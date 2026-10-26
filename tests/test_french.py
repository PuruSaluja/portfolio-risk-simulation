import hashlib
import io
import json
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import french
from portfolio_risk.data import DataError, manifest_path
from portfolio_risk.french import (
    COLUMNS,
    FILES,
    LIBRARY,
    daily_risk_free,
    fetch_factors,
    load_factors,
    main,
    parse_table,
    read_manifest,
    write_cache,
)

START = date(2024, 1, 1)
END = date(2024, 2, 29)
# The trading days of January and February 2024, less a holiday in each.
DAYS = pd.bdate_range("2024-01-02", "2024-02-29").drop(
    pd.DatetimeIndex(["2024-01-15", "2024-02-19"])
)
NOTES = "This file was created by using the 202402 CRSP database.\nSome notes.\n\n"


def daily_text(columns, rows, *, notes=NOTES):
    lines = [notes + "," + ",".join(columns)]
    for label, values in rows:
        lines.append(label + "," + ",".join(f"{v:8.2f}" for v in values))
    return "\n".join(lines) + "\n\nCopyright 2024 Eugene F. Fama and Kenneth R. French\n"


def library(days=DAYS, *, months=None, five=None, three=None, monthly_rows=None):
    """The three files' text, from seeded figures: (daily3, daily5, monthly3)."""
    rng = np.random.default_rng(0)
    labels = [d.strftime("%Y%m%d") for d in days]
    figures = np.round(rng.normal(0, 1, (len(days), 5)), 2)
    rf = np.full(len(days), 0.02)
    three = three or [
        (lab, (m, s, h, r)) for lab, (m, s, h, _, _), r in zip(labels, figures, rf, strict=True)
    ]
    five = five or [
        (lab, (m, round(s + 0.1, 2), h, w, c, r))
        for lab, (m, s, h, w, c), r in zip(labels, figures, rf, strict=True)
    ]
    monthly_rows = monthly_rows or [
        ("202312", (1.0, 0.5, 0.2, 0.43)),
        ("202401", (1.2, -0.3, -2.4, 0.47)),
        ("202402", (4.9, -0.8, -3.5, 0.42)),
    ]
    monthly = daily_text(("Mkt-RF", "SMB", "HML", "RF"), monthly_rows)
    # Monthly files go on to a table of annual figures.
    monthly = monthly.replace(
        "\n\nCopyright",
        "\n\n Annual Factors: January-December \n,Mkt-RF,SMB,HML,RF\n"
        "  2023,  1, 2, 3, 4\n\nCopyright",
    )
    return {
        "daily3": daily_text(("Mkt-RF", "SMB", "HML", "RF"), three),
        "daily5": daily_text(("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"), five),
        "monthly3": monthly,
    }


def zipped(name, text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name.replace("_CSV.zip", ".csv"), text)
    return buffer.getvalue()


def fake_download(texts):
    calls = []

    def download(url):
        calls.append(url)
        for key, name in FILES.items():
            if url == LIBRARY + name:
                return zipped(name, texts[key])
        raise AssertionError(url)

    download.calls = calls
    return download


def fetch(texts=None, start=START, end=END):
    return fetch_factors(start, end, download=fake_download(texts or library()))


def test_fetch_gives_one_row_per_day_with_every_file_s_figures() -> None:
    texts = library()
    factors, vintages = fetch(texts)
    assert list(factors.columns) == list(COLUMNS)
    assert list(factors["date"]) == [d.date() for d in DAYS]
    three = parse_table("3", texts["daily3"], ("Mkt-RF", "SMB", "HML", "RF"), 8)
    five = parse_table("5", texts["daily5"], ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"), 8)
    assert factors["mkt_rf"].tolist() == three["Mkt-RF"].tolist()
    assert factors["smb"].tolist() == three["SMB"].tolist()
    assert factors["smb5"].tolist() == five["SMB"].tolist()
    assert factors["rmw"].tolist() == five["RMW"].tolist()
    assert factors["cma"].tolist() == five["CMA"].tolist()
    assert factors["rf"].tolist() == [0.02] * len(DAYS)
    january = factors["date"].map(lambda d: d.month == 1)
    assert set(factors.loc[january, "rf_month"]) == {0.47}
    assert set(factors.loc[~january, "rf_month"]) == {0.42}
    assert vintages == {
        k: "This file was created by using the 202402 CRSP database." for k in FILES
    }


def test_fetch_asks_the_library_for_the_three_files() -> None:
    download = fake_download(library())
    fetch_factors(START, END, download=download)
    assert download.calls == [LIBRARY + name for name in FILES.values()]


def test_days_outside_the_window_are_left_out() -> None:
    factors, _ = fetch(start=date(2024, 2, 1), end=date(2024, 2, 29))
    assert factors["date"].iloc[0] == date(2024, 2, 1)
    assert len(factors) == len([d for d in DAYS if d.month == 2])


def test_data_ending_before_the_window_does_is_kept() -> None:
    factors, _ = fetch(end=date(2024, 3, 31))
    assert factors["date"].iloc[-1] == date(2024, 2, 29)


@pytest.mark.parametrize(
    ("start", "end"),
    [(date(2024, 1, 2), date(2024, 2, 29)), (date(2024, 1, 1), date(2024, 2, 28))],
)
def test_a_window_of_part_months_is_refused(start, end) -> None:
    with pytest.raises(DataError, match="not whole months"):
        fetch(start=start, end=end)


def test_a_window_that_ends_before_it_starts_is_refused() -> None:
    with pytest.raises(DataError, match="after it ends"):
        fetch(start=date(2024, 3, 1), end=date(2024, 2, 29))


def test_a_datetime_is_not_taken_for_a_date() -> None:
    with pytest.raises(DataError, match="must be a date"):
        fetch(start=datetime(2024, 1, 1))


def test_a_window_without_data_is_refused() -> None:
    with pytest.raises(DataError, match="no daily factors"):
        fetch(start=date(2025, 1, 1), end=date(2025, 1, 31))


def test_files_covering_different_days_are_refused() -> None:
    texts = library()
    texts["daily5"] = library(DAYS[:-1])["daily5"]
    with pytest.raises(DataError, match="different days.*first 20240229, 1 in all"):
        fetch(texts)


@pytest.mark.parametrize(("column", "position"), [("Mkt-RF", 0), ("HML", 2), ("RF", 5)])
def test_a_shared_figure_that_differs_between_the_files_is_refused(column, position) -> None:
    texts = library()
    rows = [line for line in texts["daily5"].splitlines() if line[:8].isdigit()]
    parts = rows[3].split(",")
    parts[position + 1] = "   99.99"
    texts["daily5"] = texts["daily5"].replace(rows[3], ",".join(parts))
    with pytest.raises(DataError, match=f"{column} differs.*on {rows[3][:8]} and 0 more"):
        fetch(texts)


def test_a_month_without_a_monthly_rate_is_refused() -> None:
    texts = library()
    texts["monthly3"] = library(monthly_rows=[("202401", (1.2, -0.3, -2.4, 0.47))])["monthly3"]
    with pytest.raises(DataError, match="no risk-free rate for 202402"):
        fetch(texts)


def test_a_date_that_is_not_one_is_refused() -> None:
    bad = [("20240132", (1.0, 1.0, 1.0, 0.02))]
    texts = library(three=bad, five=[("20240132", (1.0, 1.0, 1.0, 1.0, 1.0, 0.02))])
    with pytest.raises(DataError, match="not one"):
        fetch(texts)


def test_a_failed_download_is_reported_plainly() -> None:
    def download(url):
        raise OSError("no route to host")

    with pytest.raises(DataError, match="download of F-F_Research_Data_Factors_daily.*no route"):
        fetch_factors(START, END, download=download)


def test_a_download_that_is_not_a_zip_file_is_refused() -> None:
    with pytest.raises(DataError, match="cannot be read"):
        fetch_factors(START, END, download=lambda url: b"<html>moved</html>")


def test_a_zip_file_with_two_csv_files_is_refused() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.csv", "x")
        archive.writestr("b.csv", "y")
    with pytest.raises(DataError, match="holds 2 CSV files"):
        fetch_factors(START, END, download=lambda url: buffer.getvalue())


def test_a_file_that_is_not_ascii_is_refused() -> None:
    data = zipped("x_CSV.zip", "café")
    with pytest.raises(DataError, match="cannot be read"):
        fetch_factors(START, END, download=lambda url: data)


THREE = ("Mkt-RF", "SMB", "HML", "RF")


def test_the_table_stops_before_the_annual_figures() -> None:
    table = parse_table("m", library()["monthly3"], THREE, 6)
    assert list(table.index) == ["202312", "202401", "202402"]
    assert table.loc["202401", "RF"] == 0.47


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("notes only\n", "no table header"),
        (",Mkt-RF,SMB,RF\n20240102,1,2,3\n", "expected the columns"),
        (",Mkt-RF,SMB,HML,RF\n\n", "no rows"),
        (",Mkt-RF,SMB,HML,RF\n20240102,1,2,3\n", "has 3 figures"),
        (",Mkt-RF,SMB,HML,RF\n20240102,1,2,3,4,5\n", "has 5 figures"),
        (",Mkt-RF,SMB,HML,RF\n20240102,1,2,x,4\n", "not a number"),
        (",Mkt-RF,SMB,HML,RF\n20240102,1,2,nan,4\n", "not finite"),
        (",Mkt-RF,SMB,HML,RF\n20240103,1,2,3,4\n20240102,1,2,3,4\n", "out of order"),
        (",Mkt-RF,SMB,HML,RF\n20240102,1,2,3,4\n20240102,1,2,3,4\n", "repeated"),
    ],
)
def test_damaged_tables_are_refused(text, message) -> None:
    with pytest.raises(DataError, match=message):
        parse_table("f", text, THREE, 8)


def test_a_return_of_minus_100_percent_is_refused() -> None:
    factors, _ = fetch()
    factors.loc[3, "hml"] = -100.0
    with pytest.raises(DataError, match="hml is -100.0%, not above -100%"):
        french.validate_factors(factors, START, END)


def test_factors_outside_the_window_are_refused() -> None:
    factors, _ = fetch()
    with pytest.raises(DataError, match="outside 2024-01-03 to 2024-02-29"):
        french.validate_factors(factors, date(2024, 1, 3), END)
    with pytest.raises(DataError, match="outside 2024-01-01 to 2024-02-28"):
        french.validate_factors(factors, START, date(2024, 2, 28))


def test_a_month_whose_rate_differs_between_days_is_refused() -> None:
    factors, _ = fetch()
    factors.loc[3, "rf_month"] = 0.5
    with pytest.raises(DataError, match="2024-01: the month's risk-free rate differs"):
        french.validate_factors(factors, START, END)


def write(tmp_path: Path, factors=None) -> Path:
    cache = tmp_path / "data" / "factors.csv"
    if factors is None:
        factors, vintages = fetch()
    else:
        vintages = {}
    write_cache(
        factors,
        cache,
        start=START,
        end=END,
        vintages=vintages,
        fetched_at=datetime(2024, 3, 15, 12, 0, tzinfo=UTC),
    )
    return cache


def test_the_cache_reads_back_as_fractions_to_the_last_bit(tmp_path: Path) -> None:
    factors, _ = fetch()
    loaded = load_factors(write(tmp_path, factors))
    assert list(loaded.columns) == list(COLUMNS[1:])
    assert list(loaded.index) == list(pd.DatetimeIndex(factors["date"]))
    expected = factors.set_index(pd.DatetimeIndex(factors["date"], name="date"))[list(COLUMNS[1:])]
    pd.testing.assert_frame_equal(loaded, expected / 100.0, check_exact=True)


def test_the_manifest_records_the_window_files_and_checksum(tmp_path: Path) -> None:
    cache = write(tmp_path)
    manifest = read_manifest(cache)
    assert manifest["sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert manifest["files"] == {k: LIBRARY + n for k, n in FILES.items()}
    assert manifest["vintages"]["daily5"].endswith("202402 CRSP database.")
    assert (manifest["start"], manifest["end"]) == ("2024-01-01", "2024-02-29")
    assert (manifest["first"], manifest["last"]) == ("2024-01-02", "2024-02-29")
    assert manifest["rows"] == len(DAYS)
    assert manifest["fetched_at"] == "2024-03-15T12:00:00+00:00"
    assert b"\r\n" not in cache.read_bytes()


def test_a_changed_cache_is_refused(tmp_path: Path) -> None:
    cache = write(tmp_path)
    cache.write_bytes(cache.read_bytes().replace(b"0.02", b"0.03", 1))
    with pytest.raises(DataError, match="does not match its manifest"):
        load_factors(cache)


def test_a_cache_without_its_manifest_is_refused(tmp_path: Path) -> None:
    cache = write(tmp_path)
    manifest_path(cache).unlink()
    with pytest.raises(DataError, match="manifest.json is missing"):
        load_factors(cache)


def test_a_missing_cache_says_how_to_fetch_it(tmp_path: Path) -> None:
    cache = write(tmp_path)
    cache.unlink()
    with pytest.raises(DataError, match="python -m portfolio_risk.french fetch"):
        load_factors(cache)


def test_a_prices_manifest_is_not_taken_for_a_factors_one(tmp_path: Path) -> None:
    cache = write(tmp_path)
    manifest = json.loads(manifest_path(cache).read_text(encoding="utf-8"))
    del manifest["first"]
    manifest_path(cache).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataError, match="not a factors manifest"):
        load_factors(cache)


def rewrite_with_matching_checksum(cache: Path, text: str, **changes) -> None:
    cache.write_bytes(text.encode("utf-8"))
    manifest = read_manifest(cache)
    manifest["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest.update(changes)
    manifest_path(cache).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.replace("rf_month", "rf_monthly"), "expected the columns"),
        (lambda t: t.replace("2024-01-03", "2024-01-3x"), "cannot be parsed"),
        (lambda t: t.replace(",0.02,", ",x,", 1), "cannot be parsed"),
        (lambda t: "\n".join(t.splitlines()[:-1]) + "\n", "differ from its manifest"),
        (lambda t: "\n".join(t.splitlines()[:3] + t.splitlines()[4:]) + "\n", "differ from"),
    ],
)
def test_a_damaged_cache_is_refused_even_with_a_matching_checksum(
    tmp_path: Path, change, message
) -> None:
    cache = write(tmp_path)
    rewrite_with_matching_checksum(cache, change(cache.read_text(encoding="utf-8")))
    with pytest.raises(DataError, match=message):
        load_factors(cache)


def test_the_daily_rate_compounds_to_each_month_s_rate(tmp_path: Path) -> None:
    factors = load_factors(write(tmp_path))
    rate = daily_risk_free(factors)
    assert rate.index.equals(factors.index)
    for _, days in rate.groupby(rate.index.to_period("M")):
        assert days.nunique() == 1
        expected = factors.loc[days.index[0], "rf_month"]
        assert np.prod(1.0 + days.to_numpy()) - 1.0 == pytest.approx(expected, rel=1e-13)
    january = rate[rate.index.month == 1]
    assert len(january) == 21
    assert january.iloc[0] == pytest.approx(1.0047 ** (1 / 21) - 1, rel=1e-15)


def test_a_negative_monthly_rate_gives_a_negative_daily_rate(tmp_path: Path) -> None:
    factors = load_factors(write(tmp_path))
    factors["rf_month"] = -0.0002
    assert (daily_risk_free(factors) < 0).all()


def library_download(monkeypatch, texts=None) -> None:
    monkeypatch.setattr(french, "_download", fake_download(texts or library()))


def test_fetch_and_verify_from_the_command_line(tmp_path: Path, monkeypatch, capsys) -> None:
    library_download(monkeypatch)
    out = tmp_path / "factors.csv"
    window = ["--start", "2024-01-01", "--end", "2024-02-29"]
    assert main(["fetch", *window, "--out", str(out)]) == 0
    assert f"{len(DAYS)} days, 2024-01-02 to 2024-02-29" in capsys.readouterr().out
    assert main(["verify", "--cache", str(out)]) == 0
    assert "matches its manifest" in capsys.readouterr().out


def test_fetching_the_same_data_leaves_the_manifest_as_it_was(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    library_download(monkeypatch)
    out = tmp_path / "factors.csv"
    window = ["--start", "2024-01-01", "--end", "2024-02-29"]
    main(["fetch", *window, "--out", str(out)])
    # An older fetch time, so a rewritten manifest would differ from it.
    manifest = read_manifest(out)
    manifest["fetched_at"] = "2024-03-01T00:00:00+00:00"
    manifest_path(out).write_text(json.dumps(manifest), encoding="utf-8")
    before = manifest_path(out).read_bytes()
    out.unlink()
    assert main(["fetch", *window, "--out", str(out)]) == 0
    assert "Same data as the previous manifest" in capsys.readouterr().out
    assert manifest_path(out).read_bytes() == before
    assert load_factors(out) is not None


def test_new_data_rewrites_the_manifest(tmp_path: Path, monkeypatch, capsys) -> None:
    library_download(monkeypatch)
    out = tmp_path / "factors.csv"
    main(["fetch", "--start", "2024-01-01", "--end", "2024-02-29", "--out", str(out)])
    assert main(["fetch", "--start", "2024-02-01", "--end", "2024-02-29", "--out", str(out)]) == 0
    assert "Differs from the previous manifest" in capsys.readouterr().out
    assert read_manifest(out)["first"] == "2024-02-01"


def test_command_line_errors_are_reported_without_a_traceback(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    library_download(monkeypatch)
    out = tmp_path / "factors.csv"
    assert main(["fetch", "--start", "2024-01-02", "--end", "2024-02-29", "--out", str(out)]) == 1
    assert capsys.readouterr().err.startswith("error: the window 2024-01-02 to 2024-02-29")
    assert main(["verify", "--cache", str(out)]) == 1
    assert "manifest.json is missing" in capsys.readouterr().err
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert main(["fetch", "--out", str(blocker / "factors.csv"), "--end", "2024-02-29"]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_a_folder_given_as_the_factor_cache_is_refused_as_one(tmp_path: Path) -> None:
    for read in (french.load_factors, french.read_manifest):
        with pytest.raises(DataError, match="is a folder; the cache is a file"):
            read(tmp_path)
