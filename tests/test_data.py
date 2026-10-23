import hashlib
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import data
from portfolio_risk.data import (
    COLUMNS,
    DataError,
    fetch_prices,
    load_cache,
    main,
    manifest_path,
    read_manifest,
    validate_tickers,
    wide,
    write_cache,
)

START = date(2024, 1, 1)
END = date(2024, 1, 12)
# The trading days in that window (1 January is a holiday).
DAYS = pd.bdate_range("2024-01-02", "2024-01-12")


def yahoo_frame(tickers=("AAA", "BBB"), days=DAYS, *, capital_gains=True):
    """A frame shaped like yfinance.download(..., auto_adjust=False, actions=True)."""
    fields = ["Adj Close", "Close", "Dividends", "High", "Low", "Open", "Stock Splits", "Volume"]
    if capital_gains:
        fields.insert(1, "Capital Gains")
    columns = pd.MultiIndex.from_product([fields, list(tickers)], names=["Price", "Ticker"])
    frame = pd.DataFrame(0.0, index=pd.DatetimeIndex(days, name="Date"), columns=columns)
    for i, ticker in enumerate(tickers):
        close = 100.0 + 10 * i + np.arange(len(days))
        for field in ("Adj Close", "Close", "High", "Low", "Open"):
            frame[(field, ticker)] = close
        frame[("Volume", ticker)] = 1000.0
    return frame


def fake_download(frame):
    calls = []

    def download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        return frame

    download.calls = calls
    return download


def fetch(frame, tickers=("AAA", "BBB"), start=START, end=END):
    return fetch_prices(tickers, start, end, download=fake_download(frame))


def test_fetch_returns_one_row_per_ticker_and_day_in_the_order_given() -> None:
    frame = yahoo_frame(("AAA", "BBB"))
    frame.loc["2024-01-08", ("Dividends", "BBB")] = 0.25
    frame.loc["2024-01-10", ("Stock Splits", "AAA")] = 2.0
    prices = fetch(frame, ("BBB", "AAA"))
    assert list(prices.columns) == list(COLUMNS)
    assert list(prices["ticker"]) == ["BBB"] * 9 + ["AAA"] * 9
    assert list(prices["date"][:9]) == list(DAYS.date)
    assert prices["close"].iloc[0] == 110.0
    bbb = prices[prices["ticker"] == "BBB"].set_index("date")
    assert bbb.loc[date(2024, 1, 8), "dividend"] == 0.25
    aaa = prices[prices["ticker"] == "AAA"].set_index("date")
    assert aaa.loc[date(2024, 1, 10), "split"] == 2.0


def test_fetch_asks_for_unadjusted_closes_and_an_end_day_inclusive() -> None:
    download = fake_download(yahoo_frame())
    fetch_prices(("AAA", "BBB"), START, END, download=download)
    tickers, kwargs = download.calls[0]
    assert tickers == ["AAA", "BBB"]
    assert kwargs["start"] == "2024-01-01"
    assert kwargs["end"] == "2024-01-13"  # yfinance's end is exclusive
    assert kwargs["auto_adjust"] is False
    assert kwargs["actions"] is True


def test_capital_gains_are_zero_when_yahoo_leaves_the_column_out() -> None:
    prices = fetch(yahoo_frame(capital_gains=False))
    assert (prices["capital_gain"] == 0).all()


def test_a_day_without_a_close_for_one_ticker_is_refused() -> None:
    frame = yahoo_frame()
    frame.loc["2024-01-09", ("Close", "AAA")] = np.nan
    with pytest.raises(DataError, match="AAA: no price on 2024-01-09 of the days the others"):
        fetch(frame)


def test_timezone_aware_dates_become_calendar_dates() -> None:
    frame = yahoo_frame()
    frame.index = frame.index.tz_localize("America/New_York")
    prices = fetch(frame)
    assert prices["date"].iloc[0] == date(2024, 1, 2)


@pytest.mark.parametrize(
    ("tickers", "message"),
    [
        ((), "no tickers"),
        ("SPY", "not one string"),
        (("SPY", "spy"), "not a ticker symbol: 'spy'"),
        (("SPY", ""), "not a ticker symbol: ''"),
        (("SPY", "AGG", "SPY"), "tickers repeated: SPY"),
        (("SPY", 5), "not a ticker symbol: 5"),
    ],
)
def test_bad_ticker_lists_are_refused(tickers, message) -> None:
    with pytest.raises(DataError, match=message):
        validate_tickers(tickers)


def test_symbols_with_dots_carets_and_dashes_are_accepted() -> None:
    assert validate_tickers(["BRK-B", "^GSPC", "RDS.A", "EURUSD=X"]) == (
        "BRK-B",
        "^GSPC",
        "RDS.A",
        "EURUSD=X",
    )


def test_a_window_that_ends_before_it_starts_is_refused() -> None:
    with pytest.raises(DataError, match="starts .2024-01-12. after it ends .2024-01-01."):
        fetch(yahoo_frame(), start=END, end=START)


def test_a_window_that_reaches_today_is_refused() -> None:
    download = fake_download(yahoo_frame())
    with pytest.raises(DataError, match=r"ends on 2024-01-12, not before today \(2024-01-12"):
        fetch_prices(("AAA", "BBB"), START, END, download=download, today=date(2024, 1, 12))
    with pytest.raises(DataError, match="not before today"):
        fetch_prices(("AAA", "BBB"), START, date(2024, 1, 20), download=download, today=END)
    assert download.calls == []
    prices = fetch_prices(("AAA", "BBB"), START, END, download=download, today=date(2024, 1, 13))
    assert len(prices) == 18


def test_today_defaults_to_the_date_in_new_york() -> None:
    new_york = pd.Timestamp.now(tz="America/New_York").date()
    download = fake_download(yahoo_frame())
    with pytest.raises(DataError, match=re.escape(f"not before today ({new_york} in New York)")):
        fetch_prices(("AAA", "BBB"), START, new_york, download=download)


def test_a_datetime_is_not_taken_for_a_date() -> None:
    with pytest.raises(DataError, match="start must be a date"):
        fetch(yahoo_frame(), start=datetime(2024, 1, 1))


def test_an_empty_download_is_refused() -> None:
    with pytest.raises(DataError, match="Yahoo returned no prices for AAA, BBB"):
        fetch(pd.DataFrame())


def test_a_ticker_yahoo_has_no_prices_for_is_named() -> None:
    frame = yahoo_frame(("AAA", "BBB"))
    frame[("Close", "BBB")] = np.nan  # how yfinance reports an unknown symbol
    with pytest.raises(DataError, match="no prices for BBB from 2024-01-01"):
        fetch(frame)


def test_a_ticker_missing_from_the_columns_is_named() -> None:
    with pytest.raises(DataError, match="no prices for CCC"):
        fetch(yahoo_frame(("AAA", "BBB")), ("AAA", "CCC"))


def test_download_errors_are_reported_plainly() -> None:
    def download(tickers, **kwargs):
        raise ConnectionError("no route to host")

    with pytest.raises(DataError, match="the download failed: ConnectionError: no route"):
        fetch_prices(("AAA",), START, END, download=download)


def test_a_download_without_a_needed_field_is_refused() -> None:
    frame = yahoo_frame().drop(columns="Dividends", level=0)
    with pytest.raises(DataError, match="no Dividends column"):
        fetch(frame)


def test_single_level_columns_are_refused() -> None:
    frame = yahoo_frame()
    frame.columns = [f"{a} {b}" for a, b in frame.columns]
    with pytest.raises(DataError, match=r"\(field, ticker\) columns"):
        fetch(frame)


def test_a_repeated_date_is_refused() -> None:
    frame = yahoo_frame()
    frame = pd.concat([frame, frame.iloc[[3]]]).sort_index()
    with pytest.raises(DataError, match="repeats a date"):
        fetch(frame)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("Close", 0.0, "close 0.0 is not positive"),
        ("Close", -5.0, "close -5.0 is not positive"),
        ("Close", np.inf, "close is inf, not a number"),
        ("Dividends", -0.1, "dividend -0.1 is negative"),
        ("Dividends", np.nan, "dividend is nan, not a number"),
        ("Stock Splits", -2.0, "split -2.0 is negative"),
        ("Volume", np.nan, "volume is nan, not a number"),
    ],
)
def test_bad_values_are_refused_with_the_ticker_and_day(field, value, message) -> None:
    frame = yahoo_frame()
    frame.loc["2024-01-05", (field, "BBB")] = value
    with pytest.raises(DataError, match=f"BBB on 2024-01-05: {message}"):
        fetch(frame)


def test_a_ticker_that_starts_more_than_a_week_late_is_refused() -> None:
    frame = yahoo_frame()
    frame.loc[:"2024-01-08", ("Close", "BBB")] = np.nan  # first price 9 January
    with pytest.raises(DataError, match="BBB: the first price is on 2024-01-09, more than 7"):
        fetch(frame)


def test_funds_that_all_start_within_a_week_are_accepted() -> None:
    frame = yahoo_frame()
    frame.loc[:"2024-01-05", ("Close", slice(None))] = np.nan  # both start on 8 January
    prices = fetch(frame)
    assert set(prices.groupby("ticker")["date"].first()) == {date(2024, 1, 8)}


def test_a_ticker_that_stops_before_the_others_is_refused() -> None:
    days = pd.bdate_range("2024-01-02", "2024-01-31")
    frame = yahoo_frame(days=days)
    frame.loc["2024-01-19":, ("Close", "BBB")] = np.nan  # last price 18 January
    with pytest.raises(DataError, match="BBB: no price on 2024-01-19 and on 8 more of the days"):
        fetch(frame, end=date(2024, 1, 31))


def test_a_ticker_that_stops_a_day_early_is_refused() -> None:
    frame = yahoo_frame()
    frame.loc["2024-01-12", ("Close", "BBB")] = np.nan
    with pytest.raises(DataError, match="BBB: no price on 2024-01-12 of the days"):
        fetch(frame)


def test_a_gap_is_refused_naming_its_first_day_and_length() -> None:
    days = pd.bdate_range("2024-01-02", "2024-01-31")
    frame = yahoo_frame(days=days)
    frame.loc["2024-01-09":"2024-01-16", ("Close", "AAA")] = np.nan
    with pytest.raises(DataError, match="AAA: no price on 2024-01-09 and on 5 more of the days"):
        fetch(frame, end=date(2024, 1, 31))


def test_days_no_fund_traded_are_not_missing() -> None:
    frame = yahoo_frame()
    frame.loc["2024-01-09", ("Close", slice(None))] = np.nan  # a closure for every fund
    prices = fetch(frame)
    assert (prices["ticker"] == "AAA").sum() == (prices["ticker"] == "BBB").sum() == 8


def test_prices_outside_the_window_are_refused() -> None:
    days = pd.bdate_range("2024-01-02", "2024-01-15")
    with pytest.raises(DataError, match="outside 2024-01-01 to 2024-01-12"):
        fetch(yahoo_frame(days=days))


def test_wide_gives_a_date_by_ticker_table() -> None:
    prices = fetch(yahoo_frame(), ("BBB", "AAA"))
    table = wide(prices, "close")
    assert list(table.columns) == ["BBB", "AAA"]
    assert table.index[0] == pd.Timestamp("2024-01-02")
    assert table.loc["2024-01-03", "AAA"] == 101.0
    with pytest.raises(DataError, match="not a price column: 'ticker'"):
        wide(prices, "ticker")


# The cache.


def write(tmp_path: Path, prices=None) -> Path:
    prices = fetch(yahoo_frame()) if prices is None else prices
    cache = tmp_path / "prices.csv"
    write_cache(
        prices,
        cache,
        start=START,
        end=END,
        source="test",
        fetched_at=datetime(2024, 1, 13, 12, tzinfo=UTC),
    )
    return cache


def test_the_cache_reads_back_exactly(tmp_path: Path) -> None:
    frame = yahoo_frame()
    frame.loc["2024-01-08", ("Dividends", "BBB")] = 0.123456789012345
    frame.loc["2024-01-09", ("Close", "AAA")] = 101.30000305175781  # float32-precise
    prices = fetch(frame)
    cache = write(tmp_path, prices)
    back = load_cache(cache)
    pd.testing.assert_frame_equal(back, prices, check_exact=True)


def test_closes_of_full_precision_read_back_to_the_last_bit(tmp_path: Path) -> None:
    # pandas' default float parser reads some numbers of 17 digits a last bit off.
    frame = yahoo_frame()
    rng = np.random.default_rng(0)
    frame[("Close", "AAA")] = 100.0 * np.exp(rng.normal(0.0, 0.1, len(DAYS)))
    frame[("Close", "BBB")] = rng.uniform(1.0, 1000.0, len(DAYS))
    prices = fetch(frame)
    back = load_cache(write(tmp_path, prices))
    assert (back["close"].to_numpy() == prices["close"].to_numpy()).all()
    pd.testing.assert_frame_equal(back, prices, check_exact=True)


def test_a_ticker_of_digits_alone_reads_back_as_text(tmp_path: Path) -> None:
    prices = fetch(yahoo_frame(("0700", "1299")), ("0700", "1299"))
    back = load_cache(write(tmp_path, prices))
    assert list(dict.fromkeys(back["ticker"])) == ["0700", "1299"]
    pd.testing.assert_frame_equal(back, prices, check_exact=True)


def test_whole_number_columns_are_written_as_floats(tmp_path: Path) -> None:
    prices = fetch(yahoo_frame())
    as_ints = prices.astype({"volume": "int64", "split": "int64"})
    back = load_cache(write(tmp_path, as_ints))
    pd.testing.assert_frame_equal(back, prices, check_exact=True)


def test_a_true_or_false_column_is_still_refused(tmp_path: Path) -> None:
    prices = fetch(yahoo_frame())
    prices["split"] = False
    with pytest.raises(DataError, match="split is False, not a number"):
        write(tmp_path, prices)


def test_the_manifest_records_the_window_rows_and_checksum(tmp_path: Path) -> None:
    frame = yahoo_frame()
    frame.loc["2024-01-08", ("Dividends", "BBB")] = 0.25
    cache = write(tmp_path, fetch(frame))
    manifest = json.loads(manifest_path(cache).read_text(encoding="utf-8"))
    assert manifest_path(cache).name == "prices.manifest.json"
    assert manifest["start"] == "2024-01-01" and manifest["end"] == "2024-01-12"
    assert manifest["fetched_at"] == "2024-01-13T12:00:00+00:00"
    assert manifest["rows"] == 18
    assert manifest["sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert manifest["tickers"]["BBB"] == {
        "rows": 9,
        "first": "2024-01-02",
        "last": "2024-01-12",
        "dividends": 1,
        "capital_gains": 0,
        "splits": 0,
    }


def test_the_cache_has_unix_line_endings_so_its_checksum_is_the_same_everywhere(
    tmp_path: Path,
) -> None:
    raw = write(tmp_path).read_bytes()
    assert b"\r" not in raw
    assert raw.startswith(b"date,ticker,close,dividend,capital_gain,split,volume\n")


def test_the_same_prices_give_the_same_file(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = write(tmp_path / "a").read_bytes()
    b = write(tmp_path / "b").read_bytes()
    assert a == b


def test_a_changed_cache_is_refused(tmp_path: Path) -> None:
    cache = write(tmp_path)
    text = cache.read_text(encoding="utf-8").replace("2024-01-05,AAA,103.0", "2024-01-05,AAA,103.5")
    cache.write_bytes(text.encode("utf-8"))
    with pytest.raises(DataError, match="does not match its manifest"):
        load_cache(cache)


def test_a_cache_with_windows_line_endings_is_refused_as_changed(tmp_path: Path) -> None:
    cache = write(tmp_path)
    cache.write_bytes(cache.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(DataError, match="does not match its manifest"):
        load_cache(cache)


def test_a_cache_without_its_manifest_is_refused(tmp_path: Path) -> None:
    cache = write(tmp_path)
    manifest_path(cache).unlink()
    with pytest.raises(DataError, match="prices.manifest.json is missing"):
        load_cache(cache)


def test_a_missing_cache_says_how_to_fetch_it(tmp_path: Path) -> None:
    cache = write(tmp_path)
    cache.unlink()
    with pytest.raises(DataError, match="fetch it with: python -m portfolio_risk.data fetch"):
        load_cache(cache)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "cannot be read"),
        (b"{not json", "cannot be read"),
        (b"\xff\xfe", "cannot be read"),
        (b"[]", "is not a prices manifest"),
        (b'{"start": "2024-01-01"}', "is not a prices manifest"),
        (
            b'{"start": "", "end": "", "file": "", "sha256": "", "rows": 0, "tickers": {}}',
            "lists no tickers",
        ),
    ],
)
def test_a_damaged_manifest_is_refused(tmp_path: Path, content, message) -> None:
    cache = write(tmp_path)
    manifest_path(cache).write_bytes(content)
    with pytest.raises(DataError, match=message):
        read_manifest(cache)


def rewrite_with_matching_checksum(cache: Path, text: str, **manifest_changes) -> None:
    """Change the cache and update its manifest's checksum, as a careless edit might."""
    cache.write_bytes(text.encode("utf-8"))
    manifest = json.loads(manifest_path(cache).read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    manifest.update(manifest_changes)
    manifest_path(cache).write_text(json.dumps(manifest), encoding="utf-8")


def test_a_cache_whose_rows_differ_from_the_manifest_is_refused(tmp_path: Path) -> None:
    cache = write(tmp_path)
    lines = cache.read_text(encoding="utf-8").splitlines(keepends=True)
    # The last day dropped for both funds (header, then AAA's nine days, then BBB's).
    rewrite_with_matching_checksum(cache, "".join(lines[:9] + lines[10:-1]))
    with pytest.raises(DataError, match="rows or dates differ from its manifest"):
        load_cache(cache)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("2024-01-05,AAA,103.0", "2024-01-05,AAA,", "cannot be parsed"),
        ("2024-01-05,AAA,103.0", "2024-01-05,AAA,abc", "cannot be parsed"),
        ("2024-01-05,AAA,103.0", "2024-01-05,AAA,-103.0", "is not positive"),
        ("2024-01-05,AAA", "2024-13-05,AAA", "month must be in 1..12"),
        ("2024-01-05,AAA", "2024-01-04,AAA", "repeated or out of order"),
        ("date,ticker", "day,ticker", "expected the columns"),
    ],
)
def test_a_damaged_cache_is_refused_even_with_a_matching_checksum(
    tmp_path: Path, old, new, message
) -> None:
    cache = write(tmp_path)
    text = cache.read_text(encoding="utf-8")
    assert old in text
    rewrite_with_matching_checksum(cache, text.replace(old, new, 1))
    with pytest.raises(DataError, match=message):
        load_cache(cache)


def test_a_ticker_order_other_than_the_manifest_s_is_refused(tmp_path: Path) -> None:
    cache = write(tmp_path)
    manifest = json.loads(manifest_path(cache).read_text(encoding="utf-8"))
    tickers = dict(reversed(list(manifest["tickers"].items())))
    manifest_path(cache).write_text(json.dumps(manifest | {"tickers": tickers}), encoding="utf-8")
    with pytest.raises(DataError, match="expected the tickers BBB, AAA in that order"):
        load_cache(cache)


def test_writing_refuses_prices_that_fail_validation(tmp_path: Path) -> None:
    prices = fetch(yahoo_frame())
    prices.loc[3, "close"] = -1.0
    with pytest.raises(DataError, match="is not positive"):
        write(tmp_path, prices)
    assert not (tmp_path / "prices.csv").exists()


# The command line.


def test_verify_reports_a_good_cache(tmp_path: Path, capsys) -> None:
    cache = write(tmp_path)
    assert main(["verify", "--cache", str(cache)]) == 0
    out = capsys.readouterr().out
    assert "18 rows, 2024-01-01 to 2024-01-12" in out
    assert "matches its manifest" in out


def test_verify_reports_a_bad_cache_without_a_traceback(tmp_path: Path, capsys) -> None:
    cache = write(tmp_path)
    cache.write_bytes(b"garbage")
    assert main(["verify", "--cache", str(cache)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "does not match its manifest" in err
    assert "Traceback" not in err


def backdate_manifest(cache: Path) -> bytes:
    """Give the manifest an old fetch time, so a rewrite shows even within one second."""
    manifest = json.loads(manifest_path(cache).read_text(encoding="utf-8"))
    manifest["fetched_at"] = "2000-01-01T00:00:00+00:00"
    manifest_path(cache).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path(cache).read_bytes()


def test_fetch_writes_the_cache_and_compares_with_the_previous_manifest(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    frame = yahoo_frame(("AAA", "BBB"))
    monkeypatch.setattr(data, "fetch_prices", lambda tickers, start, end: fetch(frame, tickers))
    cache = tmp_path / "out" / "prices.csv"
    args = ["fetch", "--tickers", "AAA, BBB", "--start", "2024-01-01", "--end", "2024-01-12"]
    assert main([*args, "--out", str(cache)]) == 0
    assert "SHA-256" in capsys.readouterr().out
    assert len(load_cache(cache)) == 18
    first = backdate_manifest(cache)
    assert main([*args, "--out", str(cache)]) == 0
    assert "Same data as the previous manifest" in capsys.readouterr().out
    assert manifest_path(cache).read_bytes() == first  # fetched_at and all left alone
    frame.loc["2024-01-05", ("Close", "AAA")] = 99.0
    assert main([*args, "--out", str(cache)]) == 0
    assert "Differs from the previous manifest." in capsys.readouterr().out
    assert manifest_path(cache).read_bytes() != first


def test_fetching_the_same_data_into_a_fresh_clone_changes_only_the_untracked_cache(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    frame = yahoo_frame(("AAA", "BBB"))
    monkeypatch.setattr(data, "fetch_prices", lambda tickers, start, end: fetch(frame, tickers))
    cache = tmp_path / "prices.csv"
    args = ["fetch", "--tickers", "AAA,BBB", "--start", "2024-01-01", "--end", "2024-01-12"]
    assert main([*args, "--out", str(cache)]) == 0
    committed = backdate_manifest(cache)
    cache.unlink()  # as in a clone: the manifest is committed, the prices are not
    assert main([*args, "--out", str(cache)]) == 0
    assert manifest_path(cache).read_bytes() == committed
    assert len(load_cache(cache)) == 18


def test_a_new_window_rewrites_the_manifest_even_with_the_same_prices(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    frame = yahoo_frame(("AAA", "BBB"))
    monkeypatch.setattr(data, "fetch_prices", lambda tickers, start, end: fetch(frame, tickers))
    cache = tmp_path / "prices.csv"
    args = ["fetch", "--tickers", "AAA,BBB", "--end", "2024-01-12", "--out", str(cache)]
    assert main([*args, "--start", "2024-01-01"]) == 0
    assert main([*args, "--start", "2023-12-30"]) == 0  # same prices, a wider window
    assert "Differs from the previous manifest." in capsys.readouterr().out
    assert read_manifest(cache)["start"] == "2023-12-30"
    assert len(load_cache(cache)) == 18


def test_fetch_reports_a_failed_download_without_a_traceback(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def failing(tickers, start, end):
        raise DataError("the download failed: ConnectionError: offline")

    monkeypatch.setattr(data, "fetch_prices", failing)
    assert main(["fetch", "--out", str(tmp_path / "prices.csv")]) == 1
    assert capsys.readouterr().err == "error: the download failed: ConnectionError: offline\n"
    assert not (tmp_path / "prices.csv").exists()


def test_a_bad_date_on_the_command_line_is_refused(capsys) -> None:
    with pytest.raises(SystemExit) as exit_:
        main(["fetch", "--start", "2024-02-30"])
    assert exit_.value.code == 2
    assert "not a date (YYYY-MM-DD): '2024-02-30'" in capsys.readouterr().err


def test_fetch_reports_an_unwritable_cache_without_a_traceback(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    frame = yahoo_frame()
    monkeypatch.setattr(data, "fetch_prices", lambda tickers, start, end: fetch(frame, tickers))
    folder = tmp_path / "prices.csv"
    folder.mkdir()  # a folder where the cache file should go
    args = ["fetch", "--tickers", "AAA,BBB", "--start", "2024-01-01", "--end", "2024-01-12"]
    assert main([*args, "--out", str(folder)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err
