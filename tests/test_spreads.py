import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import spreads
from portfolio_risk.data import DataError, manifest_path
from portfolio_risk.spreads import (
    COLUMNS,
    fetch_ranges,
    load_ranges,
    main,
    read_manifest,
    write_cache,
)

START = date(2024, 1, 1)
END = date(2024, 1, 12)
DAYS = pd.bdate_range("2024-01-02", "2024-01-12")


def yahoo_frame(tickers=("AAA", "BBB"), days=DAYS):
    """A frame shaped like yfinance.download(..., auto_adjust=False, actions=True)."""
    fields = ["Adj Close", "Close", "Dividends", "High", "Low", "Open", "Stock Splits", "Volume"]
    columns = pd.MultiIndex.from_product([fields, list(tickers)], names=["Price", "Ticker"])
    frame = pd.DataFrame(0.0, index=pd.DatetimeIndex(days, name="Date"), columns=columns)
    for i, ticker in enumerate(tickers):
        close = 100.0 + 10 * i + np.arange(len(days))
        frame[("Close", ticker)] = close
        frame[("Adj Close", ticker)] = close
        frame[("Open", ticker)] = close
        frame[("High", ticker)] = close + 0.5
        frame[("Low", ticker)] = close - 0.5
        frame[("Volume", ticker)] = 1000.0
    return frame


def fetch(frame, tickers=("AAA", "BBB")):
    return fetch_ranges(tickers, START, END, download=lambda *a, **k: frame, today=date(2024, 2, 1))


def test_fetch_keeps_each_day_s_high_low_and_close_in_ticker_order() -> None:
    ranges = fetch(yahoo_frame(), ("BBB", "AAA"))
    assert list(ranges.columns) == list(COLUMNS)
    assert list(ranges["ticker"]) == ["BBB"] * 9 + ["AAA"] * 9
    first = ranges.iloc[0]
    assert (first["date"], first["high"], first["low"], first["close"]) == (
        date(2024, 1, 2),
        110.5,
        109.5,
        110.0,
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda f: f.__setitem__(("Low", "AAA"), f[("Close", "AAA")] + 1),
            "outside the day's range",
        ),
        (
            lambda f: f.__setitem__(("High", "BBB"), f[("Close", "BBB")] - 1),
            "outside the day's range",
        ),
        (lambda f: f.__setitem__(("Low", "AAA"), 0.0), "low is 0.0, not a positive number"),
        (lambda f: f.drop(columns=("High", "BBB"), inplace=True), "no High for BBB"),
    ],
)
def test_bars_the_estimators_would_misread_are_refused(change, message) -> None:
    frame = yahoo_frame()
    change(frame)
    with pytest.raises(DataError, match=message):
        fetch(frame)


def test_a_window_ending_today_or_later_is_refused() -> None:
    with pytest.raises(DataError, match="not before today"):
        fetch_ranges(["AAA"], START, END, download=lambda *a, **k: None, today=END)
    with pytest.raises(DataError, match="the download failed"):
        fetch_ranges(["AAA"], START, END, download=_fails, today=date(2024, 2, 1))


def _fails(*args, **kwargs):
    raise ConnectionError("offline")


def cache(tmp_path: Path) -> Path:
    path = tmp_path / "ranges.csv"
    write_cache(fetch(yahoo_frame()), path, start=START, end=END, source="test")
    return path


def test_the_cache_reads_back_exactly_and_refuses_a_changed_file(tmp_path: Path) -> None:
    path = cache(tmp_path)
    ranges = load_ranges(path)
    pd.testing.assert_frame_equal(ranges, fetch(yahoo_frame()), check_exact=True)
    manifest = read_manifest(path)
    assert manifest["rows"] == 18 and manifest["tickers"]["AAA"]["first"] == "2024-01-02"
    path.write_bytes(path.read_bytes().replace(b"110.5", b"110.6", 1))
    with pytest.raises(DataError, match="does not match its manifest"):
        load_ranges(path)


def test_a_cache_without_its_manifest_or_a_folder_is_refused(tmp_path: Path) -> None:
    path = cache(tmp_path)
    manifest_path(path).unlink()
    with pytest.raises(DataError, match="manifest.json is missing"):
        load_ranges(path)
    with pytest.raises(DataError, match="is a folder"):
        load_ranges(tmp_path)
    other = tmp_path / "other.csv"
    other.write_text("x", encoding="utf-8")
    manifest_path(other).write_text(json.dumps({"rows": 1}), encoding="utf-8")
    with pytest.raises(DataError, match="not a ranges manifest"):
        load_ranges(other)


def test_verify_checks_a_cache_plainly(tmp_path: Path, capsys) -> None:
    path = cache(tmp_path)
    assert main(["verify", "--cache", str(path)]) == 0
    assert "matches its manifest" in capsys.readouterr().out
    path.write_bytes(b"date,ticker,high,low,close\n")
    assert main(["verify", "--cache", str(path)]) == 1
    assert capsys.readouterr().err.startswith("error: ")
    assert Path("data") / "ranges.csv" == spreads.DEFAULT_CACHE
