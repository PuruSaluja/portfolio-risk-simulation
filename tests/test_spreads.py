import json
import math
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


def still_mid(spread=0.002, days=60, seed=0):
    """Bars around a mid-price that never moves: the high the ask, the low the bid, the
    close one or the other at random."""
    rng = np.random.default_rng(seed)
    ask, bid = 50.0 * (1 + spread / 2), 50.0 * (1 - spread / 2)
    index = pd.bdate_range("2024-01-01", periods=days)
    close = pd.Series(np.where(rng.random(days) < 0.5, ask, bid), index=index)
    return pd.Series(ask, index=index), pd.Series(bid, index=index), close


def test_with_a_still_mid_price_both_recover_the_spread_exactly() -> None:
    high, low, close = still_mid()
    ar = spreads.abdi_ranaldo(high, low, close)
    cs = spreads.corwin_schultz(high, low, close)
    # Abdi and Ranaldo measure the log spread, Corwin and Schultz the proportional one.
    np.testing.assert_allclose(ar, np.log(1.001 / 0.999), rtol=1e-12)
    np.testing.assert_allclose(cs, 0.002, rtol=1e-12)
    assert list(ar.index.astype(str)) == ["2024-01", "2024-02", "2024-03"]


@pytest.mark.parametrize("seed", [0, 1])
def test_a_spread_wide_beside_the_day_s_movement_is_recovered(seed) -> None:
    bars = spreads.simulate(0.005, 0.002, 2000, seed=seed)
    ar = spreads.abdi_ranaldo(bars["high"], bars["low"], bars["close"])
    cs = spreads.corwin_schultz(bars["high"], bars["low"], bars["close"])
    assert ar.mean() == pytest.approx(0.005, rel=0.05)
    assert cs.mean() == pytest.approx(0.005, rel=0.10)


@pytest.mark.parametrize("seed", [0, 1])
def test_a_spread_narrow_beside_it_reads_as_many_times_wider(seed) -> None:
    # A basis point at 1% a day: the funds' case.
    bars = spreads.simulate(0.0001, 0.01, 2000, seed=seed)
    ar = spreads.abdi_ranaldo(bars["high"], bars["low"], bars["close"])
    cs = spreads.corwin_schultz(bars["high"], bars["low"], bars["close"])
    assert ar.mean() > 10 * 0.0001 and (ar == 0).mean() > 0.4
    assert cs.mean() > 20 * 0.0001


def test_windows_ending_on_a_skipped_day_are_left_out() -> None:
    high, low, close = still_mid(days=40)
    jump = high.index[10]
    # The close jumps on one day, as on an ex-distribution day.
    for series in (high, low, close):
        series.loc[jump:] *= 0.9
    with_jump = spreads.abdi_ranaldo(high, low, close)
    without = spreads.abdi_ranaldo(high, low, close, skip=pd.DatetimeIndex([jump]))
    assert with_jump.iloc[0] != pytest.approx(np.log(1.001 / 0.999), rel=1e-6)
    np.testing.assert_allclose(without, np.log(1.001 / 0.999), rtol=1e-9)
    cs = spreads.corwin_schultz(high, low, close, skip=pd.DatetimeIndex([jump]))
    np.testing.assert_allclose(cs, 0.002, rtol=1e-9)


def test_bars_the_estimators_cannot_use_are_refused() -> None:
    high, low, close = still_mid()
    for args, message in [
        ((high, low, close.iloc[1:]), "on the same dates"),
        ((high, low, close * 2), "outside its day's range"),
        ((high, low * 0, close), "positive numbers"),
        ((high.iloc[:1], low.iloc[:1], close.iloc[:1]), "two days"),
        ((high.to_numpy(), low, close), "Series on a date index"),
    ]:
        with pytest.raises(ValueError, match=message):
            spreads.abdi_ranaldo(*args)
    with pytest.raises(ValueError, match="spread from 0 to 2"):
        spreads.simulate(-0.1, 0.01, 10)
    with pytest.raises(ValueError, match="two days"):
        spreads.simulate(0.001, 0.01, 1)


def corwin_schultz_by_loop(high, low, close):
    """Corwin and Schultz's steps, one two-day window at a time."""
    k = 3 - 2 * math.sqrt(2)
    out = {}
    for t in range(len(close) - 1):
        h0, l0, c0 = high.iloc[t], low.iloc[t], close.iloc[t]
        h1, l1 = high.iloc[t + 1], low.iloc[t + 1]
        if l1 > c0:  # the second day opens above the first day's close
            h1, l1 = h1 - (l1 - c0), c0
        elif h1 < c0:  # or below it
            h1, l1 = c0, l1 + (c0 - h1)
        beta = math.log(h0 / l0) ** 2 + math.log(h1 / l1) ** 2
        gamma = math.log(max(h0, h1) / min(l0, l1)) ** 2
        alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)
        out[close.index[t]] = max(2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha)), 0.0)
    s = pd.Series(out)
    return s.groupby(s.index.to_period("M")).mean()


def test_corwin_schultz_moves_the_second_day_s_range_past_an_overnight_gap() -> None:
    bars = spreads.simulate(0.003, 0.004, 120, trades_per_day=30, seed=3)
    # Overnight gaps: every third day the whole bar moves up or down 2%.
    rng = np.random.default_rng(3)
    shift = np.cumprod(np.where(np.arange(120) % 3 == 0, rng.choice([0.98, 1.02], 120), 1.0))
    bars = bars.mul(shift, axis=0)
    high, low, close = bars["high"], bars["low"], bars["close"]
    gaps = ((low.shift(-1) > close) | (high.shift(-1) < close)).sum()
    assert gaps > 20
    np.testing.assert_allclose(
        spreads.corwin_schultz(high, low, close),
        corwin_schultz_by_loop(high, low, close),
        rtol=1e-12,
    )
