import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data import COLUMNS, write_cache
from portfolio_risk.summary import main, summarise

DAYS = pd.bdate_range("2023-01-02", "2024-12-31")


def prices(seed=0, late_start=0):
    """Two years of two funds, the second paying a quarterly dividend."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, ticker in enumerate(["AAA", "BBB"]):
        close = 50.0 * (1 + i) * np.cumprod(1 + rng.normal(0.0003, 0.01 * (1 + i), len(DAYS)))
        for j, (day, value) in enumerate(zip(DAYS, close, strict=True)):
            if i == 1 and j < late_start:
                continue
            dividend = 0.5 if i == 1 and j % 63 == 62 else 0.0
            rows.append((day.date(), ticker, float(value), dividend, 0.0, 0.0, 1e6))
    return pd.DataFrame(rows, columns=list(COLUMNS))


def test_the_figures_follow_from_the_returns() -> None:
    p = prices()
    summary = summarise(p)
    close = p.pivot(index="date", columns="ticker", values="close")
    dividend = p.pivot(index="date", columns="ticker", values="dividend")
    daily = ((close + dividend) / close.shift(1) - 1).iloc[1:]
    years = (DAYS[-1] - DAYS[0]).days / 365.25
    for ticker in ["AAA", "BBB"]:
        f = summary["funds"][ticker]
        growth = (1 + daily[ticker]).prod()
        assert f["compound_return"] == pytest.approx(growth ** (1 / years) - 1, rel=1e-9)
        price_growth = close[ticker].iloc[-1] / close[ticker].iloc[0]
        assert f["price_only_compound_return"] == pytest.approx(
            price_growth ** (1 / years) - 1, rel=1e-9
        )
        assert f["volatility_daily"] == pytest.approx(
            daily[ticker].std() * math.sqrt(252), rel=1e-12
        )
        assert f["mean_return"] == pytest.approx(daily[ticker].mean() * 252, rel=1e-12)
    assert (
        summary["funds"]["BBB"]["compound_return"]
        > summary["funds"]["BBB"]["price_only_compound_return"]
    )
    assert summary["funds"]["AAA"]["compound_return"] == pytest.approx(
        summary["funds"]["AAA"]["price_only_compound_return"], rel=1e-12
    )
    corr = daily.corr()
    assert summary["correlation"]["AAA"]["BBB"] == pytest.approx(corr.loc["AAA", "BBB"], rel=1e-9)
    assert summary["daily_returns"] == len(DAYS) - 1
    assert summary["monthly_returns"] == 24
    assert summary["first_close"] == "2023-01-02"
    assert summary["trading_days_per_year_observed"] == pytest.approx((len(DAYS) - 1) / years)


def test_the_years_start_at_the_first_close_every_fund_has() -> None:
    summary = summarise(prices(late_start=3))
    assert summary["first_close"] == DAYS[3].date().isoformat()
    years = (DAYS[-1] - DAYS[3]).days / 365.25
    assert summary["trading_days_per_year_observed"] == pytest.approx((len(DAYS) - 4) / years)


def test_monthly_volatility_uses_month_total_returns_and_twelve_a_year() -> None:
    p = prices(seed=3)
    summary = summarise(p)
    # BBB pays dividends, so its total and price returns differ.
    bbb = p[p["ticker"] == "BBB"].set_index("date")
    daily = ((bbb["close"] + bbb["dividend"]) / bbb["close"].shift(1) - 1).iloc[1:]
    monthly = (1 + daily).groupby(pd.DatetimeIndex(daily.index).to_period("M")).prod() - 1
    assert summary["funds"]["BBB"]["volatility_monthly"] == pytest.approx(
        monthly.std() * math.sqrt(12), rel=1e-9
    )


def test_the_autocorrelation_figures_come_from_the_daily_total_returns() -> None:
    p = prices(seed=5)
    summary = summarise(p)
    bbb = p[p["ticker"] == "BBB"].set_index("date")
    daily = ((bbb["close"] + bbb["dividend"]) / bbb["close"].shift(1) - 1).iloc[1:]
    d = daily.to_numpy() - daily.mean()
    f = summary["funds"]["BBB"]
    assert f["autocorrelation_lag1"] == pytest.approx(np.dot(d[:-1], d[1:]) / np.dot(d, d))
    windows = (1 + daily).rolling(21).apply(np.prod, raw=True).dropna() - 1
    assert f["volatility_21day_overlapping"] == pytest.approx(
        windows.std() * math.sqrt(252 / 21), rel=1e-9
    )
    rho = [np.dot(d[:-k], d[k:]) / np.dot(d, d) for k in range(1, 21)]
    implied = math.sqrt(1 + 2 * sum((1 - k / 21) * rho[k - 1] for k in range(1, 21)))
    assert f["volatility_ratio_implied_21day"] == pytest.approx(implied, rel=1e-9)


def test_the_command_prints_and_writes_the_figures(tmp_path: Path, capsys) -> None:
    cache = tmp_path / "prices.csv"
    manifest = write_cache(
        prices(),
        cache,
        start=date(2023, 1, 1),
        end=date(2024, 12, 31),
        source="test",
        fetched_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    out = tmp_path / "results" / "summary.json"
    assert main(["--cache", str(cache), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "521 daily returns, 2023-01-02 to 2024-12-31" in printed
    assert "Ledoit-Wolf shrinkage" in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["cache_sha256"] == manifest["sha256"]
    assert written["funds"].keys() == {"AAA", "BBB"}
    assert b"\r" not in out.read_bytes()


def test_the_command_refuses_a_damaged_cache_plainly(tmp_path: Path, capsys) -> None:
    assert main(["--cache", str(tmp_path / "prices.csv")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "manifest.json is missing" in err


def test_an_unwritable_json_file_is_reported_before_anything_is_printed(
    tmp_path: Path, capsys
) -> None:
    cache = tmp_path / "prices.csv"
    write_cache(prices(), cache, start=date(2023, 1, 1), end=date(2024, 12, 31), source="test")
    assert main(["--cache", str(cache), "--json", str(tmp_path)]) == 1  # a folder
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ") and "Traceback" not in captured.err
