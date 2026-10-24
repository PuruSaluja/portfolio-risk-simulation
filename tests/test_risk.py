import json
import math
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import french
from portfolio_risk.data import COLUMNS, write_cache
from portfolio_risk.risk import (
    downside_deviation,
    drawdowns,
    expected_shortfall,
    main,
    max_drawdown,
    measures,
    per_period_rate,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)


def series(values, start="2024-01-01") -> pd.Series:
    # Business days run past what pandas 2's nanosecond dates hold (the year 2262) after
    # about 62,000 of them, so a longer series is numbered instead.
    if len(values) > 50_000:
        return pd.Series(values, dtype="float64")
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


def random_returns(n=2_000, seed=0, scale=0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    return series(rng.standard_t(4, n) * scale + 0.0003)


def test_a_per_period_rate_compounds_back_to_the_annual_one() -> None:
    assert (1 + per_period_rate(0.05, 252)) ** 252 == pytest.approx(1.05, rel=1e-14)
    assert (1 + per_period_rate(-0.2, 12)) ** 12 == pytest.approx(0.8, rel=1e-14)
    assert per_period_rate(0.0, 252) == 0.0


@pytest.mark.parametrize("annual", [-1.0, -1.5, math.nan, math.inf, True, "0.02"])
def test_a_per_period_rate_refuses_an_annual_rate_it_cannot_compound(annual) -> None:
    with pytest.raises(ValueError, match="annual rate"):
        per_period_rate(annual, 252)


def test_sharpe_is_the_mean_excess_return_over_its_deviation_on_a_yearly_scale() -> None:
    r = random_returns()
    assert sharpe_ratio(r, 252) == pytest.approx(r.mean() / r.std() * math.sqrt(252), rel=1e-12)
    rate = 1.03 ** (1 / 252) - 1
    assert sharpe_ratio(r, 252, 0.03) == pytest.approx(
        (r.mean() - rate) / r.std() * math.sqrt(252), rel=1e-12
    )
    assert sharpe_ratio(r, 252, 0.03) < sharpe_ratio(r, 252)


def test_sharpe_refuses_returns_that_never_vary() -> None:
    with pytest.raises(ValueError, match="never vary"):
        sharpe_ratio(series([0.001] * 10), 252)


def test_downside_deviation_counts_every_return_and_only_the_shortfalls() -> None:
    r = series([0.01, -0.02, 0.03, -0.01])
    # Two shortfalls, divided by all four returns.
    assert downside_deviation(r, 1) == pytest.approx(math.sqrt((0.02**2 + 0.01**2) / 4))
    assert downside_deviation(r, 252) == pytest.approx(
        math.sqrt((0.02**2 + 0.01**2) / 4 * 252), rel=1e-12
    )
    # Against a target of 1.5% a period, 1% falls short too, and 3% does not.
    assert downside_deviation(r, 1, 0.015) == pytest.approx(
        math.sqrt((0.005**2 + 0.035**2 + 0.025**2) / 4), rel=1e-12
    )
    assert downside_deviation(series([0.01, 0.02]), 252) == 0.0


def test_sortino_is_the_mean_excess_over_the_downside_deviation() -> None:
    r = series([0.01, -0.02, 0.03, -0.01])
    assert sortino_ratio(r, 1) == pytest.approx(0.0025 / math.sqrt(0.0005 / 4), rel=1e-12)
    r = random_returns(seed=1)
    rate = 1.02 ** (1 / 252) - 1
    shortfall = np.minimum(r.to_numpy() - rate, 0)
    expected = (r.mean() - rate) * 252 / (math.sqrt(np.mean(shortfall**2)) * math.sqrt(252))
    assert sortino_ratio(r, 252, 0.02) == pytest.approx(expected, rel=1e-12)


def test_sortino_refuses_returns_that_never_fall_below_the_target() -> None:
    with pytest.raises(ValueError, match="no bound"):
        sortino_ratio(series([0.01, 0.02, 0.0]), 252)


def test_the_deepest_drawdown_has_its_peak_trough_and_recovery() -> None:
    r = series([0.1, -0.5, 0.2, 1.0, -0.1])
    # Wealth 1.1, 0.55, 0.66, 1.32, 1.188.
    worst = max_drawdown(r)
    assert worst.depth == pytest.approx(-0.5)
    assert worst.peak == r.index[0]
    assert worst.trough == r.index[1]
    assert worst.recovery == r.index[3]
    assert (worst.days_to_trough, worst.days_to_recovery) == (1, 3)


def test_a_loss_on_the_first_day_is_a_drawdown_from_the_start() -> None:
    worst = max_drawdown(series([-0.2, 0.1, 0.2]))
    assert worst.depth == pytest.approx(-0.2)
    assert worst.peak is None
    assert worst.days_to_trough == 1
    # 0.8 * 1.1 = 0.88, then 1.056: back above the start on the third day.
    assert worst.days_to_recovery == 3
    np.testing.assert_allclose(drawdowns(series([-0.2, 0.1, 0.2])), [-0.2, -0.12, 0.0], atol=1e-15)


def test_a_drawdown_that_never_recovers_has_no_recovery_date() -> None:
    r = series([0.05, -0.1, -0.1, 0.02])
    worst = max_drawdown(r)
    assert worst.peak == r.index[0] and worst.trough == r.index[2]
    assert worst.depth == pytest.approx(0.81 - 1)
    assert worst.recovery is None and worst.days_to_recovery is None


def test_the_peak_is_the_last_day_at_the_high_and_a_tie_keeps_the_first_trough() -> None:
    # Wealth 1.1, 1.1, 0.88, 1.1, 0.88: the high is reached twice before the first trough.
    r = series([0.1, 0.0, -0.2, 0.25, -0.2])
    worst = max_drawdown(r)
    assert worst.peak == r.index[1]
    assert worst.trough == r.index[2]
    assert worst.recovery == r.index[3]


def test_returns_that_never_fall_have_no_drawdown() -> None:
    worst = max_drawdown(series([0.01, 0.0, 0.02]))
    assert worst.depth == 0.0 and worst.peak is None and worst.trough is None
    assert (drawdowns(series([0.01, 0.0, 0.02])) == 0).all()


def test_drawdowns_match_the_wealth_index_over_its_running_high() -> None:
    r = random_returns(seed=2)
    wealth = (1 + r).cumprod()
    expected = wealth / wealth.cummax().clip(lower=1.0) - 1
    pd.testing.assert_series_equal(drawdowns(r), expected, check_names=False, rtol=1e-12)
    assert max_drawdown(r).depth == pytest.approx(drawdowns(r).min(), rel=1e-12)


def brute_force_var(values, level):
    """The smallest return with at least a share 1 - level of returns at or below it."""
    n = len(values)
    for x in sorted(values):
        if sum(v <= x for v in values) >= n * (1 - level) - 1e-9:
            return -x


@pytest.mark.parametrize("n", [1, 7, 20, 30, 99, 100, 101, 1000])
@pytest.mark.parametrize("level", [0.9, 0.95, 0.975, 0.99])
def test_historical_value_at_risk_is_the_lower_empirical_quantile(n, level) -> None:
    values = np.random.default_rng(n).normal(0, 0.01, n)
    assert value_at_risk(series(values), level) == brute_force_var(list(values), level)


def test_a_whole_tail_takes_that_many_returns_despite_rounding() -> None:
    # 100 * (1 - 0.95) is 5.000000000000004 in floating point.
    r = series(np.arange(1, 101) / -1000.0)
    assert value_at_risk(r, 0.95) == pytest.approx(0.096)
    assert expected_shortfall(r, 0.95) == pytest.approx((0.1 + 0.099 + 0.098 + 0.097 + 0.096) / 5)


def test_historical_expected_shortfall_counts_the_last_return_in_part() -> None:
    values = np.random.default_rng(3).normal(0, 0.01, 30)
    lowest = np.sort(values)
    # 30 returns at 95%: a tail of 1.5 returns.
    assert expected_shortfall(series(values), 0.95) == pytest.approx(
        -(lowest[0] + 0.5 * lowest[1]) / 1.5, rel=1e-12
    )
    # 10 returns at 95%: half a return, so the worst alone.
    assert expected_shortfall(series(values[:10]), 0.95) == -np.sort(values[:10])[0]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("level", [0.9, 0.95, 0.99, 0.999])
def test_expected_shortfall_is_never_below_value_at_risk(seed, level) -> None:
    r = random_returns(n=777, seed=seed)
    for method in ("historical", "normal"):
        assert expected_shortfall(r, level, method) >= value_at_risk(r, level, method)


def test_normal_measures_come_from_the_sample_mean_and_deviation() -> None:
    r = random_returns(seed=4)
    m, s = r.mean(), r.std()
    assert value_at_risk(r, 0.99, "normal") == pytest.approx(
        -(m + s * NormalDist().inv_cdf(0.01)), rel=1e-12
    )
    # The standard normal's expected shortfall at 97.5% is 2.3378 standard deviations.
    assert expected_shortfall(r, 0.975, "normal") == pytest.approx(-(m - s * 2.337802), rel=1e-6)


def test_historical_and_normal_agree_on_many_normal_returns() -> None:
    r = series(np.random.default_rng(5).normal(0.0002, 0.01, 400_000))
    for level in (0.95, 0.99):
        for measure in (value_at_risk, expected_shortfall):
            assert measure(r, level) == pytest.approx(measure(r, level, "normal"), rel=0.01)


def test_fat_tails_make_the_historical_tail_deeper_than_the_normal_one() -> None:
    r = random_returns(n=100_000, seed=6)
    assert expected_shortfall(r, 0.99) > 1.2 * expected_shortfall(r, 0.99, "normal")


@pytest.mark.parametrize("level", [0, 1, 1.5, -0.1, math.nan, True, "0.95"])
def test_a_level_outside_zero_to_one_is_refused(level) -> None:
    with pytest.raises(ValueError, match="level"):
        value_at_risk(random_returns(), level)
    with pytest.raises(ValueError, match="level"):
        expected_shortfall(random_returns(), level)


def test_an_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="historical or normal"):
        value_at_risk(random_returns(), 0.95, "cornish-fisher")
    with pytest.raises(ValueError, match="historical or normal"):
        expected_shortfall(random_returns(), 0.95, "cornish-fisher")


@pytest.mark.parametrize(
    "function",
    [
        lambda r: sharpe_ratio(r, 252),
        lambda r: downside_deviation(r, 252),
        lambda r: sortino_ratio(r, 252),
        max_drawdown,
        drawdowns,
        value_at_risk,
        expected_shortfall,
    ],
)
@pytest.mark.parametrize(
    "bad", [[0.01, math.nan, -0.02], [0.01, math.inf, -0.02], [0.01, -1.0, 0.02]]
)
def test_returns_that_cannot_compound_are_refused(function, bad) -> None:
    with pytest.raises(ValueError):
        function(series(bad))


def test_a_data_frame_is_refused_for_one_series() -> None:
    frame = pd.DataFrame({"A": [0.01, -0.02], "B": [0.0, 0.01]})
    with pytest.raises(TypeError, match="Series"):
        max_drawdown(frame)


def test_measures_gathers_every_figure_with_dates_as_text() -> None:
    r = random_returns(seed=7)
    out = measures(r, 252, risk_free=0.02)
    assert out["volatility"] == pytest.approx(r.std() * math.sqrt(252), rel=1e-12)
    assert out["sharpe_ratio"] == sharpe_ratio(r, 252, 0.02)
    assert out["sortino_ratio"] == sortino_ratio(r, 252, 0.02)
    assert out["downside_deviation"] == downside_deviation(r, 252, 0.02)
    worst = max_drawdown(r)
    assert out["max_drawdown"]["depth"] == worst.depth
    assert out["max_drawdown"]["trough"] == worst.trough.date().isoformat()
    assert set(out["tail"]) == {"95", "99"}
    assert out["tail"]["99"]["normal"]["expected_shortfall"] == expected_shortfall(
        r, 0.99, "normal"
    )


def test_a_ratio_the_returns_leave_undefined_is_none_and_a_bad_rate_is_refused() -> None:
    out = measures(series([0.01, 0.02, 0.0, 0.01]), 252)
    assert out["sortino_ratio"] is None and out["sharpe_ratio"] is not None
    assert measures(series([0.0] * 5), 252)["sharpe_ratio"] is None
    with pytest.raises(ValueError, match="annual rate"):
        measures(random_returns(), 252, risk_free=-1.0)


def prices(flat_bbb=False) -> pd.DataFrame:
    days = pd.bdate_range("2023-01-02", "2024-12-31")
    rng = np.random.default_rng(0)
    rows = []
    for i, ticker in enumerate(["AAA", "BBB"]):
        close = 50.0 * np.cumprod(1 + rng.normal(0.0003, 0.01 * (1 + i), len(days)))
        if flat_bbb and ticker == "BBB":
            close[:] = 80.0
        for day, value in zip(days, close, strict=True):
            rows.append((day.date(), ticker, float(value), 0.0, 0.0, 0.0, 1e6))
    return pd.DataFrame(rows, columns=list(COLUMNS))


def cache(tmp_path: Path, **kwargs) -> Path:
    path = tmp_path / "prices.csv"
    write_cache(prices(**kwargs), path, start=date(2023, 1, 1), end=date(2024, 12, 31), source="t")
    return path


def test_the_command_prints_and_writes_every_series(tmp_path: Path, capsys) -> None:
    path = cache(tmp_path)
    out = tmp_path / "results" / "risk.json"
    argv = ["--cache", str(path), "--json", str(out), "--weights", "AAA=0.7,BBB=0.3"]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "521 daily returns, 2023-01-03 to 2024-12-31; risk-free rate 0.00% a year" in printed
    assert "AAA 70.0%, BBB 30.0%" in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["series"].keys() == {"AAA", "BBB", "portfolio"}
    assert written["weights"] == {"AAA": 0.7, "BBB": 0.3}
    assert b"\r" not in out.read_bytes()
    # The portfolio's figures are those of its daily returns, rebalanced every close.
    p = prices().pivot(index="date", columns="ticker", values="close")
    daily = (p / p.shift(1) - 1).iloc[1:]
    portfolio = pd.Series((daily @ [0.7, 0.3]).to_numpy(), index=pd.DatetimeIndex(daily.index))
    figures = written["series"]["portfolio"]
    assert figures["sharpe_ratio"] == pytest.approx(sharpe_ratio(portfolio, 252), rel=1e-12)
    assert figures["max_drawdown"]["depth"] == pytest.approx(max_drawdown(portfolio).depth)
    monthly = (1 + portfolio).groupby(portfolio.index.to_period("M")).prod() - 1
    assert figures["sharpe_ratio_monthly"] == pytest.approx(
        monthly.mean() / monthly.std() * math.sqrt(12), rel=1e-12
    )
    years = (p.index[-1] - p.index[0]).days / 365.25
    assert figures["compound_return"] == pytest.approx(
        (1 + portfolio).prod() ** (1 / years) - 1, rel=1e-9
    )


def test_the_risk_free_rate_lowers_the_ratios(tmp_path: Path) -> None:
    path = cache(tmp_path)
    zero, five = tmp_path / "zero.json", tmp_path / "five.json"
    assert main(["--cache", str(path), "--json", str(zero)]) == 0
    assert main(["--cache", str(path), "--json", str(five), "--risk-free", "0.05"]) == 0
    a = json.loads(zero.read_text(encoding="utf-8"))["series"]["AAA"]
    b = json.loads(five.read_text(encoding="utf-8"))["series"]["AAA"]
    assert b["sharpe_ratio"] < a["sharpe_ratio"] and b["sortino_ratio"] < a["sortino_ratio"]
    assert b["max_drawdown"] == a["max_drawdown"] and b["tail"] == a["tail"]


def test_a_fund_that_never_varies_has_no_ratios(tmp_path: Path, capsys) -> None:
    out = tmp_path / "risk.json"
    assert main(["--cache", str(cache(tmp_path, flat_bbb=True)), "--json", str(out)]) == 0
    bbb = json.loads(out.read_text(encoding="utf-8"))["series"]["BBB"]
    assert bbb["sharpe_ratio"] is None and bbb["sortino_ratio"] is None
    assert bbb["max_drawdown"]["depth"] == 0.0 and bbb["max_drawdown"]["peak"] is None
    printed = [line for line in capsys.readouterr().out.splitlines() if line.startswith("BBB")]
    assert "never fell" in printed[0] and "not yet" not in printed[0]
    # A return of exactly 0 is no loss, written 0 rather than minus 0.
    for method in bbb["tail"]["99"].values():
        for value in method.values():
            assert value == 0.0 and math.copysign(1.0, value) == 1.0
    assert "-0.00%" not in printed[1]


def test_the_command_refuses_bad_input_plainly(tmp_path: Path, capsys) -> None:
    assert main(["--cache", str(tmp_path / "prices.csv")]) == 1
    assert "manifest.json is missing" in capsys.readouterr().err
    path = cache(tmp_path)
    for argv, message in [
        (["--weights", "AAA=2"], "add up to 1"),
        (["--weights", "ZZZ=1"], "no returns for ZZZ"),
        (["--risk-free", "-1"], "annual rate"),
        (["--risk-free", "nan"], "annual rate"),
        (["--json", str(tmp_path)], ""),  # a folder
    ]:
        assert main(["--cache", str(path), *argv]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error: ") and message in captured.err
        assert "Traceback" not in captured.err


def test_a_series_of_rates_is_subtracted_day_by_day() -> None:
    r = random_returns()
    rate = pd.Series(np.linspace(0.0, 0.0004, len(r)), index=r.index)
    excess = r - rate
    assert sharpe_ratio(r, 252, rate) == pytest.approx(
        excess.mean() / excess.std() * math.sqrt(252), rel=1e-12
    )
    shortfall = np.minimum(excess, 0.0)
    downside = math.sqrt((shortfall**2).mean()) * math.sqrt(252)
    assert downside_deviation(r, 252, rate) == pytest.approx(downside, rel=1e-12)
    assert sortino_ratio(r, 252, rate) == pytest.approx(excess.mean() * 252 / downside, rel=1e-12)


def test_a_constant_series_gives_what_its_annual_rate_gives() -> None:
    r = random_returns()
    daily = per_period_rate(0.04, 252)
    rate = pd.Series(daily, index=r.index)
    assert sharpe_ratio(r, 252, rate) == pytest.approx(sharpe_ratio(r, 252, 0.04), rel=1e-12)
    assert sortino_ratio(r, 252, rate) == pytest.approx(sortino_ratio(r, 252, 0.04), rel=1e-12)


def test_a_steady_return_over_a_varying_rate_has_a_sharpe_ratio() -> None:
    r = series([0.001] * 10)
    rate = pd.Series(np.linspace(0.0, 0.0009, 10), index=r.index)
    assert sharpe_ratio(r, 252, rate) > 0
    with pytest.raises(ValueError, match="never vary"):
        sharpe_ratio(r, 252, pd.Series(0.0002, index=r.index))


def test_a_series_of_rates_off_the_returns_dates_or_values_is_refused() -> None:
    r = random_returns()
    for rate, message in [
        (pd.Series(0.0, index=r.index[1:]), "on the returns' dates"),
        (pd.Series(0.0, index=r.index).where(r.index != r.index[3]), "finite"),
        (pd.Series(-1.0, index=r.index), "above -100%"),
    ]:
        for measure in (sharpe_ratio, downside_deviation, sortino_ratio):
            with pytest.raises(ValueError, match=message):
                measure(r, 252, rate)
        with pytest.raises(ValueError, match=message):
            measures(r, 252, rate)


def factor_cache(tmp_path: Path, end="2024-11-29") -> Path:
    """A factor cache over the price days up to ``end``, the bill rate rising each month."""
    days = pd.bdate_range("2023-01-02", end)
    table = pd.DataFrame({"date": days.date})
    for column in ("mkt_rf", "smb", "hml", "rf", "smb5", "rmw", "cma"):
        table[column] = 0.01
    table["rf_month"] = np.round(0.2 + 0.01 * (days.month + 12 * (days.year - 2023)), 2)
    path = tmp_path / "factors.csv"
    french.write_cache(table, path, start=date(2023, 1, 1), end=date(2024, 12, 31), vintages={})
    return path


def test_the_command_measures_against_the_bill_over_the_days_it_covers(
    tmp_path: Path, capsys
) -> None:
    path = cache(tmp_path)
    factors = factor_cache(tmp_path)
    out = tmp_path / "bills.json"
    argv = ["--cache", str(path), "--factors", str(factors), "--json", str(out)]
    assert main(argv + ["--risk-free", "bills"]) == 0
    printed = capsys.readouterr().out
    assert "499 daily returns, 2023-01-03 to 2024-11-29; risk-free rate one-month " in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["risk_free"] == "one-month Treasury bill"
    assert written["factors_sha256"] == french.read_manifest(factors)["sha256"]
    rate = french.daily_risk_free(french.load_factors(factors)).loc["2023-01-03":]
    p = prices().pivot(index="date", columns="ticker", values="close")
    daily = (p / p.shift(1) - 1).iloc[1:]
    daily.index = pd.DatetimeIndex(daily.index)
    aaa = daily["AAA"].loc[rate.index]
    figures = written["series"]["AAA"]
    assert figures["sharpe_ratio"] == pytest.approx(sharpe_ratio(aaa, 252, rate), rel=1e-12)
    assert figures["sortino_ratio"] == pytest.approx(sortino_ratio(aaa, 252, rate), rel=1e-12)
    # Monthly: each month's return against the bill compounded over the same days.
    month = aaa.index.to_period("M")
    monthly = (1 + aaa).groupby(month).prod() - 1
    bill = (1 + rate).groupby(month).prod() - 1
    excess = monthly - bill
    assert figures["sharpe_ratio_monthly"] == pytest.approx(
        excess.mean() / excess.std() * math.sqrt(12), rel=1e-12
    )
    years = (rate.index[-1] - pd.Timestamp("2023-01-02")).days / 365.25
    assert written["risk_free_compound"] == pytest.approx(
        (1 + rate).prod() ** (1 / years) - 1, rel=1e-9
    )
    assert figures["max_drawdown"]["depth"] == pytest.approx(max_drawdown(aaa).depth)


def test_a_constant_rate_leaves_the_output_as_it_was(tmp_path: Path) -> None:
    out = tmp_path / "zero.json"
    assert main(["--cache", str(cache(tmp_path)), "--json", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["risk_free"] == 0.0
    assert "risk_free_compound" not in written and "factors_sha256" not in written


def test_bills_without_a_factor_cache_are_refused_plainly(tmp_path: Path, capsys) -> None:
    argv = ["--cache", str(cache(tmp_path)), "--factors", str(tmp_path / "none.csv")]
    assert main(argv + ["--risk-free", "bills"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "none.manifest.json is missing" in captured.err


def test_rates_on_other_dates_of_the_same_number_are_refused() -> None:
    r = random_returns(n=50)
    shifted = pd.Series(0.0, index=r.index + pd.offsets.BDay(1))
    with pytest.raises(ValueError, match="on the returns' dates"):
        sharpe_ratio(r, 252, shifted)


def test_bills_missing_a_day_the_prices_have_are_refused(tmp_path: Path, capsys) -> None:
    factors = factor_cache(tmp_path)
    table = pd.read_csv(factors, dtype={"date": str})
    table = table[table["date"] != "2023-06-14"]
    table["date"] = [date.fromisoformat(d) for d in table["date"]]
    french.write_cache(table, factors, start=date(2023, 1, 1), end=date(2024, 12, 31), vintages={})
    argv = ["--cache", str(cache(tmp_path)), "--factors", str(factors), "--risk-free", "bills"]
    assert main(argv) == 1
    assert "2023-06-14 is in the returns only" in capsys.readouterr().err
