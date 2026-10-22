from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.montecarlo import historical_portfolio_returns
from portfolio_risk.scenarios import (
    MIXES,
    STRESS_PERIODS,
    _mix,
    backtest,
    stress,
    window,
)

WEIGHTS = {"A": 0.5, "B": 0.3, "C": 0.2}


def daily_returns(n=600, seed=0, start="2021-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal([0.0006, 0.0001, -0.0002], [0.015, 0.005, 0.02], size=(n, 3))
    return pd.DataFrame(values, columns=list("ABC"), index=pd.bdate_range(start, periods=n))


def wealth(r: pd.Series) -> float:
    return float((1 + r).prod())


def test_daily_rebalancing_is_the_fixed_weight_portfolio() -> None:
    r = daily_returns()
    test = backtest(r, WEIGHTS, "daily")
    np.testing.assert_allclose(
        test.returns, historical_portfolio_returns(r, WEIGHTS), rtol=0, atol=1e-15
    )
    assert test.rebalances == len(r) - 1
    assert test.turnover.iloc[-1] == 0.0


def test_buy_and_hold_ends_with_each_fund_s_own_growth() -> None:
    r = daily_returns(seed=1)
    test = backtest(r, WEIGHTS, "never")
    growth = (1 + r).prod()
    expected = sum(w * growth[t] for t, w in WEIGHTS.items())
    assert wealth(test.returns) == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(
        test.weights.iloc[-1], [WEIGHTS[t] * growth[t] / expected for t in "ABC"], rtol=1e-12
    )
    assert test.rebalances == 0 and test.turnover.sum() == 0.0


def test_a_rebalance_with_nothing_to_trade_still_counts() -> None:
    flat = pd.DataFrame(0.0, index=pd.bdate_range("2021-01-01", periods=30), columns=list("ABC"))
    test = backtest(flat, WEIGHTS, "daily")
    assert test.rebalances == 29 and test.turnover.sum() == 0.0
    assert test.rebalanced.iloc[:-1].all() and not test.rebalanced.iloc[-1]


def test_monthly_rebalancing_holds_each_month_from_the_targets() -> None:
    r = daily_returns(seed=2)
    test = backtest(r, WEIGHTS, "monthly")
    expected = 1.0
    for _, month in r.groupby(r.index.to_period("M")):
        growth = (1 + month).prod()
        expected *= sum(w * growth[t] for t, w in WEIGHTS.items())
    assert wealth(test.returns) == pytest.approx(expected, rel=1e-12)
    months = r.index.to_period("M").nunique()
    assert test.rebalances == months - 1
    rebalanced = test.rebalanced[test.rebalanced].index
    assert (rebalanced.to_period("M") != (rebalanced + pd.offsets.BDay()).to_period("M")).all()


@pytest.mark.parametrize("rule, code", [("quarterly", "Q"), ("yearly", "Y")])
def test_quarterly_and_yearly_rebalance_on_the_last_close_of_each(rule, code) -> None:
    r = daily_returns(n=900, seed=3)
    test = backtest(r, WEIGHTS, rule)
    expected = 1.0
    for _, part in r.groupby(r.index.to_period(code)):
        growth = (1 + part).prod()
        expected *= sum(w * growth[t] for t, w in WEIGHTS.items())
    assert wealth(test.returns) == pytest.approx(expected, rel=1e-12)
    assert test.rebalances == r.index.to_period(code).nunique() - 1


def test_turnover_is_half_the_weight_traded_back_to_target() -> None:
    r = daily_returns(seed=4)
    test = backtest(r, WEIGHTS, "monthly")
    target = np.array(list(WEIGHTS.values()))
    drifted = test.weights.shift(1).fillna(dict(WEIGHTS)).to_numpy() * (1 + r.to_numpy())
    drifted /= drifted.sum(axis=1, keepdims=True)
    due = test.rebalanced.to_numpy()
    np.testing.assert_allclose(
        test.turnover.to_numpy()[due], 0.5 * np.abs(drifted[due] - target).sum(axis=1), rtol=1e-12
    )
    np.testing.assert_allclose(test.weights.to_numpy()[due], np.tile(target, (due.sum(), 1)))
    np.testing.assert_allclose(test.weights.sum(axis=1), 1.0, rtol=1e-12)


def test_a_band_rebalances_only_once_a_weight_drifts_past_it() -> None:
    r = daily_returns(n=1500, seed=5)
    test = backtest(r, WEIGHTS, "band:0.03")
    target = pd.Series(WEIGHTS)
    drift = (test.weights - target).abs().max(axis=1)
    # After every close but the last, no weight is more than the band from its target.
    assert (drift.iloc[:-1] <= 0.03 + 1e-12).all()
    assert test.rebalances > 0
    held = test.weights.shift(1).fillna(dict(WEIGHTS)).to_numpy() * (1 + r.to_numpy())
    held /= held.sum(axis=1, keepdims=True)
    passed = np.abs(held - target.to_numpy()).max(axis=1) > 0.03
    passed[-1] = False
    np.testing.assert_array_equal(test.rebalanced.to_numpy(), passed)
    # A wider band trades less.
    assert backtest(r, WEIGHTS, "band:0.1").turnover.sum() < test.turnover.sum()


def test_a_band_never_rebalances_at_the_last_close() -> None:
    r = pd.DataFrame(
        {"A": [0.0, 0.0, 0.5], "B": [0.0, 0.0, 0.0], "C": [0.0, 0.0, 0.0]},
        index=pd.bdate_range("2021-01-01", periods=3),
    )
    test = backtest(r, WEIGHTS, "band:0.05")
    assert test.rebalances == 0 and test.weights.iloc[-1]["A"] == pytest.approx(0.75 / 1.25)


def test_funds_left_out_of_a_mix_stay_out() -> None:
    test = backtest(daily_returns(), {"A": 0.6, "C": 0.4}, "never")
    assert (test.weights["B"] == 0).all()


@pytest.mark.parametrize(
    "rule, message",
    [
        ("weekly", "a rule is"),
        ("band", "a rule is"),
        ("band:", "not a number"),
        ("band:x", "not a number"),
        ("band:0", "between 0 and 1"),
        ("band:1", "between 0 and 1"),
        ("band:nan", "between 0 and 1"),
    ],
)
def test_an_unknown_rule_is_refused(rule, message) -> None:
    with pytest.raises(ValueError, match=message):
        backtest(daily_returns(), WEIGHTS, rule)


def test_bad_returns_and_weights_are_refused() -> None:
    r = daily_returns()
    bad = r.copy()
    bad.iloc[3, 1] = -1.0
    with pytest.raises(ValueError, match="above -100%"):
        backtest(bad, WEIGHTS, "daily")
    with pytest.raises(ValueError, match="increasing"):
        backtest(r.iloc[::-1], WEIGHTS, "daily")
    with pytest.raises(ValueError, match="add up to 1"):
        backtest(r, {"A": 0.5}, "daily")
    with pytest.raises(ValueError, match="no returns for D"):
        backtest(r, {"D": 1.0}, "daily")


def test_a_window_runs_from_the_close_of_its_first_day_to_its_last() -> None:
    r = daily_returns()
    part = window(r, date(2021, 3, 1), date(2021, 3, 31))
    assert part.index[0] == pd.Timestamp("2021-03-02")
    assert part.index[-1] == pd.Timestamp("2021-03-31")
    with pytest.raises(ValueError, match="2021-03-06 is not a day"):
        window(r, date(2021, 3, 6), date(2021, 3, 31))  # a Saturday
    with pytest.raises(ValueError, match="end after it starts"):
        window(r, date(2021, 3, 31), date(2021, 3, 1))


def test_stress_starts_fresh_at_the_targets() -> None:
    r = daily_returns(seed=6)
    out = stress(r, WEIGHTS, "never", date(2021, 6, 1), date(2021, 9, 1))
    growth = (1 + window(r, date(2021, 6, 1), date(2021, 9, 1))).prod()
    assert out["return"] == pytest.approx(
        sum(w * growth[t] for t, w in WEIGHTS.items()) - 1, rel=1e-12
    )
    assert out["max_drawdown"] <= min(out["return"], 0.0) + 1e-15


def test_the_named_mixes_and_periods_are_well_formed() -> None:
    tickers = ["SPY", "IWM", "EFA", "EEM", "AGG", "TLT", "GLD", "VNQ"]
    columns = pd.DataFrame(0.0, index=pd.bdate_range("2005-01-03", "2026-09-30"), columns=tickers)
    for spec in MIXES:
        assert backtest(columns.iloc[:5], _mix(spec, tickers), "daily").rebalances == 4
    for start, end in STRESS_PERIODS.values():
        assert start < end and start.weekday() < 5 and end.weekday() < 5
