import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data import COLUMNS, write_cache
from portfolio_risk.montecarlo import historical_portfolio_returns
from portfolio_risk.scenarios import (
    MIXES,
    STRESS_PERIODS,
    _mix,
    backtest,
    main,
    stress,
    summarise,
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


def test_a_rule_is_one_rule_however_it_is_written() -> None:
    r = daily_returns(seed=8)
    pd.testing.assert_series_equal(
        backtest(r, WEIGHTS, "Monthly ").returns, backtest(r, WEIGHTS, "monthly").returns
    )
    pd.testing.assert_series_equal(
        backtest(r, WEIGHTS, "BAND:.05").turnover, backtest(r, WEIGHTS, "band:0.05").turnover
    )


def test_returns_without_dates_are_refused() -> None:
    # Numbers taken as dates are nanoseconds from 1970, so no month would ever end.
    r = daily_returns().reset_index(drop=True)
    with pytest.raises(ValueError, match="date index"):
        backtest(r, WEIGHTS, "monthly")


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


def test_summarise_annualises_with_the_periods_it_is_given() -> None:
    r = daily_returns(seed=7)
    test = backtest(r, WEIGHTS, "monthly")
    out = summarise(test, 250.0)
    assert out["compound_return"] == pytest.approx(
        wealth(test.returns) ** (250 / len(r)) - 1, rel=1e-12
    )
    assert out["turnover_per_year"] == pytest.approx(
        test.turnover.sum() / (len(r) / 250), rel=1e-12
    )
    assert out["volatility"] == pytest.approx(test.returns.std() * 252**0.5, rel=1e-12)
    assert out["sharpe_ratio"] == pytest.approx(
        test.returns.mean() / test.returns.std() * 252**0.5, rel=1e-12
    )
    assert out["rebalances"] == test.rebalances
    assert out["final_weights"] == pytest.approx(dict(test.weights.iloc[-1]))
    w = (1 + test.returns).cumprod()
    assert out["max_drawdown"] == pytest.approx((w / w.cummax().clip(lower=1) - 1).min())


def prices() -> pd.DataFrame:
    days = pd.bdate_range("2023-01-02", "2024-12-31")
    rng = np.random.default_rng(0)
    rows = []
    for i, ticker in enumerate(["AAA", "BBB"]):
        close = 50.0 * np.cumprod(1 + rng.normal(0.0003, 0.01 * (1 + i), len(days)))
        for day, value in zip(days, close, strict=True):
            rows.append((day.date(), ticker, float(value), 0.0, 0.0, 0.0, 1e6))
    return pd.DataFrame(rows, columns=list(COLUMNS))


def cache(tmp_path: Path) -> Path:
    path = tmp_path / "prices.csv"
    write_cache(prices(), path, start=date(2023, 1, 1), end=date(2024, 12, 31), source="t")
    return path


def test_the_command_prints_and_writes_every_mix_rule_and_period(tmp_path: Path, capsys) -> None:
    out = tmp_path / "results" / "scenarios.json"
    argv = ["--cache", str(cache(tmp_path)), "--json", str(out), "--mix", "AAA=0.7,BBB=0.3"]
    argv += ["--mix", "equal", "--rule", "never", "--rule", "monthly"]
    argv += ["--period", "2023-01-02:2023-06-30", "--period", "2024-03-01:2024-04-30"]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert (
        f"2023-01-02 to 2023-06-30 ({len(pd.bdate_range('2023-01-03', '2023-06-30'))} days)"
        in printed
    )
    written = json.loads(out.read_text(encoding="utf-8"))
    assert b"\r" not in out.read_bytes()
    assert written["mixes"] == {
        "AAA=0.7,BBB=0.3": {"AAA": 0.7, "BBB": 0.3},
        "equal": {
            "AAA": 0.5,
            "BBB": 0.5,
        },
    }
    assert set(written["history"]["equal"]) == {"never", "monthly"}
    assert written["history"]["equal"]["monthly"]["rebalances"] == 23
    assert [s["period"] for s in written["stress"]] == [
        "2023-01-02:2023-06-30",
        "2024-03-01:2024-04-30",
    ]
    # A period may start at the first close, which no return covers.
    p = prices().pivot(index="date", columns="ticker", values="close")
    p.index = pd.DatetimeIndex(p.index)
    first = written["stress"][0]
    assert first["funds"]["AAA"] == pytest.approx(
        p.loc["2023-06-30", "AAA"] / p.loc["2023-01-02", "AAA"] - 1, rel=1e-12
    )
    growth = p.loc["2023-06-30"] / p.loc["2023-01-02"]
    assert first["mixes"]["AAA=0.7,BBB=0.3"]["never"]["return"] == pytest.approx(
        0.7 * growth["AAA"] + 0.3 * growth["BBB"] - 1, rel=1e-12
    )
    days = written["history"]["equal"]["monthly"]
    returns = (p / p.shift(1) - 1).iloc[1:]
    years = (p.index[-1] - p.index[0]).days / 365.25
    assert days["turnover_per_year"] == pytest.approx(
        backtest(returns, {"AAA": 0.5, "BBB": 0.5}, "monthly").turnover.sum() / years,
        rel=1e-9,
    )
    # The whole history's compound return is per calendar year.
    years = (p.index[-1] - p.index[0]).days / 365.25
    total = 0.5 * p.iloc[-1]["AAA"] / p.iloc[0]["AAA"] + 0.5 * p.iloc[-1]["BBB"] / p.iloc[0]["BBB"]
    assert written["history"]["equal"]["never"]["compound_return"] == pytest.approx(
        total ** (1 / years) - 1, rel=1e-9
    )


def test_a_named_mix_may_be_written_in_any_case(tmp_path: Path) -> None:
    out = tmp_path / "s.json"
    argv = ["--cache", str(cache(tmp_path)), "--json", str(out), "--rule", "Daily"]
    assert main([*argv, "--mix", " EQUAL"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["mixes"] == {
        " EQUAL": {"AAA": 0.5, "BBB": 0.5}
    }


def test_named_periods_outside_the_data_are_listed_as_left_out(tmp_path: Path, capsys) -> None:
    out = tmp_path / "s.json"
    assert main(["--cache", str(cache(tmp_path)), "--mix", "equal", "--json", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["stress"] == []
    assert written["stress_periods_left_out"] == list(STRESS_PERIODS)
    printed = capsys.readouterr().out
    assert "no stress period lies within the data" in printed
    assert written["rules"] == ["daily", "monthly", "quarterly", "yearly", "never", "band:0.05"]


def test_the_command_refuses_bad_input_plainly(tmp_path: Path, capsys) -> None:
    assert main(["--cache", str(tmp_path / "prices.csv")]) == 1
    assert "manifest.json is missing" in capsys.readouterr().err
    path = cache(tmp_path)
    for argv, message in [
        ([], "no returns for EEM, EFA, IWM, SPY"),  # the named mixes need the eight funds
        (["--mix", "equal", "--mix", "equal"], "a mix is named twice"),
        (["--mix", "equal", "--rule", "never", "--rule", "never"], "a rule is named twice"),
        (["--mix", "equal", "--rule", "band:0.05", "--rule", "band:.05"], "rule is named twice"),
        (["--mix", "equal", "--rule", "Never", "--rule", "never"], "a rule is named twice"),
        (["--mix", "AAA=0.5,BBB=0.5", "--mix", "equal"], "AAA=0.5,BBB=0.5 and equal are the same"),
        (["--mix", "AAA=0.5,BBB=0.5", "--mix", "BBB=0.5,AAA=0.5"], "are the same mix"),
        (["--mix", "equty"], "a mix is equity, 60/40, equal, permanent or weights"),
        (["--mix", "equal", "--rule", "fortnightly"], "a rule is"),
        (["--mix", "equal", "--period", "2023-03-04:2023-06-30"], "2023-03-04 is not a day"),
        (["--mix", "equal", "--period", "March"], "START:END"),
        (["--mix", "equal", "--period", "2023-06-30:2023-03-01"], "end after it starts"),
        (["--mix", "equal", "--period", "2020-02-19:2020-03-23"], "not a day"),
        (
            [
                "--mix",
                "equal",
                "--period",
                "2024-03-01:2024-04-30",
                "--period",
                "2024-03-01:2024-04-30",
            ],
            "a period is named twice",
        ),
        (["--mix", "equal", "--json", str(tmp_path)], ""),  # a folder
    ]:
        assert main(["--cache", str(path), *argv]) == 1, argv
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error: ") and message in captured.err
        assert "Traceback" not in captured.err
