import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import costs
from portfolio_risk.costs import main
from portfolio_risk.data import COLUMNS, load_cache, write_cache
from portfolio_risk.montecarlo import check_weights, historical_portfolio_returns
from portfolio_risk.returns import annualised_return, observed_periods_per_year, total_returns
from portfolio_risk.risk import sharpe_ratio
from portfolio_risk.scenarios import backtest


def frame(days=300, funds=3, seed=0, scale=0.01):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=days)
    columns = [f"F{i}" for i in range(funds)]
    drift = np.linspace(-0.0002, 0.0006, funds)
    values = drift + rng.normal(0.0, scale, (days, funds)) * np.linspace(0.5, 2.0, funds)
    return pd.DataFrame(values, index=index, columns=columns)


def root(w, d, k):
    """The value left after trading, by bisection on the cost equation."""
    low, high = 0.0, 1.0
    while True:
        middle = (low + high) / 2.0
        if not low < middle < high:
            return high
        if middle + (k * np.abs(w * middle - d)).sum() < 1.0:
            low = middle
        else:
            high = middle


def brute_force(returns, weights, rule, k):
    """Net daily returns from holdings in money, paying the costs inside the loop, a band
    rule deciding from the weights the costs leave."""
    values = returns.to_numpy()
    target = check_weights(weights, list(returns.columns))
    calendar = backtest(returns, weights, rule).rebalanced.to_numpy()
    band = float(rule.split(":")[1]) if rule.startswith("band") else None
    holdings = target.copy()
    out = []
    for t in range(len(values)):
        start = holdings.sum()
        holdings = holdings * (1.0 + values[t])
        total = holdings.sum()
        d = holdings / total
        if band is None:
            due = calendar[t]
        else:
            due = t < len(values) - 1 and np.abs(d - target).max() > band
        if due:
            k_t = k.iloc[t].to_numpy() if isinstance(k, pd.DataFrame) else k
            holdings = target * root(target, d, k_t) * total
        out.append(holdings.sum() / start - 1.0)
    return np.array(out)


def test_the_cost_per_unit_is_the_commission_plus_half_the_spread() -> None:
    tickers = ["AAA", "BBB", "CCC"]
    np.testing.assert_allclose(costs.per_unit(0.0005, 0.0002, tickers), [0.0006] * 3, rtol=1e-15)
    k = costs.per_unit(0.0001, {"AAA": 0.0002, "CCC": 0.001}, tickers)
    np.testing.assert_allclose(k, [0.0002, 0.0001, 0.0006], rtol=1e-15)
    np.testing.assert_array_equal(costs.per_unit(0.0, 0.0, tickers), [0.0] * 3)


@pytest.mark.parametrize(
    "commission, spreads, message",
    [
        (0.0, {"ZZZ": 0.001}, "no returns for ZZZ"),
        (-0.0001, 0.0, "not negative"),
        (0.0, float("nan"), "finite"),
        (float("inf"), 0.0, "finite"),
        (True, 0.0, "is a number"),
        (0.0, "0.001", "is a number"),
        (0.09, 0.03, "above 10%"),
    ],
)
def test_a_cost_that_is_not_a_cost_is_refused(commission, spreads, message) -> None:
    with pytest.raises(ValueError, match=message):
        costs.per_unit(commission, spreads, ["AAA", "BBB"])


def one_trade(d, w=(0.5, 0.5)):
    """Two funds, two days: drifted to ``d`` at the first close, rebalanced there."""
    w = np.array(w)
    d = np.array(d)
    # Returns that take the target weights to d over the first day, nothing the second.
    growth = d / w
    days = pd.bdate_range("2024-01-01", periods=2)
    returns = pd.DataFrame([growth - 1.0, [0.0, 0.0]], index=days, columns=["A", "B"])
    test = backtest(returns, {"A": w[0], "B": w[1]}, "daily")
    return costs.trades(test, returns), growth @ w


def test_one_trade_by_hand() -> None:
    trades, growth = one_trade([0.6, 0.4])
    np.testing.assert_allclose(trades.drifted, [[0.6, 0.4]], rtol=1e-15)
    # Sell 0.6 - 0.5 v of A, buy 0.5 v - 0.4 of B: 0.2 traded in all, so v = 1 - 0.2 k.
    k = 0.001
    assert costs.paid(trades, k).iloc[0] == pytest.approx(0.2 * k, rel=1e-12)
    assert costs.paid(trades, k).iloc[1] == 0.0
    # One cost each: v (1 - k1 / 2 + k2 / 2) = 1 - 0.6 k1 + 0.4 k2.
    k1, k2 = 0.002, 0.0005
    v = (1 - 0.6 * k1 + 0.4 * k2) / (1 - 0.5 * k1 + 0.5 * k2)
    assert costs.paid(trades, np.array([k1, k2])).iloc[0] == pytest.approx(1 - v, rel=1e-12)
    net = costs.net_returns(trades, k)
    assert net.iloc[0] == pytest.approx(growth * (1 - 0.2 * k) - 1, rel=1e-12)
    assert net.iloc[1] == 0.0


@pytest.mark.parametrize("n", [2, 3, 8, 50])
def test_newton_s_method_lands_on_the_root(n) -> None:
    rng = np.random.default_rng(n)
    for _ in range(200):
        w = rng.dirichlet(np.ones(n))
        d = rng.dirichlet(np.ones(n))
        k = rng.uniform(0.0, costs.HIGHEST, n) * rng.choice([1e-3, 0.1, 1.0])
        trades = costs.Trades(pd.Series([0.0]), w, np.array([True]), d[None, :])
        v = 1.0 - costs.paid(trades, k).iloc[0]
        assert abs(v + (k * np.abs(w * v - d)).sum() - 1.0) <= 4e-16
        assert v == pytest.approx(root(w, d, k), abs=1e-15)


def test_newton_s_method_goes_on_while_trades_turn_from_buys_to_sells() -> None:
    # A fund far overweight and one far under, so the costs are large; the rest a
    # little underweight, by less and less, so they turn from buys at v = 1 to sells
    # nearer the root. Two Newton steps leave this one 3e-5 short; three are exact.
    n = 20
    w = np.concatenate([[0.05, 0.45], np.full(n - 2, 0.5 / (n - 2))])
    rest = w[2:] * (1 - np.geomspace(0.0749, 0.0749 * 0.00265, n - 2))
    d = np.concatenate([[1 - 0.05 - rest.sum()], [0.05], rest])
    k = np.full(n, costs.HIGHEST)
    trades = costs.Trades(pd.Series([0.0]), w, np.array([True]), d[None, :])
    v = 1.0 - costs.paid(trades, k).iloc[0]
    assert v == pytest.approx(root(w, d, k), abs=1e-15)
    # Only the overweight fund is a sale at v = 1; at the root all but one are.
    assert (w - d < 0).sum() == 1 and (w * v - d < 0).sum() == n - 1


@pytest.mark.parametrize("rule", ["daily", "monthly", "never", "band:0.03", "band:0.1"])
def test_net_returns_match_paying_the_costs_inside_the_loop(rule) -> None:
    returns = frame(days=400, funds=4, seed=1, scale=0.02)
    weights = {"F0": 0.1, "F1": 0.2, "F2": 0.3, "F3": 0.4}
    trades = costs.trades(backtest(returns, weights, rule), returns)
    for k in (0.0, 0.0005, np.array([0.0001, 0.0002, 0.003, 0.05])):
        np.testing.assert_allclose(
            costs.net_returns(trades, k), brute_force(returns, weights, rule, k), atol=1e-15
        )


@pytest.mark.parametrize("rule", ["daily", "monthly", "band:0.03"])
def test_costs_that_change_by_day_and_fund_are_paid_on_each_day(rule) -> None:
    returns = frame(days=400, funds=4, seed=4, scale=0.02)
    weights = {"F0": 0.1, "F1": 0.2, "F2": 0.3, "F3": 0.4}
    made = costs.trades(backtest(returns, weights, rule), returns)
    rng = np.random.default_rng(5)
    by_day = pd.DataFrame(
        rng.uniform(0.0, 0.01, returns.shape), index=returns.index, columns=returns.columns
    )
    np.testing.assert_allclose(
        costs.net_returns(made, by_day), brute_force(returns, weights, rule, by_day), atol=1e-15
    )
    # The same cost every day is the constant cost.
    flat = pd.DataFrame(0.002, index=returns.index, columns=returns.columns)
    assert costs.net_returns(made, flat).equals(costs.net_returns(made, 0.002))


def test_costs_by_day_off_the_backtest_s_dates_or_funds_are_refused() -> None:
    returns = frame()
    made = costs.trades(backtest(returns, {"F0": 0.5, "F1": 0.5}, "daily"), returns)
    good = pd.DataFrame(0.001, index=returns.index, columns=returns.columns)
    for bad in (good.iloc[1:], good[["F2", "F1", "F0"]], good.set_axis(["a", "b", "c"], axis=1)):
        with pytest.raises(ValueError, match="on the backtest's dates, a column per fund"):
            costs.paid(made, bad)
    negative = good.copy()
    negative.iloc[3] = -0.001
    with pytest.raises(ValueError, match="one-way cost is from 0"):
        costs.paid(made, negative)


def test_without_costs_the_returns_are_the_backtest_s() -> None:
    returns = frame()
    test = backtest(returns, {"F0": 0.5, "F2": 0.5}, "monthly")
    trades = costs.trades(test, returns)
    assert costs.net_returns(trades, 0.0).equals(test.returns.rename(None))
    assert (costs.paid(trades, 0.0) == 0.0).all()
    np.testing.assert_allclose(trades.turnover, test.turnover[test.rebalanced], atol=1e-16)


def test_the_summary_s_drag_is_the_return_costs_take() -> None:
    returns = frame(days=500, funds=3, seed=2, scale=0.015)
    test = backtest(returns, {"F0": 0.3, "F1": 0.3, "F2": 0.4}, "daily")
    trades = costs.trades(test, returns)
    out = costs.summarise(trades, 0.001, 252)
    net = costs.net_returns(trades, 0.001)
    assert out["compound_return"] == pytest.approx(annualised_return(test.returns, 252))
    assert out["net_compound_return"] == pytest.approx(annualised_return(net, 252))
    assert out["drag"] == out["compound_return"] - out["net_compound_return"] > 0.0
    years = 500 / 252
    assert out["turnover_per_year"] == pytest.approx(test.turnover.sum() / years, rel=1e-12)
    assert out["paid_per_year"] == pytest.approx(costs.paid(trades, 0.001).sum() / years)
    # Each unit of one-way turnover is a sell and a buy: about twice the cost each.
    assert out["paid_per_year"] == pytest.approx(2 * 0.001 * out["turnover_per_year"], rel=1e-3)


def test_a_held_portfolio_is_charged_under_each_rule_at_each_cost() -> None:
    returns = frame(days=500, funds=4, seed=6, scale=0.015)
    weights = {"F0": 0.1, "F1": 0.0, "F2": 0.5, "F3": 0.4}
    first = returns.index[0] - pd.Timedelta(days=3)
    per_year = observed_periods_per_year(pd.DatetimeIndex([first, *returns.index]))
    rates = pd.Series(np.linspace(0.0, 0.0002, 500), index=returns.index)
    levels = (0.0, 0.0005, 0.01)
    for risk_free in (0.0, 0.03, rates):
        out = costs.held(returns, weights, first, risk_free, levels=levels)
        assert list(out) == ["daily", "monthly", "band:0.05"]
        for rule, figures in out.items():
            test = backtest(returns, weights, rule)
            assert figures["rebalances"] == test.rebalances
            assert figures["turnover_per_year"] == pytest.approx(
                test.turnover.sum() / (500 / per_year), rel=1e-12
            )
            assert [c["cost"] for c in figures["costs"]] == list(levels)
            for c in figures["costs"]:
                # Against the loop that holds the funds in money and pays inside it.
                net = pd.Series(brute_force(returns, weights, rule, c["cost"]), returns.index)
                assert c["compound_return"] == pytest.approx(
                    annualised_return(net, per_year), rel=1e-12
                )
                assert c["sharpe_ratio"] == pytest.approx(
                    sharpe_ratio(net, 252, risk_free), rel=1e-12
                )
        # Costs only take away, and take more from the rule that trades more.
        for figures in out.values():
            compound = [c["compound_return"] for c in figures["costs"]]
            assert compound == sorted(compound, reverse=True)
        daily, monthly = (out[r]["costs"] for r in ("daily", "monthly"))
        assert (
            daily[0]["compound_return"] - daily[-1]["compound_return"]
            > monthly[0]["compound_return"] - monthly[-1]["compound_return"]
        )


def test_held_daily_without_costs_is_the_portfolio_rebalanced_every_close() -> None:
    returns = frame(days=300, funds=3, seed=7)
    weights = {"F0": 0.2, "F1": 0.3, "F2": 0.5}
    first = returns.index[0] - pd.Timedelta(days=1)
    daily = costs.held(returns, weights, first, 0.02, rules=["daily"], levels=[0.0])["daily"]
    history = historical_portfolio_returns(returns, weights)
    per_year = observed_periods_per_year(pd.DatetimeIndex([first, *returns.index]))
    zero = daily["costs"][0]
    assert zero["compound_return"] == pytest.approx(annualised_return(history, per_year), rel=1e-13)
    assert zero["sharpe_ratio"] == pytest.approx(sharpe_ratio(history, 252, 0.02), rel=1e-12)
    # Every close but the last trades.
    assert daily["rebalances"] == 299


def test_held_gives_no_ratio_for_returns_that_never_vary() -> None:
    returns = pd.DataFrame(0.0, index=pd.bdate_range("2024-01-01", periods=30), columns=["A", "B"])
    first = returns.index[0] - pd.Timedelta(days=3)
    out = costs.held(returns, {"A": 0.5, "B": 0.5}, first, levels=[0.0, 0.001])
    for figures in out.values():
        assert [c["sharpe_ratio"] for c in figures["costs"]] == [None, None]
        # Nothing drifts, so nothing is traded and nothing paid.
        assert [c["compound_return"] for c in figures["costs"]] == [0.0, 0.0]


@pytest.mark.parametrize(
    "rules, levels, message",
    [
        ([], (0.0,), "at least one rule and one cost"),
        (["daily"], (), "at least one rule and one cost"),
        (["daily", "Daily"], (0.0,), "a rule is named twice"),
        (["band:0.05", "band:.05"], (0.0,), "a rule is named twice"),
        (["daily"], (0.001, 0.001), "a cost is given twice"),
        (["daily"], (-0.001,), "not negative"),
        (["daily"], (0.2,), "above 10%"),
        (["weekly"], (0.0,), "a rule is daily"),
    ],
)
def test_held_refuses_bad_rules_and_costs(rules, levels, message) -> None:
    returns = frame(days=50)
    with pytest.raises(ValueError, match=message):
        costs.held(returns, {"F0": 1.0}, returns.index[0], rules=rules, levels=levels)


def gap(returns, weights, rules, k):
    first, second = (costs.trades(backtest(returns, weights, r), returns) for r in rules)
    return annualised_return(costs.net_returns(first, k), 252) - annualised_return(
        costs.net_returns(second, k), 252
    )


@pytest.mark.parametrize("rule", ["daily", "monthly", "yearly", "band:0.03"])
@pytest.mark.parametrize("highest", [0.01, costs.HIGHEST])
def test_the_compound_return_falls_no_faster_than_its_bound(rule, highest) -> None:
    for seed in range(3):
        returns = frame(days=500, funds=4, seed=seed, scale=0.02)
        weights = {"F0": 0.1, "F1": 0.2, "F2": 0.3, "F3": 0.4}
        made = costs.trades(backtest(returns, weights, rule), returns)
        bound = costs.fall_bound(made, 252, highest)
        k = np.linspace(0.0, highest, 401)
        compound = np.array([annualised_return(costs.net_returns(made, x), 252) for x in k])
        steepest = float(np.max(-np.diff(compound) / np.diff(k)))
        assert 0.0 < steepest <= bound
        # Close to the steepest fall near no cost, so the search's steps are not tiny.
        assert bound <= 1.35 * steepest


def test_the_first_crossing_is_found_where_a_scan_would_step_over_it() -> None:
    # A lead that dips below zero for less than a tenth of a basis point around 1.5 bp,
    # and falls no faster than 200 per unit of cost.
    def gap(k):
        return 1e-4 - 2e-4 * math.exp(-(((k - 1.5e-4) / 1e-6) ** 2))

    assert gap(1e-4) > 0 and gap(2e-4) > 0  # a basis-point scan sees a lead throughout
    found = costs.first_crossing(gap, gap(0.0), 200.0, 0.01)
    assert found == pytest.approx(1.5e-4 - math.sqrt(math.log(2)) * 1e-6, abs=1e-15)
    assert gap(found) <= 0.0 < gap(np.nextafter(found, 0.0))


def test_the_search_steps_no_further_than_the_lead_allows() -> None:
    # A lead falling at half the bound, then at the bound itself to 0 at 1.75e-3, then
    # back above 0 from 1.975e-3: a step longer than lead / slope, or one that is not
    # shortened as the lead shrinks, lands past the crossing on a lead and misses it.
    def gap(k):
        if k < 1.5e-3:
            return 1e-3 - 0.5 * k
        if k < 1.9e-3:
            return 2.5e-4 - (k - 1.5e-3)
        return -1.5e-4 + 2.0 * (k - 1.9e-3)

    assert costs.first_crossing(gap, 1e-3, 1.0, 0.01) == pytest.approx(1.75e-3, abs=1e-15)


def test_the_first_crossing_is_none_where_the_gap_never_closes() -> None:
    assert costs.first_crossing(lambda k: 1e-3 - k, 1e-3, 1.0, 0.0005) is None
    assert costs.first_crossing(lambda k: 1e-3, 1e-3, 0.0, 0.01) is None
    # Closing exactly at the highest cost counts.
    assert costs.first_crossing(lambda k: 0.01 - k, 0.01, 1.0, 0.01) == 0.01


def test_breakeven_is_where_the_lead_is_gone() -> None:
    returns = frame(days=750, funds=3, seed=3, scale=0.02)
    weights = {"F0": 1 / 3, "F1": 1 / 3, "F2": 1 / 3}
    rules = ["daily", "never"]
    found = costs.breakeven(returns, weights, rules, 252)
    assert found["gap"] == pytest.approx(gap(returns, weights, rules, 0.0)) and found["gap"] > 0
    cost = found["cost"]
    assert 0.0 < cost < 0.01
    assert gap(returns, weights, rules, cost) <= 0.0
    assert gap(returns, weights, rules, np.nextafter(cost, 0.0)) > 0.0
    # Lower costs on the scan all leave the lead.
    for k in np.arange(0.0, cost, 1e-4):
        assert gap(returns, weights, rules, k) > 0.0
    behind = costs.breakeven(returns, weights, rules[::-1], 252)
    assert behind["cost"] is None and behind["gap"] == pytest.approx(-found["gap"])
    short = costs.breakeven(returns, weights, rules, 252, highest=cost / 2)
    assert short["cost"] is None and short["highest"] == cost / 2


def test_bad_arguments_are_refused() -> None:
    returns = frame()
    weights = {"F0": 1.0}
    for rules in (["daily"], ["daily", "monthly", "never"]):
        with pytest.raises(ValueError, match="two rules"):
            costs.breakeven(returns, weights, rules, 252)
    for highest in (0.0, 0.2):
        with pytest.raises(ValueError, match="highest"):
            costs.breakeven(returns, weights, ["daily", "never"], 252, highest=highest)
    trades = costs.trades(backtest(returns, weights, "daily"), returns)
    for bad in (-0.001, 0.11, float("nan"), np.array([0.0, 0.0, 0.2])):
        with pytest.raises(ValueError, match="one-way cost"):
            costs.paid(trades, bad)
    test = backtest(returns, weights, "daily")
    with pytest.raises(ValueError, match="those the backtest was run on"):
        costs.trades(test, returns.iloc[1:])
    with pytest.raises(ValueError, match="those the backtest was run on"):
        costs.trades(test, returns.set_axis(returns.index + pd.offsets.BDay(1)))
    with pytest.raises(ValueError, match="those the backtest was run on"):
        costs.trades(test, returns[["F2", "F1", "F0"]])


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


def test_the_command_writes_every_mix_rule_and_cost_and_the_break_even_costs(
    tmp_path: Path, capsys
) -> None:
    out = tmp_path / "results" / "costs.json"
    path = cache(tmp_path)
    argv = ["--cache", str(path), "--json", str(out), "--mix", "AAA=0.7,BBB=0.3"]
    argv += ["--mix", "equal", "--rule", "daily", "--rule", "never", "--rule", "band:0.02"]
    argv += ["--cost", "0", "--cost", "0.001", "--spread", "AAA=0.002", "--compare", "daily,never"]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "plus half of each fund's spread: AAA 20 bp" in printed
    assert "daily over never: lead before costs, and the cost that ends it" in printed
    assert b"\r" not in out.read_bytes()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["costs"] == [0.0, 0.001] and written["spreads"] == {"AAA": 0.002, "BBB": 0.0}
    assert written["rules"] == ["daily", "never", "band:0.02"]
    daily = total_returns(load_cache(path))
    per_year = observed_periods_per_year(pd.DatetimeIndex(["2023-01-02", *daily.index]))
    weights = {"AAA": 0.7, "BBB": 0.3}
    for rule in written["rules"]:
        test = backtest(daily, weights, rule)
        made = costs.trades(test, daily)
        entry = written["history"]["AAA=0.7,BBB=0.3"][rule]
        assert entry["rebalances"] == test.rebalances
        assert entry["compound_return"] == pytest.approx(
            annualised_return(test.returns, per_year), rel=1e-12
        )
        for level, figures in zip([0.0, 0.001], entry["costs"], strict=True):
            expected = costs.summarise(made, np.array([level + 0.001, level]), per_year)
            assert figures["cost"] == level
            for key in ("net_compound_return", "drag", "paid_per_year"):
                assert figures[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-18)
    never = written["history"]["equal"]["never"]["costs"][1]
    assert never["drag"] == 0.0 and never["paid_per_year"] == 0.0
    daily_ = written["history"]["equal"]["daily"]["costs"]
    assert daily_[1]["drag"] > daily_[0]["drag"] > 0.0
    [found] = written["breakeven"]
    assert found["rules"] == ["daily", "never"]
    for spec, w in [("AAA=0.7,BBB=0.3", weights), ("equal", {"AAA": 0.5, "BBB": 0.5})]:
        # Break-even costs are the same for every fund and leave the spreads out.
        expected = costs.breakeven(daily, w, ["daily", "never"], per_year)
        assert found["mixes"][spec] == {k: v for k, v in expected.items() if k != "rules"}


def test_the_defaults_are_every_mix_rule_and_cost(tmp_path: Path, capsys) -> None:
    out = tmp_path / "costs.json"
    # The default mixes name funds this cache lacks.
    assert main(["--cache", str(cache(tmp_path)), "--json", str(out)]) == 1
    assert "no returns for" in capsys.readouterr().err
    assert main(["--cache", str(cache(tmp_path)), "--json", str(out), "--mix", "equal"]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["costs"] == list(costs.LEVELS)
    assert list(written["history"]["equal"]) == [
        "daily",
        "monthly",
        "quarterly",
        "yearly",
        "never",
        "band:0.05",
    ]
    assert [b["rules"] for b in written["breakeven"]] == [
        ["daily", "monthly"],
        ["daily", "band:0.05"],
        ["band:0.05", "never"],
    ]
    assert written["spreads"] == {"AAA": 0.0, "BBB": 0.0}
    assert written["history"]["equal"]["daily"]["costs"][0]["drag"] == 0.0
    # One spread for every fund: half of it is paid even at no commission.
    argv = ["--cache", str(cache(tmp_path)), "--json", str(out), "--mix", "equal"]
    assert main([*argv, "--spread", "0.0004", "--cost", "0"]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["spreads"] == {"AAA": 0.0004, "BBB": 0.0004}
    daily = total_returns(load_cache(cache(tmp_path)))
    made = costs.trades(backtest(daily, {"AAA": 0.5, "BBB": 0.5}, "daily"), daily)
    per_year = observed_periods_per_year(pd.DatetimeIndex(["2023-01-02", *daily.index]))
    assert written["history"]["equal"]["daily"]["costs"][0]["drag"] == pytest.approx(
        costs.summarise(made, 0.0002, per_year)["drag"], rel=1e-12
    )


def test_the_command_refuses_bad_input_plainly(tmp_path: Path, capsys) -> None:
    assert main(["--cache", str(tmp_path / "prices.csv")]) == 1
    assert "manifest.json is missing" in capsys.readouterr().err
    path = cache(tmp_path)
    for argv, message in [
        (["--cost", "five"], "a cost is a number such as 0.0005"),
        (["--cost", "0.001", "--cost", "0.001"], "a cost is given twice"),
        (["--cost", "0.2"], "above 10%"),
        (["--cost", "-0.001"], "a cost or spread is finite and not negative"),
        (["--cost", "nan"], "a cost or spread is finite and not negative"),
        (["--spread", "inf"], "a cost or spread is finite and not negative"),
        (["--spread", "wide"], "a spread is a number such as 0.0002"),
        (["--spread", "AAA=x"], "a spread is a number such as 0.0002"),
        (["--spread", "=0.001"], "a spread is a number such as 0.0002"),
        (["--spread", "AAA=0.001, =0.002"], "a spread is a number such as 0.0002"),
        (["--spread", "ZZZ=0.001"], "no returns for ZZZ"),
        (["--spread", "AAA=0.001,aaa=0.002"], "given twice"),
        (["--compare", "daily"], "two different rules"),
        (["--compare", "daily,Daily"], "two different rules"),
        (["--rule", "daily", "--rule", "DAILY"], "a rule is named twice"),
        (["--rule", "weekly"], "a rule is daily, monthly"),
        (["--mix", "equal"], "a mix is named twice"),
        (["--mix", "AAA=0.5,BBB=0.5"], "equal and AAA=0.5,BBB=0.5 are the same mix"),
        (["--json", str(tmp_path)], ""),  # a folder
    ]:
        assert main(["--cache", str(path), "--mix", "equal", *argv]) == 1, argv
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error: ") and message in captured.err, argv


def split_prices() -> pd.DataFrame:
    """Two funds over the start of the tick rule; BBB splits 3 for 1 on 2006-02-15."""
    days = pd.bdate_range("2006-01-02", "2006-03-31")
    rows = []
    rng = np.random.default_rng(6)
    for ticker, start in (("AAA", 20.0), ("BBB", 150.0)):
        close = start * np.cumprod(1.0 + rng.normal(0.0005, 0.01, len(days)))
        for day, value in zip(days, close, strict=True):
            split = 3.0 if ticker == "BBB" and day == pd.Timestamp("2006-02-15") else 0.0
            # The cache's closes are adjusted for later splits: BBB's are a third before.
            adjusted = value / 3.0 if ticker == "BBB" else value
            rows.append((day.date(), ticker, float(adjusted), 0.0, 0.0, split, 1e6))
    return pd.DataFrame(rows, columns=list(COLUMNS))


def test_closes_before_a_split_are_put_back_on_their_traded_basis() -> None:
    prices = split_prices()
    traded = costs.traded_closes(prices)
    cached = prices.pivot(index="date", columns="ticker", values="close")
    cached.index = pd.DatetimeIndex(cached.index)
    before = traded.index < pd.Timestamp("2006-02-15")
    np.testing.assert_allclose(traded.loc[before, "BBB"], 3 * cached.loc[before, "BBB"])
    np.testing.assert_array_equal(traded.loc[~before, "BBB"], cached.loc[~before, "BBB"])
    np.testing.assert_array_equal(traded["AAA"], cached["AAA"])


def test_the_tick_floor_is_half_a_cent_of_the_traded_close_from_the_rule() -> None:
    prices = split_prices()
    dates = total_returns(prices).index
    floor = costs.tick_floor(prices, dates)
    traded = costs.traded_closes(prices).loc[dates]
    early = dates < pd.Timestamp("2006-01-31")
    assert early.any() and (floor[early] == 0.0).all().all()
    np.testing.assert_allclose(floor[~early], 0.005 / traded[~early], rtol=1e-15)
    # A third of the cached close would give three times the floor.
    assert floor.loc["2006-02-14", "BBB"] < 0.005 / prices[prices["ticker"] == "BBB"]["close"].max()
    with pytest.raises(ValueError, match="a close for every fund"):
        costs.tick_floor(prices, dates.append(pd.DatetimeIndex(["2006-04-03"])))
    cheap = prices.assign(close=prices["close"] / 100.0)
    with pytest.raises(ValueError, match=r"below \$1"):
        costs.tick_floor(cheap, dates)


def test_the_command_charges_the_tick_floor_each_day(tmp_path: Path, capsys) -> None:
    path = tmp_path / "prices.csv"
    write_cache(split_prices(), path, start=date(2006, 1, 1), end=date(2006, 3, 31), source="t")
    out = tmp_path / "tick.json"
    argv = ["--cache", str(path), "--json", str(out), "--mix", "equal", "--rule", "daily"]
    assert main([*argv, "--spread", "Tick", "--cost", "0", "--cost", "0.001"]) == 0
    assert "plus the tick floor, half of a 0.01 tick" in capsys.readouterr().out
    written = json.loads(out.read_text(encoding="utf-8"))
    loaded = load_cache(path)
    daily = total_returns(loaded)
    floor = costs.tick_floor(loaded, daily.index)
    assert written["spreads"]["mean_one_way"]["BBB"] == pytest.approx(floor["BBB"].mean())
    after = floor.loc["2006-01-31":, "AAA"].mean()
    assert written["spreads"]["mean_one_way_from_tick_rule"]["AAA"] == pytest.approx(after)
    made = costs.trades(backtest(daily, {"AAA": 0.5, "BBB": 0.5}, "daily"), daily)
    per_year = observed_periods_per_year(pd.DatetimeIndex(["2006-01-02", *daily.index]))
    for level, entry in zip(
        [0.0, 0.001], written["history"]["equal"]["daily"]["costs"], strict=True
    ):
        expected = costs.summarise(made, floor + level, per_year)
        assert entry["drag"] == pytest.approx(expected["drag"], rel=1e-12)
    assert written["history"]["equal"]["daily"]["costs"][0]["drag"] > 0.0
    with pytest.raises(ValueError, match="or tick, not 'ticks'"):
        costs.run(loaded, ["equal"], ["daily"], spreads="ticks")
