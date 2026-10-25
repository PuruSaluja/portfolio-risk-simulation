import numpy as np
import pandas as pd
import pytest

from portfolio_risk import costs
from portfolio_risk.montecarlo import check_weights
from portfolio_risk.returns import annualised_return
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
            holdings = target * root(target, d, k) * total
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


def gap(returns, weights, rules, k):
    first, second = (costs.trades(backtest(returns, weights, r), returns) for r in rules)
    return annualised_return(costs.net_returns(first, k), 252) - annualised_return(
        costs.net_returns(second, k), 252
    )


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
