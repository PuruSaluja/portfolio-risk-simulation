"""Answers worked out by hand, which the modules must reach between them.

Each case is built so that its answer has a closed form: the tangency portfolio when no
bound binds, a fund that alternates between a rise and the fall that undoes it held
against cash (rebalancing, its costs and the cost at which it stops paying), and
minimum CVaR where the answer follows from CVaR being positively homogeneous.
"""

import math

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import costs, risk
from portfolio_risk import frontier as fr
from portfolio_risk.construction import minimum_cvar
from portfolio_risk.returns import annualised_return
from portfolio_risk.scenarios import backtest


def test_the_tangency_portfolio_is_the_inverse_covariance_times_the_excess_returns() -> None:
    # With no bound binding, w is proportional to S^-1 (mu - r) and the ratio is
    # sqrt((mu - r)' S^-1 (mu - r)).
    funds = ["A", "B", "C"]
    mu = pd.Series([0.08, 0.05, 0.06], index=funds)
    covariance = pd.DataFrame(
        [[0.04, 0.006, 0.01], [0.006, 0.01, 0.002], [0.01, 0.002, 0.0225]],
        index=funds,
        columns=funds,
    )
    for annual in (0.0, 0.02):
        rate = ((1 + annual) ** (1 / 252) - 1) * 252
        excess = (mu - rate).to_numpy()
        inverse = np.linalg.solve(covariance.to_numpy(), excess)
        expected = inverse / inverse.sum()
        assert (expected > 0).all()
        w = fr.max_sharpe(mu, covariance, annual)
        assert w.to_numpy() == pytest.approx(expected, abs=1e-12)
        assert fr.sharpe(w, mu, covariance, annual) == pytest.approx(
            math.sqrt(excess @ inverse), rel=1e-12
        )


def alternating(a: float, days: int) -> pd.DataFrame:
    """A fund that rises by ``a`` and then falls back, every two days, beside cash."""
    up, down = a, 1 / (1 + a) - 1
    fund = np.where(np.arange(days) % 2 == 0, up, down)
    return pd.DataFrame({"A": fund, "CASH": 0.0}, index=pd.bdate_range("2024-01-01", periods=days))


A, DAYS = 0.1, 252
HALF = {"A": 0.5, "CASH": 0.5}


def test_rebalancing_a_fund_that_comes_back_against_cash_by_hand() -> None:
    returns = alternating(A, DAYS)
    # Left alone, the fund ends every second day where it began: buy and hold earns
    # nothing.
    held = backtest(returns, HALF, "never")
    assert (1 + held.returns).prod() == pytest.approx(1.0, abs=1e-12)
    # Rebalanced every close, a pair of days grows by (1 + a/2)(1 - a / (2 (1 + a))),
    # which is (2 + a)^2 / (4 (1 + a)) > 1.
    daily = backtest(returns, HALF, "daily")
    pair = (2 + A) ** 2 / (4 * (1 + A))
    assert (1 + daily.returns).prod() == pytest.approx(pair ** (DAYS / 2), rel=1e-12)
    # Each close the fund has drifted to (1 + a) / (2 + a) or 1 / (2 + a): a / (2 (2 + a))
    # is traded, one way, on every close but the last.
    turnover = daily.turnover.to_numpy()
    assert turnover[:-1] == pytest.approx(np.full(DAYS - 1, A / (2 * (2 + A))), rel=1e-12)
    assert turnover[-1] == 0.0
    # Each pair falls back by a / (2 (1 + a)) from the day before, and every pair ends
    # higher than the last, so that is the deepest drawdown.
    assert risk.max_drawdown(daily.returns).depth == pytest.approx(-A / (2 * (1 + A)), rel=1e-12)
    # Two returns, alternating: the mean is their average and the sample deviation half
    # their difference, scaled by n / (n - 1).
    x, y = A / 2, -A / (2 * (1 + A))
    sd = abs(x - y) / 2 * math.sqrt(DAYS / (DAYS - 1))
    assert risk.sharpe_ratio(daily.returns, 252) == pytest.approx(
        (x + y) / 2 / sd * math.sqrt(252), rel=1e-12
    )


def test_each_rebalance_s_cost_and_the_cost_at_which_it_stops_paying_by_hand() -> None:
    returns = alternating(A, DAYS)
    made = costs.trades(backtest(returns, HALF, "daily"), returns)
    # Selling the fund's excess and buying cash, v + k (d_A - v/2) + k (v/2 - d_cash) = 1,
    # so each trade leaves v = 1 - k a / (2 + a), the same on the way up and down.
    share = A / (2 + A)
    k = 0.002
    paid = costs.paid(made, k).to_numpy()
    assert paid[:-1] == pytest.approx(np.full(DAYS - 1, k * share), rel=1e-12)
    gross = ((2 + A) ** 2 / (4 * (1 + A))) ** (DAYS / 2)
    net = (1 + costs.net_returns(made, k)).prod()
    assert net == pytest.approx(gross * (1 - k * share) ** (DAYS - 1), rel=1e-12)
    # Rebalancing stops paying where that falls to buy and hold's 1:
    # k = (1 - gross^(-1 / (n - 1))) / (a / (2 + a)).
    breakeven = (1 - gross ** (-1 / (DAYS - 1))) / share
    found = costs.breakeven(returns, HALF, ["daily", "never"], 252, highest=0.1)
    assert found["gap"] == pytest.approx(annualised_return(made.gross, 252), rel=1e-12)
    assert found["cost"] == pytest.approx(breakeven, rel=1e-12)


def test_minimum_cvar_of_a_fund_and_its_opposite_is_half_of_each() -> None:
    # w A - (1 - w) A is (2 w - 1) A, and CVaR is positively homogeneous: its CVaR is
    # |2 w - 1| times A's or -A's, least, 0, at a half each.
    rng = np.random.default_rng(0)
    fund = rng.normal(0.0003, 0.01, 1_000)
    returns = pd.DataFrame(
        {"A": fund, "B": -fund}, index=pd.bdate_range("2020-01-01", periods=1_000)
    )
    w, cvar, _ = minimum_cvar(returns, 0.95)
    assert w.to_numpy() == pytest.approx([0.5, 0.5], abs=1e-9)
    assert cvar == pytest.approx(0.0, abs=1e-12)


def test_minimum_cvar_holds_only_the_smaller_of_two_copies() -> None:
    # B is A at a third the size: w A + (1 - w) B = (1 - 2 (1 - w) / 3) A, whose CVaR is
    # least with all of B, a third of A's.
    rng = np.random.default_rng(1)
    fund = rng.normal(0.0002, 0.012, 1_000)
    returns = pd.DataFrame(
        {"A": fund, "B": fund / 3}, index=pd.bdate_range("2020-01-01", periods=1_000)
    )
    w, cvar, _ = minimum_cvar(returns, 0.95)
    assert w.to_numpy() == pytest.approx([0.0, 1.0], abs=1e-9)
    assert cvar == pytest.approx(risk.expected_shortfall(returns["A"], 0.95) / 3, rel=1e-9)
