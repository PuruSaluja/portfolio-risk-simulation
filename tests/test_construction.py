import itertools
import math

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import construction
from portfolio_risk.construction import (
    inverse_volatility,
    minimum_cvar,
    risk_contributions,
    risk_parity,
)
from portfolio_risk.risk import expected_shortfall, value_at_risk


def random_covariance(n, seed, correlation=None):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.05, 0.35, n)
    if correlation is None:
        a = rng.normal(size=(n, n + 3))
        c = a @ a.T
        d = np.sqrt(np.diag(c))
        corr = c / np.outer(d, d)
    else:
        corr = np.full((n, n), correlation)
        np.fill_diagonal(corr, 1.0)
    names = [f"F{i}" for i in range(n)]
    return pd.DataFrame(corr * np.outer(vols, vols), index=names, columns=names)


@pytest.mark.parametrize(("n", "seed"), [(2, 0), (3, 1), (5, 2), (8, 3), (20, 4), (50, 5)])
def test_risk_parity_gives_every_fund_the_same_share_of_risk(n, seed) -> None:
    covariance = random_covariance(n, seed)
    w = risk_parity(covariance)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)
    assert (w > 0).all()
    np.testing.assert_allclose(risk_contributions(w, covariance), 1.0 / n, rtol=1e-12)


def test_budgets_set_each_fund_s_share_of_risk() -> None:
    covariance = random_covariance(4, 7)
    budgets = pd.Series([4.0, 3.0, 2.0, 1.0], index=covariance.index)
    contributions = risk_contributions(risk_parity(covariance, budgets), covariance)
    np.testing.assert_allclose(contributions, [0.4, 0.3, 0.2, 0.1], rtol=1e-12)


@pytest.mark.parametrize("correlation", [-0.15, 0.0, 0.5, 0.99])
def test_with_equal_correlations_risk_parity_is_inverse_volatility(correlation) -> None:
    covariance = random_covariance(6, 11, correlation=correlation)
    np.testing.assert_allclose(risk_parity(covariance), inverse_volatility(covariance), rtol=1e-12)


def test_two_funds_hold_in_inverse_proportion_to_their_volatility_whatever_the_correlation() -> (
    None
):
    for rho in (-0.9, 0.0, 0.95):
        covariance = pd.DataFrame(
            [[0.04, rho * 0.2 * 0.05], [rho * 0.2 * 0.05, 0.0025]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        w = risk_parity(covariance)
        assert w["A"] == pytest.approx(0.05 / 0.25, rel=1e-13)


def test_inverse_volatility_weights_by_one_over_volatility() -> None:
    covariance = random_covariance(5, 3)
    w = inverse_volatility(covariance)
    vols = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(w * vols, (w * vols).iloc[0], rtol=1e-13)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)


def test_risk_parity_converges_on_badly_conditioned_matrices() -> None:
    for seed in range(20):
        # Correlations of 0.999 and a little more, so not all equal.
        common = random_covariance(10, seed, correlation=0.999)
        varied = random_covariance(10, seed + 100)
        vols = np.sqrt(np.diag(common))
        covariance = common + 0.001 * varied / np.sqrt(
            np.outer(np.diag(varied), np.diag(varied))
        ) * np.outer(vols, vols)
        w = risk_parity(covariance)
        np.testing.assert_allclose(risk_contributions(w, covariance), 0.1, rtol=1e-9)


def test_contributions_add_up_to_one() -> None:
    covariance = random_covariance(5, 9)
    w = pd.Series([0.1, 0.2, 0.3, 0.15, 0.25], index=covariance.index)
    contributions = risk_contributions(w, covariance)
    assert contributions.sum() == pytest.approx(1.0, rel=1e-14)
    s = covariance.to_numpy()
    assert contributions.iloc[2] == pytest.approx(0.3 * (s @ w.to_numpy())[2] / (w @ s @ w))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.to_numpy(), "DataFrame"),
        (lambda c: c.iloc[:, :2], "square"),
        (lambda c: c.set_axis(["X", "F1", "F2"], axis=0), "square"),
        (lambda c: c.mask(c > 0.0, np.nan), "finite"),
        (lambda c: c + np.triu(np.full((3, 3), 0.01), 1), "symmetric"),
        (lambda c: c - np.diag(np.full(3, 1.0)), "positive definite"),
    ],
)
def test_bad_covariance_matrices_are_refused(change, message) -> None:
    bad = change(random_covariance(3, 0))
    for construct in (risk_parity, inverse_volatility):
        with pytest.raises((TypeError, ValueError), match=message):
            construct(bad)


def test_bad_budgets_and_weights_are_refused() -> None:
    covariance = random_covariance(3, 0)
    for budgets in ([1.0, 0.0, 1.0], [1.0, -1.0, 2.0], [1.0, np.nan, 1.0]):
        with pytest.raises(ValueError, match="risk budget"):
            risk_parity(covariance, pd.Series(budgets, index=covariance.index))
    with pytest.raises(ValueError, match="risk budget"):
        risk_parity(covariance, pd.Series([1.0, 1.0], index=["F0", "F1"]))
    with pytest.raises(ValueError, match="a weight for every fund"):
        risk_contributions(pd.Series([0.5, 0.5], index=["F0", "F1"]), covariance)


def returns_frame(days=500, funds=3, seed=0):
    rng = np.random.default_rng(seed)
    scales = np.linspace(0.005, 0.02, funds)
    data = rng.standard_t(4, (days, funds)) * scales + 0.0003
    data[:, 1:] += 0.5 * data[:, :1]
    index = pd.bdate_range("2020-01-01", periods=days)
    return pd.DataFrame(data, index=index, columns=[f"F{i}" for i in range(funds)])


def portfolio(returns, w):
    return pd.Series(returns.to_numpy() @ w.to_numpy(), index=returns.index)


@pytest.mark.parametrize(("days", "level"), [(500, 0.95), (503, 0.95), (500, 0.99), (777, 0.9)])
def test_the_optimum_is_the_expected_shortfall_of_its_weights(days, level) -> None:
    returns = returns_frame(days)
    w, cvar, threshold = minimum_cvar(returns, level)
    assert w.sum() == pytest.approx(1.0, abs=1e-15) and (w >= 0).all()
    series = portfolio(returns, w)
    assert cvar == pytest.approx(expected_shortfall(series, level), rel=1e-9)
    tail = (1 - level) * days
    if abs(tail - round(tail)) > 1e-9:
        assert threshold == pytest.approx(value_at_risk(series, level), rel=1e-8)
    else:
        # Any threshold from the next-worst loss up to the value at risk is optimal.
        losses = np.sort(-series.to_numpy())[::-1]
        k = round(tail)
        assert losses[k] - 1e-12 <= threshold <= losses[k - 1] + 1e-12


def test_no_portfolio_on_a_grid_has_a_lower_cvar() -> None:
    returns = returns_frame(400, funds=3, seed=4)
    w, cvar, _ = minimum_cvar(returns, 0.95)
    best = math.inf
    steps = 100
    for i, j in itertools.product(range(steps + 1), repeat=2):
        if i + j <= steps:
            grid = pd.Series([i / steps, j / steps, 1 - (i + j) / steps], index=returns.columns)
            best = min(best, expected_shortfall(portfolio(returns, grid), 0.95))
    assert cvar <= best + 1e-12
    # And the grid's best is close to it: the optimum lies between grid points.
    assert best - cvar < 1e-4


def test_a_cap_holds_and_at_one_over_n_gives_equal_weights() -> None:
    returns = returns_frame(400, funds=4, seed=2)
    uncapped, low, _ = minimum_cvar(returns, 0.95)
    assert uncapped.max() > 0.4
    capped, higher, _ = minimum_cvar(returns, 0.95, max_weight=0.4)
    assert capped.max() <= 0.4 + 1e-15 and higher >= low - 1e-12
    equal, _, _ = minimum_cvar(returns, 0.95, max_weight=0.25)
    np.testing.assert_allclose(equal, 0.25, atol=1e-12)


def test_one_fund_holds_everything() -> None:
    returns = returns_frame(100, funds=1)
    w, cvar, _ = minimum_cvar(returns, 0.95)
    assert w.tolist() == [1.0]
    assert cvar == pytest.approx(expected_shortfall(returns["F0"], 0.95), rel=1e-9)


def test_minimum_cvar_refuses_what_it_cannot_solve() -> None:
    returns = returns_frame(50)
    with pytest.raises(TypeError):
        minimum_cvar(returns.to_numpy())
    for level in (0.0, 1.0, 1.5, True):
        with pytest.raises(ValueError, match="level"):
            minimum_cvar(returns, level)
    for cap in (0.3, 1.01, True):
        with pytest.raises(ValueError, match="max_weight"):
            minimum_cvar(returns, 0.95, cap)
    with pytest.raises(ValueError, match="two days"):
        minimum_cvar(returns.iloc[:1])
    with pytest.raises(ValueError, match="two days"):
        minimum_cvar(returns[[]])
    bad = returns.copy()
    bad.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        minimum_cvar(bad)


def test_risk_parity_says_so_when_it_has_not_converged(monkeypatch) -> None:
    monkeypatch.setattr(construction, "MAX_ITERATIONS", 1)
    with pytest.raises(ArithmeticError, match="did not converge"):
        risk_parity(random_covariance(8, 3))


def test_returns_that_never_lose_have_a_negative_cvar() -> None:
    # Every day a gain: the value at risk, the threshold, is below zero.
    returns = returns_frame(300).abs() + 0.001
    w, cvar, threshold = minimum_cvar(returns, 0.95)
    series = portfolio(returns, w)
    assert cvar < 0 and threshold < 0
    assert cvar == pytest.approx(expected_shortfall(series, 0.95), rel=1e-9)


def test_risk_parity_converges_where_the_last_step_is_below_rounding() -> None:
    # Near the solution the Newton decrease is far below what the objective can show;
    # a line search there sees only rounding. These include matrices on which it
    # refused the last step and stopped short (n = 3, seeds 28, 30, 38, 47, 81).
    for n in (3, 5, 8):
        for seed in range(100):
            for scale in (1.0, 1e-4, 1e4):
                covariance = random_covariance(n, seed) * scale
                w = risk_parity(covariance)
                np.testing.assert_allclose(risk_contributions(w, covariance), 1 / n, rtol=1e-11)
