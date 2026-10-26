"""The Monte Carlo estimates converge to the answers theory gives, at the rate it gives.

Each check is against a closed form for the model simulated, with a tolerance of a few
standard errors of the estimate, so a run that converges more slowly than theory, or to
the wrong place, fails.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from portfolio_risk.montecarlo import (
    BootstrapModel,
    NormalModel,
    StudentTModel,
    simulate,
)
from portfolio_risk.risk import expected_shortfall, value_at_risk

MU, SIGMA = 0.0003, 0.012


def one_fund(model=NormalModel, **kwargs):
    return model(("A",), np.array([MU]), np.array([[SIGMA**2]]), **kwargs)


def test_the_error_of_the_mean_falls_as_one_over_the_root_of_the_paths() -> None:
    # Over a horizon of h days wealth is lognormal: its mean and variance are known, so
    # the error of a simulated mean is known too, sd / sqrt(paths).
    h = 21
    mean = math.exp(h * (MU + SIGMA**2 / 2))
    sd = mean * math.sqrt(math.expm1(h * SIGMA**2))
    errors = {}
    for paths in (250, 4_000):
        estimates = np.array(
            [
                1 + simulate(one_fund(), {"A": 1.0}, h, paths, seed).horizon_returns.mean()
                for seed in range(100)
            ]
        )
        errors[paths] = math.sqrt(np.mean((estimates - mean) ** 2))
        # The root mean square of 100 errors is within about 7% of the true one; allow
        # 25% (0.86 to 1.14 of it over 20 sets of 100 other seeds).
        assert errors[paths] == pytest.approx(sd / math.sqrt(paths), rel=0.25)
    # Sixteen times the paths, a quarter of the error (3.2 to 4.6 over those sets).
    assert errors[250] / errors[4_000] == pytest.approx(4.0, rel=0.3)


@pytest.mark.parametrize("q", [0.01, 0.05, 0.5, 0.95, 0.99])
def test_horizon_quantiles_converge_to_the_lognormal_ones(q) -> None:
    h, paths = 63, 100_000
    simulated = simulate(one_fund(), {"A": 1.0}, h, paths, seed=1).horizon_returns
    z = stats.norm.ppf(q)
    exact = math.expm1(h * MU + z * math.sqrt(h) * SIGMA)
    # The standard error of a sample quantile: sqrt(q (1 - q) / n) over the density there.
    density = stats.lognorm.pdf(1 + exact, s=math.sqrt(h) * SIGMA, scale=math.exp(h * MU))
    se = math.sqrt(q * (1 - q) / paths) / density
    assert abs(np.quantile(simulated, q) - exact) < 4 * se


@pytest.mark.parametrize("level", [0.95, 0.99])
def test_one_day_value_at_risk_and_shortfall_converge_to_the_lognormal_ones(level) -> None:
    # A day's simple return is exp(X) - 1, X normal: the loss at the tail's edge and the
    # mean loss beyond it have closed forms, which the risk measures of a long simulated
    # series must reach.
    paths = 400_000
    daily = pd.Series(simulate(one_fund(), {"A": 1.0}, 1, paths, seed=2).returns[:, 0])
    tail = 1 - level
    z = stats.norm.ppf(tail)
    var = -math.expm1(MU + z * SIGMA)
    # E[exp(X) | X < mu + z sigma] = exp(mu + sigma^2 / 2) Phi(z - sigma) / tail.
    shortfall = 1 - math.exp(MU + SIGMA**2 / 2) * stats.norm.cdf(z - SIGMA) / tail
    density = stats.norm.pdf(z) / (SIGMA * (1 - var))
    assert abs(value_at_risk(daily, level) - var) < 4 * math.sqrt(tail * level / paths) / density
    losses = -daily[daily <= -var]
    assert (
        abs(expected_shortfall(daily, level) - shortfall)
        < 4 * losses.std() / math.sqrt(len(losses)) + 4 * math.sqrt(tail * level / paths) / density
    )


def test_t_horizon_log_returns_have_the_model_s_mean_and_variance() -> None:
    # Every day's log return has the model's mean and variance whatever the degrees of
    # freedom, so over h independent days they are h times those; a t with 10 degrees
    # of freedom has excess kurtosis 6 / (10 - 4) = 1 a day, 1 / h over h days.
    h, paths = 21, 100_000
    model = one_fund(StudentTModel, dof=10.0)
    logs = np.log1p(simulate(model, {"A": 1.0}, h, paths, seed=3).horizon_returns)
    mean, var = h * MU, h * SIGMA**2
    assert abs(logs.mean() - mean) < 4 * math.sqrt(var / paths)
    assert abs(logs.var(ddof=1) - var) < 4 * var * math.sqrt((2 + 1 / h) / paths)


def test_t_daily_quantiles_converge_to_the_scaled_t() -> None:
    # A day's log return is mu + sigma sqrt((dof - 2) / dof) T, T a standard t.
    dof, paths = 5.0, 200_000
    logs = np.log1p(simulate(one_fund(StudentTModel, dof=dof), {"A": 1.0}, 1, paths, 4).returns)
    scale = SIGMA * math.sqrt((dof - 2) / dof)
    for q in (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999):
        exact = MU + scale * stats.t.ppf(q, dof)
        se = math.sqrt(q * (1 - q) / paths) / (stats.t.pdf(stats.t.ppf(q, dof), dof) / scale)
        assert abs(np.quantile(logs, q) - exact) < 4 * se, q


@pytest.mark.parametrize("mean_block", [1.0, 5.0, 40.0])
def test_the_bootstrap_s_mean_day_converges_to_the_historical_mean(mean_block) -> None:
    # The stationary bootstrap wraps around, so every simulated day is any historical day
    # with the same chance and the expected daily return is the historical mean, however
    # long the blocks. Paths are independent, so their means give the standard error.
    rng = np.random.default_rng(5)
    history = rng.standard_t(4, size=(500, 2)) * 0.01 + [0.0004, -0.0001]
    model = BootstrapModel(("A", "B"), history, mean_block)
    w = np.array([0.7, 0.3])
    sim = simulate(model, {"A": 0.7, "B": 0.3}, 63, 20_000, seed=6)
    per_path = sim.returns.mean(axis=1)
    se = per_path.std(ddof=1) / math.sqrt(len(per_path))
    assert abs(per_path.mean() - (history @ w).mean()) < 4 * se
