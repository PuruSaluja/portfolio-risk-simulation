import math

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.covariance import (
    annualise,
    correlation,
    ewma_covariance,
    ewma_half_life,
    ewma_weights,
    ledoit_wolf_covariance,
    sample_covariance,
    volatilities,
)

# Ten days of three assets' returns.
SMALL = pd.DataFrame(
    [
        [0.012, -0.004, 0.020],
        [-0.008, 0.003, -0.015],
        [0.005, 0.001, 0.011],
        [0.000, -0.002, 0.004],
        [-0.021, 0.006, -0.030],
        [0.017, -0.001, 0.009],
        [0.003, 0.002, -0.006],
        [-0.006, -0.003, 0.001],
        [0.009, 0.004, 0.018],
        [-0.002, 0.000, -0.010],
    ],
    columns=["A", "B", "C"],
)


def random_returns(n=500, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.array([0.01, 0.02, 0.005, 0.015])
    mixing = np.array([[1, 0, 0, 0], [0.6, 0.8, 0, 0], [-0.3, 0.1, 0.95, 0], [0.5, 0.5, 0.2, 0.7]])
    values = rng.standard_normal((n, 4)) @ mixing.T * scale
    return pd.DataFrame(values, columns=["W", "X", "Y", "Z"])


def test_sample_covariance_is_numpy_s_unbiased_estimate() -> None:
    returns = random_returns()
    cov = sample_covariance(returns)
    np.testing.assert_allclose(cov.to_numpy(), np.cov(returns.to_numpy().T), rtol=1e-12)
    assert list(cov.index) == list(cov.columns) == ["W", "X", "Y", "Z"]


def test_every_estimate_is_exactly_symmetric() -> None:
    returns = random_returns(seed=5)
    for cov in (
        sample_covariance(returns),
        ewma_covariance(returns),
        ewma_covariance(returns, demean=True),
        ledoit_wolf_covariance(returns)[0],
    ):
        matrix = cov.to_numpy()
        assert (matrix == matrix.T).all()


def test_ewma_weights_fall_by_the_decay_and_sum_to_one() -> None:
    weights = ewma_weights(5, 0.9)
    assert weights.sum() == pytest.approx(1.0, rel=1e-15)
    np.testing.assert_allclose(weights[:-1] / weights[1:], 0.9, rtol=1e-12)
    assert weights[-1] == weights.max()  # the newest day weighs most


def test_ewma_weights_need_a_day() -> None:
    with pytest.raises(ValueError, match="at least one day"):
        ewma_weights(0, 0.94)


def test_ewma_weights_over_a_long_sample_do_not_overflow() -> None:
    weights = ewma_weights(100_000, 0.94)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0)
    assert weights[-1] == pytest.approx(0.06)


@pytest.mark.parametrize("decay", [0, 1, -0.5, 1.5, math.nan, True, "0.94"])
def test_a_decay_outside_zero_to_one_is_refused(decay) -> None:
    with pytest.raises(ValueError, match="decay must be between 0 and 1"):
        ewma_weights(10, decay)
    with pytest.raises(ValueError, match="decay must be between 0 and 1"):
        ewma_covariance(SMALL, decay)


def test_riskmetrics_decay_halves_a_weight_in_about_eleven_days() -> None:
    assert ewma_half_life(0.94) == pytest.approx(11.2, abs=0.01)
    assert ewma_half_life(0.5) == pytest.approx(1.0)


def test_ewma_covariance_matches_a_direct_weighted_sum() -> None:
    returns = SMALL.to_numpy()
    decay = 0.8
    n = len(returns)
    weights = np.array([decay ** (n - 1 - t) for t in range(n)])
    weights /= weights.sum()
    expected = sum(w * np.outer(r, r) for w, r in zip(weights, returns, strict=True))
    np.testing.assert_allclose(ewma_covariance(SMALL, decay).to_numpy(), expected, rtol=1e-12)
    mean = weights @ returns
    centred = sum(w * np.outer(r - mean, r - mean) for w, r in zip(weights, returns, strict=True))
    np.testing.assert_allclose(
        ewma_covariance(SMALL, decay, demean=True).to_numpy(), centred, rtol=1e-12
    )


def test_ewma_follows_a_change_in_volatility_that_the_sample_estimate_averages() -> None:
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.01, (400, 1))
    stormy = rng.normal(0, 0.03, (100, 1))
    returns = pd.DataFrame(np.vstack([calm, stormy]), columns=["A"])
    ewma_sd = math.sqrt(ewma_covariance(returns).iloc[0, 0])
    sample_sd = math.sqrt(sample_covariance(returns).iloc[0, 0])
    assert ewma_sd == pytest.approx(0.03, rel=0.2)
    assert sample_sd < 0.02


def test_ewma_gives_a_value_from_a_single_day() -> None:
    one = SMALL.iloc[[0]]
    np.testing.assert_allclose(
        ewma_covariance(one).to_numpy(), np.outer(one.iloc[0], one.iloc[0]), rtol=1e-15
    )


def test_ledoit_wolf_matches_scikit_learn() -> None:
    # sklearn.covariance.LedoitWolf().fit(SMALL) with scikit-learn 1.9.0.
    expected = np.array(
        [
            [1.0968375553355629e-04, -1.1130104360190605e-05, 9.6990909424518078e-05],
            [-1.1130104360190605e-05, 3.8598629552287490e-05, -1.8278009219566551e-05],
            [9.6990909424518078e-05, -1.8278009219566551e-05, 1.8980761491415620e-04],
        ]
    )
    cov, shrinkage = ledoit_wolf_covariance(SMALL)
    assert shrinkage == pytest.approx(0.28377706819880294, rel=1e-12)
    np.testing.assert_allclose(cov.to_numpy(), expected, rtol=1e-12)


def test_ledoit_wolf_keeps_the_trace_and_pulls_in_the_extreme_eigenvalues() -> None:
    returns = random_returns(n=60, seed=9)
    cov, shrinkage = ledoit_wolf_covariance(returns)
    biased = np.cov(returns.to_numpy().T, ddof=0)
    assert 0 < shrinkage < 1
    assert np.trace(cov.to_numpy()) == pytest.approx(np.trace(biased), rel=1e-12)
    shrunk, raw = np.linalg.eigvalsh(cov.to_numpy()), np.linalg.eigvalsh(biased)
    assert shrunk.min() > raw.min()
    assert shrunk.max() < raw.max()
    assert np.linalg.cond(cov.to_numpy()) < np.linalg.cond(biased)


def test_ledoit_wolf_shrinks_less_as_the_sample_grows() -> None:
    small = ledoit_wolf_covariance(random_returns(n=40, seed=4))[1]
    large = ledoit_wolf_covariance(random_returns(n=4000, seed=4))[1]
    assert large < small


def test_ledoit_wolf_leaves_returns_already_at_the_target_alone() -> None:
    # Two uncorrelated assets with equal variance: the sample covariance is the target.
    returns = pd.DataFrame({"A": [0.01, -0.01, 0.01, -0.01], "B": [0.01, 0.01, -0.01, -0.01]})
    cov, shrinkage = ledoit_wolf_covariance(returns)
    assert 0.0 <= shrinkage <= 1.0
    np.testing.assert_allclose(cov.to_numpy(), np.eye(2) * 1e-4, rtol=1e-12, atol=1e-20)


def test_ledoit_wolf_does_not_shrink_when_every_day_agrees() -> None:
    # Both days give the same outer product, so the sample covariance has no estimated error.
    returns = pd.DataFrame({"A": [0.01, -0.01], "B": [-0.02, 0.02]})
    cov, shrinkage = ledoit_wolf_covariance(returns)
    assert shrinkage == 0.0
    np.testing.assert_allclose(cov.to_numpy(), [[1e-4, -2e-4], [-2e-4, 4e-4]], rtol=1e-12)


def test_ledoit_wolf_never_shrinks_by_a_negative_amount() -> None:
    # Every day's outer product is the same, so the estimated error is zero; computed, it
    # comes out about -5e-26 here, which would give a negative shrinkage unclamped.
    v = [-0.0013210486329130189, 0.006404226504432821, 0.001049001171530397, -0.005356693731611109]
    returns = pd.DataFrame([v, [-a for a in v], v, [-a for a in v]], columns=list("ABCD"))
    cov, shrinkage = ledoit_wolf_covariance(returns)
    assert shrinkage == 0.0
    np.testing.assert_allclose(cov.to_numpy(), np.outer(v, v), rtol=1e-12)


def test_ledoit_wolf_of_one_asset_is_its_variance() -> None:
    # One asset is its own target, at no distance from it.
    returns = SMALL[["A"]]
    cov, shrinkage = ledoit_wolf_covariance(returns)
    assert shrinkage == 0.0
    assert cov.iloc[0, 0] == pytest.approx(np.var(returns["A"].to_numpy()), rel=1e-12)


def test_ledoit_wolf_shrinks_fully_when_the_error_exceeds_the_distance() -> None:
    # Variances nearly equal and estimated noisily; scikit-learn 1.9.0 also gives 1.0.
    returns = pd.DataFrame({"A": [0.01, -0.01, 0.0, 0.0], "B": [0.0, 0.0, 0.0101, -0.0101]})
    cov, shrinkage = ledoit_wolf_covariance(returns)
    assert shrinkage == 1.0
    np.testing.assert_allclose(cov.to_numpy(), np.eye(2) * 5.05025e-05, rtol=1e-12)


@pytest.mark.parametrize(
    ("function", "minimum"),
    [(sample_covariance, 2), (ledoit_wolf_covariance, 2), (ewma_covariance, 1)],
)
def test_too_few_returns_are_refused(function, minimum) -> None:
    with pytest.raises(ValueError, match=f"at least {minimum} returns per asset"):
        function(SMALL.iloc[: minimum - 1])


@pytest.mark.parametrize("function", [sample_covariance, ledoit_wolf_covariance, ewma_covariance])
def test_unusable_returns_are_refused(function) -> None:
    bad = SMALL.copy()
    bad.iloc[2, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        function(bad)
    with pytest.raises(ValueError, match="no assets"):
        function(SMALL[[]])
    with pytest.raises(TypeError, match="date-by-asset DataFrame, not ndarray"):
        function(SMALL.to_numpy())


def test_correlation_and_volatilities_from_a_covariance() -> None:
    cov = pd.DataFrame([[4.0, 1.2], [1.2, 9.0]], index=["A", "B"], columns=["A", "B"])
    corr = correlation(cov)
    assert corr.loc["A", "B"] == pytest.approx(0.2) and corr.loc["B", "A"] == pytest.approx(0.2)
    assert (np.diag(corr.to_numpy()) == 1.0).all()
    assert volatilities(cov).to_dict() == {"A": 2.0, "B": 3.0}


def test_correlation_s_diagonal_is_exactly_one_even_when_rounding_says_otherwise() -> None:
    # sqrt(2) ** 2 is 2.0000000000000004, so 2 / (sqrt(2) * sqrt(2)) is not exactly 1.
    cov = pd.DataFrame([[2.0, 0.5], [0.5, 3.0]], index=["A", "B"], columns=["A", "B"])
    assert (np.diag(correlation(cov).to_numpy()) == 1.0).all()


def test_correlation_of_the_sample_covariance_is_numpy_s() -> None:
    returns = random_returns(seed=6)
    np.testing.assert_allclose(
        correlation(sample_covariance(returns)).to_numpy(),
        np.corrcoef(returns.to_numpy().T),
        rtol=1e-12,
    )


def test_annualising_scales_every_entry_and_leaves_correlations_alone() -> None:
    cov = sample_covariance(random_returns(seed=8))
    yearly = annualise(cov, 252)
    np.testing.assert_allclose(yearly.to_numpy(), cov.to_numpy() * 252, rtol=1e-15)
    np.testing.assert_allclose(correlation(yearly).to_numpy(), correlation(cov).to_numpy())
    np.testing.assert_allclose(volatilities(yearly), volatilities(cov) * math.sqrt(252))
    with pytest.raises(ValueError, match="periods_per_year must be a positive number"):
        annualise(cov, 0)


@pytest.mark.parametrize(
    ("matrix", "labels", "message"),
    [
        ([[1.0, 0.1]], (["A"], ["A", "B"]), "is square"),
        ([[1.0, 0.1], [0.1, 1.0]], (["A", "B"], ["B", "A"]), "same assets in the same order"),
        ([[1.0, 0.1], [0.2, 1.0]], (["A", "B"], ["A", "B"]), "not symmetric"),
        ([[1.0, 0.0], [0.0, 0.0]], (["A", "B"], ["A", "B"]), "B has a variance of 0.0"),
        ([[1.0, 0.0], [0.0, -1.0]], (["A", "B"], ["A", "B"]), "B has a variance of -1.0"),
        ([[1.0, np.nan], [np.nan, 1.0]], (["A", "B"], ["A", "B"]), "NaN or infinite"),
    ],
)
def test_unusable_covariance_matrices_are_refused(matrix, labels, message) -> None:
    cov = pd.DataFrame(matrix, index=labels[0], columns=labels[1])
    for function in (correlation, volatilities, lambda c: annualise(c, 252)):
        with pytest.raises(ValueError, match=message):
            function(cov)
