"""Covariance estimates of asset returns, and the correlations and volatilities in them.

Three estimators, each taking a date-by-asset table of returns and giving an
asset-by-asset covariance matrix per period of the returns (multiply by the periods in a
year, ``annualise``, to get yearly figures):

``sample_covariance``
    The usual unbiased estimate, dividing by ``n - 1``. Every day counts the same.
``ewma_covariance``
    An exponentially weighted estimate in the RiskMetrics style: the weight of each
    day falls by the factor ``decay`` per day back, so recent days dominate (with
    0.94, RiskMetrics' daily value, a weight halves every 11.2 days). The weights are
    normalised to sum to one over the sample. RiskMetrics takes daily means to be
    zero; ``demean=True`` subtracts the weighted mean instead.
``ledoit_wolf_covariance``
    The sample covariance shrunk toward a multiple of the identity matrix by the
    amount Ledoit and Wolf (2004, "A well-conditioned estimator for large-dimensional
    covariance matrices") show minimises expected squared error. Shrinking pulls the
    extreme eigenvalues in, so the matrix is better conditioned for an optimiser to
    invert. Like the paper, it starts from the covariance that divides by ``n``.
    Returns the matrix and the shrinkage intensity, between 0 and 1.

``correlation`` and ``volatilities`` read correlations and standard deviations out of
any covariance matrix.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _values(returns: pd.DataFrame, minimum: int = 2) -> np.ndarray:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"expected a date-by-asset DataFrame, not {type(returns).__name__}")
    if returns.shape[1] == 0:
        raise ValueError("no assets")
    if len(returns) < minimum:
        raise ValueError(f"needs at least {minimum} returns per asset, got {len(returns)}")
    values = returns.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("returns include NaN or infinite values")
    return values


def _centred(values: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Deviations from the mean, exactly zero for an asset whose returns never vary.

    The mean of equal values can differ from them in the last bit, which would give such
    an asset a variance of rounding error (about 1e-36) instead of zero, and correlations
    computed from noise instead of a refusal.
    """
    centred = values - mean
    centred[:, np.ptp(values, axis=0) == 0.0] = 0.0
    return centred


def _frame(matrix: np.ndarray, returns: pd.DataFrame) -> pd.DataFrame:
    # Exactly symmetric: products summed in different orders can differ in the last bit.
    matrix = (matrix + matrix.T) / 2.0
    return pd.DataFrame(matrix, index=returns.columns, columns=returns.columns)


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """The unbiased sample covariance (dividing by ``n - 1``)."""
    values = _values(returns)
    centred = _centred(values, values.mean(axis=0))
    return _frame(centred.T @ centred / (len(values) - 1), returns)


def ewma_weights(n: int, decay: float) -> np.ndarray:
    """Weights for ``n`` days, oldest first, falling by ``decay`` per day back, summing to 1."""
    if isinstance(decay, bool) or not isinstance(decay, int | float) or not 0 < decay < 1:
        raise ValueError(f"decay must be between 0 and 1 (exclusive), not {decay!r}")
    if n < 1:
        raise ValueError("needs at least one day")
    # Built from the newest day back so a long sample underflows to 0 rather than overflowing.
    weights = decay ** np.arange(n - 1, -1, -1, dtype="float64")
    return weights / weights.sum()


def ewma_half_life(decay: float) -> float:
    """Days for a weight to halve: ``log(0.5) / log(decay)``."""
    ewma_weights(1, decay)
    return math.log(0.5) / math.log(decay)


def ewma_covariance(
    returns: pd.DataFrame, decay: float = 0.94, *, demean: bool = False
) -> pd.DataFrame:
    """The exponentially weighted covariance, newest day weighted most."""
    values = _values(returns, minimum=1)
    weights = ewma_weights(len(values), decay)
    if demean:
        values = _centred(values, weights @ values)
    return _frame((values * weights[:, None]).T @ values, returns)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """The covariance shrunk toward a scaled identity, and the shrinkage intensity used."""
    values = _values(returns)
    n, p = values.shape
    x = _centred(values, values.mean(axis=0))
    sample = x.T @ x / n
    # Target: the identity scaled by the average variance.
    mu = np.trace(sample) / p
    target = mu * np.eye(p)
    # d2: how far the sample covariance is from the target (squared Frobenius norm / p).
    d2 = np.sum((sample - target) ** 2) / p
    if d2 == 0.0:
        return _frame(sample, returns), 0.0
    # b2: the estimated error of the sample covariance itself, capped at d2. The sum over
    # days of the squared norm of x_t x_t' - S is sum(|x_t|^4) - n |S|^2, which avoids
    # building the n p-by-p outer products.
    b2_bar = (np.sum(np.sum(x**2, axis=1) ** 2) - n * np.sum(sample**2)) / (n**2 * p)
    # Rounding can leave b2_bar a hair below zero when every day's x_t x_t' equals S.
    b2 = min(max(b2_bar, 0.0), d2)
    shrinkage = b2 / d2
    return _frame(shrinkage * target + (1.0 - shrinkage) * sample, returns), float(shrinkage)


def _check_covariance(covariance: pd.DataFrame) -> np.ndarray:
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError(f"expected an asset-by-asset DataFrame, not {type(covariance).__name__}")
    matrix = covariance.to_numpy(dtype="float64")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"a covariance matrix is square, not {matrix.shape}")
    if list(covariance.index) != list(covariance.columns):
        raise ValueError("rows and columns must name the same assets in the same order")
    if not np.isfinite(matrix).all():
        raise ValueError("the matrix includes NaN or infinite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0.0):
        raise ValueError("the matrix is not symmetric")
    diagonal = np.diag(matrix)
    if (diagonal <= 0).any():
        asset = covariance.index[int(np.argmax(diagonal <= 0))]
        raise ValueError(f"{asset} has a variance of {diagonal[diagonal <= 0][0]}, not positive")
    return matrix


def volatilities(covariance: pd.DataFrame) -> pd.Series:
    """Each asset's standard deviation: the square root of the diagonal."""
    matrix = _check_covariance(covariance)
    return pd.Series(np.sqrt(np.diag(matrix)), index=covariance.index)


def correlation(covariance: pd.DataFrame) -> pd.DataFrame:
    """The correlation matrix in a covariance matrix, with an exact unit diagonal."""
    matrix = _check_covariance(covariance)
    sd = np.sqrt(np.diag(matrix))
    corr = matrix / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=covariance.index, columns=covariance.columns)


def annualise(covariance: pd.DataFrame, periods_per_year: float) -> pd.DataFrame:
    """A per-period covariance on a yearly scale, assuming returns uncorrelated over time."""
    _check_covariance(covariance)
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int | float)
        or not math.isfinite(periods_per_year)
        or periods_per_year <= 0
    ):
        raise ValueError(f"periods_per_year must be a positive number, not {periods_per_year!r}")
    return covariance * periods_per_year
