"""Portfolio construction beyond mean-variance: risk parity and minimum CVaR.

Every portfolio here is long only and fully invested, like the frontier's.

``inverse_volatility``
    Weights proportional to one over each fund's volatility: equal risk from each
    fund if the funds were uncorrelated.
``risk_parity``
    Equal risk contribution (or contributions in proportion to ``budgets``). A fund's
    contribution to the portfolio's variance is ``w_i (S w)_i``; they add up to
    ``w' S w``, and here each is the same share of it. The weights come from Spinu's
    (2013) convex form: ``y`` minimising ``y' S y / 2 - sum(b_i log y_i)`` satisfies
    ``y_i (S y)_i = b_i``, so ``w = y / sum(y)`` has contributions in proportion to
    ``b``. It is minimised by Newton's method with a backtracking line search that
    keeps ``y`` positive; the function is strictly convex, so the answer is unique.
``minimum_cvar``
    The portfolio whose historical expected shortfall (CVaR) of one day's return,
    at ``level``, is lowest: the linear programme of Rockafellar and Uryasev (2000),
    ``min a + sum(u_t) / ((1 - level) T)`` over weights ``w``, a threshold ``a`` and
    shortfalls ``u_t >= max(-r_t' w - a, 0)``, solved by HiGHS through scipy. Its
    optimum is the expected shortfall ``risk.expected_shortfall`` gives (Acerbi and
    Tasche's, the boundary day counted in part). When ``(1 - level) T`` is not a whole
    number, ``a`` is the value at risk; when it is, any ``a`` from the next-worst loss
    up to the value at risk gives the same optimum.
    A linear programme's optimum can be a face rather than a point, so other weights
    may share the same CVaR.

Neither risk parity nor minimum CVaR uses expected returns, which the frontier's
out-of-sample results showed to be the estimate least worth trusting.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Newton's method stops when every contribution is within this of its budget; it
# gets there in a handful of steps, the function being smooth and strictly convex.
TOLERANCE = 1e-14
MAX_ITERATIONS = 200
# Where rounding stops it short of that, it must still be this close (relative).
CHECK = 1e-10


def _matrix(covariance: pd.DataFrame) -> np.ndarray:
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("the covariance matrix must be a DataFrame labelled by fund")
    matrix = covariance.to_numpy(dtype=float)
    n = matrix.shape[0]
    if n == 0 or matrix.shape != (n, n) or not covariance.index.equals(covariance.columns):
        raise ValueError("the covariance matrix must be square, its rows and columns the funds")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0):
        raise ValueError("the covariance matrix must be finite and symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("the covariance matrix must be positive definite") from None
    return matrix


def inverse_volatility(covariance: pd.DataFrame) -> pd.Series:
    """Weights proportional to one over each fund's volatility."""
    inverse = 1.0 / np.sqrt(np.diag(_matrix(covariance)))
    return pd.Series(inverse / inverse.sum(), index=covariance.index, name="weight")


def risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Each fund's share of the portfolio's variance, ``w_i (S w)_i / w' S w``."""
    matrix = _matrix(covariance)
    w = weights.reindex(covariance.index).to_numpy(dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("needs a weight for every fund in the covariance matrix")
    marginal = matrix @ w
    return pd.Series(w * marginal / (w @ marginal), index=covariance.index, name="contribution")


def _budgets(budgets: pd.Series | None, index: pd.Index) -> np.ndarray:
    if budgets is None:
        return np.full(len(index), 1.0 / len(index))
    b = budgets.reindex(index).to_numpy(dtype=float)
    if not np.isfinite(b).all() or (b <= 0).any():
        raise ValueError("every fund needs a risk budget above 0")
    return b / b.sum()


def risk_parity(covariance: pd.DataFrame, budgets: pd.Series | None = None) -> pd.Series:
    """The long-only portfolio whose risk contributions are in proportion to ``budgets``
    (equal by default)."""
    matrix = _matrix(covariance)
    b = _budgets(budgets, covariance.index)
    # Starting from inverse volatility, scaled so that y' S y = 1 as at the solution.
    y = b / np.sqrt(np.diag(matrix))
    y /= math.sqrt(y @ matrix @ y)

    def objective(v: np.ndarray) -> float:
        return 0.5 * v @ matrix @ v - b @ np.log(v)

    for _ in range(MAX_ITERATIONS):
        marginal = matrix @ y
        if np.max(np.abs(y * marginal - b)) <= TOLERANCE:
            break
        gradient = marginal - b / y
        hessian = matrix + np.diag(b / y**2)
        step = np.linalg.solve(hessian, gradient)
        # The largest step that keeps every y positive, then halved until f falls enough.
        shrinking = step > 0
        t = (
            min(1.0, 0.99 * float(np.min(y[shrinking] / step[shrinking])))
            if shrinking.any()
            else 1.0
        )
        current = objective(y)
        decrease = gradient @ step
        if decrease <= 64 * np.finfo(float).eps * max(1.0, abs(current)):
            # Closer than f can show: the line search would see rounding, not the
            # decrease, and refuse a good step. This near the solution a Newton step
            # of a smooth, strictly convex function converges, so it is taken.
            y = y - t * step
            continue
        while objective(y - t * step) > current - 1e-4 * t * decrease and t > 1e-12:
            t *= 0.5
        if t <= 1e-12:
            # No step lowers f any more: rounding, near the solution.
            break
        y = y - t * step
    if np.max(np.abs(y * (matrix @ y) - b)) > CHECK * b.max():
        raise ArithmeticError("risk parity did not converge")
    w = y / y.sum()
    return pd.Series(w, index=covariance.index, name="weight")


def minimum_cvar(
    returns: pd.DataFrame, level: float = 0.95, max_weight: float = 1.0
) -> tuple[pd.Series, float, float]:
    """The weights with the lowest historical CVaR of one period's return at ``level``.

    Returns the weights, the CVaR and the value at risk at the optimum, as positive
    losses per period.
    """
    from scipy import sparse
    from scipy.optimize import linprog

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("expected a DataFrame of returns, a column per fund")
    values = returns.to_numpy(dtype=float)
    days, n = values.shape
    if n == 0 or days < 2:
        raise ValueError("needs at least one fund and two days of returns")
    if not np.isfinite(values).all():
        raise ValueError("returns include NaN or infinite values")
    if isinstance(level, bool) or not isinstance(level, int | float) or not 0.0 < level < 1.0:
        raise ValueError(f"level must be a number between 0 and 1, not {level!r}")
    if isinstance(max_weight, bool) or not isinstance(max_weight, int | float):
        raise ValueError(f"max_weight must be a number, not {max_weight!r}")
    if not 1.0 / n - 1e-15 <= max_weight <= 1.0:
        raise ValueError(f"max_weight must be from 1/{n} (all funds at the cap) to 1")
    tail = (1.0 - level) * days
    # Variables: n weights, the threshold a, then one shortfall per day.
    cost = np.concatenate([np.zeros(n), [1.0], np.full(days, 1.0 / tail)])
    # -r_t' w - a - u_t <= 0 for every day t; sparse, or the identity block alone would
    # take 240 MB for 21 years of days.
    upper = sparse.hstack(
        [sparse.csr_array(-values), sparse.csr_array(-np.ones((days, 1))), -sparse.eye_array(days)],
        format="csr",
    )
    budget = np.concatenate([np.ones(n), [0.0], np.zeros(days)])[None, :]
    bounds = [(0.0, max_weight)] * n + [(None, None)] + [(0.0, None)] * days
    solution = linprog(
        cost,
        A_ub=upper,
        b_ub=np.zeros(days),
        A_eq=budget,
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
        options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
    )
    if solution.status != 0:
        raise ArithmeticError(f"the CVaR programme was not solved: {solution.message}")
    w = np.clip(solution.x[:n], 0.0, max_weight)
    w /= w.sum()
    return (
        pd.Series(w, index=returns.columns, name="weight"),
        float(solution.fun),
        float(solution.x[n]),
    )
