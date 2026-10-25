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

``python -m portfolio_risk.construction`` compares six portfolios from the cache's
daily total returns: equal weights, inverse volatility, risk parity, the minimum-
variance and tangency portfolios of the frontier, and minimum CVaR. Each is chosen
from the whole history, and from the returns up to each of a few split dates and then
held after it, rebalanced every close: what it was in the data it was chosen from, and
what it did after. With ``--risk-free bills`` the Sharpe ratios are against the daily
Treasury bill rate, over the days the factor cache covers, and the tangency portfolio
is chosen at the bill's mean rate over the returns it is chosen from (see
``frontier.constant_rate``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_risk import french, frontier, risk
from portfolio_risk import returns as ret
from portfolio_risk.data import DEFAULT_CACHE, DataError, load_cache, read_manifest, wide
from portfolio_risk.montecarlo import historical_portfolio_returns
from portfolio_risk.risk import BILLS, Rate

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

    Returns the weights, the CVaR and the threshold ``a`` at the optimum, as positive
    losses per period; the threshold is the value at risk unless ``(1 - level)`` times
    the number of periods is a whole number (see above).
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


PORTFOLIOS = (
    "equal",
    "inverse_volatility",
    "risk_parity",
    "minimum_variance",
    "max_sharpe",
    "minimum_cvar",
)


def choose(
    daily: pd.DataFrame, risk_free: Rate = 0.0, max_weight: float = 1.0, level: float = 0.95
) -> dict[str, pd.Series]:
    """The six portfolios from ``daily`` returns, in the order of ``PORTFOLIOS``.

    The covariance is the sample one, as the frontier's default; the cap applies to the
    three optimised portfolios (minimum variance, tangency, minimum CVaR), the others
    being what their rules give. For daily rates the tangency portfolio takes their
    constant rate over ``daily``'s days (``frontier.constant_rate``).
    """
    expected, covariance = frontier.estimates(daily)
    n = daily.shape[1]
    rate = frontier.constant_rate(risk_free, daily.index)
    return {
        "equal": pd.Series(1.0 / n, index=daily.columns, name="weight"),
        "inverse_volatility": inverse_volatility(covariance),
        "risk_parity": risk_parity(covariance),
        "minimum_variance": frontier.minimum_variance(covariance, max_weight),
        "max_sharpe": frontier.max_sharpe(expected, covariance, rate, max_weight),
        "minimum_cvar": minimum_cvar(daily, level, max_weight)[0],
    }


def describe(
    daily: pd.DataFrame,
    w: pd.Series,
    first_close: pd.Timestamp,
    risk_free: Rate = 0.0,
    level: float = 0.95,
) -> dict[str, Any]:
    """A portfolio rebalanced every close over ``daily``: its returns, risk and tail.

    Contributions are each fund's share of the variance of the portfolio's daily
    returns there; CVaR and value at risk are of one day's return, historical. Daily
    rates are subtracted day by day in the Sharpe ratio.
    """
    history = historical_portfolio_returns(daily, w.to_dict())
    per_year = ret.observed_periods_per_year(pd.DatetimeIndex([first_close, *daily.index]))
    days = ret.TRADING_DAYS_PER_YEAR
    _, covariance = frontier.estimates(daily)
    return {
        "mean": float(ret.annualised_mean(history, days)),
        "volatility": float(ret.annualised_volatility(history, days)),
        "sharpe_ratio": risk.or_none(
            risk.sharpe_ratio, history, days, frontier.rates_on(risk_free, daily.index)
        ),
        "compound_return": float(ret.annualised_return(history, per_year)),
        "max_drawdown": risk.max_drawdown(history).depth,
        "value_at_risk": risk.value_at_risk(history, level),
        "cvar": risk.expected_shortfall(history, level),
        "risk_contributions": {
            str(t): float(v) for t, v in risk_contributions(w, covariance).items()
        },
    }


def _sides(daily: pd.DataFrame, split: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    before = daily.loc[daily.index <= pd.Timestamp(split)]
    after = daily.loc[daily.index > pd.Timestamp(split)]
    funds = daily.shape[1]
    # A covariance matrix of n funds needs more than n days to be positive definite.
    if min(len(before), len(after)) <= funds:
        raise ValueError(
            f"a split on {split.isoformat()} leaves returns on {len(before)} days before it "
            f"and {len(after)} after; each side needs more than {funds}, one per fund"
        )
    return before, after


def _held_out(
    daily: pd.DataFrame,
    split: date,
    first_close: pd.Timestamp,
    risk_free: Rate,
    max_weight: float,
    level: float,
) -> dict[str, Any]:
    """The six portfolios chosen from the returns up to ``split`` and held after it."""
    before, after = _sides(daily, split)
    chosen = choose(before, risk_free, max_weight, level)
    out: dict[str, Any] = {
        "date": split.isoformat(),
        "estimated_on": frontier._span(before),
        "held_on": frontier._span(after),
    }
    if isinstance(risk_free, pd.Series):
        # The rate the tangency portfolio was chosen at.
        out["constant_rate"] = frontier.constant_rate(risk_free, before.index)
    return out | {
        "portfolios": {
            name: {
                "weights": {str(t): float(v) for t, v in w.items()},
                "in_sample": describe(before, w, first_close, risk_free, level),
                "held": describe(after, w, before.index[-1], risk_free, level),
            }
            for name, w in chosen.items()
        },
    }


def run(
    prices: pd.DataFrame,
    risk_free: Rate = 0.0,
    max_weight: float = 1.0,
    level: float = 0.95,
    splits: Sequence[date] = frontier.DEFAULT_SPLITS,
) -> dict[str, Any]:
    """The six portfolios from the whole history, and chosen up to each split and held after.

    Compound returns in sample run from the first close; held after a split, from the
    split's last close. A ``risk_free`` Series holds daily rates; the returns are then
    those of the days both cover.
    """
    if len(set(splits)) != len(splits):
        raise ValueError("a split date is given twice")
    daily, risk_free = frontier.on_common_days(ret.total_returns(prices), risk_free)
    first_close = wide(prices, "close").dropna().index[0]
    held_out = [
        _held_out(daily, split, first_close, risk_free, max_weight, level) for split in splits
    ]
    chosen = choose(daily, risk_free, max_weight, level)
    bills = frontier.bill_rate(risk_free, prices, daily)
    if bills:
        bills["constant_rate"] = frontier.constant_rate(risk_free, daily.index)
    return {
        **frontier._span(daily),
        "risk_free": risk_free,
        **bills,
        "max_weight": max_weight,
        "level": level,
        "portfolios": {
            name: {
                "weights": {str(t): float(v) for t, v in w.items()},
                "in_sample": describe(daily, w, first_close, risk_free, level),
            }
            for name, w in chosen.items()
        },
        "splits": held_out,
    }


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _rows(portfolios: dict[str, Any], key: str) -> None:
    print(
        f"{'':18} {'mean':>7} {'vol':>7} {'Sharpe':>7} {'compound':>8} {'max DD':>8} "
        f"{'VaR':>6} {'CVaR':>6} {'top risk':>8}  weights"
    )
    for name, p in portfolios.items():
        d = p[key]
        top = max(d["risk_contributions"].values())
        print(
            f"{name:18} {d['mean']:>7.2%} {d['volatility']:>7.2%} {_ratio(d['sharpe_ratio']):>7} "
            f"{d['compound_return']:>8.2%} {d['max_drawdown']:>8.2%} {d['value_at_risk']:>6.2%} "
            f"{d['cvar']:>6.2%} {top:>8.1%}  {frontier._weights_text(p['weights'])}"
        )


def _print(result: dict[str, Any]) -> None:
    cap = "" if result["max_weight"] == 1.0 else f", at most {result['max_weight']:.0%} a fund"
    print(
        f"{result['daily_returns']} daily returns, {result['first_return']} to "
        f"{result['last_return']}; risk-free rate {frontier.rate_text(result)}; one day's "
        f"value at risk and CVaR at {result['level']:.0%}; long only{cap}"
    )
    print("top risk: the largest share of the portfolio's variance any one fund contributes\n")
    _rows(result["portfolios"], "in_sample")
    for split in result["splits"]:
        first, then = split["estimated_on"], split["held_on"]
        print(
            f"\nchosen from {first['first_return']} to {first['last_return']}, held from "
            f"{then['first_return']} to {then['last_return']}"
        )
        _rows(split["portfolios"], "held")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.construction", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--max-weight",
        type=float,
        default=1.0,
        help="most any fund may hold in the optimised portfolios (default 1)",
    )
    parser.add_argument(
        "--risk-free",
        type=risk._risk_free_argument,
        default=0.0,
        help="compound annual risk-free rate, for the tangency portfolio and the Sharpe "
        f'ratios (default 0), or "{BILLS}" for the daily one-month Treasury bill rate from '
        "the factor cache",
    )
    parser.add_argument(
        "--factors",
        type=Path,
        default=french.DEFAULT_CACHE,
        help=f"the factor cache, for --risk-free {BILLS}",
    )
    parser.add_argument(
        "--level", type=float, default=0.95, help="CVaR confidence level (default 0.95)"
    )
    parser.add_argument(
        "--split",
        action="append",
        help="choose the portfolios from the returns up to this date and hold them after "
        "it; may be given more than once (default: "
        f"{', '.join(d.isoformat() for d in frontier.DEFAULT_SPLITS)}; none for no split)",
    )
    parser.add_argument("--json", type=Path, help="also write the figures to this file")
    args = parser.parse_args(argv)
    try:
        splits = frontier._splits(args.split)
        prices = load_cache(args.cache)
        result = {"cache_sha256": read_manifest(args.cache)["sha256"]}
        risk_free = frontier.load_risk_free(args.risk_free, args.factors, result)
        result |= run(prices, risk_free, args.max_weight, args.level, splits)
        # Written before anything is printed, so a failure leaves no half-finished run.
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.json, "w", encoding="utf-8", newline="\n") as out:
                json.dump(result, out, indent=2, allow_nan=False)
                out.write("\n")
    except (DataError, ValueError, OSError, ArithmeticError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
