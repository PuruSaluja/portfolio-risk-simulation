"""Mean-variance optimisation: the efficient frontier of long-only, fully invested portfolios.

Every portfolio here holds each fund at a weight between 0 and ``max_weight`` (1 unless
capped), the weights adding up to 1. From expected yearly returns ``mu`` and a yearly
covariance matrix ``S`` of the funds, a portfolio ``w`` has expected return ``w' mu``
and volatility ``sqrt(w' S w)``.

``minimum_variance``
    The portfolio with the lowest volatility.
``efficient_portfolio``
    The portfolio with the lowest volatility among those whose expected return is
    exactly ``target``. Below the minimum-variance portfolio's return such a portfolio
    is dominated (another has a higher return at the same volatility), so the
    efficient frontier runs from the minimum-variance portfolio's return to the highest
    return any portfolio has.
``frontier``
    That many efficient portfolios, their target returns evenly spaced along it.
``max_sharpe``
    The portfolio with the highest expected excess return per unit of volatility, the
    tangency portfolio. It lies on the frontier, where the ratio rises to one peak and
    falls (the frontier's volatility is convex in its return). At the peak the line
    from the risk-free rate touches the frontier: ``var = (t - r) d(var / 2) / dt``,
    and the slope is minus the multiplier of the return constraint. The sign of
    ``var + (t - r) multiplier`` changes once along the frontier, from positive to
    negative, so a bisection on it finds the peak to rounding, caps or not. (A search
    on the ratio itself cannot: near its peak the ratio is flat to the last digit over
    a stretch of the frontier whose weights differ.)

Each portfolio is the exact solution of a quadratic programme, found by a primal
active-set method (Nocedal and Wright, *Numerical Optimization*, section 16.5): the
weights held at 0 or at the cap form the working set, the rest solve the equality
constrained problem, and a bound is added when a step meets it and dropped when its
multiplier has the wrong sign. The covariance matrix must be positive definite.

``estimates`` gives the inputs from daily returns: the arithmetic mean times 252 and
the sample (or Ledoit-Wolf) covariance times 252. With the sample covariance, a
portfolio's ratio here is the Sharpe ratio ``risk.sharpe_ratio`` gives its daily
returns, rebalanced every close, at the same risk-free rate.

The risk-free rate may instead be daily rates, such as the Treasury bill rate
``french.daily_risk_free`` gives. The tangency portfolio and the expected ratios need
one constant rate: ``constant_rate`` gives the one whose yearly arithmetic rate is 252
times the daily rates' mean over the returns the estimates come from, so the expected
excess return is the mean daily excess return times 252, as with a constant rate. The
realised ratios subtract each day's rate, as ``risk.sharpe_ratio`` does.

``python -m portfolio_risk.frontier`` gives the frontier, the minimum-variance and
tangency portfolios and equal weights from the cache's daily total returns, and the
same three chosen from the returns up to each of a few split dates and held after it:
what the optimiser expected of them, and what they returned. Held after a split, each
is also charged for its trades (``costs.held``): rebalanced every close, monthly and
within a 5-point band, at one-way costs of 0 to 50 basis points.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from portfolio_risk import costs, french, risk
from portfolio_risk import covariance as cov
from portfolio_risk import returns as ret
from portfolio_risk.data import (
    DEFAULT_CACHE,
    DataError,
    check_output,
    load_cache,
    prices_between,
    read_manifest,
    wide,
)
from portfolio_risk.factors import align
from portfolio_risk.montecarlo import historical_portfolio_returns
from portfolio_risk.risk import BILLS, Rate, per_period_rate

Estimator = Literal["sample", "ledoit-wolf"]
# Bisection steps for the tangency portfolio: a share of the frontier between 0 and 1
# is down to rounding after about 53.
BISECTION_STEPS = 100


def _inputs(expected: pd.Series | None, covariance: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """The covariance (checked positive definite) and expected returns as arrays."""
    matrix = cov._check_covariance(covariance)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise ValueError(
            "the covariance matrix is not positive definite (a fund whose returns are a "
            "mix of the others' makes it singular)"
        ) from None
    if expected is None:
        return (matrix,)
    if not isinstance(expected, pd.Series):
        raise TypeError(f"expected returns are a Series by fund, not {type(expected).__name__}")
    if list(expected.index) != list(covariance.index):
        raise ValueError("the expected returns and the covariance must name the same funds")
    mu = expected.to_numpy(dtype="float64")
    if not np.isfinite(mu).all():
        raise ValueError("expected returns include NaN or infinite values")
    return matrix, mu


def _cap(max_weight: float, n: int) -> float:
    if (
        isinstance(max_weight, bool)
        or not isinstance(max_weight, int | float)
        or not 0.0 < max_weight <= 1.0
    ):
        raise ValueError(f"max_weight must be above 0 and at most 1, not {max_weight!r}")
    # Room only for the rounding of a cap of 1/n (1/49 times 49 is a last bit below 1):
    # a cap any further short cannot add up to 1, and equal weights would break it.
    if max_weight * n < 1.0 - 1e-15:
        raise ValueError(
            f"{n} funds capped at {max_weight:g} cannot add up to 1 (the cap is at least 1/{n})"
        )
    return float(max_weight)


def _solve(
    q: np.ndarray, a: np.ndarray, b: np.ndarray, hi: float, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Minimise ``x'qx / 2`` subject to ``a x = b`` and ``0 <= x <= hi``.

    ``x`` must be feasible; ``q`` positive definite and ``a`` of full row rank. Returns
    the solution and the multipliers of ``a x = b``: the objective falls by about
    ``multiplier * db`` when ``b`` grows by ``db``.
    """
    n = len(x)
    m = len(b)
    x = x.astype("float64", copy=True)
    # 0: free; -1: held at 0; +1: held at the cap.
    held = np.zeros(n, dtype=int)
    for _ in range(50 * (n + m)):
        free = held == 0
        k = int(free.sum())
        g = q @ x
        kkt = np.zeros((k + m, k + m))
        kkt[:k, :k] = q[np.ix_(free, free)]
        kkt[:k, k:] = a[:, free].T
        kkt[k:, :k] = a[:, free]
        solution = np.linalg.solve(kkt, np.concatenate([-g[free], np.zeros(m)]))
        step = np.zeros(n)
        step[free] = solution[:k]
        if np.max(np.abs(step), initial=0.0) <= 1e-13:
            # At the minimum for this working set: a held bound's multiplier must push
            # against it (up from 0, down from the cap), or letting it go lowers the
            # objective.
            pull = -(g + a.T @ solution[k:]) * held
            tolerance = 1e-12 * max(1.0, float(np.max(np.abs(g))))
            worst = int(np.argmin(pull))
            if pull[worst] >= -tolerance:
                return x, solution[k:]
            held[worst] = 0
            continue
        # The longest step, up to the full one, that keeps every weight within bounds.
        length, blocking, side = 1.0, -1, 0
        for i in np.flatnonzero(free):
            if step[i] < 0.0 and -x[i] / step[i] < length:
                length, blocking, side = -x[i] / step[i], i, -1
            elif step[i] > 0.0 and (hi - x[i]) / step[i] < length:
                length, blocking, side = (hi - x[i]) / step[i], i, 1
        x += length * step
        if blocking >= 0:
            held[blocking] = side
        # Held weights exactly at their bounds, not a rounding error either side.
        x[held < 0] = 0.0
        x[held > 0] = hi
    raise RuntimeError("the active-set method did not converge")


def _weights(x: np.ndarray, covariance: pd.DataFrame) -> pd.Series:
    # Adding 0 turns minus zero into zero.
    return pd.Series(x + 0.0, index=covariance.index, name="weight")


def minimum_variance(covariance: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    """The long-only, fully invested portfolio with the lowest volatility."""
    (matrix,) = _inputs(None, covariance)
    n = len(matrix)
    hi = _cap(max_weight, n)
    x, _ = _solve(matrix, np.ones((1, n)), np.ones(1), hi, np.full(n, 1.0 / n))
    return _weights(x, covariance)


def _greedy(mu: np.ndarray, hi: float, highest: bool) -> np.ndarray:
    # Fill the funds from the highest expected return down (or lowest up) to the cap.
    x = np.zeros(len(mu))
    left = 1.0
    for i in np.argsort(-mu if highest else mu, kind="stable"):
        x[i] = min(hi, left)
        left -= x[i]
        if left <= 0.0:
            break
    return x


def return_range(expected: pd.Series, max_weight: float = 1.0) -> tuple[float, float]:
    """The lowest and highest expected return of any long-only portfolio under the cap."""
    mu = expected.to_numpy(dtype="float64")
    if not np.isfinite(mu).all() or len(mu) == 0:
        raise ValueError("expected returns must be finite, at least one fund")
    hi = _cap(max_weight, len(mu))
    return float(_greedy(mu, hi, False) @ mu), float(_greedy(mu, hi, True) @ mu)


def _ends(mu: np.ndarray, hi: float) -> tuple[np.ndarray, np.ndarray, float]:
    """The lowest- and highest-return portfolios, and how far apart their returns are."""
    low, high = _greedy(mu, hi, False), _greedy(mu, hi, True)
    return low, high, float(high @ mu - low @ mu)


def _equal_returns(mu: np.ndarray, width: float) -> bool:
    return width <= 1e-12 * max(1.0, float(np.max(np.abs(mu))))


def _on_frontier(
    matrix: np.ndarray, mu: np.ndarray, hi: float, share: float
) -> tuple[np.ndarray, float]:
    """The efficient portfolio a ``share`` of the way from the lowest to the highest return,
    and the multiplier of its return constraint, per unit of that share."""
    n = len(mu)
    low, high, width = _ends(mu, hi)
    # A point on the line from the lowest-return to the highest-return portfolio is a
    # feasible start.
    start = low + share * (high - low)
    # Fully invested, w' mu = target is w' (mu - lowest) / width = share. Returns that
    # differ in the ninth decimal leave the row mu nearly parallel to the row of ones,
    # and the system singular; the scaled row is not.
    rows = np.vstack([np.ones(n), (mu - low @ mu) / width])
    x, multipliers = _solve(matrix, rows, np.array([1.0, share]), hi, start)
    return x, float(multipliers[1])


def efficient_portfolio(
    expected: pd.Series, covariance: pd.DataFrame, target: float, max_weight: float = 1.0
) -> pd.Series:
    """The lowest-volatility portfolio whose expected return is exactly ``target``."""
    matrix, mu = _inputs(expected, covariance)
    hi = _cap(max_weight, len(mu))
    if isinstance(target, bool) or not isinstance(target, int | float):
        raise ValueError(f"the target return must be a number, not {target!r}")
    low, high, width = _ends(mu, hi)
    span = (float(low @ mu), float(high @ mu))
    scale = max(1.0, float(np.max(np.abs(mu))))
    if not span[0] - 1e-12 * scale <= target <= span[1] + 1e-12 * scale:
        raise ValueError(
            f"no portfolio has an expected return of {target:.6g}: they run from "
            f"{span[0]:.6g} to {span[1]:.6g}"
        )
    if _equal_returns(mu, width):
        # Every portfolio has the same expected return; the target asks nothing more.
        return minimum_variance(covariance, max_weight)
    share = min(max((target - span[0]) / width, 0.0), 1.0)
    x, _ = _on_frontier(matrix, mu, hi, share)
    return _weights(x, covariance)


def moments(
    weights: pd.Series, expected: pd.Series, covariance: pd.DataFrame
) -> tuple[float, float]:
    """A portfolio's expected return and volatility."""
    matrix, mu = _inputs(expected, covariance)
    if list(weights.index) != list(covariance.index):
        raise ValueError("the weights and the covariance must name the same funds")
    w = weights.to_numpy(dtype="float64")
    return float(w @ mu), float(math.sqrt(max(w @ matrix @ w, 0.0)))


def frontier(
    expected: pd.Series, covariance: pd.DataFrame, points: int = 21, max_weight: float = 1.0
) -> pd.DataFrame:
    """``points`` efficient portfolios, from the minimum-variance one to the highest return.

    One row per portfolio: its expected return, volatility and weights.
    """
    if isinstance(points, bool) or not isinstance(points, int) or points < 2:
        raise ValueError(f"points must be a whole number of at least 2, not {points!r}")
    start, _ = moments(minimum_variance(covariance, max_weight), expected, covariance)
    _, end = return_range(expected, max_weight)
    rows = []
    for target in np.linspace(start, end, points):
        w = efficient_portfolio(expected, covariance, float(min(target, end)), max_weight)
        mean, sd = moments(w, expected, covariance)
        rows.append({"expected_return": mean, "volatility": sd, **w.to_dict()})
    return pd.DataFrame(rows)


def max_sharpe(
    expected: pd.Series,
    covariance: pd.DataFrame,
    risk_free: float = 0.0,
    max_weight: float = 1.0,
    periods_per_year: float = ret.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """The portfolio with the highest ratio of expected excess return to volatility.

    ``risk_free`` is a compound annual rate, turned into the rate per period that
    compounds to it and then into the yearly arithmetic rate ``periods_per_year``
    periods of it add up to, the scale of ``expected``.
    """
    matrix, mu = _inputs(expected, covariance)
    hi = _cap(max_weight, len(mu))
    rate = per_period_rate(risk_free, periods_per_year) * periods_per_year
    low, high, width = _ends(mu, hi)
    if high @ mu <= rate:
        raise ValueError(
            f"no portfolio's expected return ({high @ mu:.4%} at most) is above the "
            f"risk-free rate ({rate:.4%}), so none has a positive Sharpe ratio"
        )
    least = minimum_variance(covariance, max_weight)
    if _equal_returns(mu, width):
        return least

    def touching(share: float) -> tuple[np.ndarray, float]:
        # Positive where the ratio still rises along the frontier, negative where it falls.
        x, multiplier = _on_frontier(matrix, mu, hi, share)
        return x, float(x @ matrix @ x + (x @ mu - rate) * multiplier / width)

    top, sign = touching(1.0)
    if sign >= 0.0:
        return _weights(top, covariance)
    # The minimum-variance portfolio's multiplier is 0, so the sign there is positive.
    a, below = max(float((least.to_numpy() - low) @ mu / width), 0.0), least.to_numpy()
    b = 1.0
    for _ in range(BISECTION_STEPS):
        middle = (a + b) / 2.0
        if not a < middle < b:
            break
        x, sign = touching(middle)
        if sign > 0.0:
            a, below = middle, x
        else:
            b = middle
    # The two ends of the bracket are now a last bit of the share apart.
    return _weights(below, covariance)


def sharpe(
    weights: pd.Series,
    expected: pd.Series,
    covariance: pd.DataFrame,
    risk_free: float = 0.0,
    periods_per_year: float = ret.TRADING_DAYS_PER_YEAR,
) -> float:
    """A portfolio's expected excess return over its volatility, the rate as in ``max_sharpe``."""
    mean, sd = moments(weights, expected, covariance)
    return (mean - per_period_rate(risk_free, periods_per_year) * periods_per_year) / sd


def estimates(daily: pd.DataFrame, estimator: Estimator = "sample") -> tuple:
    """Yearly expected returns and covariance from daily returns: means and covariance x 252."""
    days = ret.TRADING_DAYS_PER_YEAR
    expected = ret.annualised_mean(daily, days)
    if estimator == "sample":
        matrix = cov.sample_covariance(daily)
    elif estimator == "ledoit-wolf":
        matrix, _ = cov.ledoit_wolf_covariance(daily)
    else:
        raise ValueError(f"the covariance estimator is sample or ledoit-wolf, not {estimator!r}")
    return expected, cov.annualise(matrix, days)


def rates_on(risk_free: Rate, days: pd.DatetimeIndex) -> Rate:
    """``risk_free`` for returns on ``days``: a constant rate as it is, daily rates on
    those days, refused unless every one of them has a rate."""
    if not isinstance(risk_free, pd.Series):
        return risk_free
    missing = days.difference(risk_free.index)
    if len(missing):
        raise ValueError(f"no risk-free rate on {missing[0].date()} ({len(missing)} days)")
    return risk_free.loc[days]


def constant_rate(risk_free: Rate, days: pd.DatetimeIndex) -> float:
    """A compound annual rate for ``max_sharpe`` and ``sharpe``: ``risk_free`` itself,
    or for daily rates the one their mean over ``days`` compounds to in 252 days, so
    that the yearly arithmetic rate those take from it is 252 times the mean."""
    if not isinstance(risk_free, pd.Series):
        return risk_free
    mean = float(rates_on(risk_free, days).mean())
    return math.expm1(ret.TRADING_DAYS_PER_YEAR * math.log1p(mean))


def on_common_days(daily: pd.DataFrame, risk_free: Rate) -> tuple[pd.DataFrame, Rate]:
    """For daily rates, the returns and the rates over the days both cover, refused
    unless they hold the same trading days there; a constant rate leaves both as given."""
    if not isinstance(risk_free, pd.Series):
        return daily, risk_free
    daily, rates = align(daily, risk_free.to_frame("rate"))
    return daily, rates["rate"]


def bill_rate(risk_free: Rate, prices: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    """For daily rates, the keys that say so in the output: the rate's name and what it
    compounds to a year over ``daily``'s days, as the risk measures give it."""
    if not isinstance(risk_free, pd.Series):
        return {}
    first_close = wide(prices, "close").dropna().index[0]
    per_year = ret.observed_periods_per_year(pd.DatetimeIndex([first_close, *daily.index]))
    return {
        "risk_free": "one-month Treasury bill",
        "risk_free_compound": float(ret.annualised_return(risk_free, per_year)),
    }


# The ends of every third year from 2009: each split leaves at least four years on
# either side.
DEFAULT_SPLITS = tuple(date(year, 12, 31) for year in (2009, 2012, 2015, 2018, 2021))


def _choose(
    expected: pd.Series, covariance: pd.DataFrame, risk_free: float, max_weight: float
) -> dict[str, pd.Series]:
    n = len(expected)
    return {
        "minimum_variance": minimum_variance(covariance, max_weight),
        "max_sharpe": max_sharpe(expected, covariance, risk_free, max_weight),
        # Equal weight is the comparison, so it ignores the cap: 1/n is within any
        # cap the funds can meet.
        "equal": pd.Series(1.0 / n, index=expected.index, name="weight"),
    }


def _described(
    w: pd.Series, expected: pd.Series, covariance: pd.DataFrame, risk_free: float
) -> dict[str, Any]:
    mean, sd = moments(w, expected, covariance)
    return {
        "weights": {str(t): float(v) for t, v in w.items()},
        "expected_return": mean,
        "volatility": sd,
        "sharpe_ratio": sharpe(w, expected, covariance, risk_free),
    }


def _realised(daily: pd.DataFrame, w: pd.Series, risk_free: Rate, first: pd.Timestamp) -> dict:
    """What a portfolio, rebalanced every close, returned over ``daily``; daily rates are
    subtracted day by day. Under ``costs`` the same held under each of
    ``costs.HELD_RULES`` and charged each of ``costs.LEVELS``."""
    history = historical_portfolio_returns(daily, w.to_dict())
    per_year = ret.observed_periods_per_year(pd.DatetimeIndex([first, *daily.index]))
    days = ret.TRADING_DAYS_PER_YEAR
    rates = rates_on(risk_free, daily.index)
    return {
        "mean": float(ret.annualised_mean(history, days)),
        "volatility": float(ret.annualised_volatility(history, days)),
        "sharpe_ratio": risk.or_none(risk.sharpe_ratio, history, days, rates),
        "compound_return": float(ret.annualised_return(history, per_year)),
        "costs": costs.held(daily, w.to_dict(), first, rates),
    }


def _span(daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "first_return": daily.index[0].date().isoformat(),
        "last_return": daily.index[-1].date().isoformat(),
        "daily_returns": len(daily),
    }


def split_sides(daily: pd.DataFrame, split: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The returns up to the close of ``split`` and after it, refused unless each side
    has more days than there are funds."""
    before = daily.loc[daily.index <= pd.Timestamp(split)]
    after = daily.loc[daily.index > pd.Timestamp(split)]
    funds = daily.shape[1]
    # A covariance matrix of n funds needs more than n days to be positive definite.
    if min(len(before), len(after)) <= funds:
        # Where the returns run says why a side is short; with the bill rate they end
        # with the factor cache, which can be before the prices do.
        raise ValueError(
            f"a split on {split.isoformat()} leaves returns on {len(before)} days before it "
            f"and {len(after)} after; each side needs more than {funds}, one per fund (the "
            f"returns run from {daily.index[0].date()} to {daily.index[-1].date()})"
        )
    return before, after


def _held_out(
    daily: pd.DataFrame, split: date, estimator: Estimator, risk_free: Rate, max_weight: float
) -> dict[str, Any]:
    """The three portfolios chosen from the returns up to ``split`` and held after it.

    For daily rates the tangency portfolio and the expected ratios before the split take
    the constant rate of the days before it, those chosen after it of the days after.
    """
    before, after = split_sides(daily, split)
    early = estimates(before, estimator)
    late = estimates(after, estimator)
    rate_before = constant_rate(risk_free, before.index)
    rate_after = constant_rate(risk_free, after.index)
    held = _choose(*early, rate_before, max_weight)
    out: dict[str, Any] = {
        "date": split.isoformat(),
        "estimated_on": _span(before),
        "held_on": _span(after),
    }
    if isinstance(risk_free, pd.Series):
        out["constant_rate"] = {"estimated_on": rate_before, "held_on": rate_after}
    return out | {
        "portfolios": {
            name: {
                **_described(w, *early, rate_before),
                "realised": _realised(after, w, risk_free, before.index[-1]),
            }
            for name, w in held.items()
        },
        "chosen_after": {
            name: _described(w, *late, rate_after)
            for name, w in _choose(*late, rate_after, max_weight).items()
            if name != "equal"
        },
    }


def run(
    prices: pd.DataFrame,
    estimator: Estimator = "sample",
    max_weight: float = 1.0,
    points: int = 21,
    risk_free: Rate = 0.0,
    splits: Sequence[date] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    """The frontier and three portfolios from the whole history, and for each of
    ``splits`` the same portfolios chosen from the returns up to that date and held after.

    The three are the minimum-variance portfolio, the tangency portfolio and equal
    weights. Held after a split, each is rebalanced to its weights at every close. A
    ``risk_free`` Series holds daily rates; the returns are then those of the days both
    cover.
    """
    if len(set(splits)) != len(splits):
        raise ValueError("a split date is given twice")
    daily, risk_free = on_common_days(ret.total_returns(prices), risk_free)
    # Checked before the whole-history optimisation, so a bad date does not wait for it.
    held_out = [_held_out(daily, split, estimator, risk_free, max_weight) for split in splits]
    expected, covariance = estimates(daily, estimator)
    rate = constant_rate(risk_free, daily.index)
    chosen = _choose(expected, covariance, rate, max_weight)
    table = frontier(expected, covariance, points, max_weight)
    tickers = [str(t) for t in daily.columns]
    bills = bill_rate(risk_free, prices, daily)
    if bills:
        bills["constant_rate"] = rate
    result = {
        **_span(daily),
        "estimator": estimator,
        "max_weight": max_weight,
        "risk_free": risk_free,
        **bills,
        "funds": {
            str(t): {
                "expected_return": float(expected[t]),
                "volatility": float(math.sqrt(covariance.loc[t, t])),
            }
            for t in daily.columns
        },
        "portfolios": {
            name: _described(w, expected, covariance, rate) for name, w in chosen.items()
        },
        "frontier": [
            {
                "expected_return": float(row["expected_return"]),
                "volatility": float(row["volatility"]),
                "sharpe_ratio": sharpe(row[tickers], expected, covariance, rate),
                "weights": {t: float(row[t]) for t in tickers},
            }
            for _, row in table.iterrows()
        ],
        "splits": held_out,
    }
    if bills:
        # The same at a zero rate on the same days, so that a comparison with the zero
        # rate shows the rate alone.
        trimmed = prices_between(prices, daily.index[0], daily.index[-1])
        zero = run(trimmed, estimator, max_weight, points, 0.0, splits)
        result["at_zero_rate"] = risk.same_days(zero, result)
    return result


def _weights_text(weights: dict[str, float]) -> str:
    return ", ".join(f"{t} {v:.1%}" for t, v in weights.items() if v >= 0.0005) or "-"


def rate_text(result: dict[str, Any], constant_for: str = "expected ratios") -> str:
    """The risk-free rate as the commands print it; for daily rates, what the constant
    rate of their mean is used for."""
    if "risk_free_compound" not in result:
        return f"{result['risk_free']:.2%} a year"
    return (
        f"{result['risk_free']}, {result['risk_free_compound']:.2%} a year compounded "
        f"({constant_for} at its mean, {result['constant_rate']:.2%})"
    )


def _print(result: dict[str, Any]) -> None:
    cap = "" if result["max_weight"] == 1.0 else f", at most {result['max_weight']:.0%} a fund"
    print(
        f"{result['daily_returns']} daily returns, {result['first_return']} to "
        f"{result['last_return']}; {result['estimator']} covariance; risk-free rate "
        f"{rate_text(result)}; long only{cap}"
    )
    print(f"\n{'':18} {'expected':>8} {'vol':>7} {'Sharpe':>7}  weights")
    for name, p in result["portfolios"].items():
        print(
            f"{name:18} {p['expected_return']:>8.2%} {p['volatility']:>7.2%} "
            f"{p['sharpe_ratio']:>7.3f}  {_weights_text(p['weights'])}"
        )
    print(f"\nefficient frontier\n{'expected':>8} {'vol':>7} {'Sharpe':>7}  weights")
    for row in result["frontier"]:
        print(
            f"{row['expected_return']:>8.2%} {row['volatility']:>7.2%} "
            f"{row['sharpe_ratio']:>7.3f}  {_weights_text(row['weights'])}"
        )
    for split in result["splits"]:
        first, then = split["estimated_on"], split["held_on"]
        print(
            f"\nchosen from {first['first_return']} to {first['last_return']}, held from "
            f"{then['first_return']} to {then['last_return']}: expected, then realised"
        )
        print(
            f"{'':18} {'expected':>8} {'vol':>7} {'Sharpe':>7} | {'mean':>7} {'vol':>7} "
            f"{'Sharpe':>7} {'compound':>8}"
        )
        for name, p in split["portfolios"].items():
            r = p["realised"]
            sharpe_text = "-" if r["sharpe_ratio"] is None else f"{r['sharpe_ratio']:.3f}"
            print(
                f"{name:18} {p['expected_return']:>8.2%} {p['volatility']:>7.2%} "
                f"{p['sharpe_ratio']:>7.3f} | {r['mean']:>7.2%} {r['volatility']:>7.2%} "
                f"{sharpe_text:>7} {r['compound_return']:>8.2%}"
            )
        for name, p in split["portfolios"].items():
            if name != "equal":
                print(f"  {name} chosen before: {_weights_text(p['weights'])}")
                later = split["chosen_after"][name]["weights"]
                print(f"  {name} chosen after:  {_weights_text(later)}")
        costs.print_held({name: p["realised"]["costs"] for name, p in split["portfolios"].items()})


def _splits(texts: Sequence[str] | None) -> tuple[date, ...]:
    if texts is None:
        return DEFAULT_SPLITS
    if [text.strip().lower() for text in texts] == ["none"]:
        return ()
    try:
        return tuple(date.fromisoformat(text.strip()) for text in texts)
    except ValueError:
        raise ValueError(
            f"a split is a date such as 2015-12-31, or none alone, not {' '.join(texts)!r}"
        ) from None


def load_risk_free(argument: float | str, factors: Path, result: dict[str, Any]) -> Rate:
    """The rate ``--risk-free`` names: the number, or the daily bill rate from the
    factor cache, whose checksum then goes into ``result``."""
    if argument != BILLS:
        return argument
    rates = french.daily_risk_free(french.load_factors(factors))
    result["factors_sha256"] = french.read_manifest(factors)["sha256"]
    return rates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.frontier", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--covariance", choices=["sample", "ledoit-wolf"], default="sample", help="estimator"
    )
    parser.add_argument(
        "--max-weight", type=float, default=1.0, help="most any fund may hold (default 1)"
    )
    parser.add_argument(
        "--points", type=int, default=21, help="portfolios along the frontier (default 21)"
    )
    parser.add_argument(
        "--risk-free",
        type=risk._risk_free_argument,
        default=0.0,
        help=f'compound annual risk-free rate (default 0), or "{BILLS}" for the daily '
        "one-month Treasury bill rate from the factor cache",
    )
    parser.add_argument(
        "--factors",
        type=Path,
        default=french.DEFAULT_CACHE,
        help=f"the factor cache, for --risk-free {BILLS}",
    )
    parser.add_argument(
        "--split",
        action="append",
        help="choose the portfolios from the returns up to this date and hold them after "
        "it; may be given more than once (default: "
        f"{', '.join(d.isoformat() for d in DEFAULT_SPLITS)}; none for no split)",
    )
    parser.add_argument("--json", type=Path, help="also write the figures to this file")
    args = parser.parse_args(argv)
    try:
        check_output(args.json)
        splits = _splits(args.split)
        prices = load_cache(args.cache)
        result = {"cache_sha256": read_manifest(args.cache)["sha256"]}
        risk_free = load_risk_free(args.risk_free, args.factors, result)
        result |= run(prices, args.covariance, args.max_weight, args.points, risk_free, splits)
        # Written before anything is printed, so a failure leaves no half-finished run.
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.json, "w", encoding="utf-8", newline="\n") as out:
                json.dump(result, out, indent=2, allow_nan=False)
                out.write("\n")
    except (DataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
