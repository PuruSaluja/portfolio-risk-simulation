"""Risk and risk-adjusted return measures of one series of periodic returns.

Rates given per year (``risk_free``, ``target``) are compound annual rates, turned into
the rate per period that compounds to them: ``(1 + annual) ** (1 / periods_per_year) - 1``.

``sharpe_ratio``
    The mean return in excess of the risk-free rate over the standard deviation of
    returns (n - 1), times the square root of ``periods_per_year``.
``downside_deviation``
    The root mean square of the shortfalls below ``target`` (returns above it count as
    0), over all ``n`` returns, as Sortino and Price define it, times the square root of
    ``periods_per_year``.
``sortino_ratio``
    The mean return in excess of ``target``, times ``periods_per_year``, over the
    downside deviation below the same target.
``drawdowns`` and ``max_drawdown``
    The fall of the wealth index ``prod(1 + r)`` from its highest point so far. The
    index starts at 1 before the first return, so a loss on the first day is a
    drawdown from the start.
``value_at_risk`` and ``expected_shortfall``
    For a confidence ``level`` and a tail of ``a = 1 - level``, losses are positive
    numbers. Historical: value at risk is minus the ``ceil(n a)``-th lowest return, the
    smallest return with at least a share ``a`` of returns at or below it; expected
    shortfall is minus the mean of the lowest ``n a`` returns, the last one counted in
    part when ``n a`` is not whole (Acerbi and Tasche, 2002), so it is never below
    value at risk. ``n a`` within 1e-9 of a whole number counts as that number, so
    that 100 returns at 95% take the 5th lowest, not the 6th that rounding
    ``1 - 0.95`` would give. Normal: from the sample mean and standard deviation,
    ``-(mean + sd z)`` and ``-(mean - sd phi(z) / a)``, where ``z`` is the standard
    normal's ``a`` quantile.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd

from portfolio_risk import returns as ret

Method = Literal["historical", "normal"]
LEVELS = (0.95, 0.99)


def _values(returns: pd.Series, minimum: int = 1) -> np.ndarray:
    # The same checks as the returns helpers: a Series, finite, above -100%.
    return ret._series(returns, minimum)


def per_period_rate(annual: float, periods_per_year: float) -> float:
    """The rate per period that compounds to ``annual`` over ``periods_per_year`` periods."""
    ret._check_periods(periods_per_year)
    if (
        isinstance(annual, bool)
        or not isinstance(annual, int | float)
        or not math.isfinite(annual)
        or annual <= -1.0
    ):
        raise ValueError(f"an annual rate must be a finite number above -100%, not {annual!r}")
    return math.expm1(math.log1p(annual) / periods_per_year)


def sharpe_ratio(returns: pd.Series, periods_per_year: float, risk_free: float = 0.0) -> float:
    """Mean excess return over its standard deviation, on a yearly scale."""
    values = _values(returns, minimum=2)
    rate = per_period_rate(risk_free, periods_per_year)
    if np.ptp(values) == 0.0:
        raise ValueError("returns that never vary have no Sharpe ratio")
    excess = values - rate
    return float(excess.mean() / excess.std(ddof=1) * math.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series, periods_per_year: float, target: float = 0.0) -> float:
    """Root mean square shortfall below ``target`` over all returns, on a yearly scale."""
    values = _values(returns)
    rate = per_period_rate(target, periods_per_year)
    shortfall = np.minimum(values - rate, 0.0)
    return float(math.sqrt(np.mean(shortfall**2)) * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float, target: float = 0.0) -> float:
    """Mean return in excess of ``target`` over the downside deviation below it, yearly."""
    values = _values(returns)
    downside = downside_deviation(returns, periods_per_year, target)
    if downside == 0.0:
        raise ValueError("no return falls below the target, so the Sortino ratio has no bound")
    rate = per_period_rate(target, periods_per_year)
    return float((values.mean() - rate) * periods_per_year / downside)


def drawdowns(returns: pd.Series) -> pd.Series:
    """The wealth index's fall from its highest point so far, at each date (0 or below)."""
    values = _values(returns)
    wealth = np.cumprod(1.0 + values)
    # The running high includes the starting wealth of 1.
    high = np.maximum.accumulate(np.maximum(wealth, 1.0))
    return pd.Series(wealth / high - 1.0, index=returns.index, name="drawdown")


@dataclass(frozen=True)
class Drawdown:
    """The deepest fall of the wealth index: its depth (0 or below) and its dates.

    ``peak`` is the date of the high it fell from, ``None`` when that is the start
    (before the first return). ``recovery`` is the first date the index was back at
    or above that high, ``None`` if it never was. The day counts are numbers of
    returns: from the peak to the trough, and from the peak to the recovery.
    """

    depth: float
    peak: Hashable | None
    trough: Hashable | None
    recovery: Hashable | None
    days_to_trough: int
    days_to_recovery: int | None


def max_drawdown(returns: pd.Series) -> Drawdown:
    """The deepest drawdown; on a tie, the first. A depth of 0 if the index never falls."""
    values = _values(returns)
    # Position 0 is the start, before the first return.
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + values)])
    high = np.maximum.accumulate(wealth)
    fall = wealth / high - 1.0
    trough = int(np.argmin(fall))
    if fall[trough] == 0.0:
        return Drawdown(0.0, None, None, None, 0, None)
    # The last time the index stood at the high, before the trough.
    peak = int(np.flatnonzero(wealth[: trough + 1] == high[trough])[-1])
    back = np.flatnonzero(wealth[trough:] >= wealth[peak])
    recovery = trough + int(back[0]) if back.size else None
    labels = returns.index
    return Drawdown(
        depth=float(fall[trough]),
        peak=None if peak == 0 else labels[peak - 1],
        trough=labels[trough - 1],
        recovery=None if recovery is None else labels[recovery - 1],
        days_to_trough=trough - peak,
        days_to_recovery=None if recovery is None else recovery - peak,
    )


def _tail(level: float) -> float:
    if isinstance(level, bool) or not isinstance(level, int | float) or not 0.0 < level < 1.0:
        raise ValueError(f"level must be a number between 0 and 1, not {level!r}")
    return 1.0 - level


def _tail_size(n: int, tail: float) -> float:
    size = n * tail
    nearest = round(size)
    return float(nearest) if abs(size - nearest) <= 1e-9 else size


def value_at_risk(returns: pd.Series, level: float = 0.95, method: Method = "historical") -> float:
    """The loss that returns exceed with chance ``1 - level``, as a positive number."""
    tail = _tail(level)
    values = _values(returns, minimum=2 if method == "normal" else 1)
    if method == "historical":
        k = max(math.ceil(_tail_size(len(values), tail)), 1)
        return float(-np.partition(values, k - 1)[k - 1])
    if method == "normal":
        z = NormalDist().inv_cdf(tail)
        return float(-(values.mean() + values.std(ddof=1) * z))
    raise ValueError(f"method must be historical or normal, not {method!r}")


def expected_shortfall(
    returns: pd.Series, level: float = 0.95, method: Method = "historical"
) -> float:
    """The mean loss in the worst ``1 - level`` of outcomes, as a positive number."""
    tail = _tail(level)
    values = _values(returns, minimum=2 if method == "normal" else 1)
    if method == "historical":
        size = _tail_size(len(values), tail)
        whole = math.floor(size)
        lowest = np.sort(values)[: whole + 1]
        if whole == 0:
            return float(-lowest[0])
        total = lowest[:whole].sum()
        if size > whole:
            total += (size - whole) * lowest[whole]
        # A mean of returns at or below the value-at-risk return can come out a last bit
        # above it in floating point; it is never above it in fact.
        return float(max(-total / size, -lowest[math.ceil(size) - 1]))
    if method == "normal":
        z = NormalDist().inv_cdf(tail)
        density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return float(-(values.mean() - values.std(ddof=1) * density / tail))
    raise ValueError(f"method must be historical or normal, not {method!r}")


def _label(value: Hashable | None) -> Any:
    if value is None:
        return None
    return value.date().isoformat() if isinstance(value, pd.Timestamp) else value


def measures(
    returns: pd.Series,
    periods_per_year: float,
    risk_free: float = 0.0,
    levels: Sequence[float] = LEVELS,
) -> dict[str, Any]:
    """Every measure above for one series, the risk-free rate also the Sortino target.

    Value at risk and expected shortfall are of a single period's return.
    """
    worst = max_drawdown(returns)
    tails = {}
    for level in levels:
        key = f"{level * 100:g}"
        tails[key] = {
            method: {
                "value_at_risk": value_at_risk(returns, level, method),
                "expected_shortfall": expected_shortfall(returns, level, method),
            }
            for method in ("historical", "normal")
        }
    return {
        "compound_return": float(ret.annualised_return(returns, periods_per_year)),
        "volatility": float(ret.annualised_volatility(returns, periods_per_year)),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year, risk_free),
        "downside_deviation": downside_deviation(returns, periods_per_year, risk_free),
        "sortino_ratio": sortino_ratio(returns, periods_per_year, risk_free),
        "max_drawdown": {
            "depth": worst.depth,
            "peak": _label(worst.peak),
            "trough": _label(worst.trough),
            "recovery": _label(worst.recovery),
            "days_to_trough": worst.days_to_trough,
            "days_to_recovery": worst.days_to_recovery,
        },
        "tail": tails,
    }
