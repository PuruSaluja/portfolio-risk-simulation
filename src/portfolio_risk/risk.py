"""Risk and risk-adjusted return measures of one series of periodic returns.

Rates given per year (``risk_free``, ``target``) are compound annual rates, turned into
the rate per period that compounds to them: ``(1 + annual) ** (1 / periods_per_year) - 1``.
Either may instead be a Series of rates per period on the returns' dates, such as the
daily Treasury bill rate ``french.daily_risk_free`` gives, which is subtracted day by
day.

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

``python -m portfolio_risk.risk`` gives every measure for each fund in the price cache
and for a portfolio of them rebalanced daily, from daily total returns, with the Sharpe
ratio of calendar-month returns beside the daily one. ``--risk-free bills`` measures
the ratios against the one-month Treasury bill from the factor cache, over the days it
covers (the monthly ratio against the bill's return compounded over the same months);
``--json`` writes the figures to a file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd

from portfolio_risk import french
from portfolio_risk import returns as ret
from portfolio_risk.data import DEFAULT_CACHE, DataError, load_cache, read_manifest, wide
from portfolio_risk.factors import align
from portfolio_risk.montecarlo import check_weights, historical_portfolio_returns, parse_weights

Method = Literal["historical", "normal"]
LEVELS = (0.95, 0.99)
# A compound annual rate, or rates per period on the returns' dates.
Rate = float | pd.Series
BILLS = "bills"


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


def per_period_rates(rate: Rate, returns: pd.Series, periods_per_year: float) -> float | np.ndarray:
    """The rate per period: from a compound annual rate, or a Series of them as given,
    refused unless it is on the returns' dates."""
    if not isinstance(rate, pd.Series):
        return per_period_rate(rate, periods_per_year)
    ret._check_periods(periods_per_year)
    if not rate.index.equals(returns.index):
        raise ValueError("a series of risk-free rates must be on the returns' dates")
    values = rate.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("risk-free rates must be finite numbers above -100%")
    return values


def sharpe_ratio(returns: pd.Series, periods_per_year: float, risk_free: Rate = 0.0) -> float:
    """Mean excess return over its standard deviation, on a yearly scale."""
    values = _values(returns, minimum=2)
    excess = values - per_period_rates(risk_free, returns, periods_per_year)
    if np.ptp(excess) == 0.0:
        raise ValueError("excess returns that never vary have no Sharpe ratio")
    return float(excess.mean() / excess.std(ddof=1) * math.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series, periods_per_year: float, target: Rate = 0.0) -> float:
    """Root mean square shortfall below ``target`` over all returns, on a yearly scale."""
    values = _values(returns)
    shortfall = np.minimum(values - per_period_rates(target, returns, periods_per_year), 0.0)
    return float(math.sqrt(np.mean(shortfall**2)) * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float, target: Rate = 0.0) -> float:
    """Mean return in excess of ``target`` over the downside deviation below it, yearly."""
    values = _values(returns)
    downside = downside_deviation(returns, periods_per_year, target)
    if downside == 0.0:
        raise ValueError("no return falls below the target, so the Sortino ratio has no bound")
    excess = values - per_period_rates(target, returns, periods_per_year)
    return float(np.mean(excess) * periods_per_year / downside)


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
        # Adding 0 turns minus zero (a return of exactly 0) into zero.
        return float(-np.partition(values, k - 1)[k - 1]) + 0.0
    if method == "normal":
        z = NormalDist().inv_cdf(tail)
        return float(-(values.mean() + values.std(ddof=1) * z)) + 0.0
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
            return float(-lowest[0]) + 0.0
        total = lowest[:whole].sum()
        if size > whole:
            total += (size - whole) * lowest[whole]
        # A mean of returns at or below the value-at-risk return can come out a last bit
        # above it in floating point; it is never above it in fact.
        return float(max(-total / size, -lowest[math.ceil(size) - 1])) + 0.0
    if method == "normal":
        z = NormalDist().inv_cdf(tail)
        density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return float(-(values.mean() - values.std(ddof=1) * density / tail)) + 0.0
    raise ValueError(f"method must be historical or normal, not {method!r}")


def or_none(measure, *args) -> float | None:
    """``measure(*args)``, or ``None`` where the returns leave it undefined (a Sharpe
    ratio of returns that never vary, a Sortino ratio of returns never below target)."""
    try:
        return measure(*args)
    except ValueError:
        return None


def date_text(value: Hashable | None) -> Any:
    """A date as ``YYYY-MM-DD`` for a JSON file; ``None`` and other labels as they are."""
    if value is None:
        return None
    return value.date().isoformat() if isinstance(value, pd.Timestamp) else value


def measures(
    returns: pd.Series,
    periods_per_year: float,
    risk_free: Rate = 0.0,
    levels: Sequence[float] = LEVELS,
) -> dict[str, Any]:
    """Every measure above for one series, the risk-free rate also the Sortino target.

    Value at risk and expected shortfall are of a single period's return. A ratio the
    returns leave undefined (Sharpe for returns that never vary, Sortino for returns
    that never fall below the target) is ``None``.
    """
    # Checked here, so that a bad rate is refused rather than reported as undefined.
    per_period_rates(risk_free, returns, periods_per_year)
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
        "volatility": float(ret.annualised_volatility(returns, periods_per_year)),
        "sharpe_ratio": or_none(sharpe_ratio, returns, periods_per_year, risk_free),
        "downside_deviation": downside_deviation(returns, periods_per_year, risk_free),
        "sortino_ratio": or_none(sortino_ratio, returns, periods_per_year, risk_free),
        "max_drawdown": {
            "depth": worst.depth,
            "peak": date_text(worst.peak),
            "trough": date_text(worst.trough),
            "recovery": date_text(worst.recovery),
            "days_to_trough": worst.days_to_trough,
            "days_to_recovery": worst.days_to_recovery,
        },
        "tail": tails,
    }


def run(prices: pd.DataFrame, weights_text: str, risk_free: Rate = 0.0) -> dict[str, Any]:
    """The measures of every fund and of a portfolio of them, from the cache's daily returns.

    The portfolio is rebalanced to its weights at every close. Compound returns are per
    calendar year, with the trading days the data holds a year; everything else on a
    yearly scale uses 252 days, or 12 months for the Sharpe ratio of calendar-month
    returns. A ``risk_free`` Series holds daily rates; the returns are then measured
    over the days both cover, which must be the same trading days, and the monthly
    ratio is against the rates compounded over each month's days.
    """
    daily = ret.total_returns(prices)
    tickers = [str(t) for t in daily.columns]
    weights = parse_weights(weights_text, tickers)
    daily_rate = monthly_rate = risk_free
    if isinstance(risk_free, pd.Series):
        daily, rates = align(daily, risk_free.to_frame("rate"))
        daily_rate = rates["rate"]
        monthly_rate = ret.compound(daily_rate, "M")
    series = {t: daily[t] for t in tickers}
    series["portfolio"] = historical_portfolio_returns(daily, weights)
    first_close = wide(prices, "close").dropna().index[0]
    per_year = ret.observed_periods_per_year(pd.DatetimeIndex([first_close, *daily.index]))
    out = {}
    for name, returns in series.items():
        monthly = ret.compound(returns, "M")
        out[name] = {
            "compound_return": float(ret.annualised_return(returns, per_year)),
            **measures(returns, ret.TRADING_DAYS_PER_YEAR, daily_rate),
            "sharpe_ratio_monthly": or_none(sharpe_ratio, monthly, 12, monthly_rate),
        }
    result = {
        "first_return": daily.index[0].date().isoformat(),
        "last_return": daily.index[-1].date().isoformat(),
        "daily_returns": len(daily),
        "risk_free": risk_free,
        "weights": dict(zip(tickers, check_weights(weights, tickers).tolist(), strict=True)),
        "series": out,
    }
    if isinstance(risk_free, pd.Series):
        result["risk_free"] = "one-month Treasury bill"
        # Compound over the days used, as the funds' compound returns are.
        result["risk_free_compound"] = float(ret.annualised_return(daily_rate, per_year))
    return result


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _print(result: dict[str, Any]) -> None:
    weights = ", ".join(f"{t} {w:.1%}" for t, w in result["weights"].items() if w > 0)
    if "risk_free_compound" in result:
        rate = f"{result['risk_free']}, {result['risk_free_compound']:.2%} a year compounded"
    else:
        rate = f"{result['risk_free']:.2%} a year"
    print(
        f"{result['daily_returns']} daily returns, {result['first_return']} to "
        f"{result['last_return']}; risk-free rate {rate}\n"
        f"portfolio, rebalanced daily: {weights}"
    )
    print(
        f"\n{'':10} {'compound':>8} {'vol':>7} {'Sharpe':>7} {'Sh(m)':>7} {'Sortino':>7} "
        f"{'downside':>8} {'max DD':>8}  {'peak':10} {'trough':10} {'recovered':10}"
    )
    for name, m in result["series"].items():
        dd = m["max_drawdown"]
        if dd["depth"] == 0.0:
            dates = f"{'never fell':32}"
        else:
            dates = f"{dd['peak'] or 'start':10} {dd['trough']:10} {dd['recovery'] or 'not yet':10}"
        print(
            f"{name:10} {m['compound_return']:>8.2%} {m['volatility']:>7.2%} "
            f"{_ratio(m['sharpe_ratio']):>7} {_ratio(m['sharpe_ratio_monthly']):>7} "
            f"{_ratio(m['sortino_ratio']):>7} {m['downside_deviation']:>8.2%} "
            f"{dd['depth']:>8.2%}  {dates}"
        )
    levels = list(next(iter(result["series"].values()))["tail"])
    print("\nOne day's loss: value at risk / expected shortfall, historical (normal)")
    print(f"{'':10}" + "".join(f" {level + '%':>29}" for level in levels))
    for name, m in result["series"].items():
        cells = []
        for level in levels:
            h, n = m["tail"][level]["historical"], m["tail"][level]["normal"]
            cells.append(
                f"{h['value_at_risk']:.2%} ({n['value_at_risk']:.2%}) / "
                f"{h['expected_shortfall']:.2%} ({n['expected_shortfall']:.2%})"
            )
        print(f"{name:10}" + "".join(f" {c:>29}" for c in cells))


def _risk_free_argument(text: str) -> float | str:
    if text == BILLS:
        return BILLS
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"a compound annual rate such as 0.02, or {BILLS}, not {text!r}"
        ) from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.risk",
        description="Risk and risk-adjusted return of each fund and a portfolio over the cache",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--weights", default="equal", help='"equal" (the default) or e.g. SPY=0.6,AGG=0.4'
    )
    parser.add_argument(
        "--risk-free",
        type=_risk_free_argument,
        default=0.0,
        help="compound annual risk-free rate, also the Sortino target (default 0), or "
        f'"{BILLS}" for the daily one-month Treasury bill rate from the factor cache',
    )
    parser.add_argument(
        "--factors",
        type=Path,
        default=french.DEFAULT_CACHE,
        help=f"the factor cache, for --risk-free {BILLS}",
    )
    parser.add_argument("--json", type=Path, help="also write the figures to this file")
    args = parser.parse_args(argv)
    try:
        prices = load_cache(args.cache)
        result = {"cache_sha256": read_manifest(args.cache)["sha256"]}
        risk_free = args.risk_free
        if risk_free == BILLS:
            risk_free = french.daily_risk_free(french.load_factors(args.factors))
            result["factors_sha256"] = french.read_manifest(args.factors)["sha256"]
        result |= run(prices, args.weights, risk_free)
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
