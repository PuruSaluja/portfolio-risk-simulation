"""Historical scenarios: allocation mixes, rebalancing rules and stress periods.

``backtest`` follows a portfolio through the funds' daily returns. It is invested at the
target weights at the first close; each day every holding grows by its fund's return,
so the weights drift, and a rebalancing rule decides on which closes they are reset to
the targets:

``daily``
    Every close: the fixed-weight portfolio the Monte Carlo and risk modules use.
``monthly``, ``quarterly``, ``yearly``
    The last close of each calendar month, quarter or year in the data.
``never``
    Buy and hold: the weights drift for good.
``band:x``
    Any close at which a fund's weight has drifted more than ``x`` from its target
    (``band:0.05`` is 5 percentage points either way).

The last close of the data is never a rebalance: no return follows it. Turnover at a
rebalance is the one-way share of the portfolio traded, half the sum of the absolute
changes in weight. No trading costs are charged.

``MIXES`` names a few allocations of the eight funds. ``STRESS_PERIODS`` names six
falls of the S&P 500, each from its closing high to its closing low (SPY's closes
peak and bottom on the same days). ``stress`` invests a mix at its targets at the
close of a period's first day and follows it, under a rule, to the close of the last.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from portfolio_risk import returns as ret
from portfolio_risk import risk
from portfolio_risk.montecarlo import check_weights, parse_weights

MIXES = {
    "equity": "SPY=0.25,IWM=0.25,EFA=0.25,EEM=0.25",
    "60/40": "SPY=0.6,AGG=0.4",
    "equal": "equal",
    # Four assets for four economic conditions, after Harry Browne; AGG stands in for
    # his cash.
    "permanent": "SPY=0.25,TLT=0.25,GLD=0.25,AGG=0.25",
}
RULES = ("daily", "monthly", "quarterly", "yearly", "never", "band:0.05")
STRESS_PERIODS = {
    "2008 financial crisis": (date(2007, 10, 9), date(2009, 3, 9)),
    "2011 euro debt crisis": (date(2011, 4, 29), date(2011, 10, 3)),
    "2015-16 China and oil": (date(2015, 5, 21), date(2016, 2, 11)),
    "2018 fourth quarter": (date(2018, 9, 20), date(2018, 12, 24)),
    "2020 COVID crash": (date(2020, 2, 19), date(2020, 3, 23)),
    "2022 inflation and rates": (date(2022, 1, 3), date(2022, 10, 12)),
}
# pandas period codes for the calendar rules.
_CALENDAR = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}


def _band(rule: str) -> float | None:
    """The band's width for a ``band:x`` rule, ``None`` for the others."""
    name, colon, width = rule.partition(":")
    if name != "band" or not colon:
        if rule in RULES[:5]:
            return None
        raise ValueError(
            f"a rule is daily, monthly, quarterly, yearly, never or band:<width>, not {rule!r}"
        )
    try:
        value = float(width)
    except ValueError:
        raise ValueError(f"the band's width is not a number: {width!r}") from None
    if not 0.0 < value < 1.0:
        raise ValueError(f"the band's width must be between 0 and 1, not {width!r}")
    return value


def _calendar_closes(index: pd.DatetimeIndex, rule: str) -> np.ndarray:
    """Whether each close is a rebalance under a rule that does not depend on drift."""
    n = len(index)
    if rule == "daily":
        due = np.ones(n, dtype=bool)
    elif rule == "never":
        due = np.zeros(n, dtype=bool)
    else:
        periods = index.to_period(_CALENDAR[rule])
        due = np.append(periods[1:] != periods[:-1], True)
    due[-1] = False
    return due


@dataclass(frozen=True, eq=False)
class Backtest:
    """A portfolio's path: daily returns, the weights after each close, which closes
    rebalanced, and the turnover at each (0 where none, or where nothing had drifted)."""

    rule: str
    target: dict[str, float]
    returns: pd.Series
    weights: pd.DataFrame
    rebalanced: pd.Series
    turnover: pd.Series

    @property
    def rebalances(self) -> int:
        return int(self.rebalanced.sum())


def backtest(returns: pd.DataFrame, weights: Mapping[str, float], rule: str) -> Backtest:
    """Follow a portfolio invested at ``weights`` through ``returns`` under ``rule``."""
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] == 0 or len(returns) < 1:
        raise ValueError("expected a date-by-fund DataFrame with at least one day")
    index = pd.DatetimeIndex(returns.index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("dates must be increasing and unique")
    values = returns.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("returns must be finite and above -100%")
    tickers = [str(t) for t in returns.columns]
    target = check_weights(weights, tickers)
    band = _band(rule)
    due = _calendar_closes(index, "never" if band is not None else rule)
    n = len(values)
    out = np.empty(n)
    after = np.empty_like(values)
    turnover = np.zeros(n)
    rebalanced = np.zeros(n, dtype=bool)
    held = target.copy()
    for t in range(n):
        grown = held * (1.0 + values[t])
        out[t] = grown.sum() - 1.0
        drifted = grown / (1.0 + out[t])
        rebalance = due[t] or (
            band is not None and t < n - 1 and np.abs(drifted - target).max() > band
        )
        if rebalance:
            rebalanced[t] = True
            turnover[t] = 0.5 * np.abs(target - drifted).sum()
            held = target.copy()
        else:
            held = drifted
        after[t] = held
    return Backtest(
        rule,
        dict(zip(tickers, target.tolist(), strict=True)),
        pd.Series(out, index=returns.index, name="portfolio"),
        pd.DataFrame(after, index=returns.index, columns=returns.columns),
        pd.Series(rebalanced, index=returns.index, name="rebalanced"),
        pd.Series(turnover, index=returns.index, name="turnover"),
    )


def window(returns: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """The returns from the close of ``start`` to the close of ``end``, both trading days."""
    index = pd.DatetimeIndex(returns.index)
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if not first < last:
        raise ValueError(f"a period must end after it starts, not {start} to {end}")
    for day in (first, last):
        if day not in index:
            raise ValueError(f"{day.date()} is not a day with returns for every fund")
    return returns[(index > first) & (index <= last)]


def _path(test: Backtest) -> dict[str, Any]:
    worst = risk.max_drawdown(test.returns)
    return {
        "return": float(np.expm1(np.log1p(test.returns).sum())),
        "max_drawdown": worst.depth,
        "trough": risk._label(worst.trough),
    }


def summarise(test: Backtest, periods_per_year: float) -> dict[str, Any]:
    """Return, risk and trading over a whole backtest."""
    worst = risk.max_drawdown(test.returns)
    years = len(test.returns) / periods_per_year
    return {
        "compound_return": float(ret.annualised_return(test.returns, periods_per_year)),
        "volatility": float(ret.annualised_volatility(test.returns, ret.TRADING_DAYS_PER_YEAR)),
        "sharpe_ratio": risk._defined(risk.sharpe_ratio, test.returns, ret.TRADING_DAYS_PER_YEAR),
        "max_drawdown": worst.depth,
        "max_drawdown_peak": risk._label(worst.peak),
        "max_drawdown_trough": risk._label(worst.trough),
        "rebalances": test.rebalances,
        "turnover_per_year": float(test.turnover.sum() / years),
        "final_weights": {t: float(w) for t, w in test.weights.iloc[-1].items()},
    }


def stress(
    returns: pd.DataFrame, weights: Mapping[str, float], rule: str, start: date, end: date
) -> dict[str, Any]:
    """A mix invested at its targets at the close of ``start``, followed to ``end``."""
    return _path(backtest(window(returns, start, end), weights, rule))


def _mix(spec: str, tickers: Sequence[str]) -> dict[str, float]:
    return parse_weights(MIXES.get(spec, spec), tickers)
