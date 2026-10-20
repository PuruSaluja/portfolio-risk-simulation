"""Returns from the cached prices, and helpers that put them on a yearly scale.

``total_returns`` turns the cache's closes and cash payouts into daily total returns,
the way a holder who reinvests every payout earns them::

    r[t] = (close[t] + payout[t]) / close[t - 1] - 1

where ``payout`` is the dividend plus any capital-gain distribution going ex on day
``t`` (the convention CRSP uses). The closes are already adjusted for splits, so splits
need nothing further. Returns are kept only on days every fund traded; a payout on a
day that is dropped is added to the next day kept, so no cash is lost.
``price_returns`` gives the same returns without the payouts.

``compound`` turns daily returns into weekly, monthly, quarterly or yearly ones, each
labelled with the last trading day it covers. The first period starts at the first
close, so it can be shorter than the rest.

Annualisation, for ``periods_per_year`` returns a year (252 for trading days, 12 for
months):

``annualised_mean``
    The arithmetic mean times ``periods_per_year``: the expected one-year return
    that a mean-variance model uses.
``annualised_volatility``
    The sample standard deviation times the square root of ``periods_per_year``,
    which assumes returns are uncorrelated from one period to the next.
``annualised_return``
    The compound (geometric) growth rate per year, what the fund actually returned:
    ``prod(1 + r) ** (periods_per_year / n) - 1``.

``observed_periods_per_year`` counts how many periods the data really holds per
year, to check the 252 convention against the calendar.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from portfolio_risk.data import wide

TRADING_DAYS_PER_YEAR = 252
PERIODS_PER_YEAR = {"D": TRADING_DAYS_PER_YEAR, "W": 52, "M": 12, "Q": 4, "Y": 1}

Frequency = Literal["W", "M", "Q", "Y"]
# pandas period codes for each frequency; weeks end on Friday, as trading weeks do.
_PERIOD_CODES = {"W": "W-FRI", "M": "M", "Q": "Q", "Y": "Y"}

Returns = pd.Series | pd.DataFrame


def _common_closes(prices: pd.DataFrame) -> pd.DataFrame:
    close = wide(prices, "close")
    common = close.dropna()
    if len(common) < 2:
        raise ValueError("fewer than two days on which every fund has a close")
    return common


def total_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily total returns, date by ticker, on the days every fund traded."""
    payout = (wide(prices, "dividend") + wide(prices, "capital_gain")).fillna(0.0)
    common = _common_closes(prices)
    # Payouts on days that are dropped move to the next day kept.
    carried = payout.cumsum().loc[common.index].diff()
    returns = (common + carried) / common.shift(1) - 1.0
    return returns.iloc[1:]


def price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily returns from the closes alone, leaving out dividends and distributions."""
    common = _common_closes(prices)
    return (common / common.shift(1) - 1.0).iloc[1:]


def _check(returns: Returns, minimum: int = 1) -> None:
    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError(f"expected a pandas Series or DataFrame, not {type(returns).__name__}")
    values = returns.to_numpy(dtype="float64")
    if len(returns) < minimum:
        raise ValueError(f"needs at least {minimum} returns, got {len(returns)}")
    if not np.isfinite(values).all():
        raise ValueError("returns include NaN or infinite values")
    if (values <= -1.0).any():
        raise ValueError("a return of -100% or less (a price at or below zero) cannot compound")


def _check_periods(periods_per_year: float) -> None:
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int | float)
        or not math.isfinite(periods_per_year)
        or periods_per_year <= 0
    ):
        raise ValueError(f"periods_per_year must be a positive number, not {periods_per_year!r}")


def log_returns(returns: Returns) -> Returns:
    """``log(1 + r)``: returns that add up over time rather than compound."""
    _check(returns)
    return np.log1p(returns)


def compound(returns: Returns, frequency: Frequency) -> Returns:
    """Daily returns compounded into ``frequency`` periods, labelled by their last day."""
    if frequency not in _PERIOD_CODES:
        raise ValueError(f"frequency must be one of {', '.join(_PERIOD_CODES)}, not {frequency!r}")
    _check(returns)
    index = pd.DatetimeIndex(returns.index)
    periods = index.to_period(_PERIOD_CODES[frequency])
    grown = (1.0 + returns).groupby(periods).prod() - 1.0
    grown.index = pd.DatetimeIndex(index.to_series().groupby(periods).max(), name=index.name)
    return grown


def annualised_mean(returns: Returns, periods_per_year: float) -> float | pd.Series:
    """The arithmetic mean return times ``periods_per_year``."""
    _check(returns)
    _check_periods(periods_per_year)
    return returns.mean() * periods_per_year


def annualised_volatility(returns: Returns, periods_per_year: float) -> float | pd.Series:
    """The sample standard deviation (n - 1) times the square root of ``periods_per_year``."""
    _check(returns, minimum=2)
    _check_periods(periods_per_year)
    return returns.std(ddof=1) * math.sqrt(periods_per_year)


def annualised_return(returns: Returns, periods_per_year: float) -> float | pd.Series:
    """The compound growth rate per year: ``prod(1 + r) ** (periods_per_year / n) - 1``."""
    _check(returns)
    _check_periods(periods_per_year)
    # Summing logs keeps a long product from overflowing or losing precision.
    growth = np.log1p(returns).sum() * (periods_per_year / len(returns))
    return np.expm1(growth)


def observed_periods_per_year(index: pd.Index) -> float:
    """How many periods the data holds per calendar year, counting gaps between dates."""
    index = pd.DatetimeIndex(index)
    if len(index) < 2:
        raise ValueError("needs at least two dates")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("dates must be increasing and unique")
    years = (index[-1] - index[0]).days / 365.25
    return (len(index) - 1) / years
