"""Trading costs in the historical backtests: commissions, bid-ask spreads and the drag
they put on the return.

A trade of a value ``x`` in a fund costs ``k x``, ``k`` being the one-way cost per unit
traded: a commission plus half the fund's quoted bid-ask spread (buying at the ask and
selling at the bid, a round trip crosses the whole spread). Buys and sells cost the
same. At a rebalance the costs are paid out of the portfolio at the close, which then
holds exactly its target weights. With the portfolio worth 1 before trading, drifted
weights ``d`` and targets ``w``, the value ``v`` left after trading solves

    v + sum_i k_i |w_i v - d_i| = 1.

The left side is convex, piecewise linear and increasing in ``v`` (its slope is at
least ``1 - max k``), so Newton's method from ``v = 1`` (to the right of the root)
stays to the right and lands on the root of one piece after another, at most one step
per fund: ``v = (1 + sum k s d) / (1 + sum k s w)``, ``s`` the signs of the trades.

Costs scale the portfolio's value, never its weights: after a rebalance it holds the
targets whatever it paid, and between rebalances the weights drift by the funds'
returns alone. So the closes on which a rule trades, even a band rule, and the weights
it trades from are those of the cost-free backtest ``scenarios.backtest`` gives, and
``trades`` takes them from it. A day's net return is ``(1 + r) v - 1``, ``r`` the
day's return before costs and ``v`` 1 on a close without a trade.

The purchase at the first close is not charged: the backtests start invested, and it
is the same for every rule. Distributions are reinvested free, as in the total returns.

``breakeven`` gives the one-way cost, the same for every fund, at which two rules'
compound returns after costs are equal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_risk import returns as ret
from portfolio_risk.scenarios import Backtest, backtest

# The highest one-way cost per unit traded anything here accepts: 10%, far above any
# fund's, and far enough below 1 that Newton's slope stays positive.
HIGHEST = 0.1
# ``breakeven`` scans the costs in steps of a basis point before narrowing down.
SCAN_STEP = 1e-4


def per_unit(
    commission: float, spreads: float | Mapping[str, float], tickers: Sequence[str]
) -> np.ndarray:
    """Each fund's one-way cost per unit traded: the commission plus half its spread.

    ``spreads`` is one quoted bid-ask spread for every fund, or one per fund (as a share
    of the price, 0.0002 for 2 basis points); a fund left out has none.
    """
    if not isinstance(spreads, Mapping):
        spreads = dict.fromkeys(tickers, spreads)
    unknown = sorted(set(spreads) - set(tickers))
    if unknown:
        raise ValueError(f"no returns for {', '.join(unknown)}")
    values = [commission, *spreads.values()]
    if any(isinstance(v, bool) or not isinstance(v, int | float) for v in values):
        raise ValueError("a commission or spread is a number")
    if not all(math.isfinite(v) and v >= 0.0 for v in values):
        raise ValueError("a commission or spread is finite and not negative")
    k = commission + 0.5 * np.array([float(spreads.get(t, 0.0)) for t in tickers])
    if k.max(initial=0.0) > HIGHEST:
        raise ValueError(f"a one-way cost above {HIGHEST:.0%} of what is traded is refused")
    return k


@dataclass(frozen=True, eq=False)
class Trades:
    """A backtest's trades: its returns before costs, its targets, which closes trade,
    and on each of those the weights the portfolio has drifted to before trading."""

    gross: pd.Series
    target: np.ndarray
    days: np.ndarray
    drifted: np.ndarray

    @property
    def turnover(self) -> np.ndarray:
        """The one-way share traded at each trading close, as ``scenarios`` gives it."""
        return 0.5 * np.abs(self.target - self.drifted).sum(axis=1)


def trades(test: Backtest, returns: pd.DataFrame) -> Trades:
    """The trades of ``test``, a backtest of ``returns``."""
    if not returns.index.equals(test.returns.index) or list(returns.columns) != list(
        test.weights.columns
    ):
        raise ValueError("the returns must be those the backtest was run on")
    values = returns.to_numpy(dtype="float64")
    target = np.array(list(test.target.values()))
    after = test.weights.to_numpy()
    # Each close's weights before trading: the previous close's, grown by the day.
    before = np.vstack([target, after[:-1]]) * (1.0 + values)
    before /= before.sum(axis=1, keepdims=True)
    days = test.rebalanced.to_numpy()
    return Trades(test.returns, target, days, before[days])


def _check_cost(cost: float | np.ndarray, n: int) -> np.ndarray:
    k = np.broadcast_to(np.asarray(cost, dtype="float64"), (n,))
    if not np.isfinite(k).all() or (k < 0.0).any() or (k > HIGHEST).any():
        raise ValueError(f"a one-way cost is from 0 to {HIGHEST:.0%} of what is traded")
    return k


def paid(trades: Trades, cost: float | np.ndarray) -> pd.Series:
    """The share of the portfolio's value paid in costs at each close (0 where it does
    not trade), at a one-way ``cost`` per unit traded, one for all funds or one each."""
    w, d = trades.target, trades.drifted
    k = _check_cost(cost, len(w))
    v = np.ones(len(d))
    # Newton's method: exact after at most one step per fund (see above), one more to
    # spare.
    for _ in range(len(w) + 1):
        s = np.sign(w * v[:, None] - d)
        v = (1.0 + (k * s * d).sum(axis=1)) / (1.0 + (k * s * w).sum(axis=1))
    out = np.zeros(len(trades.gross))
    out[trades.days] = 1.0 - v
    return pd.Series(out, index=trades.gross.index, name="paid")


def net_returns(trades: Trades, cost: float | np.ndarray) -> pd.Series:
    """The backtest's daily returns after paying ``cost`` on every trade."""
    v = 1.0 - paid(trades, cost).to_numpy()
    return pd.Series((1.0 + trades.gross.to_numpy()) * v - 1.0, index=trades.gross.index)


def summarise(trades: Trades, cost: float | np.ndarray, periods_per_year: float) -> dict:
    """Compound return a year before and after costs, the drag (the difference), and the
    costs paid and the one-way turnover, a year."""
    years = len(trades.gross) / periods_per_year
    gross = float(ret.annualised_return(trades.gross, periods_per_year))
    net = float(ret.annualised_return(net_returns(trades, cost), periods_per_year))
    return {
        "compound_return": gross,
        "net_compound_return": net,
        "drag": gross - net,
        "paid_per_year": float(paid(trades, cost).sum() / years),
        "turnover_per_year": float(trades.turnover.sum() / years),
    }


def breakeven(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    rules: Sequence[str],
    periods_per_year: float,
    highest: float = 0.01,
) -> dict:
    """The lowest one-way cost, the same for every fund, at which the first rule's
    compound return after costs falls to the second's.

    ``gap`` is the first's lead before costs, in compound return a year. ``cost`` is
    ``None`` when the first does not lead before costs, or still leads at ``highest``.
    The costs are scanned a basis point at a time, and the first step at which the
    lead is gone is narrowed down by bisection to a last bit.
    """
    if len(rules) != 2:
        raise ValueError(f"needs two rules to compare, not {list(rules)!r}")
    if not 0.0 < highest <= HIGHEST:
        raise ValueError(f"highest is above 0 and at most {HIGHEST:.0%}, not {highest!r}")
    first, second = (trades(backtest(returns, weights, rule), returns) for rule in rules)

    def gap(k: float) -> float:
        return float(
            ret.annualised_return(net_returns(first, k), periods_per_year)
            - ret.annualised_return(net_returns(second, k), periods_per_year)
        )

    lead = gap(0.0)
    out = {"rules": list(rules), "gap": lead, "highest": highest, "cost": None}
    if lead <= 0.0:
        return out
    low = 0.0
    for high in np.arange(1, round(highest / SCAN_STEP) + 1) * SCAN_STEP:
        if gap(float(high)) <= 0.0:
            break
        low = float(high)
    else:
        return out
    high = float(high)
    while True:
        middle = (low + high) / 2.0
        if not low < middle < high:
            break
        if gap(middle) > 0.0:
            low = middle
        else:
            high = middle
    out["cost"] = high
    return out
