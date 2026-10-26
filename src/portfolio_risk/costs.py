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
compound returns after costs are equal: the lowest one, not just one. As the cost
``k`` rises the second rule's return falls, so the first's lead falls no faster than
the first's return, and that is bounded. With the portfolio worth 1 before a trade and
``tau = sum |w - d|`` traded, the sum ``A = sum |w v - d|`` is at most ``tau + (1 - v)``
and ``1 - v = k A``, so ``A <= tau / (1 - k)``; then ``dv/dk = -A / (1 + k sum s w)``
gives ``|d log v / dk| <= tau / ((1 - k)(1 - k - k tau))``. Summed over the trades and
scaled as the compound return is, that bounds how fast the return can fall
(``fall_bound``). A lead of ``g`` cannot be gone within ``g`` over that bound, so the
search steps that far each time and cannot step over a crossing.

``python -m portfolio_risk.costs`` backtests the scenario mixes under every rule over
the cache's daily total returns at a few one-way costs, and gives the break-even cost
for pairs of rules. The cache holds closing prices only, so it cannot measure what the
funds cost to trade: the costs are levels to read the results at, not estimates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_risk import returns as ret
from portfolio_risk import scenarios
from portfolio_risk.data import (
    DEFAULT_CACHE,
    DataError,
    check_output,
    load_cache,
    read_manifest,
    wide,
)
from portfolio_risk.montecarlo import check_weights
from portfolio_risk.scenarios import Backtest, backtest

# The highest one-way cost per unit traded anything here accepts: 10%, far above any
# fund's, and far enough below 1 that Newton's slope stays positive.
HIGHEST = 0.1
# The shortest step ``breakeven``'s search takes, a thousandth of a basis point: a lead
# that comes and goes within less is not looked for.
RESOLUTION = 1e-7


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
        raise ValueError("a cost or spread is a number")
    if not all(math.isfinite(v) and v >= 0.0 for v in values):
        raise ValueError("a cost or spread is finite and not negative")
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
    # The funds, in the order of ``target``'s weights.
    funds: tuple[str, ...] = ()

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
    funds = tuple(str(t) for t in test.weights.columns)
    return Trades(test.returns, target, days, before[days], funds)


# A one-way cost per unit traded: one for every fund, one per fund, or a DataFrame of
# one per fund on each of the backtest's dates.
Cost = float | np.ndarray | pd.DataFrame


def _check_cost(cost: Cost, trades: Trades) -> np.ndarray:
    """The costs as an array: one per fund, or a row per trading close."""
    if isinstance(cost, pd.DataFrame):
        if not cost.index.equals(trades.gross.index) or [str(c) for c in cost.columns] != list(
            trades.funds
        ):
            raise ValueError(
                "costs by day must be on the backtest's dates, a column per fund in its order"
            )
        k = cost.to_numpy(dtype="float64")[trades.days]
    else:
        k = np.broadcast_to(np.asarray(cost, dtype="float64"), (len(trades.target),))
    if not np.isfinite(k).all() or (k < 0.0).any() or (k > HIGHEST).any():
        raise ValueError(f"a one-way cost is from 0 to {HIGHEST:.0%} of what is traded")
    return k


def paid(trades: Trades, cost: Cost) -> pd.Series:
    """The share of the portfolio's value paid in costs at each close (0 where it does
    not trade), at a one-way ``cost`` per unit traded (see ``Cost``)."""
    w, d = trades.target, trades.drifted
    k = _check_cost(cost, trades)
    v = np.ones(len(d))
    # Newton's method: exact after at most one step per fund (see above), one more to
    # spare.
    for _ in range(len(w) + 1):
        s = np.sign(w * v[:, None] - d)
        v = (1.0 + (k * s * d).sum(axis=1)) / (1.0 + (k * s * w).sum(axis=1))
    out = np.zeros(len(trades.gross))
    out[trades.days] = 1.0 - v
    return pd.Series(out, index=trades.gross.index, name="paid")


def net_returns(trades: Trades, cost: Cost) -> pd.Series:
    """The backtest's daily returns after paying ``cost`` on every trade."""
    v = 1.0 - paid(trades, cost).to_numpy()
    return pd.Series((1.0 + trades.gross.to_numpy()) * v - 1.0, index=trades.gross.index)


def summarise(trades: Trades, cost: Cost, periods_per_year: float) -> dict:
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


def fall_bound(trades: Trades, periods_per_year: float, highest: float) -> float:
    """How fast, at most, the compound return falls per unit of one-way cost (the same
    for every fund), anywhere from no cost to ``highest`` (see above)."""
    tau = 2.0 * trades.turnover
    per_trade = tau / ((1.0 - highest) * (1.0 - highest - highest * tau))
    growth = 1.0 + float(ret.annualised_return(trades.gross, periods_per_year))
    return growth * periods_per_year / len(trades.gross) * float(per_trade.sum())


def first_crossing(
    gap: Callable[[float], float], lead: float, slope: float, highest: float
) -> float | None:
    """The lowest cost up to ``highest`` at which ``gap`` falls to 0 or below, given that
    it is ``lead`` (above 0) at no cost and falls no faster than ``slope``; ``None`` if
    it never does.

    From a cost where the gap is ``g`` it cannot reach 0 within ``g / slope``, so the
    search steps that far (at least ``RESOLUTION``), and the step on which the gap is
    gone is narrowed down by bisection to a last bit.
    """
    if slope <= 0.0:
        # It does not fall at all.
        return None
    low, value = 0.0, lead
    while True:
        high = min(low + max(value / slope, RESOLUTION), highest)
        now = gap(high)
        if now <= 0.0:
            break
        if high >= highest:
            return None
        low, value = high, now
    while True:
        middle = (low + high) / 2.0
        if not low < middle < high:
            break
        if gap(middle) > 0.0:
            low = middle
        else:
            high = middle
    return high


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
    The lead falls no faster than the first rule's return (the second's falls too), so
    ``first_crossing`` searches with the first's ``fall_bound``.
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
    if lead > 0.0:
        slope = fall_bound(first, periods_per_year, highest)
        out["cost"] = first_crossing(gap, lead, slope, highest)
    return out


# The narrowest quoted spread allowed: Rule 612 of Regulation NMS sets a minimum quoting
# increment of one cent for stocks (these funds among them) priced at $1 or more, with
# compliance from 31 January 2006. The half-cent increment adopted in 2024 for the most
# tightly quoted stocks had its compliance date postponed past the end of the data
# (to November 2026, then November 2027).
TICK = 0.01
TICK_FROM = date(2006, 1, 31)
# ``--spread`` names the floor this way.
TICK_SPREAD = "tick"


def traded_closes(prices: pd.DataFrame) -> pd.DataFrame:
    """Each fund's close as it traded that day: the cache's closes are adjusted for later
    splits, so a close before a split is multiplied back by every later split's ratio."""
    closes = wide(prices, "close")
    ratios = wide(prices, "split").reindex(closes.index).fillna(0.0)
    ratios = ratios.where(ratios > 0.0, 1.0)
    # The ratios of the splits after each day: on a split's own day the close is already
    # on the new basis.
    later = ratios[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return closes * later


def tick_floor(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """The one-way cost of crossing the narrowest quote the tick allows, half a cent, as
    a share of each fund's traded close on each of ``dates``; 0 before ``TICK_FROM``,
    when no rule set the increment. A lower bound on what an order that crosses the
    displayed quote pays, not on every way to trade: an order filled at the midpoint or
    in the closing auction pays no spread."""
    closes = traded_closes(prices).reindex(dates)
    if closes.isna().to_numpy().any():
        raise ValueError("needs a close for every fund on every date")
    if (closes < 1.0).to_numpy().any():
        raise ValueError("below $1 the tick is not a cent; the floor needs closes of $1 or more")
    floor = 0.5 * TICK / closes
    floor.loc[floor.index < pd.Timestamp(TICK_FROM)] = 0.0
    return floor


# One-way costs per unit traded the command reads the backtests at: 0 to 50 basis points.
LEVELS = (0.0, 0.0001, 0.0005, 0.001, 0.0025, 0.005)
COMPARE = (("daily", "monthly"), ("daily", "band:0.05"), ("band:0.05", "never"))


def run(
    prices: pd.DataFrame,
    mixes: Sequence[str] = tuple(scenarios.MIXES),
    rules: Sequence[str] = scenarios.RULES,
    levels: Sequence[float] = LEVELS,
    spreads: float | Mapping[str, float] | str = 0.0,
    compare: Sequence[Sequence[str]] = COMPARE,
) -> dict[str, Any]:
    """Every mix under every rule at each one-way cost ``level``, and the break-even
    cost of each pair of rules in ``compare`` for each mix.

    A fund's cost at a level is the level plus half its spread in ``spreads``, or with
    ``spreads`` of ``TICK_SPREAD`` the level plus each day's ``tick_floor``. The
    break-even costs are the same for every fund and leave ``spreads`` out.
    """
    if isinstance(spreads, str) and spreads != TICK_SPREAD:
        raise ValueError(f"spreads are numbers, one per fund, or {TICK_SPREAD}, not {spreads!r}")
    daily = ret.total_returns(prices)
    tickers = [str(t) for t in daily.columns]
    if not mixes or not rules or not levels:
        raise ValueError("needs at least one mix, one rule and one cost")
    if len(set(mixes)) != len(mixes):
        raise ValueError("a mix is named twice")
    if len({scenarios._rule_key(rule) for rule in rules}) != len(rules):
        raise ValueError("a rule is named twice")
    if len(set(levels)) != len(levels):
        raise ValueError("a cost is given twice")
    weights = {spec: scenarios._mix(spec, tickers) for spec in mixes}
    seen: dict[tuple[float, ...], str] = {}
    for spec, w in weights.items():
        key = tuple(check_weights(w, tickers).tolist())
        if key in seen:
            raise ValueError(f"{seen[key]} and {spec} are the same mix")
        seen[key] = spec
    for pair in compare:
        if len(pair) != 2 or scenarios._rule_key(pair[0]) == scenarios._rule_key(pair[1]):
            raise ValueError(f"needs two different rules to compare, not {list(pair)!r}")
    # Checked before the backtests, so a bad cost does not wait for them.
    if spreads == TICK_SPREAD:
        floor = tick_floor(prices, daily.index)
        per_level = [floor + per_unit(level, 0.0, tickers) for level in levels]
        spread_figures: Any = {
            "rule": f"half of a {TICK:g} tick on each day's traded close, from "
            f"{TICK_FROM.isoformat()}",
            "mean_one_way": {t: float(floor[t].mean()) for t in tickers},
            "mean_one_way_from_tick_rule": {
                t: float(floor.loc[pd.Timestamp(TICK_FROM) :, t].mean()) for t in tickers
            },
        }
    else:
        per_level = [per_unit(level, spreads, tickers) for level in levels]
        full = spreads if isinstance(spreads, Mapping) else dict.fromkeys(tickers, spreads)
        spread_figures = {t: float(full.get(t, 0.0)) for t in tickers}
    first_close = wide(prices, "close").dropna().index[0]
    per_year = ret.observed_periods_per_year(pd.DatetimeIndex([first_close, *daily.index]))
    history: dict[str, Any] = {}
    for spec, w in weights.items():
        history[spec] = {}
        for rule in rules:
            test = backtest(daily, w, rule)
            made = trades(test, daily)
            summaries = [summarise(made, k, per_year) for k in per_level]
            history[spec][rule] = {
                "compound_return": summaries[0]["compound_return"],
                "rebalances": test.rebalances,
                "turnover_per_year": summaries[0]["turnover_per_year"],
                "costs": [
                    {
                        "cost": level,
                        "net_compound_return": s["net_compound_return"],
                        "drag": s["drag"],
                        "paid_per_year": s["paid_per_year"],
                    }
                    for level, s in zip(levels, summaries, strict=True)
                ],
            }
    return {
        "first_return": daily.index[0].date().isoformat(),
        "last_return": daily.index[-1].date().isoformat(),
        "mixes": weights,
        "rules": list(rules),
        "costs": list(levels),
        "spreads": spread_figures,
        "history": history,
        "breakeven": [
            {
                "rules": list(pair),
                "mixes": {
                    spec: _without_rules(breakeven(daily, w, pair, per_year))
                    for spec, w in weights.items()
                },
            }
            for pair in compare
        ],
    }


def _without_rules(found: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in found.items() if key != "rules"}


def _bp(value: float) -> str:
    return f"{value * 1e4:g} bp"


def _print(result: dict[str, Any]) -> None:
    print(f"daily total returns {result['first_return']} to {result['last_return']}")
    if "rule" in result["spreads"]:
        floor = result["spreads"]["mean_one_way_from_tick_rule"]
        print(
            f"plus the tick floor, {result['spreads']['rule']}; one-way, on average from "
            "then: " + ", ".join(f"{t} {s * 1e4:.2f} bp" for t, s in floor.items())
        )
    elif any(result["spreads"].values()):
        print(
            "plus half of each fund's spread: "
            + ", ".join(f"{t} {_bp(s)}" for t, s in result["spreads"].items() if s)
        )
    levels = result["costs"]
    print(
        "compound return a year before costs, one-way turnover a year, and after a one-way "
        "cost per unit traded of"
    )
    header = "".join(f" {_bp(level):>8}" for level in levels)
    print(f"\n{'':12} {'rule':10} {'before':>8} {'turnover':>8}{header}")
    for spec, by_rule in result["history"].items():
        for rule, s in by_rule.items():
            net = "".join(f" {c['net_compound_return']:>8.2%}" for c in s["costs"])
            print(
                f"{spec:12} {rule:10} {s['compound_return']:>8.2%} "
                f"{s['turnover_per_year']:>8.1%}{net}"
            )
    for entry in result["breakeven"]:
        first, second = entry["rules"]
        print(f"\n{first} over {second}: lead before costs, and the cost that ends it")
        for spec, b in entry["mixes"].items():
            if b["cost"] is not None:
                end = f"{b['cost'] * 1e4:.1f} bp"
            elif b["gap"] <= 0.0:
                end = "no lead"
            else:
                end = f"above {_bp(b['highest'])}"
            print(f"  {spec:12} {b['gap'] * 100:>6.3f} pt  {end}")


def _costs(texts: Sequence[str] | None) -> tuple[float, ...]:
    if texts is None:
        return LEVELS
    try:
        return tuple(float(text) for text in texts)
    except ValueError:
        raise ValueError(
            f"a cost is a number such as 0.0005 (5 basis points), not {' '.join(texts)!r}"
        ) from None


def _spreads(text: str) -> float | dict[str, float] | str:
    """One spread, ``0.0002``, one per fund, ``SPY=0.0001,EEM=0.0006``, or the tick floor."""
    if text.strip().lower() == TICK_SPREAD:
        return TICK_SPREAD
    bad = ValueError(
        f"a spread is a number such as 0.0002, or one per fund such as "
        f"SPY=0.0001,EEM=0.0006, not {text!r}"
    )
    parts = [part.partition("=") for part in text.split(",")] if "=" in text else None
    try:
        if parts is None:
            return float(text)
        values = [(ticker.strip().upper(), float(value)) for ticker, _, value in parts]
    except ValueError:
        raise bad from None
    if any(not ticker for ticker, _ in values):
        raise bad
    out = dict(values)
    if len(out) != len(values):
        raise ValueError("a fund's spread is given twice")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.costs", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--mix",
        action="append",
        help=f"{', '.join(scenarios.MIXES)}, or weights such as SPY=0.6,AGG=0.4; may be "
        "given more than once (default: every named mix)",
    )
    parser.add_argument(
        "--rule",
        action="append",
        help=f"{', '.join(scenarios.RULES[:5])} or band:<width>; may be given more than "
        f"once (default: {', '.join(scenarios.RULES)})",
    )
    parser.add_argument(
        "--cost",
        action="append",
        help="a one-way cost per unit traded, such as 0.0005 for 5 basis points; may be "
        f"given more than once (default: {', '.join(f'{c:g}' for c in LEVELS)})",
    )
    parser.add_argument(
        "--spread",
        default="0",
        help="a quoted bid-ask spread, half of it added to every cost: one for every fund, "
        "such as 0.0002, one per fund, such as SPY=0.0001,EEM=0.0006, or "
        f"{TICK_SPREAD} for a one-cent spread on each day's close (default 0)",
    )
    parser.add_argument(
        "--compare",
        action="append",
        help="two rules, such as daily,monthly, whose break-even cost is found; may be "
        f"given more than once (default: {' and '.join(','.join(p) for p in COMPARE)})",
    )
    parser.add_argument("--json", type=Path, help="also write the figures to this file")
    args = parser.parse_args(argv)
    try:
        check_output(args.json)
        prices = load_cache(args.cache)
        sha256 = read_manifest(args.cache)["sha256"]
        compare = (
            COMPARE
            if args.compare is None
            else [[rule.strip() for rule in text.split(",")] for text in args.compare]
        )
        result = {
            "cache_sha256": sha256,
            **run(
                prices,
                args.mix or tuple(scenarios.MIXES),
                args.rule or scenarios.RULES,
                _costs(args.cost),
                _spreads(args.spread),
                compare,
            ),
        }
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
