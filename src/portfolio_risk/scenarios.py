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

``shuffled_gap`` asks whether the difference two rules make comes from the order of the
days: it compares the gap between them on the days as they came with the gaps on the
same days shuffled.

``python -m portfolio_risk.scenarios`` backtests every mix under every rule over the
whole history and over each stress period, from the cache's daily total returns, and
the gap between two rules with the days in order and shuffled.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_risk import returns as ret
from portfolio_risk import risk
from portfolio_risk.data import DEFAULT_CACHE, DataError, load_cache, read_manifest, wide
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
    """The band's width for a ``band:x`` rule, ``None`` for the others (any case)."""
    name, colon, width = rule.strip().lower().partition(":")
    if name != "band" or not colon:
        if name in RULES[:5]:
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


def _rule_key(rule: str) -> str:
    """A rule in one spelling, so that ``Monthly`` and ``monthly``, or ``band:.05`` and
    ``band:0.05``, are one rule."""
    band = _band(rule)
    return rule.strip().lower() if band is None else f"band:{band!r}"


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
    # Numbers read as dates would be nanoseconds from 1970, all in one month.
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns need a date index, one date per day")
    index = returns.index
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("dates must be increasing and unique")
    values = returns.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("returns must be finite and above -100%")
    tickers = [str(t) for t in returns.columns]
    target = check_weights(weights, tickers)
    key = _rule_key(rule)
    band = _band(key)
    due = _calendar_closes(index, "never" if band is not None else key)
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
        "trough": risk.date_text(worst.trough),
    }


def summarise(test: Backtest, periods_per_year: float) -> dict[str, Any]:
    """Return, risk and trading over a whole backtest."""
    worst = risk.max_drawdown(test.returns)
    years = len(test.returns) / periods_per_year
    return {
        "compound_return": float(ret.annualised_return(test.returns, periods_per_year)),
        "volatility": float(ret.annualised_volatility(test.returns, ret.TRADING_DAYS_PER_YEAR)),
        "sharpe_ratio": risk.or_none(risk.sharpe_ratio, test.returns, ret.TRADING_DAYS_PER_YEAR),
        "max_drawdown": worst.depth,
        "max_drawdown_peak": risk.date_text(worst.peak),
        "max_drawdown_trough": risk.date_text(worst.trough),
        "rebalances": test.rebalances,
        "turnover_per_year": float(test.turnover.sum() / years),
        "final_weights": {t: float(w) for t, w in test.weights.iloc[-1].items()},
    }


def stress(
    returns: pd.DataFrame, weights: Mapping[str, float], rule: str, start: date, end: date
) -> dict[str, Any]:
    """A mix invested at its targets at the close of ``start``, followed to ``end``."""
    return _path(backtest(window(returns, start, end), weights, rule))


def _check_comparison(rules: Sequence[str], shuffles: int, seed: int) -> None:
    if len(rules) != 2 or _rule_key(rules[0]) == _rule_key(rules[1]):
        raise ValueError(f"needs two different rules to compare, not {list(rules)!r}")
    if isinstance(shuffles, bool) or not isinstance(shuffles, int) or shuffles < 1:
        raise ValueError(f"shuffles must be a whole number of at least 1, not {shuffles!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a whole number of at least 0, not {seed!r}")


def shuffled_gap(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    rules: Sequence[str],
    shuffles: int,
    seed: int,
    periods_per_year: float,
) -> dict[str, Any]:
    """How much more a year the first rule returns than the second, on the days in their
    order and with the days shuffled ``shuffles`` times.

    A shuffle keeps each day's returns (every fund's together) and the dates where they
    are, so the calendar rules rebalance on the same closes; only which day's returns
    fall on which date changes, which removes any dependence between one day and the
    next. A gap that comes from that dependence stands outside the shuffled ones, which
    centre on zero. The same seed shuffles the days the same ways for any mix.
    """
    _check_comparison(rules, shuffles, seed)

    def gap(frame: pd.DataFrame) -> float:
        first, second = (backtest(frame, weights, rule).returns for rule in rules)
        return float(
            ret.annualised_return(first, periods_per_year)
            - ret.annualised_return(second, periods_per_year)
        )

    real = gap(returns)
    rng = np.random.default_rng(seed)
    values = returns.to_numpy()
    shuffled = np.array(
        [
            gap(pd.DataFrame(values[rng.permutation(len(values))], returns.index, returns.columns))
            for _ in range(shuffles)
        ]
    )
    q05, q95 = np.quantile(shuffled, [0.05, 0.95])
    return {
        "gap": real,
        "shuffled_mean": float(shuffled.mean()),
        "shuffled_q05": float(q05),
        "shuffled_q95": float(q95),
        "shuffled_at_least_gap": int(np.sum(shuffled >= real)),
    }


def _mix(spec: str, tickers: Sequence[str]) -> dict[str, float]:
    """A named mix (any case), or weights such as ``SPY=0.6,AGG=0.4``."""
    name = spec.strip().lower()
    if name in MIXES:
        return parse_weights(MIXES[name], tickers)
    if "=" not in spec:
        raise ValueError(
            f"a mix is {', '.join(MIXES)} or weights such as SPY=0.6,AGG=0.4, not {spec!r}"
        )
    return parse_weights(spec, tickers)


def _period(spec: str) -> tuple[str, date, date]:
    """A named stress period, or ``START:END`` in ISO dates."""
    if spec in STRESS_PERIODS:
        return (spec, *STRESS_PERIODS[spec])
    start, colon, end = spec.partition(":")
    try:
        return spec, date.fromisoformat(start.strip()), date.fromisoformat(end.strip())
    except ValueError:
        raise ValueError(
            f"a period is one of the named ones or START:END, e.g. 2020-02-19:2020-03-23, "
            f"not {spec!r}"
        ) from None


def run(
    prices: pd.DataFrame,
    mixes: Sequence[str] = tuple(MIXES),
    rules: Sequence[str] = RULES,
    periods: Sequence[str] | None = None,
    compare: Sequence[str] = ("daily", "monthly"),
    shuffles: int = 40,
    seed: int = 0,
) -> dict[str, Any]:
    """Backtest every mix under every rule, over the history and each stress period, and
    compare two rules with the days in order and shuffled (none if ``shuffles`` is 0).

    Without ``periods``, the named stress periods that lie within the data; those that
    do not are listed as left out.
    """
    daily = ret.total_returns(prices)
    tickers = [str(t) for t in daily.columns]
    if not mixes or not rules:
        raise ValueError("needs at least one mix and one rule")
    if len(set(mixes)) != len(mixes):
        raise ValueError("a mix is named twice")
    if len({_rule_key(rule) for rule in rules}) != len(rules):
        raise ValueError("a rule is named twice")
    weights = {spec: _mix(spec, tickers) for spec in mixes}
    seen: dict[tuple[float, ...], str] = {}
    for spec, w in weights.items():
        key = tuple(check_weights(w, tickers).tolist())
        if key in seen:
            raise ValueError(f"{seen[key]} and {spec} are the same mix")
        seen[key] = spec
    if shuffles != 0:
        # Checked before the backtests, so a bad argument does not wait for them.
        _check_comparison(compare, shuffles, seed)
    index = pd.DatetimeIndex(daily.index)
    first_close = wide(prices, "close").dropna().index[0]
    if periods is None:
        chosen = [(name, *days) for name, days in STRESS_PERIODS.items()]
        within = [p for p in chosen if p[1] >= first_close.date() and p[2] <= index[-1].date()]
        left_out = [p[0] for p in chosen if p not in within]
    else:
        within = [_period(spec) for spec in periods]
        if len({p[0] for p in within}) != len(within):
            raise ValueError("a period is named twice")
        left_out = []
    # The first close is a start the returns do not include; the windows need it.
    with_start = pd.concat([pd.DataFrame(0.0, index=[first_close], columns=daily.columns), daily])
    per_year = ret.observed_periods_per_year(pd.DatetimeIndex([first_close, *index]))
    history = {
        spec: {rule: summarise(backtest(daily, w, rule), per_year) for rule in rules}
        for spec, w in weights.items()
    }
    stressed = []
    for name, start, end in within:
        funds = window(with_start, start, end)
        stressed.append(
            {
                "period": name,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "days": len(funds),
                "funds": {t: float(np.expm1(np.log1p(funds[t]).sum())) for t in tickers},
                "mixes": {
                    spec: {rule: _path(backtest(funds, w, rule)) for rule in rules}
                    for spec, w in weights.items()
                },
            }
        )
    return {
        "first_return": index[0].date().isoformat(),
        "last_return": index[-1].date().isoformat(),
        "mixes": weights,
        "rules": list(rules),
        "history": history,
        "stress": stressed,
        "stress_periods_left_out": left_out,
        "rebalancing_gap": {
            "rules": list(compare),
            "shuffles": shuffles,
            "seed": seed,
            "mixes": {
                spec: shuffled_gap(daily, w, compare, shuffles, seed, per_year)
                for spec, w in weights.items()
            }
            if shuffles
            else {},
        },
    }


def _print(result: dict[str, Any]) -> None:
    print(f"daily total returns {result['first_return']} to {result['last_return']}")
    for spec, w in result["mixes"].items():
        print(f"  {spec}: " + ", ".join(f"{t} {v:.1%}" for t, v in w.items() if v > 0))
    print(
        f"\n{'':12} {'rule':10} {'compound':>8} {'vol':>7} {'Sharpe':>7} {'max DD':>8} "
        f"{'rebal.':>6} {'turnover/yr':>11}"
    )
    for spec, by_rule in result["history"].items():
        for rule, s in by_rule.items():
            sharpe = "-" if s["sharpe_ratio"] is None else f"{s['sharpe_ratio']:.3f}"
            print(
                f"{spec:12} {rule:10} {s['compound_return']:>8.2%} {s['volatility']:>7.2%} "
                f"{sharpe:>7} {s['max_drawdown']:>8.2%} {s['rebalances']:>6} "
                f"{s['turnover_per_year']:>11.1%}"
            )
    if not result["stress"]:
        print("\nno stress period lies within the data")
    for entry in result["stress"]:
        print(f"\n{entry['period']}, {entry['start']} to {entry['end']} ({entry['days']} days)")
        print("  funds: " + ", ".join(f"{t} {r:.1%}" for t, r in entry["funds"].items()))
        rules = result["rules"]
        print(f"  {'':12}" + "".join(f" {rule:>10}" for rule in rules))
        for spec, by_rule in entry["mixes"].items():
            print(f"  {spec:12}" + "".join(f" {by_rule[r]['return']:>10.2%}" for r in rules))
    if result["stress_periods_left_out"]:
        print("\noutside the data: " + ", ".join(result["stress_periods_left_out"]))
    gaps = result["rebalancing_gap"]
    if gaps["mixes"]:
        first, second = gaps["rules"]
        print(
            f"\n{first} over {second}, compound return a year: with the days in order, "
            f"and over {gaps['shuffles']} shuffles of them (seed {gaps['seed']})"
        )
        for spec, g in gaps["mixes"].items():
            print(
                f"  {spec:12} {g['gap'] * 100:>6.3f} pt; shuffled mean "
                f"{g['shuffled_mean'] * 100:.3f}, 5-95% {g['shuffled_q05'] * 100:.3f} to "
                f"{g['shuffled_q95'] * 100:.3f}, {g['shuffled_at_least_gap']} at least as large"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.scenarios", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--mix",
        action="append",
        help=f"{', '.join(MIXES)}, or weights such as SPY=0.6,AGG=0.4; may be given more "
        "than once (default: every named mix)",
    )
    parser.add_argument(
        "--rule",
        action="append",
        help=f"{', '.join(RULES[:5])} or band:<width>; may be given more than once "
        f"(default: {', '.join(RULES)})",
    )
    parser.add_argument(
        "--period",
        action="append",
        help="a named stress period or START:END; may be given more than once "
        "(default: the named periods within the data)",
    )
    parser.add_argument(
        "--compare",
        default="daily,monthly",
        help="two rules whose gap is compared with the days in order and shuffled "
        "(default: daily,monthly)",
    )
    parser.add_argument(
        "--shuffles", type=int, default=40, help="how many shuffles (default 40; 0 for none)"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the shuffles (default 0)")
    parser.add_argument("--json", type=Path, help="also write the figures to this file")
    args = parser.parse_args(argv)
    try:
        prices = load_cache(args.cache)
        sha256 = read_manifest(args.cache)["sha256"]
        result = {
            "cache_sha256": sha256,
            **run(
                prices,
                args.mix or tuple(MIXES),
                args.rule or RULES,
                args.period,
                [rule.strip() for rule in args.compare.split(",")],
                args.shuffles,
                args.seed,
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
