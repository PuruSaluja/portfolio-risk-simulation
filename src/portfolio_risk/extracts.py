"""Flat CSV extracts of the committed results, for Tableau and the Plotly dashboard.

``python -m portfolio_risk.extracts --out extracts`` reads the results files in
``results/`` (which the commands' ``--json`` options write) and writes one file per table
below. Each has one header row, dates as ISO ``YYYY-MM-DD``, numbers as Python writes a
float (the shortest text that reads back as the same number) and an empty field where a
figure is undefined, which Tableau's text connector reads as typed columns. Every figure
is copied from a results file, never recomputed, except ``daily.csv``.

``daily.csv`` (from the price cache)
    One row per trading day and series (each fund, and ``portfolio``: the risk
    command's weights rebalanced at every close): the day's total return, the wealth
    index after it (1 before the first return) and its drawdown from the highest point
    so far. Built from Yahoo's prices, so it is written locally and never committed.
    The cache must be the one the results were computed from (its checksum is in every
    results file); ``--no-prices`` leaves the file out.
``risk_metrics.csv``
    One row per series and rate: ``risk-metrics.json`` at a zero rate and
    ``risk-metrics-bills.json`` against the one-month Treasury bill, with the days each
    covers.
``tail_risk.csv``
    One row per series, level (0.95, 0.99) and method (historical, normal): one-day
    value at risk and expected shortfall, losses as positive numbers.
``montecarlo.csv``
    One row per source (``historical`` or a model) and scope (``daily``, or
    ``horizon``: the one-year return), from ``montecarlo.json``.
``stress.csv``
    One row per stress period and holding: each fund bought and held, and each mix
    under each rebalancing rule, from ``scenarios.json``.
``rebalancing.csv``
    One row per mix and rule over the whole history, before costs.
``trading_costs.csv``
    One row per mix, rule and one-way cost: the compound return after costs and the
    drag, from ``costs.json`` (assumed cost levels, not measured spreads).
``frontier.csv`` and ``frontier_weights.csv``
    Each efficient frontier's points (uncapped and capped at 30% a fund, zero rate),
    then their weights, one row per point and fund.
``portfolios.csv``
    Each fund and each named portfolio of the frontier files at the points it sits on,
    to plot beside the frontier.
``held_out.csv``
    One row per command, cap, rate, split and portfolio: the ratio the portfolio was
    chosen on before the split, and how it did held after the split before costs.
``held_costs.csv``
    The same portfolios held after the split under each rule and one-way cost.
``factors.csv``
    One row per series, model (ff3, ff5) and term (alpha and each factor): estimate,
    OLS and Newey-West t statistics, with the regression's R squared.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_risk import returns as ret
from portfolio_risk.data import DEFAULT_CACHE, DataError, load_cache, read_manifest
from portfolio_risk.montecarlo import historical_portfolio_returns

DEFAULT_RESULTS = Path("results")
DEFAULT_OUT = Path("extracts")

# The results files read, by what each extract calls them.
RESULTS = {
    "risk": "risk-metrics.json",
    "risk_bills": "risk-metrics-bills.json",
    "montecarlo": "montecarlo.json",
    "scenarios": "scenarios.json",
    "costs": "costs.json",
    "frontier": "frontier.json",
    "frontier_bills": "frontier-bills.json",
    "frontier_cap30": "frontier-cap30.json",
    "frontier_cap30_bills": "frontier-cap30-bills.json",
    "construction": "construction.json",
    "construction_bills": "construction-bills.json",
    "factors": "factors.json",
}
# (command, results name) for the held-out and frontier extracts.
HELD = (
    ("frontier", "frontier"),
    ("frontier", "frontier_bills"),
    ("frontier", "frontier_cap30"),
    ("frontier", "frontier_cap30_bills"),
    ("construction", "construction"),
    ("construction", "construction_bills"),
)
FRONTIERS = ("frontier", "frontier_cap30")
STATISTICS = (
    "count", "mean", "sd", "skewness", "excess_kurtosis", "probability_of_loss",
    "q0.1", "q1", "q5", "q50", "q95", "q99",
)  # fmt: skip

# Every file's columns, with the type the dashboard reads each as.
COLUMNS: dict[str, dict[str, str]] = {
    "daily": {
        "date": "date", "series": "text", "daily_return": "number", "wealth": "number",
        "drawdown": "number",
    },
    "risk_metrics": {
        "series": "text", "risk_free": "text", "first_return": "date", "last_return": "date",
        "compound_return": "number", "volatility": "number", "sharpe_ratio": "number",
        "sharpe_ratio_monthly": "number", "downside_deviation": "number",
        "sortino_ratio": "number", "max_drawdown": "number", "max_drawdown_peak": "date",
        "max_drawdown_trough": "date", "max_drawdown_recovery": "date",
    },
    "tail_risk": {
        "series": "text", "level": "number", "method": "text", "value_at_risk": "number",
        "expected_shortfall": "number",
    },
    "montecarlo": {
        "source": "text", "scope": "text",
        **{s.replace(".", "_"): "number" for s in STATISTICS},
    },
    "stress": {
        "period": "text", "start": "date", "end": "date", "days": "number",
        "holding": "text", "kind": "text", "rule": "text", "return": "number",
        "max_drawdown": "number",
    },
    "rebalancing": {
        "mix": "text", "rule": "text", "compound_return": "number", "volatility": "number",
        "sharpe_ratio": "number", "max_drawdown": "number", "rebalances": "number",
        "turnover_per_year": "number",
    },
    "trading_costs": {
        "mix": "text", "rule": "text", "cost": "number", "net_compound_return": "number",
        "drag": "number", "paid_per_year": "number",
    },
    "frontier": {
        "max_weight": "number", "point": "number", "expected_return": "number",
        "volatility": "number", "sharpe_ratio": "number",
    },
    "frontier_weights": {
        "max_weight": "number", "point": "number", "fund": "text", "weight": "number",
    },
    "portfolios": {
        "max_weight": "number", "name": "text", "kind": "text", "expected_return": "number",
        "volatility": "number", "sharpe_ratio": "number",
    },
    "held_out": {
        "command": "text", "max_weight": "number", "risk_free": "text", "split": "date",
        "portfolio": "text", "chosen_sharpe_ratio": "number", "held_sharpe_ratio": "number",
        "held_volatility": "number", "held_compound_return": "number",
    },
    "held_costs": {
        "command": "text", "max_weight": "number", "risk_free": "text", "split": "date",
        "portfolio": "text", "rule": "text", "cost": "number", "compound_return": "number",
        "sharpe_ratio": "number",
    },
    "factors": {
        "series": "text", "model": "text", "term": "text", "estimate": "number",
        "ols_t": "number", "newey_west_t": "number", "r_squared": "number",
    },
}  # fmt: skip
# Columns that may be empty: a figure the results file holds as null.
NULLABLE = {
    "risk_metrics": frozenset(
        {"sharpe_ratio", "sharpe_ratio_monthly", "sortino_ratio", "max_drawdown_peak",
         "max_drawdown_trough", "max_drawdown_recovery"}
    ),
    "stress": frozenset({"rule", "max_drawdown"}),
    "rebalancing": frozenset({"sharpe_ratio"}),
    "portfolios": frozenset({"sharpe_ratio"}),
    "held_out": frozenset({"chosen_sharpe_ratio", "held_sharpe_ratio"}),
    "held_costs": frozenset({"sharpe_ratio"}),
}  # fmt: skip
# The columns that name one row; no two rows of a file share them.
KEYS = {
    "daily": ("date", "series"),
    "risk_metrics": ("series", "risk_free"),
    "tail_risk": ("series", "level", "method"),
    "montecarlo": ("source", "scope"),
    "stress": ("period", "holding", "rule"),
    "rebalancing": ("mix", "rule"),
    "trading_costs": ("mix", "rule", "cost"),
    "frontier": ("max_weight", "point"),
    "frontier_weights": ("max_weight", "point", "fund"),
    "portfolios": ("max_weight", "name"),
    "held_out": ("command", "max_weight", "risk_free", "split", "portfolio"),
    "held_costs": ("command", "max_weight", "risk_free", "split", "portfolio", "rule", "cost"),
    "factors": ("series", "model", "term"),
}
RESULTS_FILES = tuple(n for n in COLUMNS if n != "daily")


class ExtractError(Exception):
    """The results cannot be read as the extracts need them, or a file cannot be written."""


def read_results(folder: Path) -> dict[str, dict[str, Any]]:
    """Every results file the extracts need, checked to come from one price cache."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ExtractError(f"{folder} is not a folder of results")
    results = {}
    for name, file in RESULTS.items():
        path = folder / file
        try:
            results[name] = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ExtractError(
                f"{path} is missing: write it with the command the README gives for it"
            ) from None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExtractError(f"{path} cannot be read: {exc}") from None
        if not isinstance(results[name], dict) or "cache_sha256" not in results[name]:
            raise ExtractError(f"{path} is not a results file (no cache_sha256)")
    caches = {r["cache_sha256"] for r in results.values()}
    if len(caches) != 1:
        raise ExtractError(
            f"the files in {folder} come from {len(caches)} different price caches; "
            "regenerate them from one"
        )
    return results


def risk_metrics(results: Mapping[str, Any]) -> list[dict]:
    rows = []
    for name, rate in (("risk", "zero"), ("risk_bills", "bills")):
        result = results[name]
        for series, m in result["series"].items():
            drawdown = m["max_drawdown"]
            rows.append({
                "series": series, "risk_free": rate,
                "first_return": result["first_return"], "last_return": result["last_return"],
                "compound_return": m["compound_return"], "volatility": m["volatility"],
                "sharpe_ratio": m["sharpe_ratio"],
                "sharpe_ratio_monthly": m["sharpe_ratio_monthly"],
                "downside_deviation": m["downside_deviation"],
                "sortino_ratio": m["sortino_ratio"], "max_drawdown": drawdown["depth"],
                "max_drawdown_peak": drawdown["peak"], "max_drawdown_trough": drawdown["trough"],
                "max_drawdown_recovery": drawdown["recovery"],
            })  # fmt: skip
    return rows


def tail_risk(results: Mapping[str, Any]) -> list[dict]:
    rows = []
    for series, m in results["risk"]["series"].items():
        for level, methods in m["tail"].items():
            for method, tail in methods.items():
                rows.append({
                    "series": series, "level": int(level) / 100, "method": method,
                    "value_at_risk": tail["value_at_risk"],
                    "expected_shortfall": tail["expected_shortfall"],
                })  # fmt: skip
    return rows


def montecarlo(results: Mapping[str, Any]) -> list[dict]:
    result = results["montecarlo"]
    sources = [("historical", result["historical"])]
    sources += [(m["model"], m) for m in result["models"]]
    rows = []
    for source, figures in sources:
        for scope in ("daily", "horizon"):
            stats = figures[scope]
            row = {"source": source, "scope": scope}
            row |= {s.replace(".", "_"): stats[s] for s in STATISTICS}
            rows.append(row)
    return rows


def stress(results: Mapping[str, Any]) -> list[dict]:
    rows = []
    for period in results["scenarios"]["stress"]:
        common = {
            "period": period["period"], "start": period["start"], "end": period["end"],
            "days": period["days"],
        }  # fmt: skip
        for fund, value in period["funds"].items():
            rows.append(
                common | {"holding": fund, "kind": "fund", "rule": None, "return": value,
                          "max_drawdown": None}
            )  # fmt: skip
        for mix, rules in period["mixes"].items():
            for rule, figures in rules.items():
                rows.append(
                    common | {"holding": mix, "kind": "mix", "rule": rule,
                              "return": figures["return"],
                              "max_drawdown": figures["max_drawdown"]}
                )  # fmt: skip
    return rows


def rebalancing(results: Mapping[str, Any]) -> list[dict]:
    rows = []
    for mix, rules in results["scenarios"]["history"].items():
        for rule, m in rules.items():
            rows.append({
                "mix": mix, "rule": rule, "compound_return": m["compound_return"],
                "volatility": m["volatility"], "sharpe_ratio": m["sharpe_ratio"],
                "max_drawdown": m["max_drawdown"], "rebalances": m["rebalances"],
                "turnover_per_year": m["turnover_per_year"],
            })  # fmt: skip
    return rows


def trading_costs(results: Mapping[str, Any]) -> list[dict]:
    rows = []
    for mix, rules in results["costs"]["history"].items():
        for rule, m in rules.items():
            for c in m["costs"]:
                rows.append({
                    "mix": mix, "rule": rule, "cost": c["cost"],
                    "net_compound_return": c["net_compound_return"], "drag": c["drag"],
                    "paid_per_year": c["paid_per_year"],
                })  # fmt: skip
    return rows


def frontier(results: Mapping[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    """The frontier points, their weights, and the funds and named portfolios."""
    points, weights, named = [], [], []
    for name in FRONTIERS:
        result = results[name]
        cap = result["max_weight"]
        for i, p in enumerate(result["frontier"]):
            points.append({
                "max_weight": cap, "point": i, "expected_return": p["expected_return"],
                "volatility": p["volatility"], "sharpe_ratio": p["sharpe_ratio"],
            })  # fmt: skip
            weights += [
                {"max_weight": cap, "point": i, "fund": f, "weight": w}
                for f, w in p["weights"].items()
            ]
        for fund, f in result["funds"].items():
            named.append({
                "max_weight": cap, "name": fund, "kind": "fund",
                "expected_return": f["expected_return"], "volatility": f["volatility"],
                "sharpe_ratio": f.get("sharpe_ratio"),
            })  # fmt: skip
        for portfolio, p in result["portfolios"].items():
            named.append({
                "max_weight": cap, "name": portfolio, "kind": "portfolio",
                "expected_return": p["expected_return"], "volatility": p["volatility"],
                "sharpe_ratio": p["sharpe_ratio"],
            })  # fmt: skip
    return points, weights, named


def held_out(results: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    """Each split's portfolios held after it, before costs and after them."""
    rows, costs = [], []
    for command, name in HELD:
        result = results[name]
        rate = "bills" if name.endswith("_bills") else "zero"
        for split in result["splits"]:
            for portfolio, p in split["portfolios"].items():
                if command == "frontier":
                    chosen, held = p["sharpe_ratio"], p["realised"]
                else:
                    chosen, held = p["in_sample"]["sharpe_ratio"], p["held"]
                key = {
                    "command": command, "max_weight": result["max_weight"], "risk_free": rate,
                    "split": split["date"], "portfolio": portfolio,
                }  # fmt: skip
                rows.append(
                    key | {"chosen_sharpe_ratio": chosen,
                           "held_sharpe_ratio": held["sharpe_ratio"],
                           "held_volatility": held["volatility"],
                           "held_compound_return": held["compound_return"]}
                )  # fmt: skip
                for rule, r in held["costs"].items():
                    costs += [
                        key | {"rule": rule, "cost": c["cost"],
                               "compound_return": c["compound_return"],
                               "sharpe_ratio": c["sharpe_ratio"]}
                        for c in r["costs"]
                    ]  # fmt: skip
    return rows, costs


def factors(results: Mapping[str, Any]) -> list[dict]:
    rows = []
    for series, models in results["factors"]["series"].items():
        for model, fit in models.items():
            for term, c in fit["coefficients"].items():
                rows.append({
                    "series": series, "model": model, "term": term,
                    "estimate": c["estimate"], "ols_t": c["ols_t"],
                    "newey_west_t": c["newey_west_t"], "r_squared": fit["r_squared"],
                })  # fmt: skip
    return rows


def _read(name: str, extract, files: Sequence[str], results: Mapping[str, Any]):
    """``extract(results)``, naming the results files when one lacks what it reads."""
    try:
        return extract(results)
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
        named = " and ".join(RESULTS[f] for f in files)
        raise ExtractError(
            f"{named} do not hold what the {name} extract reads "
            f"({type(exc).__name__}: {exc}); write them again with the README's commands"
        ) from None


def from_results(results: Mapping[str, Any]) -> dict[str, list[dict]]:
    """Every extract that comes from the results files alone."""
    points, weights, named = _read("frontier", frontier, FRONTIERS, results)
    held, held_costs = _read("held_out", held_out, [n for _, n in HELD], results)
    tables = {
        "risk_metrics": _read("risk_metrics", risk_metrics, ("risk", "risk_bills"), results),
        "tail_risk": _read("tail_risk", tail_risk, ("risk",), results),
        "montecarlo": _read("montecarlo", montecarlo, ("montecarlo",), results),
        "stress": _read("stress", stress, ("scenarios",), results),
        "rebalancing": _read("rebalancing", rebalancing, ("scenarios",), results),
        "trading_costs": _read("trading_costs", trading_costs, ("costs",), results),
        "frontier": points,
        "frontier_weights": weights,
        "portfolios": named,
        "held_out": held,
        "held_costs": held_costs,
        "factors": _read("factors", factors, ("factors",), results),
    }
    assert tuple(tables) == RESULTS_FILES
    return tables


def daily(prices: pd.DataFrame, weights: Mapping[str, float]) -> list[dict]:
    """Each fund's and the portfolio's daily return, wealth index and drawdown."""
    funds = ret.total_returns(prices)
    series = {str(t): funds[t] for t in funds.columns}
    series["portfolio"] = historical_portfolio_returns(funds, weights)
    rows = []
    for name, values in series.items():
        wealth = (1.0 + values).cumprod()
        # The running high includes the starting wealth of 1, as risk.drawdowns has it.
        drawdown = wealth / wealth.cummax().clip(lower=1.0) - 1.0
        for day, r, w, d in zip(values.index, values, wealth, drawdown, strict=True):
            rows.append({
                "date": day.date().isoformat(), "series": name, "daily_return": float(r),
                "wealth": float(w), "drawdown": float(d),
            })  # fmt: skip
    return rows


def _text(kind: str, value: Any) -> str:
    if value is None:
        return ""
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ExtractError(f"not a number: {value!r}")
        if not math.isfinite(value):
            raise ExtractError(f"not a finite number: {value!r}")
        return repr(value) if isinstance(value, float) else str(value)
    if not isinstance(value, str):
        raise ExtractError(f"not text: {value!r}")
    return value


def check(name: str, rows: Iterable[Mapping[str, Any]]) -> list[list[str]]:
    """``rows`` as text in ``COLUMNS[name]``'s order: every column there, no other, no
    empty field a figure needs, and no two rows naming the same thing."""
    columns = COLUMNS[name]
    nullable = NULLABLE.get(name, frozenset())
    seen: set[tuple] = set()
    lines = []
    for row in rows:
        if set(row) != set(columns):
            raise ExtractError(f"{name}: a row has columns {sorted(row)}, not {sorted(columns)}")
        empty = [c for c in columns if row[c] is None and c not in nullable]
        if empty:
            raise ExtractError(f"{name}: {', '.join(empty)} empty in {dict(row)}")
        key = tuple(row[k] for k in KEYS[name])
        if key in seen:
            raise ExtractError(f"{name}: two rows for {', '.join(map(str, key))}")
        seen.add(key)
        try:
            lines.append([_text(columns[c], row[c]) for c in columns])
        except ExtractError as exc:
            raise ExtractError(f"{name}: {exc} in {dict(row)}") from None
    if not lines:
        raise ExtractError(f"{name}: no rows")
    return lines


def write(folder: Path, tables: Mapping[str, list[dict]]) -> list[Path]:
    """Every table checked first, then written as ``<name>.csv`` (each file replaced whole)."""
    checked = {name: check(name, rows) for name, rows in tables.items()}
    folder = Path(folder)
    if folder.exists() and not folder.is_dir():
        raise ExtractError(f"{folder} is a file; --out takes a folder")
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, lines in checked.items():
        path = folder / f"{name}.csv"
        partial = path.with_name(path.name + ".partial")
        with open(partial, "w", encoding="utf-8", newline="") as fh:
            out = csv.writer(fh, lineterminator="\n")
            out.writerow(COLUMNS[name])
            out.writerows(lines)
        os.replace(partial, path)
        paths.append(path)
    return paths


def build(results_folder: Path, cache: Path | None = DEFAULT_CACHE) -> dict[str, list[dict]]:
    """The extracts, from the results files and, unless ``cache`` is None, the prices."""
    results = read_results(results_folder)
    tables = from_results(results)
    if cache is not None:
        cache_sha = read_manifest(cache)["sha256"]
        results_sha = results["risk"]["cache_sha256"]
        if cache_sha != results_sha:
            raise ExtractError(
                f"{cache} is not the cache the results were computed from (checksum "
                f"{cache_sha[:12]}..., the results' {results_sha[:12]}...); fetch the "
                "README's window or pass --no-prices"
            )
        prices = load_cache(cache)
        rows = _read("daily", lambda r: daily(prices, r["risk"]["weights"]), ("risk",), results)
        tables = {"daily": rows} | tables
    return tables


def check_out(folder: Path) -> None:
    """Refuse an ``--out`` that cannot be a folder, before any work: a file, or a path
    under a file."""
    folder = Path(folder)
    if folder.exists() and not folder.is_dir():
        raise ExtractError(f"{folder} is a file; --out takes a folder")
    for parent in folder.parents:
        if parent.exists():
            if not parent.is_dir():
                raise ExtractError(f"{folder} cannot be written: {parent} is a file")
            break


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.extracts",
        description="Flat CSV extracts of the results files, for Tableau and the dashboard",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--no-prices", action="store_true", help="leave out daily.csv, the one file from prices"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        check_out(args.out)
        tables = build(args.results, None if args.no_prices else args.cache)
        paths = write(args.out, tables)
    except (ExtractError, DataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path, rows in zip(paths, tables.values(), strict=True):
        print(f"{path}  {len(rows):,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
