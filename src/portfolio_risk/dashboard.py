"""The dashboard in ``docs/dashboard-spec.md``, drawn with Plotly from the CSV extracts.

``python -m portfolio_risk.dashboard --extracts extracts --out dashboard.html`` reads only
the files ``portfolio_risk.extracts`` writes, as a Tableau workbook on them would, and
writes one HTML page: the headline figures, then a chart per sheet of the spec. Every
plotted figure is a value from the extracts, unchanged but for costs, drawn in basis
points; the tests compare each chart's values with the files.

``daily.csv`` is built from the price cache, which is not committed. Without it the two
charts over time are left out and the page says so. Every file read must be one the
extracts' ``manifest.json`` lists, with the checksum it gives, so files from two runs are
never drawn together.

By default the page carries Plotly's JavaScript (about 4.8 MB) so it opens offline;
``--cdn`` links it from Plotly's CDN instead.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from portfolio_risk.extracts import COLUMNS, KEYS, MANIFEST, NULLABLE, RESULTS_FILES

PALETTE = (
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#eeca3b", "#9d755d",
)  # fmt: skip
"""Fund colours, in the funds' order, so a fund keeps its colour on every chart."""
PORTFOLIO = "#1f2933"
ZERO, BILLS = "#9da5ad", "#4c78a8"
QUANTILES = ("q1", "q5", "q50", "q95", "q99")
MODELS = ("historical", "normal", "t", "bootstrap", "bootstrap:252")
"""The one-year distributions shown: the data's own, and the models the README compares."""
STRESS_RULE = "monthly"
"""The rule the stress heatmap shows the mixes under, as the README's table does."""
FRONTIER_COMMAND = ("frontier", 1.0, "zero")
CONSTRUCTION_COMMAND = ("construction", 1.0, "zero")
FACTOR_MODEL = "ff5"
_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


class DashboardError(Exception):
    """The extracts cannot be read, or the page cannot be written."""


def _parse(kind: str, text: str):
    if text == "":
        return None
    if kind == "number":
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"not a finite number: {text!r}")
        return value
    if kind == "date":
        # Kept as text, which sorts and plots as a date; YYYY-MM-DD only, as written.
        try:
            if not _DATE.match(text):
                raise ValueError
            dt.date.fromisoformat(text)
        except ValueError:
            raise ValueError(f"not a date as YYYY-MM-DD: {text!r}") from None
    return text


def read_csv(path: Path, name: str) -> list[dict]:
    """One extract, with each column read as the type ``COLUMNS`` gives it."""
    types = COLUMNS[name]
    nullable = NULLABLE.get(name, frozenset())
    keys: dict[tuple, int] = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header != list(types):
                raise DashboardError(
                    f"{path} has columns {header}, expected {list(types)}: "
                    "write it again with portfolio_risk.extracts"
                )
            rows = []
            for line, values in enumerate(reader, 2):
                if len(values) != len(header):
                    raise DashboardError(f"{path} line {line} has {len(values)} fields")
                try:
                    row = {c: _parse(types[c], v) for c, v in zip(header, values, strict=True)}
                except ValueError as exc:
                    raise DashboardError(f"{path} line {line}: {exc}") from exc
                empty = [c for c, v in row.items() if v is None and c not in nullable]
                if empty:
                    raise DashboardError(f"{path} line {line}: {', '.join(empty)} empty")
                key = tuple(row[k] for k in KEYS[name])
                if key in keys:
                    raise DashboardError(
                        f"{path} line {line} repeats line {keys[key]}'s "
                        f"{', '.join(KEYS[name])}: {', '.join(map(str, key))}"
                    )
                keys[key] = line
                rows.append(row)
    except OSError as exc:
        raise DashboardError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise DashboardError(f"{path} is not UTF-8 text: {exc}") from exc
    if not rows:
        raise DashboardError(f"{path} has no rows")
    return rows


def read(folder: Path) -> dict[str, list[dict]]:
    """Every extract in ``folder``: those from the results are required, ``daily.csv``
    is read if it is there."""
    folder = Path(folder)
    if not folder.is_dir():
        raise DashboardError(f"{folder} is not a folder of extracts")
    missing = [n for n in RESULTS_FILES if not (folder / f"{n}.csv").is_file()]
    if missing:
        raise DashboardError(
            f"{folder} has no {', '.join(f'{n}.csv' for n in missing)}: "
            "write the extracts with portfolio_risk.extracts first"
        )
    names = [n for n in COLUMNS if (folder / f"{n}.csv").is_file()]
    check_manifest(folder, [f"{n}.csv" for n in names])
    tables = {n: read_csv(folder / f"{n}.csv", n) for n in names}
    if "daily" in tables:
        check_daily(tables, folder / "daily.csv")
    return tables


def check_manifest(folder: Path, files: list[str]) -> None:
    """Refuse extracts that did not all come from the run that wrote ``manifest.json``:
    a file it does not list (a ``daily.csv`` an earlier run left beside a ``--no-prices``
    run), a file changed since, or a file it lists that is missing."""
    path = folder / MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DashboardError(
            f"{folder} has no {MANIFEST}: write the extracts with portfolio_risk.extracts"
        ) from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DashboardError(f"{path} cannot be read: {exc}") from None
    listed = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(listed, dict) or not all(
        isinstance(e, dict) and isinstance(e.get("sha256"), str) for e in listed.values()
    ):
        raise DashboardError(f"{path} is not an extracts manifest (no files with checksums)")
    for file in files:
        if file not in listed:
            raise DashboardError(
                f"{folder / file} is not from the run that wrote {MANIFEST} (an earlier "
                "run's file?): write the extracts again, or remove it"
            )
        if hashlib.sha256((folder / file).read_bytes()).hexdigest() != listed[file]["sha256"]:
            raise DashboardError(
                f"{folder / file} has changed since the extracts were written: write them again"
            )
    missing = sorted(set(listed) - set(files))
    if missing:
        raise DashboardError(f"{path} lists {', '.join(missing)}, which {folder} does not have")


def check_daily(tables: Mapping[str, list[dict]], path: Path) -> None:
    """Refuse a ``daily.csv`` that is not of the risk measures' series and days, such as
    one left in the folder by an earlier run over another window (``--no-prices`` does
    not write it)."""
    zero = _where(tables["risk_metrics"], risk_free="zero")
    if not zero:
        raise DashboardError("risk_metrics.csv has no rows at the zero rate, which daily.csv is of")
    series = [r["series"] for r in zero]
    days = {r["date"] for r in tables["daily"]}
    found = list(dict.fromkeys(r["series"] for r in tables["daily"]))
    first, last = zero[0]["first_return"], zero[0]["last_return"]
    if found != series or min(days) != first or max(days) != last:
        raise DashboardError(
            f"{path} holds {', '.join(found)} from {min(days)} to {max(days)}, but the risk "
            f"measures are of {', '.join(series)} from {first} to {last}: write the "
            "extracts again with the price cache, or remove daily.csv"
        )
    for s in series:
        if sum(r["series"] == s for r in tables["daily"]) != len(days):
            raise DashboardError(f"{path} does not have every day for {s}")


def _where(rows: list[dict], **match) -> list[dict]:
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def _one(rows: list[dict], **match) -> dict:
    found = _where(rows, **match)
    if len(found) != 1:
        described = ", ".join(f"{k} {v}" for k, v in match.items())
        raise DashboardError(f"expected one row for {described}, found {len(found)}")
    return found[0]


def _fmt(value, pattern: str) -> str:
    return "n/a" if value is None else pattern.format(value)


def funds_of(tables: Mapping[str, list[dict]]) -> list[str]:
    """The funds, in the order the risk measures list them."""
    rows = _where(tables["risk_metrics"], risk_free="zero")
    return [r["series"] for r in rows if r["series"] != "portfolio"]


def tiles(tables: Mapping[str, list[dict]]) -> list[tuple[str, str]]:
    """The headline figures of the portfolio the risk command measures."""
    zero = _one(tables["risk_metrics"], series="portfolio", risk_free="zero")
    bills = _one(tables["risk_metrics"], series="portfolio", risk_free="bills")
    tail = _one(tables["tail_risk"], series="portfolio", level=0.99, method="historical")
    return [
        ("Compound return a year", _fmt(zero["compound_return"], "{:.2%}")),
        ("Volatility a year", _fmt(zero["volatility"], "{:.2%}")),
        ("Sharpe ratio, zero rate", _fmt(zero["sharpe_ratio"], "{:.2f}")),
        ("Sharpe ratio, against the bill", _fmt(bills["sharpe_ratio"], "{:.2f}")),
        ("Largest drawdown", _fmt(zero["max_drawdown"], "{:.1%}")),
        ("One-day 99% expected shortfall", _fmt(tail["expected_shortfall"], "{:.2%}")),
    ]


def figures(tables: Mapping[str, list[dict]]) -> list[tuple[str, go.Figure]]:
    """One figure per sheet of the spec, in its order."""
    out = []
    funds = funds_of(tables)
    colour = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(funds)} | {"portfolio": PORTFOLIO}
    series = [*funds, "portfolio"]

    if "daily" in tables:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06,
            subplot_titles=("Growth of $1 (log scale)", "Drawdown from the highest point"),
        )  # fmt: skip
        for s in series:
            rows = _where(tables["daily"], series=s)
            line = dict(color=colour[s], width=2.5 if s == "portfolio" else 1.2)
            dates = [r["date"] for r in rows]
            fig.add_scatter(x=dates, y=[r["wealth"] for r in rows], name=s, line=line,
                            legendgroup=s, row=1, col=1)  # fmt: skip
            fig.add_scatter(x=dates, y=[r["drawdown"] for r in rows], name=s, line=line,
                            legendgroup=s, showlegend=False, row=2, col=1)  # fmt: skip
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        fig.update_layout(height=720, legend=dict(orientation="h", y=-0.08))
        out.append(("Each fund and the portfolio over time, total return", fig))

    fig = go.Figure()
    for cap, dash in ((1.0, "solid"), (0.3, "dash")):
        points = _where(tables["frontier"], max_weight=cap)
        fig.add_scatter(x=[p["volatility"] for p in points],
                        y=[p["expected_return"] for p in points], mode="lines",
                        name=f"frontier, at most {cap:.0%} a fund", line=dict(dash=dash,
                        color=PORTFOLIO))  # fmt: skip
    named = _where(tables["portfolios"], max_weight=1.0)
    for kind, symbol in (("fund", "circle"), ("portfolio", "diamond")):
        rows = [r for r in named if r["kind"] == kind]
        fig.add_scatter(x=[r["volatility"] for r in rows], y=[r["expected_return"] for r in rows],
                        mode="markers+text", text=[r["name"] for r in rows],
                        textposition="top center", name=f"{kind}s", marker=dict(symbol=symbol,
                        size=10, color=[colour.get(r["name"], BILLS) for r in rows]))  # fmt: skip
    fig.update_layout(xaxis=dict(title="volatility a year", tickformat=".0%"),
                      yaxis=dict(title="expected return a year (sample mean)",
                                 tickformat=".0%"))  # fmt: skip
    out.append(("Efficient frontier over the whole history, zero rate (in sample)", fig))

    fig = go.Figure()
    for rate, bar_colour, label in (("zero", ZERO, "zero rate"),
                                    ("bills", BILLS, "against the one-month bill")):  # fmt: skip
        rows = [_one(tables["risk_metrics"], series=s, risk_free=rate) for s in series]
        fig.add_bar(x=series, y=[r["sharpe_ratio"] for r in rows], name=label,
                    marker_color=bar_colour,
                    text=[_fmt(r["sharpe_ratio"], "{:.2f}") for r in rows])  # fmt: skip
    fig.update_layout(barmode="group", yaxis_title="Sharpe ratio, daily returns, a year",
                      legend=dict(orientation="h", y=-0.2))  # fmt: skip
    out.append(("Sharpe ratio at a zero rate and against the Treasury bill", fig))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Value at risk", "Expected shortfall"))
    for method, bar_colour in (("historical", PORTFOLIO), ("normal", ZERO)):
        rows = [_one(tables["tail_risk"], series=s, level=0.99, method=method) for s in series]
        for col, measure in ((1, "value_at_risk"), (2, "expected_shortfall")):
            fig.add_bar(x=series, y=[r[measure] for r in rows], name=method,
                        marker_color=bar_colour, showlegend=col == 1, row=1,
                        col=col)  # fmt: skip
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.2))
    out.append(("One-day loss at 99%: the data against a normal fit", fig))

    fig = go.Figure()
    for source in MODELS:
        r = _one(tables["montecarlo"], source=source, scope="horizon")
        fig.add_box(
            x=[source], name=source, q1=[r["q5"]], median=[r["q50"]], q3=[r["q95"]],
            lowerfence=[r["q1"]], upperfence=[r["q99"]], mean=[r["mean"]],
            marker_color=PORTFOLIO if source == "historical" else BILLS,
        )  # fmt: skip
    fig.update_layout(showlegend=False, yaxis=dict(title="one-year return", tickformat=".0%"))
    out.append(
        ("One-year return: every 252-day window of the data against 10,000 simulated paths "
         "(box 5% to 95%, whiskers 1% to 99%)", fig)
    )  # fmt: skip

    stress = tables["stress"]
    periods = list(dict.fromkeys(r["period"] for r in stress))
    holdings = [*funds, *dict.fromkeys(r["holding"] for r in stress if r["kind"] == "mix")]

    def stress_value(period: str, holding: str):
        rule = None if holding in funds else STRESS_RULE
        return _one(stress, period=period, holding=holding, rule=rule)["return"]

    z = [[stress_value(p, h) for h in holdings] for p in periods]
    fig = go.Figure(go.Heatmap(
        x=holdings, y=periods, z=z, zmid=0, colorscale="RdBu", colorbar=dict(tickformat=".0%"),
        text=[[f"{v:.1%}" for v in row] for row in z], texttemplate="%{text}",
    ))  # fmt: skip
    fig.update_layout(yaxis=dict(autorange="reversed"))
    out.append(
        (f"Stress periods, each an S&P 500 fall from its closing high to its closing low: funds "
         f"bought and held, mixes rebalanced {STRESS_RULE}", fig)
    )  # fmt: skip

    costs = tables["trading_costs"]
    mixes = list(dict.fromkeys(r["mix"] for r in costs))
    rules = list(dict.fromkeys(r["rule"] for r in costs))
    fig = make_subplots(rows=1, cols=len(mixes), subplot_titles=mixes, shared_yaxes=False)
    for col, mix in enumerate(mixes, 1):
        for i, rule in enumerate(rules):
            rows = _where(costs, mix=mix, rule=rule)
            fig.add_scatter(x=[r["cost"] * 1e4 for r in rows],
                            y=[r["net_compound_return"] for r in rows], name=rule,
                            legendgroup=rule, showlegend=col == 1, mode="lines+markers",
                            line=dict(color=PALETTE[i % len(PALETTE)]), row=1,
                            col=col)  # fmt: skip
    fig.update_xaxes(title_text="one-way cost, bp")
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    out.append(("Compound return a year after trading costs, by rebalancing rule", fig))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Mean-variance (frontier command)",
                                                        "Construction methods"))  # fmt: skip
    # One colour and one legend entry per portfolio, which may appear in both panels.
    shown: dict[str, str] = {}
    for col, (command, cap, rate) in enumerate((FRONTIER_COMMAND, CONSTRUCTION_COMMAND), 1):
        rows = _where(tables["held_out"], command=command, max_weight=cap, risk_free=rate)
        for portfolio in dict.fromkeys(r["portfolio"] for r in rows):
            first = portfolio not in shown
            line_colour = shown.setdefault(portfolio, PALETTE[len(shown) % len(PALETTE)])
            these = [r for r in rows if r["portfolio"] == portfolio]
            fig.add_scatter(x=[r["split"] for r in these],
                            y=[r["held_sharpe_ratio"] for r in these], name=portfolio,
                            legendgroup=portfolio, showlegend=first, mode="lines+markers",
                            line=dict(color=line_colour), row=1, col=col)  # fmt: skip
    fig.update_xaxes(title_text="split: chosen on the days to it, held after it")
    fig.update_yaxes(title_text="Sharpe ratio held, before costs", col=1)
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    out.append(("Held out: each portfolio's Sharpe ratio after the day it was chosen", fig))

    rows = _where(tables["factors"], model=FACTOR_MODEL)
    terms = list(dict.fromkeys(r["term"] for r in rows))
    z = [[_one(rows, series=s, term=t)["estimate"] for t in terms] for s in series]
    fig = go.Figure(go.Heatmap(
        x=terms, y=series, z=z, zmid=0, colorscale="RdBu",
        text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}",
    ))  # fmt: skip
    fig.update_layout(yaxis=dict(autorange="reversed"))
    out.append(
        ("Fama-French five-factor loadings (alpha a year; factor betas)", fig)
    )  # fmt: skip
    return out


def page(tables: Mapping[str, list[dict]], *, cdn: bool = False) -> str:
    """The whole HTML page."""
    zero = _one(tables["risk_metrics"], series="portfolio", risk_free="zero")
    funds = funds_of(tables)
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Portfolio risk dashboard</title><style>",
        "body{font-family:system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px;"
        "color:#1f2933;background:#fff}",
        ".note{background:#fff8e1;border-left:4px solid #f0b400;padding:8px 12px;margin:8px 0}",
        ".tiles{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}",
        ".tile{border:1px solid #d9e2ec;border-radius:6px;padding:10px 14px;min-width:140px}",
        ".tile b{display:block;font-size:1.4em}h2{margin-top:32px;font-size:1.15em}",
        "</style></head><body>",
        "<h1>Portfolio risk dashboard</h1>",
        f"<p>Daily total returns of {html.escape(', '.join(funds))}, "
        f"{zero['first_return']} to {zero['last_return']}; the portfolio holds them in "
        f"equal parts, rebalanced at every close. Drawn from the CSV extracts with Plotly "
        f"{html.escape(plotly.__version__)}, as specified in docs/dashboard-spec.md.</p>",
        "<p class='note'><b>The prices are real</b> (Yahoo Finance, closes with dividends "
        "and splits), and every figure is measured on them. Expected returns on the frontier "
        "are sample means, so the frontier is in sample; the held-out chart is how the "
        "portfolios did after the day they were chosen.</p>",
        "<div class='tiles'>",
        *(
            f"<div class='tile'>{html.escape(t)}<b>{html.escape(v)}</b></div>"
            for t, v in tiles(tables)
        ),  # fmt: skip
        "</div>",
    ]
    if "daily" not in tables:
        parts.append(
            "<p class='note'>No daily.csv in the extracts (it is built from the price "
            "cache), so the charts over time are left out.</p>"
        )
    first = True
    for title, fig in figures(tables):
        fig.update_layout(template="plotly_white", margin=dict(t=60, l=60, r=20, b=60))
        if fig.layout.height is None:
            fig.update_layout(height=460)
        parts.append(f"<h2>{html.escape(title)}</h2>")
        include = ("cdn" if cdn else True) if first else False
        parts.append(fig.to_html(full_html=False, include_plotlyjs=include,
                                 config={"displaylogo": False}))  # fmt: skip
        first = False
    parts.append("</body></html>")
    return "\n".join(parts)


def write(tables: Mapping[str, Any], path: Path, *, cdn: bool = False) -> int:
    """Write the page to ``path``; its size in bytes."""
    path = Path(path)
    text = page(tables, cdn=cdn)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise DashboardError(f"cannot write {path}: {exc.strerror or exc}") from exc
    return len(text.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.dashboard",
        description="Draw the dashboard from the CSV extracts as one HTML page.",
    )
    parser.add_argument("--extracts", type=Path, required=True,
                        help="the folder portfolio_risk.extracts wrote")  # fmt: skip
    parser.add_argument("--out", type=Path, required=True, help="the HTML file to write")
    parser.add_argument("--cdn", action="store_true",
                        help="link Plotly's JavaScript from its CDN, not include it")  # fmt: skip
    args = parser.parse_args(argv)
    if args.out.is_dir():
        parser.error(f"--out {args.out} is a folder; name the HTML file to write")
    try:
        size = write(read(args.extracts), args.out, cdn=args.cdn)
    except DashboardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
