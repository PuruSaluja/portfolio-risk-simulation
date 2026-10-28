import hashlib
import json
from pathlib import Path

import pytest

from portfolio_risk import dashboard
from portfolio_risk.dashboard import DashboardError, figures, main, page, read, tiles
from portfolio_risk.extracts import COLUMNS, RESULTS_FILES
from portfolio_risk.extracts import main as write_extracts

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("extracts")
    argv = ["--results", str(ROOT / "results"), "--no-prices", "--out", str(out)]
    assert write_extracts(argv) == 0
    return out


@pytest.fixture(scope="module")
def tables(folder) -> dict:
    return read(folder)


@pytest.fixture(scope="module")
def drawn(tables) -> dict:
    return dict(figures(tables))


def only(rows: list[dict], **match) -> dict:
    (row,) = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return row


def titled(drawn: dict, start: str):
    (fig,) = [f for t, f in drawn.items() if t.startswith(start)]
    return fig


def test_every_extract_is_read_as_its_column_types(tables) -> None:
    assert set(tables) == set(RESULTS_FILES)
    row = tables["risk_metrics"][0]
    assert isinstance(row["sharpe_ratio"], float)
    assert row["first_return"] == "2005-01-04"
    agg = [r for r in tables["risk_metrics"] if r["series"] == "AGG"][0]
    assert agg["max_drawdown_recovery"] is None


def test_the_tiles_are_the_portfolio_s_figures(tables) -> None:
    zero = [
        r for r in tables["risk_metrics"] if (r["series"], r["risk_free"]) == ("portfolio", "zero")
    ]
    shown = dict(tiles(tables))
    assert shown["Compound return a year"] == f"{zero[0]['compound_return']:.2%}"
    assert shown["Sharpe ratio, zero rate"] == f"{zero[0]['sharpe_ratio']:.2f}"
    assert shown["Largest drawdown"] == f"{zero[0]['max_drawdown']:.1%}"
    bills = only(tables["risk_metrics"], series="portfolio", risk_free="bills")
    assert shown["Sharpe ratio, against the bill"] == f"{bills['sharpe_ratio']:.2f}"
    assert bills["sharpe_ratio"] < zero[0]["sharpe_ratio"]
    es = only(tables["tail_risk"], series="portfolio", level=0.99, method="historical")
    assert shown["One-day 99% expected shortfall"] == f"{es['expected_shortfall']:.2%}"


def test_the_frontier_chart_plots_the_extract_s_points(tables, drawn) -> None:
    fig = titled(drawn, "Efficient frontier")
    for trace, cap in zip(fig.data[:2], (1.0, 0.3), strict=True):
        points = [r for r in tables["frontier"] if r["max_weight"] == cap]
        assert list(trace.x) == [p["volatility"] for p in points]
        assert list(trace.y) == [p["expected_return"] for p in points]
    funds = [r for r in tables["portfolios"] if r["max_weight"] == 1.0 and r["kind"] == "fund"]
    assert list(fig.data[2].text) == [r["name"] for r in funds]
    assert list(fig.data[2].y) == [r["expected_return"] for r in funds]
    assert list(fig.data[2].x) == [r["volatility"] for r in funds]
    named = [r for r in tables["portfolios"] if r["max_weight"] == 1.0 and r["kind"] == "portfolio"]
    assert list(fig.data[3].text) == [r["name"] for r in named] != []
    assert list(fig.data[3].x) == [r["volatility"] for r in named]
    assert list(fig.data[3].y) == [r["expected_return"] for r in named]
    assert len(fig.data) == 4


def test_the_box_chart_draws_each_distribution_s_quantiles(tables, drawn) -> None:
    fig = titled(drawn, "One-year return")
    assert [t.x[0] for t in fig.data] == list(dashboard.MODELS)
    for trace in fig.data:
        r = only(tables["montecarlo"], source=trace.x[0], scope="horizon")
        drawn_box = (trace.lowerfence, trace.q1, trace.median, trace.q3, trace.upperfence)
        assert [v[0] for v in drawn_box] == [r[q] for q in ("q1", "q5", "q50", "q95", "q99")]


def test_the_stress_heatmap_holds_each_period_s_return(tables, drawn) -> None:
    fig = titled(drawn, "Stress periods")
    (heat,) = fig.data
    for i, period in enumerate(heat.y):
        for j, holding in enumerate(heat.x):
            rule = None if j < 8 else dashboard.STRESS_RULE
            r = only(tables["stress"], period=period, holding=holding, rule=rule)
            assert heat.z[i][j] == r["return"]
    assert len(heat.y) == 6 and len(heat.x) == 12


def test_the_held_out_and_factor_charts_plot_the_extracts(tables, drawn) -> None:
    fig = titled(drawn, "Held out")
    named = {}
    for trace in fig.data:
        command = "frontier" if trace.xaxis == "x" else "construction"
        rows = [r for r in tables["held_out"] if (r["command"], r["max_weight"], r["risk_free"],
                r["portfolio"]) == (command, 1.0, "zero", trace.name)]  # fmt: skip
        assert list(trace.y) == [r["held_sharpe_ratio"] for r in rows]
        assert named.setdefault(trace.name, trace.line.color) == trace.line.color
    assert sum(t.showlegend is not False for t in fig.data) == len(named) == 6
    (heat,) = titled(drawn, "Fama-French").data
    for i, series in enumerate(heat.y):
        for j, term in enumerate(heat.x):
            (r,) = [r for r in tables["factors"]
                    if (r["series"], r["model"], r["term"]) == (series, "ff5", term)]  # fmt: skip
            assert heat.z[i][j] == r["estimate"]


def test_costs_and_sharpe_charts_plot_the_extracts(tables, drawn) -> None:
    fig = titled(drawn, "Compound return a year after trading costs")
    assert len(fig.data) == 4 * 6
    for trace in fig.data:
        mix = fig.layout.annotations[int(trace.xaxis[1:] or 1) - 1].text
        rows = [r for r in tables["trading_costs"] if (r["mix"], r["rule"]) == (mix, trace.name)]
        assert len(rows) == 6
        assert list(trace.y) == [r["net_compound_return"] for r in rows]
        # The one value not plotted as written: the cost, in basis points.
        assert list(trace.x) == pytest.approx([r["cost"] * 1e4 for r in rows], rel=1e-15)
    fig = titled(drawn, "Sharpe ratio at a zero rate")
    for trace, rate in zip(fig.data, ("zero", "bills"), strict=True):
        rows = {r["series"]: r for r in tables["risk_metrics"] if r["risk_free"] == rate}
        assert list(trace.y) == [rows[s]["sharpe_ratio"] for s in trace.x]
        assert len(trace.x) == 9


def test_the_tail_chart_plots_each_series_99_figures(tables, drawn) -> None:
    fig = titled(drawn, "One-day loss at 99%")
    assert [(t.name, t.xaxis) for t in fig.data] == [
        ("historical", "x"), ("historical", "x2"), ("normal", "x"), ("normal", "x2"),
    ]  # fmt: skip
    for trace in fig.data:
        measure = "value_at_risk" if trace.xaxis == "x" else "expected_shortfall"
        for series, value in zip(trace.x, trace.y, strict=True):
            row = only(tables["tail_risk"], series=series, level=0.99, method=trace.name)
            assert value == row[measure]
    assert len(fig.data[0].x) == 9


def daily_rows(tables) -> list[dict]:
    series = [r["series"] for r in tables["risk_metrics"] if r["risk_free"] == "zero"]
    # Figures differ by series and day, so a series drawn from another's rows shows.
    return [
        {"date": d, "series": s, "daily_return": 0.01 * i + k, "wealth": 1 + 0.01 * i + k,
         "drawdown": -0.001 * i - k / 100}
        for k, s in enumerate(series) for i, d in enumerate(["2024-01-02", "2024-01-03"], 1)
    ]  # fmt: skip


def test_without_daily_rows_the_charts_over_time_are_left_out_and_said_to_be(tables) -> None:
    text = page(tables, cdn=True)
    assert "No daily.csv in the extracts" in text
    assert "over time" not in "".join(t for t, _ in figures(tables))
    with_daily = dict(tables) | {"daily": daily_rows(tables)}
    (title, fig), *_ = figures(with_daily)
    assert title.startswith("Each fund and the portfolio over time")
    assert len(fig.data) == 2 * 9
    rows = with_daily["daily"]
    for growth, drawdown in zip(fig.data[::2], fig.data[1::2], strict=True):
        assert growth.name == drawdown.name
        mine = [r for r in rows if r["series"] == growth.name]
        assert list(growth.x) == list(drawdown.x) == [r["date"] for r in mine]
        assert list(growth.y) == [r["wealth"] for r in mine]
        assert list(drawdown.y) == [r["drawdown"] for r in mine]
    assert [t.name for t in fig.data[::2]] == list(dict.fromkeys(r["series"] for r in rows))
    assert "No daily.csv" not in page(with_daily, cdn=True)


def test_the_command_writes_one_page(folder, tmp_path: Path, capsys) -> None:
    out = tmp_path / "page" / "dashboard.html"
    assert main(["--extracts", str(folder), "--out", str(out), "--cdn"]) == 0
    assert f"wrote {out}" in capsys.readouterr().out
    text = out.read_text(encoding="utf-8")
    assert text.count("<h2>") == 8
    assert "cdn.plot.ly" in text
    assert len(text) < 2_000_000


def test_an_out_path_that_is_a_folder_is_refused(folder, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--extracts", str(folder), "--out", str(tmp_path)])


def copy(folder: Path, tmp_path: Path) -> Path:
    target = tmp_path / "x"
    target.mkdir()
    for path in folder.iterdir():
        (target / path.name).write_bytes(path.read_bytes())
    return target


def restamp(folder: Path) -> None:
    """List every CSV file in the folder in its manifest as it now is, as a run of the
    extracts would, so a test's edit reaches the check it is aimed at."""
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"] = {
        p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in folder.glob("*.csv")
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda p: p.unlink(), "has no factors.csv"),
        (lambda p: p.write_text("a,b\n", encoding="utf-8"), "has columns"),
        (lambda p: p.write_text(",".join(COLUMNS["factors"]) + "\n", encoding="utf-8"), "no rows"),
        (
            lambda p: p.write_text(
                p.read_text(encoding="utf-8").replace(",ff3,", ",,", 1), encoding="utf-8"
            ),
            "model empty",
        ),  # fmt: skip
        (
            lambda p: p.write_text(
                p.read_text(encoding="utf-8").replace("0.9855954497197529", "nan", 1),
                encoding="utf-8",
            ),
            "not a finite number",
        ),  # fmt: skip
        (
            lambda p: p.write_text(
                p.read_text(encoding="utf-8") + "SPY,ff3,alpha,1,1,1,1\n", encoding="utf-8"
            ),
            "repeats line 2",
        ),  # fmt: skip
        (
            lambda p: p.write_text(p.read_text(encoding="utf-8") + "SPY,ff3\n", encoding="utf-8"),
            "has 2 fields",
        ),  # fmt: skip
        (lambda p: p.write_bytes(b"\xff\xfe"), "is not UTF-8 text"),
    ],
)
def test_a_missing_or_damaged_extract_is_named(folder, tmp_path, capsys, damage, message) -> None:
    target = copy(folder, tmp_path)
    damage(target / "factors.csv")
    restamp(target)
    assert main(["--extracts", str(target), "--out", str(tmp_path / "d.html")]) == 1
    err = capsys.readouterr().err
    assert message in err
    assert not (tmp_path / "d.html").exists()


def test_a_folder_that_is_not_there_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DashboardError, match="is not a folder of extracts"):
        read(tmp_path / "none")


def write_daily(folder: Path, rows: list[dict]) -> None:
    columns = list(COLUMNS["daily"])
    lines = [",".join(columns)] + [",".join(str(r[c]) for c in columns) for r in rows]
    text = "".join(f"{line}\n" for line in lines)
    (folder / "daily.csv").write_text(text, encoding="utf-8")
    restamp(folder)


def whole_daily(tables) -> list[dict]:
    zero = [r for r in tables["risk_metrics"] if r["risk_free"] == "zero"]
    days = [zero[0]["first_return"], "2015-06-01", zero[0]["last_return"]]
    return [
        {"date": d, "series": r["series"], "daily_return": 0.0, "wealth": 1.0, "drawdown": 0.0}
        for r in zero for d in days
    ]  # fmt: skip


def test_a_daily_file_over_the_risk_measures_days_is_drawn(folder, tables, tmp_path) -> None:
    target = copy(folder, tmp_path)
    write_daily(target, whole_daily(tables))
    assert len(read(target)["daily"]) == 9 * 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: [r for r in rows if r["date"] != "2026-09-30"], "to 2026-09-29"),
        (lambda rows: [r for r in rows if r["series"] != "VNQ"], "holds SPY"),
        (
            lambda rows: [r for r in rows if (r["series"], r["date"]) != ("GLD", "2015-06-01")],
            "does not have every day for GLD",
        ),  # fmt: skip
    ],
)
def test_a_daily_file_from_another_window_is_refused(
    folder, tables, tmp_path, capsys, change, message
) -> None:
    target = copy(folder, tmp_path)
    rows = change(whole_daily(tables))
    if message == "to 2026-09-29":
        rows = [r | {"date": "2026-09-29"} if r["date"] == "2015-06-01" else r for r in rows]
    write_daily(target, rows)
    assert main(["--extracts", str(target), "--out", str(tmp_path / "d.html")]) == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "d.html").exists()


@pytest.mark.parametrize("bad", ["not-a-date", "2007-13-09", "20071009", "2007-10-9"])
def test_a_date_column_holding_something_else_is_refused(folder, tmp_path, capsys, bad) -> None:
    target = copy(folder, tmp_path)
    path = target / "stress.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert ",2007-10-09," in lines[1]
    lines[1] = lines[1].replace(",2007-10-09,", f",{bad},", 1)
    path.write_text("".join(lines), encoding="utf-8")
    restamp(target)
    assert main(["--extracts", str(target), "--out", str(tmp_path / "d.html")]) == 1
    assert f"stress.csv line 2: not a date as YYYY-MM-DD: '{bad}'" in capsys.readouterr().err
    assert not (tmp_path / "d.html").exists()


def test_daily_rows_beside_no_zero_rate_risk_rows_are_refused(
    folder, tables, tmp_path, capsys
) -> None:
    target = copy(folder, tmp_path)
    write_daily(target, whole_daily(tables))
    path = target / "risk_metrics.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join([lines[0], *(ln for ln in lines[1:] if ",bills," in ln)]),
                    encoding="utf-8")  # fmt: skip
    restamp(target)
    assert main(["--extracts", str(target), "--out", str(tmp_path / "d.html")]) == 1
    assert "risk_metrics.csv has no rows at the zero rate" in capsys.readouterr().err


def test_a_daily_file_the_manifest_does_not_list_is_refused(folder, tables, tmp_path, capsys):
    # Right funds, right days: only the manifest shows another run wrote it.
    target = copy(folder, tmp_path)
    columns = list(COLUMNS["daily"])
    rows = whole_daily(tables)
    lines = [",".join(columns), *(",".join(str(r[c]) for c in columns) for r in rows)]
    (target / "daily.csv").write_text("".join(f"{x}\n" for x in lines), encoding="utf-8")
    assert main(["--extracts", str(target), "--out", str(tmp_path / "d.html")]) == 1
    err = capsys.readouterr().err
    assert f"{target / 'daily.csv'} is not from the run that wrote manifest.json" in err
    assert not (tmp_path / "d.html").exists()


def edit_factors(target: Path) -> None:
    path = target / "factors.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("0.98", "0.97", 1), encoding="utf-8")


def drop_manifest(target: Path) -> None:
    (target / "manifest.json").unlink()


def manifest_text(text: str):
    return lambda target: (target / "manifest.json").write_text(text, encoding="utf-8")


def list_a_missing_file(target: Path) -> None:
    path = target / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"]["daily.csv"] = {"sha256": "0" * 64}
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (edit_factors, "factors.csv has changed since the extracts were written"),
        (drop_manifest, "has no manifest.json: write the extracts"),
        (manifest_text("{"), "manifest.json cannot be read"),
        (manifest_text('{"files": []}'), "is not an extracts manifest"),
        (manifest_text('{"files": {"factors.csv": {}}}'), "is not an extracts manifest"),
        (list_a_missing_file, "lists daily.csv, which"),
    ],
)
def test_extracts_not_all_from_one_run_are_refused(folder, tmp_path, capsys, change, message):
    target = copy(folder, tmp_path)
    change(target)
    assert main(["--extracts", str(target), "--out", str(tmp_path / "d.html")]) == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "d.html").exists()
