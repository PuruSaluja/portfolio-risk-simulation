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
    trace = fig.data[0]
    rows = [r for r in tables["trading_costs"] if (r["mix"], r["rule"]) == ("equity", "daily")]
    assert list(trace.y) == [r["net_compound_return"] for r in rows]
    assert list(trace.x) == pytest.approx([r["cost"] * 1e4 for r in rows], rel=1e-15)
    fig = titled(drawn, "Sharpe ratio at a zero rate")
    bills = [r for r in tables["risk_metrics"] if r["risk_free"] == "bills"]
    assert list(fig.data[1].y) == [r["sharpe_ratio"] for r in bills]


def daily_rows(tables) -> list[dict]:
    series = [r["series"] for r in tables["risk_metrics"] if r["risk_free"] == "zero"]
    return [
        {"date": d, "series": s, "daily_return": 0.01 * i, "wealth": 1 + 0.01 * i,
         "drawdown": -0.001 * i}
        for s in series for i, d in enumerate(["2024-01-02", "2024-01-03"], 1)
    ]  # fmt: skip


def test_without_daily_rows_the_charts_over_time_are_left_out_and_said_to_be(tables) -> None:
    text = page(tables, cdn=True)
    assert "No daily.csv in the extracts" in text
    assert "over time" not in "".join(t for t, _ in figures(tables))
    with_daily = dict(tables) | {"daily": daily_rows(tables)}
    (title, fig), *_ = figures(with_daily)
    assert title.startswith("Each fund and the portfolio over time")
    assert len(fig.data) == 2 * 9
    assert list(fig.data[0].y) == [1.01, 1.02]
    assert list(fig.data[1].y) == [-0.001, -0.002]
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
    assert main(["--extracts", str(target), "--out", str(tmp_path / "d.html")]) == 1
    err = capsys.readouterr().err
    assert message in err
    assert not (tmp_path / "d.html").exists()


def test_a_folder_that_is_not_there_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DashboardError, match="is not a folder of extracts"):
        read(tmp_path / "none")
