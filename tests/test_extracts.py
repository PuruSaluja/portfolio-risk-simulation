import csv
import json
import math
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import extracts
from portfolio_risk import risk as risk_command
from portfolio_risk.data import COLUMNS as PRICE_COLUMNS
from portfolio_risk.data import write_cache
from portfolio_risk.extracts import (
    COLUMNS,
    KEYS,
    RESULTS,
    RESULTS_FILES,
    ExtractError,
    check,
    main,
    read_results,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def read(path: Path, name: str) -> list[dict]:
    """A written extract, each field read back as its column's type."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        assert next(reader) == list(COLUMNS[name])
        rows = []
        for values in reader:
            row = {}
            for column, text in zip(COLUMNS[name], values, strict=True):
                kind = COLUMNS[name][column]
                row[column] = None if text == "" else float(text) if kind == "number" else text
            rows.append(row)
    return rows


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> dict[str, list[dict]]:
    out = tmp_path_factory.mktemp("extracts")
    assert main(["--results", str(RESULTS_DIR), "--no-prices", "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{n}.csv" for n in RESULTS_FILES)
    return {n: read(out / f"{n}.csv", n) for n in RESULTS_FILES}


@pytest.fixture(scope="module")
def results() -> dict:
    return read_results(RESULTS_DIR)


def by_key(rows: list[dict], name: str) -> dict[tuple, dict]:
    keyed = {tuple(r[k] for k in KEYS[name]): r for r in rows}
    assert len(keyed) == len(rows)
    return keyed


def test_risk_metrics_are_the_results_figures_at_each_rate(written, results) -> None:
    rows = by_key(written["risk_metrics"], "risk_metrics")
    expected = 0
    for name, rate in (("risk", "zero"), ("risk_bills", "bills")):
        for series, m in results[name]["series"].items():
            row = rows[(series, rate)]
            expected += 1
            assert row["sharpe_ratio"] == m["sharpe_ratio"]
            assert row["sortino_ratio"] == m["sortino_ratio"]
            assert row["compound_return"] == m["compound_return"]
            assert row["max_drawdown"] == m["max_drawdown"]["depth"]
            assert row["max_drawdown_recovery"] == m["max_drawdown"]["recovery"]
            assert row["last_return"] == results[name]["last_return"]
    assert len(rows) == expected == 18
    # AGG and TLT had not recovered by the last day: an empty field, not a date.
    assert rows[("AGG", "zero")]["max_drawdown_recovery"] is None


def test_tail_risk_and_monte_carlo_rows_hold_every_figure(written, results) -> None:
    tails = by_key(written["tail_risk"], "tail_risk")
    for series, m in results["risk"]["series"].items():
        for level, methods in m["tail"].items():
            for method, tail in methods.items():
                row = tails[(series, int(level) / 100, method)]
                assert row["value_at_risk"] == tail["value_at_risk"]
                assert row["expected_shortfall"] == tail["expected_shortfall"]
    assert len(tails) == 9 * 2 * 2
    mc = by_key(written["montecarlo"], "montecarlo")
    models = {m["model"]: m for m in results["montecarlo"]["models"]}
    models["historical"] = results["montecarlo"]["historical"]
    assert len(mc) == 2 * len(models)
    for (source, scope), row in mc.items():
        stats = models[source][scope]
        assert row["q0_1"] == stats["q0.1"]
        assert row["sd"] == stats["sd"]
        assert row["count"] == stats["count"]


def test_stress_rebalancing_and_cost_rows_hold_every_figure(written, results) -> None:
    scenarios = results["scenarios"]
    stress = by_key(written["stress"], "stress")
    count = 0
    for period in scenarios["stress"]:
        for fund, value in period["funds"].items():
            row = stress[(period["period"], fund, None)]
            assert (row["kind"], row["return"], row["max_drawdown"]) == ("fund", value, None)
            count += 1
        for mix, rules in period["mixes"].items():
            for rule, f in rules.items():
                row = stress[(period["period"], mix, rule)]
                assert (row["return"], row["max_drawdown"]) == (f["return"], f["max_drawdown"])
                assert row["start"] == period["start"]
                count += 1
    assert count == len(stress)
    rebalancing = by_key(written["rebalancing"], "rebalancing")
    assert len(rebalancing) == 4 * 6
    for (mix, rule), row in rebalancing.items():
        assert row["sharpe_ratio"] == scenarios["history"][mix][rule]["sharpe_ratio"]
        assert row["rebalances"] == scenarios["history"][mix][rule]["rebalances"]
    costs = by_key(written["trading_costs"], "trading_costs")
    assert len(costs) == 4 * 6 * 6
    for (mix, rule, cost), row in costs.items():
        (c,) = [c for c in results["costs"]["history"][mix][rule]["costs"] if c["cost"] == cost]
        assert (row["net_compound_return"], row["drag"]) == (c["net_compound_return"], c["drag"])


def test_frontier_points_weights_and_named_portfolios(written, results) -> None:
    points = by_key(written["frontier"], "frontier")
    weights = by_key(written["frontier_weights"], "frontier_weights")
    named = by_key(written["portfolios"], "portfolios")
    for name in ("frontier", "frontier_cap30"):
        result = results[name]
        cap = result["max_weight"]
        for i, p in enumerate(result["frontier"]):
            assert points[(cap, i)]["volatility"] == p["volatility"]
            for fund, w in p["weights"].items():
                assert weights[(cap, i, fund)]["weight"] == w
        for portfolio, p in result["portfolios"].items():
            assert named[(cap, portfolio)]["sharpe_ratio"] == p["sharpe_ratio"]
        for fund, f in result["funds"].items():
            assert named[(cap, fund)]["expected_return"] == f["expected_return"]
    assert {cap for cap, _ in points} == {1.0, 0.3}
    assert len(points) == len(results["frontier"]["frontier"]) * 2
    assert len(weights) == len(points) * 8


def test_held_out_rows_take_the_ratio_each_command_chose_on(written, results) -> None:
    held = by_key(written["held_out"], "held_out")
    costs = by_key(written["held_costs"], "held_costs")
    split = results["frontier_bills"]["splits"][2]
    p = split["portfolios"]["max_sharpe"]
    row = held[("frontier", 1.0, "bills", split["date"], "max_sharpe")]
    assert row["chosen_sharpe_ratio"] == p["sharpe_ratio"]
    assert row["held_sharpe_ratio"] == p["realised"]["sharpe_ratio"]
    split = results["construction"]["splits"][4]
    p = split["portfolios"]["risk_parity"]
    row = held[("construction", 1.0, "zero", split["date"], "risk_parity")]
    assert row["chosen_sharpe_ratio"] == p["in_sample"]["sharpe_ratio"]
    assert row["held_compound_return"] == p["held"]["compound_return"]
    c = p["held"]["costs"]["band:0.05"]["costs"][-1]
    row = costs[("construction", 1.0, "zero", split["date"], "risk_parity", "band:0.05", 0.005)]
    assert (row["compound_return"], row["sharpe_ratio"]) == (
        c["compound_return"],
        c["sharpe_ratio"],
    )
    # 4 frontier files x 5 splits x 3 portfolios, 2 construction files x 5 x 6.
    assert len(held) == 4 * 5 * 3 + 2 * 5 * 6
    assert len(costs) == len(held) * 3 * 6


def test_factor_rows_hold_every_coefficient(written, results) -> None:
    rows = by_key(written["factors"], "factors")
    count = 0
    for series, models in results["factors"]["series"].items():
        for model, fit in models.items():
            for term, c in fit["coefficients"].items():
                row = rows[(series, model, term)]
                assert (row["estimate"], row["newey_west_t"]) == (c["estimate"], c["newey_west_t"])
                assert row["r_squared"] == fit["r_squared"]
                count += 1
    assert count == len(rows) == 9 * (4 + 6)


def test_every_number_reads_back_as_the_float_written(tmp_path: Path) -> None:
    rows = [{"source": "a", "scope": "daily"} | {c: v for c in COLUMNS["montecarlo"]
            if c not in ("source", "scope") for v in [0.1 + 0.2]}]  # fmt: skip
    rows[0]["mean"] = 5e-324
    rows[0]["sd"] = -1.7976931348623157e308
    rows[0]["count"] = 2520000
    extracts.write(tmp_path, {"montecarlo": rows})
    (back,) = read(tmp_path / "montecarlo.csv", "montecarlo")
    assert back == {k: float(v) if isinstance(v, int | float) else v for k, v in rows[0].items()}
    text = (tmp_path / "montecarlo.csv").read_text(encoding="utf-8")
    assert ",2520000," in text and "\r" not in text


def row_for(name: str, **changes) -> dict:
    row = {c: ("x" if kind == "text" else "2024-01-02" if kind == "date" else 1.0)
           for c, kind in COLUMNS[name].items()}  # fmt: skip
    return row | changes


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([row_for("rebalancing", volatility=math.nan)], "not a finite number"),
        ([row_for("rebalancing", volatility=math.inf)], "not a finite number"),
        ([row_for("rebalancing", volatility=True)], "not a number"),
        ([row_for("rebalancing", volatility="0.1")], "not a number"),
        ([row_for("rebalancing", rule=3)], "not text"),
        ([row_for("rebalancing", volatility=None)], "volatility empty"),
        ([row_for("rebalancing"), row_for("rebalancing")], "two rows for x, x"),
        ([{**row_for("rebalancing"), "extra": 1.0}], "a row has columns"),
        ([], "no rows"),
    ],
)
def test_a_table_that_cannot_be_written_as_typed_columns_is_refused(rows, message) -> None:
    with pytest.raises(ExtractError, match=message):
        check("rebalancing", rows)


def test_a_figure_the_results_hold_as_null_is_an_empty_field() -> None:
    (line,) = check("rebalancing", [row_for("rebalancing", sharpe_ratio=None)])
    assert line[list(COLUMNS["rebalancing"]).index("sharpe_ratio")] == ""


def test_nothing_is_written_when_one_table_is_refused(tmp_path: Path) -> None:
    good = [row_for("tail_risk")]
    with pytest.raises(ExtractError, match="not a finite number"):
        extracts.write(
            tmp_path / "out",
            {"tail_risk": good, "rebalancing": [row_for("rebalancing", volatility=math.nan)]},
        )
    assert not (tmp_path / "out").exists()


def copy_results(tmp_path: Path) -> Path:
    folder = tmp_path / "results"
    folder.mkdir()
    for file in RESULTS.values():
        shutil.copy(RESULTS_DIR / file, folder / file)
    return folder


def set_cache_sha(folder: Path, sha: str, files=None) -> None:
    for file in files or RESULTS.values():
        path = folder / file
        result = json.loads(path.read_text(encoding="utf-8"))
        result["cache_sha256"] = sha
        path.write_text(json.dumps(result), encoding="utf-8")


def test_results_from_two_price_caches_are_refused(tmp_path: Path, capsys) -> None:
    folder = copy_results(tmp_path)
    set_cache_sha(folder, "0" * 64, ["costs.json"])
    out = tmp_path / "out"
    assert main(["--results", str(folder), "--no-prices", "--out", str(out)]) == 1
    assert "come from 2 different price caches" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("damage", ["missing", "not json", "a list", "no checksum"])
def test_a_missing_or_damaged_results_file_is_named(tmp_path: Path, capsys, damage) -> None:
    folder = copy_results(tmp_path)
    path = folder / "scenarios.json"
    if damage == "missing":
        path.unlink()
    else:
        text = {"not json": "{", "a list": "[]", "no checksum": '{"stress": []}'}[damage]
        path.write_text(text, encoding="utf-8")
    assert main(["--results", str(folder), "--no-prices", "--out", str(tmp_path / "o")]) == 1
    err = capsys.readouterr().err
    assert str(path) in err
    assert not (tmp_path / "o").exists()


def test_a_folder_that_is_not_there_or_an_out_path_that_is_a_file_is_refused(
    tmp_path: Path, capsys
) -> None:
    assert main(["--results", str(tmp_path / "none"), "--no-prices", "--out", str(tmp_path)]) == 1
    assert "is not a folder of results" in capsys.readouterr().err
    file = tmp_path / "file.csv"
    file.write_text("x", encoding="utf-8")
    assert main(["--results", str(RESULTS_DIR), "--no-prices", "--out", str(file)]) == 1
    assert "is a file; --out takes a folder" in capsys.readouterr().err
    assert file.read_text(encoding="utf-8") == "x"


def prices() -> pd.DataFrame:
    days = pd.bdate_range("2023-01-02", "2024-12-31")
    rng = np.random.default_rng(3)
    rows = []
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        close = 40.0 * np.cumprod(1 + rng.normal(0.0002, 0.01 * (1 + i), len(days)))
        for n, (day, value) in enumerate(zip(days, close, strict=True)):
            dividend = 0.5 if (ticker == "BBB" and n % 60 == 30) else 0.0
            rows.append((day.date(), ticker, float(value), dividend, 0.0, 0.0, 1e6))
    return pd.DataFrame(rows, columns=list(PRICE_COLUMNS))


def test_daily_rows_are_the_risk_command_s_series(tmp_path: Path, capsys) -> None:
    cache = tmp_path / "prices.csv"
    manifest = write_cache(
        prices(), cache, start=date(2023, 1, 1), end=date(2024, 12, 31), source="t"
    )
    folder = copy_results(tmp_path)
    risk_json = folder / RESULTS["risk"]
    weights = "AAA=0.5,BBB=0.3,CCC=0.2"
    argv = ["--cache", str(cache), "--weights", weights, "--json", str(risk_json)]
    assert risk_command.main(argv) == 0
    set_cache_sha(folder, manifest["sha256"])
    out = tmp_path / "out"
    assert main(["--results", str(folder), "--cache", str(cache), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert f"{out / 'daily.csv'}  {521 * 4:,} rows" in printed
    rows = read(out / "daily.csv", "daily")
    frame = pd.DataFrame(rows)
    assert list(frame["series"].unique()) == ["AAA", "BBB", "CCC", "portfolio"]
    measured = json.loads(risk_json.read_text(encoding="utf-8"))["series"]
    wide = frame.pivot(index="date", columns="series", values="daily_return")
    blend = wide["AAA"] * 0.5 + wide["BBB"] * 0.3 + wide["CCC"] * 0.2
    np.testing.assert_allclose(wide["portfolio"], blend, rtol=0, atol=1e-16)
    for name, group in frame.groupby("series"):
        wealth = np.cumprod(1 + group["daily_return"].to_numpy())
        np.testing.assert_allclose(group["wealth"], wealth, rtol=1e-15)
        lowest = group.loc[group["drawdown"].idxmin()]
        assert lowest["drawdown"] == measured[name]["max_drawdown"]["depth"]
        assert lowest["date"] == measured[name]["max_drawdown"]["trough"]
        assert (group["drawdown"] <= 0).all()


def test_a_cache_the_results_did_not_come_from_is_refused(tmp_path: Path, capsys) -> None:
    cache = tmp_path / "prices.csv"
    write_cache(prices(), cache, start=date(2023, 1, 1), end=date(2024, 12, 31), source="t")
    out = tmp_path / "out"
    assert main(["--results", str(RESULTS_DIR), "--cache", str(cache), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "is not the cache the results were computed from" in err
    assert "--no-prices" in err
    assert not out.exists()


def test_a_loss_on_the_first_day_is_a_drawdown_from_the_start() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    frame = pd.DataFrame(
        [(d, "AAA", c, 0.0, 0.0, 0.0, 1e6) for d, c in zip(days, [100.0, 90.0, 94.5], strict=True)],
        columns=list(PRICE_COLUMNS),
    )
    rows = extracts.daily(frame, {"AAA": 1.0})
    fund = [r for r in rows if r["series"] == "AAA"]
    assert [r["date"] for r in fund] == ["2024-01-03", "2024-01-04"]
    assert [r["wealth"] for r in fund] == pytest.approx([0.9, 0.945], rel=1e-15)
    assert [r["drawdown"] for r in fund] == pytest.approx([-0.1, -0.055], rel=1e-14)
