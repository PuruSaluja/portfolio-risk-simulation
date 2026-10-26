import itertools
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import construction, costs, french
from portfolio_risk import frontier as fr
from portfolio_risk.construction import (
    PORTFOLIOS,
    inverse_volatility,
    main,
    minimum_cvar,
    risk_contributions,
    risk_parity,
)
from portfolio_risk.data import COLUMNS, load_cache, write_cache
from portfolio_risk.returns import total_returns
from portfolio_risk.risk import expected_shortfall, sharpe_ratio, value_at_risk


def random_covariance(n, seed, correlation=None):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.05, 0.35, n)
    if correlation is None:
        a = rng.normal(size=(n, n + 3))
        c = a @ a.T
        d = np.sqrt(np.diag(c))
        corr = c / np.outer(d, d)
    else:
        corr = np.full((n, n), correlation)
        np.fill_diagonal(corr, 1.0)
    names = [f"F{i}" for i in range(n)]
    return pd.DataFrame(corr * np.outer(vols, vols), index=names, columns=names)


@pytest.mark.parametrize(("n", "seed"), [(2, 0), (3, 1), (5, 2), (8, 3), (20, 4), (50, 5)])
def test_risk_parity_gives_every_fund_the_same_share_of_risk(n, seed) -> None:
    covariance = random_covariance(n, seed)
    w = risk_parity(covariance)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)
    assert (w > 0).all()
    np.testing.assert_allclose(risk_contributions(w, covariance), 1.0 / n, rtol=1e-12)


def test_budgets_set_each_fund_s_share_of_risk() -> None:
    covariance = random_covariance(4, 7)
    budgets = pd.Series([4.0, 3.0, 2.0, 1.0], index=covariance.index)
    contributions = risk_contributions(risk_parity(covariance, budgets), covariance)
    np.testing.assert_allclose(contributions, [0.4, 0.3, 0.2, 0.1], rtol=1e-12)


@pytest.mark.parametrize("correlation", [-0.15, 0.0, 0.5, 0.99])
def test_with_equal_correlations_risk_parity_is_inverse_volatility(correlation) -> None:
    covariance = random_covariance(6, 11, correlation=correlation)
    np.testing.assert_allclose(risk_parity(covariance), inverse_volatility(covariance), rtol=1e-12)


def test_two_funds_hold_in_inverse_proportion_to_their_volatility_whatever_the_correlation() -> (
    None
):
    for rho in (-0.9, 0.0, 0.95):
        covariance = pd.DataFrame(
            [[0.04, rho * 0.2 * 0.05], [rho * 0.2 * 0.05, 0.0025]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        w = risk_parity(covariance)
        assert w["A"] == pytest.approx(0.05 / 0.25, rel=1e-13)


def test_inverse_volatility_weights_by_one_over_volatility() -> None:
    covariance = random_covariance(5, 3)
    w = inverse_volatility(covariance)
    vols = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(w * vols, (w * vols).iloc[0], rtol=1e-13)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)


def test_risk_parity_converges_on_badly_conditioned_matrices() -> None:
    for seed in range(20):
        # Correlations of 0.999 and a little more, so not all equal.
        common = random_covariance(10, seed, correlation=0.999)
        varied = random_covariance(10, seed + 100)
        vols = np.sqrt(np.diag(common))
        covariance = common + 0.001 * varied / np.sqrt(
            np.outer(np.diag(varied), np.diag(varied))
        ) * np.outer(vols, vols)
        w = risk_parity(covariance)
        np.testing.assert_allclose(risk_contributions(w, covariance), 0.1, rtol=1e-9)


def test_contributions_add_up_to_one() -> None:
    covariance = random_covariance(5, 9)
    w = pd.Series([0.1, 0.2, 0.3, 0.15, 0.25], index=covariance.index)
    contributions = risk_contributions(w, covariance)
    assert contributions.sum() == pytest.approx(1.0, rel=1e-14)
    s = covariance.to_numpy()
    assert contributions.iloc[2] == pytest.approx(0.3 * (s @ w.to_numpy())[2] / (w @ s @ w))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.to_numpy(), "DataFrame"),
        (lambda c: c.iloc[:, :2], "square"),
        (lambda c: c.set_axis(["X", "F1", "F2"], axis=0), "square"),
        (lambda c: c.mask(c > 0.0, np.nan), "finite"),
        (lambda c: c + np.triu(np.full((3, 3), 0.01), 1), "symmetric"),
        (lambda c: c - np.diag(np.full(3, 1.0)), "positive definite"),
    ],
)
def test_bad_covariance_matrices_are_refused(change, message) -> None:
    bad = change(random_covariance(3, 0))
    for construct in (risk_parity, inverse_volatility):
        with pytest.raises((TypeError, ValueError), match=message):
            construct(bad)


def test_bad_budgets_and_weights_are_refused() -> None:
    covariance = random_covariance(3, 0)
    for budgets in ([1.0, 0.0, 1.0], [1.0, -1.0, 2.0], [1.0, np.nan, 1.0]):
        with pytest.raises(ValueError, match="risk budget"):
            risk_parity(covariance, pd.Series(budgets, index=covariance.index))
    with pytest.raises(ValueError, match="risk budget"):
        risk_parity(covariance, pd.Series([1.0, 1.0], index=["F0", "F1"]))
    with pytest.raises(ValueError, match="a weight for every fund"):
        risk_contributions(pd.Series([0.5, 0.5], index=["F0", "F1"]), covariance)


def returns_frame(days=500, funds=3, seed=0):
    rng = np.random.default_rng(seed)
    scales = np.linspace(0.005, 0.02, funds)
    data = rng.standard_t(4, (days, funds)) * scales + 0.0003
    data[:, 1:] += 0.5 * data[:, :1]
    index = pd.bdate_range("2020-01-01", periods=days)
    return pd.DataFrame(data, index=index, columns=[f"F{i}" for i in range(funds)])


def portfolio(returns, w):
    return pd.Series(returns.to_numpy() @ w.to_numpy(), index=returns.index)


@pytest.mark.parametrize(("days", "level"), [(500, 0.95), (503, 0.95), (500, 0.99), (777, 0.9)])
def test_the_optimum_is_the_expected_shortfall_of_its_weights(days, level) -> None:
    returns = returns_frame(days)
    w, cvar, threshold = minimum_cvar(returns, level)
    assert w.sum() == pytest.approx(1.0, abs=1e-15) and (w >= 0).all()
    series = portfolio(returns, w)
    assert cvar == pytest.approx(expected_shortfall(series, level), rel=1e-9)
    tail = (1 - level) * days
    if abs(tail - round(tail)) > 1e-9:
        assert threshold == pytest.approx(value_at_risk(series, level), rel=1e-8)
    else:
        # Any threshold from the next-worst loss up to the value at risk is optimal.
        losses = np.sort(-series.to_numpy())[::-1]
        k = round(tail)
        assert losses[k] - 1e-12 <= threshold <= losses[k - 1] + 1e-12


def test_no_portfolio_on_a_grid_has_a_lower_cvar() -> None:
    returns = returns_frame(400, funds=3, seed=4)
    w, cvar, _ = minimum_cvar(returns, 0.95)
    best = math.inf
    steps = 100
    for i, j in itertools.product(range(steps + 1), repeat=2):
        if i + j <= steps:
            grid = pd.Series([i / steps, j / steps, 1 - (i + j) / steps], index=returns.columns)
            best = min(best, expected_shortfall(portfolio(returns, grid), 0.95))
    assert cvar <= best + 1e-12
    # And the grid's best is close to it: the optimum lies between grid points.
    assert best - cvar < 1e-4


def test_a_cap_holds_and_at_one_over_n_gives_equal_weights() -> None:
    returns = returns_frame(400, funds=4, seed=2)
    uncapped, low, _ = minimum_cvar(returns, 0.95)
    assert uncapped.max() > 0.4
    capped, higher, _ = minimum_cvar(returns, 0.95, max_weight=0.4)
    assert capped.max() <= 0.4 + 1e-15 and higher >= low - 1e-12
    equal, _, _ = minimum_cvar(returns, 0.95, max_weight=0.25)
    np.testing.assert_allclose(equal, 0.25, atol=1e-12)


def test_one_fund_holds_everything() -> None:
    returns = returns_frame(100, funds=1)
    w, cvar, _ = minimum_cvar(returns, 0.95)
    assert w.tolist() == [1.0]
    assert cvar == pytest.approx(expected_shortfall(returns["F0"], 0.95), rel=1e-9)


def test_minimum_cvar_refuses_what_it_cannot_solve() -> None:
    returns = returns_frame(50)
    with pytest.raises(TypeError):
        minimum_cvar(returns.to_numpy())
    for level in (0.0, 1.0, 1.5, True):
        with pytest.raises(ValueError, match="level"):
            minimum_cvar(returns, level)
    for cap in (0.3, 1.01, True):
        with pytest.raises(ValueError, match="max_weight"):
            minimum_cvar(returns, 0.95, cap)
    with pytest.raises(ValueError, match="two days"):
        minimum_cvar(returns.iloc[:1])
    with pytest.raises(ValueError, match="two days"):
        minimum_cvar(returns[[]])
    bad = returns.copy()
    bad.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        minimum_cvar(bad)


def test_risk_parity_says_so_when_it_has_not_converged(monkeypatch) -> None:
    monkeypatch.setattr(construction, "MAX_ITERATIONS", 1)
    with pytest.raises(ArithmeticError, match="did not converge"):
        risk_parity(random_covariance(8, 3))


def test_returns_that_never_lose_have_a_negative_cvar() -> None:
    # Every day a gain: the value at risk, the threshold, is below zero.
    returns = returns_frame(300).abs() + 0.001
    w, cvar, threshold = minimum_cvar(returns, 0.95)
    series = portfolio(returns, w)
    assert cvar < 0 and threshold < 0
    assert cvar == pytest.approx(expected_shortfall(series, 0.95), rel=1e-9)


def prices() -> pd.DataFrame:
    days = pd.bdate_range("2023-01-02", "2024-12-31")
    rng = np.random.default_rng(0)
    common = rng.normal(0.0, 0.006, len(days))
    rows = []
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        moves = 0.0002 * (i + 1) + common * i + rng.normal(0.0, 0.004 * (1 + i), len(days))
        close = 50.0 * np.cumprod(1 + moves)
        for day, value in zip(days, close, strict=True):
            rows.append((day.date(), ticker, float(value), 0.0, 0.0, 0.0, 1e6))
    return pd.DataFrame(rows, columns=list(COLUMNS))


def cache(tmp_path: Path) -> Path:
    path = tmp_path / "prices.csv"
    write_cache(prices(), path, start=date(2023, 1, 1), end=date(2024, 12, 31), source="t")
    return path


def test_the_command_compares_the_six_portfolios_in_and_out_of_sample(
    tmp_path: Path, capsys
) -> None:
    out = tmp_path / "results" / "construction.json"
    path = cache(tmp_path)
    assert main(["--cache", str(path), "--json", str(out), "--split", "2023-12-29"]) == 0
    printed = capsys.readouterr().out
    assert "chosen from 2023-01-03 to 2023-12-29, held from 2024-01-01 to 2024-12-31" in printed
    assert b"\r" not in out.read_bytes()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert tuple(written["portfolios"]) == PORTFOLIOS
    daily = total_returns(load_cache(path))
    expected, covariance = fr.estimates(daily)
    figures = {name: p["in_sample"] for name, p in written["portfolios"].items()}
    weights = {name: pd.Series(p["weights"]) for name, p in written["portfolios"].items()}
    np.testing.assert_allclose(weights["risk_parity"], risk_parity(covariance), rtol=1e-12)
    np.testing.assert_allclose(
        weights["max_sharpe"], fr.max_sharpe(expected, covariance), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(list(figures["risk_parity"]["risk_contributions"].values()), 1 / 3)
    # Each optimum is the best of the six at what it optimises.
    for name, f in figures.items():
        assert figures["minimum_variance"]["volatility"] <= f["volatility"] + 1e-12, name
        assert figures["minimum_cvar"]["cvar"] <= f["cvar"] + 1e-12, name
        assert (f["sharpe_ratio"] or 0) <= figures["max_sharpe"]["sharpe_ratio"] + 1e-9, name
    # The held figures are those of the weights chosen before, rebalanced after the split.
    split = written["splits"][0]
    before = daily.loc[:"2023-12-29"]
    after = daily.loc["2024-01-01":]
    w = minimum_cvar(before, 0.95)[0]
    np.testing.assert_allclose(pd.Series(split["portfolios"]["minimum_cvar"]["weights"]), w)
    held = portfolio(after, w)
    got = split["portfolios"]["minimum_cvar"]["held"]
    assert got["cvar"] == pytest.approx(expected_shortfall(held, 0.95), rel=1e-12)
    assert got["volatility"] == pytest.approx(held.std() * math.sqrt(252), rel=1e-12)
    years = (after.index[-1] - before.index[-1]).days / 365.25
    assert got["compound_return"] == pytest.approx((1 + held).prod() ** (1 / years) - 1, rel=1e-9)
    # Held, each is charged for its trades under each rule; every close at no cost is the
    # portfolio above. Not in sample.
    for name, p in split["portfolios"].items():
        charged = p["held"]["costs"]
        assert charged == costs.held(after, p["weights"], before.index[-1]), name
        assert list(charged) == list(costs.HELD_RULES)
        free = charged["daily"]["costs"][0]
        assert free["compound_return"] == pytest.approx(p["held"]["compound_return"], rel=1e-12)
        assert free["sharpe_ratio"] == pytest.approx(p["held"]["sharpe_ratio"], rel=1e-12)
        assert "costs" not in p["in_sample"]
    assert "Sharpe ratio after a one-way cost per unit traded of" in printed
    in_sample = split["portfolios"]["minimum_cvar"]["in_sample"]
    assert in_sample["cvar"] == pytest.approx(minimum_cvar(before, 0.95)[1], rel=1e-9)


def test_the_cap_level_and_rate_reach_the_portfolios(tmp_path: Path) -> None:
    out = tmp_path / "c.json"
    path = cache(tmp_path)
    daily = total_returns(load_cache(path))
    expected, covariance = fr.estimates(daily)
    argv = ["--cache", str(path), "--json", str(out), "--split", "none"]
    assert main(argv + ["--max-weight", "0.45"]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["splits"] == []
    for name in ("minimum_variance", "max_sharpe", "minimum_cvar"):
        assert max(written["portfolios"][name]["weights"].values()) <= 0.45 + 1e-12
    np.testing.assert_allclose(
        pd.Series(written["portfolios"]["minimum_variance"]["weights"]),
        fr.minimum_variance(covariance, 0.45),
        atol=1e-12,
    )
    assert main(argv + ["--level", "0.99", "--risk-free", "-0.1"]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert (written["max_weight"], written["level"], written["risk_free"]) == (1.0, 0.99, -0.1)
    w = pd.Series(written["portfolios"]["minimum_cvar"]["weights"])
    assert not np.allclose(w, minimum_cvar(daily, 0.95)[0], atol=1e-3)
    np.testing.assert_allclose(w, minimum_cvar(daily, 0.99)[0], atol=1e-12)
    tangency = fr.max_sharpe(expected, covariance, -0.1)
    assert not np.allclose(tangency, fr.max_sharpe(expected, covariance), atol=1e-3)
    np.testing.assert_allclose(
        pd.Series(written["portfolios"]["max_sharpe"]["weights"]), tangency, atol=1e-12
    )
    equal = written["portfolios"]["equal"]["in_sample"]
    history = portfolio(daily, pd.Series(1 / 3, index=daily.columns))
    assert equal["cvar"] == pytest.approx(expected_shortfall(history, 0.99), rel=1e-12)
    assert equal["sharpe_ratio"] == pytest.approx(sharpe_ratio(history, 252, -0.1), rel=1e-12)
    np.testing.assert_allclose(
        pd.Series(written["portfolios"]["inverse_volatility"]["weights"]),
        inverse_volatility(covariance),
        rtol=1e-14,
    )


def test_the_command_refuses_bad_input_plainly(tmp_path: Path, capsys) -> None:
    assert main(["--cache", str(tmp_path / "prices.csv")]) == 1
    assert "manifest.json is missing" in capsys.readouterr().err
    path = cache(tmp_path)
    for argv, message in [
        ([], "on 2009-12-31 leaves returns on 0 days before it"),
        (["--split", "2023-12-29", "--split", "2023-12-29"], "given twice"),
        (["--split", "2024-12-27"], "and 2 after; each side needs more than 3"),
        (["--split", "2024-12-26"], "and 3 after; each side needs more than 3"),
        (["--split", "2023-13-01"], "a split is a date"),
        (["--split", "none", "--max-weight", "0.3"], "cannot add up to 1"),
        (["--split", "none", "--level", "1"], "level"),
        (["--json", str(tmp_path), "--split", "none"], ""),  # a folder
    ]:
        assert main(["--cache", str(path), *argv]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error: ") and message in captured.err


def test_describe_refuses_a_bad_rate_rather_than_giving_no_ratio() -> None:
    # Regression: a rate that is not a number reached the Sharpe ratio, whose refusal
    # became None, as for returns that never vary.
    daily = total_returns(prices()).iloc[:60]
    w = pd.Series(1 / 3, index=daily.columns)
    first = daily.index[0] - pd.Timedelta(days=3)
    for rate in (math.nan, math.inf, -1.0):
        with pytest.raises(ValueError, match="an annual rate must be a finite number"):
            construction.describe(daily, w, first, rate)
    # A rate equal to the portfolio's own return each day leaves an excess return that
    # never varies: no ratio, and no error.
    own = portfolio(daily, w)
    assert construction.describe(daily, w, first, own)["sharpe_ratio"] is None


def test_risk_parity_converges_where_the_last_step_is_below_rounding() -> None:
    # Near the solution the Newton decrease is far below what the objective can show;
    # a line search there sees only rounding. These include matrices on which it
    # refused the last step and stopped short (n = 3, seeds 28, 30, 38, 47, 81).
    for n in (3, 5, 8):
        for seed in range(100):
            for scale in (1.0, 1e-4, 1e4):
                covariance = random_covariance(n, seed) * scale
                w = risk_parity(covariance)
                np.testing.assert_allclose(risk_contributions(w, covariance), 1 / n, rtol=1e-11)


def factor_cache(tmp_path: Path, end="2024-11-29") -> Path:
    """A factor cache over the price days up to ``end``, the bill rate rising each month."""
    days = pd.bdate_range("2023-01-02", end)
    table = pd.DataFrame({"date": days.date})
    for column in ("mkt_rf", "smb", "hml", "rf", "smb5", "rmw", "cma"):
        table[column] = 0.01
    table["rf_month"] = np.round(0.2 + 0.03 * (days.month + 12 * (days.year - 2023)), 2)
    path = tmp_path / "factors.csv"
    french.write_cache(table, path, start=date(2023, 1, 1), end=date(2024, 12, 31), vintages={})
    return path


def test_the_command_measures_against_the_bill_over_the_days_it_covers(
    tmp_path: Path, capsys
) -> None:
    path = cache(tmp_path)
    factors = factor_cache(tmp_path)
    out = tmp_path / "bills.json"
    argv = ["--cache", str(path), "--factors", str(factors), "--json", str(out)]
    assert main([*argv, "--risk-free", "bills", "--split", "2023-03-31"]) == 0
    printed = capsys.readouterr().out
    assert "499 daily returns, 2023-01-03 to 2024-11-29; risk-free rate one-month " in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["risk_free"] == "one-month Treasury bill"
    # The constant rate picks the tangency portfolio; there are no expected ratios here.
    assert f"(tangency portfolio at its mean, {written['constant_rate']:.2%})" in printed
    assert "expected ratios" not in printed
    assert written["factors_sha256"] == french.read_manifest(factors)["sha256"]
    rate = french.daily_risk_free(french.load_factors(factors)).loc["2023-01-03":]
    daily = total_returns(load_cache(path)).loc[: rate.index[-1]]
    years = (rate.index[-1] - pd.Timestamp("2023-01-02")).days / 365.25
    assert written["risk_free_compound"] == pytest.approx(
        (1 + rate).prod() ** (1 / years) - 1, rel=1e-9
    )
    assert written["constant_rate"] == pytest.approx((1 + rate.mean()) ** 252 - 1, rel=1e-12)
    expected, covariance = fr.estimates(daily)
    tangency = fr.max_sharpe(expected, covariance, written["constant_rate"])
    np.testing.assert_allclose(
        pd.Series(written["portfolios"]["max_sharpe"]["weights"]), tangency, atol=1e-12
    )
    # In sample, every Sharpe ratio is against the bill day by day.
    for name, p in written["portfolios"].items():
        history = portfolio(daily, pd.Series(p["weights"]))
        assert p["in_sample"]["sharpe_ratio"] == pytest.approx(
            sharpe_ratio(history, 252, rate), rel=1e-12
        ), name
    split = written["splits"][0]
    before, after = daily.loc[:"2023-03-31"], daily.loc["2023-04-01":]
    rate_before = (1 + rate.loc[before.index].mean()) ** 252 - 1
    assert split["constant_rate"] == pytest.approx(rate_before, rel=1e-12)
    early = fr.estimates(before)
    held = fr.max_sharpe(*early, rate_before)
    entry = split["portfolios"]["max_sharpe"]
    np.testing.assert_allclose(pd.Series(entry["weights"]), held, atol=1e-12)
    assert not np.allclose(held, fr.max_sharpe(*early), atol=1e-3)
    assert not np.allclose(held, fr.max_sharpe(*early, written["constant_rate"]), atol=1e-3)
    history = portfolio(after, held)
    assert entry["held"]["sharpe_ratio"] == pytest.approx(
        sharpe_ratio(history, 252, rate.loc[after.index]), rel=1e-12
    )
    # After costs too, against the bill day by day.
    assert entry["held"]["costs"] == costs.held(
        after, entry["weights"], before.index[-1], rate.loc[after.index]
    )
    assert entry["in_sample"]["sharpe_ratio"] == pytest.approx(
        sharpe_ratio(portfolio(before, held), 252, rate.loc[before.index]), rel=1e-12
    )
    # The whole history's tangency portfolio, over that quarter alone.
    quarter = prices()
    quarter = quarter[quarter["date"] <= date(2023, 3, 31)]
    result = construction.run(quarter, rate, splits=())
    np.testing.assert_allclose(
        pd.Series(result["portfolios"]["max_sharpe"]["weights"]), held, atol=1e-12
    )
    assert result["constant_rate"] == pytest.approx(rate_before, rel=1e-12)
    # Beside it, the same at a zero rate over the same days.
    zero = written["at_zero_rate"]
    assert (zero["first_return"], zero["last_return"]) == ("2023-01-03", "2024-11-29")
    assert zero["risk_free"] == 0.0
    for name, p in zero["portfolios"].items():
        history = portfolio(daily, pd.Series(p["weights"]))
        assert p["in_sample"]["sharpe_ratio"] == pytest.approx(
            sharpe_ratio(history, 252), rel=1e-12
        ), name
    # Rules that do not use the rate pick the same weights at either rate.
    for name in ("equal", "inverse_volatility", "risk_parity", "minimum_variance"):
        assert zero["portfolios"][name]["weights"] == written["portfolios"][name]["weights"]


def test_a_constant_rate_adds_nothing_to_the_output(tmp_path: Path) -> None:
    out = tmp_path / "zero.json"
    argv = ["--cache", str(cache(tmp_path)), "--json", str(out), "--split", "2023-12-29"]
    assert main(argv) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["risk_free"] == 0.0 and written["last_return"] == "2024-12-31"
    for key in ("risk_free_compound", "constant_rate", "factors_sha256"):
        assert key not in written
    assert "constant_rate" not in written["splits"][0]


def test_bills_the_cache_lacks_or_that_miss_a_day_are_refused_plainly(
    tmp_path: Path, capsys
) -> None:
    path = cache(tmp_path)
    argv = ["--cache", str(path), "--split", "none", "--risk-free", "bills"]
    assert main([*argv, "--factors", str(tmp_path / "none.csv")]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "none.manifest.json is missing" in captured.err
    factors = factor_cache(tmp_path)
    table = pd.read_csv(factors, dtype={"date": str})
    table = table[table["date"] != "2023-06-14"]
    table["date"] = [date.fromisoformat(d) for d in table["date"]]
    french.write_cache(table, factors, start=date(2023, 1, 1), end=date(2024, 12, 31), vintages={})
    assert main([*argv, "--factors", str(factors)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "2023-06-14 is in the returns only" in captured.err
    with pytest.raises(SystemExit):
        main([*argv[:-1], "Bills"])
    assert "a compound annual rate such as 0.02, or bills, not 'Bills'" in capsys.readouterr().err


def test_a_split_past_the_bill_s_last_day_says_where_the_returns_end(
    tmp_path: Path, capsys
) -> None:
    # The prices run to 2024-12-31, the factor cache, and so the returns, to 2024-11-29.
    argv = ["--cache", str(cache(tmp_path)), "--factors", str(factor_cache(tmp_path))]
    assert main([*argv, "--risk-free", "bills", "--split", "2024-12-02"]) == 1
    err = capsys.readouterr().err
    assert "leaves returns on 499 days before it and 0 after" in err
    assert "(the returns run from 2023-01-03 to 2024-11-29)" in err
