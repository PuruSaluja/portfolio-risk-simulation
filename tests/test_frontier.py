import itertools
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import frontier as fr
from portfolio_risk.data import COLUMNS, load_cache, write_cache
from portfolio_risk.frontier import main
from portfolio_risk.montecarlo import historical_portfolio_returns
from portfolio_risk.returns import total_returns
from portfolio_risk.risk import sharpe_ratio


def problem(n=5, seed=0, spread=0.08):
    """Random expected returns and a positive definite covariance for ``n`` funds."""
    rng = np.random.default_rng(seed)
    names = [f"F{i}" for i in range(n)]
    loadings = rng.normal(size=(n, n + 3)) * 0.1
    matrix = loadings @ loadings.T + np.diag(rng.uniform(0.001, 0.02, n))
    mu = pd.Series(rng.uniform(0.0, spread, n), index=names)
    return mu, pd.DataFrame(matrix, index=names, columns=names)


def brute_force(mu, covariance, cap, target=None):
    """The lowest variance over every pattern of funds at 0, at the cap or free."""
    s = covariance.to_numpy()
    m = mu.to_numpy()
    n = len(m)
    rows, rhs = [np.ones(n)], [1.0]
    if target is not None:
        # Measured from the lowest return over the range, so that returns a hair apart
        # still give a row the feasibility check can tell from the row of ones.
        width = m.max() - m.min()
        rows.append((m - m.min()) / width)
        rhs.append((target - m.min()) / width)
    best, best_w = math.inf, None
    for pattern in itertools.product((0, 1, 2), repeat=n):
        pattern = np.array(pattern)
        if cap == 1.0 and (pattern == 1).any():
            continue
        free = pattern == 2
        w = np.where(pattern == 1, cap, 0.0)
        a = np.array(rows)
        b = np.array(rhs) - a @ w
        k = int(free.sum())
        if k == 0:
            if np.allclose(b, 0.0, atol=1e-12):
                variance = w @ s @ w
                if variance < best:
                    best, best_w = variance, w
            continue
        kkt = np.block(
            [[s[np.ix_(free, free)], a[:, free].T], [a[:, free], np.zeros((len(b),) * 2)]]
        )
        try:
            # Funds held at the cap pull on the free ones through the covariance.
            pull = -s[np.ix_(free, ~free)] @ w[~free]
            solution = np.linalg.lstsq(kkt, np.concatenate([pull, b]), rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        w[free] = solution[:k]
        feasible = (
            (w >= -1e-12).all() and (w <= cap + 1e-12).all() and np.allclose(a @ w, rhs, atol=1e-10)
        )
        if feasible and w @ s @ w < best:
            best, best_w = w @ s @ w, w
    return best, best_w


def best_sharpe_by_support(mu, covariance, rate=0.0):
    """The highest Sharpe ratio of long-only portfolios: for each set of funds held, the
    unconstrained tangency portfolio on them, where all its weights are positive."""
    s = covariance.to_numpy()
    excess = mu.to_numpy() - rate
    n = len(excess)
    best, best_w = -math.inf, None
    for size in range(1, n + 1):
        for support in itertools.combinations(range(n), size):
            idx = list(support)
            y = np.linalg.solve(s[np.ix_(idx, idx)], excess[idx])
            if (y <= 0).all():
                y = -y
            if not (y > 0).all():
                continue
            w = np.zeros(n)
            w[idx] = y / y.sum()
            ratio = (w @ excess) / math.sqrt(w @ s @ w)
            if ratio > best:
                best, best_w = ratio, w
    return best, best_w


def check_feasible(w, cap=1.0):
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert (w >= 0.0).all()
    assert (w <= cap).all()


def test_two_funds_minimum_variance_matches_the_closed_form() -> None:
    covariance = pd.DataFrame([[0.04, 0.006], [0.006, 0.01]], index=["A", "B"], columns=["A", "B"])
    w = fr.minimum_variance(covariance)
    a = (0.01 - 0.006) / (0.04 + 0.01 - 2 * 0.006)
    assert w.tolist() == pytest.approx([a, 1 - a], abs=1e-14)


def test_an_interior_minimum_variance_portfolio_is_the_inverse_covariance_times_one() -> None:
    mu, covariance = problem(4, seed=3)
    inverse = np.linalg.solve(covariance.to_numpy(), np.ones(4))
    expected = inverse / inverse.sum()
    assert (expected > 0).all()
    assert fr.minimum_variance(covariance).to_numpy() == pytest.approx(expected, abs=1e-13)


def test_an_interior_efficient_portfolio_matches_merton_s_closed_form() -> None:
    mu, covariance = problem(4, seed=3)
    s, m, one = covariance.to_numpy(), mu.to_numpy(), np.ones(4)
    inv_one, inv_mu = np.linalg.solve(s, one), np.linalg.solve(s, m)
    a, b, c = one @ inv_one, one @ inv_mu, m @ inv_mu
    least = inv_one / a
    target = least @ m + 0.002
    w = ((c - b * target) * inv_one + (a * target - b) * inv_mu) / (a * c - b * b)
    assert (w > 0).all()
    got = fr.efficient_portfolio(mu, covariance, target)
    assert got.to_numpy() == pytest.approx(w, abs=1e-12)
    assert fr.moments(got, mu, covariance)[0] == pytest.approx(target, abs=1e-15)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("cap", [1.0, 0.35])
def test_minimum_variance_is_the_lowest_over_every_pattern_of_bounds(seed, cap) -> None:
    mu, covariance = problem(5, seed)
    best, best_w = brute_force(mu, covariance, cap)
    w = fr.minimum_variance(covariance, cap)
    check_feasible(w, cap)
    assert w @ covariance @ w == pytest.approx(best, rel=1e-12)
    assert w.to_numpy() == pytest.approx(best_w, abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("cap", [1.0, 0.35])
def test_efficient_portfolios_are_the_lowest_variance_at_their_target(seed, cap) -> None:
    mu, covariance = problem(5, seed)
    low, high = fr.return_range(mu, cap)
    for share in (0.0, 0.1, 0.37, 0.5, 0.81, 1.0):
        target = low + share * (high - low)
        w = fr.efficient_portfolio(mu, covariance, target, cap)
        check_feasible(w, cap)
        assert w @ mu == pytest.approx(target, abs=1e-13)
        best, _ = brute_force(mu, covariance, cap, target)
        assert w @ covariance @ w == pytest.approx(best, rel=1e-11)


def test_the_return_range_fills_the_best_funds_up_to_the_cap() -> None:
    mu = pd.Series([0.05, 0.10, 0.02, 0.07], index=list("ABCD"))
    assert fr.return_range(mu) == pytest.approx((0.02, 0.10))
    assert fr.return_range(mu, 0.4) == pytest.approx(
        (0.4 * 0.02 + 0.4 * 0.05 + 0.2 * 0.07, 0.4 * 0.10 + 0.4 * 0.07 + 0.2 * 0.05)
    )


@pytest.mark.parametrize("seed", range(8))
def test_the_tangency_portfolio_is_the_best_over_every_set_of_funds(seed) -> None:
    mu, covariance = problem(5, seed)
    best, best_w = best_sharpe_by_support(mu, covariance)
    w = fr.max_sharpe(mu, covariance)
    check_feasible(w)
    assert fr.sharpe(w, mu, covariance) == pytest.approx(best, rel=1e-12)
    assert w.to_numpy() == pytest.approx(best_w, abs=1e-12)


def test_the_tangency_portfolio_takes_the_risk_free_rate_as_a_compound_annual_rate() -> None:
    mu, covariance = problem(5, seed=1)
    rate = (1.03 ** (1 / 252) - 1) * 252
    best, best_w = best_sharpe_by_support(mu, covariance, rate)
    w = fr.max_sharpe(mu, covariance, risk_free=0.03)
    assert fr.sharpe(w, mu, covariance, risk_free=0.03) == pytest.approx(best, rel=1e-12)
    assert w.to_numpy() == pytest.approx(best_w, abs=1e-12)
    assert not np.allclose(w, fr.max_sharpe(mu, covariance))


@pytest.mark.parametrize("seed", range(4))
def test_a_capped_tangency_portfolio_beats_every_capped_portfolio_tried(seed) -> None:
    mu, covariance = problem(5, seed)
    cap = 0.3
    w = fr.max_sharpe(mu, covariance, max_weight=cap)
    check_feasible(w, cap)
    best = fr.sharpe(w, mu, covariance)
    rng = np.random.default_rng(seed)
    tried = 0
    for draw in rng.dirichlet(np.ones(5), 20_000):
        if draw.max() <= cap:
            tried += 1
            assert fr.sharpe(pd.Series(draw, index=mu.index), mu, covariance) <= best + 1e-12
    assert tried > 500
    for _, row in fr.frontier(mu, covariance, 41, cap).iterrows():
        assert (row["expected_return"]) / row["volatility"] <= best + 1e-12


def test_a_tangency_portfolio_at_the_end_of_the_frontier_is_found_exactly() -> None:
    # A dominates: higher return, lower risk, and B is nearly the same bet.
    covariance = pd.DataFrame([[0.01, 0.018], [0.018, 0.04]], index=["A", "B"], columns=["A", "B"])
    mu = pd.Series([0.10, 0.01], index=["A", "B"])
    assert fr.max_sharpe(mu, covariance).tolist() == [1.0, 0.0]


def test_the_ratio_is_the_sharpe_ratio_of_the_portfolio_s_daily_returns() -> None:
    rng = np.random.default_rng(5)
    daily = pd.DataFrame(
        rng.normal(0.0004, 0.01, (1_500, 3)) + rng.normal(0, 0.005, (1_500, 1)),
        index=pd.bdate_range("2015-01-01", periods=1_500),
        columns=["A", "B", "C"],
    )
    mu, covariance = fr.estimates(daily)
    for w in (fr.max_sharpe(mu, covariance), fr.minimum_variance(covariance)):
        history = historical_portfolio_returns(daily, w.to_dict())
        assert fr.sharpe(w, mu, covariance) == pytest.approx(sharpe_ratio(history, 252), rel=1e-12)
        assert fr.sharpe(w, mu, covariance, 0.02) == pytest.approx(
            sharpe_ratio(history, 252, 0.02), rel=1e-12
        )
    best = fr.sharpe(fr.max_sharpe(mu, covariance), mu, covariance)
    for fund in daily.columns:
        assert sharpe_ratio(daily[fund], 252) <= best


def test_estimates_are_yearly_means_and_covariances() -> None:
    rng = np.random.default_rng(2)
    daily = pd.DataFrame(rng.normal(0.0003, 0.01, (400, 2)), columns=["A", "B"])
    mu, covariance = fr.estimates(daily)
    assert mu.to_numpy() == pytest.approx(daily.mean().to_numpy() * 252, rel=1e-14)
    assert covariance.to_numpy() == pytest.approx(daily.cov().to_numpy() * 252, rel=1e-12)
    _, shrunk = fr.estimates(daily, "ledoit-wolf")
    assert not np.allclose(shrunk, covariance)
    with pytest.raises(ValueError, match="sample or ledoit-wolf"):
        fr.estimates(daily, "shrunk")


def test_the_frontier_runs_from_minimum_variance_to_the_highest_return() -> None:
    mu, covariance = problem(6, seed=4)
    table = fr.frontier(mu, covariance, 25)
    assert len(table) == 25
    least = fr.minimum_variance(covariance)
    assert table.iloc[0][mu.index].to_numpy() == pytest.approx(least.to_numpy(), abs=1e-15)
    assert table["expected_return"].iloc[-1] == pytest.approx(mu.max(), abs=1e-14)
    returns = table["expected_return"].to_numpy()
    variances = table["volatility"].to_numpy() ** 2
    assert (np.diff(returns) > 0).all()
    assert (np.diff(variances) > 0).all()
    # Variance is convex in the target return: the gaps between points widen.
    assert (np.diff(variances, 2) >= -1e-15).all()
    for _, row in table.iterrows():
        check_feasible(row[mu.index].to_numpy(dtype="float64"))


def test_a_cap_of_one_over_n_leaves_only_equal_weights() -> None:
    mu, covariance = problem(3, seed=0)
    w = fr.minimum_variance(covariance, 1 / 3)
    assert w.to_numpy() == pytest.approx(np.full(3, 1 / 3), abs=1e-15)
    assert fr.max_sharpe(mu, covariance, max_weight=1 / 3).to_numpy() == pytest.approx(
        np.full(3, 1 / 3), abs=1e-12
    )


def test_one_fund_is_the_whole_portfolio() -> None:
    covariance = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    mu = pd.Series([0.05], index=["A"])
    assert fr.minimum_variance(covariance).tolist() == [1.0]
    assert fr.max_sharpe(mu, covariance).tolist() == [1.0]
    assert fr.efficient_portfolio(mu, covariance, 0.05).tolist() == [1.0]
    assert len(fr.frontier(mu, covariance, 3)) == 3


def test_equal_expected_returns_leave_the_minimum_variance_portfolio() -> None:
    _, covariance = problem(4, seed=2)
    mu = pd.Series(0.05, index=covariance.index)
    least = fr.minimum_variance(covariance)
    assert fr.efficient_portfolio(mu, covariance, 0.05).equals(least)
    assert fr.max_sharpe(mu, covariance).to_numpy() == pytest.approx(least.to_numpy(), abs=1e-15)


def test_a_singular_covariance_is_refused() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.02, 0.0], [0.02, 0.01, 0.0], [0.0, 0.0, 0.01]],
        index=list("ABC"),
        columns=list("ABC"),
    )
    with pytest.raises(ValueError, match="not positive definite"):
        fr.minimum_variance(covariance)


@pytest.mark.parametrize("cap", [0.0, -0.1, 1.5, math.nan, True, "0.5", 0.19])
def test_a_cap_the_weights_cannot_meet_is_refused(cap) -> None:
    _, covariance = problem(5)
    with pytest.raises(ValueError, match="max_weight|cannot add up"):
        fr.minimum_variance(covariance, cap)


def test_a_target_no_portfolio_reaches_is_refused() -> None:
    mu, covariance = problem(4)
    low, high = fr.return_range(mu)
    for target in (low - 0.001, high + 0.001, math.nan):
        with pytest.raises(ValueError, match="no portfolio has an expected return"):
            fr.efficient_portfolio(mu, covariance, target)
    with pytest.raises(ValueError, match="must be a number"):
        fr.efficient_portfolio(mu, covariance, "0.05")


def test_a_tangency_portfolio_needs_a_return_above_the_risk_free_rate() -> None:
    mu, covariance = problem(4)
    with pytest.raises(ValueError, match="above the risk-free rate"):
        fr.max_sharpe(mu, covariance, risk_free=float(mu.max()) + 0.01)
    with pytest.raises(ValueError, match="annual rate"):
        fr.max_sharpe(mu, covariance, risk_free=-1.0)


def test_inputs_naming_other_funds_or_not_finite_are_refused() -> None:
    mu, covariance = problem(3)
    with pytest.raises(ValueError, match="same funds"):
        fr.max_sharpe(mu.rename(lambda t: t + "x"), covariance)
    with pytest.raises(ValueError, match="NaN or infinite"):
        fr.max_sharpe(mu.where(mu.index != "F1", math.nan), covariance)
    with pytest.raises(TypeError, match="Series by fund"):
        fr.max_sharpe(mu.to_numpy(), covariance)
    with pytest.raises(ValueError, match="same funds"):
        fr.moments(pd.Series(1 / 3, index=list("XYZ")), mu, covariance)
    for points in (1, 2.0, True):
        with pytest.raises(ValueError, match="points"):
            fr.frontier(mu, covariance, points)


@pytest.mark.parametrize("spread", [1e-11, 1e-9, 1e-6])
def test_expected_returns_a_hair_apart_still_give_the_exact_portfolios(spread) -> None:
    # The row of returns is then nearly parallel to the row of ones.
    _, covariance = problem(4, seed=2)
    mu = pd.Series(0.05 + spread * np.arange(4.0), index=covariance.index)
    low, high = fr.return_range(mu)
    for share in (0.0, 0.5, 1.0):
        target = low + share * (high - low)
        best, _ = brute_force(mu, covariance, 1.0, target)
        w = fr.efficient_portfolio(mu, covariance, target)
        assert w @ covariance @ w == pytest.approx(best, rel=1e-9)
    best, best_w = best_sharpe_by_support(mu, covariance)
    # Returns equal to ten digits differ by numbers carrying their own rounding, about
    # 1e-17 in 1e-11, so the weights can be no closer than about 1e-10.
    assert fr.max_sharpe(mu, covariance).to_numpy() == pytest.approx(best_w, abs=1e-9)


@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("n", [5, 8])
def test_weights_at_a_bound_are_exactly_at_it(seed, n) -> None:
    # A step to the cap lands on it or a last bit either side, depending on the numbers.
    mu, covariance = problem(n, seed)
    for cap in (1.0, 0.3, 0.45):
        for w in (fr.minimum_variance(covariance, cap), fr.max_sharpe(mu, covariance, 0.0, cap)):
            x = w.to_numpy()
            assert (x[x < 1e-9] == 0.0).all()
            assert (x[x > cap - 1e-9] == cap).all()


def test_a_target_a_rounding_error_past_either_end_gets_that_end() -> None:
    mu, covariance = problem(4, seed=1)
    low, high = fr.return_range(mu)
    assert fr.efficient_portfolio(mu, covariance, high + 5e-13).tolist() == [0.0, 0.0, 0.0, 1.0]
    assert fr.efficient_portfolio(mu, covariance, low - 5e-13).tolist() == [0.0, 0.0, 1.0, 0.0]


def prices(flat: str | None = None) -> pd.DataFrame:
    days = pd.bdate_range("2023-01-02", "2024-12-31")
    rng = np.random.default_rng(0)
    rows = []
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        close = 50.0 * np.cumprod(1 + rng.normal(0.0002 * (i + 1), 0.006 * (1 + i), len(days)))
        if ticker == flat:
            close[:] = 50.0
        for day, value in zip(days, close, strict=True):
            rows.append((day.date(), ticker, float(value), 0.0, 0.0, 0.0, 1e6))
    return pd.DataFrame(rows, columns=list(COLUMNS))


def cache(tmp_path: Path, flat: str | None = None) -> Path:
    path = tmp_path / "prices.csv"
    write_cache(prices(flat), path, start=date(2023, 1, 1), end=date(2024, 12, 31), source="t")
    return path


def test_the_command_prints_and_writes_the_frontier_and_the_splits(tmp_path: Path, capsys) -> None:
    out = tmp_path / "results" / "frontier.json"
    path = cache(tmp_path)
    argv = ["--cache", str(path), "--json", str(out), "--split", "2023-12-29"]
    assert main([*argv, "--split", "2024-06-28", "--points", "5"]) == 0
    printed = capsys.readouterr().out
    assert "chosen from 2023-01-03 to 2023-12-29, held from 2024-01-01 to 2024-12-31" in printed
    assert b"\r" not in out.read_bytes()
    written = json.loads(out.read_text(encoding="utf-8"))
    # From the cache, as the command reads it: parsing the CSV can change a close's last bit.
    daily = total_returns(load_cache(path))
    expected, covariance = fr.estimates(daily)
    least = fr.minimum_variance(covariance)
    tangency = fr.max_sharpe(expected, covariance)
    portfolios = written["portfolios"]
    assert list(portfolios) == ["minimum_variance", "max_sharpe", "equal"]
    assert portfolios["minimum_variance"]["weights"] == least.to_dict()
    assert portfolios["max_sharpe"]["weights"] == tangency.to_dict()
    assert portfolios["equal"]["weights"] == {"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3}
    assert portfolios["max_sharpe"]["sharpe_ratio"] == pytest.approx(
        sharpe_ratio(historical_portfolio_returns(daily, tangency.to_dict()), 252), rel=1e-12
    )
    assert written["funds"]["BBB"]["expected_return"] == pytest.approx(daily["BBB"].mean() * 252)
    assert len(written["frontier"]) == 5
    assert written["frontier"][0]["weights"] == pytest.approx(least.to_dict(), abs=1e-15)
    assert written["frontier"][-1]["expected_return"] == pytest.approx(expected.max())
    assert [s["date"] for s in written["splits"]] == ["2023-12-29", "2024-06-28"]
    assert written["splits"][1]["held_on"]["first_return"] == "2024-07-01"
    split = written["splits"][0]
    assert split["estimated_on"] == {
        "first_return": "2023-01-03",
        "last_return": "2023-12-29",
        "daily_returns": 259,
    }
    assert split["held_on"]["first_return"] == "2024-01-01"
    before, after = daily.loc[:"2023-12-29"], daily.loc["2024-01-01":]
    held = fr.max_sharpe(*fr.estimates(before))
    entry = split["portfolios"]["max_sharpe"]
    assert entry["weights"] == held.to_dict()
    history = historical_portfolio_returns(after, held.to_dict())
    assert entry["realised"]["mean"] == pytest.approx(history.mean() * 252, rel=1e-12)
    assert entry["realised"]["sharpe_ratio"] == pytest.approx(sharpe_ratio(history, 252))
    years = (date(2024, 12, 31) - date(2023, 12, 29)).days / 365.25
    assert entry["realised"]["compound_return"] == pytest.approx(
        np.prod(1 + history) ** (1 / years) - 1, rel=1e-12
    )
    assert (
        split["chosen_after"]["max_sharpe"]["weights"]
        == fr.max_sharpe(*fr.estimates(after)).to_dict()
    )
    assert "equal" not in split["chosen_after"]


def test_the_command_takes_a_cap_a_rate_and_the_shrunk_covariance(tmp_path: Path) -> None:
    out = tmp_path / "f.json"
    path = cache(tmp_path)
    assert main(["--cache", str(path), "--json", str(out), "--split", "none"]) == 0
    plain = json.loads(out.read_text(encoding="utf-8"))
    assert plain["splits"] == []
    argv = ["--cache", str(path), "--json", str(out), "--split", "2023-12-29"]
    argv += ["--max-weight", "0.6", "--risk-free", "0.02", "--covariance", "ledoit-wolf"]
    assert main(argv) == 0
    capped = json.loads(out.read_text(encoding="utf-8"))
    assert (capped["max_weight"], capped["risk_free"], capped["estimator"]) == (
        0.6,
        0.02,
        "ledoit-wolf",
    )
    for row in [*capped["frontier"], *capped["portfolios"].values()]:
        assert max(row["weights"].values()) <= 0.6
    daily = total_returns(load_cache(path))
    rate = (1.02 ** (1 / 252) - 1) * 252
    expected, covariance = fr.estimates(daily, "ledoit-wolf")
    for ticker, fund in capped["funds"].items():
        assert fund["volatility"] == pytest.approx(math.sqrt(covariance.loc[ticker, ticker]))
    assert capped["funds"]["BBB"]["volatility"] != plain["funds"]["BBB"]["volatility"]
    for name, w in [
        ("max_sharpe", fr.max_sharpe(expected, covariance, 0.02, 0.6)),
        ("minimum_variance", fr.minimum_variance(covariance, 0.6)),
    ]:
        assert capped["portfolios"][name]["weights"] == w.to_dict()
        assert capped["portfolios"][name]["weights"] != plain["portfolios"][name]["weights"]
    for row in capped["frontier"]:
        assert row["sharpe_ratio"] == pytest.approx(
            (row["expected_return"] - rate) / row["volatility"], rel=1e-12
        )
    # Chosen from the shrunk estimates before the split; expected figures from the same.
    before, after = daily.loc[:"2023-12-29"], daily.loc["2024-01-01":]
    early = fr.estimates(before, "ledoit-wolf")
    entry = capped["splits"][0]["portfolios"]["max_sharpe"]
    held = fr.max_sharpe(*early, 0.02, 0.6)
    assert entry["weights"] == held.to_dict()
    mean, sd = fr.moments(held, *early)
    assert (entry["expected_return"], entry["volatility"]) == pytest.approx((mean, sd), rel=1e-12)
    history = historical_portfolio_returns(after, held.to_dict())
    assert entry["realised"]["sharpe_ratio"] == pytest.approx(
        sharpe_ratio(history, 252, 0.02), rel=1e-12
    )


def test_the_command_refuses_bad_input_plainly(tmp_path: Path, capsys) -> None:
    assert main(["--cache", str(tmp_path / "prices.csv")]) == 1
    assert "manifest.json is missing" in capsys.readouterr().err
    path = cache(tmp_path)
    for argv, message in [
        ([], "on 2009-12-31 leaves returns on 0 days before it"),  # the default splits
        (["--split", "2023-12-29", "--split", "2023-12-29"], "given twice"),
        (["--split", "2023-12-29", "--split", "none"], "or none alone"),
        (["--split", "2024-12-27"], "and 2 after; each side needs more than 3"),
        (["--split", "2023-13-01"], "a split is a date"),
        (["--split", "none", "--max-weight", "0.3"], "cannot add up to 1"),
        (["--split", "none", "--max-weight", "nan"], "max_weight"),
        (["--split", "none", "--points", "1"], "points"),
        (["--split", "none", "--risk-free", "5"], "above the risk-free rate"),
        (["--split", "none", "--json", str(tmp_path)], ""),  # a folder
    ]:
        assert main(["--cache", str(path), *argv]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error: ") and message in captured.err
        assert "Traceback" not in captured.err
    assert main(["--cache", str(cache(tmp_path / "flat", "BBB")), "--split", "none"]) == 1
    assert "BBB has a variance of 0.0" in capsys.readouterr().err


def test_a_cap_short_of_one_over_n_is_refused_however_little() -> None:
    # 0.333333333333 is 1e-12 short of 1/3: three such weights cannot add up to 1, and
    # equal weights would break it.
    mu, covariance = problem(3, seed=0)
    with pytest.raises(ValueError, match="cannot add up to 1"):
        fr.minimum_variance(covariance, 0.333333333333)
    with pytest.raises(ValueError, match="cannot add up to 1"):
        fr.max_sharpe(mu, covariance, max_weight=0.3333333333333)


@pytest.mark.parametrize("n", [3, 7, 49, 98])
def test_a_cap_of_one_over_n_in_floating_point_is_met_exactly(n) -> None:
    # 1/49 times 49 is a last bit below 1 in floating point.
    mu, covariance = problem(n, seed=1)
    cap = 1 / n
    for w in (fr.minimum_variance(covariance, cap), fr.max_sharpe(mu, covariance, 0.0, cap)):
        assert (w <= cap).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-14)
