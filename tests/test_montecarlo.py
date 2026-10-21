import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data import COLUMNS, write_cache
from portfolio_risk.montecarlo import (
    BLOCK,
    NormalModel,
    StudentTModel,
    check_weights,
    describe,
    fit_dof,
    historical_portfolio_returns,
    horizon_returns,
    log_likelihood,
    main,
    parse_weights,
    simulate,
)

TICKERS = ("A", "B", "C")
MEAN = np.array([0.0004, 0.0001, -0.0002])
COVARIANCE = np.array(
    [
        [1.0e-4, 3.0e-5, -1.0e-5],
        [3.0e-5, 4.0e-5, 0.0],
        [-1.0e-5, 0.0, 2.25e-4],
    ]
)


def model() -> NormalModel:
    return NormalModel(TICKERS, MEAN, COVARIANCE)


def daily_returns(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.multivariate_normal([0.0005, 0.0002, 0.0], COVARIANCE, size=n)
    return pd.DataFrame(
        values, columns=list(TICKERS), index=pd.bdate_range("2020-01-01", periods=n)
    )


def test_fit_takes_the_mean_and_covariance_of_the_log_returns() -> None:
    returns = daily_returns()
    fitted = NormalModel.fit(returns)
    logs = np.log1p(returns.to_numpy())
    np.testing.assert_allclose(fitted.mean, logs.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(fitted.covariance, np.cov(logs.T), rtol=1e-12)
    assert fitted.tickers == TICKERS


def test_drawn_log_returns_have_the_model_s_mean_and_covariance() -> None:
    draws = np.log1p(model().sample(np.random.default_rng(1), 2_000, 100)).reshape(-1, 3)
    n = len(draws)
    sd = np.sqrt(np.diag(COVARIANCE))
    # Within four standard errors of the mean, and of each variance and covariance.
    assert np.all(np.abs(draws.mean(axis=0) - MEAN) < 4 * sd / math.sqrt(n))
    se = np.sqrt((COVARIANCE**2 + np.outer(sd**2, sd**2)) / n)
    assert np.all(np.abs(np.cov(draws.T) - COVARIANCE) < 4 * se)


def test_days_are_drawn_independently() -> None:
    draws = np.log1p(model().sample(np.random.default_rng(2), 4_000, 50))[:, :, 0]
    d = draws - draws.mean()
    lag1 = np.sum(d[:, :-1] * d[:, 1:]) / np.sum(d * d)
    assert abs(lag1) < 4 / math.sqrt(d.size)


def test_expected_wealth_matches_the_lognormal_formula() -> None:
    # Rebalanced daily, a day's expected growth is sum(w * exp(mu + var / 2)), so over h
    # independent days the expected wealth is that to the power h.
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    w = np.array([0.5, 0.3, 0.2])
    horizon = 63
    sim = simulate(model(), weights, horizon, 20_000, seed=3)
    wealth = 1.0 + sim.horizon_returns
    expected = float(w @ np.exp(MEAN + np.diag(COVARIANCE) / 2)) ** horizon
    assert abs(wealth.mean() - expected) < 4 * wealth.std(ddof=1) / math.sqrt(len(wealth))


def test_a_single_fund_compounds_to_a_lognormal() -> None:
    sim = simulate(model(), {"C": 1.0}, 252, 20_000, seed=4)
    logs = np.log1p(sim.horizon_returns)
    mean, var = 252 * MEAN[2], 252 * COVARIANCE[2, 2]
    assert abs(logs.mean() - mean) < 4 * math.sqrt(var / len(logs))
    assert logs.var(ddof=1) == pytest.approx(var, rel=4 * math.sqrt(2 / len(logs)))


def test_a_day_s_portfolio_return_is_the_weighted_sum_of_the_funds() -> None:
    weights = {"A": 0.25, "B": 0.75}
    sim = simulate(model(), weights, 10, 5, seed=5)
    funds = model().sample(np.random.default_rng(np.random.SeedSequence(5).spawn(1)[0]), 5, 10)
    np.testing.assert_allclose(sim.returns, funds @ np.array([0.25, 0.75, 0.0]), rtol=1e-15)
    assert sim.weights == {"A": 0.25, "B": 0.75, "C": 0.0}
    np.testing.assert_allclose(
        sim.horizon_returns, np.prod(1 + sim.returns, axis=1) - 1, rtol=1e-12
    )


def test_the_seed_fixes_every_path() -> None:
    a = simulate(model(), {"A": 1.0}, 5, 10, seed=7).returns
    b = simulate(model(), {"A": 1.0}, 5, 10, seed=7).returns
    c = simulate(model(), {"A": 1.0}, 5, 10, seed=8).returns
    assert (a == b).all()
    assert not np.isclose(a, c).any()


def test_a_path_is_the_same_whatever_number_of_paths_is_asked_for() -> None:
    few = simulate(model(), {"A": 0.5, "C": 0.5}, 7, BLOCK + 3, seed=9).returns
    many = simulate(model(), {"A": 0.5, "C": 0.5}, 7, 2 * BLOCK + 50, seed=9).returns
    assert few.shape == (BLOCK + 3, 7) and many.shape == (2 * BLOCK + 50, 7)
    assert (few == many[: BLOCK + 3]).all()
    # Every path is drawn, none left at the buffer's initial contents.
    assert len(np.unique(many[:, 0])) == len(many)


def test_a_singular_covariance_still_draws() -> None:
    # B is A doubled and C never varies: rank one, no Cholesky factor.
    cov = np.array([[1e-4, 2e-4, 0.0], [2e-4, 4e-4, 0.0], [0.0, 0.0, 0.0]])
    draws = np.log1p(NormalModel(TICKERS, np.zeros(3), cov).sample(np.random.default_rng(0), 50, 4))
    np.testing.assert_allclose(draws[..., 1], 2 * draws[..., 0], rtol=1e-9, atol=1e-15)
    assert (draws[..., 2] == 0.0).all()
    assert draws[..., 0].std() > 0


def test_a_fund_that_never_varied_is_simulated_as_never_varying() -> None:
    returns = daily_returns(n=50)
    returns["B"] = 0.0001
    draws = NormalModel.fit(returns).sample(np.random.default_rng(0), 20, 30)
    np.testing.assert_allclose(draws[..., 1], 0.0001, rtol=1e-12)


@pytest.mark.parametrize(
    ("tickers", "mean", "cov", "message"),
    [
        (("A", "B", "C"), MEAN, -COVARIANCE, "not positive semi-definite"),
        (("A", "B", "C"), MEAN, COVARIANCE + np.triu(np.full((3, 3), 1e-6), 1), "not symmetric"),
        (("A", "B", "C"), MEAN[:2], COVARIANCE, "shape"),
        (("A", "B", "C"), MEAN, COVARIANCE[:2, :2], "shape"),
        (("A", "A", "C"), MEAN, COVARIANCE, "each named once"),
        ((), np.zeros(0), np.zeros((0, 0)), "at least one fund"),
        (("A", "B", "C"), np.array([0.0, np.nan, 0.0]), COVARIANCE, "finite"),
    ],
)
def test_a_model_that_cannot_be_drawn_from_is_refused(tickers, mean, cov, message) -> None:
    with pytest.raises(ValueError, match=message):
        NormalModel(tickers, mean, cov)


def test_fit_refuses_returns_it_cannot_use() -> None:
    returns = daily_returns(n=10)
    with pytest.raises(TypeError, match="DataFrame"):
        NormalModel.fit(returns["A"])
    with pytest.raises(ValueError, match="at least 2 days"):
        NormalModel.fit(returns.iloc[:1])
    bad = returns.copy()
    bad.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        NormalModel.fit(bad)
    bad.iloc[3, 1] = -1.0
    with pytest.raises(ValueError, match="-100%"):
        NormalModel.fit(bad)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"A": 0.5, "D": 0.5}, "no returns for D"),
        ({"A": 1.5, "B": -0.5}, "not negative"),
        ({"A": 0.5, "B": 0.4}, "add up to 1"),
        ({"A": math.nan, "B": 1.0}, "finite"),
    ],
)
def test_weights_are_long_only_and_add_up_to_one(weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        check_weights(weights, TICKERS)


def test_funds_left_out_of_the_weights_get_none() -> None:
    np.testing.assert_array_equal(check_weights({"C": 0.6, "A": 0.4}, TICKERS), [0.4, 0.0, 0.6])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"horizon": 0}, "horizon"),
        ({"paths": 0}, "paths"),
        ({"paths": True}, "paths"),
        ({"horizon": 2.0}, "horizon"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.5}, "seed"),
    ],
)
def test_simulate_refuses_counts_that_are_not_whole_numbers(kwargs, message) -> None:
    args = {"horizon": 5, "paths": 5, "seed": 0} | kwargs
    with pytest.raises(ValueError, match=message):
        simulate(model(), {"A": 1.0}, **args)


def test_parse_weights() -> None:
    assert parse_weights("equal", TICKERS) == {t: 1 / 3 for t in TICKERS}
    assert parse_weights("a=0.6, C=0.4", TICKERS) == {"A": 0.6, "C": 0.4}
    for text, message in [
        ("A=0.6,A=0.4", "given twice"),
        ("A=0.6,C", "look like"),
        ("A=0.6,C=x", "not a number"),
        ("A=0.6,C=0.3", "add up to 1"),
        ("D=1", "no returns for D"),
    ]:
        with pytest.raises(ValueError, match=message):
            parse_weights(text, TICKERS)


def test_historical_portfolio_returns_are_the_weighted_sum() -> None:
    returns = daily_returns(n=30)
    history = historical_portfolio_returns(returns, {"A": 0.2, "B": 0.8})
    np.testing.assert_allclose(history, 0.2 * returns["A"] + 0.8 * returns["B"], rtol=1e-15)
    assert (history.index == returns.index).all()


def test_horizon_returns_compound_every_window() -> None:
    daily = daily_returns(n=40)["A"]
    windows = (1 + daily).rolling(10).apply(np.prod, raw=True).dropna().to_numpy() - 1
    np.testing.assert_allclose(horizon_returns(daily, 10), windows, rtol=1e-10)
    assert len(horizon_returns(daily, 40)) == 1
    with pytest.raises(ValueError, match="at least 41"):
        horizon_returns(daily, 41)


def test_describe() -> None:
    values = np.array([-0.02, -0.01, 0.0, 0.01, 0.05])
    d = describe(values)
    assert d["count"] == 5
    assert d["mean"] == pytest.approx(0.006)
    assert d["sd"] == pytest.approx(values.std(ddof=1))
    assert d["probability_of_loss"] == pytest.approx(0.4)
    assert d["q50"] == 0.0
    assert d["q1"] == pytest.approx(np.quantile(values, 0.01))
    dev = values - values.mean()
    m2 = np.mean(dev**2)
    assert d["skewness"] == pytest.approx(np.mean(dev**3) / m2**1.5)
    assert d["excess_kurtosis"] == pytest.approx(np.mean(dev**4) / m2**2 - 3)
    with pytest.raises(ValueError, match="at least 2 values"):
        describe(np.array([0.01]))
    flat = describe(np.full(4, 0.01))
    assert flat["sd"] == 0.0 and flat["skewness"] == 0.0 and flat["excess_kurtosis"] == 0.0


def test_normal_draws_have_neither_skew_nor_excess_kurtosis() -> None:
    d = describe(np.log1p(model().sample(np.random.default_rng(6), 1_000, 200))[..., 0])
    n = d["count"]
    assert abs(d["skewness"]) < 4 * math.sqrt(6 / n)
    assert abs(d["excess_kurtosis"]) < 4 * math.sqrt(24 / n)


def t_model(dof=4.0) -> StudentTModel:
    return StudentTModel(TICKERS, MEAN, COVARIANCE, dof)


def test_t_draws_keep_the_model_s_covariance() -> None:
    draws = np.log1p(t_model(10.0).sample(np.random.default_rng(11), 2_000, 100)).reshape(-1, 3)
    n = len(draws)
    sd = np.sqrt(np.diag(COVARIANCE))
    assert np.all(np.abs(draws.mean(axis=0) - MEAN) < 4 * sd / math.sqrt(n))
    # A t with 10 degrees of freedom has excess kurtosis 1, which widens the standard error
    # of a variance by about a fifth over the normal's.
    se = np.sqrt((COVARIANCE**2 + np.outer(sd**2, sd**2)) / n)
    assert np.all(np.abs(np.cov(draws.T) - COVARIANCE) < 6 * se)


def test_t_draws_follow_the_t_distribution() -> None:
    # Standardised to the t's own scale, a fund's draws are t with 4 degrees of freedom,
    # whose distribution function has a closed form: P(T > 2) = 0.0580582.
    dof = 4.0
    draws = np.log1p(t_model(dof).sample(np.random.default_rng(12), 2_000, 100))[..., 1]
    t = (draws - MEAN[1]) / math.sqrt(COVARIANCE[1, 1] * (dof - 2) / dof)
    x = 2.0 / math.sqrt(1 + 2.0**2 / 4)
    tail = 0.5 - 3 / 8 * x * (1 - 2.0**2 / (12 * (1 + 2.0**2 / 4)))
    assert tail == pytest.approx(0.0580582, abs=1e-7)
    share = np.mean(t > 2.0)
    assert abs(share - tail) < 4 * math.sqrt(tail * (1 - tail) / t.size)


def test_every_fund_shares_a_day_s_t_mixing_draw() -> None:
    # With independent funds, a day far out in one fund's tail is far out in the
    # others' more often than independence would allow: the mixing draw is shared.
    cov = np.diag([1e-4, 1e-4, 1e-4])
    draws = np.log1p(
        StudentTModel(TICKERS, np.zeros(3), cov, 3.0).sample(np.random.default_rng(13), 1_000, 200)
    ).reshape(-1, 3)
    big = np.abs(draws) > 3 * 0.01
    both = np.mean(big[:, 0] & big[:, 1])
    assert both > 3 * np.mean(big[:, 0]) * np.mean(big[:, 1])
    # Yet the funds are uncorrelated: their signs agree half the time.
    agree = np.mean(np.sign(draws[:, 0]) == np.sign(draws[:, 1]))
    assert abs(agree - 0.5) < 4 * math.sqrt(0.25 / len(draws))


def test_the_t_log_likelihood_is_a_density() -> None:
    # In one dimension it integrates to 1.
    mean, cov = np.array([0.001]), np.array([[1e-4]])
    grid = np.linspace(-0.5, 0.5, 4001)
    density = [math.exp(log_likelihood(np.array([[x]]), mean, cov, 3.0)) for x in grid]
    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=2e-4)


def test_the_normal_log_likelihood_is_the_sum_of_normal_densities() -> None:
    returns = daily_returns(n=50)
    logs = np.log1p(returns.to_numpy())
    mean = logs.mean(axis=0)
    cov = np.cov(logs.T)
    inverse = np.linalg.inv(cov)
    expected = sum(
        -0.5 * (3 * math.log(2 * math.pi) + math.log(np.linalg.det(cov)) + d @ inverse @ d)
        for d in logs - mean
    )
    assert log_likelihood(logs, mean, cov) == pytest.approx(expected, rel=1e-12)
    # Many degrees of freedom: all but normal.
    assert log_likelihood(logs, mean, cov, 1e8) == pytest.approx(expected, rel=1e-6)


def test_the_likelihood_refuses_a_singular_covariance() -> None:
    cov = np.array([[1e-4, 2e-4, 0.0], [2e-4, 4e-4, 0.0], [0.0, 0.0, 1e-4]])
    with pytest.raises(ValueError, match="singular"):
        log_likelihood(np.zeros((4, 3)), np.zeros(3), cov, 5.0)


def test_fit_recovers_the_degrees_of_freedom() -> None:
    draws = t_model(5.0).sample(np.random.default_rng(14), 1, 20_000)[0]
    fitted = StudentTModel.fit(pd.DataFrame(draws, columns=list(TICKERS)))
    assert fitted.dof == pytest.approx(5.0, abs=0.5)
    np.testing.assert_allclose(fitted.mean, np.log1p(draws).mean(axis=0), rtol=1e-12)


def test_normal_returns_fit_many_degrees_of_freedom() -> None:
    draws = model().sample(np.random.default_rng(15), 1, 20_000)[0]
    fitted = StudentTModel.fit(pd.DataFrame(draws, columns=list(TICKERS)))
    assert fitted.dof > 50


def test_the_fitted_dof_is_the_likeliest() -> None:
    draws = t_model(6.0).sample(np.random.default_rng(16), 1, 3_000)[0]
    logs = np.log1p(draws)
    mean, cov = logs.mean(axis=0), np.cov(logs.T)
    dof = fit_dof(logs, mean, cov)
    best = log_likelihood(logs, mean, cov, dof)
    for nearby in (dof * 0.999, dof * 1.001, dof * 0.9, dof * 1.1):
        assert log_likelihood(logs, mean, cov, nearby) < best


def test_a_given_dof_is_kept() -> None:
    assert StudentTModel.fit(daily_returns(), dof=7.5).dof == 7.5


@pytest.mark.parametrize("dof", [2.0, 1.5, math.inf, math.nan, True, "5"])
def test_the_t_needs_more_than_two_degrees_of_freedom(dof) -> None:
    with pytest.raises(ValueError, match="dof"):
        t_model(dof)


def prices() -> pd.DataFrame:
    days = pd.bdate_range("2023-01-02", "2024-12-31")
    rng = np.random.default_rng(0)
    rows = []
    for i, ticker in enumerate(["AAA", "BBB"]):
        close = 50.0 * np.cumprod(1 + rng.normal(0.0003, 0.01 * (1 + i), len(days)))
        for day, value in zip(days, close, strict=True):
            rows.append((day.date(), ticker, float(value), 0.0, 0.0, 0.0, 1e6))
    return pd.DataFrame(rows, columns=list(COLUMNS))


def test_the_command_prints_and_writes_the_figures(tmp_path: Path, capsys) -> None:
    cache = tmp_path / "prices.csv"
    manifest = write_cache(
        prices(), cache, start=date(2023, 1, 1), end=date(2024, 12, 31), source="t"
    )
    out = tmp_path / "results" / "mc.json"
    argv = ["--cache", str(cache), "--paths", "1200", "--horizon", "21", "--json", str(out)]
    assert main([*argv, "--weights", "AAA=0.7,BBB=0.3"]) == 0
    printed = capsys.readouterr().out
    assert "1,200 paths of 21 days, seed 0" in printed
    assert "AAA 70.0%, BBB 30.0%" in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["cache_sha256"] == manifest["sha256"]
    assert written["weights"] == {"AAA": 0.7, "BBB": 0.3}
    assert written["daily"]["simulated"]["count"] == 1200 * 21
    assert written["horizon"]["simulated"]["count"] == 1200
    assert written["horizon"]["historical"]["count"] == 521 - 21 + 1
    assert b"\r" not in out.read_bytes()
    # The same seed gives the same figures.
    again = tmp_path / "again.json"
    main([*argv[:-1], str(again), "--weights", "AAA=0.7,BBB=0.3"])
    assert json.loads(again.read_text(encoding="utf-8")) == written


def test_the_command_refuses_bad_input_plainly(tmp_path: Path, capsys) -> None:
    cache = tmp_path / "prices.csv"
    assert main(["--cache", str(cache)]) == 1
    assert "manifest.json is missing" in capsys.readouterr().err
    write_cache(prices(), cache, start=date(2023, 1, 1), end=date(2024, 12, 31), source="t")
    for argv, message in [
        (["--weights", "AAA=2"], "add up to 1"),
        (["--horizon", "0"], "horizon"),
        (["--horizon", "600"], "521 daily returns hold 0 windows of 600 days"),
        (["--horizon", "521"], "hold 1 window of 521 days; needs at least 2"),
        (["--paths", "1"], "at least 2 paths"),
        (["--json", str(tmp_path)], ""),  # a folder
    ]:
        assert main(["--cache", str(cache), "--paths", "10", *argv]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error: ") and message in captured.err
        assert "Traceback" not in captured.err
