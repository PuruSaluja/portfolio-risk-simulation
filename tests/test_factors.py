import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data import COLUMNS as data_columns
from portfolio_risk.data import DataError
from portfolio_risk.data import write_cache as write_prices
from portfolio_risk.factors import (
    LABELS,
    MODELS,
    align,
    attribution,
    excess_returns,
    fit_model,
    main,
    newey_west_lags,
    regress,
    run,
    summary,
)
from portfolio_risk.french import COLUMNS as factor_columns
from portfolio_risk.french import daily_risk_free, load_factors
from portfolio_risk.french import write_cache as write_factors

MODELS_FIVE = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

DAYS = pd.bdate_range("2020-01-01", periods=400)


def factor_frame(n=400, seed=0, columns=("mkt_rf", "smb", "hml", "smb5", "rmw", "cma")):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0003, 0.01, (n, len(columns)))
    return pd.DataFrame(data, index=pd.bdate_range("2020-01-01", periods=n), columns=list(columns))


def simulated(alpha=0.0002, betas=(1.1, 0.4, -0.3), noise=0.002, n=400, seed=0):
    x = factor_frame(n, seed, columns=("a", "b", "c")[: len(betas)])
    rng = np.random.default_rng(seed + 1)
    y = alpha + x.to_numpy() @ np.array(betas) + rng.normal(0, noise, n)
    return pd.Series(y, index=x.index), x


def brute_newey_west(y, x, lags):
    """The Newey-West covariance by plain loops over observations, from the formula."""
    design = np.column_stack([np.ones(len(y)), x])
    n, k = design.shape
    beta = np.linalg.solve(design.T @ design, design.T @ y)
    e = y - design @ beta
    meat = np.zeros((k, k))
    for t in range(n):
        meat += e[t] ** 2 * np.outer(design[t], design[t])
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        for t in range(lag, n):
            g = e[t] * e[t - lag] * np.outer(design[t], design[t - lag])
            meat += w * (g + g.T)
    bread = np.linalg.inv(design.T @ design)
    return np.sqrt(np.diag(bread @ meat @ bread))


def test_the_coefficients_are_those_a_series_was_built_from() -> None:
    y, x = simulated(noise=0.0)
    fit = regress(y, x)
    assert fit.names == ("alpha", "a", "b", "c")
    np.testing.assert_allclose(fit.coefficients, [0.0002, 1.1, 0.4, -0.3], rtol=0, atol=1e-14)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)


def test_noisy_coefficients_land_near_the_truth() -> None:
    y, x = simulated(noise=0.002)
    fit = regress(y, x)
    within = np.abs(fit.coefficients - [0.0002, 1.1, 0.4, -0.3]) < 3 * fit.ols_se
    assert within.all()


def test_one_factor_matches_the_textbook_formulas() -> None:
    y, x = simulated(betas=(0.8,))
    fit = regress(y, x)
    xs, ys = x["a"].to_numpy(), y.to_numpy()
    sxx = ((xs - xs.mean()) ** 2).sum()
    slope = ((xs - xs.mean()) * (ys - ys.mean())).sum() / sxx
    intercept = ys.mean() - slope * xs.mean()
    e = ys - intercept - slope * xs
    s2 = (e @ e) / (len(ys) - 2)
    assert fit.coefficients[1] == pytest.approx(slope, rel=1e-12)
    assert fit.coefficients[0] == pytest.approx(intercept, rel=1e-10)
    assert fit.ols_se[1] == pytest.approx(math.sqrt(s2 / sxx), rel=1e-12)
    assert fit.ols_se[0] == pytest.approx(
        math.sqrt(s2 * (1 / len(ys) + xs.mean() ** 2 / sxx)), rel=1e-12
    )
    r = np.corrcoef(xs, ys)[0, 1]
    assert fit.r_squared == pytest.approx(r * r, rel=1e-12)
    assert fit.adjusted_r_squared == pytest.approx(1 - (1 - r * r) * 399 / 398, rel=1e-12)
    assert fit.residual_sd == pytest.approx(math.sqrt(s2), rel=1e-12)


@pytest.mark.parametrize("lags", [0, 1, 4, 9])
def test_newey_west_errors_match_the_formula_by_plain_loops(lags) -> None:
    y, x = simulated(n=120)
    fit = regress(y, x, lags=lags)
    expected = brute_newey_west(y.to_numpy(), x.to_numpy(), lags)
    np.testing.assert_allclose(fit.newey_west_se, expected, rtol=1e-11)
    assert fit.lags == lags


def test_newey_west_errors_widen_for_autocorrelated_residuals() -> None:
    # Residuals that persist for days: the plain OLS error understates the alpha's.
    rng = np.random.default_rng(3)
    x = factor_frame(2000, 3, columns=("a",))
    e = np.convolve(rng.normal(0, 0.002, 2004), np.ones(5), "valid")
    y = pd.Series(0.5 * x["a"].to_numpy() + e, index=x.index)
    fit = regress(y, x)
    assert fit.newey_west_se[0] > 1.8 * fit.ols_se[0]


def test_t_statistics_are_estimates_over_their_errors() -> None:
    y, x = simulated()
    fit = regress(y, x)
    np.testing.assert_array_equal(fit.ols_t, fit.coefficients / fit.ols_se)
    np.testing.assert_array_equal(fit.newey_west_t, fit.coefficients / fit.newey_west_se)


@pytest.mark.parametrize(("n", "lags"), [(1, 1), (3, 1), (100, 4), (5448, 9), (10_000, 11)])
def test_the_default_lags_follow_newey_and_west_s_rule(n, lags) -> None:
    assert newey_west_lags(n) == lags


def test_the_default_lags_are_used_when_none_are_given() -> None:
    y, x = simulated()
    assert regress(y, x).lags == newey_west_lags(400) == 5


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_bad_counts_have_no_lag_rule(n) -> None:
    with pytest.raises(ValueError, match="positive whole number"):
        newey_west_lags(n)


def test_regress_refuses_what_it_cannot_fit() -> None:
    y, x = simulated()
    with pytest.raises(TypeError):
        regress(y.to_numpy(), x)
    with pytest.raises(ValueError, match="same dates"):
        regress(y.iloc[1:], x.iloc[:-1])
    with pytest.raises(ValueError, match="no factors"):
        regress(y, x[[]])
    with pytest.raises(ValueError, match="4 observations cannot fit 4"):
        regress(y.iloc[:4], x.iloc[:4])
    collinear = x.assign(d=2 * x["a"] - x["b"])
    with pytest.raises(ValueError, match="collinear"):
        regress(y, collinear)
    with pytest.raises(ValueError, match="collinear"):
        regress(y, x.assign(d=0.01))
    with pytest.raises(ValueError, match="finite"):
        regress(y.where(y.index != y.index[5], np.nan), x)
    infinite = x.copy()
    infinite.iloc[5, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        regress(y, infinite)
    with pytest.raises(ValueError, match="never vary"):
        regress(pd.Series(0.001, index=y.index), x)


@pytest.mark.parametrize("lags", [-1, 400, 2.0, True])
def test_bad_lags_are_refused(lags) -> None:
    y, x = simulated()
    with pytest.raises(ValueError, match="lags must be a whole number from 0 to 399"):
        regress(y, x, lags=lags)


def test_the_last_lag_allowed_is_one_short_of_the_observations() -> None:
    y, x = simulated(n=12)
    assert regress(y, x, lags=11).lags == 11


def test_the_attribution_adds_up_to_the_mean_excess_return() -> None:
    y, x = simulated()
    fit = regress(y, x)
    split = attribution(fit)
    assert split["mean_excess"] == pytest.approx(y.mean() * 252, rel=1e-14)
    assert sum(split["parts"].values()) == pytest.approx(split["mean_excess"], rel=1e-12)
    assert split["parts"]["alpha"] == fit.coefficients[0] * 252
    assert split["parts"]["b"] == pytest.approx(fit.coefficients[2] * x["b"].mean() * 252)
    monthly = attribution(fit, 12)
    assert monthly["mean_excess"] == pytest.approx(y.mean() * 12, rel=1e-14)


def test_the_summary_puts_alpha_on_a_yearly_scale_and_nothing_else() -> None:
    y, x = simulated()
    fit = regress(y, x)
    out = summary(fit)
    assert out["days"] == 400
    assert out["newey_west_lags"] == 5
    assert out["coefficients"]["alpha"]["estimate"] == fit.coefficients[0] * 252
    assert out["coefficients"]["a"]["estimate"] == fit.coefficients[1]
    assert out["coefficients"]["alpha"]["ols_t"] == fit.ols_t[0]
    assert out["coefficients"]["c"]["newey_west_t"] == fit.newey_west_t[3]
    assert out["r_squared"] == fit.r_squared
    assert out["adjusted_r_squared"] == fit.adjusted_r_squared
    assert out["residual_volatility"] == fit.residual_sd * math.sqrt(252)
    assert out["attribution"] == attribution(fit)


def test_each_model_regresses_on_its_own_factors_named_as_the_library_names_them() -> None:
    factors = factor_frame()
    three = fit_model(factors["smb"] * 0.2 + 0.0001, factors, "ff3")
    five = fit_model(factors["smb5"] * 0.7 + 0.0001, factors, "ff5")
    assert three.names == ("alpha", "Mkt-RF", "SMB", "HML")
    assert five.names == ("alpha", "Mkt-RF", "SMB", "HML", "RMW", "CMA")
    # The five-factor model's SMB is the five-factor file's.
    assert five.coefficients[2] == pytest.approx(0.7, rel=1e-12)
    assert three.coefficients[2] == pytest.approx(0.2, rel=1e-12)
    assert [LABELS[c] for c in MODELS["ff5"]] == list(five.names[1:])


def test_an_unknown_model_is_refused() -> None:
    factors = factor_frame()
    with pytest.raises(ValueError, match="unknown model 'ff4'; choose from ff3, ff5"):
        fit_model(factors["smb"], factors, "ff4")


def test_excess_returns_subtract_each_day_s_rate() -> None:
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.0, -0.01]}, index=DAYS[:2])
    rate = pd.Series([0.001, 0.002], index=DAYS[:2])
    out = excess_returns(returns, rate)
    np.testing.assert_allclose(out.to_numpy(), [[0.009, -0.001], [0.018, -0.012]], rtol=1e-15)
    with pytest.raises(ValueError, match="same dates"):
        excess_returns(returns, rate.iloc[:1])


def test_align_keeps_the_days_both_cover() -> None:
    returns = pd.DataFrame({"A": range(10)}, index=DAYS[2:12], dtype=float)
    factors = factor_frame(9)
    r, f = align(returns, factors)
    assert list(r.index) == list(DAYS[2:9])
    assert r.index.equals(f.index)
    # And the other way round: the factors start later and end first.
    r, f = align(pd.DataFrame({"A": range(12)}, index=DAYS[:12], dtype=float), factors.iloc[3:])
    assert list(f.index) == list(r.index) == list(DAYS[3:9])


def test_align_refuses_a_day_only_one_side_has() -> None:
    returns = pd.DataFrame({"A": range(9)}, index=DAYS[:9], dtype=float)
    factors = factor_frame(9)
    with pytest.raises(DataError, match="2020-01-03 is in the returns only \\(1 days differ\\)"):
        align(returns, factors.drop(DAYS[2]))
    with pytest.raises(DataError, match="2020-01-03 is in the factors only"):
        align(returns.drop(DAYS[2]), factors)


def test_align_refuses_tables_with_no_day_in_common() -> None:
    returns = pd.DataFrame({"A": [0.0]}, index=DAYS[300:301])
    with pytest.raises(DataError, match="no common days"):
        align(returns, factor_frame(10))


PRICE_DAYS = pd.bdate_range("2023-01-02", "2024-12-31")


def prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i, ticker in enumerate(["AAA", "BBB"]):
        close = 50.0 * np.cumprod(1 + rng.normal(0.0003, 0.01 * (1 + i), len(PRICE_DAYS)))
        for day, value in zip(PRICE_DAYS, close, strict=True):
            rows.append((day.date(), ticker, float(value), 0.0, 0.0, 0.0, 1e6))
    return pd.DataFrame(rows, columns=list(data_columns))


def factor_table(end="2024-11-30", market_from=None) -> pd.DataFrame:
    """Factors in percent, as the cache holds them, over the price days up to ``end``;
    ``mkt_rf`` is ``market_from``'s excess return when given, so it explains it exactly."""
    days = PRICE_DAYS[end >= PRICE_DAYS]
    rng = np.random.default_rng(1)
    table = pd.DataFrame(
        np.round(rng.normal(0, 1, (len(days), 6)), 2),
        columns=["mkt_rf", "smb", "hml", "smb5", "rmw", "cma"],
    )
    table.insert(0, "date", days.date)
    month = days.month + 12 * (days.year - 2023)
    table["rf"] = 0.02
    table["rf_month"] = np.round(0.30 + 0.01 * month.to_numpy(), 2)
    if market_from is not None:
        fractions = table.set_index(pd.DatetimeIndex(table["date"]))[["rf_month"]] / 100
        rate = daily_risk_free(fractions)
        returns = market_from.loc[days[1:]]
        table.loc[1:, "mkt_rf"] = ((returns - rate.loc[days[1:]]) * 100).to_numpy()
    return table[list(factor_columns)]


def caches(tmp_path: Path, **kwargs) -> tuple[Path, Path]:
    price_path = tmp_path / "prices.csv"
    write_prices(prices(), price_path, start=date(2023, 1, 1), end=date(2024, 12, 31), source="t")
    factor_path = tmp_path / "factors.csv"
    write_factors(
        factor_table(**kwargs),
        factor_path,
        start=date(2023, 1, 1),
        end=date(2024, 12, 31),
        vintages={},
    )
    return price_path, factor_path


def daily_total(table: pd.DataFrame) -> pd.DataFrame:
    close = table.pivot(index="date", columns="ticker", values="close")[["AAA", "BBB"]]
    daily = (close / close.shift(1) - 1).iloc[1:]
    daily.index = pd.DatetimeIndex(daily.index)
    daily.columns.name = None
    return daily


def test_the_command_fits_every_series_against_the_monthly_bill_rate(
    tmp_path: Path, capsys
) -> None:
    price_path, factor_path = caches(tmp_path)
    out = tmp_path / "results" / "factors.json"
    argv = ["--cache", str(price_path), "--factors", str(factor_path), "--json", str(out)]
    assert main(argv + ["--weights", "AAA=0.7,BBB=0.3"]) == 0
    printed = capsys.readouterr().out
    # The factors end a month before the prices: the days both hold.
    assert "499 daily returns, 2023-01-03 to 2024-11-29" in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["series"].keys() == {"AAA", "BBB", "portfolio"}
    assert written["series"]["AAA"].keys() == {"ff3", "ff5"}
    assert written["models"] == {
        "ff3": ["Mkt-RF", "SMB", "HML"],
        "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    }
    assert written["weights"] == {"AAA": 0.7, "BBB": 0.3}
    assert b"\r" not in out.read_bytes()
    # The same fit, by hand: each day's return less the day's bill rate.
    factors = load_factors(factor_path)
    daily = daily_total(prices()).loc["2023-01-03":"2024-11-29"]
    rate = daily_risk_free(factors).loc[daily.index]
    portfolio = (daily @ [0.7, 0.3]) - rate
    x = factors.loc[daily.index, ["mkt_rf", "smb5", "hml", "rmw", "cma"]]
    expected = summary(regress(portfolio, x.set_axis(MODELS_FIVE, axis=1)))
    got = written["series"]["portfolio"]["ff5"]
    assert got["coefficients"]["alpha"]["estimate"] == pytest.approx(
        expected["coefficients"]["alpha"]["estimate"], rel=1e-10
    )
    assert got["r_squared"] == pytest.approx(expected["r_squared"], rel=1e-12)
    years = len(rate) / 252
    assert written["risk_free_compound"] == pytest.approx(
        np.prod(1 + rate) ** (1 / years) - 1, rel=1e-12
    )


def test_a_market_factor_built_from_a_fund_explains_it_exactly(tmp_path: Path) -> None:
    price_path, factor_path = caches(tmp_path, market_from=daily_total(prices())["AAA"])
    out = tmp_path / "factors.json"
    argv = ["--cache", str(price_path), "--factors", str(factor_path), "--json", str(out)]
    assert main(argv + ["--model", "ff3"]) == 0
    fit = json.loads(out.read_text(encoding="utf-8"))["series"]["AAA"]["ff3"]
    assert fit["coefficients"]["Mkt-RF"]["estimate"] == pytest.approx(1.0, abs=1e-12)
    assert fit["coefficients"]["alpha"]["estimate"] == pytest.approx(0.0, abs=1e-12)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-12)


def test_the_command_refuses_bad_input_plainly(tmp_path: Path, capsys) -> None:
    price_path, factor_path = caches(tmp_path)
    base = ["--cache", str(price_path), "--factors", str(factor_path)]
    for argv, message in [
        (["--model", "ff4"], "unknown model 'ff4'"),
        (["--model", "ff3", "--model", "ff3"], "named twice"),
        (["--weights", "AAA=2"], "add up to 1"),
        (["--factors", str(tmp_path / "none.csv")], "manifest.json is missing"),
        (["--json", str(tmp_path)], ""),  # a folder
    ]:
        assert main(base + argv) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error: ") and message in captured.err


def test_run_needs_a_model() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        run(prices(), pd.DataFrame(), models=())
