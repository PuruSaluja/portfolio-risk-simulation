import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data import COLUMNS
from portfolio_risk.returns import (
    PERIODS_PER_YEAR,
    annualised_mean,
    annualised_return,
    annualised_volatility,
    autocorrelation,
    compound,
    implied_volatility_ratio,
    log_returns,
    observed_periods_per_year,
    overlapping_volatility,
    price_returns,
    total_returns,
)


def cache(rows):
    """A cache-shaped frame from (date, ticker, close, dividend[, capital_gain, split]) rows."""
    out = []
    for row in rows:
        day, ticker, close, dividend, *rest = row
        capital_gain, split = (rest + [0.0, 0.0])[:2]
        out.append((date.fromisoformat(day), ticker, close, dividend, capital_gain, split, 1e6))
    return pd.DataFrame(out, columns=list(COLUMNS))


def test_a_dividend_is_added_to_the_day_it_goes_ex() -> None:
    prices = cache(
        [
            ("2024-01-02", "AAA", 100.0, 0.0),
            ("2024-01-03", "AAA", 99.0, 2.0),  # fell by less than the dividend
            ("2024-01-04", "AAA", 99.99, 0.0),
        ]
    )
    r = total_returns(prices)["AAA"]
    assert r.iloc[0] == pytest.approx(0.01, abs=1e-15)
    assert r.iloc[1] == pytest.approx(0.01, abs=1e-15)
    assert list(r.index) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]


def test_capital_gain_distributions_count_like_dividends() -> None:
    prices = cache([("2024-01-02", "AAA", 50.0, 0.0), ("2024-01-03", "AAA", 49.0, 0.5, 1.0)])
    assert total_returns(prices)["AAA"].iloc[0] == pytest.approx(0.01, abs=1e-15)


def test_price_returns_leave_the_payouts_out() -> None:
    prices = cache([("2024-01-02", "AAA", 100.0, 0.0), ("2024-01-03", "AAA", 99.0, 2.0)])
    assert price_returns(prices)["AAA"].iloc[0] == pytest.approx(-0.01, abs=1e-15)


def test_splits_change_nothing_because_the_closes_are_already_adjusted() -> None:
    rows = [("2024-01-02", "AAA", 100.0, 0.0), ("2024-01-03", "AAA", 101.0, 0.0)]
    split = [rows[0], ("2024-01-03", "AAA", 101.0, 0.0, 0.0, 2.0)]
    pd.testing.assert_frame_equal(total_returns(cache(rows)), total_returns(cache(split)))


def test_only_days_every_fund_traded_are_kept_and_a_dropped_payout_moves_on() -> None:
    prices = cache(
        [
            ("2024-01-02", "AAA", 100.0, 0.0),
            ("2024-01-03", "AAA", 98.0, 1.0),  # BBB did not trade this day
            ("2024-01-04", "AAA", 99.0, 0.0),
            ("2024-01-02", "BBB", 10.0, 0.0),
            ("2024-01-04", "BBB", 11.0, 0.0),
        ]
    )
    r = total_returns(prices)
    assert list(r.index) == [pd.Timestamp("2024-01-04")]
    assert list(r.columns) == ["AAA", "BBB"]
    # AAA's dividend on the dropped day is counted in the return to the next day kept.
    assert r.loc["2024-01-04", "AAA"] == pytest.approx(0.0, abs=1e-15)
    assert r.loc["2024-01-04", "BBB"] == pytest.approx(0.1)


def test_a_payout_before_the_first_common_day_is_not_counted() -> None:
    prices = cache(
        [
            ("2024-01-02", "AAA", 100.0, 5.0),  # BBB has not started yet
            ("2024-01-03", "AAA", 100.0, 0.0),
            ("2024-01-04", "AAA", 101.0, 0.0),
            ("2024-01-03", "BBB", 10.0, 0.0),
            ("2024-01-04", "BBB", 10.0, 0.0),
        ]
    )
    assert total_returns(prices).loc["2024-01-04", "AAA"] == pytest.approx(0.01)


@pytest.mark.parametrize("function", [total_returns, price_returns])
@pytest.mark.parametrize("bbb_days", [["2024-01-04"], ["2024-01-03"]])
def test_fewer_than_two_common_days_are_refused(function, bbb_days) -> None:
    rows = [("2024-01-02", "AAA", 100.0, 0.0), ("2024-01-03", "AAA", 101.0, 0.0)]
    rows += [(day, "BBB", 10.0, 0.0) for day in bbb_days]  # no day, or one, in common
    with pytest.raises(ValueError, match="fewer than two days"):
        function(cache(rows))


def test_log_returns_add_up_to_the_compound_return() -> None:
    r = pd.Series([0.1, -0.2, 0.05])
    assert log_returns(r).sum() == pytest.approx(math.log(1.1 * 0.8 * 1.05))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([0.1, -1.0], "-100% or less"),
        ([0.1, -1.5], "-100% or less"),
        ([0.1, np.nan], "NaN or infinite"),
        ([0.1, np.inf], "NaN or infinite"),
        ([], "at least 1"),
    ],
)
def test_returns_that_cannot_compound_are_refused(values, message) -> None:
    r = pd.Series(values, dtype="float64")
    for function in (log_returns, lambda x: annualised_return(x, 252)):
        with pytest.raises(ValueError, match=message):
            function(r)


def test_a_list_is_refused_with_a_type_error() -> None:
    with pytest.raises(TypeError, match="pandas Series or DataFrame, not list"):
        annualised_mean([0.01, 0.02], 252)


def daily(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


def test_monthly_returns_are_the_ratio_of_month_end_prices() -> None:
    rng = np.random.default_rng(1)
    r = daily(rng.normal(0.0005, 0.01, 300))
    price = 100 * (1 + r).cumprod()
    monthly = compound(r, "M")
    month_end = price.groupby(price.index.to_period("M")).last()
    expected = month_end / month_end.shift(1) - 1
    expected.iloc[0] = month_end.iloc[0] / 100 - 1  # the first month starts at the first close
    np.testing.assert_allclose(monthly.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_periods_are_labelled_by_their_last_trading_day() -> None:
    # 2024-03-29 was a Friday; 30 and 31 March fell on the weekend.
    r = daily([0.01] * 70)
    monthly = compound(r, "M")
    assert list(monthly.index[:3]) == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
        pd.Timestamp("2024-03-29"),
    ]
    weekly = compound(r, "W")
    assert weekly.index[0] == pd.Timestamp("2024-01-05")
    assert weekly.iloc[0] == pytest.approx(1.01**5 - 1)
    assert (weekly.index.dayofweek == 4).all()


def test_compound_works_on_each_column_of_a_frame() -> None:
    frame = pd.DataFrame(
        {"A": [0.1, 0.1, 0.0], "B": [0.0, -0.5, 1.0]}, index=pd.bdate_range("2024-12-30", periods=3)
    )
    yearly = compound(frame, "Y")
    assert list(yearly.index) == [pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01")]
    assert yearly.loc["2024-12-31", "A"] == pytest.approx(0.21)
    assert yearly.loc["2024-12-31", "B"] == pytest.approx(-0.5)
    assert yearly.loc["2025-01-01", "B"] == pytest.approx(1.0)


def test_an_unknown_frequency_is_refused() -> None:
    with pytest.raises(ValueError, match="frequency must be one of W, M, Q, Y, not 'D'"):
        compound(daily([0.01, 0.02]), "D")


def test_a_constant_daily_return_annualises_exactly() -> None:
    r = daily([0.0004] * 504)
    assert annualised_return(r, 252) == pytest.approx(1.0004**252 - 1, rel=1e-12)
    assert annualised_mean(r, 252) == pytest.approx(0.0004 * 252, rel=1e-12)
    assert annualised_volatility(r, 252) == pytest.approx(0.0, abs=1e-15)


def test_the_compound_rate_reproduces_the_total_growth() -> None:
    rng = np.random.default_rng(7)
    r = daily(rng.normal(0.0003, 0.012, 1000))
    rate = annualised_return(r, 252)
    assert (1 + rate) ** (1000 / 252) == pytest.approx((1 + r).prod(), rel=1e-12)


def test_volatility_is_the_sample_standard_deviation_scaled_by_root_time() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(0, 0.01, 250)
    assert annualised_volatility(daily(values), 252) == pytest.approx(
        np.std(values, ddof=1) * math.sqrt(252), rel=1e-12
    )
    assert annualised_volatility(daily(values), 12) == pytest.approx(
        np.std(values, ddof=1) * math.sqrt(12), rel=1e-12
    )


def test_the_annualised_figures_are_per_column_for_a_frame() -> None:
    frame = pd.DataFrame({"A": [0.01, 0.03], "B": [0.0, 0.0]})
    mean = annualised_mean(frame, 12)
    assert list(mean.index) == ["A", "B"]
    assert mean["A"] == pytest.approx(0.24, rel=1e-12)
    assert annualised_return(frame, 1)["B"] == 0.0


def test_volatility_needs_two_returns() -> None:
    with pytest.raises(ValueError, match="at least 2 returns, got 1"):
        annualised_volatility(daily([0.01]), 252)


@pytest.mark.parametrize("periods", [0, -12, math.inf, math.nan, True, "252"])
def test_periods_per_year_must_be_a_positive_number(periods) -> None:
    with pytest.raises(ValueError, match="periods_per_year must be a positive number"):
        annualised_mean(daily([0.01, 0.02]), periods)


def test_a_compound_return_over_a_long_history_does_not_overflow() -> None:
    r = daily([0.05] * 20000)
    assert annualised_return(r, 252) == pytest.approx(1.05**252 - 1, rel=1e-9)


def test_observed_periods_per_year_counts_the_gaps_between_dates() -> None:
    # 2023 had 260 weekdays, 2024 262: 522 days, 521 gaps over 730 days.
    days = pd.bdate_range("2023-01-02", "2024-12-31")
    assert observed_periods_per_year(days) == pytest.approx(
        521 / ((days[-1] - days[0]).days / 365.25)
    )
    months = pd.date_range("2020-01-31", periods=61, freq="ME")
    assert observed_periods_per_year(months) == pytest.approx(12, rel=1e-3)


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (pd.DatetimeIndex(["2024-01-02"]), "at least two dates"),
        (pd.DatetimeIndex(["2024-01-03", "2024-01-02"]), "increasing and unique"),
        (pd.DatetimeIndex(["2024-01-02", "2024-01-02"]), "increasing and unique"),
    ],
)
def test_observed_periods_per_year_refuses_unusable_dates(index, message) -> None:
    with pytest.raises(ValueError, match=message):
        observed_periods_per_year(index)


def test_the_conventional_periods_per_year() -> None:
    assert PERIODS_PER_YEAR == {"D": 252, "W": 52, "M": 12, "Q": 4, "Y": 1}


def test_autocorrelation_is_the_lagged_covariance_over_the_variance() -> None:
    values = np.array([0.01, -0.02, 0.015, 0.0, -0.005, 0.02])
    d = values - values.mean()
    for lag in (1, 2, 3):
        expected = np.dot(d[:-lag], d[lag:]) / np.dot(d, d)
        assert autocorrelation(pd.Series(values), lag) == pytest.approx(expected, rel=1e-12)


def test_autocorrelation_recovers_an_autoregressive_coefficient() -> None:
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.01, 20000)
    values = np.empty_like(noise)
    values[0] = noise[0]
    for t in range(1, len(noise)):
        values[t] = -0.3 * values[t - 1] + noise[t]
    series = pd.Series(values)
    assert autocorrelation(series) == pytest.approx(-0.3, abs=0.02)
    assert autocorrelation(series, 2) == pytest.approx(0.09, abs=0.02)


def test_alternating_returns_are_almost_perfectly_negatively_autocorrelated() -> None:
    series = pd.Series([0.01, -0.01] * 50)
    assert autocorrelation(series) == pytest.approx(-0.99)  # -(n - 1) / n
    assert implied_volatility_ratio(series, 2) == pytest.approx(0.1)  # sqrt(1 - 0.99)


def test_overlapping_volatility_compounds_every_window() -> None:
    values = [0.01, -0.02, 0.03, 0.005, -0.01]
    windows = [(1 + a) * (1 + b) - 1 for a, b in zip(values, values[1:], strict=False)]
    expected = np.std(windows, ddof=1) * math.sqrt(252 / 2)
    assert overlapping_volatility(pd.Series(values), 2, 252) == pytest.approx(expected, rel=1e-12)
    assert overlapping_volatility(pd.Series(values), 1, 252) == pytest.approx(
        np.std(values, ddof=1) * math.sqrt(252), rel=1e-12
    )


def test_uncorrelated_returns_keep_the_square_root_rule() -> None:
    rng = np.random.default_rng(12)
    series = pd.Series(rng.normal(0, 0.01, 50000))
    scaled = annualised_volatility(series, 252)
    assert overlapping_volatility(series, 21, 252) / scaled == pytest.approx(1.0, abs=0.03)
    assert implied_volatility_ratio(series, 21) == pytest.approx(1.0, abs=0.03)


def test_the_implied_ratio_matches_overlapping_windows_for_mean_reverting_returns() -> None:
    rng = np.random.default_rng(13)
    noise = rng.normal(0, 0.01, 50000)
    values = noise.copy()
    values[1:] -= 0.2 * noise[:-1]  # an MA(1): lag-one autocorrelation about -0.19
    series = pd.Series(values)
    ratio = overlapping_volatility(series, 21, 252) / annualised_volatility(series, 252)
    assert implied_volatility_ratio(series, 21) == pytest.approx(ratio, abs=0.02)
    assert ratio < 0.92


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_lags_and_horizons_must_be_whole_numbers_of_at_least_one(bad) -> None:
    series = pd.Series([0.01, -0.01, 0.02, 0.0, 0.01])
    with pytest.raises(ValueError, match="lag must be a whole number"):
        autocorrelation(series, bad)
    with pytest.raises(ValueError, match="horizon must be a whole number"):
        overlapping_volatility(series, bad, 252)
    with pytest.raises(ValueError, match="horizon must be a whole number"):
        implied_volatility_ratio(series, bad)


def test_the_autocorrelation_helpers_refuse_unusable_input() -> None:
    with pytest.raises(ValueError, match="never vary"):
        autocorrelation(pd.Series([0.01] * 10))
    with pytest.raises(ValueError, match="at least 4 returns, got 3"):
        autocorrelation(pd.Series([0.01, 0.02, 0.0]), 2)
    with pytest.raises(ValueError, match="at least 22 returns"):
        overlapping_volatility(pd.Series([0.01, -0.01] * 10), 21, 252)
    with pytest.raises(TypeError, match="expected a pandas Series, not DataFrame"):
        autocorrelation(pd.DataFrame({"A": [0.01, 0.0, 0.02]}))
    with pytest.raises(ValueError, match="periods_per_year"):
        overlapping_volatility(pd.Series([0.01, -0.01] * 20), 21, 0)
