"""Fama-French factor regressions: what each fund's returns owe to its factor exposures.

A fund's daily return in excess of the risk-free rate is regressed on the day's factor
returns by ordinary least squares::

    r[t] - rf[t] = alpha + sum(beta[k] * f[k][t]) + e[t]

``ff3`` is Fama and French's three factors (1993): the market's excess return
(``Mkt-RF``), small minus big (``SMB``) and high minus low book-to-market (``HML``).
``ff5`` adds robust minus weak profitability (``RMW``) and conservative minus
aggressive investment (``CMA``) (2015), with the five-factor file's ``SMB``.

Daily returns are neither independent nor of constant variance, so besides the usual
OLS standard errors ``regress`` gives Newey-West (1987) ones: heteroskedasticity- and
autocorrelation-consistent, with Bartlett weights ``1 - l / (L + 1)`` for lags ``l``
up to ``L``, by default ``floor(4 (n / 100) ** (2 / 9))`` (Newey and West, 1994; 9
for 5,448 days), and no small-sample correction.

``attribution`` splits the mean excess return exactly, since OLS with an intercept
passes through the means: ``mean(r - rf) = alpha + sum(beta[k] * mean(f[k]))``. Each
part is put on a yearly scale by multiplying by 252, as ``annualised_mean`` does, so the
parts add up to the annualised mean excess return.

The factors are US equity factors, built from NYSE, AMEX and NASDAQ stocks. For a fund
holding other assets (bonds, gold, stocks outside the US) the regression still measures
how its returns moved with them, but a low R-squared there says the factors do not
describe the fund, and its alpha is not a measure of skill.

``python -m portfolio_risk.factors`` fits each model to every fund in the price cache
and to a portfolio of them rebalanced daily, over the days both caches cover, against
the daily risk-free rate ``french.daily_risk_free`` gives; ``--json`` writes the
figures to a file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_risk import french
from portfolio_risk import returns as ret
from portfolio_risk.data import DEFAULT_CACHE, DataError, load_cache, read_manifest
from portfolio_risk.montecarlo import check_weights, historical_portfolio_returns, parse_weights
from portfolio_risk.returns import TRADING_DAYS_PER_YEAR

MODELS: dict[str, tuple[str, ...]] = {
    "ff3": ("mkt_rf", "smb", "hml"),
    "ff5": ("mkt_rf", "smb5", "hml", "rmw", "cma"),
}
# The library's names for the cache's factor columns.
LABELS = {
    "mkt_rf": "Mkt-RF",
    "smb": "SMB",
    "smb5": "SMB",
    "hml": "HML",
    "rmw": "RMW",
    "cma": "CMA",
}


def newey_west_lags(n: int) -> int:
    """Newey and West's (1994) rule for the number of lags: ``floor(4 (n / 100) ** (2/9))``."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive whole number, not {n!r}")
    return math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))


@dataclass(frozen=True)
class Regression:
    """One OLS fit. ``names`` lists ``alpha`` then the factors, in the order of every
    array; the coefficients and standard errors are per period, as the data is."""

    names: tuple[str, ...]
    coefficients: np.ndarray
    ols_se: np.ndarray
    newey_west_se: np.ndarray
    lags: int
    n: int
    r_squared: float
    adjusted_r_squared: float
    residual_sd: float
    mean_y: float
    mean_x: np.ndarray

    @property
    def ols_t(self) -> np.ndarray:
        return self.coefficients / self.ols_se

    @property
    def newey_west_t(self) -> np.ndarray:
        return self.coefficients / self.newey_west_se


def _newey_west_meat(x: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    scores = x * e[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    return meat


def regress(y: pd.Series, x: pd.DataFrame, lags: int | None = None) -> Regression:
    """OLS of ``y`` on a constant and the columns of ``x``, which share ``y``'s index."""
    if not isinstance(y, pd.Series) or not isinstance(x, pd.DataFrame):
        raise TypeError("regress takes a Series of returns and a DataFrame of factors")
    if not y.index.equals(x.index):
        raise ValueError("the returns and the factors must be on the same dates")
    k = x.shape[1]
    n = len(y)
    if k < 1:
        raise ValueError("no factors to regress on")
    if n <= k + 1:
        raise ValueError(f"{n} observations cannot fit {k + 1} coefficients with any to spare")
    values = y.to_numpy(dtype=float)
    factors = x.to_numpy(dtype=float)
    if not (np.isfinite(values).all() and np.isfinite(factors).all()):
        raise ValueError("the returns and factors must be finite numbers")
    if np.ptp(values) == 0.0:
        raise ValueError("returns that never vary have nothing to explain")
    design = np.column_stack([np.ones(n), factors])
    if np.linalg.matrix_rank(design) < k + 1:
        raise ValueError("the factors are collinear (or one never varies), so betas are not unique")
    if lags is None:
        lags = newey_west_lags(n)
    if isinstance(lags, bool) or not isinstance(lags, int) or not 0 <= lags < n:
        raise ValueError(f"lags must be a whole number from 0 to {n - 1}, not {lags!r}")
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ coefficients
    bread = np.linalg.inv(design.T @ design)
    ssr = float(residuals @ residuals)
    sst = float(((values - values.mean()) ** 2).sum())
    variance = ssr / (n - k - 1)
    ols = bread * variance
    hac = bread @ _newey_west_meat(design, residuals, lags) @ bread
    r_squared = 1.0 - ssr / sst
    return Regression(
        names=("alpha", *x.columns),
        coefficients=coefficients,
        ols_se=np.sqrt(np.diag(ols)),
        newey_west_se=np.sqrt(np.diag(hac)),
        lags=lags,
        n=n,
        r_squared=r_squared,
        adjusted_r_squared=1.0 - (1.0 - r_squared) * (n - 1) / (n - k - 1),
        residual_sd=math.sqrt(variance),
        mean_y=float(values.mean()),
        mean_x=factors.mean(axis=0),
    )


def align(returns: pd.DataFrame, factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both tables over the days they share, refused unless they hold the same trading days
    there: a day one has and the other lacks would pair a return with the wrong factors."""
    first = max(returns.index[0], factors.index[0])
    last = min(returns.index[-1], factors.index[-1])
    if first > last:
        raise DataError("the returns and the factors cover no common days")
    r = returns.loc[first:last]
    f = factors.loc[first:last]
    if not r.index.equals(f.index):
        odd = r.index.symmetric_difference(f.index)
        side = "returns" if odd[0] in r.index else "factors"
        raise DataError(
            f"the returns and the factors hold different days: {odd[0].date()} is in the "
            f"{side} only ({len(odd)} days differ)"
        )
    return r, f


def excess_returns(returns: pd.DataFrame, risk_free: pd.Series) -> pd.DataFrame:
    """Each column's return less the day's risk-free rate (on the same dates)."""
    if not returns.index.equals(risk_free.index):
        raise ValueError("the returns and the risk-free rate must be on the same dates")
    return returns.sub(risk_free, axis=0)


def attribution(fit: Regression, periods_per_year: float = TRADING_DAYS_PER_YEAR) -> dict:
    """The mean excess return split into alpha and each factor's part, on a yearly scale.

    The parts add up to ``mean_excess`` up to rounding.
    """
    parts = {"alpha": float(fit.coefficients[0] * periods_per_year)}
    for name, beta, mean in zip(fit.names[1:], fit.coefficients[1:], fit.mean_x, strict=True):
        parts[name] = float(beta * mean * periods_per_year)
    return {"mean_excess": fit.mean_y * periods_per_year, "parts": parts}


def summary(fit: Regression, periods_per_year: float = TRADING_DAYS_PER_YEAR) -> dict[str, Any]:
    """The fit as plain numbers for a JSON file: alpha on a yearly scale, every coefficient
    with both t-statistics, the fit's quality, and the attribution."""
    coefficients = {}
    for i, name in enumerate(fit.names):
        scale = periods_per_year if name == "alpha" else 1.0
        coefficients[name] = {
            "estimate": float(fit.coefficients[i] * scale),
            "ols_t": float(fit.ols_t[i]),
            "newey_west_t": float(fit.newey_west_t[i]),
        }
    return {
        "days": fit.n,
        "newey_west_lags": fit.lags,
        "coefficients": coefficients,
        "r_squared": fit.r_squared,
        "adjusted_r_squared": fit.adjusted_r_squared,
        "residual_volatility": fit.residual_sd * math.sqrt(periods_per_year),
        "attribution": attribution(fit, periods_per_year),
    }


def fit_model(excess: pd.Series, factors: pd.DataFrame, model: str) -> Regression:
    """``regress`` on one model's factors, columns named as the library names them."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {', '.join(MODELS)}")
    columns: Sequence[str] = MODELS[model]
    x = factors[list(columns)].rename(columns=LABELS)
    return regress(excess, x)


def run(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    weights_text: str = "equal",
    models: Sequence[str] = tuple(MODELS),
) -> dict[str, Any]:
    """Every model fitted to each fund's daily excess return and to the portfolio's.

    The portfolio is rebalanced to its weights at every close. The days are those both
    caches hold; the risk-free rate is ``french.daily_risk_free``'s.
    """
    if not models:
        raise ValueError("needs at least one model")
    if len(set(models)) != len(models):
        raise ValueError("a model is named twice")
    for model in models:
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; choose from {', '.join(MODELS)}")
    daily = ret.total_returns(prices)
    tickers = [str(t) for t in daily.columns]
    weights = parse_weights(weights_text, tickers)
    series = daily.assign(portfolio=historical_portfolio_returns(daily, weights))
    # From every day of each month, before the first is dropped for want of a return.
    rate = french.daily_risk_free(factors)
    series, factors = align(series, factors)
    rate = rate.loc[series.index]
    excess = excess_returns(series, rate)
    out = {
        name: {model: summary(fit_model(excess[name], factors, model)) for model in models}
        for name in excess.columns
    }
    years = len(rate) / TRADING_DAYS_PER_YEAR
    return {
        "first_return": series.index[0].date().isoformat(),
        "last_return": series.index[-1].date().isoformat(),
        "daily_returns": len(series),
        # The bill's compound rate over the days used, 252 of them a year.
        "risk_free_compound": float(np.prod(1.0 + rate.to_numpy()) ** (1.0 / years) - 1.0),
        "weights": dict(zip(tickers, check_weights(weights, tickers).tolist(), strict=True)),
        "models": {model: [LABELS[c] for c in MODELS[model]] for model in models},
        "series": out,
    }


def _print(result: dict[str, Any]) -> None:
    weights = ", ".join(f"{t} {w:.1%}" for t, w in result["weights"].items() if w > 0)
    print(
        f"{result['daily_returns']} daily returns, {result['first_return']} to "
        f"{result['last_return']}; one-month T-bill {result['risk_free_compound']:.2%} a year"
    )
    print(f"portfolio, rebalanced daily: {weights}")
    print("alpha a year, betas per unit of factor; Newey-West t-statistics in brackets")
    for model, names in result["models"].items():
        print()
        print(f"{model:10}" + "".join(f" {n:>15}" for n in ["alpha", *names]) + f" {'R2':>6}")
        for name, fits in result["series"].items():
            fit = fits[model]
            cells = []
            for i, c in enumerate(fit["coefficients"].values()):
                value = f"{c['estimate']:.2%}" if i == 0 else f"{c['estimate']:.3f}"
                cells.append(f"{value} ({c['newey_west_t']:.1f})")
            print(f"{name:10}" + "".join(f" {c:>15}" for c in cells) + f" {fit['r_squared']:>6.3f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.factors", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--factors", type=Path, default=french.DEFAULT_CACHE)
    parser.add_argument(
        "--weights", default="equal", help='"equal" (the default) or e.g. SPY=0.6,AGG=0.4'
    )
    parser.add_argument(
        "--model",
        action="append",
        help=f"{' or '.join(MODELS)}; may be given more than once (default: both)",
    )
    parser.add_argument("--json", type=Path, help="also write the figures to this file")
    args = parser.parse_args(argv)
    try:
        prices = load_cache(args.cache)
        factors = french.load_factors(args.factors)
        result = {
            "cache_sha256": read_manifest(args.cache)["sha256"],
            "factors_sha256": french.read_manifest(args.factors)["sha256"],
            **run(prices, factors, args.weights, args.model or tuple(MODELS)),
        }
        # Written before anything is printed, so a failure leaves no half-finished run.
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.json, "w", encoding="utf-8", newline="\n") as out:
                json.dump(result, out, indent=2, allow_nan=False)
                out.write("\n")
    except (DataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
