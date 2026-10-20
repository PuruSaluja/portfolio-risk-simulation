"""Each fund's return and risk over the cached history, and how the funds move together.

``python -m portfolio_risk.summary`` reads the price cache (refused unless it matches its
manifest), computes daily total returns on the days every fund traded, and prints, per
fund:

- the compound return per calendar year, total and from price alone (the difference is
  what dividends added), annualised with the periods the data actually holds per year
  rather than the 252 convention, so it is the growth a holder saw;
- the annualised mean and volatility of daily returns (252 a year), and the volatility
  of monthly returns (12 a year): if daily returns were uncorrelated from day to day
  the two volatilities would agree, so a gap measures how far the square-root-of-time
  rule is off for that fund;
- the latest EWMA volatility (decay 0.94), what the recent past implies.

Then the correlation matrix of daily returns and the Ledoit-Wolf shrinkage intensity.
``--json`` writes the same figures to a file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_risk import covariance as cov
from portfolio_risk import returns as ret
from portfolio_risk.data import DEFAULT_CACHE, DataError, load_cache, read_manifest, wide

EWMA_DECAY = 0.94


def summarise(prices: pd.DataFrame) -> dict[str, Any]:
    daily = ret.total_returns(prices)
    price_only = ret.price_returns(prices)
    monthly = ret.compound(daily, "M")
    # The returns span from the first close every fund has to the last, so count the
    # periods over that.
    first_close = wide(prices, "close").dropna().index[0]
    closes = pd.DatetimeIndex([first_close, *daily.index])
    per_year = ret.observed_periods_per_year(closes)
    ewma = cov.ewma_covariance(daily, EWMA_DECAY)
    lw, shrinkage = cov.ledoit_wolf_covariance(daily)
    sample = cov.sample_covariance(daily)
    funds = {}
    for ticker in daily.columns:
        funds[ticker] = {
            "compound_return": float(ret.annualised_return(daily[ticker], per_year)),
            "price_only_compound_return": float(
                ret.annualised_return(price_only[ticker], per_year)
            ),
            "mean_return": float(ret.annualised_mean(daily[ticker], ret.TRADING_DAYS_PER_YEAR)),
            "volatility_daily": float(
                ret.annualised_volatility(daily[ticker], ret.TRADING_DAYS_PER_YEAR)
            ),
            "volatility_monthly": float(ret.annualised_volatility(monthly[ticker], 12)),
            "volatility_ewma_latest": math.sqrt(
                float(ewma.loc[ticker, ticker]) * ret.TRADING_DAYS_PER_YEAR
            ),
        }
    eigen_sample = pd.Series(sorted(_eigenvalues(sample)))
    eigen_lw = pd.Series(sorted(_eigenvalues(lw)))
    return {
        "first_close": first_close.date().isoformat(),
        "last_close": daily.index[-1].date().isoformat(),
        "daily_returns": len(daily),
        "monthly_returns": len(monthly),
        "trading_days_per_year_observed": per_year,
        "ewma_decay": EWMA_DECAY,
        "funds": funds,
        "correlation": {
            a: {b: float(v) for b, v in row.items()}
            for a, row in cov.correlation(sample).iterrows()
        },
        "ledoit_wolf_shrinkage": shrinkage,
        "condition_number": {
            "sample": float(eigen_sample.iloc[-1] / eigen_sample.iloc[0]),
            "ledoit_wolf": float(eigen_lw.iloc[-1] / eigen_lw.iloc[0]),
        },
    }


def _eigenvalues(matrix: pd.DataFrame) -> list[float]:
    return [float(v) for v in np.linalg.eigvalsh(matrix.to_numpy())]


def _print(summary: dict[str, Any]) -> None:
    print(
        f"{summary['daily_returns']} daily returns, {summary['first_close']} to "
        f"{summary['last_close']} ({summary['trading_days_per_year_observed']:.2f} "
        "trading days a year)"
    )
    print(f"{'':6} {'compound':>9} {'price':>7} {'mean':>7} {'vol':>7} {'vol(m)':>7} {'ewma':>7}")
    for ticker, f in summary["funds"].items():
        print(
            f"{ticker:<6} {f['compound_return']:>9.2%} {f['price_only_compound_return']:>7.2%} "
            f"{f['mean_return']:>7.2%} {f['volatility_daily']:>7.2%} "
            f"{f['volatility_monthly']:>7.2%} {f['volatility_ewma_latest']:>7.2%}"
        )
    tickers = list(summary["correlation"])
    print("\nCorrelation of daily returns")
    print("       " + " ".join(f"{t:>6}" for t in tickers))
    for a in tickers:
        print(f"{a:<6} " + " ".join(f"{summary['correlation'][a][b]:>6.2f}" for b in tickers))
    print(
        f"\nLedoit-Wolf shrinkage {summary['ledoit_wolf_shrinkage']:.4f}; condition number "
        f"{summary['condition_number']['sample']:.1f} sample, "
        f"{summary['condition_number']['ledoit_wolf']:.1f} shrunk"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.summary", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--json", type=Path, help="also write the figures to this file")
    args = parser.parse_args(argv)
    try:
        prices = load_cache(args.cache)
        sha256 = read_manifest(args.cache)["sha256"]
        summary = {"cache_sha256": sha256, **summarise(prices)}
        # Written before anything is printed, so a failure leaves no half-finished run.
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.json, "w", encoding="utf-8", newline="\n") as out:
                json.dump(summary, out, indent=2)
                out.write("\n")
    except (DataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
