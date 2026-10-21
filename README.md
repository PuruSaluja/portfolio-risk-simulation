# Portfolio Simulation and Risk Analysis

Monte Carlo simulation and risk analysis of multi-asset portfolios, on real
daily market data.

> **Status:** early. The data ingestion and the returns and covariance engine
> are done; the parts below are built in order, and every figure this README
> gives comes from a run of the committed code.

## The data is real

The prices are **real market data**: daily closes and dividends of eight
exchange-traded funds from Yahoo Finance, fetched with
[yfinance](https://github.com/ranaroussi/yfinance). Nothing is generated or
simulated at this stage; the simulations to come resample or model these
returns, and say which.

| Ticker | Fund | Asset class |
| --- | --- | --- |
| SPY | SPDR S&P 500 ETF Trust | US large-cap equity |
| IWM | iShares Russell 2000 ETF | US small-cap equity |
| EFA | iShares MSCI EAFE ETF | Developed ex-US equity |
| EEM | iShares MSCI Emerging Markets ETF | Emerging-market equity |
| AGG | iShares Core US Aggregate Bond ETF | US investment-grade bonds |
| TLT | iShares 20+ Year Treasury Bond ETF | Long US Treasuries |
| GLD | SPDR Gold Shares | Gold |
| VNQ | Vanguard Real Estate ETF | US real estate (REITs) |

Every one of them traded throughout the default window, January 2005 to
September 2026 (GLD, the youngest, listed in November 2004), so the history
covers 2008, 2020 and 2022 without a fund joining part way.

**Prices are not committed.** yfinance's documentation notes that Yahoo's
data is meant for personal use, so the repository does not redistribute it.
A fetch writes the prices to a local CSV cache, and every later run reads
that file and works offline. What is committed is the cache's manifest: the
window, each fund's rows, first and last dates and dividend count, and a
SHA-256 of the file. A run refuses a cache that does not match its manifest,
so results are always computed from the data they say they were.

Yahoo revises history now and then (a corrected dividend, a late print), so a
fresh fetch on another day can differ from the committed manifest; the
manifest check then says so rather than quietly computing different figures.

### Getting the data

```bash
python -m portfolio_risk.data fetch     # download into data/prices.csv
python -m portfolio_risk.data verify    # check the cache against its manifest
```

`fetch` takes `--tickers`, `--start`, `--end` (inclusive, and before today in
New York) and `--out`, and says whether the new file matches the manifest it
replaces.
[`data/prices.manifest.json`](data/prices.manifest.json) is the committed
manifest, from a fetch on 2026-10-20: 5,470 trading days for each fund,
3 January 2005 to 30 September 2026, every fund on every day (43,760 rows).

The cache keeps each day's close (split-adjusted, not dividend-adjusted),
dividend, capital-gain distribution, split ratio and volume. The fetch
checks that every fund has prices from within a week of the window's start
and a price on every day any of them traded (so no fund joins, leaves or
drops out part way through, and no day is lost to every fund's returns);
that the window ends before today in New York, since a bar fetched during a
session holds the latest trade rather than the close; that closes are
positive and finite; and that dividends, splits and volumes are not
negative. Over the window it holds 865 dividends (none for GLD, which pays
none; 259 and 260 monthly ones for TLT and AGG), no capital-gain
distributions, and four splits: IWM 2-for-1, EFA and EEM 3-for-1 on 9 June
2005, and EEM 3-for-1 again on 24 July 2008. No fund's close moves more than
4% on a split day, so the closes are adjusted for them.

**Why Yahoo's adjusted close is not kept.** Two fetches a few seconds apart
gave adjusted closes that differed by up to 2 parts in a million on 26,377
of the 43,760 rows, while every other field was identical. Yahoo computes
it afresh on each request; kept, it would change the file's checksum on
every fetch. Total returns are computed from the close and dividends
instead (see below), and without it the same data gives the same file:
fetches on 2026-10-20, hours apart and with yfinance 0.2.58 to 1.7.0, gave the
same SHA-256 as the committed manifest.

## Returns and risk of each fund

[`returns.py`](src/portfolio_risk/returns.py) computes daily total returns from the
cache: `(close + dividend + distribution) / previous close - 1`, the convention
CRSP uses, on the days every fund traded (all 5,470 here). It compounds them into
weekly to yearly returns and annualises: the arithmetic mean times 252, the
volatility times the square root of 252, and the compound growth rate.
[`covariance.py`](src/portfolio_risk/covariance.py) estimates covariance three
ways: the sample estimate, an exponentially weighted one in the RiskMetrics style
(decay 0.94, so a day's weight halves every 11.2 days), and Ledoit-Wolf shrinkage
toward a scaled identity, which matches scikit-learn 1.9.0's `LedoitWolf` to
within 1e-12.

**Checked against Yahoo's adjusted close.** Away from ex-dividend days the daily
returns agree with those from Yahoo's adjusted close to within 2.3e-6 (the
adjusted close's own rounding). On ex-dates they differ by up to 6e-4, because
Yahoo divides today's close by the previous close less the dividend where these
add the dividend to today's close (that formula reproduces Yahoo's ex-date
returns to within 2.1e-6); over the whole window the compound growth rates agree
within 0.01 percentage points a year for every fund.

`python -m portfolio_risk.summary --json results/returns-summary.json` gives,
from 3 January 2005 to 30 September 2026
([`results/returns-summary.json`](results/returns-summary.json)):

| Fund | Compound return | Price only | Mean (daily x 252) | Volatility, daily | Volatility, 21-day windows | Volatility, calendar months | EWMA volatility, 30 Sep 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY | 10.87% | 8.87% | 12.12% | 18.88% | 15.82% | 14.79% | 10.33% |
| IWM | 8.42% | 6.99% | 10.99% | 23.97% | 21.30% | 19.82% | 12.28% |
| EFA | 6.13% | 3.14% | 8.16% | 20.96% | 17.84% | 16.69% | 12.58% |
| EEM | 7.28% | 5.20% | 10.84% | 27.62% | 22.19% | 20.90% | 21.54% |
| AGG | 2.86% | -0.37% | 2.96% | 5.20% | 4.58% | 4.49% | 5.27% |
| TLT | 2.69% | -0.60% | 3.71% | 14.53% | 13.52% | 13.62% | 11.28% |
| GLD | 10.55% | 10.55% | 11.73% | 18.29% | 17.42% | 17.30% | 25.11% |
| VNQ | 6.63% | 2.20% | 10.47% | 28.38% | 22.76% | 21.50% | 11.52% |

Compound returns are per calendar year: the data holds 251.58 trading days a
year, not the conventional 252, and annualising with 252 would overstate SPY's
by 0.02 points. "Price only" leaves dividends out; for the two bond funds it is
negative, so their interest payments made up more than their whole return. The
mean and daily volatility columns use 252, the 21-day column 252 / 21 windows a
year, and the calendar-month column 12.

**The square-root-of-time rule overstates volatility here.** Scaling daily
volatility by the square root of 252 assumes returns are uncorrelated from day
to day. They are not: every fund's lag-one autocorrelation is negative, from
-0.016 (GLD) to -0.157 (VNQ), so on average part of a day's move is reversed
over the following days. Measured on returns over every 21-trading-day window,
volatility is 0.80 to 0.95 of the scaled daily figure (SPY 15.82% against
18.88%), and the ratio the first 20 autocorrelations imply matches the measured
one to within 0.013 for every fund:

| Fund | Lag-one autocorrelation | 21-day / scaled daily, measured | Implied by lags 1-20 | Calendar months / scaled daily |
| --- | ---: | ---: | ---: | ---: |
| SPY | -0.104 | 0.838 | 0.835 | 0.783 |
| IWM | -0.073 | 0.888 | 0.898 | 0.827 |
| EFA | -0.111 | 0.851 | 0.850 | 0.796 |
| EEM | -0.120 | 0.803 | 0.794 | 0.757 |
| AGG | -0.050 | 0.880 | 0.878 | 0.863 |
| TLT | -0.029 | 0.930 | 0.924 | 0.937 |
| GLD | -0.016 | 0.952 | 0.945 | 0.946 |
| VNQ | -0.157 | 0.802 | 0.814 | 0.758 |

Calendar months give lower figures still for the equity and REIT funds and
about the same for the others. There are only 261 of them, so that estimate
is the noisier one (about 4% relative error for normal returns, more with fat
tails); the 21-day windows start on every day.

Correlations of daily returns: the four equity funds move together (0.76 to
0.89) and with REITs (0.64 to 0.76); long Treasuries move against equities
(-0.25 to -0.30) and with the aggregate bond fund (0.71); gold's largest
correlation with any other fund is 0.21. Ledoit-Wolf shrinks the sample
covariance by only 0.5% (5,469 days for 8 funds leaves little estimation error),
bringing its condition number from 234 to 193.

## Parts

| Part | State |
| --- | --- |
| Data ingestion via yfinance, cached to CSV with a manifest | done |
| Returns and covariance engine, annualisation helpers | done |
| Monte Carlo: multivariate normal, Student-t, stationary block bootstrap | to come |
| Risk metrics: Sharpe, Sortino, drawdown, VaR, CVaR, downside deviation | to come |
| Scenarios: allocation mixes, rebalancing, stress periods | to come |
| Efficient frontier, mean-variance, risk parity and CVaR optimisation | to come |
| Fama-French factor attribution (Ken French data library) | to come |
| Cost-aware backtest: transaction costs, spreads, rebalancing drag | to come |
| Tableau extracts, dashboard spec and Plotly reproduction | to come |

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -c constraints.txt -e ".[dev]"   # bin/ on Linux and macOS
pytest
ruff check . && ruff format --check .
```

The tests never reach Yahoo: they stand in for the download.

## License

MIT; see [LICENSE](LICENSE). The license covers the code, not Yahoo's data.
