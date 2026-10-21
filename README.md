# Portfolio Simulation and Risk Analysis

Monte Carlo simulation and risk analysis of multi-asset portfolios, on real
daily market data.

> **Status:** early. The data ingestion, the returns and covariance engine and
> the Monte Carlo simulation are done; the parts below are built in order, and every figure this README
> gives comes from a run of the committed code.

## The data is real

The prices are **real market data**: daily closes and dividends of eight
exchange-traded funds from Yahoo Finance, fetched with
[yfinance](https://github.com/ranaroussi/yfinance). The simulations model
or resample these returns, and say which; every comparison with history is with
these prices.

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
replaces. When it does (same prices, same window), the manifest is left as it
was, so fetching into a fresh clone changes no tracked file.
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

## Monte Carlo simulation

[`montecarlo.py`](src/portfolio_risk/montecarlo.py) fits a model to the funds'
daily total returns, draws paths of daily returns for all of them at once, and
follows a portfolio held at fixed weights and rebalanced to them at every close.
Three models, each answering for an assumption the one before makes:

- **Multivariate normal.** Daily log returns are normal, with the sample mean
  and covariance of the historical log returns, and each day is drawn
  independently. Log returns, so every path compounds to a positive price. No
  fat tails, no volatility clustering, no autocorrelation.
- **Multivariate Student-t.** The same mean and covariance, but t-shaped, with
  the degrees of freedom of greatest likelihood (the mean and covariance held
  at the sample's, so only the tails change). Every fund on a day shares one
  draw of the t's mixing variable, so a violent day is violent for all of them.
  Days are still independent. The likelihood matches scipy's `multivariate_t`
  to about 1e-15, and the fitted degrees of freedom scipy's optimiser to 7e-9. A
  fund that never varies, or one that is a mix of others, leaves the returns in
  fewer dimensions than funds; the likelihood is then the density within the
  space they span (for a fund that never varies, that of the other funds).
- **Stationary block bootstrap** (Politis and Romano, 1994). Paths are built
  from blocks of consecutive historical days, whole days at a time so the funds
  keep the moves they made together; each block starts on a random day and its
  length is geometric with a chosen mean, wrapping from the last day to the
  first. The returns are the real ones, with their tails, and within a block
  their autocorrelation and volatility clustering. By default the mean length
  is the funds' average of the length Politis and White (2004, with the 2009
  correction) choose for each, leaving out funds that never vary (any block
  reproduces them). The window is found as in the authors' reference code, and
  the lengths match the arch package's (8.0.0) to 1e-15 on these funds and to
  1e-9 on 273 of 300 random series; of the rest, 26 are where arch gives under a
  day, which is kept at 1, and one where a correlation sits at the significance
  threshold and arch's sums leave out a day. The tests also check it against the
  closed form for autoregressive and moving-average series.

Runs are reproducible: the seed fixes every path, every model draws from the
same seed, and path `i` is the same whatever number of paths is asked for (they
are drawn in blocks of 1,000, each from its own stream spawned from the seed).
The normal and t models factor the covariance by Cholesky, which is unique, so the
paths do not depend on the linear-algebra library numpy was built with: numpy
2.4.6 from Anaconda and 2.5.4 from PyPI give figures within 2e-14 of each other.

```bash
python -m portfolio_risk.montecarlo --weights equal --horizon 252 --paths 10000 --seed 0 \
    --model normal --model t --model bootstrap --model bootstrap:21
```

`--weights` also takes e.g. `SPY=0.6,AGG=0.4` (long only, adding up to 1);
`--model` defaults to `normal`, `t` and `bootstrap`, and `bootstrap:<days>` sets
the mean block length.

### Results

The eight funds in equal weights, 10,000 one-year (252-day) paths, seed 0
([`results/montecarlo.json`](results/montecarlo.json)), against the same
portfolio's history, 4 January 2005 to 30 September 2026. The t fit has 4.12
degrees of freedom (likelihood ratio against the normal 12,375, for one extra
parameter). Politis and White's lengths run from 1.8 days (GLD, none of whose
first 100 autocorrelations clears the method's significance threshold) to 12.1
(AGG and TLT), 7.8 on average.

| Daily return | Sd | Excess kurtosis | 0.1% quantile | 1% quantile | 99% quantile |
| --- | ---: | ---: | ---: | ---: | ---: |
| History | 0.871% | 12.92 | -5.07% | -2.41% | 2.35% |
| Normal | 0.872% | 0.01 | -2.61% | -1.97% | 2.09% |
| Student-t | 0.872% | 15.96 | -4.16% | -2.22% | 2.39% |
| Bootstrap, mean block 7.8 days | 0.872% | 12.86 | -5.10% | -2.43% | 2.36% |

| 252-day return | Mean | Sd | 0.1% quantile | 1% quantile | 5% quantile | Median | Chance of a loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| History | 9.07% | 11.92% | -31.82% | -25.49% | -14.62% | 9.65% | 16.1% |
| Normal | 9.46% | 15.05% | -30.31% | -21.83% | -13.40% | 8.62% | 27.6% |
| Student-t | 9.57% | 15.18% | -31.46% | -21.52% | -13.50% | 8.57% | 27.2% |
| Bootstrap, mean block 7.8 days | 9.20% | 13.57% | -32.66% | -21.39% | -12.56% | 8.83% | 24.8% |
| Bootstrap, 1 day | 8.89% | 14.83% | -30.46% | -21.06% | -13.51% | 7.93% | 28.3% |
| Bootstrap, 21 days | 9.07% | 12.90% | -31.90% | -22.07% | -12.06% | 8.99% | 22.5% |
| Bootstrap, 63 days | 8.84% | 12.31% | -38.13% | -24.17% | -12.27% | 9.21% | 20.7% |
| Bootstrap, 252 days | 8.82% | 12.05% | -32.93% | -25.17% | -13.52% | 9.25% | 17.3% |

The history rows take every day, and every window of 252 days (one starting on
each day, 5,218 in all). The windows overlap, so they hold only about 21
independent years, and their low quantiles are 2008 more than anything else.

What the comparison shows:

- **Assuming normality makes the daily tails far too thin.** The normal matches
  the mean and the standard deviation, but history's worst day in a thousand is
  a 5.07% loss and the normal's 2.61%. The t gets most of the way (4.16%), and
  the bootstrap, drawing real days, all of it (5.10%). The t's excess kurtosis
  is unstable, 11.6 to 22.4 over seeds 0 to 5, because at 4.12 degrees of
  freedom a t's fourth moment barely exists (its excess kurtosis is
  6 / (dof - 4) = 52).
- **Fat tails alone do not change the yearly picture.** Drawn independently, 252
  t days add up to nearly the same one-year distribution as 252 normal days
  (standard deviation 15.2% against 15.1%, chance of a loss 27.2% against
  27.6%), and so do 252 independently drawn real days (the 1-day bootstrap, 14.8%
  and 28.3%).
- **What the yearly picture needs is the dependence between days.** The
  portfolio's daily returns are negatively autocorrelated (lag one -0.097, like
  each fund's above), so part of each move reverses: measured over 252-day
  windows, history's volatility is 0.862 of the square-root-scaled daily
  figure. Any model that draws days independently overstates one-year risk: the
  normal's standard deviation is 15.1% against history's 11.9%, and it puts a
  loss in 27.6% of years against 16.1% of windows (4 of 21 full calendar years).
  Blocks keep that dependence: the standard deviation falls from 14.8% with
  1-day blocks to 13.6% at the Politis-White length, 12.9% at 21 days and 12.1%
  at 252. Politis and White choose blocks for the variance of a sample mean, and
  7.8 days keeps the short-lag reversals but not the longer ones.
- **Long blocks agree with history partly because they are history.** With a
  mean block of 252 days a one-year path is often a single stretch of the data,
  so the bootstrap's 1% quantile (-25.2%; -23.9% to -25.6% over seeds 0 to 5)
  and chance of a loss (17.3%) land near the historical windows' (-25.5%, 16.1%)
  largely by replaying them; with only
  about 21 independent years, it cannot show a year the data does not hold.
- The means agree: the normal's 9.46% against 9.28% expected from the fitted
  moments (`(mean of exp(mu + var / 2)) ^ 252 - 1`), 1.2 standard errors apart,
  and 9.07% over the historical windows. The 1-day bootstrap's 8.89% is 2.6
  standard errors below its exact expectation, `(1 + mean daily return) ^ 252 -
  1` = 9.28%, at seed 0; over seeds 0 to 9 the gap averages -0.11 standard
  errors.

Over seeds 0 to 5 the standard deviations and chances of a loss move by less
than the gaps the bullets above rest on. The standard deviations run 15.1% to
15.2% (normal), 15.0% to 15.3% (t), 14.8% to 15.3% (1-day blocks), 13.4% to
13.8% (Politis-White), 12.6% to 12.9% (21 days), 12.2% to 12.4% (63) and 11.8%
to 12.1% (252); the chances of a loss 27.6% to 28.7%, 27.2% to 29.3%, 27.5% to
28.4%, 24.4% to 25.4%, 22.5% to 23.6%, 20.1% to 20.8% and 17.3% to 17.8%. The
daily 0.1% quantiles: normal -2.61% to -2.62%, t -4.15% to -4.21%, bootstrap
-5.10% throughout. The one-year tail quantiles are noisier: the 0.1% quantile
rests on 10 of the 10,000 paths and moves by up to 6 points between seeds (63-day
blocks, -32.1% to -38.1%), and the 1% quantiles of the normal, the t and the
Politis-White bootstrap overlap (-21.2% to -22.5%), so the tables do not rank
those models by them.

## Parts

| Part | State |
| --- | --- |
| Data ingestion via yfinance, cached to CSV with a manifest | done |
| Returns and covariance engine, annualisation helpers | done |
| Monte Carlo: multivariate normal, Student-t, stationary block bootstrap | done |
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
