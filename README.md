# Portfolio Simulation and Risk Analysis

Monte Carlo simulation and risk analysis of multi-asset portfolios, on real
daily market data.

> **Status:** in progress. The data ingestion, the returns and covariance
> engine, the Monte Carlo simulation, the risk measures, the scenario engine,
> the mean-variance optimiser, the factor attribution, and risk parity and
> minimum CVaR are done; the parts below are built in order, and every figure
> this README gives comes from a run of the committed code.

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

### Factors and the risk-free rate

The Fama-French factors and the one-month Treasury bill rate are also **real
data**, from [Kenneth French's data
library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html):
the daily three-factor file (`Mkt-RF`, `SMB`, `HML`, `RF`), the daily
five-factor file (2x3 sorts, adding `RMW` and `CMA`) and the monthly
three-factor file, for its bill rate. Like the prices, the figures are cached
locally and not committed; the manifest
[`data/factors.manifest.json`](data/factors.manifest.json) is, with the three
files' URLs and the CRSP database each was built from.

```bash
python -m portfolio_risk.french fetch     # download into data/factors.csv
python -m portfolio_risk.french verify    # check the cache against its manifest
```

The committed manifest is from a fetch on 2026-10-24 of files built from the
August 2026 CRSP database: 5,449 trading days, 3 January 2005 to 31 August
2026. The library publishes whole months a month or two behind, so the
factors end a month before the prices; every result that uses them is over
the days both cover. The fetch checks that the two daily files hold the same
days and the same `Mkt-RF`, `HML` and `RF` (they do on every day of the
window), and that the monthly file has a rate for every month. Each file names
the CRSP database it was built from, and the library rebuilds the files from
newer ones, so a later fetch can differ from the manifest; the check then
says so.

**The daily bill rate is rebuilt from the monthly one.** The daily files give
`RF` to two decimals of a percent: 0.00, 0.01 or 0.02 on every day of the
window. Compounded, those rounded figures give 5.13% for 2025 where the
monthly file's rates compound to 4.25%, and 0.00% for 2010 against 0.10%. The
files' notes describe the daily rate as the simple daily rate that compounds to
the month's rate over the month's trading days, and it is published to two
decimals; `french.daily_risk_free` computes that rate without the rounding. Rounded, it gives
the published figure on 5,407 of the 5,449 days. Over the window it compounds
to 1.74% a year.

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
  to about 1e-15, and scipy's bounded and Brent searches find degrees of
  freedom within 4e-7 of the fitted ones, where the likelihood is flat to its
  last digit, so the fit is no finer than that. A fund that never varies, or
  one that is a mix of others, leaves the returns in fewer dimensions than
  funds; the likelihood is then the density within the space they span (for a
  fund that never varies, that of the other funds).
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
    --model normal --model t --model bootstrap --model bootstrap:1 --model bootstrap:21 \
    --model bootstrap:63 --model bootstrap:252 --json results/montecarlo.json
```

`--weights` also takes e.g. `SPY=0.6,AGG=0.4` (long only, adding up to 1);
`--model` defaults to `normal`, `t` and `bootstrap`, and `bootstrap:<days>` sets
the mean block length.

### Results

The eight funds in equal weights, 10,000 one-year (252-day) paths, seed 0
([`results/montecarlo.json`](results/montecarlo.json)), against the same
portfolio's history, 4 January 2005 to 30 September 2026. The t fit has 4.12
degrees of freedom (likelihood ratio against the normal 12,375, for one extra
parameter). Politis and White's lengths run from 1.8 days (GLD: none of its
first 100 autocorrelations clears the method's significance threshold, so its
window is the shortest the method uses, lag 1 alone) to 12.1 (AGG and TLT), 7.8
on average.

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
  largely by replaying them; with only about 21 independent years, it cannot
  show a year the data does not hold.
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

## Risk measures

[`risk.py`](src/portfolio_risk/risk.py) measures one series of returns:

- **Sharpe ratio**: the mean return in excess of a risk-free rate over the
  standard deviation, on a yearly scale.
- **Downside deviation** and the **Sortino ratio**: the root mean square of
  the shortfalls below a target (returns above it count as zero), over all
  returns, and the mean excess over the target divided by it.
- **Maximum drawdown**: the deepest fall of the wealth index from its highest
  point so far, with the dates of the peak, the trough and the recovery. The
  index starts at 1 before the first return, so a first-day loss counts.
- **Value at risk** and **expected shortfall** (CVaR), historical and normal,
  reported as positive losses. Historical value at risk is the lowest return
  with at least 5% (or 1%) of returns at or below it; expected shortfall is the
  mean of the worst 5% (1%), the boundary return counted in part when that is
  not a whole number of returns (Acerbi and Tasche, 2002), so it is never below
  value at risk. The normal figures come from the sample mean and standard
  deviation.

The tests check each measure against direct formulas and brute force, and a
pandas recalculation on the real data agrees to 2e-17.

```bash
python -m portfolio_risk.risk --weights equal --risk-free 0 --json results/risk-metrics.json
python -m portfolio_risk.risk --weights equal --risk-free bills --json results/risk-metrics-bills.json
```

`--risk-free` takes a constant annual rate, or `bills` for the daily
one-month Treasury bill rate from the factor cache (see [Factors and the
risk-free rate](#factors-and-the-risk-free-rate)), subtracted day by day; the
monthly ratio is then against the bill compounded over the same months, and
the measures cover the days the factor cache covers. The first table below is
against a zero rate over the full price history, so its ratios are of total
return; the bill-rate ratios follow it.

### Results

From the daily total returns, 4 January 2005 to 30 September 2026 (5,469
days), the eight funds and the equal-weight portfolio rebalanced daily
([`results/risk-metrics.json`](results/risk-metrics.json)):

| Series | Compound return | Volatility | Sharpe, daily | Sharpe, monthly | Sortino | Max drawdown | Peak | Trough | Recovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| SPY | 10.87% | 18.88% | 0.64 | 0.77 | 0.91 | -55.2% | 2007-10-09 | 2009-03-09 | 2012-08-16 |
| IWM | 8.42% | 23.97% | 0.46 | 0.51 | 0.65 | -58.6% | 2007-07-13 | 2009-03-09 | 2011-02-08 |
| EFA | 6.13% | 20.96% | 0.39 | 0.44 | 0.55 | -61.0% | 2007-10-31 | 2009-03-09 | 2014-06-06 |
| EEM | 7.28% | 27.62% | 0.39 | 0.44 | 0.57 | -66.4% | 2007-10-31 | 2008-11-20 | 2017-09-19 |
| AGG | 2.86% | 5.20% | 0.57 | 0.65 | 0.79 | -18.4% | 2020-08-04 | 2022-10-20 | not by 30 Sep 2026 |
| TLT | 2.69% | 14.53% | 0.26 | 0.26 | 0.37 | -48.3% | 2020-08-04 | 2023-10-19 | not by 30 Sep 2026 |
| GLD | 10.55% | 18.29% | 0.64 | 0.67 | 0.91 | -45.6% | 2011-08-22 | 2015-12-17 | 2020-07-29 |
| VNQ | 6.63% | 28.38% | 0.37 | 0.41 | 0.53 | -73.1% | 2007-02-07 | 2009-03-06 | 2012-09-13 |
| Equal weight | 8.22% | 13.83% | 0.64 | 0.75 | 0.92 | -38.3% | 2007-10-31 | 2008-11-20 | 2010-01-11 |

| One day's loss | 95% VaR, historical | 95% VaR, normal | 99% VaR, historical | 99% VaR, normal | 99% CVaR, historical | 99% CVaR, normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY | 1.75% | 1.91% | 3.46% | 2.72% | 5.09% | 3.12% |
| IWM | 2.27% | 2.44% | 3.93% | 3.47% | 5.92% | 3.98% |
| EFA | 1.94% | 2.14% | 3.81% | 3.04% | 5.71% | 3.49% |
| EEM | 2.46% | 2.82% | 4.85% | 4.01% | 7.16% | 4.59% |
| AGG | 0.45% | 0.53% | 0.80% | 0.75% | 1.33% | 0.86% |
| TLT | 1.47% | 1.49% | 2.30% | 2.12% | 2.91% | 2.43% |
| GLD | 1.80% | 1.85% | 3.18% | 2.63% | 4.46% | 3.02% |
| VNQ | 2.35% | 2.90% | 5.46% | 4.12% | 8.17% | 4.72% |
| Equal weight | 1.25% | 1.40% | 2.43% | 1.99% | 3.65% | 2.29% |

What they show:

- **A normal model gets the tail the wrong way round.** At 95% its value at risk
  is larger than the historical one for every fund (by 1.7% for TLT to 23.5%
  for VNQ): fat-tailed returns have more quiet days than a normal with the same
  standard deviation. At 99% it is smaller for every fund, and the expected
  shortfall, the mean of the worst 1% of days, is 0.58 (VNQ) to 0.83 (TLT) of
  the historical one; for the portfolio, 2.29% against 3.65%, an understatement
  of 37%. This is the daily form of the Monte Carlo section's finding.
- **Diversification shows in the drawdown more than in the ratios.** The
  equal-weight portfolio's Sharpe ratio (0.64) is SPY's and gold's, but its
  deepest fall is -38.3% against SPY's -55.2%, and it was back at its high 552
  trading days after its peak, where SPY took 1,224, EFA 1,660 and EEM 2,488.
- **The bond funds' worst fall is the recent one.** AGG and TLT peaked on 4
  August 2020 and fell 18.4% and 48.3% as rates rose; neither had recovered by
  the end of the data, so the 2008 crisis is not the worst period for every
  asset.
- **The monthly Sharpe ratio is higher than the daily one** for every fund (by
  2.5% for TLT to 20.7% for SPY), because the volatility differs more than the
  mean. Square-root-scaled daily volatility is 1.06 to 1.32 times the monthly
  figure (as the returns section measures), while the monthly mean times 12 is
  0.84 to 0.99 of the daily mean times 252, so a ratio scaled from daily data
  understates them.

**Against the Treasury bill.** Over the days the factor cache covers, 4
January 2005 to 31 August 2026 (5,448 days), with the one-month bill at 1.74%
a year compounded
([`results/risk-metrics-bills.json`](results/risk-metrics-bills.json)):

| Series | Compound return | Sharpe, daily | Sharpe, monthly | Sortino | Sharpe against zero (above) | Sortino against zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY | 10.93% | 0.55 | 0.66 | 0.78 | 0.64 | 0.91 |
| IWM | 8.72% | 0.40 | 0.44 | 0.56 | 0.46 | 0.65 |
| EFA | 6.32% | 0.32 | 0.35 | 0.44 | 0.39 | 0.55 |
| EEM | 7.32% | 0.33 | 0.36 | 0.48 | 0.39 | 0.57 |
| AGG | 3.00% | 0.26 | 0.30 | 0.36 | 0.57 | 0.79 |
| TLT | 2.96% | 0.15 | 0.15 | 0.22 | 0.26 | 0.37 |
| GLD | 10.95% | 0.57 | 0.59 | 0.80 | 0.64 | 0.91 |
| VNQ | 6.97% | 0.32 | 0.34 | 0.46 | 0.37 | 0.53 |
| Equal weight | 8.45% | 0.53 | 0.62 | 0.76 | 0.64 | 0.92 |

The window ends a month earlier than the table above's, which moves the
compound returns a little (SPY 10.93% against 10.87%); the drawdowns are the
same. Against the bill every daily Sharpe ratio is lower than against zero, by
11.5% (GLD) to 19.0% (EFA) for the equity funds and gold, by 17.1% for the
equal-weight portfolio, and by 39.4% for TLT and 53.8% for AGG: a bond fund
earning 3% a year earned little more than the bill, so most of its zero-rate
ratio was the bill's. AGG's daily Sharpe ratio, 0.57 against zero (only SPY,
gold and the portfolio were higher, at 0.64), is 0.26 against the bill, below
every equity fund's. The bill compounds to 4.25% to 5.25% a year in 2023 to
2025 and 2006 to 2007, and to at most 0.10% in each year from 2009 to 2015
and in 2021, so the gap between the two tables depends on the window.

## Scenarios: allocation mixes, rebalancing and stress periods

[`scenarios.py`](src/portfolio_risk/scenarios.py) backtests a portfolio through
the historical daily returns. It is invested at its target weights at the first
close; each day every holding grows by its fund's return, so the weights drift,
and a rule decides when they are reset to the targets: every close (`daily`,
the fixed-weight portfolio of the sections above), the last close of each
month, quarter or year, `never` (buy and hold), or `band:x`, whenever a weight
has drifted more than `x` from its target. Turnover at a rebalance is half the
weight traded. **No trading costs are charged here**; [Trading
costs](#trading-costs) charges them.

Four mixes: **equity** (SPY, IWM, EFA and EEM, 25% each), **60/40** (SPY and
AGG), **equal weight** (all eight funds) and **permanent** (SPY, TLT, GLD and
AGG, 25% each, after Harry Browne's permanent portfolio, with AGG in place of
his cash). Six stress periods, each an S&P 500 fall from its closing high to
its closing low; SPY's own closes peak and bottom on the same days in the data.
A stress test invests a mix at its targets at the close of the first day and
follows it to the close of the last.

The tests check buy-and-hold, monthly, quarterly and yearly paths against
direct products of each fund's growth, and daily rebalancing against the
fixed-weight portfolio. On the real data a separate recalculation of the
daily, monthly and buy-and-hold compound returns agrees to within 1e-15, and
every buy-and-hold stress return matches the funds' growth over the window.

```bash
python -m portfolio_risk.scenarios --json results/scenarios.json
```

`--mix` takes a named mix or weights such as `SPY=0.6,AGG=0.4`, `--rule` a
rule, and `--period` a named period or `START:END`; each may be given more than
once, and a rule or mix given twice, in any spelling, is refused. `--compare`
(default `daily,monthly`) names two rules whose gap is measured again with
the days shuffled, `--shuffles` times (default 40, from `--seed`, default 0).

### Results

4 January 2005 to 30 September 2026
([`results/scenarios.json`](results/scenarios.json)); compound returns per
calendar year, Sharpe ratios against a zero risk-free rate:

| Mix | Rebalancing | Compound return | Volatility | Sharpe | Max drawdown | Rebalances | Turnover a year |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Equity | daily | 8.57% | 21.38% | 0.49 | -58.0% | 5,468 | 49.1% |
| Equity | monthly | 8.46% | 21.34% | 0.49 | -58.5% | 260 | 10.0% |
| Equity | quarterly | 8.43% | 21.32% | 0.49 | -58.4% | 86 | 5.3% |
| Equity | yearly | 8.49% | 21.27% | 0.49 | -58.5% | 21 | 2.9% |
| Equity | never (buy and hold) | 8.48% | 21.84% | 0.48 | -59.3% | 0 | 0.0% |
| Equity | 5-point band | 8.54% | 21.37% | 0.49 | -58.0% | 8 | 2.2% |
| 60/40 | daily | 8.09% | 11.53% | 0.73 | -34.7% | 5,468 | 48.6% |
| 60/40 | monthly | 7.87% | 11.36% | 0.72 | -35.8% | 260 | 9.3% |
| 60/40 | quarterly | 7.96% | 11.26% | 0.74 | -35.0% | 86 | 6.0% |
| 60/40 | yearly | 8.02% | 11.09% | 0.75 | -33.7% | 21 | 3.2% |
| 60/40 | never (buy and hold) | 8.91% | 12.52% | 0.75 | -33.3% | 0 | 0.0% |
| 60/40 | 5-point band | 8.13% | 11.54% | 0.74 | -35.0% | 19 | 4.5% |
| Equal weight | daily | 8.22% | 13.83% | 0.64 | -38.3% | 5,468 | 81.5% |
| Equal weight | monthly | 7.72% | 13.52% | 0.62 | -39.4% | 260 | 15.8% |
| Equal weight | quarterly | 7.85% | 13.26% | 0.64 | -38.3% | 86 | 9.3% |
| Equal weight | yearly | 7.83% | 12.94% | 0.65 | -37.0% | 21 | 4.7% |
| Equal weight | never (buy and hold) | 7.68% | 13.33% | 0.62 | -38.8% | 0 | 0.0% |
| Equal weight | 5-point band | 8.41% | 13.73% | 0.66 | -38.0% | 12 | 5.1% |
| Permanent | daily | 7.56% | 8.08% | 0.94 | -21.3% | 5,468 | 67.7% |
| Permanent | monthly | 7.29% | 8.03% | 0.92 | -21.6% | 260 | 13.8% |
| Permanent | quarterly | 7.35% | 8.04% | 0.92 | -21.6% | 86 | 8.0% |
| Permanent | yearly | 7.45% | 8.03% | 0.94 | -21.3% | 21 | 4.2% |
| Permanent | never (buy and hold) | 8.14% | 9.43% | 0.88 | -21.0% | 0 | 0.0% |
| Permanent | 5-point band | 7.53% | 8.11% | 0.94 | -21.4% | 20 | 5.2% |

Returns over each stress period, the mixes rebalanced monthly:

| Period | Days | SPY | AGG | TLT | GLD | Equity | 60/40 | Equal weight | Permanent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2008 financial crisis (2007-10-09 to 2009-03-09) | 355 | -55.2% | 7.4% | 25.0% | 23.9% | -58.0% | -35.8% | -38.6% | -5.2% |
| 2011 euro debt crisis (2011-04-29 to 2011-10-03) | 108 | -18.6% | 5.3% | 34.5% | 5.6% | -25.9% | -9.6% | -11.2% | 5.9% |
| 2015-16 China and oil (2015-05-21 to 2016-02-11) | 183 | -13.0% | 2.1% | 13.2% | 2.9% | -22.6% | -7.0% | -10.5% | 1.4% |
| 2018 fourth quarter (2018-09-20 to 2018-12-24) | 65 | -19.3% | 1.6% | 4.5% | 5.0% | -18.0% | -11.2% | -9.2% | -2.2% |
| 2020 COVID crash (2020-02-19 to 2020-03-23) | 23 | -33.7% | -1.3% | 14.2% | -3.6% | -34.5% | -21.5% | -22.2% | -6.9% |
| 2022 inflation and rates (2022-01-03 to 2022-10-12) | 195 | -24.5% | -14.4% | -29.3% | -7.3% | -26.5% | -20.4% | -23.7% | -19.0% |

What they show:

- **Left alone, a mix stops being the mix.** Never rebalanced, the 60/40 ended
  88.5% SPY and 11.5% AGG, and the permanent portfolio 43.0% SPY and 40.4%
  gold, the funds that grew fastest. Both returned more than any rebalanced
  version (8.91% and 8.14% a year), and with more volatility (12.52% and
  9.43%): they were no longer the portfolios they started as. The equal-weight
  mix left alone returned the least of its six (7.68%).
- **Rebalancing daily returned more than monthly, quarterly or yearly for every
  mix**, by 0.11 (equity) to 0.50 (equal weight) points a year over monthly.
  The gap comes from the order of the days. Shuffling the days keeps each
  day's returns together and the dates where they were, so only the dependence
  between days is lost; over 40 shuffles (seed 0) the shuffled gaps average
  -0.005 to 0.003 points by mix, their 95th percentiles are 0.039 to 0.107, and
  none is as large as the real gap. Over seeds 0 to 4, 200 shuffles a mix, the
  averages run -0.025 to 0.003 and still none is as large. That fits the
  negative day-to-day autocorrelation measured above: selling what rose today
  gains when part of the rise reverses. It is one history, before costs, and
  daily rebalancing trades 48.6% to 81.5% of the portfolio a year against 9.3%
  to 15.8% monthly, so costs decide whether any of it survives ([Trading
  costs](#trading-costs): at one-way costs below 12.9 to 35.2 basis points,
  by mix). A
  5-point band traded 2.2% to 5.2% a year, close
  to yearly rebalancing (2.9% to 4.7%), and returned 8.41% for equal weight,
  the most of its six.
- **Long Treasuries rose in every stress period but 2022**, and gold and AGG
  in every one but 2020 and 2022. In 2008 long Treasuries gained 25.0% and
  gold 23.9% while SPY fell 55.2%, and the permanent portfolio lost 5.2%
  against the equity mix's 58.0%. In 2022 rising rates took TLT down
  29.3% and AGG 14.4% alongside equities, and the permanent portfolio lost
  19.0%, close to the 60/40's 20.4%.
- **Within a crash, rebalancing buys the falling asset.** Over the 2008 window,
  never rebalancing lost 30.2% for the 60/40 against 35.8% rebalanced monthly,
  and 30.7% against 38.6% for equal weight. In the shorter windows the rules
  differ by at most 1.5 points for any mix.

## Trading costs

[`costs.py`](src/portfolio_risk/costs.py) charges the scenario backtests for
their trades. A trade of a value `x` in a fund costs `k x`, where `k`, the
one-way cost per unit traded, is a commission plus half the fund's quoted
bid-ask spread (a round trip crosses the whole spread). The costs are paid
out of the portfolio at the rebalancing close, which then holds exactly its
target weights: with the portfolio worth 1 before trading, drifted weights
`d` and targets `w`, the value left `v` solves `v + sum k_i |w_i v - d_i| = 1`,
which Newton's method solves exactly. Costs scale the portfolio's value and
never its weights, so the closes a rule trades on, and the weights it trades
from, are those of the cost-free backtest, for the band rule too. The
purchase at the first close is not charged (every rule makes it), nor are
reinvested distributions; costs are proportional, with no fixed fees, market
impact or taxes.

**The costs are levels, not measurements.** The price cache holds daily
closes only, with no bid, ask, high or low, so it cannot say what these funds
cost to trade. The command reads every backtest at one-way costs of 0, 1, 5,
10, 25 and 50 basis points per unit traded, and finds the **break-even cost**,
the same for every fund, at which one rule's lead over another is gone: the
lowest such cost. The leading rule's return can fall only so fast as the cost
rises, by a bound computed from its trades, so a lead of `g` cannot vanish
within `g` over that bound; the search steps that far at a time (at least a
thousandth of a basis point) and then bisects, so it cannot step over a
crossing.

The tests check the net returns against a loop that holds the funds in money,
pays the costs inside it and decides the band rule's trades from its own
after-cost weights: they agree to 1e-15 for five rules and three cost
settings, one with a different cost for each fund. Newton's method is checked
against bisection for 2 to 50 funds. The bound is checked against the
measured fall of the return for four rules at costs up to 10%, where it is
at most 1.35 times the steepest fall, and the search against leads that dip
below zero between two basis points and recover.

```bash
python -m portfolio_risk.costs --json results/costs.json
```

`--cost` sets a level (it may be given more than once), `--spread` adds half
of a quoted spread to each, one for every fund (`0.0002`) or one each
(`SPY=0.0001,EEM=0.0006`), `--mix` and `--rule` are as in the scenarios, and
`--compare` names two rules for a break-even cost (by default `daily,monthly`,
`daily,band:0.05` and `band:0.05,never`).

### Results

4 January 2005 to 30 September 2026
([`results/costs.json`](results/costs.json)), compound return a year before
costs and after a one-way cost per unit traded of:

| Mix | Rebalancing | Before costs | Turnover a year | 1 bp | 5 bp | 10 bp | 25 bp | 50 bp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Equity | daily | 8.57% | 49.1% | 8.56% | 8.52% | 8.46% | 8.31% | 8.04% |
| Equity | monthly | 8.46% | 10.0% | 8.46% | 8.45% | 8.44% | 8.41% | 8.35% |
| Equity | quarterly | 8.43% | 5.3% | 8.43% | 8.43% | 8.42% | 8.40% | 8.38% |
| Equity | yearly | 8.49% | 2.9% | 8.49% | 8.49% | 8.48% | 8.47% | 8.46% |
| Equity | never (buy and hold) | 8.48% | 0.0% | 8.48% | 8.48% | 8.48% | 8.48% | 8.48% |
| Equity | 5-point band | 8.54% | 2.2% | 8.54% | 8.54% | 8.54% | 8.53% | 8.52% |
| 60/40 | daily | 8.09% | 48.6% | 8.08% | 8.04% | 7.99% | 7.83% | 7.57% |
| 60/40 | monthly | 7.87% | 9.3% | 7.87% | 7.86% | 7.85% | 7.82% | 7.77% |
| 60/40 | quarterly | 7.96% | 6.0% | 7.96% | 7.96% | 7.95% | 7.93% | 7.90% |
| 60/40 | yearly | 8.02% | 3.2% | 8.02% | 8.02% | 8.02% | 8.01% | 7.99% |
| 60/40 | never (buy and hold) | 8.91% | 0.0% | 8.91% | 8.91% | 8.91% | 8.91% | 8.91% |
| 60/40 | 5-point band | 8.13% | 4.5% | 8.13% | 8.12% | 8.12% | 8.10% | 8.08% |
| Equal weight | daily | 8.22% | 81.5% | 8.20% | 8.13% | 8.05% | 7.78% | 7.34% |
| Equal weight | monthly | 7.72% | 15.8% | 7.72% | 7.71% | 7.69% | 7.64% | 7.55% |
| Equal weight | quarterly | 7.85% | 9.3% | 7.85% | 7.84% | 7.83% | 7.80% | 7.75% |
| Equal weight | yearly | 7.83% | 4.7% | 7.83% | 7.82% | 7.82% | 7.80% | 7.78% |
| Equal weight | never (buy and hold) | 7.68% | 0.0% | 7.68% | 7.68% | 7.68% | 7.68% | 7.68% |
| Equal weight | 5-point band | 8.41% | 5.1% | 8.41% | 8.41% | 8.40% | 8.39% | 8.36% |
| Permanent | daily | 7.56% | 67.7% | 7.55% | 7.49% | 7.42% | 7.20% | 6.84% |
| Permanent | monthly | 7.29% | 13.8% | 7.28% | 7.27% | 7.26% | 7.21% | 7.14% |
| Permanent | quarterly | 7.35% | 8.0% | 7.35% | 7.34% | 7.33% | 7.31% | 7.26% |
| Permanent | yearly | 7.45% | 4.2% | 7.45% | 7.44% | 7.44% | 7.43% | 7.40% |
| Permanent | never (buy and hold) | 8.14% | 0.0% | 8.14% | 8.14% | 8.14% | 8.14% | 8.14% |
| Permanent | 5-point band | 7.53% | 5.2% | 7.53% | 7.53% | 7.52% | 7.51% | 7.48% |

The figures before costs are the scenarios' to the last digit. The
break-even costs:

| Mix | Daily's lead over monthly, before costs | Cost that ends it | Daily's lead over the 5-point band | Cost that ends it | 5-point band's lead over buy and hold | Cost that ends it |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Equity | 0.110 pt | 12.9 bp | 0.028 pt | 2.7 bp | 0.062 pt | above 100 bp |
| 60/40 | 0.221 pt | 26.0 bp | -0.038 pt | no lead | -0.781 pt | no lead |
| Equal weight | 0.500 pt | 35.2 bp | -0.191 pt | no lead | 0.730 pt | above 100 bp |
| Permanent | 0.274 pt | 23.6 bp | 0.027 pt | 2.0 bp | -0.601 pt | no lead |

What they show:

- **The drag is the turnover times twice the cost.** Each unit of one-way
  turnover is a sale and a purchase, so the costs paid a year are `2 k`
  times the turnover, and the fall in the compound return is that times one
  plus the return (1.07 to 1.09 times here). Rebalanced daily, a 10 bp cost
  takes 0.105 to 0.176 points a year off the return and 50 bp 0.52 to 0.88;
  monthly, 0.020 to 0.034 and 0.10 to 0.17; the 5-point band, 0.005 to 0.011
  and 0.023 to 0.056.
- **Daily rebalancing's lead over monthly survives costs below 12.9 bp
  (equity) to 35.2 bp (equal weight).** At 10 bp it is 0.025 to 0.357 points,
  against 0.110 to 0.500 before costs; at 25 bp only equal weight (0.144) and
  the 60/40 (0.008) still lead; at 50 bp monthly returns more for every mix,
  by 0.20 to 0.31 points. Equal weight rebalanced daily returns 8.22% before
  costs and 7.34% at 50 bp, below its 7.68% left alone.
- **The 5-point band is the best rebalancing rule from 5 bp up for every
  mix**, and at every cost for the 60/40 and equal weight. Daily rebalancing
  leads it for the equity and permanent mixes only below 2.7 and 2.0 bp. The
  band's lead over buy and hold for the equity and equal-weight mixes
  outlasts any one-way cost up to 1%. Buy and hold returns the most for the 60/40 and the permanent mix at
  any cost, by drifting into other portfolios (see the scenarios).

It is one history, and the costs are assumed levels: what the funds cost to
trade sets which row applies.

## Mean-variance optimisation and the efficient frontier

[`frontier.py`](src/portfolio_risk/frontier.py) finds long-only, fully
invested portfolios of the eight funds, each weight between 0 and an optional
cap, from expected yearly returns (the mean daily return times 252) and the
yearly covariance (the sample covariance times 252, or Ledoit-Wolf's): the
**minimum-variance** portfolio; the **efficient frontier**, the
lowest-volatility portfolio at each expected return from the minimum-variance
portfolio's up to the highest any portfolio has; and the **tangency**
portfolio, with the highest ratio of expected excess return to volatility.
Ratios are against a zero risk-free rate, as in the risk measures, or against
the Treasury bill (see [Against the Treasury bill](#against-the-treasury-bill)).
With the sample covariance and a constant rate a portfolio's ratio is exactly
the Sharpe ratio of its daily returns rebalanced every close: equal weight's is
0.642 here and in [`results/risk-metrics.json`](results/risk-metrics.json)
alike.

Each portfolio is the exact solution of its quadratic programme, by a primal
active-set method written in numpy. The tangency portfolio is found by
bisection on the tangency condition along the frontier, since near its peak
the ratio itself is too flat to search on. The tests check every portfolio
against brute force: the lowest variance over every pattern of funds at zero,
at the cap or free; for the tangency, the best ratio over every set of funds
held, and with a cap the homogenised problem over every pattern, to 1e-12 in
the weights. They also check closed forms where no bound binds, and expected
returns as little as 1e-11 apart.
On the real data scipy's SLSQP finds the same portfolios: variances within
4e-15 of these, relative, and tangency ratios equal to 14 significant digits,
these never the lower.

```bash
python -m portfolio_risk.frontier --json results/frontier.json
python -m portfolio_risk.frontier --max-weight 0.3 --json results/frontier-cap30.json
python -m portfolio_risk.frontier --risk-free bills --json results/frontier-bills.json
python -m portfolio_risk.frontier --risk-free bills --max-weight 0.3 --json results/frontier-cap30-bills.json
```

`--max-weight` caps every fund, `--covariance ledoit-wolf` shrinks the
covariance, `--risk-free` sets the rate (compound, a year, or `bills` for the
daily one-month bill from the factor cache) and `--points` the portfolios
along the frontier (21).

**Out of sample.** Portfolios fitted to the whole history and judged on it
look better than they are. So the command also chooses the minimum-variance
and tangency portfolios, and equal weights to compare, from the returns up to
each split date, holds them after it rebalanced daily, and reports what they
returned beside what the optimiser expected. `--split` may be given more than
once; by default the splits are the ends of 2009, 2012, 2015, 2018 and 2021
(`none` for none).

### Results

The whole history, 4 January 2005 to 30 September 2026
([`results/frontier.json`](results/frontier.json),
[`results/frontier-cap30.json`](results/frontier-cap30.json)), as the
optimiser sees it, in sample:

| Portfolio | Cap | Expected | Volatility | Sharpe | Weights |
| --- | --- | ---: | ---: | ---: | --- |
| Minimum variance | none | 3.73% | 5.01% | 0.744 | AGG 91.5%, SPY 6.8%, GLD 1.6% |
| Tangency | none | 6.74% | 6.82% | 0.988 | AGG 55.2%, SPY 22.8%, GLD 18.9%, TLT 3.0% |
| Minimum variance | 30% | 6.79% | 7.49% | 0.907 | AGG 30.0%, TLT 30.0%, SPY 26.0%, GLD 14.0% |
| Tangency | 30% | 7.99% | 8.18% | 0.976 | SPY 30.0%, AGG 30.0%, GLD 24.7%, TLT 15.3% |
| Equal weight | - | 8.87% | 13.83% | 0.642 | 12.5% each |

Uncapped, the frontier runs from 3.73% expected at 5.01% volatility to SPY
alone, 12.12% at 18.88%, and never holds IWM, EFA, EEM or VNQ.

Chosen up to each split and held after it, the Sharpe ratio the optimiser
expected, then the one realised:

| Chosen up to | Days held | Minimum variance | Tangency | Minimum variance, 30% cap | Tangency, 30% cap | Equal weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2009-12-31 | 4,211 | 0.94 → 0.68 | 1.25 → 0.64 | 0.96 → 0.88 | 1.22 → 0.59 | 0.52 → 0.73 |
| 2012-12-31 | 3,457 | 1.14 → 0.52 | 1.39 → 0.46 | 1.22 → 0.74 | 1.33 → 0.45 | 0.62 → 0.69 |
| 2015-12-31 | 2,701 | 0.99 → 0.53 | 1.10 → 0.67 | 1.05 → 0.79 | 1.05 → 0.78 | 0.57 → 0.75 |
| 2018-12-31 | 1,947 | 0.97 → 0.47 | 1.07 → 0.71 | 1.03 → 0.77 | 1.03 → 0.77 | 0.58 → 0.77 |
| 2021-12-31 | 1,190 | 0.99 → 0.12 | 1.16 → 0.36 | 1.14 → 0.34 | 1.15 → 0.35 | 0.67 → 0.53 |

and the compound return a year realised after the split:

| Chosen up to | Minimum variance | Tangency | Minimum variance, 30% cap | Tangency, 30% cap | Equal weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2009-12-31 | 3.47% | 4.25% | 6.48% | 4.84% | 8.11% |
| 2012-12-31 | 2.50% | 2.86% | 5.51% | 3.55% | 7.31% |
| 2015-12-31 | 2.74% | 4.24% | 6.25% | 6.15% | 8.50% |
| 2018-12-31 | 2.69% | 5.01% | 6.71% | 6.71% | 9.56% |
| 2021-12-31 | 0.55% | 2.61% | 2.77% | 2.88% | 5.92% |

What they show:

- **Every optimised portfolio fell short of the ratio it expected, after every
  split** (20 of 20). Most of the shortfall is in the mean: the uncapped
  tangency portfolio earned 0.34 to 0.94 of the return it expected, while its
  volatility came out 0.88 to 1.42 times the forecast. A few years of history
  estimate mean returns least precisely of all the inputs, and the optimiser
  leans hardest on them.
- **Equal weight realised a higher ratio than the uncapped minimum-variance
  and tangency portfolios after every split** (0.53 to 0.77, against 0.12 to
  0.68 and 0.36 to 0.71), and the highest compound return of all five
  portfolios after every split. It holds the most equity: 62.5% in the four
  equity funds and REITs, against at most 28.4% for the others.
- **A cap helps the minimum-variance portfolio, not the tangency.** Capped at
  30%, minimum variance realised more than equal weight after 2009, 2012 and
  2015 (0.88, 0.74 and 0.79 against 0.73, 0.69 and 0.75), about the same after
  2018 (0.770 against 0.772) and less after 2021 (0.34 against 0.53), when
  bonds fell with equities. The capped tangency beat equal weight once, after
  2015 (0.781 against 0.752).
- **The weights move with the window.** The tangency portfolio chosen up to
  2009 held 54.8% AGG, 21.9% GLD, 14.8% TLT, 7.4% EEM and 1.2% VNQ; the one
  the years after 2015 would have chosen holds 49.6% SPY and 50.4% GLD. The
  minimum-variance portfolio stays mostly AGG (80.6% to 91.1% chosen before a
  split, 93.4% to 95.0% after).

The held periods overlap (every one includes 2022 to 2026), so the five splits
are five views of one history, not five independent trials. The ratios above
take a zero risk-free rate, which flatters every portfolio and the least
volatile most; the next section measures them against the bill. Nothing here
charges trading costs.

### Against the Treasury bill

With `--risk-free bills` the realised ratios subtract the one-month bill day
by day, as the risk measures do, over the days the factor cache covers: 4
January 2005 to 31 August 2026 (5,448 days), a month shorter than above. The
tangency portfolio and the expected ratios need one constant rate, so they
take the bill's mean over the returns the portfolios are chosen from: 1.74% a
year over the whole history; 2.82%, 1.78%, 1.29%, 1.21% and 1.15% up to the
five splits; 1.42% to 3.93% after them. On the whole history that constant
rate gives the minimum-variance, tangency and equal-weight portfolios expected
ratios within 0.00013 of their Sharpe ratios against the bill day by day
(0.42767 against 0.42766, 0.80407 against 0.80419, 0.53165 against 0.53167).

The whole history
([`results/frontier-bills.json`](results/frontier-bills.json),
[`results/frontier-cap30-bills.json`](results/frontier-cap30-bills.json)), in
sample:

| Portfolio | Cap | Expected | Volatility | Sharpe | Sharpe against zero (above) | Weights |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Minimum variance | none | 3.87% | 5.01% | 0.428 | 0.744 | AGG 91.4%, SPY 6.9%, GLD 1.7% |
| Tangency | none | 10.25% | 10.60% | 0.804 | 0.988 | SPY 41.6%, GLD 35.2%, TLT 23.2% |
| Minimum variance | 30% | 6.98% | 7.48% | 0.702 | 0.907 | TLT 30.0%, AGG 30.0%, SPY 25.9%, GLD 14.1% |
| Tangency | 30% | 8.63% | 8.70% | 0.794 | 0.976 | SPY 30.0%, GLD 30.0%, AGG 27.1%, TLT 12.9% |
| Equal weight | - | 9.08% | 13.83% | 0.532 | 0.642 | 12.5% each |

The minimum-variance portfolio does not depend on the rate; its ratio falls
the most, by 42.5%, against 17.1% for equal weight. The tangency portfolio
chosen at the bill holds no AGG (55.2% at a zero rate).

Chosen up to each split and held after it, the Sharpe ratio expected, then
realised, against the bill:

| Chosen up to | Minimum variance | Tangency | Minimum variance, 30% cap | Tangency, 30% cap | Equal weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2009-12-31 | 0.48 → 0.45 | 0.94 → 0.50 | 0.60 → 0.73 | 0.93 → 0.48 | 0.38 → 0.63 |
| 2012-12-31 | 0.80 → 0.22 | 1.13 → 0.27 | 0.97 → 0.55 | 1.12 → 0.28 | 0.52 → 0.56 |
| 2015-12-31 | 0.72 → 0.18 | 0.87 → 0.44 | 0.86 → 0.57 | 0.86 → 0.54 | 0.49 → 0.60 |
| 2018-12-31 | 0.70 → 0.09 | 0.85 → 0.50 | 0.85 → 0.53 | 0.85 → 0.54 | 0.49 → 0.61 |
| 2021-12-31 | 0.74 → -0.40 | 0.98 → 0.05 | 0.97 → 0.01 | 0.98 → 0.05 | 0.59 → 0.29 |

- **19 of 20 optimised portfolios fell short of the ratio they expected.** The
  exception, the capped minimum-variance portfolio after 2009 (0.60 → 0.73),
  was chosen while the bill averaged 2.82% and held while it averaged 1.42%.
- **Equal weight realised a higher ratio than the uncapped minimum-variance
  and tangency portfolios after every split** (0.29 to 0.63, against -0.40 to
  0.45 and 0.05 to 0.50), as against a zero rate.
- **The cap's advantage mostly goes.** The capped minimum-variance portfolio
  beat equal weight only after 2009 (0.73 against 0.63; against a zero rate
  after 2009, 2012 and 2015), and the capped tangency portfolio after none.
- **The uncapped tangency portfolios hold less AGG**: 28.4% to 47.0% chosen
  before a split, against 50.5% to 61.8% at a zero rate. Capped, they hold
  AGG at or near the cap either way (30% after four splits, 28.4% after 2021
  against 30%).

## Factor attribution

[`factors.py`](src/portfolio_risk/factors.py) regresses each fund's daily
return in excess of the bill rate on the day's Fama-French factors, by
ordinary least squares:

```
r - rf = alpha + b1 (Mkt-RF) + b2 SMB + b3 HML [+ b4 RMW + b5 CMA] + e
```

the three-factor model (market, size, value; Fama and French 1993) and the
five-factor one (adding profitability and investment; 2015). Daily returns are
autocorrelated and their variance changes, so beside the OLS standard errors
it gives Newey-West ones (Bartlett weights, 9 lags for 5,448 days by Newey and
West's 1994 rule, no small-sample correction). The t-statistics below are
Newey-West's. Because OLS with an intercept passes through the means, the
mean excess return splits exactly into alpha plus each beta times its
factor's mean, all times 252.

```bash
python -m portfolio_risk.factors --json results/factors.json
```

The tests check the fit against textbook formulas and the Newey-West errors
against the formula written out as loops over the observations, at 0, 1, 4 and 9
lags. On the real data, statsmodels 0.14.6 (run outside the project) gives
the same coefficients to 1e-13, OLS errors to 3e-15 and Newey-West errors to
2.6e-13 (relative), for every fund and both models.

### Results

4 January 2005 to 31 August 2026 (5,448 days)
([`results/factors.json`](results/factors.json)). Alpha is a year's (daily
times 252); Newey-West t-statistics in brackets.

| Three factors | Alpha | Mkt-RF | SMB | HML | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPY | -0.17% (-0.7) | 0.99 (111.0) | -0.13 (-13.5) | -0.00 (-0.2) | 0.986 |
| IWM | -1.16% (-2.0) | 1.04 (143.9) | 0.86 (61.8) | 0.24 (28.1) | 0.976 |
| EFA | -3.60% (-1.8) | 0.95 (37.2) | -0.08 (-2.5) | 0.13 (5.9) | 0.775 |
| EEM | -3.46% (-1.1) | 1.18 (23.6) | -0.07 (-1.4) | 0.06 (1.7) | 0.670 |
| AGG | 1.34% (1.3) | 0.00 (0.2) | -0.00 (-0.0) | -0.05 (-4.4) | 0.016 |
| TLT | 4.59% (1.6) | -0.22 (-11.1) | 0.06 (1.6) | -0.20 (-8.1) | 0.124 |
| GLD | 9.61% (2.5) | 0.07 (2.4) | 0.04 (1.1) | -0.12 (-4.1) | 0.012 |
| VNQ | -1.95% (-0.6) | 1.04 (25.5) | 0.18 (3.5) | 0.51 (10.2) | 0.611 |
| Equal weight | 0.65% (0.5) | 0.63 (41.8) | 0.11 (6.1) | 0.07 (5.9) | 0.823 |

| Five factors | Alpha | Mkt-RF | SMB | HML | RMW | CMA | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY | -0.39% (-1.6) | 1.00 (103.6) | -0.12 (-14.0) | 0.01 (1.0) | 0.06 (6.1) | 0.04 (3.0) | 0.986 |
| IWM | -1.01% (-2.0) | 1.02 (155.5) | 0.86 (61.1) | 0.07 (8.4) | -0.08 (-6.7) | -0.01 (-0.7) | 0.979 |
| EFA | -3.66% (-1.9) | 0.96 (33.0) | -0.07 (-2.4) | 0.12 (4.4) | -0.01 (-0.4) | 0.12 (2.8) | 0.776 |
| EEM | -3.11% (-1.0) | 1.17 (20.8) | -0.08 (-1.6) | 0.08 (1.6) | -0.09 (-1.4) | 0.01 (0.1) | 0.670 |
| AGG | 1.40% (1.4) | -0.00 (-0.1) | -0.00 (-0.0) | -0.05 (-3.9) | -0.01 (-0.7) | -0.01 (-0.5) | 0.017 |
| TLT | 4.60% (1.6) | -0.22 (-10.3) | 0.06 (1.6) | -0.21 (-7.5) | -0.01 (-0.2) | -0.00 (-0.1) | 0.124 |
| GLD | 9.72% (2.6) | 0.07 (2.6) | 0.04 (0.9) | -0.15 (-4.4) | -0.06 (-1.2) | 0.12 (2.2) | 0.014 |
| VNQ | -2.17% (-0.6) | 1.04 (27.2) | 0.23 (4.0) | 0.47 (7.6) | 0.07 (1.3) | -0.01 (-0.2) | 0.613 |
| Equal weight | 0.67% (0.5) | 0.63 (38.2) | 0.11 (6.7) | 0.04 (2.8) | -0.02 (-1.0) | 0.03 (1.3) | 0.824 |

Where each fund's mean excess return came from, three factors (a year; the
parts add up to the mean):

| Series | Mean excess return | Alpha | Market | Size | Value |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPY | 10.46% | -0.17% | 10.57% | 0.06% | 0.00% |
| IWM | 9.55% | -1.16% | 11.10% | -0.37% | -0.03% |
| EFA | 6.62% | -3.60% | 10.20% | 0.03% | -0.01% |
| EEM | 9.16% | -3.46% | 12.60% | 0.03% | -0.01% |
| AGG | 1.37% | 1.34% | 0.02% | 0.00% | 0.01% |
| TLT | 2.25% | 4.59% | -2.33% | -0.02% | 0.02% |
| GLD | 10.36% | 9.61% | 0.75% | -0.02% | 0.01% |
| VNQ | 9.08% | -1.95% | 11.16% | -0.08% | -0.05% |
| Equal weight | 7.36% | 0.65% | 6.76% | -0.05% | -0.01% |

What they show:

- **The equity funds are the market, and the market paid.** The market
  factor's excess return averaged 10.71% a year over the window, so a market
  beta near 1 accounts for 10.2% to 12.6% a year of SPY's, IWM's, EFA's,
  EEM's and VNQ's mean excess returns. Size and value paid almost nothing
  (`SMB` -0.43% and `HML` -0.11% a year), so IWM's large size loading (0.86)
  and VNQ's value loading (0.51) explain how they moved, not what they
  earned.
- **SPY is what an index fund should be:** a market beta of 0.99, R² 0.986,
  and an alpha of -0.17% a year that is not distinguishable from zero (t
  -0.7); the fund charges about 0.09% a year. Its slight negative size loading
  (-0.13) is the S&P 500 holding only large companies, while `Mkt-RF` is
  every US stock.
- **Small caps lagged their factor exposures.** IWM's alpha is -1.16% a year
  with three factors and -1.01% with five (t -2.0 both): the Russell 2000
  earned less than its market, size and value loadings imply.
- **For the funds outside US equities the factors explain little.** AGG's R²
  is 0.016 and gold's 0.012, so their alphas (1.34% and 9.61% a year) are
  just their mean returns over the bill, unexplained by US stock factors, not
  skill; the factors have no bond or commodity return in them. TLT's market
  beta of -0.22 is the bond-stock hedge the correlations showed.
- **EFA and EEM** have market betas of 0.95 and 1.18 but R² of 0.775 and
  0.670: US factors miss their currencies and markets. Their alphas of -3.6%
  and -3.5% a year (t -1.8 and -1.1) are what non-US stocks earned below US
  ones with the same beta over these two decades.
- **Five factors change little here.** R² rises by at most 0.0034 (IWM), and
  profitability and investment, `RMW` and `CMA`, add at most 0.24 points a
  year to any fund's mean (EEM's `RMW` part, -0.24%). SPY's alpha moves from
  -0.17% to -0.39% because it loads on profitability (0.06), which paid 2.67%
  a year.
- **The equal-weight portfolio's alpha is 0.65% a year (t 0.5)**, the
  average of the eight funds' alphas, as a portfolio rebalanced daily to fixed
  weights must have: gold's, TLT's and AGG's, less the equity funds'. A market
  beta of 0.63 accounts for 6.76% of its 7.36% mean excess return.

The Newey-West t-statistics of the alphas are close to, or larger than, the
OLS ones (SPY -0.69 against -0.34): the residuals are negatively
autocorrelated (SPY's lag-one correlation is -0.35), which narrows rather than
widens the error of a mean. Daily regressions over one 21-year window describe
these funds over that window; with no alpha's t-statistic above 2.6, none here
is evidence that it will persist.

## Risk parity and minimum CVaR

[`construction.py`](src/portfolio_risk/construction.py) builds two kinds of
portfolio that, unlike the tangency portfolio, need no expected returns:

- **Risk parity** (equal risk contribution): each fund contributes the same
  share of the portfolio's variance, `w_i (S w)_i / w' S w = 1/n`. The weights
  come from Spinu's (2013) convex form, minimised by Newton's method; the
  answer is unique. **Inverse volatility**, weights proportional to one over
  each fund's volatility, is the same thing for uncorrelated funds and is
  shown beside it.
- **Minimum CVaR**: the portfolio whose historical expected shortfall of one
  day's return at 95%, the mean of the worst 5% of days, is lowest: Rockafellar
  and Uryasev's (2000) linear programme over 5,469 days, solved by HiGHS through
  scipy (the project's one optimiser that is not its own code).

The tests check that risk parity's contributions are equal to 1e-12 for up to
50 funds and over 900 random matrices of three scales, that they follow
unequal risk budgets, and that risk parity reduces to inverse volatility when
correlations are equal; and that the CVaR programme's optimum is the expected
shortfall `risk.expected_shortfall` gives its weights, that no portfolio on a
1% grid of three funds has a lower one, and that its threshold is the value
at risk when the tail is not a whole number of days. On the real returns the
two agree to 4e-18.

```bash
python -m portfolio_risk.construction --json results/construction.json
python -m portfolio_risk.construction --risk-free bills --json results/construction-bills.json
```

The command compares six portfolios: equal weight, inverse volatility, risk
parity, and the frontier's minimum-variance and tangency portfolios beside
minimum CVaR, each chosen from the whole history and from the returns up to
each of the frontier's five split dates, then held after it, rebalanced every
close. The covariance is the sample one and Sharpe ratios are against a zero
rate, or with `--risk-free bills` against the bill, as in the frontier
section.

### Results

In sample, 4 January 2005 to 30 September 2026
([`results/construction.json`](results/construction.json)); CVaR is of one
day's return at 95%, and "top risk" is the largest share of the portfolio's
variance any one fund contributes:

| Portfolio | Volatility | Sharpe | Compound return | Max drawdown | CVaR | Top risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Equal weight | 13.83% | 0.64 | 8.22% | -38.3% | 2.05% | 22.0% |
| Inverse volatility | 9.37% | 0.75 | 6.79% | -25.6% | 1.37% | 17.0% |
| Risk parity | 8.21% | 0.80 | 6.41% | -23.2% | 1.19% | 12.5% |
| Minimum variance | 5.01% | 0.74 | 3.66% | -17.4% | 0.71% | 91.5% |
| Tangency | 6.82% | 0.99 | 6.71% | -17.2% | 0.98% | 41.1% |
| Minimum CVaR | 5.02% | 0.75 | 3.72% | -17.4% | 0.71% | 90.9% |

Risk parity holds 35.8% AGG, 20.6% TLT, 11.9% GLD and 5.3% to 7.8% of each
equity fund; minimum CVaR holds 90.9% AGG, 6.6% SPY and 2.4% GLD, close to
minimum variance's 91.5%, 6.8% and 1.6%.

Held after each split, chosen from the returns before it:

| Sharpe ratio held after | Equal weight | Inverse volatility | Risk parity | Minimum variance | Tangency | Minimum CVaR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2009 | 0.73 | 0.82 | 0.75 | 0.68 | 0.64 | 0.79 |
| 2012 | 0.69 | 0.70 | 0.59 | 0.52 | 0.46 | 0.63 |
| 2015 | 0.75 | 0.76 | 0.66 | 0.53 | 0.67 | 0.60 |
| 2018 | 0.77 | 0.75 | 0.65 | 0.47 | 0.71 | 0.52 |
| 2021 | 0.53 | 0.43 | 0.27 | 0.12 | 0.36 | 0.10 |

| CVaR held after | Equal weight | Inverse volatility | Risk parity | Minimum variance | Tangency | Minimum CVaR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2009 | 1.69% | 1.15% | 1.08% | 0.77% | 1.01% | 0.72% |
| 2012 | 1.64% | 1.17% | 1.11% | 0.74% | 0.98% | 0.72% |
| 2015 | 1.73% | 1.24% | 1.15% | 0.79% | 0.96% | 0.77% |
| 2018 | 1.89% | 1.37% | 1.26% | 0.89% | 1.04% | 0.86% |
| 2021 | 1.71% | 1.32% | 1.28% | 0.87% | 1.13% | 0.86% |

| Compound return held after | Equal weight | Inverse volatility | Risk parity | Minimum variance | Tangency | Minimum CVaR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2009 | 8.11% | 6.27% | 5.36% | 3.47% | 4.25% | 3.81% |
| 2012 | 7.31% | 5.34% | 4.18% | 2.50% | 2.86% | 3.06% |
| 2015 | 8.50% | 6.18% | 4.99% | 2.74% | 4.24% | 3.05% |
| 2018 | 9.56% | 6.78% | 5.40% | 2.69% | 5.01% | 2.94% |
| 2021 | 5.92% | 3.69% | 2.14% | 0.55% | 2.61% | 0.43% |

What they show:

- **Equal weight is risk-concentrated.** In sample, the five equity and REIT
  funds contribute 94.7% of its variance (EEM 22.0%, VNQ 21.2%), the bonds
  next to nothing (TLT -0.9%, a hedge). Risk parity spreads it evenly and,
  over the whole history, has a higher Sharpe ratio (0.80 against 0.64) with
  a drawdown of -23.2% against -38.3%.
- **Out of sample, the simpler rule did better.** Inverse volatility had a
  higher held Sharpe ratio than risk parity after every split, and the best
  of all six after 2009, 2012 and 2015; equal weight was best after 2018 and
  2021. Risk parity beat equal weight only after 2009 (0.75 against 0.73). It
  holds more TLT (22.1% to 28.5% when chosen) than inverse volatility (13.1%
  to 17.6%), and TLT's share of its risk, 12.5% by construction in the data
  it was chosen from, was 25.6% to 38.3% after each split. TLT's correlation
  with SPY was higher after every split than before it (-0.35 to -0.45
  before, -0.28 to 0.11 after) and its volatility higher after four of the
  five, so the bonds hedged less than the weights assumed.
- **Minimum CVaR did what it aims at, by a little.** Held after every split
  its daily CVaR was lower than minimum variance's (0.72% against 0.77% after
  2009, 0.86% against 0.87% after 2021), as was its volatility; its Sharpe
  ratio was higher after four of the five (not after 2021, 0.10 against
  0.12), and its drawdown shallower after four (after 2021, -16.85% against
  -16.79%). The two portfolios' weights differ by at most 2.4 to 7.9 points
  in any fund, so the gaps are small.
- **No construction beat equal weight's compound return** after any split,
  as the frontier section found for its three: the low-risk portfolios held
  mostly AGG, which returned 2.32% a year after 2009 and -0.64% after 2021.

The held periods overlap, so the five splits are five views of one history,
and nothing here charges trading costs.

**Against the Treasury bill**, over the days the factor cache covers (to 31
August 2026, a month shorter than above), the tangency portfolio chosen at
the bill's mean rate as in the frontier section
([`results/construction-bills.json`](results/construction-bills.json)):

| Portfolio | Sharpe | Sharpe against zero (above) | Change |
| --- | ---: | ---: | ---: |
| Equal weight | 0.53 | 0.64 | -17.1% |
| Inverse volatility | 0.59 | 0.75 | -21.8% |
| Risk parity | 0.61 | 0.80 | -23.1% |
| Minimum variance | 0.43 | 0.74 | -42.5% |
| Tangency | 0.80 | 0.99 | -18.6% |
| Minimum CVaR | 0.44 | 0.75 | -41.6% |

| Sharpe ratio held after, against the bill | Equal weight | Inverse volatility | Risk parity | Minimum variance | Tangency | Minimum CVaR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2009 | 0.63 | 0.67 | 0.59 | 0.45 | 0.50 | 0.54 |
| 2012 | 0.56 | 0.52 | 0.40 | 0.22 | 0.27 | 0.33 |
| 2015 | 0.60 | 0.54 | 0.43 | 0.18 | 0.44 | 0.23 |
| 2018 | 0.61 | 0.52 | 0.41 | 0.09 | 0.50 | 0.13 |
| 2021 | 0.29 | 0.11 | -0.05 | -0.40 | 0.05 | -0.43 |

- **The AGG-heavy portfolios lose the most.** In sample, minimum variance's
  and minimum CVaR's ratios fall by 42.5% and 41.6%, the others' by 17.1% to
  23.1% (the tangency portfolio's is that of a different portfolio, chosen at
  the bill). Risk parity's 0.61 is still above equal weight's 0.53.
- **Held after a split, equal weight comes first more often.** It had the
  highest ratio of the six after 2012, 2015, 2018 and 2021, and inverse
  volatility after 2009 only (against a zero rate, inverse volatility after
  2009, 2012 and 2015). Inverse volatility still beat risk parity after every
  split, and risk parity beat equal weight after none.
- **Minimum CVaR against minimum variance is unchanged:** a higher held ratio
  after four of the five splits, not after 2021 (-0.43 against -0.40).
- After 2021, when the bill averaged 3.93% a year, risk parity, minimum
  variance and minimum CVaR had mean returns below the bill's.

## Parts

| Part | State |
| --- | --- |
| Data ingestion via yfinance, cached to CSV with a manifest | done |
| Returns and covariance engine, annualisation helpers | done |
| Monte Carlo: multivariate normal, Student-t, stationary block bootstrap | done |
| Risk metrics: Sharpe, Sortino, drawdown, VaR, CVaR, downside deviation | done |
| Scenarios: allocation mixes, rebalancing, stress periods | done |
| Efficient frontier and mean-variance optimiser, held out of sample | done |
| Fama-French factor attribution (Ken French data library), ratios against the bill rate | done |
| Risk parity and CVaR optimisation, compared out of sample | done |
| Cost-aware backtest: transaction costs, spreads, rebalancing drag | done |
| Tableau extracts, dashboard spec and Plotly reproduction | to come |

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -c constraints.txt -e ".[dev]"   # bin/ on Linux and macOS
pytest
ruff check . && ruff format --check .
```

The tests never reach Yahoo or the data library: they stand in for the
downloads.

## License

MIT; see [LICENSE](LICENSE). The license covers the code, not Yahoo's data
or the factor data, which is Eugene Fama and Kenneth French's.
