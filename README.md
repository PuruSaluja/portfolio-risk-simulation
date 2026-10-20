# Portfolio Simulation and Risk Analysis

Monte Carlo simulation and risk analysis of multi-asset portfolios, on real
daily market data.

> **Status:** early. The data ingestion is in place; the parts below are
> built in order, and every figure this README gives comes from a run of the
> committed code.

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

## Parts

| Part | State |
| --- | --- |
| Data ingestion via yfinance, cached to CSV with a manifest | in progress |
| Returns and covariance engine, annualisation helpers | to come |
| Monte Carlo: multivariate normal, Student-t, stationary block bootstrap | to come |
| Risk metrics: Sharpe, Sortino, drawdown, VaR, CVaR, downside deviation | to come |
| Scenarios: allocation mixes, rebalancing, stress periods | to come |
| Efficient frontier, mean-variance, risk parity and CVaR optimisation | to come |
| Fama-French factor attribution (Ken French data library) | to come |
| Cost-aware backtest: transaction costs, spreads, rebalancing drag | to come |
| Tableau extracts, dashboard spec and Plotly reproduction | to come |

## License

MIT; see [LICENSE](LICENSE). The license covers the code, not Yahoo's data.
