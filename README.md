# Portfolio Simulation and Risk Analysis

Monte Carlo simulation and risk analysis of multi-asset portfolios, on real
daily market data.

> **Status:** complete. Every figure this README gives
> comes from a run of the committed code on the real prices described below,
> and each results file is given with the command that writes it.

## What it found

Eight funds' daily total returns, January 2005 to September 2026; the
portfolio holds them in equal parts, rebalanced at every close. All of these
are measured on that history, not projected:

- **The equal-weight portfolio** returned 8.22% a year at 13.83% volatility,
  a Sharpe ratio of 0.64 at a zero rate and 0.53 against the one-month
  Treasury bill (over the days the bill covers, to August 2026); its deepest
  fall was -38.3%, against SPY's -55.2% ([risk measures](docs/guide.md#risk-measures)).
- **Normality understates the tails and overstates the year.** History's
  worst day in a thousand is a 5.07% loss, the normal model's 2.61% (the
  Student-t 4.16%, the bootstrap 5.10%); the portfolio's one-day 99% expected
  shortfall is 3.65%, the normal's 2.29%. Yet drawing days independently puts
  the one-year standard deviation at 15.1% against history's 11.9%, because
  daily moves partly reverse; a block bootstrap of 252-day blocks gives 12.1%
  ([Monte Carlo](docs/guide.md#monte-carlo-simulation)).
- **Optimised portfolios disappointed out of sample.** Chosen on the years
  before each of five splits and held after it, every mean-variance portfolio
  fell short of the Sharpe ratio it expected (20 of 20), and equal weight beat
  the uncapped minimum-variance and tangency portfolios after every split
  ([frontier](docs/guide.md#mean-variance-optimisation-and-the-efficient-frontier)).
  Inverse volatility beat risk parity after every split
  ([risk parity and minimum CVaR](docs/guide.md#risk-parity-and-minimum-cvar)).
- **Costs decide the rebalancing rule.** Daily rebalancing's lead over
  monthly is gone at a one-way cost of 12.9 bp (equity) to 35.2 bp (equal
  weight), and from 5 bp a 5-point band is the best of the rules that
  rebalance for every mix; for the 60/40 and the permanent mix, never
  rebalancing returns more still, by drifting into other portfolios
  ([trading costs](docs/guide.md#trading-costs)).
- **Factors explain the equity funds, not the rest.** SPY has a market beta
  of 0.99 and an R² of 0.986; the equal-weight portfolio's alpha is 0.65% a
  year with a t statistic of 0.5, no evidence of any
  ([factor attribution](docs/guide.md#factor-attribution)).
- **Daily bars cannot measure these funds' spreads.** Two published
  high-low estimators read them as 9.8 to 57 (Abdi-Ranaldo) and 8.6 to 60
  (Corwin-Schultz) times a one-cent spread, and simulation shows why
  ([a negative result](docs/guide.md#estimating-spreads-from-highs-and-lows-a-negative-result)).

The [guide](docs/guide.md) sets out how each figure is computed: the data
and its manifest first, then returns, simulation, risk measures, scenarios,
costs, optimisation, factors and construction, each with its method, the
command that writes its results file, and the results.
[The dashboard](docs/guide.md#tableau-extracts-and-dashboard) draws them together.

## The data is real

Daily closes and dividends of eight exchange-traded funds from Yahoo
Finance, fetched with [yfinance](https://github.com/ranaroussi/yfinance):
SPY, IWM, EFA, EEM (equities), AGG and TLT (bonds), GLD (gold) and VNQ (real
estate), every one trading throughout January 2005 to September 2026, so the
history covers 2008, 2020 and 2022. The Fama-French factors and the one-month
Treasury bill rate come from the Ken French data library.

**The prices are not committed.** yfinance's documentation notes that Yahoo's
data is meant for personal use, so a fetch writes them to a local cache and
the repository holds only its manifest: each fund's rows, dates and dividend
count, and a SHA-256 of the file. A run refuses a cache that does not match
it, so every result is computed from the data it says it was.

## What is in it

| Part | Guide |
| --- | --- |
| Returns, covariance and their annualisation | [returns and risk of each fund](docs/guide.md#returns-and-risk-of-each-fund) |
| Monte Carlo: multivariate normal, Student-t, stationary block bootstrap | [simulation](docs/guide.md#monte-carlo-simulation) |
| Sharpe, Sortino, drawdown, VaR, CVaR | [risk measures](docs/guide.md#risk-measures) |
| Allocation mixes, rebalancing rules, stress periods | [scenarios](docs/guide.md#scenarios-allocation-mixes-rebalancing-and-stress-periods) |
| Transaction costs, spreads and rebalancing drag | [trading costs](docs/guide.md#trading-costs) |
| Efficient frontier, held out of sample, before and after costs | [optimisation](docs/guide.md#mean-variance-optimisation-and-the-efficient-frontier) |
| Fama-French factor attribution | [factors](docs/guide.md#factor-attribution) |
| Risk parity and minimum CVaR | [construction](docs/guide.md#risk-parity-and-minimum-cvar) |
| Tableau extracts, a dashboard spec and a Plotly dashboard | [dashboard](docs/guide.md#tableau-extracts-and-dashboard) |

Each results file in [`results/`](results) is given in the guide with the
command that writes it.

## Run it

Needs Python 3.11 or later.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -c constraints.txt -e ".[dev]"   # bin/ on Linux and macOS
python -m portfolio_risk.data fetch       # the prices, about 2 s
python -m portfolio_risk.french fetch     # the factors and the bill rate
python -m portfolio_risk.risk --weights equal --risk-free 0
python -m portfolio_risk.extracts --out extracts
python -m portfolio_risk.dashboard --extracts extracts --out extracts/dashboard.html
```

A fetch says whether the new file matches the committed manifest; Yahoo
revises history now and then, and when it has, the check says so rather than
quietly computing different figures. `pytest` runs the suite, which never
reaches Yahoo or the data library; [Development](docs/guide.md#development)
says what its whole-project checks cover.

## More

- [The full guide](docs/guide.md): every part's method, command and results,
  and what each cannot show.
- [Dashboard spec](docs/dashboard-spec.md) for the Tableau version.

## License

MIT — see [LICENSE](LICENSE).
