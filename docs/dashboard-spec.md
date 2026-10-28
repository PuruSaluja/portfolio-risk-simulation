# Dashboard specification

A Tableau dashboard over the CSV extracts that `python -m portfolio_risk.extracts`
writes, and the same dashboard drawn with Plotly by `python -m portfolio_risk.dashboard`.

**It has not been built in Tableau**, as none was available. This file is the
specification a Tableau workbook would follow; the Plotly page is its
reproduction, and the tests check that every value the page plots is the
extract's value, unchanged but for costs, drawn in basis points. The page
draws each sheet's main view; the two companion views (each frontier
point's weights, and the held-out portfolios after costs) are specified
here but not drawn.

## Data sources

One text-file data source per extract, each read by Tableau's text connector
with the first row as field names. Dates are ISO `YYYY-MM-DD`; an empty field
is a null (a figure that is undefined, or a fund's stress row, which has no
rule). Set the field types as below; the numbers `level`, `point` and
`max_weight` name rows, so they are dimensions, not measures.

| Source | One row per | Dimensions | Measures |
| --- | --- | --- | --- |
| `daily.csv` | trading day and series | `date` (date), `series` | `daily_return`, `wealth`, `drawdown` |
| `risk_metrics.csv` | series and rate | `series`, `risk_free` (zero, bills), `first_return`, `last_return`, the three drawdown dates | `compound_return`, `volatility`, `sharpe_ratio`, `sharpe_ratio_monthly`, `downside_deviation`, `sortino_ratio`, `max_drawdown` |
| `tail_risk.csv` | series, level and method | `series`, `level` (0.95, 0.99), `method` (historical, normal) | `value_at_risk`, `expected_shortfall` |
| `montecarlo.csv` | source and scope | `source` (historical or a model), `scope` (daily, horizon) | `count`, `mean`, `sd`, `skewness`, `excess_kurtosis`, `probability_of_loss`, `q0_1` to `q99` |
| `stress.csv` | period and holding (and rule, for a mix) | `period`, `start`, `end`, `holding`, `kind` (fund, mix), `rule` | `days`, `return`, `max_drawdown` |
| `rebalancing.csv` | mix and rule | `mix`, `rule` | `compound_return`, `volatility`, `sharpe_ratio`, `max_drawdown`, `rebalances`, `turnover_per_year` |
| `trading_costs.csv` | mix, rule and cost | `mix`, `rule`, `cost` | `net_compound_return`, `drag`, `paid_per_year` |
| `frontier.csv` | cap and point | `max_weight` (1.0, 0.3), `point` | `expected_return`, `volatility`, `sharpe_ratio` |
| `frontier_weights.csv` | cap, point and fund | `max_weight`, `point`, `fund` | `weight` |
| `portfolios.csv` | cap and name | `max_weight`, `name`, `kind` (fund, portfolio) | `expected_return`, `volatility`, `sharpe_ratio` |
| `held_out.csv` | command, cap, rate, split and portfolio | `command` (frontier, construction), `max_weight`, `risk_free`, `split` (date), `portfolio` | `chosen_sharpe_ratio`, `held_sharpe_ratio`, `held_volatility`, `held_compound_return` |
| `held_costs.csv` | the same, and rule and cost | as `held_out.csv`, and `rule`, `cost` | `compound_return`, `sharpe_ratio` |
| `factors.csv` | series, model and term | `series`, `model` (ff3, ff5), `term` (alpha and each factor) | `estimate`, `ols_t`, `newey_west_t`, `r_squared` |

No source is joined to another; each sheet uses one. The rows are figures
already computed (ratios, compound returns, quantiles), so **no measure may be
summed across rows**: every sheet shows one row per mark, aggregated with
`ATTR()` or `MIN()` over a key that picks one row, never `SUM()`.

`daily.csv` is built from Yahoo's prices and is not committed; the extracts
command writes it from a local price cache, and the Plotly page leaves the
charts over time out when it is missing. The page refuses a `daily.csv` that
does not cover the risk measures' series and days.

`manifest.json`, which the extracts command writes last, lists each file it
wrote with its SHA-256. A workbook refreshed from the folder should check
it, as the Plotly page does: a file the manifest does not list, or lists with
another checksum, came from another run and is not shown.

## Calculated fields

| Field | Source | Formula |
| --- | --- | --- |
| `Cost (bp)` | trading_costs, held_costs | `[cost] * 10000` |
| `Shortfall from chosen` | held_out | `[held_sharpe_ratio] - [chosen_sharpe_ratio]` |
| `Is portfolio` | daily, risk_metrics, tail_risk | `[series] = "portfolio"` |
| `Rate label` | risk_metrics, held_out | `IF [risk_free] = "bills" THEN "against the one-month bill" ELSE "zero rate" END` |

## Parameters

- **Rate** (zero, bills): filters `risk_metrics`, `held_out` and `held_costs`.
- **Rule** (daily, monthly, quarterly, yearly, never, band:0.05): the rule the
  stress sheet shows the mixes under; default monthly, as the README's table.
- **Cost (bp)** (0, 1, 5, 10, 25, 50): the cost the held-costs sheet shows.

## Sheets

1. **Headline tiles.** The equal-weight portfolio (`series = portfolio`):
   compound return a year, volatility, Sharpe ratio at a zero rate and against
   the bill, largest drawdown (`risk_metrics`), and the one-day 99% historical
   expected shortfall (`tail_risk`).
2. **Over time.** `daily`: `wealth` against `date` on a log axis, one line per
   series, the portfolio drawn thicker; below it on a shared date axis,
   `drawdown`. Highlight action on `series`.
3. **Efficient frontier.** Dual axis on `volatility`: `frontier` as lines,
   one per `max_weight` (solid uncapped, dashed at 30%), and `portfolios`
   (`max_weight` 1.0) as labelled marks, circles for funds, diamonds for the
   named portfolios. Expected returns are sample means: the sheet is titled
   "in sample". A second view on `frontier_weights` shows each point's weights
   as a stacked area against `point`.
4. **Sharpe ratio at a zero rate and against the bill.** `risk_metrics`:
   bars per series, side by side by `risk_free`. The bill's rows end a month
   earlier (the factor library's last month); the tooltip shows
   `first_return` and `last_return`.
5. **One-day loss at 99%.** `tail_risk` at `level` 0.99: value at risk and
   expected shortfall per series, historical beside normal.
6. **One-year return.** `montecarlo`, `scope = horizon`, for historical,
   normal, t, bootstrap and bootstrap:252: a box per source from the
   quantiles, 5% to 95% the box, 1% to 99% the whiskers, the median marked.
   Historical is every 252-day window of the data (overlapping); each model is
   10,000 simulated paths.
7. **Stress periods.** `stress`: a heatmap, periods down, holdings across
   (funds bought and held, then the mixes under the Rule parameter), coloured
   by `return` on a diverging scale centred on zero.
8. **Trading costs.** `trading_costs`: small multiples per mix,
   `net_compound_return` against `Cost (bp)`, one line per rule. The costs
   are assumed levels, the same for every fund, not measured spreads; the
   subtitle says so.
9. **Held out.** `held_out` for the uncapped frontier and the construction
   command at the Rate parameter: `held_sharpe_ratio` against `split`, one
   line per portfolio, one colour per portfolio across both panels. Tooltip:
   `chosen_sharpe_ratio` and `Shortfall from chosen`. A companion view on
   `held_costs` at the Cost parameter shows the same after costs, by rule.
10. **Factor loadings.** `factors`, `model = ff5`: a heatmap of `estimate`,
    series down, terms across, diverging around zero; alpha is a yearly rate,
    the rest are betas. Tooltip: both t statistics and `r_squared`.

## Layout

One vertical dashboard, 1,200 px wide: the tiles across the top, then the
sheets in the order above, each full width (the over-time sheet twice the
height of the others). A text box under the title says the prices are real,
which window they cover, and that the frontier is in sample while the
held-out sheet is not.

## Labelling

- In sample and held out are never on one axis without saying which is
  which: the frontier and its portfolios are in sample; the held-out sheet's
  ratios come after the day each portfolio was chosen.
- The cost sheets name their costs as assumed levels. The measured floor (the
  one-cent tick) and the spread estimators' negative result are in the README,
  not on the dashboard.
- Ratios against a zero rate and against the bill are labelled on every mark
  that shows one.
