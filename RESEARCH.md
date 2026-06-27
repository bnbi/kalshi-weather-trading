# Live Trading Postmortem: 89 Trades in Kalshi Weather Markets

**Period:** May 6 – June 11, 2026 · **Result:** −$22.86 on ~$120 staked (−19% ROI)
**Outcome:** full diagnosis of the loss, three structural fixes, and a redesigned
strategy targeting a different edge source. This document is the quantitative
record of that process.

---

## 1. Setup

The system traded daily-high-temperature contracts (thresholds and 1°F brackets)
for Chicago, NYC, and Miami. Pipeline: multi-model ensemble (GFS, ECMWF,
Open-Meteo blend) → per-city ML correction model (Lasso/Ridge selected by
time-series CV) → Normal probability model → edge vs market ask → fractional
Kelly sizing → limit-order execution.

Every prediction and trade was logged to SQLite, which is what made this
analysis possible. **If you take one thing from this document: log everything.**

## 2. What the data showed

### 2.1 The point forecast was good

Live verified MAE over ~30 days per city, against each input source:

| City | Model MAE | GFS | ECMWF | Blend |
|------|-----------|------|-------|-------|
| Chicago | **1.54°F** | 3.00 | 1.71 | 3.30 |
| NYC | **1.36°F** | 2.07 | 1.57 | 2.11 |
| Miami | **1.40°F** | 1.33 | 2.20 | 1.28 |

The ML correction beat every individual source in 2 of 3 cities and the naive
ensemble average everywhere. Temperature prediction was not the problem.

### 2.2 The probabilities were systematically overconfident

Realized win rate by claimed probability bucket (probability of the side traded):

| Claimed | Realized | n |
|---------|----------|----|
| <20% | 0% | 10 |
| 20–40% | 15% | 13 |
| 40–60% | 0% | 4 |
| 60–80% | 33% | 6 |
| 80%+ | 73% | 56 |

Every bucket realized below its claim. Note this is *not* primarily a σ
miscalibration — the temperature residuals were nearly Gaussian with honest
variance. It is **selection bias**: trades only occur where the model disagrees
with the market, so conditioning on "we traded" selects exactly the cases where
the model is most likely wrong — the winner's curse / adverse selection.

### 2.3 The market was the better forecaster — on traded contracts

Fitting `p = w·model + (1−w)·price` by log-likelihood over all 89 settled
trades, the optimum was **w ≈ 0**: market price alone (log loss 0.386) beat
every blend including any model weight, and the raw model scored 0.770. Brier
skill score of the model vs the market baseline: **−0.76**.

This is the expected result for next-day weather: market makers see the same
public model runs. There was no information asymmetry to monetize.

### 2.4 The loss was concentrated in the largest disagreements

P&L partitioned by raw model−market disagreement:

| Rule | Trades kept | P&L |
|------|-------------|-----|
| keep all | 89 | **−$22.86** |
| skip if gap > 25¢ | 56 | **+$8.68** (skipped 33 lost −$31.54) |

The legacy code *capped* edges at 25¢ but still traded them. Larger disagreement
predicted larger losses — the market knew something (newer model runs, intraday
observations). 9 of the 10 worst trades had claimed edges above 27¢.

## 3. Fixes (deployed June 11, 2026)

1. **Credibility filter.** Disagreements >25¢ are skipped, not capped. The
   market's information advantage at extreme disagreement dominates any model
   signal.
2. **Bayesian shrinkage toward the market prior.** Decisions and Kelly sizing
   use `p̂ = 0.5·model + 0.5·price`, so a 7¢ minimum edge requires a 14¢ raw
   gap, and stake sizes reflect humbled probabilities. Combined with (1),
   the rule set retrospectively keeps 24–47 of the 89 trades at +19–28% ROI
   across w ∈ [0.4, 0.6] — robust to parameter choice, though in-sample
   (see §5).
3. **Live recalibration.** σ and mean bias for the probability model now come
   from the last 30 *live verified* prediction errors per city rather than
   in-sample CV residuals (live next-day σ ≈ 1.8–2.1°F; the old hard-coded
   floor of 2.5°F distorted bracket probabilities). Bias corrections are
   clipped to ±1.5°F.
4. **Repaired the learning loop.** The daily retrain had been silently failing
   for 3 weeks on NaN inputs (missing forecast sources from API failures).
   Missing sources are now imputed row-wise from available sources. Lesson:
   a self-learning system needs monitoring on the learning itself.

## 4. Strategy redesign: from modeling edge to attention edge

Since the data showed no exploitable modeling edge against same-information
market makers, the second strategy (`sniper.py`) targets a structurally
different source: **same-day contracts in the afternoon**, where real-time
observations from the exact settlement station (KMDW/KNYC/KMIA) often pin the
final high while thin books still quote stale prices. The model there is a
Monte Carlo over `max(observed_max + CLI-vs-METAR spike, remaining-hours
forecast)` — an information/latency edge, not a forecasting edge.

## 5. Validation discipline

The retrospective rule analysis in §3 is **in-sample**: the rules were selected
on the same 89 trades used to evaluate them, and 89 binary outcomes is a small
sample. Treat +19–28% ROI as an upper bound, not an expectation.

Consequently, the new strategy gates itself: it runs dry, logs every signal,
verifies outcomes against settled actual highs, and only trades live after
≥15 verified signals show (a) positive hypothetical P&L and (b) claimed
probabilities within 15 points of realized frequency — the exact failure mode
documented in §2.2. The gate is allowed to conclude "no edge"; that is a
valid and cheap experimental outcome.

## 6. Takeaways

- **Point accuracy ≠ P&L.** A model can beat every input source on MAE and
  still lose money if the probability layer is overconfident relative to an
  efficient counterparty.
- **Condition on trading.** Evaluate forecasts on the *traded* subset; the
  selection effect (winner's curse) is the dominant bias in any
  disagreement-triggered strategy.
- **Benchmark against the market price**, which is itself a probability
  forecast, using proper scoring rules (Brier, log loss) — not against naive
  baselines that flatter the model.
- **Shrinkage beats conviction.** Treating the market as a Bayesian prior is
  cheap insurance against phantom edges.
- **Capacity matters.** Thin books (5–10¢ spreads, shallow depth) bound the
  strategy at tens of dollars per month regardless of edge — fine for a
  research project, irrelevant as income. Knowing the capacity of an edge is
  part of evaluating it.
