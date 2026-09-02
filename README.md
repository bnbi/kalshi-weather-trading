# Kalshi Weather Trading System

[![CI](https://github.com/bnbi/kalshi-weather-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/bnbi/kalshi-weather-trading/actions/workflows/ci.yml)

An end-to-end automated trading system for Kalshi daily-high-temperature
markets: multi-model ensemble forecasting, ML bias correction, calibrated
probability estimation, Bayesian shrinkage against the market prior,
fractional-Kelly execution, and a self-validating second strategy — with full
live trade logging and an honest quantitative postmortem of real-money results.

**Read [RESEARCH.md](RESEARCH.md) first** — a postmortem of 89 live trades
covering winner's-curse diagnosis, proper-scoring-rule benchmarking against
market prices, and the structural fixes that followed. The negative result and
its analysis are the most interesting part of this project.

![Reliability diagram and live P&L](assets/calibration.png)

*The whole project in one chart (92 settled trades — the postmortem analyzed the
first 89; the three that settled later left the conclusion unchanged). Left: the
model's probabilities sit below the diagonal — overconfident — and score a worse
Brier (0.22) than the market-implied price (0.13), a −0.74 skill score. Right:
the resulting P&L: −$27.01 net of exchange fees (−$22.86 before fees, as the
postmortem originally reported). The point forecasts were good (live MAE
1.4°F); the edge died in the probability layer under adverse selection.
Regenerate with `python make_figures.py` (postmortem window; `--all` for the
full record); the underlying trades are in
[`data/live_trades.csv`](data/live_trades.csv), which now covers every settled
trade (the postmortem set is the first 92).*

## Highlights

- **Probabilistic forecasting & calibration.** Per-city correction models over
  a GFS/ECMWF/ICON/blend ensemble, trained only on rows where every source is
  a genuine **lead-1** forecast (2024 →, recency-weighted) and scored against
  **official settlement-station observations** (NOAA GHCND) — the number
  Kalshi settles on, never the provisional NWS obs-feed max or reanalysis.
  Uncertainty is day-specific: a second regressor predicts each day's σ from
  spread, season, and weather regime, and is discarded if it fails an
  out-of-sample calibration gate. Live recalibration re-scores the *current*
  model on the last 30 official days (recorded decision-time features), so a
  retrain never inherits its predecessor's bias, and everything is tracked
  with reliability tables, Brier score, log loss, and a skill score against
  the market-implied baseline.
- **Adverse-selection-aware trade selection.** Live data showed losses
  concentrated where model-market disagreement was largest (the winner's
  curse). Trades use shrinkage `p̂ = w·model + (1−w)·price` and a credibility
  filter that refuses trades beyond 25¢ of disagreement.
- **Risk management.** Fractional Kelly (quarter-Kelly) with
  percentage-of-bankroll ceilings per position, per run, and in total across
  both strategies (scale-invariant — no fixed dollar caps), edge thresholds
  net of the exact exchange fee, a minimum-size floor so a small bankroll can
  still express strong signals, orders sized to what can fill *at the limit
  price* with any unfilled remainder canceled after 20 s (no order rests all
  afternoon waiting to be adversely filled), de-duplication against open
  positions *and* resting orders, and a file-based kill switch with
  auto-engage on repeated order failures.
- **Walk-forward methodology.** Time-series CV for model selection (no
  lookahead; error-trend features lagged to what is verifiable at issue
  time), a walk-forward backtester on the live edge/sizing code, and a
  self-learning loop that re-verifies provisional actuals against GHCND as
  it publishes, syncs official days into the training set, and retrains
  *before* each day's trading.
- **Two edge hypotheses, tested separately.** (1) Next-day forecasting edge —
  tested live, found ≈ zero against same-information market makers, documented.
  (2) Same-day attention/latency edge from real-time settlement-station
  observations vs stale quotes — currently in self-validating dry-run: it
  trades live only after ≥30 signals graded on *real settlement* show
  positive EV, honest calibration, and a better Brier score than the ask.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Daily Pipeline (13:00 launchd)                  │
│                                                                  │
│  1. settle yesterday ──▶ 2. re-verify vs GHCND ──▶ retrain      │
│                                                                  │
│  3. Weather APIs ──▶ Ensemble + ML ──▶ Shrinkage vs ──▶ Kelly   │
│     GFS/ECMWF/       correction        market prior    fill-or- │
│     ICON/NWS         + day-specific σ    filter        cancel   │
│                      + live σ/bias (re-scored on official days) │
│                                                                  │
│  4. record forecasts + decision log for tomorrow's verification  │
├─────────────────────────────────────────────────────────────────┤
│         Same-Day Sniper (hourly 12:20–19:20 launchd)            │
│  settlement-station obs ──▶ Monte Carlo final-high ──▶ trade    │
│  (7 cities' stations)       distribution               near-    │
│                             gated by self-validation  certainty │
│                             on REAL settlement results          │
└─────────────────────────────────────────────────────────────────┘
```

| File | Purpose |
|------|---------|
| `weather_ensemble.py` | Multi-model ensemble, source exclusions, live σ/bias recalibration (re-scored), decision log |
| `train_model.py` | Feature engineering, time-series CV model selection, per-day σ model, residual calibration |
| `db_migrations.py` | Idempotent schema/provenance migrations (lead time, truth source, decision log) |
| `station_obs.py` | Settlement-station truth: official GHCND vs provisional NWS obs feed |
| `backfill_history.py` | One-shot bootstrap: forecast archive + station truth + lead-1 re-sourcing + retrain |
| `http_util.py` | GET with bounded retries/backoff for the public weather APIs |
| `model.py` | Contract parsing, Normal probability model |
| `find_edge.py` | Market-prior shrinkage, credibility filter, edge signals |
| `orderbook.py` | Depth, spread, slippage analysis |
| `trader.py` | Fractional Kelly sizing, limit-order execution, safety caps |
| `sniper.py` | Same-day observation-driven strategy with self-validation gate |
| `daily_learner.py` | Verify → append → retrain loop |
| `pnl_tracker.py` | Trade log, settlement checks, calibration & proper scoring report |
| `backtest.py` | Walk-forward backtester |
| `scheduler.py` | Orchestration, kill switch, budget management |
| `dashboard_app.py` | Flask monitoring dashboard (P&L, positions, calibration, kill switch) |
| `kalshi_client.py` | Kalshi REST client (RSA-PSS request signing) |

## Model details

**Ground truth:** training targets, live recalibration and the sniper gate use
only the official daily TMAX at the settlement stations via NOAA GHCND. The
NWS obs feed max is recorded as a *provisional* value for the 1–3 days until
GHCND publishes, then replaced: measured on 85 days it differed from the
official value on 41% of days (NYC 58%), with per-city biases up to −1.4°F —
a difference that flips bracket outcomes and, when it fed the live bias
correction, cooled NYC and Denver forecasts by ~0.7°F. Reanalysis (ERA5) is
never used as truth.

**Training rows:** only days where every forecast column is a genuine lead-1
value (the 2021–2023 archive is mixed-lead — same-day ICON next to lead-1 GFS
taught earlier models to lean on ICON far beyond its live skill) and the
actual is official, recency-weighted with a one-year half-life because source
biases drift as NWP models are upgraded. Live-collected days carry the live
feed's own forecasts. LA's ECMWF is excluded everywhere (inland 0.25° cell,
+6–11°F warm at LAX), and any source whose live bias exceeds 5°F is dropped
automatically.

**Features (18):** raw GFS/ECMWF/ICON/blend forecasts; ensemble mean and
spread; pairwise model differences; month and sin/cos day-of-year; wind,
humidity, cloud cover; GFS/ECMWF rolling 3-day error trends lagged two days —
the freshest errors that are actually verified when a lead-1 forecast is
issued.

**Apple WeatherKit (optional 5th source):** Apple post-processes its own
model mix rather than serving raw NWP output, so its errors are the least
correlated with GFS/ECMWF/ICON — the most useful thing to add to this
ensemble. Apple publishes no forecast *archive*, though, so it cannot be
backfilled the way ICON was; it accrues one day at a time from live
collection. Until it covers half the training set it stays out of the
feature matrix and is applied instead as a post-hoc shrinkage toward
Apple's number, with a single weight fitted on verified live rows and
capped at 0.35. Default mode is `shadow` (logged, widens σ via model
spread, never moves the point forecast); flip `WEATHERKIT_MODE` to
`blend` once `python daily_learner.py stats` reports a non-zero weight.
Configure with the `WEATHERKIT_*` settings in `config.example.py` and
verify with `python weatherkit.py`.

**Selection:** Ridge / Lasso / ElasticNet / GBM / RF compared by 5-fold
time-series CV MAE per city. Regularized linear models win consistently —
the forecast features are highly collinear.

**Uncertainty:** a gradient-boosted regressor fit on out-of-fold |residuals|
predicts a day-specific σ (clipped to 1.0–6.0°F), so a calm high-pressure day
and a pre-frontal day are not assigned the same confidence. An out-of-sample
calibration gate (time-ordered folds; observed coverage vs nominal 1σ) rejects
the σ model and falls back to the constant CV residual σ if it tests poorly.

**Probability layer:** `actual ~ Normal(prediction − live_bias, σ)`, where σ
is the max of the day-specific σ and the σ of the current model's errors over
the last 30 official days (re-scored from the decision log; bias clipped
±3°F), and probability clamps prevent phantom tail edges. Model spread is an
input to the day-σ model, not stacked on top of it — the old spread/4
inflation double-counted disagreement and manufactured fat-tail YES edge.

**Trade selection:** blended probability `0.5·model + 0.5·market`, minimum
blended edge net of the exact exchange fee (0.07·P·(1−P)), credibility filter
at 25¢, bracket-YES bets disallowed, price floors per side (the NO floor is
effectively 72¢ once the probability clamp and credibility band interact).
Rationale and evidence in [RESEARCH.md](RESEARCH.md) §3.

## Live results (the honest part)

89 settled trades, May 6 – June 11, 2026: **−$22.86** on ~$120 staked.
Diagnosis: point forecasts were good; probabilities were overconfident *on the
traded subset* due to selection effects, and the market was the better
probability forecaster (Brier skill −0.76 on traded contracts). The entire
loss sat in trades with >25¢ model-market disagreement. Full analysis,
including why the retrospective +ROI of the new rules must be discounted as
in-sample, in [RESEARCH.md](RESEARCH.md).

Current era (Aug 18, 2026 →, rebuilt pipeline): 23 settled trades, 20 wins,
**+$14.52** on $27.60 staked as of Sep 2 — a good run over a sample far too
small to distinguish from luck, led by exactly the threshold-YES bets the
postmortem rated worst. The Sep 2 audit also found that the data feeding the
probability layer carried systematic 1–2°F errors (provisional truth, mixed
lead times, a broken LA input); those are fixed, and the honest baseline for
the corrected system starts from that date.

`python pnl_tracker.py calibration --since 2026-08-18` reproduces the
reliability table and scoring metrics from the live trade log at any time.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Kalshi API auth (RSA key pair)
openssl genrsa -out kalshi_private_key.pem 2048
openssl rsa -in kalshi_private_key.pem -pubout -out kalshi_public_key.pem
# upload public key at kalshi.com/account/api, then:
cp config.example.py config.py   # add your API key id + key path

# Bootstrap: archived forecasts (re-sourced at lead-1) + official station
# observations, then trains all per-city models (idempotent, ~20 min)
python backfill_history.py

python scheduler.py run --dry-run    # verify pipeline
bash setup_schedule.sh               # daily 10am paper run via launchd
bash setup_sniper.sh                 # hourly afternoon sniper (self-gating)
python dashboard_app.py              # monitoring at localhost:5050
```

For fully hands-off operation without keeping a machine awake, the GitHub
Actions paper-trade workflow runs the pipeline on a schedule in the cloud — see
**Automation & safety** below.

## CLI reference

```bash
python scheduler.py run [--dry-run] [--max-spend N]   # main strategy
python scheduler.py --kill | --resume | --status      # kill switch
python sniper.py run [--auto|--live] [--city X]       # same-day strategy
python sniper.py report                               # validation gate status
python backtest.py --all --chart                      # walk-forward backtest (live edge/sizing code, naive market)
python pnl_tracker.py summary|daily|calibration       # performance & scoring
python find_edge.py chicago --show-all                # model vs market table
python train_all_cities.py                            # retrain models
pytest -q                                             # run the test suite
```

## Automation & safety

The system is designed to run **hands-off and money-safe by default**:

- **Serverless schedule (no machine of your own).** A GitHub Actions workflow
  ([`.github/workflows/paper-trade.yml`](.github/workflows/paper-trade.yml))
  runs the full decision pipeline daily in **paper mode** on GitHub's
  infrastructure — fetching public market and weather data, generating
  predictions and edge signals, and printing the trade plan it *would* place. It
  uses no API credentials and places no real orders.
- **Continuous integration.** Every push runs the test suite
  ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) over the core
  quantitative logic (Kelly sizing, calibration, probability model, edge +
  shrinkage, the self-validation gate) — see [`tests/`](tests/).
- **Local automation is paper by default too.** The launchd template runs
  `scheduler.py --dry-run` at 13:00 local (paper runs still settle, re-verify,
  retrain and log decisions — only the orders are simulated); the same-day
  sniper runs `--auto`, which logs every signal but trades real money only
  after its self-validation gate passes (≥30 signals graded on real
  settlement, positive EV, ≤15pt calibration gap, Brier below the ask).
- **Live trading is an explicit opt-in.** It requires placing your Kalshi
  credentials (`config.py` + RSA key) on the host and removing `--dry-run`.
  Given live results showed ≈ zero edge in next-day markets, the defaults
  prioritise a correct, observable system over chasing thin returns.

## Known limitations

1. **No information asymmetry in next-day markets** — market makers see the
   same public model runs; live results confirmed ≈ zero modeling edge there.
2. **Thin markets** — 5–10¢ spreads and shallow depth cap capacity at tens of
   dollars/month regardless of edge quality.
3. **Backtest ≠ live** — the backtester's simulated market is deliberately
   naive; it validates predictive skill vs a baseline, not dollar returns.
4. **Small live samples** — 115 settled trades, ~90 official live-verified
   days for the original three cities and ~20 for the four added in August
   (the archive helps the forecast model, not the live P&L record); every
   conclusion carries wide confidence intervals, which is why the sniper
   gates itself before risking capital. Re-grading the sniper's earlier
   signals against official settlement flipped 19 of 37 outcomes that had
   been scored against reanalysis/feed values — its track record restarts
   from Sep 2026.

## Stack

Python · scikit-learn · scipy · numpy · pandas · SQLite · Flask · launchd ·
Kalshi REST API (RSA-PSS) · NWS / Open-Meteo / NOAA NCEI / Apple
WeatherKit (ES256 JWT) APIs
