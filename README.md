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
the resulting −$23 P&L. The point forecasts were good (live MAE 1.4°F); the edge
died in the probability layer under adverse selection. Regenerate with
`python make_figures.py`; the underlying trades are in [`data/live_trades.csv`](data/live_trades.csv).*

## Highlights

- **Probabilistic forecasting & calibration.** Per-city correction models over
  a GFS/ECMWF/ICON/blend ensemble (live MAE 1.36–1.54°F, beating every
  individual source), trained on five years of archived forecasts scored
  against **official settlement-station observations** (NOAA GHCND) — the same
  sensor readings Kalshi settles on, not gridded reanalysis. Uncertainty is
  day-specific: a second regressor predicts each day's σ from spread, season,
  and weather regime, and is automatically discarded if it fails a calibration
  gate. The probability layer is further recalibrated continuously from live
  verified errors and tracked with reliability tables, Brier score, log loss,
  and a skill score against the market-implied baseline.
- **Adverse-selection-aware trade selection.** Live data showed losses
  concentrated where model-market disagreement was largest (the winner's
  curse). Trades use shrinkage `p̂ = w·model + (1−w)·price` and a credibility
  filter that refuses trades beyond 25¢ of disagreement.
- **Risk management.** Fractional Kelly (quarter-Kelly) with
  percentage-of-bankroll ceilings per position and per run (scale-invariant —
  no fixed dollar caps), edge thresholds net of exchange fees, a minimum-size
  floor so a small bankroll can still express strong signals, orderbook depth
  and spread filters, position de-duplication, file-based kill switch with
  auto-engage on repeated order failures, and idempotent order submission
  (client-order-id retry with fresh request signatures).
- **Walk-forward methodology.** Time-series CV for model selection (no
  lookahead), walk-forward backtester, and a live self-learning loop that
  verifies yesterday's predictions, appends them to the training set, and
  retrains daily.
- **Two edge hypotheses, tested separately.** (1) Next-day forecasting edge —
  tested live, found ≈ zero against same-information market makers, documented.
  (2) Same-day attention/latency edge from real-time settlement-station
  observations vs stale quotes — currently in self-validating dry-run: it
  trades live only after ≥15 verified signals demonstrate positive EV and
  honest calibration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Daily Pipeline (10:00 launchd)                  │
│                                                                  │
│  Weather APIs ──▶ Ensemble + ML ──▶ Shrinkage vs ──▶ Kelly      │
│  GFS/ECMWF/       correction        market prior     execution  │
│  ICON/NWS         + day-specific σ    filter           filters  │
│                   + live σ/bias                                 │
│                     recalibration                               │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│              Self-Learning Loop (same run)                       │
│  verify yesterday ──▶ append to training set ──▶ retrain        │
├─────────────────────────────────────────────────────────────────┤
│         Same-Day Sniper (hourly 12:20–19:20 launchd)            │
│  settlement-station obs ──▶ Monte Carlo final-high ──▶ trade    │
│  (KMDW/KNYC/KMIA)           distribution               near-    │
│                             gated by self-validation  certainty │
└─────────────────────────────────────────────────────────────────┘
```

| File | Purpose |
|------|---------|
| `weather_ensemble.py` | Multi-model ensemble + live σ/bias recalibration |
| `train_model.py` | Feature engineering, time-series CV model selection, per-day σ model, residual calibration |
| `station_obs.py` | Official settlement-station observations (NOAA GHCND + NWS obs) |
| `backfill_history.py` | One-shot bootstrap: 5y forecast archive + station truth + retrain |
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

**Ground truth:** training targets and daily verification use the official
daily TMAX at the settlement stations themselves (Midway, Central Park, Miami
Intl) via NOAA GHCND, with near-real-time NWS station observations as a
fallback for recent dates. Reanalysis (ERA5) is retained only as a reference
column — a grid average is measurably not the number the exchange settles on.

**Features (19):** raw GFS/ECMWF/ICON/blend forecasts; ensemble mean and
spread; pairwise model differences; month and sin/cos day-of-year; wind,
humidity, cloud cover; per-model rolling 3-day error trends (shifted to avoid
lookahead).

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
and a pre-frontal day are not assigned the same confidence. A calibration gate
(observed coverage vs nominal 1σ) rejects the σ model and falls back to the
constant CV residual σ if it tests poorly.

**Probability layer:** `actual ~ Normal(prediction − live_bias, σ)`, where σ
is the conservative max of the day-specific σ and the σ of the last 30 live
verified errors (bias clipped ±1.5°F), plus spread-inflation for model
disagreement and probability clamps to prevent phantom tail edges.

**Trade selection:** blended probability `0.5·model + 0.5·market`, minimum
blended edge net of exchange fees, credibility filter at 25¢, bracket-YES bets
disallowed, price floors per side. Rationale and evidence in
[RESEARCH.md](RESEARCH.md) §3.

## Live results (the honest part)

89 settled trades, May 6 – June 11, 2026: **−$22.86** on ~$120 staked.
Diagnosis: point forecasts were good; probabilities were overconfident *on the
traded subset* due to selection effects, and the market was the better
probability forecaster (Brier skill −0.76 on traded contracts). The entire
loss sat in trades with >25¢ model-market disagreement. Full analysis,
including why the retrospective +ROI of the new rules must be discounted as
in-sample, in [RESEARCH.md](RESEARCH.md).

`python pnl_tracker.py calibration` reproduces the reliability table and
scoring metrics from the live trade log at any time.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Kalshi API auth (RSA key pair)
openssl genrsa -out kalshi_private_key.pem 2048
openssl rsa -in kalshi_private_key.pem -pubout -out kalshi_public_key.pem
# upload public key at kalshi.com/account/api, then:
cp config.example.py config.py   # add your API key id + key path

# Bootstrap: 5 years of archived forecasts + official station observations,
# then trains all per-city models (idempotent, ~15 min)
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
python backtest.py --all --chart                      # walk-forward backtest
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
- **Local automation is paper by default too.** The launchd jobs run
  `scheduler.py --dry-run`; the same-day sniper runs `--auto`, which logs every
  signal but trades real money only after its self-validation gate passes
  (≥15 verified signals, positive EV, ≤15pt calibration gap).
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
4. **Small live samples** — 92 settled trades, ~55 live-verified days per
   city (the 5-year backfill helps the forecast model, not the live P&L
   record); every conclusion carries wide confidence intervals, which is why
   the sniper gates itself before risking capital.

## Stack

Python · scikit-learn · scipy · numpy · pandas · SQLite · Flask · launchd ·
Kalshi REST API (RSA-PSS) · NWS / Open-Meteo / NOAA NCEI / Apple
WeatherKit (ES256 JWT) APIs
