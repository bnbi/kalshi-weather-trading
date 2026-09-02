"""
Kalshi API Configuration Template
Copy this file to config.py and fill in your credentials.
NEVER commit config.py to git -- it contains your private key.
"""

# Your Kalshi API Key ID (from https://kalshi.com/account/api)
KALSHI_API_KEY_ID = "your-api-key-id-here"

# Path to your RSA private key file (.pem)
# Generate a 2048-bit RSA key pair and upload the public key to Kalshi
KALSHI_PRIVATE_KEY_PATH = "kalshi_private_key.pem"

# API base URLs
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"

# Start with demo mode — switch to production only when you're confident
USE_DEMO = True

# ── Schedule ───────────────────────────────────────────────────────
# Local hour of the daily launchd run. Keep in sync with the installed
# com.kalshi.weatherbot.plist; status displays and `scheduler.py daemon`
# derive "next run" from this.
RUN_HOUR = 13

# ── Sizing mode ────────────────────────────────────────────────────
# "global"   — signals from all cities pooled, sized best-edge-first
#              under one run budget (bigger, Kelly-true bets).
# "per_city" — the pre-2026-09-01 behavior: run budget split evenly
#              across cities (smaller bets).
SIZING_MODE = "global"
MAX_CONTRACTS_PER_ORDER = 15     # per-order cap; the binding limits are the
                                 # % position cap and fillable book size

# Risk parameters — percentage-based, scale with bankroll automatically.
# "Bankroll" = cash + cost of everything still open (both strategies).
KELLY_FRACTION = 0.25            # quarter-Kelly per trade
MAX_POSITION_PCT = 0.08          # max 8% of bankroll on any one position
MAX_RUN_EXPOSURE_PCT = 0.25      # max 25% of bankroll deployed per run
MAX_TOTAL_EXPOSURE_PCT = 0.40    # max 40% of bankroll open at once across
                                 # ALL runs of both strategies (the sniper
                                 # fires hourly; without this, 25%/run could
                                 # compound to half the bankroll at risk)
MAX_OPEN_POSITIONS = 6           # max TOTAL simultaneous open positions —
                                 # resting orders count; slots held come off
                                 # each run's allowance (both strategies)
MIN_EDGE_CENTS = 5               # live edge threshold (blended, net of fees)

# ── Execution ──────────────────────────────────────────────────────
# Orders are limit orders AT the ask, sized to what can fill at that
# price. Anything still unfilled after FILL_WAIT_SECONDS is canceled: a
# resting order fills exactly when the market moves against the model.
FILL_WAIT_SECONDS = 20
# Post inside the spread instead of taking the ask. Off by default — the
# fee-net edge already assumes paying the ask, and an improved order is
# adversely selected. (Unfilled improved orders are still canceled.)
IMPROVE_PRICES = False

# ── Probability layer ──────────────────────────────────────────────
# Live bias correction (°F) is clipped to ±MAX_BIAS_CORRECTION. The live
# window re-scores the CURRENT model on days with OFFICIAL (GHCND) actuals.
MAX_BIAS_CORRECTION = 3.0

# ── Apple WeatherKit (optional 5th forecast source) ──────────────
# From https://developer.apple.com/account:
#   Team ID     — top-right of the developer portal membership page
#   Service ID  — an Identifier of type "Services ID" with WeatherKit enabled
#   Key ID      — a Key with WeatherKit enabled; download its .p8 once
# Free tier is 500k calls/month; this bot uses ~7/day per collection run.
WEATHERKIT_TEAM_ID = "your-team-id-here"
WEATHERKIT_SERVICE_ID = "com.example.weatherbot"
WEATHERKIT_KEY_ID = "your-key-id-here"
WEATHERKIT_PRIVATE_KEY_PATH = "AuthKey_XXXXXXXXXX.p8"

# Master switch — set False to keep credentials but stop using the source.
WEATHERKIT_ENABLED = True

# How WeatherKit is allowed to influence live predictions:
#   "shadow"   — logged and used for uncertainty only; never moves the
#                point forecast. Correct until you have verified history.
#   "blend"    — additionally applies a learned blend weight toward Apple's
#                forecast, activated automatically once enough verified days
#                exist (see WEATHERKIT_MIN_VERIFIED_DAYS).
WEATHERKIT_MODE = "shadow"

# Verified days required before the learned blend weight switches on.
WEATHERKIT_MIN_VERIFIED_DAYS = 45

# Hard cap on that blend weight, so Apple can never dominate the ML model.
WEATHERKIT_MAX_WEIGHT = 0.35
