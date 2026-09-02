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

# Risk parameters — percentage-based, scale with bankroll automatically
KELLY_FRACTION = 0.25            # quarter-Kelly per trade
MAX_POSITION_PCT = 0.08          # max 8% of bankroll on any one position
MAX_RUN_EXPOSURE_PCT = 0.25     # max 25% of bankroll deployed per run
MAX_OPEN_POSITIONS = 10          # max TOTAL simultaneous open positions
MIN_EDGE_CENTS = 5               # live edge threshold (blended, net of fees)

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
