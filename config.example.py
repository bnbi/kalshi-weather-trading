# Kalshi API Configuration
# Copy this file to config.py and fill in your credentials
# NEVER commit config.py to git — it contains your private key

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

# Risk parameters
MAX_POSITION_SIZE_CENTS = 500    # max $5 per contract
MAX_TOTAL_EXPOSURE_CENTS = 5000  # max $50 total across all positions
MAX_OPEN_POSITIONS = 10          # max number of simultaneous positions
KELLY_FRACTION = 0.25            # use quarter-Kelly sizing
MIN_EDGE_CENTS = 5               # only trade if edge > 5 cents
