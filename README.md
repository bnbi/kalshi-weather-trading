# Kalshi Trading Bot

An automated trading system for [Kalshi](https://kalshi.com) prediction markets. Builds probabilistic models and places trades when your model disagrees with market prices.

## Project structure

```
kalshi-bot/
├── kalshi_client.py      # API client with RSA-PSS authentication
├── explore_markets.py    # Browse and inspect markets (no auth needed)
├── collect_data.py       # Collect market snapshots into SQLite
├── config.example.py     # Configuration template
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup

### 1. Install dependencies

```bash
cd kalshi-bot
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get your Kalshi API credentials

1. Go to [kalshi.com/account/api](https://kalshi.com/account/api)
2. Generate an API key — you'll get a **Key ID** and be prompted to create an RSA key pair
3. Generate a 2048-bit RSA key pair:

```bash
openssl genrsa -out kalshi_private_key.pem 2048
openssl rsa -in kalshi_private_key.pem -pubout -out kalshi_public_key.pem
```

4. Upload `kalshi_public_key.pem` to Kalshi's API settings page
5. Keep `kalshi_private_key.pem` in this directory (it's gitignored)

### 3. Configure

```bash
cp config.example.py config.py
```

Edit `config.py` with your API Key ID and private key path. Leave `USE_DEMO = True` until you're ready to trade real money.

## Usage

### Explore markets (no auth needed)

```bash
# List open markets
python explore_markets.py

# Search for weather markets
python explore_markets.py --search weather

# Inspect a specific market
python explore_markets.py --ticker KXHIGHNY-25JUN15-T80
```

### Collect data

```bash
# Collect all open markets and price snapshots
python collect_data.py

# Filter to a specific series
python collect_data.py --series KXHIGHNY

# Also collect order books
python collect_data.py --orderbooks
```

Data is stored in `kalshi_data.db` (SQLite). You can query it with any SQLite tool or directly in Python with pandas:

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect("kalshi_data.db")
df = pd.read_sql("SELECT * FROM snapshots WHERE ticker LIKE 'KXHIGH%'", conn)
```

## Roadmap

- [x] Phase 1: API client and data collection
- [ ] Phase 2: Probability model (weather forecasts)
- [ ] Phase 3: Trading logic and risk management
- [ ] Phase 4: Live trading and performance analysis
