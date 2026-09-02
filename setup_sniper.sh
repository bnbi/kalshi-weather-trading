#!/bin/bash
# Install the same-day sniper launchd job (hourly afternoon runs).
# Run once: bash setup_sniper.sh

set -e

BOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$BOT_DIR/venv/bin/python"
PLIST_SRC="$BOT_DIR/com.kalshi.sniper.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.kalshi.sniper.plist"
LABEL="com.kalshi.sniper"

echo "Setting up same-day sniper schedule (launchd)..."

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON (set up venv first)"
    exit 1
fi

mkdir -p "$BOT_DIR/logs" "$HOME/Library/LaunchAgents"

if launchctl list 2>/dev/null | grep -q "$LABEL"; then
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Rewrite the repo's placeholder paths to THIS checkout (a plain cp used
# to install broken /Users/YOUR_USERNAME/... paths).
sed "s|/Users/YOUR_USERNAME/kalshi-bot|$BOT_DIR|g" "$PLIST_SRC" > "$PLIST_DST"
if grep -q "YOUR_USERNAME" "$PLIST_DST"; then
    echo "ERROR: placeholder paths survived substitution — not loading."
    exit 1
fi
launchctl load "$PLIST_DST"

if launchctl list 2>/dev/null | grep -q "$LABEL"; then
    echo "SUCCESS — sniper runs hourly 12:20-19:20 local."
else
    echo "WARNING — job may not have loaded. Check: launchctl list | grep kalshi"
fi

echo ""
echo "The sniper starts in DRY mode and logs every signal."
echo "It flips itself live automatically once validation passes:"
echo "  >=30 signals graded on real settlement AND positive hypothetical P&L"
echo "  AND honest calibration AND a better Brier score than the ask price"
echo ""
echo "Commands:"
echo "  python sniper.py report     # check validation progress"
echo "  python sniper.py run --city chicago   # manual dry run"
echo "  python scheduler.py --kill  # kill switch stops the sniper too"
