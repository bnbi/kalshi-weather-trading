#!/bin/bash
# Setup script for automated daily trading via macOS launchd
# Run this once: bash setup_schedule.sh
#
# Why launchd instead of cron?
# - launchd is macOS-native and doesn't need Full Disk Access permissions
# - If your Mac is asleep at 7am, launchd runs the job as soon as it wakes up
# - No need for pmset hacks or keeping the lid open

set -e

BOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$BOT_DIR/venv/bin/python"
PLIST_SRC="$BOT_DIR/com.kalshi.weatherbot.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.kalshi.weatherbot.plist"
LABEL="com.kalshi.weatherbot"

echo "Setting up daily trading schedule (launchd)..."
echo "  Bot directory: $BOT_DIR"
echo "  Python: $PYTHON"
echo ""

# Check python exists
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Make sure your venv is set up:"
    echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check plist exists
if [ ! -f "$PLIST_SRC" ]; then
    echo "ERROR: Plist file not found at $PLIST_SRC"
    exit 1
fi

# Create logs directory
mkdir -p "$BOT_DIR/logs"

# Remove old cron job if present
if crontab -l 2>/dev/null | grep -q "scheduler.py"; then
    echo "Removing old cron job..."
    crontab -l 2>/dev/null | grep -v "scheduler.py" | crontab -
    echo "  Old cron job removed."
    echo ""
fi

# Unload existing launchd job if already installed
if launchctl list 2>/dev/null | grep -q "$LABEL"; then
    echo "Unloading existing launchd job..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$HOME/Library/LaunchAgents"

# Copy plist to LaunchAgents
cp "$PLIST_SRC" "$PLIST_DST"
echo "Installed plist to: $PLIST_DST"

# Load the job
launchctl load "$PLIST_DST"
echo "Loaded launchd job: $LABEL"
echo ""

# Verify
if launchctl list 2>/dev/null | grep -q "$LABEL"; then
    echo "SUCCESS — launchd job is active!"
else
    echo "WARNING — job may not have loaded. Check: launchctl list | grep kalshi"
fi

echo ""
echo "The bot will now run every day at 7:00 AM."
echo "If your Mac is asleep at 7am, it will run as soon as you open the lid."
echo ""
echo "Useful commands:"
echo "  python scheduler.py --status                    # Check if trading is active"
echo "  python scheduler.py --kill                      # STOP all trading immediately"
echo "  python scheduler.py --resume                    # Turn trading back on"
echo "  launchctl list | grep kalshi                    # Check if launchd job is loaded"
echo "  launchctl unload ~/Library/LaunchAgents/com.kalshi.weatherbot.plist  # Remove job"
echo ""
echo "Logs:"
echo "  $BOT_DIR/logs/launchd_stdout.log    # Bot output"
echo "  $BOT_DIR/logs/launchd_stderr.log    # Errors"
echo "  $BOT_DIR/logs/trader_YYYY-MM-DD.log # Per-day trading log"
echo ""
echo "Done!"
