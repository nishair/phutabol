#!/bin/bash
# Install fpl_watch.py as a macOS launchd job that runs every 30 min.
#
# Usage:
#   ./install_fpl_watch.sh [--daemon] TEAM_ID [HOURS_BEFORE]
#
# Default: a per-user LaunchAgent (runs while you're logged in — right
# for a laptop). --daemon: a system LaunchDaemon that runs from boot
# with no login needed (right for an always-on box like a Mac mini
# where auto-login stays off; needs sudo; alerts should come from a
# Telegram/ntfy channel in ~/.phutabol/notify.json since there's no
# GUI session for local notifications).
#
# HOURS_BEFORE defaults to 24 (plan lands one day before the deadline).
# Uninstall:
#   launchctl bootout gui/$UID/com.phutabol.fpl-watch      # agent
#   sudo launchctl bootout system/com.phutabol.fpl-watch   # daemon
#   then delete the .plist the install printed.
set -euo pipefail

DAEMON=0
if [ "${1:-}" = "--daemon" ]; then
    DAEMON=1
    shift
fi
if [ $# -lt 1 ]; then
    echo "usage: $0 [--daemon] TEAM_ID [HOURS_BEFORE]" >&2
    exit 1
fi

TEAM_ID="$1"
HOURS_BEFORE="${2:-24}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
LABEL="com.phutabol.fpl-watch"
LOG_DIR="$HOME/.phutabol"
mkdir -p "$LOG_DIR"

if [ "$DAEMON" = 1 ]; then
    PLIST="/Library/LaunchDaemons/$LABEL.plist"
    EXTRA_KEYS="<key>UserName</key><string>$(whoami)</string>"
else
    PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
    EXTRA_KEYS=""
    mkdir -p "$HOME/Library/LaunchAgents"
fi

CONTENT=$(cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    $EXTRA_KEYS
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON</string>
        <string>$REPO_DIR/fpl_watch.py</string>
        <string>$TEAM_ID</string>
        <string>--hours-before</string>
        <string>$HOURS_BEFORE</string>
    </array>
    <key>WorkingDirectory</key><string>$REPO_DIR</string>
    <key>StartInterval</key><integer>1800</integer>
    <key>RunAtLoad</key><true/>
    <key>StandardOutPath</key><string>$LOG_DIR/watch.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/watch.log</string>
</dict>
</plist>
EOF
)

# Clear any previous install in either scope before loading.
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
if [ "$DAEMON" = 1 ]; then
    sudo launchctl bootout "system/$LABEL" 2>/dev/null || true
    echo "$CONTENT" | sudo tee "$PLIST" > /dev/null
    sudo launchctl bootstrap system "$PLIST"
else
    echo "$CONTENT" > "$PLIST"
    launchctl bootstrap "gui/$UID" "$PLIST"
fi

echo "Installed $PLIST — runs every 30 min; log: $LOG_DIR/watch.log"
echo "Plans land in $LOG_DIR/plans/ ~${HOURS_BEFORE}h before each deadline."
