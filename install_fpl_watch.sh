#!/bin/bash
# Install fpl_watch.py as a macOS launchd agent that runs every 30 min.
#
# Usage:
#   ./install_fpl_watch.sh TEAM_ID [HOURS_BEFORE]
#
# HOURS_BEFORE defaults to 24 (plan lands one day before the deadline).
# Uninstall with:
#   launchctl bootout gui/$UID/com.phutabol.fpl-watch
#   rm ~/Library/LaunchAgents/com.phutabol.fpl-watch.plist
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 TEAM_ID [HOURS_BEFORE]" >&2
    exit 1
fi

TEAM_ID="$1"
HOURS_BEFORE="${2:-24}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
LABEL="com.phutabol.fpl-watch"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/.phutabol"

mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
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

launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID" "$PLIST"
echo "Installed. Runs every 30 min; log: $LOG_DIR/watch.log"
echo "Plans land in $LOG_DIR/plans/ ~${HOURS_BEFORE}h before each deadline."
