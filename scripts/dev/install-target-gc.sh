#!/usr/bin/env bash
# Run scripts/dev/target-gc.sh once a day (and at login) for this checkout.
# macOS: a launchd agent. Elsewhere: prints the crontab line to add.
#
#   scripts/dev/install-target-gc.sh             # install
#   scripts/dev/install-target-gc.sh --uninstall
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
gc="$repo/scripts/dev/target-gc.sh"
label="ai.aidaemon.target-gc"

if [ "$(uname)" != Darwin ]; then
  echo "Add this to your crontab (crontab -e):"
  echo "17 13 * * * $gc >/dev/null 2>&1"
  exit 0
fi

plist="$HOME/Library/LaunchAgents/$label.plist"
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
if [ "${1:-}" = --uninstall ]; then
  rm -f "$plist"
  echo "removed $label"
  exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cat >"$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>$gc</string></array>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin</string></dict>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>13</integer><key>Minute</key><integer>17</integer></dict>
  <key>RunAtLoad</key><true/>
  <key>LowPriorityIO</key><true/>
  <key>Nice</key><integer>10</integer>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/aidaemon-target-gc.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/aidaemon-target-gc.log</string>
</dict>
</plist>
PLIST
launchctl bootstrap "gui/$(id -u)" "$plist"
echo "installed $label: daily, log in ~/Library/Logs/aidaemon-target-gc.log"
