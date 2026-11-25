#!/usr/bin/env bash
# Run scripts/dev/target-gc.sh every hour (and at login) for this checkout.
# macOS: a launchd agent running a copy of the scripts (so switching or
# removing this checkout's branch never changes or breaks the job; rerun this
# to pick up a newer version). Elsewhere: prints the crontab line to add.
#
#   scripts/dev/install-target-gc.sh             # install
#   scripts/dev/install-target-gc.sh --uninstall
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
gc="$repo/scripts/dev/target-gc.sh"
label="ai.aidaemon.target-gc"
# The main checkout (worktrees come and go; every one lists all of them).
main="$(dirname "$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)")"

if [ "$(uname)" != Darwin ]; then
  echo "Add this to your crontab (crontab -e):"
  echo "17 * * * * $gc >/dev/null 2>&1"
  exit 0
fi

plist="$HOME/Library/LaunchAgents/$label.plist"
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
if [ "${1:-}" = --uninstall ]; then
  rm -f "$plist"
  echo "removed $label"
  exit 0
fi

home="$HOME/Library/Application Support/aidaemon/target-gc"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs" "$home/dev"
cp "$repo/scripts/prune-target.py" "$home/prune-target.py"
cp "$gc" "$home/dev/target-gc.sh"
chmod +x "$home/dev/target-gc.sh"
cat >"$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>$home/dev/target-gc.sh</string></array>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin</string>
    <key>AIDAEMON_REPO</key><string>$main</string>
  </dict>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><true/>
  <key>LowPriorityIO</key><true/>
  <key>Nice</key><integer>10</integer>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/aidaemon-target-gc.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/aidaemon-target-gc.log</string>
</dict>
</plist>
PLIST
launchctl bootstrap "gui/$(id -u)" "$plist"
echo "installed $label: hourly, log in ~/Library/Logs/aidaemon-target-gc.log"
