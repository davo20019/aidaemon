#!/bin/bash
#
# Package the aidaemon daemon as a signed macOS .app bundle and (re)install it
# as a launchd agent. Run this once to install, and again after every rebuild
# to refresh the bundle — re-signing with the same identity keeps your
# Accessibility / Screen Recording grants intact (they are keyed to the bundle
# id, not the binary hash).
#
# Prereq: a "aidaemon-dev" signing identity (run scripts/create-signing-identity.sh
# first). If it is missing, this falls back to ad-hoc signing, which works for an
# install-once setup but loses the grant on each rebuild.
#
# Usage:
#   scripts/package-macos-app.sh                 # uses target/release if present, else target/debug
#   scripts/package-macos-app.sh --debug         # force the debug build
#   scripts/package-macos-app.sh --build         # cargo build --release --features computer_use-macos first
#   scripts/package-macos-app.sh --build --debug # fast DEBUG build (no release opt) — for local/dev
#
# See COMPUTER_USE_MACOS.md for the full walkthrough.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HOME/Applications/aidaemon.app"
LABEL="ai.aidaemon"
BUNDLE_ID="ai.aidaemon"
IDENTITY="aidaemon-dev"
LOG_DIR="$HOME/Library/Logs/aidaemon"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
VERSION="$(grep -m1 '^version' "$PROJECT_DIR/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')"

PROFILE=""
DO_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --debug) PROFILE="debug" ;;
    --release) PROFILE="release" ;;
    --build) DO_BUILD=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [ "$DO_BUILD" = "1" ]; then
  if [ "$PROFILE" = "debug" ]; then
    # Fast dev path: skip release optimization/LTO. Compiles far quicker; the
    # binary runs slower, which is fine for local iteration.
    echo "Building DEBUG (fast, local/dev) with computer_use-macos..."
    ( cd "$PROJECT_DIR" && cargo build --features computer_use-macos )
    PROFILE="debug"
  else
    echo "Building release with computer_use-macos..."
    ( cd "$PROJECT_DIR" && cargo build --release --features computer_use-macos )
    PROFILE="release"
  fi
fi

if [ -z "$PROFILE" ]; then
  if [ -x "$PROJECT_DIR/target/release/aidaemon" ]; then PROFILE="release"; else PROFILE="debug"; fi
fi
BIN="$PROJECT_DIR/target/$PROFILE/aidaemon"
[ -x "$BIN" ] || { echo "error: $BIN not found — build first (e.g. --build)" >&2; exit 1; }

echo "Packaging $PROFILE binary -> $APP"
mkdir -p "$APP/Contents/MacOS" "$LOG_DIR"
chmod 700 "$LOG_DIR"
touch "$LOG_DIR/stderr.log"
chmod 600 "$LOG_DIR/stderr.log"
cp "$BIN" "$APP/Contents/MacOS/aidaemon"

cat > "$APP/Contents/Info.plist" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
    <key>CFBundleName</key><string>aidaemon</string>
    <key>CFBundleExecutable</key><string>aidaemon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>LSUIElement</key><true/>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
</dict>
</plist>
PLISTEOF

if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENTITY"; then
  SIGN_AS="$IDENTITY"
  echo "Signing with stable identity '$IDENTITY' (grants survive rebuilds)."
else
  SIGN_AS="-"
  echo "warning: '$IDENTITY' identity not found — using ad-hoc signature." >&2
  echo "         Grants will reset on each rebuild. Run scripts/create-signing-identity.sh." >&2
fi
codesign -f -s "$SIGN_AS" --identifier "$BUNDLE_ID" --timestamp=none "$APP"
codesign --verify --deep --strict "$APP"

# launchd hands a process a minimal PATH (/usr/bin:/bin:…), so user-installed
# tools (npm/node/wrangler via conda/homebrew) aren't found — `npm run deploy`
# style tasks fail with "command not found". Derive a PATH from the build env
# (node's bin dir + homebrew) so the daemon's terminal tool can find them.
# Portable: no hardcoded user path; resolves whatever node is installed.
NODE_BIN_DIR=""
if command -v node >/dev/null 2>&1; then NODE_BIN_DIR="$(dirname "$(command -v node)")"; fi
DAEMON_PATH="${NODE_BIN_DIR:+$NODE_BIN_DIR:}/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array><string>$APP/Contents/MacOS/aidaemon</string></array>
    <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
    <key>EnvironmentVariables</key>
    <dict><key>PATH</key><string>$DAEMON_PATH</string></dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/dev/null</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/stderr.log</string>
</dict>
</plist>
PLISTEOF

bootstrap_agent() {
  local attempt bootstrap_output
  for attempt in 1 2 3 4 5; do
    if bootstrap_output=$(launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>&1); then
      if [ "$attempt" != "1" ]; then
        echo "launchd bootstrap accepted on retry $attempt."
      fi
      return 0
    fi
    if [ "$attempt" != "5" ]; then
      sleep 0.2
    fi
  done
  printf '%s\n' "$bootstrap_output" >&2
  echo "error: launchd did not accept $PLIST after 5 attempts" >&2
  return 1
}

# (Re)load the launchd agent.
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "gui/$(id -u)/$LABEL"
  bootstrap_agent
  echo "Reloaded existing launchd agent."
else
  bootstrap_agent
  echo "Bootstrapped launchd agent."
fi

# Best-effort: re-assert the flags the local llama-server must carry (e.g.
# --mlock; see ensure-llama-flags.sh for the current set and rationale). The
# llama plist is hand-managed and not generated here, so this guards against it
# being recreated without the flag. Idempotent no-op when already present;
# never fatal to the daemon deploy.
if [[ -x "$PROJECT_DIR/scripts/ensure-llama-flags.sh" ]]; then
  "$PROJECT_DIR/scripts/ensure-llama-flags.sh" || echo "warning: ensure-llama-flags.sh failed (non-fatal)"
fi

echo
echo "Done. aidaemon.app installed and running ($VERSION, $PROFILE)."
echo "If this is your first install (or you switched signing identity), grant:"
echo "  • System Settings → Privacy & Security → Accessibility      → enable 'aidaemon'"
echo "  • System Settings → Privacy & Security → Screen Recording   → enable 'aidaemon'"
echo "then re-run this script (or restart the daemon) so Screen Recording takes effect."
echo "Full walkthrough: COMPUTER_USE_MACOS.md"
