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
  echo "Building release with computer_use-macos..."
  ( cd "$PROJECT_DIR" && cargo build --release --features computer_use-macos )
  PROFILE="release"
fi

if [ -z "$PROFILE" ]; then
  if [ -x "$PROJECT_DIR/target/release/aidaemon" ]; then PROFILE="release"; else PROFILE="debug"; fi
fi
BIN="$PROJECT_DIR/target/$PROFILE/aidaemon"
[ -x "$BIN" ] || { echo "error: $BIN not found — build first (e.g. --build)" >&2; exit 1; }

echo "Packaging $PROFILE binary -> $APP"
mkdir -p "$APP/Contents/MacOS" "$LOG_DIR"
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

cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array><string>$APP/Contents/MacOS/aidaemon</string></array>
    <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$LOG_DIR/stdout.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/stderr.log</string>
</dict>
</plist>
PLISTEOF

# (Re)load the launchd agent.
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  launchctl kickstart -k "gui/$(id -u)/$LABEL"
  echo "Restarted existing launchd agent."
else
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "Bootstrapped launchd agent."
fi

echo
echo "Done. aidaemon.app installed and running ($VERSION, $PROFILE)."
echo "If this is your first install (or you switched signing identity), grant:"
echo "  • System Settings → Privacy & Security → Accessibility      → enable 'aidaemon'"
echo "  • System Settings → Privacy & Security → Screen Recording   → enable 'aidaemon'"
echo "then re-run this script (or restart the daemon) so Screen Recording takes effect."
echo "Full walkthrough: COMPUTER_USE_MACOS.md"
