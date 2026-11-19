#!/usr/bin/env bash
# Package the AIdaemon app for macOS as a disk image: AIdaemon.app with the
# daemon inside it as a helper app, so a download needs no terminal. The app's
# "Run it here" starts the helper at login from where it is.
#
#   AIdaemon.app/Contents/MacOS/AIdaemon                 the app (aidaemon-app)
#   AIdaemon.app/Contents/Helpers/AIdaemon Daemon.app    the daemon (aidaemon),
#                                                        bundle id ai.aidaemon
#
# Usage:
#   scripts/package-macos-dmg.sh <aidaemon> <aidaemon-app> <version> <out.dmg>
#
# Signing (all optional; without them the bundle is ad hoc signed, which is
# enough to try it locally but not to open a download by double-click):
#   SIGN_IDENTITY           "Developer ID Application: ..." (or its SHA-1)
#   NOTARY_KEY_PATH         App Store Connect API key (.p8), for notarytool
#   NOTARY_KEY_ID
#   NOTARY_ISSUER
set -euo pipefail

daemon_bin="$1"
app_bin="$2"
version="$3"
out_dmg="$4"
identity="${SIGN_IDENTITY:--}"
here="$(cd "$(dirname "$0")/.." && pwd)"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
app="$stage/AIdaemon.app"
helper="$app/Contents/Helpers/AIdaemon Daemon.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$helper/Contents/MacOS"

cp "$app_bin" "$app/Contents/MacOS/AIdaemon"
cp "$daemon_bin" "$helper/Contents/MacOS/aidaemon"

# The icon, from the mascot.
icon_src="$here/aidaemon-art/aidaemon_mascot_v3_1771684480903.png"
if [ -f "$icon_src" ]; then
  iconset="$stage/AIdaemon.iconset"
  mkdir -p "$iconset"
  for size in 16 32 128 256 512; do
    sips -s format png -z "$size" "$size" "$icon_src" --out "$iconset/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -s format png -z "$double" "$double" "$icon_src" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
  done
  iconutil -c icns "$iconset" -o "$app/Contents/Resources/AIdaemon.icns"
fi

cat >"$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>ai.aidaemon.app</string>
    <key>CFBundleName</key><string>AIdaemon</string>
    <key>CFBundleDisplayName</key><string>AIdaemon</string>
    <key>CFBundleExecutable</key><string>AIdaemon</string>
    <key>CFBundleIconFile</key><string>AIdaemon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>${version}</string>
    <key>CFBundleShortVersionString</key><string>${version}</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSMicrophoneUsageDescription</key><string>AIdaemon listens to the microphone while you dictate a message.</string>
    <key>NSSpeechRecognitionUsageDescription</key><string>AIdaemon turns what you dictate into the text of your message.</string>
</dict>
</plist>
EOF

# The daemon keeps the bundle id its macOS permissions (computer use) are
# granted to.
cat >"$helper/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>ai.aidaemon</string>
    <key>CFBundleName</key><string>AIdaemon Daemon</string>
    <key>CFBundleExecutable</key><string>aidaemon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>${version}</string>
    <key>CFBundleShortVersionString</key><string>${version}</string>
    <key>LSUIElement</key><true/>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
</dict>
</plist>
EOF

# Inside out: the helper, then the app. Notarization needs the hardened
# runtime and a secure timestamp on every signature.
sign() {
  if [ "$identity" = "-" ]; then
    codesign --force --sign - "$@"
  else
    codesign --force --options runtime --timestamp --sign "$identity" "$@"
  fi
}
sign --identifier ai.aidaemon "$helper"
# The app dictates (microphone and speech recognition): its Info.plist
# explains both, and the hardened runtime allows audio input only with this
# entitlement.
sign --identifier ai.aidaemon.app --entitlements "$here/scripts/macos/AIdaemon.entitlements" "$app"
codesign --verify --deep --strict "$app"

# The disk image: the app and a link to Applications to drag it onto. The
# link goes inside one source folder: a link given as a source of its own is
# followed, and would copy all of /Applications.
volume="$stage/volume"
mkdir -p "$volume"
mv "$app" "$volume/"
app="$volume/AIdaemon.app"
ln -s /Applications "$volume/Applications"
rm -f "$out_dmg"
hdiutil create -quiet -volname "AIdaemon" -srcfolder "$volume" -ov -format UDZO "$out_dmg"
if [ "$identity" != "-" ]; then
  codesign --force --timestamp --sign "$identity" "$out_dmg"
fi

if [ -n "${NOTARY_KEY_PATH:-}" ] && [ "$identity" != "-" ]; then
  xcrun notarytool submit "$out_dmg" \
    --key "$NOTARY_KEY_PATH" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER" \
    --wait
  xcrun stapler staple "$out_dmg"
  spctl --assess --type open --context context:primary-signature --verbose "$out_dmg"
fi

echo "Packaged $out_dmg (signed with ${identity})"
