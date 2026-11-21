#!/usr/bin/env bash
# Build the signed Android app (APK and App Bundle) in CI and, when a Google
# Play service account key is given, upload the bundle to Play's internal
# testing track. Used by release.yml (every tagged release) and
# android-beta.yml (a run started by hand, from any branch).
#
# Needs (environment):
#   ANDROID_KEYSTORE_BASE64, NSH_KEYSTORE_PASSWORD   the upload key
#   ANDROID_KEY_ALIAS                                its alias (default: upload)
#   APP_VERSION                                      e.g. 0.22.0
#   PLAY_SERVICE_ACCOUNT_JSON                        optional: publish to Play
# Writes aidaemon-android.apk, its .sha256 and aidaemon-android.aab in the
# repository root.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root/clients/app"

# Android's versionCode must grow with every upload, across both workflows
# (each has its own run counter): minutes since 2026-01-01 UTC.
build=$(( ($(date -u +%s) - 1767225600) / 60 ))
version="${APP_VERSION%%-*}"
# The app shows the same version it is published as.
export AIDAEMON_APP_VERSION="$version" AIDAEMON_APP_BUILD="$build"

nsh component add android
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$ANDROID_NDK_LATEST_HOME}"
keystore="$RUNNER_TEMP/release.jks"
trap 'rm -f "$keystore" "$RUNNER_TEMP/play.json"' EXIT
echo "$ANDROID_KEYSTORE_BASE64" | base64 --decode > "$keystore"
args=(--keystore "$keystore" --key-alias "${ANDROID_KEY_ALIAS:-upload}"
      --app-version "$version" --build "$build")
nsh package --target android "${args[@]}"
nsh package --target android --format aab "${args[@]}"

cp .nsh/out/AIdaemon.apk "$root/aidaemon-android.apk"
cp .nsh/out/AIdaemon.aab "$root/aidaemon-android.aab"
(cd "$root" && shasum -a 256 aidaemon-android.apk > aidaemon-android.apk.sha256)
echo "Built $version ($build)"

if [ -n "${PLAY_SERVICE_ACCOUNT_JSON:-}" ]; then
  printf '%s' "$PLAY_SERVICE_ACCOUNT_JSON" > "$RUNNER_TEMP/play.json"
  nsh publish android "$root/aidaemon-android.aab" --credentials "$RUNNER_TEMP/play.json"
  echo "Published $version ($build) to Google Play internal testing"
else
  echo "No PLAY_SERVICE_ACCOUNT_JSON: not published to Google Play"
fi
