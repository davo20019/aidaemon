#!/usr/bin/env bash
# CI only: make the Apple Distribution certificate and the App Store
# provisioning profile from the repository's secrets usable by
# `nsh package --target ios`, and print the variables the release job reads
# (for $GITHUB_ENV). With the secrets unset (a fork, a local run) it prints
# nothing and the job skips the iOS package.
#
# Secrets (as environment variables):
#   IOS_CERTIFICATE_P12_BASE64   the Apple Distribution certificate, .p12, base64
#   IOS_CERTIFICATE_PASSWORD     its export password
#   IOS_PROFILE_BASE64           the App Store provisioning profile for ai.aidaemon.app, base64
#   APPSTORE_API_KEY_P8_BASE64   an App Store Connect API key (.p8), base64 (TestFlight upload)
set -euo pipefail

# Only the variable lines may reach stdout ($GITHUB_ENV): `security` prints
# progress ("1 identity imported.") that GitHub rejects as an env line.
exec 3>&1 1>&2

if [ -z "${IOS_CERTIFICATE_P12_BASE64:-}" ] || [ -z "${IOS_PROFILE_BASE64:-}" ]; then
  echo "No iOS signing certificate or profile in this run: no iOS package." >&2
  exit 0
fi

work="${RUNNER_TEMP:-$(mktemp -d)}"
keychain="$work/ios-signing.keychain-db"
keychain_password="$(uuidgen)"

echo "$IOS_CERTIFICATE_P12_BASE64" | base64 --decode >"$work/ios-certificate.p12"
security create-keychain -p "$keychain_password" "$keychain"
security set-keychain-settings -lut 21600 "$keychain"
security unlock-keychain -p "$keychain_password" "$keychain"
security import "$work/ios-certificate.p12" -k "$keychain" -P "$IOS_CERTIFICATE_PASSWORD" \
  -T /usr/bin/codesign -T /usr/bin/security
security set-key-partition-list -S apple-tool:,apple: -s -k "$keychain_password" "$keychain" >/dev/null
security list-keychains -d user -s "$keychain" $(security list-keychains -d user | tr -d '"')
rm -f "$work/ios-certificate.p12"

identity="$(security find-identity -v -p codesigning "$keychain" \
  | awk -F'"' '/Apple Distribution/ {print $2; exit}')"
if [ -z "$identity" ]; then
  echo "The certificate holds no Apple Distribution identity." >&2
  exit 1
fi
echo "$IOS_PROFILE_BASE64" | base64 --decode >"$work/AppStore.mobileprovision"
echo "IOS_SIGN_IDENTITY=$identity" >&3
echo "IOS_PROFILE=$work/AppStore.mobileprovision" >&3

if [ -n "${APPSTORE_API_KEY_P8_BASE64:-}" ]; then
  echo "$APPSTORE_API_KEY_P8_BASE64" | base64 --decode >"$work/appstore.p8"
  echo "NSH_APP_STORE_PRIVATE_KEY=$work/appstore.p8" >&3
fi
