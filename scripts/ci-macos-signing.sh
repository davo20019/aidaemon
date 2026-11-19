#!/usr/bin/env bash
# CI only: make the Developer ID certificate and the notary key from the
# repository's secrets usable by scripts/package-macos-dmg.sh, and print the
# variables it reads (for $GITHUB_ENV). With the secrets unset (a fork, a
# local run) it prints nothing and the package is ad hoc signed.
#
# Secrets (as environment variables):
#   MACOS_CERTIFICATE_P12_BASE64   the Developer ID Application certificate, .p12, base64
#   MACOS_CERTIFICATE_PASSWORD     its export password
#   APPSTORE_API_KEY_P8_BASE64     an App Store Connect API key (.p8), base64
#   APPSTORE_API_KEY_ID
#   APPSTORE_API_ISSUER_ID
set -euo pipefail

# Only the variable lines may reach stdout ($GITHUB_ENV): `security` prints
# progress ("1 identity imported.") that GitHub rejects as an env line.
exec 3>&1 1>&2

if [ -z "${MACOS_CERTIFICATE_P12_BASE64:-}" ]; then
  echo "No signing certificate in this run: the package will be ad hoc signed." >&2
  exit 0
fi

work="${RUNNER_TEMP:-$(mktemp -d)}"
keychain="$work/signing.keychain-db"
keychain_password="$(uuidgen)"

echo "$MACOS_CERTIFICATE_P12_BASE64" | base64 --decode >"$work/certificate.p12"
security create-keychain -p "$keychain_password" "$keychain"
security set-keychain-settings -lut 21600 "$keychain"
security unlock-keychain -p "$keychain_password" "$keychain"
security import "$work/certificate.p12" -k "$keychain" -P "$MACOS_CERTIFICATE_PASSWORD" \
  -T /usr/bin/codesign -T /usr/bin/security
security set-key-partition-list -S apple-tool:,apple: -s -k "$keychain_password" "$keychain" >/dev/null
# Search this keychain too.
security list-keychains -d user -s "$keychain" $(security list-keychains -d user | tr -d '"')
rm -f "$work/certificate.p12"

identity="$(security find-identity -v -p codesigning "$keychain" \
  | awk -F'"' '/Developer ID Application/ {print $2; exit}')"
if [ -z "$identity" ]; then
  echo "The certificate holds no Developer ID Application identity." >&2
  exit 1
fi
echo "SIGN_IDENTITY=$identity" >&3

if [ -n "${APPSTORE_API_KEY_P8_BASE64:-}" ]; then
  echo "$APPSTORE_API_KEY_P8_BASE64" | base64 --decode >"$work/notary.p8"
  echo "NOTARY_KEY_PATH=$work/notary.p8" >&3
  echo "NOTARY_KEY_ID=$APPSTORE_API_KEY_ID" >&3
  echo "NOTARY_ISSUER=$APPSTORE_API_ISSUER_ID" >&3
fi
