#!/bin/bash
#
# Create a stable self-signed code-signing identity named "aidaemon-dev" in the
# login keychain and trust it for code signing.
#
# Why: macOS ties Accessibility / Screen Recording permission grants to a code
# *identity*. An ad-hoc signature changes on every `cargo build`, which
# invalidates the grant and forces you to re-approve computer_use constantly.
# Signing the app bundle with a stable identity makes the grant survive
# rebuilds. This is a ONE-TIME setup; re-running it is harmless (it skips if the
# identity already exists).
#
# This is for local development / self-hosting. For distributing notarized
# releases, a real Apple Developer ID is the production-grade alternative.
#
# Usage:  scripts/create-signing-identity.sh
set -euo pipefail

IDENTITY_NAME="aidaemon-dev"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENTITY_NAME"; then
  echo "Signing identity '$IDENTITY_NAME' already exists — nothing to do."
  exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Generating self-signed code-signing certificate '$IDENTITY_NAME'..."
openssl req -x509 -newkey rsa:2048 \
  -keyout "$WORK/key.pem" -out "$WORK/cert.pem" \
  -days 3650 -nodes -subj "/CN=$IDENTITY_NAME" \
  -addext "keyUsage=critical,digitalSignature" \
  -addext "extendedKeyUsage=critical,codeSigning" \
  -addext "basicConstraints=critical,CA:false" >/dev/null 2>&1

openssl pkcs12 -export -legacy \
  -out "$WORK/identity.p12" \
  -inkey "$WORK/key.pem" -in "$WORK/cert.pem" \
  -name "$IDENTITY_NAME" -passout pass:aidaemon >/dev/null 2>&1

echo "Importing into the login keychain..."
security import "$WORK/identity.p12" -k "$KEYCHAIN" -P aidaemon \
  -T /usr/bin/codesign -T /usr/bin/security >/dev/null

echo "Trusting the certificate for code signing..."
security add-trusted-cert -p codeSign -k "$KEYCHAIN" "$WORK/cert.pem" >/dev/null 2>&1 || true

if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENTITY_NAME"; then
  echo "Done. Identity '$IDENTITY_NAME' is ready."
  echo "Next: scripts/package-macos-app.sh   (build + bundle + install the daemon)"
else
  echo "warning: identity not visible after import; you may need to open Keychain Access" >&2
  echo "and set the '$IDENTITY_NAME' certificate's trust for Code Signing to 'Always Trust'." >&2
  exit 1
fi
