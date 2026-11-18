#!/usr/bin/env bash
# ONNX Runtime for an Intel Mac build (x86_64-apple-darwin).
#
# The daemon's local embeddings run on ONNX Runtime through fastembed 7, which
# needs its API 24 (ONNX Runtime 1.24 or newer). Microsoft stopped publishing
# Intel Mac builds after 1.23, and ort-sys downloads none for this target, so
# the build links a static library built by the sherpa-onnx project
# (csukuangfj/onnxruntime-libs), pinned by its SHA-256. Its version matches
# the one ort-sys's own prebuilts use for every other target.
#
# Prints ORT_LIB_LOCATION=<dir> for $GITHUB_ENV: ort-sys then links that
# directory's single libonnxruntime.a statically, so nothing ships beside the
# binary.
#
# Usage: scripts/onnxruntime-intel-mac.sh <work dir> >> "$GITHUB_ENV"
set -euo pipefail

VERSION="1.31.0"
NAME="onnxruntime-osx-x86_64-static_lib-${VERSION}"
URL="https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${VERSION}/${NAME}.zip"
SHA256="2b9b4c4052a39836d41074b56ebc61cb445d938c8ab7bbc8e55a97cdc74e8552"

work="${1:?usage: $0 <work dir>}"
mkdir -p "$work"
archive="$work/${NAME}.zip"

curl --proto '=https' --tlsv1.2 -sSfL --retry 3 -o "$archive" "$URL"
echo "${SHA256}  ${archive}" | shasum -a 256 -c - >&2
unzip -q -o "$archive" "${NAME}/lib/*" -d "$work"

lib="$work/${NAME}/lib"
[ -f "$lib/libonnxruntime.a" ] || { echo "no libonnxruntime.a in ${NAME}" >&2; exit 1; }
echo "ORT_LIB_LOCATION=${lib}"
