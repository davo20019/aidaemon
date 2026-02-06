#!/bin/sh
# aidaemon installer
# Usage: curl -sSfL https://get.aidaemon.ai | bash
set -eu

REPO="davo20019/aidaemon"
INSTALL_DIR="${AIDAEMON_INSTALL_DIR:-/usr/local/bin}"
BINARY="aidaemon"

# --- helpers ---

info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$1" >&2; }
err()  { printf '\033[1;31merror:\033[0m %s\n' "$1" >&2; exit 1; }

need() {
    command -v "$1" >/dev/null 2>&1 || err "'$1' is required but not found"
}

# --- detect platform ---

detect_platform() {
    OS=$(uname -s)
    ARCH=$(uname -m)

    case "$OS" in
        Linux)  OS_NAME="linux" ;;
        Darwin) OS_NAME="macos" ;;
        *)      err "Unsupported OS: $OS" ;;
    esac

    case "$ARCH" in
        x86_64|amd64)  ARCH_NAME="x86_64" ;;
        aarch64|arm64) ARCH_NAME="aarch64" ;;
        *)             err "Unsupported architecture: $ARCH" ;;
    esac

    PLATFORM="${OS_NAME}-${ARCH_NAME}"
}

# --- get latest version ---

get_latest_version() {
    need curl

    # Use GitHub redirect to avoid API rate limits (60/hr unauthenticated)
    VERSION=$(curl --proto '=https' --tlsv1.2 -sSfI \
        "https://github.com/${REPO}/releases/latest" 2>/dev/null \
        | grep -i '^location:' | sed 's|.*/tag/||;s/[[:space:]]*$//')

    [ -n "$VERSION" ] || err "Could not determine latest version (GitHub may be unreachable)"
    info "Latest version: $VERSION"
}

# --- download and install ---

install() {
    need tar

    # Check for existing installation
    if command -v "$BINARY" >/dev/null 2>&1; then
        CURRENT=$("$BINARY" --version 2>/dev/null || echo "unknown")
        info "Existing installation found: ${CURRENT}"
        info "It will be replaced with ${VERSION}"
    fi

    TARBALL="aidaemon-${PLATFORM}.tar.gz"
    URL="https://github.com/${REPO}/releases/download/${VERSION}/${TARBALL}"
    CHECKSUM_URL="${URL}.sha256"

    info "Downloading ${TARBALL}..."
    _tmpdir=$(mktemp -d)
    trap 'rm -rf "$_tmpdir"' EXIT

    curl --proto '=https' --tlsv1.2 -sSfL "$URL" -o "${_tmpdir}/${TARBALL}" \
        || err "Download failed. No binary available for ${PLATFORM}."

    # Verify checksum
    info "Verifying checksum..."
    curl --proto '=https' --tlsv1.2 -sSfL "$CHECKSUM_URL" -o "${_tmpdir}/${TARBALL}.sha256" \
        || err "Checksum download failed"

    cd "$_tmpdir"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -c "${TARBALL}.sha256" >/dev/null 2>&1 || err "Checksum verification failed!"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -c "${TARBALL}.sha256" >/dev/null 2>&1 || err "Checksum verification failed!"
    else
        warn "No sha256sum or shasum found — skipping checksum verification"
    fi

    info "Extracting..."
    tar -xzf "${_tmpdir}/${TARBALL}" -C "$_tmpdir"

    info "Installing to ${INSTALL_DIR}/${BINARY}..."
    if [ -w "$INSTALL_DIR" ]; then
        mv "${_tmpdir}/${BINARY}" "${INSTALL_DIR}/${BINARY}"
    else
        sudo mv "${_tmpdir}/${BINARY}" "${INSTALL_DIR}/${BINARY}"
    fi
    chmod +x "${INSTALL_DIR}/${BINARY}"
}

# --- post-install ---

post_install() {
    printf '\n'
    info "aidaemon ${VERSION} installed to ${INSTALL_DIR}/${BINARY}"
    printf '\n'
    printf '  Get started:\n'
    printf '\n'
    printf '    \033[1maidaemon\033[0m                 # run the setup wizard\n'
    printf '    \033[1maidaemon install-service\033[0m  # install as systemd/launchd service\n'
    printf '\n'
    printf '  Docs: https://aidaemon.ai\n'
    printf '\n'
}

# --- main ---

main() {
    if [ "$(id -u)" -eq 0 ]; then
        warn "Running as root. Only the install step needs root privileges."
    fi

    info "Installing aidaemon..."
    detect_platform
    info "Detected platform: ${PLATFORM}"
    get_latest_version
    install
    post_install
}

main
