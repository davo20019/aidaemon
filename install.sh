#!/bin/sh
# aidaemon installer
# Usage: curl -sSfL https://get.aidaemon.ai | bash
set -eu

REPO="davo20019/aidaemon"
INSTALL_DIR="${AIDAEMON_INSTALL_DIR:-/usr/local/bin}"
BINARY="aidaemon"
APP_BINARY="aidaemon-app"

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
    install_file "${_tmpdir}/${BINARY}" "${INSTALL_DIR}/${BINARY}"

    # The app (`aidaemon app` / `aidaemon tui`); older archives lack it.
    if [ -f "${_tmpdir}/${APP_BINARY}" ]; then
        info "Installing to ${INSTALL_DIR}/${APP_BINARY}..."
        install_file "${_tmpdir}/${APP_BINARY}" "${INSTALL_DIR}/${APP_BINARY}"
    else
        info "This release has no ${APP_BINARY} for ${PLATFORM}; skipping the app"
    fi

    install_web_app
}

install_file() {
    if [ -w "$INSTALL_DIR" ]; then
        mv "$1" "$2"
        chmod +x "$2"
    else
        sudo mv "$1" "$2"
        sudo chmod +x "$2"
    fi
}

verify_sha256() {
    # $1: checksum file in the current directory. Returns non-zero on mismatch.
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -c "$1" >/dev/null 2>&1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -c "$1" >/dev/null 2>&1
    else
        warn "No sha256sum or shasum found — skipping checksum verification"
    fi
}

# The app's web build, served by the daemon at /client/v1/next/. Never fails
# the install: a missing asset (older release) or a bad download is skipped.
install_web_app() {
    WEB_TARBALL="aidaemon-app-web.tar.gz"
    WEB_URL="https://github.com/${REPO}/releases/download/${VERSION}/${WEB_TARBALL}"
    WEB_DIR="${HOME}/.aidaemon/app-web"
    _webtmp="${_tmpdir}/web"
    mkdir -p "$_webtmp"

    info "Downloading ${WEB_TARBALL}..."
    if ! curl --proto '=https' --tlsv1.2 -sSfL "$WEB_URL" -o "${_webtmp}/${WEB_TARBALL}" 2>/dev/null \
        || ! curl --proto '=https' --tlsv1.2 -sSfL "${WEB_URL}.sha256" -o "${_webtmp}/${WEB_TARBALL}.sha256" 2>/dev/null; then
        info "This release has no web app; skipping it"
        return 0
    fi

    if ! (cd "$_webtmp" && verify_sha256 "${WEB_TARBALL}.sha256"); then
        warn "Web app checksum verification failed; skipping it"
        return 0
    fi

    # Extract beside the destination, then swap it in, so the daemon never
    # serves a half-written bundle.
    mkdir -p "$(dirname "$WEB_DIR")"
    _webnew="${WEB_DIR}.new.$$"
    _webold="${WEB_DIR}.old.$$"
    rm -rf "$_webnew"
    mkdir -p "$_webnew"
    if ! tar -xzf "${_webtmp}/${WEB_TARBALL}" -C "$_webnew" || [ ! -f "${_webnew}/index.html" ]; then
        rm -rf "$_webnew"
        warn "Web app archive could not be extracted; skipping it"
        return 0
    fi
    if [ -e "$WEB_DIR" ] && ! mv "$WEB_DIR" "$_webold"; then
        rm -rf "$_webnew"
        warn "Could not replace ${WEB_DIR}; skipping the web app"
        return 0
    fi
    if ! mv "$_webnew" "$WEB_DIR"; then
        [ -e "$_webold" ] && mv "$_webold" "$WEB_DIR"
        rm -rf "$_webnew"
        warn "Could not install the web app to ${WEB_DIR}; skipping it"
        return 0
    fi
    rm -rf "$_webold"
    info "Installed the web app to ${WEB_DIR}"
}

# --- post-install ---

run_post_install_migration() {
    # Opt-out for environments that want to skip post-install migration.
    if [ "${AIDAEMON_RUN_MIGRATION:-1}" = "0" ]; then
        info "Skipping post-install migration (AIDAEMON_RUN_MIGRATION=0)"
        return 0
    fi

    BIN_PATH="${INSTALL_DIR}/${BINARY}"
    if [ ! -x "$BIN_PATH" ]; then
        warn "Installed binary not executable at ${BIN_PATH}; skipping migration"
        return 0
    fi

    try_migrate_config() {
        cfg="$1"
        [ -n "$cfg" ] || return 1
        [ -f "$cfg" ] || return 1

        info "Running DB migration with config: ${cfg}"
        if "$BIN_PATH" migrate --config "$cfg"; then
            info "Migration completed"
            return 0
        fi

        warn "Migration failed for ${cfg}; daemon will retry on next startup"
        return 1
    }

    # Optional explicit override
    if [ -n "${AIDAEMON_MIGRATE_CONFIG:-}" ]; then
        try_migrate_config "${AIDAEMON_MIGRATE_CONFIG}" && return 0
    fi

    # Common config locations for upgrades
    try_migrate_config "$PWD/config.toml" && return 0
    try_migrate_config "${HOME}/.aidaemon/config.toml" && return 0
    try_migrate_config "${HOME}/.config/aidaemon/config.toml" && return 0

    info "No existing config.toml found; migration will run automatically on first launch"
}

post_install() {
    run_post_install_migration

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
