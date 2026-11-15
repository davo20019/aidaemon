//! A client on a computer that may run the daemon itself (desktops, the
//! terminal): where the gateway is, the owner's local token, the key kept
//! after pairing with a daemon elsewhere, and starting the daemon here.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use aidaemon_client_protocol::ClientKind;

use crate::keystore::{FileKeyStore, KeyStore};
use crate::link::Auth;
use crate::signin::{PairRequest, PairingLink};

const DEFAULT_URL: &str = "ws://127.0.0.1:8788/client/v1/ws";

/// Where the gateway is: `AIDAEMON_GATEWAY_URL` (a socket or HTTP URL),
/// else the local default.
pub fn gateway_url() -> String {
    std::env::var("AIDAEMON_GATEWAY_URL")
        .ok()
        .filter(|url| !url.trim().is_empty())
        .map(|url| {
            if url.starts_with("http") {
                crate::machine::socket_url(&url)
            } else {
                url
            }
        })
        .unwrap_or_else(|| DEFAULT_URL.to_owned())
}

/// The owner's local token: from `aidaemon app` / `aidaemon tui`
/// (`AIDAEMON_CLIENT_TOKEN`), else the saved copy (`aidaemon client-token`,
/// or `AIDAEMON_CLIENT_TOKEN_FILE`).
pub fn local_token() -> Option<String> {
    if let Some(token) = std::env::var("AIDAEMON_CLIENT_TOKEN")
        .ok()
        .map(|t| t.trim().to_owned())
        .filter(|t| !t.is_empty())
    {
        return Some(token);
    }
    let path = std::env::var_os("AIDAEMON_CLIENT_TOKEN_FILE")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".aidaemon").join("client-token")))?;
    std::fs::read_to_string(path)
        .ok()
        .map(|t| t.trim().to_owned())
        .filter(|t| !t.is_empty())
}

/// Where the app keeps its key once paired with a daemon elsewhere.
pub fn device_store() -> Option<Arc<FileKeyStore>> {
    dirs::home_dir().map(|home| {
        Arc::new(FileKeyStore::new(
            home.join(".aidaemon").join("app-device.json"),
        ))
    })
}

/// How to sign in from here: the local token with the gateway's URL, else
/// the paired key with its daemon's. `None`: nothing to sign in to yet.
pub fn sign_in() -> Option<(String, Auth)> {
    if let Some(token) = local_token() {
        return Some((gateway_url(), Auth::LocalToken(token)));
    }
    let keys = device_store()?;
    let device = keys.load().ok()??;
    Some((
        crate::machine::socket_url(&device.gateway),
        Auth::Device(keys),
    ))
}

/// Pair with a daemon elsewhere from its pairing `link` and keep the key;
/// then [`sign_in`] finds it. The error is for the owner.
pub async fn pair_from_link(link: &str, name: &str, kind: ClientKind) -> Result<(), String> {
    let link = PairingLink::parse(link).ok_or_else(|| {
        "That is not a pairing link. Paste the whole link the daemon printed.".to_string()
    })?;
    let request = PairRequest {
        gateway: link.gateway,
        offer_id: link.offer_id,
        code: link.code,
        device_name: if name.trim().is_empty() {
            "Desktop".into()
        } else {
            name.trim().into()
        },
        device_kind: kind,
        fingerprint: link.fingerprint,
    };
    let device = crate::signin::pair(
        &reqwest::Client::new(),
        &request,
        Duration::from_millis(500),
    )
    .await
    .map_err(|e| e.to_string())?;
    let keys = device_store().ok_or("No home directory to keep the key in")?;
    keys.save(&device).map_err(|e| e.to_string())
}

/// The `aidaemon` binary: beside this app (the bundled download), else on
/// `PATH`.
pub fn daemon_binary() -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "aidaemon.exe"
    } else {
        "aidaemon"
    };
    let beside = std::env::current_exe().ok().and_then(|exe| {
        let dir = exe.parent()?.to_path_buf();
        // The macOS download carries it as a helper app; elsewhere it sits
        // beside this one.
        [
            dir.join("../Helpers/AIdaemon Daemon.app/Contents/MacOS")
                .join(name),
            dir.join(name),
        ]
        .into_iter()
        .find(|p| p.is_file())
    });
    beside.or_else(|| {
        std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join(name))
                .find(|path| path.is_file())
        })
    })
}

/// Run the daemon on this computer: install it to start at login (no
/// administrator rights) and wait for its token. The error is for the
/// owner.
pub async fn install_daemon() -> Result<(), String> {
    // Run from the disk image, the daemon would vanish when it is ejected.
    if cfg!(target_os = "macos")
        && std::env::current_exe()
            .ok()
            .is_some_and(|exe| exe.starts_with("/Volumes"))
    {
        return Err("Drag AIdaemon to Applications first, then open it from there.".to_string());
    }
    let binary = daemon_binary().ok_or_else(|| {
        "The aidaemon program is not beside this app. Reinstall the app, or install AIdaemon \
         with the install script."
            .to_string()
    })?;
    let output = std::process::Command::new(binary)
        .args(["service", "install"])
        .output()
        .map_err(|e| format!("Could not start AIdaemon: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    // The daemon mints the app's token as it starts.
    for _ in 0..120 {
        if local_token().is_some() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Err(
        "AIdaemon did not start within a minute. Its log is in ~/Library/Logs/aidaemon (macOS) \
         or `journalctl --user -u aidaemon` (Linux)."
            .to_string(),
    )
}
