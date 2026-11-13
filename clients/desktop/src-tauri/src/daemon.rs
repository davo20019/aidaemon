//! Finding the local daemon and, when it is not running, asking the service
//! manager that already runs it to start it. The app never spawns the daemon
//! as its own child, never stops it, and quitting the app leaves it running.

use std::path::PathBuf;
use std::process::Command;

/// The gateway's loopback address on this machine: `AIDAEMON_GATEWAY_URL`,
/// else `[client_gateway] bind`/`port` from `~/.aidaemon/config.toml`
/// (or `AIDAEMON_CONFIG`), else the default `127.0.0.1:8788`.
pub fn local_gateway() -> String {
    if let Ok(url) = std::env::var("AIDAEMON_GATEWAY_URL") {
        if !url.trim().is_empty() {
            return url.trim().trim_end_matches('/').to_string();
        }
    }
    let table = config_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| text.parse::<toml::Table>().ok())
        .and_then(|table| table.get("client_gateway").cloned());
    let bind = table
        .as_ref()
        .and_then(|t| t.get("bind"))
        .and_then(|v| v.as_str())
        .unwrap_or("127.0.0.1")
        .to_string();
    let port = table
        .as_ref()
        .and_then(|t| t.get("port"))
        .and_then(|v| v.as_integer())
        .unwrap_or(8788);
    format!("http://{bind}:{port}")
}

fn config_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("AIDAEMON_CONFIG") {
        return Some(PathBuf::from(path));
    }
    Some(dirs::home_dir()?.join(".aidaemon").join("config.toml"))
}

/// The owner's copy of the local token that `aidaemon tui` uses, when this
/// machine runs the daemon: how the first run pairs without a code.
pub fn local_token() -> Option<String> {
    if let Ok(token) = std::env::var("AIDAEMON_CLIENT_TOKEN") {
        if !token.trim().is_empty() {
            return Some(token.trim().to_string());
        }
    }
    let path = dirs::home_dir()?.join(".aidaemon").join("client-token");
    let token = std::fs::read_to_string(path).ok()?;
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_string())
}

/// Ask the service manager to start (or restart) the daemon. Returns what
/// was done, in words, or why it could not be.
pub fn start() -> Result<String, String> {
    start_with_service_manager()
}

#[cfg(target_os = "macos")]
fn start_with_service_manager() -> Result<String, String> {
    // `aidaemon install-service` writes ~/Library/LaunchAgents/ai.aidaemon.plist.
    let uid = Command::new("id")
        .arg("-u")
        .output()
        .map_err(|e| e.to_string())?;
    let uid = String::from_utf8_lossy(&uid.stdout).trim().to_string();
    let target = format!("gui/{uid}/ai.aidaemon");
    let status = Command::new("launchctl")
        .args(["kickstart", &target])
        .status()
        .map_err(|e| e.to_string())?;
    if status.success() {
        Ok("Asked launchd to start the daemon.".into())
    } else {
        Err("launchd does not know the daemon here: run `aidaemon install-service` once, \
             then `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.aidaemon.plist`."
            .into())
    }
}

#[cfg(target_os = "linux")]
fn start_with_service_manager() -> Result<String, String> {
    // `aidaemon install-service` writes the system unit `aidaemon.service`;
    // a user unit of the same name is tried first.
    for args in [
        &["--user", "start", "aidaemon"][..],
        &["start", "aidaemon"][..],
    ] {
        if Command::new("systemctl")
            .args(args)
            .status()
            .is_ok_and(|s| s.success())
        {
            return Ok("Asked systemd to start the daemon.".into());
        }
    }
    Err("systemd could not start `aidaemon`: run `aidaemon install-service` once, \
         then `sudo systemctl enable --now aidaemon`."
        .into())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn start_with_service_manager() -> Result<String, String> {
    Err("Starting the daemon from the app is not supported on this system yet: \
         start it the way you installed it."
        .into())
}
