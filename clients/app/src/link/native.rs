//! The desktops and the terminal: the shared tokio link, signed in with the
//! owner's local token when the daemon runs on this machine.

use std::path::PathBuf;

use aidaemon_client_core::link::{self, Auth, LinkConfig};
use aidaemon_client_core::machine::{ClientIdentity, Update};
use aidaemon_client_protocol::{Capability, ClientKind};
use nsh_core::platform::Idiom;
use nsh_core::prelude::*;

use crate::store::Store;

const DEFAULT_URL: &str = "ws://127.0.0.1:8788/client/v1/ws";

/// Where the gateway is: `AIDAEMON_GATEWAY_URL`, else the local default.
fn gateway_url() -> String {
    std::env::var("AIDAEMON_GATEWAY_URL")
        .ok()
        .filter(|url| !url.trim().is_empty())
        .map(|url| {
            if url.starts_with("http") {
                aidaemon_client_core::machine::socket_url(&url)
            } else {
                url
            }
        })
        .unwrap_or_else(|| DEFAULT_URL.to_owned())
}

/// The gateway's HTTP origin, from its socket URL (`ws://h:p/client/v1/ws`):
/// where a message's files are fetched.
fn http_base(socket: &str) -> String {
    let http = if let Some(rest) = socket.strip_prefix("wss://") {
        format!("https://{rest}")
    } else if let Some(rest) = socket.strip_prefix("ws://") {
        format!("http://{rest}")
    } else {
        socket.to_owned()
    };
    match http.find("/client/v1") {
        Some(at) => http[..at].to_owned(),
        None => http.trim_end_matches('/').to_owned(),
    }
}

/// The owner's local token: from `aidaemon app` / `aidaemon tui`
/// (`AIDAEMON_CLIENT_TOKEN`), else the saved copy (`aidaemon client-token`).
fn local_token() -> Option<String> {
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

/// Where this app keeps its key once paired with a daemon elsewhere.
fn device_file() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".aidaemon").join("app-device.json"))
}

fn device_store() -> Option<std::sync::Arc<aidaemon_client_core::FileKeyStore>> {
    device_file().map(|path| std::sync::Arc::new(aidaemon_client_core::FileKeyStore::new(path)))
}

pub fn start(store: Store) {
    use aidaemon_client_core::KeyStore;
    let (url, auth) = if let Some(token) = local_token() {
        (gateway_url(), Auth::LocalToken(token))
    } else if let Some(keys) = device_store()
        && let Ok(Some(device)) = keys.load()
    {
        (
            aidaemon_client_core::machine::socket_url(&device.gateway),
            Auth::Device(keys),
        )
    } else {
        // Nothing to sign in to yet: where should the agent run?
        store.needs_daemon();
        return;
    };
    store.set_gateway_base(http_base(&url));
    let kind = if idiom() == Idiom::Terminal {
        ClientKind::Tui
    } else {
        ClientKind::Desktop
    };
    let config = LinkConfig {
        url,
        auth,
        client: ClientIdentity {
            kind,
            name: "aidaemon-app".into(),
            version: Some(env!("CARGO_PKG_VERSION").into()),
        },
        device: None,
        capabilities: vec![Capability::ViewActions],
    };
    let (updates_tx, mut updates) = tokio::sync::mpsc::unbounded_channel::<Update>();
    spawn(async move {
        // The link's tasks run on the runtime behind `background`.
        let Ok(commands) = background(async move { link::spawn(config, updates_tx) }).await else {
            log::error!("could not start the link");
            return;
        };
        store.attach(Box::new(move |id, command| {
            let _ = commands.send((id, command));
        }));
        while let Some(update) = updates.recv().await {
            store.update(update);
        }
    });
}

/// The `aidaemon` binary: beside this app (the bundled download), else on
/// `PATH`.
fn daemon_binary() -> Option<PathBuf> {
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
/// administrator rights), wait for its token, then connect.
pub fn install_here(store: Store) {
    store.set_starting_daemon(Some(("Starting AIdaemon on this computer…".into(), false)));
    spawn(async move {
        let result =
            background(async move {
                // Run from the disk image, the daemon would vanish when it is
                // ejected.
                if cfg!(target_os = "macos")
                    && std::env::current_exe()
                        .ok()
                        .is_some_and(|exe| exe.starts_with("/Volumes"))
                {
                    return Err(
                        "Drag AIdaemon to Applications first, then open it from there.".to_string(),
                    );
                }
                let binary = daemon_binary().ok_or_else(|| {
                    "The aidaemon program is not beside this app. Reinstall the app, or install \
                 AIdaemon with the install script."
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
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
                Err("AIdaemon did not start within a minute. Its log is in ~/Library/Logs/aidaemon \
                 (macOS) or `journalctl --user -u aidaemon` (Linux)."
                .to_string())
            })
            .await
            .unwrap_or_else(|_| Err("Could not start AIdaemon".to_string()));
        match result {
            Ok(()) => {
                store.set_starting_daemon(None);
                store.update(Update::Connecting);
                start(store);
            }
            Err(why) => store.set_starting_daemon(Some((why, true))),
        }
    });
}

/// The parts of a pairing link: the gateway's HTTP base, the offer, the
/// code and the daemon's fingerprint. Takes the web link
/// (`http://host/client/v1/next/#offer=…&code=…&fp=…`) or an
/// `aidaemon-pair:?gateway=…` link.
pub fn parse_pairing_link(link: &str) -> Option<(String, String, String, Option<String>)> {
    let link = link.trim();
    let (base, query) = if let Some(rest) = link.strip_prefix("aidaemon-pair:?") {
        (None, rest)
    } else {
        let (url, fragment) = link.split_once('#')?;
        (
            Some(aidaemon_client_core::machine::http_base(url)),
            fragment,
        )
    };
    let mut gateway = base;
    let (mut offer, mut code, mut fp) = (None, None, None);
    for pair in query.split('&') {
        let (key, value) = pair.split_once('=')?;
        let value = percent_decode(value);
        match key {
            "gateway" => gateway = Some(value.trim_end_matches('/').to_owned()),
            "offer" => offer = Some(value),
            "code" => code = Some(value),
            "fp" => fp = Some(value),
            _ => {}
        }
    }
    Some((gateway?, offer?, code?, fp))
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(byte) = u8::from_str_radix(&text[i + 1..i + 3], 16)
        {
            out.push(byte);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Pair with a daemon elsewhere from its pairing link, keep the key, then
/// connect.
pub async fn pair_remote(store: Store, link: String, name: String) -> Result<(), String> {
    use aidaemon_client_core::KeyStore;
    let (gateway, offer_id, code, fingerprint) = parse_pairing_link(&link).ok_or_else(|| {
        "That is not a pairing link. Paste the whole link the daemon printed.".to_string()
    })?;
    let request = aidaemon_client_core::signin::PairRequest {
        gateway,
        offer_id,
        code,
        device_name: if name.trim().is_empty() {
            "Desktop".into()
        } else {
            name.trim().into()
        },
        device_kind: ClientKind::Desktop,
        fingerprint,
    };
    let device = background(async move {
        aidaemon_client_core::signin::pair(
            &reqwest::Client::new(),
            &request,
            std::time::Duration::from_millis(500),
        )
        .await
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|_| "Could not pair".to_string())??;
    let keys = device_store().ok_or("No home directory to keep the key in")?;
    keys.save(&device).map_err(|e| e.to_string())?;
    store.update(Update::Connecting);
    start(store);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn both_pairing_link_forms_read_back() {
        let web = super::parse_pairing_link(
            "http://127.0.0.1:8788/client/v1/next/#offer=of-1&code=K7Q2-9FZD&fp=abc",
        )
        .unwrap();
        assert_eq!(web.0, "http://127.0.0.1:8788");
        assert_eq!((web.1.as_str(), web.2.as_str()), ("of-1", "K7Q2-9FZD"));
        let app = super::parse_pairing_link(
            "aidaemon-pair:?gateway=https%3A%2F%2Fd.example.invalid&offer=of-2&code=AB&fp=f",
        )
        .unwrap();
        assert_eq!(app.0, "https://d.example.invalid");
        assert!(super::parse_pairing_link("hello").is_none());
    }
}
