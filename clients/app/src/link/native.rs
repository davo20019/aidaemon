//! The desktops and the terminal: the shared tokio link, signed in with the
//! owner's local token when the daemon runs on this machine.

use std::path::PathBuf;

use aidaemon_client_core::link::{self, Auth, LinkConfig};
use aidaemon_client_core::machine::{ClientIdentity, Disconnect, DropKind, Update};
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

pub fn start(store: Store) {
    let Some(token) = local_token() else {
        store.update(Update::Disconnected(Disconnect {
            kind: DropKind::Unauthorized,
            reason: "No local token. Run `aidaemon client-token` on this machine.".into(),
            retry_in: std::time::Duration::from_secs(3600),
        }));
        return;
    };
    let kind = if idiom() == Idiom::Terminal {
        ClientKind::Tui
    } else {
        ClientKind::Desktop
    };
    let config = LinkConfig {
        url: gateway_url(),
        auth: Auth::LocalToken(token),
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
