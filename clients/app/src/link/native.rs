//! The desktops and the terminal: the shared tokio link on nsh's
//! background threads. Where to sign in, pairing and starting the daemon
//! here are `aidaemon_client_core::local`'s.

use aidaemon_client_core::link::{self, LinkConfig};
use aidaemon_client_core::local;
use aidaemon_client_core::machine::{ClientIdentity, Update};
use aidaemon_client_protocol::{Capability, ClientKind};
use nsh_core::platform::Idiom;
use nsh_core::prelude::*;

use crate::store::Store;

pub fn start(store: Store) {
    let Some((url, auth)) = local::sign_in() else {
        // Nothing to sign in to yet: where should the agent run?
        store.needs_daemon();
        return;
    };
    store.set_gateway_base(aidaemon_client_core::machine::http_base(&url));
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

/// Run the daemon on this computer: install it to start at login (no
/// administrator rights), wait for its token, then connect.
pub fn install_here(store: Store) {
    store.set_starting_daemon(Some(("Starting AIdaemon on this computer…".into(), false)));
    spawn(async move {
        let result = background(local::install_daemon())
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

/// Pair with a daemon elsewhere from its pairing link, keep the key, then
/// connect.
pub async fn pair_remote(store: Store, link: String, name: String) -> Result<(), String> {
    background(async move { local::pair_from_link(&link, &name, ClientKind::Desktop).await })
        .await
        .map_err(|_| "Could not pair".to_string())??;
    store.update(Update::Connecting);
    start(store);
    Ok(())
}
