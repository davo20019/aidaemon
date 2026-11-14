//! The connection to the daemon's Client Gateway. On the desktops and in a
//! terminal it is `aidaemon-client-core`'s own link (the one the old TUI
//! uses), run on NSH's background threads; its updates come back to the engine thread and into the [`Store`].

#[cfg(target_arch = "wasm32")]
use nsh_core::prelude::*;

use crate::store::Store;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod random;
#[cfg(target_arch = "wasm32")]
pub mod web;

/// Start the link for this platform.
pub fn start(store: Store) {
    #[cfg(not(target_arch = "wasm32"))]
    native::start(store);
    #[cfg(target_arch = "wasm32")]
    web::start(store);
}

/// The pairing offer the link that opened the app carried: offer id, code
/// and the daemon's fingerprint.
pub fn offered_pairing() -> Option<(String, String, Option<String>)> {
    #[cfg(target_arch = "wasm32")]
    return web::offered_pairing();
    #[cfg(not(target_arch = "wasm32"))]
    None
}

/// Pair this device with the offer, then connect.
pub async fn pair(
    store: Store,
    offer: String,
    code: String,
    name: String,
    fingerprint: Option<String>,
) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    return web::pair(store, offer, code, name, fingerprint).await;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (store, offer, code, name, fingerprint);
        Err("On the daemon's own machine this app signs in with `aidaemon client-token`.".into())
    }
}

/// Forget this device's pairing and show the pairing screen (the daemon
/// refused it). Only the browser pairs from the app.
pub fn pair_again(store: Store) {
    #[cfg(target_arch = "wasm32")]
    spawn(async move {
        web::unpair().await;
        store.needs_pairing();
    });
    #[cfg(not(target_arch = "wasm32"))]
    let _ = store;
}

/// Whether this platform pairs from the app (the browser does).
pub const PAIRS_IN_APP: bool = cfg!(target_arch = "wasm32");
