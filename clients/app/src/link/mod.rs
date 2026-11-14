//! The connection to the daemon's Client Gateway. On the desktops and in a
//! terminal it is `aidaemon-client-core`'s own link (the one the old TUI
//! and the Tauri shell use), run on NSH's background threads; its updates
//! come back to the engine thread and into the [`Store`].

use crate::store::Store;

#[cfg(not(target_arch = "wasm32"))]
mod native;

/// Start the link for this platform.
pub fn start(store: Store) {
    #[cfg(not(target_arch = "wasm32"))]
    native::start(store);
    #[cfg(target_arch = "wasm32")]
    {
        let _ = store;
        log::warn!("the browser link is not built yet");
    }
}
