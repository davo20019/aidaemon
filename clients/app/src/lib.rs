//! AIdaemon's app: chat, approvals and the agent's running work, on the
//! desktops, the web and a terminal, from one Rust crate.

use nsh_core::prelude::*;

pub mod features;
pub mod link;
pub mod shell;
pub mod store;
pub mod ui;

/// The app's UI set (`nsh.json` `"rust"`).
#[native_module("aidaemon/native")]
fn native(registry: &mut Registry) {
    registry.register::<App>();
    registry.register::<Offline>();
}

/// Frames waiting for an [`Offline`] root, from any thread.
static FEED: std::sync::Mutex<Vec<aidaemon_client_core::Update>> =
    std::sync::Mutex::new(Vec::new());

/// Hand `updates` to the [`Offline`] root, as the link would.
pub fn feed(updates: impl IntoIterator<Item = aidaemon_client_core::Update>) {
    if let Ok(mut feed) = FEED.lock() {
        feed.extend(updates);
    }
}

/// The shell over a store fed by [`feed`] instead of a link: tests and
/// previews drive it with gateway frames.
#[component]
pub fn Offline() -> impl IntoElement {
    let store = store::Store::new();
    provide_context(store);
    set_interval(std::time::Duration::from_millis(20), move || {
        let updates = FEED
            .lock()
            .map(|mut f| std::mem::take(&mut *f))
            .unwrap_or_default();
        for update in updates {
            store.update(update);
        }
    });
    view()
        .flex_1()
        .bg(ui::colors::BACKGROUND)
        .text_color(ui::colors::TEXT)
        .child(shell::shell(store))
}

/// The root: the store, the link, the shell.
#[component]
pub fn App() -> impl IntoElement {
    let store = store::Store::new();
    provide_context(store);
    store.restore();
    link::start(store);
    view()
        .flex_1()
        .bg(ui::colors::BACKGROUND)
        .text_color(ui::colors::TEXT)
        .child(shell::shell(store))
}
