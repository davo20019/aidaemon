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

/// What the terminal prints to its normal scrollback after the app leaves
/// the alternate screen: the open conversation, so it is still there to
/// scroll, search and copy.
static PARTING: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Keep `text` to print once the terminal app has quit.
pub fn set_parting_words(text: String) {
    if let Ok(mut parting) = PARTING.lock() {
        *parting = Some(text);
    }
}

/// What [`set_parting_words`] kept, once.
pub fn take_parting_words() -> Option<String> {
    PARTING.lock().ok().and_then(|mut p| p.take())
}

/// The folder the app was started in, when that means something (the
/// terminal app, run in a project's folder).
static WORKING_FOLDER: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Start new conversations in the project of `folder`.
pub fn set_working_folder(folder: String) {
    if let Ok(mut here) = WORKING_FOLDER.lock() {
        *here = Some(folder);
    }
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

/// The ids of commands an [`Offline`] root queued, newest last, with their
/// `type`: tests answer them as the daemon would.
static SENT: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());
static SETTINGS_SENT: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());
static SETUP_SENT: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());

/// Every setup step an [`Offline`] root queued, as its request id and
/// JSON, oldest first.
pub fn setup_sent() -> Vec<(String, String)> {
    SETUP_SENT.lock().map(|s| s.clone()).unwrap_or_default()
}

/// Every settings change an [`Offline`] root queued, as JSON, oldest first.
pub fn settings_sent() -> Vec<String> {
    settings_sent_ids()
        .into_iter()
        .map(|(_, json)| json)
        .collect()
}

/// [`settings_sent`] with each change's request id.
pub fn settings_sent_ids() -> Vec<(String, String)> {
    SETTINGS_SENT.lock().map(|s| s.clone()).unwrap_or_default()
}

/// The id of the last command of `kind` (its `type`, as `get_changes`) an
/// [`Offline`] root queued.
pub fn sent(kind: &str) -> Option<String> {
    SENT.lock()
        .ok()?
        .iter()
        .rev()
        .find(|(_, k)| k == kind)
        .map(|(id, _)| id.clone())
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
        for kind in [
            "get_changes",
            "get_settings",
            "list_models",
            "set_session_model",
            "edit_queued",
            "archive_session",
            "revert_file",
            "set_session_project",
            "delete_project",
        ] {
            if let Some(id) = store.queued_id(kind)
                && let Ok(mut sent) = SENT.lock()
                && !sent.iter().any(|(i, _)| *i == id)
            {
                sent.push((id, kind.to_owned()));
            }
        }
        for (log, kind) in [(&SETTINGS_SENT, "change_settings"), (&SETUP_SENT, "setup")] {
            if let Ok(mut sent) = log.lock() {
                for (id, json) in store.queued_of(kind) {
                    if !sent.iter().any(|(i, _)| *i == id) {
                        sent.push((id, json));
                    }
                }
            }
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
    if let Some(folder) = WORKING_FOLDER.lock().ok().and_then(|mut f| f.take()) {
        store.set_here(folder);
    }
    store.restore();
    ui::face::load();
    link::start(store);
    view()
        .flex_1()
        .bg(ui::colors::BACKGROUND)
        .text_color(ui::colors::TEXT)
        .child(shell::shell(store))
}
