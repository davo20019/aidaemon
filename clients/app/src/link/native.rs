//! The desktops, the terminal and the phones: the shared tokio link on
//! nsh's background threads. Where to sign in, pairing and starting the
//! daemon here are `aidaemon_client_core::local`'s.

use aidaemon_client_core::link::{self, LinkConfig};
use aidaemon_client_core::local;
use aidaemon_client_core::machine::{ClientIdentity, Update};
use aidaemon_client_protocol::{Capability, ClientKind};
use nsh_core::platform::Idiom;
use nsh_core::prelude::*;

use crate::store::Store;

/// Which link is current: [`pair_again`] moves it on, and an older link's
/// task stops at its next update (dropping its receiver ends the link).
static LINK_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The current link's waker ([`woke`]).
static WAKER: std::sync::Mutex<Option<link::Waker>> = std::sync::Mutex::new(None);

/// Tell the current link the device woke.
pub fn woke() {
    if let Some(waker) = WAKER.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
        waker.wake();
    }
}

/// What kind of client this is, for the daemon's device list and limits.
fn client_kind() -> ClientKind {
    if super::MOBILE {
        ClientKind::Mobile
    } else if idiom() == Idiom::Terminal {
        ClientKind::Tui
    } else {
        ClientKind::Desktop
    }
}

pub fn start(store: Store) {
    use std::sync::atomic::Ordering;
    // A phone has no home folder: its key file (Android) lives in the app's
    // own data directory.
    if super::MOBILE
        && let Some(dir) = data_dir()
    {
        local::set_data_dir(dir);
    }
    let generation = LINK_GENERATION.load(Ordering::SeqCst);
    let Some((url, auth)) = local::sign_in() else {
        // Nothing to sign in to yet: where should the agent run?
        store.needs_daemon();
        return;
    };
    store.set_gateway_base(aidaemon_client_core::machine::http_base(&url));
    let bridged = (url.clone(), auth.clone());
    let kind = client_kind();
    let config = LinkConfig {
        url,
        auth,
        client: ClientIdentity {
            kind,
            name: "aidaemon-app".into(),
            version: Some(crate::link::app_version()),
        },
        device: None,
        capabilities: vec![
            Capability::ViewActions,
            Capability::Surfaces,
            Capability::Shells,
        ],
    };
    let (updates_tx, mut updates) = tokio::sync::mpsc::unbounded_channel::<Update>();
    // Owned by nothing: `start` runs from screens' tasks too (pairing, "Pair
    // again", installing the daemon), and the link must outlive the screen
    // that started it. A screen's tasks go with it: the link started from
    // the pairing screen stopped as that screen gave way, and a phone stayed
    // "connecting" until the app was opened again (live 2026-11-19).
    Owner::detached().run(|| {
        spawn(async move {
            // What the app loads by URL (pictures, files, live views,
            // shells): a daemon reached over iroh is reached through the
            // core's local bridge. Starting it opens no connection.
            let (url, auth) = bridged;
            match background(async move { local::reachable_base(&url, &auth).await }).await {
                Ok(Ok(base)) => store.set_gateway_base(base),
                Ok(Err(why)) => log::warn!("files and live views are unavailable: {why}"),
                Err(_) => log::warn!("could not start the local bridge"),
            }
            // The link's tasks run on the runtime behind `background`.
            let Ok((commands, waker)) =
                background(async move { link::spawn_wakeable(config, updates_tx) }).await
            else {
                log::error!("could not start the link");
                return;
            };
            *WAKER.lock().unwrap_or_else(|p| p.into_inner()) = Some(waker);
            store.attach(Box::new(move |id, command| {
                let _ = commands.send((id, command));
            }));
            while let Some(update) = updates.recv().await {
                if LINK_GENERATION.load(Ordering::SeqCst) != generation {
                    return;
                }
                store.update(update);
            }
        })
    });
}

/// [`super::open_outside`] here: a desktop opens the link (the app, and
/// so the bridge, stays up); a phone keeps a copy, then hands it over.
pub fn open_outside(store: Store, url: String, name: String) {
    if !super::MOBILE {
        drop(open_url(url));
        return;
    }
    spawn(async move {
        let kept = background(async move { local::keep_copy(&url, &name).await }).await;
        let path = match kept {
            Ok(Ok(path)) => path,
            Ok(Err(why)) => return store.say(why),
            Err(_) => return store.say("Could not open the file"),
        };
        // iOS opens no local path for another app; its share sheet does
        // (Open in, Save to Files, Books).
        #[cfg(target_os = "ios")]
        if share(nsh_core::share::ShareRequest::new().file(path))
            .await
            .is_err()
        {
            store.say("Could not open the file");
        }
        #[cfg(not(target_os = "ios"))]
        if let Err(why) = open_path(path).await {
            store.say(format!("Could not open the file: {why}"));
        }
    });
}

/// Forget this device's pairing (the daemon refused its key) and show the
/// screen to pair again. Only a paired device: the owner's local token is
/// renewed by the daemon, not here.
pub fn pair_again(store: Store) {
    use std::sync::atomic::Ordering;
    LINK_GENERATION.fetch_add(1, Ordering::SeqCst);
    store.attach(Box::new(|_, _| {}));
    spawn(async move {
        let _ = background(async { local::unpair() }).await;
        store.needs_daemon();
    });
}

/// Whether this device signs in with a paired key (not the local token),
/// so a refusal can be answered by pairing again.
pub fn is_paired() -> bool {
    local::local_token().is_none() && local::is_paired()
}

/// Run the daemon on this computer: install it to start at login (no
/// administrator rights), wait for its token, then connect.
pub fn install_here(store: Store) {
    store.set_starting_daemon(Some(("Starting AIDaemon on this computer…".into(), false)));
    spawn(async move {
        let result = background(local::install_daemon())
            .await
            .unwrap_or_else(|_| Err("Could not start AIDaemon".to_string()));
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
    background(async move { local::pair_from_link(&link, &name, client_kind()).await })
        .await
        .map_err(|_| "Could not pair".to_string())??;
    store.update(Update::Connecting);
    start(store);
    Ok(())
}

/// Open a live view's frame socket on the background runtime; its events
/// come back to the engine thread and into the [`Store`].
pub fn open_live(store: Store, epoch: u64, url: String, ticket: String) {
    use aidaemon_client_core::surfaces::{self, MediaEvent};
    let (events_tx, mut events) = tokio::sync::mpsc::unbounded_channel::<MediaEvent>();
    let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    // Closing drops the only sender: the socket task closes and ends.
    let sender = std::rc::Rc::new(std::cell::RefCell::new(Some(out_tx)));
    let send = {
        let sender = sender.clone();
        move |text: String| {
            if let Some(tx) = sender.borrow().as_ref() {
                let _ = tx.send(text);
            }
        }
    };
    store.media_attached(
        epoch,
        crate::store::MediaLink {
            send: Box::new(send),
            close: Box::new(move || {
                sender.borrow_mut().take();
            }),
        },
    );
    spawn(async move {
        let started = background(async move {
            tokio::spawn(surfaces::run(url, ticket, events_tx, out_rx));
        })
        .await;
        if started.is_err() {
            store.media_event(
                epoch,
                MediaEvent::Closed("could not start the live view".into()),
            );
            return;
        }
        let mut closed = false;
        while let Some(event) = events.recv().await {
            closed |= matches!(event, MediaEvent::Closed(_));
            store.media_event(epoch, event);
        }
        if !closed {
            store.media_event(epoch, MediaEvent::Closed("the connection dropped".into()));
        }
    });
}

/// Open a conversation shell's socket on the background runtime; its
/// events come back to the engine thread and into the [`Store`].
pub fn open_shell(store: Store, epoch: u64, url: String, ticket: String) {
    use aidaemon_client_core::shells::{self, ShellEvent};
    let (events_tx, mut events) = tokio::sync::mpsc::unbounded_channel::<ShellEvent>();
    let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    // Closing drops the only sender: the socket task closes and ends.
    let sender = std::rc::Rc::new(std::cell::RefCell::new(Some(out_tx)));
    let send = {
        let sender = sender.clone();
        move |text: String| {
            if let Some(tx) = sender.borrow().as_ref() {
                let _ = tx.send(text);
            }
        }
    };
    store.shell_attached(
        epoch,
        crate::store::ShellLink {
            send: Box::new(send),
            close: Box::new(move || {
                sender.borrow_mut().take();
            }),
        },
    );
    spawn(async move {
        let started = background(async move {
            tokio::spawn(shells::run(url, ticket, events_tx, out_rx));
        })
        .await;
        if started.is_err() {
            store.shell_event(
                epoch,
                ShellEvent::Closed("could not start the shell".into()),
            );
            return;
        }
        let mut closed = false;
        while let Some(event) = events.recv().await {
            closed |= matches!(event, ShellEvent::Closed(_));
            store.shell_event(epoch, event);
        }
        if !closed {
            store.shell_event(epoch, ShellEvent::Closed("the connection dropped".into()));
        }
    });
}

/// Whether the owner can pair a phone from here: a computer signed in with
/// the local token (the daemon's own).
pub fn can_offer_pairing() -> bool {
    !super::MOBILE && local::local_token().is_some()
}

/// A new pairing offer, confirmed in this app.
pub async fn offer_pairing() -> Result<aidaemon_client_protocol::PairingOffer, String> {
    background(local::owner_pairing::offer())
        .await
        .map_err(|_| "Could not start pairing".to_string())?
}

/// Where the offer stands.
pub async fn pairing_status(
    offer_id: String,
) -> Result<aidaemon_client_protocol::Enrollment, String> {
    background(async move { local::owner_pairing::status(&offer_id).await })
        .await
        .map_err(|_| "Could not read the pairing".to_string())?
}

/// Approve or deny the device waiting on the offer.
pub async fn decide_pairing(
    offer_id: String,
    approve: bool,
) -> Result<aidaemon_client_protocol::Enrollment, String> {
    background(async move { local::owner_pairing::decide(&offer_id, approve).await })
        .await
        .map_err(|_| "Could not answer the pairing".to_string())?
}
