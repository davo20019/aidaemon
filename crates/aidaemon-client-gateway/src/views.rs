//! The agent-driven view: which client connection may be moved, and the one
//! way to move it.
//!
//! A socket that declares `Capability::ViewActions` in `hello` is given a
//! private outbox. When it sends a message into an app session, that session
//! names it as its view target; a message from anywhere else (an HTTP call,
//! a socket without the capability) clears the target, and so does the
//! socket closing. [`ViewTargets::show`] puts one ephemeral `view_action`
//! event in the target's outbox and nowhere else: it is not logged, not
//! broadcast and never replayed.
//!
//! The hard limit is structural. The only thing this module can produce is
//! an [`Event::ViewAction`] frame, a [`ViewAction`] names only a place on
//! screen, and nothing here holds the approvals, the turn runner or the
//! pairing store, so no view action can answer an approval, pair or revoke
//! a device, send a message or stop work.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aidaemon_client_protocol::{Event, ServerFrame, ViewAction};
use tokio::sync::mpsc;

use crate::bus::now_ms;

/// How many view actions may wait for one slow socket.
const OUTBOX: usize = 16;

/// Why a view action was not delivered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewError {
    /// No connection that can show view actions sent this session's turn
    /// (or it has closed).
    NoClient,
    /// The connection is not reading; the action was dropped.
    Busy,
}

impl std::fmt::Display for ViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoClient => f.write_str(
                "no app connection that can be moved is attached to this session's turn",
            ),
            Self::Busy => f.write_str("the app connection is not reading right now"),
        }
    }
}

impl std::error::Error for ViewError {}

#[derive(Default)]
struct Inner {
    outboxes: HashMap<u64, mpsc::Sender<ServerFrame>>,
    /// Session id to the connection whose message started its turn.
    targets: HashMap<String, u64>,
}

/// The view-capable connections and the session each one drives.
#[derive(Default)]
pub struct ViewTargets {
    next: AtomicU64,
    inner: Mutex<Inner>,
}

/// A registered connection; dropping it forgets the connection and every
/// session it was the target of.
pub struct ViewConnection {
    pub id: u64,
    pub outbox: mpsc::Receiver<ServerFrame>,
    targets: Arc<ViewTargets>,
}

impl Drop for ViewConnection {
    fn drop(&mut self) {
        let mut inner = self.targets.lock();
        inner.outboxes.remove(&self.id);
        let id = self.id;
        inner.targets.retain(|_, target| *target != id);
    }
}

impl ViewTargets {
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Register a socket that declared the capability.
    pub fn connect(self: &Arc<Self>) -> ViewConnection {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let (tx, rx) = mpsc::channel(OUTBOX);
        self.lock().outboxes.insert(id, tx);
        ViewConnection {
            id,
            outbox: rx,
            targets: self.clone(),
        }
    }

    /// A message for `session_id` came from `connection` (`None`: from a
    /// client that cannot be moved): it, and only it, is the target now.
    pub fn message_from(&self, session_id: &str, connection: Option<u64>) {
        let mut inner = self.lock();
        match connection.filter(|id| inner.outboxes.contains_key(id)) {
            Some(id) => {
                inner.targets.insert(session_id.to_string(), id);
            }
            None => {
                inner.targets.remove(session_id);
            }
        }
    }

    /// Whether a turn of `session_id` has a connection it may move.
    pub fn can_show(&self, session_id: &str) -> bool {
        let inner = self.lock();
        inner
            .targets
            .get(session_id)
            .is_some_and(|id| inner.outboxes.contains_key(id))
    }

    /// Ask the connection behind `session_id`'s turn to move its view.
    pub fn show(
        &self,
        session_id: &str,
        action: ViewAction,
        reason: &str,
    ) -> Result<(), ViewError> {
        let outbox = {
            let inner = self.lock();
            let id = inner.targets.get(session_id).ok_or(ViewError::NoClient)?;
            inner.outboxes.get(id).cloned().ok_or(ViewError::NoClient)?
        };
        let frame = ServerFrame::Event {
            seq: None,
            at_ms: now_ms(),
            session_id: Some(session_id.to_string()),
            event: Event::ViewAction {
                action,
                reason: reason.trim().chars().take(200).collect(),
            },
        };
        outbox.try_send(frame).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => ViewError::Busy,
            mpsc::error::TrySendError::Closed(_) => ViewError::NoClient,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_sender_with_the_capability_is_the_target() {
        let views = Arc::new(ViewTargets::default());
        let mut tui = views.connect();
        let web = views.connect();
        assert!(!views.can_show("app:s1"));
        views.message_from("app:s1", Some(tui.id));
        assert!(views.can_show("app:s1"));
        views
            .show("app:s1", ViewAction::NewSession, "asked")
            .unwrap();
        assert!(tui.outbox.try_recv().is_ok());
        // A message from a client that cannot be moved clears the target.
        views.message_from("app:s1", None);
        assert_eq!(
            views.show("app:s1", ViewAction::NewSession, "asked"),
            Err(ViewError::NoClient)
        );
        // The target closing clears it too.
        views.message_from("app:s1", Some(web.id));
        drop(web);
        assert!(!views.can_show("app:s1"));
        assert!(tui.outbox.try_recv().is_err());
    }
}
