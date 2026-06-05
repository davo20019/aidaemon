//! `ApprovalBroker` — a thin, named wrapper around the raw approval channel
//! sender (`mpsc::Sender<ApprovalRequest>`).
//!
//! The broker is created once at startup from the channel's transmit half and
//! handed to every tool/struct that needs to request command approval, replacing
//! the previously hand-threaded raw `mpsc::Sender<ApprovalRequest>`. It is a
//! behavior-preserving wrapper: cloning, send semantics, and channel capacity are
//! unchanged — `send` simply delegates to the inner sender.

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::SendError;

use super::terminal::ApprovalRequest;

/// Named wrapper around the approval channel's transmit half.
///
/// Cloning is cheap (clones the inner `mpsc::Sender`) and yields a handle to the
/// same underlying channel, matching the prior raw-sender behavior.
#[derive(Clone)]
pub struct ApprovalBroker {
    tx: mpsc::Sender<ApprovalRequest>,
}

impl ApprovalBroker {
    /// Wrap an existing approval channel sender.
    pub fn new(tx: mpsc::Sender<ApprovalRequest>) -> Self {
        Self { tx }
    }

    /// Send an approval request to the channel, delegating to the inner sender.
    ///
    /// Semantics are identical to `mpsc::Sender::send`: awaits available capacity
    /// and returns `Err(SendError)` if the receiver has been dropped.
    pub async fn send(&self, req: ApprovalRequest) -> Result<(), SendError<ApprovalRequest>> {
        self.tx.send(req).await
    }
}
