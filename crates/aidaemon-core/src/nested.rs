//! Calls a running tool makes on its own behalf, judged by the run's
//! safeguards before it makes them.
//!
//! A tool that drives another program (a coding agent the run delegated
//! to) can be asked by that program for permission to act: run a command,
//! edit a file, reach the network. The tool must not answer on its own, and
//! must never ask the owner itself. It hands the action to the run as a
//! call of one of the run's own tools ([`NestedCalls::authorize`]), and the
//! run judges it exactly as it judges the model's calls: an effect the
//! owner already declined is refused first, then every hook decides, and a
//! hook that needs an answer is settled by the run's
//! [`crate::InterruptResolver`] while the tool waits. An owner's decline
//! binds the rest of the run, and every run that shares its declines.
//!
//! The call is only judged here; it is never executed by the run. Whatever
//! the program then does is the tool's own work, reported in its result.

use std::fmt;

use tokio::sync::{mpsc, oneshot};

use crate::message::ToolCall;

/// The run's answer about a nested call.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NestedVerdict {
    /// Every safeguard allowed it, or the owner approved it.
    Allow,
    /// A safeguard refused it.
    Deny { reason: String },
    /// The owner declined it (the reason starts with
    /// [`crate::OWNER_DECLINED`]); remembered for the rest of the run.
    Declined { reason: String },
    /// It needed the owner, and no answer came (or nobody can be asked in
    /// this run). Nothing is remembered as declined.
    Unanswered { reason: String },
}

impl NestedVerdict {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// Why it was not allowed; `None` when it was.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Allow => None,
            Self::Deny { reason } | Self::Declined { reason } | Self::Unanswered { reason } => {
                Some(reason)
            }
        }
    }
}

/// One call waiting to be judged.
pub(crate) struct NestedAsk {
    pub call: ToolCall,
    pub reply: oneshot::Sender<NestedVerdict>,
}

/// The way a running tool has a call judged by its run (see the module
/// docs). Present in [`crate::ToolContext::nested`] while the run drives
/// the tool's call.
#[derive(Clone)]
pub struct NestedCalls {
    tx: mpsc::UnboundedSender<NestedAsk>,
}

/// The reason a call gets when its run is no longer there to judge it.
const RUN_GONE: &str = "the run ended before this call could be checked";

impl NestedCalls {
    pub(crate) fn new(tx: mpsc::UnboundedSender<NestedAsk>) -> Self {
        Self { tx }
    }

    /// Judge `call` (a call of one of the run's own tools, describing what
    /// the driven program asks to do) and wait for the answer, which may
    /// be the owner's.
    pub async fn authorize(&self, call: ToolCall) -> NestedVerdict {
        let (reply, answer) = oneshot::channel();
        if self.tx.send(NestedAsk { call, reply }).is_err() {
            return NestedVerdict::Deny {
                reason: RUN_GONE.to_string(),
            };
        }
        answer.await.unwrap_or_else(|_| NestedVerdict::Deny {
            reason: RUN_GONE.to_string(),
        })
    }
}

impl fmt::Debug for NestedCalls {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NestedCalls").finish_non_exhaustive()
    }
}
