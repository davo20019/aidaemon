//! The turn engine: one turn on the aidaemon SDK kernel, on any
//! [`TurnHost`].
//!
//! [`TurnEngine`] is a view of a host: it derefs to `dyn TurnHost`, so the
//! engine reads the host's typed surface as its own, and it passes itself
//! wherever a `&dyn TurnHost` is wanted (`turn::task_end`, the side-call
//! gateway). It owns no state; a turn's state lives in its
//! [`TurnScope`](crate::engine::scope::TurnScope).
//!
//! Layout: `model_call` serves the kernel's provider requests,
//! `compaction` compacts a run that outgrows its window, `outcome` turns a
//! finished run into the checklist's final shape, and `events` records the
//! turn's task end and decision points.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use aidaemon_core as sdk;
use serde_json::Value;
use tracing::{info, warn};

use aidaemon_events::conversation::Message;
use aidaemon_events::{EventEmitter, EventType, TaskOutcome};
use uuid::Uuid;

use crate::engine::budgets::*;
use crate::engine::calls::{ModelCall, ModelErrorCheck};
use crate::engine::provider::{
    fit_run_values, from_provider_response, prefix_with_notes, split_injected_notes,
    to_chat_options, to_daemon_messages_from, to_daemon_tools,
};
use crate::engine::scope::{AbsorbedSteering, TurnScope};
use crate::host::TurnHost;
use crate::turn::turn_context::TurnHistory;

pub mod compaction;
mod events;
pub mod model_call;
pub mod outcome;

/// The turn engine over a host. See the module docs.
#[derive(Clone, Copy)]
pub struct TurnEngine<'h> {
    host: &'h dyn TurnHost,
}

impl<'h> TurnEngine<'h> {
    pub fn new(host: &'h dyn TurnHost) -> Self {
        Self { host }
    }

    /// The host this engine runs on.
    pub fn host(&self) -> &'h dyn TurnHost {
        self.host
    }
}

impl<'h> std::ops::Deref for TurnEngine<'h> {
    type Target = dyn TurnHost + 'h;

    fn deref(&self) -> &Self::Target {
        self.host
    }
}
