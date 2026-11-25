//! The learning opt-out as the turn applies it: the record and the model's
//! tool live in [`crate::opt_out`]; the guard that refuses memory stores on
//! an opted-out turn judges calls by the turn's memory tool facts.

use std::sync::Arc;

use aidaemon_core as sdk;
use aidaemon_policies::TurnLearning;

pub use crate::opt_out::*;
pub use crate::turn::task_planning::{set_next_message_private, take_next_message_private};

/// Refuses memory stores on a turn opted out of learning, for the daemon's
/// memory tools and a replacing memory system's alike: a call is judged by
/// its tool's [`MemoryToolFacts`](crate::turn::tool_facts::MemoryToolFacts)
/// (`stores`), never by its name. Reads still run.
pub struct PrivateTurnGuard;

impl PrivateTurnGuard {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(learning: TurnLearning) -> aidaemon_policies::PrivateTurnGuard {
        aidaemon_policies::PrivateTurnGuard::new(
            learning,
            Arc::new(|call: &sdk::ToolCall| {
                crate::turn::tool_facts::memory_tool_facts(&call.name)
                    .is_some_and(|facts| (facts.stores)(&call.arguments.to_string()))
            }),
        )
    }
}
