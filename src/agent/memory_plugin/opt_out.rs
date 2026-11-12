//! "Don't remember this": a turn opted out of learning.
//!
//! A turn is opted out in one of two typed ways, never by matching words:
//! - the user sends `/private` first: the chat's next message is opted out
//!   when its turn starts ([`take_next_message_private`]);
//! - the model, reading that the user does not want the turn remembered,
//!   calls [`SKIP_LEARNING_TOOL`] during the turn.
//!
//! Either way the turn's `aidaemon_policies::TurnLearning`, which every
//! memory system receives on its `MemoryTurn`, reads `OptedOut`, and the
//! turn's events are marked like a turn the memory policy suppressed (a
//! `memory_policy_compiled` event with access `suppressed`), so background
//! work that later passes over them (consolidation, episodes, fact
//! consolidation, history intake, summaries) leaves them out, and exact
//! history search never returns them, so once the conversation moves on
//! (`/clear`, a new session) nothing brings the turn back. Recall still
//! works on the turn: opting out is about learning, not reading.

use std::sync::Arc;

use aidaemon_core as sdk;
use aidaemon_policies::TurnLearning;
use async_trait::async_trait;

use crate::events::{EventEmitter, EventType, MemoryPipelineAccess, MemoryPolicyCompiledData};

/// The host's opt-out tool, offered under any memory system.
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use aidaemon_policies::SKIP_LEARNING_TOOL;

/// The policy event's reason for an opted-out turn.
pub(crate) const OPTED_OUT_REASON: &str = "learning_opted_out";

/// Mark the turn's events so no automatic memory pipeline learns from them.
pub(crate) async fn mark_turn_opted_out(
    emitter: &EventEmitter,
    task_id: &str,
    turn_id: &str,
) -> anyhow::Result<()> {
    emitter
        .emit(
            EventType::MemoryPolicyCompiled,
            MemoryPolicyCompiledData {
                task_id: task_id.to_string(),
                turn_id: Some(turn_id.to_string()),
                access: MemoryPipelineAccess::Suppressed,
                reason_code: OPTED_OUT_REASON.to_string(),
                retrieval_suppressed: false,
                persistence_suppressed: true,
            },
        )
        .await?;
    Ok(())
}

/// What the model is told when an opted-out turn tries to store something.
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use aidaemon_policies::PRIVATE_TURN_STORE_REFUSAL;

/// Refuses memory stores on a turn opted out of learning, for the daemon's
/// memory tools and a replacing memory system's alike: a call is judged by
/// its tool's [`MemoryToolFacts`](super::tool_facts::MemoryToolFacts)
/// (`stores`), never by its name. Reads still run.
pub(crate) struct PrivateTurnGuard;

impl PrivateTurnGuard {
    #[allow(clippy::new_ret_no_self)]
    pub(crate) fn new(learning: TurnLearning) -> aidaemon_policies::PrivateTurnGuard {
        aidaemon_policies::PrivateTurnGuard::new(
            learning,
            Arc::new(|call: &sdk::ToolCall| {
                super::tool_facts::memory_tool_facts(&call.name)
                    .is_some_and(|facts| (facts.stores)(&call.arguments.to_string()))
            }),
        )
    }
}

/// The model's way to opt the current turn out of learning: it marks the
/// turn's events like a turn the memory policy suppressed.
pub(crate) struct SkipLearningTool;

impl SkipLearningTool {
    #[allow(clippy::new_ret_no_self)]
    pub(crate) fn new(
        learning: TurnLearning,
        emitter: EventEmitter,
        task_id: String,
        turn_id: String,
    ) -> aidaemon_policies::SkipLearningTool {
        aidaemon_policies::SkipLearningTool::new(
            learning,
            Arc::new(TurnOptOut {
                emitter,
                task_id,
                turn_id,
            }),
        )
    }
}

/// Records a turn's opt-out in its events.
struct TurnOptOut {
    emitter: EventEmitter,
    task_id: String,
    turn_id: String,
}

#[async_trait]
impl aidaemon_policies::OptOutRecorder for TurnOptOut {
    async fn record_opt_out(&self) -> Result<(), String> {
        mark_turn_opted_out(&self.emitter, &self.task_id, &self.turn_id)
            .await
            .map_err(|error| error.to_string())
    }
}

fn private_key(session_id: &str) -> String {
    format!("learning_opt_out:{session_id}")
}

/// `/private`: opt the chat's next message out of learning (`on`), or cancel.
pub(crate) async fn set_next_message_private(
    state: &(impl crate::traits::SettingsStore + ?Sized),
    session_id: &str,
    on: bool,
) -> anyhow::Result<()> {
    state
        .set_setting(&private_key(session_id), if on { "next" } else { "" })
        .await
}

/// Whether `/private` opted this message out, consuming the request. A read
/// error opts the message out: a request not to remember must not be lost to
/// a storage fault.
pub(crate) async fn take_next_message_private(
    state: &(impl crate::traits::SettingsStore + ?Sized),
    session_id: &str,
) -> bool {
    let key = private_key(session_id);
    match state.get_setting(&key).await {
        Ok(Some(value)) if !value.is_empty() => {
            if let Err(error) = state.set_setting(&key, "").await {
                tracing::warn!(session_id, %error, "Could not clear the /private request");
            }
            true
        }
        Ok(_) => false,
        Err(error) => {
            tracing::warn!(session_id, %error, "Could not read the /private request; not learning from this message");
            true
        }
    }
}
