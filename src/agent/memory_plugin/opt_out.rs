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

use aidaemon_core as sdk;
use aidaemon_policies::TurnLearning;
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::events::{EventEmitter, EventType, MemoryPipelineAccess, MemoryPolicyCompiledData};

/// The host's opt-out tool, offered under any memory system.
pub(crate) const SKIP_LEARNING_TOOL: &str = "skip_learning";

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
pub(crate) const PRIVATE_TURN_STORE_REFUSAL: &str = "[PRIVATE] This message is private, so \
nothing from it is stored in memory. Do not try another way to store it; tell the user plainly \
that you won't remember it.";

/// Refuses memory stores on a turn opted out of learning, for the daemon's
/// memory tools and a replacing memory system's alike: a call is judged by
/// its tool's [`MemoryToolFacts`](super::tool_facts::MemoryToolFacts)
/// (`stores`), never by its name. Reads still run. The opt-out is read at
/// call time, so a `skip_learning` earlier in the turn covers later calls.
pub(crate) struct PrivateTurnGuard {
    learning: TurnLearning,
}

impl PrivateTurnGuard {
    pub(crate) fn new(learning: TurnLearning) -> Self {
        Self { learning }
    }
}

#[async_trait]
impl sdk::Hook for PrivateTurnGuard {
    async fn before_tool(
        &self,
        _ctx: &sdk::HookContext<'_>,
        call: &sdk::ToolCall,
        _tool: &sdk::ToolInfo,
    ) -> sdk::ToolDecision {
        if self.learning.allowed() {
            return sdk::ToolDecision::Allow;
        }
        let stores = super::tool_facts::memory_tool_facts(&call.name)
            .is_some_and(|facts| (facts.stores)(&call.arguments.to_string()));
        if stores {
            sdk::ToolDecision::Deny {
                reason: PRIVATE_TURN_STORE_REFUSAL.to_string(),
            }
        } else {
            sdk::ToolDecision::Allow
        }
    }
}

/// The model's way to opt the current turn out of learning.
pub(crate) struct SkipLearningTool {
    learning: TurnLearning,
    emitter: EventEmitter,
    task_id: String,
    turn_id: String,
}

impl SkipLearningTool {
    pub(crate) fn new(
        learning: TurnLearning,
        emitter: EventEmitter,
        task_id: String,
        turn_id: String,
    ) -> Self {
        Self {
            learning,
            emitter,
            task_id,
            turn_id,
        }
    }
}

#[async_trait]
impl sdk::Tool for SkipLearningTool {
    fn spec(&self) -> sdk::ToolSpec {
        sdk::ToolSpec {
            name: SKIP_LEARNING_TOOL.into(),
            description: "Keep the current message out of long-term memory. Call it when the \
                          user does not want what they say now remembered or learned from. \
                          Nothing from this turn is then learned, now or later; memory can \
                          still be used to answer. It applies to this turn only and cannot be \
                          undone."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> sdk::ToolAnnotations {
        sdk::ToolAnnotations::READ_ONLY
    }

    fn source_trust(&self, _arguments: &Value) -> sdk::SourceTrust {
        sdk::SourceTrust::Trusted
    }

    async fn call(&self, _arguments: Value, _ctx: &sdk::ToolContext) -> sdk::ToolResult {
        if self.learning.opt_out() {
            mark_turn_opted_out(&self.emitter, &self.task_id, &self.turn_id)
                .await
                .map_err(|error| {
                    sdk::ToolError(format!("Could not record the opt-out: {error}"))
                })?;
        }
        Ok(sdk::ToolOutput::text(
            "This turn will not be remembered or learned from.",
        ))
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
