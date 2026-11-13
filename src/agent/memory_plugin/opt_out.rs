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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use aidaemon_core as sdk;
use aidaemon_policies::{OptOutRecorder as _, TurnLearning};
use async_trait::async_trait;

use crate::events::{EventEmitter, EventType, MemoryPipelineAccess, MemoryPolicyCompiledData};

/// The host's opt-out tool, offered under any memory system.
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use aidaemon_policies::SKIP_LEARNING_TOOL;

/// The policy event's reason for an opted-out turn.
pub(crate) use aidaemon_runtime::turn::task_planning::OPTED_OUT_REASON;

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
    ) -> RetryingSkipLearning {
        Self::with_recorder(
            learning,
            Arc::new(TurnOptOut {
                emitter,
                task_id,
                turn_id,
            }),
        )
    }

    #[allow(clippy::new_ret_no_self)]
    pub(crate) fn with_recorder(
        learning: TurnLearning,
        recorder: Arc<dyn aidaemon_policies::OptOutRecorder>,
    ) -> RetryingSkipLearning {
        let mark = Arc::new(MarkUntilRecorded {
            recorder,
            recorded: AtomicBool::new(false),
        });
        RetryingSkipLearning {
            inner: aidaemon_policies::SkipLearningTool::new(learning, mark.clone()),
            mark,
        }
    }
}

/// The opt-out tool whose mark is retried until it is recorded. The turn's
/// flag flips once, and the policies tool records only on that flip, so a
/// mark that failed then would leave the turn's events unmarked for good;
/// whether the mark landed is kept apart from the flag, and a later call
/// marks again until it has.
pub(crate) struct RetryingSkipLearning {
    inner: aidaemon_policies::SkipLearningTool,
    mark: Arc<MarkUntilRecorded>,
}

#[async_trait]
impl sdk::Tool for RetryingSkipLearning {
    fn spec(&self) -> sdk::ToolSpec {
        self.inner.spec()
    }

    fn annotations(&self) -> sdk::ToolAnnotations {
        self.inner.annotations()
    }

    fn source_trust(&self, arguments: &serde_json::Value) -> sdk::SourceTrust {
        self.inner.source_trust(arguments)
    }

    async fn call(&self, arguments: serde_json::Value, ctx: &sdk::ToolContext) -> sdk::ToolResult {
        let output = self.inner.call(arguments, ctx).await?;
        self.mark
            .record_opt_out()
            .await
            .map_err(|error| sdk::ToolError(format!("Could not record the opt-out: {error}")))?;
        Ok(output)
    }
}

/// Records the opt-out once it succeeds; until then every request retries.
struct MarkUntilRecorded {
    recorder: Arc<dyn aidaemon_policies::OptOutRecorder>,
    recorded: AtomicBool,
}

#[async_trait]
impl aidaemon_policies::OptOutRecorder for MarkUntilRecorded {
    async fn record_opt_out(&self) -> Result<(), String> {
        if self.recorded.load(Ordering::Acquire) {
            return Ok(());
        }
        self.recorder.record_opt_out().await?;
        self.recorded.store(true, Ordering::Release);
        Ok(())
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

pub(crate) use aidaemon_runtime::turn::task_planning::{
    set_next_message_private, take_next_message_private,
};
