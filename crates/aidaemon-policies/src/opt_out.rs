//! "Don't remember this", as hooks and a tool any host can offer under any
//! memory system: [`PrivateTurnGuard`] refuses memory stores on a turn opted
//! out of learning, and [`SkipLearningTool`] lets the model opt the turn out.

use std::sync::Arc;

use aidaemon_core::{
    Hook, HookContext, SourceTrust, Tool, ToolAnnotations, ToolCall, ToolContext, ToolDecision,
    ToolError, ToolInfo, ToolOutput, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::memory::TurnLearning;

/// The opt-out tool's name.
pub const SKIP_LEARNING_TOOL: &str = "skip_learning";

/// What the model is told when an opted-out turn tries to store something.
/// It is worded as the reply the user should get, so the model passes it on
/// as it is instead of describing a part of the system that refused.
pub const PRIVATE_TURN_STORE_REFUSAL: &str = "Not saved: the user made this message \
private. Do not try another way to save it. Tell the user, in these words or close to them: \
This message is private, so I won't save it.";

/// Whether a tool call stores something in memory. The host answers from
/// what the call's tool declares, never from its name alone.
pub type StoresJudge = Arc<dyn Fn(&ToolCall) -> bool + Send + Sync>;

/// Refuses memory stores on a turn opted out of learning. Reads still run.
/// The opt-out is read at call time, so a [`SkipLearningTool`] call earlier
/// in the turn covers later calls.
pub struct PrivateTurnGuard {
    learning: TurnLearning,
    stores: StoresJudge,
}

impl PrivateTurnGuard {
    pub fn new(learning: TurnLearning, stores: StoresJudge) -> Self {
        Self { learning, stores }
    }
}

#[async_trait]
impl Hook for PrivateTurnGuard {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        _tool: &ToolInfo,
    ) -> ToolDecision {
        if self.learning.allowed() || !(self.stores)(call) {
            ToolDecision::Allow
        } else {
            ToolDecision::Deny {
                reason: PRIVATE_TURN_STORE_REFUSAL.to_string(),
            }
        }
    }
}

/// Where the host records that a turn was opted out, so background work that
/// later passes over the turn leaves it out.
#[async_trait]
pub trait OptOutRecorder: Send + Sync {
    async fn record_opt_out(&self) -> Result<(), String>;
}

/// The model's way to opt the current turn out of learning.
pub struct SkipLearningTool {
    learning: TurnLearning,
    recorder: Arc<dyn OptOutRecorder>,
}

impl SkipLearningTool {
    pub fn new(learning: TurnLearning, recorder: Arc<dyn OptOutRecorder>) -> Self {
        Self { learning, recorder }
    }
}

#[async_trait]
impl Tool for SkipLearningTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
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

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }

    fn source_trust(&self, _arguments: &Value) -> SourceTrust {
        SourceTrust::Trusted
    }

    async fn call(&self, _arguments: Value, _ctx: &ToolContext) -> ToolResult {
        if self.learning.opt_out() {
            self.recorder
                .record_opt_out()
                .await
                .map_err(|error| ToolError(format!("Could not record the opt-out: {error}")))?;
        }
        Ok(ToolOutput::text(
            "This turn will not be remembered or learned from.",
        ))
    }
}
