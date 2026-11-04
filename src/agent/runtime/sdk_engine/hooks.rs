//! The kernel-side adapters: hooks and the bridged daemon tools. Each one
//! only forwards to the daemon over the turn's `DaemonCall` channel.

use super::*;

/// The ledger closeout as an SDK stop hook: before a final answer ships, the
/// run ledger decides whether a still-reachable expectation needs another
/// pass. Receipts, not the reply text, are the evidence.
pub(super) struct LedgerStop {
    pub(super) calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl sdk::Hook for LedgerStop {
    async fn on_stop(&self, ctx: &sdk::HookContext<'_>, final_text: &str) -> sdk::StopDecision {
        let completed_results = ctx
            .run_receipts()
            .filter(|receipt| receipt.status == sdk::ReceiptStatus::Succeeded)
            .count();
        let failed_without_success = completed_results == 0
            && ctx
                .run_receipts()
                .last()
                .is_some_and(|receipt| receipt.status == sdk::ReceiptStatus::Failed);
        let (reply, answer) = oneshot::channel();
        if self
            .calls
            .send(DaemonCall::Stop(StopCheck {
                completed_results,
                failed_without_success,
                final_text: final_text.to_string(),
                reply,
            }))
            .await
            .is_err()
        {
            return sdk::StopDecision::Accept;
        }
        answer.await.unwrap_or(sdk::StopDecision::Accept)
    }
}

/// Whether `tool` with `arguments`, after the run's earlier `previous` calls,
/// continues a same-tool loop: 8 consecutive calls
/// (`MAX_CONSECUTIVE_SAME_TOOL`), 16 for versatile and
/// research tools, 40 for `computer_use`; diverse arguments get the higher
/// limit as a bonus, not a bypass.
/// `read_only` is the tool's typed annotation: a read-only streak in which
/// every call has different arguments (paging through files, distinct
/// lookups) observes something new each time, so it is progress, not a
/// loop; budgets, the turn cap and the stuck detector bound it.
pub(super) fn same_tool_loop(
    previous: &[(String, Value)],
    tool: &str,
    arguments: &Value,
    read_only: bool,
) -> bool {
    let mut streak: Vec<&Value> = previous
        .iter()
        .rev()
        .take_while(|(name, _)| name == tool)
        .map(|(_, args)| args)
        .collect();
    streak.push(arguments);
    let total = streak.len();
    let computer_use = tool == "computer_use";
    let versatile = matches!(
        tool,
        "read_file"
            | "search_files"
            | "manage_goal_tasks"
            | "check_environment"
            | "web_search"
            | "web_fetch"
            | "terminal"
    );
    let limit = if computer_use {
        40
    } else if versatile {
        crate::agent::MAX_CONSECUTIVE_SAME_TOOL + 8
    } else {
        crate::agent::MAX_CONSECUTIVE_SAME_TOOL
    };
    if total < limit {
        return false;
    }
    let mut unique: Vec<&Value> = Vec::new();
    for args in streak {
        if !unique.contains(&args) {
            unique.push(args);
        }
    }
    if read_only && tool != "computer_use" && unique.len() == total {
        return false;
    }
    let diverse = unique.len() * 2 > total;
    let diverse_limit = if computer_use {
        40
    } else {
        crate::agent::MAX_CONSECUTIVE_SAME_TOOL + 8
    };
    !diverse || total >= diverse_limit
}

/// The consecutive-same-tool guard. A hook cannot end the
/// run, so the looping call is refused with a redirect; if the model keeps
/// at it, the kernel's stuck detector ends the run.
pub(super) struct SameToolLimit;

#[async_trait]
impl sdk::Hook for SameToolLimit {
    async fn before_tool(
        &self,
        ctx: &sdk::HookContext<'_>,
        call: &sdk::ToolCall,
        tool: &sdk::ToolInfo,
    ) -> sdk::ToolDecision {
        let previous: Vec<(String, Value)> = ctx
            .run_receipts()
            .map(|receipt| (receipt.tool.clone(), receipt.arguments.clone()))
            .collect();
        if !same_tool_loop(
            &previous,
            &call.name,
            &call.arguments,
            tool.annotations.read_only,
        ) {
            return sdk::ToolDecision::Allow;
        }
        warn!(tool = %call.name, "SDK engine: same tool called too many consecutive times");
        sdk::ToolDecision::Deny {
            reason: format!(
                "[LOOP DETECTED] `{}` has been called too many times in a row without \
                 finishing. Stop calling it: use a different approach, or answer with what \
                 you have and the exact limitation.",
                call.name
            ),
        }
    }
}

/// Whether one exact call of `tool` is read-only: the tool is read-only as a
/// whole, or its adapter types the call as a pure observation (or run
/// bookkeeping, which changes no user state) with no mutation effects. The
/// tool still enforces `mutation_forbidden` itself.
pub(super) fn call_is_read_only(tool: &dyn crate::traits::Tool, arguments: &str) -> bool {
    tool.capabilities().read_only || semantics_are_read_only(&tool.call_semantics(arguments))
}

/// The per-call half of [`call_is_read_only`], for callers that already hold
/// the call's semantics.
pub(super) fn semantics_are_read_only(semantics: &crate::traits::ToolCallSemantics) -> bool {
    matches!(
        semantics.effect,
        crate::traits::ToolCallEffect::Observation | crate::traits::ToolCallEffect::Administrative
    ) && !semantics.mutates_state()
}

/// A daemon tool as the SDK sees it. It only forwards the call; the daemon
/// dispatcher executes it on the turn's own task.
pub(super) struct BridgedTool {
    pub(super) spec: sdk::ToolSpec,
    pub(super) tool: Arc<dyn crate::traits::Tool>,
    /// `manage_memories` offered with only its goal actions.
    pub(super) goal_actions_only: bool,
    pub(super) calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl sdk::Tool for BridgedTool {
    fn spec(&self) -> sdk::ToolSpec {
        self.spec.clone()
    }

    /// Worst case for the tool as a whole: only a tool that is read-only
    /// for every call says so.
    fn annotations(&self) -> sdk::ToolAnnotations {
        if self.tool.capabilities().read_only {
            sdk::ToolAnnotations::READ_ONLY
        } else {
            sdk::ToolAnnotations::WRITE
        }
    }

    /// Judged per call: a mixed tool such as `terminal` reads for `ls` and
    /// writes for `rm`.
    fn annotations_for(&self, arguments: &Value) -> sdk::ToolAnnotations {
        if call_is_read_only(self.tool.as_ref(), &arguments.to_string()) {
            sdk::ToolAnnotations::READ_ONLY
        } else {
            sdk::ToolAnnotations::WRITE
        }
    }

    /// Daemon tools check their own arguments (some accept aliases for
    /// schema-required keys) and report bad ones through the dispatcher.
    /// Only the goal-actions view of `manage_memories` is narrowed here; the
    /// kernel records that refusal in the run ledger.
    fn validate(&self, arguments: &Value) -> Result<(), String> {
        if self.goal_actions_only {
            crate::agent::memory_plugin::check_goal_action(arguments)?;
        }
        Ok(())
    }

    fn timeout(&self) -> Option<Duration> {
        Some(TOOL_TIMEOUT)
    }

    async fn call(&self, arguments: Value, ctx: &sdk::ToolContext) -> sdk::ToolResult {
        let (reply, answer) = oneshot::channel();
        self.calls
            .send(DaemonCall::Tool(ToolRequest {
                name: self.spec.name.clone(),
                arguments,
                call_id: ctx.call_id.clone(),
                reply,
            }))
            .await
            .map_err(|_| sdk::ToolError("tool dispatcher stopped".to_string()))?;
        answer
            .await
            .map_err(|_| sdk::ToolError("tool dispatcher dropped the call".to_string()))?
    }
}

/// Build an SDK tool spec from a daemon tool's OpenAI-style function schema.
pub(super) fn tool_spec(tool: &dyn crate::traits::Tool) -> sdk::ToolSpec {
    tool_spec_from_schema(tool, &tool.schema())
}

pub(super) fn tool_spec_from_schema(
    tool: &dyn crate::traits::Tool,
    schema: &Value,
) -> sdk::ToolSpec {
    let parameters = schema
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
    let description = schema
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_else(|| tool.description())
        .to_string();
    sdk::ToolSpec {
        name: tool.name().to_string(),
        description,
        parameters,
    }
}
