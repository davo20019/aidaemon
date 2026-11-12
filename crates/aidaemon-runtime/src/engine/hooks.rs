//! The kernel-side adapters of a turn: hooks and the bridged host tools.
//! Each one only forwards to the host over the turn's [`HostCall`] channel,
//! or adjusts the request from what the run has seen.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use aidaemon_core as sdk;
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tracing::warn;

use super::calls::{
    AfterCallCheck, BudgetCheck, GateRequest, HostCall, KernelCall, KernelNotice, ModelErrorCheck,
    StopCheck, ToolRequest,
};
use super::TOOL_TIMEOUT;

/// The ledger closeout as an SDK stop hook: before a final answer ships, the
/// run ledger decides whether a still-reachable expectation needs another
/// pass. Receipts, not the reply text, are the evidence.
pub struct LedgerStop {
    pub calls: mpsc::Sender<HostCall>,
}

#[async_trait]
impl sdk::Hook for LedgerStop {
    fn timeout(&self) -> Option<std::time::Duration> {
        Some(TOOL_TIMEOUT)
    }

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
            .send(HostCall::Stop(StopCheck {
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

/// The same-tool guard's limits: 8 consecutive calls
/// (`MAX_CONSECUTIVE_SAME_TOOL`), 16 for a `Versatile` tool, and 40 for a
/// `Stepwise` one (computer use). The rule itself is
/// `aidaemon_policies::SameToolStreakGuard`.
pub const SAME_TOOL_LIMITS: aidaemon_policies::StreakLimits = aidaemon_policies::StreakLimits {
    base: crate::turn::loop_utils::MAX_CONSECUTIVE_SAME_TOOL,
    versatile: crate::turn::loop_utils::MAX_CONSECUTIVE_SAME_TOOL + 8,
    stepwise: 40,
};

/// Each tool's consecutive-call allowance (`dispatch_facts().same_tool_streak`).
pub fn same_tool_streaks<'a>(
    tools: impl IntoIterator<Item = &'a Arc<dyn aidaemon_tools::OutcomeTool>>,
) -> std::collections::HashMap<String, aidaemon_tools::semantics::SameToolStreak> {
    tools
        .into_iter()
        .map(|tool| {
            (
                tool.name().to_string(),
                tool.dispatch_facts().same_tool_streak,
            )
        })
        .collect()
}

/// Names of the tools that are read-only for every call. The same-tool
/// guard's distinct-arguments exemption is judged by the tool's worst case
/// (`capabilities().read_only`), not the per-call annotation: a mixed tool
/// such as `terminal` types most calls read-only, and must stay limited.
pub fn read_only_tool_names<'a>(
    tools: impl IntoIterator<Item = &'a Arc<dyn aidaemon_tools::OutcomeTool>>,
) -> std::collections::HashSet<String> {
    tools
        .into_iter()
        .filter(|tool| tool.capabilities().read_only)
        .map(|tool| tool.name().to_string())
        .collect()
}

/// Withholds tools from the next request only, when a retry must answer in
/// text from the receipts it already holds.
pub struct TextOnlyNext {
    pub flag: Arc<AtomicBool>,
}

#[async_trait]
impl sdk::Hook for TextOnlyNext {
    async fn before_model(&self, _ctx: &sdk::HookContext<'_>, request: &mut sdk::ModelRequest) {
        if self.flag.swap(false, Ordering::Relaxed) {
            request.tool_choice = sdk::ToolChoice::None;
        }
    }
}

/// Charges each response to the token budgets before anything acts on it.
/// A budget is a durable balance spanning runs (and may ask the owner to
/// extend it), so it is the daemon's to judge; the kernel's own limits only
/// count this run. Served on the turn's task.
pub struct BudgetGate {
    pub calls: mpsc::Sender<HostCall>,
}

#[async_trait]
impl sdk::Hook for BudgetGate {
    /// Longer than an owner's budget card can stay open.
    fn timeout(&self) -> Option<std::time::Duration> {
        Some(std::time::Duration::from_secs(60 * 60))
    }

    async fn after_model(
        &self,
        _ctx: &sdk::HookContext<'_>,
        response: &mut sdk::ModelResponse,
    ) -> sdk::ModelDecision {
        let (reply, answer) = oneshot::channel();
        let sent = self
            .calls
            .send(HostCall::Budget(BudgetCheck {
                response: response.clone(),
                reply,
            }))
            .await;
        if sent.is_err() {
            return sdk::ModelDecision::Accept;
        }
        answer.await.unwrap_or(sdk::ModelDecision::Accept)
    }
}

/// A reasoning model that spent its whole output budget thinking was usually
/// on a hard step: give it more output room first, and lower reasoning
/// effort only when the cutoffs repeat.
#[derive(Default)]
pub struct CutoffAdjust {
    consecutive: std::sync::atomic::AtomicU8,
    raised_output_tokens: std::sync::atomic::AtomicU32,
}

impl CutoffAdjust {
    /// Consecutive cutoffs so far.
    pub fn cutoffs(&self) -> u8 {
        self.consecutive.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl sdk::Hook for CutoffAdjust {
    async fn before_model(&self, _ctx: &sdk::HookContext<'_>, request: &mut sdk::ModelRequest) {
        let raised = self.raised_output_tokens.load(Ordering::Relaxed);
        if raised > 0 {
            request.max_output_tokens = Some(
                request
                    .max_output_tokens
                    .map_or(raised, |cap| cap.max(raised)),
            );
        }
        if let Some(effort) = thinking_cutoff_effort(self.consecutive.load(Ordering::Relaxed)) {
            request.options.reasoning_effort = Some(effort);
        }
    }

    async fn after_model(
        &self,
        ctx: &sdk::HookContext<'_>,
        response: &mut sdk::ModelResponse,
    ) -> sdk::ModelDecision {
        let produced_output = !response.tool_calls.is_empty() || !response.text.trim().is_empty();
        if produced_output {
            self.consecutive.store(0, Ordering::Relaxed);
        } else if response
            .note
            .as_deref()
            .is_some_and(|note| note.contains("truncated"))
        {
            let count = self.consecutive.fetch_add(1, Ordering::Relaxed) + 1;
            let raised = raised_output_ceiling(
                u32::try_from(response.usage.output_tokens).ok(),
                self.raised_output_tokens.load(Ordering::Relaxed),
            );
            if let Some(raised) = raised {
                self.raised_output_tokens.store(raised, Ordering::Relaxed);
            }
            warn!(
                session_id = ctx.session_id,
                consecutive_truncations = count,
                raised_output_tokens = ?raised,
                next_effort = ?thinking_cutoff_effort(count),
                "SDK engine: response truncated at the token limit with no output"
            );
        }
        sdk::ModelDecision::Accept
    }
}

/// Asks the host whether a failed model call is worth another try.
/// Served on the turn's task.
pub struct ModelRetry {
    pub calls: mpsc::Sender<HostCall>,
}

#[async_trait]
impl sdk::Hook for ModelRetry {
    async fn on_model_error(
        &self,
        _ctx: &sdk::HookContext<'_>,
        error: &sdk::ProviderError,
    ) -> sdk::ModelErrorDecision {
        let (reply, answer) = oneshot::channel();
        let sent = self
            .calls
            .send(HostCall::ModelError(ModelErrorCheck {
                timed_out: matches!(error, sdk::ProviderError::Transient(_)),
                reply,
            }))
            .await;
        if sent.is_err() {
            return sdk::ModelErrorDecision::Fail;
        }
        answer.await.unwrap_or(sdk::ModelErrorDecision::Fail)
    }
}

/// Records that the run repeated itself. What to do about it stays the
/// kernel's default: nudge, then stop.
pub struct StuckNotice {
    pub notices: mpsc::UnboundedSender<KernelNotice>,
}

#[async_trait]
impl sdk::Hook for StuckNotice {
    async fn on_stuck(
        &self,
        _ctx: &sdk::HookContext<'_>,
        _signal: &sdk::StuckSignal,
    ) -> Option<sdk::StuckDecision> {
        let _ = self.notices.send(KernelNotice::Stuck);
        None
    }
}

/// The dispatcher's own gates as the last `before_tool` hook, after the
/// owner was asked (so a call approved after a wait is judged as of now):
/// the session's live mode, the untrusted-reference block, the filesystem
/// scope and the mandate authority refuse (`Deny`); deferral for new
/// project instructions, a duplicate `send_file` and a durable replay
/// answer in the tool's place (`Answer`). Served on the turn's task.
pub struct DispatchGate {
    pub calls: mpsc::Sender<HostCall>,
}

#[async_trait]
impl sdk::Hook for DispatchGate {
    fn timeout(&self) -> Option<std::time::Duration> {
        Some(TOOL_TIMEOUT)
    }

    async fn before_tool(
        &self,
        ctx: &sdk::HookContext<'_>,
        call: &sdk::ToolCall,
        _tool: &sdk::ToolInfo,
    ) -> sdk::ToolDecision {
        let (reply, answer) = oneshot::channel();
        let sent = self
            .calls
            .send(HostCall::Gate(GateRequest {
                call: KernelCall {
                    run_id: ctx.run_id,
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    call_id: call.id.clone(),
                },
                reply,
            }))
            .await;
        let stopped = || sdk::ToolDecision::Deny {
            reason: "the tool dispatcher stopped before this call was checked".to_string(),
        };
        if sent.is_err() {
            return stopped();
        }
        answer.await.unwrap_or_else(|_| stopped())
    }
}

/// Records a call the dispatcher ran: result fitting, receipt, status line,
/// learning and the checklist, on the turn's own task. It runs first among
/// the after-tool hooks, so the others see what the model will read.
pub struct AfterCallRecord {
    pub calls: mpsc::Sender<HostCall>,
}

#[async_trait]
impl sdk::Hook for AfterCallRecord {
    fn timeout(&self) -> Option<std::time::Duration> {
        Some(TOOL_TIMEOUT)
    }

    async fn after_tool(
        &self,
        ctx: &sdk::HookContext<'_>,
        receipt: &sdk::Receipt,
        output: &mut sdk::ToolOutput,
    ) {
        let (reply, answer) = oneshot::channel();
        let sent = self
            .calls
            .send(HostCall::AfterCall(AfterCallCheck {
                run_id: ctx.run_id,
                call_id: receipt.call_id.clone(),
                reply,
            }))
            .await;
        match (sent, answer.await) {
            (Ok(()), Ok(None)) => {}
            (Ok(()), Ok(Some(recorded))) => *output = recorded,
            _ => {
                *output = sdk::ToolOutput::error(
                    "the tool dispatcher stopped before recording this result",
                );
            }
        }
    }
}

/// Runs a turn's tool calls on the daemon dispatcher, on the turn's own task.
pub struct TurnDispatch {
    pub calls: mpsc::Sender<HostCall>,
}

#[async_trait]
impl aidaemon_tools::OutcomeRunner for TurnDispatch {
    async fn run(&self, tool: &str, arguments: Value, ctx: &sdk::ToolContext) -> sdk::ToolResult {
        let (reply, answer) = oneshot::channel();
        let abandoned = tokio_util::sync::CancellationToken::new();
        let _stop_when_dropped = abandoned.clone().drop_guard();
        self.calls
            .send(HostCall::Tool(ToolRequest {
                abandoned,
                call: KernelCall {
                    run_id: ctx.run_id,
                    name: tool.to_string(),
                    arguments,
                    call_id: ctx.call_id.clone(),
                },
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
pub fn tool_spec(tool: &dyn aidaemon_tools::OutcomeTool) -> sdk::ToolSpec {
    tool_spec_from_schema(tool, &tool.schema())
}

pub fn tool_spec_from_schema(
    tool: &dyn aidaemon_tools::OutcomeTool,
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

/// A host tool as a turn's kernel agent runs it: the kernel checks the call
/// with `validator`, then the host dispatcher runs it on the turn's own task.
pub fn turn_tool(
    spec: sdk::ToolSpec,
    tool: Arc<dyn aidaemon_tools::OutcomeTool>,
    validator: aidaemon_tools::CallValidator,
    calls: mpsc::Sender<HostCall>,
    trust_roots: Arc<Vec<String>>,
) -> aidaemon_tools::KernelAdapter<dyn aidaemon_tools::OutcomeTool> {
    aidaemon_tools::KernelAdapter::new(tool)
        .with_spec(spec)
        .with_trust_roots(trust_roots)
        .with_timeout(TOOL_TIMEOUT)
        .with_validator(validator)
        .with_runner(Arc::new(TurnDispatch { calls }))
}

/// The output ceiling a thinking cutoff raises to.
pub const MAX_RAISED_OUTPUT_TOKENS: u32 = 32_768;

/// The doubled output ceiling after a cutoff that used `used` tokens, or
/// `None` when it cannot grow (unknown usage, or already at the cap).
pub fn raised_output_ceiling(used: Option<u32>, current: u32) -> Option<u32> {
    let base = used.filter(|used| *used > 0)?.max(current);
    let raised = base.saturating_mul(2).min(MAX_RAISED_OUTPUT_TOKENS);
    (raised > base).then_some(raised)
}

/// Reasoning effort after `cutoffs` consecutive thinking cutoffs: unchanged
/// for the first (the raised output ceiling is the fix), then low, then off.
pub fn thinking_cutoff_effort(cutoffs: u8) -> Option<String> {
    match cutoffs {
        0 | 1 => None,
        2 => Some("low".to_string()),
        _ => Some("off".to_string()),
    }
}
