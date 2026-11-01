//! Experimental: run agent turns on the aidaemon SDK kernel
//! (`crates/aidaemon-core`) instead of the legacy loop.
//!
//! Enabled by building with the `sdk-engine` feature and setting
//! `[agent.engine] kind = "sdk"`. Every path that reaches
//! `handle_message_impl` (channel ingress, internal continuations, goal task
//! leads, spawned children) then runs one SDK run per turn.
//!
//! What the daemon keeps owning:
//! - the durable turn record: `TaskStart`, `UserMessage`, `AssistantResponse`
//!   and `TaskEnd` are emitted here, so channels get a valid response
//!   envelope and history reloads across turns;
//! - tool dispatch: every daemon tool is exposed to the SDK, but calls are
//!   executed by `Agent::execute_tool_outcome` on this turn's task, so argument
//!   enrichment, role checks, workspace grants and the tools' own approval
//!   prompts behave as in the legacy loop.
//!
//! The system prompt comes from the legacy prompt builder (identity, skills,
//! memory facts, project instructions, mandate isolation).
//!
//! Images: the current turn's image attachments go to the model on the user
//! message, and images a tool observed (`ToolObservation` attachments, such
//! as an image file `read_file` opened) go with that tool's result, both as
//! [`sdk::Image`]. Gated by `[files]` vision settings and the engine model.
//! Replayed history stays text-only, like archived turns in the legacy loop.
//!
//! What it drops (on purpose, for comparison): intent gating, completion
//! contracts and the stall/recovery ladder.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use aidaemon_core as sdk;
use async_trait::async_trait;
use futures::stream::{FuturesUnordered, StreamExt};
use futures::FutureExt;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};
use uuid::Uuid;

use super::agent_helpers::touch_heartbeat;
use super::sdk_provider::{
    fit_messages, from_provider_response, to_chat_options, to_daemon_tools, DaemonProvider,
    ModelCall,
};
use super::tool_exec::ToolExecCtx;
use super::Agent;
use crate::agent::vision::{encode_image_attachment, VISION_SKIPPED_SYSTEM_HINT};
use crate::config::{AgentEngineKind, AppConfig, VisionConfig};
use crate::events::{EventEmitter, EventType, TaskOutcome, TaskStatus};
use crate::traits::{AttachmentProvenance, Message, MessageAttachment};
use crate::types::{ChannelContext, StatusUpdate, UserRole};

const DEFAULT_MAX_TURNS: u32 = 40;
/// Prior dialogue messages (including tool calls and results) replayed into
/// each run.
const HISTORY_MESSAGES: usize = 200;
/// Replayed tool results are cut to this many characters: enough to show what
/// an earlier call returned without re-sending large outputs every turn.
const REPLAYED_TOOL_RESULT_CHARS: usize = 2_000;
/// Daemon tools enforce their own timeouts (terminal backgrounding, CLI
/// agents); this only stops a wedged call from holding the turn forever.
const TOOL_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const HEARTBEAT_TICK: Duration = Duration::from_secs(15);
const GOAL_BUDGET_EXHAUSTED: &str = "the goal's token budget is used up";

/// How turns run on the SDK kernel. Held by the root agent and inherited by
/// every child it spawns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SdkEngineSettings {
    /// `[agent.engine] model`: pins every turn to this model instead of the
    /// daemon's own selection.
    pub(crate) model_override: Option<String>,
    pub(crate) max_turns: u32,
}

impl Default for SdkEngineSettings {
    fn default() -> Self {
        Self {
            model_override: None,
            max_turns: DEFAULT_MAX_TURNS,
        }
    }
}

/// The engine settings when the config selects the SDK engine.
pub(crate) fn settings_from_config(config: &AppConfig) -> Option<SdkEngineSettings> {
    let engine = &config.agent.engine;
    if engine.kind != AgentEngineKind::Sdk {
        return None;
    }
    if !engine.api_key.trim().is_empty() {
        warn!(
            "[agent.engine] api_key is ignored: the SDK engine uses the daemon's configured \
             providers"
        );
    }
    let settings = SdkEngineSettings {
        model_override: engine
            .model
            .clone()
            .filter(|model| !model.trim().is_empty()),
        max_turns: engine.max_turns.unwrap_or(DEFAULT_MAX_TURNS).max(1),
    };
    info!(model_override = ?settings.model_override, "Agent engine: SDK kernel (experimental)");
    Some(settings)
}

/// Everything the SDK run asks of the daemon, served on the turn's own task.
pub(super) enum DaemonCall {
    Tool(ToolRequest),
    Model(ModelCall),
    Stop(StopCheck),
}

/// What the legacy bootstrap settled for a turn before the SDK run starts:
/// the recorded task, the assessed contract, the offered tools, the selected
/// model and the system prompt.
pub(in crate::agent) struct SdkTurnSetup {
    /// The persisted user text (bootstrap may have added a transcript).
    pub(in crate::agent) user_text: String,
    pub(in crate::agent) task_id: String,
    pub(in crate::agent) emitter: EventEmitter,
    pub(in crate::agent) offered_tools: HashSet<String>,
    pub(in crate::agent) model: String,
    pub(in crate::agent) instructions: String,
    pub(in crate::agent) contract: crate::agent::completion_contract::CompletionContract,
}

/// The model wants to stop; the dispatcher asks the run ledger.
pub(super) struct StopCheck {
    /// Operations of this run that completed.
    completed_results: usize,
    reply: oneshot::Sender<sdk::StopDecision>,
}

/// The ledger closeout as an SDK stop hook: before a final answer ships, the
/// run ledger decides whether a still-reachable expectation needs another
/// pass. Receipts, not the reply text, are the evidence.
struct LedgerStop {
    calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl sdk::Hook for LedgerStop {
    async fn on_stop(&self, ctx: &sdk::HookContext<'_>, _final_text: &str) -> sdk::StopDecision {
        let completed_results = ctx
            .run_receipts()
            .filter(|receipt| receipt.status == sdk::ReceiptStatus::Succeeded)
            .count();
        let (reply, answer) = oneshot::channel();
        if self
            .calls
            .send(DaemonCall::Stop(StopCheck {
                completed_results,
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
/// continues a same-tool loop, with the legacy guard's limits
/// (`tool_execution/guards.rs`): 8 consecutive calls, 16 for versatile and
/// research tools, 40 for `computer_use`; diverse arguments get the higher
/// limit as a bonus, not a bypass.
fn same_tool_loop(previous: &[(String, Value)], tool: &str, arguments: &Value) -> bool {
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
        super::MAX_CONSECUTIVE_SAME_TOOL + 8
    } else {
        super::MAX_CONSECUTIVE_SAME_TOOL
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
    let diverse = unique.len() * 2 > total;
    let diverse_limit = if computer_use {
        40
    } else {
        super::MAX_CONSECUTIVE_SAME_TOOL + 8
    };
    !diverse || total >= diverse_limit
}

/// The legacy consecutive-same-tool guard as a hook. A hook cannot end the
/// run, so the looping call is refused with a redirect; if the model keeps
/// at it, the kernel's stuck detector ends the run.
struct SameToolLimit;

#[async_trait]
impl sdk::Hook for SameToolLimit {
    async fn before_tool(
        &self,
        ctx: &sdk::HookContext<'_>,
        call: &sdk::ToolCall,
        _tool: &sdk::ToolInfo,
    ) -> sdk::ToolDecision {
        let previous: Vec<(String, Value)> = ctx
            .run_receipts()
            .map(|receipt| (receipt.tool.clone(), receipt.arguments.clone()))
            .collect();
        if !same_tool_loop(&previous, &call.name, &call.arguments) {
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

/// Extra passes the ledger may demand in one run.
const MAX_CLOSEOUT_DEMANDS: u32 = 2;

/// One tool call handed from the SDK run to the daemon dispatcher.
pub(super) struct ToolRequest {
    name: String,
    arguments: Value,
    call_id: String,
    reply: oneshot::Sender<sdk::ToolResult>,
}

/// A daemon tool as the SDK sees it. It only forwards the call; the daemon
/// dispatcher executes it on the turn's own task.
struct BridgedTool {
    spec: sdk::ToolSpec,
    read_only: bool,
    calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl sdk::Tool for BridgedTool {
    fn spec(&self) -> sdk::ToolSpec {
        self.spec.clone()
    }

    fn annotations(&self) -> sdk::ToolAnnotations {
        if self.read_only {
            sdk::ToolAnnotations::READ_ONLY
        } else {
            sdk::ToolAnnotations::WRITE
        }
    }

    /// Daemon tools check their own arguments (some accept aliases for
    /// schema-required keys) and report bad ones through the dispatcher, so
    /// the refusal reaches the run ledger. A kernel-side rejection would not.
    fn validate(&self, _arguments: &Value) -> Result<(), String> {
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
fn tool_spec(tool: &dyn crate::traits::Tool) -> sdk::ToolSpec {
    let schema = tool.schema();
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

/// Anthropic tool-use ids must match `^[a-zA-Z0-9_-]+$`; ids from other
/// providers may not. Map both sides of a pair the same way.
fn wire_call_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn replayed_tool_result(content: &str) -> String {
    if content.chars().count() <= REPLAYED_TOOL_RESULT_CHARS {
        return content.to_string();
    }
    let head: String = content.chars().take(REPLAYED_TOOL_RESULT_CHARS).collect();
    format!("{head}\n[... earlier output truncated]")
}

/// Project the daemon's stored dialogue into SDK messages.
///
/// Tool calls and their results are replayed as real tool-use pairs, so the
/// model can see the evidence behind earlier answers. A call is kept only
/// with its result directly after it (providers reject unpaired tool use),
/// and only for tools still offered. Text-only neighbours of the same role
/// are merged and the transcript starts with a user message.
fn sdk_history(history: &[Message], offered: &HashSet<String>) -> Vec<sdk::Message> {
    let mut messages: Vec<sdk::Message> = Vec::new();
    let mut index = 0;
    while index < history.len() {
        let message = &history[index];
        index += 1;
        let content = message.content.as_deref().map(str::trim).unwrap_or("");
        match message.role.as_str() {
            "user" if !content.is_empty() => push_text(&mut messages, sdk::Role::User, content),
            "assistant" => {
                let calls: Vec<crate::traits::ToolCall> = message
                    .tool_calls_json
                    .as_deref()
                    .and_then(|raw| serde_json::from_str(raw).ok())
                    .unwrap_or_default();
                // Results that answer this message's calls, in order.
                let mut results: Vec<&Message> = Vec::new();
                while index < history.len() && history[index].role == "tool" {
                    results.push(&history[index]);
                    index += 1;
                }
                let mut paired_calls = Vec::new();
                let mut paired_results = Vec::new();
                for call in &calls {
                    if !offered.contains(&call.name) {
                        continue;
                    }
                    let Some(result) = results
                        .iter()
                        .find(|result| result.tool_call_id.as_deref() == Some(call.id.as_str()))
                    else {
                        continue;
                    };
                    let id = wire_call_id(&call.id);
                    let arguments = serde_json::from_str(&call.arguments)
                        .unwrap_or_else(|_| serde_json::json!({}));
                    paired_calls.push(sdk::ToolCall::new(id.clone(), call.name.clone(), arguments));
                    paired_results.push(sdk::Message::tool_result(
                        id,
                        replayed_tool_result(result.content.as_deref().unwrap_or("")),
                    ));
                }
                if messages.is_empty() {
                    continue;
                }
                if paired_calls.is_empty() {
                    if !content.is_empty() {
                        push_text(&mut messages, sdk::Role::Assistant, content);
                    }
                    continue;
                }
                messages.push(sdk::Message::assistant(content, paired_calls));
                messages.extend(paired_results);
            }
            // Results without the call they answer, and anything else.
            _ => {}
        }
    }
    messages
}

/// Append text, merging into the previous message when it has the same role
/// and carries no tool calls.
fn push_text(messages: &mut Vec<sdk::Message>, role: sdk::Role, content: &str) {
    if messages.is_empty() && role != sdk::Role::User {
        return;
    }
    match messages.last_mut() {
        Some(last) if last.role == role && last.tool_calls.is_empty() => {
            last.content.push_str("\n\n");
            last.content.push_str(content);
        }
        _ => messages.push(match role {
            sdk::Role::User => sdk::Message::user(content),
            _ => sdk::Message::assistant(content, Vec::new()),
        }),
    }
}

/// Encode the image attachments `model` may see. Returns the images and
/// whether an eligible image was left out (vision off, a model without
/// vision, or a file that is unreadable or over the size cap).
fn sdk_images(
    attachments: &[&MessageAttachment],
    vision: &VisionConfig,
    model: &str,
) -> (Vec<sdk::Image>, bool) {
    let eligible: Vec<_> = attachments
        .iter()
        .filter(|attachment| vision.mime_allowed(&attachment.mime_type))
        .collect();
    if eligible.is_empty() {
        return (Vec::new(), false);
    }
    if !vision.model_supports_vision(model) {
        return (Vec::new(), true);
    }
    let mut images = Vec::new();
    let mut skipped = false;
    for attachment in eligible {
        match encode_image_attachment(
            Path::new(&attachment.local_path),
            &attachment.mime_type,
            vision.max_image_bytes,
        ) {
            Ok(data) => images.push(sdk::Image::base64(attachment.mime_type.clone(), data)),
            Err(error) => {
                skipped = true;
                warn!(
                    path = %attachment.local_path,
                    mime = %attachment.mime_type,
                    %error,
                    "SDK engine: image not sent to the model"
                );
            }
        }
    }
    (images, skipped)
}

/// The user-facing reply for a run that did not end with a final answer.
fn reply_for(outcome: &sdk::RunOutcome) -> (String, TaskStatus, TaskOutcome) {
    let text = outcome
        .final_text
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string);
    match &outcome.status {
        sdk::RunStatus::Completed => (
            text.unwrap_or_else(|| "Done.".to_string()),
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
        ),
        sdk::RunStatus::LimitReached { limit } => (
            text.unwrap_or_else(|| {
                format!(
                    "I stopped before finishing: the {limit:?} limit for one request was reached."
                )
            }),
            TaskStatus::Completed,
            TaskOutcome::Partial,
        ),
        sdk::RunStatus::Stuck { signal } => (
            text.unwrap_or_else(|| {
                format!("I stopped because I was repeating myself without progress ({signal:?}).")
            }),
            TaskStatus::Completed,
            TaskOutcome::Partial,
        ),
        sdk::RunStatus::Stopped { reason } => (
            text.unwrap_or_else(|| format!("I stopped: {reason}")),
            TaskStatus::Completed,
            TaskOutcome::Partial,
        ),
        sdk::RunStatus::Interrupted { interruption } => (
            format!(
                "I paused before running `{}` and could not ask for approval here.",
                interruption.call.name
            ),
            TaskStatus::Interrupted,
            TaskOutcome::Partial,
        ),
        sdk::RunStatus::Cancelled => (
            "Cancelled.".to_string(),
            TaskStatus::Cancelled,
            TaskOutcome::Failed,
        ),
        sdk::RunStatus::ProviderFailed { error } => (
            format!("The model provider failed: {error}"),
            TaskStatus::Failed,
            TaskOutcome::Failed,
        ),
        other => (
            text.unwrap_or_else(|| format!("The run ended without an answer ({other:?}).")),
            TaskStatus::Failed,
            TaskOutcome::Failed,
        ),
    }
}

/// The goal whose token budget this turn charges.
struct BudgetGoal {
    goal_id: String,
    /// Scheduled runs charge their per-run budget instead of the daily one.
    scheduled: bool,
}

/// Whether a goal's token budget is used up. A scheduled run's per-run
/// budget (when the run registered one) replaces the daily budget; a budget
/// of zero or less means no limit.
fn goal_budget_exhausted(
    daily: Option<&crate::traits::GoalTokenBudgetStatus>,
    run: Option<&crate::goal_tokens::GoalRunBudgetStatus>,
) -> bool {
    if let Some(run) = run {
        return run.effective_budget_per_check > 0
            && run.tokens_used >= run.effective_budget_per_check;
    }
    daily.is_some_and(|daily| {
        daily
            .budget_daily
            .is_some_and(|budget| budget > 0 && daily.tokens_used_today >= budget)
    })
}

/// Credit one receipt to the contract obligations it proves, as the legacy
/// tool phase does: a satisfied mutation marks its effects, and a
/// verifiable observation marks the evidence requirements it matches.
/// Returns the obligation ids the receipt closes.
#[allow(clippy::too_many_arguments)]
fn bind_receipt_to_contract(
    progress: &mut crate::agent::completion_contract::CompletionProgress,
    contract: &crate::agent::completion_contract::CompletionContract,
    call_id: &str,
    tool_name: &str,
    arguments: &str,
    result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
    status: crate::traits::ToolOutcomeStatus,
) -> Vec<String> {
    use super::tool_execution_phase as evidence;
    let semantics = &metadata.semantics;
    if metadata.invocation_stage.performed_io()
        && semantics.mutates_state()
        && !metadata.contract_rejected
        && !metadata.receipt_replayed
    {
        progress.record_mutation_attempt(status);
    }
    // Mutation before observation: a mutation invalidates older evidence, so
    // a combined receipt must not invalidate its own post-action observation.
    if status.satisfies_requested_condition() && semantics.mutates_state() {
        progress.mark_mutation_receipt(contract, semantics, call_id);
    }
    if semantics.observes_state() {
        let matched_contract = evidence::observation_matches_completion_contract(
            contract,
            semantics,
            arguments,
            result_text,
            metadata,
        );
        let matched = evidence::matching_evidence_requirement_indices(
            contract,
            tool_name,
            semantics,
            arguments,
            result_text,
            metadata,
        );
        let available = matches!(
            status,
            crate::traits::ToolOutcomeStatus::Succeeded
                | crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult
        ) || matched.iter().any(|index| {
            contract
                .evidence_requirements
                .get(*index)
                .is_some_and(|requirement| {
                    evidence::evidence_requirement_accepts_nonstandard_outcome(
                        requirement,
                        metadata,
                    )
                })
        });
        if available
            && evidence::tool_result_or_metadata_contains_verifiable_evidence(
                semantics,
                result_text,
                metadata,
            )
        {
            progress.mark_observation_receipt(contract, &matched, matched_contract, call_id);
        }
    }
    progress.completion_obligations_for_receipt(call_id)
}

/// What the ledger says when the model wants to stop.
struct StopFacts {
    /// Expectations an admissible tool could still satisfy.
    reachable_obligation_ids: Vec<String>,
    /// Operations of this run that completed.
    completed_results: usize,
    /// `completed_results` when the executor's checklist was last demanded.
    executor_demanded_at: Option<usize>,
    terminal_policy_denial: bool,
}

/// The legacy completion gate (`completion_phase.rs`), read from receipts:
/// contract expectations are demanded only while nothing has completed; the
/// executor's own open checklist items are demanded again whenever completed
/// work changed since the last demand; nothing is demanded after a terminal
/// policy denial.
fn closeout_demands_work(facts: &StopFacts) -> bool {
    if facts.terminal_policy_denial || facts.reachable_obligation_ids.is_empty() {
        return false;
    }
    let executor_demands = facts
        .reachable_obligation_ids
        .iter()
        .any(|id| id.contains("/obligation:checklist:"))
        && facts.executor_demanded_at != Some(facts.completed_results);
    facts.completed_results == 0 || executor_demands
}

/// Everything a dispatched call needs from the turn.
struct TurnScope<'a> {
    session_id: &'a str,
    task_id: &'a str,
    emitter: &'a EventEmitter,
    /// Replayed dialogue messages before this turn's user message.
    history_len: usize,
    budget_goal: Option<BudgetGoal>,
    /// Set once the goal budget is used up; later model calls are refused.
    budget_exhausted: AtomicBool,
    model_calls: AtomicU64,
    /// Extra passes the ledger demanded so far.
    closeout_demands: std::sync::atomic::AtomicU32,
    /// `completed_results` at the last checklist demand (`u64::MAX`: none).
    executor_demanded_at: AtomicU64,
    status_tx: Option<mpsc::Sender<StatusUpdate>>,
    channel_ctx: &'a ChannelContext,
    user_role: UserRole,
    /// The engine model, for vision gating of tool-observed images.
    model: &'a str,
    turn_id: &'a str,
    /// The assessed contract: its authority is enforced, its expectations
    /// are credited from receipts.
    contract: &'a crate::agent::completion_contract::CompletionContract,
    progress: std::sync::Mutex<crate::agent::completion_contract::CompletionProgress>,
    /// Set when a call moved work to the background with completion
    /// notifications on: the daemon will deliver its result later, so the
    /// reply is a background handoff, not the terminal answer.
    background_handoff: AtomicBool,
}

impl Agent {
    async fn dispatch_sdk_tool_call(&self, scope: &TurnScope<'_>, request: ToolRequest) {
        let arguments = serde_json::to_string(&request.arguments).unwrap_or_else(|_| "{}".into());
        let output = self.run_sdk_tool_call(scope, &request, &arguments).await;
        let _ = request.reply.send(Ok(output));
    }

    /// Run one call and record it as the legacy loop does: a `ToolCall`
    /// event, then the tool message with a typed receipt, so the run ledger
    /// sees every operation while the run is still going.
    async fn run_sdk_tool_call(
        &self,
        scope: &TurnScope<'_>,
        request: &ToolRequest,
        arguments: &str,
    ) -> sdk::ToolOutput {
        let mut ctx = ToolExecCtx {
            session_id: scope.session_id,
            task_id: Some(scope.task_id),
            status_tx: scope.status_tx.clone(),
            channel_visibility: scope.channel_ctx.visibility,
            channel_id: scope.channel_ctx.channel_id.as_deref(),
            project_scope: None,
            trusted: scope.channel_ctx.trusted,
            user_role: scope.user_role,
            workspace_grant: scope.channel_ctx.active_workspace_grant(scope.user_role),
            correction_preapproved: false,
            suppress_trusted_session: false,
            mandate_authority: None,
            tool_call_id: Some(&request.call_id),
            mutation_forbidden: false,
            scope_escalation: None,
        };
        let started = Instant::now();
        // The contract's authority lane (forbidden tool use or mutation,
        // allowed tools, forbidden scopes), then for mandate runs the same
        // per-call authority gate as the legacy loop, which issues the
        // bounded grant or refuses the call.
        let read_only = self
            .tools
            .iter()
            .find(|tool| tool.name() == request.name)
            .map(|tool| tool.capabilities().read_only);
        let admitted =
            super::closeout::ClosingAuthority::from_authority(scope.contract.authority())
                .admits_tool(&request.name, read_only);
        let gate = if admitted {
            self.sdk_mandate_gate(&ctx, request, arguments)
                .await
                .map_err(|refusal| (refusal, "mandate_authority"))
        } else {
            Err((
                format!(
                    "[AUTHORITY BLOCKED] `{}` is outside what this request allows. Use a \
                     permitted tool or report the limitation.",
                    request.name
                ),
                "request_authority",
            ))
        };
        let grant = match gate {
            Ok(grant) => grant,
            Err((refusal, reason_code)) => {
                info!(
                    session_id = scope.session_id,
                    task_id = scope.task_id,
                    tool = %request.name,
                    reason_code,
                    "SDK engine: authority gate refused the call"
                );
                let call = crate::traits::ToolCall {
                    id: request.call_id.clone(),
                    name: request.name.clone(),
                    arguments: arguments.to_string(),
                    extra_content: None,
                };
                let semantics = self
                    .semantics_for_exact_tool_call(&request.name, arguments)
                    .await;
                if let Err(error) = self
                    .persist_pre_dispatch_policy_denial(
                        scope.emitter,
                        scope.session_id,
                        scope.task_id,
                        &call,
                        arguments,
                        refusal.clone(),
                        reason_code,
                        semantics,
                        None,
                    )
                    .await
                {
                    warn!(task_id = scope.task_id, %error, "SDK engine: could not record a denial");
                }
                return sdk::ToolOutput::error(refusal);
            }
        };
        ctx.mandate_authority = grant.as_ref();

        let mut call_data = crate::events::ToolCallData::from_tool_call(
            request.call_id.clone(),
            request.name.clone(),
            request.arguments.clone(),
            Some(scope.task_id.to_string()),
        );
        call_data.turn_id = Some(scope.turn_id.to_string());
        if let Err(error) = scope.emitter.emit(EventType::ToolCall, call_data).await {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a tool call");
        }

        let call_semantics = self
            .semantics_for_exact_tool_call(&request.name, arguments)
            .await;
        let (output, images, metadata, receipt) = match self
            .execute_tool_outcome(&request.name, arguments, &ctx)
            .await
        {
            Ok(mut outcome) => {
                // An executed call keeps the semantics it was admitted with
                // (an observation stays an observation even when it returns a
                // negative result), as in the legacy tool phase.
                super::tool_execution_phase::complete_tool_result_semantics(
                    &request.name,
                    arguments,
                    &call_semantics,
                    &mut outcome.metadata,
                );
                let status = super::loop_utils::classify_tool_outcome_status(
                    &request.name,
                    &outcome.output,
                    Some(arguments),
                    Some(&outcome.metadata),
                );
                let evidence = super::loop_utils::tool_outcome_evidence_source(&outcome.metadata);
                let mut receipt = crate::events::ToolReceiptV1::from_metadata(
                    &outcome.metadata,
                    status,
                    evidence,
                    None,
                );
                receipt.mandate_authority = grant.clone();
                if let Ok(mut progress) = scope.progress.lock() {
                    receipt.completion_obligation_ids = bind_receipt_to_contract(
                        &mut progress,
                        scope.contract,
                        &request.call_id,
                        &request.name,
                        arguments,
                        &outcome.output,
                        &outcome.metadata,
                        status,
                    );
                }
                let mut output = outcome.output;
                let observed: Vec<&MessageAttachment> = outcome
                    .metadata
                    .attachments
                    .iter()
                    .filter(|a| a.provenance == AttachmentProvenance::ToolObservation)
                    .collect();
                let (images, skipped) = sdk_images(&observed, &self.vision_config, scope.model);
                if skipped {
                    output.push_str(
                        "\n\n[SYSTEM] The image this tool observed could not be shown to you \
                         (vision is off, unsupported, or the file is too large or unreadable).",
                    );
                }
                let backgrounded = outcome.metadata.background_started || outcome.metadata.detached;
                if backgrounded && outcome.metadata.completion_notifications_enabled {
                    scope.background_handoff.store(true, Ordering::Relaxed);
                    output.push_str(
                        "\n\n[SYSTEM] Completion notifications are enabled for this background \
                         work: its result will be delivered to this conversation automatically \
                         when it finishes.",
                    );
                }
                (output, images, Some(outcome.metadata), Some(receipt))
            }
            Err(error) => (format!("Error: {error}"), Vec::new(), None, None),
        };
        let failed = receipt
            .as_ref()
            .map(|receipt| receipt.outcome_status.is_failure() || receipt.contract_rejected)
            .unwrap_or(true);
        let duration_ms = started.elapsed().as_millis() as u64;

        let mut tool = Message::new_runtime(Uuid::new_v4().to_string(), scope.session_id, "tool");
        tool.content = Some(output.clone());
        tool.tool_name = Some(request.name.clone());
        tool.tool_call_id = Some(request.call_id.clone());
        tool.turn_id = Some(scope.turn_id.to_string());
        if let Some(metadata) = &metadata {
            tool.attachments = metadata.attachments.clone();
        }
        if let Err(error) = self
            .append_tool_message_with_receipt_event_policy(
                scope.emitter,
                &tool,
                !failed,
                duration_ms,
                failed.then(|| output.clone()),
                Some(scope.task_id),
                metadata
                    .as_ref()
                    .and_then(|metadata| metadata.persistent_output.as_deref()),
                receipt,
            )
            .await
        {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a tool result");
        }
        // The executor's own checklist becomes ledger expectations.
        if request.name == "track_requirements" && !failed {
            if let Some(declared) =
                crate::tools::track_requirements::executor_expectations_from_checklist_arguments(
                    scope.task_id,
                    arguments,
                )
            {
                if let Err(error) = scope
                    .emitter
                    .emit(EventType::ExecutorExpectationsDeclared, declared)
                    .await
                {
                    warn!(task_id = scope.task_id, %error, "SDK engine: could not record a checklist");
                }
            }
        }
        info!(
            session_id = scope.session_id,
            task_id = scope.task_id,
            tool = %request.name,
            failed,
            duration_ms,
            "SDK engine tool call"
        );
        sdk::ToolOutput {
            is_error: failed,
            ..sdk::ToolOutput::text(output).with_images(images)
        }
    }

    async fn serve_sdk_stop(&self, scope: &TurnScope<'_>, check: StopCheck) {
        let decision = self.sdk_closeout(scope, check.completed_results).await;
        let _ = check.reply.send(decision);
    }

    /// The legacy ledger-first closeout on the SDK path: replay the run
    /// ledger, ask which expectations an admissible tool could still satisfy,
    /// and demand a bounded extra pass only when the receipts say so.
    async fn sdk_closeout(
        &self,
        scope: &TurnScope<'_>,
        completed_results: usize,
    ) -> sdk::StopDecision {
        let Ok(aggregate) = self
            .event_store
            .task_run_aggregate(scope.session_id, scope.task_id)
            .await
        else {
            return sdk::StopDecision::Accept;
        };
        let iteration = scope.model_calls.load(Ordering::Relaxed) as usize;
        let authority =
            super::closeout::ClosingAuthority::from_authority(scope.contract.authority());
        let visible_tools: Vec<&str> = if scope.user_role == UserRole::Public {
            Vec::new()
        } else {
            self.tools.iter().map(|tool| tool.name()).collect()
        };
        let read_only = |name: &str| {
            self.tools
                .iter()
                .find(|tool| tool.name() == name)
                .map(|tool| tool.capabilities().read_only)
        };
        let decision = aggregate.closeout(|obligation| {
            super::closeout::obligation_admissible(
                obligation,
                &authority,
                &visible_tools,
                read_only,
            )
        });
        let reachable = match &decision {
            crate::events::CloseoutDecision::Reachable { obligation_ids } => obligation_ids.clone(),
            _ => Vec::new(),
        };
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            iteration,
            crate::events::DecisionType::PostExecutionValidation,
            format!("Ledger closeout verdict: {}", decision.verdict()),
            serde_json::json!({
                "condition": "ledger_closeout",
                "verdict": decision.verdict(),
                "reachable_obligation_ids": reachable,
                "authoritative": true,
                "engine": "sdk",
            }),
        )
        .await;
        let demanded_at = scope.executor_demanded_at.load(Ordering::Relaxed);
        let facts = StopFacts {
            reachable_obligation_ids: reachable.clone(),
            completed_results,
            executor_demanded_at: (demanded_at != u64::MAX).then_some(demanded_at as usize),
            terminal_policy_denial: aggregate.terminal_policy_denial(),
        };
        if !closeout_demands_work(&facts)
            || scope.closeout_demands.load(Ordering::Relaxed) >= MAX_CLOSEOUT_DEMANDS
        {
            return sdk::StopDecision::Accept;
        }
        let demands = scope.closeout_demands.fetch_add(1, Ordering::Relaxed) + 1;
        if reachable
            .iter()
            .any(|id| id.contains("/obligation:checklist:"))
        {
            scope
                .executor_demanded_at
                .store(completed_results as u64, Ordering::Relaxed);
        }
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            iteration,
            crate::events::DecisionType::IntentGate,
            "Ledger expectations still reachable: bounded evidence pass requested",
            serde_json::json!({
                "condition": "ledger_expectations_required",
                "reachable_obligation_ids": reachable,
                "demands": demands,
                "engine": "sdk",
            }),
        )
        .await;
        sdk::StopDecision::Continue {
            message: super::system_directives::SystemDirective::LedgerExpectationsRequired {
                expectations: aggregate.expectation_summaries(&reachable),
            }
            .render(),
        }
    }

    /// Make one model call for the SDK run through the daemon's provider
    /// stack: billing-aware retries, the fallback cascade and failover
    /// providers all apply, exactly as for a legacy-loop call.
    async fn serve_sdk_model_call(&self, scope: &TurnScope<'_>, call: ModelCall) {
        let reply = self.sdk_model_call(scope, &call.request).await;
        let _ = call.reply.send(reply);
    }

    async fn sdk_model_call(
        &self,
        scope: &TurnScope<'_>,
        request: &sdk::ModelRequest,
    ) -> Result<sdk::ModelResponse, sdk::ProviderError> {
        if scope.budget_exhausted.load(Ordering::Relaxed) {
            return Err(sdk::ProviderError::Other(GOAL_BUDGET_EXHAUSTED.into()));
        }
        let iteration = scope.model_calls.fetch_add(1, Ordering::Relaxed) + 1;
        let snapshot = self.llm_runtime.snapshot();
        let tools = to_daemon_tools(&request.tools);
        let budget = crate::memory::context_window::compute_available_budget_precomputed(
            &request.model,
            0,
            &tools,
            &self.context_window_config,
        );
        let messages = fit_messages(&request.messages, scope.history_len, budget);
        let options = to_chat_options(&request.tool_choice, request.max_output_tokens);
        let est_input_tokens =
            crate::memory::context_window::estimate_multimodal_message_tokens(&messages);
        let mut telemetry = super::llm::LlmCallTelemetry::default();
        let started = Instant::now();
        let result = self
            .call_llm_with_recovery(
                snapshot.provider(),
                snapshot.router(),
                &request.model,
                &messages,
                &tools,
                &options,
                &mut telemetry,
                None,
            )
            .await;
        let latency_ms = started.elapsed().as_millis() as u64;
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                warn!(
                    session_id = scope.session_id,
                    task_id = scope.task_id,
                    model = %request.model,
                    %error,
                    "SDK engine: model call failed after recovery"
                );
                return Err(sdk::ProviderError::Other(error.to_string()));
            }
        };
        let usage = response.usage.clone();
        let (input_tokens, output_tokens) = usage
            .as_ref()
            .map(|usage| (usage.input_tokens, usage.output_tokens))
            .unwrap_or_default();
        crate::events::record_model_call_telemetry(
            scope.emitter,
            self.state.as_ref(),
            crate::events::ModelCallTelemetryInput {
                session_id: scope.session_id.to_string(),
                task_id: scope.task_id.to_string(),
                call_purpose: None,
                iteration: Some(iteration as u32),
                llm_call: crate::events::LlmCallData {
                    call_id: None,
                    call_purpose: None,
                    task_id: scope.task_id.to_string(),
                    iteration: Some(iteration as u32),
                    model: request.model.clone(),
                    final_model: Some(telemetry.final_model.clone()),
                    fell_back: telemetry.fell_back,
                    attempts: telemetry.attempts,
                    latency_ms,
                    prompt_ms: usage.as_ref().and_then(|usage| usage.prompt_ms),
                    decode_ms: usage.as_ref().and_then(|usage| usage.decode_ms),
                    input_tokens,
                    output_tokens,
                    cached_input_tokens: usage.as_ref().and_then(|u| u.cached_input_tokens),
                    cache_creation_input_tokens: usage
                        .as_ref()
                        .and_then(|u| u.cache_creation_input_tokens),
                    fresh_input_tokens: usage.as_ref().and_then(|u| u.fresh_input_tokens()),
                    est_input_tokens: Some(est_input_tokens as u32),
                    tool_calls_count: response.tool_calls.len() as u32,
                    build_ms: None,
                    prefix_hash_system: None,
                    prefix_hash_pre_boundary: None,
                    tool_defs_hash: None,
                    offered_tools: request.tools.iter().map(|tool| tool.name.clone()).collect(),
                    chosen_tools: response
                        .tool_calls
                        .iter()
                        .map(|tc| tc.name.clone())
                        .collect(),
                    session_summary_hash: None,
                    tail_hash: None,
                    prefix_hash_archived: None,
                    boundary_pos: None,
                    message_count: Some(messages.len()),
                    projected_source_message_ids: Vec::new(),
                    projected_source_turn_ids: Vec::new(),
                    force_text: false,
                    token_usage_present: usage.is_some(),
                    token_usage_evidence: crate::events::TokenUsageEvidence::Unavailable,
                    failed: false,
                    error: None,
                    provider_error_kind: None,
                    provider_status: None,
                },
                token_usage: usage.clone(),
            },
        )
        .await;
        if let (Some(goal), Some(usage)) = (&scope.budget_goal, usage.as_ref()) {
            if self.charge_goal_budget(scope, goal, usage).await && !response.tool_calls.is_empty()
            {
                return Err(sdk::ProviderError::Other(GOAL_BUDGET_EXHAUSTED.into()));
            }
        }
        let response = from_provider_response(response);
        if !response.tool_calls.is_empty() {
            self.persist_sdk_tool_call_message(scope, &response).await;
        }
        Ok(response)
    }

    /// Record an assistant message that calls tools before the calls run, so
    /// history and the run ledger see each call with the message that made it.
    async fn persist_sdk_tool_call_message(
        &self,
        scope: &TurnScope<'_>,
        response: &sdk::ModelResponse,
    ) {
        let calls: Vec<crate::traits::ToolCall> = response
            .tool_calls
            .iter()
            .map(|call| crate::traits::ToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.to_string(),
                extra_content: None,
            })
            .collect();
        let mut assistant =
            Message::new_runtime(Uuid::new_v4().to_string(), scope.session_id, "assistant");
        let text = response.text.trim();
        assistant.content = (!text.is_empty()).then(|| text.to_string());
        assistant.tool_calls_json = serde_json::to_string(&calls).ok();
        assistant.turn_id = Some(scope.turn_id.to_string());
        if let Err(error) = self
            .append_assistant_message_with_event(scope.emitter, &assistant, scope.model, None, None)
            .await
        {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a tool-call message");
        }
    }

    /// Charge a model call to the turn's goal. Returns true when this call
    /// used up the budget; the owner is alerted once. A final answer still
    /// goes out (as the legacy closeout grace allows); further work does not.
    async fn charge_goal_budget(
        &self,
        scope: &TurnScope<'_>,
        goal: &BudgetGoal,
        usage: &crate::traits::TokenUsage,
    ) -> bool {
        let delta = usage.budget_tokens() as i64;
        let daily = match self
            .state
            .add_goal_tokens_and_get_budget_status(&goal.goal_id, delta)
            .await
        {
            Ok(status) => status,
            Err(error) => {
                warn!(goal_id = %goal.goal_id, %error, "SDK engine: goal token accounting failed");
                None
            }
        };
        let run = if goal.scheduled {
            match &self.goal_token_registry {
                Some(registry) => registry.add_run_tokens(&goal.goal_id, delta).await,
                None => None,
            }
        } else {
            None
        };
        let daily = if goal.scheduled { None } else { daily.as_ref() };
        if !goal_budget_exhausted(daily, run.as_ref()) {
            return false;
        }
        if !scope.budget_exhausted.swap(true, Ordering::Relaxed) {
            warn!(
                session_id = scope.session_id,
                goal_id = %goal.goal_id,
                "SDK engine: goal token budget exhausted"
            );
            let message = format!(
                "Token alert: goal '{}' used up its token budget. The run stopped safely before \
                 completion.",
                goal.goal_id
            );
            self.fanout_token_alert(
                Some(goal.goal_id.as_str()),
                scope.session_id,
                &message,
                Some(scope.session_id),
            )
            .await;
        }
        true
    }

    /// Outside a mandate run: `Ok(None)`. Inside one: the grant for this exact
    /// call, or the refusal text the model should see.
    async fn sdk_mandate_gate(
        &self,
        ctx: &ToolExecCtx<'_>,
        request: &ToolRequest,
        arguments: &str,
    ) -> Result<Option<crate::traits::MandateAuthorityGrant>, String> {
        let goal_id = match self.mandate_goal_id_for_tool_exec(ctx).await {
            Ok(Some(goal_id)) => goal_id,
            Ok(None) => return Ok(None),
            Err(error) => return Err(format!("[MANDATE AUTHORITY BLOCKED] {error}")),
        };
        let semantics = self
            .semantics_for_exact_tool_call(&request.name, arguments)
            .await;
        match super::tool_execution_phase::mandate_gate_for_tool_call(
            self,
            &goal_id,
            &request.call_id,
            &request.name,
            arguments,
            &semantics,
        )
        .await
        {
            Ok(super::tool_execution_phase::MandateGateDecision::NotApplicable) => Ok(None),
            Ok(super::tool_execution_phase::MandateGateDecision::Allow { grant }) => Ok(grant),
            Ok(super::tool_execution_phase::MandateGateDecision::Deny { reason }) => Err(format!(
                "[MANDATE AUTHORITY BLOCKED] `{}` cannot run in this decision cycle ({reason}). \
                 Record a current ACT, choose a permitted action, WAIT, or ASK the owner.",
                request.name
            )),
            Err(error) => Err(format!("[MANDATE AUTHORITY BLOCKED] {error}")),
        }
    }

    /// One turn on the SDK kernel, after the legacy bootstrap set it up. See
    /// the module docs for what is kept and what is dropped compared with
    /// the legacy loop.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_message_sdk(
        &self,
        engine: &SdkEngineSettings,
        setup: SdkTurnSetup,
        session_id: &str,
        attachments: &[MessageAttachment],
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
    ) -> anyhow::Result<String> {
        let started = Instant::now();
        let SdkTurnSetup {
            user_text,
            task_id,
            emitter,
            offered_tools,
            model,
            instructions,
            contract,
        } = setup;
        let user_text = user_text.as_str();
        let model = engine.model_override.clone().unwrap_or(model);
        touch_heartbeat(&heartbeat);
        let turn_id = self
            .current_turn_ids
            .read()
            .await
            .get(session_id)
            .cloned()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        // Prior dialogue; bootstrap already recorded this turn's user message.
        let history: Vec<Message> = match self
            .event_store
            .get_conversation_history_since_clear(session_id, HISTORY_MESSAGES)
            .await
        {
            Ok(history) => history
                .into_iter()
                .filter(|message| message.turn_id.as_deref() != Some(turn_id.as_str()))
                .collect(),
            Err(error) => {
                warn!(session_id, %error, "SDK engine: history unavailable");
                Vec::new()
            }
        };
        let (images, vision_skipped) = sdk_images(
            &attachments.iter().collect::<Vec<_>>(),
            &self.vision_config,
            &model,
        );
        let instructions = if vision_skipped {
            format!("{instructions}\n\n{VISION_SKIPPED_SYSTEM_HINT}")
        } else {
            instructions
        };

        let (calls_tx, mut calls_rx) = mpsc::channel::<DaemonCall>(16);
        let mut builder = sdk::Agent::builder("aidaemon")
            .model(model.clone())
            .instructions(instructions)
            .provider(DaemonProvider {
                calls: calls_tx.clone(),
            })
            .limits(sdk::Limits {
                max_turns: engine.max_turns,
                tool_timeout: TOOL_TIMEOUT,
                ..sdk::Limits::default()
            });
        // Tools: the set bootstrap offered for this turn (role, policy and
        // contract filtering already applied).
        if user_role != UserRole::Public {
            for tool in self
                .tools
                .iter()
                .filter(|tool| offered_tools.contains(tool.name()))
            {
                builder = builder.tool(BridgedTool {
                    spec: tool_spec(tool.as_ref()),
                    read_only: tool.capabilities().read_only,
                    calls: calls_tx.clone(),
                });
            }
        }
        builder = builder.hook(SameToolLimit);
        builder = builder.hook(LedgerStop {
            calls: calls_tx.clone(),
        });
        drop(calls_tx);
        let sink_status = status_tx.clone();
        let sink_heartbeat = heartbeat.clone();
        builder = builder.event_sink(move |_run, event: &sdk::Event| {
            touch_heartbeat(&sink_heartbeat);
            let Some(status_tx) = sink_status.as_ref() else {
                return;
            };
            let update = match event {
                sdk::Event::TurnStarted { turn } if *turn > 0 => {
                    StatusUpdate::Thinking(*turn as usize)
                }
                sdk::Event::ToolStarted { tool, .. } => StatusUpdate::ToolStart {
                    name: tool.clone(),
                    summary: String::new(),
                },
                sdk::Event::ToolFinished { receipt } => StatusUpdate::ToolComplete {
                    name: receipt.tool.clone(),
                    summary: format!("{:?}", receipt.status).to_lowercase(),
                },
                _ => return,
            };
            let _ = status_tx.try_send(update);
        });
        let sdk_agent = builder
            .build()
            .map_err(|error| anyhow::anyhow!("SDK engine agent: {error}"))?;

        let offered: HashSet<String> = sdk_agent
            .tool_specs()
            .iter()
            .map(|spec| spec.name.clone())
            .collect();
        let mut state = sdk::RunState::new(
            "aidaemon",
            session_id,
            sdk::Message::user(user_text).with_images(images),
        );
        let mut messages = sdk_history(&history, &offered);
        let replayed = messages.len();
        messages.append(&mut state.messages);
        state.messages = messages;

        // Drive the run and the tool dispatcher together on this task.
        let budget_goal = match self.budget_goal_id(session_id).await {
            Some(goal_id) => Some(BudgetGoal {
                scheduled: crate::agent::goal_dispatch::goal_has_scheduled_provenance(
                    &self.state,
                    &goal_id,
                    self.task_id.as_deref(),
                )
                .await,
                goal_id,
            }),
            None => None,
        };
        let scope = TurnScope {
            session_id,
            task_id: &task_id,
            status_tx: status_tx.clone(),
            channel_ctx: &channel_ctx,
            user_role,
            model: &model,
            turn_id: &turn_id,
            contract: &contract,
            progress: std::sync::Mutex::new(
                crate::agent::completion_contract::CompletionProgress::new(&contract, &task_id),
            ),
            background_handoff: AtomicBool::new(false),
            emitter: &emitter,
            history_len: replayed,
            budget_goal,
            budget_exhausted: AtomicBool::new(false),
            model_calls: AtomicU64::new(0),
            closeout_demands: std::sync::atomic::AtomicU32::new(0),
            executor_demanded_at: AtomicU64::new(u64::MAX),
        };
        // A parent or goal cancellation stops the run at its next check.
        let cancel = self
            .cancel_token
            .as_ref()
            .map(|token| token.child_token())
            .unwrap_or_default();
        let run = sdk_agent.run_from(state, cancel);
        tokio::pin!(run);
        let mut in_flight = FuturesUnordered::new();
        let mut tick = tokio::time::interval(HEARTBEAT_TICK);
        let outcome = loop {
            tokio::select! {
                outcome = &mut run => break outcome,
                Some(call) = calls_rx.recv() => match call {
                    DaemonCall::Tool(request) => {
                        in_flight.push(self.dispatch_sdk_tool_call(&scope, request).boxed());
                    }
                    DaemonCall::Model(call) => {
                        in_flight.push(self.serve_sdk_model_call(&scope, call).boxed());
                    }
                    DaemonCall::Stop(check) => {
                        in_flight.push(self.serve_sdk_stop(&scope, check).boxed());
                    }
                },
                Some(()) = in_flight.next(), if !in_flight.is_empty() => {}
                _ = tick.tick() => touch_heartbeat(&heartbeat),
            }
        };

        let (reply, status, task_outcome, error) = match outcome {
            Ok(outcome) => {
                let (reply, status, task_outcome) = reply_for(&outcome);
                info!(
                    session_id,
                    task_id,
                    status = ?outcome.status,
                    turns = outcome.turns(),
                    tool_calls = outcome.state.tool_calls,
                    "SDK engine run finished"
                );
                let usage = outcome.run_usage();
                let mut assistant =
                    Message::new_runtime(Uuid::new_v4().to_string(), session_id, "assistant");
                assistant.content = Some(reply.clone());
                assistant.turn_id = Some(turn_id.clone());
                let input_tokens = Some(usage.input_tokens.min(u32::MAX as u64) as u32);
                let output_tokens = Some(usage.output_tokens.min(u32::MAX as u64) as u32);
                if scope.background_handoff.load(Ordering::Relaxed) {
                    self.append_background_handoff_message_with_event(
                        &emitter,
                        &assistant,
                        &model,
                        input_tokens,
                        output_tokens,
                    )
                    .await?;
                } else {
                    self.append_assistant_message_with_event(
                        &emitter,
                        &assistant,
                        &model,
                        input_tokens,
                        output_tokens,
                    )
                    .await?;
                }
                let error = match &outcome.status {
                    sdk::RunStatus::ProviderFailed { error } => Some(error.clone()),
                    _ => None,
                };
                (
                    Ok(reply),
                    status,
                    task_outcome,
                    (error, outcome.turns(), outcome.state.tool_calls),
                )
            }
            Err(error) => {
                let message = error.to_string();
                (
                    Err(anyhow::anyhow!("SDK engine run failed: {message}")),
                    TaskStatus::Failed,
                    TaskOutcome::Failed,
                    (Some(message), 0, 0),
                )
            }
        };
        let (error, iterations, tool_calls_count) = error;
        // The replayed run ledger has the final say on the outcome, exactly as
        // for legacy turns.
        self.emit_task_end(
            &emitter,
            &task_id,
            status,
            task_outcome,
            started,
            iterations as usize,
            tool_calls_count as usize,
            error,
            None,
        )
        .await;
        for tool in &self.tools {
            let _ = tool.on_task_end(&task_id, session_id).await;
        }
        self.current_turn_ids.write().await.remove(session_id);
        reply
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::TaskEndData;

    fn vision(max_image_bytes: u64) -> VisionConfig {
        VisionConfig {
            enabled: true,
            max_image_bytes,
            mime_types: vec!["image/jpeg".to_string(), "image/png".to_string()],
            model_patterns: vec!["claude-".to_string()],
        }
    }

    fn attachment(file: &tempfile::NamedTempFile, mime: &str) -> MessageAttachment {
        MessageAttachment {
            local_path: file.path().to_string_lossy().into_owned(),
            filename: "photo.jpg".to_string(),
            mime_type: mime.to_string(),
            size_bytes: 4,
            ..MessageAttachment::default()
        }
    }

    #[test]
    fn image_attachments_are_encoded_for_a_vision_model() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&[0xFF, 0xD8, 0xFF, 0xE0]).unwrap();
        let photo = attachment(&file, "image/jpeg");
        let voice = attachment(&file, "audio/ogg");

        let (images, skipped) = sdk_images(&[&photo, &voice], &vision(1024), "claude-sonnet-5-5");
        assert!(!skipped);
        assert_eq!(images, vec![sdk::Image::base64("image/jpeg", "/9j/4A==")]);

        // No image attachments: nothing to send and nothing skipped.
        assert_eq!(
            sdk_images(&[&voice], &vision(1024), "claude-sonnet-5-5"),
            (Vec::new(), false)
        );
    }

    #[test]
    fn images_are_skipped_without_vision_or_over_the_cap() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&[0xFF, 0xD8, 0xFF, 0xE0]).unwrap();
        let photo = attachment(&file, "image/jpeg");

        assert_eq!(
            sdk_images(&[&photo], &vision(1024), "text-only-model"),
            (Vec::new(), true)
        );
        let mut disabled = vision(1024);
        disabled.enabled = false;
        assert_eq!(
            sdk_images(&[&photo], &disabled, "claude-sonnet-5-5"),
            (Vec::new(), true)
        );
        assert_eq!(
            sdk_images(&[&photo], &vision(2), "claude-sonnet-5-5"),
            (Vec::new(), true)
        );
    }

    fn message(role: &str, content: &str) -> Message {
        let mut message = Message::new_runtime(Uuid::new_v4().to_string(), "s", role);
        message.content = Some(content.to_string());
        message
    }

    #[test]
    fn history_starts_with_user_and_merges_same_role_neighbours() {
        let history = vec![
            message("assistant", "orphaned greeting"),
            message("user", "first"),
            message("user", "second"),
            message("tool", "tool output"),
            message("assistant", "answer"),
            message("assistant", ""),
        ];
        let projected = sdk_history(&history, &HashSet::new());
        assert_eq!(projected.len(), 2);
        assert_eq!(projected[0].role, sdk::Role::User);
        assert_eq!(projected[0].content, "first\n\nsecond");
        assert_eq!(projected[1].role, sdk::Role::Assistant);
        assert_eq!(projected[1].content, "answer");
    }

    fn assistant_calling(calls: &[(&str, &str)]) -> Message {
        let mut message = message("assistant", "");
        let calls: Vec<crate::traits::ToolCall> = calls
            .iter()
            .map(|(id, name)| crate::traits::ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: r#"{"path":"a.txt"}"#.to_string(),
                extra_content: None,
            })
            .collect();
        message.tool_calls_json = serde_json::to_string(&calls).ok();
        message
    }

    fn tool_result(call_id: &str, content: &str) -> Message {
        let mut message = message("tool", content);
        message.tool_call_id = Some(call_id.to_string());
        message
    }

    #[test]
    fn history_replays_paired_tool_calls_and_drops_unpaired_ones() {
        let offered: HashSet<String> = ["read_file".to_string()].into();
        let history = vec![
            message("user", "read it"),
            // Kept: offered tool with its result; the id is made wire-safe.
            assistant_calling(&[("call:1", "read_file"), ("call-2", "read_file")]),
            tool_result("call:1", "file body"),
            // call-2 has no result, so it is dropped from the pair.
            message("assistant", "It says: file body"),
            message("user", "again"),
            // Dropped: tool no longer offered.
            assistant_calling(&[("call-3", "old_tool")]),
            tool_result("call-3", "old output"),
            // Dropped: result without its call.
            tool_result("call-9", "stray"),
            message("assistant", "done"),
        ];
        let projected = sdk_history(&history, &offered);
        let roles: Vec<sdk::Role> = projected.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            [
                sdk::Role::User,
                sdk::Role::Assistant,
                sdk::Role::Tool,
                sdk::Role::Assistant,
                sdk::Role::User,
                sdk::Role::Assistant,
            ]
        );
        assert_eq!(projected[1].tool_calls.len(), 1);
        assert_eq!(projected[1].tool_calls[0].id, "call_1");
        assert_eq!(projected[1].tool_calls[0].arguments["path"], "a.txt");
        assert_eq!(projected[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(projected[2].content, "file body");
        assert_eq!(projected[5].content, "done");
    }

    #[test]
    fn replayed_tool_results_are_capped() {
        let long = "x".repeat(REPLAYED_TOOL_RESULT_CHARS + 50);
        let replayed = replayed_tool_result(&long);
        assert!(replayed.ends_with("[... earlier output truncated]"));
        assert!(replayed.len() < long.len());
    }

    fn owner_request(text: &str) -> crate::runtime_ports::InboundMessageRequest {
        crate::runtime_ports::InboundMessageRequest {
            session_id: "telegram:synthetic-user-1".to_string(),
            user_text: text.to_string(),
            attachments: Vec::new(),
            status_tx: None,
            user_role: UserRole::Owner,
            channel_ctx: ChannelContext::private("telegram"),
            heartbeat: None,
            ingress_timing: None,
        }
    }

    /// The whole daemon path on the SDK engine with the mock provider behind
    /// the daemon's provider stack: a tool dispatched through the bridge, the
    /// final answer delivered, and the tool result recorded.
    #[tokio::test]
    async fn sdk_engine_runs_on_mock_provider() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        use crate::traits::store_prelude::*;

        let provider = MockProvider::with_responses(vec![
            MockProvider::tool_call_response(
                "remember_fact",
                r#"{"category":"preference","key":"favorite_color","value":"teal"}"#,
            ),
            MockProvider::text_response("Saved."),
        ]);
        let harness = crate::testing::setup_test_agent_sdk(provider)
            .await
            .unwrap();
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("Remember that my favorite color is teal."))
            .await
            .unwrap();
        assert_eq!(reply.text, "Saved.");
        assert_eq!(harness.provider.call_count().await, 2);
        let results = harness
            .agent
            .event_store
            .task_event_count(
                "telegram:synthetic-user-1",
                &reply.task_id,
                EventType::ToolResult,
            )
            .await
            .unwrap();
        assert_eq!(results, 1);
        let facts = harness.state.get_facts(None).await.unwrap();
        assert!(facts.iter().any(|fact| fact.value.contains("teal")));
    }

    async fn models_harness() -> crate::testing::TestHarness {
        use crate::testing::MockProvider;
        let provider = MockProvider::with_responses(vec![
            MockProvider::text_response("one"),
            MockProvider::text_response("two"),
        ]);
        let mut harness =
            crate::testing::setup_test_agent_with_models(provider, "primary-model", "smart-model")
                .await
                .unwrap();
        harness
            .agent
            .set_sdk_engine(Some(SdkEngineSettings::default()));
        harness
    }

    async fn recorded_models(harness: &crate::testing::TestHarness) -> Vec<String> {
        harness
            .provider
            .call_log
            .lock()
            .await
            .iter()
            .map(|call| call.model.clone())
            .collect()
    }

    /// `/model` pins the model; clearing it hands selection back to the
    /// daemon's routing, as in the legacy loop.
    #[tokio::test]
    async fn sdk_engine_follows_daemon_model_selection() {
        use crate::runtime_ports::AgentIngress;
        let harness = models_harness().await;
        harness.agent.set_model("pinned-model".to_string()).await;
        harness
            .agent
            .handle_inbound_message(owner_request("hello"))
            .await
            .unwrap();
        harness.agent.clear_model_override().await;
        harness
            .agent
            .handle_inbound_message(owner_request("hello again"))
            .await
            .unwrap();
        assert_eq!(
            recorded_models(&harness).await,
            ["pinned-model", "primary-model"]
        );
    }

    #[tokio::test]
    async fn sdk_engine_records_llm_call_telemetry() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        let provider = MockProvider::with_responses(vec![
            MockProvider::tool_call_response(
                "remember_fact",
                r#"{"category":"preference","key":"tea","value":"green"}"#,
            ),
            MockProvider::text_response("Saved."),
        ]);
        let harness = crate::testing::setup_test_agent_sdk(provider)
            .await
            .unwrap();
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("Remember I like green tea."))
            .await
            .unwrap();
        // The run's own calls; bootstrap's assessment calls carry a purpose.
        let calls = task_events(&harness, &reply.task_id)
            .await
            .iter()
            .filter(|event| event.event_type == EventType::LlmCall)
            .filter(|event| event.data.get("call_purpose").is_none_or(|p| p.is_null()))
            .count();
        assert_eq!(calls, 2);
    }

    async fn task_events(
        harness: &crate::testing::TestHarness,
        task_id: &str,
    ) -> Vec<crate::events::Event> {
        harness
            .agent
            .event_store
            .query_task_events_for_session("telegram:synthetic-user-1", task_id)
            .await
            .unwrap()
    }

    fn remember(value: &str) -> crate::traits::ProviderResponse {
        crate::testing::MockProvider::tool_call_response(
            "remember_fact",
            &format!(r#"{{"category":"preference","key":"color","value":"{value}"}}"#),
        )
    }

    #[tokio::test]
    async fn sdk_engine_emits_tool_call_before_result() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        let provider =
            MockProvider::with_responses(vec![remember("teal"), MockProvider::text_response("ok")]);
        let harness = crate::testing::setup_test_agent_sdk(provider)
            .await
            .unwrap();
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("Remember teal."))
            .await
            .unwrap();
        let events = task_events(&harness, &reply.task_id).await;
        let kinds: Vec<EventType> = events
            .iter()
            .map(|event| event.event_type)
            .filter(|kind| matches!(kind, EventType::ToolCall | EventType::ToolResult))
            .collect();
        assert_eq!(kinds, [EventType::ToolCall, EventType::ToolResult]);
        let result = events
            .iter()
            .find(|event| event.event_type == EventType::ToolResult)
            .unwrap()
            .parse_data::<crate::events::ToolResultData>()
            .unwrap();
        let receipt = result.receipt.expect("tool result without a receipt");
        assert_eq!(
            receipt.invocation_stage,
            crate::traits::ToolInvocationStage::Dispatched
        );
        let aggregate = harness
            .agent
            .event_store
            .task_run_aggregate("telegram:synthetic-user-1", &reply.task_id)
            .await
            .unwrap();
        assert_eq!(aggregate.operations.len(), 1);
        assert!(
            aggregate.invariant_violations.is_empty(),
            "{:?}",
            aggregate.invariant_violations
        );
    }

    /// Review focus: a provider failure after tools ran keeps their records.
    #[tokio::test]
    async fn sdk_engine_keeps_tool_results_when_provider_fails() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        let provider = MockProvider::with_responses(vec![remember("teal")]).with_strict_responses();
        let harness = crate::testing::setup_test_agent_sdk(provider)
            .await
            .unwrap();
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("Remember teal."))
            .await
            .unwrap();
        let events = task_events(&harness, &reply.task_id).await;
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event_type == EventType::ToolResult)
                .count(),
            1
        );
        let end = events
            .iter()
            .find(|event| event.event_type == EventType::TaskEnd)
            .unwrap()
            .parse_data::<TaskEndData>()
            .unwrap();
        assert_eq!(end.status, TaskStatus::Failed);
    }

    #[tokio::test]
    async fn sdk_engine_track_requirements_declares_expectations() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        // Typed items become obligations; free-text items are only notes.
        let checklist = serde_json::json!({"items": [
            {"text": "read the first note", "status": "pending",
             "requires_observation": true, "targets": ["/tmp/synthetic/first.txt"]},
            {"text": "read the second note", "status": "pending",
             "requires_observation": true, "targets": ["/tmp/synthetic/second.txt"]}
        ]})
        .to_string();
        let provider = MockProvider::with_responses(vec![
            MockProvider::tool_call_response("track_requirements", &checklist),
            MockProvider::text_response("planned"),
            MockProvider::text_response("planned"),
            MockProvider::text_response("planned"),
        ]);
        let mut harness = crate::testing::setup_test_agent_sdk(provider)
            .await
            .unwrap();
        let plan_pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let plans = Arc::new(crate::plans::PlanStore::new(plan_pool).await.unwrap());
        harness.agent.tools.push(Arc::new(
            crate::tools::track_requirements::TrackRequirementsTool::new(plans),
        ));
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("Pick a color and save it."))
            .await
            .unwrap();
        let declared = task_events(&harness, &reply.task_id)
            .await
            .iter()
            .filter(|event| event.event_type == EventType::ExecutorExpectationsDeclared)
            .count();
        assert_eq!(declared, 1);
        let aggregate = harness
            .agent
            .event_store
            .task_run_aggregate("telegram:synthetic-user-1", &reply.task_id)
            .await
            .unwrap();
        assert_eq!(aggregate.open_executor_expectations(), 2);
    }

    /// Review focus: a cancelled parent or goal stops the SDK run before
    /// its next model call.
    #[tokio::test]
    async fn sdk_engine_stops_when_cancelled() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        let provider = MockProvider::with_responses(vec![MockProvider::text_response("late")]);
        let mut harness = crate::testing::setup_test_agent_sdk(provider)
            .await
            .unwrap();
        let token = tokio_util::sync::CancellationToken::new();
        harness.agent.cancel_token = Some(token.clone());
        token.cancel();
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("hello"))
            .await
            .unwrap();
        assert_eq!(reply.text, "Cancelled.");
        assert_eq!(harness.provider.call_count().await, 0);
        let end = task_events(&harness, &reply.task_id)
            .await
            .iter()
            .find(|event| event.event_type == EventType::TaskEnd)
            .unwrap()
            .parse_data::<TaskEndData>()
            .unwrap();
        assert_eq!(end.status, TaskStatus::Cancelled);
    }

    /// The legacy `executor_declared_checklist` integration test on the SDK
    /// engine: the model declares three typed items, does one and tries to
    /// stop; the ledger demands the open items until receipts close them.
    #[tokio::test]
    async fn sdk_engine_demands_open_checklist_items_until_receipts_close_them() {
        use crate::testing::MockProvider;
        use crate::types::ChannelVisibility;
        let workspace = tempfile::tempdir().expect("workspace");
        let cwd = workspace.path().to_string_lossy().to_string();
        let dir = workspace
            .path()
            .join("declared")
            .to_string_lossy()
            .to_string();
        let file = workspace
            .path()
            .join("declared/note.txt")
            .to_string_lossy()
            .to_string();
        let checklist_args = serde_json::json!({
            "items": [
                {"text": "create the directory", "status": "pending",
                 "mutation_effects": ["local_workspace_write"], "targets": [dir]},
                {"text": "write the note", "status": "pending",
                 "mutation_effects": ["local_workspace_write"], "targets": [file]},
                {"text": "read the note back", "status": "pending",
                 "requires_observation": true, "targets": [file]}
            ]
        })
        .to_string();
        let terminal = |command: String, key: &str, path: &str| {
            serde_json::json!({"command": command, "working_dir": cwd, key: [path]}).to_string()
        };
        let early = "phase=EARLY; directory created, stopping here.";
        let done = "phase=DONE; directory, note, and read-back all completed.";
        let provider = MockProvider::with_responses(vec![
            MockProvider::tool_call_response("track_requirements", &checklist_args),
            MockProvider::tool_call_response(
                "terminal",
                &terminal(format!("mkdir -p '{dir}'"), "write_paths", &dir),
            ),
            MockProvider::text_response(early),
            MockProvider::tool_call_response(
                "terminal",
                &terminal(
                    format!("printf SYNTHETIC_NOTE > '{file}'"),
                    "write_paths",
                    &file,
                ),
            ),
            MockProvider::tool_call_response(
                "terminal",
                &terminal(format!("cat '{file}'"), "read_paths", &file),
            ),
            MockProvider::text_response(done),
            MockProvider::text_response(done),
        ])
        // Assessor: nothing typed at all — no mutation, no observation.
        .with_task_assessments(vec![MockProvider::semantic_task_assessment(
            "check",
            false,
            false,
            &[],
            "new_request",
            "local_workspace",
        )]);
        let plan_pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let plans = Arc::new(crate::plans::PlanStore::new(plan_pool).await.unwrap());
        let track = Arc::new(crate::tools::track_requirements::TrackRequirementsTool::new(plans));
        let mut harness = crate::testing::setup_full_stack_test_agent_with_extra_tools(
            provider,
            vec![track as Arc<dyn crate::traits::Tool>],
        )
        .await
        .unwrap();
        harness
            .agent
            .set_sdk_engine(Some(SdkEngineSettings::default()));

        let session_id = "executor_declared_checklist";
        let response = harness
            .agent
            .handle_message(
                session_id,
                &format!(
                    "Create {dir}, write a note file inside it, then read it back to confirm."
                ),
                None,
                UserRole::Owner,
                ChannelContext {
                    visibility: ChannelVisibility::Private,
                    platform: "telegram".to_string(),
                    channel_name: None,
                    channel_id: None,
                    workspace_id: None,
                    sender_name: Some("Alice".to_string()),
                    sender_id: Some("telegram:synthetic-owner".to_string()),
                    channel_member_names: vec![],
                    user_id_map: std::collections::HashMap::new(),
                    workspace_grant: None,
                    trusted: false,
                },
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(&file).ok().as_deref(),
            Some("SYNTHETIC_NOTE"),
            "the declared write must have happened"
        );
        assert!(
            response.contains(done) && !response.contains(early),
            "the early stop must not ship: {response}"
        );
        let decision_rows: Vec<String> = sqlx::query_scalar(
            "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id",
        )
        .bind(session_id)
        .fetch_all(&harness.state.pool())
        .await
        .unwrap();
        let decisions: Vec<serde_json::Value> = decision_rows
            .iter()
            .filter_map(|raw| serde_json::from_str(raw).ok())
            .collect();
        assert!(decisions
            .iter()
            .any(|d| d["metadata"]["condition"] == "ledger_expectations_required"));
        let last = decisions
            .iter()
            .rfind(|d| d["metadata"]["condition"] == "ledger_closeout")
            .expect("ledger closeout recorded");
        assert_eq!(last["metadata"]["verdict"], "closed", "{last}");
    }

    #[test]
    fn closeout_demands_are_bounded_by_receipts() {
        let checklist = vec!["task:t/obligation:checklist:1".to_string()];
        let contract = vec!["task:t/obligation:evidence:0".to_string()];
        let facts = |reachable: &[String], completed, demanded_at, denial| StopFacts {
            reachable_obligation_ids: reachable.to_vec(),
            completed_results: completed,
            executor_demanded_at: demanded_at,
            terminal_policy_denial: denial,
        };
        // Open checklist items are demanded even after other work...
        assert!(closeout_demands_work(&facts(&checklist, 2, None, false)));
        // ...but only again once completed work changed since the last demand.
        assert!(!closeout_demands_work(&facts(
            &checklist,
            2,
            Some(2),
            false
        )));
        assert!(closeout_demands_work(&facts(&checklist, 3, Some(2), false)));
        // Contract expectations only while nothing has completed.
        assert!(closeout_demands_work(&facts(&contract, 0, None, false)));
        assert!(!closeout_demands_work(&facts(&contract, 1, None, false)));
        // Never after a terminal policy denial, never without reachable work.
        assert!(!closeout_demands_work(&facts(&checklist, 2, None, true)));
        assert!(!closeout_demands_work(&facts(&[], 0, None, false)));
    }

    /// An assessed contract restricting tools to `system_info`.
    fn restricted_assessment() -> crate::traits::ProviderResponse {
        let base = crate::testing::MockProvider::semantic_task_assessment(
            "answer",
            false,
            false,
            &[],
            "new_request",
            "none",
        );
        let mut value: serde_json::Value =
            serde_json::from_str(base.content.as_deref().unwrap()).unwrap();
        value["schema_version"] = 10.into();
        value["contract"]["tool_scope"] = "restricted".into();
        value["contract"]["allowed_tool_names"] = serde_json::json!(["system_info"]);
        value["contract"]["forbidden_tool_scopes"] = serde_json::json!([]);
        crate::testing::MockProvider::text_response(&value.to_string())
    }

    #[tokio::test]
    async fn sdk_engine_records_the_assessed_task_contract() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        let provider = MockProvider::with_responses(vec![MockProvider::text_response("hi")])
            .with_task_assessments(vec![MockProvider::semantic_task_assessment(
                "answer",
                false,
                false,
                &[],
                "new_request",
                "none",
            )]);
        let harness = crate::testing::setup_test_agent_sdk(provider)
            .await
            .unwrap();
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("Say hi."))
            .await
            .unwrap();
        let compiled = task_events(&harness, &reply.task_id)
            .await
            .iter()
            .filter(|event| event.event_type == EventType::TaskContractCompiled)
            .count();
        assert_eq!(compiled, 1);
        assert_eq!(reply.text, "hi");
    }

    #[tokio::test]
    async fn sdk_engine_refuses_tools_outside_the_contract_authority() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        use crate::traits::store_prelude::*;
        let provider = MockProvider::with_responses(vec![
            remember("teal"),
            MockProvider::text_response("I could not save it."),
        ])
        .with_task_assessments(vec![restricted_assessment()]);
        let harness = crate::testing::setup_test_agent_sdk(provider)
            .await
            .unwrap();
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("Only look things up; remember teal."))
            .await
            .unwrap();
        let facts = harness.state.get_facts(None).await.unwrap();
        assert!(
            !facts.iter().any(|fact| fact.value.contains("teal")),
            "a tool outside the contract authority ran"
        );
        let dispatched = task_events(&harness, &reply.task_id)
            .await
            .iter()
            .filter(|event| event.event_type == EventType::ToolResult)
            .filter_map(|event| event.parse_data::<crate::events::ToolResultData>().ok())
            .filter(|result| {
                result.receipt.as_ref().is_some_and(|receipt| {
                    receipt.invocation_stage == crate::traits::ToolInvocationStage::Dispatched
                })
            })
            .count();
        assert_eq!(dispatched, 0);
        assert_eq!(reply.text, "I could not save it.");
    }

    #[test]
    fn same_tool_streaks_follow_the_legacy_limits() {
        let calls = |tool: &str, n: usize, distinct: bool| -> Vec<(String, serde_json::Value)> {
            (0..n)
                .map(|i| {
                    let args = if distinct {
                        serde_json::json!({ "i": i })
                    } else {
                        serde_json::json!({})
                    };
                    (tool.to_string(), args)
                })
                .collect()
        };
        let same = serde_json::json!({});
        // Identical calls: the 8th is a loop.
        assert!(!same_tool_loop(
            &calls("remember_fact", 6, false),
            "remember_fact",
            &same
        ));
        assert!(same_tool_loop(
            &calls("remember_fact", 7, false),
            "remember_fact",
            &same
        ));
        // Diverse arguments earn the +8 bonus, not a bypass.
        let next = serde_json::json!({ "i": 99 });
        assert!(!same_tool_loop(
            &calls("remember_fact", 7, true),
            "remember_fact",
            &next
        ));
        assert!(same_tool_loop(
            &calls("remember_fact", 15, true),
            "remember_fact",
            &next
        ));
        // Versatile and research tools start at 16.
        assert!(!same_tool_loop(
            &calls("read_file", 14, true),
            "read_file",
            &next
        ));
        assert!(same_tool_loop(
            &calls("read_file", 15, true),
            "read_file",
            &next
        ));
        // computer_use is bounded at 40.
        assert!(!same_tool_loop(
            &calls("computer_use", 38, false),
            "computer_use",
            &same
        ));
        assert!(same_tool_loop(
            &calls("computer_use", 39, false),
            "computer_use",
            &same
        ));
        // Another tool breaks the streak.
        let mut mixed = calls("remember_fact", 7, false);
        mixed.push(("system_info".to_string(), same.clone()));
        assert!(!same_tool_loop(&mixed, "remember_fact", &same));
    }

    #[test]
    fn goal_budget_exhaustion_uses_daily_and_run_budgets() {
        use crate::traits::GoalTokenBudgetStatus;
        let daily = |used, budget| GoalTokenBudgetStatus {
            budget_per_check: None,
            budget_daily: budget,
            tokens_used_today: used,
        };
        let run = |used, budget| crate::goal_tokens::GoalRunBudgetStatus {
            effective_budget_per_check: budget,
            tokens_used: used,
            budget_extensions_count: 0,
            health: Default::default(),
        };
        assert!(!goal_budget_exhausted(Some(&daily(10, Some(100))), None));
        assert!(goal_budget_exhausted(Some(&daily(100, Some(100))), None));
        assert!(!goal_budget_exhausted(Some(&daily(10_000, None)), None));
        assert!(goal_budget_exhausted(
            Some(&daily(10, Some(100))),
            Some(&run(50, 50))
        ));
        // A scheduled run's per-run budget replaces the daily one.
        assert!(!goal_budget_exhausted(
            Some(&daily(1_000, Some(100))),
            Some(&run(10, 50))
        ));
        assert!(!goal_budget_exhausted(None, Some(&run(10, 0))));
    }

    #[test]
    fn interrupted_and_failed_runs_map_to_non_success_outcomes() {
        let mut outcome = sdk::RunOutcome {
            status: sdk::RunStatus::ProviderFailed {
                error: "overloaded".into(),
            },
            final_text: None,
            state: sdk::RunState::new("aidaemon", "s", "hi"),
        };
        let (reply, status, task_outcome) = reply_for(&outcome);
        assert!(reply.contains("overloaded"));
        assert_eq!(status, TaskStatus::Failed);
        assert_eq!(task_outcome, TaskOutcome::Failed);

        outcome.status = sdk::RunStatus::Completed;
        outcome.final_text = Some("All set.".into());
        let (reply, status, task_outcome) = reply_for(&outcome);
        assert_eq!(reply, "All set.");
        assert_eq!(status, TaskStatus::Completed);
        assert_eq!(task_outcome, TaskOutcome::Succeeded);
    }
}
