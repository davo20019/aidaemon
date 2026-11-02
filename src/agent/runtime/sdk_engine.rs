//! Agent turns on the aidaemon SDK kernel (`crates/aidaemon-core`).
//!
//! Every path that reaches `handle_message_impl` (channel ingress, internal
//! continuations, goal task leads, spawned children) runs one SDK run per
//! turn.
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
//! Not ported, on purpose: the legacy intent gate, pre-execution plan /
//! critique / evidence gates and the guided `tool_choice` forcing ladder.
//! Completion is the ledger closeout (`LedgerStop`), not those gates.

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
use super::memory_plugin::{daemon_tool_offered, memory_parts, MemoryTurn};
use super::sdk_provider::{
    fit_run, from_provider_response, prefix_with_notes, split_injected_notes, to_chat_options,
    to_daemon_tools, values_to_sdk, DaemonProvider, ModelCall,
};
use super::tool_exec::ToolExecCtx;
use super::Agent;
use crate::agent::vision::encode_image_attachment;
use crate::config::{AppConfig, ProviderKind, VisionConfig};
use crate::events::{EventEmitter, EventType, TaskOutcome, TaskStatus};
use crate::traits::{AttachmentProvenance, Message, MessageAttachment};
use crate::types::{ChannelContext, StatusUpdate, UserRole};

const DEFAULT_MAX_TURNS: u32 = 40;
/// Daemon tools enforce their own timeouts (terminal backgrounding, CLI
/// agents); this only stops a wedged call from holding the turn forever.
const TOOL_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const HEARTBEAT_TICK: Duration = Duration::from_secs(15);
/// The run ended on a terminal tool result (background handoff, executor
/// blocker, direct response) whose reply ships as is.
const TERMINAL_REPLY_STOP: &str = "terminal tool result";
/// Model-call timeouts, as in the legacy LLM phase.
const DEFAULT_MODEL_CALL_TIMEOUT: Duration = Duration::from_secs(360);
const FOREGROUND_MAX_MODEL_CALL: Duration = Duration::from_secs(90);
const MANDATE_MAX_MODEL_CALL: Duration = Duration::from_secs(840);
/// A timed-out call is retried, as the legacy loop continued after one: the
/// ceiling bounds a call so the turn regains control, not so it ends.
const MAX_MODEL_CALL_TIMEOUT_RETRIES: u32 = 2;

/// A model call that ran past its per-call ceiling.
#[derive(Debug)]
struct ModelCallTimedOut(Duration);

impl std::fmt::Display for ModelCallTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "model call timed out after {}s", self.0.as_secs())
    }
}

impl std::error::Error for ModelCallTimedOut {}
const GOAL_BUDGET_EXHAUSTED: &str = "the goal's token budget is used up";
/// Budget extension limits, as in the legacy loop.
const TASK_MAX_BUDGET_EXTENSIONS: usize = 3;
const TASK_HARD_TOKEN_CAP: u64 = 2_000_000;
const SCHEDULED_MAX_BUDGET_EXTENSIONS: usize = 1;
const SCHEDULED_HARD_TOKEN_CAP: i64 = 2_000_000;
const MANDATE_BUDGET_EXHAUSTED: &str = "the mandate cycle's token balance is exhausted";

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

/// The engine settings from `[agent.engine]`.
pub(crate) fn settings_from_config(config: &AppConfig) -> SdkEngineSettings {
    let engine = &config.agent.engine;
    if engine.kind.is_some() {
        warn!("[agent.engine] kind is ignored: the SDK kernel is the only agent loop");
    }
    let mut model_override = engine
        .model
        .clone()
        .filter(|model| !model.trim().is_empty());
    if !engine.api_key.trim().is_empty() {
        warn!(
            "[agent.engine] api_key is ignored: the SDK engine uses the daemon's configured \
             providers"
        );
        // An engine api_key marks a config written for the engine's former
        // dedicated Anthropic client; its model pin named a model for that
        // client, which the daemon's own provider may not serve.
        if config.provider.kind != ProviderKind::Anthropic {
            if let Some(model) = model_override.take() {
                warn!(
                    model = %model,
                    "[agent.engine] model is ignored: it was configured for the removed \
                     Anthropic engine client; set it without api_key to pin a model the \
                     daemon's provider serves"
                );
            }
        }
    }
    let settings = SdkEngineSettings {
        model_override,
        max_turns: engine.max_turns.unwrap_or(DEFAULT_MAX_TURNS).max(1),
    };
    info!(model_override = ?settings.model_override, "Agent engine: SDK kernel");
    settings
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
    /// The transcript the legacy message builder projected for the turn's
    /// first model call, in the daemon providers' wire format.
    pub(in crate::agent) prefix: Vec<Value>,
    pub(in crate::agent) contract: crate::agent::completion_contract::CompletionContract,
    /// Post-task learning state from bootstrap; the run adds its tool calls.
    pub(in crate::agent) learning_ctx: crate::agent::post_task::LearningContext,
    /// The assessed task shape continues inline work after a background start.
    pub(in crate::agent) continue_inline_after_background: bool,
    /// Matched skills reference untrusted external content: the legacy tool
    /// block applies, naming these skills.
    pub(in crate::agent) restrict_untrusted_reference_tools: bool,
    pub(in crate::agent) untrusted_reference_skills: Vec<String>,
    /// Discovers nested project instructions before deeper subtree work.
    pub(in crate::agent) instruction_tracker:
        Option<crate::project_instructions::ProjectInstructionTracker>,
    /// The compiled memory policy allows recall on this turn.
    pub(in crate::agent) memory_recall_allowed: bool,
}

/// The model wants to stop; the dispatcher asks the run ledger.
pub(super) struct StopCheck {
    /// Operations of this run that completed.
    completed_results: usize,
    /// The run's latest tool call failed and none succeeded.
    failed_without_success: bool,
    final_text: String,
    reply: oneshot::Sender<sdk::StopDecision>,
}

/// A replayed call's result, marked so the model does not read the earlier
/// success as something that just happened (a reminder re-"set" after the
/// user had it cancelled is still cancelled).
fn replay_notice(previous_result: &str) -> String {
    format!(
        "[Not run again: an identical call already ran earlier in this task; its earlier result is \
         repeated below. Nothing new happened now, and anything done since then (for example \
         cancelling what it created) still stands.]\n\n{previous_result}"
    )
}

/// The ledger closeout as an SDK stop hook: before a final answer ships, the
/// run ledger decides whether a still-reachable expectation needs another
/// pass. Receipts, not the reply text, are the evidence.
struct LedgerStop {
    calls: mpsc::Sender<DaemonCall>,
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
            text.unwrap_or_default(),
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

/// One tool call as post-task learning records it (the legacy
/// `tool_name(summary)` line, marked when it was rejected or failed so
/// activity summaries never claim work that did not happen).
fn learning_tool_summary(name: &str, arguments: &str, rejected: bool, failed: bool) -> String {
    let line = format!(
        "{name}({})",
        crate::agent::agent_helpers::summarize_tool_args(name, arguments)
    );
    if rejected {
        format!("{line} [REJECTED]")
    } else if failed {
        format!("{line} [FAILED]")
    } else {
        line
    }
}

/// A final reply that must not ship as is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplyDefect {
    /// No text at all.
    Empty,
    /// Only internal markers or tool envelopes; sanitizing leaves nothing.
    InternalMarkers,
    /// A pasted file page, raw data dump, or a copy of the latest tool output.
    ToolOutputPaste,
}

/// The legacy response-phase shape checks, all structural (sanitizer
/// output, harness page headers, data-dump shape, overlap with the latest
/// tool output), never wording.
fn reply_defect(text: &str, latest_tool_output: Option<&str>) -> Option<ReplyDefect> {
    if text.trim().is_empty() {
        return Some(ReplyDefect::Empty);
    }
    let sanitized = crate::tools::sanitize::sanitize_user_facing_reply(text);
    if crate::tools::sanitize::reply_gutted_by_sanitization(text.trim().chars().count(), &sanitized)
        || crate::agent::task_outcome::response_looks_like_plain_text_tool_call(text)
        || crate::agent::response_analysis::reply_has_protocol_markers(text)
    {
        return Some(ReplyDefect::InternalMarkers);
    }
    let pasted = crate::agent::response_analysis::reply_is_pasted_file_page(&sanitized)
        || crate::agent::response_analysis::reply_is_raw_data_dump(&sanitized)
        || latest_tool_output.is_some_and(|output| {
            crate::agent::response_analysis::reply_duplicates_tool_output(&sanitized, output)
        });
    pasted.then_some(ReplyDefect::ToolOutputPaste)
}

/// The reply that ships for a completed run: sanitized, and an honest
/// fallback instead of an empty or gutted answer (never a bare "Done.").
/// After tool work the fallback describes that work: the receipts, not a
/// silent model, say whether anything happened.
fn finalize_reply(text: &str, tool_calls: &[String], response_note: Option<&str>) -> String {
    if text.trim().is_empty() {
        if !tool_calls.is_empty() {
            let refs: Vec<&str> = tool_calls.iter().map(String::as_str).collect();
            return super::completion_checks::build_activity_summary_reply(&refs);
        }
        return crate::agent::agent_helpers::build_empty_response_fallback(response_note);
    }
    let sanitized = crate::tools::sanitize::sanitize_user_facing_reply(text);
    if !crate::tools::sanitize::reply_gutted_by_sanitization(
        text.trim().chars().count(),
        &sanitized,
    ) {
        return sanitized;
    }
    if tool_calls.is_empty() {
        "I ran into a formatting problem composing that answer and couldn't recover it. Ask me \
         again and I'll re-gather the details."
            .to_string()
    } else {
        let refs: Vec<&str> = tool_calls.iter().map(String::as_str).collect();
        super::completion_checks::build_activity_summary_reply(&refs)
    }
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
    /// A reachable observation obligation remains after a successful change
    /// and has not been demanded yet (the owner's verify-once rule).
    verification_after_mutation: bool,
    /// The run's latest call failed and nothing succeeded, and its recovery
    /// pass has not been used.
    unrecovered_failure: bool,
}

/// The legacy completion gate (`completion_phase.rs`), read from receipts:
/// contract expectations are demanded only while nothing has completed; the
/// executor's own open checklist items are demanded again whenever completed
/// work changed since the last demand; nothing is demanded after a terminal
/// policy denial.
fn closeout_demands_work(facts: &StopFacts) -> bool {
    if facts.terminal_policy_denial {
        return false;
    }
    if facts.unrecovered_failure && facts.completed_results == 0 {
        return true;
    }
    if facts.reachable_obligation_ids.is_empty() {
        return false;
    }
    if facts.verification_after_mutation {
        return true;
    }
    let executor_demands = facts
        .reachable_obligation_ids
        .iter()
        .any(|id| id.contains("/obligation:checklist:"))
        && facts.executor_demanded_at != Some(facts.completed_results);
    facts.completed_results == 0 || executor_demands
}

/// What a run settled, beyond its kernel status, that decides the turn's
/// final status and outcome. Shared with run-end hooks (memory writers), so
/// they judge a run exactly as the turn does.
#[derive(Default)]
pub(in crate::agent) struct TurnVerdict {
    /// The neutral handoff reply once work moved to the background.
    terminal_reply: std::sync::Mutex<Option<(String, TaskOutcome)>>,
    /// Confirmation of the latest successful external change.
    external_action_ack: std::sync::Mutex<Option<String>>,
    /// Set when the run must end failed despite a final answer.
    run_failure: std::sync::Mutex<Option<String>>,
}

impl TurnVerdict {
    fn run_failure(&self) -> Option<String> {
        self.run_failure
            .lock()
            .ok()
            .and_then(|failure| failure.clone())
    }

    /// The turn's reply, status and outcome for a finished run.
    pub(in crate::agent) fn resolve(
        &self,
        outcome: &sdk::RunOutcome,
    ) -> (String, TaskStatus, TaskOutcome) {
        let (reply, status, task_outcome) = reply_for(outcome);
        let external_ack = self
            .external_action_ack
            .lock()
            .ok()
            .and_then(|ack| ack.clone());
        let terminal = self
            .terminal_reply
            .lock()
            .ok()
            .and_then(|reply| reply.clone());
        let (reply, status, task_outcome) = match (&outcome.status, terminal) {
            (sdk::RunStatus::Stopped { reason }, Some((reply, outcome)))
                if reason == TERMINAL_REPLY_STOP =>
            {
                (reply, TaskStatus::Completed, outcome)
            }
            // A budget stop ends the run through the provider port.
            (sdk::RunStatus::ProviderFailed { error }, Some((reply, outcome)))
                if error.contains(GOAL_BUDGET_EXHAUSTED) =>
            {
                (reply, TaskStatus::Completed, outcome)
            }
            _ => (reply, status, task_outcome),
        };
        let (reply, status, task_outcome) = match (&outcome.status, external_ack) {
            (
                sdk::RunStatus::ProviderFailed { .. } | sdk::RunStatus::LimitReached { .. },
                Some(ack),
            ) => (
                super::tool_execution_phase::user_facing_external_action_ack(&ack),
                TaskStatus::Completed,
                TaskOutcome::Succeeded,
            ),
            _ => (reply, status, task_outcome),
        };
        if self.run_failure().is_some() {
            (reply, TaskStatus::Failed, TaskOutcome::Failed)
        } else {
            (reply, status, task_outcome)
        }
    }
}

impl TurnScope<'_> {
    /// End the run before the next model call and ship `reply`.
    fn end_with(&self, reply: String, outcome: TaskOutcome) {
        if let Ok(mut terminal) = self.verdict.terminal_reply.lock() {
            *terminal = Some((reply, outcome));
        }
        self.end_run.store(true, Ordering::Relaxed);
    }
}

/// Everything a dispatched call needs from the turn.
struct TurnScope<'a> {
    session_id: &'a str,
    /// The user's message for this turn.
    user_text: &'a str,
    task_id: &'a str,
    emitter: &'a EventEmitter,
    /// The projected starting transcript, sent byte-for-byte on every call.
    prefix: Vec<Value>,
    /// Kernel messages that stand for `prefix`; later ones are this run's.
    prefix_len: usize,
    /// Leading system messages of that starting transcript; any more in a
    /// request were injected by a hook.
    prefix_system_lead: usize,
    budget_goal: Option<BudgetGoal>,
    /// Set once the goal budget is used up; later model calls are refused.
    budget_exhausted: AtomicBool,
    model_calls: AtomicU64,
    /// The verify-once and failure-recovery passes have been used.
    verification_demanded: AtomicBool,
    recovery_demanded: AtomicBool,
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
    /// Shared with the memory writer, which learns from it at run end.
    learning: Arc<std::sync::Mutex<crate::agent::post_task::LearningContext>>,
    /// The model-visible output of the latest tool call, for paste checks.
    latest_tool_output: std::sync::Mutex<Option<String>>,
    /// The one reply retry has been spent.
    reply_retry_used: AtomicBool,
    /// The next model call runs with tools disabled (empty-reply retry).
    text_only_next_call: AtomicBool,
    /// The assessed task shape keeps inline work going after a detach.
    continue_inline_after_background: bool,
    /// Terminal replies, external acks and run failures: the turn's verdict.
    verdict: Arc<TurnVerdict>,
    /// Tool calls in the latest model response (a direct response needs one).
    last_batch_size: std::sync::atomic::AtomicUsize,
    /// Ends the run at the next turn boundary (a background handoff).
    end_run: Arc<AtomicBool>,
    /// Skills referencing untrusted external content are active.
    restrict_untrusted_reference_tools: bool,
    untrusted_reference_skills: Vec<String>,
    /// Billable tokens this run spent, and the task budget (0: none).
    task_tokens: AtomicU64,
    task_budget: AtomicU64,
    task_budget_extensions: std::sync::atomic::AtomicUsize,
    /// Distinct successful operations (tool and canonical arguments).
    succeeded_operations: std::sync::Mutex<HashSet<String>>,
    /// The latest successful call repeated an earlier operation.
    repeating: AtomicBool,
    /// Distinct successful operations so far, and the count at the last
    /// extension.
    succeeded_calls: std::sync::atomic::AtomicUsize,
    successes_at_task_extension: std::sync::atomic::AtomicUsize,
    /// The goal's effective daily budget after extensions, and their count.
    goal_daily_budget: std::sync::Mutex<Option<i64>>,
    goal_budget_extensions: std::sync::atomic::AtomicU32,
    /// Mandate decision correction turns used.
    mandate_commit_retries: std::sync::atomic::AtomicU32,
    /// Timed-out model calls retried in this run.
    model_call_timeouts: std::sync::atomic::AtomicU32,
    /// `send_file` dedupe keys that already succeeded in this task.
    sent_files: std::sync::Mutex<HashSet<String>>,
    /// Nested project instructions discovered just in time.
    instruction_tracker:
        Option<tokio::sync::Mutex<crate::project_instructions::ProjectInstructionTracker>>,
    /// System context added during the run, sent after the projected prefix.
    extra_system: std::sync::Mutex<Vec<String>>,
    /// The latest provider note (for example a safety finish reason).
    response_note: std::sync::Mutex<Option<String>>,
    /// Set when a call moved work to the background with completion
    /// notifications on: the daemon will deliver its result later, so the
    /// reply is a background handoff, not the terminal answer.
    background_handoff: AtomicBool,
    /// The latest well-formed final answer a closeout demand sent back for
    /// more work. If the model then replies with nothing, this ships instead
    /// of a generic activity summary.
    interrupted_answer: std::sync::Mutex<Option<String>>,
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
            mutation_forbidden: scope.contract.authority().forbids_mutation,
            scope_escalation: None,
        };
        let started = Instant::now();
        let call_semantics = self
            .semantics_for_exact_tool_call(&request.name, arguments)
            .await;
        // The contract's authority lane (forbidden tool use or mutation,
        // allowed tools, forbidden scopes), then for mandate runs the same
        // per-call authority gate as the legacy loop, which issues the
        // bounded grant or refuses the call.
        // Judged per call: a mixed tool such as `terminal` is not read-only
        // as a whole, but a call its adapter types as a pure observation is
        // (the tool still enforces `mutation_forbidden` itself).
        let read_only = self
            .tools
            .iter()
            .find(|tool| tool.name() == request.name)
            .map(|tool| {
                tool.capabilities().read_only
                    || (call_semantics.effect == crate::traits::ToolCallEffect::Observation
                        && !call_semantics.mutates_state())
            });
        let admitted =
            super::closeout::ClosingAuthority::from_authority(scope.contract.authority())
                .admits_tool(&request.name, read_only);
        // A matched skill that references untrusted external content keeps
        // the legacy block on tools that could act on it.
        let untrusted_block = scope.restrict_untrusted_reference_tools
            && crate::agent::is_untrusted_external_reference_blocked_tool(&request.name);
        let gate = if untrusted_block {
            Err((
                super::tool_execution_phase::blocked_for_untrusted_external_reference_message(
                    &request.name,
                    &scope.untrusted_reference_skills,
                ),
                "untrusted_external_reference_blocked",
            ))
        } else if admitted {
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
                if reason_code == "request_authority" && read_only != Some(true) {
                    self.with_harness_eval(|eval| eval.record_forbidden_mutation_attempt())
                        .await;
                }
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

        // Before a call enters a deeper subtree, newly applicable nested
        // project instructions are loaded and the call is deliberately not
        // executed, so the model reviews them and retries (CLAUDE.md,
        // "Project Instruction Loading").
        if let Some(deferral) = self
            .sdk_load_nested_instructions(scope, request, arguments, &call_semantics)
            .await
        {
            return deferral;
        }

        // The same file and caption go out once per task.
        let send_file_key = (request.name == "send_file")
            .then(|| super::loop_utils::extract_send_file_dedupe_key_from_args(arguments))
            .flatten();
        if let Some(key) = &send_file_key {
            let already_sent = scope.sent_files.lock().is_ok_and(|sent| sent.contains(key));
            if already_sent {
                return self.sdk_suppress_duplicate_send(scope, request).await;
            }
        }
        // Calls with side effects carry the legacy idempotency key, bound to
        // the task and the canonical arguments: a repeat reuses the durable
        // receipt instead of performing the effect twice.
        let idempotency_key = self.sdk_idempotency_key(scope, request, &call_semantics);
        let preflight = super::tool_execution_phase::durable_replay_preflight(
            self,
            scope.session_id,
            &request.name,
            arguments,
            idempotency_key.as_deref(),
        )
        .await;
        if let Some(output) = self
            .sdk_replay_or_block(scope, request, arguments, &preflight)
            .await
        {
            return output;
        }
        let mut call_data = crate::events::ToolCallData::from_tool_call(
            request.call_id.clone(),
            request.name.clone(),
            request.arguments.clone(),
            Some(scope.task_id.to_string()),
        )
        .with_policy_metadata(idempotency_key.clone(), None, None);
        call_data.turn_id = Some(scope.turn_id.to_string());
        if let Err(error) = scope.emitter.emit(EventType::ToolCall, call_data).await {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a tool call");
        }

        // User-facing activity labels, as the legacy tool phase sends them;
        // `track_requirements` shows its checklist instead.
        if request.name != "track_requirements" {
            let (name, summary) = crate::tools::sanitize::user_facing_tool_activity(
                &request.name,
                &crate::agent::agent_helpers::summarize_tool_args(&request.name, arguments),
                scope.channel_ctx.visibility,
            );
            if let Some(status_tx) = &scope.status_tx {
                let _ = status_tx.try_send(StatusUpdate::ToolStart { name, summary });
            }
        }
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
                    idempotency_key.clone(),
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
                // Tool-boundary truncation is rendered into the model-visible
                // text only, as in the legacy tool phase.
                if let Some(info) = outcome.metadata.truncation.as_ref() {
                    output.push('\n');
                    output.push_str(&crate::utils::render_truncation_notice(info));
                }
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
                // The legacy detach transition, decided at the typed receipt:
                // detached work is the handoff unless the assessed task shape
                // keeps independent inline work going.
                if super::tool_execution_phase::tool_result_indicates_background_detach(
                    &request.name,
                    &output,
                    &outcome.metadata,
                ) {
                    let directive = if scope.continue_inline_after_background {
                        super::system_directives::SystemDirective::BackgroundProcessContinue
                    } else {
                        // With notifications on, the result follows by
                        // itself: a plain handoff, no pid or tool jargon.
                        // Without them, the ack says how to check on it.
                        let reply = if outcome.metadata.completion_notifications_enabled {
                            super::post_task::build_friendly_background_handoff(scope.user_text)
                        } else {
                            super::tool_execution_phase::build_background_detach_ack(
                                &request.name,
                                &output,
                                &outcome.metadata,
                            )
                        };
                        scope.end_with(reply, TaskOutcome::Partial);
                        super::system_directives::SystemDirective::BackgroundHandoff {
                            notifications_active: outcome.metadata.completion_notifications_enabled,
                        }
                    };
                    if outcome.metadata.completion_notifications_enabled {
                        scope.background_handoff.store(true, Ordering::Relaxed);
                    }
                    output = format!("{output}\n\n{}", directive.render());
                }
                let satisfied = !status.is_failure();
                // An executor's typed blocker ends its run with the structured
                // summary; the untrusted-data wrapper is removed first so
                // reply sanitization does not empty it.
                if satisfied && request.name == "report_blocker" && self.task_id.is_some() {
                    scope.end_with(
                        super::completion_checks::strip_untrusted_wrapper(&output).to_string(),
                        TaskOutcome::Partial,
                    );
                }
                // A tool's direct response answers a root turn's lone call.
                if satisfied
                    && self.depth == 0
                    && scope.last_batch_size.load(Ordering::Relaxed) == 1
                    && !(outcome.metadata.background_started || outcome.metadata.detached)
                {
                    if let Some(reply) = outcome.metadata.direct_response.clone() {
                        scope.end_with(reply, TaskOutcome::Succeeded);
                    }
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
        if let Ok(mut latest) = scope.latest_tool_output.lock() {
            *latest = Some(output.clone());
        }
        // Budget progress counts distinct successful operations: repeating
        // an identical call is not new evidence.
        if !failed {
            let operation = format!(
                "{}:{}",
                request.name,
                crate::agent::prefix_fingerprint::hash_canonical(&request.arguments)
            );
            if let Ok(mut seen) = scope.succeeded_operations.lock() {
                let new_operation = seen.insert(operation);
                if new_operation {
                    scope.succeeded_calls.fetch_add(1, Ordering::Relaxed);
                }
                scope.repeating.store(!new_operation, Ordering::Relaxed);
            }
        }
        if let (Some(key), false) = (send_file_key, failed) {
            if let Ok(mut sent) = scope.sent_files.lock() {
                sent.insert(key);
            }
        }
        if let Some(status_tx) = &scope.status_tx {
            let update = if request.name == "track_requirements" {
                super::tool_execution_phase::parse_checklist_from_result(&output)
                    .map(|text| StatusUpdate::Checklist { text })
            } else {
                let source = if matches!(request.name.as_str(), "manage_memories" | "manage_people")
                {
                    crate::agent::agent_helpers::summarize_tool_args(&request.name, arguments)
                } else {
                    super::tool_execution_phase::summarize_completed_tool_result(&output)
                };
                let (name, summary) = crate::tools::sanitize::user_facing_tool_activity(
                    &request.name,
                    &source,
                    scope.channel_ctx.visibility,
                );
                Some(StatusUpdate::ToolComplete { name, summary })
            };
            if let Some(update) = update {
                let _ = status_tx.try_send(update);
            }
        }
        // A receipt-backed confirmation of an external change, delivered if
        // the model cannot be reached afterwards.
        let external_mutation = metadata.as_ref().is_some_and(|metadata| {
            metadata.semantics.mutates_state()
                && !(metadata.background_started || metadata.detached)
        }) && self
            .tools
            .iter()
            .find(|tool| tool.name() == request.name)
            .is_some_and(|tool| tool.capabilities().external_side_effect);
        if external_mutation {
            if let Ok(mut ack) = scope.verdict.external_action_ack.lock() {
                *ack = (!failed
                    && super::tool_execution_phase::should_build_external_action_ack(&output))
                .then(|| {
                    super::tool_execution_phase::build_external_action_completion_ack(&output)
                });
            }
        }
        if let Ok(mut learning) = scope.learning.lock() {
            let rejected = metadata
                .as_ref()
                .is_some_and(|metadata| metadata.contract_rejected);
            learning.tool_calls.push(learning_tool_summary(
                &request.name,
                arguments,
                rejected,
                failed,
            ));
        }

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
                // Mandate receipts belong to the worker task, as in the legacy
                // tool phase.
                Some(
                    self.mandate_execution
                        .as_ref()
                        .map_or(scope.task_id, |fence| fence.worker_task_id.as_str()),
                ),
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
        let mut decision = self.sdk_mandate_commit_check(scope).await;
        if decision == sdk::StopDecision::Accept {
            decision = self
                .sdk_closeout(scope, check.completed_results, check.failed_without_success)
                .await;
            if decision != sdk::StopDecision::Accept
                && reply_defect(&check.final_text, None).is_none()
            {
                if let Ok(mut interrupted) = scope.interrupted_answer.lock() {
                    *interrupted = Some(check.final_text.clone());
                }
            }
        }
        if decision == sdk::StopDecision::Accept {
            decision = self.sdk_reply_check(scope, &check.final_text);
        }
        let _ = check.reply.send(decision);
    }

    /// A mandate task lead's prose is never its decision: a missing typed
    /// commit gets one bounded correction turn, then the run fails closed,
    /// as in the legacy completion phase.
    async fn sdk_mandate_commit_check(&self, scope: &TurnScope<'_>) -> sdk::StopDecision {
        use super::completion_phase::{
            mandate_decision_commit_state, mandate_decision_completion_action,
            MandateDecisionCommitState, MandateDecisionCompletionAction,
        };
        let state = match mandate_decision_commit_state(self).await {
            Ok(state) => state,
            Err(error) => {
                warn!(task_id = scope.task_id, %error, "SDK engine: mandate decision lookup failed");
                return sdk::StopDecision::Accept;
            }
        };
        let decision_tool_available = self
            .tools
            .iter()
            .any(|tool| tool.name() == "manage_mandates");
        let retries = scope.mandate_commit_retries.load(Ordering::Relaxed) as usize;
        match mandate_decision_completion_action(state, retries, decision_tool_available) {
            MandateDecisionCompletionAction::Allow => sdk::StopDecision::Accept,
            MandateDecisionCompletionAction::Retry => {
                scope.mandate_commit_retries.fetch_add(1, Ordering::Relaxed);
                self.emit_warning_decision_point(
                    scope.emitter,
                    scope.task_id,
                    scope.model_calls.load(Ordering::Relaxed) as usize,
                    crate::events::DecisionType::PostExecutionValidation,
                    "Blocked mandate prose completion without a durable decision".to_string(),
                    serde_json::json!({
                        "condition": "mandate_decision_commit_missing",
                        "retry": retries + 1,
                        "engine": "sdk",
                    }),
                )
                .await;
                sdk::StopDecision::Continue {
                    message:
                        super::system_directives::SystemDirective::MandateDecisionCommitRequired
                            .render(),
                }
            }
            MandateDecisionCompletionAction::Reject => {
                let error = if state == MandateDecisionCommitState::Invalid {
                    "Mandate deliberator found a decision for the current run with mismatched \
                     immutable identity."
                } else {
                    "Mandate deliberator returned without committing an exact durable decision \
                     after its bounded correction turn."
                };
                if let Ok(mut failure) = scope.verdict.run_failure.lock() {
                    *failure = Some(error.to_string());
                }
                sdk::StopDecision::Accept
            }
        }
    }

    /// One shared retry for a reply that must not ship (empty, only internal
    /// markers, pasted tool output); after it, the finalizer's honest
    /// fallback applies.
    fn sdk_reply_check(&self, scope: &TurnScope<'_>, final_text: &str) -> sdk::StopDecision {
        let latest = scope
            .latest_tool_output
            .lock()
            .ok()
            .and_then(|latest| latest.clone());
        let Some(defect) = reply_defect(final_text, latest.as_deref()) else {
            return sdk::StopDecision::Accept;
        };
        if scope.reply_retry_used.swap(true, Ordering::Relaxed) {
            return sdk::StopDecision::Accept;
        }
        warn!(
            session_id = scope.session_id,
            task_id = scope.task_id,
            ?defect,
            "SDK engine: final reply rejected; retrying once"
        );
        let directive = match defect {
            // After real work an empty reply is the model treating the run as
            // finished; offering tools again invites another empty turn, so
            // the retry must answer in text from the receipts it holds.
            ReplyDefect::Empty if scope.succeeded_calls.load(Ordering::Relaxed) > 0 => {
                scope.text_only_next_call.store(true, Ordering::Relaxed);
                super::system_directives::SystemDirective::EmptyFinalReplyPlainText
            }
            ReplyDefect::Empty => super::system_directives::SystemDirective::EmptyResponseRetry,
            ReplyDefect::InternalMarkers => {
                super::system_directives::SystemDirective::FinalAnswerRejectedInternalMarkers
            }
            ReplyDefect::ToolOutputPaste => {
                super::system_directives::SystemDirective::FinalAnswerWasFilePaste
            }
        };
        sdk::StopDecision::Continue {
            message: directive.render(),
        }
    }

    /// The legacy ledger-first closeout on the SDK path: replay the run
    /// ledger, ask which expectations an admissible tool could still satisfy,
    /// and demand a bounded extra pass only when the receipts say so.
    async fn sdk_closeout(
        &self,
        scope: &TurnScope<'_>,
        completed_results: usize,
        failed_without_success: bool,
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
        let mutation_succeeded = aggregate.operations.values().any(|operation| {
            operation.mutating
                && operation.outcome == Some(crate::traits::ToolOutcomeStatus::Succeeded)
        });
        let observation_reachable = reachable.iter().any(|id| {
            aggregate.obligations.get(id).is_some_and(|obligation| {
                obligation.class == crate::events::RunObligationClass::Observe
            })
        });
        let verification_after_mutation = mutation_succeeded
            && observation_reachable
            && !scope.verification_demanded.load(Ordering::Relaxed);
        // A failure that is itself the expected observation satisfies its
        // obligation; only an uncredited failure earns the recovery pass.
        let failure_credited = aggregate
            .obligations
            .values()
            .any(|obligation| obligation.state == crate::events::RunObligationState::Satisfied);
        let unrecovered_failure = failed_without_success
            && !failure_credited
            && !scope.recovery_demanded.load(Ordering::Relaxed);
        let facts = StopFacts {
            reachable_obligation_ids: reachable.clone(),
            completed_results,
            executor_demanded_at: (demanded_at != u64::MAX).then_some(demanded_at as usize),
            terminal_policy_denial: aggregate.terminal_policy_denial(),
            verification_after_mutation,
            unrecovered_failure,
        };
        if !closeout_demands_work(&facts)
            || scope.closeout_demands.load(Ordering::Relaxed) >= MAX_CLOSEOUT_DEMANDS
        {
            return sdk::StopDecision::Accept;
        }
        let demands = scope.closeout_demands.fetch_add(1, Ordering::Relaxed) + 1;
        if verification_after_mutation {
            scope.verification_demanded.store(true, Ordering::Relaxed);
        }
        if unrecovered_failure && reachable.is_empty() {
            scope.recovery_demanded.store(true, Ordering::Relaxed);
            let evidence = scope
                .latest_tool_output
                .lock()
                .ok()
                .and_then(|output| output.clone())
                .unwrap_or_default();
            return sdk::StopDecision::Continue {
                message:
                    super::system_directives::SystemDirective::ExternalMutationRecoveryRequired(
                        evidence.chars().take(600).collect(),
                    )
                    .render(),
            };
        }
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
        let reply = loop {
            let reply = self.sdk_model_call(scope, &call.request).await;
            // Only a timed-out call comes back Transient here: the provider
            // stack has already retried everything else it can.
            let Err(sdk::ProviderError::Transient(_)) = &reply else {
                break reply;
            };
            let retries = scope.model_call_timeouts.fetch_add(1, Ordering::Relaxed);
            if retries >= MAX_MODEL_CALL_TIMEOUT_RETRIES {
                break reply;
            }
            let timeout_secs = self.sdk_model_call_timeout().as_secs();
            warn!(
                session_id = scope.session_id,
                task_id = scope.task_id,
                retry = retries + 1,
                "SDK engine: retrying a timed-out model call"
            );
            if let Ok(mut extra) = scope.extra_system.lock() {
                extra.push(
                    super::system_directives::SystemDirective::ModelCallTimedOut { timeout_secs }
                        .render(),
                );
            }
        };
        let _ = call.reply.send(reply);
    }

    /// The legacy per-call ceiling: a hung provider must not hold the turn.
    fn sdk_model_call_timeout(&self) -> Duration {
        let ceiling = if self.mandate_execution.is_some() {
            MANDATE_MAX_MODEL_CALL
        } else {
            FOREGROUND_MAX_MODEL_CALL
        };
        self.limits
            .llm_call_timeout
            .unwrap_or(DEFAULT_MODEL_CALL_TIMEOUT)
            .min(ceiling)
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
        // Hook notes (recalled memory) ride in the wire prefix; the rest of
        // the kernel transcript after the starting prefix is this run's.
        let (notes, kernel) = split_injected_notes(&request.messages, scope.prefix_system_lead);
        let run = &kernel[scope.prefix_len.min(kernel.len())..];
        let extra: Vec<Value> = scope
            .extra_system
            .lock()
            .map(|extra| {
                extra
                    .iter()
                    .map(|text| serde_json::json!({"role": "system", "content": text}))
                    .collect()
            })
            .unwrap_or_default();
        let mut prefix = prefix_with_notes(&scope.prefix, &notes);
        prefix.extend(extra);
        let messages = fit_run(&prefix, run, budget);
        let mut options = to_chat_options(&request.tool_choice, request.max_output_tokens);
        if scope.text_only_next_call.swap(false, Ordering::Relaxed) {
            options.tool_choice = crate::traits::ToolChoiceMode::None;
        }
        let est_input_tokens =
            crate::memory::context_window::estimate_multimodal_message_tokens(&messages);
        let mut telemetry = super::llm::LlmCallTelemetry::default();
        // Mandate runs share one durable token balance across the task lead
        // and its executors: reserve before the call, settle after it, as in
        // the legacy LLM phase.
        let lease = self
            .sdk_mandate_token_lease(&mut options, est_input_tokens as u32)
            .await?;
        let started = Instant::now();
        let timeout = self.sdk_model_call_timeout();
        let result = match tokio::time::timeout(
            timeout,
            self.call_llm_with_recovery(
                snapshot.provider(),
                snapshot.router(),
                &request.model,
                &messages,
                &tools,
                &options,
                &mut telemetry,
                None,
            ),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(anyhow::Error::new(ModelCallTimedOut(timeout))),
        };
        let latency_ms = started.elapsed().as_millis() as u64;
        if let Some(lease) = lease.as_ref() {
            match &result {
                Ok(response) => {
                    let delta = response
                        .usage
                        .as_ref()
                        .map(|usage| i64::try_from(usage.budget_tokens()).unwrap_or(i64::MAX))
                        .unwrap_or_else(|| {
                            lease
                                .budget_per_cycle
                                .saturating_sub(lease.tokens_used_before)
                        });
                    if let Err(error) = self
                        .state
                        .settle_mandate_run_token_lease(
                            &lease.goal_run_id,
                            &lease.lease_token,
                            delta,
                        )
                        .await
                    {
                        warn!(task_id = scope.task_id, %error, "SDK engine: could not settle mandate spend");
                    }
                }
                Err(_) => {
                    super::llm_phase::exhaust_mandate_run_token_lease_after_ambiguous_call(
                        self,
                        Some(lease),
                        scope.session_id,
                    )
                    .await;
                }
            }
        }
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
                if error.is::<ModelCallTimedOut>() {
                    return Err(sdk::ProviderError::Transient(error.to_string()));
                }
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
        if let Some(usage) = usage.as_ref() {
            if let Some(stop) = self
                .sdk_charge_budgets(scope, usage, iteration as usize)
                .await
            {
                // The legacy closeout grace: a final answer on the first call,
                // or after real progress, still ships; otherwise the budget
                // reply replaces it and no further work runs.
                let grace = response.tool_calls.is_empty()
                    && (iteration == 1 || scope.succeeded_calls.load(Ordering::Relaxed) > 0);
                if !grace {
                    scope.end_with(stop, TaskOutcome::Failed);
                    return Err(sdk::ProviderError::Other(GOAL_BUDGET_EXHAUSTED.into()));
                }
            }
        }
        if let Some(note) = response.response_note.clone() {
            if let Ok(mut latest) = scope.response_note.lock() {
                *latest = Some(note);
            }
        }
        scope
            .last_batch_size
            .store(response.tool_calls.len(), Ordering::Relaxed);
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

    /// Admit a mandate run's model call against its durable balance. Returns
    /// the lease to settle, or an error that ends the run when the balance
    /// cannot admit the call.
    async fn sdk_mandate_token_lease(
        &self,
        options: &mut crate::traits::ChatOptions,
        est_input_tokens: u32,
    ) -> Result<Option<super::llm_phase::MandateRunTokenLease>, sdk::ProviderError> {
        use super::llm_phase::{
            acquire_mandate_run_token_lease, mandate_output_token_ceiling,
            release_mandate_run_token_lease_without_dispatch, MandateRunTokenAdmission,
        };
        let lease_secs = i64::try_from(MANDATE_MAX_MODEL_CALL.as_secs())
            .unwrap_or(900)
            .saturating_add(30);
        let admission = acquire_mandate_run_token_lease(self, lease_secs, &None)
            .await
            .map_err(|error| sdk::ProviderError::Other(error.to_string()))?;
        match admission {
            MandateRunTokenAdmission::NotApplicable => Ok(None),
            MandateRunTokenAdmission::Exhausted { .. } => {
                Err(sdk::ProviderError::Other(MANDATE_BUDGET_EXHAUSTED.into()))
            }
            MandateRunTokenAdmission::Acquired(lease) => {
                let remaining = lease
                    .budget_per_cycle
                    .saturating_sub(lease.tokens_used_before);
                let Some(ceiling) = mandate_output_token_ceiling(remaining, est_input_tokens)
                else {
                    release_mandate_run_token_lease_without_dispatch(self, Some(&lease), "").await;
                    return Err(sdk::ProviderError::Other(MANDATE_BUDGET_EXHAUSTED.into()));
                };
                options.max_tokens_override = Some(
                    options
                        .max_tokens_override
                        .map_or(ceiling, |configured| configured.min(ceiling)),
                );
                options.single_attempt_fail_closed = true;
                // Durably mark the reservation dispatched before the provider
                // sees the request, so an interrupted call settles fail-closed.
                let fence = self
                    .mandate_execution
                    .as_ref()
                    .ok_or_else(|| sdk::ProviderError::Other("mandate fence missing".into()))?;
                let dispatched = self
                    .state
                    .mark_mandate_run_token_lease_dispatched(
                        &lease.goal_run_id,
                        &fence.mandate_id,
                        fence.mandate_version,
                        &lease.lease_token,
                    )
                    .await
                    .unwrap_or(false);
                if !dispatched {
                    release_mandate_run_token_lease_without_dispatch(self, Some(&lease), "").await;
                    return Err(sdk::ProviderError::Other(
                        "mandate model call was not durably reserved before provider dispatch"
                            .into(),
                    ));
                }
                Ok(Some(lease))
            }
        }
    }

    /// Charge a model call to the task, goal-daily and scheduled-run
    /// budgets. When one is used up, it extends once more if receipts show
    /// progress since the last extension (or the owner approves), as the
    /// legacy budget controls do; otherwise the owner is alerted and the
    /// honest budget reply is returned.
    async fn sdk_charge_budgets(
        &self,
        scope: &TurnScope<'_>,
        usage: &crate::traits::TokenUsage,
        iteration: usize,
    ) -> Option<String> {
        let delta = usage.budget_tokens();
        let used = scope.task_tokens.fetch_add(delta, Ordering::Relaxed) + delta;
        let successes = scope.succeeded_calls.load(Ordering::Relaxed);
        let learning = scope
            .learning
            .lock()
            .map(|learning| learning.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
        let mut directives = Vec::new();
        let stop = if let Some(stop) = self
            .sdk_task_budget(
                scope,
                used,
                successes,
                iteration,
                &learning,
                &mut directives,
            )
            .await
        {
            Some(stop)
        } else if let Some(goal) = &scope.budget_goal {
            self.sdk_goal_budget(
                scope,
                goal,
                delta as i64,
                successes,
                iteration,
                &learning,
                &mut directives,
            )
            .await
        } else {
            None
        };
        if let Ok(mut extra) = scope.extra_system.lock() {
            extra.extend(directives.iter().map(|directive| directive.render()));
        }
        stop
    }

    #[allow(clippy::too_many_arguments)]
    async fn sdk_task_budget(
        &self,
        scope: &TurnScope<'_>,
        used: u64,
        successes: usize,
        iteration: usize,
        learning: &crate::agent::post_task::LearningContext,
        directives: &mut Vec<super::system_directives::SystemDirective>,
    ) -> Option<String> {
        let budget = scope.task_budget.load(Ordering::Relaxed);
        if budget == 0 || used < budget {
            return None;
        }
        let cap = TASK_HARD_TOKEN_CAP;
        let proposed = budget
            .saturating_mul(2)
            .max(used.saturating_add(budget / 2))
            .min(cap);
        let extensions = scope.task_budget_extensions.load(Ordering::Relaxed);
        let progressed = successes > scope.successes_at_task_extension.load(Ordering::Relaxed)
            && !scope.repeating.load(Ordering::Relaxed);
        let can_extend = budget < cap && proposed > used;
        let approved = can_extend
            && if extensions < TASK_MAX_BUDGET_EXTENSIONS
                && progressed
                && scope.user_role == UserRole::Owner
            {
                true
            } else {
                self.request_budget_continue_approval(
                    scope.emitter,
                    scope.task_id,
                    iteration,
                    scope.session_id,
                    scope.user_role,
                    "task",
                    used as i64,
                    budget as i64,
                    proposed as i64,
                )
                .await
            };
        if approved {
            let extension = extensions + 1;
            scope.task_budget.store(proposed, Ordering::Relaxed);
            scope
                .task_budget_extensions
                .store(extension, Ordering::Relaxed);
            scope
                .successes_at_task_extension
                .store(successes, Ordering::Relaxed);
            self.with_harness_eval(|eval| eval.record_budget_extension())
                .await;
            directives.push(
                super::system_directives::SystemDirective::TaskBudgetAutoExtended {
                    old_budget: budget as i64,
                    new_budget: proposed as i64,
                    extension,
                    max_extensions: TASK_MAX_BUDGET_EXTENSIONS,
                },
            );
            if let Some(status_tx) = &scope.status_tx {
                let _ = status_tx.try_send(StatusUpdate::BudgetExtended {
                    old_budget: budget as i64,
                    new_budget: proposed as i64,
                    extension,
                    max_extensions: TASK_MAX_BUDGET_EXTENSIONS,
                });
            }
            info!(
                task_id = scope.task_id,
                old_budget = budget,
                new_budget = proposed,
                extension,
                "SDK engine: extended the task token budget"
            );
            return None;
        }
        let alert = format!(
            "Token alert: execution in session '{}' hit task token budget (used {} / limit {}). \
             The run was stopped to prevent overspending.",
            scope.session_id, used, budget
        );
        self.fanout_token_alert(None, scope.session_id, &alert, Some(scope.session_id))
            .await;
        Some(crate::agent::post_task::graceful_budget_response(
            learning, used,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    async fn sdk_goal_budget(
        &self,
        scope: &TurnScope<'_>,
        goal: &BudgetGoal,
        delta: i64,
        successes: usize,
        iteration: usize,
        learning: &crate::agent::post_task::LearningContext,
        directives: &mut Vec<super::system_directives::SystemDirective>,
    ) -> Option<String> {
        let status = match self
            .state
            .add_goal_tokens_and_get_budget_status(&goal.goal_id, delta)
            .await
        {
            Ok(Some(status)) => status,
            Ok(None) => return None,
            Err(error) => {
                warn!(goal_id = %goal.goal_id, %error, "SDK engine: goal token accounting failed");
                return None;
            }
        };
        let metrics = || super::graceful::ScheduledRunActivityMetrics {
            evidence_gain_count: successes,
            stall_count: usize::from(scope.repeating.load(Ordering::Relaxed)),
            consecutive_same_tool_count: 0,
            consecutive_same_tool_unique_args: 0,
            total_successful_tool_calls: successes,
        };
        if goal.scheduled {
            let registry = self.goal_token_registry.as_ref()?;
            registry.add_run_tokens(&goal.goal_id, delta).await;
            let progress = scope
                .progress
                .lock()
                .ok()
                .map(|progress| progress.clone())?;
            let run_status = registry
                .update_run_health(
                    &goal.goal_id,
                    Agent::scheduled_run_health_snapshot(
                        learning,
                        metrics(),
                        scope.contract,
                        &progress,
                    ),
                )
                .await?;
            super::goal_dispatch::persist_scheduled_run_state(
                &self.state,
                &goal.goal_id,
                None,
                &run_status,
            )
            .await;
            let mut ctx = super::graceful::ScheduledRunBudgetControlCtx {
                emitter: scope.emitter,
                task_id: scope.task_id,
                session_id: scope.session_id,
                iteration,
                goal_id: &goal.goal_id,
                status: &run_status,
                pending_system_messages: directives,
                max_budget_extensions: SCHEDULED_MAX_BUDGET_EXTENSIONS,
                hard_token_cap: SCHEDULED_HARD_TOKEN_CAP,
            };
            if let super::graceful::ScheduledRunBudgetControlOutcome::Exhausted {
                tokens_used,
                budget_per_check,
            } = self.enforce_scheduled_run_budget_control(&mut ctx).await
            {
                let alert = format!(
                    "Token alert: scheduled run for goal '{}' hit its per-run budget (used {} / \
                     limit {}). The run stopped safely before completion.",
                    goal.goal_id, tokens_used, budget_per_check
                );
                self.fanout_token_alert(
                    Some(goal.goal_id.as_str()),
                    scope.session_id,
                    &alert,
                    Some(scope.session_id),
                )
                .await;
                return Some(
                    crate::agent::post_task::graceful_scheduled_run_budget_response(
                        learning,
                        tokens_used,
                        budget_per_check,
                    ),
                );
            }
            return None;
        }
        let mut effective_daily = scope
            .goal_daily_budget
            .lock()
            .map(|budget| *budget)
            .unwrap_or(None);
        let mut extensions = scope.goal_budget_extensions.load(Ordering::Relaxed) as usize;
        let outcome = {
            let mut ctx = super::graceful::GoalBudgetControlCtx {
                emitter: scope.emitter,
                task_id: scope.task_id,
                session_id: scope.session_id,
                iteration,
                goal_id: &goal.goal_id,
                status: &status,
                user_role: scope.user_role,
                learning_ctx: learning,
                evidence_gain_count: successes,
                stall_count: usize::from(scope.repeating.load(Ordering::Relaxed)),
                consecutive_same_tool_count: 0,
                consecutive_same_tool_unique_args: 0,
                total_successful_tool_calls: successes,
                pending_system_messages: directives,
                status_tx: &scope.status_tx,
                is_scheduled_goal: false,
                effective_goal_daily_budget: &mut effective_daily,
                budget_extensions_count: &mut extensions,
                max_budget_extensions: TASK_MAX_BUDGET_EXTENSIONS,
                hard_token_cap: TASK_HARD_TOKEN_CAP as i64,
                source: super::graceful::GoalBudgetCheckSource::PostLlm,
            };
            self.enforce_goal_daily_budget_control(&mut ctx).await
        };
        if let Ok(mut budget) = scope.goal_daily_budget.lock() {
            *budget = effective_daily;
        }
        scope
            .goal_budget_extensions
            .store(extensions as u32, Ordering::Relaxed);
        if let super::graceful::GoalBudgetControlOutcome::Exhausted {
            tokens_used_today,
            budget_daily,
        } = outcome
        {
            let alert = format!(
                "Token alert: goal '{}' hit daily token budget (used {} / limit {}). Execution \
                 was stopped to prevent overspending.",
                goal.goal_id, tokens_used_today, budget_daily
            );
            self.fanout_token_alert(
                Some(goal.goal_id.as_str()),
                scope.session_id,
                &alert,
                Some(scope.session_id),
            )
            .await;
            return Some(
                crate::agent::post_task::graceful_goal_daily_budget_response(
                    learning,
                    tokens_used_today,
                    budget_daily,
                    false,
                ),
            );
        }
        None
    }

    /// The legacy step-plan idempotency key: any call that is not a pure
    /// observation and mutates, has an external side effect, or is not
    /// idempotent.
    fn sdk_idempotency_key(
        &self,
        scope: &TurnScope<'_>,
        request: &ToolRequest,
        semantics: &crate::traits::ToolCallSemantics,
    ) -> Option<String> {
        let capabilities = self
            .tools
            .iter()
            .find(|tool| tool.name() == request.name)
            .map(|tool| tool.capabilities())?;
        let pure_observation = semantics.observes_state() && !semantics.mutates_state();
        let needs_key = !pure_observation
            && (semantics.mutates_state()
                || capabilities.external_side_effect
                || !capabilities.idempotent);
        needs_key.then(|| {
            let hash = crate::agent::prefix_fingerprint::hash_canonical(&request.arguments);
            format!("exec:{}:{}:{hash}", scope.task_id, request.name)
        })
    }

    /// A duplicate `send_file` in the same task: record a no-op result and
    /// tell the model the file already went out, as the legacy guard does.
    async fn sdk_suppress_duplicate_send(
        &self,
        scope: &TurnScope<'_>,
        request: &ToolRequest,
    ) -> sdk::ToolOutput {
        info!(
            task_id = scope.task_id,
            tool_call_id = %request.call_id,
            "SDK engine: suppressed a duplicate send_file"
        );
        let mut call_data = crate::events::ToolCallData::from_tool_call(
            request.call_id.clone(),
            request.name.clone(),
            request.arguments.clone(),
            Some(scope.task_id.to_string()),
        );
        call_data.turn_id = Some(scope.turn_id.to_string());
        let _ = scope.emitter.emit(EventType::ToolCall, call_data).await;
        let result_text = "Duplicate send_file suppressed: this exact file+caption was already \
                           sent in this task.";
        let mut tool = Message::new_runtime(Uuid::new_v4().to_string(), scope.session_id, "tool");
        tool.content = Some(result_text.to_string());
        tool.tool_name = Some(request.name.clone());
        tool.tool_call_id = Some(request.call_id.clone());
        tool.turn_id = Some(scope.turn_id.to_string());
        if let Err(error) = self
            .append_tool_message_with_receipt_event_policy(
                scope.emitter,
                &tool,
                true,
                0,
                None,
                Some(scope.task_id),
                None,
                None,
            )
            .await
        {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a suppressed send");
        }
        sdk::ToolOutput::text(format!(
            "{result_text}\n\n{}",
            super::system_directives::SystemDirective::DuplicateSendFileAlreadySent.render()
        ))
    }

    /// Act on the durable preflight: replay a completed earlier result
    /// (recorded as a replayed receipt) or refuse an indeterminate repeat.
    async fn sdk_replay_or_block(
        &self,
        scope: &TurnScope<'_>,
        request: &ToolRequest,
        arguments: &str,
        preflight: &super::tool_execution_phase::DurableReplayPreflight,
    ) -> Option<sdk::ToolOutput> {
        let call = crate::traits::ToolCall {
            id: request.call_id.clone(),
            name: request.name.clone(),
            arguments: arguments.to_string(),
            extra_content: None,
        };
        if let Some(reason) = &preflight.block_reason {
            if let Err(error) = self
                .persist_pre_dispatch_outcome_with_kernel_claim(
                    scope.emitter,
                    scope.session_id,
                    scope.task_id,
                    &call,
                    arguments,
                    format!("Error: {reason}"),
                    crate::traits::ToolOutcomeStatus::Blocked,
                    crate::traits::ToolInvocationStage::RejectedBeforeDispatch,
                    false,
                    crate::traits::ToolCallSemantics::default(),
                    None,
                    None,
                    None,
                )
                .await
            {
                warn!(task_id = scope.task_id, %error, "SDK engine: could not record a blocked repeat");
            }
            return Some(sdk::ToolOutput::error(format!("Error: {reason}")));
        }
        let previous = preflight.replayed_result.as_ref()?;
        info!(
            task_id = scope.task_id,
            tool = %request.name,
            source = %previous.tool_call_id,
            "SDK engine: replayed a durable receipt instead of repeating the call"
        );
        let mut call_data = crate::events::ToolCallData::from_tool_call(
            request.call_id.clone(),
            request.name.clone(),
            request.arguments.clone(),
            Some(scope.task_id.to_string()),
        );
        call_data.turn_id = Some(scope.turn_id.to_string());
        let _ = scope.emitter.emit(EventType::ToolCall, call_data).await;
        let mut receipt = previous.receipt.clone();
        if let Some(receipt) = receipt.as_mut() {
            receipt.invocation_stage = crate::traits::ToolInvocationStage::Replayed;
            receipt.outcome_evidence = crate::events::ToolOutcomeEvidenceSource::DurableReplay;
        }
        let output = replay_notice(&previous.result);
        let mut tool = Message::new_runtime(Uuid::new_v4().to_string(), scope.session_id, "tool");
        tool.content = Some(output.clone());
        tool.tool_name = Some(request.name.clone());
        tool.tool_call_id = Some(request.call_id.clone());
        tool.turn_id = Some(scope.turn_id.to_string());
        let failed = !previous.success;
        if let Err(error) = self
            .append_tool_message_with_receipt_event_policy(
                scope.emitter,
                &tool,
                !failed,
                0,
                previous.error.clone(),
                Some(scope.task_id),
                None,
                receipt,
            )
            .await
        {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a replay");
        }
        Some(sdk::ToolOutput {
            is_error: failed,
            ..sdk::ToolOutput::text(output)
        })
    }

    /// Resolve nested project instructions for a call's targets. When new
    /// guidance applies, it joins the run's system context and the call is
    /// recorded as deferred, not executed.
    async fn sdk_load_nested_instructions(
        &self,
        scope: &TurnScope<'_>,
        request: &ToolRequest,
        arguments: &str,
        semantics: &crate::traits::ToolCallSemantics,
    ) -> Option<sdk::ToolOutput> {
        if self.mandate_execution.is_some() {
            return None;
        }
        let tracker = scope.instruction_tracker.as_ref()?;
        let targets = super::tool_execution_phase::project_instruction_targets_for_tool_call(
            &request.name,
            arguments,
            semantics,
        );
        if targets.is_empty() {
            return None;
        }
        let discovered = tracker
            .lock()
            .await
            .discover_for_targets(crate::execution::active_execution_backend(), &targets)
            .await;
        let instructions = match discovered {
            Ok(Some(instructions)) => instructions,
            Ok(None) => return None,
            Err(error) => {
                warn!(
                    task_id = scope.task_id,
                    tool = %request.name,
                    %error,
                    "SDK engine: could not discover nested project instructions"
                );
                return None;
            }
        };
        let sources = instructions.source_paths();
        if let Ok(mut extra) = scope.extra_system.lock() {
            extra.push(format!(
                "[Just-in-time Project Instructions — loaded before subtree work]\n{}",
                instructions.render_for_prompt()
            ));
        }
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            scope.model_calls.load(Ordering::Relaxed) as usize,
            crate::events::DecisionType::InstructionsSnapshot,
            format!("Loaded nested project instructions before {}", request.name),
            serde_json::json!({
                "condition": "jit_project_instructions_discovered",
                "tool": request.name,
                "targets": &targets,
                "sources": &sources,
                "action_executed": false,
            }),
        )
        .await;
        let notice = crate::agent::ToolResultNotice::ProjectInstructionsDiscovered {
            tool_name: request.name.clone(),
            sources: sources.join(", "),
        }
        .render();
        let call = crate::traits::ToolCall {
            id: request.call_id.clone(),
            name: request.name.clone(),
            arguments: arguments.to_string(),
            extra_content: None,
        };
        if let Err(error) = self
            .persist_pre_dispatch_outcome_with_kernel_claim(
                scope.emitter,
                scope.session_id,
                scope.task_id,
                &call,
                arguments,
                notice.clone(),
                crate::traits::ToolOutcomeStatus::Blocked,
                crate::traits::ToolInvocationStage::RejectedBeforeDispatch,
                false,
                crate::traits::ToolCallSemantics::default(),
                None,
                None,
                None,
            )
            .await
        {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a deferred call");
        }
        Some(sdk::ToolOutput::error(notice))
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
            prefix,
            contract,
            learning_ctx,
            continue_inline_after_background,
            instruction_tracker,
            restrict_untrusted_reference_tools,
            untrusted_reference_skills,
            memory_recall_allowed,
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

        let (calls_tx, mut calls_rx) = mpsc::channel::<DaemonCall>(16);
        let mut builder = sdk::Agent::builder("aidaemon")
            .model(model.clone())
            .provider(DaemonProvider {
                calls: calls_tx.clone(),
            })
            .limits(sdk::Limits {
                max_turns: engine.max_turns,
                tool_timeout: TOOL_TIMEOUT,
                ..sdk::Limits::default()
            });
        let memory_persistence_allowed = learning_ctx.memory_persistence_allowed;
        let learning = Arc::new(std::sync::Mutex::new(learning_ctx));
        let verdict = Arc::new(TurnVerdict::default());
        // Memory comes from the active plugin; a mandate worker gets none.
        let memory_turn = MemoryTurn {
            session_id: session_id.to_string(),
            task_id: task_id.clone(),
            emitter: emitter.clone(),
            user_text: user_text.to_string(),
            user_role,
            channel_ctx: channel_ctx.clone(),
            memory_pipeline_allowed: memory_recall_allowed,
            memory_persistence_allowed,
            mandate: self.mandate_execution.is_some(),
            depth: self.depth,
            model: model.clone(),
            learning: learning.clone(),
            verdict: verdict.clone(),
        };
        let memory = memory_parts(self.memory.as_ref(), &memory_turn).await;
        // Tools: the set bootstrap offered for this turn (role, policy and
        // contract filtering already applied), with the daemon memory tools
        // only while the active plugin owns them.
        if user_role != UserRole::Public {
            for tool in self.tools.iter().filter(|tool| {
                offered_tools.contains(tool.name()) && daemon_tool_offered(tool.name(), &memory)
            }) {
                builder = builder.tool(BridgedTool {
                    spec: tool_spec(tool.as_ref()),
                    read_only: tool.capabilities().read_only,
                    calls: calls_tx.clone(),
                });
            }
        }
        // A background handoff ends the run before the next model call.
        let end_run = Arc::new(AtomicBool::new(false));
        let end_flag = end_run.clone();
        builder = builder.stop_when(move |_: &sdk::RunSnapshot| {
            end_flag
                .load(Ordering::Relaxed)
                .then(|| TERMINAL_REPLY_STOP.to_string())
        });
        builder = builder.hook(SameToolLimit);
        builder = builder.hook(LedgerStop {
            calls: calls_tx.clone(),
        });
        // After the ledger stop, so a rejected final answer never reaches a
        // memory writer.
        for hook in memory.hooks {
            builder = builder.hook_arc(hook);
        }
        if user_role != UserRole::Public {
            for tool in memory.tools {
                builder = builder.tool_arc(tool);
            }
        }
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
                _ => return,
            };
            let _ = status_tx.try_send(update);
        });
        let sdk_agent = builder
            .build()
            .map_err(|error| anyhow::anyhow!("SDK engine agent: {error}"))?;

        // The kernel's own view of the starting transcript (for its stuck
        // detection and bookkeeping); model calls send the projected prefix
        // bytes unchanged.
        let mut state = sdk::RunState::new("aidaemon", session_id, sdk::Message::user(user_text));
        let projected = values_to_sdk(&prefix);
        if !projected.is_empty() {
            state.messages = projected;
        }
        let prefix_len = state.messages.len();
        let prefix_system_lead = state
            .messages
            .iter()
            .position(|message| message.role != sdk::Role::System)
            .unwrap_or(prefix_len);

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
        let budget_goal_for_daily = budget_goal.as_ref().map(|goal| goal.goal_id.clone());
        let scope = TurnScope {
            session_id,
            user_text,
            task_id: &task_id,
            status_tx: status_tx.clone(),
            channel_ctx: &channel_ctx,
            user_role,
            model: &model,
            turn_id: &turn_id,
            contract: &contract,
            learning: learning.clone(),
            latest_tool_output: std::sync::Mutex::new(None),
            reply_retry_used: AtomicBool::new(false),
            text_only_next_call: AtomicBool::new(false),
            response_note: std::sync::Mutex::new(None),
            continue_inline_after_background,
            instruction_tracker: instruction_tracker.map(tokio::sync::Mutex::new),
            extra_system: std::sync::Mutex::new(Vec::new()),
            sent_files: std::sync::Mutex::new(HashSet::new()),
            mandate_commit_retries: std::sync::atomic::AtomicU32::new(0),
            model_call_timeouts: std::sync::atomic::AtomicU32::new(0),
            task_tokens: AtomicU64::new(0),
            task_budget: AtomicU64::new(self.limits.task_token_budget.unwrap_or(0)),
            task_budget_extensions: std::sync::atomic::AtomicUsize::new(0),
            succeeded_calls: std::sync::atomic::AtomicUsize::new(0),
            succeeded_operations: std::sync::Mutex::new(HashSet::new()),
            repeating: AtomicBool::new(false),
            successes_at_task_extension: std::sync::atomic::AtomicUsize::new(0),
            goal_daily_budget: std::sync::Mutex::new(match &budget_goal_for_daily {
                Some(goal_id) => match &self.goal_token_registry {
                    Some(registry) => registry.get_effective_daily_budget(goal_id).await,
                    None => None,
                },
                None => None,
            }),
            goal_budget_extensions: std::sync::atomic::AtomicU32::new(0),
            restrict_untrusted_reference_tools,
            untrusted_reference_skills,
            verdict: verdict.clone(),
            last_batch_size: std::sync::atomic::AtomicUsize::new(0),
            end_run: end_run.clone(),
            progress: std::sync::Mutex::new(
                crate::agent::completion_contract::CompletionProgress::new(&contract, &task_id),
            ),
            background_handoff: AtomicBool::new(false),
            interrupted_answer: std::sync::Mutex::new(None),
            emitter: &emitter,
            prefix,
            prefix_len,
            prefix_system_lead,
            budget_goal,
            budget_exhausted: AtomicBool::new(false),
            model_calls: AtomicU64::new(0),
            closeout_demands: std::sync::atomic::AtomicU32::new(0),
            verification_demanded: AtomicBool::new(false),
            recovery_demanded: AtomicBool::new(false),
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
                let (reply, status, task_outcome) = scope.verdict.resolve(&outcome);
                let run_failure = scope.verdict.run_failure();
                let reply = if matches!(outcome.status, sdk::RunStatus::Completed) {
                    let tool_calls = scope
                        .learning
                        .lock()
                        .map(|learning| learning.tool_calls.clone())
                        .unwrap_or_default();
                    let note = scope
                        .response_note
                        .lock()
                        .ok()
                        .and_then(|note| note.clone());
                    let interrupted = scope
                        .interrupted_answer
                        .lock()
                        .ok()
                        .and_then(|answer| answer.clone());
                    let reply = match interrupted {
                        Some(answer) if reply.trim().is_empty() => answer,
                        _ => reply,
                    };
                    let reply = finalize_reply(&reply, &tool_calls, note.as_deref());
                    match channel_ctx.visibility {
                        crate::types::ChannelVisibility::Public
                        | crate::types::ChannelVisibility::PublicExternal => {
                            let (sanitized, redacted) =
                                crate::tools::sanitize::sanitize_output(&reply);
                            if redacted
                                && channel_ctx.visibility
                                    == crate::types::ChannelVisibility::PublicExternal
                            {
                                format!("{sanitized}\n\n(Some content was filtered for security)")
                            } else {
                                sanitized
                            }
                        }
                        _ => reply,
                    }
                } else {
                    reply
                };
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
                    _ => run_failure.clone(),
                };
                (
                    match run_failure {
                        Some(failure) => Err(anyhow::anyhow!(failure)),
                        None => Ok(reply),
                    },
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
        // Post-turn memory and learning run in the memory plugin's run-end
        // hook.
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

    fn engine_config(provider_kind: &str, engine: &str) -> AppConfig {
        toml::from_str(&format!(
            "[provider]\nkind = \"{provider_kind}\"\napi_key = \"test-key\"\n\n\
             [provider.models]\ndefault = \"model-a\"\n\n[agent.engine]\n{engine}"
        ))
        .expect("parse app config")
    }

    #[test]
    fn legacy_engine_model_pin_is_dropped_for_a_non_anthropic_provider() {
        let config = engine_config(
            "openai_chatgpt",
            "model = \"claude-haiku-5-5\"\napi_key = \"sk-ant-test\"",
        );
        let settings = settings_from_config(&config);
        assert_eq!(settings.model_override, None);
    }

    #[test]
    fn legacy_engine_model_pin_is_kept_for_an_anthropic_provider() {
        let config = engine_config(
            "anthropic",
            "model = \"claude-haiku-5-5\"\napi_key = \"sk-ant-test\"",
        );
        let settings = settings_from_config(&config);
        assert_eq!(settings.model_override.as_deref(), Some("claude-haiku-5-5"));
    }

    #[test]
    fn engine_model_pin_without_api_key_is_honored() {
        let config = engine_config("openai_chatgpt", "model = \"gpt-6-luna\"");
        let settings = settings_from_config(&config);
        assert_eq!(settings.model_override.as_deref(), Some("gpt-6-luna"));
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
        harness.agent.set_sdk_engine(SdkEngineSettings::default());
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

    #[tokio::test]
    async fn sdk_engine_retries_a_timed_out_model_call() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        let provider = MockProvider::with_delayed_responses(
            vec![MockProvider::text_response("Here is the answer.")],
            vec![Duration::from_secs(3), Duration::ZERO],
        );
        let harness = crate::testing::setup_test_agent_sdk_with_llm_timeout(provider, Some(1))
            .await
            .unwrap();
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("Summarize the plan."))
            .await
            .unwrap();
        assert!(
            reply.text.contains("Here is the answer."),
            "a timed-out call is retried, not the end of the run: {}",
            reply.text
        );
        let retried_with_directive = harness.provider.call_log.lock().await.iter().any(|call| {
            call.messages.iter().any(|message| {
                message
                    .get("content")
                    .and_then(Value::as_str)
                    .is_some_and(|text| text.contains("took longer than 1s"))
            })
        });
        assert!(
            retried_with_directive,
            "the retry tells the model to take a smaller step"
        );
    }

    #[tokio::test]
    async fn sdk_engine_fails_after_repeated_model_call_timeouts() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        let attempts = MAX_MODEL_CALL_TIMEOUT_RETRIES as usize + 1;
        let provider = MockProvider::with_delayed_responses(
            vec![MockProvider::text_response("Too late.")],
            vec![Duration::from_secs(3); attempts],
        );
        let harness = crate::testing::setup_test_agent_sdk_with_llm_timeout(provider, Some(1))
            .await
            .unwrap();
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("Summarize the plan."))
            .await
            .unwrap();
        assert!(
            reply.text.contains("timed out after 1s"),
            "the retry budget is bounded: {}",
            reply.text
        );
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
    /// The closeout interrupted a real answer, then the model went silent:
    /// the user gets that answer, not a generic activity summary.
    #[tokio::test]
    async fn empty_reply_after_a_closeout_demand_ships_the_interrupted_answer() {
        use crate::testing::MockProvider;
        let workspace = tempfile::tempdir().expect("workspace");
        let file = workspace
            .path()
            .join("never-read.txt")
            .to_string_lossy()
            .to_string();
        let checklist_args = serde_json::json!({
            "items": [
                {"text": "read the file back", "status": "pending",
                 "requires_observation": true, "targets": [file]}
            ]
        })
        .to_string();
        let answer = "Cancelled the Acme Corp reminder; the daily schedule is unchanged.";
        let provider = MockProvider::with_responses(vec![
            MockProvider::tool_call_response("track_requirements", &checklist_args),
            MockProvider::text_response(answer),
            MockProvider::text_response(""),
            MockProvider::text_response(""),
            MockProvider::text_response(""),
        ])
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
        let harness = crate::testing::setup_full_stack_test_agent_with_extra_tools(
            provider,
            vec![track as Arc<dyn crate::traits::Tool>],
        )
        .await
        .unwrap();

        let response = harness
            .agent
            .handle_message(
                "interrupted_answer",
                "Cancel the reminder.",
                None,
                UserRole::Owner,
                ChannelContext::private("telegram"),
                None,
            )
            .await
            .unwrap();
        assert_eq!(response, answer);
    }

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
        harness.agent.set_sdk_engine(SdkEngineSettings::default());

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
            verification_after_mutation: false,
            unrecovered_failure: false,
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
        // Owner decision: a reachable observation after a successful change
        // is demanded once even after work completed.
        let verify = StopFacts {
            verification_after_mutation: true,
            ..facts(&contract, 1, None, false)
        };
        assert!(closeout_demands_work(&verify));
        // A failed operation with nothing succeeded gets its recovery pass.
        let failed = StopFacts {
            unrecovered_failure: true,
            ..facts(&[], 0, None, false)
        };
        assert!(closeout_demands_work(&failed));
        assert!(!closeout_demands_work(&StopFacts {
            terminal_policy_denial: true,
            ..failed
        }));
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

    /// A shell-like tool: not read-only as a whole, but each call says
    /// whether it writes (`{"write": true}`), as `terminal` does.
    struct ShellProbe(Arc<std::sync::atomic::AtomicUsize>);

    #[async_trait::async_trait]
    impl crate::traits::Tool for ShellProbe {
        fn name(&self) -> &str {
            "shell_probe"
        }
        fn description(&self) -> &str {
            "Runs a synthetic command."
        }
        fn schema(&self) -> serde_json::Value {
            serde_json::json!({
                "name": "shell_probe",
                "description": "Runs a synthetic command.",
                "parameters": {"type": "object", "properties": {"write": {"type": "boolean"}}}
            })
        }
        fn capabilities(&self) -> crate::traits::ToolCapabilities {
            crate::traits::ToolCapabilities {
                read_only: false,
                external_side_effect: false,
                needs_approval: false,
                idempotent: true,
                high_impact_write: false,
            }
        }
        fn call_semantics(&self, arguments: &str) -> crate::traits::ToolCallSemantics {
            let writes = serde_json::from_str::<serde_json::Value>(arguments)
                .ok()
                .and_then(|args| args["write"].as_bool())
                .unwrap_or(true);
            if writes {
                crate::traits::ToolCallSemantics::mutation()
            } else {
                crate::traits::ToolCallSemantics::observation()
            }
        }
        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok("acme.md\nmia.txt".to_string())
        }
    }

    /// A read-only request must not lose a mixed tool such as `terminal`:
    /// the authority lane judges the exact call, as the legacy loop did.
    #[tokio::test]
    async fn read_only_contract_admits_observing_calls_of_a_mixed_tool() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        let base = MockProvider::semantic_task_assessment(
            "answer",
            false,
            true,
            &[],
            "new_request",
            "none",
        );
        let mut assessment: serde_json::Value =
            serde_json::from_str(base.content.as_deref().unwrap()).unwrap();
        assessment["schema_version"] = 10.into();
        assessment["contract"]["mutation_scope"] = "read_only".into();
        let provider = MockProvider::with_responses(vec![
            MockProvider::tool_call_response("shell_probe", r#"{"write":false}"#),
            MockProvider::tool_call_response("shell_probe", r#"{"write":true}"#),
            MockProvider::text_response("acme.md is the markdown file."),
        ])
        .with_task_assessments(vec![MockProvider::text_response(&assessment.to_string())]);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
            provider,
            vec![Arc::new(ShellProbe(calls.clone()))],
            None,
        )
        .await
        .unwrap();
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("Which file in the folder is markdown?"))
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the observing call runs; the writing call is refused"
        );
        assert_eq!(reply.text, "acme.md is the markdown file.");
    }

    #[test]
    fn replayed_results_say_they_were_not_run_again() {
        let text = replay_notice("Reminder set: \"Water the plants.\"");
        assert!(text.starts_with("[Not run again:"), "{text}");
        assert!(
            text.ends_with("Reminder set: \"Water the plants.\""),
            "{text}"
        );
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
    fn learning_records_each_call_like_the_legacy_loop() {
        let args = r#"{"path":"notes.txt"}"#;
        let ok = learning_tool_summary("read_file", args, false, false);
        assert!(ok.starts_with("read_file(") && !ok.contains('['), "{ok}");
        assert!(learning_tool_summary("read_file", args, false, true).ends_with(" [FAILED]"));
        assert!(learning_tool_summary("read_file", args, true, true).ends_with(" [REJECTED]"));
    }

    #[test]
    fn reply_checks_reject_empty_marker_only_and_pasted_replies() {
        assert_eq!(reply_defect("Here is the answer.", None), None);
        assert_eq!(reply_defect("   ", None), Some(ReplyDefect::Empty));
        assert_eq!(
            reply_defect("[SYSTEM] tool envelope only", None),
            Some(ReplyDefect::InternalMarkers)
        );
        assert_eq!(
            reply_defect("[tool_use: terminal]\nargs: {\"cmd\":\"ls\"}", None),
            Some(ReplyDefect::InternalMarkers)
        );
        let page = "File: /tmp/notes.txt (lines 1-5 of 5, 60 bytes)\n1 | alpha\n2 | beta\n\
                    3 | gamma\n4 | delta\n5 | epsilon";
        assert_eq!(reply_defect(page, None), Some(ReplyDefect::ToolOutputPaste));
        let output = "/srv/synthetic/report-alpha.txt\n/srv/synthetic/report-beta.txt\n\
                      /srv/synthetic/report-gamma.txt\n/srv/synthetic/report-delta.txt";
        assert_eq!(
            reply_defect(output, Some(output)),
            Some(ReplyDefect::ToolOutputPaste)
        );
    }

    #[test]
    fn final_replies_are_sanitized_and_never_a_bare_done() {
        let clean = finalize_reply("All set.\n[SYSTEM] internal note", &[], None);
        assert!(
            clean.contains("All set.") && !clean.contains("[SYSTEM]"),
            "{clean}"
        );
        let empty = finalize_reply("", &[], Some("SAFETY"));
        assert!(
            empty.starts_with("I wasn't able to process that request") && empty.contains("SAFETY"),
            "{empty}"
        );
        let gutted = finalize_reply("[SYSTEM] only markers", &[], None);
        assert!(gutted.contains("couldn't recover"), "{gutted}");
        let with_work = finalize_reply(
            "[SYSTEM] only markers",
            &["write_file(notes.txt)".to_string()],
            None,
        );
        assert!(!with_work.trim().is_empty() && !with_work.contains("[SYSTEM]"));
        let empty_after_work = finalize_reply("", &["write_file(notes.txt)".to_string()], None);
        assert!(
            !empty_after_work.trim().is_empty()
                && !empty_after_work.starts_with("I wasn't able to process"),
            "an empty reply after tool work describes the work: {empty_after_work}"
        );
    }

    #[tokio::test]
    async fn sdk_engine_retries_an_empty_final_reply_without_tools() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;
        let provider = MockProvider::with_responses(vec![
            remember("teal"),
            MockProvider::text_response(""),
            MockProvider::text_response("Saved your favorite color: teal."),
        ]);
        let harness = crate::testing::setup_test_agent_sdk(provider)
            .await
            .unwrap();
        let reply = harness
            .agent
            .handle_inbound_message(owner_request("Remember my favorite color is teal."))
            .await
            .unwrap();
        assert_eq!(reply.text, "Saved your favorite color: teal.");
        let calls = harness.provider.call_log.lock().await;
        let last = calls.last().expect("retry call");
        assert_eq!(
            last.options.tool_choice,
            crate::traits::ToolChoiceMode::None,
            "the empty-reply retry after work runs with tools disabled"
        );
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
