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

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
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
use crate::events::{EventEmitter, EventType, TaskEndData, TaskOutcome, TaskStartData, TaskStatus};
use crate::traits::{AttachmentProvenance, Message, MessageAttachment, ToolOutcomeStatus};
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
}

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
    status_tx: Option<mpsc::Sender<StatusUpdate>>,
    channel_ctx: &'a ChannelContext,
    user_role: UserRole,
    /// The engine model, for vision gating of tool-observed images.
    model: &'a str,
    /// Attachments each call produced, by call id, so the persisted tool
    /// message keeps them.
    tool_attachments: Mutex<HashMap<String, Vec<MessageAttachment>>>,
    /// Set when a call moved work to the background with completion
    /// notifications on: the daemon will deliver its result later, so the
    /// reply is a background handoff, not the terminal answer.
    background_handoff: AtomicBool,
}

impl Agent {
    async fn dispatch_sdk_tool_call(&self, scope: &TurnScope<'_>, request: ToolRequest) {
        let arguments = serde_json::to_string(&request.arguments).unwrap_or_else(|_| "{}".into());
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
        // Mandate runs: the same per-call authority gate as the legacy loop,
        // which issues the bounded grant or refuses the call.
        let grant = match self.sdk_mandate_gate(&ctx, &request, &arguments).await {
            Ok(grant) => grant,
            Err(refusal) => {
                info!(
                    session_id = scope.session_id,
                    task_id = scope.task_id,
                    tool = %request.name,
                    "SDK engine: mandate gate refused the call"
                );
                let _ = request.reply.send(Ok(sdk::ToolOutput::error(refusal)));
                return;
            }
        };
        ctx.mandate_authority = grant.as_ref();
        let result = match self
            .execute_tool_outcome(&request.name, &arguments, &ctx)
            .await
        {
            Ok(outcome) => {
                let is_error = matches!(
                    outcome.metadata.outcome_status,
                    Some(
                        ToolOutcomeStatus::FailedRetryable
                            | ToolOutcomeStatus::FailedPermanent
                            | ToolOutcomeStatus::Blocked
                    )
                );
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
                if !outcome.metadata.attachments.is_empty() {
                    if let Ok(mut by_call) = scope.tool_attachments.lock() {
                        by_call.insert(
                            request.call_id.clone(),
                            outcome.metadata.attachments.clone(),
                        );
                    }
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
                Ok(sdk::ToolOutput {
                    is_error,
                    ..sdk::ToolOutput::text(output).with_images(images)
                })
            }
            Err(error) => Ok(sdk::ToolOutput::error(format!("Error: {error}"))),
        };
        info!(
            session_id = scope.session_id,
            task_id = scope.task_id,
            tool = %request.name,
            failed = result.as_ref().map(|output| output.is_error).unwrap_or(true),
            duration_ms = started.elapsed().as_millis() as u64,
            "SDK engine tool call"
        );
        let _ = request.reply.send(result);
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
        Ok(from_provider_response(response))
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

    /// Record a run's tool calls and results as the legacy loop does
    /// (AssistantResponse with tool calls, then ToolResult), so later turns,
    /// history replay and diagnostics see the evidence behind the answer.
    #[allow(clippy::too_many_arguments)]
    async fn persist_sdk_tool_exchanges(
        &self,
        emitter: &EventEmitter,
        session_id: &str,
        task_id: &str,
        turn_id: &str,
        model: &str,
        produced: &[sdk::Message],
        receipts: &[sdk::Receipt],
        mut tool_attachments: HashMap<String, Vec<MessageAttachment>>,
    ) {
        let mut names: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        for message in produced {
            let persisted = match message.role {
                sdk::Role::Assistant if !message.tool_calls.is_empty() => {
                    let calls: Vec<crate::traits::ToolCall> = message
                        .tool_calls
                        .iter()
                        .map(|call| {
                            names.insert(call.id.as_str(), call.name.as_str());
                            crate::traits::ToolCall {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                arguments: call.arguments.to_string(),
                                extra_content: None,
                            }
                        })
                        .collect();
                    let mut assistant =
                        Message::new_runtime(Uuid::new_v4().to_string(), session_id, "assistant");
                    let text = message.content.trim();
                    assistant.content = (!text.is_empty()).then(|| text.to_string());
                    assistant.tool_calls_json = serde_json::to_string(&calls).ok();
                    assistant.turn_id = Some(turn_id.to_string());
                    self.append_assistant_message_with_event(emitter, &assistant, model, None, None)
                        .await
                }
                sdk::Role::Tool => {
                    let call_id = message.tool_call_id.clone().unwrap_or_default();
                    let receipt = receipts.iter().rev().find(|r| r.call_id == call_id);
                    let mut tool =
                        Message::new_runtime(Uuid::new_v4().to_string(), session_id, "tool");
                    tool.content = Some(message.content.clone());
                    tool.attachments = tool_attachments.remove(&call_id).unwrap_or_default();
                    tool.tool_name = names
                        .get(call_id.as_str())
                        .map(|name| name.to_string())
                        .or_else(|| receipt.map(|r| r.tool.clone()));
                    tool.tool_call_id = Some(call_id);
                    tool.turn_id = Some(turn_id.to_string());
                    self.append_tool_message_with_receipt_event_policy(
                        emitter,
                        &tool,
                        !message.is_error,
                        receipt.map(|r| r.duration_ms).unwrap_or_default(),
                        receipt.and_then(|r| r.reason.clone()),
                        Some(task_id),
                        None,
                        None,
                    )
                    .await
                }
                _ => continue,
            };
            if let Err(error) = persisted {
                warn!(session_id, task_id, %error, "SDK engine: could not record a tool exchange");
            }
        }
    }

    /// One turn on the SDK kernel. See the module docs for what is kept and
    /// what is dropped compared with the legacy loop.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_message_sdk(
        &self,
        engine: &SdkEngineSettings,
        session_id: &str,
        user_text: &str,
        attachments: &[MessageAttachment],
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
        parent_task_id: Option<&str>,
    ) -> anyhow::Result<String> {
        let started = Instant::now();
        let model = match &engine.model_override {
            Some(model) => model.clone(),
            None => {
                let snapshot = self.llm_runtime.snapshot();
                self.select_turn_model(
                    &snapshot,
                    snapshot.router().as_ref(),
                    crate::execution_policy::ModelProfile::Balanced,
                    session_id,
                )
                .await
                .0
            }
        };
        anyhow::ensure!(
            self.mandate_execution.is_none() || attachments.is_empty(),
            "mandate workers cannot receive attachment-derived context"
        );
        // Voice notes become text when the model cannot hear them, as in the
        // legacy bootstrap.
        let enriched = crate::agent::stt::maybe_enrich_user_text(
            user_text,
            attachments,
            &self.stt_config,
            &self.audio_config,
            &model,
        )
        .await;
        let user_text = enriched.as_str();
        touch_heartbeat(&heartbeat);
        let task_id = Uuid::new_v4().to_string();
        let turn_id = Uuid::new_v4().to_string();
        let emitter = EventEmitter::new(self.event_store.clone(), session_id.to_string())
            .with_task_id(task_id.clone());

        // Read prior dialogue before this turn's user message is recorded.
        let history = match self
            .event_store
            .get_conversation_history_since_clear(session_id, HISTORY_MESSAGES)
            .await
        {
            Ok(history) => history,
            Err(error) => {
                warn!(session_id, %error, "SDK engine: history unavailable");
                Vec::new()
            }
        };

        self.current_turn_ids
            .write()
            .await
            .insert(session_id.to_string(), turn_id.clone());
        emitter
            .emit(
                EventType::TaskStart,
                TaskStartData {
                    task_id: task_id.clone(),
                    description: user_text.chars().take(200).collect(),
                    parent_task_id: parent_task_id.map(str::to_string),
                    user_message: Some(user_text.to_string()),
                    turn_id: Some(turn_id.clone()),
                },
            )
            .await?;
        let mut user_message = Message::new_runtime(turn_id.clone(), session_id, "user");
        user_message.content = Some(user_text.to_string());
        user_message.turn_id = Some(turn_id.clone());
        user_message.attachments = attachments.to_vec();
        self.append_user_message_with_event(
            &emitter,
            &user_message,
            user_role,
            &channel_ctx,
            !attachments.is_empty(),
        )
        .await?;
        let (images, vision_skipped) = sdk_images(
            &attachments.iter().collect::<Vec<_>>(),
            &self.vision_config,
            &model,
        );

        // The legacy prompt builder: identity, matched skills, channel-scoped
        // memory facts, project instructions, and the isolated prompt for
        // mandate workers. Falls back to the base prompt if it fails.
        let tools_offered = if user_role == UserRole::Public {
            0
        } else {
            self.tools.len()
        };
        let instructions = match self
            .build_system_prompt_for_message(
                &emitter,
                &task_id,
                session_id,
                user_text,
                user_role,
                &channel_ctx,
                tools_offered,
                None,
                None,
                None,
                true,
                false,
                None,
            )
            .await
        {
            Ok((core, tail, _, skills, _)) => {
                if !skills.is_empty() {
                    info!(session_id, task_id, ?skills, "SDK engine: skills matched");
                }
                if tail.trim().is_empty() {
                    core
                } else {
                    format!("{core}\n\n{tail}")
                }
            }
            Err(error) => {
                warn!(session_id, task_id, %error, "SDK engine: prompt builder failed; using base prompt");
                self.system_prompt.clone()
            }
        };
        let instructions = if vision_skipped {
            format!("{instructions}\n\n{VISION_SKIPPED_SYSTEM_HINT}")
        } else {
            instructions
        };

        // Tools: every registered daemon tool, unless the caller may not use
        // tools at all (the dispatcher would refuse each call anyway).
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
        if user_role != UserRole::Public {
            for tool in &self.tools {
                builder = builder.tool(BridgedTool {
                    spec: tool_spec(tool.as_ref()),
                    read_only: tool.capabilities().read_only,
                    calls: calls_tx.clone(),
                });
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
            tool_attachments: Mutex::new(HashMap::new()),
            background_handoff: AtomicBool::new(false),
            emitter: &emitter,
            history_len: replayed,
            budget_goal,
            budget_exhausted: AtomicBool::new(false),
            model_calls: AtomicU64::new(0),
        };
        let run = sdk_agent.run_from(state, sdk::CancellationToken::new());
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
                // Everything this run added after the user message, except a
                // closing answer, which is persisted below as the reply.
                let mut produced =
                    &outcome.state.messages[(replayed + 1).min(outcome.state.messages.len())..];
                if produced.last().is_some_and(|message| {
                    message.role == sdk::Role::Assistant && message.tool_calls.is_empty()
                }) {
                    produced = &produced[..produced.len() - 1];
                }
                let tool_attachments = std::mem::take(
                    &mut *scope
                        .tool_attachments
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                );
                self.persist_sdk_tool_exchanges(
                    &emitter,
                    session_id,
                    &task_id,
                    &turn_id,
                    &model,
                    produced,
                    outcome.receipts(),
                    tool_attachments,
                )
                .await;
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
        let _ = emitter
            .emit(
                EventType::TaskEnd,
                TaskEndData {
                    task_id: task_id.clone(),
                    status,
                    outcome: Some(task_outcome),
                    duration_secs: started.elapsed().as_secs(),
                    iterations,
                    tool_calls_count,
                    error,
                    summary: None,
                    efficiency: None,
                    turn_id: Some(turn_id.clone()),
                    completion_proof: None,
                    harness_eval: None,
                },
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
        let calls = harness
            .agent
            .event_store
            .task_event_count(
                "telegram:synthetic-user-1",
                &reply.task_id,
                EventType::LlmCall,
            )
            .await
            .unwrap();
        assert_eq!(calls, 2);
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
