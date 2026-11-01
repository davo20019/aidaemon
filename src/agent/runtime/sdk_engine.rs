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
//! What it drops (on purpose, for comparison): intent gating, completion
//! contracts and the stall/recovery ladder.

use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use aidaemon_core as sdk;
use async_trait::async_trait;
use futures::stream::{FuturesUnordered, StreamExt};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};
use uuid::Uuid;

use super::agent_helpers::touch_heartbeat;
use super::tool_exec::ToolExecCtx;
use super::Agent;
use crate::config::{AgentEngineKind, AppConfig};
use crate::events::{EventEmitter, EventType, TaskEndData, TaskOutcome, TaskStartData, TaskStatus};
use crate::traits::{Message, ToolOutcomeStatus};
use crate::types::{ChannelContext, StatusUpdate, UserRole};

const DEFAULT_MODEL: &str = "claude-sonnet-5-5";
const DEFAULT_MAX_TURNS: u32 = 40;
/// Prior dialogue messages (including tool calls and results) replayed into
/// each run.
const HISTORY_MESSAGES: usize = 60;
/// Replayed tool results are cut to this many characters: enough to show what
/// an earlier call returned without re-sending large outputs every turn.
const REPLAYED_TOOL_RESULT_CHARS: usize = 2_000;
/// Daemon tools enforce their own timeouts (terminal backgrounding, CLI
/// agents); this only stops a wedged call from holding the turn forever.
const TOOL_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const HEARTBEAT_TICK: Duration = Duration::from_secs(15);

pub(crate) struct SdkEngine {
    provider: Arc<dyn sdk::Provider>,
    model: String,
    max_turns: u32,
}

static ENGINE: OnceLock<SdkEngine> = OnceLock::new();

/// Install the SDK engine when the config selects it. Returns whether it is
/// active. Called once at startup, before any channel starts.
pub(crate) fn install(config: &AppConfig) -> anyhow::Result<bool> {
    let engine = &config.agent.engine;
    if engine.kind != AgentEngineKind::Sdk {
        return Ok(false);
    }
    let api_key = if engine.api_key.trim().is_empty() {
        std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            anyhow::anyhow!(
                "[agent.engine] kind = \"sdk\" needs api_key (or ANTHROPIC_API_KEY in the environment)"
            )
        })?
    } else {
        engine.api_key.clone()
    };
    let provider = aidaemon_providers::Anthropic::builder()
        .api_key(api_key)
        .build()
        .map_err(|error| anyhow::anyhow!("SDK engine provider: {error}"))?;
    let model = engine
        .model
        .clone()
        .filter(|model| !model.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let installed = SdkEngine {
        provider: Arc::new(sdk::Retry::new(provider)),
        model: model.clone(),
        max_turns: engine.max_turns.unwrap_or(DEFAULT_MAX_TURNS).max(1),
    };
    if ENGINE.set(installed).is_err() {
        anyhow::bail!("SDK engine installed twice");
    }
    info!(model, "Agent engine: SDK kernel (experimental)");
    Ok(true)
}

pub(crate) fn active() -> Option<&'static SdkEngine> {
    ENGINE.get()
}

/// One tool call handed from the SDK run to the daemon dispatcher.
struct ToolRequest {
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
    calls: mpsc::Sender<ToolRequest>,
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
            .send(ToolRequest {
                name: self.spec.name.clone(),
                arguments,
                call_id: ctx.call_id.clone(),
                reply,
            })
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

/// Everything a dispatched call needs from the turn.
struct TurnScope<'a> {
    session_id: &'a str,
    task_id: &'a str,
    status_tx: Option<mpsc::Sender<StatusUpdate>>,
    channel_ctx: &'a ChannelContext,
    user_role: UserRole,
}

impl Agent {
    async fn dispatch_sdk_tool_call(&self, scope: &TurnScope<'_>, request: ToolRequest) {
        let arguments = serde_json::to_string(&request.arguments).unwrap_or_else(|_| "{}".into());
        let ctx = ToolExecCtx {
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
                Ok(sdk::ToolOutput {
                    is_error,
                    ..sdk::ToolOutput::text(outcome.output)
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
        engine: &SdkEngine,
        session_id: &str,
        user_text: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
        parent_task_id: Option<&str>,
    ) -> anyhow::Result<String> {
        let started = Instant::now();
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
        self.append_user_message_with_event(
            &emitter,
            &user_message,
            user_role,
            &channel_ctx,
            false,
        )
        .await?;

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

        // Tools: every registered daemon tool, unless the caller may not use
        // tools at all (the dispatcher would refuse each call anyway).
        let (calls_tx, mut calls_rx) = mpsc::channel::<ToolRequest>(16);
        let mut builder = sdk::Agent::builder("aidaemon")
            .model(engine.model.clone())
            .instructions(instructions)
            .provider(engine.provider.clone())
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
        let mut state = sdk::RunState::new("aidaemon", session_id, user_text);
        let mut messages = sdk_history(&history, &offered);
        let replayed = messages.len();
        messages.append(&mut state.messages);
        state.messages = messages;

        // Drive the run and the tool dispatcher together on this task.
        let scope = TurnScope {
            session_id,
            task_id: &task_id,
            status_tx: status_tx.clone(),
            channel_ctx: &channel_ctx,
            user_role,
        };
        let run = sdk_agent.run_from(state, sdk::CancellationToken::new());
        tokio::pin!(run);
        let mut in_flight = FuturesUnordered::new();
        let mut tick = tokio::time::interval(HEARTBEAT_TICK);
        let outcome = loop {
            tokio::select! {
                outcome = &mut run => break outcome,
                Some(request) = calls_rx.recv() => {
                    in_flight.push(self.dispatch_sdk_tool_call(&scope, request));
                }
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
                self.persist_sdk_tool_exchanges(
                    &emitter,
                    session_id,
                    &task_id,
                    &turn_id,
                    &engine.model,
                    produced,
                    outcome.receipts(),
                )
                .await;
                let usage = outcome.run_usage();
                let mut assistant =
                    Message::new_runtime(Uuid::new_v4().to_string(), session_id, "assistant");
                assistant.content = Some(reply.clone());
                assistant.turn_id = Some(turn_id.clone());
                self.append_assistant_message_with_event(
                    &emitter,
                    &assistant,
                    &engine.model,
                    Some(usage.input_tokens.min(u32::MAX as u64) as u32),
                    Some(usage.output_tokens.min(u32::MAX as u64) as u32),
                )
                .await?;
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

    /// Live check of the whole daemon path on the SDK engine: channel ingress
    /// envelope, a daemon tool dispatched through the bridge, and dialogue
    /// carried into the next turn. Needs a real key and installs the engine
    /// process-wide, so run it alone:
    /// `ANTHROPIC_API_KEY=... cargo test --features sdk-engine --lib live_sdk_engine -- --ignored`
    #[tokio::test]
    #[ignore = "calls the Anthropic API; needs ANTHROPIC_API_KEY"]
    async fn live_sdk_engine_runs_daemon_turns() {
        use crate::runtime_ports::{AgentIngress, InboundMessageRequest};
        use crate::traits::store_prelude::*;

        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY");
        let model =
            std::env::var("AIDAEMON_MODEL").unwrap_or_else(|_| "claude-haiku-5-5".to_string());
        let provider = aidaemon_providers::Anthropic::builder()
            .api_key(api_key)
            .build()
            .unwrap();
        let _ = ENGINE.set(SdkEngine {
            provider: Arc::new(sdk::Retry::new(provider)),
            model,
            max_turns: 10,
        });

        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let ask = |text: &str| InboundMessageRequest {
            session_id: "telegram:synthetic-user-1".to_string(),
            user_text: text.to_string(),
            attachments: Vec::new(),
            status_tx: None,
            user_role: UserRole::Owner,
            channel_ctx: ChannelContext::private("telegram"),
            heartbeat: None,
            ingress_timing: None,
        };

        let first = harness
            .agent
            .handle_inbound_message(ask(
                "Use the remember_fact tool to save that my favorite color is teal \
                 (category: preference, key: favorite_color). Then confirm in one sentence.",
            ))
            .await
            .unwrap();
        assert!(!first.response_id.is_empty() && !first.task_id.is_empty());
        let recorded_results = harness
            .agent
            .event_store
            .task_event_count(
                "telegram:synthetic-user-1",
                &first.task_id,
                EventType::ToolResult,
            )
            .await
            .unwrap();
        assert!(recorded_results >= 1, "tool result was not recorded");
        let facts = harness.state.get_facts(None).await.unwrap();
        assert!(
            facts
                .iter()
                .any(|fact| fact.value.to_lowercase().contains("teal")),
            "remember_fact was not dispatched: {facts:?}; reply: {}",
            first.text
        );

        let second = harness
            .agent
            .handle_inbound_message(ask(
                "Without using any tools: what favorite color did I mention earlier? One word.",
            ))
            .await
            .unwrap();
        assert!(
            second.text.to_lowercase().contains("teal"),
            "history not carried over: {}",
            second.text
        );
        assert_ne!(first.task_id, second.task_id);
        assert_eq!(
            harness.provider.call_count().await,
            0,
            "legacy provider must not be called on the SDK engine"
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
