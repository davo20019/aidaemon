use std::collections::HashSet;
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};

use crate::agent::{Agent, StatusUpdate};
use crate::channels::ChannelHub;
use crate::events::TaskOutcome;
use crate::traits::{AgentRole, StateStore, Tool, ToolCapabilities};
use crate::types::{ChannelContext, ChannelVisibility, UserRole};

/// A tool that allows the LLM to spawn a sub-agent for a focused task.
///
/// The sub-agent runs its own agentic loop with a dedicated session and a
/// system prompt that includes the given mission. Its final text response is
/// returned as the tool result.
///
/// The circular dependency (Agent → tools → SpawnAgentTool → Agent) is broken
/// by storing a `Weak<Agent>` inside a `OnceLock`. The weak reference is set
/// after the owning `Arc<Agent>` is constructed.
pub struct SpawnAgentTool {
    agent: OnceLock<Weak<Agent>>,
    hub: OnceLock<Weak<ChannelHub>>,
    state: Option<Arc<dyn StateStore>>,
    max_response_chars: usize,
    timeout_secs: u64,
    executor_task_runs: Arc<Mutex<HashSet<String>>>,
}

#[cfg(test)]
const BACKGROUND_PROGRESS_INTERVAL_SECS: u64 = 1;
#[cfg(not(test))]
const BACKGROUND_PROGRESS_INTERVAL_SECS: u64 = 20;

/// Fallback description used when the parent `Agent` isn't wired yet
/// (early bootstrap) or when the registry is empty.
const STATIC_SPECIALIST_ARG_DESCRIPTION: &str =
    "Optional. Pick the specialist profile matching this task. \
     Omit to let the agent infer one from mission/task text.";

/// Format the per-kind `(name, description)` pairs from
/// `SpecialistRegistry::llm_visible_kinds()` into the multi-line description
/// surfaced via the `specialist` parameter of `spawn_agent`. Kept as a free
/// function so tests can exercise the formatter without wiring a full Agent.
pub(crate) fn format_specialist_arg_description(entries: &[(&'static str, String)]) -> String {
    if entries.is_empty() {
        return STATIC_SPECIALIST_ARG_DESCRIPTION.to_string();
    }
    let mut s =
        String::from("Optional. Pick the specialist profile that best matches this task:\n");
    for (name, description) in entries {
        s.push_str("- ");
        s.push_str(name);
        s.push_str(": ");
        s.push_str(description);
        if !description.ends_with('.') {
            s.push('.');
        }
        s.push('\n');
    }
    s.push_str("Omit `specialist` to let the agent infer from the mission/task text.");
    s
}

impl SpawnAgentTool {
    /// Create a SpawnAgentTool with a known agent reference.
    #[allow(dead_code)]
    pub fn new(agent: Weak<Agent>, max_response_chars: usize, timeout_secs: u64) -> Self {
        let lock = OnceLock::new();
        let _ = lock.set(agent);
        Self {
            agent: lock,
            hub: OnceLock::new(),
            state: None,
            max_response_chars,
            timeout_secs,
            executor_task_runs: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Create a SpawnAgentTool with a deferred agent reference.
    /// Call [`set_agent`] after constructing the `Arc<Agent>`.
    pub fn new_deferred(max_response_chars: usize, timeout_secs: u64) -> Self {
        Self {
            agent: OnceLock::new(),
            hub: OnceLock::new(),
            state: None,
            max_response_chars,
            timeout_secs,
            executor_task_runs: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Set state store reference for queued background notifications when direct
    /// channel delivery is unavailable.
    pub fn with_state(mut self, state: Arc<dyn StateStore>) -> Self {
        self.state = Some(state);
        self
    }

    /// Set the agent reference. Must be called exactly once after the owning
    /// `Arc<Agent>` is constructed. Panics if called more than once.
    pub fn set_agent(&self, agent: Weak<Agent>) {
        self.agent
            .set(agent)
            .expect("SpawnAgentTool::set_agent called more than once");
    }

    fn get_agent(&self) -> anyhow::Result<std::sync::Arc<Agent>> {
        let weak = self
            .agent
            .get()
            .ok_or_else(|| anyhow::anyhow!("SpawnAgentTool: agent reference not set"))?;
        weak.upgrade()
            .ok_or_else(|| anyhow::anyhow!("SpawnAgentTool: parent agent has been dropped"))
    }

    /// Set the channel hub reference for background mode notifications.
    pub fn set_hub(&self, hub: Weak<ChannelHub>) {
        let _ = self.hub.set(hub);
    }

    fn get_hub(&self) -> Option<Arc<ChannelHub>> {
        self.hub.get().and_then(|w| w.upgrade())
    }

    /// Render the `specialist` parameter's description text, pulling each
    /// kind's frontmatter description from the live `SpecialistRegistry`
    /// (so user overrides at `~/.aidaemon/specialists/<kind>.md` flow
    /// through to the LLM-facing schema on next start).
    ///
    /// During early bootstrap the parent `Agent` may not yet be wired up
    /// (the weak ref hasn't been set). In that case we fall back to a
    /// static description with no per-kind text — the `enum` list still
    /// constrains the value, so the LLM can still pick a valid kind even
    /// without descriptions.
    fn build_specialist_arg_description(&self) -> String {
        let agent = match self.get_agent() {
            Ok(a) => a,
            Err(_) => {
                warn!(
                    "spawn_agent schema: parent agent not yet available; falling back to static specialist description"
                );
                return STATIC_SPECIALIST_ARG_DESCRIPTION.to_string();
            }
        };

        format_specialist_arg_description(&agent.specialists.llm_visible_kinds())
    }

    /// Acquire a per-task in-flight lock for executor spawns.
    async fn try_begin_executor_task(&self, task_id: &str) -> bool {
        let mut runs = self.executor_task_runs.lock().await;
        if runs.contains(task_id) {
            return false;
        }
        runs.insert(task_id.to_string());
        true
    }

    /// Release a per-task in-flight lock for executor spawns.
    async fn finish_executor_task(&self, task_id: &str) {
        self.executor_task_runs.lock().await.remove(task_id);
    }
}

/// Truncate a string to at most `max_chars` bytes without splitting a
/// multi-byte UTF-8 character. Returns the original string when it fits.
fn truncate_utf8(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        return s;
    }
    // Find the last char boundary at or before `max_chars`.
    let boundary = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= max_chars)
        .last()
        .unwrap_or(0);
    &s[..boundary]
}

fn format_background_completion(
    mission: &str,
    response: &str,
    outcome: TaskOutcome,
    max_response_chars: usize,
) -> (&'static str, String) {
    let text = truncate_utf8(response, max_response_chars);
    match outcome {
        TaskOutcome::Succeeded => (
            "completed",
            format!(
                "\u{2705} Background task complete\nMission: {}\n\n{}",
                mission, text
            ),
        ),
        TaskOutcome::Partial => (
            "partial",
            format!(
                "\u{26a0}\u{fe0f} Background task incomplete\nMission: {}\n\n{}",
                mission, text
            ),
        ),
        TaskOutcome::Failed => (
            "failed",
            format!(
                "\u{274c} Background task failed\nMission: {}\n\n{}",
                mission, text
            ),
        ),
    }
}

/// Parse a leading wait/delay prefix from a task string.
/// Examples: "wait for 2 minutes ...", "in 30 seconds ...", "after 1 hour ..."
fn parse_leading_wait_seconds(task: &str) -> Option<u64> {
    static LEADING_WAIT_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)^\s*(?:wait\s+(?:for\s+)?|in\s+|after\s+)(\d+)\s*(seconds?|secs?|s|minutes?|mins?|min|m|hours?|hrs?|h)\b",
        )
        .expect("leading wait regex should compile")
    });

    let caps = LEADING_WAIT_RE.captures(task.trim())?;
    let value: u64 = caps.get(1)?.as_str().parse().ok()?;
    let unit = caps.get(2)?.as_str().to_ascii_lowercase();
    match unit.as_str() {
        "s" | "sec" | "secs" | "second" | "seconds" => Some(value),
        "m" | "min" | "mins" | "minute" | "minutes" => Some(value.saturating_mul(60)),
        "h" | "hr" | "hrs" | "hour" | "hours" => Some(value.saturating_mul(3600)),
        _ => None,
    }
}

/// Strip a leading wait/delay prefix from a task string.
fn strip_leading_wait(task: &str) -> String {
    static STRIP_WAIT_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)^\s*(?:wait\s+(?:for\s+)?|in\s+|after\s+)\d+\s*(?:seconds?|secs?|s|minutes?|mins?|min|m|hours?|hrs?|h)\s*[,;]?\s*(?:then\s+|and\s+|,\s*)?",
        )
        .expect("strip wait regex should compile")
    });

    let remainder = STRIP_WAIT_RE.replace(task.trim(), "").to_string();
    let trimmed = remainder.trim().to_string();
    if trimmed.len() < 3 {
        String::new()
    } else {
        trimmed
    }
}

async fn deliver_background_notification(
    hub: Option<&Arc<ChannelHub>>,
    state: Option<&Arc<dyn StateStore>>,
    goal_id: &str,
    session_id: &str,
    notification_type: &str,
    message: &str,
    context: &str,
) {
    let mut delivered = false;
    if let Some(hub_arc) = hub {
        if let Err(e) = hub_arc.send_text(session_id, message).await {
            warn!(
                session_id = %session_id,
                goal_id = %goal_id,
                notification_type = %notification_type,
                error = %e,
                "{context}: direct hub delivery failed"
            );
        } else {
            delivered = true;
        }
    }

    if delivered {
        return;
    }

    if let Some(state_store) = state {
        let entry =
            crate::traits::NotificationEntry::new(goal_id, session_id, notification_type, message);
        if let Err(e) = state_store.enqueue_notification(&entry).await {
            warn!(
                session_id = %session_id,
                goal_id = %goal_id,
                notification_type = %notification_type,
                error = %e,
                "{context}: enqueue fallback failed"
            );
        }
    } else {
        warn!(
            session_id = %session_id,
            goal_id = %goal_id,
            notification_type = %notification_type,
            "{context}: no hub and no queue fallback configured; update dropped"
        );
    }
}

#[derive(Deserialize)]
struct SpawnArgs {
    /// High-level mission / role description for the sub-agent.
    mission: String,
    /// The concrete task or question the sub-agent should work on.
    task: String,
    /// When true, spawn the sub-agent in the background and return immediately.
    #[serde(default)]
    background: bool,
    /// Task ID — when provided by a task lead, the executor tracks activity against this task.
    #[serde(default)]
    task_id: Option<String>,
    /// Optional specialist profile hint. The parent LLM can pick one of the supported
    /// kinds (code, browser_verifier, artifact_writer, research, review, comms_draft,
    /// executor, generic) so the child agent uses the matching profile. Threaded
    /// through `spawn_child`; Task 12 wires it into kind resolution.
    #[serde(default)]
    specialist: Option<String>,
    /// Session ID injected by execute_tool — used for background completion notifications.
    #[serde(default)]
    _session_id: Option<String>,
    /// Channel visibility injected by execute_tool — propagated to child agents.
    #[serde(default)]
    _channel_visibility: Option<String>,
    /// User role injected by execute_tool — propagated to child agents to prevent
    /// privilege escalation (e.g., Guest user spawning Owner-level sub-agent).
    #[serde(default)]
    _user_role: Option<String>,
    /// Task ID injected by execute_tool for task-lead → executor spawning.
    #[serde(default)]
    _task_id: Option<String>,
    /// Goal ID injected by execute_tool for task-lead → executor spawning.
    #[serde(default)]
    _goal_id: Option<String>,
    /// Explicit trust flag injected by execute_tool for trusted scheduled runs.
    #[serde(default)]
    _trusted_session: Option<bool>,
    /// Parent project scope injected by execute_tool for child scope inheritance.
    #[serde(default)]
    _project_scope: Option<String>,
}

fn build_child_channel_context(args: &SpawnArgs) -> ChannelContext {
    let visibility = args
        ._channel_visibility
        .as_deref()
        .map(ChannelVisibility::from_str_lossy)
        .unwrap_or(ChannelVisibility::Internal);
    ChannelContext {
        visibility,
        platform: "internal".to_string(),
        channel_name: None,
        channel_id: None,
        sender_name: None,
        sender_id: None,
        channel_member_names: vec![],
        user_id_map: std::collections::HashMap::new(),
        trusted: args._trusted_session.unwrap_or(false),
    }
}

/// Deterministic, conservative check: does this task describe a single bounded,
/// read-only command (disk free, count files, git status, list/find largest)
/// that the orchestrator should run inline with `terminal` rather than pay a
/// ~18s cold prefill + a full sub-agent loop to delegate?
///
/// Errs HARD toward `false` (allow the spawn): a missed trivial task only costs
/// one spawn, but wrongly inlining real work would block a sub-agent that was
/// needed. Anything generative, mutating, exploratory, or multi-step → `false`.
/// No LLM involved — pure keyword rules.
pub(crate) fn should_run_inline(mission_and_task: &str) -> bool {
    let text = mission_and_task.to_ascii_lowercase();
    let has = |needles: &[&str]| {
        needles
            .iter()
            .any(|n| crate::agent::keyword_match(&text, n))
    };

    // Generative / mutating / exploratory / multi-step work needs a real agent
    // loop — never inline these.
    const SPAWN_MARKERS: &[&str] = &[
        "write",
        "create",
        "generate",
        "compose",
        "draft",
        "summary",
        "summarize",
        "a report",
        "the report",
        "publish",
        "deploy",
        "install",
        "modify",
        "delete",
        "remove",
        "push",
        "send",
        "migrate",
        "refactor",
        "build",
        "fix",
        "research",
        "analyze",
        "investigate",
        "explore",
        "review",
        "audit",
        "implement",
        "diagnose",
    ];
    if has(SPAWN_MARKERS) {
        return false;
    }
    // Multi-step markers.
    if text.contains("step 1") || text.contains(" then ") || text.matches(". ").count() >= 2 {
        return false;
    }

    // Recognized single read-only data fetch.
    const INLINE_MARKERS: &[&str] = &[
        "disk free",
        "free space",
        "disk space",
        "disk usage",
        "git branch",
        "current branch",
        "last commit",
        "git status",
        "git log",
        "largest files",
        "biggest files",
        "list files",
        "uptime",
        "memory usage",
        "cpu usage",
    ];
    if has(INLINE_MARKERS) {
        return true;
    }
    // "count … files" / "number of … files" is a file count (a command); but
    // "count the bugs" is analysis — require the "files" noun to inline.
    (has(&["count"]) || text.contains("number of")) && has(&["files"])
}

#[async_trait]
impl Tool for SpawnAgentTool {
    fn name(&self) -> &str {
        "spawn_agent"
    }

    fn description(&self) -> &str {
        "Delegate a focused task to a fresh sub-agent. PREFER delegation when the work is multi-step \
         (research → synthesize, code → test, fetch → analyze), matches a named specialist (see the \
         `specialist` enum: code, browser_verifier, research, review, artifact_writer, comms_draft, ...), \
         benefits from a fresh context window, or can run in parallel with other work. Do NOT delegate \
         trivial one-shot reads (a single read_file, a single web_search) — handle those directly. \
         The sub-agent starts with a fresh context and the same tools, so reference files by path and \
         do not paste prior tool output."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "spawn_agent",
            "description": "Delegate a focused task to a fresh sub-agent. PREFER delegation when the work \
                is multi-step (research → synthesize, code → test, fetch → analyze), matches a named \
                specialist (see the `specialist` enum: code, browser_verifier, research, review, \
                artifact_writer, comms_draft, ...), benefits from a fresh context window, or can run in \
                parallel with other work. Do NOT delegate trivial one-shot reads (a single read_file, a \
                single web_search) — handle those directly. The sub-agent starts with a fresh context and \
                the same tools, so reference files by path and do not paste prior tool output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mission": {
                        "type": "string",
                        "description": "High-level mission or role for the sub-agent \
                            (e.g. 'Research assistant focused on Python packaging')"
                    },
                    "task": {
                        "type": "string",
                        "description": "The specific task or question the sub-agent should accomplish. \
                            Keep it self-contained but concise: state the goal, success criteria, and any \
                            non-obvious constraints. Do NOT paste file contents, prior tool output, or \
                            conversation excerpts — give file paths and let the sub-agent read them itself."
                    },
                    "background": {
                        "type": "boolean",
                        "description": "When true, spawn the sub-agent in the background and return immediately. \
                            The result will be delivered through the parent session when the sub-agent finishes. \
                            Use this for long-running tasks where the user doesn't need to wait.",
                        "default": false
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to associate with this executor (used by task leads to connect executor work to task tracking)"
                    },
                    "specialist": {
                        "type": "string",
                        "enum": [
                            "code",
                            "browser_verifier",
                            "artifact_writer",
                            "research",
                            "review",
                            "comms_draft",
                            "executor",
                            "generic"
                        ],
                        "description": self.build_specialist_arg_description()
                    }
                },
                "required": ["mission", "task"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            high_impact_write: true,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        self.call_with_status(arguments, None).await
    }

    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        let args: SpawnArgs = serde_json::from_str(arguments)?;
        let agent = self.get_agent()?;

        info!(
            depth = agent.depth(),
            max_depth = agent.max_depth(),
            mission = %args.mission,
            background = args.background,
            "spawn_agent tool invoked"
        );

        // Preserve execution context that was injected by the parent agent.
        let channel_ctx = build_child_channel_context(&args);

        // Propagate parent's user role to prevent privilege escalation.
        // Default to Guest (least privilege) if not provided.
        let user_role = match args._user_role.as_deref() {
            Some("Owner") => UserRole::Owner,
            Some("Guest") => UserRole::Guest,
            _ => UserRole::Guest,
        };

        // Determine child role based on parent's role.
        let child_role = if agent.role() == AgentRole::TaskLead {
            Some(AgentRole::Executor)
        } else {
            None
        };

        // Inline gate: a single bounded read-only command (disk free, count
        // files, git status, find largest …) doesn't warrant a fresh sub-agent
        // — that pays a ~18s cold prefill + a full loop to run a ~1s command.
        // Refuse the spawn and have the orchestrator run it inline with
        // `terminal` in its warm context. Deterministic + conservative: only the
        // clearly-trivial cases are blocked; anything ambiguous still spawns.
        if child_role == Some(AgentRole::Executor) && should_run_inline(&args.task) {
            info!(task = %args.task, "spawn refused: single read-only command, run inline");
            return Ok(format!(
                "Run this inline — \"{}\" is a single read-only command, not work that needs a \
                 sub-agent. Execute it yourself with the `terminal` tool, then record the result \
                 via manage_goal_tasks(action=\"update_task\", status=\"completed\", result=...). \
                 Do NOT call spawn_agent for trivial commands.",
                args.task
            ));
        }

        // LLM-provided task_id takes priority; fall back to injected _task_id
        let task_id_ref = args.task_id.or(args._task_id.clone());
        let goal_id_ref = args._goal_id.clone();

        // TaskLead -> Executor spawns must target a concrete, pre-claimed task and
        // are deduplicated per task_id so the same work is not launched twice.
        let executor_task_id = if child_role == Some(AgentRole::Executor) {
            let Some(task_id) = task_id_ref.clone() else {
                return Ok(
                    "Blocked: TaskLead must pass task_id when spawning an executor. Claim a task first with manage_goal_tasks(action='claim_task')."
                        .to_string(),
                );
            };
            if let Err(e) = agent
                .validate_executor_task_for_spawn(&task_id, goal_id_ref.as_deref())
                .await
            {
                return Ok(format!(
                    "Blocked executor spawn for task {}: {}",
                    task_id, e
                ));
            }
            if !self.try_begin_executor_task(&task_id).await {
                return Ok(format!(
                    "Blocked: task {} already has an executor running. Wait for it to finish before spawning another.",
                    task_id
                ));
            }
            Some(task_id)
        } else {
            None
        };

        // Fast-path leading wait tasks to avoid burning tokens in a child loop.
        // This covers direct spawn_agent executor calls (which bypass task-lead
        // auto-dispatch wait interception).
        let mut effective_mission = args.mission.clone();
        let mut effective_task = args.task.clone();
        if let Some(wait_secs) = parse_leading_wait_seconds(&effective_task) {
            let remainder = strip_leading_wait(&effective_task);
            info!(
                wait_secs,
                has_remainder = !remainder.is_empty(),
                "Intercepted leading wait in spawn_agent task; sleeping locally"
            );
            tokio::time::sleep(Duration::from_secs(wait_secs)).await;

            if remainder.is_empty() {
                if let Some(ref task_id) = executor_task_id {
                    self.finish_executor_task(task_id).await;
                }
                return Ok(format!("Waited for {} second(s).", wait_secs));
            }

            effective_task = remainder.clone();
            // Keep mission aligned when it appears to mirror task input.
            if parse_leading_wait_seconds(&effective_mission).is_some()
                || effective_mission
                    .trim()
                    .eq_ignore_ascii_case(args.task.trim())
            {
                effective_mission = remainder;
            }
        }

        let approval_session_id = args._session_id.clone();

        if !args.background {
            let result = self
                .run_sync(
                    agent,
                    &effective_mission,
                    &effective_task,
                    status_tx,
                    channel_ctx,
                    user_role,
                    child_role,
                    goal_id_ref.as_deref(),
                    task_id_ref.as_deref(),
                    args._project_scope.as_deref(),
                    args.specialist.as_deref(),
                    approval_session_id.as_deref(),
                )
                .await;
            if let Some(ref task_id) = executor_task_id {
                self.finish_executor_task(task_id).await;
            }
            return result;
        }

        // Background mode: need at least one completion delivery path.
        let hub = self.get_hub();
        let state = self.state.clone();
        let hub_for_parent_delivery = hub.as_ref().map(Arc::downgrade);
        if hub.is_none() && state.is_none() {
            info!(
                "Background mode requested but no hub/state notification path is available, falling back to sync"
            );
            let result = self
                .run_sync(
                    agent,
                    &effective_mission,
                    &effective_task,
                    status_tx,
                    channel_ctx,
                    user_role,
                    child_role,
                    goal_id_ref.as_deref(),
                    task_id_ref.as_deref(),
                    args._project_scope.as_deref(),
                    args.specialist.as_deref(),
                    approval_session_id.as_deref(),
                )
                .await;
            if let Some(ref task_id) = executor_task_id {
                self.finish_executor_task(task_id).await;
            }
            return result;
        }
        let session_id = match approval_session_id.as_deref() {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => {
                info!("Background mode requested but no session_id, falling back to sync");
                let result = self
                    .run_sync(
                        agent,
                        &effective_mission,
                        &effective_task,
                        status_tx,
                        channel_ctx,
                        user_role,
                        child_role,
                        goal_id_ref.as_deref(),
                        task_id_ref.as_deref(),
                        args._project_scope.as_deref(),
                        args.specialist.as_deref(),
                        approval_session_id.as_deref(),
                    )
                    .await;
                if let Some(ref task_id) = executor_task_id {
                    self.finish_executor_task(task_id).await;
                }
                return result;
            }
        };

        let task = effective_task.clone();
        let mission = effective_mission.clone();
        let timeout_secs = self.timeout_secs;
        let max_response_chars = self.max_response_chars;
        let executor_task_runs = Arc::clone(&self.executor_task_runs);
        let executor_task_id_for_bg = executor_task_id;
        let notify_goal_id = goal_id_ref.clone().unwrap_or_else(|| "global".to_string());
        let notify_status_tx = status_tx.clone();

        tokio::spawn(async move {
            let started_at = std::time::Instant::now();
            let mut progress_interval =
                tokio::time::interval(Duration::from_secs(BACKGROUND_PROGRESS_INTERVAL_SECS));
            progress_interval.tick().await; // consume immediate tick
            let timeout_duration = Duration::from_secs(timeout_secs);
            let arg_specialist_owned: Option<String> = args.specialist.clone();
            let arg_specialist = arg_specialist_owned.as_deref();
            let mut result_fut = std::pin::pin!(tokio::time::timeout(
                timeout_duration,
                agent.spawn_child_with_outcome(
                    &mission,
                    &task,
                    status_tx.clone(),
                    channel_ctx,
                    user_role,
                    child_role,
                    goal_id_ref.as_deref(),
                    task_id_ref.as_deref(),
                    args._project_scope.as_deref(),
                    arg_specialist,
                    Some(&session_id),
                ),
            ));
            let result = loop {
                tokio::select! {
                    res = &mut result_fut => break res,
                    _ = progress_interval.tick() => {
                        let elapsed_secs = started_at.elapsed().as_secs();
                        let progress_message = format!(
                            "Background sub-agent still running after {}s.\nMission: {}",
                            elapsed_secs, mission
                        );
                        if let Some(ref tx) = notify_status_tx {
                            let _ = tx.try_send(StatusUpdate::ToolProgress {
                                name: "spawn_agent".to_string(),
                                chunk: format!(
                                    "Background sub-agent still running ({}s): {}",
                                    elapsed_secs, mission
                                ),
                            });
                        }
                        deliver_background_notification(
                            hub.as_ref(),
                            state.as_ref(),
                            &notify_goal_id,
                            &session_id,
                            "progress",
                            &progress_message,
                            "spawn_agent background progress notifier",
                        )
                        .await;
                    }
                }
            };

            let (notification_type, message) = match result {
                Ok(Ok(run)) => format_background_completion(
                    &mission,
                    &run.response,
                    run.outcome,
                    max_response_chars,
                ),
                Ok(Err(e)) => (
                    "failed",
                    format!(
                        "\u{274c} Background task failed\nMission: {}\nError: {}",
                        mission, e
                    ),
                ),
                Err(_) => {
                    // The child may have persisted a terminal outcome before
                    // the timeout cancelled it — deliver that instead of a
                    // generic timeout failure.
                    let salvaged = match task_id_ref.as_deref() {
                        Some(task_id) => {
                            agent
                                .salvage_executor_task_outcome(task_id, timeout_secs)
                                .await
                        }
                        None => None,
                    };
                    match salvaged {
                        Some(outcome) if outcome.status == "completed" => (
                            "completed",
                            format!(
                                "\u{2705} Background task complete\nMission: {}\n\n{}",
                                mission, outcome.details
                            ),
                        ),
                        Some(outcome) if outcome.status == "blocked" => (
                            "blocked",
                            format!(
                                "\u{26a0}\u{fe0f} Background task blocked\nMission: {}\n\n{}",
                                mission, outcome.details
                            ),
                        ),
                        Some(outcome) => (
                            "failed",
                            format!(
                                "\u{274c} Background task failed\nMission: {}\n\n{}",
                                mission, outcome.details
                            ),
                        ),
                        None => (
                            "failed",
                            format!(
                                "\u{23f1} Background task timed out\nMission: {}\nTimed out after {}s",
                                mission, timeout_secs
                            ),
                        ),
                    }
                }
            };
            let delivered = match agent
                .deliver_parent_text_result(
                    hub_for_parent_delivery.as_ref(),
                    &session_id,
                    &message,
                    crate::agent::ParentDeliveryKind::BackgroundSpawnResult,
                )
                .await
            {
                Ok(outcome) => outcome.sent,
                Err(e) => {
                    warn!(
                        session_id = %session_id,
                        goal_id = %notify_goal_id,
                        notification_type = %notification_type,
                        error = %e,
                        "spawn_agent background completion notifier: parent delivery failed"
                    );
                    false
                }
            };

            if !delivered {
                deliver_background_notification(
                    None,
                    state.as_ref(),
                    &notify_goal_id,
                    &session_id,
                    notification_type,
                    &message,
                    "spawn_agent background completion notifier",
                )
                .await;
            }

            if let Some(task_id) = executor_task_id_for_bg {
                executor_task_runs.lock().await.remove(&task_id);
            }
        });

        Ok(format!(
            "Sub-agent spawned in background for mission: \"{}\". \
             The result will be delivered through the parent session when it completes.",
            args.mission
        ))
    }
}

impl SpawnAgentTool {
    /// Run the sub-agent synchronously (blocking until completion or timeout).
    #[allow(clippy::too_many_arguments)]
    async fn run_sync(
        &self,
        agent: Arc<Agent>,
        mission: &str,
        task: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
        child_role: Option<AgentRole>,
        goal_id: Option<&str>,
        task_id: Option<&str>,
        project_scope: Option<&str>,
        arg_specialist: Option<&str>,
        approval_session_id: Option<&str>,
    ) -> anyhow::Result<String> {
        let timeout_duration = Duration::from_secs(self.timeout_secs);
        let result = tokio::time::timeout(
            timeout_duration,
            agent.spawn_child_with_outcome(
                mission,
                task,
                status_tx,
                channel_ctx,
                user_role,
                child_role,
                goal_id,
                task_id,
                project_scope,
                arg_specialist,
                approval_session_id,
            ),
        )
        .await;

        match result {
            Ok(Ok(run)) => {
                let response = match run.outcome {
                    TaskOutcome::Succeeded => run.response,
                    TaskOutcome::Partial => {
                        format!("\u{26a0}\u{fe0f} Sub-agent incomplete\n\n{}", run.response)
                    }
                    TaskOutcome::Failed => {
                        format!("\u{274c} Sub-agent failed\n\n{}", run.response)
                    }
                };
                let max_len = self.max_response_chars;
                if response.len() > max_len {
                    let truncated = truncate_utf8(&response, max_len);
                    Ok(format!(
                        "{}\n\n[Sub-agent response truncated at {} chars]",
                        truncated, max_len
                    ))
                } else {
                    Ok(response)
                }
            }
            Ok(Err(e)) => Ok(format!("Error: specialist failed: {}", e)),
            Err(_) => {
                if let Some(task_id) = task_id {
                    // The child may have persisted a terminal outcome (e.g.
                    // report_blocker) before the timeout cancelled it — use
                    // that instead of discarding the work.
                    if let Some(salvaged) = agent
                        .salvage_executor_task_outcome(task_id, self.timeout_secs)
                        .await
                    {
                        let prefix = match salvaged.status.as_str() {
                            "completed" => "\u{2705} Sub-agent complete",
                            "blocked" => "\u{26a0}\u{fe0f} Sub-agent blocked",
                            _ => "\u{274c} Sub-agent failed",
                        };
                        return Ok(format!("{prefix}\n\n{}", salvaged.details));
                    }
                    if child_role == Some(AgentRole::Executor) {
                        agent
                            .mark_executor_task_timeout(task_id, self.timeout_secs)
                            .await;
                    }
                }
                Ok(format!(
                    "Error: specialist timed out after {} seconds",
                    self.timeout_secs
                ))
            }
        }
    }
}

#[cfg(test)]
mod inline_gate_tests {
    use super::should_run_inline;

    #[test]
    fn inlines_single_read_only_commands() {
        // Real tasks from telemetry that each got a wasteful ~30s sub-agent.
        for t in [
            "Report disk free space on / using 'df -h /'",
            "Check current disk free space on /",
            "Count the number of .rs files in ~/projects/aidaemon/src",
            "Count how many .rs files are in ~/projects/aidaemon/src",
            "Determine the current git branch and the subject of the last commit",
            "Get the current git branch and the last commit subject",
            "Find the 3 largest files under ~/projects/aidaemon",
        ] {
            assert!(should_run_inline(t), "should inline: {t}");
        }
    }

    #[test]
    fn spawns_for_substantial_or_mutating_work() {
        for t in [
            "Write a self-test summary to ~/aidaemon-selftest.md",
            "Write a summary report to a markdown file.",
            "Research the latest Rust async patterns and summarize them",
            "Refactor the auth module across 8 files and run the tests",
            "Deploy the blog to Cloudflare",
            "Count the bugs in the codebase and analyze the root causes",
            "Investigate why scheduled goals fail to authenticate",
        ] {
            assert!(!should_run_inline(t), "should spawn (not inline): {t}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{NotificationStore, StateStore};
    use std::sync::Arc;

    #[test]
    fn truncate_utf8_ascii() {
        assert_eq!(truncate_utf8("hello world", 5), "hello");
        assert_eq!(truncate_utf8("hello", 10), "hello");
        assert_eq!(truncate_utf8("hello", 5), "hello");
    }

    #[test]
    fn truncate_utf8_multibyte() {
        // Each emoji is 4 bytes
        let s = "🔥🔥🔥";
        assert_eq!(s.len(), 12);
        // Limit 4 should include exactly the first emoji
        assert_eq!(truncate_utf8(s, 4), "🔥");
        // Limit 5 should still only include the first emoji (next starts at 4, ends at 8)
        assert_eq!(truncate_utf8(s, 5), "🔥");
        // Limit 8 should include two emojis
        assert_eq!(truncate_utf8(s, 8), "🔥🔥");
        // Limit 1 — no full character fits, but char_indices first is (0, '🔥')
        // take_while(|&i| i <= 1) → only i=0 qualifies
        assert_eq!(truncate_utf8(s, 1), "");
    }

    #[test]
    fn truncate_utf8_mixed() {
        let s = "hi🌍!";
        // 'h'=1, 'i'=1, '🌍'=4, '!'=1 → total 7
        assert_eq!(truncate_utf8(s, 3), "hi");
        assert_eq!(truncate_utf8(s, 6), "hi🌍");
        assert_eq!(truncate_utf8(s, 7), "hi🌍!");
    }

    #[test]
    fn truncate_utf8_empty() {
        assert_eq!(truncate_utf8("", 10), "");
        assert_eq!(truncate_utf8("", 0), "");
    }

    #[test]
    fn partial_background_result_is_not_labeled_complete() {
        let (notification_type, message) = format_background_completion(
            "Build and verify the site",
            "Blocked on workspace access.",
            TaskOutcome::Partial,
            8_000,
        );
        assert_eq!(notification_type, "partial");
        assert!(message.starts_with("\u{26a0}\u{fe0f} Background task incomplete"));
        assert!(!message.contains("\u{2705} Background task complete"));
    }

    #[test]
    fn successful_background_result_keeps_completion_label() {
        let (notification_type, message) = format_background_completion(
            "Build and verify the site",
            "Build passed.",
            TaskOutcome::Succeeded,
            8_000,
        );
        assert_eq!(notification_type, "completed");
        assert!(message.starts_with("\u{2705} Background task complete"));
    }

    #[test]
    fn deferred_initialization_not_set() {
        let tool = SpawnAgentTool::new_deferred(8000, 300);
        let result = tool.get_agent();
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("not set"));
    }

    #[test]
    fn config_defaults() {
        use crate::config::SubagentsConfig;
        let cfg = SubagentsConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.max_depth, 3);
        assert_eq!(cfg.max_iterations, 10);
        assert_eq!(cfg.max_response_chars, 8000);
        assert_eq!(cfg.timeout_secs, 300);
    }

    #[test]
    fn deferred_hub_not_set() {
        let tool = SpawnAgentTool::new_deferred(8000, 300);
        assert!(tool.get_hub().is_none());
    }

    #[test]
    fn spawn_args_background_default() {
        let json = r#"{"mission": "test", "task": "do stuff"}"#;
        let args: SpawnArgs = serde_json::from_str(json).unwrap();
        assert!(!args.background);
        assert!(args._session_id.is_none());
        assert!(args._channel_visibility.is_none());
    }

    #[test]
    fn spawn_args_background_true() {
        let json = r#"{"mission": "test", "task": "do stuff", "background": true, "_session_id": "tg:123"}"#;
        let args: SpawnArgs = serde_json::from_str(json).unwrap();
        assert!(args.background);
        assert_eq!(args._session_id.as_deref(), Some("tg:123"));
    }

    #[test]
    fn spawn_args_with_channel_visibility() {
        let json = r#"{"mission": "test", "task": "do stuff", "_channel_visibility": "public"}"#;
        let args: SpawnArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args._channel_visibility.as_deref(), Some("public"));
    }

    #[test]
    fn spawn_args_with_trusted_session() {
        let json = r#"{"mission":"test","task":"do stuff","_trusted_session":true,"_channel_visibility":"internal"}"#;
        let args: SpawnArgs = serde_json::from_str(json).unwrap();
        let channel_ctx = build_child_channel_context(&args);
        assert_eq!(channel_ctx.visibility, ChannelVisibility::Internal);
        assert!(channel_ctx.trusted);
    }

    #[test]
    fn parse_and_strip_leading_wait() {
        assert_eq!(
            parse_leading_wait_seconds("wait for 2 minutes then run df"),
            Some(120)
        );
        assert_eq!(
            strip_leading_wait("wait for 2 minutes then run df"),
            "run df"
        );
        assert_eq!(parse_leading_wait_seconds("in 45 sec check disk"), Some(45));
        assert_eq!(strip_leading_wait("after 1 hour, reboot"), "reboot");
    }

    #[test]
    fn strip_leading_wait_pure_wait_returns_empty() {
        assert_eq!(parse_leading_wait_seconds("wait 5 min"), Some(300));
        assert!(strip_leading_wait("wait 5 min").is_empty());
    }

    #[tokio::test]
    async fn executor_task_lock_deduplicates_concurrent_spawns() {
        let tool = SpawnAgentTool::new_deferred(8000, 300);
        assert!(tool.try_begin_executor_task("task-1").await);
        assert!(
            !tool.try_begin_executor_task("task-1").await,
            "Second acquire should be rejected while first is active"
        );
        tool.finish_executor_task("task-1").await;
        assert!(
            tool.try_begin_executor_task("task-1").await,
            "Task lock should be reusable after release"
        );
    }

    #[tokio::test]
    async fn background_notification_falls_back_to_queue_when_hub_missing() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let state_dyn: Arc<dyn StateStore> = state.clone();

        deliver_background_notification(
            None,
            Some(&state_dyn),
            "goal_spawn_test",
            "sess_spawn_test",
            "progress",
            "Background sub-agent still running after 20s.\nMission: test",
            "spawn_test",
        )
        .await;

        let pending = state.get_pending_notifications(10).await.unwrap();
        assert!(pending.iter().any(|entry| {
            entry.goal_id == "goal_spawn_test"
                && entry.session_id == "sess_spawn_test"
                && entry.notification_type == "progress"
                && entry.message.contains("still running")
        }));
    }
}

#[cfg(test)]
mod specialist_arg_tests {
    use super::*;
    use serde_json::Value;

    fn schema_props() -> Value {
        let tool = SpawnAgentTool::new_deferred(8192, 60);
        let schema = tool.schema();
        schema
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .cloned()
            .expect("schema has properties")
    }

    #[test]
    fn schema_advertises_specialist_arg_with_enum() {
        let props = schema_props();
        let specialist = props
            .get("specialist")
            .expect("specialist property declared in schema");
        let kinds = specialist
            .get("enum")
            .and_then(|v| v.as_array())
            .expect("specialist has enum array");
        let names: Vec<&str> = kinds.iter().filter_map(|v| v.as_str()).collect();
        for expected in [
            "code",
            "browser_verifier",
            "artifact_writer",
            "research",
            "review",
            "comms_draft",
            "executor",
            "generic",
        ] {
            assert!(
                names.contains(&expected),
                "missing {} from enum: {:?}",
                expected,
                names
            );
        }
        // task_lead is NOT a parent-LLM-selectable value (role-typed only).
        assert!(!names.contains(&"task_lead"));
    }

    #[test]
    fn format_specialist_arg_description_lists_every_kind_with_text() {
        let registry = crate::agent::specialists::SpecialistRegistry::load(None);
        let entries = registry.llm_visible_kinds();
        let desc = format_specialist_arg_description(&entries);

        // Each non-task_lead kind appears as a `- name:` bullet, and its
        // bundled frontmatter description text is included verbatim.
        for kind in [
            "code",
            "browser_verifier",
            "artifact_writer",
            "research",
            "review",
            "comms_draft",
            "executor",
            "generic",
        ] {
            let marker = format!("- {}:", kind);
            assert!(
                desc.contains(&marker),
                "specialist description missing bullet for {}: {}",
                kind,
                desc
            );
        }

        // task_lead must NOT appear — it's role-typed, not LLM-selectable.
        assert!(!desc.contains("- task_lead:"));
        assert!(!desc.contains("task_lead:"));

        // Spot-check the actual frontmatter descriptions surface through.
        // (Test asserts the registry's description text reached the formatter.)
        let code_desc = entries
            .iter()
            .find(|(n, _)| *n == "code")
            .map(|(_, d)| d.clone())
            .expect("code kind present");
        assert!(
            desc.contains(&code_desc),
            "code description not surfaced: {}",
            desc
        );

        // Closing line tells the model omission is allowed.
        assert!(desc.contains("Omit `specialist`"));
    }

    #[test]
    fn schema_falls_back_to_static_description_when_agent_unwired() {
        let tool = SpawnAgentTool::new_deferred(8192, 60);
        let schema = tool.schema();
        let desc = schema
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.get("specialist"))
            .and_then(|s| s.get("description"))
            .and_then(|d| d.as_str())
            .expect("specialist description string");
        assert_eq!(desc, STATIC_SPECIALIST_ARG_DESCRIPTION);
    }
}
