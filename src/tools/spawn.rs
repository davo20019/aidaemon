use std::collections::HashSet;
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use async_trait::async_trait;
use futures::FutureExt as _;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};

use crate::events::TaskOutcome;
use crate::runtime_ports::{
    ChildAgentRequest, ChildAgentRun, ChildAgentRuntime, ChildTimeLimit, ChildTimedOut,
    OutboundRouter,
};
use crate::traits::{
    AgentRole, StateStore, Tool, ToolCallContext, ToolCallOutcome, ToolCapabilities,
};
use crate::types::{ChannelContext, ChannelVisibility, StatusUpdate, UserRole};
use aidaemon_core::SourceTrust;

/// A tool that allows the LLM to spawn a sub-agent for a focused task.
///
/// The sub-agent runs its own agentic loop with a dedicated session and a
/// system prompt that includes the given mission. Its final text response is
/// returned as the tool result.
///
/// The circular dependency (runtime → tools → SpawnAgentTool → runtime) is
/// broken by storing a weak [`ChildAgentRuntime`] port inside a `OnceLock`.
/// The reference is installed after the concrete root runtime is constructed.
pub struct SpawnAgentTool {
    agent: OnceLock<Weak<dyn ChildAgentRuntime>>,
    hub: OnceLock<Weak<dyn OutboundRouter>>,
    state: Option<Arc<dyn StateStore>>,
    max_response_chars: usize,
    timeout_secs: u64,
    executor_task_runs: Arc<Mutex<HashSet<String>>>,
}

#[cfg(test)]
const BACKGROUND_PROGRESS_INTERVAL_SECS: u64 = 1;
#[cfg(not(test))]
const BACKGROUND_PROGRESS_INTERVAL_SECS: u64 = 20;

/// Fallback description used when the parent runtime isn't wired yet
/// (early bootstrap) or when the registry is empty.
const STATIC_SPECIALIST_ARG_DESCRIPTION: &str =
    "Optional. Pick the specialist profile matching this task. \
     Omit to use the broad executor profile.";

/// Format the per-kind `(name, description)` pairs from
/// `SpecialistRegistry::llm_visible_kinds()` into the multi-line description
/// surfaced via the `specialist` parameter of `spawn_agent`. Kept as a free
/// function so tests can exercise the formatter without wiring a full Agent.
pub(crate) fn format_specialist_arg_description<S: AsRef<str>>(entries: &[(S, String)]) -> String {
    if entries.is_empty() {
        return STATIC_SPECIALIST_ARG_DESCRIPTION.to_string();
    }
    let mut s =
        String::from("Optional. Pick the specialist profile that best matches this task:\n");
    for (name, description) in entries {
        s.push_str("- ");
        s.push_str(name.as_ref());
        s.push_str(": ");
        s.push_str(description);
        if !description.ends_with('.') {
            s.push('.');
        }
        s.push('\n');
    }
    s.push_str("Omit `specialist` to use the broad executor profile.");
    s
}

impl SpawnAgentTool {
    /// Create a SpawnAgentTool with a known child-runtime reference.
    #[allow(dead_code)]
    pub fn new(
        agent: Weak<dyn ChildAgentRuntime>,
        max_response_chars: usize,
        timeout_secs: u64,
    ) -> Self {
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
    /// Call [`set_agent`] after constructing the root runtime.
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

    /// The tool a child agent uses to delegate further: fully wired, because
    /// a child's own results reach the chat through the same hub.
    pub(crate) fn for_child(
        max_response_chars: usize,
        timeout_secs: u64,
        state: Arc<dyn StateStore>,
        hub: Option<Weak<dyn OutboundRouter>>,
    ) -> Self {
        let tool = Self::new_deferred(max_response_chars, timeout_secs).with_state(state);
        if let Some(hub) = hub {
            tool.set_hub(hub);
        }
        tool
    }

    /// Set state store reference for queued background notifications when direct
    /// channel delivery is unavailable.
    pub fn with_state(mut self, state: Arc<dyn StateStore>) -> Self {
        self.state = Some(state);
        self
    }

    /// Set the child-runtime reference exactly once after root construction.
    /// Panics if called more than once.
    pub fn set_agent(&self, agent: Weak<dyn ChildAgentRuntime>) {
        self.agent
            .set(agent)
            .expect("SpawnAgentTool::set_agent called more than once");
    }

    fn get_agent(&self) -> anyhow::Result<Arc<dyn ChildAgentRuntime>> {
        let weak = self
            .agent
            .get()
            .ok_or_else(|| anyhow::anyhow!("SpawnAgentTool: agent reference not set"))?;
        weak.upgrade()
            .ok_or_else(|| anyhow::anyhow!("SpawnAgentTool: parent agent has been dropped"))
    }

    /// Set the channel hub reference for background mode notifications.
    pub fn set_hub(&self, hub: Weak<dyn OutboundRouter>) {
        self.hub
            .set(hub)
            .expect("SpawnAgentTool::set_hub called more than once");
    }

    fn get_hub(&self) -> Option<Arc<dyn OutboundRouter>> {
        self.hub.get().and_then(|w| w.upgrade())
    }

    pub(crate) fn wiring_ready(&self) -> bool {
        self.agent_wired() && self.hub.get().and_then(Weak::upgrade).is_some()
    }

    pub(crate) fn agent_wired(&self) -> bool {
        self.agent.get().and_then(Weak::upgrade).is_some()
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

        format_specialist_arg_description(&agent.specialist_descriptions())
    }

    /// The working time a child spawned by this tool gets.
    fn time_limit(&self, detached: bool) -> Option<ChildTimeLimit> {
        Some(ChildTimeLimit {
            limit: Duration::from_secs(self.timeout_secs),
            detached,
        })
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

/// A child's reply cut to `max_len`, and what was left out if anything.
fn bounded_child_response(
    response: &str,
    max_len: usize,
) -> (&str, Option<crate::traits::TruncationInfo>) {
    if response.len() <= max_len {
        return (response, None);
    }
    let shown = truncate_utf8(response, max_len);
    let info = crate::traits::TruncationInfo {
        shown_chars: shown.chars().count(),
        total_chars: response.chars().count(),
        remediation_hint: Some(
            "To see the rest, delegate again and ask the sub-agent to write the full \
             result to a file and return its path, or to answer a narrower question."
                .to_string(),
        ),
    };
    (shown, Some(info))
}

fn format_background_completion(
    mission: &str,
    response: &str,
    outcome: TaskOutcome,
    max_response_chars: usize,
) -> (&'static str, String) {
    // This goes straight to the chat, past the loop that renders truncation
    // metadata for tool results.
    let text = match bounded_child_response(response, max_response_chars) {
        (shown, Some(info)) => {
            format!(
                "{shown}\n\n{}",
                crate::utils::render_truncation_notice(&info)
            )
        }
        (shown, None) => shown.to_string(),
    };
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

/// How much of a mission a background record keeps.
const BACKGROUND_MISSION_CHARS: usize = 200;

/// Tell each chat which of its background sub-agents the last shutdown lost.
pub(crate) async fn report_interrupted_background_children(state: &Arc<dyn StateStore>) {
    let spawns = match state.take_background_spawns().await {
        Ok(spawns) => spawns,
        Err(error) => {
            warn!(%error, "Could not read interrupted background sub-agents");
            return;
        }
    };
    for spawn in spawns {
        let message = format!(
            "\u{26a0}\u{fe0f} A background task was interrupted by a restart and did not finish.\n\
             Mission: {}\n\n\
             It may have done part of the work. Check before running it again.",
            spawn.mission
        );
        enqueue_background_notification(
            Some(state),
            "global",
            &spawn.session_id,
            "failed",
            &message,
            "interrupted background sub-agent notifier",
            true,
        )
        .await;
    }
}

/// Queue a background job's notification for later delivery.
#[allow(clippy::too_many_arguments)]
async fn enqueue_background_notification(
    state: Option<&Arc<dyn StateStore>>,
    goal_id: &str,
    session_id: &str,
    notification_type: &str,
    message: &str,
    context: &str,
    untrusted: bool,
) {
    let message = crate::channels::present_notification(notification_type, message);
    let Some(state_store) = state else {
        warn!(
            session_id = %session_id,
            goal_id = %goal_id,
            notification_type = %notification_type,
            "{context}: no queue configured; update dropped"
        );
        return;
    };
    let entry =
        crate::traits::NotificationEntry::new(goal_id, session_id, notification_type, &message)
            .with_untrusted(untrusted);
    if let Err(e) = state_store.enqueue_notification(&entry).await {
        warn!(
            session_id = %session_id,
            goal_id = %goal_id,
            notification_type = %notification_type,
            error = %e,
            "{context}: enqueue failed"
        );
    }
}

/// Where a background child's result goes: straight to the parent session
/// through the hub, else the queue.
struct BackgroundDelivery<'a> {
    agent: &'a dyn ChildAgentRuntime,
    hub: Option<&'a Weak<dyn OutboundRouter>>,
    state: Option<&'a Arc<dyn StateStore>>,
    session_id: &'a str,
    goal_id: &'a str,
}

impl BackgroundDelivery<'_> {
    async fn deliver(&self, notification_type: &str, message: &str, trust: SourceTrust) {
        // Without a hub the queue delivers it (and the heartbeat records
        // it in the parent session then), so it is recorded once.
        let direct = match self.hub {
            Some(hub) => {
                self.agent
                    .deliver_background_child_result(Some(hub), self.session_id, message, trust)
                    .await
            }
            None => Ok(false),
        };
        let delivered = match direct {
            Ok(sent) => sent,
            Err(e) => {
                warn!(
                    session_id = %self.session_id,
                    goal_id = %self.goal_id,
                    notification_type = %notification_type,
                    error = %e,
                    "spawn_agent background completion notifier: parent delivery failed"
                );
                false
            }
        };

        if !delivered {
            // The queued copy carries the child's taint into whichever
            // parent turn records it.
            enqueue_background_notification(
                self.state,
                self.goal_id,
                self.session_id,
                notification_type,
                message,
                "spawn_agent background completion notifier",
                trust == SourceTrust::Untrusted,
            )
            .await;
        }
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
    /// Which model tier the child runs on.
    #[serde(default)]
    model_tier: Option<ChildModelTier>,
    /// Only these tools. The child is refused any other.
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// `read_only` keeps the child from changing anything.
    #[serde(default)]
    access: Option<ChildAccess>,
    /// Files or directories the child will change.
    #[serde(default)]
    scope: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChildAccess {
    ReadOnly,
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChildModelTier {
    Fast,
    Primary,
    Smart,
}

impl From<ChildModelTier> for crate::router::Tier {
    fn from(tier: ChildModelTier) -> Self {
        match tier {
            ChildModelTier::Fast => Self::Fast,
            ChildModelTier::Primary => Self::Primary,
            ChildModelTier::Smart => Self::Smart,
        }
    }
}

/// The authority the caller asked for its child, if it asked for one.
fn requested_child_authority(args: &SpawnArgs) -> anyhow::Result<Option<aidaemon_core::Authority>> {
    if args.tools.is_none() && args.access.is_none() {
        return Ok(None);
    }
    let tools = args.tools.as_ref().map(|names| {
        names
            .iter()
            .map(|name| name.trim().to_string())
            .filter(|name| !name.is_empty())
            .collect::<std::collections::BTreeSet<_>>()
    });
    anyhow::ensure!(
        tools.as_ref().is_none_or(|names| !names.is_empty()),
        "`tools` must name at least one tool"
    );
    Ok(Some(aidaemon_core::Authority {
        level: match args.access {
            Some(ChildAccess::ReadOnly) => aidaemon_core::AccessLevel::ReadOnly,
            _ => aidaemon_core::AccessLevel::Write,
        },
        tools,
    }))
}

/// The child's channel context: the parent call's visibility and trust,
/// from the dispatcher's facts.
fn build_child_channel_context(facts: &ToolCallContext) -> ChannelContext {
    let visibility = facts
        .channel_visibility
        .unwrap_or(ChannelVisibility::Internal);
    ChannelContext {
        visibility,
        platform: "internal".to_string(),
        channel_name: None,
        channel_id: None,
        workspace_id: None,
        sender_name: None,
        sender_id: None,
        channel_member_names: vec![],
        user_id_map: std::collections::HashMap::new(),
        workspace_grant: None,
        trusted: facts.trusted_session,
    }
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
         The sub-agent starts with a fresh context and the same tools unless its specialist narrows them (a named specialist cannot delegate further), so reference files by path and \
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
                the same tools unless its specialist narrows them (a named specialist cannot delegate \
                further), so reference files by path and do not paste prior tool output.",
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
                    },
                    "model_tier": {
                        "type": "string",
                        "enum": ["fast", "primary", "smart"],
                        "description": "Optional: fast for simple lookups, smart for hard reasoning. Omit to use the same model as this agent."
                    },
                    "tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "description": "Optional: the only tools this child may use. Any other call is refused, and it can never gain a tool you lack."
                    },
                    "access": {
                        "type": "string",
                        "enum": ["read_only", "write"],
                        "description": "Optional: read_only stops the child changing anything. It can never exceed your own access."
                    },
                    "scope": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional: the files or directories this child will change. When several children will write, give them disjoint scopes so they do not collide. Advisory: an overlap is reported, never refused."
                    }
                },
                "required": ["mission", "task"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let status_tx = crate::traits::status_sender(facts);
        let facts = facts.clone();

        let args: SpawnArgs = serde_json::from_str(arguments)?;
        let project_scope = facts.project_scope.clone();
        let mut parent_authority = crate::agent::authority::parent_authority(&facts).cloned();
        // A child can be made narrower than its parent and never wider.
        if let Some(requested) = requested_child_authority(&args)? {
            let parent = parent_authority.as_mut().ok_or_else(|| {
                anyhow::anyhow!("A child's tools and access can only be set inside a turn")
            })?;
            parent.authority = parent.authority.narrow(&requested);
        }
        let agent = self.get_agent()?;

        info!(
            depth = agent.depth(),
            max_depth = agent.max_depth(),
            mission = %args.mission,
            background = args.background,
            "spawn_agent tool invoked"
        );

        // Preserve the parent call's execution context.
        let channel_ctx = build_child_channel_context(&facts);

        // Propagate parent's user role to prevent privilege escalation.
        // Default to Guest (least privilege) if not provided.
        let user_role = match facts.user_role {
            Some(UserRole::Owner) => UserRole::Owner,
            _ => UserRole::Guest,
        };

        // Determine child role based on parent's role.
        let child_role = if agent.role() == AgentRole::TaskLead {
            Some(AgentRole::Executor)
        } else {
            None
        };

        // LLM-provided task_id takes priority; fall back to the call's task
        let task_id_ref = args.task_id.clone().or(facts.task_id.clone());
        // The durable task this child works on, if any. A dispatcher call
        // always carries its turn's own id, which names no durable task, so
        // a plain child has one only when the caller named it.
        let durable_task_id = if child_role == Some(AgentRole::Executor) {
            task_id_ref.clone()
        } else {
            args.task_id.clone()
        };
        let goal_id_ref = facts.goal_id.clone();

        // Plain children count against the request's tree. A task lead's
        // executors are bounded per task instead, and keep budgets of their own.
        let family_permit = if child_role.is_some() {
            if let Some(parent) = parent_authority.as_mut() {
                parent.spawn_family = None;
            }
            None
        } else {
            match parent_authority
                .as_ref()
                .and_then(|parent| parent.spawn_family.as_ref())
            {
                Some(family) => Some(family.try_admit().map_err(|why| anyhow::anyhow!(why))?),
                None => None,
            }
        };

        // A plain child in a tree joins its roster, saying what it will change.
        let mut scope_overlap_note = String::new();
        let tree = family_permit
            .is_some()
            .then(|| {
                parent_authority
                    .as_ref()
                    .and_then(|parent| parent.spawn_family.clone())
            })
            .flatten();
        if let Some(family) = tree.as_ref() {
            let scope = agent.resolve_child_scope(args.scope.clone()).await;
            let overlaps = family.coordination().scope_overlaps(&scope);
            if !overlaps.is_empty() {
                let listed = overlaps
                    .iter()
                    .map(|(sibling, path)| format!("{path} (also {sibling})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                scope_overlap_note =
                    format!("\n\nThis child's scope overlaps running sub-agents at: {listed}.");
            }
            if let Some(parent) = parent_authority.as_mut() {
                parent.child_declaration = Some(crate::agent::authority::ChildDeclaration {
                    parent_session: facts.session_id.clone(),
                    scope,
                });
            }
        }

        // TaskLead -> Executor spawns must target a concrete, pre-claimed task and
        // are deduplicated per task_id so the same work is not launched twice.
        let executor_task_id = if child_role == Some(AgentRole::Executor) {
            let Some(task_id) = task_id_ref.clone() else {
                return Err(anyhow::anyhow!(
                    "TaskLead must pass task_id when spawning an executor. Claim a task first with manage_goal_tasks(action='claim_task')."
                ));
            };
            if let Err(e) = agent
                .validate_executor_task_for_spawn(&task_id, goal_id_ref.as_deref())
                .await
            {
                return Err(anyhow::anyhow!(
                    "Blocked executor spawn for task {}: {}",
                    task_id,
                    e
                ));
            }
            if !self.try_begin_executor_task(&task_id).await {
                return Err(anyhow::anyhow!(
                    "Task {} already has an executor running. Wait for it to finish before spawning another.",
                    task_id
                ));
            }
            Some(task_id)
        } else {
            None
        };

        let approval_session_id = facts.session().map(str::to_string);

        // Background needs a way to deliver the result and a session to
        // deliver it to; without either the child runs in the foreground.
        let hub = self.get_hub();
        let state = self.state.clone();
        let hub_for_parent_delivery = hub.as_ref().map(Arc::downgrade);
        let background_session = if !args.background {
            None
        } else if hub.is_none() && state.is_none() {
            info!(
                "Background mode requested but no hub/state notification path is available, falling back to sync"
            );
            None
        } else if let Some(id) = approval_session_id.as_deref().filter(|id| !id.is_empty()) {
            Some(agent.background_delivery_session(id))
        } else {
            info!("Background mode requested but no session_id, falling back to sync");
            None
        };
        let Some(session_id) = background_session else {
            let result = crate::agent::authority::within_permit(
                family_permit.as_ref(),
                self.run_sync(
                    agent,
                    &args.mission,
                    &args.task,
                    status_tx,
                    channel_ctx,
                    user_role,
                    child_role,
                    goal_id_ref.as_deref(),
                    task_id_ref.as_deref(),
                    durable_task_id.as_deref(),
                    project_scope.as_deref(),
                    args.specialist.as_deref(),
                    approval_session_id.as_deref(),
                    parent_authority.clone(),
                    args.model_tier.map(Into::into),
                    tree.as_ref(),
                ),
            )
            .await;
            if let Some(ref task_id) = executor_task_id {
                self.finish_executor_task(task_id).await;
            }
            return result.map(|mut outcome| {
                outcome.output.push_str(&scope_overlap_note);
                outcome
            });
        };

        let task = args.task.clone();
        let mission = args.mission.clone();
        let timeout_secs = self.timeout_secs;
        let time_limit = self.time_limit(true);
        let max_response_chars = self.max_response_chars;
        let executor_task_runs = Arc::clone(&self.executor_task_runs);
        let executor_task_id_for_bg = executor_task_id;
        let notify_goal_id = goal_id_ref.clone().unwrap_or_else(|| "global".to_string());
        let notify_status_tx = status_tx.clone();
        let tree_for_bg = tree.clone();

        let background_id = format!("background_spawn:{}", uuid::Uuid::new_v4());
        if let Some(state) = state.as_ref() {
            let record = crate::traits::BackgroundSpawn {
                id: background_id.clone(),
                session_id: session_id.clone(),
                mission: mission.chars().take(BACKGROUND_MISSION_CHARS).collect(),
                started_at: chrono::Utc::now().to_rfc3339(),
            };
            if let Err(error) = state.record_background_spawn(&record).await {
                warn!(%error, "Could not record a background sub-agent");
            }
        }
        let cancel_token = agent.background_cancel_token(&background_id).await;

        tokio::spawn(async move {
            // This task owns `family_permit`, so the slot stays taken until
            // the child has finished.
            let delivery = BackgroundDelivery {
                agent: agent.as_ref(),
                hub: hub_for_parent_delivery.as_ref(),
                state: state.as_ref(),
                session_id: &session_id,
                goal_id: &notify_goal_id,
            };
            // A panic in the child or in its delivery must not skip the
            // cleanup below, and the parent session still hears it failed.
            let finished = std::panic::AssertUnwindSafe(async {
                let started_at = std::time::Instant::now();
                let mut progress_interval =
                    tokio::time::interval(Duration::from_secs(BACKGROUND_PROGRESS_INTERVAL_SECS));
                progress_interval.tick().await; // consume immediate tick
                let mut cancelled_by_owner = false;
                // Scoped so that leaving it drops the child's future, which stops
                // the child.
                let result: anyhow::Result<ChildAgentRun> = {
                    let mut result_fut = std::pin::pin!(crate::agent::authority::within_permit(
                        family_permit.as_ref(),
                        agent.run_child(ChildAgentRequest {
                            mission: mission.clone(),
                            task: task.clone(),
                            status_tx: status_tx.clone(),
                            channel_ctx,
                            user_role,
                            child_role,
                            goal_id: goal_id_ref.clone(),
                            task_id: task_id_ref.clone(),
                            project_scope: project_scope.clone(),
                            specialist: args.specialist.clone(),
                            approval_session_id: Some(session_id.clone()),
                            parent_authority: parent_authority.clone(),
                            time_limit,
                            model_tier: args.model_tier.map(Into::into),
                        })
                    ),);
                    let owner_cancelled = async {
                        match cancel_token.as_ref() {
                            Some(token) => token.cancelled().await,
                            None => std::future::pending().await,
                        }
                    };
                    tokio::pin!(owner_cancelled);
                    loop {
                        tokio::select! {
                            res = &mut result_fut => break res,
                            _ = &mut owner_cancelled => {
                                cancelled_by_owner = true;
                                break Err(anyhow::anyhow!("cancelled by the owner"));
                            }
                            _ = progress_interval.tick() => {
                                let elapsed_secs = started_at.elapsed().as_secs();
                                if let Some(ref tx) = notify_status_tx {
                                    let _ = tx.try_send(StatusUpdate::ToolProgress {
                                        name: "spawn_agent".to_string(),
                                        chunk: format!(
                                            "Background sub-agent still running ({}s): {}",
                                            elapsed_secs, mission
                                        ),
                                    });
                                }
                            }
                        }
                    }
                };

                // Only a finished child's own record decides; any other result
                // may carry child-written text and counts as untrusted.
                let trust = match &result {
                    Ok(run) => run.trust,
                    Err(error)
                        if durable_task_id.is_none()
                            && error
                                .downcast_ref::<ChildTimedOut>()
                                .is_some_and(|t| !t.report.may_have_changed_something()) =>
                    {
                        SourceTrust::Trusted
                    }
                    Err(_) => SourceTrust::Untrusted,
                };
                let (notification_type, message) = match result {
                    _ if cancelled_by_owner => (
                        "failed",
                        format!(
                            "\u{23f9}\u{fe0f} Background task cancelled\nMission: {}\n\n\
                         It may have done part of the work.",
                            mission
                        ),
                    ),
                    Ok(run) => {
                        let (kind, mut message) = format_background_completion(
                            &mission,
                            &run.response,
                            run.outcome,
                            max_response_chars,
                        );
                        message.push_str(&coordination_trailer(tree_for_bg.as_ref(), &run.session));
                        if let Some(note) = tree_for_bg
                            .as_ref()
                            .and_then(|family| family.coordination().shared_files_note())
                        {
                            message.push_str(&note);
                        }
                        (kind, message)
                    }
                    Err(error) => match error.downcast_ref::<ChildTimedOut>() {
                        None => (
                            "failed",
                            format!(
                                "\u{274c} Background task failed\nMission: {}\nError: {}",
                                mission, error
                            ),
                        ),
                        Some(timed_out) => {
                            // The child may have persisted a terminal outcome
                            // before it was stopped — deliver that instead of a
                            // generic timeout failure.
                            let salvaged = match durable_task_id.as_deref() {
                                Some(task_id) => {
                                    agent
                                        .salvage_executor_task_outcome(task_id, timeout_secs)
                                        .await
                                }
                                None => None,
                            };
                            // Nothing to salvage: release the executor's claimed
                            // task, as a waited-on executor does.
                            if salvaged.is_none() && child_role == Some(AgentRole::Executor) {
                                if let Some(task_id) = durable_task_id.as_deref() {
                                    agent
                                        .mark_executor_task_timeout(task_id, timeout_secs)
                                        .await;
                                }
                            }
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
                                        "\u{23f1} Background task timed out\nMission: {}\n\n{}",
                                        mission,
                                        timed_out.render()
                                    ),
                                ),
                            }
                        }
                    },
                };
                delivery.deliver(notification_type, &message, trust).await;
            })
            .catch_unwind()
            .await;
            if finished.is_err() {
                warn!(
                    session_id = %session_id,
                    %background_id,
                    "Background sub-agent panicked; reporting it as failed"
                );
                let message = format!(
                    "\u{274c} Background task failed\nMission: {}\n\nIt stopped unexpectedly.",
                    mission
                );
                // The note is the daemon's own text, so it is trusted.
                let delivered = std::panic::AssertUnwindSafe(delivery.deliver(
                    "failed",
                    &message,
                    SourceTrust::Trusted,
                ))
                .catch_unwind()
                .await;
                if delivered.is_err() {
                    enqueue_background_notification(
                        state.as_ref(),
                        &notify_goal_id,
                        &session_id,
                        "failed",
                        &message,
                        "spawn_agent background completion notifier",
                        false,
                    )
                    .await;
                }
            }

            if let Some(state) = state.as_ref() {
                if let Err(error) = state.finish_background_spawn(&background_id).await {
                    warn!(%error, "Could not clear a background sub-agent's record");
                }
            }
            agent.release_background_cancel_token(&background_id).await;

            if let Some(task_id) = executor_task_id_for_bg {
                executor_task_runs.lock().await.remove(&task_id);
            }
        });

        let mut outcome = background_spawn_outcome(&args.mission);
        outcome.output.push_str(&scope_overlap_note);
        Ok(outcome)
    }
}

/// The coordination log's entries about one child, for the end of its result.
fn coordination_trailer(
    tree: Option<&crate::agent::authority::SpawnFamily>,
    session: &str,
) -> String {
    tree.and_then(|family| family.coordination().result_trailer(session))
        .unwrap_or_default()
}

/// A sub-agent now running in the background: the turn hands off to it, and
/// its result reaches the parent session when it completes.
fn background_spawn_outcome(mission: &str) -> ToolCallOutcome {
    reported(
        format!(
            "Sub-agent spawned in background for mission: \"{mission}\". \
             The result will be delivered through the parent session when it completes."
        ),
        SourceTrust::Trusted,
    )
    .handed_to_background()
}

impl crate::traits::ToolSemantics for SpawnAgentTool {
    fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
        crate::traits::ToolDispatchFacts {
            task_lead_delegation: true,
            spawns_child_agents: true,
            goal_context: crate::traits::GoalContext::Goal,
            ..Default::default()
        }
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

    /// A child's result carries the child's taint; anything the call does
    /// not report (errors) counts as untrusted.
    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &crate::traits::TrustRoots<'_>,
    ) -> SourceTrust {
        SourceTrust::Untrusted
    }
}

/// A spawn result with the trust of its content stamped for the dispatcher.
fn reported(output: String, trust: SourceTrust) -> ToolCallOutcome {
    let mut outcome = ToolCallOutcome::from_output(output);
    outcome.metadata.output_trust = Some(trust);
    outcome
}

impl SpawnAgentTool {
    /// Run the sub-agent synchronously (blocking until completion or timeout).
    #[allow(clippy::too_many_arguments)]
    async fn run_sync(
        &self,
        agent: Arc<dyn ChildAgentRuntime>,
        mission: &str,
        task: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
        child_role: Option<AgentRole>,
        goal_id: Option<&str>,
        task_id: Option<&str>,
        durable_task_id: Option<&str>,
        project_scope: Option<&str>,
        arg_specialist: Option<&str>,
        approval_session_id: Option<&str>,
        parent_authority: Option<crate::agent::authority::InheritedAuthority>,
        model_tier: Option<crate::router::Tier>,
        tree: Option<&crate::agent::authority::SpawnFamily>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let result = agent
            .run_child(ChildAgentRequest {
                mission: mission.to_string(),
                task: task.to_string(),
                status_tx,
                channel_ctx,
                user_role,
                child_role,
                goal_id: goal_id.map(str::to_string),
                task_id: task_id.map(str::to_string),
                project_scope: project_scope.map(str::to_string),
                specialist: arg_specialist.map(str::to_string),
                approval_session_id: approval_session_id.map(str::to_string),
                parent_authority,
                time_limit: self.time_limit(false),
                model_tier,
            })
            .await;

        match result {
            Ok(run) => {
                let response = match run.outcome {
                    TaskOutcome::Succeeded => run.response,
                    TaskOutcome::Partial => {
                        format!("\u{26a0}\u{fe0f} Sub-agent incomplete\n\n{}", run.response)
                    }
                    TaskOutcome::Failed => {
                        return Err(anyhow::anyhow!("Sub-agent failed: {}", run.response));
                    }
                };
                let (shown, truncation) =
                    bounded_child_response(&response, self.max_response_chars);
                let mut outcome = reported(
                    format!("{shown}{}", coordination_trailer(tree, &run.session)),
                    run.trust,
                );
                outcome.metadata.truncation = truncation;
                Ok(outcome)
            }
            Err(error) => match error.downcast_ref::<ChildTimedOut>() {
                None => Err(anyhow::anyhow!("Specialist failed: {}", error)),
                Some(timed_out) => {
                    if let Some(task_id) = durable_task_id {
                        // The child may have persisted a terminal outcome (e.g.
                        // report_blocker) before it was stopped — use that
                        // instead of discarding the work.
                        if let Some(salvaged) = agent
                            .salvage_executor_task_outcome(task_id, self.timeout_secs)
                            .await
                        {
                            let prefix = match salvaged.status.as_str() {
                                "completed" => "\u{2705} Sub-agent complete",
                                "blocked" => "\u{26a0}\u{fe0f} Sub-agent blocked",
                                _ => "\u{274c} Sub-agent failed",
                            };
                            // The child wrote these details; its taint is
                            // unknown after a timeout.
                            return Ok(reported(
                                format!("{prefix}\n\n{}", salvaged.details),
                                SourceTrust::Untrusted,
                            ));
                        }
                        if child_role == Some(AgentRole::Executor) {
                            agent
                                .mark_executor_task_timeout(task_id, self.timeout_secs)
                                .await;
                        }
                    }
                    if timed_out.report.may_have_changed_something() {
                        // Not a failure the loop may retry on its own: the
                        // parent has to look at what already happened. The
                        // action details come from the child's run.
                        return Ok(reported(
                            format!(
                                "\u{26a0}\u{fe0f} Sub-agent incomplete\n\n{}",
                                timed_out.render()
                            ),
                            SourceTrust::Untrusted,
                        ));
                    }
                    Err(anyhow::anyhow!("{timed_out}"))
                }
            },
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

    /// A background spawn ends the turn from the tool's typed outcome, not
    /// from its wording: the finish and the handoff fact are set whatever
    /// the message says.
    #[test]
    fn background_spawn_hands_the_turn_off() {
        let outcome = background_spawn_outcome("Summarize the synthetic report");
        assert!(outcome.metadata.background_handoff);
        assert!(
            !outcome.metadata.background_started,
            "a sub-agent is not a tracked process launch"
        );
        let finish = outcome.finish.expect("the turn hands off");
        assert_eq!(finish.outcome, aidaemon_core::FinishOutcome::Partial);
        assert_eq!(
            finish.reply, None,
            "the dispatcher writes the handoff reply"
        );
    }

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
    }

    #[test]
    fn spawn_args_background_true() {
        let json = r#"{"mission": "test", "task": "do stuff", "background": true}"#;
        let args: SpawnArgs = serde_json::from_str(json).unwrap();
        assert!(args.background);
    }

    fn spawn_args(extra: serde_json::Value) -> SpawnArgs {
        let mut args = serde_json::json!({"mission": "m", "task": "t"});
        args.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(args).unwrap()
    }

    #[test]
    fn a_child_is_narrowed_only_when_asked() {
        assert!(
            requested_child_authority(&spawn_args(serde_json::json!({})))
                .unwrap()
                .is_none()
        );

        let requested =
            requested_child_authority(&spawn_args(serde_json::json!({"access": "read_only"})))
                .unwrap()
                .unwrap();
        assert_eq!(requested.level, aidaemon_core::AccessLevel::ReadOnly);
        assert_eq!(requested.tools, None);

        let requested = requested_child_authority(&spawn_args(
            serde_json::json!({"tools": ["read_file", " search_files "]}),
        ))
        .unwrap()
        .unwrap();
        assert_eq!(requested.level, aidaemon_core::AccessLevel::Write);
        assert_eq!(
            requested.tools.unwrap().into_iter().collect::<Vec<_>>(),
            ["read_file", "search_files"]
        );

        assert!(
            requested_child_authority(&spawn_args(serde_json::json!({"tools": [" "]}))).is_err()
        );
    }

    #[test]
    fn asking_for_more_than_the_parent_has_gives_nothing_extra() {
        let parent = aidaemon_core::Authority::read_only();
        let requested =
            requested_child_authority(&spawn_args(serde_json::json!({"access": "write"})))
                .unwrap()
                .unwrap();
        assert_eq!(
            parent.narrow(&requested).level,
            aidaemon_core::AccessLevel::ReadOnly
        );
    }

    /// The parent turn's authority reaches spawn_agent only through the
    /// dispatcher's typed facts; an argument of the same name is ignored.
    #[test]
    fn parent_authority_comes_from_the_call_facts() {
        let parent = crate::agent::authority::InheritedAuthority {
            authority: aidaemon_core::Authority::read_only(),
            owner_level: Some(crate::config::OwnerLevel::Full),
            task_access: None,
            mode_session: Some("tg:123".to_string()),
            untrusted_context: true,
            origin_request: Some("Summarize the synthetic notes.".to_string()),
            origin_trusted: true,
            trusted_roots: Vec::new(),
            declined: vec![aidaemon_core::EffectKey::egress(
                "https",
                "h.example.invalid",
                None,
            )],
            shared_declines: None,
            spawn_family: None,
            child_declaration: None,
            parent_run: None,
        };
        let mut facts = ToolCallContext::default();
        assert!(crate::agent::authority::parent_authority(&facts).is_none());
        facts.extensions.insert(parent.clone());
        assert_eq!(
            crate::agent::authority::parent_authority(&facts),
            Some(&parent)
        );
        // The model cannot supply one: SpawnArgs has no such argument.
        let json = serde_json::json!({
            "mission": "test",
            "task": "do stuff",
            "_parent_authority": parent,
        });
        assert!(serde_json::from_value::<SpawnArgs>(json).is_ok());
    }

    #[test]
    fn child_channel_context_takes_the_call_visibility() {
        let facts = ToolCallContext {
            channel_visibility: Some(ChannelVisibility::Public),
            ..Default::default()
        };
        assert_eq!(
            build_child_channel_context(&facts).visibility,
            ChannelVisibility::Public
        );
        assert_eq!(
            build_child_channel_context(&ToolCallContext::default()).visibility,
            ChannelVisibility::Internal
        );
    }

    #[test]
    fn child_channel_context_takes_the_call_trust() {
        let facts = ToolCallContext {
            trusted_session: true,
            channel_visibility: Some(ChannelVisibility::Internal),
            ..Default::default()
        };
        let channel_ctx = build_child_channel_context(&facts);
        assert_eq!(channel_ctx.visibility, ChannelVisibility::Internal);
        assert!(channel_ctx.trusted);
        assert!(!build_child_channel_context(&ToolCallContext::default()).trusted);
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
    async fn background_notification_reaches_the_queue() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let state_dyn: Arc<dyn StateStore> = state.clone();

        enqueue_background_notification(
            Some(&state_dyn),
            "goal_spawn_test",
            "sess_spawn_test",
            "completed",
            "Background task complete.\nMission: test",
            "spawn_test",
            false,
        )
        .await;

        let pending = state.get_pending_notifications(10).await.unwrap();
        assert!(pending.iter().any(|entry| {
            entry.goal_id == "goal_spawn_test"
                && entry.session_id == "sess_spawn_test"
                && entry.notification_type == "completed"
                && entry.message.contains("Mission: test")
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

#[cfg(test)]
mod child_runtime_tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::runtime_ports::{ChildWorkReport, SalvagedTaskOutcome};
    use crate::state::SqliteStateStore;
    use crate::traits::NotificationStore;
    use crate::traits::ToolCalls as _;
    use std::sync::Mutex as StdMutex;

    /// A child runtime that answers from a script, recording what it was asked.
    struct ScriptedChild {
        response: String,
        delay: Duration,
        tasks: StdMutex<Vec<String>>,
        failure: StdMutex<Option<anyhow::Error>>,
        cancel: tokio_util::sync::CancellationToken,
        role: StdMutex<AgentRole>,
        timed_out_tasks: StdMutex<Vec<String>>,
        panics: std::sync::atomic::AtomicBool,
        /// Fails before the child begins running.
        fails_in_setup: std::sync::atomic::AtomicBool,
        salvage_lookups: StdMutex<Vec<String>>,
    }

    impl ScriptedChild {
        fn new(response: &str, delay: Duration) -> Arc<Self> {
            Arc::new(Self {
                response: response.to_string(),
                delay,
                tasks: StdMutex::new(Vec::new()),
                failure: StdMutex::new(None),
                cancel: tokio_util::sync::CancellationToken::new(),
                role: StdMutex::new(AgentRole::Orchestrator),
                timed_out_tasks: StdMutex::new(Vec::new()),
                panics: std::sync::atomic::AtomicBool::new(false),
                fails_in_setup: std::sync::atomic::AtomicBool::new(false),
                salvage_lookups: StdMutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl ChildAgentRuntime for ScriptedChild {
        fn depth(&self) -> usize {
            0
        }
        fn max_depth(&self) -> usize {
            3
        }
        fn role(&self) -> AgentRole {
            *self.role.lock().unwrap()
        }
        fn specialist_descriptions(&self) -> Vec<(String, String)> {
            Vec::new()
        }
        async fn validate_executor_task_for_spawn(
            &self,
            _task_id: &str,
            _expected_goal_id: Option<&str>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn run_child(&self, request: ChildAgentRequest) -> anyhow::Result<ChildAgentRun> {
            if self
                .fails_in_setup
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                anyhow::bail!("synthetic setup failure");
            }
            crate::agent::authority::mark_child_began();
            self.tasks.lock().unwrap().push(request.task);
            tokio::time::sleep(self.delay).await;
            if self.panics.load(std::sync::atomic::Ordering::SeqCst) {
                panic!("synthetic child panic");
            }
            if let Some(failure) = self.failure.lock().unwrap().take() {
                return Err(failure);
            }
            Ok(ChildAgentRun {
                session: "sub_agent_synthetic".to_string(),
                response: self.response.clone(),
                outcome: TaskOutcome::Succeeded,
                trust: SourceTrust::Trusted,
            })
        }
        async fn background_cancel_token(
            &self,
            _id: &str,
        ) -> Option<tokio_util::sync::CancellationToken> {
            Some(self.cancel.clone())
        }
        async fn salvage_executor_task_outcome(
            &self,
            task_id: &str,
            _timeout_secs: u64,
        ) -> Option<SalvagedTaskOutcome> {
            self.salvage_lookups
                .lock()
                .unwrap()
                .push(task_id.to_string());
            None
        }
        async fn mark_executor_task_timeout(&self, task_id: &str, _timeout_secs: u64) {
            self.timed_out_tasks
                .lock()
                .unwrap()
                .push(task_id.to_string());
        }
        async fn deliver_background_child_result(
            &self,
            _router: Option<&Weak<dyn OutboundRouter>>,
            _parent_session_id: &str,
            _text: &str,
            _trust: SourceTrust,
        ) -> anyhow::Result<bool> {
            Ok(false)
        }
    }

    fn tool_for(child: &Arc<ScriptedChild>) -> SpawnAgentTool {
        let tool = SpawnAgentTool::new_deferred(100, 30);
        let runtime: Arc<dyn ChildAgentRuntime> = child.clone();
        tool.set_agent(Arc::downgrade(&runtime));
        tool
    }

    fn facts() -> ToolCallContext {
        ToolCallContext {
            session_id: "telegram:synthetic-user-1".to_string(),
            ..Default::default()
        }
    }

    async fn notification_store() -> Arc<SqliteStateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        std::mem::forget(db_file);
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        )
    }

    /// A child's own delegations reach the chat without waiting for the queue.
    #[tokio::test]
    async fn a_childs_spawn_tool_has_the_hub() {
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store;
        let router: Arc<dyn OutboundRouter> = Arc::new(CountingHub::default());
        let tool = SpawnAgentTool::for_child(100, 30, state, Some(Arc::downgrade(&router)));
        assert!(tool.get_hub().is_some());
    }

    /// A timed-out child that already acted is not a retryable failure.
    #[tokio::test]
    async fn timeout_after_side_effects_warns_instead_of_failing() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.failure.lock().unwrap() = Some(
            ChildTimedOut {
                limit_secs: 30,
                report: ChildWorkReport {
                    changed: vec!["send_file: report.pdf".to_string()],
                    ..Default::default()
                },
            }
            .into(),
        );
        let tool = tool_for(&child);
        let args = json!({"mission": "Send", "task": "Send it"}).to_string();

        let outcome = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .expect("reported, not an error");

        assert!(outcome.output.contains("send_file: report.pdf"));
        assert!(outcome.output.contains("Do NOT run it again"));
        assert_eq!(outcome.metadata.output_trust, Some(SourceTrust::Untrusted));
    }

    #[tokio::test]
    async fn timeout_with_no_side_effects_stays_a_retryable_error() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.failure.lock().unwrap() = Some(
            ChildTimedOut {
                limit_secs: 30,
                report: ChildWorkReport::default(),
            }
            .into(),
        );
        let tool = tool_for(&child);
        let args = json!({"mission": "Read", "task": "Read it"}).to_string();

        let error = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .expect_err("nothing happened, so a retry is safe");

        assert!(error.to_string().contains("timed out after 30 seconds"));
    }

    /// A background executor that runs out of time releases its task the way
    /// a waited-on one does.
    #[tokio::test]
    async fn a_timed_out_background_executor_releases_its_task() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.role.lock().unwrap() = AgentRole::TaskLead;
        *child.failure.lock().unwrap() = Some(
            ChildTimedOut {
                limit_secs: 30,
                report: ChildWorkReport::default(),
            }
            .into(),
        );
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({
            "mission": "Execute",
            "task": "t",
            "task_id": "task-synthetic-1",
            "background": true,
        })
        .to_string();

        tool.call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        wait_for_notification(&store).await;

        assert_eq!(
            *child.timed_out_tasks.lock().unwrap(),
            vec!["task-synthetic-1".to_string()]
        );
    }

    /// A background child that panics still clears its record, frees its
    /// task, and tells the parent session it failed.
    #[tokio::test]
    async fn a_panicking_background_child_is_cleaned_up_and_reported() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        child
            .panics
            .store(true, std::sync::atomic::Ordering::SeqCst);
        *child.role.lock().unwrap() = AgentRole::TaskLead;
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({
            "mission": "Execute",
            "task": "t",
            "task_id": "task-synthetic-2",
            "background": true,
        })
        .to_string();

        tool.call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        let entry = wait_for_notification(&store).await;
        assert_eq!(entry.notification_type, "failed");
        assert_eq!(entry.session_id, "telegram:synthetic-user-1");
        assert!(!entry.untrusted, "the failure note carries no child text");

        tokio::time::timeout(Duration::from_secs(5), async {
            while !store
                .background_spawns_for_session("telegram:synthetic-user-1")
                .await
                .unwrap()
                .is_empty()
                || tool
                    .executor_task_runs
                    .lock()
                    .await
                    .contains("task-synthetic-2")
            {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the record and the task claim are released");
    }

    fn timed_out_read_only() -> anyhow::Error {
        ChildTimedOut {
            limit_secs: 30,
            report: ChildWorkReport::default(),
        }
        .into()
    }

    fn facts_on_a_turn() -> ToolCallContext {
        ToolCallContext {
            task_id: Some("task-turn-synthetic".to_string()),
            ..facts()
        }
    }

    /// The turn's own id says nothing about a plain child: a read-only
    /// timeout stays trusted, and no durable task is looked up for it.
    #[tokio::test]
    async fn a_plain_background_childs_timeout_ignores_the_turns_task_id() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.failure.lock().unwrap() = Some(timed_out_read_only());
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({"mission": "Read", "task": "t", "background": true}).to_string();

        tool.call_with_execution_context(&args, None, facts_on_a_turn())
            .await
            .unwrap();
        let entry = wait_for_notification(&store).await;

        assert!(!entry.untrusted);
        assert!(child.salvage_lookups.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_plain_foreground_childs_timeout_looks_up_no_task() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.failure.lock().unwrap() = Some(timed_out_read_only());
        let tool = tool_for(&child);
        let args = json!({"mission": "Read", "task": "t"}).to_string();

        tool.call_with_execution_context(&args, None, facts_on_a_turn())
            .await
            .expect_err("a read-only timeout stays retryable");

        assert!(child.salvage_lookups.lock().unwrap().is_empty());
    }

    /// An executor's task is real, so its timeout is still looked up.
    #[tokio::test]
    async fn an_executors_timeout_still_looks_up_its_task() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.role.lock().unwrap() = AgentRole::TaskLead;
        *child.failure.lock().unwrap() = Some(timed_out_read_only());
        let tool = tool_for(&child);
        let args =
            json!({"mission": "Execute", "task": "t", "task_id": "task-synthetic-3"}).to_string();

        tool.call_with_execution_context(&args, None, facts_on_a_turn())
            .await
            .expect_err("a read-only timeout stays retryable");

        assert_eq!(
            *child.salvage_lookups.lock().unwrap(),
            vec!["task-synthetic-3".to_string()]
        );
    }

    fn facts_in_tree(family: &crate::agent::authority::SpawnFamily) -> ToolCallContext {
        let mut call_facts = facts();
        call_facts
            .extensions
            .insert(crate::agent::authority::InheritedAuthority {
                authority: aidaemon_core::Authority::full(),
                owner_level: None,
                task_access: None,
                mode_session: None,
                untrusted_context: false,
                origin_request: None,
                origin_trusted: false,
                trusted_roots: Vec::new(),
                declined: Vec::new(),
                shared_declines: None,
                spawn_family: Some(family.clone()),
                child_declaration: None,
                parent_run: None,
            });
        call_facts
    }

    /// An overlap with a running sibling is stated, never refused, and the
    /// child's result ends with what the log says about it.
    #[tokio::test]
    async fn a_scope_overlap_and_the_childs_log_reach_the_spawn_result() {
        use crate::agent::coordination::{LogEntry, Member};
        let child = ScriptedChild::new("Finished.", Duration::ZERO);
        let tool = tool_for(&child);
        let family = crate::agent::authority::SpawnFamily::new(Default::default());
        family.coordination().add_member(Member {
            session: "sub_agent_sibling".to_string(),
            parent: "root".to_string(),
            mission: "Other work".to_string(),
            specialist: None,
            scope: vec!["/w/src".to_string()],
            running: true,
        });
        family.coordination().log(LogEntry::WriteOutsideScope {
            member: "sub_agent_synthetic".to_string(),
            path: "/w/elsewhere".to_string(),
        });
        let args = json!({"mission": "m", "task": "t", "scope": ["/w/src/lib.rs"]}).to_string();

        let outcome = tool
            .call_with_execution_context(&args, None, facts_in_tree(&family))
            .await
            .unwrap();

        assert!(
            outcome.output.starts_with("Finished."),
            "{}",
            outcome.output
        );
        assert!(
            outcome.output.contains("/w/elsewhere"),
            "{}",
            outcome.output
        );
        assert!(
            outcome.output.contains("sub_agent_sibling"),
            "{}",
            outcome.output
        );
    }

    /// Children past the request's running limit are refused, not queued.
    #[tokio::test]
    async fn a_request_cannot_run_more_children_than_its_limit() {
        let child = ScriptedChild::new("done", Duration::from_millis(300));
        let tool = tool_for(&child);
        let family =
            crate::agent::authority::SpawnFamily::new(crate::agent::authority::SpawnFamilyLimits {
                max_concurrent: 1,
                max_total: 10,
                ..Default::default()
            });
        let mut call_facts = facts();
        call_facts
            .extensions
            .insert(crate::agent::authority::InheritedAuthority {
                authority: aidaemon_core::Authority::read_only(),
                owner_level: None,
                task_access: None,
                mode_session: None,
                untrusted_context: false,
                origin_request: None,
                origin_trusted: true,
                trusted_roots: Vec::new(),
                declined: Vec::new(),
                shared_declines: None,
                spawn_family: Some(family),
                child_declaration: None,
                parent_run: None,
            });
        let args = json!({"mission": "m", "task": "t"}).to_string();

        let (first, second) = tokio::join!(
            tool.call_with_execution_context(&args, None, call_facts.clone()),
            tool.call_with_execution_context(&args, None, call_facts.clone()),
        );

        let refused = usize::from(first.is_err()) + usize::from(second.is_err());
        assert_eq!(refused, 1);
        assert_eq!(child.tasks.lock().unwrap().len(), 1);
    }

    fn family_facts(max_total: usize) -> ToolCallContext {
        let family =
            crate::agent::authority::SpawnFamily::new(crate::agent::authority::SpawnFamilyLimits {
                max_concurrent: 5,
                max_total,
                ..Default::default()
            });
        let mut call_facts = facts();
        call_facts
            .extensions
            .insert(crate::agent::authority::InheritedAuthority {
                authority: aidaemon_core::Authority::read_only(),
                owner_level: None,
                task_access: None,
                mode_session: None,
                untrusted_context: false,
                origin_request: None,
                origin_trusted: true,
                trusted_roots: Vec::new(),
                declined: Vec::new(),
                shared_declines: None,
                spawn_family: Some(family),
                child_declaration: None,
                parent_run: None,
            });
        call_facts
    }

    /// A spawn that fails before its child begins running does not use up
    /// the request's total; one whose child ran does.
    #[tokio::test]
    async fn a_failed_setup_does_not_use_up_the_requests_total() {
        let child = ScriptedChild::new("done", Duration::ZERO);
        let tool = tool_for(&child);
        let call_facts = family_facts(1);
        let args = json!({"mission": "m", "task": "t"}).to_string();

        child
            .fails_in_setup
            .store(true, std::sync::atomic::Ordering::SeqCst);
        tool.call_with_execution_context(&args, None, call_facts.clone())
            .await
            .expect_err("setup failed");
        child
            .fails_in_setup
            .store(false, std::sync::atomic::Ordering::SeqCst);
        tool.call_with_execution_context(&args, None, call_facts.clone())
            .await
            .expect("the failed setup left its place");
        tool.call_with_execution_context(&args, None, call_facts.clone())
            .await
            .expect_err("the child that ran used the only place");
        assert_eq!(child.tasks.lock().unwrap().len(), 1);
    }

    /// The same holds for a background spawn.
    #[tokio::test]
    async fn a_failed_background_setup_does_not_use_up_the_requests_total() {
        let child = ScriptedChild::new("done", Duration::ZERO);
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let call_facts = family_facts(1);
        let args = json!({"mission": "m", "task": "t", "background": true}).to_string();

        child
            .fails_in_setup
            .store(true, std::sync::atomic::Ordering::SeqCst);
        tool.call_with_execution_context(&args, None, call_facts.clone())
            .await
            .unwrap();
        wait_for_notification(&store).await;
        child
            .fails_in_setup
            .store(false, std::sync::atomic::Ordering::SeqCst);
        tokio::time::timeout(Duration::from_secs(5), async {
            while tool
                .call_with_execution_context(&args, None, call_facts.clone())
                .await
                .is_err()
            {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the failed setup gives its place back once it ends");
        tool.call_with_execution_context(&args, None, call_facts.clone())
            .await
            .expect_err("the child that ran used the only place");
    }

    async fn wait_for_notification(store: &SqliteStateStore) -> crate::traits::NotificationEntry {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let pending = store.get_pending_notifications(10).await.unwrap();
                if let Some(entry) = pending.into_iter().next() {
                    return entry;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("a notification arrives")
    }

    #[tokio::test]
    async fn a_background_child_is_on_record_only_while_it_runs() {
        let child = ScriptedChild::new("done", Duration::from_millis(400));
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({"mission": "Report", "task": "t", "background": true}).to_string();

        tool.call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        let running = store
            .background_spawns_for_session("telegram:synthetic-user-1")
            .await
            .unwrap();
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].mission, "Report");

        wait_for_notification(&store).await;
        tokio::time::timeout(Duration::from_secs(5), async {
            while !store
                .background_spawns_for_session("telegram:synthetic-user-1")
                .await
                .unwrap()
                .is_empty()
            {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the record is cleared");
    }

    #[tokio::test]
    async fn the_owner_can_cancel_a_background_child() {
        let child = ScriptedChild::new("never delivered", Duration::from_secs(600));
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({"mission": "Report", "task": "t", "background": true}).to_string();

        tool.call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        child.cancel.cancel();

        let entry = wait_for_notification(&store).await;
        assert!(entry.message.contains("cancelled"), "{}", entry.message);
        assert!(!entry.message.contains("never delivered"));
    }

    #[tokio::test]
    async fn a_restart_tells_each_chat_what_it_lost() {
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        state
            .record_background_spawn(&crate::traits::BackgroundSpawn {
                id: "background_spawn:1".to_string(),
                session_id: "telegram:synthetic-user-1".to_string(),
                mission: "Publish the synthetic report".to_string(),
                started_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();

        report_interrupted_background_children(&state).await;

        let entry = wait_for_notification(&store).await;
        assert_eq!(entry.session_id, "telegram:synthetic-user-1");
        assert!(entry.message.contains("interrupted by a restart"));
        assert!(entry.message.contains("Publish the synthetic report"));
        assert!(store.take_background_spawns().await.unwrap().is_empty());
    }

    /// A task that merely starts like a delay reaches the child at once and
    /// word for word: wording never decides whether to wait.
    #[tokio::test]
    async fn task_wording_never_delays_the_child() {
        let child = ScriptedChild::new("done", Duration::ZERO);
        let tool = tool_for(&child);
        let task = "In 5 hours of logs, find the errors";
        let args = json!({"mission": "Log review", "task": task}).to_string();

        tokio::time::timeout(
            Duration::from_secs(5),
            tool.call_with_execution_context(&args, None, facts()),
        )
        .await
        .expect("the child starts immediately")
        .unwrap();

        assert_eq!(*child.tasks.lock().unwrap(), vec![task.to_string()]);
    }

    #[tokio::test]
    async fn truncated_foreground_result_says_what_is_missing() {
        let child = ScriptedChild::new(&"x".repeat(500), Duration::ZERO);
        let tool = tool_for(&child);
        let args = json!({"mission": "Report", "task": "Write the report"}).to_string();

        let outcome = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .unwrap();

        let truncation = outcome.metadata.truncation.expect("reported out of band");
        assert_eq!((truncation.shown_chars, truncation.total_chars), (100, 500));
        assert!(!outcome.output.contains("OUTPUT TRUNCATED"));
    }

    #[test]
    fn truncated_background_result_says_what_is_missing() {
        let (_, message) =
            format_background_completion("Report", &"x".repeat(500), TaskOutcome::Succeeded, 100);
        assert!(message.contains("OUTPUT TRUNCATED"));
    }

    /// Counts what reaches the chat.
    #[derive(Default)]
    struct CountingHub {
        sent: StdMutex<Vec<String>>,
    }

    #[async_trait]
    impl OutboundRouter for CountingHub {
        async fn send_text(&self, _session_id: &str, text: &str) -> anyhow::Result<()> {
            self.sent.lock().unwrap().push(text.to_string());
            Ok(())
        }
        async fn send_text_tracked(
            &self,
            _session_id: &str,
            _text: &str,
        ) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
        async fn edit_text(
            &self,
            _session_id: &str,
            _message_id: &str,
            _text: &str,
        ) -> anyhow::Result<bool> {
            Ok(false)
        }
        async fn send_media(
            &self,
            _session_id: &str,
            _media: &crate::types::MediaMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn send_media_strict(
            &self,
            _session_id: &str,
            _media: &crate::types::MediaMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn background_status_surface(&self, _session_id: &str) -> Option<String> {
            None
        }
        async fn take_background_status_surface(&self, _session_id: &str) -> Option<String> {
            None
        }
        async fn request_inline_approval(
            &self,
            _session_id: &str,
            _description: &str,
            _risk_level: crate::tools::command_risk::RiskLevel,
            _warnings: &[String],
            _permission_mode: crate::tools::command_risk::PermissionMode,
        ) -> anyhow::Result<crate::types::ApprovalResponse> {
            anyhow::bail!("no approvals in this test")
        }
        async fn register_session_route(&self, _child: &str, _parent: &str) -> bool {
            false
        }
        async fn unregister_session_route(&self, _child: &str, _parent: &str) {}
    }

    /// A background job reports when it has something to report: its result.
    #[tokio::test]
    async fn background_child_sends_no_interim_chat_messages() {
        let child = ScriptedChild::new("all done", Duration::from_millis(2_500));
        let hub = Arc::new(CountingHub::default());
        let router: Arc<dyn OutboundRouter> = hub.clone();
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        tool.set_hub(Arc::downgrade(&router));
        let args = json!({"mission": "Report", "task": "Write it", "background": true}).to_string();

        tool.call_with_execution_context(&args, None, facts())
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let pending = store.get_pending_notifications(10).await.unwrap();
                if pending.iter().any(|n| n.notification_type == "completed") {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("the result arrives");

        let sent = hub.sent.lock().unwrap();
        assert!(sent.is_empty(), "no interim pings: {sent:?}");
    }
}
