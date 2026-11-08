//! # aidaemon-agents (experimental)
//!
//! Multi-agent composition for `aidaemon-core`, built only on its public API.
//!
//! The one pattern supported here is **agent-as-tool**: a parent agent calls a
//! child agent the way it calls any other tool, the child runs to completion
//! in its own session, and the parent receives a *condensed* result plus
//! structured evidence (status, turns, per-receipt summary, child run id).
//! Reports from Anthropic (orchestrator + condensed sub-agent results) and
//! Cognition (parallel writers and persona crews diverge) point the same way:
//! delegate bounded tasks, keep one writer, return summaries.
//!
//! Design points:
//!
//! - **Fresh session per call.** The child runs in
//!   `{parent_session}/sub/{call_id}` ([`child_session_id`]), so stores and
//!   traces can relate it to the parent without sharing a transcript.
//! - **Explicit context.** [`ContextMode::Brief`] (default) gives the child
//!   only the task and the optional `context` argument.
//!   [`ContextMode::Fork`] also prepends a parent-transcript summary, which
//!   is *caller-supplied*: a tool cannot read the parent's transcript from
//!   [`ToolContext`], so the summary comes from a construction-time closure
//!   ([`ForkSource`]). [`TranscriptTap`] is a ready-made hook that captures
//!   one for you.
//! - **Budgets roll up.** The child's usage is returned through
//!   [`ToolOutput::with_usage`], which the kernel adds to the parent run, so
//!   parent token limits cover delegated work. Per-child caps come from
//!   [`ChildBudget`] (default `max_turns` 20).
//! - **Failures are visible.** A child that ends with anything other than
//!   `Completed` becomes an error result (failed receipt) naming the status.
//! - **Sequential by default.** The tool is annotated
//!   [`ToolAnnotations::WRITE`], so two children never run concurrently
//!   unless the caller marks them read-only (e.g. research children).
//! - **Depth guard.** Nesting depth is the number of `/sub/` segments in the
//!   session id; calls beyond `max_depth` are refused.
//!
//! ## Synchronous mode and approvals
//!
//! A child that pauses on a `before_tool` interrupt cannot be resumed from
//! inside the parent's tool call, so an `Interrupted` child is reported as an
//! error. Give children tools their policies allow without asking, or run an
//! approval-requiring agent as its own top-level run.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::{
    Agent, AgentBuilder, AuthorityHook, BuildError, FinishOutcome, Hook, HookContext, Limits,
    Message, ModelRequest, Role, RunOutcome, RunState, RunStatus, Tool, ToolAnnotations,
    ToolContext, ToolError, ToolOutput, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::{json, Value};

/// The segment that separates a parent session from a child session.
pub const SUB_SEGMENT: &str = "/sub/";

/// Default cap on the condensed child result handed back to the parent.
pub const DEFAULT_MAX_RESULT_CHARS: usize = 4_000;

/// Default maximum nesting depth (a child of a child of a child).
pub const DEFAULT_MAX_DEPTH: usize = 3;

/// Session id for the child spawned by `call_id` inside `parent_session`.
pub fn child_session_id(parent_session: &str, call_id: &str) -> String {
    format!("{parent_session}{SUB_SEGMENT}{call_id}")
}

/// Nesting depth encoded in a session id: the number of `/sub/` segments.
/// A top-level session has depth 0.
pub fn session_depth(session_id: &str) -> usize {
    session_id.matches(SUB_SEGMENT).count()
}

/// Keep the head and tail of `text` within `max_chars` characters (plus a
/// one-line notice). The conclusion of a report is usually at the end and
/// the framing at the start, so both survive.
pub fn condense(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if max_chars == 0 || total <= max_chars {
        return text.to_string();
    }
    let head = max_chars * 2 / 3;
    let tail = max_chars - head;
    let omitted = total - head - tail;
    let head_text: String = text.chars().take(head).collect();
    let tail_text: String = text.chars().skip(total - tail).collect();
    format!("{head_text}\n[... {omitted} characters of sub-agent output omitted ...]\n{tail_text}")
}

/// Produces a summary of the parent's transcript for [`ContextMode::Fork`].
/// It receives the trusted context of the delegating call (session, run and
/// call ids) and returns `None` when no summary is available.
pub type ForkSource = Arc<dyn Fn(&ToolContext) -> Option<String> + Send + Sync>;

/// What the child sees besides the task.
#[derive(Clone, Default)]
pub enum ContextMode {
    /// Only the task and the optional `context` argument.
    #[default]
    Brief,
    /// The task, the optional `context` argument, and a parent-transcript
    /// summary from the given source. If the source returns `None`, this
    /// behaves like `Brief`.
    Fork(ForkSource),
}

impl fmt::Debug for ContextMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Brief => f.write_str("Brief"),
            Self::Fork(_) => f.write_str("Fork(..)"),
        }
    }
}

/// Per-child hard caps, applied when the child agent is built through
/// [`AgentTool::from_builder`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChildBudget {
    pub max_turns: u32,
    /// Billable-token cap for one child run.
    pub max_tokens: Option<u64>,
    pub max_tool_calls: Option<u32>,
    pub timeout: Option<Duration>,
}

impl Default for ChildBudget {
    fn default() -> Self {
        Self {
            max_turns: 20,
            max_tokens: None,
            max_tool_calls: None,
            timeout: None,
        }
    }
}

impl ChildBudget {
    /// Overlay this budget on `base` (tool timeout and output caps are kept).
    pub fn apply(&self, mut base: Limits) -> Limits {
        base.max_turns = self.max_turns;
        base.max_tokens = self.max_tokens;
        base.max_tool_calls = self.max_tool_calls;
        base.timeout = self.timeout;
        base
    }

    /// The budget as a complete [`Limits`], for building a child by hand.
    pub fn limits(&self) -> Limits {
        self.apply(Limits::default())
    }
}

/// A child [`Agent`] exposed as a [`Tool`]. Arguments: `{task, context?}`.
///
/// Note that the parent's `Limits::tool_timeout` (default 120 s) bounds the
/// whole child run; raise it on parents that delegate long tasks.
#[derive(Clone)]
pub struct AgentTool {
    agent: Agent,
    name: String,
    description: String,
    mode: ContextMode,
    max_result_chars: usize,
    annotations: ToolAnnotations,
    max_depth: usize,
}

/// Alias matching the name used in the design notes.
pub type SubAgentTool = AgentTool;

impl fmt::Debug for AgentTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentTool")
            .field("agent", &self.agent.id())
            .field("name", &self.name)
            .field("mode", &self.mode)
            .field("max_result_chars", &self.max_result_chars)
            .field("annotations", &self.annotations)
            .field("max_depth", &self.max_depth)
            .finish()
    }
}

impl AgentTool {
    /// Wrap an already-built agent. The child keeps the limits it was built
    /// with; call [`AgentTool::budget`] (or use [`AgentTool::from_builder`])
    /// to give it child-sized caps.
    pub fn new(agent: Agent) -> Self {
        let name = sanitize_tool_name(agent.id());
        let description = format!(
            "Delegate a self-contained task to the `{}` sub-agent. It starts with no \
             memory of this conversation: put everything it needs in `task` and \
             `context`. Returns its final answer.",
            agent.id()
        );
        Self {
            agent,
            name,
            description,
            mode: ContextMode::Brief,
            max_result_chars: DEFAULT_MAX_RESULT_CHARS,
            annotations: ToolAnnotations::WRITE,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Build the child from `builder` with `budget` applied. This replaces
    /// any limits already set on the builder.
    /// Give the wrapped agent child-sized caps, replacing its own limits.
    pub fn budget(mut self, budget: ChildBudget) -> Self {
        self.agent = self.agent.with_limits(budget.limits());
        self
    }

    pub fn from_builder(builder: AgentBuilder, budget: ChildBudget) -> Result<Self, BuildError> {
        let agent = builder.limits(budget.limits()).build()?;
        Ok(Self::new(agent))
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn context_mode(mut self, mode: ContextMode) -> Self {
        self.mode = mode;
        self
    }

    /// Shorthand for `context_mode(ContextMode::Fork(source))`.
    pub fn fork_from(self, source: ForkSource) -> Self {
        self.context_mode(ContextMode::Fork(source))
    }

    /// Cap on the condensed result (head + tail). 0 disables condensing.
    pub fn max_result_chars(mut self, chars: usize) -> Self {
        self.max_result_chars = chars;
        self
    }

    pub fn with_annotations(mut self, annotations: ToolAnnotations) -> Self {
        self.annotations = annotations;
        self
    }

    /// Mark the child read-only so several calls in one turn run
    /// concurrently. Only do this for children whose tools are all
    /// read-only (research, search, review).
    pub fn read_only(self) -> Self {
        self.with_annotations(ToolAnnotations::READ_ONLY)
    }

    /// Maximum session depth a child may run at (top level is 0, its child
    /// is 1). Calls that would exceed it are refused.
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn agent(&self) -> &Agent {
        &self.agent
    }

    fn compose_input(&self, task: &str, context: Option<&str>, ctx: &ToolContext) -> String {
        let summary = match &self.mode {
            ContextMode::Brief => None,
            ContextMode::Fork(source) => source(ctx).filter(|s| !s.trim().is_empty()),
        };
        let context = context.filter(|c| !c.trim().is_empty());
        match summary {
            None => match context {
                None => task.to_string(),
                Some(context) => format!("{task}\n\nContext:\n{context}"),
            },
            Some(summary) => {
                let mut input = format!(
                    "[Summary of the delegating agent's conversation]\n{summary}\n\n[Task]\n{task}"
                );
                if let Some(context) = context {
                    input.push_str(&format!("\n\n[Context]\n{context}"));
                }
                input
            }
        }
    }

    fn details(&self, outcome: &RunOutcome, session: &str) -> Value {
        let receipts: Vec<Value> = outcome
            .receipts()
            .iter()
            .map(|r| json!({"tool": r.tool, "status": r.status}))
            .collect();
        json!({
            "sub_agent": self.agent.id(),
            "child_run_id": outcome.state.run_id,
            "child_session_id": session,
            "status": outcome.status,
            "succeeded": outcome.is_succeeded(),
            "unmet": outcome.unmet(),
            "turns": outcome.turns(),
            "receipts": receipts,
            "usage": outcome.usage(),
        })
    }
}

/// `agent.as_tool(name, description)`.
pub trait AgentExt {
    fn as_tool(&self, name: impl Into<String>, description: impl Into<String>) -> AgentTool;
}

impl AgentExt for Agent {
    fn as_tool(&self, name: impl Into<String>, description: impl Into<String>) -> AgentTool {
        AgentTool::new(self.clone())
            .name(name)
            .description(description)
    }
}

#[async_trait]
impl Tool for AgentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The complete, self-contained task for the sub-agent."
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional facts, constraints or prior findings the sub-agent needs."
                    }
                },
                "required": ["task"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        self.annotations
    }

    fn validate(&self, arguments: &Value) -> Result<(), String> {
        let Some(object) = arguments.as_object() else {
            return Err("arguments must be a JSON object".into());
        };
        match object.get("task").and_then(Value::as_str) {
            Some(task) if !task.trim().is_empty() => {}
            Some(_) => return Err("`task` must not be empty".into()),
            None => return Err("missing required string argument: task".into()),
        }
        match object.get("context") {
            None | Some(Value::Null) | Some(Value::String(_)) => {}
            Some(_) => return Err("`context` must be a string".into()),
        }
        if let Some(extra) = object.keys().find(|k| *k != "task" && *k != "context") {
            return Err(format!("unexpected argument `{extra}`"));
        }
        Ok(())
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> ToolResult {
        let depth = session_depth(&ctx.session_id) + 1;
        if depth > self.max_depth {
            return Ok(ToolOutput::error(format!(
                "Refused: delegating to `{}` would run a sub-agent at depth {depth}, beyond the \
                 maximum of {}. Do this part of the work yourself.",
                self.name, self.max_depth
            ))
            .with_details(json!({
                "refused": "max_depth",
                "depth": depth,
                "max_depth": self.max_depth,
            })));
        }

        let task = arguments["task"].as_str().unwrap_or_default();
        let context = arguments.get("context").and_then(Value::as_str);
        let input = self.compose_input(task, context, ctx);
        let session = child_session_id(&ctx.session_id, &ctx.call_id);

        // The child answers to everything the parent does: its authority
        // stacks on the child's own, and a decline in one binds the other.
        let agent = match &ctx.authority {
            Some(authority) => self.agent.with_hook(AuthorityHook::new(authority.clone())),
            None => self.agent.clone(),
        };
        let mut state = RunState::new(agent.id(), &session, &input).with_parent(ctx.run_id);
        if let Some(declines) = &ctx.declines {
            state = state.share_declines(declines.clone());
        }
        let outcome = agent
            .run_from(state, ctx.cancel.clone())
            .await
            .map_err(|e| ToolError(format!("sub-agent `{}` could not run: {e}", self.name)))?;

        let details = self.details(&outcome, &session);
        let usage = outcome.usage();
        let output = match &outcome.status {
            // Completed means the child stopped, not that it did the work:
            // unmet requirements are passed up so the caller cannot mistake
            // a stop for a success.
            RunStatus::Completed if !outcome.unmet().is_empty() => {
                let text = condense(
                    outcome.final_text.as_deref().unwrap_or_default(),
                    self.max_result_chars,
                );
                let mut message = format!(
                    "sub-agent `{}` stopped without finishing; still unmet:",
                    self.name
                );
                for line in outcome.unmet() {
                    message.push_str("\n- ");
                    message.push_str(line);
                }
                message.push_str("\n\nIts reply:\n");
                message.push_str(&text);
                ToolOutput::text(message)
            }
            RunStatus::Completed => {
                let text = outcome.final_text.as_deref().unwrap_or_default();
                ToolOutput::text(condense(text, self.max_result_chars))
            }
            // A tool in the sub-agent ended its run: the finish reply is the
            // answer. Partial work says so; a blocked one is a failure the
            // caller should react to.
            RunStatus::Finished {
                outcome: finish, ..
            } => {
                let text = condense(
                    outcome.final_text.as_deref().unwrap_or_default(),
                    self.max_result_chars,
                );
                match finish {
                    FinishOutcome::Succeeded => ToolOutput::text(text),
                    FinishOutcome::Partial => ToolOutput::text(format!(
                        "sub-agent `{}` finished part of the task:\n{text}",
                        self.name
                    )),
                    _ => {
                        ToolOutput::error(format!("sub-agent `{}` is blocked:\n{text}", self.name))
                    }
                }
            }
            RunStatus::Interrupted { interruption } => ToolOutput::error(format!(
                "sub-agent `{}` paused waiting for approval of `{}` ({}). Sub-agents that need \
                 approval are not supported in synchronous agent-as-tool mode, so the call did \
                 not run. Do it yourself (your own approval policy applies) or ask the user.",
                self.name, interruption.call.name, interruption.reason
            )),
            status => {
                let mut message = format!(
                    "sub-agent `{}` did not complete: {}.",
                    self.name,
                    status_label(status)
                );
                if let Some(partial) = last_assistant_text(&outcome.state.messages) {
                    message.push_str("\nLast message from the sub-agent:\n");
                    message.push_str(&condense(partial, self.max_result_chars));
                }
                ToolOutput::error(message)
            }
        };
        Ok(output.with_details(details).with_usage(usage))
    }
}

/// A short machine-derived label for a terminal status.
pub fn status_label(status: &RunStatus) -> String {
    match status {
        RunStatus::Completed => "completed".into(),
        RunStatus::Interrupted { interruption } => {
            format!("interrupted before `{}`", interruption.call.name)
        }
        RunStatus::LimitReached { limit } => {
            let kind = serde_json::to_value(limit)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_else(|| format!("{limit:?}"));
            format!("limit reached ({kind})")
        }
        RunStatus::Stopped { reason } => format!("stopped ({reason})"),
        RunStatus::Stuck { signal } => format!("stuck ({signal:?})"),
        RunStatus::Cancelled => "cancelled".into(),
        RunStatus::ProviderFailed { error } => format!("provider failed ({error})"),
        RunStatus::Finished { outcome, tool } => {
            let how = match outcome {
                FinishOutcome::Succeeded => "succeeded",
                FinishOutcome::Partial => "partial",
                FinishOutcome::Blocked => "blocked",
                _ => "ended",
            };
            format!("finished by `{tool}` ({how})")
        }
        other => format!("{other:?}"),
    }
}

fn last_assistant_text(messages: &[Message]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .filter(|m| m.role == Role::Assistant)
        .map(|m| m.content.as_str())
        .find(|c| !c.trim().is_empty())
}

fn sanitize_tool_name(id: &str) -> String {
    let mut name: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    if name.is_empty() {
        name.push_str("sub_agent");
    }
    name
}

/// A hook that records a compact summary of each session's transcript so a
/// [`ContextMode::Fork`] child can see what the parent has been doing.
///
/// Register it on the **parent** and pass [`TranscriptTap::source`] to
/// [`AgentTool::fork_from`]. The summary is taken in `before_model`, so it
/// reflects the transcript up to (not including) the turn that delegates.
#[derive(Clone)]
pub struct TranscriptTap {
    summaries: Arc<Mutex<HashMap<String, String>>>,
    max_messages: usize,
    max_chars_per_message: usize,
}

impl Default for TranscriptTap {
    fn default() -> Self {
        Self::new(12, 400)
    }
}

impl TranscriptTap {
    pub fn new(max_messages: usize, max_chars_per_message: usize) -> Self {
        Self {
            summaries: Arc::new(Mutex::new(HashMap::new())),
            max_messages,
            max_chars_per_message,
        }
    }

    /// The latest summary captured for `session_id`.
    pub fn summary(&self, session_id: &str) -> Option<String> {
        self.summaries
            .lock()
            .ok()
            .and_then(|m| m.get(session_id).cloned())
    }

    /// Forget a session (call when it ends to bound memory).
    pub fn clear(&self, session_id: &str) {
        if let Ok(mut map) = self.summaries.lock() {
            map.remove(session_id);
        }
    }

    /// A [`ForkSource`] reading this tap by the delegating call's session.
    pub fn source(&self) -> ForkSource {
        let tap = self.clone();
        Arc::new(move |ctx: &ToolContext| tap.summary(&ctx.session_id))
    }

    fn summarize(&self, messages: &[Message]) -> String {
        let start = messages.len().saturating_sub(self.max_messages);
        let mut lines = Vec::new();
        for message in &messages[start..] {
            let role = match message.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool result",
            };
            let content = condense(message.content.trim(), self.max_chars_per_message);
            if !content.is_empty() {
                lines.push(format!("{role}: {content}"));
            }
            for call in &message.tool_calls {
                lines.push(format!("assistant called `{}`", call.name));
            }
        }
        lines.join("\n")
    }
}

#[async_trait]
impl Hook for TranscriptTap {
    async fn before_model(&self, ctx: &HookContext<'_>, _request: &mut ModelRequest) {
        let summary = self.summarize(ctx.messages);
        if let Ok(mut map) = self.summaries.lock() {
            map.insert(ctx.session_id.to_string(), summary);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_counts_sub_segments() {
        assert_eq!(session_depth("telegram:synthetic-user-1"), 0);
        let child = child_session_id("telegram:synthetic-user-1", "c1");
        assert_eq!(child, "telegram:synthetic-user-1/sub/c1");
        assert_eq!(session_depth(&child), 1);
        assert_eq!(session_depth(&child_session_id(&child, "c2")), 2);
    }

    #[test]
    fn condense_keeps_head_and_tail() {
        let text: String = (0..1000)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let out = condense(&text, 90);
        assert!(out.starts_with(&text[..60]));
        assert!(out.ends_with(&text[970..]));
        assert!(out.contains("910 characters"));
        assert_eq!(condense("short", 90), "short");
        assert_eq!(condense(&text, 0), text);
    }

    #[test]
    fn tool_names_are_sanitized() {
        assert_eq!(sanitize_tool_name("research agent.v2"), "research_agent_v2");
        assert_eq!(sanitize_tool_name(""), "sub_agent");
    }

    #[test]
    fn budget_overlays_limits() {
        let limits = ChildBudget {
            max_tokens: Some(500),
            ..ChildBudget::default()
        }
        .limits();
        assert_eq!(limits.max_turns, 20);
        assert_eq!(limits.max_tokens, Some(500));
    }
}
