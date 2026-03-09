use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::types::StatusUpdate;

/// Role assigned to an agent for role-based tool scoping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    /// Root agent — routes, classifies, full tool access (legacy behavior).
    Orchestrator,
    /// Plans & delegates — management tools only.
    TaskLead,
    /// Executes a single task — action tools + report_blocker.
    Executor,
}

/// Categorization of a tool for role-based scoping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRole {
    /// Terminal, web_search, web_fetch, browser, etc.
    Action,
    /// ManageGoalTasksTool, ReportBlockerTool — task lead tools.
    Management,
    /// SystemInfoTool, RememberFactTool — available to all roles.
    Universal,
}

/// Safety and execution metadata for policy-driven tool selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCapabilities {
    pub read_only: bool,
    pub external_side_effect: bool,
    pub needs_approval: bool,
    pub idempotent: bool,
    pub high_impact_write: bool,
}

/// Effect classification for a specific tool call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ToolCallEffect {
    #[default]
    Unknown,
    Administrative,
    Observation,
    Mutation,
    ObservationAndMutation,
}

impl ToolCallEffect {
    pub fn observes_state(self) -> bool {
        matches!(self, Self::Observation | Self::ObservationAndMutation)
    }

    pub fn mutates_state(self) -> bool {
        matches!(self, Self::Mutation | Self::ObservationAndMutation)
    }
}

/// How a tool call can contribute verification evidence.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ToolVerificationMode {
    #[default]
    None,
    ResultContent,
}

/// Typed target hint emitted by a tool call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolTargetHintKind {
    Url,
    Path,
    ProjectScope,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolTargetHint {
    pub kind: ToolTargetHintKind,
    pub value: String,
}

impl ToolTargetHint {
    pub fn new(kind: ToolTargetHintKind, value: impl Into<String>) -> Option<Self> {
        let value = value.into().trim().to_string();
        if value.is_empty() {
            None
        } else {
            Some(Self { kind, value })
        }
    }
}

/// Structured completion semantics for a specific tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ToolCallSemantics {
    #[serde(default)]
    pub effect: ToolCallEffect,
    #[serde(default)]
    pub verification_mode: ToolVerificationMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_hints: Vec<ToolTargetHint>,
}

impl ToolCallSemantics {
    pub fn administrative() -> Self {
        Self {
            effect: ToolCallEffect::Administrative,
            ..Self::default()
        }
    }

    pub fn observation() -> Self {
        Self {
            effect: ToolCallEffect::Observation,
            ..Self::default()
        }
    }

    pub fn mutation() -> Self {
        Self {
            effect: ToolCallEffect::Mutation,
            ..Self::default()
        }
    }

    pub fn observation_and_mutation() -> Self {
        Self {
            effect: ToolCallEffect::ObservationAndMutation,
            ..Self::default()
        }
    }

    pub fn with_verification_mode(mut self, verification_mode: ToolVerificationMode) -> Self {
        self.verification_mode = verification_mode;
        self
    }

    pub fn with_target_hint(mut self, kind: ToolTargetHintKind, value: impl Into<String>) -> Self {
        if let Some(target) = ToolTargetHint::new(kind, value) {
            self.target_hints.push(target);
        }
        self
    }

    pub fn observes_state(&self) -> bool {
        self.effect.observes_state()
    }

    pub fn mutates_state(&self) -> bool {
        self.effect.mutates_state()
    }

    pub fn can_verify_with_result_content(&self) -> bool {
        self.verification_mode == ToolVerificationMode::ResultContent
    }

    pub fn is_empty(&self) -> bool {
        self.effect == ToolCallEffect::Unknown
            && self.verification_mode == ToolVerificationMode::None
            && self.target_hints.is_empty()
    }

    pub fn merge_missing_from(&mut self, fallback: Self) {
        if self.effect == ToolCallEffect::Unknown {
            self.effect = fallback.effect;
        }
        if self.verification_mode == ToolVerificationMode::None {
            self.verification_mode = fallback.verification_mode;
        }
        if self.target_hints.is_empty() {
            self.target_hints = fallback.target_hints;
        }
    }
}

/// Structured execution metadata returned by tools.
///
/// This is intentionally minimal and backward-compatible: tools can continue
/// returning plain text while selectively populating structured fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallMetadata {
    /// Process exit code when applicable (e.g. terminal/run_command style tools).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// True when tool execution exceeded a timeout threshold.
    #[serde(default)]
    pub timed_out: bool,
    /// True when execution moved to background tracking.
    #[serde(default)]
    pub background_started: bool,
    /// True when the process is detached and intentionally long-lived.
    #[serde(default)]
    pub detached: bool,
    /// True when the tool guarantees automatic completion delivery for a
    /// backgrounded operation in the current run.
    #[serde(default)]
    pub completion_notifications_enabled: bool,
    /// Transport/runtime failure outside normal tool semantics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport_error: Option<String>,
    /// Optional final user-facing reply. When set, the root agent may close the
    /// turn directly from the tool result instead of running another LLM pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direct_response: Option<String>,
    /// Completion semantics for this specific tool call.
    #[serde(default, skip_serializing_if = "ToolCallSemantics::is_empty")]
    pub semantics: ToolCallSemantics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallOutcome {
    pub output: String,
    #[serde(default)]
    pub metadata: ToolCallMetadata,
}

impl ToolCallOutcome {
    pub fn from_output(output: String) -> Self {
        Self {
            output,
            metadata: ToolCallMetadata::default(),
        }
    }
}

impl Default for ToolCapabilities {
    fn default() -> Self {
        Self {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: false,
        }
    }
}

fn tokenized_segments(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect()
}

fn contains_token(text: &str, token: &str) -> bool {
    let token = token.to_ascii_lowercase();
    tokenized_segments(text)
        .into_iter()
        .any(|segment| segment == token)
}

fn contains_any_token(text: &str, tokens: &[&str]) -> bool {
    tokens.iter().any(|token| contains_token(text, token))
}

fn json_string_arg(arguments: &str, key: &str) -> Option<String> {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get(key)
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
}

fn identifier_action_semantics(arguments: &str) -> Option<ToolCallSemantics> {
    let action = json_string_arg(arguments, "action")?;
    let lower = action.trim().to_ascii_lowercase();

    if lower.is_empty() {
        return None;
    }

    if lower == "providers" {
        return Some(
            ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent),
        );
    }

    if contains_any_token(&lower, &["trust", "close"]) {
        return Some(ToolCallSemantics::administrative());
    }

    if contains_token(&lower, "review") {
        let approve = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| value.get("approve").and_then(|value| value.as_bool()))
            .unwrap_or(false);
        return Some(if approve {
            ToolCallSemantics::mutation()
        } else {
            ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent)
        });
    }

    if contains_any_token(
        &lower,
        &[
            "trace", "history", "status", "summary", "describe", "compare", "diagnose", "brief",
            "upcoming", "usage", "audit", "verify", "timeline", "hints", "check",
        ],
    ) {
        return Some(
            ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent),
        );
    }

    if contains_any_token(
        &lower,
        &[
            "add", "create", "set", "switch", "connect", "refresh", "register", "remove", "delete",
            "update", "edit", "write", "upsert", "install", "enable", "disable", "pause", "resume",
            "retry", "cancel", "claim", "complete", "fail", "resolve", "share", "send", "link",
            "export", "purge", "confirm", "abandon", "run", "onboard", "restore", "promote",
        ],
    ) {
        return Some(ToolCallSemantics::mutation());
    }

    if contains_any_token(
        &lower,
        &[
            "list", "read", "get", "show", "view", "search", "find", "browse", "inspect",
        ],
    ) {
        return Some(
            ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent),
        );
    }

    None
}

fn http_method_semantics(arguments: &str) -> Option<ToolCallSemantics> {
    let method = json_string_arg(arguments, "method")?;
    let lower = method.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }
    Some(match lower.as_str() {
        "get" | "head" | "options" => ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent),
        _ => ToolCallSemantics::mutation(),
    })
}

fn string_to_target_hint(key: &str, value: &str) -> Option<ToolTargetHint> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    let lower_key = key.to_ascii_lowercase();
    if matches!(
        lower_key.as_str(),
        "url" | "verify_url" | "callback_url" | "target_url" | "auth_url"
    ) || trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
    {
        return ToolTargetHint::new(ToolTargetHintKind::Url, trimmed);
    }

    if matches!(
        lower_key.as_str(),
        "path"
            | "file_path"
            | "working_dir"
            | "directory"
            | "dir"
            | "repo_path"
            | "repo_dir"
            | "resource_path"
    ) || trimmed.starts_with('/')
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
        || trimmed.starts_with("~/")
    {
        return ToolTargetHint::new(ToolTargetHintKind::Path, trimmed);
    }

    if matches!(
        lower_key.as_str(),
        "project_path" | "project_dir" | "scope" | "project_scope"
    ) {
        return ToolTargetHint::new(ToolTargetHintKind::ProjectScope, trimmed);
    }

    None
}

fn push_unique_target_hint(hints: &mut Vec<ToolTargetHint>, candidate: Option<ToolTargetHint>) {
    let Some(candidate) = candidate else {
        return;
    };
    if !hints.iter().any(|existing| existing == &candidate) {
        hints.push(candidate);
    }
}

fn collect_common_target_hints(arguments: &str) -> Vec<ToolTargetHint> {
    let parsed = match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) => map,
        _ => return Vec::new(),
    };

    let mut hints = Vec::new();
    for (key, value) in &parsed {
        match value {
            Value::String(s) => push_unique_target_hint(&mut hints, string_to_target_hint(key, s)),
            Value::Array(items) if matches!(key.as_str(), "paths" | "urls") => {
                for item in items.iter().filter_map(|item| item.as_str()) {
                    push_unique_target_hint(&mut hints, string_to_target_hint(key, item));
                }
            }
            _ => {}
        }
    }
    hints
}

fn default_semantics_from_identity(
    name: &str,
    description: &str,
    arguments: &str,
    caps: ToolCapabilities,
) -> ToolCallSemantics {
    if let Some(mut semantics) = identifier_action_semantics(arguments) {
        for target_hint in collect_common_target_hints(arguments) {
            semantics = semantics.with_target_hint(target_hint.kind, target_hint.value);
        }
        return semantics;
    }

    if let Some(mut semantics) = http_method_semantics(arguments) {
        for target_hint in collect_common_target_hints(arguments) {
            semantics = semantics.with_target_hint(target_hint.kind, target_hint.value);
        }
        return semantics;
    }

    let mut semantics = if caps.read_only {
        ToolCallSemantics::observation().with_verification_mode(ToolVerificationMode::ResultContent)
    } else {
        let identity = format!("{} {}", name, description);
        if contains_any_token(
            &identity,
            &[
                "read", "list", "show", "fetch", "search", "inspect", "trace", "status", "info",
                "metrics", "history", "usage", "brief", "browse", "check", "verify", "query",
                "view",
            ],
        ) && !contains_any_token(
            &identity,
            &[
                "create", "update", "remove", "delete", "add", "set", "write", "edit", "send",
                "share", "register", "install", "connect", "commit", "spawn", "run",
            ],
        ) {
            ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent)
        } else if contains_any_token(
            &identity,
            &[
                "create", "update", "remove", "delete", "add", "set", "write", "edit", "send",
                "share", "register", "install", "connect", "commit", "spawn", "run", "manage",
                "store", "remember", "save", "report", "blocker",
            ],
        ) || caps.external_side_effect
            || caps.high_impact_write
        {
            ToolCallSemantics::mutation()
        } else {
            ToolCallSemantics::administrative()
        }
    };

    for target_hint in collect_common_target_hints(arguments) {
        semantics = semantics.with_target_hint(target_hint.kind, target_hint.value);
    }
    semantics
}

/// Tool trait — system tools, terminal, MCP-proxied tools.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// Returns the OpenAI-format function schema as a JSON Value.
    fn schema(&self) -> Value;
    /// Execute the tool with the given JSON arguments string, returns result text.
    async fn call(&self, arguments: &str) -> anyhow::Result<String>;

    /// Execute the tool with access to a status update channel for streaming feedback.
    /// Default implementation just calls `call()` - override for tools that emit progress.
    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        // Default: ignore status channel and just call the basic method
        let _ = status_tx;
        self.call(arguments).await
    }

    /// Structured execution path used by the agent loop.
    ///
    /// Default behavior preserves compatibility for existing tools by wrapping
    /// plain text output with empty metadata.
    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let output = self.call_with_status(arguments, status_tx).await?;
        Ok(ToolCallOutcome::from_output(output))
    }

    /// Task lifecycle callback fired after the agent emits `TaskEnd`.
    /// Tools that spawn background activity can use this to clean up
    /// task-scoped resources.
    async fn on_task_end(&self, _task_id: &str, _session_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Categorize this tool for role-based scoping.
    /// Default: Action (most tools are action tools).
    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    /// Capability metadata used by the execution policy and risk gate.
    /// Defaults are intentionally conservative.
    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities::default()
    }

    /// Structured completion semantics for a specific call.
    ///
    /// Default behavior derives a conservative fallback from `capabilities()`.
    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        default_semantics_from_identity(
            self.name(),
            self.description(),
            arguments,
            self.capabilities(),
        )
    }

    /// Whether this tool is currently operational.
    ///
    /// Default: true. Override for tools with dynamic backends that may be
    /// temporarily unavailable at runtime.
    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct AlwaysAvailableTool;

    #[async_trait]
    impl Tool for AlwaysAvailableTool {
        fn name(&self) -> &str {
            "always_available"
        }

        fn description(&self) -> &str {
            "test"
        }

        fn schema(&self) -> Value {
            json!({
                "name": "always_available",
                "description": "test",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            })
        }

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }
    }

    struct UnavailableTool;

    #[async_trait]
    impl Tool for UnavailableTool {
        fn name(&self) -> &str {
            "unavailable"
        }

        fn description(&self) -> &str {
            "test"
        }

        fn schema(&self) -> Value {
            json!({
                "name": "unavailable",
                "description": "test",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            })
        }

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }

        fn is_available(&self) -> bool {
            false
        }
    }

    #[test]
    fn default_is_available_returns_true() {
        let tool = AlwaysAvailableTool;
        assert!(tool.is_available());
    }

    #[test]
    fn override_is_available_returns_false() {
        let tool = UnavailableTool;
        assert!(!tool.is_available());
    }

    struct ManageTool;

    #[async_trait]
    impl Tool for ManageTool {
        fn name(&self) -> &str {
            "manage_demo"
        }

        fn description(&self) -> &str {
            "Manage demo entities"
        }

        fn schema(&self) -> Value {
            json!({
                "name": "manage_demo",
                "description": "Manage demo entities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "path": {"type": "string"}
                    },
                    "additionalProperties": false
                }
            })
        }

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }
    }

    #[test]
    fn default_call_semantics_classifies_structural_actions() {
        let tool = ManageTool;
        let list = tool.call_semantics(r#"{"action":"list","path":"/tmp/demo"}"#);
        assert!(list.observes_state());
        assert!(!list.mutates_state());
        assert!(list.can_verify_with_result_content());
        assert_eq!(
            list.target_hints,
            vec![ToolTargetHint {
                kind: ToolTargetHintKind::Path,
                value: "/tmp/demo".to_string()
            }]
        );

        let remove = tool.call_semantics(r#"{"action":"remove","path":"/tmp/demo"}"#);
        assert!(remove.mutates_state());
        assert!(!remove.observes_state());
    }

    #[test]
    fn review_action_becomes_mutation_when_approved() {
        let tool = ManageTool;
        let review = tool.call_semantics(r#"{"action":"review","approve":true}"#);
        assert!(review.mutates_state());
        assert!(!review.observes_state());
    }

    #[test]
    fn mutation_verbs_beat_entity_nouns_in_action_names() {
        let tool = ManageTool;
        let remove = tool.call_semantics(r#"{"action":"remove_provider"}"#);
        assert!(remove.mutates_state());

        let history = tool.call_semantics(r#"{"action":"run_history"}"#);
        assert!(history.observes_state());
        assert!(!history.mutates_state());
    }

    struct RememberTool;

    #[async_trait]
    impl Tool for RememberTool {
        fn name(&self) -> &str {
            "remember_fact"
        }

        fn description(&self) -> &str {
            "Store one or more long-lived facts for later"
        }

        fn schema(&self) -> Value {
            json!({
                "name": "remember_fact",
                "description": "Store facts",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            })
        }

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }
    }

    #[test]
    fn identity_mutation_keywords_cover_non_action_tools() {
        let tool = RememberTool;
        let semantics = tool.call_semantics("{}");
        assert!(semantics.mutates_state());
    }
}
