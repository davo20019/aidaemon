use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::agent::{contains_keyword_as_words, FollowupMode, TurnContext};
use crate::traits::{
    AgentRole, ToolCallEffect, ToolCallSemantics, ToolCapabilities, ToolTargetHint,
    ToolTargetHintKind,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BudgetTier {
    None,
    Small,
    Standard,
    Extended,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPersistence {
    Ephemeral,
    Durable,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionBudgetLimit {
    Steps,
    Tokens,
    LlmCalls,
    ToolCalls,
    ValidationRounds,
    WallClock,
}

impl ExecutionBudgetLimit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Steps => "max_steps",
            Self::Tokens => "max_tokens",
            Self::LlmCalls => "max_llm_calls",
            Self::ToolCalls => "max_tool_calls",
            Self::ValidationRounds => "max_validation_rounds",
            Self::WallClock => "max_wall_clock_ms",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct TargetScope {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_targets: Vec<ToolTargetHint>,
    #[serde(default)]
    pub hard_fail_outside_scope: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    #[serde(default)]
    pub allow_tool_invocation_retry: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalRequirement {
    NotNeeded,
    Required { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StepExecutionPlan {
    pub step_id: String,
    pub description: String,
    pub plan_version: u32,
    pub primary_tool: Option<String>,
    pub expected_effect: ToolCallEffect,
    pub target_scope: TargetScope,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_targets: Vec<ToolTargetHint>,
    pub retry_policy: RetryPolicy,
    pub approval_requirement: ApprovalRequirement,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StepExecutionOutcome {
    Progress,
    NoProgress,
    RecoverableFailure,
    NonrecoverableFailure,
    BackgroundDetached,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionBudget {
    pub max_steps: usize,
    pub max_tokens: usize,
    pub max_llm_calls: usize,
    pub max_tool_calls: usize,
    pub max_validation_rounds: usize,
    pub max_wall_clock_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionState {
    pub execution_id: String,
    pub current_step: Option<StepExecutionPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_plan_version: Option<u32>,
    pub attempt_count: usize,
    pub last_tool_name: Option<String>,
    pub last_outcome: Option<StepExecutionOutcome>,
    pub background_handoff_active: bool,
    pub persisted_at: Option<DateTime<Utc>>,
    pub budget_tier: BudgetTier,
    pub budget: ExecutionBudget,
    pub persistence: ExecutionPersistence,
    #[serde(default)]
    pub budget_envelope_active: bool,
    #[serde(default)]
    pub final_response_closeout_active: bool,
    #[serde(default)]
    pub budget_started_task_tokens: u64,
    #[serde(default)]
    pub budget_started_elapsed_ms: u64,
    pub llm_calls_used: usize,
    pub tool_calls_used: usize,
    pub validation_rounds_used: usize,
    pub steps_used: usize,
}

impl ExecutionState {
    pub fn new(
        budget_tier: BudgetTier,
        budget: ExecutionBudget,
        persistence: ExecutionPersistence,
    ) -> Self {
        Self {
            execution_id: Uuid::new_v4().to_string(),
            current_step: None,
            current_plan_version: None,
            attempt_count: 0,
            last_tool_name: None,
            last_outcome: None,
            background_handoff_active: false,
            persisted_at: None,
            budget_tier,
            budget,
            persistence,
            budget_envelope_active: false,
            final_response_closeout_active: false,
            budget_started_task_tokens: 0,
            budget_started_elapsed_ms: 0,
            llm_calls_used: 0,
            tool_calls_used: 0,
            validation_rounds_used: 0,
            steps_used: 0,
        }
    }

    pub fn promote_persistence(&mut self, persistence: ExecutionPersistence) {
        if matches!(persistence, ExecutionPersistence::Durable) {
            self.persistence = ExecutionPersistence::Durable;
        }
    }

    pub fn mark_persisted_now(&mut self) {
        self.persisted_at = Some(Utc::now());
    }

    pub fn set_plan_version(&mut self, plan_version: u32) {
        if plan_version > 0 {
            self.current_plan_version = Some(plan_version);
        }
    }

    pub fn activate_budget_envelope(&mut self, task_tokens_used: u64, elapsed: Duration) {
        if self.budget_envelope_active {
            return;
        }

        self.budget_envelope_active = true;
        self.final_response_closeout_active = false;
        self.budget_started_task_tokens = task_tokens_used;
        self.budget_started_elapsed_ms = elapsed.as_millis().min(u64::MAX as u128) as u64;
    }

    pub fn execution_budget_applies(&self) -> bool {
        self.budget_envelope_active
    }

    pub fn suspend_budget_for_final_response(&mut self) {
        self.budget_envelope_active = false;
        self.final_response_closeout_active = true;
    }

    pub fn record_llm_call(&mut self) {
        self.llm_calls_used = self.llm_calls_used.saturating_add(1);
    }

    pub fn begin_step(&mut self, plan: StepExecutionPlan) {
        self.steps_used = self.steps_used.saturating_add(1);
        self.attempt_count = self.attempt_count.saturating_add(1);
        self.last_tool_name = plan.primary_tool.clone();
        self.current_step = Some(plan);
    }

    pub fn record_tool_call(&mut self) {
        self.tool_calls_used = self.tool_calls_used.saturating_add(1);
    }

    pub fn record_validation_round(&mut self) {
        self.validation_rounds_used = self.validation_rounds_used.saturating_add(1);
    }

    pub fn complete_current_step(&mut self, outcome: StepExecutionOutcome) {
        self.background_handoff_active =
            matches!(outcome, StepExecutionOutcome::BackgroundDetached);
        self.last_outcome = Some(outcome);
    }

    /// Extend the budget when a tool call completes successfully.
    ///
    /// This implements the principle "only limit when wasting tokens, not when
    /// making progress."  Each successful tool execution earns additional
    /// capacity so productive multi-step runs are never artificially stopped by
    /// the initial budget ceiling.  Stall detection, repetition guards, and
    /// wall-clock limits remain the primary defences against genuine waste.
    pub fn extend_budget_on_progress(&mut self) {
        if !self.budget_envelope_active {
            return;
        }
        const PROGRESS_EXTENSION: usize = 6;
        /// Wall-clock extension per successful tool call (30 seconds).
        /// Slow external APIs, large builds, or chained terminal commands
        /// can legitimately consume significant wall time while making
        /// real progress.
        const WALL_CLOCK_EXTENSION_MS: u64 = 30_000;
        if self.budget.max_llm_calls > 0 {
            self.budget.max_llm_calls =
                self.budget.max_llm_calls.saturating_add(PROGRESS_EXTENSION);
        }
        if self.budget.max_tool_calls > 0 {
            self.budget.max_tool_calls = self
                .budget
                .max_tool_calls
                .saturating_add(PROGRESS_EXTENSION);
        }
        if self.budget.max_steps > 0 {
            self.budget.max_steps = self.budget.max_steps.saturating_add(PROGRESS_EXTENSION);
        }
        if self.budget.max_wall_clock_ms > 0 {
            self.budget.max_wall_clock_ms = self
                .budget
                .max_wall_clock_ms
                .saturating_add(WALL_CLOCK_EXTENSION_MS);
        }
    }

    pub fn exhausted_limit(
        &self,
        task_tokens_used: u64,
        elapsed: Duration,
    ) -> Option<ExecutionBudgetLimit> {
        if !self.budget_envelope_active {
            return None;
        }

        let execution_tokens_used =
            task_tokens_used.saturating_sub(self.budget_started_task_tokens);
        let execution_elapsed_ms = (elapsed.as_millis().min(u64::MAX as u128) as u64)
            .saturating_sub(self.budget_started_elapsed_ms);

        if self.budget.max_steps > 0 && self.steps_used >= self.budget.max_steps {
            return Some(ExecutionBudgetLimit::Steps);
        }
        if self.budget.max_tokens > 0 && execution_tokens_used as usize >= self.budget.max_tokens {
            return Some(ExecutionBudgetLimit::Tokens);
        }
        if self.budget.max_llm_calls > 0 && self.llm_calls_used >= self.budget.max_llm_calls {
            return Some(ExecutionBudgetLimit::LlmCalls);
        }
        if self.budget.max_tool_calls > 0 && self.tool_calls_used >= self.budget.max_tool_calls {
            return Some(ExecutionBudgetLimit::ToolCalls);
        }
        if self.budget.max_validation_rounds > 0
            && self.validation_rounds_used >= self.budget.max_validation_rounds
        {
            return Some(ExecutionBudgetLimit::ValidationRounds);
        }
        if self.budget.max_wall_clock_ms > 0
            && execution_elapsed_ms >= self.budget.max_wall_clock_ms
        {
            return Some(ExecutionBudgetLimit::WallClock);
        }
        None
    }
}

pub fn default_execution_budget(tier: BudgetTier) -> ExecutionBudget {
    // Do not hard-stop productive runs on cumulative LLM token usage. The
    // agent already has explicit anti-waste controls (stall detection,
    // repetition guards, llm/tool call caps, wall-clock limits, and optional
    // task/daily token budgets). Keep the execution-layer token cap disabled so
    // successful multi-step work can finish.
    match tier {
        BudgetTier::None => ExecutionBudget {
            max_steps: 24,
            max_tokens: 0,
            max_llm_calls: 14,
            max_tool_calls: 24,
            max_validation_rounds: 3,
            max_wall_clock_ms: 180_000,
        },
        BudgetTier::Small => ExecutionBudget {
            max_steps: 16,
            max_tokens: 0,
            max_llm_calls: 14,
            max_tool_calls: 14,
            max_validation_rounds: 3,
            max_wall_clock_ms: 180_000,
        },
        BudgetTier::Standard => ExecutionBudget {
            max_steps: 24,
            max_tokens: 0,
            max_llm_calls: 18,
            max_tool_calls: 24,
            max_validation_rounds: 3,
            max_wall_clock_ms: 600_000,
        },
        BudgetTier::Extended => ExecutionBudget {
            max_steps: 16,
            max_tokens: 0,
            max_llm_calls: 24,
            max_tool_calls: 18,
            max_validation_rounds: 5,
            max_wall_clock_ms: 1_800_000,
        },
    }
}

pub fn select_initial_execution_budget(
    user_text: &str,
    turn_context: &TurnContext,
    depth: usize,
    role: AgentRole,
) -> (BudgetTier, &'static str, ExecutionBudget) {
    fn promote_contextual_followup_budget(
        tier: BudgetTier,
        route_kind: &'static str,
        turn_context: &TurnContext,
    ) -> (BudgetTier, &'static str, ExecutionBudget) {
        let carries_followup_context = matches!(
            turn_context.followup_mode,
            Some(FollowupMode::Followup | FollowupMode::ClarificationAnswer)
        ) && !turn_context.recent_messages.is_empty();

        if carries_followup_context && matches!(tier, BudgetTier::None | BudgetTier::Small) {
            let promoted = BudgetTier::Standard;
            return (
                promoted,
                "contextual_followup",
                default_execution_budget(promoted),
            );
        }

        (tier, route_kind, default_execution_budget(tier))
    }

    let lower = user_text.trim().to_ascii_lowercase();
    let authoring_only_content = turn_context
        .completion_contract
        .connected_content_mode
        .is_authoring_only();
    let auth_or_integration_management =
        crate::agent::intent_routing::user_text_requests_auth_or_integration_management(user_text);
    if depth > 0 || matches!(role, AgentRole::Executor | AgentRole::TaskLead) {
        let tier = BudgetTier::Extended;
        return (tier, "delegated_multi_step", default_execution_budget(tier));
    }

    let has_scoped_target = turn_context.primary_project_scope.is_some()
        || turn_context
            .completion_contract
            .primary_target_hint()
            .is_some()
        || lower.contains('/')
        || lower.contains(".rs")
        || lower.contains(".md")
        || lower.contains(".toml");

    let has_scheduled_action = [
        "schedule",
        "scheduled",
        "cron",
        "every day",
        "every week",
        "every month",
        "tomorrow",
        "next week",
        "daily",
        "weekly",
        "monthly",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));

    let has_deployment_or_external_write = auth_or_integration_management
        || (!authoring_only_content
            && [
                "deploy", "publish", "release", "restart", "schedule", "webhook", "post", "put",
                "patch", "delete", "send",
            ]
            .iter()
            .any(|kw| contains_keyword_as_words(&lower, kw)));

    let has_mutation_request = auth_or_integration_management
        || (!authoring_only_content
            && [
                "edit",
                "write",
                "update",
                "change",
                "fix",
                "implement",
                "refactor",
                "create",
                "add",
                "remove",
                "delete",
                "rename",
                "remember",
                "commit",
                "deploy",
                "restart",
                "send",
                "schedule",
            ]
            .iter()
            .any(|kw| contains_keyword_as_words(&lower, kw)));

    let has_read_only_api_lookup = contains_keyword_as_words(&lower, "api")
        && !has_mutation_request
        && !has_deployment_or_external_write
        && !has_scheduled_action;

    let has_read_only_investigation = turn_context.completion_contract.requires_observation
        || has_read_only_api_lookup
        || [
            "inspect",
            "check",
            "verify",
            "read",
            "search",
            "find",
            "list",
            "show",
            "look up",
            "investigate",
            "diagnose",
            "status",
            "logs",
        ]
        .iter()
        .any(|kw| contains_keyword_as_words(&lower, kw))
            && !has_mutation_request
            && !has_deployment_or_external_write;

    if has_scheduled_action {
        let tier = BudgetTier::Standard;
        return promote_contextual_followup_budget(tier, "scheduled_action", turn_context);
    }

    if has_deployment_or_external_write {
        let tier = BudgetTier::Standard;
        return promote_contextual_followup_budget(
            tier,
            "deployment_or_external_write",
            turn_context,
        );
    }

    if has_mutation_request {
        let tier = if has_scoped_target {
            BudgetTier::Small
        } else {
            BudgetTier::Standard
        };
        let route_kind = if has_scoped_target {
            "scoped_modification"
        } else {
            "unscoped_modification"
        };
        return promote_contextual_followup_budget(tier, route_kind, turn_context);
    }

    if has_read_only_api_lookup {
        let tier = BudgetTier::Standard;
        return promote_contextual_followup_budget(tier, "api_lookup", turn_context);
    }

    if has_read_only_investigation {
        let tier = BudgetTier::None;
        return promote_contextual_followup_budget(tier, "read_only_investigation", turn_context);
    }

    let tier = BudgetTier::None;
    let route_kind = if turn_context.completion_contract.requires_observation {
        "read_only_investigation"
    } else {
        "knowledge"
    };
    promote_contextual_followup_budget(tier, route_kind, turn_context)
}

#[allow(clippy::too_many_arguments)]
pub fn compile_step_execution_plan(
    execution_id: &str,
    plan_version: u32,
    iteration: usize,
    tool_call_id: &str,
    tool_name: &str,
    effective_arguments: &str,
    semantics: &ToolCallSemantics,
    capabilities: ToolCapabilities,
    allowed_project_scope: Option<&str>,
) -> StepExecutionPlan {
    let expected_targets = if semantics.target_hints.is_empty() {
        extract_target_hints_from_arguments(effective_arguments)
    } else {
        semantics.target_hints.clone()
    };

    let scope_applies_to_expected_targets = expected_targets.is_empty()
        || expected_targets.iter().all(|target| {
            matches!(
                target.kind,
                ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope
            )
        });

    let allowed_targets =
        if let Some(scope) = allowed_project_scope.filter(|_| scope_applies_to_expected_targets) {
            ToolTargetHint::new(ToolTargetHintKind::ProjectScope, scope)
                .into_iter()
                .collect()
        } else {
            expected_targets.clone()
        };

    let target_label = expected_targets
        .first()
        .map(|target| target.value.as_str())
        .or_else(|| allowed_targets.first().map(|target| target.value.as_str()))
        .unwrap_or("the requested target");

    let approval_requirement = if capabilities.needs_approval || capabilities.high_impact_write {
        ApprovalRequirement::Required {
            reason: format!("{} is approval-gated or high impact", tool_name),
        }
    } else {
        ApprovalRequirement::NotNeeded
    };

    let needs_idempotency =
        semantics.mutates_state() || capabilities.external_side_effect || !capabilities.idempotent;

    StepExecutionPlan {
        step_id: format!("step-{iteration}-{tool_call_id}"),
        description: format!("Run `{}` against {}", tool_name, target_label),
        plan_version: plan_version.max(1),
        primary_tool: Some(tool_name.to_string()),
        expected_effect: semantics.effect,
        target_scope: TargetScope {
            allowed_targets,
            hard_fail_outside_scope: semantics.mutates_state()
                || (capabilities.external_side_effect && scope_applies_to_expected_targets),
        },
        expected_targets,
        retry_policy: RetryPolicy {
            max_attempts: if capabilities.idempotent { 2 } else { 1 },
            allow_tool_invocation_retry: capabilities.idempotent,
        },
        approval_requirement,
        idempotency_key: needs_idempotency.then(|| {
            format!(
                "exec:{}:{}:{}:{}",
                execution_id, iteration, tool_name, tool_call_id
            )
        }),
    }
}

pub fn classify_step_execution_outcome(
    is_error: bool,
    background_detached: bool,
) -> StepExecutionOutcome {
    if background_detached {
        StepExecutionOutcome::BackgroundDetached
    } else if is_error {
        StepExecutionOutcome::RecoverableFailure
    } else {
        StepExecutionOutcome::Progress
    }
}

pub(crate) fn extract_target_hints_from_arguments(arguments: &str) -> Vec<ToolTargetHint> {
    let parsed = match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(serde_json::Value::Object(map)) => map,
        _ => return Vec::new(),
    };

    let mut targets = Vec::new();
    for key in [
        "path",
        "file_path",
        "file",
        "filename",
        "project_path",
        "project_dir",
        "repo_path",
        "repo_dir",
        "working_dir",
        "directory",
        "dir",
        "url",
        "target",
        "target_url",
    ] {
        let Some(value) = parsed.get(key).and_then(|value| value.as_str()) else {
            continue;
        };
        let candidate = match key {
            "url" | "target_url" => ToolTargetHint::new(ToolTargetHintKind::Url, value),
            "project_path" | "project_dir" | "repo_path" | "repo_dir" => {
                ToolTargetHint::new(ToolTargetHintKind::ProjectScope, value)
            }
            _ => ToolTargetHint::new(ToolTargetHintKind::Path, value),
        };
        if let Some(candidate) = candidate {
            if !targets.iter().any(|existing| existing == &candidate) {
                targets.push(candidate);
            }
        }
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{CompletionContract, CompletionTaskKind, TurnContext};
    use crate::traits::{ToolCallSemantics, ToolTargetHintKind};
    use serde_json::json;

    #[test]
    fn scoped_edit_requests_start_with_small_budget() {
        let turn_context = TurnContext {
            primary_project_scope: Some("/tmp/demo".to_string()),
            ..TurnContext::default()
        };
        let (tier, route_kind, budget) = select_initial_execution_budget(
            "edit /tmp/demo/src/main.rs",
            &turn_context,
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(tier, BudgetTier::Small);
        assert_eq!(route_kind, "scoped_modification");
        assert_eq!(budget.max_validation_rounds, 3);
    }

    #[test]
    fn delegated_work_starts_with_extended_budget() {
        let (tier, route_kind, budget) = select_initial_execution_budget(
            "fix the deployment",
            &TurnContext::default(),
            1,
            AgentRole::Executor,
        );
        assert_eq!(tier, BudgetTier::Extended);
        assert_eq!(route_kind, "delegated_multi_step");
        assert!(budget.max_tool_calls >= 16);
    }

    #[test]
    fn compile_step_plan_uses_scope_and_idempotency_for_mutations() {
        let semantics =
            ToolCallSemantics::mutation().with_target_hint(ToolTargetHintKind::Path, "src/main.rs");
        let plan = compile_step_execution_plan(
            "exec-1",
            3,
            2,
            "call-1",
            "edit_file",
            r#"{"path":"src/main.rs"}"#,
            &semantics,
            ToolCapabilities {
                read_only: false,
                external_side_effect: false,
                needs_approval: true,
                idempotent: false,
                high_impact_write: false,
            },
            Some("/repo"),
        );

        assert_eq!(plan.primary_tool.as_deref(), Some("edit_file"));
        assert_eq!(plan.plan_version, 3);
        assert_eq!(plan.target_scope.allowed_targets.len(), 1);
        assert_eq!(
            plan.target_scope.allowed_targets[0].kind,
            ToolTargetHintKind::ProjectScope
        );
        assert!(plan.target_scope.hard_fail_outside_scope);
        assert!(plan.idempotency_key.is_some());
        assert!(matches!(
            plan.approval_requirement,
            ApprovalRequirement::Required { .. }
        ));
    }

    #[test]
    fn compile_step_plan_preserves_url_targets_when_project_scope_exists() {
        let semantics = ToolCallSemantics::observation().with_target_hint(
            ToolTargetHintKind::Url,
            "https://clinicaltrials.gov/api/v2/studies",
        );
        let plan = compile_step_execution_plan(
            "exec-1",
            3,
            2,
            "call-1",
            "http_request",
            r#"{"url":"https://clinicaltrials.gov/api/v2/studies"}"#,
            &semantics,
            ToolCapabilities {
                read_only: true,
                external_side_effect: true,
                needs_approval: false,
                idempotent: true,
                high_impact_write: false,
            },
            Some("/repo"),
        );

        assert_eq!(plan.target_scope.allowed_targets.len(), 1);
        assert_eq!(
            plan.target_scope.allowed_targets[0].kind,
            ToolTargetHintKind::Url
        );
        assert_eq!(
            plan.target_scope.allowed_targets[0].value,
            "https://clinicaltrials.gov/api/v2/studies"
        );
    }

    #[test]
    fn execution_state_reports_budget_exhaustion() {
        let mut state = ExecutionState::new(
            BudgetTier::Small,
            ExecutionBudget {
                max_steps: 1,
                max_tokens: 100,
                max_llm_calls: 1,
                max_tool_calls: 1,
                max_validation_rounds: 1,
                max_wall_clock_ms: 1_000,
            },
            ExecutionPersistence::Ephemeral,
        );
        state.activate_budget_envelope(0, Duration::from_millis(0));
        state.record_llm_call();
        assert_eq!(
            state.exhausted_limit(0, Duration::from_millis(1)),
            Some(ExecutionBudgetLimit::LlmCalls)
        );
    }

    #[test]
    fn inactive_execution_budget_ignores_plain_text_token_usage() {
        let state = ExecutionState::new(
            BudgetTier::None,
            ExecutionBudget {
                max_steps: 24,
                max_tokens: 10,
                max_llm_calls: 1,
                max_tool_calls: 1,
                max_validation_rounds: 1,
                max_wall_clock_ms: 1,
            },
            ExecutionPersistence::Ephemeral,
        );

        assert_eq!(state.exhausted_limit(10_000, Duration::from_secs(30)), None);
    }

    #[test]
    fn knowledge_turns_use_none_budget() {
        let turn_context = TurnContext {
            completion_contract: CompletionContract::default(),
            ..TurnContext::default()
        };
        let (tier, route_kind, _) = select_initial_execution_budget(
            "what's the capital of france",
            &turn_context,
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(tier, BudgetTier::None);
        assert_eq!(route_kind, "knowledge");
    }

    #[test]
    fn scheduled_turns_use_standard_budget() {
        let (tier, route_kind, budget) = select_initial_execution_budget(
            "schedule a daily health check",
            &TurnContext::default(),
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(tier, BudgetTier::Standard);
        assert_eq!(route_kind, "scheduled_action");
        assert!(budget.max_validation_rounds >= 3);
    }

    #[test]
    fn read_only_investigation_uses_none_budget() {
        let (tier, route_kind, _) = select_initial_execution_budget(
            "inspect the latest logs and show me the current status",
            &TurnContext::default(),
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(tier, BudgetTier::None);
        assert_eq!(route_kind, "read_only_investigation");
    }

    #[test]
    fn api_read_requests_use_standard_budget_for_multi_step_lookups() {
        let (tier, route_kind, budget) = select_initial_execution_budget(
            "Using the clinical trials API, give me studies near Fairfax for skin cancer.",
            &TurnContext::default(),
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(tier, BudgetTier::Standard);
        assert_eq!(route_kind, "api_lookup");
        assert!(budget.max_llm_calls >= 18);
        assert_eq!(budget.max_tokens, 0);
    }

    #[test]
    fn connected_content_authoring_requests_stay_in_knowledge_lane() {
        let mut turn_context = TurnContext::default();
        turn_context.completion_contract.connected_content_mode =
            crate::agent::intent_routing::ConnectedContentMode::DraftThenDeliver;
        turn_context.completion_contract.task_kind = CompletionTaskKind::Deliver;
        turn_context.completion_contract.expects_mutation = true;
        let (tier, route_kind, _) = select_initial_execution_budget(
            "Can you post a tweet about your new stuff and make it engaging so people want to comment?",
            &turn_context,
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(tier, BudgetTier::Standard);
        assert_eq!(route_kind, "deployment_or_external_write");
    }

    #[test]
    fn account_scoped_connected_content_delivery_uses_external_write_budget() {
        let (tier, route_kind, budget) = select_initial_execution_budget(
            "Can you post a tweet on your account?",
            &TurnContext::default(),
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(tier, BudgetTier::Standard);
        assert_eq!(route_kind, "deployment_or_external_write");
        assert!(budget.max_llm_calls >= 18);
    }

    #[test]
    fn auth_management_requests_use_standard_budget() {
        let (tier, route_kind, _) = select_initial_execution_budget(
            "Reconnect my Twitter OAuth account so you can post for me.",
            &TurnContext::default(),
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(tier, BudgetTier::Standard);
        assert_eq!(route_kind, "deployment_or_external_write");
    }

    #[test]
    fn contextual_followups_start_with_standard_budget() {
        let turn_context = TurnContext {
            followup_mode: Some(FollowupMode::Followup),
            recent_messages: vec![json!({
                "role": "assistant",
                "content": "Here are 20 matching studies with short summaries."
            })],
            ..TurnContext::default()
        };
        let (tier, route_kind, budget) = select_initial_execution_budget(
            "Which one is most relevant to skin cancer?",
            &turn_context,
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(tier, BudgetTier::Standard);
        assert_eq!(route_kind, "contextual_followup");
        assert_eq!(budget.max_tokens, 0);
    }

    #[test]
    fn clarification_followups_promote_scoped_edits_to_standard_budget() {
        let turn_context = TurnContext {
            primary_project_scope: Some("/tmp/demo".to_string()),
            followup_mode: Some(FollowupMode::ClarificationAnswer),
            recent_messages: vec![json!({
                "role": "assistant",
                "content": "Which file should I update?"
            })],
            ..TurnContext::default()
        };
        let (tier, route_kind, budget) = select_initial_execution_budget(
            "Update the config in src/main.rs",
            &turn_context,
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(tier, BudgetTier::Standard);
        assert_eq!(route_kind, "contextual_followup");
        assert!(budget.max_validation_rounds >= 3);
    }

    #[test]
    fn extend_budget_on_progress_increases_limits() {
        let mut state = ExecutionState::new(
            BudgetTier::None,
            default_execution_budget(BudgetTier::None),
            ExecutionPersistence::Ephemeral,
        );
        let original_llm = state.budget.max_llm_calls;
        let original_tools = state.budget.max_tool_calls;
        let original_steps = state.budget.max_steps;
        let original_wall = state.budget.max_wall_clock_ms;

        // No extension when budget envelope is inactive
        state.extend_budget_on_progress();
        assert_eq!(state.budget.max_llm_calls, original_llm);
        assert_eq!(state.budget.max_wall_clock_ms, original_wall);

        // Extension kicks in once the envelope is active
        state.activate_budget_envelope(0, Duration::from_millis(0));
        state.extend_budget_on_progress();
        assert!(state.budget.max_llm_calls > original_llm);
        assert!(state.budget.max_tool_calls > original_tools);
        assert!(state.budget.max_steps > original_steps);
        assert!(state.budget.max_wall_clock_ms > original_wall);

        // Cumulative extensions keep growing
        let after_first = state.budget.max_llm_calls;
        let after_first_wall = state.budget.max_wall_clock_ms;
        state.extend_budget_on_progress();
        assert!(state.budget.max_llm_calls > after_first);
        assert!(state.budget.max_wall_clock_ms > after_first_wall);
    }

    #[test]
    fn productive_run_never_exhausts_budget() {
        let mut state = ExecutionState::new(
            BudgetTier::None,
            default_execution_budget(BudgetTier::None),
            ExecutionPersistence::Ephemeral,
        );
        state.activate_budget_envelope(0, Duration::from_millis(0));

        // Simulate 30 productive iterations: each records an LLM call + tool
        // call but also extends via progress.  Use realistic elapsed time
        // (~10s per iteration → 300s total) to verify wall-clock extension
        // keeps pace with real-world execution.
        for _ in 0..30 {
            state.record_llm_call();
            state.record_tool_call();
            state.extend_budget_on_progress();
        }

        // 30 iterations × ~10s each = 300s of wall time.  The base budget
        // for None tier is 180s, but 30 progress extensions add 30 × 30s =
        // 900s, giving a total wall-clock budget of 1080s — well above 300s.
        let realistic_elapsed = Duration::from_secs(300);
        assert_eq!(
            state.exhausted_limit(0, realistic_elapsed),
            None,
            "Productive run should never exhaust budget, even with realistic wall-clock time"
        );
    }
}
