use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::agent::TurnContext;
use crate::traits::{AgentRole, ToolCallEffect, ToolTargetHint};

/// Structured record of a single tool call attempt outcome.
/// Accumulated in the outcome ledger for reconciliation.
/// Note: this tracks *attempts*, not intended actions. A retry that
/// succeeds after a failure produces two entries, not one.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutcomeEntry {
    pub tool_name: String,
    pub success: bool,
    pub http_status: Option<u16>,
    /// True only when the tool has `external_side_effect` capability AND
    /// semantics indicate state mutation. Terminal-based curl calls are
    /// NOT tracked (no structured status code).
    pub is_external_mutation: bool,
    pub error_summary: Option<String>,
    pub iteration: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planned_step_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planned_step_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planned_step_description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_step_count: Option<usize>,
}

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct TargetScope {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_targets: Vec<ToolTargetHint>,
    #[serde(default)]
    pub hard_fail_outside_scope: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Maximum model/tool-call proposals for the stable operation. Unlike
    /// `max_attempts`, this includes pre-I/O validation outcomes.
    #[serde(default)]
    pub max_invocations: usize,
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
    /// Stable identity of the user-owned operation. Unlike `step_id`, this
    /// survives model retries, new tool-call IDs, and loop iterations.
    #[serde(default)]
    pub operation_key: String,
    pub description: String,
    pub plan_version: u32,
    pub primary_tool: Option<String>,
    pub expected_effect: ToolCallEffect,
    pub target_scope: TargetScope,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_targets: Vec<ToolTargetHint>,
    pub retry_policy: RetryPolicy,
    /// Optional user-owned obligation cardinality. This is intentionally
    /// separate from `operation_key`: alternative concrete strategies have
    /// different operation identities but still share an explicit global
    /// "invoke at most N times" constraint for the obligation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cardinality_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cardinality_limit: Option<usize>,
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
    /// Attempts charged per stable operation. This is the authoritative retry
    /// ledger; loop iterations and model-generated call IDs are not attempts.
    #[serde(default)]
    pub operation_attempts: BTreeMap<String, usize>,
    /// Tool-call proposals charged per stable user-owned operation. This is
    /// distinct from dispatched attempts so an explicit no-retry constraint
    /// also applies after a pre-I/O rejection.
    #[serde(default)]
    pub operation_invocations: BTreeMap<String, usize>,
    /// Proposals charged to explicit request-level cardinality constraints.
    /// This ledger is never used for ordinary retry/idempotency decisions.
    #[serde(default)]
    pub obligation_invocations: BTreeMap<String, usize>,
    /// Actual adapter dispatches grouped by effective planned tool. This is
    /// authoritative response/telemetry data and must not be reconstructed
    /// from model prose or proposal counts.
    #[serde(default)]
    pub tool_dispatches: BTreeMap<String, usize>,
    /// Monotonic revision of potentially changed domain state. Observation
    /// operation identities bind to this revision so a check can run again
    /// after a different mutation, while a mutation cannot refresh its own
    /// retry key merely by having executed.
    #[serde(default)]
    pub effect_revision: u64,
    #[serde(default)]
    pub operation_effect_epochs: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_mutation_operation_base: Option<String>,
    /// Prepared proposals that reached a real or durable-replay result.
    #[serde(default)]
    pub operation_results_observed: usize,
    /// Operation results whose typed domain outcome satisfied the request.
    #[serde(default)]
    pub completed_operation_results: usize,
    /// Proposals rejected before dispatch by a typed policy/capability
    /// boundary (scope lock, controller contract, protected host data). These
    /// are terminal observations of the request's authority, not retryable
    /// adapter failures.

    /// The runtime projected the exact persisted antecedent conversation
    /// context into this turn's prompt (typed `required_context_projected`).
    /// That projection IS the canonical retained history for an
    /// exact-history obligation; no history tool lookup can add to it.
    #[serde(default)]
    pub exact_context_projected: bool,
    /// Distinct outcome-bearing tool results that have earned additional
    /// execution runway. Persisting the keys prevents a resumed task from
    /// receiving fresh capacity for replaying evidence it already observed.
    #[serde(default)]
    pub progress_credit_keys: BTreeSet<String>,
    /// Total bounded runway extensions granted for this execution.
    #[serde(default)]
    pub progress_extensions_used: usize,
    /// Observation-only extensions are capped separately so a sequence of
    /// successful reads cannot crowd out execution and verification work.
    #[serde(default)]
    pub observation_extensions_used: usize,
    /// One-shot guard for the pre-exhaustion resource adaptation directive.
    #[serde(default)]
    pub resource_pressure_emitted: bool,
    /// Structured outcome ledger — one entry per tool call attempt.
    #[serde(default)]
    pub outcome_ledger: Vec<OutcomeEntry>,
    /// Typed targets whose prerequisite mutation was rejected before I/O.
    /// Later mutation proposals that overlap one of these targets are blocked
    /// before dispatch; a dependent operation cannot run on an unproven
    /// prerequisite merely because a new model turn proposed it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked_dependency_targets: Vec<ToolTargetHint>,
    /// Cumulative milliseconds lost to LLM provider timeouts.  These are NOT
    /// the agent's fault (provider slowness, not agent stalling) and should be
    /// excluded from the wall-clock budget check.
    #[serde(default)]
    pub provider_timeout_ms: u64,
    /// Raw tool output accumulated this turn, used by the answer-grounding
    /// gate to verify enumerated list entities were actually observed.
    /// In-memory only — never serialized into events or persisted state.
    #[serde(skip)]
    pub tool_output_evidence: String,
    /// True once the evidence buffer hit its cap. The grounding gate must
    /// skip when set — missing evidence would read as fabrication.
    #[serde(skip)]
    pub tool_output_evidence_overflow: bool,
    /// True once web_search was called this turn — marks the turn as web
    /// research for the corroboration gate.
    #[serde(skip)]
    pub web_search_used: bool,
    /// Distinct domains of source pages successfully READ this turn
    /// (web_fetch/browser with substantive content). Snippets don't count.
    #[serde(skip)]
    pub web_source_domains: std::collections::HashSet<String>,
    /// Exact successfully-read source URLs. Used to require direct citations
    /// when the user asks for a concrete number of sources.
    #[serde(skip)]
    pub web_source_urls: std::collections::HashSet<String>,
    /// True when an operational tool result should be communicated at the
    /// owner's level rather than copied as control-plane bookkeeping.
    #[serde(skip)]
    pub natural_outcome_summary_required: bool,
    /// Exact internal IDs observed during a natural-summary turn. In-memory
    /// only; durable diagnostic records remain authoritative.
    #[serde(skip)]
    pub internal_identifiers: BTreeSet<String>,
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
            operation_attempts: BTreeMap::new(),
            operation_invocations: BTreeMap::new(),
            obligation_invocations: BTreeMap::new(),
            tool_dispatches: BTreeMap::new(),
            effect_revision: 0,
            operation_effect_epochs: BTreeMap::new(),
            last_mutation_operation_base: None,
            operation_results_observed: 0,
            completed_operation_results: 0,
            exact_context_projected: false,
            progress_credit_keys: BTreeSet::new(),
            progress_extensions_used: 0,
            observation_extensions_used: 0,
            resource_pressure_emitted: false,
            outcome_ledger: Vec::new(),
            blocked_dependency_targets: Vec::new(),
            provider_timeout_ms: 0,
            tool_output_evidence: String::new(),
            tool_output_evidence_overflow: false,
            web_search_used: false,
            web_source_domains: std::collections::HashSet::new(),
            web_source_urls: std::collections::HashSet::new(),
            natural_outcome_summary_required: false,
            internal_identifiers: BTreeSet::new(),
        }
    }

    pub fn mark_persisted_now(&mut self) {
        self.persisted_at = Some(Utc::now());
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
            max_wall_clock_ms: 300_000,
        },
        BudgetTier::Small => ExecutionBudget {
            max_steps: 16,
            max_tokens: 0,
            max_llm_calls: 14,
            max_tool_calls: 14,
            max_validation_rounds: 3,
            max_wall_clock_ms: 300_000,
        },
        BudgetTier::Standard => ExecutionBudget {
            max_steps: 24,
            max_tokens: 0,
            max_llm_calls: 18,
            max_tool_calls: 24,
            max_validation_rounds: 3,
            max_wall_clock_ms: 900_000,
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
    _user_text: &str,
    _turn_context: &TurnContext,
    depth: usize,
    role: AgentRole,
) -> (BudgetTier, &'static str, ExecutionBudget) {
    if depth > 0 || matches!(role, AgentRole::Executor | AgentRole::TaskLead) {
        let tier = BudgetTier::Extended;
        return (tier, "delegated_multi_step", default_execution_budget(tier));
    }

    // Top-level turns receive the same useful execution headroom regardless of
    // wording. The model chooses whether and how to use tools; lexical cues such
    // as "schedule", file suffixes, or API names must not change orchestration.
    let tier = BudgetTier::Standard;
    (tier, "model_directed", default_execution_budget(tier))
}

#[cfg(test)]
#[path = "execution_state_tests.rs"]
mod tests;
