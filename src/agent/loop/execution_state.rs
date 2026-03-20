use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::agent::{contains_keyword_as_words, FollowupMode, TurnContext};
use crate::traits::{
    AgentRole, ToolCallEffect, ToolCallSemantics, ToolCapabilities, ToolTargetHint,
    ToolTargetHintKind,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReconciliationMode {
    AttemptLevel,
    PlannedStepLevel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReconciliationOverview {
    pub mode: ReconciliationMode,
    pub total: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub failed_step_indices: Vec<usize>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LinearIntentStep {
    pub step_id: String,
    pub step_index: usize, // 1-based
    pub tool: String,
    pub target: String,
    pub description: String,
    /// Number of tool calls made while this step was current.
    #[serde(default)]
    pub tool_calls_on_step: usize,
    /// Whether this step has been marked complete (by re-planner or outcome ledger).
    #[serde(default)]
    pub completed: bool,
    /// Evidence summary when step was marked complete.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_evidence: Option<String>,
    /// Tool call count at which the re-planner last evaluated this step.
    /// Prevents re-triggering on every single round after threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_evaluated_at: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct LinearIntentPlan {
    pub plan_version: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<LinearIntentStep>,
    #[serde(default)]
    pub current_step_cursor: usize, // 0-based cursor into `steps`
}

impl LinearIntentPlan {
    /// Record tool calls on the current step.
    pub fn record_tool_calls_on_current(&mut self, count: usize) {
        if let Some(step) = self.steps.get_mut(self.current_step_cursor) {
            step.tool_calls_on_step += count;
        }
    }

    /// Check if re-planner should fire for the current step.
    /// Triggers every REPLAN_INTERVAL tool calls (not every single call).
    pub fn current_step_needs_replan(&self) -> bool {
        const REPLAN_INTERVAL: usize = 2;
        let Some(step) = self.steps.get(self.current_step_cursor) else {
            return false;
        };
        if step.completed || step.tool_calls_on_step < 2 {
            return false;
        }
        let last_eval = step.last_evaluated_at.unwrap_or(0);
        step.tool_calls_on_step >= last_eval + REPLAN_INTERVAL
    }

    /// Mark the current step as evaluated at the current tool call count.
    pub fn mark_current_step_evaluated(&mut self) {
        if let Some(step) = self.steps.get_mut(self.current_step_cursor) {
            step.last_evaluated_at = Some(step.tool_calls_on_step);
        }
    }

    /// Mark the current step as complete and advance the cursor.
    pub fn complete_current_step_with_evidence(&mut self, evidence: String) {
        if let Some(step) = self.steps.get_mut(self.current_step_cursor) {
            step.completed = true;
            step.completion_evidence = Some(evidence);
        }
        if self.current_step_cursor < self.steps.len() {
            self.current_step_cursor += 1;
        }
    }

    /// Check if all steps are complete (cursor past last step).
    #[allow(dead_code)]
    pub fn all_steps_complete(&self) -> bool {
        self.current_step_cursor >= self.steps.len()
    }

    /// Format plan with progress markers for LLM context injection.
    pub fn format_with_progress(&self) -> String {
        let mut result = String::from("## Task Plan\n");
        for step in &self.steps {
            let marker = if step.completed {
                "[DONE]"
            } else if step.step_index == self.current_step_cursor + 1 {
                "[CURRENT]"
            } else {
                ""
            };

            if marker.is_empty() {
                result.push_str(&format!("{}. {}\n", step.step_index, step.description));
            } else {
                result.push_str(&format!(
                    "{}. {} {}",
                    step.step_index, marker, step.description
                ));
                if let Some(ref evidence) = step.completion_evidence {
                    let ev = crate::utils::truncate_str(evidence, 100);
                    result.push_str(&format!(" \u{2014} {}", ev));
                }
                result.push('\n');
            }
        }
        if result.len() > 2000 {
            result.truncate(crate::utils::floor_char_boundary(&result, 2000));
            result.push_str("\n...");
        }
        result
    }
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
    /// Structured outcome ledger — one entry per tool call attempt.
    #[serde(default)]
    pub outcome_ledger: Vec<OutcomeEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_linear_intent_plan: Option<LinearIntentPlan>,
    /// Cumulative milliseconds lost to LLM provider timeouts.  These are NOT
    /// the agent's fault (provider slowness, not agent stalling) and should be
    /// excluded from the wall-clock budget check.
    #[serde(default)]
    pub provider_timeout_ms: u64,
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
            outcome_ledger: Vec::new(),
            active_linear_intent_plan: None,
            provider_timeout_ms: 0,
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

    pub fn record_outcome(&mut self, entry: OutcomeEntry) {
        self.outcome_ledger.push(entry);
    }

    pub fn install_linear_intent_plan(&mut self, plan_version: u32, steps: Vec<LinearIntentStep>) {
        if steps.is_empty() {
            self.active_linear_intent_plan = None;
            return;
        }
        self.active_linear_intent_plan = Some(LinearIntentPlan {
            plan_version,
            steps,
            current_step_cursor: 0,
        });
    }

    pub fn current_linear_intent_step(&self) -> Option<&LinearIntentStep> {
        let plan = self.active_linear_intent_plan.as_ref()?;
        plan.steps.get(plan.current_step_cursor)
    }

    pub fn advance_linear_intent_step_after_external_success(&mut self) {
        let Some(plan) = self.active_linear_intent_plan.as_mut() else {
            return;
        };
        // Allow cursor to advance past the last step (cursor == len means all done).
        // current_linear_intent_step() will return None once cursor >= len.
        if plan.current_step_cursor < plan.steps.len() {
            plan.current_step_cursor += 1;
        }
    }

    pub fn successful_external_mutation_count(&self) -> usize {
        self.outcome_ledger
            .iter()
            .filter(|e| e.is_external_mutation && e.success)
            .count()
    }

    /// Returns failures that have NOT been corrected by a later success.
    ///
    /// A failure is "corrected" if:
    /// 1. There is a later success for the same tool_name (direct retry), OR
    /// 2. ALL remaining failures occurred before the latest success of ANY tool
    ///    (the agent recovered overall — handles tool switching, e.g. run_command→terminal).
    ///
    /// Also filters out failures with no error summary (unknown errors / false positives).
    pub fn uncorrected_failed_mutations(&self) -> Vec<&OutcomeEntry> {
        let failures: Vec<&OutcomeEntry> = self
            .outcome_ledger
            .iter()
            .filter(|e| e.is_external_mutation && !e.success)
            .collect();

        if failures.is_empty() {
            return Vec::new();
        }

        // Filter out failures with no error summary (unknown errors / false positives)
        let failures_with_summary: Vec<&OutcomeEntry> = failures
            .into_iter()
            .filter(|e| e.error_summary.is_some())
            .collect();

        if failures_with_summary.is_empty() {
            return Vec::new();
        }

        // Step 1: Remove failures corrected by a later success of the same tool
        let uncorrected: Vec<&OutcomeEntry> = failures_with_summary
            .into_iter()
            .filter(|fail| {
                !self.outcome_ledger.iter().any(|e| {
                    e.is_external_mutation
                        && e.success
                        && e.tool_name == fail.tool_name
                        && e.iteration > fail.iteration
                })
            })
            .collect();

        if uncorrected.is_empty() {
            return Vec::new();
        }

        // Step 2: If ALL remaining failures occurred before the latest success
        // (any tool), the agent recovered overall. This handles tool switching
        // (e.g., run_command fails, bot switches to terminal and succeeds).
        let max_success_iter = self
            .outcome_ledger
            .iter()
            .filter(|e| e.is_external_mutation && e.success)
            .map(|e| e.iteration)
            .max();

        if let Some(max_success) = max_success_iter {
            let still_uncorrected: Vec<&OutcomeEntry> = uncorrected
                .into_iter()
                .filter(|e| e.iteration > max_success)
                .collect();
            return still_uncorrected;
        }

        uncorrected
    }

    /// True if there are external mutation failures that were NOT corrected
    /// by later successes.
    pub fn has_uncorrected_failed_external_mutations(&self) -> bool {
        !self.uncorrected_failed_mutations().is_empty()
    }

    pub(crate) fn build_attempt_reconciliation_overview(&self) -> Option<ReconciliationOverview> {
        let uncorrected = self.uncorrected_failed_mutations();
        if uncorrected.is_empty() {
            return None;
        }

        let total = self
            .outcome_ledger
            .iter()
            .filter(|e| e.is_external_mutation)
            .count();
        let succeeded = self.successful_external_mutation_count();
        let failed = uncorrected.len();

        let mut summary = format!(
            "[SYSTEM] External mutation attempt reconciliation: {} of {} attempts succeeded, {} failed.",
            succeeded, total, failed,
        );
        for entry in &uncorrected {
            let status = entry
                .http_status
                .map(|s| format!(" (HTTP {})", s))
                .unwrap_or_default();
            let error = entry.error_summary.as_deref().unwrap_or("unknown error");
            summary.push_str(&format!(
                "\n  - {} at iteration {}{}: {}",
                entry.tool_name, entry.iteration, status, error,
            ));
        }
        // NOTE: Do NOT embed meta-instructions in the summary — the LLM may parrot
        // them verbatim to the user.  The completion phase adds behavioural guidance
        // as a separate system message when needed.

        Some(ReconciliationOverview {
            mode: ReconciliationMode::AttemptLevel,
            total,
            succeeded,
            failed,
            failed_step_indices: Vec::new(),
            summary,
        })
    }

    /// Build a reconciliation summary of external mutation attempts.
    /// Returns None if all external mutations succeeded.
    pub fn build_attempt_reconciliation_summary(&self) -> Option<String> {
        self.build_attempt_reconciliation_overview()
            .map(|overview| overview.summary)
    }

    pub(crate) fn build_reconciliation_overview(&self) -> Option<ReconciliationOverview> {
        use std::collections::{BTreeMap, BTreeSet};

        let latest_plan_version = self
            .active_linear_intent_plan
            .as_ref()
            .map(|plan| plan.plan_version)
            .or_else(|| {
                self.outcome_ledger
                    .iter()
                    .filter(|entry| entry.is_external_mutation)
                    .filter_map(|entry| entry.plan_version)
                    .max()
            });

        let Some(latest_plan_version) = latest_plan_version else {
            return self.build_attempt_reconciliation_overview();
        };

        let planned_entries: Vec<&OutcomeEntry> = self
            .outcome_ledger
            .iter()
            .filter(|entry| {
                entry.is_external_mutation
                    && entry.planned_step_id.is_some()
                    && entry.plan_version == Some(latest_plan_version)
            })
            .collect();

        if planned_entries.is_empty() {
            return self.build_attempt_reconciliation_overview();
        }

        let active_plan = self
            .active_linear_intent_plan
            .as_ref()
            .filter(|plan| plan.plan_version == latest_plan_version);

        let mut by_step: BTreeMap<(usize, String), Vec<&OutcomeEntry>> = BTreeMap::new();
        for entry in planned_entries {
            let Some(step_id) = entry.planned_step_id.clone() else {
                continue;
            };
            let step_index = entry.planned_step_index.unwrap_or(usize::MAX);
            by_step
                .entry((step_index, step_id))
                .or_default()
                .push(entry);
        }

        if by_step.is_empty() {
            return self.build_attempt_reconciliation_overview();
        }

        let expected_steps = active_plan
            .map(|plan| plan.steps.len())
            .or_else(|| {
                by_step
                    .values()
                    .flat_map(|entries| {
                        entries.iter().filter_map(|entry| entry.expected_step_count)
                    })
                    .max()
            })
            .unwrap_or(by_step.len());

        let succeeded = by_step
            .values()
            .filter(|entries| entries.iter().any(|entry| entry.success))
            .count();

        let mut failed_step_indices = BTreeSet::new();
        let mut summary = format!(
            "[SYSTEM] Planned-step reconciliation: {} of {} planned steps completed.",
            succeeded, expected_steps
        );

        if let Some(plan) = active_plan {
            for step in &plan.steps {
                let key = (step.step_index, step.step_id.clone());
                if let Some(entries) = by_step.get(&key) {
                    let attempts = entries.len();
                    let succeeded_this_step = entries.iter().any(|entry| entry.success);
                    let final_failure = entries.iter().rev().find(|entry| !entry.success);
                    if succeeded_this_step && attempts > 1 {
                        summary.push_str(&format!(
                            "\n  - Step {} ({}) succeeded after {} attempts.",
                            step.step_index, step.description, attempts
                        ));
                    } else if succeeded_this_step {
                        summary.push_str(&format!(
                            "\n  - Step {} ({}) succeeded.",
                            step.step_index, step.description
                        ));
                    } else if let Some(failure) = final_failure {
                        failed_step_indices.insert(step.step_index);
                        let status = failure
                            .http_status
                            .map(|code| format!(" (HTTP {})", code))
                            .unwrap_or_default();
                        let error = failure.error_summary.as_deref().unwrap_or("unknown error");
                        summary.push_str(&format!(
                            "\n  - Step {} ({}) failed after {} attempts{}: {}",
                            step.step_index, step.description, attempts, status, error
                        ));
                    } else {
                        failed_step_indices.insert(step.step_index);
                        summary.push_str(&format!(
                            "\n  - Step {} ({}) was not completed.",
                            step.step_index, step.description
                        ));
                    }
                } else {
                    failed_step_indices.insert(step.step_index);
                    summary.push_str(&format!(
                        "\n  - Step {} ({}) was not completed.",
                        step.step_index, step.description
                    ));
                }
            }
        } else {
            for ((step_index, _step_id), entries) in &by_step {
                let attempts = entries.len();
                let description = entries
                    .iter()
                    .find_map(|entry| entry.planned_step_description.as_deref())
                    .unwrap_or("unnamed step");
                let succeeded_this_step = entries.iter().any(|entry| entry.success);
                let final_failure = entries.iter().rev().find(|entry| !entry.success);

                if succeeded_this_step && attempts > 1 {
                    summary.push_str(&format!(
                        "\n  - Step {} ({}) succeeded after {} attempts.",
                        step_index, description, attempts
                    ));
                } else if succeeded_this_step {
                    summary.push_str(&format!(
                        "\n  - Step {} ({}) succeeded.",
                        step_index, description
                    ));
                } else if let Some(failure) = final_failure {
                    failed_step_indices.insert(*step_index);
                    let status = failure
                        .http_status
                        .map(|code| format!(" (HTTP {})", code))
                        .unwrap_or_default();
                    let error = failure.error_summary.as_deref().unwrap_or("unknown error");
                    summary.push_str(&format!(
                        "\n  - Step {} ({}) failed after {} attempts{}: {}",
                        step_index, description, attempts, status, error
                    ));
                }
            }

            for step_index in 1..=expected_steps {
                let seen = by_step.keys().any(|(idx, _)| *idx == step_index);
                if !seen {
                    failed_step_indices.insert(step_index);
                }
            }

            if expected_steps > by_step.len() {
                summary.push_str(&format!(
                    "\n  - {} planned step(s) were not completed.",
                    expected_steps - by_step.len()
                ));
            }
        }

        let failed = expected_steps.saturating_sub(succeeded);
        // NOTE: Do NOT embed meta-instructions in the summary — the LLM may parrot
        // them verbatim to the user.

        Some(ReconciliationOverview {
            mode: ReconciliationMode::PlannedStepLevel,
            total: expected_steps,
            succeeded,
            failed,
            failed_step_indices: failed_step_indices.into_iter().collect(),
            summary,
        })
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
        /// Validation round extension per successful tool call.
        /// Complex multi-step tasks legitimately trigger multiple
        /// completion-verification cycles while still making real
        /// progress (reading files, running builds, deploying).
        /// Without extending this limit, productive runs are stopped
        /// by the validation cap even though every other budget
        /// dimension has headroom.
        const VALIDATION_EXTENSION: usize = 1;
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
        if self.budget.max_validation_rounds > 0 {
            self.budget.max_validation_rounds = self
                .budget
                .max_validation_rounds
                .saturating_add(VALIDATION_EXTENSION);
        }
    }

    /// Promote budget if a captured task plan indicates higher complexity
    /// than the initial keyword-based selection predicted.
    /// Only promotes upward — never reduces an existing budget.
    pub fn promote_budget_for_plan(&mut self, step_count: usize) {
        if step_count < 3 {
            return;
        }
        if !matches!(self.budget_tier, BudgetTier::None | BudgetTier::Small) {
            return;
        }
        let promoted = default_execution_budget(BudgetTier::Standard);
        self.budget.max_llm_calls = self.budget.max_llm_calls.max(promoted.max_llm_calls);
        self.budget.max_tool_calls = self.budget.max_tool_calls.max(promoted.max_tool_calls);
        self.budget.max_steps = self.budget.max_steps.max(promoted.max_steps);
        self.budget.max_wall_clock_ms = self
            .budget
            .max_wall_clock_ms
            .max(promoted.max_wall_clock_ms);
        tracing::info!(
            step_count,
            new_max_tool_calls = self.budget.max_tool_calls,
            new_max_llm_calls = self.budget.max_llm_calls,
            "Budget promoted for multi-step task plan"
        );
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
        if self.budget.max_wall_clock_ms > 0 {
            // Subtract time lost to provider timeouts — those are external
            // delays, not agent stalling, and shouldn't penalise the budget.
            let effective_elapsed = execution_elapsed_ms.saturating_sub(self.provider_timeout_ms);
            if effective_elapsed >= self.budget.max_wall_clock_ms {
                return Some(ExecutionBudgetLimit::WallClock);
            }
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
        // Multi-step tasks that include both creation AND verification/testing
        // need more budget than simple scoped edits.
        let has_verification_step = [
            "test",
            "run",
            "verify",
            "execute",
            "demonstrate",
            "show me",
            "show the results",
            "check",
        ]
        .iter()
        .any(|kw| contains_keyword_as_words(&lower, kw));

        // Tasks that involve research or multi-phase work (web search, reading
        // existing files for context, sequential "then" steps) need Standard
        // budget even when the final target is a scoped file path.
        let has_research_or_multi_phase = [
            "search the web",
            "web search",
            "search online",
            "look up online",
            "then create",
            "then write",
            "then make",
            "and create",
            "and write",
            "and make",
        ]
        .iter()
        .any(|phrase| lower.contains(phrase))
            || (contains_keyword_as_words(&lower, "search")
                && contains_keyword_as_words(&lower, "create"));

        let needs_standard = has_verification_step || has_research_or_multi_phase;

        let tier = if has_scoped_target && !needs_standard {
            BudgetTier::Small
        } else {
            BudgetTier::Standard
        };
        let route_kind = if has_scoped_target && !needs_standard {
            "scoped_modification"
        } else if has_scoped_target {
            "scoped_modification_with_verification"
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
        // Previously BudgetTier::None — but this stripped tools for queries
        // that legitimately need them (e.g., "what time is it in tokyo?"
        // needs web_search). Use Standard as the minimum so tools are always
        // available. The model self-decides whether to use them.
        let tier = BudgetTier::Standard;
        return promote_contextual_followup_budget(tier, "read_only_investigation", turn_context);
    }

    // Previously BudgetTier::None — same rationale as above.
    let tier = BudgetTier::Standard;
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
    fn research_plus_create_gets_standard_not_small() {
        // "Search the web ... then create a file at path.md" should get Standard
        // budget, not Small, even though it has a scoped target (.md).
        let turn_context = TurnContext::default();
        let (tier, route_kind, _budget) = select_initial_execution_budget(
            "Search the web for the top 3 Rust crates then create a markdown file at ~/projects/blog/drafts/rust-crates-2025.md",
            &turn_context,
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(
            tier,
            BudgetTier::Standard,
            "research+create should get Standard budget"
        );
        assert_eq!(route_kind, "scoped_modification_with_verification");
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
    fn knowledge_turns_use_standard_budget() {
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
        assert_eq!(tier, BudgetTier::Standard);
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
    fn read_only_investigation_uses_standard_budget() {
        let (tier, route_kind, _) = select_initial_execution_budget(
            "inspect the latest logs and show me the current status",
            &TurnContext::default(),
            0,
            AgentRole::Orchestrator,
        );
        assert_eq!(tier, BudgetTier::Standard);
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
        let (tier, _route_kind, budget) = select_initial_execution_budget(
            "Which one is most relevant to skin cancer?",
            &turn_context,
            0,
            AgentRole::Orchestrator,
        );
        // Tier is Standard regardless of followup context since the
        // base tier is now Standard (no longer None/Small that needed promotion).
        assert_eq!(tier, BudgetTier::Standard);
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
        let original_validation = state.budget.max_validation_rounds;

        // No extension when budget envelope is inactive
        state.extend_budget_on_progress();
        assert_eq!(state.budget.max_llm_calls, original_llm);
        assert_eq!(state.budget.max_wall_clock_ms, original_wall);
        assert_eq!(state.budget.max_validation_rounds, original_validation);

        // Extension kicks in once the envelope is active
        state.activate_budget_envelope(0, Duration::from_millis(0));
        state.extend_budget_on_progress();
        assert!(state.budget.max_llm_calls > original_llm);
        assert!(state.budget.max_tool_calls > original_tools);
        assert!(state.budget.max_steps > original_steps);
        assert!(state.budget.max_wall_clock_ms > original_wall);
        assert!(state.budget.max_validation_rounds > original_validation);

        // Cumulative extensions keep growing
        let after_first = state.budget.max_llm_calls;
        let after_first_wall = state.budget.max_wall_clock_ms;
        let after_first_validation = state.budget.max_validation_rounds;
        state.extend_budget_on_progress();
        assert!(state.budget.max_llm_calls > after_first);
        assert!(state.budget.max_wall_clock_ms > after_first_wall);
        assert!(state.budget.max_validation_rounds > after_first_validation);
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
        // call + occasional validation round, but also extends via progress.
        // Use realistic elapsed time (~10s per iteration → 300s total) to
        // verify wall-clock extension keeps pace with real-world execution.
        for i in 0..30 {
            state.record_llm_call();
            state.record_tool_call();
            // Simulate a validation round every ~10 tool calls (realistic
            // for complex multi-step tasks).
            if i % 10 == 9 {
                state.record_validation_round();
            }
            state.extend_budget_on_progress();
        }

        // 30 iterations × ~10s each = 300s of wall time.  The base budget
        // for None tier is 180s, but 30 progress extensions add 30 × 30s =
        // 900s, giving a total wall-clock budget of 1080s — well above 300s.
        // Validation rounds: base 3, used 3, but 30 extensions of +1 each
        // give 33 total — well above the 3 used.
        let realistic_elapsed = Duration::from_secs(300);
        assert_eq!(
            state.exhausted_limit(0, realistic_elapsed),
            None,
            "Productive run should never exhaust budget, even with realistic wall-clock time"
        );
    }

    fn test_execution_state() -> ExecutionState {
        ExecutionState::new(
            BudgetTier::None,
            default_execution_budget(BudgetTier::None),
            ExecutionPersistence::Ephemeral,
        )
    }

    #[test]
    fn outcome_ledger_starts_empty() {
        let state = test_execution_state();
        assert!(state.outcome_ledger.is_empty());
    }

    #[test]
    fn outcome_ledger_records_success() {
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: true,
            http_status: Some(201),
            is_external_mutation: true,
            error_summary: None,
            iteration: 1,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        assert_eq!(state.outcome_ledger.len(), 1);
        assert!(state.outcome_ledger[0].success);
    }

    #[test]
    fn outcome_ledger_tracks_failed_external_mutations() {
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: false,
            http_status: Some(403),
            is_external_mutation: true,
            error_summary: Some("duplicate content".to_string()),
            iteration: 1,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        assert!(state.has_uncorrected_failed_external_mutations());
    }

    #[test]
    fn outcome_ledger_ignores_non_external_failures() {
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "read_file".to_string(),
            success: false,
            http_status: None,
            is_external_mutation: false,
            error_summary: Some("file not found".to_string()),
            iteration: 1,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        assert!(!state.has_uncorrected_failed_external_mutations());
    }

    #[test]
    fn attempt_reconciliation_none_when_all_succeeded() {
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: true,
            http_status: Some(201),
            is_external_mutation: true,
            error_summary: None,
            iteration: 1,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        assert!(state.build_attempt_reconciliation_summary().is_none());
    }

    #[test]
    fn attempt_reconciliation_present_when_failures_exist() {
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: true,
            http_status: Some(201),
            is_external_mutation: true,
            error_summary: None,
            iteration: 1,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: false,
            http_status: Some(403),
            is_external_mutation: true,
            error_summary: Some("duplicate content".to_string()),
            iteration: 2,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        let summary = state.build_attempt_reconciliation_summary().unwrap();
        assert!(summary.contains("attempts"));
        assert!(summary.contains("1") && summary.contains("2"));
        assert!(summary.contains("failed"));
        assert!(summary.contains("403"));
        assert!(summary.contains("duplicate content"));
    }

    #[test]
    fn attempt_reconciliation_says_attempts_not_actions() {
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: false,
            http_status: Some(403),
            is_external_mutation: true,
            error_summary: Some("dup".to_string()),
            iteration: 1,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        let summary = state.build_attempt_reconciliation_summary().unwrap();
        assert!(summary.contains("attempt"));
        assert!(!summary.contains("action"));
    }

    #[test]
    fn corrected_failure_same_tool_skips_reconciliation() {
        // Failure at iter 3, then success of SAME tool at iter 7 → corrected
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "run_command".to_string(),
            success: false,
            http_status: None,
            is_external_mutation: true,
            error_summary: Some("could not find Cargo.toml".to_string()),
            iteration: 3,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        state.record_outcome(OutcomeEntry {
            tool_name: "run_command".to_string(),
            success: true,
            http_status: None,
            is_external_mutation: true,
            error_summary: None,
            iteration: 7,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        assert!(state.uncorrected_failed_mutations().is_empty());
        assert!(!state.has_uncorrected_failed_external_mutations());
        assert!(state.build_attempt_reconciliation_summary().is_none());
    }

    #[test]
    fn corrected_failure_different_tool_skips_reconciliation() {
        // Failure via run_command at iter 9, then success via terminal at iter 15
        // → corrected (all failures before last success)
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "run_command".to_string(),
            success: false,
            http_status: None,
            is_external_mutation: true,
            error_summary: Some("could not find Cargo.toml".to_string()),
            iteration: 9,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        state.record_outcome(OutcomeEntry {
            tool_name: "terminal".to_string(),
            success: true,
            http_status: None,
            is_external_mutation: true,
            error_summary: None,
            iteration: 15,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        assert!(state.uncorrected_failed_mutations().is_empty());
        assert!(!state.has_uncorrected_failed_external_mutations());
        assert!(state.build_attempt_reconciliation_summary().is_none());
    }

    #[test]
    fn uncorrected_failure_after_last_success_triggers_reconciliation() {
        // Success at iter 5, then failure at iter 10 → uncorrected
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "terminal".to_string(),
            success: true,
            http_status: None,
            is_external_mutation: true,
            error_summary: None,
            iteration: 5,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: false,
            http_status: Some(500),
            is_external_mutation: true,
            error_summary: Some("server error".to_string()),
            iteration: 10,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        assert_eq!(state.uncorrected_failed_mutations().len(), 1);
        assert!(state.has_uncorrected_failed_external_mutations());
        assert!(state.build_attempt_reconciliation_summary().is_some());
    }

    #[test]
    fn mixed_corrected_and_uncorrected_failures() {
        // run_command FAIL at iter 3 (corrected by terminal SUCCESS at iter 15)
        // http_request FAIL at iter 20 (after last success → uncorrected)
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "run_command".to_string(),
            success: false,
            http_status: None,
            is_external_mutation: true,
            error_summary: Some("not found".to_string()),
            iteration: 3,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        state.record_outcome(OutcomeEntry {
            tool_name: "terminal".to_string(),
            success: true,
            http_status: None,
            is_external_mutation: true,
            error_summary: None,
            iteration: 15,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: false,
            http_status: Some(500),
            is_external_mutation: true,
            error_summary: Some("deploy failed".to_string()),
            iteration: 20,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        let uncorrected = state.uncorrected_failed_mutations();
        assert_eq!(uncorrected.len(), 1);
        assert_eq!(uncorrected[0].tool_name, "http_request");
        assert_eq!(uncorrected[0].iteration, 20);
        let summary = state.build_attempt_reconciliation_summary().unwrap();
        assert!(summary.contains("deploy failed"));
        assert!(!summary.contains("not found")); // corrected failure excluded
    }

    #[test]
    fn install_linear_intent_plan_sets_current_step_identity() {
        let mut state = test_execution_state();
        state.install_linear_intent_plan(
            3,
            vec![
                LinearIntentStep {
                    step_id: "plan-v3-step-1".to_string(),
                    step_index: 1,
                    tool: "http_request".to_string(),
                    target: "tweet-1".to_string(),
                    description: "Post tweet 1".to_string(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
                LinearIntentStep {
                    step_id: "plan-v3-step-2".to_string(),
                    step_index: 2,
                    tool: "http_request".to_string(),
                    target: "tweet-2".to_string(),
                    description: "Post tweet 2".to_string(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
            ],
        );
        let current = state.current_linear_intent_step().unwrap();
        assert_eq!(current.step_id, "plan-v3-step-1");
        assert_eq!(current.step_index, 1);
    }

    #[test]
    fn advance_linear_intent_step_on_success_moves_forward() {
        let mut state = test_execution_state();
        state.install_linear_intent_plan(
            1,
            vec![
                LinearIntentStep {
                    step_id: "plan-v1-step-1".to_string(),
                    step_index: 1,
                    tool: "http_request".to_string(),
                    target: "tweet-1".to_string(),
                    description: "Post tweet 1".to_string(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
                LinearIntentStep {
                    step_id: "plan-v1-step-2".to_string(),
                    step_index: 2,
                    tool: "http_request".to_string(),
                    target: "tweet-2".to_string(),
                    description: "Post tweet 2".to_string(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
            ],
        );
        // First advance: step 1 → step 2
        state.advance_linear_intent_step_after_external_success();
        let current = state.current_linear_intent_step().unwrap();
        assert_eq!(current.step_index, 2);

        // Second advance: step 2 → past end (cursor retires)
        state.advance_linear_intent_step_after_external_success();
        assert!(
            state.current_linear_intent_step().is_none(),
            "cursor should retire past the last step"
        );

        // Further advances are no-ops
        state.advance_linear_intent_step_after_external_success();
        assert!(state.current_linear_intent_step().is_none());
    }

    #[test]
    fn planned_step_reconciliation_groups_retry_under_one_step() {
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: false,
            http_status: Some(403),
            is_external_mutation: true,
            error_summary: Some("duplicate content".to_string()),
            iteration: 1,
            plan_version: Some(1),
            planned_step_id: Some("plan-v1-step-2".to_string()),
            planned_step_index: Some(2),
            planned_step_description: Some("Post tweet 2".to_string()),
            expected_step_count: Some(5),
        });
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: true,
            http_status: Some(201),
            is_external_mutation: true,
            error_summary: None,
            iteration: 2,
            plan_version: Some(1),
            planned_step_id: Some("plan-v1-step-2".to_string()),
            planned_step_index: Some(2),
            planned_step_description: Some("Post tweet 2".to_string()),
            expected_step_count: Some(5),
        });
        let summary = state.build_reconciliation_overview().unwrap().summary;
        assert!(summary.contains("step"));
        assert!(summary.contains("5"));
        assert!(summary.contains("Post tweet 2"));
        assert!(summary.contains("succeeded after 2 attempts"));
    }

    #[test]
    fn planned_step_reconciliation_uses_latest_plan_version_only() {
        let mut state = test_execution_state();
        state.install_linear_intent_plan(
            2,
            vec![
                LinearIntentStep {
                    step_id: "plan-v2-step-1".to_string(),
                    step_index: 1,
                    tool: "http_request".to_string(),
                    target: "tweet-1".to_string(),
                    description: "Post tweet 1".to_string(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
                LinearIntentStep {
                    step_id: "plan-v2-step-2".to_string(),
                    step_index: 2,
                    tool: "http_request".to_string(),
                    target: "tweet-2".to_string(),
                    description: "Post tweet 2".to_string(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
            ],
        );
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: true,
            http_status: Some(201),
            is_external_mutation: true,
            error_summary: None,
            iteration: 1,
            plan_version: Some(1),
            planned_step_id: Some("plan-v1-step-1".to_string()),
            planned_step_index: Some(1),
            planned_step_description: Some("Old tweet 1".to_string()),
            expected_step_count: Some(3),
        });
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: true,
            http_status: Some(201),
            is_external_mutation: true,
            error_summary: None,
            iteration: 2,
            plan_version: Some(2),
            planned_step_id: Some("plan-v2-step-1".to_string()),
            planned_step_index: Some(1),
            planned_step_description: Some("Post tweet 1".to_string()),
            expected_step_count: Some(2),
        });

        let overview = state.build_reconciliation_overview().unwrap();
        assert_eq!(overview.mode, ReconciliationMode::PlannedStepLevel);
        assert_eq!(overview.total, 2);
        assert_eq!(overview.succeeded, 1);
        assert_eq!(overview.failed, 1);
        assert_eq!(overview.failed_step_indices, vec![2]);
        assert!(!overview.summary.contains("Old tweet 1"));
        assert!(overview
            .summary
            .contains("Step 2 (Post tweet 2) was not completed."));
    }

    #[test]
    fn reconciliation_falls_back_to_attempt_level_without_step_identity() {
        let mut state = test_execution_state();
        state.record_outcome(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: false,
            http_status: Some(403),
            is_external_mutation: true,
            error_summary: Some("duplicate content".to_string()),
            iteration: 1,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });
        let summary = state.build_reconciliation_overview().unwrap().summary;
        assert!(summary.contains("attempt"));
    }

    #[test]
    fn promote_budget_for_plan_none_to_standard() {
        let mut state = ExecutionState::new(
            BudgetTier::None,
            default_execution_budget(BudgetTier::None),
            ExecutionPersistence::Ephemeral,
        );
        let original_llm_calls = state.budget.max_llm_calls;
        let original_wall_clock = state.budget.max_wall_clock_ms;
        state.promote_budget_for_plan(4);
        let standard = default_execution_budget(BudgetTier::Standard);
        // None tier has lower llm_calls and wall_clock than Standard
        assert!(state.budget.max_llm_calls >= standard.max_llm_calls);
        assert!(state.budget.max_llm_calls > original_llm_calls);
        assert!(state.budget.max_wall_clock_ms > original_wall_clock);
    }

    #[test]
    fn promote_budget_for_plan_small_to_standard() {
        let mut state = ExecutionState::new(
            BudgetTier::Small,
            default_execution_budget(BudgetTier::Small),
            ExecutionPersistence::Ephemeral,
        );
        state.promote_budget_for_plan(3);
        let standard = default_execution_budget(BudgetTier::Standard);
        assert!(state.budget.max_llm_calls >= standard.max_llm_calls);
    }

    #[test]
    fn no_promote_for_small_plan() {
        let mut state = ExecutionState::new(
            BudgetTier::None,
            default_execution_budget(BudgetTier::None),
            ExecutionPersistence::Ephemeral,
        );
        let original = state.budget.max_tool_calls;
        state.promote_budget_for_plan(2);
        assert_eq!(state.budget.max_tool_calls, original);
    }

    #[test]
    fn no_promote_for_standard_plus() {
        let mut state = ExecutionState::new(
            BudgetTier::Standard,
            default_execution_budget(BudgetTier::Standard),
            ExecutionPersistence::Ephemeral,
        );
        let original = state.budget.max_tool_calls;
        state.promote_budget_for_plan(5);
        assert_eq!(state.budget.max_tool_calls, original);
    }

    #[test]
    fn plan_step_replan_debounce() {
        let mut plan = LinearIntentPlan {
            plan_version: 1,
            steps: vec![LinearIntentStep {
                step_id: "s1".into(),
                step_index: 1,
                tool: String::new(),
                target: String::new(),
                description: "Explore".into(),
                tool_calls_on_step: 0,
                completed: false,
                completion_evidence: None,
                last_evaluated_at: None,
            }],
            current_step_cursor: 0,
        };

        assert!(!plan.current_step_needs_replan());
        plan.record_tool_calls_on_current(1);
        assert!(!plan.current_step_needs_replan());
        plan.record_tool_calls_on_current(1);
        assert!(plan.current_step_needs_replan());

        plan.mark_current_step_evaluated();
        assert!(!plan.current_step_needs_replan());

        plan.record_tool_calls_on_current(1);
        assert!(!plan.current_step_needs_replan());
        plan.record_tool_calls_on_current(1);
        assert!(plan.current_step_needs_replan());
    }

    #[test]
    fn plan_complete_step_advances_cursor() {
        let mut plan = LinearIntentPlan {
            plan_version: 1,
            steps: vec![
                LinearIntentStep {
                    step_id: "s1".into(),
                    step_index: 1,
                    tool: String::new(),
                    target: String::new(),
                    description: "Explore".into(),
                    tool_calls_on_step: 3,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
                LinearIntentStep {
                    step_id: "s2".into(),
                    step_index: 2,
                    tool: String::new(),
                    target: String::new(),
                    description: "Create".into(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
            ],
            current_step_cursor: 0,
        };

        plan.complete_current_step_with_evidence("Found 12 posts".into());
        assert_eq!(plan.current_step_cursor, 1);
        assert!(plan.steps[0].completed);
        assert_eq!(
            plan.steps[0].completion_evidence.as_deref(),
            Some("Found 12 posts")
        );
        assert!(!plan.all_steps_complete());

        plan.complete_current_step_with_evidence("Done".into());
        assert!(plan.all_steps_complete());
    }

    #[test]
    fn plan_format_with_progress_shows_markers() {
        let plan = LinearIntentPlan {
            plan_version: 1,
            steps: vec![
                LinearIntentStep {
                    step_id: "s1".into(),
                    step_index: 1,
                    tool: String::new(),
                    target: String::new(),
                    description: "Explore posts".into(),
                    tool_calls_on_step: 3,
                    completed: true,
                    completion_evidence: Some("Found 12 posts".into()),
                    last_evaluated_at: Some(2),
                },
                LinearIntentStep {
                    step_id: "s2".into(),
                    step_index: 2,
                    tool: String::new(),
                    target: String::new(),
                    description: "Create post 1".into(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
            ],
            current_step_cursor: 1,
        };

        let formatted = plan.format_with_progress();
        assert!(formatted.contains("[DONE] Explore posts"));
        assert!(formatted.contains("Found 12 posts"));
        assert!(formatted.contains("[CURRENT] Create post 1"));
    }
}
