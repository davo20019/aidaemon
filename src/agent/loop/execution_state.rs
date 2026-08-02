use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::agent::TurnContext;
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
            tool_output_evidence: String::new(),
            tool_output_evidence_overflow: false,
            web_search_used: false,
            web_source_domains: std::collections::HashSet::new(),
        }
    }

    /// Track web research provenance for the corroboration gate: which
    /// distinct domains were successfully read this turn. A "read" requires
    /// substantive content — errors, near-empty extractions, and
    /// extraction-failure notices don't count as having consulted a source.
    pub fn record_web_source(
        &mut self,
        tool_name: &str,
        arguments: &str,
        result_text: &str,
        is_error: bool,
    ) {
        // Minimum content for a fetch to count as a successfully read source.
        const MIN_SOURCE_CONTENT_CHARS: usize = 500;
        match tool_name {
            "web_search" if !is_error => {
                self.web_search_used = true;
            }
            "web_fetch" | "browser" => {
                if is_error
                    || result_text.len() < MIN_SOURCE_CONTENT_CHARS
                    || result_text.contains("⚠ EXTRACTION FAILED")
                {
                    return;
                }
                let Some(domain) = serde_json::from_str::<serde_json::Value>(arguments)
                    .ok()
                    .and_then(|v| v.get("url").and_then(|u| u.as_str()).map(String::from))
                    .and_then(|u| reqwest::Url::parse(&u).ok())
                    .and_then(|u| u.host_str().map(|h| h.to_lowercase()))
                else {
                    return;
                };
                self.web_source_domains
                    .insert(domain.trim_start_matches("www.").to_string());
            }
            _ => {}
        }
    }

    /// Accumulate raw tool output for the answer-grounding gate. Bounded:
    /// once the cap is reached the buffer stops growing and the overflow flag
    /// disables the gate for the rest of the turn (incomplete evidence would
    /// flag legitimately-observed entries as fabricated).
    pub fn record_tool_output_evidence(&mut self, output: &str) {
        const EVIDENCE_TOTAL_CAP: usize = 512 * 1024;
        if self.tool_output_evidence_overflow {
            return;
        }
        if self.tool_output_evidence.len() + output.len() > EVIDENCE_TOTAL_CAP {
            self.tool_output_evidence_overflow = true;
            return;
        }
        self.tool_output_evidence.push_str(output);
        self.tool_output_evidence.push('\n');
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
        self.uncorrected_failed_entries(true)
    }

    /// True if there are external mutation failures that were NOT corrected
    /// by later successes.
    pub fn has_uncorrected_failed_external_mutations(&self) -> bool {
        !self.uncorrected_failed_mutations().is_empty()
    }

    fn uncorrected_failed_entries(&self, external_mutation: bool) -> Vec<&OutcomeEntry> {
        let failures: Vec<&OutcomeEntry> = self
            .outcome_ledger
            .iter()
            .filter(|e| e.is_external_mutation == external_mutation && !e.success)
            .collect();

        if failures.is_empty() {
            return Vec::new();
        }

        let failures_with_summary: Vec<&OutcomeEntry> = failures
            .into_iter()
            .filter(|e| e.error_summary.is_some())
            .collect();

        if failures_with_summary.is_empty() {
            return Vec::new();
        }

        let uncorrected: Vec<&OutcomeEntry> = failures_with_summary
            .into_iter()
            .filter(|fail| {
                !self.outcome_ledger.iter().any(|e| {
                    // Terminal's mutation classification is a per-command
                    // heuristic (classify_shell_command): a failed
                    // `python3 -c` parse lands in the mutation ledger while
                    // the successful `grep | head` retry classifies as an
                    // observation — same tool, same goal, different class —
                    // and the failure read as "uncorrected" forever (live
                    // repro 2026-07-03, task 45a65347: the recovery pass
                    // worked, the report still said all-failed). For
                    // terminal ONLY, a later same-tool success of either
                    // class corrects; precisely-classified tools (http_request
                    // POST vs GET) keep the strict same-class rule so a read
                    // can never launder a failed write.
                    let class_matches =
                        e.is_external_mutation == external_mutation || fail.tool_name == "terminal";
                    class_matches
                        && e.success
                        && e.tool_name == fail.tool_name
                        && e.iteration > fail.iteration
                })
            })
            .collect();

        if uncorrected.is_empty() {
            return Vec::new();
        }

        let max_success_iter = self
            .outcome_ledger
            .iter()
            .filter(|e| e.is_external_mutation == external_mutation && e.success)
            .map(|e| e.iteration)
            .max();

        if let Some(max_success) = max_success_iter {
            return uncorrected
                .into_iter()
                .filter(|e| e.iteration > max_success)
                .collect();
        }

        uncorrected
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

    let call_is_pure_observation = semantics.observes_state() && !semantics.mutates_state();
    let approval_requirement = if !call_is_pure_observation
        && (capabilities.needs_approval || capabilities.high_impact_write)
    {
        ApprovalRequirement::Required {
            reason: format!("{} is approval-gated or high impact", tool_name),
        }
    } else {
        ApprovalRequirement::NotNeeded
    };

    let needs_idempotency = !call_is_pure_observation
        && (semantics.mutates_state()
            || capabilities.external_side_effect
            || !capabilities.idempotent);

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
        // Retries and crash recovery may receive a new model call id or loop
        // iteration. Bind idempotency to the durable execution plus the exact
        // canonical operation instead, so replaying the same side effect gets
        // the same key while changed arguments get a new key.
        idempotency_key: needs_idempotency.then(|| {
            let arguments = serde_json::from_str::<serde_json::Value>(effective_arguments)
                .unwrap_or_else(|_| serde_json::Value::String(effective_arguments.to_string()));
            let argument_hash = crate::agent::prefix_fingerprint::hash_canonical(&arguments);
            format!("exec:{execution_id}:{tool_name}:{argument_hash}")
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
        "resource_id",
        "url",
        "target",
        "target_url",
    ] {
        let Some(value) = parsed.get(key).and_then(|value| value.as_str()) else {
            continue;
        };
        let candidate = match key {
            "resource_id" => ToolTargetHint::new(ToolTargetHintKind::ResourceId, value),
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
#[path = "execution_state_tests.rs"]
mod tests;
