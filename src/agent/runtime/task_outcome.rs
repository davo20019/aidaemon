//! Task outcome derivation from completion evidence at the `emit_task_end` boundary.

use super::completion_contract::{CompletionContract, CompletionProgress};
use super::execution_state::ExecutionState;
use super::goal_dispatch::is_low_signal_task_lead_reply;
use super::validation_state::ValidationState;
use crate::events::TaskOutcome;
use serde_json::Value;
use std::collections::BTreeMap;

/// Why a task terminated before natural completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Variants document persisted terminal causes before every caller is migrated.
pub enum TaskTerminalCause {
    Cancelled,
    Timeout,
    Watchdog,
    HardFailure,
    UnrecoveredModelFailure,
}

/// Counts of required actions and their resolution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RequestedActionSummary {
    pub required: u32,
    pub satisfied: u32,
    pub unresolved: u32,
}

impl RequestedActionSummary {
    pub fn from_completion_state(
        _validation: &ValidationState,
        execution: &ExecutionState,
        completion: &CompletionProgress,
    ) -> Self {
        // Planner-generated success criteria and linear-plan steps are
        // advisory execution aids. They are not user-authored acceptance
        // requirements and must not turn a fulfilled answer into `partial`
        // merely because the plan cursor was never advanced.
        let mut actions: BTreeMap<String, bool> = BTreeMap::new();

        if completion.verification_pending {
            actions.insert("verification:pending".to_string(), false);
        }

        // A failed exploratory observation is not itself an unresolved user
        // requirement. Required verification is represented by
        // `verification_pending` and the completion contract. Preserve failed
        // external mutations here because they can represent an uncompleted
        // requested action.
        for entry in execution.uncorrected_failed_mutations() {
            let Some(step_id) = entry.planned_step_id.as_deref() else {
                continue;
            };
            let description_key = entry
                .planned_step_description
                .as_deref()
                .map(action_key)
                .filter(|key| !key.is_empty());
            let key = description_key.unwrap_or_else(|| format!("plan-step:{step_id}"));
            actions.entry(key).or_insert(false);
        }

        let required = actions.len() as u32;
        let satisfied = actions.values().filter(|&&satisfied| satisfied).count() as u32;
        let unresolved = required.saturating_sub(satisfied);

        if unresolved > 0 {
            tracing::info!(
                required,
                satisfied,
                unresolved,
                "Task outcome bookkeeping: unresolved required actions"
            );
            // Keys are normalized from user-request text; keep them out of
            // the plaintext INFO log (the DB is encrypted, the log is not).
            let unresolved_keys: Vec<&str> = actions
                .iter()
                .filter(|(_, &satisfied)| !satisfied)
                .map(|(key, _)| key.as_str())
                .collect();
            tracing::debug!(
                ?unresolved_keys,
                "Task outcome bookkeeping: unresolved required action keys"
            );
        }

        Self {
            required,
            satisfied,
            unresolved,
        }
    }
}

fn action_key(text: &str) -> String {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                .to_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Inputs for deriving semantic task outcome once per task end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskOutcomeDerivation {
    pub response_has_user_value: bool,
    pub required_actions: RequestedActionSummary,
    pub completion_contract_fulfilled: bool,
    pub has_unrecovered_model_error: bool,
    pub terminal_cause: Option<TaskTerminalCause>,
    /// True when this turn moved a long-running command to the background with a
    /// "will notify you when done" ack. A deferral is not a failure.
    pub deferred_to_background: bool,
}

impl TaskOutcomeDerivation {
    pub fn from_completion_state(
        validation: &ValidationState,
        execution: &ExecutionState,
        completion: &CompletionProgress,
        contract: &CompletionContract,
        response_has_user_value: bool,
        has_unrecovered_model_error: bool,
        terminal_cause: Option<TaskTerminalCause>,
    ) -> Self {
        Self {
            response_has_user_value,
            required_actions: RequestedActionSummary::from_completion_state(
                validation, execution, completion,
            ),
            completion_contract_fulfilled: completion_contract_is_fulfilled(contract, completion),
            has_unrecovered_model_error,
            terminal_cause,
            deferred_to_background: execution.background_handoff_active,
        }
    }

    pub fn derive_outcome(&self) -> TaskOutcome {
        if self.has_unrecovered_model_error {
            return TaskOutcome::Failed;
        }
        if self.terminal_cause.is_some() {
            return TaskOutcome::Failed;
        }
        if !self.response_has_user_value {
            // A long-running command moved to the background (with a "will notify
            // you when done" ack) is a deferral, not a failure.
            if self.deferred_to_background {
                return TaskOutcome::Partial;
            }
            return TaskOutcome::Failed;
        }
        if self.required_actions.unresolved > 0 || !self.completion_contract_fulfilled {
            return TaskOutcome::Partial;
        }
        TaskOutcome::Succeeded
    }
}

fn completion_contract_is_fulfilled(
    contract: &CompletionContract,
    progress: &CompletionProgress,
) -> bool {
    // Expected mutation is an outcome obligation, not merely a routing hint.
    // A task cannot be persisted as Succeeded when its mutation ledger is
    // empty, even if the model produced a useful explanatory response.
    if contract.expects_mutation && progress.mutation_count == 0 {
        return false;
    }

    let observation_is_hard_requirement =
        contract.explicit_verification_requested || !contract.verification_targets.is_empty();
    if observation_is_hard_requirement
        && contract.requires_observation
        && progress.observation_count == 0
    {
        return false;
    }
    let verification_required = contract.explicit_verification_requested;
    if verification_required
        && (progress.verification_count == 0 || progress.verification_block_count > 2)
    {
        return false;
    }
    true
}

/// Whether accepted, sanitized assistant content has user-visible value.
pub fn response_has_user_value(reply: &str, total_successful_tool_calls: usize) -> bool {
    let trimmed = reply.trim();
    if trimmed.is_empty() {
        return false;
    }
    if is_low_signal_task_lead_reply(trimmed) {
        return false;
    }
    if trimmed.starts_with("The requested action completed successfully")
        || trimmed.starts_with("The requested action finished with errors")
    {
        return false;
    }
    if total_successful_tool_calls > 0 && trimmed == "Done." {
        return false;
    }
    if response_looks_like_plain_text_tool_call(trimmed) {
        return false;
    }
    true
}

/// Detect model outputs that tried to write a tool call as plain text instead
/// of returning a structured provider tool call.
pub fn response_looks_like_plain_text_tool_call(reply: &str) -> bool {
    let trimmed = reply.trim();
    if trimmed.is_empty() {
        return false;
    }

    let candidate = strip_single_code_fence(trimmed).unwrap_or(trimmed);
    let lower = candidate.to_ascii_lowercase();
    if lower.starts_with("<|tool_call")
        || lower.starts_with("[tool_call")
        || lower.starts_with("<tool_call")
        || lower.starts_with("tool_call:")
        || (lower.starts_with("call:") && lower.contains('{'))
    {
        return true;
    }

    match serde_json::from_str::<Value>(candidate) {
        Ok(value) => looks_like_tool_call_json(&value),
        Err(_) => false,
    }
}

fn strip_single_code_fence(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    let rest = trimmed.strip_prefix("```")?;
    let body_start = rest.find('\n').map(|idx| idx + 1).unwrap_or(0);
    let body = &rest[body_start..];
    let body = body.strip_suffix("```")?;
    Some(body.trim())
}

fn looks_like_tool_call_json(value: &Value) -> bool {
    if let Some(array) = value.as_array() {
        return array.iter().any(looks_like_tool_call_json);
    }

    let Some(obj) = value.as_object() else {
        return false;
    };

    if obj.get("tool_calls").and_then(Value::as_array).is_some() {
        return true;
    }

    let has_arguments = obj.contains_key("arguments")
        || obj.contains_key("args")
        || obj.contains_key("input")
        || obj.contains_key("parameters");
    if (obj.get("name").and_then(Value::as_str).is_some()
        || obj.get("tool").and_then(Value::as_str).is_some()
        || obj.get("recipient_name").and_then(Value::as_str).is_some()
        || obj.get("toolName").and_then(Value::as_str).is_some())
        && has_arguments
    {
        return true;
    }

    obj.get("function")
        .and_then(Value::as_object)
        .or_else(|| obj.get("function_call").and_then(Value::as_object))
        .is_some_and(|function| {
            function.get("name").and_then(Value::as_str).is_some()
                && function
                    .get("arguments")
                    .is_some_and(|arguments| !arguments.is_null())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::execution_state::{
        default_execution_budget, BudgetTier, ExecutionPersistence, ExecutionState, OutcomeEntry,
    };
    use crate::agent::validation_state::ValidationState;

    fn empty_execution_state() -> ExecutionState {
        ExecutionState::new(
            BudgetTier::None,
            default_execution_budget(BudgetTier::None),
            ExecutionPersistence::Ephemeral,
        )
    }

    fn base_derivation() -> TaskOutcomeDerivation {
        TaskOutcomeDerivation {
            response_has_user_value: true,
            required_actions: RequestedActionSummary {
                required: 0,
                satisfied: 0,
                unresolved: 0,
            },
            completion_contract_fulfilled: true,
            has_unrecovered_model_error: false,
            terminal_cause: None,
            deferred_to_background: false,
        }
    }

    #[test]
    fn succeeded_when_no_required_actions_and_useful_response() {
        let d = base_derivation();
        assert_eq!(d.derive_outcome(), TaskOutcome::Succeeded);
    }

    #[test]
    fn partial_when_unresolved_required_actions() {
        let mut d = base_derivation();
        d.required_actions = RequestedActionSummary {
            required: 2,
            satisfied: 1,
            unresolved: 1,
        };
        assert_eq!(d.derive_outcome(), TaskOutcome::Partial);
    }

    #[test]
    fn partial_when_completion_contract_is_unfulfilled() {
        let mut d = base_derivation();
        d.completion_contract_fulfilled = false;
        assert_eq!(d.derive_outcome(), TaskOutcome::Partial);
    }

    #[test]
    fn expected_mutation_without_ledger_evidence_is_partial() {
        let validation = ValidationState::default();
        let execution = empty_execution_state();
        let completion = CompletionProgress::default();
        let contract = CompletionContract {
            expects_mutation: true,
            ..CompletionContract::default()
        };

        let outcome = TaskOutcomeDerivation::from_completion_state(
            &validation,
            &execution,
            &completion,
            &contract,
            true,
            false,
            None,
        )
        .derive_outcome();

        assert_eq!(outcome, TaskOutcome::Partial);
    }

    #[test]
    fn explicit_observation_without_evidence_is_partial() {
        let validation = ValidationState::default();
        let execution = empty_execution_state();
        let completion = CompletionProgress::default();
        let contract = CompletionContract {
            requires_observation: true,
            explicit_verification_requested: true,
            ..CompletionContract::default()
        };

        let outcome = TaskOutcomeDerivation::from_completion_state(
            &validation,
            &execution,
            &completion,
            &contract,
            true,
            false,
            None,
        )
        .derive_outcome();

        assert_eq!(outcome, TaskOutcome::Partial);
    }

    #[test]
    fn concrete_mutation_target_without_evidence_is_partial() {
        let validation = ValidationState::default();
        let execution = empty_execution_state();
        let completion = CompletionProgress::default();
        let contract = CompletionContract {
            expects_mutation: true,
            verification_targets: vec![crate::agent::VerificationTarget {
                kind: crate::agent::VerificationTargetKind::Path,
                value: "/tmp/result.txt".to_string(),
            }],
            ..CompletionContract::default()
        };

        let outcome = TaskOutcomeDerivation::from_completion_state(
            &validation,
            &execution,
            &completion,
            &contract,
            true,
            false,
            None,
        )
        .derive_outcome();

        assert_eq!(outcome, TaskOutcome::Partial);
    }

    #[test]
    fn failed_on_cancellation() {
        let mut d = base_derivation();
        d.terminal_cause = Some(TaskTerminalCause::Cancelled);
        assert_eq!(d.derive_outcome(), TaskOutcome::Failed);
    }

    #[test]
    fn failed_on_empty_response() {
        let mut d = base_derivation();
        d.response_has_user_value = false;
        assert_eq!(d.derive_outcome(), TaskOutcome::Failed);
    }

    #[test]
    fn deferred_to_background_is_partial_not_failed() {
        // A long command moved to background with an ack is a deferral. Even with
        // no final user-facing answer this turn, it must score Partial, not Failed.
        let mut d = base_derivation();
        d.deferred_to_background = true;
        d.response_has_user_value = false;
        assert_eq!(d.derive_outcome(), TaskOutcome::Partial);

        // With a valid final reply and no unresolved work, a previously
        // backgrounded task can complete successfully.
        let mut d = base_derivation();
        d.deferred_to_background = true;
        assert_eq!(d.derive_outcome(), TaskOutcome::Succeeded);
    }

    #[test]
    fn completed_background_followup_can_succeed() {
        let mut d = base_derivation();
        d.deferred_to_background = true;
        d.required_actions = RequestedActionSummary {
            required: 3,
            satisfied: 3,
            unresolved: 0,
        };

        assert_eq!(d.derive_outcome(), TaskOutcome::Succeeded);
    }

    #[test]
    fn genuine_error_still_fails_even_when_deferred() {
        // An unrecovered model error outranks the deferral signal.
        let mut d = base_derivation();
        d.deferred_to_background = true;
        d.has_unrecovered_model_error = true;
        assert_eq!(d.derive_outcome(), TaskOutcome::Failed);
    }

    #[test]
    fn cancelled_while_not_deferred_still_fails() {
        let mut d = base_derivation();
        d.terminal_cause = Some(TaskTerminalCause::Cancelled);
        assert_eq!(d.derive_outcome(), TaskOutcome::Failed);
    }

    #[test]
    fn incidental_tool_failure_does_not_block_informational_success() {
        let validation = ValidationState {
            active_success_criteria: vec!["answer the question".to_string()],
            matched_success_criteria: vec!["answer the question".to_string()],
            ..Default::default()
        };

        let mut execution = empty_execution_state();
        execution.outcome_ledger.push(OutcomeEntry {
            tool_name: "web_search".to_string(),
            success: false,
            http_status: None,
            is_external_mutation: false,
            error_summary: Some("timeout".to_string()),
            iteration: 1,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });

        let summary = RequestedActionSummary::from_completion_state(
            &validation,
            &execution,
            &CompletionProgress::default(),
        );
        assert_eq!(summary.unresolved, 0);

        let outcome = TaskOutcomeDerivation::from_completion_state(
            &validation,
            &execution,
            &CompletionProgress::default(),
            &CompletionContract::default(),
            true,
            false,
            None,
        )
        .derive_outcome();
        assert_eq!(outcome, TaskOutcome::Succeeded);
    }

    #[test]
    fn unlinked_uncorrected_mutation_does_not_create_required_action() {
        let validation = ValidationState::default();
        let mut execution = empty_execution_state();
        execution.outcome_ledger.push(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: false,
            http_status: Some(500),
            is_external_mutation: true,
            error_summary: Some("server error".to_string()),
            iteration: 1,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });

        let summary = RequestedActionSummary::from_completion_state(
            &validation,
            &execution,
            &CompletionProgress::default(),
        );
        assert_eq!(summary.unresolved, 0);

        let outcome = TaskOutcomeDerivation::from_completion_state(
            &validation,
            &execution,
            &CompletionProgress::default(),
            &CompletionContract::default(),
            response_has_user_value("Here is what I found on the homepage.", 2),
            false,
            None,
        )
        .derive_outcome();
        assert_eq!(outcome, TaskOutcome::Succeeded);
    }

    #[test]
    fn unrelated_completed_plan_step_does_not_satisfy_unmatched_criterion() {
        let validation = ValidationState {
            active_success_criteria: vec!["publish the release".to_string()],
            ..Default::default()
        };

        let mut execution = empty_execution_state();
        execution.install_linear_intent_plan(
            1,
            vec![crate::agent::execution_state::LinearIntentStep {
                step_id: "inspect".to_string(),
                step_index: 1,
                tool: "read_file".to_string(),
                target: "CHANGELOG.md".to_string(),
                description: "inspect the changelog".to_string(),
                tool_calls_on_step: 1,
                completed: true,
                completion_evidence: Some("read successfully".to_string()),
                last_evaluated_at: None,
            }],
        );

        let summary = RequestedActionSummary::from_completion_state(
            &validation,
            &execution,
            &CompletionProgress::default(),
        );

        assert_eq!(summary, RequestedActionSummary::default());
    }

    #[test]
    fn unlinked_failed_mutation_is_incidental_not_a_new_requirement() {
        let validation = ValidationState::default();
        let mut execution = empty_execution_state();
        execution.outcome_ledger.push(OutcomeEntry {
            tool_name: "http_request".to_string(),
            success: false,
            http_status: Some(500),
            is_external_mutation: true,
            error_summary: Some("server error".to_string()),
            iteration: 1,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });

        let summary = RequestedActionSummary::from_completion_state(
            &validation,
            &execution,
            &CompletionProgress::default(),
        );

        assert_eq!(summary, RequestedActionSummary::default());
    }

    #[test]
    fn duplicate_criterion_and_plan_step_are_counted_once() {
        let validation = ValidationState {
            active_success_criteria: vec!["inspect the homepage".to_string()],
            ..Default::default()
        };

        let mut execution = empty_execution_state();
        execution.install_linear_intent_plan(
            1,
            vec![crate::agent::execution_state::LinearIntentStep {
                step_id: "inspect-homepage".to_string(),
                step_index: 1,
                tool: "browser".to_string(),
                target: "https://example.com".to_string(),
                description: "Inspect the homepage.".to_string(),
                tool_calls_on_step: 1,
                completed: true,
                completion_evidence: Some("homepage inspected".to_string()),
                last_evaluated_at: None,
            }],
        );

        let summary = RequestedActionSummary::from_completion_state(
            &validation,
            &execution,
            &CompletionProgress::default(),
        );

        assert_eq!(summary, RequestedActionSummary::default());
    }

    #[test]
    fn advisory_answer_plan_does_not_make_fulfilled_turn_partial() {
        let validation = ValidationState {
            active_success_criteria: vec![
                "retrieve the stored relationship".to_string(),
                "answer the user directly".to_string(),
            ],
            ..Default::default()
        };

        let mut execution = empty_execution_state();
        execution.install_linear_intent_plan(
            1,
            vec![
                crate::agent::execution_state::LinearIntentStep {
                    step_id: "lookup".to_string(),
                    step_index: 1,
                    tool: "manage_memories".to_string(),
                    target: String::new(),
                    description: "retrieve the stored relationship".to_string(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
                crate::agent::execution_state::LinearIntentStep {
                    step_id: "answer".to_string(),
                    step_index: 2,
                    tool: String::new(),
                    target: String::new(),
                    description: "answer the user directly".to_string(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
            ],
        );

        let outcome = TaskOutcomeDerivation::from_completion_state(
            &validation,
            &execution,
            &CompletionProgress::default(),
            &CompletionContract::default(),
            true,
            false,
            None,
        )
        .derive_outcome();

        assert_eq!(outcome, TaskOutcome::Succeeded);
    }

    #[test]
    fn response_has_user_value_rejects_low_signal_replies() {
        assert!(!response_has_user_value("Done.", 3));
        assert!(!response_has_user_value("", 0));
        assert!(response_has_user_value(
            "The homepage shows the product catalog with 12 items.",
            2
        ));
    }

    #[test]
    fn response_has_user_value_rejects_plain_text_tool_call_leaks() {
        assert!(!response_has_user_value(
            r#"<|tool_call>call:terminal {"command":"wrangler pages deploy ./dist"}"#,
            0
        ));
        assert!(!response_has_user_value(
            r#"call:browser {"url":"https://example.com"}"#,
            0
        ));
        assert!(!response_has_user_value(
            r#"{"name":"terminal","arguments":{"command":"curl -I https://example.com"}}"#,
            0
        ));
    }

    #[test]
    fn response_has_user_value_rejects_common_tool_call_text_dialects() {
        assert!(!response_has_user_value(
            "```json\n{\"tool\":\"terminal\",\"input\":{\"command\":\"cargo test\"}}\n```",
            0
        ));
        assert!(!response_has_user_value(
            r#"[{"name":"browser","arguments":{"url":"https://example.com"}}]"#,
            0
        ));
        assert!(!response_has_user_value(
            r#"{"function_call":{"name":"terminal","arguments":{"command":"cargo fmt"}}}"#,
            0
        ));
        assert!(response_has_user_value(
            r#"{"status":"ok","summary":"Deployment finished and returned HTTP 200."}"#,
            1
        ));
    }

    #[test]
    fn pending_deploy_verification_prevents_success() {
        let validation = ValidationState::default();
        let mut execution = empty_execution_state();
        execution.outcome_ledger.push(OutcomeEntry {
            tool_name: "terminal".to_string(),
            success: true,
            http_status: None,
            is_external_mutation: true,
            error_summary: None,
            iteration: 1,
            plan_version: None,
            planned_step_id: None,
            planned_step_index: None,
            planned_step_description: None,
            expected_step_count: None,
        });

        let completion = CompletionProgress {
            mutation_count: 1,
            successful_external_mutation_count: 1,
            verification_pending: true,
            ..CompletionProgress::default()
        };

        let outcome = TaskOutcomeDerivation::from_completion_state(
            &validation,
            &execution,
            &completion,
            &CompletionContract::default(),
            response_has_user_value("Deployment complete.", 1),
            false,
            None,
        )
        .derive_outcome();

        assert_eq!(outcome, TaskOutcome::Partial);
    }
}
