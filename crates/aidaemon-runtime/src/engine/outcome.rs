//! The reply and task outcome of a run that did not end with a final answer.

use aidaemon_core as sdk;
use aidaemon_events::{TaskOutcome, TaskStatus};

/// How the kernel writes a used-up quota into `RunStatus::ProviderFailed`
/// (the `Display` of `ProviderError::QuotaExhausted`).
const QUOTA_EXHAUSTED_PREFIX: &str = "quota exhausted: ";

/// The user-facing reply for a run that did not end with a final answer.
pub fn reply_for(outcome: &sdk::RunOutcome) -> (String, TaskStatus, TaskOutcome) {
    let text = outcome
        .final_text
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string);
    match &outcome.status {
        // Completed means the model stopped and every stop hook let it;
        // it is a success only when no hook reported the objective unmet
        // and the owner declined nothing (`RunOutcome::is_succeeded`).
        // The ledger reconciles further at task end.
        sdk::RunStatus::Completed => (
            text.unwrap_or_default(),
            TaskStatus::Completed,
            if outcome.is_succeeded() {
                TaskOutcome::Succeeded
            } else {
                TaskOutcome::Partial
            },
        ),
        sdk::RunStatus::LimitReached { limit } => (
            text.unwrap_or_else(|| {
                let what = match limit {
                    sdk::LimitKind::Turns => "the number of steps",
                    sdk::LimitKind::Tokens => "the token budget",
                    sdk::LimitKind::ToolCalls => "the number of tool calls",
                    sdk::LimitKind::WallClock => "the time",
                };
                format!(
                    "I stopped before finishing: I reached the limit on {what} for one request. \
                     Ask me to continue and I'll pick up from here."
                )
            }),
            TaskStatus::Completed,
            TaskOutcome::Partial,
        ),
        sdk::RunStatus::Stuck { signal } => (
            text.unwrap_or_else(|| {
                let _ = signal;
                "I stopped because I kept repeating the same step without making progress. \
                 Tell me how you'd like to proceed, or ask me to try a different approach."
                    .to_string()
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
        // A tool ended the run with its own typed finish: the run completed,
        // and its outcome is the tool's. Work left for someone else (a
        // blocker, a background job) is partial. A finish that carried no
        // reply ships the finishing call's own output.
        sdk::RunStatus::Finished {
            outcome: finish,
            tool,
        } => (
            text.or_else(|| finishing_call_output(outcome, tool))
                .unwrap_or_else(|| format!("`{tool}` ended the run.")),
            TaskStatus::Completed,
            match finish {
                sdk::FinishOutcome::Succeeded => TaskOutcome::Succeeded,
                _ => TaskOutcome::Partial,
            },
        ),
        sdk::RunStatus::Cancelled => (
            "Cancelled.".to_string(),
            TaskStatus::Cancelled,
            TaskOutcome::Failed,
        ),
        sdk::RunStatus::ProviderFailed { error } => (
            // Provider errors can carry paths and credentials: the user gets
            // the key line, redacted. A used-up quota's line already says
            // what happened and when it ends.
            match error.strip_prefix(QUOTA_EXHAUSTED_PREFIX) {
                Some(quota) => format!(
                    "I couldn't finish: {}",
                    crate::turn::post_task::redact_error_line_for_summary(
                        quota.lines().next().unwrap_or_default()
                    )
                ),
                None => format!(
                    "I couldn't finish because the model provider returned an error: {}",
                    crate::turn::post_task::redact_error_line_for_summary(
                        error.to_string().lines().next().unwrap_or_default()
                    )
                ),
            },
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

/// The model-visible output of the latest dispatched call of `tool`, the
/// untrusted-data wrapper removed.
fn finishing_call_output(outcome: &sdk::RunOutcome, tool: &str) -> Option<String> {
    let receipt = outcome.state.receipts.iter().rev().find(|receipt| {
        receipt.tool == tool && receipt.stage == sdk::InvocationStage::Dispatched
    })?;
    outcome
        .state
        .messages
        .iter()
        .rev()
        .find(|message| message.tool_call_id.as_deref() == Some(receipt.call_id.as_str()))
        .map(|message| {
            crate::turn::completion_checks::strip_untrusted_wrapper(&message.content).to_string()
        })
        .filter(|text| !text.trim().is_empty())
}
