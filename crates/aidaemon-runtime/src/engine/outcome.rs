//! The reply and task outcome of a run that did not end with a final answer.

use aidaemon_core as sdk;
use aidaemon_events::{TaskOutcome, TaskStatus};

use super::budgets::{GOAL_BUDGET_EXHAUSTED, MANDATE_BUDGET_EXHAUSTED};
use super::stop::finalize_reply;

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

/// What a run settled, beyond its kernel status, that decides the turn's
/// final status and outcome. Shared with run-end hooks (memory writers), so
/// they judge a run exactly as the turn does.
#[derive(Default)]
pub struct TurnVerdict {
    /// The reply and outcome a used-up goal or mandate budget closes the run
    /// with; the run itself ends through the provider port.
    pub budget_stop: std::sync::Mutex<Option<(String, TaskOutcome)>>,
    /// Confirmation of the latest successful external change.
    pub external_action_ack: std::sync::Mutex<Option<String>>,
    /// Set when the run must end failed despite a final answer.
    pub run_failure: std::sync::Mutex<Option<String>>,
    /// The latest provider note (for example a safety finish reason).
    pub response_note: std::sync::Mutex<Option<String>>,
    /// The latest well-formed final answer a closeout demand sent back for
    /// more work. If the model then replies with nothing, this ships instead
    /// of a generic activity summary.
    pub interrupted_answer: std::sync::Mutex<Option<String>>,
    /// The handoff reply for a run that stopped while a job it launched was
    /// still running: that job's notifier delivers the result, so the run
    /// tells the user the work continues instead of shipping the model's own
    /// verdict on unfinished work.
    pub handoff_reply: std::sync::Mutex<Option<String>>,
}

impl TurnVerdict {
    /// The run ends on a used-up budget: the next model call fails with the
    /// budget's marker, and the turn ships `reply` with `outcome`.
    pub fn record_budget_stop(&self, reply: String, outcome: TaskOutcome) {
        if let Ok(mut stop) = self.budget_stop.lock() {
            *stop = Some((reply, outcome));
        }
    }

    pub fn record_handoff_reply(&self, reply: String) {
        if let Ok(mut handoff) = self.handoff_reply.lock() {
            *handoff = Some(reply);
        }
    }

    pub fn record_interrupted_answer(&self, answer: String) {
        if let Ok(mut interrupted) = self.interrupted_answer.lock() {
            *interrupted = Some(answer);
        }
    }

    /// The reply the user gets, with its status and outcome: [`Self::resolve`],
    /// then for a completed run the interrupted-answer fallback and the final
    /// reply checks (reply sanitization alone for a stopped run), and for
    /// every run secret redaction on public channels.
    pub fn final_reply(
        &self,
        outcome: &sdk::RunOutcome,
        tool_calls: &[crate::turn::post_task::LearnedCall],
        visibility: aidaemon_core::ChannelVisibility,
    ) -> (String, TaskStatus, TaskOutcome) {
        let (reply, status, task_outcome) = self.resolve(outcome);
        let handoff = self
            .handoff_reply
            .lock()
            .ok()
            .and_then(|reply| reply.clone());
        // A handed-off run stopped with its work still running: partial work,
        // whatever the kernel's completed status would otherwise credit. A
        // run ended by a launch's own finish is handed off too: the kernel
        // keeps a batch's first finish, so its reply named only the first
        // launch, while the recorded handoff names every run the turn left
        // running (live 2026-11-14: two helpers started, the reply named one).
        let handed_off = matches!(
            outcome.status,
            sdk::RunStatus::Completed | sdk::RunStatus::Finished { .. }
        ) && handoff.is_some();
        let task_outcome = if handed_off && task_outcome == TaskOutcome::Succeeded {
            TaskOutcome::Partial
        } else {
            task_outcome
        };
        let reply = if let (true, Some(handoff)) = (handed_off, handoff) {
            handoff
        } else if matches!(outcome.status, sdk::RunStatus::Completed) {
            let note = self.response_note.lock().ok().and_then(|note| note.clone());
            let interrupted = self
                .interrupted_answer
                .lock()
                .ok()
                .and_then(|answer| answer.clone());
            let reply = match interrupted {
                Some(answer) if reply.trim().is_empty() => answer,
                _ => reply,
            };
            finalize_reply(&reply, tool_calls, note.as_deref())
        } else {
            // A stopped run's text (the model's partial answer, or the
            // daemon's own explanation) gets the same reply sanitization; a
            // partial answer that was mostly scaffolding is not shipped.
            let sanitized = aidaemon_tools::sanitize::sanitize_user_facing_reply(&reply);
            if aidaemon_tools::sanitize::reply_gutted_by_sanitization(
                reply.trim().chars().count(),
                &sanitized,
            ) {
                "I stopped before finishing and couldn't put together a clean summary of where \
                 I got to. Ask me to continue and I'll pick up from here."
                    .to_string()
            } else {
                sanitized
            }
        };
        let reply = match visibility {
            aidaemon_core::ChannelVisibility::Public
            | aidaemon_core::ChannelVisibility::PublicExternal => {
                let (sanitized, redacted) = aidaemon_tools::sanitize::sanitize_output(&reply);
                if redacted && visibility == aidaemon_core::ChannelVisibility::PublicExternal {
                    format!("{sanitized}\n\n(Some content was filtered for security)")
                } else {
                    sanitized
                }
            }
            _ => reply,
        };
        (reply, status, task_outcome)
    }

    pub fn run_failure(&self) -> Option<String> {
        self.run_failure
            .lock()
            .ok()
            .and_then(|failure| failure.clone())
    }

    /// The turn's reply, status and outcome for a finished run.
    pub fn resolve(&self, outcome: &sdk::RunOutcome) -> (String, TaskStatus, TaskOutcome) {
        let (reply, status, task_outcome) = reply_for(outcome);
        let external_ack = self
            .external_action_ack
            .lock()
            .ok()
            .and_then(|ack| ack.clone());
        let budget_stop = self.budget_stop.lock().ok().and_then(|stop| stop.clone());
        let (reply, status, task_outcome) = match (&outcome.status, budget_stop) {
            // The budget hook stopped the run, or the mandate's balance
            // could not cover the next call.
            (sdk::RunStatus::Stopped { reason }, Some((reply, outcome)))
                if reason == GOAL_BUDGET_EXHAUSTED =>
            {
                (reply, TaskStatus::Completed, outcome)
            }
            (sdk::RunStatus::ProviderFailed { error }, Some((reply, outcome)))
                if error.contains(MANDATE_BUDGET_EXHAUSTED) =>
            {
                (reply, TaskStatus::Completed, outcome)
            }
            _ => (reply, status, task_outcome),
        };
        let (reply, status, task_outcome) = match (&outcome.status, external_ack) {
            (
                sdk::RunStatus::ProviderFailed { .. } | sdk::RunStatus::LimitReached { .. },
                Some(ack),
            )
            | (sdk::RunStatus::Stopped { .. }, Some(ack)) => (
                crate::turn::tool_dispatch::user_facing_external_action_ack(&ack),
                TaskStatus::Completed,
                TaskOutcome::Succeeded,
            ),
            _ => (reply, status, task_outcome),
        };
        // The owner declined part of what the run set out to do. However the
        // run ended, and whatever acknowledged it (a tool's finish, or an
        // external action's ack on a limit or provider stop), it never
        // succeeds over that decline.
        let task_outcome = match task_outcome {
            TaskOutcome::Succeeded if !outcome.state.declined.is_empty() => TaskOutcome::Partial,
            task_outcome => task_outcome,
        };
        if self.run_failure().is_some() {
            (reply, TaskStatus::Failed, TaskOutcome::Failed)
        } else {
            (reply, status, task_outcome)
        }
    }
}
