//! Receipt-based completion check as an `on_stop` hook.

use std::collections::HashMap;

use aidaemon_core::{Hook, HookContext, Receipt, ReceiptStatus, Role, StopDecision};
use async_trait::async_trait;

use crate::{last_user_message_index, HARNESS_MARKER};

/// One structural requirement on the receipts of the current user turn
/// (everything after the last message the user, not the harness, wrote).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requirement {
    /// At least one call succeeded. Use for task agents where finishing
    /// without doing anything is a failure, not for Q&A agents.
    RequireAnySuccess,
    /// For each distinct tool + arguments, the latest attempt must not have
    /// ended failed, timed out or indeterminate.
    RequireNoUnresolvedFailures,
}

const REJECTION_PREFIX: &str = "[harness] completion check:";

/// Rejects a final answer when the receipts do not support it, at most
/// `max_rejections` times per user turn (default 2), then accepts so a run
/// can never loop on this hook alone.
///
/// The rejection count is derived from the transcript (its own rejection
/// messages since the last user message), so it survives resume in another
/// process and needs no per-run state.
#[derive(Debug, Clone)]
pub struct ReceiptCompletionCheck {
    requirements: Vec<Requirement>,
    max_rejections: usize,
}

impl Default for ReceiptCompletionCheck {
    fn default() -> Self {
        Self::new([Requirement::RequireNoUnresolvedFailures])
    }
}

impl ReceiptCompletionCheck {
    pub fn new(requirements: impl IntoIterator<Item = Requirement>) -> Self {
        Self {
            requirements: requirements.into_iter().collect(),
            max_rejections: 2,
        }
    }

    pub fn max_rejections(mut self, max: usize) -> Self {
        self.max_rejections = max;
        self
    }

    /// What is missing, as model-facing lines; empty when satisfied.
    /// `receipts` pairs each receipt of the turn with the output text the
    /// model saw for it.
    fn problems(&self, receipts: &[(&Receipt, &str)]) -> Vec<String> {
        let mut problems = Vec::new();
        for requirement in &self.requirements {
            match requirement {
                Requirement::RequireAnySuccess => {
                    if !receipts.iter().any(|(r, _)| r.succeeded()) {
                        let tried = if receipts.is_empty() {
                            "no tool was called".to_string()
                        } else {
                            format!("{} call(s) were made and none succeeded", receipts.len())
                        };
                        problems.push(format!(
                            "No tool call has succeeded since the user's last message ({tried}). \
                             Do the requested work with your tools before finishing, or state \
                             plainly that you could not do it and why."
                        ));
                    }
                }
                Requirement::RequireNoUnresolvedFailures => {
                    for (r, output) in unresolved(receipts) {
                        let reason = match r.reason.as_deref() {
                            Some(reason) => reason.to_string(),
                            None => brief(output),
                        };
                        problems.push(format!(
                            "`{}` with arguments {} ended {} ({reason}) and was not retried \
                             successfully. Retry it, verify its effect, or tell the user it \
                             did not succeed.",
                            r.tool,
                            r.arguments,
                            status_word(r.status)
                        ));
                    }
                }
            }
        }
        problems
    }
}

/// The first line of an output, capped, for quoting in a rejection.
fn brief(output: &str) -> String {
    let line = output
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("no output");
    let mut text: String = line.chars().take(200).collect();
    if line.chars().count() > 200 {
        text.push_str("...");
    }
    text
}

fn status_word(status: ReceiptStatus) -> &'static str {
    match status {
        ReceiptStatus::Failed => "failed",
        ReceiptStatus::TimedOut => "timed out",
        ReceiptStatus::Indeterminate => "indeterminate (it may or may not have run)",
        _ => "unsuccessfully",
    }
}

/// The latest receipt per tool + arguments, kept when it is unresolved.
fn unresolved<'a>(receipts: &[(&'a Receipt, &'a str)]) -> Vec<(&'a Receipt, &'a str)> {
    let mut latest: HashMap<(String, String), usize> = HashMap::new();
    for (i, (r, _)) in receipts.iter().enumerate() {
        latest.insert((r.tool.clone(), r.arguments.to_string()), i);
    }
    let mut indices: Vec<usize> = latest.into_values().collect();
    indices.sort_unstable();
    indices
        .into_iter()
        .map(|i| receipts[i])
        .filter(|(r, _)| {
            matches!(
                r.status,
                ReceiptStatus::Failed | ReceiptStatus::TimedOut | ReceiptStatus::Indeterminate
            )
        })
        .collect()
}

#[async_trait]
impl Hook for ReceiptCompletionCheck {
    async fn on_stop(&self, ctx: &HookContext<'_>, _final_text: &str) -> StopDecision {
        let start = last_user_message_index(ctx.messages).map_or(0, |i| i + 1);
        let tail = &ctx.messages[start..];
        let rejections = tail
            .iter()
            .filter(|m| m.role == Role::User && m.content.starts_with(REJECTION_PREFIX))
            .count();
        if rejections >= self.max_rejections {
            return StopDecision::Accept;
        }
        // The kernel appends exactly one tool message and one receipt per
        // call, together, so the last N receipts belong to the N tool
        // messages of this user turn (call ids alone may repeat across turns).
        let in_turn = tail.iter().filter(|m| m.role == Role::Tool).count();
        let first = ctx.receipts.len().saturating_sub(in_turn);
        let outputs = tail
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| m.content.as_str());
        let receipts: Vec<(&Receipt, &str)> = ctx.receipts[first..].iter().zip(outputs).collect();
        let problems = self.problems(&receipts);
        if problems.is_empty() {
            return StopDecision::Accept;
        }
        debug_assert!(REJECTION_PREFIX.starts_with(HARNESS_MARKER));
        let mut message = format!(
            "{REJECTION_PREFIX} the run is not finished yet (check {} of {}).",
            rejections + 1,
            self.max_rejections
        );
        for problem in problems {
            message.push_str("\n- ");
            message.push_str(&problem);
        }
        StopDecision::Continue { message }
    }
}
