//! Text the runner shows: live status, approval prompts, run endings.

use std::collections::HashMap;

use aidaemon_core::{
    Event, FinishOutcome, Interruption, LimitKind, ReceiptStatus, RunOutcome, RunStatus,
};

const MAX_TOOL_LINES: usize = 8;

/// Live status of one run, rebuilt from its events.
#[derive(Debug, Default)]
pub(crate) struct Progress {
    tool_lines: Vec<(String, String)>,
    line_by_call: HashMap<String, usize>,
    stream: String,
}

impl Progress {
    /// Apply an event. Returns true if the rendered text changed.
    pub(crate) fn apply(&mut self, event: &Event) -> bool {
        match event {
            Event::TurnStarted { .. } => {
                let changed = !self.stream.is_empty();
                self.stream.clear();
                changed
            }
            Event::TextDelta { text, .. } => {
                self.stream.push_str(text);
                !text.is_empty()
            }
            Event::ToolStarted { call_id, tool } => {
                self.tool_lines
                    .push((call_id.clone(), format!("- Running `{tool}`")));
                if self.tool_lines.len() > MAX_TOOL_LINES {
                    self.tool_lines.remove(0);
                }
                self.reindex();
                true
            }
            Event::ToolFinished { receipt } => {
                let line = format!("- `{}`: {}", receipt.tool, receipt_word(receipt.status));
                match self.line_by_call.get(&receipt.call_id) {
                    Some(&i) => self.tool_lines[i].1 = line,
                    None => {
                        // Refused before dispatch: no ToolStarted was emitted.
                        self.tool_lines.push((receipt.call_id.clone(), line));
                        if self.tool_lines.len() > MAX_TOOL_LINES {
                            self.tool_lines.remove(0);
                        }
                        self.reindex();
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn reindex(&mut self) {
        self.line_by_call = self
            .tool_lines
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (id.clone(), i))
            .collect();
    }

    /// The status message body while the run is active.
    pub(crate) fn render(&self, max_chars: usize) -> String {
        let mut out = String::from("_Working..._");
        for (_, line) in &self.tool_lines {
            out.push('\n');
            out.push_str(line);
        }
        let stream = self.stream.trim();
        if !stream.is_empty() {
            out.push_str("\n\n");
            let room = max_chars.saturating_sub(out.chars().count() + 4);
            out.push_str(&tail_chars(stream, room));
        }
        truncate_chars(&out, max_chars)
    }

    /// The status message body after the run (the answer follows separately).
    pub(crate) fn render_final(&self, word: &str, max_chars: usize) -> String {
        let mut out = format!("_{word}_");
        for (_, line) in &self.tool_lines {
            out.push('\n');
            out.push_str(line);
        }
        truncate_chars(&out, max_chars)
    }
}

fn receipt_word(status: ReceiptStatus) -> &'static str {
    match status {
        ReceiptStatus::Succeeded => "done",
        ReceiptStatus::Failed => "failed",
        ReceiptStatus::Denied => "denied",
        ReceiptStatus::TimedOut => "timed out",
        ReceiptStatus::Cancelled => "cancelled",
        ReceiptStatus::UnknownTool => "unknown tool",
        ReceiptStatus::InvalidArguments => "invalid arguments",
        ReceiptStatus::Indeterminate => "outcome unknown",
        ReceiptStatus::Skipped => "skipped",
        ReceiptStatus::Answered => "answered without running",
    }
}

/// The first `max` characters, with an ellipsis if cut.
pub(crate) fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let keep = max.saturating_sub(3);
    let mut out: String = text.chars().take(keep).collect();
    out.push_str("...");
    out
}

/// The last `max` characters, with a leading ellipsis if cut.
fn tail_chars(text: &str, max: usize) -> String {
    let count = text.chars().count();
    if count <= max {
        return text.to_string();
    }
    let keep = max.saturating_sub(3);
    let tail: String = text.chars().skip(count - keep).collect();
    format!("...{tail}")
}

/// The Markdown body of an approval prompt: tool, pretty-printed arguments
/// (truncated) and the reason the run paused.
pub(crate) fn approval_text(interruption: &Interruption, max_argument_chars: usize) -> String {
    let args = serde_json::to_string_pretty(&interruption.call.arguments)
        .unwrap_or_else(|_| interruption.call.arguments.to_string());
    let args = truncate_chars(&args, max_argument_chars).replace("```", "'''");
    let reason = interruption.reason.trim();
    let mut out = format!(
        "**Approval needed**\n\nThe agent wants to run `{}`.",
        interruption.call.name
    );
    if !reason.is_empty() {
        out.push_str(&format!("\n\nReason: {reason}"));
    }
    out.push_str(&format!("\n\nArguments:\n```json\n{args}\n```"));
    out
}

/// The text-command fallback appended when the channel has no buttons.
pub(crate) fn approval_fallback(prompt_id: &str) -> String {
    format!("Reply `/approve {prompt_id}` to allow it or `/deny {prompt_id}` to refuse.")
}

/// A short, plain explanation for runs that did not complete.
pub(crate) fn ending_text(status: &RunStatus) -> String {
    match status {
        RunStatus::Completed => "Done.".to_string(),
        RunStatus::Interrupted { .. } => "Waiting for approval.".to_string(),
        RunStatus::LimitReached { limit } => {
            let what = match limit {
                LimitKind::Turns => "the maximum number of steps",
                LimitKind::Tokens => "its token budget",
                LimitKind::ToolCalls => "the maximum number of tool calls",
                LimitKind::WallClock => "its time limit",
                #[allow(unreachable_patterns)]
                _ => "a limit",
            };
            format!("I stopped because the run reached {what}.")
        }
        RunStatus::Stopped { reason } => {
            format!("I stopped: {}", truncate_chars(reason, 300))
        }
        RunStatus::Stuck { .. } => {
            "I stopped because I was repeating myself without making progress.".to_string()
        }
        RunStatus::Cancelled => "Stopped.".to_string(),
        // A tool ended the run; its reply, when it has one, is the answer.
        RunStatus::Finished { outcome, .. } => match outcome {
            FinishOutcome::Succeeded => "Done.".to_string(),
            FinishOutcome::Partial => "I did part of it; the rest is still underway.".to_string(),
            FinishOutcome::Blocked => "I can't go on without help.".to_string(),
            _ => "I stopped here.".to_string(),
        },
        RunStatus::ProviderFailed { error } => format!(
            "I could not get a reply from the model provider ({}). Please try again.",
            truncate_chars(error, 200)
        ),
        #[allow(unreachable_patterns)]
        _ => "The run ended unexpectedly.".to_string(),
    }
}

/// [`ending_text`] for a whole outcome: a run that stopped with unmet
/// requirements says what was left undone instead of "Done.".
pub(crate) fn outcome_ending_text(outcome: &RunOutcome) -> String {
    if outcome.is_completed() && !outcome.unmet().is_empty() {
        let mut text = "I stopped without finishing everything:".to_string();
        for line in outcome.unmet() {
            text.push_str("\n- ");
            text.push_str(&truncate_chars(line, 300));
        }
        return text;
    }
    ending_text(&outcome.status)
}

/// [`status_word`] for a whole outcome: a completed run whose objective
/// was not met reads "Partly done", like a partial finish.
pub(crate) fn outcome_word(outcome: &RunOutcome) -> &'static str {
    if outcome.is_completed() && !outcome.is_succeeded() {
        return "Partly done";
    }
    status_word(&outcome.status)
}

/// One word for the final status-message edit.
pub(crate) fn status_word(status: &RunStatus) -> &'static str {
    match status {
        RunStatus::Completed => "Done",
        RunStatus::Interrupted { .. } => "Waiting for approval",
        RunStatus::Cancelled => "Stopped",
        RunStatus::Finished { outcome, .. } => match outcome {
            FinishOutcome::Succeeded => "Done",
            FinishOutcome::Partial => "Partly done",
            FinishOutcome::Blocked => "Blocked",
            _ => "Ended",
        },
        _ => "Ended",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::ToolCall;
    use serde_json::json;

    #[test]
    fn approval_text_pretty_prints_and_truncates() {
        let interruption = Interruption {
            call: ToolCall::new(
                "c1",
                "deploy",
                json!({"target": "staging", "zz_notes": "x".repeat(500)}),
            ),
            reason: "deploys need a human".into(),
            kind: aidaemon_core::InterruptKind::default(),
            digest: None,
        };
        let text = approval_text(&interruption, 80);
        assert!(text.contains("`deploy`"));
        assert!(text.contains("Reason: deploys need a human"));
        assert!(text.contains("\"target\": \"staging\""));
        assert!(text.contains("..."));
        assert!(text.chars().count() < 300);
    }

    #[test]
    fn progress_tracks_tools_and_resets_stream_per_turn() {
        let mut p = Progress::default();
        assert!(p.apply(&Event::TextDelta {
            turn: 1,
            text: "thinking".into()
        }));
        assert!(p.render(200).contains("thinking"));
        assert!(p.apply(&Event::TurnStarted { turn: 2 }));
        assert!(!p.render(200).contains("thinking"));
        p.apply(&Event::ToolStarted {
            call_id: "c1".into(),
            tool: "search".into(),
        });
        assert!(p.render(200).contains("Running `search`"));
        assert!(p.render(10).chars().count() <= 10);
    }

    /// A run a tool finished reads by its outcome, never as an unexpected
    /// ending.
    #[test]
    fn finished_runs_read_by_their_outcome() {
        let finished = |outcome| RunStatus::Finished {
            outcome,
            tool: "file_report".into(),
        };
        assert_eq!(ending_text(&finished(FinishOutcome::Succeeded)), "Done.");
        assert_eq!(status_word(&finished(FinishOutcome::Succeeded)), "Done");
        assert_eq!(
            status_word(&finished(FinishOutcome::Partial)),
            "Partly done"
        );
        assert_eq!(status_word(&finished(FinishOutcome::Blocked)), "Blocked");
        for outcome in [FinishOutcome::Partial, FinishOutcome::Blocked] {
            assert!(!ending_text(&finished(outcome)).contains("unexpectedly"));
        }
    }

    /// Stopping is not succeeding: a completed run with unmet requirements
    /// never reads as "Done".
    #[test]
    fn completed_runs_with_unmet_requirements_read_as_partly_done() {
        let mut outcome = RunOutcome {
            status: RunStatus::Completed,
            final_text: None,
            state: aidaemon_core::RunState::new("a", "s", "go"),
        };
        assert_eq!(outcome_word(&outcome), "Done");
        assert_eq!(outcome_ending_text(&outcome), "Done.");
        outcome.state.unmet = vec!["checklist: [-] email the report".into()];
        assert_eq!(outcome_word(&outcome), "Partly done");
        let text = outcome_ending_text(&outcome);
        assert!(text.contains("without finishing") && text.contains("email the report"));
    }
}
