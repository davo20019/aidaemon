//! Live progress of a turn: what tools and the agent loop report while they
//! work, and the channel a tool call's progress goes to.

use tokio::sync::mpsc;

use crate::semantics::ToolCallContext;

/// Status updates emitted by tools and the agent loop for live feedback to the user.
#[derive(Debug, Clone)]
pub enum StatusUpdate {
    /// A chunk of the reply as the model writes it, for app clients that
    /// stream it. Only a top-level turn sends it; the runner hands it to the
    /// turn observer and never to a chat surface, its live state or a
    /// parent's card. The finished reply still arrives as the turn's answer.
    ReplyText(String),
    /// A message the sender sent while the turn ran, taken into it: for app
    /// clients, which show it in the conversation once the run reads it.
    /// Like `ReplyText`, the runner hands it to the turn observer only (a
    /// chat surface already shows what its sender typed). `origin` is the
    /// platform it was sent from.
    MessageTaken { text: String, origin: String },
    /// Sent before each LLM call (skipped on iteration 0).
    Thinking(usize),
    /// Sent before each tool execution. `kind` is what every card shows;
    /// `name` and `summary` are the call's own words for clients that list
    /// calls. Built with [`StatusUpdate::tool_start`]: the variant is
    /// non-exhaustive so a new field changes one constructor, not every
    /// sender.
    #[non_exhaustive]
    ToolStart {
        name: String,
        summary: String,
        kind: aidaemon_core::card::ActivityKind,
        /// The call's arguments as an app shows them
        /// ([`crate::shown::call_input`]); empty when there are none to show.
        input: String,
    },
    /// Streaming output chunk from a tool (e.g., CLI agent progress).
    ToolProgress { name: String, chunk: String },
    /// A tool is paused until the user completes an explicit action, such as
    /// authorizing an OAuth connection in their browser. Unlike ordinary tool
    /// progress, channels must deliver this immediately and without truncation.
    UserActionRequired { message: String },
    /// Tool execution completed with a brief summary. Built with
    /// [`StatusUpdate::tool_complete`].
    #[non_exhaustive]
    ToolComplete {
        name: String,
        summary: String,
        /// The call's result as an app shows it
        /// ([`crate::shown::call_output`]); empty when there is none to show.
        output: String,
    },
    /// Tool can be cancelled with the given task_id.
    ToolCancellable { name: String, task_id: String },
    /// Periodic summary for long-running tasks (emitted every 5 minutes).
    ProgressSummary { elapsed_mins: u64, summary: String },
    /// Warning that soft iteration limit is approaching.
    IterationWarning { current: usize, threshold: usize },
    /// A new task plan was created.
    PlanCreated {
        plan_id: String,
        description: String,
        total_steps: usize,
    },
    /// A plan step has started executing.
    PlanStepStart {
        plan_id: String,
        step_index: usize,
        total_steps: usize,
        description: String,
    },
    /// A plan step completed successfully.
    PlanStepComplete {
        plan_id: String,
        step_index: usize,
        total_steps: usize,
        description: String,
        summary: Option<String>,
    },
    /// A plan step failed.
    PlanStepFailed {
        plan_id: String,
        step_index: usize,
        description: String,
        error: String,
    },
    /// The entire plan completed successfully.
    PlanComplete {
        plan_id: String,
        description: String,
        total_steps: usize,
        duration_secs: u64,
    },
    /// The plan was abandoned by user request.
    PlanAbandoned {
        plan_id: String,
        description: String,
    },
    /// The plan was revised with new/updated steps.
    PlanRevised {
        plan_id: String,
        description: String,
        reason: String,
        new_total_steps: usize,
    },
    /// The task plan as the single live card shows it: its rendering and
    /// its step count, which decides whether the card is kept.
    Checklist(aidaemon_core::card::PlanCard),
    /// A helper's record changed: it started, took a step or ended. The
    /// view is the runtime's record of it, so the live card and the card a
    /// turn leaves behind show the same thing.
    Helper(aidaemon_core::card::HelperView),
    /// The request finished in this turn, after helpers it waited on, as
    /// the runtime measured it and already worded for the owner. `by_me`:
    /// the turn finished work itself after a helper ended ("Finished by me
    /// after the helper: Drupal install" and that work's measured line);
    /// `total`: the request's whole time, from the owner's message to the
    /// answer ("Answer sent 10m 12s after your request (5:30:24 PM to
    /// 5:40:36 PM ET)."), when it adds to the helpers' own times.
    Finished {
        by_me: Option<String>,
        total: Option<String>,
    },
    /// A step ended by handing its work to something that keeps running: a
    /// helper (`ActivityKind::Delegate`) or a background job. The work has
    /// not finished, so the step must never read as finished. Built with
    /// [`StatusUpdate::step_handed_off`].
    #[non_exhaustive]
    StepHandedOff {
        kind: aidaemon_core::card::ActivityKind,
        /// The call's name as its [`Self::ToolStart`] gave it; empty when
        /// there was no such update.
        name: String,
        /// The call's own result (its launch line) as an app shows it
        /// ([`Self::showing_result`]); empty when there is none to show.
        output: String,
    },
    /// How many lines a long-running command has printed so far, sent at
    /// most every few seconds while it runs. A helper's card line shows the
    /// count, so a long count or install visibly moves. Never the lines
    /// themselves: a command's output is its own data and can carry a
    /// credential (see [`aidaemon_core::card::Step::Output`]).
    CommandOutput {
        name: String,
        lines: u64,
        /// Its latest lines, secrets redacted, for the owner's own app only
        /// ([`command_output_tail`]); a chat's card reads `lines` and never
        /// this. `None` where the owner's app is not the reader.
        tail: Option<String>,
    },
}

/// The last lines of a command's output for the owner's own app: at most
/// 20 lines and 2,000 characters, secrets redacted.
pub fn command_output_tail(stdout: &[u8], stderr: &[u8]) -> Option<String> {
    let mut text = String::from_utf8_lossy(stdout).into_owned();
    if !stderr.is_empty() {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&String::from_utf8_lossy(stderr));
    }
    let lines: Vec<&str> = text.lines().collect();
    let tail = lines[lines.len().saturating_sub(20)..].join("\n");
    let tail: String = tail
        .chars()
        .rev()
        .take(2_000)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let tail = crate::sanitize::redact_for_owner(&tail);
    (!tail.trim().is_empty()).then_some(tail)
}

impl StatusUpdate {
    /// A call starting, with nothing to show of its arguments yet
    /// ([`Self::showing_arguments`] adds them).
    pub fn tool_start(
        name: impl Into<String>,
        summary: impl Into<String>,
        kind: aidaemon_core::card::ActivityKind,
    ) -> Self {
        Self::ToolStart {
            name: name.into(),
            summary: summary.into(),
            kind,
            input: String::new(),
        }
    }

    /// A call finished, with nothing to show of its result yet
    /// ([`Self::showing_result`] adds it).
    pub fn tool_complete(name: impl Into<String>, summary: impl Into<String>) -> Self {
        Self::ToolComplete {
            name: name.into(),
            summary: summary.into(),
            output: String::new(),
        }
    }

    /// A step that handed its work off ([`Self::StepHandedOff`]), named as
    /// its call started; [`Self::showing_result`] adds its launch result.
    pub fn step_handed_off(
        name: impl Into<String>,
        kind: aidaemon_core::card::ActivityKind,
    ) -> Self {
        Self::StepHandedOff {
            kind,
            name: name.into(),
            output: String::new(),
        }
    }

    /// A started call that shows its raw JSON arguments the way history
    /// does ([`crate::shown::call_input`]). Other updates are unchanged.
    pub fn showing_arguments(mut self, arguments: &str) -> Self {
        if let Self::ToolStart { input, .. } = &mut self {
            *input = crate::shown::call_input(&serde_json::Value::String(arguments.to_string()));
        }
        self
    }

    /// A finished or handed-off call that shows its result the way history
    /// does ([`crate::shown::call_output`]). Other updates are unchanged.
    pub fn showing_result(mut self, result: &str) -> Self {
        if let Self::ToolComplete { output, .. } | Self::StepHandedOff { output, .. } = &mut self {
            *output = crate::shown::call_output(result);
        }
        self
    }

    /// Whether a channel must deliver this update immediately instead of
    /// treating it as optional, rate-limited progress.
    pub fn requires_immediate_delivery(&self) -> bool {
        // A token budget extended on productive progress is the run's own
        // bookkeeping, never a status of its own: the owner acts on nothing
        // (live 2026-11-13: seven "Auto-extended token budget" lines in one
        // turn). It stays in the logs and the run's decision records.
        matches!(self, Self::UserActionRequired { .. })
    }
}

/// The channel a call's live progress goes to, carried in
/// [`ToolCallContext::extensions`].
#[derive(Clone)]
pub struct ToolStatus(pub mpsc::Sender<StatusUpdate>);

/// Where this call's live progress goes, if anywhere.
pub fn status_sender(facts: &ToolCallContext) -> Option<mpsc::Sender<StatusUpdate>> {
    facts
        .extensions
        .get::<ToolStatus>()
        .map(|status| status.0.clone())
}

/// `facts` with this call's progress channel added.
pub fn with_status(
    mut facts: ToolCallContext,
    status_tx: Option<mpsc::Sender<StatusUpdate>>,
) -> ToolCallContext {
    if let Some(status_tx) = status_tx {
        facts.extensions.insert(ToolStatus(status_tx));
    }
    facts
}

#[cfg(test)]
mod tests {
    use super::StatusUpdate;

    #[test]
    fn url_in_ordinary_progress_does_not_force_chat_delivery() {
        let progress = StatusUpdate::ToolProgress {
            name: "agent delegation".to_string(),
            chunk: "Checking https://blog.example.test/posts/example/".to_string(),
        };
        assert!(!progress.requires_immediate_delivery());

        let action = StatusUpdate::UserActionRequired {
            message: "[Authorize](https://auth.example.test/)".to_string(),
        };
        assert!(action.requires_immediate_delivery());
    }
}

#[cfg(test)]
mod command_output_tail_tests {
    use super::command_output_tail;

    #[test]
    fn the_owner_sees_the_last_lines_with_secrets_redacted() {
        let mut out = String::new();
        for n in 1..=30 {
            out.push_str(&format!("line {n}\n"));
        }
        out.push_str("token=sk-synthetic-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
        let tail = command_output_tail(out.as_bytes(), b"").unwrap();
        assert!(!tail.contains("line 1\n"), "{tail}");
        assert!(tail.contains("line 30"));
        assert!(!tail.contains("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"), "{tail}");
        assert_eq!(command_output_tail(b"", b""), None);
    }
}
