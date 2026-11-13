//! Live progress of a turn: what tools and the agent loop report while they
//! work, and the channel a tool call's progress goes to.

use tokio::sync::mpsc;

use crate::semantics::ToolCallContext;

/// Status updates emitted by tools and the agent loop for live feedback to the user.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum StatusUpdate {
    /// Sent before each LLM call (skipped on iteration 0).
    Thinking(usize),
    /// Sent before each tool execution. `kind` is what every card shows;
    /// `name` and `summary` are the call's own words for clients that list
    /// calls.
    ToolStart {
        name: String,
        summary: String,
        kind: aidaemon_core::card::ActivityKind,
    },
    /// Streaming output chunk from a tool (e.g., CLI agent progress).
    ToolProgress { name: String, chunk: String },
    /// A tool is paused until the user completes an explicit action, such as
    /// authorizing an OAuth connection in their browser. Unlike ordinary tool
    /// progress, channels must deliver this immediately and without truncation.
    UserActionRequired { message: String },
    /// Tool execution completed with a brief summary.
    ToolComplete { name: String, summary: String },
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
    /// Token budget was auto-extended due to productive progress.
    BudgetExtended {
        old_budget: i64,
        new_budget: i64,
        extension: usize,
        max_extensions: usize,
    },
    /// A rendered task-plan checklist to display on the single live surface.
    /// Replaces the separate checklist message previously posted by
    /// `track_requirements` directly through the hub.
    Checklist { text: String },
    /// A helper's record changed: it started, took a step or ended. The
    /// view is the runtime's record of it, so the live card and the card a
    /// turn leaves behind show the same thing.
    Helper(aidaemon_core::card::HelperView),
    /// A step ended by handing its work to something that keeps running: a
    /// helper (`ActivityKind::Delegate`) or a background job. The work has
    /// not finished, so the step must never read as finished.
    StepHandedOff {
        kind: aidaemon_core::card::ActivityKind,
    },
    /// The latest line a long-running command printed (secrets redacted),
    /// sent at most every few seconds while it runs. A helper's card line
    /// shows it, so a long count or install visibly moves.
    CommandOutput { name: String, line: String },
}

impl StatusUpdate {
    /// Whether a channel must deliver this update immediately instead of
    /// treating it as optional, rate-limited progress.
    pub fn requires_immediate_delivery(&self) -> bool {
        matches!(
            self,
            Self::UserActionRequired { .. } | Self::BudgetExtended { .. }
        )
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
