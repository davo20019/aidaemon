/// Response to an approval request from the user.
#[derive(Debug, Clone)]
pub enum ApprovalResponse {
    /// Allow this specific command only
    AllowOnce,
    /// Allow this command prefix for the current session (resets on restart)
    AllowSession,
    /// Allow this command prefix forever (persists to database)
    AllowAlways,
    /// Deny the command
    Deny,
}

/// Status updates emitted by tools and the agent loop for live feedback to the user.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum StatusUpdate {
    /// Sent before each LLM call (skipped on iteration 0).
    Thinking(usize),
    /// Sent before each tool execution.
    ToolStart { name: String, summary: String },
    /// Streaming output chunk from a tool (e.g., CLI agent progress).
    ToolProgress { name: String, chunk: String },
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
}

/// The kind of media being sent.
#[allow(dead_code)]
pub enum MediaKind {
    /// An in-memory photo (e.g. screenshot).
    Photo { data: Vec<u8> },
    /// A file on disk to send as a document.
    Document {
        file_path: String,
        filename: String,
    },
}

/// A media message to be sent through a channel.
pub struct MediaMessage {
    pub session_id: String,
    pub caption: String,
    pub kind: MediaKind,
}
