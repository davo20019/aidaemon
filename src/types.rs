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
