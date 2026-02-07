/// Visibility level of the channel the message originated from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelVisibility {
    /// Direct message (1-on-1). Full memory, no restrictions.
    Private,
    /// Small private group (e.g., Telegram group, Slack MPIM). Cautious with sensitive info.
    PrivateGroup,
    /// Public channel visible to many users. No personal memory injected.
    Public,
    /// Internal/system-initiated (scheduler, triggers, sub-agents default). Full memory.
    Internal,
}

impl std::fmt::Display for ChannelVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelVisibility::Private => write!(f, "private"),
            ChannelVisibility::PrivateGroup => write!(f, "private_group"),
            ChannelVisibility::Public => write!(f, "public"),
            ChannelVisibility::Internal => write!(f, "internal"),
        }
    }
}

impl ChannelVisibility {
    /// Parse from a string (e.g., from tool args). Falls back to `Internal`.
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "private" => Self::Private,
            "private_group" => Self::PrivateGroup,
            "public" => Self::Public,
            "internal" => Self::Internal,
            _ => Self::Internal,
        }
    }
}

/// Context about the channel/conversation where a message originated.
#[derive(Debug, Clone)]
pub struct ChannelContext {
    /// How visible is this conversation?
    pub visibility: ChannelVisibility,
    /// Platform name: "telegram", "slack", "discord", "internal"
    pub platform: String,
    /// Human-readable channel name, if available (e.g., "#general", "Team Chat")
    pub channel_name: Option<String>,
}

impl ChannelContext {
    /// Default context for private DMs.
    pub fn private(platform: &str) -> Self {
        Self {
            visibility: ChannelVisibility::Private,
            platform: platform.to_string(),
            channel_name: None,
        }
    }

    /// Context for internal/system-initiated sessions (scheduler, triggers).
    pub fn internal() -> Self {
        Self {
            visibility: ChannelVisibility::Internal,
            platform: "internal".to_string(),
            channel_name: None,
        }
    }

    /// Whether personal memory (facts, episodes, goals, etc.) should be injected.
    pub fn should_inject_personal_memory(&self) -> bool {
        matches!(
            self.visibility,
            ChannelVisibility::Private | ChannelVisibility::Internal
        )
    }
}

/// Role assigned to a user based on config (owner_ids).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserRole {
    /// Full access — listed in `users.owner_ids`.
    Owner,
    /// Allowed by channel allowlist but not an owner.
    Guest,
    /// Non-whitelisted user — conversational access only, no tools.
    Public,
}

impl std::fmt::Display for UserRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UserRole::Owner => write!(f, "owner"),
            UserRole::Guest => write!(f, "guest"),
            UserRole::Public => write!(f, "public"),
        }
    }
}

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
