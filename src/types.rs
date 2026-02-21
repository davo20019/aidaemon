use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Visibility level of the channel the message originated from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelVisibility {
    /// Direct message (1-on-1). Full memory, no restrictions.
    Private,
    /// Small private group (e.g., Telegram group, Slack MPIM). Cautious with sensitive info.
    PrivateGroup,
    /// Public channel visible to many users. No personal memory injected.
    Public,
    /// Untrusted public platform (Twitter, public APIs). Hardened security, minimal memory.
    PublicExternal,
    /// Internal/system-initiated (scheduler, triggers, sub-agents default). Full memory.
    Internal,
}

/// Privacy level for facts stored in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactPrivacy {
    /// Accessible everywhere (user's name, timezone, general preferences).
    Global,
    /// Accessible only in originating channel + DMs.
    Channel,
    /// DM-only, never shared in channels, never hinted.
    Private,
}

impl std::fmt::Display for FactPrivacy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactPrivacy::Global => write!(f, "global"),
            FactPrivacy::Channel => write!(f, "channel"),
            FactPrivacy::Private => write!(f, "private"),
        }
    }
}

impl FactPrivacy {
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "global" => Self::Global,
            "channel" => Self::Channel,
            "private" => Self::Private,
            _ => Self::Global,
        }
    }
}

impl std::fmt::Display for ChannelVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelVisibility::Private => write!(f, "private"),
            ChannelVisibility::PrivateGroup => write!(f, "private_group"),
            ChannelVisibility::Public => write!(f, "public"),
            ChannelVisibility::PublicExternal => write!(f, "public_external"),
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
            "public_external" => Self::PublicExternal,
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
    /// Stable channel identifier for memory scoping (e.g., "slack:C12345", "telegram:67890")
    pub channel_id: Option<String>,
    /// Display name of the message sender, if resolved (e.g., "Alice", "Bob Smith")
    pub sender_name: Option<String>,
    /// Platform-qualified sender ID (e.g., "slack:U04S8KSS932", "telegram:123456")
    pub sender_id: Option<String>,
    /// Display names of members in the channel (for group channels; empty for DMs)
    pub channel_member_names: Vec<String>,
    /// User ID → display name lookup (e.g., "U04S8KSS932" → "Alice") for resolving IDs in facts
    pub user_id_map: HashMap<String, String>,
    /// Whether this session is explicitly trusted (e.g., a trusted scheduled task).
    /// Trusted sessions can bypass terminal command approval for allowed commands.
    /// This must be set explicitly by the scheduler — never derived from session ID strings.
    pub trusted: bool,
}

impl ChannelContext {
    /// Default context for private DMs.
    /// Used by integration tests.
    #[cfg(test)]
    pub fn private(platform: &str) -> Self {
        Self {
            visibility: ChannelVisibility::Private,
            platform: platform.to_string(),
            channel_name: None,
            channel_id: None,
            sender_name: None,
            sender_id: None,
            channel_member_names: vec![],
            user_id_map: HashMap::new(),
            trusted: false,
        }
    }

    /// Context for internal/system-initiated sessions (scheduler, triggers).
    pub fn internal() -> Self {
        Self {
            visibility: ChannelVisibility::Internal,
            platform: "internal".to_string(),
            channel_name: None,
            channel_id: None,
            sender_name: None,
            sender_id: None,
            channel_member_names: vec![],
            user_id_map: HashMap::new(),
            trusted: false,
        }
    }

    /// Context for trusted internal sessions (e.g., explicitly trusted scheduled tasks).
    pub fn internal_trusted() -> Self {
        Self {
            visibility: ChannelVisibility::Internal,
            platform: "internal".to_string(),
            channel_name: None,
            channel_id: None,
            sender_name: None,
            sender_id: None,
            channel_member_names: vec![],
            user_id_map: HashMap::new(),
            trusted: true,
        }
    }

    /// Whether deeply personal memory (goals, patterns, profile) should be injected.
    /// Facts and episodes now use channel-scoped retrieval instead.
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

/// Distinguishes different kinds of approval requests so channels can render
/// appropriate buttons (e.g., Allow Once / Deny for commands vs Confirm / Cancel
/// for scheduled-goal confirmations).
#[derive(Debug, Clone, Default)]
pub enum ApprovalKind {
    /// Standard command approval (Allow Once / Allow Session / Allow Always / Deny).
    #[default]
    Command,
    /// Scheduled-goal confirmation (Confirm / Cancel).
    GoalConfirmation,
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
    /// Token budget was auto-extended due to productive progress.
    BudgetExtended {
        old_budget: i64,
        new_budget: i64,
        extension: usize,
        max_extensions: usize,
    },
}

/// The kind of media being sent.
#[allow(dead_code)]
pub enum MediaKind {
    /// An in-memory photo (e.g. screenshot).
    Photo { data: Vec<u8> },
    /// A file on disk to send as a document.
    Document { file_path: String, filename: String },
}

/// A media message to be sent through a channel.
pub struct MediaMessage {
    pub session_id: String,
    pub caption: String,
    pub kind: MediaKind,
}
