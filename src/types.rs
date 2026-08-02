use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Least-privilege access granted to one collaborator for one project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceAccessLevel {
    Read,
    Edit,
}

impl std::fmt::Display for WorkspaceAccessLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Edit => write!(f, "edit"),
        }
    }
}

/// An explicit, expiring workspace grant. Platform workspace, channel, and
/// user IDs are all required so a grant cannot cross Slack workspaces or chats.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceGrant {
    pub platform: String,
    pub workspace_id: String,
    pub channel_id: String,
    pub user_id: String,
    pub project_root: String,
    pub access: WorkspaceAccessLevel,
    pub expires_at: DateTime<Utc>,
}

impl std::fmt::Debug for WorkspaceGrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceGrant")
            .field("platform", &self.platform)
            .field("workspace_id", &self.workspace_id)
            .field("channel_id", &self.channel_id)
            .field("user_id", &self.user_id)
            .field("project_root", &"[SCOPED WORKSPACE]")
            .field("access", &self.access)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

impl WorkspaceGrant {
    pub const READ_TOOLS: [&'static str; 2] = ["read_file", "search_files"];
    pub const EDIT_TOOLS: [&'static str; 2] = ["write_file", "edit_file"];

    pub fn is_active(&self) -> bool {
        self.expires_at > Utc::now()
    }

    pub fn allows_tool(&self, tool_name: &str) -> bool {
        Self::READ_TOOLS.contains(&tool_name)
            || (self.access == WorkspaceAccessLevel::Edit && Self::EDIT_TOOLS.contains(&tool_name))
    }

    pub fn project_name(&self) -> &str {
        Path::new(&self.project_root)
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or("project")
    }
}

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelContext {
    /// How visible is this conversation?
    pub visibility: ChannelVisibility,
    /// Platform name: "telegram", "slack", "discord", "internal"
    pub platform: String,
    /// Human-readable channel name, if available (e.g., "#general", "Team Chat")
    pub channel_name: Option<String>,
    /// Stable channel identifier for memory scoping (e.g., "slack:C12345", "telegram:67890")
    pub channel_id: Option<String>,
    /// Stable platform-workspace identifier (e.g., a Slack team ID). Required
    /// for scoped grants so identically-shaped channel IDs cannot cross tenants.
    pub workspace_id: Option<String>,
    /// Display name of the message sender, if resolved (e.g., "Alice", "Bob Smith")
    pub sender_name: Option<String>,
    /// Platform-qualified sender ID (e.g., "slack:U04S8KSS932", "telegram:123456")
    pub sender_id: Option<String>,
    /// Display names of members in the channel (for group channels; empty for DMs)
    pub channel_member_names: Vec<String>,
    /// User ID → display name lookup (e.g., "U04S8KSS932" → "Alice") for resolving IDs in facts
    pub user_id_map: HashMap<String, String>,
    /// Explicit, expiring project access for this exact sender and channel.
    /// Absent by default; never inferred from conversational text.
    pub workspace_grant: Option<WorkspaceGrant>,
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
            workspace_id: None,
            sender_name: None,
            sender_id: None,
            channel_member_names: vec![],
            user_id_map: HashMap::new(),
            workspace_grant: None,
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
            workspace_id: None,
            sender_name: None,
            sender_id: None,
            channel_member_names: vec![],
            user_id_map: HashMap::new(),
            workspace_grant: None,
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
            workspace_id: None,
            sender_name: None,
            sender_id: None,
            channel_member_names: vec![],
            user_id_map: HashMap::new(),
            workspace_grant: None,
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

    /// Return a grant only when every immutable binding still matches this
    /// exact guest/group context and the expiry has not passed.
    pub fn active_workspace_grant(&self, user_role: UserRole) -> Option<&WorkspaceGrant> {
        let grant = self.workspace_grant.as_ref()?;
        if user_role != UserRole::Guest
            || self.visibility != ChannelVisibility::PrivateGroup
            || !grant.is_active()
            || grant.platform != self.platform
            || self.workspace_id.as_deref() != Some(grant.workspace_id.as_str())
        {
            return None;
        }

        let channel_id = self
            .channel_id
            .as_deref()?
            .strip_prefix(&format!("{}:", self.platform))?;
        let sender_id = self
            .sender_id
            .as_deref()?
            .strip_prefix(&format!("{}:", self.platform))?;

        (channel_id == grant.channel_id && sender_id == grant.user_id).then_some(grant)
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

impl UserRole {
    /// Only explicit owner traffic is allowed to update durable owner memory.
    pub fn can_persist_owner_memory(self) -> bool {
        matches!(self, Self::Owner)
    }
}

/// Distinguishes different kinds of approval requests so channels can render
/// appropriate buttons (e.g., Allow Once / Deny for commands vs Confirm / Cancel
/// for scheduled-goal confirmations).
#[derive(Debug, Clone, Copy, Default)]
pub enum ApprovalKind {
    /// Standard command approval (Allow Once / Allow Session / Allow Always / Deny).
    #[default]
    Command,
    /// Point-of-action approval that can authorize only this invocation.
    /// Channels must not offer session or persistent approval buttons.
    CommandOnce,
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
    /// A rendered task-plan checklist to display on the single live surface.
    /// Replaces the separate checklist message previously posted by
    /// `track_requirements` directly through the hub.
    Checklist { text: String },
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
    /// Optional one-shot used to report the ACTUAL delivery outcome back to the
    /// enqueuing tool. `Some(tx)` means the sender wants honest confirmation:
    /// `media_listener` replies `Ok(())` only when the channel's `send_media`
    /// succeeded, and `Err(reason)` when delivery failed / no channel was found /
    /// the message was shed under overload. `None` keeps the legacy
    /// fire-and-forget behavior (the sender does not care about the outcome).
    ///
    /// `oneshot::Sender` is neither `Clone` nor `Default`; `MediaMessage` is moved
    /// (never cloned) through the media channel, so no derive is affected.
    pub result_tx: Option<tokio::sync::oneshot::Sender<Result<(), String>>>,
}

#[cfg(test)]
mod workspace_grant_tests {
    use super::*;

    fn grant(expires_at: DateTime<Utc>) -> WorkspaceGrant {
        WorkspaceGrant {
            platform: "slack".to_string(),
            workspace_id: "T_TEST".to_string(),
            channel_id: "C_PRIVATE".to_string(),
            user_id: "U_GUEST".to_string(),
            project_root: "/private/project".to_string(),
            access: WorkspaceAccessLevel::Edit,
            expires_at,
        }
    }

    fn context(workspace_grant: WorkspaceGrant) -> ChannelContext {
        ChannelContext {
            visibility: ChannelVisibility::PrivateGroup,
            platform: "slack".to_string(),
            channel_name: Some("family".to_string()),
            channel_id: Some("slack:C_PRIVATE".to_string()),
            workspace_id: Some("T_TEST".to_string()),
            sender_name: Some("Collaborator".to_string()),
            sender_id: Some("slack:U_GUEST".to_string()),
            channel_member_names: vec![],
            user_id_map: HashMap::new(),
            workspace_grant: Some(workspace_grant),
            trusted: false,
        }
    }

    #[test]
    fn workspace_grant_requires_exact_guest_group_binding() {
        let active = grant(Utc::now() + chrono::Duration::hours(1));
        let ctx = context(active);
        assert!(ctx.active_workspace_grant(UserRole::Guest).is_some());

        let mut wrong_channel = ctx.clone();
        wrong_channel.channel_id = Some("slack:C_OTHER".to_string());
        assert!(wrong_channel
            .active_workspace_grant(UserRole::Guest)
            .is_none());

        let mut wrong_workspace = ctx.clone();
        wrong_workspace.workspace_id = Some("T_OTHER".to_string());
        assert!(wrong_workspace
            .active_workspace_grant(UserRole::Guest)
            .is_none());

        let mut wrong_sender = ctx.clone();
        wrong_sender.sender_id = Some("slack:U_OTHER".to_string());
        assert!(wrong_sender
            .active_workspace_grant(UserRole::Guest)
            .is_none());

        assert!(ctx.active_workspace_grant(UserRole::Public).is_none());
        assert!(ctx.active_workspace_grant(UserRole::Owner).is_none());
    }

    #[test]
    fn expired_grants_and_dangerous_tools_are_denied() {
        let expired = context(grant(Utc::now() - chrono::Duration::seconds(1)));
        assert!(expired.active_workspace_grant(UserRole::Guest).is_none());

        let mut read_grant = grant(Utc::now() + chrono::Duration::hours(1));
        read_grant.access = WorkspaceAccessLevel::Read;
        assert!(read_grant.allows_tool("read_file"));
        assert!(read_grant.allows_tool("search_files"));
        assert!(!read_grant.allows_tool("write_file"));
        for dangerous in [
            "terminal",
            "run_command",
            "cli_agent",
            "spawn_agent",
            "send_file",
            "git_commit",
            "manage_config",
            "manage_memories",
            "browser",
            "computer_use",
        ] {
            assert!(!read_grant.allows_tool(dangerous), "{dangerous}");
        }
    }

    #[test]
    fn workspace_grant_debug_redacts_project_root() {
        let grant = grant(Utc::now() + chrono::Duration::hours(1));
        let rendered = format!("{grant:?}");
        assert!(!rendered.contains("/private/project"));
        assert!(rendered.contains("SCOPED WORKSPACE"));
    }

    #[test]
    fn workspace_grant_round_trips_through_toml() {
        let expected = vec![grant(Utc::now() + chrono::Duration::hours(1))];
        let encoded = toml::Value::try_from(&expected).expect("serialize grant");
        let decoded: Vec<WorkspaceGrant> = encoded.try_into().expect("deserialize grant");
        assert_eq!(decoded, expected);
    }
}
