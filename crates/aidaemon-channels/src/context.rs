//! Where a message came from: the conversation's audience, platform, sender
//! and any scoped project grant. Internal to aidaemon (the daemon's channel
//! context), no stability promise.

use std::collections::HashMap;
use std::path::Path;

use aidaemon_core::{ChannelVisibility, UserRole};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
        assert!(
            wrong_channel
                .active_workspace_grant(UserRole::Guest)
                .is_none()
        );

        let mut wrong_workspace = ctx.clone();
        wrong_workspace.workspace_id = Some("T_OTHER".to_string());
        assert!(
            wrong_workspace
                .active_workspace_grant(UserRole::Guest)
                .is_none()
        );

        let mut wrong_sender = ctx.clone();
        wrong_sender.sender_id = Some("slack:U_OTHER".to_string());
        assert!(
            wrong_sender
                .active_workspace_grant(UserRole::Guest)
                .is_none()
        );

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
