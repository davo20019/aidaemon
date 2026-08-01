use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::stream::StreamExt;
use futures::SinkExt;
use serde_json::Value;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

use super::commands::{shared_commands, CommandCategory, CommandDef};
use super::formatting::{
    build_help_text, markdown_to_slack_mrkdwn, sanitize_filename, split_message,
};
use crate::agent::Agent;
use crate::channels::{should_ignore_lightweight_interjection, ChannelHub, SessionMap};
use crate::tasks::{QueueOutcome, QueuedMessage, TaskRegistry};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::traits::{Channel, ChannelCapabilities, StateStore};
use crate::types::{ApprovalResponse, MediaKind, MediaMessage};
use crate::types::{
    ChannelContext, ChannelVisibility, StatusUpdate, UserRole, WorkspaceAccessLevel, WorkspaceGrant,
};

/// Maximum message length for Slack (actual limit is 40,000 but leave margin).
const MAX_MESSAGE_LEN: usize = 39_000;
const DEFAULT_WORKSPACE_GRANT_HOURS: i64 = 24;
const MAX_WORKSPACE_GRANT_HOURS: i64 = 168;

/// Serialize config mutations across every Slack bot in this process. Without
/// one shared lock, simultaneous grants/auto-claims could overwrite each other.
static SLACK_CONFIG_WRITE_LOCK: once_cell::sync::Lazy<Mutex<()>> =
    once_cell::sync::Lazy::new(|| Mutex::new(()));

enum WorkspaceGrantMutation {
    Upsert(WorkspaceGrant),
    Revoke {
        platform: String,
        workspace_id: String,
        channel_id: String,
        user_id: String,
    },
}

async fn load_workspace_grants_from_config(
    config_path: &std::path::Path,
) -> anyhow::Result<Vec<WorkspaceGrant>> {
    let _read_guard = SLACK_CONFIG_WRITE_LOCK.lock().await;
    let content = tokio::fs::read_to_string(config_path).await?;
    let doc: toml::Table = content.parse()?;
    let grants = doc
        .get("users")
        .and_then(toml::Value::as_table)
        .and_then(|users| users.get("workspace_grants"))
        .cloned()
        .map(|value| value.try_into())
        .transpose()?
        .unwrap_or_default();
    Ok(grants)
}

/// Revalidation can only preserve or reduce a queued request's original
/// authority. A later grant must never expand its project, access level, or
/// lifetime before that queued request begins executing.
fn retain_queued_workspace_grant(
    original: &WorkspaceGrant,
    current: &WorkspaceGrant,
) -> Option<WorkspaceGrant> {
    if !original.is_active()
        || !current.is_active()
        || original.platform != current.platform
        || original.workspace_id != current.workspace_id
        || original.channel_id != current.channel_id
        || original.user_id != current.user_id
        || original.project_root != current.project_root
    {
        return None;
    }

    let mut retained = original.clone();
    retained.access = if original.access == WorkspaceAccessLevel::Read
        || current.access == WorkspaceAccessLevel::Read
    {
        WorkspaceAccessLevel::Read
    } else {
        WorkspaceAccessLevel::Edit
    };
    retained.expires_at = std::cmp::min(original.expires_at, current.expires_at);
    retained.is_active().then_some(retained)
}

async fn revalidate_queued_workspace_grant(
    config_path: &std::path::Path,
    original: &WorkspaceGrant,
) -> anyhow::Result<Option<WorkspaceGrant>> {
    let grants = load_workspace_grants_from_config(config_path).await?;
    let mut matching = grants.iter().filter(|grant| {
        grant.platform == original.platform
            && grant.workspace_id == original.workspace_id
            && grant.channel_id == original.channel_id
            && grant.user_id == original.user_id
    });
    let Some(current) = matching.next() else {
        return Ok(None);
    };
    // Ambiguous duplicate grants are a malformed configuration. Fail closed
    // instead of selecting whichever entry happened to deserialize first.
    if matching.next().is_some() {
        return Ok(None);
    }
    Ok(retain_queued_workspace_grant(original, current))
}

/// Commands available in the Slack channel (shared Core + Restart).
fn slack_commands() -> Vec<CommandDef> {
    let mut cmds: Vec<CommandDef> = shared_commands()
        .into_iter()
        .filter(|c| matches!(c.category, CommandCategory::Core))
        .collect();
    cmds.push(CommandDef {
        name: "workspace",
        description: "Grant or revoke expiring project access (owner only)",
        usage: Some("grant <@user> <read|edit> <project-path> [hours] | revoke <@user> | list"),
        category: CommandCategory::Core,
    });
    cmds.push(CommandDef {
        name: "restart",
        description: "Restart the daemon",
        usage: None,
        category: CommandCategory::Restart,
    });
    cmds.push(CommandDef {
        name: "help",
        description: "Show available commands",
        usage: None,
        category: CommandCategory::Core,
    });
    cmds
}

/// Slack channel implementation using Socket Mode (WebSocket) for receiving
/// events and the Web API (HTTP) for sending messages.
pub struct SlackChannel {
    /// Bot name fetched from Slack API (e.g., "my_bot").
    /// Populated on first connection.
    bot_name: std::sync::RwLock<String>,
    app_token: String,
    bot_token: String,
    allowed_user_ids: std::sync::RwLock<Vec<String>>,
    /// Explicit owners from `users.owner_ids.slack`. When empty, allowed users
    /// retain legacy owner behavior for backward compatibility.
    owner_user_ids: std::sync::RwLock<Vec<String>>,
    workspace_grants: std::sync::RwLock<Vec<WorkspaceGrant>>,
    use_threads: bool,
    agent: Arc<Agent>,
    config_path: PathBuf,
    pending_approvals: Mutex<HashMap<String, tokio::sync::oneshot::Sender<ApprovalResponse>>>,
    session_map: SessionMap,
    task_registry: Arc<TaskRegistry>,
    files_enabled: bool,
    inbox_dir: PathBuf,
    max_file_size_mb: u64,
    state: Arc<dyn StateStore>,
    /// HTTP client for Slack Web API calls.
    http: reqwest::Client,
    /// Our own bot user ID, resolved on first connection.
    bot_user_id: Mutex<Option<String>>,
    /// Slack team/workspace ID from `auth.test`; grants fail closed until known.
    team_id: std::sync::RwLock<Option<String>>,
    /// Reference to the channel hub for dynamic bot registration.
    channel_hub: std::sync::RwLock<Option<Weak<ChannelHub>>>,
    /// Seconds of no heartbeat before declaring the agent stuck (0 = disabled).
    watchdog_stale_threshold_secs: u64,
    /// Cache of resolved Slack user IDs to display names.
    user_cache: RwLock<HashMap<String, String>>,
    /// Cache of channel ID → channel name (process-lifetime; channel names rarely change).
    channel_name_cache: RwLock<HashMap<String, String>>,
    /// Cache of channel ID → (member display names, fetched_at). TTL: 10 minutes.
    channel_members_cache: RwLock<HashMap<String, (Vec<String>, Instant)>>,
    /// Daemon start time used for post-restart UX guardrails.
    started_at: Instant,
}

fn determine_slack_role(
    allowed_user_ids: &[String],
    owner_user_ids: &[String],
    user_id: &str,
    has_workspace_grant: bool,
    directly_addressed: bool,
) -> Option<UserRole> {
    let is_allowed = allowed_user_ids.iter().any(|id| id == user_id);
    let is_owner = if owner_user_ids.is_empty() {
        // Backward-compatible migration: existing installations historically
        // treated every Slack allowlist entry as an owner.
        is_allowed
    } else {
        owner_user_ids.iter().any(|id| id == user_id)
    };

    if is_owner {
        Some(UserRole::Owner)
    } else if is_allowed || has_workspace_grant {
        Some(UserRole::Guest)
    } else if directly_addressed {
        Some(UserRole::Public)
    } else {
        None
    }
}

impl SlackChannel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        app_token: &str,
        bot_token: &str,
        allowed_user_ids: Vec<String>,
        owner_user_ids: Vec<String>,
        workspace_grants: Vec<WorkspaceGrant>,
        use_threads: bool,
        agent: Arc<Agent>,
        config_path: PathBuf,
        session_map: SessionMap,
        task_registry: Arc<TaskRegistry>,
        files_enabled: bool,
        inbox_dir: PathBuf,
        max_file_size_mb: u64,
        state: Arc<dyn StateStore>,
        watchdog_stale_threshold_secs: u64,
    ) -> Self {
        Self {
            bot_name: std::sync::RwLock::new("slack".to_string()),
            app_token: app_token.to_string(),
            bot_token: bot_token.to_string(),
            allowed_user_ids: std::sync::RwLock::new(allowed_user_ids),
            owner_user_ids: std::sync::RwLock::new(owner_user_ids),
            workspace_grants: std::sync::RwLock::new(workspace_grants),
            use_threads,
            agent,
            config_path,
            pending_approvals: Mutex::new(HashMap::new()),
            session_map,
            task_registry,
            files_enabled,
            inbox_dir,
            max_file_size_mb,
            state,
            http: reqwest::Client::new(),
            bot_user_id: Mutex::new(None),
            team_id: std::sync::RwLock::new(None),
            channel_hub: std::sync::RwLock::new(None),
            watchdog_stale_threshold_secs,
            user_cache: RwLock::new(HashMap::new()),
            channel_name_cache: RwLock::new(HashMap::new()),
            channel_members_cache: RwLock::new(HashMap::new()),
            started_at: Instant::now(),
        }
    }

    /// Set the channel hub reference for dynamic bot registration.
    pub fn set_channel_hub(&self, hub: Weak<ChannelHub>) {
        if let Ok(mut guard) = self.channel_hub.write() {
            *guard = Some(hub);
        }
    }

    /// How long a resolved approval card stays visible before it is auto-deleted,
    /// matching Telegram's approval-card behavior. On by default (10s); override
    /// with `AIDAEMON_APPROVAL_DISMISS_SECS`, or set it to `0` to keep resolved
    /// cards in the conversation.
    fn resolved_approval_dismiss_delay() -> Option<Duration> {
        let secs = std::env::var("AIDAEMON_APPROVAL_DISMISS_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(10);
        (secs > 0).then(|| Duration::from_secs(secs))
    }

    /// Persist Slack allowlist and owner IDs together so auto-claim cannot leave
    /// a user authorized but accidentally downgraded (or vice versa).
    async fn persist_access_ids(
        &self,
        allowed_ids: &[String],
        owner_ids: &[String],
    ) -> anyhow::Result<()> {
        let _write_guard = SLACK_CONFIG_WRITE_LOCK.lock().await;
        let content = tokio::fs::read_to_string(&self.config_path).await?;
        let mut doc: toml::Table = content.parse()?;

        let ids_toml = toml::Value::Array(
            allowed_ids
                .iter()
                .map(|id| toml::Value::String(id.clone()))
                .collect(),
        );

        let mut updated = false;
        if let Some(slack) = doc.get_mut("slack").and_then(|v| v.as_table_mut()) {
            slack.insert("allowed_user_ids".to_string(), ids_toml.clone());
            updated = true;
        }
        if let Some(bots) = doc.get_mut("slack_bots").and_then(|v| v.as_array_mut()) {
            if let Some(first) = bots.first_mut().and_then(|v| v.as_table_mut()) {
                first.insert("allowed_user_ids".to_string(), ids_toml);
                updated = true;
            }
        }

        if !updated {
            anyhow::bail!("No [slack] or [[slack_bots]] section found in config");
        }

        let users = doc
            .entry("users".to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("[users] must be a TOML table"))?;
        let owner_table = users
            .entry("owner_ids".to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("[users.owner_ids] must be a TOML table"))?;
        owner_table.insert(
            "slack".to_string(),
            toml::Value::Array(
                owner_ids
                    .iter()
                    .map(|id| toml::Value::String(id.clone()))
                    .collect(),
            ),
        );

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        tokio::fs::write(&self.config_path, &new_content).await?;
        Self::restrict_config_permissions(&self.config_path);
        info!("Persisted Slack access IDs to config.toml");
        Ok(())
    }

    /// Apply one grant mutation against the latest on-disk configuration. Each
    /// Slack bot keeps a local cache for message routing, but config writes must
    /// merge from disk so one bot cannot erase another bot's grants with a
    /// stale snapshot.
    async fn mutate_workspace_grants(
        &self,
        mutation: WorkspaceGrantMutation,
    ) -> anyhow::Result<(Vec<WorkspaceGrant>, bool)> {
        let _write_guard = SLACK_CONFIG_WRITE_LOCK.lock().await;
        let content = tokio::fs::read_to_string(&self.config_path).await?;
        let mut doc: toml::Table = content.parse()?;
        let users = doc
            .entry("users".to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("[users] must be a TOML table"))?;

        let mut grants: Vec<WorkspaceGrant> = users
            .get("workspace_grants")
            .cloned()
            .map(|value| value.try_into())
            .transpose()?
            .unwrap_or_default();
        grants.retain(WorkspaceGrant::is_active);

        let affected = match mutation {
            WorkspaceGrantMutation::Upsert(grant) => {
                grants.retain(|existing| {
                    !(existing.platform == grant.platform
                        && existing.workspace_id == grant.workspace_id
                        && existing.channel_id == grant.channel_id
                        && existing.user_id == grant.user_id)
                });
                grants.push(grant);
                true
            }
            WorkspaceGrantMutation::Revoke {
                platform,
                workspace_id,
                channel_id,
                user_id,
            } => {
                let before = grants.len();
                grants.retain(|grant| {
                    !(grant.platform == platform
                        && grant.workspace_id == workspace_id
                        && grant.channel_id == channel_id
                        && grant.user_id == user_id)
                });
                grants.len() != before
            }
        };

        users.insert(
            "workspace_grants".to_string(),
            toml::Value::try_from(&grants)?,
        );

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        tokio::fs::write(&self.config_path, &new_content).await?;
        Self::restrict_config_permissions(&self.config_path);
        info!(count = grants.len(), "Persisted Slack workspace grants");
        Ok((grants, affected))
    }

    async fn load_workspace_grants(&self) -> anyhow::Result<Vec<WorkspaceGrant>> {
        load_workspace_grants_from_config(&self.config_path).await
    }

    #[cfg(unix)]
    fn restrict_config_permissions(path: &std::path::Path) {
        use std::os::unix::fs::PermissionsExt;
        if let Err(error) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
            warn!(%error, "Could not enforce owner-only config permissions");
        }
    }

    #[cfg(not(unix))]
    fn restrict_config_permissions(_path: &std::path::Path) {}

    fn is_owner_id(&self, user_id: &str) -> bool {
        let owners = self
            .owner_user_ids
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !owners.is_empty() {
            return owners.iter().any(|id| id == user_id);
        }
        drop(owners);
        self.allowed_user_ids
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .any(|id| id == user_id)
    }

    async fn active_workspace_grant(
        &self,
        channel_id: &str,
        user_id: &str,
    ) -> Option<WorkspaceGrant> {
        let team_id = self
            .team_id
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()?;

        // Refresh on every group request. Multiple configured Slack bots can
        // share one config file, and a revocation must not be delayed by another
        // bot's stale process-local cache. Read failures fail closed.
        let grants = match self.load_workspace_grants().await {
            Ok(grants) => grants,
            Err(error) => {
                warn!(%error, "Could not refresh workspace grants; denying scoped access");
                return None;
            }
        };
        let active = grants
            .iter()
            .find(|grant| {
                grant.is_active()
                    && grant.platform == "slack"
                    && grant.workspace_id == team_id
                    && grant.channel_id == channel_id
                    && grant.user_id == user_id
            })
            .cloned();
        *self
            .workspace_grants
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = grants;
        active
    }

    fn parse_slack_user_mention(token: &str) -> anyhow::Result<String> {
        let inner = token
            .strip_prefix("<@")
            .and_then(|value| value.strip_suffix('>'))
            .ok_or_else(|| anyhow::anyhow!("Use an actual Slack @mention for the collaborator"))?;
        let user_id = inner.split('|').next().unwrap_or_default();
        anyhow::ensure!(
            (user_id.starts_with('U') || user_id.starts_with('W'))
                && (3..=32).contains(&user_id.len())
                && user_id.chars().all(|ch| ch.is_ascii_alphanumeric()),
            "Invalid Slack user mention"
        );
        Ok(user_id.to_string())
    }

    async fn resolve_workspace_project_root(&self, raw_path: &str) -> anyhow::Result<String> {
        let backend = crate::execution::active_execution_backend();
        anyhow::ensure!(
            backend.kind() == crate::execution::BackendKind::Local,
            "Workspace collaboration is currently available only with the local execution backend"
        );

        let resolved = backend.resolve_path(raw_path).await?;
        let metadata = backend
            .metadata(&resolved)
            .await
            .map_err(|_| anyhow::anyhow!("Project directory does not exist"))?;
        anyhow::ensure!(metadata.is_dir(), "The project path must be a directory");

        let canonical = backend.canonicalize(&resolved).await?;
        let project_root = std::path::PathBuf::from(canonical.as_str());
        anyhow::ensure!(
            project_root.parent().is_some(),
            "The filesystem root cannot be delegated"
        );
        if let Ok(home) = backend.canonicalize(backend.home_hint()).await {
            anyhow::ensure!(
                canonical != home,
                "The home directory cannot be delegated; choose one project"
            );
        }
        anyhow::ensure!(
            !crate::tools::fs_utils::is_sensitive_path(&project_root),
            "A sensitive directory cannot be delegated"
        );
        anyhow::ensure!(
            crate::tools::fs_utils::find_nearest_project_root(&project_root)
                .is_some_and(|root| root == project_root),
            "The path must be a project root (for example, a Git, Cargo, Node, Python, Go, or similar project)"
        );

        Ok(canonical.as_str().to_string())
    }

    async fn handle_workspace_command(
        &self,
        raw_command: &str,
        channel_id: &str,
        channel_type: &str,
        user_role: UserRole,
    ) -> String {
        const USAGE: &str = "Usage:\n\
            `!workspace grant <@user> <read|edit> <project-path> [hours]`\n\
            `!workspace revoke <@user>`\n\
            `!workspace list`\n\n\
            Grants work only in this private group, expire automatically (24h default, 168h max), and never include shell, deployment, credentials, memory, browser control, configuration, or sub-agents.";

        if user_role != UserRole::Owner {
            return "Only a configured owner can manage workspace access.".to_string();
        }
        if !matches!(channel_type, "mpim" | "group") {
            return "Workspace access can only be managed from the private Slack group where it will apply."
                .to_string();
        }
        let team_id = match self
            .team_id
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
        {
            Some(team_id) => team_id,
            None => {
                return "Slack workspace identity is unavailable, so access was not changed. Try again after the connection refreshes."
                    .to_string()
            }
        };

        let tokens = match shell_words::split(raw_command) {
            Ok(tokens) => tokens,
            Err(_) => {
                return format!(
                    "Could not parse the command. Quote paths containing spaces.\n\n{USAGE}"
                )
            }
        };
        if tokens.len() < 2 {
            return USAGE.to_string();
        }

        match tokens[1].as_str() {
            "grant" => {
                if !(tokens.len() == 5 || tokens.len() == 6) {
                    return USAGE.to_string();
                }
                let user_id = match Self::parse_slack_user_mention(&tokens[2]) {
                    Ok(user_id) => user_id,
                    Err(error) => return format!("{error}\n\n{USAGE}"),
                };
                if self.is_owner_id(&user_id) {
                    return "That member is already an owner and does not need a workspace grant."
                        .to_string();
                }
                let access = match tokens[3].as_str() {
                    "read" => WorkspaceAccessLevel::Read,
                    "edit" => WorkspaceAccessLevel::Edit,
                    _ => return format!("Access must be `read` or `edit`.\n\n{USAGE}"),
                };
                let hours = if tokens.len() == 6 {
                    match tokens[5].parse::<i64>() {
                        Ok(hours) if (1..=MAX_WORKSPACE_GRANT_HOURS).contains(&hours) => hours,
                        _ => {
                            return format!(
                                "Hours must be between 1 and {MAX_WORKSPACE_GRANT_HOURS}."
                            )
                        }
                    }
                } else {
                    DEFAULT_WORKSPACE_GRANT_HOURS
                };
                let project_root = match self.resolve_workspace_project_root(&tokens[4]).await {
                    Ok(root) => root,
                    Err(error) => return format!("Access was not changed: {error}"),
                };
                let grant = WorkspaceGrant {
                    platform: "slack".to_string(),
                    workspace_id: team_id.clone(),
                    channel_id: channel_id.to_string(),
                    user_id: user_id.clone(),
                    project_root,
                    access,
                    expires_at: chrono::Utc::now() + chrono::Duration::hours(hours),
                };
                let project_name = grant.project_name().to_string();
                let next_grants = match self
                    .mutate_workspace_grants(WorkspaceGrantMutation::Upsert(grant))
                    .await
                {
                    Ok((grants, _)) => grants,
                    Err(error) => {
                        warn!(%error, "Failed to persist workspace grant");
                        return "Access was not changed because the secure configuration update failed."
                            .to_string();
                    }
                };
                *self
                    .workspace_grants
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = next_grants;
                let collaborator = self
                    .resolve_user_name(&user_id)
                    .await
                    .unwrap_or_else(|| "the collaborator".to_string());
                format!(
                    "Granted @{collaborator} {access} access to project `{project_name}` in this private group for {hours} hour(s). Shell, deployment, secrets, memory, configuration, browser control, and sub-agents remain unavailable."
                )
            }
            "revoke" => {
                if tokens.len() != 3 {
                    return USAGE.to_string();
                }
                let user_id = match Self::parse_slack_user_mention(&tokens[2]) {
                    Ok(user_id) => user_id,
                    Err(error) => return format!("{error}\n\n{USAGE}"),
                };
                let (next_grants, revoked) = match self
                    .mutate_workspace_grants(WorkspaceGrantMutation::Revoke {
                        platform: "slack".to_string(),
                        workspace_id: team_id.clone(),
                        channel_id: channel_id.to_string(),
                        user_id: user_id.clone(),
                    })
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        warn!(%error, "Failed to persist workspace revocation");
                        return "Access was not changed because the secure configuration update failed."
                            .to_string();
                    }
                };
                *self
                    .workspace_grants
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = next_grants;
                if !revoked {
                    return "No active workspace grant exists for that member in this group."
                        .to_string();
                }
                let collaborator = self
                    .resolve_user_name(&user_id)
                    .await
                    .unwrap_or_else(|| "that collaborator".to_string());
                format!("Revoked workspace access for @{collaborator} in this group.")
            }
            "list" => {
                if tokens.len() != 2 {
                    return USAGE.to_string();
                }
                let latest = match self.load_workspace_grants().await {
                    Ok(grants) => grants,
                    Err(error) => {
                        warn!(%error, "Failed to refresh workspace grant list");
                        return "Workspace grants could not be read securely; no access details were shown."
                            .to_string();
                    }
                };
                *self
                    .workspace_grants
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = latest;
                let grants = self
                    .workspace_grants
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .iter()
                    .filter(|grant| {
                        grant.is_active()
                            && grant.platform == "slack"
                            && grant.workspace_id == team_id
                            && grant.channel_id == channel_id
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                if grants.is_empty() {
                    return "No active workspace grants exist in this group.".to_string();
                }
                let mut lines = vec!["Active workspace grants in this group:".to_string()];
                for grant in grants {
                    let collaborator = self
                        .resolve_user_name(&grant.user_id)
                        .await
                        .unwrap_or_else(|| "Slack member".to_string());
                    lines.push(format!(
                        "- @{collaborator}: {} on `{}` until {} UTC",
                        grant.access,
                        grant.project_name(),
                        grant.expires_at.format("%Y-%m-%d %H:%M")
                    ));
                }
                lines.join("\n")
            }
            _ => USAGE.to_string(),
        }
    }

    /// Get the bot's name (cached after first connection).
    fn get_bot_name(&self) -> String {
        self.bot_name
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Set the bot's name (called during start() after fetching from API).
    fn set_bot_name(&self, name: String) {
        if let Ok(mut guard) = self.bot_name.write() {
            *guard = name;
        }
    }

    /// Build the session ID, prefixing with bot name if not "slack".
    fn session_id(&self, base_id: &str) -> String {
        let name = self.get_bot_name();
        if name == "slack" {
            base_id.to_string()
        } else {
            format!("{}:{}", name, base_id)
        }
    }

    /// Get the channel identifier for the session map.
    fn channel_name(&self) -> String {
        let name = self.get_bot_name();
        if name == "slack" {
            "slack".to_string()
        } else {
            format!("slack:{}", name)
        }
    }

    /// Start the Slack Socket Mode client with automatic retry on crash.
    pub async fn start_with_retry(self: Arc<Self>) {
        let mut backoff = crate::backoff::ChannelReconnectBackoff::new(
            Duration::from_secs(5),
            Duration::from_secs(60),
            Duration::from_secs(60),
        );

        loop {
            info!("Starting Slack Socket Mode client");
            let started = tokio::time::Instant::now();
            if let Err(e) = self.clone().start().await {
                warn!("Slack client error: {}", e);
            }
            let ran_for = started.elapsed();

            let delay = backoff.next_jittered_delay(ran_for, 0.10);

            warn!(
                backoff_secs = delay.as_secs(),
                ran_for_secs = ran_for.as_secs(),
                "Slack client stopped, restarting"
            );
            tokio::time::sleep(delay).await;
        }
    }

    /// Open a Socket Mode connection and process events.
    async fn start(self: Arc<Self>) -> anyhow::Result<()> {
        // Resolve our own bot user ID (for filtering self-messages)
        self.resolve_bot_info().await;

        // Request a WebSocket URL from Slack
        let wss_url = self.open_connection().await?;
        // The returned URL carries a short-lived ticket in its query string.
        // Never persist it in logs.
        info!("Slack Socket Mode connection URL obtained");

        // Connect via WebSocket
        let (ws_stream, _) = tokio_tungstenite::connect_async(&wss_url)
            .await
            .map_err(|e| anyhow::anyhow!("WebSocket connect failed: {}", e))?;

        info!("Slack WebSocket connected");

        let (mut ws_tx, mut ws_rx) = ws_stream.split();

        // Process incoming WebSocket messages
        while let Some(msg) = ws_rx.next().await {
            let msg = match msg {
                Ok(m) => m,
                Err(e) => {
                    warn!("WebSocket read error: {}", e);
                    break;
                }
            };

            match msg {
                tokio_tungstenite::tungstenite::Message::Text(text) => {
                    // tokio-tungstenite 0.29+ wraps text payloads in `Utf8Bytes`;
                    // borrow `&str` via Deref for downstream parsing.
                    let envelope: Value = match serde_json::from_str(text.as_str()) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("Failed to parse Slack envelope: {}", e);
                            continue;
                        }
                    };

                    // Acknowledge the envelope immediately
                    if let Some(envelope_id) = envelope.get("envelope_id").and_then(|v| v.as_str())
                    {
                        let ack = serde_json::json!({ "envelope_id": envelope_id });
                        let ack_msg =
                            tokio_tungstenite::tungstenite::Message::Text(ack.to_string());
                        if let Err(e) = ws_tx.send(ack_msg).await {
                            warn!("Failed to ack envelope: {}", e);
                        }
                    }

                    // Handle disconnect events
                    if envelope.get("type").and_then(|v| v.as_str()) == Some("disconnect") {
                        let reason = envelope
                            .get("reason")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        info!(reason, "Slack requested disconnect");
                        break;
                    }

                    // Dispatch by envelope type
                    let channel = Arc::clone(&self);
                    tokio::spawn(async move {
                        channel.handle_envelope(envelope).await;
                    });
                }
                tokio_tungstenite::tungstenite::Message::Ping(data) => {
                    let pong = tokio_tungstenite::tungstenite::Message::Pong(data);
                    let _ = ws_tx.send(pong).await;
                }
                tokio_tungstenite::tungstenite::Message::Close(_) => {
                    info!("Slack WebSocket closed by server");
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Call `apps.connections.open` to get a WebSocket URL.
    async fn open_connection(&self) -> anyhow::Result<String> {
        let resp = self
            .http
            .post("https://slack.com/api/apps.connections.open")
            .header("Authorization", format!("Bearer {}", self.app_token))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .send()
            .await?;

        let body: Value = resp.json().await?;
        if body.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            let error = body
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            anyhow::bail!("apps.connections.open failed: {}", error);
        }

        body.get("url")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("No URL in apps.connections.open response"))
    }

    /// Resolve the bot's own user ID and name via `auth.test`.
    async fn resolve_bot_info(&self) {
        match self.slack_api_get("auth.test").await {
            Ok(resp) => {
                if let Some(user_id) = resp.get("user_id").and_then(|v| v.as_str()) {
                    let mut guard = self.bot_user_id.lock().await;
                    *guard = Some(user_id.to_string());
                }
                if let Some(team_id) = resp.get("team_id").and_then(|v| v.as_str()) {
                    if let Ok(mut guard) = self.team_id.write() {
                        *guard = Some(team_id.to_string());
                    }
                }
                // Extract bot name from response (field is "user")
                if let Some(bot_name) = resp.get("user").and_then(|v| v.as_str()) {
                    self.set_bot_name(bot_name.to_string());
                    info!(bot_name, "Resolved Slack bot info");
                }
            }
            Err(e) => {
                warn!("Failed to resolve bot info: {}", e);
            }
        }
    }

    /// Resolve a Slack user ID to a display name via the `users.info` API.
    /// Results are cached in memory for the process lifetime.
    async fn resolve_user_name(&self, user_id: &str) -> Option<String> {
        // Check cache first
        if let Some(name) = self.user_cache.read().await.get(user_id) {
            return Some(name.clone());
        }

        // Call users.info API
        let resp = self
            .http
            .get("https://slack.com/api/users.info")
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .query(&[("user", user_id)])
            .send()
            .await
            .ok()?;
        let json: Value = resp.json().await.ok()?;
        if json.get("ok")?.as_bool()? {
            let user = json.get("user")?;
            let name = user
                .pointer("/profile/display_name")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    user.pointer("/profile/real_name")
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                })
                .or_else(|| user.get("name").and_then(|v| v.as_str()))?;
            self.user_cache
                .write()
                .await
                .insert(user_id.to_string(), name.to_string());
            Some(name.to_string())
        } else {
            None
        }
    }

    /// Convert `@DisplayName` mentions in outgoing text back to Slack's `<@USERID>` format.
    async fn restore_user_mentions(&self, text: &str) -> String {
        let cache = self.user_cache.read().await;
        restore_mentions_from_cache(text, &cache)
    }

    /// Resolve all `<@USERID>` mentions in a text string to display names.
    /// Replaces e.g. `<@U04FL1J2V6>` with `@Alice` (or leaves unchanged if unresolvable).
    async fn resolve_user_mentions(&self, text: &str) -> String {
        let mut result = text.to_string();
        // Find all <@UXXXXX> patterns
        let mut start = 0;
        while let Some(open) = result[start..].find("<@U") {
            let abs_open = start + open;
            if let Some(close) = result[abs_open..].find('>') {
                let abs_close = abs_open + close;
                let user_id = &result[abs_open + 2..abs_close]; // strip <@ and >
                if let Some(name) = self.resolve_user_name(user_id).await {
                    let mention = format!("<@{}>", user_id);
                    let replacement = format!("@{}", name);
                    result = result.replacen(&mention, &replacement, 1);
                    start = abs_open + replacement.len();
                } else {
                    start = abs_close + 1;
                }
            } else {
                break;
            }
        }
        result
    }

    /// Resolve a channel ID to a human-readable name via `conversations.info`.
    /// Results are cached for the process lifetime (channel names rarely change).
    async fn resolve_channel_name(&self, channel_id: &str) -> Option<String> {
        // Check cache first
        if let Some(name) = self.channel_name_cache.read().await.get(channel_id) {
            return Some(name.clone());
        }

        // Call conversations.info API
        let resp = self
            .http
            .get("https://slack.com/api/conversations.info")
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .query(&[("channel", channel_id)])
            .send()
            .await
            .ok()?;
        let json: Value = resp.json().await.ok()?;
        if json.get("ok")?.as_bool()? {
            let name = json
                .pointer("/channel/name")
                .and_then(|v| v.as_str())?
                .to_string();
            let display_name = format!("#{}", name);
            self.channel_name_cache
                .write()
                .await
                .insert(channel_id.to_string(), display_name.clone());
            Some(display_name)
        } else {
            debug!(
                channel_id,
                error = json
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown"),
                "Failed to resolve channel name"
            );
            None
        }
    }

    /// Resolve the display names of members in a channel via `conversations.members` + `users.info`.
    /// Results are cached with a 10-minute TTL. Limited to first 50 members.
    async fn resolve_channel_member_names(&self, channel_id: &str) -> Vec<String> {
        const MEMBERS_TTL: Duration = Duration::from_secs(600); // 10 minutes

        // Check cache (with TTL)
        if let Some((names, fetched_at)) = self.channel_members_cache.read().await.get(channel_id) {
            if fetched_at.elapsed() < MEMBERS_TTL {
                return names.clone();
            }
        }

        // Call conversations.members API (limit 50)
        let resp = match self
            .http
            .get("https://slack.com/api/conversations.members")
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .query(&[("channel", channel_id), ("limit", "50")])
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                debug!(error = %e, "Failed to fetch channel members");
                return vec![];
            }
        };

        let json: Value = match resp.json().await {
            Ok(j) => j,
            Err(_) => return vec![],
        };

        if json.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            let error = json
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            if error == "channel_not_found" || error == "method_not_allowed_for_channel_type" {
                warn!(
                    channel_id,
                    error,
                    "Failed to fetch channel members — for private channels, \
                     the bot needs the `groups:read` scope in addition to `channels:read`"
                );
            } else {
                debug!(channel_id, error, "Failed to fetch channel members");
            }
            return vec![];
        }

        let member_ids: Vec<String> = json
            .get("members")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Filter out our own bot user ID
        let bot_id = self.bot_user_id.lock().await.clone();
        let filtered_ids: Vec<String> = member_ids
            .into_iter()
            .filter(|id| bot_id.as_deref() != Some(id.as_str()))
            .collect();

        // Resolve each user ID to a display name
        let mut names = Vec::with_capacity(filtered_ids.len());
        for uid in &filtered_ids {
            if let Some(name) = self.resolve_user_name(uid).await {
                names.push(name);
            }
        }

        // Cache the result
        self.channel_members_cache
            .write()
            .await
            .insert(channel_id.to_string(), (names.clone(), Instant::now()));

        names
    }

    /// Make a GET-style Slack API call (actually POST with empty body).
    async fn slack_api_get(&self, method: &str) -> anyhow::Result<Value> {
        let url = format!("https://slack.com/api/{}", method);
        let resp = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .send()
            .await?;
        let body: Value = resp.json().await?;
        if body.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            let error = body
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            anyhow::bail!("Slack API {} failed: {}", method, error);
        }
        Ok(body)
    }

    /// Make a POST Slack API call with a JSON body.
    async fn slack_api_post(&self, method: &str, body: &Value) -> anyhow::Result<Value> {
        let url = format!("https://slack.com/api/{}", method);
        let body_bytes = serde_json::to_vec(body)?;
        let mut resp = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .header("Content-Type", "application/json; charset=utf-8")
            .body(body_bytes.clone())
            .send()
            .await?;
        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            if let Some(delay) = crate::channels::rate_limit::retry_after_from_headers(
                resp.headers(),
                crate::channels::rate_limit::default_retry_after_cap(),
            ) {
                crate::channels::rate_limit::sleep_after_rate_limit("slack", delay).await;
                resp = self
                    .http
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", self.bot_token))
                    .header("Content-Type", "application/json; charset=utf-8")
                    .body(body_bytes)
                    .send()
                    .await?;
            }
        }
        let result: Value = resp.json().await?;
        if result.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            let error = result
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            anyhow::bail!(
                "Slack API {} failed: {} (response: {})",
                method,
                error,
                result
            );
        }
        Ok(result)
    }

    /// Handle a Socket Mode envelope.
    async fn handle_envelope(&self, envelope: Value) {
        let envelope_type = match envelope.get("type").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => return,
        };

        match envelope_type {
            "events_api" => {
                if let Some(payload) = envelope.get("payload") {
                    self.handle_events_api(payload).await;
                }
            }
            "interactive" => {
                if let Some(payload) = envelope.get("payload") {
                    self.handle_interactive(payload).await;
                }
            }
            "slash_commands" => {
                if let Some(payload) = envelope.get("payload") {
                    self.handle_slash_command(payload).await;
                }
            }
            "hello" => {
                info!("Slack Socket Mode hello received");
            }
            _ => {
                debug!(envelope_type, "Unhandled Slack envelope type");
            }
        }
    }

    /// Handle an Events API payload (message events, etc.).
    async fn handle_events_api(&self, payload: &Value) {
        let event = match payload.get("event") {
            Some(e) => e,
            None => return,
        };

        let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if event_type != "message" && event_type != "app_mention" {
            return;
        }

        // Ignore message subtypes (edits, joins, bot messages, etc.)
        if event.get("subtype").is_some() {
            return;
        }

        let user = match event.get("user").and_then(|v| v.as_str()) {
            Some(u) => u.to_string(),
            None => return,
        };

        // Ignore our own messages
        {
            let bot_id = self.bot_user_id.lock().await;
            if bot_id.as_deref() == Some(&user) {
                return;
            }
        }

        let channel_id = match event.get("channel").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return,
        };

        let raw_text = event
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let channel_type = event
            .get("channel_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Check if the message mentions the bot
        let bot_mentioned = {
            let bot_id_guard = self.bot_user_id.lock().await;
            if let Some(ref bid) = *bot_id_guard {
                raw_text.contains(&format!("<@{}>", bid))
            } else {
                false
            }
        };
        let is_dm = channel_type == "im";

        // Auto-claim only when neither an allowlist nor an explicit owner exists.
        // Persist both lists together so the claimed user is unambiguously Owner.
        let auto_claimed = if is_dm {
            let mut allowed = self
                .allowed_user_ids
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut owners = self
                .owner_user_ids
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if allowed.is_empty() && owners.is_empty() {
                allowed.push(user.clone());
                owners.push(user.clone());
                true
            } else {
                false
            }
        } else {
            false
        };
        if auto_claimed {
            warn!(
                user = %user,
                "No Slack users configured — auto-claiming first DM user as owner."
            );
        }

        if auto_claimed {
            // Persist to config.toml (must be outside RwLock scope to avoid Send issues)
            let allowed_ids = self
                .allowed_user_ids
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            let owner_ids = self
                .owner_user_ids
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            if let Err(e) = self.persist_access_ids(&allowed_ids, &owner_ids).await {
                warn!(user = %user, "Failed to persist auto-claimed user ID to config: {}", e);
            }
            let _ = self.post_message(
                &channel_id,
                "Hey! You're now set as the owner. Ask me anything, give me tasks, or just chat.",
                None,
            ).await;
        }

        let allowed_snapshot = self
            .allowed_user_ids
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let owner_snapshot = self
            .owner_user_ids
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let directly_addressed = bot_mentioned || is_dm;
        let role_without_grant = determine_slack_role(
            &allowed_snapshot,
            &owner_snapshot,
            &user,
            false,
            directly_addressed,
        );
        // A grant can matter only for an addressed, non-owner group member.
        // Avoid a config read for ordinary group chatter and owner traffic.
        let workspace_grant = if matches!(channel_type, "mpim" | "group")
            && bot_mentioned
            && role_without_grant != Some(UserRole::Owner)
        {
            self.active_workspace_grant(&channel_id, &user).await
        } else {
            None
        };

        // Explicit owners retain full access. Allowlisted non-owners are Guests;
        // an active workspace grant also confers Guest status for its exact group.
        let user_role = match determine_slack_role(
            &allowed_snapshot,
            &owner_snapshot,
            &user,
            workspace_grant.is_some(),
            directly_addressed,
        ) {
            Some(role) => role,
            None => return,
        };

        // In groups, collaborators must address the bot explicitly. This keeps
        // ordinary participant-to-participant chat from becoming agent input.
        if user_role == UserRole::Guest && !is_dm && !bot_mentioned {
            return;
        }
        // An authorized owner mentioning another person (but not the bot) is
        // talking to that person, not issuing an agent request.
        if user_role == UserRole::Owner && !is_dm && !bot_mentioned && raw_text.contains("<@") {
            return;
        }

        // Strip bot @mention from text before processing
        let raw_agent_text = {
            let bot_id_guard = self.bot_user_id.lock().await;
            let stripped = if let Some(ref bid) = *bot_id_guard {
                raw_text
                    .replace(&format!("<@{}>", bid), "")
                    .trim()
                    .to_string()
            } else {
                raw_text.clone()
            };
            stripped
        };
        // Resolve remaining <@USERID> mentions to display names for ordinary
        // model input, but retain raw stable IDs for security-sensitive commands.
        let text = self.resolve_user_mentions(&raw_agent_text).await;

        let ts = event
            .get("ts")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let thread_ts = event
            .get("thread_ts")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Check for file attachments (skip for Public users)
        let mut inbound_attachments = Vec::new();
        if self.files_enabled && user_role != UserRole::Public {
            if let Some(files) = event.get("files").and_then(|v| v.as_array()) {
                match self.handle_incoming_files(files).await {
                    Ok(attachments) => inbound_attachments = attachments,
                    Err(e) => {
                        let reply_thread = self.reply_thread_ts(&ts, thread_ts.as_deref());
                        let _ = self
                            .post_message(
                                &channel_id,
                                &format!("File error: {}", e),
                                reply_thread.as_deref(),
                            )
                            .await;
                        return;
                    }
                }
            }
        }

        // Build the text to send to the agent
        let agent_text = if inbound_attachments.is_empty() {
            if text.is_empty() {
                return;
            }
            text.clone()
        } else {
            if user_role == UserRole::Owner {
                super::attachments::build_inbound_text(&text, &inbound_attachments)
            } else {
                super::attachments::build_inbound_text_without_local_paths(
                    &text,
                    &inbound_attachments,
                )
            }
        };

        // Handle slash commands sent as text (block Public users)
        // Accept both /command and !command (Slack reserves / for native slash commands)
        if raw_agent_text.starts_with('/') || raw_agent_text.starts_with('!') {
            if user_role == UserRole::Public {
                let reply_thread = self.reply_thread_ts(&ts, thread_ts.as_deref());
                let _ = self
                    .post_message(
                        &channel_id,
                        "Commands are not available for public users.",
                        reply_thread.as_deref(),
                    )
                    .await;
                return;
            }
            let session_id = self.build_session_id(&channel_id, thread_ts.as_deref());
            let reply_thread = self.reply_thread_ts(&ts, thread_ts.as_deref());
            let command_name = raw_agent_text.split_whitespace().next().unwrap_or_default();
            let (reply, buttons) = if matches!(command_name, "!workspace" | "/workspace") {
                (
                    self.handle_workspace_command(
                        &raw_agent_text,
                        &channel_id,
                        channel_type,
                        user_role,
                    )
                    .await,
                    Vec::new(),
                )
            } else {
                self.dispatch_command_with_buttons(&agent_text, &session_id, user_role)
                    .await
            };
            let mrkdwn = markdown_to_slack_mrkdwn(&reply);
            let chunks = split_message(&mrkdwn, MAX_MESSAGE_LEN);
            let last_idx = chunks.len().saturating_sub(1);
            for (i, chunk) in chunks.into_iter().enumerate() {
                // Attach buttons to the last chunk only
                if i == last_idx && !buttons.is_empty() {
                    let blocks = serde_json::json!([
                        {
                            "type": "section",
                            "text": { "type": "mrkdwn", "text": &chunk }
                        },
                        {
                            "type": "actions",
                            "elements": buttons
                        }
                    ]);
                    let _ = self
                        .post_message_with_blocks(
                            &channel_id,
                            &chunk,
                            blocks,
                            reply_thread.as_deref(),
                        )
                        .await;
                } else {
                    let _ = self
                        .post_message(&channel_id, &chunk, reply_thread.as_deref())
                        .await;
                }
            }
            return;
        }

        // Determine thread_ts for replies
        let reply_thread = if self.use_threads {
            // Reply in the message's thread. If this message was already in a thread,
            // use that thread_ts; otherwise create a new thread from this message's ts.
            Some(thread_ts.unwrap_or_else(|| ts.clone()))
        } else {
            thread_ts
        };

        let session_id = self.build_session_id(&channel_id, reply_thread.as_deref());

        // Handle cancel/stop commands - these bypass the queue
        let text_lower = agent_text.to_lowercase();
        if text_lower == "cancel" || text_lower == "stop" || text_lower == "abort" {
            if user_role != UserRole::Owner {
                let _ = self
                    .post_message(
                        &channel_id,
                        "Only the owner can cancel running work in this session.",
                        reply_thread.as_deref(),
                    )
                    .await;
                return;
            }
            let cancelled = self
                .task_registry
                .cancel_running_for_session(&session_id)
                .await;
            self.task_registry.clear_queue(&session_id).await;
            let cancelled_goals = self
                .agent
                .cancel_active_goals_for_session(&session_id)
                .await;
            if cancelled.is_empty() {
                if cancelled_goals.is_empty() {
                    let _ = self
                        .post_message(
                            &channel_id,
                            "No running task to cancel.",
                            reply_thread.as_deref(),
                        )
                        .await;
                } else if cancelled_goals.len() == 1 {
                    let response = format!("⏹️ Cancelled goal: {}", cancelled_goals[0]);
                    let _ = self
                        .post_message(&channel_id, &response, reply_thread.as_deref())
                        .await;
                } else {
                    let response = format!(
                        "⏹️ Cancelled {} goals:\n{}",
                        cancelled_goals.len(),
                        cancelled_goals
                            .iter()
                            .map(|d| format!("- {}", d))
                            .collect::<Vec<_>>()
                            .join("\n")
                    );
                    let _ = self
                        .post_message(&channel_id, &response, reply_thread.as_deref())
                        .await;
                }
            } else {
                let desc = cancelled
                    .first()
                    .map(|(_, d)| d.as_str())
                    .unwrap_or("unknown");
                let mut response = format!("⏹️ Cancelled: {}", desc);
                if !cancelled_goals.is_empty() {
                    response.push_str(&format!(" (+{} goal(s) cancelled)", cancelled_goals.len()));
                }
                let _ = self
                    .post_message(&channel_id, &response, reply_thread.as_deref())
                    .await;
            }
            return;
        }

        // Register this session with the channel hub (in-memory + persistent)
        {
            let channel_name = self.channel_name();
            {
                let mut map = self.session_map.write().await;
                map.insert(session_id.clone(), channel_name.clone());
            }
            let _ = self
                .state
                .save_session_channel(&session_id, &channel_name)
                .await;
        }

        // Build channel context from Slack channel type
        let channel_ctx = {
            let visibility = match channel_type {
                "im" => ChannelVisibility::Private,
                "mpim" | "group" => ChannelVisibility::PrivateGroup,
                _ => ChannelVisibility::Public, // "channel" or unknown defaults to public
            };
            let sender_name = self.resolve_user_name(&user).await;
            let is_dm = channel_type == "im";
            let (channel_name, channel_member_names) = if !is_dm {
                let name = self.resolve_channel_name(&channel_id).await;
                let members = self.resolve_channel_member_names(&channel_id).await;
                (name, members)
            } else {
                (None, vec![])
            };
            // Snapshot current user_cache as user_id_map for resolving IDs in facts
            let user_id_map = self.user_cache.read().await.clone();
            ChannelContext {
                visibility,
                platform: "slack".to_string(),
                channel_name,
                channel_id: Some(format!("slack:{}", channel_id)),
                workspace_id: self
                    .team_id
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone(),
                sender_name,
                sender_id: Some(format!("slack:{}", user)),
                channel_member_names,
                user_id_map,
                workspace_grant,
                trusted: false,
            }
        };

        info!(session_id, user_id = %user, "Received Slack message");

        // Stable per-message identity for dedup. Slack's `ts` is unique per
        // message and reused on Events API retries, but a user re-typing the
        // same text gets a new `ts` — so this drops genuine duplicates while
        // letting a deliberate re-ask through. Fall back to text-hash dedup when
        // `ts` is absent (malformed event).
        let dedup_key: Option<&str> = if ts.is_empty() {
            None
        } else {
            Some(ts.as_str())
        };

        // Check if a task is already running for this session - if so, queue this message
        if self.task_registry.has_running_task(&session_id).await {
            let daemon_uptime = self.started_at.elapsed();
            if should_ignore_lightweight_interjection(&agent_text, daemon_uptime) {
                let current_task = self
                    .task_registry
                    .get_running_task_description(&session_id)
                    .await
                    .unwrap_or_else(|| "processing".to_string());
                let _ = self
                    .post_message(
                        &channel_id,
                        &format!(
                            "⏳ Still working on: {}. I ignored that short check-in. \
                             Send `cancel` to stop the current task.",
                            current_task
                        ),
                        reply_thread.as_deref(),
                    )
                    .await;
                return;
            }
            // Atomic check-and-queue: if the task finished between the
            // has_running_task() check above and this call, fall through to
            // direct processing instead of stranding the message.
            match self
                .task_registry
                .queue_message_if_running(
                    &session_id,
                    QueuedMessage::new(
                        &agent_text,
                        user_role,
                        channel_ctx.clone(),
                        inbound_attachments.clone(),
                    ),
                    dedup_key,
                )
                .await
            {
                QueueOutcome::Queued(queue_pos) => {
                    // Only notify for the first 3 queued messages to avoid spam.
                    if queue_pos <= 3 {
                        let current_task = self
                            .task_registry
                            .get_running_task_description(&session_id)
                            .await
                            .unwrap_or_else(|| "processing".to_string());
                        let preview: String = agent_text.chars().take(50).collect();
                        let suffix = if agent_text.len() > 50 { "..." } else { "" };
                        let _ = self
                            .post_message(
                                &channel_id,
                                &format!(
                                    "📥 Queued ({}): \"{}{}\" | Currently: {}",
                                    queue_pos, preview, suffix, current_task
                                ),
                                reply_thread.as_deref(),
                            )
                            .await;
                    }
                    return;
                }
                QueueOutcome::Duplicate => {
                    debug!(session_id, "Dropped duplicate queued message");
                    return;
                }
                QueueOutcome::NoRunningTask => {
                    info!(
                        session_id,
                        "Task finished during queue attempt — processing message directly"
                    );
                }
            }
        }

        // Dedup gate: atomically mark this message as "seen" so concurrent
        // handlers for the same text don't ALL start direct processing.
        if !self
            .task_registry
            .mark_seen(&session_id, &agent_text, dedup_key)
            .await
        {
            debug!(
                session_id,
                "Dropped duplicate message (direct processing race)"
            );
            return;
        }

        // Create heartbeat for watchdog — agent bumps this on every activity point.
        let heartbeat = Arc::new(AtomicU64::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        ));

        // Typing indicator via reaction
        let typing_channel = channel_id.clone();
        let typing_ts = ts.clone();
        let typing_self = self.clone_for_typing();
        let typing_cancel = tokio_util::sync::CancellationToken::new();
        let typing_token = typing_cancel.clone();
        tokio::spawn(async move {
            // Add hourglass reaction as typing indicator
            let _ = typing_self
                .add_reaction(&typing_channel, &typing_ts, "hourglass_flowing_sand")
                .await;
            typing_token.cancelled().await;
            // Remove when done
            let _ = typing_self
                .remove_reaction(&typing_channel, &typing_ts, "hourglass_flowing_sand")
                .await;
        });

        // Watchdog: periodically check heartbeat staleness
        let stale_threshold_secs = self.watchdog_stale_threshold_secs;
        if stale_threshold_secs > 0 {
            let watchdog_heartbeat = heartbeat.clone();
            let watchdog_cancel = typing_cancel.clone();
            let watchdog_channel = channel_id.clone();
            let watchdog_ts = ts.clone();
            let watchdog_typing_self = self.clone_for_typing();
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(10)) => {
                            let last_hb = watchdog_heartbeat.load(Ordering::Relaxed);
                            let now = SystemTime::now().duration_since(UNIX_EPOCH)
                                .unwrap_or_default().as_secs();
                            if now.saturating_sub(last_hb) > stale_threshold_secs {
                                // Swap hourglass for warning
                                let _ = watchdog_typing_self.remove_reaction(
                                    &watchdog_channel, &watchdog_ts, "hourglass_flowing_sand"
                                ).await;
                                let _ = watchdog_typing_self.add_reaction(
                                    &watchdog_channel, &watchdog_ts, "warning"
                                ).await;
                                break;
                            }
                        }
                        _ = watchdog_cancel.cancelled() => break,
                    }
                }
            });
        }

        // Status updates
        let (status_tx, mut status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(16);
        let status_channel = channel_id.clone();
        let status_thread = reply_thread.clone();
        let status_self = self.clone_for_status();
        let is_dm = channel_ctx.visibility == ChannelVisibility::Private;
        let status_task = tokio::spawn(async move {
            let mut last_sent = tokio::time::Instant::now() - Duration::from_secs(10);
            let min_interval = Duration::from_secs(3);
            while let Some(update) = status_rx.recv().await {
                // In non-DM channels: hourglass reaction is sufficient, skip status messages
                // (except BudgetExtended which must always reach the user)
                if !is_dm && !matches!(&update, StatusUpdate::BudgetExtended { .. }) {
                    continue;
                }
                let now = tokio::time::Instant::now();
                // Skip rate limiting for ToolProgress with URLs (e.g., OAuth authorize links)
                // and for BudgetExtended (cost notifications should always be delivered)
                let has_url = matches!(&update, StatusUpdate::ToolProgress { chunk, .. }
                    if chunk.contains("https://") || chunk.contains("http://"));
                let is_budget_ext = matches!(&update, StatusUpdate::BudgetExtended { .. });
                if !has_url && !is_budget_ext && now.duration_since(last_sent) < min_interval {
                    continue;
                }
                let text = match &update {
                    StatusUpdate::Thinking(_) => "_Thinking..._".to_string(),
                    StatusUpdate::ToolStart { name, summary } => {
                        let mut chars = name.chars();
                        let cap_name = match chars.next() {
                            None => String::new(),
                            Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
                        };
                        if summary.is_empty() {
                            format!("_{}..._", cap_name)
                        } else {
                            format!("_{}: {}..._", cap_name, summary)
                        }
                    }
                    StatusUpdate::ToolProgress { name, chunk } => {
                        // Don't truncate if the chunk contains a URL (e.g., OAuth authorize links)
                        if chunk.contains("https://") || chunk.contains("http://") {
                            format!("_📤 {}_\n{}", name, chunk)
                        } else {
                            let preview: String = chunk.chars().take(100).collect();
                            if chunk.len() > 100 {
                                format!("_📤 {}: {}..._", name, preview)
                            } else {
                                format!("_📤 {}: {}_", name, preview)
                            }
                        }
                    }
                    StatusUpdate::ToolComplete { name, summary } => {
                        format!("_✓ {}: {}_", name, summary)
                    }
                    StatusUpdate::ToolCancellable { name, task_id } => {
                        format!("_⏳ {} started (task_id: {})_", name, task_id)
                    }
                    StatusUpdate::ProgressSummary {
                        elapsed_mins,
                        summary,
                    } => {
                        format!("_📊 Progress ({} min): {}_", elapsed_mins, summary)
                    }
                    StatusUpdate::IterationWarning { current, threshold } => {
                        format!(
                            "_⚠️ Approaching soft limit: {} of {} iterations_",
                            current, threshold
                        )
                    }
                    StatusUpdate::PlanCreated {
                        description,
                        total_steps,
                        ..
                    } => {
                        format!("_📋 Plan created: {} ({} steps)_", description, total_steps)
                    }
                    StatusUpdate::PlanStepStart {
                        step_index,
                        total_steps,
                        description,
                        ..
                    } => {
                        format!(
                            "_▶️ Step {}/{}: {}_",
                            step_index + 1,
                            total_steps,
                            description
                        )
                    }
                    StatusUpdate::PlanStepComplete {
                        step_index,
                        total_steps,
                        description,
                        summary,
                        ..
                    } => {
                        let base = format!(
                            "_✅ Step {}/{} done: {}",
                            step_index + 1,
                            total_steps,
                            description
                        );
                        if let Some(s) = summary {
                            format!("{} - {}_", base, s)
                        } else {
                            format!("{}_", base)
                        }
                    }
                    StatusUpdate::PlanStepFailed {
                        step_index,
                        description,
                        error,
                        ..
                    } => {
                        format!(
                            "_❌ Step {} failed: {} - {}_",
                            step_index + 1,
                            description,
                            error
                        )
                    }
                    StatusUpdate::PlanComplete {
                        description,
                        total_steps,
                        duration_secs,
                        ..
                    } => {
                        let mins = duration_secs / 60;
                        let secs = duration_secs % 60;
                        format!(
                            "_🎉 Plan complete: {} ({} steps in {}m {}s)_",
                            description, total_steps, mins, secs
                        )
                    }
                    StatusUpdate::PlanAbandoned { description, .. } => {
                        format!("_🚫 Plan abandoned: {}_", description)
                    }
                    StatusUpdate::PlanRevised {
                        description,
                        reason,
                        new_total_steps,
                        ..
                    } => {
                        format!(
                            "_🔄 Plan revised: {} ({} steps) - {}_",
                            description, new_total_steps, reason
                        )
                    }
                    StatusUpdate::BudgetExtended {
                        old_budget,
                        new_budget,
                        extension,
                        max_extensions,
                    } => {
                        format!(
                            "_💰 Auto-extended token budget {} → {} ({}/{}) — continuing._",
                            old_budget, new_budget, extension, max_extensions
                        )
                    }
                    StatusUpdate::Checklist { text } => text.clone(),
                };
                let _ = status_self
                    .post_message(&status_channel, &text, status_thread.as_deref())
                    .await;
                last_sent = tokio::time::Instant::now();
            }
        });

        // Register task for tracking
        let description: String = agent_text.chars().take(80).collect();
        let (task_id, cancel_token) = self.task_registry.register(&session_id, &description).await;
        // Associate the typing indicator with this task so cancel_running_for_session
        // also stops the typing/reaction indicator.
        self.task_registry
            .set_typing_cancel(task_id, typing_cancel.clone())
            .await;
        let registry = Arc::clone(&self.task_registry);

        let agent = Arc::clone(&self.agent);
        let reply_channel = channel_id.clone();
        let reply_thread_ts = reply_thread.clone();
        let bot_token = self.bot_token.clone();
        let http = self.http.clone();
        let config_path = self.config_path.clone();
        // Snapshot user_cache for restoring @mentions in the reply
        let user_cache_snapshot = self.user_cache.read().await.clone();
        tokio::spawn(async move {
            // Drop guard: if this task panics, ensure the typing indicator stops.
            let typing_guard_token = Arc::new(std::sync::Mutex::new(typing_cancel.clone()));
            struct TypingGuard(Arc<std::sync::Mutex<tokio_util::sync::CancellationToken>>);
            impl Drop for TypingGuard {
                fn drop(&mut self) {
                    if let Ok(token) = self.0.lock() {
                        token.cancel();
                    }
                }
            }
            let _typing_guard = TypingGuard(typing_guard_token.clone());

            let mut current_text = agent_text;
            let mut current_attachments = inbound_attachments;
            let mut current_user_role = user_role;
            let mut current_channel_ctx = channel_ctx;
            let mut current_task_id = task_id;
            let mut current_cancel_token = cancel_token;
            let mut current_status_tx = status_tx;
            let mut current_typing_cancel = typing_cancel;
            let mut current_status_task = status_task;
            let mut current_heartbeat = heartbeat;
            let task_wall_deadline = tokio::time::Instant::now() + Duration::from_secs(20 * 60);

            loop {
                let result = tokio::select! {
                    r = agent.handle_message_with_attachments(&session_id, &current_text, &current_attachments, Some(current_status_tx), current_user_role, current_channel_ctx.clone(), Some(current_heartbeat.clone())) => r,
                    _ = current_cancel_token.cancelled() => Err(anyhow::anyhow!("Task cancelled")),
                    stale_mins = super::wait_for_stale_heartbeat(current_heartbeat.clone(), stale_threshold_secs, 10), if stale_threshold_secs > 0 => {
                        Err(anyhow::anyhow!(
                            "Task auto-cancelled due to inactivity ({} minute{} without progress).",
                            stale_mins,
                            if stale_mins == 1 { "" } else { "s" }
                        ))
                    },
                    _ = tokio::time::sleep_until(task_wall_deadline) => {
                        tracing::error!(session_id = %session_id, "Task hit 20-minute hard wall-clock limit");
                        Err(anyhow::anyhow!("Task exceeded maximum wall-clock time (20 minutes). This may indicate a hang."))
                    },
                };
                current_typing_cancel.cancel();
                // Abort the status display task to prevent blocking if a background
                // CLI agent monitoring task still holds a status_tx clone.
                current_status_task.abort();

                let mut task_error: Option<String> = None;

                match result {
                    Ok(reply) => {
                        // NOTE: Task intentionally stays "running" during response
                        // sending to prevent a race where incoming messages skip the
                        // queue. Finalized below before queue check.
                        let mrkdwn = markdown_to_slack_mrkdwn(&reply);
                        // Restore @DisplayName → <@USERID> for proper Slack mentions
                        let mrkdwn = restore_mentions_from_cache(&mrkdwn, &user_cache_snapshot);
                        let chunks = split_message(&mrkdwn, MAX_MESSAGE_LEN);
                        for chunk in &chunks {
                            let _ = slack_post_message(
                                &http,
                                &bot_token,
                                &reply_channel,
                                chunk,
                                reply_thread_ts.as_deref(),
                            )
                            .await;
                        }
                    }
                    Err(e) => {
                        let error_msg = e.to_string();
                        if error_msg == "Task cancelled" {
                            registry.fail(current_task_id, &error_msg).await;
                            info!("Task #{} cancelled", current_task_id);
                            return; // Exit loop on cancellation
                        }
                        task_error = Some(error_msg.clone());
                        if error_msg.starts_with("Task auto-cancelled due to inactivity") {
                            info!("Task #{} auto-cancelled by stale watchdog", current_task_id);
                            let _ = slack_post_message(
                                &http,
                                &bot_token,
                                &reply_channel,
                                &format!("⚠️ {}", error_msg),
                                reply_thread_ts.as_deref(),
                            )
                            .await;
                        } else {
                            warn!("Agent error: {}", e);
                            let _ = slack_post_message(
                                &http,
                                &bot_token,
                                &reply_channel,
                                &format!("Error: {}", e),
                                reply_thread_ts.as_deref(),
                            )
                            .await;
                        }
                    }
                }

                // Finalize the current task AFTER sending the response/error,
                // draining the queue in the same critical section so no
                // concurrent message can be stranded between the two steps.
                if let Some(mut queued) = registry
                    .finalize_and_drain(current_task_id, &session_id, task_error.as_deref())
                    .await
                {
                    // Small delay to ensure previous message is fully committed to DB
                    tokio::time::sleep(Duration::from_millis(100)).await;

                    if queued.user_role == UserRole::Guest {
                        if let Some(original) = queued.channel_ctx.workspace_grant.clone() {
                            queued.channel_ctx.workspace_grant =
                                match revalidate_queued_workspace_grant(&config_path, &original)
                                    .await
                                {
                                    Ok(grant) => grant,
                                    Err(error) => {
                                        warn!(%error, "Could not revalidate queued workspace grant; denying scoped access");
                                        None
                                    }
                                };
                        }
                    }

                    info!(
                        session_id,
                        "Processing queued message: {}",
                        queued.text.chars().take(50).collect::<String>()
                    );
                    let _ = slack_post_message(
                        &http,
                        &bot_token,
                        &reply_channel,
                        &format!(
                            "_▶️ Processing queued: \"{}\"_",
                            queued.text.chars().take(50).collect::<String>()
                        ),
                        reply_thread_ts.as_deref(),
                    )
                    .await;

                    // Set up for next iteration
                    current_text = queued.text;
                    current_user_role = queued.user_role;
                    current_channel_ctx = queued.channel_ctx;
                    current_attachments = queued.attachments;
                    let desc: String = current_text.chars().take(80).collect();
                    let (new_task_id, new_cancel_token) =
                        registry.register(&session_id, &desc).await;
                    current_task_id = new_task_id;
                    current_cancel_token = new_cancel_token;

                    // Create new status channel for queued message
                    let (new_status_tx, mut new_status_rx) =
                        tokio::sync::mpsc::channel::<StatusUpdate>(16);
                    current_status_tx = new_status_tx;

                    let queued_status_channel = reply_channel.clone();
                    let queued_status_thread = reply_thread_ts.clone();
                    let queued_status_http = http.clone();
                    let queued_status_token = bot_token.clone();
                    current_status_task = tokio::spawn(async move {
                        let mut last_sent = tokio::time::Instant::now() - Duration::from_secs(10);
                        let min_interval = Duration::from_secs(3);
                        while let Some(update) = new_status_rx.recv().await {
                            // For queued messages, only show Thinking, ToolStart, and BudgetExtended
                            if !is_dm && !matches!(&update, StatusUpdate::BudgetExtended { .. }) {
                                continue;
                            }
                            let now = tokio::time::Instant::now();
                            let is_budget_ext =
                                matches!(&update, StatusUpdate::BudgetExtended { .. });
                            if !is_budget_ext && now.duration_since(last_sent) < min_interval {
                                continue;
                            }
                            let text = match &update {
                                StatusUpdate::Thinking(_) => "_Thinking..._".to_string(),
                                StatusUpdate::ToolStart { name, summary } => {
                                    let mut chars = name.chars();
                                    let cap_name = match chars.next() {
                                        None => String::new(),
                                        Some(f) => {
                                            f.to_uppercase().collect::<String>() + chars.as_str()
                                        }
                                    };
                                    if summary.is_empty() {
                                        format!("_{}..._", cap_name)
                                    } else {
                                        format!("_{}: {}..._", cap_name, summary)
                                    }
                                }
                                StatusUpdate::BudgetExtended {
                                    old_budget,
                                    new_budget,
                                    extension,
                                    max_extensions,
                                } => {
                                    format!(
                                        "_💰 Auto-extended token budget {} → {} ({}/{}) — continuing._",
                                        old_budget, new_budget, extension, max_extensions
                                    )
                                }
                                _ => continue,
                            };
                            let _ = slack_post_message(
                                &queued_status_http,
                                &queued_status_token,
                                &queued_status_channel,
                                &text,
                                queued_status_thread.as_deref(),
                            )
                            .await;
                            last_sent = tokio::time::Instant::now();
                        }
                    });

                    // Fresh heartbeat for queued message
                    let new_heartbeat = Arc::new(AtomicU64::new(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    ));
                    current_heartbeat = new_heartbeat;

                    // New typing cancel for queued message (no reaction-based indicator)
                    current_typing_cancel = tokio_util::sync::CancellationToken::new();
                    // Associate typing token with the new queued task for cancel support
                    registry
                        .set_typing_cancel(current_task_id, current_typing_cancel.clone())
                        .await;
                    // Update the drop guard to track the new typing token
                    if let Ok(mut guard_token) = typing_guard_token.lock() {
                        *guard_token = current_typing_cancel.clone();
                    }
                } else {
                    // No more queued messages, exit loop
                    break;
                }
            }
        });
    }

    /// Handle interactive payloads (button clicks for approvals and command buttons).
    async fn handle_interactive(&self, payload: &Value) {
        let interaction_type = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if interaction_type != "block_actions" {
            return;
        }

        let user_id = payload
            .pointer("/user/id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Only owners can approve tools or invoke privileged command buttons.
        if !self.is_owner_id(user_id) {
            warn!(user_id, "Unauthorized Slack interactive action");
            return;
        }

        let actions = match payload.get("actions").and_then(|v| v.as_array()) {
            Some(a) => a,
            None => return,
        };

        for action in actions {
            let action_id = action
                .get("action_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Handle command buttons (e.g., "cmd:restart", "cmd:reload")
            if let Some(command) = action_id.strip_prefix("cmd:") {
                let channel_id = payload
                    .pointer("/channel/id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let thread_ts = payload
                    .pointer("/message/thread_ts")
                    .or_else(|| payload.pointer("/message/ts"))
                    .and_then(|v| v.as_str());
                let session_id = self.build_session_id(channel_id, thread_ts);
                let cmd_text = format!("/{}", command);
                let reply = self
                    .dispatch_command(&cmd_text, &session_id, UserRole::Owner)
                    .await;

                // Respond via response_url to update the message
                if let Some(response_url) = payload.get("response_url").and_then(|v| v.as_str()) {
                    let mrkdwn = markdown_to_slack_mrkdwn(&reply);
                    let updated = serde_json::json!({
                        "replace_original": false,
                        "text": mrkdwn,
                    });
                    let _ = self.http.post(response_url).json(&updated).send().await;
                }
                continue;
            }

            let parts: Vec<&str> = action_id.splitn(3, ':').collect();
            if parts.len() != 3 || parts[0] != "approve" {
                continue;
            }

            let action_type = parts[1];
            let approval_id = parts[2];

            let response = match action_type {
                "once" => ApprovalResponse::AllowOnce,
                "session" => ApprovalResponse::AllowSession,
                "always" => ApprovalResponse::AllowAlways,
                "deny" => ApprovalResponse::Deny,
                _ => continue,
            };

            // Resolve only a currently pending approval. In particular, do not
            // update or auto-dismiss a stale card after a daemon restart.
            let response_tx = {
                let mut pending = self.pending_approvals.lock().await;
                pending.remove(approval_id)
            };
            let Some(response_tx) = response_tx else {
                warn!(approval_id, "Stale Slack approval callback");
                continue;
            };

            // Update the original message via response_url
            if let Some(response_url) = payload.get("response_url").and_then(|v| v.as_str()) {
                let original_text = payload
                    .pointer("/message/text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Command approval");
                let updated =
                    super::approval_render::finalize_approval_message(original_text, &response);
                let updated_payload = serde_json::json!({
                    "replace_original": true,
                    "text": updated,
                });
                let _ = self
                    .http
                    .post(response_url)
                    .json(&updated_payload)
                    .send()
                    .await;

                // Keep the resolved state visible briefly, then delete the
                // interactive source message via its short-lived response URL.
                // Slack permits response URLs to delete source messages, and
                // the default 10-second delay is well inside their validity.
                if let Some(delay) = Self::resolved_approval_dismiss_delay() {
                    let http = self.http.clone();
                    let response_url = response_url.to_string();
                    tokio::spawn(async move {
                        tokio::time::sleep(delay).await;
                        let delete_payload = serde_json::json!({
                            "delete_original": true,
                        });
                        let _ = http.post(response_url).json(&delete_payload).send().await;
                    });
                }
            }

            let _ = response_tx.send(response);
        }
    }

    /// Handle slash commands from Slack.
    async fn handle_slash_command(&self, payload: &Value) {
        let user_id = payload
            .get("user_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let channel_id = payload
            .get("channel_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let command = payload
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let text_arg = payload.get("text").and_then(|v| v.as_str()).unwrap_or("");

        // Native slash commands are privileged control-plane actions.
        if !self.is_owner_id(user_id) {
            warn!(user_id, "Unauthorized Slack slash command");
            return;
        }

        let cmd_text = if text_arg.is_empty() {
            command.to_string()
        } else {
            format!("{} {}", command, text_arg)
        };

        let session_id = self.build_session_id(channel_id, None);
        let reply = self
            .dispatch_command(&cmd_text, &session_id, UserRole::Owner)
            .await;

        // Respond via response_url if available
        if let Some(response_url) = payload.get("response_url").and_then(|v| v.as_str()) {
            let mrkdwn = markdown_to_slack_mrkdwn(&reply);
            let body = serde_json::json!({
                "response_type": "ephemeral",
                "text": mrkdwn,
            });
            let _ = self.http.post(response_url).json(&body).send().await;
        } else {
            let _ = self.post_message(channel_id, &reply, None).await;
        }
    }

    /// Dispatch a command string and return the reply text.
    /// Accepts both `/command` and `!command` syntax (Slack reserves `/` for native slash commands).
    async fn dispatch_command(&self, text: &str, session_id: &str, user_role: UserRole) -> String {
        let parts: Vec<&str> = text.splitn(2, ' ').collect();
        // Normalize !prefix to /prefix so !restart works like /restart in Slack
        let normalized_cmd = if let Some(stripped) = parts[0].strip_prefix('!') {
            format!("/{}", stripped)
        } else {
            parts[0].to_string()
        };
        let cmd = normalized_cmd.as_str();
        let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

        // Try the shared command dispatcher first.
        let ctx = crate::channels::commands::CommandContext {
            agent: Arc::clone(&self.agent),
            state: Arc::clone(&self.state),
            task_registry: Arc::clone(&self.task_registry),
            config_path: self.config_path.clone(),
        };
        if let Some(reply) = ctx.dispatch(cmd, arg, session_id, user_role).await {
            return reply;
        }

        // Channel-specific commands.
        match cmd {
            "/restart" => {
                restart_process();
                "Restart failed. You may need to restart manually.".to_string()
            }
            "/workspace" => {
                "Use `!workspace ...` as a message in the private Slack group where the grant should apply. Type `!help` for the exact syntax."
                    .to_string()
            }
            "/help" | "/start" => build_help_text(&slack_commands(), "!"),
            _ => format!(
                "Unknown command: {}\nType `!help` for available commands.",
                cmd
            ),
        }
    }

    /// Dispatch a command and return the reply text plus optional Block Kit buttons.
    async fn dispatch_command_with_buttons(
        &self,
        text: &str,
        session_id: &str,
        user_role: UserRole,
    ) -> (String, Vec<Value>) {
        let reply = self.dispatch_command(text, session_id, user_role).await;

        // Normalize the command to determine which buttons to show
        let parts: Vec<&str> = text.splitn(2, ' ').collect();
        let cmd = if let Some(stripped) = parts[0].strip_prefix('!') {
            format!("/{}", stripped)
        } else {
            parts[0].to_string()
        };

        let buttons = match cmd.as_str() {
            "/help" | "/start" => Self::command_action_buttons(&[
                ("Restart", "cmd:restart"),
                ("Reload", "cmd:reload"),
                ("Clear", "cmd:clear"),
                ("Cost", "cmd:cost"),
            ]),
            "/reload" => Self::command_action_buttons(&[("Restart", "cmd:restart")]),
            _ => vec![],
        };

        (reply, buttons)
    }

    /// Build Block Kit button elements for command actions.
    fn command_action_buttons(commands: &[(&str, &str)]) -> Vec<Value> {
        commands
            .iter()
            .map(|(label, action_id)| {
                serde_json::json!({
                    "type": "button",
                    "text": { "type": "plain_text", "text": label },
                    "action_id": action_id,
                })
            })
            .collect()
    }

    /// Build a session ID from channel and thread.
    fn build_session_id(&self, channel_id: &str, thread_ts: Option<&str>) -> String {
        let base = match thread_ts {
            Some(ts) if self.use_threads => format!("slack:{}:{}", channel_id, ts),
            _ => format!("slack:{}", channel_id),
        };
        self.session_id(&base)
    }

    /// Determine the thread_ts to use when replying.
    fn reply_thread_ts(
        &self,
        message_ts: &str,
        existing_thread_ts: Option<&str>,
    ) -> Option<String> {
        if self.use_threads {
            Some(existing_thread_ts.unwrap_or(message_ts).to_string())
        } else {
            existing_thread_ts.map(|s| s.to_string())
        }
    }

    /// Post a message to a Slack channel.
    async fn post_message(
        &self,
        channel: &str,
        text: &str,
        thread_ts: Option<&str>,
    ) -> anyhow::Result<Value> {
        let mut body = serde_json::json!({
            "channel": channel,
            "text": text,
        });
        if let Some(ts) = thread_ts {
            body["thread_ts"] = Value::String(ts.to_string());
        }
        self.slack_api_post("chat.postMessage", &body).await
    }

    /// Post a message with Block Kit blocks (e.g., buttons) to a Slack channel.
    async fn post_message_with_blocks(
        &self,
        channel: &str,
        text: &str,
        blocks: Value,
        thread_ts: Option<&str>,
    ) -> anyhow::Result<Value> {
        let mut body = serde_json::json!({
            "channel": channel,
            "text": text,
            "blocks": blocks,
        });
        if let Some(ts) = thread_ts {
            body["thread_ts"] = Value::String(ts.to_string());
        }
        self.slack_api_post("chat.postMessage", &body).await
    }

    /// Handle incoming file attachments.
    async fn handle_incoming_files(
        &self,
        files: &[Value],
    ) -> anyhow::Result<Vec<crate::traits::MessageAttachment>> {
        let mut attachments = Vec::new();

        for file in files {
            let filename = file.get("name").and_then(|v| v.as_str()).unwrap_or("file");
            let file_size = file.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
            let mime_type = file
                .get("mimetype")
                .and_then(|v| v.as_str())
                .unwrap_or("application/octet-stream");

            let max_bytes = self.max_file_size_mb * 1_048_576;
            if file_size > max_bytes {
                anyhow::bail!(
                    "File too large ({:.1} MB). Maximum is {} MB.",
                    file_size as f64 / 1_048_576.0,
                    self.max_file_size_mb
                );
            }

            let download_url = match file.get("url_private_download").and_then(|v| v.as_str()) {
                Some(u) => u,
                None => {
                    tracing::warn!(
                        filename,
                        "Slack file missing url_private_download; skipping"
                    );
                    continue;
                }
            };

            let resp = self
                .http
                .get(download_url)
                .header("Authorization", format!("Bearer {}", self.bot_token))
                .send()
                .await?;

            if !resp.status().is_success() {
                anyhow::bail!("Failed to download Slack file: HTTP {}", resp.status());
            }
            let bytes = resp.bytes().await?;

            let sanitized = sanitize_filename(filename);
            let uuid_prefix = uuid::Uuid::new_v4().to_string()[..8].to_string();
            let dest_name = format!("{}_{}", uuid_prefix, sanitized);
            let dest_path = self.inbox_dir.join(&dest_name);

            std::fs::create_dir_all(&self.inbox_dir)?;
            std::fs::write(&dest_path, &bytes)?;

            info!(
                file = %dest_path.display(),
                size = bytes.len(),
                mime = %mime_type,
                "Saved inbound Slack file"
            );

            attachments.push(super::attachments::message_attachment(
                dest_path,
                sanitized,
                mime_type.to_string(),
                bytes.len() as u64,
            ));
        }

        Ok(attachments)
    }

    /// Upload a file to Slack using the new files.getUploadURLExternal flow.
    async fn upload_file(
        &self,
        channels: &str,
        filename: &str,
        data: &[u8],
        thread_ts: Option<&str>,
    ) -> anyhow::Result<()> {
        // Step 1: Get upload URL (uses form-encoded, not JSON)
        let url = "https://slack.com/api/files.getUploadURLExternal";
        let form_resp = self
            .http
            .post(url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .form(&[
                ("filename", filename.to_string()),
                ("length", data.len().to_string()),
            ])
            .send()
            .await?;
        let resp: Value = form_resp.json().await?;
        if resp.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            let error = resp
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            anyhow::bail!(
                "Slack API files.getUploadURLExternal failed: {} (response: {})",
                error,
                resp
            );
        }
        let upload_url = resp
            .get("upload_url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("No upload_url in response"))?
            .to_string();
        let file_id = resp
            .get("file_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("No file_id in response"))?
            .to_string();

        // Step 2: Upload the file (Slack docs specify POST, not PUT)
        let upload_resp = self
            .http
            .post(&upload_url)
            .header("Content-Type", "application/octet-stream")
            .body(data.to_vec())
            .send()
            .await?;
        if !upload_resp.status().is_success() {
            let status = upload_resp.status();
            let body = upload_resp.text().await.unwrap_or_default();
            anyhow::bail!("Slack file upload failed with status {}: {}", status, body);
        }
        // Step 3: Complete the upload and share to channel
        let mut complete_params = serde_json::json!({
            "files": [{"id": file_id, "title": filename}],
            "channel_id": channels,
        });
        if let Some(ts) = thread_ts {
            complete_params["thread_ts"] = Value::String(ts.to_string());
        }
        self.slack_api_post("files.completeUploadExternal", &complete_params)
            .await?;
        Ok(())
    }

    /// Parse a session ID to get (channel_id, thread_ts).
    fn parse_session_id(&self, session_id: &str) -> (String, Option<String>) {
        // Strip bot name prefix if present: "{bot_name}:slack:..." -> "slack:..."
        // session_id() prefixes with get_bot_name(), not name()/channel_name()
        let bot_name = self.get_bot_name();
        let without_prefix = if bot_name != "slack" {
            session_id
                .strip_prefix(&format!("{}:", bot_name))
                .unwrap_or(session_id)
        } else {
            session_id
        };

        // Format: "slack:{channel_id}:{thread_ts}" or "slack:{channel_id}"
        let stripped = without_prefix
            .strip_prefix("slack:")
            .unwrap_or(without_prefix);
        let parts: Vec<&str> = stripped.splitn(2, ':').collect();
        let channel_id = parts[0].to_string();
        let thread_ts = parts.get(1).map(|s| s.to_string());
        (channel_id, thread_ts)
    }

    /// Create a lightweight clone for use in spawned tasks (typing indicator).
    fn clone_for_typing(&self) -> SlackApiHandle {
        SlackApiHandle {
            http: self.http.clone(),
            bot_token: self.bot_token.clone(),
        }
    }

    /// Create a lightweight clone for use in spawned tasks (status updates).
    fn clone_for_status(&self) -> SlackApiHandle {
        SlackApiHandle {
            http: self.http.clone(),
            bot_token: self.bot_token.clone(),
        }
    }
}

/// Lightweight handle for making Slack API calls from spawned tasks.
struct SlackApiHandle {
    http: reqwest::Client,
    bot_token: String,
}

impl SlackApiHandle {
    async fn post_message(
        &self,
        channel: &str,
        text: &str,
        thread_ts: Option<&str>,
    ) -> anyhow::Result<()> {
        slack_post_message(&self.http, &self.bot_token, channel, text, thread_ts).await?;
        Ok(())
    }

    async fn add_reaction(&self, channel: &str, ts: &str, name: &str) -> anyhow::Result<()> {
        let body = serde_json::json!({
            "channel": channel,
            "timestamp": ts,
            "name": name,
        });
        let url = "https://slack.com/api/reactions.add";
        let _ = self
            .http
            .post(url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .json(&body)
            .send()
            .await;
        Ok(())
    }

    async fn remove_reaction(&self, channel: &str, ts: &str, name: &str) -> anyhow::Result<()> {
        let body = serde_json::json!({
            "channel": channel,
            "timestamp": ts,
            "name": name,
        });
        let url = "https://slack.com/api/reactions.remove";
        let _ = self
            .http
            .post(url)
            .header("Authorization", format!("Bearer {}", self.bot_token))
            .json(&body)
            .send()
            .await;
        Ok(())
    }
}

/// Convert `@DisplayName` mentions back to `<@USERID>` using a user cache snapshot.
/// Standalone function usable outside of `&self` contexts (e.g., spawned tasks).
fn restore_mentions_from_cache(text: &str, cache: &HashMap<String, String>) -> String {
    if cache.is_empty() {
        return text.to_string();
    }
    let mut result = text.to_string();
    // Sort by name length descending so "Jane Smith" matches before "Jane"
    let mut entries: Vec<_> = cache.iter().collect();
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.1.len()));
    for (user_id, name) in entries {
        let mention = format!("@{}", name);
        if result.contains(&mention) {
            let slack_mention = format!("<@{}>", user_id);
            result = result.replace(&mention, &slack_mention);
        }
    }
    result
}

/// Free function to post a message via Slack Web API.
async fn slack_post_message(
    http: &reqwest::Client,
    bot_token: &str,
    channel: &str,
    text: &str,
    thread_ts: Option<&str>,
) -> anyhow::Result<Value> {
    let mut body = serde_json::json!({
        "channel": channel,
        "text": text,
    });
    if let Some(ts) = thread_ts {
        body["thread_ts"] = Value::String(ts.to_string());
    }
    let mut resp = http
        .post("https://slack.com/api/chat.postMessage")
        .header("Authorization", format!("Bearer {}", bot_token))
        .json(&body)
        .send()
        .await?;
    if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        if let Some(delay) = crate::channels::rate_limit::retry_after_from_headers(
            resp.headers(),
            crate::channels::rate_limit::default_retry_after_cap(),
        ) {
            crate::channels::rate_limit::sleep_after_rate_limit("slack", delay).await;
            resp = http
                .post("https://slack.com/api/chat.postMessage")
                .header("Authorization", format!("Bearer {}", bot_token))
                .json(&body)
                .send()
                .await?;
        }
    }
    let result: Value = resp.json().await?;
    if result.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        let error = result
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        anyhow::bail!("chat.postMessage failed: {}", error);
    }
    Ok(result)
}

#[async_trait]
impl Channel for SlackChannel {
    fn name(&self) -> String {
        self.channel_name()
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            markdown: true,
            inline_buttons: true,
            media: true,
            max_message_len: MAX_MESSAGE_LEN,
        }
    }

    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
        let (channel_id, thread_ts) = self.parse_session_id(session_id);
        let mrkdwn = markdown_to_slack_mrkdwn(text);
        // Convert @DisplayName back to <@USERID> so Slack renders proper mentions
        let mrkdwn = self.restore_user_mentions(&mrkdwn).await;
        for chunk in split_message(&mrkdwn, MAX_MESSAGE_LEN) {
            self.post_message(&channel_id, &chunk, thread_ts.as_deref())
                .await?;
        }
        Ok(())
    }

    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        let (channel_id, thread_ts) = self.parse_session_id(session_id);
        match &media.kind {
            MediaKind::Photo { data } => {
                self.upload_file(&channel_id, "screenshot.png", data, thread_ts.as_deref())
                    .await?;
            }
            MediaKind::Document {
                file_path,
                filename,
            } => {
                let data = tokio::fs::read(file_path).await?;
                self.upload_file(&channel_id, filename, &data, thread_ts.as_deref())
                    .await?;
            }
        }
        if !media.caption.is_empty() {
            self.post_message(&channel_id, &media.caption, thread_ts.as_deref())
                .await?;
        }
        Ok(())
    }

    async fn request_approval(
        &self,
        session_id: &str,
        command: &str,
        risk_level: RiskLevel,
        warnings: &[String],
        permission_mode: PermissionMode,
    ) -> anyhow::Result<ApprovalResponse> {
        let (channel_id, thread_ts) = self.parse_session_id(session_id);

        let approval_id = uuid::Uuid::new_v4().to_string();
        let short_id = &approval_id[..8];

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        {
            let mut pending = self.pending_approvals.lock().await;
            pending.insert(approval_id.clone(), response_tx);
            info!(
                approval_id = %short_id,
                pending_count = pending.len(),
                risk = %risk_level,
                mode = %permission_mode,
                "Stored pending Slack approval"
            );
        }

        let use_session_button =
            super::approval_render::approval_use_session_button(permission_mode, risk_level);

        let message_text = super::approval_render::build_approval_message_slack(
            command,
            risk_level,
            warnings,
            use_session_button,
        );

        // Build Block Kit message with approval buttons
        let action_buttons = if use_session_button {
            serde_json::json!([
                {
                    "type": "button",
                    "text": { "type": "plain_text", "text": "Allow Once" },
                    "action_id": format!("approve:once:{}", approval_id),
                    "style": "primary"
                },
                {
                    "type": "button",
                    "text": { "type": "plain_text", "text": "Allow Session" },
                    "action_id": format!("approve:session:{}", approval_id)
                },
                {
                    "type": "button",
                    "text": { "type": "plain_text", "text": "Deny" },
                    "action_id": format!("approve:deny:{}", approval_id),
                    "style": "danger"
                }
            ])
        } else {
            serde_json::json!([
                {
                    "type": "button",
                    "text": { "type": "plain_text", "text": "Allow Once" },
                    "action_id": format!("approve:once:{}", approval_id),
                    "style": "primary"
                },
                {
                    "type": "button",
                    "text": { "type": "plain_text", "text": "Allow Always" },
                    "action_id": format!("approve:always:{}", approval_id)
                },
                {
                    "type": "button",
                    "text": { "type": "plain_text", "text": "Deny" },
                    "action_id": format!("approve:deny:{}", approval_id),
                    "style": "danger"
                }
            ])
        };

        let blocks = serde_json::json!([
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message_text
                }
            },
            {
                "type": "actions",
                "elements": action_buttons
            }
        ]);

        let mut body = serde_json::json!({
            "channel": channel_id,
            "text": message_text,
            "blocks": blocks,
        });
        if let Some(ts) = &thread_ts {
            body["thread_ts"] = Value::String(ts.to_string());
        }

        match self.slack_api_post("chat.postMessage", &body).await {
            Ok(_) => {
                info!(approval_id = %short_id, "Approval message sent to Slack");
            }
            Err(e) => {
                warn!("Failed to send Slack approval request: {}", e);
                let mut pending = self.pending_approvals.lock().await;
                pending.remove(&approval_id);
                return Err(anyhow::anyhow!(
                    "Slack approval request could not be delivered: {e}"
                ));
            }
        }

        // Wait for response with 5-minute timeout
        info!(approval_id = %short_id, "Waiting for user approval response...");
        match tokio::time::timeout(Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                warn!(approval_id = %short_id, "Approval channel closed");
                self.pending_approvals.lock().await.remove(&approval_id);
                Err(anyhow::anyhow!("Slack approval response channel closed"))
            }
            Err(_) => {
                warn!(approval_id = %short_id, "Approval timed out after 5 minutes");
                let mut pending = self.pending_approvals.lock().await;
                pending.remove(&approval_id);
                Err(anyhow::anyhow!("Slack approval request timed out"))
            }
        }
    }
}

/// Replace the current process with a fresh instance of itself.
fn restart_process() {
    use std::os::unix::process::CommandExt;

    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            tracing::error!("Failed to get current exe path: {}", e);
            return;
        }
    };
    let args: Vec<String> = std::env::args().skip(1).collect();

    tracing::info!(exe = %exe.display(), "Exec-ing new process");

    let err = std::process::Command::new(&exe).args(&args).exec();
    tracing::error!("exec failed: {}", err);
}

/// Spawn a SlackChannel in a background task.
/// This is a separate function to avoid async type inference cycles.
pub fn spawn_slack_channel(channel: Arc<SlackChannel>) {
    tokio::spawn(async move {
        channel.start_with_retry().await;
    });
}

#[cfg(test)]
mod authorization_tests {
    use super::*;

    fn ids(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn explicit_slack_owners_are_distinct_from_allowed_guests() {
        let allowed = ids(&["U_OWNER", "U_GUEST"]);
        let owners = ids(&["U_OWNER"]);
        assert_eq!(
            determine_slack_role(&allowed, &owners, "U_OWNER", false, true),
            Some(UserRole::Owner)
        );
        assert_eq!(
            determine_slack_role(&allowed, &owners, "U_GUEST", false, true),
            Some(UserRole::Guest)
        );
    }

    #[test]
    fn legacy_allowlist_remains_owner_when_no_owner_ids_exist() {
        assert_eq!(
            determine_slack_role(&ids(&["U_LEGACY"]), &[], "U_LEGACY", false, true),
            Some(UserRole::Owner)
        );
    }

    #[test]
    fn grant_confers_guest_only_in_addressed_group_flow() {
        assert_eq!(
            determine_slack_role(&[], &ids(&["U_OWNER"]), "U_COLLAB", true, true),
            Some(UserRole::Guest)
        );
        assert_eq!(
            determine_slack_role(&[], &ids(&["U_OWNER"]), "U_OUTSIDER", false, true),
            Some(UserRole::Public)
        );
        assert_eq!(
            determine_slack_role(&[], &ids(&["U_OWNER"]), "U_OUTSIDER", false, false),
            None
        );
    }

    #[test]
    fn workspace_mentions_require_stable_slack_user_ids() {
        assert_eq!(
            SlackChannel::parse_slack_user_mention("<@U123ABC>").unwrap(),
            "U123ABC"
        );
        assert!(SlackChannel::parse_slack_user_mention("@display-name").is_err());
        assert!(SlackChannel::parse_slack_user_mention("<@../../etc>").is_err());
    }

    #[test]
    fn queued_grant_revalidation_never_expands_authority() {
        let original = WorkspaceGrant {
            platform: "slack".to_string(),
            workspace_id: "T1".to_string(),
            channel_id: "G1".to_string(),
            user_id: "U1".to_string(),
            project_root: "/tmp/project-a".to_string(),
            access: WorkspaceAccessLevel::Edit,
            expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
        };

        let mut extended = original.clone();
        extended.expires_at = chrono::Utc::now() + chrono::Duration::hours(2);
        let retained = retain_queued_workspace_grant(&original, &extended).unwrap();
        assert_eq!(retained.expires_at, original.expires_at);
        assert_eq!(retained.access, WorkspaceAccessLevel::Edit);

        let mut downgraded = extended.clone();
        downgraded.access = WorkspaceAccessLevel::Read;
        let retained = retain_queued_workspace_grant(&original, &downgraded).unwrap();
        assert_eq!(retained.access, WorkspaceAccessLevel::Read);

        let mut moved = extended;
        moved.project_root = "/tmp/project-b".to_string();
        assert!(retain_queued_workspace_grant(&original, &moved).is_none());
    }
}
