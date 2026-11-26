//! The Slack bot: [`crate::SlackChannel`] holds the Socket Mode
//! connection; this module decides who may talk to the bot (allowlist,
//! owners, workspace grants), which session a message belongs to, answers
//! `!` and `/` commands and runs turns on the shared
//! [`aidaemon_channels::turns::TurnRunner`]. Where the access lists and
//! grants are kept, and which directories may be delegated, is the
//! application's ([`SlackHost`]).

use aidaemon_channels::prompts::PromptSurface as _;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use aidaemon_channels::Channel as _;
use aidaemon_channels::bot::{BotDeps, SharedCommandDispatch};
use aidaemon_channels::context::{ChannelContext, WorkspaceAccessLevel, WorkspaceGrant};
use aidaemon_channels::hub::{
    ChannelCapabilities, ChannelHub, HubChannel, MediaKind, MediaMessage,
};
use aidaemon_channels::prompts::Prompts;
use aidaemon_channels::slash::{CommandCategory, CommandDef, build_help_text, shared_commands};
use aidaemon_channels::tasks::QueuedMessage;
use aidaemon_channels::turns::{Turn, TurnRunner, TurnSurface};
use aidaemon_core::ChannelVisibility;
use aidaemon_events::conversation::MessageAttachment;
use aidaemon_tools::semantics::{
    ApprovalResponse, GoalConfirmationStyle, PermissionMode, RiskLevel, UserRole,
};
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

use crate::format::{markdown_to_mrkdwn as markdown_to_slack_mrkdwn, split_mrkdwn};

/// What the bot needs from the application: the shared slash commands,
/// where the access lists and workspace grants are kept, and which project
/// directories may be delegated.
#[async_trait]
pub trait SlackHost: SharedCommandDispatch + 'static {
    /// `!restart`: restart the application; returns only if it could not.
    fn restart(&self) {}

    /// Keep the allowlist and owner ids together (a first user claimed the
    /// bot), so a restart keeps both.
    async fn persist_access_ids(
        &self,
        allowed_ids: &[String],
        owner_ids: &[String],
    ) -> anyhow::Result<()>;

    /// The workspace grants as stored now (another bot may have changed
    /// them).
    async fn load_workspace_grants(&self) -> anyhow::Result<Vec<WorkspaceGrant>>;

    /// Apply one change to the stored grants, merged with what is stored
    /// now; returns the active grants and whether anything matched.
    async fn mutate_workspace_grants(
        &self,
        mutation: WorkspaceGrantMutation,
    ) -> anyhow::Result<(Vec<WorkspaceGrant>, bool)>;

    /// The canonical project root `raw_path` names, if it may be delegated
    /// to a collaborator; the error says why not.
    async fn resolve_workspace_project_root(&self, raw_path: &str) -> anyhow::Result<String>;
}

/// Longest plain-text message posted here (Slack's text limit is 40,000;
/// leave margin). Text in a Block Kit section is limited to the adapter's
/// [`crate::SECTION_TEXT_LIMIT`].
const MAX_MESSAGE_LEN: usize = 39_000;
const DEFAULT_WORKSPACE_GRANT_HOURS: i64 = 24;
const MAX_WORKSPACE_GRANT_HOURS: i64 = 168;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlackInboundEventKind {
    Message,
    FileShared,
    Ignore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SlackFileShareLocation {
    message_ts: String,
    thread_ts: Option<String>,
    is_private: bool,
}

fn classify_slack_inbound_event(event: &Value) -> SlackInboundEventKind {
    match event.get("type").and_then(Value::as_str).unwrap_or("") {
        "message" | "app_mention" => match event.get("subtype").and_then(Value::as_str) {
            None | Some("file_share") => SlackInboundEventKind::Message,
            Some(_) => SlackInboundEventKind::Ignore,
        },
        "file_shared" => SlackInboundEventKind::FileShared,
        _ => SlackInboundEventKind::Ignore,
    }
}

fn slack_file_share_location(
    file: &Value,
    channel_id: &str,
    event_ts: Option<&str>,
) -> Option<SlackFileShareLocation> {
    let mut locations = Vec::new();
    for (visibility, is_private) in [("private", true), ("public", false)] {
        let Some(shares) = file
            .get("shares")
            .and_then(|value| value.get(visibility))
            .and_then(|value| value.get(channel_id))
            .and_then(Value::as_array)
        else {
            continue;
        };

        locations.extend(shares.iter().filter_map(|share| {
            let message_ts = share.get("ts").and_then(Value::as_str)?.to_string();
            let thread_ts = share
                .get("thread_ts")
                .and_then(Value::as_str)
                .map(str::to_string);
            Some(SlackFileShareLocation {
                message_ts,
                thread_ts,
                is_private,
            })
        }));
    }

    if locations.is_empty() {
        return None;
    }

    if let Some(event_ts) = event_ts {
        if let Some(exact) = locations
            .iter()
            .find(|location| location.message_ts == event_ts)
        {
            return Some(exact.clone());
        }

        if let Ok(event_number) = event_ts.parse::<f64>() {
            locations.sort_by(|left, right| {
                let left_distance = left
                    .message_ts
                    .parse::<f64>()
                    .map(|value| (value - event_number).abs())
                    .unwrap_or(f64::INFINITY);
                let right_distance = right
                    .message_ts
                    .parse::<f64>()
                    .map(|value| (value - event_number).abs())
                    .unwrap_or(f64::INFINITY);
                left_distance
                    .total_cmp(&right_distance)
                    .then_with(|| right.message_ts.cmp(&left.message_ts))
            });
            return locations.into_iter().next();
        }
    }

    locations.into_iter().max_by(|left, right| {
        match (
            left.message_ts.parse::<f64>(),
            right.message_ts.parse::<f64>(),
        ) {
            (Ok(left), Ok(right)) => left.total_cmp(&right),
            _ => left.message_ts.cmp(&right.message_ts),
        }
    })
}

fn slack_channel_type(channel: &Value) -> &'static str {
    if channel.get("is_im").and_then(Value::as_bool) == Some(true) {
        "im"
    } else if channel.get("is_mpim").and_then(Value::as_bool) == Some(true) {
        "mpim"
    } else if channel.get("is_private").and_then(Value::as_bool) == Some(true) {
        "group"
    } else {
        "channel"
    }
}

fn fallback_slack_channel_type(channel_id: &str, is_private: bool) -> &'static str {
    if channel_id.starts_with('D') {
        "im"
    } else if is_private {
        "group"
    } else {
        "channel"
    }
}

fn normalize_file_shared_message(
    mut message: Value,
    file_event: &Value,
    file: &Value,
    channel_id: &str,
    channel_type: &str,
) -> anyhow::Result<Value> {
    let message_object = message
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Slack file share message was not an object"))?;
    message_object.insert("type".to_string(), Value::String("message".to_string()));
    message_object.insert("channel".to_string(), Value::String(channel_id.to_string()));
    message_object.insert(
        "channel_type".to_string(),
        Value::String(channel_type.to_string()),
    );

    if !message_object.contains_key("user")
        && let Some(user_id) = file_event
            .get("user_id")
            .or_else(|| file_event.get("user"))
            .and_then(Value::as_str)
            .or_else(|| file.get("user").and_then(Value::as_str))
    {
        message_object.insert("user".to_string(), Value::String(user_id.to_string()));
    }

    let has_files = message_object
        .get("files")
        .and_then(Value::as_array)
        .is_some_and(|files| !files.is_empty());
    if !has_files {
        message_object.insert("files".to_string(), Value::Array(vec![file.clone()]));
    }

    Ok(message)
}

/// One change to the stored workspace grants.
pub enum WorkspaceGrantMutation {
    Upsert(WorkspaceGrant),
    Revoke {
        platform: String,
        workspace_id: String,
        channel_id: String,
        user_id: String,
    },
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
    host: &dyn SlackHost,
    original: &WorkspaceGrant,
) -> anyhow::Result<Option<WorkspaceGrant>> {
    let grants = host.load_workspace_grants().await?;
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
pub struct SlackBot {
    /// Bot name fetched from Slack API (e.g., "my_bot").
    /// Populated on first connection.
    bot_name: std::sync::RwLock<String>,
    allowed_user_ids: std::sync::RwLock<Vec<String>>,
    /// Explicit owners from `users.owner_ids.slack`. When empty, allowed users
    /// retain legacy owner behavior for backward compatibility.
    owner_user_ids: std::sync::RwLock<Vec<String>>,
    /// The one-time code a direct message must carry to claim the bot
    /// while it has neither allowed users nor owners.
    owner_claim: aidaemon_channels::claim::OwnerClaim,
    workspace_grants: std::sync::RwLock<Vec<WorkspaceGrant>>,
    use_threads: bool,
    /// Threads (`channel:thread_ts`) the bot is known to be answering in.
    bot_threads: std::sync::Mutex<std::collections::HashSet<String>>,
    host: Arc<dyn SlackHost>,
    /// Our own bot user ID, resolved on first connection.
    bot_user_id: Mutex<Option<String>>,
    /// Slack team/workspace ID from `auth.test`; grants fail closed until known.
    team_id: std::sync::RwLock<Option<String>>,
    /// Reference to the channel hub for dynamic bot registration.
    channel_hub: std::sync::RwLock<Option<Weak<ChannelHub>>>,
    /// Cache of resolved Slack user IDs to display names.
    user_cache: RwLock<HashMap<String, String>>,
    /// Cache of channel ID → channel name (process-lifetime; channel names rarely change).
    channel_name_cache: RwLock<HashMap<String, String>>,
    /// Cache of channel ID → (member display names, fetched_at). TTL: 10 minutes.
    channel_members_cache: RwLock<HashMap<String, (Vec<String>, Instant)>>,
    deps: BotDeps,
    runner: TurnRunner,
    /// Holds the Socket Mode connection and asks prompts.
    transport: Arc<crate::SlackChannel>,
    prompts: Prompts,
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

/// One Slack bot's settings.
#[derive(Clone)]
pub struct SlackBotSettings {
    /// The Socket Mode app token (`xapp-...`).
    pub app_token: String,
    /// The bot token (`xoxb-...`).
    pub bot_token: String,
    pub allowed_user_ids: Vec<String>,
    /// Explicit owners; empty: every allowed user is an owner.
    pub owner_user_ids: Vec<String>,
    pub workspace_grants: Vec<WorkspaceGrant>,
    /// Answer top-level messages in a thread.
    pub use_threads: bool,
}

impl SlackBot {
    /// The code that claims the bot while it has no allowed users or owners.
    pub fn owner_claim(&self) -> &aidaemon_channels::claim::OwnerClaim {
        &self.owner_claim
    }

    /// Writes the claim code to the log when nobody is allowed or owns it.
    pub fn announce_claim_if_unowned(&self) {
        let unowned = self
            .allowed_user_ids
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty()
            && self
                .owner_user_ids
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty();
        if unowned {
            self.owner_claim.announce();
        }
    }

    pub fn new(settings: SlackBotSettings, deps: BotDeps, host: Arc<dyn SlackHost>) -> Self {
        let SlackBotSettings {
            app_token,
            bot_token,
            allowed_user_ids,
            owner_user_ids,
            workspace_grants,
            use_threads,
        } = settings;
        Self {
            runner: TurnRunner::new(deps.turn_deps()),
            deps,
            transport: Arc::new(crate::SlackChannel::new(bot_token, app_token)),
            prompts: Prompts::default(),
            bot_name: std::sync::RwLock::new("slack".to_string()),
            allowed_user_ids: std::sync::RwLock::new(allowed_user_ids),
            owner_user_ids: std::sync::RwLock::new(owner_user_ids),
            owner_claim: aidaemon_channels::claim::OwnerClaim::new("Slack", None),
            workspace_grants: std::sync::RwLock::new(workspace_grants),
            use_threads,
            host,
            bot_user_id: Mutex::new(None),
            bot_threads: std::sync::Mutex::default(),
            team_id: std::sync::RwLock::new(None),
            channel_hub: std::sync::RwLock::new(None),
            user_cache: RwLock::new(HashMap::new()),
            channel_name_cache: RwLock::new(HashMap::new()),
            channel_members_cache: RwLock::new(HashMap::new()),
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
        let grants = match self.host.load_workspace_grants().await {
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
                );
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
                            );
                        }
                    }
                } else {
                    DEFAULT_WORKSPACE_GRANT_HOURS
                };
                let project_root = match self.host.resolve_workspace_project_root(&tokens[4]).await
                {
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
                    .host
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
                    .host
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
                let latest = match self.host.load_workspace_grants().await {
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

    /// Start the Slack Socket Mode client, restarted with backoff when it
    /// stops.
    pub async fn start_with_retry(self: Arc<Self>) {
        aidaemon_channels::bot::supervise("Slack client", || async {
            if let Err(e) = self.clone().start().await {
                warn!("Slack client error: {}", e);
            }
        })
        .await
    }

    /// Resolve the bot's identity, then hold the connection.
    async fn start(self: Arc<Self>) -> anyhow::Result<()> {
        // Our own bot user ID filters out the bot's own messages.
        self.resolve_bot_info().await;
        self.start_sdk().await
    }

    /// The SDK adapter holds the connection (acks, pings, reconnects) and
    /// hands each envelope over as Slack sent it; the handling below reads
    /// Slack's own event shapes, so it is unchanged.
    async fn start_sdk(self: Arc<Self>) -> anyhow::Result<()> {
        let transport = self.transport.clone();
        let (envelopes_tx, mut envelopes) = tokio::sync::mpsc::channel(64);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let runner = {
            let transport = transport.clone();
            let shutdown = shutdown.clone();
            tokio::spawn(async move { transport.run_envelopes(envelopes_tx, shutdown).await })
        };
        info!("Slack Socket Mode connection held by the SDK adapter");
        while let Some(envelope) = envelopes.recv().await {
            let channel = Arc::clone(&self);
            tokio::spawn(async move {
                channel
                    .handle_envelope(serde_json::json!({
                        "type": envelope.kind,
                        "payload": envelope.payload,
                    }))
                    .await;
            });
        }
        shutdown.cancel();
        match runner.await {
            Ok(result) => result.map_err(|e| anyhow::anyhow!("Slack connection: {e}")),
            Err(e) => Err(anyhow::anyhow!("Slack connection task failed: {e}")),
        }
    }

    /// Resolve the bot's own user ID and name via `auth.test`.
    async fn resolve_bot_info(&self) {
        match self.transport.api("auth.test", None).await {
            Ok(resp) => {
                if let Some(user_id) = resp.get("user_id").and_then(|v| v.as_str()) {
                    let mut guard = self.bot_user_id.lock().await;
                    *guard = Some(user_id.to_string());
                }
                if let Some(team_id) = resp.get("team_id").and_then(|v| v.as_str())
                    && let Ok(mut guard) = self.team_id.write()
                {
                    *guard = Some(team_id.to_string());
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
        let json = self
            .transport
            .api_form("users.info", &[("user", user_id.to_string())])
            .await
            .ok()?;
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

        let json = match self
            .transport
            .api_form("conversations.info", &[("channel", channel_id.to_string())])
            .await
        {
            Ok(json) => json,
            Err(error) => {
                debug!(channel_id, %error, "Failed to resolve channel name");
                return None;
            }
        };
        let name = json.pointer("/channel/name").and_then(|v| v.as_str())?;
        let display_name = format!("#{}", name);
        self.channel_name_cache
            .write()
            .await
            .insert(channel_id.to_string(), display_name.clone());
        Some(display_name)
    }

    /// Resolve the display names of members in a channel via `conversations.members` + `users.info`.
    /// Results are cached with a 10-minute TTL. Limited to first 50 members.
    async fn resolve_channel_member_names(&self, channel_id: &str) -> Vec<String> {
        const MEMBERS_TTL: Duration = Duration::from_secs(600); // 10 minutes

        // Check cache (with TTL)
        if let Some((names, fetched_at)) = self.channel_members_cache.read().await.get(channel_id)
            && fetched_at.elapsed() < MEMBERS_TTL
        {
            return names.clone();
        }

        let json = match self
            .transport
            .api_form(
                "conversations.members",
                &[
                    ("channel", channel_id.to_string()),
                    ("limit", "50".to_string()),
                ],
            )
            .await
        {
            Ok(json) => json,
            Err(error) => {
                // Without the list the model has no names to mention. For a
                // private channel the bot needs `groups:read` as well as
                // `channels:read`.
                warn!(
                    channel_id,
                    %error,
                    "Failed to fetch channel members; replies here cannot mention anyone \
                     (private channels need the groups:read scope)"
                );
                return vec![];
            }
        };

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

        info!(
            channel_id,
            members = filtered_ids.len(),
            named = names.len(),
            "Slack channel members resolved"
        );

        // Cache the result
        self.channel_members_cache
            .write()
            .await
            .insert(channel_id.to_string(), (names.clone(), Instant::now()));

        names
    }

    async fn fetch_slack_file_info(&self, file_id: &str) -> anyhow::Result<Value> {
        let params = [("file", file_id.to_string())];
        let response = self.transport.api_form("files.info", &params).await?;
        response
            .get("file")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Slack files.info omitted file {}", file_id))
    }

    async fn resolve_slack_file_metadata(&self, file: &Value) -> anyhow::Result<Value> {
        let has_download_url = file
            .get("url_private_download")
            .or_else(|| file.get("url_private"))
            .and_then(Value::as_str)
            .is_some();
        let metadata_complete = file.get("name").and_then(Value::as_str).is_some()
            && file.get("mimetype").and_then(Value::as_str).is_some()
            && file.get("size").and_then(Value::as_u64).is_some();
        if has_download_url && metadata_complete {
            return Ok(file.clone());
        }

        let file_id = file
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("Slack file metadata omitted both a URL and file ID"))?;
        self.fetch_slack_file_info(file_id).await
    }

    async fn hydrate_file_shared_event(&self, event: &Value) -> anyhow::Result<Value> {
        const RETRY_DELAYS_MS: [u64; 3] = [200, 500, 1_000];

        let mut last_error = None;
        for attempt in 0..=RETRY_DELAYS_MS.len() {
            match self.try_hydrate_file_shared_event(event).await {
                Ok(message) => return Ok(message),
                Err(error) => {
                    debug!(
                        attempt = attempt + 1,
                        error = %error,
                        "Slack file_shared event was not ready"
                    );
                    last_error = Some(error);
                }
            }

            if let Some(delay_ms) = RETRY_DELAYS_MS.get(attempt) {
                tokio::time::sleep(Duration::from_millis(*delay_ms)).await;
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Slack file share hydration failed")))
    }

    async fn try_hydrate_file_shared_event(&self, event: &Value) -> anyhow::Result<Value> {
        let channel_id = event
            .get("channel_id")
            .or_else(|| event.get("channel"))
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("Slack file_shared event omitted channel_id"))?;
        let file_id = event
            .get("file_id")
            .and_then(Value::as_str)
            .or_else(|| event.pointer("/file/id").and_then(Value::as_str))
            .ok_or_else(|| anyhow::anyhow!("Slack file_shared event omitted file_id"))?;
        let event_ts = event.get("event_ts").and_then(Value::as_str);

        let file = self.fetch_slack_file_info(file_id).await?;
        let location = slack_file_share_location(&file, channel_id, event_ts).ok_or_else(|| {
            anyhow::anyhow!(
                "Slack files.info did not expose a share for file {} in channel {}",
                file_id,
                channel_id
            )
        })?;

        let (method, mut params) = if let Some(thread_ts) = location
            .thread_ts
            .as_deref()
            .filter(|thread_ts| *thread_ts != location.message_ts)
        {
            (
                "conversations.replies",
                vec![
                    ("channel", channel_id.to_string()),
                    ("ts", thread_ts.to_string()),
                ],
            )
        } else {
            (
                "conversations.history",
                vec![("channel", channel_id.to_string())],
            )
        };
        params.extend([
            ("oldest", location.message_ts.clone()),
            ("latest", location.message_ts.clone()),
            ("inclusive", "true".to_string()),
            ("limit", "15".to_string()),
        ]);
        let history = self.transport.api_form(method, &params).await?;
        let message = history
            .get("messages")
            .and_then(Value::as_array)
            .and_then(|messages| {
                messages.iter().find(|message| {
                    message.get("ts").and_then(Value::as_str) == Some(location.message_ts.as_str())
                })
            })
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Slack {} did not return file share message {}",
                    method,
                    location.message_ts
                )
            })?;

        let channel_type = if let Some(channel_type) = event
            .get("channel_type")
            .and_then(Value::as_str)
            .filter(|channel_type| matches!(*channel_type, "im" | "mpim" | "group" | "channel"))
        {
            channel_type.to_string()
        } else {
            let params = [("channel", channel_id.to_string())];
            match self.transport.api_form("conversations.info", &params).await {
                Ok(response) => response
                    .get("channel")
                    .map(slack_channel_type)
                    .unwrap_or_else(|| fallback_slack_channel_type(channel_id, location.is_private))
                    .to_string(),
                Err(error) => {
                    warn!(
                        channel_id,
                        error = %error,
                        "Could not resolve Slack channel type for file_shared event"
                    );
                    fallback_slack_channel_type(channel_id, location.is_private).to_string()
                }
            }
        };

        normalize_file_shared_message(message, event, &file, channel_id, &channel_type)
    }

    /// Handle a Socket Mode envelope.
    /// Handle one Socket Mode envelope as the connection delivers it.
    pub async fn handle_envelope(&self, envelope: Value) {
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

    fn remember_bot_thread(&self, channel_id: &str, thread_ts: &str) {
        let mut threads = self
            .bot_threads
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Bounded: a forgotten thread is asked of Slack again.
        if threads.len() >= 4096 {
            threads.clear();
        }
        threads.insert(format!("{channel_id}:{thread_ts}"));
    }

    /// Whether the bot has been brought into this thread: it was mentioned
    /// there since start, or Slack shows a message of its own in it. A
    /// failed lookup reads as no, so the bot stays quiet.
    async fn bot_in_thread(&self, channel_id: &str, thread_ts: &str) -> bool {
        let known = self
            .bot_threads
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(&format!("{channel_id}:{thread_ts}"));
        if known {
            return true;
        }
        let Some(bot_id) = self.bot_user_id.lock().await.clone() else {
            return false;
        };
        let replies = match self
            .transport
            .api_form(
                "conversations.replies",
                &[
                    ("channel", channel_id.to_string()),
                    ("ts", thread_ts.to_string()),
                    ("limit", "200".to_string()),
                ],
            )
            .await
        {
            Ok(replies) => replies,
            Err(error) => {
                debug!(channel_id, thread_ts, %error, "Could not read a Slack thread");
                return false;
            }
        };
        let joined = replies
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                messages
                    .iter()
                    .any(|m| m.get("user").and_then(Value::as_str) == Some(bot_id.as_str()))
            });
        if joined {
            self.remember_bot_thread(channel_id, thread_ts);
        }
        joined
    }

    /// Handle an Events API payload (message events, etc.).
    async fn handle_events_api(&self, payload: &Value) {
        let mut event = match payload.get("event").cloned() {
            Some(event) => event,
            None => return,
        };

        match classify_slack_inbound_event(&event) {
            SlackInboundEventKind::Message => {}
            SlackInboundEventKind::FileShared => {
                event = match self.hydrate_file_shared_event(&event).await {
                    Ok(message) => message,
                    Err(error) => {
                        warn!(
                            file_id = event
                                .get("file_id")
                                .and_then(|value| value.as_str())
                                .unwrap_or("unknown"),
                            channel_id = event
                                .get("channel_id")
                                .and_then(|value| value.as_str())
                                .unwrap_or("unknown"),
                            error = %error,
                            "Failed to hydrate Slack file_shared event"
                        );
                        return;
                    }
                };
            }
            SlackInboundEventKind::Ignore => return,
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
        // An unowned bot is claimed only by a direct message carrying its
        // one-time claim code; any other message to it is refused.
        let unclaimed = {
            let allowed = self
                .allowed_user_ids
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let owners = self
                .owner_user_ids
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            allowed.is_empty() && owners.is_empty()
        };
        let auto_claimed = if unclaimed && is_dm {
            match self.owner_claim.attempt(&raw_text) {
                aidaemon_channels::claim::ClaimAttempt::Claimed => {
                    let mut allowed = self
                        .allowed_user_ids
                        .write()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let mut owners = self
                        .owner_user_ids
                        .write()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    allowed.push(user.clone());
                    owners.push(user.clone());
                    true
                }
                aidaemon_channels::claim::ClaimAttempt::Refused => {
                    warn!(user = %user, "Direct message to an unclaimed Slack bot without its claim code");
                    let _ = self
                        .post_message(&channel_id, &self.owner_claim.refusal(), None)
                        .await;
                    return;
                }
            }
        } else {
            false
        };
        if auto_claimed {
            warn!(user = %user, "Slack bot claimed with its claim code.");
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
            if let Err(e) = self.host.persist_access_ids(&allowed_ids, &owner_ids).await {
                warn!(user = %user, "Failed to persist auto-claimed user ID to config: {}", e);
            }
            let _ = self.post_message(
                &channel_id,
                "Hey! You're now set as the owner. Ask me anything, give me tasks, or just chat.",
                None,
            ).await;
            // The message was the claim code: nothing for the agent.
            return;
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

        // In a channel or group only a message that addresses the bot is for
        // it: a mention from anyone, or from an owner a typed command or a
        // reply in a thread the bot is already answering in. Everything else
        // is people talking to each other, the owner included.
        let thread_root = event
            .get("thread_ts")
            .and_then(|v| v.as_str())
            .filter(|root| event.get("ts").and_then(|v| v.as_str()) != Some(root));
        if !is_dm && !bot_mentioned {
            let continues = user_role == UserRole::Owner
                // Mentioning another person is talking to that person.
                && !raw_text.contains("<@")
                && (raw_text.starts_with('!')
                    || raw_text.starts_with('/')
                    || match thread_root {
                        Some(root) => self.bot_in_thread(&channel_id, root).await,
                        None => false,
                    });
            if !continues {
                return;
            }
        } else if !is_dm {
            // A mention brings the bot into this thread, or opens one.
            let root = thread_root.or_else(|| {
                self.use_threads
                    .then(|| event.get("ts").and_then(|v| v.as_str()))
                    .flatten()
            });
            if let Some(root) = root {
                self.remember_bot_thread(&channel_id, root);
            }
        }

        // Strip bot @mention from text before processing
        let raw_agent_text = {
            let bot_id_guard = self.bot_user_id.lock().await;
            if let Some(bid) = bot_id_guard.as_ref() {
                raw_text
                    .replace(&format!("<@{}>", bid), "")
                    .trim()
                    .to_string()
            } else {
                raw_text.clone()
            }
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
        if self.deps.files_enabled
            && user_role != UserRole::Public
            && let Some(files) = event.get("files").and_then(|v| v.as_array())
        {
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

        // Build the text to send to the agent
        let agent_text = if inbound_attachments.is_empty() {
            if text.is_empty() {
                return;
            }
            text.clone()
        } else {
            if user_role == UserRole::Owner {
                aidaemon_channels::attachments::build_inbound_text(&text, &inbound_attachments)
            } else {
                aidaemon_channels::attachments::build_inbound_text_without_local_paths(
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
                self.dispatch_command_with_buttons(&agent_text, &session_id, &user, user_role)
                    .await
            };
            let mrkdwn = markdown_to_slack_mrkdwn(&reply);
            // The last chunk goes in a section block when it carries
            // buttons, so every chunk then fits one.
            let limit = if buttons.is_empty() {
                MAX_MESSAGE_LEN
            } else {
                crate::SECTION_TEXT_LIMIT
            };
            let chunks = split_mrkdwn(&mrkdwn, limit);
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
                in_run: None,
                fresh_agent_session: false,
                agents: Vec::new(),
            }
        };

        self.runner
            .register_chat(&session_id, &self.channel_name(), &channel_ctx)
            .await;
        info!(session_id, user_id = %user, "Received Slack message");

        let surface = SlackSurface {
            api: self.api_handle(),
            channel: channel_id.clone(),
            thread: reply_thread.clone(),
            // The hourglass goes on the message that started the work.
            reacting_to: std::sync::Mutex::new((!ts.is_empty()).then(|| ts.clone())),
            // Restores @DisplayName to a Slack mention in replies.
            user_cache: self.user_cache.read().await.clone(),
            host: self.host.clone(),
        };
        self.runner
            .submit(
                Arc::new(surface),
                Turn {
                    session_id,
                    text: agent_text,
                    attachments: inbound_attachments,
                    user_role,
                    channel_ctx,
                    // `ts` is unique per message and reused on Events API
                    // retries; without it, dedup falls back to the text.
                    dedup_key: (!ts.is_empty()).then_some(ts),
                    ingress_timing: None,
                },
            )
            .await;
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
                    .dispatch_command(&cmd_text, &session_id, user_id, UserRole::Owner)
                    .await;

                // Respond via response_url to update the message
                if let Some(response_url) = payload.get("response_url").and_then(|v| v.as_str()) {
                    let mrkdwn = markdown_to_slack_mrkdwn(&reply);
                    let updated = serde_json::json!({
                        "replace_original": false,
                        "text": mrkdwn,
                    });
                    let _ = self.transport.respond(response_url, &updated).await;
                }
                continue;
            }
        }

        let Some(answer) = crate::SlackChannel::approval_answer(payload) else {
            return;
        };
        if !self.prompts.is_open(&answer.prompt_id).await {
            // Not a card this run asked, e.g. one left from before a restart.
            warn!(prompt = %answer.prompt_id, "Stale Slack approval callback");
            return;
        }
        let Some(answered) = self
            .prompts
            .answer(&answer.prompt_id, &answer.choice_id)
            .await
        else {
            return;
        };
        if let Some((message, text)) = answered.update {
            let transport = self.transport.clone();
            if let Err(e) = transport.edit(&message, &text).await {
                warn!("Failed to update a Slack approval message: {}", e);
            }
            // Keep the outcome visible briefly, then remove the card.
            if let Some(delay) = Self::resolved_approval_dismiss_delay() {
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    let _ = transport.delete(&message).await;
                });
            }
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
            .dispatch_command(&cmd_text, &session_id, user_id, UserRole::Owner)
            .await;

        // Respond via response_url if available
        if let Some(response_url) = payload.get("response_url").and_then(|v| v.as_str()) {
            let mrkdwn = markdown_to_slack_mrkdwn(&reply);
            let body = serde_json::json!({
                "response_type": "ephemeral",
                "text": mrkdwn,
            });
            let _ = self.transport.respond(response_url, &body).await;
        } else {
            let _ = self.post_message(channel_id, &reply, None).await;
        }
    }

    /// Dispatch a command string and return the reply text.
    /// Accepts both `/command` and `!command` syntax (Slack reserves `/` for native slash commands).
    async fn dispatch_command(
        &self,
        text: &str,
        session_id: &str,
        user_id: &str,
        user_role: UserRole,
    ) -> String {
        // The same sender id the turn's channel context carries.
        let sender_id = format!("slack:{user_id}");
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
        if let Some(reply) = self
            .host
            .dispatch(cmd, arg, session_id, Some(&sender_id), user_role)
            .await
        {
            return reply;
        }

        // Channel-specific commands.
        match cmd {
            "/restart" if user_role != UserRole::Owner => {
                "Only the owner can restart the daemon.".to_string()
            }
            "/restart" => {
                self.host.restart();
                "Restart failed. You may need to restart manually.".to_string()
            }
            "/workspace" => {
                "Use `!workspace ...` as a message in the private Slack group where the grant should apply. Type `!help` for the exact syntax."
                    .to_string()
            }
            "/help" | "/start" => build_help_text(
                &slack_commands(),
                "!",
                aidaemon_channels::slash::HelpScope::from_args(arg),
            ),
            _ => aidaemon_channels::slash::unknown_command_reply(
                &format!("!{}", cmd.trim_start_matches('/')),
                &slack_commands(),
                "!",
            ),
        }
    }

    /// Dispatch a command and return the reply text plus optional Block Kit buttons.
    async fn dispatch_command_with_buttons(
        &self,
        text: &str,
        session_id: &str,
        user_id: &str,
        user_role: UserRole,
    ) -> (String, Vec<Value>) {
        let reply = self
            .dispatch_command(text, session_id, user_id, user_role)
            .await;

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
        Ok(self.transport.api("chat.postMessage", Some(&body)).await?)
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
        Ok(self.transport.api("chat.postMessage", Some(&body)).await?)
    }

    /// Handle incoming file attachments.
    async fn handle_incoming_files(
        &self,
        files: &[Value],
    ) -> anyhow::Result<Vec<MessageAttachment>> {
        let mut attachments = Vec::new();

        for sparse_file in files {
            let file = self.resolve_slack_file_metadata(sparse_file).await?;
            let filename = file.get("name").and_then(|v| v.as_str()).unwrap_or("file");
            let file_size = file.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
            let mime_type = file
                .get("mimetype")
                .and_then(|v| v.as_str())
                .unwrap_or("application/octet-stream");

            let download_url = file
                .get("url_private_download")
                .or_else(|| file.get("url_private"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Slack file {} has no private download URL after files.info",
                        filename
                    )
                })?;
            // Through the transport: the token only goes to Slack hosts, a
            // sign-in page (no `files:read`) is refused, and the size limit
            // holds while downloading.
            let bytes = aidaemon_channels::attachments::download_inbound(
                self.transport.as_ref(),
                &aidaemon_channels::Attachment::new(filename, download_url)
                    .with_mime_type(Some(mime_type.to_string()))
                    .with_size_bytes(Some(file_size).filter(|size| *size > 0)),
                self.deps.max_file_size_mb,
            )
            .await?;
            attachments.push(aidaemon_channels::attachments::save_inbound(
                &self.deps.inbox_dir,
                filename,
                mime_type.to_string(),
                &bytes,
            )?);
        }

        Ok(attachments)
    }

    /// Upload a file to a session's conversation through the transport
    /// (the upload URL must be a Slack host).
    async fn upload_file(
        &self,
        session_id: &str,
        filename: &str,
        data: Vec<u8>,
    ) -> anyhow::Result<()> {
        self.transport
            .send_file(
                &self.conversation_for(session_id),
                &aidaemon_channels::OutboundFile {
                    name: filename.to_string(),
                    bytes: data,
                    mime_type: None,
                    caption: None,
                },
            )
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

    /// The SDK conversation a session's messages go to.
    fn conversation_for(&self, session_id: &str) -> aidaemon_channels::Conversation {
        let (channel, thread) = self.parse_session_id(session_id);
        let kind = if channel.starts_with('D') {
            aidaemon_channels::ConversationKind::Direct
        } else if thread.is_some() {
            aidaemon_channels::ConversationKind::Thread
        } else {
            aidaemon_channels::ConversationKind::Group
        };
        aidaemon_channels::Conversation {
            id: match thread {
                Some(ts) => format!("{channel}:{ts}"),
                None => channel,
            },
            kind,
        }
    }

    /// A handle for Slack API calls from spawned tasks.
    fn api_handle(&self) -> Arc<crate::SlackChannel> {
        self.transport.clone()
    }
}

/// One Slack conversation (a channel, or a thread in one), as the turn
/// runner shows it.
struct SlackSurface {
    api: Arc<crate::SlackChannel>,
    channel: String,
    thread: Option<String>,
    /// The message the working reaction goes on; taken by the first turn.
    reacting_to: std::sync::Mutex<Option<String>>,
    user_cache: HashMap<String, String>,
    host: Arc<dyn SlackHost>,
}

#[async_trait]
impl TurnSurface for SlackSurface {
    fn platform(&self) -> &'static str {
        "slack"
    }

    async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
        let mrkdwn = markdown_to_slack_mrkdwn(text);
        let mrkdwn = restore_mentions_from_cache(&mrkdwn, &self.user_cache);
        let mentions = mrkdwn.matches("<@U").count();
        if mentions > 0 || text.contains('@') {
            info!(
                channel = %self.channel,
                mentions,
                known_users = self.user_cache.len(),
                "Slack reply mentions"
            );
        }
        let mut ids = Vec::new();
        let mut failure = None;
        for chunk in split_mrkdwn(&mrkdwn, MAX_MESSAGE_LEN) {
            let mut body = serde_json::json!({ "channel": self.channel, "text": chunk });
            if let Some(ts) = self.thread.as_deref() {
                body["thread_ts"] = Value::String(ts.to_string());
            }
            match self
                .api
                .api("chat.postMessage", Some(&body))
                .await
                .map_err(anyhow::Error::from)
            {
                Ok(response) => {
                    if let Some(ts) = response.get("ts").and_then(Value::as_str) {
                        ids.push(ts.to_string());
                    }
                }
                Err(error) => {
                    warn!(%error, "Failed to send Slack message");
                    failure = Some(error);
                }
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(ids),
        }
    }

    /// Slack bots have no typing indicator: an hourglass reaction on the
    /// message that started the work, swapped for a warning if the agent
    /// goes quiet. Queued turns have no message of their own to react to.
    fn show_working(
        &self,
        stop: tokio_util::sync::CancellationToken,
        heartbeat: Arc<AtomicU64>,
        stale_secs: u64,
    ) {
        let Some(ts) = self
            .reacting_to
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        else {
            return;
        };
        let api = self.api.clone();

        let channel = self.channel.clone();
        tokio::spawn(async move {
            const WORKING: &str = "hourglass_flowing_sand";
            add_reaction(&api, &channel, &ts, WORKING).await;
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(10)), if stale_secs > 0 => {
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        if now.saturating_sub(heartbeat.load(Ordering::Relaxed)) > stale_secs {
                            remove_reaction(&api, &channel, &ts, WORKING).await;
                            add_reaction(&api, &channel, &ts, "warning").await;
                            return;
                        }
                    }
                    _ = stop.cancelled() => break,
                }
            }
            remove_reaction(&api, &channel, &ts, WORKING).await;
        });
    }

    /// A queued guest message keeps a workspace grant only if the grant is
    /// still in force.
    async fn before_queued(&self, mut queued: QueuedMessage) -> QueuedMessage {
        if queued.user_role == UserRole::Guest
            && let Some(original) = queued.channel_ctx.workspace_grant.clone()
        {
            queued.channel_ctx.workspace_grant = match revalidate_queued_workspace_grant(
                self.host.as_ref(),
                &original,
            )
            .await
            {
                Ok(grant) => grant,
                Err(error) => {
                    warn!(%error, "Could not revalidate queued workspace grant; denying scoped access");
                    None
                }
            };
        }
        queued
    }
}

/// Best effort: a reaction that fails leaves the turn as it was.
async fn add_reaction(api: &crate::SlackChannel, channel: &str, ts: &str, name: &str) {
    let body = serde_json::json!({ "channel": channel, "timestamp": ts, "name": name });
    let _ = api.api("reactions.add", Some(&body)).await;
}

async fn remove_reaction(api: &crate::SlackChannel, channel: &str, ts: &str, name: &str) {
    let body = serde_json::json!({ "channel": channel, "timestamp": ts, "name": name });
    let _ = api.api("reactions.remove", Some(&body)).await;
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

#[async_trait]
impl aidaemon_channels::prompts::PromptSurface for SlackBot {
    fn prompts(&self) -> &aidaemon_channels::prompts::Prompts {
        &self.prompts
    }

    fn platform(&self) -> &'static str {
        "Slack"
    }

    async fn prompt_target(
        &self,
        session_id: &str,
    ) -> anyhow::Result<(
        Arc<dyn aidaemon_channels::Channel>,
        aidaemon_channels::Conversation,
    )> {
        Ok((self.transport.clone(), self.conversation_for(session_id)))
    }
}

#[async_trait]
impl HubChannel for SlackBot {
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
        for chunk in split_mrkdwn(&mrkdwn, MAX_MESSAGE_LEN) {
            self.post_message(&channel_id, &chunk, thread_ts.as_deref())
                .await?;
        }
        Ok(())
    }

    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        let (channel_id, thread_ts) = self.parse_session_id(session_id);
        match &media.kind {
            MediaKind::Photo { data } => {
                self.upload_file(session_id, "screenshot.png", data.clone())
                    .await?;
            }
            MediaKind::Document {
                file_path,
                filename,
            } => {
                let data = tokio::fs::read(file_path).await?;
                self.upload_file(session_id, filename, data).await?;
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
        one_time_only: bool,
    ) -> anyhow::Result<ApprovalResponse> {
        let ask = aidaemon_channels::approval_render::ApprovalAsk {
            command,
            card: None,
            risk_level,
            warnings,
            permission_mode,
            one_time_only,
        };
        self.ask_approval(session_id, &ask).await
    }

    async fn request_approval_ask(
        &self,
        session_id: &str,
        ask: &aidaemon_channels::approval_render::ApprovalAsk<'_>,
    ) -> anyhow::Result<ApprovalResponse> {
        self.ask_approval(session_id, ask).await
    }

    async fn request_goal_confirmation(
        &self,
        session_id: &str,
        goal_description: &str,
        details: &[String],
        style: GoalConfirmationStyle,
    ) -> anyhow::Result<bool> {
        self.confirm_goal(session_id, goal_description, details, style)
            .await
    }
}

impl SlackBot {
    /// Talk to the Web API at `base` with `http`, as bot user `bot_user_id`
    /// (a test points it at a mock).
    pub fn with_api(mut self, base: &str, bot_user_id: &str, http: reqwest::Client) -> Self {
        self.bot_user_id = Mutex::new(Some(bot_user_id.to_string()));
        self.transport = Arc::new(
            crate::SlackChannel::new("xoxb-synthetic", "xapp-synthetic")
                .with_api_base(base)
                .with_http_client(http),
        );
        self
    }
}

/// Spawn a Slack bot in a background task.
/// This is a separate function to avoid async type inference cycles.
pub fn spawn_slack_channel(channel: Arc<SlackBot>) {
    tokio::spawn(async move {
        channel.start_with_retry().await;
    });
}

#[cfg(test)]
mod file_event_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn accepts_file_share_messages_without_accepting_unrelated_subtypes() {
        assert_eq!(
            classify_slack_inbound_event(&json!({"type": "message"})),
            SlackInboundEventKind::Message
        );
        assert_eq!(
            classify_slack_inbound_event(&json!({
                "type": "message",
                "subtype": "file_share",
                "files": [{"id": "F_IMAGE_1"}]
            })),
            SlackInboundEventKind::Message
        );
        assert_eq!(
            classify_slack_inbound_event(&json!({
                "type": "app_mention",
                "subtype": "file_share"
            })),
            SlackInboundEventKind::Message
        );
        assert_eq!(
            classify_slack_inbound_event(&json!({
                "type": "message",
                "subtype": "message_changed"
            })),
            SlackInboundEventKind::Ignore
        );
        assert_eq!(
            classify_slack_inbound_event(&json!({
                "type": "message",
                "subtype": "bot_message"
            })),
            SlackInboundEventKind::Ignore
        );
    }

    #[test]
    fn routes_standalone_file_shared_events_for_hydration() {
        assert_eq!(
            classify_slack_inbound_event(&json!({
                "type": "file_shared",
                "file_id": "F_IMAGE_1",
                "channel_id": "C_PROJECT"
            })),
            SlackInboundEventKind::FileShared
        );
        assert_eq!(
            classify_slack_inbound_event(&json!({"type": "reaction_added"})),
            SlackInboundEventKind::Ignore
        );
    }

    #[test]
    fn locates_the_share_closest_to_the_file_event() {
        let file = json!({
            "shares": {
                "private": {
                    "C_PROJECT": [
                        {"ts": "100.000001"},
                        {"ts": "300.000001", "thread_ts": "250.000001"}
                    ]
                },
                "public": {
                    "C_OTHER": [{"ts": "400.000001"}]
                }
            }
        });

        assert_eq!(
            slack_file_share_location(&file, "C_PROJECT", Some("300.000002")),
            Some(SlackFileShareLocation {
                message_ts: "300.000001".to_string(),
                thread_ts: Some("250.000001".to_string()),
                is_private: true,
            })
        );
        assert_eq!(
            slack_file_share_location(&file, "C_PROJECT", None),
            Some(SlackFileShareLocation {
                message_ts: "300.000001".to_string(),
                thread_ts: Some("250.000001".to_string()),
                is_private: true,
            })
        );
        assert!(slack_file_share_location(&file, "C_MISSING", None).is_none());
    }

    #[test]
    fn recovered_file_message_keeps_text_and_every_reference_image() {
        let history_message = json!({
            "type": "message",
            "subtype": "file_share",
            "user": "U_OWNER",
            "text": "<@U_BOT> use both images as references",
            "ts": "300.000001",
            "files": [
                {"id": "F_IMAGE_1", "name": "reference-1.jpg"},
                {"id": "F_IMAGE_2", "name": "reference-2.jpg"}
            ]
        });
        let file_event = json!({
            "type": "file_shared",
            "file_id": "F_IMAGE_1",
            "user_id": "U_OWNER",
            "channel_id": "C_PROJECT"
        });
        let file = json!({"id": "F_IMAGE_1", "user": "U_OWNER"});

        let normalized = normalize_file_shared_message(
            history_message,
            &file_event,
            &file,
            "C_PROJECT",
            "group",
        )
        .unwrap();

        assert_eq!(normalized["type"], "message");
        assert_eq!(normalized["channel"], "C_PROJECT");
        assert_eq!(normalized["channel_type"], "group");
        assert_eq!(normalized["user"], "U_OWNER");
        assert_eq!(normalized["text"], "<@U_BOT> use both images as references");
        assert_eq!(normalized["files"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn repeated_per_file_callbacks_recover_one_stable_message_identity() {
        let history_message = json!({
            "text": "<@U_BOT> compare these",
            "ts": "500.000001",
            "files": [
                {"id": "F_IMAGE_1"},
                {"id": "F_IMAGE_2"}
            ]
        });

        let first = normalize_file_shared_message(
            history_message.clone(),
            &json!({"file_id": "F_IMAGE_1", "user_id": "U_OWNER"}),
            &json!({"id": "F_IMAGE_1"}),
            "C_PROJECT",
            "channel",
        )
        .unwrap();
        let second = normalize_file_shared_message(
            history_message,
            &json!({"file_id": "F_IMAGE_2", "user_id": "U_OWNER"}),
            &json!({"id": "F_IMAGE_2"}),
            "C_PROJECT",
            "channel",
        )
        .unwrap();

        assert_eq!(first["ts"], second["ts"]);
        assert_eq!(first["files"], second["files"]);
        assert_eq!(first["files"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn sparse_history_message_uses_file_event_fallbacks() {
        let normalized = normalize_file_shared_message(
            json!({"text": "", "ts": "600.000001"}),
            &json!({"user_id": "U_OWNER"}),
            &json!({"id": "F_IMAGE_1", "user": "U_FILE_OWNER"}),
            "D_OWNER",
            "im",
        )
        .unwrap();

        assert_eq!(normalized["user"], "U_OWNER");
        assert_eq!(normalized["files"][0]["id"], "F_IMAGE_1");
    }

    #[test]
    fn resolves_channel_types_without_weakening_private_fallbacks() {
        assert_eq!(slack_channel_type(&json!({"is_im": true})), "im");
        assert_eq!(slack_channel_type(&json!({"is_mpim": true})), "mpim");
        assert_eq!(slack_channel_type(&json!({"is_private": true})), "group");
        assert_eq!(slack_channel_type(&json!({"is_channel": true})), "channel");
        assert_eq!(fallback_slack_channel_type("D_OWNER", true), "im");
        assert_eq!(fallback_slack_channel_type("C_PRIVATE", true), "group");
        assert_eq!(fallback_slack_channel_type("C_PUBLIC", false), "channel");
    }
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
            SlackBot::parse_slack_user_mention("<@U123ABC>").unwrap(),
            "U123ABC"
        );
        assert!(SlackBot::parse_slack_user_mention("@display-name").is_err());
        assert!(SlackBot::parse_slack_user_mention("<@../../etc>").is_err());
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
