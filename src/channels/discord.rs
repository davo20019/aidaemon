//! The Discord channel. Its logic (access, auto-claim, sessions, queueing,
//! cancellation, commands, files, approvals) is transport-neutral: replies
//! go through an [`Outlet`] and inbound traffic is normalized first, so the
//! same code runs on either transport, chosen by `[channels] discord`:
//! - `sdk` (default): the `aidaemon-channel-discord` adapter;
//! - `legacy`: serenity, kept until the SDK transport has been checked
//!   live, then deleted with this module's serenity glue.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aidaemon_channel_discord::DiscordChannel as SdkDiscord;
use aidaemon_channels::Channel as _;
use async_trait::async_trait;
use serenity::all::{
    ChannelId, Command, CommandInteraction, ComponentInteraction, Context, CreateAttachment,
    CreateButton, CreateCommand, CreateInteractionResponse, CreateInteractionResponseMessage,
    CreateMessage, EditInteractionResponse, EventHandler, GatewayIntents, GuildId, Interaction,
    Message as SerenityMessage, Ready,
};
use serenity::builder::CreateActionRow;
use serenity::model::application::ButtonStyle;
use serenity::Client;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use super::approval_render;
use super::commands::{shared_commands, CommandCategory, CommandDef};
use super::formatting::{build_help_text, sanitize_filename, split_message};
use crate::channels::{ChannelHub, ChannelRuntimeDeps, SessionMap};
use crate::config::ChannelTransport;
use crate::runtime_ports::{ChannelAgentRuntime, InboundMessageRequest};
use crate::tasks::{QueueOutcome, QueuedMessage, TaskRegistry};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::traits::{Channel, ChannelCapabilities, StateStore};
use crate::types::{ApprovalResponse, GoalConfirmationStyle, MediaKind, MediaMessage};
use crate::types::{ChannelContext, ChannelVisibility, StatusUpdate, UserRole};

/// Longest Discord message.
const MAX_MESSAGE_LEN: usize = 2000;

/// Commands available in the Discord channel (shared Core only).
fn discord_commands() -> Vec<CommandDef> {
    let mut cmds: Vec<CommandDef> = shared_commands()
        .into_iter()
        .filter(|c| matches!(c.category, CommandCategory::Core))
        .collect();
    cmds.push(CommandDef {
        name: "help",
        description: "Show available commands",
        usage: None,
        category: CommandCategory::Core,
    });
    cmds
}

/// One slash command: name, description and its optional string argument
/// (name, description, required).
struct SlashCommand {
    name: &'static str,
    description: &'static str,
    argument: Option<(&'static str, &'static str, bool)>,
}

/// The slash commands the bot registers.
const SLASH_COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "model",
        description: "Show or switch the current LLM model",
        argument: Some(("name", "Model name to switch to", false)),
    },
    SlashCommand {
        name: "models",
        description: "List available models from provider",
        argument: None,
    },
    SlashCommand {
        name: "auto",
        description: "Re-enable automatic model routing",
        argument: None,
    },
    SlashCommand {
        name: "reload",
        description: "Reload config.toml",
        argument: None,
    },
    SlashCommand {
        name: "tasks",
        description: "List running and recent tasks",
        argument: None,
    },
    SlashCommand {
        name: "cancel",
        description: "Cancel a running task",
        argument: Some(("id", "Task ID to cancel", true)),
    },
    SlashCommand {
        name: "clear",
        description: "Start fresh conversation (history kept for memory/audit)",
        argument: None,
    },
    SlashCommand {
        name: "wipe",
        description: "Permanently delete this conversation",
        argument: None,
    },
    SlashCommand {
        name: "cost",
        description: "Show token usage statistics",
        argument: None,
    },
    SlashCommand {
        name: "context",
        description: "Show conversation context coverage",
        argument: None,
    },
    SlashCommand {
        name: "help",
        description: "Show available commands",
        argument: None,
    },
];

/// Where a reply goes: one conversation, on the transport that carries it.
#[derive(Clone)]
enum Outlet {
    Serenity {
        http: Arc<serenity::http::Http>,
        channel: ChannelId,
    },
    Sdk {
        transport: Arc<SdkDiscord>,
        conversation: aidaemon_channels::Conversation,
    },
}

impl Outlet {
    /// Send one message (at most [`MAX_MESSAGE_LEN`]); returns its id.
    async fn say(&self, text: &str) -> anyhow::Result<String> {
        match self {
            Self::Serenity { http, channel } => Ok(channel.say(http, text).await?.id.to_string()),
            Self::Sdk {
                transport,
                conversation,
            } => Ok(transport
                .send(
                    conversation,
                    &aidaemon_channels::OutboundMessage::text(text),
                )
                .await?
                .message_id),
        }
    }

    /// Send `text`, split to fit; returns the ids of the parts sent and
    /// whether any part failed.
    async fn say_all(&self, text: &str) -> (Vec<String>, bool) {
        let mut ids = Vec::new();
        let mut failed = false;
        for chunk in split_message(text, MAX_MESSAGE_LEN) {
            match self.say(&chunk).await {
                Ok(id) => ids.push(id),
                Err(e) => {
                    failed = true;
                    warn!("Failed to send Discord message: {}", e);
                }
            }
        }
        (ids, failed)
    }

    async fn typing(&self) {
        match self {
            Self::Serenity { http, channel } => {
                let _ = channel.broadcast_typing(http).await;
            }
            Self::Sdk {
                transport,
                conversation,
            } => {
                let _ = transport.typing(conversation).await;
            }
        }
    }
}

/// Where an inbound file's bytes come from.
enum FileSource {
    Serenity(serenity::all::Attachment),
    Sdk(Arc<SdkDiscord>, aidaemon_channels::Attachment),
}

struct InboundFile {
    name: String,
    mime: Option<String>,
    size: u64,
    source: FileSource,
}

impl InboundFile {
    async fn download(&self) -> anyhow::Result<Vec<u8>> {
        match &self.source {
            FileSource::Serenity(attachment) => Ok(attachment.download().await?),
            FileSource::Sdk(transport, attachment) => Ok(transport.download(attachment).await?),
        }
    }
}

/// An inbound message, whichever transport delivered it.
struct Inbound {
    user_id: u64,
    user_name: String,
    /// Sent in a server channel (not a DM).
    guild: bool,
    /// The Discord channel id it was sent in.
    channel_id: String,
    message_id: String,
    content: String,
    files: Vec<InboundFile>,
}

/// What pressing an approval or confirmation button decided.
enum ButtonOutcome {
    /// Not a button this bot made.
    Ignored,
    /// The presser may not answer it.
    Unauthorized,
    /// Answered; the prompt's new text (its buttons removed).
    Resolved(String),
}

/// Discord channel implementation.
pub struct DiscordChannel {
    /// Bot username fetched from Discord API (e.g., "my_bot").
    /// Populated once the gateway is ready.
    bot_username: std::sync::RwLock<String>,
    bot_token: String,
    allowed_user_ids: std::sync::RwLock<Vec<u64>>,
    /// Discord user IDs recognized as owners (from `users.owner_ids.discord`).
    owner_user_ids: Vec<u64>,
    guild_id: Option<u64>,
    agent: Arc<dyn ChannelAgentRuntime>,
    config_path: PathBuf,
    pending_approvals: Mutex<HashMap<String, tokio::sync::oneshot::Sender<ApprovalResponse>>>,
    /// SDK transport: each open prompt's message and text, by approval id,
    /// so an answer can replace it with the outcome.
    sdk_prompts: Mutex<HashMap<String, (aidaemon_channels::MessageRef, String)>>,
    session_map: SessionMap,
    task_registry: Arc<TaskRegistry>,
    files_enabled: bool,
    inbox_dir: PathBuf,
    max_file_size_mb: u64,
    state: Arc<dyn StateStore>,
    transport: ChannelTransport,
    /// Legacy transport: stored after the client starts.
    http: Mutex<Option<Arc<serenity::http::Http>>>,
    /// SDK transport: set when it starts.
    sdk: std::sync::RwLock<Option<Arc<SdkDiscord>>>,
    /// SDK transport: direct conversations opened, by user id.
    dm_conversations: Mutex<HashMap<u64, aidaemon_channels::Conversation>>,
    /// Reference to the channel hub for dynamic bot registration.
    channel_hub: std::sync::RwLock<Option<Weak<ChannelHub>>>,
    /// Seconds of no heartbeat before declaring the agent stuck (0 = disabled).
    watchdog_stale_threshold_secs: u64,
}

impl DiscordChannel {
    pub fn new(
        bot_token: &str,
        allowed_user_ids: Vec<u64>,
        owner_user_ids: Vec<u64>,
        guild_id: Option<u64>,
        runtime: ChannelRuntimeDeps,
    ) -> Self {
        Self {
            bot_username: std::sync::RwLock::new("discord".to_string()),
            bot_token: bot_token.to_string(),
            allowed_user_ids: std::sync::RwLock::new(allowed_user_ids),
            owner_user_ids,
            guild_id,
            agent: runtime.agent,
            config_path: runtime.config_path,
            pending_approvals: Mutex::new(HashMap::new()),
            sdk_prompts: Mutex::new(HashMap::new()),
            session_map: runtime.session_map,
            task_registry: runtime.task_registry,
            files_enabled: runtime.files_enabled,
            inbox_dir: runtime.inbox_dir,
            max_file_size_mb: runtime.max_file_size_mb,
            state: runtime.state,
            transport: runtime.transports.discord,
            http: Mutex::new(None),
            sdk: std::sync::RwLock::new(None),
            dm_conversations: Mutex::new(HashMap::new()),
            channel_hub: std::sync::RwLock::new(None),
            watchdog_stale_threshold_secs: runtime.watchdog_stale_threshold_secs,
        }
    }

    /// Set the channel hub reference for dynamic bot registration.
    pub fn set_channel_hub(&self, hub: Weak<ChannelHub>) {
        if let Ok(mut guard) = self.channel_hub.write() {
            *guard = Some(hub);
        }
    }

    /// Get the bot's username (cached after the gateway is ready).
    fn get_bot_username(&self) -> String {
        self.bot_username
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Set the bot's username (once the gateway reports it).
    fn set_bot_username(&self, username: String) {
        if let Ok(mut guard) = self.bot_username.write() {
            *guard = username;
        }
    }

    /// Build the session ID for a channel/user, prefixing with bot name if not "discord".
    fn session_id(&self, base_id: &str) -> String {
        let username = self.get_bot_username();
        if username == "discord" {
            base_id.to_string()
        } else {
            format!("{}:{}", username, base_id)
        }
    }

    /// Get the channel identifier for the session map.
    fn channel_name(&self) -> String {
        let username = self.get_bot_username();
        if username == "discord" {
            "discord".to_string()
        } else {
            format!("discord:{}", username)
        }
    }

    /// The session id of a DM with `user_id` or a server channel.
    fn session_for(&self, user_id: u64, guild: bool, channel_id: &str) -> String {
        let base = if guild {
            format!("discord:ch:{channel_id}")
        } else {
            format!("discord:dm:{user_id}")
        };
        self.session_id(&base)
    }

    /// Start the Discord client with automatic retry on crash.
    /// Uses exponential backoff: 5s → 10s → 20s → 40s → 60s cap.
    pub async fn start_with_retry(self: Arc<Self>) {
        let mut backoff = crate::backoff::ChannelReconnectBackoff::new(
            Duration::from_secs(5),
            Duration::from_secs(60),
            Duration::from_secs(60),
        );

        loop {
            info!(transport = ?self.transport, "Starting Discord client");
            let started = tokio::time::Instant::now();
            match self.transport {
                ChannelTransport::Sdk => self.clone().start_sdk().await,
                ChannelTransport::Legacy => self.clone().start_legacy().await,
            }
            let ran_for = started.elapsed();

            let delay = backoff.next_jittered_delay(ran_for, 0.10);

            warn!(
                backoff_secs = delay.as_secs(),
                ran_for_secs = ran_for.as_secs(),
                "Discord client stopped, restarting"
            );
            tokio::time::sleep(delay).await;
        }
    }

    async fn start_legacy(self: Arc<Self>) {
        let intents = GatewayIntents::GUILD_MESSAGES
            | GatewayIntents::DIRECT_MESSAGES
            | GatewayIntents::MESSAGE_CONTENT;

        let handler = DiscordHandler {
            channel: Arc::clone(&self),
        };

        let mut client = match Client::builder(&self.bot_token, intents)
            .event_handler(handler)
            .await
        {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to create Discord client: {}", e);
                return;
            }
        };

        // Store the HTTP client so Channel trait methods can send messages.
        {
            let mut http = self.http.lock().await;
            *http = Some(client.http.clone());
        }

        if let Err(e) = client.start().await {
            warn!("Discord client error: {}", e);
        }
    }

    /// Run on the SDK adapter until it stops, handling each event on its
    /// own task as the legacy client does.
    async fn start_sdk(self: Arc<Self>) {
        let mut transport = SdkDiscord::new(self.bot_token.clone());
        if let Some(guild) = self.guild_id {
            transport = transport.with_command_guild(guild.to_string());
        }
        let transport = Arc::new(transport);
        if let Ok(mut slot) = self.sdk.write() {
            *slot = Some(transport.clone());
        }
        let commands: Vec<aidaemon_channels::CommandSpec> = SLASH_COMMANDS
            .iter()
            .map(|command| aidaemon_channels::CommandSpec {
                name: command.name.to_string(),
                description: command.description.to_string(),
            })
            .collect();
        if let Err(e) = transport.register_commands(&commands).await {
            warn!("Failed to register Discord slash commands: {}", e);
        }

        let (events_tx, mut events) = tokio::sync::mpsc::channel(64);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let runner = {
            let transport = transport.clone();
            let shutdown = shutdown.clone();
            tokio::spawn(async move { transport.run(events_tx, shutdown).await })
        };
        while let Some(event) = events.recv().await {
            if let Some(username) = transport.bot_username() {
                if username != self.get_bot_username() {
                    info!(username = %username, "Discord bot connected");
                    self.set_bot_username(username);
                }
            }
            let this = self.clone();
            let transport = transport.clone();
            tokio::spawn(async move {
                this.handle_sdk_event(transport, event).await;
            });
        }
        shutdown.cancel();
        match runner.await {
            Ok(Err(e)) => warn!("Discord client error: {}", e),
            Err(e) => warn!("Discord client task failed: {}", e),
            Ok(Ok(())) => {}
        }
    }

    async fn handle_sdk_event(
        self: &Arc<Self>,
        transport: Arc<SdkDiscord>,
        event: aidaemon_channels::ChannelEvent,
    ) {
        match event {
            aidaemon_channels::ChannelEvent::Message(message) => {
                let Ok(user_id) = message.sender.id.parse::<u64>() else {
                    return;
                };
                let outlet = Outlet::Sdk {
                    transport: transport.clone(),
                    conversation: message.conversation.clone(),
                };
                let inbound = Inbound {
                    user_id,
                    user_name: message.sender.display_name.clone().unwrap_or_default(),
                    guild: message.conversation.kind != aidaemon_channels::ConversationKind::Direct,
                    channel_id: message.conversation.id.clone(),
                    message_id: message.message_id.clone(),
                    content: message.text.clone(),
                    files: message
                        .attachments
                        .into_iter()
                        .map(|attachment| InboundFile {
                            name: attachment.name.clone(),
                            mime: attachment.mime_type.clone(),
                            size: attachment.size_bytes.unwrap_or(0),
                            source: FileSource::Sdk(transport.clone(), attachment),
                        })
                        .collect(),
                };
                self.handle_inbound(inbound, outlet).await;
            }
            aidaemon_channels::ChannelEvent::Command(command) => {
                let Ok(user_id) = command.sender.id.parse::<u64>() else {
                    return;
                };
                let guild =
                    command.conversation.kind != aidaemon_channels::ConversationKind::Direct;
                let reply = self
                    .slash_command_reply(
                        user_id,
                        guild,
                        &command.conversation.id,
                        &command.name,
                        &command.args,
                    )
                    .await;
                if let Err(e) = transport
                    .answer_command(&command, &truncate_chars(&reply, MAX_MESSAGE_LEN))
                    .await
                {
                    warn!("Failed to answer a Discord slash command: {}", e);
                }
            }
            aidaemon_channels::ChannelEvent::Approval(answer) => {
                let Ok(user_id) = answer.sender.id.parse::<u64>() else {
                    return;
                };
                let Some((prefix, action)) = answer.choice_id.split_once(':') else {
                    return;
                };
                let prompt = self
                    .sdk_prompts
                    .lock()
                    .await
                    .get(&answer.prompt_id)
                    .cloned();
                let Some((message, original)) = prompt else {
                    warn!(
                        approval_id = %answer.prompt_id,
                        "Stale approval callback (no pending request)"
                    );
                    return;
                };
                match self
                    .resolve_button(user_id, prefix, action, &answer.prompt_id, &original)
                    .await
                {
                    ButtonOutcome::Resolved(updated) => {
                        self.sdk_prompts.lock().await.remove(&answer.prompt_id);
                        if let Err(e) = transport
                            .edit(&message, &truncate_chars(&updated, MAX_MESSAGE_LEN))
                            .await
                        {
                            warn!("Failed to update a Discord approval message: {}", e);
                        }
                    }
                    ButtonOutcome::Unauthorized => {
                        warn!(user_id, "Unauthorized Discord approval button press");
                    }
                    ButtonOutcome::Ignored => {}
                }
            }
        }
    }

    /// Where messages for `session_id` go.
    async fn outlet_for(&self, session_id: &str) -> anyhow::Result<Outlet> {
        let base_session_id = base_session_id(session_id);
        if let Some(transport) = self.sdk_transport() {
            let conversation = if let Some(user) = base_session_id.strip_prefix("discord:dm:") {
                let user_id: u64 = user.parse()?;
                let cached = self.dm_conversations.lock().await.get(&user_id).cloned();
                match cached {
                    Some(conversation) => conversation,
                    None => {
                        let conversation = transport.open_direct(user).await?;
                        self.dm_conversations
                            .lock()
                            .await
                            .insert(user_id, conversation.clone());
                        conversation
                    }
                }
            } else if let Some(channel) = base_session_id.strip_prefix("discord:ch:") {
                aidaemon_channels::Conversation {
                    id: channel.to_string(),
                    kind: aidaemon_channels::ConversationKind::Group,
                }
            } else {
                anyhow::bail!("Invalid Discord session ID: {}", session_id)
            };
            return Ok(Outlet::Sdk {
                transport,
                conversation,
            });
        }

        let http = self.get_http().await?;
        let channel = if let Some(user_id_str) = base_session_id.strip_prefix("discord:dm:") {
            let user_id: u64 = user_id_str.parse()?;
            let user = serenity::model::id::UserId::new(user_id);
            user.create_dm_channel(&http).await?.id
        } else if let Some(channel_id_str) = base_session_id.strip_prefix("discord:ch:") {
            ChannelId::new(channel_id_str.parse()?)
        } else {
            anyhow::bail!("Invalid Discord session ID: {}", session_id)
        };
        Ok(Outlet::Serenity { http, channel })
    }

    async fn get_http(&self) -> anyhow::Result<Arc<serenity::http::Http>> {
        let guard = self.http.lock().await;
        guard
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Discord HTTP client not ready"))
    }

    fn sdk_transport(&self) -> Option<Arc<SdkDiscord>> {
        self.sdk.read().ok().and_then(|slot| slot.clone())
    }

    fn is_authorized(&self, user_id: u64) -> bool {
        let ids = self
            .allowed_user_ids
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if ids.is_empty() {
            // No allowed users configured — will be handled by auto-claim in handle_inbound.
            return false;
        }
        ids.contains(&user_id)
    }

    /// Auto-claim: when allowed_user_ids is empty OR contains only stale IDs from
    /// another platform (e.g. Telegram IDs saved for a Discord bot via /connect),
    /// register the first DM user as the allowed user and persist to DB.
    ///
    /// Only applies to dynamic bots (owner_user_ids is empty). Config-based bots
    /// with explicit owner_user_ids are never auto-claimable.
    async fn try_auto_claim(&self, user_id: u64, is_dm: bool) -> bool {
        if !is_dm {
            return false;
        }
        // Config-based bots have owner_user_ids set — never auto-claim those.
        if !self.owner_user_ids.is_empty() {
            return false;
        }

        // Claim: replace any stale IDs with this user.
        {
            let mut ids = self
                .allowed_user_ids
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if ids.contains(&user_id) {
                return true; // Already authorized.
            }
            info!(
                user_id,
                old_ids = ?*ids,
                "Auto-claiming Discord bot for DM user (replacing stale IDs)"
            );
            ids.clear();
            ids.push(user_id);
        }

        // Persist to DB so it survives restarts
        if let Err(e) = self
            .state
            .update_dynamic_bot_allowed_users(&self.bot_token, &[user_id.to_string()])
            .await
        {
            warn!("Failed to persist auto-claimed user to DB: {}", e);
        }

        true
    }

    fn spawn_typing_indicator(
        outlet: Outlet,
        typing_cancel: tokio_util::sync::CancellationToken,
        heartbeat: Arc<AtomicU64>,
        stale_threshold_secs: u64,
    ) {
        let typing_token = typing_cancel.clone();
        tokio::spawn(async move {
            loop {
                outlet.typing().await;
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(8)) => {
                        if stale_threshold_secs > 0 {
                            let last_hb = heartbeat.load(Ordering::Relaxed);
                            let now = SystemTime::now().duration_since(UNIX_EPOCH)
                                .unwrap_or_default().as_secs();
                            if now.saturating_sub(last_hb) > stale_threshold_secs {
                                break;
                            }
                        }
                    }
                    _ = typing_token.cancelled() => break,
                }
            }
        });
    }

    fn spawn_status_task(
        mut status_rx: tokio::sync::mpsc::Receiver<StatusUpdate>,
        outlet: Outlet,
        is_dm: bool,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut last_sent = tokio::time::Instant::now() - Duration::from_secs(10);
            let min_interval = Duration::from_secs(3);
            while let Some(update) = status_rx.recv().await {
                // In non-DM channels: suppress all status messages.
                // Discord's native "typing..." indicator is sufficient.
                // (except BudgetExtended which must always reach the user)
                if !is_dm && !matches!(&update, StatusUpdate::BudgetExtended { .. }) {
                    continue;
                }
                let now = tokio::time::Instant::now();
                // Explicit user actions and BudgetExtended notifications bypass
                // rate limiting. URLs in task descriptions do not change the
                // delivery semantics of ordinary progress.
                let is_immediate = update.requires_immediate_delivery();
                if !is_immediate && now.duration_since(last_sent) < min_interval {
                    continue;
                }
                let text = status_text(&update);
                let _ = outlet.say(&truncate_chars(&text, MAX_MESSAGE_LEN)).await;
                last_sent = tokio::time::Instant::now();
            }
        })
    }

    /// Handle an incoming Discord message from the legacy client.
    async fn handle_message_event(self: &Arc<Self>, ctx: &Context, msg: SerenityMessage) {
        // Ignore bot messages
        if msg.author.bot {
            return;
        }
        let outlet = Outlet::Serenity {
            http: ctx.http.clone(),
            channel: msg.channel_id,
        };
        let inbound = Inbound {
            user_id: msg.author.id.get(),
            user_name: msg.author.name.clone(),
            guild: msg.guild_id.is_some(),
            channel_id: msg.channel_id.to_string(),
            message_id: msg.id.to_string(),
            content: msg.content.clone(),
            files: msg
                .attachments
                .iter()
                .map(|attachment| InboundFile {
                    name: attachment.filename.clone(),
                    mime: attachment.content_type.clone(),
                    size: attachment.size as u64,
                    source: FileSource::Serenity(attachment.clone()),
                })
                .collect(),
        };
        self.handle_inbound(inbound, outlet).await;
    }

    /// Handle an incoming message, whichever transport delivered it.
    async fn handle_inbound(self: &Arc<Self>, msg: Inbound, outlet: Outlet) {
        let user_id = msg.user_id;
        let is_dm = !msg.guild;
        // In guild channels, let all server members through — they get Guest role
        // from determine_role(). Auth gate only applies to DMs.
        if is_dm && !self.is_authorized(user_id) && !self.try_auto_claim(user_id, is_dm).await {
            warn!(user_id, "Unauthorized Discord user attempted access");
            let _ = outlet.say("Unauthorized.").await;
            return;
        }

        let (body_text, inbound_attachments) = if msg.content.is_empty() && !msg.files.is_empty() {
            if self.files_enabled {
                match self.handle_file_message(&msg.files).await {
                    Ok(attachments) => (String::new(), attachments),
                    Err(e) => {
                        let _ = outlet.say(&format!("File error: {}", e)).await;
                        return;
                    }
                }
            } else {
                let _ = outlet.say("I can only process text messages.").await;
                return;
            }
        } else if !msg.content.is_empty() {
            let mut attachments = Vec::new();
            if self.files_enabled && !msg.files.is_empty() {
                match self.handle_file_message(&msg.files).await {
                    Ok(file_attachments) => attachments = file_attachments,
                    Err(e) => {
                        let _ = outlet.say(&format!("File error: {}", e)).await;
                        return;
                    }
                }
            }
            (msg.content.clone(), attachments)
        } else {
            return;
        };

        let user_role = super::telegram::determine_role(&self.owner_user_ids, user_id);
        let text = if user_role == UserRole::Owner {
            super::attachments::build_inbound_text(&body_text, &inbound_attachments)
        } else {
            super::attachments::build_inbound_text_without_local_paths(
                &body_text,
                &inbound_attachments,
            )
        };

        let session_id = self.session_for(user_id, msg.guild, &msg.channel_id);

        // Handle slash-style commands (text commands starting with /)
        if super::commands::is_slash_command(&text) {
            let reply = self.dispatch_command(&text, &session_id, user_role).await;
            outlet.say_all(&reply).await;
            return;
        }

        // Capture sender/channel context before any queue decision. A session
        // worker may drain messages from several group members, so each queued
        // item must carry its own authorization identity.
        let channel_ctx = ChannelContext {
            visibility: if msg.guild {
                ChannelVisibility::Public
            } else {
                ChannelVisibility::Private
            },
            platform: "discord".to_string(),
            channel_name: None,
            channel_id: Some(if msg.guild {
                format!("discord:{}", msg.channel_id)
            } else {
                format!("discord:dm:{}", user_id)
            }),
            workspace_id: None,
            sender_name: Some(msg.user_name.clone()),
            sender_id: Some(format!("discord:{}", user_id)),
            channel_member_names: vec![],
            user_id_map: std::collections::HashMap::new(),
            workspace_grant: None,
            trusted: false,
        };

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

        info!(session_id, "Received Discord message from user {}", user_id);

        // Handle cancel/stop commands - these bypass the queue.
        let text_lower = text.to_ascii_lowercase();
        if text_lower == "cancel" || text_lower == "stop" || text_lower == "abort" {
            if user_role != UserRole::Owner {
                let _ = outlet
                    .say("Only the owner can cancel running work in this session.")
                    .await;
                return;
            }

            let cancelled = self
                .task_registry
                .cancel_running_for_session(&session_id)
                .await;
            let queued_work_preserved = self.task_registry.queue_len(&session_id).await;
            let cancelled_goals = self
                .agent
                .cancel_active_goals_for_session(&session_id)
                .await;

            let response = if cancelled.is_empty() {
                if cancelled_goals.is_empty() {
                    "No running task to cancel.".to_string()
                } else if cancelled_goals.len() == 1 {
                    format!("⏹️ Cancelled goal: {}", cancelled_goals[0])
                } else {
                    format!(
                        "⏹️ Cancelled {} goals:\n{}",
                        cancelled_goals.len(),
                        cancelled_goals
                            .iter()
                            .map(|d| format!("- {}", d))
                            .collect::<Vec<_>>()
                            .join("\n")
                    )
                }
            } else {
                let desc = cancelled
                    .first()
                    .map(|(_, d)| d.as_str())
                    .unwrap_or("unknown");
                let mut response = format!("⏹️ Cancelled: {}", desc);
                if queued_work_preserved > 0 {
                    response.push_str(&format!(
                        " ({} queued request(s) will continue in order)",
                        queued_work_preserved
                    ));
                }
                if !cancelled_goals.is_empty() {
                    response.push_str(&format!(" (+{} goal(s) cancelled)", cancelled_goals.len()));
                }
                response
            };
            outlet.say_all(&response).await;
            return;
        }

        // Stable per-message identity for dedup. Discord message ids are unique
        // per message and reused on gateway redeliveries, but a user re-typing
        // the same text gets a new id — so this drops genuine duplicates while
        // letting a deliberate re-ask through.
        let dedup_key = msg.message_id.clone();

        // Check if a task is already running for this session - if so, queue this message.
        if self.task_registry.has_running_task(&session_id).await {
            // Atomic check-and-queue: if the task finished between the
            // has_running_task() check above and this call, fall through to
            // direct processing instead of stranding the message.
            match self
                .task_registry
                .queue_message_if_running(
                    &session_id,
                    QueuedMessage::new(
                        &text,
                        user_role,
                        channel_ctx.clone(),
                        inbound_attachments.clone(),
                    ),
                    Some(&dedup_key),
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
                        let notice = crate::channels::presentation::queued_message_notice(
                            &text,
                            &current_task,
                            !inbound_attachments.is_empty(),
                        );
                        outlet.say_all(&notice).await;
                    }
                    return;
                }
                QueueOutcome::Duplicate => {
                    info!(session_id, "Dropped duplicate queued message");
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
            .mark_seen(&session_id, &text, Some(&dedup_key))
            .await
        {
            debug!(
                session_id,
                "Dropped duplicate message (direct processing race)"
            );
            return;
        }

        // Register task for tracking
        let description: String = text.chars().take(80).collect();
        let (task_id, cancel_token) = self.task_registry.register(&session_id, &description).await;

        // Create heartbeat for watchdog — agent bumps this on every activity point.
        let heartbeat = Arc::new(AtomicU64::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        ));
        let stale_threshold_secs = self.watchdog_stale_threshold_secs;

        // Start typing indicator for this task.
        let typing_cancel = tokio_util::sync::CancellationToken::new();
        Self::spawn_typing_indicator(
            outlet.clone(),
            typing_cancel.clone(),
            heartbeat.clone(),
            stale_threshold_secs,
        );

        // Status updates for this task.
        let (status_tx, status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(16);
        let status_task = Self::spawn_status_task(status_rx, outlet.clone(), is_dm);

        self.task_registry
            .set_typing_cancel(task_id, typing_cancel.clone())
            .await;

        let registry = Arc::clone(&self.task_registry);

        let agent = Arc::clone(&self.agent);
        tokio::spawn(async move {
            let mut current_text = text;
            let mut current_attachments = inbound_attachments;
            let mut current_user_role = user_role;
            let mut current_channel_ctx = channel_ctx;
            let mut current_task_id = task_id;
            let mut current_cancel_token = cancel_token;
            let mut current_status_tx = status_tx;
            let mut current_status_task = status_task;
            let mut current_heartbeat = heartbeat;
            let mut current_typing_cancel = typing_cancel;
            let task_wall_deadline = tokio::time::Instant::now() + Duration::from_secs(20 * 60);

            loop {
                let result = tokio::select! {
                    r = agent.handle_inbound_message(InboundMessageRequest {
                        session_id: session_id.clone(),
                        user_text: current_text.clone(),
                        attachments: current_attachments.clone(),
                        status_tx: Some(current_status_tx.clone()),
                        user_role: current_user_role,
                        channel_ctx: current_channel_ctx.clone(),
                        heartbeat: Some(current_heartbeat.clone()),
                        ingress_timing: None,
                    }) => r,
                    _ = current_cancel_token.cancelled() => Err(anyhow::anyhow!("Task cancelled")),
                    stale_mins = super::wait_for_stale_heartbeat(current_heartbeat.clone(), stale_threshold_secs, 8), if stale_threshold_secs > 0 => {
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
                current_status_task.abort();

                let mut task_error: Option<String> = None;
                match result {
                    Ok(envelope) => {
                        let _ = agent
                            .record_response_delivery(
                                &session_id,
                                envelope.delivery(
                                    "discord",
                                    crate::events::ResponseDeliveryState::Queued,
                                    Vec::new(),
                                    None,
                                ),
                            )
                            .await;
                        let reply = crate::channels::prepare_chat_message(&envelope.text);
                        let (platform_message_ids, failed) = outlet.say_all(&reply).await;
                        let _ = agent
                            .record_response_delivery(
                                &session_id,
                                envelope.delivery(
                                    "discord",
                                    if failed {
                                        crate::events::ResponseDeliveryState::Failed
                                    } else {
                                        crate::events::ResponseDeliveryState::PlatformAcknowledged
                                    },
                                    platform_message_ids,
                                    failed.then(|| "discord_send_failed".to_string()),
                                ),
                            )
                            .await;
                    }
                    Err(e) => {
                        let error_msg = e.to_string();
                        if error_msg == "Task cancelled" {
                            task_error = Some(error_msg);
                            info!("Task #{} cancelled", current_task_id);
                        } else {
                            task_error = Some(error_msg.clone());
                            if error_msg.starts_with("Task auto-cancelled due to inactivity") {
                                info!("Task #{} auto-cancelled by stale watchdog", current_task_id);
                                outlet.say_all(&format!("⚠️ {}", error_msg)).await;
                            } else {
                                warn!("Agent error: {}", e);
                                outlet.say_all(&format!("Error: {}", e)).await;
                            }
                        }
                    }
                }

                if let Some(error) = task_error.as_ref() {
                    let status = if error == "Task cancelled" {
                        crate::events::TaskStatus::Cancelled
                    } else {
                        crate::events::TaskStatus::Failed
                    };
                    if let Err(terminalize_error) = agent
                        .terminalize_supervised_task(&session_id, status, error.clone())
                        .await
                    {
                        warn!(%terminalize_error, "Failed to persist supervised task termination");
                    }
                }

                // Finalize and drain atomically so no concurrent message can
                // be stranded between task completion and the queue check.
                if let Some(queued) = registry
                    .finalize_and_drain(current_task_id, &session_id, task_error.as_deref())
                    .await
                {
                    tokio::time::sleep(Duration::from_millis(100)).await;

                    info!(
                        session_id,
                        "Processing queued message: {}",
                        queued.text.chars().take(50).collect::<String>()
                    );
                    let preview: String = queued.text.chars().take(50).collect();
                    let suffix = if queued.text.len() > 50 { "..." } else { "" };
                    let _ = outlet
                        .say(&format!("▶️ Processing queued: \"{}{}\"", preview, suffix))
                        .await;

                    current_text = queued.text;
                    current_user_role = queued.user_role;
                    current_channel_ctx = queued.channel_ctx;
                    current_attachments = queued.attachments;
                    let desc: String = current_text.chars().take(80).collect();
                    let (new_task_id, new_cancel_token) =
                        registry.register(&session_id, &desc).await;
                    current_task_id = new_task_id;
                    current_cancel_token = new_cancel_token;

                    let (new_status_tx, new_status_rx) =
                        tokio::sync::mpsc::channel::<StatusUpdate>(16);
                    current_status_tx = new_status_tx;
                    current_status_task =
                        Self::spawn_status_task(new_status_rx, outlet.clone(), is_dm);

                    current_heartbeat = Arc::new(AtomicU64::new(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    ));
                    current_typing_cancel = tokio_util::sync::CancellationToken::new();
                    Self::spawn_typing_indicator(
                        outlet.clone(),
                        current_typing_cancel.clone(),
                        current_heartbeat.clone(),
                        stale_threshold_secs,
                    );
                    registry
                        .set_typing_cancel(current_task_id, current_typing_cancel.clone())
                        .await;
                } else {
                    break;
                }
            }
        });
    }

    /// The reply to a slash command (the transport delivers it).
    async fn slash_command_reply(
        &self,
        user_id: u64,
        guild: bool,
        channel_id: &str,
        name: &str,
        arg: &str,
    ) -> String {
        // In guild channels, let all server members through (Guest role).
        // Auth gate only applies to DMs.
        if !guild && !self.is_authorized(user_id) {
            return "Unauthorized.".to_string();
        }
        let session_id = self.session_for(user_id, guild, channel_id);
        let cmd_text = format!("/{} {}", name, arg);
        let user_role = super::telegram::determine_role(&self.owner_user_ids, user_id);
        self.dispatch_command(cmd_text.trim(), &session_id, user_role)
            .await
    }

    /// Handle a slash command interaction from the legacy client.
    async fn handle_slash_command(&self, ctx: &Context, command: &CommandInteraction) {
        let user_id = command.user.id.get();
        let guild = command.guild_id.is_some();
        if !guild && !self.is_authorized(user_id) {
            let _ = command
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content("Unauthorized.")
                            .ephemeral(true),
                    ),
                )
                .await;
            return;
        }

        // Defer to buy time for longer operations
        let _ = command
            .create_response(
                &ctx.http,
                CreateInteractionResponse::Defer(CreateInteractionResponseMessage::new()),
            )
            .await;

        let arg = command
            .data
            .options
            .first()
            .and_then(|o| o.value.as_str())
            .unwrap_or("");
        let reply = self
            .slash_command_reply(
                user_id,
                guild,
                &command.channel_id.to_string(),
                &command.data.name,
                arg,
            )
            .await;

        let _ = command
            .edit_response(&ctx.http, EditInteractionResponse::new().content(reply))
            .await;
    }

    /// Dispatch a command string to the appropriate handler and return the reply text.
    async fn dispatch_command(&self, text: &str, session_id: &str, user_role: UserRole) -> String {
        let parts: Vec<&str> = text.splitn(2, ' ').collect();
        let cmd = parts[0];
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
            "/help" | "/start" => build_help_text(&discord_commands(), "/"),
            _ => format!(
                "Unknown command: {}\nType /help for available commands.",
                cmd
            ),
        }
    }

    /// Save inbound files to the inbox.
    async fn handle_file_message(
        &self,
        files: &[InboundFile],
    ) -> anyhow::Result<Vec<crate::traits::MessageAttachment>> {
        let mut attachments = Vec::new();

        for file in files {
            let max_bytes = self.max_file_size_mb * 1_048_576;
            if file.size > max_bytes {
                anyhow::bail!(
                    "File too large ({:.1} MB). Maximum is {} MB.",
                    file.size as f64 / 1_048_576.0,
                    self.max_file_size_mb
                );
            }

            let bytes = file.download().await?;
            // The size Discord reported can be missing; the bytes decide.
            if bytes.len() as u64 > max_bytes {
                anyhow::bail!(
                    "File too large ({:.1} MB). Maximum is {} MB.",
                    bytes.len() as f64 / 1_048_576.0,
                    self.max_file_size_mb
                );
            }

            let mime_type = file.mime.as_deref().unwrap_or("application/octet-stream");

            let sanitized = sanitize_filename(&file.name);
            let uuid_prefix = uuid::Uuid::new_v4().to_string()[..8].to_string();
            let dest_name = format!("{}_{}", uuid_prefix, sanitized);
            let dest_path = self.inbox_dir.join(&dest_name);

            std::fs::create_dir_all(&self.inbox_dir)?;
            std::fs::write(&dest_path, &bytes)?;

            info!(
                file = %dest_path.display(),
                size = bytes.len(),
                mime = %mime_type,
                "Saved inbound Discord file"
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

    /// Answer a pending approval or goal confirmation from a button press.
    async fn resolve_button(
        &self,
        user_id: u64,
        prefix: &str,
        action: &str,
        approval_id: &str,
        original_content: &str,
    ) -> ButtonOutcome {
        // Approval buttons require owner or allowed-user authorization
        // (guild membership alone is not sufficient for approving actions).
        if !self.is_authorized(user_id) && !self.owner_user_ids.contains(&user_id) {
            return ButtonOutcome::Unauthorized;
        }
        if !matches!(prefix, "approve" | "goal" | "autopilot") {
            return ButtonOutcome::Ignored;
        }
        let response = if matches!(prefix, "goal" | "autopilot") {
            match action {
                "confirm" => ApprovalResponse::AllowOnce,
                "cancel" | "edit" => ApprovalResponse::Deny,
                _ => return ButtonOutcome::Ignored,
            }
        } else {
            match action {
                "once" => ApprovalResponse::AllowOnce,
                "session" => ApprovalResponse::AllowSession,
                "always" => ApprovalResponse::AllowAlways,
                "deny" => ApprovalResponse::Deny,
                _ => return ButtonOutcome::Ignored,
            }
        };

        let updated = if matches!(prefix, "goal" | "autopilot") {
            let status = match (prefix, action, &response) {
                ("autopilot", "edit", _) => {
                    "✏️ Autopilot not activated\nSend the accounts, actions, limits, or guardrails you want changed."
                }
                ("autopilot", _, ApprovalResponse::Deny) => {
                    "❌ Autopilot cancelled\nNo new autonomy was activated."
                }
                ("autopilot", _, _) => "✅ Autopilot enabled\nActivation is continuing now.",
                (_, _, ApprovalResponse::Deny) => "❌ Goal cancelled\nNothing was activated.",
                _ => "✅ Goal approved\nActivation is continuing now.",
            };
            format!("{}\n\n{status}", original_content.trim_end())
        } else {
            approval_render::finalize_approval_message(original_content, &response)
        };

        let mut pending = self.pending_approvals.lock().await;
        if let Some(tx) = pending.remove(approval_id) {
            let _ = tx.send(response);
        } else {
            warn!(approval_id, "Stale approval callback (no pending request)");
        }
        ButtonOutcome::Resolved(updated)
    }

    /// Handle a button interaction (approval flow) from the legacy client.
    async fn handle_component_interaction(
        &self,
        ctx: &Context,
        interaction: &ComponentInteraction,
    ) {
        let parts: Vec<&str> = interaction.data.custom_id.splitn(3, ':').collect();
        if parts.len() != 3 {
            return;
        }
        let outcome = self
            .resolve_button(
                interaction.user.id.get(),
                parts[0],
                parts[1],
                parts[2],
                &interaction.message.content,
            )
            .await;
        let response = match outcome {
            ButtonOutcome::Ignored => return,
            ButtonOutcome::Unauthorized => CreateInteractionResponse::Message(
                CreateInteractionResponseMessage::new()
                    .content("Unauthorized.")
                    .ephemeral(true),
            ),
            // Update the message to show the outcome and remove the buttons.
            ButtonOutcome::Resolved(updated) => CreateInteractionResponse::UpdateMessage(
                CreateInteractionResponseMessage::new()
                    .content(updated)
                    .components(vec![]),
            ),
        };
        let _ = interaction.create_response(&ctx.http, response).await;
    }

    /// Register slash commands with Discord (legacy client).
    async fn register_commands(&self, ctx: &Context) {
        let commands: Vec<CreateCommand> = SLASH_COMMANDS
            .iter()
            .map(|command| {
                let mut create = CreateCommand::new(command.name).description(command.description);
                if let Some((name, description, required)) = command.argument {
                    create = create.add_option(
                        serenity::builder::CreateCommandOption::new(
                            serenity::all::CommandOptionType::String,
                            name,
                            description,
                        )
                        .required(required),
                    );
                }
                create
            })
            .collect();

        if let Some(guild_id) = self.guild_id {
            // Guild-scoped: instant propagation
            let guild = GuildId::new(guild_id);
            if let Err(e) = guild.set_commands(&ctx.http, commands).await {
                warn!("Failed to register guild slash commands: {}", e);
            } else {
                info!(guild_id, "Registered guild slash commands");
            }
        } else {
            // Global: ~1hr propagation
            if let Err(e) = Command::set_global_commands(&ctx.http, commands).await {
                warn!("Failed to register global slash commands: {}", e);
            } else {
                info!("Registered global slash commands");
            }
        }
    }

    /// Send a prompt whose last part carries `buttons` (custom id, label,
    /// style), splitting long text over several messages. On the SDK
    /// transport the prompt is remembered under `approval_id` so its answer
    /// can replace it with the outcome.
    async fn send_prompt(
        &self,
        outlet: &Outlet,
        approval_id: &str,
        pages: &[String],
        buttons: &[(String, &str, ButtonStyle)],
    ) -> anyhow::Result<()> {
        let Some((last, leading)) = pages.split_last() else {
            anyhow::bail!("empty prompt");
        };
        for page in leading {
            outlet.say(&truncate_chars(page, MAX_MESSAGE_LEN)).await?;
        }
        match outlet {
            Outlet::Serenity { http, channel } => {
                let row = CreateActionRow::Buttons(
                    buttons
                        .iter()
                        .map(|(id, label, style)| {
                            CreateButton::new(id.clone()).label(*label).style(*style)
                        })
                        .collect(),
                );
                let message = CreateMessage::new()
                    .content(truncate_chars(last, MAX_MESSAGE_LEN))
                    .components(vec![row]);
                channel.send_message(http, message).await?;
            }
            Outlet::Sdk {
                transport,
                conversation,
            } => {
                // The adapter styles by position: green first, red last.
                let prompt = aidaemon_channels::ApprovalPrompt {
                    id: approval_id.to_string(),
                    conversation: conversation.clone(),
                    text: last.clone(),
                    choices: buttons
                        .iter()
                        .map(|(id, label, _)| aidaemon_channels::ApprovalChoice {
                            // The choice is the custom id without the
                            // approval id the prompt already carries.
                            id: id
                                .rsplit_once(':')
                                .map(|(choice, _)| choice.to_string())
                                .unwrap_or_else(|| id.clone()),
                            label: label.to_string(),
                        })
                        .collect(),
                };
                let message = transport.ask(&prompt).await?;
                self.sdk_prompts
                    .lock()
                    .await
                    .insert(approval_id.to_string(), (message, last.clone()));
            }
        }
        Ok(())
    }
}

/// The `discord:dm:{user}` or `discord:ch:{channel}` part of a session id
/// ("{bot_username}:discord:dm:{user}" for a named bot, "discord:dm:{user}"
/// for the default one).
fn base_session_id(session_id: &str) -> &str {
    ["discord:dm:", "discord:ch:"]
        .iter()
        .find_map(|marker| session_id.find(marker).map(|at| &session_id[at..]))
        .unwrap_or(session_id)
}

/// The status line for one update.
fn status_text(update: &StatusUpdate) -> String {
    match update {
        StatusUpdate::Thinking(_) => "Thinking...".to_string(),
        StatusUpdate::ToolStart { name, summary } => {
            let mut chars = name.chars();
            let cap_name = match chars.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
            };
            if summary.is_empty() {
                format!("{}...", cap_name)
            } else {
                format!("{}: {}...", cap_name, summary)
            }
        }
        StatusUpdate::ToolProgress { name, chunk } => {
            let preview: String = chunk.chars().take(100).collect();
            if chunk.chars().count() > 100 {
                format!("📤 {}: {}...", name, preview)
            } else {
                format!("📤 {}: {}", name, preview)
            }
        }
        StatusUpdate::UserActionRequired { message } => message.clone(),
        StatusUpdate::ToolComplete { name, summary } => {
            format!("✓ {}: {}", name, summary)
        }
        StatusUpdate::ToolCancellable { name, task_id } => {
            format!("⏳ {} started (task_id: {})", name, task_id)
        }
        StatusUpdate::ProgressSummary {
            elapsed_mins,
            summary,
        } => {
            format!("📊 Progress ({} min): {}", elapsed_mins, summary)
        }
        StatusUpdate::IterationWarning { current, threshold } => {
            format!(
                "⚠️ Approaching soft limit: {} of {} iterations",
                current, threshold
            )
        }
        StatusUpdate::PlanCreated {
            description,
            total_steps,
            ..
        } => {
            format!("📋 Plan created: {} ({} steps)", description, total_steps)
        }
        StatusUpdate::PlanStepStart {
            step_index,
            total_steps,
            description,
            ..
        } => {
            format!(
                "▶️ Step {}/{}: {}",
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
                "✅ Step {}/{} done: {}",
                step_index + 1,
                total_steps,
                description
            );
            if let Some(s) = summary {
                format!("{} - {}", base, s)
            } else {
                base
            }
        }
        StatusUpdate::PlanStepFailed {
            step_index,
            description,
            error,
            ..
        } => {
            format!(
                "❌ Step {} failed: {} - {}",
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
                "🎉 Plan complete: {} ({} steps in {}m {}s)",
                description, total_steps, mins, secs
            )
        }
        StatusUpdate::PlanAbandoned { description, .. } => {
            format!("🚫 Plan abandoned: {}", description)
        }
        StatusUpdate::PlanRevised {
            description,
            reason,
            new_total_steps,
            ..
        } => {
            format!(
                "🔄 Plan revised: {} ({} steps) - {}",
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
                "💰 Auto-extended token budget {} → {} ({}/{}) — continuing.",
                old_budget, new_budget, extension, max_extensions
            )
        }
        StatusUpdate::Checklist { text } => text.clone(),
    }
}

/// The first `max` characters of `text`.
fn truncate_chars(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

#[async_trait]
impl Channel for DiscordChannel {
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
        let outlet = self.outlet_for(session_id).await?;
        let mut first_err: Option<anyhow::Error> = None;
        for chunk in split_message(text, MAX_MESSAGE_LEN) {
            if let Err(e) = outlet.say(&chunk).await {
                warn!("Failed to send Discord message: {}", e);
                if first_err.is_none() {
                    first_err = Some(anyhow::anyhow!("Failed to send Discord message: {}", e));
                }
            }
        }
        if let Some(err) = first_err {
            return Err(err);
        }
        Ok(())
    }

    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        let outlet = self.outlet_for(session_id).await?;
        let (bytes, filename) = match &media.kind {
            MediaKind::Photo { data } => (data.clone(), "screenshot.png".to_string()),
            MediaKind::Document {
                file_path,
                filename,
            } => (tokio::fs::read(file_path).await?, filename.clone()),
        };
        match outlet {
            Outlet::Serenity { http, channel } => {
                let mut msg = CreateMessage::new();
                if !media.caption.is_empty() {
                    msg = msg.content(&media.caption);
                }
                msg = msg.add_file(CreateAttachment::bytes(bytes, filename));
                channel
                    .send_message(&http, msg)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to send file: {}", e))?;
            }
            Outlet::Sdk {
                transport,
                conversation,
            } => {
                transport
                    .send_file(
                        &conversation,
                        &aidaemon_channels::OutboundFile {
                            name: filename,
                            bytes,
                            mime_type: None,
                            caption: (!media.caption.is_empty()).then(|| media.caption.clone()),
                        },
                    )
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to send file: {}", e))?;
            }
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
        let outlet = self.outlet_for(session_id).await?;

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
                "Stored pending Discord approval"
            );
        }

        let use_session_button =
            approval_render::approval_use_session_button(permission_mode, risk_level);

        let mut buttons = vec![(
            format!("approve:once:{}", approval_id),
            "Allow Once",
            ButtonStyle::Primary,
        )];
        if !one_time_only {
            buttons.push(if use_session_button {
                (
                    format!("approve:session:{}", approval_id),
                    "Allow Session",
                    ButtonStyle::Success,
                )
            } else {
                (
                    format!("approve:always:{}", approval_id),
                    "Allow Always",
                    ButtonStyle::Success,
                )
            });
        }
        buttons.push((
            format!("approve:deny:{}", approval_id),
            "Deny",
            ButtonStyle::Danger,
        ));

        let text = approval_render::build_approval_message_discord(
            command,
            risk_level,
            warnings,
            use_session_button,
            one_time_only,
        );

        match self
            .send_prompt(&outlet, &approval_id, &[text], &buttons)
            .await
        {
            Ok(()) => {
                info!(approval_id = %short_id, "Approval message sent to Discord");
            }
            Err(e) => {
                warn!("Failed to send Discord approval request: {}", e);
                self.pending_approvals.lock().await.remove(&approval_id);
                return Err(anyhow::anyhow!(
                    "Discord approval request could not be delivered: {e}"
                ));
            }
        }

        // Wait for response with 5-minute timeout to prevent memory leak
        info!(approval_id = %short_id, "Waiting for user approval response...");
        let outcome = tokio::time::timeout(Duration::from_secs(300), response_rx).await;
        self.sdk_prompts.lock().await.remove(&approval_id);
        match outcome {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                warn!(approval_id = %short_id, "Approval channel closed");
                self.pending_approvals.lock().await.remove(&approval_id);
                Err(anyhow::anyhow!("Discord approval response channel closed"))
            }
            Err(_) => {
                warn!(approval_id = %short_id, "Approval timed out after 5 minutes");
                self.pending_approvals.lock().await.remove(&approval_id);
                Err(anyhow::anyhow!("Discord approval request timed out"))
            }
        }
    }

    async fn request_goal_confirmation(
        &self,
        session_id: &str,
        goal_description: &str,
        details: &[String],
        style: GoalConfirmationStyle,
    ) -> anyhow::Result<bool> {
        let outlet = self.outlet_for(session_id).await?;
        let approval_id = uuid::Uuid::new_v4().to_string();
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.pending_approvals
            .lock()
            .await
            .insert(approval_id.clone(), response_tx);

        let pages =
            approval_render::build_goal_confirmation_pages(goal_description, details, style);
        let prefix = match style {
            GoalConfirmationStyle::Standard => "goal",
            GoalConfirmationStyle::Autopilot => "autopilot",
        };
        let approve_label = match style {
            GoalConfirmationStyle::Standard => "Approve & activate",
            GoalConfirmationStyle::Autopilot => "Enable Autopilot",
        };
        let mut buttons = vec![(
            format!("{prefix}:confirm:{approval_id}"),
            approve_label,
            ButtonStyle::Primary,
        )];
        if matches!(style, GoalConfirmationStyle::Autopilot) {
            buttons.push((
                format!("{prefix}:edit:{approval_id}"),
                "Edit permissions",
                ButtonStyle::Secondary,
            ));
        }
        buttons.push((
            format!("{prefix}:cancel:{approval_id}"),
            "Cancel",
            ButtonStyle::Danger,
        ));
        if let Err(error) = self
            .send_prompt(&outlet, &approval_id, &pages, &buttons)
            .await
        {
            self.pending_approvals.lock().await.remove(&approval_id);
            return Err(anyhow::anyhow!(
                "Discord goal confirmation could not be delivered: {error}"
            ));
        }

        let outcome = tokio::time::timeout(Duration::from_secs(1800), response_rx).await;
        self.sdk_prompts.lock().await.remove(&approval_id);
        match outcome {
            Ok(Ok(response)) => Ok(!matches!(response, ApprovalResponse::Deny)),
            Ok(Err(_)) => {
                self.pending_approvals.lock().await.remove(&approval_id);
                Err(anyhow::anyhow!("Discord goal confirmation channel closed"))
            }
            Err(_) => {
                self.pending_approvals.lock().await.remove(&approval_id);
                Err(anyhow::anyhow!("Discord goal confirmation timed out"))
            }
        }
    }
}

/// Serenity event handler that bridges to our DiscordChannel methods.
struct DiscordHandler {
    channel: Arc<DiscordChannel>,
}

#[async_trait]
impl EventHandler for DiscordHandler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        let username = ready.user.name.clone();
        info!(username = %username, "Discord bot connected");
        self.channel.set_bot_username(username);
        self.channel.register_commands(&ctx).await;
    }

    async fn message(&self, ctx: Context, msg: SerenityMessage) {
        self.channel.handle_message_event(&ctx, msg).await;
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        match interaction {
            Interaction::Command(command) => {
                self.channel.handle_slash_command(&ctx, &command).await;
            }
            Interaction::Component(component) => {
                self.channel
                    .handle_component_interaction(&ctx, &component)
                    .await;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
impl DiscordChannel {
    /// Run on this SDK transport (a test points it at a mock API).
    fn use_sdk_transport(&self, transport: Arc<SdkDiscord>) {
        if let Ok(mut slot) = self.sdk.write() {
            *slot = Some(transport);
        }
    }
}

#[cfg(test)]
#[path = "discord_tests.rs"]
mod tests;

/// Spawn a DiscordChannel in a background task.
/// This is a separate function to avoid async type inference cycles.
pub fn spawn_discord_channel(channel: Arc<DiscordChannel>) {
    tokio::spawn(async move {
        channel.start_with_retry().await;
    });
}
