//! The Discord bot: the `aidaemon-channel-discord` adapter carries the
//! traffic; this module decides who may talk to the bot, which session a
//! message belongs to, and answers commands. Turns run on the shared
//! [`TurnRunner`] and prompts on the shared [`Prompts`].

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};
use std::time::Duration;

use aidaemon_channel_discord::DiscordChannel as SdkDiscord;
use aidaemon_channels::{
    Channel as _, ChannelEvent, CommandSpec, Conversation, ConversationKind, InboundMessage,
    OutboundFile, OutboundMessage,
};
use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::commands::{shared_commands, CommandCategory, CommandDef};
use super::prompts::Prompts;
use super::turns::{repeat_while_working, Turn, TurnRunner, TurnSurface};
use crate::channels::{ChannelHub, ChannelRuntimeDeps};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::traits::{ChannelCapabilities, HubChannel};
use crate::types::{ApprovalResponse, GoalConfirmationStyle, MediaKind, MediaMessage};
use crate::types::{ChannelContext, ChannelVisibility, UserRole};
use aidaemon_channels::slash::build_help_text;
use aidaemon_channels::split_message;
use aidaemon_events::attachments::sanitize_filename;

/// Longest Discord message: the adapter's limit.
const MAX_MESSAGE_LEN: usize = aidaemon_channel_discord::MAX_MESSAGE_LEN;

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

/// The slash commands the bot registers: name and description.
const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("model", "Show or switch the current LLM model"),
    ("models", "List available models from provider"),
    ("auto", "Re-enable automatic model routing"),
    ("reload", "Reload config.toml"),
    ("tasks", "List running and recent tasks"),
    ("cancel", "Stop what's running in this chat"),
    (
        "clear",
        "Start fresh conversation (history kept for memory/audit)",
    ),
    ("wipe", "Permanently delete this conversation"),
    ("cost", "Show token usage statistics"),
    ("context", "Show conversation context coverage"),
    ("help", "Show available commands"),
];

/// One Discord conversation, as the turn runner shows it.
struct Outlet {
    transport: Arc<SdkDiscord>,
    conversation: Conversation,
}

impl Outlet {
    /// Send `text`, split to fit; the ids of the parts sent.
    async fn say_all(&self, text: &str) -> anyhow::Result<Vec<String>> {
        let mut ids = Vec::new();
        let mut failure = None;
        for chunk in split_message(text, MAX_MESSAGE_LEN) {
            match self
                .transport
                .send(&self.conversation, &OutboundMessage::text(chunk))
                .await
            {
                Ok(sent) => ids.push(sent.message_id),
                Err(error) => {
                    warn!("Failed to send Discord message: {}", error);
                    failure = Some(error);
                }
            }
        }
        match failure {
            Some(error) => Err(error.into()),
            None => Ok(ids),
        }
    }
}

#[async_trait]
impl TurnSurface for Outlet {
    fn platform(&self) -> &'static str {
        "discord"
    }

    async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
        self.say_all(text).await
    }

    fn show_working(&self, stop: CancellationToken, heartbeat: Arc<AtomicU64>, stale_secs: u64) {
        let transport = self.transport.clone();
        let conversation = self.conversation.clone();
        // Discord's typing indicator lasts about ten seconds.
        repeat_while_working(
            Duration::from_secs(8),
            stop,
            heartbeat,
            stale_secs,
            move || {
                let transport = transport.clone();
                let conversation = conversation.clone();
                async move {
                    let _ = transport.typing(&conversation).await;
                }
            },
        );
    }
}

/// Discord channel implementation.
pub struct DiscordChannel {
    /// Bot username, once the gateway reports it ("discord" until then).
    bot_username: std::sync::RwLock<String>,
    bot_token: String,
    allowed_user_ids: std::sync::RwLock<Vec<u64>>,
    /// Discord user IDs recognized as owners (from `users.owner_ids.discord`).
    owner_user_ids: Vec<u64>,
    deps: ChannelRuntimeDeps,
    runner: TurnRunner,
    prompts: Prompts,
    transport: std::sync::RwLock<Arc<SdkDiscord>>,
    /// Direct conversations opened, by user id.
    dm_conversations: Mutex<HashMap<u64, Conversation>>,
    /// Reference to the channel hub for dynamic bot registration.
    channel_hub: std::sync::RwLock<Option<Weak<ChannelHub>>>,
}

impl DiscordChannel {
    pub fn new(
        bot_token: &str,
        allowed_user_ids: Vec<u64>,
        owner_user_ids: Vec<u64>,
        guild_id: Option<u64>,
        runtime: ChannelRuntimeDeps,
    ) -> Self {
        let mut transport = SdkDiscord::new(bot_token.to_string());
        if let Some(guild) = guild_id {
            transport = transport.with_command_guild(guild.to_string());
        }
        Self {
            bot_username: std::sync::RwLock::new("discord".to_string()),
            bot_token: bot_token.to_string(),
            allowed_user_ids: std::sync::RwLock::new(allowed_user_ids),
            owner_user_ids,
            runner: TurnRunner::new(runtime.turn_deps()),
            deps: runtime,
            prompts: Prompts::default(),
            transport: std::sync::RwLock::new(Arc::new(transport)),
            dm_conversations: Mutex::new(HashMap::new()),
            channel_hub: std::sync::RwLock::new(None),
        }
    }

    /// Set the channel hub reference for dynamic bot registration.
    pub fn set_channel_hub(&self, hub: Weak<ChannelHub>) {
        if let Ok(mut guard) = self.channel_hub.write() {
            *guard = Some(hub);
        }
    }

    fn transport(&self) -> Arc<SdkDiscord> {
        self.transport
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn get_bot_username(&self) -> String {
        self.bot_username
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

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

    /// Run the adapter, restarting it with backoff when it stops.
    pub async fn start_with_retry(self: Arc<Self>) {
        let mut backoff = crate::backoff::ChannelReconnectBackoff::new(
            Duration::from_secs(5),
            Duration::from_secs(60),
            Duration::from_secs(60),
        );
        loop {
            info!("Starting Discord client");
            let started = tokio::time::Instant::now();
            self.clone().run().await;
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

    /// Run the adapter until it stops, handling each event on its own task.
    async fn run(self: Arc<Self>) {
        let transport = self.transport();
        let commands: Vec<CommandSpec> = SLASH_COMMANDS
            .iter()
            .map(|(name, description)| CommandSpec {
                name: name.to_string(),
                description: description.to_string(),
            })
            .collect();
        match transport.register_commands(&commands).await {
            Ok(()) => info!("Registered Discord slash commands"),
            Err(e) => warn!("Failed to register Discord slash commands: {}", e),
        }

        let (events_tx, mut events) = tokio::sync::mpsc::channel(64);
        let shutdown = CancellationToken::new();
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
            tokio::spawn(async move { this.handle_event(event).await });
        }
        shutdown.cancel();
        match runner.await {
            Ok(Err(e)) => warn!("Discord client error: {}", e),
            Err(e) => warn!("Discord client task failed: {}", e),
            Ok(Ok(())) => {}
        }
    }

    async fn handle_event(self: &Arc<Self>, event: ChannelEvent) {
        let transport = self.transport();
        match event {
            ChannelEvent::Message(message) => self.handle_message(message).await,
            ChannelEvent::Command(command) => {
                let Ok(user_id) = command.sender.id.parse::<u64>() else {
                    return;
                };
                let guild = command.conversation.kind != ConversationKind::Direct;
                let reply = self
                    .slash_command_reply(
                        user_id,
                        guild,
                        &command.conversation.id,
                        &command.as_text(),
                    )
                    .await;
                let reply: String = reply.chars().take(MAX_MESSAGE_LEN).collect();
                if let Err(e) = transport.answer_command(&command, &reply).await {
                    warn!("Failed to answer a Discord slash command: {}", e);
                }
            }
            ChannelEvent::Approval(answer) => {
                let Ok(user_id) = answer.sender.id.parse::<u64>() else {
                    return;
                };
                if !self.prompts.is_open(&answer.prompt_id).await {
                    warn!(prompt = %answer.prompt_id, "Stale approval callback (no pending request)");
                    return;
                }
                // Answering needs an allowed user or owner; being in the
                // server is not enough.
                if !self.is_authorized(user_id) && !self.owner_user_ids.contains(&user_id) {
                    warn!(user_id, "Unauthorized Discord approval button press");
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
                    let text: String = text.chars().take(MAX_MESSAGE_LEN).collect();
                    if let Err(e) = transport.edit(&message, &text).await {
                        warn!("Failed to update a Discord approval message: {}", e);
                    }
                }
            }
        }
    }

    /// Handle an incoming message.
    async fn handle_message(self: &Arc<Self>, message: InboundMessage) {
        let Ok(user_id) = message.sender.id.parse::<u64>() else {
            return;
        };
        let guild = message.conversation.kind != ConversationKind::Direct;
        let outlet = Outlet {
            transport: self.transport(),
            conversation: message.conversation.clone(),
        };
        // Server members may talk to the bot (as guests); DMs need access.
        if !guild && !self.is_authorized(user_id) && !self.try_auto_claim(user_id).await {
            warn!(user_id, "Unauthorized Discord user attempted access");
            let _ = outlet.say_all("Unauthorized.").await;
            return;
        }

        let mut attachments = Vec::new();
        if !message.attachments.is_empty() {
            if !self.deps.files_enabled {
                if message.text.is_empty() {
                    let _ = outlet.say_all("I can only process text messages.").await;
                    return;
                }
            } else {
                match self.save_files(&message.attachments).await {
                    Ok(saved) => attachments = saved,
                    Err(e) => {
                        let _ = outlet.say_all(&format!("File error: {}", e)).await;
                        return;
                    }
                }
            }
        }
        if message.text.is_empty() && attachments.is_empty() {
            return;
        }

        let user_role = aidaemon_channels::bot::determine_role(&self.owner_user_ids, user_id);
        let text = if user_role == UserRole::Owner {
            super::attachments::build_inbound_text(&message.text, &attachments)
        } else {
            super::attachments::build_inbound_text_without_local_paths(&message.text, &attachments)
        };
        let session_id = self.session_for(user_id, guild, &message.conversation.id);

        if super::commands::is_slash_command(&text) {
            let reply = self
                .dispatch_command(&text, &session_id, user_id, user_role)
                .await;
            let _ = outlet.say_all(&reply).await;
            return;
        }

        let channel_ctx = ChannelContext {
            visibility: if guild {
                ChannelVisibility::Public
            } else {
                ChannelVisibility::Private
            },
            platform: "discord".to_string(),
            channel_name: None,
            channel_id: Some(if guild {
                format!("discord:{}", message.conversation.id)
            } else {
                format!("discord:dm:{}", user_id)
            }),
            workspace_id: None,
            sender_name: message.sender.display_name.clone(),
            sender_id: Some(format!("discord:{}", user_id)),
            channel_member_names: vec![],
            user_id_map: HashMap::new(),
            workspace_grant: None,
            trusted: false,
        };
        self.runner
            .register_session(&session_id, &self.channel_name())
            .await;
        info!(session_id, "Received Discord message from user {}", user_id);
        self.runner
            .submit(
                Arc::new(outlet),
                Turn {
                    session_id,
                    text,
                    attachments,
                    user_role,
                    channel_ctx,
                    dedup_key: Some(message.message_id),
                    ingress_timing: None,
                },
            )
            .await;
    }

    /// Where messages for `session_id` go.
    async fn conversation_for(&self, session_id: &str) -> anyhow::Result<Conversation> {
        let base = base_session_id(session_id);
        if let Some(user) = base.strip_prefix("discord:dm:") {
            let user_id: u64 = user.parse()?;
            if let Some(conversation) = self.dm_conversations.lock().await.get(&user_id) {
                return Ok(conversation.clone());
            }
            let conversation = self.transport().open_direct(user).await?;
            self.dm_conversations
                .lock()
                .await
                .insert(user_id, conversation.clone());
            return Ok(conversation);
        }
        if let Some(channel) = base.strip_prefix("discord:ch:") {
            return Ok(Conversation {
                id: channel.to_string(),
                kind: ConversationKind::Group,
            });
        }
        anyhow::bail!("Invalid Discord session ID: {}", session_id)
    }

    async fn outlet_for(&self, session_id: &str) -> anyhow::Result<Outlet> {
        Ok(Outlet {
            transport: self.transport(),
            conversation: self.conversation_for(session_id).await?,
        })
    }

    fn is_authorized(&self, user_id: u64) -> bool {
        self.allowed_user_ids
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(&user_id)
    }

    /// Auto-claim: a dynamic bot (no owner ids) whose allowlist is empty or
    /// holds only stale ids from another platform (Telegram ids saved for a
    /// Discord bot via /connect) is claimed by the first DM user, and the
    /// claim is saved. Configured bots, which have owner ids, never are.
    async fn try_auto_claim(&self, user_id: u64) -> bool {
        if !self.owner_user_ids.is_empty() {
            return false;
        }
        {
            let mut ids = self
                .allowed_user_ids
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if ids.contains(&user_id) {
                return true;
            }
            info!(
                user_id,
                old_ids = ?*ids,
                "Auto-claiming Discord bot for DM user (replacing stale IDs)"
            );
            ids.clear();
            ids.push(user_id);
        }
        if let Err(e) = self
            .deps
            .state
            .update_dynamic_bot_allowed_users(&self.bot_token, &[user_id.to_string()])
            .await
        {
            warn!("Failed to persist auto-claimed user to DB: {}", e);
        }
        true
    }

    /// The reply to a slash command `text` (the transport delivers it).
    async fn slash_command_reply(
        &self,
        user_id: u64,
        guild: bool,
        channel_id: &str,
        text: &str,
    ) -> String {
        if !guild && !self.is_authorized(user_id) {
            return "Unauthorized.".to_string();
        }
        let session_id = self.session_for(user_id, guild, channel_id);
        let user_role = aidaemon_channels::bot::determine_role(&self.owner_user_ids, user_id);
        self.dispatch_command(text, &session_id, user_id, user_role)
            .await
    }

    /// Dispatch a command string to the appropriate handler and return the reply text.
    async fn dispatch_command(
        &self,
        text: &str,
        session_id: &str,
        user_id: u64,
        user_role: UserRole,
    ) -> String {
        let (cmd, arg) = text.split_once(' ').unwrap_or((text, ""));
        // The same sender id the turn's channel context carries.
        let sender_id = format!("discord:{user_id}");
        let ctx = super::commands::CommandContext {
            agent: Arc::clone(&self.deps.agent),
            state: Arc::clone(&self.deps.state),
            task_registry: Arc::clone(&self.deps.task_registry),
            config_path: self.deps.config_path.clone(),
        };
        if let Some(reply) = ctx
            .dispatch(cmd, arg.trim(), session_id, Some(&sender_id), user_role)
            .await
        {
            return reply;
        }
        match cmd {
            "/help" | "/start" => build_help_text(
                &discord_commands(),
                "/",
                aidaemon_channels::slash::HelpScope::from_args(arg),
            ),
            _ => aidaemon_channels::slash::unknown_command_reply(cmd, &discord_commands(), "/"),
        }
    }

    /// Save inbound files to the inbox.
    async fn save_files(
        &self,
        files: &[aidaemon_channels::Attachment],
    ) -> anyhow::Result<Vec<crate::traits::MessageAttachment>> {
        let max_bytes = self.deps.max_file_size_mb * 1_048_576;
        let too_large = |bytes: u64| {
            anyhow::anyhow!(
                "File too large ({:.1} MB). Maximum is {} MB.",
                bytes as f64 / 1_048_576.0,
                self.deps.max_file_size_mb
            )
        };
        let mut attachments = Vec::new();
        for file in files {
            if file.size_bytes.unwrap_or(0) > max_bytes {
                return Err(too_large(file.size_bytes.unwrap_or(0)));
            }
            let bytes = self.transport().download(file).await?;
            // The size Discord reported can be missing; the bytes decide.
            if bytes.len() as u64 > max_bytes {
                return Err(too_large(bytes.len() as u64));
            }
            let mime_type = file
                .mime_type
                .clone()
                .unwrap_or_else(|| "application/octet-stream".to_string());
            let sanitized = sanitize_filename(&file.name);
            let dest_name = format!("{}_{}", &uuid::Uuid::new_v4().to_string()[..8], sanitized);
            let dest_path = self.deps.inbox_dir.join(&dest_name);
            std::fs::create_dir_all(&self.deps.inbox_dir)?;
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
                mime_type,
                bytes.len() as u64,
            ));
        }
        Ok(attachments)
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

#[async_trait]
impl HubChannel for DiscordChannel {
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
        self.outlet_for(session_id)
            .await?
            .say_all(text)
            .await
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("Failed to send Discord message: {}", e))
    }

    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        let conversation = self.conversation_for(session_id).await?;
        let (bytes, name) = match &media.kind {
            MediaKind::Photo { data } => (data.clone(), "screenshot.png".to_string()),
            MediaKind::Document {
                file_path,
                filename,
            } => (tokio::fs::read(file_path).await?, filename.clone()),
        };
        self.transport()
            .send_file(
                &conversation,
                &OutboundFile {
                    name,
                    bytes,
                    mime_type: None,
                    caption: (!media.caption.is_empty()).then(|| media.caption.clone()),
                },
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send file: {}", e))?;
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
        let conversation = self.conversation_for(session_id).await?;
        self.prompts
            .approval(
                self.transport(),
                &conversation,
                command,
                risk_level,
                warnings,
                permission_mode,
                one_time_only,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Discord approval request failed: {e}"))
    }

    async fn request_goal_confirmation(
        &self,
        session_id: &str,
        goal_description: &str,
        details: &[String],
        style: GoalConfirmationStyle,
    ) -> anyhow::Result<bool> {
        let conversation = self.conversation_for(session_id).await?;
        self.prompts
            .goal(
                self.transport(),
                &conversation,
                goal_description,
                details,
                style,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Discord goal confirmation failed: {e}"))
    }
}

#[cfg(test)]
impl DiscordChannel {
    /// Run on this transport (a test points it at a mock API).
    fn use_transport(&self, transport: Arc<SdkDiscord>) {
        if let Ok(mut slot) = self.transport.write() {
            *slot = transport;
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
