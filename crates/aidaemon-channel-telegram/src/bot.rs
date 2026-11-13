//! The Telegram bot: who may talk to it, which session a chat is, files,
//! commands, the live status card and replies. [`crate::TelegramBot`]
//! carries the traffic; turns run on the shared
//! [`aidaemon_channels::turns::TurnRunner`]. What is the application's own
//! (its config file, extra commands, terminal sessions) goes through a
//! [`TelegramHost`].

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock as StdRwLock, Weak};
use std::time::Duration;

pub use aidaemon_channels::bot::determine_role;
use aidaemon_channels::bot::BotDeps;
use aidaemon_channels::context::ChannelContext;
use aidaemon_channels::hub::{
    ChannelCapabilities, ChannelHub, HubChannel, MediaKind, MediaMessage,
};
use aidaemon_channels::live_status::{HubSurfaceSink, SurfaceSink};
use aidaemon_channels::ports::{AgentResponseEnvelope, InboundMessageTiming};
use aidaemon_channels::prompts::Prompts;
use aidaemon_channels::slash::{build_help_text, shared_commands, CommandCategory, CommandDef};
use aidaemon_channels::turns::{repeat_while_working, Delivered, Turn, TurnRunner, TurnSurface};
use aidaemon_channels::Channel as _;
use aidaemon_core::ChannelVisibility;
use aidaemon_events::conversation::MessageAttachment;
use aidaemon_events::payloads::ResponseDeliveryState;
use aidaemon_tools::semantics::{
    ApprovalResponse, GoalConfirmationStyle, PermissionMode, RiskLevel, UserRole,
};
use aidaemon_tools::status::StatusUpdate;
use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use crate::bot_api::{
    Bot, BotCommand, CallbackQuery, ChatAction, ChatId, InlineKeyboardButton, InlineKeyboardMarkup,
    LinkPreviewOptions, Message, MessageId, ParseMode, RequestError, Update,
};
use crate::format::{
    markdown_to_html as markdown_to_telegram_html,
    markdown_to_plain_fallback as markdown_to_telegram_plain_fallback,
    split_html as split_telegram_html, split_text as split_telegram_text,
};

/// Webhook settings for one bot; disabled means long polling.
#[derive(Debug, Clone, Default)]
pub struct TelegramWebhookSettings {
    pub enabled: bool,
    pub public_url: Option<String>,
    pub listen_addr: Option<String>,
    pub path: Option<String>,
    pub max_connections: Option<u8>,
    pub drop_pending_updates: bool,
}

/// What one of the host's own commands did.
pub enum HostCommand {
    /// Not one of the host's commands.
    NotMine,
    /// Handled; the host already answered.
    Handled,
    /// Handled; send this reply.
    Reply(String),
}

/// What the bot needs from the application beyond the agent.
#[async_trait]
pub trait TelegramHost: Send + Sync + 'static {
    /// The host's own commands, listed after the shared ones in the "/"
    /// menu and `/help`.
    fn commands(&self) -> Vec<CommandDef> {
        Vec::new()
    }

    /// The shared slash commands; `None` when `cmd` is not one.
    async fn dispatch_shared(
        &self,
        cmd: &str,
        args: &str,
        session_id: &str,
        sender_id: Option<&str>,
        user_role: UserRole,
    ) -> Option<String>;

    /// One of the host's own commands.
    async fn handle_command(
        &self,
        _bot: &Arc<TelegramBot>,
        _cmd: &str,
        _arg: &str,
        _msg: &Message,
    ) -> HostCommand {
        HostCommand::NotMine
    }

    /// Whether `cmd` runs in its own task: a command that waits for a
    /// button press would otherwise hold up the chat's updates, which carry
    /// that press.
    fn runs_detached(&self, _cmd: &str) -> bool {
        false
    }

    /// A button press the host owns (`data` is its callback data); `false`
    /// leaves it to the bot (approvals).
    async fn handle_callback(
        &self,
        _bot: &Arc<TelegramBot>,
        _query: &CallbackQuery,
        _data: &str,
        _user_id: u64,
    ) -> bool {
        false
    }

    /// A message's text before it becomes a turn; `Some` is the host's
    /// answer instead.
    async fn intercept_text(
        &self,
        _bot: &Arc<TelegramBot>,
        _chat_id: i64,
        _user_id: u64,
        _user_role: UserRole,
        _text: &str,
    ) -> Option<String> {
        None
    }

    /// The Mini App asked to continue a session on the owner's computer.
    async fn continue_on_computer(
        &self,
        _bot: &Arc<TelegramBot>,
        _msg: &Message,
        _user_id: u64,
        _relay_session_id: Option<String>,
    ) {
    }

    /// The first user to write became the owner: keep `allowed` so it
    /// survives a restart. A returned note is added to the welcome.
    async fn owner_claimed(&self, _bot: &Arc<TelegramBot>, _allowed: &[u64]) -> Option<String> {
        None
    }

    /// `/setkey` (owner, private chat): the API keys that can be replaced.
    /// `Err` is the message to show; an empty list means none.
    async fn key_choices(&self) -> Result<Vec<KeyChoice>, String> {
        Ok(Vec::new())
    }

    /// Test `key` for the choice `id` (shown as `label`) and save it only
    /// if its provider accepts it; the reply says what happened.
    async fn install_key(&self, _id: &str, label: &str, _key: &str) -> String {
        format!("The config changed, so {label} was not updated. Run /setkey again.")
    }

    /// The bot is starting (`label`: its username).
    async fn on_start(&self, _bot: &Arc<TelegramBot>, _label: &str) {}

    /// `/private` with its message: keep the sender's next message out of
    /// memory.
    async fn set_next_message_private(
        &self,
        session_id: &str,
        sender_id: Option<&str>,
        on: bool,
    ) -> anyhow::Result<()>;
}

fn disabled_link_preview() -> LinkPreviewOptions {
    LinkPreviewOptions { is_disabled: true }
}

/// One API key `/setkey` can replace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyChoice {
    /// What the button sends back.
    pub id: String,
    /// What the owner sees.
    pub label: String,
}

/// How long `/setkey` waits for the key after a provider is chosen.
const PENDING_KEY_TTL: Duration = Duration::from_secs(120);

struct PendingKey {
    id: String,
    label: String,
    expires_at: std::time::Instant,
}

/// One Telegram bot.
pub struct TelegramBot {
    /// Bot username fetched from Telegram API (e.g., "coding_bot", "debug_bot").
    /// Populated on first start() call via getMe. Uses StdRwLock for sync access in trait methods.
    bot_username: StdRwLock<String>,
    /// Cached channel name for the trait's name() method (e.g., "telegram" or "telegram:my_bot").
    cached_channel_name: StdRwLock<String>,
    /// Stable namespace used when building session IDs.
    /// Once set for a process lifetime, it must not change to avoid
    /// splitting conversation history mid-run.
    session_namespace: StdRwLock<Option<String>>,
    bot: Bot,
    bot_token: String,
    webhook: TelegramWebhookSettings,
    allowed_user_ids: StdRwLock<Vec<u64>>,
    /// Telegram user IDs recognized as owners (from `users.owner_ids.telegram`).
    owner_user_ids: Vec<u64>,
    deps: BotDeps,
    host: Arc<dyn TelegramHost>,
    /// Reference to the channel hub for dynamic bot registration.
    /// Set after construction via set_channel_hub().
    channel_hub: StdRwLock<Option<Weak<ChannelHub>>>,
    /// The Bot API server (a test points it at a mock); `None`: Telegram's.
    api_url: Option<String>,
    runner: TurnRunner,
    /// The SDK adapter: receives updates and asks prompts. Replaced with
    /// the configured one when the bot starts.
    transport: StdRwLock<Arc<crate::TelegramChannel>>,
    prompts: Prompts,
    /// Chats where the owner's next message is an API key, not a request.
    pending_keys: Mutex<HashMap<(i64, u64), PendingKey>>,
}

const TELEGRAM_EXPANDABLE_TRIGGER_CHARS: usize = 1_800;
const TELEGRAM_MAX_MESSAGE_LEN: usize = crate::TELEGRAM_MAX_MESSAGE_LEN;
const TELEGRAM_EXPANDABLE_WRAPPER_LEN: usize = "<blockquote expandable></blockquote>".len();
const TELEGRAM_WEBAPP_TYPE_AGENT_MESSAGE: &str = "aidaemon.telegram.agent_message.v1";
const TELEGRAM_WEBAPP_TYPE_CONTINUE_COMPUTER: &str = "aidaemon.telegram.open_on_computer.v1";
const TELEGRAM_WEBAPP_MAX_TEXT_CHARS: usize = 2_000;
/// What a Mini App payload asks for.
#[derive(Debug)]
pub enum TelegramWebAppAction {
    AgentMessage(String),
    ContinueOnComputer { relay_session_id: Option<String> },
}

/// All commands available in the Telegram channel (shared + Telegram-specific).
///
/// Used to register Telegram's `/` command menu via `setMyCommands` and to
/// generate `/help` output.
fn telegram_commands(host: &dyn TelegramHost) -> Vec<CommandDef> {
    let mut cmds = shared_commands();
    cmds.extend(host.commands());
    cmds.push(CommandDef {
        name: "help",
        description: "Show available commands",
        usage: None,
        category: CommandCategory::Core,
    });
    cmds
}

impl TelegramBot {
    pub fn new(
        bot_token: &str,
        allowed_user_ids: Vec<u64>,
        webhook: TelegramWebhookSettings,
        owner_user_ids: Vec<u64>,
        deps: BotDeps,
        host: Arc<dyn TelegramHost>,
    ) -> Self {
        let adapter = Arc::new(crate::TelegramChannel::new(bot_token.to_string()));
        let bot = Bot::new(adapter.clone());
        let runner = TurnRunner::new(deps.turn_deps());
        Self {
            runner,
            bot_username: StdRwLock::new("telegram".to_string()),
            cached_channel_name: StdRwLock::new("telegram".to_string()),
            session_namespace: StdRwLock::new(None),
            bot,
            bot_token: bot_token.to_string(),
            webhook,
            allowed_user_ids: StdRwLock::new(allowed_user_ids),
            owner_user_ids,
            deps,
            host,
            channel_hub: StdRwLock::new(None),
            api_url: None,
            transport: StdRwLock::new(adapter),
            prompts: Prompts::default(),
            pending_keys: Mutex::new(HashMap::new()),
        }
    }

    /// The Bot API client.
    pub fn bot(&self) -> &Bot {
        &self.bot
    }

    /// The bot's token.
    pub fn bot_token(&self) -> &str {
        &self.bot_token
    }

    /// Who may talk to the bot.
    pub fn allowed_user_ids(&self) -> &StdRwLock<Vec<u64>> {
        &self.allowed_user_ids
    }

    /// Who counts as an owner (empty: every allowed user).
    pub fn owner_user_ids(&self) -> &[u64] {
        &self.owner_user_ids
    }

    /// The hub, once wired.
    pub fn channel_hub(&self) -> Option<Arc<ChannelHub>> {
        self.channel_hub
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|weak| weak.upgrade()))
    }

    fn sdk_transport(&self) -> Arc<crate::TelegramChannel> {
        self.transport
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// The chat a session's prompts go to (the first allowed user's private
    /// chat when the session names none).
    fn conversation_for(&self, session_id: &str) -> aidaemon_channels::Conversation {
        let chat_id: i64 = aidaemon_events::session::telegram_chat_id_from_session(session_id)
            .unwrap_or_else(|| {
                self.allowed_user_ids
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .first()
                    .copied()
                    .unwrap_or(0) as i64
            });
        aidaemon_channels::Conversation {
            id: chat_id.to_string(),
            kind: if chat_id > 0 {
                aidaemon_channels::ConversationKind::Direct
            } else {
                aidaemon_channels::ConversationKind::Group
            },
        }
    }

    /// Set the channel hub reference for dynamic bot registration.
    /// Called after the hub is constructed in core.rs.
    pub fn set_channel_hub(&self, hub: Weak<ChannelHub>) {
        if let Ok(mut guard) = self.channel_hub.write() {
            *guard = Some(hub);
        }
    }

    /// Send a Markdown reply, folded into an expandable quote when long.
    pub async fn send_compact_or_full_reply(
        &self,
        bot: &Bot,
        chat_id: ChatId,
        markdown: &str,
    ) -> anyhow::Result<()> {
        send_full_or_expandable_reply(bot, chat_id, markdown)
            .await
            .map(|_| ())
    }
    /// Get the bot's username, fetching from Telegram API if not cached.
    pub async fn get_bot_username(&self) -> String {
        // Check if already fetched (not the default "telegram" placeholder)
        {
            let guard = self
                .bot_username
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if *guard != "telegram" {
                return guard.clone();
            }
        }

        // Fetch from Telegram API
        match self.bot.get_me().await {
            Ok(me) => {
                let username = me
                    .username
                    .clone()
                    .unwrap_or_else(|| "telegram".to_string());
                // Update both bot_username and cached_channel_name
                if let Ok(mut guard) = self.bot_username.write() {
                    *guard = username.clone();
                }
                // Channel name is "telegram" for single bot or "telegram:{username}" for multi-bot
                let channel_name = if username == "telegram" {
                    "telegram".to_string()
                } else {
                    format!("telegram:{}", username)
                };
                if let Ok(mut guard) = self.cached_channel_name.write() {
                    *guard = channel_name;
                }
                info!(username = %username, "Fetched bot username from Telegram");
                username
            }
            Err(e) => {
                warn!("Failed to fetch bot username: {}, using 'telegram'", e);
                "telegram".to_string()
            }
        }
    }

    /// The bot's username once known.
    pub fn cached_bot_label(&self) -> Option<String> {
        let guard = self
            .bot_username
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let username = guard.trim();
        if username.is_empty() || username == "telegram" {
            None
        } else {
            Some(username.to_string())
        }
    }

    /// Build the session ID for a chat, prefixing with bot username.
    /// Uses a stable namespace to avoid mid-run session ID drift.
    pub async fn session_id(&self, chat_id: i64) -> String {
        let namespace = self.session_namespace().await;
        if namespace == "default" {
            chat_id.to_string()
        } else {
            format!("{}:{}", namespace, chat_id)
        }
    }

    /// The stable prefix of this bot's session ids.
    pub async fn session_namespace(&self) -> String {
        {
            let guard = self
                .session_namespace
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(existing) = guard.as_ref() {
                return existing.clone();
            }
        }

        let username = self.get_bot_username().await;
        let namespace = if username == "telegram" || username == "default" {
            fallback_session_namespace_from_token(&self.bot_token)
        } else {
            username
        };

        let mut guard = self
            .session_namespace
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.is_none() {
            *guard = Some(namespace.clone());
        }
        guard.as_ref().cloned().unwrap_or(namespace)
    }

    /// Get the channel identifier for the session map.
    pub async fn channel_name(&self) -> String {
        let username = self.get_bot_username().await;
        if username == "default" {
            "telegram".to_string()
        } else {
            format!("telegram:{}", username)
        }
    }

    /// Start the Telegram dispatcher with automatic retry on crash.
    /// Uses exponential backoff: 5s → 10s → 20s → 40s → 60s cap.
    /// Resets backoff to initial after a stable run (60s+).
    pub async fn start_with_retry(self: Arc<Self>) {
        // Fetch bot username once at startup
        let bot_username = self.get_bot_username().await;

        let mut backoff = aidaemon_tools::backoff::ChannelReconnectBackoff::new(
            Duration::from_secs(5),
            Duration::from_secs(60),
            Duration::from_secs(60),
        );

        loop {
            info!(name = %bot_username, "Starting Telegram dispatcher");
            let started = tokio::time::Instant::now();
            self.clone().start().await;
            let ran_for = started.elapsed();

            let delay = backoff.next_jittered_delay(ran_for, 0.10);

            warn!(
                name = %bot_username,
                backoff_secs = delay.as_secs(),
                ran_for_secs = ran_for.as_secs(),
                "Telegram dispatcher stopped, restarting"
            );
            tokio::time::sleep(delay).await;
        }
    }

    pub async fn start(self: Arc<Self>) {
        let bot_username = self.get_bot_username().await;
        info!(name = %bot_username, "Starting Telegram channel");
        self.host.on_start(&self, &bot_username).await;

        // Register commands with Telegram so they appear in the "/" menu.
        let bot_commands: Vec<BotCommand> = telegram_commands(self.host.as_ref())
            .iter()
            .map(|c| BotCommand::new(c.name, c.description))
            .collect();
        if let Err(e) = self.bot.set_my_commands(bot_commands).await {
            warn!(error = %e, "Failed to register bot commands with Telegram");
        }

        self.start_sdk(&bot_username).await;
    }

    /// The SDK adapter receives updates (webhook or long polling, offsets,
    /// backoff, the webhook fallback) and hands each one over as the Bot API
    /// sent it, read here as `telegram_api` types so the handlers below are
    /// unchanged. Each chat's updates are handled in order, one at a time, as
    /// a dispatcher would; different chats run concurrently.
    async fn start_sdk(self: Arc<Self>, bot_username: &str) {
        let mut transport = crate::TelegramChannel::new(self.bot_token.clone())
            .bot_username(bot_username)
            .poll_timeout_secs(30);
        if let Some(api_url) = &self.api_url {
            transport = transport.api_base(api_url.clone()).poll_timeout_secs(1);
        }
        if let Some(webhook) = self.build_webhook_options(bot_username) {
            transport = transport.with_webhook(webhook);
        } else {
            // A webhook left from an earlier setup would starve long polling.
            info!(name = %bot_username, "Clearing any existing webhook before starting long polling");
            if let Err(err) = self.bot.delete_webhook().await {
                warn!(
                    name = %bot_username,
                    error = %err,
                    "Failed to delete webhook before long polling (updates may not arrive)"
                );
            }
        }
        let (updates_tx, mut updates) = mpsc::channel::<serde_json::Value>(64);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let runner = {
            let shutdown = shutdown.clone();
            let transport = Arc::new(transport);
            if let Ok(mut slot) = self.transport.write() {
                *slot = transport.clone();
            }
            tokio::spawn(async move { transport.run_updates(updates_tx, shutdown).await })
        };
        info!(name = %bot_username, "Telegram updates received by the SDK adapter");
        // One worker per chat keeps that chat's updates in order.
        let mut chats: HashMap<i64, mpsc::UnboundedSender<Update>> = HashMap::new();
        while let Some(raw) = updates.recv().await {
            let update = match serde_json::from_value::<Update>(raw) {
                Ok(update) => update,
                Err(error) => {
                    warn!(name = %bot_username, %error, "Unreadable Telegram update dropped");
                    continue;
                }
            };
            let chat = update.chat_id().map(|chat| chat.0).unwrap_or_default();
            let mut update = Some(update);
            if let Some(worker) = chats.get(&chat) {
                match worker.send(update.take().expect("set above")) {
                    Ok(()) => continue,
                    // The worker ended; start a new one below.
                    Err(returned) => update = Some(returned.0),
                }
            }
            let (worker_tx, mut worker_rx) = mpsc::unbounded_channel();
            let channel = Arc::clone(&self);
            tokio::spawn(async move {
                while let Some(update) = worker_rx.recv().await {
                    channel.dispatch_update(update).await;
                }
            });
            let _ = worker_tx.send(update.take().expect("set above"));
            chats.insert(chat, worker_tx);
        }
        shutdown.cancel();
        match runner.await {
            Ok(Err(e)) => warn!(name = %bot_username, error = %e, "Telegram updates stopped"),
            Err(e) => warn!(name = %bot_username, error = %e, "Telegram updates task failed"),
            Ok(Ok(())) => {}
        }
    }

    /// Talk to this Bot API server instead of Telegram's (tests, a local
    /// Bot API server).
    pub fn with_api_url(mut self, api_url: &str) -> Self {
        let adapter = Arc::new(
            crate::TelegramChannel::new(self.bot_token.clone()).api_base(api_url.to_string()),
        );
        self.bot = Bot::new(adapter.clone());
        self.api_url = Some(api_url.to_string());
        self.transport = StdRwLock::new(adapter);
        self
    }

    /// Run the handler for one update.
    async fn dispatch_update(self: &Arc<Self>, update: Update) {
        if let Some(msg) = update.message {
            self.handle_message(msg, self.bot.clone()).await;
        } else if let Some(query) = update.callback_query {
            self.handle_callback(query, self.bot.clone()).await;
        }
    }

    fn build_webhook_options(&self, bot_username: &str) -> Option<crate::WebhookConfig> {
        if !self.webhook.enabled {
            return None;
        }

        let public_url_raw = self
            .webhook
            .public_url
            .as_ref()
            .map(|v| v.trim())
            .filter(|v| !v.is_empty());
        let Some(public_url_raw) = public_url_raw else {
            warn!(
                name = %bot_username,
                "Telegram webhook enabled but `public_url` is empty; using long polling"
            );
            return None;
        };
        let public_url = match reqwest::Url::parse(public_url_raw) {
            Ok(value) => value,
            Err(err) => {
                warn!(
                    name = %bot_username,
                    error = %err,
                    public_url = %public_url_raw,
                    "Invalid Telegram webhook public URL; using long polling"
                );
                return None;
            }
        };
        if public_url.scheme() != "https" {
            warn!(
                name = %bot_username,
                public_url = %public_url_raw,
                "Telegram webhook public URL must use HTTPS; using long polling"
            );
            return None;
        }

        let listen_addr_raw = self
            .webhook
            .listen_addr
            .as_ref()
            .map(|v| v.trim())
            .filter(|v| !v.is_empty());
        let Some(listen_addr_raw) = listen_addr_raw else {
            warn!(
                name = %bot_username,
                "Telegram webhook enabled but `listen_addr` is empty; using long polling"
            );
            return None;
        };
        let listen_addr = match listen_addr_raw.parse::<std::net::SocketAddr>() {
            Ok(value) => value,
            Err(err) => {
                warn!(
                    name = %bot_username,
                    listen_addr = %listen_addr_raw,
                    error = %err,
                    "Invalid Telegram webhook listen address; using long polling"
                );
                return None;
            }
        };

        // The URL goes to setWebhook verbatim and the listener serves its
        // path, so a configured path is part of the URL.
        let path = self
            .webhook
            .path
            .as_ref()
            .map(|v| v.trim())
            .filter(|v| !v.is_empty())
            .map(|v| v.to_string());
        let webhook_url = if let Some(ref path) = path {
            let current_path = public_url.path().trim_end_matches('/');
            // Only append if the URL doesn't already include the path
            if current_path.is_empty() || current_path == "/" {
                let mut url = public_url.clone();
                url.set_path(path);
                url
            } else {
                public_url
            }
        } else {
            public_url
        };
        // The adapter generates the secret Telegram echoes back.
        let mut opts = crate::WebhookConfig::new(listen_addr, webhook_url.to_string());
        if let Some(max_connections) = self.webhook.max_connections {
            if (1..=100).contains(&max_connections) {
                opts.max_connections = Some(max_connections);
            } else {
                warn!(
                    name = %bot_username,
                    max_connections,
                    "Telegram webhook `max_connections` must be 1..=100; ignoring"
                );
            }
        }
        opts.drop_pending_updates = self.webhook.drop_pending_updates;
        Some(opts)
    }

    /// How long a resolved approval card stays visible before it is auto-deleted,
    /// so an answered prompt doesn't linger and get mistaken for one that still
    /// needs action. On by default (10s); override with the
    /// `AIDAEMON_APPROVAL_DISMISS_SECS` env var, or set it to `0` to keep
    /// resolved cards in the chat.
    fn resolved_approval_dismiss_delay() -> Option<Duration> {
        let secs = std::env::var("AIDAEMON_APPROVAL_DISMISS_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(10);
        (secs > 0).then(|| Duration::from_secs(secs))
    }

    /// Handle callback query from inline keyboard buttons.
    async fn handle_callback(self: &Arc<Self>, q: CallbackQuery, bot: Bot) {
        // Authorization check: only allowed users can approve/deny commands.
        // Fail-closed: deny if no users configured or user not in list.
        let user_id = q.from.id.0;
        let is_authorized = {
            let allowed = self
                .allowed_user_ids
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            !allowed.is_empty() && allowed.contains(&user_id)
        };
        if !is_authorized {
            warn!(user_id, "Unauthorized callback from user");
            let _ = bot
                .answer_callback_query(q.id)
                .text(format!("Unauthorized. Your ID: {}", user_id))
                .await;
            return;
        }

        let data = match q.data {
            Some(ref d) => d.clone(),
            None => return,
        };

        if self.host.handle_callback(self, &q, &data, user_id).await {
            return;
        }

        if let Some(choice) = data.strip_prefix("setkey:") {
            self.handle_setkey_choice(&q, choice, user_id, &bot).await;
            return;
        }

        let Some((prompt_id, choice_id)) = self.sdk_transport().approval_choice(&data) else {
            let _ = bot
                .answer_callback_query(q.id)
                .text("This action is no longer valid. Please run the command again.")
                .show_alert(true)
                .await;
            return;
        };
        let answered = if self.prompts.is_open(&prompt_id).await {
            self.prompts.answer(&prompt_id, &choice_id).await
        } else {
            None
        };
        let Some(answered) = answered else {
            warn!(prompt = %prompt_id, "Stale approval callback (no pending request)");
            let _ = bot
                .answer_callback_query(q.id.clone())
                .text("Approval expired. Please run the command again.")
                .show_alert(true)
                .await;
            if let Some(m) = q.message {
                let original = m.text().unwrap_or("");
                if !original.contains("Approval expired") {
                    let _ = bot
                        .edit_message_text(
                            m.chat.id,
                            m.id,
                            format!(
                                "{}\n\n⚠️ Approval expired or daemon restarted. Run the command again to create a new approval prompt.",
                                original
                            ),
                        )
                        .await;
                }
            }
            return;
        };
        let _ = bot.answer_callback_query(q.id).text(&answered.label).await;
        if let Some((message, text)) = answered.update {
            let transport = self.sdk_transport();
            if let Err(e) = transport.edit(&message, &text).await {
                warn!("Failed to update a Telegram approval message: {}", e);
            }
            // Remove an answered approval after a moment, so it isn't
            // mistaken for one still waiting.
            if answered.approval {
                if let Some(delay) = Self::resolved_approval_dismiss_delay() {
                    tokio::spawn(async move {
                        tokio::time::sleep(delay).await;
                        let _ = transport.delete(&message).await;
                    });
                }
            }
        }
    }

    async fn handle_command(self: &Arc<Self>, text: &str, msg: &Message, bot: &Bot) {
        let parts: Vec<&str> = text.splitn(2, ' ').collect();
        let cmd_raw = parts[0];
        let cmd = cmd_raw.split('@').next().unwrap_or(cmd_raw);
        let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

        if cmd == "/setkey" && self.host.commands().iter().any(|c| c.name == "setkey") {
            self.handle_setkey_command(msg, bot).await;
            return;
        }

        // The host's own commands, then the shared ones.
        let reply = match self.host.handle_command(self, cmd, arg, msg).await {
            HostCommand::Handled => return,
            HostCommand::Reply(reply) => Some(reply),
            HostCommand::NotMine => {
                let session_id = self.session_id(msg.chat.id.0).await;
                let user_id = msg.from.as_ref().map(|user| user.id.0).unwrap_or(0);
                let user_role = determine_role(&self.owner_user_ids, user_id);
                let sender_id = msg.from.as_ref().map(|u| format!("telegram:{}", u.id.0));
                self.host
                    .dispatch_shared(cmd, arg, &session_id, sender_id.as_deref(), user_role)
                    .await
            }
        };
        let reply = match reply {
            Some(reply) => reply,
            None if cmd == "/help" || cmd == "/start" => {
                let help = build_help_text(
                    &telegram_commands(self.host.as_ref()),
                    "/",
                    aidaemon_channels::slash::HelpScope::from_args(arg),
                );
                send_markdown_chunks_or_fallback(bot, msg.chat.id, &help).await;
                return;
            }
            None => aidaemon_channels::slash::unknown_command_reply(
                cmd_raw,
                &telegram_commands(self.host.as_ref()),
                "/",
            ),
        };

        for chunk in split_telegram_text(&reply, 4096) {
            let _ = bot.send_message(msg.chat.id, chunk).await;
        }
    }

    /// `/setkey`: owner only and in a private chat, where a pasted key is not
    /// visible to anyone else before it is deleted.
    async fn handle_setkey_command(&self, msg: &Message, bot: &Bot) {
        let user_id = msg.from.as_ref().map(|user| user.id.0).unwrap_or(0);
        if determine_role(&self.owner_user_ids, user_id) != UserRole::Owner {
            let _ = bot
                .send_message(msg.chat.id, "Only the owner can change API keys.")
                .await;
            return;
        }
        if !msg.chat.is_private() {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    "Use /setkey in a private chat with the bot, so the key is not shown to this group.",
                )
                .await;
            return;
        }
        self.pending_keys
            .lock()
            .await
            .remove(&(msg.chat.id.0, user_id));
        let targets = match self.host.key_choices().await {
            Ok(targets) => targets,
            Err(error) => {
                let _ = bot.send_message(msg.chat.id, error).await;
                return;
            }
        };
        if targets.is_empty() {
            let _ = bot
                .send_message(msg.chat.id, "No configured provider uses an API key.")
                .await;
            return;
        }
        let mut rows: Vec<Vec<InlineKeyboardButton>> = targets
            .iter()
            .map(|target| {
                vec![InlineKeyboardButton::callback(
                    target.label.clone(),
                    format!("setkey:{}", target.id),
                )]
            })
            .collect();
        rows.push(vec![InlineKeyboardButton::callback(
            "Cancel",
            "setkey:cancel",
        )]);
        let _ = bot
            .send_message(
                msg.chat.id,
                "Which provider's API key do you want to replace?",
            )
            .reply_markup(InlineKeyboardMarkup::new(rows))
            .await;
    }

    async fn handle_setkey_choice(
        &self,
        query: &CallbackQuery,
        choice: &str,
        user_id: u64,
        bot: &Bot,
    ) {
        let Some(message) = query
            .message
            .as_ref()
            .filter(|message| message.chat.is_private())
            .filter(|_| determine_role(&self.owner_user_ids, user_id) == UserRole::Owner)
        else {
            let _ = bot
                .answer_callback_query(query.id.clone())
                .text("Only the owner can do this, in a private chat.")
                .await;
            return;
        };
        let key = (message.chat.id.0, user_id);
        let target = self
            .host
            .key_choices()
            .await
            .ok()
            .and_then(|targets| targets.into_iter().find(|target| target.id == choice));
        let text = match target {
            Some(target) => {
                let label = target.label.clone();
                self.pending_keys.lock().await.insert(
                    key,
                    PendingKey {
                        id: target.id,
                        label: label.clone(),
                        expires_at: std::time::Instant::now() + PENDING_KEY_TTL,
                    },
                );
                format!(
                    "{label}: paste the new API key as your next message. I'll delete it right away and test it before saving. Send /cancel to stop. This expires in 2 minutes."
                )
            }
            None => {
                self.pending_keys.lock().await.remove(&key);
                if choice == "cancel" {
                    "Key entry cancelled.".to_string()
                } else {
                    "That provider is no longer in the config. Run /setkey again.".to_string()
                }
            }
        };
        let _ = bot.answer_callback_query(query.id.clone()).await;
        let _ = bot
            .edit_message_text(message.chat.id, message.id, text)
            .await;
    }

    /// Consume the owner's message when a key is expected. `true` means it
    /// was the key (or `/cancel`) and must go no further.
    async fn capture_pending_key(&self, msg: &Message, user_id: u64, bot: &Bot) -> bool {
        let pending = {
            let mut pending_keys = self.pending_keys.lock().await;
            if pending_keys.is_empty() {
                return false;
            }
            pending_keys.retain(|_, pending| pending.expires_at > std::time::Instant::now());
            pending_keys.remove(&(msg.chat.id.0, user_id))
        };
        let Some(pending) = pending else {
            return false;
        };
        let text = msg.text().map(str::trim).unwrap_or_default();
        if text.is_empty() || text.starts_with('/') {
            let _ = bot.send_message(msg.chat.id, "Key entry cancelled.").await;
            return text.split('@').next() == Some("/cancel");
        }
        let delete_note = match bot.delete_message(msg.chat.id, msg.id).await {
            Ok(_) => "",
            Err(_) => "\n\nI couldn't delete your message, so please delete it yourself.",
        };
        let reply = self
            .host
            .install_key(&pending.id, &pending.label, text)
            .await;
        let _ = bot
            .send_message(msg.chat.id, format!("{reply}{delete_note}"))
            .await;
        true
    }

    /// Handle an incoming file/photo/audio/video/voice message.
    /// Downloads the file, saves to inbox, and returns attachment metadata.
    async fn handle_file_message(&self, msg: &Message) -> anyhow::Result<MessageAttachment> {
        // The file, its size, and the name and type it is saved under.
        let (file, filename, mime_type) = if let Some(doc) = msg.document() {
            (
                doc,
                doc.file_name.as_deref().unwrap_or("document"),
                doc.mime_type
                    .as_deref()
                    .unwrap_or("application/octet-stream"),
            )
        } else if let Some(photo) = msg.photo().and_then(|sizes| sizes.last()) {
            // The last size is the largest.
            (photo, "photo.jpg", "image/jpeg")
        } else if let Some(audio) = msg.audio() {
            (
                audio,
                audio.file_name.as_deref().unwrap_or("audio.mp3"),
                audio.mime_type.as_deref().unwrap_or("audio/mpeg"),
            )
        } else if let Some(video) = msg.video() {
            (
                video,
                video.file_name.as_deref().unwrap_or("video.mp4"),
                video.mime_type.as_deref().unwrap_or("video/mp4"),
            )
        } else if let Some(voice) = msg.voice() {
            (
                voice,
                "voice.ogg",
                voice.mime_type.as_deref().unwrap_or("audio/ogg"),
            )
        } else {
            anyhow::bail!("Unsupported message type. I can process text, files, photos, audio, video, and voice messages.");
        };
        let (filename, mime_type) = (filename.to_string(), mime_type.to_string());

        // Check file size before downloading
        let max_bytes = self.deps.max_file_size_mb * 1_048_576;
        let file_size = file.file_size.unwrap_or(0);
        if file_size > max_bytes {
            anyhow::bail!(
                "File too large ({:.1} MB). Maximum is {} MB.",
                file_size as f64 / 1_048_576.0,
                self.deps.max_file_size_mb
            );
        }

        let bytes = self
            .sdk_transport()
            .download(
                &aidaemon_channels::Attachment::new(filename.clone(), file.file_id.clone())
                    .with_mime_type(Some(mime_type.clone()))
                    .with_size_bytes(file.file_size),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to download file from Telegram: {e}"))?;
        if bytes.len() as u64 > max_bytes {
            anyhow::bail!(
                "File too large ({:.1} MB). Maximum is {} MB.",
                bytes.len() as f64 / 1_048_576.0,
                self.deps.max_file_size_mb
            );
        }

        let (filename, mime_type) = if msg.photo().is_some() {
            aidaemon_channels::attachments::telegram_photo_filename_and_mime(&bytes)
        } else {
            (filename, mime_type)
        };

        // Sanitize filename: strip path separators, null bytes, limit length
        let sanitized = aidaemon_events::attachments::sanitize_filename(&filename);
        let uuid_prefix = uuid::Uuid::new_v4().to_string()[..8].to_string();
        let dest_name = format!("{}_{}", uuid_prefix, sanitized);
        let dest_path = self.deps.inbox_dir.join(&dest_name);

        // Ensure inbox directory exists
        std::fs::create_dir_all(&self.deps.inbox_dir)?;

        // Write file
        std::fs::write(&dest_path, &bytes)?;

        info!(
            file = %dest_path.display(),
            size = bytes.len(),
            mime = %mime_type,
            "Saved inbound file"
        );

        Ok(aidaemon_channels::attachments::message_attachment(
            dest_path,
            sanitized,
            mime_type,
            bytes.len() as u64,
        ))
    }

    /// Read a Mini App payload.
    pub fn parse_web_app_action(raw: &str) -> Option<TelegramWebAppAction> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }

        let parsed: serde_json::Value = serde_json::from_str(trimmed).ok()?;
        Self::parse_web_app_action_value(&parsed)
    }

    fn parse_web_app_action_value(value: &serde_json::Value) -> Option<TelegramWebAppAction> {
        match value {
            serde_json::Value::String(inner) => {
                let trimmed = inner.trim();
                if trimmed.is_empty() {
                    return None;
                }
                let nested: serde_json::Value = serde_json::from_str(trimmed).ok()?;
                Self::parse_web_app_action_value(&nested)
            }
            serde_json::Value::Array(items) => {
                items.iter().find_map(Self::parse_web_app_action_value)
            }
            serde_json::Value::Object(map) => {
                let action_type = map
                    .get("type")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .unwrap_or("");

                let relay_session_id = map
                    .get("relay_session_id")
                    .or_else(|| map.get("relaySessionId"))
                    .or_else(|| map.get("relay_session"))
                    .or_else(|| map.get("relaySession"))
                    .or_else(|| map.get("session"))
                    .or_else(|| map.get("session_id"))
                    .or_else(|| map.get("sessionId"))
                    .or_else(|| map.get("sid"))
                    .and_then(|v| {
                        if let Some(raw) = v.as_str() {
                            let trimmed = raw.trim();
                            if trimmed.is_empty() {
                                None
                            } else {
                                Some(trimmed.to_string())
                            }
                        } else if let Some(obj) = v.as_object() {
                            obj.get("id")
                                .or_else(|| obj.get("session_id"))
                                .or_else(|| obj.get("sessionId"))
                                .and_then(|inner| inner.as_str())
                                .map(str::trim)
                                .filter(|inner| !inner.is_empty())
                                .map(|inner| inner.to_string())
                        } else {
                            None
                        }
                    });

                if action_type == TELEGRAM_WEBAPP_TYPE_CONTINUE_COMPUTER
                    || action_type == "aidaemon.telegram.continue_on_computer.v1"
                {
                    return Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id });
                }
                if action_type == TELEGRAM_WEBAPP_TYPE_AGENT_MESSAGE {
                    let text = map
                        .get("text")
                        .and_then(|v| v.as_str())
                        .map(str::trim)
                        .unwrap_or("");
                    if text.is_empty() {
                        return None;
                    }
                    let clipped = text.chars().take(TELEGRAM_WEBAPP_MAX_TEXT_CHARS).collect();
                    return Some(TelegramWebAppAction::AgentMessage(clipped));
                }

                // Some Telegram clients/wrappers may nest the actual payload under another key.
                for key in ["data", "payload", "message", "web_app_data", "webAppData"] {
                    if let Some(nested) = map.get(key) {
                        if let Some(action) = Self::parse_web_app_action_value(nested) {
                            return Some(action);
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }
    async fn handle_message(self: &Arc<Self>, msg: Message, bot: Bot) {
        let transport_received_at = chrono::Utc::now();
        let ingress_timing = InboundMessageTiming::new(Some(msg.date), transport_received_at);
        let user_id = msg.from.as_ref().map(|u| u.id.0).unwrap_or(0);

        // Authorization check with first-user auto-claim (DM only).
        // When allowed_user_ids is empty, the FIRST user to send a private
        // message becomes the owner. Group messages are rejected until an
        // owner is established, preventing someone from claiming ownership
        // by adding the bot to a group.
        let is_private = msg.chat.is_private();
        let auth_result = {
            let allowed = self
                .allowed_user_ids
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if allowed.is_empty() {
                if is_private {
                    warn!(
                        user_id,
                        "No allowed_user_ids configured — auto-claiming first DM user as owner."
                    );
                    drop(allowed);
                    let mut allowed = self
                        .allowed_user_ids
                        .write()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    check_auth(&mut allowed, user_id)
                } else {
                    // Group message before any owner is set — reject
                    AuthResult::Unauthorized
                }
            } else if allowed.contains(&user_id) {
                AuthResult::Authorized
            } else {
                AuthResult::Unauthorized
            }
        };

        match auth_result {
            AuthResult::AutoClaimed => {
                // The host keeps it (config.toml) so it survives restarts.
                let ids = self
                    .allowed_user_ids
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                let note = self.host.owner_claimed(self, &ids).await;
                let mut welcome = "Hey! You're now set as the owner. Ask me anything, give me tasks, or just chat."
                    .to_string();
                if let Some(note) = note {
                    welcome.push_str("\n\n");
                    welcome.push_str(&note);
                }
                let _ = bot.send_message(msg.chat.id, welcome).await;
                // Fall through to process the message normally
            }
            AuthResult::Unauthorized => {
                warn!(user_id, "Unauthorized user attempted access");
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!(
                            "Unauthorized. Your Telegram user ID is <code>{}</code>.\n\n\
                             To grant access, add it to <code>allowed_user_ids</code> in config.toml:\n\
                             <pre>[telegram]\nallowed_user_ids = [{}]</pre>",
                            user_id, user_id
                        ),
                    )
                    .parse_mode(ParseMode::Html)
                    .await;
                return;
            }
            AuthResult::Authorized => {} // continue
        }

        // A key being entered must reach nothing else: no agent, no event log.
        if self.capture_pending_key(&msg, user_id, &bot).await {
            return;
        }

        let user_role = determine_role(&self.owner_user_ids, user_id);

        let mut inbound_attachments = Vec::new();
        let mut body_text = String::new();

        if let Some(web_app_data) = msg.web_app_data() {
            match Self::parse_web_app_action(&web_app_data.data) {
                Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                    self.host
                        .continue_on_computer(self, &msg, user_id, relay_session_id)
                        .await;
                    return;
                }
                Some(TelegramWebAppAction::AgentMessage(text)) => body_text = text,
                None => {
                    let _ = bot
                        .send_message(
                            msg.chat.id,
                            "Mini App payload was not recognized. Try opening a new terminal session with /agent open.",
                        )
                        .await;
                    return;
                }
            }
        } else if let Some(t) = msg.text() {
            if let Some(action) = Self::parse_web_app_action(t) {
                match action {
                    TelegramWebAppAction::ContinueOnComputer { relay_session_id } => {
                        self.host
                            .continue_on_computer(self, &msg, user_id, relay_session_id)
                            .await;
                        return;
                    }
                    TelegramWebAppAction::AgentMessage(text) => body_text = text,
                }
            } else {
                body_text = t.to_string();
            }
        } else if !self.deps.files_enabled {
            let _ = bot
                .send_message(msg.chat.id, "I can only process text messages.")
                .await;
            return;
        }

        if self.deps.files_enabled
            && (msg.document().is_some()
                || msg.photo().is_some()
                || msg.audio().is_some()
                || msg.video().is_some()
                || msg.voice().is_some())
        {
            match self.handle_file_message(&msg).await {
                Ok(attachment) => inbound_attachments.push(attachment),
                Err(e) => {
                    let _ = bot
                        .send_message(msg.chat.id, format!("File error: {}", e))
                        .await;
                    return;
                }
            }
            if body_text.is_empty() {
                if let Some(caption) = msg.caption() {
                    body_text = caption.to_string();
                }
            }
        } else if body_text.is_empty() {
            let _ = bot
                .send_message(msg.chat.id, "I can only process text messages.")
                .await;
            return;
        }

        let text = if user_role == UserRole::Owner {
            aidaemon_channels::attachments::build_inbound_text(&body_text, &inbound_attachments)
        } else {
            aidaemon_channels::attachments::build_inbound_text_without_local_paths(
                &body_text,
                &inbound_attachments,
            )
        };

        // Handle slash commands.
        // A command the host runs detached (`/setup`) is spawned in its own
        // task because it may call `request_approval`, which blocks waiting
        // for a Telegram callback query.  Teloxide dispatches updates per-chat sequentially,
        // so running it inline would deadlock: the callback can't be delivered
        // until the message handler returns, but the message handler is waiting
        // for that callback.
        // `/private` sent with its message ("/private what's my balance?",
        // "/privateWhat's…") keeps that message private and handles it now;
        // "/private/private" is just `/private`.
        let text = match aidaemon_channels::slash::private_request(&text) {
            Some(aidaemon_channels::slash::PrivateRequest::With(message)) => {
                let session_id = self.session_id(msg.chat.id.0).await;
                let sender_id = format!("telegram:{user_id}");
                if let Err(error) = self
                    .host
                    .set_next_message_private(&session_id, Some(&sender_id), true)
                    .await
                {
                    warn!(%error, "Could not keep a /private message out of memory");
                    let _ = bot
                        .send_message(
                            msg.chat.id,
                            "I couldn't keep that message private, so I didn't read it. \
                             Please send it again.",
                        )
                        .await;
                    return;
                }
                let _ = bot
                    .send_message(msg.chat.id, "🔒 I won't remember or learn from this one.")
                    .await;
                message.to_string()
            }
            Some(aidaemon_channels::slash::PrivateRequest::Next) => "/private".to_string(),
            None => text,
        };
        if aidaemon_channels::slash::is_slash_command(&text) {
            // Slash commands `return` before the message dedup gate below, so a
            // duplicate Telegram delivery of the same update (same message id)
            // would execute the command twice — e.g. two "Context cleared."
            // replies, a double `/cancel`. Apply the same msg.id-keyed dedup
            // here. Distinct sends (different message ids) are never deduped, so
            // intentional repeats still work.
            let cmd_session_id = self.session_id(msg.chat.id.0).await;
            let cmd_dedup_key = msg.id.0.to_string();
            if !self
                .deps
                .task_registry
                .mark_seen(&cmd_session_id, &text, Some(&cmd_dedup_key))
                .await
            {
                debug!(session_id = %cmd_session_id, "Dropped duplicate slash command");
                return;
            }
            let detached = {
                let cmd = text.split_whitespace().next().unwrap_or("");
                let cmd = cmd.split('@').next().unwrap_or(cmd);
                self.host.runs_detached(cmd)
            };
            if detached {
                let channel = Arc::clone(self);
                let text = text.clone();
                let msg = msg.clone();
                let bot = bot.clone();
                tokio::spawn(async move {
                    channel.handle_command(&text, &msg, &bot).await;
                });
            } else {
                self.handle_command(&text, &msg, &bot).await;
            }
            return;
        }

        if let Some(lite_reply) = self
            .host
            .intercept_text(self, msg.chat.id.0, user_id, user_role, &text)
            .await
        {
            for chunk in split_telegram_text(&lite_reply, 4096) {
                let _ = bot.send_message(msg.chat.id, chunk).await;
            }
            return;
        }

        // Use chat ID as session ID, prefixed with bot name if multi-bot
        let session_id = self.session_id(msg.chat.id.0).await;
        let channel_id = aidaemon_events::session::derive_channel_id_from_session(&session_id);

        // Build channel context from Telegram chat type
        let channel_ctx = {
            // A supergroup with a public username is public; a channel always.
            let visibility = match msg.chat.kind.as_str() {
                "private" => ChannelVisibility::Private,
                "supergroup" if msg.chat.username.is_some() => ChannelVisibility::Public,
                "channel" => ChannelVisibility::Public,
                _ => ChannelVisibility::PrivateGroup,
            };
            ChannelContext {
                visibility,
                platform: "telegram".to_string(),
                channel_name: msg.chat.title().map(|s| s.to_string()),
                channel_id,
                workspace_id: None,
                sender_name: msg.from.as_ref().map(|u| match &u.last_name {
                    Some(last) => format!("{} {}", u.first_name, last),
                    None => u.first_name.clone(),
                }),
                sender_id: msg.from.as_ref().map(|u| format!("telegram:{}", u.id.0)),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                workspace_grant: None,
                trusted: false,
            }
        };

        self.runner
            .register_session(&session_id, &self.channel_name().await)
            .await;
        info!(session_id, "Received message from user {}", user_id);

        // The hub backs the live status message. Startup wires it before
        // any traffic arrives, so its absence only degrades to plain status
        // messages.
        let hub = self
            .channel_hub
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|weak| weak.upgrade()));
        if hub.is_none() {
            warn!(
                session_id = %session_id,
                "Telegram status surface disabled: ChannelHub is not wired"
            );
        }
        // The turn's one status card, which the turn runner owns.
        let card = hub
            .as_ref()
            .map(|hub| Arc::new(HubSurfaceSink::new(hub.clone(), session_id.clone())));
        let surface = TelegramSurface {
            bot,
            chat_id: msg.chat.id,
            session_id: session_id.clone(),
            hub,
            card,
        };
        self.runner
            .submit(
                Arc::new(surface),
                Turn {
                    session_id,
                    text,
                    attachments: inbound_attachments,
                    user_role,
                    channel_ctx,
                    // `message_id` is unique per chat and reused on
                    // redeliveries; the same text typed again gets a new one.
                    dedup_key: Some(msg.id.0.to_string()),
                    ingress_timing: Some(ingress_timing),
                },
            )
            .await;
    }
}

/// One Telegram chat, as the turn runner shows it.
struct TelegramSurface {
    bot: Bot,
    chat_id: ChatId,
    session_id: String,
    hub: Option<Arc<ChannelHub>>,
    /// Where the turn's status card goes; `None` without the hub, when
    /// status goes out as plain messages.
    card: Option<Arc<HubSurfaceSink>>,
}

#[async_trait]
impl TurnSurface for TelegramSurface {
    fn platform(&self) -> &'static str {
        "telegram"
    }

    async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
        send_markdown_chunks_or_fallback_result(&self.bot, self.chat_id, text).await
    }

    async fn edit(&self, message_id: &str, text: &str) -> anyhow::Result<bool> {
        let Ok(mid) = message_id.parse::<i32>() else {
            return Ok(false);
        };
        Ok(self
            .bot
            .edit_message_text(self.chat_id, MessageId(mid), text)
            .link_preview_options(disabled_link_preview())
            .await
            .is_ok())
    }

    /// Telegram's typing indicator lasts about five seconds.
    fn show_working(
        &self,
        stop: tokio_util::sync::CancellationToken,
        heartbeat: Arc<AtomicU64>,
        stale_secs: u64,
    ) {
        let bot = self.bot.clone();
        let chat_id = self.chat_id;
        repeat_while_working(
            Duration::from_secs(3),
            stop,
            heartbeat,
            stale_secs,
            move || {
                let bot = bot.clone();
                async move {
                    let _ = bot.send_chat_action(chat_id, ChatAction::Typing).await;
                }
            },
        );
    }

    fn card_sink(&self) -> Option<Arc<dyn SurfaceSink>> {
        self.card.clone().map(|sink| sink as Arc<dyn SurfaceSink>)
    }

    /// A card that stays in the chat can still change after the turn: a
    /// background job settles the steps it was handed, a helper its header.
    async fn card_kept(&self, message_id: &str) {
        if let Some(hub) = &self.hub {
            hub.register_checklist_surface(&self.session_id, message_id)
                .await;
        }
    }

    /// Without the hub there is no card: plain status messages.
    fn show_status(
        &self,
        updates: mpsc::Receiver<StatusUpdate>,
        private: bool,
    ) -> Result<tokio::task::JoinHandle<()>, mpsc::Receiver<StatusUpdate>> {
        Ok(tokio::spawn(run_legacy_status_consumer(
            updates,
            self.bot.clone(),
            self.chat_id,
            private,
        )))
    }

    /// Final answers are new messages: a progress edit is not a delivery
    /// (Telegram can acknowledge editMessageText while a client delays
    /// showing it).
    async fn deliver(&self, _envelope: &AgentResponseEnvelope, reply: &str) -> Delivered {
        match send_full_or_expandable_reply(&self.bot, self.chat_id, reply).await {
            Ok(message_ids) => Delivered {
                state: ResponseDeliveryState::PlatformAcknowledged,
                message_ids,
                error: None,
            },
            Err(e) => {
                warn!("Failed to send Telegram message: {}", e);
                Delivered {
                    state: ResponseDeliveryState::Failed,
                    message_ids: Vec::new(),
                    error: Some("telegram_send_failed".to_string()),
                }
            }
        }
    }
}

fn fallback_session_namespace_from_token(bot_token: &str) -> String {
    let raw_id = bot_token
        .split_once(':')
        .map(|(id, _)| id)
        .unwrap_or(bot_token)
        .trim();
    let sanitized_id: String = raw_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    if sanitized_id.is_empty() {
        "telegram".to_string()
    } else {
        format!("tg{}", sanitized_id)
    }
}

#[async_trait]
impl HubChannel for TelegramBot {
    fn name(&self) -> String {
        self.cached_channel_name
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            markdown: true,
            inline_buttons: true,
            media: true,
            max_message_len: 4096,
        }
    }

    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
        let chat_id: i64 = aidaemon_events::session::telegram_chat_id_from_session(session_id)
            .unwrap_or_else(|| {
                self.allowed_user_ids
                    .read()
                    .unwrap()
                    .first()
                    .copied()
                    .unwrap_or(0) as i64
            });
        self.send_compact_or_full_reply(&self.bot, ChatId(chat_id), text)
            .await
    }

    async fn send_text_tracked(
        &self,
        session_id: &str,
        text: &str,
    ) -> anyhow::Result<Option<String>> {
        let chat_id: i64 = aidaemon_events::session::telegram_chat_id_from_session(session_id)
            .unwrap_or_else(|| {
                self.allowed_user_ids
                    .read()
                    .unwrap()
                    .first()
                    .copied()
                    .unwrap_or(0) as i64
            });
        // Render markdown → Telegram HTML so the live surface displays formatted
        // content (the final reply is edited into this message at turn end, and it
        // carries markdown). Fall back to a plain send if Telegram rejects the
        // HTML. The returned message id is captured for later in-place editing.
        let html = markdown_to_telegram_html(text);
        let msg = match self
            .bot
            .send_message(ChatId(chat_id), &html)
            .parse_mode(ParseMode::Html)
            .link_preview_options(disabled_link_preview())
            .await
        {
            Ok(m) => m,
            Err(_) => {
                self.bot
                    .send_message(ChatId(chat_id), text)
                    .link_preview_options(disabled_link_preview())
                    .await?
            }
        };
        Ok(Some(msg.id.0.to_string()))
    }

    async fn edit_text(
        &self,
        session_id: &str,
        message_id: &str,
        text: &str,
    ) -> anyhow::Result<bool> {
        let chat_id: i64 = aidaemon_events::session::telegram_chat_id_from_session(session_id)
            .unwrap_or_else(|| {
                self.allowed_user_ids
                    .read()
                    .unwrap()
                    .first()
                    .copied()
                    .unwrap_or(0) as i64
            });
        let mid: i32 = match message_id.parse() {
            Ok(v) => v,
            Err(_) => return Ok(false),
        };
        // Render markdown → Telegram HTML so an edited-in final reply displays
        // formatted (matches the normal send path). Fall back to a plain edit if
        // Telegram rejects the HTML (a rare conversion edge), so an edit is never
        // lost to a formatting error.
        let html = markdown_to_telegram_html(text);
        match self
            .bot
            .edit_message_text(ChatId(chat_id), MessageId(mid), &html)
            .parse_mode(ParseMode::Html)
            .link_preview_options(disabled_link_preview())
            .await
        {
            Ok(_) => Ok(true),
            // Telegram rejects an edit whose text is identical to the current
            // message — that's benign (nothing to change), so treat it as success.
            Err(e) if e.to_string().contains("message is not modified") => Ok(true),
            Err(_) => match self
                .bot
                .edit_message_text(ChatId(chat_id), MessageId(mid), text)
                .link_preview_options(disabled_link_preview())
                .await
            {
                Ok(_) => Ok(true),
                Err(e) if e.to_string().contains("message is not modified") => Ok(true),
                Err(e) => Err(e.into()),
            },
        }
    }

    async fn delete_message(&self, session_id: &str, message_id: &str) -> anyhow::Result<bool> {
        let chat_id: i64 = aidaemon_events::session::telegram_chat_id_from_session(session_id)
            .unwrap_or_else(|| {
                self.allowed_user_ids
                    .read()
                    .unwrap()
                    .first()
                    .copied()
                    .unwrap_or(0) as i64
            });
        let mid: i32 = match message_id.parse() {
            Ok(v) => v,
            Err(_) => return Ok(false),
        };
        self.bot
            .delete_message(ChatId(chat_id), MessageId(mid))
            .await
            .map(|_| true)
            .map_err(Into::into)
    }

    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        let chat_id: i64 = aidaemon_events::session::telegram_chat_id_from_session(session_id)
            .unwrap_or_else(|| {
                self.allowed_user_ids
                    .read()
                    .unwrap()
                    .first()
                    .copied()
                    .unwrap_or(0) as i64
            });
        let (name, bytes, mime_type) = match &media.kind {
            MediaKind::Photo { data } => (
                "screenshot.png".to_string(),
                data.clone(),
                Some("image/png".to_string()),
            ),
            MediaKind::Document {
                file_path,
                filename,
            } => {
                let bytes = tokio::fs::read(file_path).await?;
                let mime_type = outbound_content_type(&bytes);
                (filename.clone(), bytes, mime_type)
            }
        };
        let conversation = aidaemon_channels::Conversation {
            id: chat_id.to_string(),
            kind: if chat_id > 0 {
                aidaemon_channels::ConversationKind::Direct
            } else {
                aidaemon_channels::ConversationKind::Group
            },
        };
        self.sdk_transport()
            .send_file(
                &conversation,
                &aidaemon_channels::OutboundFile {
                    name,
                    bytes,
                    mime_type,
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
        info!(session_id, command, risk = %risk_level, mode = %permission_mode, "Approval requested");
        let transport = self.sdk_transport();
        self.prompts
            .approval(
                transport,
                &self.conversation_for(session_id),
                command,
                risk_level,
                warnings,
                permission_mode,
                one_time_only,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Telegram approval request failed: {e}"))
    }

    async fn request_approval_ask(
        &self,
        session_id: &str,
        ask: &aidaemon_channels::approval_render::ApprovalAsk<'_>,
    ) -> anyhow::Result<ApprovalResponse> {
        info!(session_id, card = ask.card.is_some(), risk = %ask.risk_level, "Approval requested");
        self.prompts
            .approval_ask(
                self.sdk_transport(),
                &self.conversation_for(session_id),
                ask,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Telegram approval request failed: {e}"))
    }

    async fn request_goal_confirmation(
        &self,
        session_id: &str,
        goal_description: &str,
        details: &[String],
        style: GoalConfirmationStyle,
    ) -> anyhow::Result<bool> {
        let transport = self.sdk_transport();
        self.prompts
            .goal(
                transport,
                &self.conversation_for(session_id),
                goal_description,
                details,
                style,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Telegram goal confirmation failed: {e}"))
    }
}

/// Send a message with HTML parse mode, falling back to plain text on failure.
async fn send_html_or_fallback(
    bot: &Bot,
    chat_id: ChatId,
    html: &str,
    plain: &str,
) -> Result<Message, RequestError> {
    match bot
        .send_message(chat_id, html)
        .parse_mode(ParseMode::Html)
        .link_preview_options(disabled_link_preview())
        .await
    {
        Ok(message) => Ok(message),
        Err(e) => {
            warn!("HTML send failed, falling back to plain text: {}", e);
            let message = bot
                .send_message(chat_id, plain)
                .link_preview_options(disabled_link_preview())
                .await?;
            Ok(message)
        }
    }
}

async fn send_markdown_chunks_or_fallback_result(
    bot: &Bot,
    chat_id: ChatId,
    markdown: &str,
) -> anyhow::Result<Vec<String>> {
    let html = markdown_to_telegram_html(markdown);
    let html_chunks = split_telegram_html(&html, 4096);
    let plain_chunks = split_telegram_text(&markdown_to_telegram_plain_fallback(markdown), 4096);
    let mut first_err: Option<anyhow::Error> = None;
    let mut message_ids = Vec::new();

    for (i, html_chunk) in html_chunks.iter().enumerate() {
        let plain_chunk = plain_chunks
            .get(i)
            .map(|s| s.as_str())
            .unwrap_or(html_chunk.as_str());
        match send_html_or_fallback(bot, chat_id, html_chunk, plain_chunk).await {
            Ok(message) => message_ids.push(message.id.to_string()),
            Err(e) => {
                warn!("Failed to send Telegram message: {}", e);
                if first_err.is_none() {
                    first_err = Some(anyhow::anyhow!("Failed to send Telegram message: {}", e));
                }
            }
        }
    }

    if let Some(err) = first_err {
        return Err(err);
    }
    Ok(message_ids)
}

/// Convert markdown to Telegram HTML and send with plain-text fallback.
pub async fn send_markdown_chunks_or_fallback(bot: &Bot, chat_id: ChatId, markdown: &str) {
    if let Err(e) = send_markdown_chunks_or_fallback_result(bot, chat_id, markdown).await {
        warn!("Failed to send Telegram message: {}", e);
    }
}

async fn send_full_or_expandable_reply(
    bot: &Bot,
    chat_id: ChatId,
    markdown: &str,
) -> anyhow::Result<Vec<String>> {
    let plain = markdown_to_telegram_plain_fallback(markdown);
    if plain.chars().count() > TELEGRAM_EXPANDABLE_TRIGGER_CHARS {
        return send_expandable_blockquote_reply(bot, chat_id, markdown).await;
    }
    send_markdown_chunks_or_fallback_result(bot, chat_id, markdown).await
}

async fn send_expandable_blockquote_reply(
    bot: &Bot,
    chat_id: ChatId,
    markdown: &str,
) -> anyhow::Result<Vec<String>> {
    let html_body = markdown_to_telegram_html(markdown);
    let plain = markdown_to_telegram_plain_fallback(markdown);
    let max_chunk = TELEGRAM_MAX_MESSAGE_LEN - TELEGRAM_EXPANDABLE_WRAPPER_LEN;
    let html_chunks = split_telegram_html(&html_body, max_chunk);
    let plain_chunks = split_telegram_text(&plain, max_chunk);
    let mut first_err: Option<anyhow::Error> = None;
    let mut message_ids = Vec::new();
    for (i, html_chunk) in html_chunks.iter().enumerate() {
        let wrapped = format!("<blockquote expandable>{}</blockquote>", html_chunk);
        let plain_chunk = plain_chunks
            .get(i)
            .map(|s| s.as_str())
            .unwrap_or(html_chunk.as_str());
        match send_html_or_fallback(bot, chat_id, &wrapped, plain_chunk).await {
            Ok(message) => message_ids.push(message.id.to_string()),
            Err(e) => {
                warn!("Failed to send expandable Telegram message: {}", e);
                if first_err.is_none() {
                    first_err = Some(anyhow::anyhow!("Failed to send Telegram message: {}", e));
                }
            }
        }
    }

    if let Some(err) = first_err {
        return Err(err);
    }
    Ok(message_ids)
}

/// Spawn a TelegramBot in a background task.
/// This is a separate function to avoid async type inference cycles.
pub fn spawn_telegram_channel(channel: Arc<TelegramBot>) {
    tokio::spawn(async move {
        channel.start_with_retry().await;
    });
}

/// Result of the authorization check for an incoming message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthResult {
    /// User is in the allow list — proceed normally.
    Authorized,
    /// No users were configured and this user was auto-claimed as owner.
    AutoClaimed,
    /// User is not allowed.
    Unauthorized,
}

/// Pure authorization check against the allow list with first-user auto-claim.
/// When `allowed` is empty, pushes `user_id` into it and returns `AutoClaimed`.
pub fn check_auth(allowed: &mut Vec<u64>, user_id: u64) -> AuthResult {
    if allowed.is_empty() {
        allowed.push(user_id);
        AuthResult::AutoClaimed
    } else if allowed.contains(&user_id) {
        AuthResult::Authorized
    } else {
        AuthResult::Unauthorized
    }
}

fn is_indicator_only_status(update: &StatusUpdate) -> bool {
    matches!(update, StatusUpdate::Thinking(_))
}

/// Legacy (pre-LiveStatus) status consumer used only when the `ChannelHub` is
/// not wired and the single live surface is unavailable. Sends a fresh message
/// per status update (the same plain text every channel's status line uses) with the historical
/// `dm_status_count`/`MAX_DM_STATUS_MESSAGES` spam cap. This is the defensive
/// fallback shared by both the primary and queue-drain sites.
async fn run_legacy_status_consumer(
    mut rx: tokio::sync::mpsc::Receiver<StatusUpdate>,
    bot: Bot,
    chat_id: ChatId,
    is_dm: bool,
) {
    let mut last_sent = tokio::time::Instant::now() - Duration::from_secs(10);
    let min_interval = Duration::from_secs(3);
    let mut dm_status_count: u32 = 0;
    const MAX_DM_STATUS_MESSAGES: u32 = 6;
    while let Some(update) = rx.recv().await {
        if is_indicator_only_status(&update) {
            let _ = bot.send_chat_action(chat_id, ChatAction::Typing).await;
            continue;
        }
        // In non-DM channels, suppress progress chatter.
        if !is_dm {
            if matches!(&update, StatusUpdate::BudgetExtended { .. }) {
                // Fall through — cost notifications must reach the user
            } else {
                continue;
            }
        }
        let now = tokio::time::Instant::now();
        // Explicit user actions and BudgetExtended notifications must always be
        // delivered. Ordinary progress remains rate-limited even when the task
        // description happens to include a URL.
        let is_immediate = update.requires_immediate_delivery();
        // Hard cap on DM status messages to prevent notification spam.
        // BudgetExtended and explicit user actions always bypass the cap.
        if !is_immediate && dm_status_count >= MAX_DM_STATUS_MESSAGES {
            // After the cap, just send typing indicator instead
            let _ = bot.send_chat_action(chat_id, ChatAction::Typing).await;
            continue;
        }
        if !is_immediate && now.duration_since(last_sent) < min_interval {
            continue;
        }
        let Some(text) = aidaemon_channels::turns::status_text(&update) else {
            continue;
        };
        let _ = bot.send_message(chat_id, text).await;
        dm_status_count += 1;
        // Re-send typing indicator immediately after each status message.
        // Telegram clears the typing indicator when a message is sent, so
        // without this there's a visible gap until the typing loop's next
        // 4-second tick.
        let _ = bot.send_chat_action(chat_id, ChatAction::Typing).await;
        last_sent = tokio::time::Instant::now();
    }
}

/// The content type a file is sent with, from its bytes: an image is an
/// image whichever tool sent it (a screenshot the browser took, or a PNG a
/// sub-agent saved and `send_file` delivered), so the transport shows it as
/// a photo; anything else goes as a document.
fn outbound_content_type(bytes: &[u8]) -> Option<String> {
    aidaemon_tools::sniff_image_mime(bytes).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file is sent by what it is: a PNG or JPEG gets an image type (shown
    /// as a photo), anything else none (sent as a document).
    #[test]
    fn a_sent_files_type_comes_from_its_content() {
        let png = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0];
        assert_eq!(outbound_content_type(&png).as_deref(), Some("image/png"));
        assert_eq!(
            outbound_content_type(&[0xFF, 0xD8, 0xFF, 0xE0]).as_deref(),
            Some("image/jpeg")
        );
        assert_eq!(outbound_content_type(b"%PDF-1.7 synthetic"), None);
        assert_eq!(outbound_content_type(b"homepage.png but text"), None);
    }

    /// Telegram's path: with the hub wired, a chat keeps one card and the
    /// turn runner owns how it ends (the same rule as every path); a kept
    /// card is registered for what outlives the turn to settle, and a card
    /// that became a handoff's reply for its later follow-up. Without the
    /// hub there is no card.
    #[tokio::test]
    async fn a_telegram_chat_hands_its_card_to_the_runner() {
        let hub = Arc::new(ChannelHub::new(Vec::new(), Arc::default()));
        let surface = |hub: Option<Arc<ChannelHub>>| TelegramSurface {
            bot: Bot::new(Arc::new(crate::TelegramChannel::new("000000:SYNTHETIC"))),
            chat_id: ChatId(1),
            session_id: "telegram:synthetic".into(),
            card: hub.as_ref().map(|hub| {
                Arc::new(HubSurfaceSink::new(
                    hub.clone(),
                    "telegram:synthetic".into(),
                ))
            }),
            hub,
        };
        assert!(surface(None).card_sink().is_none());
        let with_card = surface(Some(hub.clone()));
        assert!(with_card.card_sink().is_some());
        with_card.card_kept("41").await;
        assert_eq!(
            hub.checklist_surface("telegram:synthetic").await.as_deref(),
            Some("41")
        );
    }

    #[test]
    fn thinking_status_uses_native_telegram_indicator_only() {
        assert!(is_indicator_only_status(&StatusUpdate::Thinking(2)));
    }

    #[test]
    fn auth_empty_list_auto_claims_first_user() {
        let mut allowed = vec![];
        assert_eq!(check_auth(&mut allowed, 111), AuthResult::AutoClaimed);
        assert_eq!(allowed, vec![111]);
    }

    #[test]
    fn auth_allowed_user_is_authorized() {
        let mut allowed = vec![111, 222];
        assert_eq!(check_auth(&mut allowed, 222), AuthResult::Authorized);
    }

    #[test]
    fn auth_unknown_user_is_unauthorized() {
        let mut allowed = vec![111];
        assert_eq!(check_auth(&mut allowed, 999), AuthResult::Unauthorized);
        // List should not be modified
        assert_eq!(allowed, vec![111]);
    }

    #[test]
    fn auth_second_user_after_auto_claim_is_unauthorized() {
        let mut allowed = vec![];
        assert_eq!(check_auth(&mut allowed, 111), AuthResult::AutoClaimed);
        assert_eq!(check_auth(&mut allowed, 222), AuthResult::Unauthorized);
    }

    #[test]
    fn auth_same_user_after_auto_claim_is_authorized() {
        let mut allowed = vec![];
        assert_eq!(check_auth(&mut allowed, 111), AuthResult::AutoClaimed);
        assert_eq!(check_auth(&mut allowed, 111), AuthResult::Authorized);
    }

    // --- determine_role ---

    #[test]
    fn role_no_owner_ids_defaults_to_owner() {
        assert_eq!(determine_role(&[], 111), UserRole::Owner);
    }

    #[test]
    fn role_user_in_owner_ids_is_owner() {
        assert_eq!(determine_role(&[111, 222], 111), UserRole::Owner);
    }

    #[test]
    fn role_user_not_in_owner_ids_is_guest() {
        assert_eq!(determine_role(&[111], 222), UserRole::Guest);
    }

    #[test]
    fn parse_web_app_action_handles_continue_on_computer() {
        let payload = r#"{"type":"aidaemon.telegram.open_on_computer.v1"}"#;
        match TelegramBot::parse_web_app_action(payload) {
            Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                assert!(relay_session_id.is_none());
            }
            other => panic!("unexpected parsed action: {:?}", other.map(|_| "unknown")),
        }
    }

    #[test]
    fn parse_web_app_action_extracts_agent_message() {
        let payload = r#"{"type":"aidaemon.telegram.agent_message.v1","text":"Continue fixing the failing tests."}"#;
        match TelegramBot::parse_web_app_action(payload) {
            Some(TelegramWebAppAction::AgentMessage(text)) => {
                assert_eq!(text, "Continue fixing the failing tests.");
            }
            other => panic!("unexpected parsed action: {:?}", other.map(|_| "unknown")),
        }
    }

    #[test]
    fn parse_web_app_action_extracts_continue_session_id() {
        let payload =
            r#"{"type":"aidaemon.telegram.open_on_computer.v1","relay_session_id":"sess_123"}"#;
        match TelegramBot::parse_web_app_action(payload) {
            Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                assert_eq!(relay_session_id.as_deref(), Some("sess_123"));
            }
            other => panic!("unexpected parsed action: {:?}", other.map(|_| "unknown")),
        }
    }

    #[test]
    fn parse_web_app_action_extracts_continue_session_id_camel_case() {
        let payload =
            r#"{"type":"aidaemon.telegram.open_on_computer.v1","relaySessionId":"sess_camel"}"#;
        match TelegramBot::parse_web_app_action(payload) {
            Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                assert_eq!(relay_session_id.as_deref(), Some("sess_camel"));
            }
            other => panic!("unexpected parsed action: {:?}", other.map(|_| "unknown")),
        }
    }

    #[test]
    fn parse_web_app_action_extracts_continue_session_id_from_object() {
        let payload =
            r#"{"type":"aidaemon.telegram.open_on_computer.v1","session":{"id":"sess_obj"}}"#;
        match TelegramBot::parse_web_app_action(payload) {
            Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                assert_eq!(relay_session_id.as_deref(), Some("sess_obj"));
            }
            other => panic!("unexpected parsed action: {:?}", other.map(|_| "unknown")),
        }
    }

    #[test]
    fn parse_web_app_action_accepts_legacy_session_id_key() {
        let payload =
            r#"{"type":"aidaemon.telegram.open_on_computer.v1","session_id":"sess_legacy"}"#;
        match TelegramBot::parse_web_app_action(payload) {
            Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                assert_eq!(relay_session_id.as_deref(), Some("sess_legacy"));
            }
            other => panic!("unexpected parsed action: {:?}", other.map(|_| "unknown")),
        }
    }

    #[test]
    fn parse_web_app_action_accepts_nested_payload_string() {
        let payload = r#"{"data":"{\"type\":\"aidaemon.telegram.open_on_computer.v1\",\"relay_session_id\":\"sess_nested\"}"}"#;
        match TelegramBot::parse_web_app_action(payload) {
            Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                assert_eq!(relay_session_id.as_deref(), Some("sess_nested"));
            }
            other => panic!("unexpected parsed action: {:?}", other.map(|_| "unknown")),
        }
    }

    #[test]
    fn parse_web_app_action_rejects_unknown_payload() {
        let payload = r#"{"type":"unknown","text":"hello"}"#;
        assert!(TelegramBot::parse_web_app_action(payload).is_none());
    }

    #[test]
    fn fallback_session_namespace_uses_bot_id_prefix() {
        let ns = fallback_session_namespace_from_token("123456789:ABCDEF");
        assert_eq!(ns, "tg123456789");
    }

    #[test]
    fn fallback_session_namespace_sanitizes_invalid_chars() {
        let ns = fallback_session_namespace_from_token("12$34:^secret");
        assert_eq!(ns, "tg1234");
    }
}
