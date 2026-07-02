use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock as StdRwLock, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::future::BoxFuture;
use teloxide::error_handlers::{ErrorHandler, LoggingErrorHandler};
use teloxide::prelude::*;
use teloxide::types::{
    BotCommand, ButtonRequest, ChatAction, InlineKeyboardButton, InlineKeyboardMarkup, InputFile,
    KeyboardButton, KeyboardMarkup, ParseMode, WebAppInfo,
};
use teloxide::update_listeners::webhooks;
use teloxide::RequestError;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use super::approval_render;
use super::commands::{shared_commands, CommandCategory, CommandDef};
use super::formatting::{
    build_help_text, html_escape, markdown_to_telegram_html, markdown_to_telegram_plain_fallback,
    sanitize_filename, split_message,
};
use super::telegram_bootstrap_signing::{
    random_terminal_bootstrap_nonce, sign_terminal_tenant_bot_bootstrap_proof,
    terminal_tenant_bot_bootstrap_url,
};
use crate::agent::Agent;
use crate::channels::{should_ignore_lightweight_interjection, ChannelHub, SessionMap};
#[cfg(feature = "discord")]
use crate::channels::{spawn_discord_channel, DiscordChannel};
#[cfg(feature = "slack")]
use crate::channels::{spawn_slack_channel, SlackChannel};
use crate::config::{AppConfig, TelegramWebhookConfig};
use crate::tasks::{QueueOutcome, TaskRegistry};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::traits::{Channel, ChannelCapabilities, StateStore};
use crate::types::{
    ApprovalResponse, ChannelContext, ChannelVisibility, MediaKind, MediaMessage, StatusUpdate,
    UserRole,
};

pub struct TelegramChannel {
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
    webhook: TelegramWebhookConfig,
    allowed_user_ids: StdRwLock<Vec<u64>>,
    /// Telegram user IDs recognized as owners (from `users.owner_ids.telegram`).
    owner_user_ids: Vec<u64>,
    agent: Arc<Agent>,
    config_path: PathBuf,
    /// Pending approvals keyed by a unique callback ID.
    pending_approvals: Mutex<HashMap<String, tokio::sync::oneshot::Sender<ApprovalResponse>>>,
    /// Shared session map — maps session_id to channel name.
    session_map: SessionMap,
    /// Task registry for tracking background agent work.
    task_registry: Arc<TaskRegistry>,
    /// Whether file transfer is enabled.
    files_enabled: bool,
    /// Directory for saving inbound files.
    inbox_dir: PathBuf,
    /// Max file size in MB for inbound files.
    max_file_size_mb: u64,
    /// State store for querying token usage.
    state: Arc<dyn StateStore>,
    /// Reference to the channel hub for dynamic bot registration.
    /// Set after construction via set_channel_hub().
    channel_hub: StdRwLock<Option<Weak<ChannelHub>>>,
    /// Seconds of no heartbeat before declaring the agent stuck (0 = disabled).
    watchdog_stale_threshold_secs: u64,
    /// URL for the Telegram Mini App terminal frontend.
    terminal_web_app_url: String,
    /// Terminal-lite session manager.
    terminal_lite: crate::terminal_lite::TerminalLiteManager,
    /// Daemon start time used for post-restart UX guardrails.
    started_at: Instant,
}

#[derive(Debug, Default)]
struct SetupLoginResult {
    success: bool,
    timed_out: bool,
    error: Option<String>,
    lines: Vec<String>,
    urls: Vec<String>,
}

use crate::wizard::CloudflaredZoneValidation;

use crate::cli_agent_flags::{self, AGENT_FLAGS_CACHE_TTL_SECS, SUPPORTED_TERMINAL_AGENTS};

const DEFAULT_TERMINAL_AGENT: &str = "codex";
const TELEGRAM_EXPANDABLE_TRIGGER_CHARS: usize = 1_800;
const TELEGRAM_MAX_MESSAGE_LEN: usize = 4096;
const TELEGRAM_EXPANDABLE_WRAPPER_LEN: usize = "<blockquote expandable></blockquote>".len();
const TELEGRAM_WEBAPP_TYPE_AGENT_MESSAGE: &str = "aidaemon.telegram.agent_message.v1";
const TELEGRAM_WEBAPP_TYPE_CONTINUE_COMPUTER: &str = "aidaemon.telegram.open_on_computer.v1";
const TELEGRAM_WEBAPP_MAX_TEXT_CHARS: usize = 2_000;
static LOW_LATENCY_RESTART_SCHEDULED: AtomicBool = AtomicBool::new(false);

enum TelegramWebAppAction {
    AgentMessage(String),
    ContinueOnComputer { relay_session_id: Option<String> },
}

/// All commands available in the Telegram channel (shared + Telegram-specific).
///
/// Used to register Telegram's `/` command menu via `setMyCommands` and to
/// generate `/help` output.
fn telegram_commands() -> Vec<CommandDef> {
    let mut cmds = shared_commands();
    cmds.extend(vec![
        CommandDef {
            name: "restart",
            description: "Restart the daemon",
            usage: None,
            category: CommandCategory::Restart,
        },
        CommandDef {
            name: "connect",
            description: "Add a new bot channel",
            usage: Some("/connect <platform> <token>"),
            category: CommandCategory::Connect,
        },
        CommandDef {
            name: "setup",
            description: "Setup wizard (webhooks, low-latency)",
            usage: Some("/setup [lowlatency]"),
            category: CommandCategory::Connect,
        },
        CommandDef {
            name: "bots",
            description: "List connected bots",
            usage: None,
            category: CommandCategory::Connect,
        },
        CommandDef {
            name: "agent",
            description: "Start or manage CLI agent sessions",
            usage: Some("/agent [codex|claude|gemini|opencode] [dir]"),
            category: CommandCategory::Terminal,
        },
        CommandDef {
            name: "terminal",
            description: "Terminal bridge commands",
            usage: Some("/terminal [lite|start|open]"),
            category: CommandCategory::Terminal,
        },
        CommandDef {
            name: "help",
            description: "Show available commands",
            usage: None,
            category: CommandCategory::Core,
        },
    ]);
    cmds
}

impl TelegramChannel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bot_token: &str,
        allowed_user_ids: Vec<u64>,
        webhook: TelegramWebhookConfig,
        owner_user_ids: Vec<u64>,
        agent: Arc<Agent>,
        config_path: PathBuf,
        session_map: SessionMap,
        task_registry: Arc<TaskRegistry>,
        files_enabled: bool,
        inbox_dir: PathBuf,
        max_file_size_mb: u64,
        state: Arc<dyn StateStore>,
        watchdog_stale_threshold_secs: u64,
        terminal_web_app_url: String,
        terminal_allowed_prefixes: Vec<String>,
    ) -> Self {
        let bot = Bot::new(bot_token);
        let terminal_allowed_prefixes = terminal_allowed_prefixes
            .into_iter()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .collect::<HashSet<_>>();
        Self {
            bot_username: StdRwLock::new("telegram".to_string()),
            cached_channel_name: StdRwLock::new("telegram".to_string()),
            session_namespace: StdRwLock::new(None),
            bot,
            bot_token: bot_token.to_string(),
            webhook,
            allowed_user_ids: StdRwLock::new(allowed_user_ids),
            owner_user_ids,
            agent,
            config_path,
            pending_approvals: Mutex::new(HashMap::new()),
            session_map,
            task_registry,
            files_enabled,
            inbox_dir,
            max_file_size_mb,
            state,
            channel_hub: StdRwLock::new(None),
            watchdog_stale_threshold_secs,
            terminal_web_app_url,
            terminal_lite: crate::terminal_lite::TerminalLiteManager::new(
                terminal_allowed_prefixes,
            ),
            started_at: Instant::now(),
        }
    }

    /// Set the channel hub reference for dynamic bot registration.
    /// Called after the hub is constructed in core.rs.
    pub fn set_channel_hub(&self, hub: Weak<ChannelHub>) {
        if let Ok(mut guard) = self.channel_hub.write() {
            *guard = Some(hub);
        }
    }

    async fn send_compact_or_full_reply(
        &self,
        bot: &Bot,
        chat_id: ChatId,
        markdown: &str,
    ) -> anyhow::Result<()> {
        send_full_or_expandable_reply(bot, chat_id, markdown).await
    }

    /// Persist the current allowed_user_ids list to config.toml.
    /// Handles both `[telegram]` and `[[telegram_bots]]` config formats.
    async fn persist_allowed_user_ids(&self, ids: &[u64]) -> anyhow::Result<()> {
        let content = tokio::fs::read_to_string(&self.config_path).await?;
        let mut doc: toml::Table = content.parse()?;

        let ids_toml = toml::Value::Array(
            ids.iter()
                .map(|&id| toml::Value::Integer(id as i64))
                .collect(),
        );

        // Update whichever config format exists: [telegram] or [[telegram_bots]]
        let mut updated = false;
        if let Some(telegram) = doc.get_mut("telegram").and_then(|v| v.as_table_mut()) {
            telegram.insert("allowed_user_ids".to_string(), ids_toml.clone());
            updated = true;
        }
        if let Some(bots) = doc.get_mut("telegram_bots").and_then(|v| v.as_array_mut()) {
            if let Some(first) = bots.first_mut().and_then(|v| v.as_table_mut()) {
                first.insert("allowed_user_ids".to_string(), ids_toml);
                updated = true;
            }
        }

        if !updated {
            anyhow::bail!("No [telegram] or [[telegram_bots]] section found in config");
        }

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        tokio::fs::write(&self.config_path, &new_content).await?;
        info!("Persisted allowed_user_ids to config.toml");
        Ok(())
    }

    /// Get the bot's username, fetching from Telegram API if not cached.
    async fn get_bot_username(&self) -> String {
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

    fn cached_bot_label(&self) -> Option<String> {
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

    async fn sync_terminal_tenant_bot_token(
        &self,
        bot_token: &str,
        user_id: u64,
        label: Option<&str>,
    ) -> Result<(), String> {
        let url = terminal_tenant_bot_bootstrap_url(&self.terminal_web_app_url)?;
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| format!("failed to read system clock: {}", err))?
            .as_secs() as i64;
        let nonce = random_terminal_bootstrap_nonce(16);
        let sig = sign_terminal_tenant_bot_bootstrap_proof(bot_token, user_id, ts, &nonce)?;
        let mut payload = serde_json::json!({
            "user_id": user_id,
            "bot_token": bot_token,
            "ts": ts,
            "nonce": nonce,
            "sig": sig,
        });
        if let Some(label) = label
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        {
            if let Some(map) = payload.as_object_mut() {
                map.insert(
                    "label".to_string(),
                    serde_json::Value::String(label.to_string()),
                );
            }
        }

        let client = reqwest::Client::builder()
            .user_agent("aidaemon-terminal-bot-bootstrap/1.0")
            .timeout(Duration::from_secs(8))
            .build()
            .map_err(|err| format!("failed to build HTTP client: {}", err))?;
        let response = client
            .post(url)
            .json(&payload)
            .send()
            .await
            .map_err(|err| format!("terminal bot sync request failed: {}", err))?;
        let status = response.status();
        let parsed: serde_json::Value = response
            .json()
            .await
            .map_err(|err| format!("terminal bot sync returned invalid JSON: {}", err))?;
        if status.is_success() && parsed.get("ok").and_then(|value| value.as_bool()) == Some(true) {
            return Ok(());
        }

        let message = parsed
            .get("message")
            .and_then(|value| value.as_str())
            .or_else(|| parsed.get("error").and_then(|value| value.as_str()))
            .unwrap_or("unknown error");
        Err(format!(
            "terminal bot sync failed (status {}): {}",
            status, message
        ))
    }

    async fn sync_terminal_tenant_bot_token_for_allowed_users(&self, label: Option<&str>) {
        let allowed_user_ids = self
            .allowed_user_ids
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        for user_id in allowed_user_ids {
            if let Err(err) = self
                .sync_terminal_tenant_bot_token(&self.bot_token, user_id, label)
                .await
            {
                warn!(
                    error = %err,
                    user_id,
                    "Failed to auto-sync Telegram bot token during channel startup"
                );
            }
        }
    }

    /// Build the session ID for a chat, prefixing with bot username.
    /// Uses a stable namespace to avoid mid-run session ID drift.
    async fn session_id(&self, chat_id: i64) -> String {
        let namespace = self.session_namespace().await;
        if namespace == "default" {
            chat_id.to_string()
        } else {
            format!("{}:{}", namespace, chat_id)
        }
    }

    async fn session_namespace(&self) -> String {
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
    async fn channel_name(&self) -> String {
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

        let mut backoff = crate::backoff::ChannelReconnectBackoff::new(
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
        self.sync_terminal_tenant_bot_token_for_allowed_users(Some(bot_username.as_str()))
            .await;

        // Register commands with Telegram so they appear in the "/" menu.
        let bot_commands: Vec<BotCommand> = telegram_commands()
            .iter()
            .map(|c| BotCommand::new(c.name, c.description))
            .collect();
        if let Err(e) = self.bot.set_my_commands(bot_commands).await {
            warn!(error = %e, "Failed to register bot commands with Telegram");
        }

        let handler = dptree::entry()
            .branch(Update::filter_message().endpoint({
                let channel = Arc::clone(&self);
                move |msg: teloxide::types::Message, bot: Bot| {
                    let channel = Arc::clone(&channel);
                    async move {
                        channel.handle_message(msg, bot).await;
                        respond(())
                    }
                }
            }))
            .branch(Update::filter_callback_query().endpoint({
                let channel = Arc::clone(&self);
                move |q: CallbackQuery, bot: Bot| {
                    let channel = Arc::clone(&channel);
                    async move {
                        channel.handle_callback(q, bot).await;
                        respond(())
                    }
                }
            }));

        let mut dispatcher = Dispatcher::builder(self.bot.clone(), handler)
            .enable_ctrlc_handler()
            .build();

        if let Some((listen_addr, webhook_opts)) = self.build_webhook_options(&bot_username) {
            info!(
                name = %bot_username,
                listen_addr = %listen_addr,
                "Starting Telegram webhook listener"
            );
            match webhooks::axum_to_router(self.bot.clone(), webhook_opts).await {
                Ok((listener, stop_flag, router)) => {
                    match tokio::net::TcpListener::bind(listen_addr).await {
                        Ok(tcp_listener) => {
                            tokio::spawn(async move {
                                if let Err(err) = axum::serve(tcp_listener, router)
                                    .with_graceful_shutdown(stop_flag)
                                    .await
                                {
                                    warn!(error = %err, "Telegram webhook server stopped with error");
                                }
                            });
                            dispatcher
                                .dispatch_with_listener(
                                    listener,
                                    LoggingErrorHandler::with_custom_text(
                                        "An error from the Telegram webhook listener",
                                    ),
                                )
                                .await;
                            return;
                        }
                        Err(err) => {
                            warn!(
                                name = %bot_username,
                                listen_addr = %listen_addr,
                                error = %err,
                                hint = "for multi-bot webhook mode, use a unique listen_addr per bot",
                                "Failed to bind Telegram webhook listener, falling back to long polling"
                            );
                            let _ = self.bot.delete_webhook().send().await;
                        }
                    }
                }
                Err(err) => {
                    warn!(
                        name = %bot_username,
                        error = %err,
                        "Failed to initialize Telegram webhook mode, falling back to long polling"
                    );
                    let _ = self.bot.delete_webhook().send().await;
                }
            }
        }

        // Clear any stale webhook before starting long polling. If a webhook
        // was previously set (e.g., via `/setup lowlatency apply`) but we're
        // now falling back to polling, Telegram will keep sending updates to
        // the old webhook URL and getUpdates will return nothing.
        info!(name = %bot_username, "Clearing any existing webhook before starting long polling");
        if let Err(err) = self.bot.delete_webhook().send().await {
            warn!(
                name = %bot_username,
                error = %err,
                "Failed to delete webhook before long polling (updates may not arrive)"
            );
        }

        // Long polling. Use an explicit polling listener + a loud error handler
        // (teloxide's default logs under a target our filter drops, which is how
        // a wedged getUpdates went silent). A watchdog runs alongside: if
        // getUpdates keeps failing past the threshold, it returns, which drops
        // the dispatch future and lets `start_with_retry` reconnect with a fresh
        // getUpdates connection instead of staying silently dead.
        let error_count = Arc::new(AtomicU64::new(0));
        let err_handler = Arc::new(PollingErrorHandler {
            bot_username: bot_username.clone(),
            error_count: Arc::clone(&error_count),
        });
        let listener = teloxide::update_listeners::polling_default(self.bot.clone()).await;
        tokio::select! {
            _ = dispatcher.dispatch_with_listener(listener, err_handler) => {}
            _ = poll_health_watchdog(error_count, bot_username.clone()) => {}
        }
    }

    fn build_webhook_options(
        &self,
        bot_username: &str,
    ) -> Option<(std::net::SocketAddr, webhooks::Options)> {
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

        // Teloxide sends the URL verbatim to Telegram's setWebhook, so we
        // must include the path in the URL. The `opts.path()` call only sets
        // the local axum route, NOT the URL registered with Telegram.
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
        let mut opts = webhooks::Options::new(listen_addr, webhook_url);
        if let Some(path) = path {
            opts = opts.path(path);
        }
        if let Some(max_connections) = self.webhook.max_connections {
            if (1..=100).contains(&max_connections) {
                opts = opts.max_connections(max_connections);
            } else {
                warn!(
                    name = %bot_username,
                    max_connections,
                    "Telegram webhook `max_connections` must be 1..=100; ignoring"
                );
            }
        }
        if self.webhook.drop_pending_updates {
            opts = opts.drop_pending_updates();
        }
        Some((listen_addr, opts))
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
    async fn handle_callback(&self, q: CallbackQuery, bot: Bot) {
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

        if data == "agent:share" || data.starts_with("agent:share:") {
            let explicit_relay_session_id = data
                .strip_prefix("agent:share:")
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(|v| v.to_string());
            let _ = bot
                .answer_callback_query(q.id.clone())
                .text("Generating share code...")
                .await;
            if let Some(teloxide::types::MaybeInaccessibleMessage::Regular(m)) = q.message {
                self.send_agent_share_code(&bot, m.chat.id, user_id, explicit_relay_session_id)
                    .await;
            } else {
                let _ = bot
                    .answer_callback_query(q.id)
                    .text("Open the chat and run /agent share.")
                    .show_alert(true)
                    .await;
            }
            return;
        }

        // Parse callback data: "approve:{once|session|always|deny}:{id}"
        // or "goal:{confirm|cancel}:{id}"
        let parts: Vec<&str> = data.splitn(3, ':').collect();
        if parts.len() != 3 || (parts[0] != "approve" && parts[0] != "goal") || parts[2].is_empty()
        {
            let _ = bot
                .answer_callback_query(q.id)
                .text("This action is no longer valid. Please run the command again.")
                .show_alert(true)
                .await;
            return;
        }

        let prefix = parts[0];
        let action = parts[1];
        let approval_id = parts[2];

        let (response, label) = if prefix == "goal" {
            match action {
                "confirm" => (ApprovalResponse::AllowOnce, "Confirmed ✅".to_string()),
                "cancel" => (ApprovalResponse::Deny, "Cancelled ❌".to_string()),
                _ => {
                    let _ = bot
                        .answer_callback_query(q.id)
                        .text("This confirmation action is no longer valid.")
                        .show_alert(true)
                        .await;
                    return;
                }
            }
        } else {
            let response = match action {
                "once" => ApprovalResponse::AllowOnce,
                "session" => ApprovalResponse::AllowSession,
                "always" => ApprovalResponse::AllowAlways,
                "deny" => ApprovalResponse::Deny,
                _ => {
                    let _ = bot
                        .answer_callback_query(q.id)
                        .text("This approval action is no longer valid.")
                        .show_alert(true)
                        .await;
                    return;
                }
            };
            let (icon, label) = approval_render::response_status(&response);
            (response, format!("{icon} {label}"))
        };

        // Send the response (check pending before updating the message)
        let mut pending = self.pending_approvals.lock().await;
        let response_tx = pending.remove(approval_id);
        drop(pending);

        let Some(response_tx) = response_tx else {
            warn!(approval_id, "Stale approval callback (no pending request)");
            let _ = bot
                .answer_callback_query(q.id.clone())
                .text("Approval expired. Please run the command again.")
                .show_alert(true)
                .await;
            if let Some(teloxide::types::MaybeInaccessibleMessage::Regular(m)) = q.message {
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

        // Acknowledge the callback and update the message
        let _ = bot.answer_callback_query(q.id).text(&label).await;

        if let Some(teloxide::types::MaybeInaccessibleMessage::Regular(m)) = q.message {
            let original = m.text().unwrap_or("");
            let updated = approval_render::finalize_approval_message(original, &response);
            let _ = bot.edit_message_text(m.chat.id, m.id, updated).await;

            // Auto-dismiss the resolved card after a short delay so an answered
            // approval doesn't linger and get mistaken for one still awaiting a
            // response. This runs only after the user has responded, so a
            // still-pending prompt is never removed out from under them.
            if let Some(delay) = Self::resolved_approval_dismiss_delay() {
                let bot = bot.clone();
                let chat_id = m.chat.id;
                let msg_id = m.id;
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    let _ = bot.delete_message(chat_id, msg_id).await;
                });
            }
        }

        let _ = response_tx.send(response);
    }

    async fn handle_command(&self, text: &str, msg: &teloxide::types::Message, bot: &Bot) {
        let parts: Vec<&str> = text.splitn(2, ' ').collect();
        let cmd_raw = parts[0];
        let cmd = cmd_raw.split('@').next().unwrap_or(cmd_raw);
        let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

        if cmd == "/terminal" || cmd == "/agent" {
            self.handle_terminal_command(arg, msg, bot, cmd).await;
            return;
        }

        // Try the shared command dispatcher first.
        let ctx = crate::channels::commands::CommandContext {
            agent: Arc::clone(&self.agent),
            state: Arc::clone(&self.state),
            task_registry: Arc::clone(&self.task_registry),
            config_path: self.config_path.clone(),
        };
        let session_id = self.session_id(msg.chat.id.0).await;
        if let Some(reply) = ctx.dispatch(cmd, arg, &session_id).await {
            for chunk in split_message(&reply, 4096) {
                let _ = bot.send_message(msg.chat.id, chunk).await;
            }
            return;
        }

        // Channel-specific commands.
        let reply = match cmd {
            "/restart" => {
                let _ = bot.send_message(msg.chat.id, "Restarting...").await;
                info!("Restart requested via Telegram");
                restart_process();
                // If exec fails, we're still alive
                "Restart failed. You may need to restart manually.".to_string()
            }
            "/connect" => {
                self.handle_connect_command(arg, msg.from.as_ref().map(|u| u.id.0).unwrap_or(0))
                    .await
            }
            "/setup" => {
                self.handle_setup_command(
                    arg,
                    msg.from.as_ref().map(|u| u.id.0).unwrap_or(0),
                    msg.chat.id.0,
                )
                .await
            }
            "/bots" => self.handle_bots_command().await,
            "/help" | "/start" => build_help_text(&telegram_commands(), "/"),
            _ => format!(
                "Unknown command: {}\nType /help for available commands.",
                cmd_raw
            ),
        };

        for chunk in split_message(&reply, 4096) {
            let _ = bot.send_message(msg.chat.id, chunk).await;
        }
    }

    fn terminal_help_text() -> String {
        "Terminal mode (owner only)\n\n\
         Usage:\n\
         /terminal\n\
         /terminal <codex|claude|gemini|opencode> [working_dir]\n\
         /terminal <codex|claude|gemini|opencode> [working_dir] [agent_flags...]\n\
         /terminal <codex|claude|gemini|opencode> [working_dir] -- [agent_flags...]\n\
         /terminal start <agent> [working_dir]\n\
         /terminal start <agent> [working_dir] [agent_flags...]\n\
         /terminal start <agent> [working_dir] -- [agent_flags...]\n\
         /terminal lite [start] [working_dir]\n\
         /terminal lite status\n\
         /terminal lite stop\n\
         /terminal open\n\
         /terminal help\n\n\
         Examples:\n\
         /terminal codex\n\
         /terminal codex --chrome\n\
         /terminal codex ~/projects/aidaemon --chrome --dangerously-skip-permissions\n\
         /terminal codex ~/projects/aidaemon -- --chrome --dangerously-skip-permissions\n\
         /terminal opencode ~/projects/aidaemon\n\
         /terminal lite ~/projects/aidaemon\n\
         /terminal claude ~/projects/aidaemon\n\
         /terminal codex \"~/projects/my app\""
            .to_string()
    }

    fn agent_help_text() -> String {
        "Agent session mode (owner only)\n\n\
         Usage:\n\
         /agent\n\
         /agent <codex|claude|gemini|opencode> [working_dir]\n\
         /agent <codex|claude|gemini|opencode> [working_dir] [agent_flags...]\n\
         /agent <codex|claude|gemini|opencode> [working_dir] -- [agent_flags...]\n\
         /agent start <agent> [working_dir]\n\
         /agent start <agent> [working_dir] [agent_flags...]\n\
         /agent flags <agent> [refresh]\n\
         /agent defaults\n\
         /agent defaults set <agent> [agent_flags...]\n\
         /agent defaults clear [agent|all]\n\
         /agent share [relay_session_id]\n\
         /agent resume <code>\n\
         /agent open\n\
         /agent help\n\n\
         Examples:\n\
         /agent codex\n\
         /agent codex --chrome\n\
         /agent codex ~/projects/aidaemon --chrome --dangerously-skip-permissions\n\
         /agent claude ~/projects/aidaemon\n\
         /agent opencode ~/projects/aidaemon\n\
         /agent flags codex\n\
         /agent flags codex refresh\n\
         /agent share\n\
         /agent resume ABCDEFGHJKLM\n\
         /agent defaults set codex --chrome --dangerously-skip-permissions\n\n\
         Tip: add `--no-default-flags` to bypass saved flags once.\n\n\
         For chat-based shell mode, use:\n\
         /terminal lite [working_dir]"
            .to_string()
    }

    async fn handle_agent_flags_command(
        &self,
        args: Vec<String>,
        msg: &teloxide::types::Message,
        bot: &Bot,
        user_id: u64,
    ) {
        if args.is_empty() {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    "Usage: /agent flags <agent> [refresh]\nExamples:\n/agent flags codex\n/agent flags codex refresh\n\nSupported agents: codex, claude, gemini, opencode",
                )
                .await;
            return;
        }

        let (agent_raw, refresh) = if args
            .first()
            .map(|v| v.eq_ignore_ascii_case("refresh"))
            .unwrap_or(false)
        {
            (args.get(1).cloned().unwrap_or_default(), true)
        } else {
            let refresh = args
                .get(1)
                .map(|v| v.eq_ignore_ascii_case("refresh"))
                .unwrap_or(false);
            (args[0].clone(), refresh)
        };

        let Some(agent) = cli_agent_flags::normalize_terminal_agent_name(&agent_raw) else {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    "Unknown agent. Use codex/claude/gemini/opencode.",
                )
                .await;
            return;
        };

        let namespace = self.session_namespace().await;
        if !refresh {
            if let Some(cached) =
                cli_agent_flags::load_agent_flags_cache(&*self.state, &namespace, user_id, &agent)
                    .await
            {
                let age = chrono::Utc::now().timestamp() - cached.updated_at_unix;
                if (0..=AGENT_FLAGS_CACHE_TTL_SECS).contains(&age) {
                    let pages = cli_agent_flags::format_agent_flag_docs(&agent, &cached.docs, true);
                    for page in pages {
                        send_markdown_chunks_or_fallback(bot, msg.chat.id, &page).await;
                    }
                    return;
                }
            }
        }

        let _ = bot
            .send_message(msg.chat.id, format!("Discovering flags for {}...", agent))
            .await;

        match cli_agent_flags::discover_agent_flags(&agent).await {
            Ok(docs) => {
                let _ = cli_agent_flags::save_agent_flags_cache(
                    &*self.state,
                    &namespace,
                    user_id,
                    &agent,
                    &docs,
                )
                .await;
                let pages = cli_agent_flags::format_agent_flag_docs(&agent, &docs, false);
                for page in pages {
                    send_markdown_chunks_or_fallback(bot, msg.chat.id, &page).await;
                }
            }
            Err(err) => {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!("Failed to discover flags for {}: {}", agent, err),
                    )
                    .await;
            }
        }
    }

    async fn handle_agent_defaults_command(
        &self,
        mut args: Vec<String>,
        msg: &teloxide::types::Message,
        bot: &Bot,
        user_id: u64,
    ) {
        let subcommand = args
            .first()
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_else(|| "show".to_string());

        if !args.is_empty() {
            args.remove(0);
        }

        if subcommand == "set" {
            let Some(agent_raw) = args.first() else {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Usage: /agent defaults set <agent> [agent_flags...]",
                    )
                    .await;
                return;
            };
            let Some(agent) = cli_agent_flags::normalize_terminal_agent_name(agent_raw) else {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Unknown agent. Use codex/claude/gemini/opencode.",
                    )
                    .await;
                return;
            };
            let cleaned = cli_agent_flags::normalize_terminal_agent_args(
                args.into_iter().skip(1).collect::<Vec<_>>(),
            );
            let (cleaned, rewrote_permission_flag) =
                crate::normalize_terminal_agent_permission_aliases(Some(agent.as_str()), cleaned);
            if cleaned.is_empty() {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "No flags provided. Example: /agent defaults set codex --chrome",
                    )
                    .await;
                return;
            }
            let namespace = self.session_namespace().await;
            let mut defaults = cli_agent_flags::load_terminal_agent_defaults(
                &*self.state,
                &namespace,
                msg.chat.id.0,
                user_id,
            )
            .await;
            defaults.insert(agent.clone(), cleaned.clone());
            match cli_agent_flags::save_terminal_agent_defaults(
                &*self.state,
                &namespace,
                msg.chat.id.0,
                user_id,
                &defaults,
            )
            .await
            {
                Ok(_) => {
                    let saved_message = if rewrote_permission_flag {
                        format!(
                            "Saved default flags for {}:\n`{}`\n\nNormalized `--allow-dangerously-skip-permissions` to `--dangerously-skip-permissions` for Claude.",
                            agent,
                            cleaned.join(" ")
                        )
                    } else {
                        format!(
                            "Saved default flags for {}:\n`{}`",
                            agent,
                            cleaned.join(" ")
                        )
                    };
                    let _ = bot.send_message(msg.chat.id, saved_message).await;
                }
                Err(err) => {
                    warn!(error = %err, "Failed to save /agent defaults");
                    let _ = bot
                        .send_message(msg.chat.id, "Failed to save defaults.")
                        .await;
                }
            }
            return;
        }

        if subcommand == "clear" {
            let namespace = self.session_namespace().await;
            let mut defaults = cli_agent_flags::load_terminal_agent_defaults(
                &*self.state,
                &namespace,
                msg.chat.id.0,
                user_id,
            )
            .await;
            if defaults.is_empty() {
                let _ = bot.send_message(msg.chat.id, "No saved defaults.").await;
                return;
            }

            let target = args
                .first()
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_else(|| "all".to_string());
            if target == "all" {
                defaults.clear();
                if cli_agent_flags::save_terminal_agent_defaults(
                    &*self.state,
                    &namespace,
                    msg.chat.id.0,
                    user_id,
                    &defaults,
                )
                .await
                .is_ok()
                {
                    let _ = bot
                        .send_message(msg.chat.id, "Cleared all agent defaults.")
                        .await;
                } else {
                    let _ = bot
                        .send_message(msg.chat.id, "Failed to clear defaults.")
                        .await;
                }
                return;
            }

            let Some(agent) = cli_agent_flags::normalize_terminal_agent_name(&target) else {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Usage: /agent defaults clear [agent|all]\nAgents: codex/claude/gemini/opencode",
                    )
                    .await;
                return;
            };
            if defaults.remove(&agent).is_none() {
                let _ = bot
                    .send_message(msg.chat.id, format!("No saved defaults for {}.", agent))
                    .await;
                return;
            }
            if cli_agent_flags::save_terminal_agent_defaults(
                &*self.state,
                &namespace,
                msg.chat.id.0,
                user_id,
                &defaults,
            )
            .await
            .is_ok()
            {
                let _ = bot
                    .send_message(msg.chat.id, format!("Cleared defaults for {}.", agent))
                    .await;
            } else {
                let _ = bot
                    .send_message(msg.chat.id, "Failed to clear defaults.")
                    .await;
            }
            return;
        }

        let namespace = self.session_namespace().await;
        let defaults = cli_agent_flags::load_terminal_agent_defaults(
            &*self.state,
            &namespace,
            msg.chat.id.0,
            user_id,
        )
        .await;
        if defaults.is_empty() {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    "No saved agent defaults yet.\nUse `/agent defaults set <agent> [flags...]`.",
                )
                .await;
            return;
        }
        let mut lines = vec!["Saved agent defaults for this chat:".to_string()];
        for agent in SUPPORTED_TERMINAL_AGENTS {
            if let Some(args) = defaults.get(*agent) {
                lines.push(format!("- {}: `{}`", agent, args.join(" ")));
            }
        }
        lines.push("Clear with `/agent defaults clear [agent|all]`.".to_string());
        let _ = bot.send_message(msg.chat.id, lines.join("\n")).await;
    }

    async fn handle_terminal_lite_command(
        &self,
        args: Vec<String>,
        msg: &teloxide::types::Message,
        bot: &Bot,
        user_id: u64,
    ) {
        let reply = self
            .terminal_lite
            .start_session(msg.chat.id.0, user_id, args)
            .await;
        let _ = bot.send_message(msg.chat.id, reply).await;
    }

    async fn handle_terminal_lite_input(
        &self,
        chat_id: i64,
        user_id: u64,
        user_role: UserRole,
        text: &str,
    ) -> Option<String> {
        self.terminal_lite
            .handle_input(chat_id, user_id, user_role, text)
            .await
    }

    async fn handle_terminal_command(
        &self,
        arg: &str,
        msg: &teloxide::types::Message,
        bot: &Bot,
        invoked_cmd: &str,
    ) {
        let user_id = msg.from.as_ref().map(|u| u.id.0).unwrap_or(0);
        if determine_role(&self.owner_user_ids, user_id) != UserRole::Owner {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    format!("Only the owner can use {} in this chat.", invoked_cmd),
                )
                .await;
            return;
        }

        let trimmed = arg.trim();
        let mut tokens = if trimmed.is_empty() {
            Vec::new()
        } else {
            shell_words::split(trimmed).unwrap_or_else(|_| {
                trimmed
                    .split_whitespace()
                    .map(std::string::ToString::to_string)
                    .collect()
            })
        };

        if invoked_cmd == "/agent"
            && matches!(
                tokens.first().map(|s| s.to_ascii_lowercase()).as_deref(),
                Some("defaults")
            )
        {
            tokens.remove(0);
            self.handle_agent_defaults_command(tokens, msg, bot, user_id)
                .await;
            return;
        }

        if invoked_cmd == "/agent"
            && matches!(
                tokens.first().map(|s| s.to_ascii_lowercase()).as_deref(),
                Some("flags")
            )
        {
            tokens.remove(0);
            self.handle_agent_flags_command(tokens, msg, bot, user_id)
                .await;
            return;
        }

        if matches!(
            tokens.first().map(|s| s.to_ascii_lowercase()).as_deref(),
            Some("lite")
        ) {
            if invoked_cmd == "/agent" {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Lite mode stays under `/terminal lite`.\nUse `/agent ...` for full Codex/Claude/Gemini sessions.",
                    )
                    .await;
                return;
            }
            tokens.remove(0);
            self.handle_terminal_lite_command(tokens, msg, bot, user_id)
                .await;
            return;
        }

        if let Some(first) = tokens.first() {
            let first = first.to_ascii_lowercase();
            if first == "help" {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        if invoked_cmd == "/agent" {
                            Self::agent_help_text()
                        } else {
                            Self::terminal_help_text()
                        },
                    )
                    .await;
                return;
            }
            if matches!(first.as_str(), "status" | "interrupt" | "stop") {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!(
                            "Use the Mini App toolbar for status/interrupt/stop in v1.\nRun {} open to reconnect.",
                            invoked_cmd
                        ),
                    )
                    .await;
                return;
            }
        }

        if invoked_cmd == "/agent"
            && matches!(
                tokens.first().map(|s| s.to_ascii_lowercase()).as_deref(),
                Some("resume")
            )
        {
            tokens.remove(0);
            let code = tokens
                .first()
                .map(|v| v.trim())
                .filter(|v| !v.is_empty())
                .map(|v| v.to_string());
            self.send_agent_resume_prompt(bot, msg.chat.id, user_id, code)
                .await;
            return;
        }

        if invoked_cmd == "/agent"
            && matches!(
                tokens.first().map(|s| s.to_ascii_lowercase()).as_deref(),
                Some("share")
            )
        {
            tokens.remove(0);
            let relay_session_id = tokens
                .first()
                .map(|v| v.trim())
                .filter(|v| !v.is_empty())
                .map(|v| v.to_string());
            self.send_agent_share_code(bot, msg.chat.id, user_id, relay_session_id)
                .await;
            return;
        }

        let mut start_requested = false;
        if let Some(first) = tokens.first() {
            let first = first.to_ascii_lowercase();
            if first == "open" || first == "start" {
                start_requested = first == "start";
                tokens.remove(0);
            }
        }

        let mut agent: Option<String> = None;
        let mut cwd_parts: Vec<String> = Vec::new();
        let mut agent_args: Vec<String> = Vec::new();
        let mut rewrote_permission_flag = false;
        let mut had_explicit_arg_delimiter = false;
        let mut used_saved_args = false;
        let mut saved_args_updated = false;

        // Backward-compat: `/terminal ... -- [flags...]`
        if let Some(idx) = tokens.iter().position(|t| t == "--") {
            had_explicit_arg_delimiter = true;
            if idx + 1 < tokens.len() {
                agent_args.extend(
                    tokens[(idx + 1)..]
                        .iter()
                        .map(|value| value.trim().to_string())
                        .filter(|value| !value.is_empty() && !value.contains('\0')),
                );
            }
            tokens.truncate(idx);
        }

        if let Some(first) = tokens.first() {
            let candidate = first.to_ascii_lowercase();
            if SUPPORTED_TERMINAL_AGENTS.contains(&candidate.as_str()) {
                agent = Some(candidate);
                tokens.remove(0);
            } else if first.starts_with('/')
                || first.starts_with('~')
                || first.starts_with('.')
                || first.contains('/')
            {
                agent = Some(DEFAULT_TERMINAL_AGENT.to_string());
            } else if first.starts_with('-') {
                // Flags-only launch defaults to codex.
                agent = Some(DEFAULT_TERMINAL_AGENT.to_string());
            } else {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!(
                            "Unknown terminal agent: `{}`\n\n{}",
                            first,
                            if invoked_cmd == "/agent" {
                                Self::agent_help_text()
                            } else {
                                Self::terminal_help_text()
                            }
                        ),
                    )
                    .await;
                return;
            }
            if had_explicit_arg_delimiter {
                for token in tokens {
                    let trimmed = token.trim();
                    if trimmed.is_empty() || trimmed.contains('\0') {
                        continue;
                    }
                    cwd_parts.push(trimmed.to_string());
                }
            } else if let Some(flag_idx) = tokens.iter().position(|token| token.starts_with('-')) {
                for token in &tokens[..flag_idx] {
                    let trimmed = token.trim();
                    if trimmed.is_empty() || trimmed.contains('\0') {
                        continue;
                    }
                    cwd_parts.push(trimmed.to_string());
                }
                for token in &tokens[flag_idx..] {
                    let trimmed = token.trim();
                    if trimmed.is_empty() || trimmed.contains('\0') {
                        continue;
                    }
                    agent_args.push(trimmed.to_string());
                }
            } else {
                for token in tokens {
                    let trimmed = token.trim();
                    if trimmed.is_empty() || trimmed.contains('\0') {
                        continue;
                    }
                    cwd_parts.push(trimmed.to_string());
                }
            }
        } else if start_requested {
            agent = Some(DEFAULT_TERMINAL_AGENT.to_string());
        } else if !agent_args.is_empty() {
            // Flags imply a terminal launch context; default agent to codex.
            agent = Some(DEFAULT_TERMINAL_AGENT.to_string());
        }
        agent_args = cli_agent_flags::normalize_terminal_agent_args(agent_args);
        let skip_saved_defaults = cli_agent_flags::strip_no_default_flag(&mut agent_args);
        let (normalized_agent_args, rewrote) =
            crate::normalize_terminal_agent_permission_aliases(agent.as_deref(), agent_args);
        agent_args = normalized_agent_args;
        rewrote_permission_flag |= rewrote;

        let namespace = self.session_namespace().await;
        if invoked_cmd == "/agent" && agent_args.is_empty() && !skip_saved_defaults {
            if let Some(agent_name) = agent.as_deref() {
                let defaults = cli_agent_flags::load_terminal_agent_defaults(
                    &*self.state,
                    &namespace,
                    msg.chat.id.0,
                    user_id,
                )
                .await;
                if let Some(saved) = defaults.get(agent_name) {
                    agent_args = cli_agent_flags::normalize_terminal_agent_args(saved.clone());
                    let (normalized_agent_args, rewrote) =
                        crate::normalize_terminal_agent_permission_aliases(
                            agent.as_deref(),
                            agent_args,
                        );
                    agent_args = normalized_agent_args;
                    rewrote_permission_flag |= rewrote;
                    used_saved_args = !agent_args.is_empty();
                }
            }
        } else if invoked_cmd == "/agent" && !agent_args.is_empty() {
            if let Some(agent_name) = agent.as_deref() {
                let mut defaults = cli_agent_flags::load_terminal_agent_defaults(
                    &*self.state,
                    &namespace,
                    msg.chat.id.0,
                    user_id,
                )
                .await;
                let changed = defaults.get(agent_name) != Some(&agent_args);
                if changed {
                    defaults.insert(agent_name.to_string(), agent_args.clone());
                    if let Err(err) = cli_agent_flags::save_terminal_agent_defaults(
                        &*self.state,
                        &namespace,
                        msg.chat.id.0,
                        user_id,
                        &defaults,
                    )
                    .await
                    {
                        warn!(error = %err, "Failed to persist /agent default flags");
                    } else {
                        saved_args_updated = true;
                    }
                }
            }
        }

        let cwd = if cwd_parts.is_empty() {
            None
        } else {
            Some(cwd_parts.join(" "))
        };
        let has_launch_intent = agent.is_some() || cwd.is_some() || !agent_args.is_empty();
        #[cfg(not(feature = "terminal-bridge"))]
        let _ = has_launch_intent;

        let current_bot_label = self.cached_bot_label();
        let mini_app_sync_warning = match self
            .sync_terminal_tenant_bot_token(&self.bot_token, user_id, current_bot_label.as_deref())
            .await
        {
            Ok(()) => None,
            Err(err) => {
                warn!(
                    error = %err,
                    user_id,
                    invoked_cmd,
                    "Failed to auto-sync Telegram bot token before opening Mini App"
                );
                Some(
                    "Mini App auth sync did not complete. If Telegram shows an auth mismatch, retry once or save the token in the app."
                        .to_string(),
                )
            }
        };

        #[cfg(feature = "terminal-bridge")]
        let prestarted_relay_session_id = if invoked_cmd == "/agent" && has_launch_intent {
            match crate::terminal_bridge::request_local_start_session(
                agent.as_deref().unwrap_or(DEFAULT_TERMINAL_AGENT),
                cwd.as_deref(),
                &agent_args,
            )
            .await
            {
                Ok(session_id) => Some(session_id),
                Err(err) => {
                    warn!(
                        error = %err,
                        chat_id = msg.chat.id.0,
                        invoked_cmd,
                        "Failed to pre-start local bridge session for Telegram Mini App"
                    );
                    None
                }
            }
        } else {
            None
        };
        #[cfg(not(feature = "terminal-bridge"))]
        let prestarted_relay_session_id: Option<String> = None;

        let mut web_app_url = match reqwest::Url::parse(self.terminal_web_app_url.trim()) {
            Ok(url) => url,
            Err(err) => {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!(
                            "Terminal web app URL is invalid in config: {}\nCurrent value: {}",
                            err, self.terminal_web_app_url
                        ),
                    )
                    .await;
                return;
            }
        };

        if web_app_url.scheme() != "https" {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    format!(
                        "Terminal web app URL must use HTTPS. Current value: {}",
                        self.terminal_web_app_url
                    ),
                )
                .await;
            return;
        }

        let telegram_session_id = self.session_id(msg.chat.id.0).await;
        {
            let mut query = web_app_url.query_pairs_mut();
            query.append_pair("telegram_session_id", &telegram_session_id);
            if let Some(agent) = agent.as_deref() {
                query.append_pair("agent", agent);
            }
            if let Some(cwd) = cwd.as_deref() {
                if !cwd.trim().is_empty() {
                    query.append_pair("cwd", cwd);
                }
            }
            if let Some(relay_session_id) = prestarted_relay_session_id.as_deref() {
                query.append_pair("relay_session_id", relay_session_id);
            }
            for arg in &agent_args {
                query.append_pair("arg", arg);
            }
            if start_requested {
                query.append_pair("autostart", "1");
            }
        }

        let mini_app_host = web_app_url.host_str().unwrap_or("unknown");
        let mini_app_base = if let Some(port) = web_app_url.port() {
            format!("{}://{}:{}", web_app_url.scheme(), mini_app_host, port)
        } else {
            format!("{}://{}", web_app_url.scheme(), mini_app_host)
        };

        let mut summary_lines = vec![
            if invoked_cmd == "/agent" {
                "🤖 <b>Agent Session</b>".to_string()
            } else {
                "🖥️ <b>Terminal Mode</b>".to_string()
            },
            String::new(),
            format!(
                "Mini App host: <code>{}</code>",
                html_escape(&mini_app_base)
            ),
        ];
        if let Some(agent) = agent.as_deref() {
            summary_lines.push(format!("Agent: <code>{}</code>", html_escape(agent)));
        } else {
            summary_lines.push(format!(
                "Agent: choose in app (default {})",
                DEFAULT_TERMINAL_AGENT
            ));
        }
        if let Some(cwd) = cwd.as_deref() {
            let folder_name = Path::new(cwd)
                .file_name()
                .and_then(|value| value.to_str())
                .filter(|value| !value.is_empty())
                .unwrap_or("custom");
            summary_lines.push(format!(
                "Working dir: <code>{}</code> (full path sent only to Mini App)",
                html_escape(folder_name)
            ));
        }
        if !agent_args.is_empty() {
            summary_lines.push(format!(
                "Agent args: <code>{}</code> (values sent only to Mini App)",
                agent_args.len()
            ));
            if used_saved_args {
                summary_lines.push(
                    "Using saved defaults for this chat. Add <code>--no-default-flags</code> to bypass once."
                        .to_string(),
                );
            } else if saved_args_updated {
                summary_lines.push("Saved as new defaults for this chat and agent.".to_string());
            }
            if rewrote_permission_flag {
                summary_lines.push(
                    "Normalized <code>--allow-dangerously-skip-permissions</code> to <code>--dangerously-skip-permissions</code> for Claude."
                        .to_string(),
                );
            }
        }
        if prestarted_relay_session_id.is_some() {
            summary_lines.push(
                "Prepared a local relay session in advance so the Mini App can attach immediately."
                    .to_string(),
            );
        }
        summary_lines.push(String::new());
        summary_lines.push(
            if invoked_cmd == "/agent" {
                "Tap 📱 Open in Mini App to launch the encrypted agent UI."
            } else {
                "Tap 🖥️ Open Terminal to launch the encrypted session UI."
            }
            .to_string(),
        );
        if let Some(warning) = mini_app_sync_warning.as_deref() {
            summary_lines.push(warning.to_string());
        }
        if invoked_cmd == "/agent" {
            summary_lines.push(
                "💻 Continue on Computer sends a one-time resume code (expires in about 5 minutes)."
                    .to_string(),
            );
        }

        let open_button = InlineKeyboardButton::web_app(
            if invoked_cmd == "/agent" {
                "📱 Open in Mini App"
            } else {
                "🖥️ Open Terminal"
            },
            WebAppInfo {
                url: web_app_url.clone(),
            },
        );
        let keyboard = if invoked_cmd == "/agent" {
            InlineKeyboardMarkup::new(vec![vec![
                open_button,
                InlineKeyboardButton::callback("💻 Continue on Computer", "agent:share"),
            ]])
        } else {
            InlineKeyboardMarkup::new(vec![vec![open_button]])
        };

        let html_message = summary_lines.join("\n");
        let plain_message = format!(
            "{}\nMini App host: {}\n\nUse the Open button to launch. Continue on Computer generates a one-time code that expires in about 5 minutes.",
            if invoked_cmd == "/agent" {
                "Agent session"
            } else {
                "Terminal mode"
            },
            mini_app_base
        );

        if bot
            .send_message(msg.chat.id, html_message)
            .parse_mode(ParseMode::Html)
            .reply_markup(keyboard.clone())
            .await
            .is_err()
        {
            let _ = bot
                .send_message(msg.chat.id, plain_message)
                .reply_markup(keyboard)
                .await;
        }

        if invoked_cmd == "/agent" {
            let reply_keyboard =
                KeyboardMarkup::new(vec![vec![KeyboardButton::new("📱 Open in Mini App")
                    .request(ButtonRequest::WebApp(WebAppInfo {
                        url: web_app_url.clone(),
                    }))]])
                .resize_keyboard()
                .one_time_keyboard();

            let _ = bot
                .send_message(
                    msg.chat.id,
                    "Tip: if in-app handoff buttons look disabled, open from this keyboard button. Telegram only enables Mini App send-back in keyboard launch mode.",
                )
                .reply_markup(reply_keyboard)
                .await;
        }
    }

    /// Handle an incoming file/photo/audio/video/voice message.
    /// Downloads the file, saves to inbox, and returns attachment metadata.
    async fn handle_file_message(
        &self,
        msg: &teloxide::types::Message,
        bot: &Bot,
    ) -> anyhow::Result<crate::traits::MessageAttachment> {
        // Extract file_id, file_size, filename, and mime_type from the message
        let (file_id, file_size, filename, mime_type) = if let Some(doc) = msg.document() {
            (
                doc.file.id.clone(),
                doc.file.size as u64,
                doc.file_name
                    .clone()
                    .unwrap_or_else(|| "document".to_string()),
                doc.mime_type
                    .as_ref()
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "application/octet-stream".to_string()),
            )
        } else if let Some(photos) = msg.photo() {
            // Last photo in the array is the largest
            let photo = photos
                .last()
                .ok_or_else(|| anyhow::anyhow!("Empty photo array"))?;
            (
                photo.file.id.clone(),
                photo.file.size as u64,
                "photo.jpg".to_string(),
                "image/jpeg".to_string(),
            )
        } else if let Some(audio) = msg.audio() {
            (
                audio.file.id.clone(),
                audio.file.size as u64,
                audio
                    .file_name
                    .clone()
                    .unwrap_or_else(|| "audio.mp3".to_string()),
                audio
                    .mime_type
                    .as_ref()
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "audio/mpeg".to_string()),
            )
        } else if let Some(video) = msg.video() {
            (
                video.file.id.clone(),
                video.file.size as u64,
                video
                    .file_name
                    .clone()
                    .unwrap_or_else(|| "video.mp4".to_string()),
                video
                    .mime_type
                    .as_ref()
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "video/mp4".to_string()),
            )
        } else if let Some(voice) = msg.voice() {
            (
                voice.file.id.clone(),
                voice.file.size as u64,
                "voice.ogg".to_string(),
                voice
                    .mime_type
                    .as_ref()
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "audio/ogg".to_string()),
            )
        } else {
            anyhow::bail!("Unsupported message type. I can process text, files, photos, audio, video, and voice messages.");
        };

        // Check file size before downloading
        let max_bytes = self.max_file_size_mb * 1_048_576;
        if file_size > max_bytes {
            anyhow::bail!(
                "File too large ({:.1} MB). Maximum is {} MB.",
                file_size as f64 / 1_048_576.0,
                self.max_file_size_mb
            );
        }

        // Get file info from Telegram
        let file = bot.get_file(file_id).await?;
        let file_path_on_server = file.path;

        // Download via HTTP (simpler than teloxide's Download trait)
        let download_url = format!(
            "https://api.telegram.org/file/bot{}/{}",
            self.bot_token, file_path_on_server
        );
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap_or_default();
        let response = client.get(&download_url).send().await?;
        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download file from Telegram: HTTP {}",
                response.status()
            );
        }
        let bytes = response.bytes().await?;

        let (filename, mime_type) = if msg.photo().is_some() {
            super::attachments::telegram_photo_filename_and_mime(&bytes)
        } else {
            (filename, mime_type)
        };

        // Sanitize filename: strip path separators, null bytes, limit length
        let sanitized = sanitize_filename(&filename);
        let uuid_prefix = uuid::Uuid::new_v4().to_string()[..8].to_string();
        let dest_name = format!("{}_{}", uuid_prefix, sanitized);
        let dest_path = self.inbox_dir.join(&dest_name);

        // Ensure inbox directory exists
        std::fs::create_dir_all(&self.inbox_dir)?;

        // Write file
        std::fs::write(&dest_path, &bytes)?;

        info!(
            file = %dest_path.display(),
            size = bytes.len(),
            mime = %mime_type,
            "Saved inbound file"
        );

        Ok(super::attachments::message_attachment(
            dest_path,
            sanitized,
            mime_type,
            bytes.len() as u64,
        ))
    }

    /// Handle /setup command for owner-only operational setup workflows.
    /// Usage:
    ///   /setup lowlatency status
    ///   /setup lowlatency reauth
    ///   /setup lowlatency plan <base-domain>
    ///   /setup lowlatency apply <base-domain>
    async fn handle_setup_command(&self, arg: &str, user_id: u64, chat_id: i64) -> String {
        if determine_role(&self.owner_user_ids, user_id) != UserRole::Owner {
            return "Only the owner can run /setup commands.".to_string();
        }

        let usage = "Setup commands (owner only)\n\n\
            Usage:\n\
            /setup lowlatency status\n\
            /setup lowlatency reauth\n\
            /setup lowlatency plan <base-domain>\n\
            /setup lowlatency apply <base-domain>\n\n\
            Examples:\n\
            /setup lowlatency status\n\
            /setup lowlatency reauth\n\
            /setup lowlatency plan bots.example.com\n\
            /setup lowlatency apply bots.example.com\n\n\
            Notes:\n\
            - `plan` is dry-run only (no config changes).\n\
            - `apply` updates local webhook config for all Telegram bots and keeps polling fallback enabled.\n\
            - If `cloudflared` or `wrangler` are missing, `apply` can ask to install them.\n\
            - If `cloudflared` or `wrangler` are not authenticated, `apply` can ask to run login flows.\n\
            - `apply` verifies your `cloudflared` zone context before DNS routing.\n\
            - `reauth` runs `cloudflared tunnel login` to refresh zone/account selection.\n\
            - `apply` asks for approval before running Cloudflare tunnel/DNS commands.\n\
            - When approved and provisioning succeeds, `apply` starts the tunnel and runs `/restart` automatically.\n\
            - If denied, the command returns manual next steps."
            .to_string();

        let mut parts: Vec<&str> = arg.split_whitespace().collect();
        if parts.is_empty() {
            return usage;
        }

        if matches!(
            parts.first().map(|v| v.to_ascii_lowercase()),
            Some(value) if value == "lowlatency" || value == "low-latency" || value == "telegram-webhook"
        ) {
            let _ = parts.remove(0);
        }
        if parts.is_empty() {
            return usage;
        }

        let action = parts[0].to_ascii_lowercase();
        let config_path = self.config_path.clone();

        match action.as_str() {
            "status" => match tokio::task::spawn_blocking({
                let config_path = config_path.clone();
                move || crate::wizard::low_latency_status_summary(&config_path)
            })
            .await
            {
                Ok(Ok(text)) => text,
                Ok(Err(err)) => format!("Setup status failed: {}", err),
                Err(err) => format!("Setup status task failed: {}", err),
            },
            "reauth" => {
                if !crate::wizard::command_exists("cloudflared").await {
                    return "Cloudflared is not installed.\nInstall it first (for example `brew install cloudflared`), then rerun `/setup lowlatency reauth`.".to_string();
                }
                let session_id = self.session_id(chat_id).await;
                let login_spec = crate::wizard::SetupCommandSpec {
                    program: "cloudflared".to_string(),
                    args: vec!["tunnel".to_string(), "login".to_string()],
                };
                let login_approval_text = format!(
                    "Run cloudflared re-authentication now?\n\n{}",
                    crate::wizard::format_setup_command(&login_spec)
                );
                let login_warnings = vec![
                    "This opens a one-time Cloudflare authorization flow.".to_string(),
                    "Use this when DNS routing is targeting the wrong zone/account.".to_string(),
                ];
                let login_approval = match self
                    .request_approval(
                        &session_id,
                        &login_approval_text,
                        RiskLevel::High,
                        &login_warnings,
                        PermissionMode::Cautious,
                    )
                    .await
                {
                    Ok(value) => value,
                    Err(err) => {
                        return format!(
                            "Could not request approval for cloudflared re-authentication: {}\nRun `cloudflared tunnel login` manually if needed.",
                            err
                        )
                    }
                };
                if matches!(login_approval, ApprovalResponse::Deny) {
                    return "Cloudflared re-authentication skipped (approval denied).".to_string();
                }
                match self.run_cloudflared_login_and_verify(chat_id).await {
                    Ok(report) => format!("Cloudflared re-authentication completed.\n{}", report),
                    Err(err) => format!(
                        "{}\nRun `cloudflared tunnel login` manually, ensure it succeeds, then rerun your setup command.",
                        err
                    ),
                }
            }
            "plan" => {
                let Some(domain) = parts.get(1).map(|v| v.to_string()) else {
                    return "Usage: /setup lowlatency plan <base-domain>\nExample: /setup lowlatency plan bots.example.com".to_string();
                };
                let dynamic_bots = self.state.get_dynamic_bots().await.unwrap_or_default();
                let config_path = config_path.clone();
                match tokio::task::spawn_blocking(move || {
                    crate::wizard::low_latency_plan_from_base_domain_with_dynamic(
                        &config_path,
                        &domain,
                        &dynamic_bots,
                    )
                })
                .await
                {
                    Ok(Ok(text)) => text,
                    Ok(Err(err)) => format!("Setup plan failed: {}", err),
                    Err(err) => format!("Setup plan task failed: {}", err),
                }
            }
            "apply" => {
                let Some(domain) = parts.get(1).map(|v| v.to_string()) else {
                    return "Usage: /setup lowlatency apply <base-domain>\nExample: /setup lowlatency apply bots.example.com".to_string();
                };
                let dynamic_bots = self.state.get_dynamic_bots().await.unwrap_or_default();
                let apply_text = match tokio::task::spawn_blocking({
                    let config_path = config_path.clone();
                    let domain = domain.clone();
                    let dynamic_bots = dynamic_bots.clone();
                    move || {
                        crate::wizard::low_latency_apply_from_base_domain_with_dynamic(
                            &config_path,
                            &domain,
                            &dynamic_bots,
                        )
                    }
                })
                .await
                {
                    Ok(Ok(text)) => text,
                    Ok(Err(err)) => return format!("Setup apply failed: {}", err),
                    Err(err) => return format!("Setup apply task failed: {}", err),
                };

                let command_specs = match tokio::task::spawn_blocking({
                    let config_path = config_path.clone();
                    let domain = domain.clone();
                    let dynamic_bots = dynamic_bots.clone();
                    move || {
                        crate::wizard::low_latency_cloudflared_commands_from_base_domain_with_dynamic(
                            &config_path,
                            &domain,
                            &dynamic_bots,
                        )
                    }
                })
                .await
                {
                    Ok(Ok(cmds)) => cmds,
                    Ok(Err(err)) => {
                        return format!(
                            "{}\n\nCould not prepare Cloudflare commands: {}\nRun manual next steps above, then /restart to apply webhook listeners.",
                            apply_text, err
                        )
                    }
                    Err(err) => {
                        return format!(
                            "{}\n\nSetup command preparation failed: {}\nRun manual next steps above, then /restart to apply webhook listeners.",
                            apply_text, err
                        )
                    }
                };

                if command_specs.is_empty() {
                    return format!(
                        "{}\n\nNo Cloudflare commands were generated.\nRun manual next steps above, then /restart to apply webhook listeners.",
                        apply_text
                    );
                }

                let session_id = self.session_id(chat_id).await;
                let has_brew = crate::wizard::command_exists("brew").await;
                let has_npm = crate::wizard::command_exists("npm").await;
                let mut install_specs: Vec<(String, crate::wizard::SetupCommandSpec)> = Vec::new();
                let cloudflared_present = crate::wizard::command_exists("cloudflared").await;
                if !cloudflared_present {
                    if let Some(spec) =
                        crate::wizard::installer_for_missing_tool("cloudflared", has_brew, has_npm)
                    {
                        install_specs.push(("cloudflared".to_string(), spec));
                    } else {
                        return format!(
                            "{}\n\n`cloudflared` is not installed and no supported automatic installer was detected.\nInstall it manually (for example `brew install cloudflared`), then rerun /setup lowlatency apply <base-domain>.",
                            apply_text
                        );
                    }
                }
                let wrangler_present = crate::wizard::command_exists("wrangler").await;
                if !wrangler_present {
                    if let Some(spec) =
                        crate::wizard::installer_for_missing_tool("wrangler", has_brew, has_npm)
                    {
                        install_specs.push(("wrangler".to_string(), spec));
                    }
                }

                let mut install_report = String::new();
                if !install_specs.is_empty() {
                    let install_preview = install_specs
                        .iter()
                        .map(|(tool, spec)| {
                            format!("{}: {}", tool, crate::wizard::format_setup_command(spec))
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let install_approval_text = format!(
                        "Install missing setup dependencies now?\n\n{}",
                        install_preview
                    );
                    let install_warnings = vec![
                        "This will install CLI tools on your host.".to_string(),
                        "Deny to keep manual setup.".to_string(),
                    ];
                    let install_approval = match self
                        .request_approval(
                            &session_id,
                            &install_approval_text,
                            RiskLevel::High,
                            &install_warnings,
                            PermissionMode::Cautious,
                        )
                        .await
                    {
                        Ok(value) => value,
                        Err(err) => {
                            return format!(
                                "{}\n\nCould not request approval for dependency installation: {}\nRun manual next steps above, then /restart to apply webhook listeners.",
                                apply_text, err
                            )
                        }
                    };

                    if matches!(install_approval, ApprovalResponse::Deny) {
                        if !cloudflared_present {
                            return format!(
                                "{}\n\nAutomatic setup skipped because install approval was denied and `cloudflared` is required.\nInstall `cloudflared` manually, then rerun /setup lowlatency apply <base-domain>.",
                                apply_text
                            );
                        }
                        install_report =
                            "Dependency installation skipped (approval denied).".to_string();
                    } else {
                        let specs: Vec<crate::wizard::SetupCommandSpec> =
                            install_specs.iter().map(|(_, spec)| spec.clone()).collect();
                        let (lines, had_failures) =
                            Self::run_setup_commands(&specs, 600, false).await;
                        install_report =
                            format!("Dependency installation results:\n{}", lines.join("\n"));
                        if had_failures {
                            let cloudflared_after =
                                crate::wizard::command_exists("cloudflared").await;
                            if !cloudflared_after {
                                return format!(
                                    "{}\n\n{}\n\n`cloudflared` is still missing, so automatic provisioning cannot continue.\nInstall it manually and rerun /setup lowlatency apply <base-domain>.",
                                    apply_text, install_report
                                );
                            }
                        }
                    }
                }

                let mut cloudflared_auth_report = String::new();
                let mut wrangler_auth_report = String::new();
                let cloudflared_available = crate::wizard::command_exists("cloudflared").await;
                if cloudflared_available
                    && !crate::wizard::command_succeeds("cloudflared", &["tunnel", "list"], 20)
                        .await
                {
                    let login_spec = crate::wizard::SetupCommandSpec {
                        program: "cloudflared".to_string(),
                        args: vec!["tunnel".to_string(), "login".to_string()],
                    };
                    let login_preview = crate::wizard::format_setup_command(&login_spec);
                    let login_approval_text = format!(
                        "Cloudflared is not authenticated. Run login now?\n\n{}",
                        login_preview
                    );
                    let login_warnings = vec![
                        "This opens a one-time Cloudflare authorization flow.".to_string(),
                        "You may need to open a URL from your browser/phone and approve access."
                            .to_string(),
                    ];
                    let login_approval = match self
                        .request_approval(
                            &session_id,
                            &login_approval_text,
                            RiskLevel::High,
                            &login_warnings,
                            PermissionMode::Cautious,
                        )
                        .await
                    {
                        Ok(value) => value,
                        Err(err) => {
                            return format!(
                                "{}\n\nCould not request approval for cloudflared authentication: {}\nRun `cloudflared tunnel login` manually, then rerun /setup lowlatency apply <base-domain>.",
                                apply_text, err
                            )
                        }
                    };
                    if matches!(login_approval, ApprovalResponse::Deny) {
                        return format!(
                            "{}\n\nAutomatic setup skipped because cloudflared authentication was denied.\nRun `cloudflared tunnel login` manually, then rerun /setup lowlatency apply <base-domain>.",
                            apply_text
                        );
                    }
                    match self.run_cloudflared_login_and_verify(chat_id).await {
                        Ok(report) => {
                            cloudflared_auth_report = report;
                        }
                        Err(err) => {
                            return format!(
                                "{}\n\n{}\nRun `cloudflared tunnel login` manually, ensure it succeeds, then rerun /setup lowlatency apply <base-domain>.",
                                apply_text, err
                            );
                        }
                    }
                }

                let route_hosts = crate::wizard::setup_route_dns_hosts(&command_specs);
                match crate::wizard::validate_cloudflared_zone_for_hosts(&route_hosts).await {
                    CloudflaredZoneValidation::Match { zone_name } => {
                        let note = format!("Cloudflared zone context verified: `{}`.", zone_name);
                        if cloudflared_auth_report.is_empty() {
                            cloudflared_auth_report = note;
                        } else {
                            cloudflared_auth_report =
                                format!("{}\n{}", cloudflared_auth_report, note);
                        }
                    }
                    CloudflaredZoneValidation::Unknown { reason } => {
                        if !reason.is_empty() {
                            let note = format!("Cloudflared zone preflight: {}.", reason);
                            if cloudflared_auth_report.is_empty() {
                                cloudflared_auth_report = note;
                            } else {
                                cloudflared_auth_report =
                                    format!("{}\n{}", cloudflared_auth_report, note);
                            }
                        }
                    }
                    CloudflaredZoneValidation::Mismatch {
                        zone_name,
                        mismatched_hosts,
                    } => {
                        let sample = mismatched_hosts
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "unknown".to_string());
                        let login_spec = crate::wizard::SetupCommandSpec {
                            program: "cloudflared".to_string(),
                            args: vec!["tunnel".to_string(), "login".to_string()],
                        };
                        let reauth_prompt = format!(
                            "Cloudflared is authenticated for zone `{}`, but planned webhook host `{}` is outside that zone.\nRe-authenticate now to select the correct zone/account?\n\n{}",
                            zone_name,
                            sample,
                            crate::wizard::format_setup_command(&login_spec)
                        );
                        let warnings = vec![
                            "Without re-auth, DNS records can be created in the wrong zone."
                                .to_string(),
                            "Deny to abort automatic provisioning safely.".to_string(),
                        ];
                        let approval = match self
                            .request_approval(
                                &session_id,
                                &reauth_prompt,
                                RiskLevel::High,
                                &warnings,
                                PermissionMode::Cautious,
                            )
                            .await
                        {
                            Ok(value) => value,
                            Err(err) => {
                                return format!(
                                    "{}\n\nCloudflared zone mismatch detected: current zone `{}` does not include planned host(s): {}.\nCould not request re-auth approval: {}\nAutomatic provisioning stopped to prevent wrong DNS records.\nRun `cloudflared tunnel login` with the correct account/zone, then rerun /setup lowlatency apply <base-domain>.",
                                    apply_text,
                                    zone_name,
                                    mismatched_hosts.join(", "),
                                    err
                                )
                            }
                        };
                        if matches!(approval, ApprovalResponse::Deny) {
                            return format!(
                                "{}\n\nCloudflared zone mismatch detected: current zone `{}` does not include planned host(s): {}.\nAutomatic provisioning stopped to prevent wrong DNS records.\nRun `cloudflared tunnel login` with the correct account/zone, then rerun /setup lowlatency apply <base-domain>.",
                                apply_text,
                                zone_name,
                                mismatched_hosts.join(", ")
                            );
                        }
                        let reauth_report = match self
                            .run_cloudflared_login_and_verify(chat_id)
                            .await
                        {
                            Ok(report) => report,
                            Err(err) => {
                                return format!(
                                    "{}\n\n{}\nRun `cloudflared tunnel login` manually with the correct account/zone, then rerun /setup lowlatency apply <base-domain>.",
                                    apply_text, err
                                );
                            }
                        };
                        match crate::wizard::validate_cloudflared_zone_for_hosts(&route_hosts).await
                        {
                            CloudflaredZoneValidation::Match {
                                zone_name: refreshed_zone,
                            } => {
                                let note = format!(
                                    "Cloudflared re-authentication completed.\n{}\nCloudflared zone context verified: `{}`.",
                                    reauth_report, refreshed_zone
                                );
                                if cloudflared_auth_report.is_empty() {
                                    cloudflared_auth_report = note;
                                } else {
                                    cloudflared_auth_report =
                                        format!("{}\n{}", cloudflared_auth_report, note);
                                }
                            }
                            CloudflaredZoneValidation::Mismatch {
                                zone_name: refreshed_zone,
                                mismatched_hosts: refreshed_hosts,
                            } => {
                                return format!(
                                    "{}\n\nCloudflared re-authentication completed but zone mismatch remains.\nCurrent zone: `{}`\nPlanned host(s): {}\nAutomatic provisioning stopped to prevent wrong DNS records.\nRun `cloudflared tunnel login` again with the correct account/zone, then rerun /setup lowlatency apply <base-domain>.",
                                    apply_text,
                                    refreshed_zone,
                                    refreshed_hosts.join(", ")
                                );
                            }
                            CloudflaredZoneValidation::Unknown { reason } => {
                                let note = if reason.is_empty() {
                                    format!(
                                        "Cloudflared re-authentication completed.\n{}",
                                        reauth_report
                                    )
                                } else {
                                    format!(
                                        "Cloudflared re-authentication completed.\n{}\nCloudflared zone preflight after re-auth: {}.",
                                        reauth_report, reason
                                    )
                                };
                                if cloudflared_auth_report.is_empty() {
                                    cloudflared_auth_report = note;
                                } else {
                                    cloudflared_auth_report =
                                        format!("{}\n{}", cloudflared_auth_report, note);
                                }
                            }
                        }
                    }
                }

                let wrangler_available = crate::wizard::command_exists("wrangler").await;
                if wrangler_available
                    && !crate::wizard::command_succeeds("wrangler", &["whoami", "--json"], 20).await
                {
                    let login_spec = crate::wizard::SetupCommandSpec {
                        program: "wrangler".to_string(),
                        args: vec!["login".to_string()],
                    };
                    let login_preview = crate::wizard::format_setup_command(&login_spec);
                    let login_approval_text = format!(
                        "Wrangler is not authenticated. Run login now?\n\n{}",
                        login_preview
                    );
                    let login_warnings = vec![
                        "This opens a one-time Cloudflare authorization flow for wrangler."
                            .to_string(),
                        "You may need to open a URL from your browser/phone and approve access."
                            .to_string(),
                    ];
                    let login_approval = match self
                        .request_approval(
                            &session_id,
                            &login_approval_text,
                            RiskLevel::High,
                            &login_warnings,
                            PermissionMode::Cautious,
                        )
                        .await
                    {
                        Ok(value) => value,
                        Err(err) => {
                            wrangler_auth_report = format!(
                                "Wrangler authentication was skipped (approval request failed: {}).",
                                err
                            );
                            ApprovalResponse::Deny
                        }
                    };

                    if matches!(login_approval, ApprovalResponse::Deny) {
                        if wrangler_auth_report.is_empty() {
                            wrangler_auth_report =
                                "Wrangler authentication skipped (approval denied).".to_string();
                        }
                    } else {
                        let login_result = self
                            .run_login_command_with_live_output(
                                chat_id,
                                &login_spec,
                                600,
                                "Starting wrangler authentication flow.",
                            )
                            .await;
                        if login_result.timed_out {
                            wrangler_auth_report =
                                "Wrangler login timed out after 600s; finish `wrangler login` manually if you need wrangler actions."
                                    .to_string();
                        } else if let Some(err) = login_result.error {
                            wrangler_auth_report = format!(
                                "Wrangler login failed: {}. Run `wrangler login` manually if needed.",
                                err
                            );
                        } else {
                            let wrangler_ok = login_result.success
                                && crate::wizard::command_succeeds(
                                    "wrangler",
                                    &["whoami", "--json"],
                                    20,
                                )
                                .await;
                            let summary =
                                crate::wizard::summarize_setup_log_lines(&login_result.lines);
                            if wrangler_ok {
                                wrangler_auth_report = if let Some(url) = login_result.urls.first()
                                {
                                    if summary.is_empty() {
                                        format!("Wrangler authentication completed.\nURL: {}", url)
                                    } else {
                                        format!(
                                            "Wrangler authentication completed.\nURL: {}\n{}",
                                            url, summary
                                        )
                                    }
                                } else if summary.is_empty() {
                                    "Wrangler authentication completed.".to_string()
                                } else {
                                    format!("Wrangler authentication completed.\n{}", summary)
                                };
                            } else {
                                wrangler_auth_report = if summary.is_empty() {
                                    "Wrangler authentication did not complete; run `wrangler login` manually if needed.".to_string()
                                } else {
                                    format!(
                                        "Wrangler authentication did not complete.\n{}\nRun `wrangler login` manually if needed.",
                                        summary
                                    )
                                };
                            }
                        }
                    }
                }

                let command_preview = command_specs
                    .iter()
                    .map(crate::wizard::format_setup_command)
                    .collect::<Vec<_>>()
                    .join("\n");
                let approval_command = format!(
                    "Run Cloudflare provisioning commands now for `{}`?\n\n{}",
                    domain, command_preview
                );
                let warnings = vec![
                    "This will execute cloudflared commands on your host.".to_string(),
                    "Deny to keep the manual setup flow.".to_string(),
                ];
                let approval = match self
                    .request_approval(
                        &session_id,
                        &approval_command,
                        RiskLevel::High,
                        &warnings,
                        PermissionMode::Cautious,
                    )
                    .await
                {
                    Ok(value) => value,
                    Err(err) => {
                        return format!(
                            "{}\n\nCould not request approval for automatic Cloudflare setup: {}\nRun manual next steps above, then /restart to apply webhook listeners.",
                            apply_text, err
                        )
                    }
                };
                if matches!(approval, ApprovalResponse::Deny) {
                    let mut out = apply_text;
                    if !install_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&install_report);
                    }
                    if !cloudflared_auth_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&cloudflared_auth_report);
                    }
                    if !wrangler_auth_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&wrangler_auth_report);
                    }
                    out.push_str(
                        "\n\nAutomatic Cloudflare setup skipped (approval denied).\nRun manual next steps above, then /restart to apply webhook listeners.",
                    );
                    out
                } else {
                    let (result_lines, had_failures) =
                        Self::run_setup_commands(&command_specs, 60, true).await;
                    let mut out = apply_text.clone();
                    let mut automatic_tunnel_startup_ok = false;
                    let mut automatic_tunnel_startup_failed = false;
                    if !install_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&install_report);
                    }
                    if !cloudflared_auth_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&cloudflared_auth_report);
                    }
                    if !wrangler_auth_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&wrangler_auth_report);
                    }
                    if had_failures {
                        out.push_str(
                            "\n\nAutomatic Cloudflare provisioning finished with failures:\n",
                        );
                        out.push_str(&result_lines.join("\n"));
                        out.push_str(
                            "\n\nRun the failed command(s) manually, then start your tunnel process and run /restart to apply webhook listeners.",
                        );
                    } else {
                        out.push_str("\n\nAutomatic Cloudflare provisioning completed:\n");
                        out.push_str(&result_lines.join("\n"));
                        let ingress_routes = match tokio::task::spawn_blocking({
                            let config_path = config_path.clone();
                            let domain = domain.clone();
                            let dynamic_bots = dynamic_bots.clone();
                            move || {
                                crate::wizard::low_latency_cloudflared_ingress_routes_from_base_domain_with_dynamic(
                                    &config_path,
                                    &domain,
                                    &dynamic_bots,
                                )
                            }
                        })
                        .await
                        {
                            Ok(Ok(routes)) => routes,
                            Ok(Err(err)) => {
                                automatic_tunnel_startup_failed = true;
                                out.push_str(
                                    "\n\nAutomatic tunnel startup failed:\n",
                                );
                                out.push_str(&format!(
                                    "Could not derive ingress routes from config: {}",
                                    err
                                ));
                                out.push_str(
                                    "\nRun `cloudflared tunnel run aidaemon-telegram` manually after reviewing your tunnel config.",
                                );
                                Vec::new()
                            }
                            Err(err) => {
                                automatic_tunnel_startup_failed = true;
                                out.push_str(
                                    "\n\nAutomatic tunnel startup failed:\n",
                                );
                                out.push_str(&format!(
                                    "Ingress route task failed: {}",
                                    err
                                ));
                                out.push_str(
                                    "\nRun `cloudflared tunnel run aidaemon-telegram` manually after reviewing your tunnel config.",
                                );
                                Vec::new()
                            }
                        };

                        if !ingress_routes.is_empty() {
                            match crate::wizard::start_cloudflared_tunnel_background(
                                "aidaemon-telegram",
                                &ingress_routes,
                            )
                            .await
                            {
                                Ok(tunnel_status) => {
                                    automatic_tunnel_startup_ok = true;
                                    out.push_str("\n\nAutomatic tunnel startup completed:\n");
                                    out.push_str(&tunnel_status);
                                }
                                Err(err) => {
                                    automatic_tunnel_startup_failed = true;
                                    out.push_str("\n\nAutomatic tunnel startup failed:\n");
                                    out.push_str(&format!(
                                        "FAIL cloudflared tunnel run aidaemon-telegram - {}",
                                        err
                                    ));
                                    out.push_str(
                                        "\nRun `cloudflared tunnel run aidaemon-telegram` manually after reviewing your tunnel config.",
                                    );
                                }
                            }
                        }

                        let restart_scheduled =
                            Self::schedule_low_latency_restart(Duration::from_secs(3));

                        // When the full automation succeeds end-to-end, replace
                        // verbose output with a concise success summary.
                        if automatic_tunnel_startup_ok && !automatic_tunnel_startup_failed {
                            // Extract public_url lines from the original apply text
                            let webhook_urls: Vec<&str> = apply_text
                                .lines()
                                .filter(|l| l.trim_start().starts_with("public_url:"))
                                .map(|l| l.trim())
                                .collect();
                            out = String::new();
                            out.push_str("Webhook mode setup complete.\n");
                            if !webhook_urls.is_empty() {
                                out.push('\n');
                                for url in &webhook_urls {
                                    out.push_str(&format!("  {}\n", url));
                                }
                            }
                            out.push_str("\nTunnel running. Restarting daemon to switch from polling to webhooks...");
                        } else if restart_scheduled {
                            out.push_str(
                                "\n\nAutomatic daemon restart scheduled in ~3s to activate Telegram webhook listeners.",
                            );
                        } else {
                            out.push_str(
                                "\n\nDaemon restart was already scheduled. Webhook listeners will be activated shortly.",
                            );
                        }
                    }
                    out
                }
            }
            "help" => usage,
            _ => usage,
        }
    }

    #[cfg(test)]
    fn strip_low_latency_next_steps(text: &str) -> String {
        let marker = "\nNext steps:\n";
        if let Some(index) = text.find(marker) {
            text[..index].trim_end().to_string()
        } else {
            text.to_string()
        }
    }

    async fn backup_cloudflared_origin_cert_if_present(
        cert_path: &Path,
    ) -> Result<Option<PathBuf>, String> {
        let metadata = match tokio::fs::metadata(cert_path).await {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(format!(
                    "failed to inspect cloudflared cert at `{}`: {}",
                    cert_path.display(),
                    err
                ))
            }
        };
        if !metadata.is_file() {
            return Err(format!(
                "cloudflared cert path `{}` exists but is not a file",
                cert_path.display()
            ));
        }

        let parent = cert_path.parent().ok_or_else(|| {
            format!(
                "cannot determine parent directory for cloudflared cert `{}`",
                cert_path.display()
            )
        })?;
        let file_name = cert_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                format!(
                    "cannot derive file name for cloudflared cert `{}`",
                    cert_path.display()
                )
            })?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for attempt in 0..20 {
            let backup_name = if attempt == 0 {
                format!("{}.bak.{}", file_name, timestamp)
            } else {
                format!("{}.bak.{}.{}", file_name, timestamp, attempt)
            };
            let backup_path = parent.join(backup_name);
            match tokio::fs::metadata(&backup_path).await {
                Ok(_) => continue,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => {
                    return Err(format!(
                        "failed to inspect cloudflared backup path `{}`: {}",
                        backup_path.display(),
                        err
                    ))
                }
            }

            match tokio::fs::rename(cert_path, &backup_path).await {
                Ok(_) => return Ok(Some(backup_path)),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(err) => {
                    return Err(format!(
                        "failed to back up cloudflared cert `{}` to `{}`: {}",
                        cert_path.display(),
                        backup_path.display(),
                        err
                    ))
                }
            }
        }

        Err(format!(
            "failed to back up cloudflared cert `{}`: too many backup name collisions",
            cert_path.display()
        ))
    }

    async fn prepare_cloudflared_origin_cert_for_login() -> Result<Option<String>, String> {
        let Some(cert_path) = crate::wizard::cloudflared_origin_cert_path() else {
            return Ok(None);
        };
        let backup_path = Self::backup_cloudflared_origin_cert_if_present(&cert_path).await?;
        Ok(backup_path.map(|path| {
            format!(
                "Existing cloudflared origin cert was backed up to `{}` before re-authentication.",
                path.display()
            )
        }))
    }

    async fn read_setup_command_stream<R>(reader: R, tx: mpsc::UnboundedSender<String>)
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        let mut lines = BufReader::new(reader).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }
    }

    async fn run_login_command_with_live_output(
        &self,
        chat_id: i64,
        spec: &crate::wizard::SetupCommandSpec,
        timeout_secs: u64,
        intro: &str,
    ) -> SetupLoginResult {
        let mut result = SetupLoginResult::default();
        let _ = self
            .bot
            .send_message(
                ChatId(chat_id),
                format!(
                    "{}\nCommand: {}",
                    intro,
                    crate::wizard::format_setup_command(spec)
                ),
            )
            .await;

        let mut child = match Command::new(&spec.program)
            .args(&spec.args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(err) => {
                result.error = Some(err.to_string());
                return result;
            }
        };

        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let mut stream_tasks = Vec::new();
        if let Some(stdout) = child.stdout.take() {
            let tx_clone = tx.clone();
            stream_tasks.push(tokio::spawn(async move {
                Self::read_setup_command_stream(stdout, tx_clone).await;
            }));
        }
        if let Some(stderr) = child.stderr.take() {
            let tx_clone = tx.clone();
            stream_tasks.push(tokio::spawn(async move {
                Self::read_setup_command_stream(stderr, tx_clone).await;
            }));
        }
        drop(tx);

        let bot = self.bot.clone();
        let announcer = tokio::spawn(async move {
            let mut lines = Vec::new();
            let mut urls = Vec::new();
            let mut seen_urls = HashSet::new();
            while let Some(line) = rx.recv().await {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if lines.len() < 300 {
                    lines.push(trimmed.to_string());
                }
                for url in crate::wizard::extract_urls(trimmed) {
                    if seen_urls.insert(url.clone()) {
                        urls.push(url.clone());
                        let _ = bot
                            .send_message(
                                ChatId(chat_id),
                                format!("Open this URL to continue auth:\n{}", url),
                            )
                            .await;
                    }
                }
            }
            (lines, urls)
        });

        let wait_outcome =
            tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait()).await;
        match wait_outcome {
            Ok(Ok(status)) => {
                result.success = status.success();
            }
            Ok(Err(err)) => {
                result.error = Some(err.to_string());
            }
            Err(_) => {
                result.timed_out = true;
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
        }

        for task in stream_tasks {
            let _ = task.await;
        }
        match announcer.await {
            Ok((lines, urls)) => {
                result.lines = lines;
                result.urls = urls;
            }
            Err(err) => {
                if result.error.is_none() {
                    result.error = Some(format!("failed to capture auth output: {}", err));
                }
            }
        }
        result
    }

    async fn run_cloudflared_login_and_verify(&self, chat_id: i64) -> Result<String, String> {
        let preflight_note = Self::prepare_cloudflared_origin_cert_for_login().await?;
        let login_spec = crate::wizard::SetupCommandSpec {
            program: "cloudflared".to_string(),
            args: vec!["tunnel".to_string(), "login".to_string()],
        };
        let login_result = self
            .run_login_command_with_live_output(
                chat_id,
                &login_spec,
                600,
                "Starting cloudflared authentication flow.",
            )
            .await;
        if login_result.timed_out {
            return Err(
                "Cloudflared login timed out after 600s. Complete authorization and retry."
                    .to_string(),
            );
        }
        if let Some(err) = login_result.error {
            return Err(format!("Failed to run cloudflared login: {}", err));
        }

        let auth_ok = login_result.success
            && crate::wizard::command_succeeds("cloudflared", &["tunnel", "list"], 20).await;
        if !auth_ok {
            let summary = crate::wizard::summarize_setup_log_lines(&login_result.lines);
            let mut details = Vec::new();
            if let Some(note) = preflight_note.clone() {
                details.push(note);
            }
            if !summary.is_empty() {
                details.push(summary);
            }
            if !login_result.urls.is_empty() {
                details.push(format!(
                    "Authorization URL: {}",
                    login_result.urls.join(", ")
                ));
            }
            let detail_block = if details.is_empty() {
                String::new()
            } else {
                format!("\n{}", details.join("\n"))
            };
            return Err(format!(
                "Cloudflared authentication did not complete successfully.{}",
                detail_block
            ));
        }

        let summary = crate::wizard::summarize_setup_log_lines(&login_result.lines);
        let mut report_lines = vec!["Cloudflared authentication completed.".to_string()];
        if let Some(url) = login_result.urls.first() {
            report_lines.push(format!("URL: {}", url));
        }
        if let Some(note) = preflight_note {
            report_lines.push(note);
        }
        if !summary.is_empty() {
            report_lines.push(summary);
        }
        Ok(report_lines.join("\n"))
    }

    async fn run_setup_commands(
        specs: &[crate::wizard::SetupCommandSpec],
        timeout_secs: u64,
        treat_existing_resource_as_success: bool,
    ) -> (Vec<String>, bool) {
        let mut lines = Vec::with_capacity(specs.len());
        let mut had_failures = false;
        for spec in specs {
            let display = crate::wizard::format_setup_command(spec);
            let run_result = tokio::time::timeout(
                Duration::from_secs(timeout_secs),
                Command::new(&spec.program)
                    .args(&spec.args)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .output(),
            )
            .await;
            match run_result {
                Ok(Ok(output)) => {
                    let summary = crate::wizard::summarize_setup_command_output(
                        &output.stdout,
                        &output.stderr,
                    );
                    let combined = format!(
                        "{}\n{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                    let treated_as_success = output.status.success()
                        || (treat_existing_resource_as_success
                            && crate::wizard::cloudflared_reports_existing_resource(&combined));
                    if treated_as_success {
                        if summary.is_empty() {
                            lines.push(format!("OK   {}", display));
                        } else {
                            lines.push(format!("OK   {} - {}", display, summary));
                        }
                    } else {
                        had_failures = true;
                        if summary.is_empty() {
                            lines.push(format!(
                                "FAIL {} - exited with status {}",
                                display, output.status
                            ));
                        } else {
                            lines.push(format!("FAIL {} - {}", display, summary));
                        }
                    }
                }
                Ok(Err(err)) => {
                    had_failures = true;
                    lines.push(format!("FAIL {} - {}", display, err));
                }
                Err(_) => {
                    had_failures = true;
                    lines.push(format!(
                        "FAIL {} - timed out after {}s",
                        display, timeout_secs
                    ));
                }
            }
        }
        (lines, had_failures)
    }

    fn schedule_low_latency_restart(delay: Duration) -> bool {
        if LOW_LATENCY_RESTART_SCHEDULED.swap(true, Ordering::SeqCst) {
            return false;
        }
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            info!("Automatic restart requested after low-latency setup");
            restart_process();
            LOW_LATENCY_RESTART_SCHEDULED.store(false, Ordering::SeqCst);
        });
        true
    }

    /// Handle /connect command - add a new bot dynamically.
    /// Usage: /connect telegram <bot_token>
    ///        /connect discord <bot_token>
    ///        /connect slack <bot_token> <app_token>
    async fn handle_connect_command(&self, arg: &str, user_id: u64) -> String {
        let parts: Vec<&str> = arg.split_whitespace().collect();

        if parts.is_empty() {
            return "Add a new bot to this agent.\n\n\
                Usage:\n\
                /connect telegram <bot_token>\n\
                /connect discord <bot_token>\n\
                /connect slack <bot_token> <app_token>\n\n\
                The new bot will use the same allowed users as this bot.\n\
                After adding, run /restart to activate the new bot."
                .to_string();
        }

        let channel_type = parts[0].to_lowercase();

        match channel_type.as_str() {
            "telegram" => {
                if parts.len() < 2 {
                    return "Usage: /connect telegram <bot_token>\n\n\
                        Get a token from @BotFather on Telegram."
                        .to_string();
                }
                let token = parts[1];
                self.connect_telegram_bot(token, user_id).await
            }
            "discord" => {
                if parts.len() < 2 {
                    return "Usage: /connect discord <bot_token>\n\n\
                        Get a token from Discord Developer Portal."
                        .to_string();
                }
                let token = parts[1];
                self.connect_discord_bot(token, user_id).await
            }
            "slack" => {
                if parts.len() < 3 {
                    return "Usage: /connect slack <bot_token> <app_token>\n\n\
                        Get tokens from Slack App Management."
                        .to_string();
                }
                let bot_token = parts[1];
                let app_token = parts[2];
                self.connect_slack_bot(bot_token, app_token, user_id).await
            }
            _ => {
                format!(
                    "Unknown channel type: {}\n\n\
                    Supported types: telegram, discord, slack",
                    channel_type
                )
            }
        }
    }

    /// Connect a new Telegram bot by validating its token.
    async fn connect_telegram_bot(&self, token: &str, user_id: u64) -> String {
        // Validate the token by calling getMe
        let bot_username = match super::connect::validate_telegram_token(token).await {
            Ok(name) => name,
            Err(e) => return e,
        };

        // Check if this bot is already connected
        match super::connect::check_bot_exists(self.state.as_ref(), "telegram", token).await {
            Ok(true) => {
                return format!(
                    "Bot @{} is already connected.\n\nUse /bots to see all connected bots.",
                    bot_username
                );
            }
            Ok(false) => {}
            Err(e) => {
                warn!("Failed to check existing bots: {}", e);
            }
        }

        // Save the new bot with same allowed users as current bot
        let allowed_user_ids_str: Vec<String> = self
            .allowed_user_ids
            .read()
            .unwrap()
            .iter()
            .map(|id| id.to_string())
            .collect();

        let new_bot = crate::traits::DynamicBot {
            id: 0, // Will be set by database
            channel_type: "telegram".to_string(),
            bot_token: token.to_string(),
            app_token: None,
            allowed_user_ids: allowed_user_ids_str,
            extra_config: serde_json::json!({"username": bot_username}).to_string(),
            created_at: String::new(), // Will be set by database
        };

        let db_id = match super::connect::persist_dynamic_bot(self.state.as_ref(), &new_bot).await {
            Ok(id) => id,
            Err(e) => {
                warn!("Failed to save bot: {}", e);
                return e;
            }
        };

        info!(
            bot = %bot_username,
            id = db_id,
            added_by = user_id,
            "New Telegram bot connected"
        );

        let mini_app_sync_warning = match self
            .sync_terminal_tenant_bot_token(token, user_id, Some(bot_username.as_str()))
            .await
        {
            Ok(()) => None,
            Err(err) => {
                warn!(
                    error = %err,
                    bot = %bot_username,
                    added_by = user_id,
                    "Failed to auto-sync Telegram bot token with terminal mini app backend"
                );
                Some(
                    "Mini App auth sync failed. If you hit a Telegram auth mismatch, run /agent from this bot once or save the token manually in the Mini App."
                        .to_string(),
                )
            }
        };

        // Try to spawn the bot immediately if we have a hub reference
        let hub_ref = self.channel_hub.read().ok().and_then(|g| g.clone());
        if let Some(weak_hub) = hub_ref {
            if let Some(hub) = weak_hub.upgrade() {
                // Create the new channel with same config as this one
                let new_channel = Arc::new(TelegramChannel::new(
                    token,
                    self.allowed_user_ids
                        .read()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .clone(),
                    TelegramWebhookConfig::default(),
                    self.owner_user_ids.clone(),
                    Arc::clone(&self.agent),
                    self.config_path.clone(),
                    self.session_map.clone(),
                    Arc::clone(&self.task_registry),
                    self.files_enabled,
                    self.inbox_dir.clone(),
                    self.max_file_size_mb,
                    Arc::clone(&self.state),
                    self.watchdog_stale_threshold_secs,
                    self.terminal_web_app_url.clone(),
                    self.terminal_lite.allowed_prefixes(),
                ));

                // Give the new channel a reference to the hub too
                new_channel.set_channel_hub(weak_hub);

                // Register with the hub
                let channel_name = hub
                    .register_channel(new_channel.clone() as Arc<dyn Channel>)
                    .await;
                info!(channel = %channel_name, "Registered new Telegram bot with hub");

                // Spawn the bot in the background using helper to avoid type cycles
                spawn_telegram_channel(new_channel);

                return format!(
                    "✓ Bot @{} connected and started!\n\n\
                    The bot is now active and ready to receive messages.\n\
                    Use /bots to see all connected bots.{}",
                    bot_username,
                    mini_app_sync_warning
                        .as_deref()
                        .map(|value| format!("\n\n{}", value))
                        .unwrap_or_default()
                );
            }
        }

        // Fallback: hub not available, require restart
        format!(
            "✓ Bot @{} connected!\n\n\
            Run /restart to activate the new bot.\n\
            Use /bots to see all connected bots.{}",
            bot_username,
            mini_app_sync_warning
                .as_deref()
                .map(|value| format!("\n\n{}", value))
                .unwrap_or_default()
        )
    }

    /// Connect a new Discord bot by validating its token.
    #[cfg(feature = "discord")]
    async fn connect_discord_bot(&self, token: &str, user_id: u64) -> String {
        // Validate the token by making a test API call
        let bot_name = match super::connect::validate_discord_token(token).await {
            Ok(name) => name,
            Err(e) => return e,
        };

        // Check if already connected
        match super::connect::check_bot_exists(self.state.as_ref(), "discord", token).await {
            Ok(true) => {
                return format!(
                    "Bot {} is already connected.\n\nUse /bots to see all connected bots.",
                    bot_name
                );
            }
            Ok(false) => {}
            Err(e) => {
                warn!("Failed to check existing bots: {}", e);
            }
        }

        // Discord user IDs differ from Telegram IDs — save empty and let
        // the Discord bot auto-claim the first DM user as the owner.
        let new_bot = crate::traits::DynamicBot {
            id: 0,
            channel_type: "discord".to_string(),
            bot_token: token.to_string(),
            app_token: None,
            allowed_user_ids: vec![],
            extra_config: "{}".to_string(),
            created_at: String::new(),
        };

        let db_id = match super::connect::persist_dynamic_bot(self.state.as_ref(), &new_bot).await {
            Ok(id) => id,
            Err(e) => return e,
        };

        info!(
            bot = %bot_name,
            id = db_id,
            added_by = user_id,
            "New Discord bot connected"
        );

        // Try to spawn the bot immediately if we have a hub reference
        let hub_ref = self.channel_hub.read().ok().and_then(|g| g.clone());
        if let Some(weak_hub) = hub_ref {
            if let Some(hub) = weak_hub.upgrade() {
                // Create the new Discord channel with empty allowed_user_ids —
                // the bot will auto-claim the first DM user.
                let new_channel = Arc::new(DiscordChannel::new(
                    token,
                    vec![], // Empty: auto-claim on first DM
                    vec![], // Owner set on auto-claim
                    None,   // No guild_id for dynamic bots
                    Arc::clone(&self.agent),
                    self.config_path.clone(),
                    self.session_map.clone(),
                    Arc::clone(&self.task_registry),
                    self.files_enabled,
                    self.inbox_dir.clone(),
                    self.max_file_size_mb,
                    Arc::clone(&self.state),
                    self.watchdog_stale_threshold_secs,
                ));

                // Give the new channel a reference to the hub
                new_channel.set_channel_hub(weak_hub);

                // Register with the hub
                let channel_name = hub
                    .register_channel(new_channel.clone() as Arc<dyn Channel>)
                    .await;
                info!(channel = %channel_name, "Registered new Discord bot with hub");

                // Spawn the bot in the background
                spawn_discord_channel(new_channel);

                return format!(
                    "✓ Discord bot {} connected and started!\n\n\
                    Send a DM to the bot on Discord to claim it as yours.\n\
                    Use /bots to see all connected bots.",
                    bot_name
                );
            }
        }

        // Fallback: hub not available, require restart
        format!(
            "✓ Discord bot {} connected!\n\n\
            Run /restart to activate the new bot.\n\
            Use /bots to see all connected bots.",
            bot_name
        )
    }

    /// Connect a new Discord bot (stub when feature disabled).
    #[cfg(not(feature = "discord"))]
    async fn connect_discord_bot(&self, token: &str, _user_id: u64) -> String {
        match super::connect::validate_discord_token(token).await {
            Ok(_) => unreachable!("discord feature is disabled"),
            Err(e) => format!(
                "{}\n\nRebuild with `cargo build --features discord` to enable Discord bots.",
                e
            ),
        }
    }

    /// Connect a new Slack bot.
    #[cfg(feature = "slack")]
    async fn connect_slack_bot(&self, bot_token: &str, app_token: &str, user_id: u64) -> String {
        // Validate the bot token by calling auth.test
        let (bot_name, team_name) =
            match super::connect::validate_slack_tokens(bot_token, app_token).await {
                Ok(pair) => pair,
                Err(e) => return e,
            };

        // Check if already connected
        match super::connect::check_bot_exists(self.state.as_ref(), "slack", bot_token).await {
            Ok(true) => {
                return format!(
                    "Slack bot {} ({}) is already connected.\n\nUse /bots to see all connected bots.",
                    bot_name, team_name
                );
            }
            Ok(false) => {}
            Err(e) => {
                warn!("Failed to check existing bots: {}", e);
            }
        }

        let allowed_user_ids_str = vec![user_id.to_string()];

        let new_bot = crate::traits::DynamicBot {
            id: 0,
            channel_type: "slack".to_string(),
            bot_token: bot_token.to_string(),
            app_token: Some(app_token.to_string()),
            allowed_user_ids: allowed_user_ids_str.clone(),
            extra_config: "{}".to_string(),
            created_at: String::new(),
        };

        let db_id = match super::connect::persist_dynamic_bot(self.state.as_ref(), &new_bot).await {
            Ok(id) => id,
            Err(e) => return e,
        };

        info!(
            bot = %bot_name,
            team = %team_name,
            id = db_id,
            added_by = user_id,
            "New Slack bot connected"
        );

        // Try to spawn the bot immediately if we have a hub reference
        let hub_ref = self.channel_hub.read().ok().and_then(|g| g.clone());
        if let Some(weak_hub) = hub_ref {
            if let Some(hub) = weak_hub.upgrade() {
                // Create the new Slack channel with same config as this Telegram channel
                let new_channel = Arc::new(SlackChannel::new(
                    app_token,
                    bot_token,
                    allowed_user_ids_str, // Slack uses String user IDs
                    false,                // use_threads default
                    Arc::clone(&self.agent),
                    self.config_path.clone(),
                    self.session_map.clone(),
                    Arc::clone(&self.task_registry),
                    self.files_enabled,
                    self.inbox_dir.clone(),
                    self.max_file_size_mb,
                    Arc::clone(&self.state),
                    self.watchdog_stale_threshold_secs,
                ));

                // Give the new channel a reference to the hub
                new_channel.set_channel_hub(weak_hub);

                // Register with the hub
                let channel_name = hub
                    .register_channel(new_channel.clone() as Arc<dyn Channel>)
                    .await;
                info!(channel = %channel_name, "Registered new Slack bot with hub");

                // Spawn the bot in the background
                spawn_slack_channel(new_channel);

                return format!(
                    "✓ Slack bot {} ({}) connected and started!\n\n\
                    The bot is now active and ready to receive messages.\n\
                    Use /bots to see all connected bots.",
                    bot_name, team_name
                );
            }
        }

        // Fallback: hub not available, require restart
        format!(
            "✓ Slack bot {} ({}) connected!\n\n\
            Run /restart to activate the new bot.\n\
            Use /bots to see all connected bots.",
            bot_name, team_name
        )
    }

    /// Connect a new Slack bot (stub when feature disabled).
    #[cfg(not(feature = "slack"))]
    async fn connect_slack_bot(&self, bot_token: &str, app_token: &str, _user_id: u64) -> String {
        match super::connect::validate_slack_tokens(bot_token, app_token).await {
            Ok(_) => unreachable!("slack feature is disabled"),
            Err(e) => format!(
                "{}\n\nRebuild with `cargo build --features slack` to enable Slack bots.",
                e
            ),
        }
    }

    /// Handle /bots command - list all connected bots.
    async fn handle_bots_command(&self) -> String {
        let mut bots_list = vec![];

        // Add current bot (from config)
        let current_username = self.get_bot_username().await;
        bots_list.push(format!(
            "• telegram:@{} (this bot, from config)",
            current_username
        ));

        // Add dynamic bots from database
        match super::connect::list_dynamic_bots(self.state.as_ref()).await {
            Ok(bots) => {
                for bot in bots {
                    let bot_info = match bot.channel_type.as_str() {
                        "telegram" => {
                            // Try to get username for display
                            let test_bot = Bot::new(&bot.bot_token);
                            match test_bot.get_me().await {
                                Ok(me) => {
                                    let username = me
                                        .username
                                        .clone()
                                        .unwrap_or_else(|| "unknown".to_string());
                                    format!("• telegram:@{} (id: {})", username, bot.id)
                                }
                                Err(_) => format!("• telegram:<invalid token> (id: {})", bot.id),
                            }
                        }
                        "discord" => format!("• discord (id: {})", bot.id),
                        "slack" => format!("• slack (id: {})", bot.id),
                        other => format!("• {} (id: {})", other, bot.id),
                    };
                    bots_list.push(bot_info);
                }
            }
            Err(e) => {
                return format!("Failed to list bots: {}", e);
            }
        }

        if bots_list.len() == 1 {
            format!(
                "Connected bots:\n{}\n\nUse /connect to add more bots.",
                bots_list.join("\n")
            )
        } else {
            format!(
                "Connected bots ({}):\n{}\n\n\
                Tip: Dynamic bots activate after /restart.",
                bots_list.len(),
                bots_list.join("\n")
            )
        }
    }

    fn parse_web_app_action(raw: &str) -> Option<TelegramWebAppAction> {
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

    async fn build_terminal_attach_handoff_message(
        &self,
        relay_session_id: &str,
        user_id: u64,
    ) -> anyhow::Result<String> {
        let handoff = crate::agent_handoff::create_handoff_code(
            self.state.as_ref(),
            relay_session_id,
            user_id,
        )
        .await?;
        let command = format!("aidaemon attach {}", handoff.code);
        let shell_token = |value: &str| -> String {
            if !value.is_empty()
                && value.chars().all(|ch| {
                    ch.is_ascii_alphanumeric()
                        || matches!(ch, '_' | '-' | '.' | '/' | ':' | '=' | '+' | ',' | '@')
                })
            {
                value.to_string()
            } else {
                format!("'{}'", value.replace('\'', r"'\''"))
            }
        };
        let exact_command = std::env::current_exe().ok().and_then(|exe| {
            let exe_text = exe.to_string_lossy().trim().to_string();
            if exe_text.is_empty() {
                None
            } else {
                Some(format!(
                    "{} attach {}",
                    shell_token(&exe_text),
                    handoff.code
                ))
            }
        });
        let command_section = if let Some(exact) = exact_command {
            format!(
                "Run this on your computer:\n\
                 <pre>{}</pre>\n\n\
                 If <code>aidaemon</code> points to an older install, run:\n\
                 <pre>{}</pre>\n\n",
                html_escape(&command),
                html_escape(&exact)
            )
        } else {
            format!(
                "Run this on your computer:\n\
                 <pre>{}</pre>\n\n",
                html_escape(&command)
            )
        };
        Ok(format!(
            "🖥️ <b>Continue In Native Terminal</b>\n\n\
             {}\
             Session: <code>{}</code>\n\
             Expires in about 5 minutes. Code is one-time use.",
            command_section,
            html_escape(relay_session_id)
        ))
    }

    async fn resolve_continue_relay_session_id(
        &self,
        chat_id: i64,
        relay_session_id: Option<String>,
    ) -> Option<String> {
        if let Some(value) = relay_session_id
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
        {
            return Some(value);
        }

        let telegram_session_id = self.session_id(chat_id).await;
        match crate::agent_handoff::resolve_relay_for_telegram_session(
            self.state.as_ref(),
            &telegram_session_id,
        )
        .await
        {
            Ok(Some(value)) => Some(value),
            Ok(None) => None,
            Err(err) => {
                warn!(
                    error = %err,
                    telegram_session_id = %telegram_session_id,
                    "Failed to resolve relay mapping for continue-on-computer"
                );
                None
            }
        }
    }

    async fn handle_continue_on_computer_action(
        &self,
        bot: &Bot,
        msg: &teloxide::types::Message,
        user_id: u64,
        relay_session_id: Option<String>,
    ) {
        self.send_agent_share_code(bot, msg.chat.id, user_id, relay_session_id)
            .await;
    }

    async fn send_agent_share_code(
        &self,
        bot: &Bot,
        chat_id: ChatId,
        user_id: u64,
        relay_session_id: Option<String>,
    ) {
        let resolved_relay = if let Some(value) = relay_session_id
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
        {
            Some(value)
        } else {
            let mapped = self
                .resolve_continue_relay_session_id(chat_id.0, None)
                .await;
            if mapped.is_some() {
                mapped
            } else {
                match crate::agent_handoff::get_last_active_relay_session_id(
                    self.state.as_ref(),
                    user_id,
                )
                .await
                {
                    Ok(value) => value,
                    Err(err) => {
                        warn!(
                            error = %err,
                            "Failed to load last active relay session id for /agent share"
                        );
                        None
                    }
                }
            }
        };

        let Some(relay_session_id) = resolved_relay else {
            let _ = bot
                .send_message(
                    chat_id,
                    "Could not identify the active terminal session. Keep the Mini App session open and try Continue on Computer again.",
                )
                .await;
            return;
        };

        match self
            .build_terminal_attach_handoff_message(&relay_session_id, user_id)
            .await
        {
            Ok(reply) => {
                let _ = bot
                    .send_message(chat_id, reply)
                    .parse_mode(ParseMode::Html)
                    .await;
            }
            Err(err) => {
                warn!(error = %err, "Failed to create terminal attach handoff code");
                let _ = bot
                    .send_message(
                        chat_id,
                        "Failed to create continue-on-computer code. Please try again.",
                    )
                    .await;
            }
        }
    }

    async fn send_agent_resume_prompt(
        &self,
        bot: &Bot,
        chat_id: ChatId,
        user_id: u64,
        code: Option<String>,
    ) {
        let Some(code) = code.map(|v| v.trim().to_string()).filter(|v| !v.is_empty()) else {
            let _ = bot
                .send_message(chat_id, "Usage: /agent resume <code>")
                .await;
            return;
        };

        let handoff = match crate::agent_handoff::resolve_handoff_code(self.state.as_ref(), &code)
            .await
        {
            Ok(value) => value,
            Err(err) => {
                let _ = bot
                        .send_message(
                            chat_id,
                            format!(
                                "Resume code is invalid or expired: {}. Generate a fresh code from your computer with `aidaemon share`.",
                                err
                            ),
                        )
                        .await;
                return;
            }
        };

        if handoff.owner_user_id != user_id {
            let _ = bot
                .send_message(
                    chat_id,
                    "That resume code belongs to a different Telegram user.",
                )
                .await;
            return;
        }

        if let Err(err) =
            crate::agent_handoff::consume_handoff_code(self.state.as_ref(), &code).await
        {
            let _ = bot
                .send_message(
                    chat_id,
                    format!("Resume code could not be consumed: {}.", err),
                )
                .await;
            return;
        }

        let relay_session_id = handoff.relay_session_id.trim().to_string();
        let telegram_session_id = self.session_id(chat_id.0).await;
        if let Err(err) = crate::agent_handoff::bind_telegram_session_to_relay(
            self.state.as_ref(),
            &telegram_session_id,
            &relay_session_id,
        )
        .await
        {
            warn!(
                error = %err,
                relay_session_id = %relay_session_id,
                telegram_session_id = %telegram_session_id,
                "Failed to bind relay session from /agent resume"
            );
            let _ = bot
                .send_message(
                    chat_id,
                    "Failed to bind the resume session. Please generate a new code and retry.",
                )
                .await;
            return;
        }

        let current_bot_label = self.cached_bot_label();
        let mini_app_sync_warning = match self
            .sync_terminal_tenant_bot_token(&self.bot_token, user_id, current_bot_label.as_deref())
            .await
        {
            Ok(()) => None,
            Err(err) => {
                warn!(
                    error = %err,
                    user_id,
                    relay_session_id = %relay_session_id,
                    "Failed to auto-sync Telegram bot token before /agent resume"
                );
                Some(
                    "If Telegram reports an auth mismatch, retry once or save the token manually in the Mini App."
                        .to_string(),
                )
            }
        };

        let mut web_app_url = match reqwest::Url::parse(&self.terminal_web_app_url) {
            Ok(url) => url,
            Err(err) => {
                warn!(error = %err, "Invalid terminal_web_app_url during /agent resume");
                let _ = bot
                    .send_message(
                        chat_id,
                        "Resume code accepted. Run `/agent open` to continue in Mini App.",
                    )
                    .await;
                return;
            }
        };
        {
            let mut query = web_app_url.query_pairs_mut();
            query.append_pair("telegram_session_id", &telegram_session_id);
            query.append_pair("relay_session_id", &relay_session_id);
            query.append_pair("autostart", "1");
        }

        let keyboard = InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::web_app(
                "📱 Open in Mini App",
                WebAppInfo {
                    url: web_app_url.clone(),
                },
            ),
            InlineKeyboardButton::callback(
                "💻 Continue on Computer",
                format!("agent:share:{}", relay_session_id),
            ),
        ]]);

        let _ = bot
            .send_message(
                chat_id,
                format!(
                    "✅ <b>Resume Code Accepted</b>\n\nSession: <code>{}</code>\nTap Open in Mini App to continue on your phone.{}",
                    html_escape(&relay_session_id),
                    mini_app_sync_warning
                        .as_deref()
                        .map(|value| format!("\n\n{}", html_escape(value)))
                        .unwrap_or_default()
                ),
            )
            .parse_mode(ParseMode::Html)
            .reply_markup(keyboard)
            .await;

        let reply_keyboard =
            KeyboardMarkup::new(vec![vec![KeyboardButton::new("📱 Open in Mini App")
                .request(ButtonRequest::WebApp(WebAppInfo {
                    url: web_app_url.clone(),
                }))]])
            .resize_keyboard()
            .one_time_keyboard();

        let _ = bot
            .send_message(
                chat_id,
                "Tip: use this keyboard launch if Mini App send-back actions are unavailable in inline mode.",
            )
            .reply_markup(reply_keyboard)
            .await;
    }

    async fn handle_message(self: &Arc<Self>, msg: teloxide::types::Message, bot: Bot) {
        let user_id = msg.from.as_ref().map(|u| u.id.0).unwrap_or(0);

        // Authorization check with first-user auto-claim (DM only).
        // When allowed_user_ids is empty, the FIRST user to send a private
        // message becomes the owner. Group messages are rejected until an
        // owner is established, preventing someone from claiming ownership
        // by adding the bot to a group.
        let is_private = matches!(msg.chat.kind, teloxide::types::ChatKind::Private(_));
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
                // Persist to config.toml so it survives restarts
                let ids = self
                    .allowed_user_ids
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                if let Err(e) = self.persist_allowed_user_ids(&ids).await {
                    warn!(
                        user_id,
                        "Failed to persist auto-claimed user ID to config: {}", e
                    );
                }
                #[cfg(feature = "terminal-bridge")]
                let mut bridge_hotstart_failed = false;
                #[cfg(feature = "terminal-bridge")]
                match AppConfig::load(&self.config_path) {
                    Ok(config) => {
                        if config.terminal.effective_bridge_enabled()
                            && !crate::terminal_bridge::spawn_if_configured(
                                &config,
                                self.state.clone(),
                            )
                        {
                            bridge_hotstart_failed = true;
                        }
                    }
                    Err(err) => {
                        warn!(
                            user_id,
                            error = %err,
                            "Failed to reload config for terminal bridge hot-start after auto-claim"
                        );
                        bridge_hotstart_failed = true;
                    }
                }
                #[cfg(feature = "terminal-bridge")]
                let mut welcome = "Hey! You're now set as the owner. Ask me anything, give me tasks, or just chat."
                    .to_string();
                #[cfg(not(feature = "terminal-bridge"))]
                let welcome = "Hey! You're now set as the owner. Ask me anything, give me tasks, or just chat."
                    .to_string();
                #[cfg(feature = "terminal-bridge")]
                if bridge_hotstart_failed {
                    welcome.push_str(
                        "\n\nI couldn't auto-enable the /agent bridge right now. If /agent doesn't open, run /restart.",
                    );
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

        let user_role = determine_role(&self.owner_user_ids, user_id);

        let mut inbound_attachments = Vec::new();
        let mut body_text = String::new();

        if let Some(web_app_data) = msg.web_app_data() {
            match Self::parse_web_app_action(&web_app_data.data) {
                Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                    self.handle_continue_on_computer_action(&bot, &msg, user_id, relay_session_id)
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
                        self.handle_continue_on_computer_action(
                            &bot,
                            &msg,
                            user_id,
                            relay_session_id,
                        )
                        .await;
                        return;
                    }
                    TelegramWebAppAction::AgentMessage(text) => body_text = text,
                }
            } else {
                body_text = t.to_string();
            }
        } else if !self.files_enabled {
            let _ = bot
                .send_message(msg.chat.id, "I can only process text messages.")
                .await;
            return;
        }

        if self.files_enabled
            && (msg.document().is_some()
                || msg.photo().is_some()
                || msg.audio().is_some()
                || msg.video().is_some()
                || msg.voice().is_some())
        {
            match self.handle_file_message(&msg, &bot).await {
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

        let text = super::attachments::build_inbound_text(&body_text, &inbound_attachments);

        // Handle slash commands.
        // `/setup` is spawned in a separate task because `handle_setup_command`
        // may call `request_approval`, which blocks waiting for a Telegram
        // callback query.  Teloxide dispatches updates per-chat sequentially,
        // so running it inline would deadlock: the callback can't be delivered
        // until the message handler returns, but the message handler is waiting
        // for that callback.
        if text.starts_with('/') {
            // Slash commands `return` before the message dedup gate below, so a
            // duplicate Telegram delivery of the same update (same message id)
            // would execute the command twice — e.g. two "Context cleared."
            // replies, a double `/cancel`. Apply the same msg.id-keyed dedup
            // here. Distinct sends (different message ids) are never deduped, so
            // intentional repeats still work.
            let cmd_session_id = self.session_id(msg.chat.id.0).await;
            let cmd_dedup_key = msg.id.0.to_string();
            if !self
                .task_registry
                .mark_seen(&cmd_session_id, &text, Some(&cmd_dedup_key))
                .await
            {
                debug!(session_id = %cmd_session_id, "Dropped duplicate slash command");
                return;
            }
            let is_setup = {
                let cmd = text.split_whitespace().next().unwrap_or("");
                let cmd = cmd.split('@').next().unwrap_or(cmd);
                cmd == "/setup"
            };
            if is_setup {
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
            .handle_terminal_lite_input(msg.chat.id.0, user_id, user_role, &text)
            .await
        {
            for chunk in split_message(&lite_reply, 4096) {
                let _ = bot.send_message(msg.chat.id, chunk).await;
            }
            return;
        }

        // Use chat ID as session ID, prefixed with bot name if multi-bot
        let session_id = self.session_id(msg.chat.id.0).await;
        let channel_id = crate::session::derive_channel_id_from_session(&session_id);

        // Register this session with the channel hub so outbound messages
        // (approvals, media, notifications) route back to this Telegram bot.
        {
            let channel_name = self.channel_name().await;
            {
                let mut map = self.session_map.write().await;
                map.insert(session_id.clone(), channel_name.clone());
            }
            let _ = self
                .state
                .save_session_channel(&session_id, &channel_name)
                .await;
        }

        // Build channel context from Telegram chat type
        let channel_ctx = {
            use teloxide::types::{ChatKind, PublicChatKind};
            let visibility = match &msg.chat.kind {
                ChatKind::Private(_) => ChannelVisibility::Private,
                ChatKind::Public(public) => match &public.kind {
                    PublicChatKind::Group => ChannelVisibility::PrivateGroup,
                    PublicChatKind::Supergroup(sg) => {
                        if sg.username.is_some() {
                            ChannelVisibility::Public
                        } else {
                            ChannelVisibility::PrivateGroup
                        }
                    }
                    PublicChatKind::Channel(_) => ChannelVisibility::Public,
                },
            };
            ChannelContext {
                visibility,
                platform: "telegram".to_string(),
                channel_name: msg.chat.title().map(|s| s.to_string()),
                channel_id,
                sender_name: msg.from.as_ref().map(|u| match &u.last_name {
                    Some(last) => format!("{} {}", u.first_name, last),
                    None => u.first_name.clone(),
                }),
                sender_id: msg.from.as_ref().map(|u| format!("telegram:{}", u.id.0)),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            }
        };

        // Handle cancel/stop commands - these bypass the queue
        let text_lower = text.to_lowercase();
        if text_lower == "cancel" || text_lower == "stop" || text_lower == "abort" {
            if user_role != UserRole::Owner {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Only the owner can cancel running work in this session.",
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
                    let _ = bot
                        .send_message(msg.chat.id, "No running task to cancel.")
                        .await;
                } else if cancelled_goals.len() == 1 {
                    let _ = bot
                        .send_message(
                            msg.chat.id,
                            format!("⏹️ Cancelled goal: {}", cancelled_goals[0]),
                        )
                        .await;
                } else {
                    let _ = bot
                        .send_message(
                            msg.chat.id,
                            format!(
                                "⏹️ Cancelled {} goals:\n{}",
                                cancelled_goals.len(),
                                cancelled_goals
                                    .iter()
                                    .map(|d| format!("- {}", d))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            ),
                        )
                        .await;
                }
            } else {
                let desc = cancelled
                    .first()
                    .map(|(_, d)| d.as_str())
                    .unwrap_or("unknown");
                let queue_cleared = self.task_registry.queue_len(&session_id).await;
                let mut response = format!("⏹️ Cancelled: {}", desc);
                if queue_cleared > 0 {
                    response.push_str(&format!(" (+{} queued messages cleared)", queue_cleared));
                }
                if !cancelled_goals.is_empty() {
                    response.push_str(&format!(" (+{} goal(s) cancelled)", cancelled_goals.len()));
                }
                let _ = bot.send_message(msg.chat.id, response).await;
            }
            return;
        }

        // Stable per-message identity for dedup. Telegram's `message_id` is
        // unique per chat and reused on webhook/poll redeliveries, but a user
        // re-typing the same text gets a NEW message_id — so this drops genuine
        // duplicates while letting a deliberate re-ask through.
        let dedup_key = msg.id.0.to_string();

        // Check if a task is already running - if so, queue this message
        if self.task_registry.has_running_task(&session_id).await {
            let daemon_uptime = self.started_at.elapsed();
            if should_ignore_lightweight_interjection(&text, daemon_uptime) {
                let current_task = self
                    .task_registry
                    .get_running_task_description(&session_id)
                    .await
                    .unwrap_or_else(|| "processing".to_string());
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!(
                            "⏳ Still working on: {}. I ignored that short check-in. \
                             Send `cancel` to stop the current task.",
                            current_task
                        ),
                    )
                    .await;
                return;
            }
            // Atomic check-and-queue: if the task finished between the
            // has_running_task() check above and this call, the registry
            // reports NoRunningTask and we fall through to direct processing
            // instead of stranding the message in an undrained queue.
            match self
                .task_registry
                .queue_message_if_running(&session_id, &text, Some(&dedup_key))
                .await
            {
                QueueOutcome::Queued(queue_pos) => {
                    // Only notify the user for the first 3 queued messages to avoid spam
                    // (long messages fragmented by Telegram Web can produce 10+ fragments).
                    if queue_pos <= 3 {
                        let current_task = self
                            .task_registry
                            .get_running_task_description(&session_id)
                            .await
                            .unwrap_or_else(|| "processing".to_string());
                        let preview: String = text.chars().take(50).collect();
                        let suffix = if text.len() > 50 { "..." } else { "" };
                        let _ = bot
                            .send_message(
                                msg.chat.id,
                                format!(
                                    "📥 Queued ({}): \"{}{}\" | Currently: {}",
                                    queue_pos, preview, suffix, current_task
                                ),
                            )
                            .await;
                    }
                    return;
                }
                QueueOutcome::Duplicate => {
                    // Duplicate message detected — silently ignore
                    debug!(session_id, "Dropped duplicate queued message");
                    return;
                }
                QueueOutcome::NoRunningTask => {
                    info!(
                        session_id,
                        "Task finished during queue attempt — processing message directly"
                    );
                    // Fall through to direct processing below.
                }
            }
        }

        // Dedup gate: atomically mark this message as "seen" so concurrent
        // webhook handlers for the same text don't ALL start direct processing.
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

        info!(session_id, "Received message from user {}", user_id);

        // Create heartbeat for watchdog — agent bumps this on every activity point.
        let heartbeat = Arc::new(AtomicU64::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        ));

        // Send typing indicator immediately, then repeat every 3s while agent works.
        // Telegram's typing indicator expires after ~5s, so 3s keeps it continuous
        // even if the occasional send_chat_action call fails silently.
        // Also monitors the heartbeat: if it goes stale, notify user and stop typing.
        let typing_bot = bot.clone();
        let typing_chat_id = msg.chat.id;
        let typing_cancel = tokio_util::sync::CancellationToken::new();
        let typing_token = typing_cancel.clone();
        let heartbeat_for_typing = heartbeat.clone();
        let stale_threshold_secs = self.watchdog_stale_threshold_secs;
        tokio::spawn(async move {
            // Hard maximum: stop typing after 30 minutes regardless of heartbeat/cancel.
            // Prevents infinite typing loops from leaked heartbeat keepers.
            let deadline = tokio::time::Instant::now() + Duration::from_secs(30 * 60);
            loop {
                let _ = typing_bot
                    .send_chat_action(typing_chat_id, ChatAction::Typing)
                    .await;
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(3)) => {
                        if stale_threshold_secs > 0 {
                            let last_hb = heartbeat_for_typing.load(Ordering::Relaxed);
                            let now = SystemTime::now().duration_since(UNIX_EPOCH)
                                .unwrap_or_default().as_secs();
                            if now.saturating_sub(last_hb) > stale_threshold_secs {
                                break; // Stop typing indicator
                            }
                        }
                        if tokio::time::Instant::now() >= deadline {
                            tracing::warn!(chat_id = %typing_chat_id, "Typing indicator reached 30-minute hard limit");
                            break;
                        }
                    }
                    _ = typing_token.cancelled() => break,
                }
            }
        });

        // Upgrade the channel hub weak reference while `self` is still in scope.
        // The hub is needed for the live-status surface (Task 4/5).
        let status_hub = self
            .channel_hub
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|weak| weak.upgrade()));

        // Status update channel — agent emits updates, we display them rate-limited.
        let (status_tx, status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(16);
        let is_dm = channel_ctx.visibility == ChannelVisibility::Private;

        // Build the single live-status surface (owner + hub-backed sink) when the hub
        // is available. The normal core.rs startup path wires the hub before any channel
        // traffic arrives, so the None branch is purely defensive.
        let (live_owner, live_sink): (
            Option<std::sync::Arc<tokio::sync::Mutex<crate::channels::live_status::LiveStatus>>>,
            Option<std::sync::Arc<crate::channels::live_status::HubSurfaceSink>>,
        ) = if let Some(hub) = status_hub.clone() {
            let owner = std::sync::Arc::new(tokio::sync::Mutex::new(
                crate::channels::live_status::LiveStatus::new(),
            ));
            let sink = std::sync::Arc::new(crate::channels::live_status::HubSurfaceSink::new(
                hub,
                session_id.clone(),
            ));
            (Some(owner), Some(sink))
        } else {
            warn!(
                session_id = %session_id,
                "Telegram status surface disabled: ChannelHub is not wired"
            );
            (None, None)
        };

        let status_task = match (&live_owner, &live_sink) {
            (Some(owner), Some(sink)) => tokio::spawn(run_status_consumer(
                status_rx,
                owner.clone(),
                sink.clone(),
                bot.clone(),
                msg.chat.id,
                is_dm,
            )),
            _ => {
                // Fallback: hub unavailable — legacy consumer keeps basic progress
                // visible so user requests still work in degraded mode.
                tokio::spawn(run_legacy_status_consumer(
                    status_rx,
                    bot.clone(),
                    msg.chat.id,
                    is_dm,
                ))
            }
        };

        // Register this task for tracking and cancellation.
        let description: String = text.chars().take(80).collect();
        let (task_id, cancel_token) = self.task_registry.register(&session_id, &description).await;
        // Associate the typing indicator with this task so cancel_running_for_session
        // also stops the typing loop (fixes typing persisting after cancel/panic).
        self.task_registry
            .set_typing_cancel(task_id, typing_cancel.clone())
            .await;
        let registry = Arc::clone(&self.task_registry);

        // Spawn the agent work in a separate task so the dispatcher can continue
        // processing other updates (especially callback queries for tool approval).
        let agent = Arc::clone(&self.agent);
        let chat_id = msg.chat.id;
        tokio::spawn(async move {
            // Drop guard: if this task panics, ensure the typing indicator stops.
            // Without this, a panic in handle_message() would skip the explicit
            // typing_cancel.cancel() call, leaving the typing loop running.
            // Uses Arc<Mutex<>> so the guard always tracks the *current* typing token
            // even when queued messages replace it.
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

            let mut current_text = text;
            let inbound_attachments = inbound_attachments;
            let mut attachments_sent = false;
            let mut current_task_id = task_id;
            let mut current_cancel_token = cancel_token;
            let mut current_status_tx = status_tx;
            let mut current_typing_cancel = typing_cancel;
            let mut current_status_task = status_task;
            let mut current_heartbeat = heartbeat;
            // live_owner/live_sink are carried through all queue-drain iterations so
            // that Task 6 can finalize the surface after the agent reply is ready.
            let live_owner = live_owner;
            let live_sink = live_sink;
            // Hard wall-clock deadline: no single message handling can exceed this,
            // regardless of heartbeat or LLM watchdog state. This catches hangs in
            // non-LLM code paths (DB queries, bootstrap phase, etc.) that the
            // heartbeat-based watchdog cannot detect.
            let task_wall_deadline = tokio::time::Instant::now() + Duration::from_secs(20 * 60);

            loop {
                let attachments_for_call = if attachments_sent {
                    &[][..]
                } else {
                    &inbound_attachments[..]
                };
                let result = tokio::select! {
                    r = agent.handle_message_with_attachments(&session_id, &current_text, attachments_for_call, Some(current_status_tx), user_role, channel_ctx.clone(), Some(current_heartbeat.clone())) => r,
                    _ = current_cancel_token.cancelled() => Err(anyhow::anyhow!("Task cancelled")),
                    stale_mins = super::wait_for_stale_heartbeat(current_heartbeat.clone(), stale_threshold_secs, 4), if stale_threshold_secs > 0 => {
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
                attachments_sent = true;
                current_typing_cancel.cancel();
                // Abort the status display task to prevent blocking if a background
                // CLI agent monitoring task still holds a status_tx clone.
                current_status_task.abort();

                // Track whether this iteration ended in error (for deferred finalization).
                let mut task_error: Option<String> = None;

                match result {
                    Ok(reply) => {
                        // NOTE: Task intentionally stays "running" during response
                        // sending. This prevents a race condition where incoming
                        // messages see no running task (has_running_task = false),
                        // skip the queue, and spawn concurrent tasks — silently
                        // dropping themselves. Finalized below before queue check.
                        if !reply.trim().is_empty() {
                            let handled = if let (Some(live_owner), Some(live_sink)) =
                                (live_owner.as_ref(), live_sink.as_ref())
                            {
                                use crate::channels::live_status::SurfaceSink;
                                let sink_ref: &dyn SurfaceSink = live_sink.as_ref();
                                live_owner
                                    .lock()
                                    .await
                                    .finalize_text(sink_ref, &reply)
                                    .await
                            } else {
                                false
                            };
                            // A delivered background handoff ("⏳ Still on it — …")
                            // becomes the session's status surface: the terminal
                            // notifier EDITS it into the completion ping instead of
                            // stacking a third message. Registered before reset()
                            // clears the surface id below.
                            if handled && crate::agent::is_friendly_background_handoff(&reply) {
                                if let (Some(hub), Some(live_owner)) =
                                    (status_hub.as_ref(), live_owner.as_ref())
                                {
                                    if let Some(id) = live_owner.lock().await.surface_message_id() {
                                        hub.register_background_status_surface(&session_id, &id)
                                            .await;
                                    }
                                }
                            }
                            if !handled {
                                if let Err(e) =
                                    send_full_or_expandable_reply(&bot, chat_id, &reply).await
                                {
                                    warn!("Failed to send Telegram message: {}", e);
                                }
                            }
                        } else if let (Some(live_owner), Some(live_sink)) =
                            (live_owner.as_ref(), live_sink.as_ref())
                        {
                            // Empty reply: resolve any live surface so it doesn't stay
                            // stuck on "⏳ Running…" forever.
                            use crate::channels::live_status::SurfaceSink;
                            let sink_ref: &dyn SurfaceSink = live_sink.as_ref();
                            live_owner.lock().await.finalize_done(sink_ref).await;
                        }
                        if let Some(live_owner) = live_owner.as_ref() {
                            live_owner.lock().await.reset();
                        }
                    }
                    Err(e) => {
                        let error_msg = e.to_string();
                        // Cancellation: fail immediately and exit (queue already
                        // cleared by the /cancel handler).
                        if error_msg == "Task cancelled" {
                            registry.fail(current_task_id, &error_msg).await;
                            info!("Task #{} cancelled", current_task_id);
                            return; // Exit loop on cancellation
                        }
                        // Other errors: defer fail() so the task stays "running"
                        // while we send the error message (same race-prevention
                        // logic as the Ok path).
                        task_error = Some(error_msg.clone());
                        if error_msg.starts_with("Task auto-cancelled due to inactivity") {
                            info!("Task #{} auto-cancelled by stale watchdog", current_task_id);
                            let _ = bot.send_message(chat_id, format!("⚠️ {}", error_msg)).await;
                        } else {
                            warn!("Agent error: {}", e);
                            let _ = bot.send_message(chat_id, format!("Error: {}", e)).await;
                        }
                        if let (Some(live_owner), Some(live_sink)) =
                            (live_owner.as_ref(), live_sink.as_ref())
                        {
                            use crate::channels::live_status::SurfaceSink;
                            let sink_ref: &dyn SurfaceSink = live_sink.as_ref();
                            let _ = live_owner
                                .lock()
                                .await
                                .finalize_text(sink_ref, &format!("⚠️ {error_msg}"))
                                .await;
                            live_owner.lock().await.reset();
                        } else if let Some(live_owner) = live_owner.as_ref() {
                            live_owner.lock().await.reset();
                        }
                    }
                }

                // Finalize the current task AFTER sending the response/error,
                // and drain the queue in the SAME critical section: a message
                // arriving concurrently is then either drained here or sees
                // NoRunningTask and processes directly — never stranded in a
                // queue nothing will drain.
                if let Some(queued) = registry
                    .finalize_and_drain(current_task_id, &session_id, task_error.as_deref())
                    .await
                {
                    // Small delay to ensure previous message is fully committed to DB
                    tokio::time::sleep(Duration::from_millis(100)).await;

                    info!(
                        session_id,
                        "Processing queued message: {}",
                        queued.text.chars().take(50).collect::<String>()
                    );
                    let _ = bot
                        .send_message(
                            chat_id,
                            format!(
                                "▶️ Processing queued: \"{}\"",
                                queued.text.chars().take(50).collect::<String>()
                            ),
                        )
                        .await;

                    // Set up for next iteration
                    current_text = queued.text;
                    let desc: String = current_text.chars().take(80).collect();
                    let (new_task_id, new_cancel_token) =
                        registry.register(&session_id, &desc).await;
                    current_task_id = new_task_id;
                    current_cancel_token = new_cancel_token;

                    // Create new status channel for the queued turn.
                    let (new_status_tx, new_status_rx) =
                        tokio::sync::mpsc::channel::<StatusUpdate>(16);
                    current_status_tx = new_status_tx;

                    // Reset the live surface and spawn a fresh consumer for the
                    // queued turn, reusing the same owner/sink (Task 6 will finalize).
                    current_status_task = match (&live_owner, &live_sink) {
                        (Some(owner), Some(sink)) => {
                            owner.lock().await.reset();
                            tokio::spawn(run_status_consumer(
                                new_status_rx,
                                owner.clone(),
                                sink.clone(),
                                bot.clone(),
                                chat_id,
                                is_dm,
                            ))
                        }
                        _ => {
                            // Legacy fallback: hub unavailable — shared legacy consumer
                            // so queued turns still show progress in degraded mode.
                            tokio::spawn(run_legacy_status_consumer(
                                new_status_rx,
                                bot.clone(),
                                chat_id,
                                is_dm,
                            ))
                        }
                    };

                    // Fresh heartbeat for queued message
                    let new_heartbeat = Arc::new(AtomicU64::new(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    ));
                    current_heartbeat = new_heartbeat.clone();

                    // New typing indicator with watchdog
                    let typing_bot = bot.clone();
                    let new_typing_cancel = tokio_util::sync::CancellationToken::new();
                    current_typing_cancel = new_typing_cancel.clone();
                    // Associate typing token with the new queued task for cancel support
                    registry
                        .set_typing_cancel(current_task_id, new_typing_cancel.clone())
                        .await;
                    // Update the drop guard to track the new typing token
                    if let Ok(mut guard_token) = typing_guard_token.lock() {
                        *guard_token = new_typing_cancel.clone();
                    }
                    let heartbeat_for_queued = new_heartbeat;
                    tokio::spawn(async move {
                        let deadline = tokio::time::Instant::now() + Duration::from_secs(30 * 60);
                        loop {
                            let _ = typing_bot
                                .send_chat_action(chat_id, ChatAction::Typing)
                                .await;
                            tokio::select! {
                                _ = tokio::time::sleep(Duration::from_secs(3)) => {
                                    if stale_threshold_secs > 0 {
                                        let last_hb = heartbeat_for_queued.load(Ordering::Relaxed);
                                        let now = SystemTime::now().duration_since(UNIX_EPOCH)
                                            .unwrap_or_default().as_secs();
                                        if now.saturating_sub(last_hb) > stale_threshold_secs {
                                            break;
                                        }
                                    }
                                    if tokio::time::Instant::now() >= deadline {
                                        tracing::warn!(chat_id = %chat_id, "Queued typing indicator reached 30-minute hard limit");
                                        break;
                                    }
                                }
                                _ = new_typing_cancel.cancelled() => break,
                            }
                        }
                    });
                } else {
                    // No more queued messages, exit loop
                    break;
                }
            }
        });
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
impl Channel for TelegramChannel {
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
        let chat_id: i64 = crate::session::telegram_chat_id_from_session(session_id)
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
        let chat_id: i64 = crate::session::telegram_chat_id_from_session(session_id)
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
        let msg = match retry_telegram_rate_limit(|| {
            self.bot
                .send_message(ChatId(chat_id), &html)
                .parse_mode(ParseMode::Html)
        })
        .await
        {
            Ok(m) => m,
            Err(_) => {
                retry_telegram_rate_limit(|| self.bot.send_message(ChatId(chat_id), text)).await?
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
        let chat_id: i64 = crate::session::telegram_chat_id_from_session(session_id)
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
            .edit_message_text(ChatId(chat_id), teloxide::types::MessageId(mid), &html)
            .parse_mode(ParseMode::Html)
            .await
        {
            Ok(_) => Ok(true),
            // Telegram rejects an edit whose text is identical to the current
            // message — that's benign (nothing to change), so treat it as success.
            Err(e) if e.to_string().contains("message is not modified") => Ok(true),
            Err(_) => match self
                .bot
                .edit_message_text(ChatId(chat_id), teloxide::types::MessageId(mid), text)
                .await
            {
                Ok(_) => Ok(true),
                Err(e) if e.to_string().contains("message is not modified") => Ok(true),
                Err(e) => Err(e.into()),
            },
        }
    }

    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        let chat_id: i64 = crate::session::telegram_chat_id_from_session(session_id)
            .unwrap_or_else(|| {
                self.allowed_user_ids
                    .read()
                    .unwrap()
                    .first()
                    .copied()
                    .unwrap_or(0) as i64
            });
        match &media.kind {
            MediaKind::Photo { data } => {
                let photo = InputFile::memory(data.clone()).file_name("screenshot.png");
                retry_telegram_rate_limit(|| {
                    self.bot
                        .send_photo(ChatId(chat_id), photo.clone())
                        .caption(&media.caption)
                })
                .await
                .map_err(|e| anyhow::anyhow!("Failed to send photo: {}", e))?;
            }
            MediaKind::Document {
                file_path,
                filename,
            } => {
                let doc = InputFile::file(file_path).file_name(filename.clone());
                let mut req = self.bot.send_document(ChatId(chat_id), doc);
                if !media.caption.is_empty() {
                    req = req.caption(&media.caption);
                }
                retry_telegram_rate_limit(|| req.clone())
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to send document: {}", e))?;
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
    ) -> anyhow::Result<ApprovalResponse> {
        let chat_id: i64 = crate::session::telegram_chat_id_from_session(session_id)
            .unwrap_or_else(|| {
                self.allowed_user_ids
                    .read()
                    .unwrap()
                    .first()
                    .copied()
                    .unwrap_or(0) as i64
            });

        info!(session_id, command, chat_id, risk = %risk_level, mode = %permission_mode, "Approval requested");

        let approval_id = uuid::Uuid::new_v4().to_string();
        let short_id = &approval_id[..8];

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        // Store the response sender
        {
            let mut pending = self.pending_approvals.lock().await;
            pending.insert(approval_id.clone(), response_tx);
            info!(approval_id = %short_id, pending_count = pending.len(), "Stored pending approval");
        }

        // Determine which buttons to show based on permission_mode and risk_level
        // - Default mode: Critical gets [Once, Session, Deny], others get [Once, Always, Deny]
        // - Cautious mode: All get [Once, Session, Deny]
        // - YOLO mode: All get [Once, Always, Deny]
        let use_session_button =
            approval_render::approval_use_session_button(permission_mode, risk_level);

        let keyboard = approval_render::build_approval_keyboard(&approval_id, use_session_button);

        let text = approval_render::build_approval_message_text(
            command,
            risk_level,
            warnings,
            use_session_button,
        );

        match self
            .bot
            .send_message(ChatId(chat_id), &text)
            .parse_mode(ParseMode::Html)
            .reply_markup(keyboard)
            .await
        {
            Ok(_) => {
                info!(approval_id = %short_id, "Approval message sent to Telegram");
            }
            Err(e) => {
                warn!("Failed to send approval request: {}", e);
                // Remove pending approval since we couldn't ask
                let mut pending = self.pending_approvals.lock().await;
                pending.remove(&approval_id);
                return Ok(ApprovalResponse::Deny);
            }
        }

        // Wait for response with 5-minute timeout to prevent memory leak
        info!(approval_id = %short_id, "Waiting for user approval response...");
        match tokio::time::timeout(Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                warn!(approval_id = %short_id, "Approval channel closed");
                Ok(ApprovalResponse::Deny)
            }
            Err(_) => {
                warn!(approval_id = %short_id, "Approval timed out after 5 minutes");
                let mut pending = self.pending_approvals.lock().await;
                pending.remove(&approval_id);
                Ok(ApprovalResponse::Deny)
            }
        }
    }

    async fn request_goal_confirmation(
        &self,
        session_id: &str,
        goal_description: &str,
        details: &[String],
    ) -> anyhow::Result<bool> {
        let chat_id: i64 = crate::session::telegram_chat_id_from_session(session_id)
            .unwrap_or_else(|| {
                self.allowed_user_ids
                    .read()
                    .unwrap()
                    .first()
                    .copied()
                    .unwrap_or(0) as i64
            });

        let approval_id = uuid::Uuid::new_v4().to_string();
        let short_id = &approval_id[..8];

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        {
            let mut pending = self.pending_approvals.lock().await;
            pending.insert(approval_id.clone(), response_tx);
        }

        let keyboard = approval_render::build_goal_confirmation_keyboard(&approval_id);

        let text = approval_render::build_goal_confirmation_text(goal_description, details);

        match self
            .bot
            .send_message(ChatId(chat_id), &text)
            .parse_mode(ParseMode::Html)
            .reply_markup(keyboard)
            .await
        {
            Ok(_) => {
                info!(approval_id = %short_id, "Goal confirmation message sent");
            }
            Err(e) => {
                warn!("Failed to send goal confirmation: {}", e);
                let mut pending = self.pending_approvals.lock().await;
                pending.remove(&approval_id);
                return Ok(false);
            }
        }

        // Wait with 30-minute timeout (generous window to avoid race conditions
        // when users confirm near the timeout boundary).
        match tokio::time::timeout(Duration::from_secs(1800), response_rx).await {
            Ok(Ok(response)) => Ok(matches!(
                response,
                ApprovalResponse::AllowOnce
                    | ApprovalResponse::AllowSession
                    | ApprovalResponse::AllowAlways
            )),
            Ok(Err(_)) => {
                warn!(approval_id = %short_id, "Goal confirmation channel closed");
                Ok(false)
            }
            Err(_) => {
                warn!(approval_id = %short_id, "Goal confirmation timed out after 30 minutes");
                let mut pending = self.pending_approvals.lock().await;
                pending.remove(&approval_id);
                Ok(false)
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

    // This replaces the current process — does not return on success
    let err = std::process::Command::new(&exe).args(&args).exec();
    tracing::error!("exec failed: {}", err);
}

/// Send a message with HTML parse mode, falling back to plain text on failure.
async fn send_html_or_fallback(
    bot: &Bot,
    chat_id: ChatId,
    html: &str,
    plain: &str,
) -> Result<(), teloxide::RequestError> {
    match retry_telegram_rate_limit(|| bot.send_message(chat_id, html).parse_mode(ParseMode::Html))
        .await
    {
        Ok(_) => Ok(()),
        Err(e) => {
            warn!("HTML send failed, falling back to plain text: {}", e);
            retry_telegram_rate_limit(|| bot.send_message(chat_id, plain)).await?;
            Ok(())
        }
    }
}

async fn retry_telegram_rate_limit<Op, Req, T>(mut op: Op) -> Result<T, RequestError>
where
    Op: FnMut() -> Req,
    Req: std::future::IntoFuture<Output = Result<T, RequestError>>,
{
    match op().into_future().await {
        Ok(value) => Ok(value),
        Err(RequestError::RetryAfter(after)) => {
            let delay = crate::channels::rate_limit::bounded_retry_after(
                after.duration(),
                crate::channels::rate_limit::default_retry_after_cap(),
            );
            crate::channels::rate_limit::sleep_after_rate_limit("telegram", delay).await;
            op().into_future().await
        }
        Err(err) => Err(err),
    }
}

async fn send_markdown_chunks_or_fallback_result(
    bot: &Bot,
    chat_id: ChatId,
    markdown: &str,
) -> anyhow::Result<()> {
    let html = markdown_to_telegram_html(markdown);
    let html_chunks = split_message(&html, 4096);
    let plain_chunks = split_message(&markdown_to_telegram_plain_fallback(markdown), 4096);
    let mut first_err: Option<anyhow::Error> = None;

    for (i, html_chunk) in html_chunks.iter().enumerate() {
        let plain_chunk = plain_chunks
            .get(i)
            .map(|s| s.as_str())
            .unwrap_or(html_chunk.as_str());
        if let Err(e) = send_html_or_fallback(bot, chat_id, html_chunk, plain_chunk).await {
            warn!("Failed to send Telegram message: {}", e);
            if first_err.is_none() {
                first_err = Some(anyhow::anyhow!("Failed to send Telegram message: {}", e));
            }
        }
    }

    if let Some(err) = first_err {
        return Err(err);
    }
    Ok(())
}

/// Convert markdown to Telegram HTML and send with plain-text fallback.
async fn send_markdown_chunks_or_fallback(bot: &Bot, chat_id: ChatId, markdown: &str) {
    if let Err(e) = send_markdown_chunks_or_fallback_result(bot, chat_id, markdown).await {
        warn!("Failed to send Telegram message: {}", e);
    }
}

async fn send_full_or_expandable_reply(
    bot: &Bot,
    chat_id: ChatId,
    markdown: &str,
) -> anyhow::Result<()> {
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
) -> anyhow::Result<()> {
    let html_body = markdown_to_telegram_html(markdown);
    let plain = markdown_to_telegram_plain_fallback(markdown);
    let max_chunk = TELEGRAM_MAX_MESSAGE_LEN - TELEGRAM_EXPANDABLE_WRAPPER_LEN;
    let html_chunks = split_message(&html_body, max_chunk);
    let plain_chunks = split_message(&plain, max_chunk);
    let mut first_err: Option<anyhow::Error> = None;
    for (i, html_chunk) in html_chunks.iter().enumerate() {
        let wrapped = format!("<blockquote expandable>{}</blockquote>", html_chunk);
        let plain_chunk = plain_chunks
            .get(i)
            .map(|s| s.as_str())
            .unwrap_or(html_chunk.as_str());
        if let Err(e) = send_html_or_fallback(bot, chat_id, &wrapped, plain_chunk).await {
            warn!("Failed to send expandable Telegram message: {}", e);
            if first_err.is_none() {
                first_err = Some(anyhow::anyhow!("Failed to send Telegram message: {}", e));
            }
        }
    }

    if let Some(err) = first_err {
        return Err(err);
    }
    Ok(())
}

/// Spawn a TelegramChannel in a background task.
/// This is a separate function to avoid async type inference cycles.
pub fn spawn_telegram_channel(channel: Arc<TelegramChannel>) {
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

/// Pure role determination: Owner when `owner_ids` is empty (all allowed users
/// are owners) or when the user is explicitly listed; Guest otherwise.
pub fn determine_role(owner_ids: &[u64], user_id: u64) -> UserRole {
    if owner_ids.is_empty() || owner_ids.contains(&user_id) {
        UserRole::Owner
    } else {
        UserRole::Guest
    }
}

fn is_indicator_only_status(update: &StatusUpdate) -> bool {
    matches!(update, StatusUpdate::Thinking(_))
}

/// Single status consumer used at both the primary and queue-drain sites.
///
/// Ordinary progress events are folded into the `LiveStatus` single-surface owner
/// (one message created lazily and edited in place). BudgetExtended and URL-bearing
/// OAuth links are must-deliver and sent as their own messages. All existing gating
/// is preserved: `is_indicator_only_status`, non-DM suppression, 3 s `min_interval`,
/// and the BudgetExtended / URL bypass.
async fn run_status_consumer(
    mut rx: tokio::sync::mpsc::Receiver<StatusUpdate>,
    owner: std::sync::Arc<tokio::sync::Mutex<crate::channels::live_status::LiveStatus>>,
    sink: std::sync::Arc<crate::channels::live_status::HubSurfaceSink>,
    bot: Bot,
    chat_id: ChatId,
    is_dm: bool,
) {
    use crate::channels::live_status::SurfaceSink;
    let mut last_sent = tokio::time::Instant::now() - Duration::from_secs(10);
    let min_interval = Duration::from_secs(3);
    while let Some(update) = rx.recv().await {
        if is_indicator_only_status(&update) {
            let _ = bot.send_chat_action(chat_id, ChatAction::Typing).await;
            continue;
        }
        let has_url = matches!(&update, StatusUpdate::ToolProgress { chunk, .. }
            if chunk.contains("https://") || chunk.contains("http://"));
        let is_budget_ext = matches!(&update, StatusUpdate::BudgetExtended { .. });
        // Non-DM channels: suppress progress except BudgetExtended (unchanged behaviour).
        if !is_dm && !is_budget_ext {
            continue;
        }
        // BudgetExtended and URL-bearing OAuth links are must-deliver: send as
        // their own message (never folded into the live surface).
        if is_budget_ext || has_url {
            let text = match &update {
                StatusUpdate::BudgetExtended {
                    old_budget,
                    new_budget,
                    extension,
                    max_extensions,
                } => format!(
                    "💰 Auto-extended token budget {old_budget} → {new_budget} \
                     ({extension}/{max_extensions}) — continuing."
                ),
                StatusUpdate::ToolProgress { name, chunk } => format!("📤 {name}\n{chunk}"),
                _ => String::new(),
            };
            if !text.is_empty() {
                let _ = bot.send_message(chat_id, text).await;
            }
            continue;
        }
        // Ordinary progress → single live surface, gated by min_interval.
        // Exception: the plan checklist is the highest-value content and edits in
        // place, so it must never be dropped by the throttle. It typically arrives
        // within a second of the track_requirements ToolStart (well inside
        // min_interval), which previously discarded it and the plan never showed.
        let now = tokio::time::Instant::now();
        let is_checklist = matches!(&update, StatusUpdate::Checklist { .. });
        if !is_checklist && now.duration_since(last_sent) < min_interval {
            continue;
        }
        let sink_ref: &dyn SurfaceSink = sink.as_ref();
        // Single-writer invariant: exactly one consumer task touches `owner` per
        // turn and turns are sequential (the previous task is aborted before the
        // next is spawned), so holding the lock across the network send below is
        // safe — there is never a second writer to contend with.
        match &update {
            StatusUpdate::Checklist { text } => {
                owner
                    .lock()
                    .await
                    .set_checklist(sink_ref, text.clone())
                    .await;
            }
            StatusUpdate::ToolStart { name, summary } => {
                let line = if summary.is_empty() {
                    format!("{name}…")
                } else {
                    format!("{name}: {summary}…")
                };
                owner.lock().await.set_activity(sink_ref, line).await;
            }
            StatusUpdate::ToolComplete { name, .. } => {
                owner
                    .lock()
                    .await
                    .set_activity(sink_ref, format!("{name}…"))
                    .await;
            }
            StatusUpdate::PlanStepStart { description, .. }
            | StatusUpdate::PlanStepComplete { description, .. } => {
                owner
                    .lock()
                    .await
                    .set_activity(sink_ref, description.clone())
                    .await;
            }
            _ => {
                continue; // other Plan*/Thinking variants: no separate message
            }
        }
        last_sent = tokio::time::Instant::now();
        let _ = bot.send_chat_action(chat_id, ChatAction::Typing).await;
    }
}

/// Legacy (pre-LiveStatus) status consumer used only when the `ChannelHub` is
/// not wired and the single live surface is unavailable. Sends a fresh message
/// per status update (heavyweight `StatusUpdate` match) with the historical
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
        // Skip rate limiting for ToolProgress with URLs (e.g., OAuth authorize links)
        // and for BudgetExtended (cost notifications should always be delivered)
        let has_url = matches!(&update, StatusUpdate::ToolProgress { chunk, .. }
            if chunk.contains("https://") || chunk.contains("http://"));
        let is_budget_ext = matches!(&update, StatusUpdate::BudgetExtended { .. });
        // Hard cap on DM status messages to prevent notification spam.
        // BudgetExtended and URL-containing messages always bypass the cap.
        if !has_url && !is_budget_ext && dm_status_count >= MAX_DM_STATUS_MESSAGES {
            // After the cap, just send typing indicator instead
            let _ = bot.send_chat_action(chat_id, ChatAction::Typing).await;
            continue;
        }
        if !has_url && !is_budget_ext && now.duration_since(last_sent) < min_interval {
            continue;
        }
        let text = match &update {
            StatusUpdate::Thinking(_) => continue,
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
                // Don't truncate if the chunk contains a URL (e.g., OAuth authorize links)
                if chunk.contains("https://") || chunk.contains("http://") {
                    format!("📤 {}\n{}", name, chunk)
                } else {
                    let preview: String = chunk.chars().take(100).collect();
                    if chunk.len() > 100 {
                        format!("📤 {}: {}...", name, preview)
                    } else {
                        format!("📤 {}: {}", name, preview)
                    }
                }
            }
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

/// Tracks whether Telegram long-polling has been failing continuously long
/// enough to be considered "wedged" — e.g. a persistent getUpdates 409 after a
/// restart that teloxide's internal retry never recovers from. Pure and
/// testable: the caller supplies the current cumulative error count and `now`.
#[derive(Default)]
struct PollHealth {
    streak_started_at: Option<Instant>,
    last_count: u64,
}

impl PollHealth {
    fn new() -> Self {
        Self::default()
    }

    /// Returns true when getUpdates has been erroring continuously for at least
    /// `threshold`. A tick with no new errors clears the streak (teloxide
    /// recovered on its own).
    fn observe(&mut self, current_count: u64, now: Instant, threshold: Duration) -> bool {
        if current_count > self.last_count {
            let started = *self.streak_started_at.get_or_insert(now);
            self.last_count = current_count;
            now.saturating_duration_since(started) >= threshold
        } else {
            self.streak_started_at = None;
            self.last_count = current_count;
            false
        }
    }
}

/// Error handler for the Telegram polling listener. teloxide's default handler
/// logs under the `teloxide` target, which our tracing filter drops — that's how
/// a wedged getUpdates went completely silent. This logs loudly under our own
/// target and counts errors so the watchdog can force a reconnect.
struct PollingErrorHandler {
    bot_username: String,
    error_count: Arc<AtomicU64>,
}

impl<E: std::fmt::Debug> ErrorHandler<E> for PollingErrorHandler {
    fn handle_error(self: Arc<Self>, error: E) -> BoxFuture<'static, ()> {
        let n = self.error_count.fetch_add(1, Ordering::Relaxed) + 1;
        // Loud on the first error, then periodically, so a sustained wedge is
        // visible without flooding the log on every retry.
        if n == 1 || n.is_multiple_of(10) {
            warn!(
                name = %self.bot_username,
                consecutive_errors = n,
                error = ?error,
                "Telegram getUpdates polling error"
            );
        }
        Box::pin(async {})
    }
}

/// Watchdog that returns once Telegram polling has been failing continuously
/// past the wedge threshold, so the caller (`start_with_retry`) can restart the
/// dispatcher with a fresh getUpdates connection. A healthy or merely quiet bot
/// keeps it parked indefinitely (it only trips on *sustained* errors).
async fn poll_health_watchdog(error_count: Arc<AtomicU64>, bot_username: String) {
    const TICK: Duration = Duration::from_secs(30);
    const WEDGE_THRESHOLD: Duration = Duration::from_secs(120);
    let mut health = PollHealth::new();
    loop {
        tokio::time::sleep(TICK).await;
        let count = error_count.load(Ordering::Relaxed);
        if health.observe(count, Instant::now(), WEDGE_THRESHOLD) {
            warn!(
                name = %bot_username,
                total_errors = count,
                threshold_secs = WEDGE_THRESHOLD.as_secs(),
                "Telegram long-polling wedged (getUpdates failing continuously); restarting dispatcher to re-establish the connection"
            );
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poll_health_flags_wedge_after_sustained_errors() {
        let mut h = PollHealth::new();
        let t0 = Instant::now();
        let threshold = Duration::from_secs(120);
        // Errors begin.
        assert!(!h.observe(1, t0, threshold));
        // Still erroring, but within the threshold — give teloxide time to recover.
        assert!(!h.observe(3, t0 + Duration::from_secs(60), threshold));
        // Errors sustained past the threshold — wedged, force a restart.
        assert!(h.observe(5, t0 + Duration::from_secs(121), threshold));
    }

    #[test]
    fn poll_health_clears_streak_when_errors_stop() {
        let mut h = PollHealth::new();
        let t0 = Instant::now();
        let threshold = Duration::from_secs(120);
        assert!(!h.observe(1, t0, threshold));
        // A tick with no new errors means teloxide recovered: clear the streak.
        assert!(!h.observe(1, t0 + Duration::from_secs(60), threshold));
        // A later error starts a fresh streak, so it is not immediately a wedge
        // even though wall-clock is now past t0 + threshold.
        assert!(!h.observe(2, t0 + Duration::from_secs(130), threshold));
    }

    #[test]
    fn thinking_status_uses_native_telegram_indicator_only() {
        assert!(is_indicator_only_status(&StatusUpdate::Thinking(2)));
    }

    #[test]
    fn normalize_agent_permission_aliases_maps_claude_allow_flag() {
        let args = vec![
            "--allow-dangerously-skip-permissions".to_string(),
            "--model".to_string(),
            "sonnet".to_string(),
        ];
        let (normalized, rewrote) =
            crate::normalize_terminal_agent_permission_aliases(Some("claude"), args);
        assert!(rewrote);
        assert_eq!(
            normalized,
            vec![
                "--dangerously-skip-permissions".to_string(),
                "--model".to_string(),
                "sonnet".to_string()
            ]
        );
    }

    #[test]
    fn normalize_agent_permission_aliases_leaves_non_claude_args_unchanged() {
        let args = vec!["--allow-dangerously-skip-permissions".to_string()];
        let (normalized, rewrote) =
            crate::normalize_terminal_agent_permission_aliases(Some("codex"), args.clone());
        assert!(!rewrote);
        assert_eq!(normalized, args);
    }

    // --- check_auth ---

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
        match TelegramChannel::parse_web_app_action(payload) {
            Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                assert!(relay_session_id.is_none());
            }
            other => panic!("unexpected parsed action: {:?}", other.map(|_| "unknown")),
        }
    }

    #[test]
    fn parse_web_app_action_extracts_agent_message() {
        let payload = r#"{"type":"aidaemon.telegram.agent_message.v1","text":"Continue fixing the failing tests."}"#;
        match TelegramChannel::parse_web_app_action(payload) {
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
        match TelegramChannel::parse_web_app_action(payload) {
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
        match TelegramChannel::parse_web_app_action(payload) {
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
        match TelegramChannel::parse_web_app_action(payload) {
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
        match TelegramChannel::parse_web_app_action(payload) {
            Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                assert_eq!(relay_session_id.as_deref(), Some("sess_legacy"));
            }
            other => panic!("unexpected parsed action: {:?}", other.map(|_| "unknown")),
        }
    }

    #[test]
    fn parse_web_app_action_accepts_nested_payload_string() {
        let payload = r#"{"data":"{\"type\":\"aidaemon.telegram.open_on_computer.v1\",\"relay_session_id\":\"sess_nested\"}"}"#;
        match TelegramChannel::parse_web_app_action(payload) {
            Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                assert_eq!(relay_session_id.as_deref(), Some("sess_nested"));
            }
            other => panic!("unexpected parsed action: {:?}", other.map(|_| "unknown")),
        }
    }

    #[test]
    fn parse_web_app_action_rejects_unknown_payload() {
        let payload = r#"{"type":"unknown","text":"hello"}"#;
        assert!(TelegramChannel::parse_web_app_action(payload).is_none());
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

    #[test]
    fn strip_low_latency_next_steps_removes_manual_instructions() {
        let text = "Low-latency webhook config applied (local config only).\nBackup: config.toml.lowlatency.bak\n\nNext steps:\n1. step one\n2. step two\n";
        let stripped = TelegramChannel::strip_low_latency_next_steps(text);
        assert_eq!(
            stripped,
            "Low-latency webhook config applied (local config only).\nBackup: config.toml.lowlatency.bak"
        );
    }

    #[test]
    fn strip_low_latency_next_steps_leaves_text_without_marker() {
        let text = "Low-latency webhook config applied (local config only).\nBackup: config.toml.lowlatency.bak\n";
        let stripped = TelegramChannel::strip_low_latency_next_steps(text);
        assert_eq!(stripped, text);
    }

    #[tokio::test]
    async fn backup_cloudflared_origin_cert_moves_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");
        tokio::fs::write(&cert_path, "dummy-cert").await.unwrap();

        let backup = TelegramChannel::backup_cloudflared_origin_cert_if_present(&cert_path)
            .await
            .unwrap()
            .unwrap();

        assert!(tokio::fs::metadata(&cert_path).await.is_err());
        let backup_content = tokio::fs::read_to_string(&backup).await.unwrap();
        assert_eq!(backup_content, "dummy-cert");
    }

    #[tokio::test]
    async fn backup_cloudflared_origin_cert_is_noop_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");

        let backup = TelegramChannel::backup_cloudflared_origin_cert_if_present(&cert_path)
            .await
            .unwrap();
        assert!(backup.is_none());
    }

    // --- persist_allowed_user_ids (config file update) ---

    #[tokio::test]
    async fn persist_updates_legacy_telegram_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        let initial = r#"
[telegram]
bot_token = "test-token"
allowed_user_ids = []
"#;
        tokio::fs::write(&config_path, initial).await.unwrap();

        // Parse and re-write with the helper logic
        let content = tokio::fs::read_to_string(&config_path).await.unwrap();
        let mut doc: toml::Table = content.parse().unwrap();
        let ids = [12345u64];
        let ids_toml = toml::Value::Array(
            ids.iter()
                .map(|&id| toml::Value::Integer(id as i64))
                .collect(),
        );
        if let Some(tg) = doc.get_mut("telegram").and_then(|v| v.as_table_mut()) {
            tg.insert("allowed_user_ids".to_string(), ids_toml);
        }
        let new_content = toml::to_string_pretty(&toml::Value::Table(doc)).unwrap();
        tokio::fs::write(&config_path, &new_content).await.unwrap();

        // Verify
        let saved = tokio::fs::read_to_string(&config_path).await.unwrap();
        let doc: toml::Table = saved.parse().unwrap();
        let ids_val = doc["telegram"]["allowed_user_ids"].as_array().unwrap();
        assert_eq!(ids_val.len(), 1);
        assert_eq!(ids_val[0].as_integer().unwrap(), 12345);
    }

    #[tokio::test]
    async fn persist_updates_telegram_bots_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        let initial = r#"
[[telegram_bots]]
bot_token = "test-token"
allowed_user_ids = []
"#;
        tokio::fs::write(&config_path, initial).await.unwrap();

        let content = tokio::fs::read_to_string(&config_path).await.unwrap();
        let mut doc: toml::Table = content.parse().unwrap();
        let ids = [67890u64];
        let ids_toml = toml::Value::Array(
            ids.iter()
                .map(|&id| toml::Value::Integer(id as i64))
                .collect(),
        );
        if let Some(bots) = doc.get_mut("telegram_bots").and_then(|v| v.as_array_mut()) {
            if let Some(first) = bots.first_mut().and_then(|v| v.as_table_mut()) {
                first.insert("allowed_user_ids".to_string(), ids_toml);
            }
        }
        let new_content = toml::to_string_pretty(&toml::Value::Table(doc)).unwrap();
        tokio::fs::write(&config_path, &new_content).await.unwrap();

        let saved = tokio::fs::read_to_string(&config_path).await.unwrap();
        let doc: toml::Table = saved.parse().unwrap();
        let bots = doc["telegram_bots"].as_array().unwrap();
        let ids_val = bots[0]["allowed_user_ids"].as_array().unwrap();
        assert_eq!(ids_val.len(), 1);
        assert_eq!(ids_val[0].as_integer().unwrap(), 67890);
    }
}
