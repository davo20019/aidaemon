use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock as StdRwLock, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use chrono::Utc;
use once_cell::sync::Lazy;
use regex::Regex;
use teloxide::prelude::*;
use teloxide::types::{
    ButtonRequest, ChatAction, InlineKeyboardButton, InlineKeyboardMarkup, InputFile,
    KeyboardButton, KeyboardMarkup, ParseMode, WebAppInfo,
};
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use super::formatting::{
    build_help_text, format_number, html_escape, markdown_to_telegram_html, sanitize_filename,
    split_message, strip_latex,
};
use crate::agent::Agent;
use crate::channels::{should_ignore_lightweight_interjection, ChannelHub, SessionMap};
#[cfg(feature = "discord")]
use crate::channels::{spawn_discord_channel, DiscordChannel};
#[cfg(feature = "slack")]
use crate::channels::{spawn_slack_channel, SlackChannel};
use crate::config::AppConfig;
use crate::tasks::TaskRegistry;
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
    /// Chat-only terminal-lite sessions keyed by Telegram chat id.
    terminal_lite_sessions: Mutex<HashMap<i64, TerminalLiteSession>>,
    /// Commands allowed in `/terminal lite`, derived from `[terminal].allowed_prefixes`.
    terminal_allowed_prefixes: HashSet<String>,
    /// Daemon start time used for post-restart UX guardrails.
    started_at: Instant,
}

#[derive(Debug, Clone)]
struct TerminalLiteSession {
    owner_user_id: u64,
    cwd: PathBuf,
    shell: String,
    preferred_agent: Option<String>,
    started_at: Instant,
    busy: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct AgentFlagDoc {
    flag: String,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct AgentFlagsCacheEntry {
    updated_at_unix: i64,
    #[serde(default)]
    flags: Vec<String>,
    #[serde(default)]
    docs: Vec<AgentFlagDoc>,
}

const TERMINAL_LITE_MAX_OUTPUT_CHARS: usize = 12_000;
const TERMINAL_LITE_TIMEOUT_SECS: u64 = 90;
const DEFAULT_TERMINAL_AGENT: &str = "codex";
const SUPPORTED_TERMINAL_AGENTS: &[&str] = &["codex", "claude", "gemini", "opencode"];
const MAX_TERMINAL_AGENT_ARGS: usize = 24;
const MAX_TERMINAL_AGENT_ARG_CHARS: usize = 256;
const AGENT_FLAGS_CACHE_TTL_SECS: i64 = 24 * 60 * 60;
const MAX_DISCOVERED_AGENT_FLAGS: usize = 512;
const MAX_DISCOVERED_AGENT_FLAG_CHARS: usize = 96;
const MAX_DISCOVERED_AGENT_FLAG_DESC_CHARS: usize = 240;
const AGENT_FLAGS_PAGE_SIZE: usize = 12;
const TELEGRAM_EXPANDABLE_TRIGGER_CHARS: usize = 1_800;
const TELEGRAM_MAX_MESSAGE_LEN: usize = 4096;
const TELEGRAM_EXPANDABLE_WRAPPER_LEN: usize = "<blockquote expandable></blockquote>".len();
const TELEGRAM_EXPANDABLE_MAX_ESCAPED_CHARS: usize =
    TELEGRAM_MAX_MESSAGE_LEN - TELEGRAM_EXPANDABLE_WRAPPER_LEN;
const TELEGRAM_WEBAPP_TYPE_AGENT_MESSAGE: &str = "aidaemon.telegram.agent_message.v1";
const TELEGRAM_WEBAPP_TYPE_CONTINUE_COMPUTER: &str = "aidaemon.telegram.open_on_computer.v1";
const TELEGRAM_WEBAPP_MAX_TEXT_CHARS: usize = 2_000;

enum TelegramWebAppAction {
    AgentMessage(String),
    ContinueOnComputer { relay_session_id: Option<String> },
}

impl TelegramChannel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bot_token: &str,
        allowed_user_ids: Vec<u64>,
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
            terminal_lite_sessions: Mutex::new(HashMap::new()),
            terminal_allowed_prefixes,
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

        let initial_backoff = Duration::from_secs(5);
        let max_backoff = Duration::from_secs(60);
        let stable_threshold = Duration::from_secs(60);
        let mut backoff = initial_backoff;

        loop {
            info!(name = %bot_username, "Starting Telegram dispatcher");
            let started = tokio::time::Instant::now();
            self.clone().start().await;
            let ran_for = started.elapsed();

            // If the dispatcher ran for long enough, it was a stable session —
            // reset backoff so the next crash recovers quickly.
            if ran_for >= stable_threshold {
                backoff = initial_backoff;
            }

            warn!(
                name = %bot_username,
                backoff_secs = backoff.as_secs(),
                ran_for_secs = ran_for.as_secs(),
                "Telegram dispatcher stopped, restarting"
            );
            tokio::time::sleep(backoff).await;
            backoff = std::cmp::min(backoff * 2, max_backoff);
        }
    }

    pub async fn start(self: Arc<Self>) {
        let bot_username = self.get_bot_username().await;
        info!(name = %bot_username, "Starting Telegram channel");

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

        Dispatcher::builder(self.bot.clone(), handler)
            .enable_ctrlc_handler()
            .build()
            .dispatch()
            .await;
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
        if parts.len() != 3 || (parts[0] != "approve" && parts[0] != "goal") {
            return;
        }

        let prefix = parts[0];
        let action = parts[1];
        let approval_id = parts[2];

        let (response, label) = if prefix == "goal" {
            match action {
                "confirm" => (ApprovalResponse::AllowOnce, "Confirmed ✅"),
                "cancel" => (ApprovalResponse::Deny, "Cancelled ❌"),
                _ => return,
            }
        } else {
            let response = match action {
                "once" => ApprovalResponse::AllowOnce,
                "session" => ApprovalResponse::AllowSession,
                "always" => ApprovalResponse::AllowAlways,
                "deny" => ApprovalResponse::Deny,
                _ => return,
            };
            let label = match &response {
                ApprovalResponse::AllowOnce => "Allowed (once)",
                ApprovalResponse::AllowSession => "Allowed (this session)",
                ApprovalResponse::AllowAlways => "Allowed (always)",
                ApprovalResponse::Deny => "Denied",
            };
            (response, label)
        };

        // Send the response
        let mut pending = self.pending_approvals.lock().await;
        if let Some(tx) = pending.remove(approval_id) {
            let _ = tx.send(response);
        } else {
            warn!(approval_id, "Stale approval callback (no pending request)");
        }

        // Acknowledge the callback and update the message
        let _ = bot.answer_callback_query(q.id).text(label).await;

        if let Some(teloxide::types::MaybeInaccessibleMessage::Regular(m)) = q.message {
            let original = m.text().unwrap_or("");
            let _ = bot
                .edit_message_text(m.chat.id, m.id, format!("{} — {}", original, label))
                .await;
        }
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

        let reply = match cmd {
            "/model" => {
                if arg.is_empty() {
                    let current = self.agent.current_model().await;
                    format!(
                        "Current model: {}\n\nUsage: /model <model-name>\nExample: /model gemini-3-pro-preview",
                        current
                    )
                } else {
                    self.agent.set_model(arg.to_string()).await;
                    format!(
                        "Model switched to: {}\nAuto-routing disabled. Use /auto to re-enable.",
                        arg
                    )
                }
            }
            "/models" => match self.agent.list_models().await {
                Ok(models) => {
                    if models.is_empty() {
                        "No models found from provider.".to_string()
                    } else {
                        let current = self.agent.current_model().await;
                        let list: Vec<String> = models
                            .iter()
                            .map(|m| {
                                if *m == current {
                                    format!("• {} (active)", m)
                                } else {
                                    format!("• {}", m)
                                }
                            })
                            .collect();
                        format!("Available models:\n{}", list.join("\n"))
                    }
                }
                Err(e) => format!("Failed to list models: {}", e),
            },
            "/auto" => {
                self.agent.clear_model_override().await;
                "Auto-routing re-enabled. Model will be selected automatically based on query complexity.".to_string()
            }
            "/reload" => match AppConfig::load(&self.config_path) {
                Ok(new_config) => match self.agent.reload_provider(&new_config).await {
                    Ok(status) => format!("Config reloaded. {}", status),
                    Err(e) => format!("Provider reload failed: {}", e),
                },
                Err(e) => {
                    // Config is broken — try to auto-restore from backup
                    let backup = self.config_path.with_extension("toml.bak");
                    if backup.exists() {
                        if tokio::fs::copy(&backup, &self.config_path).await.is_ok() {
                            format!(
                                "Config reload failed: {}\n\nAuto-restored from backup. Config is back to the previous working state.",
                                e
                            )
                        } else {
                            format!(
                                "Config reload failed: {}\n\nBackup restore also failed. Manual intervention needed.",
                                e
                            )
                        }
                    } else {
                        format!("Config reload failed: {}\n\nNo backup available.", e)
                    }
                }
            },
            "/restart" => {
                let _ = bot.send_message(msg.chat.id, "Restarting...").await;
                info!("Restart requested via Telegram");
                restart_process();
                // If exec fails, we're still alive
                "Restart failed. You may need to restart manually.".to_string()
            }
            "/tasks" => {
                let session_id = self.session_id(msg.chat.id.0).await;
                let entries = self.task_registry.list_for_session(&session_id).await;
                if entries.is_empty() {
                    "No tasks found.".to_string()
                } else {
                    let lines: Vec<String> = entries
                        .iter()
                        .map(|e| {
                            let elapsed = match e.finished_at {
                                Some(fin) => {
                                    let d = fin - e.started_at;
                                    format!("{}s", d.num_seconds())
                                }
                                None => {
                                    let d = Utc::now() - e.started_at;
                                    format!("{}s elapsed", d.num_seconds())
                                }
                            };
                            format!("#{} [{}] {} ({})", e.id, e.status, e.description, elapsed)
                        })
                        .collect();
                    lines.join("\n")
                }
            }
            "/cancel" => {
                if arg.is_empty() {
                    "Usage: /cancel <task-id>\nExample: /cancel 1".to_string()
                } else {
                    match arg.parse::<u64>() {
                        Ok(task_id) => {
                            if self.task_registry.cancel(task_id).await {
                                format!("Task #{} cancelled.", task_id)
                            } else {
                                format!("Task #{} not found or not running.", task_id)
                            }
                        }
                        Err(_) => "Invalid task ID. Usage: /cancel <task-id>".to_string(),
                    }
                }
            }
            "/clear" => {
                let session_id = self.session_id(msg.chat.id.0).await;
                match self.agent.clear_session(&session_id).await {
                    Ok(_) => "Context cleared. Starting fresh.".to_string(),
                    Err(e) => format!("Failed to clear context: {}", e),
                }
            }
            "/cost" => self.handle_cost_command().await,
            "/connect" => {
                self.handle_connect_command(arg, msg.from.as_ref().map(|u| u.id.0).unwrap_or(0))
                    .await
            }
            "/bots" => self.handle_bots_command().await,
            "/help" | "/start" => build_help_text(true, true, true, "/"),
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

    fn terminal_lite_help_text() -> String {
        "Terminal lite mode (owner only, chat-based)\n\n\
         Commands:\n\
         /terminal lite start [working_dir]\n\
         /terminal lite start <codex|claude|gemini|opencode> [working_dir]\n\
         /terminal lite [working_dir]\n\
         /terminal lite <codex|claude|gemini|opencode> [working_dir]\n\
         /terminal lite status\n\
         /terminal lite stop\n\n\
         After start, every non-slash message in this chat is treated as a shell command.\n\
         Built-ins: cd <path>, exit, quit\n\
         Note: interactive agent TUIs (codex/claude/gemini/opencode) require full /terminal Mini App mode.\n\
         Timeout: 90s per command."
            .to_string()
    }

    fn normalize_terminal_agent_name(value: &str) -> Option<String> {
        let v = value.trim().to_ascii_lowercase();
        if SUPPORTED_TERMINAL_AGENTS.contains(&v.as_str()) {
            Some(v)
        } else {
            None
        }
    }

    fn normalize_terminal_agent_args(values: Vec<String>) -> Vec<String> {
        values
            .into_iter()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty() && !v.contains('\0'))
            .map(|v| {
                v.chars()
                    .take(MAX_TERMINAL_AGENT_ARG_CHARS)
                    .collect::<String>()
            })
            .take(MAX_TERMINAL_AGENT_ARGS)
            .collect()
    }

    fn normalize_discovered_agent_flags(values: Vec<String>) -> Vec<String> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for value in values {
            let cleaned = value.trim().to_string();
            if cleaned.is_empty() || !cleaned.starts_with("--") || cleaned.contains('\0') {
                continue;
            }
            let clipped = cleaned
                .chars()
                .take(MAX_DISCOVERED_AGENT_FLAG_CHARS)
                .collect::<String>();
            if seen.insert(clipped.clone()) {
                out.push(clipped);
            }
            if out.len() >= MAX_DISCOVERED_AGENT_FLAGS {
                break;
            }
        }
        out
    }

    fn normalize_agent_flag_docs(values: Vec<AgentFlagDoc>) -> Vec<AgentFlagDoc> {
        let mut out: Vec<AgentFlagDoc> = Vec::new();
        let mut seen: HashMap<String, usize> = HashMap::new();

        for value in values {
            let normalized_flags = Self::normalize_discovered_agent_flags(vec![value.flag]);
            let Some(flag) = normalized_flags.first().cloned() else {
                continue;
            };

            let description = value
                .description
                .map(|d| d.split_whitespace().collect::<Vec<_>>().join(" "))
                .map(|d| d.trim().to_string())
                .filter(|d| !d.is_empty())
                .map(|d| {
                    d.chars()
                        .take(MAX_DISCOVERED_AGENT_FLAG_DESC_CHARS)
                        .collect::<String>()
                });

            if let Some(idx) = seen.get(&flag).copied() {
                let existing = out.get_mut(idx).expect("index from seen must exist");
                let replace = match (&existing.description, &description) {
                    (None, Some(_)) => true,
                    (Some(old), Some(new)) => new.len() > old.len(),
                    _ => false,
                };
                if replace {
                    existing.description = description;
                }
            } else {
                seen.insert(flag.clone(), out.len());
                out.push(AgentFlagDoc { flag, description });
                if out.len() >= MAX_DISCOVERED_AGENT_FLAGS {
                    break;
                }
            }
        }

        out
    }

    fn format_agent_flag_docs(agent: &str, docs: &[AgentFlagDoc], cached: bool) -> Vec<String> {
        if docs.is_empty() {
            return vec![format!(
                "### No flags found for `{}`{}\nTry `/agent flags {} refresh`.",
                agent,
                if cached { " (cached)" } else { "" },
                agent
            )];
        }

        let page_size = AGENT_FLAGS_PAGE_SIZE.max(1);
        let total = docs.len();
        let total_pages = total.div_ceil(page_size);
        let mut pages = Vec::new();

        for (idx, chunk) in docs.chunks(page_size).enumerate() {
            let mut lines = Vec::new();
            lines.push(format!(
                "### Flags for `{}`{}",
                agent,
                if cached { " (cached)" } else { "" }
            ));
            lines.push(format!("**{} total**", total));
            if total_pages > 1 {
                let start = idx * page_size + 1;
                let end = start + chunk.len() - 1;
                lines.push(format!(
                    "**Showing {}-{} of {} (page {}/{})**",
                    start,
                    end,
                    total,
                    idx + 1,
                    total_pages
                ));
            }
            lines.push(String::new());

            for doc in chunk {
                lines.push(format!("- `{}`", doc.flag));
                if let Some(desc) = doc.description.as_deref() {
                    lines.push(format!("Description: {}", desc));
                }
                lines.push(String::new());
            }

            if idx + 1 == total_pages {
                lines.push(
                    "Set defaults with `/agent defaults set <agent> [flags...]`.".to_string(),
                );
                lines.push("Bypass once with `--no-default-flags`.".to_string());
                lines.push("Refresh with `/agent flags <agent> refresh`.".to_string());
            }

            while lines.last().map(|line| line.is_empty()).unwrap_or(false) {
                lines.pop();
            }
            pages.push(lines.join("\n"));
        }

        pages
    }

    fn strip_no_default_flag(agent_args: &mut Vec<String>) -> bool {
        let before = agent_args.len();
        agent_args.retain(|arg| arg != "--no-default-flags");
        before != agent_args.len()
    }

    async fn terminal_agent_defaults_key(&self, chat_id: i64, user_id: u64) -> String {
        let scope = self.session_namespace().await;
        format!("telegram:agent_defaults:{}:{}:{}", scope, user_id, chat_id)
    }

    async fn load_terminal_agent_defaults(
        &self,
        chat_id: i64,
        user_id: u64,
    ) -> HashMap<String, Vec<String>> {
        let key = self.terminal_agent_defaults_key(chat_id, user_id).await;
        let raw = match self.state.get_setting(&key).await {
            Ok(Some(v)) => v,
            _ => return HashMap::new(),
        };
        let parsed: HashMap<String, Vec<String>> =
            serde_json::from_str(&raw).unwrap_or_else(|_| HashMap::new());
        let mut sanitized = HashMap::new();
        for (agent, args) in parsed {
            let Some(agent_name) = Self::normalize_terminal_agent_name(&agent) else {
                continue;
            };
            let cleaned = Self::normalize_terminal_agent_args(args);
            if cleaned.is_empty() {
                continue;
            }
            sanitized.insert(agent_name, cleaned);
        }
        sanitized
    }

    async fn save_terminal_agent_defaults(
        &self,
        chat_id: i64,
        user_id: u64,
        defaults: &HashMap<String, Vec<String>>,
    ) -> anyhow::Result<()> {
        let key = self.terminal_agent_defaults_key(chat_id, user_id).await;
        let serialized = serde_json::to_string(defaults)?;
        self.state.set_setting(&key, &serialized).await
    }

    async fn agent_flags_cache_key(&self, user_id: u64, agent: &str) -> String {
        let scope = self.session_namespace().await;
        format!(
            "telegram:agent_flags_cache:{}:{}:{}",
            scope,
            user_id,
            agent.to_ascii_lowercase()
        )
    }

    async fn load_agent_flags_cache(
        &self,
        user_id: u64,
        agent: &str,
    ) -> Option<AgentFlagsCacheEntry> {
        let key = self.agent_flags_cache_key(user_id, agent).await;
        let raw = self.state.get_setting(&key).await.ok().flatten()?;
        let mut parsed: AgentFlagsCacheEntry = serde_json::from_str(&raw).ok()?;
        let mut docs = Self::normalize_agent_flag_docs(parsed.docs.clone());
        if docs.is_empty() && !parsed.flags.is_empty() {
            docs = Self::normalize_discovered_agent_flags(parsed.flags.clone())
                .into_iter()
                .map(|flag| AgentFlagDoc {
                    flag,
                    description: None,
                })
                .collect();
        }
        if docs.is_empty() {
            return None;
        }
        parsed.docs = docs.clone();
        parsed.flags = docs.iter().map(|d| d.flag.clone()).collect();
        Some(parsed)
    }

    async fn save_agent_flags_cache(
        &self,
        user_id: u64,
        agent: &str,
        docs: &[AgentFlagDoc],
    ) -> anyhow::Result<()> {
        let key = self.agent_flags_cache_key(user_id, agent).await;
        let normalized_docs = Self::normalize_agent_flag_docs(docs.to_vec());
        let payload = AgentFlagsCacheEntry {
            updated_at_unix: chrono::Utc::now().timestamp(),
            flags: normalized_docs.iter().map(|d| d.flag.clone()).collect(),
            docs: normalized_docs,
        };
        let serialized = serde_json::to_string(&payload)?;
        self.state.set_setting(&key, &serialized).await
    }

    fn extract_long_flags_from_help(help_text: &str) -> Vec<String> {
        static LONG_FLAG_RE: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"--[a-zA-Z0-9][a-zA-Z0-9\-]*").expect("valid long flag regex")
        });
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for cap in LONG_FLAG_RE.find_iter(help_text) {
            let flag = cap.as_str().to_string();
            if seen.insert(flag.clone()) {
                out.push(flag);
            }
        }
        out
    }

    fn extract_flag_docs_from_help(help_text: &str) -> Vec<AgentFlagDoc> {
        static LONG_FLAG_RE: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"--[a-zA-Z0-9][a-zA-Z0-9\-]*").expect("valid long flag regex")
        });
        let mut docs = Vec::new();
        for raw_line in help_text.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let matches = LONG_FLAG_RE.find_iter(line).collect::<Vec<_>>();
            if matches.is_empty() {
                continue;
            }
            let first_end = matches[0].end();
            let desc_raw = line
                .get(first_end..)
                .unwrap_or("")
                .trim()
                .trim_start_matches([':', ';', ',', '|', '-', ' ']);
            let description = if desc_raw.is_empty() {
                None
            } else {
                Some(desc_raw.to_string())
            };
            for m in matches {
                docs.push(AgentFlagDoc {
                    flag: m.as_str().to_string(),
                    description: description.clone(),
                });
            }
        }
        Self::normalize_agent_flag_docs(docs)
    }

    async fn discover_agent_flags(agent: &str) -> anyhow::Result<Vec<AgentFlagDoc>> {
        let run_help_cmd = |help_arg: &str| {
            let mut cmd = Command::new(agent);
            cmd.arg(help_arg);
            cmd.stdin(Stdio::null());
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());
            cmd.kill_on_drop(true);
            cmd.env_remove("CLAUDECODE");
            cmd.env_remove("CLAUDE_CODE");
            cmd
        };

        let output =
            match tokio::time::timeout(Duration::from_secs(10), run_help_cmd("--help").output())
                .await
            {
                Ok(Ok(v)) => v,
                Ok(Err(err)) => {
                    if err.kind() == std::io::ErrorKind::NotFound {
                        anyhow::bail!(
                            "`{}` is not installed or not in PATH on this machine.",
                            agent
                        );
                    }
                    match tokio::time::timeout(Duration::from_secs(10), run_help_cmd("-h").output())
                        .await
                    {
                        Ok(Ok(v)) => v,
                        Ok(Err(second_err)) => {
                            anyhow::bail!("Failed to run `{}` help: {}", agent, second_err)
                        }
                        Err(_) => {
                            anyhow::bail!("`{} -h` timed out while fetching help output.", agent)
                        }
                    }
                }
                Err(_) => anyhow::bail!("`{} --help` timed out while fetching help output.", agent),
            };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let combined = if stdout.is_empty() {
            stderr.clone()
        } else if stderr.is_empty() {
            stdout.clone()
        } else {
            format!("{}\n{}", stdout, stderr)
        };

        let flags = Self::extract_long_flags_from_help(&combined);
        if flags.is_empty() {
            anyhow::bail!(
                "No long-form flags were detected in `{}` help output. Try running `{}` manually.",
                agent,
                agent
            );
        }
        let docs = Self::extract_flag_docs_from_help(&combined);
        if docs.is_empty() {
            let fallback = Self::normalize_discovered_agent_flags(flags)
                .into_iter()
                .map(|flag| AgentFlagDoc {
                    flag,
                    description: None,
                })
                .collect::<Vec<_>>();
            return Ok(fallback);
        }
        Ok(docs)
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

        let Some(agent) = Self::normalize_terminal_agent_name(&agent_raw) else {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    "Unknown agent. Use codex/claude/gemini/opencode.",
                )
                .await;
            return;
        };

        if !refresh {
            if let Some(cached) = self.load_agent_flags_cache(user_id, &agent).await {
                let age = chrono::Utc::now().timestamp() - cached.updated_at_unix;
                if (0..=AGENT_FLAGS_CACHE_TTL_SECS).contains(&age) {
                    let pages = Self::format_agent_flag_docs(&agent, &cached.docs, true);
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

        match Self::discover_agent_flags(&agent).await {
            Ok(docs) => {
                let _ = self.save_agent_flags_cache(user_id, &agent, &docs).await;
                let pages = Self::format_agent_flag_docs(&agent, &docs, false);
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
            let Some(agent) = Self::normalize_terminal_agent_name(agent_raw) else {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Unknown agent. Use codex/claude/gemini/opencode.",
                    )
                    .await;
                return;
            };
            let cleaned =
                Self::normalize_terminal_agent_args(args.into_iter().skip(1).collect::<Vec<_>>());
            if cleaned.is_empty() {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "No flags provided. Example: /agent defaults set codex --chrome",
                    )
                    .await;
                return;
            }
            let mut defaults = self
                .load_terminal_agent_defaults(msg.chat.id.0, user_id)
                .await;
            defaults.insert(agent.clone(), cleaned.clone());
            match self
                .save_terminal_agent_defaults(msg.chat.id.0, user_id, &defaults)
                .await
            {
                Ok(_) => {
                    let _ = bot
                        .send_message(
                            msg.chat.id,
                            format!(
                                "Saved default flags for {}:\n`{}`",
                                agent,
                                cleaned.join(" ")
                            ),
                        )
                        .await;
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
            let mut defaults = self
                .load_terminal_agent_defaults(msg.chat.id.0, user_id)
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
                if self
                    .save_terminal_agent_defaults(msg.chat.id.0, user_id, &defaults)
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

            let Some(agent) = Self::normalize_terminal_agent_name(&target) else {
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
            if self
                .save_terminal_agent_defaults(msg.chat.id.0, user_id, &defaults)
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

        let defaults = self
            .load_terminal_agent_defaults(msg.chat.id.0, user_id)
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

    fn is_terminal_lite_interactive_agent_command(text: &str) -> Option<String> {
        let parts = shell_words::split(text).unwrap_or_else(|_| {
            text.split_whitespace()
                .map(std::string::ToString::to_string)
                .collect()
        });
        parts
            .first()
            .and_then(|v| Self::normalize_terminal_agent_name(v))
    }

    fn default_terminal_lite_shell(&self) -> String {
        std::env::var("SHELL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "/bin/bash".to_string())
    }

    fn is_shell_env_assignment(token: &str) -> bool {
        let Some((name, _)) = token.split_once('=') else {
            return false;
        };
        if name.is_empty() {
            return false;
        }
        for (idx, ch) in name.chars().enumerate() {
            let is_valid = if idx == 0 {
                ch.is_ascii_alphabetic() || ch == '_'
            } else {
                ch.is_ascii_alphanumeric() || ch == '_'
            };
            if !is_valid {
                return false;
            }
        }
        true
    }

    fn extract_terminal_lite_command_name(text: &str) -> Option<String> {
        let parts = shell_words::split(text).unwrap_or_else(|_| {
            text.split_whitespace()
                .map(std::string::ToString::to_string)
                .collect()
        });
        for token in parts {
            let trimmed = token.trim();
            if trimmed.is_empty() {
                continue;
            }
            if Self::is_shell_env_assignment(trimmed) {
                continue;
            }
            return Some(trimmed.to_string());
        }
        None
    }

    fn contains_shell_control_operators(text: &str) -> bool {
        let mut chars = text.chars().peekable();
        let mut in_single = false;
        let mut in_double = false;
        let mut escaped = false;

        while let Some(ch) = chars.next() {
            if escaped {
                escaped = false;
                continue;
            }

            if ch == '\\' && !in_single {
                escaped = true;
                continue;
            }

            if ch == '\'' && !in_double {
                in_single = !in_single;
                continue;
            }

            if ch == '"' && !in_single {
                in_double = !in_double;
                continue;
            }

            // Command substitution works inside double quotes, so block it
            // unless we're inside single quotes.
            if ch == '$' && !in_single && matches!(chars.peek(), Some('(')) {
                return true;
            }
            if ch == '`' && !in_single {
                return true;
            }

            if in_single || in_double {
                continue;
            }

            if matches!(ch, ';' | '|' | '&' | '>' | '<' | '\n' | '\r') {
                return true;
            }
        }

        false
    }

    fn validate_terminal_lite_command(&self, text: &str) -> Result<(), String> {
        if self.terminal_allowed_prefixes.contains("*") {
            return Ok(());
        }
        if self.terminal_allowed_prefixes.is_empty() {
            return Err(
                "Terminal lite is disabled because `[terminal].allowed_prefixes` is empty."
                    .to_string(),
            );
        }

        let raw = Self::extract_terminal_lite_command_name(text)
            .ok_or_else(|| "Could not determine command name.".to_string())?;
        let command_name = Path::new(&raw)
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or(raw.as_str())
            .trim()
            .to_ascii_lowercase();

        if command_name.is_empty() {
            return Err("Could not determine command name.".to_string());
        }
        if Self::contains_shell_control_operators(text) {
            return Err(
                "Shell operators are not allowed in `/terminal lite` commands (use `/terminal` full mode)."
                    .to_string(),
            );
        }
        if self.terminal_allowed_prefixes.contains(&command_name) {
            return Ok(());
        }

        let mut allowed = self
            .terminal_allowed_prefixes
            .iter()
            .filter(|value| value.as_str() != "*")
            .cloned()
            .collect::<Vec<_>>();
        allowed.sort();
        Err(format!(
            "Command `{}` is not allowed in `/terminal lite`.\nAllowed commands: {}",
            command_name,
            allowed.join(", ")
        ))
    }

    fn resolve_terminal_lite_cwd(raw: Option<&str>) -> anyhow::Result<PathBuf> {
        let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::resolve_terminal_lite_cwd_from_base(&base, raw.unwrap_or("").trim())
    }

    fn resolve_terminal_lite_cwd_from_base(base: &Path, raw: &str) -> anyhow::Result<PathBuf> {
        let resolved = if raw.is_empty() || raw == "~" {
            dirs::home_dir().unwrap_or_else(|| base.to_path_buf())
        } else if let Some(rest) = raw.strip_prefix("~/") {
            dirs::home_dir()
                .map(|home| home.join(rest))
                .unwrap_or_else(|| base.join(rest))
        } else {
            let p = PathBuf::from(raw);
            if p.is_absolute() {
                p
            } else {
                base.join(p)
            }
        };
        let canonical = resolved
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("invalid working dir '{}': {}", resolved.display(), e))?;
        if !canonical.is_dir() {
            anyhow::bail!("'{}' is not a directory", canonical.display());
        }
        Ok(canonical)
    }

    async fn handle_terminal_lite_command(
        &self,
        mut args: Vec<String>,
        msg: &teloxide::types::Message,
        bot: &Bot,
        user_id: u64,
    ) {
        let subcommand = args
            .first()
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_else(|| "start".to_string());

        if subcommand == "help" {
            let _ = bot
                .send_message(msg.chat.id, Self::terminal_lite_help_text())
                .await;
            return;
        }

        if subcommand == "status" {
            let sessions = self.terminal_lite_sessions.lock().await;
            if let Some(session) = sessions.get(&msg.chat.id.0) {
                let elapsed = session.started_at.elapsed().as_secs();
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!(
                            "Terminal lite is active.\nOwner: {}\nWorking dir: {}\nShell: {}\nPreferred agent: {}\nBusy: {}\nUptime: {}s",
                            session.owner_user_id,
                            session.cwd.display(),
                            session.shell,
                            session.preferred_agent.as_deref().unwrap_or("none"),
                            if session.busy { "yes" } else { "no" },
                            elapsed
                        ),
                    )
                    .await;
            } else {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Terminal lite is not active. Start with `/terminal lite start`.",
                    )
                    .await;
            }
            return;
        }

        if subcommand == "stop" {
            let mut sessions = self.terminal_lite_sessions.lock().await;
            if sessions.remove(&msg.chat.id.0).is_some() {
                let _ = bot
                    .send_message(msg.chat.id, "Terminal lite stopped for this chat.")
                    .await;
            } else {
                let _ = bot
                    .send_message(msg.chat.id, "Terminal lite is not active.")
                    .await;
            }
            return;
        }

        if subcommand == "start" {
            args.remove(0);
        }

        let preferred_agent = args
            .first()
            .and_then(|value| Self::normalize_terminal_agent_name(value));
        if preferred_agent.is_some() && !args.is_empty() {
            args.remove(0);
        }

        let cwd_arg = if args.is_empty() {
            None
        } else {
            Some(args.join(" "))
        };
        let cwd = match Self::resolve_terminal_lite_cwd(cwd_arg.as_deref()) {
            Ok(v) => v,
            Err(err) => {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!("Failed to start terminal lite: {}", err),
                    )
                    .await;
                return;
            }
        };

        let shell = self.default_terminal_lite_shell();
        let session = TerminalLiteSession {
            owner_user_id: user_id,
            cwd: cwd.clone(),
            shell: shell.clone(),
            preferred_agent: preferred_agent.clone(),
            started_at: Instant::now(),
            busy: false,
        };

        {
            let mut sessions = self.terminal_lite_sessions.lock().await;
            sessions.insert(msg.chat.id.0, session);
        }

        let _ = bot
            .send_message(
                msg.chat.id,
                format!(
                    "Terminal lite started.\nWorking dir: {}\nShell: {}\nPreferred agent: {}\n\nSend commands as chat messages.\nUse `/terminal lite stop` to stop.",
                    cwd.display(),
                    shell,
                    preferred_agent.as_deref().unwrap_or("none")
                ),
            )
            .await;
    }

    async fn handle_terminal_lite_input(
        &self,
        chat_id: i64,
        user_id: u64,
        user_role: UserRole,
        text: &str,
    ) -> Option<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        let mut sessions = self.terminal_lite_sessions.lock().await;
        let session = sessions.get_mut(&chat_id)?;

        if user_role != UserRole::Owner || session.owner_user_id != user_id {
            return Some("Only the owner can use terminal lite in this chat.".to_string());
        }

        if session.busy {
            return Some(
                "Terminal lite is busy with the previous command. Wait or run `/terminal lite stop`."
                    .to_string(),
            );
        }

        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            sessions.remove(&chat_id);
            return Some("Terminal lite stopped.".to_string());
        }

        if let Some(reply) = Self::terminal_lite_try_handle_cd(session, trimmed) {
            return Some(reply);
        }

        if let Some(agent) = Self::is_terminal_lite_interactive_agent_command(trimmed) {
            return Some(format!(
                    "`{}` is an interactive TUI and is not supported in `/terminal lite`.\nUse full mode instead: `/terminal {} {}`",
                agent,
                agent,
                session.cwd.display()
            ));
        }

        if let Err(err) = self.validate_terminal_lite_command(trimmed) {
            return Some(err);
        }

        let snapshot = session.clone();
        session.busy = true;
        drop(sessions);

        let reply = Self::run_terminal_lite_command(&snapshot, trimmed).await;

        let mut sessions = self.terminal_lite_sessions.lock().await;
        if let Some(active) = sessions.get_mut(&chat_id) {
            active.busy = false;
        }
        Some(reply)
    }

    fn terminal_lite_try_handle_cd(
        session: &mut TerminalLiteSession,
        text: &str,
    ) -> Option<String> {
        let parts = shell_words::split(text).unwrap_or_else(|_| {
            text.split_whitespace()
                .map(std::string::ToString::to_string)
                .collect()
        });
        if parts.is_empty() || parts[0] != "cd" {
            return None;
        }
        let target = if parts.len() <= 1 {
            "~".to_string()
        } else {
            parts[1].clone()
        };
        match Self::resolve_terminal_lite_cwd_from_base(&session.cwd, target.trim()) {
            Ok(path) => {
                session.cwd = path.clone();
                Some(format!("cwd -> {}", path.display()))
            }
            Err(err) => Some(format!("cd: {}", err)),
        }
    }

    async fn run_terminal_lite_command(
        session: &TerminalLiteSession,
        command_text: &str,
    ) -> String {
        let mut cmd = Command::new(&session.shell);
        if cfg!(windows) {
            cmd.arg("-NoLogo").arg("-Command").arg(command_text);
        } else {
            cmd.arg("-lc").arg(command_text);
        }
        cmd.current_dir(&session.cwd);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.kill_on_drop(true);
        cmd.env_remove("CLAUDECODE");
        cmd.env_remove("CLAUDE_CODE");
        if !cfg!(windows) {
            cmd.env("TERM", "xterm-256color");
        }

        let output = match tokio::time::timeout(
            Duration::from_secs(TERMINAL_LITE_TIMEOUT_SECS),
            cmd.output(),
        )
        .await
        {
            Ok(Ok(v)) => v,
            Ok(Err(err)) => {
                return format!("Failed to run command: {}", err);
            }
            Err(_) => {
                return format!(
                    "⏱️ Command timed out after {}s: {}",
                    TERMINAL_LITE_TIMEOUT_SECS, command_text
                );
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let mut body = String::new();
        body.push_str("$ ");
        body.push_str(command_text);
        body.push_str("\n\n");

        if stdout.trim().is_empty() && stderr.trim().is_empty() {
            body.push_str("(no output)\n");
        } else {
            if !stdout.is_empty() {
                body.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !stdout.ends_with('\n') && !stdout.is_empty() {
                    body.push('\n');
                }
                body.push_str(&stderr);
            }
        }

        if body.chars().count() > TERMINAL_LITE_MAX_OUTPUT_CHARS {
            let clipped: String = body.chars().take(TERMINAL_LITE_MAX_OUTPUT_CHARS).collect();
            body = format!(
                "{}\n\n[output truncated to {} chars]",
                clipped, TERMINAL_LITE_MAX_OUTPUT_CHARS
            );
        }

        let exit_code = output.status.code().unwrap_or(-1);
        body.push_str(&format!("\n[exit {}]", exit_code));
        body
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
        agent_args = Self::normalize_terminal_agent_args(agent_args);
        let skip_saved_defaults = Self::strip_no_default_flag(&mut agent_args);

        if invoked_cmd == "/agent" && agent_args.is_empty() && !skip_saved_defaults {
            if let Some(agent_name) = agent.as_deref() {
                let defaults = self
                    .load_terminal_agent_defaults(msg.chat.id.0, user_id)
                    .await;
                if let Some(saved) = defaults.get(agent_name) {
                    agent_args = Self::normalize_terminal_agent_args(saved.clone());
                    used_saved_args = !agent_args.is_empty();
                }
            }
        } else if invoked_cmd == "/agent" && !agent_args.is_empty() {
            if let Some(agent_name) = agent.as_deref() {
                let mut defaults = self
                    .load_terminal_agent_defaults(msg.chat.id.0, user_id)
                    .await;
                let changed = defaults.get(agent_name) != Some(&agent_args);
                if changed {
                    defaults.insert(agent_name.to_string(), agent_args.clone());
                    if let Err(err) = self
                        .save_terminal_agent_defaults(msg.chat.id.0, user_id, &defaults)
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
    /// Downloads the file, saves to inbox, and returns a context string for the agent.
    async fn handle_file_message(
        &self,
        msg: &teloxide::types::Message,
        bot: &Bot,
    ) -> anyhow::Result<String> {
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
        let response = reqwest::get(&download_url).await?;
        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download file from Telegram: HTTP {}",
                response.status()
            );
        }
        let bytes = response.bytes().await?;

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

        // Build context string for the agent
        let size_display = if bytes.len() > 1_048_576 {
            format!("{:.1} MB", bytes.len() as f64 / 1_048_576.0)
        } else {
            format!("{:.0} KB", bytes.len() as f64 / 1024.0)
        };

        let caption = msg.caption().unwrap_or("");
        let mut context = format!(
            "[File received: {} ({}, {})\nSaved to: {}]",
            sanitized,
            size_display,
            mime_type,
            dest_path.display()
        );
        if !caption.is_empty() {
            context.push('\n');
            context.push_str(caption);
        }

        Ok(context)
    }

    async fn handle_cost_command(&self) -> String {
        use std::collections::HashMap as StdHashMap;

        let now = Utc::now();
        let since_24h = (now - chrono::Duration::hours(24))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let since_7d = (now - chrono::Duration::days(7))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();

        let records_24h = match self.state.get_token_usage_since(&since_24h).await {
            Ok(r) => r,
            Err(e) => return format!("Failed to query token usage: {}", e),
        };
        let records_7d = match self.state.get_token_usage_since(&since_7d).await {
            Ok(r) => r,
            Err(e) => return format!("Failed to query token usage: {}", e),
        };

        let (input_24h, output_24h) = records_24h.iter().fold((0i64, 0i64), |(i, o), r| {
            (i + r.input_tokens, o + r.output_tokens)
        });
        let (input_7d, output_7d) = records_7d.iter().fold((0i64, 0i64), |(i, o), r| {
            (i + r.input_tokens, o + r.output_tokens)
        });

        // Top models (by total tokens in 7d)
        let mut model_totals: StdHashMap<&str, i64> = StdHashMap::new();
        for r in &records_7d {
            *model_totals.entry(&r.model).or_insert(0) += r.input_tokens + r.output_tokens;
        }
        let mut models_sorted: Vec<(&&str, &i64)> = model_totals.iter().collect();
        models_sorted.sort_by(|a, b| b.1.cmp(a.1));

        let mut reply = format!(
            "Token usage (last 24h):\n  Input:  {} tokens\n  Output: {} tokens\n\n\
             Token usage (last 7d):\n  Input:  {} tokens\n  Output: {} tokens",
            format_number(input_24h),
            format_number(output_24h),
            format_number(input_7d),
            format_number(output_7d),
        );

        if !models_sorted.is_empty() {
            reply.push_str("\n\nTop models (7d):");
            for (model, total) in models_sorted.iter().take(5) {
                reply.push_str(&format!("\n  {}: {} tokens", model, format_number(**total)));
            }
        }

        reply
    }

    /// Auto-send file attachments referenced as absolute paths in a reply.
    async fn send_referenced_files_from_reply(
        bot: &Bot,
        chat_id: ChatId,
        reply: &str,
        files_enabled: bool,
    ) {
        let candidate_paths = extract_candidate_file_paths(reply);
        if candidate_paths.is_empty() {
            return;
        }

        if !files_enabled {
            let _ = bot
                .send_message(
                    chat_id,
                    "I found file path(s) in my response, but file attachments are disabled in config.",
                )
                .await;
            return;
        }

        const MAX_FILES_PER_REPLY: usize = 3;
        let sendable_paths: HashSet<String> = crate::agent::extract_file_paths_from_text(reply)
            .into_iter()
            .collect();
        let mut seen = HashSet::new();
        let mut sent = 0usize;
        let mut skipped = 0usize;

        for path in candidate_paths {
            if sent >= MAX_FILES_PER_REPLY {
                skipped += 1;
                continue;
            }
            if !seen.insert(path.clone()) {
                continue;
            }

            if !sendable_paths.contains(&path) {
                debug!(
                    file = %path,
                    "Skipping non-sendable auto-detected file path from reply"
                );
                continue;
            }

            let file_name = std::path::Path::new(&path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "file".to_string());

            let doc = InputFile::file(std::path::PathBuf::from(&path)).file_name(file_name.clone());
            match bot
                .send_document(chat_id, doc)
                .caption(format!("📎 {}", file_name))
                .await
            {
                Ok(_) => {
                    sent += 1;
                }
                Err(e) => {
                    warn!(file = %path, error = %e, "Failed to send referenced file");
                    let _ = bot
                        .send_message(
                            chat_id,
                            format!("I found `{}` but couldn't upload it: {}.", file_name, e),
                        )
                        .await;
                }
            }
        }

        if skipped > 0 {
            let _ = bot
                .send_message(
                    chat_id,
                    format!(
                        "I found additional files but only sent the first {} attachments.",
                        MAX_FILES_PER_REPLY
                    ),
                )
                .await;
        }

        if sent > 0 {
            let _ = bot
                .send_message(
                    chat_id,
                    format!(
                        "Sent {} file attachment{}.",
                        sent,
                        if sent == 1 { "" } else { "s" }
                    ),
                )
                .await;
        }
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
        let test_bot = Bot::new(token);
        let me = match test_bot.get_me().await {
            Ok(me) => me,
            Err(e) => {
                return format!(
                    "Invalid token: {}\n\nMake sure you copied the full token from @BotFather.",
                    e
                );
            }
        };

        let bot_username = me.username.clone().unwrap_or_else(|| "unknown".to_string());

        // Check if this bot is already connected
        match self.state.get_dynamic_bots().await {
            Ok(bots) => {
                for existing in &bots {
                    if existing.channel_type == "telegram" && existing.bot_token == token {
                        return format!(
                            "Bot @{} is already connected.\n\nUse /bots to see all connected bots.",
                            bot_username
                        );
                    }
                }
            }
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
            extra_config: "{}".to_string(),
            created_at: String::new(), // Will be set by database
        };

        let db_id = match self.state.add_dynamic_bot(&new_bot).await {
            Ok(id) => id,
            Err(e) => {
                warn!("Failed to save bot: {}", e);
                return format!("Failed to save bot configuration: {}", e);
            }
        };

        info!(
            bot = %bot_username,
            id = db_id,
            added_by = user_id,
            "New Telegram bot connected"
        );

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
                    self.terminal_allowed_prefixes.iter().cloned().collect(),
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
                    Use /bots to see all connected bots.",
                    bot_username
                );
            }
        }

        // Fallback: hub not available, require restart
        format!(
            "✓ Bot @{} connected!\n\n\
            Run /restart to activate the new bot.\n\
            Use /bots to see all connected bots.",
            bot_username
        )
    }

    /// Connect a new Discord bot by validating its token.
    #[cfg(feature = "discord")]
    async fn connect_discord_bot(&self, token: &str, user_id: u64) -> String {
        // Validate the token by making a test API call
        let client = reqwest::Client::new();
        let response = client
            .get("https://discord.com/api/v10/users/@me")
            .header("Authorization", format!("Bot {}", token))
            .send()
            .await;

        let bot_name = match response {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<serde_json::Value>().await {
                    Ok(json) => json["username"].as_str().unwrap_or("unknown").to_string(),
                    Err(_) => "unknown".to_string(),
                }
            }
            Ok(resp) => {
                return format!(
                    "Invalid Discord token (HTTP {}). Make sure you copied the bot token from Discord Developer Portal.",
                    resp.status()
                );
            }
            Err(e) => {
                return format!("Failed to validate token: {}", e);
            }
        };

        // Check if already connected
        match self.state.get_dynamic_bots().await {
            Ok(bots) => {
                for existing in &bots {
                    if existing.channel_type == "discord" && existing.bot_token == token {
                        return format!(
                            "Bot {} is already connected.\n\nUse /bots to see all connected bots.",
                            bot_name
                        );
                    }
                }
            }
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

        let db_id = match self.state.add_dynamic_bot(&new_bot).await {
            Ok(id) => id,
            Err(e) => {
                return format!("Failed to save bot configuration: {}", e);
            }
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
    async fn connect_discord_bot(&self, _token: &str, _user_id: u64) -> String {
        "Discord support is not enabled in this build.\n\n\
        Rebuild with `cargo build --features discord` to enable Discord bots."
            .to_string()
    }

    /// Connect a new Slack bot.
    /// Connect a new Slack bot.
    #[cfg(feature = "slack")]
    async fn connect_slack_bot(&self, bot_token: &str, app_token: &str, user_id: u64) -> String {
        // Validate the bot token by calling auth.test
        let client = reqwest::Client::new();
        let response = client
            .get("https://slack.com/api/auth.test")
            .header("Authorization", format!("Bearer {}", bot_token))
            .send()
            .await;

        let (bot_name, team_name) = match response {
            Ok(resp) => match resp.json::<serde_json::Value>().await {
                Ok(json) => {
                    if json["ok"].as_bool() != Some(true) {
                        return format!(
                            "Invalid Slack token: {}\n\nMake sure you have the correct bot token.",
                            json["error"].as_str().unwrap_or("unknown error")
                        );
                    }
                    (
                        json["user"].as_str().unwrap_or("unknown").to_string(),
                        json["team"].as_str().unwrap_or("unknown").to_string(),
                    )
                }
                Err(e) => {
                    return format!("Failed to parse Slack response: {}", e);
                }
            },
            Err(e) => {
                return format!("Failed to validate Slack token: {}", e);
            }
        };

        // Check if already connected
        match self.state.get_dynamic_bots().await {
            Ok(bots) => {
                for existing in &bots {
                    if existing.channel_type == "slack" && existing.bot_token == bot_token {
                        return format!(
                            "Slack bot {} ({}) is already connected.\n\nUse /bots to see all connected bots.",
                            bot_name, team_name
                        );
                    }
                }
            }
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

        let db_id = match self.state.add_dynamic_bot(&new_bot).await {
            Ok(id) => id,
            Err(e) => {
                return format!("Failed to save bot configuration: {}", e);
            }
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
    async fn connect_slack_bot(&self, _bot_token: &str, _app_token: &str, _user_id: u64) -> String {
        "Slack support is not enabled in this build.\n\n\
        Rebuild with `cargo build --features slack` to enable Slack bots."
            .to_string()
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
        match self.state.get_dynamic_bots().await {
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
                    "✅ <b>Resume Code Accepted</b>\n\nSession: <code>{}</code>\nTap Open in Mini App to continue on your phone.",
                    html_escape(&relay_session_id)
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

    async fn handle_message(&self, msg: teloxide::types::Message, bot: Bot) {
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

        let text = if let Some(web_app_data) = msg.web_app_data() {
            match Self::parse_web_app_action(&web_app_data.data) {
                Some(TelegramWebAppAction::ContinueOnComputer { relay_session_id }) => {
                    self.handle_continue_on_computer_action(&bot, &msg, user_id, relay_session_id)
                        .await;
                    return;
                }
                Some(TelegramWebAppAction::AgentMessage(text)) => text,
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
                    TelegramWebAppAction::AgentMessage(text) => text,
                }
            } else {
                t.to_string()
            }
        } else if self.files_enabled {
            match self.handle_file_message(&msg, &bot).await {
                Ok(file_text) => file_text,
                Err(e) => {
                    let _ = bot
                        .send_message(msg.chat.id, format!("File error: {}", e))
                        .await;
                    return;
                }
            }
        } else {
            let _ = bot
                .send_message(msg.chat.id, "I can only process text messages.")
                .await;
            return;
        };

        // Handle slash commands
        if text.starts_with('/') {
            self.handle_command(&text, &msg, &bot).await;
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
                channel_id: Some(format!("telegram:{}", msg.chat.id.0)),
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
            let queue_result = self.task_registry.queue_message(&session_id, &text).await;
            match queue_result {
                Some(queue_pos) => {
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
                None => {
                    // Duplicate message detected — silently ignore
                    debug!(session_id, "Dropped duplicate queued message");
                }
            }
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

        // Status update channel — agent emits updates, we display them rate-limited.
        let (status_tx, mut status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(16);
        let status_bot = bot.clone();
        let status_chat_id = msg.chat.id;
        let is_dm = channel_ctx.visibility == ChannelVisibility::Private;
        let status_task = tokio::spawn(async move {
            let mut last_sent = tokio::time::Instant::now() - Duration::from_secs(10);
            let min_interval = Duration::from_secs(3);
            let mut sent_thinking = false;
            while let Some(update) = status_rx.recv().await {
                // In non-DM channels: only send one "Thinking..." then suppress
                if !is_dm {
                    if matches!(&update, StatusUpdate::BudgetExtended { .. }) {
                        // Fall through — cost notifications must reach the user
                    } else {
                        if !sent_thinking
                            && matches!(
                                update,
                                StatusUpdate::Thinking(_) | StatusUpdate::ToolStart { .. }
                            )
                        {
                            let _ = status_bot.send_message(status_chat_id, "Thinking...").await;
                            let _ = status_bot
                                .send_chat_action(status_chat_id, ChatAction::Typing)
                                .await;
                            sent_thinking = true;
                            last_sent = tokio::time::Instant::now();
                        }
                        continue;
                    }
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
                    StatusUpdate::Thinking(_) => "Thinking...".to_string(),
                    StatusUpdate::ToolStart { name, summary } => {
                        if summary.is_empty() {
                            format!("Using {}...", name)
                        } else {
                            format!("Using {}: {}...", name, summary)
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
                };
                let _ = status_bot.send_message(status_chat_id, text).await;
                // Re-send typing indicator immediately after each status message.
                // Telegram clears the typing indicator when a message is sent, so
                // without this there's a visible gap until the typing loop's next
                // 4-second tick.
                let _ = status_bot
                    .send_chat_action(status_chat_id, ChatAction::Typing)
                    .await;
                last_sent = tokio::time::Instant::now();
            }
        });

        // Register this task for tracking and cancellation.
        let description: String = text.chars().take(80).collect();
        let (task_id, cancel_token) = self.task_registry.register(&session_id, &description).await;
        // Associate the typing indicator with this task so cancel_running_for_session
        // also stops the typing loop (fixes typing persisting after cancel/panic).
        self.task_registry
            .set_typing_cancel(task_id, typing_cancel.clone())
            .await;
        let registry = Arc::clone(&self.task_registry);
        let files_enabled = self.files_enabled;

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
            let mut current_task_id = task_id;
            let mut current_cancel_token = cancel_token;
            let mut current_status_tx = status_tx;
            let mut current_typing_cancel = typing_cancel;
            let mut current_status_task = status_task;
            let mut current_heartbeat = heartbeat;

            loop {
                let result = tokio::select! {
                    r = agent.handle_message(&session_id, &current_text, Some(current_status_tx), user_role, channel_ctx.clone(), Some(current_heartbeat.clone())) => r,
                    _ = current_cancel_token.cancelled() => Err(anyhow::anyhow!("Task cancelled")),
                    stale_mins = super::wait_for_stale_heartbeat(current_heartbeat.clone(), stale_threshold_secs, 4), if stale_threshold_secs > 0 => {
                        Err(anyhow::anyhow!(
                            "Task auto-cancelled due to inactivity ({} minute{} without progress).",
                            stale_mins,
                            if stale_mins == 1 { "" } else { "s" }
                        ))
                    },
                };
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
                            if let Err(e) =
                                send_full_or_expandable_reply(&bot, chat_id, &reply).await
                            {
                                warn!("Failed to send Telegram message: {}", e);
                            }
                            TelegramChannel::send_referenced_files_from_reply(
                                &bot,
                                chat_id,
                                &reply,
                                files_enabled,
                            )
                            .await;
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
                    }
                }

                // Finalize the current task AFTER sending the response/error.
                // This closes the race window where incoming messages could see
                // no running task and spawn concurrent handlers.
                if let Some(ref err) = task_error {
                    registry.fail(current_task_id, err).await;
                } else {
                    registry.complete(current_task_id).await;
                }

                // Check if there are queued messages to process
                if let Some(queued) = registry.pop_queued_message(&session_id).await {
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

                    // Create new status channel and typing indicator
                    let (new_status_tx, mut new_status_rx) =
                        tokio::sync::mpsc::channel::<StatusUpdate>(16);
                    current_status_tx = new_status_tx;

                    let status_bot = bot.clone();
                    current_status_task = tokio::spawn(async move {
                        let mut last_sent = tokio::time::Instant::now() - Duration::from_secs(10);
                        let min_interval = Duration::from_secs(3);
                        let mut sent_thinking = false;
                        while let Some(update) = new_status_rx.recv().await {
                            // In non-DM channels: only send one "Thinking..." then suppress
                            // (except BudgetExtended which must always reach the user)
                            if !is_dm {
                                if matches!(&update, StatusUpdate::BudgetExtended { .. }) {
                                    // Fall through — cost notifications must reach the user
                                } else {
                                    if !sent_thinking
                                        && matches!(
                                            update,
                                            StatusUpdate::Thinking(_)
                                                | StatusUpdate::ToolStart { .. }
                                        )
                                    {
                                        let _ =
                                            status_bot.send_message(chat_id, "Thinking...").await;
                                        sent_thinking = true;
                                        last_sent = tokio::time::Instant::now();
                                    }
                                    continue;
                                }
                            }
                            let now = tokio::time::Instant::now();
                            let is_budget_ext =
                                matches!(&update, StatusUpdate::BudgetExtended { .. });
                            if !is_budget_ext && now.duration_since(last_sent) < min_interval {
                                continue;
                            }
                            let text = match &update {
                                StatusUpdate::Thinking(_) => "Thinking...".to_string(),
                                StatusUpdate::ToolStart { name, summary } => {
                                    if summary.is_empty() {
                                        format!("Using {}...", name)
                                    } else {
                                        format!("Using {}: {}...", name, summary)
                                    }
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
                                _ => continue, // Skip other status updates for queued messages
                            };
                            let _ = status_bot.send_message(chat_id, text).await;
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
                self.bot
                    .send_photo(ChatId(chat_id), photo)
                    .caption(&media.caption)
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
                req.await
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
        let use_session_button = match permission_mode {
            PermissionMode::Cautious => true,
            PermissionMode::Default => risk_level >= RiskLevel::Critical,
            PermissionMode::Yolo => false,
        };

        let keyboard = if use_session_button {
            InlineKeyboardMarkup::new(vec![vec![
                InlineKeyboardButton::callback(
                    "Allow Once",
                    format!("approve:once:{}", approval_id),
                ),
                InlineKeyboardButton::callback(
                    "Allow Session",
                    format!("approve:session:{}", approval_id),
                ),
                InlineKeyboardButton::callback("Deny", format!("approve:deny:{}", approval_id)),
            ]])
        } else {
            InlineKeyboardMarkup::new(vec![vec![
                InlineKeyboardButton::callback(
                    "Allow Once",
                    format!("approve:once:{}", approval_id),
                ),
                InlineKeyboardButton::callback(
                    "Allow Always",
                    format!("approve:always:{}", approval_id),
                ),
                InlineKeyboardButton::callback("Deny", format!("approve:deny:{}", approval_id)),
            ]])
        };

        let escaped_cmd = html_escape(command);

        // Build message with risk info
        let (risk_icon, risk_label) = match risk_level {
            RiskLevel::Safe => ("ℹ️", "New command"),
            RiskLevel::Medium => ("⚠️", "Medium risk"),
            RiskLevel::High => ("🔶", "High risk"),
            RiskLevel::Critical => ("🚨", "Critical risk"),
        };

        let mut text = format!(
            "{} <b>{}</b>\n\n<code>{}</code>",
            risk_icon, risk_label, escaped_cmd
        );

        if !warnings.is_empty() {
            text.push('\n');
            for warning in warnings {
                text.push_str(&format!("\n• {}", html_escape(warning)));
            }
        }

        // Add explanation based on which button is shown
        if use_session_button {
            text.push_str("\n\n<i>\"Allow Session\" approves this command type until restart.</i>");
        } else {
            text.push_str("\n\n<i>\"Allow Always\" permanently approves this command type.</i>");
        }

        text.push_str(&format!("\n\n<i>[{}]</i>", short_id));

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

        let keyboard = InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::callback("Confirm ✅", format!("goal:confirm:{}", approval_id)),
            InlineKeyboardButton::callback("Cancel ❌", format!("goal:cancel:{}", approval_id)),
        ]]);

        let escaped_desc = html_escape(goal_description);
        let mut text = format!(
            "📅 <b>Confirm scheduled goal</b>\n\n<code>{}</code>",
            escaped_desc
        );

        for detail in details {
            text.push_str(&format!("\n• {}", html_escape(detail)));
        }

        text.push_str(&format!("\n\n<i>[{}]</i>", short_id));

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
    match bot
        .send_message(chat_id, html)
        .parse_mode(ParseMode::Html)
        .await
    {
        Ok(_) => Ok(()),
        Err(e) => {
            warn!("HTML send failed, falling back to plain text: {}", e);
            bot.send_message(chat_id, plain).await?;
            Ok(())
        }
    }
}

async fn send_markdown_chunks_or_fallback_result(
    bot: &Bot,
    chat_id: ChatId,
    markdown: &str,
) -> anyhow::Result<()> {
    let html = markdown_to_telegram_html(markdown);
    let html_chunks = split_message(&html, 4096);
    let plain_chunks = split_message(&strip_latex(markdown), 4096);
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
    let plain = strip_latex(markdown);
    if plain.chars().count() > TELEGRAM_EXPANDABLE_TRIGGER_CHARS {
        return send_expandable_blockquote_reply(bot, chat_id, &plain).await;
    }
    send_markdown_chunks_or_fallback_result(bot, chat_id, markdown).await
}

async fn send_expandable_blockquote_reply(
    bot: &Bot,
    chat_id: ChatId,
    plain: &str,
) -> anyhow::Result<()> {
    let chunks = split_for_expandable_blockquote(plain, TELEGRAM_EXPANDABLE_MAX_ESCAPED_CHARS);
    let mut first_err: Option<anyhow::Error> = None;
    for chunk in chunks {
        let escaped = html_escape(&chunk);
        let html = format!("<blockquote expandable>{}</blockquote>", escaped);
        if let Err(e) = send_html_or_fallback(bot, chat_id, &html, &chunk).await {
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

fn split_for_expandable_blockquote(text: &str, max_escaped_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut current_escaped_len = 0usize;

    for ch in text.chars() {
        let add = match ch {
            '&' => 5,       // "&amp;"
            '<' | '>' => 4, // "&lt;" / "&gt;"
            _ => 1,
        };
        if current_escaped_len + add > max_escaped_chars && !current.is_empty() {
            out.push(current);
            current = String::new();
            current_escaped_len = 0;
        }
        current.push(ch);
        current_escaped_len += add;
    }

    if !current.is_empty() {
        out.push(current);
    }

    if out.is_empty() {
        out.push(String::new());
    }
    out
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

fn extract_candidate_file_paths(text: &str) -> Vec<String> {
    static PATH_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(/[\w./-]+\.\w{1,10})").unwrap());
    let mut out = Vec::new();
    for cap in PATH_RE.captures_iter(text) {
        let token = cap[1].trim_matches(|c: char| {
            c.is_whitespace() || matches!(c, ')' | ']' | '}' | ',' | ';' | ':' | '!' | '?')
        });
        if token.starts_with('/') && token.contains('.') {
            out.push(token.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_lite_detects_shell_control_operators() {
        assert!(TelegramChannel::contains_shell_control_operators(
            "ls && rm -rf /tmp/demo"
        ));
        assert!(TelegramChannel::contains_shell_control_operators(
            "echo $(whoami)"
        ));
        assert!(TelegramChannel::contains_shell_control_operators(
            "echo `whoami`"
        ));
    }

    #[test]
    fn terminal_lite_allows_simple_commands_without_operators() {
        assert!(!TelegramChannel::contains_shell_control_operators(
            "ls -la /tmp"
        ));
        assert!(!TelegramChannel::contains_shell_control_operators(
            "FOO=bar env"
        ));
        assert!(!TelegramChannel::contains_shell_control_operators(
            "echo ';' '|' '>'"
        ));
    }

    #[test]
    fn format_agent_flag_docs_paginates_and_keeps_footer_actions() {
        let docs = (1..=13)
            .map(|n| AgentFlagDoc {
                flag: format!("--flag-{}", n),
                description: Some(format!("Description {}", n)),
            })
            .collect::<Vec<_>>();

        let pages = TelegramChannel::format_agent_flag_docs("claude", &docs, false);
        assert_eq!(pages.len(), 2);
        assert!(pages[0].contains("Showing 1-12 of 13 (page 1/2)"));
        assert!(pages[1].contains("Showing 13-13 of 13 (page 2/2)"));
        assert!(pages[1].contains("Set defaults with `/agent defaults set <agent> [flags...]`."));
        assert!(pages[1].contains("Bypass once with `--no-default-flags`."));
        assert!(pages[1].contains("Refresh with `/agent flags <agent> refresh`."));
    }

    #[test]
    fn format_agent_flag_docs_includes_cached_badge() {
        let docs = vec![AgentFlagDoc {
            flag: "--print".to_string(),
            description: Some("Output format.".to_string()),
        }];
        let pages = TelegramChannel::format_agent_flag_docs("claude", &docs, true);
        assert_eq!(pages.len(), 1);
        assert!(pages[0].contains("Flags for `claude` (cached)"));
    }

    #[test]
    fn split_for_expandable_blockquote_keeps_short_text_in_one_chunk() {
        let chunks = split_for_expandable_blockquote("short reply", 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "short reply");
    }

    #[test]
    fn split_for_expandable_blockquote_respects_escaped_limit() {
        let text = "A&B<C>D";
        // Escaped length is 1 + 5 + 1 + 4 + 1 + 4 + 1 = 17
        let chunks = split_for_expandable_blockquote(text, 10);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "A&B");
        assert_eq!(chunks[1], "<C>D");
    }

    #[test]
    fn split_for_expandable_blockquote_handles_empty_text() {
        let chunks = split_for_expandable_blockquote("", 10);
        assert_eq!(chunks, vec![String::new()]);
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
    fn extract_candidate_file_paths_handles_trailing_punctuation() {
        let text = "I found it at /tmp/test-docs/sample-resume.pdf.";
        let paths = extract_candidate_file_paths(text);
        assert_eq!(paths, vec!["/tmp/test-docs/sample-resume.pdf"]);
    }

    #[test]
    fn extract_candidate_file_paths_extracts_multiple_paths() {
        let text = "Primary: `/tmp/aidaemon/report.md` and backup: /tmp/aidaemon/report.csv, done.";
        let paths = extract_candidate_file_paths(text);
        assert_eq!(
            paths,
            vec!["/tmp/aidaemon/report.md", "/tmp/aidaemon/report.csv"]
        );
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
