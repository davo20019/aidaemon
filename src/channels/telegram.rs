use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock as StdRwLock, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use chrono::Utc;
use once_cell::sync::Lazy;
use regex::Regex;
use teloxide::prelude::*;
use teloxide::types::{
    ChatAction, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, ParseMode,
};
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
    /// Daemon start time used for post-restart UX guardrails.
    started_at: Instant,
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
    ) -> Self {
        let bot = Bot::new(bot_token);
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
        let cmd = parts[0];
        let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

        let reply = match cmd {
            "/model" => {
                if arg.is_empty() {
                    let current = self.agent.current_model().await;
                    format!("Current model: {}\n\nUsage: /model <model-name>\nExample: /model gemini-3-pro-preview", current)
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
            "/reload" => {
                match AppConfig::load(&self.config_path) {
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
                                format!("Config reload failed: {}\n\nBackup restore also failed. Manual intervention needed.", e)
                            }
                        } else {
                            format!("Config reload failed: {}\n\nNo backup available.", e)
                        }
                    }
                }
            }
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
            "/help" | "/start" => build_help_text(true, true, "/"),
            _ => format!(
                "Unknown command: {}\nType /help for available commands.",
                cmd
            ),
        };

        for chunk in split_message(&reply, 4096) {
            let _ = bot.send_message(msg.chat.id, chunk).await;
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
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Hey! You're now set as the owner. Ask me anything, give me tasks, or just chat.",
                    )
                    .await;
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

        let text = if let Some(t) = msg.text() {
            t.to_string()
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

        // Use chat ID as session ID, prefixed with bot name if multi-bot
        let session_id = self.session_id(msg.chat.id.0).await;

        // Register this session with the channel hub so outbound messages
        // (approvals, media, notifications) route back to this Telegram bot.
        {
            let channel_name = self.channel_name().await;
            let mut map = self.session_map.write().await;
            map.insert(session_id.clone(), channel_name.clone());
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
                            let html = markdown_to_telegram_html(&reply);
                            // Split long messages (Telegram limit is 4096 chars)
                            let html_chunks = split_message(&html, 4096);
                            let plain_chunks = split_message(&strip_latex(&reply), 4096);
                            for (i, html_chunk) in html_chunks.iter().enumerate() {
                                let plain_chunk = plain_chunks
                                    .get(i)
                                    .map(|s| s.as_str())
                                    .unwrap_or(html_chunk.as_str());
                                if let Err(e) =
                                    send_html_or_fallback(&bot, chat_id, html_chunk, plain_chunk)
                                        .await
                                {
                                    warn!("Failed to send Telegram message: {}", e);
                                }
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
        let html = markdown_to_telegram_html(text);
        let plain = strip_latex(text);
        let mut first_err: Option<anyhow::Error> = None;
        for chunk in split_message(&html, 4096) {
            if let Err(e) = send_html_or_fallback(&self.bot, ChatId(chat_id), &chunk, &plain).await
            {
                warn!("Failed to send message: {}", e);
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

        // Wait with 5-minute timeout
        match tokio::time::timeout(Duration::from_secs(300), response_rx).await {
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
                warn!(approval_id = %short_id, "Goal confirmation timed out");
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
