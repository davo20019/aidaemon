use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock as StdRwLock, Weak};
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use teloxide::prelude::*;
use teloxide::types::{ChatAction, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, ParseMode};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::agent::Agent;
use crate::channels::{ChannelHub, SessionMap};
#[cfg(feature = "discord")]
use crate::channels::{DiscordChannel, spawn_discord_channel};
#[cfg(feature = "slack")]
use crate::channels::{SlackChannel, spawn_slack_channel};
use super::formatting::{build_help_text, markdown_to_telegram_html, html_escape, split_message, format_number, sanitize_filename};
use crate::config::AppConfig;
use crate::tasks::TaskRegistry;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::traits::{Channel, ChannelCapabilities, StateStore};
use crate::types::{ApprovalResponse, ChannelContext, ChannelVisibility, MediaKind, MediaMessage, StatusUpdate, UserRole};

pub struct TelegramChannel {
    /// Bot username fetched from Telegram API (e.g., "coding_bot", "debug_bot").
    /// Populated on first start() call via getMe. Uses StdRwLock for sync access in trait methods.
    bot_username: StdRwLock<String>,
    /// Cached channel name for the trait's name() method (e.g., "telegram" or "telegram:my_bot").
    cached_channel_name: StdRwLock<String>,
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
}

impl TelegramChannel {
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
    ) -> Self {
        let bot = Bot::new(bot_token);
        Self {
            bot_username: StdRwLock::new("telegram".to_string()),
            cached_channel_name: StdRwLock::new("telegram".to_string()),
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
            ids.iter().map(|&id| toml::Value::Integer(id as i64)).collect(),
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
            let guard = self.bot_username.read().unwrap();
            if *guard != "telegram" {
                return guard.clone();
            }
        }

        // Fetch from Telegram API
        match self.bot.get_me().await {
            Ok(me) => {
                let username = me.username.clone().unwrap_or_else(|| "telegram".to_string());
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
    /// Single bot setups use just the chat_id for backward compatibility.
    async fn session_id(&self, chat_id: i64) -> String {
        let username = self.get_bot_username().await;
        if username == "default" {
            chat_id.to_string()
        } else {
            format!("{}:{}", username, chat_id)
        }
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
            .branch(
                Update::filter_message().endpoint({
                    let channel = Arc::clone(&self);
                    move |msg: teloxide::types::Message, bot: Bot| {
                        let channel = Arc::clone(&channel);
                        async move {
                            channel.handle_message(msg, bot).await;
                            respond(())
                        }
                    }
                }),
            )
            .branch(
                Update::filter_callback_query().endpoint({
                    let channel = Arc::clone(&self);
                    move |q: CallbackQuery, bot: Bot| {
                        let channel = Arc::clone(&channel);
                        async move {
                            channel.handle_callback(q, bot).await;
                            respond(())
                        }
                    }
                }),
            );

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
            let allowed = self.allowed_user_ids.read().unwrap();
            !allowed.is_empty() && allowed.contains(&user_id)
        };
        if !is_authorized {
            warn!(user_id, "Unauthorized callback from user");
            let _ = bot.answer_callback_query(q.id).text(format!("Unauthorized. Your ID: {}", user_id)).await;
            return;
        }

        let data = match q.data {
            Some(ref d) => d.clone(),
            None => return,
        };

        // Parse callback data: "approve:{once|session|always|deny}:{id}"
        let parts: Vec<&str> = data.splitn(3, ':').collect();
        if parts.len() != 3 || parts[0] != "approve" {
            return;
        }

        let action = parts[1];
        let approval_id = parts[2];

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
                    format!("Model switched to: {}\nAuto-routing disabled. Use /auto to re-enable.", arg)
                }
            }
            "/models" => {
                match self.agent.list_models().await {
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
                }
            }
            "/auto" => {
                self.agent.clear_model_override().await;
                "Auto-routing re-enabled. Model will be selected automatically based on query complexity.".to_string()
            }
            "/reload" => {
                match AppConfig::load(&self.config_path) {
                    Ok(new_config) => {
                        let new_model = new_config.provider.models.primary.clone();
                        let old_model = self.agent.current_model().await;
                        self.agent.set_model(new_model.clone()).await;
                        self.agent.clear_model_override().await;
                        format!(
                            "Config reloaded. Auto-routing re-enabled.\nModel: {} -> {}",
                            old_model, new_model
                        )
                    }
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
            "/cost" => {
                self.handle_cost_command().await
            }
            "/connect" => {
                self.handle_connect_command(arg, msg.from.as_ref().map(|u| u.id.0).unwrap_or(0)).await
            }
            "/bots" => {
                self.handle_bots_command().await
            }
            "/help" | "/start" => {
                build_help_text(true, true)
            }
            _ => format!("Unknown command: {}\nType /help for available commands.", cmd),
        };

        let _ = bot.send_message(msg.chat.id, reply).await;
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
                doc.file_name.clone().unwrap_or_else(|| "document".to_string()),
                doc.mime_type
                    .as_ref()
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "application/octet-stream".to_string()),
            )
        } else if let Some(photos) = msg.photo() {
            // Last photo in the array is the largest
            let photo = photos.last().ok_or_else(|| anyhow::anyhow!("Empty photo array"))?;
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
        let file_path_on_server = file
            .path;

        // Download via HTTP (simpler than teloxide's Download trait)
        let download_url = format!(
            "https://api.telegram.org/file/bot{}/{}",
            self.bot_token, file_path_on_server
        );
        let response = reqwest::get(&download_url).await?;
        if !response.status().is_success() {
            anyhow::bail!("Failed to download file from Telegram: HTTP {}", response.status());
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
        let since_24h = (now - chrono::Duration::hours(24)).format("%Y-%m-%d %H:%M:%S").to_string();
        let since_7d = (now - chrono::Duration::days(7)).format("%Y-%m-%d %H:%M:%S").to_string();

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
                return format!("Invalid token: {}\n\nMake sure you copied the full token from @BotFather.", e);
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
        let allowed_user_ids_str: Vec<String> = self.allowed_user_ids.read().unwrap().iter().map(|id| id.to_string()).collect();

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
                    self.allowed_user_ids.read().unwrap().clone(),
                    self.owner_user_ids.clone(),
                    Arc::clone(&self.agent),
                    self.config_path.clone(),
                    self.session_map.clone(),
                    Arc::clone(&self.task_registry),
                    self.files_enabled,
                    self.inbox_dir.clone(),
                    self.max_file_size_mb,
                    Arc::clone(&self.state),
                ));

                // Give the new channel a reference to the hub too
                new_channel.set_channel_hub(weak_hub);

                // Register with the hub
                let channel_name = hub.register_channel(new_channel.clone() as Arc<dyn Channel>).await;
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

        // Discord uses user IDs differently - for now, use current user_id
        let allowed_user_ids_str = vec![user_id.to_string()];

        let new_bot = crate::traits::DynamicBot {
            id: 0,
            channel_type: "discord".to_string(),
            bot_token: token.to_string(),
            app_token: None,
            allowed_user_ids: allowed_user_ids_str,
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
                // Create the new Discord channel with same config as this Telegram channel
                let new_channel = Arc::new(DiscordChannel::new(
                    token,
                    vec![user_id], // Discord uses u64 user IDs
                    None,          // No guild_id for dynamic bots
                    Arc::clone(&self.agent),
                    self.config_path.clone(),
                    self.session_map.clone(),
                    Arc::clone(&self.task_registry),
                    self.files_enabled,
                    self.inbox_dir.clone(),
                    self.max_file_size_mb,
                    Arc::clone(&self.state),
                ));

                // Give the new channel a reference to the hub
                new_channel.set_channel_hub(weak_hub);

                // Register with the hub
                let channel_name = hub.register_channel(new_channel.clone() as Arc<dyn Channel>).await;
                info!(channel = %channel_name, "Registered new Discord bot with hub");

                // Spawn the bot in the background
                spawn_discord_channel(new_channel);

                return format!(
                    "✓ Discord bot {} connected and started!\n\n\
                    The bot is now active and ready to receive messages.\n\
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
            Ok(resp) => {
                match resp.json::<serde_json::Value>().await {
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
                }
            }
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
                ));

                // Give the new channel a reference to the hub
                new_channel.set_channel_hub(weak_hub);

                // Register with the hub
                let channel_name = hub.register_channel(new_channel.clone() as Arc<dyn Channel>).await;
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
        bots_list.push(format!("• telegram:@{} (this bot, from config)", current_username));

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
                                    let username = me.username.clone().unwrap_or_else(|| "unknown".to_string());
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

        // Authorization check with first-user auto-claim.
        let auth_result = {
            let allowed = self.allowed_user_ids.read().unwrap();
            if allowed.is_empty() {
                drop(allowed);
                let mut allowed = self.allowed_user_ids.write().unwrap();
                check_auth(&mut allowed, user_id)
            } else if allowed.contains(&user_id) {
                AuthResult::Authorized
            } else {
                AuthResult::Unauthorized
            }
        };

        match auth_result {
            AuthResult::AutoClaimed => {
                // Persist to config.toml so it survives restarts
                let ids = self.allowed_user_ids.read().unwrap().clone();
                if let Err(e) = self.persist_allowed_user_ids(&ids).await {
                    warn!(user_id, "Failed to persist auto-claimed user ID to config: {}", e);
                }
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!(
                            "Welcome! You're the first user — your Telegram ID \
                             (<code>{}</code>) has been saved as owner.",
                            user_id
                        ),
                    )
                    .parse_mode(ParseMode::Html)
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
                    let _ = bot.send_message(msg.chat.id, format!("File error: {}", e)).await;
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
            let mut map = self.session_map.write().await;
            map.insert(session_id.clone(), self.channel_name().await);
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
            }
        };

        // Handle cancel/stop commands - these bypass the queue
        let text_lower = text.to_lowercase();
        if text_lower == "cancel" || text_lower == "stop" {
            let cancelled = self.task_registry.cancel_running_for_session(&session_id).await;
            self.task_registry.clear_queue(&session_id).await;
            if cancelled.is_empty() {
                let _ = bot.send_message(msg.chat.id, "No running task to cancel.").await;
            } else {
                let desc = cancelled.first().map(|(_, d)| d.as_str()).unwrap_or("unknown");
                let queue_cleared = self.task_registry.queue_len(&session_id).await;
                let mut response = format!("⏹️ Cancelled: {}", desc);
                if queue_cleared > 0 {
                    response.push_str(&format!(" (+{} queued messages cleared)", queue_cleared));
                }
                let _ = bot.send_message(msg.chat.id, response).await;
            }
            return;
        }

        // Check if a task is already running - if so, queue this message
        if self.task_registry.has_running_task(&session_id).await {
            let queue_pos = self.task_registry.queue_message(&session_id, &text).await;
            let current_task = self.task_registry.get_running_task_description(&session_id).await
                .unwrap_or_else(|| "processing".to_string());
            let preview: String = text.chars().take(50).collect();
            let suffix = if text.len() > 50 { "..." } else { "" };
            let _ = bot.send_message(
                msg.chat.id,
                format!("📥 Queued ({}): \"{}{}\" | Currently: {}", queue_pos, preview, suffix, current_task)
            ).await;
            return;
        }

        info!(session_id, "Received message from user {}", user_id);

        // Send typing indicator immediately, then repeat every 4s while agent works.
        // Telegram's typing indicator expires after ~5s, so 4s keeps it continuous.
        let typing_bot = bot.clone();
        let typing_chat_id = msg.chat.id;
        let typing_cancel = tokio_util::sync::CancellationToken::new();
        let typing_token = typing_cancel.clone();
        tokio::spawn(async move {
            loop {
                let _ = typing_bot.send_chat_action(typing_chat_id, ChatAction::Typing).await;
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(4)) => {}
                    _ = typing_token.cancelled() => break,
                }
            }
        });

        // Status update channel — agent emits updates, we display them rate-limited.
        let (status_tx, mut status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(16);
        let status_bot = bot.clone();
        let status_chat_id = msg.chat.id;
        let status_task = tokio::spawn(async move {
            let mut last_sent = tokio::time::Instant::now() - Duration::from_secs(10);
            let min_interval = Duration::from_secs(3);
            while let Some(update) = status_rx.recv().await {
                let now = tokio::time::Instant::now();
                if now.duration_since(last_sent) < min_interval {
                    continue;
                }
                let text = match &update {
                    StatusUpdate::Thinking(iter) => format!("Thinking... (step {})", iter + 1),
                    StatusUpdate::ToolStart { name, summary } => {
                        if summary.is_empty() {
                            format!("Using {}...", name)
                        } else {
                            format!("Using {}: {}...", name, summary)
                        }
                    }
                    StatusUpdate::ToolProgress { name, chunk } => {
                        let preview: String = chunk.chars().take(100).collect();
                        if chunk.len() > 100 {
                            format!("📤 {}: {}...", name, preview)
                        } else {
                            format!("📤 {}: {}", name, preview)
                        }
                    }
                    StatusUpdate::ToolComplete { name, summary } => {
                        format!("✓ {}: {}", name, summary)
                    }
                    StatusUpdate::ToolCancellable { name, task_id } => {
                        format!("⏳ {} started (task_id: {})", name, task_id)
                    }
                    StatusUpdate::ProgressSummary { elapsed_mins, summary } => {
                        format!("📊 Progress ({} min): {}", elapsed_mins, summary)
                    }
                    StatusUpdate::IterationWarning { current, threshold } => {
                        format!("⚠️ Approaching soft limit: {} of {} iterations", current, threshold)
                    }
                    StatusUpdate::PlanCreated { description, total_steps, .. } => {
                        format!("📋 Plan created: {} ({} steps)", description, total_steps)
                    }
                    StatusUpdate::PlanStepStart { step_index, total_steps, description, .. } => {
                        format!("▶️ Step {}/{}: {}", step_index + 1, total_steps, description)
                    }
                    StatusUpdate::PlanStepComplete { step_index, total_steps, description, summary, .. } => {
                        let base = format!("✅ Step {}/{} done: {}", step_index + 1, total_steps, description);
                        if let Some(s) = summary {
                            format!("{} - {}", base, s)
                        } else {
                            base
                        }
                    }
                    StatusUpdate::PlanStepFailed { step_index, description, error, .. } => {
                        format!("❌ Step {} failed: {} - {}", step_index + 1, description, error)
                    }
                    StatusUpdate::PlanComplete { description, total_steps, duration_secs, .. } => {
                        let mins = duration_secs / 60;
                        let secs = duration_secs % 60;
                        format!("🎉 Plan complete: {} ({} steps in {}m {}s)", description, total_steps, mins, secs)
                    }
                    StatusUpdate::PlanAbandoned { description, .. } => {
                        format!("🚫 Plan abandoned: {}", description)
                    }
                    StatusUpdate::PlanRevised { description, reason, new_total_steps, .. } => {
                        format!("🔄 Plan revised: {} ({} steps) - {}", description, new_total_steps, reason)
                    }
                };
                let _ = status_bot.send_message(status_chat_id, text).await;
                last_sent = tokio::time::Instant::now();
            }
        });

        // Register this task for tracking and cancellation.
        let description: String = text.chars().take(80).collect();
        let (task_id, cancel_token) = self.task_registry.register(&session_id, &description).await;
        let registry = Arc::clone(&self.task_registry);

        // Spawn the agent work in a separate task so the dispatcher can continue
        // processing other updates (especially callback queries for tool approval).
        let agent = Arc::clone(&self.agent);
        let chat_id = msg.chat.id;
        tokio::spawn(async move {
            let mut current_text = text;
            let mut current_task_id = task_id;
            let mut current_cancel_token = cancel_token;
            let mut current_status_tx = status_tx;
            let mut current_typing_cancel = typing_cancel;
            let mut current_status_task = status_task;

            loop {
                let result = tokio::select! {
                    r = agent.handle_message(&session_id, &current_text, Some(current_status_tx), user_role, channel_ctx.clone()) => r,
                    _ = current_cancel_token.cancelled() => Err(anyhow::anyhow!("Task cancelled")),
                };
                current_typing_cancel.cancel();
                let _ = current_status_task.await;

                match result {
                    Ok(reply) => {
                        registry.complete(current_task_id).await;
                        // Skip sending empty replies (e.g., scheduled tasks with no output)
                        if !reply.trim().is_empty() {
                            let html = markdown_to_telegram_html(&reply);
                            // Split long messages (Telegram limit is 4096 chars)
                            let html_chunks = split_message(&html, 4096);
                            let plain_chunks = split_message(&reply, 4096);
                            for (i, html_chunk) in html_chunks.iter().enumerate() {
                                let plain_chunk = plain_chunks.get(i).map(|s| s.as_str()).unwrap_or(html_chunk.as_str());
                                if let Err(e) = send_html_or_fallback(&bot, chat_id, html_chunk, plain_chunk).await {
                                    warn!("Failed to send Telegram message: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let error_msg = e.to_string();
                        registry.fail(current_task_id, &error_msg).await;
                        // Don't notify user about task cancellation
                        if error_msg == "Task cancelled" {
                            info!("Task #{} cancelled", current_task_id);
                            return; // Exit loop on cancellation
                        }
                        warn!("Agent error: {}", e);
                        let _ = bot
                            .send_message(chat_id, format!("Error: {}", e))
                            .await;
                    }
                }

                // Check if there are queued messages to process
                if let Some(queued) = registry.pop_queued_message(&session_id).await {
                    // Small delay to ensure previous message is fully committed to DB
                    tokio::time::sleep(Duration::from_millis(100)).await;

                    info!(session_id, "Processing queued message: {}", queued.text.chars().take(50).collect::<String>());
                    let _ = bot.send_message(chat_id, format!("▶️ Processing queued: \"{}\"",
                        queued.text.chars().take(50).collect::<String>())).await;

                    // Set up for next iteration
                    current_text = queued.text;
                    let desc: String = current_text.chars().take(80).collect();
                    let (new_task_id, new_cancel_token) = registry.register(&session_id, &desc).await;
                    current_task_id = new_task_id;
                    current_cancel_token = new_cancel_token;

                    // Create new status channel and typing indicator
                    let (new_status_tx, mut new_status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(16);
                    current_status_tx = new_status_tx;

                    let status_bot = bot.clone();
                    current_status_task = tokio::spawn(async move {
                        let mut last_sent = tokio::time::Instant::now() - Duration::from_secs(10);
                        let min_interval = Duration::from_secs(3);
                        while let Some(update) = new_status_rx.recv().await {
                            let now = tokio::time::Instant::now();
                            if now.duration_since(last_sent) < min_interval {
                                continue;
                            }
                            let text = match &update {
                                StatusUpdate::Thinking(iter) => format!("Thinking... (step {})", iter + 1),
                                StatusUpdate::ToolStart { name, summary } => {
                                    if summary.is_empty() {
                                        format!("Using {}...", name)
                                    } else {
                                        format!("Using {}: {}...", name, summary)
                                    }
                                }
                                _ => continue, // Skip other status updates for queued messages
                            };
                            let _ = status_bot.send_message(chat_id, text).await;
                            last_sent = tokio::time::Instant::now();
                        }
                    });

                    // New typing indicator
                    let typing_bot = bot.clone();
                    let new_typing_cancel = tokio_util::sync::CancellationToken::new();
                    current_typing_cancel = new_typing_cancel.clone();
                    tokio::spawn(async move {
                        loop {
                            let _ = typing_bot.send_chat_action(chat_id, ChatAction::Typing).await;
                            tokio::select! {
                                _ = tokio::time::sleep(Duration::from_secs(4)) => {}
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

#[async_trait]
impl Channel for TelegramChannel {
    fn name(&self) -> String {
        self.cached_channel_name.read().unwrap().clone()
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
        let chat_id: i64 = session_id.parse().unwrap_or_else(|_| {
            self.allowed_user_ids.read().unwrap().first().copied().unwrap_or(0) as i64
        });
        let html = markdown_to_telegram_html(text);
        for chunk in split_message(&html, 4096) {
            if let Err(e) = send_html_or_fallback(&self.bot, ChatId(chat_id), &chunk, text).await {
                warn!("Failed to send message: {}", e);
            }
        }
        Ok(())
    }

    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        let chat_id: i64 = session_id.parse().unwrap_or_else(|_| {
            self.allowed_user_ids.read().unwrap().first().copied().unwrap_or(0) as i64
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
            MediaKind::Document { file_path, filename } => {
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
        let chat_id: i64 = session_id.parse().unwrap_or_else(|_| {
            self.allowed_user_ids.read().unwrap().first().copied().unwrap_or(0) as i64
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
                InlineKeyboardButton::callback(
                    "Deny",
                    format!("approve:deny:{}", approval_id),
                ),
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
                InlineKeyboardButton::callback(
                    "Deny",
                    format!("approve:deny:{}", approval_id),
                ),
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
            text.push_str("\n");
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
async fn send_html_or_fallback(bot: &Bot, chat_id: ChatId, html: &str, plain: &str) -> Result<(), teloxide::RequestError> {
    match bot.send_message(chat_id, html).parse_mode(ParseMode::Html).await {
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
        let ids = vec![12345u64];
        let ids_toml = toml::Value::Array(
            ids.iter().map(|&id| toml::Value::Integer(id as i64)).collect(),
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
        let ids = vec![67890u64];
        let ids_toml = toml::Value::Array(
            ids.iter().map(|&id| toml::Value::Integer(id as i64)).collect(),
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
