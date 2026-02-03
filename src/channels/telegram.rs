use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use teloxide::prelude::*;
use teloxide::types::{ChatAction, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, ParseMode};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::agent::{Agent, StatusUpdate};
use crate::channels::SessionMap;
use crate::config::AppConfig;
use crate::tasks::TaskRegistry;
use crate::traits::{Channel, ChannelCapabilities, StateStore};
use crate::types::{ApprovalResponse, MediaKind, MediaMessage};

pub struct TelegramChannel {
    bot: Bot,
    bot_token: String,
    allowed_user_ids: Vec<u64>,
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
}

impl TelegramChannel {
    pub fn new(
        bot_token: &str,
        allowed_user_ids: Vec<u64>,
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
            bot,
            bot_token: bot_token.to_string(),
            allowed_user_ids,
            agent,
            config_path,
            pending_approvals: Mutex::new(HashMap::new()),
            session_map,
            task_registry,
            files_enabled,
            inbox_dir,
            max_file_size_mb,
            state,
        }
    }

    /// Start the Telegram dispatcher with automatic retry on crash.
    /// Uses exponential backoff: 5s → 10s → 20s → 40s → 60s cap.
    /// Resets backoff to initial after a stable run (60s+).
    pub async fn start_with_retry(self: Arc<Self>) {
        let initial_backoff = Duration::from_secs(5);
        let max_backoff = Duration::from_secs(60);
        let stable_threshold = Duration::from_secs(60);
        let mut backoff = initial_backoff;

        loop {
            info!("Starting Telegram dispatcher");
            let started = tokio::time::Instant::now();
            self.clone().start().await;
            let ran_for = started.elapsed();

            // If the dispatcher ran for long enough, it was a stable session —
            // reset backoff so the next crash recovers quickly.
            if ran_for >= stable_threshold {
                backoff = initial_backoff;
            }

            warn!(
                backoff_secs = backoff.as_secs(),
                ran_for_secs = ran_for.as_secs(),
                "Telegram dispatcher stopped, restarting"
            );
            tokio::time::sleep(backoff).await;
            backoff = std::cmp::min(backoff * 2, max_backoff);
        }
    }

    pub async fn start(self: Arc<Self>) {
        info!("Starting Telegram channel");

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
        let user_id = q.from.id.0;
        if !self.allowed_user_ids.is_empty() && !self.allowed_user_ids.contains(&user_id) {
            warn!(user_id, "Unauthorized callback from user");
            let _ = bot.answer_callback_query(q.id).text("Unauthorized.").await;
            return;
        }

        let data = match q.data {
            Some(ref d) => d.clone(),
            None => return,
        };

        // Parse callback data: "approve:{once|always|deny}:{id}"
        let parts: Vec<&str> = data.splitn(3, ':').collect();
        if parts.len() != 3 || parts[0] != "approve" {
            return;
        }

        let action = parts[1];
        let approval_id = parts[2];

        let response = match action {
            "once" => ApprovalResponse::AllowOnce,
            "always" => ApprovalResponse::AllowAlways,
            "deny" => ApprovalResponse::Deny,
            _ => return,
        };

        let label = match &response {
            ApprovalResponse::AllowOnce => "Allowed (once)",
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
                let session_id = msg.chat.id.0.to_string();
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
            "/cost" => {
                self.handle_cost_command().await
            }
            "/help" | "/start" => {
                "Available commands:\n\
                /model — Show current model\n\
                /model <name> — Switch to a different model (disables auto-routing)\n\
                /models — List available models from provider\n\
                /auto — Re-enable automatic model routing by query complexity\n\
                /reload — Reload config.toml (applies model changes, re-enables auto-routing)\n\
                /restart — Restart the daemon (picks up new binary, config, MCP servers)\n\
                /tasks — List running and recent tasks\n\
                /cancel <id> — Cancel a running task\n\
                /cost — Show token usage statistics\n\
                /help — Show this help message"
                    .to_string()
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

    async fn handle_message(&self, msg: teloxide::types::Message, bot: Bot) {
        let user_id = msg.from.as_ref().map(|u| u.id.0).unwrap_or(0);

        // Authorization check
        if !self.allowed_user_ids.is_empty() && !self.allowed_user_ids.contains(&user_id) {
            warn!(user_id, "Unauthorized user attempted access");
            let _ = bot
                .send_message(msg.chat.id, "Unauthorized.")
                .await;
            return;
        }

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

        // Use chat ID as session ID
        let session_id = msg.chat.id.0.to_string();

        // Register this session with the channel hub so outbound messages
        // (approvals, media, notifications) route back to Telegram.
        {
            let mut map = self.session_map.write().await;
            map.insert(session_id.clone(), "telegram".to_string());
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
            let result = tokio::select! {
                r = agent.handle_message(&session_id, &text, Some(status_tx)) => r,
                _ = cancel_token.cancelled() => Err(anyhow::anyhow!("Task cancelled")),
            };
            typing_cancel.cancel();
            // status_tx is dropped here (moved into handle_message), ending the receiver task.
            let _ = status_task.await;

            match result {
                Ok(reply) => {
                    registry.complete(task_id).await;
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
                Err(e) => {
                    let error_msg = e.to_string();
                    registry.fail(task_id, &error_msg).await;
                    warn!("Agent error: {}", e);
                    let _ = bot
                        .send_message(chat_id, format!("Error: {}", e))
                        .await;
                }
            }
        });
    }
}

#[async_trait]
impl Channel for TelegramChannel {
    fn name(&self) -> &str {
        "telegram"
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
            self.allowed_user_ids.first().copied().unwrap_or(0) as i64
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
            self.allowed_user_ids.first().copied().unwrap_or(0) as i64
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
    ) -> anyhow::Result<ApprovalResponse> {
        let chat_id: i64 = session_id.parse().unwrap_or_else(|_| {
            self.allowed_user_ids.first().copied().unwrap_or(0) as i64
        });

        info!(session_id, command, chat_id, "Approval requested for command");

        let approval_id = uuid::Uuid::new_v4().to_string();
        let short_id = &approval_id[..8];

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        // Store the response sender
        {
            let mut pending = self.pending_approvals.lock().await;
            pending.insert(approval_id.clone(), response_tx);
            info!(approval_id = %short_id, pending_count = pending.len(), "Stored pending approval");
        }

        // Send inline keyboard
        let keyboard = InlineKeyboardMarkup::new(vec![vec![
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
        ]]);

        let escaped_cmd = html_escape(command);
        let text = format!(
            "Command requires approval:\n\n<code>{}</code>\n\n[{}]",
            escaped_cmd, short_id
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

        info!(approval_id = %short_id, "Waiting for user approval response...");
        response_rx
            .await
            .map_err(|_| anyhow::anyhow!("Approval response channel closed"))
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

/// Convert common LLM markdown to Telegram-compatible HTML.
fn markdown_to_telegram_html(md: &str) -> String {
    let mut result = String::with_capacity(md.len() + md.len() / 4);
    let lines: Vec<&str> = md.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Fenced code blocks: ```lang\n...\n```
        if line.starts_with("```") {
            i += 1;
            let mut code = String::new();
            while i < lines.len() && !lines[i].starts_with("```") {
                if !code.is_empty() {
                    code.push('\n');
                }
                code.push_str(lines[i]);
                i += 1;
            }
            if i < lines.len() {
                i += 1; // skip closing ```
            }
            // HTML-escape the code content
            let escaped = html_escape(&code);
            result.push_str("<pre><code>");
            result.push_str(&escaped);
            result.push_str("</code></pre>");
            result.push('\n');
            continue;
        }

        // Process a non-code line: escape HTML first, then apply inline formatting
        let escaped = html_escape(line);

        // Heading lines: ### heading → <b>heading</b>
        if let Some(heading) = strip_heading(&escaped) {
            result.push_str("<b>");
            result.push_str(&heading);
            result.push_str("</b>");
            result.push('\n');
            i += 1;
            continue;
        }

        // Unordered list markers: "- " or "* " at start → "• "
        let processed = if escaped.starts_with("- ") {
            format!("• {}", &escaped[2..])
        } else if escaped.starts_with("* ") {
            format!("• {}", &escaped[2..])
        } else {
            escaped
        };

        // Inline formatting
        let processed = convert_inline_formatting(&processed);

        result.push_str(&processed);
        result.push('\n');
        i += 1;
    }

    // Remove trailing newline
    if result.ends_with('\n') {
        result.pop();
    }
    result
}

/// Escape `<`, `>`, `&` for Telegram HTML.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Strip markdown heading prefix (e.g. "### Foo" → "Foo"). Returns None if not a heading.
fn strip_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if trimmed.starts_with('#') {
        let after_hashes = trimmed.trim_start_matches('#');
        if after_hashes.starts_with(' ') {
            return Some(after_hashes.trim_start().to_string());
        }
    }
    None
}

/// Apply inline markdown formatting: bold, italic, inline code, links.
fn convert_inline_formatting(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Inline code: `code`
        if chars[i] == '`' {
            if let Some(end) = find_char(&chars, '`', i + 1) {
                result.push_str("<code>");
                let inner: String = chars[i + 1..end].iter().collect();
                result.push_str(&inner);
                result.push_str("</code>");
                i = end + 1;
                continue;
            }
        }

        // Bold: **text**
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_double_char(&chars, '*', i + 2) {
                result.push_str("<b>");
                let inner: String = chars[i + 2..end].iter().collect();
                result.push_str(&inner);
                result.push_str("</b>");
                i = end + 2;
                continue;
            }
        }

        // Link: [text](url)
        if chars[i] == '[' {
            if let Some((text, url, end)) = parse_link(&chars, i) {
                result.push_str("<a href=\"");
                result.push_str(&url);
                result.push_str("\">");
                result.push_str(&text);
                result.push_str("</a>");
                i = end;
                continue;
            }
        }

        // Italic: _text_ (but not inside words like some_var_name)
        if chars[i] == '_' && (i == 0 || chars[i - 1] == ' ') {
            if let Some(end) = find_char(&chars, '_', i + 1) {
                if end + 1 >= len || chars[end + 1] == ' ' || chars[end + 1] == '.' || chars[end + 1] == ',' {
                    result.push_str("<i>");
                    let inner: String = chars[i + 1..end].iter().collect();
                    result.push_str(&inner);
                    result.push_str("</i>");
                    i = end + 1;
                    continue;
                }
            }
        }

        // Single *italic* (not **)
        if chars[i] == '*' && (i + 1 >= len || chars[i + 1] != '*') {
            if let Some(end) = find_single_star(&chars, i + 1) {
                result.push_str("<i>");
                let inner: String = chars[i + 1..end].iter().collect();
                result.push_str(&inner);
                result.push_str("</i>");
                i = end + 1;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }
    result
}

fn find_char(chars: &[char], c: char, start: usize) -> Option<usize> {
    for j in start..chars.len() {
        if chars[j] == c {
            return Some(j);
        }
    }
    None
}

fn find_double_char(chars: &[char], c: char, start: usize) -> Option<usize> {
    let mut j = start;
    while j + 1 < chars.len() {
        if chars[j] == c && chars[j + 1] == c {
            return Some(j);
        }
        j += 1;
    }
    None
}

fn find_single_star(chars: &[char], start: usize) -> Option<usize> {
    for j in start..chars.len() {
        if chars[j] == '*' && (j + 1 >= chars.len() || chars[j + 1] != '*') {
            return Some(j);
        }
    }
    None
}

fn parse_link(chars: &[char], start: usize) -> Option<(String, String, usize)> {
    // [text](url)
    let close_bracket = find_char(chars, ']', start + 1)?;
    if close_bracket + 1 >= chars.len() || chars[close_bracket + 1] != '(' {
        return None;
    }
    let close_paren = find_char(chars, ')', close_bracket + 2)?;
    let text: String = chars[start + 1..close_bracket].iter().collect();
    let url: String = chars[close_bracket + 2..close_paren].iter().collect();
    Some((text, url, close_paren + 1))
}

/// Split a message into chunks respecting Telegram's max length.
/// Prefers splitting at paragraph boundaries, then line boundaries.
/// Never splits inside HTML tags or code blocks.
fn split_message(text: &str, max_len: usize) -> Vec<String> {
    if text.len() <= max_len {
        return vec![text.to_string()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= max_len {
            chunks.push(remaining.to_string());
            break;
        }

        let search_region = &remaining[..max_len];

        // Try paragraph boundary first
        let split_at = search_region.rfind("\n\n")
            .map(|p| p + 1)  // include first \n, second starts next chunk
            // Then try line boundary
            .or_else(|| search_region.rfind('\n'))
            // Last resort: split at max_len
            .unwrap_or(max_len);

        // Ensure we don't split inside an HTML tag
        let split_at = adjust_for_html_tags(search_region, split_at);

        let (chunk, rest) = remaining.split_at(split_at);
        let chunk = chunk.trim_end();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }
        remaining = rest.trim_start_matches('\n');
    }

    chunks
}

/// If the split point is inside an HTML tag, move it before the tag start.
fn adjust_for_html_tags(text: &str, split_at: usize) -> usize {
    let bytes = text.as_bytes();
    // Walk backward from split_at to check if we're inside a tag
    let mut j = split_at;
    while j > 0 {
        j -= 1;
        if bytes[j] == b'>' {
            // We're outside a tag, safe to split at original point
            return split_at;
        }
        if bytes[j] == b'<' {
            // We're inside a tag — split before it
            return j;
        }
    }
    split_at
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

/// Format a number with comma separators (e.g. 12450 → "12,450").
fn format_number(n: i64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Sanitize a filename: remove path separators, null bytes, and limit length.
fn sanitize_filename(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .filter(|c| *c != '/' && *c != '\\' && *c != '\0')
        .collect();
    // Limit to 200 chars, preserving extension
    if sanitized.len() <= 200 {
        sanitized
    } else if let Some(dot_pos) = sanitized.rfind('.') {
        let ext = &sanitized[dot_pos..];
        if ext.len() < 20 {
            let stem_len = 200 - ext.len();
            format!("{}{}", &sanitized[..stem_len], ext)
        } else {
            sanitized[..200].to_string()
        }
    } else {
        sanitized[..200].to_string()
    }
}
