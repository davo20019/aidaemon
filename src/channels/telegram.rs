use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use teloxide::prelude::*;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup};
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};

use crate::agent::Agent;
use crate::config::AppConfig;
use crate::tools::terminal::{ApprovalRequest, ApprovalResponse};

pub struct TelegramChannel {
    bot: Bot,
    allowed_user_ids: Vec<u64>,
    agent: Arc<Agent>,
    config_path: PathBuf,
    approval_rx: Mutex<mpsc::Receiver<ApprovalRequest>>,
    /// Pending approvals keyed by a unique callback ID.
    pending_approvals: Mutex<HashMap<String, tokio::sync::oneshot::Sender<ApprovalResponse>>>,
}

impl TelegramChannel {
    pub fn new(
        bot_token: &str,
        allowed_user_ids: Vec<u64>,
        agent: Arc<Agent>,
        config_path: PathBuf,
        approval_rx: mpsc::Receiver<ApprovalRequest>,
    ) -> Self {
        let bot = Bot::new(bot_token);
        Self {
            bot,
            allowed_user_ids,
            agent,
            config_path,
            approval_rx: Mutex::new(approval_rx),
            pending_approvals: Mutex::new(HashMap::new()),
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

        // Spawn approval request listener
        let self_for_approvals = Arc::clone(&self);
        tokio::spawn(async move {
            self_for_approvals.approval_listener().await;
        });

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

    /// Listens for approval requests from the terminal tool and sends
    /// inline keyboard prompts to the first allowed user.
    async fn approval_listener(self: Arc<Self>) {
        loop {
            let request = {
                let mut rx = self.approval_rx.lock().await;
                rx.recv().await
            };

            let request = match request {
                Some(r) => r,
                None => break, // channel closed
            };

            let approval_id = uuid::Uuid::new_v4().to_string();
            let short_id = &approval_id[..8];

            // Store the response sender
            {
                let mut pending = self.pending_approvals.lock().await;
                pending.insert(approval_id.clone(), request.response_tx);
            }

            // Send inline keyboard to the first allowed user
            let chat_id = self
                .allowed_user_ids
                .first()
                .copied()
                .unwrap_or(0) as i64;

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

            let text = format!(
                "Command requires approval:\n\n`{}`\n\n[{}]",
                request.command, short_id
            );

            if let Err(e) = self
                .bot
                .send_message(ChatId(chat_id), text)
                .reply_markup(keyboard)
                .await
            {
                warn!("Failed to send approval request: {}", e);
                // Remove pending approval since we couldn't ask
                let mut pending = self.pending_approvals.lock().await;
                if let Some(tx) = pending.remove(&approval_id) {
                    let _ = tx.send(ApprovalResponse::Deny);
                }
            }
        }
    }

    /// Handle callback query from inline keyboard buttons.
    async fn handle_callback(&self, q: CallbackQuery, bot: Bot) {
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
                    format!("Current model: {}\n\nUsage: /model <model-name>\nExample: /model gemini-2.5-pro-preview-06-05", current)
                } else {
                    self.agent.set_model(arg.to_string()).await;
                    format!("Model switched to: {}", arg)
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
            "/reload" => {
                match AppConfig::load(&self.config_path) {
                    Ok(new_config) => {
                        let new_model = new_config.provider.models.primary.clone();
                        // Test the new model with a lightweight call
                        let old_model = self.agent.current_model().await;
                        self.agent.set_model(new_model.clone()).await;
                        format!(
                            "Config reloaded.\nModel: {} -> {}",
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
            "/help" | "/start" => {
                "Available commands:\n\
                /model — Show current model\n\
                /model <name> — Switch to a different model\n\
                /models — List available models from provider\n\
                /reload — Reload config.toml (applies model changes)\n\
                /help — Show this help message"
                    .to_string()
            }
            _ => format!("Unknown command: {}\nType /help for available commands.", cmd),
        };

        let _ = bot.send_message(msg.chat.id, reply).await;
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

        let text = match msg.text() {
            Some(t) => t.to_string(),
            None => {
                let _ = bot
                    .send_message(msg.chat.id, "I can only process text messages for now.")
                    .await;
                return;
            }
        };

        // Handle slash commands
        if text.starts_with('/') {
            self.handle_command(&text, &msg, &bot).await;
            return;
        }

        // Use chat ID as session ID
        let session_id = msg.chat.id.0.to_string();

        info!(session_id, "Received message from user {}", user_id);

        match self.agent.handle_message(&session_id, &text).await {
            Ok(reply) => {
                // Split long messages (Telegram limit is 4096 chars)
                for chunk in split_message(&reply, 4096) {
                    if let Err(e) = bot.send_message(msg.chat.id, chunk).await {
                        warn!("Failed to send Telegram message: {}", e);
                    }
                }
            }
            Err(e) => {
                warn!("Agent error: {}", e);
                let _ = bot
                    .send_message(msg.chat.id, format!("Error: {}", e))
                    .await;
            }
        }
    }

    /// Send a proactive message to a chat (used by triggers/events).
    pub async fn send_to_chat(&self, chat_id: i64, text: &str) -> anyhow::Result<()> {
        self.bot
            .send_message(ChatId(chat_id), text)
            .await?;
        Ok(())
    }
}

fn split_message(text: &str, max_len: usize) -> Vec<&str> {
    if text.len() <= max_len {
        return vec![text];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let end = std::cmp::min(start + max_len, text.len());
        chunks.push(&text[start..end]);
        start = end;
    }
    chunks
}
