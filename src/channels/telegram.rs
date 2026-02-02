use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use teloxide::prelude::*;
use teloxide::types::{ChatAction, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, ParseMode};
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};

use crate::agent::Agent;
use crate::config::AppConfig;
use crate::tools::browser::MediaMessage;
use crate::tools::terminal::{ApprovalRequest, ApprovalResponse};

pub struct TelegramChannel {
    bot: Bot,
    allowed_user_ids: Vec<u64>,
    agent: Arc<Agent>,
    config_path: PathBuf,
    approval_rx: Mutex<mpsc::Receiver<ApprovalRequest>>,
    /// Pending approvals keyed by a unique callback ID.
    pending_approvals: Mutex<HashMap<String, tokio::sync::oneshot::Sender<ApprovalResponse>>>,
    media_rx: Mutex<mpsc::Receiver<MediaMessage>>,
}

impl TelegramChannel {
    pub fn new(
        bot_token: &str,
        allowed_user_ids: Vec<u64>,
        agent: Arc<Agent>,
        config_path: PathBuf,
        approval_rx: mpsc::Receiver<ApprovalRequest>,
        media_rx: mpsc::Receiver<MediaMessage>,
    ) -> Self {
        let bot = Bot::new(bot_token);
        Self {
            bot,
            allowed_user_ids,
            agent,
            config_path,
            approval_rx: Mutex::new(approval_rx),
            pending_approvals: Mutex::new(HashMap::new()),
            media_rx: Mutex::new(media_rx),
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

        // Spawn media message listener (screenshots)
        let self_for_media = Arc::clone(&self);
        tokio::spawn(async move {
            self_for_media.media_listener().await;
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

            let escaped_cmd = html_escape(&request.command);
            let text = format!(
                "Command requires approval:\n\n<code>{}</code>\n\n[{}]",
                escaped_cmd, short_id
            );

            if let Err(e) = self
                .bot
                .send_message(ChatId(chat_id), &text)
                .parse_mode(ParseMode::Html)
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

    /// Listens for media messages (screenshots) and sends them as photos.
    async fn media_listener(self: Arc<Self>) {
        loop {
            let msg = {
                let mut rx = self.media_rx.lock().await;
                rx.recv().await
            };

            let msg = match msg {
                Some(m) => m,
                None => break, // channel closed
            };

            let photo = InputFile::memory(msg.photo_bytes).file_name("screenshot.png");
            if let Err(e) = self
                .bot
                .send_photo(ChatId(msg.chat_id), photo)
                .caption(msg.caption)
                .await
            {
                warn!("Failed to send screenshot photo: {}", e);
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
            "/restart" => {
                let _ = bot.send_message(msg.chat.id, "Restarting...").await;
                info!("Restart requested via Telegram");
                restart_process();
                // If exec fails, we're still alive
                "Restart failed. You may need to restart manually.".to_string()
            }
            "/help" | "/start" => {
                "Available commands:\n\
                /model — Show current model\n\
                /model <name> — Switch to a different model\n\
                /models — List available models from provider\n\
                /reload — Reload config.toml (applies model changes)\n\
                /restart — Restart the daemon (picks up new binary, config, MCP servers)\n\
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

        let result = self.agent.handle_message(&session_id, &text).await;
        typing_cancel.cancel();

        match result {
            Ok(reply) => {
                let html = markdown_to_telegram_html(&reply);
                // Split long messages (Telegram limit is 4096 chars)
                let html_chunks = split_message(&html, 4096);
                let plain_chunks = split_message(&reply, 4096);
                for (i, html_chunk) in html_chunks.iter().enumerate() {
                    let plain_chunk = plain_chunks.get(i).map(|s| s.as_str()).unwrap_or(html_chunk.as_str());
                    if let Err(e) = send_html_or_fallback(&bot, msg.chat.id, html_chunk, plain_chunk).await {
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
        let html = markdown_to_telegram_html(text);
        for chunk in split_message(&html, 4096) {
            if let Err(e) = send_html_or_fallback(&self.bot, ChatId(chat_id), &chunk, text).await {
                warn!("Failed to send proactive message: {}", e);
            }
        }
        Ok(())
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
