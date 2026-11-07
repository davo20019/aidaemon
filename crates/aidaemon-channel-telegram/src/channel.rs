//! The [`TelegramChannel`] adapter.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use aidaemon_channels::{
    ApprovalAnswer, ApprovalPrompt, Attachment, Channel, ChannelCapabilities, ChannelError,
    ChannelEvent, CommandSpec, Conversation, MessageRef, OutboundFile, OutboundMessage,
};

use crate::api::{ApiError, BotApi, InlineKeyboardButton, SentMessage, Update, User};
use crate::callback::CallbackCodec;
use crate::format::{html_to_plain, markdown_to_html, split_html};
use crate::mapping::{
    conversation_for_chat, map_message, parse_conversation_id, sender_for, BotIdentity,
};

/// Telegram's message length limit (UTF-16 code units).
pub const TELEGRAM_MAX_MESSAGE_LEN: usize = 4096;

const DEFAULT_API_BASE: &str = "https://api.telegram.org";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Uploads and downloads carry whole files.
const FILE_TIMEOUT: Duration = Duration::from_secs(120);
/// Longest caption Telegram accepts on a file.
const CAPTION_MAX_LEN: usize = 1024;
/// Largest file the hosted Bot API lets a bot download.
pub const MAX_DOWNLOAD_BYTES: u64 = 20 * 1024 * 1024;
/// Longest `retry_after` that `send` waits out by itself before surfacing
/// [`ChannelError::RateLimited`].
const MAX_SEND_RETRY_WAIT: Duration = Duration::from_secs(10);

/// A Telegram bot, driven by long polling over the Bot HTTP API.
///
/// ```no_run
/// use aidaemon_channel_telegram::TelegramChannel;
/// let channel = TelegramChannel::new("000000:SYNTHETIC")
///     .with_name("support-bot")
///     .poll_timeout_secs(50);
/// ```
pub struct TelegramChannel {
    name: String,
    token: String,
    api_base: String,
    poll_timeout_secs: u64,
    configured_username: Option<String>,
    client: Option<reqwest::Client>,
    backoff_initial: Duration,
    backoff_max: Duration,
    state: Arc<Shared>,
}

#[derive(Default)]
struct Shared {
    identity: RwLock<Option<BotIdentity>>,
    codec: CallbackCodec,
    api: RwLock<Option<BotApi>>,
}

impl TelegramChannel {
    /// A channel for the bot with this token. Defaults: name `telegram`,
    /// API base `https://api.telegram.org`, 30 s long-poll timeout, bot
    /// username fetched with `getMe` when `run` starts.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            name: "telegram".to_string(),
            token: token.into(),
            api_base: DEFAULT_API_BASE.to_string(),
            poll_timeout_secs: 30,
            configured_username: None,
            client: None,
            backoff_initial: Duration::from_secs(1),
            backoff_max: Duration::from_secs(60),
            state: Arc::new(Shared::default()),
        }
    }

    /// Channel name, used as the session-id prefix (named `with_name` so it
    /// does not shadow [`Channel::name`]).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Bot API base URL (for a local Bot API server or a test mock).
    pub fn api_base(mut self, base: impl Into<String>) -> Self {
        self.api_base = base.into();
        self.reset_api();
        self
    }

    /// Long-poll timeout passed to `getUpdates`.
    pub fn poll_timeout_secs(mut self, secs: u64) -> Self {
        self.poll_timeout_secs = secs;
        self
    }

    /// The bot's username (with or without `@`). When set, `run` skips
    /// `getMe`; replies are then recognised by username.
    pub fn bot_username(mut self, username: impl Into<String>) -> Self {
        let username = username.into().trim_start_matches('@').to_string();
        *self
            .state
            .identity
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Some(BotIdentity {
            id: None,
            username: username.clone(),
        });
        self.configured_username = Some(username);
        self
    }

    /// Use a preconfigured HTTP client (proxies, custom TLS roots, ...).
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self.reset_api();
        self
    }

    /// Delay bounds for reconnecting after transient polling errors
    /// (exponential between them). Defaults: 1 s to 60 s.
    pub fn retry_backoff(mut self, initial: Duration, max: Duration) -> Self {
        self.backoff_initial = initial;
        self.backoff_max = max.max(initial);
        self
    }

    fn reset_api(&mut self) {
        *self.state.api.write().unwrap_or_else(|e| e.into_inner()) = None;
    }

    fn api(&self) -> BotApi {
        if let Some(api) = self
            .state
            .api
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            return api;
        }
        let client = self.client.clone().unwrap_or_default();
        let api = BotApi::new(client, &self.api_base, &self.token);
        *self.state.api.write().unwrap_or_else(|e| e.into_inner()) = Some(api.clone());
        api
    }

    fn identity(&self) -> BotIdentity {
        self.state
            .identity
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .unwrap_or_default()
    }

    async fn call<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        body: &Value,
    ) -> Result<T, ApiError> {
        self.api().call(method, body, REQUEST_TIMEOUT).await
    }

    /// `call`, waiting out one short 429 before giving up.
    async fn call_with_retry<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        body: &Value,
    ) -> Result<T, ApiError> {
        match self.call(method, body).await {
            Err(ApiError::Api {
                code: 429,
                retry_after: Some(secs),
                ..
            }) if Duration::from_secs(secs) <= MAX_SEND_RETRY_WAIT => {
                tokio::time::sleep(Duration::from_secs(secs)).await;
                self.call(method, body).await
            }
            other => other,
        }
    }

    /// Resolve the bot's identity with `getMe`, retrying transient errors.
    async fn resolve_identity(&self, shutdown: &CancellationToken) -> Result<bool, ChannelError> {
        if self.configured_username.is_some() {
            return Ok(true);
        }
        let mut delay = self.backoff_initial;
        let empty = json!({});
        loop {
            let result = tokio::select! {
                _ = shutdown.cancelled() => return Ok(false),
                r = self.call::<User>("getMe", &empty) => r,
            };
            match result {
                Ok(me) => {
                    *self
                        .state
                        .identity
                        .write()
                        .unwrap_or_else(|e| e.into_inner()) = Some(BotIdentity {
                        id: Some(me.id),
                        username: me.username.unwrap_or_default(),
                    });
                    return Ok(true);
                }
                Err(e) if e.is_auth() => return Err(e.into()),
                Err(e) => {
                    let wait = retry_wait(&e, delay);
                    warn!(channel = %self.name, error = %e, "getMe failed; retrying in {wait:?}");
                    if !sleep_or_shutdown(wait, shutdown).await {
                        return Ok(false);
                    }
                    delay = (delay * 2).min(self.backoff_max);
                }
            }
        }
    }

    async fn handle_update(&self, update: Update, events: &mpsc::Sender<ChannelEvent>) -> bool {
        if let Some(msg) = update.message {
            let identity = self.identity();
            if let Some(inbound) = map_message(&msg, &identity) {
                return events.send(ChannelEvent::Message(inbound)).await.is_ok();
            }
            return true;
        }
        if let Some(query) = update.callback_query {
            let decoded = query
                .data
                .as_deref()
                .and_then(|d| self.state.codec.decode(d));
            // Always answer so the client stops showing a spinner.
            let answer = match decoded {
                Some(_) => json!({ "callback_query_id": query.id }),
                None => json!({
                    "callback_query_id": query.id,
                    "text": "This button has expired.",
                }),
            };
            if let Err(e) = self.call::<Value>("answerCallbackQuery", &answer).await {
                debug!(channel = %self.name, error = %e, "answerCallbackQuery failed");
            }
            let (Some((prompt_id, choice_id)), Some(msg)) = (decoded, query.message) else {
                return true;
            };
            let Some(sender) = sender_for(Some(&query.from), None) else {
                return true;
            };
            let conversation =
                conversation_for_chat(&msg.chat, msg.is_topic_message, msg.message_thread_id);
            let event = ChannelEvent::Approval(ApprovalAnswer {
                prompt_id,
                choice_id,
                sender,
                conversation,
            });
            return events.send(event).await.is_ok();
        }
        true
    }

    /// Send one HTML chunk, falling back to plain text if Telegram rejects
    /// the markup.
    async fn send_chunk(
        &self,
        chat_id: i64,
        thread_id: Option<i64>,
        reply_to: Option<i64>,
        html: &str,
        reply_markup: Option<&Value>,
    ) -> Result<i64, ApiError> {
        let mut body = json!({
            "chat_id": chat_id,
            "text": html,
            "parse_mode": "HTML",
        });
        if let Some(thread) = thread_id {
            body["message_thread_id"] = json!(thread);
        }
        if let Some(reply) = reply_to {
            body["reply_to_message_id"] = json!(reply);
            body["allow_sending_without_reply"] = json!(true);
        }
        if let Some(markup) = reply_markup {
            body["reply_markup"] = markup.clone();
        }
        match self
            .call_with_retry::<SentMessage>("sendMessage", &body)
            .await
        {
            Ok(sent) => Ok(sent.message_id),
            Err(e) if e.is_parse_error() => {
                debug!(channel = %self.name, error = %e, "HTML rejected; resending as plain text");
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("parse_mode");
                }
                body["text"] = json!(html_to_plain(html));
                self.call_with_retry::<SentMessage>("sendMessage", &body)
                    .await
                    .map(|s| s.message_id)
            }
            Err(e) => Err(e),
        }
    }

    /// Send Markdown text as one or more messages; `reply_markup` goes on
    /// the last one. Returns the last message id.
    async fn send_markdown(
        &self,
        conversation: &Conversation,
        text: &str,
        reply_to: Option<&str>,
        reply_markup: Option<&Value>,
    ) -> Result<MessageRef, ChannelError> {
        let (chat_id, thread_id) = parse_conversation_id(&conversation.id).ok_or_else(|| {
            ChannelError::Other(format!(
                "invalid telegram conversation id: {}",
                conversation.id
            ))
        })?;
        let reply_to = reply_to.and_then(|r| r.parse::<i64>().ok());
        let html = markdown_to_html(text);
        let mut chunks = split_html(&html, TELEGRAM_MAX_MESSAGE_LEN);
        if chunks.is_empty() {
            // Telegram rejects empty messages; send something visible.
            chunks.push(if text.trim().is_empty() {
                "\u{2063}".to_string()
            } else {
                crate::format::html_escape(text.trim())
            });
        }
        let last = chunks.len() - 1;
        let mut last_id = 0;
        for (n, chunk) in chunks.iter().enumerate() {
            let markup = if n == last { reply_markup } else { None };
            let reply = if n == 0 { reply_to } else { None };
            last_id = self
                .send_chunk(chat_id, thread_id, reply, chunk, markup)
                .await?;
        }
        Ok(MessageRef {
            conversation_id: conversation.id.clone(),
            message_id: last_id.to_string(),
        })
    }
}

fn retry_wait(e: &ApiError, backoff: Duration) -> Duration {
    match e {
        ApiError::Api {
            code: 429,
            retry_after: Some(secs),
            ..
        } => Duration::from_secs(*secs),
        _ => backoff,
    }
}

/// Sleep unless shutdown fires first; returns false on shutdown.
async fn sleep_or_shutdown(wait: Duration, shutdown: &CancellationToken) -> bool {
    tokio::select! {
        _ = shutdown.cancelled() => false,
        _ = tokio::time::sleep(wait) => true,
    }
}

#[async_trait]
impl Channel for TelegramChannel {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            edit_messages: true,
            buttons: true,
            typing_indicator: true,
            threads: true,
            delete_messages: true,
            files: true,
            commands: true,
            max_message_len: TELEGRAM_MAX_MESSAGE_LEN,
        }
    }

    async fn run(
        &self,
        events: mpsc::Sender<ChannelEvent>,
        shutdown: CancellationToken,
    ) -> Result<(), ChannelError> {
        if !self.resolve_identity(&shutdown).await? {
            return Ok(());
        }
        let api = self.api();
        let poll_timeout = Duration::from_secs(self.poll_timeout_secs);
        let http_timeout = poll_timeout + Duration::from_secs(10);
        let mut offset: Option<i64> = None;
        let mut delay = self.backoff_initial;

        loop {
            let mut body = json!({
                "timeout": self.poll_timeout_secs,
                "allowed_updates": ["message", "callback_query"],
            });
            if let Some(offset) = offset {
                body["offset"] = json!(offset);
            }
            let result = tokio::select! {
                _ = shutdown.cancelled() => return Ok(()),
                _ = events.closed() => return Ok(()),
                r = api.call::<Vec<Update>>("getUpdates", &body, http_timeout) => r,
            };
            match result {
                Ok(updates) => {
                    delay = self.backoff_initial;
                    for update in updates {
                        offset = Some(offset.unwrap_or(i64::MIN).max(update.update_id + 1));
                        if !self.handle_update(update, &events).await {
                            return Ok(()); // receiver dropped
                        }
                        if shutdown.is_cancelled() {
                            return Ok(());
                        }
                    }
                }
                Err(e) if e.is_auth() => return Err(e.into()),
                Err(e) => {
                    let wait = retry_wait(&e, delay);
                    warn!(channel = %self.name, error = %e, "getUpdates failed; retrying in {wait:?}");
                    if !sleep_or_shutdown(wait, &shutdown).await {
                        return Ok(());
                    }
                    if e.code() != Some(429) {
                        delay = (delay * 2).min(self.backoff_max);
                    }
                }
            }
        }
    }

    async fn send(
        &self,
        conversation: &Conversation,
        message: &OutboundMessage,
    ) -> Result<MessageRef, ChannelError> {
        self.send_markdown(
            conversation,
            &message.text,
            message.reply_to.as_deref(),
            None,
        )
        .await
    }

    async fn edit(&self, message: &MessageRef, text: &str) -> Result<(), ChannelError> {
        let (chat_id, _) = parse_conversation_id(&message.conversation_id).ok_or_else(|| {
            ChannelError::Other(format!(
                "invalid telegram conversation id: {}",
                message.conversation_id
            ))
        })?;
        let message_id: i64 = message.message_id.parse().map_err(|_| {
            ChannelError::Other(format!(
                "invalid telegram message id: {}",
                message.message_id
            ))
        })?;
        // An edit replaces one message, so keep only what fits.
        let html = split_html(&markdown_to_html(text), TELEGRAM_MAX_MESSAGE_LEN)
            .into_iter()
            .next()
            .unwrap_or_else(|| "\u{2063}".to_string());
        let mut body = json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "text": html,
            "parse_mode": "HTML",
        });
        let result = match self
            .call_with_retry::<Value>("editMessageText", &body)
            .await
        {
            Err(e) if e.is_parse_error() => {
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("parse_mode");
                }
                body["text"] = json!(html_to_plain(&html));
                self.call_with_retry::<Value>("editMessageText", &body)
                    .await
            }
            other => other,
        };
        match result {
            Ok(_) => Ok(()),
            Err(e) if e.is_not_modified() => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    async fn ask(&self, prompt: &ApprovalPrompt) -> Result<MessageRef, ChannelError> {
        let buttons: Vec<InlineKeyboardButton> = prompt
            .choices
            .iter()
            .map(|choice| InlineKeyboardButton {
                text: choice.label.clone(),
                callback_data: self.state.codec.encode(&prompt.id, &choice.id),
            })
            .collect();
        // Up to three buttons share a row; more get one row each.
        let rows: Vec<Vec<InlineKeyboardButton>> = if buttons.len() <= 3 {
            vec![buttons]
        } else {
            buttons.into_iter().map(|b| vec![b]).collect()
        };
        if prompt.choices.is_empty() {
            return self
                .send_markdown(&prompt.conversation, &prompt.text, None, None)
                .await;
        }
        let markup = json!({ "inline_keyboard": rows });
        self.send_markdown(&prompt.conversation, &prompt.text, None, Some(&markup))
            .await
    }

    async fn typing(&self, conversation: &Conversation) -> Result<(), ChannelError> {
        let Some((chat_id, thread_id)) = parse_conversation_id(&conversation.id) else {
            return Ok(());
        };
        let mut body = json!({ "chat_id": chat_id, "action": "typing" });
        if let Some(thread) = thread_id {
            body["message_thread_id"] = json!(thread);
        }
        self.call::<Value>("sendChatAction", &body)
            .await
            .map(|_| ())
            .map_err(Into::into)
    }

    async fn delete(&self, message: &MessageRef) -> Result<(), ChannelError> {
        let (chat_id, _) = parse_conversation_id(&message.conversation_id).ok_or_else(|| {
            ChannelError::Other(format!(
                "invalid telegram conversation id: {}",
                message.conversation_id
            ))
        })?;
        let message_id: i64 = message.message_id.parse().map_err(|_| {
            ChannelError::Other(format!(
                "invalid telegram message id: {}",
                message.message_id
            ))
        })?;
        self.call_with_retry::<Value>(
            "deleteMessage",
            &json!({ "chat_id": chat_id, "message_id": message_id }),
        )
        .await
        .map(|_| ())
        .map_err(Into::into)
    }

    /// Sent as a document, so the file arrives as it is (no recompression).
    async fn send_file(
        &self,
        conversation: &Conversation,
        file: &OutboundFile,
    ) -> Result<MessageRef, ChannelError> {
        let (chat_id, thread_id) = parse_conversation_id(&conversation.id).ok_or_else(|| {
            ChannelError::Other(format!(
                "invalid telegram conversation id: {}",
                conversation.id
            ))
        })?;
        let mut part =
            reqwest::multipart::Part::bytes(file.bytes.clone()).file_name(file.name.clone());
        if let Some(mime) = file.mime_type.as_deref() {
            part = part
                .mime_str(mime)
                .map_err(|e| ChannelError::Other(format!("invalid mime type {mime}: {e}")))?;
        }
        let mut form = reqwest::multipart::Form::new()
            .text("chat_id", chat_id.to_string())
            .part("document", part);
        if let Some(thread) = thread_id {
            form = form.text("message_thread_id", thread.to_string());
        }
        if let Some(caption) = file.caption.as_deref().filter(|c| !c.trim().is_empty()) {
            // Plain text: an HTML caption cut at the limit could break a tag.
            let plain = html_to_plain(&markdown_to_html(caption));
            let caption: String = plain.chars().take(CAPTION_MAX_LEN).collect();
            form = form.text("caption", caption);
        }
        let sent: SentMessage = self
            .api()
            .call_multipart("sendDocument", form, FILE_TIMEOUT)
            .await?;
        Ok(MessageRef {
            conversation_id: conversation.id.clone(),
            message_id: sent.message_id.to_string(),
        })
    }

    /// `source` is a Telegram file id: resolved with `getFile`, then
    /// fetched from the file endpoint.
    async fn download(&self, attachment: &Attachment) -> Result<Vec<u8>, ChannelError> {
        if attachment
            .size_bytes
            .is_some_and(|size| size > MAX_DOWNLOAD_BYTES)
        {
            return Err(ChannelError::Other(format!(
                "telegram bots cannot download files over {MAX_DOWNLOAD_BYTES} bytes"
            )));
        }
        let file: Value = self
            .call_with_retry("getFile", &json!({ "file_id": attachment.source }))
            .await?;
        let path = file
            .get("file_path")
            .and_then(Value::as_str)
            .ok_or_else(|| ChannelError::Other("telegram getFile: no file_path".into()))?;
        self.api()
            .download(path, FILE_TIMEOUT)
            .await
            .map_err(Into::into)
    }

    /// The bot's `/` menu (`setMyCommands`).
    async fn register_commands(&self, commands: &[CommandSpec]) -> Result<(), ChannelError> {
        let commands: Vec<Value> = commands
            .iter()
            .map(|command| {
                json!({
                    "command": command.name,
                    "description": command.description.chars().take(256).collect::<String>(),
                })
            })
            .collect();
        self.call_with_retry::<Value>("setMyCommands", &json!({ "commands": commands }))
            .await
            .map(|_| ())
            .map_err(Into::into)
    }
}
