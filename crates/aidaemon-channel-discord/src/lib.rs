//! # aidaemon-channel-discord (experimental)
//!
//! A Discord [`Channel`] for `aidaemon-channels`, talking to the Discord
//! Gateway (v10, JSON) over a WebSocket and to the REST API over HTTP. It is
//! a transport only: sessions, authorization and approvals live in the
//! runner.
//!
//! ```no_run
//! use aidaemon_channel_discord::DiscordChannel;
//!
//! let channel = DiscordChannel::new("bot-token").with_name("discord_main");
//! ```
//!
//! Thread detection: Discord's MESSAGE_CREATE does not say whether the
//! channel is a thread. The adapter learns thread ids from
//! THREAD_CREATE/THREAD_UPDATE/THREAD_LIST_SYNC/GUILD_CREATE (which need the
//! [`intents::GUILDS`] intent) and from a `channel_type` field when present;
//! otherwise guild messages are reported as
//! [`ConversationKind::Group`](aidaemon_channels::ConversationKind::Group).

mod format;
mod gateway;
mod mapping;
mod rest;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_channels::{
    ApprovalPrompt, Channel, ChannelCapabilities, ChannelError, ChannelEvent, Conversation,
    MessageRef, OutboundMessage,
};
use async_trait::async_trait;
use reqwest::Method;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub use format::split_message;
pub use mapping::{decode_custom_id, encode_custom_id, CUSTOM_ID_MAX};

/// Longest message Discord accepts, in characters.
pub const MAX_MESSAGE_LEN: usize = 2000;
/// Default REST base URL.
pub const DEFAULT_API_BASE: &str = "https://discord.com/api/v10";

/// Gateway intent bits.
pub mod intents {
    pub const GUILDS: u64 = 1 << 0;
    pub const GUILD_MESSAGES: u64 = 1 << 9;
    pub const DIRECT_MESSAGES: u64 = 1 << 12;
    /// Privileged: enable "Message Content Intent" in the developer portal.
    pub const MESSAGE_CONTENT: u64 = 1 << 15;
    /// GUILD_MESSAGES | DIRECT_MESSAGES | MESSAGE_CONTENT.
    pub const DEFAULT: u64 = GUILD_MESSAGES | DIRECT_MESSAGES | MESSAGE_CONTENT;
}

/// Discord channel adapter. Cheap to share behind an `Arc`.
pub struct DiscordChannel {
    name: String,
    rest: rest::Rest,
    gateway_url: Mutex<Option<String>>,
    intents: u64,
    backoff_initial: Duration,
    backoff_max: Duration,
    state: Arc<Mutex<gateway::GatewayState>>,
}

impl DiscordChannel {
    /// A channel for `bot_token` (without the `Bot ` prefix).
    pub fn new(bot_token: impl Into<String>) -> Self {
        Self {
            name: "discord".to_string(),
            rest: rest::Rest {
                http: reqwest::Client::new(),
                api_base: DEFAULT_API_BASE.to_string(),
                token: bot_token.into(),
            },
            gateway_url: Mutex::new(None),
            intents: intents::DEFAULT,
            backoff_initial: Duration::from_secs(1),
            backoff_max: Duration::from_secs(60),
            state: Arc::new(Mutex::new(gateway::GatewayState::default())),
        }
    }

    /// Channel name (session-id prefix). Default `discord`.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// REST base URL. Default [`DEFAULT_API_BASE`].
    pub fn with_api_base(mut self, api_base: impl Into<String>) -> Self {
        self.rest.api_base = api_base.into();
        self
    }

    /// Gateway URL (for example `wss://gateway.discord.gg`). Default: looked
    /// up once with `GET /gateway/bot`. Query parameters `v=10&encoding=json`
    /// are appended unless the URL already has a query.
    pub fn with_gateway_url(self, gateway_url: impl Into<String>) -> Self {
        *self.gateway_url.lock().unwrap() = Some(gateway_url.into());
        self
    }

    /// Gateway intents bitmask. Default [`intents::DEFAULT`].
    pub fn with_intents(mut self, intents: u64) -> Self {
        self.intents = intents;
        self
    }

    /// Reconnect backoff: first wait and cap (doubling in between). The
    /// first wait also scales the 1-5 s pause after an Invalid Session.
    /// Default 1 s and 60 s.
    pub fn with_reconnect_backoff(mut self, initial: Duration, max: Duration) -> Self {
        self.backoff_initial = initial;
        self.backoff_max = max.max(initial);
        self
    }

    /// Use a preconfigured HTTP client (proxies, timeouts, test setups).
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.rest.http = client;
        self
    }

    /// The bot's user id, known after the gateway's READY event.
    pub fn bot_user_id(&self) -> Option<String> {
        self.state.lock().unwrap().bot_user_id.clone()
    }

    async fn post_message(
        &self,
        channel_id: &str,
        body: &Value,
    ) -> Result<MessageRef, ChannelError> {
        let resp = self
            .rest
            .request(
                Method::POST,
                &format!("/channels/{channel_id}/messages"),
                Some(body),
            )
            .await?;
        let message_id = resp
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| ChannelError::Other("discord: message response without id".into()))?;
        Ok(MessageRef {
            conversation_id: channel_id.to_string(),
            message_id: message_id.to_string(),
        })
    }

    /// Send all but the last chunk of `text` as plain messages and return
    /// the last chunk (to be sent by the caller with extra fields).
    async fn send_leading_chunks(
        &self,
        channel_id: &str,
        text: &str,
        reply_to: Option<&str>,
    ) -> Result<(String, Option<String>), ChannelError> {
        if text.trim().is_empty() {
            return Err(ChannelError::Other(
                "discord: cannot send an empty message".into(),
            ));
        }
        let mut chunks = split_message(text, MAX_MESSAGE_LEN);
        let last = chunks.pop().unwrap_or_default();
        let mut reply_to = reply_to.map(str::to_string);
        for chunk in chunks {
            let body = message_body(&chunk, reply_to.take().as_deref());
            self.post_message(channel_id, &body).await?;
        }
        Ok((last, reply_to))
    }
}

fn message_body(content: &str, reply_to: Option<&str>) -> Value {
    let mut body = json!({
        "content": content,
        "allowed_mentions": {"parse": []},
    });
    if let Some(id) = reply_to {
        body["message_reference"] = json!({"message_id": id, "fail_if_not_exists": false});
    }
    body
}

fn approval_components(prompt: &ApprovalPrompt) -> Result<Value, ChannelError> {
    if prompt.choices.is_empty() {
        return Err(ChannelError::Other("approval prompt has no choices".into()));
    }
    if prompt.choices.len() > 25 {
        return Err(ChannelError::Other(
            "discord supports at most 25 buttons per message".into(),
        ));
    }
    let mut buttons = Vec::with_capacity(prompt.choices.len());
    for (i, choice) in prompt.choices.iter().enumerate() {
        let custom_id = encode_custom_id(&prompt.id, &choice.id).ok_or_else(|| {
            ChannelError::Other(format!(
                "approval ids too long for a discord custom_id ({CUSTOM_ID_MAX} chars max)"
            ))
        })?;
        // By position: first = success (green), second = danger (red).
        let style = match i {
            0 => 3,
            1 => 4,
            _ => 2,
        };
        buttons.push(json!({
            "type": 2,
            "style": style,
            "label": format::truncate_chars(&choice.label, 80),
            "custom_id": custom_id,
        }));
    }
    let rows: Vec<Value> = buttons
        .chunks(5)
        .map(|row| json!({"type": 1, "components": row}))
        .collect();
    Ok(Value::Array(rows))
}

#[async_trait]
impl Channel for DiscordChannel {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            edit_messages: true,
            buttons: true,
            typing_indicator: true,
            threads: true,
            delete_messages: false,
            files: false,
            commands: false,
            max_message_len: MAX_MESSAGE_LEN,
        }
    }

    async fn run(
        &self,
        events: mpsc::Sender<ChannelEvent>,
        shutdown: CancellationToken,
    ) -> Result<(), ChannelError> {
        self.run_gateway(events, shutdown).await
    }

    async fn send(
        &self,
        conversation: &Conversation,
        message: &OutboundMessage,
    ) -> Result<MessageRef, ChannelError> {
        let (last, reply_to) = self
            .send_leading_chunks(&conversation.id, &message.text, message.reply_to.as_deref())
            .await?;
        self.post_message(&conversation.id, &message_body(&last, reply_to.as_deref()))
            .await
    }

    /// Replaces the content and removes any buttons (so an answered
    /// approval prompt cannot be pressed again). Text over 2000 characters
    /// is truncated.
    async fn edit(&self, message: &MessageRef, text: &str) -> Result<(), ChannelError> {
        let body = json!({
            "content": format::truncate_chars(text, MAX_MESSAGE_LEN),
            "allowed_mentions": {"parse": []},
            "components": [],
        });
        self.rest
            .request(
                Method::PATCH,
                &format!(
                    "/channels/{}/messages/{}",
                    message.conversation_id, message.message_id
                ),
                Some(&body),
            )
            .await
            .map(|_| ())
    }

    async fn ask(&self, prompt: &ApprovalPrompt) -> Result<MessageRef, ChannelError> {
        let components = approval_components(prompt)?;
        let channel_id = &prompt.conversation.id;
        let (last, _) = self
            .send_leading_chunks(channel_id, &prompt.text, None)
            .await?;
        let mut body = message_body(&last, None);
        body["components"] = components;
        self.post_message(channel_id, &body).await
    }

    async fn typing(&self, conversation: &Conversation) -> Result<(), ChannelError> {
        self.rest
            .request(
                Method::POST,
                &format!("/channels/{}/typing", conversation.id),
                None,
            )
            .await
            .map(|_| ())
    }
}
