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
//!
//! # Sharding and compression
//!
//! One connection, with no `shard` field, is the default. A large bot uses
//! [`DiscordChannel::with_shard_count`] or
//! [`DiscordChannel::with_recommended_shards`]: one connection per shard,
//! each with its own session and resume, and identifies paced to the limits
//! `GET /gateway/bot` reports. With the `zlib` feature,
//! [`DiscordChannel::with_zlib_compression`] turns on `zlib-stream` transport
//! compression.

mod format;
mod gateway;
mod mapping;
mod rest;
#[cfg(feature = "zlib")]
mod zlib;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_channels::download::DownloadClient;
use aidaemon_channels::split::truncate_for_edit;
use aidaemon_channels::{
    ApprovalPrompt, Attachment, Channel, ChannelCapabilities, ChannelError, ChannelEvent,
    CommandInvocation, CommandSpec, Conversation, DisplayNameCache, MessageRef, OutboundFile,
    OutboundMessage, DEFAULT_MAX_DOWNLOAD_BYTES,
};
use async_trait::async_trait;
use reqwest::Method;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub use format::split_message;
pub use mapping::{decode_custom_id, encode_custom_id, CUSTOM_ID_MAX};

/// Discord's limit for a file sent by a bot without boosts, in bytes.
pub const MAX_FILE_BYTES: usize = 10 * 1024 * 1024;

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
    /// Longest message sent, at most [`MAX_MESSAGE_LEN`].
    max_message_len: usize,
    strip_latex: bool,
    names: DisplayNameCache,
    shards: gateway::ShardMode,
    identify_interval: Duration,
    #[cfg(feature = "zlib")]
    zlib: bool,
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
                files: DownloadClient::default(),
            },
            gateway_url: Mutex::new(None),
            intents: intents::DEFAULT,
            backoff_initial: Duration::from_secs(1),
            backoff_max: Duration::from_secs(60),
            state: Arc::new(Mutex::new(gateway::GatewayState::default())),
            max_message_len: MAX_MESSAGE_LEN,
            strip_latex: true,
            names: DisplayNameCache::default(),
            shards: gateway::ShardMode::Single,
            identify_interval: Duration::from_secs(5),
            #[cfg(feature = "zlib")]
            zlib: false,
        }
    }

    /// Whether [`aidaemon_channels::ChannelRunner`] rewrites LaTeX in the
    /// text it sends here as plain text (code excepted). Default on.
    pub fn with_latex_stripping(mut self, on: bool) -> Self {
        self.strip_latex = on;
        self
    }

    /// Run `count` shards (at least 1), one connection each.
    pub fn with_shard_count(mut self, count: u32) -> Self {
        self.shards = gateway::ShardMode::Fixed(count.max(1));
        self
    }

    /// Run as many shards as `GET /gateway/bot` recommends when `run` starts.
    pub fn with_recommended_shards(mut self) -> Self {
        self.shards = gateway::ShardMode::Recommended;
        self
    }

    /// The wait between identifies in one rate-limit bucket. Default 5 s,
    /// Discord's rule; only a local test server has reason to change it.
    pub fn with_identify_interval(mut self, interval: Duration) -> Self {
        self.identify_interval = interval;
        self
    }

    /// Ask for `zlib-stream` transport compression. Default off.
    #[cfg(feature = "zlib")]
    pub fn with_zlib_compression(mut self, on: bool) -> Self {
        self.zlib = on;
        self
    }

    /// Split messages at `len` characters (1 to [`MAX_MESSAGE_LEN`]).
    pub fn with_max_message_len(mut self, len: usize) -> Self {
        self.max_message_len = len.clamp(1, MAX_MESSAGE_LEN);
        self
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

    /// HTTP settings (proxies, custom TLS roots, ...) for file downloads.
    /// A builder rather than a client, because the adapter must control how
    /// redirects are followed.
    pub fn with_download_client(mut self, builder: reqwest::ClientBuilder) -> Self {
        self.rest.files = DownloadClient::from_builder(builder);
        self
    }

    /// How many shards the last `run` started with (`None` before one).
    pub fn shard_count(&self) -> Option<u32> {
        match self.state.lock().unwrap().shards.len() {
            0 => None,
            count => u32::try_from(count).ok(),
        }
    }

    pub(crate) async fn fetch_name(&self, user_id: &str) -> Result<Option<String>, ChannelError> {
        let user = self
            .rest
            .request(Method::GET, &format!("/users/{user_id}"), None)
            .await?;
        Ok(["global_name", "username"]
            .iter()
            .find_map(|key| user.get(*key).and_then(Value::as_str))
            .filter(|name| !name.is_empty())
            .map(str::to_string))
    }

    /// The bot's user id, known after the gateway's READY event.
    pub fn bot_user_id(&self) -> Option<String> {
        self.state.lock().unwrap().bot_user_id.clone()
    }

    /// The bot's username, known after the gateway's READY event.
    pub fn bot_username(&self) -> Option<String> {
        self.state.lock().unwrap().bot_username.clone()
    }

    /// Register the command menu in this guild (it shows at once) instead
    /// of globally (which can take an hour to appear).
    pub fn with_command_guild(self, guild_id: impl Into<String>) -> Self {
        self.state.lock().unwrap().command_guild = Some(guild_id.into());
        self
    }

    /// The direct conversation with a user, opened if needed
    /// (`POST /users/@me/channels`).
    pub async fn open_direct(&self, user_id: &str) -> Result<Conversation, ChannelError> {
        let channel = self
            .rest
            .request(
                Method::POST,
                "/users/@me/channels",
                Some(&json!({ "recipient_id": user_id })),
            )
            .await?;
        let id = channel
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| ChannelError::Other("discord: DM channel without id".into()))?;
        Ok(Conversation {
            id: id.to_string(),
            kind: aidaemon_channels::ConversationKind::Direct,
        })
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
        let mut chunks = split_message(text, self.max_message_len);
        let last = chunks.pop().unwrap_or_default();
        let mut reply_to = reply_to.map(str::to_string);
        for chunk in chunks {
            let body = message_body(&chunk, reply_to.take().as_deref());
            self.post_message(channel_id, &body).await?;
        }
        Ok((last, reply_to))
    }
}

/// Replace the application's global command menu with `commands`. Each
/// command takes one optional free-text `args` option.
pub(crate) async fn put_commands(
    rest: &rest::Rest,
    application_id: &str,
    guild_id: Option<&str>,
    commands: &[CommandSpec],
) -> Result<(), ChannelError> {
    let body: Vec<Value> = commands
        .iter()
        .map(|command| {
            json!({
                "type": 1,
                "name": command.name,
                "description": format::truncate_chars(&command.description, 100),
                "options": [{
                    "type": 3,
                    "name": "args",
                    "description": "Arguments",
                    "required": false,
                }],
            })
        })
        .collect();
    let path = match guild_id {
        Some(guild) => format!("/applications/{application_id}/guilds/{guild}/commands"),
        None => format!("/applications/{application_id}/commands"),
    };
    rest.request(Method::PUT, &path, Some(&Value::Array(body)))
        .await
        .map(|_| ())
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
        // By position: first = success (green), last = danger (red, the
        // refusal), any in between = primary (blue).
        let last = prompt.choices.len() - 1;
        let style = match i {
            0 => 3,
            i if i == last => 4,
            _ => 1,
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
            delete_messages: true,
            files: true,
            commands: true,
            max_message_len: self.max_message_len,
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
    /// approval prompt cannot be pressed again).
    async fn edit(&self, message: &MessageRef, text: &str) -> Result<(), ChannelError> {
        let body = json!({
            "content": truncate_for_edit(text, self.max_message_len),
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

    async fn remove_buttons(&self, message: &MessageRef) -> Result<(), ChannelError> {
        self.rest
            .request(
                Method::PATCH,
                &format!(
                    "/channels/{}/messages/{}",
                    message.conversation_id, message.message_id
                ),
                Some(&json!({ "components": [] })),
            )
            .await
            .map(|_| ())
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

    async fn delete(&self, message: &MessageRef) -> Result<(), ChannelError> {
        self.rest
            .request(
                Method::DELETE,
                &format!(
                    "/channels/{}/messages/{}",
                    message.conversation_id, message.message_id
                ),
                None,
            )
            .await
            .map(|_| ())
    }

    async fn send_file(
        &self,
        conversation: &Conversation,
        file: &OutboundFile,
    ) -> Result<MessageRef, ChannelError> {
        if file.bytes.len() > MAX_FILE_BYTES {
            return Err(ChannelError::Other(format!(
                "discord: {} is {} bytes; bots may send at most {MAX_FILE_BYTES}",
                file.name,
                file.bytes.len()
            )));
        }
        let caption = file
            .caption
            .as_deref()
            .map(|caption| format::truncate_chars(caption, self.max_message_len))
            .unwrap_or_default();
        let payload = json!({
            "content": caption,
            "allowed_mentions": {"parse": []},
        });
        let channel_id = &conversation.id;
        let resp = self
            .rest
            .upload(
                &format!("/channels/{channel_id}/messages"),
                &payload,
                &file.name,
                &file.bytes,
                file.mime_type.as_deref(),
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

    async fn download(&self, attachment: &Attachment) -> Result<Vec<u8>, ChannelError> {
        self.download_limited(attachment, DEFAULT_MAX_DOWNLOAD_BYTES)
            .await
    }

    async fn download_limited(
        &self,
        attachment: &Attachment,
        max_bytes: u64,
    ) -> Result<Vec<u8>, ChannelError> {
        if attachment.size_bytes.is_some_and(|size| size > max_bytes) {
            return Err(ChannelError::TooLarge {
                limit_bytes: max_bytes,
            });
        }
        self.rest.download(&attachment.source, max_bytes).await
    }

    fn strips_latex(&self) -> bool {
        self.strip_latex
    }

    /// Cached; the name comes from `GET /users/{id}`.
    async fn display_name(&self, user_id: &str) -> Result<Option<String>, ChannelError> {
        self.names
            .lookup(user_id, || self.fetch_name(user_id))
            .await
    }

    /// Registered now when the gateway is ready, else on its READY.
    async fn register_commands(&self, commands: &[CommandSpec]) -> Result<(), ChannelError> {
        let (application_id, guild) = {
            let mut st = self.state.lock().unwrap();
            st.commands = Some(commands.to_vec());
            (st.application_id.clone(), st.command_guild.clone())
        };
        match application_id {
            Some(app) => put_commands(&self.rest, &app, guild.as_deref(), commands).await,
            None => Ok(()),
        }
    }

    /// Fills in the deferred reply the adapter acknowledged the invocation
    /// with.
    async fn answer_command(
        &self,
        command: &CommandInvocation,
        text: &str,
    ) -> Result<(), ChannelError> {
        let Some((application_id, token)) = command.handle.split_once('/') else {
            return Err(ChannelError::Other(
                "discord: not a command handle from this adapter".into(),
            ));
        };
        let body = json!({
            "content": format::truncate_chars(text, self.max_message_len),
            "allowed_mentions": {"parse": []},
        });
        self.rest
            .request(
                Method::PATCH,
                &format!("/webhooks/{application_id}/{token}/messages/@original"),
                Some(&body),
            )
            .await
            .map(|_| ())
    }
}
