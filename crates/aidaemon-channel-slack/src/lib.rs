//! # aidaemon-channel-slack (experimental)
//!
//! A Slack [`Channel`] for `aidaemon-channels`, built on Socket Mode for
//! inbound events and the Web API for sending.
//!
//! The adapter is transport only: it maps Slack events to
//! [`ChannelEvent`]s and runner requests to Web API calls. Sessions,
//! authorization and approvals live in the runner.
//!
//! ```no_run
//! use aidaemon_channel_slack::SlackChannel;
//!
//! let slack = SlackChannel::new("xoxb-...", "xapp-...")
//!     .with_name("slack-work")
//!     .with_reply_in_thread(true);
//! # let _ = slack;
//! ```
//!
//! Conversation ids are the Slack channel id (`C...`, `D...`) or
//! `"{channel}:{thread_ts}"` for threads. Messages the bot is asked to
//! answer at the top level of a channel are answered in a new thread when
//! `reply_in_thread` is on (the default), so their conversation id is
//! `"{channel}:{ts}"`.

mod events;
pub mod format;

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Duration;

use aidaemon_channels::download::{DownloadClient, DownloadRequest};
use aidaemon_channels::split::truncate_for_edit;
use aidaemon_channels::{
    ApprovalPrompt, Attachment, Channel, ChannelCapabilities, ChannelError, ChannelEvent,
    CommandInvocation, Conversation, DisplayNameCache, MessageRef, OutboundFile, OutboundMessage,
    DEFAULT_MAX_DOWNLOAD_BYTES,
};
use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::events::{
    map_events_api, map_interactive, map_slash_command, split_conversation_id, ApprovalValue,
    MapContext, SeenMessages, APPROVAL_ACTION_PREFIX,
};
use crate::format::{markdown_to_mrkdwn, split_mrkdwn, truncate_chars};

/// Default Slack Web API base URL.
pub const DEFAULT_API_BASE: &str = "https://slack.com/api";
/// Longest message Slack accepts comfortably (reported in capabilities).
pub const MAX_MESSAGE_LEN: usize = 4000;
/// Longest text a Block Kit section accepts; outbound chunks never exceed it.
pub const SECTION_TEXT_LIMIT: usize = 3000;

const BUTTON_LABEL_LIMIT: usize = 75;
const BUTTON_VALUE_LIMIT: usize = 2000;
const MAX_BUTTONS: usize = 25;
const MAX_RETRY_WAIT_SECS: u64 = 30;
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(120);
/// Pressed prompts whose text is kept for [`Channel::remove_buttons`].
const PRESSED_PROMPTS_KEPT: usize = 256;
/// Slack pings Socket Mode clients regularly; silence this long means the
/// connection is dead.
const IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Slack errors that mean the token is unusable.
const AUTH_ERRORS: &[&str] = &[
    "invalid_auth",
    "not_authed",
    "account_inactive",
    "token_revoked",
    "token_expired",
    "not_allowed_token_type",
];

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Slack Socket Mode + Web API channel.
pub struct SlackChannel {
    bot_token: String,
    app_token: String,
    name: String,
    api_base: String,
    reply_in_thread: bool,
    reconnect_initial: Duration,
    reconnect_max: Duration,
    http: reqwest::Client,
    download_client: DownloadClient,
    bot_user_id: Mutex<Option<String>>,
    /// Longest chunk sent, at most [`SECTION_TEXT_LIMIT`].
    max_message_len: usize,
    /// The latest inbound message per conversation: (channel, ts). The
    /// typing substitute reacts to it.
    last_inbound: Mutex<HashMap<String, (String, String)>>,
    /// Messages wearing the typing reaction, by conversation.
    reacting: Mutex<HashMap<String, (String, String)>>,
    /// `(ts, text)` of recently pressed prompts. `chat.update` needs the
    /// text to remove a message's buttons, and a press carries it.
    pressed_prompts: Mutex<VecDeque<(String, String)>>,
    strip_latex: bool,
    names: DisplayNameCache,
}

impl std::fmt::Debug for SlackChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlackChannel")
            .field("name", &self.name)
            .field("api_base", &self.api_base)
            .field("reply_in_thread", &self.reply_in_thread)
            .finish_non_exhaustive()
    }
}

/// A Socket Mode envelope as Slack sent it, already acknowledged (see
/// [`SlackChannel::run_envelopes`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SlackEnvelope {
    /// `events_api`, `interactive`, `slash_commands`, ...
    pub kind: String,
    pub payload: Value,
}

/// Where a session delivers what it receives.
#[derive(Clone, Copy)]
enum Sink<'a> {
    /// Mapped to [`ChannelEvent`]s.
    Events(&'a mpsc::Sender<ChannelEvent>),
    /// As sent.
    Envelopes(&'a mpsc::Sender<SlackEnvelope>),
}

impl Sink<'_> {
    fn is_closed(&self) -> bool {
        match self {
            Self::Events(tx) => tx.is_closed(),
            Self::Envelopes(tx) => tx.is_closed(),
        }
    }

    async fn closed(&self) {
        match self {
            Self::Events(tx) => tx.closed().await,
            Self::Envelopes(tx) => tx.closed().await,
        }
    }
}

enum SessionEnd {
    /// Shutdown requested or the event receiver went away.
    Stop,
    /// Connection ended; `immediate` when Slack asked us to reconnect.
    Reconnect { immediate: bool, saw_hello: bool },
}

impl SlackChannel {
    /// `bot_token` (`xoxb-...`) is used for the Web API, `app_token`
    /// (`xapp-...`, with `connections:write`) for Socket Mode.
    pub fn new(bot_token: impl Into<String>, app_token: impl Into<String>) -> Self {
        Self {
            bot_token: bot_token.into(),
            app_token: app_token.into(),
            name: "slack".to_string(),
            api_base: DEFAULT_API_BASE.to_string(),
            reply_in_thread: true,
            reconnect_initial: Duration::from_secs(1),
            reconnect_max: Duration::from_secs(60),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            download_client: DownloadClient::default(),
            bot_user_id: Mutex::new(None),
            max_message_len: SECTION_TEXT_LIMIT,
            last_inbound: Mutex::new(HashMap::new()),
            reacting: Mutex::new(HashMap::new()),
            pressed_prompts: Mutex::new(VecDeque::new()),
            strip_latex: true,
            names: DisplayNameCache::default(),
        }
    }

    /// Whether [`aidaemon_channels::ChannelRunner`] rewrites LaTeX in the
    /// text it sends here as plain text (code excepted). Default on.
    pub fn with_latex_stripping(mut self, on: bool) -> Self {
        self.strip_latex = on;
        self
    }

    /// Split messages at `len` characters (1 to [`SECTION_TEXT_LIMIT`], the
    /// most one Block Kit section holds).
    pub fn with_max_message_len(mut self, len: usize) -> Self {
        self.max_message_len = len.clamp(1, SECTION_TEXT_LIMIT);
        self
    }

    /// The reaction the typing substitute shows.
    const TYPING_REACTION: &'static str = "hourglass_flowing_sand";

    /// Take the typing reaction off the conversation's message, if it has
    /// one. Best effort.
    async fn clear_typing(&self, conversation_id: &str) {
        let target = self
            .reacting
            .lock()
            .ok()
            .and_then(|mut reacting| reacting.remove(conversation_id));
        if let Some((channel, ts)) = target {
            let body =
                json!({ "channel": channel, "timestamp": ts, "name": Self::TYPING_REACTION });
            if let Err(e) = self.post_bot("reactions.remove", &body).await {
                debug!(error = %e, "could not remove the typing reaction");
            }
        }
    }

    /// Channel name used as the session-id prefix (default `slack`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Web API base URL (default `https://slack.com/api`). Mainly for tests.
    pub fn with_api_base(mut self, api_base: impl Into<String>) -> Self {
        self.api_base = api_base.into().trim_end_matches('/').to_string();
        self
    }

    /// The HTTP client for Web API calls (for example one that bypasses a
    /// proxy in tests).
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = client;
        self
    }

    /// HTTP settings (proxies, custom TLS roots, ...) for file downloads,
    /// which carry the bot token. A builder rather than a client, because
    /// the adapter must control how redirects are followed.
    pub fn with_download_client(mut self, builder: reqwest::ClientBuilder) -> Self {
        self.download_client = DownloadClient::from_builder(builder);
        self
    }

    /// The approval answer in an `interactive` envelope payload, for a
    /// host that reads envelopes itself ([`SlackChannel::run_envelopes`]).
    /// `None` unless it is a press on a button made by `ask`.
    pub fn approval_answer(payload: &Value) -> Option<aidaemon_channels::ApprovalAnswer> {
        match events::map_interactive(payload)? {
            ChannelEvent::Approval(answer) => Some(answer),
            _ => None,
        }
    }

    /// Answer addressed top-level channel messages in a new thread
    /// (default true).
    pub fn with_reply_in_thread(mut self, reply_in_thread: bool) -> Self {
        self.reply_in_thread = reply_in_thread;
        self
    }

    /// Reconnect backoff: first delay and cap (defaults 1s and 60s).
    pub fn with_reconnect_backoff(mut self, initial: Duration, max: Duration) -> Self {
        self.reconnect_initial = initial;
        self.reconnect_max = max.max(initial);
        self
    }

    /// The bot's own user id, once `run` has resolved it via `auth.test`.
    pub fn bot_user_id(&self) -> Option<String> {
        self.bot_user_id.lock().ok().and_then(|g| g.clone())
    }

    /// Call a Web API method with a JSON body. Retries once after a
    /// `ratelimited` response, honouring `Retry-After` (capped).
    async fn call(
        &self,
        method: &str,
        token: &str,
        body: Option<&Value>,
    ) -> Result<Value, ChannelError> {
        self.call_with(method, token, |request| match body {
            Some(body) => request.json(body),
            None => request.header("Content-Type", "application/x-www-form-urlencoded"),
        })
        .await
    }

    /// A Web API call with form-encoded fields (methods that take no JSON).
    async fn call_form(
        &self,
        method: &str,
        fields: &[(&str, String)],
    ) -> Result<Value, ChannelError> {
        self.call_with(method, &self.bot_token, |request| request.form(fields))
            .await
    }

    /// Whether `url` is Slack's own (or this channel's configured API
    /// host): the bot token and command answers go nowhere else.
    fn slack_url(&self, url: &str) -> bool {
        reqwest::Url::parse(url).is_ok_and(|parsed| self.slack_host(&parsed))
    }

    fn slack_host(&self, parsed: &reqwest::Url) -> bool {
        let Some(host) = parsed.host_str() else {
            return false;
        };
        let configured = reqwest::Url::parse(&self.api_base)
            .ok()
            .and_then(|base| base.host_str().map(str::to_string));
        (parsed.scheme() == "https" && (host == "slack.com" || host.ends_with(".slack.com")))
            || configured.as_deref() == Some(host)
    }

    async fn call_with(
        &self,
        method: &str,
        token: &str,
        body: impl Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder,
    ) -> Result<Value, ChannelError> {
        let url = format!("{}/{}", self.api_base, method);
        for attempt in 0..2 {
            let request = body(self.http.post(&url).bearer_auth(token));
            let response = request
                .send()
                .await
                .map_err(|e| ChannelError::Transport(format!("{method}: {e}")))?;
            let status = response.status();
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<u64>().ok());
            let text = response
                .text()
                .await
                .map_err(|e| ChannelError::Transport(format!("{method}: {e}")))?;
            let parsed: Option<Value> = serde_json::from_str(&text).ok();
            let error = parsed
                .as_ref()
                .and_then(|v| v.get("error"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let error = error.as_deref();

            if status == reqwest::StatusCode::TOO_MANY_REQUESTS || error == Some("ratelimited") {
                if attempt == 0 {
                    let wait = retry_after.unwrap_or(1).min(MAX_RETRY_WAIT_SECS);
                    warn!(method, wait, "Slack rate limited; retrying once");
                    tokio::time::sleep(Duration::from_secs(wait)).await;
                    continue;
                }
                return Err(ChannelError::RateLimited {
                    retry_after_secs: retry_after,
                });
            }

            let Some(value) = parsed.as_ref() else {
                return Err(ChannelError::Transport(format!(
                    "{method}: HTTP {status} with non-JSON body"
                )));
            };
            if value.get("ok").and_then(Value::as_bool) == Some(true) {
                return Ok(value.clone());
            }
            let error = error.unwrap_or("unknown_error");
            if AUTH_ERRORS.contains(&error) {
                return Err(ChannelError::Auth(format!("{method}: {error}")));
            }
            if status.is_server_error() {
                return Err(ChannelError::Transport(format!(
                    "{method}: HTTP {status} ({error})"
                )));
            }
            return Err(ChannelError::Other(format!("{method}: {error}")));
        }
        unreachable!("the second attempt always returns")
    }

    async fn post_bot(&self, method: &str, body: &Value) -> Result<Value, ChannelError> {
        self.call(method, &self.bot_token, Some(body)).await
    }

    async fn ensure_bot_user_id(&self) -> Result<String, ChannelError> {
        if let Some(id) = self.bot_user_id() {
            return Ok(id);
        }
        let resp = self.call("auth.test", &self.bot_token, None).await?;
        let id = resp
            .get("user_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ChannelError::Other("auth.test: no user_id".into()))?
            .to_string();
        info!(bot_user_id = %id, "resolved Slack bot user");
        if let Ok(mut guard) = self.bot_user_id.lock() {
            *guard = Some(id.clone());
        }
        Ok(id)
    }

    async fn open_socket(&self) -> Result<WsStream, ChannelError> {
        let resp = self
            .call("apps.connections.open", &self.app_token, None)
            .await?;
        let url = resp
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| ChannelError::Other("apps.connections.open: no url".into()))?;
        // The URL carries a short-lived ticket; never log it.
        let (ws, _) = tokio_tungstenite::connect_async(url)
            .await
            .map_err(|e| ChannelError::Transport(format!("websocket connect: {e}")))?;
        Ok(ws)
    }

    async fn session(
        &self,
        ws: WsStream,
        bot_user_id: &str,
        sink: Sink<'_>,
        shutdown: &CancellationToken,
        seen: &mut SeenMessages,
    ) -> SessionEnd {
        let (mut tx, mut rx) = ws.split();
        let mut saw_hello = false;
        let reconnect = |immediate, saw_hello| SessionEnd::Reconnect {
            immediate,
            saw_hello,
        };

        loop {
            let frame = tokio::select! {
                _ = shutdown.cancelled() => {
                    let _ = tx.send(Message::Close(None)).await;
                    return SessionEnd::Stop;
                }
                _ = sink.closed() => {
                    let _ = tx.send(Message::Close(None)).await;
                    return SessionEnd::Stop;
                }
                frame = tokio::time::timeout(IDLE_TIMEOUT, rx.next()) => frame,
            };
            let message = match frame {
                Err(_) => {
                    warn!("Slack socket idle; reconnecting");
                    return reconnect(false, saw_hello);
                }
                Ok(None) => return reconnect(false, saw_hello),
                Ok(Some(Err(e))) => {
                    warn!(error = %e, "Slack socket read error");
                    return reconnect(false, saw_hello);
                }
                Ok(Some(Ok(message))) => message,
            };

            let text = match message {
                Message::Text(text) => text,
                Message::Ping(data) => {
                    let _ = tx.send(Message::Pong(data)).await;
                    continue;
                }
                Message::Close(_) => return reconnect(false, saw_hello),
                _ => continue,
            };

            let envelope: Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(e) => {
                    warn!(error = %e, "unparseable Slack envelope");
                    continue;
                }
            };

            // Ack first: Slack redelivers envelopes not acked within 3s.
            if let Some(id) = envelope.get("envelope_id").and_then(Value::as_str) {
                let ack = json!({ "envelope_id": id }).to_string();
                if let Err(e) = tx.send(Message::Text(ack)).await {
                    warn!(error = %e, "failed to ack Slack envelope");
                    return reconnect(false, saw_hello);
                }
            }

            let payload = envelope.get("payload").unwrap_or(&Value::Null);
            let kind = envelope.get("type").and_then(Value::as_str);
            if let (Sink::Envelopes(envelopes), Some(kind)) = (sink, kind) {
                if !matches!(kind, "hello" | "disconnect") {
                    let envelope = SlackEnvelope {
                        kind: kind.to_string(),
                        payload: payload.clone(),
                    };
                    tokio::select! {
                        _ = shutdown.cancelled() => return SessionEnd::Stop,
                        sent = envelopes.send(envelope) => {
                            if sent.is_err() {
                                return SessionEnd::Stop;
                            }
                        }
                    }
                    continue;
                }
            }
            let Sink::Events(events) = sink else {
                // hello and disconnect, handled below.
                match kind {
                    Some("hello") => saw_hello = true,
                    Some("disconnect") => {
                        let _ = tx.send(Message::Close(None)).await;
                        return reconnect(true, saw_hello);
                    }
                    _ => {}
                }
                continue;
            };
            let event = match kind {
                Some("hello") => {
                    debug!("Slack socket hello");
                    saw_hello = true;
                    None
                }
                Some("disconnect") => {
                    let reason = envelope
                        .get("reason")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    info!(reason, "Slack requested reconnect");
                    let _ = tx.send(Message::Close(None)).await;
                    return reconnect(true, saw_hello);
                }
                Some("events_api") => {
                    let ctx = MapContext {
                        bot_user_id: Some(bot_user_id),
                        reply_in_thread: self.reply_in_thread,
                    };
                    map_events_api(payload, &ctx, seen)
                }
                Some("interactive") => {
                    let event = map_interactive(payload);
                    self.remember_pressed(payload, &event);
                    event
                }
                Some("slash_commands") => map_slash_command(payload),
                other => {
                    debug!(envelope_type = ?other, "ignoring Slack envelope");
                    None
                }
            };

            if let Some(ChannelEvent::Message(message)) = &event {
                let (channel, _) = split_conversation_id(&message.conversation.id);
                if let Ok(mut last) = self.last_inbound.lock() {
                    last.insert(
                        message.conversation.id.clone(),
                        (channel.to_string(), message.message_id.clone()),
                    );
                }
            }
            if let Some(mut event) = event {
                // Slack's events carry no name. The envelope is already
                // acknowledged and the lookup is cached and time-limited.
                self.names
                    .prepare_event(&mut event, |id| async move { self.fetch_name(&id).await })
                    .await;
                tokio::select! {
                    _ = shutdown.cancelled() => return SessionEnd::Stop,
                    sent = events.send(event) => {
                        if sent.is_err() {
                            return SessionEnd::Stop;
                        }
                    }
                }
            }
        }
    }

    async fn connect_and_serve(
        &self,
        sink: Sink<'_>,
        shutdown: &CancellationToken,
        seen: &mut SeenMessages,
    ) -> Result<SessionEnd, ChannelError> {
        let bot_user_id = self.ensure_bot_user_id().await?;
        let ws = self.open_socket().await?;
        info!(channel = %self.name, "Slack socket connected");
        Ok(self.session(ws, &bot_user_id, sink, shutdown, seen).await)
    }

    async fn fetch_name(&self, user_id: &str) -> Result<Option<String>, ChannelError> {
        let info = self
            .call_form("users.info", &[("user", user_id.to_string())])
            .await?;
        let profile = info.get("user").and_then(|user| user.get("profile"));
        Ok(["display_name", "real_name"]
            .iter()
            .find_map(|key| {
                profile
                    .and_then(|p| p.get(*key))
                    .and_then(Value::as_str)
                    .filter(|name| !name.trim().is_empty())
            })
            .map(str::to_string))
    }

    /// Hold the Socket Mode connection (acknowledging every envelope,
    /// answering pings, following `disconnect`, reconnecting with backoff)
    /// and hand each envelope over as Slack sent it, for a host that reads
    /// Slack's own event shapes. Like [`Channel::run`] otherwise.
    pub async fn run_envelopes(
        &self,
        envelopes: mpsc::Sender<SlackEnvelope>,
        shutdown: CancellationToken,
    ) -> Result<(), ChannelError> {
        self.serve(Sink::Envelopes(&envelopes), shutdown).await
    }

    async fn serve(&self, sink: Sink<'_>, shutdown: CancellationToken) -> Result<(), ChannelError> {
        let mut seen = SeenMessages::default();
        let mut backoff = self.reconnect_initial;

        loop {
            if shutdown.is_cancelled() || sink.is_closed() {
                return Ok(());
            }
            let outcome = tokio::select! {
                _ = shutdown.cancelled() => return Ok(()),
                outcome = self.connect_and_serve(sink, &shutdown, &mut seen) => outcome,
            };
            match outcome {
                Ok(SessionEnd::Stop) => return Ok(()),
                Ok(SessionEnd::Reconnect {
                    immediate,
                    saw_hello,
                }) => {
                    if saw_hello {
                        backoff = self.reconnect_initial;
                    }
                    if immediate {
                        continue;
                    }
                }
                Err(ChannelError::Auth(reason)) => return Err(ChannelError::Auth(reason)),
                Err(e) => warn!(error = %e, "Slack connection failed"),
            }

            debug!(
                delay_ms = backoff.as_millis() as u64,
                "Slack reconnect backoff"
            );
            tokio::select! {
                _ = shutdown.cancelled() => return Ok(()),
                _ = tokio::time::sleep(backoff) => {}
            }
            backoff = (backoff * 2).min(self.reconnect_max);
        }
    }

    /// Keep a pressed prompt's text until its buttons are removed.
    fn remember_pressed(&self, payload: &Value, event: &Option<ChannelEvent>) {
        let Some(ChannelEvent::Approval(answer)) = event else {
            return;
        };
        let (Some(message), Some(text)) = (
            answer.message.as_ref(),
            payload
                .get("message")
                .and_then(|m| m.get("text"))
                .and_then(Value::as_str),
        ) else {
            return;
        };
        if let Ok(mut pressed) = self.pressed_prompts.lock() {
            if pressed.len() >= PRESSED_PROMPTS_KEPT {
                pressed.pop_front();
            }
            pressed.push_back((message.message_id.clone(), text.to_string()));
        }
    }

    fn section(text: &str) -> Value {
        json!({ "type": "section", "text": { "type": "mrkdwn", "text": text } })
    }
}

#[async_trait]
impl Channel for SlackChannel {
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
        self.serve(Sink::Events(&events), shutdown).await
    }

    async fn send(
        &self,
        conversation: &Conversation,
        message: &OutboundMessage,
    ) -> Result<MessageRef, ChannelError> {
        let (channel, thread_ts) = split_conversation_id(&conversation.id);
        let mrkdwn = markdown_to_mrkdwn(&message.text);
        if mrkdwn.trim().is_empty() {
            return Err(ChannelError::Other("cannot send an empty message".into()));
        }

        let mut last_ts = None;
        self.clear_typing(&conversation.id).await;
        for chunk in split_mrkdwn(&mrkdwn, self.max_message_len) {
            let mut body = json!({
                "channel": channel,
                "text": chunk,
                "blocks": [Self::section(&chunk)],
            });
            if let Some(ts) = thread_ts {
                body["thread_ts"] = json!(ts);
            }
            let resp = self.post_bot("chat.postMessage", &body).await?;
            last_ts = resp.get("ts").and_then(Value::as_str).map(str::to_string);
        }

        Ok(MessageRef {
            conversation_id: conversation.id.clone(),
            message_id: last_ts
                .ok_or_else(|| ChannelError::Other("chat.postMessage: no ts".into()))?,
        })
    }

    async fn edit(&self, message: &MessageRef, text: &str) -> Result<(), ChannelError> {
        let (channel, _) = split_conversation_id(&message.conversation_id);
        let mrkdwn = truncate_for_edit(&markdown_to_mrkdwn(text), self.max_message_len);
        let body = json!({
            "channel": channel,
            "ts": message.message_id,
            "text": mrkdwn,
            "blocks": [Self::section(&mrkdwn)],
        });
        self.post_bot("chat.update", &body).await.map(|_| ())
    }

    async fn ask(&self, prompt: &ApprovalPrompt) -> Result<MessageRef, ChannelError> {
        if prompt.choices.is_empty() {
            return Err(ChannelError::Other("approval prompt has no choices".into()));
        }
        let (channel, thread_ts) = split_conversation_id(&prompt.conversation.id);
        self.clear_typing(&prompt.conversation.id).await;
        let mrkdwn = truncate_chars(&markdown_to_mrkdwn(&prompt.text), SECTION_TEXT_LIMIT);

        let mut buttons = Vec::new();
        for (index, choice) in prompt.choices.iter().take(MAX_BUTTONS).enumerate() {
            let value = serde_json::to_string(&ApprovalValue {
                prompt_id: prompt.id.clone(),
                choice_id: choice.id.clone(),
            })
            .map_err(|e| ChannelError::Other(e.to_string()))?;
            if value.chars().count() > BUTTON_VALUE_LIMIT {
                return Err(ChannelError::Other(
                    "approval prompt/choice ids too long for a Slack button".into(),
                ));
            }
            let mut button = json!({
                "type": "button",
                "action_id": format!("{APPROVAL_ACTION_PREFIX}_{index}"),
                "text": {
                    "type": "plain_text",
                    "text": truncate_chars(&choice.label, BUTTON_LABEL_LIMIT),
                    "emoji": true,
                },
                "value": value,
            });
            if index == 0 {
                button["style"] = json!("primary");
            }
            buttons.push(button);
        }

        let mut body = json!({
            "channel": channel,
            "text": mrkdwn,
            "blocks": [
                Self::section(&mrkdwn),
                { "type": "actions", "block_id": APPROVAL_ACTION_PREFIX, "elements": buttons },
            ],
        });
        if let Some(ts) = thread_ts {
            body["thread_ts"] = json!(ts);
        }
        let resp = self.post_bot("chat.postMessage", &body).await?;
        let ts = resp
            .get("ts")
            .and_then(Value::as_str)
            .ok_or_else(|| ChannelError::Other("chat.postMessage: no ts".into()))?;
        Ok(MessageRef {
            conversation_id: prompt.conversation.id.clone(),
            message_id: ts.to_string(),
        })
    }

    /// `chat.update` with only the prompt's text. Works on a prompt whose
    /// button was pressed since this process started; others are refused,
    /// as Slack cannot drop buttons without being sent the text again.
    async fn remove_buttons(&self, message: &MessageRef) -> Result<(), ChannelError> {
        let text = self.pressed_prompts.lock().ok().and_then(|mut pressed| {
            let at = pressed
                .iter()
                .position(|(ts, _)| *ts == message.message_id)?;
            pressed.remove(at).map(|(_, text)| text)
        });
        let Some(text) = text else {
            return Err(ChannelError::Other(
                "slack: this prompt's text is not known".into(),
            ));
        };
        let (channel, _) = split_conversation_id(&message.conversation_id);
        let body = json!({
            "channel": channel,
            "ts": message.message_id,
            "text": text,
            "blocks": [Self::section(&text)],
        });
        self.post_bot("chat.update", &body).await.map(|_| ())
    }

    /// Slack has no typing indicator for bots: the person's latest message
    /// gets an hourglass reaction until the next reply.
    async fn typing(&self, conversation: &Conversation) -> Result<(), ChannelError> {
        let target = {
            let last = self
                .last_inbound
                .lock()
                .ok()
                .and_then(|last| last.get(&conversation.id).cloned());
            let Some(target) = last else {
                return Ok(());
            };
            let Ok(mut reacting) = self.reacting.lock() else {
                return Ok(());
            };
            if reacting.get(&conversation.id) == Some(&target) {
                return Ok(());
            }
            reacting.insert(conversation.id.clone(), target.clone());
            target
        };
        let body =
            json!({ "channel": target.0, "timestamp": target.1, "name": Self::TYPING_REACTION });
        match self.post_bot("reactions.add", &body).await {
            Ok(_) => Ok(()),
            Err(ChannelError::Other(reason)) if reason.contains("already_reacted") => Ok(()),
            Err(e) => {
                if let Ok(mut reacting) = self.reacting.lock() {
                    reacting.remove(&conversation.id);
                }
                Err(e)
            }
        }
    }

    fn strips_latex(&self) -> bool {
        self.strip_latex
    }

    /// Cached; the name comes from `users.info`.
    async fn display_name(&self, user_id: &str) -> Result<Option<String>, ChannelError> {
        self.names
            .lookup(user_id, || self.fetch_name(user_id))
            .await
    }

    async fn delete(&self, message: &MessageRef) -> Result<(), ChannelError> {
        let (channel, _) = split_conversation_id(&message.conversation_id);
        let body = json!({ "channel": channel, "ts": message.message_id });
        self.post_bot("chat.delete", &body).await.map(|_| ())
    }

    /// Slack's external upload: ask for an upload URL, send the bytes
    /// there, then share the file in the conversation with its caption.
    /// The returned ref's `message_id` is the Slack file id.
    async fn send_file(
        &self,
        conversation: &Conversation,
        file: &OutboundFile,
    ) -> Result<MessageRef, ChannelError> {
        let (channel, thread_ts) = split_conversation_id(&conversation.id);
        self.clear_typing(&conversation.id).await;
        let ticket = self
            .call_form(
                "files.getUploadURLExternal",
                &[
                    ("filename", file.name.clone()),
                    ("length", file.bytes.len().to_string()),
                ],
            )
            .await?;
        let upload_url = ticket
            .get("upload_url")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ChannelError::Other("files.getUploadURLExternal: no upload_url".into())
            })?;
        let file_id = ticket
            .get("file_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ChannelError::Other("files.getUploadURLExternal: no file_id".into()))?
            .to_string();
        if !self.slack_url(upload_url) {
            return Err(ChannelError::Other(
                "files.getUploadURLExternal: upload URL is not a Slack host".into(),
            ));
        }
        let uploaded = self
            .http
            .post(upload_url)
            .header("Content-Type", "application/octet-stream")
            .body(file.bytes.clone())
            .send()
            .await
            .map_err(|e| ChannelError::Transport(format!("file upload: {e}")))?;
        if !uploaded.status().is_success() {
            return Err(ChannelError::Transport(format!(
                "file upload: HTTP {}",
                uploaded.status()
            )));
        }
        let mut complete = json!({
            "files": [{ "id": file_id, "title": file.name }],
            "channel_id": channel,
        });
        if let Some(ts) = thread_ts {
            complete["thread_ts"] = json!(ts);
        }
        if let Some(caption) = file.caption.as_deref().filter(|c| !c.trim().is_empty()) {
            complete["initial_comment"] = json!(markdown_to_mrkdwn(caption));
        }
        self.post_bot("files.completeUploadExternal", &complete)
            .await?;
        Ok(MessageRef {
            conversation_id: conversation.id.clone(),
            message_id: file_id,
        })
    }

    async fn download(&self, attachment: &Attachment) -> Result<Vec<u8>, ChannelError> {
        self.download_limited(attachment, DEFAULT_MAX_DOWNLOAD_BYTES)
            .await
    }

    /// Slack file URLs need the bot token; it is only sent to Slack hosts,
    /// on the first request and on any redirect.
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
        let downloaded = self
            .download_client
            .fetch(DownloadRequest {
                url: &attachment.source,
                bearer_token: Some(&self.bot_token),
                allowed: &|target| self.slack_host(target),
                max_bytes,
                timeout: DOWNLOAD_TIMEOUT,
                user_agent: None,
            })
            .await?;
        // Without the `files:read` scope Slack answers 200 with its sign-in
        // page, which must not pass for the file.
        let is_html = |mime: &str| mime.trim_start().starts_with("text/html");
        if downloaded.content_type.as_deref().is_some_and(is_html)
            && !attachment.mime_type.as_deref().is_some_and(is_html)
        {
            return Err(ChannelError::Auth(
                "slack served a sign-in page instead of the file; the bot needs files:read"
                    .to_string(),
            ));
        }
        Ok(downloaded.bytes)
    }

    /// Slack's slash commands are declared in the app manifest; there is
    /// no API to change them.
    async fn register_commands(
        &self,
        _commands: &[aidaemon_channels::CommandSpec],
    ) -> Result<(), ChannelError> {
        Ok(())
    }

    /// Answer through the command's `response_url`, visible only to the
    /// person who ran it.
    async fn answer_command(
        &self,
        command: &CommandInvocation,
        text: &str,
    ) -> Result<(), ChannelError> {
        if !self.slack_url(&command.handle) {
            return Err(ChannelError::Other(
                "slack: command response URL is not a Slack host".into(),
            ));
        }
        let body = json!({
            "response_type": "ephemeral",
            "text": truncate_chars(&markdown_to_mrkdwn(text), MAX_MESSAGE_LEN),
        });
        let response = self
            .http
            .post(&command.handle)
            .json(&body)
            .send()
            .await
            .map_err(|e| ChannelError::Transport(format!("command response: {e}")))?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(ChannelError::Other(format!(
                "command response: HTTP {}",
                response.status()
            )))
        }
    }
}

/// Check a bot token with `auth.test`; returns `(bot_name, team_name)`.
pub async fn validate_tokens(
    bot_token: &str,
    _app_token: &str,
) -> Result<(String, String), String> {
    let client = reqwest::Client::new();
    let response = client
        .get("https://slack.com/api/auth.test")
        .header("Authorization", format!("Bearer {}", bot_token))
        .send()
        .await
        .map_err(|e| format!("Failed to validate Slack token: {}", e))?;

    let json: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Slack response: {}", e))?;

    if json["ok"].as_bool() != Some(true) {
        return Err(format!(
            "Invalid Slack token: {}\n\nMake sure you have the correct bot token.",
            json["error"].as_str().unwrap_or("unknown error")
        ));
    }

    Ok((
        json["user"].as_str().unwrap_or("unknown").to_string(),
        json["team"].as_str().unwrap_or("unknown").to_string(),
    ))
}
