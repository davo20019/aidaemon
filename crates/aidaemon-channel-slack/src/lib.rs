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

use std::sync::Mutex;
use std::time::Duration;

use aidaemon_channels::{
    ApprovalPrompt, Channel, ChannelCapabilities, ChannelError, ChannelEvent, Conversation,
    MessageRef, OutboundMessage,
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
    map_events_api, map_interactive, split_conversation_id, ApprovalValue, MapContext,
    SeenMessages, APPROVAL_ACTION_PREFIX,
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
    bot_user_id: Mutex<Option<String>>,
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
            bot_user_id: Mutex::new(None),
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
        let url = format!("{}/{}", self.api_base, method);
        for attempt in 0..2 {
            let mut request = self.http.post(&url).bearer_auth(token);
            request = match body {
                Some(body) => request.json(body),
                None => request.header("Content-Type", "application/x-www-form-urlencoded"),
            };
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
        events: &mpsc::Sender<ChannelEvent>,
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
                _ = events.closed() => {
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
            let event = match envelope.get("type").and_then(Value::as_str) {
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
                Some("interactive") => map_interactive(payload),
                other => {
                    debug!(envelope_type = ?other, "ignoring Slack envelope");
                    None
                }
            };

            if let Some(event) = event {
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
        events: &mpsc::Sender<ChannelEvent>,
        shutdown: &CancellationToken,
        seen: &mut SeenMessages,
    ) -> Result<SessionEnd, ChannelError> {
        let bot_user_id = self.ensure_bot_user_id().await?;
        let ws = self.open_socket().await?;
        info!(channel = %self.name, "Slack socket connected");
        Ok(self.session(ws, &bot_user_id, events, shutdown, seen).await)
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
            typing_indicator: false,
            threads: true,
            max_message_len: MAX_MESSAGE_LEN,
        }
    }

    async fn run(
        &self,
        events: mpsc::Sender<ChannelEvent>,
        shutdown: CancellationToken,
    ) -> Result<(), ChannelError> {
        let mut seen = SeenMessages::default();
        let mut backoff = self.reconnect_initial;

        loop {
            if shutdown.is_cancelled() || events.is_closed() {
                return Ok(());
            }
            let outcome = tokio::select! {
                _ = shutdown.cancelled() => return Ok(()),
                outcome = self.connect_and_serve(&events, &shutdown, &mut seen) => outcome,
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
        for chunk in split_mrkdwn(&mrkdwn, SECTION_TEXT_LIMIT) {
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
        let mrkdwn = truncate_chars(&markdown_to_mrkdwn(text), SECTION_TEXT_LIMIT);
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
}
