//! DiscordChannel against the channel conformance kit: REST is the shared
//! `common::MockHttp`, the gateway is a local WebSocket that completes the
//! Hello/Identify (or Resume) handshake and forwards injected dispatches.
//! No network access.

mod common;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use aidaemon_channels::conformance::{check_channel, ChannelHarness};
use aidaemon_channels::{Attachment, ApprovalPrompt, Conversation, ConversationKind};
use common::{channel, MockHttp, BOT_ID, CHANNEL_ID, GUILD_ID};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{watch, Notify};
use tokio_tungstenite::tungstenite::Message;

const DM_CHANNEL: &str = "100000000000000400";

fn alice() -> Value {
    json!({"id": "100000000000000001", "username": "alice", "global_name": "Alice Rivera"})
}

/// Frames waiting for the adapter, delivered so that none can be lost when
/// it reconnects. A new connection takes over at once (the previous one
/// stops, even if the mock has not yet noticed that client went away), and
/// a frame stays owned by the connection that sent it until the client
/// confirms it; frames a connection never had confirmed go back to the front
/// of the queue when it ends. Without this, a connection whose client had
/// already disconnected could take a frame, "send" it into a dead socket and
/// drop it, a race that only showed up on some platforms.
struct Inbox<T> {
    queue: std::sync::Mutex<VecDeque<T>>,
    notify: Notify,
    generation: watch::Sender<u64>,
}

impl<T> Inbox<T> {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            queue: std::sync::Mutex::new(VecDeque::new()),
            notify: Notify::new(),
            generation: watch::channel(0).0,
        })
    }

    fn push(&self, frame: T) {
        self.queue.lock().unwrap().push_back(frame);
        self.notify.notify_waiters();
    }

    /// Put unconfirmed frames back, ahead of newer ones, in their order.
    fn requeue(&self, frames: Vec<T>) {
        let mut queue = self.queue.lock().unwrap();
        for frame in frames.into_iter().rev() {
            queue.push_front(frame);
        }
        drop(queue);
        self.notify.notify_waiters();
    }

    /// Make the calling connection the only one taking frames. The returned
    /// receiver changes when a newer connection takes over.
    fn take_over(&self) -> watch::Receiver<u64> {
        self.generation.send_modify(|g| *g += 1);
        self.generation.subscribe()
    }

    async fn pop(&self) -> T {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(frame) = self.queue.lock().unwrap().pop_front() {
                return frame;
            }
            notified.await;
        }
    }
}

/// Gateway endpoint: Hello, then READY for Identify or RESUMED for Resume,
/// then injected `(event name, data)` dispatches with increasing sequence
/// numbers, sent to the newest connection. Heartbeats are acknowledged; the
/// sequence number a heartbeat carries confirms every dispatch up to it.
struct InjectGateway {
    url: String,
    inbox: Arc<Inbox<(String, Value)>>,
}

impl InjectGateway {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let inbox: Arc<Inbox<(String, Value)>> = Inbox::new();
        let shared = inbox.clone();
        let seq = Arc::new(AtomicU64::new(1));
        let resume_url = url.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let (inbox, seq, resume_url) = (shared.clone(), seq.clone(), resume_url.clone());
                tokio::spawn(async move {
                    let Ok(ws) = tokio_tungstenite::accept_async(stream).await else {
                        return;
                    };
                    let (mut out, mut inbound) = ws.split();
                    let mut superseded = inbox.take_over();
                    // Dispatches sent on this connection and not yet
                    // confirmed by a heartbeat, with their sequence numbers.
                    let mut unconfirmed: Vec<(u64, (String, Value))> = Vec::new();
                    let send = |v: Value| Message::Text(v.to_string());
                    let hello = json!({"op": 10, "d": {"heartbeat_interval": 45_000}});
                    if out.send(send(hello)).await.is_err() {
                        return;
                    }
                    loop {
                        tokio::select! {
                            biased;
                            _ = superseded.changed() => break,
                            msg = inbound.next() => {
                                let text = match msg {
                                    Some(Ok(Message::Text(t))) => t,
                                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                                    Some(Ok(_)) => continue,
                                };
                                let frame: Value = serde_json::from_str(&text).unwrap_or_default();
                                let s = seq.fetch_add(1, Ordering::SeqCst);
                                let reply = match frame["op"].as_u64() {
                                    Some(1) => {
                                        if let Some(last) = frame["d"].as_u64() {
                                            unconfirmed.retain(|(sent, _)| *sent > last);
                                        }
                                        json!({"op": 11})
                                    }
                                    Some(2) => json!({"op": 0, "s": s, "t": "READY", "d": {
                                        "v": 10,
                                        "user": {"id": BOT_ID, "username": "aidaemon", "bot": true},
                                        "session_id": "synthetic-session-1",
                                        "resume_gateway_url": resume_url,
                                        "guilds": [],
                                    }}),
                                    Some(6) => json!({"op": 0, "s": s, "t": "RESUMED", "d": null}),
                                    _ => continue,
                                };
                                if out.send(send(reply)).await.is_err() {
                                    break;
                                }
                            }
                            item = inbox.pop() => {
                                let s = seq.fetch_add(1, Ordering::SeqCst);
                                let frame = json!({"op": 0, "s": s, "t": item.0, "d": item.1});
                                unconfirmed.push((s, item));
                                if out.send(send(frame)).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    // Unconfirmed dispatches go to the next connection. The
                    // adapter may see one twice; the kit matches probes by
                    // their unique text, so a duplicate is harmless.
                    inbox.requeue(unconfirmed.into_iter().map(|(_, item)| item).collect());
                });
            }
        });
        Self { url, inbox }
    }
}

struct DiscordHarness {
    http: MockHttp,
    gateway: InjectGateway,
    next_id: AtomicU64,
}

impl DiscordHarness {
    fn snowflake(&self) -> String {
        format!(
            "{}",
            100000000000000500u64 + self.next_id.fetch_add(1, Ordering::SeqCst)
        )
    }

    fn dispatch(&self, t: &str, d: Value) {
        self.gateway.inbox.push((t.to_string(), d));
    }
}

#[async_trait::async_trait]
impl ChannelHarness for DiscordHarness {
    fn direct_conversation(&self) -> Conversation {
        Conversation {
            id: DM_CHANNEL.to_string(),
            kind: ConversationKind::Direct,
        }
    }

    async fn inject_direct_message(&self, text: &str) {
        let id = self.snowflake();
        self.dispatch(
            "MESSAGE_CREATE",
            json!({"id": id, "type": 0, "channel_id": DM_CHANNEL, "content": text,
                   "author": alice(), "mentions": []}),
        );
    }

    async fn inject_group_message(&self, text: &str, addressed: bool) {
        let id = self.snowflake();
        let (content, mentions) = if addressed {
            (
                format!("<@{BOT_ID}> {text}"),
                json!([{"id": BOT_ID, "bot": true}]),
            )
        } else {
            (text.to_string(), json!([]))
        };
        self.dispatch(
            "MESSAGE_CREATE",
            json!({"id": id, "type": 0, "channel_id": CHANNEL_ID, "guild_id": GUILD_ID,
                   "content": content, "author": alice(), "mentions": mentions}),
        );
    }

    async fn press_button(&self, prompt: &ApprovalPrompt, choice_id: &str) {
        let label = &prompt
            .choices
            .iter()
            .find(|c| c.id == choice_id)
            .expect("choice exists")
            .label;
        // The mock's id for a response is its position among the requests.
        let (custom_id, message_id) = self
            .http
            .requests()
            .iter()
            .enumerate()
            .rev()
            .find_map(|(position, r)| {
                let button = r.body["components"]
                    .as_array()?
                    .iter()
                    .filter_map(|row| row["components"].as_array().cloned())
                    .flatten()
                    .find(|button| button["label"] == json!(label))?;
                let id = 100000000000001001u64 + position as u64;
                Some((button["custom_id"].clone(), id.to_string()))
            })
            .expect("the prompt's button was sent");
        let id = self.snowflake();
        self.dispatch(
            "INTERACTION_CREATE",
            json!({
                "id": id, "type": 3, "token": "synthetic-interaction-token",
                "application_id": BOT_ID, "channel_id": prompt.conversation.id,
                "user": alice(),
                "message": {"id": message_id, "channel_id": prompt.conversation.id},
                "data": {"custom_id": custom_id, "component_type": 2}
            }),
        );
    }

    async fn sent_messages(&self) -> Vec<String> {
        self.http
            .requests()
            .iter()
            .filter(|r| r.method == "POST" && r.path.ends_with("/messages"))
            .map(|r| r.body["content"].as_str().unwrap_or_default().to_string())
            .collect()
    }

    async fn edited_messages(&self) -> Option<Vec<String>> {
        Some(
            self.http
                .requests()
                .iter()
                .filter(|r| r.method == "PATCH")
                .map(|r| r.body["content"].as_str().unwrap_or_default().to_string())
                .collect(),
        )
    }

    async fn inject_reply_message(&self, text: &str, parent_message_id: &str) -> bool {
        let id = self.snowflake();
        self.dispatch(
            "MESSAGE_CREATE",
            json!({"id": id, "type": 19, "channel_id": DM_CHANNEL, "content": text,
                   "author": alice(), "mentions": [],
                   "message_reference": {"message_id": parent_message_id}}),
        );
        true
    }

    /// The kit fetches it twice; the responses are scripted for that.
    async fn downloadable_attachment(&self) -> Option<(Attachment, Vec<u8>)> {
        let body = json!("synthetic attachment");
        self.http.push(200, body.clone());
        self.http.push(200, body.clone());
        Some((
            Attachment::new("conf.bin", format!("{}/attachments/conf.bin", self.http.base)),
            body.to_string().into_bytes(),
        ))
    }
}

#[tokio::test]
async fn discord_channel_conforms() {
    let harness = DiscordHarness {
        http: MockHttp::start().await,
        gateway: InjectGateway::start().await,
        next_id: AtomicU64::new(1),
    };
    let discord = Arc::new(channel(&harness.http).with_gateway_url(harness.gateway.url.clone()));
    let report = check_channel(discord, &harness).await;
    report.assert_all_passed();
    assert_eq!(report.passed.len(), 17, "{report}");
    assert!(report.notes.is_empty(), "{report}");
}
