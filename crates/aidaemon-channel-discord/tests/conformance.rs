//! DiscordChannel against the channel conformance kit: REST is the shared
//! `common::MockHttp`, the gateway is a local WebSocket that completes the
//! Hello/Identify (or Resume) handshake and forwards injected dispatches.
//! No network access.

mod common;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use aidaemon_channels::conformance::{check_channel, ChannelHarness};
use aidaemon_channels::{ApprovalPrompt, Conversation, ConversationKind};
use common::{channel, MockHttp, BOT_ID, CHANNEL_ID, GUILD_ID};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::tungstenite::Message;

const DM_CHANNEL: &str = "100000000000000400";

fn alice() -> Value {
    json!({"id": "100000000000000001", "username": "alice", "global_name": "Alice Rivera"})
}

/// Gateway endpoint: Hello, then READY for Identify or RESUMED for Resume,
/// then injected `(event name, data)` dispatches with increasing sequence
/// numbers. Heartbeats are acknowledged. One connection at a time.
struct InjectGateway {
    url: String,
    tx: mpsc::UnboundedSender<(String, Value)>,
}

impl InjectGateway {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::unbounded_channel::<(String, Value)>();
        let queue = Arc::new(Mutex::new(rx));
        let seq = Arc::new(AtomicU64::new(1));
        let resume_url = url.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let (queue, seq, resume_url) = (queue.clone(), seq.clone(), resume_url.clone());
                tokio::spawn(async move {
                    let Ok(ws) = tokio_tungstenite::accept_async(stream).await else {
                        return;
                    };
                    let (mut out, mut inbound) = ws.split();
                    let mut queue = queue.lock().await;
                    let send = |v: Value| Message::Text(v.to_string());
                    let hello = json!({"op": 10, "d": {"heartbeat_interval": 45_000}});
                    if out.send(send(hello)).await.is_err() {
                        return;
                    }
                    loop {
                        tokio::select! {
                            biased;
                            msg = inbound.next() => {
                                let text = match msg {
                                    Some(Ok(Message::Text(t))) => t,
                                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return,
                                    Some(Ok(_)) => continue,
                                };
                                let frame: Value = serde_json::from_str(&text).unwrap_or_default();
                                let s = seq.fetch_add(1, Ordering::SeqCst);
                                let reply = match frame["op"].as_u64() {
                                    Some(1) => json!({"op": 11}),
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
                                    return;
                                }
                            }
                            item = queue.recv() => {
                                let Some((t, d)) = item else { return };
                                let s = seq.fetch_add(1, Ordering::SeqCst);
                                let frame = json!({"op": 0, "s": s, "t": t, "d": d});
                                if out.send(send(frame)).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                });
            }
        });
        Self { url, tx }
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
        self.gateway.tx.send((t.to_string(), d)).unwrap();
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
        let custom_id = self
            .http
            .requests()
            .iter()
            .rev()
            .filter_map(|r| r.body["components"].as_array().cloned())
            .flatten()
            .filter_map(|row| row["components"].as_array().cloned())
            .flatten()
            .find(|button| button["label"] == json!(label))
            .and_then(|button| button["custom_id"].as_str().map(str::to_string))
            .expect("the prompt's button was sent");
        let id = self.snowflake();
        self.dispatch(
            "INTERACTION_CREATE",
            json!({
                "id": id, "type": 3, "token": "synthetic-interaction-token",
                "application_id": BOT_ID, "channel_id": prompt.conversation.id,
                "user": alice(),
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
    assert_eq!(report.passed.len(), 12, "{report}");
    assert!(report.notes.is_empty(), "{report}");
}
