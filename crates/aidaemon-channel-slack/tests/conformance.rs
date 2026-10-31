//! SlackChannel against the channel conformance kit: the Web API is the
//! shared `support::MockApi`, Socket Mode is a local WebSocket whose frames
//! are injected on demand. No network access.

mod support;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use aidaemon_channel_slack::SlackChannel;
use aidaemon_channels::conformance::{check_channel, ChannelHarness};
use aidaemon_channels::{ApprovalPrompt, Conversation, ConversationKind};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use support::{MockApi, Response, APP_TOKEN, BOT_TOKEN, BOT_USER};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::tungstenite::Message;

const DM: &str = "D000SYNTH";
const GROUP: &str = "C000SYNTH";
const ALICE: &str = "U000ALICE";

/// A Socket Mode endpoint that sends `hello`, then forwards injected
/// envelopes to whichever client is connected (one at a time).
struct InjectSocket {
    url: String,
    tx: mpsc::UnboundedSender<Value>,
}

impl InjectSocket {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/socket", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::unbounded_channel::<Value>();
        let queue = Arc::new(Mutex::new(rx));
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let queue = queue.clone();
                tokio::spawn(async move {
                    let Ok(ws) = tokio_tungstenite::accept_async(stream).await else {
                        return;
                    };
                    let (mut out, mut inbound) = ws.split();
                    // One connection drains the queue at a time.
                    let mut queue = queue.lock().await;
                    let hello = json!({ "type": "hello", "num_connections": 1 });
                    if out.send(Message::Text(hello.to_string())).await.is_err() {
                        return;
                    }
                    loop {
                        tokio::select! {
                            // Notice a closing client before handing it a frame.
                            biased;
                            msg = inbound.next() => match msg {
                                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return,
                                Some(Ok(_)) => {} // acks
                            },
                            frame = queue.recv() => {
                                let Some(frame) = frame else { return };
                                if out.send(Message::Text(frame.to_string())).await.is_err() {
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

struct SlackHarness {
    api: MockApi,
    socket: InjectSocket,
    seq: AtomicU64,
}

impl SlackHarness {
    async fn start() -> Self {
        let socket = InjectSocket::start().await;
        let socket_url = socket.url.clone();
        let api = MockApi::start(move |req, prior| match req.method() {
            "auth.test" => Response::json(json!({
                "ok": true, "user_id": BOT_USER, "team_id": "T000SYNTH", "user": "aidaemon"
            })),
            "apps.connections.open" => Response::json(json!({ "ok": true, "url": socket_url })),
            _ => Response::json(json!({ "ok": true, "ts": format!("1700000001.{prior:06}") })),
        })
        .await;
        Self {
            api,
            socket,
            seq: AtomicU64::new(1),
        }
    }

    fn next(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::SeqCst)
    }

    fn inject_event(&self, event: Value) {
        let n = self.next();
        let envelope = json!({
            "envelope_id": format!("env-{n}"),
            "type": "events_api",
            "accepts_response_payload": false,
            "payload": { "team_id": "T000SYNTH", "type": "event_callback", "event": event }
        });
        self.socket.tx.send(envelope).unwrap();
    }
}

#[async_trait::async_trait]
impl ChannelHarness for SlackHarness {
    fn direct_conversation(&self) -> Conversation {
        Conversation {
            id: DM.to_string(),
            kind: ConversationKind::Direct,
        }
    }

    async fn inject_direct_message(&self, text: &str) {
        let ts = format!("1700000000.{:06}", self.next());
        self.inject_event(json!({
            "type": "message", "user": ALICE, "channel": DM, "channel_type": "im",
            "ts": ts, "text": text
        }));
    }

    async fn inject_group_message(&self, text: &str, addressed: bool) {
        let ts = format!("1700000000.{:06}", self.next());
        let text = if addressed {
            format!("<@{BOT_USER}> {text}")
        } else {
            text.to_string()
        };
        self.inject_event(json!({
            "type": "message", "user": ALICE, "channel": GROUP, "channel_type": "channel",
            "ts": ts, "text": text
        }));
    }

    async fn press_button(&self, prompt: &ApprovalPrompt, choice_id: &str) {
        let label = &prompt
            .choices
            .iter()
            .find(|c| c.id == choice_id)
            .expect("choice exists")
            .label;
        let button = self
            .api
            .calls("chat.postMessage")
            .iter()
            .rev()
            .filter_map(|req| req.json()["blocks"].as_array().cloned())
            .flatten()
            .filter_map(|block| block["elements"].as_array().cloned())
            .flatten()
            .find(|el| el["text"]["text"] == json!(label))
            .expect("the prompt's button was posted");
        let n = self.next();
        self.socket
            .tx
            .send(json!({
                "envelope_id": format!("env-{n}"),
                "type": "interactive",
                "payload": {
                    "type": "block_actions",
                    "user": { "id": ALICE, "name": "alice" },
                    "channel": { "id": prompt.conversation.id },
                    "message": { "ts": "1700000001.000000" },
                    "actions": [{ "action_id": button["action_id"], "value": button["value"] }]
                }
            }))
            .unwrap();
    }

    async fn sent_messages(&self) -> Vec<String> {
        self.api
            .calls("chat.postMessage")
            .iter()
            .map(|req| req.json()["text"].as_str().unwrap_or_default().to_string())
            .collect()
    }
}

#[tokio::test]
async fn slack_channel_conforms() {
    let harness = SlackHarness::start().await;
    let channel = Arc::new(
        SlackChannel::new(BOT_TOKEN, APP_TOKEN)
            .with_api_base(&harness.api.base)
            .with_reconnect_backoff(Duration::from_millis(10), Duration::from_millis(50)),
    );
    let report = check_channel(channel, &harness).await;
    report.assert_all_passed();
    assert_eq!(report.passed.len(), 12, "{report}");
    assert!(report.notes.is_empty(), "{report}");
}
