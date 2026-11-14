//! SlackChannel against the channel conformance kit: the Web API is the
//! shared `support::MockApi`, Socket Mode is a local WebSocket whose frames
//! are injected on demand. No network access.

mod support;

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use aidaemon_channel_slack::SlackChannel;
use aidaemon_channels::conformance::{ChannelHarness, check_channel};
use aidaemon_channels::{ApprovalPrompt, Attachment, Conversation, ConversationKind};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use support::{APP_TOKEN, BOT_TOKEN, BOT_USER, MockApi, Response};
use tokio::net::TcpListener;
use tokio::sync::{Notify, watch};
use tokio_tungstenite::tungstenite::Message;

const DM: &str = "D000SYNTH";
const GROUP: &str = "C000SYNTH";
const ALICE: &str = "U000ALICE";
const FILE_BODY: &str = "synthetic attachment";

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

/// A Socket Mode endpoint that sends `hello`, then forwards injected
/// envelopes to the newest connected client. Envelopes are confirmed by the
/// adapter's `{"envelope_id": ...}` acks.
struct InjectSocket {
    url: String,
    inbox: Arc<Inbox<Value>>,
}

impl InjectSocket {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/socket", listener.local_addr().unwrap());
        let inbox: Arc<Inbox<Value>> = Inbox::new();
        let shared = inbox.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let inbox = shared.clone();
                tokio::spawn(async move {
                    let Ok(ws) = tokio_tungstenite::accept_async(stream).await else {
                        return;
                    };
                    let (mut out, mut inbound) = ws.split();
                    let mut superseded = inbox.take_over();
                    let mut unacked: Vec<Value> = Vec::new();
                    let hello = json!({ "type": "hello", "num_connections": 1 });
                    if out.send(Message::text(hello.to_string())).await.is_err() {
                        return;
                    }
                    loop {
                        tokio::select! {
                            biased;
                            _ = superseded.changed() => break,
                            msg = inbound.next() => match msg {
                                Some(Ok(Message::Text(text))) => {
                                    let ack: Value = serde_json::from_str(&text).unwrap_or_default();
                                    if let Some(id) = ack["envelope_id"].as_str() {
                                        unacked.retain(|frame| frame["envelope_id"] != id);
                                    }
                                }
                                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                                Some(Ok(_)) => {}
                            },
                            frame = inbox.pop() => {
                                let text = frame.to_string();
                                unacked.push(frame);
                                if out.send(Message::text(text)).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    inbox.requeue(unacked);
                });
            }
        });
        Self { url, inbox }
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
            "conf.bin" => Response::json(json!(FILE_BODY)),
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
        self.socket.inbox.push(envelope);
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

    async fn inject_named_message(&self, text: &str, display_name: &str) -> bool {
        let ts = format!("1700000000.{:06}", self.next());
        self.inject_event(json!({
            "type": "message", "user": ALICE, "channel": DM, "channel_type": "im",
            "ts": ts, "text": text, "user_profile": { "display_name": display_name }
        }));
        true
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
        // The mock's `ts` for a post is its position among the posts.
        let (button, ts) = self
            .api
            .calls("chat.postMessage")
            .iter()
            .enumerate()
            .rev()
            .find_map(|(position, req)| {
                let button = req.json()["blocks"]
                    .as_array()?
                    .iter()
                    .filter_map(|block| block["elements"].as_array().cloned())
                    .flatten()
                    .find(|el| el["text"]["text"] == json!(label))?;
                Some((button, format!("1700000001.{position:06}")))
            })
            .expect("the prompt's button was posted");
        let n = self.next();
        self.socket.inbox.push(json!({
                "envelope_id": format!("env-{n}"),
                "type": "interactive",
                "payload": {
                    "type": "block_actions",
                    "user": { "id": ALICE, "name": "alice" },
                    "channel": { "id": prompt.conversation.id },
                    "message": { "ts": ts, "text": prompt.text },
                    "actions": [{ "action_id": button["action_id"], "value": button["value"] }]
                }
        }));
    }

    async fn sent_messages(&self) -> Vec<String> {
        self.api
            .calls("chat.postMessage")
            .iter()
            .map(|req| req.json()["text"].as_str().unwrap_or_default().to_string())
            .collect()
    }

    async fn edited_messages(&self) -> Option<Vec<String>> {
        Some(
            self.api
                .calls("chat.update")
                .iter()
                .map(|req| req.json()["text"].as_str().unwrap_or_default().to_string())
                .collect(),
        )
    }

    async fn inject_reply_message(&self, text: &str, parent_message_id: &str) -> bool {
        let ts = format!("1700000000.{:06}", self.next());
        self.inject_event(json!({
            "type": "message", "user": ALICE, "channel": DM, "channel_type": "im",
            "ts": ts, "thread_ts": parent_message_id, "text": text
        }));
        true
    }

    async fn downloadable_attachment(&self) -> Option<(Attachment, Vec<u8>)> {
        Some((
            Attachment::new("conf.bin", format!("{}/files/conf.bin", self.api.base)),
            serde_json::to_vec(FILE_BODY).unwrap(),
        ))
    }
}

#[tokio::test]
async fn slack_channel_conforms() {
    let harness = SlackHarness::start().await;
    let channel = Arc::new(
        SlackChannel::new(BOT_TOKEN, APP_TOKEN)
            .with_api_base(&harness.api.base)
            .with_download_client(reqwest::Client::builder().no_proxy())
            .with_reconnect_backoff(Duration::from_millis(10), Duration::from_millis(50)),
    );
    let report = check_channel(channel, &harness).await;
    report.assert_all_passed();
    assert_eq!(report.passed.len(), 21, "{report}");
    assert!(report.notes.is_empty(), "{report}");
}
