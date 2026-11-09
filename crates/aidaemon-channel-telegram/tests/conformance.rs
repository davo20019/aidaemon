//! TelegramChannel against the channel conformance kit, driven through a
//! local mock of the Bot API. No network access.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_channel_telegram::TelegramChannel;
use aidaemon_channels::conformance::{check_channel, ChannelHarness};
use aidaemon_channels::{ApprovalPrompt, Attachment, Conversation, ConversationKind};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{StatusCode, Uri};
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Value};

const TOKEN: &str = "000000:SYNTHETIC";
const BOT_USERNAME: &str = "synthetic_helper_bot";
const DM_CHAT: i64 = 100000001;
const GROUP_CHAT: i64 = -1000000000001;
const FILE_BODY: &str = "synthetic attachment";

/// Bot API mock: `getUpdates` hands out whatever was injected (or idles for
/// 50 ms), `sendMessage` is recorded and gets a fresh message id.
#[derive(Default)]
struct Mock {
    pending: Mutex<VecDeque<Value>>,
    sent: Mutex<Vec<Value>>,
    edits: Mutex<Vec<Value>>,
    next_id: AtomicI64,
}

async fn handler(State(mock): State<Arc<Mock>>, uri: Uri, body: Bytes) -> impl IntoResponse {
    if uri.path().starts_with(&format!("/file/bot{TOKEN}/")) {
        return (StatusCode::OK, Json(json!(FILE_BODY)));
    }
    let Some(method) = uri.path().strip_prefix(&format!("/bot{TOKEN}/")) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "ok": false, "error_code": 404, "description": "Not Found" })),
        );
    };
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let result = match method {
        "getMe" => json!({
            "id": 900000001, "is_bot": true, "first_name": "Helper", "username": BOT_USERNAME
        }),
        "getUpdates" => {
            let updates: Vec<Value> = mock.pending.lock().unwrap().drain(..).collect();
            if updates.is_empty() {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            json!(updates)
        }
        "sendMessage" => {
            let id = 5000 + mock.next_id.fetch_add(1, Ordering::SeqCst);
            let mut recorded = body;
            recorded["sent_as"] = json!(id);
            mock.sent.lock().unwrap().push(recorded);
            json!({ "message_id": id })
        }
        "editMessageText" => {
            mock.edits.lock().unwrap().push(body);
            json!(true)
        }
        "getFile" => json!({ "file_id": "synthetic-file", "file_path": "documents/conf.bin" }),
        _ => json!(true),
    };
    (
        StatusCode::OK,
        Json(json!({ "ok": true, "result": result })),
    )
}

struct TelegramHarness {
    mock: Arc<Mock>,
    next_update: AtomicI64,
}

impl TelegramHarness {
    async fn start() -> (Self, String) {
        let mock = Arc::new(Mock::default());
        let app = axum::Router::new()
            .fallback(handler)
            .with_state(mock.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let harness = Self {
            mock,
            next_update: AtomicI64::new(1),
        };
        (harness, format!("http://{addr}"))
    }

    fn push(&self, kind: &str, payload: impl FnOnce(i64) -> Value) {
        let id = self.next_update.fetch_add(1, Ordering::SeqCst);
        self.mock
            .pending
            .lock()
            .unwrap()
            .push_back(json!({ "update_id": id, kind: payload(id) }));
    }
}

fn alice() -> Value {
    json!({ "id": 100000001, "is_bot": false, "first_name": "Alice", "last_name": "Rivera" })
}

fn dm_chat() -> Value {
    json!({ "id": DM_CHAT, "type": "private", "first_name": "Alice" })
}

#[async_trait::async_trait]
impl ChannelHarness for TelegramHarness {
    fn direct_conversation(&self) -> Conversation {
        Conversation {
            id: DM_CHAT.to_string(),
            kind: ConversationKind::Direct,
        }
    }

    async fn inject_direct_message(&self, text: &str) {
        self.push("message", |id| {
            json!({ "message_id": id, "date": 1700000000, "chat": dm_chat(), "from": alice(), "text": text })
        });
    }

    async fn inject_group_message(&self, text: &str, addressed: bool) {
        let chat = json!({ "id": GROUP_CHAT, "type": "supergroup", "title": "Acme Corp Team" });
        self.push("message", |id| {
            let mut message =
                json!({ "message_id": id, "date": 1700000000, "chat": chat, "from": alice() });
            if addressed {
                let mention = format!("@{BOT_USERNAME}");
                message["text"] = json!(format!("{mention} {text}"));
                message["entities"] =
                    json!([{ "type": "mention", "offset": 0, "length": mention.len() }]);
            } else {
                message["text"] = json!(text);
            }
            message
        });
    }

    async fn press_button(&self, prompt: &ApprovalPrompt, choice_id: &str) {
        let label = &prompt
            .choices
            .iter()
            .find(|c| c.id == choice_id)
            .expect("choice exists")
            .label;
        let (data, message_id) = self
            .mock
            .sent
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find_map(|body| {
                let button = body["reply_markup"]["inline_keyboard"]
                    .as_array()?
                    .iter()
                    .flat_map(|row| row.as_array().cloned().unwrap_or_default())
                    .find(|button| button["text"] == json!(label))?;
                Some((button["callback_data"].clone(), body["sent_as"].clone()))
            })
            .expect("the prompt's button was sent");
        self.push("callback_query", |id| {
            json!({
                "id": format!("cbq-{id}"),
                "from": alice(),
                "message": { "message_id": message_id, "date": 1700000000, "chat": dm_chat() },
                "data": data,
            })
        });
    }

    async fn sent_messages(&self) -> Vec<String> {
        self.mock
            .sent
            .lock()
            .unwrap()
            .iter()
            .map(|body| body["text"].as_str().unwrap_or_default().to_string())
            .collect()
    }

    async fn edited_messages(&self) -> Option<Vec<String>> {
        let edits = self.mock.edits.lock().unwrap();
        Some(
            edits
                .iter()
                .map(|body| body["text"].as_str().unwrap_or_default().to_string())
                .collect(),
        )
    }

    async fn inject_reply_message(&self, text: &str, parent_message_id: &str) -> bool {
        let parent: i64 = parent_message_id.parse().expect("telegram ids are numbers");
        self.push("message", |id| {
            json!({
                "message_id": id, "date": 1700000000, "chat": dm_chat(), "from": alice(), "text": text,
                "reply_to_message": { "message_id": parent, "date": 1700000000, "chat": dm_chat() },
            })
        });
        true
    }

    async fn downloadable_attachment(&self) -> Option<(Attachment, Vec<u8>)> {
        Some((
            Attachment::new("conf.bin", "synthetic-file"),
            serde_json::to_vec(FILE_BODY).unwrap(),
        ))
    }
}

#[tokio::test]
async fn telegram_channel_conforms() {
    let (harness, base) = TelegramHarness::start().await;
    let channel = Arc::new(
        TelegramChannel::new(TOKEN)
            .api_base(base)
            .poll_timeout_secs(1)
            .retry_backoff(Duration::from_millis(10), Duration::from_millis(50))
            .http_client(reqwest::Client::builder().no_proxy().build().unwrap())
            .download_client(reqwest::Client::builder().no_proxy()),
    );
    let report = check_channel(channel, &harness).await;
    report.assert_all_passed();
    assert_eq!(report.passed.len(), 17, "{report}");
    assert!(report.notes.is_empty(), "{report}");
}
