//! The Telegram channel on the SDK transport, against a local mock of the
//! Bot API: the adapter receives the update, the daemon's handler answers.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::Uri;
use axum::Json;
use serde_json::{json, Value};

use super::*;
use crate::testing::{setup_test_agent, MockProvider};

const TOKEN: &str = "000000:SYNTHETIC";
const OWNER: u64 = 100000001;

#[derive(Default)]
struct MockApi {
    calls: StdMutex<Vec<(String, Value)>>,
    updates: StdMutex<VecDeque<Value>>,
}

async fn handler(State(mock): State<Arc<MockApi>>, uri: Uri, body: Bytes) -> Json<Value> {
    // Bot API method names are matched case-insensitively.
    let method = uri
        .path()
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    mock.calls.lock().unwrap().push((method.clone(), body));
    let result = match method.as_str() {
        "getme" => json!({
            "id": 900000001, "is_bot": true, "first_name": "Helper",
            "username": "synthetic_helper_bot"
        }),
        "getupdates" => {
            let next = mock.updates.lock().unwrap().pop_front();
            match next {
                Some(update) => json!([update]),
                None => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    json!([])
                }
            }
        }
        "sendmessage" => json!({
            "message_id": 5001, "date": 1700000000,
            "chat": { "id": OWNER, "type": "private", "first_name": "Alice" },
            "text": "ok"
        }),
        _ => json!(true),
    };
    Json(json!({ "ok": true, "result": result }))
}

async fn start_mock() -> (Arc<MockApi>, String) {
    let mock = Arc::new(MockApi::default());
    let app = axum::Router::new()
        .fallback(handler)
        .with_state(mock.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (mock, format!("http://{addr}"))
}

async fn channel(
    base: &str,
    provider: MockProvider,
) -> (Arc<TelegramChannel>, crate::testing::TestHarness) {
    let mut harness = setup_test_agent(provider).await.unwrap();
    let agent: Arc<dyn ChannelAgentRuntime> = Arc::new(std::mem::replace(
        &mut harness.agent,
        setup_test_agent(MockProvider::new()).await.unwrap().agent,
    ));
    let inbox = tempfile::tempdir().unwrap().keep();
    let channel = Arc::new(
        TelegramChannel::new(
            TOKEN,
            vec![OWNER],
            TelegramWebhookConfig::default(),
            vec![OWNER],
            ChannelRuntimeDeps {
                agent,
                config_path: inbox.join("config.toml"),
                session_map: Arc::default(),
                task_registry: Arc::new(TaskRegistry::new(50)),
                files_enabled: false,
                inbox_dir: inbox,
                max_file_size_mb: 10,
                state: harness.state.clone(),
                watchdog_stale_threshold_secs: 0,
            },
            String::new(),
            Vec::new(),
        )
        .with_api_url(base),
    );
    (channel, harness)
}

impl MockApi {
    /// Wait for a call to `method` whose body matches `pred`.
    async fn wait_for(&self, method: &str, pred: impl Fn(&Value) -> bool) -> Value {
        for _ in 0..500 {
            let found = self
                .calls
                .lock()
                .unwrap()
                .iter()
                .find(|(m, body)| m == method && pred(body))
                .map(|(_, body)| body.clone());
            if let Some(body) = found {
                return body;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("no {method} call; saw {:#?}", self.calls.lock().unwrap());
    }
}

fn text_update(id: u64, message_id: u64, text: &str) -> Value {
    json!({
        "update_id": id,
        "message": {
            "message_id": message_id,
            "date": 1700000000,
            "chat": { "id": OWNER, "type": "private", "first_name": "Alice" },
            "from": { "id": OWNER, "is_bot": false, "first_name": "Alice", "last_name": "Rivera" },
            "text": text
        }
    })
}

#[tokio::test]
async fn an_owner_message_runs_the_agent_and_the_reply_is_sent() {
    let (mock, base) = start_mock().await;
    mock.updates
        .lock()
        .unwrap()
        .push_back(text_update(1, 10, "hi there"));
    let provider =
        MockProvider::with_responses(vec![MockProvider::text_response("Hello from the agent.")]);
    let (channel, _harness) = channel(&base, provider).await;
    let running = tokio::spawn(channel.clone().start());
    let reply = mock
        .wait_for("sendmessage", |body| {
            body["text"]
                .as_str()
                .is_some_and(|text| text.contains("Hello from the agent."))
        })
        .await;
    assert_eq!(reply["chat_id"], json!(OWNER));
    mock.wait_for("sendchataction", |body| body["action"] == "typing")
        .await;
    running.abort();
}

#[tokio::test]
async fn an_approval_is_asked_with_buttons_and_answered_by_a_press() {
    let (mock, base) = start_mock().await;
    let (channel, _harness) = channel(&base, MockProvider::new()).await;
    let running = tokio::spawn(channel.clone().start());
    // The bot has started (and its adapter is the one answering).
    mock.wait_for("getupdates", |_| true).await;
    let asking = channel.clone();
    let asked = tokio::spawn(async move {
        asking
            .request_approval(
                &format!("telegram:{OWNER}"),
                "rm -rf build/",
                RiskLevel::High,
                &[],
                PermissionMode::Default,
                false,
            )
            .await
    });
    let card = mock
        .wait_for("sendmessage", |body| body["reply_markup"].is_object())
        .await;
    let buttons = card["reply_markup"]["inline_keyboard"][0].clone();
    let labels: Vec<&str> = buttons
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["text"].as_str().unwrap())
        .collect();
    assert_eq!(labels, ["Allow Once", "Allow Always", "Deny"]);
    mock.updates.lock().unwrap().push_back(json!({
        "update_id": 2,
        "callback_query": {
            "id": "cb-1",
            "from": { "id": OWNER, "is_bot": false, "first_name": "Alice" },
            "chat_instance": "synthetic",
            "data": buttons[0]["callback_data"],
            "message": {
                "message_id": 5001,
                "date": 1700000000,
                "chat": { "id": OWNER, "type": "private", "first_name": "Alice" },
                "text": "card"
            }
        }
    }));
    let answer = tokio::time::timeout(Duration::from_secs(10), asked)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(answer, ApprovalResponse::AllowOnce));
    mock.wait_for("answercallbackquery", |_| true).await;
    let edited = mock.wait_for("editmessagetext", |_| true).await;
    assert!(edited["text"]
        .as_str()
        .unwrap()
        .contains("Allowed this once"));
    running.abort();
}

#[tokio::test]
async fn an_update_received_by_the_sdk_adapter_reaches_the_daemons_handler() {
    let (mock, base) = start_mock().await;
    mock.updates.lock().unwrap().push_back(json!({
        "update_id": 1,
        "message": {
            "message_id": 10,
            "date": 1700000000,
            "chat": { "id": OWNER, "type": "private", "first_name": "Alice" },
            "from": { "id": OWNER, "is_bot": false, "first_name": "Alice", "last_name": "Rivera" },
            "text": "/help"
        }
    }));
    let (channel, _harness) = channel(&base, MockProvider::new()).await;
    let running = tokio::spawn(channel.clone().start());

    let mut reply = None;
    for _ in 0..500 {
        reply = mock
            .calls
            .lock()
            .unwrap()
            .iter()
            .find(|(method, body)| method == "sendmessage" && body["chat_id"] == json!(OWNER))
            .map(|(_, body)| body.clone());
        if reply.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    running.abort();
    let seen: Vec<String> = mock
        .calls
        .lock()
        .unwrap()
        .iter()
        .map(|(m, b)| format!("{m} {b}"))
        .collect();
    let reply = reply.unwrap_or_else(|| panic!("the handler answered /help; calls: {seen:#?}"));
    assert!(
        reply["text"].as_str().unwrap_or_default().contains("/help"),
        "{reply}"
    );
    let calls = mock.calls.lock().unwrap();
    assert!(
        calls.iter().any(|(method, _)| method == "deletewebhook"),
        "polling clears a stale webhook first"
    );
    assert!(calls.iter().any(|(method, _)| method == "getupdates"));
}
