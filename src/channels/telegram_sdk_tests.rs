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
    // teloxide names methods in PascalCase (`SendMessage`), the adapter in
    // camelCase (`getUpdates`).
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
    let mut harness = setup_test_agent(MockProvider::new()).await.unwrap();
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
                transports: Default::default(),
            },
            String::new(),
            Vec::new(),
        )
        .with_api_url(&base),
    );
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

/// Updates arrive as `serde_json::Value`, whose keys are sorted; teloxide
/// needs `update_id` first. Without the rebuild every update parses as an
/// unknown kind and is never handled.
#[test]
fn updates_handed_over_as_values_parse_as_teloxide_updates() {
    let raw = json!({
        "update_id": 1,
        "message": {
            "message_id": 10,
            "date": 1700000000,
            "chat": { "id": OWNER, "type": "private", "first_name": "Alice" },
            "from": { "id": OWNER, "is_bot": false, "first_name": "Alice", "last_name": "Rivera" },
            "text": "/help"
        }
    });
    let naive = serde_json::from_value::<teloxide::types::Update>(raw.clone()).unwrap();
    assert!(
        matches!(naive.kind, teloxide::types::UpdateKind::Error(_)),
        "if teloxide stops needing the order, parse_update can go"
    );
    let update = parse_update(raw).unwrap();
    assert_eq!(update.id.0, 1);
    assert!(matches!(
        update.kind,
        teloxide::types::UpdateKind::Message(_)
    ));
}
