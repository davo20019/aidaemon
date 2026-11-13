//! The Telegram channel on the SDK transport, against a local mock of the
//! Bot API: the adapter receives the update, the daemon's handler answers.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::Uri;
use serde_json::{Value, json};

use super::*;
use crate::testing::{MockProvider, setup_test_agent};

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
) -> (Arc<TelegramBot>, crate::testing::TestHarness) {
    let (channel, harness, _config_path) = channel_with_config(base, provider).await;
    (channel, harness)
}

/// A bot on the mock, and the path of its (not yet written) config file.
async fn channel_with_config(
    base: &str,
    provider: MockProvider,
) -> (Arc<TelegramBot>, crate::testing::TestHarness, PathBuf) {
    let mut harness = setup_test_agent(provider).await.unwrap();
    let agent: Arc<dyn DaemonAgentRuntime> = Arc::new(std::mem::replace(
        &mut harness.agent,
        setup_test_agent(MockProvider::new()).await.unwrap().agent,
    ));
    let inbox = tempfile::tempdir().unwrap().keep();
    let config_path = inbox.join("config.toml");
    let channel = Arc::new(
        new_telegram_bot(
            TOKEN,
            vec![OWNER],
            TelegramWebhookConfig::default(),
            vec![OWNER],
            ChannelRuntimeDeps {
                agent,
                config_path: config_path.clone(),
                session_map: Arc::default(),
                task_registry: Arc::new(TaskRegistry::new(50)),
                files_enabled: false,
                inbox_dir: inbox,
                max_file_size_mb: 10,
                state: harness.state.clone(),
                watchdog_stale_threshold_secs: 0,
                watch: Default::default(),
            },
            String::new(),
            Vec::new(),
        )
        .with_api_url(base),
    );
    (channel, harness, config_path)
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
    assert_eq!(labels, ["Allow", "Always allow", "Deny"]);
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
    assert!(
        edited["text"]
            .as_str()
            .unwrap()
            .contains("Allowed this once")
    );
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

fn button_update(id: u64, data: &str) -> Value {
    json!({
        "update_id": id,
        "callback_query": {
            "id": format!("cb-{id}"),
            "from": { "id": OWNER, "is_bot": false, "first_name": "Alice" },
            "chat_instance": "synthetic",
            "data": data,
            "message": {
                "message_id": 5001, "date": 1700000000,
                "chat": { "id": OWNER, "type": "private", "first_name": "Alice" },
                "text": "Which provider?"
            }
        }
    })
}

/// A provider that accepts only `synthetic-good-key`.
async fn key_checking_provider() -> String {
    async fn models(headers: axum::http::HeaderMap) -> axum::http::StatusCode {
        let good = headers
            .get("x-api-key")
            .is_some_and(|key| key == "synthetic-good-key");
        if good {
            axum::http::StatusCode::OK
        } else {
            axum::http::StatusCode::UNAUTHORIZED
        }
    }
    let app = axum::Router::new().route("/v1/models", axum::routing::get(models));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

async fn setkey_channel(
    base: &str,
    provider_base: &str,
) -> (Arc<TelegramBot>, crate::testing::TestHarness, PathBuf) {
    let (channel, harness, config_path) = channel_with_config(base, MockProvider::new()).await;
    std::fs::write(
        &config_path,
        format!(
            "[provider]\nkind = \"openai_chatgpt\"\napi_key = \"\"\n\n\
             # Claude\n[[provider.fallbacks]]\nkind = \"anthropic\"\n\
             base_url = \"{provider_base}\"\napi_key = \"old\"\n"
        ),
    )
    .unwrap();
    (channel, harness, config_path)
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_good_key_is_saved_and_never_reaches_the_agent() {
    let _keychain = crate::testing::KeychainIsolation::acquire();
    let (mock, base) = start_mock().await;
    let provider_base = key_checking_provider().await;
    let (channel, harness, config_path) = setkey_channel(&base, &provider_base).await;
    mock.updates.lock().unwrap().extend([
        text_update(1, 10, "/setkey"),
        button_update(2, "setkey:f0"),
        text_update(3, 11, "synthetic-good-key"),
    ]);
    let running = tokio::spawn(channel.clone().start());

    let menu = mock
        .wait_for("sendmessage", |body| {
            body["text"]
                .as_str()
                .is_some_and(|t| t.contains("Which provider"))
        })
        .await;
    assert!(menu.to_string().contains("Fallback 1: Anthropic"), "{menu}");
    assert!(!menu.to_string().contains("openai_chatgpt"), "{menu}");
    mock.wait_for("deletemessage", |body| body["message_id"] == json!(11))
        .await;
    let done = mock
        .wait_for("sendmessage", |body| {
            body["text"]
                .as_str()
                .is_some_and(|t| t.contains("saved and tested"))
        })
        .await;
    assert!(!done.to_string().contains("synthetic-good-key"), "{done}");
    running.abort();

    let saved = std::fs::read_to_string(&config_path).unwrap();
    assert!(saved.contains("api_key = \"keychain\""), "{saved}");
    assert!(saved.contains("# Claude"), "{saved}");
    assert_eq!(
        crate::config::resolve_from_keychain("provider_fallback_0_api_key").unwrap(),
        "synthetic-good-key"
    );
    assert!(harness.provider.call_log.lock().await.is_empty());
    let calls = mock.calls.lock().unwrap();
    assert!(
        !calls.iter().any(|(method, body)| method == "sendmessage"
            && body.to_string().contains("synthetic-good-key"))
    );
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rejected_key_changes_nothing() {
    let _keychain = crate::testing::KeychainIsolation::acquire();
    let (mock, base) = start_mock().await;
    let provider_base = key_checking_provider().await;
    let (channel, _harness, config_path) = setkey_channel(&base, &provider_base).await;
    let before = std::fs::read_to_string(&config_path).unwrap();
    mock.updates.lock().unwrap().extend([
        button_update(1, "setkey:f0"),
        text_update(2, 11, "synthetic-bad-key"),
    ]);
    let running = tokio::spawn(channel.clone().start());
    mock.wait_for("deletemessage", |body| body["message_id"] == json!(11))
        .await;
    mock.wait_for("sendmessage", |body| {
        body["text"]
            .as_str()
            .is_some_and(|t| t.contains("rejected"))
    })
    .await;
    running.abort();
    assert_eq!(std::fs::read_to_string(&config_path).unwrap(), before);
    assert!(crate::config::resolve_from_keychain("provider_fallback_0_api_key").is_err());
}
