//! The Discord channel on the SDK transport, against a local mock of the
//! Discord REST API. Inbound events are handed to the channel directly, as
//! the gateway would deliver them.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use aidaemon_channels::{
    ApprovalAnswer, ChannelEvent, CommandInvocation, Conversation, ConversationKind,
    InboundMessage, Sender,
};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

use super::*;
use crate::runtime_ports::DaemonAgentRuntime;
use crate::tasks::TaskRegistry;
use crate::testing::{setup_test_agent, MockProvider};

const OWNER: u64 = 100_000_000_000_001_001;
const STRANGER: u64 = 100_000_000_000_002_002;

#[derive(Debug, Clone)]
struct Recorded {
    method: String,
    path: String,
    body: Value,
}

/// Serve one connection's requests in order.
async fn serve(
    socket: tokio::net::TcpStream,
    recorded: Arc<StdMutex<Vec<Recorded>>>,
    scripted: Arc<StdMutex<VecDeque<Value>>>,
    next_id: Arc<std::sync::atomic::AtomicU64>,
) {
    let mut reader = BufReader::new(socket);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
            return;
        }
        let mut parts = line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();
        let mut length = 0usize;
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header).await.unwrap_or(0) == 0 {
                return;
            }
            let header = header.trim_end();
            if header.is_empty() {
                break;
            }
            if let Some((key, value)) = header.split_once(':') {
                if key.eq_ignore_ascii_case("content-length") {
                    length = value.trim().parse().unwrap_or(0);
                }
            }
        }
        let mut body = vec![0u8; length];
        if reader.read_exact(&mut body).await.is_err() {
            return;
        }
        recorded.lock().unwrap().push(Recorded {
            method,
            path,
            body: serde_json::from_slice(&body).unwrap_or(Value::Null),
        });
        let id = next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let response = scripted
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| json!({ "id": format!("{}", 100_000_000_000_009_000u64 + id) }));
        let text = response.to_string();
        let out = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{text}",
            text.len()
        );
        if reader.get_mut().write_all(out.as_bytes()).await.is_err() {
            return;
        }
    }
}

/// A minimal Discord REST mock: records requests, answers each with a
/// scripted body or a message object with a fresh id.
struct MockDiscord {
    base: String,
    requests: Arc<StdMutex<Vec<Recorded>>>,
    script: Arc<StdMutex<VecDeque<Value>>>,
}

impl MockDiscord {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests: Arc<StdMutex<Vec<Recorded>>> = Arc::default();
        let script: Arc<StdMutex<VecDeque<Value>>> = Arc::default();
        let (recorded, scripted) = (requests.clone(), script.clone());
        tokio::spawn(async move {
            let next_id: Arc<std::sync::atomic::AtomicU64> = Arc::default();
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    return;
                };
                let (recorded, scripted, next_id) =
                    (recorded.clone(), scripted.clone(), next_id.clone());
                tokio::spawn(serve(socket, recorded, scripted, next_id));
            }
        });
        Self {
            base,
            requests,
            script,
        }
    }

    fn push(&self, body: Value) {
        self.script.lock().unwrap().push_back(body);
    }

    fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }

    /// Wait for a request matching `pred`.
    async fn wait_for(&self, pred: impl Fn(&Recorded) -> bool) -> Recorded {
        for _ in 0..500 {
            if let Some(found) = self.requests().into_iter().find(|r| pred(r)) {
                return found;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("no matching request; saw {:?}", self.requests());
    }

    fn transport(&self) -> Arc<SdkDiscord> {
        Arc::new(
            SdkDiscord::new("synthetic.discord.token")
                .with_api_base(self.base.clone())
                .with_http_client(reqwest::Client::builder().no_proxy().build().unwrap()),
        )
    }
}

/// A channel on the SDK transport against `mock`. Dynamic bots have no
/// owner ids (and so auto-claim); configured ones have `owners`.
async fn channel(
    mock: &MockDiscord,
    provider: MockProvider,
    allowed: Vec<u64>,
    owners: Vec<u64>,
) -> (Arc<DiscordChannel>, crate::testing::TestHarness) {
    let mut harness = setup_test_agent(provider).await.unwrap();
    let agent: Arc<dyn DaemonAgentRuntime> = Arc::new(std::mem::replace(
        &mut harness.agent,
        setup_test_agent(MockProvider::new()).await.unwrap().agent,
    ));
    let inbox = tempfile::tempdir().unwrap().keep();
    let channel = Arc::new(DiscordChannel::new(
        "synthetic.discord.token",
        allowed,
        owners,
        None,
        ChannelRuntimeDeps {
            agent,
            config_path: inbox.join("config.toml"),
            session_map: Arc::default(),
            task_registry: Arc::new(TaskRegistry::new(50)),
            files_enabled: true,
            inbox_dir: inbox,
            max_file_size_mb: 10,
            state: harness.state.clone(),
            watchdog_stale_threshold_secs: 0,
        },
    ));
    channel.use_transport(mock.transport());
    (channel, harness)
}

fn direct_message(from: u64, text: &str) -> ChannelEvent {
    ChannelEvent::Message(InboundMessage::new(
        Conversation {
            id: "100000000000000700".to_string(),
            kind: ConversationKind::Direct,
        },
        Sender {
            id: from.to_string(),
            display_name: Some("Alice Rivera".to_string()),
            is_bot: false,
        },
        format!("m-{from}-{text}"),
        text,
    ))
}

#[tokio::test]
async fn an_owner_dm_runs_the_agent_and_the_reply_goes_back_over_the_sdk() {
    let mock = MockDiscord::start().await;
    let provider =
        MockProvider::with_responses(vec![MockProvider::text_response("Hello from the agent.")]);
    let (channel, _harness) = channel(&mock, provider, vec![OWNER], vec![OWNER]).await;
    channel
        .handle_event(direct_message(OWNER, "hi there"))
        .await;

    let reply = mock
        .wait_for(|r| {
            r.method == "POST"
                && r.path == "/channels/100000000000000700/messages"
                && r.body["content"] == "Hello from the agent."
        })
        .await;
    assert_eq!(reply.body["allowed_mentions"], json!({"parse": []}));
    assert!(
        mock.requests()
            .iter()
            .any(|r| r.path == "/channels/100000000000000700/typing"),
        "typing is shown while the agent works"
    );
}

#[tokio::test]
async fn a_dm_from_someone_else_is_refused_before_the_agent_runs() {
    let mock = MockDiscord::start().await;
    let (channel, harness) = channel(&mock, MockProvider::new(), vec![OWNER], vec![OWNER]).await;
    channel
        .handle_event(direct_message(STRANGER, "let me in"))
        .await;
    let refusal = mock
        .wait_for(|r| r.path == "/channels/100000000000000700/messages")
        .await;
    assert_eq!(refusal.body["content"], "Unauthorized.");
    assert_eq!(harness.provider.call_count().await, 0);
}

#[tokio::test]
async fn a_dynamic_bot_is_claimed_by_its_first_dm() {
    let mock = MockDiscord::start().await;
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("Claimed.")]);
    let (channel, _harness) = channel(&mock, provider, vec![], vec![]).await;
    channel
        .handle_event(direct_message(STRANGER, "hello"))
        .await;
    mock.wait_for(|r| r.body["content"] == "Claimed.").await;
    assert!(channel.is_authorized(STRANGER));
}

#[tokio::test]
async fn a_slash_command_fills_in_its_deferred_reply() {
    let mock = MockDiscord::start().await;
    let (channel, _harness) = channel(&mock, MockProvider::new(), vec![OWNER], vec![OWNER]).await;
    channel
        .handle_event(ChannelEvent::Command(CommandInvocation {
            conversation: Conversation {
                id: "100000000000000700".to_string(),
                kind: ConversationKind::Direct,
            },
            sender: Sender {
                id: OWNER.to_string(),
                display_name: None,
                is_bot: false,
            },
            name: "help".to_string(),
            args: String::new(),
            handle: "100000000000000009/synthetic-command-token".to_string(),
        }))
        .await;
    let answer = mock.wait_for(|r| r.method == "PATCH").await;
    assert_eq!(
        answer.path,
        "/webhooks/100000000000000009/synthetic-command-token/messages/@original"
    );
    assert!(answer.body["content"].as_str().unwrap().contains("/help"));
}

#[tokio::test]
async fn an_approval_is_asked_with_buttons_and_answered_by_a_press() {
    let mock = MockDiscord::start().await;
    let (channel, _harness) = channel(&mock, MockProvider::new(), vec![OWNER], vec![OWNER]).await;
    // Opening the DM returns its channel.
    mock.push(json!({ "id": "100000000000000700", "type": 1 }));

    let asking = channel.clone();
    let asked = tokio::spawn(async move {
        asking
            .request_approval(
                &format!("discord:dm:{OWNER}"),
                "rm -rf build/",
                RiskLevel::High,
                &[],
                PermissionMode::Default,
                false,
            )
            .await
    });
    let opened = mock.wait_for(|r| r.path == "/users/@me/channels").await;
    assert_eq!(opened.body, json!({ "recipient_id": OWNER.to_string() }));
    let prompt = mock
        .wait_for(|r| {
            r.path == "/channels/100000000000000700/messages" && r.body["components"].is_array()
        })
        .await;
    let buttons = prompt.body["components"][0]["components"]
        .as_array()
        .unwrap();
    let labels: Vec<&str> = buttons
        .iter()
        .map(|b| b["label"].as_str().unwrap())
        .collect();
    assert_eq!(labels, vec!["Allow Once", "Allow Always", "Deny"]);
    let (prompt_id, choice) =
        aidaemon_channel_discord::decode_custom_id(buttons[0]["custom_id"].as_str().unwrap())
            .unwrap();
    assert_eq!(choice, "once");

    // Someone who is not authorized cannot answer it.
    let press = |from: u64| {
        ChannelEvent::Approval(ApprovalAnswer::new(
            prompt_id.clone(),
            "once",
            Sender {
                id: from.to_string(),
                display_name: None,
                is_bot: false,
            },
            Conversation {
                id: "100000000000000700".to_string(),
                kind: ConversationKind::Direct,
            },
        ))
    };
    channel.handle_event(press(STRANGER)).await;
    assert!(!asked.is_finished());

    channel.handle_event(press(OWNER)).await;
    let answer = tokio::time::timeout(Duration::from_secs(5), asked)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(answer, ApprovalResponse::AllowOnce));
    // The prompt is replaced with its outcome and loses its buttons.
    let edited = mock.wait_for(|r| r.method == "PATCH").await;
    assert_eq!(edited.body["components"], json!([]));
}

#[test]
fn session_ids_resolve_for_default_and_named_bots() {
    assert_eq!(base_session_id("discord:dm:42"), "discord:dm:42");
    assert_eq!(base_session_id("helper_bot:discord:dm:42"), "discord:dm:42");
    assert_eq!(base_session_id("helper_bot:discord:ch:77"), "discord:ch:77");
    assert_eq!(base_session_id("slack:C1"), "slack:C1");
}
