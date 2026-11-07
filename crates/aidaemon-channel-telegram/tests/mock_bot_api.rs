//! TelegramChannel against a local mock of the Bot API. No network access.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aidaemon_channel_telegram::{TelegramChannel, MAX_CALLBACK_DATA};
use aidaemon_channels::{
    ApprovalChoice, ApprovalPrompt, Channel, ChannelError, ChannelEvent, Conversation,
    ConversationKind, MessageRef, OutboundMessage,
};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{StatusCode, Uri};
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const TOKEN: &str = "000000:SYNTHETIC";
const BOT_ID: i64 = 900000001;
const BOT_USERNAME: &str = "synthetic_helper_bot";
const DM_CHAT: i64 = 100000001;
const GROUP_CHAT: i64 = -1000000000001;

#[derive(Clone, Debug)]
struct Call {
    method: String,
    body: Value,
    /// The body as sent (multipart uploads are not JSON).
    raw: String,
    at: Instant,
}

#[derive(Default)]
struct Mock {
    calls: Mutex<Vec<Call>>,
    scripted: Mutex<HashMap<String, VecDeque<(u16, Value)>>>,
    next_message_id: AtomicI64,
}

impl Mock {
    fn script(&self, method: &str, status: u16, body: Value) {
        self.scripted
            .lock()
            .unwrap()
            .entry(method.to_string())
            .or_default()
            .push_back((status, body));
    }

    fn script_ok(&self, method: &str, result: Value) {
        self.script(method, 200, json!({ "ok": true, "result": result }));
    }

    fn calls(&self, method: &str) -> Vec<Call> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.method == method)
            .cloned()
            .collect()
    }

    async fn wait_for_calls(&self, method: &str, n: usize) -> Vec<Call> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let calls = self.calls(method);
            if calls.len() >= n {
                return calls;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {n} {method} calls"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

async fn handler(State(mock): State<Arc<Mock>>, uri: Uri, body: Bytes) -> axum::response::Response {
    let path = uri.path();
    // The file endpoint `getFile` paths are fetched from.
    if let Some(file) = path.strip_prefix(&format!("/file/bot{TOKEN}/")) {
        mock.calls.lock().unwrap().push(Call {
            method: format!("file:{file}"),
            body: Value::Null,
            raw: String::new(),
            at: Instant::now(),
        });
        return (StatusCode::OK, "synthetic file bytes").into_response();
    }
    json_handler(mock, path, body).await.into_response()
}

async fn json_handler(mock: Arc<Mock>, path: &str, body: Bytes) -> impl IntoResponse {
    let prefix = format!("/bot{TOKEN}/");
    let Some(method) = path.strip_prefix(&prefix) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "ok": false, "error_code": 404, "description": "Not Found" })),
        );
    };
    let raw = String::from_utf8_lossy(&body).into_owned();
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    mock.calls.lock().unwrap().push(Call {
        method: method.to_string(),
        body,
        raw,
        at: Instant::now(),
    });
    let scripted = mock
        .scripted
        .lock()
        .unwrap()
        .get_mut(method)
        .and_then(|q| q.pop_front());
    if let Some((status, body)) = scripted {
        return (StatusCode::from_u16(status).unwrap(), Json(body));
    }
    let result = match method {
        "getMe" => json!({
            "id": BOT_ID, "is_bot": true, "first_name": "Helper", "username": BOT_USERNAME
        }),
        "getUpdates" => {
            // Simulate an idle long poll.
            tokio::time::sleep(Duration::from_millis(50)).await;
            json!([])
        }
        "sendMessage" | "sendDocument" => {
            json!({ "message_id": 5000 + mock.next_message_id.fetch_add(1, Ordering::SeqCst) })
        }
        "getFile" => json!({ "file_id": "synthetic-file", "file_path": "documents/notes.txt" }),
        _ => json!(true),
    };
    (
        StatusCode::OK,
        Json(json!({ "ok": true, "result": result })),
    )
}

async fn start_mock() -> (Arc<Mock>, String) {
    let mock = Arc::new(Mock::default());
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

fn channel(base: &str) -> TelegramChannel {
    TelegramChannel::new(TOKEN)
        .api_base(base)
        .poll_timeout_secs(1)
        .retry_backoff(Duration::from_millis(10), Duration::from_millis(50))
        .http_client(reqwest::Client::builder().no_proxy().build().unwrap())
}

struct Running {
    events: mpsc::Receiver<ChannelEvent>,
    shutdown: CancellationToken,
    handle: tokio::task::JoinHandle<Result<(), ChannelError>>,
}

impl Running {
    async fn next(&mut self) -> ChannelEvent {
        tokio::time::timeout(Duration::from_secs(10), self.events.recv())
            .await
            .expect("timed out waiting for event")
            .expect("channel closed")
    }

    async fn stop(self) -> Result<(), ChannelError> {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(10), self.handle)
            .await
            .expect("run did not stop")
            .unwrap()
    }
}

fn spawn(channel: Arc<TelegramChannel>) -> Running {
    let (tx, rx) = mpsc::channel(64);
    let shutdown = CancellationToken::new();
    let token = shutdown.clone();
    let handle = tokio::spawn(async move { channel.run(tx, token).await });
    Running {
        events: rx,
        shutdown,
        handle,
    }
}

fn alice() -> Value {
    json!({ "id": 100000001, "is_bot": false, "first_name": "Alice", "last_name": "Rivera" })
}

fn jordan() -> Value {
    json!({ "id": 100000002, "is_bot": false, "first_name": "Jordan", "last_name": "Lee" })
}

fn bot_user() -> Value {
    json!({ "id": BOT_ID, "is_bot": true, "first_name": "Helper", "username": BOT_USERNAME })
}

fn dm_chat() -> Value {
    json!({ "id": DM_CHAT, "type": "private", "first_name": "Alice" })
}

fn group_chat() -> Value {
    json!({ "id": GROUP_CHAT, "type": "supergroup", "title": "Acme Corp Team" })
}

fn message(id: i64, chat: Value, from: Value, text: &str) -> Value {
    json!({ "message_id": id, "date": 1700000000, "chat": chat, "from": from, "text": text })
}

fn expect_message(event: ChannelEvent) -> aidaemon_channels::InboundMessage {
    match event {
        ChannelEvent::Message(m) => m,
        other => panic!("expected message, got {other:?}"),
    }
}

#[tokio::test]
async fn maps_updates_and_advances_offset() {
    let (mock, base) = start_mock().await;

    let dm = message(1, dm_chat(), alice(), "hello there");

    let mut not_addressed = message(2, group_chat(), alice(), "ping @someone_else please");
    not_addressed["entities"] = json!([{ "type": "mention", "offset": 5, "length": 13 }]);

    // Username appears as a substring but not as a mention entity.
    let substring_only = message(
        3,
        group_chat(),
        jordan(),
        "mail synthetic_helper_bot@example.com",
    );

    let mut mentioned = message(
        4,
        group_chat(),
        alice(),
        "@synthetic_helper_bot what's the status?",
    );
    mentioned["entities"] = json!([{ "type": "mention", "offset": 0, "length": 21 }]);

    let mut reply_to_bot = message(5, group_chat(), jordan(), "and tomorrow?");
    reply_to_bot["reply_to_message"] = message(4999, group_chat(), bot_user(), "Earlier answer");

    let mut forum = message(6, group_chat(), alice(), "topic chatter");
    forum["is_topic_message"] = json!(true);
    forum["message_thread_id"] = json!(42);
    // Telegram links topic messages to the topic-creation service message,
    // which here was created by the bot: not a reply to the bot.
    forum["reply_to_message"] = json!({
        "message_id": 42, "date": 1700000000, "chat": group_chat(), "from": bot_user(),
        "is_topic_message": true, "message_thread_id": 42,
        "forum_topic_created": { "name": "Planning", "icon_color": 7322096 }
    });

    let mut command = message(7, group_chat(), alice(), "/help@synthetic_helper_bot now");
    command["entities"] = json!([{ "type": "bot_command", "offset": 0, "length": 26 }]);

    let mut other_bot_command = message(8, group_chat(), alice(), "/help@another_bot");
    other_bot_command["entities"] = json!([{ "type": "bot_command", "offset": 0, "length": 17 }]);

    let photo = json!({
        "message_id": 9, "date": 1700000000, "chat": dm_chat(), "from": alice(),
        "caption": "look at this",
        "photo": [
            { "file_id": "photo-small", "file_unique_id": "u-small", "width": 90, "height": 90, "file_size": 1000 },
            { "file_id": "photo-large", "file_unique_id": "u-large", "width": 1280, "height": 960, "file_size": 90000 }
        ]
    });
    let document = json!({
        "message_id": 10, "date": 1700000000, "chat": dm_chat(), "from": alice(),
        "document": { "file_id": "doc-1", "file_unique_id": "u-doc", "file_name": "report.pdf",
                      "mime_type": "application/pdf", "file_size": 2048 }
    });
    // A sticker-only message carries no text or attachment we map: skipped.
    let sticker = json!({
        "message_id": 11, "date": 1700000000, "chat": dm_chat(), "from": alice(),
        "sticker": { "file_id": "stk" }
    });

    let updates: Vec<Value> = [
        dm,
        not_addressed,
        substring_only,
        mentioned,
        reply_to_bot,
        forum,
        command,
        other_bot_command,
        photo,
    ]
    .into_iter()
    .enumerate()
    .map(|(n, m)| json!({ "update_id": 700 + n as i64, "message": m }))
    .collect();
    mock.script_ok("getUpdates", json!(updates));
    mock.script_ok(
        "getUpdates",
        json!([
            { "update_id": 709, "message": document },
            { "update_id": 710, "message": sticker },
        ]),
    );

    let ch = Arc::new(channel(&base));
    let mut run = spawn(ch);

    let m = expect_message(run.next().await);
    assert_eq!(
        m.conversation,
        Conversation {
            id: DM_CHAT.to_string(),
            kind: ConversationKind::Direct
        }
    );
    assert!(m.addressed);
    assert_eq!(m.text, "hello there");
    assert_eq!(m.message_id, "1");
    assert_eq!(m.sender.id, "100000001");
    assert_eq!(m.sender.display_name.as_deref(), Some("Alice Rivera"));
    assert!(!m.sender.is_bot);

    let m = expect_message(run.next().await);
    assert_eq!(m.conversation.kind, ConversationKind::Group);
    assert_eq!(m.conversation.id, GROUP_CHAT.to_string());
    assert!(
        !m.addressed,
        "mention of another user must not address the bot"
    );
    assert_eq!(m.text, "ping @someone_else please");

    let m = expect_message(run.next().await);
    assert!(
        !m.addressed,
        "substring without a mention entity is not addressing"
    );

    let m = expect_message(run.next().await);
    assert!(m.addressed);
    assert_eq!(m.text, "what's the status?");

    let m = expect_message(run.next().await);
    assert!(m.addressed, "reply to a bot message addresses the bot");

    let m = expect_message(run.next().await);
    assert_eq!(
        m.conversation,
        Conversation {
            id: format!("{GROUP_CHAT}:42"),
            kind: ConversationKind::Thread
        }
    );
    assert!(
        !m.addressed,
        "topic-creation link is not a reply to the bot"
    );

    let m = expect_message(run.next().await);
    assert!(m.addressed);
    assert_eq!(m.text, "/help now");

    let m = expect_message(run.next().await);
    assert!(!m.addressed, "command aimed at another bot");

    let m = expect_message(run.next().await);
    assert_eq!(m.text, "look at this");
    assert_eq!(m.attachments.len(), 1);
    assert_eq!(m.attachments[0].source, "photo-large");
    assert_eq!(m.attachments[0].mime_type.as_deref(), Some("image/jpeg"));
    assert_eq!(m.attachments[0].size_bytes, Some(90000));

    let m = expect_message(run.next().await);
    assert_eq!(m.text, "");
    assert_eq!(m.attachments.len(), 1);
    assert_eq!(m.attachments[0].name, "report.pdf");
    assert_eq!(m.attachments[0].source, "doc-1");
    assert_eq!(
        m.attachments[0].mime_type.as_deref(),
        Some("application/pdf")
    );

    // Third poll must acknowledge everything up to update 710.
    let polls = mock.wait_for_calls("getUpdates", 3).await;
    assert_eq!(polls[0].body.get("offset"), None);
    assert_eq!(polls[1].body["offset"], json!(709));
    assert_eq!(polls[2].body["offset"], json!(711));
    assert_eq!(
        polls[0].body["allowed_updates"],
        json!(["message", "callback_query"])
    );
    assert_eq!(polls[0].body["timeout"], json!(1));
    assert_eq!(mock.calls("getMe").len(), 1);

    // The sticker (update 710) was acknowledged but produced no event.
    assert!(run.events.try_recv().is_err());
    run.stop().await.unwrap();
}

#[tokio::test]
async fn configured_username_skips_get_me() {
    let (mock, base) = start_mock().await;
    let mut reply = message(1, group_chat(), alice(), "thanks");
    reply["reply_to_message"] = message(2, group_chat(), bot_user(), "done");
    mock.script_ok("getUpdates", json!([{ "update_id": 1, "message": reply }]));

    let ch = Arc::new(channel(&base).bot_username("@synthetic_helper_bot"));
    let mut run = spawn(ch);
    let m = expect_message(run.next().await);
    assert!(m.addressed);
    run.stop().await.unwrap();
    assert!(mock.calls("getMe").is_empty());
}

#[tokio::test]
async fn callback_query_becomes_approval_and_is_answered() {
    let (mock, base) = start_mock().await;
    let ch = Arc::new(channel(&base));

    let thread = Conversation {
        id: format!("{GROUP_CHAT}:42"),
        kind: ConversationKind::Thread,
    };
    let prompt = ApprovalPrompt {
        id: "approval-7f3c".to_string(),
        conversation: thread.clone(),
        text: "Run `terminal` command?".to_string(),
        choices: vec![
            ApprovalChoice {
                id: "allow_once".into(),
                label: "Allow once".into(),
            },
            ApprovalChoice {
                id: "deny".into(),
                label: "Deny".into(),
            },
        ],
    };
    let sent = ch.ask(&prompt).await.unwrap();
    let ask_call = &mock.calls("sendMessage")[0];
    let data = ask_call.body["reply_markup"]["inline_keyboard"][0][1]["callback_data"]
        .as_str()
        .unwrap()
        .to_string();

    mock.script_ok(
        "getUpdates",
        json!([{
            "update_id": 50,
            "callback_query": {
                "id": "cbq-1",
                "from": jordan(),
                "chat_instance": "ci-1",
                "data": data,
                "message": {
                    "message_id": sent.message_id.parse::<i64>().unwrap(),
                    "date": 1700000000,
                    "chat": group_chat(),
                    "from": bot_user(),
                    "is_topic_message": true,
                    "message_thread_id": 42,
                    "text": "Run terminal command?"
                }
            }
        }]),
    );

    let mut run = spawn(ch);
    match run.next().await {
        ChannelEvent::Approval(answer) => {
            assert_eq!(answer.prompt_id, "approval-7f3c");
            assert_eq!(answer.choice_id, "deny");
            assert_eq!(answer.sender.id, "100000002");
            assert_eq!(answer.conversation, thread);
        }
        other => panic!("expected approval, got {other:?}"),
    }
    let answers = mock.wait_for_calls("answerCallbackQuery", 1).await;
    assert_eq!(answers[0].body["callback_query_id"], json!("cbq-1"));
    run.stop().await.unwrap();
}

#[tokio::test]
async fn unknown_callback_data_is_answered_without_event() {
    let (mock, base) = start_mock().await;
    mock.script_ok(
        "getUpdates",
        json!([
            { "update_id": 1, "callback_query": {
                "id": "cbq-x", "from": alice(), "chat_instance": "ci", "data": "a2:999",
                "message": message(3, dm_chat(), bot_user(), "old prompt") } },
            { "update_id": 2, "message": message(4, dm_chat(), alice(), "after") }
        ]),
    );
    let mut run = spawn(Arc::new(channel(&base)));
    let m = expect_message(run.next().await);
    assert_eq!(m.text, "after");
    let answers = mock.wait_for_calls("answerCallbackQuery", 1).await;
    assert_eq!(answers[0].body["callback_query_id"], json!("cbq-x"));
    run.stop().await.unwrap();
}

#[tokio::test]
async fn unauthorized_returns_auth_error() {
    let (mock, base) = start_mock().await;
    mock.script(
        "getMe",
        401,
        json!({ "ok": false, "error_code": 401, "description": "Unauthorized" }),
    );
    let run = spawn(Arc::new(channel(&base)));
    let result = tokio::time::timeout(Duration::from_secs(10), run.handle)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(result, Err(ChannelError::Auth(_))), "{result:?}");

    // Also when the token is revoked while polling.
    mock.script(
        "getUpdates",
        401,
        json!({ "ok": false, "error_code": 401, "description": "Unauthorized" }),
    );
    let run = spawn(Arc::new(channel(&base).bot_username(BOT_USERNAME)));
    let result = tokio::time::timeout(Duration::from_secs(10), run.handle)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(result, Err(ChannelError::Auth(_))), "{result:?}");
}

#[tokio::test]
async fn rate_limit_retry_after_is_honoured() {
    let (mock, base) = start_mock().await;
    mock.script(
        "getUpdates",
        429,
        json!({ "ok": false, "error_code": 429,
                "description": "Too Many Requests: retry after 1",
                "parameters": { "retry_after": 1 } }),
    );
    let run = spawn(Arc::new(channel(&base).bot_username(BOT_USERNAME)));
    let polls = mock.wait_for_calls("getUpdates", 2).await;
    let gap = polls[1].at - polls[0].at;
    // Backoff is 10 ms in tests, so only retry_after explains the wait.
    assert!(gap >= Duration::from_millis(950), "retried after {gap:?}");
    run.stop().await.unwrap();
}

#[tokio::test]
async fn transient_errors_back_off_and_recover() {
    let (mock, base) = start_mock().await;
    mock.script(
        "getUpdates",
        502,
        json!({ "ok": false, "error_code": 502, "description": "Bad Gateway" }),
    );
    mock.script("getUpdates", 500, json!("not json envelope"));
    mock.script_ok(
        "getUpdates",
        json!([{ "update_id": 3, "message": message(1, dm_chat(), alice(), "back") }]),
    );
    let mut run = spawn(Arc::new(channel(&base).bot_username(BOT_USERNAME)));
    let m = expect_message(run.next().await);
    assert_eq!(m.text, "back");
    run.stop().await.unwrap();
}

#[tokio::test]
async fn shutdown_and_dropped_receiver_stop_run() {
    let (_mock, base) = start_mock().await;
    let run = spawn(Arc::new(channel(&base).bot_username(BOT_USERNAME)));
    tokio::time::sleep(Duration::from_millis(100)).await;
    run.stop().await.unwrap();

    let (tx, rx) = mpsc::channel(1);
    drop(rx);
    let ch = channel(&base).bot_username(BOT_USERNAME);
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        ch.run(tx, CancellationToken::new()),
    )
    .await
    .expect("run should stop when the receiver is dropped");
    assert!(result.is_ok());
}

#[tokio::test]
async fn send_converts_markdown_and_targets_threads() {
    let (mock, base) = start_mock().await;
    let ch = channel(&base);
    let thread = Conversation {
        id: format!("{GROUP_CHAT}:42"),
        kind: ConversationKind::Thread,
    };
    let msg = OutboundMessage {
        text: "**Done** for Alice Rivera: `a < b` & [docs](https://example.com)".to_string(),
        reply_to: Some("77".to_string()),
    };
    let r = ch.send(&thread, &msg).await.unwrap();
    assert_eq!(r.conversation_id, thread.id);
    let call = &mock.calls("sendMessage")[0];
    assert_eq!(call.body["chat_id"], json!(GROUP_CHAT));
    assert_eq!(call.body["message_thread_id"], json!(42));
    assert_eq!(call.body["reply_to_message_id"], json!(77));
    assert_eq!(call.body["parse_mode"], json!("HTML"));
    assert_eq!(
        call.body["text"],
        json!("<b>Done</b> for Alice Rivera: <code>a &lt; b</code> &amp; <a href=\"https://example.com\">docs</a>")
    );
    assert_eq!(r.message_id, "5000");
}

#[tokio::test]
async fn send_splits_long_messages() {
    let (mock, base) = start_mock().await;
    let ch = channel(&base);
    let dm = Conversation {
        id: DM_CHAT.to_string(),
        kind: ConversationKind::Direct,
    };
    let text = (0..800)
        .map(|n| format!("- item **{n}** of the synthetic list"))
        .collect::<Vec<_>>()
        .join("\n");
    let msg = OutboundMessage {
        text,
        reply_to: Some("12".to_string()),
    };
    let r = ch.send(&dm, &msg).await.unwrap();
    let calls = mock.calls("sendMessage");
    assert!(
        calls.len() >= 2,
        "expected several chunks, got {}",
        calls.len()
    );
    for (n, call) in calls.iter().enumerate() {
        let text = call.body["text"].as_str().unwrap();
        assert!(text.encode_utf16().count() <= 4096);
        assert_eq!(text.matches("<b>").count(), text.matches("</b>").count());
        if n == 0 {
            assert_eq!(call.body["reply_to_message_id"], json!(12));
        } else {
            assert!(call.body.get("reply_to_message_id").is_none());
        }
    }
    // The returned ref is the last chunk's.
    assert_eq!(r.message_id, (5000 + calls.len() as i64 - 1).to_string());
}

#[tokio::test]
async fn parse_error_falls_back_to_plain_text() {
    let (mock, base) = start_mock().await;
    mock.script(
        "sendMessage",
        400,
        json!({ "ok": false, "error_code": 400,
                "description": "Bad Request: can't parse entities: Unsupported start tag \"x\" at byte offset 3" }),
    );
    let ch = channel(&base);
    let dm = Conversation {
        id: DM_CHAT.to_string(),
        kind: ConversationKind::Direct,
    };
    ch.send(
        &dm,
        &OutboundMessage::text("**hi** <there> & [site](https://example.com)"),
    )
    .await
    .unwrap();
    let calls = mock.calls("sendMessage");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].body["parse_mode"], json!("HTML"));
    assert!(calls[1].body.get("parse_mode").is_none());
    assert_eq!(
        calls[1].body["text"],
        json!("hi <there> & site (https://example.com)")
    );
}

#[tokio::test]
async fn send_reports_errors() {
    let (mock, base) = start_mock().await;
    mock.script(
        "sendMessage",
        403,
        json!({ "ok": false, "error_code": 403, "description": "Forbidden: bot was blocked by the user" }),
    );
    let ch = channel(&base);
    let dm = Conversation {
        id: DM_CHAT.to_string(),
        kind: ConversationKind::Direct,
    };
    let err = ch
        .send(&dm, &OutboundMessage::text("hi"))
        .await
        .unwrap_err();
    assert!(matches!(err, ChannelError::Other(_)), "{err:?}");

    mock.script(
        "sendMessage",
        429,
        json!({ "ok": false, "error_code": 429, "description": "Too Many Requests",
                "parameters": { "retry_after": 120 } }),
    );
    let err = ch
        .send(&dm, &OutboundMessage::text("hi"))
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            ChannelError::RateLimited {
                retry_after_secs: Some(120)
            }
        ),
        "{err:?}"
    );
}

#[tokio::test]
async fn edit_ignores_not_modified() {
    let (mock, base) = start_mock().await;
    let ch = channel(&base);
    let r = MessageRef {
        conversation_id: format!("{GROUP_CHAT}:42"),
        message_id: "5001".to_string(),
    };
    ch.edit(&r, "**updated**").await.unwrap();
    let call = &mock.calls("editMessageText")[0];
    assert_eq!(call.body["chat_id"], json!(GROUP_CHAT));
    assert_eq!(call.body["message_id"], json!(5001));
    assert_eq!(call.body["text"], json!("<b>updated</b>"));
    assert_eq!(call.body["parse_mode"], json!("HTML"));

    mock.script(
        "editMessageText",
        400,
        json!({ "ok": false, "error_code": 400,
                "description": "Bad Request: message is not modified: specified new message content and reply markup are exactly the same" }),
    );
    ch.edit(&r, "**updated**").await.unwrap();

    mock.script(
        "editMessageText",
        400,
        json!({ "ok": false, "error_code": 400, "description": "Bad Request: message to edit not found" }),
    );
    assert!(ch.edit(&r, "x").await.is_err());
}

#[tokio::test]
async fn ask_renders_inline_keyboard_with_compact_callback_data() {
    let (mock, base) = start_mock().await;
    let ch = channel(&base);
    let dm = Conversation {
        id: DM_CHAT.to_string(),
        kind: ConversationKind::Direct,
    };
    let long_id = format!("approval-{}", "x".repeat(80));
    for prompt_id in ["p-1", long_id.as_str()] {
        let prompt = ApprovalPrompt {
            id: prompt_id.to_string(),
            conversation: dm.clone(),
            text: "Allow **write_file** to `notes.md`?".to_string(),
            choices: vec![
                ApprovalChoice {
                    id: "allow_once".into(),
                    label: "Allow once".into(),
                },
                ApprovalChoice {
                    id: "allow_session".into(),
                    label: "Allow for session".into(),
                },
                ApprovalChoice {
                    id: "deny".into(),
                    label: "Deny".into(),
                },
            ],
        };
        ch.ask(&prompt).await.unwrap();
    }
    let calls = mock.calls("sendMessage");
    assert_eq!(calls.len(), 2);
    for call in &calls {
        assert_eq!(
            call.body["text"],
            json!("Allow <b>write_file</b> to <code>notes.md</code>?")
        );
        let rows = call.body["reply_markup"]["inline_keyboard"]
            .as_array()
            .unwrap();
        let buttons: Vec<&Value> = rows.iter().flat_map(|r| r.as_array().unwrap()).collect();
        let labels: Vec<&str> = buttons
            .iter()
            .map(|b| b["text"].as_str().unwrap())
            .collect();
        assert_eq!(labels, ["Allow once", "Allow for session", "Deny"]);
        let mut seen = std::collections::HashSet::new();
        for b in &buttons {
            let data = b["callback_data"].as_str().unwrap();
            assert!(data.len() <= MAX_CALLBACK_DATA, "{data} too long");
            assert!(seen.insert(data.to_string()));
        }
    }
}

#[tokio::test]
async fn typing_sends_chat_action() {
    let (mock, base) = start_mock().await;
    let ch = channel(&base);
    let thread = Conversation {
        id: format!("{GROUP_CHAT}:42"),
        kind: ConversationKind::Thread,
    };
    ch.typing(&thread).await.unwrap();
    let call = &mock.calls("sendChatAction")[0];
    assert_eq!(
        call.body,
        json!({ "chat_id": GROUP_CHAT, "action": "typing", "message_thread_id": 42 })
    );
    let caps = ch.capabilities();
    assert!(caps.edit_messages && caps.buttons && caps.typing_indicator && caps.threads);
    assert_eq!(caps.max_message_len, 4096);
    assert_eq!(ch.name(), "telegram");
    assert_eq!(channel(&base).with_name("support").name(), "support");
}

#[tokio::test]
async fn delete_removes_the_message() {
    let (mock, base) = start_mock().await;
    channel(&base)
        .delete(&MessageRef {
            conversation_id: DM_CHAT.to_string(),
            message_id: "5001".to_string(),
        })
        .await
        .unwrap();
    let calls = mock.calls("deleteMessage");
    assert_eq!(
        calls[0].body,
        json!({ "chat_id": DM_CHAT, "message_id": 5001 })
    );
}

#[tokio::test]
async fn send_file_uploads_a_document_with_a_plain_caption() {
    let (mock, base) = start_mock().await;
    let conversation = Conversation {
        id: DM_CHAT.to_string(),
        kind: ConversationKind::Direct,
    };
    let sent = channel(&base)
        .send_file(
            &conversation,
            &aidaemon_channels::OutboundFile {
                name: "report.txt".to_string(),
                bytes: b"synthetic report for Acme Corp".to_vec(),
                mime_type: Some("text/plain".to_string()),
                caption: Some("The **weekly** report".to_string()),
            },
        )
        .await
        .unwrap();
    assert_eq!(sent.conversation_id, DM_CHAT.to_string());
    let calls = mock.calls("sendDocument");
    assert_eq!(calls.len(), 1);
    let raw = &calls[0].raw;
    assert!(raw.contains("filename=\"report.txt\""), "{raw}");
    assert!(raw.contains("synthetic report for Acme Corp"));
    assert!(
        raw.contains("The weekly report"),
        "the caption is plain text: {raw}"
    );
    assert!(raw.contains(&DM_CHAT.to_string()));
}

#[tokio::test]
async fn download_resolves_the_file_id_then_fetches_it() {
    let (mock, base) = start_mock().await;
    let bytes = channel(&base)
        .download(&aidaemon_channels::Attachment {
            name: "notes.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            size_bytes: Some(20),
            source: "synthetic-file".to_string(),
        })
        .await
        .unwrap();
    assert_eq!(bytes, b"synthetic file bytes");
    assert_eq!(
        mock.calls("getFile")[0].body,
        json!({ "file_id": "synthetic-file" })
    );
    assert_eq!(mock.calls("file:documents/notes.txt").len(), 1);

    // Over the Bot API's download limit: refused without a call.
    let err = channel(&base)
        .download(&aidaemon_channels::Attachment {
            name: "big.bin".to_string(),
            mime_type: None,
            size_bytes: Some(aidaemon_channel_telegram::MAX_DOWNLOAD_BYTES + 1),
            source: "synthetic-big".to_string(),
        })
        .await
        .unwrap_err();
    assert!(matches!(err, ChannelError::Other(_)));
    assert_eq!(mock.calls("getFile").len(), 1);
}

#[tokio::test]
async fn register_commands_sets_the_bot_menu() {
    let (mock, base) = start_mock().await;
    channel(&base)
        .register_commands(&[aidaemon_channels::CommandSpec {
            name: "status".to_string(),
            description: "Show what the agent is doing".to_string(),
        }])
        .await
        .unwrap();
    assert_eq!(
        mock.calls("setMyCommands")[0].body,
        json!({ "commands": [{ "command": "status", "description": "Show what the agent is doing" }] })
    );
}

/// A free local port (bound, then released).
#[cfg(feature = "webhook")]
async fn free_addr() -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap()
}

#[cfg(feature = "webhook")]
#[tokio::test]
async fn webhook_mode_registers_the_hook_and_accepts_only_signed_deliveries() {
    let (mock, base) = start_mock().await;
    let listen = free_addr().await;
    let mut webhook = aidaemon_channel_telegram::WebhookConfig::new(
        listen,
        "https://bot.example.test/hooks/synthetic",
    );
    webhook.secret = Some("synthetic-secret_1".to_string());
    webhook.drop_pending_updates = true;
    let mut run = spawn(Arc::new(channel(&base).with_webhook(webhook)));

    let registered = mock.wait_for_calls("setWebhook", 1).await;
    assert_eq!(
        registered[0].body["url"],
        "https://bot.example.test/hooks/synthetic"
    );
    assert_eq!(registered[0].body["secret_token"], "synthetic-secret_1");
    assert_eq!(registered[0].body["drop_pending_updates"], true);

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let update =
        json!({ "update_id": 1, "message": message(1, dm_chat(), alice(), "hello by hook") });
    let url = format!("http://{listen}/hooks/synthetic");
    // Wait for the listener, then deliver without the secret: refused.
    let mut unsigned = None;
    for _ in 0..100 {
        if let Ok(resp) = client.post(&url).json(&update).send().await {
            unsigned = Some(resp.status());
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(unsigned, Some(reqwest::StatusCode::UNAUTHORIZED));

    let signed = client
        .post(&url)
        .header("X-Telegram-Bot-Api-Secret-Token", "synthetic-secret_1")
        .json(&update)
        .send()
        .await
        .unwrap();
    assert_eq!(signed.status(), reqwest::StatusCode::OK);
    let m = expect_message(run.next().await);
    assert_eq!(m.text, "hello by hook");
    assert!(
        mock.calls("getUpdates").is_empty(),
        "no polling in webhook mode"
    );
    run.stop().await.unwrap();
}

#[cfg(feature = "webhook")]
#[tokio::test]
async fn webhook_mode_falls_back_to_polling_when_the_listener_cannot_bind() {
    let (mock, base) = start_mock().await;
    let taken = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let webhook = aidaemon_channel_telegram::WebhookConfig::new(
        taken.local_addr().unwrap(),
        "https://bot.example.test/hooks/synthetic",
    );
    let run = spawn(Arc::new(channel(&base).with_webhook(webhook)));
    mock.wait_for_calls("getUpdates", 1).await;
    assert_eq!(mock.calls("deleteWebhook").len(), 1);
    assert!(mock.calls("setWebhook").is_empty());
    run.stop().await.unwrap();
    drop(taken);
}

#[tokio::test]
async fn display_name_comes_from_the_private_chat() {
    let (mock, base) = start_mock().await;
    mock.script_ok(
        "getChat",
        json!({ "id": 100000001, "type": "private", "first_name": "Alice", "last_name": "Rivera" }),
    );
    let ch = channel(&base);
    assert_eq!(
        ch.display_name("100000001").await.unwrap().as_deref(),
        Some("Alice Rivera")
    );
    mock.script(
        "getChat",
        400,
        json!({ "ok": false, "error_code": 400, "description": "Bad Request: chat not found" }),
    );
    assert_eq!(ch.display_name("100000009").await.unwrap(), None);
    assert_eq!(ch.display_name("not-a-number").await.unwrap(), None);
}

#[tokio::test]
async fn a_shorter_message_length_splits_sooner() {
    let (mock, base) = start_mock().await;
    let ch = channel(&base).with_max_message_len(100);
    assert_eq!(ch.capabilities().max_message_len, 100);
    let conversation = Conversation {
        id: DM_CHAT.to_string(),
        kind: ConversationKind::Direct,
    };
    ch.send(&conversation, &OutboundMessage::text("word ".repeat(60)))
        .await
        .unwrap();
    let sends = mock.calls("sendMessage");
    assert!(sends.len() >= 3, "{} sends", sends.len());
    for send in sends {
        assert!(send.body["text"].as_str().unwrap().encode_utf16().count() <= 100);
    }
}

/// The raw mode hands every update over as the Bot API sent it, kinds the
/// adapter does not map included, and still advances the offset.
#[tokio::test]
async fn run_updates_hands_over_each_update_as_sent() {
    let (mock, base) = start_mock().await;
    let edited =
        json!({ "update_id": 41, "edited_message": message(9, dm_chat(), alice(), "fixed") });
    let plain = json!({ "update_id": 42, "message": message(10, dm_chat(), alice(), "hello") });
    mock.script_ok("getUpdates", json!([edited.clone(), plain.clone()]));
    let ch = Arc::new(channel(&base));
    let (tx, mut rx) = mpsc::channel(8);
    let shutdown = CancellationToken::new();
    let stop = shutdown.clone();
    let run = tokio::spawn(async move { ch.run_updates(tx, stop).await });

    let first = tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first, edited);
    let second = tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(second, plain);
    let polls = mock.wait_for_calls("getUpdates", 2).await;
    assert_eq!(polls[1].body["offset"], 43);

    shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
