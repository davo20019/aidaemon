mod support;

use std::time::Duration;

use aidaemon_channel_slack::SlackChannel;
use aidaemon_channels::{Channel, ChannelError, ChannelEvent, ConversationKind};
use serde_json::{json, Value};
use support::*;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

fn channel(api: &MockApi) -> SlackChannel {
    SlackChannel::new(BOT_TOKEN, APP_TOKEN)
        .with_api_base(&api.base)
        .with_reconnect_backoff(Duration::from_millis(10), Duration::from_millis(50))
}

fn hello() -> Value {
    json!({ "type": "hello", "num_connections": 1 })
}

fn events_api(envelope_id: &str, event: Value) -> Value {
    json!({
        "envelope_id": envelope_id,
        "type": "events_api",
        "accepts_response_payload": false,
        "payload": { "team_id": "T000SYNTH", "type": "event_callback", "event": event }
    })
}

fn channel_message(ts: &str, text: &str) -> Value {
    json!({
        "type": "message", "user": "U000ALICE", "channel": "C000SYNTH",
        "channel_type": "channel", "ts": ts, "text": text
    })
}

async fn next_event(rx: &mut mpsc::Receiver<ChannelEvent>) -> ChannelEvent {
    timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("timed out waiting for event")
        .expect("event channel closed")
}

#[tokio::test]
async fn acks_every_envelope_and_maps_events() {
    let approval_value = json!({ "p": "prompt-7", "c": "deny" }).to_string();
    let socket = MockSocket::start(vec![vec![
        hello(),
        events_api(
            "env-1",
            channel_message("1700000000.000100", "<@U000BOT> summarize the thread"),
        ),
        // Same ts delivered again as app_mention: deduplicated.
        events_api(
            "env-2",
            json!({
                "type": "app_mention", "user": "U000ALICE", "channel": "C000SYNTH",
                "ts": "1700000000.000100", "text": "<@U000BOT> summarize the thread"
            }),
        ),
        // Bot-authored and edited messages are skipped.
        events_api(
            "env-3",
            json!({
                "type": "message", "subtype": "bot_message", "bot_id": "B000SYNTH",
                "channel": "C000SYNTH", "ts": "1700000000.000200", "text": "beep"
            }),
        ),
        events_api(
            "env-4",
            json!({
                "type": "message", "subtype": "message_changed", "channel": "C000SYNTH",
                "ts": "1700000000.000300", "message": { "text": "edited" }
            }),
        ),
        events_api(
            "env-5",
            json!({
                "type": "message", "user": "U000ALICE", "channel": "D000SYNTH",
                "channel_type": "im", "ts": "1700000000.000400", "text": "hi in private"
            }),
        ),
        events_api(
            "env-6",
            json!({
                "type": "message", "user": "U000ALICE", "channel": "C000SYNTH",
                "channel_type": "channel", "ts": "1700000000.000500",
                "thread_ts": "1700000000.000100", "text": "follow-up in thread"
            }),
        ),
        json!({
            "envelope_id": "env-7",
            "type": "interactive",
            "payload": {
                "type": "block_actions",
                "user": { "id": "U000ALICE", "name": "alice" },
                "channel": { "id": "C000SYNTH" },
                "message": { "ts": "1700000000.000600" },
                "actions": [{ "action_id": "aidaemon_approval_1", "value": approval_value }]
            }
        }),
        json!({ "envelope_id": "env-8", "type": "slash_commands", "payload": {} }),
    ]])
    .await;
    let api = standard_api(socket.url.clone()).await;
    let slack = channel(&api);

    let (tx, mut rx) = mpsc::channel(16);
    let shutdown = CancellationToken::new();
    let stop = shutdown.clone();
    let run = tokio::spawn(async move { slack.run(tx, stop).await });

    // 1. Mention in a channel: addressed, stripped, new thread.
    let ChannelEvent::Message(m) = next_event(&mut rx).await else {
        panic!("expected message")
    };
    assert!(m.addressed);
    assert_eq!(m.text, "summarize the thread");
    assert_eq!(m.conversation.kind, ConversationKind::Thread);
    assert_eq!(m.conversation.id, "C000SYNTH:1700000000.000100");

    // 2. DM.
    let ChannelEvent::Message(m) = next_event(&mut rx).await else {
        panic!("expected message")
    };
    assert_eq!(m.conversation.kind, ConversationKind::Direct);
    assert_eq!(m.conversation.id, "D000SYNTH");
    assert!(m.addressed);

    // 3. Unaddressed reply inside an existing thread.
    let ChannelEvent::Message(m) = next_event(&mut rx).await else {
        panic!("expected message")
    };
    assert_eq!(m.conversation.kind, ConversationKind::Thread);
    assert_eq!(m.conversation.id, "C000SYNTH:1700000000.000100");
    assert!(!m.addressed);

    // 4. Button press.
    let ChannelEvent::Approval(a) = next_event(&mut rx).await else {
        panic!("expected approval")
    };
    assert_eq!(a.prompt_id, "prompt-7");
    assert_eq!(a.choice_id, "deny");
    assert_eq!(a.sender.id, "U000ALICE");
    assert_eq!(a.conversation.kind, ConversationKind::Group);
    assert_eq!(a.conversation.id, "C000SYNTH");

    let acks = socket.wait_for_acks(8).await;
    let expected: Vec<String> = (1..=8).map(|i| format!("env-{i}")).collect();
    assert_eq!(acks, expected);

    // Nothing else (duplicates, bot, edit, slash command) was forwarded.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(rx.try_recv().is_err());

    shutdown.cancel();
    timeout(Duration::from_secs(5), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();

    let auth = api.calls("auth.test");
    assert_eq!(auth.len(), 1);
    assert_eq!(
        auth[0].header("authorization"),
        Some("Bearer xoxb-synthetic")
    );
    let open = api.calls("apps.connections.open");
    assert_eq!(open.len(), 1);
    assert_eq!(
        open[0].header("authorization"),
        Some("Bearer xapp-synthetic")
    );
}

#[tokio::test]
async fn disconnect_triggers_reconnect() {
    let socket = MockSocket::start(vec![
        vec![
            hello(),
            json!({ "type": "disconnect", "reason": "refresh_requested" }),
        ],
        vec![
            hello(),
            events_api(
                "env-after",
                channel_message("1700000000.000900", "after reconnect"),
            ),
        ],
    ])
    .await;
    let api = standard_api(socket.url.clone()).await;
    let slack = channel(&api).with_reply_in_thread(false);

    let (tx, mut rx) = mpsc::channel(4);
    let shutdown = CancellationToken::new();
    let stop = shutdown.clone();
    let run = tokio::spawn(async move { slack.run(tx, stop).await });

    let ChannelEvent::Message(m) = next_event(&mut rx).await else {
        panic!("expected message")
    };
    assert_eq!(m.text, "after reconnect");
    assert_eq!(m.conversation.kind, ConversationKind::Group);
    assert_eq!(socket.connections(), 2);
    assert_eq!(api.calls("apps.connections.open").len(), 2);
    // The bot user id is resolved once, not per connection.
    assert_eq!(api.calls("auth.test").len(), 1);
    assert_eq!(socket.wait_for_acks(1).await, vec!["env-after".to_string()]);

    shutdown.cancel();
    timeout(Duration::from_secs(5), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn transient_connect_failure_is_retried() {
    let socket = MockSocket::start(vec![vec![
        hello(),
        events_api("env-1", channel_message("1700000000.000100", "made it")),
    ]])
    .await;
    let url = socket.url.clone();
    let api = MockApi::start(move |req, prior| match req.method() {
        "auth.test" => Response::json(json!({ "ok": true, "user_id": BOT_USER })),
        "apps.connections.open" if prior == 0 => {
            Response::json(json!({ "ok": false, "error": "internal_error" })).status(500)
        }
        _ => Response::json(json!({ "ok": true, "url": url })),
    })
    .await;
    let slack = channel(&api);

    let (tx, mut rx) = mpsc::channel(4);
    let shutdown = CancellationToken::new();
    let stop = shutdown.clone();
    let run = tokio::spawn(async move { slack.run(tx, stop).await });

    let ChannelEvent::Message(m) = next_event(&mut rx).await else {
        panic!("expected message")
    };
    assert_eq!(m.text, "made it");
    assert_eq!(api.calls("apps.connections.open").len(), 2);

    shutdown.cancel();
    timeout(Duration::from_secs(5), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn invalid_auth_is_fatal() {
    let api =
        MockApi::start(|_, _| Response::json(json!({ "ok": false, "error": "invalid_auth" })))
            .await;
    let slack = channel(&api);
    let (tx, _rx) = mpsc::channel(4);
    let result = timeout(
        Duration::from_secs(5),
        slack.run(tx, CancellationToken::new()),
    )
    .await
    .expect("run should return");
    match result {
        Err(ChannelError::Auth(reason)) => assert!(reason.contains("invalid_auth")),
        other => panic!("expected Auth, got {other:?}"),
    }
}

#[tokio::test]
async fn not_authed_app_token_is_fatal() {
    let api = MockApi::start(|req, _| match req.method() {
        "auth.test" => Response::json(json!({ "ok": true, "user_id": BOT_USER })),
        _ => Response::json(json!({ "ok": false, "error": "not_authed" })),
    })
    .await;
    let slack = channel(&api);
    let (tx, _rx) = mpsc::channel(4);
    let result = timeout(
        Duration::from_secs(5),
        slack.run(tx, CancellationToken::new()),
    )
    .await
    .expect("run should return");
    assert!(matches!(result, Err(ChannelError::Auth(_))), "{result:?}");
}

#[tokio::test]
async fn shutdown_stops_run_while_connected() {
    let socket = MockSocket::start(vec![vec![hello()]]).await;
    let api = standard_api(socket.url.clone()).await;
    let slack = channel(&api);
    let (tx, _rx) = mpsc::channel(4);
    let shutdown = CancellationToken::new();
    let stop = shutdown.clone();
    let run = tokio::spawn(async move { slack.run(tx, stop).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(socket.connections(), 1);
    shutdown.cancel();
    timeout(Duration::from_secs(5), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn slash_commands_become_command_events() {
    let socket = MockSocket::start(vec![vec![
        hello(),
        json!({
            "envelope_id": "env-cmd",
            "type": "slash_commands",
            "payload": {
                "command": "/status",
                "text": "  verbose ",
                "channel_id": "D000SYNTH",
                "user_id": "U000ALICE",
                "user_name": "alice",
                "response_url": "https://hooks.slack.com/commands/T000SYNTH/1/synthetic"
            }
        }),
    ]])
    .await;
    let api = standard_api(socket.url.clone()).await;
    let slack = channel(&api);
    let (tx, mut rx) = mpsc::channel(4);
    let shutdown = CancellationToken::new();
    let stop = shutdown.clone();
    let run = tokio::spawn(async move { slack.run(tx, stop).await });

    let ChannelEvent::Command(command) = next_event(&mut rx).await else {
        panic!("expected a command")
    };
    assert_eq!(command.name, "status");
    assert_eq!(command.args, "verbose");
    assert_eq!(command.conversation.kind, ConversationKind::Direct);
    assert_eq!(command.conversation.id, "D000SYNTH");
    assert_eq!(command.sender.id, "U000ALICE");
    assert_eq!(
        command.handle,
        "https://hooks.slack.com/commands/T000SYNTH/1/synthetic"
    );
    assert_eq!(socket.wait_for_acks(1).await, vec!["env-cmd".to_string()]);

    shutdown.cancel();
    timeout(Duration::from_secs(5), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
