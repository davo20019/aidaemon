//! LaTeX, display names and slash commands against the local Web API and
//! Socket Mode mocks.

mod support;

use std::sync::Arc;
use std::time::Duration;

use aidaemon_channel_slack::SlackChannel;
use aidaemon_channels::{
    AccessPolicy, Channel, ChannelEvent, ChannelRunner, CommandContext, CommandReply,
    DEFAULT_REFUSAL,
};
use aidaemon_core::Agent;
use aidaemon_core::testing::{ScriptedProvider, text};
use serde_json::{Value, json};
use support::*;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

const OWNER: &str = "U000ALICE";

fn channel(api: &MockApi) -> SlackChannel {
    SlackChannel::new(BOT_TOKEN, APP_TOKEN)
        .with_api_base(&api.base)
        .with_reconnect_backoff(Duration::from_millis(10), Duration::from_millis(50))
}

fn hello() -> Value {
    json!({ "type": "hello", "num_connections": 1 })
}

fn dm(envelope: &str, ts: &str, user: &str, body: &str) -> Value {
    json!({
        "envelope_id": envelope,
        "type": "events_api",
        "payload": { "type": "event_callback", "event": {
            "type": "message", "user": user, "channel": "D000SYNTH",
            "channel_type": "im", "ts": ts, "text": body
        }}
    })
}

fn slash(envelope: &str, command: &str, user: &str, base: &str) -> Value {
    json!({
        "envelope_id": envelope,
        "type": "slash_commands",
        "payload": {
            "command": command, "text": " three days ", "channel_id": "D000SYNTH",
            "user_id": user, "response_url": format!("{base}/response/{envelope}")
        }
    })
}

/// The standard Slack API, with `users.info` answered by `lookup`.
async fn api_with_lookup(
    socket_url: String,
    lookup: impl Fn(usize) -> Response + Send + Sync + 'static,
) -> MockApi {
    MockApi::start(move |request, prior| match request.method() {
        "auth.test" => Response::json(json!({"ok": true, "user_id": BOT_USER})),
        "apps.connections.open" => Response::json(json!({"ok": true, "url": socket_url})),
        "users.info" => lookup(prior),
        "chat.postMessage" => {
            Response::json(json!({"ok": true, "ts": format!("1700000099.{prior:06}")}))
        }
        _ => Response::json(json!({"ok": true})),
    })
    .await
}

fn alice_profile() -> Response {
    Response::json(json!({"ok": true, "user": {"id": OWNER, "profile": {
        "display_name": "", "real_name": "Alice\nRivera\u{202E}"
    }}}))
}

async fn start(
    channel: SlackChannel,
) -> (
    mpsc::Receiver<ChannelEvent>,
    CancellationToken,
    tokio::task::JoinHandle<Result<(), aidaemon_channels::ChannelError>>,
) {
    let (tx, rx) = mpsc::channel(16);
    let shutdown = CancellationToken::new();
    let token = shutdown.clone();
    let task = tokio::spawn(async move { channel.run(tx, token).await });
    (rx, shutdown, task)
}

async fn next_sender(rx: &mut mpsc::Receiver<ChannelEvent>) -> aidaemon_channels::Sender {
    match timeout(Duration::from_secs(5), rx.recv()).await.unwrap() {
        Some(ChannelEvent::Message(m)) => m.sender,
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn names_come_from_users_info_cleaned_and_cached() {
    let socket = MockSocket::start(vec![vec![
        hello(),
        dm("e1", "1700000000.000001", OWNER, "one"),
        dm("e2", "1700000000.000002", OWNER, "two"),
    ]])
    .await;
    let api = api_with_lookup(socket.url.clone(), |_| alice_profile()).await;
    let (mut rx, shutdown, task) = start(channel(&api)).await;
    for _ in 0..2 {
        let sender = next_sender(&mut rx).await;
        assert_eq!(sender.id, OWNER, "identity stays the platform id");
        assert_eq!(sender.display_name.as_deref(), Some("Alice Rivera"));
    }
    assert_eq!(api.calls("users.info").len(), 1, "looked up once");
    shutdown.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn a_failed_lookup_falls_back_to_the_id_without_losing_the_message() {
    let socket = MockSocket::start(vec![vec![
        hello(),
        dm("e1", "1700000000.000001", OWNER, "one"),
        dm("e2", "1700000000.000002", OWNER, "two"),
    ]])
    .await;
    let api = api_with_lookup(socket.url.clone(), |_| {
        Response::json(json!({"ok": false, "error": "user_not_found"}))
    })
    .await;
    let (mut rx, shutdown, task) = start(channel(&api)).await;
    for _ in 0..2 {
        let sender = next_sender(&mut rx).await;
        assert_eq!(sender.display_name.as_deref(), Some(OWNER));
    }
    assert_eq!(api.calls("users.info").len(), 1, "failure remembered");
    shutdown.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn the_trait_lookup_is_cached() {
    let api = api_with_lookup("ws://unused".to_string(), |_| alice_profile()).await;
    let slack = channel(&api);
    for _ in 0..2 {
        assert_eq!(
            slack.display_name(OWNER).await.unwrap().as_deref(),
            Some("Alice Rivera")
        );
    }
    assert_eq!(api.calls("users.info").len(), 1);
}

struct Hosted {
    shutdown: CancellationToken,
    task: tokio::task::JoinHandle<()>,
    provider: Arc<ScriptedProvider>,
}

fn host(slack: SlackChannel, reply: &str) -> Hosted {
    let provider = Arc::new(ScriptedProvider::new(vec![text(reply)]));
    let agent = Agent::builder("assistant")
        .allow_unguarded()
        .provider(provider.clone())
        .model("synthetic-model")
        .build()
        .unwrap();
    let runner = ChannelRunner::builder(agent)
        .channel(Arc::new(slack))
        .access(AccessPolicy::new().owner("slack", OWNER))
        .command("uptime", "How long", |context: CommandContext| async move {
            CommandReply::text(format!("Up [{}]", context.args))
        })
        .build()
        .unwrap();
    let shutdown = CancellationToken::new();
    let token = shutdown.clone();
    let task = tokio::spawn(async move { runner.run(token).await });
    Hosted {
        shutdown,
        task,
        provider,
    }
}

async fn wait_calls(api: &MockApi, method: &str, n: usize) -> Vec<Request> {
    for _ in 0..500 {
        let calls = api.calls(method);
        if calls.len() >= n {
            return calls;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{method}: {:?}", api.calls(method));
}

#[tokio::test]
async fn commands_arrive_by_their_own_event_and_keep_the_access_check() {
    // The response URLs point at the API mock, which hands out the socket's
    // address, so one of them has to learn the other's after starting.
    let socket_url = Arc::new(std::sync::Mutex::new(String::new()));
    let handed_out = socket_url.clone();
    let api = MockApi::start(move |request, _| match request.method() {
        "auth.test" => Response::json(json!({"ok": true, "user_id": BOT_USER})),
        "apps.connections.open" => {
            Response::json(json!({"ok": true, "url": handed_out.lock().unwrap().clone()}))
        }
        _ => Response::json(json!({"ok": true})),
    })
    .await;
    let socket = MockSocket::start(vec![vec![
        hello(),
        slash("c1", "/uptime", "U000STRANGER", &api.base),
        slash("c2", "/uptime", OWNER, &api.base),
    ]])
    .await;
    *socket_url.lock().unwrap() = socket.url.clone();
    let hosted = host(channel(&api), "unused");
    let answers = wait_calls(&api, "c2", 1).await;
    assert_eq!(answers[0].json()["text"], "Up [three days]");
    let refused = wait_calls(&api, "c1", 1).await;
    assert_eq!(refused[0].json()["text"], DEFAULT_REFUSAL);
    assert!(hosted.provider.requests().is_empty());
    hosted.shutdown.cancel();
    hosted.task.await.unwrap();
}

#[tokio::test]
async fn the_runner_rewrites_latex_unless_the_channel_opts_out() {
    for (opt_out, expected) in [(false, "So x² and `$y^2$`"), (true, "So $x^2$ and `$y^2$`")] {
        let socket = MockSocket::start(vec![vec![
            hello(),
            dm("e1", "1700000000.000001", OWNER, "math?"),
        ]])
        .await;
        let api = api_with_lookup(socket.url.clone(), |_| alice_profile()).await;
        let hosted = host(
            channel(&api).with_latex_stripping(!opt_out),
            "So $x^2$ and `$y^2$`",
        );
        let posted = wait_calls(&api, "chat.postMessage", 1).await;
        assert!(
            posted[0].body.contains(expected),
            "{expected}: {}",
            posted[0].body
        );
        hosted.shutdown.cancel();
        hosted.task.await.unwrap();
    }
}

#[tokio::test]
async fn slack_has_no_menu_to_set() {
    let api = api_with_lookup("ws://unused".to_string(), |_| alice_profile()).await;
    let before = api.requests().len();
    channel(&api)
        .register_commands(&[aidaemon_channels::CommandSpec::new("uptime", "How long")])
        .await
        .unwrap();
    assert_eq!(api.requests().len(), before, "configured in the manifest");
}
