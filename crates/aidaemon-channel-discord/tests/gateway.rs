mod common;

use std::sync::Arc;
use std::time::Duration;

use aidaemon_channel_discord::{encode_custom_id, intents, DiscordChannel};
use aidaemon_channels::{Channel, ChannelError, ChannelEvent, ConversationKind};
use common::*;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

struct Running {
    channel: Arc<DiscordChannel>,
    events: mpsc::Receiver<ChannelEvent>,
    shutdown: CancellationToken,
    task: JoinHandle<Result<(), ChannelError>>,
}

fn start(channel: DiscordChannel) -> Running {
    let channel = Arc::new(channel);
    let (tx, events) = mpsc::channel(16);
    let shutdown = CancellationToken::new();
    let (c, s) = (channel.clone(), shutdown.clone());
    let task = tokio::spawn(async move { c.run(tx, s).await });
    Running {
        channel,
        events,
        shutdown,
        task,
    }
}

impl Running {
    async fn next_event(&mut self) -> ChannelEvent {
        tokio::time::timeout(Duration::from_secs(5), self.events.recv())
            .await
            .expect("no event")
            .expect("channel closed")
    }

    async fn finish(self) -> Result<(), ChannelError> {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(5), self.task)
            .await
            .expect("run did not stop")
            .unwrap()
    }

    async fn result(self) -> Result<(), ChannelError> {
        tokio::time::timeout(Duration::from_secs(5), self.task)
            .await
            .expect("run did not return")
            .unwrap()
    }
}

fn alice() -> Value {
    json!({"id": "100000000000000001", "username": "alice", "global_name": "Alice Rivera"})
}

#[tokio::test]
async fn hello_sends_identify_then_heartbeats() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let run = start(channel(&http).with_gateway_url(gw.url.clone()));

    let mut conn = gw.accept().await;
    conn.hello(150).await;
    let identify = conn.recv_any().await;
    assert_eq!(identify["op"], 2);
    assert_eq!(identify["d"]["token"], TOKEN);
    assert_eq!(identify["d"]["intents"], intents::DEFAULT);
    assert_eq!(
        intents::DEFAULT,
        intents::GUILD_MESSAGES | intents::DIRECT_MESSAGES | intents::MESSAGE_CONTENT
    );

    for _ in 0..2 {
        let hb = conn.recv_any().await;
        assert_eq!(hb["op"], 1);
        assert!(hb["d"].is_null(), "no sequence received yet");
        conn.send(json!({"op": 11})).await;
    }
    run.finish().await.unwrap();
}

#[tokio::test]
async fn ready_stores_bot_id_and_messages_are_mapped() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let mut run = start(channel(&http).with_gateway_url(gw.url.clone()));

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    assert_eq!(conn.recv().await["op"], 2);
    conn.ready("synthetic-session-1", &gw.url).await;

    // A bot author is skipped; then an addressed guild message arrives.
    conn.dispatch(
        2,
        "MESSAGE_CREATE",
        json!({"id": "100000000000000101", "type": 0, "channel_id": CHANNEL_ID,
               "guild_id": GUILD_ID, "content": "beep",
               "author": {"id": "100000000000000005", "bot": true}, "mentions": []}),
    )
    .await;
    conn.dispatch(
        3,
        "MESSAGE_CREATE",
        json!({"id": "100000000000000102", "type": 0, "channel_id": CHANNEL_ID,
               "guild_id": GUILD_ID, "content": format!("<@{BOT_ID}> status?"),
               "author": alice(), "mentions": [{"id": BOT_ID, "bot": true}]}),
    )
    .await;
    match run.next_event().await {
        ChannelEvent::Message(m) => {
            assert_eq!(m.message_id, "100000000000000102");
            assert_eq!(m.text, "status?");
            assert!(m.addressed);
            assert_eq!(m.conversation.kind, ConversationKind::Group);
            assert_eq!(m.conversation.id, CHANNEL_ID);
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(run.channel.bot_user_id().as_deref(), Some(BOT_ID));

    // Thread learned from THREAD_CREATE.
    conn.dispatch(
        4,
        "THREAD_CREATE",
        json!({"id": "100000000000000210", "type": 11, "guild_id": GUILD_ID}),
    )
    .await;
    conn.dispatch(
        5,
        "MESSAGE_CREATE",
        json!({"id": "100000000000000103", "type": 0, "channel_id": "100000000000000210",
               "guild_id": GUILD_ID, "content": "in thread", "author": alice(), "mentions": []}),
    )
    .await;
    match run.next_event().await {
        ChannelEvent::Message(m) => {
            assert_eq!(m.conversation.kind, ConversationKind::Thread);
            assert!(!m.addressed);
        }
        other => panic!("unexpected {other:?}"),
    }
    run.finish().await.unwrap();
}

#[tokio::test]
async fn reconnect_op7_resumes_with_session_and_seq() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let resume_gw = MockGateway::start().await;
    let run = start(channel(&http).with_gateway_url(gw.url.clone()));

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    assert_eq!(conn.recv().await["op"], 2);
    conn.ready("synthetic-session-1", &resume_gw.url).await;
    conn.dispatch(7, "TYPING_START", json!({})).await;
    conn.send(json!({"op": 7, "d": null})).await;

    let mut conn2 = resume_gw.accept().await;
    conn2.hello(45_000).await;
    let resume = conn2.recv().await;
    assert_eq!(resume["op"], 6);
    assert_eq!(resume["d"]["session_id"], "synthetic-session-1");
    assert_eq!(resume["d"]["seq"], 7);
    assert_eq!(resume["d"]["token"], TOKEN);
    conn2.dispatch(8, "RESUMED", json!({})).await;
    run.finish().await.unwrap();
}

#[tokio::test]
async fn invalid_session_not_resumable_reidentifies() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let resume_gw = MockGateway::start().await;
    let run = start(channel(&http).with_gateway_url(gw.url.clone()));

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    assert_eq!(conn.recv().await["op"], 2);
    conn.ready("synthetic-session-1", &resume_gw.url).await;
    conn.send(json!({"op": 9, "d": false})).await;

    // Not resumable: back to the main gateway with a fresh Identify.
    let mut conn2 = gw.accept().await;
    conn2.hello(45_000).await;
    let identify = conn2.recv().await;
    assert_eq!(identify["op"], 2);
    run.finish().await.unwrap();
}

#[tokio::test]
async fn invalid_session_resumable_resumes() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let resume_gw = MockGateway::start().await;
    let run = start(channel(&http).with_gateway_url(gw.url.clone()));

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    conn.recv().await;
    conn.ready("synthetic-session-2", &resume_gw.url).await;
    conn.send(json!({"op": 9, "d": true})).await;

    let mut conn2 = resume_gw.accept().await;
    conn2.hello(45_000).await;
    let resume = conn2.recv().await;
    assert_eq!(resume["op"], 6);
    assert_eq!(resume["d"]["session_id"], "synthetic-session-2");
    run.finish().await.unwrap();
}

#[tokio::test]
async fn missing_heartbeat_ack_reconnects() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let run = start(channel(&http).with_gateway_url(gw.url.clone()));

    let mut conn = gw.accept().await;
    conn.hello(30).await;
    assert_eq!(conn.recv().await["op"], 2);
    conn.ready("synthetic-session-3", &gw.url).await;
    // Never acknowledge heartbeats: the client must drop the zombie
    // connection and resume on a new one.
    let mut conn2 = gw.accept().await;
    conn2.hello(45_000).await;
    let resume = conn2.recv().await;
    assert_eq!(resume["op"], 6);
    assert_eq!(resume["d"]["session_id"], "synthetic-session-3");
    run.finish().await.unwrap();
}

#[tokio::test]
async fn close_4004_is_auth_error() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let run = start(channel(&http).with_gateway_url(gw.url.clone()));

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    conn.recv().await;
    conn.close(4004).await;
    assert!(matches!(run.result().await, Err(ChannelError::Auth(_))));
}

#[tokio::test]
async fn close_4014_is_fatal_other() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let run = start(channel(&http).with_gateway_url(gw.url.clone()));

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    conn.recv().await;
    conn.close(4014).await;
    match run.result().await {
        Err(ChannelError::Other(msg)) => assert!(msg.contains("4014")),
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn other_close_codes_reconnect() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let run = start(channel(&http).with_gateway_url(gw.url.clone()));

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    conn.recv().await;
    conn.close(4000).await;
    let mut conn2 = gw.accept().await;
    conn2.hello(45_000).await;
    assert_eq!(conn2.recv().await["op"], 2, "no session yet, so identify");
    run.finish().await.unwrap();
}

#[tokio::test]
async fn interaction_becomes_approval_and_is_acknowledged() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let mut run = start(channel(&http).with_gateway_url(gw.url.clone()));

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    conn.recv().await;
    conn.ready("synthetic-session-1", &gw.url).await;
    conn.dispatch(
        2,
        "INTERACTION_CREATE",
        json!({
            "id": "100000000000000500",
            "type": 3,
            "token": "synthetic-interaction-token",
            "application_id": "100000000000000009",
            "guild_id": GUILD_ID,
            "channel_id": CHANNEL_ID,
            "member": {"user": alice()},
            "data": {"custom_id": encode_custom_id("prompt-7", "approve").unwrap(),
                     "component_type": 2}
        }),
    )
    .await;
    match run.next_event().await {
        ChannelEvent::Approval(a) => {
            assert_eq!(a.prompt_id, "prompt-7");
            assert_eq!(a.choice_id, "approve");
            assert_eq!(a.sender.id, "100000000000000001");
            assert_eq!(a.conversation.id, CHANNEL_ID);
        }
        other => panic!("unexpected {other:?}"),
    }
    let reqs = http.wait_for(1).await;
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].method, "POST");
    assert_eq!(
        reqs[0].path,
        "/interactions/100000000000000500/synthetic-interaction-token/callback"
    );
    assert_eq!(reqs[0].body, json!({"type": 6}));
    run.finish().await.unwrap();
}

#[tokio::test]
async fn gateway_url_is_fetched_when_not_configured() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    http.push(
        200,
        json!({"url": gw.url, "shards": 1, "session_start_limit": {"remaining": 1000}}),
    );
    let run = start(channel(&http));

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    assert_eq!(conn.recv().await["op"], 2);
    let reqs = http.requests();
    assert_eq!(reqs[0].method, "GET");
    assert_eq!(reqs[0].path, "/gateway/bot");
    assert_eq!(
        reqs[0].headers.get("authorization").map(String::as_str),
        Some("Bot synthetic.discord.token")
    );
    run.finish().await.unwrap();
}

#[tokio::test]
async fn gateway_lookup_401_is_auth_error() {
    let http = MockHttp::start().await;
    http.push(401, json!({"message": "401: Unauthorized", "code": 0}));
    let run = start(channel(&http));
    assert!(matches!(run.result().await, Err(ChannelError::Auth(_))));
}

#[tokio::test]
async fn dropped_receiver_stops_run() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let channel = Arc::new(channel(&http).with_gateway_url(gw.url.clone()));
    let (tx, rx) = mpsc::channel(1);
    let c = channel.clone();
    let task = tokio::spawn(async move { c.run(tx, CancellationToken::new()).await });

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    conn.recv().await;
    drop(rx);
    conn.dispatch(
        2,
        "MESSAGE_CREATE",
        json!({"id": "100000000000000104", "type": 0, "channel_id": CHANNEL_ID,
               "content": "hi", "author": alice()}),
    )
    .await;
    let res = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("run did not stop")
        .unwrap();
    assert!(res.is_ok());
}

#[tokio::test]
async fn slash_command_becomes_a_command_acked_at_once_and_answered_later() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let mut run = start(channel(&http).with_gateway_url(gw.url.clone()));

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    conn.recv().await;
    conn.ready("synthetic-session-1", &gw.url).await;
    conn.dispatch(
        2,
        "INTERACTION_CREATE",
        json!({
            "id": "100000000000000501",
            "type": 2,
            "token": "synthetic-command-token",
            "application_id": "100000000000000009",
            "guild_id": GUILD_ID,
            "channel_id": CHANNEL_ID,
            "member": {"user": alice()},
            "data": {"name": "model", "options": [{"name": "args", "type": 3, "value": " list "}]}
        }),
    )
    .await;
    let command = match run.next_event().await {
        ChannelEvent::Command(command) => command,
        other => panic!("unexpected {other:?}"),
    };
    assert_eq!(command.name, "model");
    assert_eq!(command.args, "list");
    assert_eq!(command.as_text(), "/model list");
    assert_eq!(command.sender.id, "100000000000000001");
    assert_eq!(command.conversation.id, CHANNEL_ID);
    let reqs = http.wait_for(1).await;
    assert_eq!(
        reqs[0].path,
        "/interactions/100000000000000501/synthetic-command-token/callback"
    );
    assert_eq!(reqs[0].body, json!({"type": 5}), "a deferred reply");

    run.channel
        .answer_command(&command, "Models: synthetic-a, synthetic-b")
        .await
        .unwrap();
    let reqs = http.wait_for(2).await;
    assert_eq!(reqs[1].method, "PATCH");
    assert_eq!(
        reqs[1].path,
        "/webhooks/100000000000000009/synthetic-command-token/messages/@original"
    );
    assert_eq!(reqs[1].body["content"], "Models: synthetic-a, synthetic-b");
    run.finish().await.unwrap();
}

#[tokio::test]
async fn commands_asked_for_before_ready_are_registered_on_ready() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let run = start(channel(&http).with_gateway_url(gw.url.clone()));
    let commands = vec![aidaemon_channels::CommandSpec {
        name: "status".to_string(),
        description: "Show what the agent is doing".to_string(),
    }];
    run.channel.register_commands(&commands).await.unwrap();
    assert!(http.requests().is_empty(), "nothing is sent before READY");

    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    conn.recv().await;
    conn.ready("synthetic-session-1", &gw.url).await;
    let reqs = http.wait_for(1).await;
    assert_eq!(reqs[0].method, "PUT");
    assert_eq!(reqs[0].path, format!("/applications/{BOT_ID}/commands"));
    assert_eq!(reqs[0].body[0]["name"], "status");
    assert_eq!(reqs[0].body[0]["options"][0]["name"], "args");

    // Once ready, a new menu is registered right away.
    run.channel.register_commands(&commands).await.unwrap();
    assert_eq!(http.wait_for(2).await[1].method, "PUT");
    run.finish().await.unwrap();
}

#[tokio::test]
async fn guild_commands_register_in_the_guild_and_ready_names_the_bot() {
    let http = MockHttp::start().await;
    let gw = MockGateway::start().await;
    let run = start(
        channel(&http)
            .with_gateway_url(gw.url.clone())
            .with_command_guild(GUILD_ID),
    );
    run.channel
        .register_commands(&[aidaemon_channels::CommandSpec {
            name: "help".to_string(),
            description: "Show available commands".to_string(),
        }])
        .await
        .unwrap();
    let mut conn = gw.accept().await;
    conn.hello(45_000).await;
    conn.recv().await;
    conn.ready("synthetic-session-1", &gw.url).await;
    let reqs = http.wait_for(1).await;
    assert_eq!(
        reqs[0].path,
        format!("/applications/{BOT_ID}/guilds/{GUILD_ID}/commands")
    );
    assert_eq!(run.channel.bot_username().as_deref(), Some("aidaemon"));
    run.finish().await.unwrap();
}
