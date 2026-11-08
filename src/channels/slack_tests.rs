//! The Slack bot's message path, against a local mock of the Slack Web
//! API. Envelopes are handed to the channel as the Socket Mode connection
//! delivers them.

use std::time::Duration;

use serde_json::{json, Value};

use super::*;
use crate::channels::mock_api::{MockApi, Recorded};
use crate::runtime_ports::ChannelAgentRuntime;
use crate::testing::{setup_test_agent, MockProvider};

const BOT: &str = "U0SYNTHBOT";
const OWNER: &str = "U0SYNTHOWNER";
const STRANGER: &str = "U0SYNTHOTHER";

fn respond(request: &Recorded) -> Value {
    if request.path.starts_with("/users.info") {
        return json!({ "ok": true, "user": { "name": "alice", "profile": { "display_name": "Alice Rivera" } } });
    }
    json!({ "ok": true, "ts": "1700000000.000900" })
}

async fn channel(
    api: &MockApi,
    provider: MockProvider,
) -> (Arc<SlackChannel>, crate::testing::TestHarness) {
    let mut harness = setup_test_agent(provider).await.unwrap();
    let agent: Arc<dyn ChannelAgentRuntime> = Arc::new(std::mem::replace(
        &mut harness.agent,
        setup_test_agent(MockProvider::new()).await.unwrap().agent,
    ));
    let inbox = tempfile::tempdir().unwrap().keep();
    let channel = SlackChannel::new(
        "xapp-synthetic",
        "xoxb-synthetic",
        vec![OWNER.to_string()],
        vec![OWNER.to_string()],
        Vec::new(),
        false,
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
            transports: Default::default(),
        },
    )
    .with_api(&api.base, BOT);
    (Arc::new(channel), harness)
}

fn message(from: &str, channel: &str, channel_type: &str, text: &str, ts: &str) -> Value {
    json!({
        "type": "events_api",
        "payload": { "event": {
            "type": "message",
            "user": from,
            "channel": channel,
            "channel_type": channel_type,
            "text": text,
            "ts": ts,
        }}
    })
}

fn posted(request: &Recorded, text: &str) -> bool {
    request.path == "/chat.postMessage" && request.json()["text"] == text
}

#[tokio::test]
async fn an_owner_dm_runs_the_agent_and_the_reply_is_posted() {
    let api = MockApi::start(respond).await;
    let provider =
        MockProvider::with_responses(vec![MockProvider::text_response("Hello from the agent.")]);
    let (channel, _harness) = channel(&api, provider).await;

    channel
        .handle_envelope(message(
            OWNER,
            "D0SYNTH",
            "im",
            "hi there",
            "1700000000.000100",
        ))
        .await;

    let reply = api.wait_for(|r| posted(r, "Hello from the agent.")).await;
    assert_eq!(reply.json()["channel"], "D0SYNTH");
    // The hourglass marks the work and is taken off when it is done.
    let added = api.wait_for(|r| r.path == "/reactions.add").await;
    assert_eq!(added.json()["name"], "hourglass_flowing_sand");
    assert_eq!(added.json()["timestamp"], "1700000000.000100");
    api.wait_for(|r| r.path == "/reactions.remove").await;
}

#[tokio::test]
async fn group_chatter_that_does_not_address_the_bot_is_ignored() {
    let api = MockApi::start(respond).await;
    let (channel, harness) = channel(&api, MockProvider::new()).await;
    channel
        .handle_envelope(message(
            STRANGER,
            "C0SYNTH",
            "channel",
            "lunch?",
            "1700000000.000200",
        ))
        .await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(harness.provider.call_count().await, 0);
    assert!(!api.requests().iter().any(|r| r.path == "/chat.postMessage"));
}

#[tokio::test]
async fn a_cancel_word_with_nothing_running_says_so() {
    let api = MockApi::start(respond).await;
    let (channel, harness) = channel(&api, MockProvider::new()).await;
    channel
        .handle_envelope(message(OWNER, "D0SYNTH", "im", "stop", "1700000000.000300"))
        .await;
    api.wait_for(|r| posted(r, "No running task to cancel."))
        .await;
    assert_eq!(harness.provider.call_count().await, 0);
}

#[tokio::test]
async fn an_approval_card_is_answered_by_an_owner_press_and_shows_its_outcome() {
    let api = MockApi::start(respond).await;
    let (channel, _harness) = channel(&api, MockProvider::new()).await;
    let asking = channel.clone();
    let asked = tokio::spawn(async move {
        asking
            .request_approval(
                "slack:D0SYNTH",
                "rm -rf build/",
                RiskLevel::High,
                &[],
                PermissionMode::Default,
                false,
            )
            .await
    });
    let card = api
        .wait_for(|r| r.path == "/chat.postMessage" && r.json()["blocks"].is_array())
        .await;
    assert_eq!(card.method, "POST");
    let buttons = card.json()["blocks"][1]["elements"].clone();
    let labels: Vec<String> = buttons
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["text"]["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(labels, ["Allow Once", "Allow Always", "Deny"]);
    let press = |from: &str| {
        json!({
            "type": "interactive",
            "payload": {
                "type": "block_actions",
                "user": { "id": from },
                "channel": { "id": "D0SYNTH" },
                "message": { "ts": "1700000000.000900" },
                "actions": [ buttons[0].clone() ],
            }
        })
    };

    // Only an owner may answer.
    channel.handle_envelope(press(STRANGER)).await;
    assert!(!asked.is_finished());
    channel.handle_envelope(press(OWNER)).await;
    let answer = tokio::time::timeout(Duration::from_secs(5), asked)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(answer, ApprovalResponse::AllowOnce));
    let updated = api.wait_for(|r| r.path == "/chat.update").await;
    assert!(updated.json()["text"]
        .as_str()
        .unwrap()
        .contains("Allowed this once"));
}
