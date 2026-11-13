//! The Slack bot's message path, against a local mock of the Slack Web
//! API. Envelopes are handed to the channel as the Socket Mode connection
//! delivers them.

use std::time::Duration;

use serde_json::{json, Value};

use super::*;
use crate::channels::mock_api::{MockApi, Recorded};
use crate::runtime_ports::DaemonAgentRuntime;
use crate::tasks::TaskRegistry;
use crate::testing::{setup_test_agent, MockProvider};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::traits::HubChannel;
use crate::types::ApprovalResponse;

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
    let agent: Arc<dyn DaemonAgentRuntime> = Arc::new(std::mem::replace(
        &mut harness.agent,
        setup_test_agent(MockProvider::new()).await.unwrap().agent,
    ));
    let inbox = tempfile::tempdir().unwrap().keep();
    let channel = new_slack_bot(
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
            watch: Default::default(),
        },
    )
    .with_api(&api.base, BOT, MockApi::client());
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
    api.wait_for(|r| posted(r, "Nothing is running in this chat right now."))
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
    assert_eq!(labels, ["Allow", "Always allow", "Deny"]);
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

/// A typed card renders as Telegram's and Discord's do, in Slack's blocks: the
/// plain headline and why the owner is asked in the main section, the exact
/// command under them in a secondary context block, then the buttons.
#[tokio::test]
async fn a_typed_approval_card_puts_the_command_under_the_summary() {
    let api = MockApi::start(respond).await;
    let (channel, _harness) = channel(&api, MockProvider::new()).await;
    let card = aidaemon_core::ApprovalCard {
        asked_by: Some("Drupal install".to_string()),
        action: aidaemon_core::CardAction::Command {
            line: "ls -la claude-drupal && ddev describe <site>".to_string(),
            title: None,
            looks_inside: vec!["claude-drupal".to_string()],
        },
        reason: Some(aidaemon_core::AskReason::ReadOutsideContent {
            check_says: Some("This isn't part of your request.".to_string()),
        }),
        impact: None,
    };
    let asking = channel.clone();
    let asked = tokio::spawn(async move {
        asking
            .request_approval_ask(
                "slack:D0SYNTH",
                &aidaemon_channels::approval_render::ApprovalAsk {
                    command: "summary a channel without cards would show",
                    card: Some(&card),
                    risk_level: RiskLevel::High,
                    warnings: &[],
                    permission_mode: PermissionMode::Default,
                    one_time_only: true,
                },
            )
            .await
    });
    let posted = api
        .wait_for(|r| r.path == "/chat.postMessage" && r.json()["blocks"].is_array())
        .await
        .json();
    asked.abort();
    let blocks = posted["blocks"].as_array().unwrap();
    assert_eq!(blocks.len(), 3, "{posted}");

    assert_eq!(blocks[0]["type"], "section");
    let summary = blocks[0]["text"]["text"].as_str().unwrap();
    assert!(summary.starts_with(
        "*Drupal install wants to look inside your claude-drupal folder.*\n\
         It read content from outside, like a web page, so I'm checking with you"
    ));
    assert!(summary.contains("_Safety check: This isn't part of your request._"));
    for gone in ["```", "ls -la", "summary a channel", "details"] {
        assert!(!summary.contains(gone), "{gone}: {summary}");
    }
    assert_eq!(posted["text"], summary);

    assert_eq!(blocks[1]["type"], "context");
    assert_eq!(
        blocks[1]["elements"][0],
        json!({
            "type": "mrkdwn",
            "text": "```\nls -la claude-drupal &amp;&amp; ddev describe &lt;site&gt;\n```",
        })
    );

    assert_eq!(blocks[2]["type"], "actions");
    let labels: Vec<&str> = blocks[2]["elements"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["text"]["text"].as_str().unwrap())
        .collect();
    assert_eq!(labels, ["Allow", "Deny"]);
}
