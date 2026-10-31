mod support;

use aidaemon_channel_slack::SlackChannel;
use aidaemon_channels::{
    ApprovalChoice, ApprovalPrompt, Channel, ChannelError, Conversation, ConversationKind,
    MessageRef, OutboundMessage,
};
use serde_json::json;
use support::*;

fn channel(api: &MockApi) -> SlackChannel {
    SlackChannel::new(BOT_TOKEN, APP_TOKEN).with_api_base(&api.base)
}

fn thread() -> Conversation {
    Conversation {
        id: "C000SYNTH:1700000000.000100".into(),
        kind: ConversationKind::Thread,
    }
}

fn group() -> Conversation {
    Conversation {
        id: "C000SYNTH".into(),
        kind: ConversationKind::Group,
    }
}

async fn ok_api() -> MockApi {
    MockApi::start(|_, prior| {
        Response::json(json!({ "ok": true, "ts": format!("1700000001.{prior:06}") }))
    })
    .await
}

#[tokio::test]
async fn capabilities_and_name() {
    let slack = SlackChannel::new(BOT_TOKEN, APP_TOKEN);
    assert_eq!(slack.name(), "slack");
    let caps = slack.capabilities();
    assert!(caps.edit_messages && caps.buttons && caps.threads);
    assert!(!caps.typing_indicator);
    assert_eq!(caps.max_message_len, 4000);
    assert_eq!(slack.with_name("slack-work").name(), "slack-work");
}

#[tokio::test]
async fn send_converts_markdown_and_carries_thread_ts() {
    let api = ok_api().await;
    let slack = channel(&api);
    let r = slack
        .send(
            &thread(),
            &OutboundMessage::text("**Done** & see [log](https://example.com/log)"),
        )
        .await
        .unwrap();
    assert_eq!(r.conversation_id, "C000SYNTH:1700000000.000100");
    assert_eq!(r.message_id, "1700000001.000000");

    let calls = api.calls("chat.postMessage");
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].header("authorization"),
        Some("Bearer xoxb-synthetic")
    );
    let body = calls[0].json();
    assert_eq!(body["channel"], "C000SYNTH");
    assert_eq!(body["thread_ts"], "1700000000.000100");
    let expected = "*Done* &amp; see <https://example.com/log|log>";
    assert_eq!(body["text"], expected);
    assert_eq!(body["blocks"][0]["type"], "section");
    assert_eq!(body["blocks"][0]["text"]["type"], "mrkdwn");
    assert_eq!(body["blocks"][0]["text"]["text"], expected);
}

#[tokio::test]
async fn send_to_group_has_no_thread_ts() {
    let api = ok_api().await;
    channel(&api)
        .send(&group(), &OutboundMessage::text("hello"))
        .await
        .unwrap();
    let body = api.calls("chat.postMessage")[0].json();
    assert_eq!(body["channel"], "C000SYNTH");
    assert!(body.get("thread_ts").is_none());
}

#[tokio::test]
async fn send_splits_long_messages() {
    let api = ok_api().await;
    let paragraph = "lorem ipsum ".repeat(200); // 2400 chars
    let text = vec![paragraph; 3].join("\n\n");
    let r = channel(&api)
        .send(&thread(), &OutboundMessage::text(text))
        .await
        .unwrap();
    let calls = api.calls("chat.postMessage");
    assert_eq!(calls.len(), 3);
    for call in &calls {
        let body = call.json();
        assert!(body["text"].as_str().unwrap().chars().count() <= 3000);
        assert_eq!(body["thread_ts"], "1700000000.000100");
    }
    // The ref points at the last part.
    assert_eq!(r.message_id, "1700000001.000002");
}

#[tokio::test]
async fn rate_limit_is_retried_once() {
    let api = MockApi::start(|_, prior| {
        if prior == 0 {
            Response::json(json!({ "ok": false, "error": "ratelimited" }))
                .status(429)
                .header("Retry-After", "0")
        } else {
            Response::json(json!({ "ok": true, "ts": "1700000001.000001" }))
        }
    })
    .await;
    let r = channel(&api)
        .send(&group(), &OutboundMessage::text("hi"))
        .await
        .unwrap();
    assert_eq!(r.message_id, "1700000001.000001");
    assert_eq!(api.calls("chat.postMessage").len(), 2);
}

#[tokio::test]
async fn persistent_rate_limit_returns_rate_limited() {
    // Slack can also report rate limiting in a 200 body.
    let api = MockApi::start(|_, _| {
        Response::json(json!({ "ok": false, "error": "ratelimited" })).header("Retry-After", "0")
    })
    .await;
    let err = channel(&api)
        .send(&group(), &OutboundMessage::text("hi"))
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            ChannelError::RateLimited {
                retry_after_secs: Some(0)
            }
        ),
        "{err:?}"
    );
    assert_eq!(api.calls("chat.postMessage").len(), 2);
}

#[tokio::test]
async fn invalid_auth_on_send_is_auth_error() {
    let api =
        MockApi::start(|_, _| Response::json(json!({ "ok": false, "error": "invalid_auth" })))
            .await;
    let err = channel(&api)
        .send(&group(), &OutboundMessage::text("hi"))
        .await
        .unwrap_err();
    assert!(matches!(err, ChannelError::Auth(_)), "{err:?}");
}

#[tokio::test]
async fn other_api_errors_are_reported() {
    let api =
        MockApi::start(|_, _| Response::json(json!({ "ok": false, "error": "channel_not_found" })))
            .await;
    let err = channel(&api)
        .send(&group(), &OutboundMessage::text("hi"))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("channel_not_found"), "{err}");
}

#[tokio::test]
async fn edit_calls_chat_update() {
    let api = ok_api().await;
    let message = MessageRef {
        conversation_id: "C000SYNTH:1700000000.000100".into(),
        message_id: "1700000001.000000".into(),
    };
    channel(&api)
        .edit(&message, "**updated** <text>")
        .await
        .unwrap();
    let calls = api.calls("chat.update");
    assert_eq!(calls.len(), 1);
    let body = calls[0].json();
    assert_eq!(body["channel"], "C000SYNTH");
    assert_eq!(body["ts"], "1700000001.000000");
    assert_eq!(body["text"], "*updated* &lt;text&gt;");
}

#[tokio::test]
async fn ask_renders_buttons_with_encoded_values() {
    let api = ok_api().await;
    let prompt = ApprovalPrompt {
        id: "prompt-42".into(),
        conversation: thread(),
        text: "Run `rm -rf build/`?".into(),
        choices: vec![
            ApprovalChoice {
                id: "approve".into(),
                label: "Approve".into(),
            },
            ApprovalChoice {
                id: "deny".into(),
                label: "Deny".into(),
            },
        ],
    };
    let r = channel(&api).ask(&prompt).await.unwrap();
    assert_eq!(r.conversation_id, "C000SYNTH:1700000000.000100");
    assert_eq!(r.message_id, "1700000001.000000");

    let body = api.calls("chat.postMessage")[0].json();
    assert_eq!(body["thread_ts"], "1700000000.000100");
    assert_eq!(body["blocks"][0]["type"], "section");
    assert_eq!(body["blocks"][0]["text"]["text"], "Run `rm -rf build/`?");
    let actions = &body["blocks"][1];
    assert_eq!(actions["type"], "actions");
    let buttons = actions["elements"].as_array().unwrap();
    assert_eq!(buttons.len(), 2);
    assert_eq!(buttons[0]["style"], "primary");
    assert!(buttons[1].get("style").is_none());
    assert_eq!(buttons[0]["text"]["text"], "Approve");
    assert_eq!(buttons[1]["text"]["text"], "Deny");
    for (button, choice) in buttons.iter().zip(["approve", "deny"]) {
        assert!(button["action_id"]
            .as_str()
            .unwrap()
            .starts_with("aidaemon_approval"));
        let value: serde_json::Value =
            serde_json::from_str(button["value"].as_str().unwrap()).unwrap();
        assert_eq!(value, json!({ "p": "prompt-42", "c": choice }));
    }
}
