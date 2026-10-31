mod common;

use aidaemon_channel_discord::decode_custom_id;
use aidaemon_channels::{
    ApprovalChoice, ApprovalPrompt, Channel, ChannelError, Conversation, ConversationKind,
    MessageRef, OutboundMessage,
};
use common::*;
use serde_json::json;

fn conv() -> Conversation {
    Conversation {
        id: CHANNEL_ID.to_string(),
        kind: ConversationKind::Group,
    }
}

#[tokio::test]
async fn capabilities_and_name() {
    let http = MockHttp::start().await;
    let ch = channel(&http);
    assert_eq!(ch.name(), "discord");
    let caps = ch.capabilities();
    assert!(caps.edit_messages && caps.buttons && caps.typing_indicator && caps.threads);
    assert_eq!(caps.max_message_len, 2000);
    assert_eq!(
        channel(&http).with_name("discord_ops").name(),
        "discord_ops"
    );
}

#[tokio::test]
async fn send_splits_and_reopens_fences() {
    let http = MockHttp::start().await;
    let ch = channel(&http);
    let mut text = String::from("Here is the script:\n```python\n");
    for i in 0..150 {
        text.push_str(&format!("print('row {i:04} for Alice Rivera')\n"));
    }
    text.push_str("```\nDone.");
    let msg = OutboundMessage {
        text,
        reply_to: Some("100000000000000102".into()),
    };
    let r = ch.send(&conv(), &msg).await.unwrap();

    let reqs = http.requests();
    assert!(reqs.len() >= 2, "expected a split, got {}", reqs.len());
    for (i, req) in reqs.iter().enumerate() {
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, format!("/channels/{CHANNEL_ID}/messages"));
        assert_eq!(
            req.headers.get("authorization").map(String::as_str),
            Some("Bot synthetic.discord.token")
        );
        let content = req.body["content"].as_str().unwrap();
        assert!(content.chars().count() <= 2000);
        assert_eq!(content.matches("```").count() % 2, 0, "unbalanced fences");
        assert_eq!(req.body["allowed_mentions"], json!({"parse": []}));
        if i == 0 {
            assert_eq!(
                req.body["message_reference"]["message_id"],
                "100000000000000102"
            );
        } else {
            assert!(req.body.get("message_reference").is_none());
            assert!(content.starts_with("```python\n"));
        }
    }
    assert!(reqs.last().unwrap().body["content"]
        .as_str()
        .unwrap()
        .ends_with("Done."));
    // The returned ref is the last part's.
    assert_eq!(r.conversation_id, CHANNEL_ID);
    assert_eq!(
        r.message_id,
        format!("{}", 100000000000001000u64 + reqs.len() as u64)
    );
}

#[tokio::test]
async fn send_short_message_has_allowed_mentions() {
    let http = MockHttp::start().await;
    let ch = channel(&http);
    ch.send(&conv(), &OutboundMessage::text("hi @everyone"))
        .await
        .unwrap();
    let reqs = http.requests();
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].body["content"], "hi @everyone");
    assert_eq!(reqs[0].body["allowed_mentions"], json!({"parse": []}));
}

#[tokio::test]
async fn rate_limit_is_retried_once() {
    let http = MockHttp::start().await;
    http.push(
        429,
        json!({"message": "You are being rate limited.", "retry_after": 0.05, "global": false}),
    );
    let ch = channel(&http);
    let r = ch
        .send(&conv(), &OutboundMessage::text("hello"))
        .await
        .unwrap();
    assert_eq!(http.requests().len(), 2);
    assert!(!r.message_id.is_empty());
}

#[tokio::test]
async fn second_rate_limit_surfaces() {
    let http = MockHttp::start().await;
    for _ in 0..2 {
        http.push(
            429,
            json!({"message": "You are being rate limited.", "retry_after": 0.05, "global": false}),
        );
    }
    let ch = channel(&http);
    let err = ch
        .send(&conv(), &OutboundMessage::text("hello"))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        ChannelError::RateLimited {
            retry_after_secs: Some(1)
        }
    ));
    assert_eq!(http.requests().len(), 2);
}

#[tokio::test]
async fn unauthorized_is_auth_error() {
    let http = MockHttp::start().await;
    http.push(401, json!({"message": "401: Unauthorized", "code": 0}));
    let err = channel(&http)
        .send(&conv(), &OutboundMessage::text("hello"))
        .await
        .unwrap_err();
    assert!(matches!(err, ChannelError::Auth(_)));
}

#[tokio::test]
async fn ask_renders_buttons_with_custom_ids() {
    let http = MockHttp::start().await;
    let ch = channel(&http);
    let prompt = ApprovalPrompt {
        id: "prompt-7".into(),
        conversation: conv(),
        text: "Run `rm -rf build/`?".into(),
        choices: vec![
            ApprovalChoice {
                id: "once".into(),
                label: "Allow once".into(),
            },
            ApprovalChoice {
                id: "no".into(),
                label: "Not now".into(),
            },
            ApprovalChoice {
                id: "always".into(),
                label: "Always".into(),
            },
        ],
    };
    let r = ch.ask(&prompt).await.unwrap();
    assert_eq!(r.conversation_id, CHANNEL_ID);

    let reqs = http.requests();
    assert_eq!(reqs.len(), 1);
    let body = &reqs[0].body;
    assert_eq!(body["content"], "Run `rm -rf build/`?");
    let rows = body["components"].as_array().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["type"], 1);
    let buttons = rows[0]["components"].as_array().unwrap();
    let styles: Vec<u64> = buttons
        .iter()
        .map(|b| b["style"].as_u64().unwrap())
        .collect();
    // By position, not label text: "Not now" is still danger.
    assert_eq!(styles, vec![3, 4, 2]);
    for (b, choice) in buttons.iter().zip(&prompt.choices) {
        assert_eq!(b["type"], 2);
        assert_eq!(b["label"], choice.label.as_str());
        let cid = b["custom_id"].as_str().unwrap();
        assert!(cid.len() <= 100);
        assert_eq!(
            decode_custom_id(cid),
            Some(("prompt-7".to_string(), choice.id.clone()))
        );
    }
}

#[tokio::test]
async fn ask_rejects_ids_too_long_for_custom_id() {
    let http = MockHttp::start().await;
    let prompt = ApprovalPrompt {
        id: "p".repeat(120),
        conversation: conv(),
        text: "Approve?".into(),
        choices: vec![ApprovalChoice {
            id: "yes".into(),
            label: "Yes".into(),
        }],
    };
    assert!(channel(&http).ask(&prompt).await.is_err());
    assert!(http.requests().is_empty());
}

#[tokio::test]
async fn edit_patches_message_and_clears_buttons() {
    let http = MockHttp::start().await;
    let ch = channel(&http);
    let msg = MessageRef {
        conversation_id: CHANNEL_ID.into(),
        message_id: "100000000000000700".into(),
    };
    ch.edit(&msg, "Approved by Alice Rivera").await.unwrap();
    let reqs = http.requests();
    assert_eq!(reqs[0].method, "PATCH");
    assert_eq!(
        reqs[0].path,
        format!("/channels/{CHANNEL_ID}/messages/100000000000000700")
    );
    assert_eq!(reqs[0].body["content"], "Approved by Alice Rivera");
    assert_eq!(reqs[0].body["components"], json!([]));
}

#[tokio::test]
async fn typing_posts_to_typing_endpoint() {
    let http = MockHttp::start().await;
    http.push(204, json!(null));
    channel(&http).typing(&conv()).await.unwrap();
    let reqs = http.requests();
    assert_eq!(reqs[0].method, "POST");
    assert_eq!(reqs[0].path, format!("/channels/{CHANNEL_ID}/typing"));
}
