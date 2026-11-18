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
    assert!(
        reqs.last().unwrap().body["content"]
            .as_str()
            .unwrap()
            .ends_with("Done.")
    );
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
    let prompt = ApprovalPrompt::new(
        "prompt-7",
        conv(),
        "Run `rm -rf build/`?",
        vec![
            ApprovalChoice {
                id: "once".to_string(),
                label: "Allow once".to_string(),
            },
            ApprovalChoice {
                id: "always".to_string(),
                label: "Always".to_string(),
            },
            ApprovalChoice {
                id: "no".to_string(),
                label: "Not now".to_string(),
            },
        ],
    );
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
    // By position, not label text: the last ("Not now") is danger.
    assert_eq!(styles, vec![3, 1, 4], "green first, red last, blue between");
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
    let prompt = ApprovalPrompt::new(
        "p".repeat(120),
        conv(),
        "Approve?",
        vec![ApprovalChoice {
            id: "yes".to_string(),
            label: "Yes".to_string(),
        }],
    );
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

#[tokio::test]
async fn delete_removes_the_message() {
    let http = MockHttp::start().await;
    http.push(204, json!(null));
    let ch = channel(&http);
    ch.delete(&MessageRef {
        conversation_id: CHANNEL_ID.to_string(),
        message_id: "100000000000000777".to_string(),
    })
    .await
    .unwrap();
    let reqs = http.requests();
    assert_eq!(reqs[0].method, "DELETE");
    assert_eq!(
        reqs[0].path,
        format!("/channels/{CHANNEL_ID}/messages/100000000000000777")
    );
}

#[tokio::test]
async fn send_file_uploads_the_file_with_its_caption() {
    let http = MockHttp::start().await;
    let ch = channel(&http);
    let sent = ch
        .send_file(
            &conv(),
            &aidaemon_channels::OutboundFile {
                name: "report.txt".to_string(),
                bytes: b"synthetic report for Acme Corp".to_vec(),
                mime_type: Some("text/plain".to_string()),
                caption: Some("The weekly report".to_string()),
            },
        )
        .await
        .unwrap();
    assert_eq!(sent.conversation_id, CHANNEL_ID);
    let reqs = http.requests();
    assert_eq!(reqs[0].method, "POST");
    assert_eq!(reqs[0].path, format!("/channels/{CHANNEL_ID}/messages"));
    assert!(reqs[0].headers["content-type"].starts_with("multipart/form-data"));
    let raw = String::from_utf8_lossy(&reqs[0].raw);
    assert!(raw.contains("filename=\"report.txt\""), "{raw}");
    assert!(raw.contains("synthetic report for Acme Corp"));
    assert!(raw.contains("name=\"payload_json\""));
    assert!(raw.contains("The weekly report"));
}

#[tokio::test]
async fn send_file_refuses_a_file_over_the_limit() {
    let http = MockHttp::start().await;
    let ch = channel(&http);
    let err = ch
        .send_file(
            &conv(),
            &aidaemon_channels::OutboundFile {
                name: "big.bin".to_string(),
                bytes: vec![0; aidaemon_channel_discord::MAX_FILE_BYTES + 1],
                mime_type: None,
                caption: None,
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, ChannelError::Other(_)));
    assert!(http.requests().is_empty());
}

#[tokio::test]
async fn download_fetches_the_attachment() {
    let http = MockHttp::start().await;
    http.push(200, json!("synthetic attachment bytes"));
    let ch = channel(&http);
    let bytes = ch
        .download(
            &aidaemon_channels::Attachment::new(
                "notes.txt",
                format!("{}/attachments/1/2/notes.txt", http.base),
            )
            .with_mime_type(Some("text/plain".to_string())),
        )
        .await
        .unwrap();
    assert_eq!(bytes, b"\"synthetic attachment bytes\"");
    assert_eq!(http.requests()[0].method, "GET");
}

#[tokio::test]
async fn display_name_prefers_the_global_name() {
    let http = MockHttp::start().await;
    http.push(
        200,
        json!({"id": "100000000000000001", "username": "alice", "global_name": "Alice Rivera"}),
    );
    let name = channel(&http)
        .display_name("100000000000000001")
        .await
        .unwrap();
    assert_eq!(name.as_deref(), Some("Alice Rivera"));
    assert_eq!(http.requests()[0].path, "/users/100000000000000001");
}

#[tokio::test]
async fn a_shorter_message_length_splits_sooner() {
    let http = MockHttp::start().await;
    let ch = channel(&http).with_max_message_len(100);
    assert_eq!(ch.capabilities().max_message_len, 100);
    ch.send(&conv(), &OutboundMessage::text("word ".repeat(60)))
        .await
        .unwrap();
    let reqs = http.requests();
    assert!(reqs.len() >= 3);
    for req in reqs {
        assert!(req.body["content"].as_str().unwrap().chars().count() <= 100);
    }
    assert_eq!(
        channel(&http)
            .with_max_message_len(99_999)
            .capabilities()
            .max_message_len,
        2000
    );
}

#[tokio::test]
async fn open_direct_creates_the_dm_channel() {
    let http = MockHttp::start().await;
    http.push(200, json!({"id": "100000000000000900", "type": 1}));
    let conversation = channel(&http)
        .open_direct("100000000000000001")
        .await
        .unwrap();
    assert_eq!(conversation.id, "100000000000000900");
    assert_eq!(conversation.kind, ConversationKind::Direct);
    let reqs = http.requests();
    assert_eq!(reqs[0].path, "/users/@me/channels");
    assert_eq!(reqs[0].body, json!({"recipient_id": "100000000000000001"}));
}

#[tokio::test]
async fn a_file_off_discord_is_not_fetched() {
    let http = MockHttp::start().await;
    let err = channel(&http)
        .download(&aidaemon_channels::Attachment::new(
            "notes.txt",
            "https://files.example.test/notes.txt",
        ))
        .await
        .unwrap_err();
    assert!(matches!(err, ChannelError::Other(_)), "{err}");
    assert!(http.requests().is_empty());
}

#[tokio::test]
async fn a_download_stops_at_the_limit() {
    let http = MockHttp::start().await;
    http.push(200, json!("0123456789"));
    let err = channel(&http)
        .download_limited(
            &aidaemon_channels::Attachment::new(
                "notes.txt",
                format!("{}/attachments/1/2/notes.txt", http.base),
            ),
            4,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, ChannelError::TooLarge { limit_bytes: 4 }),
        "{err}"
    );
}

#[tokio::test]
async fn a_download_sends_no_token() {
    let http = MockHttp::start().await;
    http.push(200, json!("bytes"));
    channel(&http)
        .download(&aidaemon_channels::Attachment::new(
            "notes.txt",
            format!("{}/attachments/1/2/notes.txt", http.base),
        ))
        .await
        .unwrap();
    assert!(!http.requests()[0].headers.contains_key("authorization"));
}

#[tokio::test]
async fn remove_buttons_patches_only_the_components() {
    let http = MockHttp::start().await;
    let target = MessageRef {
        conversation_id: CHANNEL_ID.to_string(),
        message_id: "100000000000000800".to_string(),
    };
    channel(&http).remove_buttons(&target).await.unwrap();
    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "PATCH");
    assert_eq!(
        requests[0].path,
        format!("/channels/{CHANNEL_ID}/messages/100000000000000800")
    );
    assert_eq!(requests[0].body, json!({ "components": [] }));
}
