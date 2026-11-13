mod support;

use aidaemon_channel_slack::SlackChannel;
use aidaemon_channels::{
    ApprovalChoice, ApprovalPrompt, Channel, ChannelError, Conversation, ConversationKind,
    MessageRef, OutboundMessage,
};
use serde_json::json;
use support::*;

fn channel(api: &MockApi) -> SlackChannel {
    SlackChannel::new(BOT_TOKEN, APP_TOKEN)
        .with_api_base(&api.base)
        .with_download_client(reqwest::Client::builder().no_proxy())
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
    assert!(
        caps.typing_indicator,
        "the hourglass reaction stands in for typing"
    );
    assert_eq!(caps.max_message_len, 3000, "one Block Kit section");
    let short = SlackChannel::new(BOT_TOKEN, APP_TOKEN).with_max_message_len(500);
    assert_eq!(short.capabilities().max_message_len, 500);
    let clamped = SlackChannel::new(BOT_TOKEN, APP_TOKEN).with_max_message_len(99_999);
    assert_eq!(clamped.capabilities().max_message_len, 3000);
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

/// A prompt ending in a collapsed block shows that block's content as a
/// secondary context block between the summary and the buttons.
#[tokio::test]
async fn ask_puts_a_collapsed_block_under_the_summary() {
    let api = ok_api().await;
    let prompt = ApprovalPrompt {
        id: "prompt-43".into(),
        conversation: thread(),
        text: "**Your assistant wants to run a command on your computer.**\n\n\
               ```details\nrm -rf build/ && echo <done>\n```"
            .into(),
        choices: vec![ApprovalChoice {
            id: "once".into(),
            label: "Allow".into(),
        }],
    };
    channel(&api).ask(&prompt).await.unwrap();
    let body = api.calls("chat.postMessage")[0].json();
    let blocks = body["blocks"].as_array().unwrap();
    assert_eq!(blocks.len(), 3);
    assert_eq!(
        blocks[0]["text"]["text"],
        "*Your assistant wants to run a command on your computer.*"
    );
    assert_eq!(blocks[1]["type"], "context");
    assert_eq!(
        blocks[1]["elements"][0]["text"],
        "```\nrm -rf build/ &amp;&amp; echo &lt;done&gt;\n```"
    );
    assert_eq!(blocks[2]["type"], "actions");
}

#[tokio::test]
async fn delete_calls_chat_delete() {
    let api = ok_api().await;
    channel(&api)
        .delete(&MessageRef {
            conversation_id: "C000SYNTH:1700000000.000100".into(),
            message_id: "1700000001.000005".into(),
        })
        .await
        .unwrap();
    let calls = api.calls("chat.delete");
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].json(),
        json!({ "channel": "C000SYNTH", "ts": "1700000001.000005" })
    );
}

#[tokio::test]
async fn send_file_uses_the_external_upload_flow() {
    let api = MockApi::start(|req, _| match req.method() {
        "files.getUploadURLExternal" => {
            let host = req.header("host").unwrap_or_default().to_string();
            Response::json(json!({
                "ok": true,
                "upload_url": format!("http://{host}/upload/F000SYNTH"),
                "file_id": "F000SYNTH",
            }))
        }
        _ => Response::json(json!({ "ok": true })),
    })
    .await;
    let sent = channel(&api)
        .send_file(
            &thread(),
            &aidaemon_channels::OutboundFile {
                name: "report.txt".into(),
                bytes: b"synthetic report for Acme Corp".to_vec(),
                mime_type: Some("text/plain".into()),
                caption: Some("The **weekly** report".into()),
            },
        )
        .await
        .unwrap();
    assert_eq!(sent.message_id, "F000SYNTH");

    let ticket = api.calls("files.getUploadURLExternal");
    assert_eq!(ticket.len(), 1);
    assert!(
        ticket[0].body.contains("filename=report.txt"),
        "{}",
        ticket[0].body
    );
    assert!(ticket[0].body.contains("length=30"), "{}", ticket[0].body);
    let upload = api.calls("F000SYNTH");
    assert_eq!(upload.len(), 1);
    assert_eq!(upload[0].body, "synthetic report for Acme Corp");
    let complete = api.calls("files.completeUploadExternal");
    assert_eq!(complete.len(), 1);
    let body = complete[0].json();
    assert_eq!(body["files"][0]["id"], "F000SYNTH");
    assert_eq!(body["channel_id"], "C000SYNTH");
    assert_eq!(body["thread_ts"], "1700000000.000100");
    assert_eq!(body["initial_comment"], "The *weekly* report");
}

#[tokio::test]
async fn send_file_refuses_an_upload_url_off_slack() {
    let api = MockApi::start(|req, _| match req.method() {
        "files.getUploadURLExternal" => Response::json(json!({
            "ok": true,
            "upload_url": "https://uploads.example.test/F000SYNTH",
            "file_id": "F000SYNTH",
        })),
        _ => Response::json(json!({ "ok": true })),
    })
    .await;
    let err = channel(&api)
        .send_file(
            &group(),
            &aidaemon_channels::OutboundFile {
                name: "a.txt".into(),
                bytes: b"x".to_vec(),
                mime_type: None,
                caption: None,
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, ChannelError::Other(_)));
    assert!(api.calls("files.completeUploadExternal").is_empty());
}

#[tokio::test]
async fn download_sends_the_bot_token_only_to_slack_hosts() {
    let api = MockApi::start(|_, _| Response::json(json!("synthetic bytes"))).await;
    let slack = channel(&api);
    let host = api.base.trim_end_matches("/api").to_string();
    let bytes = slack
        .download(&aidaemon_channels::Attachment::new(
            "notes.txt",
            format!("{host}/files-pri/T000SYNTH-F000SYNTH/notes.txt"),
        ))
        .await
        .unwrap();
    assert_eq!(bytes, b"\"synthetic bytes\"");
    assert_eq!(
        api.requests()[0].header("authorization"),
        Some("Bearer xoxb-synthetic")
    );

    let err = slack
        .download(&aidaemon_channels::Attachment::new(
            "notes.txt",
            "https://files.example.test/notes.txt",
        ))
        .await
        .unwrap_err();
    assert!(matches!(err, ChannelError::Other(_)));
    assert_eq!(api.requests().len(), 1, "no request to another host");
}

#[tokio::test]
async fn answer_command_posts_an_ephemeral_reply_to_the_response_url() {
    let api = ok_api().await;
    let slack = channel(&api);
    let host = api.base.trim_end_matches("/api").to_string();
    let command = aidaemon_channels::CommandInvocation {
        conversation: group(),
        sender: aidaemon_channels::Sender {
            id: "U000ALICE".into(),
            display_name: None,
            is_bot: false,
        },
        name: "status".into(),
        args: String::new(),
        handle: format!("{host}/commands/T000SYNTH/1/synthetic"),
    };
    slack
        .answer_command(&command, "All **quiet**")
        .await
        .unwrap();
    let reqs = api.requests();
    assert_eq!(reqs[0].path, "/commands/T000SYNTH/1/synthetic");
    assert_eq!(
        reqs[0].json(),
        json!({ "response_type": "ephemeral", "text": "All *quiet*" })
    );

    let elsewhere = aidaemon_channels::CommandInvocation {
        handle: "https://hooks.example.test/x".into(),
        ..command
    };
    assert!(slack.answer_command(&elsewhere, "hi").await.is_err());
    assert_eq!(api.requests().len(), 1);
}

#[tokio::test]
async fn display_name_reads_the_profile() {
    let api = MockApi::start(|_, _| {
        Response::json(json!({
            "ok": true,
            "user": { "id": "U000ALICE", "profile": { "display_name": "", "real_name": "Alice Rivera" } }
        }))
    })
    .await;
    let name = channel(&api).display_name("U000ALICE").await.unwrap();
    assert_eq!(name.as_deref(), Some("Alice Rivera"));
    assert!(api.calls("users.info")[0].body.contains("user=U000ALICE"));
}

#[tokio::test]
async fn a_shorter_message_length_splits_sooner() {
    let api = ok_api().await;
    let slack = channel(&api).with_max_message_len(100);
    let text = "word ".repeat(60);
    slack
        .send(&group(), &OutboundMessage::text(text))
        .await
        .unwrap();
    let posts = api.calls("chat.postMessage");
    assert!(posts.len() >= 3, "{} posts", posts.len());
    for post in posts {
        assert!(post.json()["text"].as_str().unwrap().chars().count() <= 100);
    }
}

fn file_at(api: &MockApi) -> aidaemon_channels::Attachment {
    let host = api.base.trim_end_matches("/api");
    aidaemon_channels::Attachment::new("notes.txt", format!("{host}/files-pri/F000SYNTH/notes.txt"))
}

#[tokio::test]
async fn a_redirect_off_slack_is_refused_and_the_token_is_not_followed() {
    let api = MockApi::start(|_, _| {
        Response::json(json!({}))
            .status(302)
            .header("Location", "https://files.example.test/stolen")
    })
    .await;
    let err = channel(&api).download(&file_at(&api)).await.unwrap_err();
    assert_eq!(api.requests().len(), 1);
    let shown = format!("{err} {err:?}");
    assert!(shown.contains("refused"), "{shown}");
    assert!(!shown.contains(BOT_TOKEN), "{shown}");
    assert!(!shown.contains("example.test"), "{shown}");
}

#[tokio::test]
async fn a_download_stops_at_the_limit() {
    let api = MockApi::start(|_, _| Response::json(json!("0123456789"))).await;
    let err = channel(&api)
        .download_limited(&file_at(&api), 4)
        .await
        .unwrap_err();
    assert!(
        matches!(err, ChannelError::TooLarge { limit_bytes: 4 }),
        "{err}"
    );
}

#[tokio::test]
async fn slacks_sign_in_page_does_not_pass_for_the_file() {
    let api = MockApi::start(|_, _| {
        Response::json(json!("<html>sign in</html>"))
            .header("Content-Type", "text/html; charset=utf-8")
    })
    .await;
    let err = channel(&api).download(&file_at(&api)).await.unwrap_err();
    assert!(matches!(err, ChannelError::Auth(_)), "{err}");
    assert!(err.to_string().contains("files:read"), "{err}");

    // A file that really is HTML is fine.
    let bytes = channel(&api)
        .download(&file_at(&api).with_mime_type(Some("text/html".to_string())))
        .await
        .unwrap();
    assert!(!bytes.is_empty());
}

#[tokio::test]
async fn a_long_edit_fits_the_configured_length_and_is_marked() {
    let api = ok_api().await;
    let slack = channel(&api).with_max_message_len(100);
    let target = MessageRef {
        conversation_id: "C000SYNTH".into(),
        message_id: "1700000000.000200".into(),
    };
    slack.edit(&target, &"word ".repeat(100)).await.unwrap();
    let calls = api.calls("chat.update");
    assert_eq!(calls.len(), 1);
    let text = calls[0].json()["text"].as_str().unwrap().to_string();
    assert_eq!(text.chars().count(), 100);
    assert!(text.ends_with('\u{2026}'));
}

#[tokio::test]
async fn remove_buttons_needs_a_prompt_it_saw_pressed() {
    let api = ok_api().await;
    let target = MessageRef {
        conversation_id: "C000SYNTH".into(),
        message_id: "1700000000.000200".into(),
    };
    let err = channel(&api).remove_buttons(&target).await.unwrap_err();
    assert!(matches!(err, ChannelError::Other(_)), "{err}");
    assert!(api.calls("chat.update").is_empty());
}
