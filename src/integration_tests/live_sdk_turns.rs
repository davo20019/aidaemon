// Live checks of whole agent turns on the SDK kernel, against the real
// ChatGPT backend with a cheap model. Ignored by default:
//
//   AIDAEMON_LIVE_TEST=1 RUST_MIN_STACK=8388608 cargo test --lib live_sdk_turn \
//       -- --ignored --nocapture --test-threads=1
//
// Override the model with AIDAEMON_LIVE_MODEL=<id>.

mod live_sdk_turns {
    use std::sync::Arc;

    use crate::config::ProviderKind;
    use crate::testing::{setup_live_agent, LiveHarness};
    
    use crate::types::{ChannelContext, UserRole};

    fn live_model() -> String {
        std::env::var("AIDAEMON_LIVE_MODEL").unwrap_or_else(|_| "gpt-5.6-terra".to_string())
    }

    async fn live_agent() -> Option<LiveHarness> {
        if std::env::var("AIDAEMON_LIVE_TEST").as_deref() != Ok("1") {
            eprintln!("SKIPPED: set AIDAEMON_LIVE_TEST=1 to run live tests");
            return None;
        }
        if !crate::providers::chatgpt_login::is_connected() {
            eprintln!("SKIPPED: no ChatGPT login found — run `aidaemon auth login openai`");
            return None;
        }
        let provider = crate::providers::factory::chatgpt_backend(None, None).unwrap();
        Some(
            setup_live_agent(Arc::new(provider), ProviderKind::OpenaiChatgpt, &live_model())
                .await
                .unwrap(),
        )
    }

    async fn ask(harness: &LiveHarness, session: &str, text: &str) -> String {
        let reply = tokio::time::timeout(
            std::time::Duration::from_secs(240),
            harness.agent.handle_message(
                session,
                text,
                None,
                UserRole::Owner,
                ChannelContext::private("telegram"),
                None,
            ),
        )
        .await
        .expect("turn timed out")
        .expect("turn failed");
        eprintln!("\n>>> {text}\n<<< {reply}\n");
        trace(harness, session).await;
        reply
    }

    /// Tool calls, results, decision points and the task end of the session,
    /// so a failed live check shows what the model was offered and did.
    async fn trace(harness: &LiveHarness, session: &str) {
        let since = chrono::Utc::now() - chrono::Duration::hours(1);
        for event in harness
            .event_store
            .query_events(session, since)
            .await
            .unwrap_or_default()
        {
            let kind = format!("{:?}", event.event_type);
            if !matches!(
                kind.as_str(),
                "ToolCall" | "ToolResult" | "DecisionPoint" | "TaskEnd" | "LlmCall"
            ) {
                continue;
            }
            let mut data = event.data.to_string();
            data.truncate(600);
            eprintln!("   [{kind}] {data}");
        }
    }

    #[tokio::test]
    #[ignore = "hits the real ChatGPT backend; run with --ignored"]
    async fn live_sdk_turn_plain_answer() {
        let Some(h) = live_agent().await else { return };
        let reply = ask(&h, "telegram:live-1", "What is 17 * 23? Reply with just the number.").await;
        assert!(reply.contains("391"), "{reply}");
    }

    #[tokio::test]
    #[ignore = "hits the real ChatGPT backend; run with --ignored"]
    async fn live_sdk_turn_remember_then_recall() {
        let Some(h) = live_agent().await else { return };
        let session = "telegram:live-2";
        ask(&h, session, "Remember that my favorite color is teal.").await;
        // Personal statements may land in the people graph rather than the
        // facts table, so the check is the recall, from a fresh session.
        let reply = ask(&h, "telegram:live-2b", "What's my favorite color?").await;
        assert!(reply.to_lowercase().contains("teal"), "{reply}");
    }

    #[tokio::test]
    #[ignore = "hits the real ChatGPT backend; run with --ignored"]
    async fn live_sdk_turn_write_then_read_file() {
        let Some(h) = live_agent().await else { return };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        let reply = ask(
            &h,
            "telegram:live-3",
            &format!(
                "Create the file {} containing exactly the text `alpha beta gamma`, then read it \
                 back and tell me what it contains.",
                path.display()
            ),
        )
        .await;
        let written = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(written.trim(), "alpha beta gamma");
        assert!(reply.contains("alpha beta gamma"), "{reply}");
    }

    #[tokio::test]
    #[ignore = "hits the real ChatGPT backend; run with --ignored"]
    async fn live_sdk_turn_terminal_count() {
        let Some(h) = live_agent().await else { return };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("three.txt");
        std::fs::write(&path, "one\ntwo\nthree\nfour\nfive\n").unwrap();
        let reply = ask(
            &h,
            "telegram:live-4",
            &format!(
                "Use a shell command to count the lines in {}. Tell me the number.",
                path.display()
            ),
        )
        .await;
        assert!(reply.contains('5'), "{reply}");
    }

    #[tokio::test]
    #[ignore = "hits the real ChatGPT backend; run with --ignored"]
    async fn live_sdk_turn_multi_step_listing() {
        let Some(h) = live_agent().await else { return };
        let dir = tempfile::tempdir().unwrap();
        for name in ["mia.txt", "zoe.txt", "acme.md"] {
            std::fs::write(dir.path().join(name), name).unwrap();
        }
        let reply = ask(
            &h,
            "telegram:live-5",
            &format!(
                "List the files in {} and tell me which of them is a markdown file.",
                dir.path().display()
            ),
        )
        .await;
        assert!(reply.contains("acme.md"), "{reply}");
    }

    #[tokio::test]
    #[ignore = "hits the real ChatGPT backend; run with --ignored"]
    async fn live_sdk_turn_follow_up_keeps_context() {
        let Some(h) = live_agent().await else { return };
        let session = "telegram:live-6";
        ask(&h, session, "My daughters are Mia and Zoe. Mia is 9.").await;
        let reply = ask(&h, session, "How old is the older one I just mentioned?").await;
        assert!(reply.contains('9'), "{reply}");
    }
}
