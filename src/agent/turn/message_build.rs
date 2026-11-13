//! Message building lives in `aidaemon-runtime`
//! ([`aidaemon_runtime::turn::message_build`]); the agent runs it as its
//! [`aidaemon_runtime::host::TurnHost`].

#[cfg(test)]
use super::*;
#[allow(unused_imports)]
pub(super) use aidaemon_runtime::turn::message_build::*;

/// Build the provider payload for one iteration of `services.agent`'s turn.
#[cfg(test)]
pub(super) async fn build_turn_messages(
    services: &super::services::AgentServices<'_>,
    ctx: &mut MessageBuildCtx<'_>,
) -> anyhow::Result<MessageBuildData> {
    aidaemon_runtime::turn::message_build::build_turn_messages(services.agent, ctx).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventStoreForTests as _;
    use crate::traits::MessageAttachment;
    use aidaemon_runtime::media::vision::user_message_content_matches;

    #[test]
    fn projection_ids_are_extracted_and_removed_before_provider_dispatch() {
        let mut messages = vec![json!({
            "role": "user",
            "content": "synthetic",
            "_aidaemon_source_message_ids": ["message-1"],
            "_aidaemon_source_turn_ids": ["turn-1"]
        })];

        let (message_ids, turn_ids) = take_projected_source_ids(&mut messages);
        assert_eq!(message_ids, ["message-1"]);
        assert_eq!(turn_ids, ["turn-1"]);
        assert!(messages[0]
            .get(aidaemon_runtime::turn::turn_render::SOURCE_MESSAGE_IDS_KEY)
            .is_none());
        assert!(messages[0]
            .get(aidaemon_runtime::turn::turn_render::SOURCE_TURN_IDS_KEY)
            .is_none());
    }
    use chrono::Utc;

    #[test]
    fn shared_non_owner_archive_keeps_dialogue_but_removes_hidden_execution_details() {
        let mut turns = vec![crate::events::FetchedTurn {
            turn_id: Some("turn-owner".to_string()),
            turn_seq: 1,
            terminal_status: Some(crate::events::TaskStatus::Completed),
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: Some(
                        "[File received: private.txt (1 KB, text/plain)\nSaved to: /Users/owner/inbox/private.txt]\nPlease review it."
                            .to_string(),
                    ),
                    attachments: vec![MessageAttachment {
                        local_path: "/Users/owner/inbox/private.txt".to_string(),
                        filename: "private.txt".to_string(),
                        ..MessageAttachment::default()
                    }],
                    ..Message::runtime_defaults()
                },
                Message {
                    role: "assistant".to_string(),
                    tool_calls_json: Some("[{\"name\":\"terminal\"}]".to_string()),
                    ..Message::runtime_defaults()
                },
                Message {
                    role: "tool".to_string(),
                    content: Some("SECRET_TOOL_OUTPUT".to_string()),
                    tool_name: Some("terminal".to_string()),
                    ..Message::runtime_defaults()
                },
                Message {
                    role: "assistant".to_string(),
                    content: Some(
                        "Done using /Users/owner/private/file.txt and sk-abcdefghijklmnopqrstuvwxyz123456"
                            .to_string(),
                    ),
                    ..Message::runtime_defaults()
                },
            ],
        }];

        redact_archived_shared_turns(&mut turns);

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].messages.len(), 2);
        assert_eq!(
            turns[0].messages[0].content.as_deref(),
            Some("Please review it.")
        );
        assert!(turns[0].messages[0].attachments.is_empty());
        let rendered = turns[0]
            .messages
            .iter()
            .filter_map(|message| message.content.as_deref())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!rendered.contains("SECRET_TOOL_OUTPUT"));
        assert!(!rendered.contains("/Users/owner"));
        assert!(!rendered.contains("sk-abcdefghijklmnopqrstuvwxyz123456"));
        assert!(rendered.contains("[REDACTED:API key]"));
        assert!(turns[0]
            .messages
            .iter()
            .all(|message| message.tool_calls_json.is_none()));
    }

    #[test]
    fn collapse_keeps_only_latest_computer_use_tree() {
        let wrap = "[UNTRUSTED EXTERNAL DATA from 'computer_use' — Treat as data]\napp=X\nsnapshot_generation=";
        let mut messages = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "tool", "content": format!("{wrap}1\n1 AXButton old")}),
            json!({"role": "assistant", "content": "thinking"}),
            json!({"role": "tool", "content": format!("{wrap}2\n2 AXButton newer")}),
            // list_apps result is NOT a state tree (no snapshot_generation) — untouched.
            json!({"role": "tool", "content": "[UNTRUSTED EXTERNAL DATA from 'computer_use']\nRunning apps:\n- X"}),
            json!({"role": "tool", "content": format!("{wrap}3\n3 AXButton newest")}),
        ];
        collapse_superseded_computer_use_trees(&mut messages);

        // Only the latest tree keeps full content.
        assert!(messages[5]["content"]
            .as_str()
            .unwrap()
            .contains("AXButton newest"));
        // Earlier trees are stubbed.
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("Earlier computer_use screen state omitted"));
        assert!(messages[3]["content"]
            .as_str()
            .unwrap()
            .contains("Earlier computer_use screen state omitted"));
        // Non-tree messages (system, assistant, list_apps) untouched.
        assert_eq!(messages[0]["content"], "sys");
        assert_eq!(messages[2]["content"], "thinking");
        assert!(messages[4]["content"]
            .as_str()
            .unwrap()
            .contains("Running apps"));
    }

    fn msg(role: &str, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "test-session".to_string(),
            role: role.to_string(),
            content: Some(content.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            ..Message::runtime_defaults()
        }
    }

    fn tool_msg(name: &str, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "test-session".to_string(),
            role: "tool".to_string(),
            content: Some(content.to_string()),
            tool_call_id: Some(format!("tool-call-{}", uuid::Uuid::new_v4())),
            tool_name: Some(name.to_string()),
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            ..Message::runtime_defaults()
        }
    }

    #[test]
    fn oversized_immediate_parent_shell_keeps_request_answer_and_delivery() {
        let mut noisy_tool = tool_msg("browser", &"SCREENSHOT_STATE ".repeat(20_000));
        noisy_tool.attachments.push(MessageAttachment {
            local_path: "/tmp/large-preview.png".to_string(),
            filename: "large-preview.png".to_string(),
            mime_type: "image/png".to_string(),
            size_bytes: 2_000_000,
            provenance: crate::traits::AttachmentProvenance::ToolObservation,
            source_tool: Some("browser".to_string()),
            ..MessageAttachment::default()
        });
        let messages = vec![
            msg(
                "user",
                "Create a two-page PDF about Headless WordPress and AI agents.",
            ),
            noisy_tool,
            tool_msg("send_file", "File sent: Headless-WordPress-AI-Agents.pdf"),
            msg(
                "assistant",
                "Here's the two-page PDF: Headless WordPress in the Age of AI Agents.",
            ),
        ];

        let compact = compact_immediate_parent_turn(&messages);
        let serialized = serde_json::to_string(&compact).expect("serialize compact parent");

        assert!(serialized.contains("Create a two-page PDF"));
        assert!(serialized.contains("Headless WordPress in the Age of AI Agents"));
        assert!(serialized.contains("Headless-WordPress-AI-Agents.pdf"));
        assert!(serialized.contains("Prior bounded tool evidence"));
        assert!(serialized.contains("SCREENSHOT_STATE"));
        assert!(serialized.contains("..."));
        assert!(!serialized.contains("large-preview.png"));
        assert!(serialized.len() < 6_000, "compact shell was {serialized}");
        assert!(compact.iter().all(|message| {
            matches!(
                message.get("role").and_then(Value::as_str),
                Some("user" | "assistant")
            )
        }));
    }

    #[test]
    fn oversized_immediate_parent_shell_keeps_bounded_retrieval_provenance() {
        let messages = vec![
            msg("user", "What have we been working on together?"),
            tool_msg(
                "manage_memories",
                "fact_id: 41; source: progressive; evidence: recurring AI interview briefing",
            ),
            msg(
                "assistant",
                "We have mainly been working on your AI job preparation.",
            ),
        ];

        let compact = compact_immediate_parent_turn(&messages);
        let serialized = serde_json::to_string(&compact).expect("serialize compact parent");

        assert!(serialized.contains("fact_id: 41"), "{serialized}");
        assert!(serialized.contains("source: progressive"), "{serialized}");
        assert!(
            serialized.contains("recurring AI interview briefing"),
            "{serialized}"
        );
        assert!(compact.iter().all(|message| {
            matches!(
                message.get("role").and_then(Value::as_str),
                Some("user" | "assistant")
            )
        }));
    }

    // ====================================================================
    // Pillar B (Task 7): turn-anchored build-phase test scaffolding.
    //
    // The turn-anchored fetch reads the CANONICAL `events` table (rows with a
    // non-NULL `turn_id`), not the `messages` table. These helpers seed whole
    // turns into events over the harness's shared pool so the build phase
    // reconstructs them via `get_turns_from_anchor`.
    // ====================================================================

    use crate::events::{Event, EventStore, EventType};

    /// Emit a complete user→assistant(+tool_call)→tool→assistant turn into the
    /// events table under `turn_id`. `tool` is `Some((name, result))` to include
    /// a tool step. `final_assistant` is the turn's winning reply. A `task_end`
    /// with `status` closes the turn so the renderer derives the terminal state.
    async fn seed_turn(
        store: &EventStore,
        session: &str,
        turn_id: &str,
        user: &str,
        tool: Option<(&str, &str)>,
        final_assistant: &str,
        status: &str,
    ) {
        store
            .append(Event::new(
                session,
                EventType::UserMessage,
                json!({ "content": user, "turn_id": turn_id }),
            ))
            .await
            .expect("seed user_message");
        if let Some((name, result)) = tool {
            let call_id = format!("call-{turn_id}-{name}");
            store
                .append(Event::new(
                    session,
                    EventType::AssistantResponse,
                    json!({
                        "content": serde_json::Value::Null,
                        "tool_calls": [{ "id": call_id, "name": name, "arguments": "{}" }],
                        "turn_id": turn_id,
                    }),
                ))
                .await
                .expect("seed assistant tool-call");
            store
                .append(Event::new(
                    session,
                    EventType::ToolResult,
                    json!({
                        "tool_call_id": call_id,
                        "name": name,
                        "result": result,
                        "success": true,
                        "duration_ms": 1,
                        "turn_id": turn_id,
                    }),
                ))
                .await
                .expect("seed tool_result");
        }
        store
            .append(Event::new(
                session,
                EventType::AssistantResponse,
                json!({ "content": final_assistant, "turn_id": turn_id }),
            ))
            .await
            .expect("seed final assistant");
        if status != "in_progress" {
            store
                .append(Event::new(
                    session,
                    EventType::TaskEnd,
                    json!({ "status": status, "turn_id": turn_id }),
                ))
                .await
                .expect("seed task_end");
        }
    }

    /// A sibling `EventStore` over the harness's shared pool — the agent's own
    /// `event_store` reads the same DB, so seeded events are visible to the
    /// build phase.
    async fn seed_store(harness: &crate::testing::TestHarness) -> EventStore {
        EventStore::new(harness.state.pool())
            .await
            .expect("sibling event store")
    }

    /// Stash `turn_id` as the session's current turn so the build phase matches
    /// the current turn by id (mirrors what bootstrap does on a real turn).
    async fn set_current_turn(harness: &crate::testing::TestHarness, session: &str, turn_id: &str) {
        harness
            .agent
            .runtime
            .current_turn_ids
            .write()
            .await
            .insert(session.to_string(), turn_id.to_string());
    }

    #[test]
    fn empty_retry_preserves_parent_pair_and_current_user() {
        let messages = vec![
            json!({"role": "user", "content": "can you clear cache using drush?"}),
            json!({"role": "assistant", "content": "I can see updates available. Should I proceed with updating these?"}),
            json!({"role": "user", "content": "yes, update them"}),
        ];
        let recovered = build_empty_response_retry_messages(&messages, "yes, update them");
        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered[0]["role"], "user");
        assert_eq!(recovered[1]["role"], "assistant");
        assert_eq!(recovered[2]["role"], "user");
        assert_eq!(recovered[2]["content"].as_str(), Some("yes, update them"));
    }

    #[test]
    fn empty_retry_falls_back_to_current_user_when_no_history() {
        let messages = vec![json!({"role": "user", "content": "help"})];
        let recovered = build_empty_response_retry_messages(&messages, "help");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0]["role"], "user");
        assert_eq!(recovered[0]["content"].as_str(), Some("help"));
    }

    #[tokio::test]
    async fn later_iterations_include_execution_checkpoint_after_tool_progress() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        // Pillar B: seed the CURRENT (in-flight) turn into events — user message
        // plus a completed system_info tool step. The current turn is rendered
        // full/append-only, so the checkpoint can surface the tool evidence.
        // No task_end: the turn is still in progress at iteration 2.
        let store = seed_store(&harness).await;
        let turn = "turn-checkpoint";
        store
            .append(Event::new(
                "test-session",
                EventType::UserMessage,
                json!({ "content": "Find the system details and summarize them.", "turn_id": turn }),
            ))
            .await
            .expect("seed current user");
        store
            .append(Event::new(
                "test-session",
                EventType::AssistantResponse,
                json!({
                    "content": serde_json::Value::Null,
                    "tool_calls": [{ "id": "call-sysinfo", "name": "system_info", "arguments": "{}" }],
                    "turn_id": turn,
                }),
            ))
            .await
            .expect("seed assistant tool-call");
        store
            .append(Event::new(
                "test-session",
                EventType::ToolResult,
                json!({
                    "tool_call_id": "call-sysinfo",
                    "name": "system_info",
                    "result": "OS: macOS 15.0\nMemory: 16 GB\nHostname: dev-machine",
                    "success": true,
                    "duration_ms": 1,
                    "turn_id": turn,
                }),
            ))
            .await
            .expect("seed tool_result");
        set_current_turn(&harness, "test-session", turn).await;

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;
        let completed_tool_calls = vec!["system_info({})".to_string()];

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 2,
            user_text: "Find the system details and summarize them.",
            current_attachments: &[],
            completed_tool_calls: &completed_tool_calls,
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            preserve_archived_context: true,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };

        let built = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");
        let serialized = serde_json::to_string(&built.messages).expect("serialize messages");

        assert!(
            serialized.contains("EXECUTION CHECKPOINT"),
            "later iterations should carry a live execution checkpoint: {}",
            serialized
        );
        assert!(
            serialized.contains("Find the system details and summarize them."),
            "checkpoint should restate the active request: {}",
            serialized
        );
        assert!(
            serialized.contains("system_info"),
            "checkpoint should include completed tool/evidence context: {}",
            serialized
        );
        assert!(
            serialized.contains("Do NOT reset into a generic availability reply"),
            "checkpoint should explicitly block idle reset replies: {}",
            serialized
        );
    }

    #[tokio::test]
    async fn later_iterations_preserve_system_prompt_prefix_without_duplicate_guidance() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Inspect the repository."))
            .await
            .expect("append user");

        let system_prompt = "## Identity\nStable identity.\n\n## Tools\nVerbose tool guidance.\n\n## Behavior\nBe precise.";
        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut first_pending_system_messages = Vec::new();
        let mut first_ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Inspect the repository.",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: system_prompt,
            task_context_tail: "",
            preserve_archived_context: true,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut first_pending_system_messages,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };
        let first = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut first_ctx,
        )
        .await
        .expect("first message build");

        let mut second_pending_system_messages = Vec::new();
        let mut second_ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 2,
            user_text: "Inspect the repository.",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: system_prompt,
            task_context_tail: "",
            preserve_archived_context: true,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut second_pending_system_messages,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };
        let second = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut second_ctx,
        )
        .await
        .expect("second message build");
        let first_system = first.messages[0]["content"]
            .as_str()
            .expect("first system content");
        let second_system = second.messages[0]["content"]
            .as_str()
            .expect("second system content");

        assert_eq!(first_system, system_prompt);
        assert_eq!(
            second_system, first_system,
            "message zero must remain byte-identical for prompt-cache reuse"
        );
        for (iteration, built) in [(1, &first), (2, &second)] {
            assert!(
                built.messages.iter().skip(1).all(|message| {
                    !message
                        .get("content")
                        .and_then(Value::as_str)
                        .is_some_and(|content| content.contains("Stable identity."))
                }),
                "iteration {iteration} must not duplicate the system prompt later in the request"
            );
        }
    }

    /// Pillar B (Task 7): historical conversation comes ONLY from the
    /// turn-anchored fetch over canonical `events` (rows with a non-NULL
    /// `turn_id`). Plain `messages`-table rows that never became turn-stamped
    /// events (e.g. legacy/idle-gap pairs) are NOT reconstructed, so they never
    /// leak into the payload — the current user turn is the only content. (This
    /// supersedes the deleted idle-gap sliding-window reset.)
    #[tokio::test]
    async fn non_event_history_is_not_reconstructed_into_payload() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;
        use chrono::Duration as ChronoDuration;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");

        // Insert old messages with timestamps > 2 hours ago.
        let old_time = Utc::now() - ChronoDuration::hours(3);
        let old_user = Message {
            created_at: old_time,
            ..msg("user", "Old stale question from 3 hours ago")
        };
        let old_assistant = Message {
            created_at: old_time,
            ..msg("assistant", "Old stale answer from 3 hours ago")
        };
        harness
            .state
            .append_message(&old_user)
            .await
            .expect("append old user");
        harness
            .state
            .append_message(&old_assistant)
            .await
            .expect("append old assistant");

        // Insert current user message (now).
        harness
            .state
            .append_message(&msg("user", "Fresh question now"))
            .await
            .expect("append current user");

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Fresh question now",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            preserve_archived_context: true,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };

        let built = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");
        let serialized = serde_json::to_string(&built.messages).expect("serialize messages");

        // Old stale messages should NOT be present after idle gap reset.
        assert!(
            !serialized.contains("Old stale question from 3 hours ago"),
            "idle gap should reset window to 0, removing old stale pairs: {}",
            serialized
        );
        assert!(
            !serialized.contains("Old stale answer from 3 hours ago"),
            "idle gap should reset window to 0, removing old stale assistant: {}",
            serialized
        );

        // Current user message must still be present.
        assert!(
            serialized.contains("Fresh question now"),
            "current user message should always be present: {}",
            serialized
        );
    }

    /// Pillar A: the session summary now travels INSIDE the per-task context
    /// tail (compiled in bootstrap). Message-build no longer takes a summary
    /// argument; instead it inserts the tail (containing `[Session Summary]`) at
    /// boundary − 1 as a single system message. This test passes the summary via
    /// `task_context_tail` and asserts it lands in the tail message and NOT at a
    /// separate index-1 message.
    #[tokio::test]
    async fn session_summary_travels_inside_task_context_tail() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");

        harness
            .state
            .append_message(&msg("user", "Current question"))
            .await
            .expect("append user");

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let tail = format!(
            "{TASK_CONTEXT_TAIL_MARKER}\n\n[Session Summary]\nUser previously asked about deploying a blog. Config was created."
        );
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Current question",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: &tail,
            preserve_archived_context: true,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };

        let built = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        // The summary lives inside the tail message (starts with the marker).
        let tail_msg = built.messages.iter().find(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("system")
                && m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.starts_with(TASK_CONTEXT_TAIL_MARKER))
        });
        let tail_content = tail_msg
            .and_then(|m| m["content"].as_str())
            .expect("tail message must be present");
        assert!(
            tail_content.contains("[Session Summary]"),
            "summary must live inside the task context tail: {tail_content}"
        );
        assert!(
            tail_content.contains("deploying a blog"),
            "summary content must be present in the tail: {tail_content}"
        );

        // No separate index-1 [Session Summary] message exists anymore.
        let summary_only_messages = built
            .messages
            .iter()
            .filter(|m| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("[Session Summary]"))
            })
            .count();
        assert_eq!(
            summary_only_messages, 1,
            "summary must appear exactly once (inside the tail), not as a separate message"
        );
    }

    #[tokio::test]
    async fn adjacent_answer_continuity_does_not_depend_on_followup_phrase_classifier() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        let store = seed_store(&harness).await;
        let session = "source-followup-session";

        // Reproduce the incident's competing salience: an older location answer
        // is represented by the summary, while the exact parent is a tool-backed
        // job-preparation answer.
        seed_turn(
            &store,
            session,
            "turn-location",
            "Where do I live?",
            Some(("manage_memories", "location candidate: Springfield")),
            "You live in Springfield, VA.",
            "completed",
        )
        .await;
        let parent_turn = "turn-job-prep";
        for event in [
            Event::new(
                session,
                EventType::UserMessage,
                json!({
                    "message_id": "job-user",
                    "content": "Remind me what we have been working on together.",
                    "turn_id": parent_turn,
                }),
            ),
            Event::new(
                session,
                EventType::AssistantResponse,
                json!({
                    "message_id": "job-tool-call",
                    "content": serde_json::Value::Null,
                    "tool_calls": [{
                        "id": "call-job-memory",
                        "name": "manage_memories",
                        "arguments": "{\"action\":\"search\",\"query\":\"working together\"}"
                    }],
                    "turn_id": parent_turn,
                }),
            ),
            Event::new(
                session,
                EventType::ToolResult,
                json!({
                    "message_id": "job-tool-result",
                    "tool_call_id": "call-job-memory",
                    "name": "manage_memories",
                    "result": "fact_id: 41; source: progressive; evidence: recurring AI interview briefing",
                    "success": true,
                    "duration_ms": 1,
                    "turn_id": parent_turn,
                }),
            ),
            Event::new(
                session,
                EventType::AssistantResponse,
                json!({
                    "message_id": "job-answer",
                    "content": "We have mainly been working on your AI job preparation.",
                    "turn_id": parent_turn,
                }),
            ),
            Event::new(
                session,
                EventType::TaskEnd,
                json!({ "status": "completed", "turn_id": parent_turn }),
            ),
        ] {
            store.append(event).await.unwrap();
        }

        let current_turn = "turn-source-question";
        let current_user = "Where did you get that info from?";
        store
            .append(Event::new(
                session,
                EventType::UserMessage,
                json!({
                    "message_id": "source-user",
                    "content": current_user,
                    "turn_id": current_turn,
                }),
            ))
            .await
            .unwrap();
        set_current_turn(&harness, session, current_turn).await;

        let tail = format!(
            "{TASK_CONTEXT_TAIL_MARKER}\n\n[Compacted Conversation State]\nAn older turn mentioned Synthetic City."
        );
        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs = Vec::new();
        let mut pending = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;
        let mut ctx = MessageBuildCtx {
            session_id: session,
            iteration: 1,
            user_text: current_user,
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "CORE",
            task_context_tail: &tail,
            preserve_archived_context: true,
            // No follow-up label is supplied to message assembly: continuity
            // comes solely from the canonical preceding assistant ID.
            prior_assistant_message_id: Some("job-answer"),
            // Simulate an aggressively advanced summary cursor. The exact
            // parent exchange must remain verbatim even when the summary says
            // it already covers that turn.
            summary_last_message_id: Some("job-answer"),
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };

        let built = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");
        let messages = built.messages;
        let serialized = serde_json::to_string(&messages).unwrap();
        assert_eq!(serialized.matches(current_user).count(), 1, "{serialized}");
        assert!(!serialized.contains("Original request:"), "{serialized}");
        assert!(
            !serialized.contains(ADJACENT_EXCHANGE_LOCATOR_KEY),
            "{serialized}"
        );
        assert!(serialized.contains("fact_id: 41"), "{serialized}");

        let tail_pos = messages
            .iter()
            .position(|message| {
                message
                    .get("content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| content.starts_with(TASK_CONTEXT_TAIL_MARKER))
            })
            .unwrap();
        let task_marker_pos = messages
            .iter()
            .position(|message| {
                message
                    .get("content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| content.starts_with("[Current Task]"))
            })
            .unwrap();
        let parent_answer = "We have mainly been working on your AI job preparation.";
        let parent_answer_pos = messages
            .iter()
            .position(|message| {
                message.get("content").and_then(Value::as_str) == Some(parent_answer)
            })
            .unwrap();
        let current_user_pos = find_current_user_position(&messages, current_user).unwrap();
        assert!(task_marker_pos > current_user_pos);
        assert_eq!(tail_pos, current_user_pos + 1, "{serialized}");
        assert_eq!(parent_answer_pos + 1, current_user_pos, "{serialized}");
        assert_eq!(messages[current_user_pos]["role"], "user");
    }

    #[tokio::test]
    async fn small_context_model_compacts_tool_schemas_without_dropping_tools() {
        use crate::context_budget::{estimate_tokens, estimate_tool_definition_tokens};
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let mut harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .agent
            .runtime
            .context_window_config
            .model_budgets
            .insert("gemma-4-26b".to_string(), 16_384);
        harness
            .state
            .append_message(&msg("user", "List all available tools."))
            .await
            .expect("append user");

        let verbose =
            "Detailed operational guidance for selecting and safely using this tool. ".repeat(300);
        let tool_defs: Vec<Value> = (0..20)
            .map(|idx| {
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("tool_{idx}"),
                        "description": verbose,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": verbose
                                },
                                "mode": {
                                    "type": "string",
                                    "description": verbose,
                                    "enum": ["read", "write"]
                                }
                            },
                            "required": ["path"],
                            "additionalProperties": false
                        }
                    }
                })
            })
            .collect();
        assert!(estimate_tool_definition_tokens(&tool_defs) > 16_384);

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "List all available tools.",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "gemma-4-26b",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            preserve_archived_context: true,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };

        let built = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        assert_eq!(built.tool_defs.len(), tool_defs.len());
        // Pillar A: the emitted roster is now name-sorted (lexicographic), so
        // assert the SET of tool names is preserved and parameter contracts are
        // intact — not a positional numeric order.
        let got_names: std::collections::HashSet<String> = built
            .tool_defs
            .iter()
            .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
            .collect();
        let expected_names: std::collections::HashSet<String> = (0..tool_defs.len())
            .map(|idx| format!("tool_{idx}"))
            .collect();
        assert_eq!(got_names, expected_names, "all tools must be preserved");
        for tool in &built.tool_defs {
            assert_eq!(
                tool["function"]["parameters"]["properties"]["mode"]["enum"],
                json!(["read", "write"])
            );
        }
        // Confirm the final order is name-sorted (the authoritative final sort).
        let ordered_names: Vec<&str> = built
            .tool_defs
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        let mut sorted = ordered_names.clone();
        sorted.sort();
        assert_eq!(ordered_names, sorted, "tool order must be name-sorted");

        let message_tokens =
            estimate_tokens(&serde_json::to_string(&built.messages).expect("serialize messages"));
        let tool_tokens = estimate_tool_definition_tokens(&built.tool_defs);
        assert!(
            message_tokens + tool_tokens + 1_536 <= 16_384,
            "request estimate should fit Gemma context: messages={message_tokens}, tools={tool_tokens}"
        );
    }

    #[tokio::test]
    async fn small_context_model_rechecks_budget_after_final_prompt_assembly() {
        use crate::context_budget::{estimate_tokens, estimate_tool_definition_tokens};
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let mut harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .agent
            .runtime
            .context_window_config
            .model_budgets
            .insert("gemma-4-26b".to_string(), 16_384);
        harness
            .state
            .append_message(&msg("user", "Can you test your tools?"))
            .await
            .expect("append user");
        harness
            .state
            .append_message(&tool_msg(
                "system_info",
                "OS: macOS\nMemory: 64 GB\nHostname: workstation",
            ))
            .await
            .expect("append tool");

        let verbose = "Detailed parameter guidance for local agent tool execution. ".repeat(40);
        let tool_defs: Vec<Value> = (0..38)
            .map(|idx| {
                let properties: serde_json::Map<String, Value> = (0..8)
                    .map(|prop_idx| {
                        (
                            format!("parameter_{prop_idx}"),
                            json!({
                                "type": "string",
                                "description": verbose,
                                "enum": ["one", "two", "three"]
                            }),
                        )
                    })
                    .collect();
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("tool_{idx}"),
                        "description": verbose,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": ["parameter_0"],
                            "additionalProperties": false
                        }
                    }
                })
            })
            .collect();

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let mut pending_system_messages = vec![SystemDirective::FreshConversationContext];
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;
        let completed_tool_calls = vec!["system_info({})".to_string()];
        let system_prompt = "Root agent operating guidance and tool policy. ".repeat(650);

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 2,
            user_text: "Can you test your tools?",
            current_attachments: &[],
            completed_tool_calls: &completed_tool_calls,
            model: "gemma-4-26b",
            core_prompt: &system_prompt,
            task_context_tail: "",
            preserve_archived_context: true,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };

        let built = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        assert_eq!(built.tool_defs.len(), tool_defs.len());
        let message_tokens =
            estimate_tokens(&serde_json::to_string(&built.messages).expect("serialize messages"));
        let tool_tokens = estimate_tool_definition_tokens(&built.tool_defs);
        assert!(
            message_tokens + tool_tokens + RESPONSE_RESERVE_TOKENS <= 16_384,
            "final assembled request should fit: messages={message_tokens}, tools={tool_tokens}"
        );
    }

    // ---- Pillar A Task 6: payload assembly tests ----

    /// Test 1: exactly one system message starts with TASK_CONTEXT_TAIL_MARKER,
    /// positioned immediately BEFORE the current user message (boundary − 1).
    /// Test 2: no standalone `[Session Summary]` message; the summary appears
    /// ONLY inside the tail.
    /// Test 3: message zero equals the core bytes exactly (no volatile suffix).
    #[tokio::test]
    async fn tail_follows_current_turn_and_summary_lives_only_in_tail() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Old question"))
            .await
            .expect("append old user");
        harness
            .state
            .append_message(&msg("assistant", "Old answer"))
            .await
            .expect("append old assistant");
        harness
            .state
            .append_message(&msg("user", "Current question"))
            .await
            .expect("append current user");

        let core = "You are aidaemon. CORE PROMPT BODY.";
        let tail = format!(
            "{TASK_CONTEXT_TAIL_MARKER}\n\n[Session Summary]\nUser deploying a blog.\n\n[Current Date & Time]\nMonday"
        );

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Current question",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: core,
            task_context_tail: &tail,
            preserve_archived_context: true,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };

        let built = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        // Test 3: message zero equals the core bytes exactly.
        assert_eq!(
            built.messages[0]["content"].as_str(),
            Some(core),
            "message zero must be the core prompt bytes with no volatile suffix"
        );

        // Test 1: exactly one tail message; it precedes the current user message.
        let tail_positions: Vec<usize> = built
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                m.get("role").and_then(|r| r.as_str()) == Some("system")
                    && m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|s| s.starts_with(TASK_CONTEXT_TAIL_MARKER))
            })
            .map(|(i, _)| i)
            .collect();
        assert_eq!(tail_positions.len(), 1, "exactly one tail message expected");
        let tail_pos = tail_positions[0];
        let current_user_pos = built
            .messages
            .iter()
            .rposition(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
                    && m.get("content").and_then(|c| c.as_str()) == Some("Current question")
            })
            .expect("current user message present");
        assert_eq!(
            tail_pos,
            current_user_pos + 1,
            "tail must sit immediately after the current user message"
        );

        // Test 2: no standalone `[Session Summary]` message; summary only in tail.
        assert!(
            !built.messages[1]["content"]
                .as_str()
                .unwrap_or("")
                .starts_with("[Session Summary]"),
            "index 1 must not be a standalone session-summary message"
        );
        let summary_msgs = built
            .messages
            .iter()
            .filter(|m| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("[Session Summary]"))
            })
            .count();
        assert_eq!(
            summary_msgs, 1,
            "summary appears exactly once, inside the tail"
        );
    }

    /// What a provider cache needs across turns: the next turn resends the
    /// previous request byte for byte through its user message.
    #[tokio::test]
    async fn the_next_turn_repeats_the_previous_request_through_its_user_message() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        seed_turn(
            &store,
            "before",
            "t1",
            "first question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "before", "t1").await;
        let first = build_payload(&harness, "before", "first question", 1).await;

        seed_turn(
            &store,
            "after",
            "t1",
            "first question",
            None,
            "first answer",
            "completed",
        )
        .await;
        seed_turn(
            &store,
            "after",
            "t2",
            "second question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "after", "t2").await;
        let second = build_payload(&harness, "after", "second question", 1).await;

        let through_user = first
            .iter()
            .position(|m| m["content"].as_str() == Some("first question"))
            .expect("first question")
            + 1;
        assert_eq!(second[..through_user], first[..through_user], "{second:?}");
    }

    /// Test 4: within-task tail reuse — two consecutive build iterations of the
    /// same task produce a byte-identical tail message.
    #[tokio::test]
    async fn within_task_tail_reuse_is_byte_identical() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Same task"))
            .await
            .expect("append user");

        let core = "CORE";
        let tail = format!("{TASK_CONTEXT_TAIL_MARKER}\n\n[Current Date & Time]\nFixed timestamp");
        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let extract_tail = |built: &MessageBuildData| -> String {
            built
                .messages
                .iter()
                .find(|m| {
                    m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|s| s.starts_with(TASK_CONTEXT_TAIL_MARKER))
                })
                .and_then(|m| m["content"].as_str())
                .expect("tail present")
                .to_string()
        };

        let mut p1 = Vec::new();
        let mut ctx1 = MessageBuildCtx {
            session_id: "reuse-session",
            iteration: 1,
            user_text: "Same task",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: core,
            task_context_tail: &tail,
            preserve_archived_context: true,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut p1,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };
        let built1 = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx1,
        )
        .await
        .expect("build 1");

        let mut p2 = Vec::new();
        let mut ctx2 = MessageBuildCtx {
            session_id: "reuse-session",
            iteration: 2,
            user_text: "Same task",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: core,
            task_context_tail: &tail,
            preserve_archived_context: true,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut p2,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };
        let built2 = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx2,
        )
        .await
        .expect("build 2");

        assert_eq!(
            extract_tail(&built1),
            extract_tail(&built2),
            "tail must be byte-identical across within-task iterations"
        );
    }

    /// Test 5: the final emitted tool order is name-sorted even when the input
    /// roster is unsorted — proving the sort happens as the final op before
    /// MessageBuildData, after any mutation.
    #[tokio::test]
    async fn final_tool_order_is_name_sorted_after_mutations() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Do work"))
            .await
            .expect("append user");

        // Deliberately unsorted roster (zebra, alpha, mango).
        let tool_defs: Vec<Value> = ["zebra_tool", "alpha_tool", "mango_tool"]
            .iter()
            .map(|name| {
                json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "x",
                        "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
                    }
                })
            })
            .collect();

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let mut pending = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "sort-session",
            iteration: 1,
            user_text: "Do work",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "CORE",
            task_context_tail: "",
            preserve_archived_context: true,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };
        let built = build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("build");

        let names: Vec<&str> = built
            .tool_defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str())
            .collect();
        assert_eq!(
            names,
            vec!["alpha_tool", "mango_tool", "zebra_tool"],
            "final tool_defs must be name-sorted"
        );
    }

    // ====================================================================
    // Pillar B (Task 7): turn-anchored build-phase tests (the 10 from the plan).
    // ====================================================================

    use crate::execution_policy::PolicyBundle;
    use crate::testing::{setup_test_agent, MockProvider, TestHarness};

    /// Build a payload for `session`/`user_text` at `iteration`, using the
    /// turn-anchored path. Returns the built messages.
    async fn build_payload(
        harness: &TestHarness,
        session: &str,
        user_text: &str,
        iteration: usize,
    ) -> Vec<Value> {
        build_payload_with_continuity(harness, session, user_text, iteration, true).await
    }

    async fn build_payload_with_continuity(
        harness: &TestHarness,
        session: &str,
        user_text: &str,
        iteration: usize,
        preserve_archived_context: bool,
    ) -> Vec<Value> {
        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;
        let mut ctx = MessageBuildCtx {
            session_id: session,
            iteration,
            user_text,
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "CORE-PROMPT-BYTES",
            task_context_tail: "[Task Context] tail",
            preserve_archived_context,
            prior_assistant_message_id: None,
            summary_last_message_id: None,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            redact_archived_shared_context: false,
            share_known_resources: false,
            status_tx: &status_tx,
        };
        build_turn_messages(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build")
        .messages
    }

    fn count_occurrences(messages: &[Value], needle: &str) -> usize {
        serde_json::to_string(messages)
            .unwrap_or_default()
            .matches(needle)
            .count()
    }

    /// 1. Archived turns are whole and in Archived form, positioned between the
    ///    core (index 0) and the `[Task Context]` tail.
    #[tokio::test]
    async fn archived_turns_are_whole_and_in_archived_form() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        // Two completed archived turns.
        seed_turn(
            &store,
            "s1",
            "t1",
            "first question",
            Some(("terminal", "exit_code: 0")),
            "first answer",
            "completed",
        )
        .await;
        seed_turn(
            &store,
            "s1",
            "t2",
            "second question",
            Some(("read_file", "10 lines")),
            "second answer",
            "completed",
        )
        .await;
        // Current turn (in events) under turn id t3.
        seed_turn(
            &store,
            "s1",
            "t3",
            "third question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s1", "t3").await;

        let messages = build_payload(&harness, "s1", "third question", 1).await;
        let serialized = serde_json::to_string(&messages).unwrap();

        // Core at index 0.
        assert_eq!(messages[0]["content"].as_str(), Some("CORE-PROMPT-BYTES"));
        // Both archived user messages present (whole/full).
        assert!(serialized.contains("first question"), "{serialized}");
        assert!(serialized.contains("second question"), "{serialized}");
        // Archived tool results are SUMMARIZED (compact 1-liner form).
        assert!(
            messages.iter().any(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("tool")
                    && m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|c| c.starts_with("terminal:"))
            }),
            "archived terminal result should be summarized: {serialized}"
        );
        // Exactly one tail marker, and the archived turns precede it.
        let tail_pos = messages.iter().position(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|c| c.starts_with(TASK_CONTEXT_TAIL_MARKER))
        });
        let tail_pos = tail_pos.expect("tail present");
        let first_q_pos = messages
            .iter()
            .position(|m| m.get("content").and_then(|c| c.as_str()) == Some("first question"))
            .expect("first archived user present");
        assert!(
            first_q_pos < tail_pos,
            "archived turns must precede the tail"
        );
    }

    #[tokio::test]
    async fn new_request_keeps_recent_dialogue_but_not_prior_tool_output() {
        // Top-level chat agent (depth 0); executor-mode harnesses keep the
        // strict delegated-worker floor.
        let harness = crate::testing::setup_test_agent_with_models(
            MockProvider::new(),
            "mock-model",
            "mock-model",
        )
        .await
        .unwrap();
        let store = seed_store(&harness).await;
        let session = "new-request-dialogue";
        seed_turn(
            &store,
            session,
            "kids-turn",
            "My kids?",
            Some(("manage_people", "person_record=stale-observation-7f3a")),
            "You have two daughters: Mia and Zoe. Mia is the older one.",
            "completed",
        )
        .await;
        seed_turn(
            &store,
            session,
            "ages-turn",
            "How old are they?",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, session, "ages-turn").await;

        // The relationship producer failed to link the turns: a new request.
        let payload =
            build_payload_with_continuity(&harness, session, "How old are they?", 1, false).await;
        let payload_json = serde_json::to_string(&payload).unwrap();
        assert!(payload_json.contains("My kids?"), "{payload_json}");
        assert!(
            payload_json.contains("Mia and Zoe"),
            "the referent of \"they\" must stay visible: {payload_json}"
        );
        assert!(
            !payload_json.contains("stale-observation-7f3a"),
            "an earlier task's tool output must not reach a new request: {payload_json}"
        );
        assert_eq!(
            store.session_context_boundary(session).await.unwrap(),
            None,
            "a new-request classification must not write the explicit /clear boundary"
        );

        // After a restart the in-memory anchor is gone; the dialogue survives.
        harness.agent.runtime.turn_anchors.write().await.clear();
        let after_restart =
            build_payload_with_continuity(&harness, session, "How old are they?", 1, false).await;
        assert!(serde_json::to_string(&after_restart)
            .unwrap()
            .contains("Mia and Zoe"));
    }

    #[test]
    fn new_request_dialogue_window_is_recent_bounded_and_tool_free() {
        let now = chrono::Utc::now();
        let turn = |seq: i64, age_hours: i64| {
            let at = now - chrono::Duration::hours(age_hours);
            let message = |role: &str, content: &str| Message {
                id: format!("m-{seq}-{role}"),
                session_id: "window".to_string(),
                role: role.to_string(),
                content: Some(content.to_string()),
                created_at: at,
                turn_id: Some(format!("t-{seq}")),
                ..Message::runtime_defaults()
            };
            crate::events::FetchedTurn {
                turn_id: Some(format!("t-{seq}")),
                turn_seq: seq,
                messages: vec![
                    message("user", &format!("question {seq}")),
                    Message {
                        tool_call_id: Some(format!("call-{seq}")),
                        tool_name: Some("terminal".to_string()),
                        ..message("tool", &format!("tool output {seq}"))
                    },
                    message("assistant", &format!("answer {seq}")),
                ],
                terminal_status: None,
            }
        };
        // One day-old turn, then more recent turns than the window holds.
        let mut turns = vec![turn(1, 30)];
        turns.extend((2..=10).map(|seq| turn(seq, 1)));

        let dropped = retain_recent_dialogue_for_new_request(&mut turns, now);

        assert_eq!(turns.len(), NEW_REQUEST_DIALOGUE_TURNS);
        assert_eq!(dropped, 10 - NEW_REQUEST_DIALOGUE_TURNS);
        assert_eq!(
            turns.iter().map(|turn| turn.turn_seq).collect::<Vec<_>>(),
            (5..=10).collect::<Vec<_>>(),
            "the newest turns are kept, oldest first"
        );
        assert!(turns
            .iter()
            .flat_map(|turn| &turn.messages)
            .all(|message| message.role != "tool"));
        assert!(turns[0]
            .messages
            .iter()
            .any(|message| message.content.as_deref() == Some("answer 5")));
    }

    #[tokio::test]
    async fn delegated_worker_new_task_keeps_strict_floor_and_followup_cannot_resurrect_older_task()
    {
        // Executor-mode harness (depth 1): delegated workers keep the strict
        // floor that top-level chat relaxes to a dialogue window.
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        seed_turn(
            &store,
            "new-task-floor",
            "old-task",
            "Read package metadata and use the old output format.",
            None,
            "phase=OLD\npackage_name=stale",
            "completed",
        )
        .await;
        seed_turn(
            &store,
            "new-task-floor",
            "current-task",
            "Observe a command outcome and report the current receipt.",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "new-task-floor", "current-task").await;

        let current = build_payload_with_continuity(
            &harness,
            "new-task-floor",
            "Observe a command outcome and report the current receipt.",
            1,
            false,
        )
        .await;
        let current_json = serde_json::to_string(&current).unwrap();
        assert!(current_json.contains("Observe a command outcome"));
        assert!(!current_json.contains("phase=OLD"));
        assert!(!current_json.contains("package_name=stale"));

        let floor = harness
            .agent
            .runtime
            .turn_anchors
            .read()
            .await
            .get("new-task-floor")
            .copied()
            .expect("new-task floor recorded");
        let retained = store
            .get_turns_from_anchor("new-task-floor", floor)
            .await
            .unwrap();
        assert!(retained
            .iter()
            .all(|turn| turn.turn_id.as_deref() != Some("old-task")));

        // Simulate a daemon restart. The in-memory anchor disappears, while
        // the typed new-task floor must remain authoritative in SQLite.
        harness.agent.runtime.turn_anchors.write().await.clear();

        seed_turn(
            &store,
            "new-task-floor",
            "followup-task",
            "Use only the immediately preceding receipt.",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "new-task-floor", "followup-task").await;
        let followup = build_payload_with_continuity(
            &harness,
            "new-task-floor",
            "Use only the immediately preceding receipt.",
            1,
            true,
        )
        .await;
        let followup_json = serde_json::to_string(&followup).unwrap();
        assert!(followup_json.contains("Observe a command outcome"));
        assert!(!followup_json.contains("phase=OLD"));
        assert!(!followup_json.contains("package_name=stale"));
    }

    #[tokio::test]
    async fn cold_start_keeps_prior_turn_with_inline_preview_image() {
        use std::io::Write;

        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        let mut preview = tempfile::NamedTempFile::new().expect("preview tempfile");
        preview
            .write_all(&vec![0xA5; 359_123])
            .expect("write preview bytes");

        let session = "s-cold-multimodal";
        let prior_turn = "t-pdf";
        let call_id = "call-browser-preview";
        store
            .append(Event::new(
                session,
                EventType::UserMessage,
                json!({
                    "content": "Create a two-page PDF about Headless WordPress and AI agents.",
                    "turn_id": prior_turn,
                }),
            ))
            .await
            .unwrap();
        store
            .append(Event::new(
                session,
                EventType::AssistantResponse,
                json!({
                    "content": serde_json::Value::Null,
                    "tool_calls": [{"id": call_id, "name": "browser", "arguments": "{}"}],
                    "turn_id": prior_turn,
                }),
            ))
            .await
            .unwrap();
        store
            .append(Event::new(
                session,
                EventType::ToolResult,
                json!({
                    "tool_call_id": call_id,
                    "name": "browser",
                    "result": "Rendered Headless-WordPress-AI-Agents preview",
                    "success": true,
                    "duration_ms": 1,
                    "turn_id": prior_turn,
                    "attachments": [{
                        "local_path": preview.path().to_string_lossy(),
                        "filename": "Headless-WordPress-AI-Agents-preview.png",
                        "mime_type": "image/png",
                        "size_bytes": 359123,
                        "provenance": "tool_observation",
                        "source_tool": "browser"
                    }],
                }),
            ))
            .await
            .unwrap();
        store
            .append(Event::new(
                session,
                EventType::AssistantResponse,
                json!({
                    "content": "Here's the two-page PDF: Headless WordPress in the Age of AI Agents.",
                    "turn_id": prior_turn,
                }),
            ))
            .await
            .unwrap();
        store
            .append(Event::new(
                session,
                EventType::TaskEnd,
                json!({"status": "completed", "turn_id": prior_turn}),
            ))
            .await
            .unwrap();
        seed_turn(
            &store,
            session,
            "t-current",
            "Make it 1 page PDF instead.",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, session, "t-current").await;

        let messages = build_payload(&harness, session, "Make it 1 page PDF instead.", 1).await;
        let serialized = serde_json::to_string(&messages).unwrap();

        assert!(
            serialized.contains("Create a two-page PDF about Headless WordPress"),
            "cold-start anchor dropped the prior request"
        );
        assert!(
            serialized.contains("Headless WordPress in the Age of AI Agents"),
            "cold-start anchor dropped the prior answer"
        );
        assert!(
            serialized.contains("data:image/png;base64,"),
            "archived preview should remain available to the follow-up"
        );
    }

    /// 2. Cross-turn archived stability: once turn 1 is older than the
    ///    immediate parent, its bytes are identical when built in turn 3 vs
    ///    turn 4 (render-cache hit; no fp_mismatch). The immediate parent is a
    ///    deliberate continuity suffix and graduates into this stable prefix
    ///    on the following turn.
    #[tokio::test]
    async fn cross_turn_archived_render_is_byte_stable() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        seed_turn(
            &store,
            "s2",
            "t1",
            "marker-ONE question",
            Some(("terminal", "exit_code: 0")),
            "answer one",
            "completed",
        )
        .await;

        seed_turn(
            &store,
            "s2",
            "t2",
            "marker-TWO question",
            None,
            "answer two",
            "completed",
        )
        .await;

        // Build turn 3. Turn 1 is now in the stable archived prefix, while
        // turn 2 is the continuity suffix immediately before the current user.
        seed_turn(
            &store,
            "s2",
            "t3",
            "marker-THREE question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s2", "t3").await;
        let build_a = build_payload(&harness, "s2", "marker-THREE question", 1).await;

        // Close t3, open t4 as current. Turn 1 must remain byte-identical.
        store
            .append(Event::new(
                "s2",
                EventType::TaskEnd,
                json!({"status":"completed","turn_id":"t3"}),
            ))
            .await
            .unwrap();
        seed_turn(
            &store,
            "s2",
            "t4",
            "marker-FOUR question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s2", "t4").await;
        let build_b = build_payload(&harness, "s2", "marker-FOUR question", 1).await;

        // The rendered archived-turn-1 region (everything up to its 'answer one')
        // must be byte-identical across both builds.
        let extract_t1 = |msgs: &[Value]| -> Vec<Value> {
            let end = msgs
                .iter()
                .position(|m| m.get("content").and_then(|c| c.as_str()) == Some("answer one"))
                .expect("answer one present");
            msgs[..=end].to_vec()
        };
        assert_eq!(
            extract_t1(&build_a),
            extract_t1(&build_b),
            "archived turn 1 must be byte-identical across builds (render-cache hit)"
        );
    }

    /// 3. The current turn is full/append-only (NOT summarized).
    #[tokio::test]
    async fn current_turn_is_full_not_summarized() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        let long_user = "DETAILED current request ".repeat(8);
        store
            .append(Event::new(
                "s3",
                EventType::UserMessage,
                json!({"content": long_user, "turn_id":"tc"}),
            ))
            .await
            .unwrap();
        set_current_turn(&harness, "s3", "tc").await;

        let messages = build_payload(&harness, "s3", &long_user, 1).await;
        // The full (untruncated) current user text is present verbatim.
        assert!(
            messages.iter().any(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
                    && m.get("content").and_then(|c| c.as_str()) == Some(long_user.as_str())
            }),
            "current user message must be full/verbatim"
        );
    }

    /// 4. Tail + core position preserved: exactly one tail before the immediate
    ///    parent exchange, message zero equals the core bytes, and the parent's
    ///    final answer remains adjacent to the current user.
    #[tokio::test]
    async fn tail_and_core_positions_preserved() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        seed_turn(
            &store,
            "s4",
            "t1",
            "old question",
            None,
            "old answer",
            "completed",
        )
        .await;
        seed_turn(
            &store,
            "s4",
            "t2",
            "current question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s4", "t2").await;

        let messages = build_payload(&harness, "s4", "current question", 1).await;
        assert_eq!(messages[0]["content"].as_str(), Some("CORE-PROMPT-BYTES"));
        let tail_positions: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|c| c.starts_with(TASK_CONTEXT_TAIL_MARKER))
            })
            .map(|(i, _)| i)
            .collect();
        assert_eq!(tail_positions.len(), 1, "exactly one tail marker");
        let tail_idx = tail_positions[0];
        assert_eq!(
            messages[tail_idx - 1]["content"].as_str(),
            Some("current question"),
            "tail must sit immediately after the current user message"
        );
        let parent_answer_idx = messages
            .iter()
            .position(|message| message["content"].as_str() == Some("old answer"))
            .expect("parent answer");
        assert_eq!(
            messages[parent_answer_idx + 1]["content"].as_str(),
            Some("current question"),
            "parent assistant answer must be immediately before current user"
        );
    }

    /// 5. Eviction advances the anchor: a tiny archived budget evicts the oldest
    ///    whole turns; no archived turn is partially trimmed.
    #[tokio::test]
    async fn eviction_advances_anchor_and_evicts_whole_turns() {
        let mut harness = setup_test_agent(MockProvider::new()).await.unwrap();
        // Tiny model budget so archived_budget is small but positive.
        harness.agent.runtime.context_window_config.default_budget = 5900;
        let store = seed_store(&harness).await;
        // Several archived turns with sizeable content so they exceed the budget.
        let big = "lots of detail ".repeat(20);
        for i in 1..=4 {
            seed_turn(
                &store,
                "s5",
                &format!("t{i}"),
                &format!("OLDMARK-{i} {big}"),
                None,
                &format!("answer {i}"),
                "completed",
            )
            .await;
        }
        seed_turn(
            &store,
            "s5",
            "tcur",
            "current request",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s5", "tcur").await;

        let anchor_before = harness
            .agent
            .runtime
            .turn_anchors
            .read()
            .await
            .get("s5")
            .copied();
        let messages = build_payload(&harness, "s5", "current request", 1).await;
        let anchor_after = harness
            .agent
            .runtime
            .turn_anchors
            .read()
            .await
            .get("s5")
            .copied();

        assert!(anchor_after.is_some(), "anchor recorded");
        assert!(
            anchor_after != anchor_before || anchor_before.is_none(),
            "eviction should advance the anchor"
        );
        // The oldest turn marker must have been evicted (whole-turn).
        let serialized = serde_json::to_string(&messages).unwrap();
        assert!(
            !serialized.contains("OLDMARK-1"),
            "oldest whole turn should be evicted: {serialized}"
        );
        // Current request always present.
        assert!(serialized.contains("current request"));
    }

    /// 6. Late write re-renders exactly one turn: appending a tool message under
    ///    an already-archived turn flips that turn's content_fp (one re-render),
    ///    the OTHER archived turn stays byte-identical.
    #[tokio::test]
    async fn late_write_re_renders_only_the_affected_turn() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        seed_turn(
            &store,
            "s6",
            "t1",
            "alpha question",
            Some(("terminal", "exit_code: 0")),
            "alpha answer",
            "completed",
        )
        .await;
        seed_turn(
            &store,
            "s6",
            "t2",
            "beta question",
            Some(("read_file", "5 lines")),
            "beta answer",
            "completed",
        )
        .await;
        seed_turn(
            &store,
            "s6",
            "t3",
            "current question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s6", "t3").await;

        let first = build_payload(&harness, "s6", "current question", 1).await;
        // Late write under already-archived t1: a complete assistant tool-call +
        // its tool result (a realistic background-notifier append). Both share
        // turn_id t1 and land with ids higher than t1's original rows.
        store
            .append(Event::new(
                "s6",
                EventType::AssistantResponse,
                json!({
                    "content": serde_json::Value::Null,
                    "tool_calls": [{ "id": "late-call", "name": "web_search", "arguments": "{}" }],
                    "turn_id": "t1",
                }),
            ))
            .await
            .unwrap();
        store
            .append(Event::new(
                "s6",
                EventType::ToolResult,
                json!({
                    "tool_call_id": "late-call",
                    "name": "web_search",
                    "result": "LATEWRITE-marker",
                    "success": true,
                    "duration_ms": 1,
                    "turn_id": "t1",
                }),
            ))
            .await
            .unwrap();
        let second = build_payload(&harness, "s6", "current question", 1).await;

        // t2's archived region (its user..answer) must be byte-identical across builds.
        let extract_region = |msgs: &[Value], start_marker: &str, end_marker: &str| -> Vec<Value> {
            let s = msgs
                .iter()
                .position(|m| m.get("content").and_then(|c| c.as_str()) == Some(start_marker))
                .expect("start marker");
            let e = msgs
                .iter()
                .position(|m| m.get("content").and_then(|c| c.as_str()) == Some(end_marker))
                .expect("end marker");
            msgs[s..=e].to_vec()
        };
        assert_eq!(
            extract_region(&first, "beta question", "beta answer"),
            extract_region(&second, "beta question", "beta answer"),
            "unaffected archived turn t2 must stay byte-identical"
        );
        // t1's archived region (alpha question .. just before beta question) MUST
        // change: the late tool step adds a new summarized tool message at the END
        // of t1 (highest id within the turn), so the region capture must extend
        // past "alpha answer" up to the start of the next turn.
        let extract_t1 = |msgs: &[Value]| -> Vec<Value> {
            let s = msgs
                .iter()
                .position(|m| m.get("content").and_then(|c| c.as_str()) == Some("alpha question"))
                .expect("alpha question");
            let e = msgs
                .iter()
                .position(|m| m.get("content").and_then(|c| c.as_str()) == Some("beta question"))
                .expect("beta question");
            msgs[s..e].to_vec()
        };
        let first_t1 = extract_t1(&first);
        let second_t1 = extract_t1(&second);
        assert_ne!(
            first_t1, second_t1,
            "affected archived turn t1 must be re-rendered after the late write"
        );
        // The late web_search step appears as a new summarized tool message.
        assert!(
            second_t1.iter().any(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("tool")
                    && m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|c| c.starts_with("web_search:"))
            }),
            "late web_search step must appear (summarized) in the re-rendered turn"
        );
    }

    /// 7. No synthetic user / no age_collapse fingerprint on the happy path.
    #[tokio::test]
    async fn no_synthetic_user_and_no_age_collapse_on_happy_path() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        seed_turn(
            &store,
            "s7",
            "t1",
            "current question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s7", "t1").await;

        let messages = build_payload(&harness, "s7", "current question", 1).await;
        let serialized = serde_json::to_string(&messages).unwrap();
        // No synthetic-user id leaks into content (the in-process fallback did
        // not fire because the current turn is present in the fetch).
        assert!(
            !serialized.contains("synthetic-current-user-"),
            "no synthetic user id on the happy path: {serialized}"
        );
        // No legacy age_collapse stage marker anywhere.
        assert!(!serialized.contains("age_collapse"));
    }

    /// 8. Current-turn fallback fires when the current row is absent: simulate
    ///    the race (current turn not yet in events) and assert the payload still
    ///    ends in the current user turn (full/append-only).
    #[tokio::test]
    async fn current_turn_fallback_injects_when_row_absent() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        // One archived turn exists, but the CURRENT turn's user row is NOT yet
        // committed to events (the documented race). current_turn_ids points at
        // a turn id with no rows.
        seed_turn(
            &store,
            "s8",
            "t1",
            "old question",
            None,
            "old answer",
            "completed",
        )
        .await;
        set_current_turn(&harness, "s8", "t-not-committed").await;

        let messages = build_payload(&harness, "s8", "RACE current text", 1).await;
        // Payload must still end in the current user message (full/append-only).
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"));
        assert!(
            last_user.is_some_and(
                |m| m.get("content").and_then(|c| c.as_str()) == Some("RACE current text")
            ),
            "fallback must inject the current user message: {:?}",
            serde_json::to_string(&messages)
        );
    }

    /// 9. Render cache prunes on eviction: after an eviction advances the anchor,
    ///    `turn_renders[session]` no longer holds entries for evicted turn ids.
    #[tokio::test]
    async fn render_cache_prunes_evicted_turns() {
        let mut harness = setup_test_agent(MockProvider::new()).await.unwrap();
        harness.agent.runtime.context_window_config.default_budget = 5900;
        let store = seed_store(&harness).await;
        let big = "detail ".repeat(20);
        for i in 1..=4 {
            seed_turn(
                &store,
                "s9",
                &format!("t{i}"),
                &format!("turn {i} {big}"),
                None,
                &format!("answer {i}"),
                "completed",
            )
            .await;
        }
        seed_turn(
            &store,
            "s9",
            "tcur",
            "current request",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s9", "tcur").await;

        let _ = build_payload(&harness, "s9", "current request", 1).await;
        let cache = harness.agent.runtime.turn_renders.read().await;
        let session_cache = cache.get("s9");
        // The oldest evicted turn ids must NOT remain cached (no unbounded growth).
        if let Some(sc) = session_cache {
            assert!(
                !sc.contains_key("t1"),
                "evicted turn t1 must be pruned from the render cache"
            );
        }
        // Anchor advanced past t1.
        let anchor = harness
            .agent
            .runtime
            .turn_anchors
            .read()
            .await
            .get("s9")
            .copied();
        assert!(anchor.is_some());
    }

    /// 10. No duplicate pinned-history path: unique content markers appear at
    ///     most once, and every emitted historical marker belongs to a turn at
    ///     or after the anchor.
    #[tokio::test]
    async fn no_duplicate_pinned_history_path() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        // More than the former 20-message recency window, each with a unique marker.
        for i in 1..=12 {
            seed_turn(
                &store,
                "s10",
                &format!("t{i}"),
                &format!("UNIQUEMARK-{i:02}-X"),
                None,
                &format!("reply-{i}"),
                "completed",
            )
            .await;
        }
        seed_turn(
            &store,
            "s10",
            "tcur",
            "current request UNIQUEMARK-CUR",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s10", "tcur").await;

        let messages = build_payload(&harness, "s10", "current request UNIQUEMARK-CUR", 1).await;

        // Every marker present occurs at most once (no duplicate canonical-history path).
        for i in 1..=12 {
            let needle = format!("UNIQUEMARK-{i:02}-X");
            let occ = count_occurrences(&messages, &needle);
            assert!(
                occ <= 1,
                "marker {needle} must occur at most once, got {occ}"
            );
        }

        // Every emitted historical marker belongs to a turn whose turn_seq >= anchor.
        let anchor = harness
            .agent
            .runtime
            .turn_anchors
            .read()
            .await
            .get("s10")
            .copied()
            .expect("anchor recorded");
        let store2 = seed_store(&harness).await;
        let turns = store2.get_turns_from_anchor("s10", anchor).await.unwrap();
        let serialized = serde_json::to_string(&messages).unwrap();
        for i in 1..=12 {
            let needle = format!("UNIQUEMARK-{i:02}-X");
            if serialized.contains(&needle) {
                let belongs = turns.iter().any(|t| {
                    t.turn_seq >= anchor
                        && t.messages
                            .iter()
                            .any(|m| m.content.as_deref().is_some_and(|c| c.contains(&needle)))
                });
                assert!(
                    belongs,
                    "emitted marker {needle} must belong to a turn >= anchor"
                );
            }
        }

        // Compile-time guarantee that the pinned_memories plumbing is gone:
        // MessageBuildCtx has no such field (this references all current fields).
        let _assert_no_pinned = |c: &MessageBuildCtx| {
            let MessageBuildCtx {
                session_id: _,
                iteration: _,
                user_text: _,
                current_attachments: _,
                completed_tool_calls: _,
                model: _,
                core_prompt: _,
                task_context_tail: _,
                preserve_archived_context: _,
                prior_assistant_message_id: _,
                summary_last_message_id: _,
                tool_defs: _,
                policy_bundle: _,
                pending_system_messages: _,
                empty_response_retry_pending: _,
                redact_archived_shared_context: _,
                share_known_resources: _,
                status_tx: _,
            } = c;
        };
    }

    #[test]
    fn fresh_context_isolation_preserves_multimodal_user_message() {
        let user_text = "What do you see here?";
        let mut messages = vec![
            json!({"role": "system", "content": "core"}),
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }),
        ];
        let non_system_non_user_count = messages
            .iter()
            .filter(|m| {
                let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
                role != "system" && role != "user"
            })
            .count();
        assert_eq!(non_system_non_user_count, 0);
        messages.retain(|m| {
            m.get("role").and_then(|r| r.as_str()) != Some("user")
                || m.get("content")
                    .is_some_and(|content| user_message_content_matches(content, user_text))
        });
        assert_eq!(messages.len(), 2);
        assert!(
            messages
                .iter()
                .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("user")),
            "multimodal user message must survive fresh-context retain"
        );
    }
}
