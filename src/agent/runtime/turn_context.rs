//! Turn-context assembly and the session history appends live in
//! `aidaemon-runtime` ([`aidaemon_runtime::turn::turn_context`]); the agent
//! gets them as a [`aidaemon_runtime::host::TurnHost`].

#[cfg(test)]
use super::*;
#[cfg(test)]
use super::{completion_contract, followup};
#[cfg(test)]
use crate::traits::ToolCallSemantics;
#[cfg(test)]
use aidaemon_runtime::turn::project_scope::drop_inbox_scopes;
#[allow(unused_imports)]
pub(in crate::agent) use aidaemon_runtime::turn::turn_context::*;

#[cfg(test)]
mod tests {
    use super::completion_contract::CompletionTaskKind;
    use super::followup::FollowupMode;
    use super::*;
    use chrono::Utc;

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

    #[tokio::test]
    async fn claimed_pre_dispatch_denial_uses_kernel_admission_and_typed_enforcement() {
        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .expect("test harness");
        let task_id = "task-claimed-pre-dispatch";
        let session_id = "session-claimed-pre-dispatch";
        let emitter =
            crate::events::EventEmitter::new(harness.agent.event_store.clone(), session_id)
                .with_task_id(task_id);
        let call = ToolCall {
            id: "call-claimed-pre-dispatch".to_string(),
            name: "write_file".to_string(),
            arguments: r#"{"path":"/synthetic/denied","content":"x"}"#.to_string(),
            extra_content: None,
        };
        let claim = crate::events::TaskKernelOperationClaim {
            operation_id: call.id.clone(),
            stable_operation_key: "synthetic:denied-write".to_string(),
            tool_name: call.name.clone(),
            obligation_ids: Vec::new(),
            max_attempts: 1,
            max_invocations: 1,
            idempotency_key: Some("synthetic:denied-write:key".to_string()),
            operation_lineage: None,
        };
        let denial = crate::traits::ToolAccessDenial {
            reason_code: "controller_scope_contract_rejected".to_string(),
            enforcement: crate::traits::ToolAccessEnforcement::ControllerEnforced,
            exit_code: None,
            proposed_evidence: Vec::new(),
        };

        harness
            .agent
            .persist_pre_dispatch_outcome_with_kernel_claim(
                &emitter,
                session_id,
                task_id,
                &call,
                &call.arguments,
                "synthetic scope refusal".to_string(),
                crate::traits::ToolOutcomeStatus::Blocked,
                crate::traits::ToolInvocationStage::RejectedBeforeDispatch,
                true,
                ToolCallSemantics::default(),
                None,
                Some(denial.clone()),
                Some(&claim),
            )
            .await
            .expect("persist claimed pre-dispatch outcome");

        let events = harness
            .agent
            .event_store
            .query_task_events_for_session(session_id, task_id)
            .await
            .expect("query events");
        assert_eq!(events.len(), 2);
        let persisted_call = events[0]
            .parse_data::<crate::events::ToolCallData>()
            .expect("typed call");
        assert_eq!(
            persisted_call.stable_operation_key.as_deref(),
            Some("synthetic:denied-write")
        );
        assert_eq!(
            persisted_call.idempotency_key.as_deref(),
            Some("synthetic:denied-write:key")
        );
        let result = events[1]
            .parse_data::<ToolResultData>()
            .expect("typed result");
        let receipt = result.receipt.expect("canonical receipt");
        assert!(receipt.contract_rejected);
        assert_eq!(receipt.access_denial, Some(denial));
    }

    #[test]
    fn recent_parent_messages_strip_intent_gate_payload() {
        let history = vec![
            msg("user", "Build a site"),
            msg(
                "assistant",
                "Sure, I can help.\n[INTENT_GATE] {\"can_answer_now\":false}",
            ),
        ];
        let messages = extract_recent_parent_messages(&history, 6);
        assert_eq!(messages.len(), 2);
        let assistant_content = messages[1]
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(!assistant_content.contains("[INTENT_GATE]"));
        assert_eq!(assistant_content, "Sure, I can help.");
    }
    #[test]
    fn recent_parent_messages_include_recent_api_and_file_tool_summaries() {
        let history = vec![
            msg("user", "Look up those studies."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some(crate::tools::sanitize::wrap_untrusted_output(
                    "http_request",
                    "HTTP 200 OK\ncontent-type: application/json\n\n{\"studies\":[]}",
                )),
                tool_call_id: Some("call-http".to_string()),
                tool_name: Some("http_request".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some("File sent: studies.json (127 KB)".to_string()),
                tool_call_id: Some("call-file".to_string()),
                tool_name: Some("send_file".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg("assistant", "I fetched the results and sent the file."),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| content.contains("http_request: HTTP 200 OK"))
        }));
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| content.contains("send_file: File sent: studies.json"))
        }));
    }
    #[test]
    fn recent_parent_messages_preserve_http_error_detail_beyond_headers() {
        let history = vec![
            msg("user", "Check that trial ID."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some(crate::tools::sanitize::wrap_untrusted_output(
                    "http_request",
                    "HTTP 404 Not Found\ncontent-type: application/json\n\nNot Found: NCT05178195",
                )),
                tool_call_id: Some("call-http".to_string()),
                tool_name: Some("http_request".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg(
                "assistant",
                "The API could not find that study, but I should confirm the exact error.",
            ),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| {
                        content
                            .contains("http_request: HTTP 404 Not Found | Not Found: NCT05178195")
                    })
        }));
    }
    #[test]
    fn recent_parent_messages_include_web_fetch_error_summaries() {
        let history = vec![
            msg("user", "Check that page."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some(crate::tools::sanitize::wrap_untrusted_output(
                    "web_fetch",
                    "Error fetching https://example.com/missing: HTTP 404 Not Found",
                )),
                tool_call_id: Some("call-web-fetch".to_string()),
                tool_name: Some("web_fetch".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg("assistant", "That page fetch failed."),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row.get("content").and_then(|v| v.as_str()).is_some_and(|content| {
                    content.contains("web_fetch: Error fetching https://example.com/missing: HTTP 404 Not Found")
                    })
        }));
    }
    #[test]
    fn recent_parent_messages_include_generic_terminal_evidence_summary() {
        let history = vec![
            msg("user", "Run the tests."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some("pytest\nAssertionError: expected 1 but got 2".to_string()),
                tool_call_id: Some("call-terminal".to_string()),
                tool_name: Some("terminal".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg("assistant", "The tests failed."),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| {
                        content.contains("terminal: pytest | AssertionError: expected 1 but got 2")
                    })
        }));
    }
    #[tokio::test]
    async fn bound_goal_workspace_beats_mission_text_scope_extraction() {
        use crate::testing::{setup_test_agent_root, MockProvider};
        use crate::traits::store_prelude::*;

        // Root-mode agent: the binding applies to non-executor roles (the
        // task lead and direct runs); executors keep their provisioned
        // attempt workspace.
        let mut harness = setup_test_agent_root(MockProvider::new())
            .await
            .expect("test harness");
        let workspace = tempfile::tempdir().expect("bound workspace");
        let workspace_path = workspace.path().to_string_lossy().to_string();
        let mut goal = crate::traits::Goal::new_finite("Manage the blog daily", "test-session");
        goal.context = Some(
            serde_json::json!({
                "project_scope": workspace_path,
                "project_scope_bound_at": "2026-08-27T00:00:00Z",
            })
            .to_string(),
        );
        harness.state.create_goal(&goal).await.expect("create goal");
        harness.agent.set_test_goal_id(Some(goal.id.clone()));

        // The mission text explicitly names an unrelated repository path (the
        // classic failure: instructions mention `src/content/posts` and a
        // hostname containing another project's name). The typed durable
        // workspace binding must win over anything extracted from text.
        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history(
                "test-session",
                "Each day manage the blog at https://blog.acme-daemon.dev/. Create the \
                 post in /tmp/synthetic-alice/projects/acme-daemon/src/content/posts, then build \
                 and deploy with the documented workflow.",
            )
            .await;

        assert_eq!(
            turn_context.primary_project_scope.as_deref(),
            Some(workspace_path.as_str()),
            "typed goal workspace binding must be the authoritative scope"
        );
        assert_eq!(
            turn_context.explicit_workspace(),
            Some(workspace_path.as_str()),
            "a goal's bind_workspace binding is explicit"
        );
        assert_eq!(
            turn_context.authorized_project_scopes,
            vec![workspace_path.clone()],
            "text-extracted paths outside the binding are not authorized"
        );

        // Without a binding, text extraction keeps working as before.
        harness.agent.set_test_goal_id(None);
        let unbound = harness
            .agent
            .build_turn_context_from_recent_history(
                "test-session",
                "Create the post in /tmp/synthetic-alice/projects/acme-daemon/src/content/posts.",
            )
            .await;
        assert_eq!(
            unbound.primary_project_scope.as_deref(),
            Some("/tmp/synthetic-alice/projects/acme-daemon/src/content/posts"),
        );
    }

    /// Lay out a synthetic bound workspace shaped like a static-site checkout.
    /// The daemon cwd (this repository) also has `src/`, so an un-anchored
    /// `src/content/posts` would silently resolve there — the regression.
    fn synthetic_site_workspace() -> (tempfile::TempDir, String) {
        let workspace = tempfile::tempdir().expect("bound workspace");
        std::fs::create_dir_all(workspace.path().join(".git")).expect("workspace marker");
        std::fs::create_dir_all(workspace.path().join("src/content/posts")).expect("posts dir");
        let path = workspace.path().to_string_lossy().to_string();
        (workspace, path)
    }

    #[tokio::test]
    async fn relative_paths_in_mission_text_anchor_to_the_bound_goal_workspace() {
        use crate::testing::{setup_test_agent_root, MockProvider};
        use crate::traits::store_prelude::*;

        let mut harness = setup_test_agent_root(MockProvider::new())
            .await
            .expect("test harness");
        let (_workspace, workspace_path) = synthetic_site_workspace();
        let mut goal = crate::traits::Goal::new_finite("Manage the blog daily", "test-session");
        goal.context = Some(
            serde_json::json!({
                "project_scope": workspace_path,
                "project_scope_bound_at": "2026-08-27T00:00:00Z",
            })
            .to_string(),
        );
        harness.state.create_goal(&goal).await.expect("create goal");
        harness.agent.set_test_goal_id(Some(goal.id.clone()));
        let cwd = std::env::current_dir().expect("cwd");
        assert!(
            cwd.join("src").is_dir(),
            "precondition: cwd has a competing src/"
        );

        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history(
                "test-session",
                "Write today's post as a new markdown file in src/content/posts, then build.",
            )
            .await;

        assert_eq!(
            turn_context.bound_workspace.as_deref(),
            Some(workspace_path.as_str())
        );
        assert_eq!(
            turn_context.primary_project_scope.as_deref(),
            Some(workspace_path.as_str())
        );
        assert_eq!(
            turn_context.authorized_project_scopes,
            vec![workspace_path.clone()]
        );
        let path_targets = turn_context
            .completion_contract
            .verification_targets
            .iter()
            .filter(|target| {
                target.kind == super::completion_contract::VerificationTargetKind::Path
            })
            .map(|target| target.value.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            path_targets,
            vec![format!("{workspace_path}/src/content/posts")],
            "verification target anchored to the bound workspace, not the daemon cwd"
        );
    }

    #[tokio::test]
    async fn executor_inherited_workspace_anchors_relative_handoff_paths() {
        use crate::testing::{setup_test_agent, MockProvider};

        // `setup_test_agent` builds an executor-role agent: the workspace a
        // parent hands to it is its typed binding.
        let mut harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        let (_workspace, workspace_path) = synthetic_site_workspace();
        harness.agent.inherited_project_scope = Some(workspace_path.clone());

        for handoff in [
            "Create the post file in src/content/posts and run the build.",
            "target_dir=src/content/posts action=create_post then verify the build output",
        ] {
            let turn_context = harness
                .agent
                .build_turn_context_from_recent_history("test-session", handoff)
                .await;
            assert_eq!(
                turn_context.bound_workspace.as_deref(),
                Some(workspace_path.as_str()),
                "{handoff}"
            );
            assert_eq!(
                turn_context.explicit_workspace(),
                None,
                "a workspace a parent handed down is inferred: {handoff}"
            );
            assert_eq!(
                turn_context.primary_project_scope.as_deref(),
                Some(workspace_path.as_str()),
                "{handoff}"
            );
            assert_eq!(
                turn_context.authorized_project_scopes,
                vec![workspace_path.clone()],
                "{handoff}"
            );
        }

        // Negative control: a handoff naming another repository cannot move the
        // executor out of its provisioned workspace.
        let (_elsewhere, elsewhere_path) = synthetic_site_workspace();
        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history(
                "test-session",
                &format!("Create the post in {elsewhere_path}/src/content/posts."),
            )
            .await;
        assert_eq!(
            turn_context.primary_project_scope.as_deref(),
            Some(workspace_path.as_str())
        );
        assert_eq!(
            turn_context.authorized_project_scopes,
            vec![workspace_path.clone()]
        );

        // Without an inherited workspace there is no binding and the explicit
        // path in the handoff scopes the turn as before.
        harness.agent.inherited_project_scope = None;
        let unbound = harness
            .agent
            .build_turn_context_from_recent_history(
                "test-session",
                &format!("Create the post in {elsewhere_path}/src/content/posts."),
            )
            .await;
        assert_eq!(unbound.bound_workspace, None);
        assert_eq!(
            unbound.primary_project_scope.as_deref(),
            Some(elsewhere_path.as_str()),
            "existing path promotes to its own project root"
        );
    }

    #[tokio::test]
    async fn new_task_context_excludes_unrelated_parent_messages() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg(
                "user",
                "Please work in ~/projects/blog.aidaemon.ai/src/content/posts",
            ))
            .await
            .expect("append prior user");
        harness
            .state
            .append_message(&msg("assistant", "Which posts should I update?"))
            .await
            .expect("append prior assistant");

        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history("test-session", "Why?")
            .await;

        assert_eq!(turn_context.followup_mode, Some(FollowupMode::NewTask));
        assert!(turn_context.visible_antecedent_user_message_id.is_none());
    }

    #[tokio::test]
    async fn unresolved_dialogue_state_carries_original_completion_obligations() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::{
            DialogueState, DialogueStateStore, OpenRequest, OpenRequestStatus, UserTurnKind,
            UserTurnSummary,
        };

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        let now = Utc::now();
        let mut state = DialogueState::new("test-session");
        state.open_request = Some(OpenRequest {
            user_message_id: "request-1".to_string(),
            text: "Fix the scheduler bug in this repository and verify the tests.".to_string(),
            status: OpenRequestStatus::PartiallyAnswered,
            task_id: Some("task-1".to_string()),
            project_scope: None,
            completion_contract: Some(crate::traits::RequestCompletionContract {
                scope_task_id: Some("task-1".to_string()),
                adopted_from_task_ids: Vec::new(),
                task_kind: crate::traits::RequestTaskKind::Diagnose,
                expects_mutation: true,
                required_mutation_effects: crate::traits::ToolMutationEffects::LOCAL_SOURCE_WRITE,
                forbids_mutation: false,
                forbids_tool_use: false,
                allowed_tool_names: Vec::new(),
                forbidden_tool_scopes: Vec::new(),
                dispatch_stop_rules: Vec::new(),
                required_response_fields: Vec::new(),
                response_contract: None,
                forbidden_actions: Vec::new(),
                requires_observation: true,
                requires_reverification_after_mutation: true,
                explicit_verification_requested: true,
                minimum_sources: 0,
                requires_primary_sources: false,
                requires_exact_history: false,
                evidence_requirements: Vec::new(),
                adopted_evidence_bindings: Vec::new(),
                verification_targets: Vec::new(),
            }),
            opened_at: now,
            resolved_at: None,
        });
        state.last_user_turn = Some(UserTurnSummary {
            message_id: "followup-1".to_string(),
            kind: UserTurnKind::Followup,
            text: "Fixed?".to_string(),
        });
        harness
            .state
            .upsert_dialogue_state(&state)
            .await
            .expect("persist dialogue state");

        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history("test-session", "Fixed?")
            .await;

        assert_eq!(turn_context.followup_mode, Some(FollowupMode::Followup));
        assert!(turn_context
            .goal_user_text
            .contains(&state.open_request.unwrap().text));
        assert_eq!(
            turn_context.completion_contract.task_kind,
            CompletionTaskKind::Diagnose
        );
        assert!(turn_context.completion_contract.expects_mutation);
        assert!(turn_context
            .completion_contract
            .required_mutation_effects
            .contains(crate::traits::ToolMutationEffects::LOCAL_SOURCE_WRITE));
    }

    #[tokio::test]
    async fn internal_continuation_output_cannot_replace_request_verification_target() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::{
            DialogueState, DialogueStateStore, MessageStore, OpenRequest, OpenRequestStatus,
        };

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        let original = "Find all the information about the Brunswick Allenton 8-foot pool table, including its Pros/cons.";
        let now = Utc::now();
        let mut state = DialogueState::new("test-session");
        state.open_request = Some(OpenRequest {
            user_message_id: "request-1".to_string(),
            text: original.to_string(),
            status: OpenRequestStatus::InProgress,
            task_id: Some("task-1".to_string()),
            project_scope: None,
            completion_contract: Some(crate::traits::RequestCompletionContract {
                scope_task_id: Some("task-1".to_string()),
                adopted_from_task_ids: Vec::new(),
                task_kind: crate::traits::RequestTaskKind::Find,
                expects_mutation: false,
                required_mutation_effects: crate::traits::ToolMutationEffects::NONE,
                forbids_mutation: false,
                forbids_tool_use: false,
                allowed_tool_names: Vec::new(),
                forbidden_tool_scopes: Vec::new(),
                dispatch_stop_rules: Vec::new(),
                required_response_fields: Vec::new(),
                response_contract: None,
                forbidden_actions: Vec::new(),
                requires_observation: true,
                requires_reverification_after_mutation: false,
                explicit_verification_requested: false,
                minimum_sources: 0,
                requires_primary_sources: false,
                requires_exact_history: false,
                evidence_requirements: Vec::new(),
                adopted_evidence_bindings: Vec::new(),
                verification_targets: Vec::new(),
            }),
            opened_at: now,
            resolved_at: None,
        });
        harness
            .state
            .upsert_dialogue_state(&state)
            .await
            .expect("persist dialogue state");

        let continuation = "[Background command completed]\nExit status: completed\nOutput:\nSee https://goo.gle/gemini-cli-auth-docs#workspace-gca";
        harness
            .state
            .append_message(&Message {
                content: Some(continuation.to_string()),
                annotations: vec![crate::traits::MessageAnnotation::InternalContinuation],
                ..Message::new_runtime("continuation-1", "test-session", "user")
            })
            .await
            .expect("append continuation evidence");

        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history_with_origin("test-session", continuation, true)
            .await;

        assert_eq!(turn_context.followup_mode, Some(FollowupMode::Followup));
        assert_eq!(
            turn_context.completion_contract.task_kind,
            CompletionTaskKind::Find
        );
        assert!(turn_context.completion_contract.requires_observation);
        assert!(turn_context
            .completion_contract
            .verification_targets
            .is_empty());
        assert!(turn_context.goal_user_text.contains(original));
        assert!(turn_context.goal_user_text.contains("workspace-gca"));
    }

    async fn turn_context_after_request(antecedent: &str, current: &str) -> TurnContext {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::{
            DialogueState, DialogueStateStore, MessageStore, UserTurnKind, UserTurnSummary,
        };

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        for (id, role, text) in [
            ("m0", "user", antecedent),
            ("m1", "assistant", "Looked into it."),
            ("m-current", "user", current),
        ] {
            harness
                .state
                .append_message(&Message {
                    content: Some(text.to_string()),
                    ..Message::new_runtime(id, "test-session", role)
                })
                .await
                .expect("append history");
        }
        let mut state = DialogueState::new("test-session");
        state.last_user_turn = Some(UserTurnSummary {
            message_id: "m-current".to_string(),
            kind: UserTurnKind::Followup,
            text: current.to_string(),
        });
        harness
            .state
            .upsert_dialogue_state(&state)
            .await
            .expect("persist dialogue state");
        harness
            .agent
            .build_turn_context_from_recent_history("test-session", current)
            .await
    }

    /// The saved path in a received-file notice is daemon storage. A follow-up
    /// must not confine itself there, however it is worded.
    #[tokio::test]
    async fn a_received_file_path_never_becomes_the_followup_scope() {
        let inbox = crate::testing::TEST_INBOX_DIR;
        std::fs::create_dir_all(inbox).unwrap();
        let antecedent = format!(
            "[File received: photo.jpg (29 KB, image/jpeg)\nSaved to: {inbox}/photo.jpg]\nWhy did this fail?"
        );
        for current in ["Yes, you are approved", "go ahead and fix it"] {
            let turn_context = turn_context_after_request(&antecedent, current).await;
            assert_eq!(turn_context.followup_mode, Some(FollowupMode::Followup));
            assert_eq!(turn_context.primary_project_scope, None, "{current}");
            assert!(
                turn_context.authorized_project_scopes.is_empty(),
                "{current}"
            );
        }
    }

    #[tokio::test]
    async fn a_real_path_beside_a_received_file_still_carries_over() {
        let inbox = crate::testing::TEST_INBOX_DIR;
        std::fs::create_dir_all(inbox).unwrap();
        let project = tempfile::TempDir::new().unwrap();
        let project_path = project.path().to_string_lossy().to_string();
        let antecedent = format!(
            "[File received: photo.jpg (29 KB, image/jpeg)\nSaved to: {inbox}/photo.jpg]\nFix this in {project_path}"
        );
        let turn_context = turn_context_after_request(&antecedent, "Yes, go ahead").await;
        assert_eq!(
            turn_context.primary_project_scope.as_deref(),
            Some(project_path.as_str())
        );
    }

    #[test]
    fn inbox_scopes_drop_and_others_stay() {
        let inbox = std::path::Path::new("/tmp/synthetic-inbox");
        let mut scopes = vec![
            "/tmp/synthetic-inbox".to_string(),
            "/tmp/synthetic-inbox/sub".to_string(),
            "/tmp/synthetic-inbox-sibling".to_string(),
            "/tmp/project".to_string(),
        ];
        drop_inbox_scopes(&mut scopes, Some(inbox));
        assert_eq!(scopes, ["/tmp/synthetic-inbox-sibling", "/tmp/project"]);
        drop_inbox_scopes(&mut scopes, None);
        assert_eq!(scopes.len(), 2);
    }

    #[tokio::test]
    async fn followup_inherits_antecedent_scope_over_older_project_root() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::{
            DialogueState, DialogueStateStore, MessageStore, UserTurnKind, UserTurnSummary,
        };

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        // An older turn names a real project root (has a manifest); the
        // antecedent request names a scratch dir with no project marker.
        let repo = tempfile::TempDir::new().unwrap();
        std::fs::write(repo.path().join("Cargo.toml"), "[package]\n").unwrap();
        let scratch = tempfile::TempDir::new().unwrap();
        let repo_path = repo.path().to_string_lossy().to_string();
        let scratch_path = scratch.path().to_string_lossy().to_string();

        let turns = [
            ("user", format!("Look at the build script in {repo_path}")),
            ("assistant", "Done.".to_string()),
            (
                "user",
                format!("In {scratch_path} build a tiny Python package with tests"),
            ),
            (
                "assistant",
                "Created the package; 6 tests pass.".to_string(),
            ),
        ];
        for (index, (role, text)) in turns.iter().enumerate() {
            harness
                .state
                .append_message(&Message {
                    content: Some(text.clone()),
                    ..Message::new_runtime(format!("m{index}"), "test-session", *role)
                })
                .await
                .expect("append history");
        }
        let current = "now add a command-line entry point and prove it works";
        harness
            .state
            .append_message(&Message {
                content: Some(current.to_string()),
                ..Message::new_runtime("m-current", "test-session", "user")
            })
            .await
            .expect("append current");
        let mut state = DialogueState::new("test-session");
        state.last_user_turn = Some(UserTurnSummary {
            message_id: "m-current".to_string(),
            kind: UserTurnKind::Followup,
            text: current.to_string(),
        });
        harness
            .state
            .upsert_dialogue_state(&state)
            .await
            .expect("persist dialogue state");

        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history("test-session", current)
            .await;

        assert_eq!(turn_context.followup_mode, Some(FollowupMode::Followup));
        assert_eq!(
            turn_context.primary_project_scope.as_deref(),
            Some(scratch_path.as_str())
        );
        assert_eq!(turn_context.authorized_project_scopes, vec![scratch_path]);
    }

    #[tokio::test]
    async fn build_turn_context_does_not_reuse_old_local_scope_for_external_followup() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg(
                "user",
                "Please work in /Users/jordan/projects/springfield-il-site and inspect the logs.",
            ))
            .await
            .expect("append old local user");
        harness
            .state
            .append_message(&msg(
                "assistant",
                "I am currently locked to /Users/jordan/projects/springfield-il-site.",
            ))
            .await
            .expect("append old local assistant");
        harness
            .state
            .append_message(&msg(
                "user",
                "Find melanoma clinical trials recruiting near Springfield, Illinois.",
            ))
            .await
            .expect("append external user");
        harness
            .state
            .append_message(&msg(
                "assistant",
                "I found 10 recruiting trials near Springfield, VA with male participants.",
            ))
            .await
            .expect("append external assistant");

        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history(
                "test-session",
                "Give me all the info about the top 2.",
            )
            .await;

        assert_eq!(turn_context.primary_project_scope, None);
    }
    #[tokio::test]
    async fn build_turn_context_keeps_detailed_schedule_request_separate_from_previous_task() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg(
                "user",
                "Can you post your daily blog post on your blog?",
            ))
            .await
            .expect("append prior user");
        harness
            .state
            .append_message(&msg("assistant", "Would you like me to schedule that too?"))
            .await
            .expect("append prior assistant");

        let current = "Can you set up a daily scheduled task at 6:00 am to publish the blog with honest reflections about recent errors and fixes.";
        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history("test-session", current)
            .await;

        assert_eq!(turn_context.followup_mode, Some(FollowupMode::NewTask));
        // goal_user_text always contains the current request
        assert!(
            turn_context.goal_user_text.contains(current),
            "goal_user_text should contain current request: {}",
            turn_context.goal_user_text
        );
        assert_eq!(
            turn_context.completion_contract.task_kind,
            CompletionTaskKind::Conversational,
            "bootstrap context must not infer schedule semantics from request words"
        );
        assert!(!turn_context.completion_contract.requires_observation);
    }

    #[tokio::test]
    async fn build_turn_context_does_not_carry_prior_user_into_new_task_goal_text() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg(
                "user",
                "Yes, just testing your handling of garbled input, nothing needed.",
            ))
            .await
            .expect("append prior user");
        harness
            .state
            .append_message(&msg("assistant", "Understood."))
            .await
            .expect("append prior assistant");

        let current = "Check whether thinking output is being exposed to the user.";
        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history("test-session", current)
            .await;

        assert_eq!(turn_context.followup_mode, Some(FollowupMode::NewTask));
        assert_eq!(turn_context.goal_user_text, current);
        assert!(!turn_context.goal_user_text.contains("Original request:"));
        assert!(!turn_context.goal_user_text.contains("Current request:"));
    }

    #[tokio::test]
    async fn persistence_override_keeps_raw_tool_result_ephemeral() {
        use crate::events::{EventEmitter, EventType, ToolResultData};
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        let session_id = "ephemeral-history-result";
        let emitter = EventEmitter::new(harness.agent.event_store.clone(), session_id);
        let message = Message {
            content: Some("verbatim historical evidence".to_string()),
            tool_call_id: Some("call-1".to_string()),
            tool_name: Some("search_history".to_string()),
            ..Message::new_runtime("result-1", session_id, "tool")
        };
        harness
            .agent
            .append_tool_message_with_result_event_policy(
                &emitter,
                &message,
                true,
                1,
                None,
                Some("task-1"),
                Some("search_history returned 1 authorized item"),
            )
            .await
            .unwrap();

        let events = harness
            .agent
            .event_store
            .query_recent_events(session_id, 10)
            .await
            .unwrap();
        let tool_result = events
            .iter()
            .find(|event| event.event_type == EventType::ToolResult)
            .unwrap()
            .parse_data::<ToolResultData>()
            .unwrap();
        assert_eq!(
            tool_result.result,
            "search_history returned 1 authorized item"
        );
        assert!(!tool_result.result.contains("historical evidence"));

        let working = harness.state.get_history(session_id, 10).await.unwrap();
        assert!(working
            .iter()
            .any(|message| message.content.as_deref() == Some("verbatim historical evidence")));
    }

    #[tokio::test]
    async fn tool_artifact_gets_durable_resource_identity() {
        use crate::events::{EventEmitter, EventType, ResourceRegisteredData};
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::{AttachmentProvenance, MessageAttachment, MessageStore};

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        let temp = tempfile::tempdir().expect("tempdir");
        let artifact = temp.path().join("report.pdf");
        std::fs::write(&artifact, b"%PDF-1.7 resource test").expect("write artifact");
        let session_id = "resource-registration";
        let emitter = EventEmitter::new(harness.agent.event_store.clone(), session_id);
        let message = Message {
            content: Some("Rendered a PDF.".to_string()),
            tool_call_id: Some("browser-call".to_string()),
            tool_name: Some("browser".to_string()),
            attachments: vec![MessageAttachment {
                local_path: artifact.to_string_lossy().into_owned(),
                filename: "report.pdf".to_string(),
                mime_type: "application/pdf".to_string(),
                size_bytes: std::fs::metadata(&artifact).unwrap().len(),
                provenance: AttachmentProvenance::ToolArtifact,
                source_tool: Some("browser".to_string()),
                ..MessageAttachment::default()
            }],
            ..Message::new_runtime("result-resource", session_id, "tool")
        };

        harness
            .agent
            .append_tool_message_with_result_event(
                &emitter,
                &message,
                true,
                5,
                None,
                Some("task-resource"),
            )
            .await
            .unwrap();

        let events = harness
            .agent
            .event_store
            .query_events_by_types(session_id, &[EventType::ResourceRegistered], 10)
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        let registered = events[0].parse_data::<ResourceRegisteredData>().unwrap();
        assert!(registered.resource_id.starts_with("res_"));
        assert_eq!(registered.locator, artifact.to_string_lossy());
        assert!(registered.sha256.is_some());
        assert_eq!(
            registered.produced_by_tool_call_id.as_deref(),
            Some("browser-call")
        );
        assert!(harness
            .agent
            .event_store
            .get_resource(session_id, &registered.resource_id)
            .await
            .unwrap()
            .is_some());

        let history = harness.state.get_history(session_id, 10).await.unwrap();
        assert!(history[0]
            .content
            .as_deref()
            .unwrap_or_default()
            .contains(&registered.resource_id));
    }
}
