// ==================== A conversation answers with the model it is set to ====================
//
// A client's picker pins a model and an effort per conversation (protocol
// 1.16). The root turn calls that model at that effort; another
// conversation keeps the daemon's own choice.

#[tokio::test]
async fn a_pinned_conversation_answers_with_its_model_and_effort() {
    use aidaemon_runtime::session_model::{self, PinnedModel, SessionModelSetting};
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("Pinned answer."),
        MockProvider::text_response("Automatic answer."),
    ]);
    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();
    session_model::set(
        harness.state.as_ref(),
        "app:pinned",
        &SessionModelSetting {
            pinned: Some(PinnedModel {
                provider: session_model::MAIN_PROVIDER.into(),
                model: "smart-model".into(),
            }),
            effort: Some("high".into()),
            agent: None,
        },
    )
    .await
    .unwrap();

    for session in ["app:pinned", "app:automatic"] {
        harness
            .agent
            .handle_message(
                session,
                "Hello!",
                None,
                UserRole::Owner,
                ChannelContext::private("app"),
                None,
            )
            .await
            .unwrap();
    }

    let calls = harness.provider.call_log.lock().await.clone();
    assert_eq!(calls.len(), 2, "{calls:?}");
    assert_eq!(calls[0].model, "smart-model");
    assert_eq!(calls[0].options.reasoning_effort_override.as_deref(), Some("high"));
    assert_eq!(calls[1].model, "primary-model");
    assert_eq!(calls[1].options.reasoning_effort_override, None);
}

/// A pin to a model the runtime no longer has reads as automatic: a config
/// change never strands a conversation.
#[tokio::test]
async fn a_pin_the_runtime_lost_answers_automatically() {
    use aidaemon_runtime::session_model::{self, PinnedModel, SessionModelSetting};
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("Answer.")]);
    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();
    session_model::set(
        harness.state.as_ref(),
        "app:stale",
        &SessionModelSetting {
            pinned: Some(PinnedModel {
                provider: session_model::MAIN_PROVIDER.into(),
                model: "removed-model".into(),
            }),
            effort: None,
            agent: None,
        },
    )
    .await
    .unwrap();
    harness
        .agent
        .handle_message(
            "app:stale",
            "Hello!",
            None,
            UserRole::Owner,
            ChannelContext::private("app"),
            None,
        )
        .await
        .unwrap();
    let calls = harness.provider.call_log.lock().await.clone();
    assert_eq!(calls[0].model, "primary-model");
}

/// A conversation's pin wins over a daemon-wide `/model`; a conversation on
/// automatic follows `/model`.
#[tokio::test]
async fn a_pin_beats_a_daemon_wide_model() {
    use aidaemon_runtime::session_model::{self, PinnedModel, SessionModelSetting};
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("Pinned."),
        MockProvider::text_response("Global."),
    ]);
    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();
    harness.agent.engine().set_model("smart-model".to_string()).await;
    session_model::set(
        harness.state.as_ref(),
        "app:pinned",
        &SessionModelSetting {
            pinned: Some(PinnedModel {
                provider: session_model::MAIN_PROVIDER.into(),
                model: "primary-model".into(),
            }),
            effort: None,
            agent: None,
        },
    )
    .await
    .unwrap();
    for session in ["app:pinned", "app:automatic"] {
        harness
            .agent
            .handle_message(
                session,
                "Hello!",
                None,
                UserRole::Owner,
                ChannelContext::private("app"),
                None,
            )
            .await
            .unwrap();
    }
    let calls = harness.provider.call_log.lock().await.clone();
    assert_eq!(calls[0].model, "primary-model");
    assert_eq!(calls[1].model, "smart-model");
}

/// A conversation set to a coding agent hands each message, as it is, to
/// that agent through the usual `cli_agent` run, and no model is called:
/// not for the turn, and not to plan it first.
#[tokio::test]
async fn a_coding_agent_conversation_takes_the_message_with_no_model_call() {
    use aidaemon_runtime::session_model::{self, SessionModelSetting};
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "A model answered.",
    )]);
    let extra_tools = vec![
        Arc::new(MockTool::new("cli_agent", "delegation tool", "Codex finished the change."))
            as Arc<dyn crate::traits::Tool>,
    ];
    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(provider, extra_tools, None)
        .await
        .unwrap();
    session_model::set(
        harness.state.as_ref(),
        "app:direct",
        &SessionModelSetting {
            agent: Some("codex".into()),
            ..SessionModelSetting::default()
        },
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "app:direct",
            "Rename the contact form's submit button to Send",
            None,
            UserRole::Owner,
            ChannelContext::private("app"),
            None,
        )
        .await
        .unwrap();

    let calls = harness.provider.call_log.lock().await.clone();
    assert!(calls.is_empty(), "no model call expected: {calls:?}");
    // The reply is the run's own text, not a summary in its place.
    assert_eq!(reply, "Codex finished the change.");
    let history = harness.state.get_history("app:direct", 20).await.unwrap();
    let launched = history.iter().any(|message| {
        message.tool_calls_json.as_deref().is_some_and(|calls| {
            let calls: serde_json::Value = serde_json::from_str(calls).unwrap_or_default();
            calls.as_array().is_some_and(|calls| {
                calls.iter().any(|call| {
                    let args = call
                        .get("arguments")
                        .and_then(|a| a.as_str().map(|s| serde_json::from_str(s).unwrap_or_default()).or_else(|| Some(a.clone())))
                        .unwrap_or_default();
                    call.get("name").and_then(|n| n.as_str()) == Some("cli_agent")
                        && args["tool"] == "codex"
                        && args["prompt"] == "Rename the contact form's submit button to Send"
                })
            })
        })
    });
    assert!(launched, "the message should go to codex as it is: {history:?}");
}

/// A run that could not start is reported as the run said, never as a
/// generic "I completed 1 action".
#[tokio::test]
async fn a_coding_agent_conversation_reports_a_refused_run_as_it_is() {
    use aidaemon_runtime::session_model::{self, SessionModelSetting};
    let provider = MockProvider::with_responses(vec![]);
    let refusal = "BLOCKED: another run is working in this folder.";
    let extra_tools = vec![
        Arc::new(MockTool::new("cli_agent", "delegation tool", refusal))
            as Arc<dyn crate::traits::Tool>,
    ];
    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(provider, extra_tools, None)
        .await
        .unwrap();
    session_model::set(
        harness.state.as_ref(),
        "app:direct-refused",
        &SessionModelSetting {
            agent: Some("codex".into()),
            ..SessionModelSetting::default()
        },
    )
    .await
    .unwrap();
    let reply = harness
        .agent
        .handle_message(
            "app:direct-refused",
            "Add a footer",
            None,
            UserRole::Owner,
            ChannelContext::private("app"),
            None,
        )
        .await
        .unwrap();
    assert!(harness.provider.call_log.lock().await.is_empty());
    assert!(reply.contains("another run is working"), "{reply}");
    assert!(!reply.contains("completed 1 action"), "{reply}");
}

/// An agent's result shown with no turn written around it (a conversation
/// set to that agent) stays in history, so the chat shows it again after a
/// reload, and the untrusted output joins a turn recorded tainted.
#[tokio::test]
async fn a_coding_agent_result_delivered_as_it_is_stays_in_history() {
    use aidaemon_runtime::spawn::ConversationRuntime;
    let provider = MockProvider::with_responses(vec![]);
    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(provider, Vec::new(), None)
        .await
        .unwrap();

    harness
        .agent
        .record_delivered_result(
            "app:direct",
            "The submit button now reads Send.",
            aidaemon_core::SourceTrust::Untrusted,
        )
        .await
        .unwrap();

    let history = harness.state.get_history("app:direct", 20).await.unwrap();
    let note = history
        .iter()
        .find(|message| message.role == "assistant")
        .expect("the result is kept");
    let content = note.content.as_deref().unwrap_or_default();
    assert_eq!(
        aidaemon_runtime::turn::notes::parent_note_shown_text(content),
        Some("The submit button now reads Send.")
    );
    let tainted = crate::agent::untrusted::tainted_turns(harness.state.as_ref(), "app:direct")
        .await
        .unwrap();
    assert!(tainted.iter().any(|turn| Some(turn) == note.turn_id.as_ref()));
}
