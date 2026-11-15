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
