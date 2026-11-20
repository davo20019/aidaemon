// ==================== Reasoning summaries in llm_call events ====================

/// A turn's `llm_call` event keeps the model's reasoning summary, with any
/// credential in it redacted, so a run can be read after the fact.
#[tokio::test]
async fn an_llm_call_event_keeps_the_reasoning_summary_redacted() {
    let mut reply = MockProvider::text_response("The members are the employee IDs.");
    reply.thinking = Some(
        "The schema wants member_ids as written. token=github_pat_abcdefghijklmnopqrstuvwxyz0123"
            .to_string(),
    );
    let provider = MockProvider::with_responses(vec![reply]);
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, vec![])
        .await
        .unwrap();
    let session_id = "llm_call_reasoning";
    harness
        .agent
        .handle_message(
            session_id,
            "Which values go in members?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    let rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'llm_call' ORDER BY id",
    )
    .bind(session_id)
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();
    let kept: Vec<String> = rows
        .iter()
        .filter_map(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .filter_map(|data| data["reasoning"].as_str().map(str::to_string))
        .collect();
    assert_eq!(kept.len(), 1, "{rows:?}");
    assert!(kept[0].starts_with("The schema wants member_ids as written."), "{kept:?}");
    assert!(!kept[0].contains("github_pat_"), "{kept:?}");
}
