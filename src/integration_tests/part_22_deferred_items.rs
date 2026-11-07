// ==================== Deferred checklist items keep the request open ====================
//
// A reply that says "deferred" is not a completed request. The executor's own
// checklist decides whether the work happened; deferring every item must leave
// the owner's request partially answered so a follow-up can resume it.

#[tokio::test]
async fn deferring_every_checklist_item_leaves_the_request_partially_answered() {
    let session_id = "deferred_items_request";
    let declare = json!({
        "items": [
            {"text": "Remove the three entries", "status": "pending", "mutation_effects": ["destructive"]},
            {"text": "Verify they are gone", "status": "pending", "requires_observation": true, "depends_on": [0]}
        ]
    })
    .to_string();
    let defer = json!({
        "items": [
            {"text": "Remove the three entries", "status": "deferred", "mutation_effects": ["destructive"], "note": "blocked"},
            {"text": "Verify they are gone", "status": "deferred", "requires_observation": true, "depends_on": [0], "note": "not run"}
        ]
    })
    .to_string();
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("track_requirements", &declare),
        MockProvider::tool_call_response("track_requirements", &defer),
        MockProvider::text_response("I deferred both steps; the removal was blocked."),
    ])
    .with_task_assessments(vec![MockProvider::semantic_task_assessment(
        "answer",
        true,
        false,
        &[],
        "new_request",
        "none",
    )]);
    let harness = setup_test_agent(provider).await.unwrap();

    let _ = harness
        .agent
        .handle_message(
            session_id,
            "Remove the three incorrect entries and verify they are gone",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let dialogue = harness
        .state
        .get_dialogue_state(session_id)
        .await
        .unwrap()
        .expect("dialogue state should be persisted");
    // Deferred work leaves the request owed: `blocked` or `partially_answered`
    // are both open states. Only `answered` would close it.
    let status = dialogue.open_request.as_ref().map(|request| request.status);
    assert!(
        matches!(
            status,
            Some(crate::traits::OpenRequestStatus::Blocked)
                | Some(crate::traits::OpenRequestStatus::PartiallyAnswered)
        ),
        "deferred work must not close the request: {status:?}"
    );
    let owed = dialogue
        .unfinished_requests
        .iter()
        .chain(dialogue.open_request.iter())
        .count();
    assert!(owed >= 1, "the owed request must stay in the lineage");
}
