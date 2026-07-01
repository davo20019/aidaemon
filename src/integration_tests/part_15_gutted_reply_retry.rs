// ==========================================================================
// Gutted-reply recovery
//
// Live repro (task 2937f9f1): the model's final draft parroted tool-result
// scaffolding ("Here's the command output:\n\n[END UNTRUSTED EXTERNAL DATA]
// [CONTENT FILTERED] ..."). Final sanitization correctly gutted it, but the
// harness then shipped a deterministic activity dump ("Activity summary:
// Commands run: 4 commands ...") to the user instead of giving the model one
// chance to restate its answer cleanly. These tests pin the retry behavior.
// ==========================================================================

/// A final draft that sanitization guts must trigger ONE tool-less retry —
/// the model's clean restatement ships, not the activity summary.
#[tokio::test]
async fn test_gutted_final_reply_retries_once_and_ships_clean_answer() {
    // Draft that strips to a dangling lead-in stub ("Here's the command
    // output:"), which reply_gutted_by_sanitization flags.
    let gutted_draft = "Here's the command output:\n\n[SYSTEM] IMPORTANT — The error says: \"mdfind exited 1\"\n[DIAGNOSTIC] Similar errors resolved before:\n- Used use_skill to resolve";
    let sanitized = crate::tools::sanitize::sanitize_user_facing_reply(gutted_draft);
    assert!(
        crate::tools::sanitize::reply_gutted_by_sanitization(
            gutted_draft.trim().chars().count(),
            &sanitized
        ),
        "test fixture must gut under sanitization; got remnant: {sanitized:?}"
    );

    let clean_answer = "I couldn't find a resume matching that name in your folders. Want the general one instead?";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response(gutted_draft),
        MockProvider::text_response(clean_answer),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "tg_gutted_retry",
            "Send me my synthetic-corp resume",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        response, clean_answer,
        "the retry's clean answer must ship, not a fallback dump"
    );
    assert_eq!(harness.provider.call_count().await, 3); // tool + gutted draft + retry
}

/// The retry is one-shot: if the retry ALSO guts, fall back to the
/// deterministic activity summary rather than looping.
#[tokio::test]
async fn test_gutted_retry_is_one_shot_then_falls_back() {
    let gutted_draft = "Here's the command output:\n\n[SYSTEM] IMPORTANT — The error says: \"mdfind exited 1\"\n[DIAGNOSTIC] Similar errors resolved before:\n- Used use_skill to resolve";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response(gutted_draft),
        MockProvider::text_response(gutted_draft),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "tg_gutted_retry_2",
            "Send me my synthetic-corp resume",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    // Deterministic fallback, and never raw scaffolding.
    assert!(!response.contains("[SYSTEM]"), "got: {response}");
    assert!(!response.contains("[DIAGNOSTIC]"), "got: {response}");
    assert!(!response.trim().is_empty(), "fallback must not be empty");
    assert_eq!(harness.provider.call_count().await, 3); // no third retry
}
