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

// ==========================================================================
// Tool-boundary truncation notice rendering (single loop site)
//
// Tools attach `TruncationInfo` as metadata with no embedded notice text.
// The loop must render the instructional notice into the model-visible tool
// message AFTER the outcome ledger (error_summary, record_outcome, failure
// classification) has already consumed the clean content.
// ==========================================================================

/// A tool whose output was truncated (metadata set, no embedded notice)
/// must (a) reach the model WITH the rendered notice and (b) leave the
/// outcome ledger's error_summary free of notice text.
#[tokio::test]
async fn test_truncated_tool_output_renders_notice_but_keeps_ledger_clean() {
    let truncated_tool = Arc::new(
        MockTool::new(
            "big_probe",
            "returns truncated output",
            "partial output\nError: disk full",
        )
        .with_metadata(crate::traits::ToolCallMetadata {
            truncation: Some(crate::traits::TruncationInfo {
                shown_chars: 30,
                total_chars: 900,
                remediation_hint: None,
            }),
            ..Default::default()
        }),
    ) as Arc<dyn crate::traits::Tool>;

    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("big_probe", "{}"),
        MockProvider::text_response("The probe failed: disk full."),
    ]);
    let harness =
        setup_test_agent_with_extra_tools_and_llm_timeout(provider, vec![truncated_tool], None)
            .await
            .unwrap();

    let _ = harness
        .agent
        .handle_message(
            "tg_trunc_meta",
            "Run the big probe",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    // (a) The model saw the rendered notice on the tool message of the
    // SECOND llm call.
    let calls = harness.provider.call_log.lock().await;
    let second = &calls.last().expect("two calls").messages;
    let tool_msg = second
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
        .expect("tool message present");
    let content = tool_msg
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    assert!(
        content.contains("OUTPUT TRUNCATED"),
        "model must see notice"
    );
    assert!(content.contains("Error: disk full"));
}
