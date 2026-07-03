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

/// A final reply that is a verbatim read_file page (spilled-result paging
/// derailment) gets ONE retry with an answer-don't-paste directive; the
/// retry's real answer ships.
#[tokio::test]
async fn test_pasted_file_page_reply_retries_once_and_ships_answer() {
    // Real page shape: harness header + line-numbered content (live repro:
    // page 5 of a spilled clinical-trials JSON shipped as the "answer").
    let pasted_page = "Done. Here is the output:\n\nFile: /tmp/tool_results/http_request-abc.txt (lines 672-810 of 1066, 39192 bytes, modified 2026-07-02T15:44:38Z)\n672 |         },\n673 |         {\n674 |           \"city\": \"Springfield\",\n675 |           \"country\": \"Freedonia\",\n676 |           \"facility\": \"Synthetic Medical Center ( Site 0001)\"\n677 |         },\n678 |         {\n679 |           \"city\": \"Shelbyville\"\n680 |         }";
    let clean_answer =
        "Two trials are near Springfield: Synthetic Medical Center (Site 0001) and one more.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response(pasted_page),
        MockProvider::text_response(clean_answer),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "tg_paste_retry",
            "Which trials are near Springfield?",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, clean_answer, "the retry's answer must ship");
    assert_eq!(harness.provider.call_count().await, 3);
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

/// A final reply that promises imminent action while the task made ZERO tool
/// calls (live repro: "I'm searching the ClinicalTrials.gov API ... now..."
/// then task end, nothing running) gets ONE retry directing the model to do
/// the work or admit inability. The retry's real answer ships.
#[tokio::test]
async fn test_unbacked_action_promise_retries_once() {
    let promissory =
        "I'm searching the ClinicalTrials.gov API specifically for recruiting skin cancer trials in the Fairfax area now...";
    let clean_answer = "I couldn't reach ClinicalTrials.gov from here — want me to try the web search tool instead?";
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(promissory),
        MockProvider::text_response(clean_answer),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "tg_promise_retry",
            "Find recruiting skin cancer trials near Fairfax",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, clean_answer, "the retry's answer must ship");
    assert_eq!(harness.provider.call_count().await, 2);
}

/// The same promissory phrasing AFTER real tool work ships untouched — the
/// gate is scoped to the zero-tool fabrication class only.
#[tokio::test]
async fn test_action_promise_after_tool_work_ships() {
    let promissory_after_work = "Checking the remaining registries now...";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response(promissory_after_work),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "tg_promise_with_work",
            "Find recruiting skin cancer trials near Fairfax",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, promissory_after_work);
    assert_eq!(harness.provider.call_count().await, 2, "no retry");
}
