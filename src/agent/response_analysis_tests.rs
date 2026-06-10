use super::*;

#[test]
fn reply_defers_file_access_detects_upload_requests() {
    for reply in [
        "I don't have access to that specific PDF file yet. Could you please upload the file or provide the full path to it on your system?",
        "Please provide the path to the file and I'll take a look.",
        "Could you attach the file so I can read it?",
        "I don't have direct access to that file just by its name.",
    ] {
        assert!(
            reply_defers_file_access(reply),
            "should detect deferred file access: {reply:?}"
        );
    }
}

#[test]
fn reply_defers_file_access_ignores_real_answers() {
    for reply in [
        "The offer letter is from WebFirst for a Lead Developer position starting in July.",
        "I found the file and read it — here's the summary.",
        "Done. The script now parses the path argument correctly.",
    ] {
        assert!(
            !reply_defers_file_access(reply),
            "false positive on: {reply:?}"
        );
    }
}

#[test]
fn user_text_references_file_detects_filenames_and_paths() {
    for text in [
        "Can you read the file and tell me what it's about? David Loor  WebFirst Offer Letter Lead Developer (1).pdf",
        "summarize ~/Downloads/report.docx",
        "what's in /tmp/output.log",
        "check notes.md please",
    ] {
        assert!(
            user_text_references_file(text),
            "should detect file reference: {text:?}"
        );
    }
}

#[test]
fn user_text_references_file_ignores_plain_chat() {
    for text in [
        "write me a poem about rust",
        "what's my cat's name?",
        "how do I improve my resume for tech jobs",
    ] {
        assert!(
            !user_text_references_file(text),
            "false positive on: {text:?}"
        );
    }
}

#[test]
fn test_extract_intent_gate_single_line_json() {
    let input = "Answer first.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[\"deployment_url\"]}";
    let (cleaned, gate) = extract_intent_gate(input);
    assert_eq!(cleaned, "Answer first.");
    let gate = gate.expect("expected parsed intent gate");
    assert_eq!(gate.can_answer_now, Some(false));
    assert_eq!(gate.needs_tools, Some(true));
    assert_eq!(gate.needs_clarification, Some(false));
    assert_eq!(gate.missing_info, vec!["deployment_url".to_string()]);
}

#[test]
fn test_extract_intent_gate_two_line_json() {
    let input = "Answer first.\n[INTENT_GATE]\n{\"can_answer_now\":true,\"needs_tools\":false,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[]}";
    let (cleaned, gate) = extract_intent_gate(input);
    assert_eq!(cleaned, "Answer first.");
    let gate = gate.expect("expected parsed intent gate");
    assert_eq!(gate.can_answer_now, Some(true));
    assert_eq!(gate.needs_tools, Some(false));
}

#[test]
fn test_extract_intent_gate_trailing_json_braces_in_strings() {
    // Fallback path: no [INTENT_GATE] marker, trailing JSON in a code fence.
    // The JSON contains a '{' inside a string, which breaks naive brace-counting.
    let input = "Answer here.\n\n```json\n{\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":true,\"clarifying_question\":\"contains { brace\",\"missing_info\":[\"deployment_url\"],\"complexity\":\"simple\"}\n```";
    let (cleaned, gate) = extract_intent_gate(input);
    assert_eq!(cleaned, "Answer here.");
    let gate = gate.expect("expected parsed intent gate");
    assert_eq!(gate.can_answer_now, Some(false));
    assert_eq!(gate.needs_tools, Some(true));
    assert_eq!(gate.needs_clarification, Some(true));
    assert_eq!(gate.missing_info, vec!["deployment_url".to_string()]);
}

#[test]
fn test_infer_intent_gate_no_textual_fallback_inference() {
    // With lexical fallback inference disabled, missing model fields remain None.
    let gate = infer_intent_gate("check the site", "I can look it up.");
    assert_eq!(gate.can_answer_now, None);
    assert_eq!(gate.needs_tools, None);
    assert_eq!(gate.needs_clarification, None);
}

#[test]
fn test_infer_intent_gate_path_still_forces_tools() {
    // Deterministic fallback: filesystem paths always require tools.
    let gate = infer_intent_gate("check /tmp/app.log", "I can look it up.");
    assert_eq!(gate.can_answer_now, Some(false));
    assert_eq!(gate.needs_tools, Some(true));
    assert_eq!(gate.needs_clarification, Some(false));
}

#[test]
fn test_user_text_references_filesystem_path_ignores_fractions_and_shorthand() {
    assert!(!user_text_references_filesystem_path("3/4"));
    assert!(!user_text_references_filesystem_path("2/14"));
    assert!(!user_text_references_filesystem_path("yes/no"));
    assert!(!user_text_references_filesystem_path("w/o"));
}

#[test]
fn test_user_text_references_filesystem_path_detects_common_paths_and_files() {
    assert!(user_text_references_filesystem_path(
        "/Users/alice/project/file.txt"
    ));
    assert!(user_text_references_filesystem_path("~/project/file.txt"));
    assert!(user_text_references_filesystem_path(
        "src/agent/main_loop.rs"
    ));
    assert!(user_text_references_filesystem_path("Cargo.toml"));
    assert!(user_text_references_filesystem_path(
        r"C:\\Users\\alice\\file.txt"
    ));
}

#[test]
fn test_user_explicitly_requests_local_file_inspection_detects_explicit_requests() {
    assert!(user_explicitly_requests_local_file_inspection(
        "Inspect Cargo.toml and read src/main.rs"
    ));
    assert!(user_explicitly_requests_local_file_inspection(
        "Search the repo for OAuth callback code"
    ));
}

#[test]
fn test_user_explicitly_requests_local_file_inspection_does_not_flag_api_only_turns() {
    assert!(!user_explicitly_requests_local_file_inspection(
        "Use the Twitter API to post a tweet"
    ));
    assert!(!user_explicitly_requests_local_file_inspection(
        "Check the connected API status"
    ));
}

#[test]
fn test_infer_intent_gate_does_not_guess_clarification_from_text() {
    let gate = infer_intent_gate("update the site", "Could you clarify which site you mean?");
    assert_eq!(gate.needs_clarification, None);
}

#[test]
fn test_infer_intent_gate_does_not_infer_schedule_from_user_text() {
    let gate = infer_intent_gate("send me a reminder in 2h", "Let me do that.");
    assert!(gate.schedule.is_none());
    assert!(gate.schedule_type.is_none());
}

#[test]
fn test_sanitize_response_analysis_strips_marker_and_pseudo_tool_block() {
    let input = "I recall it was deployed to Cloudflare Workers.\n\n\
                 [TEXT_ONLY_RESPONSE_MODE]\n\
                 [tool_use: terminal]\n\
                 cmd: find $HOME -name wrangler.toml\n\
                 args: {\"x\":1}";
    let out = sanitize_response_analysis(input);
    assert!(out.contains("I recall it was deployed to Cloudflare Workers."));
    assert!(!out.contains("TEXT_ONLY_RESPONSE_MODE"));
    assert!(!out.contains("[tool_use:"));
    assert!(!out.contains("cmd:"));
    assert!(!out.contains("args:"));
}

#[test]
fn test_sanitize_response_analysis_keeps_normal_cmd_text_without_tool_block() {
    let input = "Run this command manually:\ncmd: wrangler whoami";
    let out = sanitize_response_analysis(input);
    assert!(out.contains("cmd: wrangler whoami"));
}

#[test]
fn test_sanitize_response_analysis_strips_arguments_name_terminal_block() {
    let input = "I'll check config.\n\narguments:\nname: terminal";
    let out = sanitize_response_analysis(input);
    assert_eq!(out, "I'll check config.");
}

#[test]
fn test_sanitize_response_analysis_strips_echoed_important_instruction() {
    let input = "I don't have the exact URL yet.\n\n\
        [IMPORTANT: You are being consulted for your knowledge and reasoning. Respond with TEXT ONLY. Do NOT call any functions or tools. Do NOT output functionCall or tool_use blocks. Answer the user's question directly from your knowledge and the context provided.]";
    let out = sanitize_response_analysis(input);
    assert_eq!(out, "I don't have the exact URL yet.");
}

#[test]
fn test_looks_like_deferred_action_response_detects_planning_text() {
    // Action promises — any verb after "I'll" / "Let me" / "I will" that isn't knowledge-only
    assert!(looks_like_deferred_action_response(
        "I'll check the configuration for the Cloudflare Worker."
    ));
    assert!(looks_like_deferred_action_response(
        "Let me search and get back to you."
    ));
    assert!(looks_like_deferred_action_response(
        "I'll create a Python script to check the status."
    ));
    assert!(looks_like_deferred_action_response(
        "I'll run the tests and report back."
    ));
    assert!(looks_like_deferred_action_response(
        "Let me write a script for that."
    ));
    assert!(looks_like_deferred_action_response(
        "I will deploy the changes now."
    ));
    assert!(looks_like_deferred_action_response(
        "I'll need to check the full content of the audit report."
    ));
    assert!(looks_like_deferred_action_response(
        "I'll retrieve the complete text now."
    ));
    assert!(looks_like_deferred_action_response(
        "Let me read the file and send it to you."
    ));
    assert!(looks_like_deferred_action_response(
        "Shall I scan your projects folder?"
    ));
    assert!(looks_like_deferred_action_response(
        "Would you like me to install the dependencies?"
    ));
    assert!(looks_like_deferred_action_response(
        "I'll find your resume and send it over right away. Starting the send-resume workflow."
    ));
    // Structural markers
    assert!(looks_like_deferred_action_response(
        "I recall deploying to Workers.\n\n[Consultation]\nTo find the URL, I would typically inspect wrangler.toml."
    ));

    // Knowledge-only verbs — these DON'T need tools
    assert!(!looks_like_deferred_action_response(
        "I'll explain how it works."
    ));
    assert!(!looks_like_deferred_action_response(
        "Let me describe the architecture."
    ));
    assert!(!looks_like_deferred_action_response(
        "I will summarize the key points for you."
    ));
    assert!(!looks_like_deferred_action_response(
        "I'll clarify what that means."
    ));

    // Not action promises at all
    assert!(!looks_like_deferred_action_response(
        "The URL is https://example.workers.dev"
    ));
    assert!(!looks_like_deferred_action_response(
        "I checked the configuration already and it looks fine."
    ));
    assert!(!looks_like_deferred_action_response(
        "The searching process was completed successfully."
    ));
}

#[test]
fn test_has_action_promise() {
    // Action verbs
    assert!(has_action_promise("i'll create a script"));
    assert!(has_action_promise("i will run the tests"));
    assert!(has_action_promise("let me check the file"));
    assert!(has_action_promise("i’ll find your resume and send it"));
    assert!(has_action_promise("shall i scan the folder"));
    assert!(has_action_promise("would you like me to install it"));

    // Knowledge verbs — not action promises
    assert!(!has_action_promise("i'll explain the concept"));
    assert!(!has_action_promise("let me describe it"));
    assert!(!has_action_promise("i will summarize the results"));
    assert!(!has_action_promise("i'll clarify that for you"));
    assert!(!has_action_promise("i'll provide an overview"));
    assert!(!has_action_promise("i'll be happy to help"));

    // No prefix pattern at all
    assert!(!has_action_promise("the file is located at /tmp/test"));
    assert!(!has_action_promise("here is the answer"));
}

#[test]
fn test_is_short_user_correction_detects_simple_correction() {
    assert!(is_short_user_correction("You did send me the pdf"));
    assert!(is_short_user_correction("that's right"));
}

#[test]
fn test_is_short_user_correction_ignores_new_action_requests() {
    assert!(!is_short_user_correction(
        "You did send me the pdf, can you make it nicer?"
    ));
    assert!(!is_short_user_correction("Please regenerate the PDF"));
}

#[test]
fn test_classify_stall_detects_deferred_no_tool_loop() {
    let learning_ctx = LearningContext {
        user_text: "Can you make the PDF nicer?".to_string(),
        intent_domains: vec![],
        tool_calls: vec![],
        errors: vec![(DEFERRED_NO_TOOL_ERROR_MARKER.to_string(), false)],
        first_error: None,
        recovery_actions: vec![],
        start_time: Utc::now(),
        completed_naturally: false,
        explicit_positive_signals: 0,
        explicit_negative_signals: 0,
        task_outcome: None,
        replay_notes: Vec::new(),
    };

    let (label, suggestion) = Agent::classify_stall(&learning_ctx);
    assert_eq!(label, "Deferred No-Tool Loop");
    assert!(suggestion.contains("rephrasing"));
}

#[test]
fn test_parse_wait_task_seconds_parses_supported_units() {
    assert_eq!(parse_wait_task_seconds("Wait for 5 minutes."), Some(300));
    assert_eq!(parse_wait_task_seconds("wait for 45 sec"), Some(45));
    assert_eq!(parse_wait_task_seconds("WAIT FOR 2 hours"), Some(7200));
}

#[test]
fn test_parse_wait_task_seconds_ignores_non_wait_tasks() {
    assert_eq!(parse_wait_task_seconds("Send the second joke."), None);
    assert_eq!(parse_wait_task_seconds("Wait until tomorrow."), None);
}

#[test]
fn test_sanitize_response_analysis_strips_consultation_heading() {
    let input =
        "I don't have the URL yet.\n\n[Consultation]\nTo find it I'd inspect wrangler.toml.";
    let out = sanitize_response_analysis(input);
    assert!(!out.contains("[Consultation]"));
}

#[test]
fn test_extract_intent_gate_bare_json_without_marker() {
    let input = "The capital of France is Paris.\n{\"complexity\":\"knowledge\"}";
    let (cleaned, gate) = extract_intent_gate(input);
    assert_eq!(cleaned, "The capital of France is Paris.");
    let gate = gate.expect("expected parsed intent gate from bare JSON");
    assert_eq!(gate.complexity.as_deref(), Some("knowledge"));
}

#[test]
fn test_extract_intent_gate_code_fenced_json() {
    let input = "The capital of France is Paris.\n```json\n{\"complexity\":\"knowledge\"}\n```";
    let (cleaned, gate) = extract_intent_gate(input);
    assert_eq!(cleaned, "The capital of France is Paris.");
    let gate = gate.expect("expected parsed intent gate from fenced JSON");
    assert_eq!(gate.complexity.as_deref(), Some("knowledge"));
}

#[test]
fn test_extract_intent_gate_bare_json_with_spaces() {
    let input = "Answer here.\n\n{ \"complexity\": \"simple\", \"can_answer_now\": false, \"needs_tools\": true }";
    let (cleaned, gate) = extract_intent_gate(input);
    assert!(!cleaned.contains("complexity"));
    let gate = gate.expect("expected parsed intent gate");
    assert_eq!(gate.complexity.as_deref(), Some("simple"));
    assert_eq!(gate.can_answer_now, Some(false));
}

#[test]
fn test_extract_intent_gate_multiline_bare_json() {
    let input = "The largest planet is Jupiter.\n\n{\n  \"complexity\": \"knowledge\"\n}";
    let (cleaned, gate) = extract_intent_gate(input);
    assert_eq!(cleaned, "The largest planet is Jupiter.");
    let gate = gate.expect("expected parsed intent gate from multi-line JSON");
    assert_eq!(gate.complexity.as_deref(), Some("knowledge"));
}

#[test]
fn test_extract_intent_gate_bare_json_does_not_strip_unrelated_json() {
    // JSON that doesn't contain intent gate fields should NOT be stripped
    let input = "Here is the data:\n{\"name\":\"Alice\",\"age\":30}";
    let (cleaned, gate) = extract_intent_gate(input);
    assert!(gate.is_none());
    assert!(cleaned.contains("{\"name\":\"Alice\""));
}

#[test]
fn test_is_substantive_text_response_accepts_real_content() {
    // A greeting with enough substance should be accepted
    assert!(is_substantive_text_response(
        "Hola! Claro que sí, puedo hablar en español. ¿En qué puedo ayudarte hoy?",
        50
    ));

    // A capability listing should be accepted
    assert!(is_substantive_text_response(
        "Here are my main capabilities:\n\
         1. I can run terminal commands\n\
         2. I can search the web\n\
         3. I can read and write files\n\
         4. I can manage your schedule",
        50
    ));

    // A joke should be accepted
    assert!(is_substantive_text_response(
        "Here's a joke for you: Why do programmers prefer dark mode? Because light attracts bugs!",
        50
    ));
}

#[test]
fn test_is_substantive_text_response_rejects_short_text() {
    // Too short to be substantive
    assert!(!is_substantive_text_response("Sure!", 50));
    assert!(!is_substantive_text_response("OK, I can do that.", 50));
}

#[test]
fn test_is_substantive_text_response_rejects_pure_deferrals() {
    // Pure deferral text: every line is an action promise
    assert!(!is_substantive_text_response(
        "I'll search the web for that information and get back to you right away.",
        50
    ));

    // Short deferral
    assert!(!is_substantive_text_response(
        "Let me check that for you.",
        50
    ));
}

#[test]
fn test_is_substantive_text_response_accepts_mixed_content() {
    // Some deferred-looking phrasing mixed with real content — the
    // substantive parts exceed the min_len threshold.
    assert!(is_substantive_text_response(
        "I'll help you with that.\n\n\
         The capital of France is Paris. It is the largest city in France \
         and serves as the country's political, economic, and cultural center.",
        50
    ));
}

#[test]
fn test_claims_completed_side_effect_detects_past_tense_action_claims() {
    // The exact fabrication from the 2026-06-06 attribution run, turn 10:
    // zero tool calls were made, yet the model claimed the deletion happened.
    assert!(claims_completed_side_effect(
        "I have deleted the cachetest2 folder entirely."
    ));
    assert!(claims_completed_side_effect(
        "I've removed west.txt and cleaned up the directory."
    ));
    assert!(claims_completed_side_effect("I deleted the folder."));
    assert!(claims_completed_side_effect(
        "Done — I created the directory and wrote all four files for you."
    ));
    assert!(claims_completed_side_effect(
        "I have successfully executed the script."
    ));
    // Unicode apostrophe, same as has_action_promise handling.
    assert!(claims_completed_side_effect(
        "I\u{2019}ve updated north.txt."
    ));
}

#[test]
fn test_claims_completed_side_effect_ignores_non_claims() {
    // Future-tense promises are deferred actions, not completion claims.
    assert!(!claims_completed_side_effect("I'll delete the folder."));
    assert!(!claims_completed_side_effect(
        "Let me remove that file for you."
    ));
    // Plain informational replies.
    assert!(!claims_completed_side_effect(
        "Here are the contents of the three files."
    ));
    assert!(!claims_completed_side_effect(
        "The folder was already empty, nothing to do."
    ));
    // "I have" followed by a non-action word is not a claim.
    assert!(!claims_completed_side_effect(
        "I have a summary of the files for you."
    ));
    assert!(!claims_completed_side_effect(
        "I have to check the file before answering."
    ));
    // Knowledge verbs in past tense are not side effects.
    assert!(!claims_completed_side_effect(
        "I have explained the differences between the files above."
    ));
}

#[test]
fn test_claims_delegation_started_detects_unverified_agent_claims() {
    assert!(claims_delegation_started(
        "I've initiated a deep analysis using a specialized review agent."
    ));
    assert!(claims_delegation_started(
        "I started a research sub-agent to investigate this."
    ));
    assert!(claims_delegation_started(
        "A specialist agent is now running in the background."
    ));
    assert!(!claims_delegation_started(
        "I can analyze the resume directly."
    ));
    assert!(!claims_delegation_started(
        "The review agent pattern is useful for complex tasks."
    ));
}
