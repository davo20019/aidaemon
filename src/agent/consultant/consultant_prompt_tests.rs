use super::*;

#[test]
fn test_strip_markdown_section_removes_target_heading() {
    let prompt =
        "## Identity\nKeep this\n## Tools\nDrop this\nline2\n## Built-in Channels\nKeep channels";
    let stripped = strip_markdown_section(prompt, "## Tools");
    assert!(stripped.contains("## Identity"));
    assert!(stripped.contains("## Built-in Channels"));
    assert!(!stripped.contains("Drop this"));
    assert!(!stripped.contains("line2"));
}

#[test]
fn test_build_consultant_system_prompt_adds_marker_and_strips_tools() {
    let prompt = "## Identity\nA\n## Tool Selection Guide\nB\n## Tools\nC\n## Behavior\nD";
    let consultant = build_consultant_system_prompt(prompt, ConsultantPromptStyle::Full);
    assert!(consultant.contains(CONSULTANT_TEXT_ONLY_MARKER));
    assert!(consultant.contains("## Identity"));
    assert!(consultant.contains("## Behavior"));
    assert!(!consultant.contains("## Tool Selection Guide"));
    assert!(!consultant.contains("## Tools"));
}

#[test]
fn test_build_tool_loop_system_prompt_strips_heavy_sections() {
    let prompt = "## Identity\nA\n## Tool Selection Guide\nB\n## Tools\nC\n## Behavior\nD";

    let standard = build_tool_loop_system_prompt(prompt, ToolLoopPromptStyle::Standard);
    assert!(standard.contains("## Identity"));
    assert!(standard.contains("## Tool Selection Guide"));
    assert!(standard.contains("## Behavior"));
    assert!(!standard.contains("## Tools"));

    let lite = build_tool_loop_system_prompt(prompt, ToolLoopPromptStyle::Lite);
    assert!(lite.contains("## Identity"));
    assert!(lite.contains("## Behavior"));
    assert!(!lite.contains("## Tool Selection Guide"));
    assert!(!lite.contains("## Tools"));
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
fn test_sanitize_consultant_analysis_strips_marker_and_pseudo_tool_block() {
    let input = "I recall it was deployed to Cloudflare Workers.\n\n\
                 [CONSULTANT_TEXT_ONLY_MODE]\n\
                 [tool_use: terminal]\n\
                 cmd: find $HOME -name wrangler.toml\n\
                 args: {\"x\":1}";
    let out = sanitize_consultant_analysis(input);
    assert!(out.contains("I recall it was deployed to Cloudflare Workers."));
    assert!(!out.contains("CONSULTANT_TEXT_ONLY_MODE"));
    assert!(!out.contains("[tool_use:"));
    assert!(!out.contains("cmd:"));
    assert!(!out.contains("args:"));
}

#[test]
fn test_sanitize_consultant_analysis_keeps_normal_cmd_text_without_tool_block() {
    let input = "Run this command manually:\ncmd: wrangler whoami";
    let out = sanitize_consultant_analysis(input);
    assert!(out.contains("cmd: wrangler whoami"));
}

#[test]
fn test_sanitize_consultant_analysis_strips_arguments_name_terminal_block() {
    let input = "I'll check config.\n\narguments:\nname: terminal";
    let out = sanitize_consultant_analysis(input);
    assert_eq!(out, "I'll check config.");
}

#[test]
fn test_sanitize_consultant_analysis_strips_echoed_important_instruction() {
    let input = "I don't have the exact URL yet.\n\n\
        [IMPORTANT: You are being consulted for your knowledge and reasoning. Respond with TEXT ONLY. Do NOT call any functions or tools. Do NOT output functionCall or tool_use blocks. Answer the user's question directly from your knowledge and the context provided.]";
    let out = sanitize_consultant_analysis(input);
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
fn test_sanitize_consultant_analysis_strips_consultation_heading() {
    let input =
        "I don't have the URL yet.\n\n[Consultation]\nTo find it I'd inspect wrangler.toml.";
    let out = sanitize_consultant_analysis(input);
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
