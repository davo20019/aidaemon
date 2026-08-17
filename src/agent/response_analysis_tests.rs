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
fn test_infer_intent_gate_no_textual_fallback_inference() {
    // With lexical fallback inference disabled, an unarmed turn stays None.
    let gate = infer_intent_gate("check the site", "I can look it up.");
    assert_eq!(gate.needs_tools, None);
}

#[test]
fn test_infer_intent_gate_path_still_forces_tools() {
    // Deterministic rule: filesystem paths always require tools.
    let gate = infer_intent_gate("check /tmp/app.log", "I can look it up.");
    assert_eq!(gate.needs_tools, Some(true));
}

#[test]
fn test_user_text_references_filesystem_path_ignores_fractions_and_shorthand() {
    for prose in [
        "3/4",
        "2/14",
        "yes/no",
        "w/o",
        "Pros/cons",
        "input/output",
        "client/server",
        "read/write",
        "pass/fail",
    ] {
        assert!(
            !user_text_references_filesystem_path(prose),
            "unanchored prose compound became a filesystem reference: {prose}"
        );
    }
}

#[test]
fn test_user_text_references_filesystem_path_detects_common_paths_and_files() {
    assert!(user_text_references_filesystem_path(
        "/Users/alice/project/file.txt"
    ));
    assert!(user_text_references_filesystem_path("~/project/file.txt"));
    assert!(user_text_references_filesystem_path(
        "How many projects are at ~/projects?"
    ));
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
fn detects_incomplete_retry_plan_despite_length() {
    let response = "I've started breaking down your goal into specific tasks. I've created a plan \
to first research the 2026 AI job market and then synthesize that into your personalized morning \
briefing.\n\nI attempted to launch a research specialist to begin the first phase, but the request \
timed out. I'm monitoring the system and will retry the research task as soon as the connection is \
stable.\n\nCurrent Plan:\n1. Research Phase: Deep dive into trends, roles, and skills.\n\
2. Synthesis Phase: Organize findings into a morning briefing.";
    assert!(looks_like_incomplete_retry_plan(response));
}

#[test]
fn completed_briefing_with_next_steps_is_not_an_incomplete_retry_plan() {
    let response = "Market Snapshot: AI hiring is concentrating around applied engineering. \
Target Roles: GenAI engineer and AI product manager. Interview Edge: prepare concrete evaluation \
and deployment examples. Next steps: tailor these findings to your experience.";
    assert!(!looks_like_incomplete_retry_plan(response));
}

#[test]
fn queued_background_specialist_ack_is_not_an_incomplete_retry_plan() {
    let response = "A research specialist is running in the background. The result will be \
delivered through this session when it completes.";
    assert!(!looks_like_incomplete_retry_plan(response));
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
        memory_persistence_allowed: true,
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
    assert!(suggestion.contains("Automatic tool and model recovery"));
    assert!(!suggestion.contains('?'));
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
