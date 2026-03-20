use super::*;

fn gate_with_answer(can_answer: bool) -> IntentGateDecision {
    IntentGateDecision {
        can_answer_now: Some(can_answer),
        needs_tools: Some(!can_answer),
        needs_clarification: Some(false),
        clarifying_question: None,
        missing_info: vec![],
        complexity: None,
        cancel_intent: None,
        cancel_scope: None,
        is_acknowledgment: None,
        schedule: None,
        schedule_type: None,
        schedule_cron: None,
        domains: vec![],
    }
}

#[test]
fn test_parse_intent_gate_is_acknowledgment() {
    // The LLM classifies acknowledgments via the intent gate JSON —
    // no hardcoded word lists needed, works in any language.
    let gate = parse_intent_gate_json(r#"{"complexity":"knowledge","is_acknowledgment":true}"#);
    assert_eq!(gate.unwrap().is_acknowledgment, Some(true));

    let gate = parse_intent_gate_json(r#"{"complexity":"simple","is_acknowledgment":false}"#);
    assert_eq!(gate.unwrap().is_acknowledgment, Some(false));

    // Missing field → None (backward compatible)
    let gate = parse_intent_gate_json(r#"{"complexity":"simple"}"#);
    assert_eq!(gate.unwrap().is_acknowledgment, None);
}

#[test]
fn test_parse_intent_gate_cancel_intent() {
    let gate = parse_intent_gate_json(r#"{"complexity":"simple","cancel_intent":true}"#)
        .expect("expected parsed intent gate");
    assert_eq!(gate.cancel_intent, Some(true));
    assert_eq!(gate.cancel_scope, None);

    let gate =
        parse_intent_gate_json(r#"{"complexity":"simple"}"#).expect("expected parsed intent gate");
    assert_eq!(gate.cancel_intent, None);
}

#[test]
fn test_parse_intent_gate_cancel_scope() {
    let gate = parse_intent_gate_json(
        r#"{"complexity":"simple","cancel_intent":true,"cancel_scope":"targeted"}"#,
    )
    .expect("expected parsed intent gate");
    assert_eq!(gate.cancel_intent, Some(true));
    assert_eq!(gate.cancel_scope.as_deref(), Some("targeted"));

    let gate = parse_intent_gate_json(
        r#"{"complexity":"simple","cancel_intent":true,"cancel_scope":"generic"}"#,
    )
    .expect("expected parsed intent gate");
    assert_eq!(gate.cancel_scope.as_deref(), Some("generic"));

    let gate = parse_intent_gate_json(
        r#"{"complexity":"simple","cancel_intent":true,"cancel_scope":"unexpected"}"#,
    )
    .expect("expected parsed intent gate");
    assert_eq!(gate.cancel_scope, None);
}

#[test]
fn test_classify_intent_complexity_knowledge() {
    let gate = gate_with_answer(true);
    let (complexity, tools) = classify_intent_complexity("What's my name?", &gate);
    assert_eq!(complexity, IntentComplexity::Knowledge);
    assert!(tools.is_empty());
}

#[test]
fn test_classify_complexity_knowledge_requires_no_tools() {
    let mut gate = gate_with_answer(true);
    gate.needs_tools = Some(true);
    let (complexity, _) = classify_intent_complexity("Send me my resume", &gate);
    assert_eq!(complexity, IntentComplexity::Simple);
}

#[test]
fn test_classify_intent_complexity_simple() {
    let gate = gate_with_answer(false);
    let (complexity, _tools) = classify_intent_complexity("run ls -la", &gate);
    assert_eq!(complexity, IntentComplexity::Simple);
}

#[test]
fn test_classify_intent_scheduled_one_shot() {
    let mut gate = gate_with_answer(false);
    gate.schedule = Some("in 2h".to_string());
    gate.schedule_type = Some("one_shot".to_string());
    let (complexity, _) = classify_intent_complexity("remind me in 2h", &gate);
    assert!(matches!(
        complexity,
        IntentComplexity::Scheduled {
            is_one_shot: true,
            ..
        }
    ));
}

#[test]
fn test_classify_intent_scheduled_recurring() {
    let mut gate = gate_with_answer(false);
    gate.schedule = Some("every 6h".to_string());
    gate.schedule_type = Some("recurring".to_string());
    let (complexity, _) = classify_intent_complexity("monitor every 6h", &gate);
    assert!(matches!(
        complexity,
        IntentComplexity::Scheduled {
            is_one_shot: false,
            ..
        }
    ));
}

#[test]
fn test_classify_intent_scheduled_with_llm_cron() {
    let mut gate = gate_with_answer(false);
    gate.schedule = Some("3 times per day".to_string());
    gate.schedule_type = Some("recurring".to_string());
    gate.schedule_cron = Some("0 */8 * * *".to_string());
    let (complexity, _) = classify_intent_complexity("post 3 times per day", &gate);
    assert!(matches!(
        complexity,
        IntentComplexity::Scheduled {
            schedule_cron: Some(ref cron),
            ..
        } if cron == "0 */8 * * *"
    ));
}

#[test]
fn test_classify_intent_scheduled_with_cron_only() {
    let mut gate = gate_with_answer(false);
    gate.schedule = None;
    gate.schedule_type = Some("recurring".to_string());
    gate.schedule_cron = Some("0 */8 * * *".to_string());
    let (complexity, _) = classify_intent_complexity("post repeatedly", &gate);
    assert!(matches!(
        complexity,
        IntentComplexity::Scheduled {
            schedule_raw: ref raw,
            schedule_cron: Some(ref cron),
            ..
        } if raw == "0 */8 * * *" && cron == "0 */8 * * *"
    ));
}

#[test]
fn test_classify_intent_recurring_without_timing_returns_scheduled_missing_timing() {
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("complex".to_string());
    gate.schedule = None;
    gate.schedule_type = Some("recurring".to_string());
    gate.schedule_cron = None;
    let (complexity, _) =
        classify_intent_complexity("monitor my account and post 3 times per day", &gate);
    assert_eq!(complexity, IntentComplexity::ScheduledMissingTiming);
}

#[test]
fn test_classify_intent_schedule_takes_priority() {
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("complex".to_string());
    gate.schedule = Some("daily at 9am".to_string());
    gate.schedule_type = Some("recurring".to_string());
    let (complexity, _) = classify_intent_complexity("daily at 9am monitor deploy", &gate);
    assert!(matches!(complexity, IntentComplexity::Scheduled { .. }));
}

#[test]
fn test_classify_intent_no_schedule_stays_simple() {
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("simple".to_string());
    gate.schedule = None;
    gate.schedule_type = None;
    let (complexity, _) = classify_intent_complexity("check status now", &gate);
    assert_eq!(complexity, IntentComplexity::Simple);
}

#[test]
fn test_classify_intent_complexity_complex() {
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("complex".to_string());
    // Genuinely complex: long, persistent multi-session project description
    let (complexity, _) = classify_intent_complexity(
        "I need you to build a new microservice that handles user authentication. This should include JWT token generation, refresh token rotation, rate limiting, database schema design, API documentation, integration tests, load testing, and a CI/CD pipeline. Deploy to staging first, then production after review.",
        &gate,
    );
    assert_eq!(complexity, IntentComplexity::Complex);
}

#[test]
fn test_classify_intent_medium_complex_trusted() {
    // Messages over 50 chars with complexity="complex" are trusted as Complex
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("complex".to_string());
    let (complexity, _) = classify_intent_complexity(
        "Build me a website with authentication and deploy it to Vercel with a custom domain setup",
        &gate,
    );
    assert_eq!(complexity, IntentComplexity::Complex);
}

#[test]
fn test_classify_intent_compound_with_complexity_stays_complex() {
    // No lexical guardrail downgrades: respect explicit model complexity.
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("complex".to_string());
    let (complexity, _) =
        classify_intent_complexity("deploy the app and then set up monitoring", &gate);
    assert_eq!(complexity, IntentComplexity::Complex);
}

#[test]
fn test_classify_intent_sequential_tool_request_stays_complex_when_marked() {
    // Respect explicit model complexity even for numbered task lists.
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("complex".to_string());
    let (complexity, _) = classify_intent_complexity(
        "I need you to do a complex multi-step project: 1) Run \"ls -la /tmp\" on the terminal, 2) Search the web for \"Rust async traits 2025\", 3) Run \"df -h\" on the terminal, 4) Write a report combining all the findings to /tmp/full_report.txt.",
        &gate,
    );
    assert_eq!(complexity, IntentComplexity::Complex);
}

#[test]
fn test_classify_knowledge_downgraded_to_simple_when_cant_answer() {
    // When can_answer_now=false but complexity="knowledge", the model can't
    // answer from context. Downgrade to Simple so tools can try (memory,
    // manage_people, etc.) instead of returning a fallback message.
    let gate = IntentGateDecision {
        can_answer_now: Some(false),
        needs_tools: Some(false),
        needs_clarification: Some(false),
        clarifying_question: None,
        missing_info: vec![],
        complexity: Some("knowledge".to_string()),
        cancel_intent: None,
        cancel_scope: None,
        is_acknowledgment: None,
        schedule: None,
        schedule_type: None,
        schedule_cron: None,
        domains: vec![],
    };
    let (complexity, _) = classify_intent_complexity("Who is bella?", &gate);
    assert_eq!(complexity, IntentComplexity::Simple);
}

#[test]
fn test_classify_unknown_complexity_defaults_simple() {
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("unknown_value".to_string());
    let (complexity, _) = classify_intent_complexity("do something", &gate);
    assert_eq!(complexity, IntentComplexity::Simple);
}

#[test]
fn test_classify_no_complexity_defaults_simple() {
    let gate = gate_with_answer(false);
    assert!(gate.complexity.is_none());
    let (complexity, _) = classify_intent_complexity("do something", &gate);
    assert_eq!(complexity, IntentComplexity::Simple);
}

#[test]
fn test_classify_missing_complexity_promotes_cross_project_analysis_to_complex() {
    let gate = gate_with_answer(false);
    assert!(gate.complexity.is_none());
    let (complexity, _) = classify_intent_complexity(
        "Compare the package.json files across all my projects, identify shared dependencies, calculate total node_modules disk usage, and summarize version conflicts.",
        &gate,
    );
    assert_eq!(complexity, IntentComplexity::Complex);
}

#[test]
fn test_classify_missing_complexity_keeps_simple_requests_simple() {
    let gate = gate_with_answer(false);
    assert!(gate.complexity.is_none());
    let (complexity, _) = classify_intent_complexity("run ls -la", &gate);
    assert_eq!(complexity, IntentComplexity::Simple);
}

#[test]
fn test_classify_complexity_knowledge_field_with_can_answer() {
    // When can_answer_now=true, knowledge complexity stays Knowledge
    let mut gate = gate_with_answer(true);
    gate.complexity = Some("knowledge".to_string());
    let (complexity, _) = classify_intent_complexity("what is rust?", &gate);
    assert_eq!(complexity, IntentComplexity::Knowledge);
}

#[test]
fn test_classify_complexity_no_guardrail_downgrade_for_acknowledgments() {
    // No lexical guardrail downgrades: respect explicit model complexity.
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("complex".to_string());
    for msg in &[
        "ok cool thanks",
        "sure",
        "thanks!",
        "yes please",
        "got it!",
        "hello there",
    ] {
        let (complexity, _) = classify_intent_complexity(msg, &gate);
        assert_eq!(
            complexity,
            IntentComplexity::Complex,
            "'{msg}' should remain Complex"
        );
    }
}

#[test]
fn test_classify_complexity_no_guardrail_downgrade_for_short_commands() {
    // No lexical guardrail downgrades: respect explicit model complexity.
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("complex".to_string());
    for msg in &["run ls -la", "echo hello", "check the status"] {
        let (complexity, _) = classify_intent_complexity(msg, &gate);
        assert_eq!(
            complexity,
            IntentComplexity::Complex,
            "'{msg}' should remain Complex"
        );
    }
}

#[test]
fn test_classify_complexity_guardrail_allows_real_complex() {
    // Genuinely complex multi-step requests (50+ chars, persistent projects) should still be Complex
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("complex".to_string());
    let (complexity, _) = classify_intent_complexity(
        "Build a REST API with authentication, set up a PostgreSQL database with migrations, create the Terraform infrastructure for AWS deployment, configure CI/CD with GitHub Actions, add comprehensive integration tests, set up monitoring with CloudWatch, and prepare documentation for the team.",
        &gate,
    );
    assert_eq!(complexity, IntentComplexity::Complex);
}

#[test]
fn test_parse_intent_gate_with_complexity() {
    let json = r#"{"can_answer_now": false, "needs_tools": true, "complexity": "complex"}"#;
    let parsed = parse_intent_gate_json(json).unwrap();
    assert_eq!(parsed.complexity.as_deref(), Some("complex"));
}

#[test]
fn test_parse_intent_gate_with_schedule() {
    let json = r#"{"can_answer_now": false, "needs_tools": true, "schedule": "every 6h", "schedule_type": "recurring", "schedule_cron": "0 */6 * * *"}"#;
    let parsed = parse_intent_gate_json(json).unwrap();
    assert_eq!(parsed.schedule.as_deref(), Some("every 6h"));
    assert_eq!(parsed.schedule_type.as_deref(), Some("recurring"));
    assert_eq!(parsed.schedule_cron.as_deref(), Some("0 */6 * * *"));
}

#[test]
fn test_parse_intent_gate_with_domains() {
    let json =
        r#"{"can_answer_now": false, "needs_tools": true, "domains": ["Rust", "docker", "rust"]}"#;
    let parsed = parse_intent_gate_json(json).unwrap();
    assert_eq!(
        parsed.domains,
        vec!["rust".to_string(), "docker".to_string()]
    );
}

#[test]
fn test_parse_intent_gate_backward_compat() {
    // Old JSON without complexity field should parse fine with None
    let json = r#"{"can_answer_now": true, "needs_tools": false}"#;
    let parsed = parse_intent_gate_json(json).unwrap();
    assert!(parsed.complexity.is_none());
    assert!(parsed.schedule.is_none());
    assert!(parsed.schedule_type.is_none());
    assert!(parsed.schedule_cron.is_none());
}

#[test]
fn test_detect_schedule_heuristic_in_time() {
    let detected = detect_schedule_heuristic("remind me in 2h");
    assert_eq!(detected, Some(("in 2h".to_string(), true)));
}

#[test]
fn test_detect_schedule_heuristic_recurring() {
    let detected = detect_schedule_heuristic("monitor API every 6h");
    assert_eq!(detected, Some(("every 6h".to_string(), false)));
}

#[test]
fn test_detect_schedule_heuristic_tomorrow() {
    let detected = detect_schedule_heuristic("check deployment tomorrow at 9am");
    assert_eq!(detected, Some(("tomorrow at 9am".to_string(), true)));
}

#[test]
fn test_detect_schedule_heuristic_today_with_timezone() {
    let detected = detect_schedule_heuristic("send me a note today at 11:09pm EST");
    assert_eq!(detected, Some(("today at 11:09pm EST".to_string(), true)));
}

#[test]
fn test_detect_schedule_heuristic_each_interval() {
    let detected = detect_schedule_heuristic("give me 2 jokes. 1 each 5 minutes.");
    assert_eq!(detected, Some(("each 5 minutes".to_string(), false)));
}

#[test]
fn test_detect_schedule_heuristic_no_schedule() {
    let detected = detect_schedule_heuristic("check deployment status now");
    assert!(detected.is_none());
}

#[test]
fn test_detect_schedule_heuristic_ignores_schedule_reference_query() {
    let detected = detect_schedule_heuristic(
        "i want you to give me the details about this scheduled goal: \
         \"English Research: Researching English pronunciation/phonetics relevant to Spanish \
         (3 recurring slots daily: 5 AM, 12 PM, and 7 PM EST).\"",
    );
    assert!(detected.is_none());
}

#[test]
fn test_looks_like_recurring_intent_without_timing_times_per_day() {
    assert!(looks_like_recurring_intent_without_timing(
        "create 3 posts per language 3 times per day"
    ));
}

#[test]
fn test_looks_like_recurring_intent_without_timing_false_when_timed() {
    assert!(!looks_like_recurring_intent_without_timing(
        "monitor API every 6h"
    ));
}

#[test]
fn test_internal_maintenance_intent_detects_legacy_phrases() {
    assert!(is_internal_maintenance_intent(
        "Maintain knowledge base: process embeddings, consolidate memories, decay old facts"
    ));
    assert!(is_internal_maintenance_intent(
        "Maintain memory health: prune old events, clean up retention, remove stale data"
    ));
}

#[test]
fn test_internal_maintenance_intent_ignores_normal_requests() {
    assert!(!is_internal_maintenance_intent(
        "Build a full-stack website with auth and CI/CD"
    ));
    assert!(!is_internal_maintenance_intent(
        "monitor api every 6h and send status updates"
    ));
}

#[test]
fn test_contains_keyword_as_words() {
    // Exact word match
    assert!(contains_keyword_as_words("deploy the app", "deploy"));
    assert!(contains_keyword_as_words("please build it now", "build"));
    // Multi-word keyword match
    assert!(contains_keyword_as_words("set up monitoring", "set up"));
    assert!(contains_keyword_as_words(
        "create a project from scratch",
        "create a project"
    ));
    // Should NOT match derived forms
    assert!(!contains_keyword_as_words("the deployed site", "deploy"));
    assert!(!contains_keyword_as_words("deployment configs", "deploy"));
    assert!(!contains_keyword_as_words("building blocks", "build"));
    assert!(!contains_keyword_as_words(
        "implementation details",
        "implement"
    ));
    assert!(!contains_keyword_as_words("refactoring code", "refactor"));
    // Punctuation should act as word boundary
    assert!(contains_keyword_as_words(
        "build, test, and deploy.",
        "deploy"
    ));
    assert!(contains_keyword_as_words("(deploy)", "deploy"));
}

#[test]
fn test_detect_schedule_heuristic_ignores_memory_storage_with_date() {
    // "Remember my birthday is October 15" should NOT trigger scheduling
    let detected =
        detect_schedule_heuristic("Remember that my birthday is October 15 and I love sushi");
    assert!(
        detected.is_none(),
        "Memory-storage intent with date should not trigger schedule: got {:?}",
        detected
    );
}

#[test]
fn test_detect_schedule_heuristic_ignores_remember_my_date() {
    let detected = detect_schedule_heuristic("Remember my anniversary is June 20th");
    assert!(detected.is_none());
}

#[test]
fn test_detect_schedule_heuristic_ignores_note_that_date() {
    let detected = detect_schedule_heuristic("Note that I was born on March 5th");
    assert!(detected.is_none());
}

#[test]
fn test_detect_schedule_heuristic_allows_remind_me_with_date() {
    // "Remind me on October 15" IS a scheduling request
    let detected = detect_schedule_heuristic("Remind me to buy a gift on October 15");
    assert!(
        detected.is_some(),
        "Scheduling intent should still trigger: got None"
    );
}

#[test]
fn test_detect_schedule_heuristic_allows_schedule_with_memory_verb() {
    // "Remember to remind me" — has both memory and scheduling verbs, scheduling wins
    let detected = detect_schedule_heuristic("Remember that you need to remind me on March 5th");
    assert!(
        detected.is_some(),
        "When both memory and scheduling verbs present, scheduling should win"
    );
}

#[test]
fn test_detect_schedule_heuristic_ignores_remember_these_facts_with_date() {
    // "remember these facts" includes a birthday date — should NOT trigger scheduling
    let detected = detect_schedule_heuristic(
        "I want you to remember these important facts about me: 1) My favorite programming language is Rust, \
         2) I prefer dark mode, 3) My birthday is July 15th, 4) I'm allergic to shellfish, 5) My dog's name is Luna.",
    );
    assert!(
        detected.is_none(),
        "Remember-these-facts with embedded date should not trigger schedule: got {:?}",
        detected
    );
}

#[test]
fn test_detect_schedule_heuristic_ignores_facts_about_me_with_date() {
    // "facts about me" context without explicit memory verb
    let detected = detect_schedule_heuristic(
        "Here are some facts about me: my birthday is March 10th and I like coffee",
    );
    assert!(
        detected.is_none(),
        "Facts-about-me context with date should not trigger schedule: got {:?}",
        detected
    );
}

#[test]
fn test_detect_schedule_heuristic_ignores_store_these_with_date() {
    let detected = detect_schedule_heuristic(
        "Store these details: I was born on December 25th, I work at Acme Corp",
    );
    assert!(
        detected.is_none(),
        "Store-these with date should not trigger schedule"
    );
}

#[test]
fn test_detect_schedule_heuristic_compound_message_with_date() {
    // Compound message where date is in a fact-storage sub-task
    let detected = detect_schedule_heuristic(
        "I need you to do 3 things: (1) Remember that my birthday is October 15. \
         (2) Check the blog post. (3) Create a Python script.",
    );
    assert!(
        detected.is_none(),
        "Compound message with memory intent should not trigger schedule"
    );
}

#[test]
fn test_detect_schedule_heuristic_ignores_background_command_completion() {
    // Background command output contains dates from find/ls output
    let detected = detect_schedule_heuristic(
        "[Background command completed]\n\
         Command: `cd '/Users/test/projects' && chmod +x script.sh && ./script.sh`\n\
         Output:\nMar 16 13:22:51 2026 - /Users/test/projects/file.db\n\
         Jan 5 09:00:00 2026 - /Users/test/projects/data.csv",
    );
    assert!(
        detected.is_none(),
        "Background command output with dates should not trigger schedule"
    );
}

#[test]
fn test_detect_schedule_heuristic_ignores_file_edit_with_dates() {
    // User asks to finish a file that mentions dates in its content scope.
    let detected = detect_schedule_heuristic(
        "I noticed my social-media-plan.md got cut off mid-sentence. Can you finish it? \
         Read the current file, then append the remaining content starting from where it \
         was truncated. The plan should cover the full 2 weeks (March 18 through March 31) \
         as originally intended.",
    );
    assert!(
        detected.is_none(),
        "File editing request with dates should not trigger schedule"
    );
}

#[test]
fn test_detect_schedule_heuristic_ignores_file_creation_with_dates() {
    // User asks to create a file covering a date range.
    let detected = detect_schedule_heuristic(
        "Create a file ~/projects/blog/social-media-plan.md with a 2-week calendar \
         from March 18 to March 31 that promotes my blog posts.",
    );
    assert!(
        detected.is_none(),
        "File creation request with dates should not trigger schedule"
    );
}

#[test]
fn test_detect_schedule_heuristic_ignores_truncated_file_fix() {
    // User says a file was truncated and needs fixing.
    let detected = detect_schedule_heuristic(
        "The file output.json was truncated on March 20. Read it and complete the \
         missing entries through March 31.",
    );
    assert!(
        detected.is_none(),
        "Truncated file fix request with dates should not trigger schedule"
    );
}

#[test]
fn test_detect_schedule_heuristic_still_works_with_file_and_schedule_verb() {
    // A genuine scheduling request that also mentions a file.
    let detected =
        detect_schedule_heuristic("Remind me on March 18 to edit the file ~/projects/plan.md");
    assert!(
        detected.is_some(),
        "Scheduling request with file ref should still trigger schedule"
    );
}

#[test]
fn test_detect_schedule_heuristic_ignores_past_tense_date_recall() {
    // "What did we talk about ... March 3rd?" is a recall query, not scheduling.
    // The bare month-day "March 3rd" should NOT trigger scheduling without a verb.
    let detected =
        detect_schedule_heuristic("What did we talk about two weeks ago around March 3rd?");
    assert!(
        detected.is_none(),
        "Past-tense recall with bare date should not trigger schedule: got {:?}",
        detected
    );
}

#[test]
fn test_detect_schedule_heuristic_ignores_bare_date_without_verb() {
    // Bare month-day in various non-scheduling contexts.
    for input in [
        "The deadline was March 10",
        "We deployed on March 12",
        "What happened on January 5th?",
        "The incident on February 20th needs a postmortem",
        "My birthday is March 15",
    ] {
        let detected = detect_schedule_heuristic(input);
        assert!(
            detected.is_none(),
            "Bare date without scheduling verb should not trigger schedule for: {input}"
        );
    }
}

#[test]
fn test_detect_schedule_heuristic_fires_with_scheduling_verb_and_date() {
    // Month-day WITH a scheduling verb should still fire.
    for (input, expected_some) in [
        ("Remind me on March 5th to check the server", true),
        ("Schedule a review for October 15", true),
        ("Alert me on January 20th about the renewal", true),
        ("Notify me on December 1st when the sale starts", true),
        ("Check what happened on March 3rd", false),
    ] {
        let detected = detect_schedule_heuristic(input);
        assert_eq!(
            detected.is_some(),
            expected_some,
            "Expected is_some={expected_some} for: {input}, got {:?}",
            detected
        );
    }
}
