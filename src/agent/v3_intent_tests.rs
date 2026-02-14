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
fn test_classify_intent_recurring_without_timing_no_heuristic_schedule() {
    let mut gate = gate_with_answer(false);
    gate.complexity = Some("complex".to_string());
    gate.schedule = None;
    gate.schedule_type = Some("recurring".to_string());
    gate.schedule_cron = None;
    let (complexity, _) =
        classify_intent_complexity("monitor my account and post 3 times per day", &gate);
    assert_eq!(complexity, IntentComplexity::Complex);
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
