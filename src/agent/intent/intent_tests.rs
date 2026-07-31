use super::*;

#[test]
fn test_classify_intent_complexity_simple_command() {
    assert_eq!(
        classify_intent_complexity("run ls -la"),
        IntentComplexity::Simple
    );
}

#[test]
fn test_classify_intent_plain_requests_stay_simple() {
    for msg in ["do something", "check status now", "what is rust?"] {
        assert_eq!(
            classify_intent_complexity(msg),
            IntentComplexity::Simple,
            "'{msg}' should be Simple"
        );
    }
}

#[test]
fn test_classify_intent_scheduled_one_shot_heuristic() {
    let complexity = classify_intent_complexity("remind me in 2h");
    assert!(matches!(
        complexity,
        IntentComplexity::Scheduled {
            ref schedule_raw,
            is_one_shot: true,
        } if schedule_raw == "in 2h"
    ));
}

#[test]
fn test_classify_intent_scheduled_recurring_heuristic() {
    let complexity = classify_intent_complexity("monitor API every 6h");
    assert!(matches!(
        complexity,
        IntentComplexity::Scheduled {
            ref schedule_raw,
            is_one_shot: false,
        } if schedule_raw == "every 6h"
    ));
}

#[test]
fn test_classify_intent_recurring_without_timing_returns_scheduled_missing_timing() {
    assert_eq!(
        classify_intent_complexity("monitor my account and post 3 times per day"),
        IntentComplexity::ScheduledMissingTiming
    );
}

#[test]
fn test_classify_intent_cross_project_analysis_promoted_to_complex() {
    assert_eq!(
        classify_intent_complexity(
            "Compare the package.json files across all my projects, identify shared \
             dependencies, calculate total node_modules disk usage, and summarize \
             version conflicts."
        ),
        IntentComplexity::Complex
    );
}

#[test]
fn test_build_deploy_and_url_handoff_stays_with_primary_agent() {
    assert_eq!(
        classify_intent_complexity(
            "Can you create a website about Ecuador in the ~/projects folder and push to a \
             worker on cloudflare and send me the URL when it's done. Make it look nice."
        ),
        IntentComplexity::Simple
    );
}

#[test]
fn test_small_single_artifact_creation_stays_simple() {
    assert_eq!(
        classify_intent_complexity("Create a file named notes.txt."),
        IntentComplexity::Simple
    );
}

#[test]
fn structured_task_shape_can_promote_semantic_durable_work() {
    let fallback = IntentComplexity::Simple;
    let (complexity, accepted) = refine_intent_complexity_with_task_shape(
        fallback,
        IntentTaskShape {
            execution_mode: Some("durable"),
            confidence: Some("high"),
            independent_workstreams: Some(1),
            requires_background_continuation: Some(true),
        },
    );

    assert!(accepted);
    assert_eq!(complexity, IntentComplexity::Complex);
}

#[test]
fn structured_task_shape_can_demote_long_but_cohesive_work() {
    let fallback = IntentComplexity::Complex;
    let (complexity, accepted) = refine_intent_complexity_with_task_shape(
        fallback,
        IntentTaskShape {
            execution_mode: Some("inline"),
            confidence: Some("medium"),
            independent_workstreams: Some(1),
            requires_background_continuation: Some(false),
        },
    );

    assert!(accepted);
    assert_eq!(complexity, IntentComplexity::Simple);
}

#[test]
fn structured_multistage_delivery_without_background_need_stays_inline() {
    let (complexity, accepted) = refine_intent_complexity_with_task_shape(
        IntentComplexity::Simple,
        IntentTaskShape {
            execution_mode: Some("inline"),
            confidence: Some("high"),
            independent_workstreams: Some(1),
            requires_background_continuation: Some(false),
        },
    );
    assert!(accepted);
    assert_eq!(complexity, IntentComplexity::Simple);
}

#[test]
fn structured_task_shape_rejects_unsupported_durable_label() {
    let fallback = IntentComplexity::Simple;
    let (complexity, accepted) = refine_intent_complexity_with_task_shape(
        fallback,
        IntentTaskShape {
            execution_mode: Some("durable"),
            confidence: Some("high"),
            independent_workstreams: Some(1),
            requires_background_continuation: Some(false),
        },
    );

    assert!(!accepted);
    assert_eq!(complexity, IntentComplexity::Simple);
}

#[test]
fn structured_task_shape_rejects_low_confidence_or_inconsistent_output() {
    let (low_confidence, accepted) = refine_intent_complexity_with_task_shape(
        IntentComplexity::Simple,
        IntentTaskShape {
            execution_mode: Some("durable"),
            confidence: Some("low"),
            independent_workstreams: Some(3),
            requires_background_continuation: Some(true),
        },
    );
    assert!(!accepted);
    assert_eq!(low_confidence, IntentComplexity::Simple);

    let (inconsistent, accepted) = refine_intent_complexity_with_task_shape(
        IntentComplexity::Complex,
        IntentTaskShape {
            execution_mode: Some("inline"),
            confidence: Some("high"),
            independent_workstreams: Some(2),
            requires_background_continuation: Some(false),
        },
    );
    assert!(!accepted);
    assert_eq!(inconsistent, IntentComplexity::Complex);
}

#[test]
fn structured_task_shape_never_overrides_parsed_schedule() {
    let scheduled = IntentComplexity::Scheduled {
        schedule_raw: "every 6h".to_string(),
        is_one_shot: false,
    };
    let (complexity, accepted) = refine_intent_complexity_with_task_shape(
        scheduled.clone(),
        IntentTaskShape {
            execution_mode: Some("inline"),
            confidence: Some("high"),
            independent_workstreams: Some(1),
            requires_background_continuation: Some(false),
        },
    );

    assert!(!accepted);
    assert_eq!(complexity, scheduled);
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
fn test_detect_schedule_heuristic_ignores_background_command_still_running() {
    // Still-running notices carry server output that can include timestamps.
    let detected = detect_schedule_heuristic(
        "[Background command still running]\n\
         Command: `cd ~/projects/app && npm run dev`\n\
         Running for: 2m 20s\n\
         Output so far:\n✓ Ready in 2.8s (Mar 16 13:22:51 2026)\n\
         ⚠ Port 3000 is in use, trying 3001 instead.",
    );
    assert!(
        detected.is_none(),
        "Still-running background output with dates should not trigger schedule"
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
