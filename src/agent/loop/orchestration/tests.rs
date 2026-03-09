use super::memory_scope::{is_low_signal_goal_text, scope_goal_memory_to_project_hints};
use crate::traits::{Fact, Procedure};
use chrono::Utc;

fn fact(category: &str, key: &str, value: &str) -> Fact {
    Fact {
        id: 1,
        category: category.to_string(),
        key: key.to_string(),
        value: value.to_string(),
        source: "test".to_string(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        superseded_at: None,
        recall_count: 0,
        last_recalled_at: None,
        channel_id: None,
        privacy: crate::types::FactPrivacy::Global,
    }
}

fn procedure(name: &str, trigger: &str) -> Procedure {
    Procedure {
        id: 1,
        name: name.to_string(),
        trigger_pattern: trigger.to_string(),
        steps: vec!["step".to_string()],
        success_count: 1,
        failure_count: 0,
        avg_duration_secs: None,
        last_used_at: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}
#[test]
fn scope_goal_memory_keeps_only_matching_project_hints() {
    let facts = vec![
        fact("project", "test-project-framework", "Tailwind"),
        fact("project", "oaxaca-mezcal-tours-framework", "React"),
    ];
    let procs = vec![
        procedure("test-project deploy", "deploy test-project"),
        procedure("mezcal deploy", "deploy oaxaca-mezcal-tours"),
    ];

    let (scoped_facts, scoped_procs) =
        scope_goal_memory_to_project_hints(facts, procs, &["test-project".to_string()]);

    assert_eq!(scoped_facts.len(), 1);
    assert!(scoped_facts[0].key.contains("test-project"));
    assert_eq!(scoped_procs.len(), 1);
    assert!(scoped_procs[0].name.contains("test-project"));
}

#[test]
fn scope_goal_memory_returns_empty_when_project_hints_do_not_match() {
    let facts = vec![fact("project", "oaxaca-mezcal-tours-framework", "React")];
    let procs = vec![procedure("mezcal deploy", "deploy oaxaca-mezcal-tours")];

    let (scoped_facts, scoped_procs) =
        scope_goal_memory_to_project_hints(facts, procs, &["test-project".to_string()]);

    assert!(scoped_facts.is_empty());
    assert!(scoped_procs.is_empty());
}

#[test]
fn low_signal_goal_text_detects_vague_prompt() {
    assert!(is_low_signal_goal_text(
        "You are a senior designer and frontend developer. Do what you consider the best."
    ));
}

#[test]
fn low_signal_goal_text_keeps_specific_goal() {
    assert!(!is_low_signal_goal_text(
        "Build a full-stack website and deploy to AWS us-east-1."
    ));
}
