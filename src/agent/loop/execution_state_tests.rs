//! Tests for `execution_state` (split out from the original module via `#[path]`).
//!
//! Moved verbatim — no logic changes. Included as a child test module of
//! `execution_state` so `use super::*;` continues to resolve against it.

use super::*;
use crate::agent::{CompletionContract, CompletionTaskKind, FollowupMode, TurnContext};
use crate::traits::{
    ToolCallMetadata, ToolCallSemantics, ToolResultPresentation, ToolTargetHintKind,
};

#[test]
fn top_level_requests_use_model_directed_standard_budget() {
    let turn_context = TurnContext {
        primary_project_scope: Some("/tmp/demo".to_string()),
        ..TurnContext::default()
    };
    let (tier, route_kind, budget) = select_initial_execution_budget(
        "edit /tmp/demo/src/main.rs",
        &turn_context,
        0,
        AgentRole::Orchestrator,
    );
    assert_eq!(tier, BudgetTier::Standard);
    assert_eq!(route_kind, "model_directed");
    assert_eq!(budget.max_validation_rounds, 3);
}

#[test]
fn research_plus_create_gets_standard_not_small() {
    // "Search the web ... then create a file at path.md" should get Standard
    // budget, not Small, even though it has a scoped target (.md).
    let turn_context = TurnContext::default();
    let (tier, route_kind, _budget) = select_initial_execution_budget(
        "Search the web for the top 3 Rust crates then create a markdown file at ~/projects/blog/drafts/rust-crates-2025.md",
        &turn_context,
        0,
        AgentRole::Orchestrator,
    );
    assert_eq!(
        tier,
        BudgetTier::Standard,
        "research+create should get Standard budget"
    );
    assert_eq!(route_kind, "model_directed");
}

#[test]
fn delegated_work_starts_with_extended_budget() {
    let (tier, route_kind, budget) = select_initial_execution_budget(
        "fix the deployment",
        &TurnContext::default(),
        1,
        AgentRole::Executor,
    );
    assert_eq!(tier, BudgetTier::Extended);
    assert_eq!(route_kind, "delegated_multi_step");
    assert!(budget.max_tool_calls >= 16);
}

#[test]
fn compile_step_plan_uses_scope_and_idempotency_for_mutations() {
    let semantics =
        ToolCallSemantics::mutation().with_target_hint(ToolTargetHintKind::Path, "src/main.rs");
    let plan = compile_step_execution_plan(
        "exec-1",
        "operation-1".to_string(),
        None,
        3,
        2,
        "call-1",
        "edit_file",
        r#"{"path":"src/main.rs"}"#,
        &semantics,
        &Default::default(),
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: false,
        },
        &[String::from("/repo")],
    );

    assert_eq!(plan.primary_tool.as_deref(), Some("edit_file"));
    assert_eq!(plan.plan_version, 3);
    assert_eq!(plan.target_scope.allowed_targets.len(), 1);
    assert_eq!(
        plan.target_scope.allowed_targets[0].kind,
        ToolTargetHintKind::ProjectScope
    );
    assert!(plan.target_scope.hard_fail_outside_scope);
    assert!(plan.idempotency_key.is_some());
    assert!(matches!(
        plan.approval_requirement,
        ApprovalRequirement::Required { .. }
    ));
}

#[test]
fn operation_identity_refreshes_only_after_a_different_mutation() {
    let mut state = ExecutionState::new(
        BudgetTier::Standard,
        default_execution_budget(BudgetTier::Standard),
        ExecutionPersistence::Durable,
    );
    let capabilities = ToolCapabilities {
        read_only: false,
        external_side_effect: false,
        needs_approval: false,
        idempotent: false,
        high_impact_write: false,
    };
    let stage_mutation = |state: &mut ExecutionState, base: &str, tool: &str| {
        let key = state.bind_operation_to_effect_revision(base);
        let plan = compile_step_execution_plan(
            "exec-1",
            key,
            None,
            1,
            1,
            "call-1",
            tool,
            "{}",
            &ToolCallSemantics::mutation(),
            &Default::default(),
            capabilities,
            &[],
        );
        state.stage_step(plan);
        assert!(state.begin_staged_step());
        state.record_current_mutation_transition();
    };

    let first_a = state.bind_operation_to_effect_revision("invocation:a");
    stage_mutation(&mut state, "invocation:a", "edit_file");
    assert_eq!(
        state.bind_operation_to_effect_revision("invocation:a"),
        first_a,
        "an operation must not refresh its own retry identity"
    );
    assert!(state
        .bind_operation_to_effect_revision("invocation:check")
        .ends_with(":effect_revision:1"));

    stage_mutation(&mut state, "invocation:b", "write_file");
    assert!(state
        .bind_operation_to_effect_revision("invocation:a")
        .ends_with(":effect_revision:2"));
}

#[test]
fn compile_step_plan_preserves_url_targets_when_project_scope_exists() {
    let semantics = ToolCallSemantics::observation().with_target_hint(
        ToolTargetHintKind::Url,
        "https://clinicaltrials.gov/api/v2/studies",
    );
    let plan = compile_step_execution_plan(
        "exec-1",
        "operation-1".to_string(),
        None,
        3,
        2,
        "call-1",
        "http_request",
        r#"{"url":"https://clinicaltrials.gov/api/v2/studies"}"#,
        &semantics,
        &Default::default(),
        ToolCapabilities {
            read_only: true,
            external_side_effect: true,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        },
        &[String::from("/repo")],
    );

    assert_eq!(plan.target_scope.allowed_targets.len(), 1);
    assert_eq!(
        plan.target_scope.allowed_targets[0].kind,
        ToolTargetHintKind::Url
    );
    assert_eq!(
        plan.target_scope.allowed_targets[0].value,
        "https://clinicaltrials.gov/api/v2/studies"
    );
}

#[test]
fn pure_terminal_observation_does_not_inherit_static_high_impact_approval() {
    let plan = compile_step_execution_plan(
        "exec-1",
        "operation-1".to_string(),
        None,
        1,
        1,
        "call-1",
        "terminal",
        r#"{"command":"node --version"}"#,
        &ToolCallSemantics::observation(),
        &Default::default(),
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        },
        &[String::from("/repo")],
    );
    assert_eq!(plan.approval_requirement, ApprovalRequirement::NotNeeded);
    assert!(plan.idempotency_key.is_none());
}

#[test]
fn operation_retry_budget_separates_invocations_from_dispatched_attempts() {
    let compile = |call_id: &str| {
        compile_step_execution_plan(
            "exec-1",
            "invocation:exec-1:stable-operation".to_string(),
            None,
            1,
            1,
            call_id,
            "terminal",
            r#"{"action":"run","command":"/usr/bin/true","working_dir":"/tmp"}"#,
            &ToolCallSemantics::observation(),
            &Default::default(),
            ToolCapabilities {
                read_only: false,
                external_side_effect: true,
                needs_approval: true,
                idempotent: true,
                high_impact_write: true,
            },
            &[],
        )
    };
    let mut state = ExecutionState::new(
        BudgetTier::Small,
        default_execution_budget(BudgetTier::Small),
        ExecutionPersistence::Ephemeral,
    );

    state.stage_step(compile("proposal-rejected-before-io"));
    assert!(state.begin_staged_step());
    // No dispatch was recorded, so a corrected proposal still owns the first
    // and only operation attempt.
    state.stage_step(compile("corrected-proposal"));
    assert!(state.begin_staged_step());
    state.record_current_operation_dispatch();

    state.stage_step(compile("model-retry-with-new-id"));
    assert!(!state.begin_staged_step());
    assert_eq!(
        state
            .operation_attempts
            .get("invocation:exec-1:stable-operation"),
        Some(&1)
    );
    assert_eq!(
        state
            .operation_invocations
            .get("invocation:exec-1:stable-operation"),
        Some(&2)
    );
    assert_eq!(state.tool_dispatches.get("terminal"), Some(&1));
}

#[test]
fn explicit_single_invocation_limit_counts_pre_io_rejection() {
    let plan = compile_step_execution_plan(
        "exec-1",
        "invocation:exec-1:false".to_string(),
        Some(("contract:task-1:requirements:0".to_string(), 1)),
        1,
        1,
        "proposal-1",
        "run_command",
        r#"{"command":"/usr/bin/false","working_dir":"/tmp"}"#,
        &ToolCallSemantics::observation(),
        &Default::default(),
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        },
        &[],
    );
    let mut state = ExecutionState::new(
        BudgetTier::Small,
        default_execution_budget(BudgetTier::Small),
        ExecutionPersistence::Ephemeral,
    );

    state.stage_step(plan.clone());
    assert!(state.begin_staged_step());
    // The adapter rejects before I/O, so no execution attempt is charged.
    assert!(state.operation_attempts.is_empty());

    state.stage_step(plan);
    assert!(!state.begin_staged_step());
    assert_eq!(
        state
            .obligation_invocations
            .get("contract:task-1:requirements:0"),
        Some(&1)
    );
}

#[test]
fn explicit_cardinality_is_shared_across_distinct_concrete_strategies() {
    let compile = |operation: &str, command: &str| {
        compile_step_execution_plan(
            "exec-1",
            operation.to_string(),
            Some(("contract:task-1:requirements:0".to_string(), 1)),
            1,
            1,
            operation,
            "run_command",
            command,
            &ToolCallSemantics::observation(),
            &Default::default(),
            ToolCapabilities {
                read_only: true,
                external_side_effect: false,
                needs_approval: false,
                idempotent: true,
                high_impact_write: false,
            },
            &[],
        )
    };
    let mut state = ExecutionState::new(
        BudgetTier::Small,
        default_execution_budget(BudgetTier::Small),
        ExecutionPersistence::Ephemeral,
    );

    state.stage_step(compile(
        "invocation:exec-1:first",
        r#"{"command":"/usr/bin/false"}"#,
    ));
    assert!(state.begin_staged_step());
    state.stage_step(compile(
        "invocation:exec-1:alternative",
        r#"{"command":"/usr/bin/true"}"#,
    ));
    assert!(!state.begin_staged_step());
    assert!(!state
        .operation_invocations
        .contains_key("invocation:exec-1:alternative"));
}

#[test]
fn execution_state_reports_budget_exhaustion() {
    let mut state = ExecutionState::new(
        BudgetTier::Small,
        ExecutionBudget {
            max_steps: 1,
            max_tokens: 100,
            max_llm_calls: 1,
            max_tool_calls: 1,
            max_validation_rounds: 1,
            max_wall_clock_ms: 1_000,
        },
        ExecutionPersistence::Ephemeral,
    );
    state.activate_budget_envelope(0, Duration::from_millis(0));
    state.record_llm_call();
    assert_eq!(
        state.exhausted_limit(0, Duration::from_millis(1)),
        Some(ExecutionBudgetLimit::LlmCalls)
    );
}

#[test]
fn inactive_execution_budget_ignores_plain_text_token_usage() {
    let state = ExecutionState::new(
        BudgetTier::None,
        ExecutionBudget {
            max_steps: 24,
            max_tokens: 10,
            max_llm_calls: 1,
            max_tool_calls: 1,
            max_validation_rounds: 1,
            max_wall_clock_ms: 1,
        },
        ExecutionPersistence::Ephemeral,
    );

    assert_eq!(state.exhausted_limit(10_000, Duration::from_secs(30)), None);
}

#[test]
fn provider_delay_counts_toward_owner_visible_wall_clock() {
    let mut state = ExecutionState::new(
        BudgetTier::Small,
        ExecutionBudget {
            max_steps: 10,
            max_tokens: 10_000,
            max_llm_calls: 10,
            max_tool_calls: 10,
            max_validation_rounds: 10,
            max_wall_clock_ms: 1_000,
        },
        ExecutionPersistence::Ephemeral,
    );
    state.activate_budget_envelope(0, Duration::from_millis(100));
    state.provider_timeout_ms = 800;

    assert_eq!(
        state.remaining_wall_clock(Duration::from_millis(900)),
        Some(Duration::from_millis(200))
    );
    assert_eq!(
        state.exhausted_limit(0, Duration::from_millis(1_100)),
        Some(ExecutionBudgetLimit::WallClock)
    );
    assert_eq!(
        state.remaining_wall_clock(Duration::from_millis(1_100)),
        Some(Duration::ZERO)
    );
}

#[test]
fn knowledge_turns_use_standard_budget() {
    let turn_context = TurnContext {
        completion_contract: CompletionContract::default(),
        ..TurnContext::default()
    };
    let (tier, route_kind, _) = select_initial_execution_budget(
        "what's the capital of france",
        &turn_context,
        0,
        AgentRole::Orchestrator,
    );
    assert_eq!(tier, BudgetTier::Standard);
    assert_eq!(route_kind, "model_directed");
}

#[test]
fn scheduled_turns_use_standard_budget() {
    let (tier, route_kind, budget) = select_initial_execution_budget(
        "schedule a daily health check",
        &TurnContext::default(),
        0,
        AgentRole::Orchestrator,
    );
    assert_eq!(tier, BudgetTier::Standard);
    assert_eq!(route_kind, "model_directed");
    assert!(budget.max_validation_rounds >= 3);
}

#[test]
fn read_only_investigation_uses_standard_budget() {
    let (tier, route_kind, _) = select_initial_execution_budget(
        "inspect the latest logs and show me the current status",
        &TurnContext::default(),
        0,
        AgentRole::Orchestrator,
    );
    assert_eq!(tier, BudgetTier::Standard);
    assert_eq!(route_kind, "model_directed");
}

#[test]
fn api_read_requests_use_standard_budget_for_multi_step_lookups() {
    let (tier, route_kind, budget) = select_initial_execution_budget(
        "Using the clinical trials API, give me studies near Fairfax for skin cancer.",
        &TurnContext::default(),
        0,
        AgentRole::Orchestrator,
    );
    assert_eq!(tier, BudgetTier::Standard);
    assert_eq!(route_kind, "model_directed");
    assert!(budget.max_llm_calls >= 18);
    assert_eq!(budget.max_tokens, 0);
}

#[test]
fn connected_content_authoring_requests_stay_in_knowledge_lane() {
    let mut turn_context = TurnContext::default();
    turn_context.completion_contract.connected_content_mode =
        crate::agent::intent_routing::ConnectedContentMode::DraftThenDeliver;
    turn_context.completion_contract.task_kind = CompletionTaskKind::Deliver;
    turn_context.completion_contract.expects_mutation = true;
    let (tier, route_kind, _) = select_initial_execution_budget(
        "Can you post a tweet about your new stuff and make it engaging so people want to comment?",
        &turn_context,
        0,
        AgentRole::Orchestrator,
    );
    assert_eq!(tier, BudgetTier::Standard);
    assert_eq!(route_kind, "model_directed");
}

#[test]
fn account_scoped_connected_content_delivery_uses_external_write_budget() {
    let (tier, route_kind, budget) = select_initial_execution_budget(
        "Can you post a tweet on your account?",
        &TurnContext::default(),
        0,
        AgentRole::Orchestrator,
    );
    assert_eq!(tier, BudgetTier::Standard);
    assert_eq!(route_kind, "model_directed");
    assert!(budget.max_llm_calls >= 18);
}

#[test]
fn auth_management_requests_use_standard_budget() {
    let (tier, route_kind, _) = select_initial_execution_budget(
        "Reconnect my Twitter OAuth account so you can post for me.",
        &TurnContext::default(),
        0,
        AgentRole::Orchestrator,
    );
    assert_eq!(tier, BudgetTier::Standard);
    assert_eq!(route_kind, "model_directed");
}

#[test]
fn contextual_followups_start_with_standard_budget() {
    let turn_context = TurnContext {
        followup_mode: Some(FollowupMode::Followup),
        ..TurnContext::default()
    };
    let (tier, _route_kind, budget) = select_initial_execution_budget(
        "Which one is most relevant to skin cancer?",
        &turn_context,
        0,
        AgentRole::Orchestrator,
    );
    // Tier is Standard regardless of followup context since the
    // base tier is now Standard (no longer None/Small that needed promotion).
    assert_eq!(tier, BudgetTier::Standard);
    assert_eq!(budget.max_tokens, 0);
}

#[test]
fn clarification_followups_promote_scoped_edits_to_standard_budget() {
    let turn_context = TurnContext {
        primary_project_scope: Some("/tmp/demo".to_string()),
        followup_mode: Some(FollowupMode::ClarificationAnswer),
        ..TurnContext::default()
    };
    let (tier, route_kind, budget) = select_initial_execution_budget(
        "Update the config in src/main.rs",
        &turn_context,
        0,
        AgentRole::Orchestrator,
    );
    assert_eq!(tier, BudgetTier::Standard);
    assert_eq!(route_kind, "model_directed");
    assert!(budget.max_validation_rounds >= 3);
}

#[test]
fn extend_budget_on_progress_increases_limits() {
    let mut state = ExecutionState::new(
        BudgetTier::None,
        default_execution_budget(BudgetTier::None),
        ExecutionPersistence::Ephemeral,
    );
    let original_llm = state.budget.max_llm_calls;
    let original_tools = state.budget.max_tool_calls;
    let original_steps = state.budget.max_steps;
    let original_wall = state.budget.max_wall_clock_ms;
    let original_validation = state.budget.max_validation_rounds;

    // No extension when budget envelope is inactive
    assert!(!state.extend_budget_on_progress(
        "write_file",
        &crate::traits::ToolCallSemantics::mutation(),
        "created /tmp/result"
    ));
    assert_eq!(state.budget.max_llm_calls, original_llm);
    assert_eq!(state.budget.max_wall_clock_ms, original_wall);
    assert_eq!(state.budget.max_validation_rounds, original_validation);

    // Extension kicks in once the envelope is active
    state.activate_budget_envelope(0, Duration::from_millis(0));
    assert!(state.extend_budget_on_progress(
        "write_file",
        &crate::traits::ToolCallSemantics::mutation(),
        "created /tmp/result"
    ));
    assert!(state.budget.max_llm_calls > original_llm);
    assert!(state.budget.max_tool_calls > original_tools);
    assert!(state.budget.max_steps > original_steps);
    assert!(state.budget.max_wall_clock_ms > original_wall);
    assert!(state.budget.max_validation_rounds > original_validation);

    // Replaying the same result earns no additional runway.
    let after_first = state.budget.max_llm_calls;
    let after_first_wall = state.budget.max_wall_clock_ms;
    let after_first_validation = state.budget.max_validation_rounds;
    assert!(!state.extend_budget_on_progress(
        "write_file",
        &crate::traits::ToolCallSemantics::mutation(),
        "created /tmp/result"
    ));
    assert_eq!(state.budget.max_llm_calls, after_first);
    assert_eq!(state.budget.max_wall_clock_ms, after_first_wall);
    assert_eq!(state.budget.max_validation_rounds, after_first_validation);

    // A distinct verified outcome can extend the bounded envelope again.
    assert!(state.extend_budget_on_progress(
        "terminal",
        &crate::traits::ToolCallSemantics::observation()
            .with_verification_mode(crate::traits::ToolVerificationMode::ResultContent),
        "tests passed"
    ));
    assert!(state.budget.max_llm_calls > after_first);
    assert!(state.budget.max_wall_clock_ms > after_first_wall);
    assert!(state.budget.max_validation_rounds > after_first_validation);
}

#[test]
fn progress_extensions_are_bounded_and_observation_credit_is_capped() {
    let mut state = ExecutionState::new(
        BudgetTier::None,
        default_execution_budget(BudgetTier::None),
        ExecutionPersistence::Ephemeral,
    );
    state.activate_budget_envelope(0, Duration::from_millis(0));

    let observation = crate::traits::ToolCallSemantics::observation();
    for i in 0..30 {
        state.extend_budget_on_progress("read_file", &observation, &format!("evidence {i}"));
    }
    assert_eq!(state.observation_extensions_used, 4);
    assert_eq!(state.progress_extensions_used, 4);

    let mutation = crate::traits::ToolCallSemantics::mutation();
    for i in 0..30 {
        state.extend_budget_on_progress("write_file", &mutation, &format!("artifact {i}"));
    }
    assert_eq!(state.progress_extensions_used, 12);
}

#[test]
fn resource_pressure_reports_the_most_constrained_dimension_once() {
    let mut state = ExecutionState::new(
        BudgetTier::Standard,
        default_execution_budget(BudgetTier::Standard),
        ExecutionPersistence::Ephemeral,
    );
    state.activate_budget_envelope(0, Duration::ZERO);
    state.budget.max_steps = 10;
    state.budget.max_tool_calls = 10;
    state.steps_used = 8;
    state.tool_calls_used = 9;

    let pressure = state
        .resource_pressure(0, Duration::from_secs(1))
        .expect("pressure at 90 percent");
    assert_eq!(pressure.limit, ExecutionBudgetLimit::ToolCalls);
    assert_eq!(pressure.pct, 90);

    state.resource_pressure_emitted = true;
    assert!(state.resource_pressure(0, Duration::from_secs(1)).is_none());
}

fn test_execution_state() -> ExecutionState {
    ExecutionState::new(
        BudgetTier::None,
        default_execution_budget(BudgetTier::None),
        ExecutionPersistence::Ephemeral,
    )
}

#[test]
fn outcome_ledger_starts_empty() {
    let state = test_execution_state();
    assert!(state.outcome_ledger.is_empty());
}

#[test]
fn outcome_ledger_records_success() {
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: true,
        http_status: Some(201),
        is_external_mutation: true,
        error_summary: None,
        iteration: 1,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    assert_eq!(state.outcome_ledger.len(), 1);
    assert!(state.outcome_ledger[0].success);
}

#[test]
fn outcome_ledger_tracks_failed_external_mutations() {
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: false,
        http_status: Some(403),
        is_external_mutation: true,
        error_summary: Some("duplicate content".to_string()),
        iteration: 1,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    assert!(state.has_uncorrected_failed_external_mutations());
}

#[test]
fn outcome_ledger_ignores_non_external_failures() {
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "read_file".to_string(),
        success: false,
        http_status: None,
        is_external_mutation: false,
        error_summary: Some("file not found".to_string()),
        iteration: 1,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    assert!(!state.has_uncorrected_failed_external_mutations());
}

#[test]
fn attempt_reconciliation_none_when_all_succeeded() {
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: true,
        http_status: Some(201),
        is_external_mutation: true,
        error_summary: None,
        iteration: 1,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    assert!(state.build_attempt_reconciliation_summary().is_none());
}

#[test]
fn attempt_reconciliation_present_when_failures_exist() {
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: true,
        http_status: Some(201),
        is_external_mutation: true,
        error_summary: None,
        iteration: 1,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: false,
        http_status: Some(403),
        is_external_mutation: true,
        error_summary: Some("duplicate content".to_string()),
        iteration: 2,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    let summary = state.build_attempt_reconciliation_summary().unwrap();
    assert!(summary.contains("attempts"));
    assert!(summary.contains("1") && summary.contains("2"));
    assert!(summary.contains("failed"));
    assert!(summary.contains("403"));
    assert!(summary.contains("duplicate content"));
}

#[test]
fn attempt_reconciliation_says_attempts_not_actions() {
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: false,
        http_status: Some(403),
        is_external_mutation: true,
        error_summary: Some("dup".to_string()),
        iteration: 1,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    let summary = state.build_attempt_reconciliation_summary().unwrap();
    assert!(summary.contains("attempt"));
    assert!(!summary.contains("action"));
}

#[test]
fn corrected_failure_same_tool_skips_reconciliation() {
    // Failure at iter 3, then success of SAME tool at iter 7 → corrected
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "run_command".to_string(),
        success: false,
        http_status: None,
        is_external_mutation: true,
        error_summary: Some("could not find Cargo.toml".to_string()),
        iteration: 3,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    state.record_outcome(OutcomeEntry {
        tool_name: "run_command".to_string(),
        success: true,
        http_status: None,
        is_external_mutation: true,
        error_summary: None,
        iteration: 7,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    assert!(state.uncorrected_failed_mutations().is_empty());
    assert!(!state.has_uncorrected_failed_external_mutations());
    assert!(state.build_attempt_reconciliation_summary().is_none());
}

#[test]
fn corrected_failure_different_tool_skips_reconciliation() {
    // Failure via run_command at iter 9, then success via terminal at iter 15
    // → corrected (all failures before last success)
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "run_command".to_string(),
        success: false,
        http_status: None,
        is_external_mutation: true,
        error_summary: Some("could not find Cargo.toml".to_string()),
        iteration: 9,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    state.record_outcome(OutcomeEntry {
        tool_name: "terminal".to_string(),
        success: true,
        http_status: None,
        is_external_mutation: true,
        error_summary: None,
        iteration: 15,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    assert!(state.uncorrected_failed_mutations().is_empty());
    assert!(!state.has_uncorrected_failed_external_mutations());
    assert!(state.build_attempt_reconciliation_summary().is_none());
}

#[test]
fn uncorrected_failure_after_last_success_triggers_reconciliation() {
    // Success at iter 5, then failure at iter 10 → uncorrected
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "terminal".to_string(),
        success: true,
        http_status: None,
        is_external_mutation: true,
        error_summary: None,
        iteration: 5,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: false,
        http_status: Some(500),
        is_external_mutation: true,
        error_summary: Some("server error".to_string()),
        iteration: 10,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    assert_eq!(state.uncorrected_failed_mutations().len(), 1);
    assert!(state.has_uncorrected_failed_external_mutations());
    assert!(state.build_attempt_reconciliation_summary().is_some());
}

#[test]
fn mixed_corrected_and_uncorrected_failures() {
    // run_command FAIL at iter 3 (corrected by terminal SUCCESS at iter 15)
    // http_request FAIL at iter 20 (after last success → uncorrected)
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "run_command".to_string(),
        success: false,
        http_status: None,
        is_external_mutation: true,
        error_summary: Some("not found".to_string()),
        iteration: 3,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    state.record_outcome(OutcomeEntry {
        tool_name: "terminal".to_string(),
        success: true,
        http_status: None,
        is_external_mutation: true,
        error_summary: None,
        iteration: 15,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: false,
        http_status: Some(500),
        is_external_mutation: true,
        error_summary: Some("deploy failed".to_string()),
        iteration: 20,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    let uncorrected = state.uncorrected_failed_mutations();
    assert_eq!(uncorrected.len(), 1);
    assert_eq!(uncorrected[0].tool_name, "http_request");
    assert_eq!(uncorrected[0].iteration, 20);
    let summary = state.build_attempt_reconciliation_summary().unwrap();
    assert!(summary.contains("deploy failed"));
    assert!(!summary.contains("not found")); // corrected failure excluded
}

#[test]
fn install_linear_intent_plan_sets_current_step_identity() {
    let mut state = test_execution_state();
    state.install_linear_intent_plan(
        3,
        vec![
            LinearIntentStep {
                step_id: "plan-v3-step-1".to_string(),
                step_index: 1,
                tool: "http_request".to_string(),
                target: "tweet-1".to_string(),
                description: "Post tweet 1".to_string(),
                tool_calls_on_step: 0,
                completed: false,
                completion_evidence: None,
                last_evaluated_at: None,
            },
            LinearIntentStep {
                step_id: "plan-v3-step-2".to_string(),
                step_index: 2,
                tool: "http_request".to_string(),
                target: "tweet-2".to_string(),
                description: "Post tweet 2".to_string(),
                tool_calls_on_step: 0,
                completed: false,
                completion_evidence: None,
                last_evaluated_at: None,
            },
        ],
    );
    let current = state.current_linear_intent_step().unwrap();
    assert_eq!(current.step_id, "plan-v3-step-1");
    assert_eq!(current.step_index, 1);
}

#[test]
fn advance_linear_intent_step_on_success_moves_forward() {
    let mut state = test_execution_state();
    state.install_linear_intent_plan(
        1,
        vec![
            LinearIntentStep {
                step_id: "plan-v1-step-1".to_string(),
                step_index: 1,
                tool: "http_request".to_string(),
                target: "tweet-1".to_string(),
                description: "Post tweet 1".to_string(),
                tool_calls_on_step: 0,
                completed: false,
                completion_evidence: None,
                last_evaluated_at: None,
            },
            LinearIntentStep {
                step_id: "plan-v1-step-2".to_string(),
                step_index: 2,
                tool: "http_request".to_string(),
                target: "tweet-2".to_string(),
                description: "Post tweet 2".to_string(),
                tool_calls_on_step: 0,
                completed: false,
                completion_evidence: None,
                last_evaluated_at: None,
            },
        ],
    );
    assert!(state.linear_intent_plan_has_remaining_steps());
    // First advance: step 1 → step 2
    state.advance_linear_intent_step_after_external_success();
    assert!(state.linear_intent_plan_has_remaining_steps());
    let current = state.current_linear_intent_step().unwrap();
    assert_eq!(current.step_index, 2);

    // Second advance: step 2 → past end (cursor retires)
    state.advance_linear_intent_step_after_external_success();
    assert!(!state.linear_intent_plan_has_remaining_steps());
    assert!(
        state.current_linear_intent_step().is_none(),
        "cursor should retire past the last step"
    );

    // Further advances are no-ops
    state.advance_linear_intent_step_after_external_success();
    assert!(state.current_linear_intent_step().is_none());
}

#[test]
fn planned_step_reconciliation_groups_retry_under_one_step() {
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: false,
        http_status: Some(403),
        is_external_mutation: true,
        error_summary: Some("duplicate content".to_string()),
        iteration: 1,
        plan_version: Some(1),
        planned_step_id: Some("plan-v1-step-2".to_string()),
        planned_step_index: Some(2),
        planned_step_description: Some("Post tweet 2".to_string()),
        expected_step_count: Some(5),
    });
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: true,
        http_status: Some(201),
        is_external_mutation: true,
        error_summary: None,
        iteration: 2,
        plan_version: Some(1),
        planned_step_id: Some("plan-v1-step-2".to_string()),
        planned_step_index: Some(2),
        planned_step_description: Some("Post tweet 2".to_string()),
        expected_step_count: Some(5),
    });
    let summary = state.build_reconciliation_overview().unwrap().summary;
    assert!(summary.contains("step"));
    assert!(summary.contains("5"));
    assert!(summary.contains("Post tweet 2"));
    assert!(summary.contains("succeeded after 2 attempts"));
}

#[test]
fn planned_step_reconciliation_uses_latest_plan_version_only() {
    let mut state = test_execution_state();
    state.install_linear_intent_plan(
        2,
        vec![
            LinearIntentStep {
                step_id: "plan-v2-step-1".to_string(),
                step_index: 1,
                tool: "http_request".to_string(),
                target: "tweet-1".to_string(),
                description: "Post tweet 1".to_string(),
                tool_calls_on_step: 0,
                completed: false,
                completion_evidence: None,
                last_evaluated_at: None,
            },
            LinearIntentStep {
                step_id: "plan-v2-step-2".to_string(),
                step_index: 2,
                tool: "http_request".to_string(),
                target: "tweet-2".to_string(),
                description: "Post tweet 2".to_string(),
                tool_calls_on_step: 0,
                completed: false,
                completion_evidence: None,
                last_evaluated_at: None,
            },
        ],
    );
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: true,
        http_status: Some(201),
        is_external_mutation: true,
        error_summary: None,
        iteration: 1,
        plan_version: Some(1),
        planned_step_id: Some("plan-v1-step-1".to_string()),
        planned_step_index: Some(1),
        planned_step_description: Some("Old tweet 1".to_string()),
        expected_step_count: Some(3),
    });
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: true,
        http_status: Some(201),
        is_external_mutation: true,
        error_summary: None,
        iteration: 2,
        plan_version: Some(2),
        planned_step_id: Some("plan-v2-step-1".to_string()),
        planned_step_index: Some(1),
        planned_step_description: Some("Post tweet 1".to_string()),
        expected_step_count: Some(2),
    });

    let overview = state.build_reconciliation_overview().unwrap();
    assert_eq!(overview.mode, ReconciliationMode::PlannedStepLevel);
    assert_eq!(overview.total, 2);
    assert_eq!(overview.succeeded, 1);
    assert_eq!(overview.failed, 1);
    assert_eq!(overview.failed_step_indices, vec![2]);
    assert!(!overview.summary.contains("Old tweet 1"));
    assert!(overview
        .summary
        .contains("Step 2 (Post tweet 2) was not completed."));
}

#[test]
fn reconciliation_falls_back_to_attempt_level_without_step_identity() {
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: false,
        http_status: Some(403),
        is_external_mutation: true,
        error_summary: Some("duplicate content".to_string()),
        iteration: 1,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    let summary = state.build_reconciliation_overview().unwrap().summary;
    assert!(summary.contains("attempt"));
}

#[test]
fn promote_budget_for_plan_none_to_standard() {
    let mut state = ExecutionState::new(
        BudgetTier::None,
        default_execution_budget(BudgetTier::None),
        ExecutionPersistence::Ephemeral,
    );
    let original_llm_calls = state.budget.max_llm_calls;
    let original_wall_clock = state.budget.max_wall_clock_ms;
    state.promote_budget_for_plan(4);
    let standard = default_execution_budget(BudgetTier::Standard);
    // None tier has lower llm_calls and wall_clock than Standard
    assert!(state.budget.max_llm_calls >= standard.max_llm_calls);
    assert!(state.budget.max_llm_calls > original_llm_calls);
    assert!(state.budget.max_wall_clock_ms > original_wall_clock);
}

#[test]
fn promote_budget_for_plan_small_to_standard() {
    let mut state = ExecutionState::new(
        BudgetTier::Small,
        default_execution_budget(BudgetTier::Small),
        ExecutionPersistence::Ephemeral,
    );
    state.promote_budget_for_plan(3);
    let standard = default_execution_budget(BudgetTier::Standard);
    assert!(state.budget.max_llm_calls >= standard.max_llm_calls);
}

#[test]
fn no_promote_for_small_plan() {
    let mut state = ExecutionState::new(
        BudgetTier::None,
        default_execution_budget(BudgetTier::None),
        ExecutionPersistence::Ephemeral,
    );
    let original = state.budget.max_tool_calls;
    state.promote_budget_for_plan(2);
    assert_eq!(state.budget.max_tool_calls, original);
}

#[test]
fn no_promote_for_standard_plus() {
    let mut state = ExecutionState::new(
        BudgetTier::Standard,
        default_execution_budget(BudgetTier::Standard),
        ExecutionPersistence::Ephemeral,
    );
    let original = state.budget.max_tool_calls;
    state.promote_budget_for_plan(5);
    assert_eq!(state.budget.max_tool_calls, original);
}

#[test]
fn plan_step_replan_debounce() {
    let mut plan = LinearIntentPlan {
        plan_version: 1,
        steps: vec![LinearIntentStep {
            step_id: "s1".into(),
            step_index: 1,
            tool: String::new(),
            target: String::new(),
            description: "Explore".into(),
            tool_calls_on_step: 0,
            completed: false,
            completion_evidence: None,
            last_evaluated_at: None,
        }],
        current_step_cursor: 0,
    };

    assert!(!plan.current_step_needs_replan());
    plan.record_tool_calls_on_current(1);
    assert!(!plan.current_step_needs_replan());
    plan.record_tool_calls_on_current(1);
    assert!(plan.current_step_needs_replan());

    plan.mark_current_step_evaluated();
    assert!(!plan.current_step_needs_replan());

    plan.record_tool_calls_on_current(1);
    assert!(!plan.current_step_needs_replan());
    plan.record_tool_calls_on_current(1);
    assert!(plan.current_step_needs_replan());
}

#[test]
fn plan_complete_step_advances_cursor() {
    let mut plan = LinearIntentPlan {
        plan_version: 1,
        steps: vec![
            LinearIntentStep {
                step_id: "s1".into(),
                step_index: 1,
                tool: String::new(),
                target: String::new(),
                description: "Explore".into(),
                tool_calls_on_step: 3,
                completed: false,
                completion_evidence: None,
                last_evaluated_at: None,
            },
            LinearIntentStep {
                step_id: "s2".into(),
                step_index: 2,
                tool: String::new(),
                target: String::new(),
                description: "Create".into(),
                tool_calls_on_step: 0,
                completed: false,
                completion_evidence: None,
                last_evaluated_at: None,
            },
        ],
        current_step_cursor: 0,
    };

    plan.complete_current_step_with_evidence("Found 12 posts".into());
    assert_eq!(plan.current_step_cursor, 1);
    assert!(plan.steps[0].completed);
    assert_eq!(
        plan.steps[0].completion_evidence.as_deref(),
        Some("Found 12 posts")
    );
    assert!(!plan.all_steps_complete());

    plan.complete_current_step_with_evidence("Done".into());
    assert!(plan.all_steps_complete());
}

#[test]
fn plan_format_with_progress_shows_markers() {
    let plan = LinearIntentPlan {
        plan_version: 1,
        steps: vec![
            LinearIntentStep {
                step_id: "s1".into(),
                step_index: 1,
                tool: String::new(),
                target: String::new(),
                description: "Explore posts".into(),
                tool_calls_on_step: 3,
                completed: true,
                completion_evidence: Some("Found 12 posts".into()),
                last_evaluated_at: Some(2),
            },
            LinearIntentStep {
                step_id: "s2".into(),
                step_index: 2,
                tool: String::new(),
                target: String::new(),
                description: "Create post 1".into(),
                tool_calls_on_step: 0,
                completed: false,
                completion_evidence: None,
                last_evaluated_at: None,
            },
        ],
        current_step_cursor: 1,
    };

    let formatted = plan.format_with_progress();
    assert!(formatted.contains("[DONE] Explore posts"));
    assert!(formatted.contains("Found 12 posts"));
    assert!(formatted.contains("[CURRENT] Create post 1"));
}

#[test]
fn web_source_tracking_counts_distinct_successful_domains() {
    let mut state = test_execution_state();
    assert!(!state.web_search_used);
    assert!(state.web_source_domains.is_empty());

    state.record_web_source("web_search", r#"{"query":"x"}"#, "1. result...", false);
    assert!(state.web_search_used);
    assert!(
        state.web_source_domains.is_empty(),
        "search is not a read source"
    );

    let page = "x".repeat(600);
    state.record_web_source(
        "web_fetch",
        r#"{"url":"https://en.wikipedia.org/wiki/X"}"#,
        &page,
        false,
    );
    state.record_web_source(
        "web_fetch",
        r#"{"url":"https://en.wikipedia.org/wiki/Y"}"#,
        &page,
        false,
    );
    assert_eq!(state.web_source_domains.len(), 1, "same domain dedups");

    state.record_web_source(
        "browser",
        r#"{"url":"https://espn.com/squad"}"#,
        &page,
        false,
    );
    assert_eq!(state.web_source_domains.len(), 2);
    assert_eq!(state.web_source_urls.len(), 3);
    assert_eq!(
        state.cited_web_source_count(
            "Sources: [X](https://en.wikipedia.org/wiki/X) and https://espn.com/squad"
        ),
        2
    );

    // Failures and junk extractions don't count as read sources.
    state.record_web_source(
        "web_fetch",
        r#"{"url":"https://blocked.example.com/a"}"#,
        "Error fetching: HTTP 403",
        true,
    );
    let junk = format!(
        "Content from x:\n\nhi\n\n[⚠ EXTRACTION FAILED — junk]{}",
        "p".repeat(600)
    );
    state.record_web_source(
        "web_fetch",
        r#"{"url":"https://spa.example.com/b"}"#,
        &junk,
        false,
    );
    state.record_web_source(
        "web_fetch",
        r#"{"url":"https://thin.example.com/c"}"#,
        "tiny",
        false,
    );
    assert_eq!(state.web_source_domains.len(), 2);

    // Non-web tools are ignored entirely.
    state.record_web_source("terminal", r#"{"command":"ls"}"#, &"z".repeat(600), false);
    assert_eq!(state.web_source_domains.len(), 2);
}

#[test]
fn terminal_cross_class_success_corrects_failed_mutation() {
    // Live repro (task 45a65347): a failed `python3 -c` parse classified as
    // mutation; the successful `grep | head` retry classified as observation.
    // Same tool, same goal — the failure must count as corrected.
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "terminal".to_string(),
        success: false,
        http_status: None,
        is_external_mutation: true,
        error_summary: Some("JSONDecodeError: Expecting value".to_string()),
        iteration: 6,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    state.record_outcome(OutcomeEntry {
        tool_name: "terminal".to_string(),
        success: true,
        http_status: None,
        is_external_mutation: false, // grep classified as observation
        error_summary: None,
        iteration: 9,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    assert!(
        state.uncorrected_failed_mutations().is_empty(),
        "terminal cross-class retry success must correct the failure"
    );
}

#[test]
fn http_cross_class_success_does_not_launder_failed_write() {
    // Precisely-classified tools keep the strict rule: a successful GET can
    // never correct a failed POST — that would let a read launder a write.
    let mut state = test_execution_state();
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: false,
        http_status: Some(500),
        is_external_mutation: true,
        error_summary: Some("HTTP 500".to_string()),
        iteration: 3,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    state.record_outcome(OutcomeEntry {
        tool_name: "http_request".to_string(),
        success: true,
        http_status: Some(200),
        is_external_mutation: false, // GET
        error_summary: None,
        iteration: 5,
        plan_version: None,
        planned_step_id: None,
        planned_step_index: None,
        planned_step_description: None,
        expected_step_count: None,
    });
    assert_eq!(
        state.uncorrected_failed_mutations().len(),
        1,
        "a read must never correct a failed write for precisely-classified tools"
    );
}

#[test]
fn natural_tool_presentation_tracks_exact_unrequested_internal_ids() {
    let goal_id = "265636d3-b6e3-424a-839e-daebbf031067";
    let run_id = "34453851-0ad1-4707-b527-736a497196c5";
    let mut state = test_execution_state();
    state.record_tool_output_evidence(&format!("Scheduled goal: {goal_id}"));
    state.record_tool_result_presentation(
        &ToolCallMetadata {
            presentation: Some(ToolResultPresentation::NaturalSummary),
            internal_identifiers: vec![run_id.to_string()],
            ..ToolCallMetadata::default()
        },
        &format!("Queued run {run_id}"),
    );

    assert!(state.natural_outcome_summary_required);
    assert_eq!(
        state.unrequested_internal_identifiers(
            &format!("Started it. Goal {goal_id}; run {run_id}."),
            "Start the existing daily blog run."
        ),
        vec![goal_id.to_string(), run_id.to_string()]
    );
    assert_eq!(
        state.unrequested_internal_identifiers(
            &format!("Run {run_id} is queued."),
            &format!("Start run {run_id} and show its diagnostics.")
        ),
        Vec::<String>::new()
    );
}

#[test]
fn explicit_diagnostic_presentation_disables_natural_summary_gate() {
    let id = "34453851-0ad1-4707-b527-736a497196c5";
    let mut state = test_execution_state();
    state.record_tool_result_presentation(
        &ToolCallMetadata {
            presentation: Some(ToolResultPresentation::NaturalSummary),
            internal_identifiers: vec![id.to_string()],
            ..ToolCallMetadata::default()
        },
        "queued",
    );
    state.record_tool_result_presentation(
        &ToolCallMetadata {
            presentation: Some(ToolResultPresentation::DiagnosticDetail),
            ..ToolCallMetadata::default()
        },
        "diagnostics requested",
    );

    assert!(!state.natural_outcome_summary_required);
    assert!(state
        .unrequested_internal_identifiers(&format!("Run {id}"), "show diagnostics")
        .is_empty());
}

#[test]
fn nonrecoverable_failure_remains_open_until_a_later_satisfying_result() {
    let mut state = test_execution_state();
    state.complete_current_step(StepExecutionOutcome::NonrecoverableFailure);
    assert!(state.has_unresolved_nonrecoverable_failure());

    state.completed_operation_results = 1;
    assert!(!state.has_unresolved_nonrecoverable_failure());
}
