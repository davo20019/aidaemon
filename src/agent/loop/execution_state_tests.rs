//! Tests for `execution_state` (split out from the original module via `#[path]`).
//!
//! Moved verbatim — no logic changes. Included as a child test module of
//! `execution_state` so `use super::*;` continues to resolve against it.

use super::*;
use crate::agent::{CompletionContract, CompletionTaskKind, FollowupMode, TurnContext};

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
