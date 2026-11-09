//! The daemon as an `aidaemon-eval` [`Subject`]: runs a scenario through
//! `Agent::handle_message` with a strict scripted provider and checks the
//! daemon's own snapshot assertions.

use std::collections::BTreeMap;
use std::fmt::Display;

use aidaemon_core::ReceiptStatus;
use aidaemon_eval::{CallRecord, FactoryError, Scenario, ScriptedReply, Subject, Trajectory};
use anyhow::Context;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::events::{Event, EventStore};
use crate::harness_eval::fixture::{collect_run_result, FixtureSeed, HarnessEvalRunResult};
use crate::state::sqlite::SqliteStateStore;
use crate::testing::{
    setup_test_agent, setup_test_agent_orchestrator, setup_test_agent_with_models, MockProvider,
};
use crate::traits::{Goal, GoalStore, ProviderResponse, ToolInvocationStage, ToolOutcomeStatus};
use crate::types::{ChannelContext, UserRole};

pub type DaemonScenario = Scenario<DaemonSetup, DaemonExpect>;

fn default_owner() -> String {
    "owner".to_string()
}

fn is_owner(role: &str) -> bool {
    role == "owner"
}

fn is_empty_seed(seed: &FixtureSeed) -> bool {
    seed.goals.is_empty()
}

fn is_false(value: &bool) -> bool {
    !value
}

/// The state a daemon scenario starts from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonSetup {
    /// Defaults to one derived from the scenario name and run number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default = "default_owner", skip_serializing_if = "is_owner_ref")]
    pub user_role: String,
    /// Use the orchestrator-mode test harness (orchestration routing).
    #[serde(default, skip_serializing_if = "is_false")]
    pub orchestrator: bool,
    /// Use non-uniform primary/smart model tiers (routing tests).
    #[serde(default, skip_serializing_if = "is_false")]
    pub routing_models: bool,
    /// Session mode set before the turn, as `/readonly` or `/mode` would
    /// (`"readonly"` or `"unrestricted"`). Unset leaves the owner default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_mode: Option<String>,
    /// DB seed data applied before `handle_message`.
    #[serde(default, skip_serializing_if = "is_empty_seed")]
    pub seed: FixtureSeed,
}

fn is_owner_ref(role: &String) -> bool {
    is_owner(role)
}

impl Default for DaemonSetup {
    fn default() -> Self {
        Self {
            session_id: None,
            user_role: default_owner(),
            orchestrator: false,
            routing_models: false,
            session_mode: None,
            seed: FixtureSeed::default(),
        }
    }
}

/// Assertions only the daemon can answer: its `TaskEnd` record and
/// harness-eval snapshot.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonExpect {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orchestration_route: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_required_predicted: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Every LLM call, auxiliary ones included.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_calls_min: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_calls_max: Option<u32>,
    /// Main-loop calls only. The upper bound is the crate's `max_turns`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_llm_calls_min: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls_min: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls_max: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weighted_tokens_max: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forbidden_mutation_attempts_max: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forbids_mutation: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_steps_completed_min: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_accuracy_min: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_yield_min: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_fulfillment_min: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_efficiency_min: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overall_min: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direct_return: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_fallthrough: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guard_fired: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decision_types_seen: Vec<String>,
}

/// Runs a scenario against a fresh test daemon. A scenario with no `script`
/// runs with an empty one, so any model call fails it.
pub struct DaemonSubject;

#[async_trait]
impl Subject for DaemonSubject {
    type Setup = DaemonSetup;
    type ExpectExt = DaemonExpect;
    type Observed = HarnessEvalRunResult;

    async fn run(
        &self,
        scenario: &DaemonScenario,
        run: usize,
    ) -> Result<Trajectory<HarnessEvalRunResult>, FactoryError> {
        run_with_trace(scenario, run)
            .await
            .map(|(result, _events)| trajectory_of(result))
            .map_err(|err| format!("{err:#}").into())
    }

    fn check_ext(
        &self,
        expect: &DaemonExpect,
        trajectory: &Trajectory<HarnessEvalRunResult>,
    ) -> Vec<String> {
        check_daemon_expect(expect, &trajectory.observed)
    }
}

/// Run a scenario and also return the whole session trace in append order.
pub async fn run_with_trace(
    scenario: &DaemonScenario,
    run: usize,
) -> anyhow::Result<(HarnessEvalRunResult, Vec<Event>)> {
    let setup = scenario.setup.clone().unwrap_or_default();
    let session_id = setup
        .session_id
        .clone()
        .unwrap_or_else(|| format!("eval-{}-{run}", scenario.name));
    let responses = scenario
        .script
        .iter()
        .flatten()
        .map(mock_response)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let provider = MockProvider::with_responses(responses).with_strict_responses();

    let harness = if setup.orchestrator {
        setup_test_agent_orchestrator(provider).await?
    } else if setup.routing_models {
        setup_test_agent_with_models(provider, "primary-model", "smart-model").await?
    } else {
        setup_test_agent(provider).await?
    };

    apply_seed(&harness.state, &session_id, &setup.seed).await?;

    if let Some(raw) = &setup.session_mode {
        crate::agent::authority::set_session_mode(
            harness.state.as_ref(),
            &session_id,
            parse_session_mode(raw)?,
            chrono::Utc::now(),
        )
        .await?;
    }

    let user_role = parse_user_role(&setup.user_role)?;
    let response = harness
        .agent
        .handle_message(
            &session_id,
            &scenario.input,
            None,
            user_role,
            ChannelContext::private("test"),
            None,
        )
        .await;
    harness
        .provider
        .assert_response_script_not_exhausted()
        .await?;
    let response = response?;
    harness.provider.assert_response_script_consumed().await?;

    let event_store = EventStore::new(harness.state.pool()).await?;
    let result = collect_result(&event_store, &session_id, &response).await?;
    let mut session_events = event_store
        .query_recent_events(&session_id, usize::MAX >> 1)
        .await?;
    session_events.sort_by_key(|event| event.id);
    Ok((result, session_events))
}

pub(super) async fn collect_result(
    event_store: &EventStore,
    session_id: &str,
    response: &str,
) -> anyhow::Result<HarnessEvalRunResult> {
    // Locate the terminal task, then fetch its entire scoped trace. A recency
    // window silently drops early calls from long runs and invalidates counts.
    let ends = event_store
        .query_recent_task_ends(session_id, false, 1)
        .await?;
    let end = ends.first().context("missing TaskEnd event")?;
    let task_id = end
        .task_id
        .as_deref()
        .context("TaskEnd missing task identity")?;
    let events = event_store
        .query_task_events_for_session(session_id, task_id)
        .await?;
    collect_run_result(&events, response)
}

/// The mock serves one tool call per response.
fn mock_response(reply: &ScriptedReply) -> anyhow::Result<ProviderResponse> {
    let calls = reply.tool_calls.as_deref().unwrap_or_default();
    match (reply.text.as_deref(), calls) {
        (Some(text), []) => Ok(MockProvider::text_response(text)),
        (None, [call]) => Ok(MockProvider::tool_call_response(
            &call.name,
            &call.arguments.to_json_string(),
        )),
        _ => anyhow::bail!("a daemon script reply is one text or one tool call"),
    }
}

async fn apply_seed(
    state: &SqliteStateStore,
    session_id: &str,
    seed: &FixtureSeed,
) -> anyhow::Result<()> {
    for goal_spec in &seed.goals {
        let mut goal = Goal::new_finite(&goal_spec.description, session_id);
        goal.status = goal_spec.status.clone();
        state.create_goal(&goal).await?;
    }
    Ok(())
}

/// Unknown modes are rejected for the same reason as unknown roles.
fn parse_session_mode(raw: &str) -> anyhow::Result<crate::agent::authority::SessionMode> {
    use crate::agent::authority::SessionMode;
    match raw.to_ascii_lowercase().as_str() {
        "readonly" => Ok(SessionMode::ReadOnly),
        "unrestricted" => Ok(SessionMode::Unrestricted),
        other => anyhow::bail!(
            "unknown scenario session_mode {other:?} (expected readonly or unrestricted)"
        ),
    }
}

/// Unknown roles are rejected: silently running a typo'd role as Owner would
/// evaluate the most privileged path under a scenario meant to test a lesser one.
fn parse_user_role(raw: &str) -> anyhow::Result<UserRole> {
    match raw.to_ascii_lowercase().as_str() {
        "owner" => Ok(UserRole::Owner),
        "guest" => Ok(UserRole::Guest),
        "public" => Ok(UserRole::Public),
        other => {
            anyhow::bail!("unknown scenario user_role {other:?} (expected owner, guest, or public)")
        }
    }
}

/// The neutral record of a daemon run. Proposals come from `ToolCall` events
/// and receipts from `ToolResult` events; the two can differ.
pub fn trajectory_of(result: HarnessEvalRunResult) -> Trajectory<HarnessEvalRunResult> {
    let outcome = result.task_end.effective_outcome().as_str();
    Trajectory {
        status: result.task_end.status.as_str().to_string(),
        status_detail: format!("outcome {outcome}"),
        succeeded: outcome == crate::events::TaskOutcome::Succeeded.as_str(),
        unmet: Vec::new(),
        final_text: Some(result.response_text.clone()),
        proposed: result.tool_names.clone(),
        receipts: result
            .tool_results
            .iter()
            .map(|(tool, receipt)| CallRecord {
                tool: tool.clone(),
                status: receipt
                    .as_ref()
                    .and_then(|r| receipt_status(r.outcome_status)),
                dispatched: receipt
                    .as_ref()
                    .is_some_and(|r| r.invocation_stage == ToolInvocationStage::Dispatched),
            })
            .collect(),
        turns: result.execution_llm_calls,
        billable_tokens: result.harness_eval.cost.total_input_tokens
            + result.harness_eval.cost.total_output_tokens,
        run_failures: Vec::new(),
        observed: result,
    }
}

/// `None` for the daemon's outcomes the kernel has no status for.
fn receipt_status(status: ToolOutcomeStatus) -> Option<ReceiptStatus> {
    match status {
        ToolOutcomeStatus::Succeeded => Some(ReceiptStatus::Succeeded),
        ToolOutcomeStatus::FailedRetryable | ToolOutcomeStatus::FailedPermanent => {
            Some(ReceiptStatus::Failed)
        }
        ToolOutcomeStatus::Blocked => Some(ReceiptStatus::Denied),
        ToolOutcomeStatus::CompletedWithNegativeResult | ToolOutcomeStatus::Backgrounded => None,
    }
}

fn check_eq<T: PartialEq + Display>(
    failures: &mut Vec<String>,
    label: &str,
    expected: Option<&T>,
    actual: &T,
) {
    if let Some(expected) = expected.filter(|expected| *expected != actual) {
        failures.push(format!("{label}: expected {expected}, got {actual}"));
    }
}

fn check_min<T: PartialOrd + Display>(
    failures: &mut Vec<String>,
    label: &str,
    min: Option<T>,
    actual: T,
) {
    if let Some(min) = min.filter(|min| actual < *min) {
        failures.push(format!("{label}: expected >= {min}, got {actual}"));
    }
}

fn check_max<T: PartialOrd + Display>(
    failures: &mut Vec<String>,
    label: &str,
    max: Option<T>,
    actual: T,
) {
    if let Some(max) = max.filter(|max| actual > *max) {
        failures.push(format!("{label}: expected <= {max}, got {actual}"));
    }
}

fn check_score(failures: &mut Vec<String>, label: &str, min: Option<f32>, actual: f32) {
    if let Some(min) = min.filter(|min| actual + f32::EPSILON < *min) {
        failures.push(format!("{label}: expected >= {min:.2}, got {actual:.2}"));
    }
}

/// Check the daemon's snapshot assertions. Returns every failure.
pub fn check_daemon_expect(expect: &DaemonExpect, result: &HarnessEvalRunResult) -> Vec<String> {
    let eval = &result.harness_eval;
    let routing = &eval.routing;
    let contract = &eval.quality.contract;
    let scores = &eval.scores;
    let tool_calls = result.task_end.tool_calls_count;
    let f = &mut Vec::new();

    check_eq(
        f,
        "orchestration_route",
        expect.orchestration_route.as_ref(),
        &eval.orchestration_route,
    );
    check_eq(
        f,
        "tools_required_predicted",
        expect.tools_required_predicted.as_ref(),
        &routing.tools_required_predicted,
    );
    check_eq(
        f,
        "outcome",
        expect.outcome.as_ref(),
        &result.task_end.effective_outcome().as_str().to_string(),
    );
    check_eq(
        f,
        "stop_reason",
        expect.stop_reason.as_ref(),
        &eval.quality.stop_reason,
    );
    check_min(f, "llm_calls_min", expect.llm_calls_min, result.llm_calls);
    check_max(f, "llm_calls_max", expect.llm_calls_max, result.llm_calls);
    check_min(
        f,
        "execution_llm_calls_min",
        expect.execution_llm_calls_min,
        result.execution_llm_calls,
    );
    check_min(f, "tool_calls_min", expect.tool_calls_min, tool_calls);
    check_max(f, "tool_calls_max", expect.tool_calls_max, tool_calls);
    check_max(
        f,
        "weighted_tokens_max",
        expect.weighted_tokens_max,
        eval.cost.weighted_tokens,
    );
    check_eq(
        f,
        "forbids_mutation",
        expect.forbids_mutation.as_ref(),
        &contract.forbids_mutation,
    );
    check_max(
        f,
        "forbidden_mutation_attempts_max",
        expect.forbidden_mutation_attempts_max,
        contract.forbidden_mutation_attempts,
    );
    check_min(
        f,
        "plan_steps_completed_min",
        expect.plan_steps_completed_min,
        eval.progress.plan_steps_completed.unwrap_or(0),
    );
    check_score(
        f,
        "routing_accuracy_min",
        expect.routing_accuracy_min,
        scores.routing_accuracy,
    );
    check_score(
        f,
        "progress_yield_min",
        expect.progress_yield_min,
        scores.progress_yield,
    );
    check_score(
        f,
        "contract_fulfillment_min",
        expect.contract_fulfillment_min,
        scores.contract_fulfillment,
    );
    check_score(
        f,
        "cost_efficiency_min",
        expect.cost_efficiency_min,
        scores.cost_efficiency,
    );
    check_score(f, "overall_min", expect.overall_min, scores.overall);
    check_eq(
        f,
        "direct_return",
        expect.direct_return.as_ref(),
        &routing.direct_return_attempted,
    );
    check_eq(
        f,
        "response_fallthrough",
        expect.response_fallthrough.as_ref(),
        &routing.response_fallthrough,
    );

    for guard in &expect.guard_fired {
        let fired = match guard.as_str() {
            "RepetitiveCallDetection" => eval.progress.repetition_guard_fires > 0,
            "Stall" | "stall" => eval.progress.stall_guard_fires > 0,
            other => {
                f.push(format!("guard_fired: unknown guard {other}"));
                continue;
            }
        };
        if !fired {
            f.push(format!("guard_fired: expected {guard} to fire"));
        }
    }

    for decision_type in &expect.decision_types_seen {
        if !result.decision_types.contains(decision_type) {
            f.push(format!(
                "decision_types_seen: expected {decision_type}, got {:?}",
                result.decision_types
            ));
        }
    }

    std::mem::take(f)
}

/// Tool name counts, for building expectations in tests.
pub fn count_names(names: &[String]) -> BTreeMap<String, u32> {
    let mut counts = BTreeMap::new();
    for name in names {
        *counts.entry(name.clone()).or_insert(0) += 1;
    }
    counts
}
