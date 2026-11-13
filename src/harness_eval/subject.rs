//! The daemon as an `aidaemon-eval` [`Subject`]: runs a scenario through
//! `Agent::handle_message` with a strict scripted provider and checks the
//! daemon's own snapshot assertions.

use crate::events::EventStoreForTests as _;
use std::fmt::Display;
use std::path::PathBuf;

use aidaemon_core::ReceiptStatus;
use aidaemon_eval::{CallRecord, FactoryError, Scenario, ScriptedReply, Subject, Trajectory};
use anyhow::Context;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::events::{Event, EventStore};
use crate::harness_eval::recorder::{collect_run_result, HarnessEvalRunResult};
use crate::state::SqliteStateStore;
use crate::testing::{
    setup_test_agent, setup_test_agent_orchestrator, setup_test_agent_with_models, MockProvider,
};
use crate::traits::{Goal, GoalStore, ProviderResponse, ToolInvocationStage, ToolOutcomeStatus};
use crate::types::{ChannelContext, UserRole};

pub type DaemonScenario = Scenario<DaemonSetup, DaemonExpect>;

/// The suite the daemon's eval run loads.
pub fn scenarios_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/harness_eval/scenarios")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ScenarioSeed {
    #[serde(default)]
    pub goals: Vec<SeedGoal>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedGoal {
    pub description: String,
    #[serde(default = "default_active_status")]
    pub status: String,
}

fn default_active_status() -> String {
    "active".to_string()
}

fn default_owner() -> String {
    "owner".to_string()
}

fn is_owner(role: &str) -> bool {
    role == "owner"
}

fn is_empty_seed(seed: &ScenarioSeed) -> bool {
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
    #[serde(default = "default_owner", skip_serializing_if = "is_owner")]
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
    pub seed: ScenarioSeed,
}

impl Default for DaemonSetup {
    fn default() -> Self {
        Self {
            session_id: None,
            user_role: default_owner(),
            orchestrator: false,
            routing_models: false,
            session_mode: None,
            seed: ScenarioSeed::default(),
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
    seed: &ScenarioSeed,
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

#[cfg(test)]
mod tests {
    use aidaemon_eval::{check_trajectory, CaseResult, Suite};

    use super::*;
    use crate::harness_eval::recorder::record_scenario_from_events;

    fn parse(yaml: &str) -> DaemonScenario {
        DaemonScenario::parse_yaml(yaml, "inline.yaml").unwrap()
    }

    async fn run(scenario: &DaemonScenario) -> CaseResult {
        scenario.run_on(&DaemonSubject, 0).await
    }

    fn suite() -> Vec<DaemonScenario> {
        Suite::<DaemonSetup, DaemonExpect>::load_dir(scenarios_dir())
            .unwrap()
            .scenarios
    }

    fn failed(result: &CaseResult, label: &str) -> bool {
        !result.passed && result.failures.iter().any(|f| f.contains(label))
    }

    #[tokio::test]
    async fn harness_eval_scenario_suite() {
        let scenarios = suite();
        assert!(
            scenarios.len() >= 17,
            "expected at least 17 scenarios, got {}",
            scenarios.len()
        );
        let mut failures = Vec::new();
        for scenario in &scenarios {
            let result = run(scenario).await;
            if !result.passed {
                failures.push(format!("{}: {:?}", scenario.name, result.failures));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn typos_in_daemon_blocks_fail_at_load() {
        for yaml in [
            "name: typo\ninput: Hello\nexpect:\n  ext:\n    tool_calls_mx: 0\n",
            "name: typo\ninput: Hello\nsetup:\n  session_idd: s1\n",
            "name: typo\ninput: Hello\nmock_responses:\n  - text: Hi\n",
        ] {
            assert!(
                DaemonScenario::parse_yaml(yaml, "inline.yaml").is_err(),
                "{yaml}"
            );
        }
    }

    #[tokio::test]
    async fn harness_eval_checks_false_fallthrough_expectation() {
        let scenario = parse(
            "name: false_fallthrough\ninput: Hello\nsetup:\n  session_id: synthetic-fallthrough\nscript:\n  - text: Hello\nexpect:\n  ext:\n    response_fallthrough: false\n",
        );
        let mut trajectory = DaemonSubject.run(&scenario, 0).await.unwrap();
        let ext = scenario.expect.ext.as_ref().unwrap();
        trajectory
            .observed
            .harness_eval
            .routing
            .response_fallthrough = false;
        assert!(DaemonSubject.check_ext(ext, &trajectory).is_empty());
        trajectory
            .observed
            .harness_eval
            .routing
            .response_fallthrough = true;
        assert!(!DaemonSubject.check_ext(ext, &trajectory).is_empty());
    }

    #[tokio::test]
    async fn harness_eval_rejects_exhausted_response_script() {
        let scenario = parse("name: empty_script\ninput: Hello\n");
        let error = DaemonSubject
            .run(&scenario, 0)
            .await
            .expect_err("unexpected model calls must fail the scenario");
        assert!(error.to_string().contains("script exhausted"), "{error}");
    }

    /// The recorder scopes to the right task and user message for every suite
    /// scenario, writes YAML the eval crate accepts, and rebuilds the exact
    /// script when no gate intervened. Runs where gates bounce or synthesize
    /// replies are known to diverge (see `record_scenario_from_events`), so
    /// they are not asserted here.
    #[tokio::test]
    async fn recorder_writes_scenarios_that_load_and_replay() {
        let session = |s: &DaemonScenario| s.setup.as_ref().unwrap().session_id.clone().unwrap();
        for scenario in suite() {
            let (_, events) = run_with_trace(&scenario, 0).await.unwrap();
            let recorded = record_scenario_from_events(&session(&scenario), &events, None).unwrap();
            assert_eq!(recorded.input, scenario.input, "{}", scenario.name);
            let yaml = serde_saphyr::to_string(&recorded).unwrap();
            DaemonScenario::parse_yaml(&yaml, "recorded.yaml").unwrap();
        }
        for name in ["mock_tool_then_reply", "mutation_called_once_in_order"] {
            let path = scenarios_dir().join(format!("{name}.yaml"));
            let scenario = DaemonScenario::parse_file(&path).unwrap();
            let (result, events) = run_with_trace(&scenario, 0).await.unwrap();
            let recorded = record_scenario_from_events(
                &session(&scenario),
                &events,
                Some(&result.task_end.task_id),
            )
            .unwrap();
            let yaml = serde_saphyr::to_string(&recorded).unwrap();
            let mut replay = DaemonScenario::parse_yaml(&yaml, "recorded.yaml").unwrap();
            assert_eq!(replay.script, scenario.script, "{name}");
            // The draft's own expectations must hold on the run it was
            // recorded from (floored scores, receipt-backed counts).
            let original = scenario.setup.as_ref().unwrap();
            let setup = replay.setup.as_mut().unwrap();
            setup.orchestrator = original.orchestrator;
            setup.routing_models = original.routing_models;
            let result = run(&replay).await;
            assert!(result.passed, "{name}: {:?}", result.failures);
        }
    }

    #[tokio::test]
    async fn harness_eval_rejects_unconsumed_response_script() {
        let scenario = parse(
            "name: leftover_script\ninput: Hello\nsetup:\n  session_id: synthetic-leftover-script\nscript:\n  - text: Hello.\n  - text: Never requested.\n",
        );
        let error = DaemonSubject
            .run(&scenario, 0)
            .await
            .expect_err("a run that stops before its script ends must fail");
        assert!(error.to_string().contains("1 execution"), "{error}");
    }

    #[tokio::test]
    async fn harness_eval_rejects_unknown_user_role() {
        let scenario = parse(
            "name: bad_role\ninput: Hello\nsetup:\n  session_id: synthetic-bad-role\n  user_role: gest\nscript:\n  - text: Hello.\n",
        );
        let error = DaemonSubject
            .run(&scenario, 0)
            .await
            .expect_err("a typo'd role must not silently run as owner");
        assert!(error.to_string().contains("gest"), "{error}");
    }

    const REPEATED_SYSTEM_INFO: &str = r#"
name: dispatch_counts
input: Check system info
setup:
  session_id: eval_dispatch_counts_01
script:
  - tool_calls:
      - name: system_info
        arguments: "{}"
  - tool_calls:
      - name: system_info
        arguments: "{}"
  - text: Done.
"#;

    /// Proposals and dispatches diverge when the loop replays or suppresses a
    /// repeated call; each expectation must read its own lane.
    #[tokio::test]
    async fn harness_eval_dispatch_counts_read_receipts_not_proposals() {
        let mut scenario = parse(REPEATED_SYSTEM_INFO);
        scenario
            .expect
            .proposed
            .tool_call_counts
            .insert("system_info".into(), 2);
        let trajectory = DaemonSubject.run(&scenario, 0).await.unwrap();
        assert!(check_trajectory(&scenario.expect, &trajectory).is_empty());
        let dispatched = &trajectory.observed.dispatched_tool_names;
        assert!(
            !dispatched.is_empty() && dispatched.len() <= trajectory.proposed.len(),
            "dispatches must be a receipt-backed subset of proposals: proposed {:?}, dispatched {dispatched:?}",
            trajectory.proposed,
        );
        scenario
            .expect
            .tool_dispatch_counts
            .insert("system_info".into(), 3);
        let failures = check_trajectory(&scenario.expect, &trajectory);
        assert!(failures.iter().any(|f| f.contains("tool_dispatch_counts")));
    }

    #[tokio::test]
    async fn harness_eval_collects_calls_before_the_recent_event_window() {
        use serde_json::json;

        use crate::events::EventType;
        let session_id = "synthetic-long-trace";
        let scenario = parse(&format!(
            "name: long_trace\ninput: Hello\nsetup:\n  session_id: {session_id}\nscript:\n  - text: Hello\n"
        ));
        let (baseline, _) = run_with_trace(&scenario, 0).await.unwrap();
        let db = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", db.path().display()))
            .await
            .unwrap();
        let store = EventStore::new(pool).await.unwrap();
        let task_id = &baseline.task_end.task_id;
        for index in 0..2 {
            store
                .append(Event::new(
                    session_id,
                    EventType::ToolCall,
                    json!({
                        "task_id": task_id, "tool_call_id": format!("call-{index}"),
                        "name": "remember_fact", "arguments": {}
                    }),
                ))
                .await
                .unwrap();
        }
        store
            .append(Event::new(
                session_id,
                EventType::LlmCall,
                json!({"task_id": task_id, "model": "mock-model", "latency_ms": 0}),
            ))
            .await
            .unwrap();
        for _ in 0..220 {
            store
                .append(Event::new(
                    session_id,
                    EventType::UserMessage,
                    json!({
                        "task_id": task_id, "content": "synthetic progress"
                    }),
                ))
                .await
                .unwrap();
        }
        store
            .append(Event::new(
                session_id,
                EventType::TaskEnd,
                serde_json::to_value(&baseline.task_end).unwrap(),
            ))
            .await
            .unwrap();
        let result = collect_result(&store, session_id, "Hello").await.unwrap();
        assert_eq!(result.tool_names, vec!["remember_fact", "remember_fact"]);
        assert_eq!(result.llm_calls, 1);
        let trajectory = trajectory_of(result);
        let mut expect = scenario.expect.clone();
        expect
            .proposed
            .tool_call_counts
            .insert("remember_fact".into(), 1);
        assert!(!check_trajectory(&expect, &trajectory).is_empty());
        expect.proposed.tool_call_counts.clear();
        expect.proposed.tools_not_used.push("remember_fact".into());
        assert!(!check_trajectory(&expect, &trajectory).is_empty());
    }

    const TWO_TOOLS: &str = r#"
name: order
input: Check system info then remember it
setup:
  session_id: eval_order_01
script:
  - tool_calls:
      - name: system_info
        arguments: "{}"
  - tool_calls:
      - name: remember_fact
        arguments: '{"category":"project","key":"host","value":"synthetic"}'
  - text: Done.
"#;

    /// Two distinct tools, asserted in the order they were actually called.
    #[tokio::test]
    async fn harness_eval_enforces_tool_order_end_to_end() {
        let mut scenario = parse(TWO_TOOLS);
        scenario.expect.proposed.tools_in_order =
            vec!["system_info".into(), "remember_fact".into()];
        for tool in ["system_info", "remember_fact"] {
            scenario
                .expect
                .proposed
                .tool_call_counts
                .insert(tool.into(), 1);
        }
        let result = run(&scenario).await;
        assert!(result.passed, "{:?}", result.failures);
    }

    /// Same run, reversed expectation: must be rejected.
    #[tokio::test]
    async fn harness_eval_rejects_wrong_tool_order_end_to_end() {
        let mut scenario = parse(TWO_TOOLS);
        scenario.expect.proposed.tools_in_order =
            vec!["remember_fact".into(), "system_info".into()];
        assert!(failed(&run(&scenario).await, "tools_in_order"));
    }

    /// The duplicate-side-effect guard: same tool invoked twice, count says once.
    #[tokio::test]
    async fn harness_eval_rejects_duplicate_tool_call_end_to_end() {
        let mut scenario = parse(REPEATED_SYSTEM_INFO);
        scenario
            .expect
            .proposed
            .tool_call_counts
            .insert("system_info".into(), 1);
        assert!(failed(&run(&scenario).await, "tool_call_counts"));
    }

    #[tokio::test]
    async fn harness_eval_basic_conversational_scenario() {
        let scenario = parse(
            "name: basic_conversational\ninput: Hello there\nsetup:\n  session_id: eval_basic_01\nscript:\n  - text: Hello there.\nexpect:\n  ext:\n    outcome: succeeded\n    llm_calls_min: 1\n",
        );
        let result = run(&scenario).await;
        assert!(result.passed, "{:?}", result.failures);
    }
}
