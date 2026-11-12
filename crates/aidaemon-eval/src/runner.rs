//! Running one scenario and checking its expectations.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aidaemon_core::testing::ScriptedProvider;
use aidaemon_core::{Agent, InvocationStage, Provider, ReceiptStatus, RunOutcome};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::scenario::{status_name, Expect, NoExtension, Scenario};

/// Error type factories may return.
pub type FactoryError = Box<dyn std::error::Error + Send + Sync>;

/// Builds the agent under test for one run of a scenario.
///
/// The runner hands `build` the provider to use: a fresh
/// [`ScriptedProvider`] for scripted scenarios, or whatever
/// [`AgentFactory::live_provider`] returns for the rest. Call
/// [`Scenario::configure`] on your builder so the scenario's instructions and
/// limits apply.
pub trait AgentFactory: Send + Sync {
    fn build(
        &self,
        scenario: &Scenario,
        provider: Arc<dyn Provider>,
    ) -> Result<Agent, FactoryError>;

    /// Provider for scenarios without a `script`. Called once per run, so it
    /// may hand out a fresh client each time or share one. `None` (the
    /// default) makes unscripted scenarios fail with a clear message.
    fn live_provider(&self, _scenario: &Scenario) -> Option<Arc<dyn Provider>> {
        None
    }
}

/// An [`AgentFactory`] made from a closure, with an optional live provider.
pub struct FnFactory<F> {
    build: F,
    live: Option<Arc<dyn Provider>>,
}

impl<F> FnFactory<F>
where
    F: Fn(&Scenario, Arc<dyn Provider>) -> Result<Agent, FactoryError> + Send + Sync,
{
    pub fn new(build: F) -> Self {
        Self { build, live: None }
    }

    /// Use `provider` for every unscripted scenario.
    pub fn with_live_provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.live = Some(provider);
        self
    }
}

impl<F> AgentFactory for FnFactory<F>
where
    F: Fn(&Scenario, Arc<dyn Provider>) -> Result<Agent, FactoryError> + Send + Sync,
{
    fn build(
        &self,
        scenario: &Scenario,
        provider: Arc<dyn Provider>,
    ) -> Result<Agent, FactoryError> {
        (self.build)(scenario, provider)
    }

    fn live_provider(&self, _scenario: &Scenario) -> Option<Arc<dyn Provider>> {
        self.live.clone()
    }
}

/// The result of one run of one scenario.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseResult {
    pub scenario: String,
    pub passed: bool,
    /// Every failed expectation, in a stable order.
    pub failures: Vec<String>,
    /// snake_case run status, or `error` if the run never produced one.
    pub status: String,
    pub final_text: Option<String>,
    /// Tool names in receipt order.
    pub tools: Vec<String>,
    pub turns: u32,
    pub billable_tokens: u64,
    pub duration_ms: f64,
}

/// One tool call that got a receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallRecord {
    pub tool: String,
    /// `None` when the subject's receipts do not say.
    pub status: Option<ReceiptStatus>,
    /// The tool's code was invoked. A call refused or rejected at the
    /// boundary has a receipt but was not dispatched.
    pub dispatched: bool,
}

/// What a run did, in terms any subject can report: the neutral record the
/// expectations are checked against.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory<O = ()> {
    /// snake_case name of how the run ended. For kernel runs this is one of
    /// the [`crate::ExpectedStatus`] names.
    pub status: String,
    /// More about `status`, shown when an expectation on it fails.
    pub status_detail: String,
    /// Whether the run's objective was met.
    pub succeeded: bool,
    /// Requirements still open when the run stopped.
    pub unmet: Vec<String>,
    pub final_text: Option<String>,
    /// Tool names the model proposed, in order. A subject whose loop suppresses
    /// or replays calls can have more proposals than receipts.
    pub proposed: Vec<String>,
    /// Every call that got a receipt, in receipt order.
    pub receipts: Vec<CallRecord>,
    pub turns: u32,
    pub billable_tokens: u64,
    /// Problems with the run itself rather than with the agent's behavior,
    /// for example a script that was not fully consumed. Any entry fails
    /// the case.
    pub run_failures: Vec<String>,
    /// Whatever else the subject observed, for [`Subject::check_ext`].
    pub observed: O,
}

impl Trajectory {
    /// The trajectory of a finished kernel run.
    pub fn from_outcome(outcome: &RunOutcome) -> Self {
        Self {
            status: status_name(&outcome.status).to_string(),
            status_detail: format!("{:?}", outcome.status),
            succeeded: outcome.is_succeeded(),
            unmet: outcome.unmet().to_vec(),
            final_text: outcome.final_text.clone(),
            proposed: outcome
                .state
                .messages
                .iter()
                .flat_map(|message| &message.tool_calls)
                .map(|call| call.name.clone())
                .collect(),
            receipts: outcome
                .receipts()
                .iter()
                .map(|r| CallRecord {
                    tool: r.tool.clone(),
                    status: Some(r.status),
                    dispatched: r.stage == InvocationStage::Dispatched,
                })
                .collect(),
            turns: outcome.turns(),
            billable_tokens: outcome.usage().billable_tokens(),
            run_failures: Vec::new(),
            observed: (),
        }
    }
}

/// The system under test: runs a scenario once and reports a [`Trajectory`].
///
/// [`AgentFactory`] covers agents built on the kernel. Implement this
/// directly for anything else, such as an application that drives its own
/// turn loop around the kernel.
#[async_trait]
pub trait Subject: Send + Sync {
    /// The scenario's `setup:` block. Use [`NoExtension`] for none.
    type Setup: Send + Sync + 'static;
    /// The scenario's `expect.ext:` block. Use [`NoExtension`] for none.
    type ExpectExt: Send + Sync + 'static;
    /// What the subject reports beyond the neutral fields.
    type Observed: Send;

    /// `run` is the run number within the suite, for per-run session ids.
    async fn run(
        &self,
        scenario: &Scenario<Self::Setup, Self::ExpectExt>,
        run: usize,
    ) -> Result<Trajectory<Self::Observed>, FactoryError>;

    /// Check the scenario's `expect.ext:` block. Returns every failure.
    fn check_ext(
        &self,
        _expect: &Self::ExpectExt,
        _trajectory: &Trajectory<Self::Observed>,
    ) -> Vec<String> {
        Vec::new()
    }
}

/// The [`Subject`] for agents built by an [`AgentFactory`].
pub struct AgentSubject<'a> {
    factory: &'a dyn AgentFactory,
}

impl<'a> AgentSubject<'a> {
    pub fn new(factory: &'a dyn AgentFactory) -> Self {
        Self { factory }
    }
}

#[async_trait]
impl Subject for AgentSubject<'_> {
    type Setup = NoExtension;
    type ExpectExt = NoExtension;
    type Observed = ();

    async fn run(&self, scenario: &Scenario, run: usize) -> Result<Trajectory, FactoryError> {
        Ok(run_agent(self.factory, scenario, run).await?.1)
    }
}

/// Run `scenario` once on an agent from `factory`: the kernel outcome and
/// its trajectory (with any script problems in `run_failures`).
pub(crate) async fn run_agent(
    factory: &dyn AgentFactory,
    scenario: &Scenario,
    run: usize,
) -> Result<(RunOutcome, Trajectory), FactoryError> {
    let script = scenario.script.as_ref().map(|_| {
        let responses = scenario.scripted_responses();
        (responses.len(), Arc::new(ScriptedProvider::new(responses)))
    });
    let provider: Arc<dyn Provider> = match (&script, factory.live_provider(scenario)) {
        (Some((_, scripted)), _) => scripted.clone(),
        (None, Some(live)) => live,
        (None, None) => {
            return Err("scenario has no `script` and the factory supplied no live provider".into())
        }
    };
    let agent = factory
        .build(scenario, provider)
        .map_err(|err| format!("agent factory failed: {err}"))?;
    let session = format!("eval-{}-{run}", scenario.name);
    let outcome = agent
        .run(&session, &scenario.input)
        .await
        .map_err(|err| format!("run failed: {err}"))?;

    let mut trajectory = Trajectory::from_outcome(&outcome);
    if let Some((len, scripted)) = &script {
        let calls = scripted.requests().len();
        if calls > *len {
            trajectory.run_failures.push(format!(
                "script ran out: the model was called {calls} times but the script has {len} replies"
            ));
        } else if scripted.remaining() > 0 {
            trajectory.run_failures.push(format!(
                "script not fully consumed: {} of {len} replies unused",
                scripted.remaining()
            ));
        }
    }
    Ok((outcome, trajectory))
}

impl CaseResult {
    fn errored<S, X>(scenario: &Scenario<S, X>, message: String, elapsed: Duration) -> Self {
        Self {
            scenario: scenario.name.clone(),
            passed: false,
            failures: vec![message],
            status: "error".into(),
            final_text: None,
            tools: Vec::new(),
            turns: 0,
            billable_tokens: 0,
            duration_ms: elapsed.as_secs_f64() * 1000.0,
        }
    }
}

impl Scenario {
    /// Run once with a session id derived from the scenario name.
    pub async fn run(&self, factory: &dyn AgentFactory) -> CaseResult {
        self.run_on(&AgentSubject::new(factory), 0).await
    }
}

impl<S, X> Scenario<S, X> {
    /// Run once on `subject` and check the expectations.
    pub async fn run_on<T>(&self, subject: &T, run: usize) -> CaseResult
    where
        T: Subject<Setup = S, ExpectExt = X>,
    {
        let started = Instant::now();
        let trajectory = match subject.run(self, run).await {
            Ok(trajectory) => trajectory,
            Err(err) => return CaseResult::errored(self, err.to_string(), started.elapsed()),
        };
        let elapsed = started.elapsed();

        let mut failures = trajectory.run_failures.clone();
        failures.extend(check_trajectory(&self.expect, &trajectory));
        if let Some(ext) = &self.expect.ext {
            failures.extend(subject.check_ext(ext, &trajectory));
        }

        CaseResult {
            scenario: self.name.clone(),
            passed: failures.is_empty(),
            failures,
            status: trajectory.status,
            final_text: trajectory.final_text,
            tools: trajectory.receipts.into_iter().map(|r| r.tool).collect(),
            turns: trajectory.turns,
            billable_tokens: trajectory.billable_tokens,
            duration_ms: elapsed.as_secs_f64() * 1000.0,
        }
    }
}

fn status_str(status: Option<ReceiptStatus>) -> String {
    match status {
        Some(status) => serde_json::to_value(status)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| format!("{status:?}")),
        None => "unreported".to_string(),
    }
}

fn is_failure(status: Option<ReceiptStatus>) -> bool {
    status.is_some_and(|s| !matches!(s, ReceiptStatus::Succeeded | ReceiptStatus::Denied))
}

/// Check every expectation against a kernel outcome. Returns all failures
/// (empty means pass). Public so callers can assert on outcomes they produced
/// themselves, for example after a resume.
pub fn check_expectations(expect: &Expect, outcome: &RunOutcome) -> Vec<String> {
    check_trajectory(expect, &Trajectory::from_outcome(outcome))
}

/// Check every expectation against a trajectory. Returns all failures (empty
/// means pass).
pub fn check_trajectory<O, X>(expect: &Expect<X>, trajectory: &Trajectory<O>) -> Vec<String> {
    let mut failures = Vec::new();
    let receipts = &trajectory.receipts;
    let tools: Vec<&str> = receipts.iter().map(|r| r.tool.as_str()).collect();
    let actual_status = trajectory.status.as_str();

    if let Some(expected) = expect.status {
        if expected.as_str() != actual_status {
            failures.push(format!(
                "status: expected `{}`, got `{actual_status}` ({})",
                expected.as_str(),
                trajectory.status_detail
            ));
        }
    }

    if let Some(expected) = expect.succeeded {
        if trajectory.succeeded != expected {
            failures.push(format!(
                "succeeded: expected `{expected}`, got `{}` (status {actual_status}, unmet: {:?})",
                trajectory.succeeded, trajectory.unmet
            ));
        }
    }

    let text = trajectory.final_text.as_deref().unwrap_or_default();
    let lower = text.to_lowercase();
    let shown = || {
        if trajectory.final_text.is_none() {
            "no final text".to_string()
        } else {
            format!("final text: {}", truncate(text, 200))
        }
    };
    for needle in &expect.final_text_contains {
        if !lower.contains(&needle.to_lowercase()) {
            failures.push(format!(
                "final_text_contains: missing `{needle}` ({})",
                shown()
            ));
        }
    }
    for needle in &expect.final_text_contains_exact {
        if !text.contains(needle.as_str()) {
            failures.push(format!(
                "final_text_contains_exact: missing `{needle}` ({})",
                shown()
            ));
        }
    }
    for needle in &expect.final_text_not_contains {
        if lower.contains(&needle.to_lowercase()) {
            failures.push(format!(
                "final_text_not_contains: found `{needle}` ({})",
                shown()
            ));
        }
    }

    check_tool_lane(
        &mut failures,
        "",
        ToolLaneView {
            used: &expect.tools_used,
            not_used: &expect.tools_not_used,
            in_order: &expect.tools_in_order,
            counts: &expect.tool_call_counts,
        },
        &tools,
    );
    let proposed: Vec<&str> = trajectory.proposed.iter().map(String::as_str).collect();
    check_tool_lane(
        &mut failures,
        "proposed.",
        ToolLaneView {
            used: &expect.proposed.tools_used,
            not_used: &expect.proposed.tools_not_used,
            in_order: &expect.proposed.tools_in_order,
            counts: &expect.proposed.tool_call_counts,
        },
        &proposed,
    );

    for (tool, expected) in &expect.tool_dispatch_counts {
        let actual = receipts
            .iter()
            .filter(|r| r.dispatched && &r.tool == tool)
            .count();
        if actual != *expected as usize {
            failures.push(format!(
                "tool_dispatch_counts: `{tool}` expected {expected} dispatch(es), got {actual}"
            ));
        }
    }

    for (tool, expected) in &expect.receipt_statuses {
        let actual: Vec<String> = receipts
            .iter()
            .filter(|r| &r.tool == tool)
            .map(|r| status_str(r.status))
            .collect();
        let wanted: Vec<String> = expected.iter().map(|s| status_str(Some(*s))).collect();
        if actual != wanted {
            failures.push(format!(
                "receipt_statuses: `{tool}` expected [{}], got [{}]",
                wanted.join(", "),
                actual.join(", ")
            ));
        }
    }

    if let Some(max) = expect.max_turns {
        if trajectory.turns > max {
            failures.push(format!(
                "max_turns: used {} turns, limit {max}",
                trajectory.turns
            ));
        }
    }
    if let Some(max) = expect.max_billable_tokens {
        let used = trajectory.billable_tokens;
        if used > max {
            failures.push(format!(
                "max_billable_tokens: used {used} billable tokens, limit {max}"
            ));
        }
    }

    if expect.no_failed_receipts {
        let mut bad: BTreeMap<String, usize> = BTreeMap::new();
        for r in receipts.iter().filter(|r| is_failure(r.status)) {
            *bad.entry(format!("{} {}", r.tool, status_str(r.status)))
                .or_default() += 1;
        }
        if !bad.is_empty() {
            let list: Vec<String> = bad
                .into_iter()
                .map(|(k, n)| if n > 1 { format!("{k} x{n}") } else { k })
                .collect();
            failures.push(format!(
                "no_failed_receipts: failed receipts: {}",
                list.join(", ")
            ));
        }
    }

    failures
}

struct ToolLaneView<'a> {
    used: &'a [String],
    not_used: &'a [String],
    in_order: &'a [String],
    counts: &'a BTreeMap<String, u32>,
}

/// Check one lane's tool assertions. `prefix` names the lane in failures.
fn check_tool_lane(
    failures: &mut Vec<String>,
    prefix: &str,
    expect: ToolLaneView<'_>,
    tools: &[&str],
) {
    let called = || format!("[{}]", tools.join(", "));
    for tool in expect.used {
        if !tools.contains(&tool.as_str()) {
            failures.push(format!(
                "{prefix}tools_used: `{tool}` was never called (calls: {})",
                called()
            ));
        }
    }
    for tool in expect.not_used {
        let count = tools.iter().filter(|t| **t == tool).count();
        if count > 0 {
            failures.push(format!(
                "{prefix}tools_not_used: `{tool}` was called {count} time(s)"
            ));
        }
    }

    if !expect.in_order.is_empty() {
        let mut want = expect.in_order.iter().peekable();
        for tool in tools {
            if want.peek().is_some_and(|w| w == tool) {
                want.next();
            }
        }
        if let Some(missing) = want.next() {
            failures.push(format!(
                "{prefix}tools_in_order: expected subsequence [{}], stopped matching at `{missing}` (calls: {})",
                expect.in_order.join(", "),
                called()
            ));
        }
    }

    for (tool, expected) in expect.counts {
        let actual = tools.iter().filter(|t| **t == tool).count();
        if actual != *expected as usize {
            failures.push(format!(
                "{prefix}tool_call_counts: `{tool}` expected {expected} call(s), got {actual}"
            ));
        }
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max).collect();
    format!("{head}...")
}
