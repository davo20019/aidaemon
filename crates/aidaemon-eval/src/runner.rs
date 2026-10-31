//! Running one scenario and checking its expectations.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aidaemon_core::testing::ScriptedProvider;
use aidaemon_core::{Agent, Provider, ReceiptStatus, RunOutcome};
use serde::{Deserialize, Serialize};

use crate::scenario::{status_name, Expect, Scenario};

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

impl CaseResult {
    fn errored(scenario: &Scenario, message: String, elapsed: Duration) -> Self {
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
        self.run_numbered(factory, 0).await
    }

    pub(crate) async fn run_numbered(&self, factory: &dyn AgentFactory, run: usize) -> CaseResult {
        let started = Instant::now();
        let script = self.script.as_ref().map(|_| {
            let responses = self.scripted_responses();
            (responses.len(), Arc::new(ScriptedProvider::new(responses)))
        });
        let provider: Arc<dyn Provider> = match (&script, factory.live_provider(self)) {
            (Some((_, scripted)), _) => scripted.clone(),
            (None, Some(live)) => live,
            (None, None) => {
                return CaseResult::errored(
                    self,
                    "scenario has no `script` and the factory supplied no live provider".into(),
                    started.elapsed(),
                )
            }
        };
        let agent = match factory.build(self, provider) {
            Ok(agent) => agent,
            Err(err) => {
                return CaseResult::errored(
                    self,
                    format!("agent factory failed: {err}"),
                    started.elapsed(),
                )
            }
        };
        let session = format!("eval-{}-{run}", self.name);
        let outcome = match agent.run(&session, &self.input).await {
            Ok(outcome) => outcome,
            Err(err) => {
                return CaseResult::errored(self, format!("run failed: {err}"), started.elapsed())
            }
        };
        let elapsed = started.elapsed();

        let mut failures = Vec::new();
        if let Some((len, scripted)) = &script {
            let calls = scripted.requests().len();
            if calls > *len {
                failures.push(format!(
                    "script ran out: the model was called {calls} times but the script has {len} replies"
                ));
            } else if scripted.remaining() > 0 {
                failures.push(format!(
                    "script not fully consumed: {} of {len} replies unused",
                    scripted.remaining()
                ));
            }
        }
        failures.extend(check_expectations(&self.expect, &outcome));

        CaseResult {
            scenario: self.name.clone(),
            passed: failures.is_empty(),
            failures,
            status: status_name(&outcome.status).to_string(),
            final_text: outcome.final_text.clone(),
            tools: outcome.receipts().iter().map(|r| r.tool.clone()).collect(),
            turns: outcome.turns(),
            billable_tokens: outcome.usage().billable_tokens(),
            duration_ms: elapsed.as_secs_f64() * 1000.0,
        }
    }
}

fn status_str(status: ReceiptStatus) -> String {
    serde_json::to_value(status)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{status:?}"))
}

fn is_failure(status: ReceiptStatus) -> bool {
    !matches!(status, ReceiptStatus::Succeeded | ReceiptStatus::Denied)
}

/// Check every expectation against an outcome. Returns all failures (empty
/// means pass). Public so callers can assert on outcomes they produced
/// themselves, for example after a resume.
pub fn check_expectations(expect: &Expect, outcome: &RunOutcome) -> Vec<String> {
    let mut failures = Vec::new();
    let receipts = outcome.receipts();
    let tools: Vec<&str> = receipts.iter().map(|r| r.tool.as_str()).collect();
    let actual_status = status_name(&outcome.status);

    if let Some(expected) = expect.status {
        if expected.as_str() != actual_status {
            failures.push(format!(
                "status: expected `{}`, got `{actual_status}` ({:?})",
                expected.as_str(),
                outcome.status
            ));
        }
    }

    let text = outcome.final_text.as_deref().unwrap_or_default();
    let lower = text.to_lowercase();
    let shown = || {
        if outcome.final_text.is_none() {
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
    for needle in &expect.final_text_not_contains {
        if lower.contains(&needle.to_lowercase()) {
            failures.push(format!(
                "final_text_not_contains: found `{needle}` ({})",
                shown()
            ));
        }
    }

    let called = || format!("[{}]", tools.join(", "));
    for tool in &expect.tools_used {
        if !tools.contains(&tool.as_str()) {
            failures.push(format!(
                "tools_used: `{tool}` was never called (calls: {})",
                called()
            ));
        }
    }
    for tool in &expect.tools_not_used {
        let count = tools.iter().filter(|t| **t == tool).count();
        if count > 0 {
            failures.push(format!(
                "tools_not_used: `{tool}` was called {count} time(s)"
            ));
        }
    }

    if !expect.tools_in_order.is_empty() {
        let mut want = expect.tools_in_order.iter().peekable();
        for tool in &tools {
            if want.peek().is_some_and(|w| w == tool) {
                want.next();
            }
        }
        if let Some(missing) = want.next() {
            failures.push(format!(
                "tools_in_order: expected subsequence [{}], stopped matching at `{missing}` (calls: {})",
                expect.tools_in_order.join(", "),
                called()
            ));
        }
    }

    for (tool, expected) in &expect.tool_call_counts {
        let actual = tools.iter().filter(|t| **t == tool).count();
        if actual != *expected as usize {
            failures.push(format!(
                "tool_call_counts: `{tool}` expected {expected} call(s), got {actual}"
            ));
        }
    }

    for (tool, expected) in &expect.receipt_statuses {
        let actual: Vec<String> = receipts
            .iter()
            .filter(|r| &r.tool == tool)
            .map(|r| status_str(r.status))
            .collect();
        let wanted: Vec<String> = expected.iter().map(|s| status_str(*s)).collect();
        if actual != wanted {
            failures.push(format!(
                "receipt_statuses: `{tool}` expected [{}], got [{}]",
                wanted.join(", "),
                actual.join(", ")
            ));
        }
    }

    if let Some(max) = expect.max_turns {
        if outcome.turns() > max {
            failures.push(format!(
                "max_turns: used {} turns, limit {max}",
                outcome.turns()
            ));
        }
    }
    if let Some(max) = expect.max_billable_tokens {
        let used = outcome.usage().billable_tokens();
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

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max).collect();
    format!("{head}...")
}
