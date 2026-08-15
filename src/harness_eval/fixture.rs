use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::events::{DecisionPointData, Event, EventType, HarnessEvalSnapshot, TaskEndData};

fn default_owner() -> String {
    "owner".to_string()
}

fn empty_json() -> String {
    "{}".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessEvalFixture {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub session_id: String,
    pub user_text: String,
    #[serde(default = "default_owner")]
    pub user_role: String,
    /// When true, use orchestrator-mode test harness (required for orchestration routing).
    #[serde(default)]
    pub orchestrator: bool,
    /// When true, use non-uniform primary/smart model tiers (routing tests).
    #[serde(default)]
    pub routing_models: bool,
    #[serde(default)]
    pub mock_responses: Vec<MockResponseSpec>,
    /// Optional semantic task-assessment responses, kept separate from the
    /// main-loop queue so fixtures never depend on lexical mock inference.
    #[serde(default)]
    pub task_assessments: Vec<MockResponseSpec>,
    pub expect: ExpectBlock,
    /// Optional DB seed data applied before handle_message.
    #[serde(default)]
    pub seed: FixtureSeed,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FixtureSeed {
    #[serde(default)]
    pub goals: Vec<SeedGoal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedGoal {
    pub description: String,
    #[serde(default = "default_active_status")]
    pub status: String,
}

fn default_active_status() -> String {
    "active".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExpectBlock {
    #[serde(default)]
    pub orchestration_route: Option<String>,
    #[serde(default)]
    pub tools_required_predicted: Option<bool>,
    #[serde(default)]
    pub tools_used: Vec<String>,
    #[serde(default)]
    pub tools_not_used: Vec<String>,
    /// Tools that must appear in this relative order (subsequence match —
    /// unrelated calls may interleave).
    #[serde(default)]
    pub tools_in_order: Vec<String>,
    /// Exact call count per tool. Guards duplicate-side-effect regressions,
    /// e.g. `{twitter_post: 1}`. Tools omitted here are unconstrained.
    #[serde(default)]
    pub tool_call_counts: BTreeMap<String, u32>,
    #[serde(default)]
    pub outcome: Option<String>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub llm_calls_min: Option<u32>,
    #[serde(default)]
    pub llm_calls_max: Option<u32>,
    #[serde(default)]
    pub tool_calls_min: Option<u32>,
    #[serde(default)]
    pub tool_calls_max: Option<u32>,
    #[serde(default)]
    pub weighted_tokens_max: Option<u64>,
    #[serde(default)]
    pub forbidden_mutation_attempts_max: Option<u32>,
    #[serde(default)]
    pub plan_steps_completed_min: Option<u32>,
    #[serde(default)]
    pub routing_accuracy_min: Option<f32>,
    #[serde(default)]
    pub progress_yield_min: Option<f32>,
    #[serde(default)]
    pub contract_fulfillment_min: Option<f32>,
    #[serde(default)]
    pub cost_efficiency_min: Option<f32>,
    #[serde(default)]
    pub overall_min: Option<f32>,
    #[serde(default)]
    pub direct_return: Option<bool>,
    #[serde(default)]
    pub response_fallthrough: Option<bool>,
    #[serde(default)]
    pub guard_fired: Vec<String>,
    #[serde(default)]
    pub decision_types_seen: Vec<String>,
    #[serde(default)]
    pub response_contains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MockResponseSpec {
    Text { text: String },
    ToolCall { tool_call: ToolCallSpec },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallSpec {
    pub name: String,
    #[serde(default = "empty_json")]
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct HarnessEvalRunResult {
    pub response_text: String,
    pub task_end: TaskEndData,
    pub harness_eval: HarnessEvalSnapshot,
    pub llm_calls: u32,
    pub tool_names: Vec<String>,
    pub decision_types: Vec<String>,
}

pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/harness_eval/fixtures")
}

pub fn load_fixture_file(path: &Path) -> anyhow::Result<HarnessEvalFixture> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read fixture {}", path.display()))?;
    parse_fixture_yaml(&raw)
}

pub fn load_fixtures_dir(dir: &Path) -> anyhow::Result<Vec<(PathBuf, HarnessEvalFixture)>> {
    let mut out = Vec::new();
    if !dir.is_dir() {
        return Ok(out);
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext == "yaml" || ext == "yml")
        })
        .collect();
    paths.sort();
    for path in paths {
        let fixture = load_fixture_file(&path)?;
        out.push((path, fixture));
    }
    Ok(out)
}

pub fn parse_fixture_yaml(raw: &str) -> anyhow::Result<HarnessEvalFixture> {
    serde_yaml::from_str(raw).context("parse harness eval fixture YAML")
}

pub fn collect_run_result(
    events: &[Event],
    response_text: &str,
) -> anyhow::Result<HarnessEvalRunResult> {
    let task_end_event = events
        .iter()
        .rev()
        .find(|event| event.event_type == EventType::TaskEnd)
        .context("missing TaskEnd event")?;
    let task_end = task_end_event
        .parse_data::<TaskEndData>()
        .context("parse TaskEnd")?;
    let harness_eval = task_end
        .harness_eval
        .clone()
        .context("TaskEnd missing harness_eval snapshot")?;
    let task_id = &task_end.task_id;

    let llm_calls = events
        .iter()
        .filter(|event| {
            event.event_type == EventType::LlmCall && event.task_id.as_deref() == Some(task_id)
        })
        .count() as u32;

    let mut tool_names = Vec::new();
    for event in events
        .iter()
        .filter(|event| event.event_type == EventType::ToolCall)
    {
        if event.task_id.as_deref() != Some(task_id.as_str()) {
            continue;
        }
        if let Ok(data) = event.parse_data::<crate::events::ToolCallData>() {
            tool_names.push(data.name);
        }
    }

    let mut decision_types = Vec::new();
    for event in events
        .iter()
        .filter(|event| event.event_type == EventType::DecisionPoint)
    {
        if event.task_id.as_deref() != Some(task_id.as_str()) {
            continue;
        }
        if let Ok(data) = event.parse_data::<DecisionPointData>() {
            decision_types.push(format!("{:?}", data.decision_type));
        }
    }

    Ok(HarnessEvalRunResult {
        response_text: response_text.to_string(),
        task_end,
        harness_eval,
        llm_calls,
        tool_names,
        decision_types,
    })
}

pub fn assert_expectations(
    fixture: &HarnessEvalFixture,
    result: &HarnessEvalRunResult,
) -> anyhow::Result<()> {
    let expect = &fixture.expect;
    let eval = &result.harness_eval;

    if let Some(expected) = &expect.orchestration_route {
        anyhow::ensure!(
            eval.orchestration_route == *expected,
            "[{}] orchestration_route: expected {expected}, got {}",
            fixture.name,
            eval.orchestration_route
        );
    }

    if let Some(expected) = expect.tools_required_predicted {
        anyhow::ensure!(
            eval.routing.tools_required_predicted == expected,
            "[{}] tools_required_predicted: expected {expected}, got {}",
            fixture.name,
            eval.routing.tools_required_predicted
        );
    }

    if !expect.tools_used.is_empty() {
        for tool in &expect.tools_used {
            anyhow::ensure!(
                result.tool_names.iter().any(|name| name == tool),
                "[{}] tools_used: expected to call {tool}, got {:?}",
                fixture.name,
                result.tool_names
            );
        }
    }
    for tool in &expect.tools_not_used {
        anyhow::ensure!(
            !result.tool_names.iter().any(|name| name == tool),
            "[{}] tools_not_used: {tool} was called; got {:?}",
            fixture.name,
            result.tool_names
        );
    }

    if !expect.tools_in_order.is_empty() {
        assert_tool_order(&fixture.name, &expect.tools_in_order, &result.tool_names)?;
    }
    if !expect.tool_call_counts.is_empty() {
        assert_tool_counts(&fixture.name, &expect.tool_call_counts, &result.tool_names)?;
    }

    if let Some(expected) = &expect.outcome {
        let actual = result.task_end.effective_outcome().as_str();
        anyhow::ensure!(
            actual == expected,
            "[{}] outcome: expected {expected}, got {actual}",
            fixture.name
        );
    }

    if let Some(min) = expect.llm_calls_min {
        anyhow::ensure!(
            result.llm_calls >= min,
            "[{}] llm_calls_min: expected >= {min}, got {}",
            fixture.name,
            result.llm_calls
        );
    }
    if let Some(max) = expect.llm_calls_max {
        anyhow::ensure!(
            result.llm_calls <= max,
            "[{}] llm_calls_max: expected <= {max}, got {}",
            fixture.name,
            result.llm_calls
        );
    }

    let tool_calls = result.task_end.tool_calls_count;
    if let Some(min) = expect.tool_calls_min {
        anyhow::ensure!(
            tool_calls >= min,
            "[{}] tool_calls_min: expected >= {min}, got {tool_calls}",
            fixture.name
        );
    }
    if let Some(max) = expect.tool_calls_max {
        anyhow::ensure!(
            tool_calls <= max,
            "[{}] tool_calls_max: expected <= {max}, got {tool_calls}",
            fixture.name
        );
    }
    if let Some(max) = expect.weighted_tokens_max {
        anyhow::ensure!(
            eval.cost.weighted_tokens <= max,
            "[{}] weighted_tokens_max: expected <= {max}, got {}",
            fixture.name,
            eval.cost.weighted_tokens
        );
    }
    if let Some(max) = expect.forbidden_mutation_attempts_max {
        anyhow::ensure!(
            eval.quality.contract.forbidden_mutation_attempts <= max,
            "[{}] forbidden_mutation_attempts_max: expected <= {max}, got {}",
            fixture.name,
            eval.quality.contract.forbidden_mutation_attempts
        );
    }
    if let Some(min) = expect.plan_steps_completed_min {
        let actual = eval.progress.plan_steps_completed.unwrap_or(0);
        anyhow::ensure!(
            actual >= min,
            "[{}] plan_steps_completed_min: expected >= {min}, got {actual}",
            fixture.name
        );
    }

    assert_score_min(
        &fixture.name,
        "routing_accuracy",
        expect.routing_accuracy_min,
        eval.scores.routing_accuracy,
    )?;
    assert_score_min(
        &fixture.name,
        "progress_yield",
        expect.progress_yield_min,
        eval.scores.progress_yield,
    )?;
    assert_score_min(
        &fixture.name,
        "contract_fulfillment",
        expect.contract_fulfillment_min,
        eval.scores.contract_fulfillment,
    )?;
    assert_score_min(
        &fixture.name,
        "cost_efficiency",
        expect.cost_efficiency_min,
        eval.scores.cost_efficiency,
    )?;
    assert_score_min(
        &fixture.name,
        "overall",
        expect.overall_min,
        eval.scores.overall,
    )?;

    if let Some(direct_return) = expect.direct_return {
        anyhow::ensure!(
            eval.routing.direct_return_attempted == direct_return,
            "[{}] direct_return: expected {direct_return}, got {}",
            fixture.name,
            eval.routing.direct_return_attempted
        );
    }

    if let Some(expected) = &expect.stop_reason {
        anyhow::ensure!(
            eval.quality.stop_reason == *expected,
            "[{}] stop_reason: expected {expected}, got {}",
            fixture.name,
            eval.quality.stop_reason
        );
    }

    if expect.response_fallthrough == Some(true) {
        anyhow::ensure!(
            eval.routing.response_fallthrough,
            "[{}] expected response_fallthrough=true",
            fixture.name
        );
    }

    for guard in &expect.guard_fired {
        let fired = match guard.as_str() {
            "RepetitiveCallDetection" => eval.progress.repetition_guard_fires > 0,
            "Stall" | "stall" => eval.progress.stall_guard_fires > 0,
            other => {
                return Err(anyhow::anyhow!(
                    "[{}] unknown guard_fired value: {other}",
                    fixture.name
                ));
            }
        };
        anyhow::ensure!(
            fired,
            "[{}] guard_fired: expected {guard} to fire",
            fixture.name
        );
    }

    for decision_type in &expect.decision_types_seen {
        anyhow::ensure!(
            result
                .decision_types
                .iter()
                .any(|seen| seen.contains(decision_type)),
            "[{}] decision_types_seen: expected {decision_type}, got {:?}",
            fixture.name,
            result.decision_types
        );
    }

    for needle in &expect.response_contains {
        anyhow::ensure!(
            result.response_text.contains(needle),
            "[{}] response_contains: expected substring {needle:?}",
            fixture.name
        );
    }

    Ok(())
}

/// Assert `expected` appears as a subsequence of `actual` tool calls.
///
/// Unrelated calls may interleave; only relative order is checked. A tool
/// listed twice must appear twice, in that order.
fn assert_tool_order(
    fixture_name: &str,
    expected: &[String],
    actual: &[String],
) -> anyhow::Result<()> {
    let mut cursor = 0usize;
    for tool in expected {
        match actual[cursor..].iter().position(|name| name == tool) {
            Some(offset) => cursor += offset + 1,
            None => anyhow::bail!(
                "[{fixture_name}] tools_in_order: expected {expected:?} as a subsequence, \
                 but {tool} does not appear after position {cursor}; got {actual:?}"
            ),
        }
    }
    Ok(())
}

/// Assert each named tool was called exactly the expected number of times.
/// Tools absent from `expected` are unconstrained.
fn assert_tool_counts(
    fixture_name: &str,
    expected: &BTreeMap<String, u32>,
    actual: &[String],
) -> anyhow::Result<()> {
    for (tool, want) in expected {
        let got = actual
            .iter()
            .filter(|name| name.as_str() == tool.as_str())
            .count() as u32;
        anyhow::ensure!(
            got == *want,
            "[{fixture_name}] tool_call_counts: expected {tool} called {want} time(s), \
             got {got}; all calls: {actual:?}"
        );
    }
    Ok(())
}

fn assert_score_min(
    fixture_name: &str,
    label: &str,
    min: Option<f32>,
    actual: f32,
) -> anyhow::Result<()> {
    if let Some(min) = min {
        anyhow::ensure!(
            actual + f32::EPSILON >= min,
            "[{fixture_name}] {label}_min: expected >= {min:.2}, got {actual:.2}"
        );
    }
    Ok(())
}

/// Build a fixture YAML from a completed task (structural expect block only).
pub fn build_recorded_fixture(
    name: &str,
    session_id: &str,
    user_text: &str,
    eval: &HarnessEvalSnapshot,
    task_end: &TaskEndData,
    tool_names: &[String],
) -> HarnessEvalFixture {
    HarnessEvalFixture {
        name: name.to_string(),
        description: "Recorded from production run (structural expect only)".to_string(),
        session_id: session_id.to_string(),
        user_text: user_text.to_string(),
        user_role: "owner".to_string(),
        orchestrator: false,
        routing_models: false,
        mock_responses: Vec::new(),
        task_assessments: Vec::new(),
        seed: FixtureSeed::default(),
        expect: ExpectBlock {
            orchestration_route: Some(eval.orchestration_route.clone()),
            tools_required_predicted: Some(eval.routing.tools_required_predicted),
            tools_used: tool_names.to_vec(),
            tools_not_used: Vec::new(),
            tools_in_order: Vec::new(),
            tool_call_counts: BTreeMap::new(),
            outcome: Some(task_end.effective_outcome().as_str().to_string()),
            stop_reason: Some(eval.quality.stop_reason.clone()),
            llm_calls_min: None,
            llm_calls_max: Some(eval.cost.llm_calls),
            tool_calls_min: None,
            tool_calls_max: Some(task_end.tool_calls_count),
            weighted_tokens_max: Some(eval.cost.weighted_tokens),
            forbidden_mutation_attempts_max: Some(
                eval.quality.contract.forbidden_mutation_attempts,
            ),
            plan_steps_completed_min: eval.progress.plan_steps_completed,
            routing_accuracy_min: Some(round_score(eval.scores.routing_accuracy)),
            progress_yield_min: Some(round_score(eval.scores.progress_yield)),
            contract_fulfillment_min: Some(round_score(eval.scores.contract_fulfillment)),
            cost_efficiency_min: Some(round_score(eval.scores.cost_efficiency)),
            overall_min: Some(round_score(eval.scores.overall)),
            direct_return: Some(eval.routing.direct_return_attempted),
            response_fallthrough: Some(eval.routing.response_fallthrough),
            guard_fired: Vec::new(),
            decision_types_seen: Vec::new(),
            response_contains: Vec::new(),
        },
    }
}

fn round_score(value: f32) -> f32 {
    (value * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(raw: &[&str]) -> Vec<String> {
        raw.iter().map(|s| s.to_string()).collect()
    }

    fn counts(raw: &[(&str, u32)]) -> BTreeMap<String, u32> {
        raw.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn tools_in_order_rejects_reversed_sequence() {
        let err = assert_tool_order(
            "fx",
            &names(&["read_file", "write_file"]),
            &names(&["write_file", "read_file"]),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("tools_in_order"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tools_in_order_allows_unrelated_calls_between() {
        assert_tool_order(
            "fx",
            &names(&["read_file", "write_file"]),
            &names(&["read_file", "system_info", "write_file"]),
        )
        .unwrap();
    }

    #[test]
    fn tools_in_order_requires_each_repeat_of_a_tool() {
        let err = assert_tool_order(
            "fx",
            &names(&["read_file", "write_file", "read_file"]),
            &names(&["read_file", "write_file"]),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("tools_in_order"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tools_in_order_rejects_missing_tool() {
        let err = assert_tool_order(
            "fx",
            &names(&["read_file", "write_file"]),
            &names(&["read_file"]),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("write_file"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tool_call_counts_rejects_duplicate_side_effect() {
        let err = assert_tool_counts(
            "fx",
            &counts(&[("twitter_post", 1)]),
            &names(&["twitter_post", "twitter_post"]),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("twitter_post") && err.to_string().contains("got 2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tool_call_counts_accepts_exact_match() {
        assert_tool_counts(
            "fx",
            &counts(&[("twitter_post", 1), ("read_file", 2)]),
            &names(&["read_file", "twitter_post", "read_file"]),
        )
        .unwrap();
    }

    #[test]
    fn tool_call_counts_zero_requires_absence() {
        let err = assert_tool_counts(
            "fx",
            &counts(&[("twitter_post", 0)]),
            &names(&["twitter_post"]),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("twitter_post"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tool_call_counts_ignores_unlisted_tools() {
        assert_tool_counts(
            "fx",
            &counts(&[("read_file", 1)]),
            &names(&["read_file", "system_info"]),
        )
        .unwrap();
    }

    #[test]
    fn parse_fixture_yaml_reads_order_and_count_assertions() {
        let yaml = r#"
name: ordering
session_id: s1
user_text: Post it
expect:
  tools_in_order: [read_file, twitter_post]
  tool_call_counts:
    twitter_post: 1
"#;
        let fixture = parse_fixture_yaml(yaml).unwrap();
        assert_eq!(
            fixture.expect.tools_in_order,
            names(&["read_file", "twitter_post"])
        );
        assert_eq!(
            fixture.expect.tool_call_counts.get("twitter_post"),
            Some(&1)
        );
    }

    #[test]
    fn parse_minimal_fixture_yaml() {
        let yaml = r#"
name: hello
session_id: s1
user_text: Hello
expect:
  outcome: succeeded
"#;
        let fixture = parse_fixture_yaml(yaml).unwrap();
        assert_eq!(fixture.name, "hello");
        assert_eq!(fixture.expect.outcome.as_deref(), Some("succeeded"));
    }
}
