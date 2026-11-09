//! Reads a task's trace into a run result, and records a production trace as
//! a draft eval scenario.
//!
//! `aidaemon-eval` is a dev-dependency, so `db_probe` cannot use its types.
//! The `Recorded*` structs write the same YAML; a test in
//! [`crate::harness_eval`] parses their output as a real scenario.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Context;
use serde::Serialize;

use crate::events::{
    DecisionPointData, Event, EventType, HarnessEvalSnapshot, TaskEndData, ToolReceiptV1,
};

#[derive(Debug, Clone)]
pub struct HarnessEvalRunResult {
    pub response_text: String,
    pub task_end: TaskEndData,
    pub harness_eval: HarnessEvalSnapshot,
    pub llm_calls: u32,
    /// LLM calls without an auxiliary `call_purpose`, i.e. execution-loop turns.
    pub execution_llm_calls: u32,
    pub tool_names: Vec<String>,
    /// Tools whose result receipt reports `ToolInvocationStage::Dispatched`.
    pub dispatched_tool_names: Vec<String>,
    /// Every tool result in order, with its receipt when it has one.
    pub tool_results: Vec<(String, Option<ToolReceiptV1>)>,
    pub decision_types: Vec<String>,
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

    // Unparseable trace events are errors, not skips: dropping one would
    // silently undercount calls and let count/absence assertions pass.
    let task_events = |event_type: EventType| {
        events.iter().filter(move |event| {
            event.event_type == event_type && event.task_id.as_deref() == Some(task_id.as_str())
        })
    };
    let mut llm_calls = 0u32;
    let mut execution_llm_calls = 0u32;
    for event in task_events(EventType::LlmCall) {
        let data = event
            .parse_data::<crate::events::LlmCallData>()
            .with_context(|| format!("parse LlmCall event {}", event.id))?;
        llm_calls += 1;
        if data.call_purpose.is_none() {
            execution_llm_calls += 1;
        }
    }

    let mut tool_names = Vec::new();
    for event in task_events(EventType::ToolCall) {
        let data = event
            .parse_data::<crate::events::ToolCallData>()
            .with_context(|| format!("parse ToolCall event {}", event.id))?;
        tool_names.push(data.name);
    }

    let mut dispatched_tool_names = Vec::new();
    let mut tool_results = Vec::new();
    for event in task_events(EventType::ToolResult) {
        let data = event
            .parse_data::<crate::events::ToolResultData>()
            .with_context(|| format!("parse ToolResult event {}", event.id))?;
        if data.receipt.as_ref().is_some_and(|receipt| {
            receipt.invocation_stage == crate::traits::ToolInvocationStage::Dispatched
        }) {
            dispatched_tool_names.push(data.name.clone());
        }
        tool_results.push((data.name, data.receipt));
    }

    let mut decision_types = Vec::new();
    for event in task_events(EventType::DecisionPoint) {
        let data = event
            .parse_data::<DecisionPointData>()
            .with_context(|| format!("parse DecisionPoint event {}", event.id))?;
        decision_types.push(format!("{:?}", data.decision_type));
    }

    Ok(HarnessEvalRunResult {
        response_text: response_text.to_string(),
        task_end,
        harness_eval,
        llm_calls,
        execution_llm_calls,
        tool_names,
        dispatched_tool_names,
        tool_results,
        decision_types,
    })
}

/// Where recorded drafts are written by default. Deliberately outside the
/// suite's `scenarios` directory, which is auto-loaded: a draft needs a human
/// check against its trace first (see [`record_scenario_from_events`]).
pub fn recorded_scenarios_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/harness_eval/recorded")
}

/// Leading comment for recorded drafts, explaining what must be added before
/// the draft is promoted into the suite.
pub const RECORDED_SCENARIO_HEADER: &str = "\
# DRAFT recorded from a production trace. Not auto-loaded by the suite.
# 1. `script` is rebuilt from the responses the daemon SHIPPED, not
#    the raw model output: replies bounced by gates and calls blocked by guards
#    are missing, and daemon-synthesized replies appear as model text.
#    Check the script against the trace (db_probe --task <id>).
# Run the scenario until it passes, then move it into tests/harness_eval/scenarios/.
";

/// A draft scenario, in the format of `aidaemon_eval::Scenario` with the
/// daemon's `setup:` and `expect.ext:` blocks.
#[derive(Debug, Clone, Serialize)]
pub struct RecordedScenario {
    pub name: String,
    pub description: String,
    pub input: String,
    pub setup: RecordedSetup,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub script: Vec<RecordedReply>,
    pub expect: RecordedExpect,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecordedSetup {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecordedReply {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<RecordedCall>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecordedCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecordedExpect {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub final_text_contains_exact: Vec<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub tool_dispatch_counts: BTreeMap<String, u32>,
    pub max_turns: u32,
    pub proposed: RecordedProposed,
    pub ext: RecordedExt,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecordedProposed {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools_in_order: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecordedExt {
    pub orchestration_route: String,
    pub tools_required_predicted: bool,
    pub outcome: String,
    pub stop_reason: String,
    pub tool_calls_max: u32,
    pub forbids_mutation: bool,
    pub forbidden_mutation_attempts_max: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_steps_completed_min: Option<u32>,
    pub routing_accuracy_min: f32,
    pub progress_yield_min: f32,
    pub contract_fulfillment_min: f32,
    pub cost_efficiency_min: f32,
    pub overall_min: f32,
    pub direct_return: bool,
    pub response_fallthrough: bool,
}

/// Build a draft scenario from one task's canonical trace.
///
/// `events` must be in append (`id`) order and may span the whole session;
/// only the selected task's events feed the script and expectations. The task
/// is `task_id` when given, else the last task that recorded a harness-eval
/// snapshot. The execution-model script is rebuilt from `AssistantResponse`
/// events, which are the post-gate projection of the run: they only equal
/// the model's raw script when no gate bounced, blocked, or synthesized a
/// reply. `LlmCall` events do not carry the response, so nothing better is
/// available from the trace today. Expectations are structural and
/// receipt-backed, and score minimums are floored so a replay of the same
/// behavior cannot fail on rounding.
pub fn record_scenario_from_events(
    session_id: &str,
    events: &[Event],
    task_id: Option<&str>,
) -> anyhow::Result<RecordedScenario> {
    let (end_index, task_end) = events
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, event)| event.event_type == EventType::TaskEnd)
        .filter(|(_, event)| task_id.is_none_or(|id| event.task_id.as_deref() == Some(id)))
        .find_map(|(index, event)| {
            let end = event.parse_data::<TaskEndData>().ok()?;
            end.harness_eval.is_some().then_some((index, end))
        })
        .context("no TaskEnd with a harness_eval snapshot for this session/task")?;
    let eval = task_end
        .harness_eval
        .clone()
        .context("TaskEnd missing harness_eval snapshot")?;
    let task_id = task_end.task_id.as_str();
    let in_task = |event: &&Event| event.task_id.as_deref() == Some(task_id);
    let first_task_index = events
        .iter()
        .position(|event| in_task(&event))
        .unwrap_or(end_index);

    // The opening user message may be written before the task identity
    // exists; fall back to the latest one preceding the task's first event.
    let input = events
        .iter()
        .filter(in_task)
        .chain(events[..first_task_index].iter().rev())
        .find(|event| event.event_type == EventType::UserMessage)
        .and_then(|event| event.parse_data::<crate::events::UserMessageData>().ok())
        .map(|message| message.content)
        .or_else(|| task_end.summary.clone())
        .context("could not resolve the task's user message")?;

    let mut script = Vec::new();
    for event in events
        .iter()
        .filter(in_task)
        .filter(|event| event.event_type == EventType::AssistantResponse)
    {
        let data = event
            .parse_data::<crate::events::AssistantResponseData>()
            .with_context(|| format!("parse AssistantResponse event {}", event.id))?;
        let tool_calls = data.tool_calls.unwrap_or_default();
        if tool_calls.is_empty() {
            if let Some(text) = data.content.filter(|text| !text.trim().is_empty()) {
                script.push(RecordedReply {
                    text: Some(text),
                    tool_calls: Vec::new(),
                });
            }
        }
        // The mock serves one tool call per response, so a parallel batch
        // becomes consecutive turns. Order is kept; batching is not.
        for call in tool_calls {
            script.push(RecordedReply {
                text: None,
                tool_calls: vec![RecordedCall {
                    name: call.name,
                    arguments: call.arguments.to_string(),
                }],
            });
        }
    }

    let run = collect_run_result(
        &events
            .iter()
            .filter(|event| in_task(event))
            .cloned()
            .collect::<Vec<_>>(),
        "",
    )?;
    let mut tool_dispatch_counts = BTreeMap::new();
    for name in &run.dispatched_tool_names {
        *tool_dispatch_counts.entry(name.clone()).or_insert(0) += 1;
    }

    Ok(RecordedScenario {
        name: task_id.chars().take(32).collect(),
        description: "Recorded from production run (draft; structural expect only)".to_string(),
        input,
        setup: RecordedSetup {
            session_id: session_id.to_string(),
        },
        script,
        expect: RecordedExpect {
            final_text_contains_exact: Vec::new(),
            tool_dispatch_counts,
            max_turns: run.execution_llm_calls,
            proposed: RecordedProposed {
                tools_in_order: run.tool_names.clone(),
            },
            ext: RecordedExt {
                orchestration_route: eval.orchestration_route.clone(),
                tools_required_predicted: eval.routing.tools_required_predicted,
                outcome: task_end.effective_outcome().as_str().to_string(),
                stop_reason: eval.quality.stop_reason.clone(),
                tool_calls_max: task_end.tool_calls_count,
                forbids_mutation: eval.quality.contract.forbids_mutation,
                forbidden_mutation_attempts_max: eval.quality.contract.forbidden_mutation_attempts,
                plan_steps_completed_min: eval.progress.plan_steps_completed,
                routing_accuracy_min: floor_score(eval.scores.routing_accuracy),
                progress_yield_min: floor_score(eval.scores.progress_yield),
                contract_fulfillment_min: floor_score(eval.scores.contract_fulfillment),
                cost_efficiency_min: floor_score(eval.scores.cost_efficiency),
                overall_min: floor_score(eval.scores.overall),
                direct_return: eval.routing.direct_return_attempted,
                response_fallthrough: eval.routing.response_fallthrough,
            },
        },
    })
}

/// Round down to two decimals. Rounding to nearest could record a minimum
/// above the observed score (0.845 -> 0.85) and fail the identical replay.
fn floor_score(value: f32) -> f32 {
    (value * 100.0).floor() / 100.0
}
