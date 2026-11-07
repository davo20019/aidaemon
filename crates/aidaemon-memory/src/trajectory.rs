//! Learning from what tasks did: a host's task history as typed trace
//! events, and the procedures, error solutions and expertise memory learns
//! from them.

use std::collections::HashMap;
use std::sync::Arc;

use aidaemon_core::{Message, ModelRequest, Provider};
use chrono::Utc;
use tracing::warn;

use crate::learning::{error_pattern, ErrorSolution, Procedure};

/// How a task ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskEnding {
    Succeeded,
    Partial,
    Failed,
}

/// One event of a session's task history, as the host recorded it.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceEvent {
    TaskStarted {
        task_id: String,
        /// The request the task serves.
        description: String,
    },
    ToolCalled {
        task_id: Option<String>,
        name: String,
        /// A short description of the call's arguments.
        summary: Option<String>,
    },
    ToolReturned {
        task_id: Option<String>,
        name: String,
        succeeded: bool,
        result: String,
    },
    /// An error the run hit (with the tool involved, if any).
    Failed {
        task_id: Option<String>,
        message: String,
        tool: Option<String>,
    },
    TaskEnded {
        task_id: String,
        outcome: TaskEnding,
        duration_secs: f64,
    },
}

impl TraceEvent {
    fn task_id(&self) -> Option<&str> {
        match self {
            Self::TaskStarted { task_id, .. } | Self::TaskEnded { task_id, .. } => Some(task_id),
            Self::ToolCalled { task_id, .. }
            | Self::ToolReturned { task_id, .. }
            | Self::Failed { task_id, .. } => task_id.as_deref(),
        }
    }
}

/// A successful task with at least two tool calls: a candidate procedure.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcedureCandidate {
    pub task_id: String,
    pub description: String,
    /// `tool: summary`, one per call, in order.
    pub steps: Vec<String>,
    pub duration_secs: f64,
}

/// The successful tasks in `trace` worth turning into procedures, in the
/// order they started.
pub fn procedure_candidates(trace: &[TraceEvent]) -> Vec<ProcedureCandidate> {
    let mut order: Vec<&str> = Vec::new();
    let mut by_task: HashMap<&str, Vec<&TraceEvent>> = HashMap::new();
    for event in trace {
        if let Some(task_id) = event.task_id() {
            if !by_task.contains_key(task_id) {
                order.push(task_id);
            }
            by_task.entry(task_id).or_default().push(event);
        }
    }
    let mut candidates = Vec::new();
    for task_id in order {
        let events = &by_task[task_id];
        let description = events.iter().find_map(|event| match event {
            TraceEvent::TaskStarted { description, .. } => Some(description.clone()),
            _ => None,
        });
        let ending = events.iter().find_map(|event| match event {
            TraceEvent::TaskEnded {
                outcome,
                duration_secs,
                ..
            } => Some((*outcome, *duration_secs)),
            _ => None,
        });
        let (Some(description), Some((TaskEnding::Succeeded, duration_secs))) =
            (description, ending)
        else {
            continue;
        };
        let steps: Vec<String> = events
            .iter()
            .filter_map(|event| match event {
                TraceEvent::ToolCalled { name, summary, .. } => Some(format!(
                    "{name}: {}",
                    summary.clone().unwrap_or_else(|| "...".to_string())
                )),
                _ => None,
            })
            .collect();
        if steps.len() >= 2 {
            candidates.push(ProcedureCandidate {
                task_id: task_id.to_string(),
                description,
                steps,
                duration_secs,
            });
        }
    }
    candidates
}

/// A procedure observed once, from a successful task.
pub fn procedure_from_task(candidate: &ProcedureCandidate) -> Procedure {
    let now = Utc::now();
    let short_id: String = candidate.task_id.chars().take(8).collect();
    Procedure {
        id: 0,
        name: format!("auto_{short_id}"),
        trigger_pattern: candidate.description.clone(),
        steps: candidate.steps.clone(),
        success_count: 1,
        failure_count: 0,
        avg_duration_secs: Some(candidate.duration_secs as f32),
        last_used_at: Some(now),
        created_at: now,
        updated_at: now,
    }
}

/// A procedure from a completed plan's finished steps, or `None` for fewer
/// than two.
pub fn procedure_from_plan(
    plan_id: &str,
    description: &str,
    completed_steps: Vec<String>,
) -> Option<Procedure> {
    if completed_steps.len() < 2 {
        return None;
    }
    let now = Utc::now();
    let short_id: String = plan_id.chars().take(8).collect();
    Some(Procedure {
        id: 0,
        name: format!("plan_{short_id}"),
        trigger_pattern: description.to_string(),
        steps: completed_steps,
        success_count: 1,
        failure_count: 0,
        avg_duration_secs: None,
        last_used_at: Some(now),
        created_at: now,
        updated_at: now,
    })
}

/// Enough evidence to keep a procedure: at least two steps, observed
/// working, never failing.
pub fn meets_evidence_gate(procedure: &Procedure) -> bool {
    procedure.steps.len() >= 2 && procedure.failure_count == 0 && procedure.success_count >= 1
}

/// A procedure is a duplicate when an existing trigger is this similar.
const DUPLICATE_SIMILARITY: f32 = 0.85;

/// Whether a new procedure's trigger embedding duplicates an existing one.
pub fn duplicates_existing_trigger(trigger: &[f32], existing: &[Vec<f32>]) -> bool {
    existing
        .iter()
        .any(|other| crate::embed::cosine(trigger, other) > DUPLICATE_SIMILARITY)
}

/// Error-solution pairs: an error followed, within the next nine events,
/// by a successful tool result. A later success is a correlation, not
/// proof, so a pair starts with no credit; it earns it on verified reuse.
pub fn error_solutions(trace: &[TraceEvent]) -> Vec<ErrorSolution> {
    let mut solutions = Vec::new();
    for (index, event) in trace.iter().enumerate() {
        let TraceEvent::Failed { message, tool, .. } = event else {
            continue;
        };
        let recovery = trace
            .iter()
            .take(trace.len().min(index + 10))
            .skip(index + 1)
            .find_map(|next| match next {
                TraceEvent::ToolReturned {
                    name,
                    succeeded: true,
                    result,
                    ..
                } => Some((name.clone(), truncate(result, 200))),
                _ => None,
            });
        if let Some((recovered_with, summary)) = recovery {
            let now = Utc::now();
            solutions.push(ErrorSolution {
                id: 0,
                error_pattern: error_pattern(message),
                domain: tool.clone(),
                solution_summary: format!("Used {recovered_with} to resolve"),
                solution_steps: Some(vec![summary]),
                success_count: 0,
                failure_count: 0,
                last_used_at: Some(now),
                created_at: now,
            });
        }
    }
    solutions
}

/// One expertise observation per ended task: the domain its tools belong
/// to, and whether it succeeded.
pub fn expertise_updates(trace: &[TraceEvent]) -> Vec<(String, bool)> {
    trace
        .iter()
        .filter_map(|event| match event {
            TraceEvent::TaskEnded {
                task_id, outcome, ..
            } => task_domain(task_id, trace)
                .map(|domain| (domain, *outcome == TaskEnding::Succeeded)),
            _ => None,
        })
        .collect()
}

/// The domain of a task, from the tools it called.
fn task_domain(task_id: &str, trace: &[TraceEvent]) -> Option<String> {
    let tools: Vec<&str> = trace
        .iter()
        .filter_map(|event| match event {
            TraceEvent::ToolCalled {
                task_id: Some(id),
                name,
                ..
            } if id == task_id => Some(name.as_str()),
            _ => None,
        })
        .collect();
    if tools.is_empty() {
        return None;
    }
    Some(
        if tools.contains(&"terminal") {
            if tools.iter().any(|tool| tool.contains("git")) {
                "git"
            } else {
                "shell"
            }
        } else if tools
            .iter()
            .any(|tool| *tool == "web_search" || *tool == "web_fetch")
        {
            "research"
        } else if tools.contains(&"browser") {
            "browser"
        } else {
            "general"
        }
        .to_string(),
    )
}

/// Rewrites an observed procedure into a reusable one with a model: a
/// purpose name, a general trigger and clean steps.
#[derive(Clone)]
pub struct ProcedureEnhancer {
    provider: Arc<dyn Provider>,
    model: String,
}

impl ProcedureEnhancer {
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }

    /// The enhanced procedure, or `None` when the model fails, declines it
    /// as too specific (`skip`), or returns no name or trigger. Steps whose
    /// count differs from the original by more than one are not trusted;
    /// the original steps are kept.
    pub async fn enhance(
        &self,
        procedure: &Procedure,
        task_description: &str,
    ) -> Option<Procedure> {
        let tool_sequence = procedure
            .steps
            .iter()
            .enumerate()
            .map(|(i, step)| format!("{}. {step}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let duration = procedure
            .avg_duration_secs
            .map(|d| format!("{d:.0}"))
            .unwrap_or_else(|| "unknown".to_string());
        let step_count = procedure.steps.len();
        let user_prompt = format!(
            "Given a task that completed successfully, generate a clean procedure definition.\n\n\
             Task: {task_description}\n\
             Tools used:\n{tool_sequence}\n\
             Duration: {duration}s\n\n\
             Respond with JSON:\n\
             {{\n\
               \"name\": \"descriptive-kebab-case-name\",\n\
               \"trigger_pattern\": \"general description of when to use this\",\n\
               \"steps\": [\"Step 1: ...\", \"Step 2: ...\"],\n\
               \"skip\": false\n\
             }}\n\n\
             Rules:\n\
             - name: 2-4 word kebab-case describing PURPOSE (not tools used)\n\
             - trigger_pattern: General description, not the exact task\n\
             - steps: Actionable instructions including tool names and key parameters\n\
             - skip: true if too context-specific to be reusable\n\
             - Step count must match original ({step_count} steps)"
        );
        let request = ModelRequest {
            model: self.model.clone(),
            messages: vec![
                Message::system(
                    "You generate clean procedure definitions from successful task sequences.\nRespond with ONLY valid JSON.",
                ),
                Message::user(user_prompt),
            ],
            ..Default::default()
        };
        let response = match self.provider.complete(&request).await {
            Ok(response) => response,
            Err(e) => {
                warn!("LLM enhancement failed for procedure: {e}");
                return None;
            }
        };
        let text = response.text.trim();
        let json = match (text.find('{'), text.rfind('}')) {
            (Some(start), Some(end)) if end >= start => &text[start..=end],
            _ => text,
        };
        let parsed: serde_json::Value = match serde_json::from_str(json) {
            Ok(value) => value,
            Err(e) => {
                warn!("Failed to parse LLM procedure enhancement: {e}");
                return None;
            }
        };
        if parsed
            .get("skip")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return None;
        }
        let text_field = |key: &str| {
            parsed
                .get(key)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let (name, trigger_pattern) = (text_field("name"), text_field("trigger_pattern"));
        if name.is_empty() || trigger_pattern.is_empty() {
            return None;
        }
        let steps: Vec<String> = parsed
            .get("steps")
            .and_then(serde_json::Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let steps = if steps.is_empty() || steps.len().abs_diff(procedure.steps.len()) > 1 {
            procedure.steps.clone()
        } else {
            steps
        };
        Some(Procedure {
            name,
            trigger_pattern,
            steps,
            ..procedure.clone()
        })
    }
}

/// `s` cut to at most `max_len` bytes on a character boundary, ending in
/// "..." when cut.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    let target = max_len.saturating_sub(3);
    let end = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= target)
        .last()
        .unwrap_or(0);
    format!("{}...", &s[..end])
}

#[cfg(test)]
mod tests {
    use aidaemon_core::testing::{text, ScriptedProvider};

    use super::*;

    fn task(task_id: &str, outcome: TaskEnding, tools: &[&str]) -> Vec<TraceEvent> {
        let mut trace = vec![TraceEvent::TaskStarted {
            task_id: task_id.into(),
            description: format!("Do {task_id}"),
        }];
        for tool in tools {
            trace.push(TraceEvent::ToolCalled {
                task_id: Some(task_id.into()),
                name: (*tool).into(),
                summary: Some("args".into()),
            });
        }
        trace.push(TraceEvent::TaskEnded {
            task_id: task_id.into(),
            outcome,
            duration_secs: 12.0,
        });
        trace
    }

    #[test]
    fn only_successful_multi_step_tasks_become_procedures() {
        let mut trace = task(
            "task-ok-1234",
            TaskEnding::Succeeded,
            &["terminal", "edit_file"],
        );
        trace.extend(task(
            "task-failed",
            TaskEnding::Failed,
            &["terminal", "edit_file"],
        ));
        trace.extend(task("task-short", TaskEnding::Succeeded, &["terminal"]));
        let candidates = procedure_candidates(&trace);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].steps, ["terminal: args", "edit_file: args"]);
        let procedure = procedure_from_task(&candidates[0]);
        assert_eq!(procedure.name, "auto_task-ok-");
        assert!(meets_evidence_gate(&procedure));
    }

    #[test]
    fn a_recovered_error_becomes_an_uncredited_solution() {
        let trace = vec![
            TraceEvent::Failed {
                task_id: None,
                message: "connection refused at /srv/app/db.rs:10:4".into(),
                tool: Some("terminal".into()),
            },
            TraceEvent::ToolReturned {
                task_id: None,
                name: "terminal".into(),
                succeeded: false,
                result: "still down".into(),
            },
            TraceEvent::ToolReturned {
                task_id: None,
                name: "run_command".into(),
                succeeded: true,
                result: "database started".into(),
            },
        ];
        let solutions = error_solutions(&trace);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].solution_summary, "Used run_command to resolve");
        assert!(solutions[0].error_pattern.contains("<path>"));
        assert_eq!(solutions[0].success_count, 0);
    }

    #[test]
    fn expertise_follows_the_tools_a_task_used() {
        let mut trace = task("t1", TaskEnding::Succeeded, &["terminal", "git_commit"]);
        trace.extend(task("t2", TaskEnding::Failed, &["web_search"]));
        assert_eq!(
            expertise_updates(&trace),
            [("git".to_string(), true), ("research".to_string(), false)]
        );
    }

    #[test]
    fn plans_and_duplicates() {
        assert!(procedure_from_plan("plan-1", "deploy", vec!["build".into()]).is_none());
        let plan = procedure_from_plan(
            "plan-123456789",
            "deploy",
            vec!["build".into(), "ship".into()],
        )
        .unwrap();
        assert_eq!(plan.name, "plan_plan-123");
        assert!(duplicates_existing_trigger(&[1.0, 0.0], &[vec![1.0, 0.01]]));
        assert!(!duplicates_existing_trigger(&[1.0, 0.0], &[vec![0.0, 1.0]]));
    }

    #[tokio::test]
    async fn enhancement_keeps_original_steps_when_the_count_drifts() {
        let original = procedure_from_task(
            &procedure_candidates(&task(
                "t1",
                TaskEnding::Succeeded,
                &["terminal", "edit_file"],
            ))[0],
        );
        let reply = r#"{"name":"fix-build","trigger_pattern":"when the build breaks",
            "steps":["a","b","c","d"],"skip":false}"#;
        let enhanced =
            ProcedureEnhancer::new(Arc::new(ScriptedProvider::new([text(reply)])), "mock")
                .enhance(&original, "Do t1")
                .await
                .unwrap();
        assert_eq!(enhanced.name, "fix-build");
        assert_eq!(enhanced.steps, original.steps);
        let skipped = ProcedureEnhancer::new(
            Arc::new(ScriptedProvider::new([text(r#"{"skip":true}"#)])),
            "mock",
        )
        .enhance(&original, "Do t1")
        .await;
        assert!(skipped.is_none());
    }
}
