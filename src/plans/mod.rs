//! Persistent Task Plans for reliable multi-step task execution.
//!
//! This module provides a planning system that:
//! - Persists multi-step task state to SQLite
//! - Enables recovery from crashes/restarts mid-task
//! - Provides explicit step-by-step progress tracking
//! - Supports resume capability for long-running tasks

pub mod detection;
mod generation;
mod recovery;
mod store;
mod tracking;

pub use store::PlanStore;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::traits::ToolMutationEffects;
use crate::utils::truncate_str;

/// A persistent multi-step task plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPlan {
    /// Unique identifier (UUID)
    pub id: String,

    /// Session this plan belongs to
    pub session_id: String,

    /// Human-readable description (from user request)
    pub description: String,

    /// The original user message that triggered this plan
    pub trigger_message: String,

    /// Ordered list of steps
    pub steps: Vec<PlanStep>,

    /// Index of current step (0-based)
    pub current_step: usize,

    /// Overall plan status
    pub status: PlanStatus,

    /// Arbitrary checkpoint data (tool outputs, intermediate results)
    pub checkpoint: JsonValue,

    /// Why the plan was created (complexity heuristic result)
    pub creation_reason: String,

    /// Task ID from event store (links plan to events)
    pub task_id: Option<String>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl TaskPlan {
    /// Create a new task plan with the given steps.
    pub fn new(
        session_id: impl Into<String>,
        trigger_message: impl Into<String>,
        description: impl Into<String>,
        steps: Vec<String>,
        creation_reason: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        let plan_steps: Vec<PlanStep> = steps
            .into_iter()
            .enumerate()
            .map(|(index, description)| PlanStep {
                index,
                description,
                depends_on: index.checked_sub(1).into_iter().collect(),
                required_mutation_effects: ToolMutationEffects::NONE,
                requires_observation: false,
                expected_targets: Vec::new(),
                evidence_receipt_ids: Vec::new(),
                status: if index == 0 {
                    StepStatus::InProgress
                } else {
                    StepStatus::Pending
                },
                tool_call_ids: Vec::new(),
                result_summary: None,
                error: None,
                retry_count: 0,
                started_at: if index == 0 { Some(now) } else { None },
                completed_at: None,
            })
            .collect();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.into(),
            description: description.into(),
            trigger_message: trigger_message.into(),
            steps: plan_steps,
            current_step: 0,
            status: PlanStatus::InProgress,
            checkpoint: JsonValue::Object(serde_json::Map::new()),
            creation_reason: creation_reason.into(),
            task_id: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Get the current step, if any.
    pub fn current_step_ref(&self) -> Option<&PlanStep> {
        self.steps.get(self.current_step)
    }

    /// Get the current step mutably, if any.
    pub fn current_step_mut(&mut self) -> Option<&mut PlanStep> {
        self.steps.get_mut(self.current_step)
    }

    /// Keep the user-facing execution pointer aligned with checklist state.
    /// While work remains, exactly one pending step is promoted to in-progress.
    pub(crate) fn sync_active_step(&mut self) {
        if let Some(index) = self
            .steps
            .iter()
            .position(|step| step.status == StepStatus::InProgress)
        {
            self.current_step = index;
            return;
        }

        if let Some(index) = self
            .ready_step_indices()
            .ok()
            .and_then(|indices| indices.into_iter().next())
        {
            self.current_step = index;
            if self.status == PlanStatus::InProgress {
                let step = &mut self.steps[index];
                step.status = StepStatus::InProgress;
                step.started_at.get_or_insert_with(Utc::now);
            }
        } else if self.is_finished() {
            self.status = self.resolved_status();
        }
        self.updated_at = Utc::now();
    }

    /// Calculate total duration in seconds.
    pub fn duration_secs(&self) -> u64 {
        (self.updated_at - self.created_at).num_seconds().max(0) as u64
    }

    /// Count completed steps.
    pub fn completed_steps(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.status == StepStatus::Completed)
            .count()
    }

    /// Check if all steps are done (completed, failed, or skipped).
    pub fn is_finished(&self) -> bool {
        self.steps.iter().all(|s| {
            matches!(
                s.status,
                StepStatus::Completed
                    | StepStatus::Failed
                    | StepStatus::Skipped
                    | StepStatus::Deferred
            )
        })
    }

    /// Derive the terminal plan state from typed step outcomes. A deferred
    /// item is resolved for scheduling purposes but is not successful work.
    pub fn resolved_status(&self) -> PlanStatus {
        if self
            .steps
            .iter()
            .any(|step| step.status == StepStatus::Failed)
        {
            PlanStatus::Failed
        } else if self
            .steps
            .iter()
            .any(|step| step.status == StepStatus::Deferred)
        {
            PlanStatus::PartiallyCompleted
        } else {
            PlanStatus::Completed
        }
    }

    /// Validate the persisted step dependency graph.
    pub fn validate_graph(&self) -> Result<(), String> {
        self.step_graph().map(|_| ())
    }

    /// Pending steps whose prerequisites have reached accepted terminal state.
    /// The current executor selects one deterministically; task-level workers
    /// can use the same graph shape for actual parallel branches.
    pub fn ready_step_indices(&self) -> Result<Vec<usize>, String> {
        let graph = self.step_graph()?;
        Ok(self
            .steps
            .iter()
            .filter(|step| {
                step.status == StepStatus::Pending
                    && graph.dependencies_satisfied(&plan_step_node_id(step.index))
            })
            .map(|step| step.index)
            .collect())
    }

    fn step_graph(&self) -> Result<crate::execution_graph::ExecutionGraph, String> {
        use crate::execution_graph::{
            ExecutionEdgeKind, ExecutionGraph, ExecutionNodeKind, ExecutionNodeState,
        };

        let mut graph = ExecutionGraph::default();
        for (position, step) in self.steps.iter().enumerate() {
            if step.index != position {
                return Err(format!(
                    "plan step index {} does not match position {}",
                    step.index, position
                ));
            }
            let state = match step.status {
                StepStatus::Pending => ExecutionNodeState::Pending,
                StepStatus::InProgress => ExecutionNodeState::Running,
                StepStatus::Completed | StepStatus::Skipped | StepStatus::Deferred => {
                    ExecutionNodeState::Satisfied
                }
                StepStatus::Failed => ExecutionNodeState::Failed,
            };
            graph.add_node(
                plan_step_node_id(step.index),
                ExecutionNodeKind::PlanStep,
                state,
            )?;
        }
        for step in &self.steps {
            let mut unique_dependencies = std::collections::BTreeSet::new();
            for dependency in &step.depends_on {
                if !unique_dependencies.insert(*dependency) {
                    return Err(format!(
                        "plan step {} lists dependency {} more than once",
                        step.index, dependency
                    ));
                }
                if *dependency >= self.steps.len() {
                    return Err(format!(
                        "plan step {} depends on missing step {}",
                        step.index, dependency
                    ));
                }
                graph.add_edge(
                    &plan_step_node_id(step.index),
                    &plan_step_node_id(*dependency),
                    ExecutionEdgeKind::DependsOn,
                    None,
                )?;
            }
        }
        Ok(graph)
    }

    /// Advance to the next step. Returns true if there was a next step.
    pub fn advance_to_next_step(&mut self) -> bool {
        if let Some(next_index) = self
            .ready_step_indices()
            .ok()
            .and_then(|indices| indices.into_iter().next())
        {
            self.current_step = next_index;
            if let Some(step) = self.steps.get_mut(self.current_step) {
                step.status = StepStatus::InProgress;
                step.started_at = Some(Utc::now());
            }
            self.updated_at = Utc::now();
            true
        } else if self.is_finished() {
            self.status = self.resolved_status();
            self.updated_at = Utc::now();
            false
        } else {
            self.updated_at = Utc::now();
            false
        }
    }

    /// Mark the current step as completed with a summary.
    pub fn complete_current_step(&mut self, summary: Option<String>) {
        if let Some(step) = self.current_step_mut() {
            step.status = StepStatus::Completed;
            step.completed_at = Some(Utc::now());
            step.result_summary = summary;
        }
        self.updated_at = Utc::now();
    }

    /// Mark the current step as failed with an error message.
    pub fn fail_current_step(&mut self, error: String) {
        if let Some(step) = self.current_step_mut() {
            step.status = StepStatus::Failed;
            step.completed_at = Some(Utc::now());
            step.error = Some(error);
        }
        self.updated_at = Utc::now();
    }

    /// Increment retry count for current step and reset to in-progress.
    pub fn retry_current_step(&mut self) {
        if let Some(step) = self.current_step_mut() {
            step.retry_count += 1;
            step.status = StepStatus::InProgress;
            step.error = None;
            step.started_at = Some(Utc::now());
            step.completed_at = None;
        }
        self.updated_at = Utc::now();
    }

    /// Steps still requiring action (not completed/deferred/failed/skipped).
    pub fn unchecked_steps(&self) -> Vec<&PlanStep> {
        self.steps
            .iter()
            .filter(|s| matches!(s.status, StepStatus::Pending | StepStatus::InProgress))
            .collect()
    }

    /// Compact, channel-friendly checklist for posting to the user at start.
    pub fn render_compact_checklist(&self) -> String {
        let (heading, footer) = match self.status {
            PlanStatus::Planning => (
                "🧭 Planning your task",
                "I’ll update this message as I work.",
            ),
            PlanStatus::InProgress => (
                "⚙️ Working on your task",
                "I’ll update this message as I work.",
            ),
            PlanStatus::Paused => (
                "⏸️ Task paused",
                "Work will continue when the task resumes.",
            ),
            // A completed checklist means the execution plan is resolved. It
            // is not the durable task terminal: the final assistant response
            // and TaskEnd event are still outstanding at this point.
            PlanStatus::Completed => (
                "✅ Planned steps complete",
                "Finalizing the response and task outcome…",
            ),
            PlanStatus::PartiallyCompleted => (
                "⚠️ Plan ended with unresolved steps",
                "The final response must explain each deferred item and its blocker.",
            ),
            PlanStatus::Failed => ("❌ Task stopped", "The plan could not be completed."),
            PlanStatus::Abandoned => ("⏹️ Task cancelled", "No more work is in progress."),
        };
        let total = self.steps.len();
        let position = self.current_step.saturating_add(1).min(total);
        let mut out = format!("{heading}\n\n");
        if self.status == PlanStatus::InProgress && total > 0 {
            out.push_str(&format!("Step {position} of {total}\n"));
        } else {
            out.push_str(&format!(
                "{} of {total} steps complete\n",
                self.completed_steps()
            ));
        }

        let first_pending = self
            .steps
            .iter()
            .position(|step| step.status == StepStatus::Pending);
        for (index, step) in self.steps.iter().enumerate() {
            let block = match step.status {
                StepStatus::Completed => format!("✅ {}", step.description),
                StepStatus::InProgress => format!("▶️ Working now\n{}", step.description),
                StepStatus::Deferred => format!("⏭️ Deferred\n{}", step.description),
                StepStatus::Failed => format!("⚠️ Needs attention\n{}", step.description),
                StepStatus::Skipped => format!("➖ Skipped\n{}", step.description),
                StepStatus::Pending if Some(index) == first_pending => {
                    format!("○ Up next\n{}", step.description)
                }
                StepStatus::Pending => format!("○ {}", step.description),
            };
            out.push_str(&format!("\n{block}\n"));
        }
        out.push_str(&format!("\n{footer}"));
        out
    }

    /// End-of-turn recap of what got done vs what remains/deferred.
    pub fn render_recap(&self) -> String {
        let mut out = String::new();
        for step in &self.steps {
            let mark = match step.status {
                StepStatus::Completed => "✅",
                StepStatus::Deferred => "⏭️ (deferred)",
                StepStatus::Failed => "⚠️ (failed)",
                _ => "☐ (not done)",
            };
            out.push_str(&format!("{} {}\n", mark, step.description));
        }
        out.trim_end().to_string()
    }

    /// Format the plan for system prompt injection.
    pub fn format_for_prompt(&self) -> String {
        let mut lines = vec![
            "## Incomplete Task Plan".to_string(),
            format!("**Task:** {}", self.description),
            format!(
                "**Status:** {} ({}/{} steps complete)",
                self.status.as_str(),
                self.completed_steps(),
                self.steps.len()
            ),
            "**Steps:**".to_string(),
        ];

        for step in &self.steps {
            let marker = match step.status {
                StepStatus::Completed => "[x]",
                StepStatus::InProgress => "[>]",
                StepStatus::Failed => "[!]",
                StepStatus::Skipped => "[-]",
                StepStatus::Deferred => "[~]",
                StepStatus::Pending => "[ ]",
            };
            let mut line = format!("  {} {}. {}", marker, step.index + 1, step.description);
            if let Some(ref error) = step.error {
                line.push_str(&format!(" (Error: {})", truncate_str(error, 50)));
            }
            lines.push(line);
        }

        lines.push(String::new());
        lines.push(
            "If the user wants to continue this task, resume from the current step.".to_string(),
        );
        lines.push(
            "If the user provides NEW REQUIREMENTS or CHANGES (e.g., 'use X instead', 'also add Y'), \
             use plan_manager with action='revise' to update the remaining steps accordingly."
                .to_string(),
        );
        lines.push(
            "If they want to start over or do something else, use plan_manager to abandon it first."
                .to_string(),
        );

        lines.join("\n")
    }
}

/// A single step in a task plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Step index (0-based)
    pub index: usize,

    /// Human-readable description
    pub description: String,

    /// Indices of prerequisite steps. Missing on legacy rows means the steps
    /// are independent; newly generated plans persist explicit serial edges.
    #[serde(default)]
    pub depends_on: Vec<usize>,

    /// Typed evidence contract for automatic completion. If empty/false, only
    /// an explicit checklist update may complete the step.
    #[serde(default)]
    pub required_mutation_effects: ToolMutationEffects,
    #[serde(default)]
    pub requires_observation: bool,
    /// Exact canonical targets (path, URL, or resource ID), never prose.
    #[serde(default)]
    pub expected_targets: Vec<String>,
    /// Tool-call receipt IDs that proved this step.
    #[serde(default)]
    pub evidence_receipt_ids: Vec<String>,

    /// Step status
    pub status: StepStatus,

    /// Tool calls made during this step (event IDs or tool:timestamp)
    pub tool_call_ids: Vec<String>,

    /// Summary of what was accomplished
    pub result_summary: Option<String>,

    /// Error message if step failed
    pub error: Option<String>,

    /// Number of retry attempts
    pub retry_count: u32,

    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct ChecklistItem {
    pub description: String,
    pub status: StepStatus,
    pub depends_on: Vec<usize>,
    pub required_mutation_effects: ToolMutationEffects,
    pub requires_observation: bool,
    pub expected_targets: Vec<String>,
}

fn plan_step_node_id(index: usize) -> String {
    format!("plan-step:{index}")
}

impl PlanStep {
    /// Calculate step duration in seconds (if completed).
    pub fn duration_secs(&self) -> Option<u64> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some((end - start).num_seconds().max(0) as u64),
            _ => None,
        }
    }
}

/// Overall plan status.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    strum::IntoStaticStr,
    strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum PlanStatus {
    /// LLM is generating steps
    Planning,
    /// Executing steps
    InProgress,
    /// Paused (session ended, waiting for user, or interrupted)
    Paused,
    /// All steps completed successfully
    Completed,
    /// Scheduling is resolved, but one or more requested steps were deferred.
    PartiallyCompleted,
    /// Unrecoverable failure
    Failed,
    /// User explicitly cancelled
    Abandoned,
}

impl PlanStatus {
    pub fn as_str(&self) -> &'static str {
        (*self).into()
    }

    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }

    /// Check if this status represents an incomplete/active plan.
    pub fn is_incomplete(&self) -> bool {
        matches!(
            self,
            PlanStatus::Planning | PlanStatus::InProgress | PlanStatus::Paused
        )
    }
}

/// Status of a single step.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    strum::IntoStaticStr,
    strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum StepStatus {
    /// Not yet started
    Pending,
    /// Currently executing
    InProgress,
    /// Completed successfully
    Completed,
    /// Failed (may retry)
    Failed,
    /// Skipped (dependency failed or user skipped)
    Skipped,
    /// Explicitly abandoned by the model (counts as resolved, not unchecked).
    Deferred,
}

impl StepStatus {
    pub fn as_str(&self) -> &'static str {
        (*self).into()
    }

    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_plan() {
        let plan = TaskPlan::new(
            "session_123",
            "Deploy the app to production",
            "Production deployment",
            vec![
                "Run tests".to_string(),
                "Build container".to_string(),
                "Deploy".to_string(),
            ],
            "high_stakes",
        );

        assert_eq!(plan.steps.len(), 3);
        assert_eq!(plan.current_step, 0);
        assert_eq!(plan.status, PlanStatus::InProgress);
        assert_eq!(plan.steps[0].status, StepStatus::InProgress);
        assert_eq!(plan.steps[1].status, StepStatus::Pending);
        assert_eq!(plan.steps[2].status, StepStatus::Pending);
    }

    #[test]
    fn test_advance_steps() {
        let mut plan = TaskPlan::new(
            "session_123",
            "test",
            "Test task",
            vec!["Step 1".to_string(), "Step 2".to_string()],
            "test",
        );

        plan.complete_current_step(Some("Done".to_string()));
        assert_eq!(plan.steps[0].status, StepStatus::Completed);

        assert!(plan.advance_to_next_step());
        assert_eq!(plan.current_step, 1);
        assert_eq!(plan.steps[1].status, StepStatus::InProgress);

        plan.complete_current_step(None);
        assert!(!plan.advance_to_next_step());
        assert_eq!(plan.status, PlanStatus::Completed);
    }

    #[test]
    fn sync_active_step_promotes_the_next_pending_item() {
        let mut plan = TaskPlan::new(
            "session_123",
            "test",
            "Test task",
            vec!["First".to_string(), "Second".to_string()],
            "test",
        );
        plan.steps[0].status = StepStatus::Completed;
        plan.steps[1].status = StepStatus::Pending;
        plan.current_step = 0;

        plan.sync_active_step();

        assert_eq!(plan.current_step, 1);
        assert_eq!(plan.steps[1].status, StepStatus::InProgress);
    }

    #[test]
    fn dependency_graph_supports_parallel_branches_and_a_join() {
        let mut plan = TaskPlan::new(
            "session_123",
            "test",
            "Graph task",
            vec!["Left".into(), "Right".into(), "Join".into()],
            "test",
        );
        for step in &mut plan.steps {
            step.status = StepStatus::Pending;
        }
        plan.steps[0].depends_on.clear();
        plan.steps[1].depends_on.clear();
        plan.steps[2].depends_on = vec![0, 1];

        assert_eq!(plan.ready_step_indices().unwrap(), vec![0, 1]);
        plan.steps[0].status = StepStatus::Completed;
        assert_eq!(plan.ready_step_indices().unwrap(), vec![1]);
        plan.steps[1].status = StepStatus::Completed;
        assert_eq!(plan.ready_step_indices().unwrap(), vec![2]);
    }

    #[test]
    fn dependency_graph_rejects_cycles() {
        let mut plan = TaskPlan::new(
            "session_123",
            "test",
            "Cyclic task",
            vec!["One".into(), "Two".into()],
            "test",
        );
        plan.steps[0].depends_on = vec![1];
        plan.steps[1].depends_on = vec![0];

        assert!(plan.validate_graph().unwrap_err().contains("cycle"));
    }

    #[test]
    fn test_format_for_prompt() {
        let mut plan = TaskPlan::new(
            "session_123",
            "Deploy",
            "Deploy to production",
            vec![
                "Run tests".to_string(),
                "Build".to_string(),
                "Deploy".to_string(),
            ],
            "test",
        );

        plan.complete_current_step(Some("Tests passed".to_string()));
        plan.advance_to_next_step();

        let formatted = plan.format_for_prompt();
        assert!(formatted.contains("Incomplete Task Plan"));
        assert!(formatted.contains("[x] 1. Run tests"));
        assert!(formatted.contains("[>] 2. Build"));
        assert!(formatted.contains("[ ] 3. Deploy"));
    }

    #[test]
    fn test_deferred_status_round_trip() {
        assert_eq!(StepStatus::Deferred.as_str(), "deferred");
        assert_eq!(StepStatus::from_str("deferred"), Some(StepStatus::Deferred));
    }

    #[test]
    fn test_unchecked_steps_excludes_completed_and_deferred() {
        let mut plan = TaskPlan::new(
            "s",
            "t",
            "d",
            vec!["a".into(), "b".into(), "c".into()],
            "track_requirements",
        );
        plan.steps[0].status = StepStatus::Completed;
        plan.steps[1].status = StepStatus::Deferred;
        plan.steps[2].status = StepStatus::Pending;
        let unchecked = plan.unchecked_steps();
        assert_eq!(unchecked.len(), 1);
        assert_eq!(unchecked[0].description, "c");
    }

    #[test]
    fn test_render_compact_checklist_and_recap() {
        let mut plan = TaskPlan::new(
            "s",
            "t",
            "d",
            vec!["create script".into(), "send the file".into()],
            "track_requirements",
        );
        plan.steps[0].status = StepStatus::Completed;
        plan.steps[1].status = StepStatus::InProgress;
        plan.current_step = 1;
        let checklist = plan.render_compact_checklist();
        assert!(checklist.starts_with("⚙️ Working on your task"));
        assert!(checklist.contains("Step 2 of 2\n\n"));
        assert!(checklist.contains("✅ create script"));
        assert!(checklist.contains("▶️ Working now\nsend the file"));
        assert!(checklist.contains("I’ll update this message as I work."));
        assert!(checklist.contains("create script"));
        assert!(checklist.contains("send the file"));
        let recap = plan.render_recap();
        assert!(recap.contains("create script"));
        assert!(recap.contains("send the file"));
        assert!(recap.contains("not done"));
    }

    #[test]
    fn completed_plan_does_not_announce_durable_task_completion() {
        let mut plan = TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            vec!["observe result".into()],
            "track_requirements",
        );
        plan.steps[0].status = StepStatus::Completed;
        plan.status = PlanStatus::Completed;

        let checklist = plan.render_compact_checklist();
        assert!(checklist.starts_with("✅ Planned steps complete"));
        assert!(checklist.contains("Finalizing the response and task outcome"));
        assert!(!checklist.contains("Task complete"));
    }

    #[test]
    fn deferred_steps_resolve_to_partial_plan_with_visible_blocker_requirement() {
        let mut plan = TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            vec!["first action".into(), "second action".into()],
            "track_requirements",
        );
        for step in &mut plan.steps {
            step.status = StepStatus::Deferred;
        }
        plan.sync_active_step();

        assert_eq!(plan.status, PlanStatus::PartiallyCompleted);
        let checklist = plan.render_compact_checklist();
        assert!(checklist.starts_with("⚠️ Plan ended with unresolved steps"));
        assert!(checklist.contains("explain each deferred item and its blocker"));
        assert!(!checklist.contains("Planned steps complete"));
    }

    #[test]
    fn test_plan_status_is_incomplete() {
        assert!(PlanStatus::Planning.is_incomplete());
        assert!(PlanStatus::InProgress.is_incomplete());
        assert!(PlanStatus::Paused.is_incomplete());
        assert!(!PlanStatus::Completed.is_incomplete());
        assert!(!PlanStatus::PartiallyCompleted.is_incomplete());
        assert!(!PlanStatus::Failed.is_incomplete());
        assert!(!PlanStatus::Abandoned.is_incomplete());
    }
}
