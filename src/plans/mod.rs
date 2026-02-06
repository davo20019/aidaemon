//! Persistent Task Plans for reliable multi-step task execution.
//!
//! This module provides a planning system that:
//! - Persists multi-step task state to SQLite
//! - Enables recovery from crashes/restarts mid-task
//! - Provides explicit step-by-step progress tracking
//! - Supports resume capability for long-running tasks

mod store;
mod detection;
mod generation;
mod tracking;
mod recovery;

pub use store::PlanStore;
pub use detection::{should_create_plan, get_plan_suggestion_prompt};
pub use generation::generate_plan_steps;
pub use tracking::StepTracker;
pub use recovery::PlanRecovery;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

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
                StepStatus::Completed | StepStatus::Failed | StepStatus::Skipped
            )
        })
    }

    /// Advance to the next step. Returns true if there was a next step.
    pub fn advance_to_next_step(&mut self) -> bool {
        if self.current_step + 1 < self.steps.len() {
            self.current_step += 1;
            if let Some(step) = self.steps.get_mut(self.current_step) {
                step.status = StepStatus::InProgress;
                step.started_at = Some(Utc::now());
            }
            self.updated_at = Utc::now();
            true
        } else {
            self.status = PlanStatus::Completed;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    /// LLM is generating steps
    Planning,
    /// Executing steps
    InProgress,
    /// Paused (session ended, waiting for user, or interrupted)
    Paused,
    /// All steps completed successfully
    Completed,
    /// Unrecoverable failure
    Failed,
    /// User explicitly cancelled
    Abandoned,
}

impl PlanStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            PlanStatus::Planning => "planning",
            PlanStatus::InProgress => "in_progress",
            PlanStatus::Paused => "paused",
            PlanStatus::Completed => "completed",
            PlanStatus::Failed => "failed",
            PlanStatus::Abandoned => "abandoned",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "planning" => Some(PlanStatus::Planning),
            "in_progress" => Some(PlanStatus::InProgress),
            "paused" => Some(PlanStatus::Paused),
            "completed" => Some(PlanStatus::Completed),
            "failed" => Some(PlanStatus::Failed),
            "abandoned" => Some(PlanStatus::Abandoned),
            _ => None,
        }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
}

impl StepStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            StepStatus::Pending => "pending",
            StepStatus::InProgress => "in_progress",
            StepStatus::Completed => "completed",
            StepStatus::Failed => "failed",
            StepStatus::Skipped => "skipped",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(StepStatus::Pending),
            "in_progress" => Some(StepStatus::InProgress),
            "completed" => Some(StepStatus::Completed),
            "failed" => Some(StepStatus::Failed),
            "skipped" => Some(StepStatus::Skipped),
            _ => None,
        }
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
    fn test_plan_status_is_incomplete() {
        assert!(PlanStatus::Planning.is_incomplete());
        assert!(PlanStatus::InProgress.is_incomplete());
        assert!(PlanStatus::Paused.is_incomplete());
        assert!(!PlanStatus::Completed.is_incomplete());
        assert!(!PlanStatus::Failed.is_incomplete());
        assert!(!PlanStatus::Abandoned.is_incomplete());
    }
}
