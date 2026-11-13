//! Persistent Task Plans for reliable multi-step task execution.
//!
//! This module provides a planning system that:
//! - Persists multi-step task state to SQLite
//! - Keeps a task's checklist across a restart (the resume itself is built
//!   from the event log, see `agent/runtime/resume.rs`)
//! - Provides explicit step-by-step progress tracking
//! - Supports resume capability for long-running tasks

pub mod checklist_args;
pub mod detection;
mod store;
#[allow(dead_code)]
mod tracking;

pub use store::{migrate_task_plans, PlanStore};

/// Start of the user-facing render in this tool's output. The agent loop
/// locates it to lift the checklist onto the live status surface, so any
/// model-only notice must precede it.
pub const CHECKLIST_HEADER: &str = "Checklist updated";

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::traits::ToolMutationEffects;
pub use aidaemon_core::card::StepEvidence;
use aidaemon_events::text::truncate_str;

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
                background_jobs: Vec::new(),
                waits_on_jobs: false,
                not_done: None,
                evidence: StepEvidence::Settled,
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
    pub fn sync_active_step(&mut self) {
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

    /// Completed action steps no receipt verified: the only steps the card
    /// flags ([`PlanStep::unverified`]).
    pub fn unverified_steps(&self) -> usize {
        self.steps.iter().filter(|s| s.unverified()).count()
    }

    /// The steps at `indices` were confirmed by typed evidence (the run's
    /// receipts proved their checklist items): no longer said done only.
    /// Returns whether any step changed.
    pub fn confirm_steps(&mut self, indices: &std::collections::BTreeSet<usize>) -> bool {
        let mut changed = false;
        for step in &mut self.steps {
            if step.evidence == StepEvidence::ModelSaid && indices.contains(&step.index) {
                step.evidence = StepEvidence::Settled;
                changed = true;
            }
        }
        if changed {
            self.updated_at = Utc::now();
        }
        changed
    }

    /// The helper meant for the steps at `indices` never started (its
    /// `spawn_agent` call failed): nothing works on them, so they are not
    /// done, and the card says they didn't run. A step already done or set
    /// aside is left alone. Returns whether any step changed.
    pub fn helper_did_not_start(&mut self, indices: &[usize], reason: &str) -> bool {
        let now = Utc::now();
        let mut changed = false;
        for &index in indices {
            let Some(step) = self.steps.get_mut(index) else {
                continue;
            };
            if matches!(
                step.status,
                StepStatus::Completed | StepStatus::Skipped | StepStatus::Deferred
            ) || (step.status == StepStatus::Background && !step.background_jobs.is_empty())
            {
                continue;
            }
            step.status = StepStatus::Failed;
            step.evidence = StepEvidence::HelperNotStarted;
            step.error = Some(reason.to_string());
            step.completed_at = Some(now);
            changed = true;
        }
        if changed {
            if self.is_finished() {
                self.status = self.resolved_status();
            } else {
                self.sync_active_step();
            }
            self.updated_at = now;
        }
        changed
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

    fn step_graph(&self) -> Result<aidaemon_events::execution_graph::ExecutionGraph, String> {
        use aidaemon_events::execution_graph::{
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
                StepStatus::InProgress | StepStatus::Background => ExecutionNodeState::Running,
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

    /// Whether a background job is still working on any step.
    pub fn has_background_steps(&self) -> bool {
        self.steps
            .iter()
            .any(|step| step.status == StepStatus::Background)
    }

    /// Hand the steps at `indices` to the background job `job_id`. A step
    /// owns its runs in order and its outcome is its latest run's: a step
    /// an earlier run failed or stopped short on is taken over by this one
    /// (a retry), and only this job's end settles it. A step already done
    /// or skipped is left alone. Returns whether any step changed.
    pub fn link_background_job(&mut self, indices: &[usize], job_id: &str) -> bool {
        let mut changed = false;
        for &index in indices {
            let Some(step) = self.steps.get_mut(index) else {
                continue;
            };
            if matches!(step.status, StepStatus::Completed | StepStatus::Skipped) {
                continue;
            }
            step.evidence = StepEvidence::Settled;
            step.error = None;
            step.not_done = None;
            step.status = StepStatus::Background;
            step.started_at.get_or_insert_with(Utc::now);
            step.completed_at = None;
            if !step.background_jobs.iter().any(|job| job == job_id) {
                step.background_jobs.push(job_id.to_string());
            }
            changed = true;
        }
        if changed {
            // A step whose helper didn't start may have resolved the plan;
            // it is being worked on again.
            if !self.status.is_incomplete() {
                self.status = PlanStatus::InProgress;
            }
            self.updated_at = Utc::now();
        }
        changed
    }

    /// The turn handed its work to background jobs `job_ids` without naming
    /// the steps they do (`checklist_items`): every step still open (marked
    /// handed off with no job yet, or already waiting on an earlier job)
    /// waits on them, so the card reads "continuing in the background", never
    /// "stopped short", while they run. Their typed ends settle such a step
    /// ([`PlanStep::waits_on_jobs`]). Returns whether anything changed.
    pub fn wait_on_background_jobs(&mut self, job_ids: &[String]) -> bool {
        if job_ids.is_empty() {
            return false;
        }
        let mut changed = false;
        for step in &mut self.steps {
            // A step already waiting on an earlier job waits on this one too
            // (live 2026-11-13: two helpers started by one turn; the second
            // was never on the plan, so the first's end settled the steps
            // while the second still ran, and its own end found no plan).
            // A step the model marked handed off ("background") before any
            // launch took it waits on the launches too: the mark says it
            // delegated, the jobs are what it delegated to (live 2026-11-13:
            // two such steps were skipped here, no run was ever linked to
            // them, and the turn's end set them aside as "not done" while
            // both helpers went on to succeed).
            let open = matches!(step.status, StepStatus::Pending | StepStatus::InProgress)
                || (step.status == StepStatus::Deferred && step.not_done.is_none())
                || (step.status == StepStatus::Background
                    && (step.waits_on_jobs || step.background_jobs.is_empty()));
            if !open {
                continue;
            }
            step.status = StepStatus::Background;
            step.waits_on_jobs = true;
            step.completed_at = None;
            for job in job_ids {
                if !step.background_jobs.contains(job) {
                    step.background_jobs.push(job.clone());
                }
            }
            changed = true;
        }
        if changed {
            if self.is_finished() {
                self.status = self.resolved_status();
            }
            self.updated_at = Utc::now();
        }
        changed
    }

    /// Record the end of background job `job_id`, as its run record says.
    /// A step handed to it is completed once every job on it succeeded,
    /// failed only when one failed (a typed failure), and not done when one
    /// stopped before finishing. A step that only waited on the job settles
    /// from the typed ends of the jobs it waited on: done once every one
    /// succeeded, not done (with the plain reason) as soon as one failed or
    /// stopped. The plan resolves once nothing is left to run. Returns
    /// whether any step changed.
    pub fn settle_background_job(&mut self, job_id: &str, end: JobEnd) -> bool {
        let now = Utc::now();
        let mut changed = false;
        for step in &mut self.steps {
            if step.status != StepStatus::Background
                || !step.background_jobs.iter().any(|job| job == job_id)
            {
                continue;
            }
            changed = true;
            step.background_jobs.retain(|job| job != job_id);
            if step.waits_on_jobs {
                // Live 2026-11-13: a step that only waited on a background
                // job read "(not done)" after the job succeeded. Its runs'
                // typed ends are what settle it.
                let not_done = match end {
                    JobEnd::Succeeded if step.background_jobs.is_empty() => None,
                    JobEnd::Succeeded => continue,
                    JobEnd::Stopped => Some(NotDoneReason::HelperStopped),
                    JobEnd::Failed => Some(NotDoneReason::WaitedOnFailed),
                };
                step.waits_on_jobs = false;
                step.background_jobs.clear();
                step.completed_at = Some(now);
                match not_done {
                    None => {
                        step.status = StepStatus::Completed;
                        step.evidence = StepEvidence::Settled;
                    }
                    Some(reason) => {
                        step.status = StepStatus::Deferred;
                        step.not_done = Some(reason);
                    }
                }
                continue;
            }
            match end {
                JobEnd::Succeeded if step.background_jobs.is_empty() => {
                    // The helper's run record completed it.
                    step.status = StepStatus::Completed;
                    step.evidence = StepEvidence::Settled;
                    step.completed_at = Some(now);
                }
                JobEnd::Succeeded => {}
                JobEnd::Stopped => {
                    step.background_jobs.clear();
                    step.status = StepStatus::Deferred;
                    step.not_done = Some(NotDoneReason::HelperStopped);
                    step.completed_at = Some(now);
                }
                JobEnd::Failed => {
                    step.background_jobs.clear();
                    step.status = StepStatus::Failed;
                    step.completed_at = Some(now);
                }
            }
        }
        if changed {
            if self.is_finished() {
                self.status = self.resolved_status();
            } else {
                self.sync_active_step();
            }
            self.updated_at = now;
        }
        changed
    }

    /// Its turn is over and nothing it handed work to is still running: a
    /// step still open (pending, in progress, or waiting on a job no longer
    /// running) can no longer be done by anything, so it settles as not done
    /// ([`NotDoneReason::LeftOpen`]) and the plan resolves. A card never
    /// stays "working" on a step nothing is working on (live 2026-11-13: a
    /// "Report each helper's measured time" step held a card on "Working ·
    /// step 4 of 4" for 14 minutes after both helpers ended). Returns
    /// whether anything changed.
    pub fn settle_open_steps(&mut self) -> bool {
        let now = Utc::now();
        let mut changed = false;
        for step in &mut self.steps {
            if matches!(
                step.status,
                StepStatus::Pending | StepStatus::InProgress | StepStatus::Background
            ) {
                step.status = StepStatus::Deferred;
                step.waits_on_jobs = false;
                step.background_jobs.clear();
                step.not_done = Some(NotDoneReason::LeftOpen);
                step.completed_at = Some(now);
                changed = true;
            }
        }
        if changed {
            self.status = self.resolved_status();
            self.updated_at = now;
        }
        changed
    }

    /// The run's ledger says nothing ever took on the steps at `indices`
    /// ([`NotDoneReason::NotAttempted`]): each settles as not done with that
    /// reason, unless it is done or skipped. A step that only waited on the
    /// turn's jobs without being handed to one waits no longer: no job is
    /// doing it. Returns whether anything changed.
    pub fn mark_not_attempted(&mut self, indices: &std::collections::BTreeSet<usize>) -> bool {
        let now = Utc::now();
        let mut changed = false;
        for &index in indices {
            let Some(step) = self.steps.get_mut(index) else {
                continue;
            };
            if matches!(step.status, StepStatus::Completed | StepStatus::Skipped)
                || (step.status == StepStatus::Background && !step.waits_on_jobs)
            {
                continue;
            }
            step.status = StepStatus::Deferred;
            step.waits_on_jobs = false;
            step.background_jobs.clear();
            step.not_done = Some(NotDoneReason::NotAttempted);
            step.completed_at = Some(now);
            changed = true;
        }
        if changed {
            if self.is_finished() {
                self.status = self.resolved_status();
            } else {
                self.sync_active_step();
            }
            self.updated_at = now;
        }
        changed
    }

    /// Record that the steps not done in this plan were held back because
    /// the turn could not make changes: a failed or deferred step becomes a
    /// deferred step with [`NotDoneReason::ChangesOff`], which the card shows
    /// as correctly not done. Returns whether anything changed.
    pub fn hold_back_unfinished_steps(&mut self, reason: NotDoneReason) -> bool {
        let mut changed = false;
        for step in &mut self.steps {
            if matches!(step.status, StepStatus::Failed | StepStatus::Deferred)
                && step.not_done.is_none()
            {
                step.status = StepStatus::Deferred;
                step.not_done = Some(reason);
                step.completed_at.get_or_insert_with(Utc::now);
                changed = true;
            }
        }
        if changed {
            if self.is_finished() {
                self.status = self.resolved_status();
            }
            self.updated_at = Utc::now();
        }
        changed
    }

    /// The reason every step left undone shares, when all of them were held
    /// back for a known reason (and at least one was).
    fn shared_hold_reason(&self) -> Option<NotDoneReason> {
        let mut reason = None;
        for step in &self.steps {
            match (step.status, step.not_done) {
                (StepStatus::Completed | StepStatus::Skipped, _) => {}
                (StepStatus::Deferred, Some(found)) if reason.is_none_or(|r| r == found) => {
                    reason = Some(found);
                }
                _ => return None,
            }
        }
        reason
    }

    /// The plan as the card shows it: [`Self::render_compact_checklist`] and
    /// its step count, which decides whether the card is kept.
    pub fn card(&self) -> aidaemon_core::card::PlanCard {
        aidaemon_core::card::PlanCard {
            text: self.render_compact_checklist(),
            steps: self.steps.len(),
            done: self.completed_steps(),
            settled: !self
                .steps
                .iter()
                .any(|step| matches!(step.status, StepStatus::Pending | StepStatus::InProgress)),
        }
    }

    /// Compact, channel-friendly checklist for the live status card.
    ///
    /// One header line carries the state and the count; the steps follow, a
    /// blank line apart so wrapped descriptions stay readable on a phone. A
    /// one-step plan is just that step (a header and a "1 of 1" would only
    /// repeat it). A fully completed card starts with ✅
    /// ([`is_completed_checklist`]), so the card drops the turn's own step
    /// line. A step that turned out not to be
    /// needed, or that was held back for a known reason, is never shown as a
    /// failure.
    pub fn render_compact_checklist(&self) -> String {
        let total = self.steps.len();
        let unverified = self.unverified_steps();
        let in_background = self.has_background_steps();
        let finished = !matches!(
            self.status,
            PlanStatus::Planning | PlanStatus::InProgress | PlanStatus::Paused
        );
        // Every completed step counts as done; the action steps nothing
        // verified are said once, in plain words, on the final card only.
        let done = self.completed_steps();
        let said_note = if unverified > 0 && finished && !in_background {
            format!("; I couldn't verify {unverified}")
        } else {
            String::new()
        };
        let not_needed = self
            .steps
            .iter()
            .filter(|step| step.status == StepStatus::Skipped)
            .count();
        let held = self.shared_hold_reason();
        if total == 1 && (self.status != PlanStatus::PartiallyCompleted || held.is_some()) {
            let step = &self.steps[0];
            let mark = match (self.status, step.status) {
                (_, StepStatus::Completed) if step.unverified() => "☑️",
                (_, StepStatus::Completed) => "✅",
                (_, StepStatus::Skipped) => "✅",
                _ if held.is_some() => "⏭️",
                (PlanStatus::Failed, _) | (_, StepStatus::Failed) => "❌",
                (PlanStatus::Abandoned, _) => "⏹️",
                (PlanStatus::Paused, _) => "⏸️",
                _ => "⏳",
            };
            return match (step.status, held) {
                (StepStatus::Completed, _) if step.unverified() => {
                    format!("{mark} {} ({UNVERIFIED})", step.description)
                }
                (StepStatus::Failed, _) if step.helper_not_started() => {
                    format!("{mark} {} ({HELPER_NOT_STARTED})", step.description)
                }
                (StepStatus::Background, _) => {
                    format!("{mark} {} (running in the background)", step.description)
                }
                (StepStatus::Skipped, _) => format!("{mark} {} (not needed)", step.description),
                (_, Some(reason)) => {
                    format!("{mark} {} (not done: {})", step.description, reason.plain())
                }
                _ => format!("{mark} {}", step.description),
            };
        }
        let header = match self.status {
            // Work handed to a job that reports back is still being done:
            // never "stopped short" while it runs.
            _ if in_background => {
                format!("⏳ Continuing in the background · {done} of {total} done{said_note}")
            }
            // Everything left was held back for a known reason: the turn did
            // what it was allowed to, which is not a failure.
            _ if finished && held.is_some() => {
                let reason = held.map(NotDoneReason::plain).unwrap_or_default();
                format!("⏭️ Not done: {reason} · {done} of {total} done{said_note}")
            }
            PlanStatus::Planning => "🧭 Planning…".to_string(),
            PlanStatus::InProgress => {
                let position = self.current_step.saturating_add(1).min(total);
                format!("⚙️ Working · step {position} of {total}")
            }
            PlanStatus::Paused => format!("⏸️ Paused · {done} of {total} done{said_note}"),
            // Every step resolved, and an action step only on the model's
            // word: never a ✅ the evidence does not back.
            PlanStatus::Completed if unverified > 0 => {
                let needed_note = if not_needed > 0 {
                    format!(", {not_needed} not needed")
                } else {
                    String::new()
                };
                format!("☑️ Finished · {done} of {total} done{said_note}{needed_note}")
            }
            // The execution plan is resolved; the reply is still on its way,
            // which the typing indicator shows. Never claim the task is over.
            PlanStatus::Completed if not_needed > 0 => {
                format!("✅ Done · {done} done, {not_needed} not needed")
            }
            PlanStatus::Completed => format!("✅ Done · {done} of {total} steps"),
            PlanStatus::PartiallyCompleted => {
                format!("⚠️ Stopped short · {done} of {total} done{said_note}")
            }
            PlanStatus::Failed => format!("❌ Stopped · {done} of {total} done{said_note}"),
            PlanStatus::Abandoned => {
                format!("⏹️ Cancelled · {done} of {total} done{said_note}")
            }
        };
        let mut out = header;
        for step in &self.steps {
            let line = match (step.status, step.not_done) {
                (StepStatus::Completed, _) if step.unverified() => {
                    format!("☑️ {} ({UNVERIFIED})", step.description)
                }
                (StepStatus::Completed, _) => format!("✅ {}", step.description),
                (StepStatus::Failed, _) if step.helper_not_started() => {
                    format!("⚠️ {} ({HELPER_NOT_STARTED})", step.description)
                }
                (StepStatus::InProgress, _) => format!("▶️ {}", step.description),
                (StepStatus::Background, _) => {
                    format!("⏳ {} (running in the background)", step.description)
                }
                (StepStatus::Deferred, Some(reason)) => {
                    format!("⏭️ {} (not done: {})", step.description, reason.plain())
                }
                // `deferred` is the model's typed "set aside": the owner
                // reads what it means for them.
                (StepStatus::Deferred, None) => format!("⏭️ {} (not done)", step.description),
                (StepStatus::Failed, _) => format!("⚠️ {} (couldn't be done)", step.description),
                (StepStatus::Skipped, _) => format!("➖ {} (not needed)", step.description),
                (StepStatus::Pending, _) => format!("○ {}", step.description),
            };
            // A blank line between steps: descriptions wrap on a phone, and
            // without a gap the wrapped lines run into the next step.
            out.push_str("\n\n");
            out.push_str(&line);
        }
        if self.status == PlanStatus::PartiallyCompleted && !in_background && held.is_none() {
            out.push_str("\n\nThe reply below says what was left undone and why.");
        }
        // A finished card says plainly what never ran.
        let not_run: Vec<&str> = self
            .steps
            .iter()
            .filter(|step| step.status == StepStatus::Failed && step.helper_not_started())
            .map(|step| step.description.as_str())
            .collect();
        if finished && !in_background && !not_run.is_empty() {
            out.push_str("\n\nDidn't run: ");
            out.push_str(&not_run.join("; "));
        }
        out
    }

    /// End-of-turn recap of what got done vs what remains/deferred.
    pub fn render_recap(&self) -> String {
        let mut out = String::new();
        for step in &self.steps {
            let mark = match step.status {
                StepStatus::Completed => "✅",
                StepStatus::Deferred => "⏭️ (not done)",
                StepStatus::Failed => "⚠️ (failed)",
                StepStatus::Background => "⏳ (running in the background)",
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
                StepStatus::Background => "[bg]",
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

    /// Background jobs (their launch ids) still working on this step. Set at
    /// launch from the launch's typed `checklist_items`; each job's end
    /// settles the step ([`TaskPlan::settle_background_job`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub background_jobs: Vec<String>,

    /// The step was left open while the turn's background jobs ran, not
    /// handed to them by a launch's `checklist_items`: it reads as running
    /// while they do and settles from their typed ends: done once every one
    /// succeeded, not done (with the reason) once one failed or stopped.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub waits_on_jobs: bool,

    /// Why a step that was not done was left undone, when the daemon knows it
    /// from typed state (never from the step's text). Such a step was held
    /// back correctly, so the card does not show it as a failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_done: Option<NotDoneReason>,

    /// How its state is known: settled by typed evidence, completed on the
    /// executing model's word alone (its checklist update said done and no
    /// receipt or helper's result proved it; evidence clears it), or handed
    /// to a helper that never started (a later launch takes it again).
    #[serde(default, skip_serializing_if = "StepEvidence::is_settled")]
    pub evidence: StepEvidence,

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
    /// Completed on the model's word alone, and a step that declared a
    /// side effect: the one kind of step the card flags. A step that
    /// changes nothing (a check, a read) needs no receipt: the reply says
    /// what it found.
    pub fn unverified(&self) -> bool {
        self.status == StepStatus::Completed
            && self.evidence == StepEvidence::ModelSaid
            && !self.required_mutation_effects.is_empty()
    }

    /// It was handed to a helper that never started.
    pub fn helper_not_started(&self) -> bool {
        self.evidence == StepEvidence::HelperNotStarted
    }

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

    #[allow(clippy::should_implement_trait)]
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
    /// Handed to a background job that reports back: still being worked on,
    /// outside this run. Only the job's end moves it to completed or failed.
    Background,
}

impl StepStatus {
    pub fn as_str(&self) -> &'static str {
        (*self).into()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

/// How the card marks an action step nothing verified.
const UNVERIFIED: &str = "I couldn't verify this one";
/// How the card marks a step whose helper never started.
const HELPER_NOT_STARTED: &str = "didn't run: its helper didn't start";

/// Why a step was correctly left undone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotDoneReason {
    /// The turn could not change anything (read-only access or a request
    /// that forbids changes), and this step needed a change.
    ChangesOff,
    /// The background helper it was handed ended before finishing it (its
    /// run record says stopped: not a failure).
    HelperStopped,
    /// Its turn ended and nothing was left working on it.
    LeftOpen,
    /// A background job it waited on failed (its run record says so).
    WaitedOnFailed,
    /// Nothing ever took it on: no call did it, no job was launched for it,
    /// and the owner was not asked (the run's ledger says so).
    NotAttempted,
}

impl NotDoneReason {
    fn plain(self) -> &'static str {
        match self {
            Self::ChangesOff => "read-only is on",
            Self::HelperStopped => "its helper stopped before it finished",
            Self::LeftOpen => "nothing was left working on it",
            Self::WaitedOnFailed => "the work it waited on failed",
            Self::NotAttempted => "not attempted",
        }
    }
}

/// How a background job ended, as its run record says: what its steps
/// become. Only `Failed` marks a step as couldn't be done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobEnd {
    Succeeded,
    Stopped,
    Failed,
}

/// Whether a card from [`TaskPlan::render_compact_checklist`] shows every
/// step finished (the channel's live status reads cards the same way).
pub use aidaemon_channels::live_status::is_completed_checklist;

#[cfg(test)]
mod tests {
    use super::*;

    /// A step owns its runs in order and its outcome is its latest run's:
    /// a step an earlier helper failed or stopped short on is taken over by
    /// a later one for the same step, and that run's success completes it.
    #[test]
    fn a_retried_step_settles_from_its_latest_run() {
        for first_end in [JobEnd::Stopped, JobEnd::Failed] {
            let mut plan = TaskPlan::new(
                "session_123",
                "Install two sites",
                "Two sites",
                vec![
                    "Install WordPress".to_string(),
                    "Install Drupal".to_string(),
                ],
                "low",
            );
            assert!(plan.link_background_job(&[1], "job-1"));
            assert!(plan.settle_background_job("job-1", first_end));
            assert_ne!(plan.steps[1].status, StepStatus::Completed);
            assert!(plan.link_background_job(&[1], "job-2"), "{first_end:?}");
            assert_eq!(plan.steps[1].status, StepStatus::Background);
            assert_eq!(plan.steps[1].not_done, None);
            assert!(plan.settle_background_job("job-2", JobEnd::Succeeded));
            assert_eq!(plan.steps[1].status, StepStatus::Completed);
        }
    }

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

    /// A second job a turn starts joins the steps already waiting on the
    /// first; once both ended, a step left open settles as not done and the
    /// plan resolves: nothing reads "working" with nothing working.
    #[test]
    fn open_steps_settle_once_nothing_runs() {
        let mut plan = TaskPlan::new(
            "session_123",
            "test",
            "Two installs",
            vec!["Install one".into(), "Install two".into()],
            "test",
        );
        for step in &mut plan.steps {
            step.status = StepStatus::Pending;
        }
        assert!(plan.wait_on_background_jobs(&["job-1".to_string()]));
        assert!(plan.wait_on_background_jobs(&["job-2".to_string()]));
        assert_eq!(plan.steps[0].background_jobs, vec!["job-1", "job-2"]);
        plan.settle_background_job("job-1", JobEnd::Succeeded);
        assert_eq!(plan.steps[0].status, StepStatus::Background);
        plan.steps.push(PlanStep {
            status: StepStatus::InProgress,
            description: "Report each time".into(),
            background_jobs: Vec::new(),
            waits_on_jobs: false,
            ..plan.steps[0].clone()
        });
        plan.settle_background_job("job-2", JobEnd::Succeeded);
        assert!(!plan.is_finished());
        assert!(plan.settle_open_steps());
        assert!(plan.is_finished());
        assert_eq!(plan.steps[2].not_done, Some(NotDoneReason::LeftOpen));
        assert!(plan
            .render_compact_checklist()
            .contains("nothing was left working on it"));
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
        assert_eq!(
            checklist,
            "⚙️ Working · step 2 of 2\n\n✅ create script\n\n▶️ send the file"
        );
        assert!(!is_completed_checklist(&checklist));
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
        // One step: the step itself, no header or "1 of 1".
        assert_eq!(checklist, "✅ observe result");
        assert!(is_completed_checklist(&checklist));
        assert!(!checklist.contains("Task complete"));

        let mut plan = TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            vec!["first".into(), "second".into()],
            "track_requirements",
        );
        for step in &mut plan.steps {
            step.status = StepStatus::Completed;
        }
        plan.status = PlanStatus::Completed;
        let checklist = plan.render_compact_checklist();
        assert_eq!(checklist, "✅ Done · 2 of 2 steps\n\n✅ first\n\n✅ second");
        assert!(is_completed_checklist(&checklist));
        assert!(!checklist.contains("Finalizing"));
    }

    /// Steps handed to background jobs read as still being done: the card
    /// never says "stopped short" or "not done" while they run, whatever
    /// verdict the run itself recorded.
    #[test]
    fn handed_off_steps_render_as_running_in_the_background() {
        let mut plan = TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            vec![
                "inspect".into(),
                "first setup".into(),
                "second setup".into(),
                "report both".into(),
            ],
            "track_requirements",
        );
        plan.steps[0].status = StepStatus::Completed;
        assert!(plan.link_background_job(&[1, 3], "job-a"));
        assert!(plan.link_background_job(&[2, 3], "job-b"));
        // The run itself ends partial: work continues elsewhere.
        plan.status = PlanStatus::PartiallyCompleted;

        let card = plan.render_compact_checklist();
        assert_eq!(
            card,
            "⏳ Continuing in the background · 1 of 4 done\n\n✅ inspect\n\n\
             ⏳ first setup (running in the background)\n\n\
             ⏳ second setup (running in the background)\n\n\
             ⏳ report both (running in the background)"
        );
        assert!(!is_completed_checklist(&card));

        // One job done: its own step is done; the shared one waits for both.
        assert!(plan.settle_background_job("job-a", JobEnd::Succeeded));
        assert_eq!(plan.steps[1].status, StepStatus::Completed);
        assert_eq!(plan.steps[3].status, StepStatus::Background);
        assert!(plan
            .render_compact_checklist()
            .starts_with("⏳ Continuing in the background · 2 of 4 done"));

        // The other fails: its step and the shared one failed, never done.
        assert!(plan.settle_background_job("job-b", JobEnd::Failed));
        assert_eq!(plan.steps[2].status, StepStatus::Failed);
        assert_eq!(plan.steps[3].status, StepStatus::Failed);
        assert_eq!(plan.status, PlanStatus::Failed);
        assert!(plan
            .render_compact_checklist()
            .starts_with("❌ Stopped · 2 of 4 done"));
        // A job nothing waits on changes nothing.
        assert!(!plan.settle_background_job("job-c", JobEnd::Succeeded));
    }

    /// A turn that hands work to background jobs without linking its open
    /// steps never reads "stopped short" or "not done" while they run, and
    /// once every job succeeded the steps waiting on them are done.
    #[test]
    fn open_steps_wait_on_the_turns_background_jobs_and_settle_from_their_ends() {
        let mut plan = TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            vec![
                "check tools".into(),
                "install site a".into(),
                "install site b".into(),
            ],
            "track_requirements",
        );
        plan.steps[0].status = StepStatus::Completed;
        plan.steps[1].status = StepStatus::Deferred;
        plan.steps[2].status = StepStatus::Deferred;
        plan.status = PlanStatus::PartiallyCompleted;
        assert!(plan
            .render_compact_checklist()
            .starts_with("⚠️ Stopped short"));

        let jobs = vec!["job-a".to_string(), "job-b".to_string()];
        assert!(plan.wait_on_background_jobs(&jobs));
        let card = plan.render_compact_checklist();
        assert!(
            card.starts_with("⏳ Continuing in the background · 1 of 3 done"),
            "{card}"
        );
        assert!(
            !card.contains("Stopped short") && !card.contains("not done"),
            "{card}"
        );

        assert!(plan.settle_background_job("job-a", JobEnd::Succeeded));
        assert!(plan
            .render_compact_checklist()
            .starts_with("⏳ Continuing in the background"));
        assert!(plan.settle_background_job("job-b", JobEnd::Succeeded));
        assert_eq!(plan.steps[1].status, StepStatus::Completed);
        assert_eq!(plan.steps[2].status, StepStatus::Completed);
        assert_eq!(plan.status, PlanStatus::Completed);
        let card = plan.render_compact_checklist();
        assert!(!card.contains("not done"), "{card}");
        assert!(is_completed_checklist(&card), "{card}");
    }

    /// A step waiting on runs is not done, with the plain reason, as soon as
    /// one of them failed; a later success of another run does not undo it.
    #[test]
    fn a_step_waiting_on_a_failed_run_is_not_done_with_the_reason() {
        let mut plan = TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            vec!["wait for the build".into()],
            "track_requirements",
        );
        let jobs = vec!["job-a".to_string(), "job-b".to_string()];
        assert!(plan.wait_on_background_jobs(&jobs));
        assert!(plan.settle_background_job("job-a", JobEnd::Failed));
        assert_eq!(plan.steps[0].status, StepStatus::Deferred);
        assert_eq!(plan.steps[0].not_done, Some(NotDoneReason::WaitedOnFailed));
        assert!(!plan.settle_background_job("job-b", JobEnd::Succeeded));
        assert!(plan.is_finished());
        let card = plan.render_compact_checklist();
        assert!(
            card.contains("wait for the build (not done: the work it waited on failed)"),
            "{card}"
        );
    }

    /// Live 2026-11-13: one helper launched for the Drupal step; the
    /// WordPress step was never taken on. The card says so with its typed
    /// reason, and the running helper still settles its own step.
    #[test]
    fn a_step_nothing_took_on_reads_not_attempted() {
        let mut plan = TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            vec!["Install Drupal".into(), "Install WordPress".into()],
            "track_requirements",
        );
        plan.link_background_job(&[0], "job-a");
        assert!(plan.mark_not_attempted(&std::collections::BTreeSet::from([1])));
        assert_eq!(plan.steps[1].not_done, Some(NotDoneReason::NotAttempted));
        assert_eq!(plan.steps[0].status, StepStatus::Background);
        let card = plan.render_compact_checklist();
        assert!(
            card.contains("Install WordPress (not done: not attempted)"),
            "{card}"
        );
        assert!(plan.settle_background_job("job-a", JobEnd::Succeeded));
        assert_eq!(plan.steps[0].status, StepStatus::Completed);
        assert_eq!(plan.steps[1].not_done, Some(NotDoneReason::NotAttempted));
    }

    #[test]
    fn background_steps_that_all_finish_complete_the_plan() {
        let mut plan = TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            vec!["first".into(), "second".into()],
            "track_requirements",
        );
        plan.link_background_job(&[0, 1], "job-a");
        plan.status = PlanStatus::PartiallyCompleted;
        assert!(plan.settle_background_job("job-a", JobEnd::Succeeded));
        assert_eq!(plan.status, PlanStatus::Completed);
        assert!(is_completed_checklist(&plan.render_compact_checklist()));
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
        assert!(checklist.starts_with("⚠️ Stopped short · 0 of 2 done"));
        assert!(checklist.contains("⏭️ first action (not done)"));
        assert!(
            !checklist.contains("deferred"),
            "no internal status word: {checklist}"
        );
        assert!(!is_completed_checklist(&checklist));
        assert!(checklist.contains("what was left undone and why"));
        // The footer speaks to the user; it is not an instruction to the model.
        assert!(!checklist.contains("must"));
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
