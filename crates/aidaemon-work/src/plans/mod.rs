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

pub use store::{PlanStore, migrate_task_plans};

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
                waited_jobs: Vec::new(),
                made_folders: Vec::new(),
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
            step.waited_jobs.clear();
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
            // Its own run now: the jobs it waited on no longer settle it.
            step.waited_jobs.clear();
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

    /// The steps a new helper would do that a helper still running already
    /// does, each with the running jobs on it. `indices` are the steps the
    /// new launch names (`checklist_items`). A launch that names none is
    /// judged against every open step: it is covered only when each of them
    /// is already being done (live 2026-11-14: the turn launched a WordPress
    /// and a Drupal helper, then a third helper for the WordPress install
    /// while the first still ran). A step whose helpers all ended is not
    /// covered, whatever its status still says, so a step an earlier helper
    /// ended without finishing can be launched again. `running` says whether
    /// a job is still running (the run records know that, not the plan).
    pub fn steps_covered_by_running_jobs(
        &self,
        indices: &[usize],
        running: impl Fn(&str) -> bool,
    ) -> Vec<CoveredStep> {
        let covered = |step: &PlanStep| -> Option<CoveredStep> {
            if step.status != StepStatus::Background {
                return None;
            }
            let jobs: Vec<String> = step
                .background_jobs
                .iter()
                .filter(|job| running(job))
                .cloned()
                .collect();
            (!jobs.is_empty()).then(|| CoveredStep {
                index: step.index,
                description: step.description.clone(),
                jobs,
            })
        };
        if !indices.is_empty() {
            return indices
                .iter()
                .filter_map(|index| self.steps.get(*index))
                .filter_map(covered)
                .collect();
        }
        let open: Vec<&PlanStep> = self
            .steps
            .iter()
            .filter(|step| {
                !matches!(
                    step.status,
                    StepStatus::Completed
                        | StepStatus::Skipped
                        | StepStatus::Failed
                        | StepStatus::Deferred
                )
            })
            .collect();
        let all: Vec<CoveredStep> = open.iter().filter_map(|step| covered(step)).collect();
        if open.is_empty() || all.len() < open.len() {
            return Vec::new();
        }
        all
    }

    /// The turn handed its work to background jobs `job_ids` without naming
    /// the steps they do (`checklist_items`): every step still open (marked
    /// handed off with no job yet, or already waiting on an earlier job)
    /// waits on them, so the card reads "continuing in the background", never
    /// "stopped short", while they run. Their typed ends settle such a step
    /// ([`PlanStep::waited_jobs`]). Returns whether anything changed.
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
                || (step.status == StepStatus::Background && step.background_jobs.is_empty());
            if !open {
                continue;
            }
            step.status = StepStatus::Background;
            step.completed_at = None;
            for job in job_ids {
                if !step.waited_jobs.contains(job) {
                    step.waited_jobs.push(job.clone());
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

    /// The folders background job `job_id` made (its claim evidence), on
    /// the steps it took on, before its end settles them. Returns whether
    /// any step changed.
    pub fn note_job_folders(&mut self, job_id: &str, folders: &[String]) -> bool {
        let mut changed = false;
        for step in &mut self.steps {
            if !step.background_jobs.iter().any(|job| job == job_id) {
                continue;
            }
            for folder in folders {
                if !step.made_folders.contains(folder) {
                    step.made_folders.push(folder.clone());
                    changed = true;
                }
            }
        }
        if changed {
            self.updated_at = Utc::now();
        }
        changed
    }

    /// Record the end of background job `job_id`, as its run record says.
    /// A step handed to several jobs (or waiting on them) settles once the
    /// last of them ended, from all their typed ends: completed when every
    /// one succeeded, failed when one failed (a step handed to it; a step
    /// that only waited on it is not done, with that reason), and not done
    /// when one stopped before finishing or ended short because the owner
    /// declined one of its steps. While others still run it stays in the
    /// background, so a later run's success is never lost (live 2026-11-13:
    /// the WordPress helper's end marked steps both helpers shared as not
    /// done, the Drupal helper then finished them, and the card read "1 of
    /// 5 done" above its helpers' finished lines). The plan resolves once
    /// nothing is left to run. Returns whether any step changed.
    pub fn settle_background_job(&mut self, job_id: &str, end: JobEnd) -> bool {
        let now = Utc::now();
        let mut changed = false;
        let short = match end {
            JobEnd::Succeeded => None,
            JobEnd::Declined => Some(NotDoneReason::OwnerDeclined),
            JobEnd::StoppedByOwner => Some(NotDoneReason::OwnerStopped),
            JobEnd::Stopped => Some(NotDoneReason::HelperStopped),
            JobEnd::Failed => Some(NotDoneReason::WaitedOnFailed),
        };
        // The worse of two ends: a failure, then a stop, then a decline.
        let rank = |reason: NotDoneReason| match reason {
            NotDoneReason::WaitedOnFailed => 3,
            NotDoneReason::HelperStopped => 2,
            _ => 1,
        };
        for step in &mut self.steps {
            let took_on = step.background_jobs.iter().any(|job| job == job_id);
            let waited = step.waited_jobs.iter().any(|job| job == job_id);
            if step.status != StepStatus::Background || !(took_on || waited) {
                continue;
            }
            changed = true;
            step.background_jobs.retain(|job| job != job_id);
            step.waited_jobs.retain(|job| job != job_id);
            // A step a job of its own took on settles from its own runs
            // only: an end it merely waited on is not its outcome (live
            // 2026-11-14: Drupal's success marked the WordPress step done).
            if !took_on && !step.background_jobs.is_empty() {
                continue;
            }
            if let Some(short) = short {
                step.not_done = Some(match step.not_done {
                    Some(earlier) if rank(earlier) >= rank(short) => earlier,
                    _ => short,
                });
            }
            if !step.background_jobs.is_empty() || (!took_on && !step.waited_jobs.is_empty()) {
                continue;
            }
            // Live 2026-11-13: a step that only waited on a background job
            // read "(not done)" after the job succeeded. Its runs' typed
            // ends are what settle it.
            step.waited_jobs.clear();
            step.completed_at = Some(now);
            match step.not_done {
                None => {
                    step.status = StepStatus::Completed;
                    step.evidence = StepEvidence::Settled;
                }
                Some(NotDoneReason::WaitedOnFailed) if took_on => {
                    step.status = StepStatus::Failed;
                    step.not_done = None;
                }
                Some(_) => step.status = StepStatus::Deferred,
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
    ///
    /// While a job a launch handed a step to has not ended (it is still
    /// listed on the step), the plan's work is still going and nothing
    /// settles: that job's own end settles its step
    /// ([`Self::settle_background_job`]) and then the rest, however the
    /// card's runs were counted when this settle ran. Live 2026-11-13: the
    /// first helper's end settled the card while the second, launched for
    /// the WordPress step, was not yet counted on it; the card read
    /// "Stopped short · 1 of 3 (not done: nothing was left working on it)",
    /// then "Continuing", then "Stopped".
    pub fn settle_open_steps(&mut self) -> bool {
        if self
            .steps
            .iter()
            .any(|step| step.status == StepStatus::Background && step.has_background_jobs())
        {
            return false;
        }
        let now = Utc::now();
        let mut changed = false;
        for step in &mut self.steps {
            if matches!(
                step.status,
                StepStatus::Pending | StepStatus::InProgress | StepStatus::Background
            ) {
                step.status = StepStatus::Deferred;
                step.waited_jobs.clear();
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
                || (step.status == StepStatus::Background && !step.background_jobs.is_empty())
            {
                continue;
            }
            step.status = StepStatus::Deferred;
            step.waited_jobs.clear();
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
            finished_steps: self
                .steps
                .iter()
                .filter(|step| matches!(step.status, StepStatus::Completed | StepStatus::Skipped))
                .map(|step| step.index)
                .collect(),
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
            let described = step.described();
            return match (step.status, held) {
                (StepStatus::Completed, _) if step.unverified() => {
                    format!("{mark} {described} ({UNVERIFIED})")
                }
                (StepStatus::Failed, _) if step.helper_not_started() => {
                    format!("{mark} {described} ({HELPER_NOT_STARTED})")
                }
                (StepStatus::Background, _) => {
                    format!("{mark} {described} (running in the background)")
                }
                (StepStatus::Skipped, _) => format!("{mark} {described} (not needed)"),
                (_, Some(reason)) => {
                    format!("{mark} {described} (not done: {})", reason.plain())
                }
                _ => format!("{mark} {described}"),
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
                    format!("☑️ {} ({UNVERIFIED})", step.described())
                }
                (StepStatus::Completed, _) => format!("✅ {}", step.described()),
                // Its row says only that it didn't run; why is said once,
                // under the rows, however many steps it applies to.
                (StepStatus::Failed, _) if step.helper_not_started() => {
                    format!("⚠️ {} (didn't run)", step.described())
                }
                (StepStatus::InProgress, _) => format!("▶️ {}", step.described()),
                (StepStatus::Background, _) => {
                    format!("⏳ {} (running in the background)", step.described())
                }
                (StepStatus::Deferred, Some(reason)) => {
                    format!("⏭️ {} (not done: {})", step.described(), reason.plain())
                }
                // `deferred` is the model's typed "set aside": the owner
                // reads what it means for them.
                (StepStatus::Deferred, None) => format!("⏭️ {} (not done)", step.described()),
                (StepStatus::Failed, _) => format!("⚠️ {} (couldn't be done)", step.described()),
                (StepStatus::Skipped, _) => format!("➖ {} (not needed)", step.described()),
                (StepStatus::Pending, _) => format!("○ {}", step.described()),
            };
            // A blank line between steps: descriptions wrap on a phone, and
            // without a gap the wrapped lines run into the next step.
            out.push_str("\n\n");
            out.push_str(&line);
        }
        if self.status == PlanStatus::PartiallyCompleted && !in_background && held.is_none() {
            out.push_str("\n\nThe reply below says what was left undone and why.");
        }
        // Why the steps marked "didn't run" never ran: said once, not on
        // each row (live 2026-11-14: "(didn't run: its helper didn't
        // start)" twice on one card, then the same steps again below).
        if self
            .steps
            .iter()
            .any(|step| step.status == StepStatus::Failed && step.helper_not_started())
        {
            out.push_str("\n\nDidn't run: its helper didn't start.");
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

/// A checklist step a running helper already does
/// ([`TaskPlan::steps_covered_by_running_jobs`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoveredStep {
    pub index: usize,
    pub description: String,
    /// The running jobs (launch ids) on it.
    pub jobs: Vec<String>,
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

    /// Background jobs (their launch ids) that took this step on and still
    /// work on it: a launch named it in its typed `checklist_items`. Each
    /// job's end settles the step ([`TaskPlan::settle_background_job`]), and
    /// only these make a new helper for the step a duplicate.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub background_jobs: Vec<String>,

    /// Background jobs this step only waits on: it was left open while the
    /// turn's jobs ran, and no launch named it. It reads as running while
    /// they do and, while no job of its own took it on, settles from their
    /// typed ends: done once every one succeeded, not done (with the reason)
    /// once one failed or stopped. Waiting on a job is not being done by it:
    /// it never makes a launch for the step a duplicate.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub waited_jobs: Vec<String>,

    /// The folders the runs that took this step on made for it, from their
    /// claim evidence (what each found absent and then created), recorded
    /// as each run ended. The card names them, so a run that made another
    /// folder than the step's words named shows the real one (live
    /// 2026-11-14: the card read claude-drupal-12, the site was made in
    /// claude-company-drupal-5879).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub made_folders: Vec<String>,

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
    /// Its words on the card, with the folders its runs made that its words
    /// do not already name: the real folder, from the claim evidence.
    pub fn described(&self) -> String {
        let unnamed: Vec<&str> = self
            .made_folders
            .iter()
            .map(|folder| {
                std::path::Path::new(folder)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(folder)
            })
            .filter(|name| !self.description.contains(name))
            .collect();
        match unnamed.as_slice() {
            [] => self.description.clone(),
            [one] => format!("{} (made {one})", self.description),
            many => format!("{} (made {})", self.description, many.join(", ")),
        }
    }

    /// Completed on the model's word alone, and a step that declared a
    /// side effect: the one kind of step the card flags. A step that
    /// changes nothing (a check, a read) needs no receipt: the reply says
    /// what it found.
    pub fn unverified(&self) -> bool {
        self.status == StepStatus::Completed
            && self.evidence == StepEvidence::ModelSaid
            && !self.required_mutation_effects.is_empty()
    }

    /// Any background job is still listed on it: one that took it on, or
    /// one it waits on.
    pub fn has_background_jobs(&self) -> bool {
        !self.background_jobs.is_empty() || !self.waited_jobs.is_empty()
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
    /// The owner declined a step the helper it was handed asked to take,
    /// and it ended without finishing (its run record says so): the
    /// owner's answer, never a failure.
    OwnerDeclined,
    /// The owner stopped the helper it was handed before it finished: a
    /// stop, never a failure.
    OwnerStopped,
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
            Self::OwnerDeclined => "you declined a step its helper asked to take",
            Self::OwnerStopped => "you stopped its helper",
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
    /// It stopped short because the owner declined one of its steps.
    Declined,
    /// The owner stopped it before it finished.
    StoppedByOwner,
    Failed,
}

/// Whether a card from [`TaskPlan::render_compact_checklist`] shows every
/// step finished (the channel's live status reads cards the same way).
pub use aidaemon_channels::live_status::is_completed_checklist;

#[cfg(test)]
mod tests {
    use super::*;

    fn two_sites() -> TaskPlan {
        TaskPlan::new(
            "session_123",
            "Install two sites",
            "Two sites",
            vec![
                "Install WordPress".to_string(),
                "Install Drupal".to_string(),
            ],
            "low",
        )
    }

    /// A step a running helper does is covered: a second launch for it is
    /// a duplicate. Once that helper ended without finishing it, it is open
    /// to a retry again.
    #[test]
    fn a_step_a_running_helper_does_is_covered_until_it_ends() {
        let mut plan = two_sites();
        plan.link_background_job(&[0], "job-wp");
        plan.link_background_job(&[1], "job-drupal");
        let running = |job: &str| job == "job-wp" || job == "job-drupal";

        let covered = plan.steps_covered_by_running_jobs(&[0], running);
        assert_eq!(
            covered,
            vec![CoveredStep {
                index: 0,
                description: "Install WordPress".to_string(),
                jobs: vec!["job-wp".to_string()],
            }]
        );
        // Every open step is being done: a launch naming none covers nothing new.
        assert_eq!(plan.steps_covered_by_running_jobs(&[], running).len(), 2);

        // The WordPress helper stopped short: its step may be launched again.
        plan.settle_background_job("job-wp", JobEnd::Stopped);
        let only_drupal = |job: &str| job == "job-drupal";
        assert!(
            plan.steps_covered_by_running_jobs(&[0], only_drupal)
                .is_empty()
        );
        // A retry names its step: unnamed, it reads as more of what still runs.
        let unnamed = plan.steps_covered_by_running_jobs(&[], only_drupal);
        assert_eq!(
            unnamed.iter().map(|step| step.index).collect::<Vec<_>>(),
            [1]
        );
    }

    /// A step whose helper is gone (a restart lost its run record) is not
    /// covered, even while its status still reads as in the background.
    #[test]
    fn a_step_whose_helper_is_gone_is_not_covered() {
        let mut plan = two_sites();
        plan.link_background_job(&[0, 1], "job-lost");
        assert!(
            plan.steps_covered_by_running_jobs(&[0], |_| false)
                .is_empty()
        );
        assert!(
            plan.steps_covered_by_running_jobs(&[], |_| false)
                .is_empty()
        );
    }

    /// A launch that names no steps is covered only when every open step
    /// already has a running helper: with one step free, it may be that one.
    #[test]
    fn an_unnamed_launch_with_a_free_step_is_not_covered() {
        let mut plan = two_sites();
        plan.link_background_job(&[0], "job-wp");
        assert!(
            plan.steps_covered_by_running_jobs(&[], |job| job == "job-wp")
                .is_empty()
        );
        // Naming the covered step is still a duplicate.
        assert_eq!(
            plan.steps_covered_by_running_jobs(&[0, 1], |job| job == "job-wp")
                .len(),
            1
        );
    }

    /// A step a launch left waiting on the turn's jobs (no `checklist_items`)
    /// is not being done by them: it waits, and a launch for it is not a
    /// duplicate (live 2026-11-14: the WordPress helper was refused because
    /// its step waited on the Drupal job).
    #[test]
    fn steps_waiting_on_unnamed_launches_are_not_covered() {
        let mut plan = two_sites();
        plan.wait_on_background_jobs(&["job-a".to_string(), "job-b".to_string()]);
        assert!(plan.steps_covered_by_running_jobs(&[], |_| true).is_empty());
        assert!(
            plan.steps_covered_by_running_jobs(&[0], |_| true)
                .is_empty()
        );
    }

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
        assert_eq!(plan.steps[0].waited_jobs, vec!["job-1", "job-2"]);
        plan.settle_background_job("job-1", JobEnd::Succeeded);
        assert_eq!(plan.steps[0].status, StepStatus::Background);
        plan.steps.push(PlanStep {
            status: StepStatus::InProgress,
            description: "Report each time".into(),
            background_jobs: Vec::new(),
            waited_jobs: Vec::new(),
            ..plan.steps[0].clone()
        });
        plan.settle_background_job("job-2", JobEnd::Succeeded);
        assert!(!plan.is_finished());
        assert!(plan.settle_open_steps());
        assert!(plan.is_finished());
        assert_eq!(plan.steps[2].not_done, Some(NotDoneReason::LeftOpen));
        assert!(
            plan.render_compact_checklist()
                .contains("nothing was left working on it")
        );
    }

    /// Live 2026-11-13: two helpers launched, one per site. The first one's
    /// end settled the card before the second was counted on it, and its
    /// step read "nothing was left working on it" while it ran. A step a
    /// launch took on waits for that launch's own end.
    #[test]
    fn a_step_a_launch_took_on_is_never_left_open() {
        let mut plan = TaskPlan::new(
            "session_123",
            "test",
            "Two screenshots",
            vec![
                "Drupal homepage screenshot".into(),
                "WordPress homepage screenshot".into(),
                "Report each time".into(),
            ],
            "test",
        );
        assert!(plan.link_background_job(&[0], "job-drupal"));
        assert!(plan.link_background_job(&[1], "job-wordpress"));
        assert!(plan.settle_background_job("job-drupal", JobEnd::Succeeded));
        assert!(!plan.settle_open_steps());
        assert_eq!(plan.steps[1].status, StepStatus::Background);
        assert_eq!(plan.steps[2].not_done, None);
        assert_eq!(plan.steps[1].not_done, None);
        assert!(!plan.is_finished());
        let card = plan.render_compact_checklist();
        assert!(!card.contains("Stopped short"), "{card}");
        assert!(!card.contains("nothing was left working on it"), "{card}");
        assert!(plan.settle_background_job("job-wordpress", JobEnd::Succeeded));
        assert_eq!(plan.steps[1].status, StepStatus::Completed);
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
        assert!(
            plan.render_compact_checklist()
                .starts_with("⏳ Continuing in the background · 2 of 4 done")
        );

        // The other fails: its step and the shared one failed, never done.
        assert!(plan.settle_background_job("job-b", JobEnd::Failed));
        assert_eq!(plan.steps[2].status, StepStatus::Failed);
        assert_eq!(plan.steps[3].status, StepStatus::Failed);
        assert_eq!(plan.status, PlanStatus::Failed);
        assert!(
            plan.render_compact_checklist()
                .starts_with("❌ Stopped · 2 of 4 done")
        );
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
        assert!(
            plan.render_compact_checklist()
                .starts_with("⚠️ Stopped short")
        );

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
        assert!(
            plan.render_compact_checklist()
                .starts_with("⏳ Continuing in the background")
        );
        assert!(plan.settle_background_job("job-b", JobEnd::Succeeded));
        assert_eq!(plan.steps[1].status, StepStatus::Completed);
        assert_eq!(plan.steps[2].status, StepStatus::Completed);
        assert_eq!(plan.status, PlanStatus::Completed);
        let card = plan.render_compact_checklist();
        assert!(!card.contains("not done"), "{card}");
        assert!(is_completed_checklist(&card), "{card}");
    }

    /// A step waiting on runs settles once the last of them ended: not
    /// done, with the plain reason, when one of them failed, even though
    /// another succeeded.
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
        assert_eq!(plan.steps[0].status, StepStatus::Background);
        assert!(!plan.is_finished());
        assert!(plan.settle_background_job("job-b", JobEnd::Succeeded));
        assert_eq!(plan.steps[0].status, StepStatus::Deferred);
        assert_eq!(plan.steps[0].not_done, Some(NotDoneReason::WaitedOnFailed));
        assert!(plan.is_finished());
        let card = plan.render_compact_checklist();
        assert!(
            card.contains("wait for the build (not done: the work it waited on failed)"),
            "{card}"
        );
    }

    /// Live 2026-11-13: the owner declined the WordPress helper's password
    /// rotation; it ended short while the Drupal helper still ran, and the
    /// card read "Not done: its helper stopped before it finished · 1 of 5
    /// done" above its helpers' finished lines. A step both helpers shared
    /// waits for both; the WordPress step reads why it was not done; the
    /// header's count is the rows' count.
    #[test]
    fn a_declined_helper_reads_as_declined_and_shared_steps_wait_for_every_run() {
        let mut plan = TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            vec![
                "Install Drupal".into(),
                "Install WordPress".into(),
                "Screenshot both homepages".into(),
            ],
            "track_requirements",
        );
        assert!(plan.link_background_job(&[0, 2], "job-drupal"));
        assert!(plan.link_background_job(&[1, 2], "job-wordpress"));
        assert!(plan.settle_background_job("job-wordpress", JobEnd::Declined));
        assert_eq!(plan.steps[1].not_done, Some(NotDoneReason::OwnerDeclined));
        assert_eq!(plan.steps[2].status, StepStatus::Background);
        assert!(plan.settle_background_job("job-drupal", JobEnd::Succeeded));
        assert_eq!(plan.steps[0].status, StepStatus::Completed);
        assert_eq!(plan.steps[2].status, StepStatus::Deferred);
        assert_eq!(plan.steps[2].not_done, Some(NotDoneReason::OwnerDeclined));
        let card = plan.render_compact_checklist();
        assert!(
            card.starts_with(
                "⏭️ Not done: you declined a step its helper asked to take · 1 of 3 done"
            ),
            "{card}"
        );
        let done_rows = card.lines().filter(|line| line.starts_with('✅')).count();
        assert_eq!(done_rows, plan.card().done, "{card}");
    }

    /// Live 2026-11-14: the Drupal step read claude-drupal-12, but its
    /// helper made claude-company-drupal-5879. The folder a job's claim
    /// evidence recorded goes on the steps it took on, and the card names
    /// it when the step's words don't; a step that already names it reads
    /// as before, and a step the job never took on gets nothing.
    #[test]
    fn a_step_names_the_folder_its_run_really_made() {
        let mut plan = TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            vec![
                "Install Drupal in claude-drupal-12".into(),
                "Install WordPress in claude-wordpress-1".into(),
            ],
            "track_requirements",
        );
        assert!(plan.link_background_job(&[0], "job-drupal"));
        assert!(plan.link_background_job(&[1], "job-wordpress"));
        assert!(plan.note_job_folders(
            "job-drupal",
            &["/Users/x/projects/claude-company-drupal-5879".to_string()]
        ));
        assert!(plan.note_job_folders(
            "job-wordpress",
            &["/Users/x/projects/claude-wordpress-1".to_string()]
        ));
        assert!(plan.settle_background_job("job-drupal", JobEnd::Succeeded));
        assert!(plan.settle_background_job("job-wordpress", JobEnd::Succeeded));
        let card = plan.render_compact_checklist();
        assert!(
            card.contains(
                "✅ Install Drupal in claude-drupal-12 (made claude-company-drupal-5879)"
            ),
            "{card}"
        );
        assert!(
            card.contains("✅ Install WordPress in claude-wordpress-1\n")
                || card.ends_with("✅ Install WordPress in claude-wordpress-1"),
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

    fn parallel_installs() -> TaskPlan {
        let mut plan = TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            vec![
                "install the drupal site".into(),
                "install the wordpress site".into(),
                "report both".into(),
            ],
            "track_requirements",
        );
        // The model marked both installs handed off before either launch.
        plan.steps[0].status = StepStatus::Background;
        plan.steps[1].status = StepStatus::Background;
        plan
    }

    /// Live 2026-11-14: two parallel helpers, Drupal and WordPress. The
    /// Drupal launch named only its own step, yet the WordPress step, left
    /// waiting on the turn's launches, counted as already being done by the
    /// Drupal job and the WordPress launch was refused. Waiting on a run is
    /// not being done by it.
    #[test]
    fn a_second_parallel_launch_is_accepted_when_the_first_named_only_its_own_step() {
        let mut plan = parallel_installs();
        assert!(plan.link_background_job(&[0], "job-drupal"));
        // The checklist is posted again while Drupal runs: the open steps
        // wait on the turn's launch.
        plan.wait_on_background_jobs(&["job-drupal".to_string()]);
        let running = |_: &str| true;
        assert!(
            plan.steps_covered_by_running_jobs(&[1], running).is_empty(),
            "the WordPress step only waits on Drupal: {:?}",
            plan.steps[1]
        );
        assert_eq!(
            plan.steps_covered_by_running_jobs(&[0], running)
                .iter()
                .map(|covered| covered.index)
                .collect::<Vec<_>>(),
            vec![0],
            "Drupal's own step is still guarded"
        );
        assert!(plan.link_background_job(&[1], "job-wordpress"));
        // Drupal's end settles only its own step; WordPress's is its run's.
        assert!(plan.settle_background_job("job-drupal", JobEnd::Succeeded));
        assert_eq!(plan.steps[0].status, StepStatus::Completed);
        assert_eq!(plan.steps[1].status, StepStatus::Background);
        assert!(plan.settle_background_job("job-wordpress", JobEnd::Failed));
        assert_eq!(plan.steps[1].status, StepStatus::Failed);
    }

    /// A step that only waits on a running job never blocks a launch, named
    /// or not.
    #[test]
    fn a_step_that_only_waits_never_blocks_a_launch() {
        let mut plan = parallel_installs();
        plan.steps[0].status = StepStatus::Pending;
        plan.steps[1].status = StepStatus::Pending;
        plan.wait_on_background_jobs(&["job-first".to_string()]);
        let running = |_: &str| true;
        assert!(plan.steps_covered_by_running_jobs(&[], running).is_empty());
        assert!(plan.steps_covered_by_running_jobs(&[1], running).is_empty());
    }

    /// The WordPress step's own helper never started while it waited on
    /// Drupal's: Drupal's success does not make it done, and the header
    /// counts it as not done (live 2026-11-14: "Done · 4 of 4 steps").
    #[test]
    fn a_waiting_step_whose_own_helper_never_started_is_not_done() {
        let mut plan = parallel_installs();
        assert!(plan.link_background_job(&[0], "job-drupal"));
        plan.wait_on_background_jobs(&["job-drupal".to_string()]);
        assert!(plan.helper_did_not_start(&[1], "the launch was refused"));
        assert!(plan.settle_background_job("job-drupal", JobEnd::Succeeded));
        assert_eq!(plan.steps[0].status, StepStatus::Completed);
        assert_ne!(plan.steps[1].status, StepStatus::Completed);
        let card = plan.render_compact_checklist();
        assert!(!card.contains("Done ·"), "{card}");
        assert!(!card.contains("3 of 3"), "{card}");
        assert!(card.contains("didn't run"), "{card}");
        assert!(!card.contains("running in the background"), "{card}");
    }
}
