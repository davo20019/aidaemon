//! Learning from what tasks did: a host's task history as typed trace
//! events, and the procedures, error solutions and expertise memory learns
//! from them.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use aidaemon_core::{Message, ModelRequest, Provider};
use chrono::{DateTime, Utc};
use serde_json::Value;
use tracing::{info, warn};

use crate::embed::Embedder;
use crate::learning::{ErrorSolution, Procedure, embedding_of, error_pattern};
use crate::stores::MemoryStores;
use crate::text::truncate_chars;

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
        /// The call's arguments, when the host's record of the call is whole.
        arguments: Option<Value>,
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
    /// A message from the user. A host leaves out the ones it wrote itself
    /// (continuations of a run, for example).
    UserMessage { content: String, at: DateTime<Utc> },
    /// A check the run went through.
    DecisionPoint(DecisionPoint),
}

impl TraceEvent {
    /// The task whose work this event records. User messages and decision
    /// points are not work: they never affect which tasks exist, their
    /// order, or how far an error looks for its recovery.
    fn work_task_id(&self) -> Option<&str> {
        match self {
            Self::TaskStarted { task_id, .. } | Self::TaskEnded { task_id, .. } => Some(task_id),
            Self::ToolCalled { task_id, .. }
            | Self::ToolReturned { task_id, .. }
            | Self::Failed { task_id, .. } => task_id.as_deref(),
            Self::UserMessage { .. } | Self::DecisionPoint(_) => None,
        }
    }

    fn is_work(&self) -> bool {
        !matches!(self, Self::UserMessage { .. } | Self::DecisionPoint(_))
    }
}

/// A check a task's run went through, with what the host recorded about it.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionPoint {
    pub task_id: String,
    /// A stable name for the kind of check, such as `tool_budget_block`.
    pub code: String,
    /// Whether it was at least a warning. Anything lighter teaches nothing.
    pub warning: bool,
    pub details: DecisionDetails,
    /// How the host's last end record for the task says it ended; `None`
    /// when it has none or cannot read it.
    pub task_ending: Option<TaskEnding>,
}

/// The fields of a decision point the failure patterns read. Absent or
/// mistyped values read as their default.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DecisionDetails {
    pub tool: Option<String>,
    pub count: u64,
    pub action: Option<String>,
    pub consecutive_count: u64,
    pub unique_args: u64,
    pub is_diverse: bool,
    pub tools: Vec<String>,
    pub prior_semantic_failures: u64,
    pub prior_calls: u64,
    pub required_evidence_kind: Option<String>,
    pub target: Option<String>,
    pub stall_mode: Option<String>,
}

impl DecisionDetails {
    /// The fields of a host's free-form decision metadata.
    pub fn from_metadata(metadata: &Value) -> Self {
        let text = |key: &str| {
            metadata
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        let number = |key: &str| metadata.get(key).and_then(Value::as_u64).unwrap_or(0);
        Self {
            tool: text("tool"),
            count: number("count"),
            action: text("action"),
            consecutive_count: number("consecutive_count"),
            unique_args: number("unique_args"),
            is_diverse: metadata
                .get("is_diverse")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            tools: metadata
                .get("tools")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            prior_semantic_failures: number("prior_semantic_failures"),
            prior_calls: number("prior_calls"),
            required_evidence_kind: text("required_evidence_kind"),
            target: text("target"),
            stall_mode: text("stall_mode"),
        }
    }
}

/// A plan the session finished, with the steps that completed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletedPlan {
    pub id: String,
    pub description: String,
    pub completed_steps: Vec<String>,
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
        if let Some(task_id) = event.work_task_id() {
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
    let work: Vec<&TraceEvent> = trace.iter().filter(|event| event.is_work()).collect();
    let mut solutions = Vec::new();
    for (index, event) in work.iter().copied().enumerate() {
        let TraceEvent::Failed { message, tool, .. } = event else {
            continue;
        };
        let recovery = work
            .iter()
            .copied()
            .take(work.len().min(index + 10))
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

/// A lesson from a decision point that signals trouble.
#[derive(Debug, Clone, PartialEq)]
pub struct FailurePattern {
    pub trigger_context: String,
    pub action: String,
    pub description: String,
    pub confidence: f32,
}

/// The failure patterns in `trace`: one per warning-level decision point
/// with a known code, and per task at most one for each trigger and action.
pub fn failure_patterns(trace: &[TraceEvent]) -> Vec<FailurePattern> {
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut patterns = Vec::new();
    for (index, event) in trace.iter().enumerate() {
        let TraceEvent::DecisionPoint(decision) = event else {
            continue;
        };
        if !decision.warning {
            continue;
        }
        let recovery_tool = trace[index + 1..].iter().find_map(|next| match next {
            TraceEvent::ToolReturned {
                task_id: Some(task_id),
                name,
                succeeded: true,
                ..
            } if *task_id == decision.task_id => Some(name.as_str()),
            _ => None,
        });
        let Some(pattern) = failure_pattern(decision, recovery_tool) else {
            continue;
        };
        if seen.insert((
            decision.task_id.clone(),
            pattern.trigger_context.clone(),
            pattern.action.clone(),
        )) {
            patterns.push(pattern);
        }
    }
    patterns
}

fn failure_pattern(
    decision: &DecisionPoint,
    recovery_tool: Option<&str>,
) -> Option<FailurePattern> {
    let code = decision.code.as_str();
    let details = &decision.details;
    let tool = details.tool.as_deref().unwrap_or("unknown_tool");
    let recovery_suffix = recovery_tool
        .map(|tool| format!(" Previous successful recoveries switched to {tool}."))
        .unwrap_or_default();
    let confidence = |base: f32| failure_confidence(base, decision.task_ending, recovery_tool);
    let pattern = |trigger_context: String, action: &str, description: String, base: f32| {
        Some(FailurePattern {
            trigger_context,
            action: action.to_string(),
            description,
            confidence: confidence(base),
        })
    };

    match code {
        "repetitive_call_detection" => {
            let count = details.count;
            let hard_stop = details.action.as_deref().unwrap_or("redirect") == "hard_stop";
            let description = if count > 0 {
                format!(
                    "After {count} repeated calls to {tool}, stop retrying the same operation and pivot to a different tool or summarize the blocker.{recovery_suffix}"
                )
            } else {
                format!(
                    "Repeated calls to {tool} tend to become a dead-end loop; pivot to a different tool or summarize the blocker sooner.{recovery_suffix}"
                )
            };
            pattern(
                format!("{code}:{tool}"),
                if hard_stop {
                    "stop_retrying_identical_calls"
                } else {
                    "pivot_to_alternate_tool_or_strategy"
                },
                description,
                if hard_stop { 0.74 } else { 0.62 },
            )
        }
        "consecutive_same_tool_detection" => {
            let description = if details.is_diverse {
                format!(
                    "Even varied {tool} calls can become a loop after {} consecutive uses; switch tools or summarize progress before continuing.{recovery_suffix}",
                    details.consecutive_count
                )
            } else {
                format!(
                    "Long {tool} streaks with only {} distinct argument sets usually indicate a loop; change tools or adjust strategy before continuing.{recovery_suffix}",
                    details.unique_args
                )
            };
            pattern(
                format!("{code}:{tool}"),
                "switch_tools_before_long_streaks",
                description,
                if details.is_diverse { 0.54 } else { 0.68 },
            )
        }
        "alternating_pattern_detection" => {
            let mut tools = details.tools.clone();
            if tools.is_empty() {
                tools.push("unknown_pair".to_string());
            }
            tools.sort();
            let tool_pair = tools.join(" <-> ");
            pattern(
                format!("{code}:{}", tools.join("+")),
                "break_alternating_tool_loops",
                format!(
                    "Alternating between {tool_pair} with low call diversity usually indicates a loop; commit to one path or choose a third tool instead of bouncing between them.{recovery_suffix}"
                ),
                0.72,
            )
        }
        "tool_budget_block" => {
            let semantic_failures = details.prior_semantic_failures;
            let description = if semantic_failures > 0 {
                format!(
                    "Once {tool} has repeated the same semantic failure {semantic_failures} times, stop retrying it and switch strategy earlier.{recovery_suffix}"
                )
            } else {
                format!(
                    "After {} repeated {tool} calls without enough progress, stop retrying the blocked tool and switch strategy earlier.{recovery_suffix}",
                    details.prior_calls
                )
            };
            pattern(
                format!("{code}:{tool}"),
                "stop_retrying_blocked_tool",
                description,
                if semantic_failures > 0 { 0.74 } else { 0.68 },
            )
        }
        "missing_pre_execution_evidence" => {
            let evidence_kind = details
                .required_evidence_kind
                .as_deref()
                .unwrap_or("direct evidence");
            let target = details.target.as_deref().unwrap_or("current target");
            pattern(
                format!("{code}:{tool}:{evidence_kind}"),
                "gather_direct_evidence_before_mutation",
                format!(
                    "Before using {tool}, gather {evidence_kind} for {target} instead of mutating from memory or inference alone.{recovery_suffix}"
                ),
                0.82,
            )
        }
        "plan_rejected" => pattern(
            code.to_string(),
            "replan_first_risky_step_before_execution",
            format!(
                "When the structured first-step plan is rejected, revise the target or first action before executing again.{recovery_suffix}"
            ),
            0.74,
        ),
        "critique_rejected" => pattern(
            code.to_string(),
            "address_critique_before_retry",
            format!(
                "If the critique pass rejects the first risky action, fix the missing evidence or target assumptions before retrying.{recovery_suffix}"
            ),
            0.76,
        ),
        "target_scope_violation" => pattern(
            format!("{code}:{tool}"),
            "confirm_target_scope_before_mutation",
            format!(
                "If {tool} attempts to act outside the compiled target scope, stop and confirm the correct repo/file target before retrying.{recovery_suffix}"
            ),
            0.86,
        ),
        "tool_contract_violation" => pattern(
            format!("{code}:{tool}"),
            "fix_tool_arguments_before_retry",
            format!(
                "Deterministic argument-contract failures on {tool} should be corrected locally before retrying the step.{recovery_suffix}"
            ),
            0.72,
        ),
        "contradictory_file_evidence" => pattern(
            code.to_string(),
            "recheck_conflicting_state_before_completion",
            format!(
                "When file evidence contradicts itself, re-read the target before claiming completion instead of trusting stale state.{recovery_suffix}"
            ),
            0.80,
        ),
        "verification_pending" | "verification_unavailable_in_phase" => pattern(
            code.to_string(),
            "run_verification_before_claiming_success",
            format!(
                "Do not claim success while verification is pending; either run the check or surface a precise partial result.{recovery_suffix}"
            ),
            0.82,
        ),
        "validation_budget_exhausted" | "execution_budget_exhausted" => pattern(
            code.to_string(),
            "reduce_scope_when_budget_exhausts",
            format!(
                "When execution or validation budget is exhausted, reduce scope or checkpoint progress instead of pushing through unverified completion.{recovery_suffix}"
            ),
            0.70,
        ),
        "pre_tool_deferral_stall" => pattern(
            code.to_string(),
            "ask_clarifying_question_or_take_next_tool_step",
            format!(
                "Repeatedly deferring before taking any tool step leads to failure; ask one concrete clarification question or take the next tool action earlier.{recovery_suffix}"
            ),
            0.72,
        ),
        "stall" => pattern(
            format!(
                "{code}:{}",
                details.stall_mode.as_deref().unwrap_or("generic")
            ),
            "summarize_progress_or_change_strategy",
            format!(
                "When iterations stop producing new evidence, stop looping, summarize what is known, and change strategy instead of continuing the same pattern.{recovery_suffix}"
            ),
            0.76,
        ),
        "hard_iteration_cap" => pattern(
            code.to_string(),
            "finish_before_hard_iteration_cap",
            format!(
                "As the hard iteration cap approaches, stop repeating work and return the best partial answer or next concrete step earlier.{recovery_suffix}"
            ),
            0.60,
        ),
        "task_timeout" => pattern(
            code.to_string(),
            "checkpoint_and_reduce_scope_before_timeout",
            format!(
                "Long-running tasks that approach the timeout should checkpoint progress and reduce scope earlier instead of continuing the same loop.{recovery_suffix}"
            ),
            0.64,
        ),
        "task_token_budget" => pattern(
            code.to_string(),
            "reduce_token_spend_and_checkpoint",
            format!(
                "When task token spend stays high, reduce scope, checkpoint findings, or stop earlier instead of consuming the full task budget.{recovery_suffix}"
            ),
            0.68,
        ),
        "scheduled_run_budget" => pattern(
            code.to_string(),
            "checkpoint_scheduled_run_before_budget_exhaustion",
            format!(
                "Scheduled runs should bail out once progress slows; checkpoint the current state instead of burning the full per-run token budget.{recovery_suffix}"
            ),
            0.70,
        ),
        "goal_daily_token_budget" => pattern(
            code.to_string(),
            "checkpoint_goal_work_before_daily_budget_exhaustion",
            format!(
                "For recurring goal work, avoid spending the full daily budget in one pass; checkpoint progress and defer the rest earlier.{recovery_suffix}"
            ),
            0.70,
        ),
        "daily_token_budget" => pattern(
            code.to_string(),
            "checkpoint_before_daily_budget_exhaustion",
            format!(
                "When the session is nearing the daily token budget, checkpoint useful progress and stop earlier instead of exhausting the remaining budget.{recovery_suffix}"
            ),
            0.70,
        ),
        "task_token_budget_warning" => pattern(
            code.to_string(),
            "reduce_token_spend_before_budget_warning",
            format!(
                "High token-spend phases should trigger earlier summarization or scope reduction before the task budget is exhausted.{recovery_suffix}"
            ),
            0.45,
        ),
        "soft_iteration_warning" => pattern(
            code.to_string(),
            "adjust_strategy_before_soft_iteration_warning",
            format!(
                "When the soft iteration warning appears, summarize progress or change strategy instead of continuing the same loop.{recovery_suffix}"
            ),
            0.45,
        ),
        _ => None,
    }
}

fn failure_confidence(
    base: f32,
    task_ending: Option<TaskEnding>,
    recovery_tool: Option<&str>,
) -> f32 {
    let ending_bonus = match task_ending {
        Some(TaskEnding::Failed) => 0.08,
        Some(TaskEnding::Partial) => 0.04,
        Some(TaskEnding::Succeeded) | None => 0.0,
    };
    let recovery_bonus = if recovery_tool.is_some() { 0.03 } else { 0.0 };
    (base + ending_bonus + recovery_bonus).clamp(0.1, 0.96)
}

/// What one pass over a trace wrote.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LearningReport {
    pub procedures: usize,
    pub error_solutions: usize,
    pub expertise_updates: usize,
    pub failure_patterns: usize,
}

/// The tool that writes facts in the daemon's own memory, whose calls say
/// which user message a fact came from.
const REMEMBER_FACT_TOOL: &str = "remember_fact";

/// Learns from what a session's tasks did and writes it through the stores:
/// procedures, error solutions, expertise, failure patterns and the
/// provenance of facts written along the way.
pub struct TaskLearner {
    stores: Arc<dyn MemoryStores>,
    embedder: Option<Arc<dyn Embedder>>,
    enhancer: Option<ProcedureEnhancer>,
    evidence_gate: bool,
}

impl TaskLearner {
    pub fn new(stores: Arc<dyn MemoryStores>) -> Self {
        Self {
            stores,
            embedder: None,
            enhancer: None,
            evidence_gate: false,
        }
    }

    /// Recognizes duplicates and stores trigger embeddings with `embedder`.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Rewrites procedures into reusable ones with `enhancer`.
    pub fn with_enhancer(mut self, enhancer: ProcedureEnhancer) -> Self {
        self.enhancer = Some(enhancer);
        self
    }

    /// Keeps only procedures with enough evidence ([`meets_evidence_gate`]).
    pub fn with_evidence_gate(mut self, enforce: bool) -> Self {
        self.evidence_gate = enforce;
        self
    }

    /// Learn from `trace` and the session's finished `plans`. A row that
    /// cannot be written is logged and skipped.
    pub async fn learn(&self, trace: &[TraceEvent], plans: &[CompletedPlan]) -> LearningReport {
        let mut report = LearningReport::default();
        self.learn_procedures(trace, &mut report).await;
        self.learn_plans(plans, &mut report).await;
        for solution in error_solutions(trace) {
            match self
                .stores
                .record_error_solution_observation(&solution)
                .await
            {
                Ok(()) => report.error_solutions += 1,
                Err(e) => warn!("Failed to save error solution: {e}"),
            }
        }
        for (domain, success) in expertise_updates(trace) {
            match self.stores.record_task_outcome(&domain, success).await {
                Ok(()) => report.expertise_updates += 1,
                Err(e) => warn!("Failed to update expertise: {e}"),
            }
        }
        for pattern in failure_patterns(trace) {
            match self
                .stores
                .record_behavior_pattern(
                    "failure",
                    &pattern.description,
                    Some(&pattern.trigger_context),
                    Some(&pattern.action),
                    pattern.confidence,
                    1,
                )
                .await
            {
                Ok(()) => report.failure_patterns += 1,
                Err(error) => warn!(
                    trigger_context = %pattern.trigger_context,
                    action = %pattern.action,
                    %error,
                    "Failed to record failure pattern from decision point"
                ),
            }
        }
        self.stamp_fact_provenance(trace).await;
        report
    }

    /// Candidates are all checked against what was stored before this pass,
    /// and enhanced, before the first one is saved.
    async fn learn_procedures(&self, trace: &[TraceEvent], report: &mut LearningReport) {
        let mut procedures = Vec::new();
        for candidate in procedure_candidates(trace) {
            if self.is_duplicate(&candidate.description).await {
                continue;
            }
            let procedure = procedure_from_task(&candidate);
            procedures.push(self.enhanced(procedure, &candidate.description).await);
        }
        for procedure in procedures {
            if self.evidence_gate && !meets_evidence_gate(&procedure) {
                info!(
                    trigger = %procedure.trigger_pattern,
                    steps = procedure.steps.len(),
                    "Skipping procedure without sufficient evidence"
                );
                continue;
            }
            match self.stores.record_procedure_observation(&procedure).await {
                Ok(()) => {
                    self.store_trigger_embedding(&procedure).await;
                    report.procedures += 1;
                }
                Err(e) => warn!("Failed to save procedure: {e}"),
            }
        }
    }

    /// Plans go one by one, so each sees the ones saved before it.
    async fn learn_plans(&self, plans: &[CompletedPlan], report: &mut LearningReport) {
        for plan in plans {
            if plan.completed_steps.len() < 2 || self.is_duplicate(&plan.description).await {
                continue;
            }
            let Some(procedure) =
                procedure_from_plan(&plan.id, &plan.description, plan.completed_steps.clone())
            else {
                continue;
            };
            let procedure = self.enhanced(procedure, &plan.description).await;
            if self.evidence_gate && !meets_evidence_gate(&procedure) {
                info!(
                    trigger = %procedure.trigger_pattern,
                    "Skipping plan-based procedure without evidence"
                );
                continue;
            }
            match self.stores.record_procedure_observation(&procedure).await {
                Ok(()) => {
                    self.store_trigger_embedding(&procedure).await;
                    report.procedures += 1;
                }
                Err(e) => warn!("Failed to save plan-based procedure: {e}"),
            }
        }
    }

    /// Whether a stored procedure's trigger is close to `description`. Any
    /// failure to tell reads as not a duplicate.
    async fn is_duplicate(&self, description: &str) -> bool {
        let Some(embedder) = &self.embedder else {
            return false;
        };
        let Ok(new_embedding) = embedding_of(embedder.as_ref(), description).await else {
            return false;
        };
        let Ok(existing) = self.stores.procedure_trigger_embeddings().await else {
            return false;
        };
        duplicates_existing_trigger(&new_embedding, &existing)
    }

    /// `procedure` rewritten by the model, or unchanged when there is no
    /// model or it declines.
    async fn enhanced(&self, procedure: Procedure, task_description: &str) -> Procedure {
        let Some(enhancer) = &self.enhancer else {
            return procedure;
        };
        enhancer
            .enhance(&procedure, task_description)
            .await
            .unwrap_or(procedure)
    }

    async fn store_trigger_embedding(&self, procedure: &Procedure) {
        let Some(embedder) = &self.embedder else {
            return;
        };
        if let Ok(embedding) = embedding_of(embedder.as_ref(), &procedure.trigger_pattern).await {
            let _ = self
                .stores
                .set_procedure_trigger_embedding(&procedure.name, &embedding)
                .await;
        }
    }

    /// Mark each fact written by the remember tool as stated by the user
    /// (their last message contains the value) or derived.
    async fn stamp_fact_provenance(&self, trace: &[TraceEvent]) {
        let mut last_user_message: Option<(&str, DateTime<Utc>)> = None;
        for event in trace {
            match event {
                TraceEvent::UserMessage { content, at } => {
                    last_user_message = Some((content, *at));
                }
                TraceEvent::ToolCalled {
                    name,
                    arguments: Some(arguments),
                    ..
                } if name == REMEMBER_FACT_TOOL => {
                    for (category, key, value) in remembered_facts(arguments) {
                        let mut source = "derived";
                        let mut source_excerpt = None;
                        let mut first_seen_at = None;
                        if let Some((message, at)) = last_user_message {
                            let lower_value = value.to_lowercase();
                            if lower_value.len() > 3
                                && message.to_lowercase().contains(&lower_value)
                            {
                                source = "user_stated";
                            }
                            source_excerpt = Some(truncate_chars(message, 200));
                            first_seen_at = Some(at);
                        }
                        let _ = self
                            .stores
                            .stamp_fact_provenance(
                                &category,
                                &key,
                                source,
                                first_seen_at,
                                source_excerpt.as_deref(),
                            )
                            .await;
                    }
                }
                _ => {}
            }
        }
    }
}

/// The `(category, key, value)` entries of a remember call: its `facts`
/// list when it has one, otherwise the call's own fields.
fn remembered_facts(arguments: &Value) -> Vec<(String, String, String)> {
    let entry = |fact: &Value| {
        let field = |name: &str| fact.get(name).and_then(Value::as_str).map(str::to_string);
        Some((field("category")?, field("key")?, field("value")?))
    };
    match arguments.get("facts").and_then(Value::as_array) {
        Some(facts) => facts.iter().filter_map(entry).collect(),
        None => entry(arguments).into_iter().collect(),
    }
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
    use aidaemon_core::testing::{ScriptedProvider, text};

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
                arguments: None,
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

    fn decision(code: &str, metadata: Value, ending: Option<TaskEnding>) -> TraceEvent {
        TraceEvent::DecisionPoint(DecisionPoint {
            task_id: "t1".into(),
            code: code.into(),
            warning: true,
            details: DecisionDetails::from_metadata(&metadata),
            task_ending: ending,
        })
    }

    fn returned(task_id: &str, name: &str, succeeded: bool) -> TraceEvent {
        TraceEvent::ToolReturned {
            task_id: Some(task_id.into()),
            name: name.into(),
            succeeded,
            result: "done".into(),
        }
    }

    #[test]
    fn failure_confidence_adds_outcome_and_recovery_and_stays_clamped() {
        let trace = vec![
            decision(
                "repetitive_call_detection",
                serde_json::json!({"tool": "terminal", "count": 4, "action": "hard_stop"}),
                Some(TaskEnding::Failed),
            ),
            returned("t1", "terminal", false),
            returned("other", "web_fetch", true),
            returned("t1", "edit_file", true),
        ];
        let patterns = failure_patterns(&trace);
        assert_eq!(patterns.len(), 1);
        assert_eq!(
            patterns[0].trigger_context,
            "repetitive_call_detection:terminal"
        );
        assert_eq!(patterns[0].action, "stop_retrying_identical_calls");
        assert!(
            patterns[0]
                .description
                .ends_with("Previous successful recoveries switched to edit_file.")
        );
        assert!((patterns[0].confidence - (0.74 + 0.08 + 0.03)).abs() < 1e-5);
        assert!(
            (failure_confidence(0.95, Some(TaskEnding::Failed), Some("x")) - 0.96).abs() < 1e-6
        );
    }

    #[test]
    fn failure_patterns_drop_unknown_codes_light_severities_and_repeats() {
        let mut light = decision("stall", serde_json::json!({}), None);
        if let TraceEvent::DecisionPoint(decision) = &mut light {
            decision.warning = false;
        }
        let trace = vec![
            light,
            decision("not_a_known_code", serde_json::json!({}), None),
            decision("stall", serde_json::json!({"stall_mode": "idle"}), None),
            decision("stall", serde_json::json!({"stall_mode": "idle"}), None),
        ];
        let patterns = failure_patterns(&trace);
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].trigger_context, "stall:idle");
    }

    #[test]
    fn decision_metadata_reads_mistyped_values_as_defaults() {
        let details = DecisionDetails::from_metadata(&serde_json::json!({
            "tool": 7, "count": "four", "is_diverse": "yes", "tools": ["a", 1, "b"]
        }));
        assert_eq!(details.tool, None);
        assert_eq!(details.count, 0);
        assert!(!details.is_diverse);
        assert_eq!(details.tools, ["a", "b"]);
    }

    /// Decision points and user messages are not work: an error still finds
    /// a recovery nine work events away however many sit in between.
    #[test]
    fn only_work_counts_toward_the_recovery_window() {
        let failed = TraceEvent::Failed {
            task_id: None,
            message: "boom".into(),
            tool: None,
        };
        let mut trace = vec![failed.clone()];
        for _ in 0..12 {
            trace.push(decision("stall", serde_json::json!({}), None));
            trace.push(TraceEvent::UserMessage {
                content: "still there?".into(),
                at: Utc::now(),
            });
        }
        trace.push(returned("t1", "terminal", true));
        assert_eq!(error_solutions(&trace).len(), 1);

        let mut too_far = vec![failed];
        too_far.extend(std::iter::repeat_n(returned("t1", "terminal", false), 9));
        too_far.push(returned("t1", "terminal", true));
        assert!(error_solutions(&too_far).is_empty());
    }

    /// A task's first decision point must not move it ahead of a task that
    /// started working earlier.
    #[test]
    fn decision_points_do_not_reorder_tasks() {
        let mut trace = vec![decision("stall", serde_json::json!({}), None)];
        trace.extend(task("t2", TaskEnding::Succeeded, &["a", "b"]));
        trace.extend(task("t1", TaskEnding::Succeeded, &["a", "b"]));
        let order: Vec<String> = procedure_candidates(&trace)
            .into_iter()
            .map(|candidate| candidate.task_id)
            .collect();
        assert_eq!(order, ["t2", "t1"]);
    }

    #[test]
    fn truncate_cuts_on_a_character_boundary() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(
            truncate("hello world, this is a long string", 10),
            "hello w..."
        );
    }

    mod learner {
        use std::sync::Arc;

        use sqlx::SqlitePool;
        use sqlx::sqlite::SqlitePoolOptions;

        use super::super::*;
        use crate::sqlite::SqliteMemory;
        use crate::stores::{FactStore, MemoryStores};
        use crate::{HashEmbedder, LearningStore};

        async fn memory() -> (SqlitePool, SqliteMemory) {
            let pool = SqlitePoolOptions::new()
                .max_connections(1)
                .connect("sqlite::memory:")
                .await
                .unwrap();
            let embedder = Arc::new(HashEmbedder::new(384));
            crate::sqlite::migrate(&pool).await.unwrap();
            (pool.clone(), SqliteMemory::new(pool, embedder))
        }

        fn stores(memory: &SqliteMemory) -> Arc<dyn MemoryStores> {
            Arc::new(memory.clone())
        }

        fn remember(arguments: serde_json::Value, name: &str) -> TraceEvent {
            TraceEvent::ToolCalled {
                task_id: None,
                name: name.into(),
                summary: None,
                arguments: Some(arguments),
            }
        }

        #[tokio::test]
        async fn a_known_procedure_gains_a_success_and_keeps_its_definition() {
            let (pool, memory) = memory().await;
            let trace = super::task(
                "task-one",
                TaskEnding::Succeeded,
                &["terminal", "edit_file"],
            );
            let learner = TaskLearner::new(stores(&memory));
            assert_eq!(learner.learn(&trace, &[]).await.procedures, 1);
            let mut changed = super::task("task-one", TaskEnding::Succeeded, &["a", "b", "c"]);
            if let TraceEvent::TaskStarted { description, .. } = &mut changed[0] {
                *description = "another description".into();
            }
            assert_eq!(learner.learn(&changed, &[]).await.procedures, 1);
            let row: (String, i64, String) =
                sqlx::query_as("SELECT trigger_pattern, success_count, steps FROM procedures")
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            assert_eq!(row.0, "Do task-one");
            assert_eq!(row.1, 2);
            assert!(row.2.contains("edit_file"));
        }

        #[tokio::test]
        async fn expertise_counts_and_leaves_the_level_alone() {
            let (pool, memory) = memory().await;
            let learner = TaskLearner::new(stores(&memory));
            let trace = super::task("t1", TaskEnding::Failed, &["web_search"]);
            learner.learn(&trace, &[]).await;
            learner.learn(&trace, &[]).await;
            let row: (i64, i64, i64, String, f64) = sqlx::query_as(
                "SELECT tasks_attempted, tasks_succeeded, tasks_failed, current_level, confidence_score
                 FROM expertise WHERE domain = 'research'",
            )
            .fetch_one(&pool)
            .await
            .unwrap();
            assert_eq!((row.0, row.1, row.2), (2, 0, 2));
            assert_eq!(row.3, "novice");
            assert!((row.4 - 0.1).abs() < 1e-6);
        }

        #[tokio::test]
        async fn duplicates_of_stored_triggers_are_skipped_only_with_an_embedder() {
            let (_pool, memory) = memory().await;
            let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(384));
            let learner = TaskLearner::new(stores(&memory)).with_embedder(embedder);
            let plan = CompletedPlan {
                id: "plan-1".into(),
                description: "Deploy the Acme Corp site".into(),
                completed_steps: vec!["build".into(), "ship".into()],
            };
            let duplicate = CompletedPlan {
                id: "plan-2".into(),
                ..plan.clone()
            };
            let report = learner.learn(&[], &[plan, duplicate]).await;
            assert_eq!(report.procedures, 1);
            assert_eq!(
                memory.procedure_trigger_embeddings().await.unwrap().len(),
                1
            );
        }

        #[tokio::test]
        async fn remember_calls_stamp_provenance_and_other_tools_do_not() {
            let (pool, memory) = memory().await;
            for key in ["city", "team", "untouched"] {
                memory
                    .upsert_fact(
                        "profile",
                        key,
                        "value",
                        "agent",
                        None,
                        crate::records::FactPrivacy::Global,
                    )
                    .await
                    .unwrap();
            }
            let at = Utc::now();
            let trace = vec![
                remember(
                    serde_json::json!({"category": "profile", "key": "team", "value": "x"}),
                    "remember_fact",
                ),
                TraceEvent::UserMessage {
                    content: "Alice Rivera lives in Lisbon".into(),
                    at,
                },
                remember(
                    serde_json::json!({"facts": [
                        {"category": "profile", "key": "city", "value": "LISBON"}
                    ]}),
                    "remember_fact",
                ),
                remember(
                    serde_json::json!({"category": "profile", "key": "untouched", "value": "Lisbon"}),
                    "another_memory_tool",
                ),
            ];
            TaskLearner::new(stores(&memory)).learn(&trace, &[]).await;
            let rows: Vec<(String, String, Option<String>, Option<String>)> = sqlx::query_as(
                "SELECT key, source, first_seen_at, source_excerpt FROM facts ORDER BY key",
            )
            .fetch_all(&pool)
            .await
            .unwrap();
            assert_eq!(rows[0].0, "city");
            assert_eq!(rows[0].1, "user_stated");
            assert_eq!(rows[0].2.as_deref(), Some(at.to_rfc3339().as_str()));
            assert_eq!(rows[0].3.as_deref(), Some("Alice Rivera lives in Lisbon"));
            assert_eq!(rows[1].0, "team");
            assert_eq!(rows[1].1, "derived");
            assert_eq!(rows[1].2, None);
            assert_eq!(rows[2].1, "agent");
        }

        #[tokio::test]
        async fn the_backfill_embeds_what_has_no_current_embedding() {
            let (pool, memory) = memory().await;
            sqlx::query(
                "INSERT INTO error_solutions (error_pattern, domain, solution_summary, created_at)
                 VALUES ('connection refused', '', 'Used terminal to resolve', '2026-01-01')",
            )
            .execute(&pool)
            .await
            .unwrap();
            let embedder = HashEmbedder::new(384);
            assert!(
                crate::backfill_learning_embeddings(&memory, &embedder)
                    .await
                    .unwrap()
            );
            assert!(
                !crate::backfill_learning_embeddings(&memory, &embedder)
                    .await
                    .unwrap()
            );
            let embedded: (Option<Vec<u8>>,) =
                sqlx::query_as("SELECT error_embedding FROM error_solutions")
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            assert!(embedded.0.is_some());
        }
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
