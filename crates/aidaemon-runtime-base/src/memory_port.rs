//! The turn's memory port and run records, below the host: what a memory
//! system is handed for a turn ([`MemoryTurn`]) and what it adds
//! ([`MemoryParts`]), the run-end writer seam ([`MemoryWrites`]), what a run
//! teaches ([`LearningContext`]), how the turn judges its run
//! ([`TurnVerdict`], whose resolution lives in `engine::outcome`), the
//! compaction store and the execution budget. The modules that use them
//! (`turn::memory`, `turn::post_task`, ...) re-export them.

use std::sync::Arc;

use aidaemon_channels::context::ChannelContext;
use aidaemon_core as sdk;
use aidaemon_events::TaskOutcome;
use aidaemon_tools::semantics::UserRole;
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};

/// What is known about a turn when its message arrives: enough to start
/// recall, not yet its task or its memory policy.
#[derive(Clone)]
pub struct MemoryWarmup {
    pub session_id: String,
    pub user_text: String,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    pub depth: usize,
    /// The project the conversation is placed in, when it may keep project
    /// memory (`memory_project`).
    pub project: Option<aidaemon_core::ProjectRef>,
}

/// Everything a plugin needs to scope memory for one turn.
#[derive(Clone)]
pub struct MemoryTurn {
    pub session_id: String,
    pub task_id: String,
    /// The conversation turn this run answers. It is over when the run ends,
    /// before its `task_end` is appended.
    pub turn_id: String,
    /// The turn's event emitter (decision points, memory telemetry).
    pub emitter: aidaemon_events::EventEmitter,
    /// The persisted user message for the turn.
    pub user_text: String,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    /// The compiled policy allows reading memory.
    pub memory_pipeline_allowed: bool,
    /// The compiled policy allows writing memory.
    pub memory_persistence_allowed: bool,
    /// A mandate worker: no memory at all.
    pub mandate: bool,
    pub depth: usize,
    pub model: String,
    /// The run's learning context; the dispatcher adds its tool calls.
    pub learning: Arc<std::sync::Mutex<LearningContext>>,
    /// How the turn judges the run's final status.
    pub verdict: Arc<TurnVerdict>,
    /// Whether the turn may be learned from ("don't remember this"), shared
    /// with whatever memory system serves the turn.
    pub learning_consent: aidaemon_policies::TurnLearning,
    /// The project the turn works in, when it may keep project memory
    /// (`memory_project`); `None` otherwise.
    pub project: Option<aidaemon_core::ProjectRef>,
}

/// What a plugin adds to one turn.
#[derive(Default)]
pub struct MemoryParts {
    pub hooks: Vec<Arc<dyn sdk::Hook>>,
    pub tools: Vec<Arc<dyn sdk::Tool>>,
    /// Daemon tools this plugin owns; offered only while it is the active
    /// plugin. Filled from [`MemoryPlugin::owned_daemon_tools`].
    pub owned_daemon_tools: &'static [&'static str],
}

#[async_trait]
pub trait MemoryPlugin: Send + Sync {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts;

    /// Starts recall for a message as it arrives, before the turn's memory
    /// policy is settled. It must write nothing and return at once, and what
    /// it reads may reach the turn only through [`MemoryPlugin::turn`], when
    /// that turn may read memory and recalls for the same inputs.
    fn warm(&self, _warmup: &MemoryWarmup) {}

    /// The daemon memory tools (`DAEMON_MEMORY_TOOLS`) this plugin owns.
    fn owned_daemon_tools(&self) -> &'static [&'static str] {
        &[]
    }

    /// This is the daemon's own memory, so the daemon's prompts about facts
    /// and its learning from finished runs apply. Another memory system may
    /// have neither.
    fn is_daemons_own(&self) -> bool {
        false
    }
}

/// The daemon's memory side effects, behind a seam so the writer's decisions
/// are testable without spawning extraction.
pub trait MemoryWrites: Send + Sync {
    /// Progressive fact extraction from the user text and the final reply.
    /// Facts the agent stored itself since `run_started_at` are not
    /// extracted a second time.
    fn extract_facts(
        &self,
        turn: &MemoryTurn,
        reply: &str,
        run_started_at: chrono::DateTime<chrono::Utc>,
    );
    /// Incremental conversation summarization.
    fn summarize(&self, turn: &MemoryTurn);
    /// Post-task learning (procedures, expertise, failure patterns).
    fn learn(&self, learning: LearningContext);
    /// Durable knowledge from a completed delegated task's activity log,
    /// extracted with `model`; a tainted run teaches nothing. What applies
    /// only to `project` (the project the work belongs to) is kept as its
    /// memory when the host keeps project memory.
    fn learn_from_task(
        &self,
        task: aidaemon_work::traits::Task,
        model: String,
        taint: RunTaint,
        project: Option<aidaemon_core::ProjectRef>,
    );
}

/// Whether the run a task finished in read untrusted content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunTaint {
    Clean,
    Tainted,
}

#[derive(Clone)]
pub struct LearningContext {
    pub user_text: String,
    /// False when the task's compiled capability policy suppresses all
    /// automatic memory persistence. Canonical task/event telemetry is still
    /// recorded; only learned procedures, expertise, and failure patterns are
    /// skipped.
    pub memory_persistence_allowed: bool,
    pub intent_domains: Vec<String>,
    pub tool_calls: Vec<LearnedCall>,
    pub errors: Vec<(String, bool)>, // (error_text, was_recovered)
    pub first_error: Option<String>,
    pub recovery_actions: Vec<String>,
    pub start_time: chrono::DateTime<Utc>,
    pub completed_naturally: bool,
    /// Semantic outcome emitted at task end; authoritative for learning when set.
    pub task_outcome: Option<TaskOutcome>,
}

/// One call as post-task learning records it: its `tool_name(summary)`
/// line (marked when it was rejected or failed, as memory reads it), and the
/// typed facts the recap counts it by: the tool's `recap` kind and whether
/// it ran to completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearnedCall {
    pub line: String,
    /// What the call's arguments say, in a few words.
    pub summary: String,
    pub recap: aidaemon_tools::semantics::RecapKind,
    /// Refused before it ran.
    pub rejected: bool,
    /// Ran and failed.
    pub failed: bool,
}

/// What a run settled, beyond its kernel status, that decides the turn's
/// final status and outcome. Shared with run-end hooks (memory writers), so
/// they judge a run exactly as the turn does.
#[derive(Default)]
pub struct TurnVerdict {
    /// The reply and outcome a used-up goal or mandate budget closes the run
    /// with; the run itself ends through the provider port.
    pub budget_stop: std::sync::Mutex<Option<(String, TaskOutcome)>>,
    /// Confirmation of the latest successful external change.
    pub external_action_ack: std::sync::Mutex<Option<String>>,
    /// Set when the run must end failed despite a final answer.
    pub run_failure: std::sync::Mutex<Option<String>>,
    /// The latest provider note (for example a safety finish reason).
    pub response_note: std::sync::Mutex<Option<String>>,
    /// The latest well-formed final answer a closeout demand sent back for
    /// more work. If the model then replies with nothing, this ships instead
    /// of a generic activity summary.
    pub interrupted_answer: std::sync::Mutex<Option<String>>,
    /// The handoff reply for a run that stopped while a job it launched was
    /// still running: that job's notifier delivers the result, so the run
    /// tells the user the work continues instead of shipping the model's own
    /// verdict on unfinished work.
    pub handoff_reply: std::sync::Mutex<Option<String>>,
    /// The coding agents this turn started answered it themselves
    /// ([`crate::run_state::RunStates::agents_answered`]). When the model
    /// did nothing after them, the turn ships no reply of its own: each
    /// agent's message, its report checked, is the answer, and the model's
    /// closing words would only repeat it (live 2026-11-23: told not to, it
    /// restated a clean result two runs of two).
    pub answered_by_agents: std::sync::atomic::AtomicBool,
    /// The files those agents' runs changed, by their records: reading them
    /// back after the agents is no work of the turn's own.
    pub agents_edited: std::sync::Mutex<Vec<String>>,
    /// The turn of a conversation set to a coding agent: its reply is what
    /// the agent's run returned, as it is (no reply checks, no activity
    /// summary in its place).
    pub direct_agent: std::sync::atomic::AtomicBool,
}

/// What a compaction reads and writes besides the turns: the summary table,
/// and the settings that record a summary's untrusted-content taint.
pub trait CompactionStore:
    aidaemon_events::stores::ConversationSummaryStore + aidaemon_tools::services::SettingsStore
{
}

impl<T> CompactionStore for T where
    T: aidaemon_events::stores::ConversationSummaryStore
        + aidaemon_tools::services::SettingsStore
        + ?Sized
{
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionBudget {
    pub max_steps: usize,
    pub max_tokens: usize,
    pub max_llm_calls: usize,
    pub max_tool_calls: usize,
    pub max_validation_rounds: usize,
    pub max_wall_clock_ms: u64,
}

impl MemoryTurn {
    /// Worker sessions keep summaries but never learn owner facts.
    pub fn memory_isolated(&self) -> bool {
        self.depth > 0 || aidaemon_events::is_memory_isolated_worker_session(&self.session_id)
    }
}

impl LearnedCall {
    pub fn new(
        name: &str,
        summary: String,
        recap: aidaemon_tools::semantics::RecapKind,
        rejected: bool,
        failed: bool,
    ) -> Self {
        let line = format!("{name}({summary})");
        let line = if rejected {
            format!("{line} [REJECTED]")
        } else if failed {
            format!("{line} [FAILED]")
        } else {
            line
        };
        Self {
            line,
            summary,
            recap,
            rejected,
            failed,
        }
    }

    /// The call ran to completion: activity a recap may claim.
    pub fn completed(&self) -> bool {
        !self.rejected && !self.failed
    }
}

impl LearningContext {
    /// What the full memory system learns from: this run, with the first
    /// error stripped of the daemon's scaffolding lines.
    pub fn into_run_learning(self) -> aidaemon_memory::full::writer::RunLearning {
        aidaemon_memory::full::writer::RunLearning {
            user_text: self.user_text,
            persistence_allowed: self.memory_persistence_allowed,
            intent_domains: self.intent_domains,
            tool_calls: self.tool_calls.into_iter().map(|call| call.line).collect(),
            errors: self.errors,
            first_error: self.first_error.map(|error| {
                error
                    .lines()
                    .filter(|line| !crate::message_content::is_internal_scaffolding_line(line))
                    .collect::<Vec<_>>()
                    .join("\n")
            }),
            recovery_actions: self.recovery_actions,
            completed_naturally: self.completed_naturally,
            succeeded: self.task_outcome.map(|outcome| outcome.task_success()),
        }
    }
}

impl TurnVerdict {
    /// The run ends on a used-up budget: the next model call fails with the
    /// budget's marker, and the turn ships `reply` with `outcome`.
    pub fn record_budget_stop(&self, reply: String, outcome: TaskOutcome) {
        if let Ok(mut stop) = self.budget_stop.lock() {
            *stop = Some((reply, outcome));
        }
    }

    pub fn record_handoff_reply(&self, reply: String) {
        if let Ok(mut handoff) = self.handoff_reply.lock() {
            *handoff = Some(reply);
        }
    }

    pub fn record_interrupted_answer(&self, answer: String) {
        if let Ok(mut interrupted) = self.interrupted_answer.lock() {
            *interrupted = Some(answer);
        }
    }

    pub fn run_failure(&self) -> Option<String> {
        self.run_failure
            .lock()
            .ok()
            .and_then(|failure| failure.clone())
    }
}
