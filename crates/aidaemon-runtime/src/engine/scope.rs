//! One turn's state on the kernel: everything the host's handlers read
//! and update while the run goes, built once per turn by the host.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};

use serde_json::Value;

use crate::engine::outcome::TurnVerdict;

/// A message the same sender sent while the run was going, taken into it
/// (`sdk_take_steering`). It is answered once a model call that carried it,
/// with tools offered, returned; one never answered goes back to the channel
/// queue at run end (`sdk_return_unanswered_steering`).
pub struct AbsorbedSteering {
    pub message: aidaemon_channels::tasks::QueuedMessage,
    pub answered: bool,
}

/// The goal whose token budget this turn charges.
pub struct BudgetGoal {
    pub goal_id: String,
    /// Scheduled runs charge their per-run budget instead of the daily one.
    pub scheduled: bool,
}

/// Everything a dispatched call needs from the turn.
pub struct TurnScope<'a> {
    pub session_id: &'a str,
    pub daemon_paths: &'a aidaemon_tools::path_class::DaemonPaths,
    pub execution_backend: &'a dyn aidaemon_tools::ExecutionBackend,
    /// The MCP-registry tools bridged for this turn (daemon tools are
    /// `Agent::tools`); see [`Agent::turn_tool`].
    pub mcp_tools: Vec<Arc<dyn aidaemon_tools::OutcomeTool>>,
    /// Redacts secrets (the SDK's shapes and the configured literals) from
    /// the arguments of calls the kernel refused before they are written to
    /// the ledger.
    pub ledger_redactor: aidaemon_policies::Redactor,
    /// The user's message for this turn.
    pub user_text: &'a str,
    pub task_id: &'a str,
    pub emitter: &'a aidaemon_events::EventEmitter,
    /// The projected starting transcript, sent byte-for-byte on every call.
    pub prefix: Vec<Value>,
    /// Kernel messages that stand for `prefix`; later ones are this run's.
    pub prefix_len: usize,
    pub budget_goal: Option<BudgetGoal>,
    /// Set once the goal budget is used up; later model calls are refused.
    pub model_calls: AtomicU64,
    /// The verify-once and failure-recovery passes have been used.
    pub verification_demanded: AtomicBool,
    pub recovery_demanded: AtomicBool,
    /// The run was sent back once because declared work was never
    /// attempted (`RunAggregate::unattempted_declared_work`).
    pub unattempted_work_demanded: AtomicBool,
    /// Declared items the run ended without attempting, after its one pass
    /// (index, the step in the model's own words): their typed reason, "not
    /// attempted", reaches the card and the reply.
    /// Each with its checklist index; `None`: one of the owner's own units
    /// of work, which has no step on the card.
    pub not_attempted: std::sync::Mutex<Vec<(Option<usize>, String)>>,
    /// Extra passes the ledger demanded so far.
    pub closeout_demands: std::sync::atomic::AtomicU32,
    /// The latest closeout let the run stop after a demand while the ledger
    /// still held reachable expectations: the reply in hand was written to
    /// the daemon's demand, not to the user.
    pub closeout_ended_unmet: AtomicBool,
    /// `completed_results` at the last checklist demand (`u64::MAX`: none).
    pub executor_demanded_at: AtomicU64,
    /// Checklist items the model has not finished or set aside, as last
    /// declared through `track_requirements`.
    pub open_checklist_items: AtomicU64,
    /// The model has declared a checklist in this run, finished or not.
    pub checklist_declared: AtomicBool,
    pub status_tx: Option<tokio::sync::mpsc::Sender<aidaemon_tools::status::StatusUpdate>>,
    pub channel_ctx: &'a aidaemon_channels::context::ChannelContext,
    pub user_role: aidaemon_tools::semantics::UserRole,
    /// The engine model, for vision gating of tool-observed images.
    pub model: &'a str,
    pub turn_id: &'a str,
    /// The turn's contract: its expectations are credited from receipts.
    pub contract: &'a crate::turn::completion_contract::CompletionContract,
    pub progress: std::sync::Mutex<crate::turn::completion_contract::CompletionProgress>,
    /// Shared with the memory writer, which learns from it at run end.
    pub learning: Arc<std::sync::Mutex<crate::turn::post_task::LearningContext>>,
    /// The turn's learning consent, handed to the sub-agents it spawns.
    pub learning_consent: aidaemon_policies::TurnLearning,
    /// The model-visible output of the latest tool call, for paste checks.
    pub latest_tool_output: std::sync::Mutex<Option<String>>,
    /// The one reply retry has been spent.
    pub reply_retry_used: AtomicBool,
    /// The next model call runs with tools disabled (empty-reply retry).
    pub text_only_next_call: Arc<AtomicBool>,
    /// The usage of the latest model call, until its budget check takes it.
    pub pending_charge: std::sync::Mutex<Option<aidaemon_events::provider::TokenUsage>>,
    /// Where calls without a directory run; refined by the model's own
    /// explicit directories (see `SdkTurnSetup::known_project_dir`).
    pub known_project_dir: Arc<std::sync::Mutex<Option<String>>>,
    /// Terminal calls in the canonical form the kernel's approval rules
    /// judged, by run and call id: the dispatcher executes exactly these,
    /// with the scope they were judged with.
    pub judged_terminal_calls: crate::approvals::JudgedCalls,
    /// How `CanonicalCallHook` canonicalized each call, by run and call id.
    pub canonical_calls: crate::engine::tool_calls::CanonicalCalls,
    /// What the dispatcher's gates decided for each call, by run and call
    /// id (`DispatchGate`); execution takes a cleared call's grant here.
    pub gate_outcomes: crate::engine::tool_calls::GateOutcomes,
    /// Calls that ran and wait for the after-tool hook to record them.
    pub executed_calls: crate::engine::tool_calls::ExecutedCalls,
    /// The run's declines as the kernel last showed them to the approval
    /// rules: a child spawned from this turn starts with them.
    pub run_declines: crate::approvals::RunDeclines,
    /// The live decline set this turn's run shares with its parent (when it
    /// is a child) and with the children it spawns: the kernel refuses any
    /// of them an effect the owner declined in another.
    pub shared_declines: aidaemon_core::SharedDeclines,
    /// The request's sub-agent tree (a root turn starts one).
    pub spawn_family: crate::authority::SpawnFamily,
    pub run_id: aidaemon_core::RunId,
    /// The single-project scope lock passed to tools.
    pub project_scope: Option<String>,
    /// See `SdkTurnSetup::authority`.
    pub authority: &'a crate::authority::ResolvedAuthority,
    /// The chat's live `/readonly` ceiling, read by the hook per call and by
    /// the dispatcher's `mutation_forbidden`.
    pub ceiling: Arc<crate::authority::SessionModeCeiling>,
    /// The owner's trusted file roots: configured roots, the working
    /// directory, the project scope and directory, and the task's write
    /// grants.
    pub trust_roots: Vec<String>,
    /// Where relative file paths resolve (see `turn_cwd`).
    pub cwd: String,
    /// A dispatched call returned untrusted content this turn.
    pub untrusted_seen: AtomicBool,
    /// The root request egress is judged against (a child's is its root's),
    /// and whether the owner wrote it.
    pub origin_request: &'a str,
    pub origin_trusted: bool,
    pub authorized_scopes: Vec<String>,
    pub scopes_are_guess: bool,
    /// See `SdkTurnSetup::projected_source_message_ids`.
    pub projected_source_message_ids: Vec<String>,
    pub projected_source_turn_ids: Vec<String>,
    /// Terminal replies, external acks and run failures: the turn's verdict.
    pub verdict: Arc<TurnVerdict>,
    /// Daemon memory tools the active memory plugin owns on this turn.
    pub owned_memory_tools: &'static [&'static str],
    /// Tool calls in the latest model response (a direct response needs one).
    pub last_batch_size: std::sync::atomic::AtomicUsize,
    /// Skills referencing untrusted external content are active.
    pub restrict_untrusted_reference_tools: bool,
    pub untrusted_reference_skills: Vec<String>,
    /// Distinct successful operations (tool and canonical arguments).
    pub succeeded_operations: std::sync::Mutex<HashSet<String>>,
    /// The latest successful call repeated an earlier operation.
    pub repeating: AtomicBool,
    /// Distinct successful operations so far (the request's whole tree
    /// counts its own in its task budget, [`crate::authority::SpawnFamily::budget`]).
    pub succeeded_calls: std::sync::atomic::AtomicUsize,
    /// The goal's effective daily budget after extensions, and their count.
    pub goal_daily_budget: std::sync::Mutex<Option<i64>>,
    pub goal_budget_extensions: std::sync::atomic::AtomicU32,
    /// The one ask to check a file more than one member changed was made.
    pub combined_verification_demands: std::sync::atomic::AtomicU32,
    /// Mandate decision correction turns used.
    pub mandate_commit_retries: std::sync::atomic::AtomicU32,
    /// Verify-before-claim demands for an unverified GUI coordinate click.
    pub gui_verify_demands: std::sync::atomic::AtomicU32,
    /// A stop held for the run's unreported jobs waited its longest once:
    /// the next stop is not held again.
    pub unreported_jobs_wait_spent: std::sync::atomic::AtomicBool,
    /// The one 80% budget warning (task or scheduled run) was given.
    pub budget_warning_sent: AtomicBool,
    /// Timed-out model calls retried in this run.
    pub model_call_timeouts: std::sync::atomic::AtomicU32,
    /// The latest model call was rejected as too long for the model's
    /// window (now recorded); the call is retried once compacted.
    pub context_overflowed: AtomicBool,
    pub context_overflow_retries: std::sync::atomic::AtomicU32,
    /// The typed error of the latest model call that failed after recovery;
    /// the kernel keeps only its text.
    pub provider_failure: std::sync::Mutex<Option<crate::chat_bridge::ProviderError>>,
    /// `send_file` dedupe keys that already succeeded in this task.
    pub sent_files: std::sync::Mutex<HashSet<String>>,
    /// Nested project instructions discovered just in time.
    pub instruction_tracker:
        Option<tokio::sync::Mutex<crate::turn::project_instructions::ProjectInstructionTracker>>,
    /// System context raised during the run and not yet sent; the next model
    /// call pins it at the run's current end (`placed_directives`).
    pub extra_system: std::sync::Mutex<Vec<String>>,
    /// When the turn started, and the notices delivered since then that a
    /// model call already carried: each model call reads the new ones, so
    /// the reply agrees with what the owner was told meanwhile.
    pub notices_seen: std::sync::Mutex<(
        chrono::DateTime<chrono::Utc>,
        std::collections::HashSet<String>,
    )>,
    /// Directives and mid-turn user messages, with the run position they
    /// arrived at.
    pub placed_directives: std::sync::Mutex<Vec<(usize, Value)>>,
    /// Mid-turn messages taken into the run, and whether a model call
    /// answered each; unanswered ones are re-queued at run end.
    pub absorbed_steering: std::sync::Mutex<Vec<AbsorbedSteering>>,
    /// The run's summarized older rounds, once it outgrew the context.
    pub compaction: std::sync::Mutex<Option<crate::engine::compaction::RunCompaction>>,
    /// Set when a call moved work to the background with completion
    /// notifications on: the daemon will deliver its result later, so the
    /// reply is a background handoff, not the terminal answer.
    pub background_handoff: AtomicBool,
    /// Background jobs this run launched with completion notifications on
    /// (not detached services) whose end it has not observed: each typed
    /// launch receipt adds one, each typed settle receipt (the job's result
    /// read back, or the job cancelled) takes one away. While any is open, a
    /// stop is a handoff: the job's result reaches the session through its
    /// notifier, so the run must not close the request on its own.
    pub open_background_jobs: AtomicUsize,
    /// The first line of the latest call that left a job running for this
    /// run ("Started in the background: …"): the reply of a run that stops
    /// while the job runs.
    pub background_ack: std::sync::Mutex<Option<String>>,
    /// The names of the runs this turn left running in the background, in
    /// launch order (each call's typed `background_run_label`): the turn's
    /// launch line names them all
    /// ([`crate::background_run::started_group_in_background`]).
    pub background_runs: std::sync::Mutex<Vec<String>>,
    /// The latest helper the run waited on, once it ended: from then on
    /// any change the run makes itself is its own work after the helper,
    /// measured to the run's end, and the request's total is measured to
    /// the answer.
    pub after_helper: std::sync::Mutex<Option<AfterHelper>>,
}

/// The run's work after a helper it waited on ended, from typed records:
/// when the helper ended, its name, and whether the run then changed
/// anything itself (a later call that succeeded and was not read-only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AfterHelper {
    pub since: chrono::DateTime<chrono::Utc>,
    /// The name the owner reads for the helper.
    pub helper: String,
    /// The run finished work itself after the helper.
    pub did_work: bool,
    /// How many helpers the run has waited on so far, this one included.
    pub helpers: usize,
}
