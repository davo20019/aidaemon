//! One turn's state on the kernel: everything the host's handlers read
//! and update while the run goes, built once per turn by the host.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};

use crate::turn::closeout::StopDemand;

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
    /// `Agent::tools`); see `Agent::turn_tool`.
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
    /// What each stop-time ask has spent in this run (`StopDemand`).
    pub stop_budgets: StopBudgets,
    /// Declared items the run ended without attempting, after its one pass
    /// (index, the step in the model's own words): their typed reason, "not
    /// attempted", reaches the card and the reply.
    /// Each with its checklist index; `None`: one of the owner's own units
    /// of work, which has no step on the card.
    pub not_attempted: std::sync::Mutex<Vec<(Option<usize>, String)>>,
    /// The plan's usage limit stopped the chosen model and another answered:
    /// the reply says so, once per limit in a conversation.
    pub answered_past_quota: std::sync::Mutex<Option<crate::llm::turn_calls::QuotaFallback>>,
    /// The model that wrote the latest successful answer (a failover's
    /// when one answered): the reply is recorded under it, so a client can
    /// show which model wrote it.
    pub answered_by: std::sync::Mutex<Option<String>>,
    /// A call that fell back from the model it asked for: (that model, the
    /// one that answered). The turn's later calls for the same model start
    /// at the one that answered, instead of asking the failed one again
    /// first and bouncing between them.
    pub took_over: std::sync::Mutex<Option<(String, String)>>,
    /// The latest closeout let the run stop after a demand while the ledger
    /// still held reachable expectations: the reply in hand was written to
    /// the daemon's demand, not to the user.
    pub closeout_ended_unmet: AtomicBool,
    /// The sender's own message was taken into the run since the
    /// closeout's latest demand: the reply after it answers them, so a
    /// demand that ended unmet before it no longer makes that reply one
    /// written to the daemon (live 2026-11-17: "focus on the AID part",
    /// taken mid-run, drew a reply the stale flag sent back to be rewritten
    /// text-only, and the run ended saying the revision was not made).
    pub user_spoke_since_demand: AtomicBool,
    /// `completed_results` at the last checklist demand (`u64::MAX`: none).
    pub executor_demanded_at: AtomicU64,
    /// Checklist items the model has not finished or set aside, as last
    /// declared through `track_requirements`.
    pub open_checklist_items: AtomicU64,
    /// The one checklist item that changes something and that the model
    /// marked `in_progress` in its latest checklist (`u64::MAX`: none, or
    /// more than one): where a change a call made without naming an item
    /// goes ([`aidaemon_tools::semantics::ToolCallMetadata::changes_untyped`]).
    pub in_progress_change_item: AtomicU64,
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
    /// The next model call runs with tools disabled (empty-reply retry).
    pub text_only_next_call: Arc<AtomicBool>,
    /// The usage of the latest model call, until its budget check takes it.
    pub pending_charge: std::sync::Mutex<Option<aidaemon_events::provider::TokenUsage>>,
    /// Where calls without a directory run; refined by the model's own
    /// explicit directories (see `SdkTurnSetup::known_project_dir`).
    pub known_project_dir: Arc<std::sync::Mutex<Option<String>>>,
    /// The coding agent this conversation is set to (root turns the owner
    /// sent only): its model calls are answered without a model, the first
    /// with the one `cli_agent` run that takes the owner's message.
    pub direct_agent: Option<String>,
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
    /// The project whose memory the turn keeps (`memory_project`), passed
    /// to tools as `ToolCallContext::project`.
    pub project: Option<aidaemon_core::ProjectRef>,
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
    pub untrusted_seen: Arc<AtomicBool>,
    /// Owned fallback judge for detached/background coding-agent runs whose
    /// permission requests arrive after the launching tool batch returns.
    pub detached_judge: Option<Arc<crate::approvals::DetachedNestedJudge>>,
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
    /// Source files this run changed in a repository since its last
    /// command (`StopDemand::CodeChangeUnchecked`).
    pub unchecked_code: std::sync::Mutex<Vec<String>>,
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
    /// Directives and mid-turn user messages, with the run position (an
    /// index into the kernel's run messages) they arrived at.
    pub placed_directives: std::sync::Mutex<Vec<(usize, Value)>>,
    /// Mid-turn messages taken into the run, and whether a model call
    /// answered each; unanswered ones are re-queued at run end.
    pub absorbed_steering: std::sync::Mutex<Vec<AbsorbedSteering>>,
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
    /// launch order (each call's typed `background_run_labels`): the turn's
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

/// Whether a file a run wrote is code its project can build or test: not
/// prose (documentation, notes, plain text), and inside a git repository.
pub fn is_checkable_code(path: &std::path::Path) -> bool {
    const PROSE: [&str; 7] = ["md", "markdown", "mdx", "txt", "rst", "adoc", "org"];
    let prose = path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| PROSE.iter().any(|p| ext.eq_ignore_ascii_case(p)));
    !prose
        && path
            .ancestors()
            .skip(1)
            .any(|dir| dir.join(".git").exists())
}

#[cfg(test)]
mod checkable_code_tests {
    use super::is_checkable_code;

    #[test]
    fn code_in_a_repository_counts_and_prose_or_loose_files_do_not() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("acme-site");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(repo.join("src")).unwrap();
        assert!(is_checkable_code(&repo.join("src/contact.rs")));
        assert!(is_checkable_code(&repo.join("web/contact.module")));
        assert!(!is_checkable_code(&repo.join("README.md")));
        assert!(!is_checkable_code(&repo.join("notes/Plan.TXT")));
        assert!(!is_checkable_code(&tmp.path().join("loose.rs")));
    }
}

/// What each stop-time ask has spent in this run, one row per
/// [`StopDemand`]: budgets live in [`StopDemand::budget`], spends here.
pub struct StopBudgets([std::sync::atomic::AtomicU32; StopDemand::ALL.len()]);

impl Default for StopBudgets {
    fn default() -> Self {
        Self(std::array::from_fn(|_| {
            std::sync::atomic::AtomicU32::new(0)
        }))
    }
}

impl StopBudgets {
    /// How many times `demand` was asked in this run.
    pub fn used(&self, demand: StopDemand) -> u32 {
        self.0[demand.index()].load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether `demand` has no ask left.
    pub fn spent(&self, demand: StopDemand) -> bool {
        self.used(demand) >= demand.budget()
    }

    /// Count one ask of `demand`; returns the asks so far.
    pub fn spend(&self, demand: StopDemand) -> u32 {
        self.0[demand.index()].fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
    }

    /// Spend one ask of `demand` if any is left.
    pub fn try_spend(&self, demand: StopDemand) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let row = &self.0[demand.index()];
        let mut used = row.load(Relaxed);
        while used < demand.budget() {
            match row.compare_exchange_weak(used, used + 1, Relaxed, Relaxed) {
                Ok(_) => return true,
                Err(now) => used = now,
            }
        }
        false
    }
}

#[cfg(test)]
mod stop_budget_tests {
    use super::{StopBudgets, StopDemand};

    /// Every ask has its own row, and an ask past its budget is refused.
    #[test]
    fn each_ask_spends_its_own_budget() {
        for (row, demand) in StopDemand::ALL.into_iter().enumerate() {
            assert_eq!(demand.index(), row, "{demand:?} out of order");
        }
        let budgets = StopBudgets::default();
        assert!(budgets.try_spend(StopDemand::PartialWithoutObstacle));
        assert!(!budgets.try_spend(StopDemand::PartialWithoutObstacle));
        assert!(budgets.spent(StopDemand::PartialWithoutObstacle));
        assert_eq!(budgets.used(StopDemand::ReplyRepair), 0);
        for _ in 0..3 {
            assert!(budgets.try_spend(StopDemand::GuiClickVerification));
        }
        assert!(budgets.spent(StopDemand::GuiClickVerification));
    }
}
