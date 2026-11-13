//! [`DaemonMemoryWriter`]: post-turn memory and learning as a run-end hook,
//! the full memory system's writer with the daemon as its host.

use std::sync::{Arc, Mutex};

use super::memory::MemoryTurn;
use super::post_task::LearningContext;
use aidaemon_core as sdk;
use aidaemon_events::TaskStatus;

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
}

/// Which writes a writer owns: the memory plugin owns fact extraction; the
/// turn owns summarization and learning.
pub use aidaemon_memory::full::writer::WriterSettings;

/// The full memory system's run-end writer
/// ([`FullWriter`](aidaemon_memory::full::writer::FullWriter)) on a daemon
/// turn: extraction and summarization only for a completed turn, learning for
/// every run. A turn opted out of learning is neither extracted nor learned
/// from; its summary leaves the turn out (the turn's events are marked
/// suppressed).
pub struct DaemonMemoryWriter;

impl DaemonMemoryWriter {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        turn: MemoryTurn,
        settings: WriterSettings,
        writes: Arc<dyn MemoryWrites>,
    ) -> aidaemon_memory::full::writer::FullWriter {
        let writer_turn = aidaemon_memory::full::writer::WriterTurn {
            no_memory: turn.mandate,
            memory_isolated: turn.memory_isolated(),
            persistence_allowed: turn.memory_persistence_allowed,
            user_text: turn.user_text.clone(),
            consent: turn.learning_consent.clone(),
        };
        aidaemon_memory::full::writer::FullWriter::new(
            writer_turn,
            settings,
            Arc::new(DaemonRunEnd {
                turn,
                writes,
                ended: Mutex::new(None),
            }),
        )
    }
}

/// The daemon as the writer's host: the turn's verdict decides how the run
/// ended; the writes go to [`MemoryWrites`].
struct DaemonRunEnd {
    turn: MemoryTurn,
    writes: Arc<dyn MemoryWrites>,
    /// The learning context as the run ended, and the run's outcome.
    ended: Mutex<Option<(LearningContext, aidaemon_events::TaskOutcome)>>,
}

impl aidaemon_memory::full::writer::WriterHost for DaemonRunEnd {
    fn run_end(&self, outcome: &sdk::RunOutcome) -> aidaemon_memory::full::writer::RunEnd {
        let turn = &self.turn;
        let learning = turn
            .learning
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let (reply, status, task_outcome) =
            turn.verdict
                .final_reply(outcome, &learning.tool_calls, turn.channel_ctx.visibility);
        *self
            .ended
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((learning, task_outcome));
        aidaemon_memory::full::writer::RunEnd {
            reply: aidaemon_tools::sanitize::sanitize_user_facing_reply(&reply),
            completed: status == TaskStatus::Completed,
        }
    }

    fn extract_facts(&self, reply: &str, run_started_at: chrono::DateTime<chrono::Utc>) {
        self.writes.extract_facts(&self.turn, reply, run_started_at);
    }

    fn summarize(&self) {
        self.writes.summarize(&self.turn);
    }

    fn learn(&self, completed: bool) {
        let ended = self
            .ended
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        let Some((mut learning, task_outcome)) = ended else {
            return;
        };
        learning.completed_naturally = completed;
        learning.task_outcome = Some(task_outcome);
        self.writes.learn(learning);
    }
}
