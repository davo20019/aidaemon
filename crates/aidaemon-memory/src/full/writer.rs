//! The full memory system's run-end writer: once a run's outcome is final,
//! extract durable facts from a completed turn and learn from the run
//! (expertise, procedures, error solutions).
//!
//! [`FullWriter`] decides what to write; the host says how the run ended and
//! runs the writes with its own models through [`WriterHost`]. The writes
//! themselves are here: [`extract_progressive`] (with the host's provenance
//! through [`ExtractionProvenance`]) and [`learn_from_run`].

use std::sync::Arc;

use aidaemon_core::{ChannelVisibility, Hook, HookContext, Provider, RunOutcome, UserRole};
use aidaemon_policies::TurnLearning;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tracing::{debug, warn};

use crate::progressive::{Exchange, ProgressiveExtractor, may_extract, owner_writes_during_run};
use crate::stores::MemoryStores;

/// What a finished run did, for learning.
#[derive(Debug, Clone, Default)]
pub struct RunLearning {
    pub user_text: String,
    /// False when the turn's policy suppresses automatic memory: nothing is
    /// learned.
    pub persistence_allowed: bool,
    pub intent_domains: Vec<String>,
    /// `"tool_name(summary)"` per call, in order.
    pub tool_calls: Vec<String>,
    /// Each error, and whether the run recovered from it.
    pub errors: Vec<(String, bool)>,
    /// The first error, for an error solution.
    pub first_error: Option<String>,
    pub recovery_actions: Vec<String>,
    pub completed_naturally: bool,
    pub explicit_positive_signals: u32,
    pub explicit_negative_signals: u32,
    /// The run's own verdict on its task, when it has one: authoritative.
    pub succeeded: Option<bool>,
}

impl RunLearning {
    /// Whether the run's task succeeded: its verdict, else the user's
    /// explicit signals, else a natural completion with every error
    /// recovered.
    pub fn task_success(&self) -> bool {
        if let Some(succeeded) = self.succeeded {
            succeeded
        } else if self.explicit_negative_signals > 0 {
            false
        } else if self.explicit_positive_signals > 0 {
            true
        } else {
            let unrecovered = self.errors.iter().filter(|(_, recovered)| !recovered);
            self.completed_naturally && unrecovered.count() == 0
        }
    }
}

/// Learn from a finished run: expertise for its domains, the procedure it
/// followed (successful or not, so repeated bad paths accumulate failures)
/// and, when it recovered from an error, the solution.
pub async fn learn_from_run(store: &dyn MemoryStores, run: RunLearning) -> anyhow::Result<()> {
    if !run.persistence_allowed {
        return Ok(());
    }
    let task_success = run.task_success();

    let domains = crate::detect_domains(&run.intent_domains);
    for domain in &domains {
        let error = if !task_success {
            run.errors.first().map(|(e, _)| e.as_str())
        } else {
            None
        };
        if let Err(e) = store.increment_expertise(domain, task_success, error).await {
            warn!(domain = %domain, error = %e, "Failed to update expertise");
        }
    }

    if run.tool_calls.len() >= 2 {
        let generalized = crate::generalize_procedure(&run.tool_calls);
        let base_name = crate::procedure_base_name(&generalized);
        let keyed_name = crate::procedure_keyed_name(&base_name, &generalized);
        let procedure = crate::new_procedure(
            keyed_name,
            crate::extract_trigger_pattern(&run.user_text),
            generalized,
            task_success,
        );
        if let Err(e) = store.upsert_procedure(&procedure).await {
            warn!(procedure = %procedure.name, error = %e, "Failed to save procedure");
        }
    }

    if let Some(error) = run.first_error.clone()
        && !run.recovery_actions.is_empty()
    {
        let solution = crate::new_error_solution(
            crate::error_pattern(&error),
            domains.into_iter().next(),
            crate::summarize_solution(&run.recovery_actions),
            Some(run.recovery_actions.clone()),
        );
        if let Err(e) = store.insert_error_solution(&solution).await {
            warn!(error_pattern = %solution.error_pattern, error = %e, "Failed to save error solution");
        }
    }

    Ok(())
}

/// Whether a completed turn may be extracted from and summarized:
/// `(extract, summarize)`. A turn with no memory (`no_memory`) or whose
/// policy forbids automatic memory gets neither. Memory-isolated sessions
/// (workers whose "user" turn is a delegated brief) keep their summaries
/// but never extract owner facts.
pub fn post_completion_plan(
    no_memory: bool,
    memory_isolated: bool,
    persistence_allowed: bool,
    progressive_facts: bool,
    worth_extracting: bool,
    summarization: bool,
) -> (bool, bool) {
    if no_memory || !persistence_allowed {
        return (false, false);
    }
    (
        !memory_isolated && progressive_facts && worth_extracting,
        summarization,
    )
}

/// One exchange to extract facts from.
#[derive(Debug, Clone)]
pub struct ExtractionRun {
    /// The host's id for the task, for provenance.
    pub task_id: String,
    pub user_text: String,
    pub reply: String,
    pub channel_id: Option<String>,
    pub visibility: ChannelVisibility,
    pub user_role: UserRole,
    /// Facts the agent stored itself since then are not extracted again.
    pub run_started_at: DateTime<Utc>,
}

/// What an extraction stored, for the host's audit trail.
#[derive(Debug, Clone)]
pub struct ExtractionRecord {
    pub source_task_id: String,
    /// The host's event the facts came from, if it found one.
    pub source_event_id: Option<i64>,
    pub channel_id: Option<String>,
    /// The facts written.
    pub facts: Vec<serde_json::Value>,
    /// The start of the user's message.
    pub source_excerpt: String,
}

/// Where extracted facts came from, kept by the host (its event log).
#[async_trait]
pub trait ExtractionProvenance: Send + Sync {
    /// The host's record of the user message `task_id` answered, bound to
    /// the facts just written. `None` when it has none.
    async fn source_event(&self, task_id: &str) -> anyhow::Result<Option<i64>>;

    /// Keep a record of one extraction's writes.
    async fn record(&self, record: ExtractionRecord) -> anyhow::Result<()>;
}

/// One extractor per process, so its two-call concurrency limit holds
/// across every turn. The provider is given per call (the host may switch
/// models), the limit is not.
fn extractor(provider: Arc<dyn Provider>, model: &str) -> ProgressiveExtractor {
    static PERMITS: std::sync::LazyLock<Arc<tokio::sync::Semaphore>> =
        std::sync::LazyLock::new(|| Arc::new(tokio::sync::Semaphore::new(2)));
    ProgressiveExtractor::with_permits(provider, model, PERMITS.clone())
}

/// Extract durable facts from one exchange and write them: the model
/// proposes, the store keeps what is new, and `provenance` (if any) binds
/// them to their source and records the write.
pub async fn extract_progressive(
    provider: Arc<dyn Provider>,
    model: &str,
    store: Arc<dyn MemoryStores>,
    provenance: Option<Arc<dyn ExtractionProvenance>>,
    run: ExtractionRun,
) {
    if !may_extract(run.user_role, run.visibility) {
        return;
    }
    let extractor = extractor(provider, model);
    let already_stored = owner_writes_during_run(store.as_ref(), run.run_started_at).await;
    let exchange = Exchange {
        user_text: &run.user_text,
        assistant_reply: &run.reply,
        channel_id: run.channel_id.as_deref(),
        already_stored: &already_stored,
    };
    let facts = match extractor.propose(&exchange).await {
        Ok(facts) => facts,
        Err(e) => {
            debug!(error = %e, "Progressive fact extraction failed");
            return;
        }
    };
    if facts.is_empty() {
        return; // Expected for most interactions.
    }
    let written = extractor.write(store.as_ref(), &exchange, facts).await;
    if written.is_empty() {
        return;
    }
    let Some(provenance) = provenance else {
        return;
    };
    let source_task_id = run.task_id.clone();
    let source_excerpt = crate::text::truncate_chars(&run.user_text, 200);
    let source_event_id = match provenance.source_event(&source_task_id).await {
        Ok(Some(event_id)) => {
            if let Err(error) = store.repair_memory_projections().await {
                debug!(%error, %source_task_id, "Failed to bind selected message provenance to extracted memory");
            }
            Some(event_id)
        }
        Ok(None) => {
            debug!(%source_task_id, "Selected memory write had no policy-eligible canonical user event");
            None
        }
        Err(error) => {
            debug!(%error, %source_task_id, "Failed to project selected message provenance");
            None
        }
    };
    let record = ExtractionRecord {
        source_task_id,
        source_event_id,
        channel_id: run.channel_id,
        facts: written,
        source_excerpt,
    };
    if let Err(e) = provenance.record(record).await {
        debug!(error = %e, "Failed to record memory_write event");
    }
}

/// Which writes a [`FullWriter`] owns.
#[derive(Debug, Clone, Copy)]
pub struct WriterSettings {
    /// Extract facts from a completed turn.
    pub progressive_facts: bool,
    /// Ask the host to summarize the conversation after a completed turn.
    pub summarization: bool,
    /// Learn from every run.
    pub learning: bool,
}

/// What the writer knows about its turn.
#[derive(Debug, Clone)]
pub struct WriterTurn {
    /// The turn has no memory at all (nothing is written or learned).
    pub no_memory: bool,
    /// A worker session: summaries, but no owner facts.
    pub memory_isolated: bool,
    /// The turn's policy allows automatic memory.
    pub persistence_allowed: bool,
    /// The user's message.
    pub user_text: String,
    /// Whether the turn may be learned from ("don't remember this").
    pub consent: TurnLearning,
}

/// How a run ended, as the host judges it.
#[derive(Debug, Clone)]
pub struct RunEnd {
    /// The reply the user saw.
    pub reply: String,
    /// The turn completed (its reply stands).
    pub completed: bool,
}

/// The host side of the run-end writer: how the run ended, and the writes
/// run with the host's models and records.
pub trait WriterHost: Send + Sync {
    /// The run as it ended. Called once, at run end.
    fn run_end(&self, outcome: &RunOutcome) -> RunEnd;
    /// Extract facts from the turn's exchange, in the background (see
    /// [`extract_progressive`]).
    fn extract_facts(&self, reply: &str, run_started_at: DateTime<Utc>);
    /// The host's conversation summary, after a completed turn. The
    /// conversation is the host's, so the writer only says when.
    fn summarize(&self) {}
    /// Learn from the run, in the background (see [`learn_from_run`]).
    fn learn(&self, completed: bool);
}

/// Writes memory once per run, after the outcome is final: extraction and
/// summarization only for a completed turn, learning for every run. A turn
/// opted out of learning is neither extracted nor learned from; its summary
/// is the host's to keep clean.
pub struct FullWriter {
    turn: WriterTurn,
    settings: WriterSettings,
    host: Arc<dyn WriterHost>,
    /// When the run began: the writer is built as the run is assembled,
    /// before any tool call, so every memory write the run makes is at or
    /// after it.
    run_started_at: DateTime<Utc>,
}

impl FullWriter {
    pub fn new(turn: WriterTurn, settings: WriterSettings, host: Arc<dyn WriterHost>) -> Self {
        Self {
            turn,
            settings,
            host,
            run_started_at: Utc::now(),
        }
    }
}

#[async_trait]
impl Hook for FullWriter {
    async fn on_run_end(&self, _ctx: &HookContext<'_>, outcome: &RunOutcome) {
        let turn = &self.turn;
        let end = self.host.run_end(outcome);
        if end.completed {
            let (extract, summarize) = post_completion_plan(
                turn.no_memory,
                turn.memory_isolated,
                turn.persistence_allowed,
                self.settings.progressive_facts,
                crate::progressive::worth_extracting(&turn.user_text),
                self.settings.summarization,
            );
            if extract && turn.consent.allowed() {
                self.host.extract_facts(&end.reply, self.run_started_at);
            }
            if summarize {
                self.host.summarize();
            }
        }
        if turn.no_memory || !self.settings.learning || !turn.consent.allowed() {
            return;
        }
        self.host.learn(end.completed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_turn_without_memory_writes_nothing() {
        assert_eq!(
            post_completion_plan(true, false, true, true, true, true),
            (false, false)
        );
        assert_eq!(
            post_completion_plan(false, false, true, true, true, true),
            (true, true)
        );
        assert_eq!(
            post_completion_plan(false, false, false, true, true, true),
            (false, false)
        );
        assert_eq!(
            post_completion_plan(false, true, true, true, true, true),
            (false, true)
        );
    }

    #[test]
    fn the_runs_verdict_decides_success_when_it_has_one() {
        let run = RunLearning {
            completed_naturally: true,
            succeeded: Some(false),
            ..RunLearning::default()
        };
        assert!(!run.task_success());
        let run = RunLearning {
            succeeded: None,
            ..run
        };
        assert!(run.task_success());
        let run = RunLearning {
            errors: vec![("boom".into(), false)],
            ..run
        };
        assert!(!run.task_success());
    }

    #[tokio::test]
    async fn learning_records_the_procedure_a_run_followed() {
        let store = crate::full::tools::test_support::memory().await;
        learn_from_run(
            store.as_ref(),
            RunLearning {
                user_text: "deploy the site".into(),
                persistence_allowed: true,
                tool_calls: vec!["terminal(git)".into(), "terminal(npm)".into()],
                completed_naturally: true,
                ..RunLearning::default()
            },
        )
        .await
        .unwrap();
        let procedures = crate::LearningStore::get_relevant_procedures(store.as_ref(), "deploy", 5)
            .await
            .unwrap();
        assert_eq!(procedures.len(), 1, "{procedures:?}");
    }
}
