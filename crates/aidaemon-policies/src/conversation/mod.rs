//! Conversation compaction across runs: once a conversation nears the
//! model's context window, older whole turns are folded into a rolling
//! summary and the model sees that summary instead of them.
//!
//! | Piece | Role |
//! |---|---|
//! | [`compaction_threshold`], [`wait_threshold`], [`verbatim_tokens`], [`turns_to_summarize`] | The policy: when to compact and which turns to fold in |
//! | [`Compactor`] | Runs a compaction: per-session serialization, background passes with coalescing, waiting |
//! | [`ConversationSource`] | Where the turns come from (a run transcript, an event store) |
//! | [`SummaryStore`] | Where the summary lives ([`InMemorySummaries`], or your database) |
//! | [`Summarizer`] | Who writes it ([`ProviderSummarizer`]: a structured, provenance-carrying state) |
//! | [`ConversationCompaction`] | Hook for agents whose transcript carries the whole conversation |
//! | [`SummaryNote`] | Hook that only places the summary, for apps that already omit summarized turns |

mod compactor;
mod hooks;
mod summarizer;

use async_trait::async_trait;

use aidaemon_core::Message;

use crate::context_window::estimate_tokens;

pub use compactor::{CompactionJobs, CompactionLimits, Compactor};
pub use hooks::{render_summary_note, ConversationCompaction, SummaryNote};
pub use summarizer::{elide_middle, ProviderSummarizer, COMPACTED_STATE_HEADINGS};

/// History size, in tokens, at which a conversation is compacted after a
/// run: 60% of the model's context window.
pub fn compaction_threshold(window_tokens: usize) -> usize {
    window_tokens.saturating_mul(3) / 5
}

/// History size at which a model call waits for compaction instead of
/// leaving it to the pass after the run: 80% of the window.
pub fn wait_threshold(window_tokens: usize) -> usize {
    window_tokens.saturating_mul(4) / 5
}

/// Tokens of recent whole turns kept verbatim after compacting at
/// `threshold`: a quarter of it.
pub fn verbatim_tokens(threshold: usize) -> usize {
    threshold / 4
}

/// How many leading turns of the unsummarized history to fold into the
/// summary. `turn_tokens` are the turns' sizes, oldest first.
///
/// Nothing while the history is under `threshold`. Otherwise the longest
/// run of leading turns that still leaves at least `keep_tokens` of whole
/// turns verbatim. The newest turn is always kept; while it is still in
/// progress (`newest_finished` false) the turn before it is kept too, so
/// the model sees the exchange it is continuing. A run's end is what
/// finishes its turn: an app that records completion later still passes
/// `true` from `on_run_end`.
pub fn turns_to_summarize(
    turn_tokens: &[usize],
    newest_finished: bool,
    threshold: usize,
    keep_tokens: usize,
) -> usize {
    let total: usize = turn_tokens.iter().sum();
    if total < threshold {
        return 0;
    }
    let protected = if newest_finished { 1 } else { 2 };
    let target = total.saturating_sub(keep_tokens);
    let mut summarized = 0usize;
    let mut count = 0;
    for tokens in turn_tokens
        .iter()
        .take(turn_tokens.len().saturating_sub(protected))
    {
        if summarized + tokens > target {
            break;
        }
        summarized += tokens;
        count += 1;
    }
    count
}

/// One whole turn of a conversation: a user message and everything the
/// agent did in reply.
#[derive(Debug, Clone, PartialEq)]
pub struct ConversationTurn {
    /// The position the summary's cursor takes when this is the last turn
    /// folded in, as the source counts (a transcript offset, a sequence
    /// number). Increases from turn to turn.
    pub seq: i64,
    pub messages: Vec<Message>,
    /// Stable ids of `messages`, in order, when the source has them. They
    /// become the summary's source handles. Empty otherwise.
    pub message_ids: Vec<String>,
    /// Estimated size. [`ConversationTurn::new`] estimates it; a source
    /// with a better estimate sets it.
    pub tokens: usize,
}

impl ConversationTurn {
    pub fn new(seq: i64, messages: Vec<Message>) -> Self {
        let tokens = estimate_tokens(&messages);
        Self {
            seq,
            messages,
            message_ids: Vec::new(),
            tokens,
        }
    }

    pub fn with_message_ids(mut self, ids: Vec<String>) -> Self {
        self.message_ids = ids;
        self
    }

    pub fn with_tokens(mut self, tokens: usize) -> Self {
        self.tokens = tokens;
        self
    }
}

/// A session's rolling summary of every turn through `cursor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationSummary {
    pub text: String,
    /// The `seq` of the last turn folded in.
    pub cursor: i64,
    /// Messages the summary stands for, across every pass.
    pub message_count: usize,
    /// Id of the last message folded in, when the source has ids.
    pub last_message_id: Option<String>,
    /// Checksum of the history through `cursor`, for sources that cannot
    /// otherwise tell their history changed under the summary
    /// ([`ConversationSource::fingerprint`]). Zero when unused.
    pub fingerprint: u64,
}

/// The unsummarized turns a [`ConversationSource`] returns.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PendingTurns {
    /// Oldest first.
    pub turns: Vec<ConversationTurn>,
    /// The newest turn has ended (see [`turns_to_summarize`]).
    pub newest_finished: bool,
    /// The stored summary no longer describes this history (it was
    /// replaced or edited); `turns` then start from the beginning and the
    /// new summary does not build on the old one.
    pub summary_stale: bool,
}

/// Where a conversation's turns come from.
#[async_trait]
pub trait ConversationSource: Send + Sync {
    /// The turns after `summary` (every turn when `None`), oldest first.
    async fn turns_after(
        &self,
        session_id: &str,
        summary: Option<&ConversationSummary>,
    ) -> Result<PendingTurns, String>;

    /// Checksum of the history through the turn whose `seq` is `cursor`,
    /// stored with the summary. Sources with stable ids keep the default.
    fn fingerprint(&self, _cursor: i64) -> u64 {
        0
    }
}

/// Where summaries live. Implement it over your own database to keep them
/// across restarts; [`InMemorySummaries`] keeps them for the process.
#[async_trait]
pub trait SummaryStore: Send + Sync {
    async fn load(&self, session_id: &str) -> Option<ConversationSummary>;
    async fn save(&self, session_id: &str, summary: ConversationSummary);
}

/// Process-local [`SummaryStore`].
#[derive(Default)]
pub struct InMemorySummaries {
    summaries: std::sync::Mutex<std::collections::HashMap<String, ConversationSummary>>,
}

#[async_trait]
impl SummaryStore for InMemorySummaries {
    async fn load(&self, session_id: &str) -> Option<ConversationSummary> {
        self.summaries.lock().ok()?.get(session_id).cloned()
    }

    async fn save(&self, session_id: &str, summary: ConversationSummary) {
        if let Ok(mut summaries) = self.summaries.lock() {
            summaries.insert(session_id.to_string(), summary);
        }
    }
}

#[async_trait]
impl<S: SummaryStore + ?Sized> SummaryStore for std::sync::Arc<S> {
    async fn load(&self, session_id: &str) -> Option<ConversationSummary> {
        (**self).load(session_id).await
    }

    async fn save(&self, session_id: &str, summary: ConversationSummary) {
        (**self).save(session_id, summary).await
    }
}

/// Writes the summary: folds `turns` into `previous` (the summary of every
/// turn before them) and returns the new cumulative summary.
#[async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(
        &self,
        previous: Option<&str>,
        turns: &[ConversationTurn],
    ) -> Result<String, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_summarized_under_the_threshold() {
        assert_eq!(turns_to_summarize(&[100, 100, 100], true, 301, 0), 0);
        assert_eq!(turns_to_summarize(&[100, 100, 100], true, 300, 0), 2);
    }

    #[test]
    fn at_least_keep_tokens_of_whole_turns_stay_verbatim() {
        let turns = [100, 100, 100, 100];
        assert_eq!(turns_to_summarize(&turns, true, 1, 150), 2);
        assert_eq!(turns_to_summarize(&turns, true, 1, 200), 2);
        assert_eq!(turns_to_summarize(&turns, true, 1, 201), 1);
    }

    #[test]
    fn the_newest_turn_and_while_open_its_predecessor_stay_verbatim() {
        let turns = [100, 100, 100];
        assert_eq!(turns_to_summarize(&turns, true, 1, 0), 2);
        assert_eq!(turns_to_summarize(&turns, false, 1, 0), 1);
        assert_eq!(turns_to_summarize(&[100], true, 1, 0), 0);
    }

    #[test]
    fn thresholds_are_fractions_of_the_window() {
        assert_eq!(compaction_threshold(200_000), 120_000);
        assert_eq!(wait_threshold(200_000), 160_000);
        assert_eq!(verbatim_tokens(120_000), 30_000);
    }
}
