//! [`Compactor`]: runs compaction passes for any [`ConversationSource`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::watch;

use aidaemon_core::{MessageOrigin, Role};

use super::{
    ConversationSource, ConversationSummary, ConversationTurn, Summarizer, SummaryStore,
    compaction_threshold, turns_to_summarize, verbatim_tokens,
};

/// Starts the part of a summary that holds what the user said, unsummarized.
const PINNED_HEADING: &str = "\n\n## What the user said, word for word\n";
/// Longest single message kept in the pinned list.
const MAX_PINNED_MESSAGE_CHARS: usize = 600;
/// Most text the pinned list may hold; the oldest entries go first.
const MAX_PINNED_CHARS: usize = 6_000;

/// A summary's own text, and the pinned entries stored after it.
fn split_pinned(text: &str) -> (Option<&str>, Vec<String>) {
    let Some((summary, list)) = text.split_once(PINNED_HEADING) else {
        return (Some(text), Vec::new());
    };
    let pinned = list
        .lines()
        .filter_map(|line| line.strip_prefix("- "))
        .map(str::to_string)
        .collect();
    (Some(summary), pinned)
}

/// Add what the user typed in `turns`. Selection is by who wrote a message,
/// never by what it says, so no standing instruction depends on a classifier.
fn pin_from_turns(pinned: &mut Vec<String>, turns: &[ConversationTurn]) {
    for message in turns.iter().flat_map(|turn| &turn.messages) {
        let typed = message.role == Role::User
            && matches!(message.origin, None | Some(MessageOrigin::User));
        if !typed {
            continue;
        }
        let line: String = message
            .content
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(MAX_PINNED_MESSAGE_CHARS)
            .collect();
        if !line.is_empty() && !pinned.contains(&line) {
            pinned.push(line);
        }
    }
    trim_pinned(pinned);
}

fn trim_pinned(pinned: &mut Vec<String>) {
    let mut total: usize = pinned.iter().map(String::len).sum();
    let mut drop = 0;
    while total > MAX_PINNED_CHARS && drop < pinned.len() {
        total -= pinned[drop].len();
        drop += 1;
    }
    pinned.drain(..drop);
}

fn with_pinned(mut summary: String, pinned: &[String]) -> String {
    if pinned.is_empty() {
        return summary;
    }
    summary.push_str(PINNED_HEADING);
    for line in pinned {
        summary.push_str("- ");
        summary.push_str(line);
        summary.push('\n');
    }
    summary
}

/// How big the history may grow before a pass compacts it, and how much of
/// it stays verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionLimits {
    /// Compact once the unsummarized turns reach this many tokens.
    pub threshold: usize,
    /// Keep at least this many tokens of recent whole turns verbatim.
    pub keep_tokens: usize,
}

impl CompactionLimits {
    /// The defaults for a model with this context window:
    /// [`compaction_threshold`] and [`verbatim_tokens`].
    pub fn for_window(window_tokens: usize) -> Self {
        let threshold = compaction_threshold(window_tokens);
        Self {
            threshold,
            keep_tokens: verbatim_tokens(threshold),
        }
    }
}

/// The per-session state every [`Compactor`] sharing it agrees on: one pass
/// at a time per session, and at most one background pass running with one
/// queued behind it. Clone it into compactors built per call (an app whose
/// summarizer depends on the current model) so they still coordinate.
#[derive(Clone, Default)]
pub struct CompactionJobs {
    shared: Arc<Shared>,
}

#[derive(Default)]
struct Shared {
    locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    running: Mutex<HashMap<String, Running>>,
}

struct Running {
    /// The pass to run once the current one ends; a newer request replaces
    /// an older one, since it sees at least the same turns.
    next: Option<Queued>,
    done: watch::Receiver<bool>,
}

struct Queued {
    compactor: Compactor,
    source: Arc<dyn ConversationSource>,
    limits: CompactionLimits,
}

impl CompactionJobs {
    fn lock(&self, session_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self
            .shared
            .locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        locks.retain(|_, lock| Arc::strong_count(lock) > 1);
        locks.entry(session_id.to_string()).or_default().clone()
    }

    /// True while a background pass for the session is running or queued.
    pub fn is_running(&self, session_id: &str) -> bool {
        self.shared
            .running
            .lock()
            .is_ok_and(|running| running.contains_key(session_id))
    }

    /// Wait until the session's background passes, including any queued
    /// while waiting, have ended.
    pub async fn wait(&self, session_id: &str) {
        let done = self
            .shared
            .running
            .lock()
            .ok()
            .and_then(|running| running.get(session_id).map(|r| r.done.clone()));
        if let Some(mut done) = done {
            while !*done.borrow() {
                if done.changed().await.is_err() {
                    break;
                }
            }
        }
    }

    /// Wait for every session's background passes.
    pub async fn drain(&self) {
        let sessions: Vec<String> = self
            .shared
            .running
            .lock()
            .map(|running| running.keys().cloned().collect())
            .unwrap_or_default();
        for session in sessions {
            self.wait(&session).await;
        }
    }
}

/// Runs compaction: loads the session's summary, asks the source for the
/// turns after it, decides with [`turns_to_summarize`], folds the chosen
/// turns in with the [`Summarizer`] and stores the result.
#[derive(Clone)]
pub struct Compactor {
    summarizer: Arc<dyn Summarizer>,
    store: Arc<dyn SummaryStore>,
    jobs: CompactionJobs,
    pin_user_messages: bool,
}

impl Compactor {
    pub fn new(summarizer: impl Summarizer + 'static, store: impl SummaryStore + 'static) -> Self {
        Self::from_arcs(Arc::new(summarizer), Arc::new(store))
    }

    pub fn from_arcs(summarizer: Arc<dyn Summarizer>, store: Arc<dyn SummaryStore>) -> Self {
        Self {
            summarizer,
            store,
            jobs: CompactionJobs::default(),
            pin_user_messages: false,
        }
    }

    /// Keep what the user typed, word for word, after each summary. A
    /// summary loses standing instructions ("never delete those") more often
    /// than it loses the task; the list is bounded, oldest out first.
    pub fn pin_user_messages(mut self) -> Self {
        self.pin_user_messages = true;
        self
    }

    /// Coordinate with other compactors through shared [`CompactionJobs`].
    pub fn with_jobs(mut self, jobs: CompactionJobs) -> Self {
        self.jobs = jobs;
        self
    }

    pub fn jobs(&self) -> &CompactionJobs {
        &self.jobs
    }

    pub fn store(&self) -> &Arc<dyn SummaryStore> {
        &self.store
    }

    /// One pass, now. Returns the new summary, or `None` when the history
    /// is under the threshold or has nothing old enough to fold in.
    pub async fn compact(
        &self,
        session_id: &str,
        source: &dyn ConversationSource,
        limits: CompactionLimits,
    ) -> Result<Option<ConversationSummary>, String> {
        let lock = self.jobs.lock(session_id);
        let _guard = lock.lock().await;
        let stored = self.store.load(session_id).await;
        let pending = source.turns_after(session_id, stored.as_ref()).await?;
        let previous = stored.filter(|_| !pending.summary_stale);
        let sizes: Vec<usize> = pending.turns.iter().map(|turn| turn.tokens).collect();
        let count = turns_to_summarize(
            &sizes,
            pending.newest_finished,
            limits.threshold,
            limits.keep_tokens,
        );
        if count == 0 {
            return Ok(None);
        }
        let folded = &pending.turns[..count];
        let (earlier_summary, mut pinned) = match previous.as_ref() {
            Some(summary) => split_pinned(&summary.text),
            None => (None, Vec::new()),
        };
        let text = self.summarizer.summarize(earlier_summary, folded).await?;
        let text = if self.pin_user_messages {
            pin_from_turns(&mut pinned, folded);
            with_pinned(text, &pinned)
        } else {
            text
        };
        let last = &folded[count - 1];
        let summary = ConversationSummary {
            text,
            cursor: last.seq,
            message_count: previous.as_ref().map_or(0, |s| s.message_count)
                + folded.iter().map(|t| t.messages.len()).sum::<usize>(),
            last_message_id: last.message_ids.last().cloned(),
            fingerprint: source.fingerprint(last.seq),
        };
        self.store.save(session_id, summary.clone()).await;
        Ok(Some(summary))
    }

    /// A pass in the background. While one runs for the session, the
    /// request is queued (replacing any queued before it) and runs next.
    pub fn spawn(
        &self,
        session_id: &str,
        source: Arc<dyn ConversationSource>,
        limits: CompactionLimits,
    ) {
        let queued = Queued {
            compactor: self.clone(),
            source,
            limits,
        };
        let jobs = self.jobs.clone();
        let done_tx = {
            let mut running = jobs
                .shared
                .running
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(slot) = running.get_mut(session_id) {
                slot.next = Some(queued);
                return;
            }
            let (tx, rx) = watch::channel(false);
            running.insert(
                session_id.to_string(),
                Running {
                    next: None,
                    done: rx,
                },
            );
            tx
        };
        let session_id = session_id.to_string();
        tokio::spawn(async move {
            let mut job = queued;
            loop {
                match job
                    .compactor
                    .compact(&session_id, job.source.as_ref(), job.limits)
                    .await
                {
                    Ok(Some(summary)) => tracing::info!(
                        session_id,
                        cursor = summary.cursor,
                        messages = summary.message_count,
                        "conversation compacted"
                    ),
                    Ok(None) => {}
                    Err(error) => {
                        tracing::warn!(session_id, %error, "conversation compaction failed")
                    }
                }
                let next = {
                    let mut running = jobs
                        .shared
                        .running
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    match running
                        .get_mut(&session_id)
                        .and_then(|slot| slot.next.take())
                    {
                        Some(next) => Some(next),
                        None => {
                            running.remove(&session_id);
                            None
                        }
                    }
                };
                match next {
                    Some(next) => job = next,
                    None => break,
                }
            }
            let _ = done_tx.send(true);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{ConversationTurn, InMemorySummaries, PendingTurns};
    use aidaemon_core::Message;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Turns numbered 1..=n of `tokens` each, after the summary's cursor.
    struct Numbered {
        turns: usize,
        tokens: usize,
        newest_finished: bool,
    }

    #[async_trait]
    impl ConversationSource for Numbered {
        async fn turns_after(
            &self,
            _session_id: &str,
            summary: Option<&ConversationSummary>,
        ) -> Result<PendingTurns, String> {
            let after = summary.map_or(0, |s| s.cursor);
            Ok(PendingTurns {
                turns: (after + 1..=self.turns as i64)
                    .map(|seq| {
                        ConversationTurn::new(seq, vec![Message::user(format!("turn {seq}"))])
                            .with_message_ids(vec![format!("m{seq}")])
                            .with_tokens(self.tokens)
                    })
                    .collect(),
                newest_finished: self.newest_finished,
                summary_stale: false,
            })
        }
    }

    #[derive(Clone, Default)]
    struct Folding {
        calls: Arc<AtomicUsize>,
        delay: Duration,
    }

    #[async_trait]
    impl Summarizer for Folding {
        async fn summarize(
            &self,
            previous: Option<&str>,
            turns: &[ConversationTurn],
        ) -> Result<String, String> {
            tokio::time::sleep(self.delay).await;
            self.calls.fetch_add(1, Ordering::SeqCst);
            let seqs: Vec<String> = turns.iter().map(|t| t.seq.to_string()).collect();
            Ok(format!("{}+{}", previous.unwrap_or(""), seqs.join(",")))
        }
    }

    fn limits(threshold: usize, keep_tokens: usize) -> CompactionLimits {
        CompactionLimits {
            threshold,
            keep_tokens,
        }
    }

    #[tokio::test]
    async fn a_pass_folds_old_turns_and_the_next_builds_on_it() {
        let compactor = Compactor::new(Folding::default(), InMemorySummaries::default());
        let source = Numbered {
            turns: 6,
            tokens: 100,
            newest_finished: true,
        };
        let first = compactor
            .compact("s", &source, limits(500, 200))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first.text, "+1,2,3,4");
        assert_eq!(first.cursor, 4);
        assert_eq!(first.message_count, 4);
        assert_eq!(first.last_message_id.as_deref(), Some("m4"));
        // Two turns left: under the threshold.
        assert!(
            compactor
                .compact("s", &source, limits(500, 200))
                .await
                .unwrap()
                .is_none()
        );
        let grown = Numbered {
            turns: 12,
            ..source
        };
        let second = compactor
            .compact("s", &grown, limits(500, 200))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second.text, "+1,2,3,4+5,6,7,8,9,10");
        assert_eq!(second.message_count, 10);
    }

    /// Forgets everything it is given: the worst a summary can do.
    #[derive(Clone, Default)]
    struct Forgetful {
        previous_seen: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl Summarizer for Forgetful {
        async fn summarize(
            &self,
            previous: Option<&str>,
            _turns: &[ConversationTurn],
        ) -> Result<String, String> {
            self.previous_seen
                .lock()
                .unwrap()
                .push(previous.unwrap_or_default().to_string());
            Ok("a summary that lost the details".into())
        }
    }

    #[tokio::test]
    async fn what_the_user_said_survives_every_pass_word_for_word() {
        let summarizer = Forgetful::default();
        let compactor =
            Compactor::new(summarizer.clone(), InMemorySummaries::default()).pin_user_messages();
        let source = Numbered {
            turns: 6,
            tokens: 100,
            newest_finished: true,
        };
        let first = compactor
            .compact("s", &source, limits(500, 200))
            .await
            .unwrap()
            .unwrap();
        assert!(first.text.contains("- turn 1\n"));
        assert!(first.text.starts_with("a summary that lost the details"));

        let grown = Numbered {
            turns: 12,
            ..source
        };
        let second = compactor
            .compact("s", &grown, limits(500, 200))
            .await
            .unwrap()
            .unwrap();
        assert!(second.text.contains("- turn 1\n"), "{}", second.text);
        assert!(second.text.contains("- turn 10"));
        assert_eq!(second.text.matches("said, word for word").count(), 1);
        // The summarizer works from the summary alone, not the pinned list.
        assert!(!summarizer.previous_seen.lock().unwrap()[1].contains("word for word"));
    }

    #[test]
    fn the_pinned_list_keeps_the_newest_when_it_is_full() {
        let mut pinned: Vec<String> = (0..500).map(|i| format!("instruction {i}")).collect();
        trim_pinned(&mut pinned);
        assert!(pinned.iter().map(String::len).sum::<usize>() <= MAX_PINNED_CHARS);
        assert_eq!(pinned.last().unwrap(), "instruction 499");
        assert!(!pinned.contains(&"instruction 0".to_string()));
    }

    #[tokio::test]
    async fn the_limits_for_a_window_follow_the_policy() {
        assert_eq!(
            CompactionLimits::for_window(200_000),
            limits(120_000, 30_000)
        );
    }

    /// Passes requested while one runs collapse into one queued pass, and
    /// waiting covers the queued pass too.
    #[tokio::test]
    async fn background_passes_coalesce_and_wait_covers_them() {
        let summarizer = Folding {
            delay: Duration::from_millis(30),
            ..Folding::default()
        };
        let calls = summarizer.calls.clone();
        let compactor = Compactor::new(summarizer, InMemorySummaries::default());
        for turns in [6, 8, 10, 12] {
            compactor.spawn(
                "s",
                Arc::new(Numbered {
                    turns,
                    tokens: 100,
                    newest_finished: true,
                }),
                limits(300, 100),
            );
        }
        assert!(compactor.jobs().is_running("s"));
        compactor.jobs().wait("s").await;
        assert!(!compactor.jobs().is_running("s"));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "the first pass and the latest queued one"
        );
        let summary = compactor.store().load("s").await.unwrap();
        assert_eq!(summary.cursor, 11);
    }

    #[tokio::test]
    async fn compactors_sharing_jobs_coordinate() {
        let jobs = CompactionJobs::default();
        let a = Compactor::new(
            Folding {
                delay: Duration::from_millis(30),
                ..Folding::default()
            },
            InMemorySummaries::default(),
        )
        .with_jobs(jobs.clone());
        a.spawn(
            "s",
            Arc::new(Numbered {
                turns: 6,
                tokens: 100,
                newest_finished: true,
            }),
            limits(300, 100),
        );
        assert!(jobs.is_running("s"));
        jobs.drain().await;
        assert!(!jobs.is_running("s"));
    }

    #[tokio::test]
    async fn a_failed_summary_stores_nothing() {
        struct Failing;
        #[async_trait]
        impl Summarizer for Failing {
            async fn summarize(
                &self,
                _: Option<&str>,
                _: &[ConversationTurn],
            ) -> Result<String, String> {
                Err("provider down".into())
            }
        }
        let compactor = Compactor::new(Failing, InMemorySummaries::default());
        let source = Numbered {
            turns: 6,
            tokens: 100,
            newest_finished: true,
        };
        assert!(compactor.compact("s", &source, limits(1, 0)).await.is_err());
        assert!(compactor.store().load("s").await.is_none());
    }
}
