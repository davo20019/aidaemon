//! Settling a run the host left unfinished (a crash or a restart) before the
//! next turn of its session starts.
//!
//! The kernel checkpoints a run's state into the host's store. A run whose
//! last checkpoint is `in_flight` stopped mid tool batch: the kernel finishes
//! that batch under its own rules (idempotent calls run again, others are
//! marked indeterminate rather than repeated). A run that is `pending` was
//! waiting for an owner decision that no longer exists in memory. It is
//! declined without a card and ended there, because the owner has moved on
//! and sent a new message: the turn says what was not done instead. The
//! recovered transcript is not reused: the next turn is built
//! from the host's own projection, as every turn is, so nothing is
//! duplicated.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock, Mutex};

use aidaemon_core as sdk;
use aidaemon_core::InterruptResolver as _;
use aidaemon_store_sqlite::SqliteStore;
use tokio_util::sync::CancellationToken;
use tracing::warn;

/// What the owner is told about requests a cut-off run did not carry out:
/// each card's question (its first sentence), and that the owner declined
/// nothing.
pub fn not_done_note(not_done: &[String]) -> Option<String> {
    const MAX_ITEM_CHARS: usize = 120;
    const MAX_ITEMS: usize = 3;
    if not_done.is_empty() {
        return None;
    }
    let question = |summary: &String| {
        let summary = summary.trim();
        // A sentence ends at `?`, a line break, or a `.` before a space (not
        // the dots of a host name).
        let mut chars = summary.char_indices().peekable();
        let mut end = summary.len();
        while let Some((at, c)) = chars.next() {
            let next_is_space = chars.peek().is_none_or(|(_, next)| next.is_whitespace());
            if c == '?' || c == '\n' || (c == '.' && next_is_space) {
                end = at + c.len_utf8();
                break;
            }
        }
        let first = summary[..end].trim();
        if first.chars().count() > MAX_ITEM_CHARS {
            let cut: String = first.chars().take(MAX_ITEM_CHARS).collect();
            format!("{}\u{2026}", cut.trim_end())
        } else {
            first.to_string()
        }
    };
    let mut listed: Vec<String> = not_done.iter().take(MAX_ITEMS).map(question).collect();
    if not_done.len() > MAX_ITEMS {
        listed.push(format!("and {} more", not_done.len() - MAX_ITEMS));
    }
    let listed = listed.join("; ");
    // A question keeps its own mark; only an unpunctuated list gets a period.
    let stop = if listed.ends_with(['?', '.', '!', '\u{2026}']) {
        ""
    } else {
        "."
    };
    Some(format!(
        "\u{26a0}\u{fe0f} The daemon restarted while an approval was waiting for you, so it was \
         not done (you declined nothing): {listed}{stop} Ask again if you still want it."
    ))
}

/// Runs whose turn was dropped (a cancel, the inactivity watchdog) and whose
/// end is still being written to the journal.
static ENDING: LazyLock<Mutex<HashSet<sdk::RunId>>> = LazyLock::new(Mutex::default);

fn is_ending(run_id: sdk::RunId) -> bool {
    ENDING
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .contains(&run_id)
}

/// Ends the run in the journal if its turn is dropped before the run returns.
/// A dropped turn was stopped on purpose, so the next message must not resume
/// it. A process crash runs no destructor and is still recovered.
pub struct EndIfDropped {
    store: Option<Arc<SqliteStore>>,
    run_id: sdk::RunId,
}

impl EndIfDropped {
    pub fn new(store: Option<Arc<SqliteStore>>, run_id: sdk::RunId) -> Self {
        Self { store, run_id }
    }

    /// The run returned on its own.
    pub fn disarm(mut self) {
        self.store = None;
    }
}

impl Drop for EndIfDropped {
    fn drop(&mut self) {
        let Some(store) = self.store.take() else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let run_id = self.run_id;
        ENDING
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(run_id);
        runtime.spawn(async move {
            let ended = sdk::Event::Finished {
                status: sdk::RunStatus::Cancelled,
                usage: sdk::Usage::default(),
                turns: 0,
            };
            if let Err(error) = store.append(run_id, &ended).await {
                warn!(%run_id, %error, "Could not record the end of a stopped run");
            }
            ENDING
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&run_id);
        });
    }
}

/// The session's most recent run, when it is unfinished.
pub async fn stranded_run(store: Option<&SqliteStore>, session_id: &str) -> Option<sdk::RunState> {
    let store = store?;
    let state = match store.latest_state_for_session(session_id).await {
        Ok(Some(state)) if state.pending.is_some() || state.in_flight.is_some() => state,
        Ok(_) => return None,
        Err(error) => {
            warn!(session_id, %error, "Could not read the session's last run; nothing is recovered");
            return None;
        }
    };
    if is_ending(state.run_id) {
        return None;
    }
    // The kernel clears `in_flight` when a run ends; a row written before
    // that still has it, and restarting it would undo a cancel.
    match store.run_ended(state.run_id).await {
        Ok(false) => Some(state),
        Ok(true) => None,
        Err(error) => {
            warn!(session_id, %error, "Could not read how the session's last run ended; leaving it");
            None
        }
    }
}

/// Run `state` to its end. `None` means the owner gave no answer, so the run
/// stays unfinished for a later turn.
///
/// The recovered run shares no decline set (`sdk::SharedDeclines` is a
/// runtime handle, never stored): the parent and children it shared one
/// with ended with the process, and the turn about to start is a new run
/// whose declines start empty. Its own `declined` record still binds it
/// while it settles, and still keeps it from ending `Succeeded`.
pub async fn settle(
    agent: &sdk::Agent,
    state: sdk::RunState,
    cancel: CancellationToken,
    resolver: &dyn sdk::InterruptResolver,
) -> Result<Option<sdk::RunOutcome>, sdk::CoreError> {
    // A declined owner decision ends the old run: going on would spend model
    // calls on a request the owner has moved past, and decline, with nobody
    // asked, every later card it reached. The first one cancels the run, so
    // it settles only what it had started.
    let run_cancel = cancel.child_token();
    let resolver = EndsAtOwnerQuestion {
        inner: resolver,
        run: run_cancel.clone(),
    };
    let outcome = match state.pending.clone() {
        Some(pending) => {
            let Some(resolution) = resolver.resolve(&state, &pending.interruption).await else {
                return Ok(None);
            };
            agent
                .resume_resolving(state, resolution, run_cancel, &resolver)
                .await?
        }
        None => agent.run_resolving(state, run_cancel, &resolver).await?,
    };
    if matches!(outcome.status, sdk::RunStatus::Interrupted { .. }) {
        return Ok(None);
    }
    Ok(Some(outcome))
}

/// Settles a recovered run's interruptions with `inner`, and ends the run at
/// the first owner question: nobody is there to answer it.
struct EndsAtOwnerQuestion<'a> {
    inner: &'a dyn sdk::InterruptResolver,
    run: CancellationToken,
}

#[async_trait::async_trait]
impl sdk::InterruptResolver for EndsAtOwnerQuestion<'_> {
    async fn resolve(
        &self,
        state: &sdk::RunState,
        interruption: &sdk::Interruption,
    ) -> Option<sdk::Resolution> {
        let resolution = self.inner.resolve(state, interruption).await;
        if matches!(interruption.kind, sdk::InterruptKind::OwnerApproval(_)) {
            self.run.cancel();
        }
        resolution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_turn_that_was_stopped_is_not_resumed_by_the_next_message() {
        let store = Arc::new(SqliteStore::open_in_memory().await.unwrap());
        let session = "telegram:synthetic-user-1";
        let mut state = sdk::RunState::new("aidaemon", session, "A long request.");
        state.in_flight = Some(Default::default());
        store.upsert_state(&state).await.unwrap();
        assert!(stranded_run(Some(&store), session).await.is_some());

        drop(EndIfDropped::new(Some(store.clone()), state.run_id));
        assert!(
            stranded_run(Some(&store), session).await.is_none(),
            "stopped at once, before the journal write lands"
        );

        while is_ending(state.run_id) {
            tokio::task::yield_now().await;
        }
        assert!(store.run_ended(state.run_id).await.unwrap());
        assert!(stranded_run(Some(&store), session).await.is_none());
    }

    #[test]
    fn the_not_done_note_names_each_question_briefly() {
        let note = not_done_note(&[
            "Let the browser open demo.example.test, a site on this computer? Approve only if you asked for it.".to_string(),
            format!("Run terminal {}", "x".repeat(400)),
        ])
        .unwrap();
        assert!(note.contains("Let the browser open demo.example.test, a site on this computer?"));
        assert!(!note.contains("Approve only"), "{note}");
        assert!(note.contains("you declined nothing"), "{note}");
        assert!(!note.contains("?."), "{note}");
        assert!(note.chars().count() < 400, "{note}");
        assert!(not_done_note(&[]).is_none());
    }

    struct Declines;

    #[async_trait::async_trait]
    impl sdk::InterruptResolver for Declines {
        async fn resolve(
            &self,
            _state: &sdk::RunState,
            _interruption: &sdk::Interruption,
        ) -> Option<sdk::Resolution> {
            Some(sdk::Resolution::Deny {
                reason: "cut off".to_string(),
            })
        }
    }

    fn interruption(kind: sdk::InterruptKind) -> sdk::Interruption {
        sdk::Interruption {
            call: sdk::ToolCall::new("c1", "browser", serde_json::json!({})),
            reason: "r".to_string(),
            kind,
            digest: None,
        }
    }

    /// A recovered run ends at the first owner question it reaches: nobody
    /// is there to answer, so it must not go on declining later cards.
    #[tokio::test]
    async fn a_recovered_run_ends_at_its_first_owner_question() {
        let state = sdk::RunState::new("aidaemon", "s", "Earlier request.");
        let resolver = EndsAtOwnerQuestion {
            inner: &Declines,
            run: CancellationToken::new(),
        };
        resolver
            .resolve(&state, &interruption(sdk::InterruptKind::Untrusted))
            .await;
        assert!(
            !resolver.run.is_cancelled(),
            "a verifier check is not the owner's"
        );
        let owner = sdk::InterruptKind::OwnerApproval(sdk::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: "Open a site".to_string(),
            subject: "browser".to_string(),
            risk: sdk::ApprovalRisk::High,
            warnings: Vec::new(),
        });
        let answer = resolver.resolve(&state, &interruption(owner)).await;
        assert!(matches!(answer, Some(sdk::Resolution::Deny { .. })));
        assert!(resolver.run.is_cancelled());
    }

    #[tokio::test]
    async fn a_run_that_returned_is_not_ended_again() {
        let store = Arc::new(SqliteStore::open_in_memory().await.unwrap());
        let state = sdk::RunState::new("aidaemon", "s", "Hello.");
        EndIfDropped::new(Some(store.clone()), state.run_id).disarm();
        tokio::task::yield_now().await;
        assert!(!store.run_ended(state.run_id).await.unwrap());
    }
}
