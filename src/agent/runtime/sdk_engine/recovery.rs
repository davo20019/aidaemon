//! Settling a run the daemon left unfinished (a crash or a restart) before the
//! next turn of its session starts.
//!
//! The kernel checkpoints a run's state into the daemon's store. A run whose
//! last checkpoint is `in_flight` stopped mid tool batch: the kernel finishes
//! that batch under its own rules (idempotent calls run again, others are
//! marked indeterminate rather than repeated). A run that is `pending` was
//! waiting for an owner decision that no longer exists in memory. It is
//! declined without a card and ended there, because the owner has moved on
//! and sent a new message: the turn says what was not done instead. The
//! recovered transcript is not reused: the next turn is built
//! from the daemon's own projection, as every turn is, so nothing is
//! duplicated.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock, Mutex};

use aidaemon_core as sdk;
use aidaemon_store_sqlite::SqliteStore;
use tokio_util::sync::CancellationToken;
use tracing::warn;

/// What the owner is told about requests a cut-off run did not carry out.
pub(super) fn not_done_note(not_done: &[String]) -> Option<String> {
    const MAX_CHARS: usize = 300;
    if not_done.is_empty() {
        return None;
    }
    let listed: String = not_done.join("; ").chars().take(MAX_CHARS).collect();
    Some(format!(
        "⚠️ An earlier request was cut off before you answered, so it was not done: {listed}. Ask again if you still want it."
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
pub(super) struct EndIfDropped {
    store: Option<Arc<SqliteStore>>,
    run_id: sdk::RunId,
}

impl EndIfDropped {
    pub(super) fn new(store: Option<Arc<SqliteStore>>, run_id: sdk::RunId) -> Self {
        Self { store, run_id }
    }

    /// The run returned on its own.
    pub(super) fn disarm(mut self) {
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
pub(super) async fn stranded_run(
    store: Option<&SqliteStore>,
    session_id: &str,
) -> Option<sdk::RunState> {
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
pub(super) async fn settle(
    agent: &sdk::Agent,
    state: sdk::RunState,
    cancel: CancellationToken,
    resolver: &dyn sdk::InterruptResolver,
) -> Result<Option<sdk::RunOutcome>, sdk::CoreError> {
    let mut outcome = match state.pending.clone() {
        Some(pending) => {
            let Some(resolution) = resolver.resolve(&state, &pending.interruption).await else {
                return Ok(None);
            };
            // A declined owner decision ends the old run here; going on would
            // spend model calls on a request the owner has moved past.
            let resume_cancel = if matches!(
                pending.interruption.kind,
                sdk::InterruptKind::OwnerApproval(_)
            ) {
                let ended = CancellationToken::new();
                ended.cancel();
                ended
            } else {
                cancel.clone()
            };
            agent
                .resume_with_cancel(state, resolution, resume_cancel)
                .await?
        }
        None => agent.run_resolving(state, cancel.clone(), resolver).await?,
    };
    while let sdk::RunStatus::Interrupted { interruption } = &outcome.status {
        let Some(resolution) = resolver.resolve(&outcome.state, interruption).await else {
            return Ok(None);
        };
        outcome = agent
            .resume_with_cancel(outcome.state, resolution, cancel.clone())
            .await?;
    }
    Ok(Some(outcome))
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

    #[tokio::test]
    async fn a_run_that_returned_is_not_ended_again() {
        let store = Arc::new(SqliteStore::open_in_memory().await.unwrap());
        let state = sdk::RunState::new("aidaemon", "s", "Hello.");
        EndIfDropped::new(Some(store.clone()), state.run_id).disarm();
        tokio::task::yield_now().await;
        assert!(!store.run_ended(state.run_id).await.unwrap());
    }
}
