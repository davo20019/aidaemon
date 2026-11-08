//! Settling a run the daemon left unfinished (a crash or a restart) before the
//! next turn of its session starts.
//!
//! The kernel checkpoints a run's state into the daemon's store. A run whose
//! last checkpoint is `in_flight` stopped mid tool batch: the kernel finishes
//! that batch under its own rules (idempotent calls run again, others are
//! marked indeterminate rather than repeated). A run that is `pending` was
//! waiting for an owner decision that no longer exists in memory, so it is
//! settled through the same resolver a live turn uses (a fresh card, or a
//! denial). The recovered transcript is not reused: the next turn is built
//! from the daemon's own projection, as every turn is, so nothing is
//! duplicated.

use aidaemon_core as sdk;
use aidaemon_store_sqlite::SqliteStore;
use tokio_util::sync::CancellationToken;
use tracing::warn;

/// The session's most recent run, when it is unfinished.
pub(super) async fn stranded_run(
    store: Option<&SqliteStore>,
    session_id: &str,
) -> Option<sdk::RunState> {
    let store = store?;
    match store.latest_state_for_session(session_id).await {
        Ok(Some(state)) if state.pending.is_some() => Some(state),
        // The kernel now clears `in_flight` when a run ends; a row written
        // before that still has it, and restarting it would undo a cancel.
        Ok(Some(state)) if state.in_flight.is_some() => match store.run_ended(state.run_id).await {
            Ok(false) => Some(state),
            Ok(true) => None,
            Err(error) => {
                warn!(session_id, %error, "Could not read how the session's last run ended; leaving it alone");
                None
            }
        },
        Ok(_) => None,
        Err(error) => {
            warn!(session_id, %error, "Could not read the session's last run; nothing is recovered");
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
            agent
                .resume_with_cancel(state, resolution, cancel.clone())
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
