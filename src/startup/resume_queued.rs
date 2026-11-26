//! Messages the daemon accepted to wait for their turn outlive a restart.
//!
//! The task registry publishes its queues on each change
//! ([`TaskRegistry::journal_to`]); they are kept in the state store, and
//! at boot what the last run left waiting goes back on its sessions'
//! queues. Nothing is running then to drain them: once a chat's cut-off
//! turn has been continued (`resume_turns`), its waiting messages run, in
//! order, as their own turns.

use std::sync::Arc;

use aidaemon_channels::ports::OutboundRouter;
use aidaemon_channels::restored::RouterSurface;
use aidaemon_channels::tasks::{QueueSnapshot, TaskRegistry};
use aidaemon_channels::turns::TurnRunner;
use tracing::{info, warn};

use crate::channels::SessionMap;

/// Where the waiting messages are kept.
const QUEUES_KEY: &str = "runtime.queued_messages";

/// A session whose waiting messages came back, and the platform its chat
/// is on (the first message's context).
pub struct Restored {
    session_id: String,
    platform: String,
}

/// Keep `registry`'s queues in `state` from now on, and put back what the
/// last run of the daemon left waiting. Call before any channel starts, so
/// a message that arrives first queues behind the restored ones.
pub async fn keep_queues(
    registry: &TaskRegistry,
    state: Arc<dyn crate::traits::StateStore>,
) -> Vec<Restored> {
    let stored = match state.get_setting(QUEUES_KEY).await {
        Ok(Some(raw)) => serde_json::from_str::<QueueSnapshot>(&raw).unwrap_or_else(|error| {
            warn!(%error, "The waiting messages kept at the last stop could not be read");
            QueueSnapshot::new()
        }),
        Ok(None) => QueueSnapshot::new(),
        Err(error) => {
            warn!(%error, "Could not read the messages left waiting at the last stop");
            QueueSnapshot::new()
        }
    };
    let (journal, mut changes) = tokio::sync::watch::channel(QueueSnapshot::new());
    registry.journal_to(journal);
    tokio::spawn(async move {
        while changes.changed().await.is_ok() {
            let raw = serde_json::to_string(&*changes.borrow_and_update());
            match raw {
                Ok(raw) => {
                    if let Err(error) = state.set_setting(QUEUES_KEY, &raw).await {
                        warn!(%error, "Could not keep the waiting messages");
                    }
                }
                Err(error) => warn!(%error, "Could not write down the waiting messages"),
            }
        }
    });
    let restored: Vec<Restored> = stored
        .iter()
        .filter_map(|(session_id, queue)| {
            queue.first().map(|first| Restored {
                session_id: session_id.clone(),
                platform: first.channel_ctx.platform.clone(),
            })
        })
        .collect();
    if !restored.is_empty() {
        info!(
            sessions = restored.len(),
            "Restored messages the last stop left waiting"
        );
    }
    registry.restore_queues(stored).await;
    restored
}

/// Run each restored session's waiting messages once its cut-off turn (if
/// one was continued, `continuations`) has ended. A chat the hub cannot
/// reach any more keeps nothing waiting.
pub async fn run_restored(
    restored: Vec<Restored>,
    continuations: Vec<(String, tokio::task::JoinHandle<()>)>,
    runner: TurnRunner,
    registry: Arc<TaskRegistry>,
    hub: std::sync::Weak<dyn OutboundRouter>,
    sessions: SessionMap,
) {
    let mut continuations: std::collections::HashMap<_, _> = continuations.into_iter().collect();
    for Restored {
        session_id,
        platform,
    } in restored
    {
        if !sessions.read().await.contains_key(&session_id) {
            warn!(%session_id, "Messages left waiting in a chat no bot reaches; dropped");
            registry.clear_queue(&session_id).await;
            continue;
        }
        let Some(router) = hub.upgrade() else {
            return;
        };
        let continued = continuations.remove(&session_id);
        let runner = runner.clone();
        tokio::spawn(async move {
            if let Some(continued) = continued {
                let _ = continued.await;
            }
            let surface = Arc::new(RouterSurface::new(router, session_id.clone(), &platform));
            runner.run_restored(surface, &session_id).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SqliteStateStore;
    use aidaemon_channels::context::ChannelContext;
    use aidaemon_channels::tasks::QueuedMessage;
    use aidaemon_tools::semantics::UserRole;

    /// A message accepted to wait is in the state store as soon as it
    /// queues, and the next boot puts it back on its session's queue.
    #[tokio::test]
    async fn a_waiting_message_is_kept_and_comes_back_next_boot() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let path = db_file.path().to_str().unwrap().to_string();
        let embedding = Arc::new(crate::memory::embeddings::EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(&path, 100, None, embedding)
                .await
                .unwrap(),
        );
        aidaemon_tools::testing::delete_at_exit(db_file.into_temp_path().keep().unwrap());
        let state: Arc<dyn crate::traits::StateStore> = state;

        let before = TaskRegistry::new(10);
        assert!(keep_queues(&before, state.clone()).await.is_empty());
        let (_task, _) = before.register("app:s1", "running").await;
        let message = QueuedMessage::new(
            "also add tests",
            UserRole::Owner,
            ChannelContext::private("app"),
            Vec::new(),
        );
        before
            .queue_message_if_running("app:s1", message, Some("r1"))
            .await;
        let mut kept = false;
        for _ in 0..200 {
            if let Ok(Some(raw)) = state.get_setting(QUEUES_KEY).await
                && raw.contains("also add tests")
            {
                kept = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert!(kept, "the waiting message reached the state store");

        // The next boot.
        let after = TaskRegistry::new(10);
        let restored = keep_queues(&after, state).await;
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].session_id, "app:s1");
        assert_eq!(restored[0].platform, "app");
        let waiting = after.pop_queued_message("app:s1").await.unwrap();
        assert_eq!(waiting.text, "also add tests");
        assert_eq!(waiting.user_role, UserRole::Owner);
    }
}
