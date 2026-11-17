//! Turns a process stop cut off are finished after the restart.
//!
//! A chat turn the daemon was answering when it stopped (a deploy, a crash,
//! a restart) never replied: the owner saw "Thinking…" and then nothing.
//! At boot each such turn gets one internal continuation in its chat, under
//! the same serializer and delivery as every background continuation. Its
//! message says, from the run's own record, which calls were running and
//! may or may not have taken effect; the turn start settles the old run
//! without running any of them again (`recovery::settle`).
//!
//! Which turns: top-level runs that never ended and were last checkpointed
//! after the previous boot, each still its session's latest run, in a chat
//! the hub can reach. A turn this pass continued that was cut off again is
//! not continued a second time (a turn that crashes the daemon would loop):
//! the chat is told instead.

use std::collections::HashSet;
use std::sync::{Arc, Weak};

use crate::channels::SessionMap;
use aidaemon_channels::ports::OutboundRouter;
use aidaemon_runtime::engine::recovery;
use aidaemon_runtime::spawn::{ChildAgentRuntime, RunsAnswer};
use aidaemon_store_sqlite::SqliteStore;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// The previous boot's mark and the chats it continued.
const BOOT_KEY: &str = "runtime.resume_cut_off_turns";

#[derive(Default, Serialize, Deserialize)]
struct BootRecord {
    /// The run store's clock when that boot started.
    boot_at: String,
    /// Chats that boot continued; cut off again, they are not continued.
    continued: Vec<String>,
}

pub struct ResumeDeps {
    pub store: SqliteStore,
    pub state: Arc<dyn crate::traits::StateStore>,
    pub event_store: Arc<crate::events::EventStore>,
    pub sessions: SessionMap,
    pub hub: Weak<dyn OutboundRouter>,
    pub agent: Arc<dyn ChildAgentRuntime>,
}

/// Reads the previous boot's record and writes this boot's mark. Called
/// before anything this boot can checkpoint a run, so the next boot's list
/// holds only runs this process started or touched.
pub async fn mark_boot(
    store: &SqliteStore,
    state: &dyn crate::traits::StateStore,
) -> Option<(String, Vec<String>)> {
    let previous = match state.get_setting(BOOT_KEY).await {
        Ok(Some(raw)) => serde_json::from_str::<BootRecord>(&raw).ok(),
        Ok(None) => None,
        Err(error) => {
            warn!(%error, "Could not read the previous boot's mark; no cut-off turn is resumed");
            None
        }
    };
    let boot_at = match store.clock().await {
        Ok(now) => now,
        Err(error) => {
            warn!(%error, "Could not read the run store's clock; no cut-off turn is resumed");
            return None;
        }
    };
    write_record(
        state,
        &BootRecord {
            boot_at,
            continued: Vec::new(),
        },
    )
    .await;
    previous.map(|record| (record.boot_at, record.continued))
}

/// Read-modify-write of this boot's record, one at a time.
async fn update_record(
    state: &dyn crate::traits::StateStore,
    change: impl FnOnce(&mut BootRecord),
) {
    static RECORD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let _held = RECORD.lock().await;
    let Ok(Some(raw)) = state.get_setting(BOOT_KEY).await else {
        return;
    };
    let Ok(mut record) = serde_json::from_str::<BootRecord>(&raw) else {
        return;
    };
    change(&mut record);
    write_record(state, &record).await;
}

async fn write_record(state: &dyn crate::traits::StateStore, record: &BootRecord) {
    let Ok(raw) = serde_json::to_string(record) else {
        return;
    };
    if let Err(error) = state.set_setting(BOOT_KEY, &raw).await {
        warn!(%error, "Could not record this boot's mark");
    }
}

/// Continues each chat turn the last stop cut off (see the module docs).
pub async fn resume_cut_off_turns(deps: ResumeDeps, previous: Option<(String, Vec<String>)>) {
    let Some((since, continued_last_boot)) = previous else {
        return;
    };
    let runs = match deps.store.unfinished_runs_since(&since).await {
        Ok(runs) => runs,
        Err(error) => {
            warn!(%error, "Could not list the turns the last stop cut off");
            return;
        }
    };
    let continued_last_boot: HashSet<String> = continued_last_boot.into_iter().collect();
    let Some(hub) = deps.hub.upgrade() else {
        return;
    };
    let mut to_continue = Vec::new();
    for run in runs {
        let chat = run.session_id.clone();
        // Only chats the hub can answer in: a background or goal session has
        // its own recovery.
        if !deps.sessions.read().await.contains_key(&chat) {
            continue;
        }
        if continued_last_boot.contains(&chat) {
            info!(session_id = %chat, "A continued turn was cut off again; telling the chat");
            let note = "\u{26a0}\u{fe0f} The daemon restarted again while I was finishing your \
                        last request, so I stopped retrying it. Send it again, or say continue, \
                        to pick it up.";
            if let Err(error) = hub.send_text(&chat, note).await {
                warn!(session_id = %chat, %error, "Could not tell the chat its turn was cut off");
            }
            continue;
        }
        to_continue.push(run);
    }
    if to_continue.is_empty() {
        return;
    }
    // Recorded before the turns run: if one takes the daemon down, the next
    // boot does not run it again.
    update_record(&*deps.state, |record| {
        record.continued = to_continue
            .iter()
            .map(|run| run.session_id.clone())
            .collect();
    })
    .await;
    for run in to_continue {
        let chat = run.session_id.clone();
        let parent_task_id = deps
            .event_store
            .get_active_task(&chat)
            .await
            .ok()
            .flatten()
            .and_then(|event| event.task_id);
        info!(session_id = %chat, run_id = %run.run_id, "Continuing a turn the last stop cut off");
        let agent = deps.agent.clone();
        let followup = recovery::resume_note(&run);
        let hub = Arc::downgrade(&hub);
        let state = deps.state.clone();
        // Chats are continued side by side, each on its own task.
        tokio::spawn(async move {
            let answered = agent
                .answer_after_runs(RunsAnswer {
                    hub: Some(hub),
                    chat: chat.clone(),
                    parent_task_id,
                    followup,
                    clock: None,
                })
                .await;
            // The continuation ended on its own: a later cut-off in this
            // chat is a new turn's, and is continued again.
            update_record(&*state, |record| {
                record.continued.retain(|continued| continued != &chat);
            })
            .await;
            match answered {
                Ok(true) => info!(session_id = %chat, "Answered a turn the last stop cut off"),
                Ok(false) => warn!(session_id = %chat, "The continued turn gave no reply"),
                Err(error) => warn!(session_id = %chat, %error, "Continuing a cut-off turn failed"),
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventStore;
    use crate::events::EventStoreForTests as _;
    use crate::state::SqliteStateStore;
    use crate::traits::store_prelude::*;
    use aidaemon_runtime::spawn::{ChildAgentRequest, ChildAgentRun};
    use aidaemon_tools::semantics::AgentRole;
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Mutex as StdMutex;

    #[derive(Default)]
    struct Answers(
        StdMutex<Vec<RunsAnswer>>,
        Arc<aidaemon_runtime::run_state::RunStates>,
    );

    #[async_trait]
    impl ChildAgentRuntime for Answers {
        fn depth(&self) -> usize {
            0
        }
        fn max_depth(&self) -> usize {
            3
        }
        fn role(&self) -> AgentRole {
            AgentRole::Orchestrator
        }
        fn specialist_descriptions(&self) -> Vec<(String, String)> {
            Vec::new()
        }
        async fn run_child(&self, _request: ChildAgentRequest) -> anyhow::Result<ChildAgentRun> {
            anyhow::bail!("not used")
        }
        async fn validate_executor_task_for_spawn(
            &self,
            _task_id: &str,
            _expected_goal_id: Option<&str>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn salvage_executor_task_outcome(
            &self,
            _task_id: &str,
            _timeout_secs: u64,
        ) -> Option<aidaemon_runtime::spawn::SalvagedTaskOutcome> {
            None
        }
        async fn deliver_background_child_result(
            &self,
            _router: Option<&Weak<dyn OutboundRouter>>,
            _parent_session_id: &str,
            _text: &str,
            _trust: aidaemon_core::SourceTrust,
        ) -> anyhow::Result<bool> {
            Ok(false)
        }
        fn helper_runs(&self) -> Arc<aidaemon_runtime::run_state::RunStates> {
            self.1.clone()
        }
        async fn mark_executor_task_timeout(&self, _task_id: &str, _timeout_secs: u64) {}
        async fn answer_after_runs(&self, answer: RunsAnswer) -> anyhow::Result<bool> {
            self.0.lock().unwrap().push(answer);
            Ok(true)
        }
    }

    #[derive(Default)]
    struct Sent(StdMutex<Vec<(String, String)>>);

    #[async_trait]
    impl OutboundRouter for Sent {
        async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
            self.0
                .lock()
                .unwrap()
                .push((session_id.to_string(), text.to_string()));
            Ok(())
        }
        async fn send_text_tracked(&self, _: &str, _: &str) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
        async fn edit_text(&self, _: &str, _: &str, _: &str) -> anyhow::Result<bool> {
            Ok(false)
        }
        async fn send_media(&self, _: &str, _: &crate::types::MediaMessage) -> anyhow::Result<()> {
            Ok(())
        }
        async fn send_media_strict(
            &self,
            _: &str,
            _: &crate::types::MediaMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn request_inline_approval(
            &self,
            _: &str,
            _: &str,
            _: crate::tools::command_risk::RiskLevel,
            _: &[String],
            _: crate::tools::command_risk::PermissionMode,
        ) -> anyhow::Result<crate::types::ApprovalResponse> {
            anyhow::bail!("no approvals here")
        }
        async fn register_session_route(&self, _: &str, _: &str) -> bool {
            false
        }
        async fn unregister_session_route(&self, _: &str, _: &str) {}
    }

    async fn settings() -> Arc<SqliteStateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let path = db_file.path().to_str().unwrap().to_string();
        let embedding = Arc::new(crate::memory::embeddings::EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(&path, 100, None, embedding)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        state
    }

    async fn boot(
        store: &SqliteStore,
        state: &Arc<SqliteStateStore>,
        sessions: &SessionMap,
        agent: &Arc<Answers>,
        hub: &Arc<Sent>,
    ) {
        let dyn_state = state.clone() as Arc<dyn crate::traits::StateStore>;
        let previous = mark_boot(store, &*dyn_state).await;
        let router: Arc<dyn OutboundRouter> = hub.clone();
        resume_cut_off_turns(
            ResumeDeps {
                store: SqliteStore::from_pool(store_pool(store)).await.unwrap(),
                state: dyn_state,
                event_store: Arc::new(EventStore::new(state.pool()).await.unwrap()),
                sessions: sessions.clone(),
                hub: Arc::downgrade(&router),
                agent: agent.clone(),
            },
            previous,
        )
        .await;
        // The continuations run on their own tasks.
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
    }

    fn store_pool(store: &SqliteStore) -> sqlx::SqlitePool {
        store.pool().clone()
    }

    #[tokio::test]
    async fn a_cut_off_chat_turn_is_continued_once_and_then_the_chat_is_told() {
        let store = SqliteStore::open_in_memory().await.unwrap();
        let state = settings().await;
        let chat = "app:synthetic-chat-1";
        let sessions: SessionMap = Arc::new(tokio::sync::RwLock::new(HashMap::from([(
            chat.to_string(),
            "app".to_string(),
        )])));
        let agent = Arc::new(Answers::default());
        let hub = Arc::new(Sent::default());

        // First boot: nothing to compare against, nothing continued.
        boot(&store, &state, &sessions, &agent, &hub).await;
        assert!(agent.0.lock().unwrap().is_empty());

        // A turn in a chat and one in a session no chat routes to, both cut off.
        store
            .upsert_state(&aidaemon_core::RunState::new(
                "aidaemon",
                chat,
                "Draw a logo.",
            ))
            .await
            .unwrap();
        store
            .upsert_state(&aidaemon_core::RunState::new(
                "aidaemon",
                "background:synthetic",
                "Helper work.",
            ))
            .await
            .unwrap();
        boot(&store, &state, &sessions, &agent, &hub).await;
        {
            let answers = agent.0.lock().unwrap();
            assert_eq!(answers.len(), 1, "only the chat is continued");
            assert_eq!(answers[0].chat, chat);
            assert!(answers[0].followup.contains("reply it is still owed"));
        }

        // The continuation was cut off before it ended: the next boot does
        // not run it again, and the chat hears why.
        let mut record: BootRecord =
            serde_json::from_str(&state.get_setting(BOOT_KEY).await.unwrap().unwrap()).unwrap();
        record.continued = vec![chat.to_string()];
        write_record(&*state, &record).await;
        store
            .upsert_state(&aidaemon_core::RunState::new(
                "aidaemon",
                chat,
                "continuation",
            ))
            .await
            .unwrap();
        boot(&store, &state, &sessions, &agent, &hub).await;
        assert_eq!(agent.0.lock().unwrap().len(), 1, "not continued twice");
        let sent = hub.0.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert!(sent[0].1.contains("stopped retrying"), "{}", sent[0].1);
    }
}
