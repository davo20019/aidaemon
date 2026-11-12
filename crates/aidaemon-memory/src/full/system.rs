//! [`FullMemory`]: the full memory system as an
//! `aidaemon_policies::MemorySystem`.

use std::sync::Arc;

use aidaemon_core::{ChannelVisibility, Hook};
use aidaemon_policies::{MemoryInjection, MemoryParts, MemorySystem, MemoryTurn};
use async_trait::async_trait;

use super::recall::{FullRecall, RecallHost, RecallSource, RecallTurn};
use crate::render::RecallGoal;

/// The host's facts about one turn, given to [`FullMemory`] as the turn's
/// [`HostTurn`](aidaemon_policies::HostTurn). Without them the system works
/// from the [`MemoryTurn`] alone: the `"owner"` scope is the owner's private
/// chat, any other a private group, with no goals and people off.
#[derive(Clone)]
pub struct FullTurn {
    pub recall: RecallTurn,
    pub host: Arc<dyn RecallHost>,
    /// Further run hooks the host adds with this memory (its run-end
    /// writer).
    pub hooks: Vec<Arc<dyn Hook>>,
}

/// The full memory system over the memory stores: per turn, the memory
/// block injected into the request (see [`FullRecall`]).
pub struct FullMemory {
    recall: Arc<RecallSource>,
}

impl FullMemory {
    pub fn new(recall: Arc<RecallSource>) -> Self {
        Self { recall }
    }
}

/// A host with nothing of its own: no goals, people intelligence off.
struct NoHost;

#[async_trait]
impl RecallHost for NoHost {
    async fn active_goals(&self) -> anyhow::Result<Vec<RecallGoal>> {
        Ok(Vec::new())
    }

    async fn people_enabled(&self) -> anyhow::Result<bool> {
        Ok(false)
    }
}

/// What recall knows about a turn given only the [`MemoryTurn`].
fn plain_turn(turn: &MemoryTurn) -> RecallTurn {
    let owner = turn.scope == "owner";
    RecallTurn {
        session_id: turn.session_id.clone(),
        task_id: String::new(),
        user_text: None,
        owner,
        personal: owner,
        visibility: if owner {
            ChannelVisibility::Private
        } else {
            ChannelVisibility::PrivateGroup
        },
        channel_id: None,
        sender_id: None,
        allowed: turn.read,
        depth: usize::from(turn.explicit_writes_only),
    }
}

#[async_trait]
impl MemorySystem for FullMemory {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        let (recall_turn, host, extra): (RecallTurn, Arc<dyn RecallHost>, Vec<Arc<dyn Hook>>) =
            match turn.host.get::<FullTurn>() {
                Some(full) => (full.recall.clone(), full.host.clone(), full.hooks.clone()),
                None => (plain_turn(turn), Arc::new(NoHost), Vec::new()),
            };
        let recall = FullRecall::new(self.recall.clone(), host, recall_turn);
        let mut hooks: Vec<Arc<dyn Hook>> = vec![Arc::new(MemoryInjection::new(recall))];
        hooks.extend(extra);
        MemoryParts {
            hooks,
            tools: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::full::RecallSettings;
    use crate::sqlite::SqliteMemory;

    #[tokio::test]
    async fn an_owner_turn_recalls_its_facts_without_host_facts() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        let store = SqliteMemory::new(pool, Arc::new(crate::HashEmbedder::new(384)));
        crate::stores::FactStore::upsert_fact(
            &store,
            "preference",
            "favorite_color",
            "teal",
            "user",
            None,
            crate::records::FactPrivacy::Global,
        )
        .await
        .unwrap();
        let source = Arc::new(RecallSource::new(
            Arc::new(store),
            RecallSettings {
                max_facts: 10,
                assistant_name: None,
            },
        ));
        let turn = MemoryTurn {
            session_id: "test:synthetic-user".into(),
            scope: "owner".into(),
            read: true,
            write: true,
            explicit_writes_only: false,
            learning: Default::default(),
            host: Default::default(),
        };
        let parts = FullMemory::new(source.clone()).turn(&turn).await;
        assert_eq!(parts.hooks.len(), 1);
        assert!(parts.tools.is_empty());
        let recall = FullRecall::new(source, Arc::new(NoHost), plain_turn(&turn));
        let note = aidaemon_policies::Recall::render(
            &recall,
            "test:synthetic-user",
            "what is my favorite color?",
            5,
        )
        .await
        .unwrap();
        assert!(note.contains("teal"), "{note}");
    }
}
