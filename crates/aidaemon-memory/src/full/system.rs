//! [`FullMemory`]: the full memory system as an
//! `aidaemon_policies::MemorySystem`.

use std::sync::Arc;

use aidaemon_core::{ChannelVisibility, Hook, Tool};
use aidaemon_policies::{MemoryInjection, MemoryParts, MemorySystem, MemoryTurn};
use aidaemon_tools::services::SettingsStore;
use async_trait::async_trait;

use super::recall::{FullRecall, RecallHost, RecallSettings, RecallSource, RecallTurn};
use super::tools;
use crate::render::RecallGoal;
use crate::stores::MemoryStores;

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
/// block injected into the request (see [`FullRecall`]) and, on a turn that
/// may read or write memory, its tools.
pub struct FullMemory {
    recall: Arc<RecallSource>,
    tools: Vec<Arc<dyn Tool>>,
}

impl FullMemory {
    /// Recall only. A host that offers the memory tools through its own
    /// dispatcher (see [`tools::outcome_tools`]) starts here.
    pub fn new(recall: Arc<RecallSource>) -> Self {
        Self {
            recall,
            tools: Vec::new(),
        }
    }

    /// The whole system over `store`: recall with `settings`, and the four
    /// memory tools ([`tools::kernel_tools`]). `host_settings` holds the
    /// `people_enabled` switch.
    pub fn over(
        store: Arc<dyn MemoryStores>,
        settings: RecallSettings,
        host_settings: Arc<dyn SettingsStore>,
    ) -> Self {
        Self::new(Arc::new(RecallSource::new(store.clone(), settings)))
            .with_tools(tools::kernel_tools(store, host_settings))
    }

    /// The whole system in a SQLite file (created and migrated if needed),
    /// or in a private in-memory database when `path` is `None`. Embeddings
    /// come from `FastEmbedder` with the `fastembed` feature, else from
    /// [`HashEmbedder`](crate::HashEmbedder); people intelligence's switch is
    /// kept in memory ([`tools::LocalSettings`]).
    pub async fn open(path: Option<&std::path::Path>) -> anyhow::Result<Self> {
        use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
        let pool = match path {
            Some(path) => {
                let options = SqliteConnectOptions::new()
                    .filename(path)
                    .create_if_missing(true)
                    .journal_mode(SqliteJournalMode::Wal)
                    .busy_timeout(std::time::Duration::from_secs(10));
                SqlitePoolOptions::new()
                    .max_connections(4)
                    .connect_with(options)
                    .await?
            }
            None => {
                // One connection that never closes: the database lives in it.
                SqlitePoolOptions::new()
                    .max_connections(1)
                    .min_connections(1)
                    .idle_timeout(None)
                    .max_lifetime(None)
                    .connect("sqlite::memory:")
                    .await?
            }
        };
        crate::sqlite::migrate(&pool).await?;
        #[cfg(feature = "fastembed")]
        let embedder: Arc<dyn crate::Embedder> = Arc::new(crate::FastEmbedder::new());
        #[cfg(not(feature = "fastembed"))]
        let embedder: Arc<dyn crate::Embedder> = Arc::new(crate::HashEmbedder::new(384));
        let store = Arc::new(crate::sqlite::SqliteMemory::new(pool, embedder));
        Ok(Self::over(
            store,
            RecallSettings {
                max_facts: DEFAULT_MAX_FACTS,
                assistant_name: None,
            },
            Arc::new(tools::LocalSettings::default()),
        ))
    }

    /// Offer `tools` on every turn that may read or write memory.
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }
}

/// How many facts recall renders at most, by default (the daemon's
/// default too).
pub const DEFAULT_MAX_FACTS: usize = 20;

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
            tools: if turn.read || turn.write {
                self.tools.clone()
            } else {
                Vec::new()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[tokio::test]
    async fn the_whole_system_offers_its_tools_on_a_memory_turn() {
        let store = crate::full::tools::test_support::memory().await;
        let system = FullMemory::over(
            store,
            RecallSettings {
                max_facts: 10,
                assistant_name: None,
            },
            Arc::new(crate::full::tools::test_support::Settings::default()),
        );
        let mut turn = MemoryTurn {
            session_id: "test:synthetic-user".into(),
            scope: "owner".into(),
            read: true,
            write: true,
            explicit_writes_only: false,
            learning: Default::default(),
            host: Default::default(),
        };
        let names = |parts: &MemoryParts| -> Vec<String> {
            parts.tools.iter().map(|tool| tool.spec().name).collect()
        };
        let parts = system.turn(&turn).await;
        let mut offered = names(&parts);
        offered.sort();
        let mut expected: Vec<String> = crate::full::tools::TOOL_NAMES
            .iter()
            .map(|name| name.to_string())
            .collect();
        expected.sort();
        assert_eq!(offered, expected);

        turn.read = false;
        turn.write = false;
        assert!(system.turn(&turn).await.tools.is_empty());
    }
}
