//! [`MemoryRecall`]: plugs [`Memory`] into
//! `aidaemon_policies::MemoryInjection` through the `Recall` trait.

use aidaemon_policies::Recall;
use async_trait::async_trait;

use crate::memory::Memory;

/// An `aidaemon_policies::Recall` backed by [`Memory`]: top facts first,
/// then relevant episodes, as one-line snippets. Errors are logged and
/// yield no snippets, so a memory outage never breaks a run.
///
/// ```
/// use aidaemon_memory::{HashEmbedder, InMemoryMemoryStore, Memory, MemoryRecall};
///
/// let memory = Memory::new(InMemoryMemoryStore::new(), HashEmbedder::default());
/// let hook = aidaemon_policies::MemoryInjection::new(MemoryRecall::new(memory));
/// # let _ = hook;
/// ```
pub struct MemoryRecall {
    memory: Memory,
    max_episodes: usize,
}

impl MemoryRecall {
    pub fn new(memory: Memory) -> Self {
        Self {
            memory,
            max_episodes: 2,
        }
    }

    /// At most this many episode snippets (default 2), within the limit.
    pub fn max_episodes(mut self, n: usize) -> Self {
        self.max_episodes = n;
        self
    }
}

#[async_trait]
impl Recall for MemoryRecall {
    async fn recall(&self, session_id: &str, query: &str, limit: usize) -> Vec<String> {
        if limit == 0 || query.trim().is_empty() {
            return Vec::new();
        }
        let scope = self.memory.scope_for(session_id);
        let mut out: Vec<String> = match self.memory.recall_facts(&scope, query, limit).await {
            Ok(facts) => facts
                .into_iter()
                .map(|f| {
                    format!(
                        "{} {}: {}",
                        f.fact.category,
                        f.fact.key.replace('_', " "),
                        f.fact.value
                    )
                })
                .collect(),
            Err(error) => {
                tracing::warn!(%error, "fact recall failed");
                Vec::new()
            }
        };
        let room = limit.saturating_sub(out.len()).min(self.max_episodes);
        if room > 0 {
            match self.memory.recall_episodes(&scope, query, room).await {
                Ok(episodes) => out.extend(episodes.into_iter().map(|e| {
                    format!(
                        "Earlier conversation ({}): {}",
                        e.episode.created_at.format("%Y-%m-%d"),
                        e.episode.summary
                    )
                })),
                Err(error) => tracing::warn!(%error, "episode recall failed"),
            }
        }
        out
    }
}
