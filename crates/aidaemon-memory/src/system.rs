//! [`ScopedMemory`]: this crate as a whole `aidaemon_policies::MemorySystem`.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use aidaemon_core::{FnService, Schedule, Service, ServiceError};
use aidaemon_policies::{MemoryInjection, MemoryParts, MemorySystem, MemoryTurn};
use async_trait::async_trait;

use crate::memory::{Memory, MemoryBuilder};
use crate::recall::MemoryRecall;
use crate::tools::{RecallTool, RememberTool};
use crate::writer::MemoryWriter;

/// [`Memory`] as a pluggable memory system: per turn, injected recall and
/// the `recall` tool when the turn may read, the `remember` tool when it may
/// write, and run-end fact extraction unless writes are explicit only.
///
/// Each session reads and writes the scope its latest turn was given, so a
/// host can map many sessions to one memory (the owner's direct messages)
/// and keep others apart (each group conversation).
///
/// ```
/// use aidaemon_memory::{HashEmbedder, InMemoryMemoryStore, Memory, ScopedMemory};
///
/// let builder = Memory::builder(
///     std::sync::Arc::new(InMemoryMemoryStore::new()),
///     std::sync::Arc::new(HashEmbedder::default()),
/// );
/// let system = ScopedMemory::new(builder);
/// # let _ = system;
/// ```
#[derive(Clone)]
pub struct ScopedMemory {
    memory: Memory,
    /// Each session's scope, set when its turn starts.
    scopes: Arc<RwLock<SessionScopes>>,
}

/// How often recall decay and retention run.
const DECAY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(24 * 3600);

/// Sessions whose scope is kept. Run-end extraction resolves the scope after
/// the turn, so an entry must outlive its turn; the least recently started
/// session is evicted past this bound, far beyond any number of turns in
/// flight at once.
const MAX_SESSION_SCOPES: usize = 1024;

/// Session scopes, bounded by [`MAX_SESSION_SCOPES`].
#[derive(Default)]
struct SessionScopes {
    /// Session id → (scope, turn-start sequence).
    entries: HashMap<String, (String, u64)>,
    next: u64,
}

impl SessionScopes {
    fn get(&self, session_id: &str) -> Option<&str> {
        self.entries
            .get(session_id)
            .map(|(scope, _)| scope.as_str())
    }

    fn insert(&mut self, session_id: String, scope: String) {
        self.next += 1;
        self.entries.insert(session_id, (scope, self.next));
        if self.entries.len() > MAX_SESSION_SCOPES {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, (_, seq))| *seq)
                .map(|(session, _)| session.clone());
            if let Some(oldest) = oldest {
                self.entries.remove(&oldest);
            }
        }
    }
}

/// The scope of a session no turn has scoped yet.
fn session_scope(session_id: &str) -> String {
    format!("session:{session_id}")
}

impl ScopedMemory {
    /// Builds the memory from `builder`, with a scope resolver that follows
    /// the scope each session's turn was given. Any resolver already set on
    /// `builder` is replaced.
    pub fn new(builder: MemoryBuilder) -> Self {
        let scopes: Arc<RwLock<SessionScopes>> = Arc::default();
        let resolve = scopes.clone();
        let memory = builder
            .scope_resolver(move |session_id| {
                resolve
                    .read()
                    .ok()
                    .and_then(|scopes| scopes.get(session_id).map(str::to_string))
                    .unwrap_or_else(|| session_scope(session_id))
            })
            .build();
        Self { memory, scopes }
    }

    /// The memory behind this system.
    pub fn memory(&self) -> &Memory {
        &self.memory
    }
}

#[async_trait]
impl MemorySystem for ScopedMemory {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        if let Ok(mut scopes) = self.scopes.write() {
            scopes.insert(turn.session_id.clone(), turn.scope.clone());
        }
        let mut parts = MemoryParts::default();
        if turn.read {
            parts
                .hooks
                .push(Arc::new(MemoryInjection::new(MemoryRecall::new(
                    self.memory.clone(),
                ))));
            parts
                .tools
                .push(Arc::new(RecallTool::new(self.memory.clone())));
        }
        if turn.write {
            parts
                .tools
                .push(Arc::new(RememberTool::new(self.memory.clone())));
            if !turn.explicit_writes_only {
                parts
                    .hooks
                    .push(Arc::new(MemoryWriter::new(self.memory.clone())));
            }
        }
        parts
    }

    fn services(&self) -> Vec<Arc<dyn Service>> {
        let memory = self.memory.clone();
        vec![Arc::new(FnService::new(
            "memory_decay",
            Schedule::Every(DECAY_INTERVAL),
            move || {
                let memory = memory.clone();
                async move {
                    memory
                        .decay(chrono::Utc::now())
                        .await
                        .map(|_| ())
                        .map_err(|error| ServiceError::new(error.to_string()))
                }
            },
        ))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HashEmbedder, InMemoryMemoryStore};

    #[test]
    fn session_scopes_evict_the_least_recently_started_session() {
        let mut scopes = SessionScopes::default();
        for index in 0..MAX_SESSION_SCOPES {
            scopes.insert(format!("session-{index}"), "owner".to_string());
        }
        // A new turn in the oldest session makes session-1 the oldest.
        scopes.insert("session-0".to_string(), "owner".to_string());
        scopes.insert("session-new".to_string(), "session:new".to_string());

        assert_eq!(scopes.entries.len(), MAX_SESSION_SCOPES);
        assert_eq!(scopes.get("session-0"), Some("owner"));
        assert_eq!(scopes.get("session-1"), None);
        assert_eq!(scopes.get("session-new"), Some("session:new"));
    }

    fn system() -> ScopedMemory {
        ScopedMemory::new(Memory::builder(
            Arc::new(InMemoryMemoryStore::new()),
            Arc::new(HashEmbedder::default()),
        ))
    }

    fn turn(read: bool, write: bool, explicit_writes_only: bool) -> MemoryTurn {
        MemoryTurn {
            session_id: "telegram:synthetic-user-1".to_string(),
            scope: "owner".to_string(),
            read,
            write,
            explicit_writes_only,
        }
    }

    fn tool_names(parts: &MemoryParts) -> Vec<String> {
        parts.tools.iter().map(|tool| tool.spec().name).collect()
    }

    #[tokio::test]
    async fn parts_follow_the_turn_permissions() {
        let system = system();

        let all = system.turn(&turn(true, true, false)).await;
        assert_eq!(all.hooks.len(), 2, "injection and writer");
        assert_eq!(tool_names(&all), ["recall", "remember"]);

        let explicit = system.turn(&turn(true, true, true)).await;
        assert_eq!(explicit.hooks.len(), 1, "no writer");
        assert_eq!(tool_names(&explicit), ["recall", "remember"]);

        let read_only = system.turn(&turn(true, false, false)).await;
        assert_eq!(read_only.hooks.len(), 1);
        assert_eq!(tool_names(&read_only), ["recall"]);

        let none = system.turn(&turn(false, false, false)).await;
        assert!(none.hooks.is_empty() && none.tools.is_empty());
    }

    #[tokio::test]
    async fn sessions_resolve_to_the_scope_their_turn_was_given() {
        let system = system();
        assert_eq!(
            system.memory().scope_for("telegram:synthetic-user-1"),
            "session:telegram:synthetic-user-1"
        );
        system.turn(&turn(true, true, false)).await;
        assert_eq!(
            system.memory().scope_for("telegram:synthetic-user-1"),
            "owner"
        );
    }
}
