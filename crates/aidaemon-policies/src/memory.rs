//! Memory as pluggable seams: a whole [`MemorySystem`] contributes hooks and
//! tools to each turn, and [`MemoryInjection`] is a `before_model` hook over
//! a [`Recall`] backend.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use aidaemon_core::{Hook, HookContext, Message, ModelRequest, Tool};
use async_trait::async_trait;

use crate::{insert_after_system, last_user_message_index};

/// A whole long-term memory system, replaceable as one unit.
///
/// The host (an app built on the SDK) decides who may see what: it resolves
/// the turn's [`MemoryTurn::scope`] and permissions from its own notion of
/// users and channels. The memory system only decides how to remember and
/// recall within that scope, by returning the hooks and tools it wants on
/// the turn. `aidaemon-memory` is one implementation; any other backend
/// (a vector database, a hosted memory service, an external memory server)
/// implements this trait in its own crate.
#[async_trait]
pub trait MemorySystem: Send + Sync {
    /// The hooks and tools this memory system adds to `turn`.
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts;
}

/// What a [`MemorySystem`] may do on one turn, as the host decided it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryTurn {
    /// The conversation the turn belongs to.
    pub session_id: String,
    /// The memory the turn reads and writes, shared by every session the
    /// host maps to it (for example one scope for the owner's direct
    /// messages, and one per group conversation).
    pub scope: String,
    /// The turn may read memory (recall, injection).
    pub read: bool,
    /// The turn may write memory.
    pub write: bool,
    /// Writes come only from explicit tool calls: no background learning
    /// from the transcript (worker and child sessions).
    pub explicit_writes_only: bool,
}

/// What a [`MemorySystem`] adds to one turn.
#[derive(Default)]
pub struct MemoryParts {
    pub hooks: Vec<Arc<dyn Hook>>,
    pub tools: Vec<Arc<dyn Tool>>,
}

impl std::fmt::Debug for MemoryParts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryParts")
            .field("hooks", &self.hooks.len())
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.spec().name)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// A memory backend: return up to `limit` snippets relevant to `query`.
#[async_trait]
pub trait Recall: Send + Sync {
    async fn recall(&self, session_id: &str, query: &str, limit: usize) -> Vec<String>;

    /// The system note [`MemoryInjection`] adds for `query`, or `None` for no
    /// note. The default lists the [`Recall::recall`] snippets under a
    /// "may be outdated" banner; a backend with its own layout (sections,
    /// authoritative profile facts) overrides this.
    async fn render(&self, session_id: &str, query: &str, limit: usize) -> Option<String> {
        let snippets = self.recall(session_id, query, limit).await;
        if snippets.is_empty() {
            return None;
        }
        let mut note = String::from(
            "[Recalled memory — background that may be outdated or irrelevant. Use it only if it \
             helps with the current request, and prefer what the user says now.]",
        );
        for snippet in snippets.iter().take(limit) {
            note.push_str("\n- ");
            note.push_str(snippet.trim());
        }
        Some(note)
    }
}

/// Injects recalled snippets for the latest user message as a system note
/// placed after the leading system messages. Request only: the stored
/// transcript never contains the note.
///
/// The query is the latest message the *user* wrote (harness feedback is
/// skipped). Results are cached per session and query, so the backend is
/// asked once per user message, not once per turn.
pub struct MemoryInjection<R: Recall> {
    recall: R,
    limit: usize,
    cache: Mutex<HashMap<String, (String, Option<String>)>>,
}

impl<R: Recall> MemoryInjection<R> {
    pub fn new(recall: R) -> Self {
        Self {
            recall,
            limit: 5,
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn backend(&self) -> &R {
        &self.recall
    }

    async fn note(&self, session_id: &str, query: &str) -> Option<String> {
        if let Some((cached_query, note)) = self
            .cache
            .lock()
            .ok()
            .and_then(|c| c.get(session_id).cloned())
        {
            if cached_query == query {
                return note;
            }
        }
        let note = self.recall.render(session_id, query, self.limit).await;
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(session_id.to_string(), (query.to_string(), note.clone()));
        }
        note
    }
}

#[async_trait]
impl<R: Recall> Hook for MemoryInjection<R> {
    async fn before_model(&self, ctx: &HookContext<'_>, request: &mut ModelRequest) {
        let Some(index) = last_user_message_index(ctx.messages) else {
            return;
        };
        let query = ctx.messages[index].content.clone();
        if let Some(note) = self.note(ctx.session_id, &query).await {
            insert_after_system(&mut request.messages, Message::hook_note(note));
        }
    }
}

/// A tiny in-process [`Recall`] with word-overlap scoring, for tests and
/// examples. Entries are per session, plus global entries visible to all.
#[derive(Default)]
pub struct InMemoryRecall {
    entries: Mutex<Vec<(Option<String>, String)>>,
    queries: Mutex<u32>,
}

impl InMemoryRecall {
    pub fn new() -> Self {
        Self::default()
    }

    /// Remember `text` for one session.
    pub fn remember(&self, session_id: &str, text: impl Into<String>) {
        if let Ok(mut e) = self.entries.lock() {
            e.push((Some(session_id.to_string()), text.into()));
        }
    }

    /// Remember `text` for every session.
    pub fn remember_global(&self, text: impl Into<String>) {
        if let Ok(mut e) = self.entries.lock() {
            e.push((None, text.into()));
        }
    }

    /// How many times `recall` was called.
    pub fn query_count(&self) -> u32 {
        self.queries.lock().map(|q| *q).unwrap_or(0)
    }
}

fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3)
        .map(str::to_lowercase)
        .collect()
}

#[async_trait]
impl Recall for InMemoryRecall {
    async fn recall(&self, session_id: &str, query: &str, limit: usize) -> Vec<String> {
        if let Ok(mut q) = self.queries.lock() {
            *q += 1;
        }
        let query_words = words(query);
        let entries = self.entries.lock().map(|e| e.clone()).unwrap_or_default();
        let mut scored: Vec<(usize, usize, String)> = entries
            .into_iter()
            .enumerate()
            .filter(|(_, (session, _))| session.as_deref().is_none_or(|s| s == session_id))
            .map(|(i, (_, text))| (words(&text).intersection(&query_words).count(), i, text))
            .filter(|(score, _, _)| *score > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.into_iter().take(limit).map(|(_, _, t)| t).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn word_overlap_ranks_and_scopes_by_session() {
        let recall = InMemoryRecall::new();
        recall.remember(
            "telegram:synthetic-user-1",
            "Alice Rivera prefers window seats",
        );
        recall.remember(
            "telegram:synthetic-user-1",
            "Alice Rivera's daughters are Mia and Zoe",
        );
        recall.remember(
            "telegram:synthetic-user-2",
            "Jordan Lee prefers aisle seats",
        );
        recall.remember_global("Acme Corp travel policy requires economy seats");

        let got = recall
            .recall("telegram:synthetic-user-1", "Book seats for Alice", 5)
            .await;
        assert_eq!(got[0], "Alice Rivera prefers window seats");
        assert!(got.contains(&"Acme Corp travel policy requires economy seats".to_string()));
        assert!(!got.iter().any(|s| s.contains("Jordan")));
        assert!(recall
            .recall("telegram:synthetic-user-1", "zzz", 5)
            .await
            .is_empty());
    }
}
