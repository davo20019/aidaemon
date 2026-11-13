//! Memory as pluggable seams: a whole [`MemorySystem`] contributes hooks and
//! tools to each turn, and [`MemoryInjection`] is a `before_model` hook over
//! a [`Recall`] backend.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use aidaemon_core::{Hook, HookContext, Message, ModelRequest, Service, Tool};
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

    /// Standing instructions for the model on how this memory system works.
    /// Memory systems differ, so the host says nothing about remembering
    /// except what this returns. It should not change between turns.
    fn guidance(&self) -> Option<String> {
        None
    }

    /// Learn from the host's retained conversations, oldest first, in
    /// batches. This is how a system starts out knowing what a previous one
    /// knew: memory systems share no record format, but any of them can read
    /// a conversation. The host resumes where the last batch ended.
    async fn absorb_history(&self, _messages: &[PastMessage]) -> Result<HistoryIntake, String> {
        Ok(HistoryIntake::NotSupported)
    }

    /// The background work this memory system needs (consolidation, expiry,
    /// indexing). The host runs these in place of its own memory jobs.
    fn services(&self) -> Vec<Arc<dyn Service>> {
        Vec::new()
    }
}

/// Who said a [`PastMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PastRole {
    User,
    Assistant,
}

/// One message from the host's retained conversations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PastMessage {
    pub session_id: String,
    /// The memory it belongs to, as the host would scope a turn today.
    pub scope: String,
    pub role: PastRole,
    pub text: String,
    /// RFC 3339.
    pub at: String,
}

/// What became of a batch given to [`MemorySystem::absorb_history`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryIntake {
    Absorbed,
    /// This system does not learn from history. The host stops offering.
    NotSupported,
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
    /// Whether the turn may be learned from. Read it when about to learn
    /// (at run end, in background work), not when the turn starts: the
    /// user or the model may opt the turn out while it runs.
    pub learning: TurnLearning,
    /// What the host knows about the turn beyond the above, for a memory
    /// system built to read it (see [`HostTurn`]). Empty by default.
    pub host: HostTurn,
}

impl MemoryTurn {
    /// The owner's turn in `session_id`: the `"owner"` memory, read and
    /// written, learned from at run end.
    pub fn owner(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            scope: "owner".to_string(),
            read: true,
            write: true,
            explicit_writes_only: false,
            learning: Default::default(),
            host: Default::default(),
        }
    }
}

/// Host-specific facts about a turn, for a memory system that knows its
/// host's types (the turn's full audience, a diagnostics sink). A system
/// that does not ask for them works from the [`MemoryTurn`] fields alone.
/// Not compared: two turns with the same fields are equal whatever they
/// carry here.
#[derive(Clone, Default)]
pub struct HostTurn(Option<Arc<dyn std::any::Any + Send + Sync>>);

impl HostTurn {
    pub fn new<T: std::any::Any + Send + Sync>(facts: T) -> Self {
        Self(Some(Arc::new(facts)))
    }

    /// The facts, if the host gave a `T`.
    pub fn get<T: std::any::Any>(&self) -> Option<&T> {
        self.0.as_deref().and_then(|facts| facts.downcast_ref())
    }
}

impl std::fmt::Debug for HostTurn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() {
            "HostTurn(..)"
        } else {
            "HostTurn(none)"
        })
    }
}

impl PartialEq for HostTurn {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for HostTurn {}

/// Whether a turn may be learned from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Learning {
    #[default]
    Allowed,
    /// "Don't remember this": nothing from the turn may become long-term
    /// memory (facts, episodes, procedures, people, skills), now or later.
    /// Recall is unaffected, and so are writes the turn makes on purpose
    /// with a memory tool.
    OptedOut,
}

/// A turn's [`Learning`], shared by the host and the memory system for the
/// whole turn. The host opts the turn out (an explicit command before the
/// message, or the model's call to the host's opt-out tool); a memory system
/// checks [`TurnLearning::allowed`] before it learns anything from the turn.
/// Opting out cannot be undone within the turn.
#[derive(Clone, Default)]
pub struct TurnLearning(Arc<AtomicBool>);

impl TurnLearning {
    pub fn new(learning: Learning) -> Self {
        let handle = Self::default();
        if learning == Learning::OptedOut {
            handle.opt_out();
        }
        handle
    }

    pub fn get(&self) -> Learning {
        if self.0.load(Ordering::Acquire) {
            Learning::OptedOut
        } else {
            Learning::Allowed
        }
    }

    pub fn allowed(&self) -> bool {
        self.get() == Learning::Allowed
    }

    /// Opt the turn out. Returns whether this call changed it.
    pub fn opt_out(&self) -> bool {
        !self.0.swap(true, Ordering::AcqRel)
    }
}

impl std::fmt::Debug for TurnLearning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("TurnLearning").field(&self.get()).finish()
    }
}

/// Compares the current values.
impl PartialEq for TurnLearning {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl Eq for TurnLearning {}

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
            && cached_query == query
        {
            return note;
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

    #[test]
    fn opting_out_is_shared_by_every_holder_of_the_turn() {
        let host = TurnLearning::default();
        let system = host.clone();
        assert_eq!(system.get(), Learning::Allowed);
        assert!(host.opt_out());
        assert!(!host.opt_out(), "already opted out");
        assert_eq!(system.get(), Learning::OptedOut);
        assert!(!system.allowed());
        assert!(!TurnLearning::new(Learning::OptedOut).allowed());
    }

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
        assert!(
            recall
                .recall("telegram:synthetic-user-1", "zzz", 5)
                .await
                .is_empty()
        );
    }
}
