//! The store traits for a session's messages, its dialogue state, its
//! conversation summary and the rendered prompt snapshots that
//! `instructions_snapshot` events point at. The host's SQLite store
//! implements them.

use async_trait::async_trait;

use crate::conversation::{ConversationSummary, Message, MessageAnnotation};
use crate::dialogue::DialogueState;

/// Session message storage and context retrieval.
#[async_trait]
pub trait MessageStore: Send + Sync {
    /// Append a message to the session history hot window.
    /// Canonical persistence is handled via emitted events.
    async fn append_message(&self, msg: &Message) -> anyhow::Result<()>;

    /// Add `annotation` to a stored user message (identified by its id,
    /// which is also its turn id), in the hot window and in the canonical
    /// event. A store that cannot annotate ignores it: annotations added
    /// this way are advisory rendering hints.
    async fn annotate_user_message(
        &self,
        _session_id: &str,
        _message_id: &str,
        _annotation: MessageAnnotation,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get recent messages for a session from working memory.
    async fn get_history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<Message>>;

    /// Get context using Tri-Hybrid retrieval (Recency + Vector + Salience).
    /// Default implementation just calls `get_history`.
    ///
    /// Pillar B (Task 7): the last production caller (`load_initial_history`)
    /// was removed when the turn-anchored fetch took over history retention.
    /// Retained as part of the `MessageStore` contract (still exercised by the
    /// sqlite store tests); allow dead_code until a future caller or removal.
    #[allow(dead_code)]
    async fn get_context(
        &self,
        session_id: &str,
        _query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        self.get_history(session_id, limit).await
    }

    /// HARD-clear a session: delete its canonical events + conversation summary
    /// (working memory too). Destructive and irreversible — this is `/wipe`.
    /// Facts already extracted into memory are preserved.
    async fn clear_session(&self, session_id: &str) -> anyhow::Result<()>;

    /// Reset a session's conversation CONTEXT without deleting anything. The
    /// next turn starts fresh (a durable boundary hides prior messages from
    /// context retrieval), but the event history remains for the memory
    /// pipeline and audit. This is what `/clear` should do. The default falls
    /// back to the destructive clear for stores that don't implement a
    /// boundary (keeps simple/test stores functional).
    async fn clear_session_context(&self, session_id: &str) -> anyhow::Result<()> {
        self.clear_session(session_id).await
    }

    /// Advance the durable context floor without hiding the current turn.
    ///
    /// This is the lifecycle equivalent of starting a new conversation thread:
    /// earlier events remain queryable for audit and explicit history lookup,
    /// while ordinary prompt/context hydration begins after
    /// `cleared_after_event_id`. Stores without a durable boundary may keep the
    /// default no-op; production stores should also prune their hot context to
    /// `retained_turn_id` so an in-memory cache cannot bypass the boundary.
    async fn advance_session_context_boundary(
        &self,
        _session_id: &str,
        _cleared_after_event_id: i64,
        _retained_turn_id: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Durable projection of a session's open request/question state.
#[async_trait]
pub trait DialogueStateStore: Send + Sync {
    async fn get_dialogue_state(&self, session_id: &str) -> anyhow::Result<Option<DialogueState>>;

    async fn upsert_dialogue_state(&self, state: &DialogueState) -> anyhow::Result<()>;

    #[allow(dead_code)]
    async fn delete_dialogue_state(&self, session_id: &str) -> anyhow::Result<()>;
}

/// Sliding-window conversation summaries.
#[async_trait]
pub trait ConversationSummaryStore: Send + Sync {
    /// Get the conversation summary for a session.
    async fn get_conversation_summary(
        &self,
        _session_id: &str,
    ) -> anyhow::Result<Option<ConversationSummary>> {
        Ok(None)
    }

    /// Create or update a conversation summary for a session.
    async fn upsert_conversation_summary(
        &self,
        _summary: &ConversationSummary,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Rendered system-prompt snapshots, deduplicated by content hash.
///
/// Enables exact replay of past LLM calls: the `instructions_snapshot`
/// decision-point event records the core prompt's `core_hash` (plus the
/// volatile context tail inline); this store maps that hash back to the full
/// rendered core prompt text. Rows are written insert-or-ignore, so storage
/// grows only when the rendered prompt actually changes (deploys, config or
/// memory shape changes) — not per interaction.
#[async_trait]
pub trait PromptSnapshotStore: Send + Sync {
    /// Persist a rendered prompt keyed by its hash. Must be idempotent.
    async fn save_prompt_snapshot(&self, _hash: &str, _content: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Fetch a stored prompt snapshot by exact hash.
    #[allow(dead_code)] // Read path is db_probe; kept on the trait for tests/tools.
    async fn get_prompt_snapshot(&self, _hash: &str) -> anyhow::Result<Option<String>> {
        Ok(None)
    }
}
