//! What memory consolidation reads: a host's conversation history, as a
//! typed contract. The host implements it over its own event log; memory
//! never parses the host's storage.

use aidaemon_core::ChannelVisibility;
use async_trait::async_trait;
use chrono::{DateTime, Utc};

/// Who produced a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnRole {
    User,
    Assistant,
    /// A tool's result.
    Tool,
}

impl TurnRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// One turn of a session's history.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryTurn {
    /// The host's ordering id: later turns have larger ids.
    pub id: i64,
    pub role: TurnRole,
    pub content: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// A host's conversation history, as memory consolidation needs it.
///
/// "Memory-isolated" sessions (worker and mandate runs) never teach owner
/// memory; turns from tasks whose memory policy suppressed access are never
/// returned.
#[async_trait]
pub trait ConversationSource: Send + Sync {
    /// Sessions with conversation turns before `cutoff`, memory-isolated
    /// sessions left out.
    async fn sessions_with_turns_before(
        &self,
        cutoff: DateTime<Utc>,
    ) -> anyhow::Result<Vec<String>>;

    /// Whether the session is a worker or mandate run that must not teach
    /// owner memory.
    async fn is_memory_isolated(&self, session_id: &str) -> anyhow::Result<bool>;

    /// Whether every user message after turn `after_id` and before `cutoff`
    /// came from the owner, and there was at least one. A session that is
    /// memory-isolated is never owner-only.
    async fn owner_only_since(
        &self,
        session_id: &str,
        after_id: i64,
        cutoff: DateTime<Utc>,
    ) -> anyhow::Result<bool>;

    /// The largest turn id after `after_id` and before `cutoff`.
    async fn last_turn_id(
        &self,
        session_id: &str,
        after_id: i64,
        cutoff: DateTime<Utc>,
    ) -> anyhow::Result<Option<i64>>;

    /// Turns after `after_id` and before `cutoff`, oldest first, at most
    /// `limit`.
    async fn turns(
        &self,
        session_id: &str,
        after_id: i64,
        cutoff: DateTime<Utc>,
        limit: usize,
    ) -> anyhow::Result<Vec<HistoryTurn>>;

    /// The session's audience, from its latest user message.
    async fn visibility(&self, session_id: &str) -> Option<ChannelVisibility>;

    /// The channel facts learned in this session are scoped to.
    fn channel_id(&self, session_id: &str) -> Option<String>;
}

/// Per-session progress of a consolidation pass, so history is read once.
#[async_trait]
pub trait ConsolidationCursors: Send + Sync {
    /// The last turn id consolidated for the session (0 when none).
    async fn cursor(&self, session_id: &str) -> i64;

    async fn set_cursor(&self, session_id: &str, turn_id: i64) -> anyhow::Result<()>;
}
