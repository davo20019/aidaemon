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

/// One user message from the host's log, to index as a searchable span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoggedMessage {
    /// The host's id for the message; spans keep it as their source.
    pub id: i64,
    pub session_id: String,
    pub content: String,
    /// RFC 3339.
    pub created_at: String,
}

/// The host's message log, as the memory store links into it: episodes
/// record the messages they cover, and indexed messages are dropped once the
/// host no longer keeps them. Memory never reads the host's tables itself.
#[async_trait]
pub trait MessageLog: Send + Sync {
    /// The first and last user or assistant message of `session_id` from
    /// `start` to `end`, both inclusive.
    async fn message_bounds(
        &self,
        session_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> anyhow::Result<(Option<i64>, Option<i64>)>;

    /// Which of `ids` the host still keeps.
    async fn retained(&self, ids: &[i64]) -> anyhow::Result<std::collections::HashSet<i64>>;
}

/// One conversation turn from the host's log, as the background jobs read
/// it. The host leaves out its own internal turns (a run talking to itself)
/// and turns of tasks whose memory policy suppressed memory.
#[derive(Debug, Clone, PartialEq)]
pub struct LoggedTurn {
    pub role: TurnRole,
    pub content: Option<String>,
    /// For an assistant turn: the tools it called, by name, in order.
    pub tool_calls: Vec<String>,
    /// For a tool result: the tool's name.
    pub tool_name: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Which sessions [`ActivityLog::uncaptured_sessions`] returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UncapturedSessions {
    /// At least this many conversation turns since the session's latest
    /// episode ended.
    pub min_turns: usize,
    /// Only turns before this count, and the session's last such turn is
    /// before it too: the session has gone idle. `None`: any turn counts.
    pub idle_before: Option<DateTime<Utc>>,
}

/// The host's conversation log as the full memory system's background jobs
/// read it: which sessions have turns no episode covers, those turns, and
/// the recent activity patterns are learned from. Memory never reads the
/// host's tables itself.
#[async_trait]
pub trait ActivityLog: ConversationSource + ConsolidationCursors {
    /// Sessions whose conversation turns since their latest episode end
    /// pass `which`. `episode_ends` holds each session's latest episode end
    /// (RFC 3339); a session missing from it has no episode. Memory-isolated
    /// sessions may be returned: the jobs check.
    async fn uncaptured_sessions(
        &self,
        episode_ends: &[(String, String)],
        which: UncapturedSessions,
    ) -> anyhow::Result<Vec<String>>;

    /// Whether every user message after `since` (the session's latest
    /// episode; `None`: the whole session) came from the owner, and there
    /// was at least one. A memory-isolated session never is.
    async fn owner_only_after(
        &self,
        session_id: &str,
        since: Option<&crate::records::EpisodeBound>,
    ) -> anyhow::Result<bool>;

    /// The session's conversation turns after `since` (the latest episode;
    /// `None`: from the start), oldest first, at most `limit` read.
    async fn turns_after(
        &self,
        session_id: &str,
        since: Option<&crate::records::EpisodeBound>,
        limit: usize,
    ) -> anyhow::Result<Vec<LoggedTurn>>;

    /// Conversation turns of sessions that are not memory-isolated, from
    /// `since` on, newest first, at most `limit` read.
    async fn recent_turns(
        &self,
        since: DateTime<Utc>,
        limit: usize,
    ) -> anyhow::Result<Vec<LoggedTurn>>;

    /// The owner's own messages from `since` on, in sessions that are not
    /// memory-isolated, newest first, at most `limit` read.
    async fn recent_owner_messages(
        &self,
        since: DateTime<Utc>,
        limit: usize,
    ) -> anyhow::Result<Vec<LoggedTurn>>;
}
