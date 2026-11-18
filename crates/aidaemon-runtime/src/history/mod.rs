//! The conversation history the host keeps: the event log read as memory
//! needs it ([`EventHistory`]), exact-history search over it ([`search`]),
//! and the event store's seams onto both ([`HistoryEventHost`]).
//!
//! Conversation history is the host's, not a memory system's: another
//! memory system reads it through the same `aidaemon-memory` traits
//! (`MessageLog`, `ActivityLog`) this implements.

mod activity;
pub mod search;
mod source;

pub use activity::uncaptured_sessions_sql;
pub use source::EventHistory;

use sqlx::SqlitePool;

/// Channel ids the way the daemon's sessions spell them.
pub struct SessionChannelIds;

impl aidaemon_memory::sqlite::ChannelIds for SessionChannelIds {
    fn same_channel(&self, stored: &str, current: &str) -> bool {
        aidaemon_events::session::stored_channel_matches_current(stored, current)
    }

    fn channel_of_session(&self, session_id: &str) -> Option<String> {
        aidaemon_events::session::derive_channel_id_from_session(session_id)
    }
}

/// Index the user message event `event_id` as a searchable memory span, if
/// memory may learn from it.
pub async fn project_event_span(pool: &SqlitePool, event_id: i64) -> anyhow::Result<()> {
    let history = EventHistory::new(pool.clone(), None);
    if let Some(message) = history.learnable_user_message(event_id).await? {
        aidaemon_memory::sqlite::project_message_span(pool, &SessionChannelIds, &message).await?;
    }
    Ok(())
}

/// The recurrence pattern of an error message. Harness scaffolding
/// (untrusted-data envelopes, truncation notices, coaching) is the host's
/// own wrapper, not the error, so its lines are dropped first; keeping them
/// made unrelated errors share a prefix and match each other.
pub fn error_pattern(error: &str) -> String {
    let error = error
        .lines()
        .filter(|line| !crate::message_content::is_internal_scaffolding_line(line))
        .collect::<Vec<_>>()
        .join("\n");
    aidaemon_memory::error_pattern(&error)
}

/// The host's end of the event store's seams: a stored message is projected
/// into exact-history search and, when memory may learn from it, memory's
/// span index; error text loses the host's scaffolding before it is grouped.
pub struct HistoryEventHost;

#[async_trait::async_trait]
impl aidaemon_events::EventHost for HistoryEventHost {
    async fn project_history(&self, pool: &SqlitePool, event_id: i64) -> anyhow::Result<bool> {
        search::project_event(pool, event_id).await
    }

    async fn project_memory_span(&self, pool: &SqlitePool, event_id: i64) -> anyhow::Result<()> {
        project_event_span(pool, event_id).await
    }

    fn error_pattern(&self, error: &str) -> String {
        error_pattern(error)
    }
}
