pub mod binary;
pub mod core_profile;
pub mod embedding_index;
pub mod embeddings;
pub mod expertise;
pub(crate) mod history_intake;
pub mod history_source;
pub mod manager;
pub mod neighborhood;
pub mod people_intelligence;
pub mod procedures;
pub mod progressive;
pub mod retention;
#[cfg(test)]
pub mod retrieval_eval;
pub mod scoring;
pub mod task_learning;

/// Derive a canonical `channel_id` from a `session_id` string.
///
/// Used when only a session identifier is available (e.g., background learning pipelines)
/// but we still want channel-scoped memory provenance.
///
/// Session IDs commonly follow patterns like:
/// - Slack:  `slack:CHANNEL_ID:thread_ts` or `slack:CHANNEL_ID`
/// - Telegram: `bot:telegram:CHAT_ID`, `telegram:CHAT_ID`, or bare numeric chat IDs
/// - Discord: `discord:ch:ID` or `discord:dm:ID` (kept as-is)
pub fn derive_channel_id_from_session(session_id: &str) -> Option<String> {
    crate::session::derive_channel_id_from_session(session_id)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod comprehensive_tests;

/// Channel ids the way the daemon's sessions spell them.
pub struct DaemonChannelIds;

impl aidaemon_memory::sqlite::ChannelIds for DaemonChannelIds {
    fn same_channel(&self, stored: &str, current: &str) -> bool {
        crate::session::stored_channel_matches_current(stored, current)
    }

    fn channel_of_session(&self, session_id: &str) -> Option<String> {
        crate::session::derive_channel_id_from_session(session_id)
    }
}

/// The daemon's memory store over its state database. Errors keep only what
/// the failing tool said, without the daemon's own wrapper lines.
pub(crate) fn sqlite_memory(
    pool: sqlx::SqlitePool,
    embeddings: std::sync::Arc<embeddings::EmbeddingService>,
) -> aidaemon_memory::sqlite::SqliteMemory {
    fn without_scaffolding(error: &str) -> String {
        error
            .lines()
            .filter(|line| !crate::utils::is_internal_scaffolding_line(line))
            .collect::<Vec<_>>()
            .join("\n")
    }
    aidaemon_memory::sqlite::SqliteMemory::new(pool, embeddings.clone())
        .with_reranker(embeddings)
        .with_channel_ids(std::sync::Arc::new(DaemonChannelIds))
        .with_error_cleaner(without_scaffolding)
}
