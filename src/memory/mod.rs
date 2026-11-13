pub use aidaemon_memory::embeddings;
pub(crate) mod history_intake;
pub(crate) mod jobs_host;
pub mod people_intelligence;
pub mod procedures;
pub mod progressive;
pub mod retention;
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

pub use aidaemon_runtime::history::SessionChannelIds as DaemonChannelIds;

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
    aidaemon_memory::sqlite::SqliteMemory::new(pool.clone(), embeddings.clone())
        .with_message_log(std::sync::Arc::new(
            aidaemon_runtime::history::EventHistory::new(pool, None),
        ))
        .with_reranker(embeddings)
        .with_channel_ids(std::sync::Arc::new(DaemonChannelIds))
        .with_error_cleaner(without_scaffolding)
}
