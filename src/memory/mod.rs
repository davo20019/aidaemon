pub mod binary;
pub mod context_window;
pub mod embeddings;
pub mod expertise;
pub mod manager;
pub mod math;
pub mod people_intelligence;
pub mod proactive;
pub mod procedures;
pub mod retention;
pub mod scoring;
pub mod skill_promotion;
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
