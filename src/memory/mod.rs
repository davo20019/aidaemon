pub mod binary;
pub mod context_window;
pub mod core_profile;
pub mod embedding_index;
pub mod embeddings;
pub mod expertise;
pub mod graph_store;
pub mod hybrid;
pub mod manager;
pub mod math;
pub mod neighborhood;
pub mod people_intelligence;
pub mod procedures;
pub mod retention;
#[cfg(test)]
pub mod retrieval_eval;
pub mod scoring;
pub mod skill_promotion;
pub mod task_learning;

/// How long an extracted fact stays true, as declared by the extractor.
///
/// Background extractors see raw tool output, so without this they file run
/// state ("last command", "disk check result") as durable owner memory.
/// Transient observations are dropped at write time instead of being stored
/// and pinned. Anything unrecognised is durable, so an extractor that never
/// emits the field keeps its old behaviour.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FactLifetime {
    #[default]
    Durable,
    Transient,
}

impl FactLifetime {
    pub fn is_transient(self) -> bool {
        self == Self::Transient
    }
}

impl<'de> serde::Deserialize<'de> for FactLifetime {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = Option::<serde_json::Value>::deserialize(deserializer)?;
        Ok(match raw.as_ref().and_then(serde_json::Value::as_str) {
            Some(value) if value.trim().eq_ignore_ascii_case("transient") => Self::Transient,
            _ => Self::Durable,
        })
    }
}

/// Extractor prompt sentence that defines the `lifetime` field.
pub const FACT_LIFETIME_INSTRUCTION: &str = "Set \"lifetime\":\"transient\" on any fact that \
     describes state likely to change within days: the last command run, a single run's status \
     or result, current disk/CPU/queue numbers, a draft in progress, or today's plan. Omit \
     \"lifetime\" for facts that stay true (names, preferences, project locations, tech stacks).";

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
