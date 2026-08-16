//! Domain port for exact conversation-history search.

use std::collections::HashSet;

use async_trait::async_trait;
use serde::Serialize;

use crate::types::{ChannelVisibility, UserRole};

#[derive(Debug, Clone)]
pub struct HistoryScope {
    pub session_id: String,
    pub channel_id: Option<String>,
    pub visibility: ChannelVisibility,
    pub user_role: UserRole,
    pub trusted: bool,
    pub include_subagents: bool,
    pub session_filter: Option<String>,
    pub task_filter: Option<String>,
    pub snapshot_max_event_id: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryMessage {
    pub event_id: i64,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    pub role: String,
    pub content: String,
    pub created_at: String,
    pub source_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_rank: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskBookends {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective: Option<HistoryMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_objective: Option<String>,
    pub objective_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<HistoryMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_resolution: Option<String>,
    pub resolution_source: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ProjectionStats {
    pub projected: u64,
    pub orphans_removed: u64,
    pub fts_rebuilt: bool,
    pub episodes_repaired: u64,
    pub pending: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryCoverage {
    pub canonical_messages: i64,
    pub indexed_messages: i64,
    pub pending_messages: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oldest_indexed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub newest_indexed_at: Option<String>,
}

/// Exact-history operations exposed to the search tool through a domain port.
#[async_trait]
pub trait HistorySearchStore: super::EpisodeStore {
    async fn history_snapshot_max_event_id(&self) -> anyhow::Result<i64>;
    async fn history_coverage(&self) -> anyhow::Result<HistoryCoverage>;
    async fn repair_history_projection(
        &self,
        max_batches: usize,
    ) -> anyhow::Result<ProjectionStats>;
    async fn search_history(
        &self,
        query: &str,
        scope: &HistoryScope,
        limit: usize,
        semantic_sessions: &HashSet<String>,
    ) -> anyhow::Result<Vec<HistoryMessage>>;
    async fn history_context(
        &self,
        event_id: i64,
        radius: usize,
        scope: &HistoryScope,
    ) -> anyhow::Result<Vec<HistoryMessage>>;
    async fn history_event_for_message_id(
        &self,
        message_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<Option<i64>>;
    async fn history_turn(
        &self,
        turn_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<Vec<HistoryMessage>>;
    async fn history_page(
        &self,
        anchor: i64,
        older: bool,
        scope: &HistoryScope,
        limit: usize,
    ) -> anyhow::Result<Vec<HistoryMessage>>;
    async fn history_task_bookends(
        &self,
        task_id: Option<&str>,
        session_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<TaskBookends>;
}
