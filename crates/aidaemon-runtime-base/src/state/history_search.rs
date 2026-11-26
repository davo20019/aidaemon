//! History search over the state database: the projection and its queries
//! are `aidaemon_runtime::history::search`; this is the store's end.

use std::collections::HashSet;

use async_trait::async_trait;

pub(super) use crate::history::search::*;

use crate::state::types::{
    HistoryCoverage, HistoryMessage, HistoryScope, HistorySearchStore, HistoryToolActivity,
    ProjectionStats, RecentToolTurn, TaskBookends,
};

#[async_trait]
impl HistorySearchStore for super::SqliteStateStore {
    async fn history_snapshot_max_event_id(&self) -> anyhow::Result<i64> {
        snapshot_max_event_id(&self.pool()).await
    }

    async fn history_coverage(&self) -> anyhow::Result<HistoryCoverage> {
        coverage(&self.pool()).await
    }

    async fn repair_history_projection(
        &self,
        max_batches: usize,
    ) -> anyhow::Result<ProjectionStats> {
        repair_and_backfill(&self.pool(), max_batches).await
    }

    async fn search_history(
        &self,
        query: &str,
        scope: &HistoryScope,
        limit: usize,
        semantic_sessions: &HashSet<String>,
    ) -> anyhow::Result<Vec<HistoryMessage>> {
        search(&self.pool(), query, scope, limit, semantic_sessions).await
    }

    async fn history_context(
        &self,
        event_id: i64,
        radius: usize,
        scope: &HistoryScope,
    ) -> anyhow::Result<Vec<HistoryMessage>> {
        context(&self.pool(), event_id, radius, scope).await
    }

    async fn history_event_for_message_id(
        &self,
        message_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<Option<i64>> {
        event_for_message_id(&self.pool(), message_id, scope).await
    }

    async fn history_turn(
        &self,
        turn_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<Vec<HistoryMessage>> {
        turn(&self.pool(), turn_id, scope).await
    }

    async fn history_turn_tool_activity(
        &self,
        turn_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<Vec<HistoryToolActivity>> {
        turn_tool_activity(&self.pool(), turn_id, scope).await
    }

    async fn history_recent_tool_turns(
        &self,
        session_id: &str,
        turns: usize,
    ) -> anyhow::Result<Vec<RecentToolTurn>> {
        recent_tool_turns(&self.pool(), session_id, turns).await
    }

    async fn history_page(
        &self,
        anchor: i64,
        older: bool,
        scope: &HistoryScope,
        limit: usize,
    ) -> anyhow::Result<Vec<HistoryMessage>> {
        page(&self.pool(), anchor, older, scope, limit).await
    }

    async fn history_task_bookends(
        &self,
        task_id: Option<&str>,
        session_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<TaskBookends> {
        task_bookends(&self.pool(), task_id, session_id, scope).await
    }
}
