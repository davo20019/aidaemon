//! SQLite [`ApprovalStore`] for `ChannelRunner` (feature `channels`).
//!
//! The store is durable storage only. After a restart the runner reads the
//! records back with `load_all` and decides each one: still open (waits for
//! the answer, with the time it has left), expired while stopped (denied, and
//! the conversation is told), or its run is gone (dropped with a log line).
//! Who may answer is decided there too, by the conversation in the record,
//! so one chat cannot settle another's approval. This store keeps the scope
//! fields intact and makes every write atomic and idempotent.

use aidaemon_channels::{ApprovalStore, StoredApproval};
use aidaemon_core::StoreError;
use async_trait::async_trait;
use sqlx::{Row, SqlitePool};

use crate::{Error, SqliteStore, NOW};

/// Open approvals kept in the `channel_approvals` table of a [`SqliteStore`]
/// database (created by migration 2). Cheap to clone.
#[derive(Debug, Clone)]
pub struct SqliteApprovalStore {
    pool: SqlitePool,
}

impl SqliteApprovalStore {
    /// Share the store's database (and its migrations).
    pub fn new(store: &SqliteStore) -> Self {
        Self {
            pool: store.pool().clone(),
        }
    }
}

#[async_trait]
impl ApprovalStore for SqliteApprovalStore {
    /// One atomic upsert, so concurrent saves of a prompt never duplicate it.
    async fn save(&self, approval: &StoredApproval) -> Result<(), StoreError> {
        let record = serde_json::to_string(approval).map_err(Error::from)?;
        sqlx::query(&format!(
            "INSERT INTO channel_approvals
                 (prompt_id, session_id, channel, conversation_id, expires_at_ms, record, saved_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, {NOW})
             ON CONFLICT (prompt_id) DO UPDATE SET
                 session_id = excluded.session_id,
                 channel = excluded.channel,
                 conversation_id = excluded.conversation_id,
                 expires_at_ms = excluded.expires_at_ms,
                 record = excluded.record,
                 saved_at = excluded.saved_at"
        ))
        .bind(&approval.prompt_id)
        .bind(&approval.session_id)
        .bind(&approval.channel)
        .bind(&approval.conversation.id)
        .bind(i64::try_from(approval.expires_at_ms).unwrap_or(i64::MAX))
        .bind(record)
        .execute(&self.pool)
        .await
        .map_err(Error::from)?;
        Ok(())
    }

    /// Removing an id that is not there is fine: a double press settles once.
    async fn remove(&self, prompt_id: &str) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM channel_approvals WHERE prompt_id = ?1")
            .bind(prompt_id)
            .execute(&self.pool)
            .await
            .map_err(Error::from)?;
        Ok(())
    }

    /// Every open record, oldest save first. A row this build cannot decode
    /// (written by newer code) is skipped, not fatal: it could not be
    /// resumed here, and one bad row must not hide the rest.
    async fn load_all(&self) -> Result<Vec<StoredApproval>, StoreError> {
        let rows = sqlx::query("SELECT record FROM channel_approvals ORDER BY saved_at, prompt_id")
            .fetch_all(&self.pool)
            .await
            .map_err(Error::from)?;
        Ok(rows
            .iter()
            .filter_map(|row| {
                let text: String = row.get("record");
                serde_json::from_str(&text).ok()
            })
            .collect())
    }
}
