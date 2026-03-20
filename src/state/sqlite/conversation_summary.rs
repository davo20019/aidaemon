use super::*;

#[async_trait]
impl crate::traits::ConversationSummaryStore for SqliteStateStore {
    async fn get_conversation_summary(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<ConversationSummary>> {
        let row = sqlx::query(
            "SELECT session_id, summary, message_count, last_message_id, updated_at
             FROM conversation_summaries WHERE session_id = ?",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| ConversationSummary {
            session_id: r.get("session_id"),
            summary: r.get("summary"),
            message_count: r.get::<i64, _>("message_count") as usize,
            last_message_id: r.get("last_message_id"),
            updated_at: r
                .get::<String, _>("updated_at")
                .parse::<DateTime<Utc>>()
                .unwrap_or_else(|_| Utc::now()),
        }))
    }

    async fn upsert_conversation_summary(
        &self,
        summary: &ConversationSummary,
    ) -> anyhow::Result<()> {
        // Monotonic guard: only update if new message_count is higher than existing.
        // Prevents out-of-order race condition when two async compaction tasks complete
        // in wrong order — older summaries can never overwrite newer ones.
        sqlx::query(
            "INSERT INTO conversation_summaries (session_id, summary, message_count, last_message_id, updated_at)
             VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(session_id) DO UPDATE SET
               summary = excluded.summary,
               message_count = excluded.message_count,
               last_message_id = excluded.last_message_id,
               updated_at = excluded.updated_at
             WHERE excluded.message_count > conversation_summaries.message_count",
        )
        .bind(&summary.session_id)
        .bind(&summary.summary)
        .bind(summary.message_count as i64)
        .bind(&summary.last_message_id)
        .bind(summary.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
