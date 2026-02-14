use super::*;

#[async_trait]
impl crate::traits::NotificationStore for SqliteStateStore {
    async fn enqueue_notification(
        &self,
        entry: &crate::traits::NotificationEntry,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO notification_queue (id, goal_id, session_id, notification_type, priority, message, created_at, delivered_at, attempts, expires_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&entry.id)
        .bind(&entry.goal_id)
        .bind(&entry.session_id)
        .bind(&entry.notification_type)
        .bind(&entry.priority)
        .bind(&entry.message)
        .bind(&entry.created_at)
        .bind(&entry.delivered_at)
        .bind(entry.attempts)
        .bind(&entry.expires_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_pending_notifications(
        &self,
        limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::NotificationEntry>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, session_id, notification_type, priority, message, created_at, delivered_at, attempts, expires_at
             FROM notification_queue
             WHERE delivered_at IS NULL
               AND (expires_at IS NULL OR datetime(expires_at) > datetime('now'))
             ORDER BY
               CASE priority WHEN 'critical' THEN 0 ELSE 1 END ASC,
               created_at ASC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let mut entries = Vec::with_capacity(rows.len());
        for row in &rows {
            entries.push(crate::traits::NotificationEntry {
                id: row.get("id"),
                goal_id: row.get("goal_id"),
                session_id: row.get("session_id"),
                notification_type: row.get("notification_type"),
                priority: row.get("priority"),
                message: row.get("message"),
                created_at: row.get("created_at"),
                delivered_at: row.get("delivered_at"),
                attempts: row.get("attempts"),
                expires_at: row.get("expires_at"),
            });
        }
        Ok(entries)
    }

    async fn mark_notification_delivered(&self, notification_id: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE notification_queue SET delivered_at = datetime('now') WHERE id = ?")
            .bind(notification_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn increment_notification_attempt(&self, notification_id: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE notification_queue SET attempts = attempts + 1 WHERE id = ?")
            .bind(notification_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn cleanup_expired_notifications(&self) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "DELETE FROM notification_queue
             WHERE delivered_at IS NULL
               AND expires_at IS NOT NULL
               AND datetime(expires_at) <= datetime('now')",
        )
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() as i64)
    }
}
