use super::*;

/// Insert the content-free terminal notice for one mandate run on the same
/// SQLite connection/transaction as its proof-state transition.
///
/// The deterministic primary key makes a retry idempotent. A conflicting row
/// is accepted only when every immutable delivery field is identical; silently
/// treating a different terminal outcome as a duplicate would hide a state
/// machine bug and could surface the wrong owner notice.
pub(super) async fn enqueue_mandate_run_notification_on_connection(
    connection: &mut sqlx::SqliteConnection,
    notice: &crate::traits::MandateRunNotification,
) -> anyhow::Result<bool> {
    notice.validate().map_err(|error| anyhow::anyhow!(error))?;
    let entry = notice.to_notification_entry();
    if entry.priority != "critical" || entry.expires_at.is_some() {
        anyhow::bail!("mandate owner notifications must be critical and non-expiring");
    }

    let inserted = sqlx::query(
        "INSERT INTO notification_queue
            (id, goal_id, session_id, notification_type, priority, message,
             created_at, delivered_at, attempts, expires_at, task_id, action_token)
         VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0, NULL, NULL, ?)
         ON CONFLICT(id) DO NOTHING",
    )
    .bind(&entry.id)
    .bind(&entry.goal_id)
    .bind(&entry.session_id)
    .bind(&entry.notification_type)
    .bind(&entry.priority)
    .bind(&entry.message)
    .bind(&entry.created_at)
    .bind(&entry.action_token)
    .execute(&mut *connection)
    .await?
    .rows_affected();

    if inserted == 1 {
        return Ok(true);
    }

    let existing = sqlx::query(
        "SELECT goal_id, session_id, notification_type, priority, message,
                created_at, expires_at, task_id, action_token
         FROM notification_queue
         WHERE id = ?",
    )
    .bind(&entry.id)
    .fetch_optional(&mut *connection)
    .await?
    .ok_or_else(|| {
        anyhow::anyhow!(
            "mandate notification insert reported a conflict but no row exists for {}",
            entry.id
        )
    })?;

    let matches = existing.get::<String, _>("goal_id") == entry.goal_id
        && existing.get::<String, _>("session_id") == entry.session_id
        && existing.get::<String, _>("notification_type") == entry.notification_type
        && existing.get::<String, _>("priority") == entry.priority
        && existing.get::<String, _>("message") == entry.message
        && existing.get::<String, _>("created_at") == entry.created_at
        && existing.get::<Option<String>, _>("expires_at") == entry.expires_at
        && existing.get::<Option<String>, _>("task_id") == entry.task_id
        && existing.get::<Option<String>, _>("action_token") == entry.action_token;
    if !matches {
        anyhow::bail!(
            "mandate notification id {} already belongs to different immutable content",
            entry.id
        );
    }
    Ok(false)
}

#[async_trait]
impl crate::traits::NotificationStore for SqliteStateStore {
    async fn enqueue_notification(
        &self,
        entry: &crate::traits::NotificationEntry,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO notification_queue
                (id, goal_id, session_id, notification_type, priority, message,
                 created_at, delivered_at, attempts, expires_at, task_id, action_token)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        .bind(&entry.task_id)
        .bind(&entry.action_token)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn enqueue_goal_notification(
        &self,
        entry: &crate::traits::NotificationEntry,
    ) -> anyhow::Result<bool> {
        let mut tx = self.pool.begin().await?;
        let notified_at = chrono::Utc::now().to_rfc3339();
        let claimed = sqlx::query(
            "UPDATE goals
             SET notified_at = ?, notification_attempts = notification_attempts + 1
             WHERE id = ? AND notified_at IS NULL",
        )
        .bind(&notified_at)
        .bind(&entry.goal_id)
        .execute(&mut *tx)
        .await?
        .rows_affected();

        if claimed == 0 {
            tx.rollback().await?;
            return Ok(false);
        }

        sqlx::query(
            "INSERT INTO notification_queue
                (id, goal_id, session_id, notification_type, priority, message,
                 created_at, delivered_at, attempts, expires_at, task_id, action_token)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        .bind(&entry.task_id)
        .bind(&entry.action_token)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(true)
    }

    async fn get_pending_notifications(
        &self,
        limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::NotificationEntry>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, session_id, notification_type, priority, message,
                    created_at, delivered_at, attempts, expires_at, task_id, action_token
             FROM notification_queue
             WHERE delivered_at IS NULL
               AND (expires_at IS NULL OR datetime(expires_at) > datetime('now'))
               AND (
                    priority = 'critical'
                    OR (
                        attempts < 10
                        AND datetime(created_at) > datetime('now', '-24 hours')
                    )
               )
             ORDER BY
               CASE priority WHEN 'critical' THEN 0 ELSE 1 END ASC,
               julianday(created_at) DESC,
               id DESC
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
                task_id: row.get("task_id"),
                action_token: row.get("action_token"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::traits::store_prelude::*;
    use crate::traits::{
        MandateRunNotification, MandateRunNotificationKind, MandateRunProofCounts,
    };
    use std::sync::Arc;

    async fn test_store() -> (SqliteStateStore, tempfile::NamedTempFile) {
        let database = tempfile::NamedTempFile::new().unwrap();
        let store = SqliteStateStore::new(
            database.path().to_str().unwrap(),
            100,
            None,
            Arc::new(EmbeddingService::new().unwrap()),
        )
        .await
        .unwrap();
        (store, database)
    }

    fn notice(kind: MandateRunNotificationKind) -> MandateRunNotification {
        MandateRunNotification::new(
            "mandate-12345678",
            7,
            "goal-12345678",
            "run-12345678",
            "owner-session",
            kind,
            MandateRunProofCounts {
                non_root_tasks: 1,
                mutation_reservations: 1,
                succeeded_mutations: 1,
                ..MandateRunProofCounts::default()
            },
            "2026-08-02T12:00:00Z",
        )
    }

    #[tokio::test]
    async fn mandate_notice_rolls_back_with_its_enclosing_state_transaction() {
        let (store, _database) = test_store().await;
        let mut tx = store.pool.begin().await.unwrap();
        assert!(enqueue_mandate_run_notification_on_connection(
            &mut tx,
            &notice(MandateRunNotificationKind::ActSatisfied),
        )
        .await
        .unwrap());
        tx.rollback().await.unwrap();

        let count =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM notification_queue WHERE id = ?")
                .bind("mandate-run-notice:run-12345678")
                .fetch_one(&store.pool)
                .await
                .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn mandate_notice_commit_and_duplicate_retry_are_idempotent() {
        let (store, _database) = test_store().await;
        let act_notice = notice(MandateRunNotificationKind::ActSatisfied);
        let mut tx = store.pool.begin().await.unwrap();
        assert!(
            enqueue_mandate_run_notification_on_connection(&mut tx, &act_notice)
                .await
                .unwrap()
        );
        tx.commit().await.unwrap();

        let mut retry = store.pool.begin().await.unwrap();
        assert!(
            !enqueue_mandate_run_notification_on_connection(&mut retry, &act_notice)
                .await
                .unwrap()
        );
        retry.commit().await.unwrap();

        let pending = store.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, act_notice.notification_id());
        assert_eq!(pending[0].priority, "critical");
        assert!(pending[0].expires_at.is_none());
        assert_eq!(
            pending[0].action_token.as_deref(),
            Some(pending[0].id.as_str())
        );

        let mut conflicting_retry = store.pool.begin().await.unwrap();
        let conflict = enqueue_mandate_run_notification_on_connection(
            &mut conflicting_retry,
            &notice(MandateRunNotificationKind::Ask),
        )
        .await;
        assert!(conflict.is_err());
        conflicting_retry.rollback().await.unwrap();
    }

    #[tokio::test]
    async fn critical_mandate_notice_survives_unbounded_delivery_retries() {
        let (store, _database) = test_store().await;
        let notice = notice(MandateRunNotificationKind::Ask);
        let mut tx = store.pool.begin().await.unwrap();
        enqueue_mandate_run_notification_on_connection(&mut tx, &notice)
            .await
            .unwrap();
        tx.commit().await.unwrap();

        for _ in 0..12 {
            store
                .increment_notification_attempt(&notice.notification_id())
                .await
                .unwrap();
        }
        let pending = store.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].attempts, 12);

        store
            .mark_notification_delivered(&notice.notification_id())
            .await
            .unwrap();
        assert!(store
            .get_pending_notifications(10)
            .await
            .unwrap()
            .is_empty());
    }
}
