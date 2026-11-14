use super::*;

fn background_spawn_from_row(row: &sqlx::sqlite::SqliteRow) -> crate::traits::BackgroundSpawn {
    crate::traits::BackgroundSpawn {
        id: row.get("id"),
        session_id: row.get("session_id"),
        mission: row.get("mission"),
        started_at: row.get("started_at"),
    }
}

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
    // Name the automation by the name the owner gave it, else its
    // owner-confirmed objective. Every queued mandate notice passes through
    // here, so no call site can forget it.
    let named;
    let notice = if notice.objective.is_some() && notice.name.is_some() {
        notice
    } else {
        let stored = sqlx::query("SELECT name, objective FROM mandates WHERE id = ?")
            .bind(&notice.mandate_id)
            .fetch_optional(&mut *connection)
            .await?;
        let (name, objective) = stored.map_or((None, None), |row| {
            (
                row.get::<Option<String>, _>("name"),
                row.get::<Option<String>, _>("objective"),
            )
        });
        let mut filled = notice.clone();
        if filled.name.is_none() {
            filled.name = name;
        }
        if filled.objective.is_none() {
            filled.objective = objective;
        }
        named = filled;
        &named
    };
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
        // Message text is excluded: it is rendered from these fields, and a
        // retry across an upgrade may word the same outcome differently.
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
impl<S: WorkDatabase> crate::traits::NotificationStore for S {
    async fn record_background_spawn(
        &self,
        spawn: &crate::traits::BackgroundSpawn,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO background_spawns (id, session_id, mission, started_at)
             VALUES (?, ?, ?, ?)",
        )
        .bind(&spawn.id)
        .bind(&spawn.session_id)
        .bind(&spawn.mission)
        .bind(&spawn.started_at)
        .execute(self.work_pool())
        .await?;
        Ok(())
    }

    async fn finish_background_spawn(&self, id: &str) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM background_spawns WHERE id = ?")
            .bind(id)
            .execute(self.work_pool())
            .await?;
        Ok(())
    }

    async fn background_spawns_for_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Vec<crate::traits::BackgroundSpawn>> {
        let rows = sqlx::query(
            "SELECT id, session_id, mission, started_at FROM background_spawns
             WHERE session_id = ? ORDER BY started_at",
        )
        .bind(session_id)
        .fetch_all(self.work_pool())
        .await?;
        Ok(rows.iter().map(background_spawn_from_row).collect())
    }

    async fn take_background_spawns(&self) -> anyhow::Result<Vec<crate::traits::BackgroundSpawn>> {
        let rows = sqlx::query(
            "DELETE FROM background_spawns RETURNING id, session_id, mission, started_at",
        )
        .fetch_all(self.work_pool())
        .await?;
        Ok(rows.iter().map(background_spawn_from_row).collect())
    }

    async fn enqueue_notification(
        &self,
        entry: &crate::traits::NotificationEntry,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO notification_queue
                (id, goal_id, session_id, notification_type, priority, message,
                 created_at, delivered_at, attempts, expires_at, task_id, action_token,
                 untrusted)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        .bind(entry.untrusted)
        .execute(self.work_pool())
        .await?;
        Ok(())
    }

    async fn enqueue_goal_notification(
        &self,
        entry: &crate::traits::NotificationEntry,
    ) -> anyhow::Result<bool> {
        let mut tx = self.work_pool().begin_with("BEGIN IMMEDIATE").await?;
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
                 created_at, delivered_at, attempts, expires_at, task_id, action_token,
                 untrusted)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        .bind(entry.untrusted)
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
                    created_at, delivered_at, attempts, expires_at, task_id, action_token,
                    untrusted
             FROM notification_queue
             WHERE delivered_at IS NULL
               AND (expires_at IS NULL OR datetime(expires_at) > datetime('now'))
               AND (
                    notification_type NOT IN ('progress', 'status_update')
                    OR NOT EXISTS (
                        SELECT 1
                        FROM notification_queue AS terminal
                        WHERE terminal.goal_id = notification_queue.goal_id
                          AND terminal.session_id = notification_queue.session_id
                          AND terminal.notification_type IN ('completed', 'failed', 'stalled')
                          AND julianday(terminal.created_at) >= julianday(notification_queue.created_at)
                    )
               )
               AND (
                    priority = 'critical'
                    OR (
                        attempts < 10
                        AND datetime(created_at) > datetime('now', '-24 hours')
                    )
             )
             ORDER BY
               -- Normally deliver durable outcomes first. If critical work
               -- already fills this whole page, reserve the front of the page
               -- for fresh progress so a dead delivery route cannot starve
               -- every live update forever.
               CASE
                 WHEN (
                   SELECT COUNT(*) FROM notification_queue AS critical_backlog
                   WHERE critical_backlog.delivered_at IS NULL
                     AND critical_backlog.priority = 'critical'
                 ) >= ?
                 THEN CASE priority WHEN 'status_update' THEN 0 ELSE 1 END
                 ELSE CASE priority WHEN 'critical' THEN 0 ELSE 1 END
               END ASC,
               julianday(created_at) DESC,
               id DESC
             LIMIT ?",
        )
        .bind(limit)
        .bind(limit)
        .fetch_all(self.work_pool())
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
                untrusted: row.get("untrusted"),
            });
        }
        Ok(entries)
    }

    async fn mark_notification_delivered(&self, notification_id: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE notification_queue SET delivered_at = datetime('now') WHERE id = ?")
            .bind(notification_id)
            .execute(self.work_pool())
            .await?;
        Ok(())
    }

    async fn increment_notification_attempt(&self, notification_id: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE notification_queue SET attempts = attempts + 1 WHERE id = ?")
            .bind(notification_id)
            .execute(self.work_pool())
            .await?;
        Ok(())
    }

    async fn notices_delivered_since(
        &self,
        session_id: &str,
        since: chrono::DateTime<chrono::Utc>,
    ) -> anyhow::Result<Vec<crate::traits::DeliveredNotice>> {
        // `delivered_at` is SQLite's `datetime('now')`: UTC, second precision.
        let rows = sqlx::query(
            "SELECT n.id, n.notification_type, n.message, n.delivered_at,
                    (SELECT m.id FROM mandates m WHERE m.goal_id = n.goal_id LIMIT 1) AS mandate_id
             FROM notification_queue n
             WHERE n.session_id = ?
               AND n.delivered_at IS NOT NULL
               AND datetime(n.delivered_at) >= datetime(?)
             ORDER BY datetime(n.delivered_at), n.created_at",
        )
        .bind(session_id)
        .bind(since.format("%Y-%m-%d %H:%M:%S").to_string())
        .fetch_all(self.work_pool())
        .await?;
        Ok(rows
            .iter()
            .map(|row| {
                let delivered_at: String = row.get("delivered_at");
                crate::traits::DeliveredNotice {
                    id: row.get("id"),
                    notification_type: row.get("notification_type"),
                    message: row.get("message"),
                    delivered_at: chrono::NaiveDateTime::parse_from_str(
                        &delivered_at,
                        "%Y-%m-%d %H:%M:%S",
                    )
                    .map(|at| at.and_utc().to_rfc3339())
                    .unwrap_or(delivered_at),
                    mandate_id: row.get("mandate_id"),
                }
            })
            .collect())
    }

    async fn cleanup_expired_notifications(&self) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "DELETE FROM notification_queue
             WHERE delivered_at IS NULL
               AND expires_at IS NOT NULL
               AND datetime(expires_at) <= datetime('now')",
        )
        .execute(self.work_pool())
        .await?;
        Ok(result.rows_affected() as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::*;
    use crate::traits::{
        MandateRunNotification, MandateRunNotificationKind, MandateRunProofCounts,
    };

    async fn test_store() -> (SqliteWorkStore, tempfile::NamedTempFile) {
        let database = tempfile::NamedTempFile::new().unwrap();
        let store = open_for_tests(database.path().to_str().unwrap()).await;
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

    /// A queued notice names the mandate by the name the owner gave it,
    /// carries no internal reference, and a turn that started before it was
    /// delivered gets it back as typed context for its reply.
    #[tokio::test]
    async fn named_mandate_notice_reaches_the_turn_that_was_running() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_continuous(
            "Watch the status page",
            "owner-session",
            Some(10_000),
            Some(50_000),
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Every hour, check the status page and notify the owner when it is down",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.name = Some("Status watch".to_string());
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let turn_started = chrono::Utc::now() - chrono::Duration::seconds(5);
        let failed = MandateRunNotification::new(
            &mandate.id,
            mandate.version,
            &goal.id,
            "run-synthetic-1",
            "owner-session",
            MandateRunNotificationKind::ReviewFailed {
                reason: MandateFinalizationRejectReason::DeliberatorFailed,
            },
            MandateRunProofCounts::default(),
            "2026-08-02T12:00:00Z",
        )
        // A caller that names only the objective still gets the name.
        .with_objective(Some(&mandate.objective));
        let mut tx = store.pool.begin().await.unwrap();
        assert!(
            enqueue_mandate_run_notification_on_connection(&mut tx, &failed)
                .await
                .unwrap()
        );
        tx.commit().await.unwrap();
        let entry_id = failed.notification_id();
        store.mark_notification_delivered(&entry_id).await.unwrap();

        let delivered = store
            .notices_delivered_since("owner-session", turn_started)
            .await
            .unwrap();
        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].id, entry_id);
        assert_eq!(
            delivered[0].mandate_id.as_deref(),
            Some(mandate.id.as_str())
        );
        assert_eq!(
            delivered[0].message,
            "Your automation \"Status watch\": a review stopped before it finished. Ask me what \
             it got to if you want the details. It's still active and will retry automatically."
        );
        let context = crate::goals::notices_turn_context(&delivered).unwrap();
        assert!(context.contains(&mandate.id), "{context}");
        assert!(context.contains("mandate_review_failed"), "{context}");

        // Another session, and a turn that started after it, see nothing.
        assert!(
            store
                .notices_delivered_since("other-session", turn_started)
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            store
                .notices_delivered_since(
                    "owner-session",
                    chrono::Utc::now() + chrono::Duration::seconds(5)
                )
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(crate::goals::notices_turn_context(&[]), None);
    }

    #[tokio::test]
    async fn mandate_notice_rolls_back_with_its_enclosing_state_transaction() {
        let (store, _database) = test_store().await;
        let mut tx = store.pool.begin().await.unwrap();
        assert!(
            enqueue_mandate_run_notification_on_connection(
                &mut tx,
                &notice(MandateRunNotificationKind::ActSatisfied),
            )
            .await
            .unwrap()
        );
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
        assert!(
            store
                .get_pending_notifications(10)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn live_status_updates_are_not_starved_by_critical_backlog() {
        let (store, _database) = test_store().await;
        for index in 0..25 {
            let entry = crate::traits::NotificationEntry::new(
                "old-goal",
                "unroutable-session",
                "failed",
                &format!("old critical failure {index}"),
            );
            store.enqueue_notification(&entry).await.unwrap();
        }
        let progress = crate::traits::NotificationEntry::new(
            "live-goal",
            "owner-session",
            "status_update",
            "The current run is making progress.",
        );
        store.enqueue_notification(&progress).await.unwrap();

        let pending = store.get_pending_notifications(20).await.unwrap();
        assert_eq!(pending[0].id, progress.id);
        assert_eq!(pending[0].priority, "status_update");
    }

    #[tokio::test]
    async fn terminal_outcome_supersedes_older_live_status_update() {
        let (store, _database) = test_store().await;
        let mut progress = crate::traits::NotificationEntry::new(
            "goal-12345678",
            "owner-session",
            "status_update",
            "Starting the recovered run now.",
        );
        progress.created_at = "2026-08-03T14:40:06Z".to_string();
        store.enqueue_notification(&progress).await.unwrap();

        let mut completed = crate::traits::NotificationEntry::new(
            "goal-12345678",
            "owner-session",
            "completed",
            "The recovered run completed.",
        );
        completed.created_at = "2026-08-03T14:42:18Z".to_string();
        store.enqueue_notification(&completed).await.unwrap();

        let pending = store.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, completed.id);
    }
}
