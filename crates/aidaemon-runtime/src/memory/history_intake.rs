//! A new memory system learns from the retained conversations. Memory systems
//! share no record format, but any of them can read a conversation, so this
//! is how switching keeps what the daemon knew.
//!
//! Off by default: reading every conversation costs model calls. Only
//! messages present when the intake began are read, because the system sees
//! newer ones on its own turns. Privacy follows the rules a live turn has.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use aidaemon_core::{FnService, Schedule, Service, ServiceError};
use aidaemon_policies::{HistoryIntake as Intake, MemorySystem, PastMessage, PastRole};
use serde_json::Value;
use sqlx::{Row, SqlitePool};
use tracing::info;

use aidaemon_core::ChannelVisibility;

use crate::history::EventHistory;
use crate::stores::{SettingsStore, StateStore};

const BATCH: i64 = 200;
const INTERVAL: Duration = Duration::from_secs(60);

pub struct HistoryIntake {
    system: Arc<dyn MemorySystem>,
    pool: SqlitePool,
    settings: Arc<dyn SettingsStore>,
    history: EventHistory,
    cursor_key: String,
    ceiling_key: String,
    finished: AtomicBool,
}

impl HistoryIntake {
    /// `backend_id` names the memory system, so a later switch starts over.
    pub fn new(
        system: Arc<dyn MemorySystem>,
        pool: SqlitePool,
        state: Arc<dyn StateStore>,
        backend_id: &str,
    ) -> Self {
        Self {
            system,
            history: EventHistory::new(pool.clone(), Some(state.clone())),
            pool,
            settings: state,
            cursor_key: format!("memory_history_intake_cursor:{backend_id}"),
            ceiling_key: format!("memory_history_intake_ceiling:{backend_id}"),
            finished: AtomicBool::new(false),
        }
    }

    pub fn into_service(self) -> Arc<dyn Service> {
        let intake = Arc::new(self);
        Arc::new(FnService::new(
            "memory_history_intake",
            Schedule::Every(INTERVAL),
            move || {
                let intake = intake.clone();
                async move {
                    intake
                        .step()
                        .await
                        .map_err(|error| ServiceError::new(format!("{error:#}")))
                }
            },
        ))
    }

    async fn stored(&self, key: &str) -> anyhow::Result<Option<i64>> {
        Ok(self
            .settings
            .get_setting(key)
            .await?
            .and_then(|value| value.parse().ok()))
    }

    /// The last event this intake reads, fixed when it first runs.
    async fn ceiling(&self) -> anyhow::Result<i64> {
        if let Some(ceiling) = self.stored(&self.ceiling_key).await? {
            return Ok(ceiling);
        }
        let ceiling: Option<i64> = sqlx::query_scalar("SELECT MAX(id) FROM events")
            .fetch_one(&self.pool)
            .await?;
        let ceiling = ceiling.unwrap_or(0);
        self.settings
            .set_setting(&self.ceiling_key, &ceiling.to_string())
            .await?;
        Ok(ceiling)
    }

    /// The memory scope a live turn in this session would use, or `None` when
    /// the session must not reach long-term memory.
    async fn scope(&self, session_id: &str) -> anyhow::Result<Option<String>> {
        if self.history.session_is_memory_isolated(session_id).await? {
            return Ok(None);
        }
        let Some(visibility) = self
            .history
            .session_visibility_from_events(session_id)
            .await
        else {
            return Ok(None);
        };
        if visibility == ChannelVisibility::PublicExternal {
            return Ok(None);
        }
        let role: Option<String> = sqlx::query_scalar(
            "SELECT json_extract(data, '$.user_role') FROM events
             WHERE session_id = ? AND event_type = 'user_message'
             ORDER BY id DESC LIMIT 1",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();
        let personal = matches!(
            visibility,
            ChannelVisibility::Private | ChannelVisibility::Internal
        );
        Ok(Some(if role.as_deref() == Some("owner") && personal {
            "owner".to_string()
        } else {
            format!("session:{session_id}")
        }))
    }

    /// Hand one batch to the memory system and remember how far it got. A
    /// failed batch is retried on the next pass.
    async fn step(&self) -> anyhow::Result<()> {
        if self.finished.load(Ordering::Relaxed) {
            return Ok(());
        }
        let cursor = self.stored(&self.cursor_key).await?.unwrap_or(0);
        let ceiling = self.ceiling().await?;
        let rows = sqlx::query(
            "SELECT e.id, e.session_id, e.event_type, e.data, e.created_at FROM events e
             WHERE e.id > ? AND e.id <= ?
               AND e.event_type IN ('user_message', 'assistant_response')
               AND (e.task_id IS NULL OR EXISTS (
                   SELECT 1 FROM events policy
                   WHERE policy.task_id = e.task_id
                     AND policy.event_type = 'memory_policy_compiled'
                     AND json_extract(policy.data, '$.access') = 'allowed'
               ) AND NOT EXISTS (
                   SELECT 1 FROM events policy
                   WHERE policy.task_id = e.task_id
                     AND policy.event_type = 'memory_policy_compiled'
                     AND json_extract(policy.data, '$.access') = 'suppressed'
               ))
             ORDER BY e.id LIMIT ?",
        )
        .bind(cursor)
        .bind(ceiling)
        .bind(BATCH)
        .fetch_all(&self.pool)
        .await?;
        let Some(last) = rows.last().map(|row| row.get::<i64, _>("id")) else {
            self.finished.store(true, Ordering::Relaxed);
            info!("Memory history intake finished");
            return Ok(());
        };

        let mut scopes: HashMap<String, Option<String>> = HashMap::new();
        let mut batch = Vec::with_capacity(rows.len());
        for row in &rows {
            let session_id: String = row.get("session_id");
            if !scopes.contains_key(&session_id) {
                let scope = self.scope(&session_id).await?;
                scopes.insert(session_id.clone(), scope);
            }
            let Some(scope) = scopes[&session_id].clone() else {
                continue;
            };
            let data: Value = match serde_json::from_str(&row.get::<String, _>("data")) {
                Ok(data) => data,
                Err(_) => continue,
            };
            let Some(text) = data["content"]
                .as_str()
                .filter(|text| !text.trim().is_empty())
            else {
                continue;
            };
            batch.push(PastMessage {
                session_id,
                scope,
                role: if row.get::<String, _>("event_type") == "user_message" {
                    PastRole::User
                } else {
                    PastRole::Assistant
                },
                text: text.to_string(),
                at: row.get("created_at"),
            });
        }

        if !batch.is_empty() {
            match self.system.absorb_history(&batch).await {
                Ok(Intake::Absorbed) => {}
                Ok(Intake::NotSupported) => {
                    self.finished.store(true, Ordering::Relaxed);
                    info!("This memory system does not learn from history");
                    return Ok(());
                }
                Err(error) => anyhow::bail!("memory system refused the batch: {error}"),
            }
        }
        self.settings
            .set_setting(&self.cursor_key, &last.to_string())
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_policies::{MemoryParts, MemoryTurn};
    use async_trait::async_trait;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Recording {
        batches: Mutex<Vec<Vec<PastMessage>>>,
        fail: AtomicBool,
        unsupported: AtomicBool,
    }

    #[async_trait]
    impl MemorySystem for Recording {
        async fn turn(&self, _turn: &MemoryTurn) -> MemoryParts {
            MemoryParts::default()
        }

        async fn absorb_history(&self, messages: &[PastMessage]) -> Result<Intake, String> {
            if self.fail.load(Ordering::Relaxed) {
                return Err("synthetic failure".to_string());
            }
            if self.unsupported.load(Ordering::Relaxed) {
                return Ok(Intake::NotSupported);
            }
            self.batches.lock().unwrap().push(messages.to_vec());
            Ok(Intake::Absorbed)
        }
    }

    async fn setup() -> (
        tempfile::NamedTempFile,
        Arc<crate::state::SqliteStateStore>,
        Arc<Recording>,
        HistoryIntake,
    ) {
        let db = tempfile::NamedTempFile::new().unwrap();
        let embeddings = Arc::new(aidaemon_memory::embeddings::EmbeddingService::deterministic());
        let state = Arc::new(
            crate::state::SqliteStateStore::for_tests(
                db.path().to_str().unwrap(),
                100,
                None,
                embeddings,
            )
            .await
            .unwrap(),
        );
        let system = Arc::new(Recording::default());
        let intake = HistoryIntake::new(system.clone(), state.pool(), state.clone(), "test");
        (db, state, system, intake)
    }

    async fn event(
        state: &crate::state::SqliteStateStore,
        session: &str,
        kind: &str,
        text: &str,
        role: &str,
        visibility: &str,
    ) {
        let data = serde_json::json!({
            "content": text,
            "user_role": role,
            "channel_visibility": visibility,
        });
        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at)
             VALUES (?, ?, ?, '2026-01-01T00:00:00+00:00')",
        )
        .bind(session)
        .bind(kind)
        .bind(data.to_string())
        .execute(&state.pool())
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn keeps_live_privacy_rules_and_stops_at_the_ceiling() {
        let (_db, state, system, intake) = setup().await;
        event(
            &state,
            "owner-dm",
            "user_message",
            "synthetic question",
            "owner",
            "private",
        )
        .await;
        event(
            &state,
            "owner-dm",
            "assistant_response",
            "synthetic answer",
            "owner",
            "private",
        )
        .await;
        event(
            &state,
            "public-room",
            "user_message",
            "stranger text",
            "public",
            "public_external",
        )
        .await;
        event(
            &state,
            "team-room",
            "user_message",
            "group text",
            "owner",
            "public",
        )
        .await;
        event(
            &state,
            "specialist:executor:synthetic",
            "user_message",
            "worker text",
            "owner",
            "private",
        )
        .await;

        intake.step().await.unwrap();
        event(
            &state,
            "owner-dm",
            "user_message",
            "after the intake began",
            "owner",
            "private",
        )
        .await;
        intake.step().await.unwrap();
        intake.step().await.unwrap();

        let batches = system.batches.lock().unwrap();
        assert_eq!(batches.len(), 1);
        let seen: Vec<(&str, &str)> = batches[0]
            .iter()
            .map(|m| (m.scope.as_str(), m.text.as_str()))
            .collect();
        assert_eq!(
            seen,
            [
                ("owner", "synthetic question"),
                ("owner", "synthetic answer"),
                ("session:team-room", "group text"),
            ]
        );
        assert_eq!(batches[0][1].role, PastRole::Assistant);
    }

    /// A turn opted out of learning ("don't remember this") is not handed
    /// to the memory system, even though its task's policy allowed memory.
    #[tokio::test]
    async fn an_opted_out_turn_is_not_absorbed() {
        let (_db, state, system, intake) = setup().await;
        for (task, text) in [
            ("kept-task", "kept question"),
            ("private-task", "private question"),
        ] {
            let data = serde_json::json!({
                "content": text, "user_role": "owner", "channel_visibility": "private",
                "task_id": task,
            });
            let policy = serde_json::json!({"task_id": task, "access": "allowed"});
            for (kind, data) in [("memory_policy_compiled", policy), ("user_message", data)] {
                sqlx::query(
                    "INSERT INTO events (session_id, event_type, data, task_id, created_at)
                     VALUES ('owner-dm', ?, ?, ?, '2026-01-01T00:00:00+00:00')",
                )
                .bind(kind)
                .bind(data.to_string())
                .bind(task)
                .execute(&state.pool())
                .await
                .unwrap();
            }
        }
        let events = Arc::new(
            aidaemon_events::EventStore::on_paths(
                state.pool(),
                aidaemon_tools::LexicalPaths::daemon_host(),
                Arc::new(crate::history::HistoryEventHost),
            )
            .await
            .unwrap(),
        );
        crate::turn::opt_out::mark_turn_opted_out(
            &aidaemon_events::EventEmitter::new(events, "owner-dm"),
            "private-task",
            "private-turn",
        )
        .await
        .unwrap();

        intake.step().await.unwrap();
        let batches = system.batches.lock().unwrap();
        let seen: Vec<&str> = batches.iter().flatten().map(|m| m.text.as_str()).collect();
        assert_eq!(seen, ["kept question"]);
    }

    #[tokio::test]
    async fn a_failed_batch_is_retried_and_progress_survives_a_restart() {
        let (_db, state, system, intake) = setup().await;
        event(
            &state,
            "owner-dm",
            "user_message",
            "first",
            "owner",
            "private",
        )
        .await;

        system.fail.store(true, Ordering::Relaxed);
        assert!(intake.step().await.is_err());
        system.fail.store(false, Ordering::Relaxed);
        intake.step().await.unwrap();
        assert_eq!(system.batches.lock().unwrap().len(), 1);

        let restarted = HistoryIntake::new(system.clone(), state.pool(), state.clone(), "test");
        restarted.step().await.unwrap();
        assert_eq!(system.batches.lock().unwrap().len(), 1, "not read twice");
    }

    #[tokio::test]
    async fn a_system_that_cannot_learn_is_left_alone() {
        let (_db, state, system, intake) = setup().await;
        event(
            &state,
            "owner-dm",
            "user_message",
            "first",
            "owner",
            "private",
        )
        .await;
        system.unsupported.store(true, Ordering::Relaxed);
        intake.step().await.unwrap();
        assert!(intake.finished.load(Ordering::Relaxed));
    }
}
