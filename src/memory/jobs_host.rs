//! What the full memory system's background jobs
//! ([`aidaemon_memory::full::MemoryJobs`]) leave to the daemon: the side-call
//! models, learning from a session's events (the daemon's
//! [`Consolidator`]) and filing the personal goals episodes mention. The jobs
//! themselves, and when they run, are the crate's.

use std::sync::Arc;
use std::time::Duration;

use aidaemon_memory::full::{JobCall, JobSchedule, JobsHost, MemoryJobs};
use chrono::Utc;
use sqlx::{Row, SqlitePool};
use tracing::info;

use super::embeddings::EmbeddingService;
use super::history_source::EventHistory;
use crate::events::Consolidator;
use crate::llm_runtime::gateway::SideCallGateway;
use crate::llm_runtime::side_calls::SideCall;
use crate::llm_runtime::SharedLlmRuntime;
use crate::traits::StateStore;

/// The daemon's side of the memory jobs.
pub(crate) struct DaemonJobsHost {
    pool: SqlitePool,
    llm_runtime: SharedLlmRuntime,
    /// Runs the background jobs' model calls.
    side_calls: SideCallGateway,
    consolidator: Option<Arc<Consolidator>>,
}

/// The full memory system's background jobs over the daemon's store and
/// event log. Consolidation runs every `consolidation_interval`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn memory_jobs(
    pool: SqlitePool,
    state: Arc<dyn StateStore>,
    memory: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    embedding_service: Arc<EmbeddingService>,
    llm_runtime: SharedLlmRuntime,
    side_calls: SideCallGateway,
    consolidation_interval: Duration,
    consolidator: Option<Arc<Consolidator>>,
) -> Arc<MemoryJobs> {
    let history = Arc::new(EventHistory::new(pool.clone(), Some(state)));
    let host = Arc::new(DaemonJobsHost {
        pool,
        llm_runtime,
        side_calls,
        consolidator,
    });
    Arc::new(
        MemoryJobs::new(memory, history, host, embedding_service).with_schedule(JobSchedule {
            consolidation: consolidation_interval,
            ..JobSchedule::default()
        }),
    )
}

#[async_trait::async_trait]
impl JobsHost for DaemonJobsHost {
    fn model(&self, call: JobCall) -> (Arc<dyn aidaemon_core::Provider>, String) {
        let side_call = match call {
            JobCall::FactConsolidation => SideCall::FactConsolidation,
            JobCall::SessionAnalysis => SideCall::SessionAnalysis,
        };
        // The side-call gateway picks the model; this value is not sent.
        let model = self.llm_runtime.snapshot().cheap_model();
        (self.side_calls.provider(side_call), model)
    }

    async fn learn_session(&self, session_id: &str) -> anyhow::Result<()> {
        if let Some(consolidator) = &self.consolidator {
            consolidator.consolidate_session(session_id).await?;
        }
        Ok(())
    }

    async fn file_goal(&self, goal: &str, episode_id: i64, session_id: &str) -> anyhow::Result<()> {
        Self::file_goal_observation(&self.pool, goal, episode_id, session_id).await
    }
}

impl DaemonJobsHost {
    /// Extract or update a goal from episode analysis.
    async fn file_goal_observation(
        pool: &SqlitePool,
        goal_text: &str,
        source_episode_id: i64,
        session_id: &str,
    ) -> anyhow::Result<()> {
        // Check only personal goal observations. Background memory analysis must
        // never mutate or satisfy executable orchestration goals.
        let existing = sqlx::query(
            "SELECT id, description, status FROM goals
             WHERE domain = 'personal'
               AND status IN ('observed', 'active', 'abandoned', 'completed')",
        )
        .fetch_all(pool)
        .await?;

        // Simple text similarity check
        let goal_lower = goal_text.to_lowercase();
        for row in existing {
            let id: String = row.get("id");
            let description: String = row.get("description");
            let status: String = row.get("status");
            let desc_lower = description.to_lowercase();

            // Check for word overlap
            let goal_words: std::collections::HashSet<&str> =
                goal_lower.split_whitespace().collect();
            let desc_words: std::collections::HashSet<&str> =
                desc_lower.split_whitespace().collect();
            let intersection = goal_words.intersection(&desc_words).count();
            let union = goal_words.union(&desc_words).count();

            if union > 0 && (intersection as f32 / union as f32) > 0.5 {
                // Similar terminal goals are never resurrected by background analysis.
                if matches!(status.as_str(), "abandoned" | "completed") {
                    return Ok(());
                }

                // Existing observation or explicit personal goal — attach provenance.
                let now = Utc::now().to_rfc3339();
                let note = format!("Referenced in episode {}", source_episode_id);

                let notes_row = sqlx::query("SELECT progress_notes FROM goals WHERE id = ?")
                    .bind(&id)
                    .fetch_one(pool)
                    .await?;
                let notes_json: Option<String> = notes_row.get("progress_notes");
                let mut notes: Vec<String> = notes_json
                    .and_then(|j| serde_json::from_str(&j).ok())
                    .unwrap_or_default();
                notes.push(note);
                let notes_json = serde_json::to_string(&notes)?;

                sqlx::query("UPDATE goals SET progress_notes = ?, updated_at = ? WHERE id = ?")
                    .bind(&notes_json)
                    .bind(&now)
                    .bind(&id)
                    .execute(pool)
                    .await?;

                return Ok(());
            }
        }

        // Model-extracted mentions are observations, not active commitments.
        // Explicit personal goals are created through the normal goal API.
        let now = Utc::now().to_rfc3339();
        let goal_id = uuid::Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO goals
                (id, description, domain, goal_type, status, priority,
                 source_episode_id, session_id, created_at, updated_at)
             VALUES (?, ?, 'personal', 'finite', 'observed', 'medium', ?, ?, ?, ?)",
        )
        .bind(&goal_id)
        .bind(goal_text)
        .bind(source_episode_id)
        .bind(session_id)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await?;

        info!(
            goal = goal_text,
            episode_id = source_episode_id,
            "Recorded possible personal goal from episode"
        );
        Ok(())
    }
}

/// Goal review, the daemon's weekly look at stale goals. Goals are work,
/// not memory, so it runs whatever memory system is configured.
/// Review and update stale goals.
pub(crate) async fn review_goals(pool: &SqlitePool) -> anyhow::Result<()> {
    // Mark goals without progress in 14 days as potentially abandoned
    let two_weeks_ago = (Utc::now() - chrono::Duration::days(14)).to_rfc3339();

    let stale_goals =
        sqlx::query("SELECT id, description FROM goals WHERE status = 'active' AND updated_at < ?")
            .bind(&two_weeks_ago)
            .fetch_all(pool)
            .await?;

    for row in stale_goals {
        let id: String = row.get("id");
        let description: String = row.get("description");
        info!(
            goal_id = %id,
            description, "Stale goal detected - may need user input"
        );
    }

    Ok(())
}

#[cfg(test)]
#[path = "jobs_tests.rs"]
mod tests;
