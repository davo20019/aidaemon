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

use crate::consolidation::Consolidator;
use crate::history::EventHistory;
use crate::llm::SharedLlmRuntime;
use crate::llm::gateway::SideCallGateway;
use crate::llm::side_calls::SideCall;
use crate::stores::StateStore;

/// The daemon's side of the memory jobs.
pub struct DaemonJobsHost {
    pool: SqlitePool,
    llm_runtime: SharedLlmRuntime,
    /// Runs the background jobs' model calls.
    side_calls: SideCallGateway,
    consolidator: Option<Arc<Consolidator>>,
    /// Compares a mentioned goal with the open ones.
    embedder: Arc<dyn aidaemon_memory::Embedder>,
}

/// The full memory system's background jobs over the daemon's store and
/// event log. Consolidation runs every `consolidation_interval`.
#[allow(clippy::too_many_arguments)]
pub fn memory_jobs(
    pool: SqlitePool,
    state: Arc<dyn StateStore>,
    memory: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    embedder: Arc<dyn aidaemon_memory::Embedder>,
    llm_runtime: SharedLlmRuntime,
    side_calls: SideCallGateway,
    consolidation_interval: Duration,
    consolidator: Option<Arc<Consolidator>>,
    project_notes: Option<Arc<aidaemon_memory::project::ProjectNotes>>,
) -> Arc<MemoryJobs> {
    let history = Arc::new(EventHistory::new(pool.clone(), Some(state)));
    let host = Arc::new(DaemonJobsHost {
        pool,
        llm_runtime,
        side_calls,
        consolidator,
        embedder: embedder.clone(),
    });
    Arc::new(
        MemoryJobs::new(memory, history, host, embedder)
            .with_schedule(JobSchedule {
                consolidation: consolidation_interval,
                ..JobSchedule::default()
            })
            .with_project_notes(project_notes),
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
        file_goal_observation(
            &self.pool,
            self.embedder.as_ref(),
            goal,
            episode_id,
            session_id,
        )
        .await
    }
}

/// Embedding similarity at or above which a mentioned goal is the same as
/// an open personal goal. Set high: a near-miss creates one more observed
/// goal, while a false match hides a new goal ("…in 2027") in an old one.
const SAME_GOAL_MIN_SIMILARITY: f32 = 0.92;

/// Whitespace-collapsed, lowercased goal text: the identity of a goal
/// description, not a similarity.
fn goal_identity(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// File `goal_text`, which episode `source_episode_id` mentioned, as a
/// personal goal observation. Background analysis never touches
/// orchestration goals.
///
/// - A completed or abandoned personal goal with the identical description
///   is not resurrected: nothing is filed.
/// - An open (observed or active) personal goal with the identical
///   description, or the most similar one by embedding at
///   [`SAME_GOAL_MIN_SIMILARITY`] or more, gets a provenance note.
/// - Otherwise a new observation is recorded. Completed and abandoned goals
///   are never compared by similarity, so "Run a marathon in 2027" is filed
///   even though "Run a marathon in 2026" was completed.
pub async fn file_goal_observation(
    pool: &SqlitePool,
    embedder: &dyn aidaemon_memory::Embedder,
    goal_text: &str,
    source_episode_id: i64,
    session_id: &str,
) -> anyhow::Result<()> {
    let goal_text = goal_text.trim();
    if goal_text.is_empty() {
        return Ok(());
    }
    let identity = goal_identity(goal_text);
    let existing = sqlx::query(
        "SELECT id, description, status FROM goals
         WHERE domain = 'personal'
           AND status IN ('observed', 'active', 'abandoned', 'completed')",
    )
    .fetch_all(pool)
    .await?;
    let mut open: Vec<(String, String)> = Vec::new();
    for row in existing {
        let id: String = row.get("id");
        let description: String = row.get("description");
        let status: String = row.get("status");
        let terminal = matches!(status.as_str(), "abandoned" | "completed");
        if goal_identity(&description) == identity {
            if terminal {
                return Ok(());
            }
            return attach_goal_reference(pool, &id, source_episode_id).await;
        }
        if !terminal {
            open.push((id, description));
        }
    }

    if !open.is_empty() {
        let mut texts = vec![goal_text.to_string()];
        texts.extend(open.iter().map(|(_, description)| description.clone()));
        match embedder.embed(&texts).await {
            Ok(vectors) if vectors.len() == texts.len() => {
                let best = open
                    .iter()
                    .zip(&vectors[1..])
                    .map(|((id, _), vector)| (id, aidaemon_memory::cosine(&vectors[0], vector)))
                    .filter(|(_, similarity)| *similarity >= SAME_GOAL_MIN_SIMILARITY)
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((id, _)) = best {
                    return attach_goal_reference(pool, id, source_episode_id).await;
                }
            }
            Ok(_) => {}
            // Without vectors only identity matches; a possible duplicate
            // observation is cheaper than a goal lost.
            Err(error) => {
                tracing::warn!(%error, "Could not compare a mentioned goal with open goals")
            }
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

/// Note on goal `id` that episode `source_episode_id` referenced it.
async fn attach_goal_reference(
    pool: &SqlitePool,
    id: &str,
    source_episode_id: i64,
) -> anyhow::Result<()> {
    let now = Utc::now().to_rfc3339();
    let note = format!("Referenced in episode {}", source_episode_id);
    let notes_row = sqlx::query("SELECT progress_notes FROM goals WHERE id = ?")
        .bind(id)
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
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Goal review, the daemon's weekly look at stale goals. Goals are work,
/// not memory, so it runs whatever memory system is configured.
/// Review and update stale goals.
pub async fn review_goals(pool: &SqlitePool) -> anyhow::Result<()> {
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
