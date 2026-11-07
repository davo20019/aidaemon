use crate::events::{Consolidator, EventStore};
use crate::llm_runtime::SharedLlmRuntime;
use crate::memory::binary::encode_embedding;
use crate::memory::embedding_index::{
    content_hash, EmbeddingIndex, EmbeddingItem, MemoryOwner, SqliteEmbeddingIndex,
};
use crate::memory::embeddings::EmbeddingService;
use crate::memory::embeddings::EMBEDDING_MODEL_ID;
use crate::memory::history_source::EventHistory;
use crate::traits::{Message, MessageAnnotation, StateStore};
use crate::types::UserRole;
use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

pub struct MemoryManager {
    pool: SqlitePool,
    history: EventHistory,
    embedding_service: Arc<EmbeddingService>,
    llm_runtime: SharedLlmRuntime,
    consolidation_interval: Duration,
    consolidator: Option<Arc<Consolidator>>,
    event_store: Option<Arc<EventStore>>,
    state: Option<Arc<dyn StateStore>>,
}

impl MemoryManager {
    pub fn new(
        pool: SqlitePool,
        embedding_service: Arc<EmbeddingService>,
        llm_runtime: SharedLlmRuntime,
        consolidation_interval: Duration,
        consolidator: Option<Arc<Consolidator>>,
    ) -> Self {
        Self {
            history: EventHistory::new(pool.clone(), None),
            pool,
            embedding_service,
            llm_runtime,
            consolidation_interval,
            consolidator,
            event_store: None,
            state: None,
        }
    }

    pub fn with_event_store(mut self, event_store: Arc<EventStore>) -> Self {
        self.event_store = Some(event_store);
        self
    }

    /// Set the state store memory is written to.
    pub fn with_state(mut self, state: Arc<dyn StateStore>) -> Self {
        self.history = EventHistory::new(self.pool.clone(), Some(state.clone()));
        self.state = Some(state);
        self
    }

    /// Register all memory background jobs with the heartbeat coordinator.
    /// Methods are public so the heartbeat can call them individually.
    pub fn register_heartbeat_jobs(
        self: &Arc<Self>,
        heartbeat: &mut crate::heartbeat::HeartbeatCoordinator,
    ) {
        // Embedding generation (every 5s)
        let mgr = self.clone();
        heartbeat.register_job("embeddings", Duration::from_secs(5), move || {
            let m = mgr.clone();
            async move {
                let _ = m.process_embeddings().await?;
                Ok(())
            }
        });

        // Memory consolidation (configurable interval, default 6h)
        let mgr = self.clone();
        let interval = self.consolidation_interval;
        heartbeat.register_job("consolidation", interval, move || {
            let m = mgr.clone();
            async move { m.consolidate_memories().await }
        });

        // Memory decay (daily)
        let mgr = self.clone();
        heartbeat.register_job("memory_decay", Duration::from_secs(24 * 3600), move || {
            let m = mgr.clone();
            async move { m.decay_memories().await }
        });

        // Goal review (weekly)
        let mgr = self.clone();
        heartbeat.register_job(
            "goal_review",
            Duration::from_secs(7 * 24 * 3600),
            move || {
                let m = mgr.clone();
                async move { m.review_goals().await }
            },
        );

        // Episode creation (every 30 minutes) — idle sessions + long-running active sessions
        let mgr = self.clone();
        heartbeat.register_job("episodes", Duration::from_secs(30 * 60), move || {
            let m = mgr.clone();
            async move {
                if let Err(e) = m.create_episodes_for_idle_sessions().await {
                    tracing::warn!(error = %e, "Failed to create episodes for idle sessions");
                }
                m.create_episodes_for_active_long_sessions().await
            }
        });

        // Pattern detection and style analysis (every 6 hours)
        let mgr = self.clone();
        heartbeat.register_job(
            "pattern_detection",
            Duration::from_secs(6 * 3600),
            move || {
                let m = mgr.clone();
                async move { m.analyze_recent_activity().await }
            },
        );

        info!("Memory background tasks registered with heartbeat");
    }

    async fn process_embeddings(&self) -> anyhow::Result<bool> {
        let mut did_work = false;
        did_work |= self.process_procedure_embeddings().await?;
        did_work |= self.process_error_solution_embeddings().await?;
        Ok(did_work)
    }

    async fn process_procedure_embeddings(&self) -> anyhow::Result<bool> {
        let rows = sqlx::query(
            "SELECT id, trigger_pattern FROM procedures p
             WHERE trigger_pattern IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1 FROM memory_embeddings me
                   WHERE me.owner_type = 'procedure' AND me.owner_id = CAST(p.id AS TEXT)
                     AND me.embedding_purpose = 'trigger' AND me.embedding_model = ?
                     AND me.stale_at IS NULL
               )
             LIMIT 10",
        )
        .bind(EMBEDDING_MODEL_ID)
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(false);
        }

        info!(
            "Generating trigger embeddings for {} procedures",
            rows.len()
        );
        for row in rows {
            let id: i64 = row.get("id");
            let trigger: String = row.get("trigger_pattern");
            let trigger = trigger.trim();
            if trigger.is_empty() {
                continue;
            }
            match self.embedding_service.embed(trigger.to_string()).await {
                Ok(embedding) => {
                    let blob = encode_embedding(&embedding);
                    let _ = sqlx::query("UPDATE procedures SET trigger_embedding = ? WHERE id = ?")
                        .bind(blob)
                        .bind(id)
                        .execute(&self.pool)
                        .await;
                    if let Err(error) = SqliteEmbeddingIndex::new(self.pool.clone())
                        .upsert(EmbeddingItem {
                            owner: MemoryOwner::new("procedure", id),
                            purpose: "trigger".to_string(),
                            model: EMBEDDING_MODEL_ID.to_string(),
                            content_hash: content_hash(trigger),
                            embedding,
                        })
                        .await
                    {
                        warn!(procedure_id = id, %error, "Failed to project procedure embedding");
                    }
                }
                Err(e) => {
                    warn!(
                        procedure_id = id,
                        error = %e,
                        "Failed to generate trigger embedding for procedure"
                    );
                }
            }
        }

        Ok(true)
    }

    async fn process_error_solution_embeddings(&self) -> anyhow::Result<bool> {
        let rows = sqlx::query(
            "SELECT id, error_pattern FROM error_solutions e
             WHERE error_pattern IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1 FROM memory_embeddings me
                   WHERE me.owner_type = 'error_solution' AND me.owner_id = CAST(e.id AS TEXT)
                     AND me.embedding_purpose = 'error' AND me.embedding_model = ?
                     AND me.stale_at IS NULL
               )
             LIMIT 10",
        )
        .bind(EMBEDDING_MODEL_ID)
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(false);
        }

        info!("Generating embeddings for {} error solutions", rows.len());
        for row in rows {
            let id: i64 = row.get("id");
            let pat: String = row.get("error_pattern");
            let pat = pat.trim();
            if pat.is_empty() {
                continue;
            }
            match self.embedding_service.embed(pat.to_string()).await {
                Ok(embedding) => {
                    let blob = encode_embedding(&embedding);
                    let _ =
                        sqlx::query("UPDATE error_solutions SET error_embedding = ? WHERE id = ?")
                            .bind(blob)
                            .bind(id)
                            .execute(&self.pool)
                            .await;
                    if let Err(error) = SqliteEmbeddingIndex::new(self.pool.clone())
                        .upsert(EmbeddingItem {
                            owner: MemoryOwner::new("error_solution", id),
                            purpose: "error".to_string(),
                            model: EMBEDDING_MODEL_ID.to_string(),
                            content_hash: content_hash(pat),
                            embedding,
                        })
                        .await
                    {
                        warn!(error_solution_id = id, %error, "Failed to project error-solution embedding");
                    }
                }
                Err(e) => {
                    warn!(
                        error_solution_id = id,
                        error = %e,
                        "Failed to generate embedding for error solution"
                    );
                }
            }
        }

        Ok(true)
    }

    /// Consolidate durable facts from owner history (the SDK memory crate's
    /// [`FactConsolidator`] over the daemon's event log), then run the
    /// event-based extraction (procedures, error solutions, expertise) for
    /// the same sessions.
    ///
    /// [`FactConsolidator`]: aidaemon_memory::consolidation::FactConsolidator
    async fn consolidate_memories(&self) -> anyhow::Result<()> {
        let Some(state) = self.state.clone() else {
            return Ok(());
        };
        let snapshot = self.llm_runtime.snapshot();
        // Background work does not need the strongest model.
        let model = snapshot.cheap_model();
        let chain: Arc<dyn aidaemon_core::Provider> = Arc::new(snapshot.background_chain(&model));
        let provider: Arc<dyn aidaemon_core::Provider> = match &self.event_store {
            Some(event_store) => Arc::new(crate::agent::AccountedProvider {
                inner: chain,
                accounting: crate::agent::ModelCallAccounting {
                    state: state.clone(),
                    event_store: event_store.clone(),
                    purpose: "memory_consolidation",
                },
            }),
            None => chain,
        };
        let report = aidaemon_memory::consolidation::FactConsolidator::new(provider, model)
            .consolidate(
                &self.history,
                &self.history,
                state.as_ref(),
                chrono::Utc::now(),
            )
            .await?;

        if let Some(consolidator) = &self.consolidator {
            for session_id in &report.consolidated_sessions {
                // The consolidator enforces the same boundary; recheck so a
                // newly classified mandate session never enters learning.
                if self.history.session_is_memory_isolated(session_id).await? {
                    continue;
                }
                if let Err(e) = consolidator.consolidate_session(session_id).await {
                    warn!(
                        "Event consolidation failed for session {}: {}",
                        session_id, e
                    );
                }
            }
        }
        Ok(())
    }

    /// Create an episode from a session's messages: the SDK memory crate's
    /// [`SessionAnalyst`] summarizes and records it, then the daemon files the
    /// goals it found.
    ///
    /// [`SessionAnalyst`]: aidaemon_memory::episodes::SessionAnalyst
    pub async fn create_episode(
        &self,
        session_id: &str,
        messages: &[Message],
    ) -> anyhow::Result<i64> {
        anyhow::ensure!(
            !self.history.session_is_memory_isolated(session_id).await?,
            "Memory-isolated worker sessions cannot create global episodes"
        );
        let Some(state) = self.state.clone() else {
            anyhow::bail!("No memory store to record the episode in");
        };
        let entries = transcript_entries(messages);
        let start_time = messages
            .first()
            .map(|m| m.created_at)
            .unwrap_or_else(Utc::now);
        let end_time = messages
            .last()
            .map(|m| m.created_at)
            .unwrap_or_else(Utc::now);
        let snapshot = self.llm_runtime.snapshot();
        // An episode is a short summary of a finished session: fast-tier work.
        let model = snapshot.cheap_model();
        let chain: Arc<dyn aidaemon_core::Provider> = Arc::new(snapshot.background_chain(&model));
        let provider: Arc<dyn aidaemon_core::Provider> = match &self.event_store {
            Some(event_store) => Arc::new(crate::agent::AccountedProvider {
                inner: chain,
                accounting: crate::agent::ModelCallAccounting {
                    state: state.clone(),
                    event_store: event_store.clone(),
                    purpose: "episode_creation",
                },
            }),
            None => chain,
        };
        let (episode_id, analysis) =
            aidaemon_memory::episodes::SessionAnalyst::new(provider, model)
                .record_episode(
                    state.as_ref(),
                    session_id,
                    &entries,
                    messages.len(),
                    start_time,
                    end_time,
                    crate::memory::derive_channel_id_from_session(session_id),
                )
                .await?;
        for goal_text in &analysis.goals_mentioned {
            if let Err(e) = self.extract_goal(goal_text, episode_id, session_id).await {
                warn!("Failed to extract goal '{}': {}", goal_text, e);
            }
        }
        Ok(episode_id)
    }

    /// Extract or update a goal from episode analysis.
    async fn extract_goal(
        &self,
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
        .fetch_all(&self.pool)
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
                    .fetch_one(&self.pool)
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
                    .execute(&self.pool)
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
        .execute(&self.pool)
        .await?;

        info!(
            goal = goal_text,
            episode_id = source_episode_id,
            "Recorded possible personal goal from episode"
        );
        Ok(())
    }

    // ==================== Communication Style Analysis ====================

    /// Merge restated facts, then let unrecalled memory fade (the SDK
    /// memory crate's decay over the daemon store).
    pub async fn decay_memories(&self) -> anyhow::Result<()> {
        let Some(state) = &self.state else {
            return Ok(());
        };
        aidaemon_memory::stores::decay_memory(
            state.as_ref(),
            &aidaemon_memory::stores::DecayPolicy::default(),
        )
        .await?;
        info!("Applied memory decay");
        Ok(())
    }

    // ==================== Goal Review ====================

    /// Review and update stale goals.
    pub async fn review_goals(&self) -> anyhow::Result<()> {
        // Mark goals without progress in 14 days as potentially abandoned
        let two_weeks_ago = (Utc::now() - chrono::Duration::days(14)).to_rfc3339();

        let stale_goals = sqlx::query(
            "SELECT id, description FROM goals WHERE status = 'active' AND updated_at < ?",
        )
        .bind(&two_weeks_ago)
        .fetch_all(&self.pool)
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

    // ==================== Background Task Helpers ====================

    /// Create episodes for sessions that have been idle for 30+ minutes without an episode.
    async fn create_episodes_for_idle_sessions(&self) -> anyhow::Result<()> {
        let thirty_mins_ago = (Utc::now() - chrono::Duration::minutes(30)).to_rfc3339();

        // Find sessions with conversation events older than 30 mins that have
        // uncaptured events (events newer than the latest episode, or no episode
        // at all) and at least 5 such events.
        let idle_sessions: Vec<String> = sqlx::query_scalar(
            "SELECT ev.session_id
             FROM events ev
             LEFT JOIN (
                 SELECT session_id, MAX(end_time) AS latest_end
                 FROM episodes
                 GROUP BY session_id
             ) ep ON ep.session_id = ev.session_id
             WHERE ev.event_type IN ('user_message', 'assistant_response', 'tool_result')
               AND (ep.latest_end IS NULL OR ev.created_at > ep.latest_end)
               AND ev.created_at < ?
             GROUP BY ev.session_id
             HAVING COUNT(ev.id) >= 5
               AND MAX(ev.created_at) < ?",
        )
        .bind(&thirty_mins_ago)
        .bind(&thirty_mins_ago)
        .fetch_all(&self.pool)
        .await?;
        let idle_sessions = self
            .history
            .exclude_memory_isolated_sessions(idle_sessions)
            .await?;

        if idle_sessions.is_empty() {
            return Ok(());
        }

        info!(
            count = idle_sessions.len(),
            "Creating episodes for idle sessions"
        );

        for session_id in idle_sessions {
            if !self
                .history
                .session_since_last_episode_is_owner_only(&session_id)
                .await?
            {
                continue;
            }
            // Fetch messages since the latest episode (or all if none)
            let messages = self
                .fetch_session_messages_since_last_episode(&session_id, 100)
                .await?;
            if messages.len() >= 5 {
                match self.create_episode(&session_id, &messages).await {
                    Ok(episode_id) => {
                        info!(session_id, episode_id, "Created episode for idle session");
                    }
                    Err(e) => {
                        warn!(session_id, error = %e, "Failed to create episode for idle session");
                    }
                }
            }
        }

        Ok(())
    }

    /// Create episodes for active sessions that have accumulated many events
    /// without an episode. This captures context from long-running sessions
    /// before messages rotate out of the context window.
    async fn create_episodes_for_active_long_sessions(&self) -> anyhow::Result<()> {
        // Find sessions with 20+ events since their last episode (or no episode
        // at all). These sessions risk losing context as the message window rotates.
        let session_ids: Vec<String> = sqlx::query_scalar(
            "SELECT ev.session_id
             FROM events ev
             LEFT JOIN (
                 SELECT session_id, MAX(end_time) AS latest_end
                 FROM episodes
                 GROUP BY session_id
             ) ep ON ep.session_id = ev.session_id
             WHERE ev.event_type IN ('user_message', 'assistant_response', 'tool_result')
               AND (ep.latest_end IS NULL OR ev.created_at > ep.latest_end)
             GROUP BY ev.session_id
             HAVING COUNT(ev.id) >= 20",
        )
        .fetch_all(&self.pool)
        .await?;
        let session_ids = self
            .history
            .exclude_memory_isolated_sessions(session_ids)
            .await?;

        for session_id in session_ids {
            if !self
                .history
                .session_since_last_episode_is_owner_only(&session_id)
                .await?
            {
                continue;
            }
            let messages = self
                .fetch_session_messages_since_last_episode(&session_id, 100)
                .await?;
            if messages.len() >= 10 {
                match self.create_episode(&session_id, &messages).await {
                    Ok(episode_id) => {
                        info!(
                            session_id,
                            episode_id, "Created mid-session episode for long-running session"
                        );
                    }
                    Err(e) => {
                        warn!(
                            session_id,
                            error = %e,
                            "Failed to create mid-session episode"
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Fetch messages after the last stable canonical event bound. Legacy
    /// episodes without bounds fall back to end_time.
    /// If no episode exists, returns all session messages up to `limit`.
    async fn fetch_session_messages_since_last_episode(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        let suppressed_tasks = self
            .history
            .memory_suppressed_tasks_for_session(session_id)
            .await?;
        let last_episode = sqlx::query(
            "SELECT end_event_id, end_time FROM episodes
             WHERE session_id = ?
             ORDER BY COALESCE(end_event_id, 0) DESC, end_time DESC LIMIT 1",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        let rows = if let Some(end_event_id) = last_episode
            .as_ref()
            .and_then(|row| row.try_get::<Option<i64>, _>("end_event_id").ok())
            .flatten()
        {
            sqlx::query(
                "SELECT id, session_id, event_type, data, created_at, task_id
                 FROM events
                 WHERE session_id = ?
                   AND event_type IN ('user_message', 'assistant_response', 'tool_result')
                   AND id > ?
                 ORDER BY id ASC
                 LIMIT ?",
            )
            .bind(session_id)
            .bind(end_event_id)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?
        } else if let Some(since) = last_episode
            .as_ref()
            .and_then(|row| row.try_get::<String, _>("end_time").ok())
        {
            sqlx::query(
                "SELECT id, session_id, event_type, data, created_at, task_id
                 FROM events
                 WHERE session_id = ?
                   AND event_type IN ('user_message', 'assistant_response', 'tool_result')
                   AND created_at > ?
                 ORDER BY created_at ASC
                 LIMIT ?",
            )
            .bind(session_id)
            .bind(&since)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT id, session_id, event_type, data, created_at, task_id
                 FROM events
                 WHERE session_id = ?
                   AND event_type IN ('user_message', 'assistant_response', 'tool_result')
                 ORDER BY created_at ASC
                 LIMIT ?",
            )
            .bind(session_id)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?
        };

        let mut messages = Vec::with_capacity(rows.len());
        for row in rows {
            if row
                .try_get::<Option<String>, _>("task_id")?
                .is_some_and(|task_id| suppressed_tasks.contains(&task_id))
            {
                continue;
            }
            let event_id: i64 = row.get("id");
            let row_session_id: String = row.get("session_id");
            let event_type: String = row.get("event_type");
            let created_str: String = row.get("created_at");
            let created_at = DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(turn) = crate::events::turn_from_event(
                event_id,
                &row_session_id,
                &event_type,
                &data,
                created_at,
            ) {
                let mut msg = turn.clone().into_message();
                if msg
                    .effective_annotations()
                    .contains(&MessageAnnotation::InternalContinuation)
                {
                    continue;
                }
                msg.importance = crate::memory::scoring::score_turn(&turn);
                messages.push(msg);
            }
        }

        Ok(messages)
    }

    /// Analyze recent activity for pattern detection and style updates.
    async fn analyze_recent_activity(&self) -> anyhow::Result<()> {
        // Fetch messages from the last 24 hours for analysis
        let one_day_ago = (Utc::now() - chrono::Duration::hours(24)).to_rfc3339();

        let recent_messages = self.fetch_messages_since(&one_day_ago, 500).await?;
        let owner_user_messages = self
            .fetch_owner_user_messages_since(&one_day_ago, 500)
            .await?;

        if recent_messages.is_empty() {
            return Ok(());
        }

        let Some(state) = &self.state else {
            return Ok(());
        };
        // Habitual tool sequences, from the calls in order.
        let tools: Vec<String> = recent_messages
            .iter()
            .filter_map(|msg| msg.tool_calls_json.as_deref())
            .filter_map(|json| serde_json::from_str::<Vec<serde_json::Value>>(json).ok())
            .flatten()
            .filter_map(|call| {
                call.get("name")
                    .and_then(|n| n.as_str())
                    .map(str::to_string)
            })
            .collect();
        let patterns =
            aidaemon_memory::record_tool_sequence_patterns(state.as_ref(), &tools).await?;
        if !patterns.is_empty() {
            info!(count = patterns.len(), "Detected behavior patterns");
        }

        // Communication verbosity from the owner's own messages.
        let owner_texts: Vec<&str> = owner_user_messages
            .iter()
            .filter(|m| m.role == "user")
            .filter_map(|m| m.content.as_deref())
            .collect();
        if aidaemon_memory::update_communication_profile(state.as_ref(), &owner_texts)
            .await?
            .is_some()
        {
            info!("Updated user communication profile");
        }

        Ok(())
    }

    /// Fetch messages for a specific session.
    #[allow(dead_code)]
    async fn fetch_session_messages(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        let suppressed_tasks = self
            .history
            .memory_suppressed_tasks_for_session(session_id)
            .await?;
        let rows = sqlx::query(
            "SELECT id, session_id, event_type, data, created_at, task_id
             FROM events
             WHERE session_id = ?
               AND event_type IN ('user_message', 'assistant_response', 'tool_result')
             ORDER BY created_at ASC
             LIMIT ?",
        )
        .bind(session_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut messages = Vec::with_capacity(rows.len());
        for row in rows {
            if row
                .try_get::<Option<String>, _>("task_id")?
                .is_some_and(|task_id| suppressed_tasks.contains(&task_id))
            {
                continue;
            }
            let event_id: i64 = row.get("id");
            let row_session_id: String = row.get("session_id");
            let event_type: String = row.get("event_type");
            let created_str: String = row.get("created_at");
            let created_at = DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(turn) = crate::events::turn_from_event(
                event_id,
                &row_session_id,
                &event_type,
                &data,
                created_at,
            ) {
                let mut msg = turn.clone().into_message();
                if msg
                    .effective_annotations()
                    .contains(&MessageAnnotation::InternalContinuation)
                {
                    continue;
                }
                msg.importance = crate::memory::scoring::score_turn(&turn);
                messages.push(msg);
            }
        }

        Ok(messages)
    }

    /// Fetch messages from all sessions since a given time.
    async fn fetch_messages_since(
        &self,
        since: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        let suppressed_tasks = self.history.all_memory_suppressed_tasks().await?;
        let rows = sqlx::query(
            "SELECT id, session_id, event_type, data, created_at, task_id
             FROM events
             WHERE created_at >= ?
               AND event_type IN ('user_message', 'assistant_response', 'tool_result')
             ORDER BY created_at DESC
             LIMIT ?",
        )
        .bind(since)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut messages = Vec::with_capacity(rows.len());
        let mut isolation_by_session: HashMap<String, bool> = HashMap::new();
        for row in rows {
            if row
                .try_get::<Option<String>, _>("task_id")?
                .is_some_and(|task_id| suppressed_tasks.contains(&task_id))
            {
                continue;
            }
            let event_id: i64 = row.get("id");
            let session_id: String = row.get("session_id");
            let memory_isolated = match isolation_by_session.get(&session_id) {
                Some(isolated) => *isolated,
                None => {
                    let isolated = self.history.session_is_memory_isolated(&session_id).await?;
                    isolation_by_session.insert(session_id.clone(), isolated);
                    isolated
                }
            };
            if memory_isolated {
                continue;
            }
            let event_type: String = row.get("event_type");
            let created_str: String = row.get("created_at");
            let created_at = DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(turn) = crate::events::turn_from_event(
                event_id,
                &session_id,
                &event_type,
                &data,
                created_at,
            ) {
                let mut msg = turn.clone().into_message();
                if msg
                    .effective_annotations()
                    .contains(&MessageAnnotation::InternalContinuation)
                {
                    continue;
                }
                msg.importance = crate::memory::scoring::score_turn(&turn);
                messages.push(msg);
            }
        }

        Ok(messages)
    }

    async fn fetch_owner_user_messages_since(
        &self,
        since: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        let suppressed_tasks = self.history.all_memory_suppressed_tasks().await?;
        let rows = sqlx::query(
            "SELECT id, session_id, event_type, data, created_at, task_id
             FROM events
             WHERE created_at >= ?
               AND event_type = 'user_message'
             ORDER BY created_at DESC
             LIMIT ?",
        )
        .bind(since)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut messages = Vec::with_capacity(rows.len());
        let mut isolation_by_session: HashMap<String, bool> = HashMap::new();
        for row in rows {
            if row
                .try_get::<Option<String>, _>("task_id")?
                .is_some_and(|task_id| suppressed_tasks.contains(&task_id))
            {
                continue;
            }
            let event_id: i64 = row.get("id");
            let session_id: String = row.get("session_id");
            let memory_isolated = match isolation_by_session.get(&session_id) {
                Some(isolated) => *isolated,
                None => {
                    let isolated = self.history.session_is_memory_isolated(&session_id).await?;
                    isolation_by_session.insert(session_id.clone(), isolated);
                    isolated
                }
            };
            if memory_isolated {
                continue;
            }
            let event_type: String = row.get("event_type");
            let created_str: String = row.get("created_at");
            let created_at = DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if !EventHistory::parse_event_user_role(&data)
                .is_some_and(UserRole::can_persist_owner_memory)
            {
                continue;
            }

            if let Some(turn) = crate::events::turn_from_event(
                event_id,
                &session_id,
                &event_type,
                &data,
                created_at,
            ) {
                let mut msg = turn.clone().into_message();
                if msg
                    .effective_annotations()
                    .contains(&MessageAnnotation::InternalContinuation)
                {
                    continue;
                }
                msg.importance = crate::memory::scoring::score_turn(&turn);
                messages.push(msg);
            }
        }

        Ok(messages)
    }
}

/// A session's messages as episode transcript entries. Internal
/// continuations are the daemon talking to itself, never owner speech.
fn transcript_entries(messages: &[Message]) -> Vec<aidaemon_memory::episodes::TranscriptEntry> {
    use aidaemon_memory::episodes::TranscriptEntry;
    use aidaemon_memory::history::TurnRole;
    messages
        .iter()
        .filter(|msg| {
            !msg.effective_annotations()
                .contains(&MessageAnnotation::InternalContinuation)
        })
        .filter_map(|msg| {
            let role = match msg.role.as_str() {
                "user" => TurnRole::User,
                "assistant" => TurnRole::Assistant,
                "tool" => TurnRole::Tool,
                _ => return None,
            };
            let tool_calls = msg
                .tool_calls_json
                .as_deref()
                .and_then(|json| serde_json::from_str::<Vec<serde_json::Value>>(json).ok())
                .into_iter()
                .flatten()
                .filter_map(|call| {
                    call.get("name")
                        .and_then(|n| n.as_str())
                        .map(str::to_string)
                })
                .collect();
            Some(TranscriptEntry {
                role,
                content: msg.content.clone(),
                tool_calls,
                tool_name: msg.tool_name.clone(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::ProviderKind;
    use crate::llm_runtime::SharedLlmRuntime;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::testing::MockProvider;
    use crate::traits::store_prelude::*;
    use crate::traits::MessageAnnotation;
    use chrono::Utc;
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn clean_transcript_excludes_internal_continuation_as_owner_speech() {
        let messages = vec![
            Message {
                content: Some("Find the synthetic product specifications.".to_string()),
                ..Message::new_runtime("user-1", "session-1", "user")
            },
            Message {
                content: Some("Worker output from an internal retry.".to_string()),
                annotations: vec![MessageAnnotation::InternalContinuation],
                ..Message::new_runtime("continuation-1", "session-1", "user")
            },
        ];

        let transcript =
            aidaemon_memory::episodes::clean_transcript(&transcript_entries(&messages));
        assert!(transcript.contains("Find the synthetic product specifications."));
        assert!(!transcript.contains("Worker output from an internal retry."));
    }

    async fn setup_memory_manager_with_responses(
        responses: Vec<crate::traits::ProviderResponse>,
    ) -> (
        Arc<SqliteStateStore>,
        Arc<MockProvider>,
        MemoryManager,
        tempfile::NamedTempFile,
    ) {
        let db_file = tempfile::NamedTempFile::new().expect("temp db file");
        let embedding_service = Arc::new(EmbeddingService::new().expect("embedding service"));
        let store = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().expect("db path"),
                100,
                None,
                embedding_service.clone(),
            )
            .await
            .expect("state store"),
        );
        let provider = Arc::new(MockProvider::with_responses(responses));
        let llm_runtime = SharedLlmRuntime::new(
            provider.clone(),
            None,
            ProviderKind::OpenaiCompatible,
            "mock".to_string(),
        );
        let manager = MemoryManager::new(
            store.pool(),
            embedding_service,
            llm_runtime,
            Duration::from_secs(60),
            None,
        )
        .with_state(store.clone());
        (store, provider, manager, db_file)
    }

    async fn insert_owner_session_events(
        pool: &SqlitePool,
        session_id: &str,
        count: usize,
        age: chrono::Duration,
        execution_origin: Option<&str>,
        content_sentinel: &str,
    ) {
        let base = Utc::now() - age;
        for index in 0..count {
            let mut data = json!({
                "content": format!("{content_sentinel} message {index}"),
                "message_id": format!("{session_id}-message-{index}"),
                "has_attachments": false,
                "user_role": "owner",
                "channel_visibility": "private"
            });
            if index == 0 {
                if let Some(origin) = execution_origin {
                    data["execution_origin"] = json!(origin);
                }
            }
            sqlx::query(
                "INSERT INTO events (session_id, event_type, data, created_at)
                 VALUES (?, 'user_message', ?, ?)",
            )
            .bind(session_id)
            .bind(data.to_string())
            .bind((base + chrono::Duration::seconds(index as i64)).to_rfc3339())
            .execute(pool)
            .await
            .expect("insert owner session event");
        }
    }

    #[tokio::test]
    async fn mandate_sessions_are_excluded_from_fact_consolidation_by_durable_origin() {
        let response = MockProvider::text_response(
            r#"[{"category":"preference","key":"editor","value":"Helix","privacy":"global"}]"#,
        );
        let (store, provider, manager, _db_file) =
            setup_memory_manager_with_responses(vec![response]).await;
        insert_owner_session_events(
            &store.pool(),
            "ordinary-owner-fact-session",
            1,
            chrono::Duration::hours(2),
            None,
            "Important: remember that I prefer Helix as my editor.",
        )
        .await;
        insert_owner_session_events(
            &store.pool(),
            "owner-looking-mandate-fact-session",
            1,
            chrono::Duration::hours(2),
            Some("mandate"),
            "MANDATE_PRIVATE_SENTINEL must never become global memory.",
        )
        .await;

        manager
            .consolidate_memories()
            .await
            .expect("fact consolidation");

        assert_eq!(provider.call_count().await, 1);
        let calls = provider.call_log.lock().await;
        let prompt = serde_json::to_string(&calls[0].messages).expect("serialize call");
        assert!(prompt.contains("prefer Helix"));
        assert!(!prompt.contains("MANDATE_PRIVATE_SENTINEL"));
        drop(calls);

        let fact_value: String = sqlx::query_scalar(
            "SELECT value FROM facts
             WHERE category = 'preference' AND key = 'editor' AND superseded_at IS NULL",
        )
        .fetch_one(&store.pool())
        .await
        .expect("ordinary fact learned");
        assert_eq!(fact_value, "Helix");
    }

    #[tokio::test]
    async fn mandate_sessions_are_excluded_from_idle_episode_candidates() {
        let response = MockProvider::text_response(
            r#"{
                "summary":"Ordinary owner idle session",
                "topics":["ordinary"],
                "emotional_tone":"productive",
                "outcome":"resolved",
                "goals_mentioned":[],
                "emotional_intensity":0.2
            }"#,
        );
        let (store, provider, manager, _db_file) =
            setup_memory_manager_with_responses(vec![response]).await;
        insert_owner_session_events(
            &store.pool(),
            "ordinary-owner-idle-session",
            5,
            chrono::Duration::minutes(45),
            None,
            "ORDINARY_IDLE_SENTINEL",
        )
        .await;
        insert_owner_session_events(
            &store.pool(),
            "owner-looking-mandate-idle-session",
            5,
            chrono::Duration::minutes(45),
            Some("mandate"),
            "MANDATE_IDLE_SENTINEL",
        )
        .await;

        manager
            .create_episodes_for_idle_sessions()
            .await
            .expect("idle episode pass");

        assert_eq!(provider.call_count().await, 1);
        let ordinary_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM episodes WHERE session_id = ?")
                .bind("ordinary-owner-idle-session")
                .fetch_one(&store.pool())
                .await
                .expect("ordinary episode count");
        let mandate_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM episodes WHERE session_id = ?")
                .bind("owner-looking-mandate-idle-session")
                .fetch_one(&store.pool())
                .await
                .expect("mandate episode count");
        assert_eq!(ordinary_count, 1);
        assert_eq!(mandate_count, 0);
    }

    #[tokio::test]
    async fn mandate_sessions_are_excluded_from_long_episode_candidates() {
        let response = MockProvider::text_response(
            r#"{
                "summary":"Ordinary owner long session",
                "topics":["ordinary"],
                "emotional_tone":"productive",
                "outcome":"ongoing",
                "goals_mentioned":[],
                "emotional_intensity":0.2
            }"#,
        );
        let (store, provider, manager, _db_file) =
            setup_memory_manager_with_responses(vec![response]).await;
        insert_owner_session_events(
            &store.pool(),
            "ordinary-owner-long-session",
            20,
            chrono::Duration::minutes(10),
            None,
            "ORDINARY_LONG_SENTINEL",
        )
        .await;
        insert_owner_session_events(
            &store.pool(),
            "owner-looking-mandate-long-session",
            20,
            chrono::Duration::minutes(10),
            Some("mandate"),
            "MANDATE_LONG_SENTINEL",
        )
        .await;

        manager
            .create_episodes_for_active_long_sessions()
            .await
            .expect("long episode pass");

        assert_eq!(provider.call_count().await, 1);
        let ordinary_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM episodes WHERE session_id = ?")
                .bind("ordinary-owner-long-session")
                .fetch_one(&store.pool())
                .await
                .expect("ordinary episode count");
        let mandate_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM episodes WHERE session_id = ?")
                .bind("owner-looking-mandate-long-session")
                .fetch_one(&store.pool())
                .await
                .expect("mandate episode count");
        assert_eq!(ordinary_count, 1);
        assert_eq!(mandate_count, 0);
    }

    #[tokio::test]
    async fn test_consolidation_skips_public_external_sessions() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let store = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service.clone(),
            )
            .await
            .unwrap(),
        );

        // Ensure events table exists for visibility lookup.
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consolidated_at TEXT,
                task_id TEXT,
                tool_name TEXT
            )
            "#,
        )
        .execute(&store.pool())
        .await
        .unwrap();

        let session_id = "pubext_session_consolidation";
        let msg_id = uuid::Uuid::new_v4().to_string();
        // Mark the session as PublicExternal via the user_message event metadata.
        let data = serde_json::json!({
            "content": "hello world",
            "message_id": msg_id,
            "has_attachments": false,
            "user_role": "owner",
            "channel_visibility": "public_external",
            "channel_id": "twitter:ext_999",
            "platform": "twitter",
        });
        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind("user_message")
        .bind(data.to_string())
        .bind((Utc::now() - chrono::Duration::hours(2)).to_rfc3339())
        .execute(&store.pool())
        .await
        .unwrap();

        let provider = Arc::new(MockProvider::new());
        let llm_runtime = SharedLlmRuntime::new(
            provider.clone(),
            None,
            ProviderKind::OpenaiCompatible,
            "mock".to_string(),
        );
        let mgr = MemoryManager::new(
            store.pool(),
            embedding_service,
            llm_runtime,
            Duration::from_secs(60),
            None,
        )
        .with_state(store.clone());

        mgr.consolidate_memories().await.unwrap();

        // Provider should not be called for PublicExternal sessions.
        assert_eq!(provider.call_count().await, 0);

        // Cursor should advance so we don't repeatedly revisit skipped sessions.
        let cursor = store
            .get_setting(&format!("memory_fact_consolidation_cursor:{}", session_id))
            .await
            .unwrap();
        assert!(cursor.is_some());

        // No facts should be learned.
        let cnt: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM facts")
            .fetch_one(&store.pool())
            .await
            .unwrap();
        assert_eq!(cnt, 0);
    }

    #[tokio::test]
    async fn test_consolidation_skips_non_owner_sessions() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let store = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service.clone(),
            )
            .await
            .unwrap(),
        );

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consolidated_at TEXT,
                task_id TEXT,
                tool_name TEXT
            )
            "#,
        )
        .execute(&store.pool())
        .await
        .unwrap();

        let session_id = "slack:D_non_owner";
        let user_data = serde_json::json!({
            "content": "My name is Alice",
            "message_id": "msg-1",
            "has_attachments": false,
            "user_role": "public",
            "channel_visibility": "private",
            "channel_id": "slack:D_non_owner",
            "platform": "slack",
        });
        let assistant_data = serde_json::json!({
            "message_id": "msg-2",
            "content": "Nice to meet you."
        });

        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind("user_message")
        .bind(user_data.to_string())
        .bind((Utc::now() - chrono::Duration::hours(2)).to_rfc3339())
        .execute(&store.pool())
        .await
        .unwrap();

        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind("assistant_response")
        .bind(assistant_data.to_string())
        .bind((Utc::now() - chrono::Duration::hours(2) + chrono::Duration::seconds(5)).to_rfc3339())
        .execute(&store.pool())
        .await
        .unwrap();

        let provider = Arc::new(MockProvider::new());
        let llm_runtime = SharedLlmRuntime::new(
            provider.clone(),
            None,
            ProviderKind::OpenaiCompatible,
            "mock".to_string(),
        );
        let mgr = MemoryManager::new(
            store.pool(),
            embedding_service,
            llm_runtime,
            Duration::from_secs(60),
            None,
        )
        .with_state(store.clone());

        mgr.consolidate_memories().await.unwrap();

        assert_eq!(provider.call_count().await, 0);

        let cursor = store
            .get_setting(&format!("memory_fact_consolidation_cursor:{}", session_id))
            .await
            .unwrap();
        assert!(cursor.is_some());

        let cnt: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM facts")
            .fetch_one(&store.pool())
            .await
            .unwrap();
        assert_eq!(cnt, 0);
    }

    #[tokio::test]
    async fn test_fetch_owner_user_messages_since_filters_non_owner_messages() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let store = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service.clone(),
            )
            .await
            .unwrap(),
        );

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consolidated_at TEXT,
                task_id TEXT,
                tool_name TEXT
            )
            "#,
        )
        .execute(&store.pool())
        .await
        .unwrap();

        let owner_data = serde_json::json!({
            "content": "Please keep responses short.",
            "message_id": "owner-msg",
            "has_attachments": false,
            "user_role": "owner"
        });
        let public_data = serde_json::json!({
            "content": "I like pirate responses.",
            "message_id": "public-msg",
            "has_attachments": false,
            "user_role": "public"
        });

        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind("owner-session")
        .bind("user_message")
        .bind(owner_data.to_string())
        .bind((Utc::now() - chrono::Duration::minutes(10)).to_rfc3339())
        .execute(&store.pool())
        .await
        .unwrap();

        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind("public-session")
        .bind("user_message")
        .bind(public_data.to_string())
        .bind((Utc::now() - chrono::Duration::minutes(5)).to_rfc3339())
        .execute(&store.pool())
        .await
        .unwrap();

        let provider = Arc::new(MockProvider::new());
        let llm_runtime = SharedLlmRuntime::new(
            provider,
            None,
            ProviderKind::OpenaiCompatible,
            "mock".to_string(),
        );
        let mgr = MemoryManager::new(
            store.pool(),
            embedding_service,
            llm_runtime,
            Duration::from_secs(60),
            None,
        )
        .with_state(store);

        let since = (Utc::now() - chrono::Duration::hours(1)).to_rfc3339();
        let messages = mgr
            .fetch_owner_user_messages_since(&since, 10)
            .await
            .unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].content.as_deref(),
            Some("Please keep responses short.")
        );
    }

    #[tokio::test]
    async fn test_fetch_session_messages_since_last_episode_supports_legacy_annotations() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let store = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service.clone(),
            )
            .await
            .unwrap(),
        );

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consolidated_at TEXT,
                task_id TEXT,
                tool_name TEXT
            )
            "#,
        )
        .execute(&store.pool())
        .await
        .unwrap();

        let session_id = "legacy_annotations_session";
        let legacy_data = json!({
            "message_id": "legacy-msg",
            "tool_call_id": "call-legacy",
            "name": "system",
            "result": "[SYSTEM] Legacy internal note"
        });
        let annotated_data = json!({
            "message_id": "annotated-msg",
            "tool_call_id": "call-annotated",
            "name": "terminal",
            "result": "cargo test\n\n[SYSTEM] Do not retry.",
            "annotations": [{"type": "appended_system_notice"}]
        });

        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind("tool_result")
        .bind(legacy_data.to_string())
        .bind((Utc::now() - chrono::Duration::minutes(2)).to_rfc3339())
        .execute(&store.pool())
        .await
        .unwrap();

        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind("tool_result")
        .bind(annotated_data.to_string())
        .bind((Utc::now() - chrono::Duration::minutes(1)).to_rfc3339())
        .execute(&store.pool())
        .await
        .unwrap();

        let provider = Arc::new(MockProvider::new());
        let llm_runtime = SharedLlmRuntime::new(
            provider,
            None,
            ProviderKind::OpenaiCompatible,
            "mock".to_string(),
        );
        let mgr = MemoryManager::new(
            store.pool(),
            embedding_service,
            llm_runtime,
            Duration::from_secs(60),
            None,
        )
        .with_state(store);

        let messages = mgr
            .fetch_session_messages_since_last_episode(session_id, 10)
            .await
            .unwrap();

        assert_eq!(messages.len(), 2);

        let legacy = &messages[0];
        assert!(legacy.annotations.is_empty());
        assert_eq!(
            legacy.effective_annotations().as_ref(),
            [MessageAnnotation::EntireSystemNotice]
        );
        assert!(legacy.primary_content().is_none());

        let annotated = &messages[1];
        assert_eq!(
            annotated.annotations,
            vec![MessageAnnotation::AppendedSystemNotice]
        );
        assert_eq!(annotated.primary_content().as_deref(), Some("cargo test"));
    }
}
