use crate::events::{Consolidator, EventStore};
use crate::llm_runtime::SharedLlmRuntime;
use crate::memory::binary::encode_embedding;
use crate::memory::embedding_index::{
    content_hash, EmbeddingIndex, EmbeddingItem, MemoryOwner, SqliteEmbeddingIndex,
};
use crate::memory::embeddings::EmbeddingService;
use crate::memory::embeddings::EMBEDDING_MODEL_ID;
use crate::memory::history_source::EventHistory;
use crate::memory::scoring::calculate_episode_importance;
use crate::traits::{BehaviorPattern, Message, MessageAnnotation, StateStore, UserProfile};
use crate::types::UserRole;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::json;
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

    /// Generate a clean transcript from messages, compressing tool calls.
    pub fn get_clean_transcript(messages: &[Message]) -> String {
        let mut transcript = String::new();

        for msg in messages {
            if msg
                .effective_annotations()
                .contains(&MessageAnnotation::InternalContinuation)
            {
                continue;
            }
            match msg.role.as_str() {
                "user" => {
                    if let Some(content) = &msg.content {
                        transcript.push_str(&format!("User: {}\n\n", content));
                    }
                }
                "assistant" => {
                    // Check for tool calls
                    if let Some(tc_json) = &msg.tool_calls_json {
                        if let Ok(tool_calls) =
                            serde_json::from_str::<Vec<serde_json::Value>>(tc_json)
                        {
                            for tc in &tool_calls {
                                if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
                                    transcript.push_str(&format!("[Action: {}]\n", name));
                                }
                            }
                        }
                    }
                    // Include text content if present
                    if let Some(content) = &msg.content {
                        if !content.trim().is_empty() {
                            let truncated = crate::utils::truncate_str(content, 500);
                            transcript.push_str(&format!("Assistant: {}\n\n", truncated));
                        }
                    }
                }
                "tool" => {
                    // Compress tool outputs
                    if let Some(content) = &msg.content {
                        if let Some(name) = &msg.tool_name {
                            let summary = if content.len() > 200 {
                                format!("[{} output: {} chars]", name, content.len())
                            } else {
                                format!(
                                    "[{}: {}]",
                                    name,
                                    content.chars().take(100).collect::<String>()
                                )
                            };
                            transcript.push_str(&format!("{}\n", summary));
                        }
                    }
                }
                _ => {}
            }
        }

        transcript
    }

    /// Create an episode from a session's messages.
    pub async fn create_episode(
        &self,
        session_id: &str,
        messages: &[Message],
    ) -> anyhow::Result<i64> {
        anyhow::ensure!(
            !self.history.session_is_memory_isolated(session_id).await?,
            "Memory-isolated worker sessions cannot create global episodes"
        );
        if messages.is_empty() {
            return Err(anyhow::anyhow!("No messages to create episode from"));
        }

        let clean_transcript = Self::get_clean_transcript(messages);

        // Call LLM to analyze the session
        let analysis = self.analyze_session(&clean_transcript).await?;

        // Calculate importance
        let has_errors = messages.iter().any(|m| {
            m.content.as_ref().is_some_and(|c| {
                c.to_lowercase().contains("error") || c.to_lowercase().contains("failed")
            })
        });
        let has_decisions = messages.iter().any(|m| {
            m.content.as_ref().is_some_and(|c| {
                let lower = c.to_lowercase();
                lower.contains("decided")
                    || lower.contains("let's go with")
                    || lower.contains("i'll use")
            })
        });
        let has_goals = !analysis.goals_mentioned.is_empty();

        let importance = calculate_episode_importance(
            messages.len() as i32,
            has_errors,
            has_decisions,
            has_goals,
            analysis.emotional_intensity,
        );

        // Get time bounds
        let start_time = messages
            .first()
            .map(|m| m.created_at)
            .unwrap_or_else(Utc::now);
        let end_time = messages
            .last()
            .map(|m| m.created_at)
            .unwrap_or_else(Utc::now);
        let now = Utc::now();
        let bounds = sqlx::query(
            "SELECT MIN(id) AS start_event_id, MAX(id) AS end_event_id
             FROM events
             WHERE session_id = ?
               AND event_type IN ('user_message', 'assistant_response')
               AND created_at >= ? AND created_at <= ?",
        )
        .bind(session_id)
        .bind(start_time.to_rfc3339())
        .bind(end_time.to_rfc3339())
        .fetch_one(&self.pool)
        .await?;
        let start_event_id: Option<i64> = bounds.try_get("start_event_id").unwrap_or(None);
        let end_event_id: Option<i64> = bounds.try_get("end_event_id").unwrap_or(None);
        let channel_id = crate::memory::derive_channel_id_from_session(session_id);

        // Insert episode. Multiple episodes per session are allowed (mid-session
        // episodes for long-running conversations + final idle-session episode).
        let topics_json = serde_json::to_string(&analysis.topics).ok();
        let result = sqlx::query(
            "INSERT INTO episodes
                (session_id, summary, topics, emotional_tone, outcome, importance,
                 recall_count, message_count, start_time, end_time, created_at,
                 channel_id, start_event_id, end_event_id)
             VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind(&analysis.summary)
        .bind(&topics_json)
        .bind(&analysis.emotional_tone)
        .bind(&analysis.outcome)
        .bind(importance)
        .bind(messages.len() as i32)
        .bind(start_time.to_rfc3339())
        .bind(end_time.to_rfc3339())
        .bind(now.to_rfc3339())
        .bind(channel_id)
        .bind(start_event_id)
        .bind(end_event_id)
        .execute(&self.pool)
        .await?;

        if result.rows_affected() == 0 {
            // Shouldn't happen now that unique constraint is removed, but handle gracefully.
            info!(session_id, "Episode insert returned 0 rows affected");
            let existing: Option<(i64,)> = sqlx::query_as(
                "SELECT id FROM episodes WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            )
            .bind(session_id)
            .fetch_optional(&self.pool)
            .await?;
            let episode_id = existing.map(|(id,)| id).unwrap_or(0);
            return Ok(episode_id);
        }

        let episode_id = result.last_insert_rowid();

        // Generate and store embedding for the summary
        if let Ok(embedding) = self.embedding_service.embed(analysis.summary.clone()).await {
            let blob = encode_embedding(&embedding);
            sqlx::query("UPDATE episodes SET embedding = ? WHERE id = ?")
                .bind(blob)
                .bind(episode_id)
                .execute(&self.pool)
                .await?;
            // Keep the canonical span and replaceable embedding index in sync
            // with the legacy episode column during the rolling migration.
            if let Some(state) = &self.state {
                let _ = state.project_episode_memory(episode_id).await;
            }
        }

        // Extract goals
        for goal_text in &analysis.goals_mentioned {
            if let Err(e) = self.extract_goal(goal_text, episode_id, session_id).await {
                warn!("Failed to extract goal '{}': {}", goal_text, e);
            }
        }

        info!(
            session_id,
            episode_id,
            topics = ?analysis.topics,
            emotional_tone = ?analysis.emotional_tone,
            outcome = ?analysis.outcome,
            goals = analysis.goals_mentioned.len(),
            "Created episode"
        );

        Ok(episode_id)
    }

    /// Analyze a session transcript using LLM.
    async fn analyze_session(&self, transcript: &str) -> anyhow::Result<SessionAnalysis> {
        let system_prompt = r#"You are a memory system analyzing a conversation. Extract:
1. A brief summary (1-2 sentences)
2. Main topics discussed (list of keywords)
3. Emotional tone (one of: neutral, productive, frustrated, exploratory, celebratory)
4. Outcome (one of: resolved, ongoing, abandoned, learning)
5. Possible enduring personal goals the user explicitly stated for themselves

For goals_mentioned, return an empty list for ordinary one-off requests, implementation
steps, delegated work, quoted examples, prohibitions, review checklists, or specialist/system
instructions. Only include a goal when the user's own words describe an intention that should
remain relevant beyond this conversation.

Respond with ONLY valid JSON:
{
  "summary": "...",
  "topics": ["topic1", "topic2"],
  "emotional_tone": "...",
  "outcome": "...",
  "goals_mentioned": ["goal1", "goal2"],
  "emotional_intensity": 0.5
}

emotional_intensity is 0.0-1.0 scale (0=calm, 1=highly emotional)"#;

        let llm_messages = vec![
            json!({"role": "system", "content": system_prompt}),
            json!({"role": "user", "content": format!("Analyze this conversation:\n\n{}", transcript)}),
        ];

        let runtime_snapshot = self.llm_runtime.snapshot();
        // An episode is a short summary of a finished session: fast-tier work.
        // Running it on the primary model spent the most expensive tokens on
        // the least read memory.
        let model = runtime_snapshot
            .router()
            .map(|router| router.select(crate::router::Tier::Fast).to_string())
            .unwrap_or_else(|| runtime_snapshot.primary_model());
        let call_start = std::time::Instant::now();
        let response = runtime_snapshot
            .background_chat(&model, &llm_messages, &[])
            .await?;

        if let (Some(event_store), Some(state)) = (&self.event_store, &self.state) {
            crate::events::record_background_model_call_telemetry(
                event_store.clone(),
                state.as_ref(),
                "background:episode_creation",
                "episode_creation",
                &model,
                &response,
                call_start.elapsed(),
            )
            .await;
        }

        let text = response
            .content
            .ok_or_else(|| anyhow::anyhow!("Empty response from LLM"))?;

        // Parse JSON response
        let trimmed = text.trim();
        let json_str = if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                &trimmed[start..=end]
            } else {
                trimmed
            }
        } else {
            trimmed
        };

        let analysis: SessionAnalysis =
            serde_json::from_str(json_str).unwrap_or_else(|_| SessionAnalysis {
                summary: "Session with various tasks".to_string(),
                topics: vec!["general".to_string()],
                emotional_tone: "neutral".to_string(),
                outcome: "ongoing".to_string(),
                goals_mentioned: vec![],
                emotional_intensity: 0.3,
            });

        Ok(analysis)
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

    /// Analyze user's communication style from recent sessions.
    pub async fn analyze_communication_style(
        &self,
        recent_messages: &[Message],
    ) -> anyhow::Result<UserProfile> {
        let user_messages: Vec<&Message> = recent_messages
            .iter()
            .filter(|m| m.role == "user")
            .collect();

        if user_messages.is_empty() {
            return self.get_or_create_profile().await;
        }

        // Analyze verbosity (average message length)
        let avg_length: f32 = user_messages
            .iter()
            .filter_map(|m| m.content.as_ref())
            .map(|c| c.len() as f32)
            .sum::<f32>()
            / user_messages.len() as f32;

        let verbosity = if avg_length > 200.0 {
            "detailed"
        } else if avg_length > 50.0 {
            "medium"
        } else {
            "brief"
        };

        // Analyze tone (check for politeness markers)
        let politeness_count: usize = user_messages
            .iter()
            .filter_map(|m| m.content.as_ref())
            .filter(|c| {
                let lower = c.to_lowercase();
                lower.contains("please") || lower.contains("thank") || lower.contains("could you")
            })
            .count();

        let tone = if politeness_count > user_messages.len() / 2 {
            "formal"
        } else {
            "casual"
        };

        // Analyze explanation preference
        let question_count: usize = user_messages
            .iter()
            .filter_map(|m| m.content.as_ref())
            .filter(|c| {
                let lower = c.to_lowercase();
                lower.contains("why") || lower.contains("explain") || lower.contains("how does")
            })
            .count();

        let prefers_explanations = question_count > user_messages.len() / 4;

        // Get or create profile
        let mut profile = self.get_or_create_profile().await?;
        profile.verbosity_preference = verbosity.to_string();
        profile.tone_preference = tone.to_string();
        profile.prefers_explanations = prefers_explanations;
        profile.updated_at = Utc::now();

        // Update in database
        self.update_profile(&profile).await?;

        Ok(profile)
    }

    async fn get_or_create_profile(&self) -> anyhow::Result<UserProfile> {
        let row = sqlx::query(
            "SELECT id, verbosity_preference, explanation_depth, tone_preference, emoji_preference, typical_session_length, active_hours, common_workflows, asks_before_acting, prefers_explanations, likes_suggestions, updated_at
             FROM user_profile LIMIT 1"
        )
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let active_hours_json: Option<String> = row.get("active_hours");
            let workflows_json: Option<String> = row.get("common_workflows");
            let updated_str: String = row.get("updated_at");

            Ok(UserProfile {
                id: row.get("id"),
                verbosity_preference: row.get("verbosity_preference"),
                explanation_depth: row.get("explanation_depth"),
                tone_preference: row.get("tone_preference"),
                emoji_preference: row.get("emoji_preference"),
                typical_session_length: row.get("typical_session_length"),
                active_hours: active_hours_json.and_then(|j| serde_json::from_str(&j).ok()),
                common_workflows: workflows_json.and_then(|j| serde_json::from_str(&j).ok()),
                asks_before_acting: row.get::<i32, _>("asks_before_acting") == 1,
                prefers_explanations: row.get::<i32, _>("prefers_explanations") == 1,
                likes_suggestions: row.get::<i32, _>("likes_suggestions") == 1,
                updated_at: DateTime::parse_from_rfc3339(&updated_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
        } else {
            let now = Utc::now().to_rfc3339();
            sqlx::query("INSERT INTO user_profile (updated_at) VALUES (?)")
                .bind(&now)
                .execute(&self.pool)
                .await?;

            Ok(UserProfile {
                id: 1,
                verbosity_preference: "medium".to_string(),
                explanation_depth: "moderate".to_string(),
                tone_preference: "neutral".to_string(),
                emoji_preference: "none".to_string(),
                typical_session_length: None,
                active_hours: None,
                common_workflows: None,
                asks_before_acting: false,
                prefers_explanations: true,
                likes_suggestions: false,
                updated_at: Utc::now(),
            })
        }
    }

    async fn update_profile(&self, profile: &UserProfile) -> anyhow::Result<()> {
        let active_hours_json = profile
            .active_hours
            .as_ref()
            .map(|h| serde_json::to_string(h).unwrap_or_default());
        let workflows_json = profile
            .common_workflows
            .as_ref()
            .map(|w| serde_json::to_string(w).unwrap_or_default());
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            "UPDATE user_profile SET verbosity_preference = ?, explanation_depth = ?, tone_preference = ?, emoji_preference = ?, typical_session_length = ?, active_hours = ?, common_workflows = ?, asks_before_acting = ?, prefers_explanations = ?, likes_suggestions = ?, updated_at = ? WHERE id = ?"
        )
        .bind(&profile.verbosity_preference)
        .bind(&profile.explanation_depth)
        .bind(&profile.tone_preference)
        .bind(&profile.emoji_preference)
        .bind(profile.typical_session_length)
        .bind(&active_hours_json)
        .bind(&workflows_json)
        .bind(profile.asks_before_acting as i32)
        .bind(profile.prefers_explanations as i32)
        .bind(profile.likes_suggestions as i32)
        .bind(&now)
        .bind(profile.id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    // ==================== Behavior Pattern Detection ====================

    /// Detect behavior patterns from recent tool usage.
    pub async fn detect_patterns(
        &self,
        recent_messages: &[Message],
    ) -> anyhow::Result<Vec<BehaviorPattern>> {
        // Extract tool call sequences
        let mut tool_sequences: Vec<(String, DateTime<Utc>)> = Vec::new();

        for msg in recent_messages {
            if let Some(tc_json) = &msg.tool_calls_json {
                if let Ok(tool_calls) = serde_json::from_str::<Vec<serde_json::Value>>(tc_json) {
                    for tc in tool_calls {
                        if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
                            tool_sequences.push((name.to_string(), msg.created_at));
                        }
                    }
                }
            }
        }

        if tool_sequences.len() < 3 {
            return Ok(vec![]);
        }

        // Detect sequential patterns (A → B happens frequently)
        let mut pair_counts: HashMap<(String, String), i32> = HashMap::new();
        for window in tool_sequences.windows(2) {
            let pair = (window[0].0.clone(), window[1].0.clone());
            *pair_counts.entry(pair).or_insert(0) += 1;
        }

        let mut patterns = Vec::new();
        let now = Utc::now();

        for ((tool_a, tool_b), count) in pair_counts {
            if count >= 3 {
                let confidence = (count as f32 / tool_sequences.len() as f32).min(0.9);

                // Check if pattern already exists
                let existing = sqlx::query(
                    "SELECT id FROM behavior_patterns WHERE pattern_type = 'sequence' AND trigger_context = ? AND action = ?"
                )
                .bind(&tool_a)
                .bind(&tool_b)
                .fetch_optional(&self.pool)
                .await?;

                if let Some(row) = existing {
                    let id: i64 = row.get("id");
                    sqlx::query(
                        "UPDATE behavior_patterns SET occurrence_count = occurrence_count + ?, confidence = MIN(0.95, confidence + ?), last_seen_at = ? WHERE id = ?"
                    )
                    .bind(count)
                    .bind(confidence * 0.1)
                    .bind(now.to_rfc3339())
                    .bind(id)
                    .execute(&self.pool)
                    .await?;
                } else {
                    sqlx::query(
                        "INSERT INTO behavior_patterns (pattern_type, description, trigger_context, action, confidence, occurrence_count, last_seen_at, created_at)
                         VALUES ('sequence', ?, ?, ?, ?, ?, ?, ?)"
                    )
                    .bind(format!("After {} often use {}", tool_a, tool_b))
                    .bind(&tool_a)
                    .bind(&tool_b)
                    .bind(confidence)
                    .bind(count)
                    .bind(now.to_rfc3339())
                    .bind(now.to_rfc3339())
                    .execute(&self.pool)
                    .await?;
                }

                patterns.push(BehaviorPattern {
                    id: 0,
                    pattern_type: "sequence".to_string(),
                    description: format!("After {} often use {}", tool_a, tool_b),
                    trigger_context: Some(tool_a),
                    action: Some(tool_b),
                    confidence,
                    occurrence_count: count,
                    last_seen_at: Some(now),
                    created_at: now,
                });
            }
        }

        Ok(patterns)
    }

    // ==================== Memory Decay ====================

    /// Apply decay to old memories (reduce recall counts over time).
    pub async fn decay_memories(&self) -> anyhow::Result<()> {
        // Merge paraphrased duplicates first so decay and salience see one
        // row per piece of knowledge.
        if let Some(state) = &self.state {
            if let Err(error) = state.fold_restated_facts().await {
                warn!(%error, "Failed to fold restated facts");
            }
        }

        // Facts not shown to the model for two weeks lose a tenth of their
        // salience per day (at least one point). A linear one-point decay let
        // a fact recalled a hundred times stay pinned for months after it
        // stopped mattering.
        let fourteen_days_ago = (Utc::now() - chrono::Duration::days(14)).to_rfc3339();
        sqlx::query(
            "UPDATE facts SET recall_count = MAX(0, recall_count - MAX(1, recall_count / 10))
             WHERE recall_count > 0 AND (last_recalled_at IS NULL OR last_recalled_at < ?)",
        )
        .bind(&fourteen_days_ago)
        .execute(&self.pool)
        .await?;

        let thirty_days_ago = (Utc::now() - chrono::Duration::days(30)).to_rfc3339();

        // Decay episodes
        sqlx::query(
            "UPDATE episodes SET recall_count = MAX(0, recall_count - 1)
             WHERE recall_count > 0 AND (last_recalled_at IS NULL OR last_recalled_at < ?)",
        )
        .bind(&thirty_days_ago)
        .execute(&self.pool)
        .await?;

        // Decay behavior pattern confidence
        sqlx::query(
            "UPDATE behavior_patterns SET confidence = MAX(0.1, confidence - 0.05)
             WHERE confidence > 0.1 AND (last_seen_at IS NULL OR last_seen_at < ?)",
        )
        .bind(&thirty_days_ago)
        .execute(&self.pool)
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

        // Detect behavior patterns
        let patterns = self.detect_patterns(&recent_messages).await?;
        if !patterns.is_empty() {
            info!(count = patterns.len(), "Detected behavior patterns");
        }

        // Update communication style profile
        if !owner_user_messages.is_empty() {
            let _profile = self
                .analyze_communication_style(&owner_user_messages)
                .await?;
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

/// Derive a channel_id from a session_id string.
/// Session IDs follow patterns like:
///   - "slack:CHANNEL_ID:thread_ts" or "slack:CHANNEL_ID"
///   - "bot:telegram:CHAT_ID" or "telegram:CHAT_ID" or just "CHAT_ID"
///   - "discord:ch:ID" or "discord:dm:ID"
#[derive(Debug, Deserialize)]
struct SessionAnalysis {
    #[serde(
        default = "default_session_summary",
        deserialize_with = "deserialize_session_summary"
    )]
    summary: String,
    #[serde(default, deserialize_with = "deserialize_lossy_string_vec")]
    topics: Vec<String>,
    #[serde(
        default = "default_emotional_tone",
        deserialize_with = "deserialize_emotional_tone"
    )]
    emotional_tone: String,
    #[serde(
        default = "default_session_outcome",
        deserialize_with = "deserialize_session_outcome"
    )]
    outcome: String,
    #[serde(default, deserialize_with = "deserialize_lossy_string_vec")]
    goals_mentioned: Vec<String>,
    #[serde(
        default = "default_emotional_intensity",
        deserialize_with = "deserialize_emotional_intensity"
    )]
    emotional_intensity: f32,
}

fn default_session_summary() -> String {
    "Session with various tasks".to_string()
}

fn default_emotional_tone() -> String {
    "neutral".to_string()
}

fn default_session_outcome() -> String {
    "ongoing".to_string()
}

fn default_emotional_intensity() -> f32 {
    0.3
}

fn deserialize_string_value<'de, D>(deserializer: D, fallback: &str) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(value
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback)
        .to_string())
}

fn deserialize_session_summary<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_string_value(deserializer, "Session with various tasks")
}

fn deserialize_emotional_tone<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_string_value(deserializer, "neutral")
}

fn deserialize_session_outcome<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_string_value(deserializer, "ongoing")
}

fn deserialize_lossy_string_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let values = match value {
        serde_json::Value::Array(values) => values,
        serde_json::Value::String(value) => vec![serde_json::Value::String(value)],
        _ => return Ok(Vec::new()),
    };
    Ok(values
        .into_iter()
        .filter_map(|value| value.as_str().map(str::trim).map(str::to_string))
        .filter(|value| !value.is_empty())
        .collect())
}

fn deserialize_emotional_intensity<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let intensity = value
        .as_f64()
        .map(|value| value as f32)
        .or_else(|| value.as_str().and_then(|value| value.parse::<f32>().ok()))
        .unwrap_or_else(default_emotional_intensity);
    Ok(intensity.clamp(0.0, 1.0))
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

        let transcript = MemoryManager::get_clean_transcript(&messages);
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

    #[test]
    fn session_analysis_preserves_valid_fields_when_siblings_are_null_or_malformed() {
        let analysis: SessionAnalysis = serde_json::from_str(
            r#"{
                "summary": {"unexpected":"object"},
                "topics": ["deployments", null, {"bad":true}, "reliability"],
                "emotional_tone": null,
                "outcome": "",
                "goals_mentioned": "Improve release reliability",
                "emotional_intensity": "1.7"
            }"#,
        )
        .unwrap();
        assert_eq!(analysis.summary, "Session with various tasks");
        assert_eq!(analysis.topics, ["deployments", "reliability"]);
        assert_eq!(analysis.emotional_tone, "neutral");
        assert_eq!(analysis.outcome, "ongoing");
        assert_eq!(analysis.goals_mentioned, ["Improve release reliability"]);
        assert_eq!(analysis.emotional_intensity, 1.0);
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
