use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use sqlx::{SqlitePool, Row};
use tracing::{error, info, warn};
use serde::Deserialize;
use serde_json::json;
use chrono::{DateTime, Utc};
use crate::memory::embeddings::EmbeddingService;
use crate::memory::scoring::calculate_episode_importance;
use crate::traits::{BehaviorPattern, Message, ModelProvider, UserProfile};

pub struct MemoryManager {
    pool: SqlitePool,
    embedding_service: Arc<EmbeddingService>,
    provider: Arc<dyn ModelProvider>,
    fast_model: String,
    consolidation_interval: Duration,
}

impl MemoryManager {
    pub fn new(
        pool: SqlitePool,
        embedding_service: Arc<EmbeddingService>,
        provider: Arc<dyn ModelProvider>,
        fast_model: String,
        consolidation_interval: Duration,
    ) -> Self {
        Self {
            pool,
            embedding_service,
            provider,
            fast_model,
            consolidation_interval,
        }
    }

    pub fn start_background_tasks(self: Arc<Self>) {
        // Embedding generation loop (every 5s)
        let manager = self.clone();
        tokio::spawn(async move {
            info!("Starting memory background tasks...");
            loop {
                if let Err(e) = manager.process_embeddings().await {
                    error!("Error processing embeddings: {}", e);
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });

        // Memory consolidation loop (configurable interval, default 6h)
        let manager = self.clone();
        let interval = self.consolidation_interval;
        tokio::spawn(async move {
            info!(
                interval_hours = interval.as_secs() / 3600,
                "Starting memory consolidation loop..."
            );
            loop {
                tokio::time::sleep(interval).await;
                if let Err(e) = manager.consolidate_memories().await {
                    error!("Error during memory consolidation: {}", e);
                }
            }
        });

        // Memory decay loop (daily at 3 AM or every 24h)
        let manager = self.clone();
        tokio::spawn(async move {
            info!("Starting memory decay loop (daily)...");
            loop {
                // Wait 24 hours
                tokio::time::sleep(Duration::from_secs(24 * 3600)).await;
                if let Err(e) = manager.decay_memories().await {
                    error!("Error during memory decay: {}", e);
                }
            }
        });

        // Goal review loop (weekly)
        let manager = self.clone();
        tokio::spawn(async move {
            info!("Starting goal review loop (weekly)...");
            loop {
                // Wait 7 days
                tokio::time::sleep(Duration::from_secs(7 * 24 * 3600)).await;
                if let Err(e) = manager.review_goals().await {
                    error!("Error during goal review: {}", e);
                }
            }
        });

        // Episode creation loop (every 30 minutes - check for idle sessions)
        let manager = self.clone();
        tokio::spawn(async move {
            info!("Starting episode creation loop (every 30m)...");
            loop {
                tokio::time::sleep(Duration::from_secs(30 * 60)).await;
                if let Err(e) = manager.create_episodes_for_idle_sessions().await {
                    error!("Error creating episodes for idle sessions: {}", e);
                }
            }
        });

        // Pattern detection and style analysis loop (every 6 hours)
        let manager = self.clone();
        tokio::spawn(async move {
            info!("Starting pattern detection loop (every 6h)...");
            loop {
                tokio::time::sleep(Duration::from_secs(6 * 3600)).await;
                if let Err(e) = manager.analyze_recent_activity().await {
                    error!("Error during pattern/style analysis: {}", e);
                }
            }
        });
    }

    async fn process_embeddings(&self) -> anyhow::Result<bool> {
        // Fetch messages without embeddings AND without errors
        // LIMIT 10 to check incrementally.
        let rows = sqlx::query(
            "SELECT id, content FROM messages WHERE embedding IS NULL AND embedding_error IS NULL AND content IS NOT NULL LIMIT 10"
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(false);
        }

        info!("Generating embeddings for {} messages", rows.len());

        for row in rows {
            let id: String = row.get("id");
            let content: Option<String> = row.get("content");

            if let Some(text) = content {
                match self.embedding_service.embed(text).await {
                    Ok(embedding) => {
                        // Serialize to JSON bytes
                        let blob = serde_json::to_vec(&embedding)?;
                        sqlx::query(
                            "UPDATE messages SET embedding = ? WHERE id = ?"
                        )
                        .bind(blob)
                        .bind(id)
                        .execute(&self.pool)
                        .await?;
                    }
                    Err(e) => {
                        let err_msg = e.to_string();
                        error!("Failed to generate embedding for message {}: {}", id, err_msg);
                        sqlx::query(
                            "UPDATE messages SET embedding_error = ? WHERE id = ?"
                        )
                        .bind(err_msg)
                        .bind(id)
                        .execute(&self.pool)
                        .await?;
                    }
                }
            }
        }

        Ok(true)
    }

    async fn consolidate_memories(&self) -> anyhow::Result<()> {
        // Find unconsolidated high-importance messages older than 1 hour, grouped by session
        let one_hour_ago = chrono::Utc::now() - chrono::Duration::hours(1);
        let cutoff = one_hour_ago.to_rfc3339();

        let rows = sqlx::query(
            "SELECT id, session_id, role, content, created_at
             FROM messages
             WHERE importance >= 0.7
               AND consolidated_at IS NULL
               AND created_at < ?
               AND content IS NOT NULL
             ORDER BY session_id, created_at ASC"
        )
        .bind(&cutoff)
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(());
        }

        // Group by session_id
        let mut sessions: std::collections::HashMap<String, Vec<(String, String, String)>> =
            std::collections::HashMap::new();

        for row in &rows {
            let id: String = row.get("id");
            let session_id: String = row.get("session_id");
            let role: String = row.get("role");
            let content: String = row.get("content");
            let entry = sessions.entry(session_id).or_default();
            // Cap at 30 messages per session
            if entry.len() < 30 {
                entry.push((id, role, content));
            }
        }

        let total_messages: usize = sessions.values().map(|v| v.len()).sum();
        info!(
            "Consolidation: processing {} messages from {} sessions",
            total_messages,
            sessions.len()
        );

        let system_prompt = "You are a memory consolidation system. Given a conversation excerpt, \
            extract durable facts worth remembering long-term. Output ONLY a JSON array: \
            [{\"category\": \"...\", \"key\": \"...\", \"value\": \"...\"}]. \
            Categories:\n\
            - user: Personal info (name, location, job)\n\
            - preference: Tool, workflow, and communication preferences\n\
            - project: Projects, tech stacks, goals\n\
            - technical: Environment details, installed tools\n\
            - relationship: Communication patterns with the AI\n\
            - behavior: Observed tool-usage patterns and recurring workflows\n\n\
            For \"behavior\", look for:\n\
            - Which tools the user prefers for specific tasks\n\
            - Recurring workflows or action sequences\n\
            - Types of tasks frequently delegated\n\n\
            Only extract facts useful in future conversations. If nothing worth remembering, return [].";

        for (session_id, messages) in &sessions {
            // Build conversation text for this session batch
            let conversation: String = messages
                .iter()
                .map(|(_id, role, content)| format!("{}: {}", role, content))
                .collect::<Vec<_>>()
                .join("\n");

            let llm_messages = vec![
                json!({"role": "system", "content": system_prompt}),
                json!({"role": "user", "content": conversation}),
            ];

            // Call LLM with fast model, no tools
            match self.provider.chat(&self.fast_model, &llm_messages, &[]).await {
                Ok(response) => {
                    if let Some(text) = &response.content {
                        match parse_consolidation_response(text) {
                            Ok(facts) => {
                                for fact in &facts {
                                    if let Err(e) = self.upsert_fact(
                                        &fact.category,
                                        &fact.key,
                                        &fact.value,
                                    ).await {
                                        warn!(
                                            "Failed to upsert consolidated fact [{}/{}]: {}",
                                            fact.category, fact.key, e
                                        );
                                    }
                                }
                                info!(
                                    "Consolidation: extracted {} facts from session {}",
                                    facts.len(),
                                    session_id
                                );
                            }
                            Err(e) => {
                                warn!(
                                    "Failed to parse consolidation response for session {}: {} — messages will be retried next cycle",
                                    session_id, e
                                );
                            }
                        }
                    }

                    // Only mark messages as consolidated when facts were successfully parsed
                    // On parse failure, messages remain unconsolidated for retry next cycle
                    if response.content.as_ref().map_or(false, |text| parse_consolidation_response(text).is_ok()) {
                        let now = chrono::Utc::now().to_rfc3339();
                        for (id, _role, _content) in messages {
                            let _ = sqlx::query(
                                "UPDATE messages SET consolidated_at = ? WHERE id = ?"
                            )
                            .bind(&now)
                            .bind(id)
                            .execute(&self.pool)
                            .await;
                        }
                    }
                }
                Err(e) => {
                    error!(
                        "LLM call failed during consolidation for session {}: {}",
                        session_id, e
                    );
                    // Don't mark as consolidated — retry next cycle
                }
            }
        }

        Ok(())
    }

    async fn upsert_fact(&self, category: &str, key: &str, value: &str) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();

        // Use supersession logic: find existing current fact
        let existing = sqlx::query(
            "SELECT id, value FROM facts WHERE category = ? AND key = ? AND superseded_at IS NULL"
        )
        .bind(category)
        .bind(key)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = existing {
            let old_value: String = row.get("value");
            let old_id: i64 = row.get("id");

            if old_value != value {
                // Mark old as superseded
                sqlx::query("UPDATE facts SET superseded_at = ? WHERE id = ?")
                    .bind(&now)
                    .bind(old_id)
                    .execute(&self.pool)
                    .await?;

                // Insert new fact
                sqlx::query(
                    "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count)
                     VALUES (?, ?, ?, 'consolidation', ?, ?, 0)"
                )
                .bind(category)
                .bind(key)
                .bind(value)
                .bind(&now)
                .bind(&now)
                .execute(&self.pool)
                .await?;
            }
        } else {
            // No existing fact - insert new
            sqlx::query(
                "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count)
                 VALUES (?, ?, ?, 'consolidation', ?, ?, 0)"
            )
            .bind(category)
            .bind(key)
            .bind(value)
            .bind(&now)
            .bind(&now)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    // ==================== Episode Creation ====================

    /// Generate a clean transcript from messages, compressing tool calls.
    pub fn get_clean_transcript(messages: &[Message]) -> String {
        let mut transcript = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "user" => {
                    if let Some(content) = &msg.content {
                        transcript.push_str(&format!("User: {}\n\n", content));
                    }
                }
                "assistant" => {
                    // Check for tool calls
                    if let Some(tc_json) = &msg.tool_calls_json {
                        if let Ok(tool_calls) = serde_json::from_str::<Vec<serde_json::Value>>(tc_json) {
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
                            let truncated = if content.len() > 500 {
                                format!("{}...", &content[..500])
                            } else {
                                content.clone()
                            };
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
                                format!("[{}: {}]", name, content.chars().take(100).collect::<String>())
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
    pub async fn create_episode(&self, session_id: &str, messages: &[Message]) -> anyhow::Result<i64> {
        if messages.is_empty() {
            return Err(anyhow::anyhow!("No messages to create episode from"));
        }

        let clean_transcript = Self::get_clean_transcript(messages);

        // Call LLM to analyze the session
        let analysis = self.analyze_session(&clean_transcript).await?;

        // Calculate importance
        let has_errors = messages.iter().any(|m| {
            m.content.as_ref().map_or(false, |c| {
                c.to_lowercase().contains("error") || c.to_lowercase().contains("failed")
            })
        });
        let has_decisions = messages.iter().any(|m| {
            m.content.as_ref().map_or(false, |c| {
                let lower = c.to_lowercase();
                lower.contains("decided") || lower.contains("let's go with") || lower.contains("i'll use")
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
        let start_time = messages.first().map(|m| m.created_at).unwrap_or_else(Utc::now);
        let end_time = messages.last().map(|m| m.created_at).unwrap_or_else(Utc::now);
        let now = Utc::now();

        // Insert episode
        let topics_json = serde_json::to_string(&analysis.topics).ok();
        let result = sqlx::query(
            "INSERT INTO episodes (session_id, summary, topics, emotional_tone, outcome, importance, recall_count, message_count, start_time, end_time, created_at)
             VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)"
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
        .execute(&self.pool)
        .await?;

        let episode_id = result.last_insert_rowid();

        // Generate and store embedding for the summary
        if let Ok(embedding) = self.embedding_service.embed(analysis.summary.clone()).await {
            let blob = serde_json::to_vec(&embedding)?;
            sqlx::query("UPDATE episodes SET embedding = ? WHERE id = ?")
                .bind(blob)
                .bind(episode_id)
                .execute(&self.pool)
                .await?;
        }

        // Extract goals
        for goal_text in &analysis.goals_mentioned {
            if let Err(e) = self.extract_goal(goal_text, episode_id).await {
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
5. Any goals mentioned or worked toward

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

        let response = self.provider.chat(&self.fast_model, &llm_messages, &[]).await?;

        let text = response.content.ok_or_else(|| anyhow::anyhow!("Empty response from LLM"))?;

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

        let analysis: SessionAnalysis = serde_json::from_str(json_str)
            .unwrap_or_else(|_| SessionAnalysis {
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
    async fn extract_goal(&self, goal_text: &str, source_episode_id: i64) -> anyhow::Result<()> {
        // Check for similar existing goal
        let existing = sqlx::query(
            "SELECT id, description FROM goals WHERE status = 'active'"
        )
        .fetch_all(&self.pool)
        .await?;

        // Simple text similarity check
        let goal_lower = goal_text.to_lowercase();
        for row in existing {
            let id: i64 = row.get("id");
            let description: String = row.get("description");
            let desc_lower = description.to_lowercase();

            // Check for word overlap
            let goal_words: std::collections::HashSet<&str> = goal_lower.split_whitespace().collect();
            let desc_words: std::collections::HashSet<&str> = desc_lower.split_whitespace().collect();
            let intersection = goal_words.intersection(&desc_words).count();
            let union = goal_words.union(&desc_words).count();

            if union > 0 && (intersection as f32 / union as f32) > 0.5 {
                // Similar goal exists - add progress note
                let now = Utc::now().to_rfc3339();
                let note = format!("Referenced in episode {}", source_episode_id);

                let notes_row = sqlx::query("SELECT progress_notes FROM goals WHERE id = ?")
                    .bind(id)
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
                    .bind(id)
                    .execute(&self.pool)
                    .await?;

                return Ok(());
            }
        }

        // No similar goal - create new
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO goals (description, status, priority, source_episode_id, created_at, updated_at)
             VALUES (?, 'active', 'medium', ?, ?, ?)"
        )
        .bind(goal_text)
        .bind(source_episode_id)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        info!(goal = goal_text, episode_id = source_episode_id, "Created new goal");
        Ok(())
    }

    // ==================== Communication Style Analysis ====================

    /// Analyze user's communication style from recent sessions.
    pub async fn analyze_communication_style(&self, recent_messages: &[Message]) -> anyhow::Result<UserProfile> {
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
            .sum::<f32>() / user_messages.len() as f32;

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
                asks_before_acting: true,
                prefers_explanations: true,
                likes_suggestions: false,
                updated_at: Utc::now(),
            })
        }
    }

    async fn update_profile(&self, profile: &UserProfile) -> anyhow::Result<()> {
        let active_hours_json = profile.active_hours.as_ref().map(|h| serde_json::to_string(h).unwrap_or_default());
        let workflows_json = profile.common_workflows.as_ref().map(|w| serde_json::to_string(w).unwrap_or_default());
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
    pub async fn detect_patterns(&self, recent_messages: &[Message]) -> anyhow::Result<Vec<BehaviorPattern>> {
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
        // Decay facts not recalled in the last 30 days
        let thirty_days_ago = (Utc::now() - chrono::Duration::days(30)).to_rfc3339();

        sqlx::query(
            "UPDATE facts SET recall_count = MAX(0, recall_count - 1)
             WHERE recall_count > 0 AND (last_recalled_at IS NULL OR last_recalled_at < ?)"
        )
        .bind(&thirty_days_ago)
        .execute(&self.pool)
        .await?;

        // Decay episodes
        sqlx::query(
            "UPDATE episodes SET recall_count = MAX(0, recall_count - 1)
             WHERE recall_count > 0 AND (last_recalled_at IS NULL OR last_recalled_at < ?)"
        )
        .bind(&thirty_days_ago)
        .execute(&self.pool)
        .await?;

        // Decay behavior pattern confidence
        sqlx::query(
            "UPDATE behavior_patterns SET confidence = MAX(0.1, confidence - 0.05)
             WHERE confidence > 0.1 AND (last_seen_at IS NULL OR last_seen_at < ?)"
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
            "SELECT id, description FROM goals WHERE status = 'active' AND updated_at < ?"
        )
        .bind(&two_weeks_ago)
        .fetch_all(&self.pool)
        .await?;

        for row in stale_goals {
            let id: i64 = row.get("id");
            let description: String = row.get("description");
            info!(goal_id = id, description, "Stale goal detected - may need user input");
        }

        Ok(())
    }

    // ==================== Background Task Helpers ====================

    /// Create episodes for sessions that have been idle for 30+ minutes without an episode.
    async fn create_episodes_for_idle_sessions(&self) -> anyhow::Result<()> {
        let thirty_mins_ago = (Utc::now() - chrono::Duration::minutes(30)).to_rfc3339();

        // Find sessions with messages older than 30 mins that don't have episodes yet
        // and have at least 5 messages (meaningful sessions)
        let idle_sessions: Vec<String> = sqlx::query_scalar(
            "SELECT DISTINCT m.session_id
             FROM messages m
             LEFT JOIN episodes e ON e.session_id = m.session_id
             WHERE m.created_at < ?
               AND e.id IS NULL
             GROUP BY m.session_id
             HAVING COUNT(m.id) >= 5
               AND MAX(m.created_at) < ?"
        )
        .bind(&thirty_mins_ago)
        .bind(&thirty_mins_ago)
        .fetch_all(&self.pool)
        .await?;

        if idle_sessions.is_empty() {
            return Ok(());
        }

        info!(count = idle_sessions.len(), "Creating episodes for idle sessions");

        for session_id in idle_sessions {
            let messages = self.fetch_session_messages(&session_id, 100).await?;
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

    /// Analyze recent activity for pattern detection and style updates.
    async fn analyze_recent_activity(&self) -> anyhow::Result<()> {
        // Fetch messages from the last 24 hours for analysis
        let one_day_ago = (Utc::now() - chrono::Duration::hours(24)).to_rfc3339();

        let recent_messages = self.fetch_messages_since(&one_day_ago, 500).await?;

        if recent_messages.is_empty() {
            return Ok(());
        }

        // Detect behavior patterns
        let patterns = self.detect_patterns(&recent_messages).await?;
        if !patterns.is_empty() {
            info!(count = patterns.len(), "Detected behavior patterns");
        }

        // Update communication style profile
        let _profile = self.analyze_communication_style(&recent_messages).await?;
        info!("Updated user communication profile");

        Ok(())
    }

    /// Fetch messages for a specific session.
    async fn fetch_session_messages(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<Message>> {
        let rows = sqlx::query(
            "SELECT id, session_id, role, content, tool_call_id, tool_name, tool_calls_json, created_at, importance
             FROM messages
             WHERE session_id = ?
             ORDER BY created_at ASC
             LIMIT ?"
        )
        .bind(session_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let messages = rows.into_iter().map(|row| {
            let created_str: String = row.get("created_at");
            Message {
                id: row.get("id"),
                session_id: row.get("session_id"),
                role: row.get("role"),
                content: row.get("content"),
                tool_call_id: row.get("tool_call_id"),
                tool_name: row.get("tool_name"),
                tool_calls_json: row.get("tool_calls_json"),
                created_at: DateTime::parse_from_rfc3339(&created_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                importance: row.get("importance"),
                embedding: None,
            }
        }).collect();

        Ok(messages)
    }

    /// Fetch messages from all sessions since a given time.
    async fn fetch_messages_since(&self, since: &str, limit: usize) -> anyhow::Result<Vec<Message>> {
        let rows = sqlx::query(
            "SELECT id, session_id, role, content, tool_call_id, tool_name, tool_calls_json, created_at, importance
             FROM messages
             WHERE created_at >= ?
             ORDER BY created_at DESC
             LIMIT ?"
        )
        .bind(since)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let messages = rows.into_iter().map(|row| {
            let created_str: String = row.get("created_at");
            Message {
                id: row.get("id"),
                session_id: row.get("session_id"),
                role: row.get("role"),
                content: row.get("content"),
                tool_call_id: row.get("tool_call_id"),
                tool_name: row.get("tool_name"),
                tool_calls_json: row.get("tool_calls_json"),
                created_at: DateTime::parse_from_rfc3339(&created_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                importance: row.get("importance"),
                embedding: None,
            }
        }).collect();

        Ok(messages)
    }
}

#[derive(Debug, Deserialize)]
struct ExtractedFact {
    category: String,
    key: String,
    value: String,
}

#[derive(Debug, Deserialize)]
struct SessionAnalysis {
    summary: String,
    topics: Vec<String>,
    emotional_tone: String,
    outcome: String,
    #[serde(default)]
    goals_mentioned: Vec<String>,
    #[serde(default = "default_emotional_intensity")]
    emotional_intensity: f32,
}

fn default_emotional_intensity() -> f32 {
    0.3
}

fn parse_consolidation_response(text: &str) -> anyhow::Result<Vec<ExtractedFact>> {
    // Try to find JSON array in the response (LLM may include markdown fences)
    let trimmed = text.trim();
    let json_str = if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            &trimmed[start..=end]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    let facts: Vec<ExtractedFact> = serde_json::from_str(json_str)?;
    Ok(facts)
}
