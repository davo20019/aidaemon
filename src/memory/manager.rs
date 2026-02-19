use crate::config::PeopleConfig;
use crate::events::Consolidator;
use crate::llm_runtime::SharedLlmRuntime;
use crate::memory::binary::encode_embedding;
use crate::memory::embeddings::EmbeddingService;
use crate::memory::scoring::calculate_episode_importance;
use crate::traits::{BehaviorPattern, Message, Person, StateStore, UserProfile};
use crate::types::{ChannelVisibility, FactPrivacy};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::json;
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

pub struct MemoryManager {
    pool: SqlitePool,
    embedding_service: Arc<EmbeddingService>,
    llm_runtime: SharedLlmRuntime,
    consolidation_interval: Duration,
    consolidator: Option<Arc<Consolidator>>,
    state: Option<Arc<dyn StateStore>>,
    people_config: PeopleConfig,
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
            pool,
            embedding_service,
            llm_runtime,
            consolidation_interval,
            consolidator,
            state: None,
            people_config: PeopleConfig::default(),
        }
    }

    /// Set the state store for people fact routing.
    pub fn with_state(mut self, state: Arc<dyn StateStore>) -> Self {
        self.state = Some(state);
        self
    }

    /// Set the people config for controlling auto-extraction.
    pub fn with_people_config(mut self, config: PeopleConfig) -> Self {
        self.people_config = config;
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
            "SELECT id, trigger_pattern FROM procedures WHERE trigger_embedding IS NULL AND trigger_pattern IS NOT NULL LIMIT 10",
        )
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
            "SELECT id, error_pattern FROM error_solutions WHERE error_embedding IS NULL AND error_pattern IS NOT NULL LIMIT 10",
        )
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

    async fn session_visibility_from_events(&self, session_id: &str) -> Option<ChannelVisibility> {
        let row = sqlx::query(
            r#"
            SELECT data
            FROM events
            WHERE session_id = ?
              AND event_type = 'user_message'
            ORDER BY id DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await;

        let row = match row {
            Ok(r) => r,
            Err(e) => {
                if e.to_string().contains("no such table: events") {
                    return None;
                }
                warn!(session_id, error = %e, "Failed to fetch session visibility from events");
                return None;
            }
        }?;

        let raw: String = row.get("data");
        let val: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let vis = val.get("channel_visibility").and_then(|v| v.as_str())?;
        Some(ChannelVisibility::from_str_lossy(vis))
    }

    fn fact_consolidation_cursor_key(session_id: &str) -> String {
        format!("memory_fact_consolidation_cursor:{}", session_id)
    }

    async fn get_fact_consolidation_cursor(&self, session_id: &str) -> i64 {
        let Some(state) = &self.state else {
            return 0;
        };
        let key = Self::fact_consolidation_cursor_key(session_id);
        match state.get_setting(&key).await {
            Ok(Some(raw)) => raw.parse::<i64>().unwrap_or(0),
            _ => 0,
        }
    }

    async fn set_fact_consolidation_cursor(
        &self,
        session_id: &str,
        event_id: i64,
    ) -> anyhow::Result<()> {
        let Some(state) = &self.state else {
            return Ok(());
        };
        let key = Self::fact_consolidation_cursor_key(session_id);
        state.set_setting(&key, &event_id.to_string()).await
    }

    async fn list_candidate_sessions_for_fact_consolidation(
        &self,
        cutoff: &str,
    ) -> anyhow::Result<Vec<String>> {
        let rows = sqlx::query_scalar(
            r#"
            SELECT DISTINCT session_id
            FROM events
            WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
              AND created_at < ?
            "#,
        )
        .bind(cutoff)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    async fn fetch_fact_consolidation_messages(
        &self,
        session_id: &str,
        cutoff: &str,
        after_event_id: i64,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::events::ConversationTurn>> {
        let rows = sqlx::query(
            r#"
            SELECT id, event_type, data, created_at
            FROM events
            WHERE session_id = ?
              AND event_type IN ('user_message', 'assistant_response', 'tool_result')
              AND id > ?
              AND created_at < ?
            ORDER BY id ASC
            LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(after_event_id)
        .bind(cutoff)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let event_id: i64 = row.get("id");
            let event_type: String = row.get("event_type");
            let created_str: String = row.get("created_at");
            let created_at = DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        session_id,
                        event_id,
                        event_type = %event_type,
                        error = %e,
                        "Skipping malformed event payload during memory consolidation"
                    );
                    continue;
                }
            };

            if let Some(turn) =
                crate::events::turn_from_event(event_id, session_id, &event_type, &data, created_at)
            {
                out.push(turn);
            }
        }
        Ok(out)
    }

    async fn consolidate_memories(&self) -> anyhow::Result<()> {
        // Consolidate conversation events older than 1 hour.
        // Per-session cursors avoid rescanning already-processed history.
        let one_hour_ago = chrono::Utc::now() - chrono::Duration::hours(1);
        let cutoff = one_hour_ago.to_rfc3339();

        let candidate_sessions = self
            .list_candidate_sessions_for_fact_consolidation(&cutoff)
            .await?;
        if candidate_sessions.is_empty() {
            return Ok(());
        }

        // session_id -> selected high-importance messages (event_id, role, content)
        let mut sessions: std::collections::HashMap<String, Vec<(i64, String, String)>> =
            std::collections::HashMap::new();
        // session_id -> max event id seen this cycle (for cursor advancement)
        let mut session_max_seen: std::collections::HashMap<String, i64> =
            std::collections::HashMap::new();

        for session_id in candidate_sessions {
            let cursor = self.get_fact_consolidation_cursor(&session_id).await;
            let events = self
                .fetch_fact_consolidation_messages(&session_id, &cutoff, cursor, 300)
                .await?;
            if events.is_empty() {
                continue;
            }

            if let Some(max_seen) = events.last().map(|turn| turn.event_id) {
                session_max_seen.insert(session_id.clone(), max_seen);
            }

            let entry = sessions.entry(session_id).or_default();
            for turn in events {
                let Some(content) = turn.content.clone() else {
                    continue;
                };
                if content.trim().is_empty() {
                    continue;
                }
                let importance = crate::memory::scoring::score_turn(&turn);
                if importance >= 0.7 && entry.len() < 30 {
                    entry.push((turn.event_id, turn.role.as_str().to_string(), content));
                }
            }
        }

        let total_messages: usize = sessions.values().map(|v| v.len()).sum();
        info!(
            "Consolidation: processing {} messages from {} sessions (event-native)",
            total_messages,
            sessions.len()
        );

        // If no high-importance candidates were found, advance cursors so we
        // don't repeatedly rescan low-signal events.
        if sessions.is_empty() {
            for (session_id, max_seen) in &session_max_seen {
                let _ = self
                    .set_fact_consolidation_cursor(session_id, *max_seen)
                    .await;
            }
            return Ok(());
        }

        for (session_id, max_seen) in &session_max_seen {
            if !sessions.contains_key(session_id) {
                let _ = self
                    .set_fact_consolidation_cursor(session_id, *max_seen)
                    .await;
            }
        }

        // PublicExternal sessions are untrusted: do not learn durable facts from them.
        // (Progressive extraction is already disabled; this closes the consolidation path.)
        let mut skipped_public_external: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        let system_prompt = "You are a memory consolidation system. Given a conversation excerpt, \
            extract durable facts worth remembering long-term. Output ONLY a JSON array: \
            [{\"category\": \"...\", \"key\": \"...\", \"value\": \"...\", \"privacy\": \"...\"}]. \
            Categories:\n\
            - user: Personal info about the OWNER (name, location, job)\n\
            - preference: Tool, workflow, and communication preferences\n\
            - project: Projects, tech stacks, goals\n\
            - technical: Environment details, installed tools\n\
            - relationship: Communication patterns with the AI\n\
            - behavior: Observed tool-usage patterns and recurring workflows\n\
            - people: Information about OTHER individuals mentioned or participating in conversation\n\n\
            For \"behavior\", look for:\n\
            - Which tools the user prefers for specific tasks\n\
            - Recurring workflows or action sequences\n\
            - Types of tasks frequently delegated\n\n\
            For \"people\", extract:\n\
            - Names and relationships mentioned (e.g., \"my wife Aracely\", \"coworker Juan\")\n\
            - Personal details about others (birthdays, preferences, interests, jobs)\n\
            - Important dates related to people\n\
            - Format the key as \"person_name:detail_type\" (e.g., \"aracely:birthday\", \"juan:job\")\n\
            - Include a \"person_name\" field with just the person's name\n\
            - NEVER extract health info, financial details, political opinions, or religious beliefs about people\n\
            Example: {\"category\": \"people\", \"key\": \"aracely:birthday\", \"value\": \"March 15\", \"privacy\": \"private\", \"person_name\": \"Aracely\"}\n\n\
            Also classify each fact's privacy:\n\
            - \"global\": General facts useful anywhere (name, job, timezone, tech preferences)\n\
            - \"channel\": Context-specific facts from this conversation\n\
            - \"private\": Sensitive personal info the user would want kept private\n\
            Default to \"channel\" if unsure. People facts should default to \"private\".\n\n\
            Only extract facts useful in future conversations. If nothing worth remembering, return [].";

        for (session_id, messages) in &sessions {
            if self
                .session_visibility_from_events(session_id)
                .await
                .is_some_and(|v| v == ChannelVisibility::PublicExternal)
            {
                skipped_public_external.insert(session_id.clone());
                if let Some(max_seen) = session_max_seen.get(session_id) {
                    let _ = self
                        .set_fact_consolidation_cursor(session_id, *max_seen)
                        .await;
                }
                info!(
                    session_id = session_id.as_str(),
                    "Skipping memory fact consolidation for PublicExternal session"
                );
                continue;
            }

            // Build conversation text for this session batch
            let conversation: String = messages
                .iter()
                .map(|(_event_id, role, content)| format!("{}: {}", role, content))
                .collect::<Vec<_>>()
                .join("\n");

            let llm_messages = vec![
                json!({"role": "system", "content": system_prompt}),
                json!({"role": "user", "content": conversation}),
            ];

            // Call LLM with fast model, no tools
            let runtime_snapshot = self.llm_runtime.snapshot();
            let fast_model = runtime_snapshot.fast_model();
            match runtime_snapshot
                .provider()
                .chat(&fast_model, &llm_messages, &[])
                .await
            {
                Ok(response) => {
                    // Track token usage for background LLM calls
                    if let (Some(state), Some(usage)) = (&self.state, &response.usage) {
                        let _ = state
                            .record_token_usage("background:memory_consolidation", usage)
                            .await;
                    }
                    if let Some(text) = &response.content {
                        match parse_consolidation_response(text) {
                            Ok(facts) => {
                                let ch_id =
                                    crate::memory::derive_channel_id_from_session(session_id);
                                for fact in &facts {
                                    // Route "people" facts to the person_facts table
                                    if fact.category == "people" {
                                        self.route_people_fact(fact).await;
                                        continue;
                                    }

                                    let privacy = fact
                                        .privacy
                                        .as_deref()
                                        .map(FactPrivacy::from_str_lossy)
                                        .unwrap_or(FactPrivacy::Channel);
                                    if let Err(e) = self
                                        .upsert_fact(
                                            &fact.category,
                                            &fact.key,
                                            &fact.value,
                                            ch_id.as_deref(),
                                            privacy,
                                        )
                                        .await
                                    {
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
                                if let Some(max_seen) = session_max_seen.get(session_id) {
                                    let _ = self
                                        .set_fact_consolidation_cursor(session_id, *max_seen)
                                        .await;
                                }
                            }
                            Err(e) => {
                                warn!(
                                    "Failed to parse consolidation response for session {}: {} — events will be retried next cycle",
                                    session_id, e
                                );
                            }
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

        // After fact extraction, run event-based extraction (procedures, errors, expertise)
        // via the Consolidator. This unifies both pipelines into a single cycle.
        if let Some(ref consolidator) = self.consolidator {
            let session_ids: Vec<String> = sessions
                .keys()
                .filter(|sid| !skipped_public_external.contains(*sid))
                .cloned()
                .collect();
            for session_id in &session_ids {
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

    /// Check if two person names likely refer to the same person.
    /// Handles first-name-only match, case-insensitive, prefix match.
    fn names_likely_match(candidate: &str, existing: &str) -> bool {
        let c = candidate.trim().to_lowercase();
        let e = existing.trim().to_lowercase();

        if c == e {
            return true;
        }

        // First-name match: "John" matches "John Smith"
        let c_parts: Vec<&str> = c.split_whitespace().collect();
        let e_parts: Vec<&str> = e.split_whitespace().collect();

        // Single-name candidate matches multi-name existing (first name)
        if c_parts.len() == 1 && e_parts.len() > 1 && c_parts[0] == e_parts[0] {
            return true;
        }
        // Multi-name candidate matches single-name existing (first name)
        if e_parts.len() == 1 && c_parts.len() > 1 && c_parts[0] == e_parts[0] {
            return true;
        }

        false
    }

    /// Find a person by exact name first, then fuzzy match against all people + aliases.
    /// Returns the matched person and upgrades name to more complete form when found.
    async fn find_or_match_person(
        state: &Arc<dyn StateStore>,
        name: &str,
    ) -> anyhow::Result<Option<Person>> {
        // Exact match first
        if let Some(person) = state.find_person_by_name(name).await? {
            return Ok(Some(person));
        }

        // Fuzzy match against all people
        let all_people = state.get_all_people().await?;
        for person in &all_people {
            if Self::names_likely_match(name, &person.name) {
                return Ok(Some(person.clone()));
            }
            // Check aliases too
            for alias in &person.aliases {
                if Self::names_likely_match(name, alias) {
                    return Ok(Some(person.clone()));
                }
            }
        }

        Ok(None)
    }

    /// Normalize a person name: trim, title-case.
    fn normalize_person_name(name: &str) -> String {
        name.split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => {
                        first.to_uppercase().to_string() + &chars.as_str().to_lowercase()
                    }
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Route a "people" category fact to the person_facts table.
    async fn route_people_fact(&self, fact: &ExtractedFact) {
        let state = match &self.state {
            Some(s) => s,
            None => return,
        };

        // Check runtime setting first, fall back to config
        let people_enabled = match state.get_setting("people_enabled").await {
            Ok(Some(val)) => val == "true",
            _ => self.people_config.enabled,
        };
        if !people_enabled || !self.people_config.auto_extract {
            return;
        }

        let raw_name = match &fact.person_name {
            Some(name) if !name.is_empty() => name.clone(),
            _ => {
                // Try to extract person name from the key (format: "person_name:detail_type")
                match fact.key.split(':').next() {
                    Some(name) if !name.is_empty() => name.to_string(),
                    _ => return,
                }
            }
        };

        // Name normalization: trim, title-case, reject single-char names
        let person_name = Self::normalize_person_name(&raw_name);
        if person_name.len() < 2 {
            return;
        }

        // Extract the detail type from the key
        let detail_key = fact.key.split(':').nth(1).unwrap_or(&fact.key);

        // Check restricted categories
        if self
            .people_config
            .restricted_categories
            .iter()
            .any(|rc| rc == detail_key)
        {
            info!("Skipping restricted people fact category: {}", detail_key);
            return;
        }

        // Check if this detail type is in allowed auto-extract categories
        if !self
            .people_config
            .auto_extract_categories
            .iter()
            .any(|ac| ac == detail_key)
        {
            // Still allow common detail types not in the explicit list
            let always_allowed = ["name", "relationship", "nickname", "job", "role"];
            if !always_allowed.contains(&detail_key) {
                return;
            }
        }

        // Find or create the person (with fuzzy name matching)
        let person_id = match Self::find_or_match_person(state, &person_name).await {
            Ok(Some(p)) => {
                // If we matched via fuzzy and the new name is more complete, upgrade
                if p.name.len() < person_name.len()
                    && Self::names_likely_match(&person_name, &p.name)
                {
                    let mut updated = p.clone();
                    updated.name = person_name.clone();
                    updated.updated_at = chrono::Utc::now();
                    if let Err(e) = state.upsert_person(&updated).await {
                        warn!("Failed to upgrade person name to '{}': {}", person_name, e);
                    } else {
                        info!(
                            "Upgraded person name '{}' → '{}' (ID: {})",
                            p.name, person_name, p.id
                        );
                    }
                }
                p.id
            }
            Ok(None) => {
                // Auto-create with minimal info
                let person = Person {
                    id: 0,
                    name: person_name.clone(),
                    aliases: vec![],
                    relationship: None,
                    platform_ids: std::collections::HashMap::new(),
                    notes: None,
                    communication_style: None,
                    language_preference: None,
                    last_interaction_at: None,
                    interaction_count: 0,
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                };
                match state.upsert_person(&person).await {
                    Ok(id) => {
                        info!(
                            "Auto-created person '{}' (ID: {}) from consolidation",
                            person_name, id
                        );
                        id
                    }
                    Err(e) => {
                        warn!("Failed to auto-create person '{}': {}", person_name, e);
                        return;
                    }
                }
            }
            Err(e) => {
                warn!("Failed to look up person '{}': {}", person_name, e);
                return;
            }
        };

        // Store the fact with low confidence (auto-extracted)
        if let Err(e) = state
            .upsert_person_fact(
                person_id,
                detail_key,
                &fact.key,
                &fact.value,
                "consolidation",
                0.7,
            )
            .await
        {
            warn!(
                "Failed to store people fact [{}/{}] for '{}': {}",
                detail_key, fact.key, person_name, e
            );
        } else {
            debug!(
                "Stored people fact [{}/{}] for '{}' (confidence: 0.7)",
                detail_key, fact.key, person_name
            );
        }
    }

    async fn upsert_fact(
        &self,
        category: &str,
        key: &str,
        value: &str,
        channel_id: Option<&str>,
        privacy: FactPrivacy,
    ) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        let privacy_str = privacy.to_string();

        // Use an IMMEDIATE transaction to make the SELECT → UPDATE → INSERT atomic.
        // This prevents concurrent consolidation cycles from losing facts.
        let mut tx = self.pool.begin().await?;

        // Use supersession logic: find existing current fact
        let existing = sqlx::query(
            "SELECT id, value FROM facts WHERE category = ? AND key = ? AND superseded_at IS NULL",
        )
        .bind(category)
        .bind(key)
        .fetch_optional(&mut *tx)
        .await?;

        if let Some(row) = existing {
            let old_value: String = row.get("value");
            let old_id: i64 = row.get("id");

            if old_value != value {
                // Mark old as superseded
                sqlx::query("UPDATE facts SET superseded_at = ? WHERE id = ?")
                    .bind(&now)
                    .bind(old_id)
                    .execute(&mut *tx)
                    .await?;

                // Insert new fact — ignore duplicate entry errors (code 2067)
                let insert_result = sqlx::query(
                    "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count, channel_id, privacy)
                     VALUES (?, ?, ?, 'consolidation', ?, ?, 0, ?, ?)"
                )
                .bind(category)
                .bind(key)
                .bind(value)
                .bind(&now)
                .bind(&now)
                .bind(channel_id)
                .bind(&privacy_str)
                .execute(&mut *tx)
                .await;

                match insert_result {
                    Ok(_) => {}
                    Err(sqlx::Error::Database(ref db_err))
                        if db_err.code().as_deref() == Some("2067") =>
                    {
                        // Duplicate entry — fact already exists, ignore
                    }
                    Err(e) => {
                        tx.rollback().await.ok();
                        return Err(e.into());
                    }
                }
            }
        } else {
            // Check for previously-deleted (superseded) fact — don't re-create via consolidation.
            // This respects the user's explicit deletion. They can still re-add facts
            // explicitly via the remember_fact tool which uses StateStore::upsert_fact.
            let was_deleted = sqlx::query(
                "SELECT id FROM facts WHERE category = ? AND key = ? AND superseded_at IS NOT NULL LIMIT 1",
            )
            .bind(category)
            .bind(key)
            .fetch_optional(&mut *tx)
            .await?;

            if was_deleted.is_some() {
                tx.commit().await?;
                return Ok(());
            }

            // No existing fact - insert new (ignore duplicate entry errors from concurrent inserts)
            let insert_result = sqlx::query(
                "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count, channel_id, privacy)
                 VALUES (?, ?, ?, 'consolidation', ?, ?, 0, ?, ?)"
            )
            .bind(category)
            .bind(key)
            .bind(value)
            .bind(&now)
            .bind(&now)
            .bind(channel_id)
            .bind(&privacy_str)
            .execute(&mut *tx)
            .await;

            match insert_result {
                Ok(_) => {}
                Err(sqlx::Error::Database(ref db_err))
                    if db_err.code().as_deref() == Some("2067") =>
                {
                    // Duplicate entry — fact already exists, ignore
                }
                Err(e) => {
                    tx.rollback().await.ok();
                    return Err(e.into());
                }
            }
        }
        tx.commit().await?;
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

        // Insert episode. Multiple episodes per session are allowed (mid-session
        // episodes for long-running conversations + final idle-session episode).
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

        let runtime_snapshot = self.llm_runtime.snapshot();
        let fast_model = runtime_snapshot.fast_model();
        let response = runtime_snapshot
            .provider()
            .chat(&fast_model, &llm_messages, &[])
            .await?;

        // Track token usage for background LLM calls
        if let (Some(state), Some(usage)) = (&self.state, &response.usage) {
            let _ = state
                .record_token_usage("background:episode_creation", usage)
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
    async fn extract_goal(&self, goal_text: &str, source_episode_id: i64) -> anyhow::Result<()> {
        // Check for similar existing goal (including abandoned/completed to prevent resurrection)
        let existing = sqlx::query(
            "SELECT id, description, status FROM goals WHERE status IN ('active', 'abandoned', 'completed')"
        )
        .fetch_all(&self.pool)
        .await?;

        // Simple text similarity check
        let goal_lower = goal_text.to_lowercase();
        for row in existing {
            let id: i64 = row.get("id");
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
                // Similar goal found — if abandoned or completed, respect that and skip
                if status != "active" {
                    return Ok(());
                }

                // Active goal — add progress note
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

        info!(
            goal = goal_text,
            episode_id = source_episode_id,
            "Created new goal"
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
        // Decay facts not recalled in the last 30 days
        let thirty_days_ago = (Utc::now() - chrono::Duration::days(30)).to_rfc3339();

        sqlx::query(
            "UPDATE facts SET recall_count = MAX(0, recall_count - 1)
             WHERE recall_count > 0 AND (last_recalled_at IS NULL OR last_recalled_at < ?)",
        )
        .bind(&thirty_days_ago)
        .execute(&self.pool)
        .await?;

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
            let id: i64 = row.get("id");
            let description: String = row.get("description");
            info!(
                goal_id = id,
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

        if idle_sessions.is_empty() {
            return Ok(());
        }

        info!(
            count = idle_sessions.len(),
            "Creating episodes for idle sessions"
        );

        for session_id in idle_sessions {
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

        for session_id in session_ids {
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

    /// Fetch messages for a session since its last episode's end_time.
    /// If no episode exists, returns all session messages up to `limit`.
    async fn fetch_session_messages_since_last_episode(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        let last_episode_end: Option<String> =
            sqlx::query_scalar("SELECT MAX(end_time) FROM episodes WHERE session_id = ?")
                .bind(session_id)
                .fetch_optional(&self.pool)
                .await?
                .flatten();

        let rows = if let Some(ref since) = last_episode_end {
            sqlx::query(
                "SELECT id, session_id, event_type, data, created_at
                 FROM events
                 WHERE session_id = ?
                   AND event_type IN ('user_message', 'assistant_response', 'tool_result')
                   AND created_at > ?
                 ORDER BY created_at ASC
                 LIMIT ?",
            )
            .bind(session_id)
            .bind(since)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT id, session_id, event_type, data, created_at
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
    #[allow(dead_code)]
    async fn fetch_session_messages(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        let rows = sqlx::query(
            "SELECT id, session_id, event_type, data, created_at
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
        let rows = sqlx::query(
            "SELECT id, session_id, event_type, data, created_at
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
        for row in rows {
            let event_id: i64 = row.get("id");
            let session_id: String = row.get("session_id");
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
                msg.importance = crate::memory::scoring::score_turn(&turn);
                messages.push(msg);
            }
        }

        Ok(messages)
    }
}

#[derive(Debug, Deserialize)]
struct ExtractedFact {
    category: String,
    key: String,
    value: String,
    privacy: Option<String>,
    /// For "people" category: the person's name
    person_name: Option<String>,
}

/// Derive a channel_id from a session_id string.
/// Session IDs follow patterns like:
///   - "slack:CHANNEL_ID:thread_ts" or "slack:CHANNEL_ID"
///   - "bot:telegram:CHAT_ID" or "telegram:CHAT_ID" or just "CHAT_ID"
///   - "discord:ch:ID" or "discord:dm:ID"
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::ProviderKind;
    use crate::llm_runtime::SharedLlmRuntime;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::testing::MockProvider;
    use crate::traits::store_prelude::*;
    use chrono::Utc;
    use std::sync::Arc;

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
}
