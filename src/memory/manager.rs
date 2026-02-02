use std::sync::Arc;
use std::time::Duration;
use sqlx::{SqlitePool, Row};
use tracing::{error, info, warn};
use serde::Deserialize;
use serde_json::json;
use crate::memory::embeddings::EmbeddingService;
use crate::traits::ModelProvider;

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
                                    "Failed to parse consolidation response for session {}: {}",
                                    session_id, e
                                );
                            }
                        }
                    }

                    // Mark all messages in this batch as consolidated regardless of parse outcome
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
        sqlx::query(
            "INSERT INTO facts (category, key, value, source, created_at, updated_at)
             VALUES (?, ?, ?, 'consolidation', ?, ?)
             ON CONFLICT(category, key) DO UPDATE SET value = excluded.value, source = excluded.source, updated_at = excluded.updated_at"
        )
        .bind(category)
        .bind(key)
        .bind(value)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct ExtractedFact {
    category: String,
    key: String,
    value: String,
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
