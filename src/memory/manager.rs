use std::sync::Arc;
use std::time::Duration;
use sqlx::{SqlitePool, Row};
use tracing::{error, info};
use crate::memory::embeddings::EmbeddingService;

pub struct MemoryManager {
    pool: SqlitePool,
    embedding_service: Arc<EmbeddingService>,
}

impl MemoryManager {
    pub fn new(pool: SqlitePool, embedding_service: Arc<EmbeddingService>) -> Self {
        Self {
            pool,
            embedding_service,
        }
    }

    pub fn start_background_tasks(self: Arc<Self>) {
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
}
