use super::*;

#[async_trait]
impl crate::traits::PromptSnapshotStore for SqliteStateStore {
    async fn save_prompt_snapshot(&self, hash: &str, content: &str) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR IGNORE INTO prompt_snapshots (hash, content, created_at) VALUES (?, ?, ?)",
        )
        .bind(hash)
        .bind(content)
        .bind(chrono::Utc::now().to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_prompt_snapshot(&self, hash: &str) -> anyhow::Result<Option<String>> {
        let row = sqlx::query("SELECT content FROM prompt_snapshots WHERE hash = ?")
            .bind(hash)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| r.get("content")))
    }
}
