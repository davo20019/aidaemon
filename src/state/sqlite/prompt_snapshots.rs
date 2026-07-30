use super::*;

#[async_trait]
impl crate::traits::PromptSnapshotStore for SqliteStateStore {
    async fn save_prompt_snapshot(&self, hash: &str, content: &str) -> anyhow::Result<()> {
        const MAX_PROMPT_SNAPSHOTS: i64 = 500;
        sqlx::query(
            "INSERT OR IGNORE INTO prompt_snapshots (hash, content, created_at) VALUES (?, ?, ?)",
        )
        .bind(hash)
        .bind(content)
        .bind(chrono::Utc::now().to_rfc3339())
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "DELETE FROM prompt_snapshots
             WHERE hash NOT IN (
                 SELECT hash FROM prompt_snapshots
                 ORDER BY created_at DESC, hash DESC
                 LIMIT ?
             )",
        )
        .bind(MAX_PROMPT_SNAPSHOTS)
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
