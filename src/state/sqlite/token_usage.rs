use super::*;

#[async_trait]
impl crate::traits::TokenUsageStore for SqliteStateStore {
    async fn record_token_usage(&self, session_id: &str, usage: &TokenUsage) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO token_usage (session_id, model, input_tokens, output_tokens, created_at)
             VALUES (?, ?, ?, ?, datetime('now'))",
        )
        .bind(session_id)
        .bind(&usage.model)
        .bind(usage.input_tokens as i64)
        .bind(usage.output_tokens as i64)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_token_usage_since(&self, since: &str) -> anyhow::Result<Vec<TokenUsageRecord>> {
        let rows = sqlx::query(
            "SELECT model, input_tokens, output_tokens, created_at
             FROM token_usage WHERE created_at >= ? ORDER BY created_at DESC",
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            records.push(TokenUsageRecord {
                model: row.get("model"),
                input_tokens: row.get("input_tokens"),
                output_tokens: row.get("output_tokens"),
                created_at: row.get("created_at"),
            });
        }
        Ok(records)
    }

    async fn get_token_usage_by_session(
        &self,
        since: &str,
    ) -> anyhow::Result<Vec<(String, i64, i64, i64)>> {
        let rows = sqlx::query(
            "SELECT session_id, SUM(input_tokens) as total_input, \
             SUM(output_tokens) as total_output, COUNT(*) as request_count \
             FROM token_usage WHERE created_at >= ? \
             GROUP BY session_id ORDER BY (total_input + total_output) DESC",
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            results.push((
                row.get::<String, _>("session_id"),
                row.get::<i64, _>("total_input"),
                row.get::<i64, _>("total_output"),
                row.get::<i64, _>("request_count"),
            ));
        }
        Ok(results)
    }
}
