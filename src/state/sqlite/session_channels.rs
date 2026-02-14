use super::*;

#[async_trait]
impl crate::traits::SessionChannelStore for SqliteStateStore {
    async fn save_session_channel(
        &self,
        session_id: &str,
        channel_name: &str,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO session_channels (session_id, channel_name, updated_at)
             VALUES (?, ?, datetime('now'))
             ON CONFLICT(session_id) DO UPDATE SET
                channel_name = excluded.channel_name,
                updated_at = excluded.updated_at",
        )
        .bind(session_id)
        .bind(channel_name)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn load_session_channels(&self) -> anyhow::Result<Vec<(String, String)>> {
        let rows = sqlx::query("SELECT session_id, channel_name FROM session_channels")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .iter()
            .map(|r| {
                (
                    r.get::<String, _>("session_id"),
                    r.get::<String, _>("channel_name"),
                )
            })
            .collect())
    }
}
