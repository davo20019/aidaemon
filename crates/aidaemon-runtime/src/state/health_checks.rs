use super::*;

#[async_trait]
impl crate::state::types::HealthCheckStore for SqliteStateStore {
    async fn health_check(&self) -> anyhow::Result<()> {
        sqlx::query("SELECT 1").execute(&self.pool).await?;
        Ok(())
    }
}
