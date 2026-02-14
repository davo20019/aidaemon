use super::*;

#[async_trait]
impl crate::traits::LegacyGoalStore for SqliteStateStore {
    async fn get_active_goals(&self) -> anyhow::Result<Vec<Goal>> {
        SqliteStateStore::get_active_goals(self).await
    }

    async fn update_goal(
        &self,
        goal_id: i64,
        status: Option<&str>,
        progress_note: Option<&str>,
    ) -> anyhow::Result<()> {
        SqliteStateStore::update_goal(self, goal_id, status, progress_note).await
    }
}
