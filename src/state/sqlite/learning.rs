use super::*;

#[async_trait]
impl crate::traits::LearningStore for SqliteStateStore {
    async fn get_behavior_patterns(
        &self,
        min_confidence: f32,
    ) -> anyhow::Result<Vec<BehaviorPattern>> {
        SqliteStateStore::get_behavior_patterns(self, min_confidence).await
    }

    async fn record_behavior_pattern(
        &self,
        pattern_type: &str,
        description: &str,
        trigger_context: Option<&str>,
        action: Option<&str>,
        confidence_hint: f32,
        occurrence_delta: i32,
    ) -> anyhow::Result<()> {
        SqliteStateStore::record_behavior_pattern(
            self,
            pattern_type,
            description,
            trigger_context,
            action,
            confidence_hint,
            occurrence_delta,
        )
        .await
    }

    async fn get_relevant_procedures(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Procedure>> {
        SqliteStateStore::get_relevant_procedures(self, query, limit).await
    }

    async fn get_relevant_error_solutions(
        &self,
        error: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ErrorSolution>> {
        SqliteStateStore::get_relevant_error_solutions(self, error, limit).await
    }

    async fn get_all_expertise(&self) -> anyhow::Result<Vec<Expertise>> {
        SqliteStateStore::get_all_expertise(self).await
    }

    async fn get_user_profile(&self) -> anyhow::Result<Option<UserProfile>> {
        Ok(Some(SqliteStateStore::get_user_profile(self).await?))
    }

    async fn get_trusted_command_patterns(&self) -> anyhow::Result<Vec<(String, i32)>> {
        SqliteStateStore::get_trusted_command_patterns(self).await
    }

    async fn increment_expertise(
        &self,
        domain: &str,
        success: bool,
        error: Option<&str>,
    ) -> anyhow::Result<()> {
        SqliteStateStore::increment_expertise(self, domain, success, error).await
    }

    async fn upsert_procedure(&self, procedure: &Procedure) -> anyhow::Result<i64> {
        SqliteStateStore::insert_procedure(self, procedure).await
    }

    async fn update_procedure_outcome(
        &self,
        procedure_id: i64,
        success: bool,
        duration: Option<f32>,
    ) -> anyhow::Result<()> {
        SqliteStateStore::update_procedure(self, procedure_id, success, None, duration).await
    }

    async fn insert_error_solution(&self, solution: &ErrorSolution) -> anyhow::Result<i64> {
        SqliteStateStore::insert_error_solution(self, solution).await
    }

    async fn update_error_solution_outcome(
        &self,
        solution_id: i64,
        success: bool,
    ) -> anyhow::Result<()> {
        SqliteStateStore::update_error_solution(self, solution_id, success).await
    }
}
