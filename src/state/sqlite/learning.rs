use super::*;

use aidaemon_memory::{MemoryError, Result as MemoryResult};

/// The daemon store implements the SDK memory crate's learning contract.
#[async_trait]
impl crate::traits::LearningStore for SqliteStateStore {
    async fn get_behavior_patterns(
        &self,
        min_confidence: f32,
    ) -> MemoryResult<Vec<BehaviorPattern>> {
        SqliteStateStore::get_behavior_patterns(self, min_confidence)
            .await
            .map_err(store_error)
    }

    async fn record_behavior_pattern(
        &self,
        pattern_type: &str,
        description: &str,
        trigger_context: Option<&str>,
        action: Option<&str>,
        confidence_hint: f32,
        occurrence_delta: i32,
    ) -> MemoryResult<()> {
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
        .map_err(store_error)
    }

    async fn get_relevant_procedures(
        &self,
        query: &str,
        limit: usize,
    ) -> MemoryResult<Vec<Procedure>> {
        SqliteStateStore::get_relevant_procedures(self, query, limit)
            .await
            .map_err(store_error)
    }

    async fn get_relevant_error_solutions(
        &self,
        error: &str,
        limit: usize,
    ) -> MemoryResult<Vec<ErrorSolution>> {
        SqliteStateStore::get_relevant_error_solutions(self, error, limit)
            .await
            .map_err(store_error)
    }

    async fn get_user_profile(&self) -> MemoryResult<Option<UserProfile>> {
        SqliteStateStore::get_user_profile(self)
            .await
            .map(Some)
            .map_err(store_error)
    }

    async fn increment_expertise(
        &self,
        domain: &str,
        success: bool,
        error: Option<&str>,
    ) -> MemoryResult<()> {
        SqliteStateStore::increment_expertise(self, domain, success, error)
            .await
            .map_err(store_error)
    }

    async fn upsert_procedure(&self, procedure: &Procedure) -> MemoryResult<i64> {
        SqliteStateStore::insert_procedure(self, procedure)
            .await
            .map_err(store_error)
    }

    async fn update_procedure_outcome(
        &self,
        procedure_id: i64,
        success: bool,
        duration: Option<f32>,
    ) -> MemoryResult<()> {
        SqliteStateStore::update_procedure(self, procedure_id, success, None, duration)
            .await
            .map_err(store_error)
    }

    async fn insert_error_solution(&self, solution: &ErrorSolution) -> MemoryResult<i64> {
        SqliteStateStore::insert_error_solution(self, solution)
            .await
            .map_err(store_error)
    }

    async fn update_error_solution_outcome(
        &self,
        solution_id: i64,
        success: bool,
    ) -> MemoryResult<()> {
        SqliteStateStore::update_error_solution(self, solution_id, success)
            .await
            .map_err(store_error)
    }
}

/// The daemon store's errors as the SDK memory contract reports them.
fn store_error(error: anyhow::Error) -> MemoryError {
    MemoryError::Store(format!("{error:#}"))
}
