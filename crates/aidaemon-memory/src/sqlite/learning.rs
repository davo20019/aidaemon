use super::*;

use crate::error::{MemoryError, Result as MemoryResult};

/// The daemon store implements the SDK memory crate's learning contract.
#[async_trait]
impl LearningStore for SqliteMemory {
    async fn get_behavior_patterns(
        &self,
        min_confidence: f32,
    ) -> MemoryResult<Vec<BehaviorPattern>> {
        SqliteMemory::get_behavior_patterns(self, min_confidence)
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
        SqliteMemory::record_behavior_pattern(
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
        SqliteMemory::get_relevant_procedures(self, query, limit)
            .await
            .map_err(store_error)
    }

    async fn get_relevant_error_solutions(
        &self,
        error: &str,
        limit: usize,
    ) -> MemoryResult<Vec<ErrorSolution>> {
        SqliteMemory::get_relevant_error_solutions(self, error, limit)
            .await
            .map_err(store_error)
    }

    async fn get_user_profile(&self) -> MemoryResult<Option<UserProfile>> {
        SqliteMemory::get_user_profile(self)
            .await
            .map(Some)
            .map_err(store_error)
    }

    async fn update_user_profile(&self, profile: &UserProfile) -> MemoryResult<()> {
        SqliteMemory::update_user_profile(self, profile)
            .await
            .map_err(store_error)
    }

    async fn increment_expertise(
        &self,
        domain: &str,
        success: bool,
        error: Option<&str>,
    ) -> MemoryResult<()> {
        SqliteMemory::increment_expertise(self, domain, success, error)
            .await
            .map_err(store_error)
    }

    async fn upsert_procedure(&self, procedure: &Procedure) -> MemoryResult<i64> {
        SqliteMemory::insert_procedure(self, procedure)
            .await
            .map_err(store_error)
    }

    async fn update_procedure_outcome(
        &self,
        procedure_id: i64,
        success: bool,
        duration: Option<f32>,
    ) -> MemoryResult<()> {
        SqliteMemory::update_procedure(self, procedure_id, success, None, duration)
            .await
            .map_err(store_error)
    }

    async fn insert_error_solution(&self, solution: &ErrorSolution) -> MemoryResult<i64> {
        SqliteMemory::insert_error_solution(self, solution)
            .await
            .map_err(store_error)
    }

    async fn update_error_solution_outcome(
        &self,
        solution_id: i64,
        success: bool,
    ) -> MemoryResult<()> {
        SqliteMemory::update_error_solution(self, solution_id, success)
            .await
            .map_err(store_error)
    }
}

/// Fading of unrecalled facts, episodes and behavior patterns.
#[async_trait]
impl crate::stores::MemoryDecay for SqliteMemory {
    async fn decay(
        &self,
        policy: &crate::stores::DecayPolicy,
        now: chrono::DateTime<chrono::Utc>,
    ) -> anyhow::Result<()> {
        let fact_cutoff = (now - chrono::Duration::days(policy.fact_idle_days)).to_rfc3339();
        sqlx::query(
            "UPDATE facts SET recall_count = MAX(0, recall_count - MAX(1, recall_count / ?))
             WHERE recall_count > 0 AND (last_recalled_at IS NULL OR last_recalled_at < ?)",
        )
        .bind(policy.fact_decay_divisor)
        .bind(&fact_cutoff)
        .execute(&self.pool)
        .await?;
        let episode_cutoff = (now - chrono::Duration::days(policy.episode_idle_days)).to_rfc3339();
        sqlx::query(
            "UPDATE episodes SET recall_count = MAX(0, recall_count - 1)
             WHERE recall_count > 0 AND (last_recalled_at IS NULL OR last_recalled_at < ?)",
        )
        .bind(&episode_cutoff)
        .execute(&self.pool)
        .await?;
        let pattern_cutoff = (now - chrono::Duration::days(policy.pattern_idle_days)).to_rfc3339();
        sqlx::query(
            "UPDATE behavior_patterns SET confidence = MAX(0.1, confidence - ?)
             WHERE confidence > 0.1 AND (last_seen_at IS NULL OR last_seen_at < ?)",
        )
        .bind(policy.pattern_step)
        .bind(&pattern_cutoff)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

/// The daemon store's errors as the SDK memory contract reports them.
fn store_error(error: anyhow::Error) -> MemoryError {
    MemoryError::Store(format!("{error:#}"))
}
