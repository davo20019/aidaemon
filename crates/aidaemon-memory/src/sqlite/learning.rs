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

    async fn record_procedure_observation(&self, procedure: &Procedure) -> MemoryResult<()> {
        self.observe_procedure(procedure).await.map_err(store_error)
    }

    async fn record_error_solution_observation(
        &self,
        solution: &ErrorSolution,
    ) -> MemoryResult<()> {
        self.observe_error_solution(solution)
            .await
            .map_err(store_error)
    }

    async fn record_task_outcome(&self, domain: &str, success: bool) -> MemoryResult<()> {
        self.observe_task_outcome(domain, success)
            .await
            .map_err(store_error)
    }

    async fn procedure_trigger_embeddings(&self) -> MemoryResult<Vec<Vec<f32>>> {
        let blobs: Vec<Vec<u8>> = sqlx::query_scalar(
            "SELECT trigger_embedding FROM procedures WHERE trigger_embedding IS NOT NULL",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|error| store_error(error.into()))?;
        Ok(blobs
            .iter()
            .filter_map(|blob| decode_embedding(blob).ok())
            .collect())
    }

    async fn set_procedure_trigger_embedding(
        &self,
        name: &str,
        embedding: &[f32],
    ) -> MemoryResult<()> {
        sqlx::query("UPDATE procedures SET trigger_embedding = ? WHERE name = ?")
            .bind(encode_embedding(embedding))
            .bind(name)
            .execute(&self.pool)
            .await
            .map_err(|error| store_error(error.into()))?;
        Ok(())
    }

    async fn procedures_missing_embedding(&self, limit: usize) -> MemoryResult<Vec<(i64, String)>> {
        sqlx::query_as(
            "SELECT id, trigger_pattern FROM procedures p
             WHERE trigger_pattern IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1 FROM memory_embeddings me
                   WHERE me.owner_type = 'procedure' AND me.owner_id = CAST(p.id AS TEXT)
                     AND me.embedding_purpose = 'trigger' AND me.embedding_model = ?
                     AND me.stale_at IS NULL
               )
             LIMIT ?",
        )
        .bind(self.embedding_service.model_id())
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| store_error(error.into()))
    }

    async fn save_procedure_embedding(
        &self,
        id: i64,
        trigger: &str,
        embedding: Vec<f32>,
    ) -> MemoryResult<()> {
        let _ = sqlx::query("UPDATE procedures SET trigger_embedding = ? WHERE id = ?")
            .bind(encode_embedding(&embedding))
            .bind(id)
            .execute(&self.pool)
            .await;
        self.upsert_owner_embedding("procedure", id, "trigger", trigger, embedding)
            .await
            .map_err(store_error)
    }

    async fn error_solutions_missing_embedding(
        &self,
        limit: usize,
    ) -> MemoryResult<Vec<(i64, String)>> {
        sqlx::query_as(
            "SELECT id, error_pattern FROM error_solutions e
             WHERE error_pattern IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1 FROM memory_embeddings me
                   WHERE me.owner_type = 'error_solution' AND me.owner_id = CAST(e.id AS TEXT)
                     AND me.embedding_purpose = 'error' AND me.embedding_model = ?
                     AND me.stale_at IS NULL
               )
             LIMIT ?",
        )
        .bind(self.embedding_service.model_id())
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| store_error(error.into()))
    }

    async fn save_error_solution_embedding(
        &self,
        id: i64,
        pattern: &str,
        embedding: Vec<f32>,
    ) -> MemoryResult<()> {
        let _ = sqlx::query("UPDATE error_solutions SET error_embedding = ? WHERE id = ?")
            .bind(encode_embedding(&embedding))
            .bind(id)
            .execute(&self.pool)
            .await;
        self.upsert_owner_embedding("error_solution", id, "error", pattern, embedding)
            .await
            .map_err(store_error)
    }
}

/// What finished tasks taught, written the way the event consolidation
/// always has, so existing rows keep their meaning.
impl SqliteMemory {
    async fn observe_procedure(&self, procedure: &Procedure) -> anyhow::Result<()> {
        let steps_json = serde_json::to_string(&procedure.steps)?;
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            r#"
            INSERT INTO procedures (name, trigger_pattern, steps, success_count, failure_count,
                                    avg_duration_secs, last_used_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                success_count = success_count + 1,
                last_used_at = excluded.last_used_at,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&procedure.name)
        .bind(&procedure.trigger_pattern)
        .bind(&steps_json)
        .bind(procedure.success_count)
        .bind(procedure.failure_count)
        .bind(procedure.avg_duration_secs)
        .bind(&now)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn observe_error_solution(&self, solution: &ErrorSolution) -> anyhow::Result<()> {
        let steps_json = solution
            .solution_steps
            .as_ref()
            .map(|steps| serde_json::to_string(steps).unwrap_or_default());
        let domain = solution.domain.clone().unwrap_or_default();
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            r#"
            INSERT INTO error_solutions (error_pattern, domain, solution_summary, solution_steps,
                                         success_count, failure_count, last_used_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(error_pattern, domain, solution_summary) DO UPDATE SET
                solution_steps = COALESCE(excluded.solution_steps, error_solutions.solution_steps),
                success_count = error_solutions.success_count + excluded.success_count,
                failure_count = error_solutions.failure_count + excluded.failure_count,
                last_used_at = excluded.last_used_at
            "#,
        )
        .bind(&solution.error_pattern)
        .bind(&domain)
        .bind(&solution.solution_summary)
        .bind(&steps_json)
        .bind(solution.success_count)
        .bind(solution.failure_count)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn observe_task_outcome(&self, domain: &str, success: bool) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        let outcome_column = if success {
            "tasks_succeeded"
        } else {
            "tasks_failed"
        };
        let updated = sqlx::query(&format!(
            "UPDATE expertise
             SET tasks_attempted = tasks_attempted + 1,
                 {outcome_column} = {outcome_column} + 1,
                 last_task_at = ?,
                 updated_at = ?
             WHERE domain = ?"
        ))
        .bind(&now)
        .bind(&now)
        .bind(domain)
        .execute(&self.pool)
        .await?;
        if updated.rows_affected() == 0 {
            sqlx::query(
                "INSERT INTO expertise (domain, tasks_attempted, tasks_succeeded, tasks_failed,
                                        current_level, confidence_score, last_task_at, created_at, updated_at)
                 VALUES (?, 1, ?, ?, 'novice', 0.1, ?, ?, ?)",
            )
            .bind(domain)
            .bind(i32::from(success))
            .bind(i32::from(!success))
            .bind(&now)
            .bind(&now)
            .bind(&now)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
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
