use super::*;

#[async_trait]
impl crate::traits::TokenUsageStore for SqliteStateStore {
    async fn record_token_usage(
        &self,
        session_id: &str,
        usage: &TokenUsage,
        call_id: Option<&str>,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO token_usage (
                session_id, model, input_tokens, output_tokens,
                cached_input_tokens, cache_creation_input_tokens, call_id, created_at
             )
             VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(session_id)
        .bind(&usage.model)
        .bind(usage.input_tokens as i64)
        .bind(usage.output_tokens as i64)
        .bind(usage.cached_input_tokens.map(i64::from))
        .bind(usage.cache_creation_input_tokens.map(i64::from))
        .bind(call_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_token_usage_since(&self, since: &str) -> anyhow::Result<Vec<TokenUsageRecord>> {
        let rows = sqlx::query(
            "SELECT model, input_tokens, output_tokens,
                    cached_input_tokens, cache_creation_input_tokens, call_id, created_at
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
                cached_input_tokens: row
                    .try_get::<Option<i64>, _>("cached_input_tokens")
                    .unwrap_or(None),
                cache_creation_input_tokens: row
                    .try_get::<Option<i64>, _>("cache_creation_input_tokens")
                    .unwrap_or(None),
                call_id: row.try_get::<Option<String>, _>("call_id").unwrap_or(None),
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

    async fn ensure_mandate_run_token_budget(
        &self,
        goal_run_id: &str,
        mandate_id: &str,
        mandate_version: i64,
        budget_per_cycle: i64,
    ) -> anyhow::Result<(i64, i64)> {
        anyhow::ensure!(
            budget_per_cycle > 0,
            "mandate cycle token budget must be positive"
        );
        anyhow::ensure!(
            mandate_version > 0,
            "mandate authority version must be positive"
        );
        let now = chrono::Utc::now().to_rfc3339();

        sqlx::query(
            "INSERT INTO mandate_run_token_budgets
                (goal_run_id, budget_per_cycle, tokens_used, created_at, updated_at)
             SELECT gr.id, ?, 0, ?, ?
             FROM goal_runs gr
             JOIN goals g ON g.id = gr.goal_id
             JOIN mandates m ON m.goal_id = gr.goal_id
             WHERE gr.id = ?
               AND gr.trigger_type = 'mandate'
               AND gr.status = 'running'
               AND m.id = ?
               AND m.version = ?
               AND m.status = 'active'
               AND m.confirmed_at IS NOT NULL
               AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
               AND g.budget_per_check = ?
             ON CONFLICT(goal_run_id) DO NOTHING",
        )
        .bind(budget_per_cycle)
        .bind(&now)
        .bind(&now)
        .bind(goal_run_id)
        .bind(mandate_id)
        .bind(mandate_version)
        .bind(&now)
        .bind(budget_per_cycle)
        .execute(&self.pool)
        .await?;

        let row = sqlx::query(
            "SELECT b.budget_per_cycle, b.tokens_used
             FROM mandate_run_token_budgets b
             JOIN goal_runs gr ON gr.id = b.goal_run_id
             JOIN mandates m ON m.goal_id = gr.goal_id
             WHERE b.goal_run_id = ?
               AND gr.trigger_type = 'mandate'
               AND gr.status = 'running'
               AND m.id = ?
               AND m.version = ?
               AND m.status = 'active'
               AND m.confirmed_at IS NOT NULL
               AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))",
        )
        .bind(goal_run_id)
        .bind(mandate_id)
        .bind(mandate_version)
        .bind(&now)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!("mandate run token budget fence is not live for run {goal_run_id}")
        })?;
        let stored_budget = row.get::<i64, _>("budget_per_cycle");
        let tokens_used = row.get::<i64, _>("tokens_used");
        anyhow::ensure!(
            stored_budget == budget_per_cycle,
            "mandate run token budget is immutable for run {goal_run_id} (stored {stored_budget}, requested {budget_per_cycle})"
        );
        Ok((stored_budget, tokens_used))
    }

    async fn try_acquire_mandate_run_token_lease(
        &self,
        goal_run_id: &str,
        mandate_id: &str,
        mandate_version: i64,
        lease_token: &str,
        lease_secs: i64,
    ) -> anyhow::Result<(bool, i64, i64)> {
        anyhow::ensure!(
            !lease_token.trim().is_empty(),
            "mandate token lease is required"
        );
        anyhow::ensure!(
            mandate_version > 0,
            "mandate authority version must be positive"
        );
        let lease_secs = lease_secs.clamp(1, 900);
        let now = chrono::Utc::now();
        let now_text = now.to_rfc3339();
        let expires_at = (now + chrono::Duration::seconds(lease_secs)).to_rfc3339();

        // A dispatched call pessimistically reserved the remaining balance
        // before I/O. If its lease expires, retain that exhausted balance and
        // clear only the stale call identity. Actual spend is unknowable.
        sqlx::query(
            "UPDATE mandate_run_token_budgets
             SET call_lease_token = NULL,
                 call_lease_expires_at = NULL,
                 call_dispatched = 0,
                 call_tokens_used_before = NULL,
                 updated_at = ?
             WHERE goal_run_id = ?
               AND call_lease_token IS NOT NULL
               AND call_dispatched = 1
               AND call_lease_expires_at IS NOT NULL
               AND julianday(call_lease_expires_at) <= julianday(?)",
        )
        .bind(&now_text)
        .bind(goal_run_id)
        .bind(&now_text)
        .execute(&self.pool)
        .await?;

        // A lease that expired before dispatch did not perform provider I/O;
        // recover it without charging or poisoning the cycle.
        sqlx::query(
            "UPDATE mandate_run_token_budgets
             SET call_lease_token = NULL,
                 call_lease_expires_at = NULL,
                 call_dispatched = 0,
                 call_tokens_used_before = NULL,
                 updated_at = ?
             WHERE goal_run_id = ?
               AND call_lease_token IS NOT NULL
               AND call_dispatched = 0
               AND call_lease_expires_at IS NOT NULL
               AND julianday(call_lease_expires_at) <= julianday(?)",
        )
        .bind(&now_text)
        .bind(goal_run_id)
        .bind(&now_text)
        .execute(&self.pool)
        .await?;

        let acquired = sqlx::query(
            "UPDATE mandate_run_token_budgets
             SET call_lease_token = ?, call_lease_expires_at = ?,
                 call_dispatched = 0, call_tokens_used_before = NULL,
                 updated_at = ?
             WHERE goal_run_id = ?
               AND tokens_used < budget_per_cycle
               AND call_lease_token IS NULL
               AND EXISTS (
                    SELECT 1
                    FROM goal_runs gr
                    JOIN mandates m ON m.goal_id = gr.goal_id
                    WHERE gr.id = mandate_run_token_budgets.goal_run_id
                      AND gr.trigger_type = 'mandate'
                      AND gr.status = 'running'
                      AND m.id = ?
                      AND m.version = ?
                      AND m.status = 'active'
                      AND m.confirmed_at IS NOT NULL
                      AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
               )
             RETURNING budget_per_cycle, tokens_used",
        )
        .bind(lease_token)
        .bind(&expires_at)
        .bind(&now_text)
        .bind(goal_run_id)
        .bind(mandate_id)
        .bind(mandate_version)
        .bind(&now_text)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = acquired {
            return Ok((
                true,
                row.get::<i64, _>("tokens_used"),
                row.get::<i64, _>("budget_per_cycle"),
            ));
        }

        let row = sqlx::query(
            "SELECT b.budget_per_cycle, b.tokens_used
             FROM mandate_run_token_budgets b
             JOIN goal_runs gr ON gr.id = b.goal_run_id
             JOIN mandates m ON m.goal_id = gr.goal_id
             WHERE b.goal_run_id = ?
               AND gr.trigger_type = 'mandate'
               AND gr.status = 'running'
               AND m.id = ?
               AND m.version = ?
               AND m.status = 'active'
               AND m.confirmed_at IS NOT NULL
               AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))",
        )
        .bind(goal_run_id)
        .bind(mandate_id)
        .bind(mandate_version)
        .bind(&now_text)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!("mandate run token lease fence is not live for run {goal_run_id}")
        })?;
        Ok((
            false,
            row.get::<i64, _>("tokens_used"),
            row.get::<i64, _>("budget_per_cycle"),
        ))
    }

    async fn mark_mandate_run_token_lease_dispatched(
        &self,
        goal_run_id: &str,
        mandate_id: &str,
        mandate_version: i64,
        lease_token: &str,
    ) -> anyhow::Result<bool> {
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE mandate_run_token_budgets
             SET call_tokens_used_before = tokens_used,
                 tokens_used = budget_per_cycle,
                 call_dispatched = 1,
                 updated_at = datetime('now')
             WHERE goal_run_id = ?
               AND call_lease_token = ?
               AND call_dispatched = 0
               AND call_tokens_used_before IS NULL
               AND call_lease_expires_at IS NOT NULL
               AND julianday(call_lease_expires_at) > julianday(?)
               AND EXISTS (
                    SELECT 1
                    FROM goal_runs gr
                    JOIN mandates m ON m.goal_id = gr.goal_id
                    WHERE gr.id = mandate_run_token_budgets.goal_run_id
                      AND gr.trigger_type = 'mandate'
                      AND gr.status = 'running'
                      AND m.id = ?
                      AND m.version = ?
                      AND m.status = 'active'
                      AND m.confirmed_at IS NOT NULL
                      AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
               )",
        )
        .bind(goal_run_id)
        .bind(lease_token)
        .bind(&now)
        .bind(mandate_id)
        .bind(mandate_version)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() == 1)
    }

    async fn settle_mandate_run_token_lease(
        &self,
        goal_run_id: &str,
        lease_token: &str,
        delta_tokens: i64,
    ) -> anyhow::Result<(i64, i64)> {
        anyhow::ensure!(delta_tokens >= 0, "mandate token charge cannot be negative");
        let now = chrono::Utc::now().to_rfc3339();
        let row = sqlx::query(
            "UPDATE mandate_run_token_budgets
             SET tokens_used = CASE
                    WHEN call_tokens_used_before >= 9223372036854775807 - ?
                    THEN 9223372036854775807
                    ELSE call_tokens_used_before + ?
                 END,
                 call_lease_token = NULL,
                 call_lease_expires_at = NULL,
                 call_dispatched = 0,
                 call_tokens_used_before = NULL,
                 updated_at = ?
             WHERE goal_run_id = ?
               AND call_lease_token = ?
               AND call_dispatched = 1
               AND call_tokens_used_before IS NOT NULL
             RETURNING budget_per_cycle, tokens_used",
        )
        .bind(delta_tokens)
        .bind(delta_tokens)
        .bind(&now)
        .bind(goal_run_id)
        .bind(lease_token)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "mandate run token lease was lost before settlement for run {goal_run_id}"
            )
        })?;
        Ok((
            row.get::<i64, _>("tokens_used"),
            row.get::<i64, _>("budget_per_cycle"),
        ))
    }

    async fn release_mandate_run_token_lease(
        &self,
        goal_run_id: &str,
        lease_token: &str,
    ) -> anyhow::Result<bool> {
        let result = sqlx::query(
            "UPDATE mandate_run_token_budgets
             SET call_lease_token = NULL, call_lease_expires_at = NULL,
                 call_dispatched = 0, call_tokens_used_before = NULL,
                 updated_at = datetime('now')
             WHERE goal_run_id = ? AND call_lease_token = ?
               AND call_dispatched = 0
               AND call_tokens_used_before IS NULL",
        )
        .bind(goal_run_id)
        .bind(lease_token)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() == 1)
    }
}
