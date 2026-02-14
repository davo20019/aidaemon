use super::*;

#[async_trait]
impl crate::traits::DynamicCliAgentStore for SqliteStateStore {
    async fn save_dynamic_cli_agent(
        &self,
        agent: &crate::traits::DynamicCliAgent,
    ) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO dynamic_cli_agents (name, command, args_json, description, timeout_secs, max_output_chars, enabled, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
             ON CONFLICT(name) DO UPDATE SET command=excluded.command, args_json=excluded.args_json,
             description=excluded.description, timeout_secs=excluded.timeout_secs,
             max_output_chars=excluded.max_output_chars, enabled=excluded.enabled",
        )
        .bind(&agent.name)
        .bind(&agent.command)
        .bind(&agent.args_json)
        .bind(&agent.description)
        .bind(agent.timeout_secs.map(|v| v as i64))
        .bind(agent.max_output_chars.map(|v| v as i64))
        .bind(agent.enabled)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn list_dynamic_cli_agents(&self) -> anyhow::Result<Vec<crate::traits::DynamicCliAgent>> {
        let rows = sqlx::query(
            "SELECT id, name, command, args_json, description, timeout_secs, max_output_chars, enabled, created_at
             FROM dynamic_cli_agents ORDER BY created_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut agents = Vec::new();
        for row in rows {
            agents.push(crate::traits::DynamicCliAgent {
                id: row.get::<i64, _>("id"),
                name: row.get::<String, _>("name"),
                command: row.get::<String, _>("command"),
                args_json: row.get::<String, _>("args_json"),
                description: row.get::<String, _>("description"),
                timeout_secs: row.get::<Option<i64>, _>("timeout_secs").map(|v| v as u64),
                max_output_chars: row
                    .get::<Option<i64>, _>("max_output_chars")
                    .map(|v| v as usize),
                enabled: row.get::<bool, _>("enabled"),
                created_at: row.get::<String, _>("created_at"),
            });
        }
        Ok(agents)
    }

    async fn delete_dynamic_cli_agent(&self, id: i64) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM dynamic_cli_agents WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn update_dynamic_cli_agent(
        &self,
        agent: &crate::traits::DynamicCliAgent,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE dynamic_cli_agents SET command = ?, args_json = ?, description = ?, timeout_secs = ?, max_output_chars = ?, enabled = ? WHERE id = ?",
        )
        .bind(&agent.command)
        .bind(&agent.args_json)
        .bind(&agent.description)
        .bind(agent.timeout_secs.map(|v| v as i64))
        .bind(agent.max_output_chars.map(|v| v as i64))
        .bind(agent.enabled)
        .bind(agent.id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn log_cli_agent_start(
        &self,
        session_id: &str,
        agent_name: &str,
        prompt_summary: &str,
        working_dir: Option<&str>,
    ) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO cli_agent_invocations (session_id, agent_name, prompt_summary, working_dir, started_at)
             VALUES (?, ?, ?, ?, datetime('now'))",
        )
        .bind(session_id)
        .bind(agent_name)
        .bind(prompt_summary)
        .bind(working_dir)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn log_cli_agent_complete(
        &self,
        id: i64,
        exit_code: Option<i32>,
        output_summary: &str,
        success: bool,
        duration_secs: f64,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE cli_agent_invocations SET completed_at = datetime('now'), exit_code = ?, output_summary = ?, success = ?, duration_secs = ? WHERE id = ?",
        )
        .bind(exit_code)
        .bind(output_summary)
        .bind(success)
        .bind(duration_secs)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_cli_agent_invocations(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::traits::CliAgentInvocation>> {
        let rows = sqlx::query(
            "SELECT id, session_id, agent_name, prompt_summary, working_dir, started_at, completed_at, exit_code, output_summary, success, duration_secs
             FROM cli_agent_invocations ORDER BY started_at DESC LIMIT ?",
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut invocations = Vec::new();
        for row in rows {
            invocations.push(crate::traits::CliAgentInvocation {
                id: row.get::<i64, _>("id"),
                session_id: row.get::<String, _>("session_id"),
                agent_name: row.get::<String, _>("agent_name"),
                prompt_summary: row.get::<String, _>("prompt_summary"),
                working_dir: row.get::<Option<String>, _>("working_dir"),
                started_at: row.get::<String, _>("started_at"),
                completed_at: row.get::<Option<String>, _>("completed_at"),
                exit_code: row.get::<Option<i32>, _>("exit_code"),
                output_summary: row.get::<Option<String>, _>("output_summary"),
                success: row.get::<Option<bool>, _>("success"),
                duration_secs: row.get::<Option<f64>, _>("duration_secs"),
            });
        }
        Ok(invocations)
    }

    async fn cleanup_stale_cli_agent_invocations(&self, max_age_hours: i64) -> anyhow::Result<u64> {
        if max_age_hours <= 0 {
            return Ok(0);
        }
        let note = format!(
            "Auto-closed stale invocation (no completion recorded; older than {} hour(s)).",
            max_age_hours
        );
        let result = sqlx::query(
            "UPDATE cli_agent_invocations
             SET completed_at = datetime('now'),
                 exit_code = NULL,
                 output_summary = ?,
                 success = 0,
                 duration_secs = (julianday('now') - julianday(started_at)) * 86400.0
             WHERE completed_at IS NULL
               AND started_at < datetime('now', '-' || ? || ' hours')",
        )
        .bind(note)
        .bind(max_age_hours)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }
}
