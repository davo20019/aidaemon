use super::*;

#[async_trait]
impl crate::state::types::DynamicCliAgentStore for SqliteStateStore {
    async fn save_dynamic_cli_agent(
        &self,
        agent: &crate::state::types::DynamicCliAgent,
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

    async fn list_dynamic_cli_agents(
        &self,
    ) -> anyhow::Result<Vec<crate::state::types::DynamicCliAgent>> {
        let rows = sqlx::query(
            "SELECT id, name, command, args_json, description, timeout_secs, max_output_chars, enabled, created_at
             FROM dynamic_cli_agents ORDER BY created_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut agents = Vec::new();
        for row in rows {
            agents.push(crate::state::types::DynamicCliAgent {
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
        agent: &crate::state::types::DynamicCliAgent,
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
        task_id: Option<&str>,
        session_id: &str,
        agent_name: &str,
        prompt_summary: &str,
        working_dir: Option<&str>,
    ) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO cli_agent_invocations (task_id, session_id, agent_name, prompt_summary, working_dir, started_at)
             VALUES (?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(task_id)
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

    async fn note_cli_agent_run(
        &self,
        id: i64,
        run_id: &str,
        mode: Option<&str>,
    ) -> anyhow::Result<()> {
        sqlx::query("UPDATE cli_agent_invocations SET run_id = ?, mode = ? WHERE id = ?")
            .bind(run_id)
            .bind(mode)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn note_cli_agent_task(&self, id: i64, task: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE cli_agent_invocations SET task = ? WHERE id = ?")
            .bind(task)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn set_cli_agent_thread(&self, id: i64, thread: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE cli_agent_invocations SET thread_id = ? WHERE id = ?")
            .bind(thread)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn set_cli_agent_keeps_session(&self, id: i64, kept: bool) -> anyhow::Result<()> {
        sqlx::query("UPDATE cli_agent_invocations SET keeps_session = ? WHERE id = ?")
            .bind(i64::from(kept))
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn set_cli_agent_session(&self, id: i64, agent_session: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE cli_agent_invocations SET agent_session_id = ? WHERE id = ?")
            .bind(agent_session)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn record_cli_agent_event(
        &self,
        id: i64,
        seq: i64,
        kind: &str,
        data: &str,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO cli_agent_events (invocation_id, seq, kind, data) VALUES (?, ?, ?, ?)",
        )
        .bind(id)
        .bind(seq)
        .bind(kind)
        .bind(data)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn cli_agent_run(
        &self,
        run_id: &str,
    ) -> anyhow::Result<Option<aidaemon_tools::services::CliAgentRun>> {
        use sqlx::Row;
        let row = sqlx::query(
            "SELECT id, session_id, agent_name, agent_session_id, working_dir, prompt_summary, task,
                    output_summary, success, COALESCE(keeps_session, 0) AS keeps_session, thread_id
             FROM cli_agent_invocations WHERE run_id = ? ORDER BY id DESC LIMIT 1",
        )
        .bind(run_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|row| aidaemon_tools::services::CliAgentRun {
            invocation: row.get("id"),
            session_id: row
                .get::<Option<String>, _>("session_id")
                .unwrap_or_default(),
            agent_name: row.get("agent_name"),
            agent_session: row.get("agent_session_id"),
            working_dir: row.get("working_dir"),
            prompt_summary: row.get("prompt_summary"),
            task: row.get("task"),
            output_summary: row.get("output_summary"),
            success: row.get("success"),
            keeps_session: row.get::<i64, _>("keeps_session") != 0,
            thread: row.get("thread_id"),
        }))
    }

    async fn get_cli_agent_events(
        &self,
        id: i64,
    ) -> anyhow::Result<Vec<aidaemon_tools::services::CliAgentEvent>> {
        use sqlx::Row;
        let rows = sqlx::query(
            "SELECT seq, at, kind, data FROM cli_agent_events WHERE invocation_id = ? ORDER BY seq",
        )
        .bind(id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|row| aidaemon_tools::services::CliAgentEvent {
                seq: row.get("seq"),
                at: row.get("at"),
                kind: row.get("kind"),
                data: row.get("data"),
            })
            .collect())
    }

    async fn get_cli_agent_invocations(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::state::types::CliAgentInvocation>> {
        let rows = sqlx::query(
            "SELECT id, task_id, session_id, agent_name, prompt_summary, working_dir, started_at, completed_at, exit_code, output_summary, success, duration_secs
             FROM cli_agent_invocations ORDER BY started_at DESC LIMIT ?",
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut invocations = Vec::new();
        for row in rows {
            invocations.push(crate::state::types::CliAgentInvocation {
                id: row.get::<i64, _>("id"),
                task_id: row.get::<Option<String>, _>("task_id"),
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
