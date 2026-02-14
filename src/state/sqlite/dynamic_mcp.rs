use super::*;

#[async_trait]
impl crate::traits::DynamicMcpServerStore for SqliteStateStore {
    async fn save_dynamic_mcp_server(
        &self,
        server: &crate::traits::DynamicMcpServer,
    ) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO dynamic_mcp_servers (name, command, args_json, env_keys_json, triggers_json, enabled, created_at)
             VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
             ON CONFLICT(name) DO UPDATE SET command=excluded.command, args_json=excluded.args_json,
             env_keys_json=excluded.env_keys_json, triggers_json=excluded.triggers_json, enabled=excluded.enabled"
        )
        .bind(&server.name)
        .bind(&server.command)
        .bind(&server.args_json)
        .bind(&server.env_keys_json)
        .bind(&server.triggers_json)
        .bind(server.enabled)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn list_dynamic_mcp_servers(
        &self,
    ) -> anyhow::Result<Vec<crate::traits::DynamicMcpServer>> {
        let rows = sqlx::query(
            "SELECT id, name, command, args_json, env_keys_json, triggers_json, enabled, created_at
             FROM dynamic_mcp_servers ORDER BY created_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut servers = Vec::new();
        for row in rows {
            servers.push(crate::traits::DynamicMcpServer {
                id: row.get::<i64, _>("id"),
                name: row.get::<String, _>("name"),
                command: row.get::<String, _>("command"),
                args_json: row.get::<String, _>("args_json"),
                env_keys_json: row.get::<String, _>("env_keys_json"),
                triggers_json: row.get::<String, _>("triggers_json"),
                enabled: row.get::<bool, _>("enabled"),
                created_at: row.get::<String, _>("created_at"),
            });
        }
        Ok(servers)
    }

    async fn delete_dynamic_mcp_server(&self, id: i64) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM dynamic_mcp_servers WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn update_dynamic_mcp_server(
        &self,
        server: &crate::traits::DynamicMcpServer,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE dynamic_mcp_servers SET command = ?, args_json = ?, env_keys_json = ?, triggers_json = ?, enabled = ? WHERE id = ?"
        )
        .bind(&server.command)
        .bind(&server.args_json)
        .bind(&server.env_keys_json)
        .bind(&server.triggers_json)
        .bind(server.enabled)
        .bind(server.id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
