use super::*;

#[async_trait]
impl crate::traits::OAuthStore for SqliteStateStore {
    async fn save_oauth_connection(
        &self,
        conn: &crate::traits::OAuthConnection,
    ) -> anyhow::Result<i64> {
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "INSERT INTO oauth_connections (service, auth_type, username, scopes, token_expires_at, created_at, updated_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(service) DO UPDATE SET \
             auth_type = excluded.auth_type, username = excluded.username, scopes = excluded.scopes, \
             token_expires_at = excluded.token_expires_at, updated_at = excluded.updated_at",
        )
        .bind(&conn.service)
        .bind(&conn.auth_type)
        .bind(&conn.username)
        .bind(&conn.scopes)
        .bind(&conn.token_expires_at)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn save_pending_oauth_flow(
        &self,
        flow: &crate::traits::PendingOAuthFlow,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO pending_oauth_flows (state, service, code_verifier, session_id, created_at) \
             VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT(state) DO UPDATE SET \
             service = excluded.service, code_verifier = excluded.code_verifier, \
             session_id = excluded.session_id, created_at = excluded.created_at",
        )
        .bind(&flow.state)
        .bind(&flow.service)
        .bind(&flow.code_verifier)
        .bind(&flow.session_id)
        .bind(&flow.created_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_oauth_connection(
        &self,
        service: &str,
    ) -> anyhow::Result<Option<crate::traits::OAuthConnection>> {
        let row = sqlx::query(
            "SELECT id, service, auth_type, username, scopes, token_expires_at, created_at, updated_at \
             FROM oauth_connections WHERE service = ?",
        )
        .bind(service)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| crate::traits::OAuthConnection {
            id: r.get("id"),
            service: r.get("service"),
            auth_type: r.get("auth_type"),
            username: r.try_get("username").unwrap_or(None),
            scopes: r.get("scopes"),
            token_expires_at: r.try_get("token_expires_at").unwrap_or(None),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
        }))
    }

    async fn get_pending_oauth_flow(
        &self,
        state: &str,
    ) -> anyhow::Result<Option<crate::traits::PendingOAuthFlow>> {
        let row = sqlx::query(
            "SELECT state, service, code_verifier, session_id, created_at \
             FROM pending_oauth_flows WHERE state = ?",
        )
        .bind(state)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| crate::traits::PendingOAuthFlow {
            state: r.get("state"),
            service: r.get("service"),
            code_verifier: r.try_get("code_verifier").unwrap_or(None),
            session_id: r.get("session_id"),
            created_at: r.get("created_at"),
        }))
    }

    async fn list_oauth_connections(&self) -> anyhow::Result<Vec<crate::traits::OAuthConnection>> {
        let rows = sqlx::query(
            "SELECT id, service, auth_type, username, scopes, token_expires_at, created_at, updated_at \
             FROM oauth_connections ORDER BY service ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| crate::traits::OAuthConnection {
                id: r.get("id"),
                service: r.get("service"),
                auth_type: r.get("auth_type"),
                username: r.try_get("username").unwrap_or(None),
                scopes: r.get("scopes"),
                token_expires_at: r.try_get("token_expires_at").unwrap_or(None),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
            })
            .collect())
    }

    async fn list_pending_oauth_flows(
        &self,
    ) -> anyhow::Result<Vec<crate::traits::PendingOAuthFlow>> {
        let rows = sqlx::query(
            "SELECT state, service, code_verifier, session_id, created_at \
             FROM pending_oauth_flows ORDER BY created_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| crate::traits::PendingOAuthFlow {
                state: r.get("state"),
                service: r.get("service"),
                code_verifier: r.try_get("code_verifier").unwrap_or(None),
                session_id: r.get("session_id"),
                created_at: r.get("created_at"),
            })
            .collect())
    }

    async fn delete_oauth_connection(&self, service: &str) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM oauth_connections WHERE service = ?")
            .bind(service)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn delete_pending_oauth_flow(&self, state: &str) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM pending_oauth_flows WHERE state = ?")
            .bind(state)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn update_oauth_token_expiry(
        &self,
        service: &str,
        expires_at: Option<&str>,
    ) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "UPDATE oauth_connections SET token_expires_at = ?, updated_at = ? WHERE service = ?",
        )
        .bind(expires_at)
        .bind(&now)
        .bind(service)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
