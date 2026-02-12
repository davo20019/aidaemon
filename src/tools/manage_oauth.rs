use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::warn;

use crate::oauth::OAuthGateway;
use crate::traits::{StateStore, Tool};
use crate::types::StatusUpdate;

use std::sync::Arc;

pub struct ManageOAuthTool {
    gateway: OAuthGateway,
    state_store: Arc<dyn StateStore>,
}

impl ManageOAuthTool {
    pub fn new(gateway: OAuthGateway, state_store: Arc<dyn StateStore>) -> Self {
        Self {
            gateway,
            state_store,
        }
    }
}

#[async_trait]
impl Tool for ManageOAuthTool {
    fn name(&self) -> &str {
        "manage_oauth"
    }

    fn description(&self) -> &str {
        "Connect and manage OAuth services (Twitter, GitHub, etc.)"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_oauth",
            "description": "Connect and manage OAuth-authenticated external services. Use 'connect' to start an OAuth flow, 'list' to see connections, 'remove' to disconnect, 'set_credentials' to store app credentials, 'refresh' to refresh expired tokens, 'providers' to see available services.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["connect", "list", "remove", "set_credentials", "refresh", "providers"],
                        "description": "Action to perform"
                    },
                    "service": {
                        "type": "string",
                        "description": "Service name (e.g., 'twitter', 'github')"
                    },
                    "client_id": {
                        "type": "string",
                        "description": "OAuth client/app ID (for set_credentials action)"
                    },
                    "client_secret": {
                        "type": "string",
                        "description": "OAuth client/app secret (for set_credentials action)"
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        // Default call without status — connect won't work well here
        self.call_with_status(arguments, None).await
    }

    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments).unwrap_or(json!({}));
        let action = args["action"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: action"))?;

        match action {
            "providers" => self.handle_providers().await,
            "list" => self.handle_list().await,
            "connect" => {
                let service = args["service"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("'connect' requires 'service' parameter"))?;
                let session_id = args["_session_id"].as_str().unwrap_or("unknown");
                self.handle_connect(service, session_id, status_tx).await
            }
            "remove" => {
                let service = args["service"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("'remove' requires 'service' parameter"))?;
                self.handle_remove(service).await
            }
            "set_credentials" => {
                let service = args["service"]
                    .as_str()
                    .ok_or_else(|| {
                        anyhow::anyhow!("'set_credentials' requires 'service' parameter")
                    })?;
                let client_id = args["client_id"]
                    .as_str()
                    .ok_or_else(|| {
                        anyhow::anyhow!("'set_credentials' requires 'client_id' parameter")
                    })?;
                let client_secret = args["client_secret"]
                    .as_str()
                    .ok_or_else(|| {
                        anyhow::anyhow!("'set_credentials' requires 'client_secret' parameter")
                    })?;
                self.handle_set_credentials(service, client_id, client_secret)
                    .await
            }
            "refresh" => {
                let service = args["service"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("'refresh' requires 'service' parameter"))?;
                self.handle_refresh(service).await
            }
            _ => Ok(format!(
                "Unknown action '{}'. Use: connect, list, remove, set_credentials, refresh, providers",
                action
            )),
        }
    }
}

impl ManageOAuthTool {
    async fn handle_providers(&self) -> anyhow::Result<String> {
        let providers = self.gateway.list_providers().await;
        if providers.is_empty() {
            return Ok("No OAuth providers configured. Enable OAuth in config.toml and register providers.".to_string());
        }
        let mut result = String::from("Available OAuth providers:\n");
        for (name, display_name) in &providers {
            let has_creds = OAuthGateway::has_credentials(name);
            let cred_status = if has_creds {
                "credentials set"
            } else {
                "credentials needed"
            };
            result.push_str(&format!(
                "  - {} ({}) [{}]\n",
                display_name, name, cred_status
            ));
        }
        result.push_str(
            "\nTo connect: first set credentials with 'set_credentials', then use 'connect'.",
        );
        Ok(result)
    }

    async fn handle_list(&self) -> anyhow::Result<String> {
        let connections = self.state_store.list_oauth_connections().await?;
        if connections.is_empty() {
            return Ok("No OAuth connections. Use 'connect' to link a service.".to_string());
        }
        let mut result = String::from("Connected OAuth services:\n");
        for conn in &connections {
            let username = conn
                .username
                .as_deref()
                .map(|u| format!(" ({})", u))
                .unwrap_or_default();
            let expires = conn
                .token_expires_at
                .as_deref()
                .map(|e| format!(" [expires: {}]", e))
                .unwrap_or_default();
            result.push_str(&format!(
                "  - {}{} [{}]{}\n",
                conn.service, username, conn.auth_type, expires
            ));
        }
        Ok(result)
    }

    async fn handle_connect(
        &self,
        service: &str,
        session_id: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        // Start OAuth flow
        let (authorize_url, result_rx) =
            self.gateway.start_oauth2_flow(service, session_id).await?;

        // Send the authorize URL as a progress update so the user sees it
        if let Some(ref tx) = status_tx {
            let msg = format!("Click this link to authorize:\n{}", authorize_url);
            let _ = tx
                .send(StatusUpdate::ToolProgress {
                    name: "manage_oauth".to_string(),
                    chunk: msg,
                })
                .await;
        }

        // Wait for the callback (up to 10 minutes)
        match tokio::time::timeout(OAuthGateway::flow_timeout(), result_rx).await {
            Ok(Ok(result)) => Ok(result.message),
            Ok(Err(_)) => Ok("OAuth flow was cancelled.".to_string()),
            Err(_) => {
                warn!(service = %service, "OAuth flow timed out");
                Ok("OAuth flow timed out (10 minutes). Please try again.".to_string())
            }
        }
    }

    async fn handle_remove(&self, service: &str) -> anyhow::Result<String> {
        // Check if connection exists
        let conn = self.state_store.get_oauth_connection(service).await?;
        if conn.is_none() {
            return Ok(format!("No OAuth connection found for '{}'", service));
        }
        self.gateway.remove_connection(service).await
    }

    async fn handle_set_credentials(
        &self,
        service: &str,
        client_id: &str,
        client_secret: &str,
    ) -> anyhow::Result<String> {
        // Verify provider exists
        if self.gateway.get_provider(service).await.is_none() {
            return Ok(format!(
                "Unknown provider '{}'. Use 'providers' to see available services.",
                service
            ));
        }

        let id_key = format!("oauth_{}_client_id", service);
        let secret_key = format!("oauth_{}_client_secret", service);

        crate::config::store_in_keychain(&id_key, client_id)?;
        crate::config::store_in_keychain(&secret_key, client_secret)?;

        Ok(format!(
            "Credentials stored for '{}'. You can now use 'connect' to authorize.",
            service
        ))
    }

    async fn handle_refresh(&self, service: &str) -> anyhow::Result<String> {
        let conn = self.state_store.get_oauth_connection(service).await?;
        if conn.is_none() {
            return Ok(format!("No OAuth connection found for '{}'", service));
        }
        self.gateway.refresh_token(service).await
    }
}
