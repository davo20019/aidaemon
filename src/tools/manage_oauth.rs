use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::{AppConfig, OAuthProviderConfig};
use crate::oauth::{OAuthGateway, OAuthType};
use crate::traits::{StateStore, Tool, ToolCallMetadata, ToolCallOutcome, ToolCapabilities};
use crate::types::{ApprovalResponse, StatusUpdate};

use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;

#[derive(Deserialize, Default)]
struct ManageOAuthArgs {
    action: String,
    service: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    display_name: Option<String>,
    auth_type: Option<String>,
    authorize_url: Option<String>,
    token_url: Option<String>,
    scopes: Option<Vec<String>>,
    allowed_domains: Option<Vec<String>>,
    confirm_disconnect: Option<bool>,
    #[serde(default)]
    _session_id: String,
}

pub struct ManageOAuthTool {
    gateway: OAuthGateway,
    state_store: Arc<dyn StateStore>,
    config_path: PathBuf,
    approval_tx: mpsc::Sender<ApprovalRequest>,
}

impl ManageOAuthTool {
    pub fn new(
        gateway: OAuthGateway,
        state_store: Arc<dyn StateStore>,
        config_path: PathBuf,
        approval_tx: mpsc::Sender<ApprovalRequest>,
    ) -> Self {
        Self {
            gateway,
            state_store,
            config_path,
            approval_tx,
        }
    }

    fn validate_service_name(raw: &str) -> anyhow::Result<String> {
        let trimmed = raw.trim();
        anyhow::ensure!(!trimmed.is_empty(), "Service name must not be empty");
        anyhow::ensure!(
            trimmed
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-')),
            "Service name '{}' is invalid. Use only letters, numbers, '_' or '-'.",
            trimmed
        );
        Ok(trimmed.to_ascii_lowercase())
    }

    fn normalize_domains(domains: &[String]) -> anyhow::Result<Vec<String>> {
        let normalized: Vec<String> = domains
            .iter()
            .map(|domain| domain.trim().trim_start_matches('.').to_ascii_lowercase())
            .filter(|domain| !domain.is_empty())
            .map(|domain| {
                anyhow::ensure!(
                    !domain.contains("://")
                        && !domain.contains('/')
                        && !domain.contains(char::is_whitespace),
                    "Allowed domain '{}' must be a bare host/domain, not a URL or path.",
                    domain
                );
                Ok(domain)
            })
            .collect::<anyhow::Result<_>>()?;
        anyhow::ensure!(
            !normalized.is_empty(),
            "`allowed_domains` must contain at least one hostname."
        );
        Ok(normalized)
    }

    fn validate_https_url(raw: &str, label: &str) -> anyhow::Result<String> {
        let trimmed = raw.trim();
        let parsed = reqwest::Url::parse(trimmed)
            .map_err(|err| anyhow::anyhow!("Invalid {}: {}", label, err))?;
        anyhow::ensure!(
            parsed.scheme() == "https",
            "{} must be an https:// URL.",
            label
        );
        Ok(parsed.to_string())
    }

    fn normalize_scopes(scopes: &[String]) -> Vec<String> {
        scopes
            .iter()
            .map(|scope| scope.trim().to_string())
            .filter(|scope| !scope.is_empty())
            .collect()
    }

    fn pending_state_from_authorize_url(authorize_url: &str) -> Option<String> {
        let parsed = reqwest::Url::parse(authorize_url).ok()?;
        parsed
            .query_pairs()
            .find_map(|(key, value)| (key == "state").then(|| value.into_owned()))
    }

    async fn send_connect_progress(status_tx: Option<&mpsc::Sender<StatusUpdate>>, chunk: String) {
        if let Some(tx) = status_tx {
            let _ = tx
                .send(StatusUpdate::ToolProgress {
                    name: "manage_oauth".to_string(),
                    chunk,
                })
                .await;
        }
    }

    fn validate_auth_type(raw: Option<&str>) -> anyhow::Result<String> {
        let normalized = raw.unwrap_or("oauth2_pkce").trim().to_ascii_lowercase();
        match normalized.as_str() {
            "oauth2_pkce" | "oauth2" | "pkce" => Ok("oauth2_pkce".to_string()),
            "oauth2_authorization_code" | "authorization_code" | "auth_code" => {
                Ok("oauth2_authorization_code".to_string())
            }
            "oauth2_client_credentials" | "client_credentials" => {
                Ok("oauth2_client_credentials".to_string())
            }
            "oauth1a" => anyhow::bail!(
                "Interactive OAuth 1.0a provider onboarding is not supported yet. Use `manage_http_auth` for manual OAuth 1.0a tokens."
            ),
            other => anyhow::bail!(
                "Unknown auth_type '{}'. Supported values: oauth2_pkce, oauth2_authorization_code, oauth2_client_credentials.",
                other
            ),
        }
    }

    async fn load_config_doc(&self) -> anyhow::Result<toml::Table> {
        let content = tokio::fs::read_to_string(&self.config_path).await?;
        Ok(content.parse()?)
    }

    fn validate_config(content: &str) -> Result<(), String> {
        let _doc: toml::Table = content
            .parse()
            .map_err(|e| format!("Invalid TOML syntax: {}", e))?;
        let expanded = crate::config::expand_env_vars(content).map_err(|e| format!("{}", e))?;
        toml::from_str::<AppConfig>(&expanded)
            .map_err(|e| format!("Invalid config structure: {}", e))?;
        Ok(())
    }

    #[cfg(unix)]
    fn set_owner_only_permissions(path: &Path) {
        use std::os::unix::fs::PermissionsExt;

        if let Ok(metadata) = std::fs::metadata(path) {
            let mut perms = metadata.permissions();
            perms.set_mode(0o600);
            let _ = std::fs::set_permissions(path, perms);
        }
    }

    #[cfg(not(unix))]
    fn set_owner_only_permissions(_path: &Path) {}

    async fn create_backup(&self) -> anyhow::Result<()> {
        let bak = self.config_path.with_extension("toml.bak");
        let bak1 = self.config_path.with_extension("toml.bak.1");
        let bak2 = self.config_path.with_extension("toml.bak.2");

        if bak1.exists() {
            let _ = tokio::fs::rename(&bak1, &bak2).await;
        }
        if bak.exists() {
            let _ = tokio::fs::rename(&bak, &bak1).await;
        }
        tokio::fs::copy(&self.config_path, &bak).await?;
        Self::set_owner_only_permissions(&bak);
        Self::set_owner_only_permissions(&bak1);
        Self::set_owner_only_permissions(&bak2);
        Ok(())
    }

    async fn save_config_doc(&self, doc: &toml::Table) -> anyhow::Result<()> {
        let rendered = toml::to_string_pretty(&toml::Value::Table(doc.clone()))?;
        if let Err(err) = Self::validate_config(&rendered) {
            anyhow::bail!("Refused to save invalid config: {}", err);
        }
        if let Err(err) = self.create_backup().await {
            warn!(error = %err, "Failed to create config backup before manage_oauth change");
        }
        tokio::fs::write(&self.config_path, &rendered).await?;
        Self::set_owner_only_permissions(&self.config_path);
        Ok(())
    }

    fn oauth_table_mut(doc: &mut toml::Table) -> anyhow::Result<&mut toml::Table> {
        doc.entry("oauth")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("`oauth` is not a table"))
    }

    fn providers_table_mut(doc: &mut toml::Table) -> anyhow::Result<&mut toml::Table> {
        Self::oauth_table_mut(doc)?
            .entry("providers")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("`oauth.providers` is not a table"))
    }

    fn providers_table(doc: &toml::Table) -> Option<&toml::Table> {
        doc.get("oauth")
            .and_then(toml::Value::as_table)
            .and_then(|oauth| oauth.get("providers"))
            .and_then(toml::Value::as_table)
    }

    async fn request_approval(
        &self,
        session_id: &str,
        description: &str,
        warnings: Vec<String>,
    ) -> anyhow::Result<ApprovalResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                command: description.to_string(),
                session_id: session_id.to_string(),
                risk_level: RiskLevel::High,
                warnings,
                permission_mode: PermissionMode::Default,
                response_tx,
                kind: Default::default(),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;

        match tokio::time::timeout(Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Ok(ApprovalResponse::Deny),
            Err(_) => Ok(ApprovalResponse::Deny),
        }
    }

    fn credential_commands(service: &str) -> (String, String) {
        (
            format!("aidaemon keychain set oauth_{}_client_id", service),
            format!("aidaemon keychain set oauth_{}_client_secret", service),
        )
    }

    fn callback_access_warning(callback_url: &str) -> Option<String> {
        let parsed = reqwest::Url::parse(callback_url).ok()?;
        let host = parsed.host_str()?.trim().trim_matches(['[', ']']);
        let is_local = matches!(host, "localhost" | "127.0.0.1" | "::1");
        if !is_local {
            return None;
        }

        Some(
            "callback_url points at localhost. This OAuth flow must be completed in a browser on the same machine running aidaemon. It will not finish from a phone or another device unless you expose the callback server and set `oauth.callback_url` to a reachable public URL.".to_string(),
        )
    }

    fn reconnect_preserves_existing_connection_note(service: &str) -> String {
        format!(
            "An existing OAuth connection for '{}' will stay active unless the new authorization flow completes successfully.",
            service
        )
    }

    fn disconnect_confirmation_message(service: &str) -> String {
        format!(
            "Refusing to disconnect '{}' without `confirm_disconnect=true`. Use `connect` to refresh permissions or replace tokens without dropping the current connection first.",
            service
        )
    }

    async fn handle_providers(&self) -> anyhow::Result<String> {
        let doc = self.load_config_doc().await?;
        let custom_names: std::collections::HashSet<String> = Self::providers_table(&doc)
            .map(|table| table.keys().cloned().collect())
            .unwrap_or_default();

        let mut providers = self.gateway.list_providers().await;
        providers.sort_by(|a, b| a.0.cmp(&b.0));
        if providers.is_empty() {
            return Ok(
                "No OAuth providers registered yet. Use 'register_provider' to add one."
                    .to_string(),
            );
        }
        let mut result = String::from("Available OAuth providers:\n");
        for (name, display_name) in &providers {
            let has_creds = OAuthGateway::has_credentials(name);
            let cred_status = if has_creds {
                "credentials set"
            } else {
                "credentials needed"
            };
            let source = if custom_names.contains(name) {
                "custom"
            } else {
                "built-in"
            };
            result.push_str(&format!(
                "  - {} ({}) [{}; {}]\n",
                display_name, name, source, cred_status
            ));
        }
        result.push_str(
            "\nUse 'describe_provider' for setup details, 'set_credentials' to store client credentials, and 'connect' to authorize.",
        );
        Ok(result)
    }

    async fn handle_describe_provider(&self, service: &str) -> anyhow::Result<String> {
        let provider = match self.gateway.get_provider(service).await {
            Some(provider) => provider,
            None => {
                return Ok(format!(
                    "Unknown OAuth provider '{}'. Use 'providers' to see available services.",
                    service
                ));
            }
        };
        let auth_type = match provider.auth_type {
            OAuthType::OAuth2Pkce => "oauth2_pkce",
            OAuthType::OAuth2AuthorizationCode => "oauth2_authorization_code",
            OAuthType::OAuth2ClientCredentials => "oauth2_client_credentials",
            OAuthType::OAuth1a => "oauth1a",
        };
        let has_creds = OAuthGateway::has_credentials(service);
        let source = if crate::oauth::providers::get_builtin_provider(service).is_some() {
            "built-in"
        } else {
            "custom"
        };
        let (client_id_cmd, client_secret_cmd) = Self::credential_commands(service);
        let callback_url = self.gateway.callback_url();
        let callback_warning = if provider.auth_type == OAuthType::OAuth2ClientCredentials {
            None
        } else {
            Self::callback_access_warning(&callback_url)
        };

        let mut lines = vec![
            format!("OAuth provider `{}`", service),
            format!("- display_name: {}", provider.display_name),
            format!("- source: {}", source),
            format!("- auth_type: {}", auth_type),
            format!(
                "- authorize_url: {}",
                if provider.authorize_url.is_empty() {
                    "(not used)".to_string()
                } else {
                    provider.authorize_url.clone()
                }
            ),
            format!("- token_url: {}", provider.token_url),
            format!(
                "- callback_url: {}",
                if provider.auth_type == OAuthType::OAuth2ClientCredentials {
                    "(not used)".to_string()
                } else {
                    callback_url
                }
            ),
            format!("- allowed_domains: {}", provider.allowed_domains.join(", ")),
            format!(
                "- scopes: {}",
                if provider.scopes.is_empty() {
                    "(none)".to_string()
                } else {
                    provider.scopes.join(", ")
                }
            ),
            format!(
                "- client credentials: {}",
                if has_creds { "stored" } else { "missing" }
            ),
        ];

        if let Some(warning) = callback_warning {
            lines.push(format!("- note: {}", warning));
        }

        if !has_creds {
            lines.push(String::new());
            lines.push("Store the client credentials securely with:".to_string());
            lines.push(format!("- `{}`", client_id_cmd));
            lines.push(format!("- `{}`", client_secret_cmd));
        }

        lines.push(String::new());
        lines.push(match provider.auth_type {
            OAuthType::OAuth2ClientCredentials => {
                "After credentials are stored, use `connect` to fetch a token immediately."
                    .to_string()
            }
            _ => {
                "After credentials are stored, use `connect` to start the browser authorization flow."
                    .to_string()
            }
        });
        Ok(lines.join("\n"))
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
        let has_existing_connection = self
            .state_store
            .get_oauth_connection(service)
            .await?
            .is_some();
        if let Some(provider) = self.gateway.get_provider(service).await {
            match provider.auth_type {
                OAuthType::OAuth1a => {
                    return Ok(format!(
                        "Provider '{}' is configured as OAuth 1.0a, which is not supported by the interactive OAuth connector yet. Use manage_http_auth for manual OAuth 1.0a tokens.",
                        service
                    ));
                }
                OAuthType::OAuth2ClientCredentials => {
                    return self.gateway.connect_client_credentials(service).await;
                }
                OAuthType::OAuth2Pkce | OAuthType::OAuth2AuthorizationCode => {}
            }
        }

        let (authorize_url, result_rx) =
            self.gateway.start_oauth2_flow(service, session_id).await?;
        let pending_state = Self::pending_state_from_authorize_url(&authorize_url);
        let callback_warning = Self::callback_access_warning(&self.gateway.callback_url());
        let reconnect_note = has_existing_connection
            .then(|| Self::reconnect_preserves_existing_connection_note(service));

        let mut authorize_parts = Vec::new();
        if let Some(note) = &reconnect_note {
            authorize_parts.push(note.clone());
        }
        if let Some(warning) = &callback_warning {
            authorize_parts.push(format!("Note: {}", warning));
        }
        authorize_parts.push(format!("Click this link to authorize:\n{}", authorize_url));
        Self::send_connect_progress(status_tx.as_ref(), authorize_parts.join("\n\n")).await;

        let compose_result = |body: String| {
            let mut parts = Vec::new();
            if let Some(note) = &reconnect_note {
                parts.push(note.clone());
            }
            if let Some(warning) = &callback_warning {
                parts.push(format!("Note: {}", warning));
            }
            parts.push(body);
            parts.join("\n\n")
        };

        match tokio::time::timeout(OAuthGateway::flow_timeout(), result_rx).await {
            Ok(Ok(result)) => {
                Self::send_connect_progress(
                    status_tx.as_ref(),
                    format!(
                        "Browser authorization completed. Returning to chat.\n{}",
                        result.message
                    ),
                )
                .await;
                Ok(compose_result(result.message))
            }
            Ok(Err(_)) => {
                Self::send_connect_progress(
                    status_tx.as_ref(),
                    "OAuth browser authorization was cancelled.".to_string(),
                )
                .await;
                Ok(compose_result("OAuth flow was cancelled.".to_string()))
            }
            Err(_) => {
                warn!(service = %service, "OAuth flow timed out");
                if let Some(state) = pending_state.as_deref() {
                    if let Err(err) = self
                        .gateway
                        .expire_pending_flow(
                            state,
                            Some(OAuthGateway::expired_flow_message().to_string()),
                        )
                        .await
                    {
                        warn!(
                            service = %service,
                            state = %state,
                            error = %err,
                            "Failed to expire timed out OAuth flow"
                        );
                    }
                }
                Self::send_connect_progress(
                    status_tx.as_ref(),
                    OAuthGateway::expired_flow_message().to_string(),
                )
                .await;
                Ok(compose_result(
                    OAuthGateway::expired_flow_message().to_string(),
                ))
            }
        }
    }

    async fn handle_remove(
        &self,
        service: &str,
        confirm_disconnect: bool,
        session_id: &str,
    ) -> anyhow::Result<String> {
        let conn = self.state_store.get_oauth_connection(service).await?;
        if conn.is_none() {
            return Ok(format!("No OAuth connection found for '{}'", service));
        }
        if !confirm_disconnect {
            return Ok(Self::disconnect_confirmation_message(service));
        }
        let approval_description = format!("Disconnect OAuth service '{}'", service);
        match self
            .request_approval(
                session_id,
                &approval_description,
                vec![
                    "Deletes the stored OAuth connection and tokens".to_string(),
                    "Leaves the service unavailable until it is connected again".to_string(),
                ],
            )
            .await?
        {
            ApprovalResponse::AllowOnce
            | ApprovalResponse::AllowSession
            | ApprovalResponse::AllowAlways => {}
            ApprovalResponse::Deny => {
                return Ok("OAuth disconnection denied by user.".to_string());
            }
        }
        self.gateway.remove_connection(service).await
    }

    async fn handle_register_provider(
        &self,
        service: &str,
        args: &ManageOAuthArgs,
    ) -> anyhow::Result<String> {
        if crate::oauth::providers::get_builtin_provider(service).is_some() {
            return Ok(format!(
                "'{}' is a built-in OAuth provider. Use it directly instead of re-registering it.",
                service
            ));
        }

        let auth_type = Self::validate_auth_type(args.auth_type.as_deref())?;
        let authorize_url = match auth_type.as_str() {
            "oauth2_client_credentials" => String::new(),
            _ => Self::validate_https_url(
                args.authorize_url.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'register_provider' requires 'authorize_url'")
                })?,
                "authorize_url",
            )?,
        };
        let token_url = Self::validate_https_url(
            args.token_url
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("'register_provider' requires 'token_url'"))?,
            "token_url",
        )?;
        let allowed_domains =
            Self::normalize_domains(args.allowed_domains.as_deref().ok_or_else(|| {
                anyhow::anyhow!("'register_provider' requires 'allowed_domains'")
            })?)?;
        let scopes = Self::normalize_scopes(args.scopes.as_deref().unwrap_or(&[]));
        let session_id = if args._session_id.is_empty() {
            "unknown"
        } else {
            args._session_id.as_str()
        };
        let approval_description = format!(
            "Register custom OAuth provider '{}' (authorize={}, token={})",
            service, authorize_url, token_url
        );
        match self
            .request_approval(
                session_id,
                &approval_description,
                vec![
                    "Modifies OAuth configuration".to_string(),
                    "Can route credentialed API traffic to the configured domains".to_string(),
                ],
            )
            .await?
        {
            ApprovalResponse::AllowOnce
            | ApprovalResponse::AllowSession
            | ApprovalResponse::AllowAlways => {}
            ApprovalResponse::Deny => {
                return Ok("OAuth provider registration denied by user.".to_string());
            }
        }

        let display_name = args
            .display_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string);

        let mut doc = self.load_config_doc().await?;
        {
            let oauth_table = Self::oauth_table_mut(&mut doc)?;
            oauth_table.insert("enabled".to_string(), toml::Value::Boolean(true));
        }
        {
            let providers_table = Self::providers_table_mut(&mut doc)?;
            let mut provider_table = toml::Table::new();
            if let Some(ref display_name) = display_name {
                provider_table.insert(
                    "display_name".to_string(),
                    toml::Value::String(display_name.clone()),
                );
            }
            provider_table.insert(
                "auth_type".to_string(),
                toml::Value::String(auth_type.clone()),
            );
            provider_table.insert(
                "authorize_url".to_string(),
                toml::Value::String(authorize_url.clone()),
            );
            provider_table.insert(
                "token_url".to_string(),
                toml::Value::String(token_url.clone()),
            );
            provider_table.insert(
                "scopes".to_string(),
                toml::Value::Array(scopes.iter().cloned().map(toml::Value::String).collect()),
            );
            provider_table.insert(
                "allowed_domains".to_string(),
                toml::Value::Array(
                    allowed_domains
                        .iter()
                        .cloned()
                        .map(toml::Value::String)
                        .collect(),
                ),
            );
            providers_table.insert(service.to_string(), toml::Value::Table(provider_table));
        }
        self.save_config_doc(&doc).await?;

        let uses_callback = auth_type != "oauth2_client_credentials";
        let provider_config = OAuthProviderConfig {
            display_name: display_name.clone(),
            auth_type,
            authorize_url,
            token_url,
            scopes,
            allowed_domains,
        };
        self.gateway
            .register_config_provider(service, &provider_config)
            .await;

        let (client_id_cmd, client_secret_cmd) = Self::credential_commands(service);
        let provider_label = display_name.as_deref().unwrap_or(service);
        info!(service = %service, "Registered custom OAuth provider");
        Ok(format!(
            "Registered custom OAuth provider `{}`.\n- callback_url: {}\n\nNext:\n- `{}`\n- `{}`\n- then use `connect` for `{}`.",
            provider_label,
            if uses_callback {
                self.gateway.callback_url()
            } else {
                "(not used)".to_string()
            },
            client_id_cmd,
            client_secret_cmd,
            service
        ))
    }

    async fn handle_remove_provider(
        &self,
        service: &str,
        session_id: &str,
    ) -> anyhow::Result<String> {
        if crate::oauth::providers::get_builtin_provider(service).is_some() {
            return Ok(format!(
                "'{}' is a built-in OAuth provider and cannot be removed.",
                service
            ));
        }
        if self
            .state_store
            .get_oauth_connection(service)
            .await?
            .is_some()
        {
            return Ok(format!(
                "Provider '{}' still has an active OAuth connection. Disconnect it first with action='remove', then remove the provider definition.",
                service
            ));
        }

        let mut doc = self.load_config_doc().await?;
        let Some(providers_table) = Self::providers_table_mut(&mut doc).ok() else {
            return Ok(format!(
                "Custom OAuth provider '{}' was not found.",
                service
            ));
        };
        if !providers_table.contains_key(service) {
            return Ok(format!(
                "Custom OAuth provider '{}' was not found.",
                service
            ));
        }

        let approval_description = format!(
            "Remove custom OAuth provider '{}' and delete its client credentials",
            service
        );
        match self
            .request_approval(
                session_id,
                &approval_description,
                vec![
                    "Deletes OAuth provider configuration".to_string(),
                    "Deletes stored client credentials for this provider".to_string(),
                ],
            )
            .await?
        {
            ApprovalResponse::AllowOnce
            | ApprovalResponse::AllowSession
            | ApprovalResponse::AllowAlways => {}
            ApprovalResponse::Deny => {
                return Ok("OAuth provider removal denied by user.".to_string());
            }
        }

        providers_table.remove(service);
        if providers_table.is_empty() {
            if let Some(oauth_table) = doc.get_mut("oauth").and_then(toml::Value::as_table_mut) {
                oauth_table.remove("providers");
            }
        }
        self.save_config_doc(&doc).await?;

        let _ = self.gateway.unregister_provider(service).await;
        for suffix in ["client_id", "client_secret"] {
            let key = format!("oauth_{}_{}", service, suffix);
            if let Err(err) = crate::config::delete_from_keychain(&key) {
                warn!(key = %key, error = %err, "Failed to delete OAuth client credential after provider removal");
            }
        }

        Ok(format!(
            "Removed custom OAuth provider '{}' and deleted its stored client credentials.",
            service
        ))
    }

    async fn handle_set_credentials(
        &self,
        service: &str,
        client_id: &str,
        client_secret: &str,
    ) -> anyhow::Result<String> {
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

#[async_trait]
impl Tool for ManageOAuthTool {
    fn name(&self) -> &str {
        "manage_oauth"
    }

    fn description(&self) -> &str {
        "Connect and manage OAuth services, including custom providers"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_oauth",
            "description": "Manage OAuth providers, credentials, connections, and token refresh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["connect", "list", "remove", "set_credentials", "refresh", "providers", "describe_provider", "register_provider", "remove_provider"],
                        "description": "Action"
                    },
                    "service": {
                        "type": "string",
                        "description": "Provider name"
                    },
                    "client_id": {
                        "type": "string",
                        "description": "OAuth client/app ID"
                    },
                    "client_secret": {
                        "type": "string",
                        "description": "OAuth client/app secret"
                    },
                    "display_name": {
                        "type": "string",
                        "description": "Optional provider label"
                    },
                    "auth_type": {
                        "type": "string",
                        "enum": ["oauth2_pkce", "oauth2_authorization_code", "oauth2_client_credentials"],
                        "description": "Custom provider auth type"
                    },
                    "authorize_url": {
                        "type": "string",
                        "description": "Authorize URL"
                    },
                    "token_url": {
                        "type": "string",
                        "description": "Token URL"
                    },
                    "scopes": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional scopes for register_provider. Ignored for built-in providers and connect/list/remove actions."
                    },
                    "allowed_domains": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Allowed API domains for register_provider."
                    },
                    "confirm_disconnect": {
                        "type": "boolean",
                        "description": "Required with true when action='remove'. Use remove only for intentional disconnection; use connect to refresh or replace tokens."
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        self.call_with_status(arguments, None).await
    }

    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        let args: ManageOAuthArgs = serde_json::from_str(arguments)?;
        match args.action.as_str() {
            "providers" => self.handle_providers().await,
            "describe_provider" => {
                let service = Self::validate_service_name(
                    args.service
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("'describe_provider' requires 'service'"))?,
                )?;
                self.handle_describe_provider(&service).await
            }
            "list" => self.handle_list().await,
            "connect" => {
                let service = Self::validate_service_name(
                    args.service
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("'connect' requires 'service' parameter"))?,
                )?;
                let session_id = if args._session_id.is_empty() {
                    "unknown"
                } else {
                    args._session_id.as_str()
                };
                self.handle_connect(&service, session_id, status_tx).await
            }
            "remove" => {
                let service = Self::validate_service_name(
                    args.service
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("'remove' requires 'service' parameter"))?,
                )?;
                let session_id = if args._session_id.is_empty() {
                    "unknown"
                } else {
                    args._session_id.as_str()
                };
                self.handle_remove(
                    &service,
                    args.confirm_disconnect.unwrap_or(false),
                    session_id,
                )
                .await
            }
            "register_provider" => {
                let service = Self::validate_service_name(
                    args.service.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("'register_provider' requires 'service' parameter")
                    })?,
                )?;
                self.handle_register_provider(&service, &args).await
            }
            "remove_provider" => {
                let service = Self::validate_service_name(
                    args.service.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("'remove_provider' requires 'service' parameter")
                    })?,
                )?;
                let session_id = if args._session_id.is_empty() {
                    "unknown"
                } else {
                    args._session_id.as_str()
                };
                self.handle_remove_provider(&service, session_id).await
            }
            "set_credentials" => {
                let service = Self::validate_service_name(
                    args.service.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("'set_credentials' requires 'service' parameter")
                    })?,
                )?;
                let client_id = args
                    .client_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'set_credentials' requires 'client_id'"))?;
                let client_secret = args.client_secret.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'set_credentials' requires 'client_secret'")
                })?;
                self.handle_set_credentials(&service, client_id, client_secret)
                    .await
            }
            "refresh" => {
                let service = Self::validate_service_name(
                    args.service
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("'refresh' requires 'service'"))?,
                )?;
                self.handle_refresh(&service).await
            }
            other => Ok(format!(
                "Unknown action '{}'. Use: connect, list, remove, set_credentials, refresh, providers, describe_provider, register_provider, remove_provider.",
                other
            )),
        }
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: ManageOAuthArgs = serde_json::from_str(arguments)?;
        let output = self.call_with_status(arguments, status_tx).await?;
        let mut outcome = ToolCallOutcome::from_output(output.clone());
        if args.action == "connect" {
            outcome.metadata = ToolCallMetadata {
                direct_response: Some(output),
                ..ToolCallMetadata::default()
            };
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::time::Duration;

    use axum::{extract::Form, routing::post, Json, Router};
    use once_cell::sync::Lazy;
    use tempfile::NamedTempFile;
    use tokio::net::TcpListener;

    use crate::memory::embeddings::EmbeddingService;
    use crate::oauth::{OAuthProvider, SharedHttpProfiles};
    use crate::state::SqliteStateStore;

    static ENV_LOCK: Lazy<std::sync::Mutex<()>> = Lazy::new(|| std::sync::Mutex::new(()));

    fn restore_env_var(name: &str, old_value: Option<String>) {
        if let Some(value) = old_value {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
    }

    async fn test_tool(config_path: PathBuf) -> anyhow::Result<(ManageOAuthTool, OAuthGateway)> {
        let db_file = NamedTempFile::new()?;
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new()?);
        let state = Arc::new(SqliteStateStore::new(&db_path, 32, None, embedding_service).await?);
        let profiles: SharedHttpProfiles =
            Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));
        let gateway = OAuthGateway::new(
            state.clone() as Arc<dyn StateStore>,
            profiles,
            "http://localhost:8080".to_string(),
        );
        let (approval_tx, mut approval_rx) = mpsc::channel::<ApprovalRequest>(4);
        tokio::spawn(async move {
            while let Some(request) = approval_rx.recv().await {
                let _ = request.response_tx.send(ApprovalResponse::AllowOnce);
            }
        });
        Ok((
            ManageOAuthTool::new(
                gateway.clone(),
                state as Arc<dyn StateStore>,
                config_path,
                approval_tx,
            ),
            gateway,
        ))
    }

    fn write_minimal_config(path: &Path) {
        std::fs::write(path, "[provider]\napi_key = \"test-key\"\n").unwrap();
    }

    #[tokio::test]
    async fn register_provider_persists_and_hot_registers_custom_oauth_provider() {
        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let (tool, gateway) = test_tool(config_file.path().to_path_buf()).await.unwrap();

        let result = tool
            .call(
                r#"{
                    "action": "register_provider",
                    "service": "linear",
                    "display_name": "Linear",
                    "authorize_url": "https://linear.app/oauth/authorize",
                    "token_url": "https://api.linear.app/oauth/token",
                    "scopes": ["read", "write"],
                    "allowed_domains": ["api.linear.app"],
                    "_session_id": "test"
                }"#,
            )
            .await
            .unwrap();

        assert!(result.contains("Registered custom OAuth provider `Linear`."));
        assert!(result.contains("oauth_linear_client_id"));
        let saved = std::fs::read_to_string(config_file.path()).unwrap();
        assert!(saved.contains("[oauth.providers.linear]"));
        assert!(saved.contains("enabled = true"));

        let provider = gateway
            .get_provider("linear")
            .await
            .expect("provider registered");
        assert_eq!(provider.display_name, "Linear");
        assert_eq!(provider.allowed_domains, vec!["api.linear.app"]);
    }

    #[tokio::test]
    async fn register_provider_supports_client_credentials_auth_type() {
        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let (tool, gateway) = test_tool(config_file.path().to_path_buf()).await.unwrap();

        let result = tool
            .call(
                r#"{
                    "action": "register_provider",
                    "service": "corpapi",
                    "display_name": "Corp API",
                    "auth_type": "oauth2_client_credentials",
                    "token_url": "https://auth.example.com/oauth/token",
                    "scopes": ["read"],
                    "allowed_domains": ["api.example.com"],
                    "_session_id": "test"
                }"#,
            )
            .await
            .unwrap();

        assert!(result.contains("Registered custom OAuth provider `Corp API`."));
        assert!(result.contains("callback_url: (not used)"));
        let saved = std::fs::read_to_string(config_file.path()).unwrap();
        assert!(saved.contains("auth_type = \"oauth2_client_credentials\""));

        let provider = gateway
            .get_provider("corpapi")
            .await
            .expect("provider registered");
        assert_eq!(provider.auth_type, OAuthType::OAuth2ClientCredentials);
        assert!(provider.authorize_url.is_empty());
    }

    #[tokio::test]
    async fn remove_provider_deletes_custom_provider_and_client_credentials() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config_file = NamedTempFile::new().unwrap();
        std::fs::write(
            config_file.path(),
            r#"[provider]
api_key = "test-key"

[oauth]
enabled = true

[oauth.providers.linear]
display_name = "Linear"
auth_type = "oauth2_pkce"
authorize_url = "https://linear.app/oauth/authorize"
token_url = "https://api.linear.app/oauth/token"
scopes = ["read"]
allowed_domains = ["api.linear.app"]
"#,
        )
        .unwrap();

        let env_file = NamedTempFile::new().unwrap();
        std::fs::write(
            env_file.path(),
            "OAUTH_LINEAR_CLIENT_ID=abc\nOAUTH_LINEAR_CLIENT_SECRET=def\n",
        )
        .unwrap();
        let old_no_keychain = std::env::var("AIDAEMON_NO_KEYCHAIN").ok();
        let old_runtime_env = std::env::var(crate::RUNTIME_ENV_FILE_ENV_KEY).ok();
        std::env::set_var("AIDAEMON_NO_KEYCHAIN", "1");
        std::env::set_var(
            crate::RUNTIME_ENV_FILE_ENV_KEY,
            env_file.path().to_string_lossy().to_string(),
        );

        let (tool, gateway) = test_tool(config_file.path().to_path_buf()).await.unwrap();
        let config_provider = OAuthProviderConfig {
            display_name: Some("Linear".to_string()),
            auth_type: "oauth2_pkce".to_string(),
            authorize_url: "https://linear.app/oauth/authorize".to_string(),
            token_url: "https://api.linear.app/oauth/token".to_string(),
            scopes: vec!["read".to_string()],
            allowed_domains: vec!["api.linear.app".to_string()],
        };
        gateway
            .register_config_provider("linear", &config_provider)
            .await;

        let result = tool
            .call(r#"{"action":"remove_provider","service":"linear","_session_id":"test"}"#)
            .await
            .unwrap();

        restore_env_var("AIDAEMON_NO_KEYCHAIN", old_no_keychain);
        restore_env_var(crate::RUNTIME_ENV_FILE_ENV_KEY, old_runtime_env);

        assert!(result.contains("Removed custom OAuth provider 'linear'"));
        let saved = std::fs::read_to_string(config_file.path()).unwrap();
        assert!(!saved.contains("[oauth.providers.linear]"));
        assert!(gateway.get_provider("linear").await.is_none());
        let env_content = std::fs::read_to_string(env_file.path()).unwrap();
        assert!(!env_content.contains("OAUTH_LINEAR_CLIENT_ID"));
        assert!(!env_content.contains("OAUTH_LINEAR_CLIENT_SECRET"));
    }

    #[test]
    fn callback_access_warning_detects_localhost_callback_urls() {
        let warning =
            ManageOAuthTool::callback_access_warning("http://localhost:8080/oauth/callback")
                .expect("localhost warning");
        assert!(warning.contains("same machine running aidaemon"));

        let warning =
            ManageOAuthTool::callback_access_warning("http://127.0.0.1:8080/oauth/callback")
                .expect("loopback warning");
        assert!(warning.contains("reachable public URL"));

        assert!(ManageOAuthTool::callback_access_warning(
            "https://auth.example.com/oauth/callback"
        )
        .is_none());
    }

    #[tokio::test]
    async fn describe_provider_surfaces_localhost_callback_warning() {
        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let (tool, gateway) = test_tool(config_file.path().to_path_buf()).await.unwrap();
        gateway
            .register_provider(crate::oauth::providers::get_builtin_provider("twitter").unwrap())
            .await;

        let result = tool
            .call(r#"{"action":"describe_provider","service":"twitter"}"#)
            .await
            .unwrap();

        assert!(result.contains("callback_url: http://localhost:8080/oauth/callback"));
        assert!(result.contains("same machine running aidaemon"));
    }

    #[tokio::test]
    async fn connect_reports_browser_completion_via_status_updates() {
        let _guard = ENV_LOCK.lock().unwrap();

        async fn token_handler(
            Form(form): Form<HashMap<String, String>>,
        ) -> Json<serde_json::Value> {
            assert_eq!(
                form.get("grant_type").map(String::as_str),
                Some("authorization_code")
            );
            Json(serde_json::json!({
                "access_token": "connected-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600
            }))
        }

        let app = Router::new().route("/oauth/token", post(token_handler));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let (tool, gateway) = test_tool(config_file.path().to_path_buf()).await.unwrap();
        gateway
            .register_provider(OAuthProvider {
                name: "linear".to_string(),
                display_name: "Linear".to_string(),
                auth_type: OAuthType::OAuth2AuthorizationCode,
                authorize_url: "https://linear.app/oauth/authorize".to_string(),
                token_url: format!("http://{addr}/oauth/token"),
                scopes: vec!["read".to_string()],
                allowed_domains: vec!["api.linear.app".to_string()],
            })
            .await;

        let env_file = NamedTempFile::new().unwrap();
        std::fs::write(
            env_file.path(),
            "OAUTH_LINEAR_CLIENT_ID=abc\nOAUTH_LINEAR_CLIENT_SECRET=def\n",
        )
        .unwrap();
        let old_no_keychain = std::env::var("AIDAEMON_NO_KEYCHAIN").ok();
        let old_runtime_env = std::env::var(crate::RUNTIME_ENV_FILE_ENV_KEY).ok();
        std::env::set_var("AIDAEMON_NO_KEYCHAIN", "1");
        std::env::set_var(
            crate::RUNTIME_ENV_FILE_ENV_KEY,
            env_file.path().to_string_lossy().to_string(),
        );

        let (status_tx, mut status_rx) = mpsc::channel::<StatusUpdate>(8);
        let tool = Arc::new(tool);
        let tool_task = {
            let tool = tool.clone();
            tokio::spawn(async move {
                tool.call_with_status_outcome(
                    r#"{"action":"connect","service":"linear","_session_id":"telegram:123"}"#,
                    Some(status_tx),
                )
                .await
            })
        };

        let first_update = tokio::time::timeout(Duration::from_secs(2), status_rx.recv())
            .await
            .unwrap()
            .unwrap();
        let authorize_chunk = match first_update {
            StatusUpdate::ToolProgress { chunk, .. } => chunk,
            other => panic!("unexpected first status update: {other:?}"),
        };
        assert!(authorize_chunk.contains("Click this link to authorize:"));
        let authorize_url = authorize_chunk
            .lines()
            .find(|line| line.starts_with("https://"))
            .unwrap()
            .trim();
        let state = reqwest::Url::parse(authorize_url)
            .unwrap()
            .query_pairs()
            .find_map(|(key, value)| (key == "state").then(|| value.into_owned()))
            .unwrap();

        let callback_result = gateway
            .handle_callback(&state, Some("auth-code"), None)
            .await
            .unwrap();
        assert!(callback_result.contains("Connected to Linear"));

        let second_update = tokio::time::timeout(Duration::from_secs(2), status_rx.recv())
            .await
            .unwrap()
            .unwrap();
        let completion_chunk = match second_update {
            StatusUpdate::ToolProgress { chunk, .. } => chunk,
            other => panic!("unexpected completion status update: {other:?}"),
        };
        assert!(completion_chunk.contains("Browser authorization completed"));
        assert!(completion_chunk.contains("Connected to Linear"));

        let tool_outcome = tool_task.await.unwrap().unwrap();

        restore_env_var("AIDAEMON_NO_KEYCHAIN", old_no_keychain);
        restore_env_var(crate::RUNTIME_ENV_FILE_ENV_KEY, old_runtime_env);

        assert!(tool_outcome.output.contains("Connected to Linear"));
        assert_eq!(
            tool_outcome.metadata.direct_response,
            Some(tool_outcome.output)
        );
    }

    #[tokio::test]
    async fn remove_requires_explicit_confirmation() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config_file = NamedTempFile::new().unwrap();
        let env_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let old_no_keychain = std::env::var("AIDAEMON_NO_KEYCHAIN").ok();
        let old_runtime_env = std::env::var(crate::RUNTIME_ENV_FILE_ENV_KEY).ok();
        std::env::set_var("AIDAEMON_NO_KEYCHAIN", "1");
        std::env::set_var(
            crate::RUNTIME_ENV_FILE_ENV_KEY,
            env_file.path().to_string_lossy().to_string(),
        );
        let (tool, _gateway) = test_tool(config_file.path().to_path_buf()).await.unwrap();

        tool.state_store
            .save_oauth_connection(&crate::traits::OAuthConnection {
                id: 0,
                service: "twitter".to_string(),
                auth_type: "oauth2_pkce".to_string(),
                username: None,
                scopes: r#"["tweet.read","tweet.write","users.read","offline.access"]"#.to_string(),
                token_expires_at: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();

        let result = tool
            .call(r#"{"action":"remove","service":"twitter","_session_id":"test"}"#)
            .await
            .unwrap();

        restore_env_var("AIDAEMON_NO_KEYCHAIN", old_no_keychain);
        restore_env_var(crate::RUNTIME_ENV_FILE_ENV_KEY, old_runtime_env);

        assert!(result.contains("confirm_disconnect=true"));
        assert!(tool
            .state_store
            .get_oauth_connection("twitter")
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn remove_with_explicit_confirmation_disconnects_service() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config_file = NamedTempFile::new().unwrap();
        let env_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let old_no_keychain = std::env::var("AIDAEMON_NO_KEYCHAIN").ok();
        let old_runtime_env = std::env::var(crate::RUNTIME_ENV_FILE_ENV_KEY).ok();
        std::env::set_var("AIDAEMON_NO_KEYCHAIN", "1");
        std::env::set_var(
            crate::RUNTIME_ENV_FILE_ENV_KEY,
            env_file.path().to_string_lossy().to_string(),
        );
        let (tool, _gateway) = test_tool(config_file.path().to_path_buf()).await.unwrap();

        tool.state_store
            .save_oauth_connection(&crate::traits::OAuthConnection {
                id: 0,
                service: "twitter".to_string(),
                auth_type: "oauth2_pkce".to_string(),
                username: None,
                scopes: r#"["tweet.read","tweet.write","users.read","offline.access"]"#.to_string(),
                token_expires_at: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();

        let result = tool
            .call(
                r#"{"action":"remove","service":"twitter","confirm_disconnect":true,"_session_id":"test"}"#,
            )
            .await
            .unwrap();

        restore_env_var("AIDAEMON_NO_KEYCHAIN", old_no_keychain);
        restore_env_var(crate::RUNTIME_ENV_FILE_ENV_KEY, old_runtime_env);

        assert!(result.contains("Disconnected from twitter"));
        assert!(tool
            .state_store
            .get_oauth_connection("twitter")
            .await
            .unwrap()
            .is_none());
    }
}
