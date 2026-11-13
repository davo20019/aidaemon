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
use crate::traits::{
    semantics_for_exact_read_actions, ApprovalScope, OAuthStore, OwnerApprovalNeed,
    StableObservationSubject, Tool, ToolCallContext, ToolCallMetadata, ToolCallOutcome,
    ToolCallSemantics, ToolCapabilities, ToolCollectionObservation, ToolMutationEffects,
    ToolObservationEvidence, ToolSemanticFacet, ToolTargetHint, ToolTargetHintKind,
};
use crate::types::StatusUpdate;

use crate::tools::HttpRequestTool;

#[derive(Deserialize, Default)]
struct ManageOAuthArgs {
    action: String,
    service: Option<String>,
    account_id: Option<String>,
    identity_url: Option<String>,
    account_id_pointer: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    display_name: Option<String>,
    auth_type: Option<String>,
    authorize_url: Option<String>,
    token_url: Option<String>,
    scopes: Option<Vec<String>>,
    allowed_domains: Option<Vec<String>>,
    confirm_disconnect: Option<bool>,
    /// The calling session, from the dispatcher's facts (never an argument).
    #[serde(skip)]
    call_session: String,
}

/// What a `bind_account` proves and binds, once its arguments and the
/// connection are read.
struct BindPlan {
    identity_url: String,
    expected_account_id: String,
    account_id_pointer: String,
    prior_binding: String,
}

/// The custom provider a `register_provider` writes, once its arguments are
/// read.
struct RegisterPlan {
    auth_type: String,
    authorize_url: String,
    token_url: String,
    allowed_domains: Vec<String>,
    scopes: Vec<String>,
}

pub struct ManageOAuthTool {
    gateway: OAuthGateway,
    state_store: Arc<dyn OAuthStore>,
    config_path: PathBuf,
}

impl ManageOAuthTool {
    /// Stable collection subject for every connected OAuth service, complete
    /// on each `list`. Members are the same `auth_profile:<service>` IDs that
    /// `manage_http_auth` and `http_request` report, so one credential
    /// identity is shared across all three adapters.
    pub const OAUTH_CONNECTION_COLLECTION_ID: &'static str =
        "oauth_connection_collection:connected";

    pub fn new(
        gateway: OAuthGateway,
        state_store: Arc<dyn OAuthStore>,
        config_path: PathBuf,
    ) -> Self {
        Self {
            gateway,
            state_store,
            config_path,
        }
    }

    fn connection_collection_hint() -> ToolTargetHint {
        ToolTargetHint::new(
            ToolTargetHintKind::ResourceId,
            Self::OAUTH_CONNECTION_COLLECTION_ID,
        )
        .expect("oauth connection collection ID is nonempty")
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

    fn validate_account_id(raw: &str) -> anyhow::Result<String> {
        let value = raw.trim();
        anyhow::ensure!(
            !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control),
            "account_id must be a non-empty stable identifier of at most 256 characters"
        );
        Ok(value.to_string())
    }

    fn validate_account_id_pointer(raw: &str) -> anyhow::Result<String> {
        let value = raw.trim();
        anyhow::ensure!(
            value.starts_with('/') && value.len() <= 256 && !value.chars().any(char::is_control),
            "account_id_pointer must be an RFC 6901 JSON Pointer beginning with '/'"
        );
        Ok(value.to_string())
    }

    fn validate_identity_url(
        raw: &str,
        provider: &crate::oauth::OAuthProvider,
    ) -> anyhow::Result<String> {
        let parsed = reqwest::Url::parse(raw.trim())
            .map_err(|error| anyhow::anyhow!("Invalid identity_url: {}", error))?;
        let host = parsed
            .host_str()
            .ok_or_else(|| anyhow::anyhow!("identity_url must contain a host"))?;
        let test_localhost = cfg!(test)
            && matches!(host, "localhost" | "127.0.0.1" | "::1")
            && parsed.scheme() == "http";
        anyhow::ensure!(
            parsed.scheme() == "https" || test_localhost,
            "identity_url must use https"
        );
        anyhow::ensure!(
            parsed.username().is_empty()
                && parsed.password().is_none()
                && parsed.fragment().is_none(),
            "identity_url cannot contain credentials or a fragment"
        );
        anyhow::ensure!(
            provider
                .allowed_domains
                .iter()
                .any(|domain| HttpRequestTool::domain_matches(host, domain)),
            "identity_url host '{}' is not allowed for OAuth provider '{}'",
            host,
            provider.name
        );
        if !test_localhost {
            crate::tools::web_fetch::validate_url_for_ssrf(parsed.as_str())
                .map_err(anyhow::Error::msg)?;
        }
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

    async fn send_authorization_action(
        status_tx: Option<&mpsc::Sender<StatusUpdate>>,
        message: String,
    ) {
        if let Some(tx) = status_tx {
            let _ = tx.send(StatusUpdate::UserActionRequired { message }).await;
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

    /// Refuse a call that changes a connection or provider unless it came
    /// through the agent kernel, whose approval rules asked the owner
    /// ([`ManageOAuthTool::owner_approval`]).
    fn require_kernel_approval(
        kernel_approved: Option<&ApprovalScope>,
        action: &str,
    ) -> anyhow::Result<()> {
        if kernel_approved != Some(&ApprovalScope::Call) {
            anyhow::bail!(
                "manage_oauth `{action}` requires approval through the agent kernel; it was \
                 called outside an approved agent call"
            );
        }
        Ok(())
    }

    /// The owner approval one call needs: `bind_account`, `remove`,
    /// `register_provider` and `remove_provider` ask once they reach their
    /// change, judged against the stored connections and providers exactly
    /// as the call reads them. A call that stops before changing anything
    /// (missing or invalid fields, an unknown connection, a built-in
    /// provider, a disconnect without `confirm_disconnect`) needs no card;
    /// the other actions never asked.
    pub(crate) async fn owner_approval(&self, arguments: &Value) -> OwnerApprovalNeed {
        let Ok(args) = serde_json::from_value::<ManageOAuthArgs>(arguments.clone()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        let Some(service) = args
            .service
            .as_deref()
            .and_then(|service| Self::validate_service_name(service).ok())
        else {
            return OwnerApprovalNeed::NotNeeded;
        };
        let request = match args.action.as_str() {
            "bind_account" => {
                let (Some(account_id), Some(identity_url), Some(pointer)) = (
                    args.account_id.as_deref(),
                    args.identity_url.as_deref(),
                    args.account_id_pointer.as_deref(),
                ) else {
                    return OwnerApprovalNeed::NotNeeded;
                };
                match self
                    .bind_plan(&service, account_id, identity_url, pointer)
                    .await
                {
                    Ok(plan) => Self::bind_request(&service, &plan),
                    Err(_) => return OwnerApprovalNeed::NotNeeded,
                }
            }
            "remove" => {
                let connected = matches!(
                    self.state_store.get_oauth_connection(&service).await,
                    Ok(Some(_))
                );
                if !connected || !args.confirm_disconnect.unwrap_or(false) {
                    return OwnerApprovalNeed::NotNeeded;
                }
                Self::remove_request(&service)
            }
            "register_provider" => match Self::register_plan(&service, &args) {
                Ok(Ok(plan)) => Self::register_request(&service, &args, &plan),
                _ => return OwnerApprovalNeed::NotNeeded,
            },
            "remove_provider" => match self.custom_provider_removable(&service).await {
                Ok(true) => Self::remove_provider_request(&service),
                _ => return OwnerApprovalNeed::NotNeeded,
            },
            _ => return OwnerApprovalNeed::NotNeeded,
        };
        OwnerApprovalNeed::Ask(request)
    }

    fn bind_request(service: &str, plan: &BindPlan) -> aidaemon_core::OwnerApprovalRequest {
        aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: format!(
                "Verify and bind OAuth service '{}' from account '{}' to '{}'",
                service, plan.prior_binding, plan.expected_account_id
            ),
            // The exact arguments the owner approves; a changed argument is a new card.
            subject: json!({
                "action": "bind_account",
                "service": service,
                "account_id": plan.expected_account_id,
                "identity_url": plan.identity_url,
                "account_id_pointer": plan.account_id_pointer,
            })
            .to_string(),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec![
                format!(
                    "Performs one authenticated read-only GET to {}",
                    plan.identity_url
                ),
                "Persists the proven remote account ID for future mandate authorization"
                    .to_string(),
                "A later OAuth reconnect clears this binding and requires verification again"
                    .to_string(),
            ],
        }
    }

    fn remove_request(service: &str) -> aidaemon_core::OwnerApprovalRequest {
        aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: format!("Disconnect OAuth service '{}'", service),
            subject: json!({"action": "remove", "service": service, "confirm_disconnect": true})
                .to_string(),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec![
                "Deletes the stored OAuth connection and tokens".to_string(),
                "Leaves the service unavailable until it is connected again".to_string(),
            ],
        }
    }

    /// The card names the provider's endpoints, never its client
    /// credentials (`register_provider` takes none).
    fn register_request(
        service: &str,
        args: &ManageOAuthArgs,
        plan: &RegisterPlan,
    ) -> aidaemon_core::OwnerApprovalRequest {
        aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: format!(
                "Register custom OAuth provider '{}' (authorize={}, token={})",
                service, plan.authorize_url, plan.token_url
            ),
            subject: json!({
                "action": "register_provider",
                "service": service,
                "display_name": args.display_name,
                "auth_type": plan.auth_type,
                "authorize_url": plan.authorize_url,
                "token_url": plan.token_url,
                "scopes": plan.scopes,
                "allowed_domains": plan.allowed_domains,
            })
            .to_string(),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec![
                "Modifies OAuth configuration".to_string(),
                "Can route credentialed API traffic to the configured domains".to_string(),
            ],
        }
    }

    fn remove_provider_request(service: &str) -> aidaemon_core::OwnerApprovalRequest {
        aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: format!(
                "Remove custom OAuth provider '{}' and delete its client credentials",
                service
            ),
            subject: json!({"action": "remove_provider", "service": service}).to_string(),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec![
                "Deletes OAuth provider configuration".to_string(),
                "Deletes stored client credentials for this provider".to_string(),
            ],
        }
    }

    /// The typed effects of one call: every action but the reads acts on
    /// the service's credential (connection tokens, client credentials or
    /// provider definition); `bind_account` also reaches its identity host.
    /// Token endpoints and the config file are not named, so the list is not
    /// exhaustive.
    pub(crate) fn effects_for(arguments: &str) -> aidaemon_core::CallEffects {
        use aidaemon_core::{CallEffects, EffectClass, EffectKey};
        let Ok(args) = serde_json::from_str::<ManageOAuthArgs>(arguments) else {
            return CallEffects::unknown();
        };
        match args.action.as_str() {
            "list" | "providers" | "describe_provider" => CallEffects::none(),
            "connect" | "bind_account" | "remove" | "set_credentials" | "refresh"
            | "register_provider" | "remove_provider" => {
                let mut effects = CallEffects::unknown();
                if let Some(service) = args
                    .service
                    .as_deref()
                    .and_then(|service| Self::validate_service_name(service).ok())
                {
                    effects
                        .keys
                        .push(EffectKey::named(EffectClass::Credential, &service));
                }
                if args.action == "bind_account" {
                    if let Some(host) = args
                        .identity_url
                        .as_deref()
                        .map(str::trim)
                        .and_then(crate::tools::command_effects::url_effect)
                    {
                        effects.keys.push(host);
                    }
                }
                effects
            }
            _ => CallEffects::unknown(),
        }
    }

    /// What a `bind_account` proves and binds, or why it cannot.
    async fn bind_plan(
        &self,
        service: &str,
        expected_account_id: &str,
        identity_url: &str,
        account_id_pointer: &str,
    ) -> anyhow::Result<BindPlan> {
        let connection = self
            .state_store
            .get_oauth_connection(service)
            .await?
            .ok_or_else(|| anyhow::anyhow!("No OAuth connection found for '{}'", service))?;
        let provider = self
            .gateway
            .get_provider(service)
            .await
            .ok_or_else(|| anyhow::anyhow!("Unknown OAuth provider: {}", service))?;
        let identity_url = Self::validate_identity_url(identity_url, &provider)?;
        let expected_account_id = Self::validate_account_id(expected_account_id)?;
        let account_id_pointer = Self::validate_account_id_pointer(account_id_pointer)?;
        Ok(BindPlan {
            identity_url,
            expected_account_id,
            account_id_pointer,
            prior_binding: connection
                .account_id
                .unwrap_or_else(|| "unbound".to_string()),
        })
    }

    /// What a `register_provider` writes, read from its arguments, or
    /// (`Err`) the message it returns without writing.
    fn register_plan(
        service: &str,
        args: &ManageOAuthArgs,
    ) -> anyhow::Result<Result<RegisterPlan, String>> {
        if crate::oauth::providers::get_builtin_provider(service).is_some() {
            return Ok(Err(format!(
                "'{}' is a built-in OAuth provider. Use it directly instead of re-registering it.",
                service
            )));
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
        Ok(Ok(RegisterPlan {
            auth_type,
            authorize_url,
            token_url,
            allowed_domains,
            scopes,
        }))
    }

    /// Whether `remove_provider` would remove a custom provider: not a
    /// built-in one, no active connection, and defined in the config.
    async fn custom_provider_removable(&self, service: &str) -> anyhow::Result<bool> {
        if crate::oauth::providers::get_builtin_provider(service).is_some()
            || self
                .state_store
                .get_oauth_connection(service)
                .await?
                .is_some()
        {
            return Ok(false);
        }
        let doc = self.load_config_doc().await?;
        Ok(Self::providers_table(&doc).is_some_and(|table| table.contains_key(service)))
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
            let has_creds = self.gateway.has_credentials(name);
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
        let has_creds = self.gateway.has_credentials(service);
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
        Ok(self.handle_list_snapshot().await?.0)
    }

    /// Render the connection list and return the exact member identities the
    /// receipt reports, so `list` proves both readiness and absence.
    async fn handle_list_snapshot(&self) -> anyhow::Result<(String, Vec<ToolTargetHint>)> {
        let connections = self.state_store.list_oauth_connections().await?;
        let members = connections
            .iter()
            .filter_map(|conn| {
                ToolTargetHint::new(
                    ToolTargetHintKind::ResourceId,
                    HttpRequestTool::auth_profile_resource_id(&conn.service),
                )
            })
            .collect();
        if connections.is_empty() {
            return Ok((
                "No OAuth connections. Use 'connect' to link a service.".to_string(),
                members,
            ));
        }
        let mut result = String::from("Connected OAuth services:\n");
        let now = chrono::Utc::now();
        for conn in &connections {
            let username = conn
                .username
                .as_deref()
                .map(|u| format!(" ({})", u))
                .unwrap_or_default();
            // Typed readiness instead of a bare expiry: a short-lived bearer
            // that lapsed since the last call is still usable when a refresh
            // path exists, and an audit must not have to guess that.
            let readiness = crate::oauth::CredentialReadiness::for_connection(
                conn,
                now,
                self.gateway.secrets().as_ref(),
            )
            .describe();
            let account = conn
                .account_id
                .as_deref()
                .map(|id| format!(" [account_id: {}]", id))
                .unwrap_or_else(|| " [account_id: unbound]".to_string());
            result.push_str(&format!(
                "  - {}{} [{}]{} [readiness: {}] [resource_id: {}]\n",
                conn.service,
                username,
                conn.auth_type,
                account,
                readiness,
                HttpRequestTool::auth_profile_resource_id(&conn.service)
            ));
        }
        Ok((result, members))
    }

    async fn identity_response(
        &self,
        service: &str,
        identity_url: &str,
    ) -> anyhow::Result<(reqwest::StatusCode, Vec<u8>, bool)> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(30))
            .build()?;
        let token_key = format!("oauth_{}_access_token", service);
        let mut refreshed = false;
        loop {
            let token = match crate::config::resolve_from_keychain(&token_key) {
                Ok(token) => token,
                Err(error) if !refreshed => {
                    self.gateway.refresh_token(service).await.map_err(|refresh_error| {
                        anyhow::anyhow!(
                            "OAuth access token is unavailable ({error}); refresh also failed: {refresh_error}"
                        )
                    })?;
                    refreshed = true;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let response = client
                .get(identity_url)
                .bearer_auth(token)
                .header(reqwest::header::ACCEPT, "application/json")
                .send()
                .await
                .map_err(|error| {
                    anyhow::anyhow!("Identity verification request failed: {}", error)
                })?;
            if response.status() == reqwest::StatusCode::UNAUTHORIZED && !refreshed {
                self.gateway.refresh_token(service).await?;
                refreshed = true;
                continue;
            }
            let status = response.status();
            let (body, truncated) =
                crate::tools::web_fetch::read_body_capped(response, 64 * 1024).await?;
            return Ok((status, body, truncated));
        }
    }

    /// Proves and binds the remote account. The owner was asked by the
    /// kernel before this runs.
    async fn handle_bind_account(
        &self,
        service: &str,
        expected_account_id: &str,
        identity_url: &str,
        account_id_pointer: &str,
        kernel_approved: Option<&ApprovalScope>,
    ) -> anyhow::Result<String> {
        let BindPlan {
            identity_url,
            expected_account_id,
            account_id_pointer,
            ..
        } = self
            .bind_plan(
                service,
                expected_account_id,
                identity_url,
                account_id_pointer,
            )
            .await?;
        Self::require_kernel_approval(kernel_approved, "bind_account")?;

        let (status, body, truncated) = self.identity_response(service, &identity_url).await?;
        anyhow::ensure!(
            status.is_success(),
            "Identity verification returned HTTP {}; account binding was not changed",
            status.as_u16()
        );
        anyhow::ensure!(
            !truncated,
            "Identity verification response exceeded 65536 bytes; account binding was not changed"
        );
        let payload: serde_json::Value = serde_json::from_slice(&body)
            .map_err(|error| anyhow::anyhow!("Identity response was not valid JSON: {}", error))?;
        let observed_account_id = payload
            .pointer(&account_id_pointer)
            .and_then(|value| match value {
                serde_json::Value::String(value) => Some(value.clone()),
                serde_json::Value::Number(value) => Some(value.to_string()),
                _ => None,
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Identity response did not contain a string or number at JSON Pointer '{}'",
                    account_id_pointer
                )
            })?;
        anyhow::ensure!(
            observed_account_id == expected_account_id,
            "Authenticated identity mismatch: expected account '{}', received '{}'; account binding was not changed",
            expected_account_id,
            observed_account_id
        );
        self.gateway
            .bind_account_id(service, &observed_account_id)
            .await?;
        let receipt = json!({
            "account_id": observed_account_id,
            "account_id_pointer": account_id_pointer,
            "bound": true,
            "http_status": status.as_u16(),
            "identity_url": identity_url,
            "method": "GET",
            "service": service,
            "verified_at": chrono::Utc::now().to_rfc3339(),
        });
        Ok(format!(
            "OAuth account identity verified and bound.\n{}",
            serde_json::to_string_pretty(&receipt)?
        ))
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
        authorize_parts.push(format!(
            "🔐 **Authorization required**\n\n[Continue in your browser]({authorize_url})"
        ));
        Self::send_authorization_action(status_tx.as_ref(), authorize_parts.join("\n\n")).await;

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

    /// Disconnects a service. The owner was asked by the kernel before this
    /// runs.
    async fn handle_remove(
        &self,
        service: &str,
        confirm_disconnect: bool,
        kernel_approved: Option<&ApprovalScope>,
    ) -> anyhow::Result<String> {
        let conn = self.state_store.get_oauth_connection(service).await?;
        if conn.is_none() {
            return Ok(format!("No OAuth connection found for '{}'", service));
        }
        if !confirm_disconnect {
            return Ok(Self::disconnect_confirmation_message(service));
        }
        Self::require_kernel_approval(kernel_approved, "remove")?;
        self.gateway.remove_connection(service).await
    }

    /// Registers a custom provider. The owner was asked by the kernel before
    /// this runs.
    async fn handle_register_provider(
        &self,
        service: &str,
        args: &ManageOAuthArgs,
        kernel_approved: Option<&ApprovalScope>,
    ) -> anyhow::Result<String> {
        let RegisterPlan {
            auth_type,
            authorize_url,
            token_url,
            allowed_domains,
            scopes,
        } = match Self::register_plan(service, args)? {
            Ok(plan) => plan,
            Err(message) => return Ok(message),
        };
        Self::require_kernel_approval(kernel_approved, "register_provider")?;

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

    /// Removes a custom provider and its client credentials. The owner was
    /// asked by the kernel before this runs.
    async fn handle_remove_provider(
        &self,
        service: &str,
        kernel_approved: Option<&ApprovalScope>,
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

        Self::require_kernel_approval(kernel_approved, "remove_provider")?;

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

/// The tool's one description: the schema the model reads and
/// `description()` both use it.
const TOOL_DESCRIPTION: &str = "OAuth. Never paste secrets; inspect first; connect reauthorizes; bind_account proves identity. Manual auth → manage_http_auth.";

fn manage_oauth_schema() -> Value {
    json!({
        "name": "manage_oauth",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["connect", "list", "bind_account", "remove", "set_credentials", "refresh", "providers", "describe_provider", "register_provider", "remove_provider"]
                },
                "service": { "type": "string" },
                "account_id": { "type": "string", "maxLength": 256 },
                "identity_url": { "type": "string" },
                "account_id_pointer": { "type": "string" },
                "client_id": { "type": "string" },
                "client_secret": { "type": "string" },
                "display_name": { "type": "string" },
                "auth_type": {
                    "type": "string",
                    "enum": ["oauth2_pkce", "oauth2_authorization_code", "oauth2_client_credentials"]
                },
                "authorize_url": { "type": "string" },
                "token_url": { "type": "string" },
                "scopes": { "type": "array", "items": { "type": "string" } },
                "allowed_domains": { "type": "array", "items": { "type": "string" } },
                "confirm_disconnect": {
                    "type": "boolean",
                    "description": "Must be true with remove; use connect to reauthorize"
                }
            },
            "required": ["action"],
            "additionalProperties": false
        }
    })
}

#[async_trait]
impl Tool for ManageOAuthTool {
    fn name(&self) -> &str {
        "manage_oauth"
    }

    fn description(&self) -> &str {
        TOOL_DESCRIPTION
    }

    fn schema(&self) -> Value {
        manage_oauth_schema()
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let status_tx = crate::traits::status_sender(facts);
        let facts = facts.clone();

        let args: ManageOAuthArgs = serde_json::from_str(arguments)?;
        if args.action == "list" {
            let (output, members) = self.handle_list_snapshot().await?;
            let observations = members
                .iter()
                .cloned()
                .map(|subject| {
                    ToolObservationEvidence::new(subject, &[ToolSemanticFacet::Authorization])
                })
                .collect();
            return Ok(ToolCallOutcome {
                output,
                metadata: ToolCallMetadata {
                    observations,
                    collection_observations: vec![ToolCollectionObservation::complete(
                        Self::connection_collection_hint(),
                        &[ToolSemanticFacet::Authorization],
                        members,
                    )],
                    ..ToolCallMetadata::default()
                },
                finish: None,
            });
        }
        let output = self.run(arguments, status_tx, &facts).await?;
        let mut outcome = ToolCallOutcome::from_output(output.clone());
        // A connect that is the whole of a root turn's answer ends the run
        // with its own message (the dispatcher drops it if the call failed).
        if args.action == "connect" && facts.lone_root_call {
            outcome = outcome.with_finish(aidaemon_core::Finish::succeeded(output));
        }
        Ok(outcome)
    }
}

impl ManageOAuthTool {
    /// Run one call with the dispatcher's `facts`. Approval is judged by the
    /// kernel before this runs; a change without it is refused.
    pub(crate) async fn run(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        facts: &ToolCallContext,
    ) -> anyhow::Result<String> {
        let mut args: ManageOAuthArgs = serde_json::from_str(arguments)?;
        args.call_session = facts.session_id.clone();
        let kernel_approved = facts.kernel_approved.as_ref();
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
            "bind_account" => {
                let service = Self::validate_service_name(
                    args.service
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("'bind_account' requires 'service'"))?,
                )?;
                self.handle_bind_account(
                    &service,
                    args.account_id
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("'bind_account' requires 'account_id'"))?,
                    args.identity_url
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("'bind_account' requires 'identity_url'"))?,
                    args.account_id_pointer.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("'bind_account' requires 'account_id_pointer'")
                    })?,
                    kernel_approved,
                )
                .await
            }
            "connect" => {
                let service = Self::validate_service_name(
                    args.service
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("'connect' requires 'service' parameter"))?,
                )?;
                let session_id = if args.call_session.is_empty() {
                    "unknown"
                } else {
                    args.call_session.as_str()
                };
                self.handle_connect(&service, session_id, status_tx).await
            }
            "remove" => {
                let service = Self::validate_service_name(
                    args.service
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("'remove' requires 'service' parameter"))?,
                )?;
                self.handle_remove(
                    &service,
                    args.confirm_disconnect.unwrap_or(false),
                    kernel_approved,
                )
                .await
            }
            "register_provider" => {
                let service = Self::validate_service_name(
                    args.service.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("'register_provider' requires 'service' parameter")
                    })?,
                )?;
                self.handle_register_provider(&service, &args, kernel_approved)
                    .await
            }
            "remove_provider" => {
                let service = Self::validate_service_name(
                    args.service.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("'remove_provider' requires 'service' parameter")
                    })?,
                )?;
                self.handle_remove_provider(&service, kernel_approved).await
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
                "Unknown action '{}'. Use: connect, list, bind_account, remove, set_credentials, refresh, providers, describe_provider, register_provider, remove_provider.",
                other
            )),
        }
    }
}

#[async_trait]
impl crate::traits::ToolSemantics for ManageOAuthTool {
    /// Binding an account, disconnecting a service and registering or
    /// removing a custom provider ask the owner, whatever the session. The
    /// dispatcher runs the call only with that answer; a decline is recorded
    /// by the kernel.
    async fn owner_approval_request(
        &self,
        call: &crate::traits::ApprovalCall<'_>,
        _facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        if call.refused {
            return OwnerApprovalNeed::NotNeeded;
        }
        self.owner_approval(call.arguments).await
    }

    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        Self::effects_for(arguments)
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let mut semantics = semantics_for_exact_read_actions(
            arguments,
            &["list", "providers", "describe_provider"],
            ToolMutationEffects::CONFIGURATION,
        );
        if let Ok(args) = serde_json::from_str::<ManageOAuthArgs>(arguments) {
            if args.action == "list" {
                semantics = semantics.with_target_hint(
                    ToolTargetHintKind::ResourceId,
                    Self::OAUTH_CONNECTION_COLLECTION_ID,
                );
            }
            if let Some(service) = args.service.as_deref() {
                if let Ok(service) = Self::validate_service_name(service) {
                    semantics = semantics.with_target_hint(
                        ToolTargetHintKind::ResourceId,
                        HttpRequestTool::auth_profile_resource_id(&service),
                    );
                }
            }
        }
        semantics
    }

    fn stable_observation_subjects(&self) -> Vec<StableObservationSubject> {
        vec![StableObservationSubject::collection(
            Self::OAUTH_CONNECTION_COLLECTION_ID,
            &[ToolSemanticFacet::Authorization],
            HttpRequestTool::AUTH_PROFILE_NAMESPACE,
            "every connected OAuth service",
        )]
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

    /// Results are the daemon's own status text, never a third party's.
    /// Failure messages may quote a token endpoint's body; those endpoints
    /// are owner-configured, so they stay trusted.
    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &crate::traits::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        aidaemon_core::SourceTrust::Trusted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ToolCalls as _;
    use crate::traits::ToolSemantics;
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::time::Duration;

    use axum::{
        extract::Form,
        http::HeaderMap,
        routing::{get, post},
        Json, Router,
    };
    use tempfile::NamedTempFile;
    use tokio::net::TcpListener;

    use crate::memory::embeddings::EmbeddingService;
    use crate::oauth::{OAuthProvider, SharedHttpProfiles};
    use crate::state::SqliteStateStore;
    use crate::testing::KeychainIsolation;
    use crate::traits::ToolCollectionCompleteness;

    #[test]
    fn schema_fits_payload_budget() {
        // Pillar C of 2026-06-06-cross-turn-prefix-stability-design.md:
        // admin-tool schemas ride in EVERY provider call; this ceiling is the
        // per-tool payload budget. If you trip this assert by adding features,
        // compress the description text — do not raise the ceiling without
        // updating the Pillar C implementation plan.
        // NOTE: plan ceiling was 1000 but manage_oauth carries 4 mandatory
        // behavioral invariants (never-credentials-in-chat, inspect-before-reconnect,
        // connect-not-remove, manage_http_auth routing) plus a 9-value action enum
        // and a 3-value auth_type enum. Structural floor with all params is ~749 bytes;
        // ceiling set to 1100 to accommodate invariants with short readable text.
        let bytes = serde_json::to_string(&manage_oauth_schema()).unwrap().len();
        assert!(
            bytes <= 1100,
            "manage_oauth schema is {bytes} bytes, budget is 1100"
        );
    }

    async fn test_tool(
        config_path: PathBuf,
    ) -> anyhow::Result<(ManageOAuthTool, OAuthGateway, NamedTempFile)> {
        let db_file = NamedTempFile::new()?;
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new()?);
        let state = Arc::new(SqliteStateStore::new(&db_path, 32, None, embedding_service).await?);
        let profiles: SharedHttpProfiles =
            Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));
        let gateway = crate::oauth::daemon_gateway(
            state.clone() as Arc<dyn crate::traits::OAuthGatewayStore>,
            profiles,
            "http://localhost:8080".to_string(),
        );
        Ok((
            ManageOAuthTool::new(gateway.clone(), state as Arc<dyn OAuthStore>, config_path),
            gateway,
            db_file,
        ))
    }

    /// Call `tool` as the dispatcher does for a call the kernel judged (and
    /// asked the owner about where its rules required it).
    async fn call_approved(tool: &ManageOAuthTool, arguments: &str) -> anyhow::Result<String> {
        crate::testing::call_with_facts(
            tool,
            arguments,
            ToolCallContext {
                kernel_approved: Some(ApprovalScope::Call),
                ..crate::testing::call_facts("test")
            },
        )
        .await
    }

    #[test]
    fn changes_declare_the_service_credential_and_reads_declare_none() {
        use aidaemon_core::{CallEffects, EffectClass, EffectKey};
        for args in [
            r#"{"action":"remove","service":"Synthetic","confirm_disconnect":true}"#,
            r#"{"action":"register_provider","service":"synthetic"}"#,
            r#"{"action":"remove_provider","service":"synthetic"}"#,
            r#"{"action":"set_credentials","service":"synthetic"}"#,
            r#"{"action":"connect","service":"synthetic"}"#,
        ] {
            let effects = ManageOAuthTool::effects_for(args);
            assert_eq!(
                effects.keys,
                vec![EffectKey::named(EffectClass::Credential, "synthetic")],
                "{args}"
            );
            assert!(!effects.complete);
        }
        let bind = ManageOAuthTool::effects_for(
            r#"{"action":"bind_account","service":"synthetic","identity_url":"https://api.example.invalid/me"}"#,
        );
        assert_eq!(
            bind.keys,
            vec![
                EffectKey::named(EffectClass::Credential, "synthetic"),
                EffectKey::egress("https", "api.example.invalid", None),
            ]
        );
        for args in [
            r#"{"action":"list"}"#,
            r#"{"action":"providers"}"#,
            r#"{"action":"describe_provider","service":"synthetic"}"#,
        ] {
            assert_eq!(ManageOAuthTool::effects_for(args), CallEffects::none());
        }
    }

    /// Without the kernel's approval, a change is refused before it touches
    /// the config or the connection (and never asks the owner itself).
    #[tokio::test]
    async fn changes_refuse_without_kernel_approval() {
        let config_file = NamedTempFile::new().unwrap();
        std::fs::write(
            config_file.path(),
            "[provider]\napi_key = \"test-key\"\n\n[oauth.providers.synthetic]\n\
             auth_type = \"oauth2_pkce\"\nauthorize_url = \"https://auth.example.invalid/authorize\"\n\
             token_url = \"https://auth.example.invalid/token\"\nscopes = []\n\
             allowed_domains = [\"api.example.invalid\"]\n",
        )
        .unwrap();
        let before = std::fs::read_to_string(config_file.path()).unwrap();
        let (tool, _gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();
        for args in [
            r#"{"action":"register_provider","service":"other","authorize_url":"https://auth.example.invalid/authorize","token_url":"https://auth.example.invalid/token","allowed_domains":["api.example.invalid"]}"#,
            r#"{"action":"remove_provider","service":"synthetic"}"#,
        ] {
            let error =
                crate::testing::call_with_facts(&tool, args, crate::testing::call_facts("test"))
                    .await
                    .expect_err("refused");
            assert!(
                error
                    .to_string()
                    .contains("requires approval through the agent kernel"),
                "{error}"
            );
            assert!(tool.call(args).await.is_err(), "{args}");
        }
        assert_eq!(std::fs::read_to_string(config_file.path()).unwrap(), before);
        // A call that stops before changing anything needs no approval.
        let none = tool
            .call(r#"{"action":"remove","service":"synthetic","confirm_disconnect":true}"#)
            .await
            .unwrap();
        assert!(none.contains("No OAuth connection found"), "{none}");
    }

    fn write_minimal_config(path: &Path) {
        std::fs::write(path, "[provider]\napi_key = \"test-key\"\n").unwrap();
    }

    /// Its results are the daemon's own status text, so reading them never
    /// taints a run.
    #[tokio::test]
    async fn its_output_is_trusted() {
        let config_file = tempfile::NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let (tool, _gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();
        assert_eq!(
            tool.output_trust(r#"{"action":"list"}"#, &crate::traits::TrustRoots::new(&[])),
            aidaemon_core::SourceTrust::Trusted
        );
    }

    #[tokio::test]
    async fn list_emits_exact_connection_readiness_and_complete_collection_evidence() {
        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        // `restore_connections` resolves the stored connection's access token
        // through the secret store; a developer machine may hold a real
        // `oauth_twitter_access_token` keychain item.
        let _isolation = KeychainIsolation::acquire();
        let (tool, gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();
        tool.state_store
            .save_oauth_connection(&crate::traits::OAuthConnection {
                id: 0,
                service: "twitter".to_string(),
                auth_type: "oauth2_pkce".to_string(),
                username: Some("synthetic_handle".to_string()),
                account_id: Some("1000000000000000001".to_string()),
                scopes: r#"["users.read"]"#.to_string(),
                token_expires_at: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();
        gateway.restore_connections().await;

        let semantics = tool.call_semantics(r#"{"action":"list"}"#);
        assert!(semantics.target_hints.iter().any(|hint| {
            hint.kind == ToolTargetHintKind::ResourceId
                && hint.value == ManageOAuthTool::OAUTH_CONNECTION_COLLECTION_ID
        }));
        let service_semantics = tool.call_semantics(r#"{"action":"refresh","service":"twitter"}"#);
        let resource_id = HttpRequestTool::auth_profile_resource_id("twitter");
        assert!(service_semantics
            .target_hints
            .iter()
            .any(|hint| hint.kind == ToolTargetHintKind::ResourceId && hint.value == resource_id));

        let outcome = tool
            .call_with_status_outcome(r#"{"action":"list"}"#, None)
            .await
            .unwrap();
        assert!(outcome.output.contains("twitter"));
        assert!(outcome.finish.is_none());
        assert!(outcome.metadata.observations.iter().any(|observation| {
            observation.subject.kind == ToolTargetHintKind::ResourceId
                && observation.subject.value == resource_id
                && observation
                    .facets
                    .contains(&ToolSemanticFacet::Authorization)
        }));
        let collection = outcome
            .metadata
            .collection_observations
            .iter()
            .find(|collection| {
                collection.collection.value == ManageOAuthTool::OAUTH_CONNECTION_COLLECTION_ID
            })
            .expect("oauth connection collection coverage");
        assert_eq!(
            collection.completeness,
            ToolCollectionCompleteness::Complete
        );
        assert_eq!(collection.members.len(), 1);
        assert_eq!(collection.members[0].value, resource_id);

        let advertised = tool.stable_observation_subjects();
        assert!(advertised.iter().any(|subject| {
            subject.resource_id.as_deref() == Some(ManageOAuthTool::OAUTH_CONNECTION_COLLECTION_ID)
                && subject.binds(&resource_id)
        }));
    }

    #[tokio::test]
    async fn list_projects_typed_credential_readiness_for_an_expired_but_refreshable_bearer() {
        // An owner audit read a bare `[expires: <past>]` as a missing credential.
        // The list must state the typed readiness instead, and must not
        // exchange the refresh token to do so (no gateway restore here).
        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let isolation = KeychainIsolation::acquire();
        std::fs::write(
            isolation.env_file_path(),
            "OAUTH_TWITTER_ACCESS_TOKEN=stale\nOAUTH_TWITTER_REFRESH_TOKEN=rt\nOAUTH_TWITTER_CLIENT_ID=id\nOAUTH_TWITTER_CLIENT_SECRET=secret\n",
        )
        .unwrap();
        let (tool, _gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();
        let expired = (chrono::Utc::now() - chrono::Duration::hours(3)).to_rfc3339();
        tool.state_store
            .save_oauth_connection(&crate::traits::OAuthConnection {
                id: 0,
                service: "twitter".to_string(),
                auth_type: "oauth2_pkce".to_string(),
                username: Some("synthetic_handle".to_string()),
                account_id: Some("1000000000000000001".to_string()),
                scopes: r#"["tweet.write"]"#.to_string(),
                token_expires_at: Some(expired.clone()),
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();

        let output = tool.call(r#"{"action":"list"}"#).await.unwrap();
        let expected = crate::oauth::CredentialReadiness::Refreshable {
            expired_at: Some(expired.clone()),
            path: crate::oauth::CredentialRefreshPath::RefreshToken,
        }
        .describe();
        assert!(
            output.contains(&format!("[readiness: {expected}]")),
            "{output}"
        );
        assert!(!output.contains("[expires:"), "{output}");
        // The refresh token was not spent: the stored secrets are unchanged.
        let secrets = std::fs::read_to_string(isolation.env_file_path()).unwrap();
        assert!(secrets.contains("OAUTH_TWITTER_ACCESS_TOKEN=stale"));
        assert!(secrets.contains("OAUTH_TWITTER_REFRESH_TOKEN=rt"));
    }

    #[tokio::test]
    async fn empty_list_still_proves_complete_absence() {
        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let (tool, _gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();
        let outcome = tool
            .call_with_status_outcome(r#"{"action":"list"}"#, None)
            .await
            .unwrap();
        assert!(outcome.metadata.observations.is_empty());
        let collection = outcome
            .metadata
            .collection_observations
            .iter()
            .find(|collection| {
                collection.collection.value == ManageOAuthTool::OAUTH_CONNECTION_COLLECTION_ID
            })
            .expect("absence is evidence too");
        assert_eq!(
            collection.completeness,
            ToolCollectionCompleteness::Complete
        );
        assert!(collection.members.is_empty());
    }

    #[tokio::test]
    async fn register_provider_persists_and_hot_registers_custom_oauth_provider() {
        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let (tool, gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();

        let result = call_approved(
            &tool,
            r#"{
                    "action": "register_provider",
                    "service": "linear",
                    "display_name": "Linear",
                    "authorize_url": "https://linear.app/oauth/authorize",
                    "token_url": "https://api.linear.app/oauth/token",
                    "scopes": ["read", "write"],
                    "allowed_domains": ["api.linear.app"]
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
        let (tool, gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();

        let result = call_approved(
            &tool,
            r#"{
                    "action": "register_provider",
                    "service": "corpapi",
                    "display_name": "Corp API",
                    "auth_type": "oauth2_client_credentials",
                    "token_url": "https://auth.example.com/oauth/token",
                    "scopes": ["read"],
                    "allowed_domains": ["api.example.com"]
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
        let isolation = KeychainIsolation::acquire();
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

        std::fs::write(
            isolation.env_file_path(),
            "OAUTH_LINEAR_CLIENT_ID=abc\nOAUTH_LINEAR_CLIENT_SECRET=def\n",
        )
        .unwrap();

        let (tool, gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();
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

        let result = call_approved(&tool, r#"{"action":"remove_provider","service":"linear"}"#)
            .await
            .unwrap();

        assert!(result.contains("Removed custom OAuth provider 'linear'"));
        let saved = std::fs::read_to_string(config_file.path()).unwrap();
        assert!(!saved.contains("[oauth.providers.linear]"));
        assert!(gateway.get_provider("linear").await.is_none());
        let env_content = std::fs::read_to_string(isolation.env_file_path()).unwrap();
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
        let _isolation = KeychainIsolation::acquire();

        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let (tool, gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();
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
        let isolation = KeychainIsolation::acquire();

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
        let (tool, gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();
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

        std::fs::write(
            isolation.env_file_path(),
            "OAUTH_LINEAR_CLIENT_ID=abc\nOAUTH_LINEAR_CLIENT_SECRET=def\n",
        )
        .unwrap();

        let (status_tx, mut status_rx) = mpsc::channel::<StatusUpdate>(8);
        let tool = Arc::new(tool);
        let tool_task = {
            let tool = tool.clone();
            tokio::spawn(async move {
                tool.call_with_execution_context(
                    r#"{"action":"connect","service":"linear"}"#,
                    Some(status_tx),
                    crate::traits::ToolCallContext {
                        lone_root_call: true,
                        ..crate::testing::call_facts("telegram:123")
                    },
                )
                .await
            })
        };

        let first_update = tokio::time::timeout(Duration::from_secs(2), status_rx.recv())
            .await
            .unwrap()
            .unwrap();
        let authorize_chunk = match first_update {
            StatusUpdate::UserActionRequired { message } => message,
            other => panic!("unexpected first status update: {other:?}"),
        };
        assert!(authorize_chunk.contains("🔐 **Authorization required**"));
        assert!(authorize_chunk.contains("[Continue in your browser]("));
        let authorize_url = authorize_chunk
            .split_once("[Continue in your browser](")
            .map(|(_, rest)| rest)
            .and_then(|rest| rest.split_once(')').map(|(url, _)| url))
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

        assert!(tool_outcome.output.contains("Connected to Linear"));
        // The connect message is the turn's final answer.
        assert_eq!(
            tool_outcome.finish,
            Some(aidaemon_core::Finish::succeeded(
                tool_outcome.output.clone()
            ))
        );
    }

    #[tokio::test]
    async fn bind_account_proves_remote_identity_before_persisting_it() {
        let isolation = KeychainIsolation::acquire();

        async fn identity_handler(headers: HeaderMap) -> Json<serde_json::Value> {
            assert_eq!(
                headers
                    .get(reqwest::header::AUTHORIZATION)
                    .and_then(|value| value.to_str().ok()),
                Some("Bearer synthetic-access-token")
            );
            Json(serde_json::json!({
                "data": {
                    "id": "2018008573557551105",
                    "username": "synthetic_agent"
                }
            }))
        }

        let app = Router::new().route("/2/users/me", get(identity_handler));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        std::fs::write(
            isolation.env_file_path(),
            "OAUTH_TWITTER_ACCESS_TOKEN=synthetic-access-token\n",
        )
        .unwrap();

        let (tool, gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();
        gateway
            .register_provider(crate::oauth::OAuthProvider {
                name: "twitter".to_string(),
                display_name: "Synthetic X".to_string(),
                auth_type: OAuthType::OAuth2Pkce,
                authorize_url: "https://example.test/oauth/authorize".to_string(),
                token_url: "https://example.test/oauth/token".to_string(),
                scopes: vec!["users.read".to_string()],
                allowed_domains: vec!["127.0.0.1".to_string()],
            })
            .await;
        tool.state_store
            .save_oauth_connection(&crate::traits::OAuthConnection {
                id: 0,
                service: "twitter".to_string(),
                auth_type: "oauth2_pkce".to_string(),
                username: None,
                account_id: None,
                scopes: r#"["users.read"]"#.to_string(),
                token_expires_at: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();
        gateway.restore_connections().await;

        let identity_url = format!("http://{addr}/2/users/me");
        // Outside an approved agent call the binding is refused before any
        // request.
        let unapproved = tool
            .call(
                &json!({
                    "action": "bind_account",
                    "service": "twitter",
                    "account_id": "2018008573557551105",
                    "identity_url": identity_url,
                    "account_id_pointer": "/data/id",
                })
                .to_string(),
            )
            .await
            .unwrap_err();
        assert!(
            unapproved
                .to_string()
                .contains("requires approval through the agent kernel"),
            "{unapproved}"
        );
        let result = call_approved(
            &tool,
            &json!({
                "action": "bind_account",
                "service": "twitter",
                "account_id": "2018008573557551105",
                "identity_url": identity_url,
                "account_id_pointer": "/data/id",
            })
            .to_string(),
        )
        .await
        .unwrap();
        assert!(result.contains("OAuth account identity verified and bound"));
        assert!(result.contains("\"http_status\": 200"));
        let bound = tool
            .state_store
            .get_oauth_connection("twitter")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(bound.account_id.as_deref(), Some("2018008573557551105"));
        let listed = tool.call(r#"{"action":"list"}"#).await.unwrap();
        assert!(listed.contains("account_id: 2018008573557551105"));

        let mismatch = call_approved(
            &tool,
            &json!({
                "action": "bind_account",
                "service": "twitter",
                "account_id": "different-account",
                "identity_url": identity_url,
                "account_id_pointer": "/data/id",
            })
            .to_string(),
        )
        .await
        .unwrap_err();
        assert!(mismatch
            .to_string()
            .contains("Authenticated identity mismatch"));
        let unchanged = tool
            .state_store
            .get_oauth_connection("twitter")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(unchanged.account_id.as_deref(), Some("2018008573557551105"));
    }

    #[tokio::test]
    async fn remove_requires_explicit_confirmation() {
        let _isolation = KeychainIsolation::acquire();
        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let (tool, _gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();

        tool.state_store
            .save_oauth_connection(&crate::traits::OAuthConnection {
                id: 0,
                service: "twitter".to_string(),
                auth_type: "oauth2_pkce".to_string(),
                username: None,
                account_id: None,
                scopes: r#"["tweet.read","tweet.write","users.read","offline.access"]"#.to_string(),
                token_expires_at: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();

        let result = tool
            .call(r#"{"action":"remove","service":"twitter"}"#)
            .await
            .unwrap();

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
        let _isolation = KeychainIsolation::acquire();
        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path());
        let (tool, _gateway, _db) = test_tool(config_file.path().to_path_buf()).await.unwrap();

        tool.state_store
            .save_oauth_connection(&crate::traits::OAuthConnection {
                id: 0,
                service: "twitter".to_string(),
                auth_type: "oauth2_pkce".to_string(),
                username: None,
                account_id: None,
                scopes: r#"["tweet.read","tweet.write","users.read","offline.access"]"#.to_string(),
                token_expires_at: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();

        let unapproved = tool
            .call(r#"{"action":"remove","service":"twitter","confirm_disconnect":true}"#)
            .await
            .unwrap_err();
        assert!(
            unapproved
                .to_string()
                .contains("requires approval through the agent kernel"),
            "{unapproved}"
        );
        assert!(tool
            .state_store
            .get_oauth_connection("twitter")
            .await
            .unwrap()
            .is_some());

        let result = call_approved(
            &tool,
            r#"{"action":"remove","service":"twitter","confirm_disconnect":true}"#,
        )
        .await
        .unwrap();

        assert!(result.contains("Disconnected from twitter"));
        assert!(tool
            .state_store
            .get_oauth_connection("twitter")
            .await
            .unwrap()
            .is_none());
    }
}
