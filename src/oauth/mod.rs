pub mod providers;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use sha2::Digest;
use tokio::sync::{oneshot, RwLock};
use tracing::{info, warn};

use crate::config::{HttpAuthProfile, HttpAuthType, OAuthProviderConfig};
use crate::traits::StateStore;

/// Timeout for OAuth flow completion (10 minutes).
const FLOW_TIMEOUT_SECS: u64 = 600;

/// OAuth type enum.
#[derive(Debug, Clone, PartialEq)]
pub enum OAuthType {
    OAuth2Pkce,
    OAuth2AuthorizationCode,
    OAuth2ClientCredentials,
    OAuth1a,
}

/// Defines an OAuth provider's endpoints and configuration.
#[derive(Debug, Clone)]
pub struct OAuthProvider {
    pub name: String,
    pub display_name: String,
    pub auth_type: OAuthType,
    pub authorize_url: String,
    pub token_url: String,
    pub scopes: Vec<String>,
    pub allowed_domains: Vec<String>,
}

/// Result of a completed OAuth flow.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OAuthFlowResult {
    pub service: String,
    pub username: Option<String>,
    pub message: String,
}

/// A pending OAuth flow awaiting callback.
struct PendingFlow {
    provider_name: String,
    code_verifier: Option<String>,
    #[allow(dead_code)]
    session_id: String,
    #[allow(dead_code)]
    created_at: chrono::DateTime<chrono::Utc>,
    result_tx: oneshot::Sender<OAuthFlowResult>,
}

/// Shared HTTP auth profiles map type.
pub type SharedHttpProfiles = Arc<RwLock<HashMap<String, HttpAuthProfile>>>;

/// The OAuth gateway manages OAuth flows and stores tokens.
#[derive(Clone)]
pub struct OAuthGateway {
    providers: Arc<RwLock<HashMap<String, OAuthProvider>>>,
    pending_flows: Arc<RwLock<HashMap<String, PendingFlow>>>,
    state_store: Arc<dyn StateStore>,
    http_profiles: SharedHttpProfiles,
    callback_base_url: String,
}

impl OAuthGateway {
    pub fn new(
        state_store: Arc<dyn StateStore>,
        http_profiles: SharedHttpProfiles,
        callback_base_url: String,
    ) -> Self {
        Self {
            providers: Arc::new(RwLock::new(HashMap::new())),
            pending_flows: Arc::new(RwLock::new(HashMap::new())),
            state_store,
            http_profiles,
            callback_base_url,
        }
    }

    /// Register a built-in OAuth provider.
    pub async fn register_provider(&self, provider: OAuthProvider) {
        let mut providers = self.providers.write().await;
        providers.insert(provider.name.clone(), provider);
    }

    /// Register a custom provider from config.
    pub async fn register_config_provider(&self, name: &str, config: &OAuthProviderConfig) {
        let auth_type = match config.auth_type.as_str() {
            "oauth2_authorization_code" | "authorization_code" | "auth_code" => {
                OAuthType::OAuth2AuthorizationCode
            }
            "oauth2_client_credentials" | "client_credentials" => {
                OAuthType::OAuth2ClientCredentials
            }
            "oauth1a" => OAuthType::OAuth1a,
            _ => OAuthType::OAuth2Pkce,
        };
        let provider = OAuthProvider {
            name: name.to_string(),
            display_name: config
                .display_name
                .clone()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| name.to_string()),
            auth_type,
            authorize_url: config.authorize_url.clone(),
            token_url: config.token_url.clone(),
            scopes: config.scopes.clone(),
            allowed_domains: config.allowed_domains.clone(),
        };
        self.register_provider(provider).await;
    }

    /// Remove a registered OAuth provider definition from the live gateway.
    pub async fn unregister_provider(&self, name: &str) -> bool {
        let mut providers = self.providers.write().await;
        providers.remove(name).is_some()
    }

    /// List all registered providers.
    pub async fn list_providers(&self) -> Vec<(String, String)> {
        let providers = self.providers.read().await;
        providers
            .values()
            .map(|p| (p.name.clone(), p.display_name.clone()))
            .collect()
    }

    /// Get a provider by name.
    pub async fn get_provider(&self, name: &str) -> Option<OAuthProvider> {
        let providers = self.providers.read().await;
        providers.get(name).cloned()
    }

    pub fn callback_url(&self) -> String {
        Self::normalize_callback_url(&self.callback_base_url)
    }

    fn normalize_callback_url(callback_base_url: &str) -> String {
        let trimmed = callback_base_url.trim().trim_end_matches('/');
        if trimmed.ends_with("/oauth/callback") {
            trimmed.to_string()
        } else {
            format!("{trimmed}/oauth/callback")
        }
    }

    /// Generate a PKCE code verifier (43-128 chars, URL-safe).
    fn generate_code_verifier() -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&bytes)
    }

    /// Generate PKCE code challenge from verifier: BASE64URL(SHA256(verifier)).
    fn generate_code_challenge(verifier: &str) -> String {
        let hash = sha2::Sha256::digest(verifier.as_bytes());
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hash)
    }

    /// Generate a random state parameter for CSRF protection.
    fn generate_state() -> String {
        uuid::Uuid::new_v4().to_string()
    }

    /// Check if client credentials exist in keychain for a service.
    pub fn has_credentials(service: &str) -> bool {
        let client_id_key = format!("oauth_{}_client_id", service);
        crate::config::resolve_from_keychain(&client_id_key).is_ok()
    }

    /// Get client credentials from keychain.
    fn get_credentials(service: &str) -> anyhow::Result<(String, String)> {
        let client_id_key = format!("oauth_{}_client_id", service);
        let client_secret_key = format!("oauth_{}_client_secret", service);
        let client_id = crate::config::resolve_from_keychain(&client_id_key).map_err(|_| {
            anyhow::anyhow!(
                "Client ID not found in keychain. Set it with: aidaemon keychain set {}",
                client_id_key
            )
        })?;
        let client_secret =
            crate::config::resolve_from_keychain(&client_secret_key).map_err(|_| {
                anyhow::anyhow!(
                    "Client secret not found in keychain. Set it with: aidaemon keychain set {}",
                    client_secret_key
                )
            })?;
        Ok((client_id, client_secret))
    }

    /// Start an interactive OAuth authorization flow.
    /// Returns the authorize URL and a receiver for the flow result.
    pub async fn start_oauth2_flow(
        &self,
        service: &str,
        session_id: &str,
    ) -> anyhow::Result<(String, oneshot::Receiver<OAuthFlowResult>)> {
        let provider = self
            .get_provider(service)
            .await
            .ok_or_else(|| anyhow::anyhow!("Unknown OAuth provider: {}", service))?;

        if !matches!(
            provider.auth_type,
            OAuthType::OAuth2Pkce | OAuthType::OAuth2AuthorizationCode
        ) {
            return Err(anyhow::anyhow!(
                "Provider '{}' does not support interactive OAuth authorization flows",
                service
            ));
        }

        let (client_id, _) = Self::get_credentials(service)?;

        let state = Self::generate_state();
        let code_verifier = if provider.auth_type == OAuthType::OAuth2Pkce {
            Some(Self::generate_code_verifier())
        } else {
            None
        };

        let callback_url = self.callback_url();
        let scopes = provider.scopes.join(" ");

        let mut authorize_url = format!(
            "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}",
            provider.authorize_url,
            urlencoded(&client_id),
            urlencoded(&callback_url),
            urlencoded(&scopes),
            urlencoded(&state),
        );
        if let Some(ref verifier) = code_verifier {
            let code_challenge = Self::generate_code_challenge(verifier);
            authorize_url.push_str(&format!(
                "&code_challenge={}&code_challenge_method=S256",
                urlencoded(&code_challenge)
            ));
        }

        let (result_tx, result_rx) = oneshot::channel();

        let flow = PendingFlow {
            provider_name: service.to_string(),
            code_verifier,
            session_id: session_id.to_string(),
            created_at: chrono::Utc::now(),
            result_tx,
        };

        {
            let mut flows = self.pending_flows.write().await;
            flows.insert(state, flow);
        }

        Ok((authorize_url, result_rx))
    }

    fn oauth_type_label(auth_type: &OAuthType) -> &'static str {
        match auth_type {
            OAuthType::OAuth2Pkce => "oauth2_pkce",
            OAuthType::OAuth2AuthorizationCode => "oauth2_authorization_code",
            OAuthType::OAuth2ClientCredentials => "oauth2_client_credentials",
            OAuthType::OAuth1a => "oauth1a",
        }
    }

    async fn store_connected_bearer_profile(
        &self,
        service: &str,
        provider: &OAuthProvider,
        access_token: &str,
        refresh_token: Option<&str>,
        expires_in: Option<u64>,
    ) -> anyhow::Result<String> {
        let at_key = format!("oauth_{}_access_token", service);
        crate::config::store_in_keychain(&at_key, access_token)?;

        if let Some(rt) = refresh_token {
            let rt_key = format!("oauth_{}_refresh_token", service);
            crate::config::store_in_keychain(&rt_key, rt)?;
        }

        let expires_at = expires_in
            .map(|secs| (chrono::Utc::now() + chrono::Duration::seconds(secs as i64)).to_rfc3339());

        let conn = crate::traits::OAuthConnection {
            id: 0,
            service: service.to_string(),
            auth_type: Self::oauth_type_label(&provider.auth_type).to_string(),
            username: None,
            scopes: serde_json::to_string(&provider.scopes).unwrap_or_default(),
            token_expires_at: expires_at,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };
        self.state_store.save_oauth_connection(&conn).await?;

        let profile = HttpAuthProfile {
            auth_type: HttpAuthType::Bearer,
            allowed_domains: provider.allowed_domains.clone(),
            api_key: None,
            api_secret: None,
            access_token: None,
            access_token_secret: None,
            user_id: None,
            token: Some(access_token.to_string()),
            header_name: None,
            header_value: None,
            username: None,
            password: None,
        };
        {
            let mut profiles = self.http_profiles.write().await;
            profiles.insert(service.to_string(), profile);
        }

        Ok(format!(
            "Connected to {}! Use `http_request` with auth_profile=\"{}\" to make API calls.",
            provider.display_name, service
        ))
    }

    async fn exchange_client_credentials_token(
        &self,
        service: &str,
        provider: &OAuthProvider,
    ) -> anyhow::Result<(String, Option<String>, Option<u64>)> {
        let (client_id, client_secret) = Self::get_credentials(service)?;

        let mut params = HashMap::new();
        params.insert("grant_type", "client_credentials".to_string());
        if !provider.scopes.is_empty() {
            params.insert("scope", provider.scopes.join(" "));
        }

        let client = reqwest::Client::new();
        let resp = client
            .post(&provider.token_url)
            .basic_auth(&client_id, Some(&client_secret))
            .form(&params)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Client credentials token request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Client credentials token exchange failed (HTTP {}): {}",
                status,
                body
            );
        }

        let token_data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse token response: {}", e))?;

        let access_token = token_data["access_token"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No access_token in response"))?
            .to_string();
        let refresh_token = token_data["refresh_token"]
            .as_str()
            .map(ToString::to_string);
        let expires_in = token_data["expires_in"].as_u64();
        Ok((access_token, refresh_token, expires_in))
    }

    pub async fn connect_client_credentials(&self, service: &str) -> anyhow::Result<String> {
        let provider = self
            .get_provider(service)
            .await
            .ok_or_else(|| anyhow::anyhow!("Unknown OAuth provider: {}", service))?;

        anyhow::ensure!(
            provider.auth_type == OAuthType::OAuth2ClientCredentials,
            "Provider '{}' is not configured for oauth2_client_credentials",
            service
        );

        let (access_token, refresh_token, expires_in) = self
            .exchange_client_credentials_token(service, &provider)
            .await?;
        self.store_connected_bearer_profile(
            service,
            &provider,
            &access_token,
            refresh_token.as_deref(),
            expires_in,
        )
        .await
    }

    /// Handle an OAuth callback from the browser redirect.
    pub async fn handle_callback(
        &self,
        state: &str,
        code: Option<&str>,
        error: Option<&str>,
    ) -> anyhow::Result<String> {
        // Extract the pending flow
        let flow = {
            let mut flows = self.pending_flows.write().await;
            flows
                .remove(state)
                .ok_or_else(|| anyhow::anyhow!("Unknown or expired OAuth state parameter"))?
        };

        // Check for error from provider
        if let Some(err) = error {
            let msg = format!("OAuth authorization denied: {}", err);
            let _ = flow.result_tx.send(OAuthFlowResult {
                service: flow.provider_name.clone(),
                username: None,
                message: msg.clone(),
            });
            return Ok(msg);
        }

        let code = code.ok_or_else(|| anyhow::anyhow!("No authorization code in callback"))?;

        let provider = self
            .get_provider(&flow.provider_name)
            .await
            .ok_or_else(|| anyhow::anyhow!("Provider '{}' not found", flow.provider_name))?;

        // Exchange code for tokens
        let (client_id, client_secret) = Self::get_credentials(&flow.provider_name)?;
        let callback_url = self.callback_url();

        let mut params = HashMap::new();
        params.insert("grant_type", "authorization_code".to_string());
        params.insert("code", code.to_string());
        params.insert("redirect_uri", callback_url);

        if let Some(ref verifier) = flow.code_verifier {
            params.insert("code_verifier", verifier.clone());
        }

        let client = reqwest::Client::new();
        let resp = client
            .post(&provider.token_url)
            .basic_auth(&client_id, Some(&client_secret))
            .form(&params)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Token exchange request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let msg = format!("Token exchange failed (HTTP {}): {}", status, body);
            let _ = flow.result_tx.send(OAuthFlowResult {
                service: flow.provider_name.clone(),
                username: None,
                message: msg.clone(),
            });
            return Ok(msg);
        }

        let token_data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse token response: {}", e))?;

        let access_token = token_data["access_token"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No access_token in response"))?;
        let refresh_token = token_data["refresh_token"].as_str();
        let expires_in = token_data["expires_in"].as_u64();
        let message = self
            .store_connected_bearer_profile(
                &flow.provider_name,
                &provider,
                access_token,
                refresh_token,
                expires_in,
            )
            .await?;
        info!(
            service = %flow.provider_name,
            "OAuth connection established"
        );

        let _ = flow.result_tx.send(OAuthFlowResult {
            service: flow.provider_name.clone(),
            username: None,
            message: message.clone(),
        });

        Ok(message)
    }

    /// Remove expired pending flows (older than 10 minutes).
    pub async fn cleanup_expired_flows(&self) {
        let cutoff = chrono::Utc::now() - chrono::Duration::seconds(FLOW_TIMEOUT_SECS as i64);
        let mut flows = self.pending_flows.write().await;
        let expired: Vec<String> = flows
            .iter()
            .filter(|(_, f)| f.created_at < cutoff)
            .map(|(k, _)| k.clone())
            .collect();
        for key in &expired {
            if let Some(flow) = flows.remove(key) {
                let _ = flow.result_tx.send(OAuthFlowResult {
                    service: flow.provider_name,
                    username: None,
                    message: "OAuth flow expired (timeout)".to_string(),
                });
            }
        }
        if !expired.is_empty() {
            info!(count = expired.len(), "Cleaned up expired OAuth flows");
        }
    }

    /// Restore connections from DB + keychain on startup.
    pub async fn restore_connections(&self) {
        match self.state_store.list_oauth_connections().await {
            Ok(connections) => {
                for conn in connections {
                    let at_key = format!("oauth_{}_access_token", conn.service);
                    match crate::config::resolve_from_keychain(&at_key) {
                        Ok(token) => {
                            // Look up provider for allowed_domains
                            let allowed_domains = self
                                .get_provider(&conn.service)
                                .await
                                .map(|p| p.allowed_domains.clone())
                                .unwrap_or_default();

                            let profile = HttpAuthProfile {
                                auth_type: HttpAuthType::Bearer,
                                allowed_domains,
                                api_key: None,
                                api_secret: None,
                                access_token: None,
                                access_token_secret: None,
                                user_id: None,
                                token: Some(token),
                                header_name: None,
                                header_value: None,
                                username: None,
                                password: None,
                            };
                            {
                                let mut profiles = self.http_profiles.write().await;
                                profiles.insert(conn.service.clone(), profile);
                            }
                            info!(service = %conn.service, "Restored OAuth connection");
                        }
                        Err(_) => {
                            warn!(
                                service = %conn.service,
                                "OAuth connection in DB but token not found in keychain"
                            );
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Failed to restore OAuth connections: {}", e);
            }
        }
    }

    /// Refresh an expired access token using the refresh token.
    pub async fn refresh_token(&self, service: &str) -> anyhow::Result<String> {
        let provider = self
            .get_provider(service)
            .await
            .ok_or_else(|| anyhow::anyhow!("Unknown OAuth provider: {}", service))?;

        if provider.auth_type == OAuthType::OAuth2ClientCredentials {
            let (access_token, refresh_token, expires_in) = self
                .exchange_client_credentials_token(service, &provider)
                .await?;
            self.store_connected_bearer_profile(
                service,
                &provider,
                &access_token,
                refresh_token.as_deref(),
                expires_in,
            )
            .await?;
            info!(service = %service, "OAuth client-credentials token refreshed");
            return Ok(format!("Token refreshed for {}", service));
        }

        let rt_key = format!("oauth_{}_refresh_token", service);
        let refresh_token = crate::config::resolve_from_keychain(&rt_key)
            .map_err(|_| anyhow::anyhow!("No refresh token found for '{}'", service))?;

        let (client_id, client_secret) = Self::get_credentials(service)?;

        let mut params = HashMap::new();
        params.insert("grant_type", "refresh_token".to_string());
        params.insert("refresh_token", refresh_token);

        let client = reqwest::Client::new();
        let resp = client
            .post(&provider.token_url)
            .basic_auth(&client_id, Some(&client_secret))
            .form(&params)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Refresh request failed: {}", e))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Token refresh failed: {}", body));
        }

        let token_data: serde_json::Value = resp.json().await?;
        let new_access_token = token_data["access_token"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No access_token in refresh response"))?;

        // Store new access token
        let at_key = format!("oauth_{}_access_token", service);
        crate::config::store_in_keychain(&at_key, new_access_token)?;

        // Update refresh token if a new one was issued
        if let Some(new_rt) = token_data["refresh_token"].as_str() {
            let rt_key = format!("oauth_{}_refresh_token", service);
            crate::config::store_in_keychain(&rt_key, new_rt)?;
        }

        // Update expiry
        let expires_in = token_data["expires_in"].as_u64();
        let expires_at = expires_in
            .map(|secs| (chrono::Utc::now() + chrono::Duration::seconds(secs as i64)).to_rfc3339());
        self.state_store
            .update_oauth_token_expiry(service, expires_at.as_deref())
            .await?;

        // Update in-memory profile
        {
            let mut profiles = self.http_profiles.write().await;
            if let Some(profile) = profiles.get_mut(service) {
                profile.token = Some(new_access_token.to_string());
            }
        }

        info!(service = %service, "OAuth token refreshed");
        Ok(format!("Token refreshed for {}", service))
    }

    /// Remove an OAuth connection: delete from DB, keychain, and profiles.
    pub async fn remove_connection(&self, service: &str) -> anyhow::Result<String> {
        // Remove from DB
        self.state_store.delete_oauth_connection(service).await?;

        // Remove tokens from keychain (best effort)
        for suffix in &["access_token", "refresh_token"] {
            let key = format!("oauth_{}_{}", service, suffix);
            let _ = crate::config::delete_from_keychain(&key);
        }

        // Remove from in-memory profiles
        {
            let mut profiles = self.http_profiles.write().await;
            profiles.remove(service);
        }

        info!(service = %service, "OAuth connection removed");
        Ok(format!("Disconnected from {}", service))
    }

    /// Get the flow timeout in seconds (for tool to know how long to wait).
    pub fn flow_timeout() -> Duration {
        Duration::from_secs(FLOW_TIMEOUT_SECS)
    }
}

/// Simple URL-encoding for query parameter values.
fn urlencoded(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 2);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(byte as char);
            }
            _ => {
                result.push_str(&format!("%{:02X}", byte));
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::sync::Arc;

    use axum::{extract::Form, routing::post, Json, Router};
    use once_cell::sync::Lazy;
    use tempfile::NamedTempFile;
    use tokio::net::TcpListener;

    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::StateStore;

    static ENV_LOCK: Lazy<std::sync::Mutex<()>> = Lazy::new(|| std::sync::Mutex::new(()));

    fn restore_env_var(name: &str, old_value: Option<String>) {
        if let Some(value) = old_value {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
    }

    async fn test_gateway() -> anyhow::Result<OAuthGateway> {
        let db_file = NamedTempFile::new()?;
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new()?);
        let state = Arc::new(SqliteStateStore::new(&db_path, 32, None, embedding_service).await?);
        let profiles: SharedHttpProfiles = Arc::new(RwLock::new(HashMap::new()));
        Ok(OAuthGateway::new(
            state as Arc<dyn StateStore>,
            profiles,
            "http://localhost:8080".to_string(),
        ))
    }

    #[test]
    fn test_pkce_code_verifier_length() {
        let verifier = OAuthGateway::generate_code_verifier();
        assert!(verifier.len() >= 43);
        assert!(verifier.len() <= 128);
    }

    #[test]
    fn test_pkce_code_challenge_is_deterministic() {
        let verifier = "test_verifier_string";
        let c1 = OAuthGateway::generate_code_challenge(verifier);
        let c2 = OAuthGateway::generate_code_challenge(verifier);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_pkce_code_challenge_uses_sha256() {
        // Known test vector: SHA256("test") = base64url encoded
        let challenge = OAuthGateway::generate_code_challenge("test");
        // SHA256("test") = 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
        // base64url: n4bQgYhMfWWaL-qgxVrQFaO_TxsrC4Is0V1sFbDwCgg
        assert_eq!(challenge, "n4bQgYhMfWWaL-qgxVrQFaO_TxsrC4Is0V1sFbDwCgg");
    }

    #[test]
    fn test_state_parameter_uniqueness() {
        let s1 = OAuthGateway::generate_state();
        let s2 = OAuthGateway::generate_state();
        assert_ne!(s1, s2);
        // Should be a valid UUID
        assert!(uuid::Uuid::parse_str(&s1).is_ok());
    }

    #[test]
    fn test_urlencoded() {
        assert_eq!(urlencoded("hello"), "hello");
        assert_eq!(urlencoded("hello world"), "hello%20world");
        assert_eq!(urlencoded("a=b&c=d"), "a%3Db%26c%3Dd");
        assert_eq!(
            urlencoded("https://example.com"),
            "https%3A%2F%2Fexample.com"
        );
    }

    #[test]
    fn test_callback_url_accepts_base_url() {
        assert_eq!(
            OAuthGateway::normalize_callback_url("http://localhost:8080"),
            "http://localhost:8080/oauth/callback"
        );
    }

    #[test]
    fn test_callback_url_accepts_full_callback_url() {
        assert_eq!(
            OAuthGateway::normalize_callback_url("http://localhost:8080/oauth/callback"),
            "http://localhost:8080/oauth/callback"
        );
    }

    #[tokio::test]
    async fn authorization_code_flow_omits_pkce_challenge() {
        let _guard = ENV_LOCK.lock().unwrap();
        let gateway = test_gateway().await.unwrap();
        gateway
            .register_provider(OAuthProvider {
                name: "linear".to_string(),
                display_name: "Linear".to_string(),
                auth_type: OAuthType::OAuth2AuthorizationCode,
                authorize_url: "https://linear.app/oauth/authorize".to_string(),
                token_url: "https://api.linear.app/oauth/token".to_string(),
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

        let (authorize_url, _) = gateway.start_oauth2_flow("linear", "test").await.unwrap();

        restore_env_var("AIDAEMON_NO_KEYCHAIN", old_no_keychain);
        restore_env_var(crate::RUNTIME_ENV_FILE_ENV_KEY, old_runtime_env);

        assert!(authorize_url.contains("response_type=code"));
        assert!(!authorize_url.contains("code_challenge="));
    }

    #[tokio::test]
    async fn client_credentials_flow_stores_connection_and_profile() {
        let _guard = ENV_LOCK.lock().unwrap();

        async fn token_handler(
            Form(form): Form<HashMap<String, String>>,
        ) -> Json<serde_json::Value> {
            assert_eq!(
                form.get("grant_type").map(String::as_str),
                Some("client_credentials")
            );
            Json(serde_json::json!({
                "access_token": "test-access",
                "expires_in": 3600
            }))
        }

        let app = Router::new().route("/oauth/token", post(token_handler));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let gateway = test_gateway().await.unwrap();
        gateway
            .register_provider(OAuthProvider {
                name: "service".to_string(),
                display_name: "Service".to_string(),
                auth_type: OAuthType::OAuth2ClientCredentials,
                authorize_url: String::new(),
                token_url: format!("http://{}/oauth/token", addr),
                scopes: vec!["read".to_string()],
                allowed_domains: vec!["api.example.com".to_string()],
            })
            .await;

        let env_file = NamedTempFile::new().unwrap();
        std::fs::write(
            env_file.path(),
            "OAUTH_SERVICE_CLIENT_ID=abc\nOAUTH_SERVICE_CLIENT_SECRET=def\n",
        )
        .unwrap();
        let old_no_keychain = std::env::var("AIDAEMON_NO_KEYCHAIN").ok();
        let old_runtime_env = std::env::var(crate::RUNTIME_ENV_FILE_ENV_KEY).ok();
        std::env::set_var("AIDAEMON_NO_KEYCHAIN", "1");
        std::env::set_var(
            crate::RUNTIME_ENV_FILE_ENV_KEY,
            env_file.path().to_string_lossy().to_string(),
        );

        let result = gateway.connect_client_credentials("service").await.unwrap();

        restore_env_var("AIDAEMON_NO_KEYCHAIN", old_no_keychain);
        restore_env_var(crate::RUNTIME_ENV_FILE_ENV_KEY, old_runtime_env);

        assert!(result.contains("Connected to Service"));
        let conn = gateway
            .state_store
            .get_oauth_connection("service")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(conn.auth_type, "oauth2_client_credentials");
        let profiles = gateway.http_profiles.read().await;
        let profile = profiles.get("service").unwrap();
        assert_eq!(profile.token.as_deref(), Some("test-access"));
    }
}
