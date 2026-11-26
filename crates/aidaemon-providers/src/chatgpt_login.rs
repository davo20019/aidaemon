//! ChatGPT subscription login (OpenAI Sign in with ChatGPT for open-source
//! clients), behind the `chatgpt-login` feature.
//!
//! Lets an agent call OpenAI's public Responses API (`https://api.openai.com/v1`)
//! using a user's ChatGPT Go/Plus/Pro/Business plan instead of a metered OpenAI
//! API key, following OpenAI's documented open-source dynamic client registration
//! flow (`client_id=dynamic_agent_client`, persistent `ext_agent_host_id`,
//! `agent_name_hint`, and `resource=https://api.openai.com/v1`).
//!
//! Flow: OAuth 2.0 authorization code + PKCE + OpenID Connect (`nonce`) against
//! `https://auth.openai.com/api/accounts/authorize`, with the callback delivered
//! to an HTTP loopback listener on `127.0.0.1` (`http://127.0.0.1:<port>/auth/callback`).
//!
//! Credentials live in an [`aidaemon_secrets::SecretStore`] under
//! `oauth_openai_chatgpt_*`. [`ChatGptCredentialManager`] owns them for the
//! process and serializes refresh-token rotation.

use std::sync::Arc;
use std::time::Duration;

use aidaemon_secrets::{SecretStore, names};
use anyhow::{Context, Result, anyhow};
use base64::Engine;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing::{debug, warn};

/// OIDC issuer for OpenAI authentication.
pub const ISSUER: &str = "https://auth.openai.com";
/// Authorization endpoint (user-facing Sign in with ChatGPT).
pub const AUTHORIZE_URL: &str = "https://auth.openai.com/api/accounts/authorize";
/// Token endpoint (code exchange and refresh).
pub const TOKEN_URL: &str = "https://auth.openai.com/api/accounts/oauth/token";
/// Token revocation endpoint, used on logout.
pub const REVOKE_URL: &str = "https://auth.openai.com/api/accounts/oauth/revoke";

/// Public OpenAI API resource audience and base URL for ChatGPT plan usage.
pub const API_RESOURCE: &str = "https://api.openai.com/v1";
/// Public OpenAI API root for ChatGPT plan usage.
pub const OPENAI_API_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI's first-time dynamic client registration entrypoint for open-source
/// clients. Never persisted or reused as an issued client ID.
pub const DYNAMIC_CLIENT_ID: &str = "dynamic_agent_client";
/// Loopback callback path required by the dynamic registration flow.
pub const CALLBACK_PATH: &str = "/auth/callback";
/// Default port used for paste-mode sign-in (when no local listener is bound).
pub const CALLBACK_PORT: u16 = 1455;
/// Default loopback callback URI for paste-mode sign-in (strictly `127.0.0.1`,
/// never `localhost`).
pub const PASTE_REDIRECT_URI: &str = "http://127.0.0.1:1455/auth/callback";
/// Scopes requested for Sign in with ChatGPT identity + plan usage.
pub const SCOPES: &str =
    "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
/// Scope that authorizes ChatGPT plan usage on OpenAI's public Responses API.
pub const PLAN_USAGE_SCOPE: &str = "chatgpt.tokens.use.direct";

/// Legacy first-party Codex CLI client ID; tokens issued under it must not be
/// reused by third-party clients.
const LEGACY_CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// Service name used to namespace keychain entries.
pub const SERVICE: &str = "openai_chatgpt";

/// Namespaced claim on the id_token that may carry ChatGPT account details.
const AUTH_CLAIM_NAMESPACE: &str = "https://api.openai.com/auth";

/// How long to wait for the user to complete the browser sign-in.
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(900);
/// How long one connection to the callback listener may take to send its
/// request line.
const CALLBACK_READ_TIMEOUT: Duration = Duration::from_secs(10);

/// Refresh this far ahead of actual expiry so an in-flight request cannot
/// straddle the boundary.
const REFRESH_SKEW: chrono::Duration = chrono::Duration::minutes(5);

/// Cooldown after a rejected refresh so bad credentials fail fast.
const AUTH_REJECTION_COOLDOWN: chrono::Duration = chrono::Duration::minutes(1);

/// Clock skew tolerance when validating `id_token` expiry.
const ID_TOKEN_EXP_SKEW_SECS: i64 = 60;

/// What a host names in the Sign in with ChatGPT flow, and an optional
/// pre-registered OAuth client ID.
#[derive(Debug, Clone)]
pub struct LoginSettings {
    /// Optional pre-registered OAuth client ID. `None` uses OpenAI's official
    /// open-source dynamic registration (`dynamic_agent_client`).
    pub client_id: Option<String>,
    /// Stable application name sent as `agent_name_hint` on initial
    /// registration and shown on the browser callback page.
    pub app_name: String,
    /// The command that signs in again, named in errors.
    pub login_command: String,
}

impl LoginSettings {
    pub fn new(app_name: impl Into<String>, login_command: impl Into<String>) -> Self {
        Self {
            client_id: None,
            app_name: app_name.into(),
            login_command: login_command.into(),
        }
    }

    fn no_login(&self) -> anyhow::Error {
        anyhow!(
            "No ChatGPT subscription login found (or an older login needs upgrading to \
             Sign in with ChatGPT). Run `{}` to connect your ChatGPT account.",
            self.login_command
        )
    }
}

fn key(suffix: &str) -> String {
    names::oauth(SERVICE, suffix)
}

/// Stored ChatGPT subscription credentials from Sign in with ChatGPT.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatGptCredentials {
    pub access_token: String,
    pub refresh_token: String,
    /// Verified account identifier from the ID token (`chatgpt_account_id` or `sub`).
    pub account_id: String,
    /// Issued OAuth client ID bound to this registration (never `dynamic_agent_client`).
    pub client_id: String,
    pub expires_at: DateTime<Utc>,
}

impl ChatGptCredentials {
    /// True when the access token is expired or close enough that a request
    /// started now might outlive it.
    pub fn needs_refresh(&self, now: DateTime<Utc>) -> bool {
        self.expires_at <= now + REFRESH_SKEW
    }
}

/// Process-wide owner of ChatGPT subscription credentials.
///
/// OpenAI rotates refresh tokens. Keeping the refresh lock here prevents
/// concurrent callers from refreshing the same token and invalidating one
/// another.
pub struct ChatGptCredentialManager {
    state: Mutex<CredentialState>,
    token_url: String,
    revoke_url: String,
    persist_refreshes: bool,
    secrets: Arc<dyn SecretStore>,
    settings: LoginSettings,
}

#[derive(Default)]
struct CredentialState {
    credentials: Option<ChatGptCredentials>,
    rejection: Option<CredentialRejection>,
}

#[derive(Clone)]
struct CredentialRejection {
    access_token: String,
    observed_at: DateTime<Utc>,
    detail: String,
}

impl ChatGptCredentialManager {
    /// The process's credential owner over `secrets`. Make one per process
    /// and share it: two managers would race refresh-token rotation.
    pub fn new(secrets: Arc<dyn SecretStore>, settings: LoginSettings) -> Self {
        Self {
            state: Mutex::new(CredentialState::default()),
            token_url: TOKEN_URL.to_string(),
            revoke_url: REVOKE_URL.to_string(),
            persist_refreshes: true,
            secrets,
            settings,
        }
    }

    pub fn settings(&self) -> &LoginSettings {
        &self.settings
    }

    /// Persist credentials to the secret store.
    pub fn store_credentials(&self, creds: &ChatGptCredentials) -> Result<()> {
        store_credentials(self.secrets.as_ref(), creds)
    }

    /// Load stored credentials, if a valid Sign in with ChatGPT login exists.
    pub fn load_credentials(&self) -> Option<ChatGptCredentials> {
        load_credentials(self.secrets.as_ref())
    }

    /// Whether a valid Sign in with ChatGPT login is stored.
    pub fn is_connected(&self) -> bool {
        self.load_credentials().is_some()
    }

    /// Persistent opaque `ext_agent_host_id` for this host (`urn:uuid:...`),
    /// generated once and reused across sign-ins.
    pub fn host_id(&self) -> Result<String> {
        ensure_host_id(self.secrets.as_ref())
    }

    /// Remove stored session credentials and best-effort revoke the refresh
    /// token. Keeps the host's `ext_agent_host_id` and issued `client_id`
    /// registration intact so a later sign-in reuses them without repeating
    /// dynamic client registration.
    pub async fn logout(&self, client: &reqwest::Client) -> Result<()> {
        let mut state = self.state.lock().await;
        let current = state.credentials.take().or_else(|| self.load_credentials());
        state.rejection = None;

        if let Some(creds) = current {
            match client
                .post(&self.revoke_url)
                .form(&[
                    ("client_id", creds.client_id.as_str()),
                    ("token", creds.refresh_token.as_str()),
                    ("token_type_hint", "refresh_token"),
                ])
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {}
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    warn!(
                        %status,
                        body = %truncate_for_error(&body),
                        "Remote ChatGPT token revocation returned non-success; clearing local session"
                    );
                }
                Err(error) => {
                    warn!(
                        %error,
                        "Remote ChatGPT token revocation failed; clearing local session"
                    );
                }
            }
        }

        let mut delete_errors = Vec::new();
        for suffix in [
            "access_token",
            "refresh_token",
            "account_id",
            "expires_at",
            "id_token",
            "subject",
        ] {
            if let Err(error) = self.secrets.delete(&key(suffix)) {
                delete_errors.push(format!("{suffix}: {error}"));
            }
        }
        if !delete_errors.is_empty() {
            return Err(anyhow!(
                "failed to remove stored ChatGPT session entries: {}",
                delete_errors.join(", ")
            ));
        }
        Ok(())
    }

    /// Run the full Sign in with ChatGPT flow and persist the resulting
    /// credentials.
    pub async fn login(
        &self,
        client: &reqwest::Client,
        mode: CallbackMode,
        announce: impl FnOnce(&str),
    ) -> Result<ChatGptCredentials> {
        let attempt = self
            .start_login(matches!(mode, CallbackMode::Loopback))
            .await?;
        announce(&attempt.url);
        match mode {
            CallbackMode::Loopback => self.finish_on_loopback(client, attempt).await,
            CallbackMode::Paste(read) => {
                let pasted = read(&attempt.url)?;
                self.finish_with_redirect(client, &attempt, &pasted).await
            }
        }
    }

    /// Start a sign-in in two steps, for a host that shows the address in
    /// an app instead of a terminal. With `loopback`, an ephemeral loopback
    /// port on `127.0.0.1` is bound now; finish with
    /// [`Self::finish_on_loopback`]. Otherwise finish with the address the
    /// browser ended on ([`Self::finish_with_redirect`]).
    pub async fn start_login(&self, loopback: bool) -> Result<LoginAttempt> {
        let host_id = self.host_id()?;
        let pkce = PkceChallenge::generate();
        let (listener, redirect_uri) = if loopback {
            let listener = bind_callback_listener().await?;
            let port = listener
                .local_addr()
                .context("could not read callback listener port")?
                .port();
            (Some(listener), loopback_redirect_uri(port))
        } else {
            (None, PASTE_REDIRECT_URI.to_string())
        };
        let existing = self.load_credentials();
        let configured_client_id = match self
            .settings
            .client_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
        {
            Some(id) if id == DYNAMIC_CLIENT_ID || id == LEGACY_CODEX_CLIENT_ID => {
                return Err(anyhow!(
                    "configured ChatGPT client_id `{id}` is not allowed; unset it to use dynamic registration"
                ));
            }
            Some(id) => Some(id.to_string()),
            None => None,
        };
        let pending_client_id = configured_client_id
            .or_else(|| existing.as_ref().map(|c| c.client_id.clone()))
            .or_else(|| load_registered_client_id(self.secrets.as_ref()));
        let expected_subject = existing
            .as_ref()
            .and_then(|_| load_subject(self.secrets.as_ref()));
        let id_token_hint = if pending_client_id.is_some() && existing.is_some() {
            load_id_token(self.secrets.as_ref())
        } else {
            None
        };
        let url = build_authorize_url(&AuthorizeParams {
            client_id: pending_client_id.as_deref(),
            app_name: &self.settings.app_name,
            host_id: &host_id,
            redirect_uri: &redirect_uri,
            challenge: &pkce.challenge,
            state: &pkce.state,
            nonce: &pkce.nonce,
            id_token_hint: id_token_hint.as_deref(),
        });
        Ok(LoginAttempt {
            url,
            pkce,
            redirect_uri,
            pending_client_id,
            expected_subject,
            listener,
        })
    }

    /// Wait for the browser to come back to the loopback listener, then
    /// exchange the code and store the credentials.
    pub async fn finish_on_loopback(
        &self,
        client: &reqwest::Client,
        mut attempt: LoginAttempt,
    ) -> Result<ChatGptCredentials> {
        let listener = attempt
            .listener
            .take()
            .ok_or_else(|| anyhow!("this sign-in does not listen on this machine"))?;
        let callback =
            await_callback(listener, &attempt.pkce.state, &self.settings.app_name).await?;
        self.finish_with_callback(client, &attempt, &callback).await
    }

    /// Complete a pasted sign-in with the redirect URL the browser ended on.
    pub async fn finish_with_redirect(
        &self,
        client: &reqwest::Client,
        attempt: &LoginAttempt,
        redirect: &str,
    ) -> Result<ChatGptCredentials> {
        let callback = parse_callback(redirect, &attempt.pkce.state)?;
        self.finish_with_callback(client, attempt, &callback).await
    }

    async fn finish_with_callback(
        &self,
        client: &reqwest::Client,
        attempt: &LoginAttempt,
        callback: &CallbackResult,
    ) -> Result<ChatGptCredentials> {
        let issued_client_id = resolve_issued_client_id(
            attempt.pending_client_id.as_deref(),
            callback.client_id.as_deref(),
        )?;
        // Persist the issued client registration before code exchange so a
        // retry after a failed exchange (e.g. `invalid_grant`) reuses it
        // instead of re-registering dynamically.
        self.secrets.set(&key("client_id"), &issued_client_id)?;
        let exchanged = exchange_code_at(
            client,
            &self.token_url,
            &issued_client_id,
            &callback.code,
            &attempt.pkce.verifier,
            &attempt.redirect_uri,
            &attempt.pkce.nonce,
            attempt.expected_subject.as_deref(),
        )
        .await?;
        let mut state = self.state.lock().await;
        self.store_credentials(&exchanged.credentials)?;
        if let Some(id_token) = exchanged.id_token.as_deref() {
            let _ = self.secrets.set(&key("id_token"), id_token);
        }
        if let Some(subject) = exchanged.subject.as_deref() {
            let _ = self.secrets.set(&key("subject"), subject);
        }
        state.credentials = Some(exchanged.credentials.clone());
        state.rejection = None;
        Ok(exchanged.credentials)
    }

    async fn refresh(
        &self,
        client: &reqwest::Client,
        current: &ChatGptCredentials,
    ) -> Result<ChatGptCredentials> {
        let expected_subject = load_subject(self.secrets.as_ref());
        let refreshed = refresh_credentials_at(
            client,
            &self.settings,
            current,
            expected_subject.as_deref(),
            &self.token_url,
        )
        .await?;
        if self.persist_refreshes {
            if let Some(id_token) = refreshed.id_token.as_deref() {
                let _ = self.secrets.set(&key("id_token"), id_token);
            }
            if let Some(subject) = refreshed.subject.as_deref() {
                let _ = self.secrets.set(&key("subject"), subject);
            }
        }
        Ok(refreshed.credentials)
    }

    /// Return a usable access token, refreshing and persisting it under one
    /// process-wide lock when it is close to expiry.
    pub async fn usable_credentials(&self, client: &reqwest::Client) -> Result<ChatGptCredentials> {
        let mut state = self.state.lock().await;

        if state.credentials.is_none() {
            state.credentials = self.load_credentials();
        }

        let current = state
            .credentials
            .clone()
            .ok_or_else(|| self.settings.no_login())?;

        if let Some(rejection) = state.rejection.as_ref().filter(|rejection| {
            rejection.access_token == current.access_token
                && rejection.observed_at + AUTH_REJECTION_COOLDOWN > Utc::now()
        }) {
            return Err(anyhow!(rejection.detail.clone()));
        }

        if !current.needs_refresh(Utc::now()) {
            return Ok(current);
        }

        debug!("Refreshing ChatGPT subscription access token");
        match self.refresh(client, &current).await {
            Ok(refreshed) => {
                // Persist immediately: OpenAI rotates refresh tokens and the
                // old token may already be dead.
                if self.persist_refreshes
                    && let Err(error) = self.store_credentials(&refreshed)
                {
                    warn!(%error, "Refreshed ChatGPT tokens could not be persisted");
                }
                state.credentials = Some(refreshed.clone());
                state.rejection = None;
                Ok(refreshed)
            }
            Err(error) => {
                state.rejection = Some(CredentialRejection {
                    access_token: current.access_token,
                    observed_at: Utc::now(),
                    detail: error.to_string(),
                });
                Err(error)
            }
        }
    }

    /// Recover after the resource server rejects an access token even though
    /// its locally stored expiry is still in the future.
    pub async fn refresh_after_rejection(
        &self,
        client: &reqwest::Client,
        rejected_access_token: &str,
    ) -> Result<ChatGptCredentials> {
        self.refresh_after_rejection_with(
            rejected_access_token,
            self.persist_refreshes,
            |current| async move { self.refresh(client, &current).await },
        )
        .await
    }

    async fn refresh_after_rejection_with<F, Fut>(
        &self,
        rejected_access_token: &str,
        persist: bool,
        refresh: F,
    ) -> Result<ChatGptCredentials>
    where
        F: FnOnce(ChatGptCredentials) -> Fut,
        Fut: std::future::Future<Output = Result<ChatGptCredentials>>,
    {
        let mut state = self.state.lock().await;
        if state.credentials.is_none() {
            state.credentials = self.load_credentials();
        }
        let current = state
            .credentials
            .clone()
            .ok_or_else(|| self.settings.no_login())?;

        if current.access_token != rejected_access_token {
            return Ok(current);
        }
        if let Some(rejection) = state.rejection.as_ref().filter(|rejection| {
            rejection.access_token == current.access_token
                && rejection.observed_at + AUTH_REJECTION_COOLDOWN > Utc::now()
        }) {
            return Err(anyhow!(rejection.detail.clone()));
        }

        debug!("Refreshing ChatGPT credentials after remote authentication rejection");
        match refresh(current.clone()).await {
            Ok(refreshed) => {
                if persist && let Err(error) = self.store_credentials(&refreshed) {
                    warn!(%error, "Remotely recovered ChatGPT tokens could not be persisted");
                }
                state.credentials = Some(refreshed.clone());
                state.rejection = None;
                Ok(refreshed)
            }
            Err(error) => {
                state.rejection = Some(CredentialRejection {
                    access_token: current.access_token,
                    observed_at: Utc::now(),
                    detail: error.to_string(),
                });
                Err(error)
            }
        }
    }

    /// Record that the one post-refresh replay was also rejected.
    pub async fn mark_rejected(&self, rejected_access_token: &str, detail: impl Into<String>) {
        let mut state = self.state.lock().await;
        if state
            .credentials
            .as_ref()
            .is_some_and(|credentials| credentials.access_token == rejected_access_token)
        {
            state.rejection = Some(CredentialRejection {
                access_token: rejected_access_token.to_string(),
                observed_at: Utc::now(),
                detail: detail.into(),
            });
        }
    }

    /// A manager holding `credentials` in memory that never persists a
    /// refresh, for tests.
    pub fn with_credentials(credentials: ChatGptCredentials) -> Self {
        Self {
            state: Mutex::new(CredentialState {
                credentials: Some(credentials),
                rejection: None,
            }),
            token_url: TOKEN_URL.to_string(),
            revoke_url: REVOKE_URL.to_string(),
            persist_refreshes: false,
            secrets: Arc::new(aidaemon_secrets::MemorySecretStore::new()),
            settings: LoginSettings::new("the agent", "the login command"),
        }
    }
}

/// Raw token endpoint response.
#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    /// Absent on some refresh responses; the caller keeps the previous token.
    #[serde(default)]
    refresh_token: Option<String>,
    /// Carries OIDC identity claims. Absent on some refresh responses.
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    expires_in: Option<i64>,
    #[serde(default)]
    scope: Option<String>,
}

/// Decode a JWT payload segment.
///
/// The token is received directly over TLS from `auth.openai.com`'s token
/// endpoint, and its `iss`, `aud`, `exp`, and `nonce` claims are validated
/// against this authorization attempt.
fn decode_jwt_payload(token: &str) -> Result<serde_json::Value> {
    let payload = token
        .split('.')
        .nth(1)
        .ok_or_else(|| anyhow!("token is not a JWT (expected three dot-separated segments)"))?;
    let normalized = payload.trim_end_matches('=');
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(normalized)
        .context("token payload is not valid base64url")?;
    serde_json::from_slice(&bytes).context("token payload is not valid JSON")
}

/// Verified identity extracted from an OpenAI OIDC `id_token`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VerifiedIdentity {
    account_id: String,
    subject: String,
}

/// Validate an ID token's claims (`iss`, `aud`, `exp`, `sub`, optional `nonce`,
/// and optional returning `expected_subject`).
///
/// The token is obtained directly from `https://auth.openai.com/api/accounts/oauth/token`
/// over TLS during the Authorization Code + PKCE exchange (OIDC Core 1.0 §3.1.3.7);
/// JWKS asymmetric signature verification is not performed locally.
fn validate_id_token_identity(
    id_token: &str,
    expected_client_id: &str,
    expected_nonce: Option<&str>,
    expected_subject: Option<&str>,
    now: DateTime<Utc>,
) -> Result<VerifiedIdentity> {
    let claims = decode_jwt_payload(id_token)?;
    let iss = claims
        .get("iss")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if iss != ISSUER {
        return Err(anyhow!("id_token issuer `{iss}` does not match `{ISSUER}`"));
    }
    let aud_matches = match claims.get("aud") {
        Some(serde_json::Value::String(aud)) => aud == expected_client_id,
        Some(serde_json::Value::Array(auds)) => {
            auds.iter().any(|v| v.as_str() == Some(expected_client_id))
        }
        _ => false,
    };
    if !aud_matches {
        return Err(anyhow!(
            "id_token audience does not match issued client_id `{expected_client_id}`"
        ));
    }
    let exp = claims
        .get("exp")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| anyhow!("id_token has no `exp` claim"))?;
    if exp + ID_TOKEN_EXP_SKEW_SECS <= now.timestamp() {
        return Err(anyhow!("id_token is expired"));
    }
    if let Some(expected_nonce) = expected_nonce {
        let nonce = claims
            .get("nonce")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("id_token has no `nonce` claim"))?;
        if nonce != expected_nonce {
            return Err(anyhow!("id_token nonce mismatch"));
        }
    }
    let subject = claims
        .get("sub")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("id_token has no `sub` claim"))?
        .to_string();
    if let Some(expected) = expected_subject
        && subject != expected
    {
        return Err(anyhow!(
            "signed-in ChatGPT account does not match the currently connected account; \
             sign out first to switch accounts"
        ));
    }
    let account_id = account_id_from_claims(&claims)?;
    Ok(VerifiedIdentity {
        account_id,
        subject,
    })
}

/// Validate an ID token's claims (`iss`, `aud`, `exp`, `sub`, and optional
/// `nonce`) and return the account identifier (`chatgpt_account_id` if present,
/// else `sub`).
pub fn validate_id_token(
    id_token: &str,
    expected_client_id: &str,
    expected_nonce: Option<&str>,
    now: DateTime<Utc>,
) -> Result<String> {
    validate_id_token_identity(id_token, expected_client_id, expected_nonce, None, now)
        .map(|identity| identity.account_id)
}

/// Pull the account id out of an id_token (`chatgpt_account_id` if present,
/// else `sub`).
pub fn account_id_from_id_token(id_token: &str) -> Result<String> {
    let claims = decode_jwt_payload(id_token)?;
    account_id_from_claims(&claims)
}

fn account_id_from_claims(claims: &serde_json::Value) -> Result<String> {
    if let Some(id) = claims
        .get(AUTH_CLAIM_NAMESPACE)
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(|id| id.as_str())
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return Ok(id.to_string());
    }
    claims
        .get("sub")
        .and_then(|id| id.as_str())
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            anyhow!(
                "id_token has neither `{AUTH_CLAIM_NAMESPACE}.chatgpt_account_id` nor `sub` — \
                 the account identity could not be read"
            )
        })
}

/// Confirm that the granted scopes include ChatGPT plan usage
/// (`chatgpt.tokens.use.direct`).
fn ensure_plan_usage_scope(token_scope: Option<&str>, access_token: &str) -> Result<()> {
    let has_scope = |raw: &str| raw.split_whitespace().any(|s| s == PLAN_USAGE_SCOPE);
    if let Some(scope) = token_scope.map(str::trim).filter(|s| !s.is_empty()) {
        if has_scope(scope) {
            return Ok(());
        }
        return Err(anyhow!(
            "ChatGPT plan usage permission (`{PLAN_USAGE_SCOPE}`) was not granted. \
             Enable \"Use your ChatGPT plan\" when signing in, or use an OpenAI API key."
        ));
    }
    if let Ok(claims) = decode_jwt_payload(access_token)
        && let Some(scope) = claims.get("scope").and_then(|v| v.as_str())
    {
        if has_scope(scope) {
            return Ok(());
        }
        return Err(anyhow!(
            "ChatGPT plan usage permission (`{PLAN_USAGE_SCOPE}`) was not granted. \
             Enable \"Use your ChatGPT plan\" when signing in, or use an OpenAI API key."
        ));
    }
    Err(anyhow!(
        "token response omitted granted scopes; cannot confirm `{PLAN_USAGE_SCOPE}` permission"
    ))
}

/// A sign-in started with [`ChatGptCredentialManager::start_login`].
pub struct LoginAttempt {
    /// Where the owner signs in.
    pub url: String,
    pkce: PkceChallenge,
    redirect_uri: String,
    pending_client_id: Option<String>,
    expected_subject: Option<String>,
    listener: Option<TcpListener>,
}

impl LoginAttempt {
    /// It waits on this machine's loopback callback listener (else for a
    /// pasted redirect address).
    pub fn on_loopback(&self) -> bool {
        self.listener.is_some()
    }
}

/// Generate a PKCE verifier/challenge pair plus CSRF `state` and OIDC `nonce`.
pub struct PkceChallenge {
    pub verifier: String,
    pub challenge: String,
    pub state: String,
    pub nonce: String,
}

impl PkceChallenge {
    pub fn generate() -> Self {
        use rand::RngExt;
        use sha2::Digest;

        let mut rng = rand::rng();
        let bytes: Vec<u8> = (0..32).map(|_| rng.random()).collect();
        let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&bytes);
        let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(sha2::Sha256::digest(verifier.as_bytes()));
        Self {
            verifier,
            challenge,
            state: uuid::Uuid::new_v4().to_string(),
            nonce: uuid::Uuid::new_v4().to_string(),
        }
    }
}

/// Parameters for building the Sign in with ChatGPT authorization URL.
pub struct AuthorizeParams<'a> {
    /// Issued client ID for returning sign-in, or `None` for initial dynamic
    /// registration (`client_id=dynamic_agent_client`).
    pub client_id: Option<&'a str>,
    /// Application name sent as `agent_name_hint` on initial registration.
    pub app_name: &'a str,
    /// Persistent host identifier (`urn:uuid:...`).
    pub host_id: &'a str,
    /// Exact loopback callback URI (`http://127.0.0.1:<port>/auth/callback`).
    pub redirect_uri: &'a str,
    pub challenge: &'a str,
    pub state: &'a str,
    pub nonce: &'a str,
    /// Optional retained ID token for returning sign-in.
    pub id_token_hint: Option<&'a str>,
}

/// Build the loopback callback URI for `port` (strictly `127.0.0.1`, never
/// `localhost`).
pub fn loopback_redirect_uri(port: u16) -> String {
    format!("http://127.0.0.1:{port}{CALLBACK_PATH}")
}

/// Build the URL the user opens to authorize the agent via Sign in with ChatGPT.
pub fn build_authorize_url(params: &AuthorizeParams<'_>) -> String {
    let mut url = reqwest::Url::parse(AUTHORIZE_URL).expect("AUTHORIZE_URL is a valid URL");
    {
        let mut pairs = url.query_pairs_mut();
        match params.client_id.map(str::trim).filter(|id| !id.is_empty()) {
            Some(client_id) => {
                pairs.append_pair("client_id", client_id);
                if let Some(hint) = params
                    .id_token_hint
                    .map(str::trim)
                    .filter(|h| !h.is_empty())
                {
                    pairs.append_pair("id_token_hint", hint);
                }
            }
            None => {
                pairs
                    .append_pair("client_id", DYNAMIC_CLIENT_ID)
                    .append_pair("agent_name_hint", params.app_name);
            }
        }
        pairs
            .append_pair("ext_agent_host_id", params.host_id)
            .append_pair("response_type", "code")
            .append_pair("redirect_uri", params.redirect_uri)
            .append_pair("scope", SCOPES)
            .append_pair("resource", API_RESOURCE)
            .append_pair("state", params.state)
            .append_pair("nonce", params.nonce)
            .append_pair("code_challenge", params.challenge)
            .append_pair("code_challenge_method", "S256");
    }
    url.to_string()
}

/// Parsed query parameters from the OAuth redirect callback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallbackResult {
    pub code: String,
    /// Present on initial dynamic registration (`oaiapp_...`); may be omitted
    /// on returning reauthorization.
    pub client_id: Option<String>,
}

/// Extract `code` (and any issued `client_id`) from a redirect URL or request
/// target, validating `state`.
pub fn parse_callback(input: &str, expected_state: &str) -> Result<CallbackResult> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("no redirect URL provided"));
    }
    let absolute = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        let query = trimmed.trim_start_matches('?');
        let query = query
            .split_once('?')
            .map(|(_, q)| q)
            .unwrap_or(query.trim_start_matches(CALLBACK_PATH));
        format!("{PASTE_REDIRECT_URI}?{}", query.trim_start_matches('?'))
    };

    let url = reqwest::Url::parse(&absolute).context("could not parse the redirect URL")?;
    let mut code = None;
    let mut state = None;
    let mut client_id = None;
    let mut error = None;
    let mut error_description = None;
    for (k, v) in url.query_pairs() {
        match k.as_ref() {
            "code" => code = Some(v.to_string()),
            "state" => state = Some(v.to_string()),
            "client_id" => {
                let value = v.trim();
                if !value.is_empty() {
                    client_id = Some(value.to_string());
                }
            }
            "error" => error = Some(v.to_string()),
            "error_description" => error_description = Some(v.to_string()),
            _ => {}
        }
    }

    let state = state.ok_or_else(|| anyhow!("redirect URL has no `state` parameter"))?;
    if state != expected_state {
        return Err(anyhow!(
            "state mismatch — the redirect does not belong to this login attempt"
        ));
    }
    if let Some(err) = error {
        let detail = error_description.unwrap_or_else(|| "no description".to_string());
        return Err(anyhow!("authorization was denied: {err} ({detail})"));
    }
    let code = code.ok_or_else(|| anyhow!("redirect URL has no `code` parameter"))?;
    Ok(CallbackResult { code, client_id })
}

/// Resolve the issued `client_id` for token exchange.
///
/// Initial dynamic registration requires a non-empty issued client ID from the
/// callback (never `dynamic_agent_client`). Returning reauthorization keeps
/// the pending client ID and rejects a mismatched one.
pub fn resolve_issued_client_id(pending: Option<&str>, returned: Option<&str>) -> Result<String> {
    let returned = returned.map(str::trim).filter(|id| !id.is_empty());
    match pending.map(str::trim).filter(|id| !id.is_empty()) {
        None => match returned {
            Some(id) if id != DYNAMIC_CLIENT_ID && id != LEGACY_CODEX_CLIENT_ID => {
                Ok(id.to_string())
            }
            Some(id) => Err(anyhow!(
                "registration callback returned invalid client_id `{id}`"
            )),
            None => Err(anyhow!(
                "registration callback did not include an issued `client_id`"
            )),
        },
        Some(expected) => match returned {
            Some(got) if got != expected => Err(anyhow!(
                "callback client_id `{got}` does not match the pending registration `{expected}`"
            )),
            _ => Ok(expected.to_string()),
        },
    }
}

struct ExchangedTokens {
    credentials: ChatGptCredentials,
    id_token: Option<String>,
    subject: Option<String>,
}

fn credentials_from_response(
    resp: TokenResponse,
    client_id: &str,
    expected_nonce: Option<&str>,
    expected_subject: Option<&str>,
    previous: Option<&ChatGptCredentials>,
) -> Result<ExchangedTokens> {
    if previous.is_none() || resp.scope.is_some() {
        ensure_plan_usage_scope(resp.scope.as_deref(), &resp.access_token)?;
    }

    let refresh_token = resp
        .refresh_token
        .or_else(|| previous.map(|p| p.refresh_token.clone()))
        .ok_or_else(|| anyhow!("token response contained no refresh_token"))?;

    let (account_id, subject) = match resp.id_token.as_deref() {
        Some(id_token) => {
            let identity = validate_id_token_identity(
                id_token,
                client_id,
                expected_nonce,
                expected_subject,
                Utc::now(),
            )?;
            (identity.account_id, Some(identity.subject))
        }
        None => {
            let account_id = previous.map(|p| p.account_id.clone()).ok_or_else(|| {
                anyhow!("token response contained no id_token to verify account identity")
            })?;
            (account_id, None)
        }
    };

    let expires_in = resp.expires_in.unwrap_or(3600).max(0);
    Ok(ExchangedTokens {
        credentials: ChatGptCredentials {
            access_token: resp.access_token,
            refresh_token,
            account_id,
            client_id: client_id.to_string(),
            expires_at: Utc::now() + chrono::Duration::seconds(expires_in),
        },
        id_token: resp.id_token,
        subject,
    })
}

/// Exchange an authorization code for Sign in with ChatGPT tokens.
pub async fn exchange_code(
    client: &reqwest::Client,
    client_id: &str,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
    expected_nonce: &str,
) -> Result<(ChatGptCredentials, Option<String>)> {
    let exchanged = exchange_code_at(
        client,
        TOKEN_URL,
        client_id,
        code,
        verifier,
        redirect_uri,
        expected_nonce,
        None,
    )
    .await?;
    Ok((exchanged.credentials, exchanged.id_token))
}

#[allow(clippy::too_many_arguments)]
async fn exchange_code_at(
    client: &reqwest::Client,
    token_url: &str,
    client_id: &str,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
    expected_nonce: &str,
    expected_subject: Option<&str>,
) -> Result<ExchangedTokens> {
    let resp = client
        .post(token_url)
        .form(&[
            ("grant_type", "authorization_code"),
            ("client_id", client_id),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("code_verifier", verifier),
            ("resource", API_RESOURCE),
        ])
        .send()
        .await
        .context("token exchange request failed")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!(
            "token exchange returned {status}: {}",
            truncate_for_error(&body)
        ));
    }
    let parsed: TokenResponse =
        serde_json::from_str(&body).context("token exchange returned unexpected JSON")?;
    credentials_from_response(
        parsed,
        client_id,
        Some(expected_nonce),
        expected_subject,
        None,
    )
}

/// Exchange a refresh token for a fresh access token.
///
/// Per OpenAI's SIWC specification, refresh omits `scope` to retain the grant
/// and includes `resource=https://api.openai.com/v1` and the issued `client_id`.
async fn refresh_credentials_at(
    client: &reqwest::Client,
    settings: &LoginSettings,
    previous: &ChatGptCredentials,
    expected_subject: Option<&str>,
    token_url: &str,
) -> Result<ExchangedTokens> {
    let resp = client
        .post(token_url)
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", previous.client_id.as_str()),
            ("refresh_token", previous.refresh_token.as_str()),
            ("resource", API_RESOURCE),
        ])
        .send()
        .await
        .context("token refresh request failed")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!(
            "token refresh returned {status}: {} — sign in again with `{}`",
            truncate_for_error(&body),
            settings.login_command
        ));
    }
    let parsed: TokenResponse =
        serde_json::from_str(&body).context("token refresh returned unexpected JSON")?;
    credentials_from_response(
        parsed,
        &previous.client_id,
        None,
        expected_subject,
        Some(previous),
    )
}

fn truncate_for_error(body: &str) -> String {
    const LIMIT: usize = 300;
    let trimmed = body.trim();
    if trimmed.chars().count() <= LIMIT {
        return trimmed.to_string();
    }
    let head: String = trimmed.chars().take(LIMIT).collect();
    format!("{head}…")
}

/// Read or create the persistent opaque `ext_agent_host_id` for this host.
pub fn ensure_host_id(secrets: &dyn SecretStore) -> Result<String> {
    if let Ok(existing) = secrets.get(&key("host_id")) {
        let value = existing.expose().trim();
        if !value.is_empty() {
            return Ok(value.to_string());
        }
    }
    let host_id = format!("urn:uuid:{}", uuid::Uuid::new_v4());
    secrets.set(&key("host_id"), &host_id)?;
    Ok(host_id)
}

fn load_id_token(secrets: &dyn SecretStore) -> Option<String> {
    let token = secrets.get(&key("id_token")).ok()?.into_exposed();
    (!token.trim().is_empty()).then_some(token)
}

fn load_subject(secrets: &dyn SecretStore) -> Option<String> {
    let sub = secrets.get(&key("subject")).ok()?.into_exposed();
    let trimmed = sub.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn load_registered_client_id(secrets: &dyn SecretStore) -> Option<String> {
    let client_id = secrets.get(&key("client_id")).ok()?.into_exposed();
    let trimmed = client_id.trim();
    if trimmed.is_empty() || trimmed == DYNAMIC_CLIENT_ID || trimmed == LEGACY_CODEX_CLIENT_ID {
        return None;
    }
    Some(trimmed.to_string())
}

/// Persist credentials to `secrets`.
pub fn store_credentials(secrets: &dyn SecretStore, creds: &ChatGptCredentials) -> Result<()> {
    secrets.set(&key("access_token"), &creds.access_token)?;
    secrets.set(&key("refresh_token"), &creds.refresh_token)?;
    secrets.set(&key("account_id"), &creds.account_id)?;
    secrets.set(&key("client_id"), &creds.client_id)?;
    secrets.set(&key("expires_at"), &creds.expires_at.to_rfc3339())?;
    Ok(())
}

/// Load stored credentials from `secrets`, if a valid Sign in with ChatGPT
/// login exists. Pre-migration tokens without an issued `client_id` (or tied to
/// the legacy Codex client ID) are treated as disconnected so the user signs in
/// through the official flow.
pub fn load_credentials(secrets: &dyn SecretStore) -> Option<ChatGptCredentials> {
    let access_token = secrets.get(&key("access_token")).ok()?.into_exposed();
    let refresh_token = secrets.get(&key("refresh_token")).ok()?.into_exposed();
    let account_id = secrets.get(&key("account_id")).ok()?.into_exposed();
    let client_id = secrets.get(&key("client_id")).ok()?.into_exposed();
    let expires_at = secrets
        .get(&key("expires_at"))
        .ok()
        .and_then(|raw| DateTime::parse_from_rfc3339(raw.expose()).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|| Utc::now() - chrono::Duration::hours(1));

    if access_token.is_empty()
        || refresh_token.is_empty()
        || account_id.is_empty()
        || client_id.is_empty()
        || client_id == DYNAMIC_CLIENT_ID
        || client_id == LEGACY_CODEX_CLIENT_ID
    {
        return None;
    }
    Some(ChatGptCredentials {
        access_token,
        refresh_token,
        account_id,
        client_id,
        expires_at,
    })
}

fn callback_page(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

/// Bind an ephemeral loopback listener on `127.0.0.1` so sign-in never
/// collides with another tool's port.
pub async fn bind_callback_listener() -> Result<TcpListener> {
    TcpListener::bind(("127.0.0.1", 0))
        .await
        .context("could not bind a local loopback callback port on 127.0.0.1")
}

/// Wait for OpenAI to redirect the browser back to the loopback listener.
async fn await_callback(
    listener: TcpListener,
    expected_state: &str,
    app_name: &str,
) -> Result<CallbackResult> {
    let accept = async {
        loop {
            let (mut socket, _) = listener.accept().await.context("callback accept failed")?;
            let mut buf = vec![0u8; 8192];
            let n = tokio::time::timeout(CALLBACK_READ_TIMEOUT, socket.read(&mut buf))
                .await
                .ok()
                .and_then(Result::ok)
                .unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let target = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or_default()
                .to_string();

            if !target.starts_with(CALLBACK_PATH) {
                let _ = socket
                    .write_all(callback_page("<p>Waiting for the sign-in redirect…</p>").as_bytes())
                    .await;
                let _ = socket.shutdown().await;
                continue;
            }

            if !callback_has_state(&target, expected_state) {
                let _ = socket
                    .write_all(
                        callback_page("<h2>This redirect is not for the sign-in in progress.</h2>")
                            .as_bytes(),
                    )
                    .await;
                let _ = socket.shutdown().await;
                continue;
            }
            let result = parse_callback(&target, expected_state);
            let page = match &result {
                Ok(_) => callback_page(&format!(
                    "<h2>{} is connected to your ChatGPT account.</h2>\
                     <p>You can close this tab and return to the app or terminal. \
                     Manage usage anytime at <a href=\"https://chatgpt.com/settings/usage\">\
                     chatgpt.com/settings/usage</a>.</p>",
                    html_escape(app_name)
                )),
                Err(e) => callback_page(&format!(
                    "<h2>Sign-in failed.</h2><p>{}</p>",
                    html_escape(&e.to_string())
                )),
            };
            let _ = socket.write_all(page.as_bytes()).await;
            let _ = socket.shutdown().await;
            return result;
        }
    };

    tokio::time::timeout(CALLBACK_TIMEOUT, accept)
        .await
        .map_err(|_| {
            anyhow!(
                "timed out after {} seconds waiting for the ChatGPT sign-in redirect",
                CALLBACK_TIMEOUT.as_secs()
            )
        })?
}

fn callback_has_state(target: &str, expected_state: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(&format!("http://127.0.0.1{target}")) else {
        return false;
    };
    url.query_pairs()
        .any(|(key, value)| key == "state" && value == expected_state)
}

fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Reads the redirect URL back from the user, given the authorization URL.
pub type RedirectReader = Box<dyn FnOnce(&str) -> Result<String> + Send>;

/// How the authorization code gets back to us.
pub enum CallbackMode {
    /// Listen on an ephemeral loopback port on `127.0.0.1`.
    Loopback,
    /// Read the pasted redirect URL from the supplied closure.
    Paste(RedirectReader),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn a_silent_or_foreign_callback_does_not_end_the_sign_in() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let wait =
            tokio::spawn(
                async move { await_callback(listener, "state-synthetic-1", "Test").await },
            );
        let _silent = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut foreign = tokio::net::TcpStream::connect(addr).await.unwrap();
        foreign
            .write_all(b"GET /auth/callback?code=x&state=other HTTP/1.1\r\n\r\n")
            .await
            .unwrap();
        let mut sink = Vec::new();
        let _ = foreign.read_to_end(&mut sink).await;
        assert!(!wait.is_finished());
        let mut real = tokio::net::TcpStream::connect(addr).await.unwrap();
        real.write_all(
            b"GET /auth/callback?code=code-synthetic&state=state-synthetic-1&client_id=oaiapp_synthetic HTTP/1.1\r\n\r\n",
        )
        .await
        .unwrap();
        let callback = wait.await.unwrap().unwrap();
        assert_eq!(callback.code, "code-synthetic");
        assert_eq!(callback.client_id.as_deref(), Some("oaiapp_synthetic"));
    }

    fn fake_jwt(claims: serde_json::Value) -> String {
        let encode = |v: &serde_json::Value| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(v.to_string().as_bytes())
        };
        format!(
            "{}.{}.{}",
            encode(&json!({"alg": "RS256", "typ": "JWT"})),
            encode(&claims),
            "not-a-real-signature"
        )
    }

    #[test]
    fn validates_id_token_claims_and_reads_account_identity() {
        let now = Utc::now();
        let token = fake_jwt(json!({
            "iss": ISSUER,
            "aud": "oaiapp_synthetic",
            "sub": "user-synthetic-1",
            "nonce": "nonce-1",
            "exp": (now + chrono::Duration::hours(1)).timestamp(),
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-synthetic-1",
            }
        }));
        assert_eq!(
            validate_id_token(&token, "oaiapp_synthetic", Some("nonce-1"), now).unwrap(),
            "acct-synthetic-1"
        );
        // Falls back to `sub` when `chatgpt_account_id` is absent.
        let sub_only = fake_jwt(json!({
            "iss": ISSUER,
            "aud": "oaiapp_synthetic",
            "sub": "user-synthetic-2",
            "nonce": "nonce-1",
            "exp": (now + chrono::Duration::hours(1)).timestamp(),
        }));
        assert_eq!(
            validate_id_token(&sub_only, "oaiapp_synthetic", Some("nonce-1"), now).unwrap(),
            "user-synthetic-2"
        );
        // Rejects wrong audience, issuer, nonce, or expired token.
        assert!(validate_id_token(&sub_only, "other_client", Some("nonce-1"), now).is_err());
        assert!(validate_id_token(&sub_only, "oaiapp_synthetic", Some("wrong"), now).is_err());
        let expired = fake_jwt(json!({
            "iss": ISSUER,
            "aud": "oaiapp_synthetic",
            "sub": "user-synthetic-2",
            "exp": (now - chrono::Duration::hours(1)).timestamp(),
        }));
        assert!(validate_id_token(&expired, "oaiapp_synthetic", None, now).is_err());
    }

    #[test]
    fn rejects_non_jwt() {
        assert!(account_id_from_id_token("not-a-jwt").is_err());
    }

    #[test]
    fn authorize_url_uses_dynamic_registration_and_siwc_parameters() {
        let redirect = loopback_redirect_uri(54321);
        let url = build_authorize_url(&AuthorizeParams {
            client_id: None,
            app_name: "aidaemon",
            host_id: "urn:uuid:11111111-2222-4333-8444-555555555555",
            redirect_uri: &redirect,
            challenge: "challenge-value",
            state: "state-value",
            nonce: "nonce-value",
            id_token_hint: None,
        });
        let parsed = reqwest::Url::parse(&url).unwrap();
        assert_eq!(parsed.path(), "/api/accounts/authorize");
        let params: std::collections::HashMap<_, _> = parsed.query_pairs().into_owned().collect();
        assert_eq!(params["client_id"], DYNAMIC_CLIENT_ID);
        assert_eq!(params["agent_name_hint"], "aidaemon");
        assert_eq!(
            params["ext_agent_host_id"],
            "urn:uuid:11111111-2222-4333-8444-555555555555"
        );
        assert_eq!(params["resource"], API_RESOURCE);
        assert_eq!(
            params["redirect_uri"],
            "http://127.0.0.1:54321/auth/callback"
        );
        assert_eq!(params["nonce"], "nonce-value");
        assert_eq!(params["code_challenge"], "challenge-value");
        assert_eq!(params["code_challenge_method"], "S256");
        assert_eq!(params["state"], "state-value");
        assert!(params["scope"].contains(PLAN_USAGE_SCOPE));
        assert!(params["scope"].contains("offline_access"));
    }

    #[test]
    fn reauth_url_reuses_issued_client_id_without_agent_name_hint() {
        let url = build_authorize_url(&AuthorizeParams {
            client_id: Some("oaiapp_issued_1"),
            app_name: "aidaemon",
            host_id: "urn:uuid:11111111-2222-4333-8444-555555555555",
            redirect_uri: PASTE_REDIRECT_URI,
            challenge: "challenge-value",
            state: "state-value",
            nonce: "nonce-value",
            id_token_hint: Some("prior-id-token"),
        });
        let parsed = reqwest::Url::parse(&url).unwrap();
        let params: std::collections::HashMap<_, _> = parsed.query_pairs().into_owned().collect();
        assert_eq!(params["client_id"], "oaiapp_issued_1");
        assert!(!params.contains_key("agent_name_hint"));
        assert_eq!(params["id_token_hint"], "prior-id-token");
    }

    #[test]
    fn parses_full_redirect_url_with_issued_client_id() {
        let cb = parse_callback(
            "http://127.0.0.1:1455/auth/callback?code=abc123&state=st-1&client_id=oaiapp_1",
            "st-1",
        )
        .unwrap();
        assert_eq!(cb.code, "abc123");
        assert_eq!(cb.client_id.as_deref(), Some("oaiapp_1"));
    }

    #[test]
    fn resolves_and_validates_issued_client_id() {
        assert_eq!(
            resolve_issued_client_id(None, Some("oaiapp_123")).unwrap(),
            "oaiapp_123"
        );
        assert!(resolve_issued_client_id(None, None).is_err());
        assert!(resolve_issued_client_id(None, Some(DYNAMIC_CLIENT_ID)).is_err());
        assert!(resolve_issued_client_id(None, Some(LEGACY_CODEX_CLIENT_ID)).is_err());
        assert_eq!(
            resolve_issued_client_id(Some("oaiapp_123"), None).unwrap(),
            "oaiapp_123"
        );
        assert_eq!(
            resolve_issued_client_id(Some("oaiapp_123"), Some("oaiapp_123")).unwrap(),
            "oaiapp_123"
        );
        assert!(resolve_issued_client_id(Some("oaiapp_123"), Some("oaiapp_other")).is_err());
    }

    #[test]
    fn rejects_missing_plan_usage_scope() {
        assert!(
            ensure_plan_usage_scope(Some("openid profile email"), "not-a-jwt")
                .unwrap_err()
                .to_string()
                .contains(PLAN_USAGE_SCOPE)
        );
        assert!(
            ensure_plan_usage_scope(
                Some(
                    "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
                ),
                "not-a-jwt"
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_state_mismatch_and_surfaces_denial() {
        let err = parse_callback("/auth/callback?code=abc&state=other", "st-1")
            .unwrap_err()
            .to_string();
        assert!(err.contains("state mismatch"), "unexpected error: {err}");
        let denied = parse_callback(
            "/auth/callback?error=access_denied&error_description=User+declined&state=st-1",
            "st-1",
        )
        .unwrap_err()
        .to_string();
        assert!(
            denied.contains("access_denied"),
            "unexpected error: {denied}"
        );
    }

    #[test]
    fn load_credentials_rejects_legacy_tokens_without_issued_client_id() {
        let store = aidaemon_secrets::MemorySecretStore::new();
        store.set(&key("access_token"), "old-access").unwrap();
        store.set(&key("refresh_token"), "old-refresh").unwrap();
        store.set(&key("account_id"), "acct-1").unwrap();
        store
            .set(
                &key("expires_at"),
                &(Utc::now() + chrono::Duration::hours(1)).to_rfc3339(),
            )
            .unwrap();
        // Legacy login had no client_id stored -> treated as disconnected.
        assert!(load_credentials(&store).is_none());
        // Legacy Codex CLI client ID -> also rejected.
        store
            .set(&key("client_id"), LEGACY_CODEX_CLIENT_ID)
            .unwrap();
        assert!(load_credentials(&store).is_none());
        // Issued SIWC client ID -> accepted.
        store.set(&key("client_id"), "oaiapp_synthetic").unwrap();
        let loaded = load_credentials(&store).expect("valid SIWC credentials");
        assert_eq!(loaded.client_id, "oaiapp_synthetic");
    }

    #[test]
    fn host_id_is_created_once_and_reused() {
        let store = aidaemon_secrets::MemorySecretStore::new();
        let first = ensure_host_id(&store).unwrap();
        assert!(first.starts_with("urn:uuid:"), "{first}");
        let second = ensure_host_id(&store).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn refresh_keeps_previous_refresh_token_and_account_when_omitted() {
        let previous = ChatGptCredentials {
            access_token: "old-access".into(),
            refresh_token: "keep-me".into(),
            account_id: "acct-synthetic-1".into(),
            client_id: "oaiapp_synthetic".into(),
            expires_at: Utc::now(),
        };
        let resp = TokenResponse {
            access_token: "new-access".into(),
            refresh_token: None,
            id_token: None,
            expires_in: Some(3600),
            scope: None,
        };
        let out = credentials_from_response(resp, &previous.client_id, None, None, Some(&previous))
            .unwrap()
            .credentials;
        assert_eq!(out.access_token, "new-access");
        assert_eq!(out.refresh_token, "keep-me");
        assert_eq!(out.account_id, "acct-synthetic-1");
        assert_eq!(out.client_id, "oaiapp_synthetic");
        assert!(out.expires_at > Utc::now() + chrono::Duration::minutes(50));
    }

    #[test]
    fn refresh_adopts_rotated_refresh_token() {
        let previous = ChatGptCredentials {
            access_token: "old-access".into(),
            refresh_token: "old-refresh".into(),
            account_id: "acct-synthetic-1".into(),
            client_id: "oaiapp_synthetic".into(),
            expires_at: Utc::now(),
        };
        let resp = TokenResponse {
            access_token: "new-access".into(),
            refresh_token: Some("rotated-refresh".into()),
            id_token: None,
            expires_in: Some(3600),
            scope: None,
        };
        let out = credentials_from_response(resp, &previous.client_id, None, None, Some(&previous))
            .unwrap()
            .credentials;
        assert_eq!(out.refresh_token, "rotated-refresh");
    }

    #[test]
    fn returning_subject_mismatch_is_rejected() {
        let now = Utc::now();
        let token = fake_jwt(json!({
            "iss": ISSUER,
            "aud": "oaiapp_synthetic",
            "sub": "user-synthetic-other",
            "nonce": "nonce-1",
            "exp": (now + chrono::Duration::hours(1)).timestamp(),
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-synthetic-2",
            }
        }));
        let err = validate_id_token_identity(
            &token,
            "oaiapp_synthetic",
            Some("nonce-1"),
            Some("user-synthetic-original"),
            now,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("sign out first"), "{err}");
    }

    #[tokio::test]
    async fn logout_preserves_issued_client_id_and_host_id_for_reauth() {
        let secrets = Arc::new(aidaemon_secrets::MemorySecretStore::new());
        let settings = LoginSettings::new("aidaemon", "aidaemon auth login openai");
        let mut manager = ChatGptCredentialManager::new(secrets.clone(), settings);
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let closed_addr = listener.local_addr().unwrap();
        drop(listener);
        manager.revoke_url = format!("http://{closed_addr}/revoke");

        let host_id = manager.host_id().unwrap();
        manager
            .store_credentials(&synthetic_credentials("active-access"))
            .unwrap();
        secrets.set(&key("subject"), "user-synthetic-1").unwrap();
        secrets.set(&key("id_token"), "synthetic-id-token").unwrap();

        // Revocation fails against an unreachable loopback address (offline
        // logout) while still clearing local session entries and preserving client_id.
        let client = reqwest::Client::new();
        manager.logout(&client).await.unwrap();

        assert!(!manager.is_connected());
        assert_eq!(manager.host_id().unwrap(), host_id);
        assert_eq!(
            load_registered_client_id(secrets.as_ref()).as_deref(),
            Some("oaiapp_synthetic")
        );
        assert!(load_subject(secrets.as_ref()).is_none());
        assert!(load_id_token(secrets.as_ref()).is_none());

        // Subsequent sign-in reuses the issued client_id without re-registering.
        let attempt = manager.start_login(false).await.unwrap();
        let parsed = reqwest::Url::parse(&attempt.url).unwrap();
        let params: std::collections::HashMap<_, _> = parsed.query_pairs().into_owned().collect();
        assert_eq!(params["client_id"], "oaiapp_synthetic");
        assert!(!params.contains_key("agent_name_hint"));
    }

    #[tokio::test]
    async fn start_login_rejects_legacy_client_id_override() {
        let secrets = Arc::new(aidaemon_secrets::MemorySecretStore::new());
        let mut settings = LoginSettings::new("aidaemon", "aidaemon auth login openai");
        settings.client_id = Some(LEGACY_CODEX_CLIENT_ID.to_string());
        let manager = ChatGptCredentialManager::new(secrets, settings);
        let err = manager.start_login(false).await.err().unwrap().to_string();
        assert!(err.contains("not allowed"), "{err}");
    }

    #[tokio::test]
    async fn failed_code_exchange_still_persists_issued_client_id() {
        let secrets = Arc::new(aidaemon_secrets::MemorySecretStore::new());
        let settings = LoginSettings::new("aidaemon", "aidaemon auth login openai");
        let mut manager = ChatGptCredentialManager::new(secrets.clone(), settings);
        // Point token_url at an unused local port so exchange_code fails immediately.
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let closed_addr = listener.local_addr().unwrap();
        drop(listener);
        manager.token_url = format!("http://{closed_addr}/token");

        let attempt = manager.start_login(false).await.unwrap();
        let redirect = format!(
            "{PASTE_REDIRECT_URI}?code=synthetic-code&state={}&client_id=oaiapp_issued_once",
            attempt.pkce.state
        );
        let client = reqwest::Client::new();
        assert!(
            manager
                .finish_with_redirect(&client, &attempt, &redirect)
                .await
                .is_err()
        );
        assert_eq!(
            load_registered_client_id(secrets.as_ref()).as_deref(),
            Some("oaiapp_issued_once")
        );
    }

    #[test]
    fn needs_refresh_uses_skew() {
        let now = Utc::now();
        let soon = ChatGptCredentials {
            access_token: "a".into(),
            refresh_token: "r".into(),
            account_id: "acct-synthetic-1".into(),
            client_id: "oaiapp_synthetic".into(),
            expires_at: now + chrono::Duration::minutes(1),
        };
        assert!(soon.needs_refresh(now));

        let later = ChatGptCredentials {
            expires_at: now + chrono::Duration::minutes(30),
            ..soon.clone()
        };
        assert!(!later.needs_refresh(now));
    }

    fn synthetic_credentials(access_token: &str) -> ChatGptCredentials {
        ChatGptCredentials {
            access_token: access_token.to_string(),
            refresh_token: "synthetic-refresh".into(),
            account_id: "acct-synthetic-1".into(),
            client_id: "oaiapp_synthetic".into(),
            expires_at: Utc::now() + chrono::Duration::hours(1),
        }
    }

    #[tokio::test]
    async fn remote_rejection_forces_refresh_before_local_expiry() {
        let manager = ChatGptCredentialManager::with_credentials(synthetic_credentials("old"));
        let refreshed = manager
            .refresh_after_rejection_with("old", false, |_current| async {
                Ok(synthetic_credentials("rotated"))
            })
            .await
            .expect("forced refresh");

        assert_eq!(refreshed.access_token, "rotated");
    }

    #[tokio::test]
    async fn stale_rejection_reuses_concurrently_rotated_credentials() {
        let manager = ChatGptCredentialManager::with_credentials(synthetic_credentials("rotated"));
        let refresh_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls = refresh_calls.clone();
        let credentials = manager
            .refresh_after_rejection_with("old", false, move |_current| async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(synthetic_credentials("unexpected"))
            })
            .await
            .expect("newer credential should be reusable");

        assert_eq!(credentials.access_token, "rotated");
        assert_eq!(refresh_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn failed_remote_recovery_opens_bounded_auth_circuit() {
        let manager = ChatGptCredentialManager::with_credentials(synthetic_credentials("bad"));
        let first = manager
            .refresh_after_rejection_with("bad", false, |_current| async {
                Err(anyhow!("synthetic refresh rejection"))
            })
            .await;
        assert!(first.is_err());

        let refresh_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls = refresh_calls.clone();
        let second = manager
            .refresh_after_rejection_with("bad", false, move |_current| async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(synthetic_credentials("unexpected"))
            })
            .await;
        assert!(second.is_err());
        assert_eq!(refresh_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn pkce_challenge_is_sha256_of_verifier() {
        use sha2::Digest;
        let pkce = PkceChallenge::generate();
        let expected = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(sha2::Sha256::digest(pkce.verifier.as_bytes()));
        assert_eq!(pkce.challenge, expected);
        assert_ne!(pkce.verifier, pkce.challenge);
        assert!(!pkce.nonce.is_empty());
    }
}
