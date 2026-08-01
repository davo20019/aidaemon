//! ChatGPT subscription auth (Codex OAuth).
//!
//! Lets aidaemon drive the ChatGPT Codex backend with a user's ChatGPT
//! Plus/Pro/Business subscription instead of a metered OpenAI API key.
//!
//! Flow: OAuth 2.0 authorization code + PKCE against `auth.openai.com`, with the
//! authorization code delivered to a fixed loopback redirect
//! (`http://localhost:1455/auth/callback`). Both the port and the path are fixed
//! by OpenAI's client registration, so this cannot reuse [`crate::oauth::OAuthGateway`],
//! whose callbacks are all derived from one configurable `<base>/oauth/callback`.
//!
//! `client_id` below is a *public* OAuth client identifier, not a secret — it is
//! transmitted in the query string of every authorization request. Set
//! `AIDAEMON_CHATGPT_CLIENT_ID` to override it if OpenAI rotates the value.
//!
//! Credentials live in the OS keychain via [`crate::config::store_in_keychain`],
//! which falls back to the `.env` file when `AIDAEMON_NO_KEYCHAIN=1`, so headless
//! installs (launchd, systemd, containers) still work.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing::{debug, warn};

/// Authorization endpoint (user-facing sign-in).
pub const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
/// Token endpoint (code exchange and refresh).
pub const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
/// Token revocation endpoint, used on logout.
pub const REVOKE_URL: &str = "https://auth.openai.com/oauth/revoke";

/// ChatGPT's subscription-backed Codex API root. Chat and image adapters share
/// this root, but remain independent consumers of the OAuth credentials.
pub const CODEX_BACKEND_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

/// Public OAuth client identifier for ChatGPT/Codex sign-in. Not a secret.
pub const DEFAULT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
/// Loopback redirect registered for the client. Port and path are not ours to pick.
pub const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
/// Port component of [`REDIRECT_URI`].
pub const CALLBACK_PORT: u16 = 1455;
/// Scopes required for subscription-backed model access. `offline_access` is what
/// yields a refresh token; without it the daemon would need re-login hourly.
pub const SCOPES: &str = "openid profile email offline_access";

/// Service name used to namespace keychain entries.
pub const SERVICE: &str = "openai_chatgpt";

/// Namespaced claim on the id_token that carries ChatGPT account details.
const AUTH_CLAIM_NAMESPACE: &str = "https://api.openai.com/auth";

/// How long to wait for the user to complete the browser sign-in.
///
/// 15 minutes, matching Codex's own device-auth window. Five was not enough in
/// practice: signing in to ChatGPT, clearing 2FA, and approving can easily
/// outlast it, and the failure costs the user the whole flow.
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(900);

/// Refresh this far ahead of actual expiry so an in-flight request cannot
/// straddle the boundary.
const REFRESH_SKEW: chrono::Duration = chrono::Duration::minutes(5);

fn client_id() -> String {
    std::env::var("AIDAEMON_CHATGPT_CLIENT_ID")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_CLIENT_ID.to_string())
}

fn key(suffix: &str) -> String {
    format!("oauth_{SERVICE}_{suffix}")
}

/// Stored ChatGPT subscription credentials.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatGptCredentials {
    pub access_token: String,
    pub refresh_token: String,
    /// ChatGPT account id, sent as the `chatgpt-account-id` request header.
    pub account_id: String,
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
/// OpenAI rotates refresh tokens. Keeping the refresh lock here, rather than in
/// an individual model or image provider, prevents concurrent adapters from
/// refreshing the same token and invalidating one another.
pub struct ChatGptCredentialManager {
    credentials: Mutex<Option<ChatGptCredentials>>,
}

impl ChatGptCredentialManager {
    fn new() -> Self {
        Self {
            credentials: Mutex::new(None),
        }
    }

    /// Return a usable access token, refreshing and persisting it under one
    /// process-wide lock when it is close to expiry.
    pub async fn usable_credentials(&self, client: &reqwest::Client) -> Result<ChatGptCredentials> {
        let mut guard = self.credentials.lock().await;

        if guard.is_none() {
            *guard = load_credentials();
        }

        let current = guard.clone().ok_or_else(|| {
            anyhow!(
                "No ChatGPT subscription login found. Run `aidaemon auth login openai` to connect \
                 your ChatGPT account."
            )
        })?;

        if !current.needs_refresh(Utc::now()) {
            return Ok(current);
        }

        debug!("Refreshing ChatGPT subscription access token");
        match refresh_credentials(client, &current).await {
            Ok(refreshed) => {
                // Persist immediately: OpenAI rotates refresh tokens and the
                // old token may already be dead.
                if let Err(error) = store_credentials(&refreshed) {
                    warn!(%error, "Refreshed ChatGPT tokens could not be persisted");
                }
                *guard = Some(refreshed.clone());
                Ok(refreshed)
            }
            Err(error) => {
                // Reload storage on the next call instead of retrying a token
                // that is already known to be unusable.
                *guard = None;
                Err(error)
            }
        }
    }

    async fn replace_cached(&self, credentials: Option<ChatGptCredentials>) {
        *self.credentials.lock().await = credentials;
    }

    #[cfg(test)]
    pub(crate) fn with_credentials(credentials: ChatGptCredentials) -> Self {
        Self {
            credentials: Mutex::new(Some(credentials)),
        }
    }
}

static SHARED_CREDENTIAL_MANAGER: Lazy<Arc<ChatGptCredentialManager>> =
    Lazy::new(|| Arc::new(ChatGptCredentialManager::new()));

/// Shared credential owner used by every ChatGPT subscription-backed adapter.
pub fn shared_credential_manager() -> Arc<ChatGptCredentialManager> {
    SHARED_CREDENTIAL_MANAGER.clone()
}

/// Raw token endpoint response.
#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    /// Absent on some refresh responses; the caller keeps the previous token.
    #[serde(default)]
    refresh_token: Option<String>,
    /// Carries the ChatGPT account id. Absent on refresh responses.
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    expires_in: Option<i64>,
}

/// Decode a JWT payload segment without verifying the signature.
///
/// Verification is deliberately skipped: this is our own token, received
/// directly over TLS from the issuer, and we are reading a claim rather than
/// making a trust decision about a third party's assertion.
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

/// Pull the ChatGPT account id out of an id_token.
///
/// Claim path: `https://api.openai.com/auth` -> `chatgpt_account_id`.
pub fn account_id_from_id_token(id_token: &str) -> Result<String> {
    let claims = decode_jwt_payload(id_token)?;
    claims
        .get(AUTH_CLAIM_NAMESPACE)
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(|id| id.as_str())
        .filter(|id| !id.is_empty())
        .map(|id| id.to_string())
        .ok_or_else(|| {
            anyhow!(
                "id_token has no `{AUTH_CLAIM_NAMESPACE}.chatgpt_account_id` claim — \
                 the account may not have ChatGPT access"
            )
        })
}

/// Generate a PKCE verifier/challenge pair plus a CSRF state value.
pub struct PkceChallenge {
    pub verifier: String,
    pub challenge: String,
    pub state: String,
}

impl PkceChallenge {
    pub fn generate() -> Self {
        use rand::Rng;
        use sha2::Digest;

        let mut rng = rand::thread_rng();
        let bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
        let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&bytes);
        let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(sha2::Sha256::digest(verifier.as_bytes()));
        Self {
            verifier,
            challenge,
            state: uuid::Uuid::new_v4().to_string(),
        }
    }
}

/// Build the URL the user opens to authorize aidaemon.
pub fn build_authorize_url(challenge: &str, state: &str) -> String {
    let mut url = reqwest::Url::parse(AUTHORIZE_URL).expect("AUTHORIZE_URL is a valid URL");
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", &client_id())
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPES)
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state)
        .append_pair("prompt", "login");
    url.to_string()
}

/// Extract `code` from a redirect the user pasted back, validating `state`.
///
/// Accepts a full redirect URL, a bare `?code=...&state=...` query string, or
/// the raw request target the loopback listener saw.
pub fn parse_callback(input: &str, expected_state: &str) -> Result<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("no redirect URL provided"));
    }
    // reqwest::Url needs an absolute URL; graft relative forms onto the redirect base.
    let absolute = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        let query = trimmed.trim_start_matches('?');
        let query = query
            .split_once('?')
            .map(|(_, q)| q)
            .unwrap_or(query.trim_start_matches("/auth/callback"));
        format!("{REDIRECT_URI}?{}", query.trim_start_matches('?'))
    };

    let url = reqwest::Url::parse(&absolute).context("could not parse the redirect URL")?;
    let mut code = None;
    let mut state = None;
    let mut error = None;
    let mut error_description = None;
    for (k, v) in url.query_pairs() {
        match k.as_ref() {
            "code" => code = Some(v.to_string()),
            "state" => state = Some(v.to_string()),
            "error" => error = Some(v.to_string()),
            "error_description" => error_description = Some(v.to_string()),
            _ => {}
        }
    }

    if let Some(err) = error {
        let detail = error_description.unwrap_or_else(|| "no description".to_string());
        return Err(anyhow!("authorization was denied: {err} ({detail})"));
    }
    let state = state.ok_or_else(|| anyhow!("redirect URL has no `state` parameter"))?;
    if state != expected_state {
        return Err(anyhow!(
            "state mismatch — the redirect does not belong to this login attempt"
        ));
    }
    code.ok_or_else(|| anyhow!("redirect URL has no `code` parameter"))
}

fn credentials_from_response(
    resp: TokenResponse,
    previous: Option<&ChatGptCredentials>,
) -> Result<ChatGptCredentials> {
    let refresh_token = resp
        .refresh_token
        .or_else(|| previous.map(|p| p.refresh_token.clone()))
        .ok_or_else(|| anyhow!("token response contained no refresh_token"))?;

    let account_id = match resp.id_token.as_deref() {
        Some(id_token) => account_id_from_id_token(id_token)?,
        None => previous.map(|p| p.account_id.clone()).ok_or_else(|| {
            anyhow!("token response contained no id_token to read account id from")
        })?,
    };

    // Default conservatively when the server omits expires_in: a short window
    // just means we refresh sooner than necessary.
    let expires_in = resp.expires_in.unwrap_or(3600).max(0);
    Ok(ChatGptCredentials {
        access_token: resp.access_token,
        refresh_token,
        account_id,
        expires_at: Utc::now() + chrono::Duration::seconds(expires_in),
    })
}

/// Exchange an authorization code for tokens.
pub async fn exchange_code(
    client: &reqwest::Client,
    code: &str,
    verifier: &str,
) -> Result<ChatGptCredentials> {
    let resp = client
        .post(TOKEN_URL)
        .form(&[
            ("grant_type", "authorization_code"),
            ("client_id", &client_id()),
            ("code", code),
            ("redirect_uri", REDIRECT_URI),
            ("code_verifier", verifier),
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
    credentials_from_response(parsed, None)
}

/// Exchange a refresh token for a fresh access token.
///
/// OpenAI rotates refresh tokens, so the returned credentials must be persisted —
/// continuing to use the old refresh token can invalidate the chain.
pub async fn refresh_credentials(
    client: &reqwest::Client,
    previous: &ChatGptCredentials,
) -> Result<ChatGptCredentials> {
    let resp = client
        .post(TOKEN_URL)
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", &client_id()),
            ("refresh_token", &previous.refresh_token),
            ("scope", SCOPES),
        ])
        .send()
        .await
        .context("token refresh request failed")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!(
            "token refresh returned {status}: {} — sign in again with \
             `aidaemon auth login openai`",
            truncate_for_error(&body)
        ));
    }
    let parsed: TokenResponse =
        serde_json::from_str(&body).context("token refresh returned unexpected JSON")?;
    credentials_from_response(parsed, Some(previous))
}

/// Keep error text short; token endpoints can echo large bodies.
fn truncate_for_error(body: &str) -> String {
    const LIMIT: usize = 300;
    let trimmed = body.trim();
    if trimmed.chars().count() <= LIMIT {
        return trimmed.to_string();
    }
    let head: String = trimmed.chars().take(LIMIT).collect();
    format!("{head}…")
}

/// Persist credentials to the keychain (or `.env` when the keychain is disabled).
pub fn store_credentials(creds: &ChatGptCredentials) -> Result<()> {
    crate::config::store_in_keychain(&key("access_token"), &creds.access_token)?;
    crate::config::store_in_keychain(&key("refresh_token"), &creds.refresh_token)?;
    crate::config::store_in_keychain(&key("account_id"), &creds.account_id)?;
    crate::config::store_in_keychain(&key("expires_at"), &creds.expires_at.to_rfc3339())?;
    Ok(())
}

/// Load stored credentials, if a login has happened on this machine.
pub fn load_credentials() -> Option<ChatGptCredentials> {
    let access_token = crate::config::resolve_from_keychain(&key("access_token")).ok()?;
    let refresh_token = crate::config::resolve_from_keychain(&key("refresh_token")).ok()?;
    let account_id = crate::config::resolve_from_keychain(&key("account_id")).ok()?;
    let expires_at = crate::config::resolve_from_keychain(&key("expires_at"))
        .ok()
        .and_then(|raw| DateTime::parse_from_rfc3339(&raw).ok())
        .map(|dt| dt.with_timezone(&Utc))
        // An unreadable expiry should force a refresh, not a panic or a stale token.
        .unwrap_or_else(|| Utc::now() - chrono::Duration::hours(1));

    if access_token.is_empty() || refresh_token.is_empty() || account_id.is_empty() {
        return None;
    }
    Some(ChatGptCredentials {
        access_token,
        refresh_token,
        account_id,
        expires_at,
    })
}

/// Remove stored credentials and best-effort revoke the refresh token.
pub async fn logout(client: &reqwest::Client) -> Result<()> {
    if let Some(creds) = load_credentials() {
        let _ = client
            .post(REVOKE_URL)
            .form(&[
                ("client_id", client_id().as_str()),
                ("token", creds.refresh_token.as_str()),
                ("token_type_hint", "refresh_token"),
            ])
            .send()
            .await;
    }
    for suffix in ["access_token", "refresh_token", "account_id", "expires_at"] {
        let _ = crate::config::delete_from_keychain(&key(suffix));
    }
    shared_credential_manager().replace_cached(None).await;
    Ok(())
}

/// Whether a ChatGPT subscription login exists on this machine.
pub fn is_connected() -> bool {
    load_credentials().is_some()
}

/// Minimal HTTP response for the loopback callback.
fn callback_page(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

/// Bind the fixed loopback port before sending the user to the browser, so a
/// port conflict surfaces up front instead of after they have signed in.
pub async fn bind_callback_listener() -> Result<TcpListener> {
    TcpListener::bind(("127.0.0.1", CALLBACK_PORT))
        .await
        .with_context(|| {
            format!(
                "could not bind 127.0.0.1:{CALLBACK_PORT} — that exact port is required by \
                 OpenAI's registered redirect URI. Close whatever is using it (another Codex \
                 or aidaemon login), or re-run with --paste."
            )
        })
}

/// Wait for OpenAI to redirect the browser back to the loopback listener.
async fn await_callback(listener: TcpListener, expected_state: &str) -> Result<String> {
    let accept = async {
        loop {
            let (mut socket, _) = listener.accept().await.context("callback accept failed")?;
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            // Request line: "GET /auth/callback?code=...&state=... HTTP/1.1"
            let target = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or_default()
                .to_string();

            // Browsers also request /favicon.ico; ignore anything but the callback.
            if !target.starts_with("/auth/callback") {
                let _ = socket
                    .write_all(callback_page("<p>Waiting for the sign-in redirect…</p>").as_bytes())
                    .await;
                let _ = socket.shutdown().await;
                continue;
            }

            let result = parse_callback(&target, expected_state);
            let page = match &result {
                Ok(_) => callback_page(
                    "<h2>aidaemon is connected to your ChatGPT account.</h2>\
                     <p>You can close this tab and return to the terminal.</p>",
                ),
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

/// Escape text interpolated into the callback page.
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
    /// Listen on the registered loopback port.
    Loopback,
    /// Read the pasted redirect URL from the supplied closure. Used for headless
    /// installs and when the port is unavailable.
    Paste(RedirectReader),
}

/// Run the full login flow and persist the resulting credentials.
///
/// `announce` receives the authorization URL. The URL is printed rather than
/// opened: there is no portable way to launch a browser, and the daemon
/// routinely runs where no browser exists.
pub async fn login(
    client: &reqwest::Client,
    mode: CallbackMode,
    announce: impl FnOnce(&str),
) -> Result<ChatGptCredentials> {
    let pkce = PkceChallenge::generate();
    let url = build_authorize_url(&pkce.challenge, &pkce.state);

    let code = match mode {
        CallbackMode::Loopback => {
            // Bind before announcing so a port conflict cannot strand the user
            // mid-sign-in.
            let listener = bind_callback_listener().await?;
            announce(&url);
            await_callback(listener, &pkce.state).await?
        }
        CallbackMode::Paste(read) => {
            announce(&url);
            let pasted = read(&url)?;
            parse_callback(&pasted, &pkce.state)?
        }
    };

    let creds = exchange_code(client, &code, &pkce.verifier).await?;
    store_credentials(&creds)?;
    shared_credential_manager()
        .replace_cached(Some(creds.clone()))
        .await;
    Ok(creds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build a synthetic, unsigned JWT. No real tokens in fixtures.
    fn fake_jwt(claims: serde_json::Value) -> String {
        let encode = |v: &serde_json::Value| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(v.to_string().as_bytes())
        };
        format!(
            "{}.{}.{}",
            encode(&json!({"alg": "none", "typ": "JWT"})),
            encode(&claims),
            "not-a-real-signature"
        )
    }

    #[test]
    fn extracts_account_id_from_namespaced_claim() {
        let token = fake_jwt(json!({
            "sub": "user-synthetic-1",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-synthetic-1",
                "chatgpt_plan_type": "plus"
            }
        }));
        assert_eq!(
            account_id_from_id_token(&token).unwrap(),
            "acct-synthetic-1"
        );
    }

    #[test]
    fn rejects_id_token_without_chatgpt_account() {
        let token = fake_jwt(json!({
            "sub": "user-synthetic-1",
            "https://api.openai.com/auth": { "user_id": "user-synthetic-1" }
        }));
        let err = account_id_from_id_token(&token).unwrap_err().to_string();
        assert!(
            err.contains("chatgpt_account_id"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_non_jwt() {
        assert!(account_id_from_id_token("not-a-jwt").is_err());
    }

    #[test]
    fn authorize_url_carries_pkce_and_registered_redirect() {
        let url = build_authorize_url("challenge-value", "state-value");
        let parsed = reqwest::Url::parse(&url).unwrap();
        let params: std::collections::HashMap<_, _> = parsed.query_pairs().into_owned().collect();
        assert_eq!(params["response_type"], "code");
        assert_eq!(params["code_challenge"], "challenge-value");
        assert_eq!(params["code_challenge_method"], "S256");
        assert_eq!(params["state"], "state-value");
        assert_eq!(params["redirect_uri"], REDIRECT_URI);
        assert!(params["scope"].contains("offline_access"));
    }

    #[test]
    fn parses_full_redirect_url() {
        let code = parse_callback(
            "http://localhost:1455/auth/callback?code=abc123&state=st-1",
            "st-1",
        )
        .unwrap();
        assert_eq!(code, "abc123");
    }

    #[test]
    fn parses_bare_request_target() {
        let code = parse_callback("/auth/callback?code=abc123&state=st-1", "st-1").unwrap();
        assert_eq!(code, "abc123");
    }

    #[test]
    fn rejects_state_mismatch() {
        let err = parse_callback("/auth/callback?code=abc&state=other", "st-1")
            .unwrap_err()
            .to_string();
        assert!(err.contains("state mismatch"), "unexpected error: {err}");
    }

    #[test]
    fn surfaces_authorization_denial() {
        let err = parse_callback(
            "/auth/callback?error=access_denied&error_description=User+declined&state=st-1",
            "st-1",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("access_denied"), "unexpected error: {err}");
    }

    #[test]
    fn refresh_keeps_previous_refresh_token_and_account_when_omitted() {
        let previous = ChatGptCredentials {
            access_token: "old-access".into(),
            refresh_token: "keep-me".into(),
            account_id: "acct-synthetic-1".into(),
            expires_at: Utc::now(),
        };
        let resp = TokenResponse {
            access_token: "new-access".into(),
            refresh_token: None,
            id_token: None,
            expires_in: Some(3600),
        };
        let out = credentials_from_response(resp, Some(&previous)).unwrap();
        assert_eq!(out.access_token, "new-access");
        assert_eq!(out.refresh_token, "keep-me");
        assert_eq!(out.account_id, "acct-synthetic-1");
        assert!(out.expires_at > Utc::now() + chrono::Duration::minutes(50));
    }

    #[test]
    fn refresh_adopts_rotated_refresh_token() {
        let previous = ChatGptCredentials {
            access_token: "old-access".into(),
            refresh_token: "old-refresh".into(),
            account_id: "acct-synthetic-1".into(),
            expires_at: Utc::now(),
        };
        let resp = TokenResponse {
            access_token: "new-access".into(),
            refresh_token: Some("rotated-refresh".into()),
            id_token: None,
            expires_in: Some(3600),
        };
        let out = credentials_from_response(resp, Some(&previous)).unwrap();
        assert_eq!(out.refresh_token, "rotated-refresh");
    }

    #[test]
    fn needs_refresh_uses_skew() {
        let now = Utc::now();
        let soon = ChatGptCredentials {
            access_token: "a".into(),
            refresh_token: "r".into(),
            account_id: "acct-synthetic-1".into(),
            expires_at: now + chrono::Duration::minutes(1),
        };
        assert!(soon.needs_refresh(now));

        let later = ChatGptCredentials {
            expires_at: now + chrono::Duration::minutes(30),
            ..soon.clone()
        };
        assert!(!later.needs_refresh(now));
    }

    #[test]
    fn pkce_challenge_is_sha256_of_verifier() {
        use sha2::Digest;
        let pkce = PkceChallenge::generate();
        let expected = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(sha2::Sha256::digest(pkce.verifier.as_bytes()));
        assert_eq!(pkce.challenge, expected);
        assert_ne!(pkce.verifier, pkce.challenge);
    }
}
