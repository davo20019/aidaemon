//! Terminal-tenant bootstrap for the Telegram Mini App: HMAC-SHA256 proofs,
//! nonces and the worker URL used to sync a bot token to the terminal
//! tenant, and how the worker's answer is read.

use base64::Engine;
use hmac::{Hmac, KeyInit, Mac};
use rand::Rng;
use rand::rand_core::UnwrapErr;
use rand::rngs::SysRng;
use sha2::Sha256;

const TERMINAL_TENANT_BOT_BOOTSTRAP_DEVICE_ID: &str = "tenant-bot-bootstrap";

type HmacSha256 = Hmac<Sha256>;

pub(super) fn random_terminal_bootstrap_nonce(num_bytes: usize) -> String {
    let mut bytes = vec![0u8; num_bytes.max(1)];
    UnwrapErr(SysRng).fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn terminal_tenant_bot_bootstrap_signing_input(user_id: u64, ts: i64, nonce: &str) -> String {
    format!(
        "v1\nuser_id={}\ndevice_id={}\nts={}\nnonce={}",
        user_id, TERMINAL_TENANT_BOT_BOOTSTRAP_DEVICE_ID, ts, nonce
    )
}

pub(super) fn sign_terminal_tenant_bot_bootstrap_proof(
    bot_token: &str,
    user_id: u64,
    ts: i64,
    nonce: &str,
) -> Result<String, String> {
    let input = terminal_tenant_bot_bootstrap_signing_input(user_id, ts, nonce);
    let mut mac = <HmacSha256 as KeyInit>::new_from_slice(bot_token.as_bytes())
        .map_err(|_| "invalid HMAC key".to_string())?;
    mac.update(input.as_bytes());
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes()))
}

pub(super) fn terminal_tenant_bot_bootstrap_url(
    terminal_web_app_url: &str,
) -> Result<reqwest::Url, String> {
    let mut url = reqwest::Url::parse(terminal_web_app_url.trim())
        .map_err(|err| format!("invalid terminal web app URL: {}", err))?;
    if url.scheme() != "https" {
        return Err("terminal web app URL must use HTTPS".to_string());
    }
    url.set_query(None);
    url.set_fragment(None);
    url.set_path("/v1/tenant/bot-token/bootstrap");
    Ok(url)
}

/// Whether the worker accepted a bot-token sync, and whether a failure is
/// worth retrying.
pub(super) fn interpret_terminal_bot_sync_response(
    status: reqwest::StatusCode,
    content_type: Option<&str>,
    body: &[u8],
) -> (Result<(), String>, bool) {
    let retryable_status =
        status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
    let parsed = match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(parsed) => parsed,
        Err(error) => {
            let content_type = content_type.unwrap_or("unknown");
            return (
                Err(format!(
                    "terminal bot sync returned non-JSON response (status {}, content-type {}, {} bytes): {}",
                    status,
                    content_type,
                    body.len(),
                    error
                )),
                retryable_status || status.is_success(),
            );
        }
    };
    if status.is_success() && parsed.get("ok").and_then(|value| value.as_bool()) == Some(true) {
        return (Ok(()), false);
    }

    let message = parsed
        .get("message")
        .and_then(|value| value.as_str())
        .or_else(|| parsed.get("error").and_then(|value| value.as_str()))
        .unwrap_or("unknown error");
    (
        Err(format!(
            "terminal bot sync failed (status {}): {}",
            status,
            aidaemon_tools::sanitize::redact_secrets(message)
        )),
        retryable_status,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_tenant_bot_bootstrap_url_targets_worker_endpoint() {
        let url = terminal_tenant_bot_bootstrap_url(
            "https://terminal.aidaemon.ai/app?tgWebAppData=abc#fragment",
        )
        .unwrap();
        assert_eq!(
            url.as_str(),
            "https://terminal.aidaemon.ai/v1/tenant/bot-token/bootstrap"
        );
    }

    #[test]
    fn terminal_tenant_bot_bootstrap_signature_is_stable() {
        let sig = sign_terminal_tenant_bot_bootstrap_proof(
            "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcd",
            301753035,
            1_700_000_000,
            "deadbeefcafebabe",
        )
        .unwrap();
        assert_eq!(sig, "wdb3Oj1hWbvz373tj4nBZrudZKP_nFsmf8LZvWMwvOo");
    }

    #[test]
    fn terminal_bot_sync_response_accepts_success_json() {
        let (result, retryable) = interpret_terminal_bot_sync_response(
            reqwest::StatusCode::OK,
            Some("application/json"),
            br#"{"ok":true}"#,
        );
        assert!(result.is_ok());
        assert!(!retryable);
    }

    #[test]
    fn terminal_bot_sync_response_classifies_transient_non_json() {
        let (result, retryable) = interpret_terminal_bot_sync_response(
            reqwest::StatusCode::BAD_GATEWAY,
            Some("text/html"),
            b"upstream unavailable",
        );
        let error = result.unwrap_err();
        assert!(error.contains("status 502 Bad Gateway"));
        assert!(error.contains("content-type text/html"));
        assert!(retryable);
    }

    #[test]
    fn terminal_bot_sync_response_does_not_retry_auth_rejection() {
        let (result, retryable) = interpret_terminal_bot_sync_response(
            reqwest::StatusCode::UNAUTHORIZED,
            Some("application/json"),
            br#"{"ok":false,"error":"invalid proof"}"#,
        );
        assert!(result.unwrap_err().contains("invalid proof"));
        assert!(!retryable);
    }
}
