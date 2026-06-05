//! Terminal-tenant bootstrap signing for the Telegram channel.
//!
//! Provides HMAC-SHA256 signing, nonce generation, and web-app URL
//! construction used when bootstrapping a terminal tenant bot token.
//! Extracted verbatim from `telegram.rs` (behavior-preserving move).

use base64::Engine;
use hmac::{Hmac, Mac};
use rand::rngs::OsRng;
use rand::RngCore;
use sha2::Sha256;

pub(super) const TERMINAL_TENANT_BOT_BOOTSTRAP_DEVICE_ID: &str = "tenant-bot-bootstrap";

type HmacSha256 = Hmac<Sha256>;

pub(super) fn random_terminal_bootstrap_nonce(num_bytes: usize) -> String {
    let mut bytes = vec![0u8; num_bytes.max(1)];
    OsRng.fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

pub(super) fn terminal_tenant_bot_bootstrap_signing_input(
    user_id: u64,
    ts: i64,
    nonce: &str,
) -> String {
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
    let mut mac = <HmacSha256 as Mac>::new_from_slice(bot_token.as_bytes())
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
}
