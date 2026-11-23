//! Authenticating to the daemon broker: bot tokens, signed bootstrap
//! proofs and connect tokens.

use super::*;

#[derive(Clone)]
pub(crate) enum BridgeAuth {
    StaticToken(String),
    BotProof {
        mint_url: String,
        bot_tokens: Vec<String>,
        fallback_static_token: Option<String>,
    },
}

#[derive(Debug, Serialize)]
pub(crate) struct DaemonBootstrapMintRequest<'a> {
    pub(crate) user_id: &'a str,
    pub(crate) device_id: &'a str,
    pub(crate) ts: i64,
    pub(crate) nonce: String,
    pub(crate) sig: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DaemonConnectTokenResponse {
    pub(crate) ok: bool,
    pub(crate) daemon_connect_token: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) message: Option<String>,
}

pub(crate) fn merge_daemon_bot_tokens(
    configured_tokens: &[String],
    dynamic_bots: &[RuntimeBot],
    user_id: &str,
) -> Vec<String> {
    let mut out = Vec::new();
    for token in configured_tokens {
        let trimmed = token.trim();
        if trimmed.is_empty()
            || out
                .iter()
                .any(|existing: &String| existing.as_str() == trimmed)
        {
            continue;
        }
        out.push(trimmed.to_string());
    }

    for bot in dynamic_bots {
        if bot.channel_type != "telegram" {
            continue;
        }
        if !bot.allowed_user_ids.is_empty()
            && !bot
                .allowed_user_ids
                .iter()
                .any(|allowed| allowed.trim() == user_id)
        {
            continue;
        }
        let token = bot.bot_token.trim();
        if token.is_empty()
            || out
                .iter()
                .any(|existing: &String| existing.as_str() == token)
        {
            continue;
        }
        out.push(token.to_string());
    }

    out
}

pub(crate) async fn resolve_runtime_daemon_bot_tokens(
    configured_tokens: &[String],
    host: &dyn BridgeHost,
    user_id: &str,
) -> Vec<String> {
    match host.runtime_bots().await {
        Ok(dynamic_bots) => merge_daemon_bot_tokens(configured_tokens, &dynamic_bots, user_id),
        Err(err) => {
            warn!(
                error = %err,
                "Failed to load dynamic Telegram bots for daemon bootstrap auth"
            );
            configured_tokens.to_vec()
        }
    }
}

pub(crate) fn daemon_connect_token_mint_url(ws_url: &str) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(ws_url)?;
    match url.scheme() {
        "wss" => {
            url.set_scheme("https")
                .map_err(|_| anyhow::anyhow!("failed to convert wss to https"))?;
        }
        other => anyhow::bail!(
            "daemon token mint requires secure websocket origin (expected wss://, found {}://)",
            other
        ),
    }
    url.set_query(None);
    url.set_fragment(None);
    url.set_path("/v1/daemon/connect-token/daemon-auth");
    Ok(url.to_string())
}

pub(crate) async fn resolve_connect_token(
    auth: BridgeAuth,
    host: Arc<dyn BridgeHost>,
    user_id: String,
    device_id: String,
    http_client: reqwest::Client,
) -> anyhow::Result<String> {
    match auth {
        BridgeAuth::StaticToken(token) => Ok(token),
        BridgeAuth::BotProof {
            mint_url,
            bot_tokens,
            fallback_static_token,
        } => {
            let resolved_bot_tokens =
                resolve_runtime_daemon_bot_tokens(&bot_tokens, host.as_ref(), &user_id).await;
            if resolved_bot_tokens.is_empty() {
                if let Some(static_token) = fallback_static_token {
                    warn!(
                        "No configured or dynamic Telegram bot tokens available; falling back to static daemon token"
                    );
                    return Ok(static_token);
                }
                return Err(anyhow::anyhow!(
                    "no Telegram bot token available for daemon bootstrap auth"
                ));
            }
            match mint_connect_token_from_bot_proof(
                &http_client,
                &mint_url,
                &resolved_bot_tokens,
                &user_id,
                &device_id,
            )
            .await
            {
                Ok(token) => Ok(token),
                Err(err) => {
                    if let Some(static_token) = fallback_static_token {
                        warn!(
                            error = %err,
                            "Auto-bootstrap daemon token mint failed; falling back to static daemon token"
                        );
                        Ok(static_token)
                    } else {
                        Err(err)
                    }
                }
            }
        }
    }
}

pub(crate) async fn mint_connect_token_from_bot_proof(
    http_client: &reqwest::Client,
    mint_url: &str,
    bot_tokens: &[String],
    user_id: &str,
    device_id: &str,
) -> anyhow::Result<String> {
    let mut last_error: Option<anyhow::Error> = None;
    let ts = chrono::Utc::now().timestamp();
    let nonce = random_nonce_hex(16);

    for bot_token in bot_tokens {
        let signatures = match daemon_bootstrap_signature_candidates(
            bot_token, user_id, device_id, ts, &nonce,
        ) {
            Ok(v) => v,
            Err(err) => {
                last_error = Some(err);
                continue;
            }
        };

        for (sig_idx, signature) in signatures.into_iter().enumerate() {
            let body = DaemonBootstrapMintRequest {
                user_id,
                device_id,
                ts,
                nonce: nonce.clone(),
                sig: signature,
            };

            let response = match http_client.post(mint_url).json(&body).send().await {
                Ok(resp) => resp,
                Err(err) => {
                    last_error = Some(anyhow::anyhow!(
                        "failed to request daemon connect token: {}",
                        err
                    ));
                    continue;
                }
            };

            let status = response.status();
            let parsed: DaemonConnectTokenResponse = match response.json().await {
                Ok(v) => v,
                Err(err) => {
                    last_error = Some(anyhow::anyhow!(
                        "daemon token endpoint returned invalid JSON (status {}): {}",
                        status,
                        err
                    ));
                    continue;
                }
            };

            if status.is_success() && parsed.ok {
                if let Some(token) = parsed.daemon_connect_token
                    && !token.trim().is_empty()
                {
                    return Ok(token);
                }
                last_error = Some(anyhow::anyhow!(
                    "daemon token endpoint response missing token (status {})",
                    status
                ));
                continue;
            }

            if status.as_u16() == 401 || status.as_u16() == 403 {
                if sig_idx == 0 {
                    debug!(
                        status = status.as_u16(),
                        error = parsed.error.as_deref().unwrap_or("unknown"),
                        "daemon bootstrap primary signature rejected; trying legacy compatibility signature"
                    );
                } else {
                    debug!(
                        status = status.as_u16(),
                        error = parsed.error.as_deref().unwrap_or("unknown"),
                        "daemon bootstrap rejected for one configured telegram bot token"
                    );
                }
                last_error = Some(anyhow::anyhow!(
                    "daemon bootstrap rejected: {}",
                    parsed
                        .message
                        .as_deref()
                        .or(parsed.error.as_deref())
                        .unwrap_or("unauthorized")
                ));
                continue;
            }

            return Err(anyhow::anyhow!(
                "daemon token mint failed (status {}): {}",
                status,
                parsed
                    .message
                    .as_deref()
                    .or(parsed.error.as_deref())
                    .unwrap_or("unknown error")
            ));
        }
    }

    Err(last_error.unwrap_or_else(|| {
        anyhow::anyhow!("no Telegram bot token available for daemon bootstrap auth")
    }))
}
