//! Whether a coding agent's account is out of usage, and until when: asked
//! of the agent itself, after a run fails, so a run that stopped on its
//! plan's limit says so instead of reading as a crash.
//!
//! Only the agent's typed answer counts, never the words of its error
//! message.

use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// How to ask an agent about its usage ([`crate::CodingAgent::usage_check`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageCheck {
    /// Codex's app server: `account/rateLimits/read`.
    Codex,
}

/// The agent's account may not do ordinary work until `resets_at_ms`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsageLimit {
    /// When the window that ran out resets, ms since the epoch; `None`:
    /// the agent did not say.
    pub resets_at_ms: Option<u64>,
}

/// The longest an agent is given to answer.
const ANSWER_WITHIN: Duration = Duration::from_secs(20);

/// Ask the agent launched as `command` (after a launcher's `prefix`)
/// whether its account is out of usage. `None`: it is not, or it could not
/// be asked, or did not say.
pub async fn usage_limit(
    check: UsageCheck,
    command: &str,
    prefix: &[String],
) -> Option<UsageLimit> {
    tokio::time::timeout(ANSWER_WITHIN, ask(check, command, prefix))
        .await
        .ok()
        .flatten()
}

async fn ask(check: UsageCheck, command: &str, prefix: &[String]) -> Option<UsageLimit> {
    let UsageCheck::Codex = check;
    let mut child = tokio::process::Command::new(command)
        .args(prefix)
        .arg("app-server")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let stdout = child.stdout.take()?;
    for request in [
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "clientInfo": {"name": "aidaemon", "version": env!("CARGO_PKG_VERSION")},
        }}),
        json!({"jsonrpc": "2.0", "method": "initialized"}),
        // The counts alone: the reset-credit details are a second lookup.
        json!({"jsonrpc": "2.0", "id": 2, "method": "account/rateLimits/read",
            "params": {"excludeResetCreditDetails": true}}),
    ] {
        let mut line = request.to_string();
        line.push('\n');
        stdin.write_all(line.as_bytes()).await.ok()?;
    }
    let mut lines = BufReader::new(stdout).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if message.get("id").and_then(Value::as_u64) == Some(2) {
            return codex_limit(message.get("result")?);
        }
    }
    None
}

/// Codex's `account/rateLimits/read` answer: out of usage when it says
/// ordinary usage is not allowed or names the limit reached; it resets when
/// the latest full window does (`resetsAt`, seconds).
pub(crate) fn codex_limit(answer: &Value) -> Option<UsageLimit> {
    let snapshot = answer.get("rateLimits")?;
    let refused = answer.get("ordinaryUsageAllowed").and_then(Value::as_bool) == Some(false)
        || snapshot
            .get("rateLimitReachedType")
            .is_some_and(|reached| !reached.is_null());
    if !refused {
        return None;
    }
    let resets_at_ms = ["primary", "secondary"]
        .iter()
        .filter_map(|window| snapshot.get(*window))
        .filter(|window| window.get("usedPercent").and_then(Value::as_i64) >= Some(100))
        .filter_map(|window| window.get("resetsAt").and_then(Value::as_u64))
        .max()
        .map(|seconds| seconds.saturating_mul(1000));
    Some(UsageLimit { resets_at_ms })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The answer Codex 0.170.0 gave on a Plus plan whose 5-hour window ran
    /// out (live, 2026-11-26), trimmed.
    fn limited() -> Value {
        json!({
            "ordinaryUsageAllowed": false,
            "rateLimits": {
                "limitId": "codex",
                "primary": {"usedPercent": 100, "windowDurationMins": 300, "resetsAt": 1_795_666_077},
                "secondary": {"usedPercent": 43, "windowDurationMins": 10080, "resetsAt": 1_796_047_161},
                "credits": {"hasCredits": false, "unlimited": false, "balance": "0"},
                "planType": "plus",
                "rateLimitReachedType": "rate_limit_reached"
            }
        })
    }

    #[test]
    fn a_full_window_is_out_of_usage_until_it_resets() {
        assert_eq!(
            codex_limit(&limited()),
            Some(UsageLimit {
                resets_at_ms: Some(1_795_666_077_000)
            })
        );
    }

    #[test]
    fn usage_left_is_not_a_limit() {
        let mut fine = limited();
        fine["ordinaryUsageAllowed"] = json!(true);
        fine["rateLimits"]["rateLimitReachedType"] = Value::Null;
        fine["rateLimits"]["primary"]["usedPercent"] = json!(62);
        assert_eq!(codex_limit(&fine), None);
    }

    #[test]
    fn a_refusal_with_no_full_window_has_no_reset_time() {
        let mut refused = limited();
        refused["rateLimits"]["primary"]["usedPercent"] = json!(80);
        assert_eq!(
            codex_limit(&refused),
            Some(UsageLimit { resets_at_ms: None })
        );
    }
}

/// Asks the installed Codex for real: run by hand (`--ignored`) while its
/// plan is out of usage to see the typed answer read end to end.
#[cfg(test)]
mod live {
    #[tokio::test]
    #[ignore = "asks the installed Codex's account"]
    async fn the_installed_codex_says_whether_it_is_out_of_usage() {
        let limit = super::usage_limit(super::UsageCheck::Codex, "codex", &[]).await;
        println!("{limit:?}");
    }
}
