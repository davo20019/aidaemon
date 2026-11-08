//! Deferred re-run of a user request that failed only because the model
//! provider was unavailable before any tool work committed.
//!
//! The in-call retry ladder waits out short outages. When it is exhausted and
//! nothing has committed, the request is a pure re-runnable input: persist it
//! as a typed record, tell the user once that it is queued, and let the
//! heartbeat re-run it with a growing delay. A retry that fails the same way
//! re-queues itself with `attempts + 1` through the very same fallback path,
//! so no failure detection depends on reply wording. Attempts are capped;
//! the final failure is reported honestly and the record is dropped.

use serde::{Deserialize, Serialize};

use crate::types::UserRole;

const SETTING_KEY: &str = "deferred_provider_retries";
/// Attempts before the request is abandoned (initial failure counts as 0).
pub(crate) const MAX_DEFERRED_ATTEMPTS: u16 = 4;
/// Delay before the first automatic re-run; each later attempt triples it
/// (2 min, 6 min, 18 min, 54 min ≈ 80 minutes of coverage).
const BASE_DELAY_SECS: i64 = 120;
/// Never queue more than this many distinct requests; the oldest is dropped.
const MAX_QUEUE: usize = 20;
/// A used-up quota is waited out only when it resets within this long; a
/// request re-run days later would answer a question nobody is waiting on.
const MAX_QUOTA_WAIT: std::time::Duration = std::time::Duration::from_secs(12 * 60 * 60);
/// Run a quota-deferred request this long after the stated reset.
const QUOTA_RESET_SLACK_SECS: i64 = 60;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct DeferredRequest {
    pub session_id: String,
    pub user_text: String,
    pub user_role: String,
    pub failed_at: String,
    pub attempts: u16,
    pub failure_kind: String,
    /// The earliest re-run, when the failure stated it (a quota's reset);
    /// otherwise the growing delay applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_before: Option<String>,
}

impl DeferredRequest {
    fn role(&self) -> UserRole {
        match self.user_role.as_str() {
            "owner" => UserRole::Owner,
            "guest" => UserRole::Guest,
            _ => UserRole::Public,
        }
    }

    fn delay_secs(&self) -> i64 {
        BASE_DELAY_SECS.saturating_mul(3_i64.saturating_pow(u32::from(self.attempts)))
    }

    pub(crate) fn due(&self, now: chrono::DateTime<chrono::Utc>) -> bool {
        if let Some(not_before) = self
            .not_before
            .as_deref()
            .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok())
        {
            return now >= not_before.with_timezone(&chrono::Utc);
        }
        chrono::DateTime::parse_from_rfc3339(&self.failed_at)
            .map(|failed_at| {
                (now - failed_at.with_timezone(&chrono::Utc)).num_seconds() >= self.delay_secs()
            })
            .unwrap_or(true)
    }
}

fn role_label(role: UserRole) -> &'static str {
    match role {
        UserRole::Owner => "owner",
        UserRole::Guest => "guest",
        UserRole::Public => "public",
    }
}

/// Whether an error is a provider infrastructure failure worth waiting out.
/// Authentication and billing failures need the owner, not time; a used-up
/// quota is waited out when it resets soon enough (or does not say when).
pub(crate) fn is_deferrable_provider_failure(error: &anyhow::Error) -> Option<&'static str> {
    let provider = error.downcast_ref::<crate::providers::ProviderError>()?;
    let kind = provider.kind;
    let waitable = match kind {
        crate::providers::ProviderErrorKind::RateLimit
        | crate::providers::ProviderErrorKind::Timeout
        | crate::providers::ProviderErrorKind::Network
        | crate::providers::ProviderErrorKind::ServerError => true,
        crate::providers::ProviderErrorKind::QuotaExhausted => {
            provider.resets_at.is_none_or(|at| {
                at.duration_since(std::time::SystemTime::now())
                    .map_or(true, |left| left <= MAX_QUOTA_WAIT)
            })
        }
        _ => false,
    };
    waitable.then(|| kind_label(kind))
}

/// When a deferred re-run may start, if the failure says: just after a
/// used-up quota's reset.
fn not_before(error: &anyhow::Error) -> Option<String> {
    let provider = error.downcast_ref::<crate::providers::ProviderError>()?;
    let reset = chrono::DateTime::<chrono::Utc>::from(provider.resets_at?);
    Some((reset + chrono::Duration::seconds(QUOTA_RESET_SLACK_SECS)).to_rfc3339())
}

/// What the user is told when their request was queued (`attempt` 0) or
/// re-queued after another failure.
pub(crate) fn deferral_notice(attempt: u16, error: &anyhow::Error) -> String {
    let notice = queued_request_notice(attempt, error);
    match error
        .downcast_ref::<crate::providers::ProviderError>()
        .and_then(crate::providers::ProviderError::fallback_failures_note)
    {
        Some(note) => format!("{notice}\n\n{note}"),
        None => notice,
    }
}

fn queued_request_notice(attempt: u16, error: &anyhow::Error) -> String {
    let quota = error
        .downcast_ref::<crate::providers::ProviderError>()
        .filter(|e| e.kind == crate::providers::ProviderErrorKind::QuotaExhausted);
    if let Some(quota) = quota {
        return match quota.resets_at {
            Some(at) => format!(
                "The model plan's usage limit has been reached; it resets {}. I saved this request and will run it automatically after the reset, then reply here.",
                crate::providers::describe_reset(at, std::time::SystemTime::now())
            ),
            None => "The model plan's usage limit has been reached. I saved this request and will retry it automatically, then reply here.".to_string(),
        };
    }
    match attempt {
        0 => "The model provider is unavailable right now. I saved this request and will retry it automatically once the provider recovers, then reply here.".to_string(),
        attempt => format!(
            "The model provider is still unavailable (automatic retry {attempt} of {MAX_DEFERRED_ATTEMPTS}). I will keep retrying with a longer wait and reply here when it succeeds."
        ),
    }
}

fn kind_label(kind: crate::providers::ProviderErrorKind) -> &'static str {
    match kind {
        crate::providers::ProviderErrorKind::QuotaExhausted => "quota_exhausted",
        crate::providers::ProviderErrorKind::RateLimit => "rate_limit",
        crate::providers::ProviderErrorKind::Timeout => "timeout",
        crate::providers::ProviderErrorKind::Network => "network",
        crate::providers::ProviderErrorKind::ServerError => "server_error",
        _ => "other",
    }
}

impl crate::agent::Agent {
    async fn load_deferred_requests(&self) -> Vec<DeferredRequest> {
        self.state
            .get_setting(SETTING_KEY)
            .await
            .ok()
            .flatten()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default()
    }

    async fn store_deferred_requests(&self, requests: &[DeferredRequest]) -> anyhow::Result<()> {
        let raw = serde_json::to_string(requests)?;
        self.state.set_setting(SETTING_KEY, &raw).await
    }

    /// Queue (or re-queue) a request whose provider failed before any tool
    /// work committed. Returns the attempt number that will run next, or
    /// `None` when the request is not deferrable or has exhausted its cap.
    pub(crate) async fn defer_request_for_provider_recovery(
        &self,
        session_id: &str,
        user_text: &str,
        user_role: UserRole,
        error: &anyhow::Error,
    ) -> Option<u16> {
        let failure_kind = is_deferrable_provider_failure(error)?;
        if user_text.trim().is_empty() {
            return None;
        }
        let mut requests = self.load_deferred_requests().await;
        let attempts = requests
            .iter()
            .find(|request| request.session_id == session_id && request.user_text == user_text)
            .map(|request| request.attempts.saturating_add(1))
            .unwrap_or(0);
        requests.retain(|request| {
            !(request.session_id == session_id && request.user_text == user_text)
        });
        if attempts >= MAX_DEFERRED_ATTEMPTS {
            let _ = self.store_deferred_requests(&requests).await;
            return None;
        }
        requests.push(DeferredRequest {
            session_id: session_id.to_string(),
            user_text: user_text.to_string(),
            user_role: role_label(user_role).to_string(),
            failed_at: chrono::Utc::now().to_rfc3339(),
            attempts,
            failure_kind: failure_kind.to_string(),
            not_before: not_before(error),
        });
        while requests.len() > MAX_QUEUE {
            requests.remove(0);
        }
        match self.store_deferred_requests(&requests).await {
            Ok(()) => Some(attempts),
            Err(error) => {
                tracing::warn!(%error, "Failed to persist deferred provider retry");
                None
            }
        }
    }

    /// Heartbeat entry: re-run every due deferred request. A record is removed
    /// before dispatch so a crash cannot replay it twice; a repeated provider
    /// failure re-queues it through the normal fallback with `attempts + 1`.
    pub(crate) async fn retry_deferred_provider_requests(&self) -> usize {
        let requests = self.load_deferred_requests().await;
        if requests.is_empty() {
            return 0;
        }
        let now = chrono::Utc::now();
        let (due, waiting): (Vec<_>, Vec<_>) =
            requests.into_iter().partition(|request| request.due(now));
        if due.is_empty() {
            return 0;
        }
        if let Err(error) = self.store_deferred_requests(&waiting).await {
            tracing::warn!(%error, "Failed to dequeue deferred provider retries");
            return 0;
        }
        let mut dispatched = 0;
        for request in due {
            tracing::info!(
                session_id = %request.session_id,
                attempt = request.attempts + 1,
                failure_kind = %request.failure_kind,
                "Re-running request deferred by a provider outage"
            );
            let conversation = crate::runtime_ports::ConversationRequest {
                session_id: request.session_id.clone(),
                user_text: request.user_text.clone(),
                status_tx: None,
                user_role: request.role(),
                channel_ctx: crate::types::ChannelContext::internal(),
                heartbeat: None,
                parent_task_id: None,
                parent_tool_call_id: None,
                parent_result_id: None,
            };
            match self.handle_internal_continuation(&conversation).await {
                Ok(reply) => {
                    dispatched += 1;
                    let reply = crate::tools::sanitize::sanitize_user_facing_reply(&reply);
                    if reply.trim().is_empty() {
                        continue;
                    }
                    let hub = self.hub.read().await.clone();
                    if let Some(hub) = hub.and_then(|weak| weak.upgrade()) {
                        if let Err(error) = hub.send_text(&request.session_id, &reply).await {
                            tracing::warn!(%error, session_id = %request.session_id, "Failed to deliver deferred retry reply");
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, session_id = %request.session_id, "Deferred provider retry errored");
                }
            }
        }
        dispatched
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quota(resets_in: Option<std::time::Duration>) -> anyhow::Error {
        let mut error = crate::providers::ProviderError::from_status(429, "plan cap");
        error.kind = crate::providers::ProviderErrorKind::QuotaExhausted;
        error.resets_at = resets_in.map(|left| std::time::SystemTime::now() + left);
        anyhow::Error::new(error)
    }

    #[test]
    fn the_notice_names_why_each_other_model_failed() {
        use crate::providers::{ModelFailure, ProviderError, ProviderErrorKind};
        let failure = |model: &str, kind, status| ModelFailure {
            model: model.to_string(),
            kind,
            status: Some(status),
        };
        let mut error = ProviderError::from_status(429, "plan cap");
        error.kind = ProviderErrorKind::QuotaExhausted;
        error.fallback_failures = vec![
            failure("synthetic-haiku", ProviderErrorKind::Auth, 401),
            failure("synthetic-sonnet", ProviderErrorKind::Auth, 401),
            failure("synthetic-kimi", ProviderErrorKind::Billing, 402),
        ];
        // The chain's error reaches the notice wrapped in its reply context.
        let notice = deferral_notice(0, &anyhow::Error::new(error).context("reply text"));
        assert!(notice.contains("usage limit"), "{notice}");
        assert!(
            notice.contains("- synthetic-haiku, synthetic-sonnet: API key rejected (401)"),
            "{notice}"
        );
        assert!(
            notice.contains("- synthetic-kimi: out of credits (402)"),
            "{notice}"
        );
        assert!(notice.contains("/setkey"), "{notice}");

        assert!(!deferral_notice(0, &quota(None)).contains("Other models"));
    }

    #[test]
    fn a_quota_is_waited_out_only_when_it_resets_soon() {
        let hour = std::time::Duration::from_secs(3600);
        assert_eq!(
            is_deferrable_provider_failure(&quota(Some(2 * hour))),
            Some("quota_exhausted")
        );
        assert_eq!(
            is_deferrable_provider_failure(&quota(None)),
            Some("quota_exhausted")
        );
        assert_eq!(
            is_deferrable_provider_failure(&quota(Some(72 * hour))),
            None
        );
        assert!(not_before(&quota(Some(hour))).is_some());
        assert!(not_before(&quota(None)).is_none());
        assert!(deferral_notice(0, &quota(Some(hour))).contains("after the reset"));
        assert!(deferral_notice(0, &quota(None)).contains("usage limit"));
    }

    #[test]
    fn deferred_delay_grows_geometrically_and_gates_due() {
        let base = chrono::Utc::now();
        let mut request = DeferredRequest {
            session_id: "telegram:synthetic-user-1".to_string(),
            user_text: "what is the package version".to_string(),
            user_role: "owner".to_string(),
            failed_at: base.to_rfc3339(),
            attempts: 0,
            failure_kind: "server_error".to_string(),
            not_before: None,
        };
        assert_eq!(request.delay_secs(), 120);
        assert!(!request.due(base + chrono::Duration::seconds(60)));
        assert!(request.due(base + chrono::Duration::seconds(120)));
        request.attempts = 2;
        assert_eq!(request.delay_secs(), 1080);
        assert!(!request.due(base + chrono::Duration::seconds(1000)));
        assert!(request.due(base + chrono::Duration::seconds(1080)));
        assert_eq!(request.role(), UserRole::Owner);
    }

    #[test]
    fn only_waitable_provider_failures_are_deferrable() {
        let server = anyhow::Error::from(crate::providers::ProviderError::from_status(
            503,
            "synthetic 503",
        ));
        assert_eq!(
            is_deferrable_provider_failure(&server),
            Some("server_error")
        );
        let auth = anyhow::Error::from(crate::providers::ProviderError::from_status(
            401,
            "synthetic 401",
        ));
        assert_eq!(is_deferrable_provider_failure(&auth), None);
        let plain = anyhow::anyhow!("not a provider error");
        assert_eq!(is_deferrable_provider_failure(&plain), None);
    }
}
