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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct DeferredRequest {
    pub session_id: String,
    pub user_text: String,
    pub user_role: String,
    pub failed_at: String,
    pub attempts: u16,
    pub failure_kind: String,
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
/// Authentication and billing failures need the owner, not time.
pub(crate) fn is_deferrable_provider_failure(error: &anyhow::Error) -> Option<&'static str> {
    let provider = error.downcast_ref::<crate::providers::ProviderError>()?;
    let kind = provider.kind;
    let waitable = matches!(
        kind,
        crate::providers::ProviderErrorKind::RateLimit
            | crate::providers::ProviderErrorKind::Timeout
            | crate::providers::ProviderErrorKind::Network
            | crate::providers::ProviderErrorKind::ServerError
    );
    waitable.then(|| kind_label(kind))
}

fn kind_label(kind: crate::providers::ProviderErrorKind) -> &'static str {
    match kind {
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
