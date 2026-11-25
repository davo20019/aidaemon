//! A request whose model provider failed before any tool work committed:
//! what the user is told, and the queue that re-runs it once the provider
//! recovers ([`DeferredRetries`], on any [`TurnHost`]).
//!
//! The in-call retry ladder waits out short outages. When it is exhausted and
//! nothing has committed, the request is a pure re-runnable input: persist it
//! as a typed record, tell the user once that it is queued, and let the
//! heartbeat re-run it with a growing delay. A retry that fails the same way
//! re-queues itself with `attempts + 1` through the very same fallback path,
//! so no failure detection depends on reply wording. Attempts are capped;
//! the final failure is reported honestly and the record is dropped.

use aidaemon_channels::context::ChannelContext;
use aidaemon_core::ChannelVisibility;
use aidaemon_tools::semantics::UserRole;
use serde::{Deserialize, Serialize};

use crate::host::TurnHost;

/// Attempts before the request is abandoned (initial failure counts as 0).
pub const MAX_DEFERRED_ATTEMPTS: u16 = 4;

/// What the user is told when their request was queued (`attempt` 0) or
/// re-queued after another failure.
pub fn deferral_notice(attempt: u16, error: &anyhow::Error) -> String {
    let notice = queued_request_notice(attempt, error);
    match error
        .downcast_ref::<crate::chat_bridge::ProviderError>()
        .and_then(crate::chat_bridge::ProviderError::fallback_failures_note)
    {
        Some(note) => format!("{notice}\n\n{note}"),
        None => notice,
    }
}

fn queued_request_notice(attempt: u16, error: &anyhow::Error) -> String {
    let quota = error
        .downcast_ref::<crate::chat_bridge::ProviderError>()
        .filter(|e| e.kind == crate::chat_bridge::ProviderErrorKind::QuotaExhausted);
    if let Some(quota) = quota {
        return match quota.resets_at {
            Some(at) => format!(
                "The model plan's usage limit has been reached; it resets {}. I saved this request and will run it automatically after the reset, then reply here.",
                crate::chat_bridge::describe_reset(at, std::time::SystemTime::now())
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

const SETTING_KEY: &str = "deferred_provider_retries";

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
pub struct DeferredRequest {
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
    /// Where the request was made. A re-run answers in that same place, so
    /// it keeps that place's privacy rules.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<DeferredChannel>,
}

/// The parts of a channel a re-run needs. A workspace grant is not kept: it
/// expires, and the re-run simply goes without one.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeferredChannel {
    visibility: String,
    platform: String,
    channel_name: Option<String>,
    channel_id: Option<String>,
    workspace_id: Option<String>,
    sender_name: Option<String>,
    sender_id: Option<String>,
}

impl DeferredChannel {
    fn of(context: &ChannelContext) -> Self {
        Self {
            visibility: context.visibility.to_string(),
            platform: context.platform.clone(),
            channel_name: context.channel_name.clone(),
            channel_id: context.channel_id.clone(),
            workspace_id: context.workspace_id.clone(),
            sender_name: context.sender_name.clone(),
            sender_id: context.sender_id.clone(),
        }
    }

    fn context(&self) -> ChannelContext {
        let mut context = ChannelContext::internal();
        context.visibility = ChannelVisibility::from_str_lossy(&self.visibility);
        context.platform = self.platform.clone();
        context.channel_name = self.channel_name.clone();
        context.channel_id = self.channel_id.clone();
        context.workspace_id = self.workspace_id.clone();
        context.sender_name = self.sender_name.clone();
        context.sender_id = self.sender_id.clone();
        context
    }
}

impl DeferredRequest {
    /// The channel to re-run in. A record saved before channels were kept
    /// is treated as public, the most restrictive reading.
    fn channel_context(&self) -> ChannelContext {
        self.channel.as_ref().map_or_else(
            || {
                let mut context = ChannelContext::internal();
                context.visibility = ChannelVisibility::Public;
                context
            },
            DeferredChannel::context,
        )
    }

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

    pub fn due(&self, now: chrono::DateTime<chrono::Utc>) -> bool {
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
pub fn is_deferrable_provider_failure(error: &anyhow::Error) -> Option<&'static str> {
    let provider = error.downcast_ref::<crate::chat_bridge::ProviderError>()?;
    let kind = provider.kind;
    let waitable = match kind {
        crate::chat_bridge::ProviderErrorKind::RateLimit
        | crate::chat_bridge::ProviderErrorKind::Timeout
        | crate::chat_bridge::ProviderErrorKind::Network
        | crate::chat_bridge::ProviderErrorKind::ServerError => true,
        crate::chat_bridge::ProviderErrorKind::QuotaExhausted => {
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
    let provider = error.downcast_ref::<crate::chat_bridge::ProviderError>()?;
    let reset = chrono::DateTime::<chrono::Utc>::from(provider.resets_at?);
    Some((reset + chrono::Duration::seconds(QUOTA_RESET_SLACK_SECS)).to_rfc3339())
}

fn kind_label(kind: crate::chat_bridge::ProviderErrorKind) -> &'static str {
    match kind {
        crate::chat_bridge::ProviderErrorKind::QuotaExhausted => "quota_exhausted",
        crate::chat_bridge::ProviderErrorKind::RateLimit => "rate_limit",
        crate::chat_bridge::ProviderErrorKind::Timeout => "timeout",
        crate::chat_bridge::ProviderErrorKind::Network => "network",
        crate::chat_bridge::ProviderErrorKind::ServerError => "server_error",
        _ => "other",
    }
}

async fn load_deferred_requests(
    settings: &dyn aidaemon_tools::services::SettingsStore,
) -> Vec<DeferredRequest> {
    settings
        .get_setting(SETTING_KEY)
        .await
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

async fn store_deferred_requests(
    settings: &dyn aidaemon_tools::services::SettingsStore,
    requests: &[DeferredRequest],
) -> anyhow::Result<()> {
    let raw = serde_json::to_string(requests)?;
    settings.set_setting(SETTING_KEY, &raw).await
}

/// The queue of requests waiting for the model provider to recover, kept in
/// the host's settings store and re-run by the heartbeat.
#[async_trait::async_trait]
pub trait DeferredRetries: TurnHost {
    /// Queue (or re-queue) a request whose provider failed before any tool
    /// work committed, to run again once the provider recovers: the attempt
    /// that will run next, or `None` when the request is not deferrable or
    /// has exhausted its cap.
    async fn defer_request_for_provider_recovery(
        &self,
        session_id: &str,
        user_text: &str,
        user_role: UserRole,
        channel_ctx: &ChannelContext,
        error: &anyhow::Error,
    ) -> Option<u16> {
        let failure_kind = is_deferrable_provider_failure(error)?;
        if user_text.trim().is_empty() {
            return None;
        }
        let mut requests = load_deferred_requests(self.settings()).await;
        let attempts = requests
            .iter()
            .find(|request| request.session_id == session_id && request.user_text == user_text)
            .map(|request| request.attempts.saturating_add(1))
            .unwrap_or(0);
        requests.retain(|request| {
            !(request.session_id == session_id && request.user_text == user_text)
        });
        if attempts >= MAX_DEFERRED_ATTEMPTS {
            let _ = store_deferred_requests(self.settings(), &requests).await;
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
            channel: Some(DeferredChannel::of(channel_ctx)),
        });
        while requests.len() > MAX_QUEUE {
            requests.remove(0);
        }
        match store_deferred_requests(self.settings(), &requests).await {
            Ok(()) => Some(attempts),
            Err(error) => {
                tracing::warn!(%error, "Failed to persist deferred provider retry");
                None
            }
        }
    }
}

impl<H: TurnHost + ?Sized> DeferredRetries for H {}

impl crate::engine::turn_engine::TurnEngine<'_> {
    /// Heartbeat entry: re-run every due deferred request; how many ran. A
    /// record is removed before dispatch so a crash cannot replay it twice; a
    /// repeated provider failure re-queues it through the normal fallback
    /// with `attempts + 1`.
    pub async fn retry_deferred_provider_requests(&self) -> usize {
        let requests = load_deferred_requests(self.settings()).await;
        if requests.is_empty() {
            return 0;
        }
        let now = chrono::Utc::now();
        let (due, waiting): (Vec<_>, Vec<_>) =
            requests.into_iter().partition(|request| request.due(now));
        if due.is_empty() {
            return 0;
        }
        if let Err(error) = store_deferred_requests(self.settings(), &waiting).await {
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
            let rerun = self
                .handle_message_impl(
                    &request.session_id,
                    &request.user_text,
                    &[],
                    None,
                    request.role(),
                    request.channel_context(),
                    None,
                    None,
                    true,
                    None,
                    None,
                    None,
                )
                .await;
            match rerun {
                Ok(reply) => {
                    dispatched += 1;
                    let reply = aidaemon_tools::sanitize::sanitize_user_facing_reply(&reply);
                    if reply.trim().is_empty() {
                        continue;
                    }
                    let hub = self.hub().read().await.clone();
                    if let Some(hub) = hub.and_then(|weak| weak.upgrade())
                        && let Err(error) = hub.send_text(&request.session_id, &reply).await
                    {
                        tracing::warn!(%error, session_id = %request.session_id, "Failed to deliver deferred retry reply");
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
        let mut error = crate::chat_bridge::ProviderError::from_status(429, "plan cap");
        error.kind = crate::chat_bridge::ProviderErrorKind::QuotaExhausted;
        error.resets_at = resets_in.map(|left| std::time::SystemTime::now() + left);
        anyhow::Error::new(error)
    }

    #[test]
    fn the_notice_names_why_each_other_model_failed() {
        use crate::chat_bridge::{ModelFailure, ProviderError, ProviderErrorKind};
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
            channel: None,
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
    fn a_re_run_keeps_the_channel_it_came_from() {
        let mut group = ChannelContext::internal();
        group.visibility = ChannelVisibility::PublicExternal;
        group.platform = "slack".to_string();
        group.channel_id = Some("slack:synthetic-room".to_string());
        group.sender_id = Some("slack:synthetic-user-1".to_string());

        let stored = serde_json::to_string(&DeferredChannel::of(&group)).unwrap();
        let context = serde_json::from_str::<DeferredChannel>(&stored)
            .unwrap()
            .context();

        assert_eq!(context.visibility, ChannelVisibility::PublicExternal);
        assert_eq!(context.channel_id, group.channel_id);
        assert_eq!(context.sender_id, group.sender_id);
    }

    #[test]
    fn a_record_without_a_channel_is_treated_as_public() {
        let request: DeferredRequest = serde_json::from_value(serde_json::json!({
            "session_id": "synthetic",
            "user_text": "hello",
            "user_role": "owner",
            "failed_at": "2026-01-01T00:00:00Z",
            "attempts": 0,
            "failure_kind": "outage",
        }))
        .unwrap();
        assert_eq!(
            request.channel_context().visibility,
            ChannelVisibility::Public
        );
    }

    #[test]
    fn only_waitable_provider_failures_are_deferrable() {
        let server = anyhow::Error::from(crate::chat_bridge::ProviderError::from_status(
            503,
            "synthetic 503",
        ));
        assert_eq!(
            is_deferrable_provider_failure(&server),
            Some("server_error")
        );
        let auth = anyhow::Error::from(crate::chat_bridge::ProviderError::from_status(
            401,
            "synthetic 401",
        ));
        assert_eq!(is_deferrable_provider_failure(&auth), None);
        let plain = anyhow::anyhow!("not a provider error");
        assert_eq!(is_deferrable_provider_failure(&plain), None);
    }
}
