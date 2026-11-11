//! [`QuotaGuard`]: remembers that an account cannot answer for a while.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

use crate::provider::{ModelRequest, ModelResponse, Provider, ProviderError, StreamEvent};

/// How long a used-up quota is remembered when the provider does not say
/// when it resets.
const QUOTA_UNKNOWN_RESET: Duration = Duration::from_secs(15 * 60);
/// Longest a stated reset is trusted (a weekly plan window, with room).
const QUOTA_MAX_COOLDOWN: Duration = Duration::from_secs(8 * 24 * 60 * 60);
/// How long an account that cannot pay is remembered (credits may be added).
const BILLING_COOLDOWN: Duration = Duration::from_secs(10 * 60);
/// How long a rejected credential is remembered: long enough that the other
/// models of one call are not each sent, short enough to notice a fix.
const AUTH_COOLDOWN: Duration = Duration::from_secs(60);

/// An account (or one model of it) that will not answer until `until`.
#[derive(Debug, Clone, PartialEq)]
pub struct Cooldown {
    /// The model the cooldown covers, or `None` for the whole account.
    pub model: Option<String>,
    pub until: SystemTime,
    /// The error that started it, returned for every call it blocks.
    pub error: ProviderError,
}

/// A provider wrapper for one account. After a failure that no other model
/// of the account can avoid (a used-up quota, an account that cannot pay, a
/// rejected credential) it answers every call with that error, without
/// sending it, until the cooldown ends: until the quota's stated reset, or
/// for a fixed time when there is none.
///
/// Wrap each configured account once and share the guard (it is cheap to
/// clone through an `Arc`): every caller then learns from the first
/// failure, and a [`crate::Failover`] chain moves past the account's other
/// models at once instead of retrying each.
pub struct QuotaGuard {
    inner: Arc<dyn Provider>,
    cooldowns: Mutex<HashMap<Option<String>, Cooldown>>,
}

impl QuotaGuard {
    pub fn new(inner: impl Provider + 'static) -> Self {
        Self::from_arc(Arc::new(inner))
    }

    pub fn from_arc(inner: Arc<dyn Provider>) -> Self {
        Self {
            inner,
            cooldowns: Mutex::new(HashMap::new()),
        }
    }

    /// The cooldowns in force now.
    pub fn active(&self) -> Vec<Cooldown> {
        let now = SystemTime::now();
        let mut cooldowns = self.lock();
        cooldowns.retain(|_, cooldown| cooldown.until > now);
        cooldowns.values().cloned().collect()
    }

    /// Forget every cooldown (say, after the owner changed the credential).
    pub fn clear(&self) {
        self.lock().clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Option<String>, Cooldown>> {
        self.cooldowns
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The error blocking `model` now, if any.
    fn blocked(&self, model: &str) -> Option<ProviderError> {
        let now = SystemTime::now();
        let mut cooldowns = self.lock();
        cooldowns.retain(|_, cooldown| cooldown.until > now);
        cooldowns
            .get(&None)
            .or_else(|| cooldowns.get(&Some(model.to_string())))
            .map(|cooldown| cooldown.error.clone())
    }

    /// Start a cooldown if `error` says waiting is the only cure.
    fn record(&self, model: &str, error: &ProviderError) {
        let now = SystemTime::now();
        let (key, lasts) = match error {
            ProviderError::QuotaExhausted {
                resets_at,
                per_model,
                ..
            } => {
                let lasts = match resets_at {
                    Some(at) => match at.duration_since(now) {
                        Ok(left) => left.min(QUOTA_MAX_COOLDOWN),
                        // Already past: the clock or the provider is off.
                        // Remember it briefly rather than not at all.
                        Err(_) => AUTH_COOLDOWN,
                    },
                    None => QUOTA_UNKNOWN_RESET,
                };
                (per_model.then(|| model.to_string()), lasts)
            }
            // A stated affordable output means a smaller request may pass.
            ProviderError::Billing {
                affordable_tokens: None,
                ..
            } => (None, BILLING_COOLDOWN),
            ProviderError::Auth(_) => (None, AUTH_COOLDOWN),
            _ => return,
        };
        // Every call the cooldown blocks is refused here, before it is
        // sent, whatever the original answer was.
        let mut refusal = error.clone();
        if let ProviderError::QuotaExhausted { in_stream, .. } = &mut refusal {
            *in_stream = false;
        }
        self.lock().insert(
            key.clone(),
            Cooldown {
                model: key,
                until: now + lasts,
                error: refusal,
            },
        );
    }
}

#[async_trait]
impl Provider for QuotaGuard {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        if let Some(error) = self.blocked(&request.model) {
            return Err(error);
        }
        let result = self.inner.complete(request).await;
        if let Err(error) = &result {
            self.record(&request.model, error);
        }
        result
    }

    fn unavailable_until(&self, model: &str) -> Option<SystemTime> {
        let now = SystemTime::now();
        let mut cooldowns = self.lock();
        cooldowns.retain(|_, cooldown| cooldown.until > now);
        [
            cooldowns.get(&None),
            cooldowns.get(&Some(model.to_string())),
        ]
        .into_iter()
        .flatten()
        .map(|cooldown| cooldown.until)
        .max()
    }

    fn stream<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxStream<'a, Result<StreamEvent, ProviderError>> {
        if let Some(error) = self.blocked(&request.model) {
            return stream::iter([Err(error)]).boxed();
        }
        self.inner
            .stream(request)
            .inspect(move |event| {
                if let Err(error) = event {
                    self.record(&request.model, error);
                }
            })
            .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{text, ScriptedProvider};
    use crate::{Failover, FailoverStep};

    fn request(model: &str) -> ModelRequest {
        ModelRequest {
            model: model.into(),
            messages: vec![crate::Message::user("hello")],
            ..Default::default()
        }
    }

    fn capped(resets_at: Option<SystemTime>, per_model: bool) -> ProviderError {
        ProviderError::QuotaExhausted {
            message: "plan cap".into(),
            resets_at,
            per_model,
            in_stream: false,
        }
    }

    /// A plan cap on the first model blocks the account's other models
    /// without sending them, until the stated reset.
    #[tokio::test]
    async fn an_account_cap_blocks_every_model_without_sending() {
        let reset = SystemTime::now() + Duration::from_secs(3600);
        let inner = Arc::new(ScriptedProvider::with_results(vec![
            Err(capped(Some(reset), false)),
            Ok(text("never sent")),
        ]));
        let guard = QuotaGuard::from_arc(inner.clone());

        assert_eq!(
            guard.complete(&request("model-a")).await,
            Err(capped(Some(reset), false))
        );
        assert_eq!(
            guard.complete(&request("model-b")).await,
            Err(capped(Some(reset), false))
        );
        assert_eq!(inner.requests().len(), 1);
        let active = guard.active();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].model, None);
        assert!(active[0].until >= reset - Duration::from_secs(1));
    }

    /// A per-model quota (a daily allowance of one model) leaves the
    /// account's other models alone.
    #[tokio::test]
    async fn a_per_model_cap_blocks_only_that_model() {
        let inner = Arc::new(ScriptedProvider::with_results(vec![
            Err(capped(None, true)),
            Ok(text("other model answers")),
        ]));
        let guard = QuotaGuard::from_arc(inner.clone());

        assert!(guard.complete(&request("model-a")).await.is_err());
        assert_eq!(
            guard.stream(&request("model-a")).collect::<Vec<_>>().await,
            vec![Err(capped(None, true))]
        );
        let other = guard.complete(&request("model-b")).await.unwrap();
        assert_eq!(other.text, "other model answers");
        assert_eq!(inner.requests().len(), 2);
    }

    /// Throttling, outages and a billing error that names a smaller
    /// affordable request are not remembered.
    #[tokio::test]
    async fn retryable_and_affordable_errors_start_no_cooldown() {
        let inner = Arc::new(ScriptedProvider::with_results(vec![
            Err(ProviderError::RateLimited { retry_after: None }),
            Err(ProviderError::Transient("blip".into())),
            Err(ProviderError::Billing {
                message: "402".into(),
                affordable_tokens: Some(600),
            }),
            Ok(text("answered")),
        ]));
        let guard = QuotaGuard::from_arc(inner.clone());
        for _ in 0..3 {
            assert!(guard.complete(&request("model-a")).await.is_err());
        }
        assert!(guard.active().is_empty());
        assert_eq!(
            guard.complete(&request("model-a")).await.unwrap().text,
            "answered"
        );
    }

    #[tokio::test]
    async fn an_unpayable_account_and_a_rejected_key_are_remembered() {
        let broke = QuotaGuard::new(ScriptedProvider::with_results(vec![Err(
            ProviderError::Billing {
                message: "no credits".into(),
                affordable_tokens: None,
            },
        )]));
        assert!(broke.complete(&request("m")).await.is_err());
        assert!(matches!(
            broke.complete(&request("other")).await,
            Err(ProviderError::Billing { .. })
        ));

        let rejected = QuotaGuard::new(ScriptedProvider::with_results(vec![Err(
            ProviderError::Auth("revoked".into()),
        )]));
        assert!(rejected
            .stream(&request("m"))
            .next()
            .await
            .unwrap()
            .is_err());
        assert_eq!(rejected.active().len(), 1);
        rejected.clear();
        assert!(rejected.active().is_empty());
    }

    /// In a chain, a capped account's remaining models fail at once and the
    /// next account answers: no backoff is spent on models that share the
    /// cap.
    #[tokio::test(start_paused = true)]
    async fn a_failover_chain_moves_past_a_capped_account_at_once() {
        let account = Arc::new(QuotaGuard::new(ScriptedProvider::with_results(vec![Err(
            capped(None, false),
        )])));
        let other = ScriptedProvider::new(vec![text("from the other provider")]);
        let chain = Failover::new(vec![
            FailoverStep::from_arc(account.clone())
                .model("plan-a")
                .attempts(3),
            FailoverStep::from_arc(account.clone())
                .model("plan-b")
                .attempts(3),
            FailoverStep::new(other).model("other-model"),
        ]);
        let started = tokio::time::Instant::now();
        let response = chain.complete(&request("plan-a")).await.unwrap();
        assert_eq!(response.text, "from the other provider");
        assert_eq!(started.elapsed(), Duration::ZERO);
    }

    /// A used-up account says until when every model of it cannot answer;
    /// one that has not failed may answer now.
    #[tokio::test]
    async fn a_cooldown_says_until_when_the_model_cannot_answer() {
        let reset = SystemTime::now() + Duration::from_secs(3_600);
        let guard = QuotaGuard::new(ScriptedProvider::with_results(vec![Err(
            ProviderError::QuotaExhausted {
                message: "limit".into(),
                resets_at: Some(reset),
                per_model: false,
                in_stream: true,
            },
        )]));
        assert_eq!(guard.unavailable_until("any-model"), None);
        let _ = guard.complete(&request("model-a")).await;
        let until = guard
            .unavailable_until("model-b")
            .expect("account cooldown");
        assert!(until >= reset - Duration::from_secs(1) && until <= reset);
        // A blocked call is refused before it is sent: provably unspent.
        let refusal = guard.complete(&request("model-b")).await.unwrap_err();
        assert!(refusal.proves_nothing_spent(), "{refusal:?}");
    }
}
