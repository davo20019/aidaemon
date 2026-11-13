//! A turn's model calls: the model the turn runs on (an owner's pin, the
//! router's profile pick, or the agent's own), and one call with the
//! daemon's recovery as a kernel failover chain.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aidaemon_events::provider::ChatOptions;
use aidaemon_providers::routing::Router;
use serde_json::Value;
use tracing::{info, warn};

use crate::chat_bridge::{ProviderError, ProviderErrorKind, ProviderRecoveryRoute};
use crate::host::TurnHost;
use crate::policy::metrics::{provider_kind_metric_label, record_llm_payload_invalid_metric};

/// Maximum number of retries for transient LLM errors.
const MAX_LLM_RETRIES: u32 = 3;
/// Transient-error retries when the request has committed no tool work.
const MAX_LLM_RETRIES_UNCOMMITTED: u32 = 5;
/// Cap on a single transient-error backoff sleep.
const TRANSIENT_RETRY_MAX_DELAY_SECS: u64 = 30;
/// Base delay for exponential backoff on transient errors (seconds).
const RETRY_BASE_DELAY_SECS: u64 = 2;
/// Single retry budget for malformed payloads that are likely deterministic
/// (shape/unknown). Parse errors use transient retry + fallback recovery.
const MAX_MALFORMED_PAYLOAD_RETRIES: u32 = 1;
/// Small delay before malformed-payload retry to smooth transient gateway glitches.
const MALFORMED_PAYLOAD_RETRY_DELAY_SECS: u64 = 1;

/// Telemetry captured during an LLM call with recovery, surfaced to the agent
/// loop so it can be persisted as an `LlmCall` event. Populated via an out-param
/// so the many return points in `call_llm_with_recovery` stay untouched.
#[derive(Debug, Clone, Default)]
pub struct LlmCallTelemetry {
    /// Model that actually produced the successful response.
    pub final_model: String,
    /// Whether recovery cascaded to a different model/provider.
    pub fell_back: bool,
    /// Total provider calls attempted (1 = succeeded on the first try).
    pub attempts: u32,
}

fn malformed_reason_label(
    reason: Option<crate::chat_bridge::MalformedResponseReason>,
) -> &'static str {
    match reason {
        Some(crate::chat_bridge::MalformedResponseReason::Parse) => "parse",
        Some(crate::chat_bridge::MalformedResponseReason::Shape) => "shape",
        None => "unknown",
    }
}

/// The daemon's recovery rules as a kernel [`aidaemon_core::FailoverPolicy`]:
/// each provider error's recovery route decides whether to retry the same
/// step (with backoff, or with a rewritten request) or move to the next model.
/// One instance serves one call and records what it saw for telemetry.
struct DaemonRecovery {
    billing_failed_models: Arc<tokio::sync::RwLock<HashMap<String, Instant>>>,
    uncommitted_work: bool,
    provider_label: &'static str,
    /// Models attempted, in order, and whether the run left the first one.
    attempted: std::sync::Mutex<Vec<String>>,
    /// The first used-up quota the call met: when every later model also
    /// fails, it is the failure to report (with its reset), not the last
    /// fallback's error.
    quota: std::sync::Mutex<Option<aidaemon_core::ProviderError>>,
    /// The latest failure of each model tried, in first-attempt order.
    failures: std::sync::Mutex<Vec<crate::chat_bridge::ModelFailure>>,
}

impl DaemonRecovery {
    fn failures(&self) -> Vec<crate::chat_bridge::ModelFailure> {
        self.failures
            .lock()
            .map(|failures| failures.clone())
            .unwrap_or_default()
    }
}

/// How long a billing failure is cached before the model is tried again
/// (credits may have been added).
const BILLING_FAIL_CACHE_TTL: Duration = Duration::from_secs(10 * 60);
/// The smallest output cap worth retrying with after a 402.
const MIN_USEFUL_TOKENS: u64 = 512;

fn has_media(request: &aidaemon_core::ModelRequest) -> bool {
    request
        .messages
        .iter()
        .any(|m| !m.images.is_empty() || !m.audio.is_empty())
}

fn without_media(request: &aidaemon_core::ModelRequest) -> aidaemon_core::ModelRequest {
    let mut request = request.clone();
    for message in &mut request.messages {
        message.images.clear();
        message.audio.clear();
    }
    request
}

/// The request with the provider's defaults instead of advanced options
/// (response format, forced tool choice, reasoning effort, output cap).
fn with_default_options(
    request: &aidaemon_core::ModelRequest,
) -> Option<aidaemon_core::ModelRequest> {
    // The cache key only routes prompt-cache lookups and adds no field a
    // provider could reject, so the retry keeps the conversation's cache.
    let defaults = aidaemon_core::ModelRequest {
        model: request.model.clone(),
        messages: request.messages.clone(),
        tools: request.tools.clone(),
        options: aidaemon_core::RequestOptions {
            cache_key: request.options.cache_key.clone(),
            ..Default::default()
        },
        ..Default::default()
    };
    (defaults != *request).then_some(defaults)
}

impl aidaemon_core::FailoverPolicy for DaemonRecovery {
    fn before_attempt(
        &self,
        request: &aidaemon_core::ModelRequest,
    ) -> Result<(), aidaemon_core::ProviderError> {
        if let Ok(mut attempted) = self.attempted.lock() {
            attempted.push(request.model.clone());
        }
        let cached = self
            .billing_failed_models
            .try_read()
            .ok()
            .and_then(|cache| cache.get(&request.model).copied())
            .is_some_and(|failed_at| failed_at.elapsed() < BILLING_FAIL_CACHE_TTL);
        if cached {
            info!(
                model = %request.model,
                "Skipping model — cached billing failure"
            );
            return Err(aidaemon_core::ProviderError::Billing {
                message: format!("Cached billing failure for {}", request.model),
                affordable_tokens: None,
            });
        }
        Ok(())
    }

    fn after_error(
        &self,
        request: &aidaemon_core::ModelRequest,
        error: &aidaemon_core::ProviderError,
        attempt: u32,
        _attempts: u32,
    ) -> aidaemon_core::FailoverAction {
        use aidaemon_core::FailoverAction::{Next, Retry, RetryWith, Stop};
        let model = request.model.as_str();
        let daemon_error = crate::chat_bridge::sdk_backed::to_daemon_error(error.clone());
        warn!(
            model,
            attempt,
            kind = ?daemon_error.kind,
            status = ?daemon_error.status,
            "LLM call failed: {}",
            daemon_error
        );
        if let Ok(mut failures) = self.failures.lock() {
            let failure = crate::chat_bridge::ModelFailure {
                model: model.to_string(),
                kind: daemon_error.kind,
                status: daemon_error.status,
            };
            match failures.iter_mut().find(|known| known.model == model) {
                Some(known) => *known = failure,
                None => failures.push(failure),
            }
        }
        if matches!(error, aidaemon_core::ProviderError::QuotaExhausted { .. }) {
            if let Ok(mut quota) = self.quota.lock() {
                quota.get_or_insert_with(|| error.clone());
            }
        }
        // Too long for this model's window: retrying cannot help. Record
        // the window so the caller shrinks the request.
        if let aidaemon_core::ProviderError::ContextOverflow { limit, .. } = error {
            let attempted = crate::context_budget::estimate_multimodal_message_tokens(
                &crate::chat_bridge::sdk_backed::to_chat_messages(&request.messages),
            );
            crate::settings::record_context_overflow(model, limit.map(|l| l as usize), attempted);
            return Stop;
        }
        let backoff = |base: Duration, max: Duration| {
            aidaemon_tools::backoff::exponential_backoff(base, attempt.saturating_sub(1), max)
        };
        match daemon_error.recovery_route(false) {
            ProviderRecoveryRoute::Fail => Stop,
            // A 402 that says what it can afford: retry once with that cap,
            // then remember the failure and move on.
            ProviderRecoveryRoute::RetryReducedBudgetThenFallback => {
                if let aidaemon_core::ProviderError::Billing {
                    affordable_tokens: Some(affordable),
                    ..
                } = error
                {
                    if *affordable >= MIN_USEFUL_TOKENS && request.max_output_tokens.is_none() {
                        let mut capped = request.clone();
                        capped.max_output_tokens =
                            Some(affordable.saturating_sub(100).max(MIN_USEFUL_TOKENS) as u32);
                        return RetryWith {
                            delay: Duration::ZERO,
                            request: Box::new(capped),
                        };
                    }
                }
                if let Ok(mut cache) = self.billing_failed_models.try_write() {
                    cache.insert(model.to_string(), Instant::now());
                    cache.retain(|_, at| at.elapsed() < BILLING_FAIL_CACHE_TTL);
                }
                Next
            }
            // A rejected request: drop media once, then advanced options once.
            ProviderRecoveryRoute::AdaptRequestOnce => {
                if has_media(request) {
                    warn!(
                        model,
                        "Provider rejected multimodal payload; retrying text-only"
                    );
                    return RetryWith {
                        delay: Duration::ZERO,
                        request: Box::new(without_media(request)),
                    };
                }
                match with_default_options(request) {
                    Some(defaults) => {
                        warn!(
                            model,
                            "Provider rejected advanced options; retrying with defaults"
                        );
                        RetryWith {
                            delay: Duration::ZERO,
                            request: Box::new(defaults),
                        }
                    }
                    None => Stop,
                }
            }
            ProviderRecoveryRoute::RetryRateLimitThenFallback => {
                if attempt > MAX_LLM_RETRIES {
                    return Next;
                }
                let base = Duration::from_secs(daemon_error.retry_after_secs.unwrap_or(5));
                Retry(backoff(base, Duration::from_secs(120)))
            }
            ProviderRecoveryRoute::RetryTransientThenFallback => {
                // A server error on a multimodal payload: try once without
                // the media before backing off.
                if daemon_error.kind == ProviderErrorKind::ServerError && has_media(request) {
                    warn!(
                        model,
                        "Server error on a multimodal payload; retrying text-only"
                    );
                    return RetryWith {
                        delay: Duration::ZERO,
                        request: Box::new(without_media(request)),
                    };
                }
                // Before any tool work has committed, an outage is cheap to
                // wait out; after, keep the short ladder so a half-done task
                // closes from its receipts.
                let max = if self.uncommitted_work {
                    MAX_LLM_RETRIES_UNCOMMITTED
                } else {
                    MAX_LLM_RETRIES
                };
                if attempt > max {
                    return Next;
                }
                Retry(aidaemon_tools::backoff::jittered_delay(
                    backoff(
                        Duration::from_secs(RETRY_BASE_DELAY_SECS),
                        Duration::from_secs(TRANSIENT_RETRY_MAX_DELAY_SECS),
                    ),
                    0.2,
                ))
            }
            ProviderRecoveryRoute::RetryMalformedThenFallback => {
                record_llm_payload_invalid_metric(self.provider_label, model, "parse");
                if attempt > MAX_LLM_RETRIES {
                    return Next;
                }
                Retry(backoff(
                    Duration::from_secs(RETRY_BASE_DELAY_SECS),
                    Duration::MAX,
                ))
            }
            // A wrongly shaped reply is likely deterministic: one quick
            // retry, then give up without other models.
            ProviderRecoveryRoute::RetryMalformedThenFail => {
                record_llm_payload_invalid_metric(
                    self.provider_label,
                    model,
                    malformed_reason_label(daemon_error.malformed_reason),
                );
                if attempt > MAX_MALFORMED_PAYLOAD_RETRIES {
                    return Stop;
                }
                Retry(Duration::from_secs(MALFORMED_PAYLOAD_RETRY_DELAY_SECS))
            }
            ProviderRecoveryRoute::Fallback => Next,
        }
    }
}

/// Per-attempt deadline for each fallback model. Prevents a single hung
/// provider from blocking the rest of the chain.
const CASCADE_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(120);
/// Attempts a step may make; the policy's per-route limits decide sooner.
const MAX_STEP_ATTEMPTS: u32 = 16;

/// The models to fall back to after `model`, in order: the stored
/// fallback model, then the router's models, each once.
async fn fallback_models(host: &dyn TurnHost, model: &str, router: Option<&Router>) -> Vec<String> {
    let stored =
        match tokio::time::timeout(Duration::from_secs(2), host.fallback_model_lock().read()).await
        {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!("Timed out acquiring fallback_model lock while picking fallback");
                host.llm_runtime().snapshot().primary_model()
            }
        };
    let mut out: Vec<String> = Vec::new();
    for candidate in std::iter::once(stored).chain(
        router
            .map(|router| router.all_models_ordered())
            .unwrap_or_default(),
    ) {
        if candidate != model && !out.contains(&candidate) {
            out.push(candidate);
        }
    }
    out
}

/// One model call with the daemon's recovery, run as a kernel failover
/// chain: `model` on `provider` first (within `first_attempt_deadline`),
/// then the other models of its router, then each configured failover
/// provider's models (each within its own deadline). The chain is the
/// app's policy; [`DaemonRecovery`] decides retries per error:
///
/// - RateLimit, Timeout/Network/Server, malformed parse: backoff retries,
///   then the next model (a server error on media first retries without
///   it);
/// - BadRequest: retry once without media, then once with default
///   options, then fail;
/// - Billing: retry once with the affordable cap, then cache the failure
///   and move on (cached models are skipped for ten minutes);
/// - NotFound: next model; malformed shape: one retry, then fail;
/// - Auth or a used-up quota: the next provider (each backend's
///   `QuotaGuard` fails the account's other models without a request,
///   and keeps failing them until the quota resets); when every model
///   fails, the quota and its reset are what is reported;
/// - a context overflow (recorded for the caller): fail.
///
/// `pin_model` (an active computer-use task) keeps the call on `model`.
#[allow(clippy::too_many_arguments)]
pub async fn call_llm_with_recovery(
    host: &dyn TurnHost,
    provider: crate::llm::ModelBackend,
    router: Option<Router>,
    model: &str,
    messages: &[Value],
    tool_defs: &[Value],
    options: &ChatOptions,
    telemetry: &mut LlmCallTelemetry,
    pin_model: Option<&str>,
    first_attempt_deadline: Duration,
) -> anyhow::Result<aidaemon_events::provider::ProviderResponse> {
    use aidaemon_core::{Failover, FailoverStep};

    let mut steps = vec![FailoverStep::from_arc(provider.clone())
        .model(model)
        .deadline(first_attempt_deadline)
        .attempts(MAX_STEP_ATTEMPTS)];
    if pin_model.is_none() {
        for fallback in fallback_models(host, model, router.as_ref()).await {
            steps.push(
                FailoverStep::from_arc(provider.clone())
                    .model(fallback)
                    .deadline(CASCADE_ATTEMPT_TIMEOUT)
                    .attempts(MAX_STEP_ATTEMPTS),
            );
        }
        for target in host.llm_runtime().snapshot().failover_targets() {
            for fallback in target.all_models_ordered() {
                steps.push(
                    FailoverStep::from_arc(target.provider())
                        .model(fallback)
                        .deadline(CASCADE_ATTEMPT_TIMEOUT)
                        .attempts(MAX_STEP_ATTEMPTS),
                );
            }
        }
    }
    let recovery = Arc::new(DaemonRecovery {
        billing_failed_models: host.billing_failed_models().clone(),
        uncommitted_work: options.uncommitted_work,
        provider_label: provider_kind_metric_label(host.llm_runtime().snapshot().provider_kind()),
        attempted: std::sync::Mutex::new(Vec::new()),
        quota: std::sync::Mutex::new(None),
        failures: std::sync::Mutex::new(Vec::new()),
    });
    let chain = Failover::new(steps).policy(RecoveryHandle(recovery.clone()));

    let request =
        crate::chat_bridge::sdk_backed::to_sdk_request(model, messages, tool_defs, options);
    let result = crate::chat_bridge::sdk_backed::send(&chain, &request).await;

    let attempted = recovery
        .attempted
        .lock()
        .map(|models| models.clone())
        .unwrap_or_default();
    telemetry.attempts = attempted.len().max(1) as u32;
    telemetry.final_model = attempted
        .last()
        .cloned()
        .unwrap_or_else(|| model.to_string());
    telemetry.fell_back = attempted.iter().any(|attempt| attempt != model);

    match result {
        Ok(response) => {
            // The config works: stamp it as last known good.
            host.stamp_lastgood().await;
            Ok(crate::chat_bridge::sdk_backed::to_provider_response(
                response,
                &telemetry.final_model,
            ))
        }
        Err(error) => match error.downcast::<ProviderError>() {
            Ok(mut provider_err) => {
                let quota = recovery.quota.lock().ok().and_then(|quota| quota.clone());
                if let Some(quota) = quota {
                    return Err(quota_failure(quota, provider_err, recovery.failures()));
                }
                if telemetry.fell_back {
                    provider_err.fallback_failures = recovery.failures();
                }
                let terminal = provider_err.kind == ProviderErrorKind::Auth
                    || options.single_attempt_fail_closed
                    || crate::chat_bridge::context_overflow(&provider_err.to_string()).is_some();
                if terminal && !telemetry.fell_back {
                    return Err(provider_err.into());
                }
                let message = if telemetry.fell_back {
                    provider_err.recovery_failed_message()
                } else {
                    provider_err.user_message()
                };
                Err(anyhow::Error::new(provider_err).context(message))
            }
            Err(other) => Err(other),
        },
    }
}

/// The error for a call that met a used-up quota and then failed on every
/// other model: the quota (and its reset) is what the user can act on, so it
/// is the typed error; what the fallbacks answered follows it.
fn quota_failure(
    quota: aidaemon_core::ProviderError,
    last: ProviderError,
    failures: Vec<crate::chat_bridge::ModelFailure>,
) -> anyhow::Error {
    let mut quota = crate::chat_bridge::sdk_backed::to_daemon_error(quota);
    // The call spent nothing only if its last attempt did too.
    quota.spent_nothing &= last.spent_nothing;
    quota.fallback_failures = failures
        .into_iter()
        .filter(|failure| failure.kind != ProviderErrorKind::QuotaExhausted)
        .collect();
    let mut message = quota.user_message();
    if last.kind != ProviderErrorKind::QuotaExhausted {
        message.push_str(&format!(
            " The fallback providers also failed: {}",
            last.recovery_failed_message()
        ));
    }
    anyhow::Error::new(quota).context(message)
}

/// Shares one [`DaemonRecovery`] between the chain and the caller, which
/// reads what it recorded.
struct RecoveryHandle(Arc<DaemonRecovery>);

impl aidaemon_core::FailoverPolicy for RecoveryHandle {
    fn before_attempt(
        &self,
        request: &aidaemon_core::ModelRequest,
    ) -> Result<(), aidaemon_core::ProviderError> {
        self.0.before_attempt(request)
    }

    fn after_error(
        &self,
        request: &aidaemon_core::ModelRequest,
        error: &aidaemon_core::ProviderError,
        attempt: u32,
        attempts: u32,
    ) -> aidaemon_core::FailoverAction {
        self.0.after_error(request, error, attempt, attempts)
    }
}

pub async fn model_override_active(host: &dyn TurnHost, session_id: &str) -> bool {
    match tokio::time::timeout(Duration::from_secs(2), host.model_override_lock().read()).await {
        Ok(guard) => *guard,
        Err(_) => {
            warn!(
                session_id,
                "Timed out acquiring model_override lock; treating the model as pinned"
            );
            true
        }
    }
}

pub async fn select_turn_model(
    host: &dyn TurnHost,
    snapshot: &crate::llm::LlmRuntimeSnapshot,
    router: Option<&aidaemon_providers::routing::Router>,
    profile: crate::policy::execution::ModelProfile,
    session_id: &str,
) -> (String, bool) {
    let is_override =
        match tokio::time::timeout(Duration::from_secs(2), host.model_override_lock().read()).await
        {
            Ok(guard) => *guard,
            Err(_) => {
                warn!(
                    session_id,
                    "Timed out acquiring model_override lock while selecting bootstrap model"
                );
                false
            }
        };
    let own_model = || async {
        match tokio::time::timeout(Duration::from_secs(2), host.model_lock().read()).await {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!(
                    session_id,
                    "Timed out acquiring model lock while selecting a model"
                );
                snapshot.primary_model()
            }
        }
    };
    if is_override {
        return (own_model().await, false);
    }
    let autonomous = host
        .turn_runtime()
        .trust_tier_for_model(&snapshot.primary_model())
        == crate::policy::trust_tier::ModelTrustTier::Autonomous;
    if autonomous {
        return (snapshot.primary_model(), false);
    }
    if let Some(router) = router {
        return (router.select_for_profile(profile).to_string(), true);
    }
    if host.depth() == 0 {
        (snapshot.primary_model(), false)
    } else {
        (own_model().await, false)
    }
}
