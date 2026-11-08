use super::*;

/// Telemetry captured during an LLM call with recovery, surfaced to the agent
/// loop so it can be persisted as an `LlmCall` event. Populated via an out-param
/// so the many return points in `call_llm_with_recovery` stay untouched.
#[derive(Debug, Clone, Default)]
pub(in crate::agent) struct LlmCallTelemetry {
    /// Model that actually produced the successful response.
    pub final_model: String,
    /// Whether recovery cascaded to a different model/provider.
    pub fell_back: bool,
    /// Total provider calls attempted (1 = succeeded on the first try).
    pub attempts: u32,
}

fn malformed_reason_label(
    reason: Option<crate::providers::MalformedResponseReason>,
) -> &'static str {
    match reason {
        Some(crate::providers::MalformedResponseReason::Parse) => "parse",
        Some(crate::providers::MalformedResponseReason::Shape) => "shape",
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
    failures: std::sync::Mutex<Vec<crate::providers::ModelFailure>>,
}

impl DaemonRecovery {
    fn failures(&self) -> Vec<crate::providers::ModelFailure> {
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
    let defaults = aidaemon_core::ModelRequest {
        model: request.model.clone(),
        messages: request.messages.clone(),
        tools: request.tools.clone(),
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
        let daemon_error = crate::providers::sdk_backed::to_daemon_error(error.clone());
        warn!(
            model,
            attempt,
            kind = ?daemon_error.kind,
            status = ?daemon_error.status,
            "LLM call failed: {}",
            daemon_error
        );
        if let Ok(mut failures) = self.failures.lock() {
            let failure = crate::providers::ModelFailure {
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
                &crate::providers::sdk_backed::to_chat_messages(&request.messages),
            );
            crate::config::record_context_overflow(model, limit.map(|l| l as usize), attempted);
            return Stop;
        }
        let backoff = |base: Duration, max: Duration| {
            crate::backoff::exponential_backoff(base, attempt.saturating_sub(1), max)
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
                if attempt > Agent::MAX_LLM_RETRIES {
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
                    Agent::MAX_LLM_RETRIES_UNCOMMITTED
                } else {
                    Agent::MAX_LLM_RETRIES
                };
                if attempt > max {
                    return Next;
                }
                Retry(crate::backoff::jittered_delay(
                    backoff(
                        Duration::from_secs(Agent::RETRY_BASE_DELAY_SECS),
                        Duration::from_secs(Agent::TRANSIENT_RETRY_MAX_DELAY_SECS),
                    ),
                    0.2,
                ))
            }
            ProviderRecoveryRoute::RetryMalformedThenFallback => {
                record_llm_payload_invalid_metric(self.provider_label, model, "parse");
                if attempt > Agent::MAX_LLM_RETRIES {
                    return Next;
                }
                Retry(backoff(
                    Duration::from_secs(Agent::RETRY_BASE_DELAY_SECS),
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
                if attempt > Agent::MAX_MALFORMED_PAYLOAD_RETRIES {
                    return Stop;
                }
                Retry(Duration::from_secs(
                    Agent::MALFORMED_PAYLOAD_RETRY_DELAY_SECS,
                ))
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

// impl-Agent justification: LLM call recovery over llm_runtime/model/billing_failed_models.
impl Agent {
    /// The models to fall back to after `model`, in order: the stored
    /// fallback model, then the router's models, each once.
    async fn fallback_models(&self, model: &str, router: Option<&Router>) -> Vec<String> {
        let stored =
            match tokio::time::timeout(Duration::from_secs(2), self.fallback_model.read()).await {
                Ok(guard) => guard.clone(),
                Err(_) => {
                    warn!("Timed out acquiring fallback_model lock while picking fallback");
                    self.llm_runtime.snapshot().primary_model()
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
    pub(super) async fn call_llm_with_recovery(
        &self,
        provider: crate::llm_runtime::ModelBackend,
        router: Option<Router>,
        model: &str,
        messages: &[Value],
        tool_defs: &[Value],
        options: &ChatOptions,
        telemetry: &mut LlmCallTelemetry,
        pin_model: Option<&str>,
        first_attempt_deadline: Duration,
    ) -> anyhow::Result<crate::traits::ProviderResponse> {
        use aidaemon_core::{Failover, FailoverStep};

        let mut steps = vec![FailoverStep::from_arc(provider.clone())
            .model(model)
            .deadline(first_attempt_deadline)
            .attempts(MAX_STEP_ATTEMPTS)];
        if pin_model.is_none() {
            for fallback in self.fallback_models(model, router.as_ref()).await {
                steps.push(
                    FailoverStep::from_arc(provider.clone())
                        .model(fallback)
                        .deadline(CASCADE_ATTEMPT_TIMEOUT)
                        .attempts(MAX_STEP_ATTEMPTS),
                );
            }
            for target in self.llm_runtime.snapshot().failover_targets() {
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
            billing_failed_models: self.billing_failed_models.clone(),
            uncommitted_work: options.uncommitted_work,
            provider_label: provider_kind_metric_label(self.llm_runtime.snapshot().provider_kind()),
            attempted: std::sync::Mutex::new(Vec::new()),
            quota: std::sync::Mutex::new(None),
            failures: std::sync::Mutex::new(Vec::new()),
        });
        let chain = Failover::new(steps).policy(RecoveryHandle(recovery.clone()));

        let request =
            crate::providers::sdk_backed::to_sdk_request(model, messages, tool_defs, options);
        let result = crate::providers::sdk_backed::send(&chain, &request).await;

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
                self.stamp_lastgood().await;
                Ok(crate::providers::sdk_backed::to_provider_response(
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
                        || crate::providers::context_overflow(&provider_err.to_string()).is_some();
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
}

/// The error for a call that met a used-up quota and then failed on every
/// other model: the quota (and its reset) is what the user can act on, so it
/// is the typed error; what the fallbacks answered follows it.
fn quota_failure(
    quota: aidaemon_core::ProviderError,
    last: ProviderError,
    failures: Vec<crate::providers::ModelFailure>,
) -> anyhow::Error {
    let mut quota = crate::providers::sdk_backed::to_daemon_error(quota);
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use serde_json::Value;
    use tokio::sync::Mutex;

    use crate::llm_runtime::ProviderRuntimeTarget;
    use crate::providers::ProviderError;
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::traits::store_prelude::*;
    use crate::traits::{ChatOptions, Goal, Mandate, MandateAuthority, ProviderResponse, Task};

    struct ScriptedProvider {
        responses: Mutex<Vec<anyhow::Result<ProviderResponse>>>,
        models_called: Mutex<Vec<String>>,
    }

    impl ScriptedProvider {
        fn with_results(responses: Vec<anyhow::Result<ProviderResponse>>) -> Self {
            Self {
                responses: Mutex::new(responses),
                models_called: Mutex::new(Vec::new()),
            }
        }

        async fn models_called(&self) -> Vec<String> {
            self.models_called.lock().await.clone()
        }
    }

    crate::testing::chat_json_provider!(ScriptedProvider);

    impl ScriptedProvider {
        async fn respond(
            &self,
            model: &str,
            _messages: &[Value],
            _tools: &[Value],
            _options: &ChatOptions,
        ) -> anyhow::Result<ProviderResponse> {
            self.models_called.lock().await.push(model.to_string());
            let mut responses = self.responses.lock().await;
            if responses.is_empty() {
                return Err(ProviderError::from_status(500, "script exhausted").into());
            }
            responses.remove(0)
        }
    }

    #[tokio::test]
    async fn interpret_background_result_returns_model_summary() {
        // The tool-less interpretation returns the model's one-line summary so
        // the notifier can deliver a contextual answer instead of a bare value.
        let provider = MockProvider::with_responses(vec![MockProvider::text_response(
            "That's 345 lines matching \"resume\" — a raw text-match count, not a file count.",
        )]);
        let harness = setup_test_agent(provider)
            .await
            .expect("build test harness");

        let summary = harness
            .agent
            .interpret_background_result("ls -R ~/projects | grep -i resume | wc -l", "345")
            .await;

        assert_eq!(
            summary.as_deref(),
            Some(
                "That's 345 lines matching \"resume\" — a raw text-match count, not a file count."
            )
        );
    }

    #[tokio::test]
    async fn interpret_background_result_falls_back_to_none_on_empty() {
        // Empty model output → None, so the caller falls back to the raw result
        // and the answer is never lost.
        let provider = MockProvider::with_responses(vec![MockProvider::text_response("   ")]);
        let harness = setup_test_agent(provider)
            .await
            .expect("build test harness");

        let summary = harness
            .agent
            .interpret_background_result("echo done", "done")
            .await;

        assert!(summary.is_none());
    }

    #[tokio::test]
    async fn a_hanging_primary_falls_back_within_its_own_deadline() {
        /// Never answers.
        struct Hanging;

        #[async_trait::async_trait]
        impl aidaemon_core::Provider for Hanging {
            fn name(&self) -> &str {
                "hanging"
            }

            async fn complete(
                &self,
                _: &aidaemon_core::ModelRequest,
            ) -> Result<aidaemon_core::ModelResponse, aidaemon_core::ProviderError> {
                std::future::pending().await
            }
        }

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("build test harness");
        let alternate = Arc::new(ScriptedProvider::with_results(vec![Ok(
            MockProvider::text_response("Answered by the failover provider"),
        )]));
        let hanging: crate::llm_runtime::ModelBackend = Arc::new(Hanging);
        harness.agent.llm_runtime.swap(
            hanging.clone(),
            None,
            crate::config::ProviderKind::OpenaiCompatible,
            "primary-model".to_string(),
            vec![ProviderRuntimeTarget::new(
                alternate.clone() as crate::llm_runtime::ModelBackend,
                None,
                crate::config::ProviderKind::Anthropic,
                "secondary-model".to_string(),
            )],
            crate::providers::sdk_backed::ModelList::Fixed(Vec::new()),
        );

        // No other model on the hanging provider: the chain is the primary,
        // then the failover provider.
        *harness.agent.fallback_model.write().await = "primary-model".to_string();

        let mut telemetry = super::LlmCallTelemetry::default();
        let started = std::time::Instant::now();
        let response = harness
            .agent
            .call_llm_with_recovery(
                hanging,
                None,
                "primary-model",
                &[],
                &[],
                &ChatOptions::default(),
                &mut telemetry,
                None,
                std::time::Duration::from_millis(200),
            )
            .await
            .expect("the failover provider answers");
        assert_eq!(
            response.content.as_deref(),
            Some("Answered by the failover provider")
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "the primary must give up at its own deadline, took {:?}",
            started.elapsed()
        );
        assert!(telemetry.fell_back);
        assert_eq!(telemetry.final_model, "secondary-model");
        assert_eq!(alternate.models_called().await, vec!["secondary-model"]);
    }

    #[tokio::test]
    async fn recovery_can_failover_to_another_provider() {
        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("build test harness");

        let primary_provider = Arc::new(ScriptedProvider::with_results(vec![
            Err(ProviderError::from_status(500, "primary-1").into()),
            Err(ProviderError::from_status(500, "primary-2").into()),
            Err(ProviderError::from_status(500, "primary-3").into()),
            Err(ProviderError::from_status(500, "primary-4").into()),
        ]));
        let alternate_provider = Arc::new(ScriptedProvider::with_results(vec![Ok(
            MockProvider::text_response("Recovered via failover provider"),
        )]));

        harness.agent.llm_runtime.swap(
            primary_provider.clone() as crate::llm_runtime::ModelBackend,
            None,
            crate::config::ProviderKind::OpenaiCompatible,
            "primary-model".to_string(),
            vec![ProviderRuntimeTarget::new(
                alternate_provider.clone() as crate::llm_runtime::ModelBackend,
                None,
                crate::config::ProviderKind::Anthropic,
                "secondary-model".to_string(),
            )],
            crate::providers::sdk_backed::ModelList::Fixed(Vec::new()),
        );

        let mut telemetry = super::LlmCallTelemetry::default();
        let resp = harness
            .agent
            .call_llm_with_recovery(
                primary_provider as crate::llm_runtime::ModelBackend,
                None,
                "primary-model",
                &[],
                &[],
                &ChatOptions::default(),
                &mut telemetry,
                None,
                std::time::Duration::from_secs(60),
            )
            .await
            .expect("recover via alternate provider");

        assert_eq!(
            resp.content.as_deref(),
            Some("Recovered via failover provider")
        );
        assert_eq!(
            alternate_provider.models_called().await,
            vec!["secondary-model".to_string()]
        );
        assert!(telemetry.fell_back, "expected fallback to be recorded");
        assert_eq!(telemetry.final_model, "secondary-model");
        assert!(
            telemetry.attempts >= 2,
            "expected multiple attempts, got {}",
            telemetry.attempts
        );
    }

    fn plan_cap(resets_at: std::time::SystemTime) -> ProviderError {
        let mut error = ProviderError::from_status(429, "The usage limit has been reached");
        error.kind = crate::providers::ProviderErrorKind::QuotaExhausted;
        error.resets_at = Some(resets_at);
        error
    }

    /// A capped plan: the primary account's other models are not sent (its
    /// guard answers for them), no backoff is spent, and another provider
    /// answers.
    #[tokio::test]
    async fn a_capped_plan_moves_to_another_provider_without_waiting() {
        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("build test harness");
        let reset = std::time::SystemTime::now() + std::time::Duration::from_secs(7200);
        let primary_inner = Arc::new(ScriptedProvider::with_results(vec![
            Err(plan_cap(reset).into()),
            Ok(MockProvider::text_response(
                "the capped account must not be sent",
            )),
        ]));
        let primary: crate::llm_runtime::ModelBackend =
            Arc::new(aidaemon_core::QuotaGuard::from_arc(primary_inner.clone()));
        let alternate = Arc::new(ScriptedProvider::with_results(vec![Ok(
            MockProvider::text_response("answered by the other provider"),
        )]));
        let router = crate::router::Router::new(crate::config::ModelsConfig {
            default_model: "plan-model".into(),
            fallback_models: vec!["plan-model-small".into(), "plan-model-mini".into()],
            ..Default::default()
        });
        harness.agent.llm_runtime.swap(
            primary.clone(),
            Some(router.clone()),
            crate::config::ProviderKind::OpenaiChatgpt,
            "plan-model".to_string(),
            vec![ProviderRuntimeTarget::new(
                alternate.clone() as crate::llm_runtime::ModelBackend,
                None,
                crate::config::ProviderKind::Anthropic,
                "other-model".to_string(),
            )],
            crate::providers::sdk_backed::ModelList::Fixed(Vec::new()),
        );

        let started = Instant::now();
        let mut telemetry = super::LlmCallTelemetry::default();
        let response = harness
            .agent
            .call_llm_with_recovery(
                primary,
                Some(router),
                "plan-model",
                &[],
                &[],
                &ChatOptions::default(),
                &mut telemetry,
                None,
                std::time::Duration::from_secs(60),
            )
            .await
            .expect("another provider answers");

        assert_eq!(
            response.content.as_deref(),
            Some("answered by the other provider")
        );
        assert_eq!(
            primary_inner.models_called().await,
            vec!["plan-model".to_string()]
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(telemetry.final_model, "other-model");
    }

    /// When every other provider fails too, the plan cap and its reset are
    /// the reported failure, and what the fallback said follows it.
    #[tokio::test]
    async fn a_capped_plan_is_the_reported_failure_when_fallbacks_fail() {
        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("build test harness");
        let reset = std::time::SystemTime::now() + std::time::Duration::from_secs(7200);
        let primary: crate::llm_runtime::ModelBackend =
            Arc::new(aidaemon_core::QuotaGuard::from_arc(Arc::new(
                ScriptedProvider::with_results(vec![Err(plan_cap(reset).into())]),
            )));
        let broke = Arc::new(ScriptedProvider::with_results(vec![Err(
            ProviderError::from_status(402, "insufficient credits").into(),
        )]));
        harness.agent.llm_runtime.swap(
            primary.clone(),
            None,
            crate::config::ProviderKind::OpenaiChatgpt,
            "plan-model".to_string(),
            vec![ProviderRuntimeTarget::new(
                broke as crate::llm_runtime::ModelBackend,
                None,
                crate::config::ProviderKind::OpenaiCompatible,
                "broke-model".to_string(),
            )],
            crate::providers::sdk_backed::ModelList::Fixed(Vec::new()),
        );

        let mut telemetry = super::LlmCallTelemetry::default();
        let error = harness
            .agent
            .call_llm_with_recovery(
                primary,
                None,
                "plan-model",
                &[],
                &[],
                &ChatOptions::default(),
                &mut telemetry,
                None,
                std::time::Duration::from_secs(60),
            )
            .await
            .expect_err("every provider fails");

        let typed = error
            .downcast_ref::<ProviderError>()
            .expect("typed provider error");
        assert_eq!(
            typed.kind,
            crate::providers::ProviderErrorKind::QuotaExhausted
        );
        assert_eq!(typed.resets_at, Some(reset));
        let message = error.to_string();
        assert!(
            // "resets at 14:00" or, past midnight, "resets on Nov 8 at 00:00".
            message.contains("usage limit has been reached; it resets "),
            "{message}"
        );
        assert!(
            message.contains("fallback providers also failed"),
            "{message}"
        );
        assert!(message.contains("billing"), "{message}");
    }

    #[tokio::test]
    async fn single_attempt_budget_fence_never_retries_or_falls_back() {
        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("build test harness");
        let goal = Goal::new_continuous(
            "Single-attempt mandate provider call",
            "owner-single-attempt",
            Some(100),
            Some(1_000),
        );
        let mandate = Mandate::new(
            &goal.id,
            None,
            "Never retry ambiguous provider spend",
            &goal.session_id,
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        harness
            .state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let run = harness
            .state
            .start_goal_run(&goal.id, "mandate", None, None)
            .await
            .unwrap();
        let task_id = uuid::Uuid::new_v4().to_string();
        harness
            .state
            .create_task(&Task {
                id: task_id.clone(),
                goal_id: goal.id.clone(),
                description: "Own the synthetic mandate run".to_string(),
                status: "pending".to_string(),
                priority: "high".to_string(),
                task_order: 0,
                parallel_group: None,
                depends_on: None,
                agent_id: None,
                context: None,
                result: None,
                error: None,
                blocker: None,
                idempotent: false,
                retry_count: 0,
                max_retries: 0,
                created_at: chrono::Utc::now().to_rfc3339(),
                started_at: None,
                completed_at: None,
            })
            .await
            .unwrap();
        harness
            .state
            .claim_task_with_lease(&task_id, "budget-test-worker", None, 300)
            .await
            .unwrap()
            .expect("synthetic mandate task claim");
        harness
            .state
            .ensure_mandate_run_token_budget(&run.id, &mandate.id, mandate.version, 100)
            .await
            .unwrap();
        assert!(
            harness
                .state
                .try_acquire_mandate_run_token_lease(
                    &run.id,
                    &mandate.id,
                    mandate.version,
                    "one-provider-attempt",
                    60,
                )
                .await
                .unwrap()
                .0
        );
        assert!(harness
            .state
            .mark_mandate_run_token_lease_dispatched(
                &run.id,
                &mandate.id,
                mandate.version,
                "one-provider-attempt",
            )
            .await
            .unwrap());
        let provider = Arc::new(ScriptedProvider::with_results(vec![
            Err(ProviderError::from_status(500, "ambiguous provider failure").into()),
            Ok(MockProvider::text_response("retry must not run")),
        ]));
        let options = ChatOptions {
            max_tokens_override: Some(100),
            single_attempt_fail_closed: true,
            ..ChatOptions::default()
        };
        let mut telemetry = super::LlmCallTelemetry::default();

        let result = harness
            .agent
            .call_llm_with_recovery(
                provider.clone() as crate::llm_runtime::ModelBackend,
                None,
                "primary-model",
                &[],
                &[],
                &options,
                &mut telemetry,
                None,
                std::time::Duration::from_secs(60),
            )
            .await;

        assert!(result.is_err());
        assert_eq!(provider.models_called().await.len(), 1);
        let reserved_after_error: i64 = sqlx::query_scalar(
            "SELECT tokens_used FROM mandate_run_token_budgets WHERE goal_run_id = ?",
        )
        .bind(&run.id)
        .fetch_one(&harness.state.pool())
        .await
        .unwrap();
        assert_eq!(reserved_after_error, 100);
    }

    #[tokio::test]
    async fn single_attempt_budget_fence_does_not_cascade_cached_billing_failure() {
        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("build test harness");
        let primary = Arc::new(ScriptedProvider::with_results(vec![Ok(
            MockProvider::text_response("primary must not run while cached"),
        )]));
        let alternate = Arc::new(ScriptedProvider::with_results(vec![Ok(
            MockProvider::text_response("fallback must not run"),
        )]));
        harness.agent.llm_runtime.swap(
            primary.clone() as crate::llm_runtime::ModelBackend,
            None,
            crate::config::ProviderKind::OpenaiCompatible,
            "primary-model".to_string(),
            vec![ProviderRuntimeTarget::new(
                alternate.clone() as crate::llm_runtime::ModelBackend,
                None,
                crate::config::ProviderKind::Anthropic,
                "secondary-model".to_string(),
            )],
            crate::providers::sdk_backed::ModelList::Fixed(Vec::new()),
        );
        harness
            .agent
            .billing_failed_models
            .write()
            .await
            .insert("primary-model".to_string(), Instant::now());
        let options = ChatOptions {
            max_tokens_override: Some(100),
            single_attempt_fail_closed: true,
            ..ChatOptions::default()
        };
        let mut telemetry = super::LlmCallTelemetry::default();

        let result = harness
            .agent
            .call_llm_with_recovery(
                primary.clone() as crate::llm_runtime::ModelBackend,
                None,
                "primary-model",
                &[],
                &[],
                &options,
                &mut telemetry,
                None,
                std::time::Duration::from_secs(60),
            )
            .await;

        assert!(result.is_err());
        assert!(primary.models_called().await.is_empty());
        assert!(alternate.models_called().await.is_empty());
    }
}
