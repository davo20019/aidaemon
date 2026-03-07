use super::*;

fn malformed_reason_label(
    reason: Option<crate::providers::MalformedResponseReason>,
) -> &'static str {
    match reason {
        Some(crate::providers::MalformedResponseReason::Parse) => "parse",
        Some(crate::providers::MalformedResponseReason::Shape) => "shape",
        None => "unknown",
    }
}

impl Agent {
    /// Pick a fallback model, skipping `failed_model` and any models in the `exclude` list.
    /// Tries stored fallback first, then cycles through router ordered models
    /// (`default` + `fallback[]`).
    pub(super) async fn pick_fallback_excluding(
        &self,
        failed_model: &str,
        exclude: &[&str],
        router: Option<&Router>,
    ) -> Option<String> {
        let stored =
            match tokio::time::timeout(Duration::from_secs(2), self.fallback_model.read()).await {
                Ok(guard) => guard.clone(),
                Err(_) => {
                    warn!("Timed out acquiring fallback_model lock while picking fallback");
                    self.llm_runtime.snapshot().primary_model()
                }
            };
        if stored != failed_model && !exclude.contains(&stored.as_str()) {
            return Some(stored);
        }
        // Stored fallback is the same or excluded — try router candidates.
        if let Some(router) = router {
            for candidate in router.all_models_ordered() {
                if candidate != failed_model && !exclude.contains(&candidate.as_str()) {
                    return Some(candidate);
                }
            }
        }
        None
    }

    /// Try provider-local model fallbacks first, then configured alternate
    /// providers and their model chains. On success, returns the response for
    /// this call only (no persistent provider/model downgrade).
    #[allow(clippy::too_many_arguments)]
    async fn cascade_fallback(
        &self,
        provider: &Arc<dyn ModelProvider>,
        router: Option<&Router>,
        failed_model: &str,
        messages: &[Value],
        tool_defs: &[Value],
        options: &ChatOptions,
        last_err: &ProviderError,
    ) -> anyhow::Result<crate::traits::ProviderResponse> {
        let mut tried: Vec<String> = vec![failed_model.to_string()];
        let mut final_err = last_err.clone();
        let mut attempt = 1;
        let runtime_snapshot = self.llm_runtime.snapshot();
        let primary_provider_kind = runtime_snapshot.provider_kind();
        let failover_targets = runtime_snapshot.failover_targets();

        loop {
            let exclude_refs: Vec<&str> = tried.iter().map(|s| s.as_str()).collect();
            let fallback = match self
                .pick_fallback_excluding(failed_model, &exclude_refs, router)
                .await
            {
                Some(fallback_model) => fallback_model,
                None => break,
            };

            warn!(
                provider_kind = ?primary_provider_kind,
                model = %fallback,
                attempt,
                "Primary-provider fallback attempt"
            );

            match provider
                .chat_with_options(&fallback, messages, tool_defs, options)
                .await
            {
                Ok(resp) => {
                    self.stamp_lastgood().await;
                    return Ok(resp);
                }
                Err(retry_err) => match retry_err.downcast::<ProviderError>() {
                    Ok(provider_err) => {
                        final_err = provider_err;
                        tried.push(fallback);
                        attempt += 1;
                    }
                    Err(other) => return Err(other),
                },
            }
        }

        for target in failover_targets {
            for model in target.all_models_ordered() {
                warn!(
                    provider_kind = ?target.provider_kind(),
                    model = %model,
                    attempt,
                    "Provider failover attempt"
                );

                match target
                    .provider()
                    .chat_with_options(&model, messages, tool_defs, options)
                    .await
                {
                    Ok(resp) => {
                        self.stamp_lastgood().await;
                        return Ok(resp);
                    }
                    Err(retry_err) => match retry_err.downcast::<ProviderError>() {
                        Ok(provider_err) => {
                            final_err = provider_err;
                            attempt += 1;
                        }
                        Err(other) => return Err(other),
                    },
                }
            }
        }

        Err(anyhow::anyhow!("{}", final_err.recovery_failed_message()))
    }

    #[allow(clippy::too_many_arguments)]
    async fn retry_malformed_response<F>(
        &self,
        provider: &Arc<dyn ModelProvider>,
        model: &str,
        messages: &[Value],
        tool_defs: &[Value],
        options: &ChatOptions,
        provider_label: &str,
        max_retries: u32,
        wait_for_attempt: F,
        retry_strategy: &str,
    ) -> anyhow::Result<Option<crate::traits::ProviderResponse>>
    where
        F: Fn(u32) -> u64,
    {
        for attempt in 0..max_retries {
            let wait = wait_for_attempt(attempt);
            info!(
                wait_secs = wait,
                attempt = attempt + 1,
                max = max_retries,
                strategy = retry_strategy,
                "Retrying after malformed provider response"
            );
            tokio::time::sleep(Duration::from_secs(wait)).await;
            match provider
                .chat_with_options(model, messages, tool_defs, options)
                .await
            {
                Ok(resp) => {
                    self.stamp_lastgood().await;
                    return Ok(Some(resp));
                }
                Err(retry_err) => {
                    let retry_provider_err = match retry_err.downcast::<ProviderError>() {
                        Ok(pe) => pe,
                        Err(other) => return Err(other),
                    };
                    if retry_provider_err.kind == ProviderErrorKind::MalformedResponse {
                        let retry_reason =
                            malformed_reason_label(retry_provider_err.malformed_reason);
                        record_llm_payload_invalid_metric(provider_label, model, retry_reason);
                        continue;
                    }
                    return Err(anyhow::anyhow!(
                        "{}",
                        retry_provider_err.recovery_failed_message()
                    ));
                }
            }
        }

        Ok(None)
    }

    /// Attempt an LLM call with error-classified recovery:
    /// - RateLimit → exponential backoff retries, then cascade fallback
    /// - Timeout/Network/ServerError → exponential backoff retries, then cascade fallback
    /// - MalformedResponse(Parse) → exponential backoff retries, then cascade fallback
    /// - MalformedResponse(Shape/Unknown) → single retry (no cascade fallback)
    /// - NotFound → cascade fallback immediately
    /// - Auth/Billing → return user-facing error immediately
    ///
    /// Cascade fallback now means:
    /// 1. Other models on the current provider
    /// 2. Configured failover providers and their model chains
    pub(super) async fn call_llm_with_recovery(
        &self,
        provider: Arc<dyn ModelProvider>,
        router: Option<Router>,
        model: &str,
        messages: &[Value],
        tool_defs: &[Value],
        options: &ChatOptions,
    ) -> anyhow::Result<crate::traits::ProviderResponse> {
        match provider
            .chat_with_options(model, messages, tool_defs, options)
            .await
        {
            Ok(resp) => {
                // Config works — stamp as last known good (best-effort, non-blocking)
                self.stamp_lastgood().await;
                Ok(resp)
            }
            Err(e) => {
                // Try to downcast to our classified ProviderError
                let provider_err = match e.downcast::<ProviderError>() {
                    Ok(pe) => pe,
                    Err(other) => return Err(other), // not a provider error, propagate
                };

                warn!(
                    kind = ?provider_err.kind,
                    status = ?provider_err.status,
                    "LLM call failed: {}",
                    provider_err
                );
                let provider_label =
                    provider_kind_metric_label(self.llm_runtime.snapshot().provider_kind());

                match provider_err.kind {
                    // --- Non-retryable: tell the user, stop ---
                    ProviderErrorKind::Auth | ProviderErrorKind::Billing => {
                        Err(anyhow::anyhow!("{}", provider_err.user_message()))
                    }
                    ProviderErrorKind::BadRequest => {
                        if *options != ChatOptions::default() {
                            warn!(
                                model,
                                response_mode = ?options.response_mode,
                                tool_choice = ?options.tool_choice,
                                "Provider rejected advanced chat options; retrying once with default options"
                            );
                            match provider.chat(model, messages, tool_defs).await {
                                Ok(resp) => {
                                    self.stamp_lastgood().await;
                                    return Ok(resp);
                                }
                                Err(retry_err) => {
                                    warn!(
                                        model,
                                        error = %retry_err,
                                        "Fallback retry without advanced options also failed"
                                    );
                                }
                            }
                        }
                        Err(anyhow::anyhow!("{}", provider_err.user_message()))
                    }

                    // --- Rate limit: exponential backoff, then cascade fallback ---
                    ProviderErrorKind::RateLimit => {
                        let base_wait = provider_err.retry_after_secs.unwrap_or(5);
                        for attempt in 0..Self::MAX_LLM_RETRIES {
                            let wait = (base_wait * 2u64.pow(attempt)).min(120); // cap at 120s
                            info!(
                                wait_secs = wait,
                                attempt = attempt + 1,
                                max = Self::MAX_LLM_RETRIES,
                                "Rate limited, waiting before retry"
                            );
                            tokio::time::sleep(Duration::from_secs(wait)).await;
                            match provider
                                .chat_with_options(model, messages, tool_defs, options)
                                .await
                            {
                                Ok(resp) => {
                                    self.stamp_lastgood().await;
                                    return Ok(resp);
                                }
                                Err(_) => continue,
                            }
                        }
                        // All retries exhausted — cascade through fallback models
                        warn!("Rate limit retries exhausted, trying cascade fallback");
                        self.cascade_fallback(
                            &provider,
                            router.as_ref(),
                            model,
                            messages,
                            tool_defs,
                            options,
                            &provider_err,
                        )
                        .await
                    }

                    // --- Timeout / Network / Server: exponential backoff, then cascade ---
                    ProviderErrorKind::Timeout
                    | ProviderErrorKind::Network
                    | ProviderErrorKind::ServerError => {
                        for attempt in 0..Self::MAX_LLM_RETRIES {
                            let wait = Self::RETRY_BASE_DELAY_SECS * 2u64.pow(attempt); // 2s, 4s, 8s
                            info!(
                                wait_secs = wait,
                                attempt = attempt + 1,
                                max = Self::MAX_LLM_RETRIES,
                                "Retrying after transient error"
                            );
                            tokio::time::sleep(Duration::from_secs(wait)).await;
                            match provider
                                .chat_with_options(model, messages, tool_defs, options)
                                .await
                            {
                                Ok(resp) => {
                                    self.stamp_lastgood().await;
                                    return Ok(resp);
                                }
                                Err(_) => continue,
                            }
                        }
                        // All retries exhausted — cascade through fallback models
                        warn!("Transient error retries exhausted, trying cascade fallback");
                        self.cascade_fallback(
                            &provider,
                            router.as_ref(),
                            model,
                            messages,
                            tool_defs,
                            options,
                            &provider_err,
                        )
                        .await
                    }

                    // --- Malformed payload: reason-aware recovery ---
                    ProviderErrorKind::MalformedResponse => {
                        let reason = malformed_reason_label(provider_err.malformed_reason);
                        record_llm_payload_invalid_metric(provider_label, model, reason);

                        if provider_err.malformed_reason
                            == Some(crate::providers::MalformedResponseReason::Parse)
                        {
                            // Parse failures can be transient (gateway/proxy/body corruption).
                            // Use the same resilient policy as other transient provider failures.
                            let retry_result = self
                                .retry_malformed_response(
                                    &provider,
                                    model,
                                    messages,
                                    tool_defs,
                                    options,
                                    provider_label,
                                    Self::MAX_LLM_RETRIES,
                                    |attempt| Self::RETRY_BASE_DELAY_SECS * 2u64.pow(attempt),
                                    "parse_exponential_backoff",
                                )
                                .await?;
                            if let Some(resp) = retry_result {
                                return Ok(resp);
                            }
                            warn!("Malformed parse retries exhausted, trying cascade fallback");
                            return self
                                .cascade_fallback(
                                    &provider,
                                    router.as_ref(),
                                    model,
                                    messages,
                                    tool_defs,
                                    options,
                                    &provider_err,
                                )
                                .await;
                        }

                        // Shape/unknown malformed responses are likely deterministic schema issues.
                        // Retry once quickly, then fail fast without model cascade.
                        let retry_result = self
                            .retry_malformed_response(
                                &provider,
                                model,
                                messages,
                                tool_defs,
                                options,
                                provider_label,
                                Self::MAX_MALFORMED_PAYLOAD_RETRIES,
                                |_| Self::MALFORMED_PAYLOAD_RETRY_DELAY_SECS,
                                "shape_or_unknown_single_retry",
                            )
                            .await?;
                        if let Some(resp) = retry_result {
                            return Ok(resp);
                        }

                        Err(anyhow::anyhow!("{}", provider_err.user_message()))
                    }

                    // --- NotFound (bad model name): cascade fallback immediately ---
                    ProviderErrorKind::NotFound => {
                        warn!(
                            bad_model = model,
                            "Model not found, trying cascade fallback"
                        );
                        self.cascade_fallback(
                            &provider,
                            router.as_ref(),
                            model,
                            messages,
                            tool_defs,
                            options,
                            &provider_err,
                        )
                        .await
                    }

                    // --- Unknown: propagate ---
                    ProviderErrorKind::Unknown => {
                        Err(anyhow::anyhow!("{}", provider_err.user_message()))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use serde_json::Value;
    use tokio::sync::Mutex;

    use crate::llm_runtime::ProviderRuntimeTarget;
    use crate::providers::ProviderError;
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::traits::{ChatOptions, ModelProvider, ProviderResponse};

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

    #[async_trait]
    impl ModelProvider for ScriptedProvider {
        async fn chat(
            &self,
            model: &str,
            messages: &[Value],
            tools: &[Value],
        ) -> anyhow::Result<ProviderResponse> {
            self.chat_with_options(model, messages, tools, &ChatOptions::default())
                .await
        }

        async fn chat_with_options(
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

        async fn list_models(&self) -> anyhow::Result<Vec<String>> {
            Ok(vec!["scripted-model".to_string()])
        }
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
            primary_provider.clone() as Arc<dyn ModelProvider>,
            None,
            crate::config::ProviderKind::OpenaiCompatible,
            "primary-model".to_string(),
            vec![ProviderRuntimeTarget::new(
                alternate_provider.clone() as Arc<dyn ModelProvider>,
                None,
                crate::config::ProviderKind::Anthropic,
                "secondary-model".to_string(),
            )],
        );

        let resp = harness
            .agent
            .call_llm_with_recovery(
                primary_provider as Arc<dyn ModelProvider>,
                None,
                "primary-model",
                &[],
                &[],
                &ChatOptions::default(),
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
    }
}
