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
        let stored = self.fallback_model.read().await.clone();
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

    /// Try up to 2 different fallback models after retries are exhausted.
    /// On success, returns the response for this call only (no persistent model downgrade).
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

        for attempt in 1..=2 {
            let exclude_refs: Vec<&str> = tried.iter().map(|s| s.as_str()).collect();
            let fallback = match self
                .pick_fallback_excluding(failed_model, &exclude_refs, router)
                .await
            {
                Some(f) => f,
                None => break, // no more candidates
            };

            warn!(
                fallback = %fallback,
                attempt,
                "Cascade fallback attempt"
            );

            match provider
                .chat_with_options(&fallback, messages, tool_defs, options)
                .await
            {
                Ok(resp) => return Ok(resp),
                Err(_) => {
                    tried.push(fallback);
                }
            }
        }

        Err(anyhow::anyhow!("{}", last_err.user_message()))
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
                    return Err(anyhow::anyhow!("{}", retry_provider_err.user_message()));
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
