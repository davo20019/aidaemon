use super::*;

impl Agent {
    fn should_run_graduation_check(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let last = self.last_graduation_check_epoch.load(Ordering::Relaxed);
        if now.saturating_sub(last) < 3600 {
            return false;
        }
        self.last_graduation_check_epoch
            .compare_exchange(last, now, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    pub(super) async fn maybe_retire_classify_query(&self, session_id: &str) {
        if !self.policy_config.classify_retirement_enabled {
            return;
        }
        if self.classify_query_retired.load(Ordering::Relaxed) {
            return;
        }
        if !self.should_run_graduation_check() {
            return;
        }
        let report = match self
            .event_store
            .policy_graduation_report(self.policy_config.classify_retirement_window_days)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!(session_id, error = %e, "Failed policy graduation check");
                return;
            }
        };

        let max_divergence = self.policy_config.classify_retirement_max_divergence as f64;
        let passed = report.gate_passes(max_divergence);
        info!(
            session_id,
            observed_days = report.observed_days,
            window_days = report.window_days,
            divergence_rate = report.divergence_rate,
            completion_rate_current = report.current.completion_rate,
            completion_rate_previous = report.previous.completion_rate,
            error_rate_current = report.current.error_rate,
            error_rate_previous = report.previous.error_rate,
            stall_rate_current = report.current.stall_rate,
            stall_rate_previous = report.previous.stall_rate,
            passed,
            "Policy graduation evaluation"
        );
        if passed {
            self.classify_query_retired.store(true, Ordering::Relaxed);
            info!(
                session_id,
                "Policy graduation gate passed - classify_query() retired for routing"
            );
        }
    }

    /// Pick a fallback model, skipping `failed_model` and any models in the `exclude` list.
    /// Tries stored fallback first, then cycles through router tiers.
    pub(super) async fn pick_fallback_excluding(
        &self,
        failed_model: &str,
        exclude: &[&str],
    ) -> Option<String> {
        let stored = self.fallback_model.read().await.clone();
        if stored != failed_model && !exclude.contains(&stored.as_str()) {
            return Some(stored);
        }
        // Stored fallback is the same or excluded — try the router tiers
        if let Some(ref router) = self.router {
            for tier in &[
                crate::router::Tier::Primary,
                crate::router::Tier::Smart,
                crate::router::Tier::Fast,
            ] {
                let candidate = router.select(*tier).to_string();
                if candidate != failed_model && !exclude.contains(&candidate.as_str()) {
                    return Some(candidate);
                }
            }
        }
        None
    }

    /// Try up to 2 different fallback models after retries are exhausted.
    /// On success, switches the active model.
    async fn cascade_fallback(
        &self,
        failed_model: &str,
        messages: &[Value],
        tool_defs: &[Value],
        last_err: &ProviderError,
    ) -> anyhow::Result<crate::traits::ProviderResponse> {
        let mut tried: Vec<String> = vec![failed_model.to_string()];

        for attempt in 1..=2 {
            let exclude_refs: Vec<&str> = tried.iter().map(|s| s.as_str()).collect();
            let fallback = match self
                .pick_fallback_excluding(failed_model, &exclude_refs)
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

            match self.provider.chat(&fallback, messages, tool_defs).await {
                Ok(resp) => {
                    *self.model.write().await = fallback;
                    self.stamp_lastgood().await;
                    return Ok(resp);
                }
                Err(_) => {
                    tried.push(fallback);
                }
            }
        }

        Err(anyhow::anyhow!("{}", last_err.user_message()))
    }

    /// Attempt an LLM call with error-classified recovery:
    /// - RateLimit → exponential backoff retries, then cascade fallback
    /// - Timeout/Network/ServerError → exponential backoff retries, then cascade fallback
    /// - NotFound → cascade fallback immediately
    /// - Auth/Billing → return user-facing error immediately
    pub(super) async fn call_llm_with_recovery(
        &self,
        model: &str,
        messages: &[Value],
        tool_defs: &[Value],
    ) -> anyhow::Result<crate::traits::ProviderResponse> {
        match self.provider.chat(model, messages, tool_defs).await {
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

                match provider_err.kind {
                    // --- Non-retryable: tell the user, stop ---
                    ProviderErrorKind::Auth
                    | ProviderErrorKind::Billing
                    | ProviderErrorKind::BadRequest => {
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
                            match self.provider.chat(model, messages, tool_defs).await {
                                Ok(resp) => {
                                    self.stamp_lastgood().await;
                                    return Ok(resp);
                                }
                                Err(_) => continue,
                            }
                        }
                        // All retries exhausted — cascade through fallback models
                        warn!("Rate limit retries exhausted, trying cascade fallback");
                        self.cascade_fallback(model, messages, tool_defs, &provider_err)
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
                            match self.provider.chat(model, messages, tool_defs).await {
                                Ok(resp) => {
                                    self.stamp_lastgood().await;
                                    return Ok(resp);
                                }
                                Err(_) => continue,
                            }
                        }
                        // All retries exhausted — cascade through fallback models
                        warn!("Transient error retries exhausted, trying cascade fallback");
                        self.cascade_fallback(model, messages, tool_defs, &provider_err)
                            .await
                    }

                    // --- NotFound (bad model name): cascade fallback immediately ---
                    ProviderErrorKind::NotFound => {
                        warn!(
                            bad_model = model,
                            "Model not found, trying cascade fallback"
                        );
                        self.cascade_fallback(model, messages, tool_defs, &provider_err)
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
