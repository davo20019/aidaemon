//! A turn's model calls with the daemon's recovery live in `aidaemon-runtime`
//! ([`aidaemon_runtime::llm::turn_calls`]).

pub(in crate::agent) use aidaemon_runtime::llm::turn_calls::*;
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

    /// A test agent whose side calls go through a gateway over `provider`,
    /// and the event store that gateway records into.
    async fn agent_with(
        provider: crate::llm_runtime::ModelBackend,
    ) -> (crate::testing::TestHarness, Arc<crate::events::EventStore>) {
        let mut harness = setup_test_agent(MockProvider::new())
            .await
            .expect("build test harness");
        let (gateway, events) =
            crate::llm_runtime::gateway::test_support::gateway_with(provider).await;
        harness.agent.side_calls = gateway;
        (harness, events)
    }

    #[tokio::test]
    async fn interpretation_is_recorded_and_falls_back_to_none_on_failure() {
        use crate::llm_runtime::gateway::test_support::{
            failed_calls, ok_calls, provider_answering, provider_failing,
        };

        let (harness, events) = agent_with(provider_answering("12 files matched.")).await;
        assert_eq!(
            harness
                .agent
                .interpret_background_result("rg -c TODO", "12")
                .await
                .as_deref(),
            Some("12 files matched.")
        );
        assert_eq!(ok_calls(&events, "background_interpretation").await, 1);

        let (harness, events) = agent_with(provider_failing("503")).await;
        assert_eq!(
            harness
                .agent
                .interpret_background_result("rg -c TODO", "12")
                .await,
            None
        );
        assert!(failed_calls(&events, "background_interpretation").await >= 1);
        assert_eq!(ok_calls(&events, "background_interpretation").await, 0);
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
        let response = super::call_llm_with_recovery(
            &harness.agent,
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
        let resp = super::call_llm_with_recovery(
            &harness.agent,
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
        let response = super::call_llm_with_recovery(
            &harness.agent,
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
        let error = super::call_llm_with_recovery(
            &harness.agent,
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

        let result = super::call_llm_with_recovery(
            &harness.agent,
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

        let result = super::call_llm_with_recovery(
            &harness.agent,
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

    /// A budget-fenced call (one physical attempt per model) still reaches
    /// the configured fallback when the primary refused before doing any
    /// work; an error that may have been paid for still ends the call.
    #[tokio::test]
    async fn fenced_call_falls_back_only_past_an_unspent_refusal() {
        async fn run(primary_error: ProviderError) -> (bool, usize, usize, bool) {
            let harness = setup_test_agent(MockProvider::new())
                .await
                .expect("build test harness");
            // The primary account's other models answer the same way.
            let primary = Arc::new(ScriptedProvider::with_results(
                (0..8)
                    .map(|_| Err(primary_error.clone().into()))
                    .collect::<Vec<_>>(),
            ));
            let alternate = Arc::new(ScriptedProvider::with_results(vec![Ok(
                MockProvider::text_response("from the fallback"),
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
            let options = ChatOptions {
                max_tokens_override: Some(100),
                single_attempt_fail_closed: true,
                fallback_when_unspent: true,
                ..ChatOptions::default()
            };
            let mut telemetry = super::LlmCallTelemetry::default();
            let result = super::call_llm_with_recovery(
                &harness.agent,
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
            let spent_nothing = result
                .as_ref()
                .err()
                .and_then(|error| error.downcast_ref::<ProviderError>())
                .is_some_and(|error| error.spent_nothing);
            (
                result.is_ok(),
                primary.models_called().await.len().min(1),
                alternate.models_called().await.len(),
                spent_nothing,
            )
        }

        let mut quota = ProviderError::from_status(429, "usage limit reached");
        quota.kind = crate::providers::ProviderErrorKind::QuotaExhausted;
        assert_eq!(run(quota).await, (true, 1, 1, false));
        assert_eq!(
            run(ProviderError::from_status(401, "bad key")).await,
            (true, 1, 1, false)
        );
        let (ok, primary, alternate, spent_nothing) =
            run(ProviderError::from_status(503, "overloaded")).await;
        assert_eq!((ok, primary, alternate), (false, 1, 0));
        assert!(!spent_nothing, "a server error may have been paid for");
    }

    /// Mandate reviews are postponed only while no configured model can
    /// answer: one used-up account is not enough when another can.
    #[tokio::test]
    async fn no_model_can_answer_only_when_every_account_is_cooling_down() {
        use aidaemon_core::Provider as _;
        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("build test harness");
        let used_up = |reset: std::time::SystemTime| {
            Arc::new(aidaemon_core::QuotaGuard::new(
                aidaemon_core::testing::ScriptedProvider::with_results(vec![Err(
                    aidaemon_core::ProviderError::QuotaExhausted {
                        message: "limit".into(),
                        resets_at: Some(reset),
                        per_model: false,
                        in_stream: false,
                    },
                )]),
            ))
        };
        let soon = std::time::SystemTime::now() + std::time::Duration::from_secs(600);
        let later = soon + std::time::Duration::from_secs(600);
        let primary = used_up(later);
        let alternate = used_up(soon);
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
        let request = |model: &str| aidaemon_core::ModelRequest {
            model: model.into(),
            ..Default::default()
        };
        assert_eq!(harness.agent.next_model_answer_at(), None);
        let _ = primary.complete(&request("primary-model")).await;
        assert_eq!(
            harness.agent.next_model_answer_at(),
            None,
            "the alternate can"
        );
        let _ = alternate.complete(&request("secondary-model")).await;
        let at = harness.agent.next_model_answer_at().expect("nobody can");
        assert!(at <= soon && at > soon - std::time::Duration::from_secs(5));
    }
}
