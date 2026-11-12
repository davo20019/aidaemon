//! The untrusted-content rule, daemon side: the pure rule lives in
//! `aidaemon-runtime`; the agent supplies a mandate run's origin and settles
//! a turn's interruptions with the verifier.

use tracing::warn;

use super::Agent;
pub(crate) use aidaemon_runtime::untrusted::*;

impl Agent {
    /// The confirmed mandate as this run's owner request; `None` outside a
    /// mandate run. A run whose fence no longer matches an owner-confirmed
    /// policy has no owner request at all, so its egress is refused.
    pub(in crate::agent) async fn mandate_origin(&self) -> Option<MandateOrigin> {
        let fence = self.mandate_execution.as_ref()?;
        let mandate = match self.state.get_mandate(&fence.mandate_id).await {
            Ok(mandate) => mandate,
            Err(error) => {
                warn!(mandate_id = %fence.mandate_id, %error, "Mandate unreadable for the egress check");
                None
            }
        };
        Some(
            match mandate.filter(|mandate| {
                mandate.version == fence.mandate_version
                    && mandate.authority == fence.authority
                    && mandate.confirmed_at.is_some()
            }) {
                Some(mandate) => MandateOrigin {
                    request: crate::mandates::verifier_request(&mandate),
                    trusted: true,
                },
                None => MandateOrigin {
                    request: format!("Stale or unconfirmed mandate {}.", fence.mandate_id),
                    trusted: false,
                },
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use serde_json::json;

    fn quota_exhausted() -> crate::providers::ProviderError {
        let mut error = crate::providers::ProviderError::from_status(429, "plan limit");
        error.kind = crate::providers::ProviderErrorKind::QuotaExhausted;
        error
    }

    /// A harness whose primary fails with `primary_error` and whose failover
    /// provider has `failover` to answer with, for checking who was asked.
    async fn verify_with_primary_error(
        primary_error: crate::providers::ProviderError,
        failover: Vec<crate::traits::ProviderResponse>,
    ) -> (VerifierVerdict, Vec<String>) {
        use crate::llm_runtime::{ModelBackend, ProviderRuntimeTarget};
        use crate::testing::MockProvider;
        let harness = crate::testing::setup_test_agent(MockProvider::new())
            .await
            .unwrap();
        let primary: ModelBackend =
            Arc::new(MockProvider::new().with_leading_provider_errors(vec![primary_error]));
        let second = Arc::new(MockProvider::with_responses(failover));
        harness.agent.llm_runtime.swap(
            primary,
            None,
            crate::config::ProviderKind::OpenaiChatgpt,
            "plan-model".to_string(),
            vec![ProviderRuntimeTarget::new(
                second.clone() as ModelBackend,
                None,
                crate::config::ProviderKind::Anthropic,
                "failover-model".to_string(),
            )],
            crate::providers::sdk_backed::ModelList::Fixed(Vec::new()),
        );
        let verdict = verify_untrusted_call(
            &harness.agent.side_calls,
            "Send Jordan Lee the summary.",
            true,
            "send_probe",
            "Sends a synthetic message.",
            &json!({"to": "jordan@example.invalid"}),
            &tokio_util::sync::CancellationToken::new(),
        )
        .await;
        let asked = second
            .call_log
            .lock()
            .await
            .iter()
            .map(|call| call.model.clone())
            .collect();
        (verdict, asked)
    }

    /// A provider that cannot serve must not disable the check while another
    /// provider can still judge the call.
    /// A fence with no matching owner-confirmed policy has no owner request.
    #[tokio::test]
    async fn a_mandate_run_without_a_confirmed_policy_has_no_owner_request() {
        use crate::testing::MockProvider;
        let mut harness = crate::testing::setup_test_agent(MockProvider::new())
            .await
            .unwrap();
        assert!(harness.agent.mandate_origin().await.is_none());
        let attempt: crate::traits::TaskAttempt = serde_json::from_value(json!({
            "id": "attempt-synthetic",
            "task_id": "task-synthetic",
            "goal_run_id": "run-synthetic",
            "worker_profile_id": null,
            "worker_instance_id": "worker-synthetic",
            "lease_token": "lease-synthetic",
            "status": "running",
            "lease_expires_at": "2026-11-10T00:00:00Z",
            "last_heartbeat_at": "2026-11-10T00:00:00Z",
            "workspace_id": null,
            "started_at": "2026-11-10T00:00:00Z",
            "completed_at": null,
        }))
        .unwrap();
        harness.agent.set_test_mandate_execution(
            "mandate-that-does-not-exist",
            1,
            crate::traits::MandateAuthority::default(),
            "goal-synthetic",
            "root-synthetic",
            "root-attempt-synthetic",
            &attempt,
        );

        let origin = harness.agent.mandate_origin().await.unwrap();

        assert!(!origin.trusted);
    }

    #[tokio::test]
    async fn the_verifier_moves_to_a_failover_provider_when_the_plan_is_out() {
        use crate::testing::MockProvider;
        let (verdict, asked) = verify_with_primary_error(
            quota_exhausted(),
            vec![MockProvider::text_response(
                r#"{"consistent":true,"reason":"asked"}"#,
            )],
        )
        .await;
        assert!(verdict.consistent, "{verdict:?}");
        assert_eq!(asked, ["failover-model"]);
    }

    /// The failover provider's own verdict stands: moving providers never
    /// turns a refusal into an approval.
    #[tokio::test]
    async fn a_failover_refusal_still_refuses() {
        use crate::testing::MockProvider;
        let (verdict, _) = verify_with_primary_error(
            quota_exhausted(),
            vec![MockProvider::text_response(
                r#"{"consistent":false,"reason":"not asked"}"#,
            )],
        )
        .await;
        assert!(!verdict.consistent);
        assert_eq!(verdict.reason, "not asked");
    }

    /// With every provider down the check still fails closed.
    #[tokio::test]
    async fn no_provider_left_is_could_not_verify() {
        let (verdict, asked) = verify_with_primary_error(quota_exhausted(), vec![]).await;
        assert_eq!(verdict, could_not_verify());
        assert_eq!(asked, ["failover-model"]);
    }

    /// A failure tied to this request stays on the primary, so the content
    /// being judged cannot steer the check onto another model.
    #[tokio::test]
    async fn a_request_failure_does_not_change_the_judge() {
        use crate::testing::MockProvider;
        let (verdict, asked) = verify_with_primary_error(
            crate::providers::ProviderError::from_status(400, "rejected"),
            vec![MockProvider::text_response(
                r#"{"consistent":true,"reason":"asked"}"#,
            )],
        )
        .await;
        assert_eq!(verdict, could_not_verify());
        assert!(asked.is_empty(), "{asked:?}");
    }

    /// A turn cancelled while the verifier is deciding stops waiting at
    /// once and the call is refused.
    #[tokio::test]
    async fn a_cancelled_turn_stops_waiting_for_the_verifier() {
        use crate::testing::MockProvider;
        let harness = crate::testing::setup_test_agent(MockProvider::with_delayed_responses(
            vec![MockProvider::text_response(
                r#"{"consistent":true,"reason":"asked"}"#,
            )],
            vec![Duration::from_secs(60)],
        ))
        .await
        .unwrap();
        let cancel = tokio_util::sync::CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });
        let started = std::time::Instant::now();
        let verdict = verify_untrusted_call(
            &harness.agent.side_calls,
            "Send Jordan Lee the summary.",
            true,
            "send_probe",
            "Sends a synthetic message.",
            &json!({"to": "jordan@example.invalid"}),
            &cancel,
        )
        .await;
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "returns promptly"
        );
        assert_eq!(
            verdict,
            VerifierVerdict {
                consistent: false,
                reason: "cancelled".to_string(),
                decided: false,
            }
        );
    }
}
