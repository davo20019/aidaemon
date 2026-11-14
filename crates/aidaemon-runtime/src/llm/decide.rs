//! `SideCallGateway::decide`: the one door to a decision backend. It skips
//! a decision that is off, bounds the backend by the decision's deadline,
//! records every call, and returns answers only in the form the decision's
//! spec allows the caller to act on.

use std::sync::Arc;
use std::time::Duration;

#[cfg(any(test, feature = "testing"))]
use aidaemon_decisions::DecisionResponse;
use aidaemon_decisions::{Decider, DecisionRequest, ProviderDecider};
use serde_json::json;
use tokio::time::Instant;

use super::decisions::{Decision, DecisionOutcome, DefaultReason, GatedResponse};
use super::gateway::SideCallGateway;
use super::side_calls::SideCall;
use crate::settings::{DecisionBackendKind, DecisionMode, DecisionModes, DecisionsConfig};

/// Which backend answers.
#[derive(Clone)]
enum Backend {
    /// A chat model through the side-call door; `model` empty means the
    /// configured cheap model at the time of the call.
    Provider { model: String },
    /// A backend that needs no provider (Gemini, a scripted one in tests).
    Decider(Arc<dyn Decider>),
}

/// The configured decision backend and each decision's mode.
#[derive(Clone)]
pub struct DecisionLayer {
    backend: Backend,
    modes: DecisionModes,
}

impl DecisionLayer {
    /// The layer `[decisions]` describes, or `None` when it is off or
    /// cannot be built (a missing key turns it off, never fails startup).
    #[cfg(test)]
    pub fn from_config(config: &DecisionsConfig) -> Option<Self> {
        Self::from_config_with_key(config, None)
    }

    /// Like `from_config`; `provider_key` is used for
    /// the Gemini backend when `[decisions]` sets no key of its own.
    pub fn from_config_with_key(
        config: &DecisionsConfig,
        provider_key: Option<&str>,
    ) -> Option<Self> {
        let backend = match config.backend {
            DecisionBackendKind::Off => return None,
            DecisionBackendKind::Provider => Backend::Provider {
                model: config.model.trim().to_string(),
            },
            DecisionBackendKind::Gemini => {
                let key = Some(config.api_key.trim())
                    .filter(|key| !key.is_empty())
                    .or(provider_key.map(str::trim).filter(|key| !key.is_empty()));
                let Some(key) = key else {
                    tracing::warn!(
                        "decisions backend is gemini but no api_key is set and no google_genai provider is configured; decision layer off"
                    );
                    return None;
                };
                let mut decider = aidaemon_decisions::GeminiDecider::new(key);
                if !config.model.trim().is_empty() {
                    decider = decider.with_model(config.model.trim());
                }
                Backend::Decider(Arc::new(decider))
            }
        };
        Some(Self {
            backend,
            modes: config.mode,
        })
    }

    /// A layer over a ready backend, for tests.
    #[cfg(test)]
    pub fn with_decider(decider: Arc<dyn Decider>, modes: DecisionModes) -> Self {
        Self {
            backend: Backend::Decider(decider),
            modes,
        }
    }
}

/// Where a decision's telemetry is recorded: the turn's session and task
/// when there is one, else `background:decision:<purpose>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecisionContext<'a> {
    pub session_id: Option<&'a str>,
    pub task_id: Option<&'a str>,
}

impl<'a> DecisionContext<'a> {
    pub fn turn(session_id: &'a str, task_id: &'a str) -> Self {
        Self {
            session_id: Some(session_id),
            task_id: Some(task_id),
        }
    }
}

impl SideCallGateway {
    /// Sets (or clears) the decision backend. Called once at startup.
    pub fn set_decisions(&self, layer: Option<DecisionLayer>) {
        if let Some(layer) = layer.as_ref() {
            tracing::info!(
                backend = match &layer.backend {
                    Backend::Provider { .. } => "provider".to_string(),
                    Backend::Decider(decider) => decider.name().to_string(),
                },
                modes = ?layer.modes,
                "decision layer configured"
            );
        }
        *self
            .decisions
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = layer;
    }

    fn decision_layer(&self) -> Option<DecisionLayer> {
        self.decisions
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// The decision's mode; `Off` when no backend is configured.
    pub fn decision_mode(&self, decision: Decision) -> DecisionMode {
        self.decision_layer()
            .map(|layer| decision.mode_in(&layer.modes))
            .unwrap_or_default()
    }

    /// Asks the decision backend. Never fails: with no backend, the
    /// decision off, a timeout or an error, it returns `UseDefault` and the
    /// caller does what it did without a decision layer.
    pub async fn decide(
        &self,
        decision: Decision,
        request: DecisionRequest,
        context: DecisionContext<'_>,
    ) -> DecisionOutcome {
        let spec = decision.spec();
        let Some(layer) = self.decision_layer() else {
            return DecisionOutcome::UseDefault(DefaultReason::Disabled);
        };
        let mode = decision.mode_in(&layer.modes);
        if mode == DecisionMode::Off || request.questions.is_empty() {
            return DecisionOutcome::UseDefault(DefaultReason::Disabled);
        }
        let decider: Arc<dyn Decider> = match &layer.backend {
            Backend::Decider(decider) => decider.clone(),
            Backend::Provider { model } => {
                let model = if model.is_empty() {
                    self.cheap_model_now()
                } else {
                    model.clone()
                };
                Arc::new(ProviderDecider::new(
                    self.provider(SideCall::DecisionAsk),
                    model,
                ))
            }
        };
        let records_its_own_call = matches!(layer.backend, Backend::Provider { .. });
        let started = Instant::now();
        let result = tokio::time::timeout(spec.deadline, decider.decide(&request)).await;
        let elapsed = started.elapsed();
        let acting = mode == DecisionMode::On;
        let outcome = match result {
            Ok(Ok(response)) => {
                let gated = GatedResponse::new(response, &spec, acting);
                if acting {
                    DecisionOutcome::Decided(gated)
                } else {
                    DecisionOutcome::Shadow(gated)
                }
            }
            Ok(Err(error)) => {
                tracing::warn!(purpose = spec.purpose, %error, "decision failed; using default");
                DecisionOutcome::UseDefault(DefaultReason::Failed(error.to_string()))
            }
            Err(_) => {
                tracing::warn!(purpose = spec.purpose, deadline = ?spec.deadline, "decision timed out; using default");
                DecisionOutcome::UseDefault(DefaultReason::Timeout)
            }
        };
        self.record_decision(
            decision,
            mode,
            decider.name(),
            &request,
            &outcome,
            elapsed,
            context,
            !records_its_own_call,
        );
        outcome
    }

    fn cheap_model_now(&self) -> String {
        self.runtime_handle().snapshot().cheap_model()
    }

    /// Writes one decision-point event (and, for a backend that is not a
    /// side call, one `llm_call` row) in the background, so telemetry
    /// never delays the turn.
    #[allow(clippy::too_many_arguments)]
    fn record_decision(
        &self,
        decision: Decision,
        mode: DecisionMode,
        backend: &str,
        request: &DecisionRequest,
        outcome: &DecisionOutcome,
        elapsed: Duration,
        context: DecisionContext<'_>,
        write_llm_call: bool,
    ) {
        let spec = decision.spec();
        let purpose = format!("decision:{}", spec.purpose);
        let session = context
            .session_id
            .map(str::to_string)
            .unwrap_or_else(|| format!("background:{purpose}"));
        let task = context
            .task_id
            .map(str::to_string)
            .unwrap_or_else(|| session.clone());
        let metadata = decision_metadata(decision, mode, backend, request, outcome, elapsed);
        let summary = match outcome {
            DecisionOutcome::Decided(_) => format!("{purpose} answered"),
            DecisionOutcome::Shadow(_) => format!("{purpose} answered (shadow)"),
            DecisionOutcome::UseDefault(reason) => format!("{purpose} used default: {reason:?}"),
        };
        let usage = outcome.gated().map(|gated| gated.response.usage);
        let error = match outcome {
            DecisionOutcome::UseDefault(DefaultReason::Failed(error)) => Some(error.clone()),
            DecisionOutcome::UseDefault(DefaultReason::Timeout) => Some(format!(
                "decision exceeded its {:?} deadline",
                spec.deadline
            )),
            _ => None,
        };
        let backend = backend.to_string();
        let event_store = self.event_store_handle();
        let state = self.state_handle();
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        handle.spawn(async move {
            let emitter = aidaemon_events::EventEmitter::new(event_store.clone(), session.clone())
                .with_task_id(task.clone());
            let data = aidaemon_events::DecisionPointData {
                decision_type: aidaemon_events::DecisionType::GateTelemetry,
                task_id: task.clone(),
                iteration: 0,
                severity: aidaemon_events::DiagnosticSeverity::Info,
                code: Some("decision_layer".to_string()),
                metadata,
                summary,
            };
            if let Err(error) = emitter
                .emit(aidaemon_events::EventType::DecisionPoint, data)
                .await
            {
                tracing::warn!(%error, "decision telemetry write failed");
            }
            if !write_llm_call {
                return;
            }
            match (usage, error) {
                (_, Some(error)) => {
                    aidaemon_events::model_call_telemetry::record_background_model_call_failure(
                        event_store,
                        state.as_ref(),
                        &session,
                        &purpose,
                        &backend,
                        &error,
                        elapsed,
                    )
                    .await;
                }
                (Some(usage), None) => {
                    let response = aidaemon_events::provider::ProviderResponse {
                        content: None,
                        tool_calls: Vec::new(),
                        usage: Some(aidaemon_events::provider::TokenUsage {
                            input_tokens: usage.input_tokens as u32,
                            output_tokens: usage.output_tokens as u32,
                            cached_input_tokens: None,
                            cache_creation_input_tokens: None,
                            model: backend.clone(),
                            prompt_ms: None,
                            decode_ms: None,
                        }),
                        thinking: None,
                        response_note: None,
                    };
                    aidaemon_events::model_call_telemetry::record_background_model_call_telemetry(
                        event_store,
                        state.as_ref(),
                        &session,
                        &purpose,
                        &backend,
                        &response,
                        elapsed,
                    )
                    .await;
                }
                (None, None) => {}
            }
        });
    }
}

/// The decision-point metadata: per question its value, confidence,
/// source and whether it was (or in shadow, would have been) acted on.
fn decision_metadata(
    decision: Decision,
    mode: DecisionMode,
    backend: &str,
    request: &DecisionRequest,
    outcome: &DecisionOutcome,
    elapsed: Duration,
) -> serde_json::Value {
    let spec = decision.spec();
    let (outcome_tag, answers) = match outcome {
        DecisionOutcome::Decided(gated) => ("decided", answers_json(gated)),
        DecisionOutcome::Shadow(gated) => ("shadow", answers_json(gated)),
        DecisionOutcome::UseDefault(DefaultReason::Disabled) => ("disabled", json!({})),
        DecisionOutcome::UseDefault(DefaultReason::Failed(_)) => ("failed", json!({})),
        DecisionOutcome::UseDefault(DefaultReason::Timeout) => ("timeout", json!({})),
    };
    json!({
        "condition": "decision_layer",
        "decision": spec.purpose,
        "effect": match spec.effect {
            super::decisions::DecisionEffect::SkipOptional => "skip_optional",
            super::decisions::DecisionEffect::Route => "route",
            super::decisions::DecisionEffect::EscalateOnly => "escalate_only",
        },
        "mode": match mode {
            DecisionMode::Off => "off",
            DecisionMode::Shadow => "shadow",
            DecisionMode::On => "on",
        },
        "backend": backend,
        "outcome": outcome_tag,
        "latency_ms": elapsed.as_millis() as u64,
        "questions": request.questions.len(),
        "answers": answers,
    })
}

fn answers_json(gated: &GatedResponse) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (key, answer) in &gated.response.answers {
        let value = match &answer.value {
            aidaemon_decisions::AnswerValue::YesNo(p) => json!(p),
            aidaemon_decisions::AnswerValue::Choice(option) => json!(option),
            aidaemon_decisions::AnswerValue::Score(score) => json!(score),
            _ => serde_json::Value::Null,
        };
        out.insert(
            key.clone(),
            json!({
                "value": value,
                "confidence": answer.confidence,
                "source": answer.confidence_source.as_str(),
                "trusted": gated.trusted(key).is_some(),
                "acted": gated.acted(key).is_some(),
            }),
        );
    }
    serde_json::Value::Object(out)
}

/// Turns a backend's answers into a gated response without a call, for
/// tests of callers.
#[cfg(any(test, feature = "testing"))]
pub fn gated_for_tests(
    decision: Decision,
    response: DecisionResponse,
    acting: bool,
) -> DecisionOutcome {
    let gated = GatedResponse::new(response, &decision.spec(), acting);
    if acting {
        DecisionOutcome::Decided(gated)
    } else {
        DecisionOutcome::Shadow(gated)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::Duration;

    use aidaemon_decisions::testing::ScriptedDecider;
    use aidaemon_decisions::{
        Answer, ConfidenceSource, Decider, DeciderCapabilities, DecisionError, DecisionRequest,
        DecisionResponse, Question,
    };
    use async_trait::async_trait;

    use super::*;
    use crate::llm::decisions::digest_keys;
    use crate::llm::gateway::test_support::{gateway_with, provider_answering};
    use crate::settings::DecisionModes;

    fn digest_request() -> DecisionRequest {
        DecisionRequest::new("Current user message:\nwhat is my name?")
            .ask(digest_keys::IDENTITY, Question::yes_no("identity?"))
    }

    fn identity_yes() -> BTreeMap<String, Answer> {
        let mut answers = BTreeMap::new();
        answers.insert(
            digest_keys::IDENTITY.to_string(),
            Answer::yes_no(0.95, ConfidenceSource::Calibrated),
        );
        answers
    }

    fn modes(turn_digest: DecisionMode) -> DecisionModes {
        DecisionModes {
            turn_digest,
            ..DecisionModes::default()
        }
    }

    async fn gateway(decider: Arc<dyn Decider>, mode: DecisionMode) -> SideCallGateway {
        let (gateway, _events) = gateway_with(provider_answering("unused")).await;
        gateway.set_decisions(Some(DecisionLayer::with_decider(decider, modes(mode))));
        gateway
    }

    #[tokio::test]
    async fn no_backend_means_the_default() {
        let (gateway, _events) = gateway_with(provider_answering("unused")).await;
        let outcome = gateway
            .decide(
                Decision::TurnDigest,
                digest_request(),
                DecisionContext::default(),
            )
            .await;
        assert!(matches!(
            outcome,
            DecisionOutcome::UseDefault(DefaultReason::Disabled)
        ));
        assert_eq!(
            gateway.decision_mode(Decision::TurnDigest),
            DecisionMode::Off
        );
    }

    #[tokio::test]
    async fn an_off_decision_never_calls_the_backend() {
        let decider = Arc::new(ScriptedDecider::new(ConfidenceSource::Calibrated).always(0.9));
        let gateway = gateway(decider.clone(), DecisionMode::Off).await;
        let outcome = gateway
            .decide(
                Decision::TurnDigest,
                digest_request(),
                DecisionContext::default(),
            )
            .await;
        assert!(matches!(outcome, DecisionOutcome::UseDefault(_)));
        assert!(decider.calls().is_empty());
    }

    #[tokio::test]
    async fn shadow_answers_are_recorded_but_not_acted_on() {
        let decider =
            Arc::new(ScriptedDecider::new(ConfidenceSource::Calibrated).then(Ok(identity_yes())));
        let gateway = gateway(decider.clone(), DecisionMode::Shadow).await;
        let outcome = gateway
            .decide(
                Decision::TurnDigest,
                digest_request(),
                DecisionContext::default(),
            )
            .await;
        assert!(matches!(outcome, DecisionOutcome::Shadow(_)));
        assert!(outcome.acted(digest_keys::IDENTITY).is_none());
        assert_eq!(decider.calls().len(), 1);
    }

    #[tokio::test]
    async fn an_on_decision_acts_on_trusted_answers_and_records_them() {
        let decider =
            Arc::new(ScriptedDecider::new(ConfidenceSource::Calibrated).then(Ok(identity_yes())));
        let (gateway, events) = gateway_with(provider_answering("unused")).await;
        gateway.set_decisions(Some(DecisionLayer::with_decider(
            decider,
            modes(DecisionMode::On),
        )));
        let outcome = gateway
            .decide(
                Decision::TurnDigest,
                digest_request(),
                DecisionContext::turn("session-a", "task-a"),
            )
            .await;
        assert!(outcome.acted(digest_keys::IDENTITY).is_some());
        // Telemetry is written in the background.
        let mut recorded = false;
        for _ in 0..50 {
            let rows = events
                .query_task_events_for_session("session-a", "task-a")
                .await
                .unwrap();
            if rows.iter().any(|event| {
                event.event_type == aidaemon_events::EventType::DecisionPoint
                    && event
                        .data
                        .to_string()
                        .contains("\"decision\":\"turn_digest\"")
            }) {
                recorded = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(recorded, "the decision point was not recorded");
    }

    #[tokio::test]
    async fn a_failing_backend_means_the_default() {
        let decider = Arc::new(
            ScriptedDecider::new(ConfidenceSource::Calibrated)
                .then(Err(DecisionError::Transient("down".into()))),
        );
        let gateway = gateway(decider, DecisionMode::On).await;
        let outcome = gateway
            .decide(
                Decision::TurnDigest,
                digest_request(),
                DecisionContext::default(),
            )
            .await;
        assert!(matches!(
            outcome,
            DecisionOutcome::UseDefault(DefaultReason::Failed(_))
        ));
    }

    struct Hanging;

    #[async_trait]
    impl Decider for Hanging {
        fn name(&self) -> &str {
            "hanging"
        }
        fn capabilities(&self) -> DeciderCapabilities {
            DeciderCapabilities::new(10, ConfidenceSource::Calibrated)
        }
        async fn decide(&self, _: &DecisionRequest) -> Result<DecisionResponse, DecisionError> {
            std::future::pending().await
        }
    }

    #[tokio::test]
    async fn a_hung_backend_is_cut_off_at_the_deadline() {
        let gateway = gateway(Arc::new(Hanging), DecisionMode::On).await;
        let outcome = gateway
            .decide(
                Decision::TurnDigest,
                digest_request(),
                DecisionContext::default(),
            )
            .await;
        assert!(matches!(
            outcome,
            DecisionOutcome::UseDefault(DefaultReason::Timeout)
        ));
    }

    #[tokio::test]
    async fn the_provider_backend_answers_through_the_side_call_door() {
        let (gateway, events) =
            gateway_with(provider_answering(r#"{"identity":{"p_yes":0.9}}"#)).await;
        gateway.set_decisions(DecisionLayer::from_config(
            &crate::settings::DecisionsConfig {
                backend: crate::settings::DecisionBackendKind::Provider,
                model: "cheap-model".to_string(),
                api_key: String::new(),
                mode: modes(DecisionMode::On),
            },
        ));
        let outcome = gateway
            .decide(
                Decision::TurnDigest,
                digest_request(),
                DecisionContext::default(),
            )
            .await;
        let gated = outcome.gated().expect("answered");
        let answer = gated.observed(digest_keys::IDENTITY).unwrap();
        assert_eq!(answer.confidence_source, ConfidenceSource::SelfReported);
        // Identity acts on any source.
        assert!(outcome.acted(digest_keys::IDENTITY).is_some());
        assert!(
            crate::llm::gateway::test_support::ok_calls(
                &events,
                SideCall::DecisionAsk.spec().purpose
            )
            .await
                >= 1
        );
    }

    #[test]
    fn a_gemini_backend_reuses_the_gemini_provider_key() {
        let config = crate::settings::DecisionsConfig {
            backend: crate::settings::DecisionBackendKind::Gemini,
            ..Default::default()
        };
        assert!(DecisionLayer::from_config_with_key(&config, Some("synthetic-key")).is_some());
        assert!(DecisionLayer::from_config_with_key(&config, Some("  ")).is_none());
    }

    #[test]
    fn a_gemini_backend_without_a_key_is_off() {
        let config = crate::settings::DecisionsConfig {
            backend: crate::settings::DecisionBackendKind::Gemini,
            ..Default::default()
        };
        assert!(DecisionLayer::from_config(&config).is_none());
        assert!(DecisionLayer::from_config(&Default::default()).is_none());
    }
}
