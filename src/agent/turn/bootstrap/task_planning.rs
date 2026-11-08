use crate::providers::ChatJson;
use crate::traits::ProviderResponse;
use crate::utils::truncate_str;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::{Duration, Instant};
use tracing::warn;

/// Build bounded relationship-assessment context. This is not provider
/// transcript carryover: stable message IDs let the router identify one
/// exact antecedent, while fresh tasks still discard all archived prose after
/// the relationship is finalized.
pub(crate) fn relationship_conversation_context(
    _followup_mode: Option<crate::agent::FollowupMode>,
    session_summary: Option<&str>,
    recent_messages: &[serde_json::Value],
) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(summary) = session_summary.filter(|summary| !summary.is_empty()) {
        parts.push(format!("[Session Summary] {summary}"));
    }
    for message in recent_messages {
        let role = message
            .get("role")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let content = message
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if !content.is_empty() {
            let message_id = message
                .get("message_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown");
            parts.push(format!(
                "- [message_id={message_id}] {}: {}",
                role.chars().next().unwrap_or('?').to_uppercase(),
                content
            ));
        }
    }
    (!parts.is_empty()).then(|| parts.join("\n"))
}

/// The relationship router's typed dialogue edge: how this turn relates to
/// persisted dialogue state. It never authorizes actions or decides completion.
#[derive(Debug, Clone)]
pub(crate) struct PlannedTaskShape {
    /// `low`, `medium`, or `high`.
    pub confidence: Option<String>,
    /// Relationship of this turn to persisted dialogue state.
    pub request_relationship: Option<String>,
    /// Exact prior user-message ID selected from relationship-assessment
    /// context. Required for `continuation`; null for independent requests.
    pub antecedent_user_message_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskRelationshipResponse {
    schema_version: u16,
    confidence: String,
    request_relationship: String,
    #[serde(default)]
    antecedent_user_message_id: Option<String>,
}

const TASK_RELATIONSHIP_SCHEMA_VERSION: u16 = 1;

fn confidence_is_sufficient(value: Option<&str>) -> bool {
    matches!(
        value.map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "medium" | "high")
    )
}

pub(crate) fn planned_task_relationship_is_complete(shape: &PlannedTaskShape) -> bool {
    let relationship = shape
        .request_relationship
        .as_deref()
        .map(str::trim)
        .unwrap_or_default();
    let antecedent = shape
        .antecedent_user_message_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    confidence_is_sufficient(shape.confidence.as_deref())
        && match relationship {
            // The semantic compiler owns the relationship class. Exact
            // identity belongs to the deterministic dialogue-state resolver,
            // which may bind an omitted ID only when one unambiguous request
            // antecedent exists and otherwise fails closed to a new request.
            "continuation" => true,
            "new_request" | "clarification_answer" | "courtesy" => antecedent.is_none(),
            _ => false,
        }
}

/// Automatic-memory policy for one task, compiled once from typed lifecycle
/// state. Prompt retrieval, message-span projection, and post-task learning
/// must all consume this one value; none of those stages is allowed to
/// independently reinterpret the user's wording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryPipelinePolicy {
    Allowed,
    SuppressedByCurrentContract,
    /// No person wrote this turn's text (background continuation, runtime
    /// resume, mandate cycle, control command). Recall stays on so the agent
    /// still knows who it works for; persistence stays off so it cannot learn
    /// from its own synthetic text.
    RecallOnlyNonUserTurn,
}

impl MemoryPipelinePolicy {
    /// Whether automatic memory may be written or learned from this task.
    pub(crate) const fn allows_memory(self) -> bool {
        matches!(self, Self::Allowed)
    }

    /// Whether stored memory may be recalled into this task's prompt.
    pub(crate) const fn allows_recall(self) -> bool {
        matches!(self, Self::Allowed | Self::RecallOnlyNonUserTurn)
    }

    pub(crate) const fn reason_code(self) -> &'static str {
        match self {
            Self::Allowed => "allowed_by_compiled_contract",
            Self::SuppressedByCurrentContract => "inherited_contract_denies_user_memory",
            Self::RecallOnlyNonUserTurn => "non_user_turn_recall_only",
        }
    }
}

/// Compile the task-local automatic-memory policy from typed lifecycle state:
/// a contract that forbids user memory (inherited from a request that said so)
/// suppresses it; otherwise a person's turn keeps full memory and a turn no
/// person wrote may only recall.
pub(crate) fn compile_memory_pipeline_policy(
    current_contract: &crate::agent::CompletionContract,
    user_authored_turn: bool,
) -> MemoryPipelinePolicy {
    if current_contract
        .forbidden_tool_scopes
        .contains(&crate::traits::ToolSemanticScope::UserMemory)
    {
        MemoryPipelinePolicy::SuppressedByCurrentContract
    } else if user_authored_turn {
        MemoryPipelinePolicy::Allowed
    } else {
        MemoryPipelinePolicy::RecallOnlyNonUserTurn
    }
}

// Aggregate producer deadline. Candidates are bounded failover attempts under
// one logical relationship envelope; valid results are selected by configured
// priority, never by response wording or latency.
const TASK_RELATIONSHIP_TOTAL_TIMEOUT: Duration = Duration::from_secs(45);

/// JSON-object mode is rejected by a number of OpenAI-compatible gateways
/// unless the prompt contains a provider-specific lowercase marker.  The
/// daemon's semantic producers should not depend on prompt spelling.  A
/// permissive named schema requests the same object protocol while leaving
/// the domain parser as the authoritative validator for optional fields.
fn typed_object_response_mode(name: &str) -> crate::traits::ResponseMode {
    crate::traits::ResponseMode::JsonSchema {
        name: name.to_string(),
        schema: json!({
            "type": "object",
            "additionalProperties": true
        }),
        strict: false,
    }
}

/// Divide the remaining relationship envelope fairly: every configured candidate receives a bounded share, and unused
/// time is naturally returned to later candidates after a fast failure.
fn relationship_attempt_timeout(
    remaining: std::time::Duration,
    candidates_left: usize,
) -> Option<std::time::Duration> {
    (candidates_left > 0 && !remaining.is_zero())
        .then(|| remaining / u32::try_from(candidates_left).unwrap_or(u32::MAX))
}

#[derive(Clone, Copy)]
pub(crate) struct PlannerTelemetryCtx<'a> {
    pub emitter: &'a crate::events::EventEmitter,
    pub state: &'a dyn crate::traits::TokenUsageStore,
    pub session_id: &'a str,
    pub task_id: &'a str,
}

struct AuxiliaryAttemptTelemetry {
    attempt: u32,
    fell_back: bool,
    validation_error: Option<String>,
}

async fn record_auxiliary_model_call(
    telemetry: Option<PlannerTelemetryCtx<'_>>,
    call_purpose: &str,
    model: &str,
    response: &ProviderResponse,
    latency_ms: u64,
    attempt_telemetry: AuxiliaryAttemptTelemetry,
) {
    let Some(telemetry) = telemetry else {
        return;
    };
    let (
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_creation_input_tokens,
        fresh_input_tokens,
    ) = response
        .usage
        .as_ref()
        .map(|usage| {
            (
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_input_tokens,
                usage.cache_creation_input_tokens,
                usage.fresh_input_tokens(),
            )
        })
        .unwrap_or((0, 0, None, None, None));
    crate::events::record_model_call_telemetry(
        telemetry.emitter,
        telemetry.state,
        crate::events::ModelCallTelemetryInput {
            session_id: telemetry.session_id.to_string(),
            task_id: telemetry.task_id.to_string(),
            call_purpose: Some(call_purpose.to_string()),
            iteration: Some(0),
            llm_call: crate::events::LlmCallData {
                call_id: None,
                call_purpose: Some(call_purpose.to_string()),
                task_id: telemetry.task_id.to_string(),
                iteration: Some(0),
                model: model.to_string(),
                final_model: Some(model.to_string()),
                fell_back: attempt_telemetry.fell_back,
                attempts: attempt_telemetry.attempt,
                latency_ms,
                prompt_ms: None,
                decode_ms: None,
                input_tokens,
                output_tokens,
                cached_input_tokens,
                cache_creation_input_tokens,
                fresh_input_tokens,
                est_input_tokens: None,
                tool_calls_count: response.tool_calls.len() as u32,
                offered_tools: Vec::new(),
                chosen_tools: Vec::new(),
                build_ms: None,
                prefix_hash_system: None,
                prefix_hash_pre_boundary: None,
                tool_defs_hash: None,
                session_summary_hash: None,
                tail_hash: None,
                prefix_hash_archived: None,
                boundary_pos: None,
                message_count: None,
                projected_source_message_ids: Vec::new(),
                projected_source_turn_ids: Vec::new(),
                force_text: false,
                token_usage_present: response.usage.is_some(),
                token_usage_evidence: crate::events::TokenUsageEvidence::Unavailable,
                failed: attempt_telemetry.validation_error.is_some(),
                error: attempt_telemetry.validation_error,
                provider_error_kind: None,
                provider_status: None,
            },
            token_usage: response.usage.clone(),
        },
    )
    .await;
}

#[allow(clippy::too_many_arguments)]
async fn record_auxiliary_model_failure(
    telemetry: Option<PlannerTelemetryCtx<'_>>,
    call_purpose: &str,
    model: &str,
    latency_ms: u64,
    attempt: u32,
    fell_back: bool,
    est_input_tokens: Option<u32>,
    provider_error_kind: Option<crate::providers::ProviderErrorKind>,
    provider_status: Option<u16>,
    error: impl Into<String>,
) {
    let Some(telemetry) = telemetry else {
        return;
    };
    crate::events::record_model_call_telemetry(
        telemetry.emitter,
        telemetry.state,
        crate::events::ModelCallTelemetryInput {
            session_id: telemetry.session_id.to_string(),
            task_id: telemetry.task_id.to_string(),
            call_purpose: Some(call_purpose.to_string()),
            iteration: Some(0),
            llm_call: crate::events::LlmCallData {
                call_id: None,
                call_purpose: Some(call_purpose.to_string()),
                task_id: telemetry.task_id.to_string(),
                iteration: Some(0),
                model: model.to_string(),
                final_model: Some(model.to_string()),
                fell_back,
                attempts: attempt,
                latency_ms,
                prompt_ms: None,
                decode_ms: None,
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: None,
                cache_creation_input_tokens: None,
                fresh_input_tokens: None,
                // Failed calls cannot report provider usage, but the message
                // build already gives us a bounded cost estimate. Persist it
                // as estimate-only telemetry so timeout/error calls are not
                // invisible in cost reconciliation.
                est_input_tokens,
                tool_calls_count: 0,
                offered_tools: Vec::new(),
                chosen_tools: Vec::new(),
                build_ms: None,
                prefix_hash_system: None,
                prefix_hash_pre_boundary: None,
                tool_defs_hash: None,
                session_summary_hash: None,
                tail_hash: None,
                prefix_hash_archived: None,
                boundary_pos: None,
                message_count: None,
                projected_source_message_ids: Vec::new(),
                projected_source_turn_ids: Vec::new(),
                force_text: false,
                token_usage_present: false,
                token_usage_evidence: crate::events::TokenUsageEvidence::Unavailable,
                failed: true,
                error: Some(error.into()),
                provider_error_kind,
                provider_status,
            },
            token_usage: None,
        },
    )
    .await;
}

/// Resolve only the typed dialogue edge. This deliberately cannot produce
/// completion obligations, authority, tool policy, or execution mode. The
/// caller still validates an exact antecedent against persisted dialogue state
/// before committing it.
#[cfg(test)]
pub(crate) async fn generate_task_relationship(
    provider: crate::llm_runtime::ModelBackend,
    model: &str,
    user_text: &str,
    conversation_context: &str,
    telemetry: Option<PlannerTelemetryCtx<'_>>,
) -> Option<PlannedTaskShape> {
    let models = [model.to_string()];
    generate_task_relationship_candidates(
        provider,
        &models,
        user_text,
        conversation_context,
        telemetry,
    )
    .await
}

/// Resolve a dialogue relationship through the configured model lane. This
/// producer returns only a typed edge; it never installs a task contract or
/// imports prior authority. Candidates are attempted in configured priority
/// order, with a bounded fair share so a slow primary cannot starve every
/// relationship attempt and a fast fallback cannot win by latency alone.
pub(crate) async fn generate_task_relationship_candidates(
    provider: crate::llm_runtime::ModelBackend,
    model_candidates: &[String],
    user_text: &str,
    conversation_context: &str,
    telemetry: Option<PlannerTelemetryCtx<'_>>,
) -> Option<PlannedTaskShape> {
    if conversation_context.trim().is_empty() {
        return None;
    }
    let system = "You are a task relationship router. Return only valid JSON. You classify one typed dialogue edge and cannot authorize actions or decide completion.";
    let prompt = format!(
        "Recent conversation with stable message IDs:\n{conversation_context}\n\n\
         Current user request:\n{user_text}\n\n\
         Return exactly: {{\"schema_version\":1,\"confidence\":\"low|medium|high\",\
         \"request_relationship\":\"new_request|continuation|clarification_answer|courtesy\",\
         \"antecedent_user_message_id\":null|\"message-id\"}}.\n\
         Use continuation only when the current request advances, retries, or asks about one prior request shown above; copy that prior USER row's exact message_id. Otherwise use new_request and null. Courtesy and clarification_answer also require null. Judge the whole meaning, not isolated words.",
        conversation_context = truncate_str(conversation_context, 6000),
        user_text = truncate_str(user_text, 4000),
    );
    let messages = vec![
        json!({ "role": "system", "content": system }),
        json!({ "role": "user", "content": prompt }),
    ];
    let est_input_tokens = Some(
        crate::context_budget::estimate_multimodal_message_tokens(&messages).min(u32::MAX as usize)
            as u32,
    );
    let options = crate::traits::ChatOptions {
        response_mode: typed_object_response_mode("task_relationship"),
        tool_choice: crate::traits::ToolChoiceMode::None,
        max_tokens_override: Some(1024),
        reasoning_effort_override: Some("low".to_string()),
        // Relationship routing is optional, but it must not hide provider
        // retries inside its aggregate deadline. The model supplies only a
        // typed edge; one physical attempt per candidate keeps timeout and
        // cost telemetry attributable to the right candidate.
        single_attempt_fail_closed: true,
        ..crate::traits::ChatOptions::default()
    };
    let deadline = Instant::now() + TASK_RELATIONSHIP_TOTAL_TIMEOUT;
    for (candidate_index, model) in model_candidates.iter().enumerate() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let Some(attempt_timeout) = relationship_attempt_timeout(
            remaining,
            model_candidates.len().saturating_sub(candidate_index),
        ) else {
            break;
        };
        let call_start = Instant::now();
        let response = match tokio::time::timeout(
            attempt_timeout,
            provider.chat_with_options(model, &messages, &[], &options),
        )
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                let provider_error = error.downcast_ref::<crate::providers::ProviderError>();
                let prevents_model_fallback = provider_error
                    .is_some_and(crate::providers::ProviderError::prevents_model_fallback);
                record_auxiliary_model_failure(
                    telemetry,
                    "task_relationship",
                    model,
                    call_start.elapsed().as_millis() as u64,
                    (candidate_index + 1) as u32,
                    candidate_index > 0,
                    est_input_tokens,
                    provider_error.map(|error| error.kind),
                    provider_error.and_then(|error| error.status),
                    format!("provider_error: {error}"),
                )
                .await;
                warn!(%error, %model, "Task relationship fallback failed");
                if prevents_model_fallback {
                    break;
                }
                continue;
            }
            Err(_) => {
                record_auxiliary_model_failure(
                    telemetry,
                    "task_relationship",
                    model,
                    call_start.elapsed().as_millis() as u64,
                    (candidate_index + 1) as u32,
                    candidate_index > 0,
                    est_input_tokens,
                    None,
                    None,
                    "timeout",
                )
                .await;
                warn!(
                    %model,
                    timeout_ms = attempt_timeout.as_millis(),
                    "Task relationship fallback timed out"
                );
                continue;
            }
        };
        let latency_ms = call_start.elapsed().as_millis() as u64;
        let Some(json) =
            crate::utils::extract_json_object(response.content.as_deref().unwrap_or(""))
        else {
            record_auxiliary_model_call(
                telemetry,
                "task_relationship",
                model,
                &response,
                latency_ms,
                AuxiliaryAttemptTelemetry {
                    attempt: (candidate_index + 1) as u32,
                    fell_back: candidate_index > 0,
                    validation_error: Some("missing_json_object".to_string()),
                },
            )
            .await;
            continue;
        };
        let parsed: TaskRelationshipResponse = match serde_json::from_str(&json) {
            Ok(parsed) => parsed,
            Err(error) => {
                record_auxiliary_model_call(
                    telemetry,
                    "task_relationship",
                    model,
                    &response,
                    latency_ms,
                    AuxiliaryAttemptTelemetry {
                        attempt: (candidate_index + 1) as u32,
                        fell_back: candidate_index > 0,
                        validation_error: Some(error.to_string()),
                    },
                )
                .await;
                continue;
            }
        };
        let valid = parsed.schema_version == TASK_RELATIONSHIP_SCHEMA_VERSION
            && matches!(parsed.confidence.as_str(), "low" | "medium" | "high")
            && matches!(
                parsed.request_relationship.as_str(),
                "new_request" | "continuation" | "clarification_answer" | "courtesy"
            );
        let antecedent = parsed
            .antecedent_user_message_id
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        // The model owns semantic classification, not graph identity. An exact
        // message ID is useful evidence when it is available, but omission is
        // not a classification failure: the state resolver can bind a
        // continuation only when exactly one persisted request node is possible.
        let valid =
            valid && (parsed.request_relationship == "continuation" || antecedent.is_none());
        record_auxiliary_model_call(
            telemetry,
            "task_relationship",
            model,
            &response,
            latency_ms,
            AuxiliaryAttemptTelemetry {
                attempt: (candidate_index + 1) as u32,
                fell_back: candidate_index > 0,
                validation_error: (!valid).then(|| "invalid_typed_relationship".to_string()),
            },
        )
        .await;
        if !valid {
            continue;
        }
        return Some(PlannedTaskShape {
            confidence: Some(parsed.confidence),
            request_relationship: Some(parsed.request_relationship),
            antecedent_user_message_id: antecedent,
        });
    }
    None
}

/// Determine whether a turn bypasses the relationship router. Semantic
/// interpretation belongs to the model. Only protocol-level acknowledgments
/// and exact emergency control commands bypass it.
#[cfg(test)]
fn should_skip_planning(
    _task_kind: &crate::agent::CompletionTaskKind,
    user_text: &str,
    is_acknowledgment: bool,
) -> bool {
    planning_skip_reason(user_text, is_acknowledgment).is_some()
}

pub(crate) fn planning_skip_reason(
    user_text: &str,
    is_acknowledgment: bool,
) -> Option<&'static str> {
    if is_acknowledgment {
        return Some("acknowledgment");
    }

    // Exact control commands are handled by the typed bootstrap protocol.
    let lower = user_text.trim().to_lowercase();
    let control_commands = [
        "cancel",
        "stop",
        "nevermind",
        "never mind",
        "abort",
        "forget it",
        "forget that",
        "nvm",
    ];
    if control_commands.iter().any(|cmd| lower == *cmd) {
        return Some("control_command");
    }

    None
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[derive(Default)]
    struct ModelInvariantAuthFailureProvider {
        calls: std::sync::Mutex<Vec<String>>,
    }

    crate::testing::chat_json_provider!(ModelInvariantAuthFailureProvider);

    impl ModelInvariantAuthFailureProvider {
        async fn respond(
            &self,
            model: &str,
            _messages: &[serde_json::Value],
            _tools: &[serde_json::Value],
            _options: &crate::traits::ChatOptions,
        ) -> anyhow::Result<ProviderResponse> {
            self.calls.lock().unwrap().push(model.to_string());
            Err(
                crate::providers::ProviderError::from_status(401, "synthetic auth rejection")
                    .into(),
            )
        }
    }

    #[test]
    fn conversational_turn_reaches_the_relationship_router() {
        use crate::agent::CompletionTaskKind;
        assert!(!should_skip_planning(
            &CompletionTaskKind::Conversational,
            "hello there how are you doing today",
            false
        ));
    }

    #[test]
    fn short_turn_reaches_the_relationship_router() {
        use crate::agent::CompletionTaskKind;
        assert!(!should_skip_planning(
            &CompletionTaskKind::Find,
            "hi",
            false
        ));
    }

    #[test]
    fn direct_recall_question_reaches_the_relationship_router() {
        use crate::agent::CompletionTaskKind;
        assert_eq!(planning_skip_reason("What's my dad's name?", false), None);
        assert!(!should_skip_planning(
            &CompletionTaskKind::Answer,
            "What's the source of that?",
            false
        ));
    }

    #[test]
    fn artifact_request_reaches_the_relationship_router() {
        let request = "Can you create a pptx about Ecuador. Make it beautiful as I will present in my daughter's school.";

        assert_eq!(planning_skip_reason(request, false), None);
    }

    #[test]
    fn test_should_skip_control_commands() {
        use crate::agent::CompletionTaskKind;
        assert!(should_skip_planning(
            &CompletionTaskKind::Conversational,
            "cancel",
            false
        ));
        assert!(should_skip_planning(
            &CompletionTaskKind::Conversational,
            "stop",
            false
        ));
        assert!(should_skip_planning(
            &CompletionTaskKind::Conversational,
            "nevermind",
            false
        ));
        assert!(should_skip_planning(
            &CompletionTaskKind::Conversational,
            "nvm",
            false
        ));
        assert!(should_skip_planning(
            &CompletionTaskKind::Conversational,
            "abort",
            false
        ));
    }

    #[test]
    fn timed_request_reaches_the_relationship_router() {
        use crate::agent::CompletionTaskKind;
        assert_eq!(
            planning_skip_reason("Check deployment tomorrow at 9am", false),
            None
        );
        assert!(!should_skip_planning(
            &CompletionTaskKind::Change,
            "Check deployment tomorrow at 9am",
            false
        ));
    }

    #[test]
    fn cadence_inside_draft_only_request_is_not_a_schedule_route() {
        let request = "Prepare a mandate draft that reviews every 15 minutes. Do not schedule, create, or activate anything.";
        assert_eq!(planning_skip_reason(request, false), None);
    }

    #[test]
    fn test_should_skip_acknowledgment() {
        use crate::agent::CompletionTaskKind;
        assert!(should_skip_planning(
            &CompletionTaskKind::Change,
            "ok thanks that looks good",
            true
        ));
    }

    #[test]
    fn test_should_not_skip_investigative_question() {
        use crate::agent::CompletionTaskKind;
        assert!(!should_skip_planning(
            &CompletionTaskKind::Find,
            "Can you figure out how the Twitter scheduled task got removed?",
            false
        ));
    }

    #[test]
    fn test_should_not_skip_change_task() {
        use crate::agent::CompletionTaskKind;
        assert!(!should_skip_planning(
            &CompletionTaskKind::Change,
            "Create a website about zeigarnik effect and deploy to cloudflare",
            false
        ));
    }

    #[test]
    fn long_non_english_request_reaches_the_relationship_router() {
        let request = "Construye una aplicación completa, configura la entrega externa, \
            comprueba el resultado final y comparte el enlace cuando todo esté listo.";

        assert_eq!(planning_skip_reason(request, false), None);
    }

    #[test]
    fn relationship_attempt_budget_reserves_time_for_every_remaining_failover() {
        let total = std::time::Duration::from_secs(45);
        assert_eq!(relationship_attempt_timeout(total, 3), Some(total / 3));
        assert_eq!(
            relationship_attempt_timeout(std::time::Duration::from_secs(30), 2),
            Some(std::time::Duration::from_secs(15))
        );
        assert_eq!(
            relationship_attempt_timeout(std::time::Duration::from_secs(14), 1),
            Some(std::time::Duration::from_secs(14)),
            "the final configured candidate receives all remaining time"
        );
    }

    #[tokio::test]
    async fn relationship_lane_does_not_amplify_model_invariant_auth_failure() {
        let models = vec![
            "synthetic-primary".to_string(),
            "synthetic-fallback-a".to_string(),
            "synthetic-fallback-b".to_string(),
        ];
        let provider = Arc::new(ModelInvariantAuthFailureProvider::default());
        let relationship = generate_task_relationship_candidates(
            provider.clone(),
            &models,
            "Continue the synthetic task.",
            "- [message_id=synthetic-1] U: Start the synthetic task.",
            None,
        )
        .await;

        assert!(relationship.is_none());
        assert_eq!(
            provider.calls.lock().unwrap().as_slice(),
            ["synthetic-primary"]
        );
    }

    #[tokio::test]
    async fn relationship_fallback_returns_only_an_exact_typed_dialogue_edge() {
        let response = crate::testing::MockProvider::text_response(
            r#"{
                "schema_version": 1,
                "confidence": "high",
                "request_relationship": "continuation",
                "antecedent_user_message_id": "message-setup-1"
            }"#,
        );
        let provider =
            crate::testing::MockProvider::new().with_relationship_assessments(vec![response]);

        let shape = generate_task_relationship(
            Arc::new(provider),
            "local-model",
            "Return the values from that setup.",
            "- [message_id=message-setup-1] USER: Synthetic setup\n\
             - [message_id=message-reply-1] ASSISTANT: Acknowledged",
            None,
        )
        .await
        .expect("typed relationship");

        assert_eq!(shape.request_relationship.as_deref(), Some("continuation"));
        assert_eq!(
            shape.antecedent_user_message_id.as_deref(),
            Some("message-setup-1")
        );
    }

    #[tokio::test]
    async fn relationship_fallback_leaves_missing_identity_to_state_resolver() {
        let response = crate::testing::MockProvider::text_response(
            r#"{
                "schema_version": 1,
                "confidence": "high",
                "request_relationship": "continuation",
                "antecedent_user_message_id": null
            }"#,
        );
        let provider =
            crate::testing::MockProvider::new().with_relationship_assessments(vec![response]);

        let shape = generate_task_relationship(
            Arc::new(provider),
            "local-model",
            "Continue.",
            "- [message_id=message-setup-1] USER: Synthetic setup",
            None,
        )
        .await
        .expect("semantic continuation should survive an omitted optional identity");
        assert_eq!(shape.request_relationship.as_deref(), Some("continuation"));
        assert!(shape.antecedent_user_message_id.is_none());
        assert!(planned_task_relationship_is_complete(&shape));
    }

    #[test]
    fn memory_policy_is_compiled_once_from_typed_task_capabilities() {
        let user_memory = crate::traits::ToolSemanticScope::UserMemory;
        let current_contract = crate::agent::CompletionContract::default();
        let person = compile_memory_pipeline_policy(&current_contract, true);
        assert_eq!(
            person,
            MemoryPipelinePolicy::Allowed,
            "a person's turn keeps full memory unless a contract denies it"
        );
        assert!(person.allows_recall() && person.allows_memory());

        let synthetic = compile_memory_pipeline_policy(&current_contract, false);
        assert_eq!(synthetic, MemoryPipelinePolicy::RecallOnlyNonUserTurn);
        assert!(synthetic.allows_recall());
        assert!(
            !synthetic.allows_memory(),
            "the agent must not learn from its own synthetic text"
        );

        let mut inherited_denial = current_contract.clone();
        inherited_denial.forbidden_tool_scopes.push(user_memory);
        for user_authored_turn in [true, false] {
            let denied = compile_memory_pipeline_policy(&inherited_denial, user_authored_turn);
            assert_eq!(denied, MemoryPipelinePolicy::SuppressedByCurrentContract);
            assert!(!denied.allows_recall() && !denied.allows_memory());
        }
    }
}
