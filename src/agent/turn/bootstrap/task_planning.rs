use crate::llm_runtime::gateway::SideCallGateway;
use crate::llm_runtime::side_calls::SideCall;
use crate::utils::truncate_str;
use aidaemon_core as sdk;
use serde::{Deserialize, Serialize};
use serde_json::json;
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

/// Automatic-memory policy for one task (`aidaemon_memory::policy`).
pub(crate) use aidaemon_memory::policy::MemoryPipelinePolicy;

/// Compile the task-local automatic-memory policy from typed lifecycle state:
/// whether the contract forbids user memory (inherited from a request that
/// said so) and whether a person wrote the turn.
pub(crate) fn compile_memory_pipeline_policy(
    current_contract: &crate::agent::CompletionContract,
    user_authored_turn: bool,
) -> MemoryPipelinePolicy {
    MemoryPipelinePolicy::compile(
        if user_authored_turn {
            aidaemon_memory::policy::TurnAuthor::Person
        } else {
            aidaemon_memory::policy::TurnAuthor::Synthetic
        },
        current_contract
            .forbidden_tool_scopes
            .contains(&crate::traits::ToolSemanticScope::UserMemory),
    )
}

/// The turn's one memory-policy event. A turn opted out of learning is
/// recorded as suppressed with the opt-out's reason whatever the compiled
/// policy says: the reason is what keeps the turn out of history search, so
/// it must not depend on the turn's memory being allowed otherwise. Recall
/// still follows the policy (opting out is about learning, not reading).
pub(crate) fn memory_policy_compiled_data(
    task_id: &str,
    turn_id: &str,
    policy: &MemoryPipelinePolicy,
    learning_opted_out: bool,
) -> crate::events::MemoryPolicyCompiledData {
    let persistence_allowed = policy.allows_memory() && !learning_opted_out;
    crate::events::MemoryPolicyCompiledData {
        task_id: task_id.to_string(),
        turn_id: Some(turn_id.to_string()),
        // `access` excludes the turn from every automatic memory pipeline
        // (`memory_suppressed_turn_ids`), so it tracks persistence, not
        // recall.
        access: if persistence_allowed {
            crate::events::MemoryPipelineAccess::Allowed
        } else {
            crate::events::MemoryPipelineAccess::Suppressed
        },
        reason_code: if learning_opted_out {
            crate::agent::memory_plugin::OPTED_OUT_REASON.to_string()
        } else {
            policy.reason_code().to_string()
        },
        retrieval_suppressed: !policy.allows_recall(),
        persistence_suppressed: !persistence_allowed,
    }
}

/// JSON-object mode is rejected by a number of OpenAI-compatible gateways
/// unless the prompt contains a provider-specific lowercase marker.  The
/// daemon's semantic producers should not depend on prompt spelling.  A
/// permissive named schema requests the same object protocol while leaving
/// the domain parser as the authoritative validator for optional fields.
fn typed_object_response_format(name: &str) -> sdk::ResponseFormat {
    sdk::ResponseFormat::JsonSchema {
        name: name.to_string(),
        schema: json!({
            "type": "object",
            "additionalProperties": true
        }),
        strict: false,
    }
}

/// Resolve only the typed dialogue edge. This deliberately cannot produce
/// completion obligations, authority, tool policy, or execution mode. The
/// caller still validates an exact antecedent against persisted dialogue
/// state before committing it.
///
/// One tool-less side call through the gateway, which picks the model chain,
/// bounds every attempt by the spec's total deadline and records each one.
/// Any failure, an unreadable reply or an invalid edge returns `None`: the
/// turn is then treated as a new request.
pub(crate) async fn generate_task_relationship(
    side_calls: &SideCallGateway,
    user_text: &str,
    conversation_context: &str,
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
    let request = sdk::ModelRequest {
        // The side call's spec picks the model.
        model: String::new(),
        messages: vec![sdk::Message::system(system), sdk::Message::user(prompt)],
        max_output_tokens: Some(1024),
        options: sdk::RequestOptions {
            response_format: typed_object_response_format("task_relationship"),
            reasoning_effort: Some("low".to_string()),
            // Failover across the chain's models stays on: a used-up
            // primary quota must not cost the turn its relationship edge.
            single_attempt: false,
            fallback_when_unspent: false,
            cache_key: None,
        },
        ..Default::default()
    };
    let response = match side_calls
        .provider(SideCall::RelationshipRouter)
        .complete(&request)
        .await
    {
        Ok(response) => response,
        Err(error) => {
            warn!(%error, "Task relationship router got no answer");
            return None;
        }
    };
    let Some(json) = crate::utils::extract_json_object(&response.text) else {
        warn!("Task relationship router reply held no JSON object");
        return None;
    };
    let parsed: TaskRelationshipResponse = match serde_json::from_str(&json) {
        Ok(parsed) => parsed,
        Err(error) => {
            warn!(%error, "Task relationship router reply did not parse");
            return None;
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
    let valid = valid && (parsed.request_relationship == "continuation" || antecedent.is_none());
    if !valid {
        warn!("Task relationship router returned an invalid typed relationship");
        return None;
    }
    Some(PlannedTaskShape {
        confidence: Some(parsed.confidence),
        request_relationship: Some(parsed.request_relationship),
        antecedent_user_message_id: antecedent,
    })
}

/// The turn digest's request: one decision call per user message. The
/// relationship question is asked only when there is a conversation to
/// relate to; its options carry the antecedent, one `continuation_<n>`
/// option per earlier user message the conversation context names, so the
/// class and the antecedent come back as one answer.
pub(crate) struct TurnDigestRequest {
    pub request: aidaemon_decisions::DecisionRequest,
    /// `continuation_<n>` option -> user message id.
    pub antecedents: std::collections::BTreeMap<String, String>,
}

const RELATIONSHIP_CLASSES: [(&str, &str); 3] = [
    (
        "new_request",
        "The message starts a new, separate request, even if it is on a similar topic as earlier ones.",
    ),
    (
        "clarification_answer",
        "The message answers a question the assistant just asked.",
    ),
    (
        "courtesy",
        "Thanks, greetings or acknowledgements with no request.",
    ),
];

const TIERS: [(&str, &str); 3] = [
    (
        "fast",
        "Answerable right away from general knowledge or the conversation: no tools, no live or personal data, low stakes.",
    ),
    (
        "primary",
        "Needs tools, live or personal data, or a few steps of moderate work.",
    ),
    (
        "smart",
        "Needs deep reasoning, long multi-step work, code changes, or carries high stakes.",
    ),
];

/// Earlier user message ids named in the relationship context, in order,
/// without repeats. Reads the context's own row formats (`[message_id=X] U:`
/// and owed `USER message_id="X"` rows); it never reads message wording.
pub(crate) fn relationship_candidate_ids(conversation_context: &str) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    for line in conversation_context.lines() {
        let line = line.trim_start();
        let id = if let Some(rest) = line.strip_prefix("- [message_id=") {
            rest.split_once("] U:").map(|(id, _)| id)
        } else if let Some(rest) = line.strip_prefix("USER message_id=\"") {
            rest.split_once('"').map(|(id, _)| id)
        } else {
            None
        };
        if let Some(id) = id
            .map(str::trim)
            .filter(|id| !id.is_empty() && *id != "unknown")
        {
            if !ids.iter().any(|known| known == id) {
                ids.push(id.to_string());
            }
        }
    }
    ids
}

/// Builds the turn digest. `conversation_context` is the router's context
/// when the relationship should be asked, `None` otherwise.
pub(crate) fn turn_digest_request(
    user_text: &str,
    conversation_context: Option<&str>,
) -> TurnDigestRequest {
    use crate::llm_runtime::decisions::digest_keys;
    use aidaemon_decisions::{DecisionRequest, Question};

    let mut antecedents = std::collections::BTreeMap::new();
    let input = match conversation_context.filter(|context| !context.trim().is_empty()) {
        Some(context) => format!(
            "Recent conversation with stable message IDs:\n{}\n\nCurrent user message:\n{}",
            truncate_str(context, 6000),
            truncate_str(user_text, 4000)
        ),
        None => format!("Current user message:\n{}", truncate_str(user_text, 4000)),
    };
    let mut request = DecisionRequest::new(input);
    if let Some(context) = conversation_context.filter(|context| !context.trim().is_empty()) {
        let mut options: Vec<(String, String)> = RELATIONSHIP_CLASSES
            .iter()
            .map(|(key, description)| (key.to_string(), description.to_string()))
            .collect();
        // The newest candidates matter most; the choice holds at most 255.
        let ids = relationship_candidate_ids(context);
        let skip = ids.len().saturating_sub(64);
        for (n, id) in ids.into_iter().skip(skip).enumerate() {
            let key = format!("continuation_{n}");
            options.push((
                key.clone(),
                format!(
                    "The message advances, retries, corrects or asks about the earlier user message with message_id={id}."
                ),
            ));
            antecedents.insert(key, id);
        }
        request.insert(
            digest_keys::RELATIONSHIP,
            Question::choice(
                "How does the current user message relate to the conversation? Judge the whole meaning, not isolated words.",
                options,
            ),
        );
    }
    request.insert(
        digest_keys::IDENTITY,
        Question::yes_no_described(
            "Does the current user message state, correct or ask about who the user, the assistant, or the user's family members are (names, relationships)?",
            "Yes: it states or asks about such an identity, in any language.",
            "No: it does not, even if it mentions names or family in passing, or quotes text for another purpose.",
        ),
    );
    request.insert(
        digest_keys::TIER,
        Question::choice(
            "Which model tier should handle the current user message?",
            TIERS,
        ),
    );
    TurnDigestRequest {
        request,
        antecedents,
    }
}

/// The relationship edge the digest decided, when its answer may be acted
/// on. Shaped like the router's answer, so the rest of the turn is the same.
pub(crate) fn digest_relationship(
    outcome: &crate::llm_runtime::decisions::DecisionOutcome,
    antecedents: &std::collections::BTreeMap<String, String>,
) -> Option<PlannedTaskShape> {
    let answer = outcome.acted(crate::llm_runtime::decisions::digest_keys::RELATIONSHIP)?;
    let chosen = answer.chosen()?;
    let (relationship, antecedent) = match antecedents.get(chosen) {
        Some(id) => ("continuation", Some(id.clone())),
        None if RELATIONSHIP_CLASSES.iter().any(|(key, _)| *key == chosen) => (chosen, None),
        None => return None,
    };
    let shape = PlannedTaskShape {
        confidence: Some(
            if answer.confidence >= 0.8 {
                "high"
            } else {
                "medium"
            }
            .to_string(),
        ),
        request_relationship: Some(relationship.to_string()),
        antecedent_user_message_id: antecedent,
    };
    planned_task_relationship_is_complete(&shape).then_some(shape)
}

/// Whether the digest flags the message as identity-critical.
pub(crate) fn digest_identity(outcome: &crate::llm_runtime::decisions::DecisionOutcome) -> bool {
    outcome
        .acted(crate::llm_runtime::decisions::digest_keys::IDENTITY)
        .and_then(aidaemon_decisions::Answer::p_yes)
        .is_some_and(|p| p >= 0.5)
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
    use crate::llm_runtime::gateway::test_support::*;
    use std::sync::Arc;

    /// An opted-out turn's one policy event carries the opt-out's reason
    /// whatever else suppressed its memory, so history search (which keys on
    /// that reason) leaves it out; recall still follows the policy.
    #[test]
    fn an_opted_out_turn_records_the_opt_out_whatever_its_policy() {
        use aidaemon_memory::policy::TurnAuthor;
        let opted_out = crate::agent::memory_plugin::OPTED_OUT_REASON;
        for (author, forbidden) in [
            (TurnAuthor::Person, false),
            (TurnAuthor::Person, true),
            (TurnAuthor::Synthetic, false),
        ] {
            let policy = MemoryPipelinePolicy::compile(author, forbidden);
            let data = memory_policy_compiled_data("task-1", "turn-1", &policy, true);
            assert_eq!(data.reason_code, opted_out, "{policy:?}");
            assert_eq!(data.access, crate::events::MemoryPipelineAccess::Suppressed);
            assert!(data.persistence_suppressed);
            assert_eq!(data.retrieval_suppressed, !policy.allows_recall());

            let normal = memory_policy_compiled_data("task-1", "turn-1", &policy, false);
            assert_eq!(normal.reason_code, policy.reason_code());
            assert_eq!(normal.persistence_suppressed, !policy.allows_memory());
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

    const SETUP_CONTEXT: &str = "- [message_id=message-setup-1] USER: Synthetic setup\n\
         - [message_id=message-reply-1] ASSISTANT: Acknowledged";

    /// `None` is the router's safe default: the finalizer treats the turn as
    /// a new request.
    #[tokio::test]
    async fn a_router_that_fails_treats_the_message_as_a_new_request() {
        let (gateway, events) = gateway_with(provider_failing("503")).await;
        let relation =
            generate_task_relationship(&gateway, "and the second one?", SETUP_CONTEXT).await;
        assert!(relation.is_none());
        assert!(failed_calls(&events, SideCall::RelationshipRouter.spec().purpose).await >= 1);
        assert_eq!(
            ok_calls(&events, SideCall::RelationshipRouter.spec().purpose).await,
            0
        );
    }

    #[tokio::test]
    async fn a_router_reply_without_a_valid_edge_treats_the_message_as_a_new_request() {
        for reply in [
            "",
            "sure, continuation",
            r#"{"schema_version": 1, "confidence": "high"}"#,
            r#"{"schema_version": 1, "confidence": "high", "request_relationship": "new_request", "antecedent_user_message_id": "message-setup-1"}"#,
        ] {
            let (gateway, _) = gateway_with(provider_answering(reply)).await;
            let relation =
                generate_task_relationship(&gateway, "and the second one?", SETUP_CONTEXT).await;
            assert!(relation.is_none(), "reply {reply:?}");
        }
    }

    /// The gateway writes the router's row; the router writes none of its
    /// own. The request keeps today's output cap and carries no tools.
    #[tokio::test]
    async fn relationship_fallback_returns_only_an_exact_typed_dialogue_edge() {
        let scripted = Arc::new(
            aidaemon_core::testing::ScriptedProvider::new([]).then_repeat(
                aidaemon_core::testing::text(
                    r#"{
                        "schema_version": 1,
                        "confidence": "high",
                        "request_relationship": "continuation",
                        "antecedent_user_message_id": "message-setup-1"
                    }"#,
                ),
            ),
        );
        let (gateway, events) = gateway_with(scripted.clone()).await;

        let shape = generate_task_relationship(
            &gateway,
            "Return the values from that setup.",
            SETUP_CONTEXT,
        )
        .await
        .expect("typed relationship");

        assert_eq!(shape.request_relationship.as_deref(), Some("continuation"));
        assert_eq!(
            shape.antecedent_user_message_id.as_deref(),
            Some("message-setup-1")
        );
        let purpose = SideCall::RelationshipRouter.spec().purpose;
        assert_eq!(purpose, "task_relationship");
        assert_eq!(ok_calls(&events, purpose).await, 1);
        assert_eq!(failed_calls(&events, purpose).await, 0);
        let seen = scripted.requests();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].max_output_tokens, Some(1024));
        assert!(seen[0].tools.is_empty());
        assert_eq!(seen[0].tool_choice, sdk::ToolChoice::None);
    }

    #[tokio::test]
    async fn relationship_fallback_leaves_missing_identity_to_state_resolver() {
        let (gateway, _) = gateway_with(provider_answering(
            r#"{
                "schema_version": 1,
                "confidence": "high",
                "request_relationship": "continuation",
                "antecedent_user_message_id": null
            }"#,
        ))
        .await;

        let shape = generate_task_relationship(
            &gateway,
            "Continue.",
            "- [message_id=message-setup-1] USER: Synthetic setup",
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

#[cfg(test)]
mod digest_tests {
    use std::collections::BTreeMap;

    use aidaemon_decisions::{Answer, ConfidenceSource, DecisionResponse, QuestionKind};

    use super::*;
    use crate::llm_runtime::decide::gated_for_tests;
    use crate::llm_runtime::decisions::{digest_keys, Decision};

    const CONTEXT: &str = "[Session Summary] synthetic summary\n\
        - [message_id=msg-1] U: book a table for Alice Rivera\n\
        - [message_id=msg-2] A: Which day?\n\
        - [message_id=msg-3] U: check the build\n\n\
        Unfinished owner requests that a continuation may advance (copy the exact message_id):\n\
        USER message_id=\"msg-0\" (unfinished request): \"deploy the docs\"\n\
        USER message_id=\"msg-1\" (unfinished request): \"book a table\"";

    fn outcome(
        key: &str,
        answer: Answer,
        acting: bool,
    ) -> crate::llm_runtime::decisions::DecisionOutcome {
        let mut answers = BTreeMap::new();
        answers.insert(key.to_string(), answer);
        gated_for_tests(
            Decision::TurnDigest,
            DecisionResponse::new("test", answers, Default::default()),
            acting,
        )
    }

    #[test]
    fn candidates_are_user_rows_and_owed_requests_once_each() {
        assert_eq!(
            relationship_candidate_ids(CONTEXT),
            vec!["msg-1", "msg-3", "msg-0"]
        );
    }

    #[test]
    fn the_relationship_carries_one_option_per_candidate() {
        let digest = turn_digest_request("yes, Friday", Some(CONTEXT));
        let Some(question) = digest.request.questions.get(digest_keys::RELATIONSHIP) else {
            panic!("relationship asked");
        };
        let QuestionKind::Choice { options } = &question.kind else {
            panic!("a choice");
        };
        assert_eq!(options.len(), 3 + 3);
        assert_eq!(digest.antecedents.get("continuation_0").unwrap(), "msg-1");
        assert!(digest.request.validate().is_ok());
    }

    #[test]
    fn without_a_conversation_only_identity_and_tier_are_asked() {
        let digest = turn_digest_request("what is my daughter's name?", None);
        assert!(!digest
            .request
            .questions
            .contains_key(digest_keys::RELATIONSHIP));
        assert!(digest.request.questions.contains_key(digest_keys::IDENTITY));
        assert!(digest.request.questions.contains_key(digest_keys::TIER));
        assert!(digest.request.validate().is_ok());
    }

    #[test]
    fn a_continuation_option_becomes_the_routers_edge() {
        let digest = turn_digest_request("yes, Friday", Some(CONTEXT));
        let decided = outcome(
            digest_keys::RELATIONSHIP,
            Answer::choice("continuation_0", 0.85, ConfidenceSource::Calibrated),
            true,
        );
        let shape = digest_relationship(&decided, &digest.antecedents).unwrap();
        assert_eq!(shape.request_relationship.as_deref(), Some("continuation"));
        assert_eq!(shape.antecedent_user_message_id.as_deref(), Some("msg-1"));
        assert_eq!(shape.confidence.as_deref(), Some("high"));
    }

    #[test]
    fn a_class_answer_has_no_antecedent() {
        let digest = turn_digest_request("thanks!", Some(CONTEXT));
        let decided = outcome(
            digest_keys::RELATIONSHIP,
            Answer::choice("courtesy", 0.7, ConfidenceSource::Calibrated),
            true,
        );
        let shape = digest_relationship(&decided, &digest.antecedents).unwrap();
        assert_eq!(shape.request_relationship.as_deref(), Some("courtesy"));
        assert!(shape.antecedent_user_message_id.is_none());
        assert_eq!(shape.confidence.as_deref(), Some("medium"));
    }

    #[test]
    fn unsure_shadow_or_self_reported_answers_fall_back_to_the_router() {
        let digest = turn_digest_request("yes", Some(CONTEXT));
        for (answer, acting) in [
            (
                Answer::choice("new_request", 0.55, ConfidenceSource::Calibrated),
                true,
            ),
            (
                Answer::choice("new_request", 0.95, ConfidenceSource::Calibrated),
                false,
            ),
            (
                Answer::choice("new_request", 0.99, ConfidenceSource::SelfReported),
                true,
            ),
        ] {
            let decided = outcome(digest_keys::RELATIONSHIP, answer, acting);
            assert!(digest_relationship(&decided, &digest.antecedents).is_none());
        }
    }

    #[test]
    fn identity_is_flagged_only_on_yes() {
        let yes = outcome(
            digest_keys::IDENTITY,
            Answer::yes_no(0.7, ConfidenceSource::SelfReported),
            true,
        );
        assert!(digest_identity(&yes));
        let no = outcome(
            digest_keys::IDENTITY,
            Answer::yes_no(0.2, ConfidenceSource::Calibrated),
            true,
        );
        assert!(!digest_identity(&no));
        let shadow = outcome(
            digest_keys::IDENTITY,
            Answer::yes_no(0.9, ConfidenceSource::Calibrated),
            false,
        );
        assert!(!digest_identity(&shadow));
    }
}
