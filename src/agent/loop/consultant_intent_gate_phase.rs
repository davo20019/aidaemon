use super::consultant_direct_return::consultant_direct_return_ok;
use super::consultant_phase::ConsultantPhaseOutcome;
use super::*;
use crate::traits::ProviderResponse;

pub(super) struct ConsultantIntentGateData {
    pub intent_gate: IntentGateDecision,
    pub needs_tools: bool,
}

pub(super) enum ConsultantIntentGateOutcome {
    Return(ConsultantPhaseOutcome),
    Continue(ConsultantIntentGateData),
}

pub(super) struct ConsultantIntentGateCtx<'a> {
    pub resp: &'a mut ProviderResponse,
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub user_text: &'a str,
    pub iteration: usize,
    pub task_start: Instant,
    pub learning_ctx: &'a mut LearningContext,
    pub is_personal_memory_recall_turn: bool,
    pub is_reaffirmation_challenge_turn: bool,
    pub requests_external_verification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ConsultantRoutingContractOutcome {
    AskClarification(String),
    DirectReply(String),
    Continue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConsultantRouteReason {
    ClarificationRequired,
    ToolsRequired,
    ShortCorrectionDirectReply,
    AcknowledgmentDirectReply,
    DefaultContinue,
}

impl ConsultantRouteReason {
    fn as_str(self) -> &'static str {
        match self {
            ConsultantRouteReason::ClarificationRequired => "clarification_required",
            ConsultantRouteReason::ToolsRequired => "tools_required",
            ConsultantRouteReason::ShortCorrectionDirectReply => "short_correction_direct_reply",
            ConsultantRouteReason::AcknowledgmentDirectReply => "acknowledgment_direct_reply",
            ConsultantRouteReason::DefaultContinue => "default_continue",
        }
    }

    fn action(self) -> &'static str {
        match self {
            ConsultantRouteReason::ClarificationRequired
            | ConsultantRouteReason::ShortCorrectionDirectReply
            | ConsultantRouteReason::AcknowledgmentDirectReply => "return",
            ConsultantRouteReason::ToolsRequired | ConsultantRouteReason::DefaultContinue => {
                "continue"
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConsultantRoutingContractDecision {
    reason: ConsultantRouteReason,
    outcome: ConsultantRoutingContractOutcome,
}

fn evaluate_consultant_routing_contract(
    user_text: &str,
    intent_gate: &IntentGateDecision,
    user_is_short_correction: bool,
    needs_tools: bool,
    needs_clarification: bool,
) -> ConsultantRoutingContractDecision {
    if needs_clarification {
        // Override: if the user is clearly asking about code/files and tools are
        // needed, skip clarification and just proceed. The consultant sometimes
        // over-asks for read-only exploration like "search for TODOs" or
        // "how many lines does router.rs have?".
        let lower = user_text.to_ascii_lowercase();
        let is_exploration = needs_tools
            && (lower.contains("how many")
                || lower.contains("count")
                || lower.contains("search")
                || lower.contains("find")
                || lower.contains("grep")
                || lower.contains("list")
                || lower.contains("show me")
                || lower.contains("what's in")
                || lower.contains("read ")
                || lower.contains("look at")
                || lower.contains("check ")
                || lower.contains("todo")
                || lower.ends_with(".rs")
                || lower.ends_with(".py")
                || lower.ends_with(".js")
                || lower.ends_with(".ts")
                || lower.ends_with(".go"));
        if !is_exploration {
            let clarification = intent_gate
                .clarifying_question
                .clone()
                .filter(|q| q.contains('?'))
                .unwrap_or_else(|| {
                    default_clarifying_question(user_text, &intent_gate.missing_info)
                });
            return ConsultantRoutingContractDecision {
                reason: ConsultantRouteReason::ClarificationRequired,
                outcome: ConsultantRoutingContractOutcome::AskClarification(clarification),
            };
        }
        // Fall through to tools-required path for exploration requests
    }

    // Hard invariant: direct replies are never valid when tools are required.
    if needs_tools {
        return ConsultantRoutingContractDecision {
            reason: ConsultantRouteReason::ToolsRequired,
            outcome: ConsultantRoutingContractOutcome::Continue,
        };
    }

    if user_is_short_correction {
        return ConsultantRoutingContractDecision {
            reason: ConsultantRouteReason::ShortCorrectionDirectReply,
            outcome: ConsultantRoutingContractOutcome::DirectReply(
                "You're right — thanks for the correction.".to_string(),
            ),
        };
    }

    if intent_gate.is_acknowledgment.unwrap_or(false) {
        return ConsultantRoutingContractDecision {
            reason: ConsultantRouteReason::AcknowledgmentDirectReply,
            outcome: ConsultantRoutingContractOutcome::DirectReply("Got it.".to_string()),
        };
    }

    ConsultantRoutingContractDecision {
        reason: ConsultantRouteReason::DefaultContinue,
        outcome: ConsultantRoutingContractOutcome::Continue,
    }
}

impl Agent {
    pub(super) async fn run_consultant_intent_gate_phase(
        &self,
        ctx: &mut ConsultantIntentGateCtx<'_>,
    ) -> anyhow::Result<ConsultantIntentGateOutcome> {
        let resp = &mut *ctx.resp;
        let emitter = ctx.emitter;
        let task_id = ctx.task_id;
        let session_id = ctx.session_id;
        let user_text = ctx.user_text;
        let iteration = ctx.iteration;
        let task_start = ctx.task_start;
        let learning_ctx = &mut *ctx.learning_ctx;
        let is_personal_memory_recall_turn = ctx.is_personal_memory_recall_turn;
        let is_reaffirmation_challenge_turn = ctx.is_reaffirmation_challenge_turn;
        let requests_external_verification = ctx.requests_external_verification;
        // Try regular content first, then fall back to thinking output.
        // Gemini thinking models may put all useful content in thought
        // parts and only produce hallucinated tool calls as regular output.
        let raw_analysis = resp
            .content
            .as_ref()
            .filter(|s| !s.trim().is_empty())
            .cloned()
            .or_else(|| {
                resp.thinking
                    .as_ref()
                    .filter(|s| !s.trim().is_empty())
                    .map(|t| {
                        info!(
                        session_id,
                        thinking_len = t.len(),
                        "Consultant pass: using thinking output as fallback (no regular content)"
                    );
                        t.clone()
                    })
            })
            .unwrap_or_default();
        let (analysis_without_gate, model_intent_gate) = extract_intent_gate(&raw_analysis);
        let analysis = sanitize_consultant_analysis(&analysis_without_gate);
        let inferred_gate = infer_intent_gate(user_text, &analysis);
        let deterministic_tools_required = inferred_gate.needs_tools.unwrap_or(false);
        let mut intent_gate = merge_intent_gate_decision(model_intent_gate, inferred_gate);
        let is_acknowledgment = intent_gate.is_acknowledgment.unwrap_or(false);

        // Override: if user references a filesystem path, the consultant
        // (text-only, no tools) can never fulfil the request — force tools.
        let user_references_fs_path = user_text_references_filesystem_path(user_text);
        let user_is_short_correction = is_short_user_correction(user_text);
        // Semantic overrides — these detect intent from the LLM's BEHAVIOR,
        // not from word matching. They override the intent gate when there's
        // strong evidence the LLM needs tools.
        let had_hallucinated_tool_calls = !resp.tool_calls.is_empty();
        let analysis_defers_execution = looks_like_deferred_action_response(&analysis);

        let (mut can_answer_now, mut needs_tools, needs_clarification) = if user_references_fs_path
        {
            (false, true, false)
        } else if user_is_short_correction {
            info!(
                session_id,
                "Consultant pass: short user correction detected — forcing no-tools answer mode"
            );
            (true, false, false)
        } else if is_reaffirmation_challenge_turn && is_personal_memory_recall_turn {
            info!(
                session_id,
                "Consultant pass: reaffirmation challenge on personal recall — allowing one targeted memory check"
            );
            (false, true, false)
        } else if requests_external_verification {
            info!(
                session_id,
                "Consultant pass: user explicitly requested external verification — forcing tools mode"
            );
            (false, true, false)
        } else if had_hallucinated_tool_calls {
            // Strongest signal: the LLM literally tried to call tools
            // in text-only mode. It clearly cannot answer without them.
            info!(
                session_id,
                dropped_tool_calls = resp.tool_calls.len(),
                "Consultant pass: LLM attempted tool calls — forcing tools mode"
            );
            (false, true, false)
        } else if analysis_defers_execution && intent_gate.needs_tools.is_none() {
            // Fallback: if the model omitted needs_tools but its analysis still promises
            // concrete future actions, force tool mode rather than trusting missing fields.
            info!(
                session_id,
                "Consultant pass: deferred-action text but needs_tools was omitted — forcing tools mode"
            );
            (false, true, false)
        } else {
            (
                intent_gate.can_answer_now.unwrap_or(false),
                intent_gate.needs_tools.unwrap_or(false),
                intent_gate.needs_clarification.unwrap_or(false),
            )
        };
        if deterministic_tools_required
            && !needs_tools
            && !user_is_short_correction
            && !is_acknowledgment
        {
            info!(
                session_id,
                "Consultant pass: deterministic local-execution signal detected — forcing tools mode"
            );
            can_answer_now = false;
            needs_tools = true;
            intent_gate.can_answer_now = Some(false);
            intent_gate.needs_tools = Some(true);
        }
        if can_answer_now
            && !needs_tools
            && raw_analysis.trim().is_empty()
            && !is_acknowledgment
            && !user_is_short_correction
        {
            info!(
                session_id,
                "Consultant pass: can_answer_now=true but raw analysis was empty — forcing tool retry"
            );
            can_answer_now = false;
            needs_tools = true;
            intent_gate.can_answer_now = Some(false);
            intent_gate.needs_tools = Some(true);
        } else if can_answer_now
            && !needs_tools
            && analysis.trim().is_empty()
            && !raw_analysis.trim().is_empty()
        {
            info!(
                session_id,
                raw_len = raw_analysis.len(),
                "Consultant pass: analysis was sanitized to empty but raw was non-empty — trusting can_answer_now"
            );
        }
        // Knowledge-complexity override: if the model classified the query
        // as "knowledge" (fully answerable from training data) but still set
        // needs_tools=true, trust the complexity signal over needs_tools.
        // This prevents stalls on simple conversational queries like
        // "Tell me a joke in Spanish" that the model can answer directly.
        if needs_tools
            && !user_references_fs_path
            && !deterministic_tools_required
            && !is_acknowledgment
            && intent_gate
                .complexity
                .as_deref()
                .is_some_and(|c| c == "knowledge" || c == "simple")
        {
            // Don't override if user text contains verbs that require tool execution.
            // Queries like "Find all TODO comments" or "Search for X in the codebase"
            // may be classified as simple/knowledge but genuinely need tools.
            let lower = user_text.to_ascii_lowercase();
            let requires_action = [
                "search",
                "find",
                "grep",
                "scan",
                "check ",
                "run ",
                "execute",
                "create ",
                "write ",
                "deploy",
                "build",
                "compile",
                "install",
                "todo",
                "fixme",
                "list all",
                "count all",
                "show me",
            ]
            .iter()
            .any(|kw| lower.contains(kw));

            if !requires_action {
                info!(
                    session_id,
                    complexity = intent_gate.complexity.as_deref().unwrap_or("unknown"),
                    "Consultant pass: simple/knowledge complexity overrides needs_tools — allowing direct answer"
                );
                can_answer_now = true;
                needs_tools = false;
            } else {
                info!(
                    session_id,
                    complexity = intent_gate.complexity.as_deref().unwrap_or("unknown"),
                    "Consultant pass: knowledge-complexity override BLOCKED by action verb in user text"
                );
            }
        }

        intent_gate.can_answer_now = Some(can_answer_now);
        intent_gate.needs_tools = Some(needs_tools);
        intent_gate.needs_clarification = Some(needs_clarification);

        if analysis.len() != raw_analysis.len() {
            info!(
                session_id,
                raw_len = raw_analysis.len(),
                sanitized_len = analysis.len(),
                "Consultant pass: sanitized control/pseudo-tool text from analysis"
            );
        }

        let routing_decision = evaluate_consultant_routing_contract(
            user_text,
            &intent_gate,
            user_is_short_correction,
            needs_tools,
            needs_clarification,
        );
        match routing_decision.reason {
            ConsultantRouteReason::ClarificationRequired => {
                POLICY_METRICS
                    .consultant_route_clarification_required_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            ConsultantRouteReason::ToolsRequired => {
                POLICY_METRICS
                    .consultant_route_tools_required_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            ConsultantRouteReason::ShortCorrectionDirectReply => {
                POLICY_METRICS
                    .consultant_route_short_correction_direct_reply_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            ConsultantRouteReason::AcknowledgmentDirectReply => {
                POLICY_METRICS
                    .consultant_route_acknowledgment_direct_reply_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            ConsultantRouteReason::DefaultContinue => {
                POLICY_METRICS
                    .consultant_route_default_continue_total
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        let route_reason = routing_decision.reason.as_str();
        let route_action = routing_decision.reason.action();
        let route_reply_len = match &routing_decision.outcome {
            ConsultantRoutingContractOutcome::AskClarification(clarification) => {
                Some(clarification.trim().len())
            }
            ConsultantRoutingContractOutcome::DirectReply(reply) => Some(reply.trim().len()),
            ConsultantRoutingContractOutcome::Continue => None,
        };
        if route_action == "return" && route_reply_len == Some(0) {
            warn!(
                session_id,
                route_reason, "Consultant pass: empty direct reply candidate detected"
            );
        }

        info!(
            session_id,
            can_answer_now,
            needs_tools,
            needs_clarification,
            route_reason,
            route_action,
            missing_info = ?intent_gate.missing_info,
            domains = ?intent_gate.domains,
            "Consultant pass: intent gate decision"
        );

        if self.record_decision_points {
            self.emit_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::IntentGate,
                format!(
                    "Intent gate: answer_now={} needs_tools={} needs_clarification={} route={}",
                    can_answer_now, needs_tools, needs_clarification, route_reason
                ),
                json!({
                    "can_answer_now": can_answer_now,
                    "needs_tools": needs_tools,
                    "needs_clarification": needs_clarification,
                    "route_reason": route_reason,
                    "route_action": route_action,
                    "route_reply_len": route_reply_len,
                    "domains": intent_gate.domains.clone(),
                    "missing_info": intent_gate.missing_info.clone()
                }),
            )
            .await;
        }

        if let Some(drift_signal) =
            observe_route_reason_for_drift(session_id, route_reason, route_action, route_reply_len)
        {
            warn!(
                session_id,
                failsafe_activated = drift_signal.failsafe_activated,
                summary = %drift_signal.summary,
                "Consultant route drift monitor triggered"
            );
            if self.record_decision_points {
                self.emit_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::RouteDriftAlert,
                    drift_signal.summary.clone(),
                    json!({
                        "route_reason": route_reason,
                        "route_action": route_action,
                        "route_reply_len": route_reply_len,
                        "failsafe_activated": drift_signal.failsafe_activated
                    }),
                )
                .await;
            }
        }

        if !intent_gate.domains.is_empty() {
            learning_ctx.intent_domains = intent_gate.domains.clone();
        }

        match routing_decision.outcome {
            ConsultantRoutingContractOutcome::AskClarification(clarification) => {
                POLICY_METRICS
                    .ambiguity_detected_total
                    .fetch_add(1, Ordering::Relaxed);
                info!(
                    session_id,
                    route_reason,
                    clarification = %clarification,
                    "Consultant pass: routing contract selected clarification"
                );
                let assistant_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: "assistant".to_string(),
                    content: Some(clarification.clone()),
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.5,
                    embedding: None,
                };
                self.append_assistant_message_with_event(
                    emitter,
                    &assistant_msg,
                    "system",
                    None,
                    None,
                )
                .await?;

                self.emit_task_end(
                    emitter,
                    task_id,
                    TaskStatus::Completed,
                    task_start,
                    iteration,
                    0,
                    None,
                    Some(clarification.chars().take(200).collect()),
                )
                .await;
                return Ok(ConsultantIntentGateOutcome::Return(
                    consultant_direct_return_ok(clarification),
                ));
            }
            ConsultantRoutingContractOutcome::DirectReply(reply) => {
                info!(
                    session_id,
                    route_reason,
                    reply_len = reply.len(),
                    short_correction = user_is_short_correction,
                    is_acknowledgment,
                    "Consultant pass: routing contract selected direct reply"
                );
                let assistant_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: "assistant".to_string(),
                    content: Some(reply.clone()),
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.5,
                    embedding: None,
                };
                self.append_assistant_message_with_event(
                    emitter,
                    &assistant_msg,
                    "system",
                    None,
                    None,
                )
                .await?;

                self.emit_task_end(
                    emitter,
                    task_id,
                    TaskStatus::Completed,
                    task_start,
                    iteration,
                    0,
                    None,
                    Some(reply.chars().take(200).collect()),
                )
                .await;
                return Ok(ConsultantIntentGateOutcome::Return(
                    consultant_direct_return_ok(reply),
                ));
            }
            ConsultantRoutingContractOutcome::Continue => {
                info!(
                    session_id,
                    route_reason, "Consultant pass: routing contract selected continue"
                );
            }
        }

        Ok(ConsultantIntentGateOutcome::Continue(
            ConsultantIntentGateData {
                intent_gate,
                needs_tools,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[derive(Debug, Clone, Copy)]
    enum Expectation {
        AskClarification,
        DirectReply(&'static str),
        Continue,
    }

    fn base_intent_gate() -> IntentGateDecision {
        IntentGateDecision {
            can_answer_now: None,
            needs_tools: None,
            needs_clarification: None,
            clarifying_question: None,
            missing_info: Vec::new(),
            complexity: None,
            cancel_intent: None,
            cancel_scope: None,
            is_acknowledgment: None,
            schedule: None,
            schedule_type: None,
            schedule_cron: None,
            domains: Vec::new(),
        }
    }

    #[test]
    fn consultant_routing_contract_table_driven() {
        struct Case {
            name: &'static str,
            user_text: &'static str,
            gate: IntentGateDecision,
            short_correction: bool,
            needs_tools: bool,
            needs_clarification: bool,
            expected_reason: ConsultantRouteReason,
            expected: Expectation,
        }

        let mut clarify_with_valid_q = base_intent_gate();
        clarify_with_valid_q.clarifying_question = Some("Which environment should I use?".into());

        let mut clarify_with_invalid_q = base_intent_gate();
        clarify_with_invalid_q.clarifying_question = Some("Need environment".into());

        let mut ack_gate = base_intent_gate();
        ack_gate.is_acknowledgment = Some(true);

        let cases = vec![
            Case {
                name: "clarification_with_valid_question",
                user_text: "deploy it",
                gate: clarify_with_valid_q,
                short_correction: false,
                needs_tools: false,
                needs_clarification: true,
                expected_reason: ConsultantRouteReason::ClarificationRequired,
                expected: Expectation::AskClarification,
            },
            Case {
                name: "clarification_with_invalid_question_uses_default",
                user_text: "deploy it",
                gate: clarify_with_invalid_q,
                short_correction: false,
                needs_tools: false,
                needs_clarification: true,
                expected_reason: ConsultantRouteReason::ClarificationRequired,
                expected: Expectation::AskClarification,
            },
            Case {
                name: "ack_without_tools_direct_reply",
                user_text: "yes",
                gate: ack_gate.clone(),
                short_correction: false,
                needs_tools: false,
                needs_clarification: false,
                expected_reason: ConsultantRouteReason::AcknowledgmentDirectReply,
                expected: Expectation::DirectReply("Got it."),
            },
            Case {
                name: "ack_with_tools_falls_through",
                user_text: "yes do it",
                gate: ack_gate,
                short_correction: false,
                needs_tools: true,
                needs_clarification: false,
                expected_reason: ConsultantRouteReason::ToolsRequired,
                expected: Expectation::Continue,
            },
            Case {
                name: "short_correction_without_tools_direct_reply",
                user_text: "you did send me the file",
                gate: base_intent_gate(),
                short_correction: true,
                needs_tools: false,
                needs_clarification: false,
                expected_reason: ConsultantRouteReason::ShortCorrectionDirectReply,
                expected: Expectation::DirectReply("You're right — thanks for the correction."),
            },
            Case {
                name: "short_correction_with_tools_falls_through",
                user_text: "you did send me the file",
                gate: base_intent_gate(),
                short_correction: true,
                needs_tools: true,
                needs_clarification: false,
                expected_reason: ConsultantRouteReason::ToolsRequired,
                expected: Expectation::Continue,
            },
            Case {
                name: "default_continue",
                user_text: "check the deployment status",
                gate: base_intent_gate(),
                short_correction: false,
                needs_tools: false,
                needs_clarification: false,
                expected_reason: ConsultantRouteReason::DefaultContinue,
                expected: Expectation::Continue,
            },
        ];

        for case in cases {
            let decision = evaluate_consultant_routing_contract(
                case.user_text,
                &case.gate,
                case.short_correction,
                case.needs_tools,
                case.needs_clarification,
            );
            assert_eq!(
                decision.reason, case.expected_reason,
                "{}: expected reason {:?}, got {:?}",
                case.name, case.expected_reason, decision.reason
            );
            assert!(
                !decision.reason.as_str().trim().is_empty(),
                "{}: reason code must be non-empty",
                case.name
            );

            match (case.expected, decision.outcome) {
                (
                    Expectation::AskClarification,
                    ConsultantRoutingContractOutcome::AskClarification(q),
                ) => {
                    assert!(
                        q.contains('?'),
                        "{}: clarification must contain '?'",
                        case.name
                    );
                }
                (
                    Expectation::DirectReply(expected),
                    ConsultantRoutingContractOutcome::DirectReply(reply),
                ) => {
                    assert_eq!(reply, expected, "{}", case.name);
                    assert!(
                        !reply.trim().is_empty(),
                        "{}: direct reply must be non-empty",
                        case.name
                    );
                }
                (Expectation::Continue, ConsultantRoutingContractOutcome::Continue) => {}
                (expected, got) => panic!("{}: expected {:?}, got {:?}", case.name, expected, got),
            }
        }
    }

    #[test]
    fn consultant_routing_contract_never_direct_when_tools_needed() {
        let mut gate = base_intent_gate();
        gate.is_acknowledgment = Some(true);

        for short_correction in [false, true] {
            let decision = evaluate_consultant_routing_contract(
                "yes do it",
                &gate,
                short_correction,
                true,
                false,
            );
            assert!(
                matches!(decision.outcome, ConsultantRoutingContractOutcome::Continue),
                "needs_tools=true must never produce direct reply (short_correction={})",
                short_correction
            );
            assert_eq!(decision.reason, ConsultantRouteReason::ToolsRequired);
        }
    }

    /// Verify that the action-verb keywords used in the knowledge-complexity
    /// override guard correctly detect user text that requires tool execution.
    #[test]
    fn action_verb_guard_detects_tool_requiring_text() {
        let action_keywords: &[&str] = &[
            "search",
            "find",
            "grep",
            "scan",
            "check ",
            "run ",
            "execute",
            "create ",
            "write ",
            "deploy",
            "build",
            "compile",
            "install",
            "todo",
            "fixme",
            "list all",
            "count all",
            "show me",
        ];

        let should_block = [
            "Find all TODO or FIXME comments in the aidaemon codebase",
            "search for unused imports",
            "grep for async fn across all files",
            "Show me the deployment config",
            "Run cargo test",
            "check if the server is running",
            "create a new config file",
            "build the project with release flags",
            "List all endpoints in the API",
            "count all lines of Rust code",
        ];
        for text in should_block {
            let lower = text.to_ascii_lowercase();
            let detected = action_keywords.iter().any(|kw| lower.contains(kw));
            assert!(
                detected,
                "expected action verb guard to block '{}', but it didn't",
                text
            );
        }

        let should_allow = [
            "Tell me a joke in Spanish",
            "What is the capital of France?",
            "Explain how HTTP works",
            "How old is the universe?",
            "thanks",
            "yes",
        ];
        for text in should_allow {
            let lower = text.to_ascii_lowercase();
            let detected = action_keywords.iter().any(|kw| lower.contains(kw));
            assert!(
                !detected,
                "expected action verb guard to allow '{}', but it blocked it",
                text
            );
        }
    }

    proptest! {
        #[test]
        fn consultant_routing_contract_randomized_precedence_and_invariants(
            user_text in ".{0,40}",
            is_ack in any::<bool>(),
            short_correction in any::<bool>(),
            needs_tools in any::<bool>(),
            needs_clarification in any::<bool>(),
            clarifying_question_has_q in any::<bool>(),
        ) {
            let mut gate = base_intent_gate();
            gate.is_acknowledgment = Some(is_ack);
            gate.clarifying_question = Some(if clarifying_question_has_q {
                "Which environment should I use?".to_string()
            } else {
                "Need environment".to_string()
            });
            gate.missing_info = vec!["environment".to_string()];

            let decision = evaluate_consultant_routing_contract(
                &user_text,
                &gate,
                short_correction,
                needs_tools,
                needs_clarification,
            );

            // Contract precedence must remain stable.
            if needs_clarification {
                prop_assert_eq!(decision.reason, ConsultantRouteReason::ClarificationRequired);
                match decision.outcome {
                    ConsultantRoutingContractOutcome::AskClarification(q) => {
                        prop_assert!(q.contains('?'));
                    }
                    _ => prop_assert!(false, "needs_clarification=true must ask clarification"),
                }
            } else if needs_tools {
                prop_assert_eq!(decision.reason, ConsultantRouteReason::ToolsRequired);
                prop_assert!(matches!(decision.outcome, ConsultantRoutingContractOutcome::Continue));
            } else if short_correction {
                prop_assert_eq!(
                    decision.reason,
                    ConsultantRouteReason::ShortCorrectionDirectReply
                );
                match decision.outcome {
                    ConsultantRoutingContractOutcome::DirectReply(reply) => {
                        prop_assert!(!reply.trim().is_empty());
                    }
                    _ => prop_assert!(false, "short correction without tools must direct reply"),
                }
            } else if is_ack {
                prop_assert_eq!(
                    decision.reason,
                    ConsultantRouteReason::AcknowledgmentDirectReply
                );
                match decision.outcome {
                    ConsultantRoutingContractOutcome::DirectReply(reply) => {
                        prop_assert!(!reply.trim().is_empty());
                    }
                    _ => prop_assert!(false, "ack without tools must direct reply"),
                }
            } else {
                prop_assert_eq!(decision.reason, ConsultantRouteReason::DefaultContinue);
                prop_assert!(matches!(decision.outcome, ConsultantRoutingContractOutcome::Continue));
            }
        }
    }
}
