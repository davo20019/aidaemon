use super::types::{BootstrapCtx, BootstrapData, BootstrapOutcome};
use crate::agent::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::agent) struct GateFilterStage {
    stage: &'static str,
    before_count: usize,
    after_count: usize,
    action: &'static str,
    shadow_after_count: Option<usize>,
}

impl GateFilterStage {
    pub(in crate::agent) fn new(
        stage: &'static str,
        before_count: usize,
        after_count: usize,
        action: &'static str,
    ) -> Self {
        Self {
            stage,
            before_count,
            after_count,
            action,
            shadow_after_count: None,
        }
    }

    fn shadow(stage: &'static str, before_count: usize, shadow_after_count: usize) -> Self {
        Self {
            stage,
            before_count,
            after_count: before_count,
            action: "shadow_observed",
            shadow_after_count: Some(shadow_after_count),
        }
    }
}

fn build_tool_filter_gate_telemetry(
    stages: &[GateFilterStage],
    initial_tool_count: usize,
    final_tool_count: usize,
) -> (String, Value) {
    let removed_tool_count = initial_tool_count.saturating_sub(final_tool_count);
    let filtered_stage_count = stages
        .iter()
        .filter(|stage| stage.before_count != stage.after_count)
        .count();
    let summary = format!(
        "Tool filter gates evaluated: {initial_tool_count} -> {final_tool_count} tools ({removed_tool_count} removed)"
    );
    let stage_metadata: Vec<Value> = stages
        .iter()
        .map(|stage| {
            json!({
                "stage": stage.stage,
                "before_count": stage.before_count,
                "after_count": stage.after_count,
                "removed_count": stage.before_count.saturating_sub(stage.after_count),
                "shadow_after_count": stage.shadow_after_count,
                "action": stage.action,
            })
        })
        .collect();
    (
        summary,
        json!({
            "condition": "tool_filtering",
            "gate_family": "tool_availability",
            "action": if removed_tool_count > 0 { "filtered" } else { "passed" },
            "initial_tool_count": initial_tool_count,
            "final_tool_count": final_tool_count,
            "removed_tool_count": removed_tool_count,
            "filtered_stage_count": filtered_stage_count,
            "stages": stage_metadata,
        }),
    )
}

/// Commit one authoritative task envelope before any optional memory access,
/// tool filtering, project-instruction loading, or prompt construction.
/// Relationship, contract, and task ownership are deliberately finalized in
/// that order so a provisional ingress classification cannot leak an older
/// request into the new task or erase a background parent's lineage.
#[allow(clippy::too_many_arguments)]
async fn finalize_turn_assessment(
    agent: &Agent,
    emitter: &crate::events::EventEmitter,
    session_id: &str,
    task_id: &str,
    user_text: &str,
    internal_continuation: bool,
    structural_resume: bool,
    initial_turn_context: &TurnContext,
    relationship_fallback: Option<&super::task_planning::PlannedTaskShape>,
) -> anyhow::Result<TurnContext> {
    // Only the relationship producer may select an antecedent.
    let relationship_shape = relationship_fallback
        .filter(|shape| super::task_planning::planned_task_relationship_is_complete(shape));

    let mut requested_antecedent_user_message_id = None;
    let mut committed_antecedent_user_message_id = None;
    let mut resolved_request_user_message_id = None;
    if !internal_continuation {
        let dialogue_state = agent
            .state
            .get_dialogue_state(session_id)
            .await
            .ok()
            .flatten();
        let typed_clarification = initial_turn_context.followup_mode
            == Some(crate::agent::followup::FollowupMode::ClarificationAnswer);
        let (relationship, reason_code) = if structural_resume {
            ("continuation", "runtime_resume_edge")
        } else if let Some(shape) = relationship_shape {
            let requested = shape
                .request_relationship
                .as_deref()
                .unwrap_or("new_request");
            let requested_antecedent = shape
                .antecedent_user_message_id
                .as_deref()
                .map(str::trim)
                .filter(|message_id| !message_id.is_empty());
            requested_antecedent_user_message_id = requested_antecedent.map(str::to_string);
            let resolved_antecedent = dialogue_state.as_ref().and_then(|state| {
                if let Some(message_id) = requested_antecedent {
                    crate::agent::dialogue_state::resolve_request_antecedent(state, message_id)
                        .map(|request| (message_id.to_string(), request.user_message_id.clone()))
                } else {
                    crate::agent::dialogue_state::unambiguous_request_antecedent(state).map(
                        |request| {
                            (
                                request.user_message_id.clone(),
                                request.user_message_id.clone(),
                            )
                        },
                    )
                }
            });
            let exact_antecedent = resolved_antecedent.is_some();
            let relationship = match requested {
                "continuation" if exact_antecedent => "continuation",
                "clarification_answer" if typed_clarification => "clarification_answer",
                "courtesy" => "courtesy",
                _ => "new_request",
            };
            if relationship == "continuation" {
                committed_antecedent_user_message_id = resolved_antecedent
                    .as_ref()
                    .map(|(message_id, _)| message_id.clone());
            }
            resolved_request_user_message_id = resolved_antecedent
                .as_ref()
                .map(|(_, request_id)| request_id.clone());
            let reason = if requested == "continuation" && !exact_antecedent {
                "continuation_missing_exact_antecedent"
            } else {
                "assessment_committed"
            };
            (relationship, reason)
        } else if typed_clarification {
            ("clarification_answer", "typed_clarification_edge")
        } else {
            // A failed/unavailable language assessment cannot authorize an
            // adoption edge. Ordinary ingress therefore fails closed to a
            // fresh request; only runtime continuations and typed pending
            // questions bypass this rule.
            ("new_request", "assessment_unavailable_fresh_task")
        };

        if let Err(error) = crate::agent::dialogue_state::record_dialogue_semantic_user_turn(
            agent,
            session_id,
            user_text,
            relationship,
        )
        .await
        {
            warn!(session_id, %error, "Failed to persist finalized dialogue relationship");
        }
        agent
            .emit_decision_point(
                emitter,
                task_id,
                0,
                crate::events::DecisionType::IntentGate,
                "Finalized task relationship".to_string(),
                json!({
                    "condition": "task_relationship_finalized",
                    "relationship": relationship,
                    "reason_code": reason_code,
                    "requested_antecedent_user_message_id": requested_antecedent_user_message_id,
                    "antecedent_user_message_id": committed_antecedent_user_message_id,
                    "resolved_request_user_message_id": resolved_request_user_message_id,
                }),
            )
            .await;
    }

    // Rebuild after relationship commit. This second pass is intentional: it
    // is the first point at which context, contract inheritance, and project
    // carryover share the same authoritative relationship.
    let mut turn_context = agent
        .build_turn_context_from_recent_history_with_origin(
            session_id,
            user_text,
            internal_continuation,
        )
        .await;
    turn_context.visible_antecedent_user_message_id = committed_antecedent_user_message_id;
    if agent.mandate_execution.is_none() && !turn_context.inherited_completion_contract {
        crate::agent::retain_structural_completion_contract(&mut turn_context.completion_contract);
    }
    // Presentation follows the request lifecycle, not session recency. A new
    // external request clears it; a structural resume or a validated adoption
    // of unfinished obligations retains the parent artifact.
    if !(internal_continuation
        || structural_resume
        || turn_context.inherited_outstanding_obligations)
    {
        turn_context.completion_contract.response_contract = None;
    }
    turn_context.completion_contract.adopt_for_task(task_id);

    // Persist the finalized contract on its own correctness-critical event.
    // Decision points are an optional flight recorder and may be disabled for
    // cost/volume reasons; task outcome reconciliation must still survive a
    // provider failure, restart, or finalizer fallback with the exact same
    // typed obligation set.
    let persisted_contract =
        crate::agent::persistable_completion_contract(&turn_context.completion_contract);
    let required_invocations = turn_context
        .completion_contract
        .evidence_requirements
        .iter()
        .filter(|requirement| requirement.purpose == crate::traits::EvidencePurpose::Outcome)
        .filter_map(|requirement| requirement.receipt.clone())
        .fold(Vec::new(), |mut values, receipt| {
            if !values.contains(&receipt) {
                values.push(receipt);
            }
            values
        });
    emitter
        .emit(
            EventType::TaskContractCompiled,
            crate::events::TaskContractCompiledData {
                schema_version: crate::events::TaskContractCompiledData::SCHEMA_VERSION,
                task_id: task_id.to_string(),
                contract: persisted_contract,
                required_invocations,
            },
        )
        .await
        .map_err(|error| {
            anyhow::anyhow!(
                "could not persist correctness-critical task contract for {task_id}: {error}"
            )
        })?;

    // Task ownership is bound only after the relationship and inherited
    // contract are frozen. This prevents a new or internal child TaskStart
    // from rewriting the antecedent before it can be adopted.
    if let Err(error) =
        crate::agent::dialogue_state::record_dialogue_task_start(agent, session_id, task_id).await
    {
        warn!(session_id, task_id, %error, "Failed to bind finalized dialogue task");
    }
    if !internal_continuation {
        if let Err(error) = crate::agent::dialogue_state::record_dialogue_completion_contract(
            agent,
            session_id,
            user_text,
            &turn_context.completion_contract,
        )
        .await
        {
            warn!(session_id, %error, "Failed to persist finalized completion contract");
        }
    }

    Ok(turn_context)
}

/// Build mandate worker bootstrap state without consulting any owner-turn
/// routing surface. Mandate child agents already carry an immutable execution
/// fence and a role/authority-scoped registered tool set from spawn. Their
/// bootstrap must therefore be query-independent: no local skill matching,
/// conversation/project history, generic policy filtering, route fail-safe, or
/// model routing may reinterpret the model-authored task message.
#[allow(clippy::too_many_arguments)]
async fn build_isolated_mandate_bootstrap(
    agent: &Agent,
    session_id: &str,
    user_text: &str,
    user_role: UserRole,
    channel_ctx: &ChannelContext,
    task_id: String,
    user_msg_id: String,
    emitter: crate::events::EventEmitter,
) -> anyhow::Result<BootstrapOutcome> {
    anyhow::ensure!(
        agent.mandate_execution.is_some(),
        "isolated mandate bootstrap requires an immutable execution fence"
    );
    anyhow::ensure!(
        user_role == UserRole::Owner && channel_ctx.visibility == ChannelVisibility::Internal,
        "mandate workers require the internal owner execution channel"
    );

    // This is the registered child roster only. `tool_definitions_with_capabilities`
    // additionally performs message-triggered MCP composition, which is outside
    // v1 mandate authority even when the resulting view would later fail closed.
    let (mut tool_defs, available_capabilities) =
        agent.registered_tool_definitions_with_capabilities();
    Agent::sort_tool_definitions_by_name(&mut tool_defs);
    let base_tool_defs = tool_defs.clone();

    let llm_runtime_snapshot = agent.llm_runtime.snapshot();
    // Spawn selected this worker's model before its immutable fence was built.
    // Keep that exact model and expose no generic router to later loop phases.
    let model = match tokio::time::timeout(Duration::from_secs(2), agent.model.read()).await {
        Ok(guard) => guard.clone(),
        Err(_) => {
            warn!(
                session_id,
                "Timed out reading mandate worker model; using the runtime primary"
            );
            llm_runtime_snapshot.primary_model()
        }
    };

    let (
        core_prompt_bytes,
        fresh_task_context_tail,
        continuation_task_context_tail,
        active_skill_names,
        project_instruction_tracker,
    ) = agent
        .build_system_prompt_for_message(
            &emitter,
            &task_id,
            session_id,
            user_text,
            user_role,
            channel_ctx,
            tool_defs.len(),
            None,
            false,
            None,
        )
        .await?;
    anyhow::ensure!(
        active_skill_names.is_empty() && project_instruction_tracker.is_none(),
        "isolated mandate prompt unexpectedly returned generic instruction state"
    );

    let mut harness_eval = HarnessEvalAccumulator::new(HarnessEvalSeed {
        task_id: task_id.clone(),
        turn_id: Some(user_msg_id),
        depth: agent.depth as u32,
        parent_task_id: agent.task_id.clone(),
        goal_id: agent.goal_id.clone(),
        durable_task_id: agent.task_id.clone(),
        completion_task_kind: "mandate_protocol".to_string(),
        followup_mode: None,
        config: agent.runtime.harness_eval_config.clone(),
    });
    harness_eval.record_bootstrap(
        "isolated_mandate_continue",
        Vec::new(),
        Some(ModelProfile::Strong),
    );

    let learning_ctx = LearningContext {
        user_text: user_text.to_string(),
        memory_persistence_allowed: false,
        intent_domains: Vec::new(),
        tool_calls: Vec::new(),
        errors: Vec::new(),
        first_error: None,
        recovery_actions: Vec::new(),
        start_time: Utc::now(),
        completed_naturally: false,
        explicit_positive_signals: 0,
        explicit_negative_signals: 0,
        task_outcome: None,
    };
    let mut turn_context = TurnContext {
        // This fixed daemon string prevents a model-authored task from becoming
        // a generic follow-up, project, or completion-routing instruction.
        goal_user_text:
            "Execute the built-in bounded mandate protocol for this exact worker fence.".to_string(),
        completion_contract: CompletionContract {
            task_kind: CompletionTaskKind::Monitor,
            requires_observation: true,
            ..Default::default()
        },
        ..Default::default()
    };
    turn_context.completion_contract.adopt_for_task(&task_id);
    let policy_bundle = crate::execution_policy::PolicyBundle {
        // The generic policy is retained only because downstream loop state has
        // a common shape. Strong keeps the already-scoped roster intact; it does
        // not select a model because mandate bootstrap exposes no router.
        policy: ExecutionPolicy::for_profile(ModelProfile::Strong),
        risk_score: 1.0,
        uncertainty_score: 0.0,
        confidence: 1.0,
    };

    Ok(BootstrapOutcome::Continue(Box::new(BootstrapData {
        user_text: user_text.to_string(),
        task_id,
        memory_pipeline_policy:
            super::task_planning::MemoryPipelinePolicy::SuppressedByCurrentContract,
        resume_execution_snapshot: None,
        emitter,
        learning_ctx,
        learning: Default::default(),
        active_untrusted_external_reference_skills: Vec::new(),
        restrict_untrusted_external_reference_tools: false,
        available_capabilities,
        base_tool_defs,
        tool_defs,
        policy_bundle,
        llm_router: None,
        model,
        turn_context,
        project_instruction_tracker,
        core_prompt_bytes,
        fresh_task_context_tail,
        continuation_task_context_tail,
        session_summary: None,
        harness_eval,
    })))
}

/// Whether an owner message asks to resume an interrupted task. The
/// keyword check decides unless the decision layer is on, an interrupted
/// task exists, and the backend answers with a calibrated confidence of 0.8
/// or more. In shadow mode the decision runs beside the turn, recorded only.
async fn wants_resume(agent: &Agent, session_id: &str, user_text: &str) -> bool {
    use crate::llm_runtime::decisions::Decision;
    let keyword = is_resume_request(user_text);
    let mode = agent.side_calls.decision_mode(Decision::Resume);
    if mode == crate::config::DecisionMode::Off {
        return keyword;
    }
    // Typed state first: with no interrupted task there is nothing to ask.
    let Some(active) = agent
        .event_store
        .get_active_task(session_id)
        .await
        .ok()
        .flatten()
    else {
        return keyword;
    };
    let description = active
        .parse_data::<crate::events::TaskStartData>()
        .ok()
        .map(|data| data.description)
        .unwrap_or_default();
    let request = aidaemon_decisions::DecisionRequest::new(format!(
        "An earlier task was interrupted before it finished:\n{}\n\nCurrent user message:\n{}",
        crate::utils::truncate_str(&description, 2000),
        crate::utils::truncate_str(user_text, 2000)
    ))
    .ask(
        "resume",
        aidaemon_decisions::Question::yes_no_described(
            "Does the current user message ask to resume or continue the interrupted task?",
            "Yes: it asks to pick the interrupted task back up, in any language.",
            "No: it is about something else, or asks a new question.",
        ),
    );
    if mode == crate::config::DecisionMode::Shadow {
        let side_calls = agent.side_calls.clone();
        let session = session_id.to_string();
        tokio::spawn(async move {
            side_calls
                .decide(
                    Decision::Resume,
                    request,
                    crate::llm_runtime::decide::DecisionContext {
                        session_id: Some(&session),
                        task_id: None,
                    },
                )
                .await;
        });
        return keyword;
    }
    let outcome = agent
        .side_calls
        .decide(
            Decision::Resume,
            request,
            crate::llm_runtime::decide::DecisionContext {
                session_id: Some(session_id),
                task_id: None,
            },
        )
        .await;
    outcome
        .acted("resume")
        .and_then(aidaemon_decisions::Answer::p_yes)
        .map_or(keyword, |p| p >= 0.5)
}

/// Stores the turn digest's identity flag on the user message, so
/// archived rendering keeps it verbatim. Advisory: a failure is logged.
async fn annotate_identity(
    state: &dyn crate::traits::StateStore,
    session_id: &str,
    message_id: &str,
    outcome: &crate::llm_runtime::decisions::DecisionOutcome,
) {
    if !super::task_planning::digest_identity(outcome) {
        return;
    }
    if let Err(error) = state
        .annotate_user_message(
            session_id,
            message_id,
            crate::traits::MessageAnnotation::IdentityCritical,
        )
        .await
    {
        warn!(session_id, %error, "Could not store the identity flag");
    }
}

pub(in crate::agent) async fn run_bootstrap(
    services: &crate::agent::services::AgentServices<'_>,
    ctx: &BootstrapCtx<'_>,
) -> anyhow::Result<BootstrapOutcome> {
    let agent = services.agent;
    let session_id = ctx.session_id;
    let status_tx = ctx.status_tx.clone();
    let user_role = ctx.user_role;
    let channel_ctx = ctx.channel_ctx.clone();
    info!(session_id, "Bootstrap phase started");

    let is_mandate_execution = agent.mandate_execution.is_some();
    anyhow::ensure!(
        !is_mandate_execution || ctx.attachments.is_empty(),
        "mandate workers cannot receive attachment-derived context"
    );
    let model_for_stt = agent.llm_runtime.snapshot().primary_model();
    let user_text_enriched = if is_mandate_execution {
        ctx.user_text.to_string()
    } else {
        crate::agent::stt::maybe_enrich_user_text(
            ctx.user_text,
            ctx.attachments,
            &agent.runtime.stt_config,
            &agent.runtime.audio_config,
            &model_for_stt,
        )
        .await
    };
    let user_text = user_text_enriched.as_str();
    // Resume checkpoints can contain prior tool arguments/results and broader
    // owner authority. A collaborator starts a fresh scoped turn instead of
    // inheriting an interrupted owner task from the shared session.
    let resume_checkpoint = if !is_mandate_execution
        && user_role == UserRole::Owner
        && wants_resume(agent, session_id, user_text).await
    {
        match crate::agent::resume::build_resume_checkpoint(agent, session_id).await {
            Ok(checkpoint) => checkpoint,
            Err(e) => {
                warn!(
                    session_id,
                    error = %e,
                    "Failed to build resume checkpoint; continuing without resume context"
                );
                None
            }
        }
    } else {
        None
    };
    let resumed_from_task_id = resume_checkpoint.as_ref().map(|c| c.task_id.clone());

    // Generate task ID for this request
    let task_id = Uuid::new_v4().to_string();

    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        crate::agent::resume::mark_task_interrupted_for_resume(
            agent, session_id, checkpoint, &task_id,
        )
        .await;
        info!(
            session_id,
            resumed_task_id = %checkpoint.task_id,
            new_task_id = %task_id,
            "Recovered in-progress task from checkpoint"
        );
    }

    // Create event emitter for this session/task
    let emitter =
        crate::events::EventEmitter::new(agent.event_store.clone(), session_id.to_string())
            .with_task_id(task_id.clone());

    let task_description = if let Some(checkpoint) = resume_checkpoint.as_ref() {
        format!("resume: {}", checkpoint.description)
    } else {
        user_text.to_string()
    };

    // The user message's own id is also the turn_id for this conversation
    // turn (see the longer note below). Generate and stash it BEFORE the
    // TaskStart emit so TaskStartData carries the active turn — recovery
    // correctness (Task 1, revision-log #14) depends on TaskStart being
    // turn-stamped.
    let user_msg_id = Uuid::new_v4().to_string();
    agent
        .runtime
        .current_turn_ids
        .write()
        .await
        .insert(session_id.to_string(), user_msg_id.clone());

    // Emit TaskStart event
    let task_started_at = Utc::now();
    let _ = emitter
        .emit(
            EventType::TaskStart,
            TaskStartData {
                task_id: task_id.clone(),
                description: task_description.chars().take(200).collect(),
                parent_task_id: resumed_from_task_id
                    .or_else(|| ctx.parent_task_id.map(str::to_string)),
                user_message: Some(user_text.to_string()),
                turn_id: Some(user_msg_id.clone()),
                spawned_by: agent
                    .inherited_authority
                    .as_ref()
                    .and_then(|inherited| inherited.parent_task.clone()),
            },
        )
        .await;
    if let Some(timing) = ctx.ingress_timing {
        let millis = |later: chrono::DateTime<Utc>, earlier: chrono::DateTime<Utc>| {
            (later - earlier).num_milliseconds().max(0)
        };
        agent
            .emit_decision_point(
                &emitter,
                &task_id,
                0,
                DecisionType::GateTelemetry,
                "Recorded inbound transport lifecycle".to_string(),
                json!({
                    "condition": "inbound_transport_lifecycle",
                    "platform": channel_ctx.platform,
                    "platform_message_at": timing.platform_message_at,
                    "transport_received_at": timing.transport_received_at,
                    "queue_entered_at": timing.queue_entered_at,
                    "agent_dispatched_at": timing.agent_dispatched_at,
                    "task_started_at": task_started_at,
                    "platform_to_receiver_ms": timing.platform_message_at.map(|at| millis(timing.transport_received_at, at)),
                    "receiver_to_queue_ms": timing.queue_entered_at.map(|at| millis(at, timing.transport_received_at)),
                    "receiver_to_dispatch_ms": millis(timing.agent_dispatched_at, timing.transport_received_at),
                    "queue_wait_ms": timing.queue_entered_at.map(|at| millis(timing.agent_dispatched_at, at)),
                    "dispatch_to_task_start_ms": millis(task_started_at, timing.agent_dispatched_at),
                }),
            )
            .await;
    }
    // 1. Persist the user message
    //
    // The user message's own id is also the turn_id for this conversation
    // turn (generated and stashed in `current_turn_ids` above, before the
    // TaskStart emit). It is stashed so every subsequent message written
    // during this turn (assistant replies, tool results) is auto-stamped
    // with the same turn_id by `append_message_canonical`. This lets
    // boundary detection in `message_build` group the turn without
    // inferring from message content — historically a source of bugs when
    // the same text is sent twice or arrives out of order.
    let user_msg = Message {
        content: Some(user_text.to_string()),
        importance: 0.5, // Will be updated from turn_importance below
        turn_id: Some(user_msg_id.clone()),
        attachments: ctx.attachments.to_vec(),
        annotations: ctx
            .internal_continuation
            .then_some(crate::traits::MessageAnnotation::InternalContinuation)
            .into_iter()
            .collect(),
        ..Message::new_runtime(user_msg_id.clone(), session_id, "user")
    };
    // Calculate heuristic score immediately
    let score = aidaemon_memory::turn_importance(&user_msg.role, user_msg.content.as_deref());
    let mut user_msg = user_msg;
    user_msg.importance = score;

    agent
        .append_user_message_with_event(
            &emitter,
            &user_msg,
            user_role,
            &channel_ctx,
            !ctx.attachments.is_empty(),
        )
        .await?;

    // All generic owner-turn interpretation begins below this point. Mandate
    // workers leave here with a deterministic built-in bootstrap state, so an
    // exact `stop`, a local skill trigger, or path-bearing prior history can
    // never divert/re-scope the immutable worker protocol.
    if is_mandate_execution {
        return build_isolated_mandate_bootstrap(
            agent,
            session_id,
            user_text,
            user_role,
            &channel_ctx,
            task_id,
            user_msg_id,
            emitter,
        )
        .await;
    }

    if let Some(reply) = super::shortcuts::maybe_handle_stop_command(
        agent,
        session_id,
        user_text,
        user_role,
        &channel_ctx,
        status_tx.clone(),
        &task_id,
        &emitter,
    )
    .await?
    {
        return Ok(BootstrapOutcome::Return(Ok(reply)));
    }

    // Initialize learning context for post-task learning
    let mut learning_ctx = LearningContext {
        user_text: user_text.to_string(),
        // Finalized below after the relationship is committed, before this context can
        // reach any post-task learning path.
        memory_persistence_allowed: false,
        intent_domains: Vec::new(),
        tool_calls: Vec::new(),
        errors: Vec::new(),
        first_error: None,
        recovery_actions: Vec::new(),
        start_time: Utc::now(),
        completed_naturally: false,
        explicit_positive_signals: 0,
        explicit_negative_signals: 0,
        task_outcome: None,
    };
    if let Some((label, is_positive)) = detect_explicit_outcome_signal(user_text) {
        if is_positive {
            learning_ctx.explicit_positive_signals =
                learning_ctx.explicit_positive_signals.saturating_add(1);
        } else {
            learning_ctx.explicit_negative_signals =
                learning_ctx.explicit_negative_signals.saturating_add(1);
        }
        info!(
            session_id,
            task_id = %task_id,
            signal = label,
            "Detected explicit outcome signal in user input"
        );
    }

    // Bootstrap tool exposure happens before per-task model selection. Use the
    // runtime primary model's trust tier to decide whether policy filtering is
    // merely observed or may narrow the roster.
    let llm_runtime_snapshot = agent.llm_runtime.snapshot();
    let llm_router = llm_runtime_snapshot.router();
    let autonomous_bootstrap = agent.trust_tier_for_model(&llm_runtime_snapshot.primary_model())
        == crate::agent::trust_tier::ModelTrustTier::Autonomous;

    // Tools are owner-only by default. A Guest can receive a tiny file-tool
    // subset only through an explicit, active, channel-bound workspace grant.
    // Orchestrator (depth 0) keeps its authorized tools available from iteration 1.
    // Sub-agents (depth > 0) get tools based on their role (set in spawn_child).
    let workspace_grant = channel_ctx.active_workspace_grant(user_role);
    let tools_allowed_for_user = user_role == UserRole::Owner || workspace_grant.is_some();

    let mut available_capabilities: HashMap<String, ToolCapabilities> = HashMap::new();
    let mut base_tool_defs: Vec<Value> = Vec::new();
    let mut tool_defs: Vec<Value> = Vec::new();
    let mut tool_filter_stages: Vec<GateFilterStage> = Vec::new();
    let mut initial_tool_count = 0usize;
    if tools_allowed_for_user {
        let (mut defs, mut caps) = agent.tool_definitions_with_capabilities(user_text).await;
        initial_tool_count = defs.len();
        tool_filter_stages.push(GateFilterStage::new(
            if workspace_grant.is_some() {
                "workspace_grant"
            } else {
                "owner_role"
            },
            initial_tool_count,
            initial_tool_count,
            "passed",
        ));

        if let Some(grant) = workspace_grant {
            let before = defs.len();
            defs.retain(|definition| {
                Agent::tool_name_from_definition(definition)
                    .is_some_and(|name| grant.allows_tool(name))
            });
            caps.retain(|name, _| grant.allows_tool(name));
            tool_filter_stages.push(GateFilterStage::new(
                "workspace_tool_allowlist",
                before,
                defs.len(),
                if before == defs.len() {
                    "passed"
                } else {
                    "filtered"
                },
            ));
        }

        // Filter tools by channel visibility
        if channel_ctx.visibility == ChannelVisibility::PublicExternal {
            let before = defs.len();
            let allowed = ["web_search", "remember_fact", "system_info"];
            defs.retain(|d| {
                Agent::tool_name_from_definition(d).is_some_and(|name| allowed.contains(&name))
            });
            caps.retain(|name, _| allowed.contains(&name.as_str()));
            tool_filter_stages.push(GateFilterStage::new(
                "public_external_allowlist",
                before,
                defs.len(),
                if before == defs.len() {
                    "passed"
                } else {
                    "filtered"
                },
            ));
        }

        // Desktop control (computer_use) is an owner-machine action and must not
        // be reachable from shared/public conversations — only DMs and internal
        // sessions. This stops a channel message that merely mentions the tool
        // from launching it.
        {
            let before = defs.len();
            Agent::restrict_desktop_control_for_visibility(
                &mut defs,
                &mut caps,
                channel_ctx.visibility,
            );
            if before != defs.len() {
                tool_filter_stages.push(GateFilterStage::new(
                    "desktop_control_visibility",
                    before,
                    defs.len(),
                    "filtered",
                ));
            }
        }

        available_capabilities = caps;
        base_tool_defs = defs.clone();
        tool_defs = defs;
    } else {
        tool_filter_stages.push(GateFilterStage::new("owner_role", 0, 0, "blocked"));
    }

    // Pillar A Task 6: canonicalize the roster to name-sorted order at the
    // bootstrap boundary so all later retain/filter/widen ops begin from a
    // stable order. The authoritative final sort happens on `effective_tool_defs`
    // in message_build, but starting canonical keeps intermediate ordering
    // deterministic.
    Agent::sort_tool_definitions_by_name(&mut base_tool_defs);
    Agent::sort_tool_definitions_by_name(&mut tool_defs);

    let policy_bundle = build_policy_bundle(user_text, &available_capabilities, false);
    if !tool_defs.is_empty() {
        let shadow_filtered = agent.filter_tool_definitions_for_policy(
            &tool_defs,
            &available_capabilities,
            &policy_bundle.policy,
            policy_bundle.risk_score,
            false,
        );
        POLICY_METRICS
            .tool_exposure_samples
            .fetch_add(1, Ordering::Relaxed);
        POLICY_METRICS
            .tool_exposure_before_sum
            .fetch_add(tool_defs.len() as u64, Ordering::Relaxed);
        POLICY_METRICS
            .tool_exposure_after_sum
            .fetch_add(shadow_filtered.len() as u64, Ordering::Relaxed);
        if agent.runtime.policy_config.policy_shadow_mode {
            info!(
                session_id,
                task_id = %task_id,
                exposed_before = tool_defs.len(),
                exposed_after = shadow_filtered.len(),
                risk_score = policy_bundle.risk_score,
                profile = ?policy_bundle.policy.model_profile,
                "Policy tool filter shadow comparison"
            );
        }
        if agent.runtime.policy_config.tool_filter_enforce && !autonomous_bootstrap {
            tool_filter_stages.push(GateFilterStage::new(
                "policy_filter",
                tool_defs.len(),
                shadow_filtered.len(),
                if tool_defs.len() == shadow_filtered.len() {
                    "passed"
                } else {
                    "filtered"
                },
            ));
            tool_defs = shadow_filtered;
        } else {
            tool_filter_stages.push(GateFilterStage::shadow(
                "policy_filter",
                tool_defs.len(),
                shadow_filtered.len(),
            ));
        }
    }

    // Provider + router came from the same runtime snapshot above, keeping
    // this task internally consistent even if runtime configuration reloads.

    // Model selection: autonomous primary models interpret the request directly.
    // Lexical policy scoring remains available for guided/local-model support,
    // but must not silently swap a capable primary model based on user wording.
    let (selected_model, routed) = agent
        .select_turn_model(
            &llm_runtime_snapshot,
            llm_router.as_ref(),
            policy_bundle.policy.model_profile,
            session_id,
        )
        .await;
    if routed {
        if agent.runtime.policy_config.policy_shadow_mode {
            info!(
                session_id,
                task_id = %task_id,
                new_profile = ?policy_bundle.policy.model_profile,
                new_model = %selected_model,
                risk_score = policy_bundle.risk_score,
                uncertainty_score = policy_bundle.uncertainty_score,
                confidence = policy_bundle.confidence,
                "Policy shadow routing snapshot (profile-to-model mapping)"
            );
        }
        info!(
            routed_model = %selected_model,
            policy_profile = ?policy_bundle.policy.model_profile,
            "Selected model for task"
        );
    }
    let model = selected_model.clone();

    // Compile task relationship and memory policy before any optional memory
    // access, so policy and finalization cannot diverge.
    let non_owner_shared_context = user_role != UserRole::Owner
        && matches!(
            channel_ctx.visibility,
            ChannelVisibility::PrivateGroup
                | ChannelVisibility::Public
                | ChannelVisibility::PublicExternal
        );

    let mut session_summary = if agent.runtime.context_window_config.enabled
        && agent.mandate_execution.is_none()
        && !non_owner_shared_context
    {
        agent
            .state
            .get_conversation_summary(session_id)
            .await
            .ok()
            .flatten()
            .filter(|summary| summary.last_turn_seq.is_some())
    } else {
        None
    };

    // Resolve turn/project context before the relationship router and prompt
    // construction. The main loop reuses this exact snapshot.
    let initial_turn_context = agent
        .build_turn_context_from_recent_history_with_origin(
            session_id,
            user_text,
            ctx.internal_continuation,
        )
        .await;
    let planner_skip_reason = if ctx.internal_continuation {
        Some("internal_continuation_uses_parent_contract")
    } else if resume_checkpoint.is_some() {
        Some("runtime_resume_uses_parent_contract")
    } else if agent.mandate_execution.is_some() {
        Some("mandate_cycle_uses_only_budgeted_main_loop_calls")
    } else {
        super::task_planning::planning_skip_reason(user_text, false)
    };
    let planner_context = super::task_planning::relationship_conversation_context(
        initial_turn_context.followup_mode,
        session_summary
            .as_ref()
            .map(|summary| summary.summary.as_str()),
        &initial_turn_context.assessment_recent_messages,
    );
    // Owed owner requests are continuation targets even when the transcript
    // window no longer shows them; the router must be able to name them.
    let owed_candidates = agent
        .state
        .get_dialogue_state(session_id)
        .await
        .ok()
        .flatten()
        .and_then(|state| {
            crate::agent::dialogue_state::owed_request_candidate_rows(&state, chrono::Utc::now())
        });
    // Only an existing conversation gets the candidates: an absent context
    // means no relationship producer runs, and that must stay true.
    let planner_context = match (planner_context, owed_candidates) {
        (Some(context), Some(owed)) => Some(format!("{context}\n\n{owed}")),
        (context, _) => context,
    };
    // Ingress already reduced an unresolved request to a typed
    // follow-up/clarification edge. Do not make that persisted topology depend
    // on a second language call; the finalizer resolves the exact graph node
    // (or fails closed on ambiguity).
    let relationship_needed = !(planner_skip_reason.is_some()
        || matches!(
            initial_turn_context.followup_mode,
            Some(
                crate::agent::followup::FollowupMode::Followup
                    | crate::agent::followup::FollowupMode::ClarificationAnswer
            )
        ))
        && planner_context.is_some();
    // The optional turn digest (decision layer): one call that can stand in
    // for the relationship router and flags identity messages and the
    // model tier. It runs only on turns a person wrote. It is awaited only
    // when an answer of it is acted on this turn; otherwise it runs beside
    // the turn and its identity flag lands on the stored message.
    let digest_mode = agent
        .side_calls
        .decision_mode(crate::llm_runtime::decisions::Decision::TurnDigest);
    let digest = (digest_mode != crate::config::DecisionMode::Off && planner_skip_reason.is_none())
        .then(|| {
            super::task_planning::turn_digest_request(
                user_text,
                relationship_needed
                    .then_some(planner_context.as_deref())
                    .flatten(),
            )
        });
    let digest_awaited = digest.is_some()
        && digest_mode == crate::config::DecisionMode::On
        && (relationship_needed
            || agent
                .side_calls
                .decision_mode(crate::llm_runtime::decisions::Decision::ModelRoute)
                == crate::config::DecisionMode::On);
    let router = async {
        match planner_context.as_deref() {
            // The relationship router classifies the request; it does not
            // need the strongest model. The gateway runs it on the fast tier
            // with the failover providers behind it, so a used-up primary
            // quota does not cost the turn its relationship edge.
            Some(context) if relationship_needed => {
                super::task_planning::generate_task_relationship(
                    &agent.side_calls,
                    user_text,
                    context,
                )
                .await
            }
            _ => None,
        }
    };
    let (relationship_fallback, digest_outcome) = match digest {
        Some(digest) if digest_awaited => {
            let outcome = agent
                .side_calls
                .decide(
                    crate::llm_runtime::decisions::Decision::TurnDigest,
                    digest.request,
                    crate::llm_runtime::decide::DecisionContext::turn(session_id, &task_id),
                )
                .await;
            let decided = super::task_planning::digest_relationship(&outcome, &digest.antecedents);
            let relationship = match decided {
                Some(shape) => {
                    info!(
                        session_id,
                        relationship = ?shape.request_relationship,
                        "Turn digest decided the relationship; router skipped"
                    );
                    Some(shape)
                }
                // Below threshold, failed or timed out: today's router.
                None => router.await,
            };
            (relationship, Some(outcome))
        }
        Some(digest) => {
            let side_calls = agent.side_calls.clone();
            let state = agent.state.clone();
            let session = session_id.to_string();
            let task = task_id.clone();
            let message_id = user_msg_id.clone();
            tokio::spawn(async move {
                let outcome = side_calls
                    .decide(
                        crate::llm_runtime::decisions::Decision::TurnDigest,
                        digest.request,
                        crate::llm_runtime::decide::DecisionContext::turn(&session, &task),
                    )
                    .await;
                annotate_identity(&*state, &session, &message_id, &outcome).await;
            });
            (router.await, None)
        }
        None => (router.await, None),
    };
    if let Some(outcome) = digest_outcome.as_ref() {
        annotate_identity(&*agent.state, session_id, &user_msg_id, outcome).await;
    }
    // Model routing (decision layer): only a turn that starts a new
    // request, on the default model choice (no owner override, no policy
    // route), moves to the fast or smart tier, and only on an answer
    // confident enough for that direction. A continuation stays put.
    let model = match (digest_outcome.as_ref(), llm_router.as_ref()) {
        (Some(outcome), Some(router))
            if !routed
                && agent
                    .side_calls
                    .decision_mode(crate::llm_runtime::decisions::Decision::ModelRoute)
                    == crate::config::DecisionMode::On
                && relationship_fallback.as_ref().is_none_or(|shape| {
                    shape.request_relationship.as_deref() == Some("new_request")
                })
                && !agent.model_override_active(session_id).await =>
        {
            match outcome
                .acted(crate::llm_runtime::decisions::digest_keys::TIER)
                .and_then(crate::llm_runtime::decisions::model_route_tier)
            {
                Some(tier) if !router.select(tier).trim().is_empty() => {
                    let routed_model = router.select(tier).to_string();
                    if routed_model != model {
                        info!(
                            session_id,
                            %tier,
                            from = %model,
                            to = %routed_model,
                            "Turn digest routed the new request to another model tier"
                        );
                    }
                    routed_model
                }
                _ => model,
            }
        }
        _ => model,
    };
    let turn_context = finalize_turn_assessment(
        agent,
        &emitter,
        session_id,
        &task_id,
        user_text,
        ctx.internal_continuation,
        resume_checkpoint.is_some(),
        &initial_turn_context,
        relationship_fallback.as_ref(),
    )
    .await?;
    // "Don't remember this" is settled before the turn's memory policy is
    // recorded, so the turn's one policy event already says so: background
    // intake never sees a private turn as allowed, and a turn suppressed for
    // another reason still carries the opt-out history search excludes. A
    // sub-agent shares its parent turn's learning. `/private` sent before
    // this message opts it out; only a message the user wrote consumes it.
    let learning = agent
        .inherited_authority
        .as_ref()
        .and_then(|parent| parent.learning.clone())
        .unwrap_or_default();
    if !ctx.internal_continuation
        && agent.depth == 0
        && agent.mandate_execution.is_none()
        && crate::agent::memory_plugin::take_next_message_private(
            &*agent.state,
            session_id,
            channel_ctx.sender_id.as_deref(),
        )
        .await
    {
        learning.opt_out();
    }
    let memory_pipeline_policy = super::task_planning::compile_memory_pipeline_policy(
        &turn_context.completion_contract,
        planner_skip_reason.is_none(),
    );
    // Persistence and recall are separate: a turn no person wrote may recall
    // but never persist.
    let memory_pipeline_allowed = memory_pipeline_policy.allows_memory();
    let memory_recall_allowed = memory_pipeline_policy.allows_recall();
    learning_ctx.memory_persistence_allowed = memory_pipeline_allowed;
    emitter
        .emit(
            EventType::MemoryPolicyCompiled,
            super::task_planning::memory_policy_compiled_data(
                &task_id,
                &user_msg_id,
                &memory_pipeline_policy,
                !learning.allowed(),
            ),
        )
        .await?;
    agent
        .emit_decision_point(
            &emitter,
            &task_id,
            0,
            DecisionType::GateTelemetry,
            "Compiled automatic memory policy".to_string(),
            json!({
                "condition": "memory_policy_compiled",
                "policy": if memory_pipeline_allowed {
                    "allowed"
                } else if memory_recall_allowed {
                    "recall_only"
                } else {
                    "suppressed"
                },
                "reason_code": memory_pipeline_policy.reason_code(),
                "retrieval_suppressed": !memory_recall_allowed,
                "memory_message_projection_suppressed": !memory_recall_allowed,
                "post_task_learning_suppressed": !memory_pipeline_allowed,
            }),
        )
        .await;
    // Emergency pre-call compaction is token-driven and happens before the
    // tail/cursor snapshot is built, so this turn can safely use the refreshed
    // state. It is itself a memory read/write pipeline and is therefore gated
    // by the compiled task policy. The background pass after each reply
    // compacts at the normal threshold; the turn waits only when the history
    // nears this turn's model window (the background pass fell behind, or the
    // turn moved to a smaller-window model).
    if agent.runtime.context_window_config.enabled
        && agent.mandate_execution.is_none()
        && user_role.can_persist_owner_memory()
        && !non_owner_shared_context
        && memory_pipeline_allowed
    {
        let compaction_model = llm_router
            .as_ref()
            .map(|router| router.select(crate::router::Tier::Fast).to_string())
            .unwrap_or_else(|| model.clone());
        let token_threshold = agent
            .runtime
            .context_window_config
            .emergency_compaction_threshold_for(&model);
        let recent_tokens = agent
            .runtime
            .context_window_config
            .summary_recent_tokens_for(&model);
        match tokio::time::timeout(
            Duration::from_secs(30),
            crate::agent::conversation_compaction::compact_now(
                crate::agent::conversation_compaction::CompactionTarget {
                    side_calls: agent.side_calls.clone(),
                    fast_model: &compaction_model,
                    state: agent.state.clone(),
                    event_store: agent.event_store.clone(),
                    session_id,
                },
                token_threshold,
                recent_tokens,
            ),
        )
        .await
        {
            Ok(Ok(Some(summary))) => info!(
                session_id,
                last_turn_seq = summary.cursor,
                "Emergency token-pressure compaction refreshed current context"
            ),
            Ok(Ok(None)) => {}
            Ok(Err(error)) => warn!(session_id, %error, "Emergency compaction failed"),
            Err(_) => warn!(session_id, "Emergency compaction timed out after 30s"),
        }
    }

    session_summary = if agent.runtime.context_window_config.enabled
        && agent.mandate_execution.is_none()
        && !non_owner_shared_context
        && memory_recall_allowed
    {
        agent
            .state
            .get_conversation_summary(session_id)
            .await
            .ok()
            .flatten()
            // A legacy cursorless summary may end in the middle of an exchange.
            // Keep it out of the prompt until the canonical turn summarizer
            // rebuilds it with a safe boundary.
            .filter(|summary| summary.last_turn_seq.is_some())
    } else {
        None
    };

    let project_instruction_scope = if turn_context.allow_multi_project_scope {
        // A single instruction hierarchy must not silently govern an explicit
        // multi-repository request. Each delegated working directory can still
        // apply its own native/scoped instructions.
        None
    } else if user_role == UserRole::Owner {
        turn_context.primary_project_scope.as_deref()
    } else {
        // Collaborators may receive project guidance only from the exact root
        // already authorized by their active workspace grant.
        workspace_grant.map(|grant| grant.project_root.as_str())
    };
    let preserve_conversation_context = matches!(
        turn_context.followup_mode,
        Some(
            crate::agent::followup::FollowupMode::Followup
                | crate::agent::followup::FollowupMode::ClarificationAnswer
        )
    );

    // 2. Build system prompt ONCE before the loop: match skills + inject facts + memory
    // Returns the session-static CORE bytes (message zero) and the per-task
    // volatile TAIL separately so the assembler can place them at message 0 and
    // before the structurally preserved preceding exchange respectively.
    //
    // Pillar A Task 7 (per-task core-cache hook): the per-session core cache lives
    // INSIDE `build_system_prompt_for_message` (it is `&self` on Agent and reads
    // `self.runtime.core_prompts` directly — no signature change). On a cache HIT the
    // returned `core_prompt_bytes` are reused verbatim with no re-render; on a
    // MISS a `Core prompt invalidated component=...` line is logged there. Option
    // (b) in the plan — chosen for the smaller diff since assemble + render already
    // co-locate at that call site.
    let (
        core_prompt_bytes,
        fresh_task_context_tail,
        continuation_task_context_tail,
        active_skill_names,
        project_instruction_tracker,
    ) = agent
        .build_system_prompt_for_message(
            &emitter,
            &task_id,
            session_id,
            user_text,
            user_role,
            &channel_ctx,
            tool_defs.len(),
            resume_checkpoint.as_ref(),
            preserve_conversation_context,
            project_instruction_scope,
        )
        .await?;

    // A skill the user explicitly asked for this turn narrows the roster when
    // it is an external API reference.
    let skills_snapshot = agent.skill_cache.get();
    let active_untrusted_external_reference_skills: Vec<String> = skills_snapshot
        .iter()
        .filter(|skill| {
            active_skill_names.iter().any(|name| name == &skill.name)
                && crate::skills::is_untrusted_external_reference_skill(skill)
        })
        .map(|skill| skill.name.clone())
        .collect();
    let restrict_untrusted_external_reference_tools =
        !active_untrusted_external_reference_skills.is_empty();
    if restrict_untrusted_external_reference_tools {
        let before = tool_defs.len();
        tool_defs = filter_tool_defs_for_untrusted_external_reference(&tool_defs);
        tool_filter_stages.push(GateFilterStage::new(
            "active_untrusted_external_reference",
            before,
            tool_defs.len(),
            if before == tool_defs.len() {
                "passed"
            } else {
                "filtered"
            },
        ));
        base_tool_defs = filter_tool_defs_for_untrusted_external_reference(&base_tool_defs);
        available_capabilities
            .retain(|name, _| !is_untrusted_external_reference_blocked_tool(name));
    }

    let (tool_filter_summary, tool_filter_metadata) =
        build_tool_filter_gate_telemetry(&tool_filter_stages, initial_tool_count, tool_defs.len());
    agent
        .emit_decision_point(
            &emitter,
            &task_id,
            0,
            DecisionType::GateTelemetry,
            tool_filter_summary,
            tool_filter_metadata,
        )
        .await;

    // Pillar B (Task 7): historical conversation retention is now owned
    // entirely by the turn-anchored fetch in `message_build`
    // (`get_turns_from_anchor`). The old `load_initial_history` + pinned/recent
    // split is removed — the whole-turn anchor is the single retention
    // mechanism, so this bootstrap no longer pre-loads or pins history.

    let mut harness_eval = HarnessEvalAccumulator::new(HarnessEvalSeed {
        task_id: task_id.clone(),
        turn_id: Some(user_msg_id.clone()),
        depth: agent.depth as u32,
        parent_task_id: agent.task_id.clone(),
        goal_id: agent.goal_id.clone(),
        durable_task_id: agent.task_id.clone(),
        completion_task_kind: "conversational".to_string(),
        followup_mode: None,
        config: agent.runtime.harness_eval_config.clone(),
    });
    harness_eval.record_bootstrap(
        "default_continue",
        active_skill_names.clone(),
        Some(policy_bundle.policy.model_profile),
    );

    let data = BootstrapData {
        user_text: user_text.to_string(),
        task_id,
        memory_pipeline_policy,
        resume_execution_snapshot: resume_checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.execution_snapshot.clone()),
        emitter,
        learning_ctx,
        learning,
        active_untrusted_external_reference_skills,
        restrict_untrusted_external_reference_tools,
        available_capabilities,
        base_tool_defs,
        tool_defs,
        policy_bundle,
        llm_router,
        model,
        turn_context,
        project_instruction_tracker,
        core_prompt_bytes,
        fresh_task_context_tail,
        continuation_task_context_tail,
        session_summary,
        harness_eval,
    };

    Ok(BootstrapOutcome::Continue(Box::new(data)))
}

#[cfg(test)]
mod gate_telemetry_tests {
    use super::*;

    #[test]
    fn tool_filter_gate_telemetry_reports_removed_counts_by_stage() {
        let stages = vec![
            GateFilterStage::new("owner_role", 12, 12, "passed"),
            GateFilterStage::new("public_external_allowlist", 12, 3, "filtered"),
            GateFilterStage::new("policy_filter", 3, 2, "filtered"),
        ];

        let (summary, metadata) = build_tool_filter_gate_telemetry(&stages, 12, 2);

        assert!(summary.contains("12 -> 2"), "summary: {summary}");
        assert_eq!(metadata["condition"], "tool_filtering");
        assert_eq!(metadata["initial_tool_count"], 12);
        assert_eq!(metadata["final_tool_count"], 2);
        assert_eq!(metadata["removed_tool_count"], 10);
        assert_eq!(metadata["stages"][1]["removed_count"], 9);
        assert_eq!(metadata["stages"][2]["stage"], "policy_filter");
    }
}

#[cfg(test)]
mod mandate_bootstrap_isolation_tests {
    use super::*;
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::traits::{Mandate, MandateAuthority, MandateStore, MessageStore, TaskAttempt};

    fn attempt(id: &str, task_id: &str) -> TaskAttempt {
        TaskAttempt {
            id: id.to_string(),
            task_id: task_id.to_string(),
            goal_run_id: "mandate-bootstrap-run".to_string(),
            worker_profile_id: None,
            worker_instance_id: "mandate-bootstrap-worker".to_string(),
            lease_token: format!("lease-{id}"),
            status: "running".to_string(),
            lease_expires_at: (chrono::Utc::now() + chrono::Duration::minutes(3)).to_rfc3339(),
            last_heartbeat_at: chrono::Utc::now().to_rfc3339(),
            workspace_id: None,
            started_at: chrono::Utc::now().to_rfc3339(),
            completed_at: None,
        }
    }

    async fn run_isolated_turn(
        agent: &Agent,
        session_id: &str,
        user_text: &str,
    ) -> Box<BootstrapData> {
        let services = crate::agent::services::AgentServices::new(agent);
        match run_bootstrap(
            &services,
            &BootstrapCtx {
                session_id,
                user_text,
                attachments: &[],
                status_tx: None,
                user_role: UserRole::Owner,
                channel_ctx: &ChannelContext::internal(),
                ingress_timing: None,
                internal_continuation: false,
                parent_task_id: None,
            },
        )
        .await
        .expect("isolated mandate bootstrap")
        {
            BootstrapOutcome::Continue(data) => data,
            BootstrapOutcome::Return(_) => {
                panic!("generic bootstrap shortcut diverted a mandate worker")
            }
        }
    }

    async fn assert_durable_mandate_origin(agent: &Agent, session_id: &str) {
        let events = agent
            .event_store
            .query_events_by_types(session_id, &[EventType::UserMessage], 10)
            .await
            .expect("query mandate user message");
        let user_message = events
            .first()
            .expect("mandate bootstrap emitted a canonical user message");
        assert_eq!(
            user_message
                .data
                .get("execution_origin")
                .and_then(serde_json::Value::as_str),
            Some("mandate")
        );
    }

    fn assert_isolated(data: &BootstrapData, forbidden: &[&str]) {
        assert!(data.active_untrusted_external_reference_skills.is_empty());
        assert!(!data.restrict_untrusted_external_reference_tools);
        assert!(data.session_summary.is_none());
        assert!(data.project_instruction_tracker.is_none());
        assert!(data.llm_router.is_none());
        assert!(data.turn_context.assessment_recent_messages.is_empty());
        assert!(data.turn_context.primary_project_scope.is_none());
        assert_eq!(
            data.turn_context.goal_user_text,
            "Execute the built-in bounded mandate protocol for this exact worker fence."
        );
        let prompt = format!(
            "{}\n{}\n{}",
            data.core_prompt_bytes,
            data.fresh_task_context_tail,
            data.continuation_task_context_tail
        );
        for sentinel in forbidden {
            assert!(
                !prompt.contains(sentinel),
                "isolated bootstrap leaked {sentinel:?}"
            );
        }
    }

    #[tokio::test]
    async fn mandate_lead_and_executor_bypass_shortcuts_skills_and_history_routing() {
        let mut harness = setup_test_agent(MockProvider::new())
            .await
            .expect("setup test agent");

        let skill_dir = tempfile::tempdir().expect("skill tempdir");
        std::fs::write(
            skill_dir.path().join("private-api-guide.md"),
            "---\nname: private-api-guide\ndescription: PRIVATE SKILL DESCRIPTION\ntriggers: stop, private-api-guide\nsource: docs\n---\nPRIVATE SKILL BODY\n",
        )
        .expect("write custom skill");
        harness.agent.skills_dir = skill_dir.path().to_path_buf();
        harness.agent.skill_cache = crate::skills::SkillCache::new(skill_dir.path().to_path_buf());

        let goal = crate::traits::Goal::new_continuous(
            "Mandate bootstrap controller",
            "owner-mandate-bootstrap",
            Some(10_000),
            Some(50_000),
        );
        let authority = MandateAuthority::default();
        let mandate = Mandate::new(
            &goal.id,
            None,
            "Exercise the bounded protocol",
            &goal.session_id,
            authority.clone(),
            60,
            3_600,
            300,
        );
        harness
            .state
            .create_mandate_controller(&goal, &mandate)
            .await
            .expect("persist mandate controller");

        for session_id in ["mandate-lead-session", "mandate-executor-session"] {
            let prior = Message {
                content: Some(
                    "PRIVATE PRIOR HISTORY in /tmp/private-project with PRIVATE PROJECT SCOPE"
                        .to_string(),
                ),
                ..Message::new_runtime(uuid::Uuid::new_v4().to_string(), session_id, "assistant")
            };
            harness
                .state
                .append_message(&prior)
                .await
                .expect("persist prior history sentinel");
        }

        let root = attempt("root-attempt", "mandate-root-task");
        harness.agent.set_test_mandate_execution(
            &mandate.id,
            mandate.version,
            authority.clone(),
            &goal.id,
            &root.task_id,
            &root.id,
            &root,
        );
        let lead = run_isolated_turn(&harness.agent, "mandate-lead-session", "stop").await;
        assert_durable_mandate_origin(&harness.agent, "mandate-lead-session").await;
        assert_eq!(lead.user_text, "stop");
        assert_isolated(
            &lead,
            &[
                "PRIVATE SKILL DESCRIPTION",
                "PRIVATE SKILL BODY",
                "PRIVATE PRIOR HISTORY",
                "PRIVATE PROJECT SCOPE",
                "/tmp/private-project",
            ],
        );

        let executor = attempt("executor-attempt", "mandate-executor-task");
        harness.agent.set_test_mandate_execution(
            &mandate.id,
            mandate.version,
            authority,
            &goal.id,
            &root.task_id,
            &root.id,
            &executor,
        );
        let executor_data = run_isolated_turn(
            &harness.agent,
            "mandate-executor-session",
            "$private-api-guide inspect /tmp/private-project",
        )
        .await;
        assert_durable_mandate_origin(&harness.agent, "mandate-executor-session").await;
        assert_isolated(
            &executor_data,
            &[
                "PRIVATE SKILL DESCRIPTION",
                "PRIVATE SKILL BODY",
                "PRIVATE PRIOR HISTORY",
                "PRIVATE PROJECT SCOPE",
            ],
        );
        assert!(executor_data.core_prompt_bytes.contains("role: executor"));
    }
}
