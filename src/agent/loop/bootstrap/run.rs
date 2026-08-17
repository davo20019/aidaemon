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
    plan: Option<&super::task_planning::TaskPlan>,
    available_tool_names: &[String],
    relationship_fallback: Option<&super::task_planning::PlannedTaskShape>,
    task_assessment_attempted: bool,
    assessment_decision_type: crate::events::DecisionType,
    model: &str,
    planner_trust_tier: &str,
) -> (TurnContext, bool) {
    // Only the relationship producer may select an antecedent. The contract
    // producer's task shape remains useful for execution mode/confidence, but
    // its relationship fields have no dialogue-state authority.
    let relationship_shape = relationship_fallback
        .filter(|shape| super::task_planning::planned_task_relationship_is_complete(shape));
    let execution_shape = plan
        .and_then(|plan| plan.task_shape.as_ref())
        .filter(|shape| {
            matches!(
                shape
                    .confidence
                    .as_deref()
                    .map(|value| value.trim().to_ascii_lowercase()),
                Some(value) if matches!(value.as_str(), "medium" | "high")
            )
        });

    let mut committed_antecedent_user_message_id = None;
    if !internal_continuation {
        let dialogue_state = agent
            .state
            .get_dialogue_state(session_id)
            .await
            .ok()
            .flatten();
        let typed_clarification = initial_turn_context.followup_mode
            == Some(crate::agent::followup::FollowupMode::ClarificationAnswer);
        let (relationship, semantic_scope, reason_code) = if structural_resume {
            ("continuation", "none", "runtime_resume_edge")
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
            let resolved_antecedent = dialogue_state.as_ref().and_then(|state| {
                if let Some(message_id) = requested_antecedent {
                    crate::agent::dialogue_state::has_exact_request_antecedent(state, message_id)
                        .then_some(message_id)
                } else {
                    crate::agent::dialogue_state::unambiguous_request_antecedent(state)
                        .map(|request| request.user_message_id.as_str())
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
                committed_antecedent_user_message_id = resolved_antecedent.map(str::to_string);
            }
            let reason = if requested == "continuation" && !exact_antecedent {
                "continuation_missing_exact_antecedent"
            } else {
                "assessment_committed"
            };
            (
                relationship,
                shape.semantic_scope.as_deref().unwrap_or("none"),
                reason,
            )
        } else if typed_clarification {
            ("clarification_answer", "none", "typed_clarification_edge")
        } else {
            // A failed/unavailable language assessment cannot authorize an
            // adoption edge. Ordinary ingress therefore fails closed to a
            // fresh request; only runtime continuations and typed pending
            // questions bypass this rule.
            ("new_request", "none", "assessment_unavailable_fresh_task")
        };

        if let Err(error) = crate::agent::dialogue_state::record_dialogue_semantic_user_turn(
            agent,
            session_id,
            user_text,
            relationship,
            semantic_scope,
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
                    "semantic_scope": semantic_scope,
                    "reason_code": reason_code,
                    "antecedent_user_message_id": committed_antecedent_user_message_id,
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
    let before_contract = turn_context.completion_contract.clone();
    let mut semantic_contract_applied = false;
    let mut contract_lane_decisions = Vec::new();
    let mut compiled_evidence_requirements = Vec::new();
    let mut compiled_required_invocations = Vec::new();
    let mut compiled_authority = None;
    let mut compiled_evidence_policy = None;
    let mut compiled_filesystem_access = None;
    let mut compiled_project_scope = None;

    if let Some(shape) = execution_shape {
        turn_context.continue_inline_after_background_start = shape
            .continue_inline_after_background_start
            .unwrap_or(false);
    }

    if let Some(signals) = plan.and_then(|plan| plan.contract.as_ref()) {
        // The semantic producer assigns read/write/cwd roles, but it cannot
        // invent filesystem authority. Ground every proposed path against an
        // exact path-shaped resource identity authored in the current request.
        // This parser recognizes filesystem grammar only; it does not infer
        // intent from words or phrases.
        let structural_filesystem_resources =
            crate::agent::project_scope::extract_exact_filesystem_resources_from_text(
                user_text,
                &agent.path_aliases.projects,
            );
        let structural_project_scopes = structural_filesystem_resources.clone();
        let compiled = super::contract_compiler::compile_task_contract(
            super::contract_compiler::ContractCompilerInput {
                signals,
                task_shape: plan.and_then(|plan| plan.task_shape.as_ref()),
                available_tool_names,
                structural_filesystem_resources: &structural_filesystem_resources,
                structural_project_scopes: &structural_project_scopes,
                project_alias_roots: &agent.path_aliases.projects,
            },
        );
        contract_lane_decisions = compiled.decisions.clone();
        compiled_evidence_requirements = compiled.evidence_requirements.clone();
        compiled_required_invocations = compiled.required_invocations.clone();
        compiled_authority = Some(compiled.authority.clone());
        compiled_evidence_policy = Some(compiled.evidence_policy.clone());
        compiled_filesystem_access = compiled.filesystem_access.clone();
        compiled_project_scope = compiled.project_scope.clone();

        if let Some(core) = compiled.core.as_ref() {
            crate::agent::install_semantic_completion_contract(
                &mut turn_context.completion_contract,
                crate::agent::SemanticCompletionRequirements {
                    expects_mutation: core.expects_mutation,
                    requires_observation: core.requires_observation,
                    task_kind: core.task_kind,
                    required_mutation_effects: core.required_mutation_effects,
                },
            );
            if turn_context.inherited_completion_contract {
                turn_context.completion_contract = if turn_context.inherited_outstanding_obligations
                {
                    crate::agent::inherit_unfinished_request_contract(
                        turn_context.completion_contract,
                        &before_contract,
                    )
                } else {
                    crate::agent::inherit_request_constraints(
                        turn_context.completion_contract,
                        &before_contract,
                    )
                };
            }
            semantic_contract_applied = true;
        }
    }

    let effective_required_invocations = compiled_required_invocations.clone();
    let typed_obligation_pending =
        !compiled_evidence_requirements.is_empty() || !effective_required_invocations.is_empty();
    if agent.mandate_execution.is_none()
        && !semantic_contract_applied
        && !turn_context.inherited_completion_contract
    {
        crate::agent::retain_structural_completion_contract(
            &mut turn_context.completion_contract,
            typed_obligation_pending,
        );
    }
    if let Some(authority) = compiled_authority.as_ref() {
        crate::agent::apply_semantic_authority(
            &mut turn_context.completion_contract,
            crate::agent::SemanticAuthorityRequirements {
                mutation_scope: authority.mutation_scope,
                forbidden_actions: &authority.forbidden_actions,
                forbids_tool_use: authority.forbids_tool_use,
                allowed_tool_names: &authority.allowed_tool_names,
                forbidden_tool_scopes: &authority.forbidden_tool_scopes,
            },
        );
    }
    if let Some(policy) = compiled_evidence_policy {
        crate::agent::apply_semantic_evidence_policy(
            &mut turn_context.completion_contract,
            crate::agent::SemanticEvidencePolicy {
                minimum_sources: policy.minimum_sources,
                requires_primary_sources: policy.requires_primary_sources,
                requires_exact_history: policy.requires_exact_history,
            },
        );
    }
    if let Some(project_scope) = compiled_project_scope {
        turn_context.primary_project_scope = Some(project_scope);
    }
    turn_context.filesystem_access = compiled_filesystem_access.filter(|access| {
        access.execution_cwd.is_some()
            || !access.read_targets.is_empty()
            || !access.write_targets.is_empty()
    });
    crate::agent::append_evidence_obligations(
        &mut turn_context.completion_contract,
        &compiled_evidence_requirements,
    );
    crate::agent::append_required_invocation_obligations(
        &mut turn_context.completion_contract,
        &effective_required_invocations,
    );
    turn_context.completion_contract.adopt_for_task(task_id);

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

    if let Some(plan) = plan {
        let contract_changed = turn_context.completion_contract != before_contract;
        agent
            .emit_decision_point(
                emitter,
                task_id,
                0,
                assessment_decision_type,
                "Task assessment succeeded".to_string(),
                {
                    let mut metadata = crate::agent::hand_holding_telemetry::planner_result_metadata(
                        "succeeded",
                        model,
                        planner_trust_tier,
                        crate::agent::hand_holding_telemetry::PlannerResultStats {
                            step_count: plan.steps.len(),
                            success_criteria_count: plan.success_criteria.len(),
                            contract_present: plan.contract.is_some(),
                            contract_changed,
                        },
                        None,
                    );
                    metadata["assessment_mode"] = json!(plan.mode.as_str());
                    metadata["task_shape"] = json!(plan.task_shape.as_ref());
                    metadata["semantic_contract_applied"] = json!(semantic_contract_applied);
                    metadata["contract_lane_decisions"] = json!(contract_lane_decisions);
                    metadata["completion_contract"] = json!({
                        "task_kind": format!("{:?}", turn_context.completion_contract.task_kind).to_ascii_lowercase(),
                        "expects_mutation": turn_context.completion_contract.expects_mutation,
                        "requires_observation": turn_context.completion_contract.requires_observation,
                        "evidence_requirements": turn_context.completion_contract.evidence_requirements,
                        "required_invocations": effective_required_invocations,
                        "forbidden_tool_scopes": turn_context.completion_contract.forbidden_tool_scopes,
                        "legacy_response_fields_ignored": turn_context.completion_contract.required_response_fields,
                    });
                    metadata
                },
            )
            .await;
    } else if task_assessment_attempted {
        agent
            .emit_decision_point(
                emitter,
                task_id,
                0,
                assessment_decision_type,
                "Task assessment returned no result".to_string(),
                crate::agent::hand_holding_telemetry::planner_result_metadata(
                    "no_plan",
                    model,
                    planner_trust_tier,
                    crate::agent::hand_holding_telemetry::PlannerResultStats::empty(),
                    Some("assessment_returned_none"),
                ),
            )
            .await;
    }

    (turn_context, semantic_contract_applied)
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
    let llm_provider = llm_runtime_snapshot.provider();
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
            None,
            None,
            false,
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
        config: agent.harness_eval_config.clone(),
    });
    harness_eval.record_bootstrap(
        "isolated_mandate_continue",
        Vec::new(),
        Some(ModelProfile::Strong),
        false,
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
        replay_notes: Vec::new(),
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
        task_plan: None,
        memory_pipeline_policy:
            super::task_planning::MemoryPipelinePolicy::SuppressedByCurrentContract,
        resume_execution_snapshot: None,
        emitter,
        learning_ctx,
        active_skill_names,
        active_untrusted_external_reference_skills: Vec::new(),
        restrict_untrusted_external_reference_tools: false,
        tools_allowed_for_user: true,
        available_capabilities,
        base_tool_defs,
        tool_defs,
        policy_bundle,
        llm_provider,
        llm_router: None,
        model,
        route_failsafe_active: false,
        turn_context,
        project_instruction_tracker,
        core_prompt_bytes,
        fresh_task_context_tail,
        continuation_task_context_tail,
        session_summary: None,
        harness_eval,
    })))
}

pub(in crate::agent) async fn run_bootstrap_phase(
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
            &agent.stt_config,
            &agent.audio_config,
            &model_for_stt,
        )
        .await
    };
    let user_text = user_text_enriched.as_str();
    // Resume checkpoints can contain prior tool arguments/results and broader
    // owner authority. A collaborator starts a fresh scoped turn instead of
    // inheriting an interrupted owner task from the shared session.
    let resume_checkpoint =
        if !is_mandate_execution && user_role == UserRole::Owner && is_resume_request(user_text) {
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
    // boundary detection in `message_build_phase` group the turn without
    // inferring from message content — historically a source of bugs when
    // the same text is sent twice or arrives out of order.
    let user_msg = Message {
        content: Some(user_text.to_string()),
        importance: 0.5, // Will be updated by score_message below
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
    let score = crate::memory::scoring::score_message(&user_msg);
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
        // Finalized below after semantic assessment, before this context can
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
        replay_notes: Vec::new(),
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
    let llm_provider = llm_runtime_snapshot.provider();
    let llm_router = llm_runtime_snapshot.router();
    let autonomous_bootstrap = agent.trust_tier_for_model(&llm_runtime_snapshot.primary_model())
        == crate::agent::trust_tier::ModelTrustTier::Autonomous;

    // Tools are owner-only by default. A Guest can receive a tiny file-tool
    // subset only through an explicit, active, channel-bound workspace grant.
    // Orchestrator (depth 0) keeps its authorized tools available from iteration 1.
    // Sub-agents (depth > 0) get tools based on their role (set in spawn_child).
    let workspace_grant = channel_ctx.active_workspace_grant(user_role);
    let tools_allowed_for_user = user_role == UserRole::Owner || workspace_grant.is_some();
    let correction_mode = agent.correction_context_for_current_goal().await.is_some();

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

        if correction_mode {
            let before = defs.len();
            defs.retain(|definition| {
                Agent::tool_name_from_definition(definition).is_some_and(
                    crate::agent::correction_sandbox::tool_may_be_offered_during_correction,
                )
            });
            caps.retain(|name, _| {
                crate::agent::correction_sandbox::tool_may_be_offered_during_correction(name)
            });
            tool_filter_stages.push(GateFilterStage::new(
                "correction_sandbox_roster",
                before,
                defs.len(),
                if before == defs.len() {
                    "passed"
                } else {
                    "filtered"
                },
            ));
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
    // in message_build_phase, but starting canonical keeps intermediate ordering
    // deterministic.
    Agent::sort_tool_definitions_by_name(&mut base_tool_defs);
    Agent::sort_tool_definitions_by_name(&mut tool_defs);

    let mut policy_bundle = build_policy_bundle(user_text, &available_capabilities, false);
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
        if agent.policy_config.policy_shadow_mode {
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
        if agent.policy_config.tool_filter_enforce && !autonomous_bootstrap {
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
    let selected_model = {
        let is_override =
            match tokio::time::timeout(Duration::from_secs(2), agent.model_override.read()).await {
                Ok(guard) => *guard,
                Err(_) => {
                    warn!(
                        session_id,
                        "Timed out acquiring model_override lock while selecting bootstrap model"
                    );
                    false
                }
            };
        if !is_override {
            if autonomous_bootstrap {
                llm_runtime_snapshot.primary_model()
            } else if let Some(ref router) = llm_router {
                let new_model = router
                    .select_for_profile(policy_bundle.policy.model_profile)
                    .to_string();
                let routed_model = new_model;
                if agent.policy_config.policy_shadow_mode {
                    info!(
                        session_id,
                        task_id = %task_id,
                        new_profile = ?policy_bundle.policy.model_profile,
                        new_model = %routed_model,
                        risk_score = policy_bundle.risk_score,
                        uncertainty_score = policy_bundle.uncertainty_score,
                        confidence = policy_bundle.confidence,
                        "Policy shadow routing snapshot (profile-to-model mapping)"
                    );
                }
                info!(
                    routed_model = %routed_model,
                    policy_profile = ?policy_bundle.policy.model_profile,
                    "Selected model for task"
                );
                routed_model
            } else {
                // No router: for top-level auto mode, pick the model from the same
                // runtime snapshot as provider/router to avoid transient reload races.
                // Sub-agents keep their local model selection behavior.
                let m = if agent.depth == 0 {
                    llm_runtime_snapshot.primary_model()
                } else {
                    match tokio::time::timeout(Duration::from_secs(2), agent.model.read()).await {
                        Ok(guard) => guard.clone(),
                        Err(_) => {
                            warn!(
                                session_id,
                                "Timed out acquiring model lock while selecting bootstrap model"
                            );
                            llm_runtime_snapshot.primary_model()
                        }
                    }
                };
                m
            }
        } else {
            // Model override keeps normal loop behavior.
            let m = match tokio::time::timeout(Duration::from_secs(2), agent.model.read()).await {
                Ok(guard) => guard.clone(),
                Err(_) => {
                    warn!(
                        session_id,
                        "Timed out acquiring model lock while honoring override"
                    );
                    llm_runtime_snapshot.primary_model()
                }
            };
            m
        }
    };
    let mut model = selected_model.clone();
    let route_failsafe_active = route_failsafe_active_for_session(session_id);
    if route_failsafe_active && !autonomous_bootstrap {
        // Guided models may still use the drift fail-safe to recover capacity.
        if !matches!(policy_bundle.policy.model_profile, ModelProfile::Strong) {
            policy_bundle.policy = ExecutionPolicy::for_profile(ModelProfile::Strong);
            policy_bundle
                .policy
                .escalation_reasons
                .push("route_drift_failsafe".to_string());
        }
        if let Some(ref router) = llm_router {
            model = router.select_for_profile(ModelProfile::Strong).to_string();
        }
        if !tool_defs.is_empty() && !autonomous_bootstrap {
            tool_defs = agent.filter_tool_definitions_for_policy(
                &base_tool_defs,
                &available_capabilities,
                &policy_bundle.policy,
                policy_bundle.risk_score,
                false,
            );
        }
        warn!(
            session_id,
            model = %model,
            profile = ?policy_bundle.policy.model_profile,
            "Route drift fail-safe active: forcing strong routing policy"
        );
    } else if route_failsafe_active {
        info!(
            session_id,
            model = %model,
            "Route drift fail-safe observed without overriding autonomous model selection"
        );
    }

    // Compile task relationship and capability policy before any optional
    // memory access. The same assessment artifact is handed to the main loop;
    // policy and finalization therefore cannot diverge through two model calls.
    let non_owner_shared_context = user_role != UserRole::Owner
        && matches!(
            channel_ctx.visibility,
            ChannelVisibility::PrivateGroup
                | ChannelVisibility::Public
                | ChannelVisibility::PublicExternal
        );

    let mut session_summary = if agent.context_window_config.enabled
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

    // Resolve turn/project context before task assessment and prompt
    // construction. The main loop reuses this exact snapshot.
    let initial_turn_context = agent
        .build_turn_context_from_recent_history_with_origin(
            session_id,
            user_text,
            ctx.internal_continuation,
        )
        .await;
    let model_trust_tier = agent.trust_tier_for_model(&model);
    let planner_trust_tier = model_trust_tier.as_str();
    let assessment_mode = match model_trust_tier {
        crate::agent::trust_tier::ModelTrustTier::Guided => {
            super::task_planning::TaskAssessmentMode::GuidedPlan
        }
        crate::agent::trust_tier::ModelTrustTier::Autonomous => {
            super::task_planning::TaskAssessmentMode::AutonomousRouting
        }
    };
    let assessment_decision_type = match assessment_mode {
        super::task_planning::TaskAssessmentMode::GuidedPlan => {
            crate::events::DecisionType::HandHoldingTelemetry
        }
        super::task_planning::TaskAssessmentMode::AutonomousRouting => {
            crate::events::DecisionType::IntentGate
        }
    };
    let planner_skip_reason = if ctx.internal_continuation {
        Some("internal_continuation_uses_parent_contract")
    } else if resume_checkpoint.is_some() {
        Some("runtime_resume_uses_parent_contract")
    } else if agent.mandate_execution.is_some() {
        Some("mandate_cycle_uses_only_budgeted_main_loop_calls")
    } else {
        super::task_planning::planning_skip_reason(user_text, false)
    };
    let planner_models = llm_router
        .as_ref()
        .map_or_else(|| vec![model.clone()], |router| router.all_models_ordered());
    let planner_model = planner_models
        .first()
        .map_or(model.as_str(), String::as_str);
    let planner_context = super::task_planning::task_assessment_conversation_context(
        initial_turn_context.followup_mode,
        session_summary
            .as_ref()
            .map(|summary| summary.summary.as_str()),
        &initial_turn_context.assessment_recent_messages,
    );
    let planner_telemetry = Some(super::task_planning::PlannerTelemetryCtx {
        emitter: &emitter,
        state: agent.state.as_ref(),
        session_id,
        task_id: &task_id,
    });
    let (task_plan, task_assessment_attempted, relationship_fallback) =
        if let Some(reason) = planner_skip_reason {
            agent
                .emit_decision_point(
                    &emitter,
                    &task_id,
                    0,
                    assessment_decision_type,
                    "Task assessment skipped".to_string(),
                    crate::agent::hand_holding_telemetry::planner_skip_metadata(
                        reason,
                        &model,
                        planner_trust_tier,
                    ),
                )
                .await;
            (None, false, None)
        } else {
            agent
                .emit_decision_point(
                    &emitter,
                    &task_id,
                    0,
                    assessment_decision_type,
                    "Task assessment attempted".to_string(),
                    crate::agent::hand_holding_telemetry::planner_result_metadata(
                        "attempted",
                        planner_model,
                        planner_trust_tier,
                        crate::agent::hand_holding_telemetry::PlannerResultStats::empty(),
                        None,
                    ),
                )
                .await;
            // Contract and dialogue relationship are independent read-only
            // producers. Run them concurrently under their own deadlines so a
            // slow contract model cannot delay the lightweight relationship lane.
            // Contract authority and obligations belong to the current request
            // only. Prior conversation is visible exclusively to the relationship
            // producer; carryover must cross its typed antecedent/adoption edge.
            let contract_future = super::task_planning::generate_task_plan(
                llm_provider.clone(),
                &planner_models,
                user_text,
                assessment_mode,
                planner_telemetry,
            );
            let relationship_future = async {
                match planner_context.as_deref() {
                    Some(context) => {
                        super::task_planning::generate_task_relationship(
                            llm_provider.clone(),
                            planner_model,
                            user_text,
                            context,
                            planner_telemetry,
                        )
                        .await
                    }
                    None => None,
                }
            };
            let (task_plan, relationship) = tokio::join!(contract_future, relationship_future);
            (task_plan, true, relationship)
        };
    let available_tool_names = tool_defs
        .iter()
        .filter_map(|definition| {
            definition
                .get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| definition.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect::<Vec<_>>();
    let (turn_context, semantic_contract_applied) = finalize_turn_assessment(
        agent,
        &emitter,
        session_id,
        &task_id,
        user_text,
        ctx.internal_continuation,
        resume_checkpoint.is_some(),
        &initial_turn_context,
        task_plan.as_ref(),
        &available_tool_names,
        relationship_fallback.as_ref(),
        task_assessment_attempted,
        assessment_decision_type,
        &model,
        planner_trust_tier,
    )
    .await;
    let memory_pipeline_policy = super::task_planning::compile_memory_pipeline_policy(
        &turn_context.completion_contract,
        semantic_contract_applied,
    );
    let memory_pipeline_allowed = memory_pipeline_policy.allows_memory();
    learning_ctx.memory_persistence_allowed = memory_pipeline_allowed;
    emitter
        .emit(
            EventType::MemoryPolicyCompiled,
            crate::events::MemoryPolicyCompiledData {
                task_id: task_id.clone(),
                turn_id: Some(user_msg_id.clone()),
                access: if memory_pipeline_allowed {
                    crate::events::MemoryPipelineAccess::Allowed
                } else {
                    crate::events::MemoryPipelineAccess::Suppressed
                },
                reason_code: memory_pipeline_policy.reason_code().to_string(),
                retrieval_suppressed: !memory_pipeline_allowed,
                persistence_suppressed: !memory_pipeline_allowed,
            },
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
                "policy": if memory_pipeline_allowed { "allowed" } else { "suppressed" },
                "reason_code": memory_pipeline_policy.reason_code(),
                "retrieval_suppressed": !memory_pipeline_allowed,
                "memory_message_projection_suppressed": !memory_pipeline_allowed,
                "post_task_learning_suppressed": !memory_pipeline_allowed,
            }),
        )
        .await;
    // Identity/profile retrieval is optional memory access too; compile the
    // policy first, then fetch only the bounded categories used by the prompt.
    let owner_dm_fact_cache = if memory_pipeline_allowed
        && agent.depth == 0
        && user_role == UserRole::Owner
        && channel_ctx.should_inject_personal_memory()
    {
        let mut identity_facts = Vec::new();
        for category in &[
            "identity",
            "personal",
            "profile",
            "user",
            "assistant",
            "bot",
            "relationship",
            "preference",
            "family",
        ] {
            if let Ok(mut facts) = agent.state.get_facts(Some(category)).await {
                identity_facts.append(&mut facts);
            }
        }
        Some(identity_facts)
    } else {
        None
    };

    // Emergency pre-call compaction is token-driven and happens before the
    // tail/cursor snapshot is built, so this turn can safely use the refreshed
    // state. It is itself a memory read/write pipeline and is therefore gated
    // by the compiled task policy.
    if agent.context_window_config.enabled
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
            .context_window_config
            .summarize_token_threshold_for(&compaction_model);
        let recent_tokens = agent
            .context_window_config
            .summary_recent_tokens_for(&compaction_model);
        match tokio::time::timeout(
            Duration::from_secs(30),
            crate::memory::context_window::refresh_incremental_summarization(
                llm_provider.clone(),
                &compaction_model,
                agent.state.clone(),
                agent.event_store.clone(),
                session_id,
                token_threshold,
                recent_tokens,
            ),
        )
        .await
        {
            Ok(Ok(Some(summary))) => info!(
                session_id,
                last_turn_seq = summary.last_turn_seq,
                "Emergency token-pressure compaction refreshed current context"
            ),
            Ok(Ok(None)) => {}
            Ok(Err(error)) => warn!(session_id, %error, "Emergency compaction failed"),
            Err(_) => warn!(session_id, "Emergency compaction timed out after 30s"),
        }
    }

    session_summary = if agent.context_window_config.enabled
        && agent.mandate_execution.is_none()
        && !non_owner_shared_context
        && memory_pipeline_allowed
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
    // `self.core_prompts` directly — no signature change). On a cache HIT the
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
            owner_dm_fact_cache.as_deref(),
            session_summary.as_ref(),
            memory_pipeline_allowed,
            preserve_conversation_context,
            project_instruction_scope,
        )
        .await?;

    // An external-reference skill may narrow the roster only after it is the
    // final activated skill. Raw trigger matches are merely semantic
    // candidates and cannot independently remove tools.
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
    // entirely by the turn-anchored fetch in `message_build_phase`
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
        config: agent.harness_eval_config.clone(),
    });
    harness_eval.record_bootstrap(
        "default_continue",
        active_skill_names.clone(),
        Some(policy_bundle.policy.model_profile),
        route_failsafe_active,
    );

    let data = BootstrapData {
        user_text: user_text.to_string(),
        task_id,
        task_plan,
        memory_pipeline_policy,
        resume_execution_snapshot: resume_checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.execution_snapshot.clone()),
        emitter,
        learning_ctx,
        active_skill_names,
        active_untrusted_external_reference_skills,
        restrict_untrusted_external_reference_tools,
        tools_allowed_for_user,
        available_capabilities,
        base_tool_defs,
        tool_defs,
        policy_bundle,
        llm_provider,
        llm_router,
        model,
        route_failsafe_active,
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
        match run_bootstrap_phase(
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

    fn assert_isolated(data: &BootstrapData, forbidden: &[&str]) {
        assert!(data.active_skill_names.is_empty());
        assert!(data.active_untrusted_external_reference_skills.is_empty());
        assert!(!data.restrict_untrusted_external_reference_tools);
        assert!(data.session_summary.is_none());
        assert!(data.project_instruction_tracker.is_none());
        assert!(data.llm_router.is_none());
        assert!(!data.route_failsafe_active);
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
