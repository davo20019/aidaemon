use super::types::{BootstrapCtx, BootstrapData, BootstrapOutcome};
use crate::agent::recall_guardrails::{
    looks_like_personal_memory_recall_question, user_is_reaffirmation_challenge,
    user_requests_external_verification,
};
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

    let (core_prompt_bytes, task_context_tail, active_skill_names, project_instruction_tracker) =
        agent
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
    let turn_context = TurnContext {
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
        resume_execution_snapshot: None,
        emitter,
        learning_ctx,
        is_personal_memory_recall_turn: false,
        is_reaffirmation_challenge_turn: false,
        requests_external_verification: false,
        restrict_to_personal_memory_tools: false,
        active_skill_names,
        active_untrusted_external_reference_skills: Vec::new(),
        restrict_untrusted_external_reference_tools: false,
        personal_memory_tool_call_cap: 0,
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
        task_context_tail,
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
    let _ = emitter
        .emit(
            EventType::TaskStart,
            TaskStartData {
                task_id: task_id.clone(),
                description: task_description.chars().take(200).collect(),
                parent_task_id: resumed_from_task_id,
                user_message: Some(user_text.to_string()),
                turn_id: Some(user_msg_id.clone()),
            },
        )
        .await;
    if !is_mandate_execution {
        if let Err(err) =
            crate::agent::dialogue_state::record_dialogue_task_start(agent, session_id, &task_id)
                .await
        {
            warn!(
                session_id,
                task_id,
                error = %err,
                "Failed to record dialogue task start"
            );
        }
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

    // A current schedule proposal owns an exact confirm/cancel response. Check
    // it before the generic task-cancel shortcut so `cancel` discards the
    // proposal instead of reporting "No running task to cancel".
    if let Some(reply) = super::shortcuts::maybe_handle_pending_goal_confirmation(
        agent, session_id, user_text, user_role, &task_id, &emitter,
    )
    .await?
    {
        return Ok(BootstrapOutcome::Return(Ok(reply)));
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

    // Only fetch identity/profile categories — NOT get_facts(None) which returned
    // every fact in the DB, causing unrelated facts (Ecuador travel, WiFi router
    // tips, etc.) to bleed into prompts for unrelated queries.
    let owner_dm_fact_cache = if agent.depth == 0
        && user_role == UserRole::Owner
        && channel_ctx.should_inject_personal_memory()
    {
        let mut identity_facts = Vec::new();
        for cat in &[
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
            if let Ok(mut facts) = agent.state.get_facts(Some(cat)).await {
                identity_facts.append(&mut facts);
            }
        }
        Some(identity_facts)
    } else {
        None
    };
    // Initialize learning context for post-task learning
    let mut learning_ctx = LearningContext {
        user_text: user_text.to_string(),
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

    let mut is_personal_memory_recall_turn = looks_like_personal_memory_recall_question(user_text);
    let is_reaffirmation_challenge_turn = user_is_reaffirmation_challenge(user_text);
    if is_reaffirmation_challenge_turn && !is_personal_memory_recall_turn {
        if let Ok(history) = agent.state.get_history(session_id, 8).await {
            // Challenge turns like "Are you sure?" inherit context from the
            // immediately previous user request.
            let mut skipped_current = false;
            for msg in history.iter().rev() {
                if msg.role != "user" {
                    continue;
                }
                let Some(content) = msg.content.as_deref() else {
                    continue;
                };
                let trimmed = content.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if !skipped_current && trimmed.eq_ignore_ascii_case(user_text.trim()) {
                    skipped_current = true;
                    continue;
                }
                if looks_like_personal_memory_recall_question(trimmed) {
                    is_personal_memory_recall_turn = true;
                }
                break;
            }
        }
    }
    let requests_external_verification = user_requests_external_verification(user_text);
    // Bootstrap tool exposure happens before per-task model selection. Use the
    // runtime primary model's trust tier to decide whether policy filtering is
    // merely observed or may narrow the roster.
    let llm_runtime_snapshot = agent.llm_runtime.snapshot();
    let llm_provider = llm_runtime_snapshot.provider();
    let llm_router = llm_runtime_snapshot.router();
    let autonomous_bootstrap = agent.trust_tier_for_model(&llm_runtime_snapshot.primary_model())
        == crate::agent::trust_tier::ModelTrustTier::Autonomous;
    let skills_snapshot = agent.skill_cache.get();
    let active_untrusted_external_reference_skills =
        matched_untrusted_external_reference_skill_names(
            &skills_snapshot,
            user_text,
            user_role,
            channel_ctx.visibility,
        );
    let restrict_untrusted_external_reference_tools = !active_untrusted_external_reference_skills
        .is_empty()
        && !user_explicitly_requests_local_file_inspection(user_text);
    let restrict_to_personal_memory_tools = false;
    let personal_memory_tool_call_cap = 4;

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

        if restrict_untrusted_external_reference_tools {
            let before = defs.len();
            defs = filter_tool_defs_for_untrusted_external_reference(&defs);
            caps.retain(|name, _| !is_untrusted_external_reference_blocked_tool(name));
            tool_filter_stages.push(GateFilterStage::new(
                "untrusted_external_reference",
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
    if is_personal_memory_recall_turn
        && matches!(policy_bundle.policy.model_profile, ModelProfile::Cheap)
        && policy_bundle
            .policy
            .escalate("critical_recall_turn_requires_primary")
    {
        info!(
            session_id,
            new_profile = ?policy_bundle.policy.model_profile,
            "Escalated model profile for critical personal recall turn"
        );
    }

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

    if restrict_untrusted_external_reference_tools {
        let before = tool_defs.len();
        tool_defs = filter_tool_defs_for_untrusted_external_reference(&tool_defs);
        tool_filter_stages.push(GateFilterStage::new(
            "untrusted_external_reference_final",
            before,
            tool_defs.len(),
            if before == tool_defs.len() {
                "passed"
            } else {
                "filtered"
            },
        ));
        let before_base = base_tool_defs.len();
        base_tool_defs = filter_tool_defs_for_untrusted_external_reference(&base_tool_defs);
        tool_filter_stages.push(GateFilterStage::new(
            "untrusted_external_reference_base_final",
            before_base,
            base_tool_defs.len(),
            if before_base == base_tool_defs.len() {
                "passed"
            } else {
                "filtered"
            },
        ));
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

    // 2a. Load conversation summary BEFORE building the prompt. Pillar A: the
    // session summary now participates in the per-task context tail, so it must
    // be available when `build_system_prompt_for_message` compiles the tail.
    let non_owner_shared_context = user_role != UserRole::Owner
        && matches!(
            channel_ctx.visibility,
            ChannelVisibility::PrivateGroup
                | ChannelVisibility::Public
                | ChannelVisibility::PublicExternal
        );

    // Emergency pre-call compaction is token-driven and happens before the
    // tail/cursor snapshot is built, so this turn can safely use the refreshed
    // state. Normal maintenance remains the coalesced post-turn worker.
    if agent.context_window_config.enabled
        && agent.mandate_execution.is_none()
        && user_role.can_persist_owner_memory()
        && !non_owner_shared_context
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

    let session_summary = if agent.context_window_config.enabled
        && agent.mandate_execution.is_none()
        && !non_owner_shared_context
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

    // Resolve turn/project context before prompt construction. The main loop
    // reuses this exact snapshot so instruction scope cannot drift between the
    // prompt AIDaemon saw and the filesystem scope its tools enforce.
    let turn_context = agent
        .build_turn_context_from_recent_history(session_id, user_text)
        .await;
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
    let (core_prompt_bytes, task_context_tail, active_skill_names, project_instruction_tracker) =
        agent
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
                project_instruction_scope,
            )
            .await?;

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
        resume_execution_snapshot: resume_checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.execution_snapshot.clone()),
        emitter,
        learning_ctx,
        is_personal_memory_recall_turn,
        is_reaffirmation_challenge_turn,
        requests_external_verification,
        restrict_to_personal_memory_tools,
        active_skill_names,
        active_untrusted_external_reference_skills,
        restrict_untrusted_external_reference_tools,
        personal_memory_tool_call_cap,
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
        task_context_tail,
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
        assert!(data.turn_context.recent_messages.is_empty());
        assert!(data.turn_context.primary_project_scope.is_none());
        assert_eq!(
            data.turn_context.goal_user_text,
            "Execute the built-in bounded mandate protocol for this exact worker fence."
        );
        let prompt = format!("{}\n{}", data.core_prompt_bytes, data.task_context_tail);
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
