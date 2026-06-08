use super::types::{BootstrapCtx, BootstrapData, BootstrapOutcome};
use crate::agent::recall_guardrails::{
    detect_critical_fact_query, deterministic_reply_for_critical_query,
    extract_critical_fact_summary, filter_tool_defs_for_personal_memory, is_personal_memory_tool,
    looks_like_personal_memory_recall_question, user_is_reaffirmation_challenge,
    user_requests_external_verification,
};
use crate::agent::*;

pub(in crate::agent) async fn run_bootstrap_phase(
    services: &crate::agent::services::AgentServices<'_>,
    ctx: &BootstrapCtx<'_>,
) -> anyhow::Result<BootstrapOutcome> {
    let agent = services.agent;
    let session_id = ctx.session_id;
    let user_text = ctx.user_text;
    let status_tx = ctx.status_tx.clone();
    let user_role = ctx.user_role;
    let channel_ctx = ctx.channel_ctx.clone();
    info!(session_id, "Bootstrap phase started");
    let resume_checkpoint = if is_resume_request(user_text) {
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
    if let Err(err) =
        crate::agent::dialogue_state::record_dialogue_task_start(agent, session_id, &task_id).await
    {
        warn!(
            session_id,
            task_id,
            error = %err,
            "Failed to record dialogue task start"
        );
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
        ..Message::new_runtime(user_msg_id, session_id, "user")
    };
    // Calculate heuristic score immediately
    let score = crate::memory::scoring::score_message(&user_msg);
    let mut user_msg = user_msg;
    user_msg.importance = score;

    agent
        .append_user_message_with_event(&emitter, &user_msg, user_role, &channel_ctx, false)
        .await?;

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

    // Explicit mid-task pivots ("wait stop... actually ... instead") should
    // cancel stale in-flight work but continue handling the new instruction.
    super::shortcuts::maybe_cancel_work_for_mid_task_pivot(
        agent,
        session_id,
        user_text,
        user_role,
        &channel_ctx,
        status_tx.clone(),
        &task_id,
    )
    .await;

    if let Some(reply) = super::shortcuts::maybe_handle_pending_goal_confirmation(
        agent, session_id, user_text, user_role, &task_id, &emitter,
    )
    .await?
    {
        return Ok(BootstrapOutcome::Return(Ok(reply)));
    }

    if let Some(reply) = super::shortcuts::maybe_handle_non_resolving_confirmation_shortcut(
        agent, session_id, user_text, &task_id, &emitter,
    )
    .await?
    {
        return Ok(BootstrapOutcome::Return(Ok(reply)));
    }

    if let Some(reply) = super::shortcuts::maybe_handle_trivial_ack_shortcut(
        agent, session_id, user_text, &task_id, &emitter,
    )
    .await?
    {
        return Ok(BootstrapOutcome::Return(Ok(reply)));
    }

    if let Some(reply) = super::shortcuts::maybe_handle_time_query_shortcut(
        agent, session_id, user_text, &task_id, &emitter,
    )
    .await?
    {
        return Ok(BootstrapOutcome::Return(Ok(reply)));
    }

    // Deterministic critical-fact resolver for high-trust identity/profile recall.
    // This avoids model drift when context is compressed or the fast model is selected.
    let critical_fact_query = detect_critical_fact_query(user_text);
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
    if agent.depth == 0
        && user_role == UserRole::Owner
        && channel_ctx.should_inject_personal_memory()
    {
        if let Some(query) = critical_fact_query {
            let facts = owner_dm_fact_cache.as_deref().unwrap_or(&[]);
            let mut summary = extract_critical_fact_summary(facts);
            if summary.assistant_name.is_none() {
                summary.assistant_name = agent.system_prompt.lines().find_map(|line| {
                    let trimmed = line.trim();
                    let rest = trimmed.strip_prefix("You are ")?;
                    let candidate = rest
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .trim_matches(|c: char| matches!(c, '.' | ',' | '"' | '\'' | '`'));
                    if candidate.is_empty()
                        || candidate.len() > 40
                        || matches!(candidate.to_ascii_lowercase().as_str(), "a" | "an" | "the")
                    {
                        None
                    } else {
                        Some(candidate.to_string())
                    }
                });
            }
            let reply = deterministic_reply_for_critical_query(query, &summary);
            let reply = super::shortcuts::emit_bootstrap_direct_reply(
                agent,
                &emitter,
                &task_id,
                session_id,
                Instant::now(),
                &reply,
            )
            .await?;
            return Ok(BootstrapOutcome::Return(Ok(reply)));
        }
    }

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
    // For personal-memory recall turns, keep tool search narrow unless the
    // user explicitly asks for broader verification.
    let restrict_to_personal_memory_tools =
        is_personal_memory_recall_turn && !requests_external_verification;
    // "Are you sure?" should allow only one targeted re-check before reaffirming.
    let personal_memory_tool_call_cap =
        if is_reaffirmation_challenge_turn && is_personal_memory_recall_turn {
            1
        } else {
            4
        };

    // Tool access is owner-only.
    // Orchestrator (depth 0) now keeps tools available from iteration 1.
    // Deterministic control-plane routing still handles cancel/schedule/goal fast-paths
    // before the first LLM call.
    // Sub-agents (depth > 0) get tools based on their role (set in spawn_child).
    let tools_allowed_for_user = user_role == UserRole::Owner;

    let mut available_capabilities: HashMap<String, ToolCapabilities> = HashMap::new();
    let mut base_tool_defs: Vec<Value> = Vec::new();
    let mut tool_defs: Vec<Value> = Vec::new();
    if tools_allowed_for_user {
        let (mut defs, mut caps) = agent.tool_definitions_with_capabilities(user_text).await;

        // Filter tools by channel visibility
        if channel_ctx.visibility == ChannelVisibility::PublicExternal {
            let allowed = ["web_search", "remember_fact", "system_info"];
            defs.retain(|d| {
                Agent::tool_name_from_definition(d).is_some_and(|name| allowed.contains(&name))
            });
            caps.retain(|name, _| allowed.contains(&name.as_str()));
        }

        if restrict_to_personal_memory_tools {
            defs = filter_tool_defs_for_personal_memory(&defs);
            caps.retain(|name, _| is_personal_memory_tool(name));
        }
        if restrict_untrusted_external_reference_tools {
            defs = filter_tool_defs_for_untrusted_external_reference(&defs);
            caps.retain(|name, _| !is_untrusted_external_reference_blocked_tool(name));
        }

        available_capabilities = caps;
        base_tool_defs = defs.clone();
        tool_defs = defs;
    }

    // Pillar A Task 6: canonicalize the roster to name-sorted order at the
    // bootstrap boundary so all later retain/filter/widen ops begin from a
    // stable order. The authoritative final sort happens on `effective_tool_defs`
    // in message_build_phase, but starting canonical keeps intermediate ordering
    // deterministic.
    Agent::sort_tool_definitions_by_name(&mut base_tool_defs);
    Agent::sort_tool_definitions_by_name(&mut tool_defs);

    let mut policy_bundle = build_policy_bundle(user_text, &available_capabilities, false);
    if (critical_fact_query.is_some() || is_personal_memory_recall_turn)
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
        let shadow_filtered =
            agent.restrict_connected_api_setup_tools_for_request(user_text, &shadow_filtered);
        let shadow_filtered =
            agent.ensure_connected_api_tools_exposed(user_text, &shadow_filtered, &tool_defs);
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
        if agent.policy_config.tool_filter_enforce {
            tool_defs = shadow_filtered;
        }
    }

    if restrict_to_personal_memory_tools {
        tool_defs = filter_tool_defs_for_personal_memory(&tool_defs);
        base_tool_defs = filter_tool_defs_for_personal_memory(&base_tool_defs);
        available_capabilities.retain(|name, _| is_personal_memory_tool(name));
    }
    if restrict_untrusted_external_reference_tools {
        tool_defs = filter_tool_defs_for_untrusted_external_reference(&tool_defs);
        base_tool_defs = filter_tool_defs_for_untrusted_external_reference(&base_tool_defs);
        available_capabilities
            .retain(|name, _| !is_untrusted_external_reference_blocked_tool(name));
    }

    // Keep provider + router consistent for this task, even if runtime reloads.
    let llm_runtime_snapshot = agent.llm_runtime.snapshot();
    let llm_provider = llm_runtime_snapshot.provider();
    // All depths get the router so cascade fallback works on rate limits.
    // Other depth-0-only features (identity detection, memory loading) have
    // their own separate depth checks.
    let llm_router = llm_runtime_snapshot.router();

    // Model selection: route to the appropriate model.
    // Iteration 1 runs deterministic orchestration routing before entering
    // the normal tool-enabled loop.
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
            if let Some(ref router) = llm_router {
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
    if route_failsafe_active {
        // Fail-safe mode: bypass deterministic direct-return routing and
        // force strong profile/model selection for this turn.
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
        if !tool_defs.is_empty() {
            tool_defs = agent.filter_tool_definitions_for_policy(
                &base_tool_defs,
                &available_capabilities,
                &policy_bundle.policy,
                policy_bundle.risk_score,
                false,
            );
            tool_defs = agent.restrict_connected_api_setup_tools_for_request(user_text, &tool_defs);
            tool_defs =
                agent.ensure_connected_api_tools_exposed(user_text, &tool_defs, &base_tool_defs);
        }
        warn!(
            session_id,
            model = %model,
            profile = ?policy_bundle.policy.model_profile,
            "Route drift fail-safe active: forcing strong routing policy"
        );
    }

    // 2a. Load conversation summary BEFORE building the prompt. Pillar A: the
    // session summary now participates in the per-task context tail, so it must
    // be available when `build_system_prompt_for_message` compiles the tail.
    let session_summary = if agent.context_window_config.enabled {
        agent
            .state
            .get_conversation_summary(session_id)
            .await
            .ok()
            .flatten()
    } else {
        None
    };

    // 2. Build system prompt ONCE before the loop: match skills + inject facts + memory
    // Returns the session-static CORE bytes (message zero) and the per-task
    // volatile TAIL separately so the assembler can place them at message 0 and
    // boundary − 1 respectively.
    //
    // Pillar A Task 7 (per-task core-cache hook): the per-session core cache lives
    // INSIDE `build_system_prompt_for_message` (it is `&self` on Agent and reads
    // `self.core_prompts` directly — no signature change). On a cache HIT the
    // returned `core_prompt_bytes` are reused verbatim with no re-render; on a
    // MISS a `Core prompt invalidated component=...` line is logged there. Option
    // (b) in the plan — chosen for the smaller diff since assemble + render already
    // co-locate at that call site.
    let (core_prompt_bytes, task_context_tail, active_skill_names) = agent
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
        )
        .await?;

    // Pillar B (Task 7): historical conversation retention is now owned
    // entirely by the turn-anchored fetch in `message_build_phase`
    // (`get_turns_from_anchor`). The old `load_initial_history` + pinned/recent
    // split is removed — the whole-turn anchor is the single retention
    // mechanism, so this bootstrap no longer pre-loads or pins history.

    let data = BootstrapData {
        task_id,
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
        core_prompt_bytes,
        task_context_tail,
        session_summary,
    };

    Ok(BootstrapOutcome::Continue(Box::new(data)))
}
