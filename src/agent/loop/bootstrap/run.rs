use super::types::{BootstrapCtx, BootstrapData, BootstrapOutcome};
use crate::agent::recall_guardrails::{
    detect_critical_fact_query, deterministic_reply_for_critical_query,
    extract_critical_fact_summary, filter_tool_defs_for_personal_memory, is_personal_memory_tool,
    looks_like_personal_memory_recall_question, user_is_reaffirmation_challenge,
    user_requests_external_verification,
};
use crate::agent::*;

impl Agent {
    pub(in crate::agent) async fn run_bootstrap_phase(
        &self,
        ctx: &BootstrapCtx<'_>,
    ) -> anyhow::Result<BootstrapOutcome> {
        let session_id = ctx.session_id;
        let user_text = ctx.user_text;
        let status_tx = ctx.status_tx.clone();
        let user_role = ctx.user_role;
        let channel_ctx = ctx.channel_ctx.clone();
        let resume_checkpoint = if is_resume_request(user_text) {
            match self.build_resume_checkpoint(session_id).await {
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
            self.mark_task_interrupted_for_resume(session_id, checkpoint, &task_id)
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
            crate::events::EventEmitter::new(self.event_store.clone(), session_id.to_string())
                .with_task_id(task_id.clone());

        let task_description = if let Some(checkpoint) = resume_checkpoint.as_ref() {
            format!("resume: {}", checkpoint.description)
        } else {
            user_text.to_string()
        };

        // Emit TaskStart event
        let _ = emitter
            .emit(
                EventType::TaskStart,
                TaskStartData {
                    task_id: task_id.clone(),
                    description: task_description.chars().take(200).collect(),
                    parent_task_id: resumed_from_task_id,
                    user_message: Some(user_text.to_string()),
                },
            )
            .await;

        // 1. Persist the user message
        let user_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "user".to_string(),
            content: Some(user_text.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5, // Will be updated by score_message below
            embedding: None,
        };
        // Calculate heuristic score immediately
        let score = crate::memory::scoring::score_message(&user_msg);
        let mut user_msg = user_msg;
        user_msg.importance = score;

        self.append_user_message_with_event(&emitter, &user_msg, &channel_ctx, false)
            .await?;

        if let Some(reply) = self
            .maybe_handle_stop_command(
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
        self.maybe_cancel_work_for_mid_task_pivot(
            session_id,
            user_text,
            user_role,
            &channel_ctx,
            status_tx.clone(),
            &task_id,
        )
        .await;

        if let Some(reply) = self
            .maybe_handle_pending_goal_confirmation(
                session_id, user_text, user_role, &task_id, &emitter,
            )
            .await?
        {
            return Ok(BootstrapOutcome::Return(Ok(reply)));
        }

        if let Some(reply) = self
            .maybe_handle_trivial_ack_shortcut(session_id, user_text, &task_id, &emitter)
            .await?
        {
            return Ok(BootstrapOutcome::Return(Ok(reply)));
        }

        if let Some(reply) = self
            .maybe_handle_time_query_shortcut(session_id, user_text, &task_id, &emitter)
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
        let owner_dm_fact_cache = if self.depth == 0
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
                if let Ok(mut facts) = self.state.get_facts(Some(cat)).await {
                    identity_facts.append(&mut facts);
                }
            }
            Some(identity_facts)
        } else {
            None
        };
        if self.depth == 0
            && user_role == UserRole::Owner
            && channel_ctx.should_inject_personal_memory()
        {
            if let Some(query) = critical_fact_query {
                let facts = owner_dm_fact_cache.as_deref().unwrap_or(&[]);
                let mut summary = extract_critical_fact_summary(facts);
                if summary.assistant_name.is_none() {
                    summary.assistant_name = self.system_prompt.lines().find_map(|line| {
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
                let reply = self
                    .emit_bootstrap_direct_reply(
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

        let mut is_personal_memory_recall_turn =
            looks_like_personal_memory_recall_question(user_text);
        let is_reaffirmation_challenge_turn = user_is_reaffirmation_challenge(user_text);
        if is_reaffirmation_challenge_turn && !is_personal_memory_recall_turn {
            if let Ok(history) = self.state.get_history(session_id, 8).await {
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
            let (mut defs, mut caps) = self.tool_definitions_with_capabilities(user_text).await;

            // Filter tools by channel visibility
            if channel_ctx.visibility == ChannelVisibility::PublicExternal {
                let allowed = ["web_search", "remember_fact", "system_info"];
                defs.retain(|d| {
                    Self::tool_name_from_definition(d).is_some_and(|name| allowed.contains(&name))
                });
                caps.retain(|name, _| allowed.contains(&name.as_str()));
            }

            if restrict_to_personal_memory_tools {
                defs = filter_tool_defs_for_personal_memory(&defs);
                caps.retain(|name, _| is_personal_memory_tool(name));
            }

            available_capabilities = caps;
            base_tool_defs = defs.clone();
            tool_defs = defs;
        }

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
            let shadow_filtered = self.filter_tool_definitions_for_policy(
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
            if self.policy_config.policy_shadow_mode {
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
            if self.policy_config.tool_filter_enforce {
                tool_defs = shadow_filtered;
            }
        }

        if restrict_to_personal_memory_tools {
            tool_defs = filter_tool_defs_for_personal_memory(&tool_defs);
            base_tool_defs = filter_tool_defs_for_personal_memory(&base_tool_defs);
            available_capabilities.retain(|name, _| is_personal_memory_tool(name));
        }

        // Keep provider + router consistent for this task, even if runtime reloads.
        let llm_runtime_snapshot = self.llm_runtime.snapshot();
        let llm_provider = llm_runtime_snapshot.provider();
        let llm_router = if self.depth == 0 {
            llm_runtime_snapshot.router()
        } else {
            None
        };

        // Model selection: route to the appropriate model.
        // Consultant text-only pre-pass is disabled; iteration 1 runs deterministic
        // control-plane routing before entering the normal tool-enabled loop.
        let (selected_model, mut consultant_pass_active) = {
            let is_override = *self.model_override.read().await;
            if !is_override {
                if let Some(ref router) = llm_router {
                    let new_model = router
                        .select_for_profile(policy_bundle.policy.model_profile)
                        .to_string();
                    let routed_model = new_model;
                    if self.policy_config.policy_shadow_mode {
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
                    (routed_model, false)
                } else {
                    // No router: for top-level auto mode, pick the model from the same
                    // runtime snapshot as provider/router to avoid transient reload races.
                    // Sub-agents keep their local model selection behavior.
                    let m = if self.depth == 0 {
                        llm_runtime_snapshot.primary_model()
                    } else {
                        self.model.read().await.clone()
                    };
                    (m, false)
                }
            } else {
                // Model override keeps normal loop behavior.
                let m = self.model.read().await.clone();
                (m, false)
            }
        };
        let mut model = selected_model.clone();
        let route_failsafe_active = route_failsafe_active_for_session(session_id);
        if route_failsafe_active {
            // Fail-safe mode: bypass consultant direct-return routing and force
            // strong profile/model selection for this turn.
            consultant_pass_active = false;
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
                tool_defs = self.filter_tool_definitions_for_policy(
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
        }

        // 2. Build system prompt ONCE before the loop: match skills + inject facts + memory
        let system_prompt = self
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
            )
            .await?;

        // 2b. Retrieve Context ONCE (Optimization)
        // Canonical read path: events first, state-context fallback.
        let mut initial_history = self.load_initial_history(session_id, user_text, 50).await?;

        // Optimize: Identify "Pinned" memories (Relevant/Salient but old) to avoid re-fetching
        let recency_window = 20;
        let recent_ids: std::collections::HashSet<String> = initial_history
            .iter()
            .rev()
            .take(recency_window)
            .map(|m| m.id.clone())
            .collect();

        let pinned_memories: Vec<Message> = initial_history
            .drain(..)
            .filter(|m| !recent_ids.contains(&m.id))
            .collect();

        info!(
            session_id,
            total_context = initial_history.len(),
            pinned_old_memories = pinned_memories.len(),
            depth = self.depth,
            "Context prepared"
        );

        // 2c. Load conversation summary for context window management
        let session_summary = if self.context_window_config.enabled {
            self.state
                .get_conversation_summary(session_id)
                .await
                .ok()
                .flatten()
        } else {
            None
        };

        let data = BootstrapData {
            task_id,
            emitter,
            learning_ctx,
            is_personal_memory_recall_turn,
            is_reaffirmation_challenge_turn,
            requests_external_verification,
            restrict_to_personal_memory_tools,
            personal_memory_tool_call_cap,
            tools_allowed_for_user,
            available_capabilities,
            base_tool_defs,
            tool_defs,
            policy_bundle,
            llm_provider,
            llm_router,
            model,
            consultant_pass_active,
            route_failsafe_active,
            system_prompt,
            pinned_memories,
            session_summary,
        };

        Ok(BootstrapOutcome::Continue(Box::new(data)))
    }
}
