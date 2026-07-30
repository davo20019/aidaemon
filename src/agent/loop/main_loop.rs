use super::bootstrap_phase::{BootstrapCtx, BootstrapData, BootstrapOutcome};
use super::llm_phase::{LlmPhaseCtx, LlmPhaseOutcome};
use super::message_build_phase::{MessageBuildCtx, MessageBuildData};
use super::orchestration_phase::OrchestrationCtx;
use super::response_phase::{ResponsePhaseCtx, ResponsePhaseOutcome};
use super::stopping_phase::{StoppingPhaseCtx, StoppingPhaseOutcome};
use super::tool_execution_phase::{ToolExecutionCtx, ToolExecutionOutcome};
use super::tool_prelude_phase::{ToolPreludeCtx, ToolPreludeOutcome};
use super::*;
use crate::events::TaskOutcome;

/// Check if a cancel keyword appears in text without a preceding negation.
/// Returns false for phrases like "do not stop", "don't cancel", "never stop".
fn cancel_keyword_not_negated(text: &str, keyword: &str) -> bool {
    if !contains_keyword_as_words(text, keyword) {
        return false;
    }
    // Find the keyword position and check the 1-3 words before it for negation.
    let words: Vec<&str> = text.split_whitespace().collect();
    let kw_lower = keyword.to_ascii_lowercase();
    for (i, w) in words.iter().enumerate() {
        let normalized = w
            .trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'')
            .to_ascii_lowercase();
        if normalized == kw_lower {
            // Check up to 3 words before for negation markers.
            let start = i.saturating_sub(3);
            for word in &words[start..i] {
                let prev = word
                    .trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'')
                    .to_ascii_lowercase();
                if matches!(
                    prev.as_str(),
                    "not" | "don't" | "dont" | "no" | "never" | "shouldn't" | "without"
                ) {
                    return false;
                }
            }
        }
    }
    true
}

/// Build a user-facing message when the force-text safety net fires and there
/// is no salvageable tool output to return.
///
/// The legacy fallback ("I ran into a processing limit. Please try again or
/// rephrase your request.") is misleading: the common real cause is a terse,
/// under-specified request (e.g. a bare "web search" with no query) that the
/// model could never turn into an actionable tool call, so it spun producing
/// low-signal stubs until the safety net tripped — no tool ever ran, so there
/// is nothing to salvage. In that case, ask for the missing detail instead of
/// claiming a limit the user can do nothing about.
fn build_stuck_no_output_fallback(user_text: &str) -> String {
    let trimmed = user_text.trim();
    let lower = trimmed.to_ascii_lowercase();
    // A short message that names a lookup action but carries no object is the
    // classic "couldn't form a query" case. 60 chars comfortably covers bare
    // commands like "web search" / "look it up" without matching real queries.
    let is_terse = trimmed.chars().count() <= 60;
    let mentions_search = [
        "search",
        "look up",
        "look it up",
        "google",
        "find out",
        "web search",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));
    if is_terse && mentions_search {
        return "I wasn't sure what to search for. What would you like me to look up?".to_string();
    }
    "I wasn't able to complete that. Could you rephrase with a bit more detail about what \
     you'd like me to do?"
        .to_string()
}

fn infer_deterministic_orchestration_intent(user_text: &str) -> IntentGateDecision {
    let mut intent_gate = infer_intent_gate(user_text, "");
    let lower = user_text.trim().to_ascii_lowercase();
    let explicit_cancel_command = lower == "/cancel" || lower.starts_with("/cancel ");

    // Single-word cancel keywords ("cancel", "stop", "abort") are only treated
    // as cancel intent in SHORT messages (< 80 chars). In long task descriptions
    // they are almost always part of instructions, not commands.
    // Multi-word phrases ("never mind", "forget it", "scratch that") are
    // unambiguous regardless of message length.
    let short_msg = lower.len() < 80;
    let has_cancel_phrase = if short_msg {
        [
            "cancel",
            "stop",
            "abort",
            "never mind",
            "nevermind",
            "forget it",
            "scratch that",
        ]
        .iter()
        .any(|kw| cancel_keyword_not_negated(&lower, kw))
    } else {
        ["never mind", "nevermind", "forget it", "scratch that"]
            .iter()
            .any(|kw| cancel_keyword_not_negated(&lower, kw))
    };
    if explicit_cancel_command || has_cancel_phrase {
        let targeted_cancel = [
            "goal",
            "task",
            "job",
            "this goal",
            "that goal",
            "specific",
            "id",
        ]
        .iter()
        .any(|kw| contains_keyword_as_words(&lower, kw));
        intent_gate.cancel_intent = Some(true);
        intent_gate.cancel_scope = Some(if targeted_cancel {
            "targeted".to_string()
        } else {
            "generic".to_string()
        });
    }
    intent_gate
}

// impl-Agent justification: handle_message_impl drives the turn lifecycle and owns TurnState construction.
impl Agent {
    /// Run the agentic loop for a user message in the given session.
    /// Returns the final assistant text response.
    /// `heartbeat` is an optional atomic timestamp updated on each activity point.
    /// Channels pass `Some(heartbeat)` so the typing indicator can detect stalls;
    /// sub-agents, triggers, and tests pass `None`.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_message_impl(
        &self,
        session_id: &str,
        user_text: &str,
        attachments: &[crate::traits::MessageAttachment],
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
    ) -> anyhow::Result<String> {
        touch_heartbeat(&heartbeat);
        info!(session_id, "handle_message_impl: starting bootstrap phase");
        let services = super::services::AgentServices::new(self);

        let bootstrap_outcome = super::bootstrap_phase::run_bootstrap_phase(
            &services,
            &BootstrapCtx {
                session_id,
                user_text,
                attachments,
                status_tx: status_tx.clone(),
                user_role,
                channel_ctx: &channel_ctx,
            },
        )
        .await?;
        let BootstrapData {
            task_id,
            emitter,
            mut learning_ctx,
            is_personal_memory_recall_turn,
            is_reaffirmation_challenge_turn,
            requests_external_verification,
            restrict_to_personal_memory_tools,
            active_skill_names,
            active_untrusted_external_reference_skills,
            restrict_untrusted_external_reference_tools,
            personal_memory_tool_call_cap,
            tools_allowed_for_user,
            mut available_capabilities,
            mut base_tool_defs,
            mut tool_defs,
            mut policy_bundle,
            llm_provider,
            llm_router,
            mut model,
            route_failsafe_active,
            core_prompt_bytes,
            task_context_tail,
            mut session_summary,
            mut harness_eval,
        } = match bootstrap_outcome {
            BootstrapOutcome::Return(result) => return result,
            BootstrapOutcome::Continue(data) => *data,
        };
        let mut turn_context = self
            .build_turn_context_from_recent_history(session_id, user_text)
            .await;
        let followup_mode = turn_context
            .followup_mode
            .map(|mode| mode.as_str())
            .unwrap_or("unknown");
        let turn_context_reasons: Vec<&'static str> = turn_context
            .reasons
            .iter()
            .map(|reason| reason.as_code())
            .collect();
        let llm_user_text = if matches!(
            turn_context.followup_mode,
            Some(FollowupMode::Followup | FollowupMode::ClarificationAnswer)
        ) {
            let dialogue_state = self
                .state
                .get_dialogue_state(session_id)
                .await
                .ok()
                .flatten();
            super::dialogue_state::compose_llm_user_text(
                turn_context.followup_mode,
                dialogue_state.as_ref(),
                user_text,
                chrono::Utc::now(),
            )
        } else {
            user_text.to_string()
        };
        let mut completion_progress = CompletionProgress::new(&turn_context.completion_contract);
        harness_eval.set_completion_context(
            format!("{:?}", turn_context.completion_contract.task_kind).to_lowercase(),
            turn_context
                .followup_mode
                .map(|mode| mode.as_str().to_string()),
        );
        harness_eval.record_completion_contract(&turn_context.completion_contract);
        if self.harness_eval_enabled() {
            self.install_harness_eval(harness_eval).await;
        }
        let harness_eval_handle = if self.harness_eval_enabled() {
            Some(self.harness_eval_handle())
        } else {
            None
        };
        let (execution_budget_tier, execution_budget_route, execution_budget) =
            select_initial_execution_budget(user_text, &turn_context, self.depth, self.role);
        #[cfg(test)]
        let execution_budget = self
            .execution_budget_override
            .clone()
            .unwrap_or(execution_budget);
        let mut execution_state = ExecutionState::new(
            execution_budget_tier,
            execution_budget.clone(),
            if self.depth > 0 || self.task_id.is_some() {
                ExecutionPersistence::Durable
            } else {
                ExecutionPersistence::Ephemeral
            },
        );
        execution_state.mark_persisted_now();
        self.emit_decision_point(
            &emitter,
            &task_id,
            0,
            DecisionType::ExecutionBudgetSelection,
            "Selected initial execution budget tier".to_string(),
            json!({
                "condition": "initial_execution_budget_selected",
                "budget_tier": execution_budget_tier,
                "route_kind": execution_budget_route,
                "budget": execution_budget,
                "persistence": execution_state.persistence,
                "execution_id": execution_state.execution_id,
            }),
        )
        .await;
        self.emit_decision_point(
            &emitter,
            &task_id,
            0,
            DecisionType::ExecutionStateSnapshot,
            "Initialized execution state snapshot".to_string(),
            json!({
                "condition": "execution_state_initialized",
                "execution_state": execution_state.clone(),
            }),
        )
        .await;
        info!(
            session_id,
            followup_mode,
            reasons = ?turn_context_reasons,
            primary_project_scope = ?turn_context.primary_project_scope,
            allow_multi_project_scope = turn_context.allow_multi_project_scope,
            "Turn context resolved"
        );
        // 3. Agentic loop — runs until natural completion or safety limits.
        //
        // The whole loop runs inside a `turn` span carrying task_id + session_id
        // so every downstream log line is correlated with this request without
        // editing each call site. We use `Instrument` (not a held `enter()`
        // guard) because the guard pattern is unsound across `.await` points —
        // it would leak the span onto whatever task the worker thread polls next.
        let turn_span = tracing::info_span!("turn", task_id = %task_id, session_id = %session_id);
        tracing::Instrument::instrument(async move {
        let task_start = Instant::now();
        let mut last_progress_summary = Instant::now();
        const MAX_FORCE_TEXT_ITERATIONS: usize = 3;
        let mut approach_pivots_used: usize = 0;
        const MAX_BUDGET_EXTENSIONS: usize = 3;
        const HARD_TOKEN_CAP: i64 = 2_000_000;
        // Scheduled work must stop at the user-approved per-run budget. Any
        // increase requires an explicit approval instead of silently doubling
        // the budget for an unattended run.
        const SCHEDULED_MAX_BUDGET_EXTENSIONS: usize = 0;
        const SCHEDULED_HARD_TOKEN_CAP: i64 = 2_000_000;

        let iteration_limits = match &self.limits.iteration_config {
            IterationLimitConfig::Unlimited => super::loop_state::IterationLimitSettings {
                hard_cap: Some(HARD_ITERATION_CAP),
                soft_threshold: None,
                soft_warn_at: None,
            },
            IterationLimitConfig::Soft { threshold, warn_at } => {
                super::loop_state::IterationLimitSettings {
                    hard_cap: Some(HARD_ITERATION_CAP),
                    soft_threshold: Some(*threshold),
                    soft_warn_at: Some(*warn_at),
                }
            }
            IterationLimitConfig::Hard { initial: _, cap } => {
                super::loop_state::IterationLimitSettings {
                    hard_cap: Some(*cap),
                    soft_threshold: None,
                    soft_warn_at: None,
                }
            }
        };
        let mut turn_state = super::loop_state::TurnState {
            stall: super::loop_state::StallTracker::with_recent_capacity(RECENT_CALLS_WINDOW),
            failures: super::loop_state::FailureLedger::default(),
            recovery: super::loop_state::RecoveryState::default(),
            budget: super::loop_state::BudgetTracker::new(
                self.limits.task_token_budget,
                self.limits.daily_token_budget,
                iteration_limits,
            ),
            evidence: super::loop_state::EvidenceLedger::default(),
            reflection: super::loop_state::ReflectionState::default(),
            directives: super::loop_state::PendingDirectives::default(),
            counters: super::loop_state::LoopCounters::new(
                infer_intent_gate(user_text, "")
                    .needs_tools
                    .unwrap_or(false),
            ),
            read_files: super::loop_state::ReadFileObservationTracker::default(),
            harness_eval: None,
            eval: None,
        };
        if let Some(handle) = harness_eval_handle {
            turn_state.attach_harness_eval(handle);
        }

        // Task-start planning call: generate a structured plan before the main loop.
        // Skipped for conversational queries, short messages, and acknowledgments.
        // In tests, MockProvider silently intercepts planning calls (skip_planning_calls=true).
        {
            use super::bootstrap_phase::task_planning::{generate_task_plan, planning_skip_reason};
            use crate::agent::execution_state::LinearIntentStep;
            let planner_trust_tier = self.trust_tier_for_model(&model).as_str();
            let planner_skip_reason = planning_skip_reason(user_text, false);
            if let Some(reason) = planner_skip_reason {
                self.emit_decision_point(
                    &emitter,
                    &task_id,
                    0,
                    crate::events::DecisionType::HandHoldingTelemetry,
                    "Task planner skipped".to_string(),
                    super::hand_holding_telemetry::planner_skip_metadata(
                        reason,
                        &model,
                        planner_trust_tier,
                    ),
                )
                .await;
            } else {
                // Build conversation context from recent messages for the planner.
                // This ensures the planner sees the same narrative as the main LLM,
                // preventing intent loss across multi-hop follow-ups.
                let planner_context = {
                    let mut ctx_parts = Vec::new();
                    if let Some(ref summary) = session_summary {
                        if !summary.summary.is_empty() {
                            ctx_parts.push(format!("[Session Summary] {}", summary.summary));
                        }
                    }
                    for msg in &turn_context.recent_messages {
                        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
                        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                        if !content.is_empty() {
                            ctx_parts.push(format!(
                                "- {}: {}",
                                role.chars().next().unwrap_or('?').to_uppercase(),
                                content
                            ));
                        }
                    }
                    if ctx_parts.is_empty() {
                        None
                    } else {
                        Some(ctx_parts.join("\n"))
                    }
                };
                let planner_model = llm_router
                    .as_ref()
                    .map(|router| router.select(crate::router::Tier::Primary))
                    .unwrap_or(model.as_str());
                self.emit_decision_point(
                    &emitter,
                    &task_id,
                    0,
                    crate::events::DecisionType::HandHoldingTelemetry,
                    "Task planner attempted".to_string(),
                    super::hand_holding_telemetry::planner_result_metadata(
                        "attempted",
                        planner_model,
                        planner_trust_tier,
                        super::hand_holding_telemetry::PlannerResultStats::empty(),
                        None,
                    ),
                )
                .await;
                // Single-model deployments still run the planner. The router
                // chooses a model when present; otherwise the active model
                // reads the request instead of leaving lexical inference as
                // the only contract classifier.
                let plan_opt = generate_task_plan(
                    llm_provider.clone(),
                    planner_model,
                    user_text,
                    planner_context.as_deref(),
                    Some(super::bootstrap_phase::task_planning::PlannerTelemetryCtx {
                        emitter: &emitter,
                        state: self.state.as_ref(),
                        session_id,
                        task_id: &task_id,
                    }),
                )
                .await;
                if let Some(plan) = plan_opt {
                    let before_contract = turn_context.completion_contract.clone();
                    // The planner read the actual request (any language), so
                    // its contract classification refines the English-keyword
                    // inference. Explicit user verification requests are
                    // never relaxed (enforced inside apply).
                    if let Some(ref signals) = plan.contract {
                        let planned_kind = signals
                            .task_kind
                            .as_deref()
                            .and_then(crate::agent::parse_planned_task_kind);
                        crate::agent::apply_planned_contract_signals(
                            &mut turn_context.completion_contract,
                            signals.expects_mutation,
                            signals.requires_observation,
                            planned_kind,
                        );
                        if turn_context.completion_contract != before_contract {
                            info!(
                                session_id,
                                task_kind = ?turn_context.completion_contract.task_kind,
                                expects_mutation =
                                    turn_context.completion_contract.expects_mutation,
                                requires_observation =
                                    turn_context.completion_contract.requires_observation,
                                "Completion contract refined by planner classification"
                            );
                        }
                    }
                    let contract_changed = turn_context.completion_contract != before_contract;
                    self.emit_decision_point(
                        &emitter,
                        &task_id,
                        0,
                        crate::events::DecisionType::HandHoldingTelemetry,
                        "Task planner succeeded".to_string(),
                        super::hand_holding_telemetry::planner_result_metadata(
                            "succeeded",
                            &model,
                            planner_trust_tier,
                            super::hand_holding_telemetry::PlannerResultStats {
                                step_count: plan.steps.len(),
                                success_criteria_count: plan.success_criteria.len(),
                                contract_present: plan.contract.is_some(),
                                contract_changed,
                            },
                            None,
                        ),
                    )
                    .await;
                    let linear_steps: Vec<LinearIntentStep> = plan
                        .steps
                        .iter()
                        .enumerate()
                        .map(|(i, step)| LinearIntentStep {
                            step_id: format!("task-plan-step-{}", i + 1),
                            step_index: i + 1,
                            tool: step.tool_hint.clone().unwrap_or_default(),
                            target: String::new(),
                            description: step.description.clone(),
                            tool_calls_on_step: 0,
                            completed: false,
                            completion_evidence: None,
                            last_evaluated_at: None,
                        })
                        .collect();

                    let step_count = linear_steps.len();
                    execution_state.install_linear_intent_plan(1, linear_steps);

                    if !plan.success_criteria.is_empty() {
                        turn_state
                            .evidence
                            .validation_state_mut()
                            .set_plan(1, &plan.success_criteria);
                    }

                    execution_state.promote_budget_for_plan(step_count);

                    info!(
                        session_id,
                        goal = %plan.goal,
                        step_count,
                        "Task plan installed and budget evaluated"
                    );
                } else {
                    self.emit_decision_point(
                        &emitter,
                        &task_id,
                        0,
                        crate::events::DecisionType::HandHoldingTelemetry,
                        "Task planner returned no plan".to_string(),
                        super::hand_holding_telemetry::planner_result_metadata(
                            "no_plan",
                            &model,
                            planner_trust_tier,
                            super::hand_holding_telemetry::PlannerResultStats::empty(),
                            Some("planner_returned_none"),
                        ),
                    )
                    .await;
                }
            }
        }

        if route_failsafe_active {
            turn_state
                .directives
                .push_system_message(SystemDirective::RouteFailsafeActive);
        }
        let has_recent_tool_context = turn_context
            .recent_messages
            .iter()
            .any(|row| row.get("role").and_then(|v| v.as_str()) == Some("tool"));
        if looks_like_evidence_grounding_challenge(user_text)
            && (turn_context.followup_mode != Some(FollowupMode::NewTask)
                || has_recent_tool_context)
        {
            turn_state
                .directives
                .push_system_message(SystemDirective::EvidenceGroundingRequired);
        }
        // Short approval of the assistant's own proposal ("Yes, read it"):
        // anchor execution to the quoted proposal so the model carries out
        // exactly what it offered instead of re-planning from scratch.
        if let Some(proposal) =
            super::tool_prelude_phase::approved_proposal_text(&turn_context)
        {
            turn_state
                .directives
                .push_system_message(SystemDirective::ApprovedProposalAnchor { proposal });
        }
        // Only pin the model to the prior exchange for genuinely vague
        // challenges ("Are you sure?") — compound/new-task messages that merely
        // contain a challenge keyword must not be anchored away from their
        // actual request.
        if is_reaffirmation_challenge_turn
            && crate::agent::recall_guardrails::is_vague_reaffirmation_challenge(user_text)
        {
            if let Ok(history) = self.state.get_history(session_id, 12).await {
                if let Some(anchor) = crate::agent::recall_guardrails::resolve_reaffirmation_anchor(
                    &history, user_text,
                ) {
                    turn_state.directives.push_system_message(
                        SystemDirective::ReaffirmationChallengeAnchor {
                            prior_user_request: anchor.prior_user_request,
                            prior_assistant_reply: anchor.prior_assistant_reply,
                        },
                    );
                }
            }
        }
        // Coreference grounding: a follow-up that carries its person referent
        // only via a pronoun ("...what can you infer about her?") is prone to
        // binding the pronoun to the salient pinned-profile person instead of
        // the actual subject of the prior exchange. Anchor it to that exchange
        // and force a memory lookup before answering.
        else if crate::agent::recall_guardrails::looks_like_pronoun_referent_followup(user_text) {
            if let Ok(history) = self.state.get_history(session_id, 12).await {
                if let Some(anchor) = crate::agent::recall_guardrails::resolve_reaffirmation_anchor(
                    &history, user_text,
                ) {
                    turn_state.directives.push_system_message(
                        SystemDirective::CoreferenceGroundingRequired {
                            prior_user_request: anchor.prior_user_request,
                            prior_assistant_reply: anchor.prior_assistant_reply,
                        },
                    );
                    // Signal the denial gate: coreference fired first this turn.
                    // The two gates are mutually exclusive — coreference takes
                    // precedence so the denial gate must not also fire and
                    // inject a second, contradictory directive.
                    completion_progress.coreference_fired = true;
                }
            }
        }
        // Best-effort project directory hint (seeded from user text, refined by tool calls).
        if let Some(known_project_dir) = turn_context.primary_project_scope.clone().or_else(|| {
            super::tool_execution_phase::extract_project_dir_hint_with_aliases(
                user_text,
                &self.path_aliases.projects,
            )
        }) {
            turn_state.evidence.set_known_project_dir(known_project_dir);
        }

        // Resolve goal_id once for per-goal token budget enforcement.
        // Executors currently carry only task_id, so we may need to lookup goal_id via task.
        let resolved_goal_id: Option<String> = if let Some(gid) = self.goal_id.clone() {
            Some(gid)
        } else if let Some(ref tid) = self.task_id {
            match self.state.get_task(tid).await {
                Ok(Some(task)) => Some(task.goal_id),
                Ok(None) => {
                    warn!(
                        session_id,
                        task_id = %tid,
                        "Task not found while resolving goal_id; goal budget enforcement disabled for this run"
                    );
                    None
                }
                Err(e) => {
                    warn!(
                        session_id,
                        task_id = %tid,
                        error = %e,
                        "Failed to resolve goal_id from task; goal budget enforcement disabled for this run"
                    );
                    None
                }
            }
        } else {
            None
        };
        let is_scheduled_goal = if let Some(goal_id) = resolved_goal_id.as_deref() {
            goal_has_scheduled_provenance(&self.state, goal_id, self.task_id.as_deref()).await
        } else {
            false
        };
        let is_root_scheduled_run = if self.task_id.is_none() {
            is_scheduled_goal
        } else {
            task_has_scheduled_provenance(&self.state, self.task_id.as_deref()).await
        };
        let scheduled_goal_budget_per_check = if let Some(goal_id) = resolved_goal_id.as_deref() {
            self.state
                .get_goal(goal_id)
                .await
                .ok()
                .flatten()
                .and_then(|g| g.budget_per_check)
        } else {
            None
        };
        let active_scheduled_root_task_id = if let Some(goal_id) = resolved_goal_id.as_deref() {
            if is_scheduled_goal {
                active_scheduled_root_task_id(&self.state, goal_id).await
            } else {
                None
            }
        } else {
            None
        };
        if is_scheduled_goal {
            turn_state.budget.disable_iteration_limits();
            if let Some(registry) = self.goal_token_registry.as_ref() {
                if let Some(goal_id) = resolved_goal_id.as_deref() {
                    if is_root_scheduled_run {
                        let persisted_state = self
                            .state
                            .get_scheduled_run_state(goal_id)
                            .await
                            .ok()
                            .flatten();
                        let restored = if let Some(state) = persisted_state.as_ref() {
                            if Some(state.root_task_id.as_str())
                                == active_scheduled_root_task_id.as_deref()
                            {
                                registry
                                    .restore_run_budget(
                                        goal_id,
                                        state.effective_budget_per_check,
                                        state.tokens_used,
                                        state.budget_extensions_count,
                                        state.health.clone(),
                                    )
                                    .await
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        if restored.is_none() {
                            registry
                                .start_run_budget(goal_id, scheduled_goal_budget_per_check)
                                .await;
                            if let Some(status) = registry.get_run_budget(goal_id).await {
                                persist_scheduled_run_state(
                                    &self.state,
                                    goal_id,
                                    active_scheduled_root_task_id.as_deref(),
                                    &status,
                                )
                                .await;
                            } else {
                                clear_scheduled_run_state(&self.state, goal_id).await;
                            }
                        }
                    } else if registry.get_run_budget(goal_id).await.is_none() {
                        if let Some(state) = self
                            .state
                            .get_scheduled_run_state(goal_id)
                            .await
                            .ok()
                            .flatten()
                        {
                            let _ = registry
                                .restore_run_budget(
                                    goal_id,
                                    state.effective_budget_per_check,
                                    state.tokens_used,
                                    state.budget_extensions_count,
                                    state.health.clone(),
                                )
                                .await;
                        } else {
                            registry
                                .start_run_budget(goal_id, scheduled_goal_budget_per_check)
                                .await;
                            if let Some(status) = registry.get_run_budget(goal_id).await {
                                persist_scheduled_run_state(
                                    &self.state,
                                    goal_id,
                                    active_scheduled_root_task_id.as_deref(),
                                    &status,
                                )
                                .await;
                            }
                        }
                    }
                }
            }
            if let Some(per_check_budget) =
                scheduled_goal_budget_per_check.and_then(|v| u64::try_from(v).ok())
            {
                turn_state
                    .budget
                    .raise_effective_task_budget_to(per_check_budget);
            }
        }
        let effective_task_timeout = if is_scheduled_goal {
            None
        } else {
            self.limits.task_timeout
        };
        let max_budget_extensions = if is_scheduled_goal {
            SCHEDULED_MAX_BUDGET_EXTENSIONS
        } else {
            MAX_BUDGET_EXTENSIONS
        };
        let hard_token_cap = if is_scheduled_goal {
            SCHEDULED_HARD_TOKEN_CAP
        } else {
            HARD_TOKEN_CAP
        };
        // Runtime-only override for goal daily budget extensions.
        // Shared via GoalTokenRegistry so task-leads/executors for the same goal
        // can inherit the same temporary budget without persisting it to SQLite.
        let mut effective_goal_daily_budget: Option<i64> = if let (Some(goal_id), Some(registry)) = (
            resolved_goal_id.as_deref(),
            self.goal_token_registry.as_ref(),
        ) {
            registry.get_effective_daily_budget(goal_id).await
        } else {
            None
        };

        // Correction-execution context for this turn. `Some` ONLY when this agent
        // is running under a deliberately-dispatched remediation goal id (see
        // `correction_dispatch::dispatch_correction_remediation` +
        // `Agent::correction_context_for_current_goal`). `None` for every normal,
        // user-initiated turn — preserving the prior hardcoded `correction: None`
        // behavior byte-for-byte on non-remediation paths. Fetched once here and
        // cloned into each iteration's `ToolExecutionCtx` so the P2.4 per-call
        // sandbox gate fires on the remediation task-lead and all its executors.
        let correction_context = self.correction_context_for_current_goal().await;

        loop {
            let iteration = turn_state.counters.advance_iteration();
            touch_heartbeat(&heartbeat);
            let plan_progress = execution_state
                .active_linear_intent_plan
                .as_ref()
                .map(|plan| {
                    (
                        plan.steps.iter().filter(|step| step.completed).count() as u32,
                        plan.steps.len() as u32,
                    )
                });
            #[cfg(feature = "computer_use")]
            {
                model =
                    crate::agent::computer_use::resolve_model_for_task(&task_id, &model).await;
            }
            turn_state
                .with_harness_eval(|eval| {
                    eval.record_completion_progress(&completion_progress);
                    eval.record_iteration_progress(
                        iteration as u32,
                        turn_state.counters.total_tool_calls_attempted() as u32,
                        turn_state.counters.total_successful_tool_calls() as u32,
                        turn_state.evidence.evidence_gain_count() as u32,
                        false,
                    );
                    if let Some((completed, total)) = plan_progress {
                        eval.record_plan_progress(completed, total);
                    }
                })
                .await;

            // Check for cancellation (cascades via token hierarchy)
            if let Some(ref ct) = self.cancel_token {
                if ct.is_cancelled() {
                    info!(session_id, iteration, "Task cancelled by parent");
                    self.with_harness_eval(|eval| eval.record_stop_reason(StopReason::Cancelled))
                        .await;
                    self.emit_decision_point(
                        &emitter,
                        &task_id,
                        iteration,
                        DecisionType::StoppingCondition,
                        "Stopping condition fired: cancellation token set".to_string(),
                        json!({"condition":"cancelled"}),
                    )
                    .await;

                    // Mark remaining tasks as cancelled.
                    if let Some(ref gid) = self.goal_id {
                        if let Ok(tasks) = self.state.get_tasks_for_goal(gid).await {
                            for task in &tasks {
                                if task.status != "completed"
                                    && task.status != "failed"
                                    && task.status != "cancelled"
                                {
                                    let mut ct = task.clone();
                                    ct.status = "cancelled".to_string();
                                    let _ = self.state.update_task(&ct).await;
                                }
                            }
                        }
                    }

                    let cancel_reply = "Task cancelled.".to_string();
                    let assistant_msg = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: session_id.to_string(),
                        role: "assistant".to_string(),
                        content: Some(cancel_reply.clone()),
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.3,
                        ..Message::runtime_defaults()
                    };
                    let _ = self
                        .append_assistant_message_with_event(
                            &emitter,
                            &assistant_msg,
                            "system",
                            None,
                            None,
                        )
                        .await;

                    self.emit_task_end(
                        &emitter,
                        &task_id,
                        TaskStatus::Cancelled,
                        TaskOutcome::Failed,
                        task_start,
                        iteration,
                        0,
                        None,
                        Some(cancel_reply.clone()),
                    )
                    .await;
                    return Ok(cancel_reply);
                }
            }

            // Safety net: if force-text mode has been active for too many
            // consecutive iterations, hard-return whatever the LLM last produced.
            // This prevents infinite force-text loops where the response/completion
            // phase keeps deciding to continue despite having no tools.
            if turn_state.recovery.force_text_response() {
                let force_text_iterations = turn_state.recovery.record_force_text_iteration();
                if force_text_iterations > MAX_FORCE_TEXT_ITERATIONS {
                    warn!(
                        session_id,
                        iteration,
                        force_text_iterations,
                        "Force-text safety net: exceeded max consecutive force-text iterations, hard-stopping"
                    );
                    let fallback = super::stopping_phase::latest_non_system_tool_output_excerpt(
                        self, session_id, 2000,
                    )
                    .await
                    .unwrap_or_else(|| build_stuck_no_output_fallback(user_text));
                    let assistant_msg = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: session_id.to_string(),
                        role: "assistant".to_string(),
                        content: Some(fallback.clone()),
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.3,
                        ..Message::runtime_defaults()
                    };
                    let _ = self
                        .append_assistant_message_with_event(
                            &emitter,
                            &assistant_msg,
                            "force_text_safety_net",
                            None,
                            None,
                        )
                        .await;
                    self.emit_task_end(
                        &emitter,
                        &task_id,
                        TaskStatus::Completed,
                        TaskOutcome::Failed,
                        task_start,
                        iteration,
                        turn_state.budget.task_tokens_used() as usize,
                        None,
                        Some(fallback.clone()),
                    )
                    .await;
                    return Ok(fallback);
                }
            } else {
                turn_state.recovery.reset_force_text_iterations();
            }

            info!(
                iteration,
                session_id,
                model = %model,
                depth = self.depth,
                policy_profile = ?policy_bundle.policy.model_profile,
                verify_level = ?policy_bundle.policy.verify_level,
                approval_mode = ?policy_bundle.policy.approval_mode,
                context_budget = policy_bundle.policy.context_budget,
                tool_budget = policy_bundle.policy.tool_budget,
                policy_rev = policy_bundle.policy.policy_rev,
                risk_score = policy_bundle.risk_score,
                uncertainty_score = policy_bundle.uncertainty_score,
                "Agent loop iteration"
            );

            // Emit ThinkingStart event
            let _ = emitter
                .emit(
                    EventType::ThinkingStart,
                    ThinkingStartData {
                        iteration: iteration as u32,
                        task_id: task_id.clone(),
                        total_tool_calls: learning_ctx.tool_calls.len() as u32,
                    },
                )
                .await;

            let stopping_stall = turn_state.stall.for_stopping_phase();
            let stopping_failures = turn_state.failures.for_stopping_phase();
            let stopping_recovery = turn_state.recovery.for_stopping_phase();
            let stopping_budget = turn_state.budget.for_stopping_phase();
            let stopping_evidence = turn_state.evidence.for_stopping_phase();
            let stopping_directives = turn_state.directives.for_stopping_phase();
            let stopping_counters = turn_state.counters.for_stopping_phase();
            let stopping_outcome = super::stopping_phase::run_stopping_phase(
                &services,
                &mut StoppingPhaseCtx {
                    emitter: &emitter,
                    task_id: &task_id,
                    session_id,
                    iteration,
                    task_start,
                    learning_ctx: &mut learning_ctx,
                    hard_cap: stopping_budget.hard_cap,
                    effective_task_timeout,
                    task_tokens_used: stopping_budget.task_tokens_used,
                    effective_task_budget: stopping_budget.effective_task_budget,
                    budget_warning_sent: stopping_budget.budget_warning_sent,
                    pending_system_messages: stopping_directives.pending_system_messages,
                    budget_extensions_count: stopping_budget.budget_extensions_count,
                    user_role,
                    evidence_gain_count: stopping_evidence.evidence_gain_count,
                    approach_pivots_used,
                    stall_count: stopping_stall.stall_count,
                    deferred_no_tool_streak: stopping_counters.deferred_no_tool_streak,
                    consecutive_same_tool: stopping_stall.consecutive_same_tool,
                    consecutive_same_tool_arg_hashes: stopping_stall
                        .consecutive_same_tool_arg_hashes,
                    total_successful_tool_calls: stopping_counters.total_successful_tool_calls,
                    pending_background_ack: stopping_directives.pending_background_ack,
                    status_tx: &status_tx,
                    resolved_goal_id: &resolved_goal_id,
                    is_scheduled_goal,
                    effective_daily_budget: stopping_budget.effective_daily_budget,
                    effective_goal_daily_budget: &mut effective_goal_daily_budget,
                    successful_send_file_keys: stopping_counters.successful_send_file_keys,
                    model: &mut model,
                    soft_threshold: stopping_budget.soft_threshold,
                    soft_warn_at: stopping_budget.soft_warn_at,
                    soft_limit_warned: stopping_budget.soft_limit_warned,
                    last_progress_summary: &mut last_progress_summary,
                    tool_failure_count: stopping_failures.tool_failure_count,
                    session_summary: &mut session_summary,
                    policy_bundle: &mut policy_bundle,
                    user_text,
                    available_capabilities: &available_capabilities,
                    llm_router: &llm_router,
                    last_escalation_iteration: stopping_stall.last_escalation_iteration,
                    consecutive_clean_iterations: stopping_stall.consecutive_clean_iterations,
                    max_budget_extensions,
                    hard_token_cap,
                    execution_state: &mut execution_state,
                    force_text_response: stopping_recovery.force_text_response,
                    completion_progress: &mut completion_progress,
                    turn_context: &turn_context,
                    validation_state: stopping_evidence.validation_state,
                },
            )
            .await?;
            match stopping_outcome {
                StoppingPhaseOutcome::ContinueLoop => continue,
                StoppingPhaseOutcome::Return(result) => return result,
                StoppingPhaseOutcome::PivotApproach { failure_record } => {
                    approach_pivots_used += 1;
                    // Fresh runway: the stall evidence belongs to the failed
                    // approach. The failure stays referenced in the directive
                    // and verbatim in history — nothing is discarded.
                    turn_state.stall.reset_for_pivot();
                    turn_state.directives.push_system_message(
                        SystemDirective::ApproachPivotRequired {
                            attempt: approach_pivots_used,
                            failure_record,
                        },
                    );
                    crate::agent::heuristic_telemetry::global().record(
                        "approach_pivot",
                        &model,
                        self.trust_tier_for_model(&model),
                        crate::agent::heuristic_telemetry::HeuristicAction::Enforced,
                    );
                    continue;
                }
                StoppingPhaseOutcome::Proceed => {}
            }

            // Deterministic control-plane routing on iteration 1:
            // handle cancel/schedule/complex intents before the first LLM call.
            if iteration == 1
                && self.depth == 0
                && self.role == AgentRole::Orchestrator
                && !route_failsafe_active
            {
                let intent_gate = infer_deterministic_orchestration_intent(user_text);
                if let Some(outcome) = super::orchestration_phase::run_orchestration_phase(
                    &services,
                    &mut OrchestrationCtx {
                        emitter: &emitter,
                        task_id: &task_id,
                        session_id,
                        user_text,
                        iteration,
                        task_start,
                        task_tokens_used: turn_state.budget.task_tokens_used(),
                        pending_system_messages: turn_state
                            .directives
                            .for_message_build_phase()
                            .pending_system_messages,
                        tool_defs: &mut tool_defs,
                        base_tool_defs: &mut base_tool_defs,
                        available_capabilities: &mut available_capabilities,
                        policy_bundle: &mut policy_bundle,
                        tools_allowed_for_user,
                        llm_provider: llm_provider.clone(),
                        llm_router: llm_router.clone(),
                        model: &model,
                        user_role,
                        channel_ctx: channel_ctx.clone(),
                        status_tx: status_tx.clone(),
                        intent_gate: &intent_gate,
                        turn_context: &turn_context,
                    },
                )
                .await?
                {
                    match outcome {
                        ResponsePhaseOutcome::ContinueLoop => {
                            turn_state
                                .with_harness_eval(|eval| eval.record_response_fallthrough())
                                .await;
                            continue;
                        }
                        ResponsePhaseOutcome::Return(result) => return result,
                        ResponsePhaseOutcome::ProceedToToolExecution => {}
                    }
                }
            }

            // Inject task plan context with progress markers into the model's context.
            if let Some(ref plan) = execution_state.active_linear_intent_plan {
                if !plan.steps.is_empty() {
                    let plan_text = plan.format_with_progress();
                    turn_state
                        .directives
                        .push_system_message(SystemDirective::TaskPlanContext(plan_text));
                }
            }

            // Compaction: on the first iteration, detect whether the conversation
            // history needs compacting. IdleGap and FileUpload triggers run
            // synchronously (user just arrived or uploaded a file — latency is
            // acceptable). WindowOverflow runs asynchronously in the background
            // so it doesn't add 15s latency to every message once the conversation
            // exceeds the window size. The aging pair stays in the sliding window
            // until the background compaction completes.
            if iteration == 1 && self.context_window_config.enabled {
                // Count user-message pairs and compute idle gap.
                let history = self
                    .state
                    .get_history(session_id, 100)
                    .await
                    .unwrap_or_default();
                let total_pairs = history.iter().filter(|m| m.role == "user").count();
                let idle_gap_seconds = history
                    .last()
                    .map(|m| {
                        let now = Utc::now();
                        now.signed_duration_since(m.created_at).num_seconds().max(0) as u64
                    })
                    .unwrap_or(0);

                let window_size = self.context_window_config.summary_window;
                let compaction_trigger = super::compaction::detect_compaction_trigger(
                    total_pairs,
                    window_size,
                    idle_gap_seconds,
                    user_text,
                );

                if let Some(ref trigger) = compaction_trigger {
                    info!(
                        session_id,
                        ?trigger,
                        total_pairs,
                        idle_gap_seconds,
                        window_size,
                        "Compaction trigger detected"
                    );

                    // Convert history messages to JSON Value array for the compaction prompt.
                    let messages_to_compact: Vec<Value> = history
                        .iter()
                        .map(|m| {
                            let mut msg = json!({ "role": m.role });
                            if let Some(ref content) = m.content {
                                msg["content"] = json!(content);
                            }
                            if let Some(ref name) = m.tool_name {
                                msg["name"] = json!(name);
                            }
                            msg
                        })
                        .collect();

                    let last_message_id = history.last().map(|m| m.id.as_str()).unwrap_or("");

                    match trigger {
                        super::compaction::CompactionTrigger::WindowOverflow { .. } => {
                            // Async — don't block the user's response. The aging
                            // pair is already in the sliding window (we haven't
                            // removed it). On the next turn the summary's
                            // last_message_id will cover it.
                            let provider = llm_provider.clone();
                            let compaction_model = model.clone();
                            let state = self.state.clone();
                            let sid = session_id.to_string();
                            let summary_clone = session_summary.clone();
                            let msgs = messages_to_compact.clone();
                            let msg_count = total_pairs;
                            let last_id = last_message_id.to_string();
                            tokio::spawn(async move {
                                if let Err(e) = super::compaction::run_and_store_compaction(
                                    provider,
                                    &compaction_model,
                                    state.as_ref(),
                                    &sid,
                                    summary_clone,
                                    &msgs,
                                    msg_count,
                                    &last_id,
                                )
                                .await
                                {
                                    warn!(session_id = %sid, error = %e, "Background compaction failed");
                                }
                            });
                            info!(
                                session_id,
                                "Background compaction spawned for window overflow"
                            );
                        }
                        _ => {
                            // IdleGap / FileUpload — run synchronously (15s timeout).
                            match tokio::time::timeout(
                                Duration::from_secs(15),
                                super::compaction::run_and_store_compaction(
                                    llm_provider.clone(),
                                    &model,
                                    self.state.as_ref(),
                                    session_id,
                                    session_summary.clone(),
                                    &messages_to_compact,
                                    total_pairs,
                                    last_message_id,
                                ),
                            )
                            .await
                            {
                                Ok(Ok(())) => {
                                    // Reload the summary so message build uses the fresh version.
                                    session_summary = self
                                        .state
                                        .get_conversation_summary(session_id)
                                        .await
                                        .ok()
                                        .flatten();
                                    info!(session_id, "Synchronous compaction completed");
                                }
                                Ok(Err(e)) => {
                                    warn!(session_id, error = %e, "Synchronous compaction failed");
                                }
                                Err(_) => {
                                    warn!(session_id, "Synchronous compaction timed out (15s)");
                                }
                            }
                        }
                    }
                }
            }

            let message_build_directives = turn_state.directives.for_message_build_phase();
            let message_build_recovery = turn_state.recovery.for_message_build_phase();
            let message_build_start = Instant::now();
            let MessageBuildData {
                mut messages,
                tool_defs: effective_tool_defs,
                est_input_tokens,
            } = super::message_build_phase::run_message_build_phase(
                &services,
                &mut MessageBuildCtx {
                    session_id,
                    iteration,
                    user_text: &llm_user_text,
                    current_attachments: attachments,
                    completed_tool_calls: &learning_ctx.tool_calls,
                    model: &model,
                    core_prompt: &core_prompt_bytes,
                    task_context_tail: &task_context_tail,
                    tool_defs: &tool_defs,
                    policy_bundle: &policy_bundle,
                    pending_system_messages: message_build_directives.pending_system_messages,
                    empty_response_retry_pending: message_build_recovery
                        .empty_response_retry_pending,
                    status_tx: &status_tx,
                },
            )
            .await?;
            let context_drops = tool_defs.len().saturating_sub(effective_tool_defs.len()) as u32;
            turn_state
                .with_harness_eval(|eval| {
                    eval.record_message_build(
                        effective_tool_defs.len() as u32,
                        est_input_tokens,
                        context_drops,
                    );
                })
                .await;
            let message_build_ms = message_build_start.elapsed().as_millis() as u64;

            let llm_stall = turn_state.stall.for_llm_phase();
            let llm_recovery = turn_state.recovery.for_llm_phase();
            let llm_budget = turn_state.budget.for_llm_phase();
            let llm_evidence = turn_state.evidence.for_llm_phase();
            let llm_directives = turn_state.directives.for_llm_phase();
            let llm_counters = turn_state.counters.for_llm_phase();
            let llm_outcome = super::llm_phase::run_llm_phase(
                &services,
                &mut LlmPhaseCtx {
                    messages: &mut messages,
                    emitter: &emitter,
                    task_id: &task_id,
                    session_id,
                    user_text,
                    iteration,
                    force_text_response: llm_recovery.force_text_response,
                    task_start,
                    task_tokens_used: llm_budget.task_tokens_used,
                    learning_ctx: &mut learning_ctx,
                    pending_system_messages: llm_directives.pending_system_messages,
                    llm_provider: llm_provider.clone(),
                    llm_router: llm_router.clone(),
                    model: &model,
                    user_role,
                    tool_defs: &effective_tool_defs,
                    status_tx: &status_tx,
                    resolved_goal_id: &resolved_goal_id,
                    is_scheduled_goal,
                    effective_goal_daily_budget: &mut effective_goal_daily_budget,
                    budget_extensions_count: llm_budget.budget_extensions_count,
                    evidence_gain_count: llm_evidence.evidence_gain_count,
                    stall_count: llm_stall.stall_count,
                    consecutive_same_tool: llm_stall.consecutive_same_tool,
                    consecutive_same_tool_arg_hashes: llm_stall.consecutive_same_tool_arg_hashes,
                    total_successful_tool_calls: llm_counters.total_successful_tool_calls,
                    pending_external_action_ack: llm_directives.pending_external_action_ack,
                    heartbeat: &heartbeat,
                    empty_response_retry_pending: llm_recovery.empty_response_retry_pending,
                    empty_response_retry_note: llm_recovery.empty_response_retry_note,
                    identity_prefill_text: llm_directives.identity_prefill_text,
                    deferred_no_tool_streak: llm_counters.deferred_no_tool_streak,
                    tools_required_for_turn: llm_counters.tools_required_for_turn,
                    max_budget_extensions,
                    hard_token_cap,
                    truncated_text_prefix: llm_recovery.truncated_text_prefix,
                    provider_timeout_ms: llm_budget.provider_timeout_ms,
                    thinking_truncation_count: llm_recovery.thinking_truncation_count,
                    est_input_tokens,
                    build_ms: message_build_ms,
                },
            )
            .await?;
            let mut resp = match llm_outcome {
                LlmPhaseOutcome::ContinueLoop => {
                    if execution_state.execution_budget_applies() {
                        execution_state.record_llm_call();
                    }
                    // Propagate accumulated timeout to execution state so
                    // wall-clock budget excludes provider-caused delays.
                    execution_state.provider_timeout_ms = turn_state.budget.provider_timeout_ms();
                    continue;
                }
                LlmPhaseOutcome::Return(result) => {
                    if execution_state.execution_budget_applies() {
                        execution_state.record_llm_call();
                    }
                    return result;
                }
                LlmPhaseOutcome::Proceed(resp) => resp,
            };

            let response_stall = turn_state.stall.for_response_phase();
            let response_recovery = turn_state.recovery.for_response_phase();
            let response_evidence = turn_state.evidence.for_response_phase();
            let response_directives = turn_state.directives.for_response_phase();
            let response_counters = turn_state.counters.for_response_phase();
            let response_outcome = super::response_phase::run_response_phase(
                &services,
                &mut ResponsePhaseCtx {
                    resp: &mut resp,
                    emitter: &emitter,
                    task_id: &task_id,
                    session_id,
                    user_text,
                    iteration,
                    task_start,
                    task_tokens_used: turn_state.budget.task_tokens_used(),
                    learning_ctx: &mut learning_ctx,
                    pending_system_messages: response_directives.pending_system_messages,
                    tool_defs: &mut tool_defs,
                    base_tool_defs: &mut base_tool_defs,
                    available_capabilities: &mut available_capabilities,
                    policy_bundle: &mut policy_bundle,
                    tools_allowed_for_user,
                    is_personal_memory_recall_turn,
                    is_reaffirmation_challenge_turn,
                    requests_external_verification,
                    llm_provider: llm_provider.clone(),
                    llm_router: llm_router.clone(),
                    model: &mut model,
                    user_role,
                    channel_ctx: channel_ctx.clone(),
                    status_tx: status_tx.clone(),
                    total_successful_tool_calls: response_counters.total_successful_tool_calls,
                    stall_count: response_stall.stall_count,
                    consecutive_clean_iterations: response_stall.consecutive_clean_iterations,
                    deferred_no_tool_streak: response_counters.deferred_no_tool_streak,
                    deferred_no_tool_model_switches: response_counters
                        .deferred_no_tool_model_switches,
                    fallback_expanded_once: response_recovery.fallback_expanded_once,
                    empty_response_retry_used: response_recovery.empty_response_retry_used,
                    empty_response_retry_pending: response_recovery.empty_response_retry_pending,
                    empty_response_retry_note: response_recovery.empty_response_retry_note,
                    identity_prefill_text: response_directives.identity_prefill_text,
                    pending_background_ack: response_directives.pending_background_ack,
                    pending_external_action_ack: response_directives.pending_external_action_ack,
                    require_file_recheck_before_answer: response_evidence
                        .require_file_recheck_before_answer,
                    completion_progress: &mut completion_progress,
                    turn_context: &turn_context,
                    needs_tools_for_turn: response_counters.needs_tools_for_turn,
                    force_text_response: response_recovery.force_text_response,
                    execution_state: &mut execution_state,
                    validation_state: response_evidence.validation_state,
                },
            )
            .await?;
            match response_outcome {
                ResponsePhaseOutcome::ContinueLoop => {
                    if execution_state.execution_budget_applies() {
                        execution_state.record_llm_call();
                    }
                    continue;
                }
                ResponsePhaseOutcome::Return(result) => {
                    if execution_state.execution_budget_applies() {
                        execution_state.record_llm_call();
                    }
                    return result;
                }
                ResponsePhaseOutcome::ProceedToToolExecution => {
                    if !resp.tool_calls.is_empty() && !execution_state.execution_budget_applies() {
                        execution_state.activate_budget_envelope(
                            turn_state.budget.task_tokens_used(),
                            task_start.elapsed(),
                        );
                    }
                    if !resp.tool_calls.is_empty() || execution_state.execution_budget_applies() {
                        execution_state.record_llm_call();
                    }
                }
            }
            // === EXECUTE TOOL CALLS ===
            let tool_prelude_recovery = turn_state.recovery.for_tool_prelude_phase();
            let tool_prelude_evidence = turn_state.evidence.for_tool_prelude_phase();
            let tool_prelude_directives = turn_state.directives.for_tool_prelude_phase();
            let tool_prelude_outcome = super::tool_prelude_phase::run_tool_prelude_phase(
                &services,
                &mut ToolPreludeCtx {
                    resp: &resp,
                    emitter: &emitter,
                    task_id: &task_id,
                    session_id,
                    model: &model,
                    llm_provider: llm_provider.clone(),
                    iteration,
                    task_start,
                    learning_ctx: &mut learning_ctx,
                    evidence_state: tool_prelude_evidence.evidence_state,
                    user_text,
                    policy_bundle: &policy_bundle,
                    available_capabilities: &available_capabilities,
                    execution_state: &mut execution_state,
                    validation_state: tool_prelude_evidence.validation_state,
                    pending_system_messages: tool_prelude_directives.pending_system_messages,
                    force_text_response: tool_prelude_recovery.force_text_response,
                    turn_context: &turn_context,
                },
            )
            .await?;
            match tool_prelude_outcome {
                ToolPreludeOutcome::ContinueLoop => continue,
                ToolPreludeOutcome::Return(result) => return result,
                ToolPreludeOutcome::Proceed => {}
            }

            // Capture baseline for tracking tool calls per plan step
            let tool_calls_before_execution = learning_ctx.tool_calls.len();

            let tool_execution_stall = turn_state.stall.for_tool_execution_phase();
            let tool_execution_failures = turn_state.failures.for_tool_execution_phase();
            let tool_execution_recovery = turn_state.recovery.for_tool_execution_phase();
            let tool_execution_evidence = turn_state.evidence.for_tool_execution_phase();
            let tool_execution_reflection = turn_state.reflection.for_tool_execution_phase();
            let tool_execution_directives = turn_state.directives.for_tool_execution_phase();
            let tool_execution_counters = turn_state.counters.for_tool_execution_phase();
            let tool_execution_outcome = super::tool_execution_phase::run_tool_execution_phase(
                &services,
                &mut ToolExecutionCtx {
                    resp: &resp,
                    emitter: &emitter,
                    task_id: &task_id,
                    session_id,
                    iteration,
                    task_start,
                    learning_ctx: &mut learning_ctx,
                    task_tokens_used: turn_state.budget.task_tokens_used(),
                    user_text,
                    model: &model,
                    restrict_to_personal_memory_tools,
                    active_skill_names: &active_skill_names,
                    active_untrusted_external_reference_skills:
                        &active_untrusted_external_reference_skills,
                    restrict_untrusted_external_reference_tools,
                    is_reaffirmation_challenge_turn,
                    personal_memory_tool_call_cap,
                    base_tool_defs: &base_tool_defs,
                    available_capabilities: &available_capabilities,
                    policy_bundle: &policy_bundle,
                    status_tx: status_tx.clone(),
                    channel_ctx: &channel_ctx,
                    user_role,
                    heartbeat: &heartbeat,
                    tool_defs: &mut tool_defs,
                    total_tool_calls_attempted: tool_execution_counters.total_tool_calls_attempted,
                    total_successful_tool_calls: tool_execution_counters
                        .total_successful_tool_calls,
                    tool_failure_count: tool_execution_failures.tool_failure_count,
                    tool_failure_signatures: tool_execution_failures.tool_failure_signatures,
                    tool_transient_failure_count: tool_execution_failures
                        .tool_transient_failure_count,
                    tool_cooldown_until_iteration: tool_execution_failures
                        .tool_cooldown_until_iteration,
                    tool_call_count: tool_execution_counters.tool_call_count,
                    personal_memory_tool_calls: tool_execution_counters.personal_memory_tool_calls,
                    no_evidence_result_streak: tool_execution_evidence.no_evidence_result_streak,
                    no_evidence_tools_seen: tool_execution_evidence.no_evidence_tools_seen,
                    evidence_gain_count: tool_execution_evidence.evidence_gain_count,
                    pending_error_solution_ids: tool_execution_reflection
                        .pending_error_solution_ids,
                    tool_error_history: tool_execution_failures.tool_error_history,
                    reflection_completed: tool_execution_reflection.reflection_completed,
                    pending_reflection_recoveries: tool_execution_reflection
                        .pending_reflection_recoveries,
                    tool_failure_patterns: tool_execution_failures.tool_failure_patterns,
                    last_tool_failure: tool_execution_failures.last_tool_failure,
                    in_session_learned: tool_execution_reflection.in_session_learned,
                    unknown_tools: tool_execution_failures.unknown_tools,
                    recent_tool_calls: tool_execution_stall.recent_tool_calls,
                    consecutive_same_tool: tool_execution_stall.consecutive_same_tool,
                    consecutive_same_tool_arg_hashes: tool_execution_stall
                        .consecutive_same_tool_arg_hashes,
                    force_text_response: tool_execution_recovery.force_text_response,
                    pending_system_messages: tool_execution_directives.pending_system_messages,
                    recent_tool_names: tool_execution_stall.recent_tool_names,
                    successful_send_file_keys: tool_execution_counters.successful_send_file_keys,
                    cli_agent_boundary_injected: tool_execution_directives
                        .cli_agent_boundary_injected,
                    evidence_state: tool_execution_evidence.evidence_state,
                    pending_background_ack: tool_execution_directives.pending_background_ack,
                    pending_external_action_ack: tool_execution_directives
                        .pending_external_action_ack,
                    stall_count: tool_execution_stall.stall_count,
                    deferred_no_tool_streak: tool_execution_counters.deferred_no_tool_streak,
                    consecutive_clean_iterations: tool_execution_stall.consecutive_clean_iterations,
                    fallback_expanded_once: tool_execution_recovery.fallback_expanded_once,
                    known_project_dir: tool_execution_evidence.known_project_dir,
                    dirs_with_project_inspect_file_evidence: tool_execution_evidence
                        .dirs_with_project_inspect_file_evidence,
                    dirs_with_search_no_matches: tool_execution_evidence
                        .dirs_with_search_no_matches,
                    require_file_recheck_before_answer: tool_execution_evidence
                        .require_file_recheck_before_answer,
                    completion_progress: &mut completion_progress,
                    turn_context: &turn_context,
                    resolved_goal_id: resolved_goal_id.as_deref(),
                    tool_result_cache: tool_execution_counters.tool_result_cache,
                    execution_state: &mut execution_state,
                    validation_state: tool_execution_evidence.validation_state,
                    read_file_tracker: &mut turn_state.read_files,
                    correction: correction_context.clone(),
                },
            )
            .await?;
            match tool_execution_outcome {
                ToolExecutionOutcome::Return(result) => return result,
                ToolExecutionOutcome::NextIteration => {}
            }

            // Re-planner: track tool calls on current plan step and evaluate
            // whether the step is complete. Uses delta of learning_ctx.tool_calls
            // to count ALL calls this round (including failures).
            {
                let tool_calls_this_round = learning_ctx
                    .tool_calls
                    .len()
                    .saturating_sub(tool_calls_before_execution);
                if let Some(ref mut plan) = execution_state.active_linear_intent_plan {
                    plan.record_tool_calls_on_current(tool_calls_this_round);
                }
                if let Some(ref mut plan) = execution_state.active_linear_intent_plan {
                    if plan.current_step_needs_replan() {
                        plan.mark_current_step_evaluated();
                        if let Some(step) = plan.steps.get(plan.current_step_cursor).cloned() {
                            use super::bootstrap_phase::task_planning::{
                                evaluate_step_completion, summarize_tool_calls_for_replan,
                            };
                            let tool_summary =
                                summarize_tool_calls_for_replan(&learning_ctx.tool_calls, 8);
                            if let Some(ref router) = llm_router {
                                self.emit_decision_point(
                                    &emitter,
                                    &task_id,
                                    iteration,
                                    crate::events::DecisionType::HandHoldingTelemetry,
                                    "Replanner attempted step evaluation".to_string(),
                                    super::hand_holding_telemetry::replanner_result_metadata(
                                        "attempted",
                                        &model,
                                        self.trust_tier_for_model(&model).as_str(),
                                        step.step_index,
                                        &step.description,
                                        false,
                                        None,
                                    ),
                                )
                                .await;
                                if let Some(evidence) = evaluate_step_completion(
                                    llm_provider.clone(),
                                    router,
                                    &step.description,
                                    &tool_summary,
                                    Some(
                                        super::bootstrap_phase::task_planning::PlannerTelemetryCtx {
                                            emitter: &emitter,
                                            state: self.state.as_ref(),
                                            session_id,
                                            task_id: &task_id,
                                        },
                                    ),
                                )
                                .await
                                {
                                    self.emit_decision_point(
                                        &emitter,
                                        &task_id,
                                        iteration,
                                        crate::events::DecisionType::HandHoldingTelemetry,
                                        "Replanner advanced current step".to_string(),
                                        super::hand_holding_telemetry::replanner_result_metadata(
                                            "advanced",
                                            &model,
                                            self.trust_tier_for_model(&model).as_str(),
                                            step.step_index,
                                            &step.description,
                                            true,
                                            Some(&evidence),
                                        ),
                                    )
                                    .await;
                                    if let Some(ref mut plan) =
                                        execution_state.active_linear_intent_plan
                                    {
                                        plan.complete_current_step_with_evidence(evidence);
                                        info!(
                                            session_id,
                                            completed_step = plan.current_step_cursor - 1,
                                            "Re-planner advanced plan to next step"
                                        );
                                    }
                                } else {
                                    self.emit_decision_point(
                                        &emitter,
                                        &task_id,
                                        iteration,
                                        crate::events::DecisionType::HandHoldingTelemetry,
                                        "Replanner did not advance current step".to_string(),
                                        super::hand_holding_telemetry::replanner_result_metadata(
                                            "not_advanced",
                                            &model,
                                            self.trust_tier_for_model(&model).as_str(),
                                            step.step_index,
                                            &step.description,
                                            false,
                                            None,
                                        ),
                                    )
                                    .await;
                                }
                            }
                        }
                    }
                }
            }
        }
        }, turn_span)
        .await
    }
}

#[cfg(test)]
#[path = "characterization_tests.rs"]
mod characterization_tests;

#[cfg(test)]
mod cancel_intent_tests {
    use super::*;

    #[test]
    fn negated_stop_not_detected() {
        // "Do not stop until every test passes" should NOT trigger cancel
        assert!(!cancel_keyword_not_negated(
            "do not stop until every test passes",
            "stop"
        ));
    }

    #[test]
    fn negated_cancel_not_detected() {
        assert!(!cancel_keyword_not_negated(
            "don't cancel anything",
            "cancel"
        ));
    }

    #[test]
    fn bare_stop_detected() {
        assert!(cancel_keyword_not_negated("stop", "stop"));
    }

    #[test]
    fn bare_cancel_detected() {
        assert!(cancel_keyword_not_negated("cancel everything", "cancel"));
    }

    #[test]
    fn never_stop_not_detected() {
        assert!(!cancel_keyword_not_negated("never stop working", "stop"));
    }

    #[test]
    fn long_message_with_stop_no_cancel() {
        let msg = "write a script that does X and Y. do not stop until every test passes.";
        let intent = infer_deterministic_orchestration_intent(msg);
        assert!(!intent.cancel_intent.unwrap_or(false));
    }

    #[test]
    fn short_stop_message_triggers_cancel() {
        let intent = infer_deterministic_orchestration_intent("stop");
        assert!(intent.cancel_intent.unwrap_or(false));
    }

    #[test]
    fn short_cancel_message_triggers_cancel() {
        let intent = infer_deterministic_orchestration_intent("cancel all");
        assert!(intent.cancel_intent.unwrap_or(false));
    }

    #[test]
    fn never_mind_in_long_message() {
        // Multi-word phrases are always unambiguous
        let intent = infer_deterministic_orchestration_intent(
            "actually never mind about that whole thing I asked earlier, let me think about it",
        );
        assert!(intent.cancel_intent.unwrap_or(false));
    }
}

#[cfg(test)]
mod stuck_fallback_tests {
    use super::*;

    #[test]
    fn bare_web_search_asks_for_query() {
        // The exact case that produced the misleading "processing limit" reply:
        // a query-less "Web search" command the model could not act on.
        let msg = build_stuck_no_output_fallback("Web search");
        assert!(msg.contains("what would you like me to look up") || msg.contains("search for"));
        assert!(!msg.contains("processing limit"));
    }

    #[test]
    fn bare_look_it_up_asks_for_query() {
        let msg = build_stuck_no_output_fallback("look it up");
        assert!(msg.to_lowercase().contains("look up"));
    }

    #[test]
    fn detailed_search_query_does_not_get_clarifying_prompt() {
        // A request with an actual subject should not be treated as query-less;
        // it falls through to the generic rephrase message.
        let msg = build_stuck_no_output_fallback(
            "search the web for what Caro is a nickname for and summarize the top results",
        );
        assert!(msg.contains("rephrase"));
    }

    #[test]
    fn non_search_request_falls_through_to_generic() {
        let msg = build_stuck_no_output_fallback("build me a dashboard");
        assert!(msg.contains("rephrase"));
        assert!(!msg.contains("processing limit"));
    }

    #[test]
    fn fallback_never_emits_legacy_processing_limit_string() {
        for input in ["Web search", "do something", "", "look up cats"] {
            assert!(!build_stuck_no_output_fallback(input).contains("processing limit"));
        }
    }
}
