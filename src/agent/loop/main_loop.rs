use super::bootstrap_phase::{BootstrapCtx, BootstrapData, BootstrapOutcome};
use super::llm_phase::LlmPhaseCtx;
use super::message_build_phase::{MessageBuildCtx, MessageBuildData};
use super::response_phase::ResponsePhaseCtx;
use super::stopping_phase::StoppingPhaseCtx;
use super::tool_execution_phase::ToolExecutionCtx;
use super::tool_prelude_phase::ToolPreludeCtx;
use super::turn_transition::{TurnRestartReason, TurnTransition};
use super::*;
use crate::events::TaskOutcome;

/// Build a user-facing message when the force-text safety net fires and there
/// is no salvageable tool output to return.
///
/// This path runs only after automatic force-text recovery is exhausted. It
/// must not guess from wording that the user should restate an otherwise valid
/// request or choose a routine execution tactic.
fn build_stuck_no_output_fallback(_user_text: &str) -> String {
    "I wasn't able to complete that because automatic execution recovery produced no usable result."
        .to_string()
}

/// Build assessment context only when the dialogue lifecycle says the current
/// request depends on an earlier exchange. New tasks already carry their full
/// authored request separately; feeding unrelated history into their hard
/// completion contract can manufacture stale observation or mutation duties.
fn task_assessment_conversation_context(
    followup_mode: Option<FollowupMode>,
    session_summary: Option<&str>,
    recent_messages: &[Value],
) -> Option<String> {
    if !matches!(
        followup_mode,
        Some(FollowupMode::Followup | FollowupMode::ClarificationAnswer)
    ) {
        return None;
    }

    let mut ctx_parts = Vec::new();
    if let Some(summary) = session_summary.filter(|summary| !summary.is_empty()) {
        ctx_parts.push(format!("[Session Summary] {summary}"));
    }
    for msg in recent_messages {
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
    (!ctx_parts.is_empty()).then(|| ctx_parts.join("\n"))
}

/// Enter one agent-loop iteration and update the state that is common to every
/// phase path. Keeping this boundary small avoids threading the full turn
/// context through another driver object while making iteration entry explicit.
async fn begin_turn_iteration(
    task_id: &str,
    model: &mut String,
    turn_state: &mut super::loop_state::TurnState,
    execution_state: &ExecutionState,
    completion_progress: &CompletionProgress,
    heartbeat: &Option<Arc<AtomicU64>>,
) -> usize {
    let iteration = turn_state.counters.advance_iteration();
    touch_heartbeat(heartbeat);
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
        let resolved_model =
            crate::agent::computer_use::resolve_model_for_task(task_id, model.as_str()).await;
        *model = resolved_model;
    }
    #[cfg(not(feature = "computer_use"))]
    let _ = (task_id, model);

    turn_state
        .with_harness_eval(|eval| {
            eval.record_completion_progress(completion_progress);
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
    iteration
}

/// Apply restart-only state changes at one shared driver boundary. Most
/// restarts merely select the loop back-edge; approach pivots additionally
/// reset stale evidence and inject the next-attempt directive.
fn prepare_turn_restart(
    agent: &Agent,
    reason: TurnRestartReason,
    turn_state: &mut super::loop_state::TurnState,
    approach_pivots_used: &mut usize,
    model: &str,
) {
    let TurnRestartReason::ApproachPivot { failure_record } = reason else {
        return;
    };

    *approach_pivots_used += 1;
    turn_state.stall.reset_for_pivot();
    turn_state.failures.reset_for_pivot();
    turn_state
        .directives
        .push_system_message(SystemDirective::ApproachPivotRequired {
            attempt: *approach_pivots_used,
            failure_record,
        });
    crate::agent::heuristic_telemetry::global().record(
        "approach_pivot",
        model,
        agent.trust_tier_for_model(model),
        crate::agent::heuristic_telemetry::HeuristicAction::Enforced,
    );
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
        internal_continuation: bool,
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
                internal_continuation,
            },
        )
        .await?;
        let BootstrapData {
            user_text: canonical_user_text,
            task_id,
            resume_execution_snapshot,
            emitter,
            mut learning_ctx,
            is_reaffirmation_challenge_turn,
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
            mut turn_context,
            mut project_instruction_tracker,
            core_prompt_bytes,
            mut task_context_tail,
            session_summary,
            mut harness_eval,
        } = match bootstrap_outcome {
            BootstrapOutcome::Return(result) => return result,
            BootstrapOutcome::Continue(data) => *data,
        };
        // Bootstrap may enrich the durable user message with an STT fallback.
        // From this point onward, use exactly what was persisted so matching,
        // rendering, routing, and dialogue projection share one source of truth.
        let user_text = canonical_user_text.as_str();
        let followup_mode = turn_context
            .followup_mode
            .map(|mode| mode.as_str())
            .unwrap_or("unknown");
        let turn_context_reasons: Vec<&'static str> = turn_context
            .reasons
            .iter()
            .map(|reason| reason.as_code())
            .collect();
        let dialogue_state = self
            .state
            .get_dialogue_state(session_id)
            .await
            .ok()
            .flatten();
        // Keep the persisted user message byte-for-byte intact. Historically a
        // follow-up was rewritten into a second synthetic user message, leaving
        // both the stored raw message and the rewritten copy in the provider
        // transcript. Continuity now comes from normal transcript adjacency.
        let llm_user_text = canonical_user_text.clone();
        // Conversation continuity is topological, not phrase-classified: every
        // user turn keeps the immediately preceding assistant exchange intact.
        // Whether the wording continues that topic or starts a new task remains
        // for the model to interpret from the raw transcript.
        let prior_assistant_message_id = dialogue_state
            .as_ref()
            .and_then(|state| state.last_assistant_turn.as_ref())
            .map(|turn| turn.message_id.clone());
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
            counters: super::loop_state::LoopCounters::default(),
            read_files: super::loop_state::ReadFileObservationTracker::default(),
            harness_eval: None,
            eval: None,
        };
        // Task-start semantic assessment is authoritative for language-derived
        // completion obligations. Autonomous models receive classification
        // without step scaffolding and retain control of their approach.
        // In tests, MockProvider silently intercepts assessment calls.
        let mut semantic_contract_applied = false;
        let task_plan = {
            use super::bootstrap_phase::task_planning::{
                generate_task_plan, planned_contract_is_complete, planned_contract_is_confident,
                planned_mutation_constraints_are_grounded, planned_tool_constraints_are_grounded,
                planned_response_fields_are_grounded, planning_skip_reason, TaskAssessmentMode,
            };
            let model_trust_tier = self.trust_tier_for_model(&model);
            let planner_trust_tier = model_trust_tier.as_str();
            let assessment_mode = match model_trust_tier {
                crate::agent::trust_tier::ModelTrustTier::Guided => {
                    TaskAssessmentMode::GuidedPlan
                }
                crate::agent::trust_tier::ModelTrustTier::Autonomous => {
                    TaskAssessmentMode::AutonomousRouting
                }
            };
            let assessment_decision_type = match assessment_mode {
                TaskAssessmentMode::GuidedPlan => {
                    crate::events::DecisionType::HandHoldingTelemetry
                }
                TaskAssessmentMode::AutonomousRouting => crate::events::DecisionType::IntentGate,
            };
            let planner_skip_reason = if self.mandate_execution.is_some() {
                Some("mandate_cycle_uses_only_budgeted_main_loop_calls")
            } else {
                planning_skip_reason(user_text, false)
            };
            if let Some(reason) = planner_skip_reason {
                self.emit_decision_point(
                    &emitter,
                    &task_id,
                    0,
                    assessment_decision_type,
                    "Task assessment skipped".to_string(),
                    super::hand_holding_telemetry::planner_skip_metadata(
                        reason,
                        &model,
                        planner_trust_tier,
                    ),
                )
                .await;
                None
            } else {
                // Preserve prior narrative for genuine multi-hop follow-ups,
                // but keep a lifecycle-classified new request contract-local.
                let planner_context = task_assessment_conversation_context(
                    turn_context.followup_mode,
                    session_summary.as_ref().map(|summary| summary.summary.as_str()),
                    &turn_context.recent_messages,
                );
                let planner_model = llm_router
                    .as_ref()
                    .map(|router| router.select(crate::router::Tier::Primary))
                    .unwrap_or(model.as_str());
                self.emit_decision_point(
                    &emitter,
                    &task_id,
                    0,
                    assessment_decision_type,
                    "Task assessment attempted".to_string(),
                    super::hand_holding_telemetry::planner_result_metadata(
                        "attempted",
                        planner_model,
                        planner_trust_tier,
                        super::hand_holding_telemetry::PlannerResultStats::empty(),
                        None,
                    ),
                )
                .await;
                // This call classifies obligations only; autonomous models
                // still choose their own execution approach.
                let plan_opt = generate_task_plan(
                    llm_provider.clone(),
                    planner_model,
                    user_text,
                    planner_context.as_deref(),
                    assessment_mode,
                    Some(super::bootstrap_phase::task_planning::PlannerTelemetryCtx {
                        emitter: &emitter,
                        state: self.state.as_ref(),
                        session_id,
                        task_id: &task_id,
                    }),
                )
                .await;
                if let Some(ref plan) = plan_opt {
                    let before_contract = turn_context.completion_contract.clone();
                    if let Some(shape) = plan.task_shape.as_ref().filter(|shape| {
                        matches!(
                            shape
                                .confidence
                                .as_deref()
                                .map(|value| value.trim().to_ascii_lowercase()),
                            Some(value) if matches!(value.as_str(), "medium" | "high")
                        )
                    }) {
                        turn_context.continue_inline_after_background_start = shape
                            .continue_inline_after_background_start
                            .unwrap_or(false);
                        if let (Some(relationship), Some(semantic_scope)) = (
                            shape.request_relationship.as_deref(),
                            shape.semantic_scope.as_deref(),
                        ) {
                            match crate::agent::dialogue_state::record_dialogue_semantic_user_turn(
                                self,
                                session_id,
                                user_text,
                                relationship,
                                semantic_scope,
                            )
                            .await
                            {
                                Ok(Some(crate::traits::UserTurnKind::NewRequest)) => {
                                    turn_context.followup_mode = Some(
                                        crate::agent::followup::FollowupMode::NewTask,
                                    );
                                    if plan
                                        .contract
                                        .as_ref()
                                        .and_then(|contract| contract.project_reference.as_deref())
                                        .is_none()
                                        && !crate::agent::user_text_references_filesystem_path(
                                            user_text,
                                        )
                                    {
                                        turn_context.primary_project_scope = None;
                                    }
                                }
                                Ok(Some(crate::traits::UserTurnKind::Followup)) => {
                                    turn_context.followup_mode =
                                        Some(crate::agent::followup::FollowupMode::Followup);
                                }
                                Ok(Some(crate::traits::UserTurnKind::ClarificationAnswer)) => {
                                    turn_context.followup_mode = Some(
                                        crate::agent::followup::FollowupMode::ClarificationAnswer,
                                    );
                                }
                                Ok(_) => {}
                                Err(error) => warn!(
                                    session_id,
                                    %error,
                                    "Failed to persist semantic dialogue classification"
                                ),
                            }
                        }
                    }
                    // Install only a complete, grounded semantic contract.
                    // Partial output cannot refine a hard decision.
                    if let Some(ref signals) = plan.contract {
                        let scope = signals
                            .mutation_scope
                            .as_deref()
                            .map(|value| value.trim().to_ascii_lowercase())
                            .unwrap_or_default();
                        let declares_negative_scope =
                            matches!(scope.as_str(), "read_only" | "read-only" | "scoped");
                        let forbids_tool_use = signals
                            .tool_scope
                            .as_deref()
                            .is_some_and(|scope| scope.trim().eq_ignore_ascii_case("forbidden"));
                        let has_tool_constraints =
                            forbids_tool_use || !signals.forbidden_tool_scopes.is_empty();
                        let confident = planned_contract_is_confident(
                            signals,
                            plan.task_shape.as_ref(),
                        );
                        let complete = planned_contract_is_complete(signals);
                        let grounded = planned_mutation_constraints_are_grounded(
                            signals,
                            user_text,
                        );
                        let tool_constraint_grounded =
                            planned_tool_constraints_are_grounded(signals, user_text);
                        let response_fields_grounded =
                            planned_response_fields_are_grounded(signals, user_text);

                        if confident
                            && complete
                            && (!declares_negative_scope || grounded)
                            && (!has_tool_constraints || tool_constraint_grounded)
                            && response_fields_grounded
                        {
                            let planned_kind = signals
                                .task_kind
                                .as_deref()
                                .and_then(crate::agent::parse_planned_task_kind)
                                .expect("complete semantic contract has a valid task kind");
                            let forbidden_actions = signals
                                .forbidden_actions
                                .iter()
                                .filter_map(|action| {
                                    crate::agent::parse_planned_forbidden_action(action)
                                })
                                .collect::<Vec<_>>();
                            let required_effects = if signals.expects_mutation == Some(true) {
                                crate::agent::parse_planned_mutation_effects(
                                    signals
                                        .required_effects
                                        .as_deref()
                                        .expect("complete semantic contract has effects"),
                                )
                                .expect("complete semantic contract has valid effects")
                            } else {
                                crate::traits::ToolMutationEffects::NONE
                            };
                            crate::agent::install_semantic_completion_contract(
                                &mut turn_context.completion_contract,
                                crate::agent::SemanticCompletionRequirements {
                                    expects_mutation: signals
                                        .expects_mutation
                                        .expect("complete semantic contract"),
                                    requires_observation: signals
                                        .requires_observation
                                        .expect("complete semantic contract"),
                                    task_kind: planned_kind,
                                    required_mutation_effects: required_effects,
                                    mutation_scope: signals
                                        .mutation_scope
                                        .as_deref()
                                        .unwrap_or("allowed"),
                                    forbidden_actions: &forbidden_actions,
                                    minimum_sources: signals.minimum_sources.unwrap_or_default()
                                        as usize,
                                    requires_primary_sources: signals
                                        .requires_primary_sources
                                        .unwrap_or(false),
                                    requires_exact_history: signals
                                        .requires_exact_history
                                        .unwrap_or(false),
                                    evidence_requirements: signals
                                        .evidence_requirements
                                        .as_deref()
                                        .unwrap_or_default(),
                                    forbids_tool_use,
                                    forbidden_tool_scopes: &signals.forbidden_tool_scopes,
                                    required_response_fields: &signals.required_response_fields,
                                },
                            );
                            if turn_context.inherited_completion_contract
                                && turn_context.followup_mode
                                    != Some(crate::agent::followup::FollowupMode::NewTask)
                            {
                                turn_context.completion_contract =
                                    crate::agent::inherit_unfinished_request_contract(
                                        turn_context.completion_contract.clone(),
                                        &before_contract,
                                    );
                            }
                            if let Some(reference) = signals
                                .project_reference
                                .as_deref()
                                .map(str::trim)
                                .filter(|reference| !reference.is_empty())
                                .filter(|reference| {
                                    user_text
                                        .to_ascii_lowercase()
                                        .contains(&reference.to_ascii_lowercase())
                                })
                            {
                                if let Some(scope) =
                                    crate::tools::fs_utils::resolve_project_scope_reference(
                                        reference,
                                        &self.path_aliases.projects,
                                    )
                                {
                                    turn_context.primary_project_scope =
                                        Some(scope.to_string_lossy().to_string());
                                }
                            }
                            semantic_contract_applied = true;
                            if let Err(error) = crate::agent::dialogue_state::record_dialogue_completion_contract(
                                self,
                                session_id,
                                user_text,
                                &turn_context.completion_contract,
                            )
                            .await
                            {
                                warn!(
                                    session_id,
                                    %error,
                                    "Failed to persist semantic completion contract"
                                );
                            }
                        } else {
                            warn!(
                                session_id,
                                confident,
                                complete,
                                grounded,
                                mutation_scope = %scope,
                                "Ignored incomplete or untrusted semantic contract"
                            );
                        }
                        if turn_context.completion_contract != before_contract {
                            info!(
                                session_id,
                                task_kind = ?turn_context.completion_contract.task_kind,
                                expects_mutation =
                                    turn_context.completion_contract.expects_mutation,
                                requires_observation =
                                    turn_context.completion_contract.requires_observation,
                                "Installed semantic completion contract"
                            );
                        }
                    }
                    let contract_changed = turn_context.completion_contract != before_contract;
                    self.emit_decision_point(
                        &emitter,
                        &task_id,
                        0,
                        assessment_decision_type,
                        "Task assessment succeeded".to_string(),
                        {
                            let mut metadata =
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
                                );
                            metadata["assessment_mode"] = json!(plan.mode.as_str());
                            metadata["task_shape"] = json!(plan.task_shape.as_ref());
                            metadata["completion_contract"] = json!({
                                "task_kind": format!(
                                    "{:?}",
                                    turn_context.completion_contract.task_kind
                                )
                                .to_ascii_lowercase(),
                                "expects_mutation": turn_context
                                    .completion_contract
                                    .expects_mutation,
                                "required_mutation_effects": turn_context
                                    .completion_contract
                                    .required_mutation_effects,
                                "requires_observation": turn_context
                                    .completion_contract
                                    .requires_observation,
                                "requires_reverification_after_mutation": turn_context
                                    .completion_contract
                                    .requires_reverification_after_mutation,
                                "explicit_verification_requested": turn_context
                                    .completion_contract
                                    .explicit_verification_requested,
                                "evidence_requirements": turn_context
                                    .completion_contract
                                    .evidence_requirements,
                                "forbidden_actions": turn_context
                                    .completion_contract
                                    .forbidden_mutation_actions
                                    .iter()
                                    .map(|action| action.as_str())
                                    .collect::<Vec<_>>(),
                                "forbidden_tool_scopes": turn_context
                                    .completion_contract
                                    .forbidden_tool_scopes,
                                "required_response_fields": turn_context
                                    .completion_contract
                                    .required_response_fields,
                            });
                            metadata
                        },
                    )
                    .await;
                } else {
                    self.emit_decision_point(
                        &emitter,
                        &task_id,
                        0,
                        assessment_decision_type,
                        "Task assessment returned no result".to_string(),
                        super::hand_holding_telemetry::planner_result_metadata(
                            "no_plan",
                            &model,
                            planner_trust_tier,
                            super::hand_holding_telemetry::PlannerResultStats::empty(),
                            Some("assessment_returned_none"),
                        ),
                    )
                    .await;
                }
                plan_opt
            }
        };

        if self.mandate_execution.is_none()
            && !semantic_contract_applied
            && !turn_context.inherited_completion_contract
        {
            crate::agent::retain_structural_completion_contract(
                &mut turn_context.completion_contract,
            );
        }

        // Derive all contract-dependent state exactly once from that finalized
        // value so loop control, progress tracking, budgets, and telemetry agree.
        let mut completion_progress = CompletionProgress::new(&turn_context.completion_contract);
        if turn_context.completion_contract.forbids_tool_use {
            tool_defs.clear();
            let outstanding_needs = turn_context
                .completion_contract
                .evidence_requirements
                .iter()
                .map(crate::agent::inquiry::describe_requirement)
                .collect();
            turn_state.directives.push_system_message(
                SystemDirective::ToolUseForbiddenByRequest { outstanding_needs },
            );
        } else if !turn_context
            .completion_contract
            .forbidden_tool_scopes
            .is_empty()
        {
            let forbidden_scopes = &turn_context.completion_contract.forbidden_tool_scopes;
            tool_defs.retain(|definition| {
                let Some(name) = definition
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(serde_json::Value::as_str)
                else {
                    return false;
                };
                let scope = self
                    .tools
                    .iter()
                    .find(|tool| tool.name() == name && tool.is_available())
                    .and_then(|tool| {
                        tool.semantic_affordances()
                            .map(|affordances| affordances.scope)
                    })
                    .or_else(|| {
                        super::tool_execution_phase::fallback_tool_semantic_scope(name)
                    });
                !scope.is_some_and(|scope| forbidden_scopes.contains(&scope))
            });
        }
        let epistemic_uncertainty = crate::agent::inquiry::epistemic_uncertainty(
            &turn_context.completion_contract.evidence_requirements,
        );
        if policy_bundle.apply_epistemic_uncertainty(epistemic_uncertainty) {
            let guided_model = self.trust_tier_for_model(&model)
                == crate::agent::trust_tier::ModelTrustTier::Guided;
            let model_override_active = if epistemic_uncertainty >= 0.55 && guided_model {
                tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    self.model_override.read(),
                )
                .await
                .map(|guard| *guard)
                .unwrap_or(true)
            } else {
                true
            };
            if epistemic_uncertainty >= 0.55 && guided_model && !model_override_active {
                if let Some(router) = llm_router.as_ref() {
                    model = router
                        .select_for_profile(policy_bundle.policy.model_profile)
                        .to_string();
                }
            }
            if epistemic_uncertainty >= 0.55
                && guided_model
                && self.policy_config.tool_filter_enforce
            {
                tool_defs = self.filter_tool_definitions_for_policy(
                    &base_tool_defs,
                    &available_capabilities,
                    &policy_bundle.policy,
                    policy_bundle.risk_score,
                    false,
                );
            }
            self.emit_decision_point(
                &emitter,
                &task_id,
                0,
                DecisionType::ExecutionPlanningGate,
                "Applied evidence-derived epistemic uncertainty".to_string(),
                json!({
                    "condition": "epistemic_uncertainty_applied",
                    "uncertainty_score": policy_bundle.uncertainty_score,
                    "evidence_requirement_count": turn_context.completion_contract.evidence_requirements.len(),
                    "verify_level": policy_bundle.policy.verify_level,
                    "tool_budget": policy_bundle.policy.tool_budget,
                    "visible_tool_count": tool_defs.len(),
                    "selected_model": model,
                }),
            )
            .await;
        }
        if !turn_context.completion_contract.forbids_tool_use
            && !turn_context
            .completion_contract
            .evidence_requirements
            .is_empty()
        {
            let candidate_tools = crate::agent::inquiry::candidate_tools_for_requirements(
                &turn_context.completion_contract.evidence_requirements,
                tool_defs.iter().filter_map(|definition| {
                    definition
                        .get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(serde_json::Value::as_str)
                }),
            );
            let outstanding_needs = turn_context
                .completion_contract
                .evidence_requirements
                .iter()
                .map(crate::agent::inquiry::describe_requirement)
                .collect();
            turn_state.directives.push_system_message(
                SystemDirective::InquiryEvidenceRequired {
                    outstanding_needs,
                    candidate_tools,
                },
            );
        }
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
        if let Some(handle) = harness_eval_handle {
            turn_state.attach_harness_eval(handle);
        }

        let execution_requirement =
            ExecutionRequirement::from_finalized_contract(&turn_context.completion_contract);

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
        if let Some(snapshot) = resume_execution_snapshot {
            execution_state.execution_id = snapshot.execution_id;
            execution_state.last_outcome = snapshot.last_outcome;
            execution_state.background_handoff_active = snapshot.background_handoff_active;
            execution_state.persistence = ExecutionPersistence::Durable;
        }
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

        if let Some(plan) = task_plan.filter(|plan| {
            plan.mode.includes_step_plan() && !plan.steps.is_empty()
        }) {
            use crate::agent::execution_state::LinearIntentStep;
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
        }

        if route_failsafe_active {
            turn_state
                .directives
                .push_system_message(SystemDirective::RouteFailsafeActive);
        }
        if let Some(mandate_id) = dialogue_state
            .as_ref()
            .and_then(|state| state.last_closed_question.as_ref())
            .filter(|question| question.kind == crate::traits::QuestionKind::MandateInput)
            .and_then(|question| question.mandate_id.as_ref())
            .filter(|mandate_id| !mandate_id.is_empty())
        {
            turn_state.directives.push_system_message(
                SystemDirective::MandateOwnerInputInspectionRequired {
                    mandate_id: mandate_id.clone(),
                },
            );
        }
        let has_recent_tool_context = turn_context
            .recent_messages
            .iter()
            .any(|row| row.get("role").and_then(|v| v.as_str()) == Some("tool"));
        if looks_like_evidence_grounding_challenge(user_text)
            && (turn_context.followup_mode != Some(FollowupMode::NewTask)
                || has_recent_tool_context)
            && self
                .supervision_gate_enforced(
                    "evidence_challenge_directive",
                    &model,
                    &emitter,
                    &task_id,
                    0,
                )
                .await
        {
            turn_state
                .directives
                .push_system_message(SystemDirective::EvidenceGroundingRequired);
        }
        // Only pin the model to the prior exchange for genuinely vague
        // challenges ("Are you sure?") — compound/new-task messages that merely
        // contain a challenge keyword must not be anchored away from their
        // actual request.
        if is_reaffirmation_challenge_turn
            && crate::agent::recall_guardrails::is_vague_reaffirmation_challenge(user_text)
            && self
                .supervision_gate_enforced(
                    "reaffirmation_anchor_directive",
                    &model,
                    &emitter,
                    &task_id,
                    0,
                )
                .await
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
        else if crate::agent::recall_guardrails::looks_like_pronoun_referent_followup(user_text)
            && self
                .supervision_gate_enforced(
                    "coreference_grounding_directive",
                    &model,
                    &emitter,
                    &task_id,
                    0,
                )
                .await
        {
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
        if let Some(known_project_dir) = turn_context.primary_project_scope.clone() {
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
            SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS
        } else {
            MAX_BUDGET_EXTENSIONS
        };
        let hard_token_cap = if is_scheduled_goal {
            SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP
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
            let iteration = begin_turn_iteration(
                &task_id,
                &mut model,
                &mut turn_state,
                &execution_state,
                &completion_progress,
                &heartbeat,
            )
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

            // An unfulfilled Change/Deliver contract must retain execution
            // capability. Recovery branches may request force-text for generic
            // stall control, so clamp that shared state at the loop boundary.
            if turn_state.recovery.force_text_response()
                && !completion_contract_allows_force_text(
                    &turn_context.completion_contract,
                    &completion_progress,
                )
            {
                turn_state.recovery.set_force_text_response(false);
                turn_state.recovery.reset_force_text_iterations();
                turn_state
                    .directives
                    .for_message_build_phase()
                    .pending_system_messages
                    .push(SystemDirective::DeferredToolCallRequired);
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
                    last_failure_class: stopping_failures.last_failure_class,
                    empty_response_retry_pending: stopping_recovery
                        .empty_response_retry_pending,
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
            match stopping_outcome.into_turn_transition() {
                TurnTransition::Restart(reason) => {
                    prepare_turn_restart(
                        self,
                        reason,
                        &mut turn_state,
                        &mut approach_pivots_used,
                        &model,
                    );
                    continue;
                }
                TurnTransition::Finish(result) => return result,
                TurnTransition::Advance(()) => {}
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
                    prior_assistant_message_id: prior_assistant_message_id.as_deref(),
                    summary_last_message_id: session_summary
                        .as_ref()
                        .filter(|summary| summary.last_turn_seq.is_some())
                        .map(|summary| summary.last_message_id.as_str()),
                    tool_defs: &tool_defs,
                    policy_bundle: &policy_bundle,
                    pending_system_messages: message_build_directives.pending_system_messages,
                    empty_response_retry_pending: message_build_recovery
                        .empty_response_retry_pending,
                    redact_archived_shared_context: user_role != UserRole::Owner
                        && matches!(
                            channel_ctx.visibility,
                            ChannelVisibility::PrivateGroup
                                | ChannelVisibility::Public
                                | ChannelVisibility::PublicExternal
                        ),
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
                    execution_requirement: &execution_requirement,
                    completion_contract: &turn_context.completion_contract,
                    completion_progress: &completion_progress,
                    force_text_allowed: completion_contract_allows_force_text(
                        &turn_context.completion_contract,
                        &completion_progress,
                    ),
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
            let mut resp = match llm_outcome.into_turn_transition() {
                TurnTransition::Restart(reason) => {
                    if execution_state.execution_budget_applies() {
                        execution_state.record_llm_call();
                    }
                    // Propagate accumulated timeout to execution state so
                    // wall-clock budget excludes provider-caused delays.
                    execution_state.provider_timeout_ms = turn_state.budget.provider_timeout_ms();
                    prepare_turn_restart(
                        self,
                        reason,
                        &mut turn_state,
                        &mut approach_pivots_used,
                        &model,
                    );
                    continue;
                }
                TurnTransition::Finish(result) => {
                    if execution_state.execution_budget_applies() {
                        execution_state.record_llm_call();
                    }
                    return result;
                }
                TurnTransition::Advance(resp) => resp,
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
                    execution_requirement: &execution_requirement,
                    force_text_response: response_recovery.force_text_response,
                    execution_state: &mut execution_state,
                    validation_state: response_evidence.validation_state,
                },
            )
            .await?;
            match response_outcome.into_response_transition() {
                TurnTransition::Restart(reason) => {
                    if execution_state.execution_budget_applies() {
                        execution_state.record_llm_call();
                    }
                    prepare_turn_restart(
                        self,
                        reason,
                        &mut turn_state,
                        &mut approach_pivots_used,
                        &model,
                    );
                    continue;
                }
                TurnTransition::Finish(result) => {
                    if execution_state.execution_budget_applies() {
                        execution_state.record_llm_call();
                    }
                    return result;
                }
                TurnTransition::Advance(()) => {
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
                    project_instruction_tracker: &mut project_instruction_tracker,
                    task_context_tail: &mut task_context_tail,
                },
            )
            .await?;
            match tool_prelude_outcome.into_turn_transition() {
                TurnTransition::Restart(reason) => {
                    prepare_turn_restart(
                        self,
                        reason,
                        &mut turn_state,
                        &mut approach_pivots_used,
                        &model,
                    );
                    continue;
                }
                TurnTransition::Finish(result) => return result,
                TurnTransition::Advance(()) => {}
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
                    last_failure_class: tool_execution_failures.last_failure_class,
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
                    project_instruction_tracker: &mut project_instruction_tracker,
                    task_context_tail: &mut task_context_tail,
                    resolved_goal_id: resolved_goal_id.as_deref(),
                    is_scheduled_goal,
                    tool_result_cache: tool_execution_counters.tool_result_cache,
                    execution_state: &mut execution_state,
                    validation_state: tool_execution_evidence.validation_state,
                    read_file_tracker: &mut turn_state.read_files,
                    correction: correction_context.clone(),
                },
            )
            .await?;
            let iteration_restart_reason = match tool_execution_outcome.into_turn_transition() {
                TurnTransition::Restart(reason) => reason,
                TurnTransition::Finish(result) => return result,
                TurnTransition::Advance(never) => match never {},
            };

            // Guided-only re-planner. Autonomous models self-direct and never
            // spend an auxiliary model call evaluating their own step progress.
            if self.mandate_execution.is_none()
                && self.trust_tier_for_model(&model)
                == crate::agent::trust_tier::ModelTrustTier::Guided
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
            prepare_turn_restart(
                self,
                iteration_restart_reason,
                &mut turn_state,
                &mut approach_pivots_used,
                &model,
            );
        }
        }, turn_span)
        .await
    }
}

#[cfg(test)]
#[path = "characterization_tests.rs"]
mod characterization_tests;

#[cfg(test)]
mod stuck_fallback_tests {
    use super::*;

    #[test]
    fn new_task_assessment_excludes_unrelated_recent_context() {
        let recent = vec![json!({
            "role": "assistant",
            "content": "The previous device check requires live health evidence."
        })];
        assert_eq!(
            task_assessment_conversation_context(
                Some(FollowupMode::NewTask),
                Some("Prior device discussion"),
                &recent,
            ),
            None
        );
    }

    #[test]
    fn followup_assessment_retains_the_preceding_exchange() {
        let recent = vec![json!({
            "role": "user",
            "content": "Check the first device."
        })];
        let context =
            task_assessment_conversation_context(Some(FollowupMode::Followup), None, &recent)
                .expect("follow-up context");
        assert!(context.contains("Check the first device."));
    }

    #[test]
    fn fallback_does_not_infer_missing_details_from_request_wording() {
        let msg = build_stuck_no_output_fallback("Web search");
        assert!(msg.contains("automatic execution recovery"));
        assert!(!msg.contains('?'));
        assert!(!msg.contains("processing limit"));
    }

    #[test]
    fn fallback_is_wording_independent() {
        let msg = build_stuck_no_output_fallback("look it up");
        let detailed = build_stuck_no_output_fallback(
            "search the web for what Caro is a nickname for and summarize the top results",
        );
        assert_eq!(msg, detailed);
    }

    #[test]
    fn fallback_never_emits_legacy_processing_limit_string() {
        for input in ["Web search", "do something", "", "look up cats"] {
            assert!(!build_stuck_no_output_fallback(input).contains("processing limit"));
        }
    }
}
