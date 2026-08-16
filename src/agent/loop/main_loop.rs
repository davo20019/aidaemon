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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContinuationReceiptAssimilation {
    outcome_status: crate::traits::ToolOutcomeStatus,
    matched_requirement_indices: Vec<usize>,
    matched_contract: bool,
    observation_credited: bool,
    mutation_credited: bool,
}

/// Adopt one exact parent receipt into the child task's proof graph.
///
/// This is deliberately independent of notification prose. A receipt can
/// close child obligations only after typed task-lineage validation by the
/// caller, exact call/result correlation in EventStore, and semantic matching
/// against the child's current completion contract.
fn assimilate_continuation_receipt(
    contract: &CompletionContract,
    progress: &mut CompletionProgress,
    evidence: &crate::events::ContinuationToolEvidence,
    parent_task_id: &str,
    parent_result_id: &str,
) -> Option<ContinuationReceiptAssimilation> {
    let receipt = evidence.result.receipt.as_ref()?;
    let reportable = matches!(
        receipt.outcome_status,
        crate::traits::ToolOutcomeStatus::Succeeded
            | crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult
    );
    if !reportable {
        return Some(ContinuationReceiptAssimilation {
            outcome_status: receipt.outcome_status,
            matched_requirement_indices: Vec::new(),
            matched_contract: false,
            observation_credited: false,
            mutation_credited: false,
        });
    }

    let raw_arguments = serde_json::to_string(&evidence.call.arguments).ok()?;
    let mut metadata = receipt.to_metadata();
    super::tool_execution_phase::complete_tool_result_semantics(
        &evidence.result.name,
        &raw_arguments,
        &receipt.semantics,
        &mut metadata,
    );
    let semantics = &metadata.semantics;
    let receipt_scope = format!(
        "parent:{parent_task_id}:{}:{parent_result_id}",
        evidence.call.tool_call_id
    );

    let mutation_credited =
        receipt.outcome_status.satisfies_requested_condition() && semantics.mutates_state();
    if mutation_credited {
        progress.mark_mutation_receipt(contract, semantics, &receipt_scope);
    }

    let matched_contract = semantics.observes_state()
        && super::tool_execution_phase::observation_matches_completion_contract(
            contract,
            semantics,
            &raw_arguments,
            &evidence.result.result,
            &metadata,
        );
    let mut matched_requirement_indices = if semantics.observes_state() {
        super::tool_execution_phase::matching_evidence_requirement_indices(
            contract,
            semantics,
            &raw_arguments,
            &evidence.result.result,
            &metadata,
        )
    } else {
        Vec::new()
    };
    if semantics.observes_state() {
        for index in super::tool_execution_phase::accumulate_evidence_requirement_marker_matches(
            contract,
            progress,
            semantics,
            &raw_arguments,
            &evidence.result.result,
            &metadata,
        ) {
            if !matched_requirement_indices.contains(&index) {
                matched_requirement_indices.push(index);
            }
        }
    }
    let requirement_match = if contract.evidence_requirements.is_empty() {
        matched_contract
    } else {
        !matched_requirement_indices.is_empty()
    };
    let can_verify = semantics.observes_state()
        && super::tool_execution_phase::tool_result_or_metadata_contains_verifiable_evidence(
            semantics,
            &evidence.result.result,
            &metadata,
        );
    let observation_credited = can_verify && requirement_match;
    if can_verify {
        if contract.requires_observation && progress.verification_pending && requirement_match {
            progress.mark_verification_attempt();
        }
        progress.mark_observation_receipt(
            contract,
            &matched_requirement_indices,
            matched_contract,
            &receipt_scope,
        );
    }

    Some(ContinuationReceiptAssimilation {
        outcome_status: receipt.outcome_status,
        matched_requirement_indices,
        matched_contract,
        observation_credited,
        mutation_credited,
    })
}

#[cfg(test)]
fn task_assessment_conversation_context(
    followup_mode: Option<FollowupMode>,
    session_summary: Option<&str>,
    recent_messages: &[Value],
) -> Option<String> {
    super::bootstrap_phase::task_planning::task_assessment_conversation_context(
        followup_mode,
        session_summary,
        recent_messages,
    )
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
        continuation_parent_task_id: Option<&str>,
        continuation_parent_tool_call_id: Option<&str>,
        continuation_parent_result_id: Option<&str>,
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
                parent_task_id: continuation_parent_task_id,
            },
        )
        .await?;
        let BootstrapData {
            user_text: canonical_user_text,
            task_id,
            task_plan: bootstrap_task_plan,
            memory_pipeline_policy,
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
            turn_context,
            mut project_instruction_tracker,
            core_prompt_bytes,
            fresh_task_context_tail,
            continuation_task_context_tail,
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
        // Bootstrap finalized the immutable relationship, contract, memory policy,
        // and prompt context before this execution loop was constructed.
        let task_plan = bootstrap_task_plan;
        // The semantic assessment may refine the bootstrap relationship. Use
        // the finalized typed relationship—not the request wording or an
        // earlier provisional value—to decide whether prior turns can enter
        // the provider transcript.
        let preserve_archived_context = matches!(
            turn_context.followup_mode,
            Some(
                crate::agent::followup::FollowupMode::Followup
                    | crate::agent::followup::FollowupMode::ClarificationAnswer
            )
        );
        let mut task_context_tail = if preserve_archived_context {
            continuation_task_context_tail
        } else {
            fresh_task_context_tail
        };
        let prior_assistant_message_id = preserve_archived_context
            .then(|| {
                dialogue_state
                    .as_ref()
                    .and_then(|state| state.last_assistant_turn.as_ref())
                    .map(|turn| turn.message_id.clone())
            })
            .flatten();

        // The bootstrap policy was compiled from this exact assessment before
        // optional memory access. Reassert the deny edge after contract
        // installation so no later post-task path can widen it.
        if turn_context
            .completion_contract
            .forbidden_tool_scopes
            .contains(&crate::traits::ToolSemanticScope::UserMemory)
        {
            learning_ctx.memory_persistence_allowed = false;
        }
        debug_assert_eq!(
            learning_ctx.memory_persistence_allowed,
            memory_pipeline_policy.allows_memory()
        );

        // Derive all contract-dependent state exactly once from that finalized
        // value so loop control, progress tracking, budgets, and telemetry agree.
        debug_assert!(turn_context.completion_contract.belongs_to_task(&task_id));
        let mut completion_progress =
            CompletionProgress::new(&turn_context.completion_contract, &task_id);

        // A background completion re-enters as a child task. Import only the
        // exact terminal receipt named by that typed continuation edge, and
        // only when the finalized contract explicitly adopted the parent task.
        // The notification's natural-language body is never proof.
        if let Some(parent_task_id) = continuation_parent_task_id {
            let lineage_adopted = turn_context
                .completion_contract
                .adopted_from_task_ids
                .iter()
                .any(|candidate| candidate == parent_task_id);
            let exact_reference = continuation_parent_tool_call_id
                .zip(continuation_parent_result_id);
            let mut telemetry = json!({
                "condition": "continuation_receipt_assimilation",
                "parent_task_id": parent_task_id,
                "parent_tool_call_id": continuation_parent_tool_call_id,
                "parent_result_id": continuation_parent_result_id,
                "child_task_id": task_id,
                "lineage_adopted": lineage_adopted,
                "receipt_found": false,
                "observation_credited": false,
                "mutation_credited": false,
                "reason_code": if lineage_adopted {
                    if exact_reference.is_some() {
                        "exact_receipt_not_found"
                    } else {
                        "incomplete_receipt_reference"
                    }
                } else {
                    "parent_task_not_adopted"
                },
            });
            if lineage_adopted {
                if let Some((parent_tool_call_id, parent_result_id)) = exact_reference {
                    if let Some(evidence) = self
                        .event_store
                        .continuation_tool_evidence(
                            session_id,
                            parent_task_id,
                            parent_tool_call_id,
                            parent_result_id,
                        )
                        .await?
                    {
                        telemetry["receipt_found"] = json!(true);
                        if let Some(assimilation) = assimilate_continuation_receipt(
                            &turn_context.completion_contract,
                            &mut completion_progress,
                            &evidence,
                            parent_task_id,
                            parent_result_id,
                        ) {
                            telemetry["outcome_status"] =
                                json!(assimilation.outcome_status.as_str());
                            telemetry["matched_requirement_indices"] =
                                json!(assimilation.matched_requirement_indices);
                            telemetry["matched_contract"] = json!(assimilation.matched_contract);
                            telemetry["observation_credited"] =
                                json!(assimilation.observation_credited);
                            telemetry["mutation_credited"] =
                                json!(assimilation.mutation_credited);
                            telemetry["reason_code"] = json!(
                                if assimilation.observation_credited
                                    || assimilation.mutation_credited
                                {
                                    "receipt_assimilated"
                                } else {
                                    "receipt_did_not_match_child_obligations"
                                }
                            );
                        } else {
                            telemetry["reason_code"] = json!("receipt_missing_typed_metadata");
                        }
                    }
                }
            }
            self.emit_decision_point(
                &emitter,
                &task_id,
                0,
                DecisionType::EvidenceGate,
                "Evaluated typed parent receipt for continuation".to_string(),
                telemetry,
            )
            .await;
        }
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
        } else {
            if !turn_context
                .completion_contract
                .allowed_tool_names
                .is_empty()
            {
                let allowed_tool_names = &turn_context.completion_contract.allowed_tool_names;
                tool_defs.retain(|definition| {
                    definition
                        .get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|name| allowed_tool_names.iter().any(|allowed| allowed == name))
                });
            }
            if !turn_context
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
        // Policy filtering may rebuild definitions from `base_tool_defs`.
        // Reapply the task's hard capability boundary afterward so an
        // uncertainty-driven policy refresh cannot widen an allow-only or
        // denied-scope contract.
        if turn_context.completion_contract.forbids_tool_use {
            tool_defs.clear();
        } else if !turn_context
            .completion_contract
            .allowed_tool_names
            .is_empty()
        {
            let allowed = &turn_context.completion_contract.allowed_tool_names;
            tool_defs.retain(|definition| {
                definition
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|name| allowed.iter().any(|candidate| candidate == name))
            });
        }
        if !turn_context
            .completion_contract
            .forbidden_tool_scopes
            .is_empty()
        {
            let forbidden = &turn_context.completion_contract.forbidden_tool_scopes;
            tool_defs.retain(|definition| {
                let Some(name) = definition
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(serde_json::Value::as_str)
                else {
                    return false;
                };
                self.tools
                    .iter()
                    .find(|tool| tool.name() == name && tool.is_available())
                    .and_then(|tool| {
                        tool.semantic_affordances()
                            .map(|affordances| affordances.scope)
                    })
                    .or_else(|| {
                        super::tool_execution_phase::fallback_tool_semantic_scope(name)
                    })
                    .is_none_or(|scope| !forbidden.contains(&scope))
            });
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
                    preserve_archived_context,
                    prior_assistant_message_id: prior_assistant_message_id.as_deref(),
                    summary_last_message_id: session_summary
                        .as_ref()
                        .filter(|_| preserve_archived_context)
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

    fn detached_command_evidence(
        status: crate::traits::ToolOutcomeStatus,
        exit_code: i32,
        result: &str,
    ) -> crate::events::ContinuationToolEvidence {
        let arguments = json!({
            "command": "/usr/bin/false",
            "working_dir": "/tmp",
        });
        let mut metadata = crate::traits::ToolCallMetadata {
            outcome_status: Some(status),
            exit_code: Some(exit_code),
            semantics: crate::tools::command_semantics::classify_shell_command("/usr/bin/false"),
            ..crate::traits::ToolCallMetadata::default()
        };
        super::super::tool_execution_phase::complete_tool_result_semantics(
            "terminal",
            &serde_json::to_string(&arguments).unwrap(),
            &metadata.semantics.clone(),
            &mut metadata,
        );
        let mut receipt = crate::events::ToolReceiptV1::from_metadata(
            &metadata,
            status,
            crate::events::ToolOutcomeEvidenceSource::StructuredMetadata,
            None,
        );
        receipt.result_provenance.result_id = Some("result:synthetic-negative".to_string());
        crate::events::ContinuationToolEvidence {
            call: crate::events::ToolCallData {
                tool_call_id: "call-parent".to_string(),
                name: "terminal".to_string(),
                arguments,
                summary: None,
                task_id: Some("task-parent".to_string()),
                idempotency_key: None,
                policy_rev: None,
                risk_score: None,
                turn_id: None,
            },
            result: crate::events::ToolResultData {
                message_id: None,
                tool_call_id: "call-parent".to_string(),
                name: "terminal".to_string(),
                result: result.to_string(),
                success: true,
                duration_ms: 1,
                error: None,
                task_id: Some("task-parent".to_string()),
                annotations: Vec::new(),
                turn_id: None,
                attachments: Vec::new(),
                receipt: Some(receipt),
            },
        }
    }

    #[test]
    fn adopted_negative_parent_receipt_closes_child_outcome_obligation() {
        let mut contract = CompletionContract {
            scope_task_id: Some("task-child".to_string()),
            adopted_from_task_ids: vec!["task-parent".to_string()],
            requires_observation: true,
            evidence_requirements: vec![crate::traits::RequestEvidenceRequirement {
                summary: "Observed process outcome".to_string(),
                acceptable_scopes: vec![crate::traits::ToolSemanticScope::HostLocal],
                purpose: crate::traits::EvidencePurpose::Outcome,
                minimum_authority: crate::traits::EvidenceAuthority::Direct,
                temporal_scope: crate::traits::EvidenceTemporalScope::Current,
                required_content_markers: vec!["exit".to_string()],
                target: None,
            }],
            ..CompletionContract::default()
        };
        contract.adopt_for_task("task-child");
        let mut progress = CompletionProgress::new(&contract, "task-child");
        let evidence = detached_command_evidence(
            crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult,
            1,
            "",
        );

        let assimilation = assimilate_continuation_receipt(
            &contract,
            &mut progress,
            &evidence,
            "task-parent",
            "result:synthetic-negative",
        )
        .unwrap();

        assert!(assimilation.observation_credited);
        assert_eq!(assimilation.matched_requirement_indices, [0]);
        assert!(progress.all_evidence_requirements_satisfied());
        assert!(!progress.verification_pending);
    }

    #[test]
    fn parent_receipt_cannot_close_an_unrelated_child_requirement() {
        let contract = CompletionContract {
            scope_task_id: Some("task-child".to_string()),
            adopted_from_task_ids: vec!["task-parent".to_string()],
            requires_observation: true,
            evidence_requirements: vec![crate::traits::RequestEvidenceRequirement {
                summary: "Read package edition".to_string(),
                acceptable_scopes: vec![crate::traits::ToolSemanticScope::LocalWorkspace],
                purpose: crate::traits::EvidencePurpose::Content,
                minimum_authority: crate::traits::EvidenceAuthority::Direct,
                temporal_scope: crate::traits::EvidenceTemporalScope::Current,
                required_content_markers: vec!["edition".to_string()],
                target: None,
            }],
            ..CompletionContract::default()
        };
        let mut progress = CompletionProgress::new(&contract, "task-child");
        let evidence = detached_command_evidence(
            crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult,
            1,
            "",
        );

        let assimilation = assimilate_continuation_receipt(
            &contract,
            &mut progress,
            &evidence,
            "task-parent",
            "result:synthetic-negative",
        )
        .unwrap();

        assert!(!assimilation.observation_credited);
        assert!(progress.verification_pending);
        assert!(!progress.all_evidence_requirements_satisfied());
    }

    #[test]
    fn provisional_new_task_assessment_can_name_but_not_adopt_an_antecedent() {
        let recent = vec![json!({
            "message_id": "synthetic-prior-user",
            "role": "assistant",
            "content": "The previous device check requires live health evidence."
        })];
        let context = task_assessment_conversation_context(
            Some(FollowupMode::NewTask),
            Some("Prior device discussion"),
            &recent,
        )
        .expect("bounded relationship context");
        assert!(context.contains("message_id=synthetic-prior-user"));
        assert!(context.contains("Prior device discussion"));
    }

    #[test]
    fn followup_assessment_retains_the_preceding_exchange() {
        let recent = vec![json!({
            "message_id": "synthetic-user-1",
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
