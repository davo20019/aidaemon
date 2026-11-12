//! Stop checks: the daemon side of `DaemonCall::Stop`. The ledger closeout
//! and the stop-time demands, each judged by the closeout arbiter.

use super::*;
pub(super) use aidaemon_runtime::engine::stop::*;

/// The tools the closeout may count as reachable: the registered daemon
/// tools, minus daemon memory tools the active memory plugin does not own,
/// and the MCP tools bridged for the turn.
pub(super) fn closeout_visible_tools<'a>(
    tools: &'a [Arc<dyn crate::traits::Tool>],
    mcp_tools: &'a [Arc<dyn crate::traits::Tool>],
    user_role: UserRole,
    owned_memory_tools: &[&str],
) -> Vec<&'a str> {
    if user_role == UserRole::Public {
        return Vec::new();
    }
    let mut visible: Vec<&str> = tools
        .iter()
        .map(|tool| tool.name())
        .filter(|name| daemon_tool_offer(name, owned_memory_tools) != DaemonToolOffer::Hidden)
        .collect();
    for tool in mcp_tools {
        if !visible.contains(&tool.name()) {
            visible.push(tool.name());
        }
    }
    visible
}

impl Agent {
    pub(super) async fn serve_sdk_stop(&self, scope: &TurnScope<'_>, check: StopCheck) {
        let mut decision = self.sdk_mandate_commit_check(scope).await;
        if decision == sdk::StopDecision::Accept && self.sdk_background_handoff(scope).await {
            let _ = check.reply.send(decision);
            return;
        }
        if decision == sdk::StopDecision::Accept {
            decision = self
                .sdk_closeout(scope, check.completed_results, check.failed_without_success)
                .await;
            // The agent said the request is partial or blocked: the run stops
            // unmet, and only its reply may still need repair.
            if let sdk::StopDecision::AcceptUnmet { .. } = decision {
                let repair = self.sdk_reply_check(scope, &check.final_text).await;
                let _ = check.reply.send(if repair.permits_stop() {
                    decision
                } else {
                    repair
                });
                return;
            }
            if decision != sdk::StopDecision::Accept
                && reply_defect(&check.final_text, None).is_none()
            {
                scope
                    .verdict
                    .record_interrupted_answer(check.final_text.clone());
            }
        }
        if decision == sdk::StopDecision::Accept {
            decision = self.sdk_gui_click_check(scope).await;
        }
        if decision == sdk::StopDecision::Accept {
            decision = self.sdk_combined_verification_check(scope).await;
        }
        if decision == sdk::StopDecision::Accept {
            decision = self.sdk_reply_check(scope, &check.final_text).await;
        }
        let _ = check.reply.send(decision);
    }

    /// A stop while a job this run launched is still running is a handoff,
    /// decided from the run's typed launch and settle receipts, never from
    /// the reply: the job's notifier owns its result and re-engages the
    /// session when it ends, so the run demands no more work, runs no reply
    /// checks, and ships the handoff reply instead of the model's verdict on
    /// work that has not finished (live 2026-11-11: a run polled a running
    /// Codex job, gave up, and reported the task as failed while Codex went
    /// on to build it).
    pub(super) async fn sdk_background_handoff(&self, scope: &TurnScope<'_>) -> bool {
        if scope.open_background_jobs.load(Ordering::Relaxed) == 0 {
            return false;
        }
        scope.background_handoff.store(true, Ordering::Relaxed);
        scope.closeout_ended_unmet.store(false, Ordering::Relaxed);
        scope.verdict.record_handoff_reply(
            crate::agent::post_task::build_friendly_background_handoff(scope.user_text),
        );
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            scope.model_calls.load(Ordering::Relaxed) as usize,
            crate::events::DecisionType::PostExecutionValidation,
            "Stop with a background job still running: handed off".to_string(),
            serde_json::json!({
                "condition": "background_job_running_at_stop",
                "open_jobs": scope.open_background_jobs.load(Ordering::Relaxed),
                "engine": "sdk",
            }),
        )
        .await;
        true
    }

    /// A GUI success claim must not ship while the task's last coordinate
    /// click is unverified: a raw (x, y) click has no element identity, so
    /// only a follow-up observation (which clears the flag) can confirm it
    /// (2026-07-12: an unverified click at a guessed position was reported as
    /// a successful Like). Bounded, so a model that never looks cannot loop.
    #[cfg(feature = "computer_use")]
    pub(super) async fn sdk_gui_click_check(&self, scope: &TurnScope<'_>) -> sdk::StopDecision {
        use crate::agent::closeout::{StopDemand, StopDemandVerdict};
        if self.depth != 0
            || !crate::agent::computer_use::task_has_unverified_coordinate_click(scope.task_id)
                .await
        {
            return sdk::StopDecision::Accept;
        }
        let demand = StopDemand::GuiClickVerification;
        let used = scope.gui_verify_demands.load(Ordering::Relaxed);
        if self.sdk_stop_demand_verdict(scope, demand, used).await != StopDemandVerdict::Demand {
            return sdk::StopDecision::Accept;
        }
        scope.gui_verify_demands.fetch_add(1, Ordering::Relaxed);
        self.emit_warning_decision_point(
            scope.emitter,
            scope.task_id,
            scope.model_calls.load(Ordering::Relaxed) as usize,
            crate::events::DecisionType::PostExecutionValidation,
            "GUI success claim blocked: unverified coordinate click".to_string(),
            serde_json::json!({ "condition": "gui_coordinate_click_unverified", "engine": "sdk" }),
        )
        .await;
        sdk::StopDecision::Continue {
            message: crate::agent::system_directives::SystemDirective::GuiCoordinateClickUnverified
                .render(),
        }
    }

    #[cfg(not(feature = "computer_use"))]
    pub(super) async fn sdk_gui_click_check(&self, _scope: &TurnScope<'_>) -> sdk::StopDecision {
        sdk::StopDecision::Accept
    }

    /// A file more than one agent changed is checked once before the
    /// top-level agent reports: a read or command after the last change,
    /// recorded by the tree, closes it. Children are asked nothing.
    pub(super) async fn sdk_combined_verification_check(
        &self,
        scope: &TurnScope<'_>,
    ) -> sdk::StopDecision {
        use crate::agent::closeout::{StopDemand, StopDemandVerdict};
        let Some(coordination) = scope.spawn_family.active_coordination() else {
            return sdk::StopDecision::Accept;
        };
        if !coordination.is_root(scope.session_id) {
            return sdk::StopDecision::Accept;
        }
        let shared = coordination.unverified_shared_files();
        if shared.is_empty() {
            return sdk::StopDecision::Accept;
        }
        let demand = StopDemand::CombinedVerification;
        let used = scope.combined_verification_demands.load(Ordering::Relaxed);
        if self.sdk_stop_demand_verdict(scope, demand, used).await != StopDemandVerdict::Demand {
            return sdk::StopDecision::Accept;
        }
        scope
            .combined_verification_demands
            .fetch_add(1, Ordering::Relaxed);
        sdk::StopDecision::Continue {
            message: format!(
                "More than one agent changed these files: {}. Read them or run a check on the \
                 combined result before you report.",
                shared.join(", ")
            ),
        }
    }

    /// A mandate task lead's prose is never its decision: a missing typed
    /// commit gets one bounded correction turn when the run could still make
    /// it, then the run fails closed.
    pub(super) async fn sdk_mandate_commit_check(
        &self,
        scope: &TurnScope<'_>,
    ) -> sdk::StopDecision {
        use crate::agent::closeout::{StopDemand, StopDemandVerdict};
        use crate::agent::mandate_run::{
            mandate_decision_commit_state, MandateDecisionCommitState,
        };
        let state = match mandate_decision_commit_state(self).await {
            Ok(state) => state,
            Err(error) => {
                warn!(task_id = scope.task_id, %error, "SDK engine: mandate decision lookup failed");
                return sdk::StopDecision::Accept;
            }
        };
        let error = match state {
            MandateDecisionCommitState::NotApplicable | MandateDecisionCommitState::Committed => {
                return sdk::StopDecision::Accept;
            }
            MandateDecisionCommitState::Invalid => {
                "Mandate deliberator found a decision for the current run with mismatched \
                 immutable identity."
            }
            MandateDecisionCommitState::Missing => {
                let demand = StopDemand::MandateDecisionCommit;
                let used = scope.mandate_commit_retries.load(Ordering::Relaxed);
                match self.sdk_stop_demand_verdict(scope, demand, used).await {
                    StopDemandVerdict::Demand => {
                        scope.mandate_commit_retries.fetch_add(1, Ordering::Relaxed);
                        self.emit_warning_decision_point(
                            scope.emitter,
                            scope.task_id,
                            scope.model_calls.load(Ordering::Relaxed) as usize,
                            crate::events::DecisionType::PostExecutionValidation,
                            "Blocked mandate prose completion without a durable decision"
                                .to_string(),
                            serde_json::json!({
                                "condition": demand.condition(),
                                "retry": used + 1,
                                "engine": "sdk",
                            }),
                        )
                        .await;
                        return sdk::StopDecision::Continue {
                            message:
                                crate::agent::system_directives::SystemDirective::MandateDecisionCommitRequired
                                    .render(),
                        };
                    }
                    StopDemandVerdict::Exhausted => {
                        "Mandate deliberator returned without committing an exact durable \
                         decision after its bounded correction turn."
                    }
                    StopDemandVerdict::Unreachable => {
                        "Mandate deliberator returned without a durable decision, and this \
                         run's authority does not admit manage_mandates record_decision, so \
                         no correction turn could commit one."
                    }
                }
            }
        };
        if let Ok(mut failure) = scope.verdict.run_failure.lock() {
            *failure = Some(error.to_string());
        }
        sdk::StopDecision::Accept
    }

    /// Ask the closeout arbiter whether `demand` may be made now, against the
    /// same authority and visible tools the ledger closeout uses, and record
    /// the verdict.
    pub(super) async fn sdk_stop_demand_verdict(
        &self,
        scope: &TurnScope<'_>,
        demand: crate::agent::closeout::StopDemand,
        used: u32,
    ) -> crate::agent::closeout::StopDemandVerdict {
        let authority =
            crate::agent::closeout::ClosingAuthority::from_resolved(&scope.authority.authority)
                .without_tool_demands(scope.contract.forbids_tool_use);
        let visible_tools = closeout_visible_tools(
            &self.tools,
            &scope.mcp_tools,
            scope.user_role,
            scope.owned_memory_tools,
        );
        let find = |name: &str| self.tools.iter().find(|tool| tool.name() == name);
        let verdict = crate::agent::closeout::stop_demand_verdict(
            demand,
            used,
            &authority,
            &visible_tools,
            |name| find(name).map(|tool| tool.capabilities().read_only),
            |name, arguments| {
                find(name).is_some_and(|tool| tool.owner_control_plane_call(arguments))
            },
        );
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            scope.model_calls.load(Ordering::Relaxed) as usize,
            crate::events::DecisionType::PostExecutionValidation,
            format!("Stop demand {}: {}", demand.condition(), verdict.as_str()),
            serde_json::json!({
                "condition": "stop_demand",
                "demand": demand.condition(),
                "verdict": verdict.as_str(),
                "used": used,
                "budget": demand.budget(),
                "authoritative": true,
                "engine": "sdk",
            }),
        )
        .await;
        verdict
    }

    /// One shared retry for a reply that must not ship (empty, only internal
    /// markers, pasted tool output, or written to a closeout demand that
    /// ended unmet); after it, the finalizer's honest fallback applies.
    pub(super) async fn sdk_reply_check(
        &self,
        scope: &TurnScope<'_>,
        final_text: &str,
    ) -> sdk::StopDecision {
        let latest = scope
            .latest_tool_output
            .lock()
            .ok()
            .and_then(|latest| latest.clone());
        let defect = reply_defect(final_text, latest.as_deref()).or_else(|| {
            scope
                .closeout_ended_unmet
                .load(Ordering::Relaxed)
                .then_some(ReplyDefect::AnswersUnmetDemand)
        });
        let Some(defect) = defect else {
            return sdk::StopDecision::Accept;
        };
        let used = u32::from(scope.reply_retry_used.load(Ordering::Relaxed));
        if self
            .sdk_stop_demand_verdict(scope, crate::agent::closeout::StopDemand::ReplyRepair, used)
            .await
            != crate::agent::closeout::StopDemandVerdict::Demand
        {
            return sdk::StopDecision::Accept;
        }
        scope.reply_retry_used.store(true, Ordering::Relaxed);
        warn!(
            session_id = scope.session_id,
            task_id = scope.task_id,
            ?defect,
            "SDK engine: final reply rejected; retrying once"
        );
        let directive = match defect {
            // After real work an empty reply is the model treating the run as
            // finished; offering tools again invites another empty turn, so
            // the retry must answer in text from the receipts it holds.
            ReplyDefect::Empty if scope.succeeded_calls.load(Ordering::Relaxed) > 0 => {
                scope.text_only_next_call.store(true, Ordering::Relaxed);
                crate::agent::system_directives::SystemDirective::EmptyFinalReplyPlainText
            }
            ReplyDefect::Empty => {
                crate::agent::system_directives::SystemDirective::EmptyResponseRetry
            }
            ReplyDefect::InternalMarkers => {
                crate::agent::system_directives::SystemDirective::FinalAnswerRejectedInternalMarkers
            }
            ReplyDefect::ToolOutputPaste => {
                crate::agent::system_directives::SystemDirective::FinalAnswerWasFilePaste
            }
            // The work is over: the rewrite answers the user from what the
            // run already holds, and an empty rewrite ships this reply.
            ReplyDefect::AnswersUnmetDemand => {
                scope
                    .verdict
                    .record_interrupted_answer(final_text.to_string());
                scope.text_only_next_call.store(true, Ordering::Relaxed);
                crate::agent::system_directives::SystemDirective::FinalAnswerAfterUnmetCloseout
            }
        };
        sdk::StopDecision::Continue {
            message: directive.render(),
        }
    }

    /// The ledger-first closeout: replay the run
    /// ledger, ask which expectations an admissible tool could still satisfy,
    /// and demand a bounded extra pass only when the receipts say so.
    pub(super) async fn sdk_closeout(
        &self,
        scope: &TurnScope<'_>,
        completed_results: usize,
        failed_without_success: bool,
    ) -> sdk::StopDecision {
        let Ok(aggregate) = self
            .event_store
            .task_run_aggregate(scope.session_id, scope.task_id)
            .await
        else {
            return sdk::StopDecision::Accept;
        };
        let iteration = scope.model_calls.load(Ordering::Relaxed) as usize;
        // The agent's own typed verdict: `partial` or `blocked` ends the run
        // unmet. No more work is demanded (the agent said it cannot or will
        // not finish), and nothing later turns it into a success.
        if let Some(claim) = aggregate
            .executor_claimed_outcome
            .filter(|claim| claim.unfinished())
        {
            scope.closeout_ended_unmet.store(false, Ordering::Relaxed);
            self.emit_decision_point(
                scope.emitter,
                scope.task_id,
                iteration,
                crate::events::DecisionType::PostExecutionValidation,
                format!("Agent reported the request as {}", claim.as_str()),
                serde_json::json!({
                    "condition": "executor_claimed_outcome",
                    "claimed_outcome": claim.as_str(),
                    "engine": "sdk",
                }),
            )
            .await;
            return sdk::StopDecision::AcceptUnmet {
                unmet: vec![format!(
                    "the agent reported the request as {}",
                    claim.as_str()
                )],
            };
        }
        let authority =
            crate::agent::closeout::ClosingAuthority::from_resolved(&scope.authority.authority)
                .without_tool_demands(scope.contract.forbids_tool_use);
        let visible_tools = closeout_visible_tools(
            &self.tools,
            &scope.mcp_tools,
            scope.user_role,
            scope.owned_memory_tools,
        );
        let read_only = |name: &str| {
            self.tools
                .iter()
                .find(|tool| tool.name() == name)
                .map(|tool| tool.capabilities().read_only)
        };
        let decision = aggregate.closeout(|obligation| {
            crate::agent::closeout::obligation_admissible(
                obligation,
                &authority,
                &visible_tools,
                read_only,
            )
        });
        let reachable = match &decision {
            crate::events::CloseoutDecision::Reachable { obligation_ids } => obligation_ids.clone(),
            _ => Vec::new(),
        };
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            iteration,
            crate::events::DecisionType::PostExecutionValidation,
            format!("Ledger closeout verdict: {}", decision.verdict()),
            serde_json::json!({
                "condition": "ledger_closeout",
                "verdict": decision.verdict(),
                "proof_basis": match &decision {
                    crate::events::CloseoutDecision::Closed { proof_basis } => Some(*proof_basis),
                    _ => None,
                },
                "reachable_obligation_ids": reachable,
                "authoritative": true,
                "engine": "sdk",
            }),
        )
        .await;
        let demanded_at = scope.executor_demanded_at.load(Ordering::Relaxed);
        // Delivering a capture to the chat changes nothing an observation
        // could verify, so it does not earn the verify-once pass.
        let mutation_succeeded = aggregate.operations.values().any(|operation| {
            operation.changes_verifiable_state()
                && operation.outcome == Some(crate::traits::ToolOutcomeStatus::Succeeded)
        });
        let observation_reachable = reachable.iter().any(|id| {
            aggregate.obligations.get(id).is_some_and(|obligation| {
                obligation.class == crate::events::RunObligationClass::Observe
            })
        });
        let verification_after_mutation = mutation_succeeded
            && observation_reachable
            && !scope.verification_demanded.load(Ordering::Relaxed);
        // A failure that is itself the expected observation satisfies its
        // obligation; only an uncredited failure earns the recovery pass.
        let failure_credited = aggregate
            .obligations
            .values()
            .any(|obligation| obligation.state == crate::events::RunObligationState::Satisfied);
        let unrecovered_failure = failed_without_success
            && !failure_credited
            && !scope.recovery_demanded.load(Ordering::Relaxed);
        let facts = StopFacts {
            reachable_obligation_ids: reachable.clone(),
            completed_results,
            executor_demanded_at: (demanded_at != u64::MAX).then_some(demanded_at as usize),
            terminal_policy_denial: aggregate.terminal_policy_denial(),
            verification_after_mutation,
            unrecovered_failure,
        };
        if !closeout_demands_work(&facts)
            || scope.closeout_demands.load(Ordering::Relaxed) >= MAX_CLOSEOUT_DEMANDS
        {
            scope.closeout_ended_unmet.store(
                closeout_ended_unmet(&facts, scope.closeout_demands.load(Ordering::Relaxed)),
                Ordering::Relaxed,
            );
            return sdk::StopDecision::Accept;
        }
        scope.closeout_ended_unmet.store(false, Ordering::Relaxed);
        let demands = scope.closeout_demands.fetch_add(1, Ordering::Relaxed) + 1;
        if verification_after_mutation {
            scope.verification_demanded.store(true, Ordering::Relaxed);
        }
        if unrecovered_failure && reachable.is_empty() {
            scope.recovery_demanded.store(true, Ordering::Relaxed);
            let evidence = scope
                .latest_tool_output
                .lock()
                .ok()
                .and_then(|output| output.clone())
                .unwrap_or_default();
            return sdk::StopDecision::Continue {
                message:
                    crate::agent::system_directives::SystemDirective::ExternalMutationRecoveryRequired(
                        evidence.chars().take(600).collect(),
                    )
                    .render(),
            };
        }
        if reachable
            .iter()
            .any(|id| id.contains("/obligation:checklist:"))
        {
            scope
                .executor_demanded_at
                .store(completed_results as u64, Ordering::Relaxed);
        }
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            iteration,
            crate::events::DecisionType::IntentGate,
            "Ledger expectations still reachable: bounded evidence pass requested",
            serde_json::json!({
                "condition": "ledger_expectations_required",
                "reachable_obligation_ids": reachable,
                "demands": demands,
                "engine": "sdk",
            }),
        )
        .await;
        sdk::StopDecision::Continue {
            message: crate::agent::system_directives::SystemDirective::LedgerExpectationsRequired {
                expectations: aggregate.expectation_summaries(&reachable),
            }
            .render(),
        }
    }
}
