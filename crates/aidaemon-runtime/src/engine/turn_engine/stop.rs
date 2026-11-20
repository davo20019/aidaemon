//! Stop checks: the daemon side of `DaemonCall::Stop`. The ledger closeout
//! and the stop-time demands, each judged by the closeout arbiter.

use super::*;
pub use crate::engine::stop::*;

/// Record the declared items the run ends without attempting, with their
/// typed reason, for the card and the reply; returns the run's unmet lines.
pub fn record_not_attempted(
    scope: &TurnScope<'_>,
    aggregate: &aidaemon_events::run_aggregate::RunAggregate,
) -> Vec<String> {
    // What the judgement saw, so a "didn't start" on finished work can be
    // traced (live 2026-11-15: a run that wrote and committed its change
    // was told it never started it).
    tracing::info!(
        task_id = scope.task_id,
        unattempted = ?aggregate.unattempted_declared_work(),
        checklist_owner_units = ?aggregate.checklist_owner_units,
        owner_units = aggregate.owner_units.len(),
        mutating_operations = aggregate.operations.values().filter(|op| op.mutating).count(),
        operations = aggregate.operations.len(),
        "Declared work judged not attempted"
    );
    let items: Vec<(Option<usize>, String)> = aggregate
        .unattempted_declared_work()
        .iter()
        .filter_map(|id| {
            // One of the owner's own units: its words, no step on the card.
            if let Some(unit) = aggregate.owner_unit_of(id) {
                return Some((None, unit.what.trim().to_string()));
            }
            let index = id
                .rsplit_once("/obligation:checklist:")
                .and_then(|(_, index)| index.parse().ok())?;
            let obligation = aggregate.obligations.get(id)?;
            let step = obligation
                .step
                .clone()
                .or_else(|| obligation.summary.clone())
                .unwrap_or_default();
            Some((Some(index), step))
        })
        .collect();
    let unmet = if items.is_empty() {
        vec![UNATTEMPTED_WORK_UNMET.to_string()]
    } else {
        items
            .iter()
            .map(|(_, step)| format!("{UNATTEMPTED_WORK_UNMET}: {step}"))
            .collect()
    };
    if let Ok(mut recorded) = scope.not_attempted.lock() {
        for item in items {
            if !recorded.contains(&item) {
                recorded.push(item);
            }
        }
    }
    unmet
}

/// The tools the closeout may count as reachable: the registered daemon
/// tools, minus daemon memory tools the active memory plugin does not own,
/// and the MCP tools bridged for the turn.
pub fn closeout_visible_tools<'a>(
    tools: &'a [Arc<dyn aidaemon_tools::OutcomeTool>],
    mcp_tools: &'a [Arc<dyn aidaemon_tools::OutcomeTool>],
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

/// How often a held stop rereads whether the run's jobs have ended.
const UNREPORTED_JOB_POLL: std::time::Duration = std::time::Duration::from_millis(500);
/// The longest one held stop waits for the run's jobs. The stop hook's
/// own timeout ([`super::super::hooks::LedgerStop`]) is this plus room for
/// the rest of the judgment, so the wait ends first and the run ends held
/// rather than with an unanswered check.
pub(crate) const UNREPORTED_JOB_MAX_WAIT: std::time::Duration =
    std::time::Duration::from_secs(30 * 60);

impl TurnEngine<'_> {
    /// Judge a stop. A stop sent back for more work holds the reply the
    /// model was about to give: the kernel ships only the run's last reply,
    /// so the held one never reaches the user. Every `Continue` says so,
    /// whichever check sent it, so the model's next reply stands on its own
    /// (live 2026-11-18: a story was held for a check, and the next reply
    /// said "the full story is in my message just above").
    pub async fn serve_sdk_stop(&self, scope: &TurnScope<'_>, check: StopCheck) {
        let (inner, verdict) = tokio::sync::oneshot::channel();
        let StopCheck {
            completed_results,
            failed_without_success,
            final_text,
            reply,
        } = check;
        self.judge_sdk_stop(
            scope,
            StopCheck {
                completed_results,
                failed_without_success,
                final_text,
                reply: inner,
            },
        )
        .await;
        let Ok(decision) = verdict.await else {
            return;
        };
        let _ = reply.send(match decision {
            sdk::StopDecision::Continue { message } => sdk::StopDecision::Continue {
                message: format!(
                    "{message}\n\n{}",
                    crate::turn::system_directives::HELD_REPLY_NOT_DELIVERED
                ),
            },
            other => other,
        });
    }

    async fn judge_sdk_stop(&self, scope: &TurnScope<'_>, check: StopCheck) {
        if let Some(ended) = self.sdk_wait_for_unreported_jobs(scope).await {
            let _ = check.reply.send(sdk::StopDecision::Continue {
                message: crate::turn::system_directives::SystemDirective::BackgroundJobHeld {
                    ended,
                }
                .render(),
            });
            return;
        }
        let mut decision = self.sdk_mandate_commit_check(scope).await;
        // A stop that would hand the turn to a running job first answers for
        // every declared item: one nothing took on is launched or asked
        // about now, since the job's result answers only for its own items.
        if decision == sdk::StopDecision::Accept
            && scope.open_background_jobs.load(Ordering::Relaxed) > 0
        {
            match self.sdk_judge_unattempted_work(scope, None, false).await {
                Some(held @ sdk::StopDecision::Continue { .. }) => {
                    let _ = check.reply.send(held);
                    return;
                }
                Some(unmet) => decision = unmet,
                None => {}
            }
        }
        if matches!(
            decision,
            sdk::StopDecision::Accept | sdk::StopDecision::AcceptUnmet { .. }
        ) && self.sdk_background_handoff(scope).await
        {
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
            decision = self.sdk_code_check(scope).await;
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
    pub async fn sdk_background_handoff(&self, scope: &TurnScope<'_>) -> bool {
        if scope.open_background_jobs.load(Ordering::Relaxed) == 0 {
            return false;
        }
        scope.background_handoff.store(true, Ordering::Relaxed);
        scope.closeout_ended_unmet.store(false, Ordering::Relaxed);
        let ack = scope
            .background_ack
            .lock()
            .ok()
            .and_then(|ack| ack.clone())
            .unwrap_or_else(|| "The task is running in the background.".to_string());
        scope.verdict.record_handoff_reply(ack);
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            scope.model_calls.load(Ordering::Relaxed) as usize,
            aidaemon_events::payloads::DecisionType::PostExecutionValidation,
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

    /// Hold a stop while a job this run started is still running with
    /// nothing to report its result: the run's end would stop the job, so
    /// a reply now would describe work as running that is about to be
    /// killed (live 2026-11-12: a counting sub-agent answered "at 15 of
    /// 300, still running", its end killed the count, and the chat was told
    /// it finished). Waits, as progress, until those jobs end, at most
    /// `UNREPORTED_JOB_MAX_WAIT` in one run: `None` when nothing was
    /// running, else whether they ended, so the run reads their result (or
    /// stops them and says so) before it answers. Read from the tools' typed
    /// process state, never from the reply.
    pub async fn sdk_wait_for_unreported_jobs(&self, scope: &TurnScope<'_>) -> Option<bool> {
        use crate::turn::closeout::StopDemand;
        if scope.stop_budgets.spent(StopDemand::UnreportedJobWait) {
            return None;
        }
        let mut waited = false;
        let started = std::time::Instant::now();
        let _running = self
            .child_progress()
            .map(aidaemon_channels::owner_wait::ProgressClock::begin);
        loop {
            let mut open = 0usize;
            for tool in self.tools() {
                open += tool.jobs_ended_by_task_end(scope.task_id).await;
            }
            if open == 0 {
                return waited.then_some(true);
            }
            if waited && started.elapsed() >= UNREPORTED_JOB_MAX_WAIT {
                scope.stop_budgets.spend(StopDemand::UnreportedJobWait);
                return Some(false);
            }
            if !waited {
                waited = true;
                self.emit_decision_point(
                    scope.emitter,
                    scope.task_id,
                    scope.model_calls.load(Ordering::Relaxed) as usize,
                    aidaemon_events::payloads::DecisionType::PostExecutionValidation,
                    "Stop held until the run's unreported jobs end".to_string(),
                    serde_json::json!({
                        "condition": "unreported_job_running_at_stop",
                        "open_jobs": open,
                        "engine": "sdk",
                    }),
                )
                .await;
            }
            tokio::time::sleep(UNREPORTED_JOB_POLL).await;
        }
    }

    /// A GUI success claim must not ship while the task's last coordinate
    /// click is unverified: a raw (x, y) click has no element identity, so
    /// only a follow-up observation (which clears the flag) can confirm it
    /// (2026-07-12: an unverified click at a guessed position was reported as
    /// a successful Like). Bounded, so a model that never looks cannot loop.
    pub async fn sdk_gui_click_check(&self, scope: &TurnScope<'_>) -> sdk::StopDecision {
        use crate::turn::closeout::{StopDemand, StopDemandVerdict};
        if self.depth() != 0 {
            return sdk::StopDecision::Accept;
        }
        let unverified = match self.computer_use() {
            Some(computer_use) => {
                computer_use
                    .has_unverified_coordinate_click(scope.task_id)
                    .await
            }
            None => false,
        };
        if !unverified {
            return sdk::StopDecision::Accept;
        }
        let demand = StopDemand::GuiClickVerification;
        if self.sdk_stop_demand(scope, demand).await != StopDemandVerdict::Demand {
            return sdk::StopDecision::Accept;
        }
        self.emit_warning_decision_point(
            scope.emitter,
            scope.task_id,
            scope.model_calls.load(Ordering::Relaxed) as usize,
            aidaemon_events::payloads::DecisionType::PostExecutionValidation,
            "GUI success claim blocked: unverified coordinate click".to_string(),
            serde_json::json!({ "condition": "gui_coordinate_click_unverified", "engine": "sdk" }),
        )
        .await;
        sdk::StopDecision::Continue {
            message: crate::turn::system_directives::SystemDirective::GuiCoordinateClickUnverified
                .render(),
        }
    }

    /// A file more than one agent changed is checked once before the
    /// top-level agent reports: a read or command after the last change,
    /// recorded by the tree, closes it. Children are asked nothing.
    pub async fn sdk_combined_verification_check(
        &self,
        scope: &TurnScope<'_>,
    ) -> sdk::StopDecision {
        use crate::turn::closeout::{StopDemand, StopDemandVerdict};
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
        if self.sdk_stop_demand(scope, demand).await != StopDemandVerdict::Demand {
            return sdk::StopDecision::Accept;
        }
        sdk::StopDecision::Continue {
            message: format!(
                "More than one agent changed these files: {}. Read them or run a check on the \
                 combined result before you report.",
                shared.join(", ")
            ),
        }
    }

    /// Code this run changed is built or tested once before it reports: a
    /// command run after the last change closes it. Bounded, and never
    /// asked of a run that cannot run a command; a run that says why it
    /// could not check stops after the one ask.
    pub async fn sdk_code_check(&self, scope: &TurnScope<'_>) -> sdk::StopDecision {
        use crate::turn::closeout::{StopDemand, StopDemandVerdict};
        let unchecked = scope
            .unchecked_code
            .lock()
            .map(|files| files.clone())
            .unwrap_or_default();
        if unchecked.is_empty() {
            return sdk::StopDecision::Accept;
        }
        let demand = StopDemand::CodeChangeUnchecked;
        if self.sdk_stop_demand(scope, demand).await != StopDemandVerdict::Demand {
            return sdk::StopDecision::Accept;
        }
        const SHOWN: usize = 8;
        let mut files = unchecked
            .iter()
            .take(SHOWN)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        if unchecked.len() > SHOWN {
            files.push_str(&format!(" and {} more", unchecked.len() - SHOWN));
        }
        sdk::StopDecision::Continue {
            message: format!(
                "You changed code ({files}) and ran no command since. Before you report, run \
                 the narrowest check that covers the change (the project's build, type-check \
                 or the relevant tests) and report its result. If no check can run here, say \
                 so and why in your reply."
            ),
        }
    }

    /// A mandate task lead's prose is never its decision: a missing typed
    /// commit gets one bounded correction turn when the run could still make
    /// it, then the run fails closed.
    pub async fn sdk_mandate_commit_check(&self, scope: &TurnScope<'_>) -> sdk::StopDecision {
        use crate::engine::mandate_run::{
            MandateDecisionCommitState, mandate_decision_commit_state,
        };
        use crate::turn::closeout::{StopDemand, StopDemandVerdict};
        let state = match mandate_decision_commit_state(self.host).await {
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
                match self.sdk_stop_demand(scope, demand).await {
                    StopDemandVerdict::Demand => {
                        self.emit_warning_decision_point(
                            scope.emitter,
                            scope.task_id,
                            scope.model_calls.load(Ordering::Relaxed) as usize,
                            aidaemon_events::payloads::DecisionType::PostExecutionValidation,
                            "Blocked mandate prose completion without a durable decision"
                                .to_string(),
                            serde_json::json!({
                                "condition": demand.condition(),
                                "retry": scope.stop_budgets.used(demand),
                                "engine": "sdk",
                            }),
                        )
                        .await;
                        return sdk::StopDecision::Continue {
                            message:
                                crate::turn::system_directives::SystemDirective::MandateDecisionCommitRequired
                                    .render(),
                        };
                    }
                    StopDemandVerdict::Exhausted => {
                        "Mandate deliberator returned without committing an exact durable \
                         decision after its bounded correction turn."
                    }
                    StopDemandVerdict::Unreachable => {
                        "Mandate deliberator returned without a durable decision, and this \
                         run's authority does not admit mandate_review record_decision, so \
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
    /// same authority and visible tools the ledger closeout uses, record the
    /// verdict, and spend one ask of the run's budget when it is `Demand`.
    /// The one place a stop-time ask is judged and counted.
    pub async fn sdk_stop_demand(
        &self,
        scope: &TurnScope<'_>,
        demand: crate::turn::closeout::StopDemand,
    ) -> crate::turn::closeout::StopDemandVerdict {
        let used = scope.stop_budgets.used(demand);
        let authority =
            crate::turn::closeout::ClosingAuthority::from_resolved(&scope.authority.authority)
                .without_tool_demands(scope.contract.forbids_tool_use);
        let visible_tools = closeout_visible_tools(
            self.tools(),
            &scope.mcp_tools,
            scope.user_role,
            scope.owned_memory_tools,
        );
        let find = |name: &str| self.tools().iter().find(|tool| tool.name() == name);
        let verdict = crate::turn::closeout::stop_demand_verdict(
            demand,
            used,
            &authority,
            &visible_tools,
            |name| {
                find(name).map(|tool| crate::turn::closeout::KnownTool {
                    read_only: tool.capabilities().read_only,
                    facts: tool.dispatch_facts(),
                })
            },
            |name, arguments| {
                find(name).is_some_and(|tool| tool.owner_control_plane_call(arguments))
            },
        );
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            scope.model_calls.load(Ordering::Relaxed) as usize,
            aidaemon_events::payloads::DecisionType::PostExecutionValidation,
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
        if verdict == crate::turn::closeout::StopDemandVerdict::Demand {
            scope.stop_budgets.spend(demand);
        }
        verdict
    }

    /// One shared retry for a reply that must not ship (empty, only internal
    /// markers, pasted tool output, or written to a closeout demand that
    /// ended unmet); after it, the finalizer's honest fallback applies.
    pub async fn sdk_reply_check(
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
        if self
            .sdk_stop_demand(scope, crate::turn::closeout::StopDemand::ReplyRepair)
            .await
            != crate::turn::closeout::StopDemandVerdict::Demand
        {
            return sdk::StopDecision::Accept;
        }
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
                crate::turn::system_directives::SystemDirective::EmptyFinalReplyPlainText
            }
            ReplyDefect::Empty => {
                crate::turn::system_directives::SystemDirective::EmptyResponseRetry
            }
            ReplyDefect::InternalMarkers => {
                crate::turn::system_directives::SystemDirective::FinalAnswerRejectedInternalMarkers
            }
            ReplyDefect::ToolOutputPaste => {
                crate::turn::system_directives::SystemDirective::FinalAnswerWasFilePaste
            }
            // The work is over: the rewrite answers the user from what the
            // run already holds, and an empty rewrite ships this reply.
            ReplyDefect::AnswersUnmetDemand => {
                scope
                    .verdict
                    .record_interrupted_answer(final_text.to_string());
                scope.text_only_next_call.store(true, Ordering::Relaxed);
                crate::turn::system_directives::SystemDirective::FinalAnswerAfterUnmetCloseout
            }
        };
        sdk::StopDecision::Continue {
            message: directive.render(),
        }
    }

    /// Send a run that declared work and attempted none of it back once:
    /// do the work, or ask the owner the one question it waits on. Read from
    /// the ledger's typed obligations and receipts, never from the reply.
    async fn sdk_demand_unattempted_work(
        &self,
        scope: &TurnScope<'_>,
        aggregate: &aidaemon_events::run_aggregate::RunAggregate,
        iteration: usize,
    ) -> sdk::StopDecision {
        scope
            .stop_budgets
            .spend(crate::turn::closeout::StopDemand::UnattemptedWork);
        scope.closeout_ended_unmet.store(false, Ordering::Relaxed);
        let declared = aggregate.unattempted_declared_work();
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            iteration,
            aidaemon_events::payloads::DecisionType::PostExecutionValidation,
            "Declared work nothing took on: one pass to do it or ask the owner".to_string(),
            serde_json::json!({
                "condition": "declared_work_unattempted",
                "claimed_outcome": aggregate.executor_claimed_outcome.map(|claim| claim.as_str()),
                "obligation_ids": declared,
                "engine": "sdk",
            }),
        )
        .await;
        sdk::StopDecision::Continue {
            message: crate::turn::system_directives::SystemDirective::DeclaredWorkUnattempted {
                items: aggregate.expectation_summaries(&declared),
            }
            .render(),
        }
    }

    /// Judge declared work nothing took on when the turn is about to hand
    /// off to a background job (`launch`: the launch making the handoff now,
    /// its receipt not yet in the ledger, and the items it names). `None`:
    /// nothing to hold. `Continue`: the one pass to launch it or ask.
    /// `AcceptUnmet`: the pass was spent, and the items end "not attempted".
    ///
    /// `continuation_follows`: the job reports back, and its continuation
    /// carries the request's remaining steps from its result. Past the
    /// pass, nothing is then recorded "not attempted": a step that comes
    /// after the job (commit once the tests pass) is not yet due (live
    /// 2026-11-15: a handoff's reply said "I didn't start: Commit").
    pub async fn sdk_judge_unattempted_work(
        &self,
        scope: &TurnScope<'_>,
        launch: Option<(&str, &[usize], usize)>,
        continuation_follows: bool,
    ) -> Option<sdk::StopDecision> {
        let aggregate = self
            .events()
            .task_run_aggregate(scope.session_id, scope.task_id)
            .await
            .ok()?;
        let aggregate = match launch {
            Some((call_id, items, runs)) => aggregate.with_pending_launch(call_id, items, runs),
            None => aggregate,
        };
        let changes_permitted =
            !scope.contract.forbids_mutation && !scope.contract.forbids_tool_use;
        match unattempted_work_verdict(
            aggregate.declared_work_unattempted(),
            aggregate.executor_claimed_outcome,
            true,
            changes_permitted,
            scope
                .stop_budgets
                .spent(crate::turn::closeout::StopDemand::UnattemptedWork),
        ) {
            UnattemptedWorkVerdict::NotApplicable => None,
            UnattemptedWorkVerdict::Demand => {
                let iteration = scope.model_calls.load(Ordering::Relaxed) as usize;
                Some(
                    self.sdk_demand_unattempted_work(scope, &aggregate, iteration)
                        .await,
                )
            }
            UnattemptedWorkVerdict::StopUnmet if continuation_follows => None,
            UnattemptedWorkVerdict::StopUnmet => Some(sdk::StopDecision::AcceptUnmet {
                unmet: record_not_attempted(scope, &aggregate),
            }),
        }
    }

    /// The ledger-first closeout: replay the run
    /// ledger, ask which expectations an admissible tool could still satisfy,
    /// and demand a bounded extra pass only when the receipts say so.
    pub async fn sdk_closeout(
        &self,
        scope: &TurnScope<'_>,
        completed_results: usize,
        failed_without_success: bool,
    ) -> sdk::StopDecision {
        let Ok(aggregate) = self
            .events()
            .task_run_aggregate(scope.session_id, scope.task_id)
            .await
        else {
            return sdk::StopDecision::Accept;
        };
        let iteration = scope.model_calls.load(Ordering::Relaxed) as usize;
        let work_unattempted = aggregate.declared_work_unattempted();
        // What the closeout judged a stop on, so an early stop that went
        // unheld can be traced (live 2026-11-15: a fallback model's cut-off
        // "I'll check both…" on a change request ended the run).
        tracing::info!(
            task_id = scope.task_id,
            owner_units = aggregate.owner_units.len(),
            work_unattempted,
            claimed = ?aggregate.executor_claimed_outcome,
            mutating_operations = aggregate.operations.values().filter(|op| op.mutating).count(),
            failed_without_success,
            "Closeout judging a stop"
        );
        let changes_permitted =
            !scope.contract.forbids_mutation && !scope.contract.forbids_tool_use;
        // The agent's own typed verdict: `partial`, `blocked` or
        // `needs_owner` ends the run unmet, and nothing later turns it into
        // a success. No more work is demanded (the agent said it cannot or
        // will not finish, or asked the owner), unless it attempted none of
        // the work it declared with nothing in its way: that run goes back
        // once to do the work or ask.
        if let Some(claim) = aggregate
            .executor_claimed_outcome
            .filter(|claim| claim.unfinished())
        {
            let unattempted = unattempted_work_verdict(
                work_unattempted,
                Some(claim),
                true,
                changes_permitted,
                scope
                    .stop_budgets
                    .spent(crate::turn::closeout::StopDemand::UnattemptedWork),
            );
            if unattempted == UnattemptedWorkVerdict::Demand {
                return self
                    .sdk_demand_unattempted_work(scope, &aggregate, iteration)
                    .await;
            }
            // `partial` with nothing typed in its way and declared items
            // still open: it goes on once, or says what stops it (live
            // 2026-11-15: a coding run stopped mid-change after two edits,
            // "the edit handler is currently incomplete").
            // Its own list says what is still open.
            let open_items = scope.open_checklist_items.load(Ordering::Relaxed) as usize;
            if claim == aidaemon_events::ExecutorClaimedOutcome::Partial
                && open_items > 0
                && changes_permitted
                && !failed_without_success
                && aggregate.executor_short_of.is_none()
                && !aggregate.terminal_policy_denial()
                && !aggregate.has_background_in_flight()
                && scope
                    .stop_budgets
                    .try_spend(crate::turn::closeout::StopDemand::PartialWithoutObstacle)
            {
                self.emit_decision_point(
                    scope.emitter,
                    scope.task_id,
                    iteration,
                    aidaemon_events::payloads::DecisionType::PostExecutionValidation,
                    "Partial verdict with nothing in the way: one more pass".to_string(),
                    serde_json::json!({
                        "condition": "partial_without_obstacle",
                        "open_items": open_items,
                        "engine": "sdk",
                    }),
                )
                .await;
                return sdk::StopDecision::Continue {
                    message:
                        crate::turn::system_directives::SystemDirective::PartialWithoutObstacle {
                            open_items,
                        }
                        .render(),
                };
            }
            scope.closeout_ended_unmet.store(false, Ordering::Relaxed);
            self.emit_decision_point(
                scope.emitter,
                scope.task_id,
                iteration,
                aidaemon_events::payloads::DecisionType::PostExecutionValidation,
                format!("Agent reported the request as {}", claim.as_str()),
                serde_json::json!({
                    "condition": "executor_claimed_outcome",
                    "claimed_outcome": claim.as_str(),
                    "work_unattempted": work_unattempted,
                    "engine": "sdk",
                }),
            )
            .await;
            let mut unmet = vec![claim.unmet_line()];
            if unattempted == UnattemptedWorkVerdict::StopUnmet {
                unmet.extend(record_not_attempted(scope, &aggregate));
            }
            return sdk::StopDecision::AcceptUnmet { unmet };
        }
        let authority =
            crate::turn::closeout::ClosingAuthority::from_resolved(&scope.authority.authority)
                .without_tool_demands(scope.contract.forbids_tool_use);
        let visible_tools = closeout_visible_tools(
            self.tools(),
            &scope.mcp_tools,
            scope.user_role,
            scope.owned_memory_tools,
        );
        let known = |name: &str| {
            self.tools()
                .iter()
                .find(|tool| tool.name() == name)
                .map(|tool| crate::turn::closeout::KnownTool {
                    read_only: tool.capabilities().read_only,
                    facts: tool.dispatch_facts(),
                })
        };
        let decision = aggregate.closeout(|obligation| {
            crate::turn::closeout::obligation_admissible(
                obligation,
                &authority,
                &visible_tools,
                known,
            )
        });
        let reachable = match &decision {
            aidaemon_events::run_aggregate::CloseoutDecision::Reachable { obligation_ids } => {
                obligation_ids.clone()
            }
            _ => Vec::new(),
        };
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            iteration,
            aidaemon_events::payloads::DecisionType::PostExecutionValidation,
            format!("Ledger closeout verdict: {}", decision.verdict()),
            serde_json::json!({
                "condition": "ledger_closeout",
                "verdict": decision.verdict(),
                "proof_basis": match &decision {
                    aidaemon_events::run_aggregate::CloseoutDecision::Closed { proof_basis } => Some(*proof_basis),
                    _ => None,
                },
                "reachable_obligation_ids": reachable,
                "authoritative": true,
                "engine": "sdk",
            }),
        )
        .await;
        // Every declared step deferred, skipped or dropped, with none of
        // the work attempted: the same stop as an unfinished verdict. A
        // change the owner asked for that nothing attempted is judged now,
        // whatever else is reachable: no observation the ledger may still
        // demand does it.
        let owner_units_uncovered = !aggregate.uncovered_owner_units().is_empty();
        match unattempted_work_verdict(
            work_unattempted,
            aggregate.executor_claimed_outcome,
            !decision.work_reachable() || owner_units_uncovered,
            changes_permitted,
            scope
                .stop_budgets
                .spent(crate::turn::closeout::StopDemand::UnattemptedWork),
        ) {
            UnattemptedWorkVerdict::Demand => {
                return self
                    .sdk_demand_unattempted_work(scope, &aggregate, iteration)
                    .await;
            }
            UnattemptedWorkVerdict::StopUnmet => {
                scope.closeout_ended_unmet.store(false, Ordering::Relaxed);
                return sdk::StopDecision::AcceptUnmet {
                    unmet: record_not_attempted(scope, &aggregate),
                };
            }
            UnattemptedWorkVerdict::NotApplicable => {}
        }
        let demanded_at = scope.executor_demanded_at.load(Ordering::Relaxed);
        // Delivering a capture to the chat changes nothing an observation
        // could verify, so it does not earn the verify-once pass.
        let mutation_succeeded = aggregate.operations.values().any(|operation| {
            operation.changes_verifiable_state()
                && operation.outcome
                    == Some(aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded)
        });
        let observation_reachable = reachable.iter().any(|id| {
            aggregate.obligations.get(id).is_some_and(|obligation| {
                obligation.class == aidaemon_events::run_aggregate::RunObligationClass::Observe
            })
        });
        let verification_after_mutation = mutation_succeeded
            && observation_reachable
            && !scope
                .stop_budgets
                .spent(crate::turn::closeout::StopDemand::VerifyAfterMutation);
        // A failure that is itself the expected observation satisfies its
        // obligation; only an uncredited failure earns the recovery pass.
        let failure_credited = aggregate.obligations.values().any(|obligation| {
            obligation.state == aidaemon_events::run_aggregate::RunObligationState::Satisfied
        });
        let unrecovered_failure = failed_without_success
            && !failure_credited
            && !scope
                .stop_budgets
                .spent(crate::turn::closeout::StopDemand::FailureRecovery);
        let facts = StopFacts {
            reachable_obligation_ids: reachable.clone(),
            completed_results,
            executor_demanded_at: (demanded_at != u64::MAX).then_some(demanded_at as usize),
            terminal_policy_denial: aggregate.terminal_policy_denial(),
            verification_after_mutation,
            unrecovered_failure,
        };
        if !closeout_demands_work(&facts)
            || scope
                .stop_budgets
                .spent(crate::turn::closeout::StopDemand::LedgerExpectations)
        {
            scope.closeout_ended_unmet.store(
                closeout_ended_unmet(
                    &facts,
                    scope
                        .stop_budgets
                        .used(crate::turn::closeout::StopDemand::LedgerExpectations),
                ) && !scope.user_spoke_since_demand.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
            return sdk::StopDecision::Accept;
        }
        scope.closeout_ended_unmet.store(false, Ordering::Relaxed);
        scope
            .user_spoke_since_demand
            .store(false, Ordering::Relaxed);
        let demands = scope
            .stop_budgets
            .spend(crate::turn::closeout::StopDemand::LedgerExpectations);
        if verification_after_mutation {
            scope
                .stop_budgets
                .spend(crate::turn::closeout::StopDemand::VerifyAfterMutation);
        }
        if unrecovered_failure && reachable.is_empty() {
            scope
                .stop_budgets
                .spend(crate::turn::closeout::StopDemand::FailureRecovery);
            let evidence = scope
                .latest_tool_output
                .lock()
                .ok()
                .and_then(|output| output.clone())
                .unwrap_or_default();
            return sdk::StopDecision::Continue {
                message:
                    crate::turn::system_directives::SystemDirective::ExternalMutationRecoveryRequired(
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
            aidaemon_events::payloads::DecisionType::IntentGate,
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
            message: crate::turn::system_directives::SystemDirective::LedgerExpectationsRequired {
                expectations: aggregate.expectation_summaries(&reachable),
            }
            .render(),
        }
    }
}
