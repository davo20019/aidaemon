//! Token budgets charged per model call: task, scheduled-run and goal
//! daily budgets, with extensions on progress.

use super::*;

pub(super) const GOAL_BUDGET_EXHAUSTED: &str = "the goal's token budget is used up";
/// How often, and how far, a used-up budget may extend.
pub(super) const TASK_MAX_BUDGET_EXTENSIONS: usize = 3;
pub(super) const TASK_HARD_TOKEN_CAP: u64 = 2_000_000;
pub(super) const SCHEDULED_MAX_BUDGET_EXTENSIONS: usize = 1;
pub(super) const SCHEDULED_HARD_TOKEN_CAP: i64 = 2_000_000;
pub(super) const MANDATE_BUDGET_EXHAUSTED: &str = "the mandate cycle's token balance is exhausted";

/// The task budget's one 80% warning is due: past 80%, not yet exhausted,
/// not already given.
pub(super) fn task_budget_warning_due(used: u64, budget: u64, already_sent: bool) -> bool {
    budget > 0 && !already_sent && used >= budget.saturating_mul(80) / 100 && used < budget
}

/// The run's task token budget (0 means none). A scheduled run with a
/// per-check budget is bounded by that run budget (with its own extension
/// and alert), so the task budget does not stack on top of it. Without one,
/// the configured task budget applies.
pub(super) fn task_token_budget(configured: Option<u64>, scheduled_per_check: Option<i64>) -> u64 {
    match scheduled_per_check.filter(|budget| *budget > 0) {
        Some(_) => 0,
        None => configured.unwrap_or(0),
    }
}

impl Agent {
    /// Charge a model call to the task, goal-daily and scheduled-run
    /// budgets. When one is used up, it extends once more if receipts show
    /// progress since the last extension (or the owner approves); otherwise
    /// the owner is alerted and the
    /// honest budget reply is returned.
    pub(super) async fn sdk_charge_budgets(
        &self,
        scope: &TurnScope<'_>,
        usage: &crate::traits::TokenUsage,
        iteration: usize,
    ) -> Option<String> {
        let delta = usage.budget_tokens();
        let used = scope.task_tokens.fetch_add(delta, Ordering::Relaxed) + delta;
        let successes = scope.succeeded_calls.load(Ordering::Relaxed);
        let learning = scope
            .learning
            .lock()
            .map(|learning| learning.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
        let mut directives = Vec::new();
        let stop = if let Some(stop) = self
            .sdk_task_budget(
                scope,
                used,
                successes,
                iteration,
                &learning,
                &mut directives,
            )
            .await
        {
            Some(stop)
        } else if let Some(goal) = &scope.budget_goal {
            self.sdk_goal_budget(
                scope,
                goal,
                delta as i64,
                successes,
                iteration,
                &learning,
                &mut directives,
            )
            .await
        } else {
            None
        };
        // One warning at 80% of the task budget (scheduled runs warn on their
        // own per-run budget instead).
        let budget = scope.task_budget.load(Ordering::Relaxed);
        let scheduled = scope
            .budget_goal
            .as_ref()
            .is_some_and(|goal| goal.scheduled);
        if stop.is_none()
            && !scheduled
            && task_budget_warning_due(
                used,
                budget,
                scope.budget_warning_sent.load(Ordering::Relaxed),
            )
        {
            scope.budget_warning_sent.store(true, Ordering::Relaxed);
            let hint: String = scope.user_text.chars().take(150).collect();
            directives.push(
                crate::agent::system_directives::SystemDirective::TaskTokenBudgetWarning {
                    used,
                    budget,
                    pct: used.saturating_mul(100) / budget,
                    task_anchor: if hint.trim().is_empty() {
                        String::new()
                    } else {
                        format!(" Current task: {}", hint.trim())
                    },
                },
            );
        }
        if let Ok(mut extra) = scope.extra_system.lock() {
            extra.extend(directives.iter().map(|directive| directive.render()));
        }
        stop
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn sdk_task_budget(
        &self,
        scope: &TurnScope<'_>,
        used: u64,
        successes: usize,
        iteration: usize,
        learning: &crate::agent::post_task::LearningContext,
        directives: &mut Vec<crate::agent::system_directives::SystemDirective>,
    ) -> Option<String> {
        let budget = scope.task_budget.load(Ordering::Relaxed);
        if budget == 0 || used < budget {
            return None;
        }
        let cap = TASK_HARD_TOKEN_CAP;
        let proposed = budget
            .saturating_mul(2)
            .max(used.saturating_add(budget / 2))
            .min(cap);
        let extensions = scope.task_budget_extensions.load(Ordering::Relaxed);
        let progressed = successes > scope.successes_at_task_extension.load(Ordering::Relaxed)
            && !scope.repeating.load(Ordering::Relaxed);
        let can_extend = budget < cap && proposed > used;
        let approved = can_extend
            && if extensions < TASK_MAX_BUDGET_EXTENSIONS
                && progressed
                && scope.user_role == UserRole::Owner
            {
                true
            } else {
                self.request_budget_continue_approval(
                    scope.emitter,
                    scope.task_id,
                    iteration,
                    scope.session_id,
                    scope.user_role,
                    "task",
                    used as i64,
                    budget as i64,
                    proposed as i64,
                )
                .await
            };
        if approved {
            let extension = extensions + 1;
            scope.task_budget.store(proposed, Ordering::Relaxed);
            scope
                .task_budget_extensions
                .store(extension, Ordering::Relaxed);
            scope
                .successes_at_task_extension
                .store(successes, Ordering::Relaxed);
            self.with_harness_eval(|eval| eval.record_budget_extension())
                .await;
            directives.push(
                crate::agent::system_directives::SystemDirective::TaskBudgetAutoExtended {
                    old_budget: budget as i64,
                    new_budget: proposed as i64,
                    extension,
                    max_extensions: TASK_MAX_BUDGET_EXTENSIONS,
                },
            );
            if let Some(status_tx) = &scope.status_tx {
                let _ = status_tx.try_send(StatusUpdate::BudgetExtended {
                    old_budget: budget as i64,
                    new_budget: proposed as i64,
                    extension,
                    max_extensions: TASK_MAX_BUDGET_EXTENSIONS,
                });
            }
            info!(
                task_id = scope.task_id,
                old_budget = budget,
                new_budget = proposed,
                extension,
                "SDK engine: extended the task token budget"
            );
            return None;
        }
        let alert = format!(
            "Token alert: execution in session '{}' hit task token budget (used {} / limit {}). \
             The run was stopped to prevent overspending.",
            scope.session_id, used, budget
        );
        self.fanout_token_alert(None, scope.session_id, &alert, Some(scope.session_id))
            .await;
        Some(crate::agent::post_task::graceful_budget_response(
            learning, used,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn sdk_goal_budget(
        &self,
        scope: &TurnScope<'_>,
        goal: &BudgetGoal,
        delta: i64,
        successes: usize,
        iteration: usize,
        learning: &crate::agent::post_task::LearningContext,
        directives: &mut Vec<crate::agent::system_directives::SystemDirective>,
    ) -> Option<String> {
        let status = match self
            .state
            .add_goal_tokens_and_get_budget_status(&goal.goal_id, delta)
            .await
        {
            Ok(Some(status)) => status,
            Ok(None) => return None,
            Err(error) => {
                warn!(goal_id = %goal.goal_id, %error, "SDK engine: goal token accounting failed");
                return None;
            }
        };
        let metrics = || crate::agent::graceful::ScheduledRunActivityMetrics {
            evidence_gain_count: successes,
            stall_count: usize::from(scope.repeating.load(Ordering::Relaxed)),
            consecutive_same_tool_count: 0,
            consecutive_same_tool_unique_args: 0,
            total_successful_tool_calls: successes,
        };
        if goal.scheduled {
            let registry = self.goal_token_registry.as_ref()?;
            registry.add_run_tokens(&goal.goal_id, delta).await;
            let progress = scope
                .progress
                .lock()
                .ok()
                .map(|progress| progress.clone())?;
            let run_status = registry
                .update_run_health(
                    &goal.goal_id,
                    Agent::scheduled_run_health_snapshot(
                        learning,
                        metrics(),
                        scope.contract,
                        &progress,
                    ),
                )
                .await?;
            crate::agent::goal_dispatch::persist_scheduled_run_state(
                &self.state,
                &goal.goal_id,
                None,
                &run_status,
            )
            .await;
            // One warning at 80% of the per-run budget, so the run can wrap
            // up before it is stopped.
            if let Some(pct) = Agent::scheduled_run_budget_pressure_pct(
                &run_status,
                scope.budget_warning_sent.load(Ordering::Relaxed),
            ) {
                scope.budget_warning_sent.store(true, Ordering::Relaxed);
                directives.push(
                    crate::agent::system_directives::SystemDirective::ScheduledRunBudgetPressure {
                        used: run_status.tokens_used,
                        budget: run_status.effective_budget_per_check,
                        pct,
                    },
                );
                self.emit_warning_decision_point(
                    scope.emitter,
                    scope.task_id,
                    iteration,
                    crate::events::DecisionType::StoppingCondition,
                    "Scheduled run resource-pressure threshold reached".to_string(),
                    serde_json::json!({
                        "condition": "scheduled_run_budget_pressure",
                        "goal_id": goal.goal_id,
                        "budget": run_status.effective_budget_per_check,
                        "tokens_used": run_status.tokens_used,
                        "pct": pct,
                        "engine": "sdk",
                    }),
                )
                .await;
            }
            let mut ctx = crate::agent::graceful::ScheduledRunBudgetControlCtx {
                emitter: scope.emitter,
                task_id: scope.task_id,
                session_id: scope.session_id,
                iteration,
                goal_id: &goal.goal_id,
                status: &run_status,
                pending_system_messages: directives,
                max_budget_extensions: SCHEDULED_MAX_BUDGET_EXTENSIONS,
                hard_token_cap: SCHEDULED_HARD_TOKEN_CAP,
            };
            if let crate::agent::graceful::ScheduledRunBudgetControlOutcome::Exhausted {
                tokens_used,
                budget_per_check,
            } = self.enforce_scheduled_run_budget_control(&mut ctx).await
            {
                let alert = format!(
                    "Token alert: scheduled run for goal '{}' hit its per-run budget (used {} / \
                     limit {}). The run stopped safely before completion.",
                    goal.goal_id, tokens_used, budget_per_check
                );
                self.fanout_token_alert(
                    Some(goal.goal_id.as_str()),
                    scope.session_id,
                    &alert,
                    Some(scope.session_id),
                )
                .await;
                return Some(
                    crate::agent::post_task::graceful_scheduled_run_budget_response(
                        learning,
                        tokens_used,
                        budget_per_check,
                    ),
                );
            }
            return None;
        }
        let mut effective_daily = scope
            .goal_daily_budget
            .lock()
            .map(|budget| *budget)
            .unwrap_or(None);
        let mut extensions = scope.goal_budget_extensions.load(Ordering::Relaxed) as usize;
        let outcome = {
            let mut ctx = crate::agent::graceful::GoalBudgetControlCtx {
                emitter: scope.emitter,
                task_id: scope.task_id,
                session_id: scope.session_id,
                iteration,
                goal_id: &goal.goal_id,
                status: &status,
                user_role: scope.user_role,
                learning_ctx: learning,
                evidence_gain_count: successes,
                stall_count: usize::from(scope.repeating.load(Ordering::Relaxed)),
                consecutive_same_tool_count: 0,
                consecutive_same_tool_unique_args: 0,
                total_successful_tool_calls: successes,
                pending_system_messages: directives,
                status_tx: &scope.status_tx,
                is_scheduled_goal: false,
                effective_goal_daily_budget: &mut effective_daily,
                budget_extensions_count: &mut extensions,
                max_budget_extensions: TASK_MAX_BUDGET_EXTENSIONS,
                hard_token_cap: TASK_HARD_TOKEN_CAP as i64,
                source: crate::agent::graceful::GoalBudgetCheckSource::PostLlm,
            };
            self.enforce_goal_daily_budget_control(&mut ctx).await
        };
        if let Ok(mut budget) = scope.goal_daily_budget.lock() {
            *budget = effective_daily;
        }
        scope
            .goal_budget_extensions
            .store(extensions as u32, Ordering::Relaxed);
        if let crate::agent::graceful::GoalBudgetControlOutcome::Exhausted {
            tokens_used_today,
            budget_daily,
        } = outcome
        {
            let alert = format!(
                "Token alert: goal '{}' hit daily token budget (used {} / limit {}). Execution \
                 was stopped to prevent overspending.",
                goal.goal_id, tokens_used_today, budget_daily
            );
            self.fanout_token_alert(
                Some(goal.goal_id.as_str()),
                scope.session_id,
                &alert,
                Some(scope.session_id),
            )
            .await;
            return Some(
                crate::agent::post_task::graceful_goal_daily_budget_response(
                    learning,
                    tokens_used_today,
                    budget_daily,
                    false,
                ),
            );
        }
        None
    }
}
