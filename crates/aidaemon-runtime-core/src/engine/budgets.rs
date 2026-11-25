//! Token budget limits and the pure checks the host's budget gate reads.

pub const GOAL_BUDGET_EXHAUSTED: &str = "the goal's token budget is used up";
/// How often, and how far, a used-up budget may extend.
pub const TASK_MAX_BUDGET_EXTENSIONS: usize = 3;
pub const TASK_HARD_TOKEN_CAP: u64 = 2_000_000;
pub const SCHEDULED_MAX_BUDGET_EXTENSIONS: usize = 1;
pub const SCHEDULED_HARD_TOKEN_CAP: i64 = 2_000_000;
pub const MANDATE_BUDGET_EXHAUSTED: &str = "the mandate cycle's token balance is exhausted";

/// The task budget's one 80% warning is due: past 80%, not yet exhausted,
/// not already given.
pub fn task_budget_warning_due(used: u64, budget: u64, already_sent: bool) -> bool {
    budget > 0 && !already_sent && used >= budget.saturating_mul(80) / 100 && used < budget
}

/// The run's task token budget (0 means none). A scheduled run with a
/// per-check budget is bounded by that run budget (with its own extension
/// and alert), so the task budget does not stack on top of it. Without one,
/// the configured task budget applies.
pub fn task_token_budget(configured: Option<u64>, scheduled_per_check: Option<i64>) -> u64 {
    match scheduled_per_check.filter(|budget| *budget > 0) {
        Some(_) => 0,
        None => configured.unwrap_or(0),
    }
}

/// One request's task token budget: the whole tree's, the turn and every
/// plain sub-agent it starts. Its tokens, its limit, its extensions and its
/// progress are one record, so the tree is charged once and extended once
/// (live 2026-11-13: two helpers charged the tree's shared counter but each
/// held its own 500,000 limit and its own extension count, so each
/// compared the tree's 513,391 tokens with its own limit, one extended on
/// its own and the other asked the owner mid-run).
#[derive(Debug, Default)]
pub struct TreeBudget {
    tokens: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// 0: none. Set by the tree's first turn ([`Self::start`]).
    limit: std::sync::atomic::AtomicU64,
    started: std::sync::atomic::AtomicBool,
    /// Distinct successful operations across the tree: its progress.
    successes: std::sync::atomic::AtomicUsize,
    /// The same, per member (the turn and each helper, by session): what a
    /// member that is repeating itself compares with the rest of the tree.
    by_member: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    /// Each member's count at the last extension.
    by_member_at_last: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    warned: std::sync::atomic::AtomicBool,
    /// Extensions so far, and the progress at the last one. Held while one
    /// member decides, so a second member that hit the same limit waits for
    /// that decision instead of making its own.
    extensions: tokio::sync::Mutex<Extensions>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Extensions {
    pub count: usize,
    pub successes_at_last: usize,
}

impl TreeBudget {
    /// The tree's budget as its first turn sets it; later members keep it.
    pub fn start(&self, limit: u64) {
        if !self.started.swap(true, std::sync::atomic::Ordering::SeqCst) {
            self.limit.store(limit, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// The tree's token counter.
    pub fn tokens(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.tokens.clone()
    }

    /// Charge `delta` tokens; the tree's total after it.
    pub fn charge(&self, delta: u64) -> u64 {
        self.tokens
            .fetch_add(delta, std::sync::atomic::Ordering::Relaxed)
            + delta
    }

    pub fn used(&self) -> u64 {
        self.tokens.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn limit(&self) -> u64 {
        self.limit.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Member `member` made a new distinct successful operation.
    pub fn note_success(&self, member: &str) {
        self.successes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut by_member) = self.by_member.lock() {
            *by_member.entry(member.to_string()).or_default() += 1;
        }
    }

    /// Whether the tree is making progress for `member`, whose latest call
    /// repeated an earlier one when `repeating`. The budget is the tree's,
    /// so its progress is the tree's: something new succeeded anywhere in it
    /// since the last extension. Only a member going round in circles needs
    /// the rest of the tree to have done new work since then: a helper
    /// checking on its running command while its sibling works is waiting,
    /// not stuck, whichever of them did the very latest new thing (live
    /// 2026-11-14: asked at 500,044 of 500,000 tokens, "Nothing new has
    /// succeeded", while two helpers were installing a site each).
    pub fn progressed(&self, extensions: &Extensions, member: &str, repeating: bool) -> bool {
        let new_in_tree = self
            .successes()
            .saturating_sub(extensions.successes_at_last);
        if !repeating {
            return new_in_tree > 0;
        }
        let count = |map: &std::sync::Mutex<std::collections::HashMap<String, usize>>| {
            map.lock()
                .ok()
                .and_then(|map| map.get(member).copied())
                .unwrap_or(0)
        };
        let own_new = count(&self.by_member).saturating_sub(count(&self.by_member_at_last));
        new_in_tree > own_new
    }

    pub fn successes(&self) -> usize {
        self.successes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The tree's one 80% warning: `true` once, for the member that gives it.
    pub fn take_warning(&self) -> bool {
        let (used, limit) = (self.used(), self.limit());
        task_budget_warning_due(
            used,
            limit,
            self.warned.load(std::sync::atomic::Ordering::Relaxed),
        ) && !self.warned.swap(true, std::sync::atomic::Ordering::Relaxed)
    }

    /// Decide over the tree's extensions, one member at a time.
    pub async fn deciding(&self) -> tokio::sync::MutexGuard<'_, Extensions> {
        self.extensions.lock().await
    }

    /// Raise the limit, recording the extension and the progress at it.
    pub fn extend(&self, to: u64, extensions: &mut Extensions) {
        self.limit.store(to, std::sync::atomic::Ordering::SeqCst);
        extensions.count += 1;
        extensions.successes_at_last = self.successes();
        if let (Ok(now), Ok(mut at_last)) = (self.by_member.lock(), self.by_member_at_last.lock()) {
            *at_last = now.clone();
        }
    }
}

/// What a used-up task budget does next, under the configured
/// auto-extension policy: up to [`TASK_MAX_BUDGET_EXTENSIONS`] extensions,
/// never past [`TASK_HARD_TOKEN_CAP`], each while the tree made new
/// progress since the last.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetStep {
    /// Within the policy: extend without asking.
    Extend { to: u64 },
    /// Past the policy: more spending is the owner's decision.
    Ask { to: u64, why: AskWhy },
    /// At the hard cap: nothing can extend it.
    Stop,
}

/// Why spending more is the owner's decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AskWhy {
    /// The automatic extensions the policy allows are used up.
    ExtensionsUsed(usize),
    /// Nothing new succeeded anywhere in the request since the budget last
    /// grew.
    NoProgress,
    /// The request is not the owner's own.
    NotOwners,
}

impl AskWhy {
    /// Why the owner is asked, in plain words.
    pub fn words(self) -> String {
        match self {
            Self::ExtensionsUsed(count) => format!(
                "It already grew {count} time{} on its own, the most the budget policy allows.",
                if count == 1 { "" } else { "s" }
            ),
            Self::NoProgress => {
                "Nothing new has succeeded since the budget last grew, so it may be stuck."
                    .to_string()
            }
            Self::NotOwners => "This request did not come from you.".to_string(),
        }
    }
}

/// The policy for a used-up task budget of `limit` at `used` tokens, given
/// whether the tree made progress since its last extension
/// ([`TreeBudget::progressed`]).
pub fn task_budget_step(
    used: u64,
    limit: u64,
    extensions: Extensions,
    progressed: bool,
    owners: bool,
) -> BudgetStep {
    let cap = TASK_HARD_TOKEN_CAP;
    let to = limit
        .saturating_mul(2)
        .max(used.saturating_add(limit / 2))
        .min(cap);
    if limit >= cap || to <= used {
        return BudgetStep::Stop;
    }
    let why = if !owners {
        AskWhy::NotOwners
    } else if extensions.count >= TASK_MAX_BUDGET_EXTENSIONS {
        AskWhy::ExtensionsUsed(extensions.count)
    } else if !progressed {
        AskWhy::NoProgress
    } else {
        return BudgetStep::Extend { to };
    };
    BudgetStep::Ask { to, why }
}

// ---- Charging a model call to the run's budgets ----

use std::sync::atomic::Ordering;

use aidaemon_core as sdk;
use aidaemon_events::TaskOutcome;
use aidaemon_events::conversation::Message;
use aidaemon_tools::semantics::UserRole;
use tracing::{info, warn};
use uuid::Uuid;

use crate::engine::calls::BudgetCheck;
use crate::engine::scope::{BudgetGoal, TurnScope};
use crate::host::TurnHost;
use crate::turn::turn_context::TurnHistory;

/// Charge the latest response to the budgets, and record its tool calls
/// only if the run will act on them: a stopped response must leave no
/// call without a result in the history.
pub async fn serve_sdk_budget(host: &dyn TurnHost, scope: &TurnScope<'_>, check: BudgetCheck) {
    let usage = scope
        .pending_charge
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    let iteration = scope.model_calls.load(Ordering::Relaxed) as usize;
    let mut decision = sdk::ModelDecision::Accept;
    if let Some(usage) = usage
        && let Some(stop) = sdk_charge_budgets(host, scope, &usage, iteration).await
    {
        // Closeout grace: a final answer on the first call, or after
        // real progress, still ships; otherwise the budget reply
        // replaces it and no further work runs.
        let grace = check.response.tool_calls.is_empty()
            && (iteration == 1 || scope.succeeded_calls.load(Ordering::Relaxed) > 0);
        if !grace {
            scope.verdict.record_budget_stop(stop, TaskOutcome::Failed);
            decision = sdk::ModelDecision::Stop {
                reason: GOAL_BUDGET_EXHAUSTED.to_string(),
            };
        }
    }
    if decision == sdk::ModelDecision::Accept && !check.response.tool_calls.is_empty() {
        persist_sdk_tool_call_message(host, scope, &check.response).await;
    }
    let _ = check.reply.send(decision);
}

/// Charge a model call to the task, goal-daily and scheduled-run
/// budgets. When one is used up, it extends once more if receipts show
/// progress since the last extension (or the owner approves); otherwise
/// the owner is alerted and the
/// honest budget reply is returned.
pub async fn sdk_charge_budgets(
    host: &dyn TurnHost,
    scope: &TurnScope<'_>,
    usage: &aidaemon_events::provider::TokenUsage,
    iteration: usize,
) -> Option<String> {
    let delta = usage.budget_tokens();
    let used = scope.spawn_family.budget().charge(delta);
    let successes = scope.succeeded_calls.load(Ordering::Relaxed);
    let learning = scope
        .learning
        .lock()
        .map(|learning| learning.clone())
        .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
    let mut directives = Vec::new();
    let stop = if let Some(stop) = sdk_task_budget(
        host,
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
        sdk_goal_budget(
            host,
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
    // One warning at 80% of the request's task budget, to the member that
    // crossed it (scheduled runs warn on their own per-run budget instead).
    let tree = scope.spawn_family.budget();
    let budget = tree.limit();
    let scheduled = scope
        .budget_goal
        .as_ref()
        .is_some_and(|goal| goal.scheduled);
    if stop.is_none() && !scheduled && tree.take_warning() {
        let hint: String = scope.user_text.chars().take(150).collect();
        directives.push(
            crate::turn::system_directives::SystemDirective::TaskTokenBudgetWarning {
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

/// The tree's task budget after a charge: within it, nothing; used up,
/// the policy's step ([`task_budget_step`]). One member decides at a time,
/// so a limit two helpers hit together is extended (or asked about) once.
#[allow(clippy::too_many_arguments)]
pub async fn sdk_task_budget(
    host: &dyn TurnHost,
    scope: &TurnScope<'_>,
    used: u64,
    _successes: usize,
    iteration: usize,
    learning: &crate::turn::post_task::LearningContext,
    directives: &mut Vec<crate::turn::system_directives::SystemDirective>,
) -> Option<String> {
    let tree = scope.spawn_family.budget();
    if tree.limit() == 0 || used < tree.limit() {
        return None;
    }
    let mut extensions = tree.deciding().await;
    // Another member may have extended it meanwhile.
    let (used, budget) = (tree.used(), tree.limit());
    if budget == 0 || used < budget {
        return None;
    }
    let step = task_budget_step(
        used,
        budget,
        *extensions,
        tree.progressed(
            &extensions,
            scope.session_id,
            scope.repeating.load(Ordering::Relaxed),
        ),
        scope.user_role == UserRole::Owner,
    );
    let extended_to = match step {
        BudgetStep::Extend { to } => Some(to),
        BudgetStep::Ask { to, why } => crate::turn::budget::request_budget_continue_approval(
            host,
            scope.emitter,
            scope.task_id,
            iteration,
            scope.session_id,
            scope.user_role,
            crate::turn::budget::BudgetAsk {
                scope_label: "task",
                work: crate::turn::budget::work_words(scope.user_text),
                used: used as i64,
                current: budget as i64,
                proposed: to as i64,
                why: why.words(),
            },
        )
        .await
        .then_some(to),
        BudgetStep::Stop => None,
    };
    if let Some(proposed) = extended_to {
        tree.extend(proposed, &mut extensions);
        let extension = extensions.count;
        host.turn_runtime()
            .with_harness_eval(|eval| eval.record_budget_extension())
            .await;
        directives.push(
            crate::turn::system_directives::SystemDirective::TaskBudgetAutoExtended {
                old_budget: budget as i64,
                new_budget: proposed as i64,
                extension,
                max_extensions: TASK_MAX_BUDGET_EXTENSIONS,
            },
        );
        info!(
            task_id = scope.task_id,
            old_budget = budget,
            new_budget = proposed,
            extension,
            "SDK engine: extended the task token budget"
        );
        return None;
    }
    drop(extensions);
    let alert = format!(
        "Token alert: execution in session '{}' hit task token budget (used {} / limit {}). \
         The run was stopped to prevent overspending.",
        scope.session_id, used, budget
    );
    crate::turn::budget::fanout_token_alert(
        host,
        None,
        scope.session_id,
        &alert,
        Some(scope.session_id),
    )
    .await;
    Some(crate::turn::post_task::graceful_budget_response(
        learning, used,
    ))
}

#[allow(clippy::too_many_arguments)]
pub async fn sdk_goal_budget(
    host: &dyn TurnHost,
    scope: &TurnScope<'_>,
    goal: &BudgetGoal,
    delta: i64,
    successes: usize,
    iteration: usize,
    learning: &crate::turn::post_task::LearningContext,
    directives: &mut Vec<crate::turn::system_directives::SystemDirective>,
) -> Option<String> {
    let status = match host
        .goal_budgets()
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
    let metrics = || crate::turn::budget::ScheduledRunActivityMetrics {
        evidence_gain_count: successes,
        stall_count: usize::from(scope.repeating.load(Ordering::Relaxed)),
        consecutive_same_tool_count: 0,
        consecutive_same_tool_unique_args: 0,
        total_successful_tool_calls: successes,
    };
    if goal.scheduled {
        let registry = host.goal_tokens()?;
        registry.add_run_tokens(&goal.goal_id, delta).await;
        let progress = scope
            .progress
            .lock()
            .ok()
            .map(|progress| progress.clone())?;
        let run_status = registry
            .update_run_health(
                &goal.goal_id,
                crate::turn::budget::scheduled_run_health_snapshot(
                    learning,
                    metrics(),
                    scope.contract,
                    &progress,
                ),
            )
            .await?;
        host.persist_scheduled_run_state(&goal.goal_id, &run_status)
            .await;
        // One warning at 80% of the per-run budget, so the run can wrap
        // up before it is stopped.
        if let Some(pct) = crate::turn::budget::scheduled_run_budget_pressure_pct(
            &run_status,
            scope.budget_warning_sent.load(Ordering::Relaxed),
        ) {
            scope.budget_warning_sent.store(true, Ordering::Relaxed);
            directives.push(
                crate::turn::system_directives::SystemDirective::ScheduledRunBudgetPressure {
                    used: run_status.tokens_used,
                    budget: run_status.effective_budget_per_check,
                    pct,
                },
            );
            crate::turn::task_end::emit_warning_decision_point(
                host,
                scope.emitter,
                scope.task_id,
                iteration,
                aidaemon_events::DecisionType::StoppingCondition,
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
        let mut ctx = crate::turn::budget::ScheduledRunBudgetControlCtx {
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
        if let crate::turn::budget::ScheduledRunBudgetControlOutcome::Exhausted {
            tokens_used,
            budget_per_check,
        } = crate::turn::budget::enforce_scheduled_run_budget_control(host, &mut ctx).await
        {
            let alert = format!(
                "Token alert: scheduled run for goal '{}' hit its per-run budget (used {} / \
                 limit {}). The run stopped safely before completion.",
                goal.goal_id, tokens_used, budget_per_check
            );
            crate::turn::budget::fanout_token_alert(
                host,
                Some(goal.goal_id.as_str()),
                scope.session_id,
                &alert,
                Some(scope.session_id),
            )
            .await;
            return Some(
                crate::turn::post_task::graceful_scheduled_run_budget_response(
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
        let mut ctx = crate::turn::budget::GoalBudgetControlCtx {
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
            source: crate::turn::budget::GoalBudgetCheckSource::PostLlm,
        };
        crate::turn::budget::enforce_goal_daily_budget_control(host, &mut ctx).await
    };
    if let Ok(mut budget) = scope.goal_daily_budget.lock() {
        *budget = effective_daily;
    }
    scope
        .goal_budget_extensions
        .store(extensions as u32, Ordering::Relaxed);
    if let crate::turn::budget::GoalBudgetControlOutcome::Exhausted {
        tokens_used_today,
        budget_daily,
    } = outcome
    {
        let alert = format!(
            "Token alert: goal '{}' hit daily token budget (used {} / limit {}). Execution \
             was stopped to prevent overspending.",
            goal.goal_id, tokens_used_today, budget_daily
        );
        crate::turn::budget::fanout_token_alert(
            host,
            Some(goal.goal_id.as_str()),
            scope.session_id,
            &alert,
            Some(scope.session_id),
        )
        .await;
        return Some(crate::turn::post_task::graceful_goal_daily_budget_response(
            learning,
            tokens_used_today,
            budget_daily,
            false,
        ));
    }
    None
}

/// Record the tool calls of a response the run will act on.
pub async fn persist_sdk_tool_call_message(
    host: &dyn TurnHost,
    scope: &TurnScope<'_>,
    response: &sdk::ModelResponse,
) {
    let calls: Vec<aidaemon_events::conversation::ToolCall> = response
        .tool_calls
        .iter()
        .map(|call| aidaemon_events::conversation::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.to_string(),
            extra_content: None,
        })
        .collect();
    let mut assistant =
        Message::new_runtime(Uuid::new_v4().to_string(), scope.session_id, "assistant");
    let text = response.text.trim();
    assistant.content = (!text.is_empty()).then(|| text.to_string());
    assistant.tool_calls_json = serde_json::to_string(&calls).ok();
    assistant.turn_id = Some(scope.turn_id.to_string());
    if let Err(error) = host
        .append_assistant_message_with_event(scope.emitter, &assistant, scope.model, None, None)
        .await
    {
        warn!(task_id = scope.task_id, %error, "SDK engine: could not record a tool-call message");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live 2026-11-13: a turn's two helpers charged the request's one
    /// token counter, but each compared it with a limit of its own and
    /// extended its own. The tree has one limit and one extension count:
    /// the first member over it decides, and the second finds it extended.
    #[tokio::test]
    async fn a_request_and_its_helpers_have_one_budget() {
        let family = crate::authority::SpawnFamily::new(Default::default());
        let root = family.clone();
        let helper = family.clone();
        root.budget().start(500_000);
        // A helper's own turn does not reset the request's budget.
        helper.budget().start(500_000);
        helper.budget().start(123);
        assert_eq!(family.budget().limit(), 500_000);

        root.budget().charge(96_599);
        // WordPress works; Drupal checks on its running `ddev start`.
        helper.budget().note_success("drupal");
        helper.budget().note_success("wordpress");
        let used = helper.budget().charge(416_792);
        assert_eq!(used, 513_391);
        assert_eq!(
            family.tokens().load(std::sync::atomic::Ordering::Relaxed),
            513_391
        );

        // Drupal, repeating its check, hits the limit first: its sibling
        // did the newest work, so the tree is progressing.
        let tree = helper.budget();
        let mut extensions = tree.deciding().await;
        assert!(tree.progressed(&extensions, "drupal", true));
        let step = task_budget_step(
            used,
            tree.limit(),
            *extensions,
            tree.progressed(&extensions, "drupal", true),
            true,
        );
        assert_eq!(step, BudgetStep::Extend { to: 1_000_000 });
        tree.extend(1_000_000, &mut extensions);
        drop(extensions);
        // The other helper, over the old limit too, finds it extended.
        assert!(root.budget().used() < root.budget().limit());
        assert_eq!(root.budget().deciding().await.count, 1);
        // Nothing new since that extension: no progress.
        assert!(!tree.progressed(&*tree.deciding().await, "wordpress", false));
    }

    /// A lone run going round in circles is not progressing: its latest
    /// call repeated one it made, and nobody else did newer work.
    #[test]
    fn a_lone_run_repeating_itself_is_not_progressing() {
        let tree = TreeBudget::default();
        tree.note_success("solo");
        let fresh = Extensions::default();
        assert!(tree.progressed(&fresh, "solo", false));
        assert!(!tree.progressed(&fresh, "solo", true));
    }

    /// The tree's budget judges the tree's progress: a helper repeating its
    /// check on a running command hits the limit while its sibling has done
    /// new work since the last extension, even when the repeating helper
    /// did the very latest new thing (live 2026-11-14: "Nothing new has
    /// succeeded since the budget last grew" at 500,044 of 500,000 tokens
    /// while two helpers were installing a site each).
    #[tokio::test]
    async fn a_siblings_new_work_is_the_trees_progress() {
        let tree = TreeBudget::default();
        tree.start(500_000);
        tree.note_success("wordpress");
        tree.note_success("drupal");
        let extensions = tree.deciding().await;
        assert!(tree.progressed(&extensions, "drupal", true));
        assert!(tree.progressed(&extensions, "wordpress", true));
        drop(extensions);

        let mut extensions = tree.deciding().await;
        tree.extend(1_000_000, &mut extensions);
        // Since then only Drupal did anything new: repeating, it has no
        // one else's progress to lean on; not repeating, its work counts.
        tree.note_success("drupal");
        assert!(!tree.progressed(&extensions, "drupal", true));
        assert!(tree.progressed(&extensions, "drupal", false));
        assert!(tree.progressed(&extensions, "wordpress", true));
    }

    /// The owner is asked only past the configured policy: its extensions
    /// used up, or nothing new done since the last one, or not the owner's
    /// request. The hard cap stops without asking.
    #[test]
    fn the_owner_is_asked_only_past_the_auto_extension_policy() {
        let fresh = Extensions::default();
        assert_eq!(
            task_budget_step(513_391, 500_000, fresh, true, true),
            BudgetStep::Extend { to: 1_000_000 }
        );
        let spent = Extensions {
            count: TASK_MAX_BUDGET_EXTENSIONS,
            successes_at_last: 2,
        };
        assert_eq!(
            task_budget_step(1_000_001, 1_000_000, spent, true, true),
            BudgetStep::Ask {
                to: 2_000_000,
                why: AskWhy::ExtensionsUsed(TASK_MAX_BUDGET_EXTENSIONS)
            }
        );
        let stalled = Extensions {
            count: 1,
            successes_at_last: 4,
        };
        assert_eq!(
            task_budget_step(1_000_001, 1_000_000, stalled, false, true),
            BudgetStep::Ask {
                to: 2_000_000,
                why: AskWhy::NoProgress
            }
        );
        assert!(matches!(
            task_budget_step(513_391, 500_000, fresh, true, false),
            BudgetStep::Ask {
                why: AskWhy::NotOwners,
                ..
            }
        ));
        assert_eq!(
            task_budget_step(2_000_001, TASK_HARD_TOKEN_CAP, fresh, true, true),
            BudgetStep::Stop
        );
    }

    /// When the owner is asked, the card says what the spending is for,
    /// what it cost so far and why it is asked, in plain words.
    #[test]
    fn a_budget_question_says_what_it_is_for_and_what_it_cost() {
        let ask = crate::turn::budget::BudgetAsk {
            scope_label: "task",
            work: crate::turn::budget::work_words(
                "Install a Drupal demo site in ~/projects/claude-drupal-2026-11-13-demo-4\nUse DDEV.",
            ),
            used: 513_391,
            current: 500_000,
            proposed: 1_000_000,
            why: AskWhy::NoProgress.words(),
        };
        assert_eq!(
            ask.action(),
            "let \u{201c}Install a Drupal demo site in ~/projects/claude-drupal-2026-11-13-demo-4\u{201d} \
             keep going past its token budget, up to 1,000,000 tokens"
        );
        assert_eq!(
            ask.cost(),
            "This request and the helpers it started have used 513,391 tokens, past its budget \
             of 500,000."
        );
        assert!(!ask.action().contains("500000") && !ask.cost().contains("execution"));
    }
}
