//! Token budgets inside a turn: a goal's daily budget and a scheduled run's
//! per-run budget, their autonomous extensions on real progress, the owner's
//! approval for one more extension, and token alerts to the owner.

use std::time::Duration;

use aidaemon_events::DecisionType;
use aidaemon_tools::semantics::{ApprovalResponse, PermissionMode, RiskLevel, UserRole};
use aidaemon_tools::status::StatusUpdate;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::host::TurnHost;
use crate::turn::completion_contract::{CompletionContract, CompletionProgress};
use crate::turn::loop_utils::MAX_CONSECUTIVE_SAME_TOOL;
use crate::turn::post_task::{self, LearningContext};
use crate::turn::system_directives::SystemDirective;
use crate::turn::validation_state::{build_needs_approval_request, ApprovalState};

#[derive(Debug, Clone, Copy)]
pub enum GoalBudgetCheckSource {
    PostLlm,
}

pub struct GoalBudgetControlCtx<'a> {
    pub emitter: &'a aidaemon_events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub iteration: usize,
    pub goal_id: &'a str,
    pub status: &'a aidaemon_work::traits::GoalTokenBudgetStatus,
    pub user_role: UserRole,
    pub learning_ctx: &'a LearningContext,
    pub evidence_gain_count: usize,
    pub stall_count: usize,
    pub consecutive_same_tool_count: usize,
    pub consecutive_same_tool_unique_args: usize,
    pub total_successful_tool_calls: usize,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
    pub is_scheduled_goal: bool,
    pub effective_goal_daily_budget: &'a mut Option<i64>,
    pub budget_extensions_count: &'a mut usize,
    pub max_budget_extensions: usize,
    pub hard_token_cap: i64,
    pub source: GoalBudgetCheckSource,
}

pub enum GoalBudgetControlOutcome {
    Continue,
    Exhausted {
        tokens_used_today: i64,
        budget_daily: i64,
    },
}

pub struct ScheduledRunBudgetControlCtx<'a> {
    pub emitter: &'a aidaemon_events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub iteration: usize,
    pub goal_id: &'a str,
    pub status: &'a aidaemon_work::goal_tokens::GoalRunBudgetStatus,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub max_budget_extensions: usize,
    pub hard_token_cap: i64,
}

pub enum ScheduledRunBudgetControlOutcome {
    Continue,
    Exhausted {
        tokens_used: i64,
        budget_per_check: i64,
    },
}

pub struct ScheduledRunActivityMetrics {
    pub evidence_gain_count: usize,
    pub stall_count: usize,
    pub consecutive_same_tool_count: usize,
    pub consecutive_same_tool_unique_args: usize,
    pub total_successful_tool_calls: usize,
}

pub fn has_meaningful_budget_progress(
    evidence_gain_count: usize,
    _total_successful_tool_calls: usize,
) -> bool {
    // Transport-level success and administrative calls are not progress.
    // Only a mutation receipt or result-content verification can justify
    // spending the run's single autonomous extension.
    evidence_gain_count > 0
}

pub fn scheduled_run_health_snapshot(
    learning_ctx: &LearningContext,
    metrics: ScheduledRunActivityMetrics,
    completion_contract: &CompletionContract,
    completion_progress: &CompletionProgress,
) -> aidaemon_work::traits::ScheduledRunHealth {
    let required_mutation_progress = completion_contract.expects_mutation
        && completion_progress.mutation_count > 0
        && (completion_contract.required_mutation_effects.is_empty()
            || completion_progress
                .observed_mutation_effects
                .intersects(completion_contract.required_mutation_effects));
    aidaemon_work::traits::ScheduledRunHealth {
        evidence_gain_count: metrics.evidence_gain_count,
        total_successful_tool_calls: metrics.total_successful_tool_calls,
        completion_requires_mutation: completion_contract.expects_mutation,
        required_mutation_progress,
        completion_requires_observation: completion_contract.requires_observation,
        verification_progress: completion_progress.verification_count > 0,
        stall_count: metrics.stall_count,
        consecutive_same_tool_count: metrics.consecutive_same_tool_count,
        consecutive_same_tool_unique_args: metrics.consecutive_same_tool_unique_args,
        unrecovered_error_count: learning_ctx
            .errors
            .iter()
            .filter(|(_, recovered)| !recovered)
            .count(),
    }
}

pub fn scheduled_run_has_structural_progress(
    health: &aidaemon_work::traits::ScheduledRunHealth,
) -> bool {
    if health.completion_requires_mutation {
        return health.required_mutation_progress;
    }
    if health.completion_requires_observation {
        return health.verification_progress;
    }
    // Text-only scheduled work has no mutation/verification receipt to
    // satisfy, so bounded result evidence remains its progress signal.
    health.evidence_gain_count > 0
}

pub fn scheduled_run_metrics_are_clearly_unproductive(
    health: &aidaemon_work::traits::ScheduledRunHealth,
) -> bool {
    if health.stall_count > 1 {
        return true;
    }

    let diverse_limit = MAX_CONSECUTIVE_SAME_TOOL + 4;
    if health.consecutive_same_tool_count >= diverse_limit {
        return true;
    }
    if health.consecutive_same_tool_count >= MAX_CONSECUTIVE_SAME_TOOL {
        let is_diverse =
            health.consecutive_same_tool_unique_args * 2 > health.consecutive_same_tool_count;
        if !is_diverse {
            return true;
        }
    }

    if health.total_successful_tool_calls == 0 {
        return health.unrecovered_error_count > 0 && health.evidence_gain_count == 0;
    }

    health.unrecovered_error_count >= health.total_successful_tool_calls
}

pub fn scheduled_run_auto_extension_candidate(
    status: &aidaemon_work::goal_tokens::GoalRunBudgetStatus,
    max_budget_extensions: usize,
    hard_token_cap: i64,
) -> Option<i64> {
    let old_budget = status.effective_budget_per_check;
    let new_budget = old_budget
        .saturating_mul(2)
        .max(status.tokens_used.saturating_add(old_budget / 2))
        .min(hard_token_cap);

    let has_meaningful_progress = scheduled_run_has_structural_progress(&status.health);
    let clearly_unproductive = scheduled_run_metrics_are_clearly_unproductive(&status.health);

    if status.budget_extensions_count < max_budget_extensions
        && old_budget < hard_token_cap
        && new_budget > status.tokens_used
        && has_meaningful_progress
        && !clearly_unproductive
    {
        Some(new_budget)
    } else {
        None
    }
}

/// The percentage to warn at once a scheduled run has used 80% of its
/// per-run budget (not yet exhausted, not already warned).
pub fn scheduled_run_budget_pressure_pct(
    status: &aidaemon_work::goal_tokens::GoalRunBudgetStatus,
    warning_already_sent: bool,
) -> Option<i64> {
    let budget = status.effective_budget_per_check;
    if warning_already_sent || budget <= 0 || status.tokens_used >= budget {
        return None;
    }
    let warning_threshold = budget.saturating_mul(80) / 100;
    (status.tokens_used >= warning_threshold)
        .then(|| status.tokens_used.saturating_mul(100) / budget)
}

/// A budget extension that is the owner's decision: what the spending is
/// for, what it cost so far, what it would allow and why it is asked, all
/// in plain words on the card (live 2026-11-13: "extend the task token
/// budget from 500000 to 1000000 and continue execution (target: task token
/// budget)" reached the owner mid-run with nothing saying what it was for).
pub struct BudgetAsk<'a> {
    /// `task` (one request and its helpers) or `goal daily`.
    pub scope_label: &'a str,
    /// The work it is for, in the words it was asked in.
    pub work: String,
    pub used: i64,
    pub current: i64,
    pub proposed: i64,
    /// Why this is past the automatic policy.
    pub why: String,
}

impl BudgetAsk<'_> {
    /// What is asked, in one line.
    pub fn action(&self) -> String {
        format!(
            "let \u{201c}{}\u{201d} keep going past its token budget, up to {} tokens",
            self.work,
            thousands(self.proposed)
        )
    }

    /// What it cost so far.
    pub fn cost(&self) -> String {
        if self.scope_label == "task" {
            format!(
                "This request and the helpers it started have used {} tokens, past its budget of \
                 {}.",
                thousands(self.used),
                thousands(self.current)
            )
        } else {
            format!(
                "It has used {} tokens of its {} budget of {}.",
                thousands(self.used),
                self.scope_label,
                thousands(self.current)
            )
        }
    }
}

/// The work a request is, for the owner: its first line, cut at a word.
pub fn work_words(text: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("this request");
    const MAX: usize = 80;
    if line.chars().count() <= MAX {
        return line.to_string();
    }
    let cut: String = line.chars().take(MAX).collect();
    let cut = cut.rsplit_once(' ').map_or(cut.as_str(), |(head, _)| head);
    format!("{}\u{2026}", cut.trim_end())
}

/// `1000000` as "1,000,000".
pub fn thousands(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

/// Ask the owner to approve one budget extension past the automatic
/// policy ([`BudgetAsk`]).
///
/// Returns true only when the owner explicitly approves.
#[allow(clippy::too_many_arguments)]
pub async fn request_budget_continue_approval(
    host: &dyn TurnHost,
    emitter: &aidaemon_events::EventEmitter,
    task_id: &str,
    iteration: usize,
    session_id: &str,
    user_role: UserRole,
    ask: BudgetAsk<'_>,
) -> bool {
    let (scope_label, used_tokens, current_budget, proposed_budget) =
        (ask.scope_label, ask.used, ask.current, ask.proposed);
    if user_role != UserRole::Owner {
        return false;
    }
    if proposed_budget <= current_budget {
        return false;
    }

    let hub_weak = match tokio::time::timeout(Duration::from_secs(2), host.hub().read()).await {
        Ok(guard) => guard.clone(),
        Err(_) => {
            warn!(
                session_id,
                scope = scope_label,
                "Timed out acquiring hub lock for budget extension approval"
            );
            return false;
        }
    };
    let Some(hub_weak) = hub_weak else {
        return false;
    };
    let Some(hub_arc) = hub_weak.upgrade() else {
        return false;
    };

    let approval_request = build_needs_approval_request(
        ask.action(),
        None,
        ask.cost(),
        ask.why.clone(),
        format!(
            "If you allow it, the work continues within {} tokens.",
            thousands(proposed_budget)
        ),
        None,
    );
    let (approval_desc, warnings) = approval_request.to_inline_approval_prompt();
    crate::turn::task_end::emit_decision_point(
        host,
        emitter,
        task_id,
        iteration,
        DecisionType::BudgetAutoExtension,
        format!(
            "Requested owner approval for {} budget extension",
            scope_label
        ),
        json!({
            "condition": "budget_extension_manual_request",
            "scope_label": scope_label,
            "approval_state": ApprovalState::Requested,
            "used_tokens": used_tokens,
            "current_budget": current_budget,
            "proposed_budget": proposed_budget,
        }),
    )
    .await;

    match hub_arc
        .request_inline_approval(
            session_id,
            &approval_desc,
            RiskLevel::High,
            &warnings,
            PermissionMode::Cautious,
        )
        .await
    {
        Ok(ApprovalResponse::AllowOnce)
        | Ok(ApprovalResponse::AllowSession)
        | Ok(ApprovalResponse::AllowAlways) => true,
        Ok(ApprovalResponse::Deny) => {
            crate::turn::task_end::emit_warning_decision_point(
                host,
                emitter,
                task_id,
                iteration,
                DecisionType::BudgetAutoExtension,
                format!("Owner denied {} budget extension", scope_label),
                json!({
                    "condition": "budget_extension_manual_denied",
                    "scope_label": scope_label,
                    "approval_state": ApprovalState::Denied,
                    "used_tokens": used_tokens,
                    "current_budget": current_budget,
                    "proposed_budget": proposed_budget,
                }),
            )
            .await;
            false
        }
        Err(e) => {
            crate::turn::task_end::emit_warning_decision_point(
                host,
                emitter,
                task_id,
                iteration,
                DecisionType::BudgetAutoExtension,
                format!("Approval unavailable for {} budget extension", scope_label),
                json!({
                    "condition": "budget_extension_manual_unavailable",
                    "scope_label": scope_label,
                    "approval_state": ApprovalState::Denied,
                    "used_tokens": used_tokens,
                    "current_budget": current_budget,
                    "proposed_budget": proposed_budget,
                    "error": e.to_string(),
                }),
            )
            .await;
            warn!(
                session_id,
                scope = scope_label,
                error = %e,
                "Budget extension approval unavailable"
            );
            false
        }
    }
}

pub async fn enforce_goal_daily_budget_control(
    host: &dyn TurnHost,
    ctx: &mut GoalBudgetControlCtx<'_>,
) -> GoalBudgetControlOutcome {
    let Some(db_budget_daily) = ctx.status.budget_daily else {
        return GoalBudgetControlOutcome::Continue;
    };
    let shared_budget_daily = if let Some(registry) = host.goal_tokens() {
        registry.get_effective_daily_budget(ctx.goal_id).await
    } else {
        None
    };
    let durable_override = aidaemon_work::goal_tokens::load_goal_daily_budget_override(
        host.settings(),
        ctx.goal_id,
        db_budget_daily,
        ctx.hard_token_cap,
    )
    .await;
    if let Some(durable) = durable_override.as_ref() {
        *ctx.budget_extensions_count = (*ctx.budget_extensions_count)
            .max(durable.extensions_count.min(ctx.max_budget_extensions));
        if let Some(registry) = host.goal_tokens() {
            registry
                .set_effective_daily_budget(ctx.goal_id, durable.budget_daily)
                .await;
        }
    }
    let budget_daily = [
        Some(db_budget_daily),
        *ctx.effective_goal_daily_budget,
        shared_budget_daily,
        durable_override.map(|value| value.budget_daily),
    ]
    .into_iter()
    .flatten()
    .max()
    .unwrap_or(db_budget_daily);
    *ctx.effective_goal_daily_budget = Some(budget_daily);
    if budget_daily <= 0 || ctx.status.tokens_used_today < budget_daily {
        return GoalBudgetControlOutcome::Continue;
    }

    let old_gbudget = budget_daily;
    let new_gbudget = old_gbudget
        .saturating_mul(2)
        .max(ctx.status.tokens_used_today.saturating_add(old_gbudget / 2))
        .min(ctx.hard_token_cap);

    let productive = if ctx.is_scheduled_goal {
        // Active scheduled runs are governed by their shared per-run
        // budget above. Keep this fail-closed for any legacy caller that
        // reaches daily control directly.
        false
    } else {
        has_meaningful_budget_progress(ctx.evidence_gain_count, ctx.total_successful_tool_calls)
            && post_task::is_productive(
                ctx.learning_ctx,
                ctx.stall_count,
                ctx.consecutive_same_tool_count,
                ctx.consecutive_same_tool_unique_args,
                ctx.total_successful_tool_calls,
            )
    };

    let (auto_condition, manual_condition, source_label) = match ctx.source {
        GoalBudgetCheckSource::PostLlm => (
            "goal_daily_budget_extension_post_llm",
            "goal_daily_budget_extension_manual_post_llm",
            "post-LLM",
        ),
    };

    if *ctx.budget_extensions_count < ctx.max_budget_extensions
        && old_gbudget < ctx.hard_token_cap
        && new_gbudget > ctx.status.tokens_used_today
        && productive
    {
        *ctx.budget_extensions_count += 1;
        *ctx.effective_goal_daily_budget = Some(new_gbudget);
        if let Some(registry) = host.goal_tokens() {
            registry
                .set_effective_daily_budget(ctx.goal_id, new_gbudget)
                .await;
        }
        if let Err(error) = aidaemon_work::goal_tokens::persist_goal_daily_budget_override(
            host.settings(),
            ctx.goal_id,
            new_gbudget,
            *ctx.budget_extensions_count,
        )
        .await
        {
            warn!(
                goal_id = %ctx.goal_id,
                %error,
                "Failed to persist same-day goal budget extension"
            );
        }
        info!(
            ctx.session_id,
            goal_id = %ctx.goal_id,
            old_budget = old_gbudget,
            new_budget = new_gbudget,
            extension = *ctx.budget_extensions_count,
            source = source_label,
            "Auto-extended goal daily token budget in-memory"
        );
        ctx.pending_system_messages
            .push(SystemDirective::GoalDailyBudgetAutoExtended {
                old_budget: old_gbudget,
                new_budget: new_gbudget,
                extension: *ctx.budget_extensions_count,
                max_extensions: ctx.max_budget_extensions,
            });
        crate::turn::task_end::emit_decision_point(
            host,
            ctx.emitter,
            ctx.task_id,
            ctx.iteration,
            DecisionType::BudgetAutoExtension,
            "Auto-extended goal daily token budget on productive progress".to_string(),
            json!({
                "condition": auto_condition,
                "goal_id": ctx.goal_id,
                "old_budget": old_gbudget,
                "new_budget": new_gbudget,
                "extension": *ctx.budget_extensions_count,
                "max_extensions": ctx.max_budget_extensions,
            }),
        )
        .await;
        return GoalBudgetControlOutcome::Continue;
    }

    // Scheduled work has an owner-confirmed unattended authority envelope.
    // It may use the bounded autonomous extension above, but it must never
    // convert resource management into a live approval interruption. Once
    // that envelope is spent, stop this cycle cleanly at the hard boundary.
    if ctx.is_scheduled_goal {
        return GoalBudgetControlOutcome::Exhausted {
            tokens_used_today: ctx.status.tokens_used_today,
            budget_daily,
        };
    }

    let approved_extension =
        if old_gbudget < ctx.hard_token_cap && new_gbudget > ctx.status.tokens_used_today {
            request_budget_continue_approval(
                host,
                ctx.emitter,
                ctx.task_id,
                ctx.iteration,
                ctx.session_id,
                ctx.user_role,
                BudgetAsk {
                    scope_label: "goal daily",
                    work: "this goal's work today".to_string(),
                    used: ctx.status.tokens_used_today,
                    current: old_gbudget,
                    proposed: new_gbudget,
                    why: if *ctx.budget_extensions_count >= ctx.max_budget_extensions {
                        crate::engine::budgets::AskWhy::ExtensionsUsed(*ctx.budget_extensions_count)
                    } else {
                        crate::engine::budgets::AskWhy::NoProgress
                    }
                    .words(),
                },
            )
            .await
        } else {
            false
        };

    if approved_extension {
        *ctx.effective_goal_daily_budget = Some(new_gbudget);
        if let Some(registry) = host.goal_tokens() {
            registry
                .set_effective_daily_budget(ctx.goal_id, new_gbudget)
                .await;
        }
        if let Err(error) = aidaemon_work::goal_tokens::persist_goal_daily_budget_override(
            host.settings(),
            ctx.goal_id,
            new_gbudget,
            (*ctx.budget_extensions_count).saturating_add(1),
        )
        .await
        {
            warn!(
                goal_id = %ctx.goal_id,
                %error,
                "Failed to persist owner-approved same-day goal budget extension"
            );
        }
        ctx.pending_system_messages
            .push(SystemDirective::GoalDailyBudgetExtensionApproved {
                old_budget: old_gbudget,
                new_budget: new_gbudget,
            });
        crate::turn::task_end::emit_decision_point(
            host,
            ctx.emitter,
            ctx.task_id,
            ctx.iteration,
            DecisionType::BudgetAutoExtension,
            "Extended goal daily token budget via owner approval".to_string(),
            json!({
                "condition": manual_condition,
                "goal_id": ctx.goal_id,
                "approval_state": ApprovalState::Granted,
                "old_budget": old_gbudget,
                "new_budget": new_gbudget,
                "tokens_used_today": ctx.status.tokens_used_today,
            }),
        )
        .await;
        return GoalBudgetControlOutcome::Continue;
    }

    GoalBudgetControlOutcome::Exhausted {
        tokens_used_today: ctx.status.tokens_used_today,
        budget_daily,
    }
}

pub async fn enforce_scheduled_run_budget_control(
    host: &dyn TurnHost,
    ctx: &mut ScheduledRunBudgetControlCtx<'_>,
) -> ScheduledRunBudgetControlOutcome {
    let budget_per_check = ctx.status.effective_budget_per_check;
    if budget_per_check <= 0 || ctx.status.tokens_used < budget_per_check {
        return ScheduledRunBudgetControlOutcome::Continue;
    }

    let old_budget = budget_per_check;
    if let Some(new_budget) = scheduled_run_auto_extension_candidate(
        ctx.status,
        ctx.max_budget_extensions,
        ctx.hard_token_cap,
    )
        && let Some(registry) = host.goal_tokens() {
            let updated = registry
                .auto_extend_run_budget(ctx.goal_id, new_budget)
                .await;
            if let Some(status) = updated.as_ref() {
                host.persist_scheduled_run_state(ctx.goal_id, status).await;
            }
            let extension = updated
                .as_ref()
                .map(|status| status.budget_extensions_count)
                .unwrap_or_else(|| ctx.status.budget_extensions_count.saturating_add(1));
            info!(
                ctx.session_id,
                goal_id = %ctx.goal_id,
                old_budget,
                new_budget,
                extension,
                "Auto-extended scheduled run budget"
            );
            // This is routine internal adaptation, not an owner-attention
            // event. Keep it in the directive/event ledger instead of the
            // channel status stream so an unattended run cannot wake the
            // owner merely because it resized its own budget.
            ctx.pending_system_messages.push(
                SystemDirective::ScheduledRunBudgetAdaptationRequired {
                    old_budget,
                    new_budget,
                    extension,
                    max_extensions: ctx.max_budget_extensions,
                },
            );
            crate::turn::task_end::emit_decision_point(
                host,
                ctx.emitter,
                ctx.task_id,
                ctx.iteration,
                DecisionType::BudgetAutoExtension,
                "Auto-extended scheduled run budget on continued progress".to_string(),
                json!({
                    "condition": "scheduled_run_budget_extension",
                    "goal_id": ctx.goal_id,
                    "old_budget": old_budget,
                    "new_budget": new_budget,
                    "extension": extension,
                    "max_extensions": ctx.max_budget_extensions,
                    "tokens_used": ctx.status.tokens_used,
                }),
            )
            .await;
            return ScheduledRunBudgetControlOutcome::Continue;
        }

    ScheduledRunBudgetControlOutcome::Exhausted {
        tokens_used: ctx.status.tokens_used,
        budget_per_check,
    }
}

pub fn dedupe_alert_sessions(sessions: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for session in sessions {
        let trimmed = session.trim();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.to_string()) {
            out.push(trimmed.to_string());
        }
    }
    out
}

pub fn sanitize_alert_scope(scope: &str) -> String {
    scope
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub async fn load_default_alert_sessions(host: &dyn TurnHost) -> Vec<String> {
    match host.settings().get_setting("default_alert_sessions").await {
        Ok(Some(raw)) => match serde_json::from_str::<Vec<String>>(&raw) {
            Ok(sessions) => dedupe_alert_sessions(sessions),
            Err(e) => {
                warn!(error = %e, "Invalid default_alert_sessions setting");
                Vec::new()
            }
        },
        Ok(None) => Vec::new(),
        Err(e) => {
            warn!(error = %e, "Failed to read default_alert_sessions setting");
            Vec::new()
        }
    }
}

pub fn same_alert_destination(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    let is_non_telegram = |value: &str| {
        value.contains("slack:") || value.contains("discord:") || value.starts_with("specialist:")
    };
    if is_non_telegram(left) || is_non_telegram(right) {
        return false;
    }
    match (
        aidaemon_events::session::telegram_chat_id_from_session(left),
        aidaemon_events::session::telegram_chat_id_from_session(right),
    ) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

/// Fan-out token alerts to owner sessions plus the triggering session.
pub async fn fanout_token_alert(
    host: &dyn TurnHost,
    goal_id: Option<&str>,
    trigger_session_id: &str,
    message: &str,
    suppress_session_id: Option<&str>,
) {
    // Background goal runs deliver one consolidated terminal outcome from
    // the task lead. Child and task-lead budget checks can fire at nearly
    // the same instant; fanning out here produced duplicate alerts, often
    // addressed to internal specialist sessions. Defer those alerts to the
    // run finalizer, which has the originating user session and full task
    // outcome.
    if goal_id.is_some() && host.depth() > 0 {
        info!(
            depth = host.depth(),
            goal_id = goal_id.unwrap_or_default(),
            "Deferring background goal budget alert to terminal run notification"
        );
        return;
    }

    let mut targets = load_default_alert_sessions(host).await;
    let goal_session = if let Some(goal_id) = goal_id {
        host.goals()
            .get_goal(goal_id)
            .await
            .ok()
            .flatten()
            .map(|goal| goal.session_id)
    } else {
        None
    };
    let primary_session = goal_session.as_deref().unwrap_or(trigger_session_id);
    targets.retain(|target| !same_alert_destination(target, primary_session));
    targets.push(primary_session.to_string());
    targets = dedupe_alert_sessions(targets);

    let goal_ref = goal_id
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("token-budget:{}", sanitize_alert_scope(trigger_session_id)));

    let hub = match tokio::time::timeout(Duration::from_secs(2), host.hub().read()).await {
        Ok(guard) => guard.clone(),
        Err(_) => {
            warn!(
                trigger_session_id,
                "Timed out acquiring hub lock while faning out token alert"
            );
            None
        }
    };
    for target in targets {
        let entry = aidaemon_work::traits::NotificationEntry::new(
            &goal_ref,
            &target,
            "token_alert",
            message,
        );

        if let Err(e) = host.notifications().enqueue_notification(&entry).await {
            warn!(
                session_id = %target,
                goal_id = %goal_ref,
                error = %e,
                "Failed to enqueue token alert"
            );
            continue;
        }

        if suppress_session_id == Some(target.as_str()) {
            let _ = host
                .notifications()
                .mark_notification_delivered(&entry.id)
                .await;
            continue;
        }

        if let Some(hub_weak) = &hub
            && let Some(hub_arc) = hub_weak.upgrade()
                && hub_arc.send_text(&target, message).await.is_ok() {
                    let _ = host
                        .notifications()
                        .mark_notification_delivered(&entry.id)
                        .await;
                }
    }
}
