//! A scheduled goal's runs: whether work came from a schedule (its
//! provenance), and the persisted budget state of the scheduled run in
//! progress.

use crate::goal_tokens::GoalRunBudgetStatus;
use crate::goals::ScheduledRunState;
use crate::scheduler::SchedulerStore;

pub async fn task_has_scheduled_provenance<S: SchedulerStore + ?Sized>(
    state: &S,
    task_id: Option<&str>,
) -> bool {
    if let Some(tid) = task_id
        && let Ok(Some(run)) = state.get_goal_run_for_task(tid).await
    {
        return is_scheduled_cycle_trigger(&run.trigger_type);
    }

    false
}

/// A recovery run re-attempts a failed scheduled occurrence, so it is an
/// autonomous cycle of the scheduled goal in its own right. Treating it as
/// non-root made it inherit the failed run's persisted token spend and start
/// already over budget.
pub fn is_scheduled_cycle_trigger(trigger_type: &str) -> bool {
    matches!(trigger_type, "scheduled" | "recovery")
}

pub async fn active_scheduled_root_task_id<S: SchedulerStore + ?Sized>(
    state: &S,
    goal_id: &str,
) -> Option<String> {
    state
        .get_goal_runs(goal_id)
        .await
        .ok()?
        .into_iter()
        .filter(|run| is_scheduled_cycle_trigger(&run.trigger_type))
        .filter(|run| !matches!(run.status.as_str(), "completed" | "failed" | "cancelled"))
        .find_map(|run| run.root_task_id)
}

pub async fn goal_has_scheduled_provenance<S: SchedulerStore + ?Sized>(
    state: &S,
    goal_id: &str,
    task_id: Option<&str>,
) -> bool {
    if task_has_scheduled_provenance(state, task_id).await {
        return true;
    }

    if let Ok(schedules) = state.get_schedules_for_goal(goal_id).await
        && !schedules.is_empty()
    {
        return true;
    }

    if let Ok(runs) = state.get_goal_runs(goal_id).await
        && runs.iter().any(|run| run.trigger_type == "scheduled")
    {
        return true;
    }

    false
}

pub async fn persist_scheduled_run_state<S: SchedulerStore + ?Sized>(
    state: &S,
    goal_id: &str,
    root_task_id_hint: Option<&str>,
    status: &GoalRunBudgetStatus,
) {
    let existing = state.get_scheduled_run_state(goal_id).await.ok().flatten();
    // A caller-provided root always identifies the current run. Reusing the
    // prior persisted root here mixed token accounting across separate fires.
    let root_task_id = if let Some(root_task_id) = root_task_id_hint {
        Some(root_task_id.to_string())
    } else if let Some(record) = existing.as_ref() {
        Some(record.root_task_id.clone())
    } else {
        active_scheduled_root_task_id(state, goal_id).await
    };

    let Some(root_task_id) = root_task_id else {
        return;
    };

    let now = chrono::Utc::now().to_rfc3339();
    let created_at = existing
        .as_ref()
        .filter(|record| record.root_task_id == root_task_id)
        .map(|record| record.created_at.clone())
        .unwrap_or_else(|| now.clone());
    let record = ScheduledRunState {
        goal_id: goal_id.to_string(),
        root_task_id,
        effective_budget_per_check: status.effective_budget_per_check,
        tokens_used: status.tokens_used,
        budget_extensions_count: status.budget_extensions_count,
        health: status.health.clone(),
        created_at,
        updated_at: now,
    };
    let _ = state.upsert_scheduled_run_state(&record).await;
}

pub async fn clear_scheduled_run_state<S: SchedulerStore + ?Sized>(state: &S, goal_id: &str) {
    let _ = state.delete_scheduled_run_state(goal_id).await;
}
