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
