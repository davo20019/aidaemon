#[derive(Debug, Default)]
pub(in crate::agent) struct BudgetTracker {
    task_tokens_used: u64,
    effective_task_budget: Option<u64>,
    effective_daily_budget: Option<u64>,
    budget_extensions_count: usize,
    budget_warning_sent: bool,
    provider_timeout_ms: u64,
    soft_limit_warned: bool,
    hard_cap: Option<usize>,
    soft_threshold: Option<usize>,
    soft_warn_at: Option<usize>,
}

pub(in crate::agent) struct StoppingBudgetState<'a> {
    pub hard_cap: Option<usize>,
    pub task_tokens_used: u64,
    pub effective_task_budget: &'a mut Option<u64>,
    pub budget_warning_sent: &'a mut bool,
    pub budget_extensions_count: &'a mut usize,
    pub effective_daily_budget: &'a mut Option<u64>,
    pub soft_threshold: Option<usize>,
    pub soft_warn_at: Option<usize>,
    pub soft_limit_warned: &'a mut bool,
}

pub(in crate::agent) struct LlmBudgetState<'a> {
    pub task_tokens_used: &'a mut u64,
    pub budget_extensions_count: &'a mut usize,
    pub provider_timeout_ms: &'a mut u64,
}

impl BudgetTracker {
    pub(in crate::agent) fn new(
        task_token_budget: Option<u64>,
        daily_token_budget: Option<u64>,
        iteration_limits: IterationLimitSettings,
    ) -> Self {
        Self {
            effective_task_budget: task_token_budget,
            effective_daily_budget: daily_token_budget,
            hard_cap: iteration_limits.hard_cap,
            soft_threshold: iteration_limits.soft_threshold,
            soft_warn_at: iteration_limits.soft_warn_at,
            ..Self::default()
        }
    }

    pub(in crate::agent) fn for_stopping_phase(&mut self) -> StoppingBudgetState<'_> {
        StoppingBudgetState {
            hard_cap: self.hard_cap,
            task_tokens_used: self.task_tokens_used,
            effective_task_budget: &mut self.effective_task_budget,
            budget_warning_sent: &mut self.budget_warning_sent,
            budget_extensions_count: &mut self.budget_extensions_count,
            effective_daily_budget: &mut self.effective_daily_budget,
            soft_threshold: self.soft_threshold,
            soft_warn_at: self.soft_warn_at,
            soft_limit_warned: &mut self.soft_limit_warned,
        }
    }

    pub(in crate::agent) fn for_llm_phase(&mut self) -> LlmBudgetState<'_> {
        LlmBudgetState {
            task_tokens_used: &mut self.task_tokens_used,
            budget_extensions_count: &mut self.budget_extensions_count,
            provider_timeout_ms: &mut self.provider_timeout_ms,
        }
    }

    pub(in crate::agent) fn task_tokens_used(&self) -> u64 {
        self.task_tokens_used
    }

    pub(in crate::agent) fn add_task_tokens(&mut self, tokens: u64) -> u64 {
        self.task_tokens_used = self.task_tokens_used.saturating_add(tokens);
        self.task_tokens_used
    }

    pub(in crate::agent) fn effective_task_budget(&self) -> Option<u64> {
        self.effective_task_budget
    }

    pub(in crate::agent) fn effective_daily_budget(&self) -> Option<u64> {
        self.effective_daily_budget
    }

    pub(in crate::agent) fn set_effective_task_budget(&mut self, budget: Option<u64>) {
        self.effective_task_budget = budget;
    }

    pub(in crate::agent) fn raise_effective_task_budget_to(&mut self, minimum: u64) {
        self.effective_task_budget = Some(
            self.effective_task_budget
                .map(|budget| budget.max(minimum))
                .unwrap_or(minimum),
        );
    }

    pub(in crate::agent) fn set_effective_daily_budget(&mut self, budget: Option<u64>) {
        self.effective_daily_budget = budget;
    }

    pub(in crate::agent) fn budget_warning_sent(&self) -> bool {
        self.budget_warning_sent
    }

    pub(in crate::agent) fn set_budget_warning_sent(&mut self, sent: bool) {
        self.budget_warning_sent = sent;
    }

    pub(in crate::agent) fn budget_extensions_count(&self) -> usize {
        self.budget_extensions_count
    }

    pub(in crate::agent) fn set_budget_extensions_count(&mut self, count: usize) {
        self.budget_extensions_count = count;
    }

    pub(in crate::agent) fn increment_budget_extensions_count(&mut self) -> usize {
        self.budget_extensions_count = self.budget_extensions_count.saturating_add(1);
        self.budget_extensions_count
    }

    pub(in crate::agent) fn provider_timeout_ms(&self) -> u64 {
        self.provider_timeout_ms
    }

    pub(in crate::agent) fn add_provider_timeout_ms(&mut self, timeout_ms: u64) -> u64 {
        self.provider_timeout_ms = self.provider_timeout_ms.saturating_add(timeout_ms);
        self.provider_timeout_ms
    }

    pub(in crate::agent) fn soft_limit_warned(&self) -> bool {
        self.soft_limit_warned
    }

    pub(in crate::agent) fn set_soft_limit_warned(&mut self, warned: bool) {
        self.soft_limit_warned = warned;
    }

    pub(in crate::agent) fn hard_cap(&self) -> Option<usize> {
        self.hard_cap
    }

    pub(in crate::agent) fn soft_threshold(&self) -> Option<usize> {
        self.soft_threshold
    }

    pub(in crate::agent) fn soft_warn_at(&self) -> Option<usize> {
        self.soft_warn_at
    }

    pub(in crate::agent) fn set_iteration_limits(&mut self, limits: IterationLimitSettings) {
        self.hard_cap = limits.hard_cap;
        self.soft_threshold = limits.soft_threshold;
        self.soft_warn_at = limits.soft_warn_at;
    }

    pub(in crate::agent) fn disable_iteration_limits(&mut self) {
        self.hard_cap = None;
        self.soft_threshold = None;
        self.soft_warn_at = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::agent) struct IterationLimitSettings {
    pub hard_cap: Option<usize>,
    pub soft_threshold: Option<usize>,
    pub soft_warn_at: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::{BudgetTracker, IterationLimitSettings};

    #[test]
    fn initializes_budgets_and_iteration_limits() {
        let tracker = BudgetTracker::new(
            Some(1_000),
            Some(10_000),
            IterationLimitSettings {
                hard_cap: Some(50),
                soft_threshold: Some(30),
                soft_warn_at: Some(20),
            },
        );

        assert_eq!(tracker.effective_task_budget(), Some(1_000));
        assert_eq!(tracker.effective_daily_budget(), Some(10_000));
        assert_eq!(tracker.hard_cap(), Some(50));
        assert_eq!(tracker.soft_threshold(), Some(30));
        assert_eq!(tracker.soft_warn_at(), Some(20));
    }

    #[test]
    fn accumulates_task_tokens_with_saturation() {
        let mut tracker = BudgetTracker::default();

        assert_eq!(tracker.add_task_tokens(40), 40);
        assert_eq!(tracker.add_task_tokens(2), 42);
        tracker.add_task_tokens(u64::MAX);

        assert_eq!(tracker.task_tokens_used(), u64::MAX);
    }

    #[test]
    fn tracks_warning_and_extension_count() {
        let mut tracker = BudgetTracker::default();

        assert!(!tracker.budget_warning_sent());
        tracker.set_budget_warning_sent(true);
        assert!(tracker.budget_warning_sent());
        assert_eq!(tracker.increment_budget_extensions_count(), 1);
        tracker.set_budget_extensions_count(7);
        assert_eq!(tracker.budget_extensions_count(), 7);
    }

    #[test]
    fn accumulates_provider_timeout_with_saturation() {
        let mut tracker = BudgetTracker::default();

        assert_eq!(tracker.add_provider_timeout_ms(500), 500);
        tracker.add_provider_timeout_ms(u64::MAX);

        assert_eq!(tracker.provider_timeout_ms(), u64::MAX);
    }

    #[test]
    fn can_disable_iteration_limits_for_scheduled_runs() {
        let mut tracker = BudgetTracker::new(
            None,
            None,
            IterationLimitSettings {
                hard_cap: Some(60),
                soft_threshold: Some(40),
                soft_warn_at: Some(20),
            },
        );

        tracker.disable_iteration_limits();

        assert_eq!(tracker.hard_cap(), None);
        assert_eq!(tracker.soft_threshold(), None);
        assert_eq!(tracker.soft_warn_at(), None);
    }

    #[test]
    fn raises_effective_task_budget_without_lowering_existing_budget() {
        let mut tracker = BudgetTracker::new(
            Some(100),
            None,
            IterationLimitSettings {
                hard_cap: None,
                soft_threshold: None,
                soft_warn_at: None,
            },
        );

        tracker.raise_effective_task_budget_to(90);
        assert_eq!(tracker.effective_task_budget(), Some(100));

        tracker.raise_effective_task_budget_to(150);
        assert_eq!(tracker.effective_task_budget(), Some(150));
    }
}
