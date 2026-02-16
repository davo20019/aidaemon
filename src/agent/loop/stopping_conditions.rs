//! Stopping-condition extraction scaffolding.
//!
//! This module models the pure threshold checks that currently live inline in
//! `main_loop.rs` so they can be moved behind a dedicated gate.
#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoppingCondition {
    HardIterationCap {
        cap: usize,
        iteration: usize,
    },
    TaskTimeout {
        timeout_secs: u64,
        elapsed_secs: u64,
    },
    TaskTokenBudget {
        budget: u64,
        used: u64,
    },
    Stall {
        stall_count: usize,
        max_stall_iterations: usize,
    },
}

/// Inputs for pure stopping checks (no DB/state queries).
#[derive(Debug, Clone, Copy)]
pub struct PureStoppingInputs {
    pub iteration: usize,
    pub hard_cap: Option<usize>,
    pub timeout_secs: Option<u64>,
    pub elapsed_secs: u64,
    pub task_token_budget: Option<u64>,
    pub task_tokens_used: u64,
    pub stall_count: usize,
    pub max_stall_iterations: usize,
}

impl PureStoppingInputs {
    /// Evaluate checks in the same precedence currently used by the main loop.
    pub fn evaluate(self) -> Option<StoppingCondition> {
        if let Some(cap) = self.hard_cap {
            if self.iteration > cap {
                return Some(StoppingCondition::HardIterationCap {
                    cap,
                    iteration: self.iteration,
                });
            }
        }

        if let Some(timeout_secs) = self.timeout_secs {
            if self.elapsed_secs > timeout_secs {
                return Some(StoppingCondition::TaskTimeout {
                    timeout_secs,
                    elapsed_secs: self.elapsed_secs,
                });
            }
        }

        if let Some(budget) = self.task_token_budget {
            if self.task_tokens_used >= budget {
                return Some(StoppingCondition::TaskTokenBudget {
                    budget,
                    used: self.task_tokens_used,
                });
            }
        }

        if self.stall_count >= self.max_stall_iterations {
            return Some(StoppingCondition::Stall {
                stall_count: self.stall_count,
                max_stall_iterations: self.max_stall_iterations,
            });
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_none_when_no_threshold_is_hit() {
        let out = PureStoppingInputs {
            iteration: 2,
            hard_cap: Some(10),
            timeout_secs: Some(300),
            elapsed_secs: 12,
            task_token_budget: Some(10_000),
            task_tokens_used: 90,
            stall_count: 0,
            max_stall_iterations: 3,
        }
        .evaluate();
        assert!(out.is_none());
    }

    #[test]
    fn honors_existing_precedence_order() {
        // Both cap and timeout are tripped, but cap should win because it is
        // evaluated first in the current loop order.
        let out = PureStoppingInputs {
            iteration: 11,
            hard_cap: Some(10),
            timeout_secs: Some(1),
            elapsed_secs: 20,
            task_token_budget: Some(10),
            task_tokens_used: 999,
            stall_count: 99,
            max_stall_iterations: 3,
        }
        .evaluate();
        assert_eq!(
            out,
            Some(StoppingCondition::HardIterationCap {
                cap: 10,
                iteration: 11
            })
        );
    }

    #[test]
    fn detects_stall_when_other_checks_do_not_fire() {
        let out = PureStoppingInputs {
            iteration: 3,
            hard_cap: Some(10),
            timeout_secs: Some(300),
            elapsed_secs: 1,
            task_token_budget: Some(10_000),
            task_tokens_used: 100,
            stall_count: 3,
            max_stall_iterations: 3,
        }
        .evaluate();
        assert_eq!(
            out,
            Some(StoppingCondition::Stall {
                stall_count: 3,
                max_stall_iterations: 3
            })
        );
    }
}
