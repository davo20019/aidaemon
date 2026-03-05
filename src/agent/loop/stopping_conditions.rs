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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StallMode {
    Default,
    DeferredNoTool,
    Transient,
    EmptyResponse,
}

impl StallMode {
    pub fn as_code(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::DeferredNoTool => "deferred_no_tool",
            Self::Transient => "transient",
            Self::EmptyResponse => "empty_response",
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopControlDecision {
    HardIterationCap {
        cap: usize,
        iteration: usize,
    },
    TaskTimeout {
        timeout_secs: u64,
        elapsed_secs: u64,
    },
    PreToolDeferral {
        deferred_no_tool_streak: usize,
        max_pre_tool_deferrals: usize,
    },
    Stall {
        stall_count: usize,
        max_stall_iterations: usize,
        mode: StallMode,
    },
}

/// Inputs for the unified loop-control evaluator.
#[derive(Debug, Clone)]
pub struct LoopControlInputs<'a> {
    pub iteration: usize,
    pub hard_cap: Option<usize>,
    pub timeout_secs: Option<u64>,
    pub elapsed_secs: u64,
    pub stall_count: usize,
    pub max_stall_iterations: usize,
    pub deferred_no_tool_streak: usize,
    pub deferred_no_tool_switch_threshold: usize,
    pub deferred_no_tool_error_marker: &'a str,
    pub max_pre_tool_deferrals: usize,
    pub total_successful_tool_calls: usize,
    pub recent_errors: &'a [(String, bool)],
}

impl<'a> LoopControlInputs<'a> {
    fn stall_limit_and_mode(&self) -> (usize, StallMode) {
        let recent_errors = self
            .recent_errors
            .iter()
            .rev()
            .take(8)
            .map(|(e, _)| e.to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        if self.deferred_no_tool_streak >= self.deferred_no_tool_switch_threshold
            || recent_errors.contains(self.deferred_no_tool_error_marker)
        {
            // Give extra room for deferred/no-tool recovery windows.
            return (
                self.max_stall_iterations.saturating_add(3),
                StallMode::DeferredNoTool,
            );
        }

        let transient_signals = recent_errors.contains("rate limit")
            || recent_errors.contains("too many requests")
            || recent_errors.contains("429")
            || recent_errors.contains("timed out")
            || recent_errors.contains("timeout")
            || recent_errors.contains("network")
            || recent_errors.contains("connection")
            || recent_errors.contains("service unavailable")
            || recent_errors.contains("bad gateway")
            || recent_errors.contains("gateway timeout");
        if transient_signals {
            return (
                self.max_stall_iterations.saturating_add(2),
                StallMode::Transient,
            );
        }

        if recent_errors.contains("empty_response(") || recent_errors.contains("empty response") {
            return (
                self.max_stall_iterations.saturating_add(2),
                StallMode::EmptyResponse,
            );
        }

        (self.max_stall_iterations, StallMode::Default)
    }

    /// Evaluate loop controls in priority order.
    ///
    /// Priority:
    /// 1) hard iteration cap
    /// 2) task timeout
    /// 3) pre-tool deferral stall
    /// 4) post-tool stall
    pub fn evaluate(&self) -> Option<LoopControlDecision> {
        if let Some(cap) = self.hard_cap {
            if self.iteration > cap {
                return Some(LoopControlDecision::HardIterationCap {
                    cap,
                    iteration: self.iteration,
                });
            }
        }

        if let Some(timeout_secs) = self.timeout_secs {
            if self.elapsed_secs > timeout_secs {
                return Some(LoopControlDecision::TaskTimeout {
                    timeout_secs,
                    elapsed_secs: self.elapsed_secs,
                });
            }
        }

        if self.total_successful_tool_calls == 0
            && self.deferred_no_tool_streak >= self.max_pre_tool_deferrals
        {
            return Some(LoopControlDecision::PreToolDeferral {
                deferred_no_tool_streak: self.deferred_no_tool_streak,
                max_pre_tool_deferrals: self.max_pre_tool_deferrals,
            });
        }

        let (stall_limit, stall_mode) = self.stall_limit_and_mode();
        if self.stall_count >= stall_limit {
            return Some(LoopControlDecision::Stall {
                stall_count: self.stall_count,
                max_stall_iterations: stall_limit,
                mode: stall_mode,
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

    #[test]
    fn loop_control_prefers_pre_tool_deferral_before_stall() {
        let out = LoopControlInputs {
            iteration: 5,
            hard_cap: None,
            timeout_secs: None,
            elapsed_secs: 0,
            stall_count: 99,
            max_stall_iterations: 3,
            deferred_no_tool_streak: 6,
            deferred_no_tool_switch_threshold: 2,
            deferred_no_tool_error_marker: "deferred-action no-tool loop",
            max_pre_tool_deferrals: 6,
            total_successful_tool_calls: 0,
            recent_errors: &[],
        }
        .evaluate();
        assert_eq!(
            out,
            Some(LoopControlDecision::PreToolDeferral {
                deferred_no_tool_streak: 6,
                max_pre_tool_deferrals: 6
            })
        );
    }

    #[test]
    fn loop_control_uses_deferred_no_tool_stall_window() {
        let errors = vec![("deferred-action no-tool loop".to_string(), false)];
        let out = LoopControlInputs {
            iteration: 5,
            hard_cap: None,
            timeout_secs: None,
            elapsed_secs: 0,
            stall_count: 6,
            max_stall_iterations: 3,
            deferred_no_tool_streak: 2,
            deferred_no_tool_switch_threshold: 2,
            deferred_no_tool_error_marker: "deferred-action no-tool loop",
            max_pre_tool_deferrals: 99,
            total_successful_tool_calls: 2,
            recent_errors: &errors,
        }
        .evaluate();
        assert_eq!(
            out,
            Some(LoopControlDecision::Stall {
                stall_count: 6,
                max_stall_iterations: 6,
                mode: StallMode::DeferredNoTool
            })
        );
    }

    #[test]
    fn loop_control_detects_transient_stall_mode() {
        let errors = vec![("network timeout while calling tool".to_string(), false)];
        let out = LoopControlInputs {
            iteration: 5,
            hard_cap: None,
            timeout_secs: None,
            elapsed_secs: 0,
            stall_count: 5,
            max_stall_iterations: 3,
            deferred_no_tool_streak: 0,
            deferred_no_tool_switch_threshold: 2,
            deferred_no_tool_error_marker: "deferred-action no-tool loop",
            max_pre_tool_deferrals: 6,
            total_successful_tool_calls: 1,
            recent_errors: &errors,
        }
        .evaluate();
        assert_eq!(
            out,
            Some(LoopControlDecision::Stall {
                stall_count: 5,
                max_stall_iterations: 5,
                mode: StallMode::Transient
            })
        );
    }
}
