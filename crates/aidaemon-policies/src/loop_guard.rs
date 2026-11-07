//! The consecutive-same-tool guard: a model that calls one tool over and over
//! without making progress gets its next call refused with a redirect.
//!
//! The decision is a pure function over the run's previous calls, so the
//! same rule serves every harness built on the kernel. The limits are
//! data: a harness picks its own numbers and its own per-tool allowance.

use std::collections::{HashMap, HashSet};

use aidaemon_core::{Hook, HookContext, SameToolStreak, ToolCall, ToolDecision, ToolInfo};
use async_trait::async_trait;
use serde_json::Value;
use tracing::warn;

/// The numbers behind each [`SameToolStreak`] allowance: `Standard` uses
/// `base`, `Versatile` uses `versatile`, and `Stepwise` uses `stepwise` (with
/// no read-only exemption).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreakLimits {
    pub base: usize,
    pub versatile: usize,
    pub stepwise: usize,
}

impl Default for StreakLimits {
    fn default() -> Self {
        Self {
            base: 8,
            versatile: 16,
            stepwise: 40,
        }
    }
}

/// Whether the call about to run continues a same-tool loop.
///
/// `previous` is the run's `(tool, arguments)` history, oldest first. The
/// streak is the trailing run of calls to `tool`, plus this call. A streak
/// shorter than its limit is never a loop. Past the limit, distinct
/// arguments can still be progress: a read-only tool whose calls all differ
/// is allowed, and a streak that is mostly varied waits for the wider limit.
pub fn same_tool_loop(
    previous: &[(String, Value)],
    tool: &str,
    arguments: &Value,
    read_only: bool,
    allowance: SameToolStreak,
    limits: StreakLimits,
) -> bool {
    let mut streak: Vec<&Value> = previous
        .iter()
        .rev()
        .take_while(|(name, _)| name == tool)
        .map(|(_, args)| args)
        .collect();
    streak.push(arguments);
    let total = streak.len();
    let stepwise = allowance == SameToolStreak::Stepwise;
    let limit = match allowance {
        SameToolStreak::Stepwise => limits.stepwise,
        SameToolStreak::Versatile => limits.versatile,
        SameToolStreak::Standard => limits.base,
    };
    if total < limit {
        return false;
    }
    let mut unique: Vec<&Value> = Vec::new();
    for args in streak {
        if !unique.contains(&args) {
            unique.push(args);
        }
    }
    if read_only && !stepwise && unique.len() == total {
        return false;
    }
    let diverse = unique.len() * 2 > total;
    let diverse_limit = if stepwise {
        limits.stepwise
    } else {
        limits.versatile
    };
    !diverse || total >= diverse_limit
}

/// The loop guard as a kernel hook. It refuses the looping call with a
/// redirect; if the model keeps at it, the kernel's stuck detector ends the
/// run.
pub struct SameToolStreakGuard {
    /// Tools that are read-only for every call.
    pub read_only_tools: HashSet<String>,
    /// Each tool's allowance. A tool not listed gets [`SameToolStreak::Standard`].
    pub allowances: HashMap<String, SameToolStreak>,
    pub limits: StreakLimits,
}

#[async_trait]
impl Hook for SameToolStreakGuard {
    async fn before_tool(
        &self,
        ctx: &HookContext<'_>,
        call: &ToolCall,
        _tool: &ToolInfo,
    ) -> ToolDecision {
        let previous: Vec<(String, Value)> = ctx
            .run_receipts()
            .map(|receipt| (receipt.tool.clone(), receipt.arguments.clone()))
            .collect();
        let allowance = self.allowances.get(&call.name).copied().unwrap_or_default();
        if !same_tool_loop(
            &previous,
            &call.name,
            &call.arguments,
            self.read_only_tools.contains(&call.name),
            allowance,
            self.limits,
        ) {
            return ToolDecision::Allow;
        }
        warn!(tool = %call.name, "same tool called too many consecutive times");
        ToolDecision::Deny {
            reason: format!(
                "[LOOP DETECTED] `{}` has been called too many times in a row without \
                 finishing. Stop calling it: use a different approach, or answer with what \
                 you have and the exact limitation.",
                call.name
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn history(tool: &str, n: usize) -> Vec<(String, Value)> {
        (0..n)
            .map(|i| (tool.to_string(), json!({"i": i})))
            .collect()
    }

    #[test]
    fn default_allowance_refuses_at_the_base_limit() {
        let limits = StreakLimits::default();
        // Seven prior identical calls plus this one is eight: at the limit.
        let same = json!({"x": 1});
        let prior: Vec<(String, Value)> = (0..7).map(|_| ("t".into(), same.clone())).collect();
        assert!(same_tool_loop(
            &prior,
            "t",
            &same,
            false,
            SameToolStreak::Standard,
            limits
        ));
        assert!(!same_tool_loop(
            &prior[..6],
            "t",
            &same,
            false,
            SameToolStreak::Standard,
            limits
        ));
    }

    #[test]
    fn read_only_tool_with_distinct_arguments_is_not_a_loop_until_the_wider_limit() {
        let limits = StreakLimits::default();
        let prior = history("read", 8);
        let next = json!({"i": 99});
        assert!(!same_tool_loop(
            &prior,
            "read",
            &next,
            true,
            SameToolStreak::Standard,
            limits
        ));
    }

    #[test]
    fn stepwise_allowance_uses_its_own_limit_and_drops_the_read_only_exemption() {
        let limits = StreakLimits::default();
        let prior = history("step", 39);
        let next = json!({"i": 999});
        assert!(same_tool_loop(
            &prior,
            "step",
            &next,
            true,
            SameToolStreak::Stepwise,
            limits
        ));
    }

    #[test]
    fn a_different_tool_breaks_the_streak() {
        let limits = StreakLimits::default();
        let mut prior = history("a", 10);
        prior.push(("b".into(), json!({})));
        assert!(!same_tool_loop(
            &prior,
            "a",
            &json!({}),
            false,
            SameToolStreak::Standard,
            limits
        ));
    }
}
