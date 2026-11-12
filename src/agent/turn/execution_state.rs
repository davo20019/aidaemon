//! The run's execution state lives in `aidaemon-runtime`; the budget's
//! selection keeps the daemon's signature here.

pub(crate) use aidaemon_runtime::turn::execution_state::*;

use crate::agent::TurnContext;
use crate::traits::AgentRole;

pub(in crate::agent) fn select_initial_execution_budget(
    _user_text: &str,
    _turn_context: &TurnContext,
    depth: usize,
    role: AgentRole,
) -> (BudgetTier, &'static str, ExecutionBudget) {
    aidaemon_runtime::turn::execution_state::select_initial_execution_budget(depth, role)
}

#[cfg(test)]
#[path = "execution_state_tests.rs"]
mod tests;
