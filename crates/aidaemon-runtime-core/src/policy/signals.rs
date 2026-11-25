use std::collections::HashMap;

use super::execution::{PolicyBundle, score_risk_from_capabilities};
use aidaemon_tools::semantics::ToolCapabilities;

pub fn build_policy_bundle(
    _user_text: &str,
    available_capabilities: &HashMap<String, ToolCapabilities>,
    prior_immediate_failure: bool,
) -> PolicyBundle {
    // Bootstrap has tool capabilities and prior execution state, but it does
    // not yet have a proposed action. Do not manufacture risk, ambiguity, or
    // missing targets from request wording. Concrete calls are checked later
    // against typed capabilities, arguments, approvals, and receipts.
    let risk_score =
        score_risk_from_capabilities(&available_capabilities.values().copied().collect::<Vec<_>>());
    let uncertainty_score: f32 = if prior_immediate_failure { 0.65 } else { 0.0 };
    let confidence = (1.0 - uncertainty_score).clamp(0.0, 1.0);
    PolicyBundle::from_scores(risk_score, uncertainty_score, confidence)
}

#[cfg(test)]
#[path = "signal_tests.rs"]
mod policy_signal_tests;
