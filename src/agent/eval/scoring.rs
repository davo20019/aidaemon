use crate::events::TaskOutcome;
use crate::execution_policy::ModelProfile;

use super::accumulator::{ContractSnapshot, CostSnapshot, ProgressSnapshot, RoutingSnapshot};

/// Configurable weights and tier multipliers for harness eval scoring.
#[derive(Debug, Clone)]
pub struct HarnessEvalConfig {
    pub enabled: bool,
    pub weight_routing: f32,
    pub weight_progress: f32,
    pub weight_quality: f32,
    pub weight_cost: f32,
    pub cost_tier_cheap: f32,
    pub cost_tier_balanced: f32,
    pub cost_tier_strong: f32,
    pub cost_tier_unknown: f32,
}

impl Default for HarnessEvalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            weight_routing: 0.30,
            weight_progress: 0.25,
            weight_quality: 0.30,
            weight_cost: 0.15,
            cost_tier_cheap: 1.0,
            cost_tier_balanced: 2.5,
            cost_tier_strong: 5.0,
            cost_tier_unknown: 3.0,
        }
    }
}

impl HarnessEvalConfig {
    pub fn tier_multiplier(&self, profile: Option<ModelProfile>) -> f32 {
        match profile {
            Some(ModelProfile::Cheap) => self.cost_tier_cheap,
            Some(ModelProfile::Balanced) => self.cost_tier_balanced,
            Some(ModelProfile::Strong) => self.cost_tier_strong,
            None => self.cost_tier_unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarnessScores {
    pub routing_accuracy: f32,
    pub progress_yield: f32,
    pub contract_fulfillment: f32,
    pub cost_efficiency: f32,
    pub overall: f32,
}

pub fn compute_routing_accuracy(routing: &RoutingSnapshot) -> f32 {
    let mut score = 1.0_f32;
    if routing.orchestration_route == "direct_reply" && routing.tools_actually_used {
        score -= 0.5;
    }
    if routing.tools_required_predicted
        && !routing.tools_actually_used
        && routing.outcome != TaskOutcome::Succeeded
    {
        score -= 0.3;
    }
    if !routing.tools_required_predicted && routing.tools_actually_used {
        score -= 0.35;
    }
    if routing.route_drift_failsafe {
        score -= 0.2;
    }
    if routing.model_escalated {
        score -= 0.1;
    }
    score.clamp(0.0, 1.0)
}

pub fn compute_progress_yield(progress: &ProgressSnapshot) -> f32 {
    let iterations = progress.iterations.max(1) as f32;
    let yield_units = progress.tool_calls_succeeded as f32
        + progress.observation_count as f32
        + progress.verification_count as f32;
    let mut score = (yield_units / iterations).min(1.0);
    // Conversational/direct-reply turns may legitimately produce zero tool yield;
    // give modest credit when the loop stayed clean (no stall churn).
    if score < f32::EPSILON
        && progress.stall_guard_fires == 0
        && progress.deferred_no_tool_events <= 1
        && progress.iterations <= 3
    {
        score = (1.0 / iterations).min(0.5);
    }
    if progress.stall_guard_fires > 2 {
        score *= 0.8;
    }
    // Planner progress is advisory telemetry. Completion-contract evidence and
    // semantic task outcome determine success; a stale plan cursor must not
    // cap the progress score for otherwise productive work.
    score.clamp(0.0, 1.0)
}

/// Whether all active completion-contract obligations were satisfied.
pub fn contract_is_fulfilled(contract: &ContractSnapshot) -> bool {
    if contract.forbids_mutation
        && (contract.forbidden_mutation_attempts > 0 || contract.mutation_count > 0)
    {
        return false;
    }
    let mut checks = 0_u32;
    let mut passed = 0_u32;
    if contract.requires_observation {
        checks += 1;
        if if contract.evidence_requirements_total > 0 {
            contract.evidence_requirements_satisfied == contract.evidence_requirements_total
        } else {
            contract.observation_count > 0
        } {
            passed += 1;
        }
    }
    if contract.expects_mutation {
        checks += 1;
        if contract.mutation_count > 0 {
            passed += 1;
        }
    }
    if contract.verification_required {
        checks += 1;
        if contract.verification_count > 0 && contract.verification_blocks <= 2 {
            passed += 1;
        }
    }
    checks == 0 || passed == checks
}

pub fn compute_contract_fulfillment(contract: &ContractSnapshot, outcome: TaskOutcome) -> f32 {
    if contract.forbids_mutation
        && (contract.forbidden_mutation_attempts > 0 || contract.mutation_count > 0)
    {
        return 0.0;
    }
    if !contract.expects_mutation
        && !contract.requires_observation
        && !contract.verification_required
    {
        return match outcome {
            TaskOutcome::Succeeded => 1.0,
            TaskOutcome::Partial => 0.5,
            TaskOutcome::Failed => 0.0,
        };
    }

    let mut checks = 0_u32;
    let mut passed = 0_u32;
    if contract.requires_observation {
        checks += 1;
        if if contract.evidence_requirements_total > 0 {
            contract.evidence_requirements_satisfied == contract.evidence_requirements_total
        } else {
            contract.observation_count > 0
        } {
            passed += 1;
        }
    }
    if contract.expects_mutation {
        checks += 1;
        if contract.mutation_count > 0 {
            passed += 1;
        }
    }
    if contract.verification_required {
        checks += 1;
        if contract.verification_count > 0 && contract.verification_blocks <= 2 {
            passed += 1;
        }
    }
    if checks == 0 {
        return match outcome {
            TaskOutcome::Succeeded => 1.0,
            TaskOutcome::Partial => 0.5,
            TaskOutcome::Failed => 0.0,
        };
    }
    let base = passed as f32 / checks as f32;
    match outcome {
        TaskOutcome::Succeeded => base,
        TaskOutcome::Partial => base * 0.75,
        TaskOutcome::Failed => base * 0.25,
    }
}

pub fn compute_cost_efficiency(cost: &CostSnapshot, outcome: TaskOutcome) -> f32 {
    let weighted = cost.weighted_tokens.max(1) as f64;
    let mut score = (1.0 / (1.0 + (weighted / 1000.0).log10().max(0.0))) as f32;
    if outcome != TaskOutcome::Succeeded {
        score *= 0.5;
    }
    score.clamp(0.0, 1.0)
}

pub fn compute_overall(scores: HarnessScores, config: &HarnessEvalConfig) -> f32 {
    let weighted = config.weight_routing * scores.routing_accuracy
        + config.weight_progress * scores.progress_yield
        + config.weight_quality * scores.contract_fulfillment
        + config.weight_cost * scores.cost_efficiency;
    // A severe routing, contract, or cost failure cannot be averaged away by
    // lots of bookkeeping tool calls.
    let hard_cap = if scores.cost_efficiency < 0.15 {
        0.55
    } else if scores.routing_accuracy < 0.5 {
        0.60
    } else {
        1.0
    };
    weighted.clamp(0.0, hard_cap)
}

pub fn build_scores(
    routing: &RoutingSnapshot,
    progress: &ProgressSnapshot,
    contract: &ContractSnapshot,
    cost: &CostSnapshot,
    outcome: TaskOutcome,
    config: &HarnessEvalConfig,
) -> HarnessScores {
    let routing_accuracy = compute_routing_accuracy(routing);
    let progress_yield = compute_progress_yield(progress);
    let contract_fulfillment = compute_contract_fulfillment(contract, outcome);
    let cost_efficiency = compute_cost_efficiency(cost, outcome);
    let partial = HarnessScores {
        routing_accuracy,
        progress_yield,
        contract_fulfillment,
        cost_efficiency,
        overall: 0.0,
    };
    let mut overall = compute_overall(partial, config);
    let negative_contract_violation = contract.forbids_mutation
        && (contract.forbidden_mutation_attempts > 0 || contract.mutation_count > 0);
    if negative_contract_violation
        || (outcome != TaskOutcome::Succeeded && !contract_is_fulfilled(contract))
    {
        overall = overall.min(0.35);
    }
    HarnessScores { overall, ..partial }
}

pub fn weighted_tokens_from_raw(
    input_tokens: u64,
    output_tokens: u64,
    profile: Option<ModelProfile>,
    config: &HarnessEvalConfig,
) -> u64 {
    let raw = input_tokens.saturating_add(output_tokens);
    (raw as f64 * config.tier_multiplier(profile) as f64).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::TaskOutcome;

    fn default_routing() -> RoutingSnapshot {
        RoutingSnapshot {
            orchestration_route: "default_continue".to_string(),
            tools_required_predicted: false,
            tools_actually_used: false,
            direct_return_attempted: false,
            route_drift_failsafe: false,
            skills_activated: vec![],
            policy_profile: Some("balanced".to_string()),
            model_escalated: false,
            outcome: TaskOutcome::Succeeded,
            response_fallthrough: false,
            intent_gate_fires: 0,
            evidence_gate_blocks: 0,
            critique_replan_fires: 0,
        }
    }

    #[test]
    fn routing_penalizes_direct_reply_with_tools() {
        let mut routing = default_routing();
        routing.orchestration_route = "direct_reply".to_string();
        routing.tools_actually_used = true;
        assert!(compute_routing_accuracy(&routing) <= 0.5);
    }

    #[test]
    fn routing_penalizes_unpredicted_tool_use() {
        let mut routing = default_routing();
        routing.tools_actually_used = true;
        assert!(compute_routing_accuracy(&routing) <= 0.65);
    }

    #[test]
    fn forbidden_mutation_attempt_fails_contract_and_caps_overall() {
        let contract = ContractSnapshot {
            forbids_mutation: true,
            forbidden_mutation_attempts: 1,
            ..ContractSnapshot::default()
        };
        assert!(!contract_is_fulfilled(&contract));
        assert_eq!(
            compute_contract_fulfillment(&contract, TaskOutcome::Succeeded),
            0.0
        );
        let scores = build_scores(
            &default_routing(),
            &ProgressSnapshot::default(),
            &contract,
            &CostSnapshot::default(),
            TaskOutcome::Succeeded,
            &HarnessEvalConfig::default(),
        );
        assert!(scores.overall <= 0.35);
    }

    #[test]
    fn advisory_plan_does_not_cap_evidence_yield() {
        let progress = ProgressSnapshot {
            iterations: 10,
            tool_calls_succeeded: 30,
            plan_steps_completed: Some(0),
            plan_steps_total: Some(5),
            ..ProgressSnapshot::default()
        };
        assert_eq!(compute_progress_yield(&progress), 1.0);
    }

    #[test]
    fn cost_efficiency_penalizes_failed_outcomes() {
        let cost = CostSnapshot {
            total_input_tokens: 5000,
            total_output_tokens: 500,
            weighted_tokens: 5500,
            llm_calls: 2,
            fell_back_count: 0,
            sub_agent_weighted_tokens: 0,
            sub_agent_spawn_count: 0,
            sub_agent_failures: 0,
            tokens_failed_waste: true,
        };
        let ok = compute_cost_efficiency(&cost, TaskOutcome::Succeeded);
        let bad = compute_cost_efficiency(&cost, TaskOutcome::Failed);
        assert!(ok > bad);
    }

    #[test]
    fn tier_multiplier_scales_weighted_tokens() {
        let config = HarnessEvalConfig::default();
        let cheap = weighted_tokens_from_raw(1000, 100, Some(ModelProfile::Cheap), &config);
        let strong = weighted_tokens_from_raw(1000, 100, Some(ModelProfile::Strong), &config);
        assert!(strong > cheap);
    }

    #[test]
    fn contract_is_fulfilled_requires_mutation_when_expected() {
        let mut contract = ContractSnapshot {
            expects_mutation: true,
            mutation_count: 0,
            ..Default::default()
        };
        assert!(!contract_is_fulfilled(&contract));
        contract.mutation_count = 1;
        assert!(contract_is_fulfilled(&contract));
    }

    #[test]
    fn raw_observation_count_does_not_replace_material_evidence_closure() {
        let mut contract = ContractSnapshot {
            requires_observation: true,
            observation_count: 4,
            evidence_requirements_total: 2,
            evidence_requirements_satisfied: 1,
            ..Default::default()
        };
        assert!(!contract_is_fulfilled(&contract));
        contract.evidence_requirements_satisfied = 2;
        assert!(contract_is_fulfilled(&contract));
    }

    #[test]
    fn progress_yield_credits_clean_conversational_turns() {
        let progress = ProgressSnapshot {
            iterations: 1,
            ..Default::default()
        };
        assert!(compute_progress_yield(&progress) >= 0.5);
    }
}
