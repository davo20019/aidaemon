use crate::agent::{CompletionContract, CompletionProgress};
use crate::events::{
    HarnessEvalSnapshot, HarnessScoresPayload, QualityEvalPayload, RoutingEvalPayload,
    TaskEfficiencyData, TaskOutcome,
};
use crate::execution_policy::ModelProfile;

use super::scoring::{
    build_scores, contract_is_fulfilled, weighted_tokens_from_raw, HarnessEvalConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Budget/Timeout wired incrementally; all variants used in snapshots
pub enum StopReason {
    Completed,
    Stall,
    Budget,
    Timeout,
    Error,
    Cancelled,
    DirectReturn,
    Other,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::Completed => "completed",
            StopReason::Stall => "stall",
            StopReason::Budget => "budget",
            StopReason::Timeout => "timeout",
            StopReason::Error => "error",
            StopReason::Cancelled => "cancelled",
            StopReason::DirectReturn => "direct_return",
            StopReason::Other => "other",
        }
    }
}

#[derive(Debug, Clone)]
pub struct HarnessEvalSeed {
    pub task_id: String,
    pub turn_id: Option<String>,
    pub depth: u32,
    pub parent_task_id: Option<String>,
    pub goal_id: Option<String>,
    pub durable_task_id: Option<String>,
    pub completion_task_kind: String,
    pub followup_mode: Option<String>,
    pub config: HarnessEvalConfig,
}

#[derive(Debug, Clone)]
pub struct RoutingSnapshot {
    pub orchestration_route: String,
    pub tools_required_predicted: bool,
    pub tools_actually_used: bool,
    pub direct_return_attempted: bool,
    pub route_drift_failsafe: bool,
    pub skills_activated: Vec<String>,
    pub policy_profile: Option<String>,
    pub model_escalated: bool,
    pub outcome: TaskOutcome,
    pub response_fallthrough: bool,
    pub intent_gate_fires: u32,
    pub evidence_gate_blocks: u32,
    pub critique_replan_fires: u32,
}

#[derive(Debug, Clone, Default)]
pub struct ProgressSnapshot {
    pub iterations: u32,
    pub tool_calls_attempted: u32,
    pub tool_calls_succeeded: u32,
    pub evidence_gain_total: u32,
    pub no_progress_iterations: u32,
    pub stall_guard_fires: u32,
    pub repetition_guard_fires: u32,
    pub observation_count: u32,
    pub verification_count: u32,
    pub plan_steps_completed: Option<u32>,
    pub plan_steps_total: Option<u32>,
    pub tool_defs_count: u32,
    pub est_input_tokens: u32,
    pub context_drops: u32,
    pub deferred_no_tool_events: u32,
    pub budget_extensions: u32,
}

#[derive(Debug, Clone, Default)]
pub struct ContractSnapshot {
    pub expects_mutation: bool,
    pub forbids_mutation: bool,
    pub forbidden_mutation_attempts: u32,
    pub mutation_count: u32,
    pub requires_observation: bool,
    pub observation_count: u32,
    pub verification_required: bool,
    pub verification_count: u32,
    pub verification_blocks: u32,
}

#[derive(Debug, Clone, Default)]
pub struct CostSnapshot {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub weighted_tokens: u64,
    pub llm_calls: u32,
    pub fell_back_count: u32,
    pub sub_agent_weighted_tokens: u64,
    pub sub_agent_spawn_count: u32,
    pub sub_agent_failures: u32,
    pub tokens_failed_waste: bool,
}

/// Mutable per-task harness metrics collected during the agent loop.
#[derive(Debug, Clone)]
pub struct HarnessEvalAccumulator {
    seed: HarnessEvalSeed,
    orchestration_route: String,
    tools_required_predicted: bool,
    tools_actually_used: bool,
    direct_return_attempted: bool,
    direct_return_succeeded: bool,
    route_drift_failsafe: bool,
    skills_activated: Vec<String>,
    policy_profile: Option<String>,
    model_escalated: bool,
    model_profile: Option<ModelProfile>,
    progress: ProgressSnapshot,
    contract: ContractSnapshot,
    post_exec_validation_failures: u32,
    unrecovered_errors: u32,
    approval_denied: bool,
    stop_reason: StopReason,
    raw_input_tokens: u64,
    raw_output_tokens: u64,
    weighted_tokens: u64,
    llm_calls: u32,
    fell_back_count: u32,
    sub_agent_weighted_tokens: u64,
    sub_agent_spawn_count: u32,
    sub_agent_failures: u32,
    response_fallthrough: bool,
    intent_gate_fires: u32,
    evidence_gate_blocks: u32,
    critique_replan_fires: u32,
}

impl HarnessEvalAccumulator {
    pub fn new(seed: HarnessEvalSeed) -> Self {
        Self {
            orchestration_route: "default_continue".to_string(),
            tools_required_predicted: false,
            tools_actually_used: false,
            direct_return_attempted: false,
            direct_return_succeeded: false,
            route_drift_failsafe: false,
            skills_activated: Vec::new(),
            policy_profile: None,
            model_escalated: false,
            model_profile: None,
            progress: ProgressSnapshot::default(),
            contract: ContractSnapshot::default(),
            post_exec_validation_failures: 0,
            unrecovered_errors: 0,
            approval_denied: false,
            stop_reason: StopReason::Other,
            raw_input_tokens: 0,
            raw_output_tokens: 0,
            weighted_tokens: 0,
            llm_calls: 0,
            fell_back_count: 0,
            sub_agent_weighted_tokens: 0,
            sub_agent_spawn_count: 0,
            sub_agent_failures: 0,
            response_fallthrough: false,
            intent_gate_fires: 0,
            evidence_gate_blocks: 0,
            critique_replan_fires: 0,
            seed,
        }
    }

    pub fn set_completion_context(
        &mut self,
        completion_task_kind: impl Into<String>,
        followup_mode: Option<String>,
    ) {
        self.seed.completion_task_kind = completion_task_kind.into();
        self.seed.followup_mode = followup_mode;
    }

    pub fn record_bootstrap(
        &mut self,
        orchestration_route: impl Into<String>,
        skills_activated: Vec<String>,
        policy_profile: Option<ModelProfile>,
        route_failsafe_active: bool,
    ) {
        self.orchestration_route = orchestration_route.into();
        self.skills_activated = skills_activated;
        self.policy_profile = policy_profile.map(model_profile_label);
        self.model_profile = policy_profile;
        self.route_drift_failsafe = route_failsafe_active;
    }

    pub fn record_orchestration_route(&mut self, route: impl Into<String>, tools_required: bool) {
        self.orchestration_route = route.into();
        self.tools_required_predicted = tools_required;
    }

    pub fn record_direct_return(&mut self, attempted: bool, succeeded: bool) {
        self.direct_return_attempted = attempted;
        self.direct_return_succeeded = succeeded;
        if succeeded {
            self.stop_reason = StopReason::DirectReturn;
        }
    }

    pub fn record_response_fallthrough(&mut self) {
        self.response_fallthrough = true;
    }

    pub fn record_message_build(
        &mut self,
        tool_defs_count: u32,
        est_input_tokens: u32,
        context_drops: u32,
    ) {
        self.progress.tool_defs_count = tool_defs_count;
        self.progress.est_input_tokens = est_input_tokens;
        self.progress.context_drops = context_drops;
    }

    pub fn record_intent_gate_fire(&mut self) {
        self.intent_gate_fires = self.intent_gate_fires.saturating_add(1);
    }

    pub fn record_evidence_gate_block(&mut self) {
        self.evidence_gate_blocks = self.evidence_gate_blocks.saturating_add(1);
    }

    pub fn record_critique_replan(&mut self) {
        self.critique_replan_fires = self.critique_replan_fires.saturating_add(1);
    }

    pub fn record_deferred_no_tool_event(&mut self) {
        self.progress.deferred_no_tool_events =
            self.progress.deferred_no_tool_events.saturating_add(1);
    }

    pub fn record_budget_extension(&mut self) {
        self.progress.budget_extensions = self.progress.budget_extensions.saturating_add(1);
    }

    pub fn record_model_escalated(&mut self) {
        self.model_escalated = true;
    }

    pub fn record_iteration_progress(
        &mut self,
        iteration: u32,
        tool_calls_attempted: u32,
        tool_calls_succeeded: u32,
        evidence_gain: u32,
        no_progress: bool,
    ) {
        self.progress.iterations = iteration;
        self.progress.tool_calls_attempted = tool_calls_attempted;
        self.progress.tool_calls_succeeded = tool_calls_succeeded;
        self.progress.evidence_gain_total = evidence_gain;
        if no_progress {
            self.progress.no_progress_iterations =
                self.progress.no_progress_iterations.saturating_add(1);
        }
        if tool_calls_succeeded > 0 {
            self.tools_actually_used = true;
        }
    }

    pub fn record_stall_guard(&mut self) {
        self.progress.stall_guard_fires = self.progress.stall_guard_fires.saturating_add(1);
    }

    pub fn record_repetition_guard(&mut self) {
        self.progress.repetition_guard_fires =
            self.progress.repetition_guard_fires.saturating_add(1);
    }

    pub fn record_completion_contract(&mut self, contract: &CompletionContract) {
        self.contract.expects_mutation = contract.expects_mutation;
        self.contract.forbids_mutation = contract.forbids_mutation;
        self.contract.requires_observation = contract.requires_observation;
        self.contract.verification_required = contract.explicit_verification_requested
            || contract.requires_reverification_after_mutation;
    }

    pub fn record_forbidden_mutation_attempt(&mut self) {
        self.contract.forbidden_mutation_attempts =
            self.contract.forbidden_mutation_attempts.saturating_add(1);
    }

    pub fn record_plan_progress(&mut self, completed: u32, total: u32) {
        self.progress.plan_steps_completed = Some(completed);
        self.progress.plan_steps_total = Some(total);
    }

    pub fn record_completion_progress(&mut self, progress: &CompletionProgress) {
        self.contract.mutation_count = progress.mutation_count as u32;
        self.contract.observation_count = progress.observation_count as u32;
        self.contract.verification_count = progress.verification_count as u32;
        self.contract.verification_blocks = progress.verification_block_count as u32;
        self.progress.observation_count = progress.observation_count as u32;
        self.progress.verification_count = progress.verification_count as u32;
    }

    pub fn record_post_exec_validation_failure(&mut self) {
        self.post_exec_validation_failures = self.post_exec_validation_failures.saturating_add(1);
    }

    #[allow(dead_code)] // wired when unrecovered error paths gain eval hooks
    pub fn record_unrecovered_error(&mut self) {
        self.unrecovered_errors = self.unrecovered_errors.saturating_add(1);
    }

    pub fn record_approval_denied(&mut self) {
        self.approval_denied = true;
    }

    pub fn record_stop_reason(&mut self, reason: StopReason) {
        self.stop_reason = reason;
    }

    pub fn record_llm_efficiency(&mut self, efficiency: &TaskEfficiencyData) {
        self.raw_input_tokens = efficiency.input_tokens;
        self.raw_output_tokens = efficiency.output_tokens;
        self.llm_calls = efficiency.llm_calls as u32;
        self.fell_back_count = efficiency.fell_back_count as u32;
        self.weighted_tokens = weighted_tokens_from_raw(
            efficiency.input_tokens,
            efficiency.output_tokens,
            self.model_profile,
            &self.seed.config,
        ) + self.sub_agent_weighted_tokens;
    }

    pub fn rollup_sub_agent(&mut self, child: &HarnessEvalSnapshot) {
        self.sub_agent_spawn_count = self.sub_agent_spawn_count.saturating_add(1);
        self.sub_agent_weighted_tokens = self
            .sub_agent_weighted_tokens
            .saturating_add(child.cost.weighted_tokens);
        self.progress.tool_calls_attempted = self
            .progress
            .tool_calls_attempted
            .saturating_add(child.progress.tool_calls_attempted);
        self.progress.tool_calls_succeeded = self
            .progress
            .tool_calls_succeeded
            .saturating_add(child.progress.tool_calls_succeeded);
        self.progress.iterations = self.progress.iterations.max(child.progress.iterations);
        if child.quality.outcome == "failed" {
            self.sub_agent_failures = self.sub_agent_failures.saturating_add(1);
        }
        if child.routing.tools_actually_used {
            self.tools_actually_used = true;
        }
    }

    pub fn finalize(
        self,
        outcome: TaskOutcome,
        iterations: u32,
        tool_calls_count: u32,
        efficiency: Option<&TaskEfficiencyData>,
    ) -> HarnessEvalSnapshot {
        let mut acc = self;
        acc.progress.iterations = acc.progress.iterations.max(iterations);
        acc.progress.tool_calls_attempted = acc.progress.tool_calls_attempted.max(tool_calls_count);
        if tool_calls_count > 0 {
            acc.tools_actually_used = true;
        }
        if let Some(eff) = efficiency {
            acc.record_llm_efficiency(eff);
        }
        if acc.stop_reason == StopReason::Other {
            acc.stop_reason = match outcome {
                TaskOutcome::Succeeded | TaskOutcome::Partial => StopReason::Completed,
                TaskOutcome::Failed => StopReason::Error,
            };
        }
        if acc.direct_return_attempted && !acc.direct_return_succeeded {
            acc.stop_reason = StopReason::Error;
        }
        if outcome == TaskOutcome::Failed {
            let failed_calls = acc
                .progress
                .tool_calls_attempted
                .saturating_sub(acc.progress.tool_calls_succeeded);
            if failed_calls > 0 {
                acc.unrecovered_errors = acc.unrecovered_errors.saturating_add(failed_calls);
            }
        }

        let routing = RoutingSnapshot {
            orchestration_route: acc.orchestration_route.clone(),
            tools_required_predicted: acc.tools_required_predicted,
            tools_actually_used: acc.tools_actually_used,
            direct_return_attempted: acc.direct_return_attempted,
            route_drift_failsafe: acc.route_drift_failsafe,
            skills_activated: acc.skills_activated.clone(),
            policy_profile: acc.policy_profile.clone(),
            model_escalated: acc.model_escalated,
            outcome,
            response_fallthrough: acc.response_fallthrough,
            intent_gate_fires: acc.intent_gate_fires,
            evidence_gate_blocks: acc.evidence_gate_blocks,
            critique_replan_fires: acc.critique_replan_fires,
        };

        let cost = CostSnapshot {
            total_input_tokens: acc.raw_input_tokens,
            total_output_tokens: acc.raw_output_tokens,
            weighted_tokens: acc.weighted_tokens,
            llm_calls: acc.llm_calls,
            fell_back_count: acc.fell_back_count,
            sub_agent_weighted_tokens: acc.sub_agent_weighted_tokens,
            sub_agent_spawn_count: acc.sub_agent_spawn_count,
            sub_agent_failures: acc.sub_agent_failures,
            tokens_failed_waste: outcome != TaskOutcome::Succeeded,
        };

        let scores = build_scores(
            &routing,
            &acc.progress,
            &acc.contract,
            &cost,
            outcome,
            &acc.seed.config,
        );

        let contract_fulfilled = contract_is_fulfilled(&acc.contract);

        HarnessEvalSnapshot {
            task_id: acc.seed.task_id,
            turn_id: acc.seed.turn_id,
            depth: acc.seed.depth,
            parent_task_id: acc.seed.parent_task_id,
            goal_id: acc.seed.goal_id,
            durable_task_id: acc.seed.durable_task_id,
            completion_task_kind: acc.seed.completion_task_kind,
            orchestration_route: acc.orchestration_route,
            followup_mode: acc.seed.followup_mode,
            routing: RoutingEvalPayload {
                orchestration_route: routing.orchestration_route,
                tools_required_predicted: routing.tools_required_predicted,
                tools_actually_used: routing.tools_actually_used,
                direct_return_attempted: routing.direct_return_attempted,
                route_drift_failsafe: routing.route_drift_failsafe,
                skills_activated: routing.skills_activated,
                policy_profile: routing.policy_profile,
                model_escalated: routing.model_escalated,
                response_fallthrough: routing.response_fallthrough,
                intent_gate_fires: routing.intent_gate_fires,
                evidence_gate_blocks: routing.evidence_gate_blocks,
                critique_replan_fires: routing.critique_replan_fires,
            },
            progress: acc.progress.into(),
            quality: QualityEvalPayload {
                outcome: outcome.as_str().to_string(),
                stop_reason: acc.stop_reason.as_str().to_string(),
                contract: acc.contract.into(),
                post_exec_validation_failures: acc.post_exec_validation_failures,
                unrecovered_errors: acc.unrecovered_errors,
                approval_denied: acc.approval_denied,
                contract_fulfilled,
            },
            cost: cost.into(),
            scores: HarnessScoresPayload {
                routing_accuracy: scores.routing_accuracy,
                progress_yield: scores.progress_yield,
                contract_fulfillment: scores.contract_fulfillment,
                cost_efficiency: scores.cost_efficiency,
                overall: scores.overall,
            },
        }
    }
}

fn model_profile_label(profile: ModelProfile) -> String {
    match profile {
        ModelProfile::Cheap => "cheap".to_string(),
        ModelProfile::Balanced => "balanced".to_string(),
        ModelProfile::Strong => "strong".to_string(),
    }
}

impl From<ProgressSnapshot> for crate::events::ProgressEvalPayload {
    fn from(value: ProgressSnapshot) -> Self {
        Self {
            iterations: value.iterations,
            tool_calls_attempted: value.tool_calls_attempted,
            tool_calls_succeeded: value.tool_calls_succeeded,
            evidence_gain_total: value.evidence_gain_total,
            no_progress_iterations: value.no_progress_iterations,
            stall_guard_fires: value.stall_guard_fires,
            repetition_guard_fires: value.repetition_guard_fires,
            plan_steps_completed: value.plan_steps_completed,
            plan_steps_total: value.plan_steps_total,
            tool_defs_count: value.tool_defs_count,
            est_input_tokens: value.est_input_tokens,
            context_drops: value.context_drops,
            deferred_no_tool_events: value.deferred_no_tool_events,
            budget_extensions: value.budget_extensions,
        }
    }
}

impl From<ContractSnapshot> for crate::events::ContractFulfillmentPayload {
    fn from(value: ContractSnapshot) -> Self {
        Self {
            expects_mutation: value.expects_mutation,
            forbids_mutation: value.forbids_mutation,
            forbidden_mutation_attempts: value.forbidden_mutation_attempts,
            mutation_count: value.mutation_count,
            requires_observation: value.requires_observation,
            observation_count: value.observation_count,
            verification_required: value.verification_required,
            verification_count: value.verification_count,
            verification_blocks: value.verification_blocks,
            fulfilled: contract_is_fulfilled(&value),
        }
    }
}

impl From<CostSnapshot> for crate::events::CostEvalPayload {
    fn from(value: CostSnapshot) -> Self {
        Self {
            total_input_tokens: value.total_input_tokens,
            total_output_tokens: value.total_output_tokens,
            weighted_tokens: value.weighted_tokens,
            llm_calls: value.llm_calls,
            fell_back_count: value.fell_back_count,
            sub_agent_weighted_tokens: value.sub_agent_weighted_tokens,
            sub_agent_spawn_count: value.sub_agent_spawn_count,
            sub_agent_failures: value.sub_agent_failures,
            tokens_failed_waste: value.tokens_failed_waste,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{
        ContractFulfillmentPayload, CostEvalPayload, HarnessScoresPayload, ProgressEvalPayload,
        QualityEvalPayload, RoutingEvalPayload,
    };

    fn sample_child_snapshot(
        weighted_tokens: u64,
        tool_calls: u32,
        outcome: &str,
    ) -> HarnessEvalSnapshot {
        HarnessEvalSnapshot {
            task_id: "child".to_string(),
            turn_id: None,
            depth: 1,
            parent_task_id: Some("parent".to_string()),
            goal_id: Some("goal".to_string()),
            durable_task_id: Some("durable-child".to_string()),
            completion_task_kind: "conversational".to_string(),
            orchestration_route: "default_continue".to_string(),
            followup_mode: None,
            routing: RoutingEvalPayload {
                orchestration_route: "default_continue".to_string(),
                tools_required_predicted: false,
                tools_actually_used: tool_calls > 0,
                direct_return_attempted: false,
                route_drift_failsafe: false,
                skills_activated: Vec::new(),
                policy_profile: None,
                model_escalated: false,
                response_fallthrough: false,
                intent_gate_fires: 0,
                evidence_gate_blocks: 0,
                critique_replan_fires: 0,
            },
            progress: ProgressEvalPayload {
                iterations: 2,
                tool_calls_attempted: tool_calls,
                tool_calls_succeeded: tool_calls,
                evidence_gain_total: 0,
                no_progress_iterations: 0,
                stall_guard_fires: 0,
                repetition_guard_fires: 0,
                plan_steps_completed: None,
                plan_steps_total: None,
                tool_defs_count: 0,
                est_input_tokens: 0,
                context_drops: 0,
                deferred_no_tool_events: 0,
                budget_extensions: 0,
            },
            quality: QualityEvalPayload {
                outcome: outcome.to_string(),
                stop_reason: "completed".to_string(),
                contract: ContractFulfillmentPayload {
                    expects_mutation: false,
                    forbids_mutation: false,
                    forbidden_mutation_attempts: 0,
                    mutation_count: 0,
                    requires_observation: false,
                    observation_count: 0,
                    verification_required: false,
                    verification_count: 0,
                    verification_blocks: 0,
                    fulfilled: true,
                },
                post_exec_validation_failures: 0,
                unrecovered_errors: 0,
                approval_denied: false,
                contract_fulfilled: true,
            },
            cost: CostEvalPayload {
                total_input_tokens: 100,
                total_output_tokens: 50,
                weighted_tokens,
                llm_calls: 1,
                fell_back_count: 0,
                sub_agent_weighted_tokens: 0,
                sub_agent_spawn_count: 0,
                sub_agent_failures: 0,
                tokens_failed_waste: outcome != "succeeded",
            },
            scores: HarnessScoresPayload {
                routing_accuracy: 1.0,
                progress_yield: 1.0,
                contract_fulfillment: 1.0,
                cost_efficiency: 1.0,
                overall: 1.0,
            },
        }
    }

    #[test]
    fn rollup_sub_agent_merges_cost_and_progress() {
        let mut acc = HarnessEvalAccumulator::new(HarnessEvalSeed {
            task_id: "parent".to_string(),
            turn_id: None,
            depth: 0,
            parent_task_id: None,
            goal_id: Some("goal".to_string()),
            durable_task_id: None,
            completion_task_kind: "conversational".to_string(),
            followup_mode: None,
            config: HarnessEvalConfig::default(),
        });
        acc.rollup_sub_agent(&sample_child_snapshot(500, 2, "succeeded"));
        acc.rollup_sub_agent(&sample_child_snapshot(300, 1, "failed"));
        let snap = acc.finalize(TaskOutcome::Succeeded, 1, 3, None);
        assert_eq!(snap.cost.sub_agent_spawn_count, 2);
        assert_eq!(snap.cost.sub_agent_failures, 1);
        assert_eq!(snap.cost.sub_agent_weighted_tokens, 800);
        assert_eq!(snap.progress.tool_calls_attempted, 3);
        assert_eq!(snap.progress.tool_calls_succeeded, 3);
        assert!(snap.routing.tools_actually_used);
    }
}
