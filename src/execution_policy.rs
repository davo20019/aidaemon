use serde::{Deserialize, Serialize};

use crate::traits::ToolCapabilities;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelProfile {
    Cheap,
    Balanced,
    Strong,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VerifyLevel {
    Quick,
    Full,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalMode {
    Auto,
    AskOnce,
    AskAlways,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPolicy {
    pub model_profile: ModelProfile,
    pub verify_level: VerifyLevel,
    pub approval_mode: ApprovalMode,
    pub context_budget: usize,
    pub tool_budget: usize,
    pub policy_rev: u32,
    #[serde(default)]
    pub escalation_reasons: Vec<String>,
}

impl ExecutionPolicy {
    pub fn for_profile(profile: ModelProfile) -> Self {
        match profile {
            ModelProfile::Cheap => Self {
                model_profile: ModelProfile::Cheap,
                verify_level: VerifyLevel::Quick,
                approval_mode: ApprovalMode::Auto,
                context_budget: 6000,
                tool_budget: 6,
                policy_rev: 1,
                escalation_reasons: Vec::new(),
            },
            ModelProfile::Balanced => Self {
                model_profile: ModelProfile::Balanced,
                verify_level: VerifyLevel::Quick,
                approval_mode: ApprovalMode::AskOnce,
                context_budget: 12_000,
                tool_budget: 12,
                policy_rev: 1,
                escalation_reasons: Vec::new(),
            },
            ModelProfile::Strong => Self {
                model_profile: ModelProfile::Strong,
                verify_level: VerifyLevel::Full,
                approval_mode: ApprovalMode::AskAlways,
                context_budget: 20_000,
                tool_budget: 20,
                policy_rev: 1,
                escalation_reasons: Vec::new(),
            },
        }
    }

    pub fn escalate(&mut self, reason: impl Into<String>) -> bool {
        let reason = reason.into();
        let next = match self.model_profile {
            ModelProfile::Cheap => Some(ModelProfile::Balanced),
            ModelProfile::Balanced => Some(ModelProfile::Strong),
            ModelProfile::Strong => None,
        };
        if let Some(profile) = next {
            let mut replacement = Self::for_profile(profile);
            replacement.policy_rev = self.policy_rev.saturating_add(1);
            replacement
                .escalation_reasons
                .extend(self.escalation_reasons.iter().cloned());
            replacement.escalation_reasons.push(reason);
            *self = replacement;
            true
        } else {
            false
        }
    }

    pub fn deescalate(&mut self) -> bool {
        let next = match self.model_profile {
            ModelProfile::Strong => Some(ModelProfile::Balanced),
            ModelProfile::Balanced => Some(ModelProfile::Cheap),
            ModelProfile::Cheap => None,
        };
        if let Some(profile) = next {
            let mut replacement = Self::for_profile(profile);
            replacement.policy_rev = self.policy_rev.saturating_add(1);
            replacement
                .escalation_reasons
                .extend(self.escalation_reasons.iter().cloned());
            *self = replacement;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyBundle {
    pub policy: ExecutionPolicy,
    pub risk_score: f32,
    pub uncertainty_score: f32,
    pub confidence: f32,
}

impl PolicyBundle {
    pub fn from_scores(risk_score: f32, uncertainty_score: f32, confidence: f32) -> Self {
        let profile = profile_for_risk(risk_score, uncertainty_score);
        Self {
            policy: ExecutionPolicy::for_profile(profile),
            risk_score: clamp01(risk_score),
            uncertainty_score: clamp01(uncertainty_score),
            confidence: clamp01(confidence),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct UncertaintySignals {
    pub missing_required_slot: bool,
    pub conflicting_constraints: bool,
    pub ambiguous_wording: bool,
    pub prior_immediate_failure: bool,
}

pub fn score_uncertainty_v1(signals: UncertaintySignals) -> f32 {
    let mut score = 0.0f32;
    if signals.missing_required_slot {
        score += 0.35;
    }
    if signals.conflicting_constraints {
        score += 0.25;
    }
    if signals.ambiguous_wording {
        score += 0.20;
    }
    if signals.prior_immediate_failure {
        score += 0.20;
    }
    clamp01(score)
}

pub fn score_risk_from_capabilities(caps: &[ToolCapabilities]) -> f32 {
    if caps.is_empty() {
        return 0.25;
    }

    let mut aggregate = 0.0f32;
    for cap in caps {
        let mut local = 0.0f32;
        if !cap.read_only {
            local += 0.12;
        }
        if cap.external_side_effect {
            local += 0.18;
        }
        if cap.needs_approval {
            local += 0.07;
        }
        if !cap.idempotent {
            local += 0.05;
        }
        if cap.high_impact_write {
            local += 0.20;
        }
        aggregate += local;
    }
    let avg = aggregate / caps.len() as f32;
    clamp01(0.12 + avg)
}

pub fn profile_for_risk(risk_score: f32, uncertainty_score: f32) -> ModelProfile {
    let composite = clamp01((risk_score * 0.7) + (uncertainty_score * 0.3));
    if composite < 0.34 {
        ModelProfile::Cheap
    } else if composite < 0.67 {
        ModelProfile::Balanced
    } else {
        ModelProfile::Strong
    }
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_profile_defaults_match_plan() {
        let cheap = ExecutionPolicy::for_profile(ModelProfile::Cheap);
        assert_eq!(cheap.context_budget, 6000);
        assert_eq!(cheap.tool_budget, 6);
        assert_eq!(cheap.verify_level, VerifyLevel::Quick);
        assert_eq!(cheap.approval_mode, ApprovalMode::Auto);

        let balanced = ExecutionPolicy::for_profile(ModelProfile::Balanced);
        assert_eq!(balanced.context_budget, 12_000);
        assert_eq!(balanced.tool_budget, 12);
        assert_eq!(balanced.verify_level, VerifyLevel::Quick);
        assert_eq!(balanced.approval_mode, ApprovalMode::AskOnce);

        let strong = ExecutionPolicy::for_profile(ModelProfile::Strong);
        assert_eq!(strong.context_budget, 20_000);
        assert_eq!(strong.tool_budget, 20);
        assert_eq!(strong.verify_level, VerifyLevel::Full);
        assert_eq!(strong.approval_mode, ApprovalMode::AskAlways);
    }

    #[test]
    fn uncertainty_v1_weights() {
        let score = score_uncertainty_v1(UncertaintySignals {
            missing_required_slot: true,
            conflicting_constraints: true,
            ambiguous_wording: false,
            prior_immediate_failure: false,
        });
        assert!((score - 0.60).abs() < 0.001);
    }

    #[test]
    fn escalation_bounds() {
        let mut policy = ExecutionPolicy::for_profile(ModelProfile::Balanced);
        assert!(policy.escalate("stall"));
        assert_eq!(policy.model_profile, ModelProfile::Strong);
        assert!(!policy.escalate("already_max"));
        assert_eq!(policy.model_profile, ModelProfile::Strong);
    }

    #[test]
    fn deescalation_bounds() {
        let mut policy = ExecutionPolicy::for_profile(ModelProfile::Balanced);
        assert!(policy.deescalate());
        assert_eq!(policy.model_profile, ModelProfile::Cheap);
        assert!(!policy.deescalate());
        assert_eq!(policy.model_profile, ModelProfile::Cheap);
    }
}
