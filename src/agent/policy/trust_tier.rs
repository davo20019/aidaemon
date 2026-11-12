//! Model trust tiers live in `aidaemon-runtime`; the agent combines its
//! `[policy] trust_tier` setting with the model id.

pub use aidaemon_runtime::policy::trust_tier::{
    classify_model_trust, ModelTrustTier, TrustTierSetting,
};

impl crate::agent::Agent {
    /// Trust tier for the model handling the current call, combining the
    /// `[policy] trust_tier` config override with the model-id heuristic.
    pub(crate) fn trust_tier_for_model(&self, model: &str) -> ModelTrustTier {
        classify_model_trust(
            model,
            TrustTierSetting::parse(&self.policy_config.trust_tier),
        )
    }
}
