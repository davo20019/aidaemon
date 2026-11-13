//! Model trust tiers live in `aidaemon-runtime`; the agent combines its
//! `[policy] trust_tier` setting with the model id.

pub use aidaemon_runtime::policy::trust_tier::ModelTrustTier;

impl crate::agent::Agent {
    /// Trust tier for the model handling the current call, combining the
    /// `[policy] trust_tier` config override with the model-id heuristic.
    pub(crate) fn trust_tier_for_model(&self, model: &str) -> ModelTrustTier {
        self.runtime.trust_tier_for_model(model)
    }
}
