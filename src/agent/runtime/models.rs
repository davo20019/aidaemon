//! A provider reload from the daemon's config: the daemon builds the new
//! backends, the agent swaps them in.

use super::Agent;

/// Rebuild the provider backend + router from `config` and hot-swap them
/// without a daemon restart.
pub(in crate::agent) async fn reload_provider(
    agent: &Agent,
    config: &crate::config::AppConfig,
) -> anyhow::Result<String> {
    let bundle = crate::providers::factory::build_provider_router(config)?;
    let fallback_model = config
        .provider
        .models
        .fallback_models
        .iter()
        .find(|m| m.as_str() != bundle.primary_model)
        .cloned()
        .unwrap_or_else(|| bundle.primary_model.clone());
    agent
        .swap_provider(aidaemon_runtime::agent::ProviderSwap {
            provider: bundle.provider,
            models: bundle.models,
            primary_model: bundle.primary_model,
            router: bundle.router,
            provider_kind: bundle.provider_kind,
            failover_targets: bundle.failover_targets,
            fallback_model,
            decisions: crate::llm_runtime::decide::DecisionLayer::from_config_with_key(
                &config.decisions,
                config.gemini_provider_api_key().as_deref(),
            ),
        })
        .await
}
