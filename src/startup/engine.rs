//! The turn engine's settings from the daemon's config (`[agent.engine]`).

use tracing::{info, warn};

use crate::config::{AppConfig, ProviderKind};
use aidaemon_runtime::engine::settings::{
    log_configured_trust_roots, SdkEngineSettings, DEFAULT_MAX_TURNS,
};

/// The engine settings from `[agent.engine]`.
pub(crate) fn settings_from_config(
    config: &AppConfig,
    backend: &dyn crate::execution::ExecutionBackend,
) -> SdkEngineSettings {
    let engine = &config.agent.engine;
    if engine.kind.is_some() {
        warn!("[agent.engine] kind is ignored: the SDK kernel is the only agent loop");
    }
    if config.state.daily_token_budget.is_some() {
        warn!(
            "[state] daily_token_budget is ignored: no global daily token cap is enforced; \
             use per-goal daily budgets, subagents.task_token_budget or a provider spend limit"
        );
    }
    if config.subagents.task_timeout_secs.is_some() {
        warn!(
            "[subagents] task_timeout_secs is ignored: runs are bounded by progress and cost \
             (inactivity watchdog, per-call timeouts, limits, token budgets), not elapsed time"
        );
    }
    let mut model_override = engine
        .model
        .clone()
        .filter(|model| !model.trim().is_empty());
    if !engine.api_key.trim().is_empty() {
        warn!(
            "[agent.engine] api_key is ignored: the SDK engine uses the daemon's configured \
             providers"
        );
        // An engine api_key marks a config written for the engine's former
        // dedicated Anthropic client; its model pin named a model for that
        // client, which the daemon's own provider may not serve.
        if config.provider.kind != ProviderKind::Anthropic
            && let Some(model) = model_override.take() {
                warn!(
                    model = %model,
                    "[agent.engine] model is ignored: it was configured for the removed \
                     Anthropic engine client; set it without api_key to pin a model the \
                     daemon's provider serves"
                );
            }
    }
    let settings = SdkEngineSettings {
        model_override,
        max_turns: engine.max_turns.unwrap_or(DEFAULT_MAX_TURNS).max(1),
        owner_default: config.authority.owner_default,
        trusted_roots: config.authority.trusted_roots.clone(),
        outbound_secrets: aidaemon_runtime::approvals::ConfiguredSecrets::new(
            config.outbound_secret_values(),
        ),
    };
    info!(model_override = ?settings.model_override, "Agent engine: SDK kernel");
    log_configured_trust_roots(
        &settings.trusted_roots,
        backend.workspace_root().as_str().to_string(),
    );
    settings
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_config(provider_kind: &str, engine: &str) -> AppConfig {
        toml::from_str(&format!(
            "[provider]\nkind = \"{provider_kind}\"\napi_key = \"test-key\"\n\n\
             [provider.models]\ndefault = \"model-a\"\n\n[agent.engine]\n{engine}"
        ))
        .expect("parse app config")
    }

    #[test]
    fn legacy_engine_model_pin_is_dropped_for_a_non_anthropic_provider() {
        let config = engine_config(
            "openai_chatgpt",
            "model = \"claude-haiku-5-5\"\napi_key = \"sk-ant-test\"",
        );
        let settings =
            settings_from_config(&config, &*crate::execution::active_execution_backend());
        assert_eq!(settings.model_override, None);
    }

    #[test]
    fn legacy_engine_model_pin_is_kept_for_an_anthropic_provider() {
        let config = engine_config(
            "anthropic",
            "model = \"claude-haiku-5-5\"\napi_key = \"sk-ant-test\"",
        );
        let settings =
            settings_from_config(&config, &*crate::execution::active_execution_backend());
        assert_eq!(settings.model_override.as_deref(), Some("claude-haiku-5-5"));
    }

    #[test]
    fn engine_model_pin_without_api_key_is_honored() {
        let config = engine_config("openai_chatgpt", "model = \"gpt-6-luna\"");
        let settings =
            settings_from_config(&config, &*crate::execution::active_execution_backend());
        assert_eq!(settings.model_override.as_deref(), Some("gpt-6-luna"));
    }
}
