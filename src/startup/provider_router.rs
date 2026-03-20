use std::sync::Arc;

use tracing::info;

use crate::config::{AppConfig, ProviderConfig, ProviderKind};
use crate::llm_runtime::ProviderRuntimeTarget;
use crate::router::Router;
use crate::traits::ModelProvider;

pub struct ProviderRouterBundle {
    pub provider: Arc<dyn ModelProvider>,
    pub primary_model: String,
    pub router: Option<Router>,
    pub provider_kind: ProviderKind,
    pub failover_targets: Vec<ProviderRuntimeTarget>,
}

const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

fn provider_specific_base_url_override(config: &ProviderConfig) -> Option<&str> {
    let trimmed = config.base_url.trim();
    if trimmed.is_empty() || trimmed == OPENAI_DEFAULT_BASE_URL {
        None
    } else {
        Some(trimmed)
    }
}

fn build_provider_target(config: &ProviderConfig) -> anyhow::Result<ProviderRuntimeTarget> {
    let provider_base_override = provider_specific_base_url_override(config);

    let provider: Arc<dyn crate::traits::ModelProvider> = match config.kind {
        ProviderKind::OpenaiCompatible => Arc::new(
            crate::providers::OpenAiCompatibleProvider::new_with_all_options(
                &config.base_url,
                &config.api_key,
                config.gateway_token.as_deref(),
                config.extra_headers.clone(),
                config.max_tokens,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .with_reasoning_effort(config.reasoning_effort.clone()),
        ),
        ProviderKind::XaiNative => Arc::new(
            crate::providers::XaiNativeProvider::new_with_options(
                &config.api_key,
                provider_base_override,
                config.max_tokens,
                config.extra_headers.clone(),
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?,
        ),
        ProviderKind::GoogleGenai => Arc::new(
            crate::providers::GoogleGenAiProvider::new_with_base_url_and_headers(
                &config.api_key,
                provider_base_override,
                config.extra_headers.clone(),
            ),
        ),
        ProviderKind::Anthropic => {
            Arc::new(crate::providers::AnthropicNativeProvider::new_with_options(
                &config.api_key,
                provider_base_override,
                config.max_tokens,
                config.extra_headers.clone(),
            ))
        }
    };

    let router = Router::new(config.models.clone());
    let primary_model = router.default_model().to_string();
    let router = if router.is_uniform() {
        None
    } else {
        Some(router)
    };

    Ok(ProviderRuntimeTarget::new(
        provider,
        router,
        config.kind,
        primary_model,
    ))
}

pub fn build_provider_router(config: &AppConfig) -> anyhow::Result<ProviderRouterBundle> {
    let primary_target = build_provider_target(&config.provider)?;
    let failover_targets = config
        .provider
        .fallbacks
        .iter()
        .map(build_provider_target)
        .collect::<anyhow::Result<Vec<_>>>()?;

    let primary_router = primary_target.router();
    let primary_model = primary_target.primary_model().to_string();
    info!(
        default_model = primary_model.as_str(),
        fallbacks = ?primary_router
            .as_ref()
            .map(|router| router.fallback_models().to_vec())
            .unwrap_or_default(),
        failover_providers = failover_targets.len(),
        "Model router configured"
    );

    Ok(ProviderRouterBundle {
        provider: primary_target.provider(),
        primary_model,
        router: primary_router,
        provider_kind: primary_target.provider_kind(),
        failover_targets,
    })
}
