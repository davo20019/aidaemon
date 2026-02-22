use std::sync::Arc;

use tracing::info;

use crate::config::{AppConfig, ProviderKind};
use crate::router::Router;
use crate::traits::ModelProvider;

pub struct ProviderRouterBundle {
    pub provider: Arc<dyn ModelProvider>,
    pub primary_model: String,
}

const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

fn provider_specific_base_url_override(config: &AppConfig) -> Option<&str> {
    let trimmed = config.provider.base_url.trim();
    if trimmed.is_empty() || trimmed == OPENAI_DEFAULT_BASE_URL {
        None
    } else {
        Some(trimmed)
    }
}

pub fn build_provider_router(config: &AppConfig) -> anyhow::Result<ProviderRouterBundle> {
    let provider_base_override = provider_specific_base_url_override(config);

    let provider: Arc<dyn crate::traits::ModelProvider> = match config.provider.kind {
        ProviderKind::OpenaiCompatible => Arc::new(
            crate::providers::OpenAiCompatibleProvider::new_with_gateway_token_and_headers(
                &config.provider.base_url,
                &config.provider.api_key,
                config.provider.gateway_token.as_deref(),
                config.provider.extra_headers.clone(),
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?,
        ),
        ProviderKind::GoogleGenai => Arc::new(
            crate::providers::GoogleGenAiProvider::new_with_base_url_and_headers(
                &config.provider.api_key,
                provider_base_override,
                config.provider.extra_headers.clone(),
            ),
        ),
        ProviderKind::Anthropic => {
            Arc::new(crate::providers::AnthropicNativeProvider::new_with_options(
                &config.provider.api_key,
                provider_base_override,
                config.provider.max_tokens,
                config.provider.extra_headers.clone(),
            ))
        }
    };

    let router = Router::new(config.provider.models.clone());
    let primary_model = router.default_model().to_string();
    info!(
        default_model = router.default_model(),
        fallbacks = ?router.fallback_models(),
        "Model router configured"
    );

    Ok(ProviderRouterBundle {
        provider,
        primary_model,
    })
}
