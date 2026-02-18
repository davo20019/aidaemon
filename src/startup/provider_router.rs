use std::sync::Arc;

use tracing::info;

use crate::config::{AppConfig, ProviderKind};
use crate::router::Router;
use crate::traits::ModelProvider;

pub struct ProviderRouterBundle {
    pub provider: Arc<dyn ModelProvider>,
    pub primary_model: String,
}

pub fn build_provider_router(config: &AppConfig) -> anyhow::Result<ProviderRouterBundle> {
    let provider: Arc<dyn crate::traits::ModelProvider> = match config.provider.kind {
        ProviderKind::OpenaiCompatible => Arc::new(
            crate::providers::OpenAiCompatibleProvider::new_with_gateway_token(
                &config.provider.base_url,
                &config.provider.api_key,
                config.provider.gateway_token.as_deref(),
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?,
        ),
        ProviderKind::GoogleGenai => Arc::new(crate::providers::GoogleGenAiProvider::new(
            &config.provider.api_key,
        )),
        ProviderKind::Anthropic => Arc::new(crate::providers::AnthropicNativeProvider::new(
            &config.provider.api_key,
        )),
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
