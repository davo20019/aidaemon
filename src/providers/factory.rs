use std::sync::Arc;

use tracing::{info, warn};

use crate::config::{AppConfig, ProviderConfig, ProviderKind};
use crate::llm_runtime::ProviderRuntimeTarget;
use crate::router::Router;

use super::sdk_backed::ModelList;

pub(crate) struct ProviderRouterBundle {
    pub provider: crate::llm_runtime::ModelBackend,
    pub models: ModelList,
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

/// The daemon's ChatGPT OAuth state as the SDK backend's credentials: the
/// process-wide manager serializes refresh-token rotation across every
/// subscription-backed adapter.
struct DaemonChatGptAuth {
    manager: Arc<crate::oauth::chatgpt_codex::ChatGptCredentialManager>,
    client: reqwest::Client,
}

#[async_trait::async_trait]
impl aidaemon_providers::ChatGptAuth for DaemonChatGptAuth {
    async fn credentials(&self) -> Result<aidaemon_providers::ChatGptCredentials, String> {
        self.manager
            .usable_credentials(&self.client)
            .await
            .map(sdk_credentials)
            .map_err(|error| error.to_string())
    }

    async fn refresh_after_rejection(
        &self,
        rejected_token: &str,
    ) -> Result<aidaemon_providers::ChatGptCredentials, String> {
        self.manager
            .refresh_after_rejection(&self.client, rejected_token)
            .await
            .map(sdk_credentials)
            .map_err(|error| error.to_string())
    }

    async fn mark_rejected(&self, token: &str, detail: &str) {
        self.manager.mark_rejected(token, detail).await;
    }
}

fn sdk_credentials(
    credentials: crate::oauth::chatgpt_codex::ChatGptCredentials,
) -> aidaemon_providers::ChatGptCredentials {
    aidaemon_providers::ChatGptCredentials {
        access_token: credentials.access_token,
        account_id: credentials.account_id,
    }
}

/// The ChatGPT-subscription backend on the daemon's OAuth login.
pub(crate) fn chatgpt_backend(
    base_override: Option<&str>,
    reasoning_effort: Option<&str>,
) -> anyhow::Result<aidaemon_providers::ChatGpt> {
    let client =
        super::build_http_client(std::time::Duration::from_secs(60)).map_err(anyhow::Error::msg)?;
    let mut builder = aidaemon_providers::ChatGpt::builder()
        .base_url(base_override.unwrap_or(crate::oauth::chatgpt_codex::CODEX_BACKEND_BASE_URL))
        .auth(DaemonChatGptAuth {
            manager: crate::oauth::chatgpt_codex::shared_credential_manager(),
            client,
        });
    if let Some(effort) = reasoning_effort {
        builder = builder.reasoning_effort(effort);
    }
    builder.build().map_err(|error| anyhow::anyhow!("{error}"))
}

/// The configured models, for backends whose API the daemon does not list.
fn configured_models(config: &ProviderConfig) -> Vec<String> {
    let models = &config.models;
    let mut out = vec![
        models.primary.clone(),
        models.fast.clone(),
        models.smart.clone(),
    ];
    out.retain(|model| !model.is_empty());
    out.dedup();
    out
}

/// The SDK backend for `config`'s kind, and how its models are listed.
pub(crate) fn build_provider(
    config: &ProviderConfig,
) -> anyhow::Result<(crate::llm_runtime::ModelBackend, ModelList)> {
    use aidaemon_providers as backends;
    let config_error = |error: backends::ConfigError| anyhow::anyhow!("{error}");
    let base_override = provider_specific_base_url_override(config);
    let headers = config.extra_headers.clone().unwrap_or_default();
    let mut models = ModelList::Fixed(configured_models(config));
    let backend: Arc<dyn aidaemon_core::Provider> = match config.kind {
        ProviderKind::OpenaiCompatible => {
            if config.llama_cpp_thinking
                || config.slot_routing.enabled
                || !config.structured_answer_models.is_empty()
            {
                warn!(
                    "llama_cpp_thinking, slot_routing and structured_answer_models are no \
                     longer supported and are ignored"
                );
            }
            let mut builder = backends::OpenAiCompatible::builder()
                .base_url(&config.base_url)
                .api_key(&config.api_key)
                .timeout(std::time::Duration::from_secs(360));
            if let Some(token) = config.gateway_token.as_deref().filter(|t| !t.is_empty()) {
                builder = builder.header("cf-aig-authorization", format!("Bearer {token}"));
            }
            for (name, value) in &headers {
                builder = builder.header(name, value);
            }
            if let Some(max) = config.max_tokens {
                builder = builder.max_tokens(max);
            }
            if let Some(effort) = &config.reasoning_effort {
                builder = builder.reasoning_effort(effort);
            }
            models = ModelList::OpenAiModels {
                base_url: config.base_url.clone(),
                api_key: Some(config.api_key.clone()).filter(|key| !key.is_empty()),
            };
            Arc::new(builder.build().map_err(config_error)?)
        }
        ProviderKind::XaiNative => {
            let mut builder = backends::Xai::builder()
                .api_key(&config.api_key)
                .timeout(std::time::Duration::from_secs(120));
            if let Some(base) = base_override {
                builder = builder.base_url(base);
            }
            for (name, value) in &headers {
                builder = builder.header(name, value);
            }
            if let Some(max) = config.max_tokens {
                builder = builder.max_tokens(max);
            }
            Arc::new(builder.build().map_err(config_error)?)
        }
        ProviderKind::GoogleGenai => {
            let mut builder = backends::GoogleGenai::builder()
                .api_key(&config.api_key)
                .timeout(std::time::Duration::from_secs(360));
            if let Some(base) = base_override {
                builder = builder.base_url(base);
            }
            for (name, value) in &headers {
                builder = builder.header(name, value);
            }
            Arc::new(builder.build().map_err(config_error)?)
        }
        ProviderKind::Anthropic => {
            let mut builder = backends::Anthropic::builder()
                .api_key(&config.api_key)
                .timeout(std::time::Duration::from_secs(360));
            if let Some(base) = base_override {
                builder = builder.base_url(base);
            }
            for (name, value) in &headers {
                builder = builder.header(name, value);
            }
            if let Some(max) = config.max_tokens {
                builder = builder.max_tokens(max);
            }
            if let Some(effort) = &config.reasoning_effort {
                builder = builder.reasoning_effort(effort);
            }
            Arc::new(builder.build().map_err(config_error)?)
        }
        ProviderKind::OpenaiChatgpt => {
            if config.max_tokens.is_some() {
                warn!(
                    "provider.max_tokens is ignored by the ChatGPT subscription provider; \
                     the Codex backend accepts no output cap"
                );
            }
            models = ModelList::Fixed(
                backends::CHATGPT_KNOWN_MODELS
                    .iter()
                    .map(|model| model.to_string())
                    .collect(),
            );
            Arc::new(chatgpt_backend(
                base_override,
                config.reasoning_effort.as_deref(),
            )?)
        }
    };
    Ok((backend, models))
}

fn build_provider_target(config: &ProviderConfig) -> anyhow::Result<ProviderRuntimeTarget> {
    let (provider, models) = build_provider(config)?;

    let router = Router::new(config.models.clone());
    let primary_model = router.default_model().to_string();
    let router = (!router.is_uniform()).then_some(router);

    Ok(
        ProviderRuntimeTarget::new(provider, router, config.kind, primary_model)
            .with_models(models),
    )
}

pub(crate) fn build_provider_router(config: &AppConfig) -> anyhow::Result<ProviderRouterBundle> {
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
        models: primary_target.models().clone(),
        primary_model,
        router: primary_router,
        provider_kind: primary_target.provider_kind(),
        failover_targets,
    })
}
