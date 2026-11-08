use std::sync::{Arc, RwLock};

use crate::config::{ModelsConfig, ProviderKind};
use crate::providers::sdk_backed::ModelList;
use crate::router::Router;

/// A model backend: any SDK provider.
pub type ModelBackend = Arc<dyn aidaemon_core::Provider>;

#[derive(Clone)]
pub struct ProviderRuntimeTarget {
    provider: ModelBackend,
    router: Option<Router>,
    provider_kind: ProviderKind,
    primary_model: String,
    models: ModelList,
}

impl ProviderRuntimeTarget {
    pub fn new(
        provider: ModelBackend,
        router: Option<Router>,
        provider_kind: ProviderKind,
        primary_model: String,
    ) -> Self {
        Self {
            models: ModelList::Fixed(vec![primary_model.clone()]),
            provider,
            router,
            provider_kind,
            primary_model,
        }
    }

    /// How this provider's models are listed.
    pub(crate) fn with_models(mut self, models: ModelList) -> Self {
        self.models = models;
        self
    }

    pub(crate) fn models(&self) -> &ModelList {
        &self.models
    }

    pub fn provider(&self) -> ModelBackend {
        self.provider.clone()
    }

    pub fn router(&self) -> Option<Router> {
        self.router.clone()
    }

    pub fn provider_kind(&self) -> ProviderKind {
        self.provider_kind
    }

    pub fn primary_model(&self) -> &str {
        &self.primary_model
    }

    pub fn all_models_ordered(&self) -> Vec<String> {
        self.router
            .as_ref()
            .map(|router| router.all_models_ordered())
            .filter(|models| !models.is_empty())
            .unwrap_or_else(|| vec![self.primary_model.clone()])
    }

    /// The same models with this provider's fast tier first, for calls that
    /// do not need its strongest model.
    pub(crate) fn models_cheap_first(&self) -> Vec<String> {
        cheap_first(self.router.as_ref(), self.all_models_ordered())
    }
}

/// `models` reordered so the router's fast tier, when it is one of them,
/// comes first.
fn cheap_first(router: Option<&Router>, mut models: Vec<String>) -> Vec<String> {
    if let Some(fast) = router.map(|router| router.select(crate::router::Tier::Fast).to_string()) {
        if let Some(index) = models.iter().position(|model| *model == fast) {
            let fast = models.remove(index);
            models.insert(0, fast);
        }
    }
    models
}

#[derive(Clone)]
pub struct LlmRuntimeSnapshot {
    provider: ModelBackend,
    router: Option<Router>,
    provider_kind: ProviderKind,
    primary_model: String,
    failover_targets: Vec<ProviderRuntimeTarget>,
    models: ModelList,
}

impl LlmRuntimeSnapshot {
    /// The primary provider's model listing.
    pub(crate) fn models(&self) -> &ModelList {
        &self.models
    }

    pub fn provider(&self) -> ModelBackend {
        self.provider.clone()
    }

    pub fn router(&self) -> Option<Router> {
        self.router.clone()
    }

    pub fn provider_kind(&self) -> ProviderKind {
        self.provider_kind
    }

    pub fn primary_model(&self) -> String {
        self.primary_model.clone()
    }

    pub fn failover_targets(&self) -> Vec<ProviderRuntimeTarget> {
        self.failover_targets.clone()
    }

    /// The chain a background call (memory consolidation, summaries, skill
    /// promotion) runs on: the primary model, the router's fallbacks, then
    /// every failover provider's models, under the SDK's default policy, which
    /// moves on after a used-up quota or an auth failure. Agent turns build
    /// their own chain with the daemon's recovery policy.
    ///
    /// `first_model` (usually a primary-provider model such as a cheaper
    /// tier) is tried first; the primary provider's other models follow.
    pub(crate) fn background_chain(&self, first_model: &str) -> aidaemon_core::Failover {
        use aidaemon_core::FailoverStep;
        let mut primary_models = vec![first_model.to_string()];
        primary_models.extend(
            self.router
                .as_ref()
                .map(Router::all_models_ordered)
                .unwrap_or_else(|| vec![self.primary_model.clone()])
                .into_iter()
                .filter(|model| model != first_model),
        );
        let mut steps: Vec<FailoverStep> = primary_models
            .into_iter()
            .map(|model| FailoverStep::from_arc(self.provider.clone()).model(model))
            .collect();
        for target in &self.failover_targets {
            for model in target.models_cheap_first() {
                steps.push(FailoverStep::from_arc(target.provider()).model(model));
            }
        }
        aidaemon_core::Failover::new(steps)
    }

    /// The primary provider's fast-tier model, for background and auxiliary
    /// calls (memory consolidation, summaries, assessment) that do not need
    /// the strongest model.
    pub(crate) fn cheap_model(&self) -> String {
        self.router
            .as_ref()
            .map(|router| router.select(crate::router::Tier::Fast).to_string())
            .filter(|model| !model.trim().is_empty())
            .unwrap_or_else(|| self.primary_model.clone())
    }

    /// `primary_models` followed by every failover provider's models (fast
    /// tier first), for callers that try candidates one by one themselves.
    /// Run them on [`Self::model_routed_backend`].
    pub(crate) fn with_failover_models(&self, mut primary_models: Vec<String>) -> Vec<String> {
        for target in &self.failover_targets {
            for model in target.models_cheap_first() {
                if !primary_models.contains(&model) {
                    primary_models.push(model);
                }
            }
        }
        primary_models
    }

    /// One backend that sends each request to the provider that owns its
    /// model: the primary provider, else the first failover provider listing
    /// it. Unknown models go to the primary.
    pub(crate) fn model_routed_backend(&self) -> ModelBackend {
        let routes = self
            .failover_targets
            .iter()
            .flat_map(|target| {
                let provider = target.provider();
                target
                    .all_models_ordered()
                    .into_iter()
                    .map(move |model| (model, provider.clone()))
            })
            .collect();
        let primary_models = self
            .router
            .as_ref()
            .map(Router::all_models_ordered)
            .unwrap_or_else(|| vec![self.primary_model.clone()]);
        Arc::new(ModelRoutedBackend {
            primary: self.provider.clone(),
            primary_models,
            routes,
        })
    }

    /// One background call on [`Self::background_chain`], starting at `model`.
    pub(crate) async fn background_chat(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
    ) -> anyhow::Result<crate::traits::ProviderResponse> {
        use crate::providers::sdk_backed::ChatJson;
        self.background_chain(model)
            .chat(model, messages, tools)
            .await
    }
}

#[derive(Clone)]
struct LlmRuntimeState {
    provider: ModelBackend,
    router: Option<Router>,
    provider_kind: ProviderKind,
    primary_model: String,
    failover_targets: Vec<ProviderRuntimeTarget>,
    models: ModelList,
}

#[derive(Clone)]
pub struct SharedLlmRuntime(Arc<RwLock<LlmRuntimeState>>);

impl SharedLlmRuntime {
    #[allow(dead_code)] // Retained for single-provider call sites and tests.
    pub fn new(
        provider: ModelBackend,
        router: Option<Router>,
        provider_kind: ProviderKind,
        primary_model: String,
    ) -> Self {
        Self::new_with_failovers(provider, router, provider_kind, primary_model, Vec::new())
    }

    pub fn new_with_failovers(
        provider: ModelBackend,
        router: Option<Router>,
        provider_kind: ProviderKind,
        primary_model: String,
        failover_targets: Vec<ProviderRuntimeTarget>,
    ) -> Self {
        Self(Arc::new(RwLock::new(LlmRuntimeState {
            models: ModelList::Fixed(vec![primary_model.clone()]),
            provider,
            router,
            provider_kind,
            primary_model,
            failover_targets,
        })))
    }

    /// How the primary provider's models are listed.
    pub(crate) fn with_models(self, models: ModelList) -> Self {
        self.0.write().expect("llm runtime lock poisoned").models = models;
        self
    }

    pub fn snapshot(&self) -> LlmRuntimeSnapshot {
        let guard = self.0.read().expect("llm runtime lock poisoned");
        LlmRuntimeSnapshot {
            provider: guard.provider.clone(),
            router: guard.router.clone(),
            provider_kind: guard.provider_kind,
            primary_model: guard.primary_model.clone(),
            failover_targets: guard.failover_targets.clone(),
            models: guard.models.clone(),
        }
    }

    pub fn router(&self) -> Option<Router> {
        self.snapshot().router()
    }

    pub fn swap(
        &self,
        provider: ModelBackend,
        router: Option<Router>,
        provider_kind: ProviderKind,
        primary_model: String,
        failover_targets: Vec<ProviderRuntimeTarget>,
        models: ModelList,
    ) -> LlmRuntimeSnapshot {
        let mut guard = self.0.write().expect("llm runtime lock poisoned");
        let old = LlmRuntimeSnapshot {
            provider: guard.provider.clone(),
            router: guard.router.clone(),
            provider_kind: guard.provider_kind,
            primary_model: guard.primary_model.clone(),
            failover_targets: guard.failover_targets.clone(),
            models: std::mem::replace(&mut guard.models, models),
        };
        guard.provider = provider;
        guard.router = router;
        guard.provider_kind = provider_kind;
        guard.primary_model = primary_model;
        guard.failover_targets = failover_targets;
        old
    }
}

#[allow(dead_code)] // Retained for test helpers and lightweight single-provider wiring.
pub fn router_from_models(models: ModelsConfig) -> Option<Router> {
    let router = Router::new(models);
    if router.is_uniform() {
        None
    } else {
        Some(router)
    }
}

/// See [`LlmRuntimeSnapshot::model_routed_backend`].
struct ModelRoutedBackend {
    primary: ModelBackend,
    primary_models: Vec<String>,
    routes: Vec<(String, ModelBackend)>,
}

impl ModelRoutedBackend {
    fn backend_for(&self, model: &str) -> &ModelBackend {
        if self.primary_models.iter().any(|known| known == model) {
            return &self.primary;
        }
        self.routes
            .iter()
            .find(|(known, _)| known == model)
            .map(|(_, provider)| provider)
            .unwrap_or(&self.primary)
    }
}

#[async_trait::async_trait]
impl aidaemon_core::Provider for ModelRoutedBackend {
    fn name(&self) -> &str {
        "model-routed"
    }

    async fn complete(
        &self,
        request: &aidaemon_core::ModelRequest,
    ) -> Result<aidaemon_core::ModelResponse, aidaemon_core::ProviderError> {
        self.backend_for(&request.model).complete(request).await
    }

    fn stream<'a>(
        &'a self,
        request: &'a aidaemon_core::ModelRequest,
    ) -> futures::stream::BoxStream<
        'a,
        Result<aidaemon_core::StreamEvent, aidaemon_core::ProviderError>,
    > {
        self.backend_for(&request.model).stream(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::testing::{text, ScriptedProvider};

    /// A background call whose primary provider has used up its quota is
    /// answered by the failover provider, not dropped.
    #[tokio::test]
    async fn background_chat_fails_over_past_a_used_up_quota() {
        let primary = Arc::new(ScriptedProvider::with_results([Err(
            aidaemon_core::ProviderError::QuotaExhausted {
                message: "usage limit reached".into(),
                resets_at: None,
                per_model: false,
            },
        )]));
        let failover = Arc::new(ScriptedProvider::new([text("consolidated")]));
        let runtime = SharedLlmRuntime::new_with_failovers(
            primary.clone(),
            None,
            ProviderKind::OpenaiCompatible,
            "primary-model".to_string(),
            vec![ProviderRuntimeTarget::new(
                failover.clone(),
                None,
                ProviderKind::Anthropic,
                "failover-model".to_string(),
            )],
        );

        let response = runtime
            .snapshot()
            .background_chat(
                "primary-model",
                &[serde_json::json!({"role": "user", "content": "hi"})],
                &[],
            )
            .await
            .expect("the failover provider answers");

        assert_eq!(response.content.as_deref(), Some("consolidated"));
        assert_eq!(primary.requests().len(), 1);
        let served = failover.requests();
        assert_eq!(served.len(), 1);
        assert_eq!(served[0].model, "failover-model");
    }

    fn router(default: &str, fallback: &[&str], fast: &str) -> Router {
        Router::new(ModelsConfig {
            default_model: default.to_string(),
            fallback_models: fallback.iter().map(|m| m.to_string()).collect(),
            primary: default.to_string(),
            fast: fast.to_string(),
            smart: String::new(),
        })
    }

    fn two_provider_runtime(
        primary: Arc<ScriptedProvider>,
        failover: Arc<ScriptedProvider>,
    ) -> SharedLlmRuntime {
        SharedLlmRuntime::new_with_failovers(
            primary,
            Some(router("strong", &["cheap"], "cheap")),
            ProviderKind::OpenaiChatgpt,
            "strong".to_string(),
            vec![ProviderRuntimeTarget::new(
                failover,
                Some(router("big-claude", &["small-claude"], "small-claude")),
                ProviderKind::Anthropic,
                "big-claude".to_string(),
            )],
        )
    }

    /// Auxiliary calls start on each provider's fast tier, and callers that
    /// pick candidates themselves see the failover provider's models too.
    #[test]
    fn auxiliary_models_are_cheap_first_and_include_failover_providers() {
        let runtime = two_provider_runtime(
            Arc::new(ScriptedProvider::new([])),
            Arc::new(ScriptedProvider::new([])),
        );
        let snapshot = runtime.snapshot();
        assert_eq!(snapshot.cheap_model(), "cheap");
        assert_eq!(
            snapshot.with_failover_models(vec!["cheap".into(), "strong".into()]),
            ["cheap", "strong", "small-claude", "big-claude"]
        );
    }

    /// A model-picking caller (the relationship router) reaches the failover provider
    /// by naming one of its models, and the background chain tries the
    /// failover provider's fast tier first.
    #[tokio::test]
    async fn requests_route_to_the_provider_that_owns_the_model() {
        use crate::providers::sdk_backed::ChatJson;
        let primary = Arc::new(ScriptedProvider::new([text("from primary")]));
        let failover = Arc::new(ScriptedProvider::new([
            text("from failover"),
            text("background"),
        ]));
        let runtime = two_provider_runtime(primary.clone(), failover.clone());
        let snapshot = runtime.snapshot();
        let backend = snapshot.model_routed_backend();
        let messages = [serde_json::json!({"role": "user", "content": "hi"})];

        let answer = backend.chat("small-claude", &messages, &[]).await.unwrap();
        assert_eq!(answer.content.as_deref(), Some("from failover"));
        let answer = backend.chat("strong", &messages, &[]).await.unwrap();
        assert_eq!(answer.content.as_deref(), Some("from primary"));

        // Primary exhausted (its script is empty): the chain moves to the
        // failover provider, starting at its fast tier.
        let answer = snapshot
            .background_chat("cheap", &messages, &[])
            .await
            .unwrap();
        assert_eq!(answer.content.as_deref(), Some("background"));
        assert_eq!(failover.requests().last().unwrap().model, "small-claude");
    }
}
