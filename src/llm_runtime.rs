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
            for model in target.all_models_ordered() {
                steps.push(FailoverStep::from_arc(target.provider()).model(model));
            }
        }
        aidaemon_core::Failover::new(steps)
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

    pub fn provider(&self) -> ModelBackend {
        self.snapshot().provider()
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
}
