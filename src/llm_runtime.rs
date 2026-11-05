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
