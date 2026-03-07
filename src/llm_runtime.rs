use std::sync::{Arc, RwLock};

use crate::config::{ModelsConfig, ProviderKind};
use crate::router::{Router, Tier};
use crate::traits::ModelProvider;

#[derive(Clone)]
pub struct ProviderRuntimeTarget {
    provider: Arc<dyn ModelProvider>,
    router: Option<Router>,
    provider_kind: ProviderKind,
    primary_model: String,
}

impl ProviderRuntimeTarget {
    pub fn new(
        provider: Arc<dyn ModelProvider>,
        router: Option<Router>,
        provider_kind: ProviderKind,
        primary_model: String,
    ) -> Self {
        Self {
            provider,
            router,
            provider_kind,
            primary_model,
        }
    }

    pub fn provider(&self) -> Arc<dyn ModelProvider> {
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
    provider: Arc<dyn ModelProvider>,
    router: Option<Router>,
    provider_kind: ProviderKind,
    primary_model: String,
    failover_targets: Vec<ProviderRuntimeTarget>,
}

impl LlmRuntimeSnapshot {
    pub fn provider(&self) -> Arc<dyn ModelProvider> {
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

    pub fn fast_model(&self) -> String {
        self.router
            .as_ref()
            .map(|r| r.select(Tier::Fast).to_string())
            .unwrap_or_else(|| self.primary_model.clone())
    }
}

#[derive(Clone)]
struct LlmRuntimeState {
    provider: Arc<dyn ModelProvider>,
    router: Option<Router>,
    provider_kind: ProviderKind,
    primary_model: String,
    failover_targets: Vec<ProviderRuntimeTarget>,
}

#[derive(Clone)]
pub struct SharedLlmRuntime(Arc<RwLock<LlmRuntimeState>>);

impl SharedLlmRuntime {
    #[allow(dead_code)] // Retained for single-provider call sites and tests.
    pub fn new(
        provider: Arc<dyn ModelProvider>,
        router: Option<Router>,
        provider_kind: ProviderKind,
        primary_model: String,
    ) -> Self {
        Self::new_with_failovers(provider, router, provider_kind, primary_model, Vec::new())
    }

    pub fn new_with_failovers(
        provider: Arc<dyn ModelProvider>,
        router: Option<Router>,
        provider_kind: ProviderKind,
        primary_model: String,
        failover_targets: Vec<ProviderRuntimeTarget>,
    ) -> Self {
        Self(Arc::new(RwLock::new(LlmRuntimeState {
            provider,
            router,
            provider_kind,
            primary_model,
            failover_targets,
        })))
    }

    pub fn snapshot(&self) -> LlmRuntimeSnapshot {
        let guard = self.0.read().expect("llm runtime lock poisoned");
        LlmRuntimeSnapshot {
            provider: guard.provider.clone(),
            router: guard.router.clone(),
            provider_kind: guard.provider_kind,
            primary_model: guard.primary_model.clone(),
            failover_targets: guard.failover_targets.clone(),
        }
    }

    pub fn provider(&self) -> Arc<dyn ModelProvider> {
        self.snapshot().provider()
    }

    pub fn router(&self) -> Option<Router> {
        self.snapshot().router()
    }

    pub fn swap(
        &self,
        provider: Arc<dyn ModelProvider>,
        router: Option<Router>,
        provider_kind: ProviderKind,
        primary_model: String,
        failover_targets: Vec<ProviderRuntimeTarget>,
    ) -> LlmRuntimeSnapshot {
        let mut guard = self.0.write().expect("llm runtime lock poisoned");
        let old = LlmRuntimeSnapshot {
            provider: guard.provider.clone(),
            router: guard.router.clone(),
            provider_kind: guard.provider_kind,
            primary_model: guard.primary_model.clone(),
            failover_targets: guard.failover_targets.clone(),
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
