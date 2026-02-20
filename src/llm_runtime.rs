use std::sync::{Arc, RwLock};

use crate::config::{ModelsConfig, ProviderKind};
use crate::router::{Router, Tier};
use crate::traits::ModelProvider;

#[derive(Clone)]
pub struct LlmRuntimeSnapshot {
    provider: Arc<dyn ModelProvider>,
    router: Option<Router>,
    provider_kind: ProviderKind,
    primary_model: String,
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
}

#[derive(Clone)]
pub struct SharedLlmRuntime(Arc<RwLock<LlmRuntimeState>>);

impl SharedLlmRuntime {
    pub fn new(
        provider: Arc<dyn ModelProvider>,
        router: Option<Router>,
        provider_kind: ProviderKind,
        primary_model: String,
    ) -> Self {
        Self(Arc::new(RwLock::new(LlmRuntimeState {
            provider,
            router,
            provider_kind,
            primary_model,
        })))
    }

    pub fn snapshot(&self) -> LlmRuntimeSnapshot {
        let guard = self.0.read().expect("llm runtime lock poisoned");
        LlmRuntimeSnapshot {
            provider: guard.provider.clone(),
            router: guard.router.clone(),
            provider_kind: guard.provider_kind,
            primary_model: guard.primary_model.clone(),
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
    ) -> LlmRuntimeSnapshot {
        let mut guard = self.0.write().expect("llm runtime lock poisoned");
        let old = LlmRuntimeSnapshot {
            provider: guard.provider.clone(),
            router: guard.router.clone(),
            provider_kind: guard.provider_kind,
            primary_model: guard.primary_model.clone(),
        };
        guard.provider = provider;
        guard.router = router;
        guard.provider_kind = provider_kind;
        guard.primary_model = primary_model;
        old
    }
}

pub fn router_from_models(models: ModelsConfig) -> Option<Router> {
    let router = Router::new(models);
    if router.is_uniform() {
        None
    } else {
        Some(router)
    }
}
