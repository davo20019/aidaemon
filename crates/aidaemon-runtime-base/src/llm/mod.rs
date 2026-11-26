use std::sync::{Arc, RwLock};

use crate::chat_bridge::sdk_backed::ModelList;
use aidaemon_providers::routing::{ModelsConfig, ProviderKind, Router};

pub mod decide;
pub mod decisions;
pub mod gateway;
pub mod side_calls;
pub mod trial;

/// A model backend: any SDK provider.
pub type ModelBackend = Arc<dyn aidaemon_core::Provider>;

#[derive(Clone)]
pub struct ProviderRuntimeTarget {
    provider: ModelBackend,
    router: Option<Router>,
    provider_kind: ProviderKind,
    primary_model: String,
    models: ModelList,
    offered_models: Vec<String>,
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
            offered_models: Vec::new(),
        }
    }

    /// How this provider's models are listed.
    pub fn with_models(mut self, models: ModelList) -> Self {
        self.models = models;
        self
    }

    pub fn models(&self) -> &ModelList {
        &self.models
    }

    /// Catalog models this configured provider also accepts.
    pub fn with_offered_models(mut self, models: Vec<String>) -> Self {
        self.offered_models = models;
        self
    }

    pub fn offered_models(&self) -> &[String] {
        &self.offered_models
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
    pub fn models_cheap_first(&self) -> Vec<String> {
        cheap_first(self.router.as_ref(), self.all_models_ordered())
    }

    /// The models in the order a turn tries them when it fails over here:
    /// work that asked for the main provider's fast model starts on this
    /// provider's fast one; any other starts on its strongest, so a coding
    /// turn does not fall from the strongest model to the cheapest.
    pub fn models_for_failover(&self, wants_fast: bool) -> Vec<String> {
        if wants_fast {
            return self.models_cheap_first();
        }
        let mut models = self.all_models_ordered();
        if let Some(smart) = self.router.as_ref().map(|router| {
            router
                .select(aidaemon_providers::routing::Tier::Smart)
                .to_string()
        }) && !smart.trim().is_empty()
        {
            models.retain(|model| *model != smart);
            models.insert(0, smart);
        }
        models
    }
}

/// `models` reordered so the router's fast tier, when it is one of them,
/// comes first.
fn cheap_first(router: Option<&Router>, mut models: Vec<String>) -> Vec<String> {
    if let Some(fast) = router.map(|router| {
        router
            .select(aidaemon_providers::routing::Tier::Fast)
            .to_string()
    }) && let Some(index) = models.iter().position(|model| *model == fast)
    {
        let fast = models.remove(index);
        models.insert(0, fast);
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
    offered_models: Vec<String>,
}

impl LlmRuntimeSnapshot {
    /// The primary provider's model listing.
    pub fn models(&self) -> &ModelList {
        &self.models
    }

    pub fn offered_models(&self) -> &[String] {
        &self.offered_models
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

    /// The backend of the provider the picker names `provider_id`.
    fn backend_of(&self, provider_id: &str) -> Option<ModelBackend> {
        if provider_id == MAIN_PROVIDER {
            return Some(self.provider.clone());
        }
        // The picker's own id format ([`failover_id`]).
        let index: usize = provider_id.strip_prefix("failover-")?.parse().ok()?;
        self.failover_targets
            .get(index)
            .map(ProviderRuntimeTarget::provider)
    }

    /// Until when `model` of the provider `provider_id` will not answer (a
    /// used-up quota, an account that cannot pay, a rejected key, as its
    /// quota guard learned from a call); `None`: it may answer now.
    pub fn unavailable_until(
        &self,
        provider_id: &str,
        model: &str,
    ) -> Option<std::time::SystemTime> {
        self.backend_of(provider_id)?.unavailable_until(model)
    }

    /// When some configured model (the primary provider's, then every
    /// failover provider's) can next answer: `None` when one may answer
    /// now, otherwise the earliest time a known cooldown ends.
    pub fn next_answer_at(&self) -> Option<std::time::SystemTime> {
        let primary_models = self
            .router
            .as_ref()
            .map(|router| router.all_models_ordered())
            .filter(|models| !models.is_empty())
            .unwrap_or_else(|| vec![self.primary_model.clone()]);
        let candidates = std::iter::once((self.provider.clone(), primary_models)).chain(
            self.failover_targets
                .iter()
                .map(|target| (target.provider(), target.all_models_ordered())),
        );
        let mut earliest: Option<std::time::SystemTime> = None;
        for (provider, models) in candidates {
            for model in models {
                let until = provider.unavailable_until(&model)?;
                earliest = Some(earliest.map_or(until, |at| at.min(until)));
            }
        }
        earliest
    }

    /// The chain a background side call runs on: `first_model`, the primary
    /// provider's other models (the router's fallbacks), then every failover
    /// provider's models, under the SDK's default policy, which moves on after
    /// a used-up quota or an auth failure. Every step's provider is passed
    /// through `wrap` (the side-call gateway wraps them to record each
    /// attempt). Agent turns build their own chain with the daemon's
    /// recovery policy.
    pub(in crate::llm) fn background_chain_through(
        &self,
        first_model: &str,
        wrap: impl Fn(ModelBackend) -> ModelBackend,
    ) -> aidaemon_core::Failover {
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
            .map(|model| FailoverStep::from_arc(wrap(self.provider.clone())).model(model))
            .collect();
        for target in &self.failover_targets {
            for model in target.models_cheap_first() {
                steps.push(FailoverStep::from_arc(wrap(target.provider())).model(model));
            }
        }
        aidaemon_core::Failover::new(steps)
    }

    /// The primary provider's fast-tier model, for background and auxiliary
    /// calls (memory consolidation, summaries, assessment) that do not need
    /// the strongest model.
    pub fn cheap_model(&self) -> String {
        self.router
            .as_ref()
            .map(|router| {
                router
                    .select(aidaemon_providers::routing::Tier::Fast)
                    .to_string()
            })
            .filter(|model| !model.trim().is_empty())
            .unwrap_or_else(|| self.primary_model.clone())
    }

    /// `primary_models` followed by every failover provider's models (fast
    /// tier first), for callers that try candidates one by one themselves.
    /// Run them on [`Self::model_routed_backend`].
    pub(in crate::llm) fn with_failover_models(
        &self,
        mut primary_models: Vec<String>,
    ) -> Vec<String> {
        for target in &self.failover_targets {
            for model in target.models_cheap_first() {
                if !primary_models.contains(&model) {
                    primary_models.push(model);
                }
            }
        }
        primary_models
    }

    /// Ask for a safety judgment within one time budget, starting on the
    /// primary model and moving to the failover providers only on an
    /// infrastructure failure. A failure tied to this request stops, so the
    /// content being judged cannot pick a different judge. One attempt per
    /// model; the caller fails closed on an error, which is the last
    /// attempt's. The backend is passed through `wrap` (the side-call gateway
    /// wraps it to record each attempt as it starts).
    pub(in crate::llm) async fn ask_for_judgment_through(
        &self,
        messages: &[serde_json::Value],
        options: &aidaemon_events::provider::ChatOptions,
        budget: std::time::Duration,
        wrap: impl FnOnce(ModelBackend) -> ModelBackend,
    ) -> Result<aidaemon_events::provider::ProviderResponse, String> {
        use crate::chat_bridge::ChatJson;
        let routed = self.model_routed_backend();
        let models = available_first(
            &routed,
            self.with_failover_models(vec![self.primary_model()]),
        );
        let backend = wrap(routed);
        let deadline = std::time::Instant::now() + budget;
        let mut last_error = "no model available".to_string();
        for model in models {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let attempt = tokio::time::timeout(
                remaining,
                backend.chat_with_options(&model, messages, &[], options),
            )
            .await;
            let (error, try_next) = match attempt {
                Ok(Ok(response)) => return Ok(response),
                Ok(Err(error)) => {
                    let infrastructure = error
                        .downcast_ref::<crate::chat_bridge::ProviderError>()
                        .is_some_and(|error| error.kind.is_infrastructure_failure());
                    (error.to_string(), infrastructure)
                }
                Err(_) => (format!("timed out after {}s", budget.as_secs()), false),
            };
            tracing::warn!(%model, %error, "Safety judgment attempt failed");
            last_error = error;
            if !try_next {
                break;
            }
        }
        Err(last_error)
    }

    /// One backend that sends each request to the provider that owns its
    /// model: the primary provider, else the first failover provider listing
    /// it. Unknown models go to the primary.
    pub(in crate::llm) fn model_routed_backend(&self) -> ModelBackend {
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
}

/// `models` without those their provider already knows cannot answer now
/// (a used-up quota, a rejected key: [`aidaemon_core::QuotaGuard`]), so a
/// judgment does not pay for a known refusal on every call. When every model
/// is known to be out, all are kept: the attempts then fail at once and the
/// last error says why.
fn available_first(backend: &ModelBackend, models: Vec<String>) -> Vec<String> {
    let now = std::time::SystemTime::now();
    let (out, available): (Vec<String>, Vec<String>) = models.into_iter().partition(|model| {
        backend
            .unavailable_until(model)
            .is_some_and(|until| until > now)
    });
    if available.is_empty() {
        return out;
    }
    if !out.is_empty() {
        tracing::debug!(skipped = ?out, "Safety judgment skips models known to be out");
    }
    available
}

#[derive(Clone)]
struct LlmRuntimeState {
    provider: ModelBackend,
    router: Option<Router>,
    provider_kind: ProviderKind,
    primary_model: String,
    failover_targets: Vec<ProviderRuntimeTarget>,
    models: ModelList,
    offered_models: Vec<String>,
    /// When each provider (by its backend, [`backend_key`]) was last asked
    /// for its models; a swap forgets them all.
    offered_listed_at: std::collections::HashMap<usize, std::time::Instant>,
    /// Models a trial showed can run a turn, by backend ([`backend_key`]);
    /// a swap forgets them.
    tried: std::collections::HashSet<(usize, String)>,
    /// Providers whose account the last refresh's trial was refused (no
    /// credit, a rejected key, a used-up quota), by backend, until when the
    /// refusal is trusted without a newer trial; a swap forgets them.
    account_refused: std::collections::HashMap<usize, std::time::Instant>,
}

/// A backend's identity while the runtime holds it.
fn backend_key(backend: &ModelBackend) -> usize {
    Arc::as_ptr(backend) as *const () as usize
}

/// The picker's id of the runtime's main provider; a failover is
/// `failover-N` ([`failover_id`]).
pub const MAIN_PROVIDER: &str = "main";

/// The picker's id of the `index`th failover provider.
pub fn failover_id(index: usize) -> String {
    format!("failover-{index}")
}

#[derive(Clone)]
pub struct SharedLlmRuntime(Arc<RwLock<LlmRuntimeState>>);

impl SharedLlmRuntime {
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
            offered_models: Vec::new(),
            offered_listed_at: std::collections::HashMap::new(),
            tried: std::collections::HashSet::new(),
            account_refused: std::collections::HashMap::new(),
        })))
    }

    /// Whether `model` of the provider `provider_id` will not answer for
    /// now: its quota guard holds a cooldown for it, or the last trial of
    /// the provider's account was refused.
    pub fn unavailable(&self, provider_id: &str, model: &str) -> bool {
        let snapshot = self.snapshot();
        let Some(backend) = snapshot.backend_of(provider_id) else {
            return false;
        };
        backend.unavailable_until(model).is_some()
            || self
                .0
                .read()
                .expect("llm runtime lock poisoned")
                .account_refused
                .get(&backend_key(&backend))
                .is_some_and(|until| *until > std::time::Instant::now())
    }

    /// One real call showing `model` of the provider `provider_id` can run
    /// a turn ([`trial::trial`]; `Err`: why not, in plain words), once: a model that passed is not tried again until
    /// the providers change.
    pub async fn trial_once(&self, provider_id: &str, model: &str) -> Result<(), String> {
        let snapshot = self.snapshot();
        let backend = snapshot
            .backend_of(provider_id)
            .ok_or_else(|| format!("There is no provider {provider_id}"))?;
        let key = (backend_key(&backend), model.to_string());
        if self
            .0
            .read()
            .expect("llm runtime lock poisoned")
            .tried
            .contains(&key)
        {
            return Ok(());
        }
        trial::trial(&*backend, model)
            .await
            .map_err(|failure| failure.to_string())?;
        let mut guard = self.0.write().expect("llm runtime lock poisoned");
        // Kept only while the runtime still holds that backend.
        if Arc::ptr_eq(&guard.provider, &backend)
            || guard
                .failover_targets
                .iter()
                .any(|target| Arc::ptr_eq(&target.provider, &backend))
        {
            guard.tried.insert(key);
        }
        Ok(())
    }

    /// Whether some provider has not been asked for its models yet.
    pub fn offered_never_listed(&self) -> bool {
        let guard = self.0.read().expect("llm runtime lock poisoned");
        std::iter::once(&guard.provider)
            .chain(guard.failover_targets.iter().map(|target| &target.provider))
            .any(|backend| !guard.offered_listed_at.contains_key(&backend_key(backend)))
    }

    /// Ask each provider not asked within `max_age` for the models it
    /// offers now, and offer those (in place of the catalog's), and make one
    /// trial call on its cheapest configured model, so an account that
    /// cannot answer (no credit, a rejected key, a used-up quota) is known
    /// to its quota guard before anyone picks it. One that cannot list keeps
    /// what it offered; each is asked again after `max_age`, by one caller
    /// at a time.
    pub async fn refresh_offered_models(&self, max_age: std::time::Duration) {
        let due: Vec<(ModelBackend, ModelList, String)> = {
            let mut guard = self.0.write().expect("llm runtime lock poisoned");
            let now = std::time::Instant::now();
            let main_cheapest = cheap_first(
                guard.router.as_ref(),
                guard
                    .router
                    .as_ref()
                    .map(Router::all_models_ordered)
                    .filter(|models| !models.is_empty())
                    .unwrap_or_else(|| vec![guard.primary_model.clone()]),
            )
            .into_iter()
            .next()
            .unwrap_or_default();
            let mut providers = vec![(guard.provider.clone(), guard.models.clone(), main_cheapest)];
            providers.extend(guard.failover_targets.iter().map(|target| {
                (
                    target.provider.clone(),
                    target.models.clone(),
                    target
                        .models_cheap_first()
                        .into_iter()
                        .next()
                        .unwrap_or_default(),
                )
            }));
            providers
                .into_iter()
                .filter(|(backend, _, _)| {
                    let key = backend_key(backend);
                    let due = guard
                        .offered_listed_at
                        .get(&key)
                        .is_none_or(|at| now.saturating_duration_since(*at) >= max_age);
                    if due {
                        guard.offered_listed_at.insert(key, now);
                    }
                    due
                })
                .collect()
        };
        let asks = due.into_iter().map(|(backend, listing, cheapest)| async move {
            match listing.live().await {
                Ok(models) => {
                    self.set_offered_models(&backend, models);
                }
                Err(error) => {
                    tracing::debug!(provider = backend.name(), %error, "A provider could not list its models");
                }
            }
            if cheapest.is_empty() || backend.unavailable_until(&cheapest).is_some() {
                return;
            }
            let refused = match trial::trial(&*backend, &cheapest).await {
                Ok(()) => false,
                Err(failure) => {
                    tracing::info!(provider = backend.name(), model = %cheapest, %failure, "A provider's trial call failed");
                    failure.account_refused()
                }
            };
            let mut guard = self.0.write().expect("llm runtime lock poisoned");
            let key = backend_key(&backend);
            if refused {
                // Trusted until the next refresh can try it again.
                guard
                    .account_refused
                    .insert(key, std::time::Instant::now() + 2 * max_age);
            } else {
                guard.account_refused.remove(&key);
            }
        });
        futures::future::join_all(asks).await;
    }

    /// How the primary provider's models are listed.
    pub fn with_models(self, models: ModelList) -> Self {
        self.0.write().expect("llm runtime lock poisoned").models = models;
        self
    }

    /// The models the provider whose backend is `backend` (the primary or
    /// a failover) offers beyond its configured ones: what it listed. `false`
    /// when no provider of the runtime has that backend now (it was swapped
    /// since the listing was asked for).
    pub fn set_offered_models(&self, backend: &ModelBackend, models: Vec<String>) -> bool {
        let mut guard = self.0.write().expect("llm runtime lock poisoned");
        if Arc::ptr_eq(&guard.provider, backend) {
            guard.offered_models = models;
            return true;
        }
        match guard
            .failover_targets
            .iter_mut()
            .find(|target| Arc::ptr_eq(&target.provider, backend))
        {
            Some(target) => {
                target.offered_models = models;
                true
            }
            None => false,
        }
    }

    /// Catalog models the configured primary provider also accepts.
    pub fn with_offered_models(self, models: Vec<String>) -> Self {
        self.0
            .write()
            .expect("llm runtime lock poisoned")
            .offered_models = models;
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
            offered_models: guard.offered_models.clone(),
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
        model_lists: (ModelList, Vec<String>),
    ) -> LlmRuntimeSnapshot {
        let (models, offered_models) = model_lists;
        let mut guard = self.0.write().expect("llm runtime lock poisoned");
        let old = LlmRuntimeSnapshot {
            provider: guard.provider.clone(),
            router: guard.router.clone(),
            provider_kind: guard.provider_kind,
            primary_model: guard.primary_model.clone(),
            failover_targets: guard.failover_targets.clone(),
            models: std::mem::replace(&mut guard.models, models),
            offered_models: std::mem::replace(&mut guard.offered_models, offered_models),
        };
        guard.provider = provider;
        guard.router = router;
        guard.provider_kind = provider_kind;
        guard.primary_model = primary_model;
        guard.failover_targets = failover_targets;
        guard.offered_listed_at.clear();
        guard.tried.clear();
        guard.account_refused.clear();
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

    fn unavailable_until(&self, model: &str) -> Option<std::time::SystemTime> {
        self.backend_for(model).unavailable_until(model)
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
    use crate::chat_bridge::ChatJson;
    use aidaemon_core::testing::{ScriptedProvider, text};

    /// One call on the background chain, starting at `model`.
    async fn background_chat(
        snapshot: &LlmRuntimeSnapshot,
        model: &str,
        messages: &[serde_json::Value],
    ) -> anyhow::Result<aidaemon_events::provider::ProviderResponse> {
        snapshot
            .background_chain_through(model, |provider| provider)
            .chat_with_options(model, messages, &[], &Default::default())
            .await
    }

    /// A turn failing over to a provider configured with only its cheap
    /// model starts on that provider's strongest one; work that asked for
    /// the fast model starts on the fast one.
    #[test]
    fn a_failover_starts_on_the_tier_the_turn_asked_for() {
        let mut models = ModelsConfig {
            default_model: "claude-haiku-5-5".to_string(),
            ..ModelsConfig::default()
        };
        models.apply_defaults(&ProviderKind::Anthropic);
        let router = Router::new(models);
        let target = ProviderRuntimeTarget::new(
            Arc::new(ScriptedProvider::new([])),
            (!router.is_uniform()).then_some(router),
            ProviderKind::Anthropic,
            "claude-haiku-5-5".to_string(),
        );
        assert_eq!(
            target.models_for_failover(false),
            vec!["claude-opus-5-5", "claude-haiku-5-5"]
        );
        assert_eq!(target.models_for_failover(true)[0], "claude-haiku-5-5");
    }

    /// A background call whose primary provider has used up its quota is
    /// answered by the failover provider, not dropped.
    #[tokio::test]
    async fn background_chat_fails_over_past_a_used_up_quota() {
        let primary = Arc::new(ScriptedProvider::with_results([Err(
            aidaemon_core::ProviderError::QuotaExhausted {
                message: "usage limit reached".into(),
                resets_at: None,
                per_model: false,
                in_stream: false,
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

        let response = background_chat(
            &runtime.snapshot(),
            "primary-model",
            &[serde_json::json!({"role": "user", "content": "hi"})],
        )
        .await
        .expect("the failover provider answers");

        assert_eq!(response.content.as_deref(), Some("consolidated"));
        assert_eq!(primary.requests().len(), 1);
        let served = failover.requests();
        assert_eq!(served.len(), 1);
        assert_eq!(served[0].model, "failover-model");
    }

    /// A used-up quota on the judge's primary model is an infrastructure
    /// failure: the judgment moves on to the failover provider and is
    /// answered there. Once the primary is known to be out, later
    /// judgments skip it instead of paying for its refusal each time; a
    /// failure tied to the request still stops at the first model.
    #[tokio::test]
    async fn a_judgment_fails_over_past_a_used_up_quota_and_then_skips_it() {
        let primary = Arc::new(ScriptedProvider::with_results([Err(
            aidaemon_core::ProviderError::QuotaExhausted {
                message: "usage limit reached".into(),
                resets_at: None,
                per_model: false,
                in_stream: false,
            },
        )]));
        let guarded: ModelBackend = Arc::new(aidaemon_core::QuotaGuard::from_arc(primary.clone()));
        let verdict = r#"{"consistent":true,"reason":"asked for"}"#;
        let failover = Arc::new(ScriptedProvider::new([text(verdict), text(verdict)]));
        let runtime = SharedLlmRuntime::new_with_failovers(
            guarded,
            None,
            ProviderKind::OpenaiChatgpt,
            "primary-model".to_string(),
            vec![ProviderRuntimeTarget::new(
                failover.clone(),
                None,
                ProviderKind::Anthropic,
                "failover-model".to_string(),
            )],
        );
        let messages = [serde_json::json!({"role": "user", "content": "judge this"})];
        for _ in 0..2 {
            let answer = runtime
                .snapshot()
                .ask_for_judgment_through(
                    &messages,
                    &Default::default(),
                    std::time::Duration::from_secs(5),
                    |backend| backend,
                )
                .await
                .expect("the failover provider judges");
            assert_eq!(answer.content.as_deref(), Some(verdict));
        }
        assert_eq!(
            primary.requests().len(),
            1,
            "the out-of-quota model is asked once"
        );
        assert_eq!(failover.requests().len(), 2);

        // A request the provider rejected as malformed stops at once.
        let rejecting = Arc::new(ScriptedProvider::with_results([Err(
            aidaemon_core::ProviderError::InvalidRequest("schema rejected".into()),
        )]));
        let unused = Arc::new(ScriptedProvider::new([text(verdict)]));
        let runtime = SharedLlmRuntime::new_with_failovers(
            rejecting,
            None,
            ProviderKind::OpenaiChatgpt,
            "primary-model".to_string(),
            vec![ProviderRuntimeTarget::new(
                unused.clone(),
                None,
                ProviderKind::Anthropic,
                "failover-model".to_string(),
            )],
        );
        let refused = runtime
            .snapshot()
            .ask_for_judgment_through(
                &messages,
                &Default::default(),
                std::time::Duration::from_secs(5),
                |backend| backend,
            )
            .await;
        assert!(refused.is_err());
        assert!(unused.requests().is_empty());
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
        let primary = Arc::new(ScriptedProvider::new([text("from primary")]));
        let failover = Arc::new(ScriptedProvider::new([
            text("from failover"),
            text("background"),
        ]));
        let runtime = two_provider_runtime(primary.clone(), failover.clone());
        let snapshot = runtime.snapshot();
        let backend = snapshot.model_routed_backend();
        let messages = [serde_json::json!({"role": "user", "content": "hi"})];

        let answer = backend
            .chat_with_options("small-claude", &messages, &[], &Default::default())
            .await
            .unwrap();
        assert_eq!(answer.content.as_deref(), Some("from failover"));
        let answer = backend
            .chat_with_options("strong", &messages, &[], &Default::default())
            .await
            .unwrap();
        assert_eq!(answer.content.as_deref(), Some("from primary"));

        // Primary exhausted (its script is empty): the chain moves to the
        // failover provider, starting at its fast tier.
        let answer = background_chat(&snapshot, "cheap", &messages)
            .await
            .unwrap();
        assert_eq!(answer.content.as_deref(), Some("background"));
        assert_eq!(failover.requests().last().unwrap().model, "small-claude");
    }
}
