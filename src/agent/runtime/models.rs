use super::*;

impl Agent {
    /// Get the current model name.
    pub async fn current_model(&self) -> String {
        self.model.read().await.clone()
    }

    /// Switch the active model at runtime. Keeps the old model as fallback.
    /// Also disables auto-routing until `clear_model_override()` is called.
    pub async fn set_model(&self, model: String) {
        let mut m = self.model.write().await;
        let mut fb = self.fallback_model.write().await;
        info!(old = %*m, new = %model, "Model switched");
        *fb = m.clone();
        *m = model;
        *self.model_override.write().await = true;
    }

    /// Re-enable auto-routing after a manual model override.
    pub async fn clear_model_override(&self) {
        *self.model_override.write().await = false;
        info!("Model override cleared, auto-routing re-enabled");
    }

    /// Rebuild and hot-swap the provider backend + router without a daemon restart.
    pub async fn reload_provider(
        &self,
        config: &crate::config::AppConfig,
    ) -> anyhow::Result<String> {
        let bundle = crate::startup::provider_router::build_provider_router(config)?;
        let new_router = crate::llm_runtime::router_from_models(config.provider.models.clone());
        let new_kind = config.provider.kind.clone();
        let new_primary = bundle.primary_model.clone();
        let new_fallback = config
            .provider
            .models
            .fallback_models
            .iter()
            .find(|m| m.as_str() != new_primary)
            .cloned()
            .unwrap_or_else(|| new_primary.clone());
        let old_model = self.model.read().await.clone();

        let old_runtime = self.llm_runtime.swap(
            bundle.provider,
            new_router,
            new_kind.clone(),
            new_primary.clone(),
        );

        {
            let mut model = self.model.write().await;
            let mut fallback = self.fallback_model.write().await;
            *model = new_primary.clone();
            *fallback = new_fallback.clone();
        }
        *self.model_override.write().await = false;

        info!(
            old_provider = ?old_runtime.provider_kind(),
            new_provider = ?new_kind,
            old_model = %old_model,
            new_model = %new_primary,
            new_fallback = %new_fallback,
            "Provider runtime reloaded"
        );

        Ok(format!(
            "Provider: {:?} -> {:?}. Model: {} -> {}. Auto-routing re-enabled.",
            old_runtime.provider_kind(),
            new_kind,
            old_model,
            new_primary
        ))
    }

    /// Clear conversation history for a session, preserving facts.
    pub async fn clear_session(&self, session_id: &str) -> anyhow::Result<()> {
        self.state.clear_session(session_id).await
    }

    /// List available models from the provider.
    pub async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        self.llm_runtime.provider().list_models().await
    }

    /// Stamp the current config as "last known good" — called after a
    /// successful LLM response proves the config actually works.
    pub(super) async fn stamp_lastgood(&self) {
        let lastgood = self.config_path.with_extension("toml.lastgood");
        if let Err(e) = tokio::fs::copy(&self.config_path, &lastgood).await {
            warn!(error = %e, "Failed to stamp lastgood config");
        }
    }
}
