use super::*;

// impl-Agent justification: provider reload from the daemon config, session clears.
impl Agent {
    /// Rebuild and hot-swap the provider backend + router without a daemon restart.
    pub async fn reload_provider(
        &self,
        config: &crate::config::AppConfig,
    ) -> anyhow::Result<String> {
        let bundle = crate::providers::factory::build_provider_router(config)?;
        let new_primary = bundle.primary_model.clone();
        let new_fallback = config
            .provider
            .models
            .fallback_models
            .iter()
            .find(|m| m.as_str() != new_primary)
            .cloned()
            .unwrap_or_else(|| new_primary.clone());
        let old_model = match tokio::time::timeout(Duration::from_secs(2), self.model.read()).await
        {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!("Timed out acquiring model lock during provider reload");
                self.llm_runtime.snapshot().primary_model()
            }
        };

        let old_runtime = self.llm_runtime.swap(
            bundle.provider,
            bundle.router,
            bundle.provider_kind,
            new_primary.clone(),
            bundle.failover_targets,
            bundle.models,
        );

        {
            let mut model = tokio::time::timeout(Duration::from_secs(2), self.model.write())
                .await
                .map_err(|_| anyhow::anyhow!("timed out acquiring model write lock"))?;
            let mut fallback =
                tokio::time::timeout(Duration::from_secs(2), self.fallback_model.write())
                    .await
                    .map_err(|_| anyhow::anyhow!("timed out acquiring fallback write lock"))?;
            *model = new_primary.clone();
            *fallback = new_fallback.clone();
        }
        let manual_model_cleared;
        {
            let mut override_guard =
                tokio::time::timeout(Duration::from_secs(2), self.model_override.write())
                    .await
                    .map_err(|_| anyhow::anyhow!("timed out acquiring model_override lock"))?;
            manual_model_cleared = *override_guard;
            *override_guard = false;
        }

        // The decision layer follows the reloaded config, so a Gemini key
        // set from chat (`/setkey`) takes effect without a restart.
        self.side_calls.set_decisions(
            crate::llm_runtime::decide::DecisionLayer::from_config_with_key(
                &config.decisions,
                config.gemini_provider_api_key().as_deref(),
            ),
        );

        info!(
            old_provider = ?old_runtime.provider_kind(),
            new_provider = ?config.provider.kind,
            old_model = %old_model,
            new_model = %new_primary,
            new_fallback = %new_fallback,
            "Provider runtime reloaded"
        );

        Ok(aidaemon_providers::routing::ProviderReload {
            old_provider: old_runtime.provider_kind(),
            new_provider: config.provider.kind,
            old_model,
            new_model: new_primary,
            manual_model_cleared,
        }
        .summary())
    }

    /// HARD-clear a session (destructive: deletes events). Backs `/wipe`.
    pub async fn clear_session(&self, session_id: &str) -> anyhow::Result<()> {
        self.state.clear_session(session_id).await
    }

    /// Reset a session's conversation context non-destructively (events remain
    /// for memory + audit). Backs `/clear`. Also drops the in-memory turn
    /// anchor so the next context build re-initializes past the new boundary.
    pub async fn clear_session_context(&self, session_id: &str) -> anyhow::Result<()> {
        self.state.clear_session_context(session_id).await?;
        self.runtime.turn_anchors.write().await.remove(session_id);
        Ok(())
    }

    /// List available models from the provider.
    pub async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        self.llm_runtime.snapshot().models().list().await
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
