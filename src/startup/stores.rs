use std::sync::Arc;

use tracing::info;

use crate::config::AppConfig;
use crate::events::EventStore;
use crate::health::HealthProbeStore;
use crate::memory::embeddings::EmbeddingService;
use crate::plans::PlanStore;
use crate::state::SqliteStateStore;

pub struct StoreBundle {
    pub embedding_service: Arc<EmbeddingService>,
    pub state: Arc<SqliteStateStore>,
    pub event_store: Arc<EventStore>,
    pub plan_store: Arc<PlanStore>,
    pub health_store: Option<Arc<HealthProbeStore>>,
}

pub async fn build_stores(config: &AppConfig) -> anyhow::Result<StoreBundle> {
    let embedding_service = Arc::new(
        EmbeddingService::new().map_err(|e| anyhow::anyhow!("Failed to init embeddings: {}", e))?,
    );
    info!("Embedding service initialized (AllMiniLML6V2)");

    let state = Arc::new(
        SqliteStateStore::new(
            &config.state.db_path,
            config.state.working_memory_cap,
            config.state.encryption_key.as_deref(),
            embedding_service.clone(),
        )
        .await?,
    );
    info!("State store initialized ({})", config.state.db_path);

    if let Ok(count) = state.backfill_episode_embeddings().await {
        if count > 0 {
            info!(count, "Backfilled missing episode embeddings");
        }
    }

    if let Ok(count) = state.backfill_fact_embeddings().await {
        if count > 0 {
            info!(count, "Backfilled missing fact embeddings");
        }
    }

    let event_store = Arc::new(EventStore::new(state.pool()).await?);
    info!("Event store initialized");

    let plan_store = Arc::new(PlanStore::new(state.pool()).await?);
    info!("Plan store initialized");

    let health_store = if config.health.enabled {
        Some(Arc::new(
            HealthProbeStore::new(state.pool())
                .await
                .expect("Failed to initialize health probe store"),
        ))
    } else {
        None
    };

    info!("Plan store and event store initialized");

    Ok(StoreBundle {
        embedding_service,
        state,
        event_store,
        plan_store,
        health_store,
    })
}
