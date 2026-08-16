use std::sync::Arc;

use tracing::info;

use crate::config::AppConfig;
use crate::events::EventStore;
use crate::health::HealthProbeStore;
use crate::memory::embeddings::EmbeddingService;
use crate::plans::PlanStore;
use crate::state::SqliteStateStore;
use crate::traits::StateStore;
use sqlx::SqlitePool;

pub struct StoreBundle {
    /// Domain-facing state contract. The concrete SQLite implementation stays
    /// inside this adapter module and is not leaked through startup wiring.
    pub state: Arc<dyn StateStore>,
    /// Shared relational handle used by infrastructure components that need a
    /// pool directly (events, plans, health, nodes, and checkpoints).
    pub pool: SqlitePool,
    pub embedding_service: Arc<EmbeddingService>,
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

    // Canonical memory tables are repairable projections, not authoritative
    // state. A large legacy event history can take minutes to project, so keep
    // that migration off the channel-readiness critical path.
    let projection_state = state.clone();
    tokio::spawn(async move {
        match crate::state::sqlite::history_search::repair_and_backfill(
            &projection_state.pool(),
            20,
        )
        .await
        {
            Ok(stats)
                if stats.projected > 0
                    || stats.orphans_removed > 0
                    || stats.fts_rebuilt
                    || stats.episodes_repaired > 0 =>
            {
                tracing::info!(
                    projected = stats.projected,
                    orphans_removed = stats.orphans_removed,
                    fts_rebuilt = stats.fts_rebuilt,
                    episodes_repaired = stats.episodes_repaired,
                    pending = stats.pending,
                    "Repaired exact-history projection"
                );
            }
            Ok(_) => {}
            Err(error) => tracing::warn!(%error, "Exact-history projection repair deferred"),
        }
        // Continue large legacy backfills incrementally off the readiness
        // path. Each round is bounded and yields between batches.
        for _ in 0..20 {
            match crate::state::sqlite::history_search::backfill(&projection_state.pool(), 20).await
            {
                Ok((_, 0)) => break,
                Ok((projected, pending)) => {
                    tracing::info!(projected, pending, "Continuing exact-history backfill");
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
                Err(error) => {
                    tracing::warn!(%error, "Exact-history incremental backfill deferred");
                    break;
                }
            }
        }
        match projection_state.backfill_missing_memory_projections().await {
            Ok((facts, episodes, spans, procedures, error_solutions))
                if facts > 0
                    || episodes > 0
                    || spans > 0
                    || procedures > 0
                    || error_solutions > 0 =>
            {
                tracing::info!(
                    facts,
                    episodes,
                    spans,
                    procedures,
                    error_solutions,
                    "Backfilled canonical memory projections"
                );
            }
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(%error, "Canonical memory backfill deferred");
            }
        }
        const STRUCTURED_BACKFILL_ATTEMPTS: usize = 3;
        for attempt in 1..=STRUCTURED_BACKFILL_ATTEMPTS {
            match projection_state.backfill_structured_personal_memory().await {
                Ok(count) => {
                    if count > 0 {
                        tracing::info!(count, "Backfilled safe structured personal memories");
                    }
                    break;
                }
                Err(error) if attempt < STRUCTURED_BACKFILL_ATTEMPTS => {
                    tracing::info!(
                        %error,
                        attempt,
                        "Structured personal-memory backfill busy; retrying"
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(250 * attempt as u64))
                        .await;
                }
                Err(error) => {
                    tracing::warn!(
                        %error,
                        attempts = STRUCTURED_BACKFILL_ATTEMPTS,
                        "Structured personal-memory backfill deferred"
                    );
                }
            }
        }
    });

    Ok(StoreBundle {
        embedding_service,
        pool: state.pool(),
        state: state as Arc<dyn StateStore>,
        event_store,
        plan_store,
        health_store,
    })
}
