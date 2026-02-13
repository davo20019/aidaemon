use std::sync::Arc;
use std::time::Duration;

use crate::config::AppConfig;
use crate::events::{Consolidator, EventStore, Pruner};
use crate::memory::embeddings::EmbeddingService;
use crate::memory::manager::MemoryManager;
use crate::plans::PlanStore;
use crate::router::{Router, Tier};
use crate::state::SqliteStateStore;
use crate::traits::ModelProvider;

pub struct MemoryPipelineBundle {
    pub consolidator: Arc<Consolidator>,
    pub pruner: Arc<Pruner>,
    pub memory_manager: Arc<MemoryManager>,
}

pub fn build_memory_pipeline(
    config: &AppConfig,
    state: Arc<SqliteStateStore>,
    event_store: Arc<EventStore>,
    plan_store: Arc<PlanStore>,
    provider: Arc<dyn ModelProvider>,
    router: &Router,
    embedding_service: Arc<EmbeddingService>,
) -> MemoryPipelineBundle {
    let consolidator = Arc::new(
        Consolidator::new(
            event_store.clone(),
            plan_store,
            state.pool(),
            Some(provider.clone()),
            router.select(Tier::Fast).to_string(),
            Some(embedding_service.clone()),
        )
        .with_state(state.clone())
        .with_learning_evidence_gate(config.policy.learning_evidence_gate_enforce),
    );

    let pruner = Arc::new(Pruner::new(
        event_store,
        consolidator.clone(),
        7, // 7-day retention
    ));

    let consolidation_interval =
        Duration::from_secs(config.state.consolidation_interval_hours * 3600);
    let memory_manager = Arc::new(
        MemoryManager::new(
            state.pool(),
            embedding_service,
            provider,
            router.select(Tier::Fast).to_string(),
            consolidation_interval,
            Some(consolidator.clone()),
        )
        .with_state(state)
        .with_people_config(config.people.clone()),
    );

    MemoryPipelineBundle {
        consolidator,
        pruner,
        memory_manager,
    }
}
