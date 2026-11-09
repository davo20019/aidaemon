use std::sync::Arc;
use std::time::Duration;

use crate::config::AppConfig;
use crate::events::{Consolidator, EventStore, Pruner};
use crate::llm_runtime::gateway::SideCallGateway;
use crate::llm_runtime::SharedLlmRuntime;
use crate::memory::embeddings::EmbeddingService;
use crate::memory::manager::MemoryManager;
use crate::plans::PlanStore;
use crate::traits::StateStore;
use sqlx::SqlitePool;

pub struct MemoryPipelineBundle {
    pub consolidator: Arc<Consolidator>,
    pub pruner: Arc<Pruner>,
    pub memory_manager: Arc<MemoryManager>,
}

#[allow(clippy::too_many_arguments)]
pub fn build_memory_pipeline(
    config: &AppConfig,
    state: Arc<dyn StateStore>,
    memory_store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    pool: SqlitePool,
    event_store: Arc<EventStore>,
    plan_store: Arc<PlanStore>,
    llm_runtime: SharedLlmRuntime,
    side_calls: SideCallGateway,
    embedding_service: Arc<EmbeddingService>,
) -> MemoryPipelineBundle {
    let consolidator = Arc::new(
        Consolidator::new(
            event_store.clone(),
            plan_store,
            pool.clone(),
            memory_store.clone(),
            Some(llm_runtime.clone()),
            side_calls.clone(),
            Some(embedding_service.clone()),
        )
        .with_learning_evidence_gate(config.policy.learning_evidence_gate_enforce),
    );

    let pruner = Arc::new(Pruner::new(
        event_store.clone(),
        consolidator.clone(),
        7, // 7-day retention
    ));

    let consolidation_interval =
        Duration::from_secs(config.state.consolidation_interval_hours * 3600);
    let memory_manager = Arc::new(
        MemoryManager::new(
            pool,
            embedding_service,
            llm_runtime,
            side_calls,
            consolidation_interval,
            Some(consolidator.clone()),
        )
        .with_state(state)
        .with_memory(memory_store),
    );

    MemoryPipelineBundle {
        consolidator,
        pruner,
        memory_manager,
    }
}
