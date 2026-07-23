//! Backend-neutral graph contract for long-term memory relationships.

use async_trait::async_trait;

use crate::types::{ChannelVisibility, FactPrivacy};

#[derive(Debug, Clone)]
pub struct GraphEntityUpsert {
    pub entity_type: String,
    pub canonical_name: String,
    pub display_name: String,
    pub aliases: Vec<String>,
    pub channel_id: Option<String>,
    pub privacy: FactPrivacy,
}

#[derive(Debug, Clone)]
pub struct GraphEdgeUpsert {
    pub source_entity_id: i64,
    pub target_entity_id: i64,
    pub relation: String,
    pub source_claim_id: i64,
    pub confidence: f32,
    pub channel_id: Option<String>,
    pub privacy: FactPrivacy,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GraphSearchScope {
    pub channel_id: Option<String>,
    pub visibility: ChannelVisibility,
    pub requester_is_owner: bool,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GraphNeighbor {
    pub entity_id: i64,
    pub display_name: String,
    pub relation: String,
    pub depth: usize,
}

#[async_trait]
pub trait MemoryGraphStore: Send + Sync {
    async fn upsert_entity(&self, entity: GraphEntityUpsert) -> anyhow::Result<i64>;
    async fn upsert_edge(&self, edge: GraphEdgeUpsert) -> anyhow::Result<()>;
    async fn invalidate_claim(&self, claim_id: i64, valid_to: &str) -> anyhow::Result<()>;
    #[allow(dead_code)]
    async fn neighbors(
        &self,
        start_entity_id: i64,
        max_depth: usize,
        scope: &GraphSearchScope,
    ) -> anyhow::Result<Vec<GraphNeighbor>>;
}
