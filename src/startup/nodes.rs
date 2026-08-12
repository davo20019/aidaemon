use std::sync::Arc;

use crate::agent::Agent;
use crate::channels::ChannelHub;
use crate::config::AppConfig;
use crate::nodes::channel::NodeChannel;
use crate::nodes::gateway::{NodeGatewayState, NodeRateLimiter};
use crate::nodes::service::{AgentNodeConversationIngress, NodeService};
use crate::nodes::{auth, NodeStore};
use crate::traits::Channel;

pub async fn start(
    config: &AppConfig,
    pool: sqlx::SqlitePool,
    agent: Arc<Agent>,
    hub: Arc<ChannelHub>,
) -> anyhow::Result<()> {
    if !config.nodes.gateway.enabled {
        return Ok(());
    }
    let instance_key = auth::load_or_create_instance_key()?;
    let store = Arc::new(NodeStore::new(pool, instance_key));
    let mut service_value = NodeService::new(store.clone(), config.nodes.clone());
    if config.nodes.ota.enabled {
        let release = crate::nodes::ota::FirmwareRelease::load(&config.nodes.ota)?;
        tracing::info!(
            release_id = %release.manifest().release_id,
            version = %release.manifest().version,
            sequence = release.manifest().sequence,
            bytes = release.manifest().size_bytes,
            "validated immutable Node firmware release"
        );
        service_value = service_value.with_firmware_release(release);
    }
    if config.nodes.speech.enabled {
        service_value = service_value.with_speech(crate::nodes::speech::configured_synthesizer(
            &config.nodes.speech,
        )?);
    }
    let service = Arc::new(service_value);
    let ingress = Arc::new(AgentNodeConversationIngress::new(agent));
    hub.register_channel(Arc::new(NodeChannel::new(store)) as Arc<dyn Channel>)
        .await;
    let state = NodeGatewayState {
        service,
        ingress,
        hub,
        config: config.nodes.clone(),
        rate_limiter: Arc::new(NodeRateLimiter::default()),
    };
    tokio::spawn(async move {
        if let Err(error) = crate::nodes::gateway::serve(state).await {
            tracing::error!(%error, "Node Gateway stopped");
        }
    });
    Ok(())
}
