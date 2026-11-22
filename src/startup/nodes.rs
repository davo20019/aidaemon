use std::sync::Arc;

use crate::agent::Agent;
use crate::channels::ChannelHub;
use crate::config::AppConfig;
use crate::nodes::channel::NodeChannel;
use crate::nodes::gateway::{NodeGatewayState, NodeRateLimiter};
use crate::nodes::service::{AgentNodeConversationIngress, NodeService};
use crate::nodes::{NodeStore, auth};
use crate::traits::HubChannel;

/// The running Node service and how long a Node may be silent and still
/// count as online, for the apps' System area.
static RUNNING: std::sync::OnceLock<(Arc<NodeService>, u64)> = std::sync::OnceLock::new();

/// The owner's devices for the apps: name, kind, whether online, version
/// and how each last reached the Gateway. Empty when Nodes are off.
pub(crate) async fn infos(pool: &sqlx::SqlitePool) -> Vec<aidaemon_client_protocol::NodeInfo> {
    use aidaemon_client_protocol::{NodeConnection, NodeInfo};
    let Some((service, offline_after)) = RUNNING.get() else {
        return Vec::new();
    };
    let rows = sqlx::query_as::<_, (String, String, String, Option<String>, Option<String>)>(
        "SELECT node_id, display_name, kind, last_seen_at, last_firmware_version FROM nodes \
         WHERE revoked_at IS NULL ORDER BY display_name",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    let now = chrono::Utc::now();
    rows.into_iter()
        .map(|(node_id, name, kind, last_seen, firmware)| {
            let seen = last_seen
                .as_deref()
                .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok())
                .map(|at| at.with_timezone(&chrono::Utc));
            let online = seen.is_some_and(|at| {
                (now - at).num_seconds() <= i64::try_from(*offline_after).unwrap_or(i64::MAX)
            });
            let connection = service.link(&node_id).map(|link| match link {
                crate::nodes::domain::NodeLink::LocalNetwork => NodeConnection::LocalNetwork,
                crate::nodes::domain::NodeLink::Tunnel => NodeConnection::Tunnel,
                crate::nodes::domain::NodeLink::Local => NodeConnection::Local,
            });
            NodeInfo {
                node_id,
                name,
                kind,
                online,
                last_seen_ms: seen.and_then(|at| u64::try_from(at.timestamp_millis()).ok()),
                firmware_version: firmware.filter(|v| !v.is_empty()),
                connection,
            }
        })
        .collect()
}

pub async fn start(
    config: &AppConfig,
    pool: sqlx::SqlitePool,
    agent: Arc<Agent>,
    hub: Arc<ChannelHub>,
    endpoints: aidaemon_core::EndpointSource,
) -> anyhow::Result<()> {
    if !config.nodes.gateway.enabled {
        return Ok(());
    }
    let instance_key = auth::load_or_create_instance_key()?;
    let store = Arc::new(NodeStore::new(pool, instance_key));
    let mut service_value =
        NodeService::new(store.clone(), config.nodes.clone()).with_endpoints(endpoints);
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
    let lan_authority = if config.nodes.gateway.lan.enabled {
        let path = dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".aidaemon")
            .join("node-lan-ca.key");
        match crate::nodes::lan::LanAuthority::load_or_create(&path) {
            Ok(authority) => Some(Arc::new(authority)),
            Err(error) => {
                tracing::warn!(error = %format!("{error:#}"), "The Node Gateway's local listener is off");
                None
            }
        }
    } else {
        None
    };
    if let Some(authority) = &lan_authority {
        service_value = service_value.with_lan(crate::nodes::lan::LanEndpoint {
            port: config.nodes.gateway.lan.port,
            authority: authority.clone(),
            address: None,
        });
    }
    let service = Arc::new(service_value);
    let _ = RUNNING.set((
        service.clone(),
        config.nodes.monitoring.default_offline_after_seconds,
    ));
    let ingress = Arc::new(AgentNodeConversationIngress::new(agent));
    hub.register_channel(Arc::new(NodeChannel::new(store)) as Arc<dyn HubChannel>)
        .await;
    let state = NodeGatewayState {
        service,
        ingress,
        hub,
        config: config.nodes.clone(),
        rate_limiter: Arc::new(NodeRateLimiter::default()),
    };
    // The same routes on the local network, with the daemon's own CA.
    if let Some(authority) = lan_authority {
        let app = crate::nodes::gateway::router(state.clone());
        let port = config.nodes.gateway.lan.port;
        tokio::spawn(async move {
            if let Err(error) = crate::nodes::lan::serve(app, authority, port).await {
                tracing::error!(error = %format!("{error:#}"), "Node Gateway's local listener stopped");
            }
        });
    }
    tokio::spawn(async move {
        if let Err(error) = crate::nodes::gateway::serve(state).await {
            tracing::error!(%error, "Node Gateway stopped");
        }
    });
    Ok(())
}
