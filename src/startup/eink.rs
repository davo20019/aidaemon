//! E-ink access (`[client_gateway.eink]`): when a bridge is named, the
//! daemon dials out to it and serves paired e-ink devices through the
//! gateway's bridge router. Off unless `bridge_url` is set.

use std::sync::{Arc, OnceLock};

use aidaemon_bridge_connector::{BridgeKeys, Connector};
use aidaemon_client_gateway::{ClientGateway, ClientGatewayConfig};

static CONNECTOR: OnceLock<Arc<Connector>> = OnceLock::new();

/// Start the bridge connector if the config names a bridge. A failure
/// leaves e-ink access off and everything else serving.
pub(crate) fn start(config: &ClientGatewayConfig, gateway: Arc<ClientGateway>) {
    let Some(url) = config
        .eink
        .bridge_url
        .as_deref()
        .map(str::trim)
        .filter(|url| !url.is_empty())
    else {
        return;
    };
    // Machine format: a URL scheme. The bridge is reached only over TLS,
    // except a bridge on this computer (for development).
    let local = url.starts_with("ws://127.0.0.1") || url.starts_with("ws://localhost");
    if !url.starts_with("wss://") && !local {
        tracing::warn!("E-ink access is off: client_gateway.eink.bridge_url must be wss://");
        return;
    }
    let key_path = aidaemon_client_gateway::token::token_file(config).with_file_name("bridge-keys");
    let keys = match BridgeKeys::load_or_create(&key_path) {
        Ok(keys) => keys,
        Err(error) => {
            tracing::warn!(error = %format!("{error:#}"), "E-ink access is off: no bridge keys");
            return;
        }
    };
    let connector = Connector::start(gateway, url.to_string(), keys);
    tracing::info!(
        route = %connector.route_id_hex(),
        "E-ink access on: dialing the bridge"
    );
    let _ = CONNECTOR.set(connector);
}
