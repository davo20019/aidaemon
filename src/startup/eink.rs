//! E-ink access (`[client_gateway.eink]`): when a bridge is named, the
//! daemon dials out to it and serves paired e-ink devices through the
//! gateway's bridge router. Off unless `bridge_url` is set. Settings turns
//! it on and off while the daemon runs ([`apply`]).

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use aidaemon_bridge_connector::{BridgeKeys, Connector};
use aidaemon_client_gateway::{ClientGateway, ClientGatewayConfig};

/// The project's bridge, the one the app's Settings offers first.
#[cfg(test)]
pub(crate) const PROJECT_BRIDGE: &str = "wss://bridge.aidaemon.ai/v1/daemon";

/// The setting that names the bridge.
pub(crate) const BRIDGE_URL_KEY: &str = "client_gateway.eink.bridge_url";

/// What the connector runs with: the gateway and where its keys live.
struct Host {
    gateway: Arc<ClientGateway>,
    key_path: PathBuf,
}

/// The connector running now, and the bridge it dials.
#[derive(Default)]
pub(crate) struct Slot {
    running: Option<(String, Arc<Connector>)>,
}

static HOST: OnceLock<Host> = OnceLock::new();
static SLOT: Mutex<Slot> = Mutex::new(Slot { running: None });

/// Whether `url` may name a bridge: `wss://` only, except a bridge on this
/// computer (for development). Machine format: a URL scheme and host.
pub(crate) fn check_bridge_url(url: &str) -> Result<(), String> {
    let url = url.trim();
    let local = [
        "ws://127.0.0.1:",
        "ws://127.0.0.1/",
        "ws://localhost:",
        "ws://localhost/",
    ]
    .iter()
    .any(|prefix| url.starts_with(prefix));
    let tls = url
        .strip_prefix("wss://")
        .is_some_and(|rest| !rest.is_empty() && !rest.starts_with('/'));
    if url.chars().any(char::is_whitespace) || !(tls || local) {
        return Err(
            "The e-ink bridge must be a wss:// address (ws:// only on this computer)".to_string(),
        );
    }
    Ok(())
}

/// Start the bridge connector if the config names a bridge. A failure
/// leaves e-ink access off and everything else serving.
pub(crate) fn start(config: &ClientGatewayConfig, gateway: Arc<ClientGateway>) {
    let key_path = aidaemon_client_gateway::token::token_file(config).with_file_name("bridge-keys");
    let _ = HOST.set(Host { gateway, key_path });
    if let Err(why) = apply(config.eink.bridge_url.as_deref()) {
        tracing::warn!("E-ink access is off: {why}");
    }
}

/// Turn e-ink access to `bridge_url` (`None` or empty: off), now: the
/// connector running for another bridge stops first. `Ok(true)`: it runs.
pub(crate) fn apply(bridge_url: Option<&str>) -> Result<bool, String> {
    let host = HOST.get();
    let mut slot = SLOT.lock().unwrap_or_else(|p| p.into_inner());
    apply_in(
        &mut slot,
        host.map(|h| (&h.gateway, h.key_path.as_path())),
        bridge_url,
    )
}

/// [`apply`] on `slot`, for `host` (`None`: no gateway running, so only
/// turning off works).
pub(crate) fn apply_in(
    slot: &mut Slot,
    host: Option<(&Arc<ClientGateway>, &std::path::Path)>,
    bridge_url: Option<&str>,
) -> Result<bool, String> {
    let wanted = bridge_url.map(str::trim).filter(|url| !url.is_empty());
    if let Some(url) = wanted {
        check_bridge_url(url)?;
    }
    if slot.running.as_ref().map(|(url, _)| url.as_str()) == wanted {
        return Ok(wanted.is_some());
    }
    if let Some((_, connector)) = slot.running.take() {
        connector.stop();
    }
    let Some(url) = wanted else {
        return Ok(false);
    };
    let (gateway, key_path) = host.ok_or("The client gateway is not running")?;
    let keys = BridgeKeys::load_or_create(key_path)
        .map_err(|error| format!("No bridge keys: {error:#}"))?;
    let connector = Connector::start(gateway.clone(), url.to_string(), keys);
    tracing::info!(
        route = %connector.route_id_hex(),
        "E-ink access on: dialing the bridge"
    );
    slot.running = Some((url.to_string(), connector));
    Ok(true)
}

impl Slot {
    /// The connector running now, if any.
    #[cfg(test)]
    pub(crate) fn connector(&self) -> Option<Arc<Connector>> {
        self.running.as_ref().map(|(_, c)| c.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_tls_bridge_or_one_on_this_computer() {
        for good in [
            PROJECT_BRIDGE,
            "wss://bridge.example.test/v1/daemon",
            "ws://127.0.0.1:8790/v1/daemon",
            "ws://localhost:8790/v1/daemon",
        ] {
            assert!(check_bridge_url(good).is_ok(), "{good}");
        }
        for bad in [
            "",
            "wss://",
            "ws://bridge.example.test/v1/daemon",
            "ws://127.0.0.1.example.test/v1/daemon",
            "ws://localhost.example.test/v1/daemon",
            "https://bridge.example.test",
            "wss://bridge.example.test/v1 daemon",
            "bridge.example.test",
        ] {
            assert!(check_bridge_url(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn turning_off_with_nothing_running_is_fine_and_a_bad_url_is_refused() {
        let mut slot = Slot::default();
        assert_eq!(apply_in(&mut slot, None, None), Ok(false));
        assert_eq!(apply_in(&mut slot, None, Some("  ")), Ok(false));
        assert!(apply_in(&mut slot, None, Some("ws://bridge.example.test")).is_err());
        assert!(
            apply_in(&mut slot, None, Some(PROJECT_BRIDGE)).is_err(),
            "no gateway"
        );
        assert!(slot.connector().is_none());
    }
}
