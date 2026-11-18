//! `[exposure.*]` on the daemon: the supervisor that publishes the gateways
//! through their tunnels, started after both gateways, stopped with the
//! daemon's `shutdown` token. Read once at startup; changes apply after a
//! restart. Also `/exposure`.

use std::path::PathBuf;

use aidaemon_core::{CancellationToken, EndpointPublisher, EndpointSource};
use aidaemon_exposure::{
    Backoff, ExposedService, ExposurePlan, ExposureSupervisor, ServiceListeners,
    supervisor::StatusHandle,
};
use async_trait::async_trait;

use crate::config::AppConfig;
use crate::types::UserRole;

/// Each gateway's endpoint source, and the publisher the supervisor gets
/// when that gateway has an `[exposure]` table.
pub struct GatewayEndpoints {
    pub client: EndpointSource,
    pub node: EndpointSource,
    client_publisher: Option<EndpointPublisher>,
    node_publisher: Option<EndpointPublisher>,
}

/// The sources the two gateways start with: live where an adapter is
/// configured, the static config origins either way.
pub fn endpoints(config: &AppConfig) -> GatewayEndpoints {
    let client_static = crate::startup::client_gateway::static_origins(&config.client_gateway);
    let node_static = config.nodes.gateway.advertised_endpoints.clone();
    let (client, client_publisher) = match config.exposure.client_gateway {
        Some(_) if config.client_gateway.enabled => {
            let (source, publisher) = EndpointSource::live(client_static);
            (source, Some(publisher))
        }
        _ => (EndpointSource::fixed(client_static), None),
    };
    let (node, node_publisher) = match config.exposure.node_gateway {
        Some(_) if config.nodes.gateway.enabled => {
            let (source, publisher) = EndpointSource::live(node_static);
            (source, Some(publisher))
        }
        _ => (EndpointSource::fixed(node_static), None),
    };
    GatewayEndpoints {
        client,
        node,
        client_publisher,
        node_publisher,
    }
}

/// Where adapters keep generated tunnel configs and child pid files.
fn state_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".aidaemon")
        .join("exposure")
}

static STATUS: std::sync::OnceLock<StatusHandle> = std::sync::OnceLock::new();

/// Start the supervisor for every configured service (none configured: no
/// supervisor). Gateways never wait for it: until a tunnel is up they serve
/// their static origins.
pub fn start(
    config: &AppConfig,
    endpoints: &mut GatewayEndpoints,
    shutdown: CancellationToken,
) -> Option<ExposureSupervisor> {
    let mut listeners = ServiceListeners::new();
    if config.client_gateway.enabled {
        match config.client_gateway.bind_addr() {
            Ok(addr) => {
                if let Err(error) = listeners.register(ExposedService::ClientGateway, addr) {
                    tracing::warn!(%error, "Client Gateway listener not exposable");
                }
            }
            Err(error) => tracing::warn!(%error, "Client Gateway listener not exposable"),
        }
    }
    if config.nodes.gateway.enabled
        && let Ok(ip) = config.nodes.gateway.bind.parse::<std::net::IpAddr>()
    {
        let addr = std::net::SocketAddr::new(ip, config.nodes.gateway.port);
        if let Err(error) = listeners.register(ExposedService::NodeGateway, addr) {
            tracing::warn!(%error, "Node Gateway listener not exposable");
        }
    }
    let state_dir = state_dir();
    let mut plans = Vec::new();
    for (service, settings) in config.exposure.services() {
        let publisher = match service {
            ExposedService::ClientGateway => endpoints.client_publisher.take(),
            ExposedService::NodeGateway => endpoints.node_publisher.take(),
            ExposedService::Webhook { .. } => None,
        };
        let Some(publisher) = publisher else {
            continue;
        };
        plans.push(ExposurePlan {
            adapter: settings.build(&service, state_dir.clone()),
            service,
            publisher,
        });
    }
    if plans.is_empty() {
        return None;
    }
    let supervisor = ExposureSupervisor::spawn(plans, &listeners, shutdown, Backoff::default());
    let _ = STATUS.set(supervisor.status_handle());
    Some(supervisor)
}

/// `/exposure`: per service, adapter, state, origin, time since the last
/// change and the last failure. Read-only.
pub(crate) struct ExposureCommand;

fn ago(since: std::time::SystemTime) -> String {
    let secs = since.elapsed().map(|d| d.as_secs()).unwrap_or(0);
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m", secs / 60),
        3600..=86_399 => format!("{}h {}m", secs / 3600, (secs % 3600) / 60),
        _ => format!("{}d", secs / 86_400),
    }
}

pub(crate) fn status_text(statuses: &[aidaemon_exposure::ServiceStatus]) -> String {
    let mut text = "Exposure:".to_string();
    for status in statuses {
        text.push_str(&format!(
            "\n• {} via {}: {} for {}",
            status.service,
            status.adapter,
            status.state.as_str(),
            ago(status.since)
        ));
        if let Some(origin) = &status.origin {
            text.push_str(&format!(", at {origin}"));
        }
        if status.restarts > 0 {
            text.push_str(&format!(", {} restarts", status.restarts));
        }
        if let Some(reason) = &status.last_failure {
            text.push_str(&format!("\n  last failure: {reason}"));
        }
    }
    text
}

#[async_trait]
impl aidaemon_channels::slash::SlashCommand for ExposureCommand {
    fn handles(&self, cmd: &str) -> bool {
        cmd == "/exposure"
    }

    fn commands(&self) -> Vec<aidaemon_channels::slash::CommandDef> {
        vec![aidaemon_channels::slash::CommandDef {
            name: "exposure",
            description: "Show the tunnels publishing the gateways",
            usage: None,
            category: aidaemon_channels::slash::CommandCategory::Core,
        }]
    }

    async fn run(
        &self,
        _cmd: &str,
        _args: &str,
        call: &aidaemon_channels::slash::SlashCall<'_>,
    ) -> String {
        if call.user_role != UserRole::Owner {
            return "Only the owner can see how the gateways are exposed.".to_string();
        }
        match STATUS.get() {
            Some(handle) => status_text(&handle.status()),
            None => "Nothing is exposed: add an `[exposure.client_gateway]` or \
                     `[exposure.node_gateway]` table to config.toml (or ask me to enable \
                     exposure) and restart."
                .to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_exposure::{ServiceState, ServiceStatus};

    /// Every command the device and exposure handlers list in the menu and
    /// `/help` is one they handle.
    #[test]
    fn listed_commands_are_handled() {
        use aidaemon_channels::slash::SlashCommand;
        let handlers: [&dyn SlashCommand; 2] = [
            &ExposureCommand,
            &crate::startup::client_gateway::DeviceCommands,
        ];
        for handler in handlers {
            let commands = handler.commands();
            assert!(!commands.is_empty());
            for command in commands {
                assert!(
                    handler.handles(&format!("/{}", command.name)),
                    "{}",
                    command.name
                );
            }
        }
    }

    #[test]
    fn status_text_names_state_origin_and_failure() {
        let text = status_text(&[ServiceStatus {
            service: ExposedService::ClientGateway,
            adapter: "tailscale".into(),
            state: ServiceState::Down,
            origin: None,
            since: std::time::SystemTime::now(),
            last_failure: Some("tailscale is not installed".into()),
            restarts: 2,
        }]);
        assert!(
            text.contains("client_gateway via tailscale: down"),
            "{text}"
        );
        assert!(text.contains("2 restarts") && text.contains("not installed"));
    }
}
