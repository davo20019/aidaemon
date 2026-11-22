//! The gateway's settings (`[client_gateway]` in the daemon's config).

use std::net::IpAddr;

use aidaemon_client_protocol::Risk;
use serde::Deserialize;

/// How the Client Gateway runs. Off by default.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientGatewayConfig {
    #[serde(default)]
    pub enabled: bool,
    /// The address to listen on. Only loopback is served: a remote client
    /// reaches the gateway through a TLS tunnel to this address.
    #[serde(default = "default_bind")]
    pub bind: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Where the local access token's hash is kept (`aidaemon client-token`
    /// writes it). Empty: the daemon's default path.
    #[serde(default)]
    pub token_file: Option<String>,
    /// Events older than this are dropped from the resume log; a client
    /// further behind gets a snapshot.
    #[serde(default = "default_retention_days")]
    pub retention_days: u64,
    /// The most events the resume log keeps.
    #[serde(default = "default_max_events")]
    pub max_events: u64,
    /// The highest risk a watch may approve; above it a watch may only deny.
    #[serde(default = "default_watch_risk_ceiling")]
    pub watch_risk_ceiling: Risk,
    /// The address paired devices reach the gateway at (an owner tunnel's
    /// HTTPS URL), put in pairing links. Empty: the loopback address.
    #[serde(default)]
    pub public_url: Option<String>,
    /// A directory holding the nsh app's published web build (`nsh package
    /// --target web` in `clients/app`), served at `/client/v1/next/` from
    /// the gateway's own origin. Empty: not served.
    #[serde(default)]
    pub app_dir: Option<String>,
    /// Remote access over iroh (`[client_gateway.remote]`): the owner's
    /// apps reach this gateway from anywhere by the daemon's key, with no
    /// tunnel. On whenever the gateway is.
    #[serde(default)]
    pub remote: RemoteAccessConfig,
}

/// `[client_gateway.remote]`: serve the gateway over iroh too.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteAccessConfig {
    /// On by default: the daemon keeps a connection to its home relay
    /// (which sees its address, never its traffic). `false` turns it off.
    #[serde(default = "default_remote_enabled")]
    pub enabled: bool,
    /// Relay URLs (your own `iroh-relay`). Empty: the aidaemon project's
    /// relays (`relay.aidaemon.ai`).
    #[serde(default)]
    pub relays: Vec<String>,
}

fn default_remote_enabled() -> bool {
    true
}

impl Default for RemoteAccessConfig {
    fn default() -> Self {
        Self {
            enabled: default_remote_enabled(),
            relays: Vec::new(),
        }
    }
}

fn default_bind() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    8788
}
fn default_retention_days() -> u64 {
    7
}
fn default_max_events() -> u64 {
    100_000
}
fn default_watch_risk_ceiling() -> Risk {
    Risk::Medium
}

impl Default for ClientGatewayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bind: default_bind(),
            port: default_port(),
            token_file: None,
            retention_days: default_retention_days(),
            max_events: default_max_events(),
            watch_risk_ceiling: default_watch_risk_ceiling(),
            public_url: None,
            app_dir: None,
            remote: RemoteAccessConfig::default(),
        }
    }
}

impl ClientGatewayConfig {
    /// Where devices reach the gateway: `public_url`, else the bind address.
    pub fn public_url(&self) -> String {
        match self.public_url.as_deref().map(str::trim) {
            Some(url) if !url.is_empty() => url.trim_end_matches('/').to_string(),
            _ => match self.bind.parse::<IpAddr>() {
                Ok(IpAddr::V6(ip)) => format!("http://[{ip}]:{}", self.port),
                _ => format!("http://{}:{}", self.bind, self.port),
            },
        }
    }

    /// The path of the web client pairing links open: the nsh app when its
    /// build is served (`app_dir`), else the page saying it is not installed.
    pub fn web_app_path(&self) -> &'static str {
        match self.app_dir.as_deref().map(str::trim) {
            Some(dir) if !dir.is_empty() => "/client/v1/next/",
            _ => "/client/v1/app",
        }
    }

    /// The address to bind, refused unless it is loopback: this gateway
    /// does not serve TLS, and a client token can approve a shell command.
    pub fn bind_addr(&self) -> anyhow::Result<std::net::SocketAddr> {
        let ip: IpAddr = self
            .bind
            .parse()
            .map_err(|_| anyhow::anyhow!("client_gateway.bind must be an IP address"))?;
        anyhow::ensure!(
            ip.is_loopback(),
            "client_gateway.bind must be loopback: the gateway does not serve TLS, so reach \
             it remotely through a TLS tunnel (Tailscale, Cloudflare Tunnel) to a loopback bind"
        );
        Ok(std::net::SocketAddr::new(ip, self.port))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_access_is_on_unless_turned_off() {
        let on: ClientGatewayConfig = serde_json::from_str(r#"{"enabled": true}"#).unwrap();
        assert!(on.remote.enabled);
        assert!(on.remote.relays.is_empty());
        assert!(ClientGatewayConfig::default().remote.enabled);
        let off: ClientGatewayConfig =
            serde_json::from_str(r#"{"enabled": true, "remote": {"enabled": false}}"#).unwrap();
        assert!(!off.remote.enabled);
    }

    #[test]
    fn off_and_loopback_by_default() {
        let config: ClientGatewayConfig = serde_json::from_str("{}").unwrap();
        assert!(!config.enabled);
        assert!(config.bind_addr().unwrap().ip().is_loopback());
        assert_eq!(config.watch_risk_ceiling, Risk::Medium);
    }

    #[test]
    fn a_non_loopback_bind_is_refused() {
        let config = ClientGatewayConfig {
            bind: "0.0.0.0".into(),
            ..ClientGatewayConfig::default()
        };
        assert!(config.bind_addr().unwrap_err().to_string().contains("TLS"));
        let lan = ClientGatewayConfig {
            bind: "192.168.1.20".into(),
            ..ClientGatewayConfig::default()
        };
        assert!(lan.bind_addr().is_err());
        let v6 = ClientGatewayConfig {
            bind: "::1".into(),
            ..ClientGatewayConfig::default()
        };
        assert!(v6.bind_addr().is_ok());
    }
}
