//! `[exposure.client_gateway]` and `[exposure.node_gateway]`: which adapter
//! publishes each gateway. Read once at startup.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::adapters::{
    cloudflare::CloudflareAdapter, command::CommandAdapter, manual::ManualAdapter,
    tailscale::TailscaleAdapter,
};
use crate::{ExposedService, ExposureAdapter, OriginStability, validate_https_origin};

/// The `[exposure]` table. No table: nothing is exposed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExposureConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_gateway: Option<ServiceExposureConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_gateway: Option<ServiceExposureConfig>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloudflareMode {
    /// A named tunnel on the owner's own hostname: stable.
    #[default]
    Named,
    /// A `trycloudflare.com` quick tunnel: ephemeral, for testing.
    Quick,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TailscaleMode {
    /// Private to the owner's tailnet.
    #[default]
    Serve,
    /// Public on the internet through Tailscale Funnel.
    Funnel,
}

/// One service's adapter, tagged by `adapter`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "adapter", rename_all = "snake_case", deny_unknown_fields)]
pub enum ServiceExposureConfig {
    /// A reverse proxy the owner runs; the daemon only checks it answers.
    Manual { url: String },
    Cloudflare {
        #[serde(default)]
        mode: CloudflareMode,
        /// Named: the public hostname routed to the tunnel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hostname: Option<String>,
        /// Named: the tunnel's name or UUID.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tunnel: Option<String>,
        /// Named: the tunnel credentials file, when not in `~/.cloudflared`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        credentials_file: Option<String>,
        /// The `cloudflared` binary, when not on `PATH`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        binary: Option<String>,
    },
    Tailscale {
        #[serde(default)]
        mode: TailscaleMode,
        /// The HTTPS port on the tailnet name (default 443 for the Client
        /// Gateway, 8443 for the Node Gateway).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        https_port: Option<u16>,
        /// The `tailscale` binary, when not on `PATH`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        binary: Option<String>,
    },
    /// Any program that prints `{"url": "https://..."}` on its first stdout
    /// line and stays running. Configured by hand only.
    Command {
        command: Vec<String>,
        #[serde(default)]
        stable: bool,
    },
}

/// Who can reach the published service, for the owner's approval card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Audience {
    Public,
    Tailnet,
    Proxy,
}

impl Audience {
    pub fn describe(self) -> &'static str {
        match self {
            Self::Public => "public on the internet",
            Self::Tailnet => "private to your tailnet",
            Self::Proxy => "wherever your proxy publishes it",
        }
    }
}

/// The stable options, named in refusals.
const STABLE_OPTIONS: &str = "cloudflare (mode = \"named\"), tailscale, manual, or command with \
                              stable = true";

impl ExposureConfig {
    /// The configured services with their settings.
    pub fn services(&self) -> Vec<(ExposedService, &ServiceExposureConfig)> {
        let mut services = Vec::new();
        if let Some(config) = &self.client_gateway {
            services.push((ExposedService::ClientGateway, config));
        }
        if let Some(config) = &self.node_gateway {
            services.push((ExposedService::NodeGateway, config));
        }
        services
    }

    /// Checks at config load: each table needs its gateway on, the Node
    /// Gateway needs a stable origin, and every origin passes the HTTPS
    /// check. Two services never share a Cloudflare tunnel.
    pub fn validate(&self, client_enabled: bool, node_enabled: bool) -> Result<(), String> {
        if self.client_gateway.is_some() && !client_enabled {
            return Err(
                "exposure.client_gateway requires client_gateway.enabled = true".to_string(),
            );
        }
        if self.node_gateway.is_some() && !node_enabled {
            return Err("exposure.node_gateway requires nodes.gateway.enabled = true".to_string());
        }
        for (service, config) in self.services() {
            config
                .validate(&service)
                .map_err(|error| format!("exposure.{service}: {error}"))?;
        }
        if let (
            Some(ServiceExposureConfig::Cloudflare {
                mode: CloudflareMode::Named,
                tunnel: Some(a),
                ..
            }),
            Some(ServiceExposureConfig::Cloudflare {
                mode: CloudflareMode::Named,
                tunnel: Some(b),
                ..
            }),
        ) = (&self.client_gateway, &self.node_gateway)
            && a.trim() == b.trim()
        {
            return Err(
                "exposure.client_gateway and exposure.node_gateway must use different \
                     Cloudflare tunnels: one tunnel publishes one service"
                    .to_string(),
            );
        }
        if let (
            Some(ServiceExposureConfig::Tailscale { .. }),
            Some(ServiceExposureConfig::Tailscale { .. }),
        ) = (&self.client_gateway, &self.node_gateway)
        {
            let client = self.client_gateway.as_ref().unwrap();
            let node = self.node_gateway.as_ref().unwrap();
            if client.tailscale_port(&ExposedService::ClientGateway)
                == node.tailscale_port(&ExposedService::NodeGateway)
            {
                return Err(
                    "exposure.client_gateway and exposure.node_gateway must use different \
                     tailscale https_port values"
                        .to_string(),
                );
            }
        }
        Ok(())
    }
}

fn is_hostname(host: &str) -> bool {
    !host.is_empty()
        && host.len() <= 253
        && host.contains('.')
        && host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
        })
}

impl ServiceExposureConfig {
    /// `manual`, `cloudflare`, `tailscale` or `command`.
    pub fn adapter_name(&self) -> &'static str {
        match self {
            Self::Manual { .. } => "manual",
            Self::Cloudflare { .. } => "cloudflare",
            Self::Tailscale { .. } => "tailscale",
            Self::Command { .. } => "command",
        }
    }

    pub fn stability(&self) -> OriginStability {
        match self {
            Self::Cloudflare {
                mode: CloudflareMode::Quick,
                ..
            }
            | Self::Command { stable: false, .. } => OriginStability::Ephemeral,
            _ => OriginStability::Stable,
        }
    }

    pub fn audience(&self) -> Audience {
        match self {
            Self::Tailscale {
                mode: TailscaleMode::Serve,
                ..
            } => Audience::Tailnet,
            Self::Manual { .. } | Self::Command { .. } => Audience::Proxy,
            _ => Audience::Public,
        }
    }

    fn tailscale_port(&self, service: &ExposedService) -> Option<u16> {
        match self {
            Self::Tailscale { https_port, .. } => Some(https_port.unwrap_or(
                if matches!(service, ExposedService::NodeGateway) {
                    8443
                } else {
                    443
                },
            )),
            _ => None,
        }
    }

    pub fn validate(&self, service: &ExposedService) -> Result<(), String> {
        if service.needs_stable_origin() && self.stability() == OriginStability::Ephemeral {
            return Err(format!(
                "{service} needs an origin that survives restarts; use {STABLE_OPTIONS}"
            ));
        }
        match self {
            Self::Manual { url } => {
                validate_https_origin(url).map_err(|error| format!("url {error}"))?;
            }
            Self::Cloudflare {
                mode: CloudflareMode::Named,
                hostname,
                tunnel,
                ..
            } => {
                let hostname = hostname
                    .as_deref()
                    .map(str::trim)
                    .filter(|h| !h.is_empty())
                    .ok_or("cloudflare named tunnels need `hostname`")?;
                if !is_hostname(hostname) {
                    return Err("hostname must be a DNS name like app.example.com".to_string());
                }
                validate_https_origin(&format!("https://{hostname}"))
                    .map_err(|error| format!("hostname {error}"))?;
                if tunnel.as_deref().map(str::trim).unwrap_or("").is_empty() {
                    return Err("cloudflare named tunnels need `tunnel`".to_string());
                }
            }
            Self::Cloudflare {
                mode: CloudflareMode::Quick,
                hostname,
                tunnel,
                credentials_file,
                ..
            } => {
                if hostname.is_some() || tunnel.is_some() || credentials_file.is_some() {
                    return Err(
                        "cloudflare quick tunnels take no hostname, tunnel or credentials_file"
                            .to_string(),
                    );
                }
            }
            Self::Tailscale { https_port, .. } => {
                if https_port == &Some(0) {
                    return Err("https_port must not be zero".to_string());
                }
            }
            Self::Command { command, .. } => {
                if command.first().map(|p| p.trim().is_empty()).unwrap_or(true) {
                    return Err("command needs a program as its first entry".to_string());
                }
            }
        }
        Ok(())
    }

    /// The adapter for this config. `state_dir` holds generated tunnel
    /// configs and child pid files.
    pub fn build(&self, service: &ExposedService, state_dir: PathBuf) -> Arc<dyn ExposureAdapter> {
        match self {
            Self::Manual { url } => Arc::new(ManualAdapter::new(url.clone())),
            Self::Cloudflare {
                mode,
                hostname,
                tunnel,
                credentials_file,
                binary,
            } => Arc::new(CloudflareAdapter {
                mode: *mode,
                hostname: hostname.clone().unwrap_or_default(),
                tunnel: tunnel.clone().unwrap_or_default(),
                credentials_file: credentials_file.clone(),
                binary: binary.clone(),
                state_dir,
            }),
            Self::Tailscale { mode, binary, .. } => Arc::new(TailscaleAdapter {
                mode: *mode,
                https_port: self.tailscale_port(service).unwrap_or(443),
                binary: binary.clone(),
                state_dir,
            }),
            Self::Command { command, stable } => Arc::new(CommandAdapter {
                argv: command.clone(),
                stable: *stable,
                state_dir,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<ExposureConfig, String> {
        toml::from_str(text).map_err(|e| e.to_string())
    }

    #[test]
    fn parses_each_adapter() {
        let config = parse(
            r#"
            [client_gateway]
            adapter = "tailscale"
            [node_gateway]
            adapter = "cloudflare"
            mode = "named"
            hostname = "nodes.example.com"
            tunnel = "synthetic-tunnel"
            "#,
        )
        .unwrap();
        config.validate(true, true).unwrap();
        assert_eq!(
            config.client_gateway.as_ref().unwrap().audience(),
            Audience::Tailnet
        );
        let config = parse(
            r#"
            [client_gateway]
            adapter = "command"
            command = ["my-tunnel", "--port", "{port}"]
            "#,
        )
        .unwrap();
        config.validate(true, false).unwrap();
        assert_eq!(
            config.client_gateway.unwrap().stability(),
            OriginStability::Ephemeral
        );
    }

    #[test]
    fn unknown_fields_are_refused() {
        assert!(parse(
            "[client_gateway]\nadapter = \"manual\"\nurl = \"https://a.example.com\"\nport = 1\n"
        )
        .is_err());
        assert!(parse("[other]\nadapter = \"manual\"\n").is_err());
        assert!(parse("[client_gateway]\nadapter = \"ngrok\"\n").is_err());
    }

    #[test]
    fn node_gateway_refuses_ephemeral_adapters() {
        let config = parse("[node_gateway]\nadapter = \"cloudflare\"\nmode = \"quick\"\n").unwrap();
        let error = config.validate(false, true).unwrap_err();
        assert!(error.contains("tailscale"), "{error}");
        let config = parse("[node_gateway]\nadapter = \"command\"\ncommand = [\"x\"]\n").unwrap();
        assert!(config.validate(false, true).is_err());
        let config =
            parse("[node_gateway]\nadapter = \"command\"\ncommand = [\"x\"]\nstable = true\n")
                .unwrap();
        config.validate(false, true).unwrap();
    }

    #[test]
    fn tables_need_their_gateway_and_valid_origins() {
        let config =
            parse("[client_gateway]\nadapter = \"manual\"\nurl = \"https://a.example.com\"\n")
                .unwrap();
        assert!(config.validate(false, false).is_err());
        config.validate(true, false).unwrap();
        let config =
            parse("[client_gateway]\nadapter = \"manual\"\nurl = \"http://a.example.com/x\"\n")
                .unwrap();
        assert!(config.validate(true, false).is_err());
        let config = parse(
            "[client_gateway]\nadapter = \"cloudflare\"\nhostname = \"a.example.com/x\"\ntunnel = \"t\"\n",
        )
        .unwrap();
        assert!(config.validate(true, false).is_err());
        let shared = parse(
            "[client_gateway]\nadapter = \"cloudflare\"\nhostname = \"a.example.com\"\ntunnel = \"t\"\n\
             [node_gateway]\nadapter = \"cloudflare\"\nhostname = \"b.example.com\"\ntunnel = \"t\"\n",
        )
        .unwrap();
        assert!(shared.validate(true, true).is_err());
    }
}
