//! Exposure adapters: publish a loopback gateway through a tunnel the daemon
//! supervises, and feed the live HTTPS origin to the gateway.
//!
//! An adapter moves bytes, never authenticates: device keys, signed
//! challenges and scopes stay in the gateway protocols, and nothing here is
//! an input to an auth decision. Adapters receive a loopback address that is
//! a registered service listener ([`ServiceListeners`]), every origin they
//! report passes [`validate_https_origin`], and only adapters with a stable
//! origin may serve the Node Gateway.
//!
//! Tunnels live and die with the daemon: process adapters run their program
//! as a foreground child that is stopped with the daemon, never as a
//! background service.

use std::collections::BTreeMap;
use std::net::SocketAddr;

use async_trait::async_trait;
use tokio::sync::watch;

pub mod adapters;
pub mod config;
mod process;
pub mod supervisor;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use config::{Audience, CloudflareMode, ExposureConfig, ServiceExposureConfig, TailscaleMode};
pub use supervisor::{Backoff, ExposurePlan, ExposureSupervisor, ServiceState, ServiceStatus};

/// A service an adapter can publish.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExposedService {
    ClientGateway,
    NodeGateway,
    /// A channel webhook (reserved; nothing publishes one yet).
    Webhook {
        slug: String,
    },
}

impl ExposedService {
    /// The name in config, status and `{service}` placeholders.
    pub fn key(&self) -> String {
        match self {
            Self::ClientGateway => "client_gateway".to_string(),
            Self::NodeGateway => "node_gateway".to_string(),
            Self::Webhook { slug } => format!("webhook_{slug}"),
        }
    }

    /// The one path prefix the service's listener mounts.
    pub fn path_prefix(&self) -> Option<&'static str> {
        match self {
            Self::ClientGateway => Some("/client/v1"),
            Self::NodeGateway => Some("/node/v1"),
            Self::Webhook { .. } => None,
        }
    }

    /// Whether the service needs an origin that survives restarts (a Node
    /// that enrolled against it must find the gateway again).
    pub fn needs_stable_origin(&self) -> bool {
        matches!(self, Self::NodeGateway)
    }
}

impl std::fmt::Display for ExposedService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.key())
    }
}

/// Whether an origin survives a restart of the adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginStability {
    Stable,
    Ephemeral,
}

/// Where a published service is reached: an HTTPS origin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicEndpoint {
    /// `https://host[:port]`, no trailing slash once validated.
    pub origin: String,
    pub stability: OriginStability,
}

/// The state of one exposure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExposureHealth {
    Starting,
    Up(PublicEndpoint),
    /// Published, but a health check fails.
    Degraded {
        endpoint: PublicEndpoint,
        reason: String,
    },
    Down {
        reason: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum ExposureError {
    /// The adapter's program is not on `PATH`; `hint` says how to install it.
    #[error("{program} is not installed: {hint}")]
    NotInstalled { program: String, hint: String },
    /// The supervisor will not expose this (not a registered listener, not
    /// loopback, an ephemeral origin for a service that needs a stable one).
    #[error("refused: {0}")]
    Refused(String),
    #[error("{0}")]
    Failed(String),
}

/// A way to publish a loopback listener.
#[async_trait]
pub trait ExposureAdapter: Send + Sync {
    /// `"cloudflare"`, `"tailscale"`, ...
    fn id(&self) -> &str;
    /// Whether this adapter, as configured, yields an origin that survives
    /// restarts.
    fn stability(&self) -> OriginStability;
    /// Publish `local` (loopback). The handle owns this one exposure.
    async fn expose(
        &self,
        service: &ExposedService,
        local: SocketAddr,
    ) -> Result<Box<dyn Exposure>, ExposureError>;
}

/// One running exposure.
#[async_trait]
pub trait Exposure: Send + Sync {
    /// Current state; changes whenever the origin or health changes. A
    /// closed channel means the exposure ended.
    fn watch(&self) -> watch::Receiver<ExposureHealth>;
    async fn stop(self: Box<Self>) -> Result<(), ExposureError>;
}

/// The service listeners startup bound, the only addresses the supervisor
/// will expose. Each is loopback, and no address serves two services.
#[derive(Debug, Clone, Default)]
pub struct ServiceListeners {
    listeners: BTreeMap<ExposedService, SocketAddr>,
}

impl ServiceListeners {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `service`'s listener. Refused when `local` is not loopback
    /// or already serves another service.
    pub fn register(
        &mut self,
        service: ExposedService,
        local: SocketAddr,
    ) -> Result<(), ExposureError> {
        if !local.ip().is_loopback() {
            return Err(ExposureError::Refused(format!(
                "{service} listens on {local}, which is not loopback"
            )));
        }
        if local.port() == 0 {
            return Err(ExposureError::Refused(format!(
                "{service} has no bound port"
            )));
        }
        if let Some((other, _)) = self
            .listeners
            .iter()
            .find(|(other, addr)| **other != service && **addr == local)
        {
            return Err(ExposureError::Refused(format!(
                "{local} already serves {other}; one listener serves one service"
            )));
        }
        self.listeners.insert(service, local);
        Ok(())
    }

    pub fn get(&self, service: &ExposedService) -> Option<SocketAddr> {
        self.listeners.get(service).copied()
    }
}

/// Why an origin was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OriginError {
    #[error("origins must not exceed 192 bytes")]
    TooLong,
    #[error("must be a valid HTTPS origin")]
    NotUrl,
    #[error("must use HTTPS")]
    NotHttps,
    #[error("must be an HTTPS origin without credentials, paths, queries or fragments")]
    NotOrigin,
}

/// The longest origin accepted.
pub const MAX_ORIGIN_BYTES: usize = 192;

/// Check that `origin` is an HTTPS origin (`https://host[:port]`, an
/// optional trailing slash, nothing else) of at most 192 bytes, and return it
/// without the trailing slash. The one check for config origins and for what
/// adapters report.
pub fn validate_https_origin(origin: &str) -> Result<String, OriginError> {
    if origin.len() > MAX_ORIGIN_BYTES {
        return Err(OriginError::TooLong);
    }
    let parsed = url::Url::parse(origin).map_err(|_| OriginError::NotUrl)?;
    if parsed.scheme() != "https" || parsed.host_str().is_none() {
        return Err(OriginError::NotHttps);
    }
    if !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || parsed.path() != "/"
    {
        return Err(OriginError::NotOrigin);
    }
    Ok(origin.trim_end_matches('/').to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_check() {
        assert_eq!(
            validate_https_origin("https://app.example.com/").unwrap(),
            "https://app.example.com"
        );
        assert_eq!(
            validate_https_origin("https://host.example.ts.net:8443").unwrap(),
            "https://host.example.ts.net:8443"
        );
        assert_eq!(
            validate_https_origin("http://app.example.com"),
            Err(OriginError::NotHttps)
        );
        for bad in [
            "https://user:pw@app.example.com",
            "https://app.example.com/path",
            "https://app.example.com/?q=1",
            "https://app.example.com/#x",
        ] {
            assert_eq!(
                validate_https_origin(bad),
                Err(OriginError::NotOrigin),
                "{bad}"
            );
        }
        assert_eq!(validate_https_origin("not a url"), Err(OriginError::NotUrl));
        let long = format!("https://{}.example.com", "a".repeat(190));
        assert_eq!(validate_https_origin(&long), Err(OriginError::TooLong));
    }

    #[test]
    fn listeners_are_loopback_and_one_per_service() {
        let mut listeners = ServiceListeners::new();
        let local: SocketAddr = "127.0.0.1:8788".parse().unwrap();
        listeners
            .register(ExposedService::ClientGateway, local)
            .unwrap();
        assert!(
            listeners
                .register(ExposedService::NodeGateway, local)
                .is_err()
        );
        assert!(
            listeners
                .register(ExposedService::NodeGateway, "0.0.0.0:8787".parse().unwrap())
                .is_err()
        );
        listeners
            .register(
                ExposedService::NodeGateway,
                "127.0.0.1:8787".parse().unwrap(),
            )
            .unwrap();
        assert_eq!(listeners.get(&ExposedService::ClientGateway), Some(local));
    }
}
