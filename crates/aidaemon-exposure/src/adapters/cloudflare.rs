//! `cloudflare`: runs `cloudflared` as a supervised child. Named mode
//! publishes the owner's hostname through their tunnel, routing only the
//! service prefix to the loopback listener (everything else is a 404 at the
//! edge); quick mode opens a `trycloudflare.com` tunnel for testing.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;

use crate::config::CloudflareMode;
use crate::process::{self, Probe, ProcessOptions, ProcessSpec, Stream, Watcher};
use crate::{
    ExposedService, Exposure, ExposureAdapter, ExposureError, ExposureHealth, OriginStability,
    PublicEndpoint,
};

const INSTALL_HINT: &str = "install cloudflared \
     (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)";
const LOGIN_HINT: &str = "cloudflared has no credentials for this tunnel: run \
     `cloudflared tunnel login`, then `cloudflared tunnel create <name>` and \
     `cloudflared tunnel route dns <name> <hostname>`";

pub struct CloudflareAdapter {
    pub mode: CloudflareMode,
    pub hostname: String,
    pub tunnel: String,
    pub credentials_file: Option<String>,
    pub binary: Option<String>,
    pub state_dir: PathBuf,
}

/// A YAML scalar (JSON strings are valid YAML).
fn quoted(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn expand_home(path: &str) -> String {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest).display().to_string(),
        _ => path.to_string(),
    }
}

/// The generated config of a named tunnel: one hostname, only `prefix`
/// routed to `local`, a 404 for everything else.
pub(crate) fn named_config(
    tunnel: &str,
    credentials_file: Option<&str>,
    hostname: &str,
    prefix: &str,
    local: SocketAddr,
) -> String {
    let mut text = format!("tunnel: {}\n", quoted(tunnel));
    if let Some(file) = credentials_file.map(str::trim).filter(|f| !f.is_empty()) {
        text.push_str(&format!(
            "credentials-file: {}\n",
            quoted(&expand_home(file))
        ));
    }
    text.push_str(&format!(
        "ingress:\n  - hostname: {}\n    path: {}\n    service: {}\n  - service: http_status:404\n",
        quoted(hostname),
        quoted(&format!("^{prefix}(/.*)?$")),
        quoted(&format!("http://{local}")),
    ));
    text
}

/// The `https://<name>.trycloudflare.com` origin in a cloudflared log line.
pub(crate) fn quick_origin(line: &str) -> Option<String> {
    let start = line.find("https://")?;
    let candidate: String = line[start..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '/' | '.' | '-'))
        .collect();
    let host = candidate.strip_prefix("https://")?.trim_end_matches('/');
    (host.ends_with(".trycloudflare.com") && host != "api.trycloudflare.com")
        .then(|| format!("https://{host}"))
}

/// Whether cloudflared's `/ready` answer says the edge holds a
/// connection: HTTP 200 and `readyConnections` above zero.
pub(crate) fn ready_answer(status: u16, body: &str) -> bool {
    status == 200
        && serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(|v| v["readyConnections"].as_u64())
            .is_some_and(|n| n > 0)
}

/// Asks cloudflared's metrics server (`--metrics`) whether it is ready.
struct ReadyProbe {
    url: String,
    http: reqwest::Client,
}

#[async_trait]
impl Probe for ReadyProbe {
    async fn ready(&mut self) -> bool {
        let Ok(answer) = self.http.get(&self.url).send().await else {
            return false;
        };
        let status = answer.status().as_u16();
        let body = answer.text().await.unwrap_or_default();
        ready_answer(status, &body)
    }
}

/// A loopback port free now, for cloudflared's metrics server.
fn free_loopback_port() -> Result<u16, ExposureError> {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|addr| addr.port())
        .map_err(|error| ExposureError::Failed(format!("no port for tunnel metrics: {error}")))
}

/// Up once `/ready` says so.
struct NamedWatcher {
    endpoint: PublicEndpoint,
}

impl Watcher for NamedWatcher {
    fn on_line(&mut self, _stream: Stream, _line: &str) -> Option<ExposureHealth> {
        None
    }

    fn on_ready(&mut self) -> Option<ExposureHealth> {
        Some(ExposureHealth::Up(self.endpoint.clone()))
    }
}

/// A quick tunnel's origin appears only in its log (a URL, read as one);
/// it is up once `/ready` says so and the origin is known.
#[derive(Default)]
struct QuickWatcher {
    origin: Option<String>,
}

impl Watcher for QuickWatcher {
    fn on_line(&mut self, _stream: Stream, line: &str) -> Option<ExposureHealth> {
        if self.origin.is_none() {
            self.origin = quick_origin(line);
        }
        None
    }

    fn on_ready(&mut self) -> Option<ExposureHealth> {
        self.origin.clone().map(|origin| {
            ExposureHealth::Up(PublicEndpoint {
                origin,
                stability: OriginStability::Ephemeral,
            })
        })
    }
}

#[async_trait]
impl ExposureAdapter for CloudflareAdapter {
    fn id(&self) -> &str {
        "cloudflare"
    }

    fn stability(&self) -> OriginStability {
        match self.mode {
            CloudflareMode::Named => OriginStability::Stable,
            CloudflareMode::Quick => OriginStability::Ephemeral,
        }
    }

    async fn expose(
        &self,
        service: &ExposedService,
        local: SocketAddr,
    ) -> Result<Box<dyn Exposure>, ExposureError> {
        let program = process::require_program(
            self.binary.as_deref().unwrap_or("cloudflared"),
            INSTALL_HINT,
        )?;
        let prefix = service
            .path_prefix()
            .ok_or_else(|| ExposureError::Refused(format!("{service} has no path prefix")))?;
        tokio::fs::create_dir_all(&self.state_dir)
            .await
            .map_err(|error| ExposureError::Failed(format!("state dir: {error}")))?;
        let config_path = self
            .state_dir
            .join(format!("cloudflare-{}.yml", service.key()));
        if self.mode == CloudflareMode::Named
            && let Some(file) = self
                .credentials_file
                .as_deref()
                .map(str::trim)
                .filter(|f| !f.is_empty())
            && !std::path::Path::new(&expand_home(file)).is_file()
        {
            return Err(ExposureError::Failed(LOGIN_HINT.to_string()));
        }
        let metrics = format!("127.0.0.1:{}", free_loopback_port()?);
        let (config, args) = match self.mode {
            CloudflareMode::Named => (
                named_config(
                    &self.tunnel,
                    self.credentials_file.as_deref(),
                    &self.hostname,
                    prefix,
                    local,
                ),
                vec![
                    "--no-autoupdate".to_string(),
                    "tunnel".to_string(),
                    "--config".to_string(),
                    config_path.display().to_string(),
                    "--metrics".to_string(),
                    metrics.clone(),
                    "run".to_string(),
                    self.tunnel.clone(),
                ],
            ),
            // An own (empty) config, so the owner's ~/.cloudflared/config.yml
            // ingress is not picked up.
            CloudflareMode::Quick => (
                "no-autoupdate: true\n".to_string(),
                vec![
                    "--no-autoupdate".to_string(),
                    "tunnel".to_string(),
                    "--config".to_string(),
                    config_path.display().to_string(),
                    "--metrics".to_string(),
                    metrics.clone(),
                    "--url".to_string(),
                    format!("http://{local}"),
                ],
            ),
        };
        tokio::fs::write(&config_path, config)
            .await
            .map_err(|error| ExposureError::Failed(format!("tunnel config: {error}")))?;
        let spec = ProcessSpec { program, args };
        let options = ProcessOptions {
            startup_timeout: Duration::from_secs(60),
            assume_up_after: None,
            pidfile: Some(super::pidfile(&self.state_dir, "cloudflare", service)),
            cleanup: None,
            probe: Some((
                Duration::from_secs(1),
                Box::new(ReadyProbe {
                    url: format!("http://{metrics}/ready"),
                    http: reqwest::Client::builder()
                        .no_proxy()
                        .timeout(Duration::from_secs(3))
                        .build()
                        .map_err(|error| ExposureError::Failed(format!("http client: {error}")))?,
                }),
            )),
        };
        let exposure = match self.mode {
            CloudflareMode::Named => {
                let watcher = NamedWatcher {
                    endpoint: PublicEndpoint {
                        origin: format!("https://{}", self.hostname.trim()),
                        stability: OriginStability::Stable,
                    },
                };
                process::spawn(spec, watcher, options).await?
            }
            CloudflareMode::Quick => process::spawn(spec, QuickWatcher::default(), options).await?,
        };
        Ok(Box::new(exposure))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUICK_LOG: &str = include_str!("../../tests/fixtures/cloudflared-quick.golden");
    const NAMED_LOG: &str = include_str!("../../tests/fixtures/cloudflared-named.golden");

    #[test]
    fn named_config_routes_only_the_prefix() {
        let text = named_config(
            "synthetic-tunnel",
            None,
            "nodes.example.com",
            "/node/v1",
            "127.0.0.1:8787".parse().unwrap(),
        );
        assert_eq!(
            text,
            "tunnel: \"synthetic-tunnel\"\ningress:\n  - hostname: \"nodes.example.com\"\n    \
             path: \"^/node/v1(/.*)?$\"\n    service: \"http://127.0.0.1:8787\"\n  - service: \
             http_status:404\n"
        );
    }

    #[test]
    fn quick_watcher_reads_the_origin_then_waits_for_ready() {
        let mut watcher = QuickWatcher::default();
        assert_eq!(watcher.on_ready(), None, "no origin yet");
        assert!(
            QUICK_LOG
                .lines()
                .all(|line| watcher.on_line(Stream::Stderr, line).is_none()),
            "log lines never decide health"
        );
        assert_eq!(
            watcher.on_ready(),
            Some(ExposureHealth::Up(PublicEndpoint {
                origin: "https://synthetic-words-here.trycloudflare.com".into(),
                stability: OriginStability::Ephemeral,
            }))
        );
    }

    #[test]
    fn named_watcher_is_up_only_when_ready() {
        let mut watcher = NamedWatcher {
            endpoint: PublicEndpoint {
                origin: "https://nodes.example.com".into(),
                stability: OriginStability::Stable,
            },
        };
        assert!(
            NAMED_LOG
                .lines()
                .all(|line| watcher.on_line(Stream::Stderr, line).is_none())
        );
        assert!(matches!(watcher.on_ready(), Some(ExposureHealth::Up(_))));
    }

    #[test]
    fn ready_means_a_200_with_a_ready_connection() {
        assert!(ready_answer(
            200,
            r#"{"status":200,"readyConnections":4,"connectorId":"00000000-0000-0000-0000-000000000000"}"#
        ));
        assert!(!ready_answer(
            503,
            r#"{"status":503,"readyConnections":0,"connectorId":"00000000-0000-0000-0000-000000000000"}"#
        ));
        assert!(!ready_answer(200, r#"{"readyConnections":0}"#));
        assert!(!ready_answer(200, "Registered tunnel connection"));
    }

    #[test]
    fn quick_origin_ignores_other_urls() {
        assert_eq!(
            quick_origin("see https://developers.cloudflare.com/x"),
            None
        );
        assert_eq!(quick_origin("https://api.trycloudflare.com"), None);
    }
}
