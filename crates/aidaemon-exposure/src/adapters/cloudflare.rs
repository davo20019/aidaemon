//! `cloudflare`: runs `cloudflared` as a supervised child. Named mode
//! publishes the owner's hostname through their tunnel, routing only the
//! service prefix to the loopback listener (everything else is a 404 at the
//! edge); quick mode opens a `trycloudflare.com` tunnel for testing.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;

use crate::config::CloudflareMode;
use crate::process::{self, ProcessOptions, ProcessSpec, Stream, Watcher};
use crate::{
    ExposedService, Exposure, ExposureAdapter, ExposureError, ExposureHealth, OriginStability,
    PublicEndpoint,
};

const INSTALL_HINT: &str = "install cloudflared \
     (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)";
const LOGIN_HINT: &str = "cloudflared has no credentials for this tunnel: run \
     `cloudflared tunnel login`, then `cloudflared tunnel create <name>` and \
     `cloudflared tunnel route dns <name> <hostname>`";
/// What cloudflared logs once the edge accepted a connection.
const REGISTERED: &str = "Registered tunnel connection";

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

fn credentials_problem(line: &str) -> bool {
    let line = line.to_ascii_lowercase();
    line.contains("cannot determine default origin certificate")
        || line.contains("tunnel credentials file")
        || line.contains("origin cert")
            && (line.contains("not found") || line.contains("no such file"))
}

struct NamedWatcher {
    endpoint: PublicEndpoint,
}

impl Watcher for NamedWatcher {
    fn on_line(&mut self, stream: Stream, line: &str) -> Option<ExposureHealth> {
        if stream != Stream::Stderr {
            return None;
        }
        if line.contains(REGISTERED) {
            return Some(ExposureHealth::Up(self.endpoint.clone()));
        }
        credentials_problem(line).then(|| ExposureHealth::Down {
            reason: LOGIN_HINT.to_string(),
        })
    }
}

#[derive(Default)]
struct QuickWatcher {
    origin: Option<String>,
}

impl Watcher for QuickWatcher {
    fn on_line(&mut self, _stream: Stream, line: &str) -> Option<ExposureHealth> {
        if self.origin.is_none() {
            self.origin = quick_origin(line);
        }
        if line.contains(REGISTERED) {
            return self.origin.clone().map(|origin| {
                ExposureHealth::Up(PublicEndpoint {
                    origin,
                    stability: OriginStability::Ephemeral,
                })
            });
        }
        None
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
    fn quick_watcher_reads_the_origin_then_waits_for_registration() {
        let mut watcher = QuickWatcher::default();
        let mut states = QUICK_LOG
            .lines()
            .filter_map(|line| watcher.on_line(Stream::Stderr, line));
        assert_eq!(
            states.next(),
            Some(ExposureHealth::Up(PublicEndpoint {
                origin: "https://synthetic-words-here.trycloudflare.com".into(),
                stability: OriginStability::Ephemeral,
            }))
        );
    }

    #[test]
    fn named_watcher_is_up_on_registration() {
        let mut watcher = NamedWatcher {
            endpoint: PublicEndpoint {
                origin: "https://nodes.example.com".into(),
                stability: OriginStability::Stable,
            },
        };
        let states: Vec<_> = NAMED_LOG
            .lines()
            .filter_map(|line| watcher.on_line(Stream::Stderr, line))
            .collect();
        assert!(matches!(states.first(), Some(ExposureHealth::Up(_))));
        let down = watcher.on_line(
            Stream::Stderr,
            "ERR Cannot determine default origin certificate path. No file cert.pem",
        );
        assert!(matches!(down, Some(ExposureHealth::Down { .. })));
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
