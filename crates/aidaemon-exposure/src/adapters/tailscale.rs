//! `tailscale`: `tailscale serve` (tailnet only) or `tailscale funnel`
//! (public) as a supervised foreground child, never `--bg`: a background
//! rule would stay in `tailscaled` after the daemon exits and keep
//! publishing a dead port. The origin is the machine's tailnet name from
//! `tailscale status --json`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;

use crate::config::TailscaleMode;
use crate::process::{self, Probe, ProcessOptions, ProcessSpec, Stream, Watcher};
use crate::{
    ExposedService, Exposure, ExposureAdapter, ExposureError, ExposureHealth, OriginStability,
    PublicEndpoint,
};

const INSTALL_HINT: &str =
    "install Tailscale (https://tailscale.com/download) and run `tailscale up`";

pub struct TailscaleAdapter {
    pub mode: TailscaleMode,
    pub https_port: u16,
    pub binary: Option<String>,
    pub state_dir: PathBuf,
}

impl TailscaleAdapter {
    fn verb(&self) -> &'static str {
        match self.mode {
            TailscaleMode::Serve => "serve",
            TailscaleMode::Funnel => "funnel",
        }
    }
}

/// This machine's HTTPS origin on the tailnet, from `tailscale status
/// --json`; the reason when Tailscale is not running or logged in.
pub(crate) fn tailnet_origin(status_json: &str, port: u16) -> Result<String, String> {
    let status: serde_json::Value = serde_json::from_str(status_json)
        .map_err(|_| "tailscale status did not return JSON".to_string())?;
    let state = status["BackendState"].as_str().unwrap_or("");
    if state != "Running" {
        return Err(format!(
            "Tailscale is not connected (state {}): run `tailscale up`",
            if state.is_empty() { "unknown" } else { state }
        ));
    }
    let name = status["Self"]["DNSName"]
        .as_str()
        .map(|name| name.trim_end_matches('.'))
        .filter(|name| !name.is_empty())
        .ok_or("this machine has no tailnet DNS name: turn on MagicDNS and HTTPS certificates in the Tailscale admin console")?;
    Ok(if port == 443 {
        format!("https://{name}")
    } else {
        format!("https://{name}:{port}")
    })
}

/// Whether `tailscale serve status --json` holds this exposure: an HTTPS
/// listener on `port` whose web handler proxies to `local` (and, for
/// funnel, is open to the internet). A foreground `serve`/`funnel` keeps
/// its rules under `Foreground`, keyed by its session.
pub(crate) fn serve_status_has(status_json: &str, port: u16, local: &str, funnel: bool) -> bool {
    let Ok(status) = serde_json::from_str::<serde_json::Value>(status_json) else {
        return false;
    };
    let mut configs = vec![&status];
    if let Some(foreground) = status["Foreground"].as_object() {
        configs.extend(foreground.values());
    }
    let port_key = port.to_string();
    let suffix = format!(":{port}");
    configs.into_iter().any(|config| {
        config["TCP"][&port_key]["HTTPS"].as_bool() == Some(true)
            && config["Web"].as_object().is_some_and(|web| {
                web.iter().any(|(host_port, site)| {
                    host_port.ends_with(&suffix)
                        && site["Handlers"].as_object().is_some_and(|handlers| {
                            handlers.values().any(|handler| {
                                handler["Proxy"]
                                    .as_str()
                                    .is_some_and(|proxy| proxy.trim_end_matches('/') == local)
                            })
                        })
                        && (!funnel || config["AllowFunnel"][host_port].as_bool() == Some(true))
                })
            })
    })
}

/// Asks `tailscale serve status --json` whether the rule is live.
struct ServeStatusProbe {
    status: ProcessSpec,
    port: u16,
    local: String,
    funnel: bool,
}

#[async_trait]
impl Probe for ServeStatusProbe {
    async fn ready(&mut self) -> bool {
        match self.status.output(Duration::from_secs(5)).await {
            Ok(json) => serve_status_has(&json, self.port, &self.local, self.funnel),
            Err(_) => false,
        }
    }
}

/// Up once the probe finds the rule; the output is only kept for the
/// failure reason (Tailscale's own words when it refuses or waits).
struct ServeWatcher {
    endpoint: PublicEndpoint,
}

impl Watcher for ServeWatcher {
    fn on_line(&mut self, _stream: Stream, _line: &str) -> Option<ExposureHealth> {
        None
    }

    fn on_ready(&mut self) -> Option<ExposureHealth> {
        Some(ExposureHealth::Up(self.endpoint.clone()))
    }
}

#[async_trait]
impl ExposureAdapter for TailscaleAdapter {
    fn id(&self) -> &str {
        "tailscale"
    }

    fn stability(&self) -> OriginStability {
        OriginStability::Stable
    }

    async fn expose(
        &self,
        service: &ExposedService,
        local: SocketAddr,
    ) -> Result<Box<dyn Exposure>, ExposureError> {
        let program =
            process::require_program(self.binary.as_deref().unwrap_or("tailscale"), INSTALL_HINT)?;
        let status = ProcessSpec {
            program: program.clone(),
            args: vec!["status".into(), "--json".into()],
        }
        .output(Duration::from_secs(15))
        .await
        .map_err(|error| ExposureError::Failed(format!("{error}: {INSTALL_HINT}")))?;
        let origin = tailnet_origin(&status, self.https_port).map_err(ExposureError::Failed)?;
        let endpoint = PublicEndpoint {
            origin,
            stability: OriginStability::Stable,
        };
        let port = format!("--https={}", self.https_port);
        let target = format!("http://{local}");
        let spec = ProcessSpec {
            program: program.clone(),
            args: vec![self.verb().into(), port.clone(), target.clone()],
        };
        let probe = ServeStatusProbe {
            status: ProcessSpec {
                program: program.clone(),
                args: vec!["serve".into(), "status".into(), "--json".into()],
            },
            port: self.https_port,
            local: target,
            funnel: matches!(self.mode, TailscaleMode::Funnel),
        };
        let options = ProcessOptions {
            // A refused funnel waits for the owner to enable it: no rule
            // appears, and the reason is Tailscale's own last lines.
            startup_timeout: Duration::from_secs(30),
            assume_up_after: None,
            pidfile: Some(super::pidfile(&self.state_dir, "tailscale", service)),
            cleanup: Some(ProcessSpec {
                program,
                args: vec![self.verb().into(), port, "off".into()],
            }),
            probe: Some((Duration::from_secs(1), Box::new(probe))),
        };
        let exposure = process::spawn(spec, ServeWatcher { endpoint }, options).await?;
        Ok(Box::new(exposure))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATUS: &str = include_str!("../../tests/fixtures/tailscale-status.golden");

    #[test]
    fn origin_from_status() {
        assert_eq!(
            tailnet_origin(STATUS, 443).unwrap(),
            "https://synthetic-host.example-tailnet.ts.net"
        );
        assert_eq!(
            tailnet_origin(STATUS, 8443).unwrap(),
            "https://synthetic-host.example-tailnet.ts.net:8443"
        );
        let stopped = STATUS.replace("\"Running\"", "\"NeedsLogin\"");
        assert!(
            tailnet_origin(&stopped, 443)
                .unwrap_err()
                .contains("tailscale up")
        );
    }

    #[test]
    fn up_only_when_serve_status_holds_the_rule() {
        let local = "http://127.0.0.1:8787";
        let foreground = r#"{"Foreground":{"synthetic-session":{
            "TCP":{"443":{"HTTPS":true}},
            "Web":{"synthetic-host.example-tailnet.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:8787"}}}},
            "AllowFunnel":{"synthetic-host.example-tailnet.ts.net:443":true}}}}"#;
        assert!(serve_status_has(foreground, 443, local, false));
        assert!(serve_status_has(foreground, 443, local, true));
        // Another port, another target: not this exposure.
        assert!(!serve_status_has(foreground, 8443, local, false));
        assert!(!serve_status_has(
            foreground,
            443,
            "http://127.0.0.1:9999",
            false
        ));
        // Served on the tailnet but not opened to the internet: no funnel.
        let tailnet_only = foreground.replace("true}}}}", "false}}}}");
        assert!(serve_status_has(&tailnet_only, 443, local, false));
        assert!(!serve_status_has(&tailnet_only, 443, local, true));
        // Nothing served yet, or not JSON at all.
        assert!(!serve_status_has("{}", 443, local, false));
        assert!(!serve_status_has(
            "Available within your tailnet:",
            443,
            local,
            false
        ));
    }
}
