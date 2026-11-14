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
use crate::process::{self, ProcessOptions, ProcessSpec, Stream, Watcher};
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

struct ServeWatcher {
    endpoint: PublicEndpoint,
}

impl Watcher for ServeWatcher {
    fn on_line(&mut self, _stream: Stream, line: &str) -> Option<ExposureHealth> {
        let lower = line.to_ascii_lowercase();
        if lower.contains("available within your tailnet")
            || lower.contains("available on the internet")
        {
            return Some(ExposureHealth::Up(self.endpoint.clone()));
        }
        if lower.contains("funnel is not enabled")
            || lower.contains("https is not enabled")
            || lower.contains("not allowed")
        {
            return Some(ExposureHealth::Down {
                reason: format!("tailscale refused: {}", line.trim()),
            });
        }
        None
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
        let spec = ProcessSpec {
            program: program.clone(),
            args: vec![self.verb().into(), port.clone(), format!("http://{local}")],
        };
        let options = ProcessOptions {
            startup_timeout: Duration::from_secs(30),
            // Some versions print nothing until a request arrives.
            assume_up_after: Some((Duration::from_secs(5), ExposureHealth::Up(endpoint.clone()))),
            pidfile: Some(super::pidfile(&self.state_dir, "tailscale", service)),
            cleanup: Some(ProcessSpec {
                program,
                args: vec![self.verb().into(), port, "off".into()],
            }),
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
    fn up_when_serve_says_available() {
        let mut watcher = ServeWatcher {
            endpoint: PublicEndpoint {
                origin: "https://synthetic-host.example-tailnet.ts.net".into(),
                stability: OriginStability::Stable,
            },
        };
        assert!(
            watcher
                .on_line(Stream::Stdout, "Press Ctrl+C to exit.")
                .is_none()
        );
        assert!(matches!(
            watcher.on_line(Stream::Stdout, "Available within your tailnet:"),
            Some(ExposureHealth::Up(_))
        ));
    }
}
