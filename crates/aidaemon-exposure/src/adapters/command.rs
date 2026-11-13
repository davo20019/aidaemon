//! `command`: any program as a tunnel, configured by hand only. Contract:
//! argv only (no shell), with `{port}`, `{addr}` and `{service}` substituted;
//! the first stdout line must be `{"url": "https://..."}` within 30 s; later
//! stdout is ignored; the child must keep running, and its exit is `Down`
//! and a supervised restart.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;

use crate::process::{self, ProcessOptions, ProcessSpec, Stream, Watcher};
use crate::{
    ExposedService, Exposure, ExposureAdapter, ExposureError, ExposureHealth, OriginStability,
    PublicEndpoint,
};

pub struct CommandAdapter {
    pub argv: Vec<String>,
    pub stable: bool,
    pub state_dir: PathBuf,
}

/// `argv` with the placeholders filled in.
pub(crate) fn substitute(
    argv: &[String],
    service: &ExposedService,
    local: SocketAddr,
) -> Vec<String> {
    argv.iter()
        .map(|arg| {
            arg.replace("{port}", &local.port().to_string())
                .replace("{addr}", &local.to_string())
                .replace("{service}", &service.key())
        })
        .collect()
}

struct FirstLine {
    stability: OriginStability,
    seen: bool,
}

#[derive(serde::Deserialize)]
struct Announcement {
    url: String,
}

impl Watcher for FirstLine {
    fn on_line(&mut self, stream: Stream, line: &str) -> Option<ExposureHealth> {
        if stream != Stream::Stdout || self.seen {
            return None;
        }
        self.seen = true;
        Some(match serde_json::from_str::<Announcement>(line.trim()) {
            Ok(announcement) => ExposureHealth::Up(PublicEndpoint {
                origin: announcement.url,
                stability: self.stability,
            }),
            Err(_) => ExposureHealth::Down {
                reason: "the command's first stdout line was not {\"url\": \"https://...\"}"
                    .to_string(),
            },
        })
    }
}

#[async_trait]
impl ExposureAdapter for CommandAdapter {
    fn id(&self) -> &str {
        "command"
    }

    fn stability(&self) -> OriginStability {
        if self.stable {
            OriginStability::Stable
        } else {
            OriginStability::Ephemeral
        }
    }

    async fn expose(
        &self,
        service: &ExposedService,
        local: SocketAddr,
    ) -> Result<Box<dyn Exposure>, ExposureError> {
        let argv = substitute(&self.argv, service, local);
        let (program, args) = argv
            .split_first()
            .ok_or_else(|| ExposureError::Refused("the command is empty".to_string()))?;
        let program = process::require_program(program, "check `command` in [exposure]")?;
        let spec = ProcessSpec {
            program,
            args: args.to_vec(),
        };
        let options = ProcessOptions {
            startup_timeout: Duration::from_secs(30),
            assume_up_after: None,
            pidfile: Some(super::pidfile(&self.state_dir, "command", service)),
            cleanup: None,
        };
        let watcher = FirstLine {
            stability: self.stability(),
            seen: false,
        };
        Ok(Box::new(process::spawn(spec, watcher, options).await?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders() {
        let argv = vec![
            "tunnel".to_string(),
            "--to={addr}".to_string(),
            "{port}".to_string(),
            "{service}".to_string(),
        ];
        assert_eq!(
            substitute(
                &argv,
                &ExposedService::NodeGateway,
                "127.0.0.1:8787".parse().unwrap()
            ),
            vec!["tunnel", "--to=127.0.0.1:8787", "8787", "node_gateway"]
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn contract_with_a_shell_fixture() {
        let dir = tempfile::tempdir().unwrap();
        let script = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/command-adapter.sh"
        );
        let adapter = CommandAdapter {
            argv: vec![
                "sh".into(),
                script.into(),
                "{port}".into(),
                "{service}".into(),
            ],
            stable: true,
            state_dir: dir.path().to_path_buf(),
        };
        let exposure = adapter
            .expose(
                &ExposedService::ClientGateway,
                "127.0.0.1:8788".parse().unwrap(),
            )
            .await
            .unwrap();
        let mut rx = exposure.watch();
        let up = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let ExposureHealth::Up(endpoint) = rx.borrow_and_update().clone() {
                    return endpoint;
                }
                rx.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        assert_eq!(up.origin, "https://client-gateway-8788.example.com");
        assert_eq!(up.stability, OriginStability::Stable);
        exposure.stop().await.unwrap();
    }

    #[test]
    fn a_first_line_that_is_not_the_announcement_is_down() {
        let mut watcher = FirstLine {
            stability: OriginStability::Ephemeral,
            seen: false,
        };
        assert!(matches!(
            watcher.on_line(Stream::Stdout, "starting..."),
            Some(ExposureHealth::Down { .. })
        ));
        assert!(watcher
            .on_line(Stream::Stdout, "{\"url\": \"https://x.example.com\"}")
            .is_none());
    }
}
