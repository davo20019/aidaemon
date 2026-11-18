//! `manual`: a reverse proxy the owner runs. The configured URL is the
//! origin; a periodic HTTPS `HEAD` to the service prefix reports whether the
//! proxy still reaches the gateway.

use std::net::SocketAddr;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::watch;

use crate::{
    ExposedService, Exposure, ExposureAdapter, ExposureError, ExposureHealth, OriginStability,
    PublicEndpoint,
};

const CHECK_EVERY: Duration = Duration::from_secs(60);
const FIRST_CHECK: Duration = Duration::from_secs(5);

pub struct ManualAdapter {
    url: String,
}

impl ManualAdapter {
    pub fn new(url: String) -> Self {
        Self { url }
    }
}

/// Whether a proxy answering with `status` reached the gateway behind it:
/// gateway errors (and Cloudflare's 530) mean it did not.
fn reached(status: u16) -> bool {
    !matches!(status, 502 | 503 | 504 | 521 | 522 | 523 | 530)
}

#[async_trait]
impl ExposureAdapter for ManualAdapter {
    fn id(&self) -> &str {
        "manual"
    }

    fn stability(&self) -> OriginStability {
        OriginStability::Stable
    }

    async fn expose(
        &self,
        service: &ExposedService,
        _local: SocketAddr,
    ) -> Result<Box<dyn Exposure>, ExposureError> {
        let endpoint = PublicEndpoint {
            origin: self.url.trim_end_matches('/').to_string(),
            stability: OriginStability::Stable,
        };
        let (tx, rx) = watch::channel(ExposureHealth::Up(endpoint.clone()));
        let probe = format!(
            "{}{}/",
            endpoint.origin,
            service.path_prefix().unwrap_or_default()
        );
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| ExposureError::Failed(format!("http client: {error}")))?;
        let task = tokio::spawn(async move {
            tokio::time::sleep(FIRST_CHECK).await;
            loop {
                let health = match client.head(&probe).send().await {
                    Ok(response) if reached(response.status().as_u16()) => {
                        ExposureHealth::Up(endpoint.clone())
                    }
                    Ok(response) => ExposureHealth::Degraded {
                        endpoint: endpoint.clone(),
                        reason: format!("the proxy answered {}", response.status()),
                    },
                    Err(error) => ExposureHealth::Degraded {
                        endpoint: endpoint.clone(),
                        reason: format!("the proxy is unreachable: {error}"),
                    },
                };
                tx.send_if_modified(|current| {
                    let changed = *current != health;
                    *current = health;
                    changed
                });
                tokio::time::sleep(CHECK_EVERY).await;
            }
        });
        Ok(Box::new(ManualExposure { rx, task }))
    }
}

struct ManualExposure {
    rx: watch::Receiver<ExposureHealth>,
    task: tokio::task::JoinHandle<()>,
}

#[async_trait]
impl Exposure for ManualExposure {
    fn watch(&self) -> watch::Receiver<ExposureHealth> {
        self.rx.clone()
    }

    async fn stop(self: Box<Self>) -> Result<(), ExposureError> {
        self.task.abort();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gateway_errors_mean_the_proxy_did_not_reach_us() {
        assert!(reached(404));
        assert!(reached(401));
        assert!(!reached(502));
        assert!(!reached(530));
    }

    #[tokio::test]
    async fn starts_up_on_the_configured_origin() {
        let adapter = ManualAdapter::new("https://app.example.com/".into());
        let exposure = adapter
            .expose(
                &ExposedService::ClientGateway,
                "127.0.0.1:8788".parse().unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            *exposure.watch().borrow(),
            ExposureHealth::Up(PublicEndpoint {
                origin: "https://app.example.com".into(),
                stability: OriginStability::Stable
            })
        );
        exposure.stop().await.unwrap();
    }
}
