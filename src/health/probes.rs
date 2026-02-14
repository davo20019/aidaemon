//! Health probe types and executor implementations.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::process::Command;
use tracing::{debug, warn};

/// Type of health probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeType {
    /// HTTP/HTTPS endpoint check
    Http,
    /// Shell command execution
    Command,
    /// File existence/age check
    File,
    /// TCP port connectivity
    Port,
    /// Custom probe (reserved for future extensions)
    Custom,
}

impl ProbeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProbeType::Http => "http",
            ProbeType::Command => "command",
            ProbeType::File => "file",
            ProbeType::Port => "port",
            ProbeType::Custom => "custom",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "http" | "https" => ProbeType::Http,
            "command" | "cmd" => ProbeType::Command,
            "file" => ProbeType::File,
            "port" | "tcp" => ProbeType::Port,
            _ => ProbeType::Custom,
        }
    }
}

/// Status of a probe check result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeStatus {
    Healthy,
    Unhealthy,
    Timeout,
    Error,
}

impl ProbeStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProbeStatus::Healthy => "healthy",
            ProbeStatus::Unhealthy => "unhealthy",
            ProbeStatus::Timeout => "timeout",
            ProbeStatus::Error => "error",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "healthy" | "ok" | "up" => ProbeStatus::Healthy,
            "unhealthy" | "down" | "fail" => ProbeStatus::Unhealthy,
            "timeout" => ProbeStatus::Timeout,
            _ => ProbeStatus::Error,
        }
    }

    pub fn is_healthy(&self) -> bool {
        matches!(self, ProbeStatus::Healthy)
    }
}

/// Configuration options for probe execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbeConfig {
    /// Timeout in seconds (default: 10)
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// Expected HTTP status code (default: 200)
    #[serde(default = "default_expected_status")]
    pub expected_status: Option<u16>,

    /// Expected response body substring
    pub expected_body: Option<String>,

    /// HTTP method (default: GET)
    #[serde(default = "default_http_method")]
    pub method: String,

    /// HTTP headers to include
    #[serde(default)]
    pub headers: HashMap<String, String>,

    /// For file probe: max age in seconds (file is unhealthy if older)
    pub max_age_secs: Option<u64>,

    /// For command probe: expected exit code (default: 0)
    #[serde(default)]
    pub expected_exit_code: Option<i32>,
}

fn default_timeout_secs() -> u64 {
    10
}

fn default_expected_status() -> Option<u16> {
    Some(200)
}

fn default_http_method() -> String {
    "GET".to_string()
}

/// Health probe definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthProbe {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub probe_type: ProbeType,
    pub target: String,
    pub schedule: String,
    pub source: String,
    pub config: ProbeConfig,
    pub consecutive_failures_alert: u32,
    pub latency_threshold_ms: Option<u32>,
    pub alert_session_ids: Vec<String>,
    pub is_paused: bool,
    pub last_run_at: Option<DateTime<Utc>>,
    pub next_run_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl HealthProbe {
    /// Create a new probe with defaults.
    pub fn new(
        name: String,
        probe_type: ProbeType,
        target: String,
        schedule: String,
        source: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            description: None,
            probe_type,
            target,
            schedule,
            source,
            config: ProbeConfig::default(),
            consecutive_failures_alert: 3,
            latency_threshold_ms: None,
            alert_session_ids: Vec::new(),
            is_paused: false,
            last_run_at: None,
            next_run_at: now,
            created_at: now,
            updated_at: now,
        }
    }
}

/// Result of a probe check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub id: i64,
    pub probe_id: String,
    pub status: ProbeStatus,
    pub latency_ms: Option<u32>,
    pub error_message: Option<String>,
    pub response_body: Option<String>,
    pub checked_at: DateTime<Utc>,
}

impl ProbeResult {
    pub fn new(probe_id: String, status: ProbeStatus) -> Self {
        Self {
            id: 0,
            probe_id,
            status,
            latency_ms: None,
            error_message: None,
            response_body: None,
            checked_at: Utc::now(),
        }
    }

    pub fn with_latency(mut self, latency_ms: u32) -> Self {
        self.latency_ms = Some(latency_ms);
        self
    }

    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error_message = Some(error.into());
        self
    }

    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        let body = body.into();
        // Truncate to 1KB
        self.response_body = Some(if body.len() > 1024 {
            format!("{}...", &body[..1021])
        } else {
            body
        });
        self
    }
}

/// Probe executor that runs checks based on probe type.
pub struct ProbeExecutor;

impl ProbeExecutor {
    /// Execute a probe and return the result.
    pub async fn execute(probe: &HealthProbe) -> ProbeResult {
        let start = Instant::now();
        let timeout = Duration::from_secs(probe.config.timeout_secs);

        let result = match tokio::time::timeout(timeout, Self::execute_inner(probe)).await {
            Ok(result) => result,
            Err(_) => {
                let latency = start.elapsed().as_millis() as u32;
                ProbeResult::new(probe.id.clone(), ProbeStatus::Timeout)
                    .with_latency(latency)
                    .with_error(format!(
                        "Probe timed out after {}s",
                        probe.config.timeout_secs
                    ))
            }
        };

        let latency = start.elapsed().as_millis() as u32;
        if result.latency_ms.is_none() {
            ProbeResult {
                latency_ms: Some(latency),
                ..result
            }
        } else {
            result
        }
    }

    async fn execute_inner(probe: &HealthProbe) -> ProbeResult {
        match probe.probe_type {
            ProbeType::Http => Self::execute_http(probe).await,
            ProbeType::Command => Self::execute_command(probe).await,
            ProbeType::File => Self::execute_file(probe).await,
            ProbeType::Port => Self::execute_port(probe).await,
            ProbeType::Custom => ProbeResult::new(probe.id.clone(), ProbeStatus::Error)
                .with_error("Custom probes not implemented"),
        }
    }

    /// Execute an HTTP probe.
    async fn execute_http(probe: &HealthProbe) -> ProbeResult {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(probe.config.timeout_secs))
            .build();

        let client = match client {
            Ok(c) => c,
            Err(e) => {
                return ProbeResult::new(probe.id.clone(), ProbeStatus::Error)
                    .with_error(format!("Failed to create HTTP client: {}", e));
            }
        };

        let method = probe.config.method.to_uppercase();
        let mut request = match method.as_str() {
            "GET" => client.get(&probe.target),
            "POST" => client.post(&probe.target),
            "HEAD" => client.head(&probe.target),
            "PUT" => client.put(&probe.target),
            "DELETE" => client.delete(&probe.target),
            _ => client.get(&probe.target),
        };

        // Add custom headers
        for (key, value) in &probe.config.headers {
            request = request.header(key, value);
        }

        let start = Instant::now();
        let response = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                let latency = start.elapsed().as_millis() as u32;
                let status = if e.is_timeout() {
                    ProbeStatus::Timeout
                } else if e.is_connect() {
                    ProbeStatus::Unhealthy
                } else {
                    ProbeStatus::Error
                };
                return ProbeResult::new(probe.id.clone(), status)
                    .with_latency(latency)
                    .with_error(format!("HTTP request failed: {}", e));
            }
        };

        let latency = start.elapsed().as_millis() as u32;
        let status_code = response.status().as_u16();

        // Check expected status
        let expected = probe.config.expected_status.unwrap_or(200);
        let status_ok = status_code == expected;

        // Get response body for validation
        let body = response.text().await.unwrap_or_default();

        // Check expected body
        let body_ok = match &probe.config.expected_body {
            Some(expected_body) => body.contains(expected_body),
            None => true,
        };

        let probe_status = if status_ok && body_ok {
            ProbeStatus::Healthy
        } else {
            ProbeStatus::Unhealthy
        };

        let mut result = ProbeResult::new(probe.id.clone(), probe_status)
            .with_latency(latency)
            .with_body(&body);

        if !status_ok {
            result =
                result.with_error(format!("Expected status {}, got {}", expected, status_code));
        } else if !body_ok {
            result = result.with_error(format!(
                "Response body does not contain expected text: {:?}",
                probe.config.expected_body
            ));
        }

        result
    }

    /// Execute a command probe.
    async fn execute_command(probe: &HealthProbe) -> ProbeResult {
        let start = Instant::now();

        // Parse command (simple shell execution)
        #[cfg(unix)]
        let output = Command::new("sh")
            .arg("-c")
            .arg(&probe.target)
            .output()
            .await;

        #[cfg(windows)]
        let output = Command::new("cmd.exe")
            .arg("/C")
            .arg(&probe.target)
            .output()
            .await;

        let latency = start.elapsed().as_millis() as u32;

        match output {
            Ok(output) => {
                let exit_code = output.status.code().unwrap_or(-1);
                let expected_code = probe.config.expected_exit_code.unwrap_or(0);
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                let is_healthy = exit_code == expected_code;
                let probe_status = if is_healthy {
                    ProbeStatus::Healthy
                } else {
                    ProbeStatus::Unhealthy
                };

                let combined_output = if stderr.is_empty() {
                    stdout.to_string()
                } else {
                    format!("{}\n{}", stdout, stderr)
                };

                let mut result = ProbeResult::new(probe.id.clone(), probe_status)
                    .with_latency(latency)
                    .with_body(combined_output);

                if !is_healthy {
                    result = result.with_error(format!(
                        "Command exited with code {} (expected {})",
                        exit_code, expected_code
                    ));
                }

                result
            }
            Err(e) => ProbeResult::new(probe.id.clone(), ProbeStatus::Error)
                .with_latency(latency)
                .with_error(format!("Failed to execute command: {}", e)),
        }
    }

    /// Execute a file probe.
    async fn execute_file(probe: &HealthProbe) -> ProbeResult {
        let start = Instant::now();
        let path = std::path::Path::new(&probe.target);

        // Check if file exists
        if !path.exists() {
            let latency = start.elapsed().as_millis() as u32;
            return ProbeResult::new(probe.id.clone(), ProbeStatus::Unhealthy)
                .with_latency(latency)
                .with_error(format!("File does not exist: {}", probe.target));
        }

        // Check file age if max_age_secs is specified
        if let Some(max_age) = probe.config.max_age_secs {
            match std::fs::metadata(path) {
                Ok(meta) => match meta.modified() {
                    Ok(modified) => {
                        let age = modified.elapsed().unwrap_or_default();
                        let latency = start.elapsed().as_millis() as u32;

                        if age.as_secs() > max_age {
                            return ProbeResult::new(probe.id.clone(), ProbeStatus::Unhealthy)
                                .with_latency(latency)
                                .with_error(format!(
                                    "File is too old: {} seconds (max: {})",
                                    age.as_secs(),
                                    max_age
                                ));
                        }
                    }
                    Err(e) => {
                        let latency = start.elapsed().as_millis() as u32;
                        return ProbeResult::new(probe.id.clone(), ProbeStatus::Error)
                            .with_latency(latency)
                            .with_error(format!("Cannot read file modification time: {}", e));
                    }
                },
                Err(e) => {
                    let latency = start.elapsed().as_millis() as u32;
                    return ProbeResult::new(probe.id.clone(), ProbeStatus::Error)
                        .with_latency(latency)
                        .with_error(format!("Cannot read file metadata: {}", e));
                }
            }
        }

        let latency = start.elapsed().as_millis() as u32;
        ProbeResult::new(probe.id.clone(), ProbeStatus::Healthy).with_latency(latency)
    }

    /// Execute a port probe (TCP connectivity).
    async fn execute_port(probe: &HealthProbe) -> ProbeResult {
        let start = Instant::now();

        // Parse target as host:port
        let target = &probe.target;

        // Try to connect
        match TcpStream::connect(target).await {
            Ok(mut stream) => {
                // Connection successful - try to send/receive to ensure it's truly alive
                let latency = start.elapsed().as_millis() as u32;

                // Gracefully shutdown the connection
                let _ = stream.shutdown().await;

                debug!(target = %target, latency_ms = latency, "Port probe healthy");
                ProbeResult::new(probe.id.clone(), ProbeStatus::Healthy).with_latency(latency)
            }
            Err(e) => {
                let latency = start.elapsed().as_millis() as u32;
                warn!(target = %target, error = %e, "Port probe failed");
                ProbeResult::new(probe.id.clone(), ProbeStatus::Unhealthy)
                    .with_latency(latency)
                    .with_error(format!("Connection failed: {}", e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_type_conversion() {
        assert_eq!(ProbeType::Http.as_str(), "http");
        assert_eq!(ProbeType::from_str("http"), ProbeType::Http);
        assert_eq!(ProbeType::from_str("HTTPS"), ProbeType::Http);
        assert_eq!(ProbeType::from_str("command"), ProbeType::Command);
        assert_eq!(ProbeType::from_str("tcp"), ProbeType::Port);
        assert_eq!(ProbeType::from_str("unknown"), ProbeType::Custom);
    }

    #[test]
    fn test_probe_status_conversion() {
        assert_eq!(ProbeStatus::Healthy.as_str(), "healthy");
        assert_eq!(ProbeStatus::from_str("healthy"), ProbeStatus::Healthy);
        assert_eq!(ProbeStatus::from_str("ok"), ProbeStatus::Healthy);
        assert_eq!(ProbeStatus::from_str("down"), ProbeStatus::Unhealthy);
        assert!(ProbeStatus::Healthy.is_healthy());
        assert!(!ProbeStatus::Unhealthy.is_healthy());
    }

    #[test]
    fn test_probe_result_body_truncation() {
        let long_body = "x".repeat(2000);
        let result =
            ProbeResult::new("test".to_string(), ProbeStatus::Healthy).with_body(long_body);

        assert!(result.response_body.as_ref().unwrap().len() <= 1024);
        assert!(result.response_body.as_ref().unwrap().ends_with("..."));
    }
}
