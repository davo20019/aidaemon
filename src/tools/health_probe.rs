//! LLM tool for health probe management.

use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::info;

use crate::cron_utils::{compute_next_run, parse_schedule};
use crate::health::{HealthProbe, HealthProbeStore, ProbeConfig, ProbeExecutor, ProbeType};
use crate::traits::Tool;

pub struct HealthProbeTool {
    store: Arc<HealthProbeStore>,
}

impl HealthProbeTool {
    pub fn new(store: Arc<HealthProbeStore>) -> Self {
        Self { store }
    }
}

#[derive(Debug, Deserialize)]
struct HealthProbeArgs {
    action: String,
    name: Option<String>,
    #[serde(rename = "type")]
    probe_type: Option<String>,
    target: Option<String>,
    schedule: Option<String>,
    description: Option<String>,
    #[serde(default)]
    config: HealthProbeConfigArgs,
    consecutive_failures_alert: Option<u32>,
    latency_threshold_ms: Option<u32>,
    id: Option<String>,
    #[serde(default)]
    hours: Option<u32>,
}

#[derive(Debug, Deserialize, Default)]
struct HealthProbeConfigArgs {
    timeout_secs: Option<u64>,
    expected_status: Option<u16>,
    expected_body: Option<String>,
    method: Option<String>,
    max_age_secs: Option<u64>,
    expected_exit_code: Option<i32>,
}

#[async_trait]
impl Tool for HealthProbeTool {
    fn name(&self) -> &str {
        "health_probe"
    }

    fn description(&self) -> &str {
        "Create, manage, and monitor health probes for services, endpoints, and infrastructure"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "health_probe",
            "description": "Create, manage, and monitor health probes for HTTP endpoints, TCP ports, \
                commands, and files. Probes run on a schedule and alert when services go down or degrade.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "delete", "pause", "resume", "run_now", "status"],
                        "description": "The action to perform"
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable name for the probe (required for create)"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["http", "port", "command", "file"],
                        "description": "Type of probe: http (HTTP/HTTPS endpoint), port (TCP port), command (shell command), file (file existence/age)"
                    },
                    "target": {
                        "type": "string",
                        "description": "What to check. For http: URL, port: host:port, command: shell command, file: path"
                    },
                    "schedule": {
                        "type": "string",
                        "description": "When to run. Natural: 'every 1m', 'every 5m', 'hourly'. Or 5-field cron"
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of what this probe monitors"
                    },
                    "config": {
                        "type": "object",
                        "description": "Probe-specific configuration",
                        "properties": {
                            "timeout_secs": {
                                "type": "integer",
                                "description": "Timeout in seconds (default: 10)"
                            },
                            "expected_status": {
                                "type": "integer",
                                "description": "For HTTP: expected status code (default: 200)"
                            },
                            "expected_body": {
                                "type": "string",
                                "description": "For HTTP: expected substring in response body"
                            },
                            "method": {
                                "type": "string",
                                "description": "For HTTP: request method (default: GET)"
                            },
                            "max_age_secs": {
                                "type": "integer",
                                "description": "For file: max age in seconds before unhealthy"
                            },
                            "expected_exit_code": {
                                "type": "integer",
                                "description": "For command: expected exit code (default: 0)"
                            }
                        }
                    },
                    "consecutive_failures_alert": {
                        "type": "integer",
                        "description": "Number of consecutive failures before alerting (default: 3)"
                    },
                    "latency_threshold_ms": {
                        "type": "integer",
                        "description": "Latency threshold in ms to trigger warnings"
                    },
                    "id": {
                        "type": "string",
                        "description": "Probe ID (required for delete, pause, resume, run_now, status)"
                    },
                    "hours": {
                        "type": "integer",
                        "description": "For status action: hours of history to include (default: 24)"
                    }
                },
                "required": ["action"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: HealthProbeArgs = serde_json::from_str(arguments)?;

        match args.action.as_str() {
            "create" => self.create(args).await,
            "list" => self.list().await,
            "delete" => self.delete(args).await,
            "pause" => self.pause(args).await,
            "resume" => self.resume(args).await,
            "run_now" => self.run_now(args).await,
            "status" => self.status(args).await,
            other => Ok(format!(
                "Unknown action '{}'. Use: create, list, delete, pause, resume, run_now, status",
                other
            )),
        }
    }
}

impl HealthProbeTool {
    async fn create(&self, args: HealthProbeArgs) -> anyhow::Result<String> {
        let name = args.name.as_deref().unwrap_or("").trim();
        if name.is_empty() {
            return Ok("Error: 'name' is required for create".to_string());
        }

        let probe_type = args.probe_type.as_deref().unwrap_or("").trim();
        if probe_type.is_empty() {
            return Ok(
                "Error: 'type' is required for create (http, port, command, file)".to_string(),
            );
        }

        let target = args.target.as_deref().unwrap_or("").trim();
        if target.is_empty() {
            return Ok("Error: 'target' is required for create".to_string());
        }

        let schedule = args.schedule.as_deref().unwrap_or("").trim();
        if schedule.is_empty() {
            return Ok("Error: 'schedule' is required for create".to_string());
        }

        // Check if probe with this name already exists
        if self.store.get_probe_by_name(name).await?.is_some() {
            return Ok(format!(
                "Error: A probe with name '{}' already exists",
                name
            ));
        }

        // Parse and validate schedule
        let cron_expr = match parse_schedule(schedule) {
            Ok(expr) => expr,
            Err(e) => return Ok(format!("Error parsing schedule '{}': {}", schedule, e)),
        };

        let next_run = match compute_next_run(&cron_expr) {
            Ok(dt) => dt,
            Err(e) => return Ok(format!("Error computing next run: {}", e)),
        };

        // Build probe config
        let config = ProbeConfig {
            timeout_secs: args.config.timeout_secs.unwrap_or(10),
            expected_status: args.config.expected_status.or(Some(200)),
            expected_body: args.config.expected_body,
            method: args.config.method.unwrap_or_else(|| "GET".to_string()),
            headers: std::collections::HashMap::new(),
            max_age_secs: args.config.max_age_secs,
            expected_exit_code: args.config.expected_exit_code.or(Some(0)),
        };

        let now = Utc::now();
        let probe = HealthProbe {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            description: args.description,
            probe_type: ProbeType::from_str(probe_type),
            target: target.to_string(),
            schedule: cron_expr.clone(),
            source: "tool".to_string(),
            config,
            consecutive_failures_alert: args.consecutive_failures_alert.unwrap_or(3),
            latency_threshold_ms: args.latency_threshold_ms,
            alert_session_ids: Vec::new(), // Will inherit defaults from manager
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now,
            updated_at: now,
        };

        self.store.upsert_probe(&probe).await?;

        info!(
            name = %name,
            probe_type = %probe_type,
            target = %target,
            schedule = %schedule,
            "Created health probe via tool"
        );

        Ok(format!(
            "Created health probe:\n\
             • ID: {}\n\
             • Name: {}\n\
             • Type: {}\n\
             • Target: {}\n\
             • Schedule: {} (cron: {})\n\
             • Alert after: {} consecutive failures\n\
             • Next run: {}",
            probe.id,
            name,
            probe_type,
            target,
            schedule,
            cron_expr,
            probe.consecutive_failures_alert,
            next_run.format("%Y-%m-%d %H:%M:%S UTC")
        ))
    }

    async fn list(&self) -> anyhow::Result<String> {
        let probes = self.store.list_probes().await?;

        if probes.is_empty() {
            return Ok("No health probes configured.".to_string());
        }

        let mut output = format!("Health probes ({}):\n", probes.len());

        for probe in probes {
            let latest = self.store.get_latest_result(&probe.id).await?;
            let status_str = if probe.is_paused {
                "PAUSED".to_string()
            } else if let Some(ref result) = latest {
                result.status.as_str().to_uppercase()
            } else {
                "UNKNOWN".to_string()
            };

            let latency_str = latest
                .as_ref()
                .and_then(|r| r.latency_ms)
                .map(|ms| format!(" ({}ms)", ms))
                .unwrap_or_default();

            let last_check = latest
                .as_ref()
                .map(|r| r.checked_at.format("%Y-%m-%d %H:%M UTC").to_string())
                .unwrap_or_else(|| "never".to_string());

            output.push_str(&format!(
                "\n• {} [{}{}]\n\
                   ID: {}\n\
                   Type: {} → {}\n\
                   Schedule: {}\n\
                   Source: {}\n\
                   Last check: {}\n\
                   Next run: {}\n",
                probe.name,
                status_str,
                latency_str,
                probe.id,
                probe.probe_type.as_str(),
                probe.target,
                probe.schedule,
                probe.source,
                last_check,
                probe.next_run_at.format("%Y-%m-%d %H:%M UTC")
            ));
        }

        Ok(output)
    }

    async fn delete(&self, args: HealthProbeArgs) -> anyhow::Result<String> {
        let id = match args.id.as_deref() {
            Some(id) if !id.is_empty() => id,
            _ => return Ok("Error: 'id' is required for delete".to_string()),
        };

        // First check if it exists and get the name for logging
        let probe = self.store.get_probe(id).await?;
        if probe.is_none() {
            return Ok(format!("No health probe found with ID '{}'", id));
        }

        let deleted = self.store.delete_probe(id).await?;
        if deleted {
            info!(id = %id, "Deleted health probe");
            Ok(format!("Deleted health probe {}", id))
        } else {
            Ok(format!("No health probe found with ID '{}'", id))
        }
    }

    async fn pause(&self, args: HealthProbeArgs) -> anyhow::Result<String> {
        let id = match args.id.as_deref() {
            Some(id) if !id.is_empty() => id,
            _ => return Ok("Error: 'id' is required for pause".to_string()),
        };

        let paused = self.store.pause_probe(id).await?;
        if paused {
            info!(id = %id, "Paused health probe");
            Ok(format!("Paused health probe {}", id))
        } else {
            Ok(format!("No health probe found with ID '{}'", id))
        }
    }

    async fn resume(&self, args: HealthProbeArgs) -> anyhow::Result<String> {
        let id = match args.id.as_deref() {
            Some(id) if !id.is_empty() => id,
            _ => return Ok("Error: 'id' is required for resume".to_string()),
        };

        // Get the probe to recompute next run
        let probe = match self.store.get_probe(id).await? {
            Some(p) => p,
            None => return Ok(format!("No health probe found with ID '{}'", id)),
        };

        let next_run = match compute_next_run(&probe.schedule) {
            Ok(dt) => dt,
            Err(e) => return Ok(format!("Error computing next run: {}", e)),
        };

        let resumed = self.store.resume_probe(id, next_run).await?;
        if resumed {
            info!(id = %id, "Resumed health probe");
            Ok(format!(
                "Resumed health probe {}. Next run: {}",
                id,
                next_run.format("%Y-%m-%d %H:%M:%S UTC")
            ))
        } else {
            Ok(format!("No health probe found with ID '{}'", id))
        }
    }

    async fn run_now(&self, args: HealthProbeArgs) -> anyhow::Result<String> {
        let id = match args.id.as_deref() {
            Some(id) if !id.is_empty() => id,
            _ => return Ok("Error: 'id' is required for run_now".to_string()),
        };

        let probe = match self.store.get_probe(id).await? {
            Some(p) => p,
            None => return Ok(format!("No health probe found with ID '{}'", id)),
        };

        // Execute the probe
        let result = ProbeExecutor::execute(&probe).await;

        // Store the result
        self.store.insert_result(&result).await?;

        // Update last_run_at
        let now = Utc::now();
        let _ = self
            .store
            .update_probe_run(&probe.id, now, probe.next_run_at)
            .await;

        info!(
            name = %probe.name,
            status = %result.status.as_str(),
            latency_ms = ?result.latency_ms,
            "Ran health probe on demand"
        );

        let latency_str = result
            .latency_ms
            .map(|ms| format!("{}ms", ms))
            .unwrap_or_else(|| "N/A".to_string());

        let error_str = result
            .error_message
            .as_ref()
            .map(|e| format!("\nError: {}", e))
            .unwrap_or_default();

        Ok(format!(
            "Probe '{}' check result:\n\
             • Status: {}\n\
             • Latency: {}{}",
            probe.name,
            result.status.as_str().to_uppercase(),
            latency_str,
            error_str
        ))
    }

    async fn status(&self, args: HealthProbeArgs) -> anyhow::Result<String> {
        let id = match args.id.as_deref() {
            Some(id) if !id.is_empty() => id,
            _ => return Ok("Error: 'id' is required for status".to_string()),
        };

        let probe = match self.store.get_probe(id).await? {
            Some(p) => p,
            None => return Ok(format!("No health probe found with ID '{}'", id)),
        };

        let hours = args.hours.unwrap_or(24).min(168);
        let end = Utc::now();
        let start = end - chrono::Duration::hours(hours as i64);

        let stats = self.store.calculate_stats(&probe.id, start, end).await?;
        let consecutive_failures = self.store.count_consecutive_failures(&probe.id).await?;
        let latest = self.store.get_latest_result(&probe.id).await?;

        let current_status = if probe.is_paused {
            "PAUSED"
        } else if let Some(ref result) = latest {
            result.status.as_str()
        } else {
            "UNKNOWN"
        };

        let last_check = latest
            .as_ref()
            .map(|r| r.checked_at.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| "never".to_string());

        let last_latency = latest
            .as_ref()
            .and_then(|r| r.latency_ms)
            .map(|ms| format!("{}ms", ms))
            .unwrap_or_else(|| "N/A".to_string());

        let last_error = latest
            .as_ref()
            .and_then(|r| r.error_message.clone())
            .unwrap_or_else(|| "None".to_string());

        let degraded_str = if stats.is_degraded { " (DEGRADED)" } else { "" };

        Ok(format!(
            "Health probe status: {}\n\n\
             Name: {}\n\
             Type: {} → {}\n\
             Schedule: {}\n\
             Source: {}\n\n\
             Current Status: {}\n\
             Consecutive Failures: {}\n\
             Last Check: {}\n\
             Last Latency: {}\n\
             Last Error: {}\n\n\
             Stats (last {} hours):{}\n\
             • Checks: {}\n\
             • Healthy: {} ({:.1}% uptime)\n\
             • Avg Latency: {}\n\
             • P95 Latency: {}",
            probe.name,
            probe.name,
            probe.probe_type.as_str(),
            probe.target,
            probe.schedule,
            probe.source,
            current_status.to_uppercase(),
            consecutive_failures,
            last_check,
            last_latency,
            last_error,
            hours,
            degraded_str,
            stats.check_count,
            stats.healthy_count,
            stats.uptime_percent,
            stats
                .avg_latency_ms
                .map(|ms| format!("{}ms", ms))
                .unwrap_or_else(|| "N/A".to_string()),
            stats
                .p95_latency_ms
                .map(|ms| format!("{}ms", ms))
                .unwrap_or_else(|| "N/A".to_string()),
        ))
    }
}
