//! The Client Gateway (`/client/v1`) on the daemon: its host (sessions,
//! transcripts and slash commands from the daemon's stores) and its start.
//! Off unless `[client_gateway] enabled = true`.

use std::sync::Arc;

use aidaemon_client_gateway::{ClientGateway, ClientGatewayConfig, ClientHost, GatewayDeps};
use aidaemon_client_protocol::{HistoryMessage, SessionSummary};
use async_trait::async_trait;

use crate::channels::{ChannelHub, ChannelRuntimeDeps};
use crate::config::AppConfig;
use crate::traits::StateStore;
use crate::types::UserRole;

/// The `[client_gateway]` table of the config file, read without resolving
/// the rest (which would touch the keychain). Missing or invalid: defaults.
pub(crate) fn settings_from_file(config_path: &std::path::Path) -> ClientGatewayConfig {
    with_installed_app(
        std::fs::read_to_string(config_path)
            .ok()
            .and_then(|text| text.parse::<toml::Table>().ok())
            .and_then(|table| table.get("client_gateway").cloned())
            .and_then(|value| value.try_into().ok())
            .unwrap_or_default(),
    )
}

/// Where `install.sh` puts the app's web build.
pub(crate) fn installed_app_dir() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|home| home.join(".aidaemon").join("app-web"))
}

/// `config` serving the installed web build when it names none and one is
/// installed: an install needs no `app_dir` line.
pub(crate) fn with_installed_app(mut config: ClientGatewayConfig) -> ClientGatewayConfig {
    let unset = config
        .app_dir
        .as_deref()
        .is_none_or(|d| d.trim().is_empty());
    if unset
        && let Some(dir) = installed_app_dir()
        && dir.join("index.html").is_file()
    {
        config.app_dir = Some(dir.to_string_lossy().into_owned());
    }
    config
}

/// The local token, minting one when there is none that still works.
pub(crate) fn local_token_or_mint(config: &ClientGatewayConfig) -> anyhow::Result<String> {
    if let Some(token) = aidaemon_client_gateway::token::local_token(config) {
        return Ok(token);
    }
    let path = aidaemon_client_gateway::token::mint_local_token(config)?;
    eprintln!(
        "Minted a local Client Gateway token (saved to {}).",
        path.display()
    );
    aidaemon_client_gateway::token::local_token(config)
        .ok_or_else(|| anyhow::anyhow!("the token just minted does not verify"))
}

/// The configured `public_url`, as the origins served while no tunnel
/// publishes a live one.
pub(crate) fn static_origins(config: &ClientGatewayConfig) -> Vec<String> {
    config
        .public_url
        .iter()
        .map(|url| url.trim().trim_end_matches('/').to_string())
        .filter(|url| !url.is_empty())
        .collect()
}

/// The daemon as the gateway's host.
pub(crate) struct DaemonClientHost {
    pub state: Arc<dyn StateStore>,
    pub commands: crate::channels::commands::CommandContext,
    pub system: SystemSources,
}

/// What the app's System area reads: the daemon's models, token use, model
/// call timings, schedules, background jobs, queues and health checks.
pub(crate) struct SystemSources {
    pub started_at: std::time::Instant,
    pub models: aidaemon_providers::routing::ModelsConfig,
    pub pool: sqlx::SqlitePool,
    pub events: Arc<crate::events::EventStore>,
    pub health: Option<Arc<crate::health::HealthProbeStore>>,
    pub jobs: Option<Arc<crate::heartbeat::HeartbeatTelemetry>>,
    pub queues: Arc<crate::queue_telemetry::QueueTelemetry>,
}

/// A stored time (RFC 3339 or SQLite's `YYYY-MM-DD HH:MM:SS`, UTC) in ms.
fn stored_ms(text: &str) -> Option<u64> {
    let at = chrono::DateTime::parse_from_rfc3339(text)
        .map(|t| t.with_timezone(&chrono::Utc))
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").map(|t| t.and_utc())
        })
        .ok()?;
    u64::try_from(at.timestamp_millis()).ok()
}

impl SystemSources {
    async fn status(&self) -> aidaemon_client_protocol::SystemStatus {
        use aidaemon_client_protocol::*;
        let usage = sqlx::query_as::<_, (Option<String>, String, i64, i64, i64)>(
            "SELECT date(created_at) AS day, model, SUM(input_tokens), SUM(output_tokens), \
             COUNT(*) FROM token_usage WHERE created_at >= datetime('now', '-7 days') \
             GROUP BY day, model ORDER BY day DESC, model",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|(day, model, input, output, requests)| UsageDay {
            day: day.unwrap_or_default(),
            model,
            input_tokens: input.max(0) as u64,
            output_tokens: output.max(0) as u64,
            requests: requests.max(0) as u64,
        })
        .collect();
        let since = chrono::Utc::now() - chrono::Duration::hours(24);
        // The timings come from the latest 2000 calls; the count is all of them.
        let calls_in_window: Option<i64> = sqlx::query_scalar(
            "SELECT COUNT(*) FROM events WHERE event_type = 'llm_call' AND created_at >= ?",
        )
        .bind(since.to_rfc3339())
        .fetch_one(&self.pool)
        .await
        .ok();
        let model_calls = self
            .events
            .get_llm_stats(since)
            .await
            .ok()
            .filter(|stats| stats.total_calls > 0)
            .map(|stats| ModelCalls {
                total: calls_in_window
                    .and_then(|n| u64::try_from(n).ok())
                    .unwrap_or(stats.total_calls)
                    .max(stats.total_calls),
                failed: stats.failed_calls,
                fell_back: stats.fell_back_count,
                p50_ms: stats.p50_latency_ms,
                p95_ms: stats.p95_latency_ms,
                max_ms: stats.max_latency_ms,
            });
        type ScheduleRow = (
            String,
            String,
            String,
            String,
            i64,
            i64,
            Option<String>,
            String,
        );
        let schedules = sqlx::query_as::<_, ScheduleRow>(
            "SELECT s.goal_id, g.description, g.status, s.original_schedule, s.is_paused, \
             s.is_one_shot, s.last_run_at, s.next_run_at FROM goal_schedules s \
             JOIN goals g ON g.id = s.goal_id ORDER BY s.next_run_at ASC LIMIT 100",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(
            |(goal_id, description, status, schedule, paused, one_shot, last, next)| ScheduleInfo {
                goal_id,
                description,
                status,
                schedule,
                paused: paused != 0,
                one_shot: one_shot != 0,
                next_run_ms: stored_ms(&next),
                last_run_ms: last.as_deref().and_then(stored_ms),
            },
        )
        .collect();
        let jobs = self
            .jobs
            .as_ref()
            .map(|jobs| jobs.snapshots())
            .unwrap_or_default()
            .into_iter()
            .map(|job| JobInfo {
                name: job.name,
                every_secs: job.interval_secs,
                running: job.is_running,
                consecutive_failures: job.consecutive_failures,
                last_success_ms: job.last_success_at.as_deref().and_then(stored_ms),
                last_error: job.last_error.filter(|_| job.consecutive_failures > 0),
            })
            .collect();
        let queues = self
            .queues
            .snapshots()
            .into_iter()
            .map(|queue| QueueInfo {
                name: queue.name,
                depth: queue.current_depth as u64,
                capacity: queue.capacity as u64,
                dropped: queue.dropped_total,
                failed: queue.failed_total,
                overloaded: queue.overload_active,
            })
            .collect();
        let mut checks = Vec::new();
        if let Some(store) = &self.health {
            for probe in store.list_probes().await.unwrap_or_default() {
                let latest = store.get_latest_result(&probe.id).await.ok().flatten();
                let failures = store
                    .count_consecutive_failures(&probe.id)
                    .await
                    .unwrap_or(0);
                let status = match latest.as_ref().map(|r| &r.status) {
                    None => CheckStatus::Unknown,
                    Some(crate::health::probes::ProbeStatus::Healthy) => {
                        let slow = probe
                            .latency_threshold_ms
                            .zip(latest.as_ref().and_then(|r| r.latency_ms))
                            .is_some_and(|(limit, took)| took > limit);
                        if slow {
                            CheckStatus::Degraded
                        } else {
                            CheckStatus::Healthy
                        }
                    }
                    Some(_) => CheckStatus::Unhealthy,
                };
                checks.push(HealthCheckInfo {
                    name: probe.name,
                    target: probe.target,
                    status,
                    paused: probe.is_paused,
                    consecutive_failures: failures,
                    latency_ms: latest.as_ref().and_then(|r| r.latency_ms),
                    checked_ms: latest
                        .as_ref()
                        .and_then(|r| u64::try_from(r.checked_at.timestamp_millis()).ok()),
                    error: latest.and_then(|r| r.error_message),
                });
            }
        }
        let model = Some(self.models.default_model.clone()).filter(|m| !m.is_empty());
        SystemStatus {
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_secs: self.started_at.elapsed().as_secs(),
            model,
            fallback_models: self.models.fallback_models.clone(),
            usage,
            model_calls,
            schedules,
            jobs,
            queues,
            checks,
        }
    }
}

#[async_trait]
impl ClientHost for DaemonClientHost {
    async fn sessions(&self) -> anyhow::Result<Vec<SessionSummary>> {
        let last: std::collections::HashMap<String, String> =
            sqlx::query_as::<_, (String, String)>(
                "SELECT session_id, MAX(created_at) FROM events \
                 WHERE event_type IN ('user_message', 'assistant_response') \
                 GROUP BY session_id",
            )
            .fetch_all(&self.system.pool)
            .await
            .unwrap_or_default()
            .into_iter()
            .collect();
        Ok(self
            .state
            .load_session_channels()
            .await?
            .into_iter()
            .map(|(id, channel)| SessionSummary {
                // The app replies into chat conversations too (routed to
                // their chat by the gateway).
                writable: true,
                last_activity_ms: last.get(&id).and_then(|at| stored_ms(at)),
                id,
                channel,
                title: None,
            })
            .collect())
    }

    async fn system(&self) -> Option<aidaemon_client_protocol::SystemStatus> {
        Some(self.system.status().await)
    }

    async fn history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<HistoryMessage>> {
        let messages = self.state.get_history(session_id, limit).await?;
        Ok(messages
            .into_iter()
            .filter(|m| matches!(m.role.as_str(), "user" | "assistant"))
            .filter(|m| m.tool_calls_json.is_none())
            // Runtime continuations and system notices ride as user-role
            // turns for the model; they are not what the owner said.
            .filter(|m| {
                !m.annotations.iter().any(|a| {
                    matches!(
                        a,
                        crate::traits::MessageAnnotation::InternalContinuation
                            | crate::traits::MessageAnnotation::EntireSystemNotice
                    )
                })
            })
            .filter_map(|m| {
                let text = m.content.filter(|text| !text.trim().is_empty())?;
                Some(HistoryMessage {
                    id: m.id,
                    role: m.role,
                    text,
                    at_ms: u64::try_from(m.created_at.timestamp_millis()).unwrap_or(0),
                })
            })
            .collect())
    }

    async fn slash(&self, session_id: &str, command: &str, args: &str) -> Option<String> {
        self.commands
            .dispatch(
                &format!("/{command}"),
                args,
                session_id,
                None,
                UserRole::Owner,
            )
            .await
    }
}

/// The agent-driven view on the running gateway: `client_view` moves the
/// app connection that sent the turn's message, and only that one.
pub(crate) struct GatewayViews;

impl aidaemon_client_gateway::client_view::ClientViews for GatewayViews {
    fn can_show(&self, session_id: &str) -> bool {
        running().is_some_and(|gateway| gateway.views().can_show(session_id))
    }

    fn show(
        &self,
        session_id: &str,
        action: aidaemon_client_protocol::ViewAction,
        reason: &str,
    ) -> Result<(), String> {
        let gateway = running().ok_or_else(|| "the Client Gateway is off".to_string())?;
        gateway
            .views()
            .show(session_id, action, reason)
            .map_err(|error| error.to_string())
    }
}

/// The running gateway, for the chat's device commands; unset while it is
/// off.
static GATEWAY: std::sync::OnceLock<Arc<ClientGateway>> = std::sync::OnceLock::new();

pub(crate) fn running() -> Option<Arc<ClientGateway>> {
    GATEWAY.get().cloned()
}

/// Start the gateway when it is enabled: it sees every chat's turns and
/// approval cards from now on, and serves on its loopback port. Pairing
/// confirmations go out on `approvals`, as the agent's cards do.
pub async fn start(
    config: &AppConfig,
    pool: sqlx::SqlitePool,
    deps: &ChannelRuntimeDeps,
    hub: Arc<ChannelHub>,
    approvals: crate::tools::ApprovalBroker,
    endpoints: aidaemon_core::EndpointSource,
    system: SystemSources,
) -> anyhow::Result<()> {
    if !config.client_gateway.enabled {
        return Ok(());
    }
    let (confirm_tx, mut confirm_rx) = tokio::sync::mpsc::channel(16);
    tokio::spawn(async move {
        while let Some(request) = confirm_rx.recv().await {
            let _ = approvals.send(request).await;
        }
    });
    let log = aidaemon_store_sqlite::SqliteClientEventLog::open(pool)
        .await
        .map_err(|error| anyhow::anyhow!("client event log: {error}"))?;
    let gateway = ClientGateway::start(GatewayDeps {
        turns: aidaemon_channels::turns::TurnDeps {
            agent: deps.agent.clone(),
            session_map: deps.session_map.clone(),
            sessions: deps.state.clone(),
            task_registry: deps.task_registry.clone(),
            watchdog_stale_threshold_secs: deps.watchdog_stale_threshold_secs,
            watch: deps.watch.clone(),
        },
        hub,
        host: Arc::new(DaemonClientHost {
            state: deps.state.clone(),
            commands: deps.command_context(),
            system,
        }),
        log,
        config: with_installed_app(config.client_gateway.clone()),
        endpoints,
        token_file: aidaemon_client_gateway::token::token_file(&config.client_gateway),
        approvals: Some(confirm_tx),
    })
    .await?;
    let _ = GATEWAY.set(gateway.clone());
    tokio::spawn(async move {
        if let Err(error) = aidaemon_client_gateway::serve(gateway).await {
            tracing::error!(%error, "Client Gateway stopped");
        }
    });
    Ok(())
}

/// `/pair_device`, `/devices` and `/revoke_device <id>`: the owner's devices
/// for the Client Gateway, from any chat.
pub(crate) struct DeviceCommands;

const GATEWAY_OFF: &str = "The Client Gateway is off. Set `[client_gateway] enabled = true` in \
config.toml and restart to pair apps.";

#[async_trait]
impl aidaemon_channels::slash::SlashCommand for DeviceCommands {
    fn handles(&self, cmd: &str) -> bool {
        matches!(cmd, "/pair_device" | "/devices" | "/revoke_device")
    }

    fn commands(&self) -> Vec<aidaemon_channels::slash::CommandDef> {
        use aidaemon_channels::slash::{CommandCategory, CommandDef};
        vec![
            CommandDef {
                name: "pair_device",
                description: "Pair a phone or browser with the app",
                usage: None,
                category: CommandCategory::Core,
            },
            CommandDef {
                name: "devices",
                description: "List paired devices",
                usage: None,
                category: CommandCategory::Core,
            },
            CommandDef {
                name: "revoke_device",
                description: "Unpair a device",
                usage: Some("/revoke_device <id>"),
                category: CommandCategory::Core,
            },
        ]
    }

    async fn run(
        &self,
        cmd: &str,
        args: &str,
        call: &aidaemon_channels::slash::SlashCall<'_>,
    ) -> String {
        if call.user_role != UserRole::Owner {
            return "Only the owner can manage paired devices.".to_string();
        }
        let Some(gateway) = running() else {
            return GATEWAY_OFF.to_string();
        };
        let pairing = gateway.pairing();
        match cmd {
            "/pair_device" => match pairing.offer(Some(call.session_id.to_string())).await {
                Ok(offer) => format!(
                    "To pair a device, open {} on it and enter the code **{}** \
                     (valid 10 minutes). I'll ask you here before it can do anything.\n\n\
                     Daemon fingerprint: `{}`",
                    web_link(&gateway, &offer),
                    offer.code,
                    offer.fingerprint
                ),
                Err(error) => format!("Could not start pairing: {}", error.message),
            },
            "/devices" => match pairing.devices().await {
                Ok(devices) if devices.iter().all(|d| d.revoked) => {
                    "No paired devices. Send /pair_device to pair one.".to_string()
                }
                Ok(devices) => devices_text(&devices),
                Err(error) => format!("Could not list devices: {}", error.message),
            },
            _ => {
                let id = args.trim();
                if id.is_empty() {
                    return "Usage: /revoke_device <device id> (see /devices)".to_string();
                }
                match pairing.revoke(id).await {
                    Ok(true) => format!("Revoked `{id}`: its sessions are closed."),
                    Ok(false) => format!("No active device `{id}`."),
                    Err(error) => format!("Could not revoke: {}", error.message),
                }
            }
        }
    }
}

/// The web client's address with the offer filled in.
pub(crate) fn web_link(
    gateway: &ClientGateway,
    offer: &aidaemon_client_protocol::PairingOffer,
) -> String {
    format!(
        "{}{}#offer={}&code={}&fp={}",
        gateway.public_url(),
        gateway.web_app_path(),
        offer.offer_id,
        offer.code,
        offer.fingerprint
    )
}

fn devices_text(devices: &[aidaemon_client_protocol::DeviceInfo]) -> String {
    let mut text = "Paired devices:".to_string();
    for device in devices.iter().filter(|d| !d.revoked) {
        let scopes: Vec<String> = device
            .scopes
            .iter()
            .map(|s| format!("{s:?}").to_lowercase())
            .collect();
        text.push_str(&format!(
            "\n• {} ({:?}) `{}`: {}",
            device.name,
            device.kind,
            device.device_id,
            scopes.join(", ")
        ));
        if let Some(ceiling) = device.risk_ceiling {
            text.push_str(&format!(", approves up to {ceiling:?} risk"));
        }
    }
    text.push_str("\n\nRevoke one with /revoke_device <id>.");
    text
}
