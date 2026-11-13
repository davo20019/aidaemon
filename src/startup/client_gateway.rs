//! The Client Gateway (`/client/v1`) on the daemon: its host (sessions,
//! transcripts and slash commands from the daemon's stores) and its start.
//! Off unless `[client_gateway] enabled = true`.

use std::path::PathBuf;
use std::sync::Arc;

use aidaemon_client_gateway::{ClientGateway, ClientGatewayConfig, ClientHost, GatewayDeps};
use aidaemon_client_protocol::{HistoryMessage, SessionSummary};
use async_trait::async_trait;

use crate::channels::{ChannelHub, ChannelRuntimeDeps};
use crate::config::AppConfig;
use crate::traits::StateStore;
use crate::types::UserRole;

/// Where the local client token's hash is kept.
pub(crate) fn token_file(config: &ClientGatewayConfig) -> PathBuf {
    match config.token_file.as_deref().map(str::trim) {
        Some(path) if !path.is_empty() => PathBuf::from(shellexpand::tilde(path).into_owned()),
        _ => dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".aidaemon")
            .join("client-gateway-token.sha256"),
    }
}

/// The daemon as the gateway's host.
pub(crate) struct DaemonClientHost {
    pub state: Arc<dyn StateStore>,
    pub commands: crate::channels::commands::CommandContext,
}

#[async_trait]
impl ClientHost for DaemonClientHost {
    async fn sessions(&self) -> anyhow::Result<Vec<SessionSummary>> {
        Ok(self
            .state
            .load_session_channels()
            .await?
            .into_iter()
            .map(|(id, channel)| SessionSummary {
                writable: channel == aidaemon_client_gateway::APP_CHANNEL,
                id,
                channel,
                title: None,
                last_activity_ms: None,
            })
            .collect())
    }

    async fn history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<HistoryMessage>> {
        let messages = self.state.get_history(session_id, limit).await?;
        Ok(messages
            .into_iter()
            .filter(|m| matches!(m.role.as_str(), "user" | "assistant"))
            .filter(|m| m.tool_calls_json.is_none())
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

/// Start the gateway when it is enabled: it sees every chat's turns and
/// approval cards from now on, and serves on its loopback port.
pub async fn start(
    config: &AppConfig,
    pool: sqlx::SqlitePool,
    deps: &ChannelRuntimeDeps,
    hub: Arc<ChannelHub>,
) -> anyhow::Result<()> {
    if !config.client_gateway.enabled {
        return Ok(());
    }
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
        }),
        log,
        config: config.client_gateway.clone(),
        token_file: token_file(&config.client_gateway),
    })
    .await?;
    tokio::spawn(async move {
        if let Err(error) = aidaemon_client_gateway::serve(gateway).await {
            tracing::error!(%error, "Client Gateway stopped");
        }
    });
    Ok(())
}
