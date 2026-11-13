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

/// Where the owner's own copy of the local token is kept, next to its hash,
/// readable by the owner only, so `aidaemon tui` and `aidaemon web` need no
/// flags. Same trust as the config and the database beside it.
pub(crate) fn client_token_file(config: &ClientGatewayConfig) -> PathBuf {
    token_file(config).with_file_name("client-token")
}

/// The `[client_gateway]` table of the config file, read without resolving
/// the rest (which would touch the keychain). Missing or invalid: defaults.
pub(crate) fn settings_from_file(config_path: &std::path::Path) -> ClientGatewayConfig {
    std::fs::read_to_string(config_path)
        .ok()
        .and_then(|text| text.parse::<toml::Table>().ok())
        .and_then(|table| table.get("client_gateway").cloned())
        .and_then(|value| value.try_into().ok())
        .unwrap_or_default()
}

/// Mint a new local token (revoking the old one at once), keep its hash for
/// the gateway and the owner's copy in [`client_token_file`]. Returns the
/// copy's path; the token itself is never printed or returned.
pub(crate) fn mint_local_token(config: &ClientGatewayConfig) -> std::io::Result<PathBuf> {
    let token = aidaemon_client_gateway::token::mint(&token_file(config))?;
    let path = client_token_file(config);
    write_owner_only(&path, &format!("{token}\n"))?;
    Ok(path)
}

/// The local token: `AIDAEMON_CLIENT_TOKEN`, else the owner's copy. A copy
/// that no longer matches the gateway's hash (minted elsewhere) is `None`.
pub(crate) fn local_token(config: &ClientGatewayConfig) -> Option<String> {
    if let Ok(token) = std::env::var("AIDAEMON_CLIENT_TOKEN") {
        if !token.trim().is_empty() {
            return Some(token.trim().to_string());
        }
    }
    let token = std::fs::read_to_string(client_token_file(config)).ok()?;
    let token = token.trim().to_string();
    aidaemon_client_gateway::token::verify(&token_file(config), &token).then_some(token)
}

/// The local token, minting one when there is none that still works.
pub(crate) fn local_token_or_mint(config: &ClientGatewayConfig) -> anyhow::Result<String> {
    if let Some(token) = local_token(config) {
        return Ok(token);
    }
    let path = mint_local_token(config)?;
    eprintln!(
        "Minted a local Client Gateway token (saved to {}).",
        path.display()
    );
    local_token(config).ok_or_else(|| anyhow::anyhow!("the token just minted does not verify"))
}

fn write_owner_only(path: &std::path::Path, contents: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, contents)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(&tmp, path)
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

/// The agent-driven view on the running gateway: `client_view` moves the
/// app connection that sent the turn's message, and only that one.
pub(crate) struct GatewayViews;

impl crate::tools::client_view::ClientViews for GatewayViews {
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
        }),
        log,
        config: config.client_gateway.clone(),
        endpoints,
        token_file: token_file(&config.client_gateway),
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
        "{}/client/v1/app#offer={}&code={}&fp={}",
        gateway.public_url(),
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
