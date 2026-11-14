//! The daemon's side of the Telegram bot ([`TelegramBot`], in
//! `aidaemon-channel-telegram`): its config file, the shared slash commands
//! on its stores, its own commands (`/setkey`, `/connect`, `/bots`, `/setup`,
//! `/restart`) and the terminal surface (`aidaemon_channel_telegram::terminal`)
//! on its settings store.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock as StdRwLock, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::telegram_api::{Bot, CallbackQuery, ChatId};
pub(crate) use aidaemon_channel_telegram::bot::TelegramBot;
use aidaemon_channel_telegram::bot::{
    HostCommand, KeyChoice, TelegramHost, TelegramWebhookSettings, determine_role,
};
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tracing::{info, warn};

use super::commands::{CommandCategory, CommandDef};
#[cfg(feature = "discord")]
use crate::channels::spawn_discord_channel;
#[cfg(feature = "slack")]
use crate::channels::spawn_slack_channel;
use crate::channels::{ChannelHub, ChannelRuntimeDeps};
use crate::config::TelegramWebhookConfig;
#[cfg(test)]
use crate::runtime_ports::DaemonAgentRuntime;
#[cfg(test)]
use crate::tasks::TaskRegistry;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::traits::{HubChannel, StateStore};
use crate::types::{ApprovalResponse, UserRole};
use aidaemon_channel_telegram::terminal::TelegramTerminal;

/// A Telegram bot on the daemon's services.
pub(crate) fn new_telegram_bot(
    bot_token: &str,
    allowed_user_ids: Vec<u64>,
    webhook: TelegramWebhookConfig,
    owner_user_ids: Vec<u64>,
    runtime: ChannelRuntimeDeps,
    terminal_web_app_url: String,
    terminal_allowed_prefixes: Vec<String>,
) -> TelegramBot {
    let webhook = TelegramWebhookSettings {
        enabled: webhook.enabled,
        public_url: webhook.public_url,
        listen_addr: webhook.listen_addr,
        path: webhook.path,
        max_connections: webhook.max_connections,
        drop_pending_updates: webhook.drop_pending_updates,
    };
    let host = Arc::new(DaemonTelegramHost {
        runtime: runtime.clone(),
        terminal: TelegramTerminal::new(
            runtime.state.clone(),
            terminal_web_app_url,
            terminal_allowed_prefixes,
            crate::startup::terminal_bridge::local_agent_start(),
        ),
    });
    TelegramBot::new(
        bot_token,
        allowed_user_ids,
        webhook,
        owner_user_ids,
        runtime.bot_deps(),
        host,
    )
}

/// Spawn a Telegram bot in a background task.
/// This is a separate function to avoid async type inference cycles.
pub(crate) fn spawn_telegram_channel(channel: Arc<TelegramBot>) {
    tokio::spawn(async move {
        channel.start_with_retry().await;
    });
}

/// The daemon's services behind one Telegram bot.
pub(crate) struct DaemonTelegramHost {
    runtime: ChannelRuntimeDeps,
    /// `/agent`, `/terminal` and the Mini App handoff.
    terminal: TelegramTerminal,
}

impl DaemonTelegramHost {
    fn cx<'a>(&'a self, tg: &'a Arc<TelegramBot>) -> Cx<'a> {
        Cx {
            tg,
            bot: tg.bot().clone(),
            bot_token: tg.bot_token().to_string(),
            allowed_user_ids: tg.allowed_user_ids(),
            owner_user_ids: tg.owner_user_ids().to_vec(),
            state: self.runtime.state.clone(),
            config_path: self.runtime.config_path.clone(),
            runtime: &self.runtime,
            terminal: &self.terminal,
        }
    }

    /// Test the key against its provider and save it only if it is accepted.
    async fn install_key(&self, slot_id: &str, label: &str, key: &str) -> String {
        use super::set_key::{KeyCheck, KeySlot, check_key, key_targets, save_key};
        let slot = KeySlot::from_id(slot_id);
        let target = tokio::fs::read_to_string(&self.runtime.config_path)
            .await
            .ok()
            .and_then(|text| key_targets(&text).ok())
            .and_then(|targets| {
                targets
                    .into_iter()
                    .find(|target| Some(target.slot) == slot && target.label() == *label)
            });
        let Some(target) = target else {
            return format!("The config changed, so {label} was not updated. Run /setkey again.");
        };
        match check_key(&target, key).await {
            KeyCheck::Rejected(detail) => {
                format!("{label}: the provider rejected that key ({detail}). Nothing was changed.")
            }
            KeyCheck::Unverified(detail) => {
                format!("{label}: I couldn't verify that key ({detail}). Nothing was changed.")
            }
            KeyCheck::Accepted => match save_key(&self.runtime.config_path, target.slot, key).await
            {
                Err(error) => format!("{label}: the key works, but saving it failed: {error}"),
                Ok(stored) => {
                    let ctx = self.command_context();
                    let reload = ctx.reload_config().await.unwrap_or_else(|error| error);
                    super::set_key::key_saved_reply(&target.provider_name(), stored, &reload)
                }
            },
        }
    }

    /// The keys `/setkey` offers.
    async fn key_choices(&self) -> Result<Vec<KeyChoice>, String> {
        let targets = tokio::fs::read_to_string(&self.runtime.config_path)
            .await
            .map_err(anyhow::Error::from)
            .and_then(|text| super::set_key::key_targets(&text))
            .map_err(|error| format!("Could not read the config: {error}"))?;
        Ok(targets
            .iter()
            .map(|target| KeyChoice {
                id: target.slot.id(),
                label: target.label(),
            })
            .collect())
    }

    fn command_context(&self) -> super::commands::CommandContext {
        self.runtime.command_context()
    }
}

/// One bot and the daemon's services, for the daemon's own commands.
struct Cx<'a> {
    tg: &'a Arc<TelegramBot>,
    bot: Bot,
    bot_token: String,
    allowed_user_ids: &'a StdRwLock<Vec<u64>>,
    owner_user_ids: Vec<u64>,
    state: Arc<dyn StateStore>,
    config_path: PathBuf,
    runtime: &'a ChannelRuntimeDeps,
    terminal: &'a TelegramTerminal,
}

impl Cx<'_> {
    async fn session_id(&self, chat_id: i64) -> String {
        self.tg.session_id(chat_id).await
    }

    async fn get_bot_username(&self) -> String {
        self.tg.get_bot_username().await
    }

    async fn request_approval(
        &self,
        session_id: &str,
        command: &str,
        risk_level: RiskLevel,
        warnings: &[String],
        permission_mode: PermissionMode,
        one_time_only: bool,
    ) -> anyhow::Result<ApprovalResponse> {
        self.tg
            .request_approval(
                session_id,
                command,
                risk_level,
                warnings,
                permission_mode,
                one_time_only,
            )
            .await
    }

    fn runtime_deps(&self) -> ChannelRuntimeDeps {
        self.runtime.clone()
    }

    fn channel_hub_weak(&self) -> Option<Weak<ChannelHub>> {
        self.tg.channel_hub().map(|hub| Arc::downgrade(&hub))
    }
}

/// The daemon's commands, after the shared ones, in the "/" menu.
fn daemon_commands() -> Vec<CommandDef> {
    vec![
        CommandDef {
            name: "restart",
            description: "Restart the daemon",
            usage: None,
            category: CommandCategory::Restart,
        },
        CommandDef {
            name: "setkey",
            description: "Replace a provider's API key (works with no model available)",
            usage: Some("/setkey"),
            category: CommandCategory::Connect,
        },
        CommandDef {
            name: "connect",
            description: "Add a new bot channel",
            usage: Some("/connect <platform> <token>"),
            category: CommandCategory::Connect,
        },
        CommandDef {
            name: "setup",
            description: "Setup wizard (webhooks, low-latency)",
            usage: Some("/setup [lowlatency]"),
            category: CommandCategory::Connect,
        },
        CommandDef {
            name: "bots",
            description: "List connected bots",
            usage: None,
            category: CommandCategory::Connect,
        },
    ]
}

#[async_trait]
impl TelegramHost for DaemonTelegramHost {
    fn commands(&self) -> Vec<CommandDef> {
        let mut commands = daemon_commands();
        commands.extend(TelegramTerminal::commands());
        commands.extend(self.command_context().extension_commands());
        commands
    }

    async fn dispatch_shared(
        &self,
        cmd: &str,
        args: &str,
        session_id: &str,
        sender_id: Option<&str>,
        user_role: UserRole,
    ) -> Option<String> {
        self.command_context()
            .dispatch(cmd, args, session_id, sender_id, user_role)
            .await
    }

    async fn handle_command(
        &self,
        tg: &Arc<TelegramBot>,
        cmd: &str,
        arg: &str,
        msg: &super::telegram_api::Message,
    ) -> HostCommand {
        if self.terminal.handle_command(tg, cmd, arg, msg).await {
            return HostCommand::Handled;
        }
        let cx = self.cx(tg);
        let bot = cx.bot.clone();
        let user_id = msg.from.as_ref().map(|u| u.id.0).unwrap_or(0);
        HostCommand::Reply(match cmd {
            "/restart" => {
                let _ = bot.send_message(msg.chat.id, "Restarting...").await;
                info!("Restart requested via Telegram");
                restart_process();
                // If exec fails, we're still alive
                "Restart failed. You may need to restart manually.".to_string()
            }
            "/connect" => cx.handle_connect_command(arg, user_id).await,
            "/setup" => cx.handle_setup_command(arg, user_id, msg.chat.id.0).await,
            "/bots" => cx.handle_bots_command().await,
            _ => return HostCommand::NotMine,
        })
    }

    /// `/setup` may wait for an approval button.
    fn runs_detached(&self, cmd: &str) -> bool {
        cmd == "/setup"
    }

    async fn handle_callback(
        &self,
        tg: &Arc<TelegramBot>,
        q: &CallbackQuery,
        data: &str,
        user_id: u64,
    ) -> bool {
        self.terminal.handle_callback(tg, q, data, user_id).await
    }

    async fn key_choices(&self) -> Result<Vec<KeyChoice>, String> {
        DaemonTelegramHost::key_choices(self).await
    }

    async fn install_key(&self, id: &str, label: &str, key: &str) -> String {
        DaemonTelegramHost::install_key(self, id, label, key).await
    }

    async fn intercept_text(
        &self,
        _tg: &Arc<TelegramBot>,
        chat_id: i64,
        user_id: u64,
        user_role: UserRole,
        text: &str,
    ) -> Option<String> {
        self.terminal
            .intercept_text(chat_id, user_id, user_role, text)
            .await
    }

    async fn continue_on_computer(
        &self,
        tg: &Arc<TelegramBot>,
        msg: &super::telegram_api::Message,
        user_id: u64,
        relay_session_id: Option<String>,
    ) {
        self.terminal
            .continue_on_computer(tg, msg, user_id, relay_session_id)
            .await;
    }

    async fn owner_claimed(&self, tg: &Arc<TelegramBot>, allowed: &[u64]) -> Option<String> {
        let cx = self.cx(tg);
        let user_id = allowed.first().copied().unwrap_or(0);
        // Persist to config.toml so it survives restarts
        if let Err(e) = cx.persist_allowed_user_ids(allowed).await {
            warn!(
                user_id,
                "Failed to persist auto-claimed user ID to config: {}", e
            );
        }
        #[cfg(feature = "terminal-bridge")]
        {
            let mut bridge_hotstart_failed = false;
            match crate::config::AppConfig::load(&cx.config_path) {
                Ok(config) => {
                    if config.terminal.effective_bridge_enabled()
                        && !crate::startup::terminal_bridge::spawn_if_configured(
                            &config,
                            cx.state.clone(),
                        )
                    {
                        bridge_hotstart_failed = true;
                    }
                }
                Err(err) => {
                    warn!(
                        user_id,
                        error = %err,
                        "Failed to reload config for terminal bridge hot-start after auto-claim"
                    );
                    bridge_hotstart_failed = true;
                }
            }
            if bridge_hotstart_failed {
                return Some(
                    "I couldn't auto-enable the /agent bridge right now. If /agent doesn't open, run /restart."
                        .to_string(),
                );
            }
        }
        None
    }

    async fn on_start(&self, tg: &Arc<TelegramBot>, label: &str) {
        self.terminal.sync_bot_token(tg, Some(label)).await;
    }

    async fn set_next_message_private(
        &self,
        session_id: &str,
        sender_id: Option<&str>,
        on: bool,
    ) -> anyhow::Result<()> {
        crate::agent::memory_plugin::set_next_message_private(
            self.runtime.state.as_ref(),
            session_id,
            sender_id,
            on,
        )
        .await
    }
}

#[derive(Debug, Default)]
struct SetupLoginResult {
    success: bool,
    timed_out: bool,
    error: Option<String>,
    lines: Vec<String>,
    urls: Vec<String>,
}

use crate::wizard::CloudflaredZoneValidation;

static LOW_LATENCY_RESTART_SCHEDULED: AtomicBool = AtomicBool::new(false);

/// Replace the current process with a fresh instance of itself.
fn restart_process() {
    use std::os::unix::process::CommandExt;

    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            tracing::error!("Failed to get current exe path: {}", e);
            return;
        }
    };
    let args: Vec<String> = std::env::args().skip(1).collect();

    tracing::info!(exe = %exe.display(), "Exec-ing new process");

    // This replaces the current process — does not return on success
    let err = std::process::Command::new(&exe).args(&args).exec();
    tracing::error!("exec failed: {}", err);
}

impl Cx<'_> {
    /// Persist the current allowed_user_ids list to config.toml.
    /// Handles both `[telegram]` and `[[telegram_bots]]` config formats.
    async fn persist_allowed_user_ids(&self, ids: &[u64]) -> anyhow::Result<()> {
        let content = tokio::fs::read_to_string(&self.config_path).await?;
        let mut doc: toml::Table = content.parse()?;

        let ids_toml = toml::Value::Array(
            ids.iter()
                .map(|&id| toml::Value::Integer(id as i64))
                .collect(),
        );

        // Update whichever config format exists: [telegram] or [[telegram_bots]]
        let mut updated = false;
        if let Some(telegram) = doc.get_mut("telegram").and_then(|v| v.as_table_mut()) {
            telegram.insert("allowed_user_ids".to_string(), ids_toml.clone());
            updated = true;
        }
        if let Some(bots) = doc.get_mut("telegram_bots").and_then(|v| v.as_array_mut())
            && let Some(first) = bots.first_mut().and_then(|v| v.as_table_mut())
        {
            first.insert("allowed_user_ids".to_string(), ids_toml);
            updated = true;
        }

        if !updated {
            anyhow::bail!("No [telegram] or [[telegram_bots]] section found in config");
        }

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        tokio::fs::write(&self.config_path, &new_content).await?;
        info!("Persisted allowed_user_ids to config.toml");
        Ok(())
    }

    /// Handle /setup command for owner-only operational setup workflows.
    /// Usage:
    ///   /setup lowlatency status
    ///   /setup lowlatency reauth
    ///   /setup lowlatency plan <base-domain>
    ///   /setup lowlatency apply <base-domain>
    async fn handle_setup_command(&self, arg: &str, user_id: u64, chat_id: i64) -> String {
        if determine_role(&self.owner_user_ids, user_id) != UserRole::Owner {
            return "Only the owner can run /setup commands.".to_string();
        }

        let usage = "Setup commands (owner only)\n\n\
            Usage:\n\
            /setup lowlatency status\n\
            /setup lowlatency reauth\n\
            /setup lowlatency plan <base-domain>\n\
            /setup lowlatency apply <base-domain>\n\n\
            Examples:\n\
            /setup lowlatency status\n\
            /setup lowlatency reauth\n\
            /setup lowlatency plan bots.example.com\n\
            /setup lowlatency apply bots.example.com\n\n\
            Notes:\n\
            - `plan` is dry-run only (no config changes).\n\
            - `apply` updates local webhook config for all Telegram bots and keeps polling fallback enabled.\n\
            - If `cloudflared` or `wrangler` are missing, `apply` can ask to install them.\n\
            - If `cloudflared` or `wrangler` are not authenticated, `apply` can ask to run login flows.\n\
            - `apply` verifies your `cloudflared` zone context before DNS routing.\n\
            - `reauth` runs `cloudflared tunnel login` to refresh zone/account selection.\n\
            - `apply` asks for approval before running Cloudflare tunnel/DNS commands.\n\
            - When approved and provisioning succeeds, `apply` starts the tunnel and runs `/restart` automatically.\n\
            - If denied, the command returns manual next steps."
            .to_string();

        let mut parts: Vec<&str> = arg.split_whitespace().collect();
        if parts.is_empty() {
            return usage;
        }

        if matches!(
            parts.first().map(|v| v.to_ascii_lowercase()),
            Some(value) if value == "lowlatency" || value == "low-latency" || value == "telegram-webhook"
        ) {
            let _ = parts.remove(0);
        }
        if parts.is_empty() {
            return usage;
        }

        let action = parts[0].to_ascii_lowercase();
        let config_path = self.config_path.clone();

        match action.as_str() {
            "status" => match tokio::task::spawn_blocking({
                let config_path = config_path.clone();
                move || crate::wizard::low_latency_status_summary(&config_path)
            })
            .await
            {
                Ok(Ok(text)) => text,
                Ok(Err(err)) => format!("Setup status failed: {}", err),
                Err(err) => format!("Setup status task failed: {}", err),
            },
            "reauth" => {
                if !crate::wizard::command_exists("cloudflared").await {
                    return "Cloudflared is not installed.\nInstall it first (for example `brew install cloudflared`), then rerun `/setup lowlatency reauth`.".to_string();
                }
                let session_id = self.session_id(chat_id).await;
                let login_spec = crate::wizard::SetupCommandSpec {
                    program: "cloudflared".to_string(),
                    args: vec!["tunnel".to_string(), "login".to_string()],
                };
                let login_approval_text = format!(
                    "Run cloudflared re-authentication now?\n\n{}",
                    crate::wizard::format_setup_command(&login_spec)
                );
                let login_warnings = vec![
                    "This opens a one-time Cloudflare authorization flow.".to_string(),
                    "Use this when DNS routing is targeting the wrong zone/account.".to_string(),
                ];
                let login_approval = match self
                    .request_approval(
                        &session_id,
                        &login_approval_text,
                        RiskLevel::High,
                        &login_warnings,
                        PermissionMode::Cautious,
                        false,
                    )
                    .await
                {
                    Ok(value) => value,
                    Err(err) => {
                        return format!(
                            "Could not request approval for cloudflared re-authentication: {}\nRun `cloudflared tunnel login` manually if needed.",
                            err
                        );
                    }
                };
                if matches!(login_approval, ApprovalResponse::Deny) {
                    return "Cloudflared re-authentication skipped (approval denied).".to_string();
                }
                match self.run_cloudflared_login_and_verify(chat_id).await {
                    Ok(report) => format!("Cloudflared re-authentication completed.\n{}", report),
                    Err(err) => format!(
                        "{}\nRun `cloudflared tunnel login` manually, ensure it succeeds, then rerun your setup command.",
                        err
                    ),
                }
            }
            "plan" => {
                let Some(domain) = parts.get(1).map(|v| v.to_string()) else {
                    return "Usage: /setup lowlatency plan <base-domain>\nExample: /setup lowlatency plan bots.example.com".to_string();
                };
                let dynamic_bots = self.state.get_dynamic_bots().await.unwrap_or_default();
                let config_path = config_path.clone();
                match tokio::task::spawn_blocking(move || {
                    crate::wizard::low_latency_plan_from_base_domain_with_dynamic(
                        &config_path,
                        &domain,
                        &dynamic_bots,
                    )
                })
                .await
                {
                    Ok(Ok(text)) => text,
                    Ok(Err(err)) => format!("Setup plan failed: {}", err),
                    Err(err) => format!("Setup plan task failed: {}", err),
                }
            }
            "apply" => {
                let Some(domain) = parts.get(1).map(|v| v.to_string()) else {
                    return "Usage: /setup lowlatency apply <base-domain>\nExample: /setup lowlatency apply bots.example.com".to_string();
                };
                let dynamic_bots = self.state.get_dynamic_bots().await.unwrap_or_default();
                let apply_text = match tokio::task::spawn_blocking({
                    let config_path = config_path.clone();
                    let domain = domain.clone();
                    let dynamic_bots = dynamic_bots.clone();
                    move || {
                        crate::wizard::low_latency_apply_from_base_domain_with_dynamic(
                            &config_path,
                            &domain,
                            &dynamic_bots,
                        )
                    }
                })
                .await
                {
                    Ok(Ok(text)) => text,
                    Ok(Err(err)) => return format!("Setup apply failed: {}", err),
                    Err(err) => return format!("Setup apply task failed: {}", err),
                };

                let command_specs = match tokio::task::spawn_blocking({
                    let config_path = config_path.clone();
                    let domain = domain.clone();
                    let dynamic_bots = dynamic_bots.clone();
                    move || {
                        crate::wizard::low_latency_cloudflared_commands_from_base_domain_with_dynamic(
                            &config_path,
                            &domain,
                            &dynamic_bots,
                        )
                    }
                })
                .await
                {
                    Ok(Ok(cmds)) => cmds,
                    Ok(Err(err)) => {
                        return format!(
                            "{}\n\nCould not prepare Cloudflare commands: {}\nRun manual next steps above, then /restart to apply webhook listeners.",
                            apply_text, err
                        )
                    }
                    Err(err) => {
                        return format!(
                            "{}\n\nSetup command preparation failed: {}\nRun manual next steps above, then /restart to apply webhook listeners.",
                            apply_text, err
                        )
                    }
                };

                if command_specs.is_empty() {
                    return format!(
                        "{}\n\nNo Cloudflare commands were generated.\nRun manual next steps above, then /restart to apply webhook listeners.",
                        apply_text
                    );
                }

                let session_id = self.session_id(chat_id).await;
                let has_brew = crate::wizard::command_exists("brew").await;
                let has_npm = crate::wizard::command_exists("npm").await;
                let mut install_specs: Vec<(String, crate::wizard::SetupCommandSpec)> = Vec::new();
                let cloudflared_present = crate::wizard::command_exists("cloudflared").await;
                if !cloudflared_present {
                    if let Some(spec) =
                        crate::wizard::installer_for_missing_tool("cloudflared", has_brew, has_npm)
                    {
                        install_specs.push(("cloudflared".to_string(), spec));
                    } else {
                        return format!(
                            "{}\n\n`cloudflared` is not installed and no supported automatic installer was detected.\nInstall it manually (for example `brew install cloudflared`), then rerun /setup lowlatency apply <base-domain>.",
                            apply_text
                        );
                    }
                }
                let wrangler_present = crate::wizard::command_exists("wrangler").await;
                if !wrangler_present
                    && let Some(spec) =
                        crate::wizard::installer_for_missing_tool("wrangler", has_brew, has_npm)
                {
                    install_specs.push(("wrangler".to_string(), spec));
                }

                let mut install_report = String::new();
                if !install_specs.is_empty() {
                    let install_preview = install_specs
                        .iter()
                        .map(|(tool, spec)| {
                            format!("{}: {}", tool, crate::wizard::format_setup_command(spec))
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let install_approval_text = format!(
                        "Install missing setup dependencies now?\n\n{}",
                        install_preview
                    );
                    let install_warnings = vec![
                        "This will install CLI tools on your host.".to_string(),
                        "Deny to keep manual setup.".to_string(),
                    ];
                    let install_approval = match self
                        .request_approval(
                            &session_id,
                            &install_approval_text,
                            RiskLevel::High,
                            &install_warnings,
                            PermissionMode::Cautious,
                            false,
                        )
                        .await
                    {
                        Ok(value) => value,
                        Err(err) => {
                            return format!(
                                "{}\n\nCould not request approval for dependency installation: {}\nRun manual next steps above, then /restart to apply webhook listeners.",
                                apply_text, err
                            );
                        }
                    };

                    if matches!(install_approval, ApprovalResponse::Deny) {
                        if !cloudflared_present {
                            return format!(
                                "{}\n\nAutomatic setup skipped because install approval was denied and `cloudflared` is required.\nInstall `cloudflared` manually, then rerun /setup lowlatency apply <base-domain>.",
                                apply_text
                            );
                        }
                        install_report =
                            "Dependency installation skipped (approval denied).".to_string();
                    } else {
                        let specs: Vec<crate::wizard::SetupCommandSpec> =
                            install_specs.iter().map(|(_, spec)| spec.clone()).collect();
                        let (lines, had_failures) =
                            Self::run_setup_commands(&specs, 600, false).await;
                        install_report =
                            format!("Dependency installation results:\n{}", lines.join("\n"));
                        if had_failures {
                            let cloudflared_after =
                                crate::wizard::command_exists("cloudflared").await;
                            if !cloudflared_after {
                                return format!(
                                    "{}\n\n{}\n\n`cloudflared` is still missing, so automatic provisioning cannot continue.\nInstall it manually and rerun /setup lowlatency apply <base-domain>.",
                                    apply_text, install_report
                                );
                            }
                        }
                    }
                }

                let mut cloudflared_auth_report = String::new();
                let mut wrangler_auth_report = String::new();
                let cloudflared_available = crate::wizard::command_exists("cloudflared").await;
                if cloudflared_available
                    && !crate::wizard::command_succeeds("cloudflared", &["tunnel", "list"], 20)
                        .await
                {
                    let login_spec = crate::wizard::SetupCommandSpec {
                        program: "cloudflared".to_string(),
                        args: vec!["tunnel".to_string(), "login".to_string()],
                    };
                    let login_preview = crate::wizard::format_setup_command(&login_spec);
                    let login_approval_text = format!(
                        "Cloudflared is not authenticated. Run login now?\n\n{}",
                        login_preview
                    );
                    let login_warnings = vec![
                        "This opens a one-time Cloudflare authorization flow.".to_string(),
                        "You may need to open a URL from your browser/phone and approve access."
                            .to_string(),
                    ];
                    let login_approval = match self
                        .request_approval(
                            &session_id,
                            &login_approval_text,
                            RiskLevel::High,
                            &login_warnings,
                            PermissionMode::Cautious,
                            false,
                        )
                        .await
                    {
                        Ok(value) => value,
                        Err(err) => {
                            return format!(
                                "{}\n\nCould not request approval for cloudflared authentication: {}\nRun `cloudflared tunnel login` manually, then rerun /setup lowlatency apply <base-domain>.",
                                apply_text, err
                            );
                        }
                    };
                    if matches!(login_approval, ApprovalResponse::Deny) {
                        return format!(
                            "{}\n\nAutomatic setup skipped because cloudflared authentication was denied.\nRun `cloudflared tunnel login` manually, then rerun /setup lowlatency apply <base-domain>.",
                            apply_text
                        );
                    }
                    match self.run_cloudflared_login_and_verify(chat_id).await {
                        Ok(report) => {
                            cloudflared_auth_report = report;
                        }
                        Err(err) => {
                            return format!(
                                "{}\n\n{}\nRun `cloudflared tunnel login` manually, ensure it succeeds, then rerun /setup lowlatency apply <base-domain>.",
                                apply_text, err
                            );
                        }
                    }
                }

                let route_hosts = crate::wizard::setup_route_dns_hosts(&command_specs);
                match crate::wizard::validate_cloudflared_zone_for_hosts(&route_hosts).await {
                    CloudflaredZoneValidation::Match { zone_name } => {
                        let note = format!("Cloudflared zone context verified: `{}`.", zone_name);
                        if cloudflared_auth_report.is_empty() {
                            cloudflared_auth_report = note;
                        } else {
                            cloudflared_auth_report =
                                format!("{}\n{}", cloudflared_auth_report, note);
                        }
                    }
                    CloudflaredZoneValidation::Unknown { reason } => {
                        if !reason.is_empty() {
                            let note = format!("Cloudflared zone preflight: {}.", reason);
                            if cloudflared_auth_report.is_empty() {
                                cloudflared_auth_report = note;
                            } else {
                                cloudflared_auth_report =
                                    format!("{}\n{}", cloudflared_auth_report, note);
                            }
                        }
                    }
                    CloudflaredZoneValidation::Mismatch {
                        zone_name,
                        mismatched_hosts,
                    } => {
                        let sample = mismatched_hosts
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "unknown".to_string());
                        let login_spec = crate::wizard::SetupCommandSpec {
                            program: "cloudflared".to_string(),
                            args: vec!["tunnel".to_string(), "login".to_string()],
                        };
                        let reauth_prompt = format!(
                            "Cloudflared is authenticated for zone `{}`, but planned webhook host `{}` is outside that zone.\nRe-authenticate now to select the correct zone/account?\n\n{}",
                            zone_name,
                            sample,
                            crate::wizard::format_setup_command(&login_spec)
                        );
                        let warnings = vec![
                            "Without re-auth, DNS records can be created in the wrong zone."
                                .to_string(),
                            "Deny to abort automatic provisioning safely.".to_string(),
                        ];
                        let approval = match self
                            .request_approval(
                                &session_id,
                                &reauth_prompt,
                                RiskLevel::High,
                                &warnings,
                                PermissionMode::Cautious,
                                false,
                            )
                            .await
                        {
                            Ok(value) => value,
                            Err(err) => {
                                return format!(
                                    "{}\n\nCloudflared zone mismatch detected: current zone `{}` does not include planned host(s): {}.\nCould not request re-auth approval: {}\nAutomatic provisioning stopped to prevent wrong DNS records.\nRun `cloudflared tunnel login` with the correct account/zone, then rerun /setup lowlatency apply <base-domain>.",
                                    apply_text,
                                    zone_name,
                                    mismatched_hosts.join(", "),
                                    err
                                );
                            }
                        };
                        if matches!(approval, ApprovalResponse::Deny) {
                            return format!(
                                "{}\n\nCloudflared zone mismatch detected: current zone `{}` does not include planned host(s): {}.\nAutomatic provisioning stopped to prevent wrong DNS records.\nRun `cloudflared tunnel login` with the correct account/zone, then rerun /setup lowlatency apply <base-domain>.",
                                apply_text,
                                zone_name,
                                mismatched_hosts.join(", ")
                            );
                        }
                        let reauth_report = match self
                            .run_cloudflared_login_and_verify(chat_id)
                            .await
                        {
                            Ok(report) => report,
                            Err(err) => {
                                return format!(
                                    "{}\n\n{}\nRun `cloudflared tunnel login` manually with the correct account/zone, then rerun /setup lowlatency apply <base-domain>.",
                                    apply_text, err
                                );
                            }
                        };
                        match crate::wizard::validate_cloudflared_zone_for_hosts(&route_hosts).await
                        {
                            CloudflaredZoneValidation::Match {
                                zone_name: refreshed_zone,
                            } => {
                                let note = format!(
                                    "Cloudflared re-authentication completed.\n{}\nCloudflared zone context verified: `{}`.",
                                    reauth_report, refreshed_zone
                                );
                                if cloudflared_auth_report.is_empty() {
                                    cloudflared_auth_report = note;
                                } else {
                                    cloudflared_auth_report =
                                        format!("{}\n{}", cloudflared_auth_report, note);
                                }
                            }
                            CloudflaredZoneValidation::Mismatch {
                                zone_name: refreshed_zone,
                                mismatched_hosts: refreshed_hosts,
                            } => {
                                return format!(
                                    "{}\n\nCloudflared re-authentication completed but zone mismatch remains.\nCurrent zone: `{}`\nPlanned host(s): {}\nAutomatic provisioning stopped to prevent wrong DNS records.\nRun `cloudflared tunnel login` again with the correct account/zone, then rerun /setup lowlatency apply <base-domain>.",
                                    apply_text,
                                    refreshed_zone,
                                    refreshed_hosts.join(", ")
                                );
                            }
                            CloudflaredZoneValidation::Unknown { reason } => {
                                let note = if reason.is_empty() {
                                    format!(
                                        "Cloudflared re-authentication completed.\n{}",
                                        reauth_report
                                    )
                                } else {
                                    format!(
                                        "Cloudflared re-authentication completed.\n{}\nCloudflared zone preflight after re-auth: {}.",
                                        reauth_report, reason
                                    )
                                };
                                if cloudflared_auth_report.is_empty() {
                                    cloudflared_auth_report = note;
                                } else {
                                    cloudflared_auth_report =
                                        format!("{}\n{}", cloudflared_auth_report, note);
                                }
                            }
                        }
                    }
                }

                let wrangler_available = crate::wizard::command_exists("wrangler").await;
                if wrangler_available
                    && !crate::wizard::command_succeeds("wrangler", &["whoami", "--json"], 20).await
                {
                    let login_spec = crate::wizard::SetupCommandSpec {
                        program: "wrangler".to_string(),
                        args: vec!["login".to_string()],
                    };
                    let login_preview = crate::wizard::format_setup_command(&login_spec);
                    let login_approval_text = format!(
                        "Wrangler is not authenticated. Run login now?\n\n{}",
                        login_preview
                    );
                    let login_warnings = vec![
                        "This opens a one-time Cloudflare authorization flow for wrangler."
                            .to_string(),
                        "You may need to open a URL from your browser/phone and approve access."
                            .to_string(),
                    ];
                    let login_approval = match self
                        .request_approval(
                            &session_id,
                            &login_approval_text,
                            RiskLevel::High,
                            &login_warnings,
                            PermissionMode::Cautious,
                            false,
                        )
                        .await
                    {
                        Ok(value) => value,
                        Err(err) => {
                            wrangler_auth_report = format!(
                                "Wrangler authentication was skipped (approval request failed: {}).",
                                err
                            );
                            ApprovalResponse::Deny
                        }
                    };

                    if matches!(login_approval, ApprovalResponse::Deny) {
                        if wrangler_auth_report.is_empty() {
                            wrangler_auth_report =
                                "Wrangler authentication skipped (approval denied).".to_string();
                        }
                    } else {
                        let login_result = self
                            .run_login_command_with_live_output(
                                chat_id,
                                &login_spec,
                                600,
                                "Starting wrangler authentication flow.",
                            )
                            .await;
                        if login_result.timed_out {
                            wrangler_auth_report =
                                "Wrangler login timed out after 600s; finish `wrangler login` manually if you need wrangler actions."
                                    .to_string();
                        } else if let Some(err) = login_result.error {
                            wrangler_auth_report = format!(
                                "Wrangler login failed: {}. Run `wrangler login` manually if needed.",
                                err
                            );
                        } else {
                            let wrangler_ok = login_result.success
                                && crate::wizard::command_succeeds(
                                    "wrangler",
                                    &["whoami", "--json"],
                                    20,
                                )
                                .await;
                            let summary =
                                crate::wizard::summarize_setup_log_lines(&login_result.lines);
                            if wrangler_ok {
                                wrangler_auth_report = if let Some(url) = login_result.urls.first()
                                {
                                    if summary.is_empty() {
                                        format!("Wrangler authentication completed.\nURL: {}", url)
                                    } else {
                                        format!(
                                            "Wrangler authentication completed.\nURL: {}\n{}",
                                            url, summary
                                        )
                                    }
                                } else if summary.is_empty() {
                                    "Wrangler authentication completed.".to_string()
                                } else {
                                    format!("Wrangler authentication completed.\n{}", summary)
                                };
                            } else {
                                wrangler_auth_report = if summary.is_empty() {
                                    "Wrangler authentication did not complete; run `wrangler login` manually if needed.".to_string()
                                } else {
                                    format!(
                                        "Wrangler authentication did not complete.\n{}\nRun `wrangler login` manually if needed.",
                                        summary
                                    )
                                };
                            }
                        }
                    }
                }

                let command_preview = command_specs
                    .iter()
                    .map(crate::wizard::format_setup_command)
                    .collect::<Vec<_>>()
                    .join("\n");
                let approval_command = format!(
                    "Run Cloudflare provisioning commands now for `{}`?\n\n{}",
                    domain, command_preview
                );
                let warnings = vec![
                    "This will execute cloudflared commands on your host.".to_string(),
                    "Deny to keep the manual setup flow.".to_string(),
                ];
                let approval = match self
                    .request_approval(
                        &session_id,
                        &approval_command,
                        RiskLevel::High,
                        &warnings,
                        PermissionMode::Cautious,
                        false,
                    )
                    .await
                {
                    Ok(value) => value,
                    Err(err) => {
                        return format!(
                            "{}\n\nCould not request approval for automatic Cloudflare setup: {}\nRun manual next steps above, then /restart to apply webhook listeners.",
                            apply_text, err
                        );
                    }
                };
                if matches!(approval, ApprovalResponse::Deny) {
                    let mut out = apply_text;
                    if !install_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&install_report);
                    }
                    if !cloudflared_auth_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&cloudflared_auth_report);
                    }
                    if !wrangler_auth_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&wrangler_auth_report);
                    }
                    out.push_str(
                        "\n\nAutomatic Cloudflare setup skipped (approval denied).\nRun manual next steps above, then /restart to apply webhook listeners.",
                    );
                    out
                } else {
                    let (result_lines, had_failures) =
                        Self::run_setup_commands(&command_specs, 60, true).await;
                    let mut out = apply_text.clone();
                    let mut automatic_tunnel_startup_ok = false;
                    let mut automatic_tunnel_startup_failed = false;
                    if !install_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&install_report);
                    }
                    if !cloudflared_auth_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&cloudflared_auth_report);
                    }
                    if !wrangler_auth_report.is_empty() {
                        out.push_str("\n\n");
                        out.push_str(&wrangler_auth_report);
                    }
                    if had_failures {
                        out.push_str(
                            "\n\nAutomatic Cloudflare provisioning finished with failures:\n",
                        );
                        out.push_str(&result_lines.join("\n"));
                        out.push_str(
                            "\n\nRun the failed command(s) manually, then start your tunnel process and run /restart to apply webhook listeners.",
                        );
                    } else {
                        out.push_str("\n\nAutomatic Cloudflare provisioning completed:\n");
                        out.push_str(&result_lines.join("\n"));
                        let ingress_routes = match tokio::task::spawn_blocking({
                            let config_path = config_path.clone();
                            let domain = domain.clone();
                            let dynamic_bots = dynamic_bots.clone();
                            move || {
                                crate::wizard::low_latency_cloudflared_ingress_routes_from_base_domain_with_dynamic(
                                    &config_path,
                                    &domain,
                                    &dynamic_bots,
                                )
                            }
                        })
                        .await
                        {
                            Ok(Ok(routes)) => routes,
                            Ok(Err(err)) => {
                                automatic_tunnel_startup_failed = true;
                                out.push_str(
                                    "\n\nAutomatic tunnel startup failed:\n",
                                );
                                out.push_str(&format!(
                                    "Could not derive ingress routes from config: {}",
                                    err
                                ));
                                out.push_str(
                                    "\nRun `cloudflared tunnel run aidaemon-telegram` manually after reviewing your tunnel config.",
                                );
                                Vec::new()
                            }
                            Err(err) => {
                                automatic_tunnel_startup_failed = true;
                                out.push_str(
                                    "\n\nAutomatic tunnel startup failed:\n",
                                );
                                out.push_str(&format!(
                                    "Ingress route task failed: {}",
                                    err
                                ));
                                out.push_str(
                                    "\nRun `cloudflared tunnel run aidaemon-telegram` manually after reviewing your tunnel config.",
                                );
                                Vec::new()
                            }
                        };

                        if !ingress_routes.is_empty() {
                            match crate::wizard::start_cloudflared_tunnel_background(
                                "aidaemon-telegram",
                                &ingress_routes,
                            )
                            .await
                            {
                                Ok(tunnel_status) => {
                                    automatic_tunnel_startup_ok = true;
                                    out.push_str("\n\nAutomatic tunnel startup completed:\n");
                                    out.push_str(&tunnel_status);
                                }
                                Err(err) => {
                                    automatic_tunnel_startup_failed = true;
                                    out.push_str("\n\nAutomatic tunnel startup failed:\n");
                                    out.push_str(&format!(
                                        "FAIL cloudflared tunnel run aidaemon-telegram - {}",
                                        err
                                    ));
                                    out.push_str(
                                        "\nRun `cloudflared tunnel run aidaemon-telegram` manually after reviewing your tunnel config.",
                                    );
                                }
                            }
                        }

                        let restart_scheduled =
                            Self::schedule_low_latency_restart(Duration::from_secs(3));

                        // When the full automation succeeds end-to-end, replace
                        // verbose output with a concise success summary.
                        if automatic_tunnel_startup_ok && !automatic_tunnel_startup_failed {
                            // Extract public_url lines from the original apply text
                            let webhook_urls: Vec<&str> = apply_text
                                .lines()
                                .filter(|l| l.trim_start().starts_with("public_url:"))
                                .map(|l| l.trim())
                                .collect();
                            out = String::new();
                            out.push_str("Webhook mode setup complete.\n");
                            if !webhook_urls.is_empty() {
                                out.push('\n');
                                for url in &webhook_urls {
                                    out.push_str(&format!("  {}\n", url));
                                }
                            }
                            out.push_str("\nTunnel running. Restarting daemon to switch from polling to webhooks...");
                        } else if restart_scheduled {
                            out.push_str(
                                "\n\nAutomatic daemon restart scheduled in ~3s to activate Telegram webhook listeners.",
                            );
                        } else {
                            out.push_str(
                                "\n\nDaemon restart was already scheduled. Webhook listeners will be activated shortly.",
                            );
                        }
                    }
                    out
                }
            }
            "help" => usage,
            _ => usage,
        }
    }

    #[cfg(test)]
    fn strip_low_latency_next_steps(text: &str) -> String {
        let marker = "\nNext steps:\n";
        if let Some(index) = text.find(marker) {
            text[..index].trim_end().to_string()
        } else {
            text.to_string()
        }
    }

    async fn backup_cloudflared_origin_cert_if_present(
        cert_path: &Path,
    ) -> Result<Option<PathBuf>, String> {
        let metadata = match tokio::fs::metadata(cert_path).await {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(format!(
                    "failed to inspect cloudflared cert at `{}`: {}",
                    cert_path.display(),
                    err
                ));
            }
        };
        if !metadata.is_file() {
            return Err(format!(
                "cloudflared cert path `{}` exists but is not a file",
                cert_path.display()
            ));
        }

        let parent = cert_path.parent().ok_or_else(|| {
            format!(
                "cannot determine parent directory for cloudflared cert `{}`",
                cert_path.display()
            )
        })?;
        let file_name = cert_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                format!(
                    "cannot derive file name for cloudflared cert `{}`",
                    cert_path.display()
                )
            })?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for attempt in 0..20 {
            let backup_name = if attempt == 0 {
                format!("{}.bak.{}", file_name, timestamp)
            } else {
                format!("{}.bak.{}.{}", file_name, timestamp, attempt)
            };
            let backup_path = parent.join(backup_name);
            match tokio::fs::metadata(&backup_path).await {
                Ok(_) => continue,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => {
                    return Err(format!(
                        "failed to inspect cloudflared backup path `{}`: {}",
                        backup_path.display(),
                        err
                    ));
                }
            }

            match tokio::fs::rename(cert_path, &backup_path).await {
                Ok(_) => return Ok(Some(backup_path)),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(err) => {
                    return Err(format!(
                        "failed to back up cloudflared cert `{}` to `{}`: {}",
                        cert_path.display(),
                        backup_path.display(),
                        err
                    ));
                }
            }
        }

        Err(format!(
            "failed to back up cloudflared cert `{}`: too many backup name collisions",
            cert_path.display()
        ))
    }

    async fn prepare_cloudflared_origin_cert_for_login() -> Result<Option<String>, String> {
        let Some(cert_path) = crate::wizard::cloudflared_origin_cert_path() else {
            return Ok(None);
        };
        let backup_path = Self::backup_cloudflared_origin_cert_if_present(&cert_path).await?;
        Ok(backup_path.map(|path| {
            format!(
                "Existing cloudflared origin cert was backed up to `{}` before re-authentication.",
                path.display()
            )
        }))
    }

    async fn read_setup_command_stream<R>(reader: R, tx: mpsc::UnboundedSender<String>)
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        let mut lines = BufReader::new(reader).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }
    }

    async fn run_login_command_with_live_output(
        &self,
        chat_id: i64,
        spec: &crate::wizard::SetupCommandSpec,
        timeout_secs: u64,
        intro: &str,
    ) -> SetupLoginResult {
        let mut result = SetupLoginResult::default();
        let _ = self
            .bot
            .send_message(
                ChatId(chat_id),
                format!(
                    "{}\nCommand: {}",
                    intro,
                    crate::wizard::format_setup_command(spec)
                ),
            )
            .await;

        let mut child = match Command::new(&spec.program)
            .args(&spec.args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(err) => {
                result.error = Some(err.to_string());
                return result;
            }
        };

        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let mut stream_tasks = Vec::new();
        if let Some(stdout) = child.stdout.take() {
            let tx_clone = tx.clone();
            stream_tasks.push(tokio::spawn(async move {
                Self::read_setup_command_stream(stdout, tx_clone).await;
            }));
        }
        if let Some(stderr) = child.stderr.take() {
            let tx_clone = tx.clone();
            stream_tasks.push(tokio::spawn(async move {
                Self::read_setup_command_stream(stderr, tx_clone).await;
            }));
        }
        drop(tx);

        let bot = self.bot.clone();
        let announcer = tokio::spawn(async move {
            let mut lines = Vec::new();
            let mut urls = Vec::new();
            let mut seen_urls = HashSet::new();
            while let Some(line) = rx.recv().await {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if lines.len() < 300 {
                    lines.push(trimmed.to_string());
                }
                for url in crate::wizard::extract_urls(trimmed) {
                    if seen_urls.insert(url.clone()) {
                        urls.push(url.clone());
                        let _ = bot
                            .send_message(
                                ChatId(chat_id),
                                format!("Open this URL to continue auth:\n{}", url),
                            )
                            .await;
                    }
                }
            }
            (lines, urls)
        });

        let wait_outcome =
            tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait()).await;
        match wait_outcome {
            Ok(Ok(status)) => {
                result.success = status.success();
            }
            Ok(Err(err)) => {
                result.error = Some(err.to_string());
            }
            Err(_) => {
                result.timed_out = true;
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
        }

        for task in stream_tasks {
            let _ = task.await;
        }
        match announcer.await {
            Ok((lines, urls)) => {
                result.lines = lines;
                result.urls = urls;
            }
            Err(err) => {
                if result.error.is_none() {
                    result.error = Some(format!("failed to capture auth output: {}", err));
                }
            }
        }
        result
    }

    async fn run_cloudflared_login_and_verify(&self, chat_id: i64) -> Result<String, String> {
        let preflight_note = Self::prepare_cloudflared_origin_cert_for_login().await?;
        let login_spec = crate::wizard::SetupCommandSpec {
            program: "cloudflared".to_string(),
            args: vec!["tunnel".to_string(), "login".to_string()],
        };
        let login_result = self
            .run_login_command_with_live_output(
                chat_id,
                &login_spec,
                600,
                "Starting cloudflared authentication flow.",
            )
            .await;
        if login_result.timed_out {
            return Err(
                "Cloudflared login timed out after 600s. Complete authorization and retry."
                    .to_string(),
            );
        }
        if let Some(err) = login_result.error {
            return Err(format!("Failed to run cloudflared login: {}", err));
        }

        let auth_ok = login_result.success
            && crate::wizard::command_succeeds("cloudflared", &["tunnel", "list"], 20).await;
        if !auth_ok {
            let summary = crate::wizard::summarize_setup_log_lines(&login_result.lines);
            let mut details = Vec::new();
            if let Some(note) = preflight_note.clone() {
                details.push(note);
            }
            if !summary.is_empty() {
                details.push(summary);
            }
            if !login_result.urls.is_empty() {
                details.push(format!(
                    "Authorization URL: {}",
                    login_result.urls.join(", ")
                ));
            }
            let detail_block = if details.is_empty() {
                String::new()
            } else {
                format!("\n{}", details.join("\n"))
            };
            return Err(format!(
                "Cloudflared authentication did not complete successfully.{}",
                detail_block
            ));
        }

        let summary = crate::wizard::summarize_setup_log_lines(&login_result.lines);
        let mut report_lines = vec!["Cloudflared authentication completed.".to_string()];
        if let Some(url) = login_result.urls.first() {
            report_lines.push(format!("URL: {}", url));
        }
        if let Some(note) = preflight_note {
            report_lines.push(note);
        }
        if !summary.is_empty() {
            report_lines.push(summary);
        }
        Ok(report_lines.join("\n"))
    }

    async fn run_setup_commands(
        specs: &[crate::wizard::SetupCommandSpec],
        timeout_secs: u64,
        treat_existing_resource_as_success: bool,
    ) -> (Vec<String>, bool) {
        let mut lines = Vec::with_capacity(specs.len());
        let mut had_failures = false;
        for spec in specs {
            let display = crate::wizard::format_setup_command(spec);
            let run_result = tokio::time::timeout(
                Duration::from_secs(timeout_secs),
                Command::new(&spec.program)
                    .args(&spec.args)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .output(),
            )
            .await;
            match run_result {
                Ok(Ok(output)) => {
                    let summary = crate::wizard::summarize_setup_command_output(
                        &output.stdout,
                        &output.stderr,
                    );
                    let combined = format!(
                        "{}\n{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                    let treated_as_success = output.status.success()
                        || (treat_existing_resource_as_success
                            && crate::wizard::cloudflared_reports_existing_resource(&combined));
                    if treated_as_success {
                        if summary.is_empty() {
                            lines.push(format!("OK   {}", display));
                        } else {
                            lines.push(format!("OK   {} - {}", display, summary));
                        }
                    } else {
                        had_failures = true;
                        if summary.is_empty() {
                            lines.push(format!(
                                "FAIL {} - exited with status {}",
                                display, output.status
                            ));
                        } else {
                            lines.push(format!("FAIL {} - {}", display, summary));
                        }
                    }
                }
                Ok(Err(err)) => {
                    had_failures = true;
                    lines.push(format!("FAIL {} - {}", display, err));
                }
                Err(_) => {
                    had_failures = true;
                    lines.push(format!(
                        "FAIL {} - timed out after {}s",
                        display, timeout_secs
                    ));
                }
            }
        }
        (lines, had_failures)
    }

    fn schedule_low_latency_restart(delay: Duration) -> bool {
        if LOW_LATENCY_RESTART_SCHEDULED.swap(true, Ordering::SeqCst) {
            return false;
        }
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            info!("Automatic restart requested after low-latency setup");
            restart_process();
            LOW_LATENCY_RESTART_SCHEDULED.store(false, Ordering::SeqCst);
        });
        true
    }

    /// Handle /connect command - add a new bot dynamically.
    /// Usage: /connect telegram <bot_token>
    ///        /connect discord <bot_token>
    ///        /connect slack <bot_token> <app_token>
    async fn handle_connect_command(&self, arg: &str, user_id: u64) -> String {
        let parts: Vec<&str> = arg.split_whitespace().collect();

        if parts.is_empty() {
            return "Add a new bot to this agent.\n\n\
                Usage:\n\
                /connect telegram <bot_token>\n\
                /connect discord <bot_token>\n\
                /connect slack <bot_token> <app_token>\n\n\
                The new bot will use the same allowed users as this bot.\n\
                After adding, run /restart to activate the new bot."
                .to_string();
        }

        let channel_type = parts[0].to_lowercase();

        match channel_type.as_str() {
            "telegram" => {
                if parts.len() < 2 {
                    return "Usage: /connect telegram <bot_token>\n\n\
                        Get a token from @BotFather on Telegram."
                        .to_string();
                }
                let token = parts[1];
                self.connect_telegram_bot(token, user_id).await
            }
            "discord" => {
                if parts.len() < 2 {
                    return "Usage: /connect discord <bot_token>\n\n\
                        Get a token from Discord Developer Portal."
                        .to_string();
                }
                let token = parts[1];
                self.connect_discord_bot(token, user_id).await
            }
            "slack" => {
                if parts.len() < 3 {
                    return "Usage: /connect slack <bot_token> <app_token>\n\n\
                        Get tokens from Slack App Management."
                        .to_string();
                }
                let bot_token = parts[1];
                let app_token = parts[2];
                self.connect_slack_bot(bot_token, app_token, user_id).await
            }
            _ => {
                format!(
                    "Unknown channel type: {}\n\n\
                    Supported types: telegram, discord, slack",
                    channel_type
                )
            }
        }
    }

    /// Connect a new Telegram bot by validating its token.
    async fn connect_telegram_bot(&self, token: &str, user_id: u64) -> String {
        // Validate the token by calling getMe
        let bot_username = match super::connect::validate_telegram_token(token).await {
            Ok(name) => name,
            Err(e) => return e,
        };

        // Check if this bot is already connected
        match super::connect::check_bot_exists(self.state.as_ref(), "telegram", token).await {
            Ok(true) => {
                return format!(
                    "Bot @{} is already connected.\n\nUse /bots to see all connected bots.",
                    bot_username
                );
            }
            Ok(false) => {}
            Err(e) => {
                warn!("Failed to check existing bots: {}", e);
            }
        }

        // Save the new bot with same allowed users as current bot
        let allowed_user_ids_str: Vec<String> = self
            .allowed_user_ids
            .read()
            .unwrap()
            .iter()
            .map(|id| id.to_string())
            .collect();

        let new_bot = crate::traits::DynamicBot {
            id: 0, // Will be set by database
            channel_type: "telegram".to_string(),
            bot_token: token.to_string(),
            app_token: None,
            allowed_user_ids: allowed_user_ids_str,
            extra_config: serde_json::json!({"username": bot_username}).to_string(),
            created_at: String::new(), // Will be set by database
        };

        let db_id = match super::connect::persist_dynamic_bot(self.state.as_ref(), &new_bot).await {
            Ok(id) => id,
            Err(e) => {
                warn!("Failed to save bot: {}", e);
                return e;
            }
        };

        info!(
            bot = %bot_username,
            id = db_id,
            added_by = user_id,
            "New Telegram bot connected"
        );

        let mini_app_sync_warning = match self
            .terminal
            .sync_tenant_bot_token(token, user_id, Some(bot_username.as_str()))
            .await
        {
            Ok(()) => None,
            Err(err) => {
                warn!(
                    error = %err,
                    bot = %bot_username,
                    added_by = user_id,
                    "Failed to auto-sync Telegram bot token with terminal mini app backend"
                );
                Some(
                    "Mini App auth sync failed. If you hit a Telegram auth mismatch, run /agent from this bot once or save the token manually in the Mini App."
                        .to_string(),
                )
            }
        };

        // Try to spawn the bot immediately if we have a hub reference
        let hub_ref = self.channel_hub_weak();
        if let Some(weak_hub) = hub_ref
            && let Some(hub) = weak_hub.upgrade()
        {
            // Create the new channel with same config as this one
            let new_channel = Arc::new(new_telegram_bot(
                token,
                self.allowed_user_ids
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone(),
                TelegramWebhookConfig::default(),
                self.owner_user_ids.clone(),
                self.runtime_deps(),
                self.terminal.web_app_url().to_string(),
                self.terminal.lite_allowed_prefixes(),
            ));

            // Give the new channel a reference to the hub too
            new_channel.set_channel_hub(weak_hub);

            // Register with the hub
            let channel_name = hub
                .register_channel(new_channel.clone() as Arc<dyn HubChannel>)
                .await;
            info!(channel = %channel_name, "Registered new Telegram bot with hub");

            // Spawn the bot in the background using helper to avoid type cycles
            spawn_telegram_channel(new_channel);

            return format!(
                "✓ Bot @{} connected and started!\n\n\
                    The bot is now active and ready to receive messages.\n\
                    Use /bots to see all connected bots.{}",
                bot_username,
                mini_app_sync_warning
                    .as_deref()
                    .map(|value| format!("\n\n{}", value))
                    .unwrap_or_default()
            );
        }

        // Fallback: hub not available, require restart
        format!(
            "✓ Bot @{} connected!\n\n\
            Run /restart to activate the new bot.\n\
            Use /bots to see all connected bots.{}",
            bot_username,
            mini_app_sync_warning
                .as_deref()
                .map(|value| format!("\n\n{}", value))
                .unwrap_or_default()
        )
    }

    /// Connect a new Discord bot by validating its token.
    #[cfg(feature = "discord")]
    async fn connect_discord_bot(&self, token: &str, user_id: u64) -> String {
        // Validate the token by making a test API call
        let bot_name = match super::connect::validate_discord_token(token).await {
            Ok(name) => name,
            Err(e) => return e,
        };

        // Check if already connected
        match super::connect::check_bot_exists(self.state.as_ref(), "discord", token).await {
            Ok(true) => {
                return format!(
                    "Bot {} is already connected.\n\nUse /bots to see all connected bots.",
                    bot_name
                );
            }
            Ok(false) => {}
            Err(e) => {
                warn!("Failed to check existing bots: {}", e);
            }
        }

        // Discord user IDs differ from Telegram IDs — save empty and let
        // the Discord bot auto-claim the first DM user as the owner.
        let new_bot = crate::traits::DynamicBot {
            id: 0,
            channel_type: "discord".to_string(),
            bot_token: token.to_string(),
            app_token: None,
            allowed_user_ids: vec![],
            extra_config: "{}".to_string(),
            created_at: String::new(),
        };

        let db_id = match super::connect::persist_dynamic_bot(self.state.as_ref(), &new_bot).await {
            Ok(id) => id,
            Err(e) => return e,
        };

        info!(
            bot = %bot_name,
            id = db_id,
            added_by = user_id,
            "New Discord bot connected"
        );

        // Try to spawn the bot immediately if we have a hub reference
        let hub_ref = self.channel_hub_weak();
        if let Some(weak_hub) = hub_ref
            && let Some(hub) = weak_hub.upgrade()
        {
            // Create the new Discord channel with empty allowed_user_ids —
            // the bot will auto-claim the first DM user.
            let new_channel = Arc::new(super::discord::new_discord_bot(
                token,
                vec![], // Empty: auto-claim on first DM
                vec![], // Owner set on auto-claim
                None,   // No guild_id for dynamic bots
                self.runtime_deps(),
            ));

            // Give the new channel a reference to the hub
            new_channel.set_channel_hub(weak_hub);

            // Register with the hub
            let channel_name = hub
                .register_channel(new_channel.clone() as Arc<dyn HubChannel>)
                .await;
            info!(channel = %channel_name, "Registered new Discord bot with hub");

            // Spawn the bot in the background
            spawn_discord_channel(new_channel);

            return format!(
                "✓ Discord bot {} connected and started!\n\n\
                    Send a DM to the bot on Discord to claim it as yours.\n\
                    Use /bots to see all connected bots.",
                bot_name
            );
        }

        // Fallback: hub not available, require restart
        format!(
            "✓ Discord bot {} connected!\n\n\
            Run /restart to activate the new bot.\n\
            Use /bots to see all connected bots.",
            bot_name
        )
    }

    /// Connect a new Discord bot (stub when feature disabled).
    #[cfg(not(feature = "discord"))]
    async fn connect_discord_bot(&self, token: &str, _user_id: u64) -> String {
        match super::connect::validate_discord_token(token).await {
            Ok(_) => unreachable!("discord feature is disabled"),
            Err(e) => format!(
                "{}\n\nRebuild with `cargo build --features discord` to enable Discord bots.",
                e
            ),
        }
    }

    /// Connect a new Slack bot.
    #[cfg(feature = "slack")]
    async fn connect_slack_bot(&self, bot_token: &str, app_token: &str, user_id: u64) -> String {
        // Validate the bot token by calling auth.test
        let (bot_name, team_name) =
            match super::connect::validate_slack_tokens(bot_token, app_token).await {
                Ok(pair) => pair,
                Err(e) => return e,
            };

        // Check if already connected
        match super::connect::check_bot_exists(self.state.as_ref(), "slack", bot_token).await {
            Ok(true) => {
                return format!(
                    "Slack bot {} ({}) is already connected.\n\nUse /bots to see all connected bots.",
                    bot_name, team_name
                );
            }
            Ok(false) => {}
            Err(e) => {
                warn!("Failed to check existing bots: {}", e);
            }
        }

        let allowed_user_ids_str = vec![user_id.to_string()];

        let new_bot = crate::traits::DynamicBot {
            id: 0,
            channel_type: "slack".to_string(),
            bot_token: bot_token.to_string(),
            app_token: Some(app_token.to_string()),
            allowed_user_ids: allowed_user_ids_str.clone(),
            extra_config: "{}".to_string(),
            created_at: String::new(),
        };

        let db_id = match super::connect::persist_dynamic_bot(self.state.as_ref(), &new_bot).await {
            Ok(id) => id,
            Err(e) => return e,
        };

        info!(
            bot = %bot_name,
            team = %team_name,
            id = db_id,
            added_by = user_id,
            "New Slack bot connected"
        );

        // Try to spawn the bot immediately if we have a hub reference
        let hub_ref = self.channel_hub_weak();
        if let Some(weak_hub) = hub_ref
            && let Some(hub) = weak_hub.upgrade()
        {
            // Create the new Slack channel with same config as this Telegram channel
            let new_channel = Arc::new(super::slack::new_slack_bot(
                app_token,
                bot_token,
                allowed_user_ids_str, // Slack uses String user IDs
                Vec::new(),           // Empty preserves legacy allowed-user owner behavior
                Vec::new(),           // Dynamic bot starts without delegated workspaces
                false,                // use_threads default
                self.runtime_deps(),
            ));

            // Give the new channel a reference to the hub
            new_channel.set_channel_hub(weak_hub);

            // Register with the hub
            let channel_name = hub
                .register_channel(new_channel.clone() as Arc<dyn HubChannel>)
                .await;
            info!(channel = %channel_name, "Registered new Slack bot with hub");

            // Spawn the bot in the background
            spawn_slack_channel(new_channel);

            return format!(
                "✓ Slack bot {} ({}) connected and started!\n\n\
                    The bot is now active and ready to receive messages.\n\
                    Use /bots to see all connected bots.",
                bot_name, team_name
            );
        }

        // Fallback: hub not available, require restart
        format!(
            "✓ Slack bot {} ({}) connected!\n\n\
            Run /restart to activate the new bot.\n\
            Use /bots to see all connected bots.",
            bot_name, team_name
        )
    }

    /// Connect a new Slack bot (stub when feature disabled).
    #[cfg(not(feature = "slack"))]
    async fn connect_slack_bot(&self, bot_token: &str, app_token: &str, _user_id: u64) -> String {
        match super::connect::validate_slack_tokens(bot_token, app_token).await {
            Ok(_) => unreachable!("slack feature is disabled"),
            Err(e) => format!(
                "{}\n\nRebuild with `cargo build --features slack` to enable Slack bots.",
                e
            ),
        }
    }

    /// Handle /bots command - list all connected bots.
    async fn handle_bots_command(&self) -> String {
        let mut bots_list = vec![];
        let mut seen_credentials = std::collections::HashSet::new();

        // Add current bot (from config)
        let current_username = self.get_bot_username().await;
        seen_credentials.insert(("telegram".to_string(), self.bot_token.clone()));
        bots_list.push(format!(
            "• telegram:@{} (this bot, from config)",
            current_username
        ));

        // Add dynamic bots from database
        match super::connect::list_dynamic_bots(self.state.as_ref()).await {
            Ok(bots) => {
                for bot in bots {
                    if !seen_credentials.insert((bot.channel_type.clone(), bot.bot_token.clone())) {
                        continue;
                    }
                    let bot_info = match bot.channel_type.as_str() {
                        "telegram" => {
                            // Try to get username for display
                            match super::connect::validate_telegram_token(&bot.bot_token).await {
                                Ok(username) => {
                                    format!("• telegram:@{} (id: {})", username, bot.id)
                                }
                                Err(_) => format!("• telegram:<invalid token> (id: {})", bot.id),
                            }
                        }
                        "discord" => format!("• discord (id: {})", bot.id),
                        "slack" => format!("• slack (id: {})", bot.id),
                        other => format!("• {} (id: {})", other, bot.id),
                    };
                    bots_list.push(bot_info);
                }
            }
            Err(e) => {
                return format!("Failed to list bots: {}", e);
            }
        }

        if bots_list.len() == 1 {
            format!(
                "Connected bots:\n{}\n\nUse /connect to add more bots.",
                bots_list.join("\n")
            )
        } else {
            format!(
                "Connected bots ({}):\n{}\n\n\
                Tip: Dynamic bots activate after /restart.",
                bots_list.len(),
                bots_list.join("\n")
            )
        }
    }
}

#[cfg(test)]
#[path = "telegram_sdk_tests.rs"]
mod sdk_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_agent_permission_aliases_maps_claude_allow_flag() {
        let args = vec![
            "--allow-dangerously-skip-permissions".to_string(),
            "--model".to_string(),
            "sonnet".to_string(),
        ];
        let (normalized, rewrote) =
            crate::normalize_terminal_agent_permission_aliases(Some("claude"), args);
        assert!(rewrote);
        assert_eq!(
            normalized,
            vec![
                "--dangerously-skip-permissions".to_string(),
                "--model".to_string(),
                "sonnet".to_string()
            ]
        );
    }

    #[test]
    fn normalize_agent_permission_aliases_leaves_non_claude_args_unchanged() {
        let args = vec!["--allow-dangerously-skip-permissions".to_string()];
        let (normalized, rewrote) =
            crate::normalize_terminal_agent_permission_aliases(Some("codex"), args.clone());
        assert!(!rewrote);
        assert_eq!(normalized, args);
    }

    // --- check_auth ---

    #[test]
    fn strip_low_latency_next_steps_removes_manual_instructions() {
        let text = "Low-latency webhook config applied (local config only).\nBackup: config.toml.lowlatency.bak\n\nNext steps:\n1. step one\n2. step two\n";
        let stripped = Cx::strip_low_latency_next_steps(text);
        assert_eq!(
            stripped,
            "Low-latency webhook config applied (local config only).\nBackup: config.toml.lowlatency.bak"
        );
    }

    #[test]
    fn strip_low_latency_next_steps_leaves_text_without_marker() {
        let text = "Low-latency webhook config applied (local config only).\nBackup: config.toml.lowlatency.bak\n";
        let stripped = Cx::strip_low_latency_next_steps(text);
        assert_eq!(stripped, text);
    }

    #[tokio::test]
    async fn backup_cloudflared_origin_cert_moves_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");
        tokio::fs::write(&cert_path, "dummy-cert").await.unwrap();

        let backup = Cx::backup_cloudflared_origin_cert_if_present(&cert_path)
            .await
            .unwrap()
            .unwrap();

        assert!(tokio::fs::metadata(&cert_path).await.is_err());
        let backup_content = tokio::fs::read_to_string(&backup).await.unwrap();
        assert_eq!(backup_content, "dummy-cert");
    }

    #[tokio::test]
    async fn backup_cloudflared_origin_cert_is_noop_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");

        let backup = Cx::backup_cloudflared_origin_cert_if_present(&cert_path)
            .await
            .unwrap();
        assert!(backup.is_none());
    }

    // --- persist_allowed_user_ids (config file update) ---

    #[tokio::test]
    async fn persist_updates_legacy_telegram_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        let initial = r#"
[telegram]
bot_token = "test-token"
allowed_user_ids = []
"#;
        tokio::fs::write(&config_path, initial).await.unwrap();

        // Parse and re-write with the helper logic
        let content = tokio::fs::read_to_string(&config_path).await.unwrap();
        let mut doc: toml::Table = content.parse().unwrap();
        let ids = [12345u64];
        let ids_toml = toml::Value::Array(
            ids.iter()
                .map(|&id| toml::Value::Integer(id as i64))
                .collect(),
        );
        if let Some(tg) = doc.get_mut("telegram").and_then(|v| v.as_table_mut()) {
            tg.insert("allowed_user_ids".to_string(), ids_toml);
        }
        let new_content = toml::to_string_pretty(&toml::Value::Table(doc)).unwrap();
        tokio::fs::write(&config_path, &new_content).await.unwrap();

        // Verify
        let saved = tokio::fs::read_to_string(&config_path).await.unwrap();
        let doc: toml::Table = saved.parse().unwrap();
        let ids_val = doc["telegram"]["allowed_user_ids"].as_array().unwrap();
        assert_eq!(ids_val.len(), 1);
        assert_eq!(ids_val[0].as_integer().unwrap(), 12345);
    }

    #[tokio::test]
    async fn persist_updates_telegram_bots_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        let initial = r#"
[[telegram_bots]]
bot_token = "test-token"
allowed_user_ids = []
"#;
        tokio::fs::write(&config_path, initial).await.unwrap();

        let content = tokio::fs::read_to_string(&config_path).await.unwrap();
        let mut doc: toml::Table = content.parse().unwrap();
        let ids = [67890u64];
        let ids_toml = toml::Value::Array(
            ids.iter()
                .map(|&id| toml::Value::Integer(id as i64))
                .collect(),
        );
        if let Some(bots) = doc.get_mut("telegram_bots").and_then(|v| v.as_array_mut())
            && let Some(first) = bots.first_mut().and_then(|v| v.as_table_mut())
        {
            first.insert("allowed_user_ids".to_string(), ids_toml);
        }
        let new_content = toml::to_string_pretty(&toml::Value::Table(doc)).unwrap();
        tokio::fs::write(&config_path, &new_content).await.unwrap();

        let saved = tokio::fs::read_to_string(&config_path).await.unwrap();
        let doc: toml::Table = saved.parse().unwrap();
        let bots = doc["telegram_bots"].as_array().unwrap();
        let ids_val = bots[0]["allowed_user_ids"].as_array().unwrap();
        assert_eq!(ids_val.len(), 1);
        assert_eq!(ids_val[0].as_integer().unwrap(), 67890);
    }
}
