//! The daemon's side of the Telegram bot ([`TelegramBot`], in
//! `aidaemon-channel-telegram`): its config file, the shared slash commands
//! on its stores, and its own commands (`/setkey`, `/connect`, `/bots`,
//! `/setup`, `/restart`, `/agent`, `/terminal`) with the Mini App handoff.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock as StdRwLock, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::telegram_api::{
    Bot, ButtonRequest, CallbackQuery, ChatId, InlineKeyboardButton, InlineKeyboardMarkup,
    KeyboardButton, KeyboardMarkup, ParseMode, WebAppInfo,
};
pub(crate) use aidaemon_channel_telegram::bot::TelegramBot;
use aidaemon_channel_telegram::bot::{
    determine_role, send_markdown_chunks_or_fallback, HostCommand, KeyChoice, TelegramHost,
    TelegramWebhookSettings,
};
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tracing::{info, warn};

use super::commands::{CommandCategory, CommandDef};
use super::telegram_bootstrap_signing::{
    random_terminal_bootstrap_nonce, sign_terminal_tenant_bot_bootstrap_proof,
    terminal_tenant_bot_bootstrap_url,
};
#[cfg(feature = "discord")]
use crate::channels::spawn_discord_channel;
#[cfg(feature = "slack")]
use crate::channels::{spawn_slack_channel, SlackChannel};
use crate::channels::{ChannelHub, ChannelRuntimeDeps};
use crate::config::{AppConfig, TelegramWebhookConfig};
#[cfg(test)]
use crate::runtime_ports::DaemonAgentRuntime;
#[cfg(test)]
use crate::tasks::TaskRegistry;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::traits::{HubChannel, StateStore};
use crate::types::{ApprovalResponse, UserRole};
use aidaemon_channel_telegram::format::html_escape;

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
    let terminal_allowed_prefixes = terminal_allowed_prefixes
        .into_iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<HashSet<_>>();
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
        terminal_web_app_url,
        terminal_lite: crate::terminal_lite::TerminalLiteManager::new(terminal_allowed_prefixes),
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
    /// URL for the Telegram Mini App terminal frontend.
    terminal_web_app_url: String,
    /// Terminal-lite session manager.
    terminal_lite: crate::terminal_lite::TerminalLiteManager,
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
            terminal_web_app_url: self.terminal_web_app_url.clone(),
            terminal_lite: &self.terminal_lite,
        }
    }

    /// Test the key against its provider and save it only if it is accepted.
    async fn install_key(&self, slot_id: &str, label: &str, key: &str) -> String {
        use super::set_key::{check_key, key_targets, save_key, KeyCheck, KeySlot};
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
    terminal_web_app_url: String,
    terminal_lite: &'a crate::terminal_lite::TerminalLiteManager,
}

impl Cx<'_> {
    async fn session_id(&self, chat_id: i64) -> String {
        self.tg.session_id(chat_id).await
    }

    async fn session_namespace(&self) -> String {
        self.tg.session_namespace().await
    }

    async fn get_bot_username(&self) -> String {
        self.tg.get_bot_username().await
    }

    fn cached_bot_label(&self) -> Option<String> {
        self.tg.cached_bot_label()
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
        CommandDef {
            name: "agent",
            description: "Start or manage CLI agent sessions",
            usage: Some("/agent [codex|claude|gemini|opencode] [dir]"),
            category: CommandCategory::Terminal,
        },
        CommandDef {
            name: "terminal",
            description: "Terminal bridge commands",
            usage: Some("/terminal [lite|start|open]"),
            category: CommandCategory::Terminal,
        },
    ]
}

#[async_trait]
impl TelegramHost for DaemonTelegramHost {
    fn commands(&self) -> Vec<CommandDef> {
        daemon_commands()
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
        let cx = self.cx(tg);
        let bot = cx.bot.clone();
        let user_id = msg.from.as_ref().map(|u| u.id.0).unwrap_or(0);
        HostCommand::Reply(match cmd {
            "/terminal" | "/agent" => {
                cx.handle_terminal_command(arg, msg, &bot, cmd).await;
                return HostCommand::Handled;
            }
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
        let cx = self.cx(tg);
        let bot = cx.bot.clone();
        if data == "agent:share" || data.starts_with("agent:share:") {
            let explicit_relay_session_id = data
                .strip_prefix("agent:share:")
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(|v| v.to_string());
            let _ = bot
                .answer_callback_query(q.id.clone())
                .text("Generating share code...")
                .await;
            if let Some(m) = &q.message {
                cx.send_agent_share_code(&bot, m.chat.id, user_id, explicit_relay_session_id)
                    .await;
            } else {
                let _ = bot
                    .answer_callback_query(q.id.clone())
                    .text("Open the chat and run /agent share.")
                    .show_alert(true)
                    .await;
            }
            return true;
        }
        false
    }

    async fn key_choices(&self) -> Result<Vec<KeyChoice>, String> {
        DaemonTelegramHost::key_choices(self).await
    }

    async fn install_key(&self, id: &str, label: &str, key: &str) -> String {
        DaemonTelegramHost::install_key(self, id, label, key).await
    }

    async fn intercept_text(
        &self,
        tg: &Arc<TelegramBot>,
        chat_id: i64,
        user_id: u64,
        user_role: UserRole,
        text: &str,
    ) -> Option<String> {
        self.cx(tg)
            .handle_terminal_lite_input(chat_id, user_id, user_role, text)
            .await
    }

    async fn continue_on_computer(
        &self,
        tg: &Arc<TelegramBot>,
        msg: &super::telegram_api::Message,
        user_id: u64,
        relay_session_id: Option<String>,
    ) {
        let cx = self.cx(tg);
        let bot = cx.bot.clone();
        cx.handle_continue_on_computer_action(&bot, msg, user_id, relay_session_id)
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
            match AppConfig::load(&cx.config_path) {
                Ok(config) => {
                    if config.terminal.effective_bridge_enabled()
                        && !crate::terminal_bridge::spawn_if_configured(&config, cx.state.clone())
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
        self.cx(tg)
            .sync_terminal_tenant_bot_token_for_allowed_users(Some(label))
            .await;
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

use crate::cli_agent_flags::{self, AGENT_FLAGS_CACHE_TTL_SECS, SUPPORTED_TERMINAL_AGENTS};

const DEFAULT_TERMINAL_AGENT: &str = "codex";
static LOW_LATENCY_RESTART_SCHEDULED: AtomicBool = AtomicBool::new(false);

fn interpret_terminal_bot_sync_response(
    status: reqwest::StatusCode,
    content_type: Option<&str>,
    body: &[u8],
) -> (Result<(), String>, bool) {
    let retryable_status =
        status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
    let parsed = match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(parsed) => parsed,
        Err(error) => {
            let content_type = content_type.unwrap_or("unknown");
            return (
                Err(format!(
                    "terminal bot sync returned non-JSON response (status {}, content-type {}, {} bytes): {}",
                    status,
                    content_type,
                    body.len(),
                    error
                )),
                retryable_status || status.is_success(),
            );
        }
    };
    if status.is_success() && parsed.get("ok").and_then(|value| value.as_bool()) == Some(true) {
        return (Ok(()), false);
    }

    let message = parsed
        .get("message")
        .and_then(|value| value.as_str())
        .or_else(|| parsed.get("error").and_then(|value| value.as_str()))
        .unwrap_or("unknown error");
    (
        Err(format!(
            "terminal bot sync failed (status {}): {}",
            status,
            crate::tools::sanitize::redact_secrets(message)
        )),
        retryable_status,
    )
}

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
        if let Some(bots) = doc.get_mut("telegram_bots").and_then(|v| v.as_array_mut()) {
            if let Some(first) = bots.first_mut().and_then(|v| v.as_table_mut()) {
                first.insert("allowed_user_ids".to_string(), ids_toml);
                updated = true;
            }
        }

        if !updated {
            anyhow::bail!("No [telegram] or [[telegram_bots]] section found in config");
        }

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        tokio::fs::write(&self.config_path, &new_content).await?;
        info!("Persisted allowed_user_ids to config.toml");
        Ok(())
    }

    async fn sync_terminal_tenant_bot_token(
        &self,
        bot_token: &str,
        user_id: u64,
        label: Option<&str>,
    ) -> Result<(), String> {
        let url = terminal_tenant_bot_bootstrap_url(&self.terminal_web_app_url)?;
        let client = reqwest::Client::builder()
            .user_agent("aidaemon-terminal-bot-bootstrap/1.0")
            .timeout(Duration::from_secs(8))
            .build()
            .map_err(|err| format!("failed to build HTTP client: {}", err))?;
        let mut last_error = "terminal bot sync did not run".to_string();
        for attempt in 0..3_u64 {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|err| format!("failed to read system clock: {}", err))?
                .as_secs() as i64;
            let nonce = random_terminal_bootstrap_nonce(16);
            let sig = sign_terminal_tenant_bot_bootstrap_proof(bot_token, user_id, ts, &nonce)?;
            let mut payload = serde_json::json!({
                "user_id": user_id,
                "bot_token": bot_token,
                "ts": ts,
                "nonce": nonce,
                "sig": sig,
            });
            if let Some(label) = label
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
            {
                if let Some(map) = payload.as_object_mut() {
                    map.insert(
                        "label".to_string(),
                        serde_json::Value::String(label.to_string()),
                    );
                }
            }

            let response = match client.post(url.clone()).json(&payload).send().await {
                Ok(response) => response,
                Err(error) => {
                    last_error = format!("terminal bot sync request failed: {error}");
                    if attempt == 2 {
                        return Err(last_error);
                    }
                    let jitter = rand::random::<u64>() % 150;
                    tokio::time::sleep(Duration::from_millis(150 * (attempt + 1) + jitter)).await;
                    continue;
                }
            };
            let status = response.status();
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let body = match response.bytes().await {
                Ok(body) => body,
                Err(error) => {
                    last_error = format!(
                        "terminal bot sync response body failed (status {status}): {error}"
                    );
                    if attempt == 2 {
                        return Err(last_error);
                    }
                    let jitter = rand::random::<u64>() % 150;
                    tokio::time::sleep(Duration::from_millis(150 * (attempt + 1) + jitter)).await;
                    continue;
                }
            };
            let (result, retryable) = interpret_terminal_bot_sync_response(
                status,
                content_type.as_deref(),
                body.as_ref(),
            );
            match result {
                Ok(()) => return Ok(()),
                Err(error) => last_error = error,
            }
            if !retryable || attempt == 2 {
                return Err(last_error);
            }
            let jitter = rand::random::<u64>() % 150;
            tokio::time::sleep(Duration::from_millis(150 * (attempt + 1) + jitter)).await;
        }
        Err(last_error)
    }

    async fn sync_terminal_tenant_bot_token_for_allowed_users(&self, label: Option<&str>) {
        let allowed_user_ids = self
            .allowed_user_ids
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        for user_id in allowed_user_ids {
            if let Err(err) = self
                .sync_terminal_tenant_bot_token(&self.bot_token, user_id, label)
                .await
            {
                warn!(
                    error = %err,
                    user_id,
                    "Failed to auto-sync Telegram bot token during channel startup"
                );
            }
        }
    }

    fn terminal_help_text() -> String {
        "Terminal mode (owner only)\n\n\
         Usage:\n\
         /terminal\n\
         /terminal <codex|claude|gemini|opencode> [working_dir]\n\
         /terminal <codex|claude|gemini|opencode> [working_dir] [agent_flags...]\n\
         /terminal <codex|claude|gemini|opencode> [working_dir] -- [agent_flags...]\n\
         /terminal start <agent> [working_dir]\n\
         /terminal start <agent> [working_dir] [agent_flags...]\n\
         /terminal start <agent> [working_dir] -- [agent_flags...]\n\
         /terminal lite [start] [working_dir]\n\
         /terminal lite status\n\
         /terminal lite stop\n\
         /terminal open\n\
         /terminal help\n\n\
         Examples:\n\
         /terminal codex\n\
         /terminal codex --chrome\n\
         /terminal codex ~/projects/aidaemon --chrome --dangerously-skip-permissions\n\
         /terminal codex ~/projects/aidaemon -- --chrome --dangerously-skip-permissions\n\
         /terminal opencode ~/projects/aidaemon\n\
         /terminal lite ~/projects/aidaemon\n\
         /terminal claude ~/projects/aidaemon\n\
         /terminal codex \"~/projects/my app\""
            .to_string()
    }

    fn agent_help_text() -> String {
        "Agent session mode (owner only)\n\n\
         Usage:\n\
         /agent\n\
         /agent <codex|claude|gemini|opencode> [working_dir]\n\
         /agent <codex|claude|gemini|opencode> [working_dir] [agent_flags...]\n\
         /agent <codex|claude|gemini|opencode> [working_dir] -- [agent_flags...]\n\
         /agent start <agent> [working_dir]\n\
         /agent start <agent> [working_dir] [agent_flags...]\n\
         /agent flags <agent> [refresh]\n\
         /agent defaults\n\
         /agent defaults set <agent> [agent_flags...]\n\
         /agent defaults clear [agent|all]\n\
         /agent share [relay_session_id]\n\
         /agent resume <code>\n\
         /agent open\n\
         /agent help\n\n\
         Examples:\n\
         /agent codex\n\
         /agent codex --chrome\n\
         /agent codex ~/projects/aidaemon --chrome --dangerously-skip-permissions\n\
         /agent claude ~/projects/aidaemon\n\
         /agent opencode ~/projects/aidaemon\n\
         /agent flags codex\n\
         /agent flags codex refresh\n\
         /agent share\n\
         /agent resume ABCDEFGHJKLM\n\
         /agent defaults set codex --chrome --dangerously-skip-permissions\n\n\
         Tip: add `--no-default-flags` to bypass saved flags once.\n\n\
         For chat-based shell mode, use:\n\
         /terminal lite [working_dir]"
            .to_string()
    }

    async fn handle_agent_flags_command(
        &self,
        args: Vec<String>,
        msg: &super::telegram_api::Message,
        bot: &Bot,
        user_id: u64,
    ) {
        if args.is_empty() {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    "Usage: /agent flags <agent> [refresh]\nExamples:\n/agent flags codex\n/agent flags codex refresh\n\nSupported agents: codex, claude, gemini, opencode",
                )
                .await;
            return;
        }

        let (agent_raw, refresh) = if args
            .first()
            .map(|v| v.eq_ignore_ascii_case("refresh"))
            .unwrap_or(false)
        {
            (args.get(1).cloned().unwrap_or_default(), true)
        } else {
            let refresh = args
                .get(1)
                .map(|v| v.eq_ignore_ascii_case("refresh"))
                .unwrap_or(false);
            (args[0].clone(), refresh)
        };

        let Some(agent) = cli_agent_flags::normalize_terminal_agent_name(&agent_raw) else {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    "Unknown agent. Use codex/claude/gemini/opencode.",
                )
                .await;
            return;
        };

        let namespace = self.session_namespace().await;
        if !refresh {
            if let Some(cached) =
                cli_agent_flags::load_agent_flags_cache(&*self.state, &namespace, user_id, &agent)
                    .await
            {
                let age = chrono::Utc::now().timestamp() - cached.updated_at_unix;
                if (0..=AGENT_FLAGS_CACHE_TTL_SECS).contains(&age) {
                    let pages = cli_agent_flags::format_agent_flag_docs(&agent, &cached.docs, true);
                    for page in pages {
                        send_markdown_chunks_or_fallback(bot, msg.chat.id, &page).await;
                    }
                    return;
                }
            }
        }

        let _ = bot
            .send_message(msg.chat.id, format!("Discovering flags for {}...", agent))
            .await;

        match cli_agent_flags::discover_agent_flags(&agent).await {
            Ok(docs) => {
                let _ = cli_agent_flags::save_agent_flags_cache(
                    &*self.state,
                    &namespace,
                    user_id,
                    &agent,
                    &docs,
                )
                .await;
                let pages = cli_agent_flags::format_agent_flag_docs(&agent, &docs, false);
                for page in pages {
                    send_markdown_chunks_or_fallback(bot, msg.chat.id, &page).await;
                }
            }
            Err(err) => {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!("Failed to discover flags for {}: {}", agent, err),
                    )
                    .await;
            }
        }
    }

    async fn handle_agent_defaults_command(
        &self,
        mut args: Vec<String>,
        msg: &super::telegram_api::Message,
        bot: &Bot,
        user_id: u64,
    ) {
        let subcommand = args
            .first()
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_else(|| "show".to_string());

        if !args.is_empty() {
            args.remove(0);
        }

        if subcommand == "set" {
            let Some(agent_raw) = args.first() else {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Usage: /agent defaults set <agent> [agent_flags...]",
                    )
                    .await;
                return;
            };
            let Some(agent) = cli_agent_flags::normalize_terminal_agent_name(agent_raw) else {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Unknown agent. Use codex/claude/gemini/opencode.",
                    )
                    .await;
                return;
            };
            let cleaned = cli_agent_flags::normalize_terminal_agent_args(
                args.into_iter().skip(1).collect::<Vec<_>>(),
            );
            let (cleaned, rewrote_permission_flag) =
                crate::normalize_terminal_agent_permission_aliases(Some(agent.as_str()), cleaned);
            if cleaned.is_empty() {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "No flags provided. Example: /agent defaults set codex --chrome",
                    )
                    .await;
                return;
            }
            let namespace = self.session_namespace().await;
            let mut defaults = cli_agent_flags::load_terminal_agent_defaults(
                &*self.state,
                &namespace,
                msg.chat.id.0,
                user_id,
            )
            .await;
            defaults.insert(agent.clone(), cleaned.clone());
            match cli_agent_flags::save_terminal_agent_defaults(
                &*self.state,
                &namespace,
                msg.chat.id.0,
                user_id,
                &defaults,
            )
            .await
            {
                Ok(_) => {
                    let saved_message = if rewrote_permission_flag {
                        format!(
                            "Saved default flags for {}:\n`{}`\n\nNormalized `--allow-dangerously-skip-permissions` to `--dangerously-skip-permissions` for Claude.",
                            agent,
                            cleaned.join(" ")
                        )
                    } else {
                        format!(
                            "Saved default flags for {}:\n`{}`",
                            agent,
                            cleaned.join(" ")
                        )
                    };
                    let _ = bot.send_message(msg.chat.id, saved_message).await;
                }
                Err(err) => {
                    warn!(error = %err, "Failed to save /agent defaults");
                    let _ = bot
                        .send_message(msg.chat.id, "Failed to save defaults.")
                        .await;
                }
            }
            return;
        }

        if subcommand == "clear" {
            let namespace = self.session_namespace().await;
            let mut defaults = cli_agent_flags::load_terminal_agent_defaults(
                &*self.state,
                &namespace,
                msg.chat.id.0,
                user_id,
            )
            .await;
            if defaults.is_empty() {
                let _ = bot.send_message(msg.chat.id, "No saved defaults.").await;
                return;
            }

            let target = args
                .first()
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_else(|| "all".to_string());
            if target == "all" {
                defaults.clear();
                if cli_agent_flags::save_terminal_agent_defaults(
                    &*self.state,
                    &namespace,
                    msg.chat.id.0,
                    user_id,
                    &defaults,
                )
                .await
                .is_ok()
                {
                    let _ = bot
                        .send_message(msg.chat.id, "Cleared all agent defaults.")
                        .await;
                } else {
                    let _ = bot
                        .send_message(msg.chat.id, "Failed to clear defaults.")
                        .await;
                }
                return;
            }

            let Some(agent) = cli_agent_flags::normalize_terminal_agent_name(&target) else {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Usage: /agent defaults clear [agent|all]\nAgents: codex/claude/gemini/opencode",
                    )
                    .await;
                return;
            };
            if defaults.remove(&agent).is_none() {
                let _ = bot
                    .send_message(msg.chat.id, format!("No saved defaults for {}.", agent))
                    .await;
                return;
            }
            if cli_agent_flags::save_terminal_agent_defaults(
                &*self.state,
                &namespace,
                msg.chat.id.0,
                user_id,
                &defaults,
            )
            .await
            .is_ok()
            {
                let _ = bot
                    .send_message(msg.chat.id, format!("Cleared defaults for {}.", agent))
                    .await;
            } else {
                let _ = bot
                    .send_message(msg.chat.id, "Failed to clear defaults.")
                    .await;
            }
            return;
        }

        let namespace = self.session_namespace().await;
        let defaults = cli_agent_flags::load_terminal_agent_defaults(
            &*self.state,
            &namespace,
            msg.chat.id.0,
            user_id,
        )
        .await;
        if defaults.is_empty() {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    "No saved agent defaults yet.\nUse `/agent defaults set <agent> [flags...]`.",
                )
                .await;
            return;
        }
        let mut lines = vec!["Saved agent defaults for this chat:".to_string()];
        for agent in SUPPORTED_TERMINAL_AGENTS {
            if let Some(args) = defaults.get(*agent) {
                lines.push(format!("- {}: `{}`", agent, args.join(" ")));
            }
        }
        lines.push("Clear with `/agent defaults clear [agent|all]`.".to_string());
        let _ = bot.send_message(msg.chat.id, lines.join("\n")).await;
    }

    async fn handle_terminal_lite_command(
        &self,
        args: Vec<String>,
        msg: &super::telegram_api::Message,
        bot: &Bot,
        user_id: u64,
    ) {
        let reply = self
            .terminal_lite
            .start_session(msg.chat.id.0, user_id, args)
            .await;
        let _ = bot.send_message(msg.chat.id, reply).await;
    }

    async fn handle_terminal_lite_input(
        &self,
        chat_id: i64,
        user_id: u64,
        user_role: UserRole,
        text: &str,
    ) -> Option<String> {
        self.terminal_lite
            .handle_input(chat_id, user_id, user_role, text)
            .await
    }

    async fn handle_terminal_command(
        &self,
        arg: &str,
        msg: &super::telegram_api::Message,
        bot: &Bot,
        invoked_cmd: &str,
    ) {
        let user_id = msg.from.as_ref().map(|u| u.id.0).unwrap_or(0);
        if determine_role(&self.owner_user_ids, user_id) != UserRole::Owner {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    format!("Only the owner can use {} in this chat.", invoked_cmd),
                )
                .await;
            return;
        }

        let trimmed = arg.trim();
        let mut tokens = if trimmed.is_empty() {
            Vec::new()
        } else {
            shell_words::split(trimmed).unwrap_or_else(|_| {
                trimmed
                    .split_whitespace()
                    .map(std::string::ToString::to_string)
                    .collect()
            })
        };

        if invoked_cmd == "/agent"
            && matches!(
                tokens.first().map(|s| s.to_ascii_lowercase()).as_deref(),
                Some("defaults")
            )
        {
            tokens.remove(0);
            self.handle_agent_defaults_command(tokens, msg, bot, user_id)
                .await;
            return;
        }

        if invoked_cmd == "/agent"
            && matches!(
                tokens.first().map(|s| s.to_ascii_lowercase()).as_deref(),
                Some("flags")
            )
        {
            tokens.remove(0);
            self.handle_agent_flags_command(tokens, msg, bot, user_id)
                .await;
            return;
        }

        if matches!(
            tokens.first().map(|s| s.to_ascii_lowercase()).as_deref(),
            Some("lite")
        ) {
            if invoked_cmd == "/agent" {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        "Lite mode stays under `/terminal lite`.\nUse `/agent ...` for full Codex/Claude/Gemini sessions.",
                    )
                    .await;
                return;
            }
            tokens.remove(0);
            self.handle_terminal_lite_command(tokens, msg, bot, user_id)
                .await;
            return;
        }

        if let Some(first) = tokens.first() {
            let first = first.to_ascii_lowercase();
            if first == "help" {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        if invoked_cmd == "/agent" {
                            Self::agent_help_text()
                        } else {
                            Self::terminal_help_text()
                        },
                    )
                    .await;
                return;
            }
            if matches!(first.as_str(), "status" | "interrupt" | "stop") {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!(
                            "Use the Mini App toolbar for status/interrupt/stop in v1.\nRun {} open to reconnect.",
                            invoked_cmd
                        ),
                    )
                    .await;
                return;
            }
        }

        if invoked_cmd == "/agent"
            && matches!(
                tokens.first().map(|s| s.to_ascii_lowercase()).as_deref(),
                Some("resume")
            )
        {
            tokens.remove(0);
            let code = tokens
                .first()
                .map(|v| v.trim())
                .filter(|v| !v.is_empty())
                .map(|v| v.to_string());
            self.send_agent_resume_prompt(bot, msg.chat.id, user_id, code)
                .await;
            return;
        }

        if invoked_cmd == "/agent"
            && matches!(
                tokens.first().map(|s| s.to_ascii_lowercase()).as_deref(),
                Some("share")
            )
        {
            tokens.remove(0);
            let relay_session_id = tokens
                .first()
                .map(|v| v.trim())
                .filter(|v| !v.is_empty())
                .map(|v| v.to_string());
            self.send_agent_share_code(bot, msg.chat.id, user_id, relay_session_id)
                .await;
            return;
        }

        let mut start_requested = false;
        if let Some(first) = tokens.first() {
            let first = first.to_ascii_lowercase();
            if first == "open" || first == "start" {
                start_requested = first == "start";
                tokens.remove(0);
            }
        }

        let mut agent: Option<String> = None;
        let mut cwd_parts: Vec<String> = Vec::new();
        let mut agent_args: Vec<String> = Vec::new();
        let mut rewrote_permission_flag = false;
        let mut had_explicit_arg_delimiter = false;
        let mut used_saved_args = false;
        let mut saved_args_updated = false;

        // Backward-compat: `/terminal ... -- [flags...]`
        if let Some(idx) = tokens.iter().position(|t| t == "--") {
            had_explicit_arg_delimiter = true;
            if idx + 1 < tokens.len() {
                agent_args.extend(
                    tokens[(idx + 1)..]
                        .iter()
                        .map(|value| value.trim().to_string())
                        .filter(|value| !value.is_empty() && !value.contains('\0')),
                );
            }
            tokens.truncate(idx);
        }

        if let Some(first) = tokens.first() {
            let candidate = first.to_ascii_lowercase();
            if SUPPORTED_TERMINAL_AGENTS.contains(&candidate.as_str()) {
                agent = Some(candidate);
                tokens.remove(0);
            } else if first.starts_with('/')
                || first.starts_with('~')
                || first.starts_with('.')
                || first.contains('/')
            {
                agent = Some(DEFAULT_TERMINAL_AGENT.to_string());
            } else if first.starts_with('-') {
                // Flags-only launch defaults to codex.
                agent = Some(DEFAULT_TERMINAL_AGENT.to_string());
            } else {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!(
                            "Unknown terminal agent: `{}`\n\n{}",
                            first,
                            if invoked_cmd == "/agent" {
                                Self::agent_help_text()
                            } else {
                                Self::terminal_help_text()
                            }
                        ),
                    )
                    .await;
                return;
            }
            if had_explicit_arg_delimiter {
                for token in tokens {
                    let trimmed = token.trim();
                    if trimmed.is_empty() || trimmed.contains('\0') {
                        continue;
                    }
                    cwd_parts.push(trimmed.to_string());
                }
            } else if let Some(flag_idx) = tokens.iter().position(|token| token.starts_with('-')) {
                for token in &tokens[..flag_idx] {
                    let trimmed = token.trim();
                    if trimmed.is_empty() || trimmed.contains('\0') {
                        continue;
                    }
                    cwd_parts.push(trimmed.to_string());
                }
                for token in &tokens[flag_idx..] {
                    let trimmed = token.trim();
                    if trimmed.is_empty() || trimmed.contains('\0') {
                        continue;
                    }
                    agent_args.push(trimmed.to_string());
                }
            } else {
                for token in tokens {
                    let trimmed = token.trim();
                    if trimmed.is_empty() || trimmed.contains('\0') {
                        continue;
                    }
                    cwd_parts.push(trimmed.to_string());
                }
            }
        } else if start_requested {
            agent = Some(DEFAULT_TERMINAL_AGENT.to_string());
        } else if !agent_args.is_empty() {
            // Flags imply a terminal launch context; default agent to codex.
            agent = Some(DEFAULT_TERMINAL_AGENT.to_string());
        }
        agent_args = cli_agent_flags::normalize_terminal_agent_args(agent_args);
        let skip_saved_defaults = cli_agent_flags::strip_no_default_flag(&mut agent_args);
        let (normalized_agent_args, rewrote) =
            crate::normalize_terminal_agent_permission_aliases(agent.as_deref(), agent_args);
        agent_args = normalized_agent_args;
        rewrote_permission_flag |= rewrote;

        let namespace = self.session_namespace().await;
        if invoked_cmd == "/agent" && agent_args.is_empty() && !skip_saved_defaults {
            if let Some(agent_name) = agent.as_deref() {
                let defaults = cli_agent_flags::load_terminal_agent_defaults(
                    &*self.state,
                    &namespace,
                    msg.chat.id.0,
                    user_id,
                )
                .await;
                if let Some(saved) = defaults.get(agent_name) {
                    agent_args = cli_agent_flags::normalize_terminal_agent_args(saved.clone());
                    let (normalized_agent_args, rewrote) =
                        crate::normalize_terminal_agent_permission_aliases(
                            agent.as_deref(),
                            agent_args,
                        );
                    agent_args = normalized_agent_args;
                    rewrote_permission_flag |= rewrote;
                    used_saved_args = !agent_args.is_empty();
                }
            }
        } else if invoked_cmd == "/agent" && !agent_args.is_empty() {
            if let Some(agent_name) = agent.as_deref() {
                let mut defaults = cli_agent_flags::load_terminal_agent_defaults(
                    &*self.state,
                    &namespace,
                    msg.chat.id.0,
                    user_id,
                )
                .await;
                let changed = defaults.get(agent_name) != Some(&agent_args);
                if changed {
                    defaults.insert(agent_name.to_string(), agent_args.clone());
                    if let Err(err) = cli_agent_flags::save_terminal_agent_defaults(
                        &*self.state,
                        &namespace,
                        msg.chat.id.0,
                        user_id,
                        &defaults,
                    )
                    .await
                    {
                        warn!(error = %err, "Failed to persist /agent default flags");
                    } else {
                        saved_args_updated = true;
                    }
                }
            }
        }

        let cwd = if cwd_parts.is_empty() {
            None
        } else {
            Some(cwd_parts.join(" "))
        };
        let has_launch_intent = agent.is_some() || cwd.is_some() || !agent_args.is_empty();
        #[cfg(not(feature = "terminal-bridge"))]
        let _ = has_launch_intent;

        let current_bot_label = self.cached_bot_label();
        let mini_app_sync_warning = match self
            .sync_terminal_tenant_bot_token(&self.bot_token, user_id, current_bot_label.as_deref())
            .await
        {
            Ok(()) => None,
            Err(err) => {
                warn!(
                    error = %err,
                    user_id,
                    invoked_cmd,
                    "Failed to auto-sync Telegram bot token before opening Mini App"
                );
                Some(
                    "Mini App auth sync did not complete. If Telegram shows an auth mismatch, retry once or save the token in the app."
                        .to_string(),
                )
            }
        };

        #[cfg(feature = "terminal-bridge")]
        let prestarted_relay_session_id = if invoked_cmd == "/agent" && has_launch_intent {
            match crate::terminal_bridge::request_local_start_session(
                agent.as_deref().unwrap_or(DEFAULT_TERMINAL_AGENT),
                cwd.as_deref(),
                &agent_args,
            )
            .await
            {
                Ok(session_id) => Some(session_id),
                Err(err) => {
                    warn!(
                        error = %err,
                        chat_id = msg.chat.id.0,
                        invoked_cmd,
                        "Failed to pre-start local bridge session for Telegram Mini App"
                    );
                    None
                }
            }
        } else {
            None
        };
        #[cfg(not(feature = "terminal-bridge"))]
        let prestarted_relay_session_id: Option<String> = None;

        let mut web_app_url = match reqwest::Url::parse(self.terminal_web_app_url.trim()) {
            Ok(url) => url,
            Err(err) => {
                let _ = bot
                    .send_message(
                        msg.chat.id,
                        format!(
                            "Terminal web app URL is invalid in config: {}\nCurrent value: {}",
                            err, self.terminal_web_app_url
                        ),
                    )
                    .await;
                return;
            }
        };

        if web_app_url.scheme() != "https" {
            let _ = bot
                .send_message(
                    msg.chat.id,
                    format!(
                        "Terminal web app URL must use HTTPS. Current value: {}",
                        self.terminal_web_app_url
                    ),
                )
                .await;
            return;
        }

        let telegram_session_id = self.session_id(msg.chat.id.0).await;
        {
            let mut query = web_app_url.query_pairs_mut();
            query.append_pair("telegram_session_id", &telegram_session_id);
            if let Some(agent) = agent.as_deref() {
                query.append_pair("agent", agent);
            }
            if let Some(cwd) = cwd.as_deref() {
                if !cwd.trim().is_empty() {
                    query.append_pair("cwd", cwd);
                }
            }
            if let Some(relay_session_id) = prestarted_relay_session_id.as_deref() {
                query.append_pair("relay_session_id", relay_session_id);
            }
            for arg in &agent_args {
                query.append_pair("arg", arg);
            }
            if start_requested {
                query.append_pair("autostart", "1");
            }
        }

        let mini_app_host = web_app_url.host_str().unwrap_or("unknown");
        let mini_app_base = if let Some(port) = web_app_url.port() {
            format!("{}://{}:{}", web_app_url.scheme(), mini_app_host, port)
        } else {
            format!("{}://{}", web_app_url.scheme(), mini_app_host)
        };

        let mut summary_lines = vec![
            if invoked_cmd == "/agent" {
                "🤖 <b>Agent Session</b>".to_string()
            } else {
                "🖥️ <b>Terminal Mode</b>".to_string()
            },
            String::new(),
            format!(
                "Mini App host: <code>{}</code>",
                html_escape(&mini_app_base)
            ),
        ];
        if let Some(agent) = agent.as_deref() {
            summary_lines.push(format!("Agent: <code>{}</code>", html_escape(agent)));
        } else {
            summary_lines.push(format!(
                "Agent: choose in app (default {})",
                DEFAULT_TERMINAL_AGENT
            ));
        }
        if let Some(cwd) = cwd.as_deref() {
            let folder_name = Path::new(cwd)
                .file_name()
                .and_then(|value| value.to_str())
                .filter(|value| !value.is_empty())
                .unwrap_or("custom");
            summary_lines.push(format!(
                "Working dir: <code>{}</code> (full path sent only to Mini App)",
                html_escape(folder_name)
            ));
        }
        if !agent_args.is_empty() {
            summary_lines.push(format!(
                "Agent args: <code>{}</code> (values sent only to Mini App)",
                agent_args.len()
            ));
            if used_saved_args {
                summary_lines.push(
                    "Using saved defaults for this chat. Add <code>--no-default-flags</code> to bypass once."
                        .to_string(),
                );
            } else if saved_args_updated {
                summary_lines.push("Saved as new defaults for this chat and agent.".to_string());
            }
            if rewrote_permission_flag {
                summary_lines.push(
                    "Normalized <code>--allow-dangerously-skip-permissions</code> to <code>--dangerously-skip-permissions</code> for Claude."
                        .to_string(),
                );
            }
        }
        if prestarted_relay_session_id.is_some() {
            summary_lines.push(
                "Prepared a local relay session in advance so the Mini App can attach immediately."
                    .to_string(),
            );
        }
        summary_lines.push(String::new());
        summary_lines.push(
            if invoked_cmd == "/agent" {
                "Tap 📱 Open in Mini App to launch the encrypted agent UI."
            } else {
                "Tap 🖥️ Open Terminal to launch the encrypted session UI."
            }
            .to_string(),
        );
        if let Some(warning) = mini_app_sync_warning.as_deref() {
            summary_lines.push(warning.to_string());
        }
        if invoked_cmd == "/agent" {
            summary_lines.push(
                "💻 Continue on Computer sends a one-time resume code (expires in about 5 minutes)."
                    .to_string(),
            );
        }

        let open_button = InlineKeyboardButton::web_app(
            if invoked_cmd == "/agent" {
                "📱 Open in Mini App"
            } else {
                "🖥️ Open Terminal"
            },
            WebAppInfo {
                url: web_app_url.clone(),
            },
        );
        let keyboard = if invoked_cmd == "/agent" {
            InlineKeyboardMarkup::new(vec![vec![
                open_button,
                InlineKeyboardButton::callback("💻 Continue on Computer", "agent:share"),
            ]])
        } else {
            InlineKeyboardMarkup::new(vec![vec![open_button]])
        };

        let html_message = summary_lines.join("\n");
        let plain_message = format!(
            "{}\nMini App host: {}\n\nUse the Open button to launch. Continue on Computer generates a one-time code that expires in about 5 minutes.",
            if invoked_cmd == "/agent" {
                "Agent session"
            } else {
                "Terminal mode"
            },
            mini_app_base
        );

        if bot
            .send_message(msg.chat.id, html_message)
            .parse_mode(ParseMode::Html)
            .reply_markup(keyboard.clone())
            .await
            .is_err()
        {
            let _ = bot
                .send_message(msg.chat.id, plain_message)
                .reply_markup(keyboard)
                .await;
        }

        if invoked_cmd == "/agent" {
            let reply_keyboard =
                KeyboardMarkup::new(vec![vec![KeyboardButton::new("📱 Open in Mini App")
                    .request(ButtonRequest::WebApp(WebAppInfo {
                        url: web_app_url.clone(),
                    }))]])
                .resize_keyboard()
                .one_time_keyboard();

            let _ = bot
                .send_message(
                    msg.chat.id,
                    "Tip: if in-app handoff buttons look disabled, open from this keyboard button. Telegram only enables Mini App send-back in keyboard launch mode.",
                )
                .reply_markup(reply_keyboard)
                .await;
        }
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
                        )
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
                if !wrangler_present {
                    if let Some(spec) =
                        crate::wizard::installer_for_missing_tool("wrangler", has_brew, has_npm)
                    {
                        install_specs.push(("wrangler".to_string(), spec));
                    }
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
                            )
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
                            )
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
                                )
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
                        )
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
                ))
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
                    ))
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
                    ))
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
            .sync_terminal_tenant_bot_token(token, user_id, Some(bot_username.as_str()))
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
        if let Some(weak_hub) = hub_ref {
            if let Some(hub) = weak_hub.upgrade() {
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
                    self.terminal_web_app_url.clone(),
                    self.terminal_lite.allowed_prefixes(),
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
        if let Some(weak_hub) = hub_ref {
            if let Some(hub) = weak_hub.upgrade() {
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
        if let Some(weak_hub) = hub_ref {
            if let Some(hub) = weak_hub.upgrade() {
                // Create the new Slack channel with same config as this Telegram channel
                let new_channel = Arc::new(SlackChannel::new(
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

    async fn build_terminal_attach_handoff_message(
        &self,
        relay_session_id: &str,
        user_id: u64,
    ) -> anyhow::Result<String> {
        let handoff = crate::agent_handoff::create_handoff_code(
            self.state.as_ref(),
            relay_session_id,
            user_id,
        )
        .await?;
        let command = format!("aidaemon attach {}", handoff.code);
        let shell_token = |value: &str| -> String {
            if !value.is_empty()
                && value.chars().all(|ch| {
                    ch.is_ascii_alphanumeric()
                        || matches!(ch, '_' | '-' | '.' | '/' | ':' | '=' | '+' | ',' | '@')
                })
            {
                value.to_string()
            } else {
                format!("'{}'", value.replace('\'', r"'\''"))
            }
        };
        let exact_command = std::env::current_exe().ok().and_then(|exe| {
            let exe_text = exe.to_string_lossy().trim().to_string();
            if exe_text.is_empty() {
                None
            } else {
                Some(format!(
                    "{} attach {}",
                    shell_token(&exe_text),
                    handoff.code
                ))
            }
        });
        let command_section = if let Some(exact) = exact_command {
            format!(
                "Run this on your computer:\n\
                 <pre>{}</pre>\n\n\
                 If <code>aidaemon</code> points to an older install, run:\n\
                 <pre>{}</pre>\n\n",
                html_escape(&command),
                html_escape(&exact)
            )
        } else {
            format!(
                "Run this on your computer:\n\
                 <pre>{}</pre>\n\n",
                html_escape(&command)
            )
        };
        Ok(format!(
            "🖥️ <b>Continue In Native Terminal</b>\n\n\
             {}\
             Session: <code>{}</code>\n\
             Expires in about 5 minutes. Code is one-time use.",
            command_section,
            html_escape(relay_session_id)
        ))
    }

    async fn resolve_continue_relay_session_id(
        &self,
        chat_id: i64,
        relay_session_id: Option<String>,
    ) -> Option<String> {
        if let Some(value) = relay_session_id
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
        {
            return Some(value);
        }

        let telegram_session_id = self.session_id(chat_id).await;
        match crate::agent_handoff::resolve_relay_for_telegram_session(
            self.state.as_ref(),
            &telegram_session_id,
        )
        .await
        {
            Ok(Some(value)) => Some(value),
            Ok(None) => None,
            Err(err) => {
                warn!(
                    error = %err,
                    telegram_session_id = %telegram_session_id,
                    "Failed to resolve relay mapping for continue-on-computer"
                );
                None
            }
        }
    }

    async fn handle_continue_on_computer_action(
        &self,
        bot: &Bot,
        msg: &super::telegram_api::Message,
        user_id: u64,
        relay_session_id: Option<String>,
    ) {
        self.send_agent_share_code(bot, msg.chat.id, user_id, relay_session_id)
            .await;
    }

    async fn send_agent_share_code(
        &self,
        bot: &Bot,
        chat_id: ChatId,
        user_id: u64,
        relay_session_id: Option<String>,
    ) {
        let resolved_relay = if let Some(value) = relay_session_id
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
        {
            Some(value)
        } else {
            let mapped = self
                .resolve_continue_relay_session_id(chat_id.0, None)
                .await;
            if mapped.is_some() {
                mapped
            } else {
                match crate::agent_handoff::get_last_active_relay_session_id(
                    self.state.as_ref(),
                    user_id,
                )
                .await
                {
                    Ok(value) => value,
                    Err(err) => {
                        warn!(
                            error = %err,
                            "Failed to load last active relay session id for /agent share"
                        );
                        None
                    }
                }
            }
        };

        let Some(relay_session_id) = resolved_relay else {
            let _ = bot
                .send_message(
                    chat_id,
                    "Could not identify the active terminal session. Keep the Mini App session open and try Continue on Computer again.",
                )
                .await;
            return;
        };

        match self
            .build_terminal_attach_handoff_message(&relay_session_id, user_id)
            .await
        {
            Ok(reply) => {
                let _ = bot
                    .send_message(chat_id, reply)
                    .parse_mode(ParseMode::Html)
                    .await;
            }
            Err(err) => {
                warn!(error = %err, "Failed to create terminal attach handoff code");
                let _ = bot
                    .send_message(
                        chat_id,
                        "Failed to create continue-on-computer code. Please try again.",
                    )
                    .await;
            }
        }
    }

    async fn send_agent_resume_prompt(
        &self,
        bot: &Bot,
        chat_id: ChatId,
        user_id: u64,
        code: Option<String>,
    ) {
        let Some(code) = code.map(|v| v.trim().to_string()).filter(|v| !v.is_empty()) else {
            let _ = bot
                .send_message(chat_id, "Usage: /agent resume <code>")
                .await;
            return;
        };

        let handoff = match crate::agent_handoff::resolve_handoff_code(self.state.as_ref(), &code)
            .await
        {
            Ok(value) => value,
            Err(err) => {
                let _ = bot
                        .send_message(
                            chat_id,
                            format!(
                                "Resume code is invalid or expired: {}. Generate a fresh code from your computer with `aidaemon share`.",
                                err
                            ),
                        )
                        .await;
                return;
            }
        };

        if handoff.owner_user_id != user_id {
            let _ = bot
                .send_message(
                    chat_id,
                    "That resume code belongs to a different Telegram user.",
                )
                .await;
            return;
        }

        if let Err(err) =
            crate::agent_handoff::consume_handoff_code(self.state.as_ref(), &code).await
        {
            let _ = bot
                .send_message(
                    chat_id,
                    format!("Resume code could not be consumed: {}.", err),
                )
                .await;
            return;
        }

        let relay_session_id = handoff.relay_session_id.trim().to_string();
        let telegram_session_id = self.session_id(chat_id.0).await;
        if let Err(err) = crate::agent_handoff::bind_telegram_session_to_relay(
            self.state.as_ref(),
            &telegram_session_id,
            &relay_session_id,
        )
        .await
        {
            warn!(
                error = %err,
                relay_session_id = %relay_session_id,
                telegram_session_id = %telegram_session_id,
                "Failed to bind relay session from /agent resume"
            );
            let _ = bot
                .send_message(
                    chat_id,
                    "Failed to bind the resume session. Please generate a new code and retry.",
                )
                .await;
            return;
        }

        let current_bot_label = self.cached_bot_label();
        let mini_app_sync_warning = match self
            .sync_terminal_tenant_bot_token(&self.bot_token, user_id, current_bot_label.as_deref())
            .await
        {
            Ok(()) => None,
            Err(err) => {
                warn!(
                    error = %err,
                    user_id,
                    relay_session_id = %relay_session_id,
                    "Failed to auto-sync Telegram bot token before /agent resume"
                );
                Some(
                    "If Telegram reports an auth mismatch, retry once or save the token manually in the Mini App."
                        .to_string(),
                )
            }
        };

        let mut web_app_url = match reqwest::Url::parse(&self.terminal_web_app_url) {
            Ok(url) => url,
            Err(err) => {
                warn!(error = %err, "Invalid terminal_web_app_url during /agent resume");
                let _ = bot
                    .send_message(
                        chat_id,
                        "Resume code accepted. Run `/agent open` to continue in Mini App.",
                    )
                    .await;
                return;
            }
        };
        {
            let mut query = web_app_url.query_pairs_mut();
            query.append_pair("telegram_session_id", &telegram_session_id);
            query.append_pair("relay_session_id", &relay_session_id);
            query.append_pair("autostart", "1");
        }

        let keyboard = InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::web_app(
                "📱 Open in Mini App",
                WebAppInfo {
                    url: web_app_url.clone(),
                },
            ),
            InlineKeyboardButton::callback(
                "💻 Continue on Computer",
                format!("agent:share:{}", relay_session_id),
            ),
        ]]);

        let _ = bot
            .send_message(
                chat_id,
                format!(
                    "✅ <b>Resume Code Accepted</b>\n\nSession: <code>{}</code>\nTap Open in Mini App to continue on your phone.{}",
                    html_escape(&relay_session_id),
                    mini_app_sync_warning
                        .as_deref()
                        .map(|value| format!("\n\n{}", html_escape(value)))
                        .unwrap_or_default()
                ),
            )
            .parse_mode(ParseMode::Html)
            .reply_markup(keyboard)
            .await;

        let reply_keyboard =
            KeyboardMarkup::new(vec![vec![KeyboardButton::new("📱 Open in Mini App")
                .request(ButtonRequest::WebApp(WebAppInfo {
                    url: web_app_url.clone(),
                }))]])
            .resize_keyboard()
            .one_time_keyboard();

        let _ = bot
            .send_message(
                chat_id,
                "Tip: use this keyboard launch if Mini App send-back actions are unavailable in inline mode.",
            )
            .reply_markup(reply_keyboard)
            .await;
    }
}

#[cfg(test)]
#[path = "telegram_sdk_tests.rs"]
mod sdk_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_bot_sync_response_accepts_success_json() {
        let (result, retryable) = interpret_terminal_bot_sync_response(
            reqwest::StatusCode::OK,
            Some("application/json"),
            br#"{"ok":true}"#,
        );
        assert!(result.is_ok());
        assert!(!retryable);
    }

    #[test]
    fn terminal_bot_sync_response_classifies_transient_non_json() {
        let (result, retryable) = interpret_terminal_bot_sync_response(
            reqwest::StatusCode::BAD_GATEWAY,
            Some("text/html"),
            b"upstream unavailable",
        );
        let error = result.unwrap_err();
        assert!(error.contains("status 502 Bad Gateway"));
        assert!(error.contains("content-type text/html"));
        assert!(retryable);
    }

    #[test]
    fn terminal_bot_sync_response_does_not_retry_auth_rejection() {
        let (result, retryable) = interpret_terminal_bot_sync_response(
            reqwest::StatusCode::UNAUTHORIZED,
            Some("application/json"),
            br#"{"ok":false,"error":"invalid proof"}"#,
        );
        assert!(result.unwrap_err().contains("invalid proof"));
        assert!(!retryable);
    }

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
        if let Some(bots) = doc.get_mut("telegram_bots").and_then(|v| v.as_array_mut()) {
            if let Some(first) = bots.first_mut().and_then(|v| v.as_table_mut()) {
                first.insert("allowed_user_ids".to_string(), ids_toml);
            }
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
