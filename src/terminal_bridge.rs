//! The daemon's side of the terminal bridge (`aidaemon-terminal-bridge`):
//! its settings from `AppConfig`, and the host backed by the state store
//! (runtime bots, the relay handoff records in `agent_handoff`).

use std::sync::Arc;

pub use aidaemon_terminal_bridge::{
    request_local_start_session, run_local_attach_cli, run_local_share_cli, run_local_start_cli,
};
use aidaemon_terminal_bridge::{
    AttachHandoff, BridgeHost, BridgeSettings, ReviewAgentSettings, RuntimeBot,
};
use async_trait::async_trait;

use crate::config::AppConfig;
use crate::traits::StateStore;

/// Starts the bridge when the config allows it (see
/// [`aidaemon_terminal_bridge::spawn_if_configured`]).
pub fn spawn_if_configured(config: &AppConfig, state: Arc<dyn StateStore>) -> bool {
    aidaemon_terminal_bridge::spawn_if_configured(
        bridge_settings(config),
        Arc::new(StateBridgeHost { state }),
        crate::config::shared_secret_store(),
    )
}

fn bridge_settings(config: &AppConfig) -> BridgeSettings {
    BridgeSettings {
        enabled: config.terminal.effective_bridge_enabled(),
        user_id: resolve_daemon_user_id(config, config.terminal.effective_daemon_user_id()),
        ws_url: config.terminal.effective_daemon_ws_url(),
        shell: config.terminal.effective_daemon_shell(),
        inbox_dir: config.files.inbox_dir.clone(),
        max_upload_bytes: if config.files.enabled {
            config.files.max_file_size_mb.saturating_mul(1_048_576)
        } else {
            0
        },
        device_id: config.terminal.effective_daemon_device_id(),
        bot_tokens: resolve_daemon_bot_tokens(config),
        static_connect_token: config.terminal.effective_daemon_connect_token(),
        allow_static_token_fallback: config.terminal.effective_allow_static_token_fallback(),
        review_agents: config
            .cli_agents
            .tools
            .iter()
            .map(|(name, tool)| ReviewAgentSettings {
                name: name.clone(),
                command: tool.command.clone(),
                args: tool.args.clone(),
                timeout_secs: tool.timeout_secs,
                max_output_chars: tool.max_output_chars,
            })
            .collect(),
    }
}

/// The configured daemon user id, else the first numeric Telegram owner id,
/// else the first allowed user of a configured Telegram bot.
fn resolve_daemon_user_id(config: &AppConfig, explicit: Option<u64>) -> Option<u64> {
    if let Some(id) = explicit {
        return Some(id);
    }

    if let Some(owner_ids) = config.users.owner_ids.get("telegram") {
        for raw in owner_ids {
            if let Ok(id) = raw.parse::<u64>() {
                return Some(id);
            }
        }
    }

    config
        .all_telegram_bots()
        .iter()
        .find_map(|bot| bot.allowed_user_ids.first().copied())
}

fn resolve_daemon_bot_tokens(config: &AppConfig) -> Vec<String> {
    let mut out = Vec::new();
    for bot in config.all_telegram_bots() {
        let token = bot.bot_token.trim();
        if token.is_empty() {
            continue;
        }
        if !out.iter().any(|existing: &String| existing == token) {
            out.push(token.to_string());
        }
    }
    out
}

/// The bridge's host on the daemon's state store.
struct StateBridgeHost {
    state: Arc<dyn StateStore>,
}

fn attach_handoff(handoff: crate::agent_handoff::TerminalAttachHandoff) -> AttachHandoff {
    AttachHandoff {
        code: handoff.code,
        relay_session_id: handoff.relay_session_id,
        expires_at_unix: handoff.expires_at_unix,
    }
}

#[async_trait]
impl BridgeHost for StateBridgeHost {
    async fn runtime_bots(&self) -> anyhow::Result<Vec<RuntimeBot>> {
        Ok(self
            .state
            .get_dynamic_bots()
            .await?
            .into_iter()
            .map(|bot| RuntimeBot {
                channel_type: bot.channel_type,
                bot_token: bot.bot_token,
                allowed_user_ids: bot.allowed_user_ids,
            })
            .collect())
    }

    async fn create_handoff_code(
        &self,
        relay_session_id: &str,
        owner_user_id: u64,
    ) -> anyhow::Result<AttachHandoff> {
        crate::agent_handoff::create_handoff_code(
            self.state.as_ref(),
            relay_session_id,
            owner_user_id,
        )
        .await
        .map(attach_handoff)
    }

    async fn resolve_handoff_code(&self, code: &str) -> anyhow::Result<AttachHandoff> {
        crate::agent_handoff::resolve_handoff_code(self.state.as_ref(), code)
            .await
            .map(attach_handoff)
    }

    async fn consume_handoff_code(&self, code: &str) -> anyhow::Result<AttachHandoff> {
        crate::agent_handoff::consume_handoff_code(self.state.as_ref(), code)
            .await
            .map(attach_handoff)
    }

    async fn set_last_active_relay_session_id(
        &self,
        owner_user_id: u64,
        relay_session_id: &str,
    ) -> anyhow::Result<()> {
        crate::agent_handoff::set_last_active_relay_session_id(
            self.state.as_ref(),
            owner_user_id,
            relay_session_id,
        )
        .await
    }

    async fn get_last_active_relay_session_id(
        &self,
        owner_user_id: u64,
    ) -> anyhow::Result<Option<String>> {
        crate::agent_handoff::get_last_active_relay_session_id(self.state.as_ref(), owner_user_id)
            .await
    }

    async fn bind_telegram_session_to_relay(
        &self,
        telegram_session_id: &str,
        relay_session_id: &str,
    ) -> anyhow::Result<()> {
        crate::agent_handoff::bind_telegram_session_to_relay(
            self.state.as_ref(),
            telegram_session_id,
            relay_session_id,
        )
        .await
    }

    async fn resolve_relay_for_telegram_session(
        &self,
        telegram_session_id: &str,
    ) -> anyhow::Result<Option<String>> {
        crate::agent_handoff::resolve_relay_for_telegram_session(
            self.state.as_ref(),
            telegram_session_id,
        )
        .await
    }
}
