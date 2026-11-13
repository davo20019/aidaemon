//! The Telegram terminal surface: `/agent` (CLI agent sessions in the
//! terminal Mini App, saved flags and defaults, share and resume codes),
//! `/terminal` (the Mini App, or terminal lite: a shell in the chat) and
//! "Continue on Computer". The host gives it a settings store, the Mini App
//! URL and, when the terminal bridge runs, a way to pre-start a local session.

pub mod bootstrap;
pub mod handoff;

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aidaemon_channels::slash::{CommandCategory, CommandDef};
use aidaemon_channels::terminal_lite::TerminalLiteManager;
use aidaemon_core::UserRole;
use aidaemon_tools::cli_agent_flags::{
    self, AGENT_FLAGS_CACHE_TTL_SECS, SUPPORTED_TERMINAL_AGENTS,
};
use aidaemon_tools::permission_bypass::normalize_terminal_agent_permission_aliases;
use aidaemon_tools::services::SettingsStore;
use async_trait::async_trait;
use tracing::warn;

use crate::bot::{determine_role, send_markdown_chunks_or_fallback, TelegramBot};
use crate::bot_api::{
    Bot, ButtonRequest, CallbackQuery, ChatId, InlineKeyboardButton, InlineKeyboardMarkup,
    KeyboardButton, KeyboardMarkup, Message, ParseMode, WebAppInfo,
};
use crate::format::html_escape;
use bootstrap::{
    interpret_terminal_bot_sync_response, random_terminal_bootstrap_nonce,
    sign_terminal_tenant_bot_bootstrap_proof, terminal_tenant_bot_bootstrap_url,
};

const DEFAULT_TERMINAL_AGENT: &str = "codex";

/// Starts a CLI agent session on this computer ahead of the Mini App, so it
/// can attach at once (the terminal bridge's local start).
#[async_trait]
pub trait LocalAgentStart: Send + Sync {
    /// Starts `agent` in `cwd` with `args`; returns the relay session id.
    async fn start_session(
        &self,
        agent: &str,
        cwd: Option<&str>,
        args: &[String],
    ) -> anyhow::Result<String>;
}

/// The terminal surface of one Telegram bot.
pub struct TelegramTerminal {
    settings: Arc<dyn SettingsStore>,
    /// URL of the Telegram Mini App terminal frontend.
    web_app_url: String,
    lite: TerminalLiteManager,
    local_start: Option<Arc<dyn LocalAgentStart>>,
}

impl TelegramTerminal {
    /// A terminal surface keeping its records in `settings`. Terminal lite
    /// runs only commands starting with one of `lite_allowed_prefixes`.
    pub fn new(
        settings: Arc<dyn SettingsStore>,
        web_app_url: String,
        lite_allowed_prefixes: Vec<String>,
        local_start: Option<Arc<dyn LocalAgentStart>>,
    ) -> Self {
        let lite_allowed_prefixes = lite_allowed_prefixes
            .into_iter()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .collect::<HashSet<_>>();
        Self {
            settings,
            web_app_url,
            lite: TerminalLiteManager::new(lite_allowed_prefixes),
            local_start,
        }
    }

    /// The Mini App URL.
    pub fn web_app_url(&self) -> &str {
        &self.web_app_url
    }

    /// The command prefixes terminal lite runs.
    pub fn lite_allowed_prefixes(&self) -> Vec<String> {
        self.lite.allowed_prefixes()
    }

    /// Syncs another bot's token to the Mini App tenant for `user_id` (a bot
    /// connected at runtime).
    pub async fn sync_tenant_bot_token(
        &self,
        bot_token: &str,
        user_id: u64,
        label: Option<&str>,
    ) -> Result<(), String> {
        sync_tenant_bot_token(&self.web_app_url, bot_token, user_id, label).await
    }

    /// `/agent` and `/terminal` in the "/" menu.
    pub fn commands() -> Vec<CommandDef> {
        vec![
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

    fn cx<'a>(&'a self, tg: &'a Arc<TelegramBot>) -> Cx<'a> {
        Cx {
            tg,
            state: self.settings.clone(),
            bot_token: tg.bot_token().to_string(),
            allowed_user_ids: tg.allowed_user_ids(),
            owner_user_ids: tg.owner_user_ids().to_vec(),
            terminal_web_app_url: self.web_app_url.clone(),
            terminal_lite: &self.lite,
            local_start: self.local_start.as_deref(),
        }
    }

    /// Handles `/agent` or `/terminal` (`cmd`). Returns false for any other
    /// command.
    pub async fn handle_command(
        &self,
        tg: &Arc<TelegramBot>,
        cmd: &str,
        arg: &str,
        msg: &Message,
    ) -> bool {
        if !matches!(cmd, "/agent" | "/terminal") {
            return false;
        }
        let bot = tg.bot().clone();
        self.cx(tg)
            .handle_terminal_command(arg, msg, &bot, cmd)
            .await;
        true
    }

    /// Handles the "Continue on Computer" button (`agent:share[:<relay>]`).
    /// Returns false for any other callback.
    pub async fn handle_callback(
        &self,
        tg: &Arc<TelegramBot>,
        q: &CallbackQuery,
        data: &str,
        user_id: u64,
    ) -> bool {
        if data != "agent:share" && !data.starts_with("agent:share:") {
            return false;
        }
        let bot = tg.bot().clone();
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
            self.cx(tg)
                .send_agent_share_code(&bot, m.chat.id, user_id, explicit_relay_session_id)
                .await;
        } else {
            let _ = bot
                .answer_callback_query(q.id.clone())
                .text("Open the chat and run /agent share.")
                .show_alert(true)
                .await;
        }
        true
    }

    /// A chat message while terminal lite is on runs as a shell command;
    /// `None` when the chat has no terminal lite session.
    pub async fn intercept_text(
        &self,
        chat_id: i64,
        user_id: u64,
        user_role: UserRole,
        text: &str,
    ) -> Option<String> {
        self.lite
            .handle_input(chat_id, user_id, user_role, text)
            .await
    }

    /// Sends a one-time code to continue the chat's relay session on the
    /// computer.
    pub async fn continue_on_computer(
        &self,
        tg: &Arc<TelegramBot>,
        msg: &Message,
        user_id: u64,
        relay_session_id: Option<String>,
    ) {
        let bot = tg.bot().clone();
        self.cx(tg)
            .send_agent_share_code(&bot, msg.chat.id, user_id, relay_session_id)
            .await;
    }

    /// Syncs the bot token to the Mini App tenant for every allowed user
    /// (at bot start).
    pub async fn sync_bot_token(&self, tg: &Arc<TelegramBot>, label: Option<&str>) {
        self.cx(tg)
            .sync_terminal_tenant_bot_token_for_allowed_users(label)
            .await;
    }
}

/// One bot and the terminal surface, for one command.
struct Cx<'a> {
    tg: &'a Arc<TelegramBot>,
    state: Arc<dyn SettingsStore>,
    bot_token: String,
    allowed_user_ids: &'a StdRwLock<Vec<u64>>,
    owner_user_ids: Vec<u64>,
    terminal_web_app_url: String,
    terminal_lite: &'a TerminalLiteManager,
    local_start: Option<&'a dyn LocalAgentStart>,
}

impl Cx<'_> {
    async fn session_id(&self, chat_id: i64) -> String {
        self.tg.session_id(chat_id).await
    }

    async fn session_namespace(&self) -> String {
        self.tg.session_namespace().await
    }

    fn cached_bot_label(&self) -> Option<String> {
        self.tg.cached_bot_label()
    }

    async fn sync_terminal_tenant_bot_token(
        &self,
        bot_token: &str,
        user_id: u64,
        label: Option<&str>,
    ) -> Result<(), String> {
        sync_tenant_bot_token(&self.terminal_web_app_url, bot_token, user_id, label).await
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
        msg: &Message,
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
        if !refresh
            && let Some(cached) =
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
        msg: &Message,
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
                normalize_terminal_agent_permission_aliases(Some(agent.as_str()), cleaned);
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
        msg: &Message,
        bot: &Bot,
        user_id: u64,
    ) {
        let reply = self
            .terminal_lite
            .start_session(msg.chat.id.0, user_id, args)
            .await;
        let _ = bot.send_message(msg.chat.id, reply).await;
    }

    async fn handle_terminal_command(
        &self,
        arg: &str,
        msg: &Message,
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
            normalize_terminal_agent_permission_aliases(agent.as_deref(), agent_args);
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
                        normalize_terminal_agent_permission_aliases(agent.as_deref(), agent_args);
                    agent_args = normalized_agent_args;
                    rewrote_permission_flag |= rewrote;
                    used_saved_args = !agent_args.is_empty();
                }
            }
        } else if invoked_cmd == "/agent" && !agent_args.is_empty()
            && let Some(agent_name) = agent.as_deref() {
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

        let cwd = if cwd_parts.is_empty() {
            None
        } else {
            Some(cwd_parts.join(" "))
        };
        let has_launch_intent = agent.is_some() || cwd.is_some() || !agent_args.is_empty();

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

        let local_start = self
            .local_start
            .filter(|_| invoked_cmd == "/agent" && has_launch_intent);
        let prestarted_relay_session_id = if let Some(local_start) = local_start {
            match local_start
                .start_session(
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
            if let Some(cwd) = cwd.as_deref()
                && !cwd.trim().is_empty() {
                    query.append_pair("cwd", cwd);
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

    async fn build_terminal_attach_handoff_message(
        &self,
        relay_session_id: &str,
        user_id: u64,
    ) -> anyhow::Result<String> {
        let handoff =
            handoff::create_handoff_code(self.state.as_ref(), relay_session_id, user_id).await?;
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
        match handoff::resolve_relay_for_telegram_session(self.state.as_ref(), &telegram_session_id)
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
                match handoff::get_last_active_relay_session_id(self.state.as_ref(), user_id).await
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

        let handoff = match handoff::resolve_handoff_code(self.state.as_ref(), &code).await {
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

        if let Err(err) = handoff::consume_handoff_code(self.state.as_ref(), &code).await {
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
        if let Err(err) = handoff::bind_telegram_session_to_relay(
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

/// Syncs `bot_token` to the Mini App tenant for `user_id`, retrying
/// transient failures.
async fn sync_tenant_bot_token(
    web_app_url: &str,
    bot_token: &str,
    user_id: u64,
    label: Option<&str>,
) -> Result<(), String> {
    let url = terminal_tenant_bot_bootstrap_url(web_app_url)?;
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
            && let Some(map) = payload.as_object_mut() {
                map.insert(
                    "label".to_string(),
                    serde_json::Value::String(label.to_string()),
                );
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
                last_error =
                    format!("terminal bot sync response body failed (status {status}): {error}");
                if attempt == 2 {
                    return Err(last_error);
                }
                let jitter = rand::random::<u64>() % 150;
                tokio::time::sleep(Duration::from_millis(150 * (attempt + 1) + jitter)).await;
                continue;
            }
        };
        let (result, retryable) =
            interpret_terminal_bot_sync_response(status, content_type.as_deref(), body.as_ref());
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
