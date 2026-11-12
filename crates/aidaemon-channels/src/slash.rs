//! Slash commands shared by every chat platform: the command menu
//! ([`shared_commands`], [`build_help_text`]) and the commands that behave the
//! same everywhere ([`SharedCommands`]): models, chat operations, context,
//! `/private`, `/mode`, `/clear`, `/wipe` and usage. A host supplies its
//! services through [`CommandHost`] and adds its own commands (`/work`,
//! `/reload`...) as [`SlashCommand`]s. Internal to aidaemon, no stability
//! promise; the public contract is [`crate::channel`].

use std::collections::HashMap;
use std::sync::Arc;

use aidaemon_events::conversation::{ConversationSummary, Message};
use aidaemon_events::provider::TokenUsageRecord;
use aidaemon_tools::duration_format::{compact_chrono_duration, ZeroUnitStyle};
use aidaemon_tools::semantics::UserRole;
use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::ports::ChannelAgentRuntime;
use crate::tasks::{TaskRegistry, TaskStatus};

/// Whether a chat message is a slash command: `/name` or `/name@bot`, where
/// `name` is letters, digits and underscores (Telegram's command syntax).
/// A message that merely starts with a path (`/tmp/x.py is broken`) is not.
pub fn is_slash_command(text: &str) -> bool {
    let Some(first) = text.split_whitespace().next() else {
        return false;
    };
    let Some(rest) = first.strip_prefix('/') else {
        return false;
    };
    let name = rest.split_once('@').map_or(rest, |(name, _bot)| name);
    !name.is_empty()
        && name.starts_with(|c: char| c.is_ascii_alphabetic())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod slash_command_tests {
    use super::is_slash_command;

    #[test]
    fn commands_are_names_and_paths_are_messages() {
        for command in [
            "/help",
            "/model gpt-5.6-terra",
            "/clear",
            "/start@aidaemon_bot",
            "/set_up now",
        ] {
            assert!(is_slash_command(command), "{command}");
        }
        for message in [
            "/tmp/aidaemon-live/prime/prime.py has a bug",
            "/Users/alice/notes.txt",
            "/ what",
            "/",
            "/123",
            "hello /help",
        ] {
            assert!(!is_slash_command(message), "{message}");
        }
    }
}

/// Single source of truth for command definitions.
///
/// Each command is defined once here; the registry drives Telegram's
/// `setMyCommands` API, the `/help` output, and (in the future) Slack/Discord
/// command registration.
pub struct CommandDef {
    /// Command name without leading `/` or `!`.
    pub name: &'static str,
    /// Short description shown in Telegram's command menu and `/help` text.
    /// Must be 3-256 characters for Telegram's `BotCommand`.
    pub description: &'static str,
    /// Optional usage string shown only in `/help` (e.g. "/model [name]").
    pub usage: Option<&'static str>,
    /// Which platform group this command belongs to.
    pub category: CommandCategory,
}

/// Determines which platforms show a command.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CommandCategory {
    /// Available on all platforms.
    Core,
    /// Telegram + Slack (process restart).
    Restart,
    /// Telegram-only (dynamic bot management).
    Connect,
    /// Telegram-only (terminal/agent bridge).
    Terminal,
}

/// Returns the shared commands available on every platform.
pub fn shared_commands() -> Vec<CommandDef> {
    vec![
        CommandDef {
            name: "model",
            description: "Show or switch AI model",
            usage: Some("/model [name]"),
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "models",
            description: "List available models",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "auto",
            description: "Re-enable automatic model routing",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "reload",
            description: "Reload configuration",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "tasks",
            description: "List running tasks",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "status",
            description: "Show this chat's mode, running tasks and what's next",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "work",
            description: "View and manage durable work",
            usage: Some("/work [ready|running|blocked|show|comment|unblock|retry]"),
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "cancel",
            description: "Cancel a running task",
            usage: Some("/cancel <id>"),
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "clear",
            description: "Start fresh conversation (history kept)",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "wipe",
            description: "Permanently delete this conversation",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "cost",
            description: "Show token usage stats",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "context",
            description: "Show conversation context coverage",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "checkpoints",
            description: "List filesystem checkpoints",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "readonly",
            description: "Make this chat read-only for 4 hours",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "private",
            description: "Don't remember or learn from your next message",
            usage: Some("/private [off]"),
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "mode",
            description: "Show or change this chat's access mode",
            usage: Some("/mode [full|unrestricted|readonly]"),
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "rollback",
            description: "Preview or confirm a checkpoint rollback",
            usage: Some("/rollback [checkpoint-id|confirm TOKEN]"),
            category: CommandCategory::Core,
        },
    ]
}

/// A chat's tool authority as the owner set it with `/mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatMode {
    ReadOnly,
    Full,
    Unrestricted,
}

/// What the shared commands need from the host besides the agent.
#[async_trait]
pub trait CommandHost: Send + Sync {
    /// `/private`: whether the sender's next message in this chat is kept
    /// out of memory (`on`), or that request cancelled.
    async fn set_next_message_private(
        &self,
        session_id: &str,
        sender_id: Option<&str>,
        on: bool,
    ) -> anyhow::Result<()>;
    /// The chat's mode in force at `now` and when it ends; `None` is full.
    async fn chat_mode(
        &self,
        session_id: &str,
        now: DateTime<Utc>,
    ) -> Option<(ChatMode, Option<DateTime<Utc>>)>;
    /// Set the chat's mode from `now`; returns when it ends.
    async fn set_chat_mode(
        &self,
        session_id: &str,
        mode: ChatMode,
        now: DateTime<Utc>,
    ) -> anyhow::Result<DateTime<Utc>>;
    /// How many hours a narrowed or widened mode lasts.
    fn chat_mode_hours(&self) -> i64;
    async fn history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<Message>>;
    async fn conversation_summary(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<ConversationSummary>>;
    /// Estimated tokens of stored messages (`role`, `content`, `name`).
    fn estimate_message_tokens(&self, messages: &[serde_json::Value]) -> usize;
    fn estimate_text_tokens(&self, text: &str) -> usize;
    /// Token usage recorded since `since` (`%Y-%m-%d %H:%M:%S`, UTC).
    async fn token_usage_since(&self, since: &str) -> anyhow::Result<Vec<TokenUsageRecord>>;
    /// The owner's next scheduled item, for `/status`; `None` when nothing
    /// is scheduled (or the host keeps no schedules).
    async fn next_scheduled(&self) -> Option<ScheduledItem> {
        None
    }

    /// `at` in the owner's local time and format ("2:39 PM ET"), for the
    /// first line of `/status`; `None` when the host has no owner zone.
    fn owner_time(&self, _at: DateTime<Utc>) -> Option<String> {
        None
    }
}

/// One scheduled item as `/status` shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledItem {
    /// What runs, in the owner's words (`/status` cuts it at a word).
    pub what: String,
    /// When, already in the owner's local time ("2:39 PM ET").
    pub when: String,
}

/// What `/status` reports, read from typed state.
#[derive(Debug, Clone)]
pub struct StatusView {
    /// The current time in the owner's local format, when known.
    pub now: Option<String>,
    /// The chat's mode and when it ends; `None` is full access.
    pub mode: Option<(ChatMode, Option<DateTime<Utc>>)>,
    /// Running chat tasks: id, description, when it started.
    pub running: Vec<(u64, String, DateTime<Utc>)>,
    pub next_scheduled: Option<ScheduledItem>,
}

/// The one line that says a chat's mode, as `/mode` and `/status` show it.
fn mode_line(mode: Option<(ChatMode, Option<DateTime<Utc>>)>, now: DateTime<Utc>) -> String {
    let left = |until: Option<DateTime<Utc>>| {
        until.map_or_else(String::new, |at| {
            format!(
                " (about {} left)",
                compact_chrono_duration(at - now, ZeroUnitStyle::Trim)
            )
        })
    };
    match mode {
        Some((ChatMode::ReadOnly, until)) => format!("Mode: read-only{}.", left(until)),
        Some((ChatMode::Unrestricted, until)) => format!("Mode: unrestricted{}.", left(until)),
        Some((ChatMode::Full, _)) | None => "Mode: full.".to_string(),
    }
}

/// `/status`: the chat's mode, what is running and what is scheduled next.
pub fn render_status(view: &StatusView, now: DateTime<Utc>) -> String {
    let mode = mode_line(view.mode, now);
    let mut lines = vec![match &view.now {
        Some(local) => format!("It's {local}. {mode}"),
        None => mode,
    }];
    if view.running.is_empty() {
        lines.push("Running: nothing right now.".to_string());
    } else {
        lines.push("Running:".to_string());
        for (id, description, started) in &view.running {
            lines.push(format!(
                "• {} ({} so far, /cancel {id} to stop it)",
                crate::presentation::word_preview(description, 80),
                compact_chrono_duration(now - *started, ZeroUnitStyle::Trim)
            ));
        }
    }
    lines.push(match &view.next_scheduled {
        Some(item) => format!(
            "Next scheduled: {} – {}",
            crate::presentation::word_preview(&item.what, 80),
            item.when
        ),
        None => "Next scheduled: nothing.".to_string(),
    });
    lines.join("\n")
}

/// The command closest to an unknown `name` (without its prefix) by edit
/// distance, when one is close enough to be a likely typo.
pub fn closest_command<'a>(name: &str, commands: &'a [CommandDef]) -> Option<&'a str> {
    let name = name.to_ascii_lowercase();
    let len = name.chars().count();
    if len == 0 {
        return None;
    }
    let limit = if len <= 4 { 1 } else { 2 };
    commands
        .iter()
        .map(|command| (edit_distance(&name, command.name), command.name))
        .filter(|(distance, _)| *distance <= limit)
        .min_by_key(|(distance, _)| *distance)
        .map(|(_, command)| command)
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut current = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(ca != *cb);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        previous = current;
    }
    previous[b.len()]
}

/// The reply to a command nobody handles: a likely intended command when one
/// is close, and where to find the rest.
pub fn unknown_command_reply(cmd: &str, commands: &[CommandDef], prefix: &str) -> String {
    let name = cmd
        .trim_start_matches(prefix)
        .split('@')
        .next()
        .unwrap_or("");
    match closest_command(name, commands) {
        Some(suggestion) => format!(
            "I don't know {cmd}. Did you mean {prefix}{suggestion}? Send {prefix}help to see \
             the commands."
        ),
        None => format!("I don't know {cmd}. Send {prefix}help to see the commands."),
    }
}

/// How `/private` was sent when it carries more than the bare command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivateRequest<'a> {
    /// Only `/private` (said once or more, "/private/private"): the next
    /// message is kept private.
    Next,
    /// `/private` with the message itself ("/private what's my balance?",
    /// "/privateWhat's…"): this message is the private one.
    With(&'a str),
}

/// Read `/private` sent together with its message. `None` for anything
/// else, including `/private off` and the bare `/private`, which the
/// command itself answers.
pub fn private_request(text: &str) -> Option<PrivateRequest<'_>> {
    let mut rest = text.trim_start();
    let mut seen = 0usize;
    while let Some(after) = strip_prefix_ignore_case(rest, "/private") {
        seen += 1;
        // Telegram may append the bot name in groups: `/private@my_bot`.
        rest = match after.strip_prefix('@') {
            Some(bot) => bot.find(char::is_whitespace).map_or("", |at| &bot[at..]),
            None => after,
        };
        rest = rest.trim_start();
    }
    if seen == 0 {
        return None;
    }
    let message = rest.trim();
    if message.is_empty() {
        return (seen > 1).then_some(PrivateRequest::Next);
    }
    if message.eq_ignore_ascii_case("off") {
        return None;
    }
    Some(PrivateRequest::With(message))
}

fn strip_prefix_ignore_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let head = text.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| &text[prefix.len()..])
}

/// The commands `/help` shows first: the ones used day to day.
pub const EVERYDAY_COMMANDS: &[&str] = &["clear", "tasks", "status", "private", "readonly", "mode"];

/// How much `/help` shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpScope {
    /// The everyday commands and how to see the rest.
    Everyday,
    /// Every command the platform has (`/help all`).
    All,
}

impl HelpScope {
    pub fn from_args(args: &str) -> Self {
        if args.trim().eq_ignore_ascii_case("all") {
            Self::All
        } else {
            Self::Everyday
        }
    }
}

/// Who sent a command, and where.
pub struct SlashCall<'a> {
    pub session_id: &'a str,
    pub sender_id: Option<&'a str>,
    pub user_role: UserRole,
}

/// A command the host adds to the shared ones.
#[async_trait]
pub trait SlashCommand: Send + Sync {
    /// Whether this handles `cmd` (with its `/`).
    fn handles(&self, cmd: &str) -> bool;
    async fn run(&self, cmd: &str, args: &str, call: &SlashCall<'_>) -> String;
}

/// Shared command dispatcher for commands that behave identically across
/// Telegram, Slack, and Discord channels.
pub struct SharedCommands {
    pub agent: Arc<dyn ChannelAgentRuntime>,
    pub host: Arc<dyn CommandHost>,
    pub task_registry: Arc<TaskRegistry>,
    /// The host's own commands, asked in order for any other name.
    pub extensions: Vec<Arc<dyn SlashCommand>>,
}

impl SharedCommands {
    /// Try to handle a shared command. Returns `Some(reply)` if the command was
    /// recognised, `None` if the channel should handle it itself.
    pub async fn dispatch(
        &self,
        cmd: &str,
        args: &str,
        session_id: &str,
        sender_id: Option<&str>,
        user_role: UserRole,
    ) -> Option<String> {
        Some(match cmd {
            "/model" => self.handle_model(args).await,
            "/models" => self.handle_models().await,
            "/auto" => self.handle_auto().await,
            "/tasks" => self.handle_tasks(session_id).await,
            "/cancel" => self.handle_cancel(args).await,
            "/clear" | "/new" => self.handle_clear(session_id).await,
            "/status" => self.handle_status(session_id).await,
            "/wipe" => self.handle_wipe(session_id).await,
            "/cost" => self.handle_cost().await,
            "/context" => self.handle_context(session_id, user_role).await,
            "/readonly" => self.handle_readonly(session_id, user_role).await,
            "/private" => self.handle_private(args, session_id, sender_id).await,
            "/mode" => self.handle_mode(args, session_id, user_role).await,
            _ => {
                let extension = self.extensions.iter().find(|ext| ext.handles(cmd))?;
                let call = SlashCall {
                    session_id,
                    sender_id,
                    user_role,
                };
                extension.run(cmd, args, &call).await
            }
        })
    }

    async fn handle_status(&self, session_id: &str) -> String {
        let now = Utc::now();
        let running = self
            .task_registry
            .list_for_session(session_id)
            .await
            .into_iter()
            .filter(|entry| matches!(entry.status, TaskStatus::Running))
            .map(|entry| (entry.id, entry.description, entry.started_at))
            .collect();
        let view = StatusView {
            now: self.host.owner_time(now),
            mode: self.host.chat_mode(session_id, now).await,
            running,
            next_scheduled: self.host.next_scheduled().await,
        };
        render_status(&view, now)
    }

    /// `/private [off]`: the sender's next message in this chat is not
    /// remembered or learned from (it only narrows memory, so anyone may ask).
    /// `off` cancels. Both are the sender's own: in a shared chat nobody else
    /// consumes or cancels them.
    async fn handle_private(
        &self,
        args: &str,
        session_id: &str,
        sender_id: Option<&str>,
    ) -> String {
        let on = match args.trim().to_ascii_lowercase().as_str() {
            "" => true,
            "off" => false,
            _ => return "Usage: /private [off]".to_string(),
        };
        match self
            .host
            .set_next_message_private(session_id, sender_id, on)
            .await
        {
            Ok(()) if on => "Your next message won't be remembered or learned from. \
                             Memory can still be used to answer it."
                .to_string(),
            Ok(()) => "Cancelled: your next message is remembered as usual.".to_string(),
            Err(error) => format!("Could not set that: {error}"),
        }
    }

    async fn handle_readonly(&self, session_id: &str, user_role: UserRole) -> String {
        self.handle_mode("readonly", session_id, user_role).await
    }

    /// `/mode [full|unrestricted|readonly]`: the owner narrows or restores this
    /// chat's tool authority. Explicit command syntax, persisted with an expiry.
    async fn handle_mode(&self, args: &str, session_id: &str, user_role: UserRole) -> String {
        let session_mode_ttl_hours = self.host.chat_mode_hours();
        const NOT_OWNER: &str = "Only the owner can change the mode.";
        const USAGE: &str = "Usage: /mode [full|unrestricted|readonly]";
        if user_role != UserRole::Owner {
            return NOT_OWNER.to_string();
        }
        let now = Utc::now();
        let requested = match args.trim().to_ascii_lowercase().as_str() {
            "" => return mode_line(self.host.chat_mode(session_id, now).await, now),
            "full" => ChatMode::Full,
            "unrestricted" => ChatMode::Unrestricted,
            "readonly" | "read-only" | "read_only" => ChatMode::ReadOnly,
            _ => return USAGE.to_string(),
        };
        match self.host.set_chat_mode(session_id, requested, now).await {
            Ok(_) => match requested {
                ChatMode::ReadOnly => format!("Read-only for this chat for the next {session_mode_ttl_hours} hours, starting now, including work already running. aidaemon can look but not change anything. Send /mode full to end it early."),
                ChatMode::Unrestricted => format!("Unrestricted for this chat for the next {session_mode_ttl_hours} hours, starting with your next message."),
                ChatMode::Full => "Back to full access from your next message.".to_string(),
            },
            Err(error) => {
                tracing::warn!(session_id, %error, "Could not change the session mode");
                "Couldn't change the mode right now.".to_string()
            }
        }
    }

    async fn handle_context(&self, session_id: &str, user_role: UserRole) -> String {
        if user_role != UserRole::Owner {
            return "Only the owner can inspect conversation context coverage.".to_string();
        }
        let model = self.agent.current_model().await;
        let (enabled, model_budget, trigger_tokens, recent_tokens, anchor) =
            self.agent.context_debug_settings(session_id, &model).await;
        let history = self.host.history(session_id, 10_000).await;
        let summary = self.host.conversation_summary(session_id).await;
        match (history, summary) {
            (Ok(history), Ok(summary)) => {
                let raw_values = history
                    .iter()
                    .map(|message| {
                        serde_json::json!({
                            "role": message.role,
                            "content": message.content,
                            "name": message.tool_name,
                        })
                    })
                    .collect::<Vec<_>>();
                let exact_tokens = self.host.estimate_message_tokens(&raw_values);
                let compacted_messages = summary
                    .as_ref()
                    .map(|value| value.message_count)
                    .unwrap_or(0);
                let raw_recent_messages = history.len().saturating_sub(compacted_messages);
                let summary_line = summary.as_ref().map_or_else(
                    || "Summary: none yet".to_string(),
                    |value| {
                        format!(
                            "Summary: {} messages through turn {} / message {} (~{} tokens)",
                            value.message_count,
                            value
                                .last_turn_seq
                                .map(|seq| seq.to_string())
                                .unwrap_or_else(|| "legacy".to_string()),
                            value.last_message_id,
                            self.host.estimate_text_tokens(&value.summary)
                        )
                    },
                );
                format!(
                    "Conversation context\nEnabled: {enabled}\nModel: {model}\nModel window: ~{model_budget} tokens\nCompaction high-water: ~{trigger_tokens} tokens\nRecent raw overlap target: ~{recent_tokens} tokens\n{summary_line}\nPersisted exact messages inspected: {} (~{exact_tokens} tokens)\nEstimated uncompacted messages: {raw_recent_messages}\nCurrent raw-turn anchor: {}\n\nExact retained history remains searchable even when it is not in the active prompt.",
                    history.len(),
                    anchor
                        .map(|seq| seq.to_string())
                        .unwrap_or_else(|| "not initialized".to_string())
                )
            }
            (Err(error), _) | (_, Err(error)) => {
                format!("Could not inspect conversation context: {error}")
            }
        }
    }

    async fn handle_model(&self, arg: &str) -> String {
        if arg.is_empty() {
            let current = self.agent.current_model().await;
            format!(
                "Current model: {}\n\nUsage: /model <model-name>\nExample: /model gemini-3-pro-preview",
                current
            )
        } else {
            self.agent.set_model(arg.to_string()).await;
            format!(
                "Model switched to: {}\nAuto-routing disabled. Use /auto to re-enable.",
                arg
            )
        }
    }

    async fn handle_models(&self) -> String {
        match self.agent.list_models().await {
            Ok(models) => {
                if models.is_empty() {
                    "No models found from provider.".to_string()
                } else {
                    let current = self.agent.current_model().await;
                    let list: Vec<String> = models
                        .iter()
                        .map(|m| {
                            if *m == current {
                                format!("• {} (active)", m)
                            } else {
                                format!("• {}", m)
                            }
                        })
                        .collect();
                    format!("Available models:\n{}", list.join("\n"))
                }
            }
            Err(e) => format!("Failed to list models: {}", e),
        }
    }

    async fn handle_auto(&self) -> String {
        self.agent.clear_model_override().await;
        "Auto-routing re-enabled. Model will be selected automatically based on query complexity."
            .to_string()
    }

    async fn handle_tasks(&self, session_id: &str) -> String {
        let entries = self
            .task_registry
            .list_for_session(session_id)
            .await
            .into_iter()
            .filter(|entry| !matches!(entry.status, TaskStatus::Completed))
            .collect::<Vec<_>>();
        if entries.is_empty() {
            "No active or failed chat operations. Completed conversation turns are audit history, not durable work. Use /work to inspect persistent work items.".to_string()
        } else {
            let lines: Vec<String> = entries
                .iter()
                .map(|e| {
                    let elapsed = match e.finished_at {
                        Some(fin) => {
                            let d = fin - e.started_at;
                            format!("{}s", d.num_seconds())
                        }
                        None => {
                            let d = Utc::now() - e.started_at;
                            format!("{}s elapsed", d.num_seconds())
                        }
                    };
                    format!("#{} [{}] {} ({})", e.id, e.status, e.description, elapsed)
                })
                .collect();
            format!(
                "Chat operations:\n{}\n\nCompleted conversation turns are omitted. Use /work for durable work items.",
                lines.join("\n")
            )
        }
    }

    async fn handle_cancel(&self, arg: &str) -> String {
        if arg.is_empty() {
            "Usage: /cancel <task-id>\nExample: /cancel 1".to_string()
        } else {
            match arg.parse::<u64>() {
                Ok(task_id) => {
                    if self.task_registry.cancel(task_id).await {
                        format!("Task #{} cancelled.", task_id)
                    } else {
                        format!("Task #{} not found or not running.", task_id)
                    }
                }
                Err(_) => "Invalid task ID. Usage: /cancel <task-id>".to_string(),
            }
        }
    }

    async fn handle_clear(&self, session_id: &str) -> String {
        // Cancel any running tasks for this session so the agent loop aborts
        // immediately instead of continuing to burn tokens after /clear.
        let cancelled_tasks = self
            .task_registry
            .cancel_running_for_session(session_id)
            .await;
        // Also cancel finite orchestration goals associated with this
        // conversation. Recurring schedules and personal goals intentionally
        // survive a context clear.
        let cancelled_goals = self
            .agent
            .cancel_active_finite_work_for_session(session_id)
            .await;

        // Non-destructive: sets a context boundary so the next turn starts
        // fresh, but the event history remains for memory and audit. Use /wipe
        // to permanently delete.
        match self.agent.clear_session_context(session_id).await {
            Ok(_) => {
                let mut cancellations = Vec::new();
                if !cancelled_tasks.is_empty() {
                    cancellations.push(format!(
                        "{} running task{}",
                        cancelled_tasks.len(),
                        if cancelled_tasks.len() == 1 { "" } else { "s" }
                    ));
                }
                if !cancelled_goals.is_empty() {
                    cancellations.push(format!(
                        "{} background goal{}",
                        cancelled_goals.len(),
                        if cancelled_goals.len() == 1 { "" } else { "s" }
                    ));
                }
                let suffix = if cancellations.is_empty() {
                    String::new()
                } else {
                    format!(" ({} cancelled.)", cancellations.join(" and "))
                };
                format!(
                    "Context cleared. Starting fresh – your history is kept; use /wipe to \
                     delete this chat's history.{suffix}"
                )
            }
            Err(e) => format!("Failed to clear context: {}", e),
        }
    }

    async fn handle_wipe(&self, session_id: &str) -> String {
        let cancelled = self
            .task_registry
            .cancel_running_for_session(session_id)
            .await;

        // Destructive: erases this session's active database artifacts.
        match self.agent.clear_session(session_id).await {
            Ok(_) => {
                let suffix = if cancelled.is_empty() {
                    String::new()
                } else {
                    format!(
                        " ({} running task{} cancelled.)",
                        cancelled.len(),
                        if cancelled.len() == 1 { "" } else { "s" }
                    )
                };
                format!(
                    "Conversation erased from the active database (events, exact-search index, \
                     summaries, episodes, and raw memory spans). Facts already saved to memory \
                     are kept without raw-message provenance. Existing encrypted backups may \
                     retain older copies until their own lifecycle removes them.{suffix}"
                )
            }
            Err(e) => format!("Failed to wipe conversation: {}", e),
        }
    }

    async fn handle_cost(&self) -> String {
        let now = Utc::now();
        let since_24h = (now - chrono::Duration::hours(24))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let since_7d = (now - chrono::Duration::days(7))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();

        let records_24h = match self.host.token_usage_since(&since_24h).await {
            Ok(r) => r,
            Err(e) => return format!("Failed to query token usage: {}", e),
        };
        let records_7d = match self.host.token_usage_since(&since_7d).await {
            Ok(r) => r,
            Err(e) => return format!("Failed to query token usage: {}", e),
        };

        let (input_24h, output_24h) = records_24h.iter().fold((0i64, 0i64), |(i, o), r| {
            (i + r.input_tokens, o + r.output_tokens)
        });
        let (input_7d, output_7d) = records_7d.iter().fold((0i64, 0i64), |(i, o), r| {
            (i + r.input_tokens, o + r.output_tokens)
        });

        // Top models (by total tokens in 7d)
        let mut model_totals: HashMap<&str, i64> = HashMap::new();
        for r in &records_7d {
            *model_totals.entry(&r.model).or_insert(0) += r.input_tokens + r.output_tokens;
        }
        let mut models_sorted: Vec<(&&str, &i64)> = model_totals.iter().collect();
        models_sorted.sort_by(|a, b| b.1.cmp(a.1));

        let mut reply = format!(
            "Token usage (last 24h):\n  Input:  {} tokens\n  Output: {} tokens\n\n\
             Token usage (last 7d):\n  Input:  {} tokens\n  Output: {} tokens",
            format_number(input_24h),
            format_number(output_24h),
            format_number(input_7d),
            format_number(output_7d),
        );

        if !models_sorted.is_empty() {
            reply.push_str("\n\nTop models (7d):");
            for (model, total) in models_sorted.iter().take(5) {
                reply.push_str(&format!("\n  {}: {} tokens", model, format_number(**total)));
            }
        }

        reply
    }
}

/// Format a number with comma separators (e.g. 12450 → "12,450").
pub fn format_number(n: i64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Build the /help response text from a list of command definitions.
///
/// `commands`: the command definitions to include (platform-specific).
/// `prefix`: command prefix character ("/" for Telegram/Discord, "!" for Slack).
/// `scope`: the everyday commands, or all of them (`/help all`). The text is
/// Markdown; each channel renders it in its own format.
pub fn build_help_text(commands: &[CommandDef], prefix: &str, scope: HelpScope) -> String {
    let p = prefix;
    let shown: Vec<&CommandDef> = match scope {
        HelpScope::Everyday => EVERYDAY_COMMANDS
            .iter()
            .filter_map(|name| commands.iter().find(|cmd| cmd.name == *name))
            .collect(),
        HelpScope::All => commands.iter().collect(),
    };
    let mut text = match scope {
        HelpScope::Everyday => String::from(
            "Just ask in plain words – I'll pick the right tools. For example: \
             _\"Check my disk space every 6 hours\"_ or _\"Send me the latest log file\"_.\n\n\
             **Everyday commands**",
        ),
        HelpScope::All => String::from("**All commands**"),
    };
    for cmd in shown {
        let label = match cmd.usage {
            Some(usage) => {
                // Replace the leading "/" with the platform prefix.
                if let Some(rest) = usage.strip_prefix('/') {
                    format!("{p}{rest}")
                } else {
                    usage.to_string()
                }
            }
            None => format!("{p}{}", cmd.name),
        };
        text.push_str(&format!("\n{} – {}", label, cmd.description));
    }
    if scope == HelpScope::Everyday {
        text.push_str(&format!("\n\nSend {p}help all for every command."));
    }
    text
}

#[cfg(test)]
mod help_and_status_tests {
    use super::*;

    fn defs() -> Vec<CommandDef> {
        shared_commands()
    }

    #[test]
    fn help_shows_the_everyday_commands_and_how_to_see_the_rest() {
        let help = build_help_text(&defs(), "/", HelpScope::Everyday);
        for name in EVERYDAY_COMMANDS {
            assert!(help.contains(&format!("\n/{name}")), "{name}: {help}");
        }
        assert!(!help.contains("/models"), "{help}");
        assert!(!help.contains('`'), "{help}");
        assert!(
            help.ends_with("Send /help all for every command."),
            "{help}"
        );
        let all = build_help_text(&defs(), "!", HelpScope::All);
        assert!(all.contains("!models – List available models"), "{all}");
        assert_eq!(HelpScope::from_args(" ALL "), HelpScope::All);
        assert_eq!(HelpScope::from_args(""), HelpScope::Everyday);
    }

    #[test]
    fn an_unknown_command_suggests_the_closest_one() {
        let commands = defs();
        assert_eq!(closest_command("cleer", &commands), Some("clear"));
        assert_eq!(closest_command("stauts", &commands), Some("status"));
        assert_eq!(closest_command("weather", &commands), None);
        assert_eq!(
            unknown_command_reply("/claer", &commands, "/"),
            "I don't know /claer. Did you mean /clear? Send /help to see the commands."
        );
        assert_eq!(
            unknown_command_reply("/weather@my_bot", &commands, "/"),
            "I don't know /weather@my_bot. Send /help to see the commands."
        );
    }

    #[test]
    fn private_can_carry_its_message() {
        assert_eq!(private_request("/private"), None);
        assert_eq!(private_request("/private off"), None);
        assert_eq!(private_request("/clear"), None);
        assert_eq!(
            private_request("/private/private"),
            Some(PrivateRequest::Next)
        );
        assert_eq!(
            private_request("/privateAnother test"),
            Some(PrivateRequest::With("Another test"))
        );
        assert_eq!(
            private_request("/private@my_bot what's my balance?"),
            Some(PrivateRequest::With("what's my balance?"))
        );
        assert_eq!(
            private_request("/private /private hi"),
            Some(PrivateRequest::With("hi"))
        );
    }

    #[test]
    fn status_says_mode_running_work_and_whats_next() {
        let now = Utc::now();
        let quiet = StatusView {
            now: None,
            mode: None,
            running: Vec::new(),
            next_scheduled: None,
        };
        assert_eq!(
            render_status(&quiet, now),
            "Mode: full.\nRunning: nothing right now.\nNext scheduled: nothing."
        );
        let busy = StatusView {
            now: Some("2:05 PM ET".to_string()),
            mode: Some((ChatMode::ReadOnly, Some(now + chrono::Duration::hours(3)))),
            running: vec![(
                7,
                "Read four files".to_string(),
                now - chrono::Duration::minutes(2),
            )],
            next_scheduled: Some(ScheduledItem {
                what: "Check disk space".to_string(),
                when: "2:39 PM ET".to_string(),
            }),
        };
        let status = render_status(&busy, now);
        assert!(
            status.starts_with("It's 2:05 PM ET. Mode: read-only (about 3h left)."),
            "{status}"
        );
        assert!(
            status.contains("• Read four files (2m so far, /cancel 7 to stop it)"),
            "{status}"
        );
        assert!(
            status.ends_with("Next scheduled: Check disk space – 2:39 PM ET"),
            "{status}"
        );

        // A long description is cut at a word, never mid-word.
        let long = StatusView {
            next_scheduled: Some(ScheduledItem {
                what: "Every day, independently manage and publish to the synthetic blog at https://blog.example.test following the editorial plan".to_string(),
                when: "Fri, Nov 13, 6:00 AM ET".to_string(),
            }),
            ..busy
        };
        let status = render_status(&long, now);
        let next = status.lines().last().unwrap();
        assert!(next.contains("… – Fri, Nov 13, 6:00 AM ET"), "{next}");
        assert!(next.contains("Every day, independently manage"), "{next}");
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(12450), "12,450");
        assert_eq!(format_number(1_000_000), "1,000,000");
    }
}
