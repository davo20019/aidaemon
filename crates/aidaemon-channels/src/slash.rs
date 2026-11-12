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
            "/clear" => self.handle_clear(session_id).await,
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
        let left_suffix = |until: Option<chrono::DateTime<Utc>>| {
            until.map_or_else(String::new, |at| {
                format!(
                    " (about {} left)",
                    compact_chrono_duration(at - now, ZeroUnitStyle::Trim)
                )
            })
        };
        let requested = match args.trim().to_ascii_lowercase().as_str() {
            "" => {
                return match self.host.chat_mode(session_id, now).await {
                    Some((ChatMode::ReadOnly, until)) => {
                        format!("Mode: read-only{}.", left_suffix(until))
                    }
                    Some((ChatMode::Unrestricted, until)) => {
                        format!("Mode: unrestricted{}.", left_suffix(until))
                    }
                    Some((ChatMode::Full, _)) | None => "Mode: full.".to_string(),
                };
            }
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
                    "Context cleared. Starting fresh — your active history is kept (use /wipe to \
                     erase it from the active database).{suffix}"
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
pub fn build_help_text(commands: &[CommandDef], prefix: &str) -> String {
    let p = prefix;
    let mut text = String::from(
        "**What I can do**\n\
         \n\
         **Schedule recurring tasks**\n\
         _\"Check my disk space every 6 hours\"_\n\
         _\"Remind me to review PRs on weekdays at 9am\"_\n\
         \n\
         **Search & browse the web**\n\
         _\"What's the latest on Rust 2024 edition?\"_\n\
         _\"Summarize this article: <url>\"_\n\
         \n\
         **Run commands & manage your system**\n\
         _\"Show me docker container status\"_\n\
         _\"Deploy the staging branch\"_\n\
         \n\
         **Switch AI providers with guidance**\n\
         _\"Switch to Moonshot\"_\n\
         _\"Move me to Cloudflare Gateway and ask for what you need\"_\n\
         \n\
         **Remember things about you**\n\
         _\"I prefer dark mode and vim keybindings\"_\n\
         _\"My deploy server is 10.0.1.50\"_\n\
         \n\
         **Track your goals**\n\
         _\"I want to finish the API migration this month\"_\n\
         _\"How's my progress on learning Rust?\"_\n\
         \n\
         **Delegate coding tasks**\n\
         _\"Refactor the auth module to use JWT\"_\n\
         _\"Fix the failing tests in src/parser\"_\n\
         \n\
         **Send & receive files**\n\
         _\"Send me the latest log file\"_\n\
         _\"Analyze this CSV I'm uploading\"_\n\
         \n\
         Just ask in plain language — I'll pick the right tools.\n\
         \n\
         **Commands**",
    );

    for cmd in commands {
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
        text.push_str(&format!("\n`{}` — {}", label, cmd.description));
    }

    text
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
