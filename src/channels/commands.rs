use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;

use super::formatting::format_number;
use crate::agent::Agent;
use crate::config::AppConfig;
use crate::tasks::TaskRegistry;
use crate::traits::StateStore;

/// Single source of truth for command definitions.
///
/// Each command is defined once here; the registry drives Telegram's
/// `setMyCommands` API, the `/help` output, and (in the future) Slack/Discord
/// command registration.
pub(crate) struct CommandDef {
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
pub(crate) enum CommandCategory {
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
pub(crate) fn shared_commands() -> Vec<CommandDef> {
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
            name: "cancel",
            description: "Cancel a running task",
            usage: Some("/cancel <id>"),
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "clear",
            description: "Start fresh conversation",
            usage: None,
            category: CommandCategory::Core,
        },
        CommandDef {
            name: "cost",
            description: "Show token usage stats",
            usage: None,
            category: CommandCategory::Core,
        },
    ]
}

/// Shared command dispatcher for commands that behave identically across
/// Telegram, Slack, and Discord channels.
pub(crate) struct CommandContext {
    pub agent: Arc<Agent>,
    pub state: Arc<dyn StateStore>,
    pub task_registry: Arc<TaskRegistry>,
    pub config_path: PathBuf,
}

impl CommandContext {
    /// Try to handle a shared command. Returns `Some(reply)` if the command was
    /// recognised, `None` if the channel should handle it itself.
    pub(crate) async fn dispatch(&self, cmd: &str, args: &str, session_id: &str) -> Option<String> {
        match cmd {
            "/model" => Some(self.handle_model(args).await),
            "/models" => Some(self.handle_models().await),
            "/auto" => Some(self.handle_auto().await),
            "/reload" => Some(self.handle_reload().await),
            "/tasks" => Some(self.handle_tasks(session_id).await),
            "/cancel" => Some(self.handle_cancel(args).await),
            "/clear" => Some(self.handle_clear(session_id).await),
            "/cost" => Some(self.handle_cost().await),
            _ => None,
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

    async fn handle_reload(&self) -> String {
        match AppConfig::load(&self.config_path) {
            Ok(new_config) => match self.agent.reload_provider(&new_config).await {
                Ok(status) => format!("Config reloaded. {}", status),
                Err(e) => format!("Provider reload failed: {}", e),
            },
            Err(e) => {
                // Config is broken — try to auto-restore from backup
                let backup = self.config_path.with_extension("toml.bak");
                if backup.exists() {
                    if tokio::fs::copy(&backup, &self.config_path).await.is_ok() {
                        format!(
                            "Config reload failed: {}\n\nAuto-restored from backup. Config is back to the previous working state.",
                            e
                        )
                    } else {
                        format!(
                            "Config reload failed: {}\n\nBackup restore also failed. Manual intervention needed.",
                            e
                        )
                    }
                } else {
                    format!("Config reload failed: {}\n\nNo backup available.", e)
                }
            }
        }
    }

    async fn handle_tasks(&self, session_id: &str) -> String {
        let entries = self.task_registry.list_for_session(session_id).await;
        if entries.is_empty() {
            "No tasks found.".to_string()
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
            lines.join("\n")
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
        let cancelled = self
            .task_registry
            .cancel_running_for_session(session_id)
            .await;

        match self.agent.clear_session(session_id).await {
            Ok(_) => {
                if cancelled.is_empty() {
                    "Context cleared. Starting fresh.".to_string()
                } else {
                    format!(
                        "Context cleared. Starting fresh. ({} running task{} cancelled.)",
                        cancelled.len(),
                        if cancelled.len() == 1 { "" } else { "s" }
                    )
                }
            }
            Err(e) => format!("Failed to clear context: {}", e),
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

        let records_24h = match self.state.get_token_usage_since(&since_24h).await {
            Ok(r) => r,
            Err(e) => return format!("Failed to query token usage: {}", e),
        };
        let records_7d = match self.state.get_token_usage_since(&since_7d).await {
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
