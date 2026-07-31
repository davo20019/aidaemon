use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;

use super::formatting::format_number;
use crate::agent::Agent;
use crate::config::AppConfig;
use crate::tasks::TaskRegistry;
use crate::traits::{StateStore, TaskJournalEntry, WorkTaskSummary};
use crate::types::UserRole;

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
            name: "checkpoints",
            description: "List filesystem checkpoints",
            usage: None,
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
    pub(crate) async fn dispatch(
        &self,
        cmd: &str,
        args: &str,
        session_id: &str,
        user_role: UserRole,
    ) -> Option<String> {
        match cmd {
            "/model" => Some(self.handle_model(args).await),
            "/models" => Some(self.handle_models().await),
            "/auto" => Some(self.handle_auto().await),
            "/reload" => Some(self.handle_reload().await),
            "/tasks" => Some(self.handle_tasks(session_id).await),
            "/work" => Some(self.handle_work(args, session_id, user_role).await),
            "/cancel" => Some(self.handle_cancel(args).await),
            "/clear" => Some(self.handle_clear(session_id).await),
            "/wipe" => Some(self.handle_wipe(session_id).await),
            "/cost" => Some(self.handle_cost().await),
            "/checkpoints" => Some(self.handle_checkpoints(user_role).await),
            "/rollback" => Some(self.handle_rollback(args, session_id, user_role).await),
            _ => None,
        }
    }

    async fn handle_checkpoints(&self, user_role: UserRole) -> String {
        if user_role != UserRole::Owner {
            return "Only the owner can inspect filesystem checkpoints.".to_string();
        }
        match crate::checkpoints::active_manager() {
            Some(manager) => manager.list_text(30, None).await,
            None => "Filesystem checkpoints are disabled.".to_string(),
        }
    }

    async fn handle_rollback(&self, args: &str, session_id: &str, user_role: UserRole) -> String {
        if user_role != UserRole::Owner {
            return "Only the owner can roll back filesystem checkpoints.".to_string();
        }
        let Some(manager) = crate::checkpoints::active_manager() else {
            return "Filesystem checkpoints are disabled.".to_string();
        };
        let mut parts = args.split_whitespace();
        let first = parts.next();
        if first.is_some_and(|value| value.eq_ignore_ascii_case("confirm")) {
            let Some(token) = parts.next() else {
                return "Usage: /rollback confirm <token>".to_string();
            };
            if parts.next().is_some() {
                return "Usage: /rollback confirm <token>".to_string();
            }
            return match manager.apply_rollback(session_id, token).await {
                Ok(result) => result.render(),
                Err(error) => format!("Rollback failed: {error}"),
            };
        }
        if parts.next().is_some() {
            return "Usage: /rollback [checkpoint-id]".to_string();
        }
        match manager.prepare_rollback(session_id, first).await {
            Ok(preview) => preview.render(),
            Err(error) => format!("Could not prepare rollback: {error}"),
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

    async fn handle_work(&self, args: &str, session_id: &str, user_role: UserRole) -> String {
        if user_role != UserRole::Owner {
            return "Only the owner can view or manage durable work.".to_string();
        }

        let mut parts = args.split_whitespace();
        let action = parts.next().unwrap_or("overview").to_ascii_lowercase();
        let project_id = match self.state.get_session_work_project(session_id).await {
            Ok(project_id) => project_id,
            Err(error) => return format!("Could not resolve the active project: {error}"),
        };

        match action.as_str() {
            "overview" | "all" => self.render_work_overview(&project_id).await,
            "waiting" | "ready" | "running" | "in_progress" | "blocked" | "attention"
            | "needs_attention" | "done" => {
                let lane = match action.as_str() {
                    "running" => "in_progress",
                    "attention" => "needs_attention",
                    value => value,
                };
                self.render_work_lane(&project_id, lane).await
            }
            "show" => {
                let Some(reference) = parts.next() else {
                    return "Usage: /work show <task-id>".to_string();
                };
                let task = match self.resolve_work_task(&project_id, reference).await {
                    Ok(task) => task,
                    Err(message) => return message,
                };
                self.render_work_task(&task).await
            }
            "comment" | "unblock" => {
                let Some(reference) = parts.next() else {
                    return format!("Usage: /work {action} <task-id> <message>");
                };
                let message = parts.collect::<Vec<_>>().join(" ");
                if message.trim().is_empty() {
                    return format!("Usage: /work {action} <task-id> <message>");
                }
                let task = match self.resolve_work_task(&project_id, reference).await {
                    Ok(task) => task,
                    Err(message) => return message,
                };
                if action == "unblock" {
                    return match self
                        .state
                        .unblock_task(&task.task_id, &message, "owner", Some(session_id))
                        .await
                    {
                        Ok(true) => format!(
                            "Task {} was unblocked and returned to the ready queue.",
                            short_id(&task.task_id)
                        ),
                        Ok(false) => "That task is not currently blocked.".to_string(),
                        Err(error) => format!("Could not unblock task: {error}"),
                    };
                }
                let entry = TaskJournalEntry::new(
                    &task.project_id,
                    &task.goal_id,
                    &task.goal_run_id,
                    "comment",
                    "human",
                    "owner",
                    &message,
                )
                .with_task(Some(&task.task_id), None)
                .with_source_channel(Some(session_id));
                match self.state.append_task_journal(&entry).await {
                    Ok(()) => format!("Comment added to task {}.", short_id(&task.task_id)),
                    Err(error) => format!("Could not add comment: {error}"),
                }
            }
            "retry" | "cancel" => {
                let Some(reference) = parts.next() else {
                    return format!("Usage: /work {action} <task-id>");
                };
                let task = match self.resolve_work_task(&project_id, reference).await {
                    Ok(task) => task,
                    Err(message) => return message,
                };
                let result = if action == "retry" {
                    self.state
                        .retry_work_task(&task.task_id, "owner", Some(session_id))
                        .await
                } else {
                    self.state
                        .cancel_work_task(&task.task_id, "owner", Some(session_id))
                        .await
                };
                match result {
                    Ok(true) if action == "retry" => {
                        format!("Task {} was queued for retry.", short_id(&task.task_id))
                    }
                    Ok(true) => format!("Task {} was cancelled.", short_id(&task.task_id)),
                    Ok(false) if action == "retry" => {
                        format!("Task {} cannot be retried.", short_id(&task.task_id))
                    }
                    Ok(false) => format!("Task {} cannot be cancelled.", short_id(&task.task_id)),
                    Err(error) => format!("Could not {action} task: {error}"),
                }
            }
            "assign" => {
                let (Some(reference), Some(profile_id)) = (parts.next(), parts.next()) else {
                    return "Usage: /work assign <task-id> <profile-id>".to_string();
                };
                let task = match self.resolve_work_task(&project_id, reference).await {
                    Ok(task) => task,
                    Err(message) => return message,
                };
                match self
                    .state
                    .assign_task_worker_profile(
                        &task.task_id,
                        profile_id,
                        "owner",
                        Some(session_id),
                    )
                    .await
                {
                    Ok(true) => format!(
                        "Task {} assigned to profile {}.",
                        short_id(&task.task_id),
                        profile_id
                    ),
                    Ok(false) => "Task or worker profile was unavailable.".to_string(),
                    Err(error) => format!("Could not assign worker profile: {error}"),
                }
            }
            "workspace" => {
                let Some(reference) = parts.next() else {
                    return "Usage: /work workspace <task-id> [shared|isolated|worktree]"
                        .to_string();
                };
                let task = match self.resolve_work_task(&project_id, reference).await {
                    Ok(task) => task,
                    Err(message) => return message,
                };
                if let Some(policy) = parts.next() {
                    return match self
                        .state
                        .set_task_workspace_policy(&task.task_id, policy)
                        .await
                    {
                        Ok(true) => format!(
                            "Task {} workspace policy set to {}.",
                            short_id(&task.task_id),
                            policy
                        ),
                        Ok(false) => {
                            "Workspace policy can only change before execution.".to_string()
                        }
                        Err(error) => format!("Could not set workspace policy: {error}"),
                    };
                }
                match self.state.get_task_workspace(&task.task_id).await {
                    Ok(Some(workspace)) => format!(
                        "Task {} workspace\nPolicy: {}\nStatus: {}\nPath: {}\nBranch: {}",
                        short_id(&task.task_id),
                        workspace.policy,
                        workspace.status,
                        workspace.root_path,
                        workspace.branch_name.as_deref().unwrap_or("—")
                    ),
                    Ok(None) => match self.state.get_task_workspace_policy(&task.task_id).await {
                        Ok(policy) => format!(
                            "Task {} has policy {}; no workspace has been created yet.",
                            short_id(&task.task_id),
                            policy
                        ),
                        Err(error) => format!("Could not inspect workspace: {error}"),
                    },
                    Err(error) => format!("Could not inspect workspace: {error}"),
                }
            }
            "profiles" => match self.state.list_worker_profiles(Some(&project_id)).await {
                Ok(profiles) if profiles.is_empty() => "No enabled worker profiles.".to_string(),
                Ok(profiles) => {
                    let lines = profiles
                        .into_iter()
                        .map(|profile| {
                            format!(
                                "• {} — {} (max {}, {})",
                                profile.id,
                                profile.name,
                                profile.max_concurrency,
                                profile.workspace_policy
                            )
                        })
                        .collect::<Vec<_>>();
                    format!("Worker profiles\n{}", lines.join("\n"))
                }
                Err(error) => format!("Could not list worker profiles: {error}"),
            },
            "projects" => match self.state.list_work_projects().await {
                Ok(projects) => {
                    let lines = projects
                        .into_iter()
                        .map(|project| {
                            let marker = if project.id == project_id {
                                " (active)"
                            } else {
                                ""
                            };
                            format!("• {} — {}{}", project.id, project.name, marker)
                        })
                        .collect::<Vec<_>>();
                    format!("Projects\n{}", lines.join("\n"))
                }
                Err(error) => format!("Could not list projects: {error}"),
            },
            "project" => {
                let Some(reference_head) = parts.next() else {
                    return format!("Active project: {project_id}\nUsage: /work project <id>");
                };
                if reference_head.eq_ignore_ascii_case("create") {
                    let name = parts.collect::<Vec<_>>().join(" ");
                    if name.trim().is_empty() {
                        return "Usage: /work project create <name>".to_string();
                    }
                    return match self.state.create_work_project(&name, None).await {
                        Ok(project) => match self
                            .state
                            .set_session_work_project(session_id, &project.id)
                            .await
                        {
                            Ok(true) => format!(
                                "Created and selected project {} ({})",
                                project.name, project.id
                            ),
                            Ok(false) => {
                                "Project was created but could not be selected.".to_string()
                            }
                            Err(error) => {
                                format!("Project was created but selection failed: {error}")
                            }
                        },
                        Err(error) => format!("Could not create project: {error}"),
                    };
                }
                let reference = std::iter::once(reference_head)
                    .chain(parts)
                    .collect::<Vec<_>>()
                    .join(" ");
                let projects = match self.state.list_work_projects().await {
                    Ok(projects) => projects,
                    Err(error) => return format!("Could not list projects: {error}"),
                };
                let matches = projects
                    .into_iter()
                    .filter(|project| {
                        project.id == reference || project.name.eq_ignore_ascii_case(&reference)
                    })
                    .collect::<Vec<_>>();
                let Some(project) = matches.first() else {
                    return "Project not found. Use /work projects to list choices.".to_string();
                };
                match self
                    .state
                    .set_session_work_project(session_id, &project.id)
                    .await
                {
                    Ok(true) => format!("Active project set to {}.", project.name),
                    Ok(false) => "Project not found.".to_string(),
                    Err(error) => format!("Could not select project: {error}"),
                }
            }
            _ => work_usage(),
        }
    }

    async fn render_work_overview(&self, project_id: &str) -> String {
        match self.state.list_work_goals(project_id, false, 25).await {
            Ok(goals) if goals.is_empty() => {
                format!("No active work in project {project_id}.")
            }
            Ok(goals) => {
                let lines = goals
                    .into_iter()
                    .map(|goal| {
                        format!(
                            "• {} {} — ready {} · running {} · blocked {} · attention {} · done {}",
                            short_id(&goal.goal_id),
                            truncate_chars(&goal.description, 72),
                            goal.ready,
                            goal.in_progress,
                            goal.blocked,
                            goal.needs_attention,
                            goal.done
                        )
                    })
                    .collect::<Vec<_>>();
                format!(
                    "Active work · project {project_id}\n{}\n\nUse /work ready, /work blocked, or /work show <task-id>.",
                    lines.join("\n")
                )
            }
            Err(error) => format!("Could not load work: {error}"),
        }
    }

    async fn render_work_lane(&self, project_id: &str, lane: &str) -> String {
        match self.state.list_work_tasks(project_id, Some(lane), 50).await {
            Ok(tasks) if tasks.is_empty() => format!("No tasks in the {lane} lane."),
            Ok(tasks) => {
                let lines = tasks
                    .into_iter()
                    .map(|task| {
                        let owner = task
                            .worker_profile
                            .as_deref()
                            .map(|profile| format!(" · {profile}"))
                            .unwrap_or_default();
                        format!(
                            "• {} [{}] {}{}",
                            short_id(&task.task_id),
                            task.priority,
                            truncate_chars(&task.description, 90),
                            owner
                        )
                    })
                    .collect::<Vec<_>>();
                format!("{lane}\n{}", lines.join("\n"))
            }
            Err(error) => format!("Could not load work: {error}"),
        }
    }

    async fn resolve_work_task(
        &self,
        project_id: &str,
        reference: &str,
    ) -> Result<WorkTaskSummary, String> {
        let tasks = self
            .state
            .list_work_tasks(project_id, None, 500)
            .await
            .map_err(|error| format!("Could not load work: {error}"))?;
        let mut matches = tasks
            .into_iter()
            .filter(|task| task.task_id == reference || task.task_id.starts_with(reference))
            .collect::<Vec<_>>();
        match matches.len() {
            0 => Err("Task not found in the active project run.".to_string()),
            1 => Ok(matches.remove(0)),
            _ => Err("Task reference is ambiguous; provide more ID characters.".to_string()),
        }
    }

    async fn render_work_task(&self, task: &WorkTaskSummary) -> String {
        let journal = self
            .state
            .get_task_journal(&task.task_id, 8)
            .await
            .unwrap_or_default();
        let handoff = self
            .state
            .get_latest_task_handoff(&task.task_id)
            .await
            .ok()
            .flatten();
        let workspace = self
            .state
            .get_task_workspace(&task.task_id)
            .await
            .ok()
            .flatten();
        let mut reply = format!(
            "Task {}\n{}\nLane: {} · status: {} · priority: {}\nGoal: {}\nWorker: {}",
            short_id(&task.task_id),
            task.description,
            task.lane,
            task.status,
            task.priority,
            truncate_chars(&task.goal_description, 100),
            task.worker_profile.as_deref().unwrap_or("unassigned")
        );
        if let Some(blocker) = &task.blocker {
            reply.push_str(&format!("\nBlocker: {blocker}"));
        }
        if let Some(handoff) = handoff {
            reply.push_str(&format!("\n\nLatest handoff: {}", handoff.summary));
            if !handoff.verification.is_empty() {
                reply.push_str(&format!("\nVerified: {}", handoff.verification.join("; ")));
            }
            if let Some(risk) = handoff.remaining_risk {
                reply.push_str(&format!("\nRemaining risk: {risk}"));
            }
            if let Some(next) = handoff.next_step {
                reply.push_str(&format!("\nNext step: {next}"));
            }
        }
        if let Some(workspace) = workspace {
            reply.push_str(&format!(
                "\n\nWorkspace: {} ({})",
                workspace.root_path, workspace.status
            ));
        }
        if !journal.is_empty() {
            let lines = journal
                .into_iter()
                .map(|entry| {
                    format!(
                        "• {} · {}: {}",
                        entry.created_at,
                        entry.entry_type,
                        truncate_chars(&entry.body, 120)
                    )
                })
                .collect::<Vec<_>>();
            reply.push_str(&format!("\n\nRecent journal\n{}", lines.join("\n")));
        }
        reply
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

fn short_id(id: &str) -> &str {
    let end = id
        .char_indices()
        .nth(8)
        .map(|(index, _)| index)
        .unwrap_or(id.len());
    &id[..end]
}

fn truncate_chars(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut truncated = value
        .chars()
        .take(max.saturating_sub(1))
        .collect::<String>();
    truncated.push('…');
    truncated
}

fn work_usage() -> String {
    "Usage:\n\
     /work — active overview\n\
     /work ready|running|blocked|attention|done\n\
     /work show <task-id>\n\
     /work comment <task-id> <message>\n\
     /work unblock <task-id> <resolution>\n\
     /work retry|cancel <task-id>\n\
     /work assign <task-id> <profile-id>\n\
     /work workspace <task-id> [shared|isolated|worktree]\n\
     /work profiles|projects\n\
     /work project <id>\n\
     /work project create <name>"
        .to_string()
}
