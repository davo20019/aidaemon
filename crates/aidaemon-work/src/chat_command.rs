//! `/work` in chat: the owner's view of durable work (lanes, tasks, journal
//! comments, retries, workspaces, worker profiles and projects), over the
//! [`WorkCoordinationStore`]. A host adds it to the shared chat commands.

use std::sync::Arc;

use aidaemon_channels::slash::{SlashCall, SlashCommand};
use aidaemon_tools::semantics::UserRole;
use async_trait::async_trait;

use crate::goals::{TaskJournalEntry, WorkTaskSummary};
use crate::stores::WorkCoordinationStore;

/// The `/work` command.
pub struct WorkCommand {
    state: Arc<dyn WorkCoordinationStore>,
}

impl WorkCommand {
    pub fn new(state: Arc<dyn WorkCoordinationStore>) -> Self {
        Self { state }
    }
}

#[async_trait]
impl SlashCommand for WorkCommand {
    fn handles(&self, cmd: &str) -> bool {
        cmd == "/work"
    }

    async fn run(&self, _cmd: &str, args: &str, call: &SlashCall<'_>) -> String {
        self.handle_work(args, call.session_id, call.user_role)
            .await
    }
}

impl WorkCommand {
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
