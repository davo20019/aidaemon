//! The work a chat has running besides its turns, and stopping it
//! ([`ChatWorkControl`], on any [`TurnHost`]): its sub-agents and background
//! commands, its finite goals and background spawns.

use aidaemon_channels::ports::{ChatWork, ChatWorkKind, ChatWorkView};
use aidaemon_work::traits::Goal;

use crate::host::TurnHost;

/// An active goal that a session's stop or correction cancels: finite
/// orchestration work, never a recurring or personal goal.
pub fn session_goal_is_finite_work(goal: &Goal) -> bool {
    matches!(
        goal.status.as_str(),
        "active" | "pending" | "pending_confirmation"
    ) && goal.domain == "orchestration"
        && goal.goal_type == "finite"
}

/// Listing and stopping a chat's work.
#[async_trait::async_trait]
pub trait ChatWorkControl: TurnHost {
    /// Cancel the finite work the session started (its finite goals and
    /// background spawns); what was cancelled.
    async fn cancel_active_finite_work_for_session(&self, session_id: &str) -> Vec<String> {
        let mut cancelled = self.cancel_session_finite_goals(session_id).await;
        cancelled.extend(self.cancel_session_background_spawns(session_id).await);
        cancelled
    }

    /// Stop the session's background sub-agents; what was stopped.
    async fn cancel_session_background_spawns(&self, session_id: &str) -> Vec<String> {
        let Some(registry) = self.goal_tokens() else {
            return Vec::new();
        };
        let spawns = self
            .work()
            .background_spawns_for_session(session_id)
            .await
            .unwrap_or_default();
        let mut cancelled = Vec::new();
        for spawn in spawns {
            if registry.cancel(&spawn.id).await {
                let mission: String = spawn.mission.chars().take(80).collect();
                cancelled.push(format!("background task: {mission}"));
            }
        }
        cancelled
    }

    /// Everything running for the chat besides its turns, from typed state:
    /// its sub-agents (the running-children registry) and its background
    /// commands (each tool's process table).
    async fn chat_work(&self, session_id: &str) -> ChatWorkView {
        let mut items: Vec<ChatWork> = self
            .running_children()
            .listed(session_id)
            .into_iter()
            .map(|child| ChatWork {
                key: child.key,
                kind: ChatWorkKind::SubAgent,
                title: format!("Sub-agent: {}", child.child.label),
                started_at: child.child.started_at,
                request: child.request,
                step: child.step,
            })
            .collect();
        for (index, tool) in self.tools().iter().enumerate() {
            for job in tool.background_jobs(session_id).await {
                items.push(ChatWork {
                    key: format!("job:{index}:{}", job.key),
                    kind: ChatWorkKind::BackgroundJob,
                    title: job.title,
                    started_at: job.started_at,
                    request: job.task_id,
                    step: None,
                });
            }
        }
        items.sort_by_key(|item| item.started_at);
        ChatWorkView {
            current_request: self.running_children().current_request(session_id),
            items,
        }
    }

    /// Stop the chat's work `key` and read it back: a sub-agent is stopped
    /// with the commands it left running, and counts as stopped once it is no
    /// longer listed and none of those commands runs; a command once its
    /// process is gone.
    async fn stop_chat_work(&self, session_id: &str, key: &str) -> bool {
        const SETTLE: std::time::Duration = std::time::Duration::from_secs(5);
        const POLL: std::time::Duration = std::time::Duration::from_millis(100);
        if let Some(job) = key.strip_prefix("job:") {
            let Some((index, job_key)) = job.split_once(':') else {
                return false;
            };
            let Some(tool) = index
                .parse::<usize>()
                .ok()
                .and_then(|i| self.tools().get(i))
            else {
                return false;
            };
            if !tool
                .background_jobs(session_id)
                .await
                .iter()
                .any(|listed| listed.key == job_key)
            {
                // Not running for this chat (it ended, or it is not its).
                return true;
            }
            return tool.stop_background_job(job_key).await;
        }
        let Some(child_session) = self.running_children().stop(key) else {
            return true;
        };
        let deadline = tokio::time::Instant::now() + SETTLE;
        while self.running_children().is_listed(key) && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(POLL).await;
        }
        // A stopped child's run is dropped before its own end could stop the
        // commands it started: stop them here.
        let mut jobs_stopped = true;
        for tool in self.tools() {
            for job in tool.background_jobs(&child_session).await {
                jobs_stopped &= tool.stop_background_job(&job.key).await;
            }
        }
        !self.running_children().is_listed(key) && jobs_stopped
    }

    /// Cancel the session's finite goals in flight, with their schedules and
    /// open tasks; their descriptions.
    async fn cancel_session_finite_goals(&self, session_id: &str) -> Vec<String> {
        let store = self.work();
        let goals = store
            .get_goals_for_session(session_id)
            .await
            .unwrap_or_default();
        let mut active: Vec<&Goal> = Vec::new();
        for goal in goals
            .iter()
            .filter(|goal| session_goal_is_finite_work(goal))
        {
            // A scheduled one-shot (a reminder) is future work the owner
            // asked for, not work in flight: a stop or /clear ends it only
            // while its run is executing.
            let scheduled = store
                .get_schedules_for_goal(&goal.id)
                .await
                .map(|schedules| !schedules.is_empty())
                .unwrap_or(false);
            let running = self
                .goal_tokens()
                .is_some_and(|registry| registry.is_run_active(&goal.id));
            if scheduled && !running {
                continue;
            }
            active.push(goal);
        }
        if active.is_empty() {
            return Vec::new();
        }

        let now = chrono::Utc::now().to_rfc3339();
        let mut cancelled = Vec::new();
        for goal in active {
            if let Some(registry) = self.goal_tokens() {
                registry.cancel(&goal.id).await;
                registry.clear_run_budget(&goal.id).await;
            }
            aidaemon_work::scheduled_runs::clear_scheduled_run_state(store, &goal.id).await;

            let mut updated = goal.clone();
            updated.status = "cancelled".to_string();
            updated.updated_at = now.clone();
            updated.completed_at = Some(now.clone());
            let _ = store.update_goal(&updated).await;

            // Best-effort cleanup: cancelled goals should not retain schedules.
            if let Ok(schedules) = store.get_schedules_for_goal(&updated.id).await {
                for s in &schedules {
                    let _ = store.delete_goal_schedule(&s.id).await;
                }
            }

            // Cancel all remaining tasks for this goal.
            if let Ok(tasks) = store.get_tasks_for_goal(&updated.id).await {
                for task in tasks {
                    if task.status != "completed"
                        && task.status != "failed"
                        && task.status != "cancelled"
                    {
                        let mut t = task.clone();
                        t.status = "cancelled".to_string();
                        t.completed_at = Some(now.clone());
                        let _ = store.update_task(&t).await;
                    }
                }
            }

            cancelled.push(updated.description.chars().take(100).collect());
        }

        cancelled
    }
}

impl<H: TurnHost + ?Sized> ChatWorkControl for H {}

#[cfg(test)]
mod tests {
    use aidaemon_work::traits::Goal;

    use super::session_goal_is_finite_work;

    #[test]
    fn stop_cancels_finite_work_but_keeps_recurring_and_personal_goals() {
        let finite = Goal::new_finite("Answer the current question", "session-1");
        let recurring = Goal::new_continuous("Publish a daily blog post", "session-1", None, None);
        let personal = Goal::new_personal("Keep writing", "session-1");

        assert!(session_goal_is_finite_work(&finite));
        assert!(!session_goal_is_finite_work(&recurring));
        assert!(!session_goal_is_finite_work(&personal));
    }
}
