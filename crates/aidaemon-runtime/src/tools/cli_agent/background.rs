//! Background runs: persisting results, re-engaging the turn, and
//! `check`, `cancel` and `list`.

use super::*;

impl CliAgentTool {
    pub(super) async fn persist_delegated_cli_result_with_state(
        state: Arc<dyn CliAgentStore>,
        delegated_task_id: &str,
        response: Option<&str>,
        error: Option<&str>,
    ) {
        let latest_task = state.get_task(delegated_task_id).await.ok().flatten();
        let structured =
            derive_executor_step_result(delegated_task_id, latest_task.as_ref(), response, error);
        let task_lead_summary = structured.render_task_lead_summary();

        if let Ok(Some(mut task)) = state.get_task(delegated_task_id).await {
            if let Ok(context) =
                persist_executor_result_context(task.context.as_deref(), &structured)
            {
                task.context = Some(context);
            }

            match error {
                Some(error) => {
                    task.status = "failed".to_string();
                    task.error = Some(error.to_string());
                    task.completed_at = Some(chrono::Utc::now().to_rfc3339());
                }
                None => {
                    if matches!(structured.task_outcome, TaskValidationOutcome::TaskDone) {
                        if task
                            .result
                            .as_deref()
                            .is_none_or(|result| result.trim().is_empty())
                        {
                            if let Some(response) = response {
                                task.result = Some(response.to_string());
                            } else {
                                task.result = Some(structured.summary.clone());
                            }
                        }
                        task.status = "completed".to_string();
                        task.blocker = None;
                    } else {
                        task.result = Some(task_lead_summary.clone());
                        task.status = "blocked".to_string();
                        task.blocker = structured
                            .blocker
                            .clone()
                            .or_else(|| structured.exact_need.clone())
                            .or_else(|| Some(structured.summary.clone()));
                    }
                    task.completed_at = Some(chrono::Utc::now().to_rfc3339());
                }
            }

            let _ = state.update_task(&task).await;
        }
    }

    pub(super) fn prune_completed_map(completed: &mut HashMap<String, CompletedCliAgent>) {
        // Keep results for as long as dialogue-state can keep an unfinished
        // request open. A ten-minute cache made ordinary delayed follow-ups
        // lose successful delegated work and turn a recoverable typo into a
        // dead end.
        const COMPLETED_TTL: Duration = Duration::from_secs(12 * 60 * 60);
        const COMPLETED_CAP: usize = 128;

        completed.retain(|_, entry| entry.completed_at.elapsed() <= COMPLETED_TTL);
        if completed.len() <= COMPLETED_CAP {
            return;
        }

        let mut by_age: Vec<(String, Instant)> = completed
            .iter()
            .map(|(task_id, entry)| (task_id.clone(), entry.completed_at))
            .collect();
        by_age.sort_by_key(|(_, ts)| *ts);
        let to_remove = by_age.len().saturating_sub(COMPLETED_CAP);
        for (task_id, _) in by_age.into_iter().take(to_remove) {
            completed.remove(&task_id);
        }
    }

    pub(super) async fn build_finished_result(
        backend: &SharedExecutionBackend,
        agent: &RunningCliAgent,
    ) -> String {
        let elapsed = agent.running_since().elapsed().as_secs();
        let stdout_output = agent.stdout_buf.lock().await.clone();
        if agent.auth_failed.load(Ordering::Acquire) {
            return format!(
                "CLI agent '{}' failed after {}s.\n\n{}",
                agent.tool_name,
                elapsed,
                Self::auth_error_message(&agent.tool_name)
            );
        }
        // Its end's verdict when it was judged (a run that failed reads as
        // failed, with why), else what its output says.
        let verdict = agent.verdict.get();
        let result = match verdict {
            Some(verdict) => truncate_with_note(&verdict.text, 10000),
            None => extract_meaningful_output(&stdout_output, 10000),
        };
        let ended = match verdict {
            Some(verdict) if !verdict.success => "failed",
            _ => "finished",
        };

        // Capture git diff for finished background tasks.
        let diff_section = if let Some(base) = agent.diff_dir.get() {
            Self::capture_git_diff(backend, base)
                .await
                .map(|diff| format!("\n\n## File Changes\n```diff\n{}\n```", diff))
        } else {
            None
        };

        let mut final_result = format!(
            "CLI agent '{}' {ended} after {}s.\n\nResult:\n{}",
            agent.tool_name, elapsed, result
        );
        if let Some(diff) = diff_section {
            final_result.push_str(&diff);
        }
        if let Some(section) = agent
            .left_running
            .get()
            .and_then(left_running::LeftRunning::result_section)
        {
            final_result.push_str("\n\n");
            final_result.push_str(&section);
        }
        final_result
    }

    pub(super) fn build_background_reengagement_followup(
        tool_name: &str,
        task_id: &str,
        task_context: &str,
        status: &str,
        output: &str,
    ) -> String {
        format!(
            "[Background command completed]\n\
             Command: `cli_agent {tool_name}`\n\
             Runtime task ID: {task_id}\n\
             Delegated step: {task_context}\n\
             Exit status: {status}\n\
             Output:\n{output}\n\n\
             This delegated step was part of the user's previous unfinished request. Check the session history for the requested outcome and continue from this evidence. The exit status above is the runtime's verdict on the step; how long the step took is the runtime's to state on its card, so never repeat a time the agent reported for its own run. Delegated work is progress, not automatically fulfillment: complete every remaining implementation, action, validation, or delivery step that the original request requires. Do not merely announce that the delegated step ended, and do not ask the user to repeat the request."
        )
    }

    /// Clean up any finished CLI agent tasks.
    pub(super) async fn reap_finished(&self) {
        let finished: Vec<(String, RunningCliAgent)> = {
            let mut running = self.running.lock().await;
            let finished_ids: Vec<String> = running
                .iter()
                .filter(|(_, agent)| agent.finished.load(Ordering::Acquire))
                .map(|(id, _)| id.clone())
                .collect();

            // Also release working-dir claims for finished tasks.
            // Lock ordering: running -> working_dir_claims.
            let mut claims = lock_claims(&self.working_dir_claims);
            let mut removed: Vec<(String, RunningCliAgent)> = Vec::new();
            for task_id in finished_ids {
                if let Some(agent) = running.remove(&task_id) {
                    if let Some(ref dir) = agent.claim_dir {
                        release_claim(&mut claims, dir, &task_id);
                    }
                    removed.push((task_id, agent));
                }
            }
            removed
        };

        for (task_id, agent) in finished {
            let final_result = Self::build_finished_result(&self.backend, &agent).await;
            let mut completed = self.completed.lock().await;
            completed.insert(
                task_id.clone(),
                CompletedCliAgent {
                    result: final_result,
                    completed_at: Instant::now(),
                    session_id: agent.session_id.clone(),
                },
            );
            Self::prune_completed_map(&mut completed);
            info!(task_id, tool = %agent.tool_name, "Reaped finished CLI agent");
        }
    }

    /// Record the agent as a background run and show it on its turn's card.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn start_background_run(
        &self,
        slot: &std::sync::OnceLock<Arc<BackgroundRun>>,
        task_id: &str,
        session_id: &str,
        label: &str,
        turn_task_id: Option<&str>,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        work_began: chrono::DateTime<chrono::Utc>,
    ) {
        if session_id.trim().is_empty() || slot.get().is_some() {
            return;
        }
        let run = BackgroundRun::start(
            self.run_home(),
            crate::background_run::RunStart {
                id: &cli_run_id(task_id),
                chat: session_id,
                label,
                task_id: turn_task_id,
                status_tx,
                work_began,
            },
        )
        .await;
        let _ = slot.set(run);
    }

    pub(super) async fn persist_delegated_cli_handoff(
        state: &Arc<dyn CliAgentStore>,
        delegated_task_id: &str,
        tool_name: &str,
        prompt: &str,
        working_dir: Option<&str>,
    ) {
        let expected_targets = working_dir
            .and_then(|dir| ToolTargetHint::new(ToolTargetHintKind::ProjectScope, dir))
            .into_iter()
            .collect::<Vec<_>>();
        let handoff = ExecutorHandoff {
            task_id: delegated_task_id.to_string(),
            mission: format!("cli_agent:{tool_name}"),
            task_description: prompt.to_string(),
            target_scope: TargetScope {
                allowed_targets: expected_targets.clone(),
            },
            expected_targets,
            allowed_tools: Some(vec![format!("cli_agent:{tool_name}")]),
        };

        if let Ok(Some(mut task)) = state.get_task(delegated_task_id).await {
            task.status = "running".to_string();
            if task.started_at.is_none() {
                task.started_at = Some(chrono::Utc::now().to_rfc3339());
            }
            if let Ok(context) = persist_executor_handoff_context(task.context.as_deref(), &handoff)
            {
                task.context = Some(context);
            }
            let _ = state.update_task(&task).await;
        }
    }

    pub(super) async fn persist_delegated_cli_result(
        &self,
        delegated_task_id: &str,
        response: Option<&str>,
        error: Option<&str>,
    ) {
        Self::persist_delegated_cli_result_with_state(
            self.state.clone(),
            delegated_task_id,
            response,
            error,
        )
        .await;
    }

    /// Check on a background CLI agent task.
    pub(super) async fn handle_check(
        &self,
        task_id: &str,
        session_id: &str,
    ) -> anyhow::Result<String> {
        self.wait_for_finish(task_id, CHECK_WAIT).await;
        let running = self.running.lock().await;

        let Some(agent) = running.get(task_id) else {
            drop(running);
            if let Some(done) = self.completed.lock().await.get(task_id).cloned() {
                mark_background_settled();
                return Ok(done.result);
            }

            if let Some(invocation) = self
                .state
                .get_cli_agent_invocations(128)
                .await?
                .into_iter()
                .find(|invocation| invocation.task_id.as_deref() == Some(task_id))
            {
                let result = invocation
                    .output_summary
                    .as_deref()
                    .unwrap_or("No persisted output was recorded.");
                if invocation.completed_at.is_some() {
                    mark_background_settled();
                }
                return Ok(format!(
                    "CLI agent '{}' {}. Recovered durable result for task_id={task_id}.\n\nResult:\n{}",
                    invocation.agent_name,
                    if invocation.completed_at.is_some() {
                        "finished"
                    } else {
                        "was started but has no recorded completion"
                    },
                    result
                ));
            }

            // Background task identifiers are opaque. Models occasionally
            // copy a provider thread UUID instead of the short runtime ID.
            // Recover an unambiguous session-local task instead of converting
            // that bookkeeping miss into an abandoned user objective.
            let session_candidates = {
                let running = self.running.lock().await;
                let mut candidates = running
                    .iter()
                    .filter(|(_, agent)| agent.session_id == session_id)
                    .map(|(id, agent)| {
                        (
                            id.clone(),
                            format!("running — {}", agent.prompt_summary),
                            None,
                        )
                    })
                    .collect::<Vec<_>>();
                drop(running);

                let completed = self.completed.lock().await;
                candidates.extend(
                    completed
                        .iter()
                        .filter(|(_, done)| done.session_id == session_id)
                        .map(|(id, done)| {
                            (
                                id.clone(),
                                "finished — result available".to_string(),
                                Some(done.result.clone()),
                            )
                        }),
                );
                candidates
            };

            if session_candidates.len() == 1 {
                let (recovered_id, status, result) = &session_candidates[0];
                if let Some(result) = result {
                    mark_background_settled();
                    return Ok(format!(
                        "Requested task_id '{task_id}' was not found. Recovered the only recent CLI task for this session instead (task_id={recovered_id}, {status}).\n\n{result}"
                    ));
                }
                mark_left_in_background();
                return Ok(format!(
                    "Requested task_id '{task_id}' was not found. Recovered the only active CLI task for this session (task_id={recovered_id}, {status}). Use action=\"check\" with task_id=\"{recovered_id}\"."
                ));
            }

            let candidate_lines = session_candidates
                .iter()
                .take(8)
                .map(|(id, status, _)| format!("- task_id={id}: {status}"))
                .collect::<Vec<_>>();
            let recovery = if candidate_lines.is_empty() {
                "No recent CLI tasks are tracked for this session. Start a new in-scope run or inspect durable invocation history; do not treat this lookup miss as completion."
                    .to_string()
            } else {
                format!(
                    "Recent CLI tasks for this session:\n{}\nCheck the matching task ID; do not treat this lookup miss as completion.",
                    candidate_lines.join("\n")
                )
            };
            return Ok(format!(
                "No CLI agent task matched task_id '{task_id}'. {recovery}"
            ));
        };

        if agent.process.is_queued() {
            let waited = agent.accepted_at.elapsed().as_secs();
            mark_left_in_background();
            let what = match agent.waits_for_folder() {
                Some(holder) => {
                    format!("for {holder} to finish in its folder and starts by itself once it has")
                }
                None => "for a free slot and starts by itself when one frees".to_string(),
            };
            return Ok(format!(
                "CLI agent '{}' is queued (task_id={task_id}): it has waited {waited}s {what}. \
                 Do not start it again.\nTask: {}...",
                agent.tool_name, agent.prompt_summary
            ));
        }
        let elapsed = agent.running_since().elapsed().as_secs();
        let display_output = agent.display_buf.lock().await.clone();

        let is_running = !agent.finished.load(Ordering::Acquire);

        if !is_running {
            mark_background_settled();
            Ok(Self::build_finished_result(&self.backend, agent).await)
        } else {
            mark_left_in_background();
            Ok(format!(
                "CLI agent '{}' still running ({}s elapsed, pid={}).\n\
                 Task: {}...\n\n\
                 Partial output ({} chars):\n{}",
                agent.tool_name,
                elapsed,
                agent
                    .process
                    .handle()
                    .map(|handle| handle.display_id())
                    .unwrap_or_default(),
                agent.prompt_summary,
                display_output.len(),
                crate::tools::terminal::keep_head_and_tail(&display_output, 5000)
            ))
        }
    }

    /// Wait up to `wait` for a running task to finish, without holding the
    /// task table's lock (the stream task and other calls need it). Returns at
    /// once for a task that is unknown or already finished.
    pub(super) async fn wait_for_finish(&self, task_id: &str, wait: Duration) {
        let finished = match self.running.lock().await.get(task_id) {
            Some(agent) => agent.finished.clone(),
            None => return,
        };
        let deadline = tokio::time::Instant::now() + wait;
        while !finished.load(Ordering::Acquire) && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(CHECK_POLL).await;
        }
    }

    /// Cancel a background CLI agent task.
    pub(super) async fn handle_cancel(&self, task_id: &str) -> anyhow::Result<String> {
        let agent = {
            let mut running = self.running.lock().await;
            let Some(agent) = running.remove(task_id) else {
                return Ok(format!("No running CLI agent with task_id '{}'", task_id));
            };
            agent
        };
        self.completed.lock().await.remove(task_id);

        let display_output = agent.display_buf.lock().await.clone();
        // A queued run's time is how long it waited; a started one's, how
        // long it has run.
        let elapsed = agent.running_since().elapsed().as_secs();

        // Release working-dir claim owned by this task.
        if let Some(ref dir) = agent.claim_dir {
            self.release_working_dir_claim(dir, task_id);
        }

        // Kill its process; a queued run never starts.
        let queued = agent.process.is_queued();
        agent.process.stop(&self.backend).await;
        mark_background_settled();

        if let Some(ref delegated_task_id) = agent.delegated_task_id {
            Self::persist_delegated_cli_result_with_state(
                self.state.clone(),
                delegated_task_id,
                None,
                Some("Executor CLI run was cancelled before completion."),
            )
            .await;
        }

        if queued {
            return Ok(format!(
                "Cancelled CLI agent '{}' before it started: it waited {elapsed}s for a free \
                 slot and never ran.",
                agent.tool_name
            ));
        }
        Ok(format!(
            "Cancelled CLI agent '{}' (was running for {}s).\n\nOutput before cancellation:\n{}",
            agent.tool_name,
            elapsed,
            crate::tools::terminal::keep_head_and_tail(&display_output, 5000)
        ))
    }

    /// Cancel all CLI agent tasks for a specific session.
    pub(super) async fn handle_cancel_all(&self, session_id: &str) -> anyhow::Result<String> {
        // Find all tasks matching this session and remove them from tracking first.
        let to_cancel: Vec<(String, RunningCliAgent)> = {
            let mut running = self.running.lock().await;
            let task_ids: Vec<String> = running
                .iter()
                .filter(|(_, agent)| agent.session_id == session_id)
                .map(|(task_id, _)| task_id.clone())
                .collect();

            let mut removed = Vec::new();
            for task_id in task_ids {
                if let Some(agent) = running.remove(&task_id) {
                    removed.push((task_id, agent));
                }
            }
            removed
        };

        if to_cancel.is_empty() {
            return Ok("No running CLI agents for this session.".to_string());
        }

        let mut cancelled = Vec::new();
        for (task_id, agent) in to_cancel {
            self.completed.lock().await.remove(&task_id);
            if let Some(ref dir) = agent.claim_dir {
                self.release_working_dir_claim(dir, &task_id);
            }
            agent.process.stop(&self.backend).await;
            mark_background_settled();
            if let Some(ref delegated_task_id) = agent.delegated_task_id {
                Self::persist_delegated_cli_result_with_state(
                    self.state.clone(),
                    delegated_task_id,
                    None,
                    Some("Executor CLI run was cancelled before completion."),
                )
                .await;
            }
            cancelled.push(format!("{} ({})", agent.tool_name, task_id));
        }

        Ok(format!(
            "Cancelled {} CLI agent(s): {}",
            cancelled.len(),
            cancelled.join(", ")
        ))
    }

    /// List all running CLI agent tasks.
    pub(super) async fn handle_list(&self) -> anyhow::Result<String> {
        let running = self.running.lock().await;

        if running.is_empty() {
            return Ok("No CLI agents currently running.".to_string());
        }

        let mut lines = vec!["Running CLI agents:".to_string()];
        for (task_id, agent) in running.iter() {
            let elapsed = agent.running_since().elapsed().as_secs();
            let status = if let Some(holder) = agent
                .waits_for_folder()
                .filter(|_| agent.process.is_queued())
            {
                format!("queued, waiting for {holder} to finish in its folder")
            } else if agent.process.is_queued() {
                "queued, waiting for a free slot".to_string()
            } else if !agent.finished.load(Ordering::Acquire) {
                "running".to_string()
            } else {
                "finished".to_string()
            };
            lines.push(format!(
                "  {} - {} ({}, {}s): {}...",
                task_id, agent.tool_name, status, elapsed, agent.prompt_summary
            ));
        }

        Ok(lines.join("\n"))
    }
}
