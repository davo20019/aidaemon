use super::*;

impl Agent {
    /// Spawn a child agent with an incremented depth and a focused mission.
    ///
    /// The child runs its own agentic loop in a fresh session and returns the
    /// final text response. It inherits the parent's provider, state, model,
    /// and non-spawn tools. If the child hasn't reached max_depth it also gets
    /// its own `spawn_agent` tool so it can recurse further.
    ///
    /// When `child_role` is `Some`, tools are scoped by role:
    /// - TaskLead: Management + Universal tools + ManageGoalTasksTool + SpawnAgentTool
    /// - Executor: Action + Universal tools + ReportBlockerTool, NO SpawnAgentTool
    #[allow(clippy::too_many_arguments)]
    pub async fn spawn_child(
        self: &Arc<Self>,
        mission: &str,
        task: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
        child_role: Option<AgentRole>,
        goal_id: Option<&str>,
        task_id: Option<&str>,
    ) -> anyhow::Result<String> {
        if self.depth >= self.max_depth {
            anyhow::bail!(
                "Cannot spawn sub-agent: max recursion depth ({}) reached",
                self.max_depth
            );
        }

        let child_depth = self.depth + 1;
        let model = self.model.read().await.clone();

        // Collect parent's non-spawn tools for the child.
        // Use root_tools if available (TaskLead spawning Executor needs the full
        // unfiltered set so Action tools aren't lost through double-filtering).
        let full_tools: Vec<Arc<dyn Tool>> = self
            .root_tools
            .as_ref()
            .unwrap_or(&self.tools)
            .iter()
            .filter(|t| t.name() != "spawn_agent")
            .cloned()
            .collect();

        // Apply role-based tool scoping when child_role is specified.
        let (scoped_tools, child_system_prompt, child_root_tools) = if let Some(role) = child_role {
            match role {
                AgentRole::TaskLead => {
                    // Task leads get Management + Universal tools
                    let mut tools: Vec<Arc<dyn Tool>> = full_tools
                        .iter()
                        .filter(|t| {
                            matches!(t.tool_role(), ToolRole::Management | ToolRole::Universal)
                        })
                        .cloned()
                        .collect();
                    // Add ManageGoalTasksTool
                    if let Some(gid) = goal_id {
                        tools.push(Arc::new(crate::tools::ManageGoalTasksTool::new(
                            gid.to_string(),
                            self.state.clone(),
                        )));
                    }
                    let goal_context = if let Some(gid) = goal_id {
                        self.state
                            .get_goal(gid)
                            .await
                            .ok()
                            .flatten()
                            .and_then(|g| g.context)
                    } else {
                        None
                    };
                    // SpawnAgentTool added below (for spawning executors)
                    let prompt = Self::build_task_lead_prompt(
                        goal_id.unwrap_or("unknown"),
                        task,
                        goal_context.as_deref(),
                        child_depth,
                        self.max_depth,
                    );
                    // Pass the full unfiltered tools as root_tools so that when
                    // this TaskLead spawns Executor children, they can access
                    // Action tools that were filtered out of the TaskLead's set.
                    (tools, prompt, Some(full_tools.clone()))
                }
                AgentRole::Executor => {
                    // Executors get Action + Universal tools
                    let mut tools: Vec<Arc<dyn Tool>> = full_tools
                        .iter()
                        .filter(|t| matches!(t.tool_role(), ToolRole::Action | ToolRole::Universal))
                        .cloned()
                        .collect();
                    // Add ReportBlockerTool
                    if let Some(tid) = task_id {
                        tools.push(Arc::new(crate::tools::ReportBlockerTool::new(
                            tid.to_string(),
                            self.state.clone(),
                        )));
                    }
                    let prompt = Self::build_executor_prompt(task, child_depth, self.max_depth);
                    // Executors never get SpawnAgentTool
                    return self
                        .spawn_child_inner(
                            &tools,
                            model,
                            prompt,
                            child_depth,
                            mission,
                            task,
                            status_tx,
                            channel_ctx,
                            user_role,
                            role,
                            false, // no spawn tool
                            task_id.map(|s| s.to_string()),
                            None,
                            None, // root_tools (executors don't spawn children)
                        )
                        .await;
                }
                AgentRole::Orchestrator => {
                    // Orchestrator: full loop with spawn available (unless at max depth)
                    let at_max_depth = child_depth >= self.max_depth;
                    let depth_note = if at_max_depth {
                        "\nYou are at the maximum sub-agent depth. You CANNOT spawn further sub-agents; \
                        the `spawn_agent` tool is not available to you. Complete the task directly."
                    } else {
                        ""
                    };
                    let prompt = format!(
                        "{}\n\n## Sub-Agent Context\n\
                        You are a sub-agent (depth {}/{}) spawned to accomplish a specific mission.\n\
                        **Mission:** {}\n\n\
                        Focus exclusively on this mission. Be concise. Return your findings/results \
                        directly — they will be consumed by the parent agent.{}",
                        self.system_prompt, child_depth, self.max_depth, mission, depth_note
                    );
                    (full_tools, prompt, None)
                }
            }
        } else {
            // Legacy behavior: no role scoping
            let at_max_depth = child_depth >= self.max_depth;
            let depth_note = if at_max_depth {
                "\nYou are at the maximum sub-agent depth. You CANNOT spawn further sub-agents; \
                the `spawn_agent` tool is not available to you. Complete the task directly."
            } else {
                ""
            };
            let prompt = format!(
                "{}\n\n## Sub-Agent Context\n\
                You are a sub-agent (depth {}/{}) spawned to accomplish a specific mission.\n\
                **Mission:** {}\n\n\
                Focus exclusively on this mission. Be concise. Return your findings/results \
                directly — they will be consumed by the parent agent.{}",
                self.system_prompt, child_depth, self.max_depth, mission, depth_note
            );
            (full_tools, prompt, None)
        };

        let effective_role = child_role.unwrap_or(AgentRole::Orchestrator);
        let can_spawn = child_depth < self.max_depth && effective_role != AgentRole::Executor;

        // For TaskLead, pass goal_id; other roles get no goal context injection.
        let goal_for_child = if effective_role == AgentRole::TaskLead {
            goal_id.map(|s| s.to_string())
        } else {
            None
        };

        self.spawn_child_inner(
            &scoped_tools,
            model,
            child_system_prompt,
            child_depth,
            mission,
            task,
            status_tx,
            channel_ctx,
            user_role,
            effective_role,
            can_spawn,
            None,             // task_id (executor activity tracking)
            goal_for_child,   // goal_id (task lead context injection)
            child_root_tools, // root_tools for TaskLead → Executor inheritance
        )
        .await
    }

    /// Internal helper to create and run a child agent.
    #[allow(clippy::too_many_arguments)]
    async fn spawn_child_inner(
        self: &Arc<Self>,
        tools: &[Arc<dyn Tool>],
        model: String,
        system_prompt: String,
        child_depth: usize,
        mission: &str,
        task: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
        role: AgentRole,
        add_spawn_tool: bool,
        task_id: Option<String>,
        goal_id: Option<String>,
        root_tools: Option<Vec<Arc<dyn Tool>>>,
    ) -> anyhow::Result<String> {
        let child_session = format!("sub-{}-{}", child_depth, Uuid::new_v4());

        info!(
            parent_depth = self.depth,
            child_depth,
            child_session = %child_session,
            mission,
            ?role,
            "Spawning sub-agent"
        );

        // Emit SubAgentSpawn event
        {
            let emitter =
                crate::events::EventEmitter::new(self.event_store.clone(), child_session.clone());
            let _ = emitter
                .emit(
                    EventType::SubAgentSpawn,
                    SubAgentSpawnData {
                        child_session_id: child_session.clone(),
                        mission: mission.to_string(),
                        task: task.chars().take(500).collect(),
                        depth: child_depth as u32,
                        parent_task_id: None,
                    },
                )
                .await;
        }

        let start = std::time::Instant::now();
        // Save task_id for post-completion knowledge extraction (Phase 4)
        let saved_task_id = task_id.clone();

        let result = if add_spawn_tool {
            let spawn_tool = Arc::new(crate::tools::spawn::SpawnAgentTool::new_deferred(
                self.max_response_chars,
                self.timeout_secs,
            ));

            let mut child_tools: Vec<Arc<dyn Tool>> = tools.to_vec();
            child_tools.push(spawn_tool.clone());

            // Derive child cancel token from parent
            let child_cancel = self.cancel_token.as_ref().map(|t| t.child_token());

            let child = Arc::new(Agent::with_depth(
                self.llm_runtime.clone(),
                self.state.clone(),
                self.event_store.clone(),
                child_tools,
                model,
                system_prompt,
                self.config_path.clone(),
                self.skills_dir.clone(),
                child_depth,
                self.max_depth,
                self.iteration_config.clone(),
                self.max_iterations,
                self.max_iterations_cap,
                self.max_response_chars,
                self.timeout_secs,
                self.max_facts,
                self.task_timeout,
                self.task_token_budget,
                self.llm_call_timeout,
                self.mcp_registry.clone(),
                self.verification_tracker.clone(),
                role,
                task_id,
                goal_id,
                child_cancel,
                self.goal_token_registry.clone(),
                self.hub.read().await.clone(),
                self.record_decision_points,
                self.context_window_config.clone(),
                self.policy_config.clone(),
                root_tools,
            ));

            // Close the loop: give the spawn tool a weak ref to the child.
            spawn_tool.set_agent(Arc::downgrade(&child));

            child
                .handle_message(
                    &child_session,
                    task,
                    status_tx,
                    user_role,
                    channel_ctx,
                    None,
                )
                .await
        } else {
            // Derive child cancel token from parent
            let child_cancel = self.cancel_token.as_ref().map(|t| t.child_token());

            let child = Arc::new(Agent::with_depth(
                self.llm_runtime.clone(),
                self.state.clone(),
                self.event_store.clone(),
                tools.to_vec(),
                model,
                system_prompt,
                self.config_path.clone(),
                self.skills_dir.clone(),
                child_depth,
                self.max_depth,
                self.iteration_config.clone(),
                self.max_iterations,
                self.max_iterations_cap,
                self.max_response_chars,
                self.timeout_secs,
                self.max_facts,
                self.task_timeout,
                self.task_token_budget,
                self.llm_call_timeout,
                self.mcp_registry.clone(),
                self.verification_tracker.clone(),
                role,
                task_id,
                goal_id,
                child_cancel,
                self.goal_token_registry.clone(),
                self.hub.read().await.clone(),
                self.record_decision_points,
                self.context_window_config.clone(),
                self.policy_config.clone(),
                root_tools,
            ));

            child
                .handle_message(
                    &child_session,
                    task,
                    status_tx,
                    user_role,
                    channel_ctx,
                    None,
                )
                .await
        };

        let duration = start.elapsed();

        // Emit SubAgentComplete event
        {
            let emitter =
                crate::events::EventEmitter::new(self.event_store.clone(), child_session.clone());
            let (success, summary) = match &result {
                Ok(response) => (true, response.chars().take(200).collect()),
                Err(e) => (false, format!("{}", e)),
            };
            let _ = emitter
                .emit(
                    EventType::SubAgentComplete,
                    SubAgentCompleteData {
                        child_session_id: child_session,
                        success,
                        result_summary: summary,
                        duration_secs: duration.as_secs(),
                        parent_task_id: None,
                    },
                )
                .await;
        }

        // Spawn background knowledge extraction for completed executor tasks.
        if let Some(ref task_id) = saved_task_id {
            if result.is_ok() {
                if let Ok(Some(completed_task)) = self.state.get_task(task_id).await {
                    if completed_task.status == "completed" {
                        let state = self.state.clone();
                        let provider = self.llm_runtime.provider();
                        let model = self.fallback_model.read().await.clone();
                        let tid = task_id.clone();
                        tokio::spawn(async move {
                            if let Err(e) = crate::memory::task_learning::extract_task_knowledge(
                                state,
                                provider,
                                model,
                                completed_task,
                            )
                            .await
                            {
                                warn!(
                                    task_id = %tid,
                                    error = %e,
                                    "Task knowledge extraction failed"
                                );
                            }
                        });
                    }
                }
            }
        }

        result
    }

    /// Spawn a task lead for a goal. Called from handle_message (&self context).
    ///
    /// This is a simplified version of spawn_child that doesn't require &Arc<Self>,
    /// since handle_message takes &self. The task lead gets management + universal tools
    /// plus ManageGoalTasksTool and SpawnAgentTool (for spawning executors).
    pub(super) fn spawn_task_lead(
        &self,
        goal_id: &str,
        goal_description: &str,
        user_text: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<String>> + Send + '_>>
    {
        // Box::pin to break async recursion (handle_message -> spawn_task_lead -> handle_message)
        let goal_id = goal_id.to_string();
        let goal_description = goal_description.to_string();
        let user_text = user_text.to_string();
        Box::pin(async move {
            let goal_id = &goal_id;
            let goal_description = &goal_description;
            let user_text = &user_text;
            if self.depth >= self.max_depth {
                anyhow::bail!(
                    "Cannot spawn task lead: max recursion depth ({}) reached",
                    self.max_depth
                );
            }

            let child_depth = self.depth + 1;
            let model = self.model.read().await.clone();

            // Task leads get Management + Universal tools from parent
            let mut tl_tools: Vec<Arc<dyn Tool>> = self
                .tools
                .iter()
                .filter(|t| t.name() != "spawn_agent")
                .filter(|t| matches!(t.tool_role(), ToolRole::Management | ToolRole::Universal))
                .cloned()
                .collect();

            // Add ManageGoalTasksTool scoped to this goal
            tl_tools.push(Arc::new(crate::tools::ManageGoalTasksTool::new(
                goal_id.to_string(),
                self.state.clone(),
            )));

            // Read goal context for feed-forward (Phase 4)
            let goal_context = self
                .state
                .get_goal(goal_id)
                .await
                .ok()
                .flatten()
                .and_then(|g| g.context);

            let system_prompt = Self::build_task_lead_prompt(
                goal_id,
                user_text,
                goal_context.as_deref(),
                child_depth,
                self.max_depth,
            );

            let task_text = format!(
                "Plan and execute this goal by creating tasks and delegating to executors:\n\n{}",
                user_text
            );
            let mission = format!(
                "Task Lead for goal: {}",
                &goal_description[..goal_description.len().min(100)]
            );
            let child_session = format!("sub-{}-{}", child_depth, Uuid::new_v4());

            info!(
                parent_depth = self.depth,
                child_depth,
                child_session = %child_session,
                goal_id,
                "Spawning task lead"
            );

            // Emit SubAgentSpawn event
            {
                let emitter = crate::events::EventEmitter::new(
                    self.event_store.clone(),
                    child_session.clone(),
                );
                let _ = emitter
                    .emit(
                        EventType::SubAgentSpawn,
                        SubAgentSpawnData {
                            child_session_id: child_session.clone(),
                            mission: mission.clone(),
                            task: task_text.chars().take(500).collect(),
                            depth: child_depth as u32,
                            parent_task_id: None,
                        },
                    )
                    .await;
            }

            let start = std::time::Instant::now();

            // Task lead can spawn executors, so give it a SpawnAgentTool
            let spawn_tool = Arc::new(crate::tools::spawn::SpawnAgentTool::new_deferred(
                self.max_response_chars,
                self.timeout_secs,
            ));
            tl_tools.push(spawn_tool.clone());

            // Get a child cancellation token from the goal's token
            let child_cancel_token = if let Some(ref registry) = self.goal_token_registry {
                registry.child_token(goal_id).await
            } else {
                None
            };

            // Collect root tools (full unfiltered set) for Executor inheritance.
            // Use parent's root_tools if available, otherwise parent's full tool set.
            let root_tools_for_tl: Vec<Arc<dyn Tool>> = self
                .root_tools
                .as_ref()
                .unwrap_or(&self.tools)
                .iter()
                .filter(|t| t.name() != "spawn_agent")
                .cloned()
                .collect();

            let child = Arc::new(Agent::with_depth(
                self.llm_runtime.clone(),
                self.state.clone(),
                self.event_store.clone(),
                tl_tools,
                model,
                system_prompt,
                self.config_path.clone(),
                self.skills_dir.clone(),
                child_depth,
                self.max_depth,
                self.iteration_config.clone(),
                self.max_iterations,
                self.max_iterations_cap,
                self.max_response_chars,
                self.timeout_secs,
                self.max_facts,
                self.task_timeout,
                self.task_token_budget,
                self.llm_call_timeout,
                self.mcp_registry.clone(),
                self.verification_tracker.clone(),
                AgentRole::TaskLead,
                None,                             // task_id (task leads aren't executors)
                Some(goal_id.to_string()),        // goal_id (context injection for child)
                child_cancel_token,               // cancel_token (derived from goal token)
                self.goal_token_registry.clone(), // goal_token_registry
                self.hub.read().await.clone(),    // hub
                self.record_decision_points,
                self.context_window_config.clone(),
                self.policy_config.clone(),
                Some(root_tools_for_tl), // root_tools for Executor inheritance
            ));

            spawn_tool.set_agent(Arc::downgrade(&child));

            let result = child
                .handle_message(
                    &child_session,
                    &task_text,
                    status_tx,
                    user_role,
                    channel_ctx,
                    None,
                )
                .await;

            let duration = start.elapsed();

            // Emit SubAgentComplete event
            {
                let emitter = crate::events::EventEmitter::new(
                    self.event_store.clone(),
                    child_session.clone(),
                );
                let (success, summary) = match &result {
                    Ok(response) => (true, response.chars().take(200).collect()),
                    Err(e) => (false, format!("{}", e)),
                };
                let _ = emitter
                    .emit(
                        EventType::SubAgentComplete,
                        SubAgentCompleteData {
                            child_session_id: child_session,
                            success,
                            result_summary: summary,
                            duration_secs: duration.as_secs(),
                            parent_task_id: None,
                        },
                    )
                    .await;
            }

            result
        }) // end Box::pin(async move { ... })
    }

    /// Build system prompt for a Task Lead agent.
    fn build_task_lead_prompt(
        goal_id: &str,
        goal_description: &str,
        goal_context: Option<&str>,
        depth: usize,
        max_depth: usize,
    ) -> String {
        let mut prompt = format!(
            "You are a Task Lead managing goal: {goal_id}\n\
             Goal: {goal_description}\n\n\
             You are a sub-agent (depth {depth}/{max_depth}).\n\
             Your job is to plan and delegate work. You MUST NOT execute tasks yourself.\n\n\
             ## Workflow\n\
             1. Analyze the goal and break it into concrete tasks using manage_goal_tasks(create_task)\n\
                - Start with 2-5 tasks for the NEXT PHASE (not the entire project)\n\
                - After those tasks complete, reassess and create more tasks if the goal isn't done\n\
                - Set `depends_on` (array of task IDs) for tasks that require prior tasks to complete\n\
                - Set `parallel_group` for tasks that belong to the same logical phase\n\
                - Set `idempotent: true` for tasks safe to retry on failure\n\
                - Set `task_order` for display ordering\n\
             2. Before spawning an executor, claim the task: manage_goal_tasks(claim_task, task_id=...)\n\
                - This verifies dependencies are met and atomically reserves the task\n\
                - If claiming fails due to unmet dependencies, work on other available tasks first\n\
             3. Spawn an executor: spawn_agent(mission=..., task=..., task_id=<the task ID>)\n\
                - Always pass the task_id so executor activity is tracked\n\
             4. After each executor returns, update: manage_goal_tasks(update_task, task_id, status, result)\n\
             5. If a task fails and is idempotent: manage_goal_tasks(retry_task, task_id) then re-spawn\n\
                - If not idempotent or max retries exceeded: create alternative task or fail the goal\n\
             6. When all tasks complete: manage_goal_tasks(complete_goal, summary)\n\n\
             ## Rules\n\
             - Keep each planning step small: 2-5 tasks at a time, then iterate\n\
             - Spawn executors one at a time (sequential execution)\n\
             - Each executor gets a single, focused task\n\
             - Always check list_tasks before spawning the next executor\n\
             - If an executor reports a blocker, resolve it or adjust the plan"
        );

        if let Some(ctx) = goal_context {
            prompt.push_str(&format!(
                "\n\n## Prior Knowledge\n\
                 The following knowledge was gathered from previous tasks and may be relevant:\n{}",
                format_goal_context(ctx)
            ));
        }

        prompt
    }

    /// Build system prompt for an Executor agent.
    fn build_executor_prompt(task_description: &str, depth: usize, max_depth: usize) -> String {
        format!(
            "You are an Executor. Complete this single task and return your results.\n\n\
             You are a sub-agent (depth {depth}/{max_depth}).\n\n\
             Task: {task_description}\n\n\
             Rules:\n\
             - Focus ONLY on this task. Do not expand scope.\n\
             - EXECUTE the task immediately. Do NOT ask for permission or confirmation.\n\
             - Do NOT ask \"Shall I proceed?\" or \"Would you like me to...?\". Just do the work.\n\
             - There is no human in this loop — you are an autonomous executor.\n\
             - If you encounter ambiguity or a blocker you cannot resolve, use report_blocker immediately.\n\
             - Return the FULL content you produced — not a meta-description of what you did.\n\
             - If your task is research: return all findings, data points, and analysis in detail.\n\
             - If your task is to write a report: return the complete report text.\n\
             - If your task is to run a command: return the full output.\n\
             - NEVER return just \"I researched X\" or \"Generated a report about Y\". Return the actual content.\n\
             - Include specific outputs (file paths, data retrieved, commands run).\n\
             - If you create or write a file, include its FULL ABSOLUTE PATH in your result text.\n\
             - Do NOT spawn sub-agents."
        )
    }
}
