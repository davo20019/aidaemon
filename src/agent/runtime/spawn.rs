use super::*;
pub(crate) use aidaemon_runtime::spawn::*;

// impl-Agent justification: the host side of making a child (its own tools,
// MCP registry and agent struct); deciding and running the child is the
// engine's (`aidaemon_runtime::engine::turn_engine::spawn`).
impl Agent {
    /// Build a child of this agent from `spec` (the host side of
    /// [`aidaemon_runtime::host::TurnHost::create_child`]).
    pub(in crate::agent) async fn create_child_agent(&self, spec: ChildSpec) -> Arc<Agent> {
        let ChildSpec {
            mut tools,
            model,
            system_prompt,
            depth: child_depth,
            role,
            task_id,
            goal_id,
            mandate_execution,
            cancel_token,
            root_tools,
            add_spawn_tool,
            inherited_project_scope,
            approval_session_id,
            max_iterations_override,
            timeout_secs_override,
            inherited_authority,
            child_progress,
        } = spec;
        let hub = match tokio::time::timeout(Duration::from_secs(2), self.hub.read()).await {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!("Timed out acquiring hub lock while spawning child agent");
                None
            }
        };

        let spawn_tool = if add_spawn_tool {
            Some(Arc::new(crate::tools::spawn::SpawnAgentTool::for_child(
                self.limits.max_response_chars,
                self.limits.timeout_secs,
                self.state.clone(),
                hub.clone(),
            )))
        } else {
            None
        };

        if let Some(ref spawn_tool) = spawn_tool {
            tools.push(spawn_tool.clone());
        }
        // A plain child in a tree can hand the coordinator a decision.
        if inherited_authority
            .spawn_family
            .as_ref()
            .is_some_and(|family| family.coordinates())
        {
            tools.push(Arc::new(crate::tools::EscalateToParentTool));
        }

        let effective_max_iterations =
            max_iterations_override.unwrap_or(self.limits.max_iterations);
        let effective_timeout_secs = timeout_secs_override.unwrap_or(self.limits.timeout_secs);
        let child_mcp_registry = match mandate_execution.as_ref() {
            Some(fence) => self
                .mcp_registry
                .as_ref()
                .map(|registry| registry.scoped_to_mandate(&fence.authority)),
            None => self.mcp_registry.clone(),
        };
        let child_specialists = if mandate_execution.is_some() {
            Arc::new(crate::agent::specialists::SpecialistRegistry::load(None))
        } else {
            self.specialists.clone()
        };
        #[allow(unused_mut)]
        let mut child = Agent::with_depth(
            self.llm_runtime.clone(),
            self.side_calls.clone(),
            self.state.clone(),
            self.memory_store.clone(),
            self.event_store.clone(),
            tools,
            model,
            system_prompt,
            self.config_path.clone(),
            self.daemon_paths.clone(),
            self.execution_backend.clone(),
            self.skills_dir.clone(),
            child_depth,
            self.limits.max_depth,
            self.limits.iteration_config.clone(),
            effective_max_iterations,
            self.limits.max_iterations_cap,
            self.limits.max_response_chars,
            effective_timeout_secs,
            self.limits.max_facts,
            self.limits.task_token_budget,
            self.limits.llm_call_timeout,
            child_mcp_registry,
            role,
            task_id,
            goal_id,
            mandate_execution,
            cancel_token,
            self.goal_token_registry.clone(),
            hub,
            self.schedule_approved_sessions.clone(),
            self.billing_failed_models.clone(),
            self.runtime.record_decision_points,
            self.runtime.context_window_config.clone(),
            self.runtime.policy_config.clone(),
            self.path_aliases.clone(),
            self.inbox_dir.clone(),
            inherited_project_scope,
            approval_session_id,
            root_tools,
            child_specialists,
            self.runtime.vision_config.clone(),
            self.runtime.audio_config.clone(),
            self.runtime.stt_config.clone(),
            self.runtime.harness_eval_config.clone(),
        );
        #[cfg(test)]
        crate::agent::authority::CREATED_CHILD_INHERITANCE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(inherited_authority.clone());
        child.inherited_authority = Some(inherited_authority);
        child.child_progress = child_progress;
        child.running_children = self.running_children.clone();
        child.sdk_engine = self.sdk_engine.clone();
        child.run_store = self.run_store.clone();
        child.owner_approvals = self.owner_approvals.clone();
        child.memory = self.memory.clone();
        child.transcript_writes = self.transcript_writes.clone();
        let child = Arc::new(child);

        if let Some(spawn_tool) = spawn_tool {
            let runtime: Arc<dyn crate::runtime_ports::ChildAgentRuntime> = child.clone();
            spawn_tool.set_agent(Arc::downgrade(&runtime));
        }

        child
    }

    /// Spawn a child and return its reply (tests' shorthand for
    /// [`TurnEngine::spawn_child`](aidaemon_runtime::engine::turn_engine::TurnEngine::spawn_child)).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub async fn spawn_child(
        &self,
        mission: &str,
        task: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
        child_role: Option<AgentRole>,
        goal_id: Option<&str>,
        task_id: Option<&str>,
        inherited_project_scope: Option<&str>,
        arg_specialist: Option<&str>,
    ) -> anyhow::Result<String> {
        self.spawn_child_with_outcome(
            mission,
            task,
            status_tx,
            channel_ctx,
            user_role,
            child_role,
            goal_id,
            task_id,
            inherited_project_scope,
            arg_specialist,
            None,
            None,
            None,
            None,
        )
        .await
        .map(|result| result.response)
    }

    /// Spawn a child and return how it ended (tests' shorthand for
    /// [`TurnEngine::spawn_child`](aidaemon_runtime::engine::turn_engine::TurnEngine::spawn_child)).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn spawn_child_with_outcome(
        &self,
        mission: &str,
        task: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
        child_role: Option<AgentRole>,
        goal_id: Option<&str>,
        task_id: Option<&str>,
        inherited_project_scope: Option<&str>,
        arg_specialist: Option<&str>,
        approval_session_id: Option<&str>,
        parent_authority: Option<crate::agent::authority::InheritedAuthority>,
        time_limit: Option<ChildTimeLimit>,
        model_tier: Option<crate::router::Tier>,
    ) -> anyhow::Result<SpawnChildResult> {
        let mut request =
            crate::runtime_ports::ChildAgentRequest::new(mission, task, channel_ctx, user_role);
        request.status_tx = status_tx;
        request.child_role = child_role;
        request.goal_id = goal_id.map(str::to_string);
        request.task_id = task_id.map(str::to_string);
        request.project_scope = inherited_project_scope.map(str::to_string);
        request.specialist = arg_specialist.map(str::to_string);
        request.approval_session_id = approval_session_id.map(str::to_string);
        request.parent_authority = parent_authority;
        request.time_limit = time_limit;
        request.model_tier = model_tier;
        self.engine().spawn_child(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::card::Step;

    /// Live 2026-11-13: "the whole count ran as a single command" and the
    /// card showed nothing for 60 s. The line a helper's running command
    /// printed last (secrets redacted) is that helper's progress line.
    #[tokio::test]
    async fn a_helpers_command_output_is_its_progress_line() {
        let printed = b"1\n2\nTOKEN=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789\r42\n\n";
        let line = crate::tools::terminal::latest_output_line(printed).expect("a line");
        assert_eq!(line, "42");
        let secret = crate::tools::terminal::latest_output_line(
            b"export API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789\n",
        )
        .expect("a line");
        assert!(!secret.contains("abcdefghijklmnop"), "{secret}");
        assert_eq!(crate::tools::terminal::latest_output_line(b"\n \n"), None);

        let (parent_tx, mut parent_rx) = mpsc::channel(4);
        let counter = child_status_sender(Some(parent_tx), "Slow counter", true, None).unwrap();
        counter
            .send(StatusUpdate::CommandOutput {
                name: "terminal".into(),
                line,
            })
            .await
            .unwrap();
        drop(counter);
        match parent_rx.recv().await {
            Some(StatusUpdate::Helper(view)) => {
                assert_eq!(view.label, "Slow counter");
                assert_eq!(view.step, Some(Step::Output("42".into())));
                assert_eq!(
                    view.step.unwrap().words(),
                    "Running a command · last output: 42",
                    "the owner reads plain words"
                );
            }
            other => panic!("expected the helper's record, got {other:?}"),
        }
    }

    #[test]
    fn review_specialist_keeps_only_reading_tools() {
        let registry = crate::agent::specialists::SpecialistRegistry::load(None);
        let declared = registry
            .get(crate::traits::SpecialistKind::Review)
            .tools
            .clone()
            .expect("review declares its tools");
        let mut tools: Vec<Arc<dyn Tool>> = ["read_file", "write_file", "edit_file", "terminal"]
            .into_iter()
            .map(|name| {
                Arc::new(crate::testing::MockTool::new(name, "synthetic", "ok")) as Arc<dyn Tool>
            })
            .collect();
        aidaemon_runtime::specialists::compose::apply_specialist_tool_allowlist(
            crate::traits::SpecialistKind::Review,
            &declared,
            &mut tools,
        );
        let names: Vec<&str> = tools.iter().map(|tool| tool.name()).collect();
        assert_eq!(names, vec!["read_file"]);
    }

    /// A background result from a nested child goes to the chat the work
    /// answers to, not to the child's own session.
    #[tokio::test]
    async fn nested_background_results_go_to_the_chat_the_work_answers_to() {
        use crate::runtime_ports::ChildAgentRuntime;
        let provider = crate::testing::MockProvider::new();
        let mut harness =
            crate::testing::setup_full_stack_test_agent_with_extra_tools(provider, Vec::new())
                .await
                .unwrap();
        assert_eq!(
            harness
                .agent
                .background_delivery_session("specialist:code:1"),
            "specialist:code:1",
            "a root agent answers to its own session"
        );
        harness.agent.approval_session_id = Some("telegram:synthetic-user-1".to_string());
        assert_eq!(
            harness
                .agent
                .background_delivery_session("specialist:code:1"),
            "telegram:synthetic-user-1"
        );
    }

    #[tokio::test]
    async fn mandate_task_lead_receives_only_bounded_owner_guidance_from_goal_context() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::store_prelude::*;
        use crate::traits::{Goal, Mandate, MandateAuthority};

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("setup test harness");
        let mut goal = Goal::new_continuous("Steward the public account", "owner", None, None);
        goal.context = Some(
            serde_json::json!({
                "relevant_facts": [{"value": "PRIVATE CONTROLLER FACT"}],
                "recent_messages": ["PRIVATE CONTROLLER HISTORY"],
                "owner_guidance": [{
                    "guidance": "Prefer thoughtful replies over original posts",
                    "recorded_at": "2026-08-02T00:00:00Z"
                }]
            })
            .to_string(),
        );
        let mandate = Mandate::new(
            &goal.id,
            None,
            "Steward the public account",
            "owner",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        harness
            .state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        let spec = harness
            .agent
            .engine()
            .build_task_lead_spec(&[], &goal.id, &goal.description, 1, false, None)
            .await;
        assert!(spec
            .system_prompt
            .contains("Prefer thoughtful replies over original posts"));
        assert!(!spec.system_prompt.contains("PRIVATE CONTROLLER FACT"));
        assert!(!spec.system_prompt.contains("PRIVATE CONTROLLER HISTORY"));
    }
}
