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
    use crate::events::TaskOutcome;
    use aidaemon_core::card::{ActivityKind, Step};

    fn rendered_checklist(done: usize, total: usize) -> String {
        let steps = (0..total).map(|i| format!("synthetic step {i}")).collect();
        let mut plan = crate::plans::TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            steps,
            "track_requirements",
        );
        for step in plan.steps.iter_mut().take(done) {
            step.status = crate::plans::StepStatus::Completed;
        }
        if done == total {
            plan.status = crate::plans::PlanStatus::Completed;
        } else {
            plan.current_step = done;
        }
        plan.render_compact_checklist()
    }

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

    /// Two parallel children each declare a checklist. Neither reaches the
    /// parent's card (which would replace the parent's plan); each step
    /// becomes its own child's record with a typed kind, a step handed to
    /// the background included, and a tool ending is not a step.
    #[tokio::test]
    async fn a_childs_checklist_never_replaces_the_parents_plan() {
        let (parent_tx, mut parent_rx) = mpsc::channel(16);
        let wordpress = child_status_sender(
            Some(parent_tx.clone()),
            "Build the WordPress demo site for Acme Corp",
            false,
            None,
        )
        .unwrap();
        let drupal = child_status_sender(
            Some(parent_tx),
            "Build the Drupal demo site for Acme Corp",
            false,
            None,
        )
        .unwrap();
        wordpress
            .send(StatusUpdate::Checklist(aidaemon_core::card::PlanCard {
                text: rendered_checklist(5, 5),
                steps: 5,
            }))
            .await
            .unwrap();
        drupal
            .send(StatusUpdate::Checklist(aidaemon_core::card::PlanCard {
                text: rendered_checklist(1, 4),
                steps: 4,
            }))
            .await
            .unwrap();
        drupal
            .send(StatusUpdate::ToolStart {
                name: "running the site installer".into(),
                summary: String::new(),
                kind: ActivityKind::Run,
            })
            .await
            .unwrap();
        drupal
            .send(StatusUpdate::ToolComplete {
                name: "running the site installer".into(),
                summary: "done".into(),
            })
            .await
            .unwrap();
        drupal
            .send(StatusUpdate::StepHandedOff {
                kind: ActivityKind::Run,
            })
            .await
            .unwrap();
        drop((wordpress, drupal));

        let mut lines = Vec::new();
        while let Some(update) = parent_rx.recv().await {
            match update {
                StatusUpdate::Checklist(_) => panic!("child checklist reached the parent card"),
                StatusUpdate::Helper(view) => {
                    assert!(!view.background);
                    lines.push(format!(
                        "{}: {}",
                        view.label,
                        view.step.map(|step| step.words()).unwrap_or_default()
                    ));
                }
                other => panic!("unexpected update {other:?}"),
            }
        }
        assert_eq!(
            lines,
            vec![
                "Build the Drupal demo site for Acme Corp: Running a command".to_string(),
                "Build the Drupal demo site for Acme Corp: Running a command · goes on in the \
                 background"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn running_siblings_with_one_title_are_told_apart_by_their_task() {
        let registry = Arc::new(RunningChildren::default());
        let mission = "Run one shell command and report its exact output.";
        let _alpha = registry.enter(
            "chat-1",
            mission,
            "Run this exact shell command: sleep 120 && echo alpha-test . Report the output.",
            "child-alpha",
            None,
        );
        let _beta = registry.enter(
            "chat-1",
            mission,
            "Run this exact shell command: sleep 120 && echo beta-test . Report the output.",
            "child-beta",
            None,
        );
        let _other_chat = registry.enter("chat-2", mission, "Run: date", "child-other", None);
        let labels: Vec<String> = registry
            .for_session("chat-1")
            .into_iter()
            .map(|child| child.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                format!("{mission} (…echo alpha-test . Report the…)"),
                format!("{mission} (…echo beta-test . Report the…)"),
            ]
        );
        let alone = registry.for_session("chat-2");
        assert_eq!(alone[0].label, mission);
        drop(_beta);
        assert_eq!(registry.for_session("chat-1")[0].label, mission);
    }

    /// A child is listed under the chat turn that started it, and a stop
    /// from the chat reaches it by its key.
    #[test]
    fn a_listed_child_carries_its_request_and_can_be_stopped() {
        let registry = Arc::new(RunningChildren::default());
        let turn = registry.begin_request("chat-1", "task-a");
        let child = registry.enter("chat-1", "Counting agent A", "count", "child-a", None);
        drop(turn);
        let later = registry.enter("chat-1", "Counting agent B", "count", "child-b", None);
        let listed = registry.listed("chat-1");
        assert_eq!(listed[0].request.as_deref(), Some("task-a"));
        assert_eq!(listed[1].request, None);
        assert_eq!(registry.current_request("chat-1"), None);
        let stop = child.stopped();
        assert_eq!(registry.stop(&listed[0].key).as_deref(), Some("child-a"));
        assert!(stop.is_cancelled());
        assert!(registry.is_listed(&listed[0].key));
        drop(child);
        assert!(!registry.is_listed(&listed[0].key));
        assert!(registry.stop(&listed[0].key).is_none());
        drop(later);
    }

    #[test]
    fn a_child_status_label_fits_a_phone() {
        assert_eq!(child_status_label("  \n"), "Sub-agent");
        let long = "x".repeat(200);
        let label = child_status_label(&long);
        assert_eq!(label.chars().count(), 60);
        assert!(label.ends_with('…'));
    }

    /// A child that saw untrusted content but could not record it, or
    /// whose record cannot be read, returns an untrusted result.
    async fn synthetic_event_store() -> crate::events::EventStore {
        use crate::events::EventStoreForTests;
        let db_file = tempfile::NamedTempFile::new().expect("temp db file");
        let path = db_file.into_temp_path().keep().expect("keep temp db path");
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", path.display()))
            .await
            .expect("connect sqlite");
        crate::events::EventStore::new(pool)
            .await
            .expect("init event store")
    }

    async fn record_tool_call(
        store: &crate::events::EventStore,
        session: &str,
        id: &str,
        name: &str,
        finished: bool,
    ) {
        store
            .append(crate::events::Event::new(
                session,
                EventType::ToolCall,
                json!({ "tool_call_id": id, "name": name, "arguments": {} }),
            ))
            .await
            .expect("append tool call");
        if finished {
            store
                .append(crate::events::Event::new(
                    session,
                    EventType::ToolResult,
                    json!({
                        "tool_call_id": id,
                        "name": name,
                        "result": "ok",
                        "success": true,
                        "duration_ms": 1,
                    }),
                ))
                .await
                .expect("append tool result");
        }
    }

    /// An early change stays in the report however many read-only calls
    /// follow it.
    #[tokio::test]
    async fn a_stopped_childs_report_covers_its_whole_run() {
        let store = synthetic_event_store().await;
        let session = "specialist:executor:synthetic-1";
        let read_only: std::collections::HashSet<String> =
            ["read_file".to_string()].into_iter().collect();
        record_tool_call(&store, session, "call-write", "write_file", true).await;
        for n in 0..300 {
            record_tool_call(
                &store,
                session,
                &format!("call-read-{n}"),
                "read_file",
                true,
            )
            .await;
        }
        let report = child_work_report(&store, session, &read_only).await;
        assert!(report.may_have_changed_something());
        assert_eq!(report.changed.len(), 1);
        assert!(!report.unknown);
    }

    /// A report that cannot show it saw the run from its first record does
    /// not claim the child only read.
    #[tokio::test]
    async fn a_report_that_cannot_see_the_whole_run_is_unknown() {
        let store = synthetic_event_store().await;
        let session = "specialist:executor:synthetic-2";
        let read_only: std::collections::HashSet<String> =
            ["read_file".to_string()].into_iter().collect();
        for n in 0..6 {
            record_tool_call(
                &store,
                session,
                &format!("call-read-{n}"),
                "read_file",
                true,
            )
            .await;
        }
        let report = child_work_report_within(&store, session, &read_only, 10).await;
        assert!(report.unknown);
        assert!(report.may_have_changed_something());
        let report = child_work_report_within(&store, session, &read_only, 13).await;
        assert!(!report.unknown);
        assert!(!report.may_have_changed_something());
    }

    /// A child's end record is found however many records follow it.
    #[tokio::test]
    async fn a_childs_end_record_is_found_behind_later_records() {
        let store = synthetic_event_store().await;
        let session = "specialist:executor:synthetic-3";
        store
            .append(crate::events::Event::new(
                session,
                EventType::TaskEnd,
                json!({
                    "task_id": "task-synthetic-3",
                    "status": "completed",
                    "outcome": "succeeded",
                }),
            ))
            .await
            .expect("append task end");
        for n in 0..40 {
            record_tool_call(
                &store,
                session,
                &format!("call-late-{n}"),
                "read_file",
                false,
            )
            .await;
        }
        let record = latest_child_task_end(&store, session).await;
        assert_eq!(resolve_child_outcome(true, &record), TaskOutcome::Succeeded);
    }

    #[test]
    fn a_child_with_a_refused_write_it_never_made_is_incomplete() {
        assert_eq!(
            settle_coordinated_outcome(TaskOutcome::Succeeded, true),
            TaskOutcome::Partial
        );
        assert_eq!(
            settle_coordinated_outcome(TaskOutcome::Succeeded, false),
            TaskOutcome::Succeeded
        );
        assert_eq!(
            settle_coordinated_outcome(TaskOutcome::Failed, true),
            TaskOutcome::Failed
        );
    }

    #[test]
    fn a_reply_with_no_readable_end_record_is_not_a_success() {
        let unreadable: anyhow::Result<Option<TaskEndData>> =
            Err(anyhow::anyhow!("database is locked"));
        assert_eq!(
            resolve_child_outcome(true, &unreadable),
            TaskOutcome::Partial
        );
        assert_eq!(resolve_child_outcome(true, &Ok(None)), TaskOutcome::Partial);
        assert_eq!(resolve_child_outcome(false, &Ok(None)), TaskOutcome::Failed);
    }

    /// The child's prompt says what `child_can_spawn` decided: a specialist
    /// its caller made a coordinator is not told it cannot delegate.
    #[test]
    fn the_child_prompt_follows_the_spawn_decision() {
        let coordinator = plain_child_prompt("P", 1, 3, "Coordinate", Some("profile"), true);
        assert!(!coordinator.contains("CANNOT spawn"));
        let leaf = plain_child_prompt("P", 1, 3, "Count", Some("profile"), false);
        assert!(leaf.contains("CANNOT spawn"));
        let deepest = plain_child_prompt("P", 3, 3, "Count", None, false);
        assert!(deepest.contains("maximum sub-agent depth"));
    }

    #[test]
    fn a_named_specialist_is_a_leaf_and_a_plain_child_is_not() {
        assert!(child_can_spawn(1, 3, AgentRole::Orchestrator, false, false));
        assert!(!child_can_spawn(1, 3, AgentRole::Orchestrator, true, false));
        assert!(!child_can_spawn(
            3,
            3,
            AgentRole::Orchestrator,
            false,
            false
        ));
        assert!(!child_can_spawn(1, 3, AgentRole::Executor, false, false));
        // A caller that grants `spawn_agent` asked for a coordinator, within
        // the depth limit and never for an executor.
        assert!(child_can_spawn(1, 3, AgentRole::Orchestrator, true, true));
        assert!(!child_can_spawn(3, 3, AgentRole::Orchestrator, true, true));
        assert!(!child_can_spawn(1, 3, AgentRole::Executor, true, true));
    }

    #[test]
    fn task_lead_cannot_be_named_on_a_plain_spawn() {
        assert_eq!(named_specialist(Some("task_lead")), None);
        assert_eq!(named_specialist(Some("not-a-kind")), None);
        assert_eq!(named_specialist(None), None);
        assert_eq!(
            named_specialist(Some("review")),
            Some(crate::traits::SpecialistKind::Review)
        );
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

    #[test]
    fn an_unreadable_taint_record_counts_as_tainted() {
        use crate::memory::task_learning::RunTaint;
        assert_eq!(
            aidaemon_runtime::engine::turn_engine::spawn::run_taint(
                false,
                &Err(anyhow::anyhow!("corrupt"))
            ),
            RunTaint::Tainted
        );
        assert_eq!(
            aidaemon_runtime::engine::turn_engine::spawn::run_taint(
                false,
                &Ok(vec!["turn-1".to_string()])
            ),
            RunTaint::Tainted
        );
        assert_eq!(
            aidaemon_runtime::engine::turn_engine::spawn::run_taint(false, &Ok(Vec::new())),
            RunTaint::Clean
        );
    }

    #[test]
    fn a_run_flagged_untrusted_counts_as_tainted_even_with_an_empty_record() {
        use crate::memory::task_learning::RunTaint;
        assert_eq!(
            aidaemon_runtime::engine::turn_engine::spawn::run_taint(true, &Ok(Vec::new())),
            RunTaint::Tainted
        );
        assert_eq!(
            aidaemon_runtime::engine::turn_engine::spawn::run_taint(false, &Ok(Vec::new())),
            RunTaint::Clean
        );
    }

    #[test]
    fn a_childs_unrecorded_or_unreadable_taint_fails_closed() {
        use aidaemon_core::SourceTrust::{Trusted, Untrusted};
        assert_eq!(super::child_result_trust(false, &Ok(Vec::new())), Trusted);
        assert_eq!(super::child_result_trust(true, &Ok(Vec::new())), Untrusted);
        assert_eq!(
            super::child_result_trust(false, &Ok(vec!["turn-a".to_string()])),
            Untrusted
        );
        assert_eq!(
            super::child_result_trust(false, &Err(anyhow::anyhow!("unreadable"))),
            Untrusted
        );
    }

    use super::is_sqlite_busy;

    #[test]
    fn sqlite_busy_retry_classifier_is_narrow() {
        assert!(is_sqlite_busy(&anyhow::anyhow!(
            "error returned from database: (code: 5) database is locked"
        )));
        assert!(is_sqlite_busy(&anyhow::anyhow!("SQLITE_BUSY")));
        for code in [261, 517, 773] {
            assert!(is_sqlite_busy(&anyhow::anyhow!(
                "error returned from database: (code: {code}) database is locked"
            )));
        }
        assert!(!is_sqlite_busy(&anyhow::anyhow!(
            "error returned from database: (code: 516) unrelated failure"
        )));
        assert!(!is_sqlite_busy(&anyhow::anyhow!("executor lease was lost")));
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
