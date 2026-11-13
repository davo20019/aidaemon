#[cfg(test)]
use super::*;
#[cfg(test)]
pub(crate) use aidaemon_runtime::spawn::*;

/// Tests' shorthands for
/// [`TurnEngine::spawn_child`](aidaemon_runtime::engine::turn_engine::TurnEngine::spawn_child).
#[cfg(test)]
pub(crate) trait SpawnChildForTest {
    /// Spawn a child and return its reply.
    #[allow(clippy::too_many_arguments)]
    async fn spawn_child(
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
    ) -> anyhow::Result<String>;

    /// Spawn a child and return how it ended.
    #[allow(clippy::too_many_arguments)]
    async fn spawn_child_with_outcome(
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
    ) -> anyhow::Result<SpawnChildResult>;
}

#[cfg(test)]
impl SpawnChildForTest for Agent {
    async fn spawn_child(
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

    async fn spawn_child_with_outcome(
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
        let counter =
            child_status_sender(Some(parent_tx), "Slow counter", &[], true, None).unwrap();
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
        harness
            .agent
            .set_test_approval_session_id(Some("telegram:synthetic-user-1".to_string()));
        assert_eq!(
            harness
                .agent
                .background_delivery_session("specialist:code:1"),
            "telegram:synthetic-user-1"
        );
    }

    #[tokio::test]
    async fn mandate_task_lead_receives_only_bounded_owner_guidance_from_goal_context() {
        use crate::testing::{MockProvider, setup_test_agent};
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
        assert!(
            spec.system_prompt
                .contains("Prefer thoughtful replies over original posts")
        );
        assert!(!spec.system_prompt.contains("PRIVATE CONTROLLER FACT"));
        assert!(!spec.system_prompt.contains("PRIVATE CONTROLLER HISTORY"));
    }
}
