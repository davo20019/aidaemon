// Agents in one request's tree sharing a workspace: every test seeds its
// conflict deterministically and judges typed state, not wording.

mod coordination_tests {
    use super::*;
    use crate::agent::authority::{InheritedAuthority, SpawnFamily, SpawnPermit};
    use crate::agent::coordination::file_state;
    use crate::testing::TestHarness;

    const CHILD_A: &str = "sub_agent_synthetic_a";

    fn inherited(family: &SpawnFamily) -> InheritedAuthority {
        InheritedAuthority {
            authority: aidaemon_core::Authority::full(),
            owner_level: Some(crate::config::OwnerLevel::Full),
            task_access: None,
            mode_session: None,
            untrusted_context: false,
            origin_request: Some("Update the synthetic notes.".to_string()),
            origin_trusted: true,
            trusted_roots: Vec::new(),
            declined: Vec::new(),
            shared_declines: None,
            spawn_family: Some(family.clone()),
            child_declaration: None,
            parent_run: None,
        }
    }

    /// A tree that has a child, and the permit keeping that child counted.
    fn tree() -> (SpawnFamily, SpawnPermit) {
        let family = SpawnFamily::new(Default::default());
        let permit = family.try_admit().unwrap();
        (family, permit)
    }

    async fn harness(responses: Vec<crate::traits::ProviderResponse>) -> TestHarness {
        let backend = crate::execution::active_execution_backend;
        let tools: Vec<Arc<dyn crate::traits::Tool>> = vec![
            Arc::new(crate::tools::ReadFileTool::new(backend())),
            Arc::new(crate::tools::WriteFileTool::new(backend())),
            Arc::new(crate::tools::EditFileTool::new(backend())),
        ];
        setup_test_agent_with_extra_tools_and_llm_timeout(
            MockProvider::with_responses(responses),
            tools,
            None,
        )
        .await
        .unwrap()
    }

    /// One turn of `session` in the tree `family`.
    async fn turn(harness: &mut TestHarness, family: &SpawnFamily, session: &str) -> String {
        harness
            .agent
            .set_test_inherited_authority(inherited(family));
        harness
            .agent
            .handle_message(
                session,
                "Work on the synthetic notes.",
                None,
                UserRole::Owner,
                ChannelContext::private("test"),
                None,
            )
            .await
            .unwrap()
    }

    fn read_call(path: &str) -> crate::traits::ProviderResponse {
        MockProvider::tool_call_response("read_file", &json!({"path": path}).to_string())
    }

    fn real(path: &std::path::Path) -> String {
        std::fs::canonicalize(path)
            .unwrap()
            .to_string_lossy()
            .into_owned()
    }

    #[tokio::test]
    async fn a_read_in_a_tree_becomes_the_readers_baseline() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let (family, _permit) = tree();
        let mut harness = harness(vec![
            read_call(file.to_str().unwrap()),
            MockProvider::text_response("Read it."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        let coordination = family.active_coordination().unwrap();
        assert_eq!(
            coordination.baseline(CHILD_A, &real(&file)),
            Some(file_state(Some(b"alpha\n")))
        );
    }

    /// A lone agent has no ledger to write to.
    #[tokio::test]
    async fn a_request_with_no_child_tracks_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let family = SpawnFamily::new(Default::default());
        let mut harness = harness(vec![
            read_call(file.to_str().unwrap()),
            MockProvider::text_response("Read it."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        assert!(family.active_coordination().is_none());
    }

    fn edit_call(path: &str, old: &str, new: &str) -> crate::traits::ProviderResponse {
        MockProvider::tool_call_response(
            "edit_file",
            &json!({"path": path, "old_text": old, "new_text": new}).to_string(),
        )
    }

    fn write_call(path: &str, content: &str) -> crate::traits::ProviderResponse {
        MockProvider::tool_call_response(
            "write_file",
            &json!({"path": path, "content": content}).to_string(),
        )
    }

    const CHILD_B: &str = "sub_agent_synthetic_b";

    fn stale_refusals(family: &SpawnFamily, member: &str) -> usize {
        use crate::agent::coordination::LogEntry;
        family
            .active_coordination()
            .unwrap()
            .log_entries()
            .iter()
            .filter(|entry| {
                matches!(entry, LogEntry::StaleWriteRefused { member: m, .. } if m == member)
            })
            .count()
    }

    /// A sibling changed another part of the file: the write is refused, the
    /// writer re-reads and retries, both changes survive, and the run still
    /// ends as a success.
    #[tokio::test]
    async fn a_non_overlapping_stale_edit_is_refused_then_lands_after_a_reread() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        let p = file.to_str().unwrap();
        std::fs::write(&file, "alpha\nbeta\ngamma\n").unwrap();
        let (family, _permit) = tree();
        let mut harness = harness(vec![
            // A reads.
            read_call(p),
            MockProvider::text_response("Read."),
            // B reads, changes the first line.
            read_call(p),
            edit_call(p, "alpha", "ALPHA"),
            MockProvider::text_response("B done."),
            // A edits from its stale view, is refused, re-reads, retries.
            edit_call(p, "gamma", "GAMMA"),
            read_call(p),
            edit_call(p, "gamma", "GAMMA"),
            MockProvider::text_response("A done."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "ALPHA\nbeta\nGAMMA\n"
        );
        assert_eq!(stale_refusals(&family, CHILD_A), 1);
        assert_eq!(
            latest_task_outcome(&harness, CHILD_A).await,
            crate::events::TaskOutcome::Succeeded,
            "a refused-then-retried write is an ordinary run"
        );
    }

    /// A write over a file a sibling changed never lands silently.
    #[tokio::test]
    async fn a_full_overwrite_of_a_siblings_change_does_not_land() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        let p = file.to_str().unwrap();
        std::fs::write(&file, "alpha\n").unwrap();
        let (family, _permit) = tree();
        let mut harness = harness(vec![
            read_call(p),
            MockProvider::text_response("Read."),
            write_call(p, "from b\n"),
            MockProvider::text_response("B done."),
            write_call(p, "from a\n"),
            MockProvider::text_response("A stopped."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from b\n");
    }

    /// A change made by nobody in the tree is caught too.
    #[tokio::test]
    async fn an_unattributed_change_stops_the_write() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        let p = file.to_str().unwrap();
        std::fs::write(&file, "alpha\n").unwrap();
        let (family, _permit) = tree();
        let mut harness = harness(vec![
            read_call(p),
            MockProvider::text_response("Read."),
            write_call(p, "from a\n"),
            MockProvider::text_response("A stopped."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        std::fs::write(&file, "from outside\n").unwrap();
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from outside\n");
        assert_eq!(stale_refusals(&family, CHILD_A), 1);
    }

    /// Before the tree has a child the same interleaving runs as it always
    /// has: the last writer wins.
    #[tokio::test]
    async fn a_lone_agents_writes_are_not_judged() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        let p = file.to_str().unwrap();
        std::fs::write(&file, "alpha\n").unwrap();
        let family = SpawnFamily::new(Default::default());
        let mut harness = harness(vec![
            read_call(p),
            MockProvider::text_response("Read."),
            write_call(p, "last\n"),
            MockProvider::text_response("Done."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        std::fs::write(&file, "from outside\n").unwrap();
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "last\n");
    }

    /// A shell whose command rewrites a file, as the real terminal might.
    struct SyntheticShell;

    #[async_trait::async_trait]
    impl crate::traits::Tool for SyntheticShell {
        fn name(&self) -> &str {
            "synthetic_shell"
        }
        fn description(&self) -> &str {
            "Runs a synthetic command"
        }
        fn schema(&self) -> serde_json::Value {
            json!({
                "name": "synthetic_shell",
                "description": "Runs a synthetic command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["command"],
                    "additionalProperties": false
                }
            })
        }
        async fn execute(
            &self,
            arguments: &str,
            _facts: &crate::traits::ToolCallContext,
        ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
            let args: serde_json::Value = serde_json::from_str(arguments)?;
            if let (Some(path), Some(content)) = (args["path"].as_str(), args["content"].as_str()) {
                std::fs::write(path, content)?;
            }
            Ok(crate::traits::ToolCallOutcome::from_output(
                "ran".to_string(),
            ))
        }
    }

    impl crate::traits::ToolSemantics for SyntheticShell {
        fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
            crate::traits::ToolDispatchFacts {
                shell_command_line: true,
                ..Default::default()
            }
        }
        fn capabilities(&self) -> crate::traits::ToolCapabilities {
            crate::traits::ToolCapabilities {
                read_only: true,
                idempotent: true,
                ..Default::default()
            }
        }
    }

    async fn harness_with_shell(responses: Vec<crate::traits::ProviderResponse>) -> TestHarness {
        let backend = crate::execution::active_execution_backend;
        let tools: Vec<Arc<dyn crate::traits::Tool>> = vec![
            Arc::new(crate::tools::ReadFileTool::new(backend())),
            Arc::new(crate::tools::WriteFileTool::new(backend())),
            Arc::new(crate::tools::EditFileTool::new(backend())),
            Arc::new(SyntheticShell),
        ];
        setup_test_agent_with_extra_tools_and_llm_timeout(
            MockProvider::with_responses(responses),
            tools,
            None,
        )
        .await
        .unwrap()
    }

    fn shell_call(path: &str, content: &str) -> crate::traits::ProviderResponse {
        MockProvider::tool_call_response(
            "synthetic_shell",
            &json!({"command": "rewrite", "path": path, "content": content}).to_string(),
        )
    }

    /// An agent's own shell change does not make its own next write stale.
    #[tokio::test]
    async fn a_members_own_shell_change_keeps_its_write_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        let p = file.to_str().unwrap();
        std::fs::write(&file, "alpha\n").unwrap();
        let (family, _permit) = tree();
        let mut harness = harness_with_shell(vec![
            read_call(p),
            shell_call(p, "from shell\n"),
            edit_call(p, "from shell", "edited"),
            MockProvider::text_response("Done."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "edited\n");
        assert_eq!(stale_refusals(&family, CHILD_A), 0);
    }

    /// A shell change is a sibling's change for everyone else.
    #[tokio::test]
    async fn a_siblings_shell_change_is_attributed_and_stops_a_stale_write() {
        use crate::agent::coordination::{LogEntry, Writer};
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        let p = file.to_str().unwrap();
        std::fs::write(&file, "alpha\nbeta\n").unwrap();
        let (family, _permit) = tree();
        let mut harness = harness_with_shell(vec![
            read_call(p),
            MockProvider::text_response("Read."),
            read_call(p),
            shell_call(p, "ALPHA\nbeta\n"),
            MockProvider::text_response("B done."),
            edit_call(p, "beta", "BETA"),
            MockProvider::text_response("A stopped."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "ALPHA\nbeta\n");
        let log = family.active_coordination().unwrap().log_entries();
        assert!(matches!(
            &log[..],
            [LogEntry::StaleWriteRefused { changed_by: Some(Writer::Member(by)), .. }]
                if by == CHILD_B
        ));
    }

    /// The real child path enters the roster with what the spawn declared,
    /// and leaves it running only until it finishes.
    #[tokio::test]
    async fn a_spawned_child_joins_the_roster_with_its_scope() {
        use crate::runtime_ports::{ChildAgentRequest, ChildAgentRuntime};
        let (family, _permit) = tree();
        let harness = harness(vec![MockProvider::text_response("Child done.")]).await;
        let agent = Arc::new(harness.agent);
        let mut parent = inherited(&family);
        parent.child_declaration = Some(crate::agent::authority::ChildDeclaration {
            parent_session: "telegram:synthetic-user-1".to_string(),
            scope: vec!["/w/synthetic".to_string()],
        });
        let run = agent
            .run_child(ChildAgentRequest {
                mission: "Tidy the synthetic notes".to_string(),
                task: "Tidy them.".to_string(),
                status_tx: None,
                channel_ctx: ChannelContext::private("test"),
                user_role: UserRole::Owner,
                child_role: None,
                goal_id: None,
                task_id: None,
                project_scope: None,
                specialist: None,
                approval_session_id: None,
                parent_authority: Some(parent),
                time_limit: None,
                model_tier: None,
            })
            .await
            .unwrap();
        let roster = family.active_coordination().unwrap().roster();
        assert_eq!(roster.len(), 1);
        assert_eq!(roster[0].session, run.session);
        assert_eq!(roster[0].scope, vec!["/w/synthetic".to_string()]);
        assert_eq!(roster[0].parent, "telegram:synthetic-user-1");
        assert!(!roster[0].running, "it finished");
    }

    /// The parent's next model call carries the roster, after the transcript
    /// it has already sent.
    #[tokio::test]
    async fn the_parent_is_told_who_is_running_at_the_tail() {
        use crate::agent::coordination::Member;
        const PARENT: &str = "telegram:synthetic-user-1";
        let (family, _permit) = tree();
        family.active_coordination().unwrap().add_member(Member {
            session: CHILD_A.to_string(),
            parent: PARENT.to_string(),
            mission: "Tidy the synthetic notes".to_string(),
            specialist: None,
            scope: Vec::new(),
            running: true,
        });
        let mut harness = harness(vec![MockProvider::text_response("Waiting.")]).await;
        turn(&mut harness, &family, PARENT).await;
        let log = harness.provider.call_log.lock().await;
        let last = log.last().unwrap().messages.last().unwrap().clone();
        assert_eq!(last["role"], "system", "{last}");
        assert!(
            last["content"].as_str().unwrap().contains(CHILD_A),
            "{last}"
        );
    }

    const ROOT_SESSION: &str = "telegram:synthetic-user-1";

    fn parent_says(decision: &str, note: &str) -> crate::traits::ProviderResponse {
        MockProvider::text_response(&json!({"decision": decision, "note": note}).to_string())
    }

    /// A tree whose top-level agent is ready, a shared file both children
    /// have read, and a sibling's overwrite on top of it.
    async fn collision(
        answer: crate::traits::ProviderResponse,
    ) -> (
        tempfile::TempDir,
        std::path::PathBuf,
        SpawnFamily,
        SpawnPermit,
        TestHarness,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap().to_string();
        let (family, permit) = tree();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        let mut harness = harness(vec![
            read_call(&p),
            MockProvider::text_response("Read."),
            write_call(&p, "from b\n"),
            MockProvider::text_response("B done."),
            write_call(&p, "from a\n"),
            answer,
            MockProvider::text_response("A finished."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        (dir, file, family, permit, harness)
    }

    fn decisions(family: &SpawnFamily) -> Vec<crate::agent::coordination::ParentDecision> {
        family
            .coordination()
            .log_entries()
            .iter()
            .filter_map(|entry| match entry {
                crate::agent::coordination::LogEntry::Escalated { decision, .. } => Some(*decision),
                _ => None,
            })
            .collect()
    }

    /// The parent lifts the stale check for this one write.
    #[tokio::test]
    async fn proceed_lets_the_contested_write_land() {
        use crate::agent::coordination::ParentDecision;
        let (_dir, file, family, _permit, mut harness) =
            collision(parent_says("proceed", "Yours is the right version.")).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from a\n");
        assert_eq!(decisions(&family), vec![ParentDecision::Proceed]);
        assert_eq!(
            latest_task_outcome(&harness, CHILD_A).await,
            crate::events::TaskOutcome::Succeeded
        );
    }

    /// `proceed` only lifts the check. An edit that then cannot apply did not
    /// happen, and the child's run must not read as if it had.
    #[tokio::test]
    async fn proceed_on_an_edit_that_then_fails_stays_open() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap();
        let (family, _permit) = tree();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        let mut harness = harness(vec![
            read_call(p),
            MockProvider::text_response("Read."),
            read_call(p),
            edit_call(p, "alpha", "ALPHA"),
            MockProvider::text_response("B done."),
            edit_call(p, "alpha", "OMEGA"),
            parent_says("proceed", "Go ahead."),
            MockProvider::text_response("A finished."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "ALPHA\n");
        assert!(family.coordination().has_unlanded_write(CHILD_A));
    }

    /// Reread sends the child back to the file; the run is an ordinary one.
    #[tokio::test]
    async fn reread_refuses_the_write_and_the_retry_lands() {
        use crate::agent::coordination::ParentDecision;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap();
        let (family, _permit) = tree();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        let mut harness = harness(vec![
            read_call(p),
            MockProvider::text_response("Read."),
            write_call(p, "from b\n"),
            MockProvider::text_response("B done."),
            write_call(p, "from a\n"),
            parent_says("reread", "Keep theirs and add yours."),
            read_call(p),
            write_call(p, "from b\nfrom a\n"),
            MockProvider::text_response("A finished."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from b\nfrom a\n");
        assert_eq!(decisions(&family), vec![ParentDecision::Reread]);
        assert_eq!(
            latest_task_outcome(&harness, CHILD_A).await,
            crate::events::TaskOutcome::Succeeded
        );
    }

    /// Stop means the work did not happen, and a run that did other work
    /// says so.
    #[tokio::test]
    async fn stop_refuses_and_the_run_is_not_a_success() {
        use crate::agent::coordination::ParentDecision;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap();
        let (family, _permit) = tree();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        let mut harness = harness(vec![
            read_call(p),
            MockProvider::text_response("Read."),
            write_call(p, "from b\n"),
            MockProvider::text_response("B done."),
            read_call("/tmp/aidaemon-test/absent-synthetic-file"),
            write_call(p, "from a\n"),
            parent_says("stop", "Another agent owns this."),
            MockProvider::text_response("A stopped."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from b\n");
        assert_eq!(decisions(&family), vec![ParentDecision::Stop]);
        assert_ne!(
            latest_task_outcome(&harness, CHILD_A).await,
            crate::events::TaskOutcome::Succeeded
        );
    }

    /// An answer that is not the strict JSON is no answer.
    #[tokio::test]
    async fn an_unreadable_answer_is_no_answer() {
        use crate::agent::coordination::ParentDecision;
        let (_dir, file, family, _permit, mut harness) =
            collision(MockProvider::text_response("Go ahead, I suppose.")).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from b\n");
        assert_eq!(decisions(&family), vec![ParentDecision::NoAnswer]);
    }

    /// With no top-level agent to ask nothing is put to any model.
    #[tokio::test]
    async fn no_top_level_agent_is_no_answer() {
        use crate::agent::coordination::ParentDecision;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap();
        let (family, _permit) = tree();
        let mut harness = harness(vec![
            read_call(p),
            MockProvider::text_response("Read."),
            write_call(p, "from b\n"),
            MockProvider::text_response("B done."),
            write_call(p, "from a\n"),
            MockProvider::text_response("A stopped."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(decisions(&family), vec![ParentDecision::NoAnswer]);
    }

    /// Past the tree's cap a collision is refused without a side call.
    #[tokio::test]
    async fn the_tree_cap_answers_no_answer_without_asking() {
        use crate::agent::coordination::ParentDecision;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap();
        let family = SpawnFamily::new(crate::agent::authority::SpawnFamilyLimits {
            max_escalations_per_request: 0,
            ..Default::default()
        });
        let _permit = family.try_admit().unwrap();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        let mut harness = harness(vec![
            read_call(p),
            MockProvider::text_response("Read."),
            write_call(p, "from b\n"),
            MockProvider::text_response("B done."),
            write_call(p, "from a\n"),
            MockProvider::text_response("A stopped."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(decisions(&family), vec![ParentDecision::NoAnswer]);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from b\n");
    }

    /// The file moved again while the parent answered: `proceed` covered
    /// what the parent saw, so the write is judged afresh and does not land.
    #[tokio::test]
    async fn proceed_does_not_cover_a_file_that_changed_during_the_wait() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap().to_string();
        let (family, _permit) = tree();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        let zero = std::time::Duration::ZERO;
        let provider = MockProvider::with_delayed_responses(
            vec![
                read_call(&p),
                MockProvider::text_response("Read."),
                write_call(&p, "from b\n"),
                MockProvider::text_response("B done."),
                write_call(&p, "from a\n"),
                parent_says("proceed", "Go."),
                MockProvider::text_response("A finished."),
            ],
            vec![
                zero,
                zero,
                zero,
                zero,
                zero,
                std::time::Duration::from_millis(600),
                zero,
            ],
        );
        let backend = crate::execution::active_execution_backend;
        let tools: Vec<Arc<dyn crate::traits::Tool>> = vec![
            Arc::new(crate::tools::ReadFileTool::new(backend())),
            Arc::new(crate::tools::WriteFileTool::new(backend())),
            Arc::new(crate::tools::EditFileTool::new(backend())),
        ];
        let mut harness = setup_test_agent_with_extra_tools_and_llm_timeout(provider, tools, None)
            .await
            .unwrap();
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        let writer = {
            let file = file.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                std::fs::write(&file, "changed during the wait\n").unwrap();
            })
        };
        turn(&mut harness, &family, CHILD_A).await;
        writer.await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "changed during the wait\n"
        );
    }

    /// A tainted child's escalation marks the top-level session tainted
    /// before the question is used, and its note stays out of the log's view.
    #[tokio::test]
    async fn a_tainted_childs_escalation_taints_the_root_turn() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap().to_string();
        let (family, _permit) = tree();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        let mut harness = harness(vec![
            read_call(&p),
            MockProvider::text_response("Read."),
            write_call(&p, "from b\n"),
            MockProvider::text_response("B done."),
            write_call(&p, "from a\n"),
            parent_says("reread", "secret-marker-note"),
            MockProvider::text_response("A finished."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        let mut parent = inherited(&family);
        parent.untrusted_context = true;
        harness.agent.set_test_inherited_authority(parent);
        harness
            .agent
            .handle_message(
                CHILD_A,
                "Finish the synthetic notes.",
                None,
                UserRole::Owner,
                ChannelContext::private("test"),
                None,
            )
            .await
            .unwrap();
        let tainted = crate::agent::untrusted::tainted_turns(harness.state.as_ref(), ROOT_SESSION)
            .await
            .unwrap();
        assert_eq!(tainted, vec!["synthetic-turn".to_string()]);
        let note = family.coordination().parent_note(ROOT_SESSION);
        assert!(note.is_none_or(|note| !note.contains("secret-marker-note")));
    }

    /// `coordination = false` restores the last-writer-wins behaviour.
    #[tokio::test]
    async fn coordination_off_leaves_writes_unjudged() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap().to_string();
        let family = SpawnFamily::new(crate::agent::authority::SpawnFamilyLimits {
            coordination: false,
            ..Default::default()
        });
        let _permit = family.try_admit().unwrap();
        let mut harness = harness(vec![
            read_call(&p),
            MockProvider::text_response("Read."),
            write_call(&p, "from b\n"),
            MockProvider::text_response("B done."),
            write_call(&p, "from a\n"),
            MockProvider::text_response("A done."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from a\n");
    }

    /// A child's own question comes back as the tool result, and spends its
    /// allowance.
    #[tokio::test]
    async fn a_childs_question_is_answered_as_its_tool_result() {
        use crate::agent::coordination::ParentDecision;
        let (family, _permit) = tree();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        let backend = crate::execution::active_execution_backend;
        let tools: Vec<Arc<dyn crate::traits::Tool>> = vec![
            Arc::new(crate::tools::ReadFileTool::new(backend())),
            Arc::new(crate::tools::EscalateToParentTool),
        ];
        let mut harness = setup_test_agent_with_extra_tools_and_llm_timeout(
            MockProvider::with_responses(vec![
                MockProvider::tool_call_response(
                    "escalate_to_parent",
                    r#"{"question":"Is the header mine to change?"}"#,
                ),
                parent_says("stop", "That is not yours."),
                MockProvider::text_response("Understood."),
            ]),
            tools,
            None,
        )
        .await
        .unwrap();
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(decisions(&family), vec![ParentDecision::Stop]);
    }

    /// A plain child is offered the way to escalate.
    #[tokio::test]
    async fn only_a_plain_child_gets_the_way_to_escalate() {
        use crate::runtime_ports::{ChildAgentRequest, ChildAgentRuntime};
        let (family, _permit) = tree();
        let harness = harness(vec![MockProvider::text_response("Child done.")]).await;
        let provider = harness.provider.clone();
        let agent = Arc::new(harness.agent);
        agent
            .run_child(ChildAgentRequest {
                mission: "Tidy the synthetic notes".to_string(),
                task: "Tidy them.".to_string(),
                status_tx: None,
                channel_ctx: ChannelContext::private("test"),
                user_role: UserRole::Owner,
                child_role: None,
                goal_id: None,
                task_id: None,
                project_scope: None,
                specialist: None,
                approval_session_id: None,
                parent_authority: Some(inherited(&family)),
                time_limit: None,
                model_tier: None,
            })
            .await
            .unwrap();
        let log = provider.call_log.lock().await;
        let names = extract_tool_names(&log.last().unwrap().tools);
        assert!(
            names.contains(&"escalate_to_parent".to_string()),
            "{names:?}"
        );
    }

    /// Two writers touched one file and the root never looked again: it is
    /// asked once to check, and its read closes the demand.
    #[tokio::test]
    async fn a_file_with_two_writers_is_checked_before_the_root_stops() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap();
        let (family, _permit) = tree();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        let mut harness = harness(vec![
            write_call(p, "one\n"),
            MockProvider::text_response("A done."),
            read_call(p),
            write_call(p, "two\n"),
            MockProvider::text_response("B done."),
            // The root tries to finish, is asked to check, reads, finishes.
            MockProvider::text_response("All finished."),
            read_call(p),
            MockProvider::text_response("Checked and finished."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        let reply = turn(&mut harness, &family, ROOT_SESSION).await;
        assert_eq!(reply, "Checked and finished.");
        assert_eq!(
            family.coordination().unverified_shared_files(),
            Vec::<String>::new()
        );
    }

    /// A root that already read the shared file after the last write is not
    /// asked again.
    #[tokio::test]
    async fn a_root_that_already_looked_is_not_asked_to_check() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap();
        let (family, _permit) = tree();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        let mut harness = harness(vec![
            write_call(p, "one\n"),
            MockProvider::text_response("A done."),
            read_call(p),
            write_call(p, "two\n"),
            MockProvider::text_response("B done."),
            read_call(p),
            MockProvider::text_response("Finished."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        let reply = turn(&mut harness, &family, ROOT_SESSION).await;
        assert_eq!(reply, "Finished.");
        assert_eq!(harness.provider.call_log.lock().await.len(), 7);
    }

    #[tokio::test]
    async fn a_shell_call_by_the_root_counts_as_checking() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\n").unwrap();
        let p = file.to_str().unwrap();
        let (family, _permit) = tree();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        let mut harness = harness_with_shell(vec![
            write_call(p, "one\n"),
            MockProvider::text_response("A done."),
            read_call(p),
            write_call(p, "two\n"),
            MockProvider::text_response("B done."),
            MockProvider::tool_call_response("synthetic_shell", r#"{"command":"run the tests"}"#),
            MockProvider::text_response("Finished."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        assert_eq!(turn(&mut harness, &family, ROOT_SESSION).await, "Finished.");
    }
}
