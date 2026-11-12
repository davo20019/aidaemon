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
            learning: None,
            spawn_family: Some(family.clone()),
            child_declaration: None,
            parent_run: None,
            known_resources: Default::default(),
            card_title: None,
            lineage: Vec::new(),
            parent_task: None,
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
            precedence: None,
            claim: None,
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

    /// A tree whose top-level agent is ready, a file a sibling overwrote,
    /// and a writer that has read the sibling's version and still wants to
    /// replace it: a contest, put to the coordinator.
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
            read_call(&p),
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

    /// The coordinator lets the change through: it lands, and the side that
    /// lost cannot quietly put its own back.
    #[tokio::test]
    async fn apply_lets_the_contested_write_land_and_binds_the_other_side() {
        use crate::agent::coordination::ParentDecision;
        let (_dir, file, family, _permit, mut harness) =
            collision(parent_says("apply", "Yours is the right version.")).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from a\n");
        assert_eq!(decisions(&family), vec![ParentDecision::Apply]);
        assert_eq!(
            latest_task_outcome(&harness, CHILD_A).await,
            crate::events::TaskOutcome::Succeeded
        );
        // B reads A's version and tries to restore its own: refused by the
        // ruling, with no second question.
        let p = file.to_str().unwrap().to_string();
        harness
            .provider
            .push_responses(vec![
                read_call(&p),
                write_call(&p, "from b\n"),
                MockProvider::text_response("B gave way."),
            ])
            .await;
        turn(&mut harness, &family, CHILD_B).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from a\n");
        assert_eq!(decisions(&family), vec![ParentDecision::Apply]);
    }

    /// The live failure this replaces: the text an edit targets is gone. The
    /// writer is refused and reads again, sees for itself, and the
    /// coordinator is never asked.
    #[tokio::test]
    async fn an_edit_whose_text_is_gone_never_reaches_the_coordinator() {
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
            // Refused as stale; then, after reading, the tool's own error.
            edit_call(p, "alpha", "OMEGA"),
            read_call(p),
            edit_call(p, "alpha", "OMEGA"),
            MockProvider::text_response("A saw the line was already changed."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "ALPHA\n");
        assert_eq!(decisions(&family), Vec::new());
        assert_eq!(stale_refusals(&family, CHILD_A), 1);
        harness
            .provider
            .assert_response_script_consumed()
            .await
            .unwrap();
    }

    /// Reading again does not make a sibling's change the reader's to
    /// replace: the overwrite is put to the coordinator, who keeps the
    /// sibling's, and a second try is refused without a second question.
    #[tokio::test]
    async fn an_overwrite_after_a_reread_is_contested_and_keep_holds() {
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
            // Stale: refused with no question.
            write_call(p, "from a\n"),
            // Read, then the same overwrite: a contest.
            read_call(p),
            write_call(p, "from a\n"),
            parent_says("keep", "B's version is the one we want."),
            // Trying again changes nothing and asks nobody.
            edit_call(p, "from b", "from a"),
            MockProvider::text_response("A left it."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from b\n");
        assert_eq!(decisions(&family), vec![ParentDecision::Keep]);
        harness
            .provider
            .assert_response_script_consumed()
            .await
            .unwrap();
        // What makes the child's run `Partial` when it reports to its parent.
        assert!(family.coordination().has_unlanded_write(CHILD_A));
    }

    /// A retry that keeps the sibling's lines and adds its own is ordinary:
    /// it lands, nobody is asked, and the run is a success.
    #[tokio::test]
    async fn a_retry_that_only_adds_its_own_lines_lands_without_a_question() {
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
            read_call(p),
            write_call(p, "from b\nfrom a\n"),
            MockProvider::text_response("A finished."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from b\nfrom a\n");
        assert_eq!(decisions(&family), Vec::new());
        assert_eq!(
            latest_task_outcome(&harness, CHILD_A).await,
            crate::events::TaskOutcome::Succeeded
        );
    }

    /// The coordinator said who wins when it started the second child: a
    /// contest in the overlap is settled by that, both ways, with no
    /// question, and the coordinator is told.
    #[tokio::test]
    async fn precedence_given_at_spawn_settles_the_contest_without_a_question() {
        use crate::agent::coordination::{Member, Precedence};
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "alpha\nbeta\n").unwrap();
        let p = file.to_str().unwrap();
        let (family, _permit) = tree();
        family.coordination().bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        for (session, said) in [(CHILD_A, None), (CHILD_B, Some(Precedence::ThisChild))] {
            family.coordination().add_declared_member(
                Member {
                    session: session.to_string(),
                    parent: ROOT_SESSION.to_string(),
                    mission: "Update the synthetic notes".to_string(),
                    specialist: None,
                    scope: vec![real(&file)],
                    running: true,
                },
                None,
                said,
            );
        }
        let mut harness = harness(vec![
            // A changes the first line.
            read_call(p),
            edit_call(p, "alpha", "ALPHA-A"),
            MockProvider::text_response("A done."),
            // B, who has precedence, changes A's line and its own.
            read_call(p),
            edit_call(p, "ALPHA-A", "ALPHA-B"),
            edit_call(p, "beta", "BETA-B"),
            MockProvider::text_response("B done."),
            // A reads and tries to take both back: refused by the rule.
            read_call(p),
            write_call(p, "ALPHA-A\nBETA-A\n"),
            MockProvider::text_response("A gave way."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "ALPHA-B\nBETA-B\n"
        );
        assert_eq!(decisions(&family), Vec::new());
        let note = family.coordination().parent_note(ROOT_SESSION).unwrap();
        assert!(
            note.contains("takes precedence there") && note.contains("was refused under"),
            "{note}"
        );
    }

    /// "Instead" refuses the write and hands the child the coordinator's
    /// instruction; the run is not held open for it.
    #[tokio::test]
    async fn instead_refuses_the_write_and_passes_the_instruction_on() {
        use crate::agent::coordination::ParentDecision;
        let (_dir, file, family, _permit, mut harness) =
            collision(parent_says("instead", "Append your line under theirs.")).await;
        turn(&mut harness, &family, CHILD_A).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from b\n");
        assert_eq!(decisions(&family), vec![ParentDecision::Instead]);
        assert!(!family.coordination().has_unlanded_write(CHILD_A));
    }

    /// What the coordinator is put, and what it may answer: the lines as
    /// they were, are and would be, and only decisions that can happen.
    #[tokio::test]
    async fn the_coordinator_is_shown_the_lines_and_only_possible_decisions() {
        let (_dir, _file, family, _permit, mut harness) =
            collision(parent_says("keep", "Leave it.")).await;
        turn(&mut harness, &family, CHILD_A).await;
        let log = harness.provider.call_log.lock().await;
        let asked = log
            .iter()
            .find(|call| {
                call.messages[0]["content"]
                    .as_str()
                    .is_some_and(|system| system.starts_with("You coordinate a team"))
            })
            .expect("the coordinator was asked");
        let system = asked.messages[0]["content"].as_str().unwrap();
        for offered in ["\"apply\"", "\"keep\"", "\"stop\"", "\"instead\""] {
            assert!(system.contains(offered), "{offered}: {system}");
        }
        assert!(!system.contains("reread"), "{system}");
        let packet = asked.messages[1]["content"].as_str().unwrap();
        for fact in ["| alpha", "| from b", "| from a", CHILD_B] {
            assert!(packet.contains(fact), "{fact}: {packet}");
        }
        assert!(asked.tools.is_empty());
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
            read_call(p),
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
            read_call(p),
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
            read_call(p),
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

    /// The file moved again while the parent answered: `apply` covered
    /// what the parent saw, so the write is judged afresh and does not land.
    #[tokio::test]
    async fn apply_does_not_cover_a_file_that_changed_during_the_wait() {
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
                read_call(&p),
                write_call(&p, "from a\n"),
                parent_says("apply", "Go."),
                MockProvider::text_response("A finished."),
            ],
            vec![
                zero,
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
            read_call(&p),
            write_call(&p, "from a\n"),
            parent_says("keep", "secret-marker-note"),
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
            write_call(p, "one\ntwo\n"),
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
            write_call(p, "one\ntwo\n"),
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
            write_call(p, "one\ntwo\n"),
            MockProvider::text_response("B done."),
            MockProvider::tool_call_response("synthetic_shell", r#"{"command":"run the tests"}"#),
            MockProvider::text_response("Finished."),
        ])
        .await;
        turn(&mut harness, &family, CHILD_A).await;
        turn(&mut harness, &family, CHILD_B).await;
        assert_eq!(turn(&mut harness, &family, ROOT_SESSION).await, "Finished.");
    }

    /// One seeded conflict between two children over one file.
    struct Seeded {
        name: &'static str,
        /// What the coordinator said when it started the second child.
        rule: Option<crate::agent::coordination::Precedence>,
        /// The sessions' turns in order, and the model's replies across them.
        turns: &'static [&'static str],
        script: fn(&str) -> Vec<crate::traits::ProviderResponse>,
        /// The file at the end.
        end: &'static str,
        /// The coordinator's answers, in order.
        asked: &'static [&'static str],
    }

    /// What one run of a seeded conflict did, from the tree's typed record.
    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    struct Tally {
        end: String,
        /// The coordinator's answers, in order.
        asked: Vec<&'static str>,
        stale_refusals: usize,
        settled_without_asking: usize,
        /// Changes to a sibling's lines by a shell, seen only afterwards.
        shell_overwrites: usize,
        /// Children left with a refused change they never made.
        unlanded: usize,
        /// A sibling's change replaced by a file tool with no ruling, rule
        /// or answer behind it. Must be zero.
        undecided_overwrites: usize,
    }

    const SEED: &str = "alpha\nbeta\ngamma\n";

    async fn run_seeded(seeded: &Seeded) -> Tally {
        use crate::agent::coordination::{LogEntry, Member, ParentDecision};
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        std::fs::write(&file, SEED).unwrap();
        let p = file.to_str().unwrap();
        let (family, _permit) = tree();
        let coordination = family.coordination();
        coordination.bind_root(
            ROOT_SESSION,
            "Update the synthetic notes.",
            "synthetic-turn",
        );
        for (session, said) in [(CHILD_A, None), (CHILD_B, seeded.rule)] {
            coordination.add_declared_member(
                Member {
                    session: session.to_string(),
                    parent: ROOT_SESSION.to_string(),
                    mission: "Update the synthetic notes".to_string(),
                    specialist: None,
                    scope: vec![real(&file)],
                    running: true,
                },
                None,
                said,
            );
        }
        let mut harness = harness_with_shell((seeded.script)(p)).await;
        for session in seeded.turns {
            turn(&mut harness, &family, session).await;
        }
        harness
            .provider
            .assert_response_script_consumed()
            .await
            .unwrap_or_else(|error| panic!("{}: {error}", seeded.name));
        let log = coordination.log_entries();
        let count = |wanted: fn(&LogEntry) -> bool| log.iter().filter(|e| wanted(e)).count();
        let end = std::fs::read_to_string(&file).unwrap();
        let asked: Vec<&'static str> = decisions(&family)
            .into_iter()
            .map(ParentDecision::as_str)
            .collect();
        let decided = asked.contains(&"apply")
            || count(|e| matches!(e, LogEntry::ContestSettled { landed: true, .. })) > 0;
        let shell_overwrites = count(|e| matches!(e, LogEntry::ShellOverwrote { .. }));
        // Every scenario has B put "-B" on a line; it is gone only if
        // something replaced it.
        let replaced = !end.contains("-B");
        Tally {
            asked,
            stale_refusals: count(|e| matches!(e, LogEntry::StaleWriteRefused { .. })),
            settled_without_asking: count(|e| matches!(e, LogEntry::ContestSettled { .. })),
            shell_overwrites,
            unlanded: [CHILD_A, CHILD_B]
                .iter()
                .filter(|child| coordination.has_unlanded_write(child))
                .count(),
            undecided_overwrites: usize::from(replaced && !decided && shell_overwrites == 0),
            end,
        }
    }

    fn seeded_conflicts() -> Vec<Seeded> {
        use crate::agent::coordination::Precedence;
        vec![
            Seeded {
                name: "non_overlapping_edits",
                rule: None,
                turns: &[CHILD_A, CHILD_B, CHILD_A],
                script: |p| {
                    vec![
                        read_call(p),
                        MockProvider::text_response("Read."),
                        read_call(p),
                        edit_call(p, "beta", "BETA-B"),
                        MockProvider::text_response("Done."),
                        edit_call(p, "gamma", "GAMMA-A"),
                        read_call(p),
                        edit_call(p, "gamma", "GAMMA-A"),
                        MockProvider::text_response("Done."),
                    ]
                },
                end: "alpha\nBETA-B\nGAMMA-A\n",
                asked: &[],
            },
            Seeded {
                name: "same_line_edits",
                rule: None,
                turns: &[CHILD_A, CHILD_B, CHILD_A],
                script: |p| {
                    vec![
                        read_call(p),
                        MockProvider::text_response("Read."),
                        read_call(p),
                        edit_call(p, "beta", "BETA-B"),
                        MockProvider::text_response("Done."),
                        edit_call(p, "beta", "BETA-A"),
                        read_call(p),
                        MockProvider::text_response("B already changed it."),
                    ]
                },
                end: "alpha\nBETA-B\ngamma\n",
                asked: &[],
            },
            Seeded {
                name: "edit_after_reread_onto_a_siblings_line",
                rule: None,
                turns: &[CHILD_A, CHILD_B, CHILD_A],
                script: |p| {
                    vec![
                        read_call(p),
                        MockProvider::text_response("Read."),
                        read_call(p),
                        edit_call(p, "beta", "BETA-B"),
                        MockProvider::text_response("Done."),
                        edit_call(p, "beta", "BETA-A"),
                        read_call(p),
                        edit_call(p, "BETA-B", "BETA-A"),
                        parent_says("keep", "B's line stands."),
                        MockProvider::text_response("Left it."),
                    ]
                },
                end: "alpha\nBETA-B\ngamma\n",
                asked: &["keep"],
            },
            Seeded {
                name: "full_overwrite",
                rule: None,
                turns: &[CHILD_A, CHILD_B, CHILD_A],
                script: |p| {
                    vec![
                        read_call(p),
                        MockProvider::text_response("Read."),
                        read_call(p),
                        edit_call(p, "beta", "BETA-B"),
                        MockProvider::text_response("Done."),
                        write_call(p, "all from a\n"),
                        read_call(p),
                        write_call(p, "all from a\n"),
                        parent_says("apply", "Replace it."),
                        MockProvider::text_response("Done."),
                    ]
                },
                end: "all from a\n",
                asked: &["apply"],
            },
            Seeded {
                name: "shell_write_over_a_siblings_line",
                rule: None,
                turns: &[CHILD_B, CHILD_A],
                script: |p| {
                    vec![
                        read_call(p),
                        edit_call(p, "beta", "BETA-B"),
                        MockProvider::text_response("Done."),
                        read_call(p),
                        shell_call(p, "alpha\nfrom a shell\ngamma\n"),
                        MockProvider::text_response("Done."),
                    ]
                },
                end: "alpha\nfrom a shell\ngamma\n",
                asked: &[],
            },
            Seeded {
                name: "target_text_gone",
                rule: None,
                turns: &[CHILD_B, CHILD_A],
                script: |p| {
                    vec![
                        read_call(p),
                        edit_call(p, "beta", "BETA-B"),
                        MockProvider::text_response("Done."),
                        read_call(p),
                        edit_call(p, "beta\n", "BETA-A\n"),
                        MockProvider::text_response("The line is not there."),
                    ]
                },
                end: "alpha\nBETA-B\ngamma\n",
                asked: &[],
            },
            Seeded {
                name: "rule_resolved_contest",
                rule: Some(Precedence::RunningSiblings),
                turns: &[CHILD_B, CHILD_A, CHILD_B],
                script: |p| {
                    vec![
                        read_call(p),
                        edit_call(p, "beta", "BETA-B"),
                        MockProvider::text_response("Done."),
                        // A has precedence: its change to B's line lands.
                        read_call(p),
                        edit_call(p, "BETA-B", "BETA-A"),
                        MockProvider::text_response("Done."),
                        // B's attempt to take it back is refused.
                        read_call(p),
                        edit_call(p, "BETA-A", "BETA-B"),
                        MockProvider::text_response("Left it."),
                    ]
                },
                end: "alpha\nBETA-A\ngamma\n",
                asked: &[],
            },
            Seeded {
                name: "unruled_contest",
                rule: Some(Precedence::Shared),
                turns: &[CHILD_B, CHILD_A],
                script: |p| {
                    vec![
                        read_call(p),
                        edit_call(p, "beta", "BETA-B"),
                        MockProvider::text_response("Done."),
                        read_call(p),
                        edit_call(p, "BETA-B", "BETA-A"),
                        parent_says("instead", "Add your line under B's."),
                        edit_call(p, "gamma", "BETA-A\ngamma"),
                        MockProvider::text_response("Done."),
                    ]
                },
                end: "alpha\nBETA-B\nBETA-A\ngamma\n",
                asked: &["instead"],
            },
        ]
    }

    /// Every seeded conflict, several times over: the same seed is settled
    /// the same way each run, the coordinator is asked only about informed
    /// changes to a sibling's lines, and no file tool ever replaces a
    /// sibling's change without a ruling, a rule or an answer behind it.
    /// Run with `--nocapture` for the table.
    #[tokio::test]
    async fn seeded_conflicts_are_settled_the_same_way_every_run() {
        const RUNS: usize = 3;
        let mut asked_total = 0usize;
        let mut conflicts = 0usize;
        println!("scenario | asked | stale refused | settled unasked | shell overwrites | unlanded | undecided overwrites | agreement");
        for seeded in seeded_conflicts() {
            let mut runs = Vec::new();
            for _ in 0..RUNS {
                runs.push(run_seeded(&seeded).await);
            }
            let first = runs[0].clone();
            let agreeing = runs.iter().filter(|run| **run == first).count();
            println!(
                "{} | {:?} | {} | {} | {} | {} | {} | {agreeing}/{RUNS}",
                seeded.name,
                first.asked,
                first.stale_refusals,
                first.settled_without_asking,
                first.shell_overwrites,
                first.unlanded,
                first.undecided_overwrites,
            );
            assert_eq!(agreeing, RUNS, "{}: {runs:?}", seeded.name);
            assert_eq!(first.end, seeded.end, "{}", seeded.name);
            assert_eq!(first.asked, seeded.asked, "{}", seeded.name);
            assert_eq!(first.undecided_overwrites, 0, "{}", seeded.name);
            asked_total += first.asked.len();
            conflicts += 1;
        }
        println!("coordinator calls per seeded conflict: {asked_total}/{conflicts}");
        assert_eq!(asked_total, 3, "only the three informed contests with no rule");
    }
}
