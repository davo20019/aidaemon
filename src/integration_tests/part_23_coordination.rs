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
}
