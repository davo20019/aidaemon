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
}
