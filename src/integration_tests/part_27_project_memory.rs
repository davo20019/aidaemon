// ==========================================================================
// Project memory: a conversation placed in a project keeps what it learns
// about the project in the project's own notes, and the owner's personal
// memory is never replaced, merged or crowded by it.
// ==========================================================================

mod project_memory {
    use super::*;

    use aidaemon_memory::project::NoteKind;

    /// A project with a folder, and `session` placed in it.
    async fn place_in_project(
        harness: &crate::testing::TestHarness,
        session: &str,
        name: &str,
    ) -> (aidaemon_core::ProjectRef, tempfile::TempDir) {
        let folder = tempfile::tempdir().unwrap();
        let project = harness.state.create_work_project(name, None).await.unwrap();
        let path = dunce::canonicalize(folder.path()).unwrap();
        harness
            .state
            .update_work_project(&project.id, None, Some(Some(path.to_str().unwrap())))
            .await
            .unwrap();
        assert!(
            harness
                .state
                .set_session_work_project(session, &project.id)
                .await
                .unwrap()
        );
        let project = aidaemon_core::ProjectRef::new(project.id, name, Some(path)).unwrap();
        (project, folder)
    }

    fn remember(args: serde_json::Value) -> crate::traits::ProviderResponse {
        let mut response = MockProvider::tool_call_response("remember_fact", &args.to_string());
        response.content = Some("Saving that for later work.".to_string());
        response
    }

    /// Every message text the model was sent, across calls.
    async fn sent_texts(harness: &crate::testing::TestHarness) -> Vec<String> {
        harness
            .provider
            .call_log
            .lock()
            .await
            .iter()
            .flat_map(|call| call.messages.iter())
            .filter_map(|message| message["content"].as_str().map(str::to_string))
            .collect()
    }

    async fn personal_facts(harness: &crate::testing::TestHarness) -> Vec<(i64, String, String)> {
        let mut facts: Vec<_> = harness
            .state
            .memory_stores()
            .get_facts(None)
            .await
            .unwrap()
            .into_iter()
            .map(|fact| (fact.id, fact.key, fact.value))
            .collect();
        facts.sort();
        facts
    }

    #[tokio::test]
    async fn a_project_conversation_keeps_project_facts_in_its_notes() {
        let harness = setup_test_agent(MockProvider::with_responses(vec![
            remember(serde_json::json!({"facts": [
                {"category": "technical", "key": "test_command", "value": "Tests run with make test."},
                {"category": "technical", "key": "editor_settings", "value": "The repo pins editorconfig with 4-space indents."}
            ]})),
            MockProvider::text_response("Noted for this project."),
            MockProvider::text_response("Use make test."),
        ]))
        .await
        .unwrap();
        let memory = harness.state.memory_stores();
        for (category, key, value) in [
            ("user", "name", "Alice Rivera"),
            ("preference", "editor", "vim"),
            ("technical", "test_command", "pytest -q in the blog repo"),
        ] {
            memory
                .upsert_fact(
                    category,
                    key,
                    value,
                    "user",
                    None,
                    aidaemon_memory::FactPrivacy::Global,
                )
                .await
                .unwrap();
        }
        let before = personal_facts(&harness).await;
        let session = "app:project-memory-1";
        let (project, _folder) = place_in_project(&harness, session, "Acme Site").await;

        harness
            .agent
            .handle_message(
                session,
                "remember how tests run here and the editor settings",
                None,
                UserRole::Owner,
                ChannelContext::private("app"),
                None,
            )
            .await
            .unwrap();

        // Nothing personal changed: same rows, same values, none superseded.
        assert_eq!(personal_facts(&harness).await, before);
        let note = harness
            .project_notes
            .read(&project, NoteKind::Fact, "test_command")
            .expect("the project's note");
        assert!(note.contains("Tests run with make test."), "{note}");
        assert_eq!(harness.project_notes.entries(&project).len(), 2);

        // The next turn in the project sees the notes beside the owner's memory.
        harness
            .agent
            .handle_message(
                session,
                "how do I run the tests?",
                None,
                UserRole::Owner,
                ChannelContext::private("app"),
                None,
            )
            .await
            .unwrap();
        let texts = sent_texts(&harness).await;
        let block = texts
            .iter()
            .rev()
            .find(|text| text.contains("## Project Memory (Acme Site)"))
            .expect("the project section");
        assert!(block.contains("fact-test_command.md"), "{block}");
    }

    #[tokio::test]
    async fn a_conversation_without_a_project_remembers_as_before() {
        let harness = setup_test_agent(MockProvider::with_responses(vec![
            remember(serde_json::json!(
                {"category": "technical", "key": "test_command", "value": "make test"}
            )),
            MockProvider::text_response("Noted."),
        ]))
        .await
        .unwrap();
        // Another conversation is in a project; this one is not.
        let (project, _folder) =
            place_in_project(&harness, "app:some-other-session", "Acme Site").await;
        harness
            .agent
            .handle_message(
                "app:plain-chat",
                "remember that tests run with make test",
                None,
                UserRole::Owner,
                ChannelContext::private("app"),
                None,
            )
            .await
            .unwrap();
        let facts = personal_facts(&harness).await;
        assert!(
            facts
                .iter()
                .any(|(_, key, value)| key == "test_command" && value == "make test"),
            "{facts:?}"
        );
        assert!(harness.project_notes.entries(&project).is_empty());
        assert!(
            sent_texts(&harness)
                .await
                .iter()
                .all(|text| !text.contains("Project Memory")),
        );
    }

    #[tokio::test]
    async fn a_shared_group_in_a_project_gets_no_project_memory() {
        let harness = setup_test_agent(MockProvider::with_responses(vec![
            MockProvider::text_response("Hello."),
        ]))
        .await
        .unwrap();
        let session = "slack:group-in-project";
        let (project, _folder) = place_in_project(&harness, session, "Acme Site").await;
        harness
            .project_notes
            .write(
                &project,
                &aidaemon_memory::project::NoteWrite {
                    kind: NoteKind::Fact,
                    key: "db".into(),
                    category: Some("technical".into()),
                    description: None,
                    body: "Postgres 16".into(),
                    source: aidaemon_memory::project::NoteSource::Agent,
                },
            )
            .unwrap();
        let mut group = ChannelContext::private("slack");
        group.visibility = crate::types::ChannelVisibility::PrivateGroup;
        harness
            .agent
            .handle_message(session, "hi all", None, UserRole::Owner, group, None)
            .await
            .unwrap();
        assert!(
            sent_texts(&harness)
                .await
                .iter()
                .all(|text| !text.contains("Project Memory")),
        );
    }

    /// A sub-agent works for the chat that started it: it sees that chat's
    /// project memory, and what it remembers on purpose goes to the project.
    #[tokio::test]
    async fn a_sub_agent_of_a_project_chat_shares_its_project_memory() {
        let harness = setup_test_agent(MockProvider::with_responses(vec![
            remember(serde_json::json!(
                {"category": "technical", "key": "lint_command", "value": "Lint with make lint."}
            )),
            MockProvider::text_response("Linted and noted."),
        ]))
        .await
        .unwrap();
        let parent = "app:project-parent";
        let (project, folder) = place_in_project(&harness, parent, "Acme Site").await;
        harness
            .project_notes
            .write(
                &project,
                &aidaemon_memory::project::NoteWrite {
                    kind: NoteKind::Fact,
                    key: "test_command".into(),
                    category: Some("technical".into()),
                    description: None,
                    body: "Tests run with make test.".into(),
                    source: aidaemon_memory::project::NoteSource::Agent,
                },
            )
            .unwrap();
        let folder = dunce::canonicalize(folder.path()).unwrap();
        let agent = Arc::new(harness.agent);
        agent
            .spawn_child_with_outcome(
                "Lint the project",
                "Run the linter and remember how it runs.",
                None,
                ChannelContext::private("app"),
                UserRole::Owner,
                None,
                None,
                None,
                Some(folder.to_str().unwrap()),
                None,
                Some(parent),
                None,
                None,
                None,
            )
            .await
            .unwrap();
        let texts: Vec<String> = harness
            .provider
            .call_log
            .lock()
            .await
            .iter()
            .flat_map(|call| call.messages.iter())
            .filter_map(|message| message["content"].as_str().map(str::to_string))
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("## Project Memory (Acme Site)")),
            "the child sees the project's notes"
        );
        assert!(
            harness
                .project_notes
                .read(&project, NoteKind::Fact, "lint_command")
                .is_some()
        );
    }

    /// A server a project declared and the owner started runs under the
    /// project's name: its tools are offered on that project's turns, and
    /// never in a chat outside it or in another project.
    #[tokio::test]
    async fn a_project_servers_tools_are_offered_only_in_its_project() {
        let mut harness = setup_test_agent(MockProvider::with_responses(vec![
            MockProvider::text_response("In the project."),
            MockProvider::text_response("Outside it."),
            MockProvider::text_response("In another project."),
        ]))
        .await
        .unwrap();
        let in_project = "app:project-mcp-in";
        let (project, _folder) = place_in_project(&harness, in_project, "Acme Site").await;
        let (_other, _other_folder) =
            place_in_project(&harness, "app:project-mcp-other", "Other Site").await;
        let registry = crate::mcp::McpRegistry::new(
            harness.state.clone(),
            crate::config::secret_store().clone(),
        );
        let name = aidaemon_mcp::registry::project::registered_name(&project.id, "fake");
        registry
            .add_server(
                &name,
                crate::config::McpServerConfig {
                    command: crate::testing::fake_mcp_server_path()
                        .to_string_lossy()
                        .into_owned(),
                    args: Vec::new(),
                    env: std::collections::HashMap::new(),
                    read_only_tools: Vec::new(),
                },
                false,
            )
            .await
            .unwrap();
        harness.agent.set_test_mcp_registry(registry);
        harness.agent.set_test_tool_filter_enforce(false);
        let tool = format!("{name}__echo");
        for session in [in_project, "app:project-mcp-plain", "app:project-mcp-other"] {
            harness
                .agent
                .handle_message(
                    session,
                    "Say something short.",
                    None,
                    UserRole::Owner,
                    ChannelContext::private("app"),
                    None,
                )
                .await
                .unwrap();
        }
        let offered: Vec<String> = harness
            .provider
            .call_log
            .lock()
            .await
            .iter()
            .map(|call| serde_json::Value::Array(call.tools.clone()).to_string())
            .collect();
        assert_eq!(offered.len(), 3, "one model call per turn");
        assert!(offered[0].contains(&tool), "the project's own turn: {}", offered[0]);
        assert!(!offered[1].contains(&tool), "a chat in no project");
        assert!(!offered[2].contains(&tool), "another project's turn");
    }
}
