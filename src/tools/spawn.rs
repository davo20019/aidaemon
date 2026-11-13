//! `spawn_agent` lives in `aidaemon-runtime` ([`aidaemon_runtime::spawn_tool`]).

pub use aidaemon_runtime::spawn_tool::*;

#[cfg(test)]
mod test_scope {
    pub(super) use std::sync::{Arc, Weak};
    pub(super) use std::time::Duration;

    pub(super) use aidaemon_core::SourceTrust;
    pub(super) use aidaemon_runtime::turn::sliding_window::MEASURED_TIMING_MARK;
    pub(super) use async_trait::async_trait;
    pub(super) use serde_json::json;
    pub(super) use tokio::sync::mpsc;

    pub(super) use crate::events::TaskOutcome;
    pub(super) use crate::runtime_ports::{
        ChildAgentRequest, ChildAgentRun, ChildAgentRuntime, ChildTimedOut, OutboundRouter,
    };
    pub(super) use crate::traits::{AgentRole, StateStore, Tool, ToolCallContext};
    pub(super) use crate::types::{ChannelVisibility, StatusUpdate, UserRole};
}
#[cfg(test)]
use test_scope::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{NotificationStore, StateStore};
    use std::sync::Arc;

    /// A background spawn ends the turn from the tool's typed outcome, not
    /// from its wording: the finish and the handoff fact are set whatever
    /// the message says.
    #[test]
    fn background_spawn_hands_the_turn_off() {
        let outcome = background_spawn_outcome("Summarize the synthetic report");
        assert!(outcome.metadata.background_handoff);
        assert!(
            !outcome.metadata.background_started,
            "a sub-agent is not a tracked process launch"
        );
        let finish = outcome.finish.expect("the turn hands off");
        assert_eq!(finish.outcome, aidaemon_core::FinishOutcome::Partial);
        assert_eq!(
            finish.reply, None,
            "the dispatcher writes the handoff reply"
        );
    }

    #[test]
    fn truncate_utf8_ascii() {
        assert_eq!(truncate_utf8("hello world", 5), "hello");
        assert_eq!(truncate_utf8("hello", 10), "hello");
        assert_eq!(truncate_utf8("hello", 5), "hello");
    }

    #[test]
    fn truncate_utf8_multibyte() {
        // Each emoji is 4 bytes
        let s = "🔥🔥🔥";
        assert_eq!(s.len(), 12);
        // Limit 4 should include exactly the first emoji
        assert_eq!(truncate_utf8(s, 4), "🔥");
        // Limit 5 should still only include the first emoji (next starts at 4, ends at 8)
        assert_eq!(truncate_utf8(s, 5), "🔥");
        // Limit 8 should include two emojis
        assert_eq!(truncate_utf8(s, 8), "🔥🔥");
        // Limit 1 — no full character fits, but char_indices first is (0, '🔥')
        // take_while(|&i| i <= 1) → only i=0 qualifies
        assert_eq!(truncate_utf8(s, 1), "");
    }

    #[test]
    fn truncate_utf8_mixed() {
        let s = "hi🌍!";
        // 'h'=1, 'i'=1, '🌍'=4, '!'=1 → total 7
        assert_eq!(truncate_utf8(s, 3), "hi");
        assert_eq!(truncate_utf8(s, 6), "hi🌍");
        assert_eq!(truncate_utf8(s, 7), "hi🌍!");
    }

    #[test]
    fn truncate_utf8_empty() {
        assert_eq!(truncate_utf8("", 10), "");
        assert_eq!(truncate_utf8("", 0), "");
    }

    #[test]
    fn partial_background_result_is_not_labeled_complete() {
        let (notification_type, message) = format_background_completion(
            "Build and verify the site",
            "Blocked on workspace access.",
            TaskOutcome::Partial,
            false,
            8_000,
        );
        assert_eq!(notification_type, "partial");
        assert!(message.starts_with("\u{26a0}\u{fe0f} Didn't finish in the background"));
        assert!(!message.contains("\u{2705} Finished in the background"));

        // Its own typed verdict was done and only receipts are missing: it
        // finished, and says what its tools did not confirm.
        let (notification_type, message) = format_background_completion(
            "Count to 30",
            "Done. I counted from 1 to 30.",
            TaskOutcome::Partial,
            true,
            8_000,
        );
        assert_eq!(notification_type, "completed");
        assert!(message.starts_with("\u{2705} Finished in the background: Count to 30"));
        assert!(message.ends_with("its tools didn't confirm every step it listed."));
    }

    #[test]
    fn successful_background_result_keeps_completion_label() {
        let (notification_type, message) = format_background_completion(
            "Build and verify the site",
            "Build passed.",
            TaskOutcome::Succeeded,
            false,
            8_000,
        );
        assert_eq!(notification_type, "completed");
        assert!(message.starts_with("\u{2705} Finished in the background"));
    }

    /// A helper's title, and each plan item, names the folder the work
    /// creates, so the owner's first "Started in the background" lines say
    /// where it goes (live 2026-11-13: the new -6 folders showed up first in
    /// a status reply).
    #[tokio::test]
    async fn titles_and_plan_items_name_the_folder_the_work_creates() {
        let tool = SpawnAgentTool::new_deferred(8000, 300);
        let schema = tool.schema();
        let title = schema["parameters"]["properties"]["title"]["description"]
            .as_str()
            .unwrap();
        assert!(
            title.contains("when the work creates or works in a folder, that folder's name"),
            "{title}"
        );
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let plans = crate::tools::track_requirements::TrackRequirementsTool::new(Arc::new(
            crate::plans::PlanStore::new(pool).await.unwrap(),
        ));
        assert!(crate::traits::Tool::description(&plans)
            .contains("name the folder or file an item creates"));
    }

    #[test]
    fn deferred_initialization_not_set() {
        let tool = SpawnAgentTool::new_deferred(8000, 300);
        let result = tool.get_agent();
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("not set"));
    }

    #[test]
    fn config_defaults() {
        use crate::config::SubagentsConfig;
        let cfg = SubagentsConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.max_depth, 3);
        assert_eq!(cfg.max_iterations, 10);
        assert_eq!(cfg.max_response_chars, 8000);
        assert_eq!(cfg.timeout_secs, 300);
    }

    #[test]
    fn deferred_hub_not_set() {
        let tool = SpawnAgentTool::new_deferred(8000, 300);
        assert!(tool.get_hub().is_none());
    }

    #[test]
    fn spawn_args_background_default() {
        let json = r#"{"mission": "test", "task": "do stuff"}"#;
        let args: SpawnArgs = serde_json::from_str(json).unwrap();
        assert!(!args.background);
    }

    #[test]
    fn spawn_args_background_true() {
        let json = r#"{"mission": "test", "task": "do stuff", "background": true}"#;
        let args: SpawnArgs = serde_json::from_str(json).unwrap();
        assert!(args.background);
    }

    fn spawn_args(extra: serde_json::Value) -> SpawnArgs {
        let mut args = serde_json::json!({"mission": "m", "task": "t"});
        args.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(args).unwrap()
    }

    #[test]
    fn a_child_is_narrowed_only_when_asked() {
        assert!(
            requested_child_authority(&spawn_args(serde_json::json!({})))
                .unwrap()
                .is_none()
        );

        let requested =
            requested_child_authority(&spawn_args(serde_json::json!({"access": "read_only"})))
                .unwrap()
                .unwrap();
        assert_eq!(requested.level, aidaemon_core::AccessLevel::ReadOnly);
        assert_eq!(requested.tools, None);

        let requested = requested_child_authority(&spawn_args(
            serde_json::json!({"tools": ["read_file", " search_files "]}),
        ))
        .unwrap()
        .unwrap();
        assert_eq!(requested.level, aidaemon_core::AccessLevel::Write);
        assert_eq!(
            requested.tools.unwrap().into_iter().collect::<Vec<_>>(),
            ["read_file", "search_files"]
        );

        assert!(
            requested_child_authority(&spawn_args(serde_json::json!({"tools": [" "]}))).is_err()
        );
    }

    #[test]
    fn asking_for_more_than_the_parent_has_gives_nothing_extra() {
        let parent = aidaemon_core::Authority::read_only();
        let requested =
            requested_child_authority(&spawn_args(serde_json::json!({"access": "write"})))
                .unwrap()
                .unwrap();
        assert_eq!(
            parent.narrow(&requested).level,
            aidaemon_core::AccessLevel::ReadOnly
        );
    }

    /// The parent turn's authority reaches spawn_agent only through the
    /// dispatcher's typed facts; an argument of the same name is ignored.
    #[test]
    fn parent_authority_comes_from_the_call_facts() {
        let parent = crate::agent::authority::InheritedAuthority {
            authority: aidaemon_core::Authority::read_only(),
            owner_level: Some(crate::config::OwnerLevel::Full),
            task_access: None,
            mode_session: Some("tg:123".to_string()),
            untrusted_context: true,
            origin_request: Some("Summarize the synthetic notes.".to_string()),
            origin_trusted: true,
            trusted_roots: Vec::new(),
            declined: vec![aidaemon_core::EffectKey::egress(
                "https",
                "h.example.invalid",
                None,
            )],
            shared_declines: None,
            learning: None,
            spawn_family: None,
            child_declaration: None,
            parent_run: None,
            known_resources: Default::default(),
            card_title: None,
            lineage: Vec::new(),
            parent_task: None,
        };
        let mut facts = ToolCallContext::default();
        assert!(crate::agent::authority::parent_authority(&facts).is_none());
        facts.extensions.insert(parent.clone());
        assert_eq!(
            crate::agent::authority::parent_authority(&facts),
            Some(&parent)
        );
        // The model cannot supply one: SpawnArgs has no such argument.
        let json = serde_json::json!({
            "mission": "test",
            "task": "do stuff",
            "_parent_authority": parent,
        });
        assert!(serde_json::from_value::<SpawnArgs>(json).is_ok());
    }

    #[test]
    fn child_channel_context_takes_the_call_visibility() {
        let facts = ToolCallContext {
            channel_visibility: Some(ChannelVisibility::Public),
            ..Default::default()
        };
        assert_eq!(
            build_child_channel_context(&facts).visibility,
            ChannelVisibility::Public
        );
        assert_eq!(
            build_child_channel_context(&ToolCallContext::default()).visibility,
            ChannelVisibility::Internal
        );
    }

    #[test]
    fn child_channel_context_takes_the_call_trust() {
        let facts = ToolCallContext {
            trusted_session: true,
            channel_visibility: Some(ChannelVisibility::Internal),
            ..Default::default()
        };
        let channel_ctx = build_child_channel_context(&facts);
        assert_eq!(channel_ctx.visibility, ChannelVisibility::Internal);
        assert!(channel_ctx.trusted);
        assert!(!build_child_channel_context(&ToolCallContext::default()).trusted);
    }

    #[tokio::test]
    async fn executor_task_lock_deduplicates_concurrent_spawns() {
        let tool = SpawnAgentTool::new_deferred(8000, 300);
        assert!(tool.try_begin_executor_task("task-1").await);
        assert!(
            !tool.try_begin_executor_task("task-1").await,
            "Second acquire should be rejected while first is active"
        );
        tool.finish_executor_task("task-1").await;
        assert!(
            tool.try_begin_executor_task("task-1").await,
            "Task lock should be reusable after release"
        );
    }

    #[tokio::test]
    async fn background_notification_reaches_the_queue() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let state_dyn: Arc<dyn StateStore> = state.clone();

        enqueue_background_notification(
            Some(&(state_dyn.clone() as Arc<dyn crate::traits::NotificationStore>)),
            "goal_spawn_test",
            "sess_spawn_test",
            "completed",
            "Background task complete.\nMission: test",
            "spawn_test",
            false,
        )
        .await;

        let pending = state.get_pending_notifications(10).await.unwrap();
        assert!(pending.iter().any(|entry| {
            entry.goal_id == "goal_spawn_test"
                && entry.session_id == "sess_spawn_test"
                && entry.notification_type == "completed"
                && entry.message.contains("Mission: test")
        }));
    }
}

#[cfg(test)]
mod specialist_arg_tests {
    use super::*;
    use serde_json::Value;

    fn schema_props() -> Value {
        let tool = SpawnAgentTool::new_deferred(8192, 60);
        let schema = tool.schema();
        schema
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .cloned()
            .expect("schema has properties")
    }

    #[test]
    fn schema_advertises_specialist_arg_with_enum() {
        let props = schema_props();
        let specialist = props
            .get("specialist")
            .expect("specialist property declared in schema");
        let kinds = specialist
            .get("enum")
            .and_then(|v| v.as_array())
            .expect("specialist has enum array");
        let names: Vec<&str> = kinds.iter().filter_map(|v| v.as_str()).collect();
        for expected in [
            "code",
            "browser_verifier",
            "artifact_writer",
            "research",
            "review",
            "comms_draft",
            "executor",
            "generic",
        ] {
            assert!(
                names.contains(&expected),
                "missing {} from enum: {:?}",
                expected,
                names
            );
        }
        // task_lead is NOT a parent-LLM-selectable value (role-typed only).
        assert!(!names.contains(&"task_lead"));
    }

    #[test]
    fn format_specialist_arg_description_lists_every_kind_with_text() {
        let registry = crate::agent::specialists::SpecialistRegistry::load(None);
        let entries = registry.llm_visible_kinds();
        let desc = format_specialist_arg_description(&entries);

        // Each non-task_lead kind appears as a `- name:` bullet, and its
        // bundled frontmatter description text is included verbatim.
        for kind in [
            "code",
            "browser_verifier",
            "artifact_writer",
            "research",
            "review",
            "comms_draft",
            "executor",
            "generic",
        ] {
            let marker = format!("- {}:", kind);
            assert!(
                desc.contains(&marker),
                "specialist description missing bullet for {}: {}",
                kind,
                desc
            );
        }

        // task_lead must NOT appear — it's role-typed, not LLM-selectable.
        assert!(!desc.contains("- task_lead:"));
        assert!(!desc.contains("task_lead:"));

        // Spot-check the actual frontmatter descriptions surface through.
        // (Test asserts the registry's description text reached the formatter.)
        let code_desc = entries
            .iter()
            .find(|(n, _)| *n == "code")
            .map(|(_, d)| d.clone())
            .expect("code kind present");
        assert!(
            desc.contains(&code_desc),
            "code description not surfaced: {}",
            desc
        );

        // Closing line tells the model omission is allowed.
        assert!(desc.contains("Omit `specialist`"));
    }

    #[test]
    fn schema_falls_back_to_static_description_when_agent_unwired() {
        let tool = SpawnAgentTool::new_deferred(8192, 60);
        let schema = tool.schema();
        let desc = schema
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.get("specialist"))
            .and_then(|s| s.get("description"))
            .and_then(|d| d.as_str())
            .expect("specialist description string");
        assert_eq!(desc, STATIC_SPECIALIST_ARG_DESCRIPTION);
    }
}

#[cfg(test)]
mod child_runtime_tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::runtime_ports::{ChildWorkReport, SalvagedTaskOutcome};
    use crate::state::SqliteStateStore;
    use crate::traits::NotificationStore;
    use crate::traits::ToolCalls as _;
    use std::sync::Mutex as StdMutex;

    /// A child runtime that answers from a script, recording what it was asked.
    struct ScriptedChild {
        response: String,
        delay: Duration,
        tasks: StdMutex<Vec<String>>,
        failure: StdMutex<Option<anyhow::Error>>,
        cancel: tokio_util::sync::CancellationToken,
        role: StdMutex<AgentRole>,
        timed_out_tasks: StdMutex<Vec<String>>,
        panics: std::sync::atomic::AtomicBool,
        /// Fails before the child begins running.
        fails_in_setup: std::sync::atomic::AtomicBool,
        salvage_lookups: StdMutex<Vec<String>>,
        /// What each spawn declared about its child.
        declarations: StdMutex<Vec<crate::agent::authority::ChildDeclaration>>,
        /// The authority each child was started with.
        authorities: StdMutex<Vec<Option<aidaemon_core::Authority>>>,
    }

    impl ScriptedChild {
        fn new(response: &str, delay: Duration) -> Arc<Self> {
            Arc::new(Self {
                response: response.to_string(),
                delay,
                tasks: StdMutex::new(Vec::new()),
                failure: StdMutex::new(None),
                cancel: tokio_util::sync::CancellationToken::new(),
                role: StdMutex::new(AgentRole::Orchestrator),
                timed_out_tasks: StdMutex::new(Vec::new()),
                panics: std::sync::atomic::AtomicBool::new(false),
                fails_in_setup: std::sync::atomic::AtomicBool::new(false),
                declarations: StdMutex::new(Vec::new()),
                salvage_lookups: StdMutex::new(Vec::new()),
                authorities: StdMutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl ChildAgentRuntime for ScriptedChild {
        fn depth(&self) -> usize {
            0
        }
        fn max_depth(&self) -> usize {
            3
        }
        fn role(&self) -> AgentRole {
            *self.role.lock().unwrap()
        }
        fn specialist_descriptions(&self) -> Vec<(String, String)> {
            Vec::new()
        }
        async fn validate_executor_task_for_spawn(
            &self,
            _task_id: &str,
            _expected_goal_id: Option<&str>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn run_child(&self, request: ChildAgentRequest) -> anyhow::Result<ChildAgentRun> {
            if self
                .fails_in_setup
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                anyhow::bail!("synthetic setup failure");
            }
            crate::agent::authority::mark_child_began();
            self.authorities.lock().unwrap().push(
                request
                    .parent_authority
                    .as_ref()
                    .map(|parent| parent.authority.clone()),
            );
            if let Some(declaration) = request
                .parent_authority
                .as_ref()
                .and_then(|parent| parent.child_declaration.clone())
            {
                self.declarations.lock().unwrap().push(declaration);
            }
            self.tasks.lock().unwrap().push(request.task);
            tokio::time::sleep(self.delay).await;
            if self.panics.load(std::sync::atomic::Ordering::SeqCst) {
                panic!("synthetic child panic");
            }
            if let Some(failure) = self.failure.lock().unwrap().take() {
                return Err(failure);
            }
            Ok(ChildAgentRun {
                session: "sub_agent_synthetic".to_string(),
                response: self.response.clone(),
                outcome: TaskOutcome::Succeeded,
                trust: SourceTrust::Trusted,
                reported_done: false,
            })
        }
        async fn background_cancel_token(
            &self,
            _id: &str,
        ) -> Option<tokio_util::sync::CancellationToken> {
            Some(self.cancel.clone())
        }
        async fn salvage_executor_task_outcome(
            &self,
            task_id: &str,
            _timeout_secs: u64,
        ) -> Option<SalvagedTaskOutcome> {
            self.salvage_lookups
                .lock()
                .unwrap()
                .push(task_id.to_string());
            None
        }
        async fn mark_executor_task_timeout(&self, task_id: &str, _timeout_secs: u64) {
            self.timed_out_tasks
                .lock()
                .unwrap()
                .push(task_id.to_string());
        }
        async fn deliver_background_child_result(
            &self,
            _router: Option<&Weak<dyn OutboundRouter>>,
            _parent_session_id: &str,
            _text: &str,
            _trust: SourceTrust,
        ) -> anyhow::Result<bool> {
            Ok(false)
        }
    }

    fn tool_for(child: &Arc<ScriptedChild>) -> SpawnAgentTool {
        let tool = SpawnAgentTool::new_deferred(100, 30);
        let runtime: Arc<dyn ChildAgentRuntime> = child.clone();
        tool.set_agent(Arc::downgrade(&runtime));
        tool
    }

    fn facts() -> ToolCallContext {
        ToolCallContext {
            session_id: "telegram:synthetic-user-1".to_string(),
            ..Default::default()
        }
    }

    async fn notification_store() -> Arc<SqliteStateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        std::mem::forget(db_file);
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        )
    }

    /// A child's own delegations reach the chat without waiting for the queue.
    #[tokio::test]
    async fn a_childs_spawn_tool_has_the_hub() {
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store;
        let router: Arc<dyn OutboundRouter> = Arc::new(CountingHub::default());
        let tool = SpawnAgentTool::for_child(100, 30, state, Some(Arc::downgrade(&router)));
        assert!(tool.get_hub().is_some());
    }

    /// A timed-out child that already acted is not a retryable failure.
    #[tokio::test]
    async fn timeout_after_side_effects_warns_instead_of_failing() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.failure.lock().unwrap() = Some(
            ChildTimedOut {
                limit_secs: 30,
                report: ChildWorkReport {
                    changed: vec!["send_file: report.pdf".to_string()],
                    ..Default::default()
                },
            }
            .into(),
        );
        let tool = tool_for(&child);
        let args = json!({"mission": "Send", "task": "Send it"}).to_string();

        let outcome = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .expect("reported, not an error");

        assert!(outcome.output.contains("send_file: report.pdf"));
        assert!(outcome.output.contains("Do NOT run it again"));
        assert_eq!(outcome.metadata.output_trust, Some(SourceTrust::Untrusted));
    }

    #[tokio::test]
    async fn timeout_with_no_side_effects_stays_a_retryable_error() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.failure.lock().unwrap() = Some(
            ChildTimedOut {
                limit_secs: 30,
                report: ChildWorkReport::default(),
            }
            .into(),
        );
        let tool = tool_for(&child);
        let args = json!({"mission": "Read", "task": "Read it"}).to_string();

        let error = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .expect_err("nothing happened, so a retry is safe");

        assert!(error.to_string().contains("no progress for 30 seconds"));
    }

    /// A background executor that runs out of time releases its task the way
    /// a waited-on one does.
    #[tokio::test]
    async fn a_timed_out_background_executor_releases_its_task() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.role.lock().unwrap() = AgentRole::TaskLead;
        *child.failure.lock().unwrap() = Some(
            ChildTimedOut {
                limit_secs: 30,
                report: ChildWorkReport::default(),
            }
            .into(),
        );
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({
            "mission": "Execute",
            "task": "t",
            "task_id": "task-synthetic-1",
            "background": true,
        })
        .to_string();

        tool.call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        wait_for_notification(&store).await;

        assert_eq!(
            *child.timed_out_tasks.lock().unwrap(),
            vec!["task-synthetic-1".to_string()]
        );
    }

    /// A grant written with the provider's function namespace names the
    /// tool; a name the caller does not have is refused, not silently
    /// turned into a child with no tools.
    #[test]
    fn granted_tool_names_are_exact_or_refused() {
        assert_eq!(tool_name_from_grant(" functions.read_file "), "read_file");
        assert_eq!(tool_name_from_grant("search_files"), "search_files");
        let requested = requested_child_authority(
            &serde_json::from_value::<SpawnArgs>(serde_json::json!({
                "mission": "m",
                "task": "t",
                "tools": ["functions.read_file", "made_up"]
            }))
            .unwrap(),
        )
        .unwrap()
        .unwrap();
        let own = vec!["read_file".to_string(), "search_files".to_string()];
        assert_eq!(
            ungrantable_tools(&requested, &aidaemon_core::Authority::full(), Some(&own)),
            vec!["made_up".to_string()]
        );
        let narrow = aidaemon_core::Authority {
            level: aidaemon_core::AccessLevel::Write,
            tools: Some(["search_files".to_string()].into_iter().collect()),
        };
        let mut outside = ungrantable_tools(&requested, &narrow, Some(&own));
        outside.sort();
        assert_eq!(
            outside,
            vec!["made_up".to_string(), "read_file".to_string()]
        );
    }

    /// A spawn holding its child to read-only access is itself read-only, so
    /// a read-only coordinator may delegate; a writing spawn is not.
    #[test]
    fn a_read_only_spawn_is_a_read_only_call() {
        use crate::traits::ToolSemantics;
        let child = ScriptedChild::new("unused", Duration::ZERO);
        let tool = tool_for(&child);
        let roots = crate::traits::TrustRoots::new(&[]);
        let read = json!({"mission": "m", "task": "t", "access": "read_only"}).to_string();
        let write = json!({"mission": "m", "task": "t", "access": "write"}).to_string();
        let unset = json!({"mission": "m", "task": "t"}).to_string();
        assert!(tool.call_annotations_in(&read, &roots).read_only);
        assert!(!tool.call_annotations_in(&write, &roots).read_only);
        assert!(!tool.call_annotations_in(&unset, &roots).read_only);
    }

    /// A child whose run succeeded is a succeeded receipt even when its
    /// report is full of failure vocabulary: the typed outcome crosses back
    /// to the parent, not a scan of the wording.
    #[tokio::test]
    async fn a_succeeded_child_reports_its_typed_outcome_not_its_wording() {
        let child = ScriptedChild::new(
            "## Findings\n### Bugs\n- Error: the handler fails on empty input.\nFAILED checks: 2",
            Duration::ZERO,
        );
        let tool = tool_for(&child);
        let args = json!({"mission": "Review", "task": "t"}).to_string();
        let outcome = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(crate::traits::ToolOutcomeStatus::Succeeded)
        );
    }

    /// The parent's final reply gives each helper's start, end and duration
    /// from the daemon's measurement: a child's result leads with it, in the
    /// owner's 12-hour clock, so a cut of a long report never drops it.
    #[tokio::test]
    async fn a_childs_result_leads_with_its_measured_run_in_the_owners_clock() {
        let report = "x".repeat(200_000);
        let child = ScriptedChild::new(&report, Duration::ZERO);
        let tool = tool_for(&child);
        let args = json!({"mission": "Slow count", "task": "t"}).to_string();
        let outcome = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        let first = outcome.output.lines().next().unwrap_or_default();
        assert!(
            first.starts_with(&format!("{MEASURED_TIMING_MARK}Slow count took ")),
            "{first}"
        );
        assert!(first.contains(" AM ") || first.contains(" PM "), "{first}");
    }

    /// One call with `tasks` starts its children together: they overlap,
    /// every result comes back under its mission with when it ran, and the
    /// call succeeds only because each child did.
    #[tokio::test]
    async fn a_batch_call_runs_its_children_together() {
        use crate::traits::ToolSemantics;
        let child = ScriptedChild::new("done", Duration::from_millis(300));
        let tool = tool_for(&child);
        let args = json!({
            "tasks": [
                {"mission": "Count routes", "task": "a"},
                {"mission": "Count tests", "task": "b"},
                {"mission": "Count files", "task": "c"}
            ]
        })
        .to_string();
        let begun = std::time::Instant::now();
        let outcome = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        assert!(
            begun.elapsed() < Duration::from_millis(800),
            "three 300ms children took {:?}",
            begun.elapsed()
        );
        assert_eq!(child.tasks.lock().unwrap().len(), 3);
        for mission in ["Count routes", "Count tests", "Count files"] {
            assert!(outcome.output.contains(mission), "{}", outcome.output);
        }
        assert_eq!(outcome.output.matches(MEASURED_TIMING_MARK).count(), 3);
        assert_eq!(
            outcome
                .output
                .matches("These are the daemon's measurements")
                .count(),
            3
        );
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(crate::traits::ToolOutcomeStatus::Succeeded)
        );
        let read_only = json!({
            "access": "read_only",
            "tasks": [{"mission": "a", "task": "b"}, {"mission": "c", "task": "d"}]
        })
        .to_string();
        assert!(
            tool.call_annotations_in(&read_only, &crate::traits::TrustRoots::new(&[]))
                .read_only
        );
    }

    #[test]
    fn batch_entries_take_the_calls_defaults_and_override_them() {
        use crate::traits::ToolSemantics;
        let tasks = batch_tasks(
            &json!({
                "access": "read_only",
                "tasks": [
                    {"mission": "m1", "task": "t1"},
                    {"mission": "m2", "task": "t2", "access": "write"}
                ]
            })
            .to_string(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(tasks[0]["access"], "read_only");
        assert_eq!(tasks[1]["access"], "write");
        assert!(
            batch_tasks(&json!({"mission": "m", "task": "t"}).to_string())
                .unwrap()
                .is_none()
        );
        for bad in [
            json!({"tasks": []}),
            json!({"mission": "m", "tasks": [{"mission": "a", "task": "b"}]}),
            json!({"tasks": [{"mission": "a", "task": "b", "tasks": []}]}),
            json!({"tasks": [
                {"mission": "a", "task": "b", "background": true},
                {"mission": "c", "task": "d"}
            ]}),
        ] {
            assert!(batch_tasks(&bad.to_string()).is_err(), "{bad}");
        }
        // A write in any entry keeps the call's worst case.
        let tool = SpawnAgentTool::new_deferred(100, 30);
        let mixed = json!({"tasks": [
            {"mission": "a", "task": "b", "access": "read_only"},
            {"mission": "c", "task": "d"}
        ]})
        .to_string();
        assert!(
            !tool
                .call_annotations_in(&mixed, &crate::traits::TrustRoots::new(&[]))
                .read_only
        );
    }

    #[test]
    fn a_batch_with_a_failed_child_is_not_a_success_and_names_it() {
        let ok = || {
            let mut outcome = reported("fine".to_string(), SourceTrust::Trusted);
            outcome.metadata.outcome_status = Some(crate::traits::ToolOutcomeStatus::Succeeded);
            outcome
        };
        let combined = combine_batch(
            vec![
                ("first".to_string(), Ok(ok())),
                ("second".to_string(), Err(anyhow::anyhow!("boom"))),
            ],
            false,
        )
        .unwrap();
        assert_eq!(
            combined.metadata.outcome_status,
            Some(crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult)
        );
        assert_eq!(combined.metadata.output_trust, Some(SourceTrust::Untrusted));
        assert!(combined.output.contains("### second\nFailed: boom"));
        assert!(combine_batch(
            vec![("only".to_string(), Err(anyhow::anyhow!("boom")))],
            false
        )
        .is_err());
    }

    /// Starting helpers in the background answers with one plain line each,
    /// naming the helper: the run may ship it as its reply, so no bare
    /// heading ("Counter A") ever stands alone in the chat.
    #[test]
    fn a_background_start_reads_as_one_line_per_helper() {
        let combined = combine_batch(
            vec![
                (
                    "Counter A".to_string(),
                    Ok(background_spawn_outcome("Counter A")),
                ),
                (
                    "Counter B".to_string(),
                    Ok(background_spawn_outcome("Counter B")),
                ),
            ],
            true,
        )
        .unwrap();
        assert_eq!(
            combined.output,
            "Started in the background: Counter A. Its result will come to this chat when it \
             finishes.\nStarted in the background: Counter B. Its result will come to this chat \
             when it finishes."
        );
        assert!(!combined.output.contains('#'));
    }

    /// A background child that panics still clears its record, frees its
    /// task, and tells the parent session it failed.
    #[tokio::test]
    async fn a_panicking_background_child_is_cleaned_up_and_reported() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        child
            .panics
            .store(true, std::sync::atomic::Ordering::SeqCst);
        *child.role.lock().unwrap() = AgentRole::TaskLead;
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({
            "mission": "Execute",
            "task": "t",
            "task_id": "task-synthetic-2",
            "background": true,
        })
        .to_string();

        tool.call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        let entry = wait_for_notification(&store).await;
        assert_eq!(entry.notification_type, "failed");
        assert_eq!(entry.session_id, "telegram:synthetic-user-1");
        assert!(!entry.untrusted, "the failure note carries no child text");

        tokio::time::timeout(Duration::from_secs(5), async {
            while !store
                .background_spawns_for_session("telegram:synthetic-user-1")
                .await
                .unwrap()
                .is_empty()
                || tool
                    .executor_task_runs
                    .lock()
                    .await
                    .contains("task-synthetic-2")
            {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the record and the task claim are released");
    }

    fn timed_out_read_only() -> anyhow::Error {
        ChildTimedOut {
            limit_secs: 30,
            report: ChildWorkReport::default(),
        }
        .into()
    }

    fn facts_on_a_turn() -> ToolCallContext {
        ToolCallContext {
            task_id: Some("task-turn-synthetic".to_string()),
            ..facts()
        }
    }

    /// The turn's own id says nothing about a plain child: a read-only
    /// timeout stays trusted, and no durable task is looked up for it.
    #[tokio::test]
    async fn a_plain_background_childs_timeout_ignores_the_turns_task_id() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.failure.lock().unwrap() = Some(timed_out_read_only());
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({"mission": "Read", "task": "t", "background": true}).to_string();

        tool.call_with_execution_context(&args, None, facts_on_a_turn())
            .await
            .unwrap();
        let entry = wait_for_notification(&store).await;

        assert!(!entry.untrusted);
        assert!(child.salvage_lookups.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_plain_foreground_childs_timeout_looks_up_no_task() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.failure.lock().unwrap() = Some(timed_out_read_only());
        let tool = tool_for(&child);
        let args = json!({"mission": "Read", "task": "t"}).to_string();

        tool.call_with_execution_context(&args, None, facts_on_a_turn())
            .await
            .expect_err("a read-only timeout stays retryable");

        assert!(child.salvage_lookups.lock().unwrap().is_empty());
    }

    /// An executor's task is real, so its timeout is still looked up.
    #[tokio::test]
    async fn an_executors_timeout_still_looks_up_its_task() {
        let child = ScriptedChild::new("unused", Duration::ZERO);
        *child.role.lock().unwrap() = AgentRole::TaskLead;
        *child.failure.lock().unwrap() = Some(timed_out_read_only());
        let tool = tool_for(&child);
        let args =
            json!({"mission": "Execute", "task": "t", "task_id": "task-synthetic-3"}).to_string();

        tool.call_with_execution_context(&args, None, facts_on_a_turn())
            .await
            .expect_err("a read-only timeout stays retryable");

        assert_eq!(
            *child.salvage_lookups.lock().unwrap(),
            vec!["task-synthetic-3".to_string()]
        );
    }

    fn facts_in_tree(family: &crate::agent::authority::SpawnFamily) -> ToolCallContext {
        let mut call_facts = facts();
        call_facts
            .extensions
            .insert(crate::agent::authority::InheritedAuthority {
                authority: aidaemon_core::Authority::full(),
                owner_level: None,
                task_access: None,
                mode_session: None,
                untrusted_context: false,
                origin_request: None,
                origin_trusted: false,
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
            });
        call_facts
    }

    /// What the coordinator says about an overlap at spawn travels with the
    /// child as typed state, in spawn order, even when the sibling it
    /// overlaps has not joined the roster yet.
    #[tokio::test]
    async fn precedence_and_spawn_order_travel_with_the_child() {
        use crate::agent::coordination::Precedence;
        let child = ScriptedChild::new("Finished.", Duration::ZERO);
        let tool = tool_for(&child);
        let family = crate::agent::authority::SpawnFamily::new(Default::default());
        let first = json!({"mission": "m1", "task": "t", "scope": ["/w/src"]}).to_string();
        let second = json!({
            "mission": "m2", "task": "t", "scope": ["/w/src/lib.rs"],
            "precedence": "this_child"
        })
        .to_string();
        for args in [&first, &second] {
            tool.call_with_execution_context(args, None, facts_in_tree(&family))
                .await
                .unwrap();
        }
        let declared = child.declarations.lock().unwrap().clone();
        assert_eq!(declared.len(), 2);
        assert_eq!(declared[0].precedence, None);
        assert_eq!(declared[1].precedence, Some(Precedence::ThisChild));
        let order = |at: usize| declared[at].claim.as_ref().unwrap().order();
        assert!(order(0) < order(1));

        let unknown = json!({"mission": "m", "task": "t", "precedence": "mine"}).to_string();
        assert!(tool
            .call_with_execution_context(&unknown, None, facts_in_tree(&family))
            .await
            .is_err());
    }

    /// An overlap with a running sibling is stated, never refused, and the
    /// child's result ends with what the log says about it.
    #[tokio::test]
    async fn a_scope_overlap_and_the_childs_log_reach_the_spawn_result() {
        use crate::agent::coordination::{LogEntry, Member};
        let child = ScriptedChild::new("Finished.", Duration::ZERO);
        let tool = tool_for(&child);
        let family = crate::agent::authority::SpawnFamily::new(Default::default());
        family.coordination().add_member(Member {
            session: "sub_agent_sibling".to_string(),
            parent: "root".to_string(),
            mission: "Other work".to_string(),
            specialist: None,
            scope: vec!["/w/src".to_string()],
            running: true,
        });
        family.coordination().log(LogEntry::WriteOutsideScope {
            member: "sub_agent_synthetic".to_string(),
            path: "/w/elsewhere".to_string(),
        });
        let args = json!({"mission": "m", "task": "t", "scope": ["/w/src/lib.rs"]}).to_string();

        let outcome = tool
            .call_with_execution_context(&args, None, facts_in_tree(&family))
            .await
            .unwrap();

        // The measured run leads, then the child's own report.
        assert!(
            outcome.output.starts_with(MEASURED_TIMING_MARK)
                && outcome.output.contains("\n\nFinished."),
            "{}",
            outcome.output
        );
        assert!(
            outcome.output.contains("/w/elsewhere"),
            "{}",
            outcome.output
        );
        assert!(
            outcome.output.contains("sub_agent_sibling"),
            "{}",
            outcome.output
        );
    }

    /// Children past the request's running limit are refused, not queued.
    #[tokio::test]
    async fn a_request_cannot_run_more_children_than_its_limit() {
        let child = ScriptedChild::new("done", Duration::from_millis(300));
        let tool = tool_for(&child);
        let family =
            crate::agent::authority::SpawnFamily::new(crate::agent::authority::SpawnFamilyLimits {
                max_concurrent: 1,
                max_total: 10,
                ..Default::default()
            });
        let mut call_facts = facts();
        call_facts
            .extensions
            .insert(crate::agent::authority::InheritedAuthority {
                authority: aidaemon_core::Authority::read_only(),
                owner_level: None,
                task_access: None,
                mode_session: None,
                untrusted_context: false,
                origin_request: None,
                origin_trusted: true,
                trusted_roots: Vec::new(),
                declined: Vec::new(),
                shared_declines: None,
                learning: None,
                spawn_family: Some(family),
                child_declaration: None,
                parent_run: None,
                known_resources: Default::default(),
                card_title: None,
                lineage: Vec::new(),
                parent_task: None,
            });
        let args = json!({"mission": "m", "task": "t"}).to_string();

        let (first, second) = tokio::join!(
            tool.call_with_execution_context(&args, None, call_facts.clone()),
            tool.call_with_execution_context(&args, None, call_facts.clone()),
        );

        let refused = usize::from(first.is_err()) + usize::from(second.is_err());
        assert_eq!(refused, 1);
        assert_eq!(child.tasks.lock().unwrap().len(), 1);
    }

    fn family_facts(max_total: usize) -> ToolCallContext {
        let family =
            crate::agent::authority::SpawnFamily::new(crate::agent::authority::SpawnFamilyLimits {
                max_concurrent: 5,
                max_total,
                ..Default::default()
            });
        let mut call_facts = facts();
        call_facts
            .extensions
            .insert(crate::agent::authority::InheritedAuthority {
                authority: aidaemon_core::Authority::read_only(),
                owner_level: None,
                task_access: None,
                mode_session: None,
                untrusted_context: false,
                origin_request: None,
                origin_trusted: true,
                trusted_roots: Vec::new(),
                declined: Vec::new(),
                shared_declines: None,
                learning: None,
                spawn_family: Some(family),
                child_declaration: None,
                parent_run: None,
                known_resources: Default::default(),
                card_title: None,
                lineage: Vec::new(),
                parent_task: None,
            });
        call_facts
    }

    /// A spawn that fails before its child begins running does not use up
    /// the request's total; one whose child ran does.
    #[tokio::test]
    async fn a_failed_setup_does_not_use_up_the_requests_total() {
        let child = ScriptedChild::new("done", Duration::ZERO);
        let tool = tool_for(&child);
        let call_facts = family_facts(1);
        let args = json!({"mission": "m", "task": "t"}).to_string();

        child
            .fails_in_setup
            .store(true, std::sync::atomic::Ordering::SeqCst);
        tool.call_with_execution_context(&args, None, call_facts.clone())
            .await
            .expect_err("setup failed");
        child
            .fails_in_setup
            .store(false, std::sync::atomic::Ordering::SeqCst);
        tool.call_with_execution_context(&args, None, call_facts.clone())
            .await
            .expect("the failed setup left its place");
        tool.call_with_execution_context(&args, None, call_facts.clone())
            .await
            .expect_err("the child that ran used the only place");
        assert_eq!(child.tasks.lock().unwrap().len(), 1);
    }

    /// The same holds for a background spawn.
    #[tokio::test]
    async fn a_failed_background_setup_does_not_use_up_the_requests_total() {
        let child = ScriptedChild::new("done", Duration::ZERO);
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let call_facts = family_facts(1);
        let args = json!({"mission": "m", "task": "t", "background": true}).to_string();

        child
            .fails_in_setup
            .store(true, std::sync::atomic::Ordering::SeqCst);
        tool.call_with_execution_context(&args, None, call_facts.clone())
            .await
            .unwrap();
        wait_for_notification(&store).await;
        child
            .fails_in_setup
            .store(false, std::sync::atomic::Ordering::SeqCst);
        tokio::time::timeout(Duration::from_secs(5), async {
            while tool
                .call_with_execution_context(&args, None, call_facts.clone())
                .await
                .is_err()
            {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the failed setup gives its place back once it ends");
        tool.call_with_execution_context(&args, None, call_facts.clone())
            .await
            .expect_err("the child that ran used the only place");
    }

    async fn wait_for_notification(store: &SqliteStateStore) -> crate::traits::NotificationEntry {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let pending = store.get_pending_notifications(10).await.unwrap();
                if let Some(entry) = pending.into_iter().next() {
                    return entry;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("a notification arrives")
    }

    fn facts_with_authority(authority: aidaemon_core::Authority) -> ToolCallContext {
        let mut call_facts = facts();
        call_facts.user_role = Some(UserRole::Owner);
        call_facts
            .extensions
            .insert(crate::agent::authority::InheritedAuthority {
                authority,
                owner_level: None,
                task_access: None,
                mode_session: None,
                untrusted_context: false,
                origin_request: None,
                origin_trusted: true,
                trusted_roots: Vec::new(),
                declined: Vec::new(),
                shared_declines: None,
                learning: None,
                spawn_family: None,
                child_declaration: None,
                parent_run: None,
                known_resources: Default::default(),
                card_title: None,
                lineage: Vec::new(),
                parent_task: None,
            });
        call_facts
    }

    /// A child, background or not, starts with its parent turn's authority
    /// and never more: a full-access parent's child may run a command that
    /// changes things, a read-only parent's child may not, and a parent can
    /// narrow its child further with `access`.
    #[tokio::test]
    async fn a_child_runs_with_its_parent_turns_authority_foreground_or_background() {
        use aidaemon_core::{Authority, ToolAnnotations};
        let command = ToolAnnotations::WRITE;
        let cases = [
            (Authority::full(), json!({}), true),
            (Authority::read_only(), json!({}), false),
            (Authority::full(), json!({"access": "read_only"}), false),
        ];
        for background in [false, true] {
            for (parent, extra, allowed) in cases.clone() {
                let child = ScriptedChild::new("done", Duration::ZERO);
                let store = notification_store().await;
                let state: Arc<dyn StateStore> = store.clone();
                let tool = tool_for(&child).with_state(state);
                let mut args = json!({
                    "mission": "Run a command",
                    "task": "Run one shell command.",
                    "tools": ["terminal"],
                    "background": background,
                });
                for (key, value) in extra.as_object().unwrap() {
                    args[key] = value.clone();
                }
                tool.call_with_execution_context(
                    &args.to_string(),
                    None,
                    facts_with_authority(parent.clone()),
                )
                .await
                .unwrap();
                let given = tokio::time::timeout(Duration::from_secs(10), async {
                    loop {
                        if let Some(given) = child.authorities.lock().unwrap().first().cloned() {
                            return given;
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                })
                .await
                .expect("the child starts")
                .expect("the child carries its parent's authority");
                assert_eq!(
                    given.permits("terminal", command).is_ok(),
                    allowed,
                    "background={background} parent={parent:?} extra={extra} given={given:?}"
                );
                assert_eq!(given.level, parent.level.min(given.level));
            }
        }
    }

    #[tokio::test]
    async fn a_background_child_is_on_record_only_while_it_runs() {
        let child = ScriptedChild::new("done", Duration::from_millis(400));
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({"mission": "Report", "task": "t", "background": true}).to_string();

        tool.call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        let running = store
            .background_spawns_for_session("telegram:synthetic-user-1")
            .await
            .unwrap();
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].mission, "Report");

        wait_for_notification(&store).await;
        tokio::time::timeout(Duration::from_secs(5), async {
            while !store
                .background_spawns_for_session("telegram:synthetic-user-1")
                .await
                .unwrap()
                .is_empty()
            {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the record is cleared");
    }

    #[tokio::test]
    async fn the_owner_can_cancel_a_background_child() {
        let child = ScriptedChild::new("never delivered", Duration::from_secs(600));
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({"mission": "Report", "task": "t", "background": true}).to_string();

        tool.call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        child.cancel.cancel();

        let entry = wait_for_notification(&store).await;
        assert!(
            entry
                .message
                .contains("Cancelled in the background: Report"),
            "{}",
            entry.message
        );
        assert!(!entry.message.contains("never delivered"));
    }

    /// A helper's result and its chat footer give one duration, how long it
    /// took from its first step to its report, and the times it spans on
    /// the owner's clock, with seconds when under ten minutes so the times
    /// agree with the duration (live 2026-11-13: "Ran 2m 44s (8:34 PM ET to
    /// 8:37 PM ET)" was its first to last tool call, beside its own 152 s).
    #[test]
    fn a_helpers_time_is_its_first_step_to_its_report() {
        let started = chrono::Utc::now();
        let run = |first_step: Option<i64>, finished: i64, waited: u64| MeasuredRun {
            started,
            finished: started + chrono::Duration::seconds(finished),
            steps: first_step.map(|a| {
                (
                    started + chrono::Duration::seconds(a),
                    started + chrono::Duration::seconds(finished - 9),
                )
            }),
            waited: Duration::from_secs(waited),
        };
        let seconds = |at: i64| {
            (started + chrono::Duration::seconds(at))
                .with_timezone(&chrono::Local)
                .format("%-I:%M:%S %p")
                .to_string()
        };

        let footer = background_timing(&run(Some(7), 78, 0));
        assert_eq!(
            footer,
            format!(
                "{MEASURED_TIMING_MARK}Took 1m 11s ({} to {} {}).",
                seconds(7),
                seconds(78),
                crate::cron_utils::owner_zone_label(
                    chrono::Local::now().offset().local_minus_utc()
                )
            )
        );
        let window = run_window("Slow counter", &run(Some(7), 78, 0));
        assert!(
            window.starts_with(&format!(
                "{MEASURED_TIMING_MARK}Slow counter took 1m 11s (71s), {} to {} ",
                seconds(7),
                seconds(78)
            )),
            "{window}"
        );
        assert!(!window.contains("in all"), "{window}");

        // No steps: from when it was started.
        let footer = background_timing(&run(None, 78, 20));
        assert!(
            footer.starts_with(&format!(
                "{MEASURED_TIMING_MARK}Took 1m 18s ({} to ",
                seconds(0)
            )),
            "{footer}"
        );
        assert!(
            footer.ends_with(" It waited 20s of that for your answer."),
            "{footer}"
        );

        // Ten minutes or more: the owner's usual clock, no seconds.
        let long = background_timing(&run(Some(0), 900, 0));
        let owner_start = crate::cron_utils::owner_time(&started);
        assert!(
            long.contains(&format!("Took 15m ({owner_start} to ")),
            "{long}"
        );
    }

    #[tokio::test]
    async fn a_background_childs_result_says_how_long_it_took() {
        let child = ScriptedChild::new("synthetic report", Duration::from_millis(50));
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({"mission": "Report", "task": "t", "background": true}).to_string();
        tool.call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        let entry = wait_for_notification(&store).await;
        assert!(
            entry
                .message
                .contains(&format!("\n\n{MEASURED_TIMING_MARK}Took ")),
            "{}",
            entry.message
        );
    }

    #[tokio::test]
    async fn a_restart_tells_each_chat_what_it_lost() {
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        state
            .record_background_spawn(&crate::traits::BackgroundSpawn {
                id: "background_spawn:1".to_string(),
                session_id: "telegram:synthetic-user-1".to_string(),
                mission: "Publish the synthetic report".to_string(),
                started_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();

        report_interrupted_background_children(
            &(state.clone() as Arc<dyn crate::traits::NotificationStore>),
        )
        .await;

        let entry = wait_for_notification(&store).await;
        assert_eq!(entry.session_id, "telegram:synthetic-user-1");
        assert!(entry
            .message
            .contains("A restart stopped this background job"));
        assert!(entry.message.contains("Publish the synthetic report"));
        assert!(store.take_background_spawns().await.unwrap().is_empty());
    }

    /// A task that merely starts like a delay reaches the child at once and
    /// word for word: wording never decides whether to wait.
    #[tokio::test]
    async fn task_wording_never_delays_the_child() {
        let child = ScriptedChild::new("done", Duration::ZERO);
        let tool = tool_for(&child);
        let task = "In 5 hours of logs, find the errors";
        let args = json!({"mission": "Log review", "task": task}).to_string();

        tokio::time::timeout(
            Duration::from_secs(5),
            tool.call_with_execution_context(&args, None, facts()),
        )
        .await
        .expect("the child starts immediately")
        .unwrap();

        assert_eq!(*child.tasks.lock().unwrap(), vec![task.to_string()]);
    }

    #[tokio::test]
    async fn truncated_foreground_result_says_what_is_missing() {
        let child = ScriptedChild::new(&"x".repeat(500), Duration::ZERO);
        let tool = tool_for(&child);
        let args = json!({"mission": "Report", "task": "Write the report"}).to_string();

        let outcome = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .unwrap();

        let truncation = outcome.metadata.truncation.expect("reported out of band");
        assert_eq!((truncation.shown_chars, truncation.total_chars), (100, 500));
        assert!(!outcome.output.contains("OUTPUT TRUNCATED"));
    }

    #[test]
    fn truncated_background_result_says_what_is_missing() {
        let (_, message) = format_background_completion(
            "Report",
            &"x".repeat(500),
            TaskOutcome::Succeeded,
            false,
            100,
        );
        assert!(message.contains("OUTPUT TRUNCATED"));
    }

    /// Counts what reaches the chat.
    #[derive(Default)]
    struct CountingHub {
        sent: StdMutex<Vec<String>>,
    }

    #[async_trait]
    impl OutboundRouter for CountingHub {
        async fn send_text(&self, _session_id: &str, text: &str) -> anyhow::Result<()> {
            self.sent.lock().unwrap().push(text.to_string());
            Ok(())
        }
        async fn send_text_tracked(
            &self,
            _session_id: &str,
            _text: &str,
        ) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
        async fn edit_text(
            &self,
            _session_id: &str,
            _message_id: &str,
            _text: &str,
        ) -> anyhow::Result<bool> {
            Ok(false)
        }
        async fn send_media(
            &self,
            _session_id: &str,
            _media: &crate::types::MediaMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn send_media_strict(
            &self,
            _session_id: &str,
            _media: &crate::types::MediaMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn background_status_surface(&self, _session_id: &str) -> Option<String> {
            None
        }
        async fn take_background_status_surface(&self, _session_id: &str) -> Option<String> {
            None
        }
        async fn request_inline_approval(
            &self,
            _session_id: &str,
            _description: &str,
            _risk_level: crate::tools::command_risk::RiskLevel,
            _warnings: &[String],
            _permission_mode: crate::tools::command_risk::PermissionMode,
        ) -> anyhow::Result<crate::types::ApprovalResponse> {
            anyhow::bail!("no approvals in this test")
        }
        async fn register_session_route(&self, _child: &str, _parent: &str) -> bool {
            false
        }
        async fn unregister_session_route(&self, _child: &str, _parent: &str) {}
    }

    /// Records the plan card's edits.
    #[derive(Default)]
    struct CardHub {
        edits: StdMutex<Vec<(String, String)>>,
        /// No card is kept yet: the turn is still running.
        turn_running: std::sync::atomic::AtomicBool,
    }

    #[async_trait]
    impl OutboundRouter for CardHub {
        async fn send_text(&self, _session_id: &str, _text: &str) -> anyhow::Result<()> {
            Ok(())
        }
        async fn send_text_tracked(
            &self,
            _session_id: &str,
            _text: &str,
        ) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
        async fn edit_text(
            &self,
            _session_id: &str,
            message_id: &str,
            text: &str,
        ) -> anyhow::Result<bool> {
            self.edits
                .lock()
                .unwrap()
                .push((message_id.to_string(), text.to_string()));
            Ok(true)
        }
        async fn send_media(
            &self,
            _session_id: &str,
            _media: &crate::types::MediaMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn send_media_strict(
            &self,
            _session_id: &str,
            _media: &crate::types::MediaMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn background_status_surface(&self, _session_id: &str) -> Option<String> {
            None
        }
        async fn take_background_status_surface(&self, _session_id: &str) -> Option<String> {
            None
        }
        async fn checklist_surface(&self, _session_id: &str) -> Option<String> {
            (!self.turn_running.load(std::sync::atomic::Ordering::SeqCst))
                .then(|| "card-1".to_string())
        }
        async fn request_inline_approval(
            &self,
            _session_id: &str,
            _description: &str,
            _risk_level: crate::tools::command_risk::RiskLevel,
            _warnings: &[String],
            _permission_mode: crate::tools::command_risk::PermissionMode,
        ) -> anyhow::Result<crate::types::ApprovalResponse> {
            anyhow::bail!("no approvals in this test")
        }
        async fn register_session_route(&self, _child: &str, _parent: &str) -> bool {
            false
        }
        async fn unregister_session_route(&self, _child: &str, _parent: &str) {}
    }

    /// A helper's steps reach the card its turn left after that turn ended,
    /// under its start line; while the turn runs, its live card shows them
    /// and the kept card is left alone (live 2026-11-13: the card said only
    /// "Started in the background: Slow count to 30" for the whole count).
    #[tokio::test]
    async fn a_helpers_steps_reach_its_card_after_its_turn() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let plans = Arc::new(crate::plans::PlanStore::new(pool).await.unwrap());
        let chat = "telegram:synthetic-steps-chat";
        plans
            .upsert_checklist(
                chat,
                Some("task-steps"),
                "track_requirements",
                &[(
                    "Count slowly to 30".into(),
                    crate::plans::StepStatus::Pending,
                )],
            )
            .await
            .unwrap();
        plans
            .link_background_job(chat, Some("task-steps"), &[0], "job-steps")
            .await
            .unwrap()
            .expect("the step waits on the helper");
        let hub = Arc::new(CardHub::default());
        hub.turn_running
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let router: Arc<dyn OutboundRouter> = hub.clone();
        note_running_helper(chat, "job-steps", "Slow count to 30");
        let (parent_tx, mut parent_rx) = mpsc::channel(8);
        let helper = show_helper_steps_on_card(
            Some(parent_tx),
            HelperCard {
                hub: Some(Arc::downgrade(&router)),
                plans: Some(plans.clone()),
                plan_session: Some(chat.to_string()),
                chat: chat.to_string(),
                job: "job-steps".to_string(),
                label: "Slow count to 30".to_string(),
                at_start: (None, None),
                drawn: Default::default(),
            },
        )
        .unwrap();
        let step = |line: &str| StatusUpdate::SubAgentProgress {
            label: "Slow count to 30".into(),
            progress: line.into(),
        };

        helper.send(step("running a command: 14")).await.unwrap();
        assert!(matches!(
            parent_rx.recv().await,
            Some(StatusUpdate::SubAgentProgress { .. })
        ));
        tokio::time::sleep(HELPER_CARD_EVERY + Duration::from_millis(500)).await;
        assert!(
            hub.edits.lock().unwrap().is_empty(),
            "the turn's own card shows it"
        );

        hub.turn_running
            .store(false, std::sync::atomic::Ordering::SeqCst);
        helper.send(step("running a command: 15")).await.unwrap();
        let card = tokio::time::timeout(HELPER_CARD_EVERY * 3, async {
            loop {
                if let Some((id, text)) = hub.edits.lock().unwrap().last().cloned() {
                    assert_eq!(id, "card-1");
                    return text;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("the kept card shows the helper's step");
        assert!(card.contains("Count slowly to 30"), "{card}");
        assert!(
            card.contains("Started in the background: Slow count to 30\n🤖 running a command: 15"),
            "{card}"
        );
        helper_ended(chat, "job-steps");
    }

    /// Launch a background child for checklist items 1 and 2 of a three-item
    /// checklist, and return the plan card the chat ends up showing.
    async fn background_card_after(child: &Arc<ScriptedChild>) -> (String, crate::plans::TaskPlan) {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let plans = Arc::new(crate::plans::PlanStore::new(pool).await.unwrap());
        let session = "telegram:synthetic-user-1";
        plans
            .upsert_checklist(
                session,
                Some("task-synthetic"),
                "track_requirements",
                &[
                    (
                        "Inspect both sites".into(),
                        crate::plans::StepStatus::Completed,
                    ),
                    (
                        "Resume the first setup".into(),
                        crate::plans::StepStatus::Pending,
                    ),
                    (
                        "Send the screenshot".into(),
                        crate::plans::StepStatus::Pending,
                    ),
                ],
            )
            .await
            .unwrap();
        // The run's receipts proved the first step.
        plans
            .confirm_steps(
                session,
                "task-synthetic",
                &std::collections::BTreeSet::from([0]),
            )
            .await
            .unwrap();
        let hub = Arc::new(CardHub::default());
        let router: Arc<dyn OutboundRouter> = hub.clone();
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(child).with_state(state);
        tool.set_hub(Arc::downgrade(&router));
        tool.set_plan_store(plans.clone());
        let args = json!({
            "mission": "Setup",
            "task": "Resume it",
            "background": true,
            "checklist_items": [1, 2]
        })
        .to_string();
        let mut call_facts = facts();
        call_facts.task_id = Some("task-synthetic".to_string());

        tool.call_with_execution_context(&args, None, call_facts)
            .await
            .unwrap();
        let running = plans.get_recent_for_session(session, 1).await.unwrap();
        let running = running[0].render_compact_checklist();
        assert!(
            running.starts_with("⏳ Continuing in the background · 1 of 3 done"),
            "{running}"
        );
        assert!(running.contains("⏳ Send the screenshot (running in the background)"));

        let card = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Some((id, text)) = hub.edits.lock().unwrap().last().cloned() {
                    assert_eq!(id, "card-1", "the turn's card is edited in place");
                    return text;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("the card is updated when the child ends");
        let plan = plans
            .get_recent_for_session(session, 1)
            .await
            .unwrap()
            .remove(0);
        (card, plan)
    }

    /// A call that names its sub-agent by `title` and says what to do in
    /// `task`, leaving out `mission`, starts it (live 2026-11-13: both such
    /// calls failed on "missing field `mission`").
    #[tokio::test]
    async fn a_spawn_with_a_title_and_a_task_needs_no_mission() {
        let child = ScriptedChild::new("Done.", Duration::ZERO);
        let tool = tool_for(&child);
        let args = json!({
            "title": "Count slowly to 30",
            "task": "Count from 1 to 30, one number every 2 seconds.",
            "specialist": "executor"
        })
        .to_string();

        tool.call_with_execution_context(&args, None, facts())
            .await
            .expect("the child starts");

        assert_eq!(
            *child.tasks.lock().unwrap(),
            vec!["Count from 1 to 30, one number every 2 seconds.".to_string()]
        );
    }

    /// A background helper is named by its title from its start line to its
    /// result, not by its mission (live 2026-11-13: "Count slowly to 30"
    /// started and "Time a slow count and notify the user when finished"
    /// finished).
    #[tokio::test]
    async fn a_background_helper_keeps_its_title_from_start_to_finish() {
        let child = ScriptedChild::new("Done.", Duration::from_millis(50));
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let args = json!({
            "title": "Count slowly to 30",
            "mission": "Time a slow count and notify the user when finished.",
            "task": "Count to 30.",
            "background": true
        })
        .to_string();

        let started = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .unwrap();
        let entry = wait_for_notification(&store).await;

        assert!(
            started
                .output
                .starts_with("Started in the background: Count slowly to 30."),
            "{}",
            started.output
        );
        assert!(
            entry
                .message
                .contains("\u{2705} Finished in the background: Count slowly to 30\n"),
            "{}",
            entry.message
        );
        assert!(
            !entry.message.contains("Time a slow count"),
            "{}",
            entry.message
        );
    }

    /// A `spawn_agent` call that fails starts no helper: the checklist step
    /// it named reads as not running, the failure tells the model which step
    /// nothing works on, and a corrected call takes the step again.
    #[tokio::test]
    async fn a_failed_spawn_leaves_its_step_not_running_until_a_retry_starts() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let plans = Arc::new(crate::plans::PlanStore::new(pool).await.unwrap());
        let session = "telegram:synthetic-user-1";
        plans
            .upsert_checklist(
                session,
                Some("task-synthetic"),
                "track_requirements",
                &[
                    (
                        "Drupal install in site-a".into(),
                        crate::plans::StepStatus::InProgress,
                    ),
                    (
                        "WordPress install in site-b".into(),
                        crate::plans::StepStatus::Pending,
                    ),
                ],
            )
            .await
            .unwrap();
        let child = ScriptedChild::new("Done.", Duration::from_secs(5));
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        let router: Arc<dyn OutboundRouter> = Arc::new(CardHub::default());
        tool.set_hub(Arc::downgrade(&router));
        tool.set_plan_store(plans.clone());
        let mut call_facts = facts();
        call_facts.task_id = Some("task-synthetic".to_string());

        let failed =
            json!({"title": "WordPress install", "background": true, "checklist_items": [1]});
        let error = tool
            .call_with_execution_context(&failed.to_string(), None, call_facts.clone())
            .await
            .expect_err("no task, no helper");
        let error = error.to_string();
        assert!(
            error.contains("nothing is working on checklist item(s) 1"),
            "{error}"
        );
        let plan = plans
            .get_recent_for_session(session, 1)
            .await
            .unwrap()
            .remove(0);
        assert_eq!(plan.steps[1].status, crate::plans::StepStatus::Failed);
        assert!(plan.steps[1].helper_not_started);
        assert!(
            plan.render_compact_checklist()
                .contains("WordPress install in site-b (didn't run: its helper didn't start)"),
            "{}",
            plan.render_compact_checklist()
        );

        let retried = json!({
            "title": "WordPress install",
            "task": "Install WordPress in site-b.",
            "background": true,
            "checklist_items": [1]
        });
        tool.call_with_execution_context(&retried.to_string(), None, call_facts)
            .await
            .expect("the corrected call starts the helper");
        let plan = plans
            .get_recent_for_session(session, 1)
            .await
            .unwrap()
            .remove(0);
        assert_eq!(plan.steps[1].status, crate::plans::StepStatus::Background);
        assert!(!plan.steps[1].helper_not_started);
    }

    /// Records the card's edits when they land, each landing a while after
    /// it was sent, as a slow or rate-limited Telegram edit does.
    #[derive(Default)]
    struct SlowCardHub {
        landed: StdMutex<Vec<String>>,
        slow_ms: std::sync::atomic::AtomicU64,
    }

    #[async_trait]
    impl OutboundRouter for SlowCardHub {
        async fn send_text(&self, _session_id: &str, _text: &str) -> anyhow::Result<()> {
            Ok(())
        }
        async fn send_text_tracked(
            &self,
            _session_id: &str,
            _text: &str,
        ) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
        async fn edit_text(
            &self,
            _session_id: &str,
            _message_id: &str,
            text: &str,
        ) -> anyhow::Result<bool> {
            let wait = self.slow_ms.load(std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(wait)).await;
            self.landed.lock().unwrap().push(text.to_string());
            Ok(true)
        }
        async fn send_media(
            &self,
            _session_id: &str,
            _media: &crate::types::MediaMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn send_media_strict(
            &self,
            _session_id: &str,
            _media: &crate::types::MediaMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn background_status_surface(&self, _session_id: &str) -> Option<String> {
            None
        }
        async fn take_background_status_surface(&self, _session_id: &str) -> Option<String> {
            None
        }
        async fn checklist_surface(&self, _session_id: &str) -> Option<String> {
            Some("card-1".to_string())
        }
        async fn request_inline_approval(
            &self,
            _session_id: &str,
            _description: &str,
            _risk_level: crate::tools::command_risk::RiskLevel,
            _warnings: &[String],
            _permission_mode: crate::tools::command_risk::PermissionMode,
        ) -> anyhow::Result<crate::types::ApprovalResponse> {
            anyhow::bail!("no approvals in this test")
        }
        async fn register_session_route(&self, _child: &str, _parent: &str) -> bool {
            false
        }
        async fn unregister_session_route(&self, _child: &str, _parent: &str) {}
    }

    /// A step redraw still in flight when the helper ends lands before the
    /// finished card, never after it: the card ends on the finished state,
    /// within the redraw's time, not on the helper's last step (live
    /// 2026-11-13: "running a command: 13" stayed after the count ended).
    #[tokio::test]
    async fn a_helpers_finished_card_is_never_overwritten_by_its_last_step() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let plans = Arc::new(crate::plans::PlanStore::new(pool).await.unwrap());
        let chat = "telegram:synthetic-race-chat";
        plans
            .upsert_checklist(
                chat,
                Some("task-race"),
                "track_requirements",
                &[
                    (
                        "Count slowly to 30".into(),
                        crate::plans::StepStatus::Pending,
                    ),
                    ("Say done".into(), crate::plans::StepStatus::Pending),
                ],
            )
            .await
            .unwrap();
        plans
            .link_background_job(chat, Some("task-race"), &[0, 1], "job-race")
            .await
            .unwrap()
            .unwrap();
        let hub = Arc::new(SlowCardHub::default());
        hub.slow_ms.store(800, std::sync::atomic::Ordering::SeqCst);
        let router: Arc<dyn OutboundRouter> = hub.clone();
        let weak = Arc::downgrade(&router);
        note_running_helper(chat, "job-race", "Slow count");
        let (parent_tx, _parent_rx) = mpsc::channel(64);
        let helper = show_helper_steps_on_card(
            Some(parent_tx),
            HelperCard {
                hub: Some(weak.clone()),
                plans: Some(plans.clone()),
                plan_session: Some(chat.to_string()),
                chat: chat.to_string(),
                job: "job-race".to_string(),
                label: "Slow count".to_string(),
                at_start: (None, None),
                drawn: Default::default(),
            },
        )
        .unwrap();
        helper
            .send(StatusUpdate::SubAgentProgress {
                label: "Slow count".into(),
                progress: "running a command: 13".into(),
            })
            .await
            .unwrap();
        // The step's redraw is now on its way.
        tokio::time::sleep(Duration::from_millis(200)).await;
        hub.slow_ms.store(0, std::sync::atomic::Ordering::SeqCst);
        let ended = std::time::Instant::now();
        settle_background_steps(
            Some(&plans),
            Some(&weak),
            Some(chat),
            chat,
            "job-race",
            true,
        )
        .await;
        assert!(ended.elapsed() < Duration::from_secs(2));
        tokio::time::sleep(Duration::from_millis(1_000)).await;

        let landed = hub.landed.lock().unwrap().clone();
        let last = landed.last().expect("the finished card was drawn");
        assert!(last.starts_with("✅ Done · 2 of 2 steps"), "{landed:?}");
        assert!(!last.contains("running a command"), "{landed:?}");
        drop(helper);
    }

    /// A child that finishes its work moves the steps it was given to done
    /// on the card, from the typed launch, not from what anyone wrote.
    #[tokio::test]
    async fn finished_background_child_marks_its_steps_done() {
        let child = ScriptedChild::new("set up", Duration::from_millis(200));
        let (card, plan) = background_card_after(&child).await;
        assert!(card.starts_with("✅ Done · 3 of 3 steps"), "{card}");
        assert!(plan
            .steps
            .iter()
            .all(|step| step.status == crate::plans::StepStatus::Completed));
    }

    /// A child that fails marks its steps failed, never done.
    #[tokio::test]
    async fn failed_background_child_marks_its_steps_failed() {
        let child = ScriptedChild::new("unused", Duration::from_millis(200));
        *child.failure.lock().unwrap() = Some(anyhow::anyhow!("synthetic failure"));
        let (card, plan) = background_card_after(&child).await;
        assert!(card.starts_with("❌ Stopped · 1 of 3 done"), "{card}");
        assert!(
            card.contains("Resume the first setup (couldn't be done)"),
            "{card}"
        );
        assert_eq!(plan.steps[1].status, crate::plans::StepStatus::Failed);
        assert_eq!(plan.steps[2].status, crate::plans::StepStatus::Failed);
    }

    /// A background job reports when it has something to report: its result.
    #[tokio::test]
    async fn background_child_sends_no_interim_chat_messages() {
        let child = ScriptedChild::new("all done", Duration::from_millis(2_500));
        let hub = Arc::new(CountingHub::default());
        let router: Arc<dyn OutboundRouter> = hub.clone();
        let store = notification_store().await;
        let state: Arc<dyn StateStore> = store.clone();
        let tool = tool_for(&child).with_state(state);
        tool.set_hub(Arc::downgrade(&router));
        let args = json!({"mission": "Report", "task": "Write it", "background": true}).to_string();

        tool.call_with_execution_context(&args, None, facts())
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let pending = store.get_pending_notifications(10).await.unwrap();
                if pending.iter().any(|n| n.notification_type == "completed") {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("the result arrives");

        let sent = hub.sent.lock().unwrap();
        assert!(sent.is_empty(), "no interim pings: {sent:?}");
    }
}
