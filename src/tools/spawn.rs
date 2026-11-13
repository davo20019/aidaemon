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
    pub(super) use aidaemon_runtime::run_state::RunPhase;
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

    /// A synthetic ended run record.
    pub(super) fn ended_run(
        phase: RunPhase,
        unconfirmed: &[&str],
        first_step: Option<i64>,
        finished: i64,
        waited: u64,
    ) -> aidaemon_runtime::run_state::RunState {
        let start = chrono::DateTime::parse_from_rfc3339("2026-11-13T02:05:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        aidaemon_runtime::run_state::RunState {
            id: "background_spawn:synthetic".to_string(),
            chat: "telegram:synthetic-chat".to_string(),
            label: "Slow count".to_string(),
            phase,
            requested_at: None,
            started_at: start,
            ended_at: Some(start + chrono::Duration::seconds(finished)),
            work_began: first_step.map(|secs| start + chrono::Duration::seconds(secs)),
            waited: Duration::from_secs(waited),
            step: None,
            steps: Vec::new(),
            evidence: aidaemon_runtime::run_state::RunEvidence {
                unconfirmed: unconfirmed.iter().map(|s| s.to_string()).collect(),
                ..Default::default()
            },
            failure: (phase == RunPhase::Failed)
                .then_some(aidaemon_runtime::run_state::RunFailure::NoCause),
            card: Default::default(),
        }
    }

    /// The result's header is the run's phase, its one evidence line names
    /// only typed action steps nothing verified, and its footer is the measured
    /// time: what the helper wrote is only the body.
    #[test]
    fn a_runs_result_renders_from_its_record() {
        let stopped = render_run_result(
            &ended_run(RunPhase::Stopped, &[], None, 60, 0),
            "WordPress install",
            "The site is running.",
            8_000,
            false,
        );
        assert!(
            stopped.starts_with("\u{26a0}\u{fe0f} Didn't finish in the background: WordPress install\n\nThe site is running."),
            "{stopped}"
        );
        assert!(!stopped.contains("Finished in the background"));

        let succeeded = render_run_result(
            &ended_run(RunPhase::Succeeded, &[], Some(2), 62, 0),
            "Slow count",
            "Done.",
            8_000,
            false,
        );
        assert!(
            succeeded.starts_with("\u{2705} Finished in the background: Slow count\n\nDone."),
            "{succeeded}"
        );
        assert!(!succeeded.contains("confirm"), "{succeeded}");
        assert!(succeeded.contains("Took 1m"), "{succeeded}");

        let unconfirmed = render_run_result(
            &ended_run(
                RunPhase::Succeeded,
                &["Capture the screenshot"],
                None,
                60,
                0,
            ),
            "Site install",
            "Done.",
            8_000,
            false,
        );
        assert!(
            unconfirmed.starts_with(
                "\u{2705} Finished in the background: Site install\n\
                 Its report is the only record of one step: Capture the screenshot.\n\nDone."
            ),
            "{unconfirmed}"
        );
    }

    /// How a run ended is said once: when its card shows it, the result
    /// carries the body and the measured time, never the card's header
    /// again (live 2026-11-13: "Finished in the background: X" on the card
    /// and again on the result).
    #[test]
    fn the_result_never_repeats_the_cards_header() {
        for phase in [RunPhase::Succeeded, RunPhase::Stopped, RunPhase::Failed] {
            let run = ended_run(phase, &["Capture the screenshot"], Some(2), 62, 0);
            let message = render_run_result(&run, "Slow count", "Counted to 30.", 8_000, true);
            for header in [phase.headline(), phase.helper_headline(false)] {
                assert!(!message.contains(header), "{header:?} in {message:?}");
            }
            assert!(!message.contains("Slow count"), "{message}");
            // A failed run's line is why it failed, from its record; any
            // other's is what its report alone records.
            let lead = match phase {
                RunPhase::Failed => run.reason().expect("a failed run has its reason"),
                _ => {
                    "Its report is the only record of one step: Capture the screenshot.".to_string()
                }
            };
            assert!(
                message.starts_with(&format!("{lead}\n\nCounted to 30.")),
                "{message}"
            );
            assert!(message.contains("Took 1m"), "{message}");
        }
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
            card_steps: Vec::new(),
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
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
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
        /// Its tree's helper run records.
        runs: Arc<aidaemon_runtime::run_state::RunStates>,
        /// How its runs end when they return.
        end: StdMutex<aidaemon_runtime::run_state::RunEnd>,
        /// What its run records say of how they ended.
        evidence: StdMutex<aidaemon_runtime::run_state::RunEvidence>,
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
                runs: Default::default(),
                end: StdMutex::new(aidaemon_runtime::run_state::RunEnd::Succeeded),
                evidence: StdMutex::new(Default::default()),
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
                end: self.end.lock().unwrap().clone(),
                evidence: self.evidence.lock().unwrap().clone(),
            })
        }
        fn helper_runs(&self) -> Arc<aidaemon_runtime::run_state::RunStates> {
            self.runs.clone()
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
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
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

    /// What the parent says of a helper's start, end and duration quotes the
    /// daemon's measurement: a child's result leads with it, in the owner's
    /// 12-hour clock, so a cut of a long report never drops it.
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
                card_steps: Vec::new(),
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
                card_steps: Vec::new(),
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
                card_steps: Vec::new(),
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
                card_steps: Vec::new(),
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
                .contains("Didn't finish in the background: Report")
                && entry.message.contains("You cancelled it"),
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
        let record = |first_step: Option<i64>, finished: i64, waited: u64| {
            let mut state =
                super::tests::ended_run(RunPhase::Succeeded, &[], None, finished, waited);
            state.started_at = started;
            state.ended_at = Some(started + chrono::Duration::seconds(finished));
            state.work_began = first_step.map(|a| started + chrono::Duration::seconds(a));
            state
        };
        let seconds = |at: i64| {
            (started + chrono::Duration::seconds(at))
                .with_timezone(&chrono::Local)
                .format("%-I:%M:%S %p")
                .to_string()
        };

        let footer = background_timing(&record(Some(7), 78, 0)).unwrap();
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
        let footer = background_timing(&record(None, 78, 20)).unwrap();
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
        let long = background_timing(&record(Some(0), 900, 0)).unwrap();
        let owner_start = crate::cron_utils::owner_time(&started);
        assert!(
            long.contains(&format!("Took 15m ({owner_start} to ")),
            "{long}"
        );
    }

    /// Every duration the owner reads equals its run record's: the record's
    /// start and end, cut to the whole second, and their difference (live
    /// 2026-11-13: a 20.4 s run read "20 seconds" beside 2:21:16 to
    /// 2:21:37). The parent's note, the card's line and the result's
    /// footer all say 21s for that run.
    #[test]
    fn owner_facing_durations_equal_the_records() {
        use chrono::TimeZone;
        let started = chrono::Utc
            .with_ymd_and_hms(2026, 11, 13, 19, 21, 16)
            .unwrap()
            + chrono::Duration::milliseconds(800);
        let finished = started + chrono::Duration::milliseconds(20_400);
        let run = MeasuredRun {
            started,
            finished,
            steps: None,
            waited: Duration::ZERO,
        };
        let clock = |at: chrono::DateTime<chrono::Utc>| {
            at.with_timezone(&chrono::Local)
                .format("%-I:%M:%S %p")
                .to_string()
        };
        let (from, to) = (clock(started), clock(finished));
        assert!(
            from.contains(":21:16 ") && to.contains(":21:37 "),
            "{from} {to}"
        );

        let note = run_window("Count lines", &run);
        assert!(
            note.contains(&format!("Count lines took 21s (21s), {from} to {to} ")),
            "{note}"
        );
        let card = aidaemon_runtime::spawn_tool::took_line(run.span(), run.waited);
        assert!(
            card.starts_with(&format!("Took 21s ({from} to {to} ")),
            "{card}"
        );

        let mut record = super::tests::ended_run(RunPhase::Succeeded, &[], None, 0, 0);
        record.started_at = started;
        record.work_began = None;
        record.ended_at = Some(finished);
        assert_eq!(record.took(), Some(chrono::Duration::seconds(21)));
        let footer = background_timing(&record).unwrap();
        assert_eq!(footer, format!("{MEASURED_TIMING_MARK}{card}"));
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
        let message = render_run_result(
            &super::tests::ended_run(RunPhase::Succeeded, &[], None, 5, 0),
            "Report",
            &"x".repeat(500),
            100,
            false,
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
        /// A later turn kept this card after "card-1".
        later_card: StdMutex<Option<String>>,
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
        async fn checklist_surface(&self, _session_id: &str) -> Option<String> {
            if self.turn_running.load(std::sync::atomic::Ordering::SeqCst) {
                return None;
            }
            Some(
                self.later_card
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| "card-1".to_string()),
            )
        }
        async fn checklist_surface_after(
            &self,
            _session_id: &str,
            after: Option<&str>,
        ) -> Option<String> {
            if self.turn_running.load(std::sync::atomic::Ordering::SeqCst) {
                return None;
            }
            match after {
                None => Some("card-1".to_string()),
                Some("card-1") => self.later_card.lock().unwrap().clone(),
                Some(_) => None,
            }
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

    /// A no-plan helper that ends after a later turn kept its own card
    /// settles the card its starting turn kept, recorded on its run, and
    /// leaves the later turn's card alone (the newest card belonged to
    /// another turn).
    #[tokio::test]
    async fn a_helpers_end_edits_its_own_turns_card_not_the_newest() {
        let chat = "telegram:synthetic-own-card-chat";
        let hub = Arc::new(CardHub::default());
        let router: Arc<dyn OutboundRouter> = hub.clone();
        let weak = Arc::downgrade(&router);
        let runs = Arc::new(aidaemon_runtime::run_state::RunStates::default());
        runs.begin("job-own-card", chat, "Slow count", None);
        runs.card_before("job-own-card", None);
        // The starting turn kept card-1; then a later turn kept card-2.
        *hub.later_card.lock().unwrap() = Some("card-2".to_string());
        assert_eq!(
            hub.checklist_surface(chat).await.as_deref(),
            Some("card-2"),
            "the newest card is the later turn's"
        );
        runs.finish(
            "job-own-card",
            aidaemon_runtime::run_state::RunEnd::Succeeded,
            aidaemon_runtime::run_state::RunMeasure {
                ended_at: chrono::Utc::now(),
                work_began: None,
                waited: Duration::ZERO,
            },
            Default::default(),
            aidaemon_runtime::run_state::RunReport {
                text: "Done.".into(),
                untrusted: false,
            },
        );
        settle_background_steps(
            &runs,
            None,
            Some(&weak),
            None,
            chat,
            "job-own-card",
            crate::plans::JobEnd::Succeeded,
        )
        .await;
        let edits = hub.edits.lock().unwrap().clone();
        assert_eq!(edits.len(), 1, "{edits:?}");
        assert_eq!(edits[0].0, "card-1", "{edits:?}");
        assert!(edits[0].1.contains("Slow count"), "{edits:?}");
        // Nothing on its card runs any more: the card settled and the
        // record was let go.
        assert!(runs.get("job-own-card").is_none());
    }

    /// A run whose card can no longer be told (its chat's card history no
    /// longer holds the card it began beside, as after a restart) never
    /// draws itself over the card a later turn kept: that card shows that
    /// turn's own runs (live 2026-11-13: a shell loop's card was overwritten
    /// with a CLI agent's "Done 9m 40s" and stayed wrong).
    #[tokio::test]
    async fn a_run_with_an_unknown_card_leaves_a_later_turns_card_alone() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let plans = Arc::new(crate::plans::PlanStore::new(pool).await.unwrap());
        let chat = "telegram:synthetic-unknown-card-chat";
        plans
            .upsert_checklist(
                chat,
                Some("task-unknown"),
                "track_requirements",
                &[(
                    "Install WordPress".into(),
                    crate::plans::StepStatus::Pending,
                )],
            )
            .await
            .unwrap();
        plans
            .link_background_job(chat, Some("task-unknown"), &[0], "job-unknown")
            .await
            .unwrap()
            .expect("the step waits on the run");
        let hub = Arc::new(CardHub::default());
        let router: Arc<dyn OutboundRouter> = hub.clone();
        let weak = Arc::downgrade(&router);
        let runs = Arc::new(aidaemon_runtime::run_state::RunStates::default());
        runs.begin("job-unknown", chat, "Install WordPress", None);
        // It began beside a card the hub no longer remembers; a later turn
        // kept card-2 since.
        runs.card_before("job-unknown", Some("forgotten-card".to_string()));
        *hub.later_card.lock().unwrap() = Some("card-2".to_string());
        runs.finish(
            "job-unknown",
            aidaemon_runtime::run_state::RunEnd::Failed(
                aidaemon_runtime::run_state::RunFailure::NoCause,
            ),
            aidaemon_runtime::run_state::RunMeasure {
                ended_at: chrono::Utc::now(),
                work_began: None,
                waited: Duration::ZERO,
            },
            Default::default(),
            aidaemon_runtime::run_state::RunReport {
                text: "Blocked.".into(),
                untrusted: false,
            },
        );
        let shown = settle_background_steps(
            &runs,
            Some(&plans),
            Some(&weak),
            Some(chat),
            chat,
            "job-unknown",
            crate::plans::JobEnd::Failed,
        )
        .await;
        assert!(!shown, "no card shows it: its result says how it ended");
        let edits = hub.edits.lock().unwrap().clone();
        assert!(edits.is_empty(), "{edits:?}");
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
        let runs = Arc::new(aidaemon_runtime::run_state::RunStates::default());
        runs.begin("job-steps", chat, "Slow count to 30", None);
        let (parent_tx, mut parent_rx) = mpsc::channel(8);
        let helper = show_helper_steps_on_card(
            Some(parent_tx),
            HelperCard {
                runs: runs.clone(),
                hub: Some(Arc::downgrade(&router)),
                plans: Some(plans.clone()),
                plan_session: Some(chat.to_string()),
                chat: chat.to_string(),
                job: "job-steps".to_string(),
                label: "Slow count to 30".to_string(),
                at_start: None,
                drawn: Default::default(),
            },
        )
        .unwrap();
        let step = |line: &str| {
            StatusUpdate::Helper(aidaemon_core::card::HelperView::running(
                "Slow count to 30",
                true,
                Some(aidaemon_core::card::Step::Output(line.into())),
            ))
        };

        helper.send(step("14")).await.unwrap();
        assert!(matches!(
            parent_rx.recv().await,
            Some(StatusUpdate::Helper(view)) if view == runs.get("job-steps").unwrap().view()
        ));
        tokio::time::sleep(HELPER_CARD_EVERY + Duration::from_millis(500)).await;
        assert!(
            hub.edits.lock().unwrap().is_empty(),
            "the turn's own card shows it"
        );

        hub.turn_running
            .store(false, std::sync::atomic::Ordering::SeqCst);
        helper.send(step("15")).await.unwrap();
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
        assert!(
            card.contains(
                "⏳ Running in the background: Slow count to 30\nRunning a command · last output: 15"
            ),
            "{card}"
        );
        assert_eq!(
            runs.get("job-steps").and_then(|run| run.step),
            Some(aidaemon_core::card::Step::Output("15".into())),
            "/tasks reads the same step from the run record"
        );
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
        assert!(plan.steps[1].helper_not_started());
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
        assert!(!plan.steps[1].helper_not_started());
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
        let runs = Arc::new(aidaemon_runtime::run_state::RunStates::default());
        runs.begin("job-race", chat, "Slow count", None);
        let (parent_tx, _parent_rx) = mpsc::channel(64);
        let helper = show_helper_steps_on_card(
            Some(parent_tx),
            HelperCard {
                runs: runs.clone(),
                hub: Some(weak.clone()),
                plans: Some(plans.clone()),
                plan_session: Some(chat.to_string()),
                chat: chat.to_string(),
                job: "job-race".to_string(),
                label: "Slow count".to_string(),
                at_start: None,
                drawn: Default::default(),
            },
        )
        .unwrap();
        helper
            .send(StatusUpdate::Helper(
                aidaemon_core::card::HelperView::running(
                    "Slow count",
                    true,
                    Some(aidaemon_core::card::Step::Output("13".into())),
                ),
            ))
            .await
            .unwrap();
        // The step's redraw is now on its way.
        tokio::time::sleep(Duration::from_millis(200)).await;
        hub.slow_ms.store(0, std::sync::atomic::Ordering::SeqCst);
        let ended = std::time::Instant::now();
        runs.finish(
            "job-race",
            aidaemon_runtime::run_state::RunEnd::Succeeded,
            aidaemon_runtime::run_state::RunMeasure {
                ended_at: chrono::Utc::now(),
                work_began: None,
                waited: Duration::ZERO,
            },
            Default::default(),
            aidaemon_runtime::run_state::RunReport {
                text: "Done.".into(),
                untrusted: false,
            },
        );
        settle_background_steps(
            &runs,
            Some(&plans),
            Some(&weak),
            Some(chat),
            chat,
            "job-race",
            crate::plans::JobEnd::Succeeded,
        )
        .await;
        assert!(ended.elapsed() < Duration::from_secs(2));
        tokio::time::sleep(Duration::from_millis(1_000)).await;

        let landed = hub.landed.lock().unwrap().clone();
        let last = landed.last().expect("the finished card was drawn");
        assert!(last.starts_with("✅ Done · 2 of 2 steps"), "{landed:?}");
        assert!(!last.contains("Running a command"), "{landed:?}");
        assert!(
            last.ends_with("✅ Finished in the background: Slow count"),
            "{landed:?}"
        );
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

    /// What every surface shows for one background helper: the card while
    /// it runs, its run record then, and the chat's finish line and the
    /// final card once it ended.
    struct Surfaces {
        running_card: String,
        running_record: RunPhase,
        running_lines: Vec<aidaemon_core::card::HelperView>,
        message: String,
        final_card: String,
    }

    async fn surfaces_for(child: &Arc<ScriptedChild>, cancel: bool) -> Surfaces {
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
                Some("task-agree"),
                "track_requirements",
                &[
                    (
                        "Check the tools".into(),
                        crate::plans::StepStatus::Completed,
                    ),
                    ("Install the site".into(), crate::plans::StepStatus::Pending),
                ],
            )
            .await
            .unwrap();
        plans
            .confirm_steps(
                session,
                "task-agree",
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
            "title": "Site install",
            "task": "Install it",
            "background": true,
            "checklist_items": [1]
        })
        .to_string();
        let mut call_facts = facts();
        call_facts.task_id = Some("task-agree".to_string());
        tool.call_with_execution_context(&args, None, call_facts)
            .await
            .unwrap();

        let running_card =
            plans.get_recent_for_session(session, 1).await.unwrap()[0].render_compact_checklist();
        let records = child.runs.for_chat(session);
        assert_eq!(records.len(), 1, "one record per helper run");
        let running_record = records[0].phase;
        let running_lines: Vec<aidaemon_core::card::HelperView> = child
            .runs
            .for_chat(session)
            .iter()
            .map(aidaemon_runtime::run_state::RunState::view)
            .collect();
        if cancel {
            child.cancel.cancel();
        }
        let message = wait_for_notification(&store).await.message;
        let final_card = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Some((_, text)) = hub.edits.lock().unwrap().last().cloned() {
                    return text;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the card is redrawn when the run ends");
        Surfaces {
            running_card,
            running_record,
            running_lines,
            message,
            final_card,
        }
    }

    /// Running, succeeded, failed and stopped helpers: the card, the run
    /// record, the helper line, the finish header, its measured footer and
    /// the step's mark all say the same thing, from the one run record.
    #[tokio::test]
    async fn every_surface_agrees_on_how_a_helper_stands() {
        let cases: [(&str, RunPhase, bool, &str); 4] = [
            (
                "succeeded",
                RunPhase::Succeeded,
                false,
                "✅ Install the site",
            ),
            (
                "stopped short",
                RunPhase::Stopped,
                false,
                "⏭️ Install the site (not done: its helper stopped before it finished)",
            ),
            (
                "failed",
                RunPhase::Failed,
                false,
                "⚠️ Install the site (couldn't be done)",
            ),
            (
                "cancelled",
                RunPhase::Stopped,
                true,
                "⏭️ Install the site (not done: its helper stopped before it finished)",
            ),
        ];
        for (case, phase, cancel, step_line) in cases {
            let delay = if cancel {
                Duration::from_secs(30)
            } else {
                Duration::from_millis(300)
            };
            let child = ScriptedChild::new("The site is up.", delay);
            match phase {
                RunPhase::Failed => {
                    *child.failure.lock().unwrap() = Some(anyhow::anyhow!("synthetic failure"))
                }
                RunPhase::Stopped => {
                    *child.end.lock().unwrap() = aidaemon_runtime::run_state::RunEnd::Stopped(None)
                }
                _ => *child.end.lock().unwrap() = aidaemon_runtime::run_state::RunEnd::Succeeded,
            }
            let seen = surfaces_for(&child, cancel).await;

            // While it runs.
            assert_eq!(seen.running_record, RunPhase::Running, "{case}");
            assert!(
                seen.running_card
                    .starts_with("⏳ Continuing in the background · 1 of 2 done"),
                "{case}: {}",
                seen.running_card
            );
            assert!(
                seen.running_card
                    .contains("⏳ Install the site (running in the background)"),
                "{case}: {}",
                seen.running_card
            );
            assert_eq!(
                seen.running_lines,
                vec![
                    aidaemon_core::card::HelperView::running("Site install", true, None)
                        .for_steps(&[1])
                ],
                "{case}"
            );

            // Once it ended: how it ended is on the card, once; the result
            // carries the body and the measured time.
            assert!(
                seen.final_card
                    .contains(&format!("{}: Site install", phase.headline())),
                "{case}: {}",
                seen.final_card
            );
            for any in [RunPhase::Succeeded, RunPhase::Failed, RunPhase::Stopped] {
                assert!(
                    !seen.message.contains(any.headline()),
                    "{case}: {}",
                    seen.message
                );
            }
            assert!(seen.message.contains("Took "), "{case}: {}", seen.message);
            assert!(
                !seen.message.contains("confirm"),
                "{case}: {}",
                seen.message
            );
            assert!(
                seen.final_card.contains(step_line),
                "{case}: {}",
                seen.final_card
            );
            // The card settles to the helper's final header, the same one
            // its result starts with; one that failed or was cancelled says
            // why under it, from its record.
            let header = match (phase, cancel) {
                (RunPhase::Failed, _) => format!(
                    "{}: Site install\nIt stopped on an error: synthetic failure.",
                    phase.headline()
                ),
                (_, true) => format!(
                    "{}: Site install\nYou cancelled it before it finished.",
                    phase.headline()
                ),
                _ => format!("{}: Site install", phase.headline()),
            };
            assert!(
                seen.final_card.ends_with(&header),
                "{case}: {}",
                seen.final_card
            );
            assert!(
                !seen.final_card.contains("running in the background"),
                "{case}: {}",
                seen.final_card
            );
            if phase != RunPhase::Failed {
                assert!(
                    !seen.final_card.contains("couldn't be done"),
                    "{case}: only a typed failure reads as couldn't be done: {}",
                    seen.final_card
                );
            }
            assert!(
                child.runs.for_chat("telegram:synthetic-user-1").is_empty(),
                "{case}: the ended run's record goes once its result is out"
            );
        }
    }

    /// A fresh install never starts in a folder that already holds
    /// something: the call names the next free folder and no helper runs,
    /// and a folder that doesn't exist yet starts.
    #[tokio::test]
    async fn new_work_never_starts_in_a_folder_that_already_exists() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("site-3")).unwrap();
        std::fs::create_dir(root.path().join("site-4")).unwrap();
        let child = ScriptedChild::new("set up", Duration::from_millis(10));
        let tool = tool_for(&child);
        let taken = root.path().join("site-3").display().to_string();
        let args = json!({
            "title": "Site install in site-3",
            "task": "Install a fresh site",
            "new_folders": [taken],
        })
        .to_string();
        let error = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .expect_err("an existing folder is not a fresh one");
        let free = root.path().join("site-5").display().to_string();
        assert!(
            error
                .to_string()
                .contains(&format!("{taken} already exists"))
                && error
                    .to_string()
                    .contains(&format!("next free name is {free}")),
            "{error}"
        );
        assert!(child.tasks.lock().unwrap().is_empty(), "no helper ran");

        let args = json!({
            "title": "Site install in site-5",
            "task": "Install a fresh site",
            "new_folders": [free],
        })
        .to_string();
        tool.call_with_execution_context(
            &args,
            None,
            facts_with_authority(aidaemon_core::Authority::full()),
        )
        .await
        .expect("a folder that doesn't exist yet is fine");
        assert_eq!(child.tasks.lock().unwrap().len(), 1);

        // The spawn made the folder, empty, and told the child it is there
        // to work in.
        let made = std::path::Path::new(&free);
        assert!(made.is_dir(), "the new folder exists before the child runs");
        assert_eq!(std::fs::read_dir(made).unwrap().count(), 0);
        assert!(
            child.tasks.lock().unwrap()[0].contains(&format!("{free}, was just created")),
            "{:?}",
            child.tasks.lock().unwrap()
        );
        // Its write authority is derived from the spec, in one place: the
        // folder made for it is inside, anything else is outside.
        let declaration = child.declarations.lock().unwrap()[0].clone();
        assert_eq!(declaration.new_folders, std::slice::from_ref(&free));
        let parent = crate::agent::authority::InheritedAuthority {
            child_declaration: Some(declaration),
            ..crate::agent::authority::parent_authority(&facts_with_authority(
                aidaemon_core::Authority::full(),
            ))
            .unwrap()
            .clone()
        };
        let child_access = crate::agent::authority::narrow_for_child(
            &parent,
            crate::agent::authority::ResolvedAuthority {
                authority: aidaemon_core::Authority::full(),
                owner_level: Some(crate::config::OwnerLevel::Full),
                task_access: None,
            },
        )
        .task_access
        .expect("a child with a declared folder is confined to it");
        let writes = |path: &str| crate::traits::ToolCallAccessManifest {
            write_targets: vec![crate::traits::ToolTargetHint::new(
                crate::traits::ToolTargetHintKind::Path,
                path.to_string(),
            )
            .unwrap()],
            ..Default::default()
        };
        let check = |path: &str| {
            aidaemon_runtime::turn::tool_dispatch::access_manifest_scope_check(
                "write_file",
                &writes(path),
                Some(&child_access),
                &[],
                None,
                &crate::tools::DaemonPaths::default(),
            )
        };
        assert!(check(&format!("{free}/wp-config.php")).is_none());
        assert!(check(&format!("{free}/.ddev/config.yaml")).is_none());
        let taken_file = format!("{taken}/index.php");
        assert!(
            check(&taken_file).is_some(),
            "the earlier site next to it is outside the child's scope"
        );
    }

    /// A spawn that cannot start gives back the folders it made, so the
    /// name is free again.
    #[tokio::test]
    async fn a_spawn_that_fails_to_start_gives_its_new_folder_back() {
        let root = tempfile::tempdir().unwrap();
        let child = ScriptedChild::new("set up", Duration::from_millis(10));
        child
            .fails_in_setup
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let tool = tool_for(&child);
        let folder = root.path().join("site").display().to_string();
        let args = json!({
            "title": "Site install",
            "task": "Install a fresh site",
            "new_folders": [folder],
        })
        .to_string();
        let _ = tool.call_with_execution_context(&args, None, facts()).await;
        assert!(!std::path::Path::new(&folder).exists());
    }

    /// A helper whose run record says it did none of its work fails, with
    /// the typed reason, whatever its reply says: the parent gets the
    /// failure and the measured time worded as how long it ran before
    /// failing, never as a time to finish.
    #[tokio::test]
    async fn a_helper_that_did_nothing_fails_with_its_typed_reason() {
        let child = ScriptedChild::new(
            "**Blocked:** the folder does not exist here.",
            Duration::from_millis(10),
        );
        *child.end.lock().unwrap() = aidaemon_runtime::run_state::RunEnd::Failed(
            aidaemon_runtime::run_state::RunFailure::NothingDone {
                undone: vec!["Install WordPress with DDEV".to_string()],
            },
        );
        let tool = tool_for(&child);
        let args = json!({
            "title": "WordPress install",
            "task": "Install WordPress",
        })
        .to_string();
        let error = tool
            .call_with_execution_context(&args, None, facts())
            .await
            .expect_err("a helper that did nothing failed")
            .to_string();
        assert!(error.contains("Sub-agent failed."), "{error}");
        assert!(
            error.contains("It ended without doing any of its work")
                && error.contains("Install WordPress with DDEV"),
            "the typed reason reaches the parent: {error}"
        );
        assert!(
            error.contains("and failed.") && error.contains("not how long its work takes"),
            "{error}"
        );

        // In a batch beside a helper that succeeded, the failure is that
        // helper's section and the call is not a success.
        let ok = Ok(reported("done".to_string(), SourceTrust::Trusted));
        let combined = combine_batch(
            vec![
                (
                    "Drupal install".to_string(),
                    ok.map(|mut outcome| {
                        outcome.metadata.outcome_status =
                            Some(aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded);
                        outcome
                    }),
                ),
                (
                    "WordPress install".to_string(),
                    Err(anyhow::anyhow!(error.clone())),
                ),
            ],
            false,
        )
        .unwrap();
        assert!(combined
            .output
            .contains("### WordPress install\nFailed: Sub-agent failed."));
        assert_ne!(
            combined.metadata.outcome_status,
            Some(aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded)
        );
    }
}
