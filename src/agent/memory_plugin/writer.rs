//! [`DaemonMemoryWriter`]: post-turn memory and learning as a run-end hook.

use std::sync::Arc;

use aidaemon_core as sdk;
use async_trait::async_trait;

use super::MemoryTurn;
use crate::agent::post_task::LearningContext;
use crate::events::TaskStatus;

/// The daemon's memory side effects, behind a seam so the writer's decisions
/// are testable without spawning extraction.
pub(crate) trait MemoryWrites: Send + Sync {
    /// Progressive fact extraction from the user text and the final reply.
    fn extract_facts(&self, turn: &MemoryTurn, reply: &str);
    /// Incremental conversation summarization.
    fn summarize(&self, turn: &MemoryTurn);
    /// Post-task learning (procedures, expertise, failure patterns).
    fn learn(&self, learning: LearningContext);
}

/// Which writes this writer owns, and the context-window switches its plan
/// depends on. The memory plugin owns fact extraction; the turn owns
/// summarization and learning.
#[derive(Clone, Copy)]
pub(crate) struct WriterSettings {
    pub progressive_facts: bool,
    pub summarization: bool,
    pub learning: bool,
}

/// Writes memory once per run, after the outcome is final: extraction and
/// summarization only for a completed turn, learning for every run.
pub(crate) struct DaemonMemoryWriter {
    turn: MemoryTurn,
    settings: WriterSettings,
    writes: Arc<dyn MemoryWrites>,
}

impl DaemonMemoryWriter {
    pub(crate) fn new(
        turn: MemoryTurn,
        settings: WriterSettings,
        writes: Arc<dyn MemoryWrites>,
    ) -> Self {
        Self {
            turn,
            settings,
            writes,
        }
    }
}

#[async_trait]
impl sdk::Hook for DaemonMemoryWriter {
    async fn on_run_end(&self, _ctx: &sdk::HookContext<'_>, outcome: &sdk::RunOutcome) {
        let turn = &self.turn;
        let mut learning = turn
            .learning
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let (reply, status, task_outcome) =
            turn.verdict
                .final_reply(outcome, &learning.tool_calls, turn.channel_ctx.visibility);
        if status == TaskStatus::Completed {
            let (extract, summarize) = crate::agent::post_turn::post_completion_memory_plan(
                turn.mandate,
                turn.memory_isolated(),
                turn.memory_persistence_allowed,
                self.settings.progressive_facts,
                crate::memory::context_window::should_extract_facts(&turn.user_text),
                self.settings.summarization,
            );
            if extract {
                let reply = crate::tools::sanitize::sanitize_user_facing_reply(&reply);
                self.writes.extract_facts(turn, &reply);
            }
            if summarize {
                self.writes.summarize(turn);
            }
        }
        if turn.mandate || !self.settings.learning {
            return;
        }
        learning.completed_naturally = status == TaskStatus::Completed;
        learning.task_outcome = Some(task_outcome);
        self.writes.learn(learning);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{Arc, Mutex};

    use aidaemon_core as sdk;
    use aidaemon_core::testing::{text, ScriptedProvider};
    use async_trait::async_trait;

    use super::{DaemonMemoryWriter, MemoryWrites, WriterSettings};
    use crate::agent::memory_plugin::MemoryTurn;
    use crate::agent::post_task::LearningContext;
    use crate::types::{ChannelContext, ChannelVisibility, UserRole};

    #[derive(Default)]
    pub(crate) struct Recorded {
        pub(crate) extracted: Vec<String>,
        pub(crate) summarized: usize,
        pub(crate) learned: Vec<LearningContext>,
    }

    #[derive(Default)]
    pub(crate) struct RecordingWrites(pub(crate) Mutex<Recorded>);

    impl MemoryWrites for RecordingWrites {
        fn extract_facts(&self, _turn: &MemoryTurn, reply: &str) {
            self.0.lock().unwrap().extracted.push(reply.to_string());
        }
        fn summarize(&self, _turn: &MemoryTurn) {
            self.0.lock().unwrap().summarized += 1;
        }
        fn learn(&self, learning: LearningContext) {
            self.0.lock().unwrap().learned.push(learning);
        }
    }

    /// Sends the first final answer back for another pass.
    #[derive(Default)]
    struct ContinueOnce(Mutex<bool>);

    #[async_trait]
    impl sdk::Hook for ContinueOnce {
        async fn on_stop(
            &self,
            _ctx: &sdk::HookContext<'_>,
            _final_text: &str,
        ) -> sdk::StopDecision {
            let mut done = self.0.lock().unwrap();
            if *done {
                return sdk::StopDecision::Accept;
            }
            *done = true;
            sdk::StopDecision::Continue {
                message: "check once more".to_string(),
            }
        }
    }

    async fn turn(depth: usize, session_id: &str) -> MemoryTurn {
        let turn = crate::agent::memory_plugin::test_support::turn(
            session_id,
            "remember that my favorite color is teal",
        )
        .await;
        turn.learning
            .lock()
            .unwrap()
            .tool_calls
            .push("remember_fact(favorite_color)".to_string());
        MemoryTurn { depth, ..turn }
    }

    const SETTINGS: WriterSettings = WriterSettings {
        progressive_facts: true,
        summarization: true,
        learning: true,
    };

    async fn run(
        provider: ScriptedProvider,
        turn: MemoryTurn,
        cancel: bool,
    ) -> Arc<RecordingWrites> {
        let writes = Arc::new(RecordingWrites::default());
        let agent = sdk::Agent::builder("test")
            .allow_unguarded()
            .model("scripted")
            .provider(provider)
            .hook(ContinueOnce::default())
            .hook(DaemonMemoryWriter::new(turn, SETTINGS, writes.clone()))
            .build()
            .unwrap();
        let token = tokio_util::sync::CancellationToken::new();
        if cancel {
            token.cancel();
        }
        agent.run_with_cancel("s", "go", token).await.unwrap();
        writes
    }

    #[tokio::test]
    async fn writer_extracts_once_per_completed_turn() {
        let writes = run(
            ScriptedProvider::new([text("first answer"), text("Saved: teal.")]),
            turn(0, "telegram:synthetic-user-1").await,
            false,
        )
        .await;
        let recorded = writes.0.lock().unwrap();
        assert_eq!(recorded.extracted, vec!["Saved: teal.".to_string()]);
        assert_eq!(recorded.summarized, 1);
        assert_eq!(recorded.learned.len(), 1);
        assert!(recorded.learned[0].completed_naturally);
        assert_eq!(
            recorded.learned[0].task_outcome,
            Some(crate::events::TaskOutcome::Succeeded)
        );
    }

    struct Approve;
    #[async_trait]
    impl sdk::InterruptResolver for Approve {
        async fn resolve(
            &self,
            _: &sdk::RunState,
            _: &sdk::Interruption,
        ) -> Option<sdk::Resolution> {
            Some(sdk::Resolution::Approve)
        }
    }

    /// A turn whose egress after an untrusted read was verified and approved
    /// learns once, from the final outcome, not once per interruption.
    #[tokio::test]
    async fn a_verified_egress_turn_learns_once() {
        use aidaemon_core::testing::{call, spec, tool_calls};
        let writes = Arc::new(RecordingWrites::default());
        let fetch = sdk::FnTool::new(
            spec("fetch", "fetches", &[]),
            sdk::ToolAnnotations::READ_ONLY,
            |_: serde_json::Value| std::future::ready(Ok(sdk::ToolOutput::text("page"))),
        );
        let send = sdk::FnTool::new(
            spec("send", "sends", &[]),
            sdk::ToolAnnotations {
                egress: true,
                ..sdk::ToolAnnotations::WRITE
            },
            |_: serde_json::Value| std::future::ready(Ok(sdk::ToolOutput::text("sent"))),
        )
        .with_trust(sdk::SourceTrust::Trusted);
        let agent = sdk::Agent::builder("test")
            .allow_unguarded()
            .model("scripted")
            .provider(ScriptedProvider::new([
                tool_calls([call("c1", "fetch", serde_json::json!({}))]),
                tool_calls([call("c2", "send", serde_json::json!({}))]),
                text("Sent."),
            ]))
            .tool(fetch)
            .tool(send)
            .hook(sdk::UntrustedHook::new(
                sdk::UntrustedMode::Enforce,
                |_, content| content.to_string(),
            ))
            .hook(DaemonMemoryWriter::new(
                turn(0, "telegram:synthetic-user-1").await,
                SETTINGS,
                writes.clone(),
            ))
            .build()
            .unwrap();
        let outcome = agent
            .run_resolving(
                sdk::RunState::new("test", "s", "go"),
                tokio_util::sync::CancellationToken::new(),
                &Approve,
            )
            .await
            .unwrap();
        assert!(outcome.is_completed());
        let recorded = writes.0.lock().unwrap();
        assert_eq!(recorded.learned.len(), 1, "one run end per turn");
        assert!(recorded.learned[0].completed_naturally);
    }

    #[tokio::test]
    async fn writer_skips_failed_and_cancelled_runs() {
        let failed = run(
            ScriptedProvider::with_results([Err(sdk::ProviderError::Auth("bad key".into()))]),
            turn(0, "telegram:synthetic-user-1").await,
            false,
        )
        .await;
        let cancelled = run(
            ScriptedProvider::new([text("never")]),
            turn(0, "telegram:synthetic-user-1").await,
            true,
        )
        .await;
        for writes in [failed, cancelled] {
            let recorded = writes.0.lock().unwrap();
            assert!(recorded.extracted.is_empty());
            assert_eq!(recorded.summarized, 0);
            assert_eq!(recorded.learned.len(), 1);
            assert!(!recorded.learned[0].completed_naturally);
        }
    }

    #[tokio::test]
    async fn writer_sanitizes_reply_before_extraction() {
        let writes = run(
            ScriptedProvider::new([text("first"), text("Saved teal. [prior turn]")]),
            turn(0, "telegram:synthetic-user-1").await,
            false,
        )
        .await;
        let recorded = writes.0.lock().unwrap();
        assert_eq!(recorded.extracted.len(), 1);
        assert!(!recorded.extracted[0].contains("[prior turn]"));
    }

    #[tokio::test]
    async fn memory_isolated_session_summarizes_but_does_not_extract() {
        let writes = run(
            ScriptedProvider::new([text("first"), text("Saved: teal.")]),
            turn(1, "telegram:synthetic-user-1").await,
            false,
        )
        .await;
        let recorded = writes.0.lock().unwrap();
        assert!(recorded.extracted.is_empty());
        assert_eq!(recorded.summarized, 1);
    }

    /// The writer extracts the reply the user got: secrets redacted on a
    /// public channel, and the interrupted answer when the model's last
    /// reply was empty.
    #[tokio::test]
    async fn writer_extracts_the_finalized_reply() {
        let mut public = turn(0, "slack:synthetic-public").await;
        public.channel_ctx.visibility = ChannelVisibility::Public;
        let writes = run(
            ScriptedProvider::new([
                text("first"),
                text("Key: sk-abc123456789012345678901234567890"),
            ]),
            public,
            false,
        )
        .await;
        {
            let recorded = writes.0.lock().unwrap();
            assert_eq!(recorded.extracted.len(), 1);
            assert!(
                !recorded.extracted[0].contains("sk-abc"),
                "{:?}",
                recorded.extracted
            );
        }

        let interrupted = turn(0, "telegram:synthetic-user-1").await;
        interrupted
            .verdict
            .record_interrupted_answer("Teal it is.".to_string());
        let writes = run(
            ScriptedProvider::new([text("first"), text("")]),
            interrupted,
            false,
        )
        .await;
        assert_eq!(
            writes.0.lock().unwrap().extracted,
            vec!["Teal it is.".to_string()]
        );
    }

    /// A plugin whose writer records into `writes`. The harness is a root
    /// agent: delegated agents are memory-isolated and never extract.
    struct RecordingPlugin(Arc<RecordingWrites>);

    #[async_trait]
    impl crate::agent::memory_plugin::MemoryPlugin for RecordingPlugin {
        async fn turn(&self, turn: &MemoryTurn) -> crate::agent::memory_plugin::MemoryParts {
            crate::agent::memory_plugin::MemoryParts {
                hooks: vec![Arc::new(DaemonMemoryWriter::new(
                    turn.clone(),
                    SETTINGS,
                    self.0.clone(),
                ))],
                ..Default::default()
            }
        }

        fn owned_daemon_tools(&self) -> &'static [&'static str] {
            crate::agent::memory_plugin::DAEMON_MEMORY_TOOLS
        }
    }

    #[tokio::test]
    async fn learning_records_run_tool_calls() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;

        let mut harness = crate::testing::setup_test_agent_root(
            MockProvider::with_responses(vec![
                MockProvider::tool_call_response(
                    "remember_fact",
                    r#"{"category":"preference","key":"favorite_color","value":"teal"}"#,
                ),
                MockProvider::text_response("Saved."),
            ])
            .with_task_assessments(vec![MockProvider::semantic_task_assessment(
                "answer",
                false,
                false,
                &[],
                "new_request",
                "general",
            )]),
        )
        .await
        .unwrap();
        let writes = Arc::new(RecordingWrites::default());
        harness
            .agent
            .set_memory_plugin(Arc::new(RecordingPlugin(writes.clone())));
        harness
            .agent
            .handle_inbound_message(crate::runtime_ports::InboundMessageRequest {
                session_id: "telegram:synthetic-user-1".to_string(),
                user_text: "By the way, my favorite color is teal and I work at Acme Corp."
                    .to_string(),
                attachments: Vec::new(),
                status_tx: None,
                user_role: UserRole::Owner,
                channel_ctx: ChannelContext::private("telegram"),
                heartbeat: None,
                ingress_timing: None,
            })
            .await
            .unwrap();
        let recorded = writes.0.lock().unwrap();
        assert_eq!(recorded.learned.len(), 1);
        assert!(
            recorded.learned[0]
                .tool_calls
                .iter()
                .any(|call| call.starts_with("remember_fact")),
            "{:?}",
            recorded.learned[0].tool_calls
        );
        assert_eq!(
            recorded.extracted,
            vec!["Saved.".to_string()],
            "persist={} completed={} outcome={:?}",
            recorded.learned[0].memory_persistence_allowed,
            recorded.learned[0].completed_naturally,
            recorded.learned[0].task_outcome
        );
    }
}
