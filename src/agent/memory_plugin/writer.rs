//! [`DaemonMemoryWriter`]: post-turn memory and learning as a run-end hook
//! (`aidaemon_runtime::turn::memory_writer`).

pub(crate) use aidaemon_runtime::turn::memory_writer::{
    DaemonMemoryWriter, MemoryWrites, WriterSettings,
};

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
        /// The run start each extraction was given.
        pub(crate) extraction_windows: Vec<chrono::DateTime<chrono::Utc>>,
        pub(crate) summarized: usize,
        pub(crate) learned: Vec<LearningContext>,
    }

    #[derive(Default)]
    pub(crate) struct RecordingWrites(pub(crate) Mutex<Recorded>);

    impl MemoryWrites for RecordingWrites {
        fn extract_facts(
            &self,
            _turn: &MemoryTurn,
            reply: &str,
            run_started_at: chrono::DateTime<chrono::Utc>,
        ) {
            let mut recorded = self.0.lock().unwrap();
            recorded.extracted.push(reply.to_string());
            recorded.extraction_windows.push(run_started_at);
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
    async fn extraction_window_opens_before_the_run_and_closes_after_it() {
        let built_after = chrono::Utc::now();
        let turn = turn(0, "telegram:synthetic-user-1").await;
        let writes = run(
            ScriptedProvider::new([text("first answer"), text("Saved: teal.")]),
            turn,
            false,
        )
        .await;
        let ended_before = chrono::Utc::now();
        let recorded = writes.0.lock().unwrap();
        assert_eq!(recorded.extraction_windows.len(), 1);
        let started = recorded.extraction_windows[0];
        assert!(built_after <= started && started <= ended_before);
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

    /// "Don't remember this": an opted-out turn is neither extracted nor
    /// learned from. The summary still runs; it leaves the turn out itself.
    #[tokio::test]
    async fn an_opted_out_turn_is_not_learned_from() {
        let turn = turn(0, "telegram:synthetic-user-1").await;
        turn.learning_consent.opt_out();
        let writes = run(
            ScriptedProvider::new([text("first answer"), text("Saved: teal.")]),
            turn,
            false,
        )
        .await;
        let recorded = writes.0.lock().unwrap();
        assert!(recorded.extracted.is_empty());
        assert!(recorded.learned.is_empty());
        assert_eq!(recorded.summarized, 1);
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

    /// A run a tool finished with work done is a completed run for both
    /// memory writers: the daemon's summarizes and learns, the memory
    /// plugin's extracts facts.
    #[tokio::test]
    async fn finished_run_is_written_by_both_memory_writers() {
        use aidaemon_core::testing::{call, spec, tool_calls};
        use aidaemon_memory::{FactExtractor, HashEmbedder, InMemoryMemoryStore, Memory};
        for finish in [
            sdk::Finish::succeeded("Connected the synthetic service."),
            sdk::Finish::partial("Started; the rest runs in the background."),
        ] {
            let writes = Arc::new(RecordingWrites::default());
            let extraction = ScriptedProvider::new([text(r#"{"facts": []}"#)]);
            let memory = Memory::builder(
                Arc::new(InMemoryMemoryStore::new()),
                Arc::new(HashEmbedder::default()),
            )
            .extractor(FactExtractor::new(extraction, "small"))
            .build();
            let plugin_writer = Arc::new(aidaemon_memory::MemoryWriter::new(memory));
            let finisher = sdk::FnTool::new(
                spec("finisher", "finishes", &[]),
                sdk::ToolAnnotations::READ_ONLY,
                move |_: serde_json::Value| {
                    std::future::ready(Ok(sdk::ToolOutput::text("done").finish(finish.clone())))
                },
            );
            let agent = sdk::Agent::builder("test")
                .allow_unguarded()
                .model("scripted")
                .provider(ScriptedProvider::new([tool_calls([call(
                    "c1",
                    "finisher",
                    serde_json::json!({}),
                )])]))
                .tool(finisher)
                .hook(DaemonMemoryWriter::new(
                    turn(0, "telegram:synthetic-user-1").await,
                    SETTINGS,
                    writes.clone(),
                ))
                .hook_arc(plugin_writer.clone())
                .build()
                .unwrap();
            let outcome = agent.run("s", "go").await.unwrap();
            assert!(matches!(outcome.status, sdk::RunStatus::Finished { .. }));

            {
                let recorded = writes.0.lock().unwrap();
                assert_eq!(recorded.summarized, 1, "the daemon writer summarizes");
                assert_eq!(recorded.learned.len(), 1);
                assert!(recorded.learned[0].completed_naturally);
            }
            assert_eq!(
                plugin_writer.drain().await.len(),
                1,
                "the memory plugin's writer ingests the turn"
            );
        }
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

        let mut harness =
            crate::testing::setup_test_agent_root(MockProvider::with_responses(vec![
                MockProvider::tool_call_response(
                    "remember_fact",
                    r#"{"category":"preference","key":"favorite_color","value":"teal"}"#,
                ),
                MockProvider::text_response("Saved."),
            ]))
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
