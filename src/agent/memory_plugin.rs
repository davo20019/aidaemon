//! Long-term memory as a plugin on SDK turns.
//!
//! The agent reaches memory only through [`MemoryPlugin`]: each turn the
//! active plugin returns the hooks (recall injection, run-end writing) and
//! tools it wants, plus the daemon memory tools it owns. Swapping the plugin
//! swaps the memory backend without touching the turn code.
//!
//! Any SDK `aidaemon_policies::MemorySystem` is a plugin through
//! [`SystemMemory`]: the daemon decides the turn's scope and permissions,
//! and the system decides how to remember. `aidaemon-memory` is the
//! built-in system; [`DaemonMemory`] is the daemon's own store.

use std::sync::Arc;

#[cfg(test)]
use aidaemon_core as sdk;
use async_trait::async_trait;

use crate::traits::StateStore;
#[cfg(test)]
use crate::types::{ChannelContext, UserRole};

/// Opting a turn out of learning lives in `aidaemon-runtime`.
mod opt_out {
    pub(crate) use aidaemon_runtime::turn::opt_out::*;
}
#[cfg(test)]
mod opt_out_tests;
mod recall;
#[cfg(test)]
mod swap_tests;
mod system_memory;
/// What the turn knows about memory tools lives in `aidaemon-runtime`.
mod tool_facts {
    #[cfg(test)]
    pub(in crate::agent) use aidaemon_runtime::turn::tool_facts::*;
}
mod writer;
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use opt_out::{
    mark_turn_opted_out, set_next_message_private, take_next_message_private,
};
pub(crate) use recall::DaemonRecall;
pub(crate) use system_memory::SystemMemory;
#[cfg(test)]
pub(in crate::agent) use tool_facts::memory_tool_facts;
pub(crate) use writer::{DaemonMemoryWriter, MemoryWrites, WriterSettings};

#[cfg(test)]
pub(crate) use aidaemon_runtime::turn::memory::MANAGE_MEMORIES_GOAL_ACTIONS;
pub(crate) use aidaemon_runtime::turn::memory::{
    DAEMON_MEMORY_TOOLS, MemoryParts, MemoryPlugin, MemoryTurn,
};
#[cfg(test)]
pub(crate) use aidaemon_runtime::turn::memory::{
    DaemonToolOffer, daemon_tool_offer, goal_actions_view,
};
#[cfg(test)]
pub(crate) use aidaemon_runtime::turn::memory::{check_goal_action, memory_parts};

/// The memory system this daemon runs with.
pub(crate) struct ConfiguredMemory {
    /// What serves turns; `None` for the daemon's own memory.
    pub plugin: Option<Arc<dyn MemoryPlugin>>,
    /// Background work of a replacing system.
    pub services: Vec<Arc<dyn aidaemon_core::Service>>,
    /// What the base system prompt says about memory.
    pub prompt: crate::startup::prompt::MemoryPrompt,
}

impl ConfiguredMemory {
    fn daemons_own() -> Self {
        Self {
            plugin: None,
            services: Vec::new(),
            prompt: crate::startup::prompt::MemoryPrompt::DaemonsOwn,
        }
    }

    #[cfg_attr(not(any(test, feature = "sdk-memory")), allow(dead_code))]
    pub(crate) fn replaced_by(system: Arc<dyn aidaemon_policies::MemorySystem>) -> Self {
        Self {
            services: system.services(),
            prompt: crate::startup::prompt::MemoryPrompt::Replaced(system.guidance()),
            plugin: Some(Arc::new(SystemMemory::new(system))),
        }
    }

    /// Whether the daemon's own background memory jobs should run.
    pub fn own_memory(&self) -> bool {
        self.plugin.is_none()
    }
}

/// The memory plugin `[memory]` selects, or `None` for the daemon's own.
///
/// A backend that cannot start is logged and the daemon's own memory is
/// used, so a broken memory store never keeps the daemon down.
pub(crate) async fn configured_memory(
    config: &crate::config::MemoryConfig,
    llm_runtime: &crate::llm_runtime::SharedLlmRuntime,
    side_calls: &crate::llm_runtime::gateway::SideCallGateway,
    state: Arc<dyn StateStore>,
    pool: sqlx::SqlitePool,
) -> ConfiguredMemory {
    match config.backend {
        crate::config::MemoryBackend::Daemon => ConfiguredMemory::daemons_own(),
        #[cfg(feature = "sdk-memory")]
        crate::config::MemoryBackend::Sdk => {
            let store = match aidaemon_memory::SqliteMemoryStore::open(&config.path).await {
                Ok(store) => store,
                Err(error) => {
                    tracing::error!(
                        path = %config.path,
                        %error,
                        "[memory] backend = \"sdk\" could not open its store; \
                         using the daemon's memory"
                    );
                    return ConfiguredMemory::daemons_own();
                }
            };
            // The side-call gateway picks the model; this value is not sent.
            let fast_model = llm_runtime
                .router()
                .map(|router| router.select(crate::router::Tier::Fast).to_string())
                .unwrap_or_else(|| llm_runtime.snapshot().primary_model());
            let extractor = aidaemon_memory::FactExtractor::from_arc(
                side_calls.provider(crate::llm_runtime::side_calls::SideCall::FactExtraction),
                fast_model,
            );
            let builder = aidaemon_memory::Memory::builder(
                Arc::new(store),
                Arc::new(aidaemon_memory::FastEmbedder::new()),
            )
            .extractor(extractor);
            tracing::info!(path = %config.path, "Memory backend: aidaemon-memory");
            let system: Arc<dyn aidaemon_policies::MemorySystem> =
                Arc::new(aidaemon_memory::ScopedMemory::new(builder));
            let mut memory = ConfiguredMemory::replaced_by(system.clone());
            if config.learn_from_history {
                memory.services.push(
                    crate::memory::history_intake::HistoryIntake::new(
                        system,
                        pool,
                        state,
                        &format!("sdk:{}", config.path),
                    )
                    .into_service(),
                );
            }
            memory
        }
        #[cfg(not(feature = "sdk-memory"))]
        crate::config::MemoryBackend::Sdk => {
            let _ = (llm_runtime, side_calls, state, pool);
            tracing::warn!(
                "[memory] backend = \"sdk\" needs a build with the sdk-memory feature; \
                 using the daemon's memory"
            );
            ConfiguredMemory::daemons_own()
        }
    }
}

/// The daemon's own SQLite memory: recall through [`MemoryInjection`]
/// over [`DaemonRecall`], plus the daemon memory tools.
///
/// [`MemoryInjection`]: aidaemon_policies::MemoryInjection
pub(crate) struct DaemonMemory {
    inner: Arc<DaemonMemoryInner>,
    /// The full memory system as the daemon serves it.
    served: SystemMemory,
}

pub(crate) struct DaemonMemoryInner {
    state: Arc<dyn StateStore>,
    store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    event_store: Arc<crate::events::EventStore>,
    llm_runtime: crate::llm_runtime::SharedLlmRuntime,
    /// Runs the conversation summarizer's model calls.
    side_calls: crate::llm_runtime::gateway::SideCallGateway,
    context_window_config: crate::config::ContextWindowConfig,
    record_decision_points: bool,
    /// The full memory system's recall over `store`.
    #[cfg_attr(not(test), allow(dead_code))]
    recall: Arc<aidaemon_memory::full::RecallSource>,
}

impl DaemonMemory {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        state: Arc<dyn StateStore>,
        store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
        event_store: Arc<crate::events::EventStore>,
        llm_runtime: crate::llm_runtime::SharedLlmRuntime,
        side_calls: crate::llm_runtime::gateway::SideCallGateway,
        context_window_config: crate::config::ContextWindowConfig,
        max_facts: usize,
        record_decision_points: bool,
        system_prompt: &str,
    ) -> Self {
        let recall = Arc::new(aidaemon_memory::full::RecallSource::new(
            store.clone(),
            aidaemon_memory::full::RecallSettings {
                max_facts,
                assistant_name: super::system_prompt::infer_assistant_name_from_prompt(
                    system_prompt,
                ),
            },
        ));
        let inner = Arc::new(DaemonMemoryInner {
            recall: recall.clone(),
            state,
            store,
            event_store,
            llm_runtime,
            side_calls,
            context_window_config,
            record_decision_points,
        });
        let host_inner = inner.clone();
        let served = SystemMemory::new(Arc::new(aidaemon_memory::full::FullMemory::new(recall)))
            .with_host_facts(Arc::new(move |turn: &MemoryTurn| {
                let config = &host_inner.context_window_config;
                let writer = DaemonMemoryWriter::new(
                    turn.clone(),
                    WriterSettings {
                        progressive_facts: config.progressive_facts,
                        summarization: false,
                        learning: false,
                    },
                    host_inner.clone(),
                );
                aidaemon_policies::HostTurn::new(aidaemon_memory::full::FullTurn {
                    recall: recall::recall_turn(turn),
                    host: DaemonRecall::host(&host_inner, turn.clone()),
                    hooks: vec![Arc::new(writer)],
                })
            }));
        Self { inner, served }
    }
}

/// The daemon store's extraction, summarization and learning pipelines.
impl MemoryWrites for DaemonMemoryInner {
    fn extract_facts(
        &self,
        turn: &MemoryTurn,
        reply: &str,
        run_started_at: chrono::DateTime<chrono::Utc>,
    ) {
        let snapshot = self.llm_runtime.snapshot();
        crate::memory::progressive::spawn_progressive_extraction(
            self.side_calls.clone(),
            // The side-call gateway picks the model; this value is not sent.
            fast_model(&snapshot, &turn.model),
            self.store.clone(),
            self.event_store.clone(),
            turn.task_id.clone(),
            turn.user_text.clone(),
            reply.to_string(),
            turn.channel_ctx.channel_id.clone(),
            turn.channel_ctx.visibility,
            turn.user_role,
            run_started_at,
        );
    }

    fn summarize(&self, turn: &MemoryTurn) {
        let snapshot = self.llm_runtime.snapshot();
        // The side-call gateway picks the summary model; this value only sets
        // the token-estimate correction. The turn's model is the window the
        // history has to fit.
        let fast_model = fast_model(&snapshot, &turn.model);
        super::conversation_compaction::compact_after_reply(
            super::conversation_compaction::CompactionTarget {
                side_calls: self.side_calls.clone(),
                fast_model: &fast_model,
                state: self.state.clone(),
                event_store: self.event_store.clone(),
                session_id: &turn.session_id,
            },
            turn.turn_id.clone(),
            self.context_window_config
                .summarize_token_threshold_for(&turn.model),
            self.context_window_config
                .summary_recent_tokens_for(&turn.model),
            turn.user_role,
        );
    }

    fn learn(&self, learning: crate::agent::post_task::LearningContext) {
        aidaemon_runtime::turn::post_task::spawn_learning(self.store.clone(), learning);
    }

    fn learn_from_task(
        &self,
        task: crate::traits::Task,
        model: String,
        taint: crate::memory::task_learning::RunTaint,
    ) {
        let state = self.state.clone();
        let store = self.store.clone();
        let side_calls = self.side_calls.clone();
        tokio::spawn(async move {
            let task_id = task.id.clone();
            if let Err(error) = crate::memory::task_learning::extract_task_knowledge(
                state, store, side_calls, model, task, taint,
            )
            .await
            {
                tracing::warn!(task_id = %task_id, %error, "Task knowledge extraction failed");
            }
        });
    }
}

/// The router's fast tier, else the turn's model.
fn fast_model(snapshot: &crate::llm_runtime::LlmRuntimeSnapshot, model: &str) -> String {
    snapshot
        .router()
        .map(|router| router.select(crate::router::Tier::Fast).to_string())
        .unwrap_or_else(|| model.to_string())
}

impl DaemonMemory {
    /// The daemon store's write pipelines, for the turn's own summarization
    /// and learning.
    pub(crate) fn writes(&self) -> Arc<dyn MemoryWrites> {
        self.inner.clone()
    }
}

/// The daemon's memory is the full memory system (`aidaemon_memory::full`)
/// served like any other `MemorySystem`, through [`SystemMemory`] with the
/// daemon's facts about each turn: its audience, its recall host and its
/// run-end writer.
#[async_trait]
impl MemoryPlugin for DaemonMemory {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        self.served.turn(turn).await
    }

    fn owned_daemon_tools(&self) -> &'static [&'static str] {
        DAEMON_MEMORY_TOOLS
    }

    fn is_daemons_own(&self) -> bool {
        true
    }
}

/// Turn fixtures for memory plugin tests.
#[cfg(test)]
pub(super) mod test_support {
    use std::sync::{Arc, Mutex};

    use super::MemoryTurn;
    use crate::agent::post_task::LearningContext;
    use crate::types::{ChannelContext, UserRole};

    pub(crate) fn learning(user_text: &str) -> LearningContext {
        LearningContext {
            user_text: user_text.to_string(),
            memory_persistence_allowed: true,
            intent_domains: Vec::new(),
            tool_calls: Vec::new(),
            errors: Vec::new(),
            first_error: None,
            recovery_actions: Vec::new(),
            start_time: chrono::Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
            task_outcome: None,
        }
    }

    /// An owner-DM turn with memory allowed.
    pub(crate) async fn turn(session_id: &str, user_text: &str) -> MemoryTurn {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        use crate::events::EventStoreForTests as _;
        let events = Arc::new(crate::events::EventStore::new(pool).await.unwrap());
        MemoryTurn {
            session_id: session_id.to_string(),
            task_id: "task-1".to_string(),
            turn_id: "turn-1".to_string(),
            emitter: crate::events::EventEmitter::new(events, session_id),
            user_text: user_text.to_string(),
            user_role: UserRole::Owner,
            channel_ctx: ChannelContext::private("telegram"),
            memory_pipeline_allowed: true,
            memory_persistence_allowed: true,
            mandate: false,
            depth: 0,
            model: "mock-model".to_string(),
            learning: Arc::new(Mutex::new(learning(user_text))),
            verdict: Arc::default(),
            learning_consent: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_runtime::host::TurnHost as _;
    use std::sync::atomic::{AtomicUsize, Ordering};

    async fn turn(mandate: bool) -> MemoryTurn {
        MemoryTurn {
            mandate,
            ..test_support::turn("telegram:synthetic-user-1", "hello").await
        }
    }

    /// Counts `before_model` calls.
    #[derive(Default)]
    struct CountingHook(AtomicUsize);

    #[async_trait]
    impl sdk::Hook for CountingHook {
        async fn before_model(
            &self,
            _ctx: &sdk::HookContext<'_>,
            _request: &mut sdk::ModelRequest,
        ) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A plugin that owns no daemon tool and adds one hook.
    struct StubMemory(Arc<CountingHook>);

    #[async_trait]
    impl MemoryPlugin for StubMemory {
        async fn turn(&self, _turn: &MemoryTurn) -> MemoryParts {
            MemoryParts {
                hooks: vec![self.0.clone()],
                ..MemoryParts::default()
            }
        }
    }

    fn offered_tool_names(call: &crate::testing::MockChatCall) -> Vec<String> {
        call.tools
            .iter()
            .filter_map(|tool| {
                tool.get("function")
                    .and_then(|function| function.get("name"))
                    .or_else(|| tool.get("name"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }

    fn offered_actions(call: &crate::testing::MockChatCall, tool: &str) -> Vec<String> {
        call.tools
            .iter()
            .map(|definition| definition.get("function").unwrap_or(definition))
            .find(|function| function["name"] == tool)
            .and_then(|function| function.pointer("/parameters/properties/action/enum"))
            .and_then(serde_json::Value::as_array)
            .map(|actions| {
                actions
                    .iter()
                    .filter_map(|action| action.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn system_text(call: &crate::testing::MockChatCall) -> String {
        call.messages
            .iter()
            .filter(|message| message["role"] == "system")
            .filter_map(|message| message["content"].as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn owner_request(text: &str) -> crate::runtime_ports::InboundMessageRequest {
        crate::runtime_ports::InboundMessageRequest {
            session_id: "telegram:synthetic-user-1".to_string(),
            user_text: text.to_string(),
            attachments: Vec::new(),
            status_tx: None,
            user_role: UserRole::Owner,
            channel_ctx: ChannelContext::private("telegram"),
            heartbeat: None,
            ingress_timing: None,
        }
    }

    #[tokio::test]
    async fn mandate_turn_gets_no_memory_parts() {
        let plugin = StubMemory(Arc::default());
        assert_eq!(
            memory_parts(&plugin, &turn(false).await).await.hooks.len(),
            1
        );
        let parts = memory_parts(&plugin, &turn(true).await).await;
        assert!(parts.hooks.is_empty() && parts.tools.is_empty());
    }

    #[tokio::test]
    async fn memory_tools_follow_the_active_plugin() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;

        let harness = crate::testing::setup_test_agent_sdk(MockProvider::with_responses(vec![
            MockProvider::text_response("Hi."),
        ]))
        .await
        .unwrap();
        harness
            .agent
            .handle_inbound_message(owner_request("hello there"))
            .await
            .unwrap();
        let log = harness.provider.call_log.lock().await;
        assert!(offered_tool_names(log.last().unwrap()).contains(&"remember_fact".to_string()));
        assert!(system_text(log.last().unwrap()).contains("`manage_memories` (search/list)"));
        drop(log);

        let manage_memories: Arc<dyn crate::traits::Tool> =
            Arc::new(crate::tools::ManageMemoriesTool::new(
                harness.state.clone(),
                harness.state.memory_stores(),
            ));
        let mut harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
            MockProvider::with_responses(vec![MockProvider::text_response("Hi.")]),
            vec![manage_memories],
            None,
        )
        .await
        .unwrap();
        harness
            .agent
            .set_memory_plugin(Arc::new(StubMemory(Arc::default())));
        harness
            .agent
            .handle_inbound_message(owner_request("hello there"))
            .await
            .unwrap();
        let log = harness.provider.call_log.lock().await;
        let names = offered_tool_names(log.last().unwrap());
        for tool in ["remember_fact", "manage_people"] {
            assert!(
                !names.contains(&tool.to_string()),
                "{tool} offered: {names:?}"
            );
        }
        // manage_memories keeps only its goal actions.
        let actions = offered_actions(log.last().unwrap(), "manage_memories");
        assert!(
            actions.contains(&"list_scheduled".to_string()),
            "{actions:?}"
        );
        assert!(!actions.contains(&"search".to_string()), "{actions:?}");
        assert!(!actions.contains(&"forget".to_string()), "{actions:?}");
        let system = system_text(log.last().unwrap());
        assert!(
            system.contains("`manage_memories` (goal actions)"),
            "{system}"
        );
        assert!(!system.contains("(search/list)"), "{system}");
    }

    /// The goal-actions view keeps exactly the goal actions `manage_memories`
    /// declares and refuses the rest.
    #[tokio::test]
    async fn goal_actions_view_covers_only_goal_actions() {
        use serde_json::json;

        let harness = crate::testing::setup_test_agent_sdk(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let schema = crate::traits::Tool::schema(&crate::tools::ManageMemoriesTool::new(
            harness.state.clone(),
            harness.state.memory_stores(),
        ));
        let declared: Vec<String> = schema["parameters"]["properties"]["action"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|action| action.as_str().unwrap().to_string())
            .collect();
        for action in MANAGE_MEMORIES_GOAL_ACTIONS {
            assert!(
                declared.contains(&action.to_string()),
                "{action} not declared"
            );
        }
        let view = goal_actions_view(&schema);
        let narrowed = view["parameters"]["properties"]["action"]["enum"]
            .as_array()
            .unwrap();
        assert_eq!(narrowed.len(), MANAGE_MEMORIES_GOAL_ACTIONS.len());
        assert!(check_goal_action(&json!({"action": "pause_scheduled"})).is_ok());
        assert!(check_goal_action(&json!({"action": "forget"})).is_err());
        assert!(check_goal_action(&json!({})).is_err());
        assert_eq!(
            daemon_tool_offer("manage_memories", &[]),
            DaemonToolOffer::GoalActions
        );
        assert_eq!(
            daemon_tool_offer("remember_fact", &[]),
            DaemonToolOffer::Hidden
        );
        assert_eq!(
            daemon_tool_offer("manage_memories", DAEMON_MEMORY_TOOLS),
            DaemonToolOffer::Full
        );
        assert_eq!(daemon_tool_offer("terminal", &[]), DaemonToolOffer::Full);
    }

    #[tokio::test]
    async fn plugin_hooks_run_on_sdk_turns() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;

        let mut harness = crate::testing::setup_test_agent_sdk(MockProvider::with_responses(vec![
            MockProvider::text_response("Hi."),
        ]))
        .await
        .unwrap();
        let hook = Arc::new(CountingHook::default());
        harness
            .agent
            .set_memory_plugin(Arc::new(StubMemory(hook.clone())));
        harness
            .agent
            .handle_inbound_message(owner_request("hello there"))
            .await
            .unwrap();
        assert_eq!(hook.0.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn daemon_memory_owns_its_tools() {
        use crate::testing::MockProvider;

        let mut harness = crate::testing::setup_test_agent_sdk(MockProvider::new())
            .await
            .unwrap();
        assert_eq!(
            harness.agent.memory_plugin().owned_daemon_tools(),
            [
                "remember_fact",
                "manage_memories",
                "manage_people",
                "share_memory"
            ]
        );
        use aidaemon_runtime::engine::tool_roster::ToolRoster as _;
        let roster = |agent: &crate::agent::Agent| {
            agent
                .session_static_tool_roster(UserRole::Owner, &ChannelContext::private("telegram"))
                .into_iter()
                .map(|(name, _)| name)
                .collect::<Vec<_>>()
        };
        assert!(roster(&harness.agent).contains(&"remember_fact".to_string()));
        harness
            .agent
            .set_memory_plugin(Arc::new(StubMemory(Arc::default())));
        assert!(!roster(&harness.agent).contains(&"remember_fact".to_string()));
    }

    #[test]
    fn memory_backend_defaults_to_daemon_and_parses_sdk() {
        use crate::config::{MemoryBackend, MemoryConfig};

        let default: MemoryConfig = toml::from_str("").unwrap();
        assert_eq!(default.backend, MemoryBackend::Daemon);
        assert_eq!(default.path, "aidaemon-memory.db");
        let sdk: MemoryConfig = toml::from_str("backend = \"sdk\"\npath = \"m.db\"").unwrap();
        assert_eq!(sdk.backend, MemoryBackend::Sdk);
        assert_eq!(sdk.path, "m.db");
    }

    async fn learned_after_one_turn(replace_memory: bool) -> usize {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;

        let mut harness =
            crate::testing::setup_test_agent_root(MockProvider::with_responses(vec![
                MockProvider::text_response("Hi."),
            ]))
            .await
            .unwrap();
        let writes = Arc::new(writer::tests::RecordingWrites::default());
        if replace_memory {
            harness
                .agent
                .set_memory_plugin(Arc::new(StubMemory(Arc::default())));
        }
        harness.agent.set_test_transcript_writes(writes.clone());
        harness
            .agent
            .handle_inbound_message(owner_request("hello there"))
            .await
            .unwrap();
        let recorded = writes.0.lock().unwrap();
        assert!(recorded.extracted.is_empty());
        recorded.learned.len()
    }

    /// Learning from finished runs writes the daemon's own stores, so it
    /// belongs to the daemon's memory and stops when another system replaces it.
    #[tokio::test]
    async fn only_the_daemons_memory_learns_from_runs() {
        assert_eq!(learned_after_one_turn(false).await, 1);
        assert_eq!(learned_after_one_turn(true).await, 0);
    }
}
