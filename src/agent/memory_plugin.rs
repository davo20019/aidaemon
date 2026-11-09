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

use std::collections::HashMap;
use std::sync::Arc;

use aidaemon_core as sdk;
use async_trait::async_trait;

use crate::traits::StateStore;
use crate::types::{ChannelContext, UserRole};

mod recall;
mod system_memory;
mod writer;
pub(crate) use recall::DaemonRecall;
#[cfg_attr(not(feature = "sdk-memory"), allow(unused_imports))]
pub(crate) use system_memory::SystemMemory;
pub(crate) use writer::{DaemonMemoryWriter, MemoryWrites, WriterSettings};

/// The daemon's built-in memory tools. Offered only while the active plugin
/// lists them in [`MemoryParts::owned_daemon_tools`].
pub(crate) const DAEMON_MEMORY_TOOLS: &[&str] =
    &["remember_fact", "manage_memories", "manage_people"];

/// `manage_memories` actions that manage goals, not memory. They stay with
/// the daemon whatever the plugin: under a plugin that does not own
/// `manage_memories`, the tool is offered narrowed to these actions.
pub(crate) const MANAGE_MEMORIES_GOAL_ACTIONS: &[&str] = &[
    "create_personal_goal",
    "list_goals",
    "complete_goal",
    "abandon_goal",
    "create_scheduled_goal",
    "activate_scheduled",
    "list_scheduled",
    "list_scheduled_matching",
    "add_schedule",
    "cancel_scheduled",
    "pause_scheduled",
    "resume_scheduled",
    "retry_scheduled",
    "retry_failed_scheduled",
    "cancel_scheduled_matching",
    "retry_scheduled_matching",
    "diagnose_scheduled",
];

/// How a daemon tool is offered under the active plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DaemonToolOffer {
    Full,
    /// `manage_memories` with only [`MANAGE_MEMORIES_GOAL_ACTIONS`].
    GoalActions,
    Hidden,
}

/// How the daemon tool `name` is offered while the active plugin owns the
/// daemon memory tools `owned`.
pub(crate) fn daemon_tool_offer(name: &str, owned: &[&str]) -> DaemonToolOffer {
    if !DAEMON_MEMORY_TOOLS.contains(&name) || owned.contains(&name) {
        DaemonToolOffer::Full
    } else if name == "manage_memories" {
        DaemonToolOffer::GoalActions
    } else {
        DaemonToolOffer::Hidden
    }
}

/// `manage_memories`' function definition (bare or wrapped in `function`)
/// narrowed to its goal actions.
pub(crate) fn goal_actions_view(definition: &serde_json::Value) -> serde_json::Value {
    let mut definition = definition.clone();
    let function = match definition.get_mut("function") {
        Some(function) => function,
        None => &mut definition,
    };
    function["description"] = serde_json::Value::String(
        "Personal goals and scheduled goals: create, list, pause, resume, retry, cancel, diagnose."
            .to_string(),
    );
    if let Some(actions) = function
        .pointer_mut("/parameters/properties/action/enum")
        .and_then(serde_json::Value::as_array_mut)
    {
        actions.retain(|action| {
            action
                .as_str()
                .is_some_and(|action| MANAGE_MEMORIES_GOAL_ACTIONS.contains(&action))
        });
    }
    definition
}

/// Rejects a `manage_memories` call outside the goal-actions view.
pub(crate) fn check_goal_action(arguments: &serde_json::Value) -> Result<(), String> {
    let action = arguments
        .get("action")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if MANAGE_MEMORIES_GOAL_ACTIONS.contains(&action) {
        Ok(())
    } else {
        Err(format!(
            "manage_memories action `{action}` is not available here: it only manages goals. \
             Use the memory tools you were given for facts and preferences."
        ))
    }
}

/// Everything a plugin needs to scope memory for one turn.
#[derive(Clone)]
pub(crate) struct MemoryTurn {
    pub session_id: String,
    pub task_id: String,
    /// The conversation turn this run answers. It is over when the run ends,
    /// before its `task_end` is appended.
    pub turn_id: String,
    /// The turn's event emitter (decision points, memory telemetry).
    pub emitter: crate::events::EventEmitter,
    /// The persisted user message for the turn.
    pub user_text: String,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    /// The compiled policy allows reading memory.
    pub memory_pipeline_allowed: bool,
    /// The compiled policy allows writing memory.
    pub memory_persistence_allowed: bool,
    /// A mandate worker: no memory at all.
    pub mandate: bool,
    pub depth: usize,
    pub model: String,
    /// The run's learning context; the dispatcher adds its tool calls.
    pub learning: Arc<std::sync::Mutex<crate::agent::post_task::LearningContext>>,
    /// How the turn judges the run's final status.
    pub verdict: Arc<crate::agent::sdk_engine::TurnVerdict>,
}

impl MemoryTurn {
    /// Worker sessions keep summaries but never learn owner facts.
    pub(crate) fn memory_isolated(&self) -> bool {
        self.depth > 0 || crate::events::is_memory_isolated_worker_session(&self.session_id)
    }
}

/// What a plugin adds to one turn.
#[derive(Default)]
pub(crate) struct MemoryParts {
    pub hooks: Vec<Arc<dyn sdk::Hook>>,
    pub tools: Vec<Arc<dyn sdk::Tool>>,
    /// Daemon tools this plugin owns; offered only while it is the active
    /// plugin. Filled from [`MemoryPlugin::owned_daemon_tools`].
    pub owned_daemon_tools: &'static [&'static str],
}

#[async_trait]
pub(crate) trait MemoryPlugin: Send + Sync {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts;

    /// The daemon memory tools ([`DAEMON_MEMORY_TOOLS`]) this plugin owns.
    fn owned_daemon_tools(&self) -> &'static [&'static str] {
        &[]
    }
}

/// The parts `plugin` adds to `turn`. A mandate worker gets none, whatever
/// the plugin: its authority envelope excludes owner memory.
pub(crate) async fn memory_parts(plugin: &dyn MemoryPlugin, turn: &MemoryTurn) -> MemoryParts {
    if turn.mandate {
        return MemoryParts::default();
    }
    MemoryParts {
        owned_daemon_tools: plugin.owned_daemon_tools(),
        ..plugin.turn(turn).await
    }
}

/// The memory system this daemon runs with.
pub(crate) struct ConfiguredMemory {
    /// What serves turns; `None` for the daemon's own memory.
    pub plugin: Option<Arc<dyn MemoryPlugin>>,
    /// Background work of a replacing system.
    pub services: Vec<Arc<dyn aidaemon_core::Service>>,
}

impl ConfiguredMemory {
    fn daemons_own() -> Self {
        Self {
            plugin: None,
            services: Vec::new(),
        }
    }

    #[cfg_attr(not(any(test, feature = "sdk-memory")), allow(dead_code))]
    pub(crate) fn replaced_by(system: Arc<dyn aidaemon_policies::MemorySystem>) -> Self {
        Self {
            services: system.services(),
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
    state: Arc<dyn StateStore>,
    event_store: Arc<crate::events::EventStore>,
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
            let fast_model = llm_runtime
                .router()
                .map(|router| router.select(crate::router::Tier::Fast).to_string())
                .unwrap_or_else(|| llm_runtime.snapshot().primary_model());
            let extractor = aidaemon_memory::FactExtractor::new(
                super::sdk_provider::RuntimeProvider {
                    runtime: llm_runtime.clone(),
                    accounting: Some(super::sdk_provider::ModelCallAccounting {
                        state,
                        event_store,
                        purpose: "sdk_memory_extraction",
                    }),
                },
                fast_model,
            );
            let builder = aidaemon_memory::Memory::builder(
                Arc::new(store),
                Arc::new(aidaemon_memory::FastEmbedder::new()),
            )
            .extractor(extractor);
            tracing::info!(path = %config.path, "Memory backend: aidaemon-memory");
            ConfiguredMemory::replaced_by(Arc::new(aidaemon_memory::ScopedMemory::new(builder)))
        }
        #[cfg(not(feature = "sdk-memory"))]
        crate::config::MemoryBackend::Sdk => {
            let _ = (llm_runtime, state, event_store);
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
}

pub(crate) struct DaemonMemoryInner {
    state: Arc<dyn StateStore>,
    event_store: Arc<crate::events::EventStore>,
    llm_runtime: crate::llm_runtime::SharedLlmRuntime,
    context_window_config: crate::config::ContextWindowConfig,
    max_facts: usize,
    record_decision_points: bool,
    /// The configured assistant name (pinned among the critical facts).
    assistant_name: Option<String>,
    /// Per session, the entities the core profile selected last, so the
    /// selection stays stable across turns.
    core_profile_ids: tokio::sync::RwLock<HashMap<String, Vec<String>>>,
}

impl DaemonMemory {
    pub(crate) fn new(
        state: Arc<dyn StateStore>,
        event_store: Arc<crate::events::EventStore>,
        llm_runtime: crate::llm_runtime::SharedLlmRuntime,
        context_window_config: crate::config::ContextWindowConfig,
        max_facts: usize,
        record_decision_points: bool,
        system_prompt: &str,
    ) -> Self {
        Self {
            inner: Arc::new(DaemonMemoryInner {
                state,
                event_store,
                llm_runtime,
                context_window_config,
                max_facts,
                record_decision_points,
                assistant_name: super::system_prompt::infer_assistant_name_from_prompt(
                    system_prompt,
                ),
                core_profile_ids: tokio::sync::RwLock::new(HashMap::new()),
            }),
        }
    }
}

/// The daemon store's extraction, summarization and learning pipelines.
impl MemoryWrites for DaemonMemoryInner {
    fn extract_facts(&self, turn: &MemoryTurn, reply: &str) {
        let snapshot = self.llm_runtime.snapshot();
        crate::memory::progressive::spawn_progressive_extraction(
            self.llm_runtime.clone(),
            fast_model(&snapshot, &turn.model),
            self.state.clone(),
            self.event_store.clone(),
            turn.task_id.clone(),
            turn.user_text.clone(),
            reply.to_string(),
            turn.channel_ctx.channel_id.clone(),
            turn.channel_ctx.visibility,
            turn.user_role,
        );
    }

    fn summarize(&self, turn: &MemoryTurn) {
        let snapshot = self.llm_runtime.snapshot();
        // The fast model writes the summary; the turn's model is the window
        // the history has to fit.
        let fast_model = fast_model(&snapshot, &turn.model);
        super::conversation_compaction::compact_after_reply(
            super::conversation_compaction::CompactionTarget {
                runtime: self.llm_runtime.clone(),
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
        crate::agent::post_turn::spawn_learning(self.state.clone(), learning);
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

#[async_trait]
impl MemoryPlugin for DaemonMemory {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        let recall = DaemonRecall::new(self.inner.clone(), turn.clone());
        let config = &self.inner.context_window_config;
        let writer = DaemonMemoryWriter::new(
            turn.clone(),
            WriterSettings {
                progressive_facts: config.progressive_facts,
                summarization: false,
                learning: false,
            },
            self.inner.clone(),
        );
        MemoryParts {
            hooks: vec![
                Arc::new(aidaemon_policies::MemoryInjection::new(recall)),
                Arc::new(writer),
            ],
            ..MemoryParts::default()
        }
    }

    fn owned_daemon_tools(&self) -> &'static [&'static str] {
        DAEMON_MEMORY_TOOLS
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            Arc::new(crate::tools::ManageMemoriesTool::new(harness.state.clone()));
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
            harness.agent.memory.owned_daemon_tools(),
            ["remember_fact", "manage_memories", "manage_people"]
        );
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

    /// Summarization and learning are the turn's, not the plugin's: they run
    /// under a plugin that adds no writer.
    #[tokio::test]
    async fn transcript_memory_runs_under_any_plugin() {
        use crate::runtime_ports::AgentIngress;
        use crate::testing::MockProvider;

        let mut harness =
            crate::testing::setup_test_agent_root(MockProvider::with_responses(vec![
                MockProvider::text_response("Hi."),
            ]))
            .await
            .unwrap();
        let writes = Arc::new(writer::tests::RecordingWrites::default());
        harness
            .agent
            .set_memory_plugin(Arc::new(StubMemory(Arc::default())));
        harness.agent.set_test_transcript_writes(writes.clone());
        harness
            .agent
            .handle_inbound_message(owner_request("hello there"))
            .await
            .unwrap();
        let recorded = writes.0.lock().unwrap();
        assert_eq!(recorded.learned.len(), 1);
        assert!(recorded.extracted.is_empty());
    }
}
