//! Long-term memory as a plugin on SDK turns.
//!
//! The agent reaches memory only through [`MemoryPlugin`]: each turn the
//! active plugin returns the hooks (recall injection, run-end writing) and
//! tools it wants, plus the daemon memory tools it owns. Swapping the plugin
//! swaps the memory backend without touching the turn code.

use std::collections::HashMap;
use std::sync::Arc;

use aidaemon_core as sdk;
use async_trait::async_trait;

use crate::traits::StateStore;
use crate::types::{ChannelContext, UserRole};

mod recall;
pub(crate) use recall::DaemonRecall;

/// The daemon's built-in memory tools. Offered only while the active plugin
/// lists them in [`MemoryParts::owned_daemon_tools`].
pub(crate) const DAEMON_MEMORY_TOOLS: &[&str] =
    &["remember_fact", "manage_memories", "manage_people"];

/// Everything a plugin needs to scope memory for one turn.
#[derive(Clone)]
pub(crate) struct MemoryTurn {
    pub session_id: String,
    pub task_id: String,
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
    /// Daemon tools this plugin owns; offered only while it is the active plugin.
    pub owned_daemon_tools: &'static [&'static str],
}

#[async_trait]
pub(crate) trait MemoryPlugin: Send + Sync {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts;
}

/// The parts `plugin` adds to `turn`. A mandate worker gets none, whatever
/// the plugin: its authority envelope excludes owner memory.
pub(crate) async fn memory_parts(plugin: &dyn MemoryPlugin, turn: &MemoryTurn) -> MemoryParts {
    if turn.mandate {
        return MemoryParts::default();
    }
    plugin.turn(turn).await
}

/// Whether the daemon tool `name` is offered under `parts`.
pub(crate) fn daemon_tool_offered(name: &str, parts: &MemoryParts) -> bool {
    !DAEMON_MEMORY_TOOLS.contains(&name) || parts.owned_daemon_tools.contains(&name)
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
        max_facts: usize,
        record_decision_points: bool,
        system_prompt: &str,
    ) -> Self {
        Self {
            inner: Arc::new(DaemonMemoryInner {
                state,
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

#[async_trait]
impl MemoryPlugin for DaemonMemory {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        let recall = DaemonRecall::new(self.inner.clone(), turn.clone());
        MemoryParts {
            hooks: vec![Arc::new(aidaemon_policies::MemoryInjection::new(recall))],
            tools: Vec::new(),
            owned_daemon_tools: DAEMON_MEMORY_TOOLS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    async fn turn(mandate: bool) -> MemoryTurn {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        let events = Arc::new(crate::events::EventStore::new(pool).await.unwrap());
        MemoryTurn {
            session_id: "telegram:synthetic-user-1".to_string(),
            task_id: "task-1".to_string(),
            emitter: crate::events::EventEmitter::new(events, "telegram:synthetic-user-1"),
            user_text: "hello".to_string(),
            user_role: UserRole::Owner,
            channel_ctx: ChannelContext::private("telegram"),
            memory_pipeline_allowed: true,
            memory_persistence_allowed: true,
            mandate,
            depth: 0,
            model: "mock-model".to_string(),
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
        drop(log);

        let mut harness = crate::testing::setup_test_agent_sdk(MockProvider::with_responses(vec![
            MockProvider::text_response("Hi."),
        ]))
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
        for tool in DAEMON_MEMORY_TOOLS {
            assert!(
                !names.contains(&tool.to_string()),
                "{tool} offered: {names:?}"
            );
        }
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
}
