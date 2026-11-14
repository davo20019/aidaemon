//! Builds the daemon's [`Agent`] from its parts and the daemon's host ports:
//! its memory over the daemon stores, its file policy, OTel export and
//! computer-use pins.

use std::sync::Arc;

#[cfg(test)]
use super::Agent;
use super::AgentParts;
use aidaemon_runtime::agent::{HostPorts, TraceExport};

/// The daemon's root agent, unshared, for tests that change it before use
/// (startup assembles through `aidaemon_sdk::runtime::host`).
#[cfg(test)]
pub(crate) fn new_agent(parts: AgentParts) -> Agent {
    let ports = host_ports(&parts);
    Agent::new(parts, ports)
}

/// The daemon's ports for an agent built from `parts`.
pub(crate) fn host_ports(parts: &AgentParts) -> HostPorts {
    let deps = &parts.dependencies;
    let daemon_memory = super::memory_plugin::DaemonMemory::new(
        deps.state.clone(),
        deps.memory_store.clone(),
        deps.event_store.clone(),
        deps.llm_runtime.clone(),
        deps.side_calls.clone(),
        parts.context_window_config.clone(),
        parts.max_facts,
        parts.record_decision_points,
        &parts.system_prompt,
    );
    let transcript_writes = daemon_memory.writes();
    HostPorts {
        memory: Arc::new(daemon_memory),
        transcript_writes,
        file_policy: crate::tools::daemon_file_policy(parts.daemon_paths.clone()),
        trace: Some(Arc::new(DaemonTrace)),
        computer_use: computer_use(),
    }
}

/// The daemon's OTel export, read when a run starts.
struct DaemonTrace;

impl TraceExport for DaemonTrace {
    fn sink(&self) -> Option<Arc<dyn aidaemon_core::EventSink>> {
        crate::otel_export::sink()
    }

    fn observe_sent(&self, run: aidaemon_core::RunId, messages: &[serde_json::Value]) {
        crate::otel_export::observe_sent(run, messages);
    }
}

fn computer_use() -> Option<Arc<dyn aidaemon_runtime::host::ComputerUseState>> {
    #[cfg(feature = "computer_use")]
    {
        Some(Arc::new(super::computer_use::ComputerUsePins))
    }
    #[cfg(not(feature = "computer_use"))]
    None
}

/// Lower the token-pressure watermark in focused tests without weakening
/// production defaults; the turn's post-reply compaction reads the
/// transcript writer's own copy, rebuilt here with the lower one.
#[cfg(test)]
pub(crate) fn set_context_compaction_tokens_for_test(
    agent: &mut Agent,
    token_threshold: usize,
    recent_tokens: usize,
) {
    use aidaemon_runtime::host::TurnHost as _;
    agent.set_test_context_compaction_tokens(token_threshold, recent_tokens);
    let writes = super::memory_plugin::DaemonMemory::new(
        agent.state_store().clone(),
        agent.memory_stores().clone(),
        agent.event_store().clone(),
        agent.llm_runtime().clone(),
        agent.side_calls().clone(),
        agent.turn_runtime().context_window_config.clone(),
        0,
        false,
        "",
    )
    .writes();
    agent.set_test_transcript_writes(writes);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::events::EventStoreForTests as _;
    use crate::testing::MockProvider;
    use crate::types::{ChannelContext, UserRole};

    /// Startup's path: the daemon's parts and ports assembled through the
    /// SDK's turn-runtime preset give an agent wired to itself that answers
    /// a turn, with the engine settings and run store it was given.
    #[tokio::test]
    async fn the_daemon_assembles_through_the_runtime_preset() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let skills_dir = tempfile::TempDir::new().unwrap();
        let embedding = Arc::new(crate::memory::embeddings::EmbeddingService::deterministic());
        let state = Arc::new(
            crate::state::SqliteStateStore::for_tests(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding,
            )
            .await
            .unwrap(),
        );
        let event_store = Arc::new(crate::events::EventStore::new(state.pool()).await.unwrap());
        let provider = Arc::new(MockProvider::with_responses(vec![
            MockProvider::text_response("Assembled and answering."),
        ]));
        let llm_runtime = crate::llm_runtime::SharedLlmRuntime::new(
            provider.clone() as crate::llm_runtime::ModelBackend,
            None,
            crate::config::ProviderKind::OpenaiCompatible,
            "mock-model".to_string(),
        );
        let files = crate::config::FilesConfig::default();
        let parts = super::AgentParts {
            dependencies: super::super::AgentDependencies {
                memory_store: state.memory_stores(),
                side_calls: crate::llm_runtime::gateway::SideCallGateway::new(
                    llm_runtime.clone(),
                    state.clone(),
                    event_store.clone(),
                ),
                llm_runtime,
                state: state.clone() as Arc<dyn crate::traits::StateStore>,
                event_store,
                tools: Vec::new(),
            },
            model: "mock-model".to_string(),
            system_prompt: "You are a helpful test assistant.".to_string(),
            config_path: std::path::PathBuf::from("config.toml"),
            daemon_paths: Default::default(),
            execution_backend: crate::execution::active_execution_backend(),
            skills_dir: skills_dir.path().to_path_buf(),
            max_depth: 3,
            max_iterations: 10,
            max_iterations_cap: 20,
            max_response_chars: 8000,
            timeout_secs: 30,
            max_facts: 20,
            iteration_config: crate::config::IterationLimitConfig::Unlimited,
            task_token_budget: None,
            spawn_family: Default::default(),
            llm_call_timeout_secs: None,
            mcp_registry: None,
            goal_token_registry: None,
            hub: None,
            record_decision_points: false,
            context_window_config: Default::default(),
            policy_config: Default::default(),
            path_aliases: Default::default(),
            inbox_dir: None,
            inherited_project_scope: None,
            specialists: Arc::new(crate::agent::specialists::SpecialistRegistry::load(None)),
            interactive_slot: None,
            vision_config: crate::config::VisionConfig::from_files(&files),
            audio_config: crate::config::AudioConfig::from_files(&files),
            stt_config: crate::config::SttConfig::from_files(&files),
            harness_eval_config: (&crate::config::DiagnosticsHarnessEvalConfig::default()).into(),
        };
        let ports = super::host_ports(&parts);
        let run_store = Arc::new(
            aidaemon_store_sqlite::SqliteStore::from_pool(state.pool())
                .await
                .unwrap(),
        );
        let agent = aidaemon_sdk::runtime::host(parts, ports)
            .engine(Default::default())
            .run_store(run_store.clone())
            .build()
            .await;

        crate::startup::wiring::wire_agent_cycles(&agent, None)
            .await
            .expect("the preset wires the agent to itself");
        {
            use aidaemon_runtime::host::EngineContext as _;
            assert!(Arc::ptr_eq(agent.run_store().unwrap(), &run_store));
        }
        let reply = agent
            .handle_message(
                "telegram:synthetic-user-1",
                "Say hello.",
                None,
                UserRole::Owner,
                ChannelContext::private("synthetic-user-1"),
                None,
            )
            .await
            .unwrap();
        assert_eq!(reply, "Assembled and answering.");
    }
}
