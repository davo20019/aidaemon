//! The turn runtime, assembled (feature `runtime`): aidaemon's own agent
//! host ([`Agent`], from `aidaemon-runtime`) and the [`TurnEngine`] its turns
//! run on. This is the preset the aidaemon daemon itself builds through.
//!
//! The host supplies two things: the agent's [`AgentParts`] (its services,
//! settings and limits) and its [`HostPorts`] (memory, file policy, trace
//! export, computer use). [`host`] puts them together with the engine's
//! settings and run store, and [`HostBuilder::build`] returns the agent,
//! shared and wired to itself, ready for channels to call.
//!
//! ```ignore
//! let agent = aidaemon_sdk::runtime::host(parts, ports)
//!     .engine(settings)
//!     .run_store(store)
//!     .build()
//!     .await;
//! let reply = agent.handle_message(session, text, None, role, ctx, None).await?;
//! ```

use std::sync::Arc;

pub use aidaemon_runtime::agent::{
    Agent, AgentDependencies, AgentParts, HostPorts, ProviderSwap, TraceExport,
};
pub use aidaemon_runtime::engine::settings::SdkEngineSettings as EngineSettings;
pub use aidaemon_runtime::engine::turn_engine::TurnEngine;
pub use aidaemon_runtime::host::TurnHost;
pub use aidaemon_runtime::turn::memory::MemoryPlugin;

/// Start assembling an agent host from its parts and the host's ports.
pub fn host(parts: AgentParts, ports: HostPorts) -> HostBuilder {
    HostBuilder {
        parts,
        ports,
        engine: EngineSettings::default(),
        run_store: None,
        memory: None,
    }
}

/// An agent host being assembled; see [`host`].
#[must_use = "call .build() to get the agent"]
pub struct HostBuilder {
    parts: AgentParts,
    ports: HostPorts,
    engine: EngineSettings,
    run_store: Option<Arc<aidaemon_store_sqlite::SqliteStore>>,
    memory: Option<Arc<dyn MemoryPlugin>>,
}

impl HostBuilder {
    /// The turn engine's settings for this agent's turns and its
    /// children's. Default: [`EngineSettings::default`].
    pub fn engine(mut self, settings: EngineSettings) -> Self {
        self.engine = settings;
        self
    }

    /// Where every run journals its events and checkpoints its state.
    /// Default: none (runs are not resumable after a restart).
    pub fn run_store(mut self, store: Arc<aidaemon_store_sqlite::SqliteStore>) -> Self {
        self.run_store = Some(store);
        self
    }

    /// A memory plugin in place of the ports' memory, when `Some`.
    pub fn memory(mut self, memory: Option<Arc<dyn MemoryPlugin>>) -> Self {
        self.memory = memory;
        self
    }

    /// The agent, shared and holding a weak reference to itself (for the
    /// background work its turns start).
    pub async fn build(self) -> Arc<Agent> {
        let HostBuilder {
            parts,
            ports,
            engine,
            run_store,
            memory,
        } = self;
        let mut agent = Agent::new(parts, ports);
        agent.set_sdk_engine(engine);
        if let Some(store) = run_store {
            agent.set_run_store(store);
        }
        if let Some(memory) = memory {
            agent.set_memory_plugin(memory);
        }
        let agent = Arc::new(agent);
        agent.set_self_ref(Arc::downgrade(&agent)).await;
        agent
    }
}
