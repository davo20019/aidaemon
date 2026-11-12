//! Nodes: the domain, Gateway, store and tools live in `aidaemon-nodes`.
//! The daemon keeps its adapters: the agent behind Node turns
//! (`service::AgentNodeConversationIngress`), the instance key in the
//! daemon's secret store (`auth`) and the `aidaemon node` CLI.

pub use aidaemon_nodes::{
    announcement, channel, domain, gateway, monitoring, ota, protocol, restart, simulator, speech,
    store, tool,
};

pub mod cli;
pub mod service;

/// Node authentication, with the instance key kept in the daemon's secret store.
pub mod auth {
    pub use aidaemon_nodes::auth::*;

    /// The Gateway instance key from the daemon's secret store.
    pub fn load_or_create_instance_key() -> anyhow::Result<[u8; 32]> {
        aidaemon_nodes::auth::load_or_create_instance_key(crate::config::secret_store().as_ref())
    }
}

pub use aidaemon_nodes::{AuthenticatedNodeContext, NodeAction, NodeRecord};
pub use aidaemon_nodes::{NodeConversationIngress, NodeService, NodeStore};
