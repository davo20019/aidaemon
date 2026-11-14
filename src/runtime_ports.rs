//! Narrow runtime interfaces used at subsystem boundaries.
//!
//! These ports keep tools and outbound channel routing independent from the
//! concrete `Agent` and `ChannelHub` implementations. Startup owns the only
//! place where those implementations are connected.

use async_trait::async_trait;

#[cfg(test)]
pub(crate) use aidaemon_runtime::spawn::ChildWorkReport;
#[cfg(test)]
pub(crate) use aidaemon_runtime::spawn::{ChildAgentRequest, ChildAgentRun, SalvagedTaskOutcome};
pub(crate) use aidaemon_runtime::spawn::{ChildAgentRuntime, ConversationRuntime};
#[cfg(test)]
pub(crate) use aidaemon_runtime::spawn::{ChildTimeLimit, ChildTimedOut};

#[cfg(test)]
pub(crate) use aidaemon_channels::ports::{
    AgentIngress, InboundMessageRequest, InboundMessageTiming,
};
pub(crate) use aidaemon_channels::ports::{ChannelAgentRuntime, OutboundRouter};

/// The daemon's additions to [`ChannelAgentRuntime`]: what its own chat
/// glue (Slack project grants, `/reload`) needs from the agent.
#[async_trait]
pub(crate) trait DaemonAgentRuntime: ChannelAgentRuntime {
    /// Whether the file tools must not write `path` on this daemon.
    fn is_sensitive_path(&self, path: &std::path::Path) -> bool;
    /// The daemon's execution environment.
    fn execution_backend(&self) -> crate::execution::SharedExecutionBackend;
    async fn reload_provider(&self, config: &crate::config::AppConfig) -> anyhow::Result<String>;
}
