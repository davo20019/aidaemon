//! The agent as the daemon's own chat glue's port (`/reload`, Slack
//! project grants).

use async_trait::async_trait;

use super::Agent;
use crate::runtime_ports::DaemonAgentRuntime;

#[async_trait]
impl DaemonAgentRuntime for Agent {
    fn is_sensitive_path(&self, path: &std::path::Path) -> bool {
        aidaemon_runtime::host::EngineContext::daemon_paths(self).is_sensitive(path)
    }

    fn execution_backend(&self) -> crate::execution::SharedExecutionBackend {
        aidaemon_runtime::host::TurnHost::execution_backend_handle(self)
    }

    async fn reload_provider(&self, config: &crate::config::AppConfig) -> anyhow::Result<String> {
        super::models::reload_provider(self, config).await
    }
}
