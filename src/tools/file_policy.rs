//! The daemon's policy for the SDK file tools.

use std::path::Path;
use std::sync::Arc;

use aidaemon_core::CallEffects;
use aidaemon_tools::FilePolicy;
use async_trait::async_trait;

use super::{command_effects, fs_utils};
use crate::execution::{BackendPath, ExecutionBackend, SharedExecutionBackend};

struct DaemonFilePolicy;

#[async_trait]
impl FilePolicy for DaemonFilePolicy {
    fn is_secret(&self, path: &Path) -> bool {
        fs_utils::is_secret_path(path)
    }

    fn is_secret_ignoring_name_shape(&self, path: &Path) -> bool {
        fs_utils::is_secret_path_ignoring_name_shape(path)
    }

    fn is_sensitive(&self, path: &Path) -> bool {
        fs_utils::is_sensitive_path(path)
    }

    fn attaches_images(&self) -> bool {
        true
    }

    fn shell_tool_name(&self) -> &str {
        "terminal"
    }

    fn delivery_hint(&self) -> Option<&str> {
        Some("When the file is complete, deliver it with send_file instead of pasting it inline.")
    }

    fn keeps_backups(&self) -> bool {
        true
    }

    async fn backup_before_overwrite(&self, backend: &SharedExecutionBackend, path: &BackendPath) {
        fs_utils::backup_before_overwrite(backend, path).await;
    }

    async fn post_write_diagnostics(&self, backend: &dyn ExecutionBackend, path: &Path) -> String {
        fs_utils::post_write_diagnostics(backend, path).await
    }

    fn write_effects(&self, backend: &dyn ExecutionBackend, arguments: &str) -> CallEffects {
        command_effects::file_tool_write_effects(backend, arguments)
    }

    fn plants_configuration(&self, path: &str) -> bool {
        command_effects::is_git_config_path(path)
    }
}

pub(crate) fn daemon_file_policy() -> Arc<dyn FilePolicy> {
    Arc::new(DaemonFilePolicy)
}

/// The paths a call of a file tool reads and writes, as the guard judges them.
pub(crate) fn file_guard_paths(
    tool: &dyn crate::traits::Tool,
    arguments: &str,
) -> (Vec<String>, Vec<String>) {
    let manifest = tool.call_access_manifest(arguments);
    let values = |targets: Vec<crate::traits::ToolTargetHint>| {
        targets.into_iter().map(|target| target.value).collect()
    };
    (
        values(manifest.read_targets),
        values(manifest.write_targets),
    )
}

/// Run `tool` the way a turn does: the protected-path guard first.
#[cfg(test)]
pub(crate) async fn guarded_call(
    tool: &dyn crate::traits::Tool,
    arguments: &str,
) -> anyhow::Result<String> {
    let guard = aidaemon_tools::ProtectedPathGuard::new(
        crate::execution::active_execution_backend(),
        daemon_file_policy(),
    )
    .tool(tool.name(), tool.dispatch_facts().file_guard);
    let (reads, writes) = file_guard_paths(tool, arguments);
    if let Some(reason) = guard.refusal(tool.name(), &reads, &writes).await {
        anyhow::bail!(reason);
    }
    tool.call(arguments).await
}
