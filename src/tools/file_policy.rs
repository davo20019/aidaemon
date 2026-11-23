//! The file tools' policy is `aidaemon_tools::HostFilePolicy`, set up in
//! `startup::tool_config`. This module keeps the test helpers that run a file
//! tool behind the protected-path guard, as the dispatcher does.

#[cfg(test)]
use crate::traits::ToolCalls as _;

pub use aidaemon_tools::path_class::DaemonPaths;

pub(crate) use crate::startup::tool_config::daemon_file_policy;

#[cfg(test)]
pub(crate) async fn guarded_call(
    tool: &dyn crate::traits::Tool,
    arguments: &str,
) -> anyhow::Result<String> {
    guarded_call_protecting(DaemonPaths::default(), tool, arguments).await
}

/// [`guarded_call`] with the daemon's own files set to `paths`.
#[cfg(test)]
pub(crate) async fn guarded_call_protecting(
    paths: DaemonPaths,
    tool: &dyn crate::traits::Tool,
    arguments: &str,
) -> anyhow::Result<String> {
    let guard = aidaemon_tools::ProtectedPathGuard::new(
        crate::execution::active_execution_backend(),
        daemon_file_policy(paths),
    )
    .tool(tool.name(), tool.dispatch_facts().file_guard);
    let (reads, writes) = tool.guarded_paths(arguments);
    if let Some(reason) = guard.refusal(tool.name(), &reads, &writes).await {
        anyhow::bail!(reason);
    }
    tool.call(arguments).await
}
