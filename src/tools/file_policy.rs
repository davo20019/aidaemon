//! The file tools' policy is `aidaemon_tools::HostFilePolicy`, set up in
//! `startup::tool_config`. This module keeps the test helpers that run a file
//! tool behind the protected-path guard, as the dispatcher does.

#[cfg(test)]
use crate::traits::ToolCalls as _;
#[cfg(test)]
use std::path::Path;

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

/// Where the file tools keep pre-change copies of local files.
#[cfg(test)]
pub(crate) use crate::startup::tool_config::file_backup_root;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitive_paths_are_classified() {
        let paths = DaemonPaths::default();
        for sensitive in [
            "/home/user/.ssh/id_rsa",
            "/home/user/.env",
            "/repo/.env.production/value",
            "/repo/.dev.vars.local",
            "/repo/.git/config",
            "/repo/certificates/app.PEM",
            "/repo/terraform.tfstate.backup",
            "/home/user/.gnupg/key",
        ] {
            assert!(paths.is_sensitive(Path::new(sensitive)), "{sensitive}");
        }
        assert!(!paths.is_sensitive(Path::new("/home/user/code/main.rs")));
        assert!(!paths.is_sensitive(Path::new("/repo/src/environment.rs")));
    }

    #[test]
    fn the_daemons_own_files_are_protected_only_when_given() {
        let own = Path::new("/synthetic/daemon/config.toml");
        assert!(!DaemonPaths::default().is_sensitive(own));
        let paths = DaemonPaths::new(vec![own.to_path_buf()]);
        assert!(paths.is_sensitive(own));
        assert!(paths.is_secret(own));
    }
}
