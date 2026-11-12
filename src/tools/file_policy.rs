//! The daemon's policy for the SDK file tools.

#[cfg(test)]
use crate::traits::ToolCalls as _;
use std::path::Path;
use std::sync::Arc;

use aidaemon_core::CallEffects;
use aidaemon_tools::FilePolicy;
use async_trait::async_trait;

use super::{command_effects, fs_utils};
use crate::execution::{BackendPath, ExecutionBackend, SharedExecutionBackend};

pub use aidaemon_tools::path_class::DaemonPaths;

struct DaemonFilePolicy {
    paths: DaemonPaths,
}

#[async_trait]
impl FilePolicy for DaemonFilePolicy {
    fn is_secret(&self, path: &Path) -> bool {
        self.paths.is_secret(path)
    }

    fn is_secret_ignoring_name_shape(&self, path: &Path) -> bool {
        self.paths.is_secret_ignoring_name_shape(path)
    }

    fn is_sensitive(&self, path: &Path) -> bool {
        self.paths.is_sensitive(path)
    }

    fn attaches_images(&self) -> bool {
        true
    }

    fn refusal_hint(&self, path: &Path) -> Option<String> {
        self.paths.is_own_file(path).then(|| {
            "This is one of aidaemon's own files, which change only through manage_config: \
             read the config with manage_config action 'read' (secrets hidden), and change one \
             setting with 'set' or 'unset'."
                .to_string()
        })
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
        fs_utils::backup_before_overwrite(backend, path, file_backup_root().as_deref()).await;
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

pub(crate) fn daemon_file_policy(paths: DaemonPaths) -> Arc<dyn FilePolicy> {
    Arc::new(DaemonFilePolicy { paths })
}

/// Run `tool` the way a turn does: the protected-path guard first.
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
#[cfg(not(test))]
pub(crate) fn file_backup_root() -> Option<std::path::PathBuf> {
    Some(dirs::home_dir()?.join(".aidaemon").join("file-backups"))
}

/// Tests never write into the real home directory.
#[cfg(test)]
pub(crate) fn file_backup_root() -> Option<std::path::PathBuf> {
    Some(std::env::temp_dir().join("aidaemon-test-file-backups"))
}

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
