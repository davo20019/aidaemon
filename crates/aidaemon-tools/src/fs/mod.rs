//! `read_file`, `write_file` and `edit_file`, over an [`ExecutionBackend`].
//!
//! [`FilePolicy`] holds what differs between applications: which paths are
//! refused, whether a change is backed up, and what is reported after a write.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use aidaemon_core::CallEffects;

use crate::backend::{BackendPath, ExecutionBackend};
use crate::path_class::PathClassifier;
use crate::workspace::Workspace;

mod edit;
mod guard;
mod read;
mod workspace_backend;
mod write;

pub use edit::EditFile;
pub use guard::ProtectedPathGuard;
pub use read::{MAX_READ_CHARS, ReadFile, render_read_file_output, render_read_file_output_within};
pub use workspace_backend::WorkspaceBackend;
pub use write::WriteFile;

/// Directories to skip during recursive file walks.
pub const DEFAULT_IGNORE_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".cache",
    ".parcel-cache",
    "coverage",
    ".turbo",
    ".gradle",
    ".idea",
    ".vs",
];

pub(crate) type SharedBackend = Arc<dyn ExecutionBackend>;

/// A tool's model-facing descriptions, for an application that words them
/// for its own tool set.
#[derive(Debug, Clone)]
pub struct ToolText {
    /// One-line description.
    pub short: String,
    /// The description in the tool's schema.
    pub schema: String,
}

/// What an application decides about the file tools.
#[async_trait]
pub trait FilePolicy: Send + Sync {
    /// Files `read_file` refuses because they hold secrets.
    fn is_secret(&self, path: &Path) -> bool;
    /// [`FilePolicy::is_secret`] without name-shape rules, for `.env.example`
    /// style templates, which are judged by where they live.
    fn is_secret_ignoring_name_shape(&self, path: &Path) -> bool;
    /// Paths the file tools must not change. [`ProtectedPathGuard`] enforces
    /// these three; add it to the agent.
    fn is_sensitive(&self, path: &Path) -> bool;

    /// The path a kernel access manifest declares for a path argument.
    fn declare_path(&self, path: &str) -> String {
        path.to_string()
    }
    /// Whether images are returned as attachments the caller can deliver.
    fn attaches_images(&self) -> bool {
        false
    }
    /// The tool named in hints that suggest running a command.
    fn shell_tool_name(&self) -> &str {
        "shell"
    }
    /// A sentence added after an append, on delivering the finished file.
    fn delivery_hint(&self) -> Option<&str> {
        None
    }
    /// Whether [`FilePolicy::backup_before_overwrite`] keeps a copy.
    fn keeps_backups(&self) -> bool {
        false
    }
    /// Keep a copy of `path` before it is replaced. Best effort: a failed
    /// backup never blocks the change.
    async fn backup_before_overwrite(&self, _backend: &SharedBackend, _path: &BackendPath) {}
    /// Text appended to a successful write, such as a syntax check.
    async fn post_write_diagnostics(
        &self,
        _backend: &dyn ExecutionBackend,
        _path: &Path,
    ) -> String {
        String::new()
    }
    /// The effects of a write call.
    fn write_effects(&self, _backend: &dyn ExecutionBackend, _arguments: &str) -> CallEffects {
        CallEffects::unknown()
    }
    /// Whether writing `path` changes configuration that later commands run
    /// (git configuration, for example).
    fn plants_configuration(&self, _path: &str) -> bool {
        false
    }
    /// Where to go instead when the file tools refuse `path`: an app's own
    /// files can have a tool of their own (its config tool, for example).
    fn refusal_hint(&self, _path: &Path) -> Option<String> {
        None
    }
}

/// The policy for tools confined to a [`Workspace`].
#[derive(Debug, Clone)]
pub struct WorkspaceFilePolicy {
    workspace: Workspace,
    classifier: PathClassifier,
}

impl WorkspaceFilePolicy {
    pub fn new(workspace: Workspace) -> Self {
        Self {
            workspace,
            classifier: PathClassifier::empty(),
        }
    }
}

impl FilePolicy for WorkspaceFilePolicy {
    fn is_secret(&self, path: &Path) -> bool {
        self.classifier.is_secret(path)
    }

    fn is_secret_ignoring_name_shape(&self, path: &Path) -> bool {
        self.classifier.is_secret_ignoring_name_shape(path)
    }

    fn is_sensitive(&self, path: &Path) -> bool {
        self.classifier.is_sensitive(path)
    }

    fn declare_path(&self, path: &str) -> String {
        self.workspace.declare(path)
    }
}

/// What a [`HostFilePolicy`] says and keeps, as the application words it.
#[derive(Debug, Clone, Default)]
pub struct HostFilePolicySettings {
    /// The tool named in hints that suggest running a command.
    pub shell_tool_name: String,
    /// A sentence added after an append, on delivering the finished file.
    pub delivery_hint: Option<String>,
    /// Where to go instead when the refused path is one of the
    /// application's own files.
    pub own_file_hint: Option<String>,
    /// Keep a copy of a file before it is replaced, under `backup_root`.
    pub keeps_backups: bool,
    /// Where backups go (`None`: none is kept).
    pub backup_root: Option<std::path::PathBuf>,
}

/// The policy for file tools on the host's own filesystem: secret and
/// sensitive paths (and the application's own files) by [`DaemonPaths`],
/// images as attachments, backups before an overwrite, a syntax check after
/// a write, and the effects of a write (git configuration among them).
///
/// [`DaemonPaths`]: crate::path_class::DaemonPaths
#[derive(Debug, Clone)]
pub struct HostFilePolicy {
    paths: crate::path_class::DaemonPaths,
    settings: HostFilePolicySettings,
}

impl HostFilePolicy {
    /// A policy protecting `paths`, worded by `settings`.
    pub fn new(paths: crate::path_class::DaemonPaths, settings: HostFilePolicySettings) -> Self {
        Self { paths, settings }
    }
}

#[async_trait]
impl FilePolicy for HostFilePolicy {
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
        self.settings
            .own_file_hint
            .clone()
            .filter(|_| self.paths.is_own_file(path))
    }

    fn shell_tool_name(&self) -> &str {
        &self.settings.shell_tool_name
    }

    fn delivery_hint(&self) -> Option<&str> {
        self.settings.delivery_hint.as_deref()
    }

    fn keeps_backups(&self) -> bool {
        self.settings.keeps_backups
    }

    async fn backup_before_overwrite(&self, backend: &SharedBackend, path: &BackendPath) {
        crate::fs_utils::backup_before_overwrite(
            backend,
            path,
            self.settings.backup_root.as_deref(),
        )
        .await;
    }

    async fn post_write_diagnostics(&self, backend: &dyn ExecutionBackend, path: &Path) -> String {
        crate::fs_utils::post_write_diagnostics(backend, path).await
    }

    fn write_effects(&self, backend: &dyn ExecutionBackend, arguments: &str) -> CallEffects {
        crate::command_effects::file_tool_write_effects(backend, arguments)
    }

    fn plants_configuration(&self, path: &str) -> bool {
        crate::command_effects::is_git_config_path(path)
    }
}

pub(crate) fn workspace_parts(workspace: Workspace) -> (SharedBackend, Arc<dyn FilePolicy>) {
    let policy = Arc::new(WorkspaceFilePolicy::new(workspace.clone()));
    (Arc::new(WorkspaceBackend::new(workspace)), policy)
}

/// The path argument of a file tool, under the names models commonly use.
pub(crate) fn path_arg(arguments: &Value) -> Option<&str> {
    ["path", "file_path", "file", "filename"]
        .iter()
        .find_map(|key| arguments.get(*key).and_then(Value::as_str))
}

/// Format file content with line numbers.
pub fn format_with_line_numbers(content: &str, start_line: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total = start_line + lines.len();
    let width = total.to_string().len().max(3);
    lines
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>width$} | {}", start_line + i + 1, line, width = width))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path_class::DaemonPaths;

    #[test]
    fn host_policy_points_own_files_to_the_hint_and_protects_them() {
        let own = Path::new("/synthetic/app/config.toml");
        let policy = HostFilePolicy::new(
            DaemonPaths::new(vec![own.to_path_buf()]),
            HostFilePolicySettings {
                shell_tool_name: "terminal".to_string(),
                own_file_hint: Some("use the config tool".to_string()),
                ..Default::default()
            },
        );
        assert!(policy.is_sensitive(own));
        assert_eq!(
            policy.refusal_hint(own).as_deref(),
            Some("use the config tool")
        );
        assert_eq!(policy.refusal_hint(Path::new("/synthetic/other.rs")), None);
        assert_eq!(policy.shell_tool_name(), "terminal");
        assert!(!policy.keeps_backups());
        assert!(policy.plants_configuration("/repo/.git/config"));
    }
}
