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
pub use read::{render_read_file_output, render_read_file_output_within, ReadFile, MAX_READ_CHARS};
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
