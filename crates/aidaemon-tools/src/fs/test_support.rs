//! The file tools as the daemon assembles them, for the tests that moved
//! here from the daemon: its [`HostFilePolicy`] settings and tool wording
//! (the daemon's `startup::tool_config`), and a call behind the
//! protected-path guard, as the dispatcher makes it.

use std::sync::Arc;

use crate::backend::SharedExecutionBackend;
use crate::path_class::DaemonPaths;
use crate::testing::ToolCalls as _;
use crate::{
    EditFile, FilePolicy, HostFilePolicy, HostFilePolicySettings, OutcomeTool, ProtectedPathGuard,
    ReadFile, SearchFiles, ToolText, WriteFile,
};

/// The daemon's file policy on its host filesystem.
pub(crate) fn daemon_file_policy(paths: DaemonPaths) -> Arc<dyn FilePolicy> {
    Arc::new(HostFilePolicy::new(
        paths,
        HostFilePolicySettings {
            shell_tool_name: "terminal".to_string(),
            delivery_hint: Some(
                "When the file is complete, deliver it with send_file instead of pasting it inline."
                    .to_string(),
            ),
            own_file_hint: Some(
                "This is one of aidaemon's own files, which change only through manage_config: \
                 read the config with manage_config action 'read' (secrets hidden), and change one \
                 setting with 'set' or 'unset'."
                    .to_string(),
            ),
            keeps_backups: true,
            backup_root: file_backup_root(),
        },
    ))
}

/// Where the file tools keep pre-change copies of local files in tests.
pub(crate) fn file_backup_root() -> Option<std::path::PathBuf> {
    Some(std::env::temp_dir().join("aidaemon-test-file-backups"))
}

/// Run `tool` behind the protected-path guard with none of the daemon's own
/// files protected.
pub(crate) async fn guarded_call(
    tool: &dyn OutcomeTool,
    arguments: &str,
) -> anyhow::Result<String> {
    guarded_call_protecting(DaemonPaths::default(), tool, arguments).await
}

/// [`guarded_call`] with the daemon's own files set to `paths`.
pub(crate) async fn guarded_call_protecting(
    paths: DaemonPaths,
    tool: &dyn OutcomeTool,
    arguments: &str,
) -> anyhow::Result<String> {
    let guard = ProtectedPathGuard::new(
        crate::testing::active_execution_backend(),
        daemon_file_policy(paths),
    )
    .tool(tool.name(), tool.dispatch_facts().file_guard);
    let (reads, writes) = tool.guarded_paths(arguments);
    if let Some(reason) = guard.refusal(tool.name(), &reads, &writes).await {
        anyhow::bail!(reason);
    }
    tool.call(arguments).await
}

/// `read_file` as the daemon words it, without its own files protected.
pub(crate) struct ReadFileTool;

#[allow(clippy::new_ret_no_self)]
impl ReadFileTool {
    pub(crate) fn new(backend: SharedExecutionBackend) -> ReadFile {
        ReadFile::on_backend(backend, daemon_file_policy(DaemonPaths::default())).with_text(ToolText {
            short: "Read file contents with line numbers and metadata. Read a fitting file in full once; for large files, search first and then read one exact non-overlapping range.".to_string(),
            schema: "Read file contents with line numbers and metadata. Use this instead of terminal cat/head/tail. Read a file in full once when it fits the limit; oversized reads return the first page plus an exact continuation call to follow. For a large file with a known target, use search_files first, then read one exact surrounding range. Sequential ranges must not overlap; previously covered content is replayed from the task-local artifact.".to_string(),
        })
    }
}

/// `write_file` as the daemon words it, without its own files protected.
pub(crate) struct WriteFileTool;

#[allow(clippy::new_ret_no_self)]
impl WriteFileTool {
    pub(crate) fn new(backend: SharedExecutionBackend) -> WriteFile {
        WriteFile::on_backend(backend, daemon_file_policy(DaemonPaths::default())).with_text(ToolText {
            short: "Write or create a file with given content".to_string(),
            schema: "Write content to a file, creating it if it doesn't exist. Use this instead of terminal echo/cat for file creation. Creates a backup of existing files before overwriting. ALWAYS prefer write_file over terminal heredocs (cat > file << EOF) — heredocs trigger the approval flow.".to_string(),
        })
    }
}

/// `edit_file` as the daemon words it, without its own files protected.
pub(crate) struct EditFileTool;

#[allow(clippy::new_ret_no_self)]
impl EditFileTool {
    pub(crate) fn new(backend: SharedExecutionBackend) -> EditFile {
        EditFile::on_backend(backend, daemon_file_policy(DaemonPaths::default())).with_text(ToolText {
            short: "Find and replace text in a file".to_string(),
            schema: "Find and replace text in a file. Use this instead of terminal sed/awk. Shows context around the change. Fails safely if the text isn't found or is ambiguous. On not-found/ambiguous text, read_file the same path and retry once before asking the user.".to_string(),
        })
    }
}

/// `search_files` as the daemon words it, without its own files protected.
pub(crate) struct SearchFilesTool;

#[allow(clippy::new_ret_no_self)]
impl SearchFilesTool {
    pub(crate) fn new(backend: SharedExecutionBackend) -> SearchFiles {
        SearchFiles::on_backend(backend, daemon_file_policy(DaemonPaths::default())).with_text(ToolText {
            short: "Search for files by name pattern or content regex".to_string(),
            schema: "Search for files by name pattern (glob) and/or content (regex). Use this instead of terminal find/grep. Automatically skips .git, node_modules, target, etc.".to_string(),
        })
    }
}
