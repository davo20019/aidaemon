//! # aidaemon-tools (experimental)
//!
//! A small default tool set for [`aidaemon_core`] agents, built on lessons
//! from agent-computer interfaces (SWE-agent, Claude Code, Aider): few
//! non-overlapping tools, bounded output, actionable errors, and validation
//! before mutation.
//!
//! | Tool | Name | Annotations |
//! |---|---|---|
//! | [`ReadFile`] | `read_file` | read-only |
//! | [`WriteFile`] | `write_file` | write (idempotent) |
//! | [`EditFile`] | `edit_file` | write |
//! | [`Glob`] | `glob` | read-only |
//! | [`Grep`] | `grep` | read-only |
//! | [`Shell`] | `shell` | destructive |
//! | [`WebFetch`] | `web_fetch` | read-only |
//! | [`Checklist`] | `track_requirements` | write (agent state only) |
//!
//! Every file tool is confined to a [`Workspace`]. The shell runs through a
//! [`Sandbox`]; containment (sandboxing) lives there, while approval
//! stays a policy concern for [`aidaemon_core::Hook`]s. No tool contains
//! approval logic.
//!
//! ```no_run
//! use aidaemon_tools::{default_tools, Checklist, Workspace};
//! # fn demo(provider: impl aidaemon_core::Provider + 'static) -> Result<(), Box<dyn std::error::Error>> {
//! let workspace = Workspace::new("./my-project")?;
//! let (checklist, gate) = Checklist::with_gate();
//! let mut builder = aidaemon_core::Agent::builder("coder").allow_unguarded()
//!     .model("some-model")
//!     .provider(provider)
//!     .tool(checklist)
//!     .hook(gate);
//! for tool in default_tools(workspace) {
//!     builder = builder.tool_arc(tool);
//! }
//! let agent = builder.build()?;
//! # let _ = agent; Ok(()) }
//! ```

use std::sync::Arc;

use aidaemon_core::Tool;

pub mod backend;
mod checklist;
mod exec;
mod fs;
pub mod outcome;
pub mod path_class;
mod search;
mod search_files;
pub mod semantics;
pub mod services;
mod shell;
mod util;
pub mod web;
mod workspace;

#[cfg(test)]
mod testutil;

pub use backend::{
    shell_quote, BackendDirEntry, BackendFileType, BackendKind, BackendMetadata, BackendPath,
    CommandSpec, ExecutionBackend, ExecutionOutput, ExecutionRequest, ProcessHandle,
    SpawnedProcess, WriteMode,
};
pub use checklist::{Checklist, ChecklistGate, ChecklistItem, ChecklistState, ItemStatus};
pub use exec::{
    best_available_sandbox, BubblewrapSandbox, CommandOutput, DockerSandbox, ExecError,
    LocalSandbox, Sandbox, SandboxCommand, SeatbeltSandbox,
};
pub use fs::{
    format_with_line_numbers, render_read_file_output, render_read_file_output_within, EditFile,
    FilePolicy, ReadFile, ToolText, WorkspaceBackend, WorkspaceFilePolicy, WriteFile,
    DEFAULT_IGNORE_DIRS, MAX_READ_CHARS,
};
pub use outcome::OutcomeTool;
pub use path_class::PathClassifier;
pub use search::{Glob, Grep};
pub use search_files::SearchFiles;
pub use shell::{Shell, DEFAULT_ENV_ALLOWLIST};
pub use util::sniff_image_mime;
pub use web::WebFetch;
pub use workspace::{PathError, Workspace};

/// The standard set: `read_file`, `write_file`, `edit_file`, `glob`, `grep`,
/// `shell` (with [`best_available_sandbox`]) and `web_fetch`.
///
/// `track_requirements` is not included because it is only useful together
/// with its stop hook; add both with [`Checklist::with_gate`].
pub fn default_tools(workspace: Workspace) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(ReadFile::new(workspace.clone())),
        Arc::new(WriteFile::new(workspace.clone())),
        Arc::new(EditFile::new(workspace.clone())),
        Arc::new(Glob::new(workspace.clone())),
        Arc::new(Grep::new(workspace.clone())),
        Arc::new(Shell::sandboxed(workspace)),
        Arc::new(WebFetch::new()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_set_has_strict_unique_specs() {
        let (_d, ws) = testutil::ws();
        let tools = default_tools(ws);
        let mut names: Vec<String> = tools.iter().map(|t| t.spec().name).collect();
        names.push(Checklist::default().spec().name);
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
        for tool in tools
            .iter()
            .cloned()
            .chain([Arc::new(Checklist::default()) as Arc<dyn Tool>])
        {
            let spec = tool.spec();
            assert!(!spec.description.is_empty(), "{}", spec.name);
            assert_eq!(spec.parameters["type"], "object", "{}", spec.name);
            assert_eq!(
                spec.parameters["additionalProperties"],
                serde_json::json!(false),
                "{}",
                spec.name
            );
            // Every required key is a declared property.
            for key in spec.parameters["required"].as_array().unwrap() {
                assert!(
                    spec.parameters["properties"]
                        .get(key.as_str().unwrap())
                        .is_some(),
                    "{}: {key}",
                    spec.name
                );
            }
        }
        let read_only: Vec<bool> = tools.iter().map(|t| t.annotations().read_only).collect();
        assert_eq!(read_only, [true, false, false, true, true, false, true]);
    }
}
