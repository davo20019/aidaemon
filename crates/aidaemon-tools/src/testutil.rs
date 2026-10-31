//! Shared test fixtures.

use aidaemon_core::ToolContext;

use crate::workspace::Workspace;

pub(crate) fn ws() -> (tempfile::TempDir, Workspace) {
    let dir = tempfile::tempdir().expect("tempdir");
    let ws = Workspace::new(dir.path()).expect("workspace");
    (dir, ws)
}

pub(crate) fn ctx() -> ToolContext {
    ToolContext::new("test:synthetic-session", "call-1")
}
