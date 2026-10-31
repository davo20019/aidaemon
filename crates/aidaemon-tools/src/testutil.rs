//! Shared test fixtures.

use aidaemon_core::{CancellationToken, RunId, ToolContext};

use crate::workspace::Workspace;

pub(crate) fn ws() -> (tempfile::TempDir, Workspace) {
    let dir = tempfile::tempdir().expect("tempdir");
    let ws = Workspace::new(dir.path()).expect("workspace");
    (dir, ws)
}

pub(crate) fn ctx() -> ToolContext {
    ToolContext {
        run_id: RunId::new(),
        session_id: "test:synthetic-session".into(),
        call_id: "call-1".into(),
        turn: 1,
        cancel: CancellationToken::new(),
    }
}
