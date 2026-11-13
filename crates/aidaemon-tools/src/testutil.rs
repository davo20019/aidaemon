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

/// Sets an environment variable that no other test uses, and removes it
/// again when dropped.
pub(crate) struct ScopedVar(&'static str);

impl ScopedVar {
    pub(crate) fn set(name: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        // SAFETY: test-only, and the name is private to one test. std
        // serializes its own environment reads and writes; foreign code
        // calling getenv concurrently is a risk the test binary accepts.
        unsafe { std::env::set_var(name, value) };
        Self(name)
    }
}

impl Drop for ScopedVar {
    fn drop(&mut self) {
        // SAFETY: as in `ScopedVar::set`.
        unsafe { std::env::remove_var(self.0) };
    }
}
