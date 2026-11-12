//! Workspace-bound execution backends. They live in
//! `aidaemon_tools::execution`; this module keeps the daemon's paths to them.

#[allow(unused_imports)]
pub use aidaemon_tools::{
    shell_quote, BackendDirEntry, BackendFileType, BackendKind, BackendMetadata, BackendPath,
    CommandSpec, ExecutionBackend, ExecutionOutput, ExecutionRequest, LexicalPaths, ProcessHandle,
    SpawnedProcess, WriteMode,
};

pub use aidaemon_tools::backend::{normalize_path_lexically, SharedExecutionBackend};

/// The default local backend for isolated unit tests that construct a tool
/// without running startup. Production code never reads this: startup builds
/// the one backend for the process and passes it to whatever needs it.
#[cfg(test)]
pub use aidaemon_tools::testing::active_execution_backend;

pub use aidaemon_tools::execution::build_execution_backend;
#[cfg(test)]
pub use aidaemon_tools::execution::local_backend;
