//! The daemon's memory system lives in the crates: the store, recall and
//! jobs in `aidaemon-memory`, its host adapters in `aidaemon_runtime::memory`
//! and retention in `aidaemon_runtime::state::retention`. Kept at its old
//! path.

pub use aidaemon_memory::embeddings;
#[allow(unused_imports)]
pub use aidaemon_runtime::memory::{
    history_intake, jobs as jobs_host, people_intelligence, progressive, task_learning,
};
pub use aidaemon_runtime::state::retention;

pub mod procedures;

/// The daemon's memory store over its state database.
#[allow(unused_imports)]
pub(crate) use aidaemon_runtime::state::sqlite_memory;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod comprehensive_tests;

#[cfg(test)]
mod jobs_tests;
