//! # aidaemon-runtime
//!
//! aidaemon's turn runtime: what runs one agent turn on top of the
//! `aidaemon-core` kernel, with aidaemon's behavior as the default.
//!
//! - [`tools`]: the built-in tools (terminal, CLI agents, memory, goals,
//!   browser handoff, computer use, ...).
//! - [`agent`]: the agent that runs turns with them.
//!
//! Every other module (the model runtime, engine, turn, approvals, state and
//! the rest) lives in `aidaemon-runtime-core` and is re-exported here, so
//! `aidaemon_runtime::turn::...` and `crate::turn::...` keep their paths.
//! The split lets an edit to a tool recompile this crate alone.

// Feature unification only (cargo-hakari); keeps the unused-dependency lint quiet.
use workspace_hack as _;

pub use aidaemon_runtime_core::*;

pub mod agent;
pub mod tools;
