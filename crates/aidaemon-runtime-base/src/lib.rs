//! # aidaemon-runtime-base
//!
//! The lower half of aidaemon's turn runtime: what the turn and the engine
//! (`aidaemon-runtime-core`) are built on, with no reference to either. The
//! model runtime and side calls ([`llm`]), the host traits a turn runs on
//! ([`host`]), tool authority ([`authority`]), sub-agent coordination and
//! runs ([`coordination`], [`spawn`], [`spawn_tool`], [`run_state`]), the
//! untrusted-content rule ([`untrusted`]), the system prompt's core
//! ([`prompt`]), memory and history ([`memory`], [`history`]) and the SQLite
//! state store ([`state`]). `aidaemon-runtime-core` re-exports every module
//! here at its root; depend on `aidaemon-runtime`.

// Feature unification only (cargo-hakari); keeps the unused-dependency lint quiet.
use workspace_hack as _;

pub mod authority;
pub mod background_run;
pub mod budget_policy;
pub mod call_args;
pub mod chat_bridge;
pub mod consolidation;
pub mod context_budget;
pub mod coordination;
pub mod decisions;
pub mod device_cap;
pub mod eval;
pub mod helper_run;
pub mod history;
pub mod host;
pub mod known_resources;
pub mod llm;
pub mod markers;
pub mod media;
pub mod memory;
pub mod memory_port;
pub mod message_content;
pub mod mid_run;
pub mod opt_out;
pub mod owed_reports;
pub mod owner_level;
pub mod permissions;
pub mod policy;
pub mod prompt;
pub mod render;
pub mod run_state;
pub mod scope_check;
pub mod session_model;
pub mod settings;
pub mod spawn;
pub mod spawn_tool;
pub mod state;
pub mod steering;
pub mod stores;
pub mod terminal_grants;
pub mod text_cut;
pub mod tool_manifest;
pub mod turn_project;
pub mod turn_runtime;
pub mod untrusted;
pub mod untrusted_writes;
