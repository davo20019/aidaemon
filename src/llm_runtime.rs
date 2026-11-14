//! The model runtime and the side-call gateway live in `aidaemon-runtime`;
//! the old paths stay.

pub use aidaemon_runtime::llm::*;

#[cfg(test)]
#[path = "llm_runtime/door_tests.rs"]
mod door_tests;
