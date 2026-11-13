//! The state database is `aidaemon_runtime::state` (the SQLite end of the
//! `StateStore` facade and the state schema); kept at its old path.

pub use aidaemon_runtime::state::SqliteStateStore;

#[cfg(test)]
mod write_transaction_tests;
