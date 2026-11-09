//! The immutable event store, its payloads and the run aggregate.
//!
//! Sits on a host's SQLite pool the way `aidaemon-store-sqlite` and
//! `aidaemon_memory::sqlite` do, and owns the `events` table contract.
