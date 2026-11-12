#![recursion_limit = "256"]
//! Goals, mandates, tasks, work coordination and notifications: the domain
//! types, the store traits and their SQLite implementation.

pub mod admissibility;
pub mod authority;
pub mod chat_command;
pub mod goal_tokens;
pub mod goals;
pub mod mandates;
pub mod plans;
pub mod reminders;
pub mod schedule;
pub mod scheduler;
pub mod schema;
pub mod sqlite;
pub mod stores;
pub mod tools;
pub mod triggers;

/// Every domain type and store trait, under one path.
pub mod traits {
    pub use crate::goals::*;
    pub use crate::mandates::*;
    pub use crate::stores::*;
    pub use aidaemon_tools::semantics::*;
}

/// The mandate protocol's own control tools. Their receipts record
/// bookkeeping, never an observation of the world, so they are never cited as
/// decision or measurement evidence.
pub const MANDATE_PROTOCOL_TOOLS: &[&str] = &[
    "manage_mandates",
    "manage_goal_tasks",
    "track_requirements",
    "report_blocker",
];
