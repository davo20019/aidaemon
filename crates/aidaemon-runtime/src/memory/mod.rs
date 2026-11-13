//! The daemon's memory system as the full `aidaemon-memory` system's host:
//! its background jobs' side (`jobs`), progressive extraction with the event
//! log as provenance, learning from finished tasks, people upkeep, and a new
//! memory system's intake of the retained conversations.

pub mod history_intake;
pub mod jobs;
pub mod people_intelligence;
pub mod progressive;
pub mod task_learning;
