//! Tools that join several domains (memory with scheduled work, the
//! automation surfaces), so they live where those crates meet.

pub mod command_patterns;
pub mod config_manager;
pub mod diagnose;
pub mod escalate;
pub mod image_generation;
pub mod manage_api;
pub mod manage_automations;
pub mod manage_memories;
pub mod policy_metrics;
pub mod report_blocker;
pub mod run_command;
pub mod search_history;
mod semantic_command_risk;
pub mod set_reminder;
pub mod terminal;

/// The domain types and tool vocabulary these tools name.
pub(crate) mod traits {
    pub use aidaemon_core::ChannelVisibility;
    pub use aidaemon_memory::records::{
        HistoryCoverage, HistoryMessage, HistoryScope, HistoryToolActivity, TaskBookends,
    };
    pub use aidaemon_memory::stores::HistorySearchStore;
    pub use aidaemon_tools::semantics::*;
    pub use aidaemon_tools::OutcomeTool as Tool;
    pub use aidaemon_work::stores::{GoalStore, NotificationStore};
    pub use aidaemon_work::traits::*;
}
