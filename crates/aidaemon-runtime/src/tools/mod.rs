//! Tools that join several domains (memory with scheduled work, the
//! automation surfaces), so they live where those crates meet.

pub mod diagnose;
pub mod escalate;
pub mod image_generation;
pub mod manage_api;
pub mod manage_automations;
pub mod manage_memories;
pub mod policy_metrics;
pub mod report_blocker;
pub mod search_history;
pub mod set_reminder;

/// The domain types and tool vocabulary these tools name.
pub(crate) mod traits {
    pub use aidaemon_core::ChannelVisibility;
    pub use aidaemon_memory::records::{
        HistoryCoverage, HistoryMessage, HistoryScope, HistoryToolActivity, TaskBookends,
    };
    pub use aidaemon_memory::stores::HistorySearchStore;
    pub use aidaemon_tools::semantics::*;
    pub use aidaemon_tools::OutcomeTool as Tool;
    pub use aidaemon_work::traits::*;
}
