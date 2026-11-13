//! Tools that join several domains (memory with scheduled work, the
//! automation surfaces), so they live where those crates meet.

pub mod manage_automations;
pub mod manage_memories;
pub mod set_reminder;

/// The domain types and tool vocabulary these tools name.
pub(crate) mod traits {
    pub use aidaemon_core::ChannelVisibility;
    pub use aidaemon_tools::semantics::*;
    pub use aidaemon_tools::OutcomeTool as Tool;
    pub use aidaemon_work::traits::*;
}
