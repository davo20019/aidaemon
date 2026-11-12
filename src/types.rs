//! Shared contract types, now defined in the SDK crates. This module only
//! re-exports them under their old paths.

pub use aidaemon_channels::context::{ChannelContext, WorkspaceAccessLevel, WorkspaceGrant};
pub use aidaemon_channels::hub::{MediaKind, MediaMessage};
pub use aidaemon_core::ChannelVisibility;
#[allow(unused_imports)]
pub use aidaemon_memory::records::FactPrivacy;
#[cfg(test)]
pub use aidaemon_tools::semantics::PermissionMode;
pub use aidaemon_tools::semantics::{
    ApprovalKind, ApprovalResponse, GoalConfirmationStyle, RiskLevel, UserRole,
};
pub use aidaemon_tools::status::StatusUpdate;
pub use aidaemon_tools::TerminalConfinement;
