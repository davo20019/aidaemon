//! Shared contract types, now defined in the SDK crates. This module only
//! re-exports them under their old paths.

#[cfg(test)]
pub use aidaemon_channels::context::WorkspaceAccessLevel;
pub use aidaemon_channels::context::{ChannelContext, WorkspaceGrant};
#[cfg(test)]
pub use aidaemon_channels::hub::MediaKind;
pub use aidaemon_channels::hub::MediaMessage;
pub use aidaemon_core::ChannelVisibility;
#[allow(unused_imports)]
pub use aidaemon_memory::records::FactPrivacy;
#[cfg(test)]
pub use aidaemon_tools::semantics::PermissionMode;
#[cfg(test)]
pub use aidaemon_tools::semantics::RiskLevel;
pub use aidaemon_tools::semantics::{ApprovalKind, ApprovalResponse, UserRole};
#[cfg(test)]
pub use aidaemon_tools::status::StatusUpdate;
pub use aidaemon_tools::TerminalConfinement;
