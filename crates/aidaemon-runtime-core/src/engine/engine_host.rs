//! What the turn engine needs of its host beyond a turn host: the engine's
//! own settings, the owner's approvals, the specialist registry, and building
//! child agents. Defined here, above the host, so the host and the modules
//! below it never name engine types.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use aidaemon_channels::context::ChannelContext;
use aidaemon_tools::semantics::UserRole;
use aidaemon_tools::status::StatusUpdate;
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::host::TurnHost;

/// The agent the turn engine runs on.
#[async_trait]
pub trait EngineHost: TurnHost {
    /// How turns run on the SDK kernel; shared by every agent in the tree.
    fn engine_settings(&self) -> &std::sync::Arc<crate::engine::settings::SdkEngineSettings>;
    /// The owner's approvals: pending cards and session grants.
    fn owner_approvals(&self) -> &std::sync::Arc<crate::approvals::OwnerApprovals>;
    /// The specialist registry this agent's children are made from.
    fn specialists(&self) -> &std::sync::Arc<crate::specialists::SpecialistRegistry>;
    /// Build a child of this agent: a fresh agent sharing this one's model
    /// runtime, stores, hub, approvals and sub-agent registry.
    async fn create_child(&self, spec: crate::spawn::ChildSpec) -> std::sync::Arc<dyn ChildAgent>;
}

/// A child agent its host built: a turn host of its own that runs turns.
#[async_trait]
pub trait ChildAgent: TurnHost {
    /// Run one turn of the child.
    async fn handle_message(
        &self,
        session_id: &str,
        user_text: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
    ) -> anyhow::Result<String>;

    /// Set once any turn of the child read (or started with) untrusted
    /// content, even when the taint could not be recorded.
    fn read_untrusted(&self) -> Arc<std::sync::atomic::AtomicBool>;
}
