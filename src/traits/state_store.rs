//! The store traits, re-exported from the crates that own them, plus what
//! the app assembles for itself: the `StateStore` facade over every slice,
//! and the composite that spans crates for a tool still in `src/`
//! (`SkillsStore`: memory + settings, moves with skills in 9.6). The
//! database liveness probe (`HealthCheckStore`) serves the daemon's own
//! heartbeat.

use async_trait::async_trait;

pub use aidaemon_channels::hub::{DynamicBotStore, SessionChannelStore};
pub use aidaemon_events::provider::TokenUsageStore;
pub use aidaemon_events::stores::{
    ConversationSummaryStore, DialogueStateStore, MessageStore, PromptSnapshotStore,
};
pub use aidaemon_mcp::registry::{DynamicMcpServerStore, McpRegistryStore};
pub use aidaemon_memory::skills::SkillStore;
pub use aidaemon_memory::stores::{EpisodeStore, FactStore, PeopleStore};
pub use aidaemon_tools::services::{
    DynamicCliAgentStore, OAuthGatewayStore, OAuthStore, SettingsStore,
};
#[allow(unused_imports)]
pub use aidaemon_work::stores::*;

/// Learning storage lives in the SDK memory crate.
pub use aidaemon_memory::LearningStore;

/// Database health check — verifies the connection is alive.
#[async_trait]
pub trait HealthCheckStore: Send + Sync {
    async fn health_check(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Focused persistence contract used by skill startup and management.
pub trait SkillsStore: SkillStore + SettingsStore {}

impl<T> SkillsStore for T where T: SkillStore + SettingsStore {}

/// Facade trait kept for backwards compatibility.
///
/// This lets call sites keep using `Arc<dyn StateStore>`, while new code can
/// depend on focused store traits like `FactStore` or `PeopleStore`.
pub trait StateStore:
    Send
    + Sync
    + MessageStore
    + DialogueStateStore
    + super::HistorySearchStore
    + GoalTraceStore
    + MandateToolStore
    + aidaemon_work::scheduler::SchedulerStore
    + McpRegistryStore
    + SkillsStore
    + ReportBlockerStore
    + ScheduledGoalRunsStore
    + OAuthGatewayStore
    + TokenUsageStore
    + SkillStore
    + DynamicBotStore
    + SessionChannelStore
    + DynamicMcpServerStore
    + DynamicCliAgentStore
    + SettingsStore
    + OAuthStore
    + GoalStore
    + MandateStore
    + TaskStore
    + GoalScheduleStore
    + GoalBudgetStore
    + ScheduledRunStore
    + TaskDispatchStore
    + WorkCoordinationStore
    + GoalNotificationStore
    + ConversationSummaryStore
    + HealthCheckStore
    + NotificationStore
    + PromptSnapshotStore
{
}

impl<T> StateStore for T where
    T: Send
        + Sync
        + MessageStore
        + DialogueStateStore
        + super::HistorySearchStore
        + GoalTraceStore
        + MandateToolStore
        + McpRegistryStore
        + SkillsStore
        + ReportBlockerStore
        + ScheduledGoalRunsStore
        + OAuthGatewayStore
        + TokenUsageStore
        + SkillStore
        + DynamicBotStore
        + SessionChannelStore
        + DynamicMcpServerStore
        + DynamicCliAgentStore
        + SettingsStore
        + OAuthStore
        + GoalStore
        + MandateStore
        + TaskStore
        + GoalScheduleStore
        + GoalBudgetStore
        + ScheduledRunStore
        + TaskDispatchStore
        + WorkCoordinationStore
        + GoalNotificationStore
        + ConversationSummaryStore
        + HealthCheckStore
        + NotificationStore
        + PromptSnapshotStore
{
}
