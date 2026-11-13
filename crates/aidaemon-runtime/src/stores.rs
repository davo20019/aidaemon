//! The agent host's store: one facade over every store slice the turn
//! runtime and its parts take, so a host passes a single
//! `Arc<dyn StateStore>` where each part asks for its own trait.
//!
//! Each slice's trait lives with the crate that uses it; this module only
//! joins them. New code should depend on the focused trait it needs
//! (`FactStore`, `GoalStore`, ...), not on the facade.

pub use aidaemon_channels::hub::{DynamicBotStore, SessionChannelStore};
pub use aidaemon_events::provider::TokenUsageStore;
pub use aidaemon_events::stores::{
    ConversationSummaryStore, DialogueStateStore, MessageStore, PromptSnapshotStore,
};
pub use aidaemon_mcp::registry::{DynamicMcpServerStore, McpRegistryStore};
pub use aidaemon_memory::skills::SkillStore;
pub use aidaemon_memory::stores::{EpisodeStore, FactStore, HistorySearchStore, PeopleStore};
pub use aidaemon_memory::LearningStore;
pub use aidaemon_skills::SkillsStore;
pub use aidaemon_tools::services::{
    DynamicCliAgentStore, OAuthGatewayStore, OAuthStore, SettingsStore,
};
pub use aidaemon_work::stores::*;

/// Every store slice the agent host hands its parts, as one trait object.
///
/// Implemented for any type that implements all of the slices.
pub trait StateStore:
    Send
    + Sync
    + crate::turn::compaction::CompactionStore
    + MessageStore
    + DialogueStateStore
    + HistorySearchStore
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
        + HistorySearchStore
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

/// Import this where store-trait methods are called on a concrete store:
/// a method is only callable with its defining trait in scope.
pub mod prelude {
    pub use super::{
        ConversationSummaryStore, DialogueStateStore, DynamicBotStore, DynamicCliAgentStore,
        DynamicMcpServerStore, EpisodeStore, FactStore, GoalBudgetStore, GoalNotificationStore,
        GoalScheduleStore, GoalStore, GoalTraceStore, HealthCheckStore, HistorySearchStore,
        LearningStore, MandateStore, MandateToolStore, McpRegistryStore, MessageStore,
        NotificationStore, OAuthGatewayStore, OAuthStore, PeopleStore, PromptSnapshotStore,
        ReportBlockerStore, ScheduledGoalRunsStore, ScheduledRunStore, SessionChannelStore,
        SettingsStore, SkillStore, SkillsStore, StateStore, TaskDispatchStore, TaskStore,
        TokenUsageStore, WorkCoordinationStore,
    };
}
