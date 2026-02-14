//! Shared domain types + core interfaces (traits) used across the codebase.
//!
//! This module is intentionally kept as a thin re-export layer so that:
//! - `crate::traits::*` remains stable for call sites
//! - adding/changing one area (e.g. store traits) doesn't cause a full-file rebuild

mod channels;
mod conversation;
mod dynamic;
mod memory;
mod people;
mod provider;
mod tools;
mod trigger_event;
mod v3;

pub use channels::{Channel, ChannelCapabilities};
pub use conversation::{ConversationSummary, Message, ToolCall};
pub use dynamic::{
    CliAgentInvocation, DynamicBot, DynamicCliAgent, DynamicMcpServer, DynamicSkill, OAuthConnection,
    SkillDraft,
};
pub use memory::{BehaviorPattern, Episode, ErrorSolution, Expertise, Fact, Goal, Procedure, UserProfile};
pub use people::{Person, PersonFact};
pub use provider::{ModelProvider, ProviderResponse, TokenUsage, TokenUsageRecord};
pub use tools::{AgentRole, Tool, ToolCapabilities, ToolRole};
pub use trigger_event::TriggerEvent;
pub use v3::{GoalTokenBudgetStatus, GoalV3, NotificationEntry, TaskActivityV3, TaskV3};

mod state_store;
pub use state_store::*;

/// Import this in modules that call store-trait methods on concrete types.
///
/// `StateStore` is a facade (supertrait) used for trait objects, but Rust still
/// requires the defining trait to be in scope for method-call syntax.
pub mod store_prelude {
    #![allow(unused_imports)]
    pub use super::{
        ConversationSummaryStore, DynamicBotStore, DynamicCliAgentStore, DynamicMcpServerStore,
        EpisodeStore, FactStore, HealthCheckStore, LearningStore, LegacyGoalStore, MessageStore,
        NotificationStore, OAuthStore, PeopleStore, SessionChannelStore, SettingsStore, SkillStore,
        StateStore, TokenUsageStore, V3Store,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_capabilities_default_is_conservative() {
        let caps = ToolCapabilities::default();
        assert!(!caps.read_only);
        assert!(!caps.external_side_effect);
        assert!(caps.needs_approval);
        assert!(!caps.idempotent);
        assert!(!caps.high_impact_write);
    }
}

