//! Shared domain types + core interfaces (traits) used across the codebase.
//!
//! This module is intentionally kept as a thin re-export layer so that:
//! - `crate::traits::*` remains stable for call sites
//! - adding/changing one area (e.g. store traits) doesn't cause a full-file rebuild

mod channels;
mod conversation;
mod dialogue;
mod dynamic;
mod goals;
mod memory;
mod people;
mod provider;
mod self_correction;
mod tools;
mod trigger_event;

pub use channels::{Channel, ChannelCapabilities};
#[allow(unused_imports)]
pub use conversation::{
    extract_primary_message_content, first_primary_message_line, infer_message_annotations,
    message_content_is_structural_only, AttachmentProvenance, ConversationSummary, Message,
    MessageAnnotation, MessageAttachment, ToolCall,
};
pub use dialogue::{
    ActiveTaskRef, ActiveTaskStatus, AssistantTurnKind, AssistantTurnSummary, DialogueState,
    OpenQuestion, OpenRequest, OpenRequestStatus, QuestionKind, UserTurnKind, UserTurnSummary,
};
pub use dynamic::{
    CliAgentInvocation, DynamicBot, DynamicCliAgent, DynamicMcpServer, DynamicSkill,
    OAuthConnection, PendingOAuthFlow, SkillDraft,
};
pub use goals::{
    Goal, GoalSchedule, GoalTokenBudgetStatus, NotificationEntry, ScheduledRunHealth,
    ScheduledRunState, Task, TaskActivity,
};
#[allow(unused_imports)]
pub use memory::{
    BehaviorPattern, Episode, ErrorSolution, Expertise, ExtractedMemoryEntity,
    ExtractedMemoryGraph, ExtractedMemoryRelationship, Fact, MemoryHealthReport, Procedure,
    UserProfile,
};
pub use people::{Person, PersonFact};
pub use provider::{
    ChatOptions, ModelProvider, ProviderResponse, ResponseMode, TokenUsage, TokenUsageRecord,
    ToolChoiceMode,
};
#[allow(unused_imports)]
pub use self_correction::*;
#[allow(unused_imports)]
pub use tools::{
    AgentRole, ReadFileResultMetadata, ReadFileSelectionMetadata, SpecialistKind, Tool,
    ToolCallEffect, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolExecutionContext, ToolOutcomeStatus, ToolRole, ToolSemanticAffordances, ToolSemanticFacet,
    ToolSemanticScope, ToolTargetHint, ToolTargetHintKind, ToolVerificationMode, TruncationInfo,
};
pub use trigger_event::TriggerEvent;

mod state_store;
pub use state_store::*;

/// Import this in modules that call store-trait methods on concrete types.
///
/// `StateStore` is a facade (supertrait) used for trait objects, but Rust still
/// requires the defining trait to be in scope for method-call syntax.
pub mod store_prelude {
    #![allow(unused_imports)]
    pub use super::{
        ConversationSummaryStore, DialogueStateStore, DynamicBotStore, DynamicCliAgentStore,
        DynamicMcpServerStore, EpisodeStore, FactStore, GoalBudgetStore, GoalNotificationStore,
        GoalScheduleStore, GoalStore, HealthCheckStore, LearningStore, MessageStore,
        NotificationStore, OAuthStore, PeopleStore, PromptSnapshotStore, ScheduledRunStore,
        SessionChannelStore, SettingsStore, SkillStore, StateStore, TaskDispatchStore, TaskStore,
        TokenUsageStore,
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
