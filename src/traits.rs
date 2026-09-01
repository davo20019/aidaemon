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
mod history;
mod mandates;
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
#[allow(unused_imports)]
pub use dialogue::{
    ActiveTaskRef, ActiveTaskStatus, AdoptedEvidenceBinding, AssistantTurnKind,
    AssistantTurnSummary, DialogueState, OpenQuestion, OpenRequest, OpenRequestStatus,
    QuestionKind, RequestCollectionCoverageRequirement, RequestCompletionContract,
    RequestDispatchStopRule, RequestEvidenceRequirement, RequestForbiddenAction,
    RequestObservationTarget, RequestReceiptPredicate, RequestResponseContract, RequestTaskKind,
    RequestVerificationTarget, RequestVerificationTargetKind, RequestedOutcomeCondition,
    UserRequestBinding, UserTurnKind, UserTurnSummary,
};
pub use dynamic::{
    CliAgentInvocation, DynamicBot, DynamicCliAgent, DynamicMcpServer, DynamicSkill,
    OAuthConnection, PendingOAuthFlow, SkillDraft,
};
pub(crate) use goals::{task_execution_graph, ExpiredAttemptRecovery};
pub use goals::{
    Goal, GoalRun, GoalSchedule, GoalTokenBudgetStatus, HandoffArtifact, NotificationEntry,
    ScheduledFailureKind, ScheduledRecoveryDisposition, ScheduledRecoveryState, ScheduledRunHealth,
    ScheduledRunState, Task, TaskActivity, TaskAttempt, TaskAttemptPatch, TaskHandoff,
    TaskJournalEntry, TaskWorkspace, WorkGoalSummary, WorkProject, WorkTaskSummary, WorkerProfile,
    DEFAULT_PROJECT_ID,
};
pub use history::{
    HistoryCoverage, HistoryMessage, HistoryScope, HistorySearchStore, ProjectionStats,
    TaskBookends,
};
#[allow(unused_imports)]
pub use mandates::{
    is_runtime_fallback_rationale, Intention, IntentionStatus, Mandate, MandateActivityLevel,
    MandateAuthority, MandateAuthorityGrant, MandateAutonomyMode, MandateDecisionCycle,
    MandateDecisionOutcome, MandateFinalizationRejectReason, MandateFinalizationStaleReason,
    MandateLearningNote, MandateMutationAttempt, MandateMutationAttemptStatus,
    MandateMutationDispatchClaim, MandateMutationEvidence, MandateMutationOutcomeProjection,
    MandateMutationQuotaBlockReason, MandateMutationQuotaState, MandateMutationReservation,
    MandateMutationTarget, MandateObjectiveControl, MandateObjectiveMeasurement,
    MandateOperatingUpdates, MandateOperationKind, MandateOperationScope,
    MandateReconciliationReason, MandateReconciliationResolution, MandateRunFinalizationRequest,
    MandateRunFinalizationResult, MandateRunNotification, MandateRunNotificationKind,
    MandateRunProofCounts, MandateStatus, MandateStrategyRevision, MandateStrategyRevisionKind,
    MandateStrategySnapshot, MandateSuspension, MandateSuspensionKind, MandateTerminationKind,
    MandateWakeSignal, MandateWakeSignalKind, ObjectiveMetricDirection,
    SAFE_FALLBACK_WAIT_RATIONALE,
};
#[allow(unused_imports)]
pub use memory::{
    BehaviorPattern, Episode, ErrorSolution, Expertise, ExtractedMemoryEntity,
    ExtractedMemoryGraph, ExtractedMemoryRelationship, Fact, MemoryHealthReport,
    PersonalAliasCandidate, PersonalEntityCandidate, PersonalFactCandidate, PersonalMemoryWrite,
    PersonalMemoryWriteResult, PersonalRelationshipCandidate, Procedure, UserProfile,
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
    semantics_for_exact_read_actions, AgentRole, AuthorizationPreflightRecord,
    AuthorizationPreflightStatus, DurableReplayDecision, EvidenceAuthority, EvidencePurpose,
    EvidenceTemporalScope, PreparedToolInvocation, ReadFileResultMetadata,
    ReadFileSelectionMetadata, RuntimePreparationFailure, ScopeEscalation, SpecialistKind, Tool,
    ToolAccessDenial, ToolAccessEnforcement, ToolArgumentContractViolation, ToolCallAccessManifest,
    ToolCallEffect, ToolCallMetadata, ToolCallOperation, ToolCallOutcome, ToolCallSemantics,
    ToolCapabilities, ToolCollectionCompleteness, ToolCollectionObservation,
    ToolEvidenceCapability, ToolExecutionContext, ToolInvocationStage, ToolMutationEffects,
    ToolObservationEvidence, ToolOutcomeStatus, ToolReceiptKind, ToolResultCompleteness,
    ToolResultContentSource, ToolResultPresentation, ToolResultProvenance, ToolRole,
    ToolSemanticAffordances, ToolSemanticFacet, ToolSemanticScope, ToolTargetHint,
    ToolTargetHintKind, ToolVerificationMode, TruncationInfo,
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
        GoalScheduleStore, GoalStore, GoalTraceStore, HealthCheckStore, HistorySearchStore,
        LearningStore, MandateStore, MandateToolStore, McpRegistryStore, MessageStore,
        NotificationStore, OAuthGatewayStore, OAuthStore, PeopleStore, PeopleToolStore,
        PromptSnapshotStore, ReportBlockerStore, ScheduledGoalRunsStore, ScheduledRunStore,
        SessionChannelStore, SettingsStore, SkillStore, SkillsStore, StateStore, TaskDispatchStore,
        TaskStore, TokenUsageStore, WorkCoordinationStore,
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
