//! Shared domain types + core interfaces (traits) used across the codebase.
//!
//! This module is intentionally kept as a thin re-export layer so that:
//! - `crate::traits::*` remains stable for call sites
//! - adding/changing one area (e.g. store traits) doesn't cause a full-file rebuild

mod channels;
mod dynamic;
mod history;
mod memory;
mod people;
mod tools;

#[allow(unused_imports)]
pub use aidaemon_events::conversation::{
    extract_primary_message_content, first_primary_message_line, infer_message_annotations,
    message_content_is_structural_only, AttachmentProvenance, ConversationSummary, Message,
    MessageAnnotation, MessageAttachment, ToolCall,
};
#[allow(unused_imports)]
pub use aidaemon_events::dialogue::{
    ActiveTaskRef, ActiveTaskStatus, AdoptedEvidenceBinding, AssistantTurnKind,
    AssistantTurnSummary, DialogueState, OpenQuestion, OpenRequest, OpenRequestStatus,
    QuestionKind, RequestCollectionCoverageRequirement, RequestCompletionContract,
    RequestDispatchStopRule, RequestEvidenceRequirement, RequestForbiddenAction,
    RequestObservationTarget, RequestReceiptPredicate, RequestResponseContract, RequestTaskKind,
    RequestVerificationTarget, RequestVerificationTargetKind, RequestedOutcomeCondition,
    UserRequestBinding, UserTurnKind, UserTurnSummary,
};
#[allow(unused_imports)]
pub use aidaemon_events::provider::{
    ChatOptions, ProviderResponse, ResponseMode, TokenUsage, TokenUsageRecord, ToolChoiceMode,
};
#[allow(unused_imports)]
pub(crate) use aidaemon_work::goals::{task_execution_graph, ExpiredAttemptRecovery};
#[allow(unused_imports)]
pub use aidaemon_work::goals::{
    BackgroundSpawn, Goal, GoalRun, GoalSchedule, GoalTokenBudgetStatus, HandoffArtifact,
    NotificationEntry, ScheduledFailureKind, ScheduledRecoveryDisposition, ScheduledRecoveryState,
    ScheduledRunHealth, ScheduledRunState, Task, TaskActivity, TaskAttempt, TaskAttemptPatch,
    TaskHandoff, TaskJournalEntry, TaskWorkspace, WorkGoalSummary, WorkProject, WorkTaskSummary,
    WorkerProfile, DEFAULT_PROJECT_ID, REMOTE_CHANGE_ACTIVITY,
};
#[allow(unused_imports)]
pub use aidaemon_work::mandates::{
    mandate_owner_label, owner_question_excerpt, review_failure_is_notified, Intention,
    IntentionStatus, Mandate, MandateActivityLevel, MandateAuthority, MandateAuthorityGrant,
    MandateAutonomyMode, MandateDecisionCycle, MandateDecisionOrigin, MandateDecisionOutcome,
    MandateFinalizationRejectReason, MandateFinalizationStaleReason, MandateLearningNote,
    MandateMutationAttempt, MandateMutationAttemptStatus, MandateMutationDispatchClaim,
    MandateMutationEvidence, MandateMutationQuotaBlockReason, MandateMutationQuotaState,
    MandateMutationReservation, MandateMutationTarget, MandateObjectiveControl,
    MandateObjectiveMeasurement, MandateOperatingUpdates, MandateOperationKind,
    MandateOperationScope, MandateReconciliationReason, MandateReconciliationResolution,
    MandateRunFinalizationRequest, MandateRunFinalizationResult, MandateRunNotification,
    MandateRunNotificationKind, MandateRunProofCounts, MandateStatus, MandateStrategySnapshot,
    MandateSuspension, MandateSuspensionKind, MandateTerminationKind, MandateWakeSignal,
    MandateWakeSignalKind, ObjectiveMetricDirection, SAFE_FALLBACK_WAIT_RATIONALE,
};
#[cfg(test)]
pub use channels::ChannelCapabilities;
pub use channels::HubChannel;
#[allow(unused_imports)]
pub use dynamic::{
    CliAgentInvocation, DynamicBot, DynamicCliAgent, DynamicMcpServer, DynamicSkill,
    OAuthConnection, PendingOAuthFlow, SkillDraft,
};
#[allow(unused_imports)]
pub use history::{
    HistoryCoverage, HistoryMessage, HistoryScope, HistorySearchStore, HistoryToolActivity,
    ProjectionStats, RecentToolTurn, TaskBookends,
};
#[allow(unused_imports)]
pub use memory::{
    BehaviorPattern, Episode, ErrorSolution, Expertise, ExtractedMemoryEntity,
    ExtractedMemoryGraph, ExtractedMemoryRelationship, Fact, MemoryHealthReport,
    PersonalAliasCandidate, PersonalEntityCandidate, PersonalFactCandidate, PersonalMemoryWrite,
    PersonalMemoryWriteResult, PersonalRelationshipCandidate, Procedure, UserProfile,
};
#[allow(unused_imports)]
pub use people::{Person, PersonFact};
#[cfg(test)]
pub use tools::ToolCalls;
#[allow(unused_imports)]
pub use tools::{
    default_call_annotations, prepare_invocation, semantics_for_exact_read_actions, status_sender,
    with_status, AgentRole, ApprovalCall, ApprovalScope, AuthorizationPreflightRecord,
    AuthorizationPreflightStatus, DurableReplayDecision, EvidenceAuthority, EvidencePurpose,
    EvidenceTemporalScope, FileGuard, GoalContext, NetworkReach, OwnerApprovalNeed,
    PreparedToolInvocation, ReadFileResultMetadata, ReadFileSelectionMetadata,
    RuntimePreparationFailure, SameToolStreak, ScopeEscalation, SpecialistKind,
    StableObservationSubject, Tool, ToolAccessDenial, ToolAccessEnforcement,
    ToolArgumentContractViolation, ToolCallAccessManifest, ToolCallContext, ToolCallEffect,
    ToolCallMetadata, ToolCallOperation, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolCollectionCompleteness, ToolCollectionObservation, ToolDispatchFacts,
    ToolEvidenceCapability, ToolInvocationStage, ToolMutationEffects, ToolObservationEvidence,
    ToolOutcomeStatus, ToolReceiptKind, ToolResultCompleteness, ToolResultContentSource,
    ToolResultPresentation, ToolResultProvenance, ToolRole, ToolSemanticFacet, ToolSemanticScope,
    ToolSemantics, ToolStatus, ToolTargetHint, ToolTargetHintKind, ToolVerificationMode,
    TruncationInfo, TrustRoots, EGRESS_EFFECTS,
};

mod state_store;
pub use state_store::*;

/// Import this in modules that call store-trait methods on concrete types.
pub use aidaemon_runtime::stores::prelude as store_prelude;

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
