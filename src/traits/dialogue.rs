use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::tools::{
    EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, ToolEvidenceCapability,
    ToolInvocationStage, ToolMutationEffects, ToolOutcomeStatus, ToolReceiptKind,
    ToolSemanticScope,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestTaskKind {
    Conversational,
    Answer,
    Check,
    Find,
    Change,
    Deliver,
    Schedule,
    Monitor,
    Diagnose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestForbiddenAction {
    Create,
    Delete,
    Deploy,
    Publish,
    Post,
    Send,
}

/// Typed presentation contract for the successful terminal response.
///
/// This never proves that execution succeeded. The run aggregate may project
/// it only after every non-delivery obligation has authoritative evidence.
/// `source_message_hash` binds the assessor-produced contract to the user turn
/// it interpreted. Natural-language interpretation remains entirely in the
/// semantic producer rather than being repeated as Rust keyword rules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum RequestResponseContract {
    ExactText {
        success_text: String,
        source_message_hash: String,
    },
}

impl RequestResponseContract {
    pub fn success_text(&self) -> &str {
        match self {
            Self::ExactText { success_text, .. } => success_text,
        }
    }

    pub fn source_message_hash(&self) -> &str {
        match self {
            Self::ExactText {
                source_message_hash,
                ..
            } => source_message_hash,
        }
    }

    pub fn mode(&self) -> &'static str {
        match self {
            Self::ExactText { .. } => "exact_text",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestVerificationTargetKind {
    Url,
    Path,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestVerificationTarget {
    pub kind: RequestVerificationTargetKind,
    pub value: String,
}

/// Machine-checkable facts about the receipt that must support one evidence
/// requirement. Receipt metadata and observed subject content are deliberately
/// separate: an exit code must never be encoded as a prose marker that stdout
/// is then expected to contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestedOutcomeCondition {
    /// Any terminal disposition, including a normal success.
    AnyTerminal,
    Succeeded,
    CompletedWithNegativeResult,
    /// A dispatched adapter/runtime failure, excluding policy rejection.
    Failed,
    Blocked,
    /// A pre-I/O validation or policy rejection.
    ContractRejected,
    /// Any terminal disposition other than success. This is the canonical
    /// condition for an expected refusal/failure where the exact boundary is
    /// intentionally not predetermined by the requester.
    NonSuccessTerminal,
}

impl RequestedOutcomeCondition {
    pub fn matches(
        self,
        outcome: ToolOutcomeStatus,
        invocation_stage: ToolInvocationStage,
        contract_rejected: bool,
    ) -> bool {
        let terminal = outcome != ToolOutcomeStatus::Backgrounded;
        let terminal_observed = terminal
            && (invocation_stage.reached_dispatch()
                || contract_rejected
                || outcome == ToolOutcomeStatus::Blocked);
        match self {
            Self::AnyTerminal => terminal_observed,
            Self::Succeeded => {
                outcome == ToolOutcomeStatus::Succeeded
                    && invocation_stage.reached_dispatch()
                    && !contract_rejected
            }
            Self::CompletedWithNegativeResult => {
                outcome == ToolOutcomeStatus::CompletedWithNegativeResult
                    && invocation_stage.reached_dispatch()
                    && !contract_rejected
            }
            // A command that ran and reported a nonzero exit is a failed
            // invocation from the request's point of view ("run `false`;
            // expect it to fail") even though the adapter types it as a
            // negative result rather than a tool failure.
            Self::Failed => {
                matches!(
                    outcome,
                    ToolOutcomeStatus::FailedRetryable
                        | ToolOutcomeStatus::FailedPermanent
                        | ToolOutcomeStatus::CompletedWithNegativeResult
                ) && invocation_stage.reached_dispatch()
                    && !contract_rejected
            }
            Self::Blocked => outcome == ToolOutcomeStatus::Blocked,
            // The rejection bit is the cross-version fact. Durable replay is
            // represented by `invocation_stage=replayed`, so requiring a
            // non-dispatch stage here would make a valid persisted rejection
            // stop satisfying its original contract after hydration.
            Self::ContractRejected => contract_rejected,
            Self::NonSuccessTerminal => {
                terminal_observed && (contract_rejected || outcome != ToolOutcomeStatus::Succeeded)
            }
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestReceiptPredicate {
    /// Exact requested or effective tool identifiers that may satisfy this
    /// need. Empty accepts any tool whose evidence capability is compatible.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_names: Vec<String>,
    /// Alternative acceptable process exit codes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exit_codes: Vec<i32>,
    /// Alternative acceptable typed invocation outcomes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outcome_statuses: Vec<super::tools::ToolOutcomeStatus>,
    /// Request-level outcome policy. Semantic assessment chooses this compact
    /// condition; deterministic receipt matching evaluates it against the
    /// adapter-specific protocol. Legacy low-level fields remain readable for
    /// persisted contracts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_condition: Option<RequestedOutcomeCondition>,
    /// Whether some authoritative result content must exist.
    #[serde(default)]
    pub requires_output: bool,
    /// Required pre-I/O invocation-contract disposition when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_rejected: Option<bool>,
    /// Minimum number of distinct matching terminal receipts needed to prove
    /// this invocation obligation. Response prose and repeated copies of one
    /// receipt cannot substitute for the requested number of operations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_invocations: Option<usize>,
    /// Maximum tool-invocation proposals allowed for this user-owned
    /// operation. This counts pre-I/O validation outcomes as invocations while
    /// the execution-attempt ledger separately counts actual dispatches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_invocations: Option<usize>,
}

impl RequestReceiptPredicate {
    pub fn is_empty(&self) -> bool {
        self.tool_names.is_empty()
            && self.exit_codes.is_empty()
            && self.outcome_statuses.is_empty()
            && self.outcome_condition.is_none()
            && !self.requires_output
            && self.contract_rejected.is_none()
            && self.min_invocations.is_none()
            && self.max_invocations.is_none()
    }

    pub fn exit_matches(&self, receipt_kind: ToolReceiptKind, exit_code: Option<i32>) -> bool {
        self.exit_codes.is_empty()
            // Exit codes are a process-adapter field. Older persisted
            // contracts could carry a provider-defaulted exit-code predicate
            // for generic receipts, where it has no meaning and must not veto
            // otherwise valid typed evidence.
            || receipt_kind != ToolReceiptKind::Process
            || exit_code.is_some_and(|actual| self.exit_codes.contains(&actual))
    }

    pub fn outcome_matches(
        &self,
        outcome: ToolOutcomeStatus,
        invocation_stage: ToolInvocationStage,
        contract_rejected: bool,
        receipt_kind: ToolReceiptKind,
        exit_code: Option<i32>,
    ) -> bool {
        if let Some(condition) = self.outcome_condition {
            return condition.matches(outcome, invocation_stage, contract_rejected);
        }
        if contract_rejected && self.contract_rejected == Some(true) {
            return true;
        }
        if self.outcome_statuses.is_empty() || self.outcome_statuses.contains(&outcome) {
            return true;
        }
        // Compatibility for contracts persisted before normal nonzero process
        // exits received their own domain-outcome enum.
        outcome == ToolOutcomeStatus::CompletedWithNegativeResult
            && receipt_kind == ToolReceiptKind::Process
            && exit_code.is_some_and(|code| code != 0)
            && self
                .outcome_statuses
                .contains(&ToolOutcomeStatus::FailedPermanent)
    }

    pub fn rejection_matches(&self, contract_rejected: bool) -> bool {
        self.outcome_condition.is_some()
            || self
                .contract_rejected
                .is_none_or(|expected| contract_rejected == expected)
    }
}

/// A task-local conditional transition that closes selected execution lanes
/// after a typed receipt has occurred.
///
/// This is deliberately a receipt-to-capability edge rather than a phrase or
/// command rule. It lets a semantic producer express workflows such as “after
/// the prerequisite is permanently rejected, do not start its dependent
/// process”, while leaving unrelated recovery tools available.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestDispatchStopRule {
    pub trigger: RequestReceiptPredicate,
    /// Receipt protocol families closed by this transition. The compiler
    /// resolves these capabilities against the registered tool set and
    /// persists the resulting exact adapter names for atomic admission.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked_receipt_kinds: Vec<ToolReceiptKind>,
    /// Optional exact adapter lanes. This is useful when the request names a
    /// particular tool rather than an entire capability family.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked_tool_names: Vec<String>,
}

/// One material information need that must be supported before a request can
/// complete successfully. The natural-language summary guides investigation;
/// the typed fields are the completion invariant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestEvidenceRequirement {
    pub summary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub acceptable_scopes: Vec<ToolSemanticScope>,
    pub purpose: EvidencePurpose,
    pub minimum_authority: EvidenceAuthority,
    pub temporal_scope: EvidenceTemporalScope,
    /// Legacy advisory field/key hints. These remain for persisted schema
    /// compatibility and investigation guidance, but never decide lifecycle
    /// completion. Hard proof uses typed receipts and structural identities.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_content_markers: Vec<String>,
    /// Typed invocation/result constraints. New outcome requirements must use
    /// this field; `required_content_markers` remains subject-data matching for
    /// content and state observations plus legacy persisted contracts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt: Option<RequestReceiptPredicate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<RequestVerificationTarget>,
}

impl RequestEvidenceRequirement {
    pub fn supports_capability(&self, capability: &ToolEvidenceCapability) -> bool {
        self.acceptable_scopes.contains(&capability.scope)
            && capability.purposes.contains(&self.purpose)
            && capability.authority.satisfies(self.minimum_authority)
            && capability.temporal_scope.satisfies(self.temporal_scope)
    }
}

/// Stable proof-graph identity retained when a typed continuation adopts an
/// evidence requirement from another task. The requirement value is stored
/// with the source ID so hydration, deduplication, and local reordering never
/// turn a positional index into the wrong obligation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdoptedEvidenceBinding {
    pub source_obligation_id: String,
    pub requirement: RequestEvidenceRequirement,
}

/// Durable semantic obligations for an unresolved request. These values come
/// from a validated task assessment or structural resource identity, never
/// from replaying natural-language phrase rules on a later turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestCompletionContract {
    /// Task whose lifecycle owns every executable obligation in this contract.
    /// Legacy rows omit this value and must be explicitly adopted before use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_task_id: Option<String>,
    /// Prior task contracts deliberately adopted into this task by the typed
    /// dialogue relationship. This is audit lineage, not an authorization to
    /// execute arbitrary obligations from history.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adopted_from_task_ids: Vec<String>,
    pub task_kind: RequestTaskKind,
    pub expects_mutation: bool,
    pub required_mutation_effects: ToolMutationEffects,
    pub forbids_mutation: bool,
    /// The current request explicitly forbids every tool/lookup path.
    #[serde(default)]
    pub forbids_tool_use: bool,
    /// When non-empty, only these explicitly authorized tool identifiers may
    /// execute for the request. This represents "use only X" without
    /// collapsing the contract into an all-tools prohibition.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_tool_names: Vec<String>,
    /// Capability domains the current request explicitly excludes while
    /// leaving unrelated tools available.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forbidden_tool_scopes: Vec<ToolSemanticScope>,
    /// Conditional task-local stop transitions. A triggered rule removes only
    /// its selected tools; it cannot grant authority or block unrelated
    /// recovery capabilities.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dispatch_stop_rules: Vec<RequestDispatchStopRule>,
    /// Exact user-authored labels required in the substantive final answer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_response_fields: Vec<String>,
    /// Grounded response artifact to project after execution proof closes.
    /// Unlike legacy response-field labels, this is an exact typed value with
    /// user-message provenance and is never interpreted as execution proof.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_contract: Option<Box<RequestResponseContract>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forbidden_actions: Vec<RequestForbiddenAction>,
    pub requires_observation: bool,
    pub requires_reverification_after_mutation: bool,
    pub explicit_verification_requested: bool,
    #[serde(default)]
    pub minimum_sources: usize,
    #[serde(default)]
    pub requires_primary_sources: bool,
    #[serde(default)]
    pub requires_exact_history: bool,
    /// Material claim-support obligations. Legacy rows omit this field and
    /// retain the older generic-observation behavior.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_requirements: Vec<RequestEvidenceRequirement>,
    /// Stable cross-task proof bindings. These are lineage metadata, not
    /// evidence: a matching terminal receipt is still required to satisfy the
    /// corresponding local obligation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adopted_evidence_bindings: Vec<AdoptedEvidenceBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub verification_targets: Vec<RequestVerificationTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRequestStatus {
    Open,
    InProgress,
    PartiallyAnswered,
    Answered,
    Blocked,
    Superseded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionKind {
    Clarification,
    Approval,
    Choice,
    Confirmation,
    /// Runtime-authenticated marker for an owner mandate whose model-authored
    /// question remains isolated behind `manage_mandates(get)`.
    MandateInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssistantTurnKind {
    ClarificationQuestion,
    PartialProgress,
    SubstantiveAnswer,
    Blocked,
    Refusal,
    SystemNotice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserTurnKind {
    NewRequest,
    Followup,
    ClarificationAnswer,
    Courtesy,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveTaskStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenRequest {
    pub user_message_id: String,
    pub text: String,
    pub status: OpenRequestStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_scope: Option<ToolSemanticScope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_contract: Option<RequestCompletionContract>,
    pub opened_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenQuestion {
    pub assistant_message_id: String,
    pub text: String,
    pub kind: QuestionKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub related_user_message_id: Option<String>,
    /// Structured binding for [`QuestionKind::MandateInput`]. Keeping this out
    /// of `text` avoids parsing notification prose to recover the obligation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mandate_id: Option<String>,
    pub awaiting_user_reply: bool,
    pub asked_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantTurnSummary {
    pub message_id: String,
    pub kind: AssistantTurnKind,
    pub left_request_open: bool,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserTurnSummary {
    pub message_id: String,
    pub kind: UserTurnKind,
    pub text: String,
}

/// Typed lineage from a conversational user turn to the canonical request it
/// advances. A follow-up remains addressable by its own message ID even though
/// lifecycle ownership and completion obligations stay on the root request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserRequestBinding {
    pub user_message_id: String,
    pub request_user_message_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveTaskRef {
    pub task_id: String,
    pub status: ActiveTaskStatus,
    pub started_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DialogueState {
    pub session_id: String,
    pub schema_version: u32,
    pub revision: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_request: Option<OpenRequest>,
    /// Request that immediately preceded a provisionally independent user
    /// turn. Ingress cannot know whether ordinary language is a new request or
    /// a continuation, so it retains (rather than destroys) the exact typed
    /// antecedent until task assessment commits the relationship.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub antecedent_request: Option<OpenRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_question: Option<OpenQuestion>,
    /// The most recent open question that a user reply closed. Kept so the
    /// turn that ANSWERS a clarifying question can still see the question
    /// text after ingestion clears `open_question`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_closed_question: Option<OpenQuestion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_assistant_turn: Option<AssistantTurnSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_user_turn: Option<UserTurnSummary>,
    /// Recent typed message-to-request lineage used to canonicalize semantic
    /// antecedents selected from the bounded conversation transcript.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recent_request_bindings: Vec<UserRequestBinding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_task: Option<ActiveTaskRef>,
    pub updated_at: DateTime<Utc>,
}

impl DialogueState {
    pub const SCHEMA_VERSION: u32 = 2;

    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            schema_version: Self::SCHEMA_VERSION,
            revision: 1,
            open_request: None,
            antecedent_request: None,
            open_question: None,
            last_closed_question: None,
            last_assistant_turn: None,
            last_user_turn: None,
            recent_request_bindings: Vec::new(),
            active_task: None,
            updated_at: Utc::now(),
        }
    }

    pub fn touch(&mut self) {
        self.revision = self.revision.saturating_add(1);
        self.updated_at = Utc::now();
    }

    /// Project the persisted dialogue lifecycle into the shared obligation
    /// graph. Relationships come exclusively from stable IDs and typed status;
    /// no response wording participates in whether work remains open.
    pub(crate) fn obligation_graph(
        &self,
    ) -> Result<crate::execution_graph::ExecutionGraph, String> {
        use crate::execution_graph::{
            ExecutionEdgeKind, ExecutionGraph, ExecutionNodeKind, ExecutionNodeState,
        };

        let mut graph = ExecutionGraph::default();
        let Some(request) = self.open_request.as_ref() else {
            return Ok(graph);
        };
        let request_id = format!("dialogue-request:{}", request.user_message_id);
        let request_state = match request.status {
            OpenRequestStatus::Answered => ExecutionNodeState::Satisfied,
            OpenRequestStatus::Superseded => ExecutionNodeState::Superseded,
            OpenRequestStatus::Blocked => ExecutionNodeState::Blocked,
            OpenRequestStatus::InProgress => ExecutionNodeState::Running,
            OpenRequestStatus::Open | OpenRequestStatus::PartiallyAnswered => {
                ExecutionNodeState::Pending
            }
        };
        graph.add_node(
            request_id.clone(),
            ExecutionNodeKind::Request,
            request_state,
        )?;

        if let Some(task) = self.active_task.as_ref().filter(|task| {
            request
                .task_id
                .as_deref()
                .is_some_and(|request_task_id| request_task_id == task.task_id)
        }) {
            let task_id = format!("dialogue-task:{}", task.task_id);
            let task_state = match task.status {
                ActiveTaskStatus::Running => ExecutionNodeState::Running,
                ActiveTaskStatus::Completed => ExecutionNodeState::Satisfied,
                ActiveTaskStatus::Failed | ActiveTaskStatus::Cancelled => {
                    ExecutionNodeState::Failed
                }
            };
            graph.add_node(task_id.clone(), ExecutionNodeKind::Task, task_state)?;
            graph.add_edge(&request_id, &task_id, ExecutionEdgeKind::Requires, None)?;
        }

        if let Some(question) = self
            .open_question
            .as_ref()
            .filter(|question| question.awaiting_user_reply)
        {
            let question_id = format!("dialogue-question:{}", question.assistant_message_id);
            let input_id = format!("dialogue-input:{}", question.assistant_message_id);
            graph.add_node(
                question_id.clone(),
                ExecutionNodeKind::Obligation,
                ExecutionNodeState::Pending,
            )?;
            graph.add_node(
                input_id.clone(),
                ExecutionNodeKind::HumanInput,
                ExecutionNodeState::Pending,
            )?;
            graph.add_edge(&request_id, &question_id, ExecutionEdgeKind::Requires, None)?;
            graph.add_edge(
                &question_id,
                &input_id,
                ExecutionEdgeKind::AwaitsInput,
                None,
            )?;
        }

        Ok(graph)
    }

    pub(crate) fn has_unresolved_request_obligation(&self) -> bool {
        use crate::execution_graph::ExecutionNodeState;

        let Some(request) = self.open_request.as_ref() else {
            return false;
        };
        let request_id = format!("dialogue-request:{}", request.user_message_id);
        self.obligation_graph().is_ok_and(|graph| {
            graph
                .state(&request_id)
                .is_some_and(|state| !ExecutionNodeState::satisfies_dependency(state))
                || !graph.requirements_satisfied(&request_id)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_success_terminal_condition_spans_rejection_and_dispatched_failure() {
        let condition = RequestedOutcomeCondition::NonSuccessTerminal;
        assert!(condition.matches(
            ToolOutcomeStatus::Blocked,
            ToolInvocationStage::RejectedBeforeDispatch,
            true,
        ));
        assert!(condition.matches(
            ToolOutcomeStatus::FailedPermanent,
            ToolInvocationStage::Dispatched,
            false,
        ));
        assert!(condition.matches(
            ToolOutcomeStatus::CompletedWithNegativeResult,
            ToolInvocationStage::Dispatched,
            false,
        ));
        assert!(!condition.matches(
            ToolOutcomeStatus::Succeeded,
            ToolInvocationStage::Dispatched,
            false,
        ));
        assert!(!condition.matches(
            ToolOutcomeStatus::Backgrounded,
            ToolInvocationStage::Dispatched,
            false,
        ));
        assert!(!condition.matches(
            ToolOutcomeStatus::FailedPermanent,
            ToolInvocationStage::Unknown,
            false,
        ));
        assert!(RequestedOutcomeCondition::ContractRejected.matches(
            ToolOutcomeStatus::Blocked,
            ToolInvocationStage::Replayed,
            true,
        ));
    }

    #[test]
    fn obligation_graph_uses_typed_request_and_question_state() {
        let now = Utc::now();
        let mut state = DialogueState::new("session");
        state.open_request = Some(OpenRequest {
            user_message_id: "user-1".to_string(),
            text: "synthetic request".to_string(),
            status: OpenRequestStatus::PartiallyAnswered,
            task_id: None,
            project_scope: None,
            semantic_scope: None,
            completion_contract: None,
            opened_at: now,
            resolved_at: None,
        });
        state.open_question = Some(OpenQuestion {
            assistant_message_id: "assistant-1".to_string(),
            text: "synthetic clarification".to_string(),
            kind: QuestionKind::Clarification,
            related_user_message_id: Some("user-1".to_string()),
            mandate_id: None,
            awaiting_user_reply: true,
            asked_at: now,
        });

        assert!(state.has_unresolved_request_obligation());
        let graph = state.obligation_graph().unwrap();
        assert!(!graph.requirements_satisfied("dialogue-request:user-1"));

        state.open_question = None;
        state.open_request.as_mut().unwrap().status = OpenRequestStatus::Answered;
        assert!(!state.has_unresolved_request_obligation());
    }
}
