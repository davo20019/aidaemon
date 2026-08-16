use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::tools::{
    EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, ToolMutationEffects,
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
    /// Whether some authoritative result content must exist.
    #[serde(default)]
    pub requires_output: bool,
    /// Required pre-I/O invocation-contract disposition when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_rejected: Option<bool>,
}

impl RequestReceiptPredicate {
    pub fn is_empty(&self) -> bool {
        self.tool_names.is_empty()
            && self.exit_codes.is_empty()
            && self.outcome_statuses.is_empty()
            && !self.requires_output
            && self.contract_rejected.is_none()
    }
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
    /// Exact user-authored labels required in the substantive final answer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_response_fields: Vec<String>,
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
