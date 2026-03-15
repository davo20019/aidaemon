use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::tools::ToolSemanticScope;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_question: Option<OpenQuestion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_assistant_turn: Option<AssistantTurnSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_user_turn: Option<UserTurnSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_task: Option<ActiveTaskRef>,
    pub updated_at: DateTime<Utc>,
}

impl DialogueState {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            schema_version: Self::SCHEMA_VERSION,
            revision: 1,
            open_request: None,
            open_question: None,
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
}
