use serde::{Deserialize, Serialize};

/// What a self-correction attempt is about.
#[allow(dead_code)] // Used in Task 2+; StateStore methods and SQLite impl pending
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelfCorrectionSubjectKind {
    Task,
    Goal,
    BackgroundCommand,
}

impl SelfCorrectionSubjectKind {
    #[allow(dead_code)] // Used in Task 2+; StateStore methods and SQLite impl pending
    pub fn as_str(&self) -> &'static str {
        match self {
            SelfCorrectionSubjectKind::Task => "task",
            SelfCorrectionSubjectKind::Goal => "goal",
            SelfCorrectionSubjectKind::BackgroundCommand => "background_command",
        }
    }

    #[allow(dead_code)] // Used in Task 2+; StateStore methods and SQLite impl pending
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "task" => Some(SelfCorrectionSubjectKind::Task),
            "goal" => Some(SelfCorrectionSubjectKind::Goal),
            "background_command" => Some(SelfCorrectionSubjectKind::BackgroundCommand),
            _ => None,
        }
    }
}

/// Persisted status of a single attempt.
pub mod attempt_status {
    #[allow(dead_code)] // Used in Task 2+; StateStore methods and SQLite impl pending
    pub const BLOCKED: &str = "blocked";
    #[allow(dead_code)] // Used in Task 2+; StateStore methods and SQLite impl pending
    pub const EXECUTED: &str = "executed";
    #[allow(dead_code)] // Used in Task 2+; StateStore methods and SQLite impl pending
    pub const FAILED: &str = "failed";
    #[allow(dead_code)] // Used in Task 2+; StateStore methods and SQLite impl pending
    pub const VERIFIED_SUCCESS: &str = "verified_success";
    #[allow(dead_code)] // Used in Task 2+; StateStore methods and SQLite impl pending
    pub const GAVE_UP: &str = "gave_up";
}

/// One self-correction attempt against a subject. Durable audit + the basis for
/// repeat-blocking and the K-bound runaway backstop.
#[allow(dead_code)] // Used in Task 2+; StateStore methods and SQLite impl pending
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfCorrectionAttempt {
    pub id: i64,
    pub subject_id: String,
    pub subject_kind: String,
    pub approach_signature: String,
    pub attempt_index: i64,
    pub status: String,
    pub blocked_reason: Option<String>,
    pub evidence_ref: Option<String>,
    pub created_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subject_kind_round_trips_through_str() {
        for k in [
            SelfCorrectionSubjectKind::Task,
            SelfCorrectionSubjectKind::Goal,
            SelfCorrectionSubjectKind::BackgroundCommand,
        ] {
            assert_eq!(SelfCorrectionSubjectKind::from_str(k.as_str()), Some(k));
        }
        assert_eq!(SelfCorrectionSubjectKind::from_str("nonsense"), None);
    }
}
