//! `set_reminder` lives in `aidaemon_runtime::tools::set_reminder`; the tests here run it against the daemon's assembled state store.

pub use aidaemon_runtime::tools::set_reminder::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ToolCalls as _;
    use crate::traits::ToolOutcomeStatus;
    use crate::traits::store_prelude::*;
    use serde_json::json;
    use std::sync::Arc;

    async fn state() -> Arc<crate::state::SqliteStateStore> {
        let db = tempfile::NamedTempFile::new().unwrap();
        let path = db.path().to_str().unwrap().to_string();
        std::mem::forget(db);
        let embeddings = Arc::new(crate::memory::embeddings::EmbeddingService::deterministic());
        Arc::new(
            crate::state::SqliteStateStore::for_tests(&path, 100, None, embeddings)
                .await
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn sets_an_active_typed_reminder() {
        let state = state().await;
        let tool = SetReminderTool::new(state.clone());
        let outcome = tool
            .call_with_execution_context(
                &json!({
                    "message": "Check the oven.",
                    "when": "in 2h"
                })
                .to_string(),
                None,
                crate::traits::ToolCallContext {
                    session_id: "telegram:synthetic-user-1".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(
            outcome.output.starts_with("Reminder set"),
            "{}",
            outcome.output
        );
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        let goal = state
            .get_scheduled_goals()
            .await
            .unwrap()
            .into_iter()
            .find(|goal| goal.session_id == "telegram:synthetic-user-1")
            .unwrap();
        assert_eq!(goal.status, "active");
        assert_eq!(
            crate::reminders::message_from_context(goal.context.as_deref()).as_deref(),
            Some("Check the oven.")
        );
    }

    #[tokio::test]
    async fn non_owner_cannot_set_a_reminder() {
        let state = state().await;
        let tool = SetReminderTool::new(state.clone());
        let outcome = tool
            .call_with_execution_context(
                &json!({
                    "message": "Check the oven.",
                    "when": "in 2h"
                })
                .to_string(),
                None,
                crate::traits::ToolCallContext {
                    session_id: "telegram:synthetic-user-2".into(),
                    user_role: Some(crate::types::UserRole::Guest),
                    ..Default::default()
                },
            )
            .await;
        let failed = match outcome {
            Err(_) => true,
            Ok(outcome) => outcome.metadata.outcome_status != Some(ToolOutcomeStatus::Succeeded),
        };
        assert!(failed);
    }
}
