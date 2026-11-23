//! `set_reminder` calls against a scratch state store.

use super::*;
use crate::stores::prelude::*;
use aidaemon_tools::semantics::ToolOutcomeStatus;
use aidaemon_tools::testing::ToolCalls as _;
use serde_json::json;
use std::sync::Arc;

async fn state() -> Arc<crate::state::SqliteStateStore> {
    let db = tempfile::NamedTempFile::new().unwrap();
    let path = db.path().to_str().unwrap().to_string();
    aidaemon_tools::testing::delete_at_exit(db.into_temp_path().keep().unwrap());
    let embeddings = Arc::new(aidaemon_memory::embeddings::EmbeddingService::deterministic());
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
            aidaemon_tools::semantics::ToolCallContext {
                session_id: "telegram:synthetic-user-1".into(),
                user_role: Some(aidaemon_tools::semantics::UserRole::Owner),
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
        aidaemon_work::reminders::message_from_context(goal.context.as_deref()).as_deref(),
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
            aidaemon_tools::semantics::ToolCallContext {
                session_id: "telegram:synthetic-user-2".into(),
                user_role: Some(aidaemon_tools::semantics::UserRole::Guest),
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
