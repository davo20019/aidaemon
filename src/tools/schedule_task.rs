//! `schedule_task` lives in `aidaemon_runtime::tools::schedule_task`; the tests here run it against the daemon's assembled state store.

pub use aidaemon_runtime::tools::schedule_task::ScheduleTaskTool;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::store_prelude::*;
    use crate::traits::{
        ApprovalCall, OwnerApprovalNeed, ToolCallContext, ToolCalls as _, ToolOutcomeStatus,
        ToolSemantics,
    };
    use crate::types::{ChannelVisibility, UserRole};
    use serde_json::{Value, json};
    use std::sync::Arc;

    async fn setup_state() -> Arc<SqliteStateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        state
    }

    fn owner() -> ToolCallContext {
        ToolCallContext {
            user_role: Some(UserRole::Owner),
            session_id: "test-session".into(),
            channel_visibility: Some(ChannelVisibility::Private),
            ..Default::default()
        }
    }

    fn approved() -> ToolCallContext {
        ToolCallContext {
            kernel_approved: Some(crate::traits::ApprovalScope::Call),
            ..owner()
        }
    }

    async fn call(
        tool: &ScheduleTaskTool,
        args: Value,
        facts: ToolCallContext,
    ) -> crate::traits::ToolCallOutcome {
        tool.call_with_execution_context(&args.to_string(), None, facts)
            .await
            .unwrap()
    }

    async fn judged(
        tool: &ScheduleTaskTool,
        args: Value,
        facts: ToolCallContext,
    ) -> OwnerApprovalNeed {
        tool.owner_approval_request(&ApprovalCall::as_sent(&args), &facts)
            .await
    }

    #[tokio::test]
    async fn creates_active_recurring_work() {
        let state = setup_state().await;
        let tool = ScheduleTaskTool::new(state.clone());
        let created = call(
            &tool,
            json!({"task": "Check API health", "when": "every 6h"}),
            approved(),
        )
        .await;
        assert_eq!(
            created.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        assert!(
            created.output.starts_with("Scheduled: Check API health."),
            "{}",
            created.output
        );
        let goal = state
            .get_scheduled_goals()
            .await
            .unwrap()
            .into_iter()
            .find(|g| g.description == "Check API health")
            .unwrap();
        assert!(created.metadata.internal_identifiers.contains(&goal.id));
        assert_eq!(goal.status, "active");
        assert_eq!(goal.domain, "orchestration");
        assert_eq!(goal.goal_type, "continuous");
        assert_eq!(goal.budget_per_check, Some(400_000));
        assert_eq!(goal.budget_daily, Some(1_000_000));
        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert_eq!(schedules.len(), 1);
        assert!(!schedules[0].is_one_shot);
        assert_eq!(schedules[0].cron_expr, "0 */6 * * *");
        assert_eq!(schedules[0].tz, "local");
    }

    #[tokio::test]
    async fn creates_one_time_work_and_cleans_the_schedule_from_the_task() {
        let state = setup_state().await;
        let tool = ScheduleTaskTool::new(state.clone());
        let created = call(
            &tool,
            json!({"task": "Please send the synthetic report tomorrow at 9am", "when": "tomorrow at 9am", "name": "Report"}),
            approved(),
        )
        .await;
        assert!(
            created.output.starts_with("Scheduled \"Report\": "),
            "{}",
            created.output
        );
        let goal = state.get_scheduled_goals().await.unwrap().remove(0);
        assert_eq!(goal.goal_type, "finite");
        assert!(
            !goal.description.contains("tomorrow"),
            "{}",
            goal.description
        );
        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert!(schedules[0].is_one_shot);
    }

    #[tokio::test]
    async fn refuses_outside_the_kernel() {
        let state = setup_state().await;
        let tool = ScheduleTaskTool::new(state.clone());
        let error = tool
            .call_with_execution_context(
                &json!({"task": "Check confirmation transport", "when": "every 6h"}).to_string(),
                None,
                owner(),
            )
            .await
            .expect_err("refused outside the kernel");
        assert!(
            error
                .to_string()
                .contains("requires approval through the agent kernel"),
            "{error}"
        );
        assert!(state.get_scheduled_goals().await.unwrap().is_empty());
    }

    /// A refusal is typed as one: nothing is stored and the outcome is
    /// never a success, whatever its words.
    #[tokio::test]
    async fn refusals_store_nothing_and_are_never_a_success() {
        let state = setup_state().await;
        let tool = ScheduleTaskTool::new(state.clone());
        let args = json!({"task": "Check API health", "when": "every 6h"});
        for facts in [
            ToolCallContext {
                user_role: Some(UserRole::Guest),
                ..approved()
            },
            ToolCallContext {
                channel_visibility: Some(ChannelVisibility::Internal),
                ..approved()
            },
            ToolCallContext {
                session_id: "specialist:executor:synthetic".into(),
                ..approved()
            },
        ] {
            let refused = call(&tool, args.clone(), facts).await;
            assert_eq!(
                refused.metadata.outcome_status,
                Some(ToolOutcomeStatus::FailedPermanent),
                "{}",
                refused.output
            );
        }
        let bad = call(
            &tool,
            json!({"task": "Check API health", "when": "whenever the moon is full"}),
            approved(),
        )
        .await;
        assert_eq!(
            bad.metadata.outcome_status,
            Some(ToolOutcomeStatus::FailedPermanent)
        );
        assert!(state.get_scheduled_goals().await.unwrap().is_empty());

        // The same work twice is one automation.
        let first = call(&tool, args.clone(), approved()).await;
        assert_eq!(
            first.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        let again = call(&tool, args, approved()).await;
        assert_eq!(
            again.metadata.outcome_status,
            Some(ToolOutcomeStatus::FailedPermanent)
        );
        assert!(
            again
                .output
                .starts_with("The same scheduled work already exists")
        );
        assert_eq!(state.get_scheduled_goals().await.unwrap().len(), 1);
    }

    /// The owner is asked exactly where the call would create work that runs
    /// the agent: not for a non-owner, an internal run, a bad time or a
    /// duplicate. The card binds the exact work.
    #[tokio::test]
    async fn judges_its_own_approval() {
        let state = setup_state().await;
        let tool = ScheduleTaskTool::new(state.clone());
        let args = json!({"task": "Check API health", "when": "every 6h"});
        let OwnerApprovalNeed::Ask(request) = judged(&tool, args.clone(), owner()).await else {
            panic!("scheduled work asks");
        };
        assert_eq!(request.summary, "Check API health");
        assert_eq!(request.risk, aidaemon_core::ApprovalRisk::Medium);
        let card =
            aidaemon_core::mandate_card::MandateConfirmationCard::from_lines(&request.warnings)
                .expect("all five lines");
        assert_eq!(card.what, "Check API health");
        assert_eq!(card.how_often, "Every 6 hours");
        assert_eq!(
            card.can_do,
            "Anything you can do – it runs with your full access"
        );
        assert_eq!(card.ends, "Only when you stop it");
        assert!(card.details.iter().any(|w| w.starts_with("Next: ")));
        let OwnerApprovalNeed::Ask(other) = judged(
            &tool,
            json!({"task": "Check API health", "when": "every 2h"}),
            owner(),
        )
        .await
        else {
            panic!("scheduled work asks");
        };
        assert_ne!(request.subject, other.subject);
        for facts in [
            ToolCallContext {
                user_role: Some(UserRole::Guest),
                ..owner()
            },
            ToolCallContext {
                channel_visibility: Some(ChannelVisibility::Internal),
                ..owner()
            },
        ] {
            assert_eq!(
                judged(&tool, args.clone(), facts).await,
                OwnerApprovalNeed::NotNeeded
            );
        }
        assert_eq!(
            judged(
                &tool,
                json!({"task": "Check API health", "when": "whenever the moon is full"}),
                owner()
            )
            .await,
            OwnerApprovalNeed::NotNeeded
        );
        call(&tool, args.clone(), approved()).await;
        assert_eq!(
            judged(&tool, args, owner()).await,
            OwnerApprovalNeed::NotNeeded
        );
    }
}
