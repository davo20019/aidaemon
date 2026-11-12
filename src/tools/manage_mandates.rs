//! `manage_mandates` lives in `aidaemon_work::tools::manage_mandates`; the
//! daemon gives it the skill library for strategies. The tests here run it
//! against the daemon's assembled state store and agent.

pub use aidaemon_work::tools::manage_mandates::*;

/// The daemon's filesystem skills under `dir`, as mandate strategies read
/// them.
pub(crate) fn strategy_skills_dir(dir: std::path::PathBuf) -> std::sync::Arc<dyn StrategySkills> {
    std::sync::Arc::new(SkillsDirStrategies(dir))
}

struct SkillsDirStrategies(std::path::PathBuf);

impl StrategySkills for SkillsDirStrategies {
    fn find(&self, name: &str) -> Option<StrategySkill> {
        let skills = crate::skills::load_skills(&self.0);
        let skill = crate::skills::find_skill_by_name(&skills, name)?;
        Some(StrategySkill {
            name: skill.name.clone(),
            description: skill.description.clone(),
            body: crate::tools::sanitize::sanitize_external_content(&skill.body),
            source: skill.source.clone(),
            canonical: skill.to_markdown(),
        })
    }
}

#[cfg(test)]
mod tests {
    /// The dispatcher's facts for the owner in a private chat.
    fn owner_facts() -> crate::traits::ToolCallContext {
        crate::traits::ToolCallContext {
            session_id: "owner-session".into(),
            user_role: Some(crate::types::UserRole::Owner),
            channel_visibility: Some(crate::types::ChannelVisibility::Private),
            ..Default::default()
        }
    }

    /// The owner's facts for a call the agent kernel judged (the owner
    /// confirmed it where `owner_approval_request` asks).
    fn approved_owner_facts() -> crate::traits::ToolCallContext {
        crate::traits::ToolCallContext {
            kernel_approved: Some(crate::traits::ApprovalScope::Call),
            ..owner_facts()
        }
    }

    /// Outside the kernel (no `kernel_approved`) a create, update or
    /// transfer the owner would be asked about is refused and nothing is
    /// stored; one the call answers without committing is answered as
    /// before.
    #[tokio::test]
    async fn committing_owner_changes_refuse_without_kernel_approval() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let tool = ManageMandatesTool::new(state.clone());
        let args = json!({
            "action": "create",
            "objective": "Never activate without confirmation",
            "max_mutating_actions_per_cycle": 0,
            "success_criteria": ["The bounded source is reviewed without unauthorized action"],
        });
        assert!(matches!(
            tool.owner_approval(&args, &owner_facts()).await,
            crate::traits::OwnerApprovalNeed::Ask(_)
        ));
        let error = tool
            .call_with(&args.to_string(), owner_facts())
            .await
            .expect_err("refused outside the kernel");
        assert!(
            error
                .to_string()
                .contains("requires approval through the agent kernel"),
            "{error}"
        );
        assert!(state
            .list_mandates(Some("owner-session"), true)
            .await
            .unwrap()
            .is_empty());

        // A guest, or the owner outside a private chat, is answered without
        // a card or a commit.
        for facts in [
            crate::traits::ToolCallContext {
                user_role: Some(crate::types::UserRole::Guest),
                ..owner_facts()
            },
            crate::traits::ToolCallContext {
                channel_visibility: Some(crate::types::ChannelVisibility::Public),
                ..owner_facts()
            },
        ] {
            assert_eq!(
                tool.owner_approval(&args, &facts).await,
                crate::traits::OwnerApprovalNeed::NotNeeded
            );
            let answer = tool.call_with(&args.to_string(), facts).await.unwrap();
            assert!(answer.contains("verified private channel"), "{answer}");
        }
        for read in ["list", "draft", "get"] {
            assert_eq!(
                tool.owner_approval(&json!({"action": read}), &owner_facts())
                    .await,
                crate::traits::OwnerApprovalNeed::NotNeeded,
                "{read}"
            );
        }

        // The card binds the exact arguments and ignores injected `_` keys.
        let crate::traits::OwnerApprovalNeed::Ask(first) =
            tool.owner_approval(&args, &owner_facts()).await
        else {
            panic!("create asks");
        };
        let mut injected = args.clone();
        injected["_session_id"] = json!("synthetic-session");
        let crate::traits::OwnerApprovalNeed::Ask(same) =
            tool.owner_approval(&injected, &owner_facts()).await
        else {
            panic!("create asks");
        };
        assert_eq!(first.subject, same.subject);
    }

    use super::*;
    use crate::testing::CallWithFacts;
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::traits::*;
    use crate::types::ChannelVisibility;
    use serde_json::{json, Value};
    use std::sync::Arc;

    #[tokio::test]
    async fn mandate_list_projects_the_objective_lifecycle_readout() {
        // An owner audit that consults the mandate surface must see the same
        // authoritative schedule/run/recovery answer as scheduled_goal_runs,
        // not report the objective "unknown".
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = crate::traits::Goal::new_continuous(
            "Publish the synthetic daily report",
            "owner-session",
            None,
            None,
        );
        let mandate = Mandate::new(
            &goal.id,
            None,
            "Publish the synthetic daily report",
            "owner-session",
            MandateAuthority::default(),
            3_600,
            21_600,
            10_800,
        );
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let schedule = crate::traits::GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: "2026-09-01T10:00:00+00:00".to_string(),
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();
        let run = state
            .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
            .await
            .unwrap();
        assert!(state
            .finish_goal_run(
                &run.id,
                "completed",
                Some("Published and verified the post.")
            )
            .await
            .unwrap());
        let tool = ManageMandatesTool::new(state.clone());
        let output = tool
            .call_with(
                &json!({
                    "action": "list",
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(
            output.contains("objective status: mandate clock (next review"),
            "{output}"
        );
        assert!(output.contains("last run completed"), "{output}");
        assert!(
            output.contains("Published and verified the post."),
            "{output}"
        );
        assert!(
            output.contains("run history 1 completed / 0 failed"),
            "{output}"
        );
        // The same line answers the delegation/control question with typed
        // state: this mandate has no owner-confirmed metric loop, no
        // delegated account identities, and no recorded measurements.
        assert!(
            output.contains(
                "control active (bounded; no metric loop; no delegated account; 0 measurements)"
            ),
            "{output}"
        );
    }

    #[tokio::test]
    async fn owner_audit_keeps_scheduled_and_mandate_only_subjects_distinct() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();

        let scheduled = crate::traits::Goal::new_continuous(
            "Publish the synthetic daily digest",
            "owner-session",
            None,
            None,
        );
        state.create_goal(&scheduled).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .create_goal_schedule(&crate::traits::GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: scheduled.id.clone(),
                cron_expr: "0 6 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("daily".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: None,
                next_run_at: "2026-09-02T10:00:00+00:00".to_string(),
                created_at: now.clone(),
                updated_at: now.clone(),
            })
            .await
            .unwrap();

        let controller = crate::traits::Goal::new_continuous(
            "Review synthetic social engagement",
            "owner-session",
            None,
            None,
        );
        let mandate = Mandate::new(
            &controller.id,
            None,
            "Review synthetic social engagement",
            "owner-session",
            MandateAuthority::default(),
            3_600,
            21_600,
            10_800,
        );
        state
            .create_mandate_controller(&controller, &mandate)
            .await
            .unwrap();

        // The owner reaches the daemon through several channel sessions. A
        // scheduled objective created through a sibling owner session is part
        // of the same daemon-wide collection; hiding it while labelling the
        // coverage `complete` would prove a false absence (R53).
        let sibling = crate::traits::Goal::new_continuous(
            "Synthetic schedule created through the coding bot",
            "owner-session-coding-bot",
            None,
            None,
        );
        state.create_goal(&sibling).await.unwrap();
        state
            .create_goal_schedule(&crate::traits::GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: sibling.id.clone(),
                cron_expr: "0 7 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("daily".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: None,
                next_run_at: "2026-09-02T11:00:00+00:00".to_string(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();
        let tool = ManageMandatesTool::new(state);
        let outcome = tool
            .call_with_execution_context(
                &json!({
                    "action": "list",
                })
                .to_string(),
                None,
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let scheduled_id = crate::tools::objective_status::objective_resource_id(&scheduled.id);
        let controller_id = crate::tools::objective_status::objective_resource_id(&controller.id);
        let sibling_id = crate::tools::objective_status::objective_resource_id(&sibling.id);
        assert!(outcome.output.contains(&scheduled_id), "{}", outcome.output);
        assert!(
            outcome.output.contains(&controller_id),
            "{}",
            outcome.output
        );
        assert!(outcome.output.contains(&sibling_id), "{}", outcome.output);
        let scheduled_position = outcome
            .output
            .find(&scheduled_id)
            .expect("scheduled objective row");
        let mandate_section_position = outcome
            .output
            .find("Mandates (")
            .expect("mandate detail section");
        assert!(scheduled_position < mandate_section_position);

        let observations = &outcome.metadata.observations;
        assert!(observations
            .iter()
            .any(|observation| observation.subject.value == scheduled_id));
        assert!(observations
            .iter()
            .any(|observation| observation.subject.value == controller_id));
        assert!(observations
            .iter()
            .any(|observation| observation.subject.value == sibling_id));
        let scheduled_collection = outcome
            .metadata
            .collection_observations
            .iter()
            .find(|collection| {
                collection.collection.value
                    == crate::tools::objective_status::ObjectiveCollection::ScheduledGoals
                        .resource_id()
            })
            .unwrap();
        let mandate_collection = outcome
            .metadata
            .collection_observations
            .iter()
            .find(|collection| {
                collection.collection.value
                    == crate::tools::objective_status::ObjectiveCollection::MandateControllers
                        .resource_id()
            })
            .unwrap();
        let mut scheduled_members = scheduled_collection
            .members
            .iter()
            .map(|member| member.value.clone())
            .collect::<Vec<_>>();
        scheduled_members.sort();
        let mut expected_scheduled = vec![scheduled_id.clone(), sibling_id.clone()];
        expected_scheduled.sort();
        assert_eq!(scheduled_members, expected_scheduled);
        assert_eq!(scheduled_collection.total_count, Some(2));
        assert_eq!(
            scheduled_collection.completeness,
            crate::traits::ToolCollectionCompleteness::Complete
        );
        // Mandate authority stays session-scoped: the collection join does
        // not widen the mandate list itself.
        assert_eq!(mandate_collection.members.len(), 1);
        assert_eq!(mandate_collection.members[0].value, controller_id);
    }

    #[tokio::test]
    async fn mandate_list_stays_session_scoped_while_scheduled_collection_is_daemon_wide() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();

        let other_controller = crate::traits::Goal::new_continuous(
            "Review synthetic inbox triage",
            "owner-session-coding-bot",
            None,
            None,
        );
        let other_mandate = Mandate::new(
            &other_controller.id,
            None,
            "Review synthetic inbox triage",
            "owner-session-coding-bot",
            MandateAuthority::default(),
            3_600,
            21_600,
            10_800,
        );
        state
            .create_mandate_controller(&other_controller, &other_mandate)
            .await
            .unwrap();
        let tool = ManageMandatesTool::new(state);
        let outcome = tool
            .call_with_execution_context(
                &json!({
                    "action": "list",
                })
                .to_string(),
                None,
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(
            !outcome.output.contains(&other_mandate.id),
            "{}",
            outcome.output
        );
        let mandate_collection = outcome
            .metadata
            .collection_observations
            .iter()
            .find(|collection| {
                collection.collection.value
                    == crate::tools::objective_status::ObjectiveCollection::MandateControllers
                        .resource_id()
            })
            .unwrap();
        assert!(mandate_collection.members.is_empty());

        // Non-owner and non-private callers still get nothing at all.
        let refused = tool
            .call_with_execution_context(
                &json!({
                    "action": "list",
                })
                .to_string(),
                None,
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: None,
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(refused.output.contains("only be listed by the owner"));
        assert!(refused.metadata.collection_observations.is_empty());
    }

    #[tokio::test]
    async fn provider_default_filled_wait_payload_commits_exact_decision() {
        use crate::traits::{Goal, Task};

        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous(
            "Review a synthetic bounded source",
            "owner-session",
            Some(250_000),
            Some(2_000_000),
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Review a synthetic bounded source",
            "owner-session",
            MandateAuthority::default(),
            3_600,
            21_600,
            10_800,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let leased = state
            .claim_due_mandates(1, "provider-default-test", 300)
            .await
            .unwrap()
            .pop()
            .expect("one due mandate");
        let run_id = uuid::Uuid::new_v4().to_string();
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let root_task = Task {
            id: root_task_id.clone(),
            goal_id: goal.id.clone(),
            description: "Record one bounded WAIT decision".to_string(),
            status: "pending".to_string(),
            priority: "high".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        let run = state
            .create_mandate_review_run(
                &mandate.id,
                leased.review_lease_token.as_deref().unwrap(),
                &run_id,
                &root_task,
            )
            .await
            .unwrap();
        let attempt = state
            .claim_task_with_lease(&root_task_id, "provider-default-task-lead", None, 300)
            .await
            .unwrap()
            .unwrap();
        let tool = ManageMandatesTool::new(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "record_decision",
                    "outcome": "wait",
                    "activity_level": "quiet",
                    "rationale": "No worthwhile action is available in this review.",
                    "reconsider_minutes": 180,
                    "learning_note": "",
                    "learning_evidence_receipt_ids": [],
                    "strategy_key": "",
                    "strategy_kind": "explore",
                    "strategy_confidence_bps": 0,
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    goal_id: Some((goal.id).to_string()),
                    goal_run_id: Some((run.id).to_string()),
                    task_attempt_id: Some((attempt.id).to_string()),
                    session_id: "synthetic-task-lead".into(),
                    channel_visibility: Some(crate::types::ChannelVisibility::Internal),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains("WAIT recorded"));
        let decision = state
            .get_mandate_decision_for_run(&run.id)
            .await
            .unwrap()
            .expect("durable decision");
        assert_eq!(decision.outcome, MandateDecisionOutcome::Wait);
        assert_eq!(
            decision.rationale,
            "No worthwhile action is available in this review."
        );
        assert!(state
            .list_mandate_learning_notes(&mandate.id, 10)
            .await
            .unwrap()
            .is_empty());
        assert!(state
            .list_current_mandate_strategy(&mandate.id, 10)
            .await
            .unwrap()
            .is_empty());
    }

    /// A started review run for a bounded mandate with one action slot, its
    /// claimed root attempt, and the root task id receipts are filed under.
    struct ReviewRun {
        // Holds the temp database alive for the test's duration.
        _harness: crate::testing::TestHarness,
        state: Arc<crate::state::SqliteStateStore>,
        mandate: Mandate,
        goal_id: String,
        run_id: String,
        attempt_id: String,
        root_task_id: String,
    }

    async fn review_run(control: bool) -> ReviewRun {
        use crate::traits::{Goal, Task};
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous("Post one note about Acme", "owner-session", None, None);
        let authority = MandateAuthority::from_operation_scopes(
            true,
            vec![
                MandateOperationScope {
                    tool: "http_request".to_string(),
                    operation: crate::traits::ToolCallOperation::Get,
                    kind: crate::traits::MandateOperationKind::Observation,
                    target_prefixes: vec!["https://api.example.test/feed".to_string()],
                    allowed_query_params: Vec::new(),
                    mutation_effects: Vec::new(),
                },
                MandateOperationScope {
                    tool: "http_request".to_string(),
                    operation: crate::traits::ToolCallOperation::Post,
                    kind: crate::traits::MandateOperationKind::Mutation,
                    target_prefixes: vec!["https://api.example.test/notes".to_string()],
                    allowed_query_params: Vec::new(),
                    mutation_effects: vec![
                        "remote_mutation".to_string(),
                        "external_delivery".to_string(),
                    ],
                },
            ],
            1,
            4,
            900,
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Post one note about Acme",
            "owner-session",
            authority,
            600,
            3_600,
            600,
        );
        mandate.success_criteria = vec!["The note is grounded in the Acme feed".to_string()];
        if control {
            mandate.objective_control = Some(crate::traits::MandateObjectiveControl {
                schema_version: crate::traits::MandateObjectiveControl::SCHEMA_VERSION,
                metric_name: "notes posted".to_string(),
                unit: "note".to_string(),
                baseline_micros: 0,
                target_micros: 1_000_000,
                direction: crate::traits::ObjectiveMetricDirection::AtLeast,
                measurement_source: "metric_source:acme-notes".to_string(),
                measurement_cadence_secs: 600,
                experiment_cohort: "acme".to_string(),
                experiment_window_secs: 3_600,
                minimum_effect_micros: 1_000_000,
                max_stagnant_measurements: 3,
                run_failure_budget: 3,
                baseline_observed_at: chrono::Utc::now().to_rfc3339(),
            });
        }
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let leased = state
            .claim_due_mandates(1, "receipt-attach-test", 300)
            .await
            .unwrap()
            .pop()
            .expect("one due mandate");
        let run_id = uuid::Uuid::new_v4().to_string();
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let root_task = Task {
            id: root_task_id.clone(),
            goal_id: goal.id.clone(),
            description: "Review the Acme feed".to_string(),
            status: "pending".to_string(),
            priority: "high".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        let run = state
            .create_mandate_review_run(
                &mandate.id,
                leased.review_lease_token.as_deref().unwrap(),
                &run_id,
                &root_task,
            )
            .await
            .unwrap();
        let attempt = state
            .claim_task_with_lease(&root_task_id, "task-lead", None, 300)
            .await
            .unwrap()
            .unwrap();
        ReviewRun {
            _harness: harness,
            state,
            mandate: leased,
            goal_id: goal.id,
            run_id: run.id,
            attempt_id: attempt.id,
            root_task_id,
        }
    }

    /// File one succeeded structured receipt for `tool` under the run.
    async fn file_receipt(run: &ReviewRun, call_id: &str, tool: &str) {
        let receipt = crate::events::ToolReceiptV1::from_metadata(
            &crate::traits::ToolCallMetadata {
                outcome_status: Some(crate::traits::ToolOutcomeStatus::Succeeded),
                semantics: crate::traits::ToolCallSemantics::observation(),
                ..Default::default()
            },
            crate::traits::ToolOutcomeStatus::Succeeded,
            crate::events::ToolOutcomeEvidenceSource::StructuredMetadata,
            None,
        );
        let result = crate::events::ToolResultData {
            message_id: None,
            tool_call_id: call_id.to_string(),
            name: tool.to_string(),
            result: "ok".to_string(),
            success: true,
            duration_ms: 1,
            error: None,
            task_id: Some(run.root_task_id.clone()),
            annotations: vec![],
            turn_id: None,
            attachments: vec![],
            receipt: Some(receipt),
        };
        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at, task_id, tool_name)
             VALUES ('specialist:task_lead:synthetic', 'tool_result', ?, ?, ?, ?)",
        )
        .bind(serde_json::to_string(&result).unwrap())
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(&run.root_task_id)
        .bind(tool)
        .execute(&run.state.pool())
        .await
        .unwrap();
    }

    fn lead_facts(run: &ReviewRun) -> crate::traits::ToolCallContext {
        crate::traits::ToolCallContext {
            goal_id: Some(run.goal_id.clone()),
            goal_run_id: Some(run.run_id.clone()),
            task_attempt_id: Some(run.attempt_id.clone()),
            session_id: "specialist:task_lead:synthetic".into(),
            channel_visibility: Some(ChannelVisibility::Internal),
            ..Default::default()
        }
    }

    fn lead_args(run: &ReviewRun, fields: Value) -> String {
        let mut args = json!({
            "mandate_id": run.mandate.id,
        });
        for (key, value) in fields.as_object().unwrap() {
            args[key] = value.clone();
        }
        args.to_string()
    }

    fn act_fields(run: &ReviewRun) -> Value {
        json!({
            "action": "record_decision",
            "outcome": "act",
            "rationale": "The feed shows a fresh Acme release worth one note",
            "observations": ["The Acme feed lists release 2.0 today"],
            "evidence_receipt_ids": [],
            "intention": "Post one grounded note about Acme release 2.0",
            "value_criterion": run.mandate.success_criteria[0],
            "expected_benefit": "Readers learn about the release the day it ships",
            "risk": "Low: one factual note grounded in the feed",
            "invalidation_criteria": "Do not post if the release left the feed"
        })
    }

    fn tool_for(run: &ReviewRun) -> ManageMandatesTool {
        ManageMandatesTool::new(run.state.clone())
    }

    #[tokio::test]
    async fn record_decision_attaches_current_run_receipts_when_none_cited() {
        let run = review_run(false).await;
        file_receipt(&run, "call_read_acme_feed", "http_request").await;
        file_receipt(&run, "call_create_task", "manage_goal_tasks").await;
        let result = tool_for(&run)
            .call_with(&lead_args(&run, act_fields(&run)), lead_facts(&run))
            .await
            .unwrap();
        assert!(result.contains("ACT committed"), "{result}");
        let decision = run
            .state
            .get_mandate_decision_for_run(&run.run_id)
            .await
            .unwrap()
            .expect("durable decision");
        assert_eq!(
            decision.evidence_receipt_ids,
            vec!["call_read_acme_feed".to_string()]
        );
    }

    /// A committed WAIT under the review's fence finishes the run as
    /// succeeded with the durable decision's summary; an ACT does not, and
    /// neither does a decision recorded without the fence's mandate facts.
    #[tokio::test]
    async fn mandate_decision_finishes_succeeded() {
        let wait = json!({
            "action": "record_decision",
            "outcome": "wait",
            "rationale": "Nothing new is worth saying yet"
        });
        let fenced = |run: &ReviewRun| crate::traits::ToolCallContext {
            mandate_id: Some(run.mandate.id.clone()),
            mandate_version: Some(run.mandate.version),
            ..lead_facts(run)
        };

        let run = review_run(false).await;
        let outcome = tool_for(&run)
            .call_with_execution_context(&lead_args(&run, wait.clone()), None, fenced(&run))
            .await
            .unwrap();
        let decision = run
            .state
            .get_mandate_decision_for_run(&run.run_id)
            .await
            .unwrap()
            .expect("durable decision");
        let finish = outcome.finish.expect("a committed WAIT ends the review");
        assert_eq!(finish.outcome, aidaemon_core::FinishOutcome::Succeeded);
        assert_eq!(finish.reply, decision.non_action_summary());
        assert!(finish.reply.as_deref().is_some_and(
            |reply| reply.starts_with("WAIT recorded: Nothing new is worth saying yet")
        ));

        let unfenced = review_run(false).await;
        let outcome = tool_for(&unfenced)
            .call_with_execution_context(&lead_args(&unfenced, wait), None, lead_facts(&unfenced))
            .await
            .unwrap();
        assert!(outcome.output.contains("WAIT recorded"));
        assert!(outcome.finish.is_none(), "no fence, no finish");

        let act = review_run(false).await;
        file_receipt(&act, "call_read_acme_feed", "http_request").await;
        let outcome = tool_for(&act)
            .call_with_execution_context(&lead_args(&act, act_fields(&act)), None, fenced(&act))
            .await
            .unwrap();
        assert!(outcome.output.contains("ACT committed"));
        assert!(outcome.finish.is_none(), "an ACT review goes on");
    }

    /// A learning note whose receipt ids are omitted cites the decision's
    /// own evidence instead of being dropped with its strategy revision.
    #[tokio::test]
    async fn learning_note_without_receipt_ids_cites_the_decision_evidence() {
        let run = review_run(false).await;
        file_receipt(&run, "call_read_acme_feed", "http_request").await;
        let result = tool_for(&run)
            .call_with(
                &lead_args(
                    &run,
                    json!({
                        "action": "record_decision",
                        "outcome": "wait",
                        "rationale": "The feed has nothing new since the last note",
                        "learning_note": "The Acme feed updates in the morning",
                        "strategy_key": "acme_feed_timing",
                        "strategy_kind": "explore",
                        "strategy_confidence_bps": 6_000
                    }),
                ),
                lead_facts(&run),
            )
            .await
            .unwrap();
        assert!(result.contains("WAIT recorded"), "{result}");
        let notes = run
            .state
            .list_mandate_learning_notes(&run.mandate.id, 10)
            .await
            .unwrap();
        assert_eq!(notes.len(), 1, "the learning note was kept");
        assert_eq!(
            notes[0].evidence_receipt_ids,
            vec!["call_read_acme_feed".to_string()]
        );
        assert_eq!(
            run.state
                .list_current_mandate_strategy(&run.mandate.id, 10)
                .await
                .unwrap()
                .len(),
            1,
            "the strategy revision was kept"
        );
    }

    #[tokio::test]
    async fn act_without_any_current_run_observation_is_still_rejected() {
        let run = review_run(false).await;
        file_receipt(&run, "call_create_task", "manage_goal_tasks").await;
        let error = tool_for(&run)
            .call_with(&lead_args(&run, act_fields(&run)), lead_facts(&run))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("ACT requires at least one current-run sourced observation"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn record_measurement_attaches_current_run_receipts_when_none_cited() {
        let run = review_run(true).await;
        file_receipt(&run, "call_read_acme_feed", "http_request").await;
        let result = tool_for(&run)
            .call_with(
                &lead_args(
                    &run,
                    json!({
                        "action": "record_measurement",
                        "measurement_value_micros": 0,
                        "measurement_confidence_bps": 10_000,
                        "measurement_observed_at": chrono::Utc::now().to_rfc3339(),
                        "evidence_receipt_ids": []
                    }),
                ),
                lead_facts(&run),
            )
            .await
            .unwrap();
        assert!(result.contains("Objective measurement"), "{result}");
        let ids: String = sqlx::query_scalar(
            "SELECT evidence_receipt_ids_json FROM mandate_objective_measurements
             WHERE mandate_id = ?",
        )
        .bind(&run.mandate.id)
        .fetch_one(&run.state.pool())
        .await
        .unwrap();
        assert_eq!(ids, r#"["call_read_acme_feed"]"#);
    }

    #[tokio::test]
    async fn owner_confirmation_activates_one_unscheduled_controller_atomically() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let tool = ManageMandatesTool::new(state.clone());
        let args = r#"{
                    "action":"create",
                    "objective":"Steward @aidaemon_ai thoughtfully",
                    "duration_minutes":60,
                    "expires_at":"2099-01-01T00:00:00Z",
                    "operation_scopes":[{
                        "tool":"http_request",
                        "operation":"POST",
                        "kind":"mutation",
                        "target_prefixes":["https://api.x.com/2/tweets","auth_profile:twitter","account:12345"],
                        "mutation_effects":["remote_mutation","external_delivery"]
                    }],
                    "max_mutating_actions_per_cycle":1,
                    "max_mutating_actions_per_rolling_24h":8,
                    "min_seconds_between_mutations":900,
                    "constraints":["No fabrication"],
                    "success_criteria":["Interventions measurably improve the usefulness of the account without degrading trust"]
                }"#;

        // The kernel asks the owner on the goal card before the call runs.
        let crate::traits::OwnerApprovalNeed::Ask(request) = tool
            .owner_approval(
                &serde_json::from_str::<Value>(args).unwrap(),
                &owner_facts(),
            )
            .await
        else {
            panic!("create asks the owner");
        };
        assert!(matches!(
            ManageMandatesTool::card_kind_for(&request),
            crate::types::ApprovalKind::GoalConfirmation
        ));
        assert!(request.summary.contains("Steward @aidaemon_ai"));
        // Judging stores nothing.
        assert!(state
            .list_mandates(Some("owner-session"), true)
            .await
            .unwrap()
            .is_empty());
        assert!(request.warnings.iter().any(|warning| {
            warning.contains("Access summary:")
                && warning.contains("publish public posts and replies on X")
                && warning.contains("Only account 12345 through the saved X connection")
        }));
        assert!(request
            .warnings
            .iter()
            .any(|warning| warning == "Review depth: Balanced — standard checks before acting"));
        assert!(request
            .warnings
            .iter()
            .any(|warning| warning == "Expiration: 1 hour after activation"));
        assert!(request.warnings.iter().all(|warning| {
            !warning.contains("http_request")
                && !warning.contains("auth_profile:")
                && !warning.contains("api.x.com")
        }));
        assert!(request.warnings.iter().any(|warning| {
            warning.contains("Timing normalized")
                && warning.contains("expires_at value was ignored")
        }));
        let result = tool.call_with(args, approved_owner_facts()).await.unwrap();
        assert!(result.contains("Activated mandate"));

        let mandates = state
            .list_mandates(Some("owner-session"), false)
            .await
            .unwrap();
        assert_eq!(mandates.len(), 1);
        assert_eq!(mandates[0].status, MandateStatus::Active);
        let confirmed_at = chrono::DateTime::parse_from_rfc3339(
            mandates[0]
                .confirmed_at
                .as_deref()
                .expect("confirmation time"),
        )
        .unwrap();
        let expires_at = chrono::DateTime::parse_from_rfc3339(
            mandates[0]
                .expires_at
                .as_deref()
                .expect("activation-relative expiry"),
        )
        .unwrap();
        assert_eq!(
            expires_at.signed_duration_since(confirmed_at).num_seconds(),
            3_600
        );
        assert_ne!(
            mandates[0].expires_at.as_deref(),
            Some("2099-01-01T00:00:00Z")
        );
        let controller = state.get_goal(&mandates[0].goal_id).await.unwrap().unwrap();
        assert_eq!(controller.status, "active");
        assert_eq!(controller.domain, "orchestration");
        assert_eq!(controller.goal_type, "continuous");
        assert_eq!(controller.budget_per_check, Some(250_000));
        assert_eq!(controller.budget_daily, Some(2_000_000));
        assert!(state
            .get_schedules_for_goal(&controller.id)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn a_mandate_keeps_the_owner_given_name_and_is_found_by_it() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let tool = ManageMandatesTool::new(state.clone());
        let create = |name: &str| {
            json!({
                "action": "create",
                "name": name,
                // The objective does not repeat the name: the name is its own
                // typed field, never parsed from text.
                "objective": "Hourly availability checks for the synthetic status page",
                "duration_minutes": 60,
                "default_review_minutes": 60,
                "min_review_minutes": 60,
                "max_review_minutes": 60,
                "operation_scopes": [{
                    "tool": "http_request",
                    "operation": "GET",
                    "kind": "observation",
                    "target_prefixes": ["https://status.example.test/health"],
                    "mutation_effects": []
                }],
                "success_criteria": ["The owner hears about every outage of the status page"]
            })
            .to_string()
        };

        let mut draft: Value = serde_json::from_str(&create("synthetic-watch-2")).unwrap();
        draft["action"] = json!("draft");
        let drafted: Value = serde_json::from_str(
            &tool
                .call_with(&draft.to_string(), approved_owner_facts())
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(drafted["proposal"]["name"], json!("synthetic-watch-2"));

        let crate::traits::OwnerApprovalNeed::Ask(card) = tool
            .owner_approval(
                &serde_json::from_str::<Value>(&create("synthetic-watch-2")).unwrap(),
                &owner_facts(),
            )
            .await
        else {
            panic!("create asks the owner");
        };
        assert!(
            card.summary.contains("\"synthetic-watch-2\""),
            "{}",
            card.summary
        );

        let created = tool
            .call_with(&create("synthetic-watch-2"), approved_owner_facts())
            .await
            .unwrap();
        assert!(created.contains("\"synthetic-watch-2\""), "{created}");
        let stored = state
            .list_mandates(Some("owner-session"), false)
            .await
            .unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].name.as_deref(), Some("synthetic-watch-2"));

        // Listed under its name.
        let listed = tool
            .call_with(
                &json!({"action": "list"}).to_string(),
                approved_owner_facts(),
            )
            .await
            .unwrap();
        assert!(
            listed.contains(&format!("{} \"synthetic-watch-2\"", stored[0].id)),
            "{listed}"
        );

        // A second live mandate cannot take the same name.
        let error = tool
            .call_with(&create("Synthetic-Watch-2"), approved_owner_facts())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("already named"), "{error}");

        // Found and cancelled by the name alone.
        let summary: Value = serde_json::from_str(
            &tool
                .call_with(
                    &json!({"action": "get", "name": "SYNTHETIC-WATCH-2"}).to_string(),
                    approved_owner_facts(),
                )
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(summary["mandate"]["id"], json!(stored[0].id));
        assert_eq!(summary["mandate"]["name"], json!("synthetic-watch-2"));
        tool.call_with(
            &json!({"action": "cancel", "name": "synthetic-watch-2"}).to_string(),
            approved_owner_facts(),
        )
        .await
        .unwrap();
        let after = state.get_mandate(&stored[0].id).await.unwrap().unwrap();
        assert_eq!(after.status, crate::traits::MandateStatus::Cancelled);

        // An unknown name selects nothing.
        let missing = tool
            .call_with(
                &json!({"action": "cancel", "name": "synthetic-watch-9"}).to_string(),
                approved_owner_facts(),
            )
            .await
            .unwrap_err();
        assert!(
            missing.to_string().contains("no mandate named"),
            "{missing}"
        );
    }

    #[tokio::test]
    async fn a_mandate_that_should_tell_the_owner_carries_and_shows_the_alert_grant() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let tool = ManageMandatesTool::new(state.clone());
        let create = |owner_alerts: bool| {
            json!({
                "action": "create",
                "objective": "Check the synthetic status page hourly and tell me when it is down",
                "duration_minutes": 60,
                "default_review_minutes": 60,
                "min_review_minutes": 60,
                "max_review_minutes": 60,
                "operation_scopes": [{
                    "tool": "http_request",
                    "operation": "GET",
                    "kind": "observation",
                    "target_prefixes": ["https://status.example.test/health"],
                    "mutation_effects": []
                }],
                "owner_alerts": owner_alerts,
                "success_criteria": ["The owner hears about every outage of the status page"]
            })
            .to_string()
        };

        // The draft shows the typed grant the model chose.
        let mut draft: Value = serde_json::from_str(&create(true)).unwrap();
        draft["action"] = json!("draft");
        let drafted: Value = serde_json::from_str(
            &tool
                .call_with(&draft.to_string(), approved_owner_facts())
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(drafted["proposal"]["owner_alerts"], json!(true));

        // The owner's card says the automation will tell them what it finds.
        let crate::traits::OwnerApprovalNeed::Ask(request) = tool
            .owner_approval(
                &serde_json::from_str::<Value>(&create(true)).unwrap(),
                &owner_facts(),
            )
            .await
        else {
            panic!("create asks the owner");
        };
        assert!(
            request
                .warnings
                .iter()
                .any(|warning| warning.contains("Tell you:")),
            "{:#?}",
            request.warnings
        );
        let crate::traits::OwnerApprovalNeed::Ask(silent) = tool
            .owner_approval(
                &serde_json::from_str::<Value>(&create(false)).unwrap(),
                &owner_facts(),
            )
            .await
        else {
            panic!("create asks the owner");
        };
        assert!(!silent
            .warnings
            .iter()
            .any(|warning| warning.contains("Tell you:")));

        tool.call_with(&create(true), approved_owner_facts())
            .await
            .unwrap();
        let mandates = state
            .list_mandates(Some("owner-session"), false)
            .await
            .unwrap();
        assert_eq!(mandates.len(), 1);
        assert!(mandates[0].authority.owner_alerts);
        let summary: Value = serde_json::from_str(
            &tool
                .call_with(
                    &json!({"action": "get", "mandate_id": mandates[0].id}).to_string(),
                    approved_owner_facts(),
                )
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(summary["authority_summary"]["owner_alerts"], json!(true));
    }

    #[tokio::test]
    async fn typed_autopilot_mode_gets_distinct_version_bound_confirmation() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let tool = ManageMandatesTool::new(state.clone());
        let args = r#"{
                    "action":"create",
                    "objective":"Maintain the synthetic account without routine hand-holding",
                    "autonomy_mode":"autopilot",
                    "objective_control": {
                        "schema_version": 1,
                        "metric_name": "synthetic useful interactions",
                        "unit": "count",
                        "baseline_micros": 10000000,
                        "target_micros": 20000000,
                        "direction": "at_least",
                        "measurement_source": "metric_source:synthetic-analytics",
                        "measurement_cadence_secs": 3600,
                        "experiment_cohort": "synthetic-cohort-a",
                        "experiment_window_secs": 86400,
                        "minimum_effect_micros": 1000000,
                        "max_stagnant_measurements": 3,
                        "run_failure_budget": 3,
                        "baseline_observed_at": "2026-08-15T00:00:00Z"
                    },
                    "operation_scopes":[{
                        "tool":"http_request",
                        "operation":"POST",
                        "kind":"mutation",
                        "target_prefixes":["https://api.example.test/v1/posts","auth_profile:synthetic-social","account:synthetic-1"],
                        "allowed_query_params":[],
                        "mutation_effects":["remote_mutation","external_delivery"]
                    }],
                    "max_mutating_actions_per_cycle":2,
                    "max_mutating_actions_per_rolling_24h":12,
                    "min_seconds_between_mutations":900,
                    "success_criteria":["Each intervention provides concrete value to the synthetic audience"]
                }"#;

        let crate::traits::OwnerApprovalNeed::Ask(request) = tool
            .owner_approval(
                &serde_json::from_str::<Value>(args).unwrap(),
                &owner_facts(),
            )
            .await
        else {
            panic!("an Autopilot create asks the owner");
        };
        assert!(matches!(
            ManageMandatesTool::card_kind_for(&request),
            crate::types::ApprovalKind::AutopilotConfirmation
        ));
        assert!(request.summary.starts_with("Enable Autopilot:"));
        assert!(request.warnings.iter().any(|warning| {
            warning.contains("Autonomy mode: Autopilot")
                && warning.contains("handles routine decisions")
        }));
        assert!(request.warnings.iter().any(|warning| {
            warning.contains("Policy reference: Version 2")
                && warning.contains("rules and limits summarized here")
        }));
        assert!(request.warnings.iter().any(|warning| {
            warning
                .contains("Only account synthetic-1 through the saved synthetic social connection")
        }));
        assert!(request
            .warnings
            .iter()
            .any(|warning| warning == "Review depth: Thorough — extra care before acting"));
        assert!(request
            .warnings
            .iter()
            .any(|warning| warning.starts_with("Owner checkpoints:")));
        assert!(request
            .warnings
            .iter()
            .any(|warning| warning.starts_with("Recovery policy:")));

        let result = tool.call_with(args, approved_owner_facts()).await.unwrap();
        assert!(result.contains("Autopilot enabled for mandate"));
        let mandate = state
            .list_mandates(Some("owner-session"), false)
            .await
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(mandate.autonomy_mode, MandateAutonomyMode::Autopilot);
        assert_eq!(mandate.review_effort, "thorough");
        assert_eq!(mandate.default_review_secs, AUTOPILOT_DEFAULT_REVIEW_SECS);
        assert_eq!(mandate.version, 2);
    }

    /// A strategy skill edited while the owner's card is open is never
    /// pinned: the commit refuses and nothing is stored.
    #[tokio::test]
    async fn a_strategy_edited_after_the_card_is_refused() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let skills_dir = tempfile::TempDir::new().unwrap();
        let mut skill = crate::skills::Skill {
            name: "acme-stewardship".to_string(),
            description: "Careful account strategy".to_string(),
            triggers: vec![],
            body: "Prefer useful replies.".to_string(),
            origin: Some("custom".to_string()),
            source: Some("filesystem".to_string()),
            source_url: None,
            dir_path: None,
            resources: vec![],
        };
        crate::skills::write_skill_to_file(skills_dir.path(), &skill).unwrap();
        let tool = ManageMandatesTool::new(state.clone())
            .with_strategy_skills(Some(strategy_skills_dir(skills_dir.path().to_path_buf())));
        let expires_at = (chrono::Utc::now() + chrono::Duration::days(30)).to_rfc3339();
        let args = format!(
            r#"{{
                    "action":"create",
                    "objective":"Steward the Acme account",
                    "allow_observations":false,
                    "constraints":["No fabrication"],
                    "success_criteria":["Owner ends stewardship"],
                    "stop_conditions":["Credentials are revoked"],
                    "strategy_skill":"acme-stewardship",
                    "expires_at":"{expires_at}"
                }}"#,
        );
        let crate::traits::OwnerApprovalNeed::Ask(request) = tool
            .owner_approval(
                &serde_json::from_str::<Value>(&args).unwrap(),
                &owner_facts(),
            )
            .await
        else {
            panic!("create asks the owner");
        };
        assert!(request.subject.contains("strategy_sha256"));

        skill.body = "Post as often as possible.".to_string();
        crate::skills::write_skill_to_file(skills_dir.path(), &skill).unwrap();

        let error = tool
            .call_with(&args, approved_owner_facts())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("changed after the owner was asked"),
            "{error}"
        );
        assert!(state
            .list_mandates(Some("owner-session"), true)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn a_fixed_review_interval_is_shown_as_one_value() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let tool = ManageMandatesTool::new(harness.state.clone());
        let expires_at = (chrono::Utc::now() + chrono::Duration::days(1)).to_rfc3339();
        let args = format!(
            r#"{{
                    "action":"create",
                    "objective":"Watch a synthetic endpoint",
                    "allow_observations":false,
                    "constraints":["No fabrication"],
                    "success_criteria":["Owner ends it"],
                    "stop_conditions":["Endpoint retired"],
                    "min_review_minutes":60,
                    "max_review_minutes":60,
                    "default_review_minutes":60,
                    "expires_at":"{expires_at}"
                }}"#,
        );
        let crate::traits::OwnerApprovalNeed::Ask(request) = tool
            .owner_approval(
                &serde_json::from_str::<Value>(&args).unwrap(),
                &owner_facts(),
            )
            .await
        else {
            panic!("create asks the owner");
        };
        let rendered = request.warnings.join("\n");
        assert!(
            rendered.contains("Review interval: Every 1 hour."),
            "{rendered}"
        );
        assert!(!rendered.contains("may vary from"), "{rendered}");
    }

    #[tokio::test]
    async fn pinned_strategy_is_content_addressed_and_confirmation_is_complete() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let skills_dir = tempfile::TempDir::new().unwrap();
        let original_skill = crate::skills::Skill {
            name: "x-stewardship".to_string(),
            description: "Thoughtful account strategy".to_string(),
            triggers: vec![],
            body: "Prefer useful replies and verify outcomes.".to_string(),
            origin: Some("custom".to_string()),
            source: Some("filesystem".to_string()),
            source_url: None,
            dir_path: None,
            resources: vec![],
        };
        crate::skills::write_skill_to_file(skills_dir.path(), &original_skill).unwrap();

        let tool = ManageMandatesTool::new(state.clone())
            .with_strategy_skills(Some(strategy_skills_dir(skills_dir.path().to_path_buf())));
        let expires_at = (chrono::Utc::now() + chrono::Duration::days(30)).to_rfc3339();
        let args = format!(
            r#"{{
                    "action":"create",
                    "objective":"Steward @aidaemon_ai thoughtfully",
                    "allow_observations":false,
                    "constraints":["No fabrication"],
                    "success_criteria":["Owner ends stewardship"],
                    "stop_conditions":["Credentials are revoked"],
                    "strategy_skill":"x-stewardship",
                    "expires_at":"{expires_at}"
                }}"#,
        );

        let crate::traits::OwnerApprovalNeed::Ask(request) = tool
            .owner_approval(
                &serde_json::from_str::<Value>(&args).unwrap(),
                &owner_facts(),
            )
            .await
        else {
            panic!("create asks the owner");
        };
        for expected in [
            "Objective:",
            "Constraints:",
            "Success criteria:",
            "Stop conditions:",
            "Guidance:",
            "Access summary:",
            "Mutation limits:",
            "Review interval:",
            "Expiration:",
            "Review depth:",
            "Policy reference:",
        ] {
            assert!(
                request
                    .warnings
                    .iter()
                    .any(|warning| warning.contains(expected)),
                "missing confirmation field {expected}"
            );
        }
        let rendered = request.warnings.join("\n");
        for internal_term in [
            "sha256:",
            "Allowed targets:",
            "Exact operation scopes:",
            "allowed_query_params",
            "mutation_effects",
            "Confirmation binding:",
        ] {
            assert!(
                !rendered.contains(internal_term),
                "confirmation leaked internal term {internal_term}"
            );
        }

        tool.call_with(&args, approved_owner_facts()).await.unwrap();

        // The snapshot is pinned when the confirmed call stores the mandate:
        // a later filesystem edit does not alter it.
        let changed_skill = crate::skills::Skill {
            body: "A later filesystem edit must not alter the pinned snapshot.".to_string(),
            ..original_skill
        };
        crate::skills::write_skill_to_file(skills_dir.path(), &changed_skill).unwrap();

        let mandate = state
            .list_mandates(Some("owner-session"), false)
            .await
            .unwrap()
            .pop()
            .unwrap();
        let strategy = mandate.strategy.expect("pinned strategy");
        assert_eq!(strategy.skill_name, "x-stewardship");
        assert!(strategy.body.contains("Prefer useful replies"));
        assert!(!strategy.body.contains("later filesystem edit"));
        assert_eq!(strategy.content_sha256.len(), 64);
    }

    async fn draft_with_observation_tool(tool_name: &str) -> Value {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let tool = ManageMandatesTool::new(harness.state.clone());
        let result = tool
            .call_with(
                &json!({
                    "action": "draft",
                    "objective": "Post one note about the Acme feed",
                    "operation_scopes": [
                        {
                            "tool": tool_name,
                            "operation": "GET",
                            "kind": "observation",
                            "target_prefixes": ["http://localhost:8977/feed"]
                        },
                        {
                            "tool": "http_request",
                            "operation": "POST",
                            "kind": "mutation",
                            "target_prefixes": ["http://localhost:8977/notes"],
                            "mutation_effects": ["remote_mutation", "external_delivery"]
                        }
                    ],
                    "max_mutating_actions_per_cycle": 1,
                    "max_mutating_actions_per_rolling_24h": 1,
                    "min_seconds_between_mutations": 900,
                    "success_criteria": ["The note is grounded in the Acme feed"],
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        serde_json::from_str(&result).unwrap()
    }

    #[tokio::test]
    async fn draft_rejects_web_fetch_scope_for_localhost() {
        let value = draft_with_observation_tool("web_fetch").await;
        assert_eq!(value["ready_to_confirm"], false, "{value}");
        assert!(
            value["validation_status"]
                .as_str()
                .unwrap_or_default()
                .contains("use http_request"),
            "{value}"
        );
    }

    #[tokio::test]
    async fn draft_accepts_http_request_scope_for_localhost() {
        let value = draft_with_observation_tool("http_request").await;
        assert!(
            !value["validation_status"]
                .as_str()
                .unwrap_or_default()
                .contains("use http_request"),
            "{value}"
        );
    }

    #[tokio::test]
    async fn draft_reports_missing_authority_without_writing() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let tool = ManageMandatesTool::new(state.clone());
        let result = tool
            .call_with(
                r#"{
                    "action":"draft",
                    "objective":"Manage the account as an ongoing presence",
                    "max_mutating_actions_per_cycle":1
                }"#,
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let value: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(value["execution_mode"], "ongoing_mandate");
        assert_eq!(value["writes_performed"], false);
        assert_eq!(value["ready_to_confirm"], false);
        assert!(value["required_inputs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "operation_scopes"));
        assert!(value["required_inputs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "success_criteria"));
        assert!(state
            .list_mandates(Some("owner-session"), true)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn draft_normalizes_redundant_fixed_expiry_to_relative_duration() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let tool = ManageMandatesTool::new(state.clone());
        let result = tool
            .call_with(
                r#"{
                    "action":"draft",
                    "objective":"Steward the account for exactly 24 hours after activation",
                    "duration_minutes":1440,
                    "expires_at":"2099-01-01T00:00:00Z"
                }"#,
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let value: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(value["proposal"]["duration_minutes"], 1_440);
        assert!(value["proposal"]["expires_at"].is_null());
        assert!(value["proposal"]["timing_normalization"]
            .as_str()
            .unwrap()
            .contains("redundant expires_at value was ignored"));
        assert!(state
            .list_mandates(Some("owner-session"), true)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn draft_uses_human_review_effort_and_hides_token_arithmetic() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let tool = ManageMandatesTool::new(harness.state);
        let result = tool
            .call_with(
                r#"{
                    "action":"draft",
                    "objective":"Maintain a bounded internal posture"
                }"#,
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let value: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(
            value["proposal"]["resource_policy"]["review_effort"],
            "balanced"
        );
        assert_eq!(
            value["proposal"]["resource_policy"]["automatically_managed"],
            true
        );
        assert!(value["proposal"].get("token_budgets").is_none());

        let schema = tool.schema();
        let properties = &schema["parameters"]["properties"];
        assert!(properties.get("review_effort").is_some());
        assert!(properties.get("budget_per_cycle").is_none());
        assert!(properties.get("budget_daily").is_none());

        let error = tool
            .call_with(
                r#"{
                    "action":"draft",
                    "objective":"Do not confuse actions with tokens",
                    "budget_per_cycle":1,
                    "budget_daily":1
                }"#,
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("legacy per-review token capacity"));
    }

    #[tokio::test]
    async fn draft_requires_daily_budget_to_fund_the_review_cadence() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let tool = ManageMandatesTool::new(harness.state);

        let error = tool
            .call_with(
                r#"{
                    "action":"draft",
                    "objective":"Review every thirty minutes for one day",
                    "default_review_minutes":30,
                    "duration_minutes":1440,
                    "budget_per_cycle":30000,
                    "budget_daily":30000
                }"#,
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("cannot fund 48 default review cycle"));

        let defaulted = tool
            .call_with(
                r#"{
                    "action":"draft",
                    "objective":"Use cadence-aware defaults every thirty minutes",
                    "default_review_minutes":30,
                    "duration_minutes":1440
                }"#,
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let defaulted: Value = serde_json::from_str(&defaulted).unwrap();
        assert_eq!(
            defaulted["proposal"]["resource_policy"]["review_effort"],
            "balanced"
        );
        assert_eq!(
            defaulted["proposal"]["resource_policy"]["cadence_funded"],
            true
        );

        let result = tool
            .call_with(
                r#"{
                    "action":"draft",
                    "objective":"Review every thirty minutes for one day",
                    "default_review_minutes":30,
                    "duration_minutes":1440,
                    "budget_per_cycle":30000,
                    "budget_daily":1440000
                }"#,
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let value: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(
            value["proposal"]["resource_policy"]["review_effort"],
            "legacy_custom"
        );
        assert_eq!(
            value["proposal"]["resource_policy"]["automatically_managed"],
            false
        );
    }

    #[tokio::test]
    async fn create_rejects_legacy_cross_product_authority() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let tool = ManageMandatesTool::new(harness.state);
        let error = tool
            .call_with(
                r#"{
                    "action":"create",
                    "objective":"Do not combine unrelated reads and writes",
                    "allowed_tools":["http_request","web_fetch"],
                    "allowed_mutation_effects":["remote_mutation","external_delivery"],
                    "allowed_target_prefixes":["https://blog.aidaemon.ai/posts/","https://api.x.com/2/tweets"],
                    "max_mutating_actions_per_cycle":1,
                    "max_mutating_actions_per_rolling_24h":1,
                    "min_seconds_between_mutations":900
                }"#, crate::traits::ToolCallContext { session_id: "owner-session".into(), user_role: Some(crate::types::UserRole::Owner), channel_visibility: Some(crate::types::ChannelVisibility::Private), ..Default::default() })
            .await
            .unwrap_err();
        assert!(error.to_string().contains("operation_scopes"));
    }

    #[tokio::test]
    async fn create_only_fields_are_described_as_create_only() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let tool = ManageMandatesTool::new(harness.state);
        let schema = tool.schema();
        for field in ["source_goal_id", "priority", "duration_minutes"] {
            let description = schema["parameters"]["properties"][field]["description"]
                .as_str()
                .expect("create-only field description");
            assert!(description.contains("Create only"), "{field}");
            let description = description.to_ascii_lowercase();
            assert!(description.contains("update"), "{field}");
            assert!(description.contains("ignore"), "{field}");
        }
    }

    #[tokio::test]
    async fn policy_get_reports_typed_success_when_policy_mentions_unauthorized_behavior() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = crate::traits::Goal::new_continuous(
            "Inspect a mandate policy",
            "owner-session",
            Some(100_000),
            Some(1_000_000),
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Review a bounded source",
            "owner-session",
            MandateAuthority::default(),
            15 * 60,
            24 * 60 * 60,
            4 * 60 * 60,
        );
        mandate
            .stop_conditions
            .push("Any unauthorized API behavior.".to_string());
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let tool = ManageMandatesTool::new(state);
        let outcome = tool
            .call_with_execution_context(
                &json!({
                    "action": "get",
                    "mandate_id": mandate.id,
                    "section": "policy",
                })
                .to_string(),
                None,
                crate::traits::ToolCallContext {
                    session_id: "owner-session".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        assert!(outcome.output.contains("unauthorized API behavior"));
    }

    #[tokio::test]
    async fn owner_confirmed_effort_update_manages_controller_capacity() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = crate::traits::Goal::new_continuous(
            "Repair a mandate controller budget",
            "owner-session",
            Some(100_000),
            Some(1_000_000),
        );
        let mandate = Mandate::new(
            &goal.id,
            None,
            "Review a bounded source",
            "owner-session",
            MandateAuthority::default(),
            15 * 60,
            24 * 60 * 60,
            4 * 60 * 60,
        );
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        let tool = ManageMandatesTool::new(state.clone());
        let update = json!({
            "action": "update",
            "mandate_id": mandate.id,
            "review_effort": "thorough",
            "source_goal_id": "",
            "priority": "critical",
            "duration_minutes": 1,
            "expires_at": "2099-02-15T13:31:00Z",
        });
        let crate::traits::OwnerApprovalNeed::Ask(request) =
            tool.owner_approval(&update, &owner_facts()).await
        else {
            panic!("an update asks the owner");
        };
        assert!(request
            .warnings
            .iter()
            .any(|warning| { warning == "Review depth: Thorough — extra care before acting" }));
        // Judging changes nothing.
        assert_eq!(
            state
                .get_mandate(&mandate.id)
                .await
                .unwrap()
                .unwrap()
                .version,
            mandate.version
        );
        let result = tool
            .call_with(&update.to_string(), approved_owner_facts())
            .await
            .unwrap();
        assert!(result.contains("thorough review effort"));

        let updated_goal = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(updated_goal.budget_per_check, Some(500_000));
        assert_eq!(updated_goal.budget_daily, Some(4_000_000));
        assert_eq!(updated_goal.priority, goal.priority);
        let updated_mandate = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(updated_mandate.version, mandate.version + 1);
        assert_eq!(updated_mandate.review_effort, "thorough");
        assert_eq!(updated_mandate.source_goal_id, mandate.source_goal_id);
        assert_eq!(
            updated_mandate.expires_at.as_deref(),
            Some("2099-02-15T13:31:00Z")
        );
    }

    #[tokio::test]
    async fn policy_view_round_trips_review_bounds_into_update() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = crate::traits::Goal::new_continuous(
            "Steward a synthetic account",
            "owner-session",
            Some(100_000),
            Some(1_000_000),
        );
        let mandate = Mandate::new(
            &goal.id,
            None,
            "Review a bounded source",
            "owner-session",
            MandateAuthority::default(),
            15 * 60,
            24 * 60 * 60,
            4 * 60 * 60,
        );
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let tool = Arc::new(ManageMandatesTool::new(state.clone()));
        let get = json!({"action": "get", "mandate_id": mandate.id, "section": "policy"});
        let policy: Value = serde_json::from_str(
            &tool
                .call_with(&get.to_string(), owner_facts())
                .await
                .unwrap(),
        )
        .expect("policy view is JSON");
        assert_eq!(policy["min_review_minutes"], 15);
        assert_eq!(policy["default_review_minutes"], 240);
        assert_eq!(policy["max_review_minutes"], 1440);

        // A structured-output provider fills every field; without the bounds
        // in the policy view it invented the schema minimum (1 minute).
        let placeholder = json!({
            "action": "update",
            "mandate_id": mandate.id,
            "review_effort": "balanced",
            "min_review_minutes": 1,
            "default_review_minutes": 1,
            "max_review_minutes": 1
        });
        let error = tool
            .call_with(&placeholder.to_string(), owner_facts())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("default_review_minutes is too short"),
            "{error}"
        );
        assert!(!error.contains("legacy"), "{error}");

        let mut copied =
            json!({"action": "update", "mandate_id": mandate.id, "review_effort": "thorough"});
        for field in [
            "min_review_minutes",
            "default_review_minutes",
            "max_review_minutes",
        ] {
            copied[field] = policy[field].clone();
        }
        tool.call_with(&copied.to_string(), approved_owner_facts())
            .await
            .unwrap();
        let updated = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(updated.min_review_secs, mandate.min_review_secs);
        assert_eq!(updated.default_review_secs, mandate.default_review_secs);
        assert_eq!(updated.max_review_secs, mandate.max_review_secs);
    }

    #[tokio::test]
    async fn get_defaults_to_compact_runtime_summary_without_strategy_body() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = crate::traits::Goal::new_continuous(
            "Compact mandate status",
            "owner-session",
            None,
            None,
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Expose durable runtime state without spilling",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        let body_marker = "STRATEGY_BODY_MUST_NOT_APPEAR";
        mandate.strategy = Some(MandateStrategySnapshot {
            skill_name: "large-strategy".to_string(),
            snapshot_version: MandateStrategySnapshot::SCHEMA_VERSION,
            content_sha256: "a".repeat(64),
            description: "A deliberately large persisted strategy".to_string(),
            body: body_marker.repeat(500),
            source: Some("test".to_string()),
        });
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let tool = ManageMandatesTool::new(state);
        let result = tool
            .call_with(
                &format!(r#"{{"action":"get","mandate_id":"{}"}}"#, mandate.id),
                owner_facts(),
            )
            .await
            .unwrap();
        let value: Value = serde_json::from_str(&result).unwrap();

        assert_eq!(value["section"], "summary");
        assert_eq!(value["mandate"]["controller_goal_id"], goal.id);
        assert_eq!(
            value["owner_input_contract"]["answer_question_changes_authority"],
            false
        );
        assert_eq!(
            value["owner_input_contract"]["authority_changes_require_confirmed_update"],
            true
        );
        assert!(value.get("latest_decision").is_some());
        assert!(value.get("latest_mutation_receipts").is_some());
        assert!(!result.contains(body_marker));
        assert!(result.len() < 8_000, "summary was {} bytes", result.len());
    }

    #[tokio::test]
    async fn resume_requires_durable_confirmation_not_only_a_paused_status() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = crate::traits::Goal::new_continuous_pending(
            "Unconfirmed mandate controller",
            "owner-session",
            None,
            None,
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Do not resume without confirmation",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.status = MandateStatus::Paused;
        mandate.confirmed_at = None;
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let tool = ManageMandatesTool::new(state);
        let error = tool
            .call_with(
                &format!(r#"{{"action":"resume","mandate_id":"{}"}}"#, mandate.id),
                owner_facts(),
            )
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("unconfirmed mandate cannot be resumed"));
    }

    /// `cancel` takes effect without an approval prompt, so the tool itself
    /// must refuse anyone but the verified private owner.
    #[tokio::test]
    async fn cancel_requires_a_verified_private_owner_in_the_tool() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = crate::traits::Goal::new_continuous(
            "Active mandate controller",
            "owner-session",
            None,
            None,
        );
        let mandate = Mandate::new(
            &goal.id,
            None,
            "Keep running until the owner cancels",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let before = state
            .get_mandate(&mandate.id)
            .await
            .unwrap()
            .unwrap()
            .status;
        assert!(!before.is_terminal());
        let tool = ManageMandatesTool::new(state.clone());
        for (role, visibility) in [
            (
                crate::types::UserRole::Guest,
                crate::types::ChannelVisibility::Private,
            ),
            (
                crate::types::UserRole::Owner,
                crate::types::ChannelVisibility::Public,
            ),
        ] {
            let result = tool
                .call_with(
                    &format!(r#"{{"action":"cancel","mandate_id":"{}"}}"#, mandate.id),
                    crate::traits::ToolCallContext {
                        user_role: Some(role),
                        channel_visibility: Some(visibility),
                        ..owner_facts()
                    },
                )
                .await
                .unwrap();
            assert!(
                result.contains("only be changed by the owner in a verified private channel"),
                "{role}/{visibility}: {result}"
            );
            assert_eq!(
                state
                    .get_mandate(&mandate.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                before,
                "{role}/{visibility} must not cancel"
            );
        }
        let result = tool
            .call_with(
                &format!(r#"{{"action":"cancel","mandate_id":"{}"}}"#, mandate.id),
                owner_facts(),
            )
            .await
            .unwrap();
        assert!(result.contains("is now cancelled"), "{result}");
    }

    #[tokio::test]
    async fn ask_resume_requires_and_persists_owner_guidance() {
        use crate::traits::{
            Goal, Mandate, MandateAuthority, MandateDecisionCycle, MandateDecisionOutcome,
            MandateRunFinalizationRequest, MandateRunFinalizationResult, Task,
        };

        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous("Await owner judgment", "owner-session", None, None);
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Use owner judgment when quality is ambiguous",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::minutes(1)).to_rfc3339();
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        state
            .claim_due_mandates(1, "test-heartbeat", 300)
            .await
            .unwrap();
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let run = state
            .start_goal_run(&goal.id, "mandate", None, Some(&root_task_id))
            .await
            .unwrap();
        let created_at = chrono::Utc::now().to_rfc3339();
        state
            .create_task(&Task {
                id: root_task_id.clone(),
                goal_id: goal.id.clone(),
                description: "Deliberate and record one ASK decision".to_string(),
                status: "pending".to_string(),
                priority: "high".to_string(),
                task_order: 0,
                parallel_group: None,
                depends_on: None,
                agent_id: None,
                context: None,
                result: None,
                error: None,
                blocker: None,
                idempotent: false,
                retry_count: 0,
                max_retries: 0,
                created_at,
                started_at: None,
                completed_at: None,
            })
            .await
            .unwrap();
        let root_attempt = state
            .claim_task_with_lease(
                &root_task_id,
                "ask-resume-task-lead",
                Some("profile-task-lead"),
                7_200,
            )
            .await
            .unwrap()
            .unwrap();
        let mut ask = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Ask,
            "Owner preference is material",
            mandate.version,
        );
        ask.question = Some("Should replies be preferred?".to_string());
        state
            .record_mandate_decision(&ask, None, Some(&root_attempt.id))
            .await
            .unwrap();
        assert!(state
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("ASK decision durably recorded".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        assert!(matches!(
            state
                .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
                    mandate_id: mandate.id.clone(),
                    expected_mandate_version: mandate.version,
                    goal_run_id: run.id.clone(),
                    finalized_at: chrono::Utc::now().to_rfc3339(),
                })
                .await
                .unwrap(),
            MandateRunFinalizationResult::NonActionSatisfied {
                outcome: MandateDecisionOutcome::Ask,
                ..
            }
        ));
        let tool = ManageMandatesTool::new(state.clone());
        let missing = tool
            .call_with(
                &format!(
                    r#"{{"action":"answer_question","mandate_id":"{}"}}"#,
                    mandate.id
                ),
                owner_facts(),
            )
            .await;
        assert!(missing
            .unwrap_err()
            .to_string()
            .contains("guidance is required"));

        let resumed = tool
            .call_with(&format!(
                r#"{{"action":"answer_question","mandate_id":"{}","guidance":"Prefer thoughtful replies over original posts."}}"#,
                mandate.id
            ), owner_facts())
            .await
            .unwrap();
        assert!(resumed.contains("is active"));
        assert!(resumed.contains("Immutable authority is unchanged"));
        assert!(resumed.contains("owner-confirmed update workflow"));
        let resumed_mandate = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(resumed_mandate.status, MandateStatus::Active);
        assert_eq!(resumed_mandate.authority, mandate.authority);
        let resumed_review_at =
            chrono::DateTime::parse_from_rfc3339(&resumed_mandate.next_review_at)
                .unwrap()
                .with_timezone(&chrono::Utc);
        assert!(resumed_review_at <= chrono::Utc::now() + chrono::Duration::seconds(2));
        assert!(resumed_review_at >= chrono::Utc::now() - chrono::Duration::seconds(2));
        let controller = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(controller.status, "active");
        assert!(controller
            .context
            .as_deref()
            .unwrap()
            .contains("Prefer thoughtful replies"));

        // An unanswered ASK resumes on its own, so an answer can arrive while
        // the mandate is already active. It still lands as bounded guidance.
        let late = tool
            .call_with(&format!(
                r#"{{"action":"answer_question","mandate_id":"{}","guidance":"Keep posts under two sentences."}}"#,
                mandate.id
            ), owner_facts())
            .await
            .unwrap();
        assert!(late.contains("late answer"), "{late}");
        assert!(late.contains("Immutable authority is unchanged"));
        let after_late = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(after_late.status, MandateStatus::Active);
        assert_eq!(after_late.authority, mandate.authority);
        let context = state
            .get_goal(&goal.id)
            .await
            .unwrap()
            .unwrap()
            .context
            .unwrap();
        assert!(context.contains("Prefer thoughtful replies"));
        assert!(context.contains("Keep posts under two sentences"));
    }

    #[tokio::test]
    async fn stale_internal_decision_cannot_rebind_to_a_newer_goal_run() {
        use crate::traits::{Goal, Mandate, MandateAuthority};

        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous("Pinned decision cycle", "owner-session", None, None);
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Never move a stale decision onto a newer run",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::minutes(1)).to_rfc3339();
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        state
            .claim_due_mandates(1, "pinned-run-test", 300)
            .await
            .unwrap();

        let stale_run = state
            .start_goal_run(&goal.id, "mandate", None, None)
            .await
            .unwrap();
        state
            .finish_goal_run(&stale_run.id, "failed", Some("superseded"))
            .await
            .unwrap();
        let current_run = state
            .start_goal_run(&goal.id, "mandate", None, None)
            .await
            .unwrap();
        let tool = ManageMandatesTool::new(state.clone());
        let error = tool
            .call_with(
                r#"{"action":"record_decision","outcome":"wait","rationale":"stale"}"#,
                crate::traits::ToolCallContext {
                    goal_id: Some(goal.id.clone()),
                    goal_run_id: Some(stale_run.id.clone()),
                    task_attempt_id: Some("stale-attempt".into()),
                    session_id: "sub-task-lead".into(),
                    channel_visibility: Some(crate::types::ChannelVisibility::Internal),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("no longer executable"));
        assert!(state
            .get_mandate_decision_for_run(&current_run.id)
            .await
            .unwrap()
            .is_none());
    }
}
