//! `track_requirements` lives in `aidaemon_work::tools::track_requirements`; the tests here run it against the daemon's assembled state store.

pub use aidaemon_work::tools::track_requirements::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plans::PlanStore;
    use crate::testing::CallWithFacts;
    use crate::traits::StableObservationSubject;
    use crate::traits::Tool;
    use crate::traits::ToolSemanticFacet;
    use crate::traits::ToolSemantics;
    use serde_json::json;
    use serde_json::Value;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::sync::Arc;

    async fn test_tool() -> (TrackRequirementsTool, Arc<PlanStore>) {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let plan_store = Arc::new(PlanStore::new(pool).await.unwrap());
        // The tool only persists checklist state; UI delivery is handled by the
        // agent loop via StatusUpdate::Checklist, so there is no hub to wire here.
        (TrackRequirementsTool::new(plan_store.clone()), plan_store)
    }

    /// The card's header follows the honest outcome (live 2026-11-12: step 3
    /// found no price, yet the card read "Done · 3 of 3 steps"). An item
    /// tried and not achieved is not done, and a partial or blocked verdict
    /// never reads as done.
    #[tokio::test]
    async fn checklist_header_follows_unmet_items_and_the_claimed_outcome() {
        let (tool, _plan_store) = test_tool().await;
        let failed_item = json!({
            "items": [
                {"text": "Say the date", "status": "completed"},
                {"text": "Multiply the numbers", "status": "completed"},
                {"text": "Find the synthetic quote", "status": "failed"}
            ],
            "outcome": "done"
        })
        .to_string();
        tool.validate_arguments(&failed_item).unwrap();
        let statuses = &tool.schema()["parameters"]["properties"]["items"]["items"]["properties"]
            ["status"]["enum"];
        assert!(statuses.as_array().unwrap().contains(&json!("failed")));
        let out = tool
            .call_with(&failed_item, crate::testing::call_facts("sess-failed"))
            .await
            .unwrap();
        assert!(!out.contains("Done ·"), "{out}");
        // Ticked on the model's word only: done. Neither changes anything,
        // so neither needs a receipt and the card adds no note.
        assert!(out.contains("Stopped · 2 of 3 done\n"), "{out}");
        assert!(!out.contains("verify"), "{out}");

        let partial = json!({
            "items": [
                {"text": "Say the date", "status": "completed"},
                {"text": "Find the synthetic quote", "status": "completed"}
            ],
            "outcome": "partial"
        })
        .to_string();
        let out = tool
            .call_with(&partial, crate::testing::call_facts("sess-partial"))
            .await
            .unwrap();
        assert!(!out.contains("Done ·"), "{out}");
        assert!(out.contains("Stopped short · 2 of 2 done\n"), "{out}");

        let done = partial.replace("partial", "done");
        let out = tool
            .call_with(&done, crate::testing::call_facts("sess-done"))
            .await
            .unwrap();
        assert!(out.contains("✅ Done · 2 of 2 steps"), "{out}");

        // An action step that declared a side effect and that nothing
        // verified is the one step the card flags, in plain words.
        let action = json!({
            "items": [
                {"text": "Say the date", "status": "completed"},
                {"text": "Save the quote to quote.txt", "status": "completed",
                 "mutation_effects": ["local_workspace_write"]}
            ],
            "outcome": "done"
        })
        .to_string();
        let out = tool
            .call_with(&action, crate::testing::call_facts("sess-action"))
            .await
            .unwrap();
        assert!(
            out.contains("☑️ Finished · 2 of 2 done; I couldn't verify 1"),
            "{out}"
        );
        assert!(
            out.contains("☑️ Save the quote to quote.txt (I couldn't verify this one)"),
            "{out}"
        );
        assert!(out.contains("✅ Say the date"), "{out}");
        assert!(!out.to_lowercase().contains("tool"), "{out}");
    }

    /// Steps left open while the chat's background sub-agents run read as
    /// continuing in the background, never "stopped short", even when no
    /// launch named them (typed: the running records, not the wording).
    #[tokio::test]
    async fn open_steps_wait_on_running_background_sub_agents() {
        let (tool, _plan_store) = test_tool().await;
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let state: Arc<dyn crate::traits::StateStore> = Arc::new(
            crate::state::SqliteStateStore::for_tests(
                &db_file.path().display().to_string(),
                100,
                None,
                Arc::new(crate::memory::embeddings::EmbeddingService::deterministic()),
            )
            .await
            .unwrap(),
        );
        state
            .record_background_spawn(&crate::traits::BackgroundSpawn {
                id: "background_spawn:synthetic".to_string(),
                session_id: "sess-bg".to_string(),
                mission: "Install a synthetic site".to_string(),
                started_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();
        let tool = tool.with_state(state);
        let args = json!({
            "items": [
                {"text": "Check the tools", "status": "completed"},
                {"text": "Install the synthetic site", "status": "deferred"}
            ],
            "outcome": "partial"
        })
        .to_string();
        let out = tool
            .call_with(&args, crate::testing::call_facts("sess-bg"))
            .await
            .unwrap();
        assert!(
            out.contains("Continuing in the background · 1 of 2 done\n"),
            "{out}"
        );
        assert!(
            !out.contains("Stopped short") && !out.contains("not done"),
            "{out}"
        );

        // Another chat with nothing running keeps its own verdict.
        let out = tool
            .call_with(&args, crate::testing::call_facts("sess-quiet"))
            .await
            .unwrap();
        assert!(out.contains("Stopped short"), "{out}");
    }

    /// A step with nothing to do is finished, not failed: the card reads
    /// done and can go once the reply lands.
    #[tokio::test]
    async fn a_step_with_nothing_to_do_reads_as_not_needed() {
        let (tool, _plan_store) = test_tool().await;
        let args = json!({
            "items": [
                {"text": "Find your open orders", "status": "completed"},
                {"text": "Cancel the open orders", "status": "skipped"}
            ],
            "outcome": "done"
        })
        .to_string();
        tool.validate_arguments(&args).unwrap();
        let out = tool
            .call_with(&args, crate::testing::call_facts("sess-skip"))
            .await
            .unwrap();
        assert!(out.contains("✅ Done · 1 done, 1 not needed"), "{out}");
        assert!(
            out.contains("➖ Cancel the open orders (not needed)"),
            "{out}"
        );
        assert!(!out.contains('⚠'), "{out}");
    }

    /// Read-only correctly kept a write from happening: the card says why it
    /// was not done instead of "❌ Stopped".
    #[tokio::test]
    async fn read_only_holds_steps_back_instead_of_failing_them() {
        let (tool, plan_store) = test_tool().await;
        let args = json!({
            "items": [
                {"text": "Read your config", "status": "completed"},
                {"text": "Write the new setting", "status": "failed"}
            ],
            "outcome": "blocked"
        })
        .to_string();
        let mut facts = crate::testing::call_facts("sess-ro");
        facts.mutation_forbidden = true;
        let out = tool.call_with(&args, facts).await.unwrap();
        assert!(
            out.contains("⏭️ Not done: read-only is on · 1 of 2 done\n"),
            "{out}"
        );
        assert!(
            out.contains("⏭️ Write the new setting (not done: read-only is on)"),
            "{out}"
        );
        assert!(!out.contains('❌') && !out.contains("Stopped"), "{out}");
        let plan = plan_store
            .get_recent_for_session("sess-ro", 1)
            .await
            .unwrap();
        assert_eq!(
            plan[0].steps[1].not_done,
            Some(crate::plans::NotDoneReason::ChangesOff)
        );
    }

    /// One step is the request itself: no card, the reply says how it went.
    #[tokio::test]
    async fn a_one_step_checklist_gets_no_card() {
        let (tool, _plan_store) = test_tool().await;
        let args = json!({"items": [{"text": "Save your test name", "status": "in_progress"}]})
            .to_string();
        let out = tool
            .call_with(&args, crate::testing::call_facts("sess-one"))
            .await
            .unwrap();
        let (_, after_header) = out.split_once(CHECKLIST_HEADER).expect("header");
        let (_, card) = after_header.split_once(":\n").expect("card");
        assert!(card.trim().is_empty(), "{out}");
        assert!(out.contains("Save your test name"), "{out}");
    }

    #[tokio::test]
    async fn test_track_requirements_persists_and_returns_rendered_list() {
        let (tool, plan_store) = test_tool().await;
        let args = json!({
            "items": [
                {"text": "create script", "status": "in_progress"},
                {"text": "send the file", "status": "pending"}
            ]
        })
        .to_string();
        let out = tool
            .call_with(&args, crate::testing::call_facts("sess-1"))
            .await
            .unwrap();
        assert!(out.contains("create script"));
        assert!(out.contains("send the file"));
        let plan = plan_store
            .get_incomplete_for_session("sess-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.unchecked_steps().len(), 2);
    }

    #[tokio::test]
    async fn test_track_requirements_marks_completed_on_update() {
        let (tool, plan_store) = test_tool().await;
        let mk = |status: &str| {
            json!({
                "items": [{"text": "send the file", "status": status}]
            })
            .to_string()
        };
        tool.call_with(&mk("pending"), crate::testing::call_facts("sess-2"))
            .await
            .unwrap();
        // While still pending it is the session's active (incomplete) checklist.
        let id = plan_store
            .get_incomplete_for_session("sess-2")
            .await
            .unwrap()
            .unwrap()
            .id;

        tool.call_with(&mk("completed"), crate::testing::call_facts("sess-2"))
            .await
            .unwrap();
        // Once all items are completed the plan is auto-completed (no longer
        // "incomplete"), so fetch it by id to verify the final state.
        let plan = plan_store.get(&id).await.unwrap().unwrap();
        assert_eq!(plan.unchecked_steps().len(), 0);
        assert_eq!(plan.completed_steps(), 1);
    }

    fn synthetic_vocabulary() -> StableSubjectVocabulary {
        StableSubjectVocabulary::from_subjects(vec![
            StableObservationSubject::collection(
                "objective_collection:scheduled_goals",
                &[ToolSemanticFacet::Schedule, ToolSemanticFacet::RunState],
                "objective:",
                "every scheduled goal",
            ),
            StableObservationSubject::collection(
                "objective_collection:mandate_controllers",
                &[ToolSemanticFacet::Control],
                "objective:",
                "every mandate controller",
            ),
            StableObservationSubject::collection(
                "auth_profile_collection:configured",
                &[ToolSemanticFacet::Authorization],
                "auth_profile:",
                "every configured API auth profile",
            ),
            StableObservationSubject::namespace("res_", "session resource handles"),
        ])
    }

    /// The live 2026-09-01 audit shape: two items bound to advertised
    /// collections and a third bound to an invented credential-state ID that
    /// no adapter reports.
    fn audit_checklist_arguments(status: &str) -> Value {
        json!({
            "items": [
                {
                    "text": "Audit the schedule state and latest run outcome.",
                    "status": status,
                    "requires_observation": true,
                    "targets": [{
                        "subject": {"kind": "resource_id", "value": "objective_collection:scheduled_goals"},
                        "facets": ["schedule", "run_state"],
                        "collection_coverage": {
                            "collection": {"kind": "resource_id", "value": "objective_collection:scheduled_goals"},
                            "minimum_completeness": "complete"
                        }
                    }]
                },
                {
                    "text": "Audit objective-control presence.",
                    "status": status,
                    "requires_observation": true,
                    "targets": [{
                        "subject": {"kind": "resource_id", "value": "objective_collection:mandate_controllers"},
                        "facets": ["control"],
                        "collection_coverage": {
                            "collection": {"kind": "resource_id", "value": "objective_collection:mandate_controllers"},
                            "minimum_completeness": "complete"
                        }
                    }]
                },
                {
                    "text": "Audit owner configuration and credential readiness.",
                    "status": status,
                    "requires_observation": true,
                    "targets": ["x_owner_credential_state"]
                }
            ]
        })
    }

    #[tokio::test]
    async fn unbound_resource_id_target_is_dropped_and_reported() {
        let (tool, plan_store) = test_tool().await;
        let tool = tool.with_stable_subjects(synthetic_vocabulary());
        let raw = audit_checklist_arguments("in_progress").to_string();

        let canonical = tool.canonicalize_arguments(&raw).unwrap();
        let canonical_value: Value = serde_json::from_str(&canonical).unwrap();
        let items = canonical_value["items"].as_array().unwrap();
        assert_eq!(
            items[0]["targets"],
            audit_checklist_arguments("in_progress")["items"][0]["targets"],
            "an advertised collection binding must survive untouched"
        );
        assert_eq!(items[1]["targets"].as_array().unwrap().len(), 1);
        assert_eq!(
            items[2]["targets"].as_array().map(Vec::len),
            Some(0),
            "an invented resource ID can never be credited, so it must not become an exact obligation"
        );
        assert_eq!(
            items[2][UNBOUND_TARGETS_KEY],
            json!(["x_owner_credential_state"]),
            "the drop must be recorded, never silent"
        );

        let facts = crate::traits::ToolCallContext {
            task_id: Some("task-audit".into()),
            ..crate::testing::call_facts("sess-audit")
        };
        let out = tool.call_with(&canonical, facts).await.unwrap();
        assert!(
            out.contains("x_owner_credential_state"),
            "the model must be told which target was unbound: {out}"
        );
        assert!(
            out.contains("auth_profile_collection:configured") && out.contains("objective:"),
            "the notice must name the stable subjects the model can bind instead: {out}"
        );
        assert!(
            out.contains("Checklist updated (0/3 done):\n"),
            "the checklist header must remain the live-surface delimiter: {out}"
        );
        let plan = plan_store
            .get_incomplete_for_session("sess-audit")
            .await
            .unwrap()
            .unwrap();
        assert!(plan.steps[2].expected_targets.is_empty());
        assert_eq!(
            plan.steps[0].expected_targets,
            ["objective_collection:scheduled_goals"]
        );
    }
}
