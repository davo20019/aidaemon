// ==================== An instruction is carried out, not reported on ====================
//
// Live 2026-11-13: after `/clear`, "install Drupal and WordPress with ddev"
// was declared as two installs, the run only listed the sites a past run had
// made, re-declared the installs blocked, and answered with their state. The
// run declared work and attempted none of it: the closeout sends it back once
// to do the work or ask the owner (typed `needs_owner`), and judges that from
// receipts, never from the reply. A question about state still answers
// directly.

/// A read-only listing, as `ddev list` is typed: an observation, never an
/// attempt at the work.
struct SyntheticSiteList;

#[async_trait::async_trait]
impl crate::traits::Tool for SyntheticSiteList {
    fn name(&self) -> &str {
        "synthetic_site_list"
    }

    fn description(&self) -> &str {
        "Lists the synthetic sites"
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "synthetic_site_list",
            "description": "Lists the synthetic sites",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
        })
    }

    async fn execute(
        &self,
        _arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        Ok(crate::traits::ToolCallOutcome::from_output(
            "synthetic-wp paused; synthetic-drupal paused".to_string(),
        ))
    }
}

impl crate::traits::ToolSemantics for SyntheticSiteList {
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: true,
            idempotent: true,
            ..Default::default()
        }
    }
}

async fn unattempted_work_harness(
    provider: MockProvider,
) -> crate::testing::TestHarness {
    let probe: Arc<dyn crate::traits::Tool> = Arc::new(SyntheticSiteList);
    setup_test_agent_root_with_extra_tools_and_llm_timeout(provider, vec![probe], None)
        .await
        .unwrap()
}

async fn unattempted_work_task_end(
    harness: &crate::testing::TestHarness,
    session: &str,
) -> crate::events::TaskEndData {
    harness
        .agent
        .event_store()
        .query_events_by_types(session, &[crate::events::EventType::TaskEnd], 5)
        .await
        .unwrap()
        .into_iter()
        .find_map(|event| event.parse_data::<crate::events::TaskEndData>().ok())
        .expect("task end")
}

async fn ask_owner_turn(harness: &crate::testing::TestHarness, session: &str, text: &str) -> String {
    harness
        .agent
        .handle_message(
            session,
            text,
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap()
}

fn install_items(status: &str) -> serde_json::Value {
    json!([
        {"text": "Install Drupal in ~/projects/synthetic-drupal", "status": status, "mutation_effects": ["local_workspace_write"]},
        {"text": "Install WordPress in ~/projects/synthetic-wp", "status": status, "mutation_effects": ["local_workspace_write"]},
        {"text": "Send a screenshot of each homepage", "status": status, "mutation_effects": ["external_delivery"]}
    ])
}

/// The run reports on earlier sites instead of installing: it is sent back
/// once, asks the owner through `needs_owner`, and ends partial with that
/// question as its reply. It is never a completed request.
#[tokio::test]
async fn a_report_on_earlier_work_does_not_complete_an_instruction_to_do_it() {
    let session = "unattempted-work-asks";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "track_requirements",
            &json!({"items": install_items("pending")}).to_string(),
        ),
        MockProvider::tool_call_response("synthetic_site_list", "{}"),
        MockProvider::tool_call_response(
            "track_requirements",
            &json!({"items": install_items("deferred"), "outcome": "blocked"}).to_string(),
        ),
        MockProvider::text_response(
            "Both sites are present but paused, so their timings and screenshots remain unverified.",
        ),
        MockProvider::tool_call_response(
            "track_requirements",
            &json!({"items": install_items("deferred"), "outcome": "needs_owner"}).to_string(),
        ),
        MockProvider::text_response(
            "Those sites already exist. Install new ones, or report on the existing ones?",
        ),
        MockProvider::text_response("UNREACHED"),
    ]);
    let harness = unattempted_work_harness(provider).await;
    let reply = ask_owner_turn(
        &harness,
        session,
        "Install Drupal and WordPress with ddev in ~/projects, time them, and send a screenshot of each homepage.",
    )
    .await;

    assert_eq!(
        reply,
        "Those sites already exist. Install new ones, or report on the existing ones?"
    );
    assert_eq!(harness.provider.call_count().await, 6);
    let end = unattempted_work_task_end(&harness, session).await;
    assert_eq!(end.status, crate::events::TaskStatus::Completed);
    assert_eq!(
        end.effective_outcome(),
        crate::events::TaskOutcome::Partial
    );
    let held = harness
        .agent
        .event_store()
        .query_events_by_types(session, &[crate::events::EventType::DecisionPoint], 50)
        .await
        .unwrap()
        .into_iter()
        .filter_map(|event| event.parse_data::<crate::events::DecisionPointData>().ok())
        .any(|point| point.metadata["condition"] == "declared_work_unattempted");
    assert!(held, "the stop must be held for the unattempted work");
}

/// A run that still attempts nothing after the one pass stops unmet with the
/// typed reason, not as a success.
#[tokio::test]
async fn unattempted_work_after_the_pass_stops_with_a_typed_reason() {
    let session = "unattempted-work-stops";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "track_requirements",
            &json!({"items": install_items("pending")}).to_string(),
        ),
        MockProvider::tool_call_response(
            "track_requirements",
            &json!({"items": install_items("deferred"), "outcome": "partial"}).to_string(),
        ),
        MockProvider::text_response("The sites from earlier are there."),
        MockProvider::text_response("The sites from earlier are still there."),
        MockProvider::text_response("UNREACHED"),
    ]);
    let harness = unattempted_work_harness(provider).await;
    let reply = ask_owner_turn(&harness, session, "Install Drupal and WordPress with ddev.").await;

    assert_eq!(reply, "The sites from earlier are still there.");
    assert_eq!(harness.provider.call_count().await, 4);
    let end = unattempted_work_task_end(&harness, session).await;
    assert_eq!(
        end.effective_outcome(),
        crate::events::TaskOutcome::Partial
    );
}

/// A status question declares no work: it is answered directly, with no
/// pass, and succeeds.
#[tokio::test]
async fn a_status_question_is_still_answered_directly() {
    let session = "unattempted-work-status-question";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_site_list", "{}"),
        MockProvider::text_response("Both sites exist and are paused."),
        MockProvider::text_response("UNREACHED"),
    ]);
    let harness = unattempted_work_harness(provider).await;
    let reply = ask_owner_turn(&harness, session, "Are my DDEV sites up?").await;

    assert_eq!(reply, "Both sites exist and are paused.");
    assert_eq!(harness.provider.call_count().await, 2);
    let end = unattempted_work_task_end(&harness, session).await;
    assert_eq!(
        end.effective_outcome(),
        crate::events::TaskOutcome::Succeeded
    );
}
