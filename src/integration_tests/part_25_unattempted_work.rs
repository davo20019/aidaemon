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

    assert_eq!(
        reply,
        "The sites from earlier are still there.\n\nI didn't start these parts: \
         Install Drupal in ~/projects/synthetic-drupal; Install WordPress in \
         ~/projects/synthetic-wp; Send a screenshot of each homepage."
    );
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

// Live 2026-11-13: Drupal and WordPress were declared as two installs, one
// helper was launched naming the Drupal item, and the launch handed the turn
// off: WordPress was never started and nothing said why. Each declared item
// is judged on its own; one nothing took on is launched or asked about
// before a handoff ends the turn.

/// A notifying launch that hands the turn to the background and names the
/// declared items its job takes on.
fn item_launcher(name: &str, line: &str, items: Vec<usize>) -> crate::testing::MockTool {
    crate::testing::MockTool::new(name, "Starts a synthetic install in the background", line)
        .with_metadata(crate::traits::ToolCallMetadata {
            outcome_status: Some(crate::traits::ToolOutcomeStatus::Backgrounded),
            background_started: true,
            background_handoff: true,
            completion_notifications_enabled: true,
            checklist_items: items,
            ..crate::traits::ToolCallMetadata::default()
        })
        .with_finish(
            crate::traits::ToolCallOutcome::from_output(String::new())
                .handed_to_background()
                .finish
                .unwrap(),
        )
}

async fn launch_harness(provider: MockProvider) -> crate::testing::TestHarness {
    let tools: Vec<Arc<dyn crate::traits::Tool>> = vec![
        Arc::new(item_launcher(
            "launch_drupal",
            "Started in the background: the Drupal install.",
            vec![0],
        )),
        Arc::new(item_launcher(
            "launch_wordpress",
            "Started in the background: the WordPress install.",
            vec![1],
        )),
    ];
    setup_test_agent_root_with_extra_tools_and_llm_timeout(provider, tools, None)
        .await
        .unwrap()
}

fn two_installs(first: &str, second: &str) -> serde_json::Value {
    json!({"items": [
        {"text": "Install Drupal in ~/projects/synthetic-drupal", "status": first, "mutation_effects": ["local_workspace_write"]},
        {"text": "Install WordPress in ~/projects/synthetic-wp", "status": second, "mutation_effects": ["local_workspace_write"]}
    ]})
}

async fn held_for_unattempted_work(harness: &crate::testing::TestHarness, session: &str) -> bool {
    harness
        .agent
        .event_store()
        .query_events_by_types(session, &[crate::events::EventType::DecisionPoint], 50)
        .await
        .unwrap()
        .into_iter()
        .filter_map(|event| event.parse_data::<crate::events::DecisionPointData>().ok())
        .any(|point| point.metadata["condition"] == "declared_work_unattempted")
}

/// One of two declared installs launched: the handoff is held once, the
/// second install is launched, and only then does the turn hand off.
#[tokio::test]
async fn a_handoff_is_held_until_every_declared_item_is_taken_on() {
    let session = "unattempted-item-launched";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "track_requirements",
            &two_installs("background", "pending").to_string(),
        ),
        MockProvider::tool_call_response("launch_drupal", "{}"),
        MockProvider::tool_call_response("launch_wordpress", "{}"),
        MockProvider::text_response("UNREACHED"),
    ]);
    let harness = launch_harness(provider).await;
    let reply = ask_owner_turn(&harness, session, "Install Drupal and WordPress in parallel.").await;

    assert_eq!(harness.provider.call_count().await, 3);
    assert!(held_for_unattempted_work(&harness, session).await);
    assert!(!reply.contains("didn't start"), "{reply}");
}

/// After the one pass the second install is still not taken on: the turn
/// hands off, and that item ends with its typed reason, "not attempted",
/// named in the reply.
#[tokio::test]
async fn an_item_still_untaken_after_the_pass_ends_not_attempted() {
    let session = "unattempted-item-left";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "track_requirements",
            &two_installs("background", "pending").to_string(),
        ),
        MockProvider::tool_call_response("launch_drupal", "{}"),
        MockProvider::text_response("The Drupal install is underway."),
        MockProvider::text_response("UNREACHED"),
    ]);
    let harness = launch_harness(provider).await;
    let reply = ask_owner_turn(&harness, session, "Install Drupal and WordPress in parallel.").await;

    assert_eq!(harness.provider.call_count().await, 3);
    assert!(held_for_unattempted_work(&harness, session).await);
    assert!(
        reply.ends_with("I didn't start this part: Install WordPress in ~/projects/synthetic-wp."),
        "{reply}"
    );
    let end = unattempted_work_task_end(&harness, session).await;
    assert_eq!(
        end.effective_outcome(),
        crate::events::TaskOutcome::Partial
    );
}

// ==================== The request is the owner's words ====================
//
// Live 2026-11-13, after `/clear`: "Have two parallel agents. One installing
// Drupal and the other one WordPress for a Claude company. Both use ddev. At
// ~/projects/. Tell me how long it takes for them to finish. They should send
// a screenshot of the homepage." The run declared only checks and screenshots
// of existing sites, met that checklist, and never asked whether to install
// new ones. The request contract is compiled from the owner's message apart
// from the executor, and a change the owner asked for that only a look covers
// was not attempted.

fn owner_units_of_the_live_request() -> crate::traits::ProviderResponse {
    MockProvider::owner_units(&[
        ("Install Drupal with ddev in ~/projects", true),
        ("Install WordPress with ddev in ~/projects", true),
        ("Homepage screenshot of each site", false),
        ("How long each took", false),
    ])
}

fn screenshot_checks(outcome: Option<&str>) -> String {
    let mut args = json!({"items": [
        {"text": "Capture your Drupal homepage screenshot", "status": "completed", "requires_observation": true, "owner_units": [2]},
        {"text": "Capture your WordPress homepage screenshot", "status": "completed", "requires_observation": true, "owner_units": [2]}
    ]});
    if let Some(outcome) = outcome {
        args["outcome"] = json!(outcome);
    }
    args.to_string()
}

/// The run plans checks of the sites that exist, meets its own checklist
/// and answers: it is held for the owner's installs, and asks the owner the
/// one question through `needs_owner`.
#[tokio::test]
async fn a_change_request_answered_only_with_checks_is_held_for_the_owners_units() {
    let session = "owner-units-held";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("track_requirements", &screenshot_checks(None)),
        MockProvider::tool_call_response("synthetic_site_list", "{}"),
        MockProvider::tool_call_response("track_requirements", &screenshot_checks(Some("done"))),
        MockProvider::text_response("Both sites are up; here are their homepages."),
        MockProvider::tool_call_response(
            "track_requirements",
            &screenshot_checks(Some("needs_owner")),
        ),
        MockProvider::text_response(
            "Those sites exist. Install new ones, or check the existing ones?",
        ),
        MockProvider::text_response("UNREACHED"),
    ])
    .with_owner_units(vec![owner_units_of_the_live_request()]);
    let harness = unattempted_work_harness(provider).await;
    let reply = ask_owner_turn(
        &harness,
        session,
        "Have two parallel agents. One installing Drupal and the other one WordPress. Both use ddev. At ~/projects/. Tell me how long it takes for them to finish. They should send a screenshot of the homepage.",
    )
    .await;

    assert_eq!(
        reply,
        "Those sites exist. Install new ones, or check the existing ones?"
    );
    assert!(held_for_unattempted_work(&harness, session).await);
    // The run read the owner's units, numbered, from its first call on.
    let calls = harness.provider.call_log.lock().await;
    let first = serde_json::to_string(&calls[0].messages).unwrap();
    assert!(first.contains("[What the owner asked for]"), "{first}");
    assert!(first.contains("Install WordPress with ddev in ~/projects"));
    // The demand named the owner's install, in the owner's words.
    let demand = serde_json::to_string(&calls[4].messages).unwrap();
    assert!(
        demand.contains("Install Drupal with ddev in ~/projects"),
        "{demand}"
    );
}

/// "Screenshot my existing Drupal site" asks for no change: the check is
/// the work, and the run answers without being held.
#[tokio::test]
async fn a_request_to_check_is_satisfied_by_a_check() {
    let session = "owner-units-check";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "track_requirements",
            &json!({"items": [
                {"text": "Capture your Drupal homepage screenshot", "status": "pending", "requires_observation": true, "owner_units": [0]}
            ]})
            .to_string(),
        ),
        MockProvider::tool_call_response("synthetic_site_list", "{}"),
        MockProvider::text_response("Your Drupal site is up; here is its homepage."),
        MockProvider::text_response("UNREACHED"),
    ])
    .with_owner_units(vec![MockProvider::owner_units(&[(
        "Screenshot of my existing Drupal site",
        false,
    )])]);
    let harness = unattempted_work_harness(provider).await;
    let reply = ask_owner_turn(&harness, session, "Screenshot my existing Drupal site.").await;

    assert_eq!(reply, "Your Drupal site is up; here is its homepage.");
    assert!(!held_for_unattempted_work(&harness, session).await);
}
