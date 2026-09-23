use anyhow::Context;

use crate::events::EventStore;
use crate::harness_eval::fixture::{
    assert_expectations, collect_run_result, HarnessEvalFixture, HarnessEvalRunResult,
    MockResponseSpec,
};
use crate::state::sqlite::SqliteStateStore;
use crate::testing::{
    setup_test_agent, setup_test_agent_orchestrator, setup_test_agent_with_models, MockProvider,
};
use crate::traits::{Goal, GoalStore, ProviderResponse};
use crate::types::{ChannelContext, UserRole};

pub async fn run_fixture(fixture: &HarnessEvalFixture) -> anyhow::Result<HarnessEvalRunResult> {
    run_fixture_with_trace(fixture)
        .await
        .map(|(result, _session_events)| result)
}

/// Run a fixture and also return the whole session trace in append order.
async fn run_fixture_with_trace(
    fixture: &HarnessEvalFixture,
) -> anyhow::Result<(HarnessEvalRunResult, Vec<crate::events::Event>)> {
    let mock_responses = build_mock_responses(&fixture.mock_responses);
    let provider = MockProvider::with_responses(mock_responses)
        .with_strict_responses()
        .with_task_assessments(build_mock_responses(&fixture.task_assessments));

    let harness = if fixture.orchestrator {
        setup_test_agent_orchestrator(provider).await?
    } else if fixture.routing_models {
        setup_test_agent_with_models(provider, "primary-model", "smart-model").await?
    } else {
        setup_test_agent(provider).await?
    };

    apply_seed(&harness.state, &fixture.session_id, &fixture.seed).await?;

    let user_role = parse_user_role(&fixture.user_role)?;
    let response = harness
        .agent
        .handle_message(
            &fixture.session_id,
            &fixture.user_text,
            None,
            user_role,
            ChannelContext::private("test"),
            None,
        )
        .await;
    harness
        .provider
        .assert_response_script_not_exhausted()
        .await?;
    let response = response?;
    harness.provider.assert_response_script_consumed().await?;

    let event_store = EventStore::new(harness.state.pool()).await?;
    let result = collect_fixture_result(&event_store, &fixture.session_id, &response).await?;
    let mut session_events = event_store
        .query_recent_events(&fixture.session_id, usize::MAX >> 1)
        .await?;
    session_events.sort_by_key(|event| event.id);
    Ok((result, session_events))
}

async fn collect_fixture_result(
    event_store: &EventStore,
    session_id: &str,
    response: &str,
) -> anyhow::Result<HarnessEvalRunResult> {
    // Locate the terminal task, then fetch its entire scoped trace. A recency
    // window silently drops early calls from long runs and invalidates counts.
    let ends = event_store
        .query_recent_task_ends(session_id, false, 1)
        .await?;
    let end = ends.first().context("missing TaskEnd event")?;
    let task_id = end
        .task_id
        .as_deref()
        .context("TaskEnd missing task identity")?;
    let events = event_store
        .query_task_events_for_session(session_id, task_id)
        .await?;
    collect_run_result(&events, response)
}

async fn apply_seed(
    state: &SqliteStateStore,
    session_id: &str,
    seed: &crate::harness_eval::fixture::FixtureSeed,
) -> anyhow::Result<()> {
    for goal_spec in &seed.goals {
        let mut goal = Goal::new_finite(&goal_spec.description, session_id);
        goal.status = goal_spec.status.clone();
        state.create_goal(&goal).await?;
    }
    Ok(())
}

pub async fn run_and_assert(fixture: &HarnessEvalFixture) -> anyhow::Result<HarnessEvalRunResult> {
    let result = run_fixture(fixture).await?;
    assert_expectations(fixture, &result)?;
    Ok(result)
}

fn build_mock_responses(specs: &[MockResponseSpec]) -> Vec<ProviderResponse> {
    specs
        .iter()
        .map(|spec| match spec {
            MockResponseSpec::Text { text } => MockProvider::text_response(text),
            MockResponseSpec::ToolCall { tool_call } => {
                MockProvider::tool_call_response(&tool_call.name, &tool_call.arguments)
            }
        })
        .collect()
}

/// Unknown roles are rejected: silently running a typo'd role as Owner would
/// evaluate the most privileged path under a fixture meant to test a lesser one.
fn parse_user_role(raw: &str) -> anyhow::Result<UserRole> {
    match raw.to_ascii_lowercase().as_str() {
        "owner" => Ok(UserRole::Owner),
        "guest" => Ok(UserRole::Guest),
        "public" => Ok(UserRole::Public),
        other => {
            anyhow::bail!("unknown fixture user_role {other:?} (expected owner, guest, or public)")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness_eval::fixture::fixtures_dir;
    use crate::harness_eval::fixture::load_fixtures_dir;

    #[tokio::test]
    async fn harness_eval_fixture_suite() {
        let dir = fixtures_dir();
        let fixtures = load_fixtures_dir(&dir).expect("load fixtures");
        assert!(
            fixtures.len() >= 15,
            "expected at least 15 fixtures in {}, got {}",
            dir.display(),
            fixtures.len()
        );
        let mut failures = Vec::new();
        for (path, fixture) in fixtures {
            if let Err(error) = run_and_assert(&fixture).await {
                failures.push(format!("fixture {} failed: {error:#}", path.display()));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[tokio::test]
    async fn harness_eval_checks_false_fallthrough_expectation() {
        let fixture = crate::harness_eval::fixture::parse_fixture_yaml(
            "name: false_fallthrough\nsession_id: synthetic-fallthrough\nuser_text: Hello\nmock_responses:\n  - text: Hello\nexpect:\n  response_fallthrough: false\n",
        ).unwrap();
        let mut result = run_fixture(&fixture).await.unwrap();
        result.harness_eval.routing.response_fallthrough = false;
        assert_expectations(&fixture, &result).unwrap();
        result.harness_eval.routing.response_fallthrough = true;
        assert!(assert_expectations(&fixture, &result).is_err());
    }

    #[tokio::test]
    async fn harness_eval_rejects_exhausted_response_script() {
        let fixture = crate::harness_eval::fixture::parse_fixture_yaml(
            "name: empty_script\nsession_id: synthetic-empty-script\nuser_text: Hello\nexpect: {}\n",
        ).unwrap();
        let error = run_fixture(&fixture)
            .await
            .expect_err("unexpected model calls must fail the fixture");
        assert!(format!("{error:#}").contains("script exhausted"));
    }

    /// The recorder scopes to the right task and user message for every suite
    /// fixture, and rebuilds the exact script when no gate intervened. Runs
    /// where gates bounce or synthesize replies are known to diverge (see
    /// `record_fixture_from_events`), so they are not asserted here.
    #[tokio::test]
    async fn recorder_round_trips_user_text_and_ungated_scripts() {
        for (path, fixture) in load_fixtures_dir(&fixtures_dir()).unwrap() {
            let (_, events) = run_fixture_with_trace(&fixture).await.unwrap();
            let recorded = crate::harness_eval::fixture::record_fixture_from_events(
                &fixture.session_id,
                &events,
                None,
            )
            .unwrap();
            assert_eq!(recorded.user_text, fixture.user_text, "{}", path.display());
        }
        for name in ["mock_tool_then_reply", "mutation_called_once_in_order"] {
            let path = fixtures_dir().join(format!("{name}.yaml"));
            let fixture = crate::harness_eval::fixture::load_fixture_file(&path).unwrap();
            let (run, events) = run_fixture_with_trace(&fixture).await.unwrap();
            let recorded = crate::harness_eval::fixture::record_fixture_from_events(
                &fixture.session_id,
                &events,
                Some(&run.task_end.task_id),
            )
            .unwrap();
            let script = |specs: &[MockResponseSpec]| serde_json::to_string(specs).unwrap();
            assert_eq!(
                script(&recorded.mock_responses),
                script(&fixture.mock_responses),
                "{name}"
            );
            // The draft's own expectations must hold on the run it was
            // recorded from (floored scores, receipt-backed counts).
            let mut replay = recorded.clone();
            replay.task_assessments = fixture.task_assessments.clone();
            replay.orchestrator = fixture.orchestrator;
            replay.routing_models = fixture.routing_models;
            run_and_assert(&replay).await.unwrap();
        }
    }

    #[tokio::test]
    async fn harness_eval_rejects_unconsumed_response_script() {
        let fixture = crate::harness_eval::fixture::parse_fixture_yaml(
            "name: leftover_script\nsession_id: synthetic-leftover-script\nuser_text: Hello\nmock_responses:\n  - text: Hello.\n  - text: Never requested.\nexpect: {}\n",
        ).unwrap();
        let error = run_fixture(&fixture)
            .await
            .expect_err("a run that stops before its script ends must fail");
        assert!(
            format!("{error:#}").contains("1 execution"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test]
    async fn harness_eval_rejects_unknown_user_role() {
        let fixture = crate::harness_eval::fixture::parse_fixture_yaml(
            "name: bad_role\nsession_id: synthetic-bad-role\nuser_text: Hello\nuser_role: gest\nmock_responses:\n  - text: Hello.\nexpect: {}\n",
        ).unwrap();
        let error = run_fixture(&fixture)
            .await
            .expect_err("a typo'd role must not silently run as owner");
        assert!(format!("{error:#}").contains("gest"));
    }

    /// Proposals and dispatches diverge when the loop replays or suppresses a
    /// repeated call; each expectation must read its own lane.
    #[tokio::test]
    async fn harness_eval_dispatch_counts_read_receipts_not_proposals() {
        let fixture = crate::harness_eval::fixture::parse_fixture_yaml(
            r#"
name: dispatch_counts
session_id: eval_dispatch_counts_01
user_text: Check system info
mock_responses:
  - tool_call:
      name: system_info
      arguments: "{}"
  - tool_call:
      name: system_info
      arguments: "{}"
  - text: Done.
expect:
  tool_call_counts:
    system_info: 2
"#,
        )
        .unwrap();
        let result = run_and_assert(&fixture).await.unwrap();
        assert!(
            !result.dispatched_tool_names.is_empty()
                && result.dispatched_tool_names.len() <= result.tool_names.len(),
            "dispatches must be a receipt-backed subset of proposals: proposed {:?}, dispatched {:?}",
            result.tool_names,
            result.dispatched_tool_names
        );
        let mut wrong = fixture.clone();
        wrong
            .expect
            .tool_dispatch_counts
            .insert("system_info".into(), 3);
        let error = assert_expectations(&wrong, &result).unwrap_err();
        assert!(error.to_string().contains("tool_dispatch_counts"));
    }

    #[tokio::test]
    async fn harness_eval_collects_calls_before_the_recent_event_window() {
        use crate::events::{Event, EventType};
        use serde_json::json;
        let fixture = crate::harness_eval::fixture::parse_fixture_yaml(
            "name: long_trace\nsession_id: synthetic-long-trace\nuser_text: Hello\nmock_responses:\n  - text: Hello\nexpect: {}\n",
        ).unwrap();
        let baseline = run_fixture(&fixture).await.unwrap();
        let db = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", db.path().display()))
            .await
            .unwrap();
        let store = EventStore::new(pool).await.unwrap();
        let task_id = &baseline.task_end.task_id;
        for index in 0..2 {
            store
                .append(Event::new(
                    &fixture.session_id,
                    EventType::ToolCall,
                    json!({
                        "task_id": task_id, "tool_call_id": format!("call-{index}"),
                        "name": "remember_fact", "arguments": {}
                    }),
                ))
                .await
                .unwrap();
        }
        store
            .append(Event::new(
                &fixture.session_id,
                EventType::LlmCall,
                json!({"task_id": task_id, "model": "mock-model", "latency_ms": 0}),
            ))
            .await
            .unwrap();
        for _ in 0..220 {
            store
                .append(Event::new(
                    &fixture.session_id,
                    EventType::UserMessage,
                    json!({
                        "task_id": task_id, "content": "synthetic progress"
                    }),
                ))
                .await
                .unwrap();
        }
        store
            .append(Event::new(
                &fixture.session_id,
                EventType::TaskEnd,
                serde_json::to_value(&baseline.task_end).unwrap(),
            ))
            .await
            .unwrap();
        let result = collect_fixture_result(&store, &fixture.session_id, "Hello")
            .await
            .unwrap();
        assert_eq!(result.tool_names, vec!["remember_fact", "remember_fact"]);
        assert_eq!(result.llm_calls, 1);
        let mut expect_once = fixture.clone();
        expect_once
            .expect
            .tool_call_counts
            .insert("remember_fact".into(), 1);
        assert!(assert_expectations(&expect_once, &result).is_err());
        expect_once.expect.tool_call_counts.clear();
        expect_once
            .expect
            .tools_not_used
            .push("remember_fact".into());
        assert!(assert_expectations(&expect_once, &result).is_err());
    }

    /// Two distinct tools, asserted in the order they were actually called.
    #[tokio::test]
    async fn harness_eval_enforces_tool_order_end_to_end() {
        let fixture = crate::harness_eval::fixture::parse_fixture_yaml(
            r#"
name: order_ok
session_id: eval_order_ok_01
user_text: Check system info then remember it
mock_responses:
  - tool_call:
      name: system_info
      arguments: "{}"
  - tool_call:
      name: remember_fact
      arguments: '{"category":"project","key":"host","value":"synthetic"}'
  - text: Done.
expect:
  tools_in_order: [system_info, remember_fact]
  tool_call_counts:
    system_info: 1
    remember_fact: 1
"#,
        )
        .unwrap();
        run_and_assert(&fixture).await.unwrap();
    }

    /// Same run, reversed expectation — must be rejected.
    #[tokio::test]
    async fn harness_eval_rejects_wrong_tool_order_end_to_end() {
        let fixture = crate::harness_eval::fixture::parse_fixture_yaml(
            r#"
name: order_bad
session_id: eval_order_bad_01
user_text: Check system info then remember it
mock_responses:
  - tool_call:
      name: system_info
      arguments: "{}"
  - tool_call:
      name: remember_fact
      arguments: '{"category":"project","key":"host","value":"synthetic"}'
  - text: Done.
expect:
  tools_in_order: [remember_fact, system_info]
"#,
        )
        .unwrap();
        let err = run_and_assert(&fixture)
            .await
            .expect_err("reversed tool order must fail");
        assert!(
            err.to_string().contains("tools_in_order"),
            "unexpected error: {err:#}"
        );
    }

    /// The duplicate-side-effect guard: same tool invoked twice, count says once.
    #[tokio::test]
    async fn harness_eval_rejects_duplicate_tool_call_end_to_end() {
        let fixture = crate::harness_eval::fixture::parse_fixture_yaml(
            r#"
name: duplicate_bad
session_id: eval_duplicate_bad_01
user_text: Check system info
mock_responses:
  - tool_call:
      name: system_info
      arguments: "{}"
  - tool_call:
      name: system_info
      arguments: "{}"
  - text: Done.
expect:
  tool_call_counts:
    system_info: 1
"#,
        )
        .unwrap();
        let err = run_and_assert(&fixture)
            .await
            .expect_err("duplicate tool call must fail");
        assert!(
            err.to_string().contains("tool_call_counts"),
            "unexpected error: {err:#}"
        );
    }

    #[tokio::test]
    async fn harness_eval_basic_conversational_fixture() {
        let fixture = crate::harness_eval::fixture::parse_fixture_yaml(
            r#"
name: basic_conversational
session_id: eval_basic_01
user_text: Hello there
mock_responses:
  - text: Hello there.
expect:
  outcome: succeeded
  llm_calls_min: 1
"#,
        )
        .unwrap();
        run_and_assert(&fixture).await.unwrap();
    }
}
