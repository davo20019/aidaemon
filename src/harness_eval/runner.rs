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
    let mock_responses = build_mock_responses(&fixture.mock_responses);
    let provider = MockProvider::with_responses(mock_responses);

    let harness = if fixture.orchestrator {
        setup_test_agent_orchestrator(provider).await?
    } else if fixture.routing_models {
        setup_test_agent_with_models(provider, "primary-model", "smart-model").await?
    } else {
        setup_test_agent(provider).await?
    };

    apply_seed(&harness.state, &fixture.session_id, &fixture.seed).await?;

    let user_role = parse_user_role(&fixture.user_role);
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
        .await?;

    let event_store = EventStore::new(harness.state.pool()).await?;
    let events = event_store
        .query_recent_events(&fixture.session_id, 200)
        .await?;
    collect_run_result(&events, &response)
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

fn parse_user_role(raw: &str) -> UserRole {
    match raw.to_ascii_lowercase().as_str() {
        "guest" => UserRole::Guest,
        _ => UserRole::Owner,
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
        for (path, fixture) in fixtures {
            run_and_assert(&fixture)
                .await
                .unwrap_or_else(|err| panic!("fixture {} failed: {err:#}", path.display()));
        }
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
expect:
  outcome: succeeded
  llm_calls_min: 1
"#,
        )
        .unwrap();
        run_and_assert(&fixture).await.unwrap();
    }
}
