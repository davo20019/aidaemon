use std::sync::Arc;

use crate::events::EventStore;
use crate::harness_eval::fixture::{
    assert_expectations, collect_run_result, HarnessEvalFixture, HarnessEvalRunResult,
    MockResponseSpec,
};
use crate::testing::{setup_test_agent, setup_test_agent_orchestrator, MockProvider};
use crate::traits::ProviderResponse;
use crate::types::{ChannelContext, UserRole};

pub async fn run_fixture(fixture: &HarnessEvalFixture) -> anyhow::Result<HarnessEvalRunResult> {
    let mock_responses = build_mock_responses(&fixture.mock_responses);
    let provider = MockProvider::with_responses(mock_responses);

    let harness = if fixture.orchestrator {
        setup_test_agent_orchestrator(provider).await?
    } else {
        setup_test_agent(provider).await?
    };

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
            !fixtures.is_empty(),
            "expected at least one fixture in {}",
            dir.display()
        );
        for (path, fixture) in fixtures {
            run_and_assert(&fixture)
                .await
                .unwrap_or_else(|err| panic!("fixture {} failed: {err:#}", path.display()));
        }
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
