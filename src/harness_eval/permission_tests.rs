use crate::events::{EventStore, EventType, ToolResultData};
use crate::testing::{setup_full_stack_test_agent, MockProvider};
use crate::types::{ChannelContext, UserRole};
use serde_json::json;

#[tokio::test]
async fn harness_eval_read_only_contract_denies_write_and_preserves_file() {
    let workspace = tempfile::tempdir().unwrap();
    let target = workspace.path().join("protected.txt");
    std::fs::write(&target, "synthetic original content").unwrap();
    let base = MockProvider::semantic_task_assessment(
        "check",
        false,
        true,
        &[],
        "new_request",
        "local_workspace",
    );
    let mut assessment: serde_json::Value =
        serde_json::from_str(base.content.as_deref().unwrap()).unwrap();
    assessment["contract"]["mutation_scope"] = json!("read_only");
    assessment["contract"]["evidence_requirements"] = json!([{
        "summary": "Record the result of the attempted operation",
        "acceptable_scopes": ["local_workspace"],
        "purpose": "outcome", "minimum_authority": "direct", "temporal_scope": "historical",
        "receipt": {"tool_names": ["terminal"], "outcome_condition": "non_success_terminal",
            "requires_output": false, "min_invocations": 1, "max_invocations": 1}
    }]);
    let arguments = json!({
        "command": "printf changed > protected.txt",
        "working_dir": workspace.path(), "write_paths": [target]
    })
    .to_string();
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("terminal", &arguments),
        MockProvider::text_response("The write was denied; the file is unchanged."),
        // Supply the final checkpoint reply explicitly; no fabricated mock fallback.
        MockProvider::text_response("No mutation was performed. The protected file is unchanged."),
    ])
    .with_strict_responses()
    .with_task_assessments(vec![MockProvider::text_response(&assessment.to_string())]);
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let session = "synthetic-read-only-enforcement";
    harness
        .agent
        .handle_message(
            session,
            "Perform a read-only check. Report any denied operation without retrying.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    harness
        .provider
        .assert_response_script_not_exhausted()
        .await
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(&target).unwrap(),
        "synthetic original content"
    );
    let store = EventStore::new(harness.state.pool()).await.unwrap();
    let ends = store
        .query_recent_task_ends(session, false, 1)
        .await
        .unwrap();
    let end = ends[0].parse_data::<crate::events::TaskEndData>().unwrap();
    let eval = end.harness_eval.unwrap();
    assert!(
        eval.quality.contract.forbids_mutation,
        "typed read-only contract must be installed"
    );
    assert_eq!(eval.quality.contract.forbidden_mutation_attempts, 1);
    let events = store
        .query_task_events_for_session(session, &end.task_id)
        .await
        .unwrap();
    assert!(
        events
            .iter()
            .filter(|event| event.event_type == EventType::ToolResult)
            .filter_map(|event| event.parse_data::<ToolResultData>().ok())
            .any(|result| result.name == "terminal"
                && result
                    .receipt
                    .is_some_and(|receipt| receipt.access_denial.is_some())),
        "the attempted write must produce a typed denial receipt"
    );
}
