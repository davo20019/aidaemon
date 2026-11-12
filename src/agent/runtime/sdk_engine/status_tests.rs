//! What a run's end records: the task end's outcome, the run aggregate's
//! lifecycle phase and the reply, for runs that stop with work open, runs
//! closed by a policy refusal, and plain successes.

use std::sync::Arc;

use crate::events::{TaskEndData, TaskOutcome, TaskStatus};
use crate::testing::{MockProvider, MockTool, TestHarness};
use crate::traits::Tool;
use crate::types::{ChannelContext, UserRole};
use aidaemon_events::run_aggregate::TaskKernelPhase;

async fn task_end(harness: &TestHarness, session: &str) -> TaskEndData {
    harness
        .agent
        .event_store()
        .query_events_by_types(session, &[crate::events::EventType::TaskEnd], 5)
        .await
        .unwrap()
        .into_iter()
        .find_map(|event| event.parse_data::<TaskEndData>().ok())
        .expect("task end")
}

async fn phase(harness: &TestHarness, session: &str, task_id: &str) -> TaskKernelPhase {
    harness
        .agent
        .event_store()
        .task_run_aggregate(session, task_id)
        .await
        .unwrap()
        .lifecycle_phase()
}

async fn root(provider: MockProvider, tools: Vec<Arc<dyn Tool>>) -> TestHarness {
    crate::testing::setup_test_agent_root_with_extra_tools_and_llm_timeout(provider, tools, None)
        .await
        .unwrap()
}

async fn ask(harness: &TestHarness, session: &str, text: &str) -> String {
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

fn probe() -> Arc<dyn Tool> {
    Arc::new(MockTool::new(
        "synthetic_probe",
        "Probes a synthetic service",
        "probe ok",
    ))
}

/// The model declares a change it never makes and keeps answering "done"
/// after the closeout's demands are spent: the run stops, but it is not a
/// success anywhere it is recorded.
#[tokio::test]
async fn stop_with_checklist_open_after_demands_is_not_a_success() {
    let checklist = r#"{"items":[
        {"text":"Probe the service","status":"in_progress","requires_observation":true},
        {"text":"Write the report","status":"pending","mutation_effects":["local_workspace_write"]}
    ]}"#;
    let mut responses = vec![
        MockProvider::tool_call_response("track_requirements", checklist),
        MockProvider::tool_call_response("synthetic_probe", "{}"),
    ];
    responses.extend((0..6).map(|_| MockProvider::text_response("All done.")));
    let harness = root(MockProvider::with_responses(responses), vec![probe()]).await;
    let session = "status-open-checklist";
    let reply = ask(&harness, session, "Probe the service and write the report").await;
    assert!(!reply.trim().is_empty());
    // The closeout demanded the open item before the run gave up on it.
    assert!(harness.provider.call_count().await > 3);

    let end = task_end(&harness, session).await;
    assert_eq!(end.status, TaskStatus::Completed);
    assert_ne!(end.effective_outcome(), TaskOutcome::Succeeded);
    assert_ne!(
        phase(&harness, session, &end.task_id).await,
        TaskKernelPhase::WorkSucceeded
    );
}

/// A full success stays a success.
#[tokio::test]
async fn plain_success_is_recorded_as_success() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_probe", "{}"),
        MockProvider::text_response("The probe says ok."),
    ]);
    let harness = root(provider, vec![probe()]).await;
    let session = "status-success";
    let reply = ask(&harness, session, "Probe the service").await;

    assert_eq!(reply, "The probe says ok.");
    let end = task_end(&harness, session).await;
    assert_eq!(end.status, TaskStatus::Completed);
    assert_eq!(end.effective_outcome(), TaskOutcome::Succeeded);
    assert_ne!(
        phase(&harness, session, &end.task_id).await,
        TaskKernelPhase::Partial
    );
}

/// The only requested change was refused by policy: the work did not
/// happen, so the run is not a success, and it is final (partial, never
/// failed, so nothing retries it).
#[tokio::test]
async fn run_closed_by_policy_denial_is_not_a_success() {
    let dir = std::path::PathBuf::from("/tmp/aidaemon-test").join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&dir).unwrap();
    let blocked = dir.join("blocked.txt");
    let session = "tg_status_denied";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "write_file",
            &serde_json::json!({"path": blocked, "content": "no"}).to_string(),
        ),
        MockProvider::text_response("I could not write the file: this chat is read-only."),
    ]);
    let tool: Arc<dyn Tool> = Arc::new(crate::tools::WriteFileTool::new(
        crate::execution::active_execution_backend(),
    ));
    let harness =
        crate::testing::setup_full_stack_test_agent_with_extra_tools(provider, vec![tool])
            .await
            .unwrap();
    let agent = Arc::new(harness.agent);
    let commands = crate::channels::commands::CommandContext {
        agent: agent.clone(),
        state: harness.state.clone(),
        task_registry: Arc::new(crate::tasks::TaskRegistry::new(10)),
        config_path: std::path::PathBuf::new(),
    };
    commands
        .dispatch("/readonly", "", session, None, UserRole::Owner)
        .await
        .unwrap();
    agent
        .handle_message(
            session,
            "Write a file.",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();
    assert!(!blocked.exists());

    let end = agent
        .event_store()
        .query_events_by_types(session, &[crate::events::EventType::TaskEnd], 5)
        .await
        .unwrap()
        .into_iter()
        .find_map(|event| event.parse_data::<TaskEndData>().ok())
        .expect("task end");
    assert_eq!(
        end.completion_proof
            .as_ref()
            .and_then(|proof| proof.proof_basis),
        Some(crate::events::TaskProofBasis::PolicyDenial)
    );
    assert_eq!(end.status, TaskStatus::Completed);
    assert_eq!(end.effective_outcome(), TaskOutcome::Partial);
    let aggregate = agent
        .event_store()
        .task_run_aggregate(session, &end.task_id)
        .await
        .unwrap();
    assert_ne!(aggregate.lifecycle_phase(), TaskKernelPhase::WorkSucceeded);
    let _ = std::fs::remove_dir_all(dir);
}

/// The agent's own `blocked` ends the run there: no more work is demanded,
/// its reply ships, and the task is partial everywhere it is recorded.
#[tokio::test]
async fn claimed_blocked_ends_the_run_partial() {
    let checklist = r#"{"items":[
        {"text":"Probe the service","status":"completed","requires_observation":true},
        {"text":"Write the report","status":"pending","mutation_effects":["local_workspace_write"]}
    ],"outcome":"blocked"}"#;
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_probe", "{}"),
        MockProvider::tool_call_response("track_requirements", checklist),
        MockProvider::text_response("I can't write the report: the folder is not shared with me."),
        MockProvider::text_response("UNREACHED second answer"),
    ]);
    let harness = root(provider, vec![probe()]).await;
    let session = "status-claimed-blocked";
    let reply = ask(&harness, session, "Probe the service and write the report").await;

    assert_eq!(
        reply,
        "I can't write the report: the folder is not shared with me."
    );
    assert_eq!(harness.provider.call_count().await, 3);
    let end = task_end(&harness, session).await;
    assert_eq!(end.status, TaskStatus::Completed);
    assert_eq!(end.effective_outcome(), TaskOutcome::Partial);
    assert_eq!(
        phase(&harness, session, &end.task_id).await,
        TaskKernelPhase::Partial
    );
}

/// Receipts never upgrade the agent's own `partial`: a run that would be a
/// plain success stays partial when the agent says it is.
#[tokio::test]
async fn claimed_partial_is_not_upgraded_by_receipts() {
    let checklist = r#"{"items":[
        {"text":"Probe the service","status":"completed"},
        {"text":"Probe the backup too","status":"deferred"}
    ],"outcome":"partial"}"#;
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_probe", "{}"),
        MockProvider::tool_call_response("track_requirements", checklist),
        MockProvider::text_response("The main probe is ok; I did not get to the backup."),
    ]);
    let harness = root(provider, vec![probe()]).await;
    let session = "status-claimed-partial";
    ask(&harness, session, "Probe the service").await;

    let end = task_end(&harness, session).await;
    assert_eq!(end.effective_outcome(), TaskOutcome::Partial);
    assert_eq!(
        phase(&harness, session, &end.task_id).await,
        TaskKernelPhase::Partial
    );
}

/// A claimed `done` with the work done is a success.
#[tokio::test]
async fn claimed_done_with_the_work_done_is_a_success() {
    let checklist = r#"{"items":[
        {"text":"Probe the service","status":"completed"}
    ],"outcome":"done"}"#;
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_probe", "{}"),
        MockProvider::tool_call_response("track_requirements", checklist),
        MockProvider::text_response("The probe says ok."),
    ]);
    let harness = root(provider, vec![probe()]).await;
    let session = "status-claimed-done";
    let reply = ask(&harness, session, "Probe the service").await;

    assert_eq!(reply, "The probe says ok.");
    let end = task_end(&harness, session).await;
    assert_eq!(end.effective_outcome(), TaskOutcome::Succeeded);
}

/// Evidence downgrades a claimed `done`: a declared change no receipt shows
/// keeps the run below success.
#[tokio::test]
async fn claimed_done_with_work_open_is_not_a_success() {
    let checklist = r#"{"items":[
        {"text":"Write the report","status":"completed","mutation_effects":["local_workspace_write"]}
    ],"outcome":"done"}"#;
    let mut responses = vec![
        MockProvider::tool_call_response("synthetic_probe", "{}"),
        MockProvider::tool_call_response("track_requirements", checklist),
    ];
    responses.extend((0..6).map(|_| MockProvider::text_response("All done.")));
    let harness = root(MockProvider::with_responses(responses), vec![probe()]).await;
    let session = "status-claimed-done-open";
    ask(&harness, session, "Write the report").await;

    let end = task_end(&harness, session).await;
    assert_ne!(end.effective_outcome(), TaskOutcome::Succeeded);
    assert_ne!(
        phase(&harness, session, &end.task_id).await,
        TaskKernelPhase::WorkSucceeded
    );
}
