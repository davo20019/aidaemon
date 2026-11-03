use super::model_call::*;
use super::outcome::*;
use super::tool_calls::*;
use super::*;
use crate::events::TaskEndData;

fn vision(max_image_bytes: u64) -> VisionConfig {
    VisionConfig {
        enabled: true,
        max_image_bytes,
        mime_types: vec!["image/jpeg".to_string(), "image/png".to_string()],
        model_patterns: vec!["claude-".to_string()],
    }
}

fn attachment(file: &tempfile::NamedTempFile, mime: &str) -> MessageAttachment {
    MessageAttachment {
        local_path: file.path().to_string_lossy().into_owned(),
        filename: "photo.jpg".to_string(),
        mime_type: mime.to_string(),
        size_bytes: 4,
        ..MessageAttachment::default()
    }
}

fn engine_config(provider_kind: &str, engine: &str) -> AppConfig {
    toml::from_str(&format!(
        "[provider]\nkind = \"{provider_kind}\"\napi_key = \"test-key\"\n\n\
         [provider.models]\ndefault = \"model-a\"\n\n[agent.engine]\n{engine}"
    ))
    .expect("parse app config")
}

#[test]
fn legacy_engine_model_pin_is_dropped_for_a_non_anthropic_provider() {
    let config = engine_config(
        "openai_chatgpt",
        "model = \"claude-haiku-5-5\"\napi_key = \"sk-ant-test\"",
    );
    let settings = settings_from_config(&config);
    assert_eq!(settings.model_override, None);
}

#[test]
fn legacy_engine_model_pin_is_kept_for_an_anthropic_provider() {
    let config = engine_config(
        "anthropic",
        "model = \"claude-haiku-5-5\"\napi_key = \"sk-ant-test\"",
    );
    let settings = settings_from_config(&config);
    assert_eq!(settings.model_override.as_deref(), Some("claude-haiku-5-5"));
}

#[test]
fn engine_model_pin_without_api_key_is_honored() {
    let config = engine_config("openai_chatgpt", "model = \"gpt-6-luna\"");
    let settings = settings_from_config(&config);
    assert_eq!(settings.model_override.as_deref(), Some("gpt-6-luna"));
}

#[test]
fn image_attachments_are_encoded_for_a_vision_model() {
    use std::io::Write;
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&[0xFF, 0xD8, 0xFF, 0xE0]).unwrap();
    let photo = attachment(&file, "image/jpeg");
    let voice = attachment(&file, "audio/ogg");

    let (images, skipped) = sdk_images(&[&photo, &voice], &vision(1024), "claude-sonnet-5-5");
    assert!(!skipped);
    assert_eq!(images, vec![sdk::Image::base64("image/jpeg", "/9j/4A==")]);

    // No image attachments: nothing to send and nothing skipped.
    assert_eq!(
        sdk_images(&[&voice], &vision(1024), "claude-sonnet-5-5"),
        (Vec::new(), false)
    );
}

#[test]
fn images_are_skipped_without_vision_or_over_the_cap() {
    use std::io::Write;
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&[0xFF, 0xD8, 0xFF, 0xE0]).unwrap();
    let photo = attachment(&file, "image/jpeg");

    assert_eq!(
        sdk_images(&[&photo], &vision(1024), "text-only-model"),
        (Vec::new(), true)
    );
    let mut disabled = vision(1024);
    disabled.enabled = false;
    assert_eq!(
        sdk_images(&[&photo], &disabled, "claude-sonnet-5-5"),
        (Vec::new(), true)
    );
    assert_eq!(
        sdk_images(&[&photo], &vision(2), "claude-sonnet-5-5"),
        (Vec::new(), true)
    );
}

fn owner_request(text: &str) -> crate::runtime_ports::InboundMessageRequest {
    crate::runtime_ports::InboundMessageRequest {
        session_id: "telegram:synthetic-user-1".to_string(),
        user_text: text.to_string(),
        attachments: Vec::new(),
        status_tx: None,
        user_role: UserRole::Owner,
        channel_ctx: ChannelContext::private("telegram"),
        heartbeat: None,
        ingress_timing: None,
    }
}

/// The whole daemon path on the SDK engine with the mock provider behind
/// the daemon's provider stack: a tool dispatched through the bridge, the
/// final answer delivered, and the tool result recorded.
#[tokio::test]
async fn sdk_engine_runs_on_mock_provider() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    use crate::traits::store_prelude::*;

    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"preference","key":"favorite_color","value":"teal"}"#,
        ),
        MockProvider::text_response("Saved."),
    ]);
    let harness = crate::testing::setup_test_agent_sdk(provider)
        .await
        .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Remember that my favorite color is teal."))
        .await
        .unwrap();
    assert_eq!(reply.text, "Saved.");
    assert_eq!(harness.provider.call_count().await, 2);
    let results = harness
        .agent
        .event_store
        .task_event_count(
            "telegram:synthetic-user-1",
            &reply.task_id,
            EventType::ToolResult,
        )
        .await
        .unwrap();
    assert_eq!(results, 1);
    let facts = harness.state.get_facts(None).await.unwrap();
    assert!(facts.iter().any(|fact| fact.value.contains("teal")));
}

async fn models_harness() -> crate::testing::TestHarness {
    use crate::testing::MockProvider;
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("one"),
        MockProvider::text_response("two"),
    ]);
    let mut harness =
        crate::testing::setup_test_agent_with_models(provider, "primary-model", "smart-model")
            .await
            .unwrap();
    harness.agent.set_sdk_engine(SdkEngineSettings::default());
    harness
}

async fn recorded_models(harness: &crate::testing::TestHarness) -> Vec<String> {
    harness
        .provider
        .call_log
        .lock()
        .await
        .iter()
        .map(|call| call.model.clone())
        .collect()
}

/// `/model` pins the model; clearing it hands selection back to the
/// daemon's routing, as in the legacy loop.
#[tokio::test]
async fn sdk_engine_follows_daemon_model_selection() {
    use crate::runtime_ports::AgentIngress;
    let harness = models_harness().await;
    harness.agent.set_model("pinned-model".to_string()).await;
    harness
        .agent
        .handle_inbound_message(owner_request("hello"))
        .await
        .unwrap();
    harness.agent.clear_model_override().await;
    harness
        .agent
        .handle_inbound_message(owner_request("hello again"))
        .await
        .unwrap();
    assert_eq!(
        recorded_models(&harness).await,
        ["pinned-model", "primary-model"]
    );
}

#[tokio::test]
async fn sdk_engine_records_llm_call_telemetry() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"preference","key":"tea","value":"green"}"#,
        ),
        MockProvider::text_response("Saved."),
    ]);
    let harness = crate::testing::setup_test_agent_sdk(provider)
        .await
        .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Remember I like green tea."))
        .await
        .unwrap();
    // The run's own calls; bootstrap's assessment calls carry a purpose.
    let calls = task_events(&harness, &reply.task_id)
        .await
        .iter()
        .filter(|event| event.event_type == EventType::LlmCall)
        .filter(|event| event.data.get("call_purpose").is_none_or(|p| p.is_null()))
        .count();
    assert_eq!(calls, 2);
}

#[tokio::test]
async fn sdk_engine_retries_a_timed_out_model_call() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    let provider = MockProvider::with_delayed_responses(
        vec![MockProvider::text_response("Here is the answer.")],
        vec![Duration::from_secs(3), Duration::ZERO],
    );
    let harness = crate::testing::setup_test_agent_sdk_with_llm_timeout(provider, Some(1))
        .await
        .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Summarize the plan."))
        .await
        .unwrap();
    assert!(
        reply.text.contains("Here is the answer."),
        "a timed-out call is retried, not the end of the run: {}",
        reply.text
    );
    let retried_with_directive = harness.provider.call_log.lock().await.iter().any(|call| {
        call.messages.iter().any(|message| {
            message
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|text| text.contains("took longer than 1s"))
        })
    });
    assert!(
        retried_with_directive,
        "the retry tells the model to take a smaller step"
    );
}

#[tokio::test]
async fn sdk_engine_fails_after_repeated_model_call_timeouts() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    let attempts = MAX_MODEL_CALL_TIMEOUT_RETRIES as usize + 1;
    let provider = MockProvider::with_delayed_responses(
        vec![MockProvider::text_response("Too late.")],
        vec![Duration::from_secs(3); attempts],
    );
    let harness = crate::testing::setup_test_agent_sdk_with_llm_timeout(provider, Some(1))
        .await
        .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Summarize the plan."))
        .await
        .unwrap();
    assert!(
        reply.text.contains("timed out after 1s"),
        "the retry budget is bounded: {}",
        reply.text
    );
}

async fn task_events(
    harness: &crate::testing::TestHarness,
    task_id: &str,
) -> Vec<crate::events::Event> {
    harness
        .agent
        .event_store
        .query_task_events_for_session("telegram:synthetic-user-1", task_id)
        .await
        .unwrap()
}

fn remember(value: &str) -> crate::traits::ProviderResponse {
    crate::testing::MockProvider::tool_call_response(
        "remember_fact",
        &format!(r#"{{"category":"preference","key":"color","value":"{value}"}}"#),
    )
}

#[tokio::test]
async fn sdk_engine_emits_tool_call_before_result() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    let provider =
        MockProvider::with_responses(vec![remember("teal"), MockProvider::text_response("ok")]);
    let harness = crate::testing::setup_test_agent_sdk(provider)
        .await
        .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Remember teal."))
        .await
        .unwrap();
    let events = task_events(&harness, &reply.task_id).await;
    let kinds: Vec<EventType> = events
        .iter()
        .map(|event| event.event_type)
        .filter(|kind| matches!(kind, EventType::ToolCall | EventType::ToolResult))
        .collect();
    assert_eq!(kinds, [EventType::ToolCall, EventType::ToolResult]);
    let result = events
        .iter()
        .find(|event| event.event_type == EventType::ToolResult)
        .unwrap()
        .parse_data::<crate::events::ToolResultData>()
        .unwrap();
    let receipt = result.receipt.expect("tool result without a receipt");
    assert_eq!(
        receipt.invocation_stage,
        crate::traits::ToolInvocationStage::Dispatched
    );
    let aggregate = harness
        .agent
        .event_store
        .task_run_aggregate("telegram:synthetic-user-1", &reply.task_id)
        .await
        .unwrap();
    assert_eq!(aggregate.operations.len(), 1);
    assert!(
        aggregate.invariant_violations.is_empty(),
        "{:?}",
        aggregate.invariant_violations
    );
}

/// Review focus: a provider failure after tools ran keeps their records.
#[tokio::test]
async fn sdk_engine_keeps_tool_results_when_provider_fails() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    let provider = MockProvider::with_responses(vec![remember("teal")]).with_strict_responses();
    let harness = crate::testing::setup_test_agent_sdk(provider)
        .await
        .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Remember teal."))
        .await
        .unwrap();
    let events = task_events(&harness, &reply.task_id).await;
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type == EventType::ToolResult)
            .count(),
        1
    );
    let end = events
        .iter()
        .find(|event| event.event_type == EventType::TaskEnd)
        .unwrap()
        .parse_data::<TaskEndData>()
        .unwrap();
    assert_eq!(end.status, TaskStatus::Failed);
}

#[tokio::test]
async fn sdk_engine_track_requirements_declares_expectations() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    // Typed items become obligations; free-text items are only notes.
    let checklist = serde_json::json!({"items": [
        {"text": "read the first note", "status": "pending",
         "requires_observation": true, "targets": ["/tmp/synthetic/first.txt"]},
        {"text": "read the second note", "status": "pending",
         "requires_observation": true, "targets": ["/tmp/synthetic/second.txt"]}
    ]})
    .to_string();
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("track_requirements", &checklist),
        MockProvider::text_response("planned"),
        MockProvider::text_response("planned"),
        MockProvider::text_response("planned"),
    ]);
    let mut harness = crate::testing::setup_test_agent_sdk(provider)
        .await
        .unwrap();
    let plan_pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    let plans = Arc::new(crate::plans::PlanStore::new(plan_pool).await.unwrap());
    harness.agent.tools.push(Arc::new(
        crate::tools::track_requirements::TrackRequirementsTool::new(plans),
    ));
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Pick a color and save it."))
        .await
        .unwrap();
    let declared = task_events(&harness, &reply.task_id)
        .await
        .iter()
        .filter(|event| event.event_type == EventType::ExecutorExpectationsDeclared)
        .count();
    assert_eq!(declared, 1);
    let aggregate = harness
        .agent
        .event_store
        .task_run_aggregate("telegram:synthetic-user-1", &reply.task_id)
        .await
        .unwrap();
    assert_eq!(aggregate.open_executor_expectations(), 2);
}

/// Review focus: a cancelled parent or goal stops the SDK run before
/// its next model call.
#[tokio::test]
async fn sdk_engine_stops_when_cancelled() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("late")]);
    let mut harness = crate::testing::setup_test_agent_sdk(provider)
        .await
        .unwrap();
    let token = tokio_util::sync::CancellationToken::new();
    harness.agent.cancel_token = Some(token.clone());
    token.cancel();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("hello"))
        .await
        .unwrap();
    assert_eq!(reply.text, "Cancelled.");
    assert_eq!(harness.provider.call_count().await, 0);
    let end = task_events(&harness, &reply.task_id)
        .await
        .iter()
        .find(|event| event.event_type == EventType::TaskEnd)
        .unwrap()
        .parse_data::<TaskEndData>()
        .unwrap();
    assert_eq!(end.status, TaskStatus::Cancelled);
}

/// The legacy `executor_declared_checklist` integration test on the SDK
/// engine: the model declares three typed items, does one and tries to
/// stop; the ledger demands the open items until receipts close them.
/// The closeout interrupted a real answer, then the model went silent:
/// the user gets that answer, not a generic activity summary.
#[tokio::test]
async fn empty_reply_after_a_closeout_demand_ships_the_interrupted_answer() {
    use crate::testing::MockProvider;
    let workspace = tempfile::tempdir().expect("workspace");
    let file = workspace
        .path()
        .join("never-read.txt")
        .to_string_lossy()
        .to_string();
    let checklist_args = serde_json::json!({
        "items": [
            {"text": "read the file back", "status": "pending",
             "requires_observation": true, "targets": [file]}
        ]
    })
    .to_string();
    let answer = "Cancelled the Acme Corp reminder; the daily schedule is unchanged.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("track_requirements", &checklist_args),
        MockProvider::text_response(answer),
        MockProvider::text_response(""),
        MockProvider::text_response(""),
        MockProvider::text_response(""),
    ])
    .with_task_assessments(vec![MockProvider::semantic_task_assessment(
        "check",
        false,
        false,
        &[],
        "new_request",
        "local_workspace",
    )]);
    let plan_pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    let plans = Arc::new(crate::plans::PlanStore::new(plan_pool).await.unwrap());
    let track = Arc::new(crate::tools::track_requirements::TrackRequirementsTool::new(plans));
    let harness = crate::testing::setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![track as Arc<dyn crate::traits::Tool>],
    )
    .await
    .unwrap();

    let response = harness
        .agent
        .handle_message(
            "interrupted_answer",
            "Cancel the reminder.",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(response, answer);
}

#[tokio::test]
async fn sdk_engine_demands_open_checklist_items_until_receipts_close_them() {
    use crate::testing::MockProvider;
    use crate::types::ChannelVisibility;
    let workspace = tempfile::tempdir().expect("workspace");
    let cwd = workspace.path().to_string_lossy().to_string();
    let dir = workspace
        .path()
        .join("declared")
        .to_string_lossy()
        .to_string();
    let file = workspace
        .path()
        .join("declared/note.txt")
        .to_string_lossy()
        .to_string();
    let checklist_args = serde_json::json!({
        "items": [
            {"text": "create the directory", "status": "pending",
             "mutation_effects": ["local_workspace_write"], "targets": [dir]},
            {"text": "write the note", "status": "pending",
             "mutation_effects": ["local_workspace_write"], "targets": [file]},
            {"text": "read the note back", "status": "pending",
             "requires_observation": true, "targets": [file]}
        ]
    })
    .to_string();
    let terminal = |command: String, key: &str, path: &str| {
        serde_json::json!({"command": command, "working_dir": cwd, key: [path]}).to_string()
    };
    let early = "phase=EARLY; directory created, stopping here.";
    let done = "phase=DONE; directory, note, and read-back all completed.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("track_requirements", &checklist_args),
        MockProvider::tool_call_response(
            "terminal",
            &terminal(format!("mkdir -p '{dir}'"), "write_paths", &dir),
        ),
        MockProvider::text_response(early),
        MockProvider::tool_call_response(
            "terminal",
            &terminal(
                format!("printf SYNTHETIC_NOTE > '{file}'"),
                "write_paths",
                &file,
            ),
        ),
        MockProvider::tool_call_response(
            "terminal",
            &terminal(format!("cat '{file}'"), "read_paths", &file),
        ),
        MockProvider::text_response(done),
        MockProvider::text_response(done),
    ])
    // Assessor: nothing typed at all — no mutation, no observation.
    .with_task_assessments(vec![MockProvider::semantic_task_assessment(
        "check",
        false,
        false,
        &[],
        "new_request",
        "local_workspace",
    )]);
    let plan_pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    let plans = Arc::new(crate::plans::PlanStore::new(plan_pool).await.unwrap());
    let track = Arc::new(crate::tools::track_requirements::TrackRequirementsTool::new(plans));
    let mut harness = crate::testing::setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![track as Arc<dyn crate::traits::Tool>],
    )
    .await
    .unwrap();
    harness.agent.set_sdk_engine(SdkEngineSettings::default());

    let session_id = "executor_declared_checklist";
    let response = harness
        .agent
        .handle_message(
            session_id,
            &format!("Create {dir}, write a note file inside it, then read it back to confirm."),
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "telegram".to_string(),
                channel_name: None,
                channel_id: None,
                workspace_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:synthetic-owner".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                workspace_grant: None,
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        std::fs::read_to_string(&file).ok().as_deref(),
        Some("SYNTHETIC_NOTE"),
        "the declared write must have happened"
    );
    assert!(
        response.contains(done) && !response.contains(early),
        "the early stop must not ship: {response}"
    );
    let decision_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id",
    )
    .bind(session_id)
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();
    let decisions: Vec<serde_json::Value> = decision_rows
        .iter()
        .filter_map(|raw| serde_json::from_str(raw).ok())
        .collect();
    assert!(decisions
        .iter()
        .any(|d| d["metadata"]["condition"] == "ledger_expectations_required"));
    let last = decisions
        .iter()
        .rfind(|d| d["metadata"]["condition"] == "ledger_closeout")
        .expect("ledger closeout recorded");
    assert_eq!(last["metadata"]["verdict"], "closed", "{last}");
}

#[test]
fn closeout_demands_are_bounded_by_receipts() {
    let checklist = vec!["task:t/obligation:checklist:1".to_string()];
    let contract = vec!["task:t/obligation:evidence:0".to_string()];
    let facts = |reachable: &[String], completed, demanded_at, denial| StopFacts {
        reachable_obligation_ids: reachable.to_vec(),
        completed_results: completed,
        executor_demanded_at: demanded_at,
        terminal_policy_denial: denial,
        verification_after_mutation: false,
        unrecovered_failure: false,
    };
    // Open checklist items are demanded even after other work...
    assert!(closeout_demands_work(&facts(&checklist, 2, None, false)));
    // ...but only again once completed work changed since the last demand.
    assert!(!closeout_demands_work(&facts(
        &checklist,
        2,
        Some(2),
        false
    )));
    assert!(closeout_demands_work(&facts(&checklist, 3, Some(2), false)));
    // Contract expectations only while nothing has completed.
    assert!(closeout_demands_work(&facts(&contract, 0, None, false)));
    assert!(!closeout_demands_work(&facts(&contract, 1, None, false)));
    // Never after a terminal policy denial, never without reachable work.
    assert!(!closeout_demands_work(&facts(&checklist, 2, None, true)));
    assert!(!closeout_demands_work(&facts(&[], 0, None, false)));
    // Owner decision: a reachable observation after a successful change
    // is demanded once even after work completed.
    let verify = StopFacts {
        verification_after_mutation: true,
        ..facts(&contract, 1, None, false)
    };
    assert!(closeout_demands_work(&verify));
    // A failed operation with nothing succeeded gets its recovery pass.
    let failed = StopFacts {
        unrecovered_failure: true,
        ..facts(&[], 0, None, false)
    };
    assert!(closeout_demands_work(&failed));
    assert!(!closeout_demands_work(&StopFacts {
        terminal_policy_denial: true,
        ..failed
    }));
}

/// An assessed contract restricting tools to `system_info`.
fn restricted_assessment() -> crate::traits::ProviderResponse {
    let base = crate::testing::MockProvider::semantic_task_assessment(
        "answer",
        false,
        false,
        &[],
        "new_request",
        "none",
    );
    let mut value: serde_json::Value =
        serde_json::from_str(base.content.as_deref().unwrap()).unwrap();
    value["schema_version"] = 10.into();
    value["contract"]["tool_scope"] = "restricted".into();
    value["contract"]["allowed_tool_names"] = serde_json::json!(["system_info"]);
    value["contract"]["forbidden_tool_scopes"] = serde_json::json!([]);
    crate::testing::MockProvider::text_response(&value.to_string())
}

#[tokio::test]
async fn sdk_engine_records_the_assessed_task_contract() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("hi")])
        .with_task_assessments(vec![MockProvider::semantic_task_assessment(
            "answer",
            false,
            false,
            &[],
            "new_request",
            "none",
        )]);
    let harness = crate::testing::setup_test_agent_sdk(provider)
        .await
        .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Say hi."))
        .await
        .unwrap();
    let compiled = task_events(&harness, &reply.task_id)
        .await
        .iter()
        .filter(|event| event.event_type == EventType::TaskContractCompiled)
        .count();
    assert_eq!(compiled, 1);
    assert_eq!(reply.text, "hi");
}

#[tokio::test]
async fn sdk_engine_refuses_tools_outside_the_contract_authority() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    use crate::traits::store_prelude::*;
    let provider = MockProvider::with_responses(vec![
        remember("teal"),
        MockProvider::text_response("I could not save it."),
    ])
    .with_task_assessments(vec![restricted_assessment()]);
    let harness = crate::testing::setup_test_agent_sdk(provider)
        .await
        .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Only look things up; remember teal."))
        .await
        .unwrap();
    let facts = harness.state.get_facts(None).await.unwrap();
    assert!(
        !facts.iter().any(|fact| fact.value.contains("teal")),
        "a tool outside the contract authority ran"
    );
    let dispatched = task_events(&harness, &reply.task_id)
        .await
        .iter()
        .filter(|event| event.event_type == EventType::ToolResult)
        .filter_map(|event| event.parse_data::<crate::events::ToolResultData>().ok())
        .filter(|result| {
            result.receipt.as_ref().is_some_and(|receipt| {
                receipt.invocation_stage == crate::traits::ToolInvocationStage::Dispatched
            })
        })
        .count();
    assert_eq!(dispatched, 0);
    assert_eq!(reply.text, "I could not save it.");
}

/// A shell-like tool: not read-only as a whole, but each call says
/// whether it writes (`{"write": true}`), as `terminal` does.
struct ShellProbe(Arc<std::sync::atomic::AtomicUsize>);

#[async_trait::async_trait]
impl crate::traits::Tool for ShellProbe {
    fn name(&self) -> &str {
        "shell_probe"
    }
    fn description(&self) -> &str {
        "Runs a synthetic command."
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "shell_probe",
            "description": "Runs a synthetic command.",
            "parameters": {"type": "object", "properties": {"write": {"type": "boolean"}}}
        })
    }
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }
    fn call_semantics(&self, arguments: &str) -> crate::traits::ToolCallSemantics {
        let args = serde_json::from_str::<serde_json::Value>(arguments).unwrap_or_default();
        let writes = args["write"].as_bool().unwrap_or(true);
        if args["admin"].as_bool() == Some(true) {
            crate::traits::ToolCallSemantics::administrative()
        } else if writes {
            crate::traits::ToolCallSemantics::mutation()
        } else {
            crate::traits::ToolCallSemantics::observation()
        }
    }
    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok("acme.md\nmia.txt".to_string())
    }
}

/// A read-only request must not lose a mixed tool such as `terminal`:
/// the authority lane judges the exact call, as the legacy loop did.
#[tokio::test]
async fn read_only_contract_admits_observing_calls_of_a_mixed_tool() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    let base =
        MockProvider::semantic_task_assessment("answer", false, true, &[], "new_request", "none");
    let mut assessment: serde_json::Value =
        serde_json::from_str(base.content.as_deref().unwrap()).unwrap();
    assessment["schema_version"] = 10.into();
    assessment["contract"]["mutation_scope"] = "read_only".into();
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("shell_probe", r#"{"write":false}"#),
        MockProvider::tool_call_response("shell_probe", r#"{"write":true}"#),
        MockProvider::text_response("acme.md is the markdown file."),
    ])
    .with_task_assessments(vec![MockProvider::text_response(&assessment.to_string())]);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(ShellProbe(calls.clone()))],
        None,
    )
    .await
    .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Which file in the folder is markdown?"))
        .await
        .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the observing call runs; the writing call is refused"
    );
    assert_eq!(reply.text, "acme.md is the markdown file.");
}

/// Run bookkeeping (`track_requirements`, `report_blocker`) is typed
/// administrative: it changes no user state, so a read-only request
/// still admits it while a writing call stays refused.
#[tokio::test]
async fn read_only_contract_admits_administrative_calls() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    let base =
        MockProvider::semantic_task_assessment("answer", false, true, &[], "new_request", "none");
    let mut assessment: serde_json::Value =
        serde_json::from_str(base.content.as_deref().unwrap()).unwrap();
    assessment["schema_version"] = 10.into();
    assessment["contract"]["mutation_scope"] = "read_only".into();
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("shell_probe", r#"{"admin":true}"#),
        MockProvider::tool_call_response("shell_probe", r#"{"write":true}"#),
        MockProvider::text_response("Checklist recorded; nothing written."),
    ])
    .with_task_assessments(vec![MockProvider::text_response(&assessment.to_string())]);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(ShellProbe(calls.clone()))],
        None,
    )
    .await
    .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request(
            "Plan the review of Acme notes; change nothing.",
        ))
        .await
        .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the administrative call runs; the writing call is refused"
    );
    assert_eq!(reply.text, "Checklist recorded; nothing written.");
}

/// The model cites receipts (mandate `evidence_receipt_ids`) by the
/// `result_id` in the receipt footer, so a live tool result carries the
/// same footer replayed history does; the stored result stays raw so a
/// replay does not repeat it.
#[tokio::test]
async fn live_tool_results_carry_the_receipt_footer() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("shell_probe", r#"{"write":false}"#),
        MockProvider::text_response("acme.md is the markdown file."),
    ]);
    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(ShellProbe(Arc::new(
            std::sync::atomic::AtomicUsize::new(0),
        )))],
        None,
    )
    .await
    .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Which file in the folder is markdown?"))
        .await
        .unwrap();
    assert_eq!(reply.text, "acme.md is the markdown file.");
    let calls = harness.provider.call_log.lock().await;
    let tool_message = calls
        .last()
        .and_then(|call| {
            call.messages
                .iter()
                .rev()
                .find(|message| message["role"] == "tool")
                .cloned()
        })
        .expect("the answer call sees the tool result");
    let content = tool_message["content"].as_str().unwrap_or_default();
    assert!(content.starts_with("[Tool receipt v"), "{content}");
    assert!(content.contains("result_id=sha256:"), "{content}");
    assert!(content.contains("acme.md\nmia.txt"), "{content}");
    drop(calls);
    let stored = task_events(&harness, &reply.task_id)
        .await
        .iter()
        .filter(|event| event.event_type == EventType::ToolResult)
        .filter_map(|event| event.parse_data::<crate::events::ToolResultData>().ok())
        .map(|result| result.result)
        .collect::<Vec<_>>();
    assert_eq!(stored.len(), 1);
    assert!(stored[0].contains("acme.md\nmia.txt"), "{stored:?}");
    assert!(!stored[0].contains("[Tool receipt v"), "{stored:?}");
}

/// A goal run's scope covers its workspace, yet the spill notice sends the
/// model to the session's spill directory: the run may read its own
/// spilled results, not another session's, and gains no write there.
#[test]
fn scoped_runs_read_their_own_spilled_results_only() {
    use crate::agent::tool_execution_phase::{access_manifest_scope_check, ScopeCheck};
    use crate::traits::{ToolCallAccessManifest, ToolTargetHint, ToolTargetHintKind};
    let hint = |kind, value: String| ToolTargetHint::new(kind, value).unwrap();
    let task = ToolCallAccessManifest {
        read_targets: vec![hint(
            ToolTargetHintKind::ProjectScope,
            "/work/acme-blog".to_string(),
        )],
        write_targets: vec![hint(
            ToolTargetHintKind::ProjectScope,
            "/work/acme-blog".to_string(),
        )],
        ..Default::default()
    };
    let own = crate::tools::result_spill::session_spill_dir_for_backend(
        "specialist:research:synthetic-1",
    )
    .unwrap();
    let other = crate::tools::result_spill::session_spill_dir_for_backend(
        "specialist:research:synthetic-2",
    )
    .unwrap();
    let check = |read: Option<String>, write: Option<String>| {
        let mut call = ToolCallAccessManifest {
            read_targets: read
                .map(|path| vec![hint(ToolTargetHintKind::Path, path)])
                .unwrap_or_default(),
            write_targets: write
                .map(|path| vec![hint(ToolTargetHintKind::Path, path)])
                .unwrap_or_default(),
            ..Default::default()
        };
        grant_own_spill_reads(&mut call, "specialist:research:synthetic-1");
        access_manifest_scope_check("read_file", &call, Some(&task), &[], None)
    };
    assert!(check(Some(format!("{own}/web_fetch-1a2b3c4d.txt")), None).is_none());
    assert!(matches!(
        check(Some(format!("{other}/web_fetch-1a2b3c4d.txt")), None),
        Some(ScopeCheck::Rejected(_))
    ));
    assert!(matches!(
        check(None, Some(format!("{own}/planted.txt"))),
        Some(ScopeCheck::Rejected(_))
    ));
}

#[test]
fn replayed_results_say_they_were_not_run_again() {
    let text = replay_notice("Reminder set: \"Water the plants.\"");
    assert!(text.starts_with("[Not run again:"), "{text}");
    assert!(
        text.ends_with("Reminder set: \"Water the plants.\""),
        "{text}"
    );
}

#[test]
fn same_tool_streaks_follow_the_legacy_limits() {
    let calls = |tool: &str, n: usize, distinct: bool| -> Vec<(String, serde_json::Value)> {
        (0..n)
            .map(|i| {
                let args = if distinct {
                    serde_json::json!({ "i": i })
                } else {
                    serde_json::json!({})
                };
                (tool.to_string(), args)
            })
            .collect()
    };
    let same = serde_json::json!({});
    // Identical calls: the 8th is a loop.
    assert!(!same_tool_loop(
        &calls("remember_fact", 6, false),
        "remember_fact",
        &same
    ));
    assert!(same_tool_loop(
        &calls("remember_fact", 7, false),
        "remember_fact",
        &same
    ));
    // Diverse arguments earn the +8 bonus, not a bypass.
    let next = serde_json::json!({ "i": 99 });
    assert!(!same_tool_loop(
        &calls("remember_fact", 7, true),
        "remember_fact",
        &next
    ));
    assert!(same_tool_loop(
        &calls("remember_fact", 15, true),
        "remember_fact",
        &next
    ));
    // Versatile and research tools start at 16.
    assert!(!same_tool_loop(
        &calls("read_file", 14, true),
        "read_file",
        &next
    ));
    assert!(same_tool_loop(
        &calls("read_file", 15, true),
        "read_file",
        &next
    ));
    // computer_use is bounded at 40.
    assert!(!same_tool_loop(
        &calls("computer_use", 38, false),
        "computer_use",
        &same
    ));
    assert!(same_tool_loop(
        &calls("computer_use", 39, false),
        "computer_use",
        &same
    ));
    // Another tool breaks the streak.
    let mut mixed = calls("remember_fact", 7, false);
    mixed.push(("system_info".to_string(), same.clone()));
    assert!(!same_tool_loop(&mixed, "remember_fact", &same));
}

#[test]
fn learning_records_each_call_like_the_legacy_loop() {
    let args = r#"{"path":"notes.txt"}"#;
    let ok = learning_tool_summary("read_file", args, false, false);
    assert!(ok.starts_with("read_file(") && !ok.contains('['), "{ok}");
    assert!(learning_tool_summary("read_file", args, false, true).ends_with(" [FAILED]"));
    assert!(learning_tool_summary("read_file", args, true, true).ends_with(" [REJECTED]"));
}

#[test]
fn advisory_checklist_steps_are_deferred_once_the_run_is_done() {
    let mut plan = crate::plans::TaskPlan::new(
        "telegram:synthetic",
        "synthetic request",
        "synthetic plan",
        vec!["Summarize the already proven result".to_string()],
        "test",
    );
    assert!(reconcile_plan_projection(&mut plan));
    assert_eq!(plan.steps[0].status, crate::plans::StepStatus::Deferred);
    assert!(plan.is_finished());
}

#[test]
fn typed_checklist_steps_keep_their_state() {
    let mut plan = crate::plans::TaskPlan::new(
        "telegram:synthetic",
        "synthetic request",
        "synthetic plan",
        vec!["Observe canonical state".to_string()],
        "test",
    );
    plan.steps[0].requires_observation = true;
    assert!(!reconcile_plan_projection(&mut plan));
    assert_eq!(plan.steps[0].status, crate::plans::StepStatus::InProgress);
}

#[test]
fn task_budget_warns_once_at_eighty_percent() {
    assert!(!task_budget_warning_due(79, 100, false));
    assert!(task_budget_warning_due(80, 100, false));
    assert!(!task_budget_warning_due(90, 100, true), "only once");
    assert!(
        !task_budget_warning_due(100, 100, false),
        "exhausted stops instead"
    );
    assert!(!task_budget_warning_due(90, 0, false), "no budget");
}

#[test]
fn learning_records_errors_and_the_recovery_after_them() {
    let mut learning = crate::agent::memory_plugin::test_support::learning("build it");
    record_call_for_learning(&mut learning, "terminal(ok)".into(), "fine", false);
    assert!(learning.errors.is_empty() && learning.recovery_actions.is_empty());
    record_call_for_learning(
        &mut learning,
        "terminal(make) [FAILED]".into(),
        "Error: make: no rule",
        true,
    );
    assert_eq!(
        learning.first_error.as_deref(),
        Some("Error: make: no rule")
    );
    assert_eq!(
        learning.errors,
        vec![("Error: make: no rule".to_string(), false)]
    );
    record_call_for_learning(&mut learning, "terminal(make build)".into(), "built", false);
    assert_eq!(
        learning.recovery_actions,
        vec!["terminal(make build)".to_string()]
    );
    assert!(
        learning.errors[0].1,
        "the later success credits the error as recovered"
    );
    assert_eq!(learning.tool_calls.len(), 3);
}

#[test]
fn reply_checks_reject_empty_marker_only_and_pasted_replies() {
    assert_eq!(reply_defect("Here is the answer.", None), None);
    assert_eq!(reply_defect("   ", None), Some(ReplyDefect::Empty));
    assert_eq!(
        reply_defect("[SYSTEM] tool envelope only", None),
        Some(ReplyDefect::InternalMarkers)
    );
    assert_eq!(
        reply_defect("[tool_use: terminal]\nargs: {\"cmd\":\"ls\"}", None),
        Some(ReplyDefect::InternalMarkers)
    );
    let page = "File: /tmp/notes.txt (lines 1-5 of 5, 60 bytes)\n1 | alpha\n2 | beta\n\
                3 | gamma\n4 | delta\n5 | epsilon";
    assert_eq!(reply_defect(page, None), Some(ReplyDefect::ToolOutputPaste));
    let output = "/srv/synthetic/report-alpha.txt\n/srv/synthetic/report-beta.txt\n\
                  /srv/synthetic/report-gamma.txt\n/srv/synthetic/report-delta.txt";
    assert_eq!(
        reply_defect(output, Some(output)),
        Some(ReplyDefect::ToolOutputPaste)
    );
}

#[test]
fn final_replies_are_sanitized_and_never_a_bare_done() {
    let clean = finalize_reply("All set.\n[SYSTEM] internal note", &[], None);
    assert!(
        clean.contains("All set.") && !clean.contains("[SYSTEM]"),
        "{clean}"
    );
    let empty = finalize_reply("", &[], Some("SAFETY"));
    assert!(
        empty.starts_with("I wasn't able to process that request") && empty.contains("SAFETY"),
        "{empty}"
    );
    let gutted = finalize_reply("[SYSTEM] only markers", &[], None);
    assert!(gutted.contains("couldn't recover"), "{gutted}");
    let with_work = finalize_reply(
        "[SYSTEM] only markers",
        &["write_file(notes.txt)".to_string()],
        None,
    );
    assert!(!with_work.trim().is_empty() && !with_work.contains("[SYSTEM]"));
    let empty_after_work = finalize_reply("", &["write_file(notes.txt)".to_string()], None);
    assert!(
        !empty_after_work.trim().is_empty()
            && !empty_after_work.starts_with("I wasn't able to process"),
        "an empty reply after tool work describes the work: {empty_after_work}"
    );
}

#[tokio::test]
async fn sdk_engine_retries_an_empty_final_reply_without_tools() {
    use crate::runtime_ports::AgentIngress;
    use crate::testing::MockProvider;
    let provider = MockProvider::with_responses(vec![
        remember("teal"),
        MockProvider::text_response(""),
        MockProvider::text_response("Saved your favorite color: teal."),
    ]);
    let harness = crate::testing::setup_test_agent_sdk(provider)
        .await
        .unwrap();
    let reply = harness
        .agent
        .handle_inbound_message(owner_request("Remember my favorite color is teal."))
        .await
        .unwrap();
    assert_eq!(reply.text, "Saved your favorite color: teal.");
    let calls = harness.provider.call_log.lock().await;
    let last = calls.last().expect("retry call");
    assert_eq!(
        last.options.tool_choice,
        crate::traits::ToolChoiceMode::None,
        "the empty-reply retry after work runs with tools disabled"
    );
}

#[test]
fn interrupted_and_failed_runs_map_to_non_success_outcomes() {
    let mut outcome = sdk::RunOutcome {
        status: sdk::RunStatus::ProviderFailed {
            error: "overloaded".into(),
        },
        final_text: None,
        state: sdk::RunState::new("aidaemon", "s", "hi"),
    };
    let (reply, status, task_outcome) = reply_for(&outcome);
    assert!(reply.contains("overloaded"));
    assert_eq!(status, TaskStatus::Failed);
    assert_eq!(task_outcome, TaskOutcome::Failed);

    outcome.status = sdk::RunStatus::Completed;
    outcome.final_text = Some("All set.".into());
    let (reply, status, task_outcome) = reply_for(&outcome);
    assert_eq!(reply, "All set.");
    assert_eq!(status, TaskStatus::Completed);
    assert_eq!(task_outcome, TaskOutcome::Succeeded);
}

/// A stopped run's reply gets the same checks as a completed one: no
/// debug text, and no secret on a public channel.
#[test]
fn stopped_run_replies_are_plain_and_redacted() {
    let verdict = TurnVerdict::default();
    let mut outcome = sdk::RunOutcome {
        status: sdk::RunStatus::LimitReached {
            limit: sdk::LimitKind::ToolCalls,
        },
        final_text: Some("Partial: the deploy key is sk-abc123456789012345678901234567890".into()),
        state: sdk::RunState::new("aidaemon", "s", "hi"),
    };
    let (reply, _, task_outcome) =
        verdict.final_reply(&outcome, &[], crate::types::ChannelVisibility::Public);
    assert!(!reply.contains("sk-abc"), "secret leaked: {reply}");
    assert_eq!(task_outcome, TaskOutcome::Partial);

    outcome.final_text = None;
    let (reply, _, _) =
        verdict.final_reply(&outcome, &[], crate::types::ChannelVisibility::Private);
    assert!(reply.contains("the number of tool calls"), "{reply}");
    assert!(!reply.contains("ToolCalls"), "debug text leaked: {reply}");

    outcome.status = sdk::RunStatus::ProviderFailed {
        error: "401 from /Users/alice/.config/key: sk-abc123456789012345678901234567890".into(),
    };
    let (reply, _, _) =
        verdict.final_reply(&outcome, &[], crate::types::ChannelVisibility::Private);
    assert!(!reply.contains("sk-abc"), "secret leaked: {reply}");
    assert!(!reply.contains("/Users/alice"), "path leaked: {reply}");
}

/// The closeout judges reachability over the tools the turn offered: a
/// memory plugin that does not own the daemon memory tools hides them.
#[tokio::test]
async fn closeout_sees_only_offered_memory_tools() {
    let seed = crate::testing::setup_test_agent_sdk(crate::testing::MockProvider::new())
        .await
        .unwrap();
    let manage_memories: Arc<dyn crate::traits::Tool> =
        Arc::new(crate::tools::ManageMemoriesTool::new(seed.state.clone()));
    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        crate::testing::MockProvider::new(),
        vec![manage_memories],
        None,
    )
    .await
    .unwrap();
    let tools = &harness.agent.tools;
    let owned = closeout_visible_tools(
        tools,
        UserRole::Owner,
        crate::agent::memory_plugin::DAEMON_MEMORY_TOOLS,
    );
    assert!(owned.contains(&"remember_fact"));
    let foreign = closeout_visible_tools(tools, UserRole::Owner, &[]);
    assert!(!foreign.contains(&"remember_fact"));
    // Its goal actions stay offered under any plugin.
    assert!(foreign.contains(&"manage_memories"));
    assert!(!foreign.contains(&"manage_people"));
    assert!(closeout_visible_tools(tools, UserRole::Public, &[]).is_empty());
}
