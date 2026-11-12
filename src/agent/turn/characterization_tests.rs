use crate::testing::{
    setup_full_stack_test_agent_with_extra_tools, setup_test_agent,
    setup_test_agent_root_with_extra_tools_and_llm_timeout,
    setup_test_agent_with_extra_tools_and_llm_timeout, setup_test_agent_with_models, MockProvider,
    MockTool,
};
use crate::traits::{
    ChatOptions, EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, ProviderResponse,
    ResponseMode, TokenUsage, Tool, ToolArgumentContractViolation, ToolCall, ToolCallMetadata,
    ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolChoiceMode, ToolEvidenceCapability,
    ToolMutationEffects, ToolOutcomeStatus, ToolRole, ToolSemanticScope, ToolTargetHintKind,
    ToolVerificationMode,
};
use crate::types::{ChannelContext, UserRole};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

struct RejectingArgumentTool {
    io_calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for RejectingArgumentTool {
    fn name(&self) -> &str {
        "synthetic_contract_tool"
    }

    fn description(&self) -> &str {
        "Synthetic adapter with a deterministic argument contract"
    }

    fn schema(&self) -> Value {
        json!({
            "name": self.name(),
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": { "mode": { "type": "string" } },
                "required": ["mode"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        _arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            self.io_calls.fetch_add(1, Ordering::SeqCst);
            Ok("unexpected I/O".to_string())
        }
        .await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for RejectingArgumentTool {
    fn validate_arguments(&self, _arguments: &str) -> Result<(), ToolArgumentContractViolation> {
        Err(ToolArgumentContractViolation::new(
            "synthetic mode is rejected before I/O",
        ))
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::observation().with_verification_mode(ToolVerificationMode::ResultContent)
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            needs_approval: false,
            idempotent: true,
            ..ToolCapabilities::default()
        }
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Universal
    }
}

#[tokio::test]
async fn typed_expected_negative_receipt_completes_without_validation_retry() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "run_command",
            r#"{"command":"/usr/bin/false","working_dir":"/tmp"}"#,
        ),
        MockProvider::text_response(
            "phase=synthetic; exit=1; outcome=completed_with_negative_result",
        ),
    ]);
    let negative_receipt_tool = MockTool::new(
        "run_command",
        "Synthetic typed process observation",
        "$ /usr/bin/false (exit: 1)",
    )
    .with_role(ToolRole::Universal)
    .with_metadata(ToolCallMetadata {
        outcome_status: Some(ToolOutcomeStatus::CompletedWithNegativeResult),
        exit_code: Some(1),
        semantics: ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent),
        ..ToolCallMetadata::default()
    });
    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(negative_receipt_tool)],
        None,
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "typed-negative-receipt",
            "In working directory /tmp, use run_command exactly once and no other tool to execute /usr/bin/false; exit 1 is the expected completed result. Return phase, exit, and outcome.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let calls = harness.provider.call_log.lock().await;
    assert_eq!(
        reply,
        "phase=synthetic; exit=1; outcome=completed_with_negative_result"
    );
    assert_eq!(
        calls.len(),
        2,
        "the receipt closes verification directly: the call, then the answer"
    );
}

#[tokio::test]
async fn successful_memory_receipt_finalizes_without_a_validation_budget_loop() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "manage_memories",
            r#"{"action":"search","query":"synthetic family record"}"#,
        ),
        MockProvider::text_response("The synthetic family record is available."),
    ]);
    let semantics = ToolCallSemantics::observation()
        .with_evidence(vec![ToolEvidenceCapability {
            scope: ToolSemanticScope::UserMemory,
            purposes: vec![EvidencePurpose::Content],
            authority: EvidenceAuthority::Canonical,
            temporal_scope: EvidenceTemporalScope::Both,
        }])
        .with_verification_mode(ToolVerificationMode::ResultContent);
    let tool = MockTool::new(
        "manage_memories",
        "Read synthetic memory",
        "Synthetic family record found.",
    )
    .with_role(ToolRole::Universal)
    .with_metadata(ToolCallMetadata {
        outcome_status: Some(ToolOutcomeStatus::Succeeded),
        semantics,
        ..ToolCallMetadata::default()
    });
    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(tool)],
        None,
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "typed-memory-receipt",
            "Return the synthetic family record from authorized memory.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let calls = harness.provider.call_log.lock().await;
    assert_eq!(reply, "The synthetic family record is available.");
    assert_eq!(
        calls.len(),
        2,
        "a closed typed memory obligation must not trigger validation retries"
    );
}

#[tokio::test]
async fn typed_contract_rejection_requires_the_actual_tool_receipt() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "read_file",
            r#"{"path":"/tmp/synthetic-contract.txt","start_line":1,"end_line":12,"tail_lines":1}"#,
        ),
        MockProvider::text_response("phase=synthetic; outcome=completed_with_negative_result"),
        MockProvider::text_response("phase=synthetic; outcome=completed_with_negative_result"),
    ]);
    let harness = setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(crate::tools::ReadFileTool::new(
            crate::execution::active_execution_backend(),
        ))],
        None,
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "typed-contract-rejection",
            "Use read_file exactly once and no other tool with both range and tail modes; the contract rejection is the expected observation. Return phase and outcome.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        reply,
        "phase=synthetic; outcome=completed_with_negative_result"
    );
    let calls = harness.provider.call_log.lock().await;
    assert_eq!(
        calls.len(),
        3,
        "the call, the answer, and the one recovery pass an uncredited rejection earns"
    );
}

#[tokio::test]
async fn dispatcher_argument_rejection_is_canonical_non_success_receipt() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_contract_tool", r#"{"mode":"reject"}"#),
        MockProvider::text_response("phase=synthetic; outcome=contract_rejected"),
        MockProvider::text_response("phase=synthetic; outcome=contract_rejected"),
    ]);
    let io_calls = Arc::new(AtomicUsize::new(0));
    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(RejectingArgumentTool {
            io_calls: io_calls.clone(),
        })],
        None,
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "typed-dispatch-rejection",
            "Use synthetic_contract_tool exactly once; its contract rejection is the expected observation. Return phase and outcome.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let trace: Vec<(String, String)> = sqlx::query_as(
        "SELECT event_type, data FROM events
         WHERE session_id = 'typed-dispatch-rejection' ORDER BY id",
    )
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();
    assert_eq!(
        reply, "phase=synthetic; outcome=contract_rejected",
        "typed rejection trace: {trace:#?}"
    );
    assert_eq!(
        io_calls.load(Ordering::SeqCst),
        0,
        "rejection must precede I/O"
    );
    let data: String = sqlx::query_scalar(
        "SELECT data FROM events
         WHERE session_id = 'typed-dispatch-rejection' AND event_type = 'tool_result'
         ORDER BY id DESC LIMIT 1",
    )
    .fetch_one(&harness.state.pool())
    .await
    .unwrap();
    let result: crate::events::ToolResultData = serde_json::from_str(&data).unwrap();
    let receipt = result.receipt.expect("canonical rejection receipt");
    assert!(!result.success);
    assert!(receipt.contract_rejected);
    assert_eq!(receipt.outcome_status, ToolOutcomeStatus::Blocked);
}

#[tokio::test]
async fn inbound_transport_lifecycle_is_persisted_before_agent_work() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("timing-ok")]);
    let harness = setup_test_agent(provider).await.unwrap();
    let transport_received_at = chrono::Utc::now() - chrono::Duration::seconds(4);
    let queue_entered_at = transport_received_at + chrono::Duration::seconds(1);
    let agent_dispatched_at = queue_entered_at + chrono::Duration::seconds(2);
    let timing = crate::runtime_ports::InboundMessageTiming {
        platform_message_at: Some(transport_received_at - chrono::Duration::seconds(2)),
        transport_received_at,
        queue_entered_at: Some(queue_entered_at),
        agent_dispatched_at,
    };

    let reply = harness
        .agent
        .handle_message_with_attachments_and_ingress(
            "inbound-lifecycle",
            "Return the timing acknowledgement.",
            &[],
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
            Some(timing),
        )
        .await
        .unwrap();
    assert_eq!(reply, "timing-ok");

    let data: String = sqlx::query_scalar(
        "SELECT data FROM events
         WHERE session_id = 'inbound-lifecycle'
           AND event_type = 'decision_point'
           AND json_extract(data, '$.metadata.condition') = 'inbound_transport_lifecycle'
         ORDER BY id DESC LIMIT 1",
    )
    .fetch_one(&harness.state.pool())
    .await
    .unwrap();
    let event: Value = serde_json::from_str(&data).unwrap();
    let metadata = event.get("metadata").unwrap();
    assert_eq!(metadata.get("platform_to_receiver_ms"), Some(&json!(2000)));
    assert_eq!(metadata.get("receiver_to_queue_ms"), Some(&json!(1000)));
    assert_eq!(metadata.get("queue_wait_ms"), Some(&json!(2000)));
    assert_eq!(metadata.get("receiver_to_dispatch_ms"), Some(&json!(3000)));
    assert!(metadata
        .get("dispatch_to_task_start_ms")
        .and_then(Value::as_i64)
        .is_some_and(|millis| millis >= 0));
}

#[tokio::test]
async fn unavailable_relationship_assessments_fail_closed_to_fresh_context() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "Context acknowledged: phrase=VIOLET HARBOR; quantity=sixty-four.",
        ),
        MockProvider::text_response("The prior exchange is unavailable."),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();

    harness
        .agent
        .handle_message(
            "unknown-relationship-adjacency",
            "Context setup: phrase=VIOLET HARBOR; quantity=sixty-four.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    let reply = harness
        .agent
        .handle_message(
            "unknown-relationship-adjacency",
            "Return the phrase and quantity from the immediately preceding exchange.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "The prior exchange is unavailable.");
    let calls = harness.provider.call_log.lock().await;
    assert_eq!(calls.len(), 2);
    let second_messages = &calls[1].messages;
    assert!(!second_messages.iter().any(|message| {
        message.get("role").and_then(Value::as_str) == Some("user")
            && message
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains("VIOLET HARBOR"))
    }));
    assert!(!second_messages.iter().any(|message| {
        message.get("role").and_then(Value::as_str) == Some("assistant")
            && message
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains("quantity=sixty-four"))
    }));
    assert!(!second_messages
        .iter()
        .any(|message| message.get("role").and_then(Value::as_str) == Some("tool")));
}

#[tokio::test]
async fn legacy_output_markers_do_not_control_finalization() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "The requested fields have been reported.",
    )]);
    let harness = setup_test_agent(provider).await.unwrap();

    let reply = harness
        .agent
        .handle_message(
            "synthetic-output-contract",
            "Report owner and credential_status. Do not use tools.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "The requested fields have been reported.");
    assert_eq!(harness.provider.call_count().await, 1);
}

#[tokio::test]
async fn completed_negative_observation_is_reportable_when_semantic_assessment_is_unavailable() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_command_probe", "{}"),
        MockProvider::text_response(
            "The probe completed with exit code 127: the synthetic executable was not found.",
        ),
    ]);
    let probe = MockTool::new(
        "synthetic_command_probe",
        "Observe whether a synthetic command is available",
        "exit_code: 127\nsynthetic-command: command not found",
    )
    .with_metadata(ToolCallMetadata {
        outcome_status: Some(ToolOutcomeStatus::CompletedWithNegativeResult),
        exit_code: Some(127),
        semantics: ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent)
            .with_evidence(vec![ToolEvidenceCapability::new(
                ToolSemanticScope::HostLocal,
                &[EvidencePurpose::CurrentState, EvidencePurpose::Outcome],
                EvidenceAuthority::Direct,
                EvidenceTemporalScope::Current,
            )]),
        ..ToolCallMetadata::default()
    });
    let harness =
        setup_test_agent_with_extra_tools_and_llm_timeout(provider, vec![Arc::new(probe)], None)
            .await
            .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "synthetic-negative-observation",
            "Check whether the synthetic command is available and report the result.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(reply.contains("exit code 127"));
    assert!(reply.contains("not found"));
    assert_eq!(
        harness.provider.call_count().await,
        2,
        "a dispatched typed negative observation must complete without a planner contract"
    );
}

struct RecordingSearchFilesTool {
    calls: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Tool for RecordingSearchFilesTool {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        "Mock search_files tool for regression testing"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "search_files",
            "description": "Mock search",
            "parameters": {
                "type": "object",
                "properties": {
                    "glob": {"type": "string"},
                    "path": {"type": "string"}
                },
                "additionalProperties": true
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            self.calls.lock().await.push(arguments.to_string());
            let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
            let path = args["path"].as_str().unwrap_or(".");
            Ok(format!("No matches found (0 files scanned in {})", path))
        }
        .await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for RecordingSearchFilesTool {
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }
}

struct RecordingProjectInspectTool {
    calls: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Tool for RecordingProjectInspectTool {
    fn name(&self) -> &str {
        "project_inspect"
    }

    fn description(&self) -> &str {
        "Recording project_inspect tool for regression testing"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "project_inspect",
            "description": "Record project_inspect args",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "paths": {"type": "array", "items": {"type": "string"}}
                },
                "additionalProperties": true
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            self.calls.lock().await.push(arguments.to_string());
            let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
            let primary = args["path"]
                .as_str()
                .or_else(|| {
                    args["paths"]
                        .as_array()
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.as_str())
                })
                .unwrap_or(".");
            Ok(format!(
                "# Project: {}\n\n## Structure\n```\nindex.html\nstyles.css\n```\n",
                primary
            ))
        }
        .await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for RecordingProjectInspectTool {
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }
}

struct MockProjectInspectTool;

#[async_trait]
impl Tool for MockProjectInspectTool {
    fn name(&self) -> &str {
        "project_inspect"
    }

    fn description(&self) -> &str {
        "Mock project_inspect tool for regression testing"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "project_inspect",
            "description": "Mock inspect",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "paths": {"type": "array", "items": {"type": "string"}}
                },
                "additionalProperties": true
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
            let path = args["path"].as_str().unwrap_or(".");
            Ok(format!(
                "# Project: {}\n\n## Structure\n```\nindex.html\nstyles.css\n```\n",
                path
            ))
        }
        .await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for MockProjectInspectTool {
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }
}

struct PlayedNodeAudioTool {
    calls: Arc<AtomicUsize>,
}

impl PlayedNodeAudioTool {
    fn receipt_semantics() -> ToolCallSemantics {
        ToolCallSemantics::observation_and_mutation_with(ToolMutationEffects::EXTERNAL_DELIVERY)
            .with_verification_mode(ToolVerificationMode::ResultContent)
            .with_evidence(vec![ToolEvidenceCapability::new(
                ToolSemanticScope::ExternalRemote,
                &[EvidencePurpose::Outcome],
                EvidenceAuthority::Direct,
                EvidenceTemporalScope::Current,
            )])
    }
}

#[async_trait]
impl Tool for PlayedNodeAudioTool {
    fn name(&self) -> &str {
        "send_node_audio"
    }

    fn description(&self) -> &str {
        "Mock node audio delivery with an authoritative playback receipt"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "send_node_audio",
            "description": "Deliver audio to a synthetic node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node": {"type": "string"},
                    "text": {"type": "string"}
                },
                "required": ["node", "text"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        _arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolCallOutcome {
            output: json!({
                "node_id": "synthetic-node-1",
                "delivery_status": "played",
                "playback_complete": true
            })
            .to_string(),
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::Succeeded),
                semantics: Self::receipt_semantics(),
                ..ToolCallMetadata::default()
            },
            finish: None,
        })
    }
}

impl crate::traits::ToolSemantics for PlayedNodeAudioTool {
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        Self::receipt_semantics()
    }
}

/// `detached`: the model asked for a long-lived service (true) rather than a
/// job the daemon moved to the background after its foreground timeout.
struct BackgroundDetachTool {
    detached: bool,
}

#[async_trait]
impl Tool for BackgroundDetachTool {
    fn name(&self) -> &str {
        "background_task"
    }

    fn description(&self) -> &str {
        "Mock tool that detaches work to the background"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "background_task",
            "description": "Mock background detach",
            "parameters": {
                "type": "object",
                "properties": {
                    "job": {"type": "string"}
                },
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let status_tx = crate::traits::status_sender(facts);

        let _ = (arguments, status_tx);
        let outcome = ToolCallOutcome {
            output: "Background job started.".to_string(),
            metadata: ToolCallMetadata {
                background_started: true,
                detached: self.detached,
                timed_out: !self.detached,
                completion_notifications_enabled: true,
                ..ToolCallMetadata::default()
            },
            finish: None,
        };
        // As `terminal`: work moved to background tracking hands the turn
        // off; a detached service launch does not.
        Ok(if self.detached {
            outcome
        } else {
            outcome.handed_to_background()
        })
    }
}

impl crate::traits::ToolSemantics for BackgroundDetachTool {
    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::mutation_with(crate::traits::ToolMutationEffects::PROCESS_STATE)
    }
}

struct MockRemoteMutationTool;

#[async_trait]
impl Tool for MockRemoteMutationTool {
    fn name(&self) -> &str {
        "update_remote"
    }

    fn description(&self) -> &str {
        "Mock tool that updates a remote URL"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "update_remote",
            "description": "Mock remote update",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
            let url = args["url"].as_str().unwrap_or("https://example.com/status");
            Ok(format!("Updated {}", url))
        }
        .await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for MockRemoteMutationTool {
    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
        let url = args["url"].as_str().unwrap_or("https://example.com/status");
        ToolCallSemantics::mutation().with_target_hint(ToolTargetHintKind::Url, url)
    }
}

struct MockRemoteObservationTool;

#[async_trait]
impl Tool for MockRemoteObservationTool {
    fn name(&self) -> &str {
        "check_remote"
    }

    fn description(&self) -> &str {
        "Mock tool that checks a remote URL"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "check_remote",
            "description": "Mock remote check",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
            let url = args["url"].as_str().unwrap_or("https://example.com/status");
            Ok(format!("Verified {} shows the updated status.", url))
        }
        .await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for MockRemoteObservationTool {
    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
        let url = args["url"].as_str().unwrap_or("https://example.com/status");
        ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent)
            .with_evidence(vec![crate::traits::ToolEvidenceCapability::new(
                crate::traits::ToolSemanticScope::ExternalRemote,
                &[
                    crate::traits::EvidencePurpose::CurrentState,
                    crate::traits::EvidencePurpose::Content,
                    crate::traits::EvidencePurpose::Outcome,
                ],
                crate::traits::EvidenceAuthority::Direct,
                crate::traits::EvidenceTemporalScope::Current,
            )])
            .with_target_hint(ToolTargetHintKind::Url, url)
    }
}

struct CountingLocalWriteTool {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for CountingLocalWriteTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write a local file for project-instruction gate characterization"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "write_file",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            let args: Value = serde_json::from_str(arguments)?;
            let path = args["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing path"))?;
            let content = args["content"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing content"))?;
            self.calls.fetch_add(1, Ordering::SeqCst);
            std::fs::write(path, content)?;
            Ok(format!("Successfully wrote file: {path}"))
        }
        .await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for CountingLocalWriteTool {
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let path = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|args| args["path"].as_str().map(str::to_string))
            .unwrap_or_default();
        ToolCallSemantics::mutation_with(crate::traits::ToolMutationEffects::LOCAL_SOURCE_WRITE)
            .with_target_hint(ToolTargetHintKind::Path, path)
    }
}

#[tokio::test]
async fn nested_project_instructions_are_injected_before_first_write_executes() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    let nested = repo.join("crates/widget/src");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(repo.join("AGENTS.md"), "ROOT_ONLY_RULE").unwrap();
    std::fs::write(repo.join("crates/widget/AGENTS.md"), "JIT_ONLY_WIDGET_RULE").unwrap();
    let target = nested.join("lib.rs");
    std::fs::write(&target, "before\n").unwrap();

    let write_args = json!({
        "path": target.to_string_lossy(),
        "content": "after\n"
    })
    .to_string();
    let read_args = json!({"path": target.to_string_lossy()}).to_string();
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("write_file", &write_args),
        MockProvider::tool_call_response("read_file", &read_args),
        MockProvider::tool_call_response("write_file", &write_args),
        MockProvider::tool_call_response("read_file", &read_args),
        MockProvider::text_response("Updated and verified the widget."),
    ]);
    let write_calls = Arc::new(AtomicUsize::new(0));
    let harness = setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![
            Arc::new(CountingLocalWriteTool {
                calls: write_calls.clone(),
            }) as Arc<dyn Tool>,
            Arc::new(crate::tools::ReadFileTool::new(
                crate::execution::active_execution_backend(),
            )) as Arc<dyn Tool>,
        ],
        None,
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "jit_project_instruction_write_gate",
            &format!(
                "In project {}, update and verify the widget.",
                repo.display()
            ),
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(!reply.trim().is_empty());
    let calls = harness.provider.call_log.lock().await;
    let call_trace = calls
        .iter()
        .enumerate()
        .map(|(index, call)| format!("call {index}: {}", Value::Array(call.messages.clone())))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        write_calls.load(Ordering::SeqCst),
        1,
        "reply={reply:?}; provider calls:\n{call_trace}"
    );
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "after\n");

    assert!(!calls[0]
        .messages
        .iter()
        .any(|message| message.to_string().contains("JIT_ONLY_WIDGET_RULE")));
    assert!(calls.iter().skip(1).any(|call| {
        let payload = Value::Array(call.messages.clone()).to_string();
        payload.contains("JIT_ONLY_WIDGET_RULE") && payload.contains("deliberately NOT executed")
    }));
}

#[tokio::test]
async fn verification_characterization_blocks_completion_until_matching_observation() {
    let url = "https://example.com/status";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("update_remote", &json!({"url": url}).to_string()),
        MockProvider::text_response("Updated it."),
        MockProvider::tool_call_response("check_remote", &json!({"url": url}).to_string()),
        MockProvider::text_response("Updated and verified the status page."),
    ]);

    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![
            Arc::new(MockRemoteMutationTool) as Arc<dyn Tool>,
            Arc::new(MockRemoteObservationTool) as Arc<dyn Tool>,
        ],
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "verification_characterization",
            &format!("Update {} and verify it.", url),
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "Updated and verified the status page.");
    assert_eq!(
        harness.provider.call_count().await,
        4,
        "the premature final text should be blocked so the verification tool can run"
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    assert!(
        call_log.iter().any(|call| {
            call.messages.iter().any(|message| {
                message.get("role").and_then(|v| v.as_str()) == Some("system")
                    && message
                        .get("content")
                        .and_then(|v| v.as_str())
                        .is_some_and(|content| {
                            content.contains("final verification step")
                                || content.contains("verification")
                        })
            })
        }),
        "verification guard should inject a verification-required system directive"
    );
}

#[tokio::test]
async fn background_ack_characterization_ends_the_run_as_a_partial_handoff() {
    let provider = MockProvider::with_responses(vec![MockProvider::tool_call_response(
        "background_task",
        r#"{"job":"long-running"}"#,
    )]);

    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![Arc::new(BackgroundDetachTool { detached: false }) as Arc<dyn Tool>],
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "background_ack_characterization",
            "Start a long running background job",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let _ = reply;

    // The tool's typed handoff ends the run: no further model call.
    assert_eq!(harness.provider.call_count().await, 1);
    let results = harness
        .agent
        .event_store()
        .query_events_by_types(
            "background_ack_characterization",
            &[crate::events::EventType::ToolResult],
            20,
        )
        .await
        .unwrap();
    assert!(results.iter().any(|event| {
        event
            .parse_data::<crate::events::ToolResultData>()
            .is_ok_and(|result| {
                result.receipt.is_some_and(|receipt| {
                    receipt.outcome_status == crate::traits::ToolOutcomeStatus::Backgrounded
                })
            })
    }));
    let task_end = harness
        .agent
        .event_store()
        .query_events_by_types(
            "background_ack_characterization",
            &[crate::events::EventType::TaskEnd],
            5,
        )
        .await
        .unwrap()
        .into_iter()
        .find_map(|event| event.parse_data::<crate::events::TaskEndData>().ok())
        .expect("background parent task end");
    assert_eq!(task_end.status, crate::events::TaskStatus::Completed);
    assert_eq!(
        task_end.effective_outcome(),
        crate::events::TaskOutcome::Partial,
        "an in-flight detached invocation is a partial handoff, not a terminal failure"
    );
}

#[tokio::test]
async fn played_node_audio_receipt_closes_delivery_and_outcome_contract_in_one_call() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "send_node_audio",
            r#"{"node":"synthetic-node-1","text":"Dinner is ready."}"#,
        ),
        MockProvider::text_response("The audio was delivered and playback completed."),
    ]);
    let calls = Arc::new(AtomicUsize::new(0));
    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![Arc::new(PlayedNodeAudioTool {
            calls: calls.clone(),
        }) as Arc<dyn Tool>],
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "played_node_audio_receipt",
            "Have synthetic-node-1 say that dinner is ready and confirm it played.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "The audio was delivered and playback completed.");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the playback receipt should fulfill both obligations without a second verification call"
    );
    assert_eq!(
        harness.provider.call_log.lock().await.len(),
        2,
        "the fulfilled receipt should proceed directly to the final response"
    );
}

// Task 5: when a root-agent turn moves a command to the background, the turn's
// final user-facing answer must be a NEUTRAL handoff — never the model's premature
// give-up summary ("Activity summary", "results were not provided"). The handoff is
// enforced deterministically in the stopping phase, independent of model compliance
// (the real result is delivered out-of-band by the completion notifier).
#[tokio::test]
async fn background_detach_delivers_neutral_handoff_not_giveup_summary() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("background_task", r#"{"job":"long-running"}"#),
        // If the loop ever reached a forced-text pass, the model would give up on
        // empty stdout — this MUST NOT reach the user.
        MockProvider::text_response(
            "Activity summary: the command ran but the results were not provided.",
        ),
    ]);

    // Root (depth-0) harness mirrors a real user turn, where the deterministic
    // background handoff applies.
    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(BackgroundDetachTool { detached: false }) as Arc<dyn Tool>],
        None,
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "background_detach_neutral_handoff",
            "Start a long running background job",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let lower = reply.to_lowercase();
    assert!(
        !lower.contains("results were not provided") && !lower.contains("activity summary"),
        "the model's give-up summary must not be the final answer; got: {reply}"
    );
    assert!(
        lower.contains("background") && lower.contains("running in the background"),
        "the final answer must be the neutral background handoff; got: {reply}"
    );
}

// A launch the model detached on purpose is a long-lived service (a tunnel, a
// dev server): it never finishes, so a "results will follow" handoff strands
// the user. The run continues past the launch receipt, the model sees it, and
// the model's own report of what is running is the reply.
#[tokio::test]
async fn detached_service_launch_continues_the_run_to_the_models_report() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("background_task", r#"{"job":"tunnel"}"#),
        MockProvider::text_response(
            "The tunnel is running at https://synthetic-tunnel.example.test. Ask me to stop it anytime.",
        ),
    ]);

    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(BackgroundDetachTool { detached: true }) as Arc<dyn Tool>],
        None,
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "detached_service_launch",
            "Expose the local site through a tunnel",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        reply.contains("https://synthetic-tunnel.example.test"),
        "the model's report of the running service must be the reply; got: {reply}"
    );
    assert!(
        !reply.contains("Still on it"),
        "a detached service must not end in the background handoff; got: {reply}"
    );
    let call_log = harness.provider.call_log.lock().await.clone();
    let follow_up = call_log
        .iter()
        .rev()
        .find(|call| {
            call.messages
                .iter()
                .any(|m| m.to_string().contains("Background job started."))
        })
        .expect("the model is called again after the launch receipt");
    let transcript = follow_up
        .messages
        .iter()
        .map(|m| m.to_string())
        .collect::<String>();
    assert!(
        !transcript.contains("Do NOT call additional tools"),
        "the launch result must not tell the model to stop working"
    );
}

#[tokio::test]
async fn budget_blocked_same_tool_calls_do_not_trigger_false_consecutive_loop_stop() {
    let burst_calls: Vec<ToolCall> = (0..20)
        .map(|idx| ToolCall {
            id: format!("call_{}", idx),
            name: "project_inspect".to_string(),
            arguments: json!({"path": format!("/tmp/project_{}", idx)}).to_string(),
            extra_content: None,
        })
        .collect();

    let provider = MockProvider::with_responses(vec![
        ProviderResponse {
            content: None,
            tool_calls: burst_calls,
            usage: Some(TokenUsage {
                input_tokens: 10,
                output_tokens: 10,
                cached_input_tokens: None,
                cache_creation_input_tokens: None,
                model: "mock".to_string(),
                ..Default::default()
            }),
            thinking: None,
            response_note: None,
        },
        MockProvider::text_response("Summarized project status."),
    ]);

    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![Arc::new(MockProjectInspectTool) as Arc<dyn Tool>],
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "budget_vs_loop_ordering",
            "Inspect all these project folders and summarize",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "Summarized project status.");
}

#[tokio::test]
#[ignore = "project directory scope constraints not yet fully wired"]
async fn mixed_project_inspect_path_and_paths_preserves_primary_path_for_follow_up_tools() {
    let primary_dir = tempfile::tempdir().unwrap();
    let secondary_dir = tempfile::tempdir().unwrap();
    let primary_dir_str = primary_dir.path().to_string_lossy().to_string();
    let secondary_dir_str = secondary_dir.path().to_string_lossy().to_string();

    let search_calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let inspect_calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "project_inspect",
            &json!({
                "path": primary_dir_str,
                "paths": [primary_dir_str, secondary_dir_str]
            })
            .to_string(),
        ),
        MockProvider::tool_call_response("search_files", &json!({"glob":"*.html"}).to_string()),
        MockProvider::tool_call_response(
            "search_files",
            &json!({"glob":"*.html", "path": primary_dir.path().to_string_lossy()}).to_string(),
        ),
        MockProvider::text_response("Inspection complete."),
    ]);

    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![
            Arc::new(RecordingSearchFilesTool {
                calls: search_calls.clone(),
            }) as Arc<dyn Tool>,
            Arc::new(RecordingProjectInspectTool {
                calls: inspect_calls.clone(),
            }) as Arc<dyn Tool>,
        ],
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "mixed_project_inspect_path_paths",
            "Inspect both project folders and find HTML files",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "Inspection complete.");

    let inspect_args = inspect_calls.lock().await.clone();
    assert_eq!(inspect_args.len(), 1, "expected one project_inspect call");
    assert!(
        inspect_args[0].contains("\"path\"") && inspect_args[0].contains("\"paths\""),
        "expected mixed path+paths args in project_inspect call, got: {}",
        inspect_args[0]
    );

    let search_args = search_calls.lock().await.clone();
    assert_eq!(
        search_args.len(),
        2,
        "expected one follow-up search_files call plus required explicit re-check"
    );
    assert!(
        search_args[0].contains(&format!("\"path\":\"{}\"", primary_dir.path().display())),
        "expected first search_files call to inherit primary path from project_inspect(path), got: {}",
        search_args[0]
    );
}

#[tokio::test]
async fn replay_trace_yes_do_it_with_sanitized_response_analysis_falls_through_to_tools() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "arguments:\nname: terminal\ncommand: ls\n\
             [INTENT_GATE]\n\
             {\"complexity\":\"simple\",\"can_answer_now\":true,\"needs_tools\":true,\"is_acknowledgment\":true}",
        ),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Applied the requested changes."),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "replay_yes_do_it",
            "Yes, do it.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "Applied the requested changes.");
    assert!(
        harness.provider.call_count().await >= 3,
        "expected initial routing call + tool-call + final response path"
    );
}

#[tokio::test]
async fn prose_about_future_work_does_not_create_an_execution_obligation() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("I'll search for all Rust files with async fn first."),
        MockProvider::text_response("Next I'll inspect each file and count async functions."),
        MockProvider::text_response("I'm going to run the search now."),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Found the files and compiled the async summary."),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "replay_pre_tool_deferral",
            "Find all Rust files that contain async fn and give me the top 3 files.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "I'll search for all Rust files with async fn first.");
    assert_eq!(
        harness.provider.call_count().await,
        1,
        "prose alone must not synthesize a lifecycle obligation or retry"
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    assert!(
        !call_log
            .iter()
            .any(|entry| matches!(entry.options.response_mode, ResponseMode::JsonSchema { .. })),
        "text-only schema pass should be disabled"
    );
    // Only a finalized mutation/observation contract or a typed failed
    // operation can support execution recovery.
}

#[tokio::test]
async fn prose_deferral_does_not_create_execution_obligation_without_typed_contract() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "Need to inspect first.\n\
             [INTENT_GATE]\n\
             {\"complexity\":\"simple\",\"can_answer_now\":false,\"needs_tools\":true}",
        ),
        MockProvider::text_response("I'll inspect the machine first."),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("I'll format the final summary next."),
        MockProvider::text_response("Final summary: system inspection completed."),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "deferred_no_tool_reset_after_success",
            "Inspect my system and summarize it.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "I'll inspect the machine first.");
    assert_eq!(
        harness.provider.call_count().await,
        2,
        "the structural protocol marker is sanitized once, but ordinary prose must not drive retries"
    );

    // No finalized typed requirement or failed operation exists for this
    // request, so presentation text cannot force provider-level Required mode.
    let call_log = harness.provider.call_log.lock().await.clone();
    assert!(
        !call_log.is_empty(),
        "expected provider calls to be recorded"
    );
    assert!(
        call_log
            .iter()
            .all(|entry| !matches!(entry.options.tool_choice, ToolChoiceMode::Required)),
        "expected no Required tool-choice for a non-tool-classified user text"
    );
}

#[tokio::test]
async fn failed_specialist_delegation_gets_one_recovery_pass_and_no_success() {
    let incomplete_plan = "I've started breaking down your goal into specific tasks. I've created \
a plan to first research the 2026 AI job market and then synthesize that into your personalized \
morning briefing.\n\nI attempted to launch a research specialist, but the request timed out. I'm \
monitoring the system and will retry the research task as soon as the connection is stable.\n\n\
Current Plan:\n1. Research Phase: Deep dive into trends, roles, and skills.\n\
2. Synthesis Phase: Organize findings into a morning briefing.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "spawn_agent",
            r#"{"mission":"Research AI jobs","task":"Produce current findings"}"#,
        ),
        MockProvider::text_response(incomplete_plan),
        MockProvider::text_response(incomplete_plan),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response(
            "Market Snapshot: applied AI engineering remains the strongest target. \
Target Roles: GenAI engineer and AI product manager. Interview Edge: prepare concrete \
examples of evaluation, deployment, and agent reliability work.",
        ),
    ]);
    let spawn_tool: Arc<dyn Tool> = Arc::new(
        MockTool::new(
            "spawn_agent",
            "Mock failed specialist delegation",
            "Error: specialist timed out after 300 seconds",
        )
        .with_metadata(ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::FailedRetryable),
            ..ToolCallMetadata::default()
        }),
    );
    let harness =
        setup_test_agent_root_with_extra_tools_and_llm_timeout(provider, vec![spawn_tool], None)
            .await
            .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "failed_specialist_plan_pivot",
            "Research the 2026 AI job market and produce my morning briefing.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // The failed delegation is an unrecovered failure: the ledger closeout
    // demands one recovery pass, then the run closes from its receipts and
    // cannot report success, whatever the reply promises.
    let _ = reply;
    assert_eq!(
        harness.provider.call_count().await,
        3,
        "spawn, the first reply, and one recovery pass"
    );
    let task_end = harness
        .agent
        .event_store()
        .query_events_by_types(
            "failed_specialist_plan_pivot",
            &[crate::events::EventType::TaskEnd],
            5,
        )
        .await
        .unwrap()
        .into_iter()
        .find_map(|event| event.parse_data::<crate::events::TaskEndData>().ok())
        .expect("task end");
    assert_ne!(
        task_end.effective_outcome(),
        crate::events::TaskOutcome::Succeeded,
        "a run whose only work failed must not close as succeeded"
    );
}

#[tokio::test]
async fn provider_option_rejection_falls_back_to_default_chat() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("Got it.")])
        .rejecting_non_default_options();

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "provider_option_rejection_fallback",
            "Yes",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "Got it.");

    let call_log = harness.provider.call_log.lock().await.clone();
    assert!(!call_log.is_empty(), "expected at least one provider call");
    // Every call carries the conversation's cache key (81ca9193); it is a
    // routing hint, not a provider option, so it is compared apart.
    assert!(
        call_log.iter().all(|entry| ChatOptions {
            uncommitted_work: false,
            cache_key: None,
            ..entry.options.clone()
        } == ChatOptions::default()),
        "expected default chat options when the text-only pass is disabled"
    );
    assert!(
        call_log
            .iter()
            .all(|entry| entry.options.cache_key.as_deref()
                == Some("provider_option_rejection_fallback")),
        "every call should keep the conversation's cache key"
    );
}

#[tokio::test]
async fn emoji_only_turn_uses_model_instead_of_generic_ack_shortcut() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("😂")]);
    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "emoji_only_no_generic_ack",
            "😂",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "😂");
    assert!(
        harness.provider.call_count().await > 0,
        "emoji-only turns should be interpreted by the model, not short-circuited"
    );
}
