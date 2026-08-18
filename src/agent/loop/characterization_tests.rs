use crate::agent::policy_metrics_snapshot;
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
use crate::types::{ChannelContext, StatusUpdate, UserRole};
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

    fn validate_arguments(&self, _arguments: &str) -> Result<(), ToolArgumentContractViolation> {
        Err(ToolArgumentContractViolation::new(
            "synthetic mode is rejected before I/O",
        ))
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        self.io_calls.fetch_add(1, Ordering::SeqCst);
        Ok("unexpected I/O".to_string())
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
    let assessment = MockProvider::text_response(
        &json!({
            "schema_version": 7,
            "goal": "Observe one expected negative process result",
            "steps": [],
            "success_criteria": [],
            "contract": {
                "confidence": "high",
                "task_kind": "check",
                "expects_mutation": false,
                "requires_observation": true,
                "required_effects": [],
                "mutation_scope": "allowed",
                "forbidden_actions": [],
                "constraint_evidence": [],
                "tool_scope": "restricted",
                "allowed_tool_names": ["run_command"],
                "forbidden_tool_scopes": [],
                "tool_constraint_evidence": ["Use run_command exactly once and no other tool"],
                "required_response_fields": ["phase", "exit", "outcome"],
                "minimum_sources": 0,
                "requires_primary_sources": false,
                "requires_exact_history": false,
                "evidence_requirements": [{
                    "summary": "Observe the required process receipt",
                    "acceptable_scopes": ["host_local"],
                    "purpose": "outcome",
                    "minimum_authority": "direct",
                    "temporal_scope": "current",
                    "required_content_markers": [],
                    "receipt": {
                        "tool_names": ["run_command"],
                        "exit_codes": [1],
                        "outcome_statuses": ["failed_permanent"],
                        "requires_output": false,
                        "contract_rejected": false
                    }
                }],
                "project_reference": null
            },
            "task_shape": {
                "execution_mode": "inline",
                "confidence": "high",
                "independent_workstreams": 1,
                "requires_background_continuation": false,
                "continue_inline_after_background_start": false,
                "request_relationship": "new_request",
                "antecedent_user_message_id": null,
                "semantic_scope": "host_local"
            }
        })
        .to_string(),
    );
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "run_command",
            r#"{"command":"/usr/bin/false","working_dir":"/tmp"}"#,
        ),
        MockProvider::text_response(
            r#"{"goal":"Observe one expected negative process result","success_criteria":["One typed process receipt is returned"],"first_action":{"tool":"run_command","target":"/usr/bin/false","description":"Execute the requested observational command once"},"requires_verification":true,"risky_actions":["The command deliberately returns a nonzero process result"],"version":1}"#,
        ),
        MockProvider::text_response(
            "phase=synthetic; exit=1; outcome=completed_with_negative_result",
        ),
    ])
    .with_task_assessments(vec![assessment]);
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
        3,
        "the receipt must close verification directly after the pre-execution gate"
    );
}

#[tokio::test]
async fn successful_memory_receipt_finalizes_without_a_validation_budget_loop() {
    let assessment = MockProvider::text_response(
        &json!({
            "schema_version": 10,
            "goal": "Read one synthetic family record",
            "steps": [],
            "success_criteria": [],
            "contract": {
                "confidence": "high",
                "task_kind": "answer",
                "expects_mutation": false,
                "requires_observation": true,
                "required_effects": [],
                "mutation_scope": "forbidden",
                "forbidden_actions": [],
                "constraint_evidence": [],
                "tool_scope": "restricted",
                "allowed_tool_names": ["manage_memories"],
                "forbidden_tool_scopes": [],
                "tool_constraint_evidence": [],
                "required_response_fields": [],
                "minimum_sources": 0,
                "requires_primary_sources": false,
                "requires_exact_history": false,
                "evidence_requirements": [{
                    "summary": "Read the current synthetic family record",
                    "acceptable_scopes": ["user_memory"],
                    "purpose": "content",
                    "minimum_authority": "canonical",
                    "temporal_scope": "both",
                    "required_content_markers": [],
                    "receipt": {
                        "tool_names": ["manage_memories"],
                        "exit_codes": [],
                        "outcome_statuses": ["succeeded"],
                        "requires_output": true,
                        "contract_rejected": false,
                        "max_invocations": 1
                    },
                    "target": null
                }],
                "required_invocations": [],
                "filesystem_access": {
                    "execution_cwd": null,
                    "read_paths": [],
                    "write_paths": []
                },
                "project_reference": null
            },
            "task_shape": {
                "execution_mode": "inline",
                "confidence": "high",
                "independent_workstreams": 1,
                "requires_background_continuation": false,
                "continue_inline_after_background_start": false
            }
        })
        .to_string(),
    );
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "manage_memories",
            r#"{"action":"search","query":"synthetic family record"}"#,
        ),
        MockProvider::text_response(
            r#"{"goal":"Read one synthetic family record","success_criteria":["Canonical receipt recorded"],"first_action":{"tool":"manage_memories","target":null,"description":"Read once"},"requires_verification":true,"risky_actions":[],"version":1}"#,
        ),
        MockProvider::text_response("The synthetic family record is available."),
    ])
    .with_task_assessments(vec![assessment]);
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
        3,
        "a closed typed memory obligation must not trigger validation retries"
    );
}

#[tokio::test]
async fn explicitly_required_permanent_failure_is_observation_not_retry_signal() {
    let assessment = MockProvider::text_response(
        &json!({
            "schema_version": 7,
            "goal": "Observe one expected permanent adapter outcome",
            "steps": [],
            "success_criteria": [],
            "contract": {
                "confidence": "high",
                "task_kind": "check",
                "expects_mutation": false,
                "requires_observation": true,
                "required_effects": [],
                "mutation_scope": "allowed",
                "forbidden_actions": [],
                "constraint_evidence": [],
                "tool_scope": "restricted",
                "allowed_tool_names": ["synthetic_adapter"],
                "forbidden_tool_scopes": [],
                "tool_constraint_evidence": ["Use synthetic_adapter exactly once"],
                "required_response_fields": ["phase", "outcome"],
                "minimum_sources": 0,
                "requires_primary_sources": false,
                "requires_exact_history": false,
                "evidence_requirements": [],
                "required_invocations": [{
                    "tool_names": ["synthetic_adapter"],
                    "exit_codes": [],
                    "outcome_statuses": ["failed_permanent"],
                    "requires_output": true,
                    "contract_rejected": false
                }],
                "filesystem_access": {
                    "execution_cwd": null,
                    "read_paths": [],
                    "write_paths": []
                },
                "project_reference": null
            },
            "task_shape": {
                "execution_mode": "inline",
                "confidence": "high",
                "independent_workstreams": 1,
                "requires_background_continuation": false,
                "continue_inline_after_background_start": false,
                "request_relationship": "new_request",
                "antecedent_user_message_id": null,
                "semantic_scope": "host_local"
            }
        })
        .to_string(),
    );
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_adapter", r#"{"mode":"observe"}"#),
        MockProvider::text_response(
            r#"{"goal":"Observe one adapter receipt","success_criteria":["Receipt recorded"],"first_action":{"tool":"synthetic_adapter","target":null,"description":"Observe once"},"requires_verification":true,"risky_actions":[],"version":1}"#,
        ),
        MockProvider::text_response("phase=synthetic; outcome=failed_permanent"),
    ])
    .with_task_assessments(vec![assessment]);
    let tool = MockTool::new(
        "synthetic_adapter",
        "Synthetic failing observation",
        "adapter unavailable by construction",
    )
    .with_role(ToolRole::Universal)
    .with_metadata(ToolCallMetadata {
        outcome_status: Some(ToolOutcomeStatus::FailedPermanent),
        semantics: ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent)
            .with_evidence(vec![ToolEvidenceCapability {
                scope: ToolSemanticScope::HostLocal,
                purposes: vec![EvidencePurpose::Outcome],
                authority: EvidenceAuthority::Direct,
                temporal_scope: EvidenceTemporalScope::Current,
            }]),
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
            "typed-permanent-negative",
            "Use synthetic_adapter exactly once; failed_permanent is the expected observed outcome. Return phase and outcome.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "phase=synthetic; outcome=failed_permanent");
    assert_eq!(
        harness.provider.call_log.lock().await.len(),
        3,
        "the expected typed failure must not enter validation/retry recovery"
    );
}

#[tokio::test]
async fn typed_contract_rejection_requires_the_actual_tool_receipt() {
    let assessment = MockProvider::text_response(
        &json!({
            "schema_version": 7,
            "goal": "Observe one expected invocation-contract rejection",
            "steps": [],
            "success_criteria": [],
            "contract": {
                "confidence": "high",
                "task_kind": "check",
                "expects_mutation": false,
                "requires_observation": true,
                "required_effects": [],
                "mutation_scope": "allowed",
                "forbidden_actions": [],
                "constraint_evidence": [],
                "tool_scope": "restricted",
                "allowed_tool_names": ["read_file"],
                "forbidden_tool_scopes": [],
                "tool_constraint_evidence": ["Use read_file exactly once and no other tool"],
                "required_response_fields": ["phase", "outcome"],
                "minimum_sources": 0,
                "requires_primary_sources": false,
                "requires_exact_history": false,
                "evidence_requirements": [{
                    "summary": "Observe the required invocation rejection",
                    "acceptable_scopes": ["host_local"],
                    "purpose": "outcome",
                    "minimum_authority": "direct",
                    "temporal_scope": "current",
                    "required_content_markers": [],
                    "receipt": {
                        "tool_names": ["read_file"],
                        "exit_codes": [],
                        "outcome_statuses": ["completed_with_negative_result"],
                        "requires_output": false,
                        "contract_rejected": true
                    }
                }],
                "project_reference": null
            },
            "task_shape": {
                "execution_mode": "inline",
                "confidence": "high",
                "independent_workstreams": 1,
                "requires_background_continuation": false,
                "continue_inline_after_background_start": false,
                "request_relationship": "new_request",
                "antecedent_user_message_id": null,
                "semantic_scope": "host_local"
            }
        })
        .to_string(),
    );
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "read_file",
            r#"{"path":"/tmp/synthetic-contract.txt","start_line":1,"end_line":12,"tail_lines":1}"#,
        ),
        MockProvider::text_response("phase=synthetic; outcome=completed_with_negative_result"),
    ])
    .with_task_assessments(vec![assessment]);
    let harness = setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(crate::tools::ReadFileTool)],
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
    assert_eq!(calls.len(), 2, "zero-tool prose cannot satisfy the receipt");
}

#[tokio::test]
async fn dispatcher_argument_rejection_is_canonical_non_success_receipt() {
    let assessment = MockProvider::text_response(
        &json!({
            "schema_version": 7,
            "goal": "Observe a pre-I/O adapter contract rejection",
            "steps": [],
            "success_criteria": [],
            "contract": {
                "confidence": "high",
                "task_kind": "check",
                "expects_mutation": false,
                "requires_observation": true,
                "required_effects": [],
                "mutation_scope": "allowed",
                "forbidden_actions": [],
                "constraint_evidence": [],
                "tool_scope": "restricted",
                "allowed_tool_names": ["synthetic_contract_tool"],
                "forbidden_tool_scopes": [],
                "tool_constraint_evidence": ["Use synthetic_contract_tool exactly once"],
                "required_response_fields": ["phase", "outcome"],
                "minimum_sources": 0,
                "requires_primary_sources": false,
                "requires_exact_history": false,
                "evidence_requirements": [],
                "required_invocations": [{
                    "tool_names": ["synthetic_contract_tool"],
                    "exit_codes": [],
                    "outcome_statuses": ["completed_with_negative_result"],
                    "requires_output": true,
                    "contract_rejected": true
                }],
                "filesystem_access": {
                    "execution_cwd": null,
                    "read_paths": [],
                    "write_paths": []
                },
                "project_reference": null
            },
            "task_shape": {
                "execution_mode": "inline",
                "confidence": "high",
                "independent_workstreams": 1,
                "requires_background_continuation": false,
                "continue_inline_after_background_start": false,
                "request_relationship": "new_request",
                "antecedent_user_message_id": null,
                "semantic_scope": "host_local"
            }
        })
        .to_string(),
    );
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_contract_tool", r#"{"mode":"reject"}"#),
        MockProvider::text_response("phase=synthetic; outcome=contract_rejected"),
    ])
    .with_task_assessments(vec![assessment]);
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
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("timing-ok")])
        .with_task_assessments(vec![MockProvider::semantic_task_assessment(
            "answer",
            false,
            false,
            &[],
            "new_request",
            "none",
        )]);
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
    ])
    .with_task_assessments(vec![MockProvider::semantic_task_assessment(
        "answer",
        false,
        false,
        &[],
        "new_request",
        "none",
    )]);
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
async fn explicit_no_tool_current_fact_returns_direct_limitation_with_zero_offered_tools() {
    let assessment = MockProvider::text_response(
        &json!({
            "schema_version": 7,
            "goal": "Explain the evidence boundary for a current fact",
            "steps": [],
            "success_criteria": [],
            "contract": {
                "confidence": "high",
                "task_kind": "answer",
                "expects_mutation": false,
                "requires_observation": true,
                "required_effects": [],
                "mutation_scope": "allowed",
                "forbidden_actions": [],
                "constraint_evidence": [],
                "tool_scope": "forbidden",
                "tool_constraint_evidence": ["Do not use any tools"],
                "minimum_sources": 0,
                "requires_primary_sources": false,
                "requires_exact_history": false,
                "evidence_requirements": [{
                    "summary": "Establish the current synthetic language release",
                    "acceptable_scopes": ["external_remote"],
                    "purpose": "current_state",
                    "minimum_authority": "direct",
                    "temporal_scope": "current"
                }],
                "project_reference": null
            },
            "task_shape": {
                "execution_mode": "inline",
                "confidence": "high",
                "independent_workstreams": 1,
                "requires_background_continuation": false,
                "continue_inline_after_background_start": false,
                "request_relationship": "new_request",
                "semantic_scope": "external_remote"
            }
        })
        .to_string(),
    );
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "I cannot establish the current release from the supplied context; that would require live evidence, which you asked me not to retrieve.",
    )])
    .with_task_assessments(vec![assessment]);
    let harness = setup_test_agent(provider).await.unwrap();

    let reply = harness
        .agent
        .handle_message(
            "synthetic-no-tool-session",
            "What is the current synthetic language release? Do not use any tools.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(reply.contains("require live evidence"));
    let calls = harness.provider.call_log.lock().await;
    assert_eq!(calls.len(), 1, "no verification recovery call should occur");
    assert!(
        calls[0].tools.is_empty(),
        "the hard no-tool contract removes all tool definitions"
    );
}

#[tokio::test]
async fn legacy_output_markers_do_not_control_finalization() {
    let assessment = MockProvider::text_response(
        &json!({
            "schema_version": 7,
            "goal": "Return the requested readiness fields",
            "steps": [],
            "success_criteria": [],
            "contract": {
                "confidence": "high",
                "task_kind": "answer",
                "expects_mutation": false,
                "requires_observation": false,
                "required_effects": [],
                "mutation_scope": "allowed",
                "forbidden_actions": [],
                "constraint_evidence": [],
                "tool_scope": "forbidden",
                "forbidden_tool_scopes": [],
                "tool_constraint_evidence": ["Do not use tools"],
                "required_response_fields": ["owner", "credential_status"],
                "minimum_sources": 0,
                "requires_primary_sources": false,
                "requires_exact_history": false,
                "evidence_requirements": [],
                "project_reference": null
            },
            "task_shape": {
                "execution_mode": "inline",
                "confidence": "high",
                "independent_workstreams": 1,
                "requires_background_continuation": false,
                "continue_inline_after_background_start": false,
                "request_relationship": "new_request",
                "semantic_scope": "none"
            }
        })
        .to_string(),
    );
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "The requested fields have been reported.",
    )])
    .with_task_assessments(vec![assessment]);
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
async fn capability_specific_deny_set_hides_memory_without_disabling_other_tools() {
    let assessment = MockProvider::text_response(
        &json!({
            "schema_version": 7,
            "goal": "Explain the synthetic concept without memory",
            "steps": [],
            "success_criteria": [],
            "contract": {
                "confidence": "high",
                "task_kind": "answer",
                "expects_mutation": false,
                "requires_observation": false,
                "required_effects": [],
                "mutation_scope": "allowed",
                "forbidden_actions": [],
                "constraint_evidence": [],
                "tool_scope": "allowed",
                "forbidden_tool_scopes": ["user_memory"],
                "tool_constraint_evidence": ["Do not use memory"],
                "required_response_fields": [],
                "minimum_sources": 0,
                "requires_primary_sources": false,
                "requires_exact_history": false,
                "evidence_requirements": [],
                "project_reference": null
            },
            "task_shape": {
                "execution_mode": "inline",
                "confidence": "high",
                "independent_workstreams": 1,
                "requires_background_continuation": false,
                "continue_inline_after_background_start": false,
                "request_relationship": "new_request",
                "semantic_scope": "none"
            }
        })
        .to_string(),
    );
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "A synthetic explanation that does not depend on personal memory.",
    )])
    .with_task_assessments(vec![assessment]);
    let harness = setup_test_agent(provider).await.unwrap();

    let reply = harness
        .agent
        .handle_message(
            "synthetic-no-memory",
            "Explain the synthetic concept. Do not use memory.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(reply.contains("does not depend on personal memory"));
    let calls = harness.provider.call_log.lock().await;
    assert_eq!(calls.len(), 1);
    assert!(calls[0]
        .tools
        .iter()
        .all(|tool| tool["function"]["name"] != "manage_memories"));
    assert!(
        !calls[0].tools.is_empty(),
        "a scoped memory prohibition must not become an all-tool prohibition"
    );
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

#[tokio::test]
async fn failed_task_and_no_progress_metrics_are_observable() {
    let before = policy_metrics_snapshot();

    // Iteration 1: unknown tool call (blocked) => no successful tools => no-progress increment.
    // Iterations 2..: repeated valid tool call => repetitive-loop failure path.
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("no_such_tool", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let _ = harness
        .agent
        .handle_message(
            "metrics_failure_no_progress",
            "Run system checks repeatedly",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let after = policy_metrics_snapshot();
    let failed_tokens_delta = after
        .tokens_failed_tasks_total
        .saturating_sub(before.tokens_failed_tasks_total);
    let no_progress_delta = after
        .no_progress_iterations_total
        .saturating_sub(before.no_progress_iterations_total);

    assert!(
        failed_tokens_delta > 0,
        "expected tokens_failed_tasks_total to increase; before={} after={}",
        before.tokens_failed_tasks_total,
        after.tokens_failed_tasks_total
    );
    assert!(
        no_progress_delta >= 1,
        "expected no_progress_iterations_total to increase by at least 1; before={} after={}",
        before.no_progress_iterations_total,
        after.no_progress_iterations_total
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        self.calls.lock().await.push(arguments.to_string());
        let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
        let path = args["path"].as_str().unwrap_or(".");
        Ok(format!("No matches found (0 files scanned in {})", path))
    }

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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
        let path = args["path"].as_str().unwrap_or(".");
        Ok(format!(
            "# Project: {}\n\n## Structure\n```\nindex.html\nstyles.css\n```\n",
            path
        ))
    }

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

struct CountingSendFileTool {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for CountingSendFileTool {
    fn name(&self) -> &str {
        "send_file"
    }

    fn description(&self) -> &str {
        "Mock send_file tool for force-text characterization"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "send_file",
            "description": "Mock send file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "caption": {"type": "string"}
                },
                "required": ["file_path"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok("File sent successfully.".to_string())
    }

    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let path = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|args| {
                args.get("file_path")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_default();
        ToolCallSemantics::mutation_with(crate::traits::ToolMutationEffects::EXTERNAL_DELIVERY)
            .with_target_hint(ToolTargetHintKind::Path, path)
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

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(json!({
            "node_id": "synthetic-node-1",
            "delivery_status": "played",
            "playback_complete": true
        })
        .to_string())
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<tokio::sync::mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        Ok(ToolCallOutcome {
            output: self.call(arguments).await?,
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::Succeeded),
                semantics: Self::receipt_semantics(),
                ..ToolCallMetadata::default()
            },
        })
    }

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

struct BackgroundDetachTool;

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

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        Ok("Background job started.".to_string())
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<tokio::sync::mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let _ = (arguments, status_tx);
        Ok(ToolCallOutcome {
            output: "Background job started.".to_string(),
            metadata: ToolCallMetadata {
                background_started: true,
                detached: true,
                completion_notifications_enabled: true,
                ..ToolCallMetadata::default()
            },
        })
    }

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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
        let url = args["url"].as_str().unwrap_or("https://example.com/status");
        Ok(format!("Updated {}", url))
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
        let url = args["url"].as_str().unwrap_or("https://example.com/status");
        ToolCallSemantics::mutation().with_target_hint(ToolTargetHintKind::Url, url)
    }
}

struct CountingRemoteMutationTool {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for CountingRemoteMutationTool {
    fn name(&self) -> &str {
        "update_remote"
    }

    fn description(&self) -> &str {
        "Mock tool that updates remote state"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "update_remote",
            "description": "Mock remote update",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(format!("Updated {arguments}"))
    }

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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
        let url = args["url"].as_str().unwrap_or("https://example.com/status");
        Ok(format!("Verified {} shows the updated status.", url))
    }

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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
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
            Arc::new(crate::tools::ReadFileTool) as Arc<dyn Tool>,
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
async fn force_text_characterization_strips_tools_after_duplicate_send_file() {
    let send_file_args =
        r#"{"file_path":"/tmp/aidaemon-characterization.pdf","caption":"Characterization"}"#;
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("send_file", send_file_args),
        MockProvider::tool_call_response("send_file", send_file_args),
        MockProvider::tool_call_response("send_file", send_file_args),
        MockProvider::text_response("Done. I already sent the file."),
    ]);
    let send_file_calls = Arc::new(AtomicUsize::new(0));

    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![Arc::new(CountingSendFileTool {
            calls: send_file_calls.clone(),
        }) as Arc<dyn Tool>],
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "force_text_characterization",
            "Send me the characterization PDF",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "Done. I already sent the file.");
    assert_eq!(
        send_file_calls.load(Ordering::SeqCst),
        1,
        "duplicate send_file calls should be suppressed before force-text closeout"
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    assert!(
        call_log.last().is_some_and(|call| !call.tools.is_empty()
            && call.options.tool_choice == crate::traits::ToolChoiceMode::None),
        "force-text closeout retains tool defs (prompt-prefix stability) and \
         disables calling via tool_choice=none: {:?}",
        call_log.last().map(|call| &call.options.tool_choice)
    );
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
async fn stall_characterization_stops_repeated_unknown_tool_before_final_text() {
    let mut responses = Vec::new();
    for attempt in 1..=7 {
        responses.push({
            let mut resp = MockProvider::tool_call_response("unknown_stall_tool", "{}");
            resp.content = Some(format!("I'll retry the same tool, attempt {}.", attempt));
            resp
        });
    }
    responses.push(MockProvider::text_response("This should not be reached."));
    let provider = MockProvider::with_responses(responses);

    let harness = setup_test_agent(provider).await.unwrap();
    let reply = harness
        .agent
        .handle_message(
            "stall_characterization",
            "Use the unavailable stall tool",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        !reply.contains("This should not be reached"),
        "stall detection should stop repeated unknown-tool attempts before the scripted final text"
    );
    assert!(
        harness.provider.call_count().await < 8,
        "stall detection should stop early; provider calls: {}",
        harness.provider.call_count().await
    );
}

#[tokio::test]
async fn truncation_characterization_reassembles_mid_sentence_text_continuation() {
    let prefix = format!(
        "{} ",
        std::iter::repeat_n("partial", 205)
            .collect::<Vec<_>>()
            .join(" ")
    );
    let continuation = "and the final sentence is complete.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(&prefix),
        MockProvider::text_response(continuation),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let reply = harness
        .agent
        .handle_message(
            "truncation_characterization",
            "Draft a long status update",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, format!("{}{}", prefix, continuation));
    assert_eq!(
        harness.provider.call_count().await,
        2,
        "truncated first response should trigger exactly one continuation pass"
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    assert!(
        call_log.last().is_some_and(|call| {
            call.messages.iter().any(|message| {
                message.get("role").and_then(|v| v.as_str()) == Some("system")
                    && message
                        .get("content")
                        .and_then(|v| v.as_str())
                        .is_some_and(|content| {
                            content.contains("previous text response was cut off mid-sentence")
                                && content.contains("Continue your response")
                        })
            })
        }),
        "continuation pass should include the truncation recovery directive"
    );
}

#[tokio::test]
async fn truncation_characterization_keeps_prefix_when_short_tail_repeats_earlier_phrase() {
    let prefix = format!(
        "Which company or role are you targeting? {} The AI Expert resume is the ch",
        std::iter::repeat_n("detail", 205)
            .collect::<Vec<_>>()
            .join(" ")
    );
    let continuation = "osen one even stronger. Which company or role?";
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(&prefix),
        MockProvider::text_response(continuation),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let reply = harness
        .agent
        .handle_message(
            "truncation_short_overlapping_tail",
            "Which resume should I send?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, format!("{}{}", prefix, continuation));
}

#[tokio::test]
async fn background_ack_characterization_keeps_tools_available_for_unfulfilled_change() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("background_task", r#"{"job":"long-running"}"#),
        MockProvider::text_response("This model text should be ignored."),
    ])
    .with_task_assessments(vec![
        MockProvider::semantic_task_assessment_with_inline_continuation(
            "change",
            true,
            false,
            &["process_state"],
            "new_request",
            "host_local",
            true,
        ),
    ]);

    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![Arc::new(BackgroundDetachTool) as Arc<dyn Tool>],
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

    assert!(
        !reply.contains("This model text should be ignored"),
        "an unfulfilled change cannot close from model text alone: {reply}"
    );
    assert!(reply.contains("couldn't complete"), "{reply}");

    let call_log = harness.provider.call_log.lock().await.clone();
    assert!(
        call_log.iter().skip(1).any(|call| !call.tools.is_empty()
            && call.options.tool_choice != crate::traits::ToolChoiceMode::None),
        "an unfulfilled Change contract must retain tool definitions and execution capability"
    );
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
}

#[tokio::test]
async fn played_node_audio_receipt_closes_delivery_and_outcome_contract_in_one_call() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "send_node_audio",
            r#"{"node":"synthetic-node-1","text":"Dinner is ready."}"#,
        ),
        MockProvider::text_response("The audio was delivered and playback completed."),
    ])
    .with_task_assessments(vec![MockProvider::text_response(
        &json!({
            "schema_version": 7,
            "goal": "Deliver a spoken message and confirm playback",
            "steps": [],
            "success_criteria": [],
            "contract": {
                "confidence": "high",
                "task_kind": "deliver",
                "expects_mutation": true,
                "requires_observation": true,
                "required_effects": ["external_delivery"],
                "mutation_scope": "allowed",
                "forbidden_actions": [],
                "constraint_evidence": [],
                "tool_scope": "allowed",
                "tool_constraint_evidence": [],
                "minimum_sources": 0,
                "requires_primary_sources": false,
                "requires_exact_history": false,
                "evidence_requirements": [{
                    "summary": "Confirm the spoken message playback outcome",
                    "acceptable_scopes": ["external_remote"],
                    "purpose": "outcome",
                    "minimum_authority": "direct",
                    "temporal_scope": "current"
                }],
                "project_reference": null
            },
            "task_shape": {
                "execution_mode": "inline",
                "confidence": "high",
                "independent_workstreams": 1,
                "requires_background_continuation": false,
                "request_relationship": "new_request",
                "semantic_scope": "external_remote"
            }
        })
        .to_string(),
    )]);
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
        vec![Arc::new(BackgroundDetachTool) as Arc<dyn Tool>],
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

#[tokio::test]
async fn contradictory_file_evidence_forces_recheck_before_final_answer() {
    let project_dir = tempfile::tempdir().unwrap();
    let project_dir_str = project_dir.path().to_string_lossy().to_string();
    let search_calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("search_files", &json!({"glob":"*.html"}).to_string()),
        MockProvider::tool_call_response(
            "project_inspect",
            &json!({"path": project_dir_str}).to_string(),
        ),
        MockProvider::text_response("I couldn't find any HTML files."),
        MockProvider::tool_call_response(
            "search_files",
            &json!({"glob":"*.html", "path": project_dir_str}).to_string(),
        ),
        MockProvider::text_response(
            "After re-checking with an explicit path, I still have no HTML matches.",
        ),
    ]);

    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![
            Arc::new(RecordingSearchFilesTool {
                calls: search_calls.clone(),
            }) as Arc<dyn Tool>,
            Arc::new(MockProjectInspectTool) as Arc<dyn Tool>,
        ],
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "contradictory_file_recheck",
            &format!("Find HTML files under {}", project_dir_str),
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        reply,
        "After re-checking with an explicit path, I still have no HTML matches."
    );
    assert_eq!(harness.provider.call_count().await, 5);

    let calls = search_calls.lock().await.clone();
    assert_eq!(calls.len(), 2, "expected initial search + forced re-check");
    assert!(
        calls[0].contains("\"path\"") && calls[0].contains(&project_dir_str),
        "expected first search_files call to receive injected project path, got: {}",
        calls[0]
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    let contradiction_nudge_seen = call_log.iter().any(|entry| {
        entry.messages.iter().any(|m| {
            m.get("role").and_then(|v| v.as_str()) == Some("system")
                && m.get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|c| c.contains("Contradictory file evidence was detected"))
        })
    });
    assert!(
        contradiction_nudge_seen,
        "expected contradiction re-check system nudge in provider context"
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
async fn guided_model_forces_required_only_after_concrete_no_tool_deferral() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("I'll run the command now."),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Command inspection completed."),
    ])
    .with_task_assessments(vec![MockProvider::semantic_task_assessment(
        "check",
        false,
        true,
        &[],
        "new_request",
        "host_local",
    )]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "guided_concrete_tool_recovery",
            "Run the command pwd on this machine and summarize the result.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "Command inspection completed.");
    let call_log = harness.provider.call_log.lock().await.clone();
    assert!(
        call_log
            .iter()
            .any(|entry| matches!(entry.options.tool_choice, ToolChoiceMode::Required)),
        "guided recovery should force one tool call after the observed deferral"
    );
}

#[tokio::test]
async fn autonomous_model_gets_one_required_evidence_pass_for_execution_obligation() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("I'll run the command now."),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Command inspection completed."),
    ])
    .with_task_assessments(vec![MockProvider::semantic_task_assessment(
        "check",
        false,
        true,
        &[],
        "new_request",
        "host_local",
    )]);

    let harness = setup_test_agent_with_models(provider, "gpt-5-codex", "gpt-5-codex")
        .await
        .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "autonomous_concrete_tool_recovery",
            "Run the command pwd on this machine and summarize the result.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "Command inspection completed.");
    let call_log = harness.provider.call_log.lock().await.clone();
    assert!(
        call_log
            .iter()
            .any(|entry| matches!(entry.options.tool_choice, ToolChoiceMode::Required)),
        "typed execution obligations require one evidence pass regardless of model tier"
    );
}

#[tokio::test]
async fn planner_refined_observation_contract_requires_execution_until_observed() {
    let mut provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            r#"{
                "goal": "Compare two approaches using current system evidence",
                "steps": [
                    {"description": "Inspect current system evidence", "tool_hint": "system_info"},
                    {"description": "Compare the approaches", "tool_hint": null}
                ],
                "success_criteria": ["The comparison cites inspected evidence"],
                "contract": {
                    "confidence": "high",
                    "task_kind": "check",
                    "expects_mutation": false,
                    "requires_observation": true,
                    "mutation_scope": "allowed",
                    "forbidden_actions": [],
                    "constraint_evidence": []
                }
            }"#,
        ),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response(
            "Based on the inspected system evidence, approach A is faster; approach B is safer.",
        ),
    ]);
    provider.skip_planning_calls = false;

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "planner_refined_contract_tool_state",
            "Write a comprehensive comparison of approach A and approach B with recommendations.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        reply,
        "Based on the inspected system evidence, approach A is faster; approach B is safer."
    );
    let call_log = harness.provider.call_log.lock().await.clone();
    assert!(
        call_log.iter().any(|entry| entry
            .tools
            .iter()
            .any(|tool| tool["function"]["name"] == "system_info")),
        "the finalized observation contract must remain execution-required until evidence exists"
    );
}

#[tokio::test]
async fn typed_read_only_policy_is_not_revalidated_with_prose_matching() {
    let url = "https://example.com/status";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("update_remote", &json!({"url": url}).to_string()),
        MockProvider::tool_call_response("check_remote", &json!({"url": url}).to_string()),
        MockProvider::text_response("No remote mutation was performed."),
    ])
    .with_task_assessments(vec![MockProvider::text_response(
        r#"{
                "schema_version": 7,
                "goal": "Update remote status",
                "steps": [],
                "success_criteria": [],
                "contract": {
                    "confidence": "high",
                    "task_kind": "check",
                    "expects_mutation": false,
                    "requires_observation": true,
                    "required_effects": [],
                    "mutation_scope": "read_only",
                    "forbidden_actions": [],
                    "constraint_evidence": ["change locally but don't deploy"],
                    "tool_scope": "allowed",
                    "tool_constraint_evidence": [],
                    "minimum_sources": 0,
                    "requires_primary_sources": false,
                    "requires_exact_history": false,
                    "evidence_requirements": [{
                        "summary": "Observe the current remote status",
                        "acceptable_scopes": ["external_remote"],
                        "purpose": "current_state",
                        "minimum_authority": "direct",
                        "temporal_scope": "current"
                    }],
                    "project_reference": null
                },
                "task_shape": {
                    "execution_mode": "inline",
                    "confidence": "high",
                    "independent_workstreams": 1,
                    "requires_background_continuation": false,
                    "request_relationship": "new_request",
                    "semantic_scope": "external_remote"
                }
            }"#,
    )]);
    let calls = Arc::new(AtomicUsize::new(0));
    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![
            Arc::new(CountingRemoteMutationTool {
                calls: calls.clone(),
            }) as Arc<dyn Tool>,
            Arc::new(MockRemoteObservationTool) as Arc<dyn Tool>,
        ],
    )
    .await
    .unwrap();

    let reply = harness
        .agent
        .handle_message(
            "ungrounded_negative_contract",
            &format!("Update the status at {url}."),
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(reply, "No remote mutation was performed.");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "typed policy must not depend on a matching verbatim prose span"
    );
}

#[tokio::test]
async fn failed_specialist_plan_reply_pivots_to_direct_tools() {
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

    assert!(
        reply.contains("Market Snapshot:"),
        "unexpected reply: {reply}"
    );
    assert!(!reply.contains("monitoring the system"));
    assert!(
        harness.provider.call_count().await >= 5,
        "repeated incomplete plans should trigger another tool-backed iteration"
    );
    let calls = harness.provider.call_log.lock().await;
    assert!(
        calls.iter().any(|call| {
            call.messages.iter().any(|message| {
                message
                    .get("content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| {
                        content.contains("Specialist delegation failed")
                            && content.contains("available direct tools")
                    })
            })
        }),
        "failed delegation should inject direct-tool recovery guidance"
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
    assert!(
        call_log
            .iter()
            .all(|entry| entry.options == ChatOptions::default()),
        "expected default chat options when the text-only pass is disabled"
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
