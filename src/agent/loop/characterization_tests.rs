use crate::agent::policy_metrics_snapshot;
use crate::testing::{
    setup_full_stack_test_agent_with_extra_tools, setup_test_agent,
    setup_test_agent_root_with_extra_tools_and_llm_timeout,
    setup_test_agent_with_extra_tools_and_llm_timeout, setup_test_agent_with_models, MockProvider,
    MockTool,
};
use crate::traits::{
    ChatOptions, ProviderResponse, ResponseMode, TokenUsage, Tool, ToolCall, ToolCallMetadata,
    ToolCallOutcome, ToolCallSemantics, ToolChoiceMode, ToolTargetHintKind, ToolVerificationMode,
};
use crate::types::{ChannelContext, StatusUpdate, UserRole};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::test]
#[ignore = "tokens_failed_tasks_total / no_progress_iterations_total not yet wired to agent loop"]
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
    .with_task_assessments(vec![MockProvider::semantic_task_assessment(
        "change",
        true,
        false,
        &["process_state"],
        "new_request",
        "host_local",
    )]);

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

    assert_eq!(reply, "This model text should be ignored.");

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(
        call_log.len(),
        2,
        "background detach currently runs one forced text summary pass after the tool call"
    );
    assert!(
        call_log.last().is_some_and(|call| !call.tools.is_empty()
            && call.options.tool_choice != crate::traits::ToolChoiceMode::None),
        "an unfulfilled Change contract must retain tool definitions and execution capability"
    );
    assert!(
        call_log.last().is_some_and(|call| {
            call.messages.iter().any(|message| {
                message.get("role").and_then(|v| v.as_str()) == Some("system")
                    && message
                        .get("content")
                        .and_then(|v| v.as_str())
                        .is_some_and(|content| {
                            content.contains("A background task is now running")
                                && content.contains("completion notifications are enabled")
                        })
            })
        }),
        "background detach should carry a handoff directive into the forced text pass"
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
async fn replay_trace_deferred_planning_text_does_not_stall_before_first_tool_call() {
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

    // The agent may either:
    // 1. Run all 5 responses and return the final text (old behavior)
    // 2. Stop earlier due to deferred-no-tool detection returning an intermediate text
    // Both are acceptable — the key is no crash and a non-empty response.
    assert!(
        !reply.is_empty(),
        "Agent should return a non-empty response"
    );
    // At minimum some deferral retries should fire before recovery.
    assert!(
        harness.provider.call_count().await >= 3,
        "expected at least a few retries before deferred/no-tool recovery"
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    assert!(
        !call_log
            .iter()
            .any(|entry| matches!(entry.options.response_mode, ResponseMode::JsonSchema { .. })),
        "text-only schema pass should be disabled"
    );
    // Generic complexity is advisory, so it cannot force a tool call by itself.
    // Only a finalized mutation/observation contract or the narrow deterministic
    // intent signal can support guided-model forced recovery.
}

#[tokio::test]
async fn generic_deferred_no_tool_recovery_does_not_force_required() {
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

    assert_eq!(reply, "Final summary: system inspection completed.");

    // No finalized concrete tool requirement exists for this generic request.
    // Deferred-action recovery can still drive the model toward a tool, but it
    // must not force provider-level Required mode.
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
async fn ungrounded_negative_classifier_output_cannot_block_requested_mutation() {
    let url = "https://example.com/status";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("update_remote", &json!({"url": url}).to_string()),
        MockProvider::tool_call_response("check_remote", &json!({"url": url}).to_string()),
        MockProvider::text_response("Remote status updated."),
    ])
    .with_task_assessments(vec![MockProvider::text_response(
        r#"{
                "schema_version": 2,
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
                    "minimum_sources": 0,
                    "requires_primary_sources": false,
                    "requires_exact_history": false,
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

    assert_eq!(reply, "Remote status updated.");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "an invented restriction without a verbatim span must not suppress mutation"
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
    let spawn_tool: Arc<dyn Tool> = Arc::new(MockTool::new(
        "spawn_agent",
        "Mock failed specialist delegation",
        "Error: specialist timed out after 300 seconds",
    ));
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
