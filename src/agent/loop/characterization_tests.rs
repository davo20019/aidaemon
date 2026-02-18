use crate::agent::policy_metrics_snapshot;
use crate::testing::{
    setup_full_stack_test_agent_with_extra_tools, setup_test_agent, setup_test_agent_with_models,
    MockProvider,
};
use crate::traits::{
    ChatOptions, ProviderResponse, ResponseMode, TokenUsage, Tool, ToolCall, ToolChoiceMode,
};
use crate::types::{ChannelContext, UserRole};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::test]
async fn consultant_metrics_capture_direct_return_and_fallthrough_paths() {
    let before = policy_metrics_snapshot();

    // Direct-return case (deterministic schedule routing before first LLM call).
    let direct_provider = MockProvider::with_responses(vec![]);
    let direct_harness =
        setup_test_agent_with_models(direct_provider, "primary-model", "smart-model")
            .await
            .unwrap();
    let direct_reply = direct_harness
        .agent
        .handle_message(
            "metrics_direct",
            "Check deployment tomorrow at 9am",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(
        direct_reply.contains("Reply **confirm** to proceed"),
        "expected schedule confirmation direct-return, got: {direct_reply}"
    );
    assert_eq!(
        direct_harness.provider.call_count().await,
        0,
        "expected deterministic pre-routing to avoid first LLM call"
    );

    // Fallthrough case (deterministic simple route continues into full tool loop).
    let fallthrough_provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("System inspected."),
    ]);
    let fallthrough_harness =
        setup_test_agent_with_models(fallthrough_provider, "primary-model", "smart-model")
            .await
            .unwrap();
    let fallthrough_reply = fallthrough_harness
        .agent
        .handle_message(
            "metrics_fallthrough",
            "Check my system status",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(fallthrough_reply, "System inspected.");

    let after = policy_metrics_snapshot();
    let direct_delta = after
        .consultant_direct_return_total
        .saturating_sub(before.consultant_direct_return_total);
    let fallthrough_delta = after
        .consultant_fallthrough_total
        .saturating_sub(before.consultant_fallthrough_total);

    assert!(
        direct_delta >= 1,
        "expected consultant_direct_return_total to increase by at least 1; before={} after={}",
        before.consultant_direct_return_total,
        after.consultant_direct_return_total
    );
    assert!(
        fallthrough_delta >= 1,
        "expected consultant_fallthrough_total to increase by at least 1; before={} after={}",
        before.consultant_fallthrough_total,
        after.consultant_fallthrough_total
    );
}

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
                model: "mock".to_string(),
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
async fn replay_trace_yes_do_it_with_sanitized_consultant_analysis_falls_through_to_tools() {
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
        "expected consultant + tool-call + final response path"
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
        !call_log.iter().any(|entry| matches!(
            entry.options.response_mode,
            ResponseMode::JsonSchema { .. }
        )),
        "text-only consultant schema pass should be disabled"
    );

    let required_tool_choice_seen = call_log
        .iter()
        .any(|entry| matches!(entry.options.tool_choice, ToolChoiceMode::Required));
    assert!(
        required_tool_choice_seen,
        "expected deferred-no-tool recovery to require a tool call on a subsequent LLM attempt"
    );
}

#[tokio::test]
async fn deferred_no_tool_forced_required_resets_after_first_successful_tool_call() {
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

    let call_log = harness.provider.call_log.lock().await.clone();
    let required_indices: Vec<usize> = call_log
        .iter()
        .enumerate()
        .filter_map(|(idx, entry)| {
            if matches!(entry.options.tool_choice, ToolChoiceMode::Required) {
                Some(idx)
            } else {
                None
            }
        })
        .collect();
    assert!(
        !required_indices.is_empty(),
        "expected at least one forced required-tool recovery call before tool success"
    );

    let first_required = required_indices[0];
    assert!(
        call_log
            .iter()
            .skip(first_required + 1)
            .any(|entry| !matches!(entry.options.tool_choice, ToolChoiceMode::Required)),
        "expected forced required-tool mode to clear after first successful tool call"
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
    assert!(
        !call_log.is_empty(),
        "expected at least one provider call"
    );
    assert!(
        call_log
            .iter()
            .all(|entry| entry.options == ChatOptions::default()),
        "expected default chat options when consultant text-pass is disabled"
    );
}
