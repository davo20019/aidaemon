//! Tool-failure classification and the message-list repairs the loop runs
//! before each model call.

use std::collections::HashMap;

use aidaemon_events::traits::ToolOutcomeStatus;
use serde_json::{Value, json};
use std::borrow::Cow;
use tracing::warn;

pub fn strip_appended_diagnostics(raw: &str) -> Cow<'_, str> {
    aidaemon_events::traits::extract_primary_message_content(raw, &[])
}

/// Calls of one tool in a row before the loop asks whether the run is stuck.
pub const MAX_CONSECUTIVE_SAME_TOOL: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFailureClass {
    Semantic,
    Transient,
}

fn classify_http_status(status: u16) -> Option<ToolFailureClass> {
    if status >= 500 || matches!(status, 408 | 409 | 425 | 429) {
        return Some(ToolFailureClass::Transient);
    }
    if status >= 400 {
        return Some(ToolFailureClass::Semantic);
    }
    None
}

/// The leading markers the daemon's own tools write when they return a
/// refusal or failure as text (`Error: …`, `Blocked: …`): the daemon wrote
/// these, at the start of the result, so they are a protocol marker, not
/// the result's words. Text anywhere else in a result is the result's
/// content and never decides failure.
const DAEMON_FAILURE_MARKERS: [&str; 6] = [
    "ERROR:",
    "Error:",
    "Failed to ",
    "Request blocked:",
    "Blocked:",
    "[SYSTEM] BLOCKED:",
];

fn starts_with_daemon_failure_marker(text: &str) -> bool {
    DAEMON_FAILURE_MARKERS
        .iter()
        .any(|marker| text.starts_with(marker))
}

/// Tools whose results are file/data content, not tool execution status.
fn is_data_content_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file" | "search_files" | "read_channel_history" | "web_search" | "policy_metrics"
    )
}

/// Tools that return external data, whose bodies are the remote site's.
fn is_external_data_tool(tool_name: &str) -> bool {
    matches!(tool_name, "web_fetch" | "http_request")
}

/// A JSON result's own top-level failure fields, a machine format: a 4xx/5xx
/// `status`/`status_code`/`http_status`, `success: false`/`ok: false`, a
/// non-null `error` or a non-empty `errors`. Nested values are the result's
/// data, and no string value is read for its words.
fn classify_json_result(value: &Value) -> Option<ToolFailureClass> {
    let map = value.as_object()?;
    for key in [
        "status",
        "status_code",
        "statusCode",
        "http_status",
        "httpStatus",
    ] {
        let status = match map.get(key) {
            Some(Value::Number(n)) => n.as_u64().and_then(|v| u16::try_from(v).ok()),
            _ => None,
        };
        if let Some(kind) = status.and_then(classify_http_status) {
            return Some(kind);
        }
    }
    let failed_flag = map.get("success").and_then(Value::as_bool) == Some(false)
        || map.get("ok").and_then(Value::as_bool) == Some(false);
    let error = map.get("error").is_some_and(|error| !error.is_null());
    let errors = match map.get("errors") {
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(fields)) => !fields.is_empty(),
        _ => false,
    };
    (failed_flag || error || errors).then_some(ToolFailureClass::Semantic)
}

/// Failure read from a result's text when the tool reported nothing typed.
/// Only two things count: the daemon's own leading failure marker, and a
/// whole-result JSON object's top-level failure fields. Words inside a
/// result ("invalid", "timeout", "forbidden", an HTTP status line, an exit
/// code someone printed) are its content: they never decide failure, so a
/// result with no typed signal and neither of these is not a failure.
pub fn classify_tool_result_failure(
    tool_name: &str,
    result_text: &str,
) -> Option<ToolFailureClass> {
    let cleaned = strip_appended_diagnostics(result_text);
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        return None;
    }
    if starts_with_daemon_failure_marker(cleaned) {
        return Some(ToolFailureClass::Semantic);
    }
    if is_data_content_tool(tool_name) || is_external_data_tool(tool_name) {
        return None;
    }
    if (cleaned.starts_with('{') || cleaned.starts_with('['))
        && let Ok(value) = serde_json::from_str::<Value>(cleaned)
    {
        return classify_json_result(&value);
    }
    None
}

pub fn classify_tool_result_failure_with_args(
    tool_name: &str,
    result_text: &str,
    tool_arguments: Option<&str>,
) -> Option<ToolFailureClass> {
    if crate::turn::tool_facts::memory_tool_facts(tool_name)
        .and_then(|facts| facts.output_is_report)
        .is_some_and(|is_report| is_report(result_text, tool_arguments))
    {
        return None;
    }
    classify_tool_result_failure(tool_name, result_text)
}

/// Whether a command tool call runs a test runner, judged by the parsed
/// program and subcommand of each shell invocation (nested `sh -c` included),
/// never by a substring of the arguments: `curl …/latest` is not a test run.
fn invokes_test_runner(args_json: &str) -> bool {
    let Some(command) = extract_command_from_args(args_json) else {
        return false;
    };
    aidaemon_tools::command_risk::structural_shell_invocations(&command)
        .iter()
        .any(|invocation| {
            let program = std::path::Path::new(&invocation.program)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&invocation.program);
            let args: Vec<&str> = invocation.arguments.iter().map(String::as_str).collect();
            let subcommand = args.iter().find(|arg| !arg.starts_with('-')).copied();
            match program {
                "pytest" | "py.test" | "jest" | "vitest" | "mocha" | "rspec" | "phpunit"
                | "tox" | "nox" | "ctest" | "bats" | "cargo-nextest" => true,
                "cargo" => matches!(subcommand, Some("test" | "nextest")),
                "go" | "bun" | "deno" | "dotnet" | "mvn" | "gradle" | "gradlew" | "swift"
                | "mix" | "zig" => subcommand == Some("test"),
                "npm" | "yarn" | "pnpm" => match subcommand {
                    Some("test" | "t") => true,
                    Some("run") => args
                        .iter()
                        .skip_while(|arg| **arg != "run")
                        .nth(1)
                        .is_some_and(|script| script.split(':').next() == Some("test")),
                    _ => false,
                },
                "make" => args.iter().any(|arg| arg.split(':').next() == Some("test")),
                "python" | "python3" => args
                    .windows(2)
                    .any(|pair| pair[0] == "-m" && matches!(pair[1], "pytest" | "unittest")),
                _ => false,
            }
        })
}

pub fn classify_tool_result_failure_with_context(
    tool_name: &str,
    result_text: &str,
    tool_arguments: Option<&str>,
    metadata: Option<&aidaemon_events::traits::ToolCallMetadata>,
) -> Option<ToolFailureClass> {
    if let Some(meta) = metadata {
        if let Some(status) = meta.outcome_status {
            return failure_class_from_outcome_status(status);
        }
        if meta.transport_error.is_some() {
            // The tool raised an error and reported no outcome of its own:
            // a failure, of a kind it did not type. Whether it may be
            // retried is the tool's to say (`outcome_status`, or a typed
            // error the dispatcher maps), never read from the error's text.
            return Some(ToolFailureClass::Semantic);
        }
        if meta.background_started {
            // A command moved to background tracking sets timed_out alongside
            // background_started, but the handoff is not a failure — the
            // command is still running and its completion is delivered later.
            return None;
        }
        if meta.timed_out {
            return Some(ToolFailureClass::Transient);
        }
        if let Some(code) = meta.exit_code {
            if code != 0 {
                // Test runners exit non-zero when tests fail — that's expected
                // informational output, not a tool failure. Let the LLM read
                // and interpret the test results instead of entering a repair loop.
                if !tool_arguments.is_some_and(invokes_test_runner) {
                    return Some(ToolFailureClass::Semantic);
                }
                return None;
            }
            // Explicit terminal success should not be reclassified by text.
            if tool_name == "terminal" {
                return None;
            }
        }
        if let Some(status) = meta.http_status {
            // Structured HTTP status is the call's outcome.
            return classify_http_status(status);
        }
    }
    classify_tool_result_failure_with_args(tool_name, result_text, tool_arguments)
}

pub fn classify_tool_outcome_status(
    tool_name: &str,
    result_text: &str,
    tool_arguments: Option<&str>,
    metadata: Option<&aidaemon_events::traits::ToolCallMetadata>,
) -> ToolOutcomeStatus {
    if let Some(status) = metadata.and_then(|meta| meta.outcome_status) {
        return status;
    }
    if metadata.is_some_and(|meta| meta.background_started || meta.detached) {
        return ToolOutcomeStatus::Backgrounded;
    }
    match classify_tool_result_failure_with_context(
        tool_name,
        result_text,
        tool_arguments,
        metadata,
    ) {
        Some(ToolFailureClass::Transient) => return ToolOutcomeStatus::FailedRetryable,
        Some(ToolFailureClass::Semantic) => return ToolOutcomeStatus::FailedPermanent,
        None => {}
    }
    if metadata
        .and_then(|meta| meta.exit_code)
        .is_some_and(|code| code != 0)
    {
        return ToolOutcomeStatus::CompletedWithNegativeResult;
    }
    ToolOutcomeStatus::Succeeded
}

/// Report the strength of the evidence used by `classify_tool_outcome_status`.
/// This is persisted with the receipt so recovery and completion logic can
/// distinguish an authoritative typed outcome from a compatibility parse of
/// model-visible text.
pub fn tool_outcome_evidence_source(
    metadata: &aidaemon_events::traits::ToolCallMetadata,
) -> aidaemon_events::ToolOutcomeEvidenceSource {
    if metadata.receipt_replayed {
        return aidaemon_events::ToolOutcomeEvidenceSource::DurableReplay;
    }
    if metadata.outcome_status.is_some() {
        return aidaemon_events::ToolOutcomeEvidenceSource::ToolReported;
    }
    if metadata.exit_code.is_some()
        || metadata.timed_out
        || metadata.background_started
        || metadata.detached
        || metadata.transport_error.is_some()
        || metadata.http_status.is_some()
    {
        return aidaemon_events::ToolOutcomeEvidenceSource::StructuredMetadata;
    }
    aidaemon_events::ToolOutcomeEvidenceSource::LegacyText
}

pub fn failure_class_from_outcome_status(
    outcome_status: ToolOutcomeStatus,
) -> Option<ToolFailureClass> {
    match outcome_status {
        ToolOutcomeStatus::FailedRetryable => Some(ToolFailureClass::Transient),
        ToolOutcomeStatus::FailedPermanent | ToolOutcomeStatus::Blocked => {
            Some(ToolFailureClass::Semantic)
        }
        ToolOutcomeStatus::Succeeded
        | ToolOutcomeStatus::CompletedWithNegativeResult
        | ToolOutcomeStatus::Backgrounded => None,
    }
}

/// Merge consecutive messages with the same role (`user` or `assistant`) into
/// one message, preserving all content and tool calls.
pub fn merge_consecutive_messages(messages: &mut Vec<Value>) {
    if messages.len() <= 1 {
        return;
    }
    let mut i = 1;
    while i < messages.len() {
        let prev_role = messages[i - 1]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_string();
        let curr_role = messages[i]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_string();
        if (curr_role == "assistant" || curr_role == "user") && prev_role == curr_role {
            let prev_content_value = messages[i - 1].get("content");
            let curr_content_value = messages[i].get("content");
            // Never merge multimodal (array) or other non-string content — merging
            // would destroy image blocks.
            if !prev_content_value.is_some_and(|c| c.is_string())
                || !curr_content_value.is_some_and(|c| c.is_string())
            {
                i += 1;
                continue;
            }
            let curr_content = curr_content_value
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            let prev_content = prev_content_value
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            if curr_role == "user"
                && !prev_content.is_empty()
                && !curr_content.is_empty()
                && prev_content != curr_content
            {
                i += 1;
                continue;
            }
            let merged = if prev_content.is_empty() {
                curr_content
            } else if curr_content.is_empty() {
                prev_content
            } else {
                format!("{}\n{}", prev_content, curr_content)
            };
            messages[i - 1]["content"] = json!(merged);
            // Merge assistant tool_calls arrays if both have them.
            if curr_role == "assistant"
                && let Some(curr_tcs) = messages[i]
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .cloned()
            {
                if let Some(prev_tcs) = messages[i - 1]
                    .get_mut("tool_calls")
                    .and_then(|v| v.as_array_mut())
                {
                    prev_tcs.extend(curr_tcs);
                } else {
                    messages[i - 1]["tool_calls"] = json!(curr_tcs);
                }
            }
            messages.remove(i);
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tool_error_detection_tests {
    use super::{
        ToolFailureClass, classify_tool_result_failure, classify_tool_result_failure_with_args,
        classify_tool_result_failure_with_context,
    };
    use aidaemon_events::traits::{ToolCallMetadata, ToolOutcomeStatus};

    #[test]
    fn the_daemons_leading_failure_marker_is_a_failure() {
        for result in [
            "Error: request timed out while connecting to api",
            "Error: specialist failed: invalid research request",
            "Request blocked: tool-only parameters were embedded in the URL (auth_profile, body).",
            "Blocked: outside the workspace",
        ] {
            assert_eq!(
                classify_tool_result_failure("spawn_agent", result),
                Some(ToolFailureClass::Semantic),
                "{result}"
            );
        }
    }

    /// Words inside a result are its content: with no typed signal, an
    /// answer that quotes an error status, a timeout, an exit code or a
    /// "forbidden" is not a failure, and is not retried as one.
    #[test]
    fn words_inside_a_result_never_decide_failure() {
        for (tool, result) in [
            (
                "spawn_agent",
                "The blog post is live. `curl -I` returned:\nHTTP/2 200 \nserver: cloudflare",
            ),
            (
                "spawn_agent",
                "The request failed. Response line: HTTP/1.1 503 Service Unavailable",
            ),
            (
                "spawn_agent",
                "Fixed the handler so it returns forbidden for guests.",
            ),
            ("terminal", "Attempt 2: 503"),
            ("terminal", "$ make test (exit: 2, 22ms)"),
            ("terminal", "[Process pid=123 finished]\n[exit code: 42]"),
            ("http_request", "Status: 404 Not Found"),
            (
                "synthetic_mcp__import",
                "Imported 120 rows, 0 invalid rows, no timeout.",
            ),
            (
                "terminal",
                "$ curl -s https://api.example.com (exit: 0, 9ms)\n\n{\"error\":\"not found\"}",
            ),
            (
                "search_files",
                "Search complete: 24 files scanned, 0 patterns not found",
            ),
        ] {
            assert_eq!(
                classify_tool_result_failure(tool, result),
                None,
                "{tool}: {result}"
            );
        }
    }

    #[test]
    fn a_whole_json_result_is_read_by_its_top_level_failure_fields() {
        let error = r#"{"error":"invalid arguments: missing required field"}"#;
        assert_eq!(
            classify_tool_result_failure("manage_memories", error),
            Some(ToolFailureClass::Semantic)
        );
        assert_eq!(
            classify_tool_result_failure("synthetic_tool", r#"{"status":503}"#),
            Some(ToolFailureClass::Transient)
        );
        // Field names, nested values and string values are data.
        for result in [
            r#"{"metrics":{"tokens_failed_tasks_total":0,"llm_payload_invalid_total":0}}"#,
            r#"{"connection_state":"ok","recovery":{"watchdog_timeout_seconds":90}}"#,
            r#"{"stop_conditions":["Any unauthorized API behavior."]}"#,
            r#"{"items":[{"error":"a stored row's own field"}]}"#,
            r#"{"message":"Error: timed out (quoted)"}"#,
        ] {
            assert_eq!(
                classify_tool_result_failure("synthetic_tool", result),
                None,
                "{result}"
            );
        }
    }

    #[test]
    fn a_report_tool_output_is_data() {
        let args = r#"{"action":"list"}"#;
        let result = "**Stored Memories** (showing 2 of 2 facts)\n\n- **api_auth_resolution**: Uses manage_oauth to resolve HTTP 401 Unauthorized errors";
        assert_eq!(
            classify_tool_result_failure_with_args("manage_memories", result, Some(args)),
            None
        );
    }

    #[test]
    fn structured_exit_code_decides_a_terminal_result() {
        let success = ToolCallMetadata {
            exit_code: Some(0),
            ..Default::default()
        };
        assert_eq!(
            classify_tool_result_failure_with_context(
                "terminal",
                "Error: printed by the program, which still exited 0",
                None,
                Some(&success),
            ),
            None
        );
        let failure = ToolCallMetadata {
            exit_code: Some(2),
            ..Default::default()
        };
        assert_eq!(
            classify_tool_result_failure_with_context(
                "terminal",
                "all good text otherwise",
                None,
                Some(&failure),
            ),
            Some(ToolFailureClass::Semantic)
        );
    }

    #[test]
    fn test_runner_negative_result_is_not_counted_as_success() {
        let metadata = ToolCallMetadata {
            exit_code: Some(1),
            ..Default::default()
        };
        let status = super::classify_tool_outcome_status(
            "terminal",
            "2 tests failed",
            Some(r#"{"command":"cargo test"}"#),
            Some(&metadata),
        );
        assert_eq!(status, ToolOutcomeStatus::CompletedWithNegativeResult);
        assert!(!status.satisfies_requested_condition());
        assert!(!status.is_failure());
    }

    #[test]
    fn test_runner_is_recognized_by_command_structure() {
        for command in [
            "pytest -q",
            "cd app && npm run test:unit",
            "bash -c 'python3 -m pytest tests/'",
            "cargo nextest run",
            "make test",
            "yarn test --watch=false",
        ] {
            assert!(
                super::invokes_test_runner(&serde_json::json!({ "command": command }).to_string()),
                "{command} runs tests"
            );
        }
        for command in [
            "curl -fsS https://example.com/releases/latest",
            "cat contest_results.txt",
            "grep -r test src/",
            "ls tests/",
            "npm run build",
        ] {
            assert!(
                !super::invokes_test_runner(&serde_json::json!({ "command": command }).to_string()),
                "{command} does not run tests"
            );
        }
    }

    #[test]
    fn failing_command_mentioning_test_in_an_argument_is_a_failure() {
        let metadata = ToolCallMetadata {
            exit_code: Some(22),
            ..Default::default()
        };
        let status = super::classify_tool_outcome_status(
            "terminal",
            "curl: (22) The requested URL returned error: 404",
            Some(r#"{"command":"curl -fsS https://example.com/releases/latest"}"#),
            Some(&metadata),
        );
        assert!(status.is_failure(), "got {status:?}");
    }

    #[test]
    fn tool_supplied_outcome_status_is_authoritative() {
        let metadata = ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::Blocked),
            exit_code: Some(0),
            ..Default::default()
        };
        assert_eq!(
            super::classify_tool_outcome_status("terminal", "success", None, Some(&metadata)),
            ToolOutcomeStatus::Blocked
        );
        // An MCP result the server did not flag `isError` is a success,
        // whatever words its data holds.
        let mcp = ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::Succeeded),
            ..Default::default()
        };
        assert_eq!(
            super::classify_tool_outcome_status(
                "synthetic_mcp__import",
                "Error: 0 invalid rows",
                None,
                Some(&mcp),
            ),
            ToolOutcomeStatus::Succeeded
        );
    }

    /// A raised tool error with no typed outcome is a failure whose kind
    /// was not typed: its text never makes it retryable.
    #[test]
    fn an_untyped_transport_error_is_a_failure_never_guessed_transient() {
        for error in [
            "Path is a directory, not a file: /tmp/synthetic",
            "connection refused",
            "HTTP 503 from upstream",
        ] {
            let metadata = ToolCallMetadata {
                transport_error: Some(error.to_string()),
                ..Default::default()
            };
            assert_eq!(
                classify_tool_result_failure_with_context(
                    "web_fetch",
                    &format!("Error: {error}"),
                    None,
                    Some(&metadata),
                ),
                Some(ToolFailureClass::Semantic),
                "{error}"
            );
        }
    }

    #[test]
    fn background_handoff_is_not_classified_as_failure() {
        // A command moved to background sets timed_out=true alongside
        // background_started=true. The handoff is not a failure — the command
        // is still running and may well succeed.
        let metadata = ToolCallMetadata {
            background_started: true,
            timed_out: true,
            ..Default::default()
        };
        let classified = classify_tool_result_failure_with_context(
            "terminal",
            "Command still running after 30s. Moved to background (pid=5822).",
            None,
            Some(&metadata),
        );
        assert_eq!(classified, None);
    }

    #[test]
    fn a_typed_timeout_is_transient() {
        let metadata = ToolCallMetadata {
            timed_out: true,
            ..Default::default()
        };
        let classified =
            classify_tool_result_failure_with_context("web_fetch", "", None, Some(&metadata));
        assert_eq!(classified, Some(ToolFailureClass::Transient));
    }

    #[test]
    fn structured_http_status_decides_an_http_result() {
        for (status, expected) in [
            (403, Some(ToolFailureClass::Semantic)),
            (429, Some(ToolFailureClass::Transient)),
            (200, None),
        ] {
            let meta = ToolCallMetadata {
                http_status: Some(status),
                ..Default::default()
            };
            assert_eq!(
                classify_tool_result_failure_with_context(
                    "http_request",
                    "Error: the body of a page that talks about errors",
                    None,
                    Some(&meta),
                ),
                expected,
                "{status}"
            );
        }
    }
}

/// Fix message ordering to satisfy provider requirements:
/// 1. Drop orphaned tool messages (tool_call_id without corresponding assistant tool_call)
/// 2. Drop orphaned assistant tool_calls (without corresponding tool result)
/// 3. Merge any consecutive same-role user/assistant messages (stability)
/// 4. Ensure conversation starts with user/system (not tool)
pub fn fixup_message_ordering(messages: &mut Vec<Value>) {
    // Pass 0: initial coalescing helps reduce edge cases.
    merge_consecutive_messages(messages);

    // Pass 1: remove orphaned, out-of-order, and duplicate tool messages. A
    // provider can only consume a tool result after the matching assistant
    // tool call has appeared in the transcript. A global ID-membership check is
    // insufficient here: context fitting can retain a tool result while moving
    // or dropping its call, and a later call with the same ID must not make that
    // earlier result look valid.
    let mut pending_tool_call_ids = std::collections::HashSet::new();
    messages.retain(|m| match m.get("role").and_then(|r| r.as_str()) {
        Some("assistant") => {
            if let Some(tool_calls) = m.get("tool_calls").and_then(|v| v.as_array()) {
                pending_tool_call_ids.extend(
                    tool_calls
                        .iter()
                        .filter_map(|tc| tc.get("id").and_then(|id| id.as_str()))
                        .filter(|id| !id.is_empty())
                        .map(str::to_string),
                );
            }
            true
        }
        Some("tool") => {
            let tc_id = m
                .get("tool_call_id")
                .and_then(|id| id.as_str())
                .unwrap_or("");
            if pending_tool_call_ids.remove(tc_id) {
                true
            } else {
                warn!(
                    tool_call_id = tc_id,
                    "Dropping tool message without a preceding unmatched assistant tool call"
                );
                false
            }
        }
        _ => true,
    });

    let tool_result_ids: std::collections::HashSet<String> = messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
        .filter_map(|m| m.get("tool_call_id").and_then(|id| id.as_str()))
        .map(|s| s.to_string())
        .collect();

    // Pass 2: prune orphaned assistant tool_calls.
    for m in messages.iter_mut() {
        if m.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }
        if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()).cloned() {
            let kept: Vec<Value> = tcs
                .into_iter()
                .filter(|tc| {
                    tc.get("id")
                        .and_then(|id| id.as_str())
                        .is_some_and(|id| tool_result_ids.contains(id))
                })
                .collect();
            if kept.is_empty() {
                // Remove empty tool_calls to avoid malformed assistant tool-call message.
                m.as_object_mut().map(|o| o.remove("tool_calls"));
            } else {
                m["tool_calls"] = Value::Array(kept);
            }
        }
    }

    // Pass 3: remove empty assistant tool-call placeholders if no content and no tool_calls remain.
    messages.retain(|m| {
        if m.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            return true;
        }
        let has_content = m
            .get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|s| !s.is_empty());
        let has_tool_calls = m
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty());
        has_content || has_tool_calls
    });

    // Pass 4: merge any newly-consecutive same-role messages.
    merge_consecutive_messages(messages);

    // Ensure first non-system message is not assistant/tool if possible by trimming
    // until first user.
    if let Some(first_non_system) = messages
        .iter()
        .position(|m| m.get("role").and_then(|r| r.as_str()) != Some("system"))
    {
        let first_role = messages[first_non_system]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        if first_role != "user"
            && let Some(first_user_rel) = messages[first_non_system..]
                .iter()
                .position(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        {
            let abs_end = first_non_system + first_user_rel;
            warn!(
                dropped = abs_end - first_non_system,
                "Dropping leading non-user messages to satisfy provider ordering"
            );
            messages.drain(first_non_system..abs_end);
        }
    }

    // Pass 5: Gemini-specific - ensure assistant messages with tool_calls only follow
    // user or tool messages (not other assistant messages).
    // If we find assistant->assistant where the second has tool_calls, merge them.
    let mut i = 1;
    while i < messages.len() {
        let prev_role = messages[i - 1]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        let curr_role = messages[i]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        let curr_has_tc = messages[i]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty());

        // If current is assistant with tool_calls, and previous is also assistant
        // (which shouldn't happen after Pass 3, but just in case), merge them
        if prev_role == "assistant" && curr_role == "assistant" && curr_has_tc {
            warn!(
                "Pass 5: Found consecutive assistant messages, merging to satisfy Gemini constraint"
            );
            // Merge content
            let curr_content = messages[i]
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            let prev_content = messages[i - 1]
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            if !curr_content.is_empty() {
                let merged = if prev_content.is_empty() {
                    curr_content
                } else {
                    format!("{}\n{}", prev_content, curr_content)
                };
                messages[i - 1]["content"] = json!(merged);
            }
            // Merge tool_calls
            if let Some(curr_tcs) = messages[i]
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .cloned()
            {
                if let Some(prev_tcs) = messages[i - 1]
                    .get_mut("tool_calls")
                    .and_then(|v| v.as_array_mut())
                {
                    prev_tcs.extend(curr_tcs);
                } else {
                    messages[i - 1]["tool_calls"] = json!(curr_tcs);
                }
            }
            messages.remove(i);
        } else {
            i += 1;
        }
    }

    // Pass 6: Gemini-specific - ensure assistant messages with tool_calls only follow
    // user or tool messages. If an assistant(tc) follows another assistant(tc) or
    // a plain assistant, strip the tool_calls from the second one (keeping any content).
    // This handles edge cases not caught by Pass 5.
    let mut i = 1;
    while i < messages.len() {
        let curr_role = messages[i]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        let curr_has_tc = messages[i]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty());

        if curr_role == "assistant" && curr_has_tc {
            let prev_role = messages[i - 1]
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("");
            // Valid predecessors for assistant with tool_calls: "user" or "tool"
            if prev_role != "user" && prev_role != "tool" {
                warn!(
                    prev_role,
                    "Pass 6: Stripping tool_calls from assistant that doesn't follow user/tool"
                );
                // Remove tool_calls but keep content
                messages[i].as_object_mut().map(|o| o.remove("tool_calls"));
                // If this leaves an empty assistant, it will be caught by the retain logic
                // but since we're past that, let's check now
                let has_content = messages[i]
                    .get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| !s.is_empty());
                if !has_content {
                    messages.remove(i);
                    continue;
                }
            }
        }
        i += 1;
    }
}

/// Collapse repeated tool error payloads in the current interaction to avoid
/// runaway context growth.
///
/// Strategy: for each tool name, keep only the most recent error details and
/// replace earlier errors with a short note. A successful (non-error) tool
/// result resets the error streak for that tool.
pub fn collapse_repeated_tool_errors(messages: &mut [Value]) -> usize {
    let Some(boundary) = messages
        .iter()
        .rposition(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
    else {
        return 0;
    };

    let mut collapsed = 0usize;
    let mut last_error_idx_by_tool: HashMap<String, usize> = HashMap::new();

    for idx in boundary.saturating_add(1)..messages.len() {
        let role = messages[idx]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        if role != "tool" {
            continue;
        }
        let name = messages[idx]
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();
        if name.is_empty() {
            continue;
        }
        let content = messages[idx]
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        // The receipt line's typed outcome when the result carries one.
        let is_error = match aidaemon_events::ToolReceiptV1::outcome_in_context_header(content) {
            Some(status) => status.is_failure(),
            None => classify_tool_result_failure(&name, content).is_some(),
        };

        if is_error {
            if let Some(prev_idx) = last_error_idx_by_tool.insert(name.clone(), idx) {
                // Keep the latest error details; earlier errors become a short note.
                messages[prev_idx]["content"] = json!(format!(
                    "Error: (previous {} error collapsed; see the most recent {} error for details)",
                    name, name
                ));
                collapsed += 1;
            }
        } else {
            last_error_idx_by_tool.remove(&name);
        }
    }

    collapsed
}

/// Extract the "command" field from tool arguments JSON (for terminal tool).
pub fn extract_command_from_args(args_json: &str) -> Option<String> {
    serde_json::from_str::<Value>(args_json)
        .ok()
        .and_then(|v| v.get("command")?.as_str().map(String::from))
}

/// Build a stable dedupe key for send_file calls within a single task.
/// Key format: "{expanded_path}|{trimmed_caption}".
pub fn extract_send_file_dedupe_key_from_args(args_json: &str) -> Option<String> {
    let parsed = serde_json::from_str::<Value>(args_json).ok()?;
    let file_path = parsed.get("file_path")?.as_str()?.trim();
    if file_path.is_empty() {
        return None;
    }
    let expanded_path = shellexpand::tilde(file_path).to_string();
    let caption = parsed
        .get("caption")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    Some(format!("{}|{}", expanded_path, caption))
}

pub use aidaemon_work::triggers::is_trigger_session;

#[cfg(test)]
#[path = "message_ordering_tests.rs"]
mod message_ordering_tests;
