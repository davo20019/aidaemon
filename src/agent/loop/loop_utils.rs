use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};
use tracing::warn;

pub(super) fn strip_appended_diagnostics(raw: &str) -> &str {
    // Keep only the "core" tool output so pattern extraction and retrieval don't
    // get polluted by injected meta blocks.
    const MARKERS: [&str; 3] = ["\n\n[DIAGNOSTIC]", "\n\n[TOOL STATS]", "\n\n[SYSTEM]"];
    let mut cut_at: Option<usize> = None;
    for m in MARKERS {
        if let Some(idx) = raw.find(m) {
            cut_at = Some(cut_at.map(|c| c.min(idx)).unwrap_or(idx));
        }
    }
    let trimmed = match cut_at {
        Some(idx) => &raw[..idx],
        None => raw,
    };
    trimmed.trim()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolFailureClass {
    Semantic,
    Transient,
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

static HTTP_STATUS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)\bhttp/\d(?:\.\d+)?\s+([1-5][0-9]{2})\b|\bstatus\s+code\s+([1-5][0-9]{2})\b|\bstatus\s*[:=]\s*([1-5][0-9]{2})\b|"(?:status|status_code|statusCode|http_status|httpStatus)"\s*:\s*"?([1-5][0-9]{2})"?"#)
        .expect("http status regex must compile")
});

static EXIT_CODE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:\bexit\s*:\s*|\bexit\s*code\s*[:=]?\s*|\[exit\s*code:\s*)(-?\d+)")
        .expect("exit code regex must compile")
});

fn classify_http_status(status: u16) -> Option<ToolFailureClass> {
    if status >= 500 || matches!(status, 408 | 409 | 425 | 429) {
        return Some(ToolFailureClass::Transient);
    }
    if status >= 400 {
        return Some(ToolFailureClass::Semantic);
    }
    None
}

fn extract_status_from_value(value: &Value) -> Option<u16> {
    match value {
        Value::Number(n) => n.as_u64().and_then(|v| u16::try_from(v).ok()),
        Value::String(s) => s.parse::<u16>().ok(),
        _ => None,
    }
}

fn classify_text_error(lower: &str) -> ToolFailureClass {
    if contains_any(
        lower,
        &[
            "rate limit",
            "too many requests",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "network error",
            "connection reset",
            "connection refused",
            "connection aborted",
            "connection failed",
            "retry later",
            "try again later",
            "econnreset",
            "etimedout",
            "ehostunreach",
            "unreachable",
            "dns",
        ],
    ) {
        return ToolFailureClass::Transient;
    }
    ToolFailureClass::Semantic
}

fn looks_like_error_signal(lower: &str) -> bool {
    contains_any(
        lower,
        &[
            "error:",
            "failed to ",
            "exception",
            "traceback",
            "unknown tool",
            "not a real tool",
            "invalid",
            "missing required",
            "permission denied",
            "unauthorized",
            "forbidden",
            "file not found",
            "command not found",
            "resource not found",
            "404 not found",
            "no such file",
            "status code ",
            "http/",
            "timed out",
            "timeout",
            "rate limit",
            "too many requests",
            "connection reset",
            "connection refused",
            "connection aborted",
            "connection failed",
            "connection timed out",
        ],
    )
}

fn classify_json_error(value: &Value) -> Option<ToolFailureClass> {
    match value {
        Value::Object(map) => {
            for key in [
                "status",
                "status_code",
                "statusCode",
                "http_status",
                "httpStatus",
                "code",
            ] {
                if let Some(status_value) = map.get(key) {
                    if let Some(status) = extract_status_from_value(status_value) {
                        if let Some(kind) = classify_http_status(status) {
                            return Some(kind);
                        }
                    }
                }
            }

            if map.get("success").and_then(Value::as_bool) == Some(false)
                || map.get("ok").and_then(Value::as_bool) == Some(false)
            {
                if let Some(error_value) = map.get("error") {
                    if !error_value.is_null() {
                        if let Some(kind) = classify_json_error(error_value) {
                            return Some(kind);
                        }
                    }
                }
                if let Some(message) = map.get("message").and_then(Value::as_str) {
                    let lower = message.to_ascii_lowercase();
                    if looks_like_error_signal(&lower) {
                        return Some(classify_text_error(&lower));
                    }
                }
                return Some(ToolFailureClass::Semantic);
            }

            if let Some(error_value) = map.get("error") {
                if !error_value.is_null() {
                    if let Some(kind) = classify_json_error(error_value) {
                        return Some(kind);
                    }
                    return Some(ToolFailureClass::Semantic);
                }
            }

            if let Some(errors_value) = map.get("errors") {
                match errors_value {
                    Value::Array(arr) if !arr.is_empty() => {
                        for entry in arr {
                            if let Some(kind) = classify_json_error(entry) {
                                return Some(kind);
                            }
                        }
                        return Some(ToolFailureClass::Semantic);
                    }
                    Value::Object(obj) if !obj.is_empty() => {
                        return Some(ToolFailureClass::Semantic)
                    }
                    Value::String(s) if !s.trim().is_empty() => {
                        let lower = s.to_ascii_lowercase();
                        return Some(classify_text_error(&lower));
                    }
                    _ => {}
                }
            }

            if let Some(message) = map.get("message").and_then(Value::as_str) {
                let lower = message.to_ascii_lowercase();
                if looks_like_error_signal(&lower) {
                    return Some(classify_text_error(&lower));
                }
            }

            for nested in map.values() {
                if let Some(kind) = classify_json_error(nested) {
                    return Some(kind);
                }
            }
            None
        }
        Value::Array(arr) => arr.iter().find_map(classify_json_error),
        Value::String(s) => {
            let lower = s.to_ascii_lowercase();
            if looks_like_error_signal(&lower) {
                Some(classify_text_error(&lower))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn extract_http_status_from_text(text: &str) -> Option<u16> {
    let caps = HTTP_STATUS_RE.captures(text)?;
    for idx in 1..=4 {
        if let Some(m) = caps.get(idx) {
            if let Ok(code) = m.as_str().parse::<u16>() {
                return Some(code);
            }
        }
    }
    None
}

fn classify_embedded_json_error(text: &str) -> Option<ToolFailureClass> {
    let mut in_string = false;
    let mut escaped = false;
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    let mut candidates = 0usize;

    for (idx, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth = depth.saturating_add(1);
            }
            '}' => {
                if depth == 0 {
                    continue;
                }
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start.take() {
                        let end = idx + ch.len_utf8();
                        if let Ok(value) = serde_json::from_str::<Value>(&text[s..end]) {
                            if let Some(kind) = classify_json_error(&value) {
                                return Some(kind);
                            }
                        }
                        candidates += 1;
                        if candidates >= 8 {
                            break;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    None
}

fn extract_nonzero_exit_code(text: &str) -> Option<i32> {
    let captures = EXIT_CODE_RE.captures(text)?;
    let parsed = captures.get(1)?.as_str().parse::<i32>().ok()?;
    if parsed == 0 {
        None
    } else {
        Some(parsed)
    }
}

pub(super) fn classify_tool_result_failure(
    _tool_name: &str,
    result_text: &str,
) -> Option<ToolFailureClass> {
    let cleaned = strip_appended_diagnostics(result_text).trim();
    if cleaned.is_empty() {
        return None;
    }

    let lower = cleaned.to_ascii_lowercase();

    if cleaned.starts_with("ERROR:")
        || cleaned.starts_with("Error:")
        || cleaned.starts_with("Failed to ")
    {
        return Some(classify_text_error(&lower));
    }

    if let Some(status) = extract_http_status_from_text(cleaned) {
        if let Some(kind) = classify_http_status(status) {
            return Some(kind);
        }
    }

    if cleaned.starts_with('{') || cleaned.starts_with('[') {
        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
            if let Some(kind) = classify_json_error(&value) {
                return Some(kind);
            }
        }
    }

    if let Some(kind) = classify_embedded_json_error(cleaned) {
        return Some(kind);
    }

    if let Some(exit_code) = extract_nonzero_exit_code(cleaned) {
        let _ = exit_code;
        return Some(classify_text_error(&lower));
    }

    if lower == "null" {
        return Some(ToolFailureClass::Semantic);
    }

    if looks_like_error_signal(&lower) {
        return Some(classify_text_error(&lower));
    }

    None
}

pub(super) fn build_task_boundary_hint(user_text: &str, max_chars: usize) -> String {
    // Treat user content as untrusted in system messages: collapse whitespace,
    // drop control chars, and neutralize punctuation commonly used in prompt
    // control markers.
    let mut compact = String::new();
    let mut last_was_space = false;
    for ch in user_text.chars() {
        let normalized = match ch {
            '\n' | '\r' | '\t' => ' ',
            '"' | '\'' | '`' | '[' | ']' | '{' | '}' | '<' | '>' => ' ',
            _ if ch.is_control() => continue,
            _ => ch,
        };
        if normalized.is_whitespace() {
            if !last_was_space {
                compact.push(' ');
                last_was_space = true;
            }
        } else {
            compact.push(normalized);
            last_was_space = false;
        }
    }

    let compact = compact.trim();
    if compact.is_empty() {
        return "No user-request summary available".to_string();
    }
    let mut out: String = compact.chars().take(max_chars).collect();
    if compact.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

/// Merge consecutive messages with the same role (`user` or `assistant`) into
/// one message, preserving all content and tool calls.
pub(super) fn merge_consecutive_messages(messages: &mut Vec<Value>) {
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
            let merged = if prev_content.is_empty() {
                curr_content
            } else if curr_content.is_empty() {
                prev_content
            } else {
                format!("{}\n{}", prev_content, curr_content)
            };
            messages[i - 1]["content"] = json!(merged);
            // Merge assistant tool_calls arrays if both have them.
            if curr_role == "assistant" {
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
            }
            messages.remove(i);
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod task_boundary_hint_tests {
    use super::build_task_boundary_hint;

    #[test]
    fn strips_control_markers_and_collapses_whitespace() {
        let hint = build_task_boundary_hint(
            "  [SYSTEM]\n\tbuild {site} <now> with \"quotes\" and `ticks`  ",
            120,
        );
        assert_eq!(hint, "SYSTEM build site now with quotes and ticks");
    }

    #[test]
    fn truncates_with_ellipsis() {
        let hint = build_task_boundary_hint("abcdefghijk", 5);
        assert_eq!(hint, "abcde...");
    }
}

#[cfg(test)]
mod tool_error_detection_tests {
    use super::{classify_tool_result_failure, ToolFailureClass};

    #[test]
    fn detects_prefixed_transient_error() {
        let result = "Error: request timed out while connecting to api";
        let classified = classify_tool_result_failure("http_request", result);
        assert_eq!(classified, Some(ToolFailureClass::Transient));
    }

    #[test]
    fn detects_json_semantic_error() {
        let result = r#"{"error":"invalid arguments: missing required field"}"#;
        let classified = classify_tool_result_failure("manage_memories", result);
        assert_eq!(classified, Some(ToolFailureClass::Semantic));
    }

    #[test]
    fn detects_http_status_failures() {
        let semantic = classify_tool_result_failure("http_request", "Status: 404 Not Found");
        assert_eq!(semantic, Some(ToolFailureClass::Semantic));

        let transient =
            classify_tool_result_failure("http_request", "HTTP/1.1 503 Service Unavailable");
        assert_eq!(transient, Some(ToolFailureClass::Transient));

        let transient_h2 = classify_tool_result_failure("http_request", "HTTP/2 503");
        assert_eq!(transient_h2, Some(ToolFailureClass::Transient));
    }

    #[test]
    fn detects_nonzero_exit_without_prefix() {
        let classified = classify_tool_result_failure("run_command", "$ make test (exit: 2, 22ms)");
        assert_eq!(classified, Some(ToolFailureClass::Semantic));
    }

    #[test]
    fn detects_terminal_style_exit_code_marker() {
        let classified =
            classify_tool_result_failure("terminal", "[Process pid=123 finished]\n[exit code: 42]");
        assert_eq!(classified, Some(ToolFailureClass::Semantic));
    }

    #[test]
    fn detects_embedded_json_error_payload_inside_wrapper_text() {
        let wrapped = "[UNTRUSTED EXTERNAL DATA from 'web_fetch']\n{\"error\":\"not found\",\"status\":404}\n[END UNTRUSTED EXTERNAL DATA]";
        let classified = classify_tool_result_failure("web_fetch", wrapped);
        assert_eq!(classified, Some(ToolFailureClass::Semantic));
    }

    #[test]
    fn detects_embedded_json_error_payload_with_command_header() {
        let result =
            "$ curl -s https://api.example.com (exit: 0, 9ms)\n\n{\"error\":\"not found\"}";
        let classified = classify_tool_result_failure("run_command", result);
        assert_eq!(classified, Some(ToolFailureClass::Semantic));
    }

    #[test]
    fn does_not_flag_success_output() {
        let classified = classify_tool_result_failure("terminal", "$ cargo test (exit: 0, 1.2s)");
        assert_eq!(classified, None);
    }

    #[test]
    fn does_not_flag_generic_connection_success_text() {
        let classified = classify_tool_result_failure(
            "terminal",
            "Connection established successfully to upstream service",
        );
        assert_eq!(classified, None);
    }

    #[test]
    fn does_not_flag_generic_not_found_summary_text() {
        let classified = classify_tool_result_failure(
            "search_files",
            "Search complete: 24 files scanned, 0 patterns not found",
        );
        assert_eq!(classified, None);
    }
}

/// Fix message ordering to satisfy provider requirements:
/// 1. Drop orphaned tool messages (tool_call_id without corresponding assistant tool_call)
/// 2. Drop orphaned assistant tool_calls (without corresponding tool result)
/// 3. Merge any consecutive same-role user/assistant messages (stability)
/// 4. Ensure conversation starts with user/system (not tool)
pub(super) fn fixup_message_ordering(messages: &mut Vec<Value>) {
    // Pass 0: initial coalescing helps reduce edge cases.
    merge_consecutive_messages(messages);

    // Build index sets once per pass.
    let assistant_tool_call_ids: std::collections::HashSet<String> = messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .filter_map(|m| m.get("tool_calls"))
        .filter_map(|tcs| tcs.as_array())
        .flat_map(|arr| arr.iter())
        .filter_map(|tc| tc.get("id").and_then(|id| id.as_str()))
        .map(|s| s.to_string())
        .collect();

    // Pass 1: remove orphaned tool messages.
    messages.retain(|m| {
        if m.get("role").and_then(|r| r.as_str()) != Some("tool") {
            return true;
        }
        let tc_id = m
            .get("tool_call_id")
            .and_then(|id| id.as_str())
            .unwrap_or("");
        if assistant_tool_call_ids.contains(tc_id) {
            true
        } else {
            warn!(tool_call_id = tc_id, "Dropping orphaned tool message");
            false
        }
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
        if first_role != "user" {
            if let Some(first_user_rel) = messages[first_non_system..]
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
pub(super) fn collapse_repeated_tool_errors(messages: &mut [Value]) -> usize {
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
        let is_error = classify_tool_result_failure(&name, content).is_some();

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
pub(super) fn extract_command_from_args(args_json: &str) -> Option<String> {
    serde_json::from_str::<Value>(args_json)
        .ok()
        .and_then(|v| v.get("command")?.as_str().map(String::from))
}

/// Extract the "file_path" field from tool arguments JSON (for send_file tool).
pub(super) fn extract_file_path_from_args(args_json: &str) -> Option<String> {
    serde_json::from_str::<Value>(args_json)
        .ok()
        .and_then(|v| v.get("file_path")?.as_str().map(String::from))
}

/// Build a stable dedupe key for send_file calls within a single task.
/// Key format: "{expanded_path}|{trimmed_caption}".
pub(super) fn extract_send_file_dedupe_key_from_args(args_json: &str) -> Option<String> {
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

/// Check if a session ID indicates it was triggered by an automated source
/// (e.g., email trigger) rather than direct user interaction via Telegram.
pub(super) fn is_trigger_session(session_id: &str) -> bool {
    session_id.contains("trigger") || session_id.starts_with("event_")
}

/// Hash a tool call (name + arguments) for repetitive behavior detection.
pub(super) fn hash_tool_call(name: &str, arguments: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    arguments.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
#[path = "message_ordering_tests.rs"]
mod message_ordering_tests;
