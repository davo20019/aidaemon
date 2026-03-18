//! Deterministic 1-line summaries for tool calls and their results.
//!
//! Used by the sliding window context manager to replace verbose tool results
//! with compact summaries in old conversation pairs, preserving context while
//! reducing token count.

// Used by message_build_phase.rs for age-based tool result summarization.

use serde_json::Value;

use crate::utils::truncate_str;

/// Maximum character length for the args portion of a terminal command summary.
const COMMAND_TRUNCATE_CHARS: usize = 80;

/// Generate a deterministic 1-line summary from a tool call and its result.
///
/// Format: `"tool_name: [args_summary] -> [outcome]"`
///
/// # Arguments
/// * `tool_name` — the name of the tool that was called
/// * `args_json` — the raw JSON string of tool arguments (may be empty or invalid)
/// * `result` — the raw result string returned by the tool
///
/// # Examples
/// ```ignore
/// // "terminal: cargo test -> exit 0"
/// // "http_request: GET api.example.com/data -> 200 OK"
/// // "read_file: src/main.rs -> 245 lines"
/// // "write_file: src/utils.rs -> ok"
/// ```
pub(super) fn summarize_tool_result(tool_name: &str, args_json: &str, result: &str) -> String {
    let args_summary = extract_args_summary(tool_name, args_json);
    let outcome = extract_outcome(tool_name, result);

    if args_summary.is_empty() {
        format!("{}: -> {}", tool_name, outcome)
    } else {
        format!("{}: {} -> {}", tool_name, args_summary, outcome)
    }
}

/// Extract a short args summary based on the tool type.
fn extract_args_summary(tool_name: &str, args_json: &str) -> String {
    let parsed: Value = match serde_json::from_str(args_json) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };

    match tool_name {
        "terminal" | "run_command" => parsed
            .get("command")
            .and_then(|v| v.as_str())
            .map(|cmd| truncate_str(cmd, COMMAND_TRUNCATE_CHARS))
            .unwrap_or_default(),

        "http_request" => {
            let method = parsed
                .get("method")
                .and_then(|v| v.as_str())
                .unwrap_or("GET");
            let url = parsed.get("url").and_then(|v| v.as_str()).unwrap_or("");
            if url.is_empty() {
                method.to_string()
            } else {
                format!("{} {}", method, url)
            }
        }

        "read_file" | "write_file" | "edit_file" => extract_path_field(&parsed),

        "web_search" => parsed
            .get("query")
            .and_then(|v| v.as_str())
            .map(|q| format!("'{}'", q))
            .unwrap_or_default(),

        "remember_fact" | "manage_memories" => parsed
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),

        _ => first_string_value(&parsed),
    }
}

/// Extract the file path from tool arguments, checking common field names.
fn extract_path_field(parsed: &Value) -> String {
    for key in ["path", "file_path", "file", "filename"] {
        if let Some(s) = parsed.get(key).and_then(|v| v.as_str()) {
            return s.to_string();
        }
    }
    String::new()
}

/// Extract the first string value from a JSON object's top-level fields.
fn first_string_value(parsed: &Value) -> String {
    if let Some(obj) = parsed.as_object() {
        for (_key, val) in obj {
            if let Some(s) = val.as_str() {
                return truncate_str(s, COMMAND_TRUNCATE_CHARS);
            }
        }
    }
    String::new()
}

/// Extract a short outcome description based on the tool type and result content.
fn extract_outcome(tool_name: &str, result: &str) -> String {
    match tool_name {
        "terminal" | "run_command" => extract_terminal_outcome(result),
        "http_request" => extract_http_outcome(result),
        "read_file" => extract_read_file_outcome(result),
        "write_file" | "edit_file" => extract_write_outcome(result),
        _ => "completed".to_string(),
    }
}

/// Look for `exit_code:` or `Exit code:` pattern in terminal output, format as `exit N`.
fn extract_terminal_outcome(result: &str) -> String {
    // Search from the end since exit codes tend to appear at the bottom.
    // We search in the lowercased line and use the position to index into the
    // lowercased copy (not the original), avoiding byte-offset mismatches that
    // can arise when `to_lowercase()` changes string length.
    for line in result.lines().rev() {
        let lower = line.to_lowercase();
        if let Some(pos) = lower
            .find("exit_code:")
            .or_else(|| lower.find("exit code:"))
        {
            let after_match = &lower[pos..];
            if let Some(code) = after_match
                .split(':')
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<i64>().ok())
            {
                return format!("exit {}", code);
            }
        }
    }
    "completed".to_string()
}

/// Extract HTTP status from the first line of the result if it looks like a status.
fn extract_http_outcome(result: &str) -> String {
    let first_line = result.lines().next().unwrap_or("");

    // Check for patterns like "200 OK", "HTTP/1.1 200 OK", "Status: 404"
    if first_line.contains("HTTP") || looks_like_http_status(first_line) {
        let trimmed = first_line.trim();
        // Keep it short — truncate long first lines
        return truncate_str(trimmed, 40);
    }

    "completed".to_string()
}

/// Check if a line looks like it contains an HTTP status code (3-digit number in 1xx-5xx range).
fn looks_like_http_status(line: &str) -> bool {
    line.split_whitespace().any(|word| {
        word.len() == 3
            && word
                .chars()
                .next()
                .is_some_and(|c| ('1'..='5').contains(&c))
            && word.chars().skip(1).all(|c| c.is_ascii_digit())
    })
}

/// Count lines in the result for read_file.
fn extract_read_file_outcome(result: &str) -> String {
    if result.is_empty() {
        return "empty".to_string();
    }
    let count = result.lines().count();
    if count == 1 {
        "1 line".to_string()
    } else {
        format!("{} lines", count)
    }
}

/// Check if write/edit succeeded based on result content.
fn extract_write_outcome(result: &str) -> String {
    let lower = result.to_lowercase();
    if lower.contains("successfully") || lower.contains("success") || lower.contains("written") {
        "ok".to_string()
    } else if lower.contains("error") || lower.contains("failed") || lower.contains("denied") {
        "error".to_string()
    } else {
        "completed".to_string()
    }
}

/// Calculate how many prior conversation pairs fit within a token budget.
///
/// Returns `min(5, pairs_that_fit_in_30%_of_budget)`.
/// Iterates from most recent (end of slice) backward, accumulating token estimates
/// until 30% of the available budget is exhausted.
///
/// Each element of `skeleton_pairs` is `(user_token_estimate, assistant_token_estimate)`.
pub(super) fn calculate_window_size(
    skeleton_pairs: &[(usize, usize)],
    available_budget: usize,
) -> usize {
    let budget_30pct = available_budget * 30 / 100;
    let mut used = 0usize;
    let mut count = 0usize;

    for &(user_tokens, assistant_tokens) in skeleton_pairs.iter().rev() {
        let pair_cost = user_tokens + assistant_tokens;
        if used + pair_cost > budget_30pct {
            break;
        }
        used += pair_cost;
        count += 1;
        if count >= 5 {
            break;
        }
    }

    count
}

/// Extract a skeleton from a sequence of messages belonging to one interaction.
///
/// The skeleton preserves signal while minimizing tokens:
/// - **User messages**: kept as-is (text content preserved).
/// - **Assistant messages with text**: text content preserved, `tool_calls` stripped.
/// - **Assistant messages without text** (tool-call-only): content replaced with
///   `"[Action completed]"` so there is always a response to the user message.
///   Dropping the message entirely would leave a dangling user message which can
///   trigger "completion compulsion" in models.
/// - **Tool messages** (`role: "tool"`): dropped entirely.
/// - **System messages**: dropped.
pub(super) fn extract_skeleton(messages: &[Value]) -> Vec<Value> {
    let mut result = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

        match role {
            "user" => {
                // Keep user messages as-is
                result.push(msg.clone());
            }
            "assistant" => {
                let has_text = msg
                    .get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| !s.trim().is_empty());

                if has_text {
                    // Keep text content, strip tool_calls
                    let mut skeleton = serde_json::json!({
                        "role": "assistant",
                        "content": msg.get("content").unwrap(),
                    });
                    // Preserve any extra fields like "name" but NOT "tool_calls"
                    if let Some(obj) = msg.as_object() {
                        for (key, val) in obj {
                            if key != "role" && key != "content" && key != "tool_calls" {
                                skeleton[key] = val.clone();
                            }
                        }
                    }
                    result.push(skeleton);
                } else {
                    // No text content (tool-call-only assistant message)
                    // Replace with placeholder to prevent dangling user messages
                    result.push(serde_json::json!({
                        "role": "assistant",
                        "content": "[Action completed]",
                    }));
                }
            }
            // Drop tool messages and system messages
            _ => {}
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_with_exit_code() {
        let result = summarize_tool_result(
            "terminal",
            r#"{"command": "cargo test"}"#,
            "running 5 tests...\ntest result: ok\nexit_code: 0",
        );
        assert_eq!(result, "terminal: cargo test -> exit 0");
    }

    #[test]
    fn test_terminal_nonzero_exit() {
        let result = summarize_tool_result(
            "terminal",
            r#"{"command": "cargo build"}"#,
            "error[E0308]: mismatched types\nExit code: 1",
        );
        assert_eq!(result, "terminal: cargo build -> exit 1");
    }

    #[test]
    fn test_terminal_no_exit_code() {
        let result = summarize_tool_result("terminal", r#"{"command": "echo hello"}"#, "hello");
        assert_eq!(result, "terminal: echo hello -> completed");
    }

    #[test]
    fn test_terminal_long_command_truncated() {
        let long_cmd = "a".repeat(200);
        let args = format!(r#"{{"command": "{}"}}"#, long_cmd);
        let result = summarize_tool_result("terminal", &args, "exit_code: 0");
        // The command portion should be truncated
        assert!(result.len() < 200);
        assert!(result.contains("..."));
        assert!(result.ends_with("-> exit 0"));
    }

    #[test]
    fn test_run_command_alias() {
        let result = summarize_tool_result(
            "run_command",
            r#"{"command": "ls -la"}"#,
            "total 42\nexit_code: 0",
        );
        assert_eq!(result, "run_command: ls -la -> exit 0");
    }

    #[test]
    fn test_http_request_with_status() {
        let result = summarize_tool_result(
            "http_request",
            r#"{"method": "GET", "url": "api.example.com/data"}"#,
            "HTTP/1.1 200 OK\nContent-Type: application/json",
        );
        assert_eq!(
            result,
            "http_request: GET api.example.com/data -> HTTP/1.1 200 OK"
        );
    }

    #[test]
    fn test_http_request_status_code_only() {
        let result = summarize_tool_result(
            "http_request",
            r#"{"method": "POST", "url": "api.example.com/submit"}"#,
            "200 OK",
        );
        assert_eq!(
            result,
            "http_request: POST api.example.com/submit -> 200 OK"
        );
    }

    #[test]
    fn test_http_request_no_status() {
        let result = summarize_tool_result(
            "http_request",
            r#"{"method": "GET", "url": "example.com"}"#,
            "{\"data\": [1, 2, 3]}",
        );
        assert_eq!(result, "http_request: GET example.com -> completed");
    }

    #[test]
    fn test_http_request_default_method() {
        let result = summarize_tool_result("http_request", r#"{"url": "example.com"}"#, "200 OK");
        assert_eq!(result, "http_request: GET example.com -> 200 OK");
    }

    #[test]
    fn test_read_file_line_count() {
        let content = "line 1\nline 2\nline 3\nline 4\nline 5";
        let result = summarize_tool_result("read_file", r#"{"path": "src/main.rs"}"#, content);
        assert_eq!(result, "read_file: src/main.rs -> 5 lines");
    }

    #[test]
    fn test_read_file_single_line() {
        let result = summarize_tool_result("read_file", r#"{"path": "VERSION"}"#, "1.0.0");
        assert_eq!(result, "read_file: VERSION -> 1 line");
    }

    #[test]
    fn test_read_file_empty() {
        let result = summarize_tool_result("read_file", r#"{"path": "empty.txt"}"#, "");
        assert_eq!(result, "read_file: empty.txt -> empty");
    }

    #[test]
    fn test_read_file_many_lines() {
        let content = (1..=245)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let result = summarize_tool_result("read_file", r#"{"path": "src/main.rs"}"#, &content);
        assert_eq!(result, "read_file: src/main.rs -> 245 lines");
    }

    #[test]
    fn test_write_file_success() {
        let result = summarize_tool_result(
            "write_file",
            r#"{"path": "src/utils.rs"}"#,
            "File written successfully",
        );
        assert_eq!(result, "write_file: src/utils.rs -> ok");
    }

    #[test]
    fn test_write_file_error() {
        let result = summarize_tool_result(
            "write_file",
            r#"{"path": "/etc/hosts"}"#,
            "Error: permission denied",
        );
        assert_eq!(result, "write_file: /etc/hosts -> error");
    }

    #[test]
    fn test_edit_file_success() {
        let result = summarize_tool_result(
            "edit_file",
            r#"{"path": "src/config.rs"}"#,
            "Edit applied successfully",
        );
        assert_eq!(result, "edit_file: src/config.rs -> ok");
    }

    #[test]
    fn test_edit_file_completed() {
        let result = summarize_tool_result(
            "edit_file",
            r#"{"path": "src/config.rs"}"#,
            "Changes applied.",
        );
        assert_eq!(result, "edit_file: src/config.rs -> completed");
    }

    #[test]
    fn test_web_search() {
        let result = summarize_tool_result(
            "web_search",
            r#"{"query": "skin cancer trials DC"}"#,
            "Found 10 results...",
        );
        assert_eq!(result, "web_search: 'skin cancer trials DC' -> completed");
    }

    #[test]
    fn test_remember_fact() {
        let result = summarize_tool_result(
            "remember_fact",
            r#"{"action": "store", "fact": "User likes coffee"}"#,
            "Fact stored successfully",
        );
        assert_eq!(result, "remember_fact: store -> completed");
    }

    #[test]
    fn test_manage_memories() {
        let result = summarize_tool_result(
            "manage_memories",
            r#"{"action": "search", "query": "coffee"}"#,
            "Found 3 matching facts",
        );
        assert_eq!(result, "manage_memories: search -> completed");
    }

    #[test]
    fn test_unknown_tool_with_string_arg() {
        let result =
            summarize_tool_result("custom_tool", r#"{"task": "do_thing", "count": 5}"#, "done");
        assert_eq!(result, "custom_tool: do_thing -> completed");
    }

    #[test]
    fn test_unknown_tool_no_string_args() {
        let result = summarize_tool_result("custom_tool", r#"{"count": 5, "flag": true}"#, "done");
        assert_eq!(result, "custom_tool: -> completed");
    }

    #[test]
    fn test_empty_args() {
        let result = summarize_tool_result("custom_tool", "", "done");
        assert_eq!(result, "custom_tool: -> completed");
    }

    #[test]
    fn test_invalid_json_args() {
        let result = summarize_tool_result("terminal", "not json", "exit_code: 0");
        assert_eq!(result, "terminal: -> exit 0");
    }

    #[test]
    fn test_file_path_alias() {
        // write_file using file_path instead of path
        let result = summarize_tool_result(
            "write_file",
            r#"{"file_path": "output.txt"}"#,
            "File written successfully",
        );
        assert_eq!(result, "write_file: output.txt -> ok");
    }

    #[test]
    fn test_utf8_safety() {
        // Ensure multi-byte characters don't cause panics
        let result = summarize_tool_result(
            "terminal",
            r#"{"command": "echo '🦀🦀🦀'"}"#,
            "🦀🦀🦀\nexit_code: 0",
        );
        assert!(result.contains("echo '🦀🦀🦀'"));
        assert!(result.ends_with("-> exit 0"));
    }

    #[test]
    fn test_exit_code_case_insensitive() {
        let result = summarize_tool_result(
            "terminal",
            r#"{"command": "make"}"#,
            "Build complete\nEXIT_CODE: 0",
        );
        assert_eq!(result, "terminal: make -> exit 0");
    }

    #[test]
    fn test_http_404_status() {
        let result = summarize_tool_result(
            "http_request",
            r#"{"method": "GET", "url": "example.com/missing"}"#,
            "404 Not Found",
        );
        assert_eq!(
            result,
            "http_request: GET example.com/missing -> 404 Not Found"
        );
    }

    #[test]
    fn test_looks_like_http_status_true() {
        assert!(looks_like_http_status("200 OK"));
        assert!(looks_like_http_status("404 Not Found"));
        assert!(looks_like_http_status("500 Internal Server Error"));
    }

    #[test]
    fn test_looks_like_http_status_false() {
        assert!(!looks_like_http_status("{\"data\": 123}"));
        assert!(!looks_like_http_status("no status here"));
        assert!(!looks_like_http_status("600 is not valid"));
        assert!(!looks_like_http_status("99 too short"));
    }

    // ─── calculate_window_size tests ───

    #[test]
    fn test_window_size_all_five_fit() {
        // 5 small pairs: each ~50 tokens. Budget 10000 → 30% = 3000. All fit easily.
        let pairs = vec![(25, 25); 5];
        assert_eq!(calculate_window_size(&pairs, 10000), 5);
    }

    #[test]
    fn test_window_size_budget_limited() {
        // 5 pairs, each ~500 tokens. Budget 4000 → 30% = 1200. Only 2 fit (1000 < 1200 < 1500).
        let pairs = vec![(250, 250); 5];
        assert_eq!(calculate_window_size(&pairs, 4000), 2);
    }

    #[test]
    fn test_window_size_tiny_budget_zero() {
        // Budget 100 → 30% = 30. Each pair is 200 tokens. None fit.
        let pairs = vec![(100, 100); 3];
        assert_eq!(calculate_window_size(&pairs, 100), 0);
    }

    #[test]
    fn test_window_size_empty_pairs() {
        assert_eq!(calculate_window_size(&[], 10000), 0);
    }

    #[test]
    fn test_window_size_large_budget_caps_at_five() {
        // 10 pairs, huge budget. Should cap at 5.
        let pairs = vec![(10, 10); 10];
        assert_eq!(calculate_window_size(&pairs, 1_000_000), 5);
    }

    #[test]
    fn test_window_size_exactly_at_boundary() {
        // 30% of 1000 = 300. 3 pairs of 100 each = 300. Should fit exactly 3.
        let pairs = vec![(50, 50); 5];
        assert_eq!(calculate_window_size(&pairs, 1000), 3);
    }

    #[test]
    fn test_window_size_just_over_boundary() {
        // 30% of 1000 = 300. 3 pairs: 101 each = 303 > 300. Only 2 fit.
        let pairs = vec![(51, 50); 5];
        assert_eq!(calculate_window_size(&pairs, 1000), 2);
    }

    // ─── extract_skeleton tests ───

    #[test]
    fn test_skeleton_strips_tool_calls_from_assistant() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "Hello"}),
            serde_json::json!({
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": [{"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}}]
            }),
        ];
        let skeleton = extract_skeleton(&messages);
        assert_eq!(skeleton.len(), 2);
        assert_eq!(skeleton[0]["role"], "user");
        assert_eq!(skeleton[0]["content"], "Hello");
        assert_eq!(skeleton[1]["role"], "assistant");
        assert_eq!(skeleton[1]["content"], "Let me check that.");
        assert!(skeleton[1].get("tool_calls").is_none());
    }

    #[test]
    fn test_skeleton_drops_tool_results() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "Read my file"}),
            serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}}]
            }),
            serde_json::json!({"role": "tool", "content": "file contents here...", "tool_call_id": "tc1", "name": "read_file"}),
            serde_json::json!({"role": "assistant", "content": "Here are the contents of your file."}),
        ];
        let skeleton = extract_skeleton(&messages);
        assert_eq!(skeleton.len(), 3); // user, [Action completed], assistant
                                       // No tool messages
        assert!(skeleton.iter().all(|m| m["role"] != "tool"));
    }

    #[test]
    fn test_skeleton_replaces_empty_assistant_with_action_completed() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "Write to file"}),
            serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "tc1", "function": {"name": "write_file", "arguments": "{}"}}]
            }),
        ];
        let skeleton = extract_skeleton(&messages);
        assert_eq!(skeleton.len(), 2);
        assert_eq!(skeleton[1]["role"], "assistant");
        assert_eq!(skeleton[1]["content"], "[Action completed]");
    }

    #[test]
    fn test_skeleton_replaces_empty_string_assistant_with_action_completed() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "Do something"}),
            serde_json::json!({
                "role": "assistant",
                "content": "   ",
                "tool_calls": [{"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}]
            }),
        ];
        let skeleton = extract_skeleton(&messages);
        assert_eq!(skeleton.len(), 2);
        assert_eq!(skeleton[1]["content"], "[Action completed]");
    }

    #[test]
    fn test_skeleton_drops_system_messages() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "You are helpful."}),
            serde_json::json!({"role": "user", "content": "Hello"}),
            serde_json::json!({"role": "assistant", "content": "Hi!"}),
        ];
        let skeleton = extract_skeleton(&messages);
        assert_eq!(skeleton.len(), 2);
        assert_eq!(skeleton[0]["role"], "user");
        assert_eq!(skeleton[1]["role"], "assistant");
    }

    #[test]
    fn test_skeleton_handles_multiple_interactions() {
        let messages = vec![
            // First interaction
            serde_json::json!({"role": "user", "content": "What time is it?"}),
            serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "tc1", "function": {"name": "terminal", "arguments": r#"{"command":"date"}"#}}]
            }),
            serde_json::json!({"role": "tool", "content": "Mon Mar 17 12:00:00 UTC 2025", "tool_call_id": "tc1", "name": "terminal"}),
            serde_json::json!({"role": "assistant", "content": "It's noon on March 17th."}),
            // Second interaction
            serde_json::json!({"role": "user", "content": "Write a greeting"}),
            serde_json::json!({
                "role": "assistant",
                "content": "I'll write that for you.",
                "tool_calls": [{"id": "tc2", "function": {"name": "write_file", "arguments": "{}"}}]
            }),
            serde_json::json!({"role": "tool", "content": "File written successfully", "tool_call_id": "tc2", "name": "write_file"}),
            serde_json::json!({"role": "assistant", "content": "Done! I wrote the greeting file."}),
        ];
        let skeleton = extract_skeleton(&messages);
        // Expected: user, [Action completed], assistant, user, assistant (stripped), assistant
        assert_eq!(skeleton.len(), 6);
        assert!(skeleton.iter().all(|m| m["role"] != "tool"));
        assert_eq!(skeleton[0]["content"], "What time is it?");
        assert_eq!(skeleton[1]["content"], "[Action completed]"); // tool-call-only assistant
        assert_eq!(skeleton[2]["content"], "It's noon on March 17th.");
        assert_eq!(skeleton[3]["content"], "Write a greeting");
        assert_eq!(skeleton[4]["content"], "I'll write that for you."); // tool_calls stripped
        assert!(skeleton[4].get("tool_calls").is_none());
        assert_eq!(skeleton[5]["content"], "Done! I wrote the greeting file.");
    }

    #[test]
    fn test_skeleton_empty_input() {
        let skeleton = extract_skeleton(&[]);
        assert!(skeleton.is_empty());
    }
}
