//! Deterministic 1-line summaries for tool calls and their results.
//!
//! Used by the sliding window context manager to replace verbose tool results
//! with compact summaries in old conversation pairs, preserving context while
//! reducing token count.

// Not yet wired into the message build phase — will be used by a later task.
#![allow(dead_code)]

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
        if let Some(pos) = lower.find("exit_code:").or_else(|| lower.find("exit code:")) {
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
        let result = summarize_tool_result(
            "terminal",
            r#"{"command": "echo hello"}"#,
            "hello",
        );
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
        let result = summarize_tool_result(
            "http_request",
            r#"{"url": "example.com"}"#,
            "200 OK",
        );
        assert_eq!(result, "http_request: GET example.com -> 200 OK");
    }

    #[test]
    fn test_read_file_line_count() {
        let content = "line 1\nline 2\nline 3\nline 4\nline 5";
        let result = summarize_tool_result(
            "read_file",
            r#"{"path": "src/main.rs"}"#,
            content,
        );
        assert_eq!(result, "read_file: src/main.rs -> 5 lines");
    }

    #[test]
    fn test_read_file_single_line() {
        let result = summarize_tool_result(
            "read_file",
            r#"{"path": "VERSION"}"#,
            "1.0.0",
        );
        assert_eq!(result, "read_file: VERSION -> 1 line");
    }

    #[test]
    fn test_read_file_empty() {
        let result = summarize_tool_result(
            "read_file",
            r#"{"path": "empty.txt"}"#,
            "",
        );
        assert_eq!(result, "read_file: empty.txt -> empty");
    }

    #[test]
    fn test_read_file_many_lines() {
        let content = (1..=245).map(|i| format!("line {}", i)).collect::<Vec<_>>().join("\n");
        let result = summarize_tool_result(
            "read_file",
            r#"{"path": "src/main.rs"}"#,
            &content,
        );
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
        let result = summarize_tool_result(
            "custom_tool",
            r#"{"task": "do_thing", "count": 5}"#,
            "done",
        );
        assert_eq!(result, "custom_tool: do_thing -> completed");
    }

    #[test]
    fn test_unknown_tool_no_string_args() {
        let result = summarize_tool_result(
            "custom_tool",
            r#"{"count": 5, "flag": true}"#,
            "done",
        );
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
        assert_eq!(result, "http_request: GET example.com/missing -> 404 Not Found");
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
}
