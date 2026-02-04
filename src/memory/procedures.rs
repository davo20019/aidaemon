//! Procedural memory module for learning and recalling action sequences.
//!
//! This module handles:
//! - Learning successful action sequences from task completions
//! - Generalizing procedures to be reusable
//! - Error-solution pair learning
#![allow(dead_code)] // Reserved for future procedural learning feature

use regex::Regex;
use crate::traits::{ErrorSolution, Message, Procedure};
use chrono::Utc;
use std::sync::OnceLock;

// Pre-compiled regexes for better performance
fn path_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"/[\w/.-]+").unwrap())
}

fn url_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"https?://[^\s)]+").unwrap())
}

fn line_num_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r":\d+:\d+").unwrap())
}

/// Extract action sequence from a conversation (tool calls made).
pub fn extract_action_sequence(messages: &[Message]) -> Vec<String> {
    let mut actions = Vec::new();

    for msg in messages {
        if let Some(tc_json) = &msg.tool_calls_json {
            if let Ok(tool_calls) = serde_json::from_str::<Vec<serde_json::Value>>(tc_json) {
                for tc in tool_calls {
                    if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
                        // Include tool name and a summary of arguments
                        let args = tc.get("arguments").and_then(|a| a.as_str()).unwrap_or("{}");
                        let summary = summarize_tool_args(name, args);
                        actions.push(format!("{}({})", name, summary));
                    }
                }
            }
        }
    }

    actions
}

/// Summarize tool arguments into a brief representation.
fn summarize_tool_args(tool_name: &str, args_json: &str) -> String {
    let args: serde_json::Value = serde_json::from_str(args_json).unwrap_or(serde_json::json!({}));

    match tool_name {
        "terminal" => {
            args.get("command")
                .and_then(|c| c.as_str())
                .map(|c| {
                    // Extract command name only
                    c.split_whitespace().next().unwrap_or("cmd").to_string()
                })
                .unwrap_or_else(|| "cmd".to_string())
        }
        "remember_fact" => {
            args.get("category")
                .and_then(|c| c.as_str())
                .unwrap_or("fact")
                .to_string()
        }
        "web_search" => {
            args.get("query")
                .and_then(|q| q.as_str())
                .map(|q| {
                    let words: Vec<&str> = q.split_whitespace().take(2).collect();
                    words.join(" ")
                })
                .unwrap_or_else(|| "query".to_string())
        }
        "web_fetch" => "url".to_string(),
        "browser" => {
            args.get("action")
                .and_then(|a| a.as_str())
                .unwrap_or("action")
                .to_string()
        }
        _ => "...".to_string(),
    }
}

/// Generalize a procedure by replacing specific values with placeholders.
pub fn generalize_procedure(actions: &[String]) -> Vec<String> {
    actions
        .iter()
        .map(|action| {
            // Replace specific paths with <path>
            let generalized = path_regex().replace_all(action, "<path>").to_string();

            // Replace URLs with <url>
            let generalized = url_regex().replace_all(&generalized, "<url>").to_string();

            generalized
        })
        .collect()
}

/// Generate a procedure name from the task context using keyword extraction.
pub fn generate_procedure_name(task_context: &str) -> String {
    let lower = task_context.to_lowercase();

    // Common task patterns
    if lower.contains("build") && lower.contains("rust") {
        return "rust-build".to_string();
    }
    if lower.contains("test") {
        return "run-tests".to_string();
    }
    if lower.contains("deploy") {
        return "deploy".to_string();
    }
    if lower.contains("debug") || lower.contains("fix") {
        return "debug-fix".to_string();
    }
    if lower.contains("search") || lower.contains("find") {
        return "search".to_string();
    }
    if lower.contains("install") || lower.contains("setup") {
        return "setup".to_string();
    }
    if lower.contains("git") {
        if lower.contains("commit") {
            return "git-commit".to_string();
        }
        if lower.contains("push") {
            return "git-push".to_string();
        }
        return "git-workflow".to_string();
    }

    // Default: extract first verb-like word
    let words: Vec<&str> = task_context.split_whitespace().take(3).collect();
    words.join("-").to_lowercase()
}

/// Extract trigger pattern from task context.
pub fn extract_trigger_pattern(task_context: &str) -> String {
    // Take the first sentence or first 100 chars as the trigger
    let first_sentence = task_context
        .split('.')
        .next()
        .unwrap_or(task_context)
        .trim();

    if first_sentence.len() > 100 {
        first_sentence[..100].to_string()
    } else {
        first_sentence.to_string()
    }
}

/// Create a new Procedure from task context.
pub fn create_procedure(
    name: String,
    trigger_pattern: String,
    steps: Vec<String>,
) -> Procedure {
    let now = Utc::now();
    Procedure {
        id: 0, // Will be set by database
        name,
        trigger_pattern,
        steps,
        success_count: 1,
        failure_count: 0,
        avg_duration_secs: None,
        last_used_at: Some(now),
        created_at: now,
        updated_at: now,
    }
}

/// Extract error pattern from an error message.
pub fn extract_error_pattern(error: &str) -> String {
    // Remove line numbers and specific paths
    let pattern = line_num_regex().replace_all(error, ":<line>").to_string();
    let pattern = path_regex().replace_all(&pattern, "<path>").to_string();

    // Truncate to first 200 chars
    if pattern.len() > 200 {
        pattern[..200].to_string()
    } else {
        pattern
    }
}

/// Summarize solution actions into a brief description.
pub fn summarize_solution(actions: &[String]) -> String {
    if actions.is_empty() {
        return "No specific actions recorded".to_string();
    }

    let unique_tools: Vec<&str> = actions
        .iter()
        .filter_map(|a| a.split('(').next())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    format!(
        "Used {} tool(s): {}",
        unique_tools.len(),
        unique_tools.join(", ")
    )
}

/// Create a new ErrorSolution.
pub fn create_error_solution(
    error_pattern: String,
    domain: Option<String>,
    solution_summary: String,
    solution_steps: Option<Vec<String>>,
) -> ErrorSolution {
    let now = Utc::now();
    ErrorSolution {
        id: 0, // Will be set by database
        error_pattern,
        domain,
        solution_summary,
        solution_steps,
        success_count: 1,
        failure_count: 0,
        last_used_at: Some(now),
        created_at: now,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_error_pattern() {
        let error = "error[E0382]: borrow of moved value at /home/user/project/src/main.rs:42:10";
        let pattern = extract_error_pattern(error);
        assert!(pattern.contains("E0382"));
        assert!(pattern.contains("<path>"));
        assert!(pattern.contains("<line>"));
    }

    #[test]
    fn test_generate_procedure_name() {
        assert_eq!(generate_procedure_name("Build the Rust project"), "rust-build");
        assert_eq!(generate_procedure_name("Run the tests"), "run-tests");
        assert_eq!(generate_procedure_name("Deploy to production"), "deploy");
    }

    #[test]
    fn test_generalize_procedure() {
        let actions = vec![
            "terminal(cargo)".to_string(),
            "web_fetch(/home/user/file.rs)".to_string(),
        ];
        let generalized = generalize_procedure(&actions);
        assert!(generalized[1].contains("<path>"));
    }
}
