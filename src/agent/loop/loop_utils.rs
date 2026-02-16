use std::collections::HashMap;
use std::hash::{Hash, Hasher};

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

fn tool_content_is_error_prefix(s: &str) -> bool {
    s.starts_with("ERROR:") || s.starts_with("Error:") || s.starts_with("Failed to ")
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
        let is_error = tool_content_is_error_prefix(content);

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
