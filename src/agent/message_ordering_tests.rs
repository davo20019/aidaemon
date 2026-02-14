use super::*;
use serde_json::json;

/// Helper: assert no tool message appears without a matching assistant tool_call.
fn assert_no_orphaned_tools(messages: &[Value]) {
    let assistant_tc_ids: std::collections::HashSet<String> = messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .filter_map(|m| m.get("tool_calls"))
        .filter_map(|tcs| tcs.as_array())
        .flat_map(|arr| arr.iter())
        .filter_map(|tc| tc.get("id").and_then(|id| id.as_str()))
        .map(|s| s.to_string())
        .collect();

    for m in messages {
        if m.get("role").and_then(|r| r.as_str()) == Some("tool") {
            let tc_id = m
                .get("tool_call_id")
                .and_then(|id| id.as_str())
                .unwrap_or("");
            assert!(
                assistant_tc_ids.contains(tc_id),
                "Orphaned tool message: tool_call_id={} has no matching assistant tool_call",
                tc_id
            );
        }
    }
}

/// Helper: assert no assistant tool_call exists without a matching tool result.
fn assert_no_orphaned_tool_calls(messages: &[Value]) {
    let tool_result_ids: std::collections::HashSet<String> = messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
        .filter_map(|m| m.get("tool_call_id").and_then(|id| id.as_str()))
        .map(|s| s.to_string())
        .collect();

    for m in messages {
        if m.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }
        if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tcs {
                let id = tc.get("id").and_then(|id| id.as_str()).unwrap_or("");
                assert!(
                    tool_result_ids.contains(id),
                    "Orphaned tool_call: id={} has no matching tool result",
                    id
                );
            }
        }
    }
}

/// Helper: assert no consecutive same-role messages.
fn assert_no_consecutive_same_role(messages: &[Value]) {
    for i in 1..messages.len() {
        let prev = messages[i - 1]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        let curr = messages[i]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        if (curr == "assistant" || curr == "user") && prev == curr {
            panic!(
                "Consecutive same-role messages at index {}-{}: role={}",
                i - 1,
                i,
                curr
            );
        }
    }
}

/// Helper: assert the first non-system message is NOT a tool message.
fn assert_no_leading_tool(messages: &[Value]) {
    for m in messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "system" {
            continue;
        }
        assert_ne!(
            role, "tool",
            "First non-system message is a tool message (orphaned function_response)"
        );
        break;
    }
}

fn assert_all_invariants(messages: &[Value]) {
    assert_no_orphaned_tools(messages);
    assert_no_orphaned_tool_calls(messages);
    assert_no_consecutive_same_role(messages);
    assert_no_leading_tool(messages);
}

fn tc(id: &str, name: &str) -> Value {
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": "{}"}})
}

#[test]
fn test_clean_conversation_unchanged() {
    let mut msgs = vec![
        json!({"role": "user", "content": "hello"}),
        json!({"role": "assistant", "content": "I'll check", "tool_calls": [tc("c1", "terminal")]}),
        json!({"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "ok"}),
        json!({"role": "assistant", "content": "Done"}),
    ];
    fixup_message_ordering(&mut msgs);
    assert_eq!(msgs.len(), 4);
    assert_all_invariants(&msgs);
}

#[test]
fn test_orphaned_tool_at_start_of_window() {
    // Context window starts with tool result whose assistant is outside window.
    let mut msgs = vec![
        json!({"role": "tool", "tool_call_id": "c0", "name": "terminal", "content": "old result"}),
        json!({"role": "assistant", "content": "noted"}),
        json!({"role": "user", "content": "hello"}),
        json!({"role": "assistant", "content": "hi"}),
    ];
    fixup_message_ordering(&mut msgs);
    assert_all_invariants(&msgs);
    // The orphaned tool should be gone
    assert!(msgs
        .iter()
        .all(|m| m.get("role").and_then(|r| r.as_str()) != Some("tool")));
}

#[test]
fn test_two_orphaned_tools_at_start() {
    let mut msgs = vec![
        json!({"role": "tool", "tool_call_id": "c0", "name": "terminal", "content": "r0"}),
        json!({"role": "tool", "tool_call_id": "c1", "name": "browser", "content": "r1"}),
        json!({"role": "assistant", "content": "summary of prev"}),
        json!({"role": "user", "content": "next question"}),
        json!({"role": "assistant", "content": "answer"}),
    ];
    fixup_message_ordering(&mut msgs);
    assert_all_invariants(&msgs);
}

#[test]
fn test_orphan_drop_creates_consecutive_assistants() {
    // assistant A → tool(orphaned) → assistant B → user
    // After dropping tool, assistant A and B are consecutive → must merge.
    let mut msgs = vec![
        json!({"role": "assistant", "content": "step 1", "tool_calls": [tc("c1", "terminal")]}),
        json!({"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "result"}),
        json!({"role": "assistant", "content": "step 2", "tool_calls": [tc("c2", "browser")]}),
        // c2 tool result is missing (outside window)
        json!({"role": "user", "content": "ok"}),
        json!({"role": "assistant", "content": "done"}),
    ];
    fixup_message_ordering(&mut msgs);
    assert_all_invariants(&msgs);
}

#[test]
fn test_multiple_tool_calls_partial_orphan() {
    // Assistant has 2 tool_calls, only 1 has a result in context.
    let mut msgs = vec![
        json!({"role": "user", "content": "do stuff"}),
        json!({"role": "assistant", "content": "ok", "tool_calls": [tc("c1", "terminal"), tc("c2", "browser")]}),
        json!({"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "result1"}),
        // c2 result missing
        json!({"role": "assistant", "content": "done"}),
    ];
    fixup_message_ordering(&mut msgs);
    assert_all_invariants(&msgs);
    // c2 should be stripped from tool_calls but c1 kept
    let assistant_tc = &msgs[1];
    let tcs = assistant_tc.get("tool_calls").unwrap().as_array().unwrap();
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0]["id"], "c1");
}

#[test]
fn test_long_agentic_loop_context_window() {
    // Simulates 10 iterations with a 20-message window.
    // First few iterations' messages are outside the window.
    let mut msgs = vec![];
    // Messages 0-19 from a long conversation — window starts mid-conversation.
    // Old orphaned tool:
    msgs.push(
        json!({"role": "tool", "tool_call_id": "old_c1", "name": "terminal", "content": "old"}),
    );
    // Old assistant final response:
    msgs.push(json!({"role": "assistant", "content": "done with prev task"}));
    // New user message:
    msgs.push(json!({"role": "user", "content": "new task"}));
    // 5 iterations of assistant→tool pairs:
    for i in 0..5 {
        let cid = format!("iter_{}", i);
        msgs.push(json!({"role": "assistant", "content": format!("step {}", i), "tool_calls": [tc(&cid, "terminal")]}));
        msgs.push(json!({"role": "tool", "tool_call_id": cid, "name": "terminal", "content": format!("result {}", i)}));
    }

    fixup_message_ordering(&mut msgs);
    assert_all_invariants(&msgs);
}

#[test]
fn test_assistant_with_null_content_and_tool_calls() {
    let mut msgs = vec![
        json!({"role": "user", "content": "go"}),
        json!({"role": "assistant", "content": null, "tool_calls": [tc("c1", "write_file")]}),
        json!({"role": "tool", "tool_call_id": "c1", "name": "write_file", "content": "ok"}),
        json!({"role": "assistant", "content": "done"}),
    ];
    fixup_message_ordering(&mut msgs);
    assert_all_invariants(&msgs);
    assert_eq!(msgs.len(), 4);
}

#[test]
fn test_merge_combines_tool_calls() {
    // Two consecutive assistants with different tool_calls → merge should combine.
    let mut msgs = vec![
        json!({"role": "assistant", "content": "a", "tool_calls": [tc("c1", "t1")]}),
        json!({"role": "assistant", "content": "b", "tool_calls": [tc("c2", "t2")]}),
    ];
    merge_consecutive_messages(&mut msgs);
    assert_eq!(msgs.len(), 1);
    let tcs = msgs[0].get("tool_calls").unwrap().as_array().unwrap();
    assert_eq!(tcs.len(), 2);
}
