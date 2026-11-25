//! Task outcome derivation from completion evidence at the `emit_task_end` boundary.

use serde_json::Value;

/// Why a task terminated before natural completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Variants document persisted terminal causes before every caller is migrated.
pub enum TaskTerminalCause {
    Cancelled,
    Timeout,
    Watchdog,
    HardFailure,
    UnrecoveredModelFailure,
}

/// Detect model outputs that tried to write a tool call as plain text instead
/// of returning a structured provider tool call.
pub fn response_looks_like_plain_text_tool_call(reply: &str) -> bool {
    let trimmed = reply.trim();
    if trimmed.is_empty() {
        return false;
    }

    let candidate = strip_single_code_fence(trimmed).unwrap_or(trimmed);
    let lower = candidate.to_ascii_lowercase();
    if lower.starts_with("<|tool_call")
        || lower.starts_with("[tool_call")
        || lower.starts_with("<tool_call")
        || lower.starts_with("tool_call:")
        || (lower.starts_with("call:") && lower.contains('{'))
    {
        return true;
    }

    match serde_json::from_str::<Value>(candidate) {
        Ok(value) => looks_like_tool_call_json(&value),
        Err(_) => false,
    }
}

fn strip_single_code_fence(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    let rest = trimmed.strip_prefix("```")?;
    let body_start = rest.find('\n').map(|idx| idx + 1).unwrap_or(0);
    let body = &rest[body_start..];
    let body = body.strip_suffix("```")?;
    Some(body.trim())
}

fn looks_like_tool_call_json(value: &Value) -> bool {
    if let Some(array) = value.as_array() {
        return array.iter().any(looks_like_tool_call_json);
    }

    let Some(obj) = value.as_object() else {
        return false;
    };

    if obj.get("tool_calls").and_then(Value::as_array).is_some() {
        return true;
    }

    let has_arguments = obj.contains_key("arguments")
        || obj.contains_key("args")
        || obj.contains_key("input")
        || obj.contains_key("parameters");
    if (obj.get("name").and_then(Value::as_str).is_some()
        || obj.get("tool").and_then(Value::as_str).is_some()
        || obj.get("recipient_name").and_then(Value::as_str).is_some()
        || obj.get("toolName").and_then(Value::as_str).is_some())
        && has_arguments
    {
        return true;
    }

    obj.get("function")
        .and_then(Value::as_object)
        .or_else(|| obj.get("function_call").and_then(Value::as_object))
        .is_some_and(|function| {
            function.get("name").and_then(Value::as_str).is_some()
                && function
                    .get("arguments")
                    .is_some_and(|arguments| !arguments.is_null())
        })
}
