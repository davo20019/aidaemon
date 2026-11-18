//! Phase 0 observability — deterministic prefix fingerprinting for the
//! sliding-window cache-reuse investigation.
//!
//! The llama.cpp prefix cache only reuses a prompt prefix while the serialized
//! bytes are byte-identical to a prior request. This module computes stable
//! SHA-256 fingerprints over canonical JSON so the daemon log can attribute a
//! cache break to the *region* of the prompt that changed (message zero, the
//! pre-boundary history region, the tool definitions, or the session summary)
//! without ever emitting raw message content.
//!
//! The single source of truth is `canonical_prefix`: it is used both by the
//! provider-call fingerprint in `mandate_run` and by the hash unit tests, so the
//! tests exercise exactly the code path that runs in production.
//!
//! Canonicalization recursively sorts object keys so that two semantically
//! identical messages with differently-ordered keys hash the same. Hashes are
//! computed over the **complete** message objects (including `tool_calls`,
//! `name`, and `tool_call_id`), so a change to any of those fields flips the
//! hash even when `role` and `content` are unchanged.

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Marker prefix for the task context tail message. SHARED between the tail
/// builder (`system_prompt.rs`) and this module so the constant never drifts.
/// The tail builder re-exports this constant; do not duplicate the literal.
pub const TASK_CONTEXT_TAIL_MARKER: &str = "[Task Context]";

/// Index of the last user-role message whose content equals `user_text`.
///
/// This is the prompt boundary: everything before it is reusable history,
/// everything at/after it is the current interaction (current user message +
/// its tool chain + checkpoint). Falls back to `messages.len()` when no
/// matching user message is present, so the pre-boundary region degrades to
/// "everything after message zero" rather than panicking.
pub fn boundary_pos(messages: &[Value], user_text: &str) -> usize {
    find_current_user_boundary(messages, user_text).unwrap_or(messages.len())
}

fn find_current_user_boundary(messages: &[Value], user_text: &str) -> Option<usize> {
    messages.iter().enumerate().rev().find_map(|(i, m)| {
        if m.get("role").and_then(|r| r.as_str()) == Some("user")
            && m.get("content").is_some_and(|content| {
                crate::message_content::user_message_content_matches(content, user_text)
            })
        {
            Some(i)
        } else {
            None
        }
    })
}

/// Hash the pre-boundary history region of an intermediate build stage.
///
/// Skips a leading system-role message when present so the value is comparable
/// across stages that run before vs. after system-prompt insertion (and to the
/// provider-call `hash_pre_boundary`). The attribution value is in comparing a
/// single stage's hash across consecutive builds — a stage whose hash flips
/// while `keep_from` is stable is content mutation, not window-trim movement.
pub fn stage_pre_boundary_hash(messages: &[Value], user_text: &str) -> String {
    let boundary = boundary_pos(messages, user_text);
    let skip = usize::from(
        messages
            .first()
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            == Some("system"),
    );
    let region: Vec<Value> = messages.iter().take(boundary).skip(skip).cloned().collect();
    hash_canonical(&Value::Array(region))
}

/// SHA-256 hex digest over the canonical (recursively key-sorted) JSON form of
/// `value`. Exposed for the call site to fingerprint tool definitions and the
/// session summary as separate region hashes.
pub fn hash_canonical(value: &Value) -> String {
    let mut canonical = String::new();
    write_canonical(value, &mut canonical);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    hex::encode(hasher.finalize())
}

/// Serialize `value` into a deterministic string form with all object keys
/// recursively sorted. Independent of serde_json's `preserve_order` feature so
/// the hash is stable regardless of build configuration.
fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*key).clone()).to_string());
                out.push(':');
                write_canonical(&map[*key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stage_hash_skips_leading_system_message() {
        // Same history, one stage has a system prompt at index 0, the other
        // does not. The pre-boundary region (history before the boundary)
        // hashes the same because the leading system message is skipped.
        let with_system = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "old"}),
            json!({"role": "assistant", "content": "ans"}),
            json!({"role": "user", "content": "current question"}),
        ];
        let without_system = vec![
            json!({"role": "user", "content": "old"}),
            json!({"role": "assistant", "content": "ans"}),
            json!({"role": "user", "content": "current question"}),
        ];
        assert_eq!(
            stage_pre_boundary_hash(&with_system, "current question"),
            stage_pre_boundary_hash(&without_system, "current question"),
        );
    }

    #[test]
    fn stage_hash_changes_when_pre_boundary_content_mutates() {
        let base = vec![
            json!({"role": "user", "content": "old"}),
            json!({"role": "assistant", "content": "ans"}),
            json!({"role": "user", "content": "current question"}),
        ];
        let mut mutated = base.clone();
        mutated[1] = json!({"role": "assistant", "content": "ans (truncated…)"});
        assert_ne!(
            stage_pre_boundary_hash(&base, "current question"),
            stage_pre_boundary_hash(&mutated, "current question"),
        );
    }

    #[test]
    fn hash_canonical_is_order_independent_for_nested_objects() {
        let a = json!({"outer": {"b": 1, "a": 2}, "list": [{"y": 1, "x": 2}]});
        let b = json!({"list": [{"x": 2, "y": 1}], "outer": {"a": 2, "b": 1}});
        assert_eq!(hash_canonical(&a), hash_canonical(&b));
    }
}
