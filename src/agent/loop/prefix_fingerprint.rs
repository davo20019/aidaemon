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
//! The single source of truth is [`canonical_prefix`]: it is used both by the
//! provider-call fingerprint in `llm_phase` and by the hash unit tests, so the
//! tests exercise exactly the code path that runs in production.
//!
//! Canonicalization recursively sorts object keys so that two semantically
//! identical messages with differently-ordered keys hash the same. Hashes are
//! computed over the **complete** message objects (including `tool_calls`,
//! `name`, and `tool_call_id`), so a change to any of those fields flips the
//! hash even when `role` and `content` are unchanged.

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Compute the region fingerprints for a finalized provider message payload.
///
/// Returns `(hash_system, hash_pre_boundary, boundary_pos)`:
/// - `hash_system` — hash of message zero alone (the system prompt). Must be
///   constant across within-task consecutive calls once the cache-stable
///   system-prompt work is in place.
/// - `hash_pre_boundary` — hash of the complete message objects in
///   `[1..boundary)`. The current interaction's tool chain sits at/after the
///   boundary, so ordinary per-iteration tail growth does not flip this hash.
/// - `boundary_pos` — the index of the last user-role message whose `content`
///   equals `user_text`; if no such message exists, `messages.len()`.
///
/// Hashes never include raw message content in their output — only the hex
/// digest is returned.
pub(crate) fn canonical_prefix(messages: &[Value], user_text: &str) -> (String, String, usize) {
    let boundary = boundary_pos(messages, user_text);

    let hash_system = match messages.first() {
        Some(m) => hash_canonical(m),
        None => hash_canonical(&Value::Null),
    };

    let pre_region: Vec<Value> = messages.iter().take(boundary).skip(1).cloned().collect();
    let hash_pre_boundary = hash_canonical(&Value::Array(pre_region));

    (hash_system, hash_pre_boundary, boundary)
}

/// All region fingerprints for one finalized provider call, emitted as a
/// single structured `info!` line in `llm_phase` immediately before the
/// provider request. Hashes never carry raw content; only digests are logged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProviderCallFingerprint {
    /// Hash of message zero (the system prompt).
    pub hash_system: String,
    /// Hash of the complete pre-boundary message objects in `[1..boundary)`.
    pub hash_pre_boundary: String,
    /// Index of the boundary (last user message matching `user_text`).
    pub boundary_pos: usize,
    /// Total number of messages in the payload.
    pub message_count: usize,
    /// Hash of the effective tool definitions passed to this attempt, in the
    /// order sent. Empty string only when the call genuinely carries no
    /// tools. Force-text calls retain the tool definitions (calling is
    /// disabled via tool_choice=none), so their hash is computed normally —
    /// stability across a force-text turn is part of the signal.
    pub tool_defs_hash: String,
    /// Hash of the `[Session Summary]` message at index one, if present;
    /// empty string otherwise. Surfaced separately so index-one churn is
    /// immediately visible.
    pub session_summary_hash: String,
    /// Marks force-text iterations (plain-text required; tool calling
    /// disabled via tool_choice=none while tool definitions stay in the
    /// payload). The mode marker for attribution; it does not affect any
    /// hash in this struct.
    pub force_text: bool,
}

/// Marker prefix for the message-build session summary (see
/// `message_build_phase.rs`). Used to locate the index-one summary so its
/// churn can be hashed independently of the rest of the pre-boundary region.
const SESSION_SUMMARY_MARKER: &str = "[Session Summary]";

/// Build the [`ProviderCallFingerprint`] for the final provider payload.
///
/// `effective_tools` is the actual tool set passed to the provider. In
/// force-text mode the definitions are *retained* (see `effective_tools_for_call`
/// in `llm_phase.rs`: tool defs stay in the payload for prefix-cache stability;
/// only calling is disabled via `tool_choice=none`), so `tool_defs_hash` stays
/// stable across a force-text turn. The hash is empty only when
/// `effective_tools` is genuinely empty. The tool hash preserves the array
/// order so that a reordering (a genuine cache break) is observable; it is
/// *not* name-sorted here (that name-sorted form is the Phase 1 validity hash,
/// a different concern). `force_text` is recorded as a fingerprint tag, not
/// used to alter any hash.
pub(crate) fn provider_call_fingerprint(
    messages: &[Value],
    user_text: &str,
    effective_tools: &[Value],
    force_text: bool,
) -> ProviderCallFingerprint {
    let (hash_system, hash_pre_boundary, boundary_pos) = canonical_prefix(messages, user_text);

    let tool_defs_hash = if effective_tools.is_empty() {
        String::new()
    } else {
        hash_canonical(&Value::Array(effective_tools.to_vec()))
    };

    let session_summary_hash = messages
        .iter()
        .find(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| s.starts_with(SESSION_SUMMARY_MARKER))
        })
        .map(hash_canonical)
        .unwrap_or_default();

    ProviderCallFingerprint {
        hash_system,
        hash_pre_boundary,
        boundary_pos,
        message_count: messages.len(),
        tool_defs_hash,
        session_summary_hash,
        force_text,
    }
}

/// Index of the last user-role message whose content equals `user_text`.
///
/// This is the prompt boundary: everything before it is reusable history,
/// everything at/after it is the current interaction (current user message +
/// its tool chain + checkpoint). Falls back to `messages.len()` when no
/// matching user message is present, so the pre-boundary region degrades to
/// "everything after message zero" rather than panicking.
pub(crate) fn boundary_pos(messages: &[Value], user_text: &str) -> usize {
    messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, m)| {
            m.get("role").and_then(|r| r.as_str()) == Some("user")
                && m.get("content").and_then(|c| c.as_str()) == Some(user_text)
        })
        .map(|(i, _)| i)
        .unwrap_or(messages.len())
}

/// Hash the pre-boundary history region of an intermediate build stage.
///
/// Skips a leading system-role message when present so the value is comparable
/// across stages that run before vs. after system-prompt insertion (and to the
/// provider-call `hash_pre_boundary`). The attribution value is in comparing a
/// single stage's hash across consecutive builds — a stage whose hash flips
/// while `keep_from` is stable is content mutation, not window-trim movement.
pub(crate) fn stage_pre_boundary_hash(messages: &[Value], user_text: &str) -> String {
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
pub(crate) fn hash_canonical(value: &Value) -> String {
    let mut canonical = String::new();
    write_canonical(value, &mut canonical);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
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

    fn sample_messages() -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "You are a helpful assistant."}),
            json!({"role": "user", "content": "old question"}),
            json!({"role": "assistant", "content": "old answer"}),
            json!({"role": "user", "content": "current question"}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}}]
            }),
            json!({"role": "tool", "content": "file body", "tool_call_id": "tc1", "name": "read_file"}),
        ]
    }

    #[test]
    fn boundary_is_last_user_message_matching_user_text() {
        let messages = sample_messages();
        let (_, _, boundary) = canonical_prefix(&messages, "current question");
        // Index 3 is the current user message; the tool chain at 4/5 is excluded.
        assert_eq!(boundary, 3);
    }

    #[test]
    fn boundary_falls_back_to_len_when_user_text_absent() {
        let messages = sample_messages();
        let (_, _, boundary) = canonical_prefix(&messages, "no such message");
        assert_eq!(boundary, messages.len());
    }

    #[test]
    fn identical_inputs_produce_identical_hashes() {
        let messages = sample_messages();
        let first = canonical_prefix(&messages, "current question");
        let second = canonical_prefix(&messages.clone(), "current question");
        assert_eq!(first, second);
    }

    #[test]
    fn key_order_does_not_affect_hash() {
        let a = vec![json!({"role": "system", "alpha": 1, "beta": 2, "content": "x"})];
        // Same object, keys inserted in a different textual order.
        let b: Vec<Value> =
            vec![
                serde_json::from_str(r#"{"content":"x","beta":2,"alpha":1,"role":"system"}"#)
                    .unwrap(),
            ];
        assert_eq!(canonical_prefix(&a, "").0, canonical_prefix(&b, "").0);
    }

    #[test]
    fn pre_boundary_hash_changes_when_tool_calls_field_changes() {
        let mut messages = sample_messages();
        // Insert a pre-boundary assistant message carrying tool_calls.
        messages.insert(
            3,
            json!({
                "role": "assistant",
                "content": "thinking",
                "tool_calls": [{"id": "tcA", "function": {"name": "grep", "arguments": "{}"}}]
            }),
        );
        let baseline = canonical_prefix(&messages, "current question").1;

        // Change ONLY the tool_calls field (role + content unchanged).
        messages[3] = json!({
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [{"id": "tcB", "function": {"name": "grep", "arguments": "{}"}}]
        });
        let changed = canonical_prefix(&messages, "current question").1;

        assert_ne!(baseline, changed);
    }

    #[test]
    fn pre_boundary_hash_changes_when_tool_call_id_changes() {
        let mut messages = sample_messages();
        messages.insert(
            3,
            json!({"role": "tool", "content": "r", "tool_call_id": "id1", "name": "t"}),
        );
        let baseline = canonical_prefix(&messages, "current question").1;
        messages[3] = json!({"role": "tool", "content": "r", "tool_call_id": "id2", "name": "t"});
        let changed = canonical_prefix(&messages, "current question").1;
        assert_ne!(baseline, changed);
    }

    #[test]
    fn pre_boundary_hash_changes_when_name_field_changes() {
        let mut messages = sample_messages();
        messages.insert(
            3,
            json!({"role": "tool", "content": "r", "tool_call_id": "id", "name": "name_a"}),
        );
        let baseline = canonical_prefix(&messages, "current question").1;
        messages[3] =
            json!({"role": "tool", "content": "r", "tool_call_id": "id", "name": "name_b"});
        let changed = canonical_prefix(&messages, "current question").1;
        assert_ne!(baseline, changed);
    }

    #[test]
    fn system_hash_is_independent_of_history_growth() {
        let mut messages = sample_messages();
        let system_before = canonical_prefix(&messages, "current question").0;
        // Append tail growth (a new tool result) — system region must not move.
        messages
            .push(json!({"role": "tool", "content": "more", "tool_call_id": "tc2", "name": "x"}));
        let system_after = canonical_prefix(&messages, "current question").0;
        assert_eq!(system_before, system_after);
    }

    #[test]
    fn tail_growth_after_boundary_does_not_flip_pre_boundary_hash() {
        let mut messages = sample_messages();
        let pre_before = canonical_prefix(&messages, "current question").1;
        // Appending more tool results in the current interaction (after boundary).
        messages
            .push(json!({"role": "tool", "content": "more", "tool_call_id": "tc2", "name": "x"}));
        let pre_after = canonical_prefix(&messages, "current question").1;
        assert_eq!(pre_before, pre_after);
    }

    #[test]
    fn empty_messages_do_not_panic() {
        let (sys, pre, boundary) = canonical_prefix(&[], "anything");
        assert_eq!(boundary, 0);
        // Stable hashes for the degenerate case.
        assert_eq!(sys, hash_canonical(&Value::Null));
        assert_eq!(pre, hash_canonical(&Value::Array(vec![])));
    }

    #[test]
    fn fingerprint_is_deterministic_across_identical_inputs() {
        let messages = sample_messages();
        let tools = vec![json!({"name": "read_file", "parameters": {}})];
        let a = provider_call_fingerprint(&messages, "current question", &tools, false);
        let b = provider_call_fingerprint(&messages.clone(), "current question", &tools, false);
        assert_eq!(a, b);
    }

    #[test]
    fn force_text_keeps_tool_defs_hash_and_sets_flag() {
        let messages = sample_messages();
        let tools = vec![json!({"name": "read_file", "parameters": {}})];
        // Force-text retains the tool definitions in the payload (calling is
        // disabled via tool_choice=none) — the hash must be computed normally
        // so attribution can SEE that the defs stayed stable across the
        // force-text turn. Blanking it here would destroy the evidence that
        // the tool-def-refit fix works. The `force_text` boolean is the mode
        // marker; the hash carries the stability signal.
        let fp = provider_call_fingerprint(&messages, "current question", &tools, true);
        assert!(fp.force_text);
        let normal = provider_call_fingerprint(&messages, "current question", &tools, false);
        assert!(!normal.force_text);
        assert_ne!(fp.tool_defs_hash, "");
        assert_eq!(
            fp.tool_defs_hash, normal.tool_defs_hash,
            "same tool defs must hash identically regardless of force-text mode"
        );

        // A genuinely tool-free call still reports an empty hash.
        let empty = provider_call_fingerprint(&messages, "current question", &[], false);
        assert_eq!(empty.tool_defs_hash, "");
    }

    #[test]
    fn fingerprint_surfaces_canonical_prefix_fields() {
        let messages = sample_messages();
        let fp = provider_call_fingerprint(&messages, "current question", &[], false);
        let (sys, pre, boundary) = canonical_prefix(&messages, "current question");
        assert_eq!(fp.hash_system, sys);
        assert_eq!(fp.hash_pre_boundary, pre);
        assert_eq!(fp.boundary_pos, boundary);
        assert_eq!(fp.message_count, messages.len());
    }

    #[test]
    fn session_summary_hash_present_only_when_summary_message_exists() {
        let mut messages = sample_messages();
        let without = provider_call_fingerprint(&messages, "current question", &[], false);
        assert_eq!(without.session_summary_hash, "");

        messages.insert(
            1,
            json!({"role": "system", "content": "[Session Summary]\nUser likes coffee."}),
        );
        let with = provider_call_fingerprint(&messages, "current question", &[], false);
        assert_ne!(with.session_summary_hash, "");
    }

    #[test]
    fn tool_defs_hash_changes_when_schema_changes() {
        let messages = sample_messages();
        let tools_a = vec![json!({"name": "read_file", "parameters": {"type": "object"}})];
        let tools_b = vec![json!({"name": "read_file", "parameters": {"type": "string"}})];
        let a = provider_call_fingerprint(&messages, "current question", &tools_a, false);
        let b = provider_call_fingerprint(&messages, "current question", &tools_b, false);
        assert_ne!(a.tool_defs_hash, b.tool_defs_hash);
    }

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
