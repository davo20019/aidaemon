//! Context budgeting for the daemon's wire-format requests: token estimates
//! (calibrated per model), message and tool-definition fitting, and tool
//! result compression.
//!
//! These work on OpenAI-format JSON values. They move into the policies
//! crate's `ContextWindow` once the daemon builds requests from kernel
//! messages (PLAN 8.3); progressive fact extraction is memory and lives in
//! `memory::progressive`.

use serde_json::{json, Value};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::settings::ContextWindowConfig;

/// Rolling provider/model correction for the portable chars/4 fallback.
static TOKEN_ESTIMATE_CORRECTION: std::sync::LazyLock<
    std::sync::RwLock<HashMap<String, (f64, u32)>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(HashMap::new()));

/// Estimate token count from text using a simple heuristic (~4 chars per token).
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

pub fn record_token_estimate_calibration(model: &str, estimated: usize, actual: usize) {
    if model.trim().is_empty() || estimated == 0 || actual == 0 {
        return;
    }
    let observed = (actual as f64 / estimated as f64).clamp(0.5, 3.0);
    let mut calibration = TOKEN_ESTIMATE_CORRECTION
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = calibration
        .entry(model.to_string())
        .or_insert((observed, 0));
    entry.0 = if entry.1 == 0 {
        observed
    } else {
        entry.0 * 0.8 + observed * 0.2
    };
    entry.1 = entry.1.saturating_add(1);
}

pub fn token_estimate_correction(model: &str) -> f64 {
    TOKEN_ESTIMATE_CORRECTION
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(model)
        .map(|(factor, _)| *factor)
        .unwrap_or(1.0)
}

const MULTIMODAL_IMAGE_TOKEN_SURROGATE: usize = 1_200;
const MULTIMODAL_AUDIO_BYTES_PER_TOKEN: usize = 100;

fn estimate_audio_tokens_from_base64_len(base64_len: usize) -> usize {
    // Rough surrogate: ~1 token per 100 bytes of raw audio (base64 is ~4/3 inflation).
    let raw_bytes = base64_len.saturating_mul(3) / 4;
    raw_bytes
        .saturating_div(MULTIMODAL_AUDIO_BYTES_PER_TOKEN)
        .max(64)
}

/// Estimate tokens for a message array, substituting bounded provider-token
/// surrogates for inline multimodal bytes.
///
/// The serialized `data:` URL is transport encoding, not prompt text. Counting
/// its characters made one ordinary screenshot look larger than the entire
/// context window during cold-start anchor selection. Keep the surrounding
/// JSON/text in the chars/4 estimate, remove only the binary payload, then add
/// the explicit image/audio token surrogate as a number (rather than embedding
/// that number in a short string, which previously undercounted it again).
pub fn estimate_multimodal_message_tokens(messages: &[Value]) -> usize {
    let mut surrogate_messages = messages.to_vec();
    let mut used_surrogate = false;
    let mut multimodal_tokens = 0usize;

    for msg in surrogate_messages.iter_mut() {
        // Renderer provenance is an out-of-band telemetry sidecar carried on
        // the value only until final projection. It must not consume provider
        // budget or change compaction/eviction decisions.
        if let Some(object) = msg.as_object_mut() {
            object.retain(|key, _| !key.starts_with("_aidaemon_"));
        }
        if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        let Some(blocks) = msg.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };

        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("image_url") => {
                    multimodal_tokens =
                        multimodal_tokens.saturating_add(MULTIMODAL_IMAGE_TOKEN_SURROGATE);
                    *block = json!({
                        "type": "image_url",
                        "image_url": {"url": "[image-bytes-omitted-for-token-estimate]"}
                    });
                    used_surrogate = true;
                }
                Some("input_audio") => {
                    let b64_len = block
                        .get("input_audio")
                        .and_then(|a| a.get("data"))
                        .and_then(Value::as_str)
                        .map(str::len)
                        .unwrap_or(0);
                    multimodal_tokens = multimodal_tokens
                        .saturating_add(estimate_audio_tokens_from_base64_len(b64_len));
                    *block = json!({
                        "type": "input_audio",
                        "input_audio": {
                            "data": "[audio-bytes-omitted-for-token-estimate]",
                            "format": "unknown"
                        }
                    });
                    used_surrogate = true;
                }
                _ => {}
            }
        }
    }

    let json = match serde_json::to_string(&surrogate_messages) {
        Ok(s) => s,
        Err(_) => return estimate_tokens(&serde_json::to_string(messages).unwrap_or_default()),
    };

    if used_surrogate {
        tracing::debug!("Using multimodal surrogate token estimate for context budget");
    }
    estimate_tokens(&json).saturating_add(multimodal_tokens)
}

/// Estimate the serialized token cost of OpenAI-format tool definitions.
pub fn estimate_tool_definition_tokens(tool_defs: &[Value]) -> usize {
    let tools_json = serde_json::to_string(tool_defs).unwrap_or_default();
    estimate_tokens(&tools_json)
}

/// Return the configured total context window for a model.
pub fn model_context_budget(model: &str, config: &ContextWindowConfig) -> usize {
    config.context_budget_for(model)
}

fn truncate_description(value: &mut Value, max_chars: usize) {
    let Some(text) = value.as_str() else {
        return;
    };
    if text.chars().count() <= max_chars {
        return;
    }

    let mut truncated: String = text.chars().take(max_chars.saturating_sub(3)).collect();
    truncated.push_str("...");
    *value = Value::String(truncated);
}

fn compact_schema_metadata(value: &mut Value, description_limit: usize) {
    match value {
        Value::Object(map) => {
            for annotation in ["title", "examples", "$comment", "default"] {
                map.remove(annotation);
            }
            if let Some(description) = map.get_mut("description") {
                truncate_description(description, description_limit);
            }
            for child in map.values_mut() {
                compact_schema_metadata(child, description_limit);
            }
        }
        Value::Array(values) => {
            for child in values {
                compact_schema_metadata(child, description_limit);
            }
        }
        _ => {}
    }
}

fn compact_parameter_annotations(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for annotation in ["description", "title", "examples", "$comment", "default"] {
                map.remove(annotation);
            }
            for child in map.values_mut() {
                compact_parameter_annotations(child);
            }
        }
        Value::Array(values) => {
            for child in values {
                compact_parameter_annotations(child);
            }
        }
        _ => {}
    }
}

/// Reduce descriptive tool-schema overhead while preserving the complete callable surface.
///
/// Tool entries, function names, parameter properties, required fields, enums, and validation
/// constraints are retained. Only annotation-only metadata and verbose descriptions are reduced.
pub fn fit_tool_definitions_to_budget(tool_defs: &[Value], budget_tokens: usize) -> Vec<Value> {
    if estimate_tool_definition_tokens(tool_defs) <= budget_tokens {
        return tool_defs.to_vec();
    }

    let mut compacted = tool_defs.to_vec();
    for description_limit in [512, 256, 128, 64, 32] {
        compacted = tool_defs.to_vec();
        for definition in &mut compacted {
            compact_schema_metadata(definition, description_limit);
        }
        if estimate_tool_definition_tokens(&compacted) <= budget_tokens {
            break;
        }
    }

    if estimate_tool_definition_tokens(&compacted) > budget_tokens {
        compacted = tool_defs.to_vec();
        for definition in &mut compacted {
            if let Some(function) = definition.get_mut("function") {
                if let Some(description) = function.get_mut("description") {
                    truncate_description(description, 32);
                }
                if let Some(parameters) = function.get_mut("parameters") {
                    compact_parameter_annotations(parameters);
                }
            }
        }
    }

    compacted
}

/// Compute the available token budget for conversation history.
///
/// Subtracts system prompt, tool definitions, and response reserve from the model's
/// total context budget (looked up from config or defaulting to `default_budget`).
/// Use [`compute_available_budget`] when you have the full prompt string, or inline
/// the additive pattern when you have pre-computed per-component token counts.
pub const CONTEXT_RESPONSE_RESERVE_TOKENS: usize = 1536;

/// Compute the available message budget given a pre-computed system-token count.
/// Prefer this over [`compute_available_budget`] when the system prompt is split
/// into components whose token counts are already known (avoids a String allocation).
pub fn compute_available_budget_precomputed(
    model: &str,
    system_tokens: usize,
    tool_defs: &[Value],
    config: &ContextWindowConfig,
) -> usize {
    let total_budget = model_context_budget(model, config);
    let tools_tokens = estimate_tool_definition_tokens(tool_defs);
    total_budget.saturating_sub(system_tokens + tools_tokens + CONTEXT_RESPONSE_RESERVE_TOKENS)
}

/// Convenience wrapper: computes the system-token count from the combined prompt
/// string. Use [`compute_available_budget_precomputed`] when components are already
/// computed separately to avoid an extra allocation.
#[allow(dead_code)]
pub fn compute_available_budget(
    model: &str,
    system_prompt: &str,
    tool_defs: &[Value],
    config: &ContextWindowConfig,
) -> usize {
    compute_available_budget_precomputed(model, estimate_tokens(system_prompt), tool_defs, config)
}

fn role_quota(role: &str) -> usize {
    match role {
        "user" => 10,
        "assistant" => 10,
        "tool" => 8,
        _ => 6,
    }
}

/// Fit messages to a budget using role-aware quotas and recency ranking.
///
/// Keeps a balanced slice of user/assistant/tool context under strict budgets.
///
/// Pillar A: the session summary is no longer inserted here — it lives only in
/// the per-task context tail (`build_system_prompt_for_message`).
///
/// Pillar B (Task 7): callers scope this to the CURRENT-TURN region only —
/// archived turns are whole-turn-evicted upstream and never trimmed here.
///
/// Returns `(fitted_messages, dropped)` where `dropped` is the number of
/// messages removed from the input (`input_len - fitted_len`). The result is
/// always a subset of the input (this never adds messages), so the count is an
/// accurate "did anything get dropped?" signal — Pillar B (Task 8) keys the
/// `history_fitting` prefix-mutation attribution line off `dropped > 0`.
pub fn fit_messages_with_source_quotas(
    messages: Vec<Value>,
    budget_tokens: usize,
) -> (Vec<Value>, usize) {
    let input_len = messages.len();
    let current_tokens = estimate_multimodal_message_tokens(&messages);
    if current_tokens <= budget_tokens {
        return (messages, 0);
    }
    if messages.len() <= 2 {
        let mut compacted = messages;
        let per_message_chars = budget_tokens
            .saturating_mul(4)
            .saturating_sub(512usize.saturating_mul(compacted.len()))
            .checked_div(compacted.len().max(1))
            .unwrap_or(0)
            .max(256);
        for message in &mut compacted {
            let Some(content) = message.get_mut("content") else {
                continue;
            };
            let Some(text) = content.as_str() else {
                continue;
            };
            if text.chars().count() > per_message_chars {
                *content = Value::String(compact_summary_input(text, per_message_chars));
            }
        }
        let after_tokens = estimate_multimodal_message_tokens(&compacted);
        warn!(
            original_tokens = current_tokens,
            after_tokens,
            budget_tokens,
            "Oversized minimal current context was explicitly head/tail compacted"
        );
        return (compacted, 0);
    }

    let mut selected_indices: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut role_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    // Anchor: first user message if present, otherwise first message.
    let anchor_idx = messages
        .iter()
        .position(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .unwrap_or(0);
    selected_indices.insert(anchor_idx);
    let anchor_role = messages[anchor_idx]
        .get("role")
        .and_then(|r| r.as_str())
        .unwrap_or("unknown")
        .to_string();
    *role_counts.entry(anchor_role).or_insert(0) += 1;

    // Always keep a recent tail window — must be large enough to hold a multi-step
    // task's tool calls (write → run → verify cycles easily produce 8+ messages).
    let keep_recent = 8usize.min(messages.len());
    let start = messages.len().saturating_sub(keep_recent);
    for (idx, msg) in messages.iter().enumerate().skip(start) {
        if selected_indices.insert(idx) {
            let role = msg
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("unknown")
                .to_string();
            *role_counts.entry(role).or_insert(0) += 1;
        }
    }

    // Fill remaining budget candidates from most recent backwards with role quotas.
    for idx in (0..messages.len()).rev() {
        if selected_indices.contains(&idx) {
            continue;
        }
        let role = messages[idx]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("unknown");
        let quota = role_quota(role);
        let count = role_counts.get(role).copied().unwrap_or(0);
        if count >= quota {
            continue;
        }
        selected_indices.insert(idx);
        *role_counts.entry(role.to_string()).or_insert(0) += 1;
    }

    // Materialize selected messages in original order.
    let mut result: Vec<Value> = selected_indices
        .iter()
        .map(|idx| messages[*idx].clone())
        .collect();

    // Trim oldest non-anchor messages until under budget.
    loop {
        if estimate_multimodal_message_tokens(&result) <= budget_tokens || result.len() <= 2 {
            break;
        }

        // Keep first (anchor) and last 6 always; drop from the middle.
        // Protecting more recent messages prevents loss of current-task tool results.
        if result.len() > 7 {
            result.remove(1);
        } else {
            break;
        }
    }

    info!(
        original_count = messages.len(),
        result_count = result.len(),
        original_tokens = current_tokens,
        budget_tokens,
        "Context window: applied source quotas"
    );

    let dropped = input_len.saturating_sub(result.len());
    (result, dropped)
}

/// Compress a tool result if it exceeds the character limit.
///
/// Below `max_chars`: returns as-is.
/// Above: preserves head+tail and drops the middle.
pub fn compress_tool_result(tool_name: &str, result: &str, max_chars: usize) -> String {
    let total_chars = result.chars().count();
    if total_chars <= max_chars {
        return result.to_string();
    }

    // Keep space for marker text; preserve as much head+tail signal as possible.
    const ANNOTATION_OVERHEAD: usize = 64;
    const MIN_HEAD_CHARS: usize = 120;
    const MIN_TAIL_CHARS: usize = 80;

    if looks_like_structured_payload(result) {
        // For structured payloads (JSON API responses), keep head + tail so the
        // model sees the JSON summary/metadata at the top AND some complete items
        // from deeper in the response.  Head-heavy ratio (70/30) because the
        // summary and first items are most informative.
        let available = max_chars.saturating_sub(ANNOTATION_OVERHEAD);
        let struct_head = (available * 7) / 10;
        let struct_tail = available.saturating_sub(struct_head);

        if total_chars <= struct_head + struct_tail {
            return result.to_string();
        }

        let head_end = byte_index_after_chars(result, struct_head);
        let tail_start = byte_index_before_last_chars(result, struct_tail);
        let compressed = format!(
            "{}\n\n{}\n\n{}",
            &result[..head_end],
            aidaemon_tools::text::truncation_notice(struct_head + struct_tail, total_chars),
            &result[tail_start..]
        );

        debug!(
            tool = tool_name,
            original_len = total_chars,
            compressed_len = compressed.len(),
            "Compressed structured tool result"
        );

        return compressed;
    }

    if max_chars <= ANNOTATION_OVERHEAD + MIN_HEAD_CHARS + MIN_TAIL_CHARS {
        let head_chars = max_chars.saturating_sub(ANNOTATION_OVERHEAD).max(1);
        let head_end = byte_index_after_chars(result, head_chars);
        return format!(
            "{}\n\n{}",
            &result[..head_end],
            aidaemon_tools::text::truncation_notice(head_chars, total_chars)
        );
    }

    // Head/tail scale with the configured budget so raising
    // max_tool_result_chars genuinely increases what the model sees; only
    // the MIN floors are fixed (they keep tiny budgets usable).
    let available = max_chars.saturating_sub(ANNOTATION_OVERHEAD);
    let mut head_chars = ((available * 5) / 9).max(MIN_HEAD_CHARS);
    let mut tail_chars = available.saturating_sub(head_chars).max(MIN_TAIL_CHARS);
    if head_chars + tail_chars > available {
        tail_chars = available.saturating_sub(head_chars);
    }
    if tail_chars < MIN_TAIL_CHARS {
        tail_chars = MIN_TAIL_CHARS.min(available.saturating_sub(1));
        head_chars = available.saturating_sub(tail_chars);
    }

    if total_chars <= head_chars + tail_chars {
        return result.to_string();
    }

    let head_end = byte_index_after_chars(result, head_chars);
    let tail_start = byte_index_before_last_chars(result, tail_chars);
    let compressed = format!(
        "{}\n\n{}\n\n{}",
        &result[..head_end],
        aidaemon_tools::text::truncation_notice(head_chars + tail_chars, total_chars),
        &result[tail_start..]
    );

    debug!(
        tool = tool_name,
        original_len = total_chars,
        compressed_len = compressed.len(),
        "Compressed tool result"
    );

    compressed
}

fn looks_like_structured_payload(result: &str) -> bool {
    let trimmed = result.trim_start();
    trimmed.starts_with('{')
        || (trimmed.starts_with('[') && !trimmed.starts_with("[UNTRUSTED"))
        || result.contains("\nJSON summary:\n")
        || result.contains("\nTop-level JSON array")
}

fn byte_index_after_chars(s: &str, char_count: usize) -> usize {
    if char_count == 0 {
        return 0;
    }
    s.char_indices()
        .map(|(idx, _)| idx)
        .nth(char_count)
        .unwrap_or(s.len())
}

fn byte_index_before_last_chars(s: &str, char_count: usize) -> usize {
    if char_count == 0 {
        return s.len();
    }
    let total = s.chars().count();
    if char_count >= total {
        return 0;
    }
    byte_index_after_chars(s, total.saturating_sub(char_count))
}

pub fn compact_summary_input(content: &str, max_chars: usize) -> String {
    aidaemon_policies::elide_middle(content, max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_usage_calibrates_the_portable_token_estimator() {
        let model = "synthetic-calibration-model";
        assert_eq!(token_estimate_correction(model), 1.0);
        record_token_estimate_calibration(model, 1_000, 1_500);
        assert!((token_estimate_correction(model) - 1.5).abs() < f64::EPSILON);
        record_token_estimate_calibration(model, 1_000, 1_000);
        assert!((token_estimate_correction(model) - 1.4).abs() < 0.001);
    }

    #[test]
    fn internal_projection_metadata_does_not_change_context_budget() {
        let plain = vec![json!({"role": "user", "content": "synthetic request"})];
        let annotated = vec![json!({
            "role": "user",
            "content": "synthetic request",
            "_aidaemon_source_message_ids": ["message-1", "message-2"],
            "_aidaemon_source_turn_ids": ["turn-1"]
        })];

        assert_eq!(
            estimate_multimodal_message_tokens(&plain),
            estimate_multimodal_message_tokens(&annotated)
        );
    }

    #[test]
    fn multimodal_audio_surrogate_does_not_explode_estimate() {
        let huge_b64 = "A".repeat(1_400_000);
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "listen"},
                {"type": "input_audio", "input_audio": {"data": huge_b64, "format": "opus"}}
            ]
        })];
        let naive = estimate_tokens(&serde_json::to_string(&messages).unwrap());
        let surrogate = estimate_multimodal_message_tokens(&messages);
        assert!(
            surrogate < naive / 10,
            "surrogate {surrogate} vs naive {naive}"
        );
        assert!(
            surrogate >= 10_500,
            "audio surrogate was not added: {surrogate}"
        );
        assert!(surrogate < 50_000);
    }

    #[test]
    fn multimodal_image_surrogate_is_bounded_and_actually_charged() {
        let huge_data_url = format!("data:image/png;base64,{}", "A".repeat(800_000));
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "PDF preview"},
                {"type": "image_url", "image_url": {"url": huge_data_url}}
            ]
        })];
        let naive = estimate_tokens(&serde_json::to_string(&messages).unwrap());
        let surrogate = estimate_multimodal_message_tokens(&messages);

        assert!(
            surrogate >= MULTIMODAL_IMAGE_TOKEN_SURROGATE,
            "image surrogate was not added: {surrogate}"
        );
        assert!(
            surrogate < naive / 10,
            "surrogate {surrogate} vs naive {naive}"
        );
        assert!(surrogate < 2_000);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hi"), 1); // ceil(2/4) = 1
        assert_eq!(estimate_tokens("hello world!!"), 4); // ceil(13/4) = 4
                                                         // ~1000 chars should be ~250 tokens
        let long = "a".repeat(1000);
        assert_eq!(estimate_tokens(&long), 250);
    }

    #[test]
    fn test_fit_with_source_quotas_keeps_anchor_and_recent() {
        let mut messages = Vec::new();
        for i in 0..18 {
            let role = if i % 3 == 0 {
                "user"
            } else if i % 3 == 1 {
                "assistant"
            } else {
                "tool"
            };
            messages.push(json!({"role": role, "content": format!("msg-{i}")}));
        }

        let (result, dropped) = fit_messages_with_source_quotas(messages, 40);
        assert!(!result.is_empty());
        // Pillar B (Task 8): the fitter reports how many messages it dropped so
        // the call site can attribute the `history_fitting` prefix mutation.
        assert!(dropped > 0, "tight budget must drop at least one message");
        assert_eq!(result[0]["role"], "user");
        let tail = result.last().unwrap()["content"].as_str().unwrap();
        assert!(tail.contains("msg-17"));
        // Pillar A: no summary message is injected by the fitter anymore.
        assert!(result.iter().all(|m| {
            !m["content"]
                .as_str()
                .unwrap_or("")
                .contains("Conversation summary")
        }));
    }

    #[test]
    fn test_fit_with_source_quotas_reports_zero_dropped_when_under_budget() {
        // Pillar B (Task 8): when nothing is dropped, `dropped` must be 0 so the
        // call site stays silent (no unattributed prefix mutation logged).
        let messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
        ];
        let original_len = messages.len();
        let (result, dropped) = fit_messages_with_source_quotas(messages, 100_000);
        assert_eq!(dropped, 0);
        assert_eq!(result.len(), original_len);
    }

    #[test]
    fn test_compress_tool_result_short() {
        let short = "Hello world";
        let result = compress_tool_result("test_tool", short, 2000);
        assert_eq!(result, short);
    }

    #[test]
    fn test_compress_tool_result_long() {
        let long = format!("HEAD:{}:TAIL", "x".repeat(5000));
        let result = compress_tool_result("test_tool", &long, 2000);
        assert!(result.len() < long.len());
        assert!(result.contains("OUTPUT TRUNCATED"));
        assert!(result.contains("HEAD:"));
        assert!(result.contains(":TAIL"));
    }

    #[test]
    fn test_compress_tool_result_uses_full_configured_budget() {
        // A result just over the limit must keep most of the configured
        // budget, not get clamped to a fixed ~1800 chars regardless of config.
        let long = format!("HEAD:{}:TAIL", "x".repeat(4500));
        let result = compress_tool_result("test_tool", &long, 4000);
        let kept = result.chars().count();
        assert!(
            kept > 3000,
            "4000-char budget should retain >3000 chars, kept {kept}"
        );
    }

    #[test]
    fn test_compress_tool_result_scales_with_larger_budget() {
        // Raising max_tool_result_chars must actually increase retained
        // content (important when users configure big-context models).
        let long = format!("HEAD:{}:TAIL", "x".repeat(30000));
        let small = compress_tool_result("test_tool", &long, 4000);
        let large = compress_tool_result("test_tool", &long, 16000);
        let small_kept = small.chars().count();
        let large_kept = large.chars().count();
        assert!(
            large_kept > small_kept * 2,
            "16k budget should retain far more than 4k budget (small={small_kept}, large={large_kept})"
        );
        assert!(
            large_kept > 12000,
            "16k budget should retain >12000 chars, kept {large_kept}"
        );
        assert!(large.contains("HEAD:"));
        assert!(large.contains(":TAIL"));
        assert!(large.contains("OUTPUT TRUNCATED"));
    }

    #[test]
    fn test_compress_tool_result_tiny_budget_still_bounded() {
        let long = "y".repeat(10000);
        let result = compress_tool_result("test_tool", &long, 300);
        assert!(result.contains("OUTPUT TRUNCATED"));
        assert!(result.chars().count() < 1000);
    }

    #[test]
    fn test_compress_tool_result_keeps_head_and_tail_for_structured_payloads() {
        // Build a structured payload large enough to trigger compression
        let json_body =
            "{\n  \"items\": [\n".to_string() + &"    {\"id\":1},\n".repeat(100) + "  ]\n}";
        let structured = format!(
            "[UNTRUSTED EXTERNAL DATA from 'http_request']\nHTTP 200 OK\n\nJSON summary:\nitems: array(2 item(s))\n\n{}",
            json_body
        );
        let result = compress_tool_result("http_request", &structured, 600);
        assert!(result.contains("JSON summary:"));
        assert!(result.contains("OUTPUT TRUNCATED"));
        // Head+tail: should contain both the beginning (JSON summary) and the
        // end of the payload (closing braces from the JSON structure).
        assert!(result.contains("]\n}"));
    }

    #[test]
    fn test_compute_budget() {
        let config = ContextWindowConfig {
            default_budget: 32000,
            model_budgets: {
                let mut m = std::collections::HashMap::new();
                m.insert("big-model".to_string(), 100000);
                m
            },
            ..Default::default()
        };

        // Default model
        let budget = compute_available_budget("unknown-model", "system prompt", &[], &config);
        // 32000 - estimate_tokens("system prompt") - estimate_tokens("[]") - 1536
        let expected = 32000 - estimate_tokens("system prompt") - estimate_tokens("[]") - 1536;
        assert_eq!(budget, expected);

        // Named model with custom budget
        let budget = compute_available_budget("big-model", "system prompt", &[], &config);
        let expected = 100000 - estimate_tokens("system prompt") - estimate_tokens("[]") - 1536;
        assert_eq!(budget, expected);
    }

    #[test]
    fn test_fit_tool_definitions_under_budget_is_unchanged() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to read."
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }
            }
        })];

        let compacted = fit_tool_definitions_to_budget(&tools, 10_000);
        assert_eq!(compacted, tools);
    }

    #[test]
    fn test_fit_tool_definitions_preserves_tools_and_parameter_contracts() {
        let verbose = "Detailed operational guidance. ".repeat(200);
        let tools: Vec<Value> = (0..12)
            .map(|idx| {
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("verbose_tool_{idx}"),
                        "description": verbose,
                        "parameters": {
                            "type": "object",
                            "title": "Verbose tool input",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": verbose,
                                    "examples": ["/tmp/example"]
                                },
                                "mode": {
                                    "type": "string",
                                    "description": verbose,
                                    "enum": ["read", "write"]
                                }
                            },
                            "required": ["path", "mode"],
                            "additionalProperties": false
                        }
                    }
                })
            })
            .collect();

        let original_tokens = estimate_tool_definition_tokens(&tools);
        let compacted = fit_tool_definitions_to_budget(&tools, 2_000);

        assert_eq!(compacted.len(), tools.len());
        assert!(
            estimate_tool_definition_tokens(&compacted) < original_tokens,
            "verbose schemas should be reduced"
        );
        assert!(
            estimate_tool_definition_tokens(&compacted) <= 2_000,
            "compacted schemas should fit the requested budget"
        );

        for (idx, tool) in compacted.iter().enumerate() {
            assert_eq!(
                tool["function"]["name"],
                Value::String(format!("verbose_tool_{idx}"))
            );
            assert_eq!(
                tool["function"]["parameters"]["properties"]["path"]["type"],
                "string"
            );
            assert_eq!(
                tool["function"]["parameters"]["properties"]["mode"]["enum"],
                json!(["read", "write"])
            );
            assert_eq!(
                tool["function"]["parameters"]["required"],
                json!(["path", "mode"])
            );
            assert_eq!(
                tool["function"]["parameters"]["additionalProperties"],
                false
            );
        }
    }
}
