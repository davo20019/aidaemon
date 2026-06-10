//! Context window management: token budget enforcement, sliding-window summarization,
//! and progressive fact extraction.
//!
//! Three interconnected subsystems:
//! - **System A**: Token budget enforcement — trims conversation history to fit model limits.
//! - **System B**: Sliding-window summarization — preserves context when messages are trimmed.
//! - **System C**: Progressive fact extraction — extracts durable facts immediately after interactions.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

use crate::config::ContextWindowConfig;
use crate::events::EventStore;
use crate::traits::{ModelProvider, StateStore};
use crate::types::UserRole;

/// Maximum concurrent background extraction LLM calls.
static EXTRACTION_SEMAPHORE: std::sync::LazyLock<Semaphore> =
    std::sync::LazyLock::new(|| Semaphore::new(2));

/// A fact extracted from conversation by progressive extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineFact {
    pub category: String,
    pub key: String,
    pub value: String,
}

/// Estimate token count from text using a simple heuristic (~4 chars per token).
pub fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
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

fn surrogate_multimodal_content_tokens(content: &Value) -> Option<usize> {
    let blocks = content.as_array()?;
    let mut total = 0usize;
    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    total = total.saturating_add(estimate_tokens(text));
                }
            }
            Some("image_url") => {
                total = total.saturating_add(MULTIMODAL_IMAGE_TOKEN_SURROGATE);
            }
            Some("input_audio") => {
                let b64_len = block
                    .get("input_audio")
                    .and_then(|a| a.get("data"))
                    .and_then(|d| d.as_str())
                    .map(str::len)
                    .unwrap_or(0);
                total = total.saturating_add(estimate_audio_tokens_from_base64_len(b64_len));
            }
            _ => {}
        }
    }
    Some(total)
}

/// Estimate tokens for a message array, substituting surrogates for multimodal base64 payloads.
pub fn estimate_multimodal_message_tokens(messages: &[Value]) -> usize {
    let mut surrogate_messages = messages.to_vec();
    let mut used_surrogate = false;

    for msg in surrogate_messages.iter_mut() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        let Some(content) = msg.get("content") else {
            continue;
        };
        if let Some(tokens) = surrogate_multimodal_content_tokens(content) {
            msg["content"] = json!(format!("[multimodal-surrogate:{tokens}]"));
            used_surrogate = true;
        }
    }

    let json = match serde_json::to_string(&surrogate_messages) {
        Ok(s) => s,
        Err(_) => return estimate_tokens(&serde_json::to_string(messages).unwrap_or_default()),
    };

    if used_surrogate {
        tracing::debug!("Using multimodal surrogate token estimate for context budget");
    }
    estimate_tokens(&json)
}

/// Estimate the serialized token cost of OpenAI-format tool definitions.
pub fn estimate_tool_definition_tokens(tool_defs: &[Value]) -> usize {
    let tools_json = serde_json::to_string(tool_defs).unwrap_or_default();
    estimate_tokens(&tools_json)
}

/// Return the configured total context window for a model.
pub fn model_context_budget(model: &str, config: &ContextWindowConfig) -> usize {
    config
        .model_budgets
        .get(model)
        .copied()
        .unwrap_or(config.default_budget)
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
    let messages_json = serde_json::to_string(&messages).unwrap_or_default();
    let current_tokens = estimate_tokens(&messages_json);
    if current_tokens <= budget_tokens {
        return (messages, 0);
    }
    if messages.len() <= 2 {
        return (messages, 0);
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
        let json = serde_json::to_string(&result).unwrap_or_default();
        if estimate_tokens(&json) <= budget_tokens || result.len() <= 2 {
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
            crate::utils::truncation_notice(struct_head + struct_tail, total_chars),
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
            crate::utils::truncation_notice(head_chars, total_chars)
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
        crate::utils::truncation_notice(head_chars + tail_chars, total_chars),
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

fn message_contains_critical_fact_signal(content: &str) -> bool {
    let lower = content.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    lower.contains("my name is")
        || lower.contains("owner name")
        || lower.contains("assistant name")
        || lower.contains("bot name")
        || lower.contains("call me ")
        || lower.contains(" is myself")
        || lower.contains("daughter")
        || lower.contains("son")
        || lower.contains("children")
        || lower.contains("wife")
        || lower.contains("husband")
        || lower.contains("spouse")
        || (lower.contains("saved fact") && lower.contains("name"))
}

/// Summarize old messages using a fast LLM.
///
/// Sends messages to the LLM with a concise summarization prompt.
/// Returns 3-5 sentences preserving topics, decisions, values, and pending tasks.
pub async fn summarize_messages(
    provider: &Arc<dyn ModelProvider>,
    model: &str,
    messages: &[Value],
    state: Option<&Arc<dyn StateStore>>,
    event_store: Option<Arc<EventStore>>,
) -> anyhow::Result<String> {
    // Build a condensed representation of messages for the LLM
    let mut conversation_text = String::new();
    for msg in messages {
        let role = msg
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("unknown");
        let content = msg
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("[no content]");
        let contains_critical = message_contains_critical_fact_signal(content);
        // Truncate very long messages in the summary input (char-boundary safe)
        let max_chars = if contains_critical { 1200 } else { 500 };
        let truncated = if content.len() > max_chars {
            let mut end = max_chars;
            while !content.is_char_boundary(end) && end > 0 {
                end -= 1;
            }
            &content[..end]
        } else {
            content
        };
        let critical_prefix = if contains_critical { "[CRITICAL] " } else { "" };
        conversation_text.push_str(&format!("{}{}: {}\n", critical_prefix, role, truncated));
    }

    let llm_messages = vec![
        json!({
            "role": "system",
            "content": "You are a conversation summarizer. Be extremely concise and preserve critical identity/profile facts."
        }),
        json!({
            "role": "user",
            "content": format!(
                "Summarize this conversation concisely. Preserve: topics discussed, decisions made, \
                 important data/values mentioned, user preferences expressed, pending tasks, \
                 and critical identity/relationship updates (owner name, assistant name, spouse/children).\n\
                 Output 3-5 sentences max.\n\n{}",
                conversation_text
            )
        }),
    ];

    let call_start = std::time::Instant::now();
    let response = provider.chat(model, &llm_messages, &[]).await?;

    if let (Some(state), Some(event_store)) = (state, event_store) {
        crate::events::record_background_model_call_telemetry(
            event_store,
            state.as_ref(),
            "background:summarization",
            "summarization",
            model,
            &response,
            call_start.elapsed(),
        )
        .await;
    }

    response
        .content
        .ok_or_else(|| anyhow::anyhow!("Empty response from summarization LLM"))
}

/// Check if a user message is worth extracting facts from.
///
/// Returns `false` for trivial messages (very short, greetings, acknowledgments).
/// This prevents wasting LLM calls on messages that will never contain durable facts.
pub fn should_extract_facts(user_text: &str) -> bool {
    let trimmed = user_text.trim();

    // Too short to contain meaningful facts
    if trimmed.len() < 20 {
        return false;
    }

    // Single emoji or very short acknowledgments
    let lower = trimmed.to_lowercase();
    let trivial = [
        "ok",
        "okay",
        "thanks",
        "thank you",
        "thx",
        "yes",
        "no",
        "yep",
        "nope",
        "sure",
        "got it",
        "cool",
        "nice",
        "great",
        "good",
        "lol",
        "haha",
        "hmm",
        "ah",
        "oh",
        "right",
        "exactly",
        "agreed",
        "understood",
        "roger",
        "k",
        "kk",
        "ty",
        "np",
        "👍",
        "👋",
        "🙏",
        "✅",
        "done",
        "perfect",
        "awesome",
    ];

    if trivial.contains(&lower.as_str()) {
        return false;
    }

    true
}

/// Extract durable facts from a user-assistant interaction using a fast LLM.
///
/// Returns facts worth remembering (user preferences, personal info, project details).
/// Returns `[]` when nothing worth remembering (most interactions).
/// Rate-limited by a static semaphore (max 2 concurrent calls).
pub async fn extract_inline_facts(
    provider: &Arc<dyn ModelProvider>,
    model: &str,
    user_message: &str,
    assistant_response: &str,
    state: Option<&Arc<dyn StateStore>>,
    event_store: Option<Arc<EventStore>>,
) -> anyhow::Result<Vec<InlineFact>> {
    // Acquire semaphore permit to limit concurrent extraction calls
    let _permit = EXTRACTION_SEMAPHORE.acquire().await?;

    let llm_messages = vec![
        json!({
            "role": "system",
            "content": "You extract durable facts from conversations. Only extract facts that would be useful to remember long-term. \
                        Return a JSON array of objects with 'category', 'key', and 'value' fields.\n\n\
                        Categories: user (personal info), preference (likes/dislikes), project (project details), technical (technical facts).\n\
                        Use snake_case keys like 'dog_name', 'favorite_color', 'work_company'. Be consistent with naming.\n\n\
                        CORRECTIONS: If the user is correcting or updating previously stated information (e.g., \"actually\", \"not X, it's Y\", \
                        \"I changed\", \"I meant\"), extract the CORRECTED fact using the same key format as the original would have used. \
                        The corrected value will automatically supersede the old one.\n\n\
                        IDENTITY: The 'user' category is ONLY for facts the user states about THEMSELVES in first person. \
                        Never store another person's name (friend, family member, client, applicant, public figure) under a user \
                        identity key like 'name'. When the user talks about someone else, do not extract a 'user' fact from it.\n\n\
                        If nothing is worth remembering, return an empty array: []\n\n\
                        Examples:\n\
                        - \"My dog's name is Bella\" → [{\"category\":\"user\",\"key\":\"dog_name\",\"value\":\"Bella\"}]\n\
                        - \"Actually my dog's name is Max, not Bella\" → [{\"category\":\"user\",\"key\":\"dog_name\",\"value\":\"Max\"}]\n\
                        - \"I prefer dark mode\" → [{\"category\":\"preference\",\"key\":\"ui_theme\",\"value\":\"dark mode\"}]\n\
                        - \"My sister lives in Tokyo, not Paris\" → [{\"category\":\"user\",\"key\":\"sister_location\",\"value\":\"Tokyo\"}]\n\
                        - \"How's the weather?\" → []\n\n\
                        IMPORTANT: Return ONLY the JSON array, no other text."
        }),
        json!({
            "role": "user",
            "content": format!(
                "User said: {}\n\nAssistant replied: {}",
                truncate_for_extraction(user_message, 500),
                truncate_for_extraction(assistant_response, 500)
            )
        }),
    ];

    let call_start = std::time::Instant::now();
    let response = provider.chat(model, &llm_messages, &[]).await?;

    if let (Some(state), Some(event_store)) = (state, event_store) {
        crate::events::record_background_model_call_telemetry(
            event_store,
            state.as_ref(),
            "background:progressive_extraction",
            "progressive_extraction",
            model,
            &response,
            call_start.elapsed(),
        )
        .await;
    }

    let text = match response.content {
        Some(t) => t,
        None => return Ok(vec![]),
    };

    // Parse JSON response — be lenient with formatting
    let trimmed = text.trim();
    // Try to find JSON array in the response
    let json_str = if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            &trimmed[start..=end]
        } else {
            return Ok(vec![]);
        }
    } else {
        return Ok(vec![]);
    };

    match serde_json::from_str::<Vec<InlineFact>>(json_str) {
        Ok(facts) => {
            if !facts.is_empty() {
                info!(count = facts.len(), "Progressive extraction found facts");
            }
            Ok(facts)
        }
        Err(e) => {
            debug!(error = %e, response = trimmed, "Failed to parse extraction response");
            Ok(vec![])
        }
    }
}

/// Truncate text for extraction prompts to avoid excessive token usage.
fn truncate_for_extraction(text: &str, max_len: usize) -> &str {
    if text.len() <= max_len {
        text
    } else {
        let mut end = max_len;
        while !text.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        &text[..end]
    }
}

/// Identity-class `user` fact keys that require first-person evidence in the
/// user's own message before they may be persisted.
const USER_IDENTITY_KEYS: &[&str] = &["name", "full_name", "first_name", "last_name", "nickname"];

/// Returns true when an extracted `user` identity fact (e.g. `name`) has no
/// supporting evidence in the user's own words.
///
/// Guards against the extraction model misattributing a third-party name
/// (someone merely mentioned in conversation) — or a hallucinated one — as
/// the user's identity, which then poisons every future prompt that injects
/// the user profile.
pub(crate) fn identity_fact_lacks_user_evidence(
    category: &str,
    key: &str,
    value: &str,
    user_text: &str,
) -> bool {
    if !category.trim().eq_ignore_ascii_case("user") {
        return false;
    }
    let key_norm = key.trim().to_ascii_lowercase();
    if !USER_IDENTITY_KEYS.contains(&key_norm.as_str()) {
        return false;
    }
    let value_norm = value.trim().to_lowercase();
    if value_norm.len() <= 1 {
        return true;
    }
    !user_text.to_lowercase().contains(&value_norm)
}

/// Run progressive fact extraction in the background.
/// Spawns a tokio task that extracts facts and stores them immediately.
#[allow(clippy::too_many_arguments)]
pub fn spawn_progressive_extraction(
    provider: Arc<dyn ModelProvider>,
    fast_model: String,
    state: Arc<dyn StateStore>,
    event_store: Arc<EventStore>,
    user_text: String,
    assistant_response: String,
    channel_id: Option<String>,
    visibility: crate::types::ChannelVisibility,
    user_role: UserRole,
) {
    tokio::spawn(async move {
        // Never extract or persist owner memory from non-owners or untrusted public platforms.
        if !user_role.can_persist_owner_memory()
            || matches!(visibility, crate::types::ChannelVisibility::PublicExternal)
        {
            return;
        }

        match extract_inline_facts(
            &provider,
            &fast_model,
            &user_text,
            &assistant_response,
            Some(&state),
            Some(event_store.clone()),
        )
        .await
        {
            Ok(facts) if !facts.is_empty() => {
                let source_excerpt = crate::utils::truncate_str(&user_text, 200);
                let first_seen_at = chrono::Utc::now();
                let mut written: Vec<serde_json::Value> = Vec::new();
                for fact in facts {
                    // Identity facts (user.name etc.) must be evidenced by the
                    // user's own words — third-party names mentioned in
                    // conversation are not the user's identity.
                    if identity_fact_lacks_user_evidence(
                        &fact.category,
                        &fact.key,
                        &fact.value,
                        &user_text,
                    ) {
                        warn!(
                            key = fact.key,
                            value = fact.value,
                            "Skipping user identity fact without evidence in the user's message"
                        );
                        continue;
                    }
                    // Progressive extraction can capture personal info; default to
                    // conservative privacy unless explicitly promoted later.
                    let privacy = if fact.category.trim().eq_ignore_ascii_case("user") {
                        crate::types::FactPrivacy::Private
                    } else {
                        crate::types::FactPrivacy::Channel
                    };
                    if let Err(e) = state
                        .upsert_fact_with_provenance(
                            &fact.category,
                            &fact.key,
                            &fact.value,
                            "progressive",
                            channel_id.as_deref(),
                            privacy,
                            Some(first_seen_at),
                            Some(source_excerpt.as_str()),
                        )
                        .await
                    {
                        warn!(error = %e, key = fact.key, "Failed to store progressive fact");
                    } else {
                        written.push(json!({
                            "category": fact.category,
                            "key": fact.key,
                            "value": fact.value,
                        }));
                    }
                }
                // Flight-recorder entry so every background memory write is
                // auditable: which facts, from which excerpt, into which channel.
                if !written.is_empty() {
                    let event = crate::events::Event::new(
                        "background:progressive_extraction",
                        crate::events::EventType::DecisionPoint,
                        json!({
                            "code": "memory_write",
                            "decision_type": "memory_write",
                            "severity": "info",
                            "summary": format!(
                                "Progressive extraction stored {} fact(s)",
                                written.len()
                            ),
                            "metadata": {
                                "source": "progressive",
                                "channel_id": channel_id,
                                "facts": written,
                                "source_excerpt": source_excerpt,
                            },
                        }),
                    );
                    if let Err(e) = event_store.append(event).await {
                        debug!(error = %e, "Failed to record memory_write event");
                    }
                }
            }
            Ok(_) => {} // No facts found — expected for most interactions
            Err(e) => {
                debug!(error = %e, "Progressive fact extraction failed");
            }
        }
    });
}

/// Run incremental summarization in the background.
/// Summarizes older messages and stores the summary for future context injection.
#[allow(clippy::too_many_arguments)]
pub fn spawn_incremental_summarization(
    provider: Arc<dyn ModelProvider>,
    fast_model: String,
    state: Arc<dyn StateStore>,
    event_store: Arc<EventStore>,
    session_id: String,
    threshold: usize,
    window: usize,
    user_role: UserRole,
) {
    tokio::spawn(async move {
        if !user_role.can_persist_owner_memory() {
            return;
        }

        let history = match state.get_history(&session_id, 100).await {
            Ok(h) => h,
            Err(e) => {
                warn!(error = %e, "Failed to get history for summarization");
                return;
            }
        };

        if history.len() < threshold {
            return;
        }

        // Convert to JSON values for summarization
        let to_summarize_count = history.len().saturating_sub(window);
        if to_summarize_count == 0 {
            return;
        }

        let to_summarize: Vec<Value> = history[..to_summarize_count]
            .iter()
            .map(|m| {
                json!({
                    "role": m.role,
                    "content": m.content.as_deref().unwrap_or("")
                })
            })
            .collect();

        match summarize_messages(
            &provider,
            &fast_model,
            &to_summarize,
            Some(&state),
            Some(event_store),
        )
        .await
        {
            Ok(text) => {
                let last_msg_id = history[to_summarize_count - 1].id.clone();
                let summary = crate::traits::ConversationSummary {
                    session_id: session_id.clone(),
                    summary: text,
                    message_count: to_summarize_count,
                    last_message_id: last_msg_id,
                    updated_at: chrono::Utc::now(),
                };
                if let Err(e) = state.upsert_conversation_summary(&summary).await {
                    warn!(error = %e, "Failed to store conversation summary");
                } else {
                    info!(
                        session_id = session_id.as_str(),
                        message_count = to_summarize_count,
                        "Stored conversation summary"
                    );
                }
            }
            Err(e) => {
                warn!(error = %e, "Failed to summarize messages");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_guard_blocks_unevidenced_user_name() {
        // Third-party or hallucinated name not present in the user's words.
        assert!(identity_fact_lacks_user_evidence(
            "user",
            "name",
            "Edison Mendez",
            "Tell me about the beca applicant and their nationality"
        ));
    }

    #[test]
    fn identity_guard_allows_first_person_name() {
        assert!(!identity_fact_lacks_user_evidence(
            "user",
            "name",
            "David Loor",
            "Hi, my name is David Loor and I live in Quito"
        ));
        // Case-insensitive match.
        assert!(!identity_fact_lacks_user_evidence(
            "user",
            "Name",
            "david loor",
            "I'm David Loor"
        ));
    }

    #[test]
    fn identity_guard_ignores_non_identity_facts() {
        // Other user facts are not gated (dog_name is not the user's identity).
        assert!(!identity_fact_lacks_user_evidence(
            "user",
            "dog_name",
            "Bella",
            "what's the weather?"
        ));
        // Non-user categories are never gated.
        assert!(!identity_fact_lacks_user_evidence(
            "project",
            "name",
            "aidaemon",
            "unrelated text"
        ));
    }

    #[test]
    fn identity_guard_blocks_trivial_values() {
        assert!(identity_fact_lacks_user_evidence("user", "name", "x", "x"));
        assert!(identity_fact_lacks_user_evidence(
            "user", "name", " ", "anything"
        ));
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
        assert!(surrogate < 50_000);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hi"), 0); // 2/4 = 0
        assert_eq!(estimate_tokens("hello world!!"), 3); // 13/4 = 3
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

    #[test]
    fn test_should_extract_facts_trivial() {
        assert!(!should_extract_facts("ok"));
        assert!(!should_extract_facts("thanks"));
        assert!(!should_extract_facts("yes"));
        assert!(!should_extract_facts("lol"));
        assert!(!should_extract_facts("👍"));
        assert!(!should_extract_facts("short")); // <20 chars
        assert!(!should_extract_facts("Got it")); // <20 chars
    }

    #[test]
    fn test_should_extract_facts_meaningful() {
        assert!(should_extract_facts(
            "My dog's name is Bella and she's a golden retriever"
        ));
        assert!(should_extract_facts(
            "I work at Acme Corp in the engineering department"
        ));
        assert!(should_extract_facts(
            "Please set up a new React project with TypeScript"
        ));
    }

    #[test]
    fn test_inline_fact_deserialization() {
        let json = r#"[{"category":"user","key":"dog_name","value":"Bella"}]"#;
        let facts: Vec<InlineFact> = serde_json::from_str(json).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].category, "user");
        assert_eq!(facts[0].key, "dog_name");
        assert_eq!(facts[0].value, "Bella");
    }

    #[test]
    fn test_inline_fact_empty_array() {
        let json = "[]";
        let facts: Vec<InlineFact> = serde_json::from_str(json).unwrap();
        assert!(facts.is_empty());
    }
}
