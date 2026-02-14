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
use crate::traits::{ModelProvider, StateStore};

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

/// Compute the available token budget for conversation history.
///
/// Subtracts system prompt, tool definitions, and response reserve from the model's
/// total context budget (looked up from config or defaulting to `default_budget`).
pub fn compute_available_budget(
    model: &str,
    system_prompt: &str,
    tool_defs: &[Value],
    config: &ContextWindowConfig,
) -> usize {
    let total_budget = config
        .model_budgets
        .get(model)
        .copied()
        .unwrap_or(config.default_budget);

    let system_tokens = estimate_tokens(system_prompt);
    let tools_json = serde_json::to_string(tool_defs).unwrap_or_default();
    let tools_tokens = estimate_tokens(&tools_json);
    let response_reserve = 2048;

    total_budget.saturating_sub(system_tokens + tools_tokens + response_reserve)
}

/// Fit conversation messages into a token budget.
///
/// If messages fit within budget, returns them unchanged (no-op for short conversations).
/// If over budget:
/// 1. Keeps the first user message (anchor) + last N messages (current context)
/// 2. Injects a conversation summary (if available) after the anchor
/// 3. Drops oldest messages from the middle until under budget
#[allow(dead_code)]
pub fn fit_messages_to_budget(
    messages: Vec<Value>,
    budget_tokens: usize,
    session_summary: Option<&str>,
) -> Vec<Value> {
    // Quick check: if under budget, return as-is
    let messages_json = serde_json::to_string(&messages).unwrap_or_default();
    let current_tokens = estimate_tokens(&messages_json);

    if current_tokens <= budget_tokens {
        return messages;
    }

    let msg_count = messages.len();
    if msg_count <= 2 {
        return messages;
    }

    // Keep first user message (anchor) + last 4 messages (current context)
    let keep_recent = 4.min(msg_count - 1);
    let anchor = messages[0].clone();
    let recent: Vec<Value> = messages[msg_count - keep_recent..].to_vec();

    // Build result: anchor + optional summary + recent messages
    let mut result = Vec::with_capacity(keep_recent + 2);
    result.push(anchor);

    // Inject summary as a system message if available
    if let Some(summary) = session_summary {
        result.push(json!({
            "role": "system",
            "content": format!("[Conversation summary: {}]", summary)
        }));
    }

    result.extend(recent);

    let dropped = msg_count - result.len() + if session_summary.is_some() { 1 } else { 0 };
    info!(
        original_count = msg_count,
        result_count = result.len(),
        dropped,
        original_tokens = current_tokens,
        budget_tokens,
        "Context window: trimmed messages to fit budget"
    );

    result
}

fn role_quota(role: &str) -> usize {
    match role {
        "user" => 10,
        "assistant" => 8,
        "tool" => 4,
        _ => 6,
    }
}

/// Fit messages to a budget using role-aware quotas and recency ranking.
///
/// Compared to `fit_messages_to_budget`, this keeps a more balanced slice of
/// user/assistant/tool context under strict budgets.
pub fn fit_messages_with_source_quotas(
    messages: Vec<Value>,
    budget_tokens: usize,
    session_summary: Option<&str>,
) -> Vec<Value> {
    let messages_json = serde_json::to_string(&messages).unwrap_or_default();
    let current_tokens = estimate_tokens(&messages_json);
    if current_tokens <= budget_tokens {
        return messages;
    }
    if messages.len() <= 2 {
        return messages;
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

    // Always keep a recent tail window.
    let keep_recent = 4usize.min(messages.len());
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

    // Optional summary insertion after the anchor.
    if let Some(summary) = session_summary {
        if !summary.trim().is_empty() {
            let insert_at = 1.min(result.len());
            result.insert(
                insert_at,
                json!({
                    "role": "system",
                    "content": format!("[Conversation summary: {}]", summary)
                }),
            );
        }
    }

    // Trim oldest non-anchor messages until under budget.
    loop {
        let json = serde_json::to_string(&result).unwrap_or_default();
        if estimate_tokens(&json) <= budget_tokens || result.len() <= 2 {
            break;
        }

        // Keep first (anchor) and last 2 always; drop from the middle.
        if result.len() > 3 {
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

    result
}

/// Compress a tool result if it exceeds the character limit.
///
/// Below `max_chars`: returns as-is.
/// Above: truncates and appends annotation.
pub fn compress_tool_result(tool_name: &str, result: &str, max_chars: usize) -> String {
    if result.len() <= max_chars {
        return result.to_string();
    }

    let truncate_to = max_chars.saturating_sub(100); // Leave room for annotation
    let truncated = &result[..truncate_to];
    let compressed = format!(
        "{}\n...\n[truncated {} → {} chars]",
        truncated,
        result.len(),
        truncate_to
    );

    debug!(
        tool = tool_name,
        original_len = result.len(),
        compressed_len = compressed.len(),
        "Compressed tool result"
    );

    compressed
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
        // Truncate very long messages in the summary input
        let truncated = if content.len() > 500 {
            &content[..500]
        } else {
            content
        };
        conversation_text.push_str(&format!("{}: {}\n", role, truncated));
    }

    let llm_messages = vec![
        json!({
            "role": "system",
            "content": "You are a conversation summarizer. Be extremely concise."
        }),
        json!({
            "role": "user",
            "content": format!(
                "Summarize this conversation concisely. Preserve: topics discussed, decisions made, \
                 important data/values mentioned, user preferences expressed, pending tasks.\n\
                 Output 3-5 sentences max.\n\n{}",
                conversation_text
            )
        }),
    ];

    let response = provider.chat(model, &llm_messages, &[]).await?;

    // Track token usage for summarization LLM calls
    if let (Some(state), Some(usage)) = (state, &response.usage) {
        let _ = state
            .record_token_usage("background:summarization", usage)
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

    let response = provider.chat(model, &llm_messages, &[]).await?;

    // Track token usage for progressive extraction LLM calls
    if let (Some(state), Some(usage)) = (state, &response.usage) {
        let _ = state
            .record_token_usage("background:progressive_extraction", usage)
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
        &text[..max_len]
    }
}

/// Run progressive fact extraction in the background.
/// Spawns a tokio task that extracts facts and stores them immediately.
pub fn spawn_progressive_extraction(
    provider: Arc<dyn ModelProvider>,
    fast_model: String,
    state: Arc<dyn StateStore>,
    user_text: String,
    assistant_response: String,
    channel_id: Option<String>,
    visibility: crate::types::ChannelVisibility,
) {
    tokio::spawn(async move {
        // Never extract or persist memory from untrusted public platforms.
        if matches!(visibility, crate::types::ChannelVisibility::PublicExternal) {
            return;
        }

        match extract_inline_facts(
            &provider,
            &fast_model,
            &user_text,
            &assistant_response,
            Some(&state),
        )
        .await
        {
            Ok(facts) if !facts.is_empty() => {
                for fact in facts {
                    // Progressive extraction can capture personal info; default to
                    // conservative privacy unless explicitly promoted later.
                    let privacy = if fact.category.trim().eq_ignore_ascii_case("user") {
                        crate::types::FactPrivacy::Private
                    } else {
                        crate::types::FactPrivacy::Channel
                    };
                    if let Err(e) = state
                        .upsert_fact(
                            &fact.category,
                            &fact.key,
                            &fact.value,
                            "progressive",
                            channel_id.as_deref(),
                            privacy,
                        )
                        .await
                    {
                        warn!(error = %e, key = fact.key, "Failed to store progressive fact");
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
pub fn spawn_incremental_summarization(
    provider: Arc<dyn ModelProvider>,
    fast_model: String,
    state: Arc<dyn StateStore>,
    session_id: String,
    threshold: usize,
    window: usize,
) {
    tokio::spawn(async move {
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

        match summarize_messages(&provider, &fast_model, &to_summarize, Some(&state)).await {
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
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hi"), 0); // 2/4 = 0
        assert_eq!(estimate_tokens("hello world!!"), 3); // 13/4 = 3
                                                         // ~1000 chars should be ~250 tokens
        let long = "a".repeat(1000);
        assert_eq!(estimate_tokens(&long), 250);
    }

    #[test]
    fn test_fit_messages_under_budget() {
        let messages = vec![
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hi there"}),
        ];
        // Huge budget — messages should pass through unchanged
        let result = fit_messages_to_budget(messages.clone(), 100_000, None);
        assert_eq!(result.len(), 2);
        assert_eq!(result, messages);
    }

    #[test]
    fn test_fit_messages_over_budget() {
        let mut messages = Vec::new();
        // Create 15 messages
        for i in 0..15 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            messages.push(json!({"role": role, "content": format!("Message number {}", i)}));
        }

        // Very small budget to force trimming
        let result =
            fit_messages_to_budget(messages.clone(), 50, Some("We discussed topics A and B"));
        // Should have: anchor(1) + summary(1) + recent(4) = 6
        assert_eq!(result.len(), 6);
        // First should be the anchor (first user message)
        assert_eq!(result[0]["content"], "Message number 0");
        // Second should be the injected summary
        assert!(result[1]["content"]
            .as_str()
            .unwrap()
            .contains("Conversation summary"));
        // Last should be the last original message
        assert_eq!(result[5]["content"], "Message number 14");
    }

    #[test]
    fn test_fit_messages_over_budget_no_summary() {
        let mut messages = Vec::new();
        for i in 0..10 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            messages.push(json!({"role": role, "content": format!("Message {}", i)}));
        }

        let result = fit_messages_to_budget(messages, 50, None);
        // Should have: anchor(1) + recent(4) = 5
        assert_eq!(result.len(), 5);
        assert_eq!(result[0]["content"], "Message 0");
        assert_eq!(result[4]["content"], "Message 9");
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

        let result = fit_messages_with_source_quotas(messages, 40, Some("summary"));
        assert!(!result.is_empty());
        assert_eq!(result[0]["role"], "user");
        let tail = result.last().unwrap()["content"].as_str().unwrap();
        assert!(tail.contains("msg-17"));
    }

    #[test]
    fn test_compress_tool_result_short() {
        let short = "Hello world";
        let result = compress_tool_result("test_tool", short, 2000);
        assert_eq!(result, short);
    }

    #[test]
    fn test_compress_tool_result_long() {
        let long = "x".repeat(5000);
        let result = compress_tool_result("test_tool", &long, 2000);
        assert!(result.len() < 5000);
        assert!(result.contains("[truncated"));
        assert!(result.contains("5000"));
    }

    #[test]
    fn test_compute_budget() {
        let config = ContextWindowConfig {
            default_budget: 24000,
            model_budgets: {
                let mut m = std::collections::HashMap::new();
                m.insert("big-model".to_string(), 100000);
                m
            },
            ..Default::default()
        };

        // Default model
        let budget = compute_available_budget("unknown-model", "system prompt", &[], &config);
        // 24000 - estimate_tokens("system prompt") - estimate_tokens("[]") - 2048
        let expected = 24000 - estimate_tokens("system prompt") - estimate_tokens("[]") - 2048;
        assert_eq!(budget, expected);

        // Named model with custom budget
        let budget = compute_available_budget("big-model", "system prompt", &[], &config);
        let expected = 100000 - estimate_tokens("system prompt") - estimate_tokens("[]") - 2048;
        assert_eq!(budget, expected);
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
