use crate::router::{Router, Tier};
use crate::traits::{ConversationSummary, ModelProvider, StateStore};
use crate::utils::floor_char_boundary;
use chrono::Utc;
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::warn;

/// Why a compaction should happen right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CompactionTrigger {
    /// A pair aged out of the sliding window. Contains the message ID of the aging user message.
    WindowOverflow { aging_pair_user_msg_id: String },
    /// Session idle for >2 hours. All window pairs should be compacted.
    IdleGap,
    /// File upload without referential language. Compact messages outside window.
    FileUpload,
}

/// Referential phrases that indicate the file upload refers to earlier conversation context.
const REFERENTIAL_PHRASES: &[&str] = &[
    "mentioned",
    "discussed",
    "we were working on",
    "the file from",
    "here's the",
];

/// Detect whether compaction should be triggered.
///
/// Priority: IdleGap > FileUpload > WindowOverflow (checked in that order).
pub(crate) fn detect_compaction_trigger(
    total_pairs: usize,
    window_size: usize,
    idle_gap_seconds: u64,
    user_text: &str,
) -> Option<CompactionTrigger> {
    // 1. Idle gap — highest priority
    if idle_gap_seconds > 7200 {
        return Some(CompactionTrigger::IdleGap);
    }

    // 2. File upload without referential language
    if user_text.contains("[File received:") {
        let lower = user_text.to_lowercase();
        let has_reference = REFERENTIAL_PHRASES
            .iter()
            .any(|phrase| lower.contains(phrase));
        if !has_reference {
            return Some(CompactionTrigger::FileUpload);
        }
    }

    // 3. Window overflow
    if total_pairs > window_size {
        return Some(CompactionTrigger::WindowOverflow {
            aging_pair_user_msg_id: String::new(), // caller fills in the actual ID
        });
    }

    None
}

/// Tracks user-message IDs that are pending compaction.
///
/// The cap of 3 prevents unbounded accumulation if compaction calls are slow.
/// Currently unused — compaction runs synchronously. Will be used when async
/// compaction is implemented as a future optimization.
#[allow(dead_code)]
pub(crate) struct PendingCompaction {
    pub pair_ids: Vec<String>,
}

#[allow(dead_code)]
impl PendingCompaction {
    pub fn new() -> Self {
        Self {
            pair_ids: Vec::new(),
        }
    }

    /// Add a pair to the pending list. Returns the oldest pair ID if cap (3) is exceeded.
    pub fn add(&mut self, user_msg_id: String) -> Option<String> {
        self.pair_ids.push(user_msg_id);
        if self.pair_ids.len() > 3 {
            Some(self.pair_ids.remove(0))
        } else {
            None
        }
    }

    /// Remove pairs that have been incorporated into the summary.
    pub fn drain_completed(&mut self, last_compacted_msg_id: &str) {
        self.pair_ids.retain(|id| id != last_compacted_msg_id);
    }

    pub fn is_empty(&self) -> bool {
        self.pair_ids.is_empty()
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.pair_ids.len()
    }
}

/// Format messages for the compaction prompt.
///
/// Each message is rendered as:
/// - `User: <content>`
/// - `Assistant: <content>`
/// - `Tool: <tool_name> -> <truncated_content>`
fn format_messages_for_prompt(messages: &[Value]) -> String {
    messages
        .iter()
        .filter_map(|msg| {
            let role = msg.get("role")?.as_str()?;
            match role {
                "user" => {
                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    Some(format!("User: {}", content))
                }
                "assistant" => {
                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    if content.is_empty() {
                        // Might be a tool-call-only turn; skip if no text content
                        None
                    } else {
                        Some(format!("Assistant: {}", content))
                    }
                }
                "tool" => {
                    let tool_name = msg
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("unknown");
                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    let summary = if content.len() > 200 {
                        format!("{}...", &content[..floor_char_boundary(content, 200)])
                    } else {
                        content.to_string()
                    };
                    Some(format!("Tool: {} -> {}", tool_name, summary))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

const COMPACTION_SYSTEM_PROMPT: &str =
    "You are a conversation summarizer. Produce concise summaries that preserve specific details.";

const INITIAL_COMPACTION_USER_TEMPLATE: &str = "\
Summarize this conversation in 1500 tokens or fewer.

Preserve with exact values:
- Names, IDs, numbers, file paths, URLs
- Decisions made and their reasoning
- Task outcomes (success/failure and what was produced)
- Identity-critical information about the user

Drop:
- Conversational filler and politeness
- Detailed reasoning that led to decisions (keep the decision, drop the deliberation)
- Failed attempts that were later corrected

Conversation:
";

const INCREMENTAL_COMPACTION_USER_TEMPLATE: &str = "\
Here is the existing conversation summary and new conversation turns.
Update the summary to include the new information. Stay under 1500 tokens.
Preserve names, IDs, numbers, decisions, outcomes.

Existing summary:
";

/// Build the messages array for a compaction LLM call.
///
/// Two modes:
/// - **Initial** (no existing summary): system prompt + user prompt with all messages
/// - **Incremental** (has existing summary): system prompt + user prompt with summary + new messages
pub(crate) fn build_compaction_prompt(
    existing_summary: Option<&str>,
    messages_to_compact: &[Value],
) -> Vec<Value> {
    let formatted = format_messages_for_prompt(messages_to_compact);

    let user_content = match existing_summary {
        Some(summary) => {
            format!(
                "{}{}\n\nNew turns:\n{}",
                INCREMENTAL_COMPACTION_USER_TEMPLATE, summary, formatted
            )
        }
        None => {
            format!("{}{}", INITIAL_COMPACTION_USER_TEMPLATE, formatted)
        }
    };

    vec![
        json!({ "role": "system", "content": COMPACTION_SYSTEM_PROMPT }),
        json!({ "role": "user", "content": user_content }),
    ]
}

/// Run the compaction LLM call and return the summary text.
#[allow(dead_code)]
pub(crate) async fn run_compaction(
    provider: Arc<dyn ModelProvider>,
    router: &Router,
    existing_summary: Option<&str>,
    messages_to_compact: &[Value],
) -> anyhow::Result<String> {
    let model = router.select(Tier::Primary).to_string();
    let messages = build_compaction_prompt(existing_summary, messages_to_compact);

    let result = match tokio::time::timeout(
        std::time::Duration::from_secs(15),
        provider.chat(&model, &messages, &[]),
    )
    .await
    {
        Ok(Ok(response)) => response,
        Ok(Err(e)) => {
            warn!(error = %e, "Compaction LLM call failed");
            return Err(e);
        }
        Err(_) => {
            warn!("Compaction LLM call timed out (15s)");
            return Err(anyhow::anyhow!("Compaction LLM call timed out"));
        }
    };

    let content = result.content.unwrap_or_default().trim().to_string();

    if content.is_empty() {
        return Err(anyhow::anyhow!("Compaction LLM returned empty content"));
    }

    Ok(content)
}

/// Run compaction and persist the resulting summary.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) async fn run_and_store_compaction(
    provider: Arc<dyn ModelProvider>,
    router: &Router,
    state: &dyn StateStore,
    session_id: &str,
    existing_summary: Option<ConversationSummary>,
    messages_to_compact: &[Value],
    new_message_count: usize,
    last_message_id: &str,
) -> anyhow::Result<()> {
    let existing_text = existing_summary.as_ref().map(|s| s.summary.as_str());

    let summary_text = run_compaction(provider, router, existing_text, messages_to_compact).await?;

    let summary = ConversationSummary {
        session_id: session_id.to_string(),
        summary: summary_text,
        message_count: new_message_count,
        last_message_id: last_message_id.to_string(),
        updated_at: Utc::now(),
    };

    state.upsert_conversation_summary(&summary).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── trigger detection ──────────────────────────────────────────

    #[test]
    fn test_detect_idle_gap() {
        let result = detect_compaction_trigger(3, 5, 7201, "hello");
        assert_eq!(result, Some(CompactionTrigger::IdleGap));
    }

    #[test]
    fn test_detect_file_upload_no_reference() {
        let result =
            detect_compaction_trigger(3, 5, 0, "[File received: photo.jpg] check this out");
        assert_eq!(result, Some(CompactionTrigger::FileUpload));
    }

    #[test]
    fn test_detect_file_upload_with_reference() {
        let result = detect_compaction_trigger(
            3,
            5,
            0,
            "[File received: config.yml] here's the file we discussed",
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_detect_window_overflow() {
        let result = detect_compaction_trigger(6, 5, 0, "hello");
        assert!(matches!(
            result,
            Some(CompactionTrigger::WindowOverflow { .. })
        ));
    }

    #[test]
    fn test_detect_no_trigger() {
        let result = detect_compaction_trigger(3, 5, 0, "hello");
        assert_eq!(result, None);
    }

    #[test]
    fn test_idle_gap_takes_priority_over_overflow() {
        // Both conditions met: idle gap (7201s) AND window overflow (6 > 5)
        let result = detect_compaction_trigger(6, 5, 7201, "hello");
        assert_eq!(result, Some(CompactionTrigger::IdleGap));
    }

    // ── pending compaction ─────────────────────────────────────────

    #[test]
    fn test_pending_cap_at_3() {
        let mut pending = PendingCompaction::new();
        assert!(pending.add("msg-1".to_string()).is_none());
        assert!(pending.add("msg-2".to_string()).is_none());
        assert!(pending.add("msg-3".to_string()).is_none());
        // 4th add exceeds cap — oldest is force-dropped
        let dropped = pending.add("msg-4".to_string());
        assert_eq!(dropped, Some("msg-1".to_string()));
        assert_eq!(pending.len(), 3);
        assert_eq!(pending.pair_ids, vec!["msg-2", "msg-3", "msg-4"]);
    }

    #[test]
    fn test_pending_drain_completed() {
        let mut pending = PendingCompaction::new();
        pending.add("msg-1".to_string());
        pending.add("msg-2".to_string());
        pending.add("msg-3".to_string());

        pending.drain_completed("msg-2");
        assert_eq!(pending.len(), 2);
        assert_eq!(pending.pair_ids, vec!["msg-1", "msg-3"]);
    }

    // ── prompt building ────────────────────────────────────────────

    #[test]
    fn test_build_compaction_prompt_initial() {
        let messages = vec![
            json!({ "role": "user", "content": "What time is it?" }),
            json!({ "role": "assistant", "content": "It is 3pm." }),
        ];

        let prompt = build_compaction_prompt(None, &messages);
        assert_eq!(prompt.len(), 2);

        // System message
        let system = prompt[0]["content"].as_str().unwrap();
        assert!(system.contains("conversation summarizer"));

        // User message — initial mode, no existing summary
        let user = prompt[1]["content"].as_str().unwrap();
        assert!(user.contains("Summarize this conversation"));
        assert!(user.contains("User: What time is it?"));
        assert!(user.contains("Assistant: It is 3pm."));
        assert!(!user.contains("Existing summary"));
    }

    #[test]
    fn test_build_compaction_prompt_incremental() {
        let messages = vec![
            json!({ "role": "user", "content": "Now deploy it." }),
            json!({ "role": "assistant", "content": "Deploying now." }),
        ];

        let existing = "User asked about project setup. Created config.toml.";
        let prompt = build_compaction_prompt(Some(existing), &messages);
        assert_eq!(prompt.len(), 2);

        // System message
        let system = prompt[0]["content"].as_str().unwrap();
        assert!(system.contains("conversation summarizer"));

        // User message — incremental mode, has existing summary
        let user = prompt[1]["content"].as_str().unwrap();
        assert!(user.contains("existing conversation summary"));
        assert!(user.contains("Existing summary:"));
        assert!(user.contains(existing));
        assert!(user.contains("New turns:"));
        assert!(user.contains("User: Now deploy it."));
        assert!(user.contains("Assistant: Deploying now."));
        assert!(!user.contains("Summarize this conversation"));
    }
}
