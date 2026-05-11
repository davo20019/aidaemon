//! LLM-backed intent classifier (shadow-mode scaffolding).
//!
//! This module exposes a fast-model classification call that returns a
//! coarse-grained intent class for a user message. It is *not* wired into
//! the agent's decision path. Its purpose is to run alongside the existing
//! keyword-based heuristics so we can compare outputs and gather evidence
//! before deciding whether (and where) to replace regex-based intent
//! detection with an LLM call.
//!
//! ## Why this exists
//!
//! Intent detection in this codebase is keyword-based: a series of helpers
//! like `detect_schedule_heuristic`, `is_memory_storage_intent`, and
//! `looks_like_personal_memory_recall_question` walk shared keyword lists
//! and combine them with simple logic. The lists are now centralized (see
//! `intent_keywords.rs`), but adding edge cases still requires patching
//! regex/keyword logic in code.
//!
//! An LLM classifier could in principle handle the long tail of phrasings
//! that regex misses, at the cost of latency, money, and a new failure mode.
//! Whether that trade is worth it is an empirical question. This module
//! provides the measurement instrument.
//!
//! ## Shape
//!
//! - [`LlmIntentClass`] — coarse-grained classes mirroring the existing
//!   heuristic outputs (schedule, memory storage, memory recall, action,
//!   knowledge question, other).
//! - [`classify_intent`] — async function that calls a fast model with a
//!   compact prompt and parses a structured response. Fail-open: any
//!   transport/parse error yields `LlmIntentClass::Unknown` rather than
//!   propagating, so a shadow caller never blocks on classifier failure.
//! - [`log_intent_disagreement`] — convenience helper for shadow callers
//!   that observe a heuristic-vs-LLM disagreement.
//!
//! ## Wiring
//!
//! As of v0.9.34, no code path calls these functions in production. They
//! are designed to be plugged in behind a config flag (default off) in a
//! follow-up release. The module is shipped now so the classifier itself
//! can be reviewed and tested independently of the integration risk.

use std::time::Duration;

use serde_json::{json, Value};
use tracing::{debug, info};

use crate::traits::{ChatOptions, ModelProvider, StateStore};
use std::sync::Arc;

/// Output cap for classifier calls. A label is a single short word; 20 tokens
/// is plenty and keeps the per-call cost (and per-call affordability check
/// on metered providers like OpenRouter) tiny.
const CLASSIFIER_MAX_OUTPUT_TOKENS: u32 = 20;

/// Hard timeout for a classifier call. Keeps the shadow path from holding
/// onto the request thread even when the fast model is slow or stuck.
const CLASSIFIER_TIMEOUT: Duration = Duration::from_secs(5);

/// Coarse-grained intent classes the classifier can return.
///
/// These mirror the buckets the keyword heuristics already produce, so a
/// caller running both can compare apples to apples. `Unknown` is the
/// fail-open value: it means "the classifier did not return a usable
/// answer", not "the message has no intent".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmIntentClass {
    /// Schedule a one-time future action (e.g., "remind me at 5pm").
    ScheduleOneShot,
    /// Schedule a recurring action (e.g., "every Monday at 9am").
    ScheduleRecurring,
    /// Store a fact in memory (e.g., "remember my birthday is October 15").
    MemoryStorage,
    /// Recall personal information already stored (e.g., "what's my coffee?").
    MemoryRecall,
    /// Request an action that requires non-memory tools (write, run, search).
    Action,
    /// Pure knowledge / question request that the model can answer directly.
    KnowledgeQuestion,
    /// None of the above, or a mix that doesn't reduce cleanly.
    Other,
    /// Classifier call failed or returned an unparseable response. Callers
    /// should fall back to heuristics.
    Unknown,
}

impl LlmIntentClass {
    /// Parse the classifier's textual answer into a class. Accepts the
    /// snake_case strings the prompt instructs the model to return.
    pub(crate) fn from_response_str(s: &str) -> Self {
        match s.trim().trim_matches('"').to_ascii_lowercase().as_str() {
            "schedule_one_shot" => Self::ScheduleOneShot,
            "schedule_recurring" => Self::ScheduleRecurring,
            "memory_storage" => Self::MemoryStorage,
            "memory_recall" => Self::MemoryRecall,
            "action" => Self::Action,
            "knowledge_question" => Self::KnowledgeQuestion,
            "other" => Self::Other,
            _ => Self::Unknown,
        }
    }

    /// Stable string label for logging and disagreement reports.
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::ScheduleOneShot => "schedule_one_shot",
            Self::ScheduleRecurring => "schedule_recurring",
            Self::MemoryStorage => "memory_storage",
            Self::MemoryRecall => "memory_recall",
            Self::Action => "action",
            Self::KnowledgeQuestion => "knowledge_question",
            Self::Other => "other",
            Self::Unknown => "unknown",
        }
    }
}

/// Build the messages array for a classification call. Kept separate from
/// `classify_intent` so the prompt can be unit-tested without a provider.
pub(crate) fn build_classifier_messages(user_text: &str) -> Vec<Value> {
    // The prompt is deliberately short. Latency and cost scale with prompt
    // size; the whole point of using a fast model is to stay cheap.
    let system = "You are an intent classifier. Read the user's message and \
                  return exactly one label (no prose, no explanation, no JSON \
                  wrapping). Valid labels:\n\
                  - schedule_one_shot: trigger a one-time future action\n\
                  - schedule_recurring: trigger a repeating action\n\
                  - memory_storage: store/remember a fact about the user\n\
                  - memory_recall: recall a fact already stored\n\
                  - action: run a tool, write code, search, browse, etc.\n\
                  - knowledge_question: answer a question from general knowledge\n\
                  - other: doesn't fit any category, or is a mix\n\
                  Respond with the label only.";

    vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": user_text}),
    ]
}

/// Run the classifier. Fail-open: any error returns `LlmIntentClass::Unknown`
/// so callers in shadow mode never propagate transient classifier failures.
///
/// `state` is optional; when provided, token usage is recorded under the
/// `background:intent_classifier` key so we can audit shadow-mode cost.
#[allow(dead_code)] // shadow scaffolding — wired in a follow-up release
pub async fn classify_intent(
    provider: &dyn ModelProvider,
    fast_model: &str,
    user_text: &str,
    state: Option<&Arc<dyn StateStore>>,
) -> LlmIntentClass {
    let trimmed = user_text.trim();
    if trimmed.is_empty() {
        return LlmIntentClass::Other;
    }

    let messages = build_classifier_messages(trimmed);

    let options = ChatOptions {
        max_tokens_override: Some(CLASSIFIER_MAX_OUTPUT_TOKENS),
        ..ChatOptions::default()
    };
    let call = provider.chat_with_options(fast_model, &messages, &[], &options);
    let response = match tokio::time::timeout(CLASSIFIER_TIMEOUT, call).await {
        Ok(Ok(r)) => r,
        Ok(Err(err)) => {
            debug!(?err, "intent classifier call failed; failing open");
            return LlmIntentClass::Unknown;
        }
        Err(_) => {
            debug!(
                timeout_s = CLASSIFIER_TIMEOUT.as_secs(),
                "intent classifier timeout"
            );
            return LlmIntentClass::Unknown;
        }
    };

    if let (Some(state), Some(usage)) = (state, &response.usage) {
        let _ = state
            .record_token_usage("background:intent_classifier", usage)
            .await;
    }

    match response.content.as_deref() {
        Some(text) => LlmIntentClass::from_response_str(text),
        None => LlmIntentClass::Unknown,
    }
}

/// Convenience helper for shadow callers: log when the LLM classifier and
/// the heuristic disagree. Keeps log shape consistent so we can grep for
/// `intent_disagreement` and tally results later.
///
/// `heuristic_label` and `llm_label` are free-form strings so callers can
/// pass whatever labels their heuristic produces (the heuristic vocabulary
/// is not as clean as the LLM enum).
#[allow(dead_code)] // shadow scaffolding — wired in a follow-up release
pub fn log_intent_disagreement(user_text: &str, heuristic_label: &str, llm: LlmIntentClass) {
    if llm == LlmIntentClass::Unknown {
        // Classifier returned no signal — not a disagreement, just a miss.
        return;
    }
    if heuristic_label == llm.as_label() {
        return;
    }
    info!(
        event = "intent_disagreement",
        heuristic = heuristic_label,
        llm = llm.as_label(),
        // Truncate to keep log lines bounded; full text is recoverable
        // from message events if needed.
        user_text_preview = %crate::utils::truncate_str(user_text, 200),
        "heuristic and LLM intent classifiers disagree"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockProvider;

    #[test]
    fn parses_known_labels() {
        assert_eq!(
            LlmIntentClass::from_response_str("schedule_one_shot"),
            LlmIntentClass::ScheduleOneShot
        );
        assert_eq!(
            LlmIntentClass::from_response_str("MEMORY_STORAGE"),
            LlmIntentClass::MemoryStorage
        );
        assert_eq!(
            LlmIntentClass::from_response_str(" action "),
            LlmIntentClass::Action
        );
        assert_eq!(
            LlmIntentClass::from_response_str("\"knowledge_question\""),
            LlmIntentClass::KnowledgeQuestion
        );
    }

    #[test]
    fn rejects_unknown_labels_as_unknown() {
        assert_eq!(
            LlmIntentClass::from_response_str("garbage"),
            LlmIntentClass::Unknown
        );
        assert_eq!(
            LlmIntentClass::from_response_str(""),
            LlmIntentClass::Unknown
        );
        // The model might wrap in JSON despite instructions — that's a fail
        // case, not a panic. Caller fails open.
        assert_eq!(
            LlmIntentClass::from_response_str("{\"label\":\"action\"}"),
            LlmIntentClass::Unknown
        );
    }

    #[test]
    fn label_roundtrip_is_stable() {
        // Every non-Unknown class must round-trip through as_label →
        // from_response_str so log entries can be parsed back if needed.
        for class in [
            LlmIntentClass::ScheduleOneShot,
            LlmIntentClass::ScheduleRecurring,
            LlmIntentClass::MemoryStorage,
            LlmIntentClass::MemoryRecall,
            LlmIntentClass::Action,
            LlmIntentClass::KnowledgeQuestion,
            LlmIntentClass::Other,
        ] {
            assert_eq!(
                LlmIntentClass::from_response_str(class.as_label()),
                class,
                "round-trip failed for {class:?}"
            );
        }
    }

    #[test]
    fn message_shape_contains_user_text_and_label_vocabulary() {
        let messages = build_classifier_messages("remind me at 5pm");
        assert_eq!(messages.len(), 2);

        let system = messages[0].get("content").and_then(|c| c.as_str()).unwrap();
        // Sanity-check that every enum label appears verbatim in the
        // prompt. If a label is renamed without updating the prompt, the
        // model will produce something we can't parse.
        for label in [
            "schedule_one_shot",
            "schedule_recurring",
            "memory_storage",
            "memory_recall",
            "action",
            "knowledge_question",
            "other",
        ] {
            assert!(system.contains(label), "prompt is missing label {label:?}");
        }

        let user = messages[1].get("content").and_then(|c| c.as_str()).unwrap();
        assert_eq!(user, "remind me at 5pm");
    }

    #[tokio::test]
    async fn classify_returns_parsed_label_on_success() {
        let provider =
            MockProvider::with_responses(vec![MockProvider::text_response("memory_storage")]);
        let result = classify_intent(&provider, "fast-model", "remember my birthday", None).await;
        assert_eq!(result, LlmIntentClass::MemoryStorage);
    }

    #[tokio::test]
    async fn classify_fails_open_on_unparseable_response() {
        let provider =
            MockProvider::with_responses(vec![MockProvider::text_response("not a label")]);
        let result = classify_intent(&provider, "fast-model", "do something weird", None).await;
        assert_eq!(result, LlmIntentClass::Unknown);
    }

    #[tokio::test]
    async fn classify_short_circuits_on_empty_input() {
        // Mock will record a call if invoked; verify it isn't.
        let provider = MockProvider::with_responses(vec![]);
        let result = classify_intent(&provider, "fast-model", "   ", None).await;
        assert_eq!(result, LlmIntentClass::Other);
        assert_eq!(provider.call_count().await, 0);
    }

    #[tokio::test]
    async fn classify_handles_provider_error_by_failing_open() {
        // Empty response queue → provider returns default "Mock response",
        // which doesn't parse as any label.
        let provider = MockProvider::new();
        let result = classify_intent(&provider, "fast-model", "anything", None).await;
        assert_eq!(result, LlmIntentClass::Unknown);
    }

    #[test]
    fn disagreement_log_is_silent_on_match() {
        // No assertion possible on tracing output without a subscriber,
        // but the call must not panic and must not log for the no-op
        // cases. Smoke test for the early-return branches.
        log_intent_disagreement("do something", "action", LlmIntentClass::Action);
        log_intent_disagreement("anything", "action", LlmIntentClass::Unknown);
    }
}
