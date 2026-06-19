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

use crate::events::EventStore;
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

/// Coarse relational-recall classification used by neighborhood assembly and
/// the search-before-deny gate. Separate from `LlmIntentClass` because both
/// consumers need the *entities* the query names, which the coarse class lacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationalKind {
    /// A question about a relationship/connection between entities
    /// ("who is Caro's spouse?", "what tools does project X use?").
    Relational,
    /// A direct personal-fact recall ("what's my dog's name?").
    Recall,
    /// Neither — do nothing.
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalIntent {
    pub kind: RelationalKind,
    /// Entities the query is about, as the model named them. Possibly empty.
    pub entities: Vec<String>,
}

impl RelationalIntent {
    fn none() -> Self {
        Self {
            kind: RelationalKind::None,
            entities: Vec::new(),
        }
    }
}

/// Parse the classifier's JSON reply. Fail-open: any malformed input yields
/// `RelationalKind::None` with no entities (caller then does nothing).
pub fn parse_relational_intent(raw: &str) -> RelationalIntent {
    // Extract the first {...} span so ```json fences / prose don't break parsing.
    let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}')) else {
        return RelationalIntent::none();
    };
    if end < start {
        return RelationalIntent::none();
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw[start..=end]) else {
        return RelationalIntent::none();
    };
    let kind = match v.get("intent").and_then(|i| i.as_str()).unwrap_or("none") {
        "relational" => RelationalKind::Relational,
        "recall" => RelationalKind::Recall,
        _ => RelationalKind::None,
    };
    let entities = v
        .get("entities")
        .and_then(|e| e.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    RelationalIntent { kind, entities }
}

/// Build the messages array for a relational-intent classification call.
/// Kept separate from `classify_relational_intent` so the prompt can be
/// unit-tested without a provider.
fn build_relational_classifier_messages(user_text: &str) -> Vec<Value> {
    let system = "You classify a user message about their personal memory. \
Reply with ONLY a JSON object: {\"intent\": \"relational\"|\"recall\"|\"none\", \"entities\": [..]}. \
\"relational\" = a question about a relationship/connection between entities (e.g. \"who is Caro's spouse?\", \"who is my kid's mom?\", \"what tools does project X use?\"). \
\"recall\" = a direct fact lookup about one entity (e.g. \"what's my dog's name?\"). \
\"none\" = anything else (general knowledge, chit-chat, actions). \
\"entities\" = the people/projects/things the question is about, as named (resolve possessives to the owned entity: \"my mom\" -> \"my mom\"). Keep it short.";
    vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": user_text}),
    ]
}

/// Classify a message for relational/recall intent and extract its entities.
/// Fail-open: empty input, provider error, or timeout yields `RelationalKind::None`.
pub async fn classify_relational_intent(
    provider: &dyn ModelProvider,
    fast_model: &str,
    user_text: &str,
) -> RelationalIntent {
    let trimmed = user_text.trim();
    if trimmed.is_empty() {
        return RelationalIntent::none();
    }
    let messages = build_relational_classifier_messages(trimmed);
    let options = ChatOptions {
        max_tokens_override: Some(120),
        ..ChatOptions::default()
    };
    let call = provider.chat_with_options(fast_model, &messages, &[], &options);
    let response = match tokio::time::timeout(CLASSIFIER_TIMEOUT, call).await {
        Ok(Ok(r)) => r,
        Ok(Err(err)) => {
            debug!(?err, "relational classifier call failed; failing open");
            return RelationalIntent::none();
        }
        Err(_) => {
            debug!(
                timeout_s = CLASSIFIER_TIMEOUT.as_secs(),
                "relational classifier timeout"
            );
            return RelationalIntent::none();
        }
    };
    parse_relational_intent(response.content.as_deref().unwrap_or(""))
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
    event_store: Option<Arc<EventStore>>,
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
    let call_start = std::time::Instant::now();
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

    if let (Some(state), Some(event_store)) = (state, event_store) {
        crate::events::record_background_model_call_telemetry(
            event_store,
            state.as_ref(),
            "background:intent_classifier",
            "intent_classifier",
            fast_model,
            &response,
            call_start.elapsed(),
        )
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
        let result =
            classify_intent(&provider, "fast-model", "remember my birthday", None, None).await;
        assert_eq!(result, LlmIntentClass::MemoryStorage);
    }

    #[tokio::test]
    async fn classify_fails_open_on_unparseable_response() {
        let provider =
            MockProvider::with_responses(vec![MockProvider::text_response("not a label")]);
        let result =
            classify_intent(&provider, "fast-model", "do something weird", None, None).await;
        assert_eq!(result, LlmIntentClass::Unknown);
    }

    #[tokio::test]
    async fn classify_short_circuits_on_empty_input() {
        // Mock will record a call if invoked; verify it isn't.
        let provider = MockProvider::with_responses(vec![]);
        let result = classify_intent(&provider, "fast-model", "   ", None, None).await;
        assert_eq!(result, LlmIntentClass::Other);
        assert_eq!(provider.call_count().await, 0);
    }

    #[tokio::test]
    async fn classify_handles_provider_error_by_failing_open() {
        // Empty response queue → provider returns default "Mock response",
        // which doesn't parse as any label.
        let provider = MockProvider::new();
        let result = classify_intent(&provider, "fast-model", "anything", None, None).await;
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

    #[test]
    fn parse_relational_intent_reads_json() {
        let r = parse_relational_intent(r#"{"intent":"relational","entities":["Caro","Frank"]}"#);
        assert_eq!(r.kind, RelationalKind::Relational);
        assert_eq!(r.entities, vec!["Caro".to_string(), "Frank".to_string()]);
    }

    #[test]
    fn parse_relational_intent_tolerates_fencing_and_prose() {
        // Models often wrap JSON in ```json fences or add a sentence.
        let r = parse_relational_intent(
            "Sure!\n```json\n{\"intent\":\"recall\",\"entities\":[\"my dog\"]}\n```",
        );
        assert_eq!(r.kind, RelationalKind::Recall);
        assert_eq!(r.entities, vec!["my dog".to_string()]);
    }

    #[test]
    fn parse_relational_intent_fails_open_on_garbage() {
        let r = parse_relational_intent("not json at all");
        assert_eq!(r.kind, RelationalKind::None);
        assert!(r.entities.is_empty());
    }

    #[tokio::test]
    async fn classify_relational_intent_parses_provider_json() {
        let provider = crate::testing::MockProvider::with_responses(vec![
            crate::testing::MockProvider::text_response(
                r#"{"intent":"relational","entities":["Caro"]}"#,
            ),
        ]);
        let r = classify_relational_intent(&provider, "fast-model", "who is caro's spouse?").await;
        assert_eq!(r.kind, RelationalKind::Relational);
        assert_eq!(r.entities, vec!["Caro".to_string()]);
    }

    #[tokio::test]
    async fn classify_relational_intent_fails_open_on_empty_input() {
        let provider = crate::testing::MockProvider::new();
        let r = classify_relational_intent(&provider, "fast-model", "   ").await;
        assert_eq!(r.kind, RelationalKind::None);
    }
}
