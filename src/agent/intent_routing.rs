use regex::Regex;

use super::{IntentGateDecision, ENABLE_SCHEDULE_HEURISTICS};

/// Complexity classification for V3 orchestration routing.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum IntentComplexity {
    /// Answer from memory/knowledge, no executor needed.
    Knowledge,
    /// Simple task — falls through to full agent loop.
    Simple,
    /// Multi-step complex task, create a V3 goal and fall through to current agent loop.
    Complex,
    /// User asks for recurring/ongoing behavior but did not provide timing.
    ScheduledMissingTiming,
    /// Scheduled task intent requiring deferred/recurring goal creation.
    Scheduled {
        schedule_raw: String,
        schedule_cron: Option<String>,
        is_one_shot: bool,
        schedule_type_explicit: bool,
    },
}

/// Check if a phrase appears as complete words in text (word-boundary matching).
/// Splits on whitespace, trims surrounding punctuation (preserving apostrophes),
/// then checks for consecutive word matches. Case-insensitive.
///
/// Works for single keywords ("deploy"), multi-word phrases ("set up"),
/// and contractions ("i'll check").
pub(super) fn contains_keyword_as_words(text: &str, keyword: &str) -> bool {
    let normalize = |w: &str| -> String {
        w.trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'')
            .to_lowercase()
    };
    let text_words: Vec<String> = text
        .split_whitespace()
        .map(normalize)
        .filter(|w| !w.is_empty())
        .collect();
    let kw_words: Vec<String> = keyword
        .split_whitespace()
        .map(normalize)
        .filter(|w| !w.is_empty())
        .collect();
    if kw_words.is_empty() {
        return false;
    }
    text_words
        .windows(kw_words.len())
        .any(|window| window == kw_words.as_slice())
}

/// Detect "about this scheduled goal" meta-queries so they aren't misread as
/// fresh scheduling requests when quoted text contains timing words.
fn is_schedule_reference_query(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let mentions_schedule_subject = contains_keyword_as_words(&lower, "scheduled goal")
        || contains_keyword_as_words(&lower, "recurring goal")
        || contains_keyword_as_words(&lower, "schedule");

    if !mentions_schedule_subject {
        return false;
    }

    contains_keyword_as_words(&lower, "details about")
        || contains_keyword_as_words(&lower, "tell me about")
        || contains_keyword_as_words(&lower, "show me")
        || contains_keyword_as_words(&lower, "list")
        || contains_keyword_as_words(&lower, "what is")
        || contains_keyword_as_words(&lower, "what's")
        || contains_keyword_as_words(&lower, "explain")
        || contains_keyword_as_words(&lower, "describe")
        || lower.contains("scheduled goal:")
}

/// Detect obvious scheduling phrases in user text as a fallback when the model
/// omits schedule fields in [INTENT_GATE].
///
/// Returns (schedule_raw, is_one_shot).
pub(super) fn detect_schedule_heuristic(user_text: &str) -> Option<(String, bool)> {
    let text = user_text.trim();
    if text.is_empty() {
        return None;
    }
    if is_schedule_reference_query(text) {
        return None;
    }
    let lower = text.to_ascii_lowercase();

    // One-shot relative time, e.g. "in 2h", "in 30 minutes"
    let re_in_time =
        Regex::new(r"(?i)\bin\s+\d+\s*(?:m|min|mins|minutes?|h|hrs?|hours?)\b").ok()?;
    if let Some(m) = re_in_time.find(text) {
        return Some((m.as_str().trim().to_string(), true));
    }

    // One-shot absolute tomorrow time, e.g. "tomorrow at 9am"
    let re_tomorrow_at =
        Regex::new(r"(?i)\btomorrow\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b").ok()?;
    if let Some(m) = re_tomorrow_at.find(text) {
        return Some((m.as_str().trim().to_string(), true));
    }

    // One-shot today/tonight absolute time, optional timezone token.
    let re_today_tonight_at = Regex::new(
        r"(?i)\b(?:today|tonight)\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?(?:\s+[A-Za-z]{1,8}|[+-]\d{2}:?\d{2}|Z)?\b",
    )
    .ok()?;
    if let Some(m) = re_today_tonight_at.find(text) {
        return Some((m.as_str().trim().to_string(), true));
    }

    // One-shot tomorrow without explicit time (will likely require clarification later).
    if contains_keyword_as_words(&lower, "tomorrow") {
        return Some(("tomorrow".to_string(), true));
    }

    // Recurring intervals: "every 6h", "every 30m", "each 5 minutes"
    let re_every_interval =
        Regex::new(r"(?i)\b(?:every|each)\s+\d+\s*(?:m|min|mins|minutes?|h|hrs?|hours?)\b").ok()?;
    if let Some(m) = re_every_interval.find(text) {
        return Some((m.as_str().trim().to_string(), false));
    }

    // Recurring named schedules
    let recurring_patterns = [
        r"(?i)\bdaily\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
        r"(?i)\bweekdays?\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
        r"(?i)\bweekends?\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
    ];
    for pattern in recurring_patterns {
        let re = Regex::new(pattern).ok()?;
        if let Some(m) = re.find(text) {
            return Some((m.as_str().trim().to_string(), false));
        }
    }

    for kw in ["hourly", "daily", "weekly", "monthly"] {
        if contains_keyword_as_words(&lower, kw) {
            return Some((kw.to_string(), false));
        }
    }

    None
}

/// Detect recurring-intent language when the user did not provide concrete timing.
/// Used to prevent accidental fallback into non-recurring "complex" goals.
pub(super) fn looks_like_recurring_intent_without_timing(user_text: &str) -> bool {
    if detect_schedule_heuristic(user_text).is_some() {
        return false;
    }

    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let re_times_per = match Regex::new(r"(?i)\b\d+\s+times?\s+per\s+(day|week|month)\b") {
        Ok(re) => re,
        Err(_) => return false,
    };
    if re_times_per.is_match(user_text) {
        return true;
    }

    for kw in [
        "monitor",
        "recurring",
        "ongoing",
        "long-term",
        "long term",
        "regularly",
        "consistently",
        "every day",
        "each day",
        "per day",
        "per week",
        "per month",
    ] {
        if contains_keyword_as_words(&lower, kw) {
            return true;
        }
    }

    false
}

/// Detect legacy internal-maintenance intents that should run via native
/// heartbeat/memory jobs rather than goal orchestration.
pub(super) fn is_internal_maintenance_intent(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    if lower == "maintain knowledge base: process embeddings, consolidate memories, decay old facts"
        || lower
            == "maintain memory health: prune old events, clean up retention, remove stale data"
    {
        return true;
    }

    let knowledge_maintenance = contains_keyword_as_words(&lower, "process embeddings")
        && contains_keyword_as_words(&lower, "consolidate memories")
        && (contains_keyword_as_words(&lower, "decay old facts")
            || contains_keyword_as_words(&lower, "memory decay"));

    let memory_health = contains_keyword_as_words(&lower, "prune old events")
        && (contains_keyword_as_words(&lower, "clean up retention")
            || contains_keyword_as_words(&lower, "retention cleanup"))
        && contains_keyword_as_words(&lower, "stale data");

    knowledge_maintenance || memory_health
}

pub(super) fn infer_intent_gate(user_text: &str, _analysis: &str) -> IntentGateDecision {
    // If the user's message contains a filesystem path, the request almost
    // certainly requires tool access (terminal) — override the consultant's
    // analysis and route to the tool loop directly.
    if super::user_text_references_filesystem_path(user_text) {
        return IntentGateDecision {
            can_answer_now: Some(false),
            needs_tools: Some(true),
            needs_clarification: Some(false),
            clarifying_question: None,
            missing_info: Vec::new(),
            complexity: None,
            cancel_intent: None,
            cancel_scope: None,
            is_acknowledgment: None,
            schedule: None,
            schedule_type: None,
            schedule_cron: None,
            domains: Vec::new(),
        };
    }

    // No lexical guessing fallback: rely on explicit model intent-gate fields.
    // Missing fields simply stay None.
    IntentGateDecision {
        can_answer_now: None,
        needs_tools: None,
        needs_clarification: None,
        clarifying_question: None,
        missing_info: Vec::new(),
        complexity: None,
        cancel_intent: None,
        cancel_scope: None,
        is_acknowledgment: None,
        schedule: None,
        schedule_type: None,
        schedule_cron: None,
        domains: Vec::new(),
    }
}

#[cfg(test)]
mod intent_routing_path_override_tests {
    use super::*;

    #[test]
    fn infer_intent_gate_does_not_force_tools_for_urls() {
        let d = infer_intent_gate("https://example.com/foo/bar", "");
        assert!(
            d.needs_tools.is_none(),
            "expected needs_tools=None for URL, got {:?}",
            d.needs_tools
        );
    }

    #[test]
    fn infer_intent_gate_does_not_force_tools_for_common_slash_shorthand() {
        for text in ["3/4", "2/14", "yes/no", "w/o"] {
            let d = infer_intent_gate(text, "");
            assert!(
                d.needs_tools.is_none(),
                "expected needs_tools=None for '{}', got {:?}",
                text,
                d.needs_tools
            );
        }
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_unix_paths() {
        let d = infer_intent_gate("/Users/alice/project/file.txt", "");
        assert_eq!(d.needs_tools, Some(true));
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_tilde_paths() {
        let d = infer_intent_gate("~/project/file.txt", "");
        assert_eq!(d.needs_tools, Some(true));
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_windows_paths() {
        let d = infer_intent_gate(r"C:\Users\alice\file.txt", "");
        assert_eq!(d.needs_tools, Some(true));
    }
}

/// Classify user intent complexity for V3 routing.
///
/// Uses the LLM-provided `complexity` field from the `[INTENT_GATE]` JSON.
/// Falls back to `Simple` when the field is absent or unrecognized.
///
/// Guardrails override the LLM's "complex" classification for messages that
/// are clearly simple — the consultant LLM over-classifies short commands,
/// acknowledgments, and single-action requests as complex.
pub(super) fn classify_intent_complexity(
    user_text: &str,
    intent_gate: &IntentGateDecision,
) -> (IntentComplexity, Vec<String>) {
    // If user clearly wants recurring behavior but no timing could be extracted,
    // ask for schedule details instead of silently creating a non-recurring goal.
    if ENABLE_SCHEDULE_HEURISTICS
        && intent_gate.schedule.is_none()
        && intent_gate.schedule_cron.is_none()
        && looks_like_recurring_intent_without_timing(user_text)
    {
        return (IntentComplexity::ScheduledMissingTiming, vec![]);
    }

    // Schedule takes priority over all other classifications.
    if let Some(ref schedule_raw) = intent_gate.schedule {
        let schedule_type_explicit = intent_gate.schedule_type.is_some();
        let is_one_shot = intent_gate.schedule_type.as_deref() == Some("one_shot");
        return (
            IntentComplexity::Scheduled {
                schedule_raw: schedule_raw.clone(),
                schedule_cron: intent_gate.schedule_cron.clone(),
                is_one_shot,
                schedule_type_explicit,
            },
            vec![],
        );
    }
    if let Some(ref schedule_cron) = intent_gate.schedule_cron {
        let schedule_type_explicit = intent_gate.schedule_type.is_some();
        let is_one_shot = intent_gate.schedule_type.as_deref() == Some("one_shot");
        return (
            IntentComplexity::Scheduled {
                schedule_raw: schedule_cron.clone(),
                schedule_cron: Some(schedule_cron.clone()),
                is_one_shot,
                schedule_type_explicit,
            },
            vec![],
        );
    }

    if intent_gate.can_answer_now.unwrap_or(false) && !intent_gate.needs_tools.unwrap_or(false) {
        return (IntentComplexity::Knowledge, vec![]);
    }
    // When can_answer_now=false, don't classify as Knowledge even if
    // complexity="knowledge" — the model can't answer, so we should
    // try tools (memory search, manage_people, etc.) as Simple.
    match intent_gate.complexity.as_deref() {
        Some("knowledge") => (IntentComplexity::Simple, vec![]),
        Some("complex") => (IntentComplexity::Complex, vec![]),
        _ => (IntentComplexity::Simple, vec![]),
    }
}
