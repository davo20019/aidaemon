use std::collections::HashMap;

use super::contains_keyword_as_words;
use crate::execution_policy::{score_risk_from_capabilities, PolicyBundle};
use crate::traits::ToolCapabilities;

#[allow(dead_code)] // Kept for potential future response/fallback handling.
pub(super) fn first_question_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| line.contains('?'))
        .map(|s| s.to_string())
}

pub(super) fn default_clarifying_question(_user_text: &str, missing_info: &[String]) -> String {
    if !missing_info.is_empty() {
        return format!(
            "Could you clarify {} so I can proceed correctly?",
            missing_info.join(", ")
        );
    }
    "Which exact target should I use for this action?".to_string()
}

pub(super) fn build_policy_bundle(
    _user_text: &str,
    available_capabilities: &HashMap<String, ToolCapabilities>,
    prior_immediate_failure: bool,
) -> PolicyBundle {
    // Bootstrap has tool capabilities and prior execution state, but it does
    // not yet have a proposed action. Do not manufacture risk, ambiguity, or
    // missing targets from request wording. Concrete calls are checked later
    // against typed capabilities, arguments, approvals, and receipts.
    let risk_score =
        score_risk_from_capabilities(&available_capabilities.values().copied().collect::<Vec<_>>());
    let uncertainty_score: f32 = if prior_immediate_failure { 0.65 } else { 0.0 };
    let confidence = (1.0 - uncertainty_score).clamp(0.0, 1.0);
    PolicyBundle::from_scores(risk_score, uncertainty_score, confidence)
}

pub(super) fn detect_explicit_outcome_signal(text: &str) -> Option<(&'static str, bool)> {
    let lower = text.trim().to_ascii_lowercase();
    // Feedback learning must not reinterpret a substantive new request just
    // because it contains a courtesy word (for example, "thanks to X, deploy
    // Y"). Restrict automatic labels to short feedback-shaped turns.
    if lower.chars().count() > 240 {
        return None;
    }

    let normalized = lower
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .trim();
    let exact_positives = ["thanks", "thank you", "perfect", "got it", "that worked"];
    let explicit_positive_result = [
        "that worked",
        "it worked",
        "works now",
        "looks good",
        "exactly what i needed",
    ]
    .iter()
    .any(|phrase| contains_keyword_as_words(normalized, phrase));
    if exact_positives.contains(&normalized) || explicit_positive_result {
        return Some(("positive", true));
    }
    let negatives = [
        "that's wrong",
        "try again",
        "not what i asked",
        "you misunderstood",
    ];
    if negatives
        .iter()
        .any(|phrase| contains_keyword_as_words(normalized, phrase))
    {
        return Some(("negative", false));
    }
    None
}

pub(super) fn tool_is_side_effecting(
    name: &str,
    capabilities: &HashMap<String, ToolCapabilities>,
) -> bool {
    !capabilities
        .get(name)
        .copied()
        .unwrap_or_default()
        .read_only
}

/// Returns true if the message is a trivial acknowledgment, greeting, or
/// single imperative command that should never be routed as Complex.
#[allow(dead_code)] // Kept for potential future guardrail handling.
pub(super) fn is_trivial_message(lower: &str) -> bool {
    let trivial_prefixes = [
        "ok",
        "okay",
        "sure",
        "thanks",
        "thank you",
        "thx",
        "got it",
        "cool",
        "great",
        "nice",
        "yes",
        "no",
        "yep",
        "nope",
        "alright",
        "sounds good",
        "perfect",
        "awesome",
        "good",
        "fine",
        "right",
        "hello",
        "hi",
        "hey",
    ];
    for prefix in &trivial_prefixes {
        if lower.starts_with(prefix) {
            // Exact match or followed by whitespace/punctuation
            if lower.len() == prefix.len()
                || lower
                    .as_bytes()
                    .get(prefix.len())
                    .is_some_and(|b| !b.is_ascii_alphanumeric())
            {
                return true;
            }
        }
    }
    false
}

/// Returns true for short corrective follow-ups (not new requests), e.g.
/// "you did send me the pdf". This is a deterministic guardrail when the
/// first-pass intent gate over-predicts `needs_tools=true`.
#[cfg(test)]
pub(super) fn is_short_user_correction(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() || lower.contains('?') {
        return false;
    }

    let word_count = lower.split_whitespace().count();
    if word_count > 14 {
        return false;
    }

    // If the user is clearly asking for a fresh action, this is not a correction-only turn.
    let request_prefixes = [
        "can you ",
        "could you ",
        "would you ",
        "please ",
        "run ",
        "check ",
        "find ",
        "create ",
        "generate ",
        "make ",
        "send ",
        "open ",
        "read ",
        "write ",
        "search ",
        "install ",
        "fix ",
        "debug ",
        "build ",
        "edit ",
        "move ",
        "copy ",
        "delete ",
        "retry ",
        "try again",
        "proceed",
    ];
    if request_prefixes.iter().any(|p| lower.starts_with(p)) {
        return false;
    }
    let request_phrases = [
        " can you ",
        " could you ",
        " would you ",
        " please ",
        " try again",
        " proceed",
        " go ahead",
        " check ",
        " verify ",
        " look it up",
        " look this up",
    ];
    if request_phrases.iter().any(|p| lower.contains(p)) {
        return false;
    }

    let correction_markers = [
        "you did",
        "you already",
        "you sent",
        "you have sent",
        "you did send",
        "i already",
        "i got",
        "i received",
        "that's right",
        "thats right",
        "correct",
        "exactly",
    ];
    correction_markers.iter().any(|m| lower.contains(m))
}

/// Returns true if the message is a list of immediate tool operations that can
/// be completed in a single agent session. These should be Simple, not Complex.
#[allow(dead_code)] // Kept for potential future guardrail handling.
pub(super) fn is_sequential_tool_request(lower: &str) -> bool {
    // Check for numbered list patterns (1), 2), 3) or 1. 2. 3.)
    let has_numbered_steps = lower.contains("1)") || lower.contains("1.");
    if !has_numbered_steps {
        return false;
    }

    // Check if the steps are all immediate tool actions
    let action_verbs = [
        "run ",
        "execute ",
        "search ",
        "write ",
        "create ",
        "check ",
        "list ",
        "read ",
        "fetch ",
        "download ",
        "install ",
        "find ",
        "show ",
        "display ",
        "get ",
        "send ",
        "open ",
        "save ",
    ];
    let step_count = lower.matches([')', '.']).count().min(10); // cap to avoid false positives on prose

    // Count how many action verbs appear - if most steps are tool actions, it's sequential
    let action_count = action_verbs.iter().filter(|v| lower.contains(*v)).count();
    action_count >= 2 && step_count >= 2
}

#[cfg(test)]
#[path = "policy_signal_tests.rs"]
mod policy_signal_tests;
