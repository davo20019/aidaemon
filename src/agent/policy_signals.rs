use std::collections::HashMap;

use crate::execution_policy::{
    score_risk_from_capabilities, score_uncertainty_v1, PolicyBundle, UncertaintySignals,
};
use crate::traits::ToolCapabilities;

pub(super) fn user_text_looks_ambiguous(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();

    // If the message contains a filesystem path, the user is giving us
    // concrete location info - never treat that as ambiguous.
    if lower.contains('/') || lower.contains('\\') {
        return false;
    }

    // Only flag truly bare/short references - when the entire message
    // is basically just a pronoun or vague phrase with no actionable context.
    // Longer messages (>40 chars) have enough context for the LLM to decide.
    if lower.len() > 40 {
        return false;
    }

    let phrase_ambiguous = [
        "the site",
        "that site",
        "this site",
        "that project",
        "the project",
        "that file",
        "this file",
        "that one",
        "this one",
        "the thing",
        "that thing",
    ]
    .iter()
    .any(|p| {
        lower == *p || lower.starts_with(&format!("{} ", p)) || lower.contains(&format!(" {}", p))
    });
    if phrase_ambiguous {
        return true;
    }

    matches!(lower.as_str(), "it" | "this" | "that")
}

#[allow(dead_code)] // Kept for potential future consultant/fallback handling.
pub(super) fn first_question_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| line.contains('?'))
        .map(|s| s.to_string())
}

pub(super) fn default_clarifying_question(user_text: &str, missing_info: &[String]) -> String {
    if !missing_info.is_empty() {
        return format!(
            "Could you clarify {} so I can proceed correctly?",
            missing_info.join(", ")
        );
    }
    if user_text_looks_ambiguous(user_text) {
        return "Could you clarify exactly which site/project/file you mean?".to_string();
    }
    "Could you share the missing details I need before I proceed?".to_string()
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

fn estimate_risk_from_text(user_text: &str) -> f32 {
    let lower = user_text.to_ascii_lowercase();
    let mut score = 0.18f32;

    if contains_any(
        &lower,
        &[
            "write ",
            "edit ",
            "change ",
            "modify ",
            "create ",
            "delete ",
            "remove ",
            "fix ",
            "deploy ",
            "install ",
            "run ",
            "execute ",
            "commit ",
            "schedule ",
        ],
    ) {
        score += 0.28;
    }

    if contains_any(
        &lower,
        &[
            "api ",
            "http",
            "webhook",
            "send ",
            "post ",
            "publish ",
            "external",
            "production",
        ],
    ) {
        score += 0.20;
    }

    if contains_any(
        &lower,
        &[
            "rm ",
            "sudo",
            "drop ",
            "truncate ",
            "force",
            "dangerous",
            "overwrite",
        ],
    ) {
        score += 0.25;
    }

    score.clamp(0.0, 1.0)
}

fn infer_uncertainty_signals(user_text: &str, prior_immediate_failure: bool) -> UncertaintySignals {
    let lower = user_text.trim().to_ascii_lowercase();
    let missing_required_slot = user_text_looks_ambiguous(user_text)
        || matches!(lower.as_str(), "do it" | "handle it" | "fix it" | "run it");

    let conflicting_constraints = (lower.contains("quick") && lower.contains("detailed"))
        || (lower.contains("short") && lower.contains("comprehensive"))
        || (lower.contains("brief") && lower.contains("deep"));

    let ambiguous_wording =
        contains_any(
            &lower,
            &[
                "sometime",
                "later",
                "soon",
                "asap",
                "next week",
                "one day",
                "eventually",
                "whenever",
            ],
        ) && !contains_any(&lower, &[" at ", " on ", " by ", " cron", "every "]);

    UncertaintySignals {
        missing_required_slot,
        conflicting_constraints,
        ambiguous_wording,
        prior_immediate_failure,
    }
}

pub(super) fn build_policy_bundle_v1(
    user_text: &str,
    available_capabilities: &HashMap<String, ToolCapabilities>,
    prior_immediate_failure: bool,
) -> PolicyBundle {
    let text_risk = estimate_risk_from_text(user_text);
    let cap_risk =
        score_risk_from_capabilities(&available_capabilities.values().copied().collect::<Vec<_>>());
    let risk_score = ((text_risk * 0.7) + (cap_risk * 0.3)).clamp(0.0, 1.0);
    let uncertainty_score = score_uncertainty_v1(infer_uncertainty_signals(
        user_text,
        prior_immediate_failure,
    ));
    let confidence = (1.0 - uncertainty_score).clamp(0.0, 1.0);
    PolicyBundle::from_scores(risk_score, uncertainty_score, confidence)
}

pub(super) fn detect_explicit_outcome_signal(text: &str) -> Option<(&'static str, bool)> {
    let lower = text.to_ascii_lowercase();
    let positives = ["thanks", "perfect", "got it", "that worked"];
    if positives.iter().any(|p| lower.contains(p)) {
        return Some(("positive", true));
    }
    let negatives = [
        "that's wrong",
        "try again",
        "not what i asked",
        "you misunderstood",
    ];
    if negatives.iter().any(|n| lower.contains(n)) {
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
/// consultant intent gate over-predicts `needs_tools=true`.
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
