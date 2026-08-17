//! Centralized keyword constants for intent classification.
//!
//! These constants back the memory-storage / scheduling / recall classifiers
//! in `intent_routing` and `policy::recall_guardrails`. Centralizing them
//! ensures that adding a new verb in one place covers every classifier that
//! needs to react to it — a class of bugs we hit repeatedly when the same
//! lists drifted apart in different files.
//!
//! Each call site still owns its own combinator logic. What is shared is the
//! vocabulary, not the decision.
//!
//! ## Precision levels
//!
//! These vocabularies support advisory context and tests. They do not route an
//! autonomous turn or remove authorized tools from its roster.
//!
//! We expose two precision levels for memory-store intent:
//!
//! - [`MEMORY_STORE_STRICT_PHRASES`] — multi-word phrases. High precision,
//!   safe to use as a hard gate against other intents.
//! - [`MEMORY_STORE_LENIENT_VERBS`] — single-word imperative verbs.
//!   Broader recall; appropriate where missing a real store request is worse
//!   than over-matching.
//!
//! Callers pick the level that matches the cost of being wrong.

/// Strong multi-word phrases that clearly indicate the user is asking the
/// agent to store information.
///
/// Each entry is intentionally a multi-token phrase to minimize false
/// positives from word stems and unrelated uses (e.g., "remembers", "saved
/// me time").
#[cfg(test)]
pub(crate) const MEMORY_STORE_STRICT_PHRASES: &[&str] = &[
    "remember that",
    "remember my",
    "remember i",
    "remember these",
    "remember this",
    "remember the following",
    "note that",
    "note my",
    "note these",
    "note this",
    "save that",
    "save these",
    "save this",
    "store these",
    "store this",
    "store that",
    "keep in mind",
    "keep track of",
    "don't forget",
    "do not forget",
    "memorize",
    "learn that",
    "learn this",
    "record that",
    "record this",
    "update my",
];

/// Single-word imperative verbs that suggest a store intent. Broader than
/// [`MEMORY_STORE_STRICT_PHRASES`] and tolerated by classifiers where the
/// downside of a false positive is small (e.g., the recall guardrail just
/// keeps the tool palette wider).
#[cfg(test)]
pub(crate) const MEMORY_STORE_LENIENT_VERBS: &[&str] = &["remember", "memorize", "store", "save"];

/// Fact-storage context phrases. Even without an imperative verb, the presence
/// of one of these phrases strongly suggests the user is sharing facts to be
/// remembered (e.g., "here are some facts about me: ...").
#[cfg(test)]
pub(crate) const MEMORY_STORE_FACT_CONTEXTS: &[&str] = &[
    "facts about me",
    "things about me",
    "info about me",
    "information about me",
    "details about me",
];

/// Scheduling verb phrases. When present, the user wants the agent to trigger
/// an action at a future time. Classifiers that handle ambiguous date mentions
/// use this to break ties in favor of the scheduler.
#[cfg(test)]
pub(crate) const SCHEDULING_VERB_PHRASES: &[&str] = &[
    "schedule",
    "remind me",
    "set a reminder",
    "alert me",
    "notify me",
    "ping me",
    "wake me",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Invariant: every strict phrase has at least one lenient verb as its
    /// first token (or contains one as a token). This catches accidental
    /// drift where someone adds a strict phrase whose root verb isn't in
    /// the lenient list, which would mean the lenient classifier silently
    /// misses requests the strict classifier catches.
    #[test]
    fn strict_phrases_share_root_verbs_with_lenient_verbs() {
        let lenient: std::collections::HashSet<&str> =
            MEMORY_STORE_LENIENT_VERBS.iter().copied().collect();
        // These strict phrases use verbs intentionally absent from the
        // lenient list (their bare form is too ambiguous to safely match
        // without context). Document them explicitly.
        let expected_lenient_gaps: std::collections::HashSet<&str> =
            ["note", "keep", "don't", "do", "learn", "record", "update"]
                .into_iter()
                .collect();
        for phrase in MEMORY_STORE_STRICT_PHRASES {
            let first_word = phrase.split_whitespace().next().unwrap_or("");
            assert!(
                lenient.contains(first_word) || expected_lenient_gaps.contains(first_word),
                "strict phrase {phrase:?} has root verb {first_word:?} that is \
                 neither in MEMORY_STORE_LENIENT_VERBS nor in the documented \
                 gap list — add it to one or the other"
            );
        }
    }

    #[test]
    fn no_empty_keyword_entries() {
        for kw in MEMORY_STORE_STRICT_PHRASES
            .iter()
            .chain(MEMORY_STORE_LENIENT_VERBS.iter())
            .chain(MEMORY_STORE_FACT_CONTEXTS.iter())
            .chain(SCHEDULING_VERB_PHRASES.iter())
        {
            assert!(!kw.is_empty(), "empty keyword constant");
            assert_eq!(
                *kw,
                kw.to_ascii_lowercase(),
                "keyword {kw:?} must be lowercase"
            );
            assert_eq!(
                *kw,
                kw.trim(),
                "keyword {kw:?} has leading/trailing whitespace"
            );
        }
    }
}
