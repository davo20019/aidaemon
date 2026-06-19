//! Cheap, recall-biased pre-filter that decides whether the relational
//! classifier should run this turn. A false positive only costs one
//! fast-model call (which then returns `None`); it must never be the sole
//! authority for behavior.

use crate::agent::intent_routing::contains_keyword_as_words;

/// Relational nouns that suggest a personal-graph query (e.g., "X's spouse").
const RELATIONAL_NOUNS: &[&str] = &[
    "spouse",
    "husband",
    "wife",
    "partner",
    "boyfriend",
    "girlfriend",
    "boss",
    "manager",
    "colleague",
    "sibling",
    "brother",
    "sister",
    "parent",
    "mother",
    "father",
    "mom",
    "dad",
    "son",
    "daughter",
    "kid",
    "kids",
    "child",
    "children",
    "friend",
    "pet",
    "dog",
    "cat",
    "job",
    "company",
    "email",
    "phone",
    "address",
    "birthday",
    "age",
    "nationality",
];

/// Returns true when the text looks like a relational query about a named
/// person known to the agent (e.g., "who is caro's spouse?", "where does
/// María work?").  Identified by the possessive `'s` pattern combined with
/// a relational noun, or by interrogative+verb patterns about persons.
fn looks_like_named_person_relational_query(lower: &str) -> bool {
    // Possessive pattern: "<name>'s <relation>" — strong signal for a
    // personal-graph lookup regardless of which interrogative opens it.
    let has_possessive = lower.contains("'s");
    if has_possessive
        && RELATIONAL_NOUNS
            .iter()
            .any(|r| contains_keyword_as_words(lower, r))
    {
        return true;
    }

    // "where does X live / work" — relational even without possessive.
    let relational_verbs = ["live", "lives", "work", "works"];
    if contains_keyword_as_words(lower, "where")
        && relational_verbs
            .iter()
            .any(|v| contains_keyword_as_words(lower, v))
    {
        return true;
    }

    false
}

/// True when the text is specifically a named-person relational query
/// (e.g., "who is Caro's spouse?") as opposed to a generic recall question
/// ("what about pets?"). Used by the completion-phase denial gate to scope
/// the check to cases where a specific named entity was not looked up.
pub fn user_text_is_named_person_relational_query(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    looks_like_named_person_relational_query(&lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_for_named_person_possessive_query() {
        assert!(user_text_is_named_person_relational_query(
            "who is caro's spouse?"
        ));
    }

    #[test]
    fn fires_for_where_does_x_work() {
        assert!(user_text_is_named_person_relational_query(
            "where does María work?"
        ));
    }

    #[test]
    fn skips_general_knowledge() {
        assert!(!user_text_is_named_person_relational_query(
            "who is the president of france?"
        ));
    }
}
