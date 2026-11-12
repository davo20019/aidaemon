//! The one list of words that mark a UI action (a button, a selector, a
//! script, typed text) as consequential, for every tool that drives a UI:
//! the browser and computer use judge the same click the same way.

/// Words whose standalone presence (word-boundary match) signals an action
/// that sends, spends, deletes, publishes or changes access.
const CONSEQUENTIAL_ACTION_WORDS: &[&str] = &[
    // Sending and publishing
    "send",
    "submit",
    "publish",
    "post",
    // Money
    "buy",
    "purchase",
    "checkout",
    "pay",
    "transfer",
    // Deleting
    "delete",
    "remove",
    // Access and accounts
    "confirm",
    "authorize",
    "permission",
    "authentication",
    "account",
    "sign out",
    "log out",
];

/// Whether `text` names a consequential action as a whole word or phrase.
/// Derived forms ("submitted", "sender") do not match.
pub fn names_consequential_action(text: &str) -> bool {
    CONSEQUENTIAL_ACTION_WORDS
        .iter()
        .any(|word| crate::text::contains_keyword_as_words(text, word))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consequential_words_match_whole_words_only() {
        assert!(names_consequential_action("Send"));
        assert!(names_consequential_action("Confirm purchase."));
        assert!(names_consequential_action("click Log out"));
        assert!(!names_consequential_action("Sender details"));
        assert!(!names_consequential_action("submitted form"));
    }
}
