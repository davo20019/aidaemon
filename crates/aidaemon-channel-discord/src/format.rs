//! Outbound text handling: Discord Markdown is close enough to standard
//! Markdown that nothing is converted; long messages are split by
//! [`aidaemon_channels::split_message`] (line and paragraph boundaries, code
//! fences closed and reopened across chunks).

/// Truncate to `max_chars` characters, ending with an ellipsis if cut.
pub use aidaemon_channels::truncate_ellipsis as truncate_chars;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_with_ellipsis() {
        assert_eq!(truncate_chars("abcdef", 4), "abc…");
        assert_eq!(truncate_chars("abc", 4), "abc");
    }
}
