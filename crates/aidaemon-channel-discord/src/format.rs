//! Outbound text handling: Discord Markdown is close enough to standard
//! Markdown that nothing is converted; long messages are split by
//! [`aidaemon_channels::split_message`] (line and paragraph boundaries, code
//! fences closed and reopened across chunks).

/// Truncate to `max_chars` characters, ending with an ellipsis if cut.
pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_with_ellipsis() {
        assert_eq!(truncate_chars("abcdef", 4), "abc…");
        assert_eq!(truncate_chars("abc", 4), "abc");
    }
}
