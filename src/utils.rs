//! Common utility functions used across the codebase.

/// Returns the largest byte index ≤ `byte_limit` that falls on a UTF-8 char boundary.
///
/// Use this when you need to slice a string by approximate byte length without panicking.
/// The returned index is safe to use with `&s[..index]`.
pub fn floor_char_boundary(s: &str, byte_limit: usize) -> usize {
    if byte_limit >= s.len() {
        return s.len();
    }
    let mut i = byte_limit;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Truncates a string to at most `max_chars` characters, adding "..." if truncated.
///
/// This function is UTF-8 safe and respects character boundaries, avoiding panics
/// when truncating strings that contain multi-byte characters (like emojis).
///
/// # Arguments
/// * `s` - The string to truncate
/// * `max_chars` - Maximum number of characters (not bytes) in the result, including the "..." suffix
///
/// # Examples
/// ```
/// use aidaemon::utils::truncate_str;
///
/// assert_eq!(truncate_str("hello", 10), "hello");
/// assert_eq!(truncate_str("hello world", 8), "hello...");
/// assert_eq!(truncate_str("🦀🦀🦀🦀🦀", 4), "🦀...");
/// ```
pub fn truncate_str(s: &str, max_chars: usize) -> String {
    truncate_impl(s, max_chars, "...")
}

/// Truncates a string to at most `max_chars` characters, adding "\n... (truncated)" if truncated.
///
/// Similar to [`truncate_str`], but uses a more verbose suffix suitable for multi-line
/// CLI output where the truncation should be clearly visible on its own line.
///
/// This function is UTF-8 safe and respects character boundaries.
///
/// # Arguments
/// * `s` - The string to truncate
/// * `max_chars` - Maximum number of characters (not bytes) before the suffix
pub fn truncate_with_note(s: &str, max_chars: usize) -> String {
    truncate_impl(s, max_chars, "\n... (truncated)")
}

/// Internal implementation for string truncation.
fn truncate_impl(s: &str, max_chars: usize, suffix: &str) -> String {
    // Fast path: if string is short enough, return as-is
    // We check byte length first as a cheap filter before counting chars
    if s.len() <= max_chars {
        // Byte length is <= max_chars, so char count must also be <= max_chars
        // (each char is at least 1 byte)
        return s.to_string();
    }

    // Count actual characters
    let char_count = s.chars().count();
    if char_count <= max_chars {
        return s.to_string();
    }

    let suffix_len = suffix.chars().count();

    // Need to truncate - reserve space for suffix
    if max_chars <= suffix_len {
        // If max_chars is less than or equal to suffix length, just return truncated suffix
        return suffix.chars().take(max_chars).collect();
    }

    // Take max_chars - suffix_len characters and append suffix
    let truncated: String = s.chars().take(max_chars - suffix_len).collect();
    format!("{}{}", truncated, suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_truncation_needed() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello", 5), "hello");
        assert_eq!(truncate_str("", 10), "");
    }

    #[test]
    fn test_truncation_ascii() {
        assert_eq!(truncate_str("hello world", 8), "hello...");
        assert_eq!(truncate_str("hello world", 7), "hell...");
        assert_eq!(truncate_str("abcdefghij", 6), "abc...");
    }

    #[test]
    fn test_truncation_emoji() {
        // Each emoji is 1 character but multiple bytes
        // 5 emojis = 5 chars, so max_chars=5 means no truncation
        assert_eq!(truncate_str("🦀🦀🦀🦀🦀", 5), "🦀🦀🦀🦀🦀"); // No truncation needed
        assert_eq!(truncate_str("🦀🦀🦀🦀🦀", 4), "🦀..."); // 4-3=1 emoji + "..."
        assert_eq!(truncate_str("🦀🦀🦀🦀🦀🦀", 5), "🦀🦀..."); // 6 emojis, take 2 + "..."
        assert_eq!(truncate_str("🦀🦀🦀🦀🦀🦀🦀", 6), "🦀🦀🦀..."); // 7 emojis, take 3 + "..."
    }

    #[test]
    fn test_truncation_mixed() {
        // Mix of ASCII and emoji
        assert_eq!(truncate_str("hi 🦀 world", 8), "hi 🦀 ...");
        assert_eq!(truncate_str("⛅️ wrangler 4.62.0", 10), "⛅️ wran...");
    }

    #[test]
    fn test_edge_cases() {
        // Very small max_chars
        assert_eq!(truncate_str("hello", 3), "...");
        assert_eq!(truncate_str("hello", 2), "..");
        assert_eq!(truncate_str("hello", 1), ".");
        assert_eq!(truncate_str("hello", 0), "");

        // Exact boundary
        assert_eq!(truncate_str("hello", 5), "hello");
        assert_eq!(truncate_str("hello!", 6), "hello!");
    }

    #[test]
    fn test_unicode_various() {
        // Various multi-byte characters
        assert_eq!(truncate_str("héllo wörld", 8), "héllo...");
        assert_eq!(truncate_str("日本語テスト", 5), "日本...");
        assert_eq!(truncate_str("🌀✨⛅️🦞", 4), "🌀...");
    }

    #[test]
    fn test_variation_selectors() {
        // Emoji with variation selectors (e.g., ⛅️ is ⛅ + VS16)
        // This should not panic even if the variation selector is a separate code point
        let s = "⛅️ test";
        let result = truncate_str(s, 5);
        assert!(result.len() <= 20); // Just verify it doesn't panic
    }

    #[test]
    fn test_truncate_with_note() {
        use super::truncate_with_note;

        // No truncation needed
        assert_eq!(truncate_with_note("hello", 20), "hello");

        // Truncation with verbose suffix ("\n... (truncated)" is 16 chars)
        // String is 34 chars, max is 30, so we need to truncate
        let result = truncate_with_note("hello world this is a long string", 30);
        assert!(result.ends_with("\n... (truncated)"));
        assert!(result.starts_with("hello"));

        // Works with emojis - 10 emojis is 10 chars, max 20 means no truncation
        let result = truncate_with_note("🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀", 20);
        assert_eq!(result, "🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀"); // No truncation needed

        // Truncation with emojis - 20 emojis = 20 chars, suffix is 16 chars
        // With max_chars=20, we need content > 20 to trigger truncation
        let result = truncate_with_note("🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀", 20);
        assert!(result.contains("🦀"));
        assert!(result.ends_with("\n... (truncated)"));
    }

    mod proptest_truncate {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn truncate_result_within_limit(s in ".*", n in 0usize..500) {
                let result = truncate_str(&s, n);
                assert!(result.chars().count() <= n.max(1));
            }

            #[test]
            fn no_truncation_when_fits(s in "[a-z]{0,50}", n in 50usize..200) {
                let result = truncate_str(&s, n);
                if s.chars().count() <= n {
                    assert_eq!(result, s);
                }
            }

            #[test]
            fn truncate_never_panics(s in "\\PC{0,500}", n in 0usize..1000) {
                let _ = truncate_str(&s, n);
                let _ = truncate_with_note(&s, n);
            }
        }
    }
}

/// Extract a JSON object from LLM output, handling code fences and preamble text.
/// Tries direct parse first, then falls back to finding `{...}` bounds.
pub fn extract_json_object(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let candidate = if trimmed.starts_with("```") {
        trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```JSON")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
            .to_string()
    } else {
        trimmed.to_string()
    };
    if serde_json::from_str::<serde_json::Value>(&candidate)
        .ok()
        .is_some_and(|v| v.is_object())
    {
        return Some(candidate);
    }

    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    let sliced = raw[start..=end].trim().to_string();
    if serde_json::from_str::<serde_json::Value>(&sliced)
        .ok()
        .is_some_and(|v| v.is_object())
    {
        Some(sliced)
    } else {
        None
    }
}
