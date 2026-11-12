//! Text helpers shared by tools and event payloads.

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
/// use aidaemon_tools::text::truncate_str;
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

/// Whether `keyword` appears in `text` as whole words (case-insensitive,
/// surrounding punctuation ignored), so "send" matches "Send it!" but not
/// "sender".
pub fn contains_keyword_as_words(text: &str, keyword: &str) -> bool {
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
