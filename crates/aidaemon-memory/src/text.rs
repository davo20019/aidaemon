//! Small text helpers shared by the memory modules.

/// `s` cut to at most `max_chars` characters, ending in "..." when cut.
pub(crate) fn truncate_chars(s: &str, max_chars: usize) -> String {
    const SUFFIX: &str = "...";
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let suffix_len = SUFFIX.chars().count();
    if max_chars <= suffix_len {
        return SUFFIX.chars().take(max_chars).collect();
    }
    let kept: String = s.chars().take(max_chars - suffix_len).collect();
    format!("{kept}{SUFFIX}")
}

/// Lowercased words of a fact value, punctuation and spacing ignored, for
/// comparing values written by different extractors ("~/projects/" and
/// "~/projects" are one value).
pub(crate) fn normalize_fact_value(value: &str) -> String {
    value
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}
