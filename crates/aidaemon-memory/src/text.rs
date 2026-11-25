//! Small text helpers shared by the memory modules.

/// `s` cut to at most `max_chars` characters, ending in "..." when cut.
pub(crate) use aidaemon_tools::text::truncate_str as truncate_chars;

/// Lowercased words of a fact value, punctuation and spacing ignored, for
/// comparing values written by different extractors ("~/projects/" and
/// "~/projects" are one value).
/// The lowercase words of `text` (runs of letters and digits, any script).
pub(crate) fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
}

/// `value`'s lowercase words joined by single spaces: two values that
/// differ only in case, spacing or punctuation compare equal.
pub(crate) fn normalize_fact_value(value: &str) -> String {
    words(value).collect::<Vec<_>>().join(" ")
}

pub(crate) fn canonicalize_fact_key(key: &str) -> String {
    let raw = key.trim();
    if raw.is_empty() {
        return String::new();
    }

    let mut out = String::with_capacity(raw.len());
    let mut last_was_sep = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }

    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        raw.to_string()
    } else {
        trimmed.to_string()
    }
}

/// The one name for a personal fact several keys mean (`dob` and
/// `Date of Birth` are both `birth_date`), after [`canonicalize_fact_key`].
pub(crate) fn canonical_personal_key(key: &str) -> String {
    match canonicalize_fact_key(key).as_str() {
        "birthday" | "date_of_birth" | "dob" => "birth_date".to_string(),
        "current_residence" | "home" | "location" => "residence".to_string(),
        "place_of_birth" => "birthplace".to_string(),
        "favorite_name" | "preferred_first_name" => "preferred_name".to_string(),
        other => other.to_string(),
    }
}
