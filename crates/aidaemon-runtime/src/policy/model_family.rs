//! Model-id patterns that name a family and a version.
//!
//! The harness keeps a few short lists keyed on model ids: which models are
//! frontier (`trust_tier`), which see images and how large a window is
//! (`settings`). A pattern such as `gpt-6` or `gemini-2.5` in one of those
//! lists names a family and the version it starts at; this module makes
//! such a pattern cover every later version of that family too, so a new
//! release (`gpt-7-sol`, `gemini-4-hydrogen`) is treated like the one
//! before it without a code change.

use std::cmp::Ordering;

/// The leading version of a token: `6.2` of `6.2`, `4` of `4o`.
fn leading_version(token: &str) -> Option<Vec<u64>> {
    let end = token
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(token.len());
    let run = token[..end].trim_end_matches('.');
    if run.is_empty() {
        return None;
    }
    run.split('.').map(|part| part.parse().ok()).collect()
}

/// `(family, version)` of a pattern written as `<family>-<version>`, such as
/// `gpt-5` or `gemini-2.5`; `None` for any other pattern.
fn family_pattern(pattern: &str) -> Option<(String, Vec<u64>)> {
    let pattern = pattern.trim().to_ascii_lowercase();
    let (family, rest) = pattern.split_once('-')?;
    if family.is_empty() || !family.bytes().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    // Only a bare version: `gpt-4.1`, not `claude-opus` or `gpt-4o-mini`.
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
        return None;
    }
    Some((family.to_string(), leading_version(rest)?))
}

/// The version `model` carries right after its `family` word, e.g. `[7]` for
/// `gpt` in `openai/gpt-7-sol`.
fn version_after(model: &str, family: &str) -> Option<Vec<u64>> {
    let model = model.trim().to_ascii_lowercase();
    let name = model.rsplit('/').next().unwrap_or(&model);
    let mut tokens = name.split('-');
    while let Some(token) = tokens.next() {
        // A route such as `us.anthropic.claude` keeps its family last.
        let word = token.rsplit('.').next().unwrap_or(token);
        if word == family {
            return tokens.next().and_then(leading_version);
        }
    }
    None
}

fn compare(a: &[u64], b: &[u64]) -> Ordering {
    for i in 0..a.len().max(b.len()) {
        match a.get(i).unwrap_or(&0).cmp(b.get(i).unwrap_or(&0)) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}

/// The version of `pattern` when it names a family version that `model`
/// is at or past: `gpt-5` covers `gpt-7-sol`, `gemini-2.5` covers
/// `gemini-4-hydrogen`, and neither covers `gpt-4o` or `gpt-oss-120b`.
pub fn covered_version(pattern: &str, model: &str) -> Option<Vec<u64>> {
    let (family, from) = family_pattern(pattern)?;
    let version = version_after(model, &family)?;
    (compare(&version, &from) != Ordering::Less).then_some(from)
}

/// Whether `pattern` names a family version that `model` is at or past.
pub fn covers_later_version(pattern: &str, model: &str) -> bool {
    covered_version(pattern, model).is_some()
}

/// Of `patterns`, the one naming the latest family version `model` is at or
/// past: for `gpt-7-sol` among `gpt-4.1`, `gpt-5` and `gpt-6`, `gpt-6`.
pub fn closest_family_pattern<'a, T>(
    patterns: impl IntoIterator<Item = (&'a str, T)>,
    model: &str,
) -> Option<T> {
    patterns
        .into_iter()
        .filter_map(|(pattern, value)| covered_version(pattern, model).map(|v| (v, value)))
        .max_by(|(a, _), (b, _)| compare(a, b))
        .map(|(_, value)| value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_family_version_covers_later_versions() {
        assert!(covers_later_version("gpt-5", "gpt-7-sol"));
        assert!(covers_later_version("gpt-6", "openai/gpt-6.2-sol:batch"));
        assert!(covers_later_version("gemini-2.5", "gemini-4-hydrogen"));
        assert!(covers_later_version("gemini-2.5", "google/gemini-2.5-pro"));
        assert!(covers_later_version("grok-3", "grok-4.20"));
        assert!(covers_later_version("gpt-4", "gpt-4o-mini"));
        assert!(covers_later_version(
            "claude-3",
            "us.anthropic.claude-3-7-sonnet"
        ));
    }

    #[test]
    fn earlier_versions_other_families_and_other_shapes_are_not_covered() {
        assert!(!covers_later_version("gemini-2.5", "gemini-2-pro"));
        assert!(!covers_later_version("gpt-4.1", "gpt-4o"));
        assert!(!covers_later_version("gpt-4", "gpt-oss-120b"));
        assert!(!covers_later_version("gpt-4", "gpt-3.5-turbo"));
        assert!(!covers_later_version("gpt-5", "chatgpt-7"));
        assert!(!covers_later_version("gemini-3", "gemini-flash-latest"));
        // Patterns that are not a family version stay plain.
        assert!(!covers_later_version("claude-opus", "claude-opus-9"));
        assert!(!covers_later_version("qwen3-vl", "qwen4-vl"));
        assert!(!covers_later_version("gpt-4o", "gpt-7"));
        assert!(!covers_later_version("o3", "o5"));
    }

    #[test]
    fn the_closest_family_version_wins() {
        let windows = [("gpt-4.1", 1), ("gpt-5", 2), ("gpt-6", 3), ("grok-4", 4)];
        assert_eq!(closest_family_pattern(windows, "gpt-7-sol"), Some(3));
        assert_eq!(closest_family_pattern(windows, "gpt-5.6-terra"), Some(2));
        assert_eq!(closest_family_pattern(windows, "gpt-4o"), None);
        assert_eq!(closest_family_pattern(windows, "grok-5"), Some(4));
    }
}
