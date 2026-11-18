//! Which offered model is the newer version of one in use.
//!
//! Providers list dozens of ids: dated snapshots, previews, embeddings,
//! audio, other sizes. A successor is a model of the same line (same name,
//! same variant such as `sol`, `opus`, `pro` or `mini`) with a higher
//! version, or the stable release of a preview in use. Anything else is not
//! offered as an upgrade, so `gpt-6-sol` never becomes `gpt-6.2-luna` and
//! `claude-opus-5-5` never becomes a `claude-haiku`.

use std::cmp::Ordering;

/// Words that mark a pre-release build. A preview may succeed a preview,
/// and a stable release succeeds a preview, never the other way.
const PRERELEASE: &[&str] = &["preview", "exp", "experimental", "beta", "alpha"];

/// Words that mark a model that is not a chat model, or a special build of
/// one; an id carrying one the current model lacks is never a successor.
const SPECIAL: &[&str] = &[
    "audio",
    "realtime",
    "tts",
    "transcribe",
    "embedding",
    "embed",
    "image",
    "search",
    "instruct",
    "latest",
    "chat",
    "codex",
    "computer",
    "deep",
    "research",
    "live",
    "native",
    "thinking",
];

/// An id split into its route (`openrouter/openai/`), name, version and
/// variant words.
#[derive(Debug, Clone, PartialEq)]
struct Parsed {
    route: String,
    name: Vec<String>,
    version: Vec<u64>,
    variant: Vec<String>,
    prerelease: bool,
    dated: bool,
}

fn number_parts(token: &str) -> Option<Vec<u64>> {
    if token.is_empty() || !token.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    token.split('.').map(|p| p.parse::<u64>().ok()).collect()
}

/// A date stamp such as `20251001` or the `2025`/`10`/`01` of
/// `2025-10-01`, which marks a pinned snapshot.
fn is_date(token: &str) -> bool {
    token.len() == 8 && token.bytes().all(|b| b.is_ascii_digit())
        || token.len() == 4 && token.starts_with("20") && token.bytes().all(|b| b.is_ascii_digit())
}

fn parse(id: &str) -> Option<Parsed> {
    let lower = id.trim().to_ascii_lowercase();
    let (route, model) = match lower.rfind('/') {
        Some(at) => (lower[..=at].to_string(), &lower[at + 1..]),
        None => (String::new(), lower.as_str()),
    };
    let mut name = Vec::new();
    let mut version = Vec::new();
    let mut variant = Vec::new();
    let mut prerelease = false;
    let mut dated = false;
    let mut tokens = model.split(['-', '_', ':', '@']).peekable();
    while let Some(token) = tokens.next() {
        if token.is_empty() {
            continue;
        }
        if is_date(token) {
            dated = true;
            // The month and day of a dashed date follow.
            while tokens
                .peek()
                .is_some_and(|t| t.len() == 2 && t.bytes().all(|b| b.is_ascii_digit()))
            {
                tokens.next();
            }
            continue;
        }
        if PRERELEASE.contains(&token) {
            prerelease = true;
            continue;
        }
        // A short version run: `6`, `6.2`, or Anthropic's `5-5`.
        if let Some(parts) = number_parts(token)
            && variant.is_empty()
            && version.len() < 3
            && token.len() <= 4
        {
            version.extend(parts);
            continue;
        }
        if version.is_empty() {
            name.push(token.to_string());
        } else {
            variant.push(token.to_string());
        }
    }
    if name.is_empty() || version.is_empty() {
        return None;
    }
    Some(Parsed {
        route,
        name,
        version,
        variant,
        prerelease,
        dated,
    })
}

fn compare_versions(a: &[u64], b: &[u64]) -> Ordering {
    let len = a.len().max(b.len());
    for i in 0..len {
        let (x, y) = (
            a.get(i).copied().unwrap_or(0),
            b.get(i).copied().unwrap_or(0),
        );
        match x.cmp(&y) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}

/// Whether `candidate` is a newer model of `current`'s line.
pub fn is_successor(current: &str, candidate: &str) -> bool {
    let (Some(cur), Some(cand)) = (parse(current), parse(candidate)) else {
        return false;
    };
    if cand.route != cur.route || cand.name != cur.name || cand.variant != cur.variant {
        return false;
    }
    // A snapshot pins a version on purpose; only an undated id moves on.
    if cand.dated || (cand.prerelease && !cur.prerelease) {
        return false;
    }
    if cand
        .variant
        .iter()
        .chain(&cand.name)
        .any(|word| SPECIAL.contains(&word.as_str()))
    {
        return false;
    }
    match compare_versions(&cand.version, &cur.version) {
        Ordering::Greater => true,
        // The stable release of the preview in use.
        Ordering::Equal => cur.prerelease && !cand.prerelease,
        Ordering::Less => false,
    }
}

/// Of `offered`, the newest successor of `current`, if any.
pub fn successor<'a>(current: &str, offered: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    offered
        .into_iter()
        .filter(|candidate| is_successor(current, candidate))
        .max_by(|a, b| {
            let (pa, pb) = (parse(a), parse(b));
            match (pa, pb) {
                (Some(pa), Some(pb)) => compare_versions(&pa.version, &pb.version)
                    .then_with(|| pb.prerelease.cmp(&pa.prerelease)),
                _ => Ordering::Equal,
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(current: &str, offered: &[&'static str]) -> Option<&'static str> {
        successor(current, offered.iter().copied())
    }

    #[test]
    fn the_same_line_with_a_higher_version_succeeds() {
        let offered = [
            "gpt-6-astra",
            "gpt-6-sol",
            "gpt-6-luna",
            "gpt-6.2-astra",
            "gpt-6.2-sol",
            "gpt-5.6-sol",
        ];
        assert_eq!(next("gpt-6-sol", &offered), Some("gpt-6.2-sol"));
        assert_eq!(next("gpt-6-astra", &offered), Some("gpt-6.2-astra"));
        // No 6.2 luna yet.
        assert_eq!(next("gpt-6-luna", &offered), None);
        assert_eq!(next("gpt-6.2-sol", &offered), None);
        assert_eq!(next("gpt-5.6-sol", &offered), Some("gpt-6.2-sol"));
    }

    #[test]
    fn the_newest_successor_wins() {
        let offered = ["gpt-6.1-sol", "gpt-6.2-sol", "gpt-6.10-sol"];
        assert_eq!(next("gpt-6-sol", &offered), Some("gpt-6.10-sol"));
    }

    #[test]
    fn anthropic_style_versions_compare_by_parts() {
        let offered = [
            "claude-opus-5-5",
            "claude-opus-6",
            "claude-sonnet-6",
            "claude-opus-6-20270101",
            "claude-haiku-6",
        ];
        assert_eq!(next("claude-opus-5-5", &offered), Some("claude-opus-6"));
        assert_eq!(next("claude-sonnet-5-5", &offered), Some("claude-sonnet-6"));
        assert_eq!(next("claude-fable-5-2", &offered), None);
    }

    #[test]
    fn other_sizes_and_special_builds_never_succeed() {
        let offered = [
            "gpt-6.2-mini",
            "gpt-6.2-sol-audio",
            "gpt-6.2-sol-realtime",
            "gpt-6.2-sol-search",
            "gpt-6.2-codex",
            "gpt-6.2",
        ];
        assert_eq!(next("gpt-6-sol", &offered), None);
        assert_eq!(next("gpt-5.4-mini", &offered), Some("gpt-6.2-mini"));
    }

    #[test]
    fn previews_and_snapshots() {
        let offered = [
            "gemini-3-pro",
            "gemini-3.5-pro-preview",
            "gemini-3.5-flash",
            "gemini-3-pro-preview-06-05",
        ];
        // A stable model is not moved to a preview.
        assert_eq!(next("gemini-3-pro", &offered), None);
        // A preview moves to the newest, preview or stable.
        assert_eq!(
            next("gemini-3-pro-preview", &offered),
            Some("gemini-3.5-pro-preview")
        );
        assert_eq!(
            next("gemini-3-flash-preview", &offered),
            Some("gemini-3.5-flash")
        );
        // A dated snapshot is never offered, and a dated model moves to the line's newest.
        assert_eq!(
            next(
                "claude-sonnet-4-5-20250929",
                &["claude-sonnet-4-6", "claude-sonnet-4-6-20260101"]
            ),
            Some("claude-sonnet-4-6")
        );
    }

    #[test]
    fn routes_must_match() {
        let offered = ["openai/gpt-6.2-sol", "gpt-6.2-sol", "other/gpt-7-sol"];
        assert_eq!(
            next("openai/gpt-6-sol", &offered),
            Some("openai/gpt-6.2-sol")
        );
        assert_eq!(next("gpt-6-sol", &offered), Some("gpt-6.2-sol"));
    }

    #[test]
    fn unparseable_ids_are_left_alone() {
        assert_eq!(next("my-custom-model", &["my-custom-model-2"]), None);
        assert_eq!(next("o3", &["o4"]), None);
        assert_eq!(
            next("grok-4", &["grok-4.1", "grok-4-fast"]),
            Some("grok-4.1")
        );
    }
}
