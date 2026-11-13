//! Common utility functions used across the codebase.

pub use aidaemon_tools::text::{
    floor_char_boundary, truncation_notice, truncation_notice_with_hint,
};

pub use aidaemon_events::text::{truncate_str, truncate_with_note};

/// Render the model-facing truncation notice from structured metadata.
/// Exactly one call site in the agent loop renders this; tools must not
/// embed it in their returned output.
pub fn render_truncation_notice(info: &crate::traits::TruncationInfo) -> String {
    match info.remediation_hint.as_deref() {
        Some(hint) => truncation_notice_with_hint(info.shown_chars, info.total_chars, hint),
        None => truncation_notice(info.shown_chars, info.total_chars),
    }
}

/// Losslessly compact a pretty-printed JSON block embedded in tool-result
/// text (headers/prefix + JSON body + suffix). Pretty-printed JSON costs
/// ~2.5-3x the tokens of its compact form with identical information — on a
/// small-context local model that difference is the compose step timing out
/// versus answering. Returns `Some(new_text)` only when a parseable JSON
/// block was found AND compaction actually shrank the text; any doubt
/// (unparseable, truncated, already compact) returns `None` and the caller
/// keeps the original.
pub fn compact_embedded_json(text: &str) -> Option<String> {
    let start = text.find(['{', '['])?;
    let open = text.as_bytes()[start] as char;
    let close = if open == '{' { '}' } else { ']' };
    let end = text.rfind(close)?;
    if end <= start {
        return None;
    }
    let candidate = &text[start..=end];
    let value: serde_json::Value = serde_json::from_str(candidate).ok()?;
    let compact = value.to_string();
    if compact.chars().count() >= candidate.chars().count() {
        return None;
    }
    let mut rebuilt = String::with_capacity(text.len() - candidate.len() + compact.len());
    rebuilt.push_str(&text[..start]);
    rebuilt.push_str(&compact);
    rebuilt.push_str(&text[end + close.len_utf8()..]);
    Some(rebuilt)
}

pub use aidaemon_runtime::message_content::is_internal_scaffolding_line;

pub use aidaemon_runtime::turn::helpers::extract_json_object;

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
    fn compact_embedded_json_shrinks_pretty_api_bodies_losslessly() {
        // Live repro (task 6331508b): a pretty-printed clinical-trials JSON
        // response stayed inline (under the spill cap) and its whitespace
        // inflation helped time out the compose call.
        let pretty = "HTTP 200 OK\ncontent-type: application/json\n\n[\n  {\n    \"nctId\": \"NCT00000000\",\n    \"title\": \"Synthetic Study\",\n    \"locations\": [\n      {\n        \"city\": \"Springfield\",\n        \"state\": \"Virginia\"\n      }\n    ]\n  }\n]\n";
        let compacted = compact_embedded_json(pretty).expect("compaction applies");
        assert!(compacted.chars().count() < pretty.chars().count());
        assert!(compacted.starts_with("HTTP 200 OK"), "prefix preserved");
        assert!(
            compacted.contains("\"nctId\":\"NCT00000000\""),
            "data intact, compact"
        );
        // Lossless: parse both bodies and compare values.
        let orig_start = pretty.find('[').unwrap();
        let orig: serde_json::Value =
            serde_json::from_str(&pretty[orig_start..pretty.rfind(']').unwrap() + 1]).unwrap();
        let new_start = compacted.find('[').unwrap();
        let new: serde_json::Value =
            serde_json::from_str(&compacted[new_start..compacted.rfind(']').unwrap() + 1]).unwrap();
        assert_eq!(orig, new);
    }

    #[test]
    fn compact_embedded_json_leaves_prose_and_broken_json_alone() {
        assert!(compact_embedded_json("plain command output with no braces").is_none());
        // Truncated JSON (mid-array cut) must not be touched.
        assert!(compact_embedded_json("[\n  {\"a\": 1},\n  {\"b\"").is_none());
        // Already-compact JSON: no shrink, no change.
        assert!(compact_embedded_json("{\"a\":1}").is_none());
    }

    #[test]
    fn test_truncation_notice_reports_amounts_and_forbids_fabrication() {
        let notice = truncation_notice(100, 250);
        assert!(notice.contains("OUTPUT TRUNCATED"));
        assert!(notice.contains("100 of 250"));
        // 150 omitted = 250 - 100
        assert!(notice.contains("150 omitted"));
        // Must instruct the model not to invent the omitted content.
        assert!(notice.contains("Do NOT enumerate"));
    }

    #[test]
    fn test_truncation_notice_saturates_when_shown_exceeds_total() {
        // Defensive: shown > total must not underflow.
        let notice = truncation_notice(300, 250);
        assert!(notice.contains("0 omitted"));
    }

    #[test]
    fn test_truncation_notice_with_hint_uses_custom_remediation() {
        let notice = truncation_notice_with_hint(100, 250, "Re-fetch with a larger max_chars.");
        assert!(notice.contains("OUTPUT TRUNCATED"));
        assert!(notice.contains("100 of 250"));
        assert!(notice.contains("Do NOT enumerate"));
        assert!(notice.contains("Re-fetch with a larger max_chars."));
        // The terminal-flavored default hint must not leak into custom-hint notices.
        assert!(!notice.contains("wc -l"));
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

    #[test]
    fn render_truncation_notice_matches_legacy_format() {
        let info = crate::traits::TruncationInfo {
            shown_chars: 100,
            total_chars: 250,
            remediation_hint: None,
        };
        assert_eq!(render_truncation_notice(&info), truncation_notice(100, 250));

        let hinted = crate::traits::TruncationInfo {
            shown_chars: 100,
            total_chars: 250,
            remediation_hint: Some("Re-fetch with a larger max_chars.".to_string()),
        };
        assert_eq!(
            render_truncation_notice(&hinted),
            truncation_notice_with_hint(100, 250, "Re-fetch with a larger max_chars.")
        );
    }
}
