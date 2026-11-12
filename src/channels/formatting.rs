//! Chat formatting. Markdown conversion for each platform lives in the SDK
//! channel crates; the daemon keeps its plain-text fallback, message
//! splitting and command help.

#[cfg(feature = "slack")]
pub(crate) use aidaemon_channel_slack::format::{
    markdown_to_mrkdwn as markdown_to_slack_mrkdwn, split_mrkdwn,
};
pub(crate) use aidaemon_channel_telegram::format::{
    html_escape, markdown_to_html as markdown_to_telegram_html, split_html as split_telegram_html,
};
/// Fence-aware Markdown splitting by characters (Discord, approval pages).
pub(crate) use aidaemon_channels::split_message;

pub(crate) use aidaemon_channel_telegram::format::{
    markdown_to_plain_fallback as markdown_to_telegram_plain_fallback,
    split_text as split_telegram_text,
};

pub(crate) use aidaemon_channels::slash::build_help_text;

pub(crate) use aidaemon_events::attachments::sanitize_filename;

#[cfg(test)]
mod tests {

    #[test]
    fn plain_multi_line_replies_keep_one_field_per_line() {
        let reply =
            "phase=SYNTHETIC\nblog_schedule=absent\nlatest_run=unknown\ncontroller=not_proven";
        assert_eq!(markdown_to_telegram_html(reply), reply);
        assert_eq!(markdown_to_telegram_plain_fallback(reply), reply);
        assert_eq!(markdown_to_telegram_html(reply).lines().count(), 4);
    }
    use super::*;

    #[test]
    fn test_markdown_to_telegram_heading() {
        let result = markdown_to_telegram_html("### My Heading");
        assert!(result.contains("<b>My Heading</b>"));
    }

    #[test]
    fn test_markdown_to_telegram_bold() {
        let result = markdown_to_telegram_html("This is **bold** text");
        assert!(result.contains("<b>bold</b>"));
    }

    #[test]
    fn test_markdown_to_telegram_plain_fallback_strips_markup() {
        let result = markdown_to_telegram_plain_fallback(
            "**Option 1: Colombia**\n- Current Form: `6 points`\n[Details](https://example.com)",
        );
        assert_eq!(
            result,
            "Option 1: Colombia\n• Current Form: 6 points\nDetails (https://example.com)"
        );
    }

    #[test]
    fn test_markdown_to_telegram_code_block() {
        let result = markdown_to_telegram_html("```rust\nfn main() {}\n```");
        assert!(result.contains("<pre><code class=\"language-rust\">"));
        assert!(result.contains("fn main()"));
    }

    #[test]
    fn test_markdown_to_telegram_code_inside_bold() {
        let result = markdown_to_telegram_html("The Markdown file is **`acme.md`**.");
        assert_eq!(
            result.trim(),
            "The Markdown file is <b><code>acme.md</code></b>."
        );
        let result = markdown_to_telegram_html("See [the `README`](https://example.com) or *`x`*");
        assert!(result.contains(">the <code>README</code></a>"), "{result}");
        assert!(result.contains("<i><code>x</code></i>"), "{result}");
    }

    #[test]
    fn test_markdown_to_telegram_inline_code() {
        let result = markdown_to_telegram_html("Use `cargo build` to compile");
        assert!(result.contains("<code>cargo build</code>"));
    }

    #[test]
    fn test_markdown_to_telegram_list() {
        let result = markdown_to_telegram_html("- item one\n- item two");
        assert!(result.contains("• item one"));
        assert!(result.contains("• item two"));
    }

    #[test]
    fn test_markdown_to_telegram_link() {
        let result = markdown_to_telegram_html("[click here](https://example.com)");
        assert!(result.contains("<a href=\"https://example.com\">click here</a>"));
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("a < b & c > d"), "a &lt; b &amp; c &gt; d");
    }

    #[test]
    fn test_split_message_no_split() {
        let msgs = split_message("short", 4096);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0], "short");
    }

    #[test]
    fn test_split_message_long() {
        let long = "a".repeat(5000);
        let msgs = split_message(&long, 4096);
        assert!(msgs.len() >= 2);
        for msg in &msgs {
            assert!(msg.len() <= 4096 + 50); // small tolerance for split logic
        }
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("test.txt"), "test.txt");
        assert_eq!(sanitize_filename("path/to/file.txt"), "pathtofile.txt");
        assert_eq!(sanitize_filename("a\0b"), "ab");
    }

    #[test]
    fn test_sanitize_filename_long() {
        let long = "a".repeat(250) + ".txt";
        let result = sanitize_filename(&long);
        assert!(result.len() <= 200);
        assert!(result.ends_with(".txt"));
    }

    #[test]
    fn test_prices_survive_telegram_html() {
        let result = markdown_to_telegram_html("Price is $50 and $60");
        assert_eq!(result.trim(), "Price is $50 and $60");
    }

    #[test]
    fn test_strip_latex_in_telegram_html() {
        let result = markdown_to_telegram_html("The answer is $247 \\times 38 = 9386$.");
        assert!(result.contains("247 × 38 = 9386"));
        assert!(!result.contains('$'));
        assert!(!result.contains("\\times"));
    }

    mod proptest_formatting {
        use super::*;
        use aidaemon_channel_telegram::format::tg_len;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn markdown_to_telegram_never_panics(md in "\\PC{0,500}") {
                let _ = markdown_to_telegram_html(&md);
            }

            #[test]
            fn split_message_never_panics(text in "\\PC{0,2000}", max_len in 100usize..5000) {
                let parts = split_message(&text, max_len);
                assert_eq!(parts.is_empty(), text.trim().is_empty());
            }

            #[test]
            fn telegram_text_parts_fit_the_utf16_limit(text in "\\PC{0,3000}", max_len in 100usize..1000) {
                for part in split_telegram_text(&text, max_len) {
                    assert!(tg_len(&part) <= max_len, "{} > {max_len}", tg_len(&part));
                }
            }
        }
    }
}
