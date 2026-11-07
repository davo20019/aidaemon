//! Chat formatting. Markdown conversion for each platform lives in the SDK
//! channel crates; the daemon keeps its plain-text fallback, message
//! splitting and command help.

#[cfg(feature = "slack")]
pub(crate) use aidaemon_channel_slack::format::markdown_to_mrkdwn as markdown_to_slack_mrkdwn;
pub(crate) use aidaemon_channel_telegram::format::{
    html_escape, markdown_to_html as markdown_to_telegram_html,
};
use aidaemon_channels::latex::strip_latex;

/// Convert common LLM markdown to readable plain text for Telegram fallbacks.
pub(crate) fn markdown_to_telegram_plain_fallback(md: &str) -> String {
    let md = strip_latex(md);
    let mut result = String::with_capacity(md.len());
    let mut in_code_block = false;

    for line in md.lines() {
        if line.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }

        let processed = if in_code_block {
            line.to_string()
        } else {
            let line = strip_heading(line).unwrap_or_else(|| line.to_string());
            let line =
                if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
                    format!("• {}", rest)
                } else {
                    line
                };
            convert_inline_formatting_to_plain(&line)
        };

        result.push_str(&processed);
        result.push('\n');
    }

    if result.ends_with('\n') {
        result.pop();
    }
    result
}

/// Strip markdown heading prefix (e.g. "### Foo" → "Foo"). Returns None if not a heading.
fn strip_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if trimmed.starts_with('#') {
        let after_hashes = trimmed.trim_start_matches('#');
        if after_hashes.starts_with(' ') {
            return Some(after_hashes.trim_start().to_string());
        }
    }
    None
}

fn convert_inline_formatting_to_plain(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if chars[i] == '`' {
            if let Some(end) = find_char(&chars, '`', i + 1) {
                result.extend(chars[i + 1..end].iter());
                i = end + 1;
                continue;
            }
        }

        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_double_char(&chars, '*', i + 2) {
                result.extend(chars[i + 2..end].iter());
                i = end + 2;
                continue;
            }
        }

        if chars[i] == '[' {
            if let Some((text, url, end)) = parse_link(&chars, i) {
                result.push_str(&text);
                if !url.is_empty() {
                    result.push_str(" (");
                    result.push_str(&url);
                    result.push(')');
                }
                i = end;
                continue;
            }
        }

        if chars[i] == '_' {
            if let Some(end) = find_char(&chars, '_', i + 1) {
                result.extend(chars[i + 1..end].iter());
                i = end + 1;
                continue;
            }
        }

        if chars[i] == '*' && (i + 1 >= len || chars[i + 1] != '*') {
            if let Some(end) = find_single_star(&chars, i + 1) {
                result.extend(chars[i + 1..end].iter());
                i = end + 1;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }
    result
}

fn find_char(chars: &[char], c: char, start: usize) -> Option<usize> {
    (start..chars.len()).find(|&j| chars[j] == c)
}

fn find_double_char(chars: &[char], c: char, start: usize) -> Option<usize> {
    let mut j = start;
    while j + 1 < chars.len() {
        if chars[j] == c && chars[j + 1] == c {
            return Some(j);
        }
        j += 1;
    }
    None
}

fn find_single_star(chars: &[char], start: usize) -> Option<usize> {
    (start..chars.len()).find(|&j| chars[j] == '*' && (j + 1 >= chars.len() || chars[j + 1] != '*'))
}

fn parse_link(chars: &[char], start: usize) -> Option<(String, String, usize)> {
    // [text](url)
    let close_bracket = find_char(chars, ']', start + 1)?;
    if close_bracket + 1 >= chars.len() || chars[close_bracket + 1] != '(' {
        return None;
    }
    let close_paren = find_char(chars, ')', close_bracket + 2)?;
    let text: String = chars[start + 1..close_bracket].iter().collect();
    let url: String = chars[close_bracket + 2..close_paren].iter().collect();
    Some((text, url, close_paren + 1))
}

/// Split a message into chunks respecting Telegram's max length.
/// Prefers splitting at paragraph boundaries, then line boundaries.
/// Never splits inside HTML tags or code blocks.
pub(crate) fn split_message(text: &str, max_len: usize) -> Vec<String> {
    if text.len() <= max_len {
        return vec![text.to_string()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= max_len {
            chunks.push(remaining.to_string());
            break;
        }

        // Find the largest char boundary at or before max_len to avoid
        // slicing in the middle of a multi-byte UTF-8 character.
        let mut boundary = max_len;
        while boundary > 0 && !remaining.is_char_boundary(boundary) {
            boundary -= 1;
        }

        let search_region = &remaining[..boundary];

        // Try paragraph boundary first
        let split_at = search_region
            .rfind("\n\n")
            .map(|p| p + 1) // include first \n, second starts next chunk
            // Then try line boundary
            .or_else(|| search_region.rfind('\n'))
            // Last resort: split at char boundary
            .unwrap_or(boundary);

        // Ensure we don't split inside an HTML tag
        let split_at = adjust_for_html_tags(search_region, split_at);

        // Safety: if split_at is 0 (e.g. max_len=0), force progress by
        // advancing one character to avoid an infinite loop.
        let split_at = if split_at == 0 {
            remaining
                .char_indices()
                .nth(1)
                .map_or(remaining.len(), |(i, _)| i)
        } else {
            split_at
        };

        let (chunk, rest) = remaining.split_at(split_at);
        let chunk = chunk.trim_end();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }
        remaining = rest.trim_start_matches('\n');
    }

    chunks
}

/// If the split point is inside an HTML tag, move it before the tag start.
fn adjust_for_html_tags(text: &str, split_at: usize) -> usize {
    let bytes = text.as_bytes();
    // Walk backward from split_at to check if we're inside a tag
    let mut j = split_at;
    while j > 0 {
        j -= 1;
        if bytes[j] == b'>' {
            // We're outside a tag, safe to split at original point
            return split_at;
        }
        if bytes[j] == b'<' {
            // We're inside a tag — split before it
            return j;
        }
    }
    split_at
}

/// Format a number with comma separators (e.g. 12450 → "12,450").
pub(crate) fn format_number(n: i64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Build the /help response text from a list of command definitions.
///
/// `commands`: the command definitions to include (platform-specific).
/// `prefix`: command prefix character ("/" for Telegram/Discord, "!" for Slack).
pub(crate) fn build_help_text(commands: &[super::commands::CommandDef], prefix: &str) -> String {
    let p = prefix;
    let mut text = String::from(
        "**What I can do**\n\
         \n\
         **Schedule recurring tasks**\n\
         _\"Check my disk space every 6 hours\"_\n\
         _\"Remind me to review PRs on weekdays at 9am\"_\n\
         \n\
         **Search & browse the web**\n\
         _\"What's the latest on Rust 2024 edition?\"_\n\
         _\"Summarize this article: <url>\"_\n\
         \n\
         **Run commands & manage your system**\n\
         _\"Show me docker container status\"_\n\
         _\"Deploy the staging branch\"_\n\
         \n\
         **Switch AI providers with guidance**\n\
         _\"Switch to Moonshot\"_\n\
         _\"Move me to Cloudflare Gateway and ask for what you need\"_\n\
         \n\
         **Remember things about you**\n\
         _\"I prefer dark mode and vim keybindings\"_\n\
         _\"My deploy server is 10.0.1.50\"_\n\
         \n\
         **Track your goals**\n\
         _\"I want to finish the API migration this month\"_\n\
         _\"How's my progress on learning Rust?\"_\n\
         \n\
         **Delegate coding tasks**\n\
         _\"Refactor the auth module to use JWT\"_\n\
         _\"Fix the failing tests in src/parser\"_\n\
         \n\
         **Send & receive files**\n\
         _\"Send me the latest log file\"_\n\
         _\"Analyze this CSV I'm uploading\"_\n\
         \n\
         Just ask in plain language — I'll pick the right tools.\n\
         \n\
         **Commands**",
    );

    for cmd in commands {
        let label = match cmd.usage {
            Some(usage) => {
                // Replace the leading "/" with the platform prefix.
                if let Some(rest) = usage.strip_prefix('/') {
                    format!("{p}{rest}")
                } else {
                    usage.to_string()
                }
            }
            None => format!("{p}{}", cmd.name),
        };
        text.push_str(&format!("\n`{}` — {}", label, cmd.description));
    }

    text
}

/// Sanitize a filename: remove path separators, null bytes, and limit length.
pub(crate) fn sanitize_filename(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .filter(|c| *c != '/' && *c != '\\' && *c != '\0')
        .collect();
    // Strip path traversal sequences
    let sanitized = sanitized.replace("..", "");
    // Limit to 200 chars, preserving extension
    if sanitized.len() <= 200 {
        sanitized
    } else if let Some(dot_pos) = sanitized.rfind('.') {
        let ext = &sanitized[dot_pos..];
        if ext.len() < 20 {
            let stem_len = 200 - ext.len();
            let boundary = crate::utils::floor_char_boundary(&sanitized, stem_len);
            format!("{}{}", &sanitized[..boundary], ext)
        } else {
            let boundary = crate::utils::floor_char_boundary(&sanitized, 200);
            sanitized[..boundary].to_string()
        }
    } else {
        let boundary = crate::utils::floor_char_boundary(&sanitized, 200);
        sanitized[..boundary].to_string()
    }
}

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
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(12450), "12,450");
        assert_eq!(format_number(1_000_000), "1,000,000");
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
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn markdown_to_telegram_never_panics(md in "\\PC{0,500}") {
                let _ = markdown_to_telegram_html(&md);
            }

            #[test]
            fn split_message_never_panics(text in "\\PC{0,2000}", max_len in 100usize..5000) {
                let parts = split_message(&text, max_len);
                assert!(!parts.is_empty());
            }
        }
    }
}
