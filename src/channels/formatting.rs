/// Convert common LLM markdown to Telegram-compatible HTML.
pub(crate) fn markdown_to_telegram_html(md: &str) -> String {
    let mut result = String::with_capacity(md.len() + md.len() / 4);
    let lines: Vec<&str> = md.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Fenced code blocks: ```lang\n...\n```
        if line.starts_with("```") {
            i += 1;
            let mut code = String::new();
            while i < lines.len() && !lines[i].starts_with("```") {
                if !code.is_empty() {
                    code.push('\n');
                }
                code.push_str(lines[i]);
                i += 1;
            }
            if i < lines.len() {
                i += 1; // skip closing ```
            }
            // HTML-escape the code content
            let escaped = html_escape(&code);
            result.push_str("<pre><code>");
            result.push_str(&escaped);
            result.push_str("</code></pre>");
            result.push('\n');
            continue;
        }

        // Markdown table: collect consecutive lines starting with |
        if line.trim_start().starts_with('|') {
            let mut table_lines = Vec::new();
            while i < lines.len() && lines[i].trim_start().starts_with('|') {
                table_lines.push(lines[i]);
                i += 1;
            }
            // Parse table: first line = headers, skip separator lines (contain ---)
            let parse_row = |row: &str| -> Vec<String> {
                row.split('|')
                    .map(|cell| cell.trim().to_string())
                    .filter(|cell| !cell.is_empty())
                    .collect()
            };
            let is_separator = |row: &str| -> bool { row.contains("---") || row.contains(":--") };

            let headers: Vec<String> = if !table_lines.is_empty() {
                parse_row(table_lines[0])
            } else {
                Vec::new()
            };

            let data_rows: Vec<Vec<String>> = table_lines
                .iter()
                .skip(1)
                .filter(|line| !is_separator(line))
                .map(|line| parse_row(line))
                .collect();

            // Format each row as a card-style block
            for (ri, row) in data_rows.iter().enumerate() {
                if ri > 0 {
                    result.push('\n');
                }
                for (ci, cell) in row.iter().enumerate() {
                    let escaped_cell = html_escape(cell);
                    let formatted_cell = convert_inline_formatting(&escaped_cell);
                    if ci < headers.len() {
                        let escaped_header = html_escape(&headers[ci]);
                        result
                            .push_str(&format!("<b>{}</b>: {}\n", escaped_header, formatted_cell));
                    } else {
                        result.push_str(&formatted_cell);
                        result.push('\n');
                    }
                }
            }
            continue;
        }

        // Process a non-code line: escape HTML first, then apply inline formatting
        let escaped = html_escape(line);

        // Heading lines: ### heading → <b>heading</b>
        if let Some(heading) = strip_heading(&escaped) {
            result.push_str("<b>");
            result.push_str(&heading);
            result.push_str("</b>");
            result.push('\n');
            i += 1;
            continue;
        }

        // Unordered list markers: "- " or "* " at start → "• "
        let processed = if let Some(rest) = escaped
            .strip_prefix("- ")
            .or_else(|| escaped.strip_prefix("* "))
        {
            format!("• {}", rest)
        } else {
            escaped
        };

        // Inline formatting
        let processed = convert_inline_formatting(&processed);

        result.push_str(&processed);
        result.push('\n');
        i += 1;
    }

    // Remove trailing newline
    if result.ends_with('\n') {
        result.pop();
    }
    result
}

/// Escape `<`, `>`, `&` for Telegram HTML.
pub(crate) fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
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

/// Apply inline markdown formatting: bold, italic, inline code, links.
fn convert_inline_formatting(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Inline code: `code`
        if chars[i] == '`' {
            if let Some(end) = find_char(&chars, '`', i + 1) {
                result.push_str("<code>");
                let inner: String = chars[i + 1..end].iter().collect();
                result.push_str(&inner);
                result.push_str("</code>");
                i = end + 1;
                continue;
            }
        }

        // Bold: **text**
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_double_char(&chars, '*', i + 2) {
                result.push_str("<b>");
                let inner: String = chars[i + 2..end].iter().collect();
                result.push_str(&inner);
                result.push_str("</b>");
                i = end + 2;
                continue;
            }
        }

        // Link: [text](url)
        if chars[i] == '[' {
            if let Some((text, url, end)) = parse_link(&chars, i) {
                result.push_str("<a href=\"");
                result.push_str(&url);
                result.push_str("\">");
                result.push_str(&text);
                result.push_str("</a>");
                i = end;
                continue;
            }
        }

        // Italic: _text_ (but not inside words like some_var_name)
        if chars[i] == '_' && (i == 0 || chars[i - 1] == ' ') {
            if let Some(end) = find_char(&chars, '_', i + 1) {
                if end + 1 >= len
                    || chars[end + 1] == ' '
                    || chars[end + 1] == '.'
                    || chars[end + 1] == ','
                {
                    result.push_str("<i>");
                    let inner: String = chars[i + 1..end].iter().collect();
                    result.push_str(&inner);
                    result.push_str("</i>");
                    i = end + 1;
                    continue;
                }
            }
        }

        // Single *italic* (not **)
        if chars[i] == '*' && (i + 1 >= len || chars[i + 1] != '*') {
            if let Some(end) = find_single_star(&chars, i + 1) {
                result.push_str("<i>");
                let inner: String = chars[i + 1..end].iter().collect();
                result.push_str(&inner);
                result.push_str("</i>");
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

/// Convert common LLM markdown to Slack mrkdwn format.
///
/// Slack mrkdwn differs from standard markdown:
/// - Bold: `*bold*` (single asterisk, not double)
/// - Italic: `_italic_`
/// - Code: `` `code` `` (same)
/// - Code block: ``` ```code``` ``` (same)
/// - Links: `<url|text>` instead of `[text](url)`
/// - Headings: `*Heading*` (bold, no # prefix)
/// - Lists: use `•` for unordered
#[cfg(feature = "slack")]
pub(crate) fn markdown_to_slack_mrkdwn(md: &str) -> String {
    let mut result = String::with_capacity(md.len() + md.len() / 4);
    let lines: Vec<&str> = md.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Fenced code blocks: ```lang\n...\n``` — pass through as-is
        if line.starts_with("```") {
            result.push_str(line);
            result.push('\n');
            i += 1;
            while i < lines.len() && !lines[i].starts_with("```") {
                result.push_str(lines[i]);
                result.push('\n');
                i += 1;
            }
            if i < lines.len() {
                result.push_str(lines[i]);
                result.push('\n');
                i += 1;
            }
            continue;
        }

        // Heading lines: ### heading → *heading* (bold in Slack)
        if line.starts_with('#') {
            let trimmed = line.trim_start_matches('#').trim_start();
            if !trimmed.is_empty() {
                result.push('*');
                result.push_str(&convert_slack_inline(trimmed));
                result.push('*');
                result.push('\n');
                i += 1;
                continue;
            }
        }

        // Unordered list markers: "- " or "* " at start → "• "
        let processed =
            if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
                format!("• {}", rest)
            } else {
                line.to_string()
            };

        // Apply inline formatting conversions
        let processed = convert_slack_inline(&processed);

        result.push_str(&processed);
        result.push('\n');
        i += 1;
    }

    // Remove trailing newline
    if result.ends_with('\n') {
        result.pop();
    }
    result
}

/// Convert inline markdown to Slack mrkdwn.
/// - `**bold**` → `*bold*`
/// - `[text](url)` → `<url|text>`
/// - Inline code stays the same
#[cfg(feature = "slack")]
fn convert_slack_inline(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Inline code: `code` — pass through
        if chars[i] == '`' {
            if let Some(end) = find_char(&chars, '`', i + 1) {
                let span: String = chars[i..=end].iter().collect();
                result.push_str(&span);
                i = end + 1;
                continue;
            }
        }

        // Bold: **text** → *text*
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_double_char(&chars, '*', i + 2) {
                result.push('*');
                let inner: String = chars[i + 2..end].iter().collect();
                result.push_str(&inner);
                result.push('*');
                i = end + 2;
                continue;
            }
        }

        // Link: [text](url) → <url|text>
        if chars[i] == '[' {
            if let Some((text, url, end)) = parse_link(&chars, i) {
                result.push('<');
                result.push_str(&url);
                result.push('|');
                result.push_str(&text);
                result.push('>');
                i = end;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }
    result
}

/// Build the /help response text.
///
/// `include_restart`: whether to show the /restart command (Telegram, Slack)
/// `include_connect`: whether to show /connect and /bots (Telegram only)
/// `prefix`: command prefix character ("/" for Telegram, "!" for Slack)
pub(crate) fn build_help_text(include_restart: bool, include_connect: bool, prefix: &str) -> String {
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

    text.push_str(&format!(
        "\n\
         `{p}model` — Show or switch AI model\n\
         `{p}models` — List available models\n\
         `{p}auto` — Re-enable automatic model routing\n\
         `{p}reload` — Reload configuration",
    ));

    if include_restart {
        text.push_str(&format!("\n`{p}restart` — Restart the daemon"));
    }

    text.push_str(&format!(
        "\n\
         `{p}tasks` — List running tasks\n\
         `{p}cancel <id>` — Cancel a running task\n\
         `{p}clear` — Start fresh conversation\n\
         `{p}cost` — Show token usage stats",
    ));

    if include_connect {
        text.push_str(&format!(
            "\n\
             `{p}connect` — Add a new bot\n\
             `{p}bots` — List connected bots",
        ));
    }

    text.push_str(&format!("\n`{p}help` — Show this message"));

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
            format!("{}{}", &sanitized[..stem_len], ext)
        } else {
            sanitized[..200].to_string()
        }
    } else {
        sanitized[..200].to_string()
    }
}

#[cfg(test)]
mod tests {
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
    fn test_markdown_to_telegram_code_block() {
        let result = markdown_to_telegram_html("```rust\nfn main() {}\n```");
        assert!(result.contains("<pre><code>"));
        assert!(result.contains("fn main()"));
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
