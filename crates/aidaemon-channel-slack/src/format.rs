//! Markdown to Slack mrkdwn conversion and message splitting.
//!
//! Ported from the daemon's `src/channels/formatting.rs`, with escaping of
//! the three characters Slack treats as control characters (`&`, `<`, `>`)
//! and a character-aware splitter that keeps code fences and `<...>` tokens
//! intact.

/// Escape the characters Slack reserves for control sequences.
pub fn escape_mrkdwn(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    push_escaped(&mut out, s);
    out
}

fn push_escaped(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            other => out.push(other),
        }
    }
}

/// Convert Markdown (as produced by models) to Slack mrkdwn.
///
/// - `**bold**` / `__bold__` → `*bold*`, `*italic*` → `_italic_`,
///   `~~strike~~` → `~strike~`
/// - `[text](url)` → `<url|text>`, `<https://...>` autolinks are kept
/// - headings become bold lines, `-`/`*`/`+` bullets become `•`
/// - `> quote` lines stay quotes
/// - fenced and inline code pass through (language tags are dropped)
/// - `&`, `<`, `>` are escaped everywhere else; existing user and channel
///   mention tokens (`<@U...>`, `<#C...>`) are preserved, broadcast tokens
///   such as `<!channel>` are not.
pub fn markdown_to_mrkdwn(md: &str) -> String {
    let mut out = String::with_capacity(md.len() + md.len() / 4);
    let mut in_fence = false;

    for line in md.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            // Slack has no syntax highlighting: drop any language tag.
            out.push_str("```");
            in_fence = !in_fence;
            out.push('\n');
            continue;
        }
        if in_fence {
            push_escaped(&mut out, line);
            out.push('\n');
            continue;
        }

        if let Some(heading) = strip_heading(trimmed) {
            out.push('*');
            out.push_str(&convert_inline(heading));
            out.push('*');
            out.push('\n');
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix('>') {
            out.push('>');
            out.push_str(&convert_inline(rest));
            out.push('\n');
            continue;
        }

        let indent = &line[..line.len() - trimmed.len()];
        if let Some(rest) = ["- ", "* ", "+ "]
            .iter()
            .find_map(|marker| trimmed.strip_prefix(marker))
        {
            out.push_str(indent);
            out.push_str("• ");
            out.push_str(&convert_inline(rest));
            out.push('\n');
            continue;
        }

        out.push_str(&convert_inline(line));
        out.push('\n');
    }

    if out.ends_with('\n') {
        out.pop();
    }
    out
}

fn strip_heading(line: &str) -> Option<&str> {
    let hashes = line.chars().take_while(|&c| c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &line[hashes..];
    if !rest.starts_with(' ') {
        return None;
    }
    let rest = rest.trim();
    (!rest.is_empty()).then_some(rest)
}

fn convert_inline(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        // Inline code: copy through, escaped.
        if c == '`'
            && let Some(end) = find_char(&chars, '`', i + 1)
        {
            out.push('`');
            push_escaped(&mut out, &collect(&chars[i + 1..end]));
            out.push('`');
            i = end + 1;
            continue;
        }

        // Bold: **x** or __x__ → *x*
        if (c == '*' || c == '_')
            && i + 1 < len
            && chars[i + 1] == c
            && let Some(end) = find_double(&chars, c, i + 2)
            && end > i + 2
        {
            out.push('*');
            out.push_str(&convert_inline(&collect(&chars[i + 2..end])));
            out.push('*');
            i = end + 2;
            continue;
        }

        // Strikethrough: ~~x~~ → ~x~
        if c == '~'
            && i + 1 < len
            && chars[i + 1] == '~'
            && let Some(end) = find_double(&chars, '~', i + 2)
            && end > i + 2
        {
            out.push('~');
            out.push_str(&convert_inline(&collect(&chars[i + 2..end])));
            out.push('~');
            i = end + 2;
            continue;
        }

        // Italic: *x* → _x_ (opening star must touch a word).
        if c == '*'
            && i + 1 < len
            && !chars[i + 1].is_whitespace()
            && let Some(end) = find_single_star(&chars, i + 1)
            && !chars[end - 1].is_whitespace()
        {
            out.push('_');
            out.push_str(&convert_inline(&collect(&chars[i + 1..end])));
            out.push('_');
            i = end + 1;
            continue;
        }

        // Link: [text](url) → <url|text>
        if c == '['
            && let Some((text, url, end)) = parse_link(&chars, i)
        {
            out.push('<');
            push_escaped(&mut out, &sanitize_url(&url));
            out.push('|');
            push_escaped(&mut out, &text.replace('|', "/"));
            out.push('>');
            i = end;
            continue;
        }

        // `<...>` tokens: autolinks and user/channel mentions survive.
        if c == '<'
            && let Some(end) = find_char(&chars, '>', i + 1)
        {
            let inner = collect(&chars[i + 1..end]);
            if is_preserved_token(&inner) {
                out.push('<');
                if inner.starts_with('@') || inner.starts_with('#') {
                    out.push_str(&inner);
                } else {
                    push_escaped(&mut out, &sanitize_url(&inner));
                }
                out.push('>');
                i = end + 1;
                continue;
            }
        }

        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            other => out.push(other),
        }
        i += 1;
    }
    out
}

fn is_preserved_token(inner: &str) -> bool {
    let id_like = |rest: &str| {
        let id = rest.split('|').next().unwrap_or("");
        !id.is_empty()
            && id
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
    };
    if let Some(rest) = inner.strip_prefix('@') {
        return id_like(rest);
    }
    if let Some(rest) = inner.strip_prefix('#') {
        return id_like(rest);
    }
    (inner.starts_with("http://") || inner.starts_with("https://") || inner.starts_with("mailto:"))
        && !inner.chars().any(char::is_whitespace)
}

fn sanitize_url(url: &str) -> String {
    url.chars()
        .filter(|c| !matches!(c, '<' | '>' | '|') && !c.is_whitespace())
        .collect()
}

fn collect(chars: &[char]) -> String {
    chars.iter().collect()
}

fn find_char(chars: &[char], c: char, start: usize) -> Option<usize> {
    (start..chars.len()).find(|&j| chars[j] == c)
}

fn find_double(chars: &[char], c: char, start: usize) -> Option<usize> {
    (start..chars.len().saturating_sub(1)).find(|&j| chars[j] == c && chars[j + 1] == c)
}

fn find_single_star(chars: &[char], start: usize) -> Option<usize> {
    (start..chars.len()).find(|&j| {
        chars[j] == '*' && (j + 1 >= chars.len() || chars[j + 1] != '*') && chars[j - 1] != '*'
    })
}

fn parse_link(chars: &[char], start: usize) -> Option<(String, String, usize)> {
    let close_bracket = find_char(chars, ']', start + 1)?;
    if close_bracket + 1 >= chars.len() || chars[close_bracket + 1] != '(' {
        return None;
    }
    let close_paren = find_char(chars, ')', close_bracket + 2)?;
    let text = collect(&chars[start + 1..close_bracket]);
    let url = collect(&chars[close_bracket + 2..close_paren]);
    if url.is_empty() {
        return None;
    }
    Some((text, url, close_paren + 1))
}

const FENCE: &str = "```";

/// Split mrkdwn into chunks of at most `max_chars` characters.
///
/// Prefers paragraph, then line, then word boundaries. Never splits inside
/// a `<...>` token, and closes/reopens a code fence that spans a split so
/// every chunk renders on its own.
pub fn split_mrkdwn(text: &str, max_chars: usize) -> Vec<String> {
    // Room for a closing fence ("\n```") plus at least some content.
    let max_chars = max_chars.max(16);
    let mut chunks = Vec::new();
    let mut rest = text.to_string();
    let mut reopen = false;

    loop {
        let current = if reopen {
            format!("{FENCE}\n{rest}")
        } else {
            std::mem::take(&mut rest)
        };
        if current.chars().count() <= max_chars {
            if !current.trim().is_empty() {
                chunks.push(current);
            }
            break;
        }

        let budget = max_chars - (FENCE.len() + 1);
        let limit = current
            .char_indices()
            .nth(budget)
            .map_or(current.len(), |(b, _)| b);
        let region = &current[..limit];

        let mut split_at_space = false;
        let mut split = region
            .rfind("\n\n")
            .map(|p| p + 1)
            .or_else(|| region.rfind('\n'))
            .filter(|&p| p > 0);
        if split.is_none() {
            split = region.rfind(' ').filter(|&p| p > 0);
            split_at_space = split.is_some();
        }
        let mut split = split.unwrap_or(limit);

        // Do not cut a `<url|text>` or `<@U...>` token in half.
        if let Some(lt) = current[..split].rfind('<')
            && lt > 0
            && !current[lt..split].contains('>')
        {
            split = lt;
            split_at_space = false;
        }

        let mut chunk = current[..split].trim_end().to_string();
        let mut remainder = &current[split..];
        if split_at_space {
            remainder = remainder.strip_prefix(' ').unwrap_or(remainder);
        }
        let remainder = remainder.trim_start_matches('\n');

        reopen = chunk.matches(FENCE).count() % 2 == 1;
        if reopen {
            chunk.push('\n');
            chunk.push_str(FENCE);
        }
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }
        rest = remainder.to_string();
        if rest.is_empty() {
            break;
        }
    }

    if chunks.is_empty() {
        chunks.push(String::new());
    }
    chunks
}

/// Truncate to `max_chars` characters, marking the cut with an ellipsis.
pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bold_italic_strike_and_headings() {
        assert_eq!(markdown_to_mrkdwn("**bold** text"), "*bold* text");
        assert_eq!(markdown_to_mrkdwn("__bold__"), "*bold*");
        assert_eq!(markdown_to_mrkdwn("an *italic* word"), "an _italic_ word");
        assert_eq!(markdown_to_mrkdwn("~~gone~~"), "~gone~");
        assert_eq!(markdown_to_mrkdwn("## Status"), "*Status*");
        assert_eq!(markdown_to_mrkdwn("#hashtag"), "#hashtag");
        assert_eq!(markdown_to_mrkdwn("2 * 3 * 4"), "2 * 3 * 4");
    }

    #[test]
    fn links_and_tokens() {
        assert_eq!(
            markdown_to_mrkdwn("see [the docs](https://example.com/a?x=1&y=2)"),
            "see <https://example.com/a?x=1&amp;y=2|the docs>"
        );
        assert_eq!(
            markdown_to_mrkdwn("<https://example.com>"),
            "<https://example.com>"
        );
        assert_eq!(
            markdown_to_mrkdwn("hi <@U000ALICE> in <#C000SYNTH>"),
            "hi <@U000ALICE> in <#C000SYNTH>"
        );
        assert_eq!(markdown_to_mrkdwn("<!channel> now"), "&lt;!channel&gt; now");
    }

    #[test]
    fn escapes_control_characters() {
        assert_eq!(
            markdown_to_mrkdwn("a < b && c > d"),
            "a &lt; b &amp;&amp; c &gt; d"
        );
        assert_eq!(markdown_to_mrkdwn("`x<y`"), "`x&lt;y`");
        assert_eq!(
            markdown_to_mrkdwn("```rust\nif a < b {}\n```"),
            "```\nif a &lt; b {}\n```"
        );
        assert_eq!(markdown_to_mrkdwn("> quoted & kept"), "> quoted &amp; kept");
    }

    #[test]
    fn lists_become_bullets() {
        assert_eq!(
            markdown_to_mrkdwn("- one\n* **two**\n  + three"),
            "• one\n• *two*\n  • three"
        );
    }

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(split_mrkdwn("hello", 3000), vec!["hello".to_string()]);
    }

    #[test]
    fn splits_on_paragraphs_within_limit() {
        let para = "word ".repeat(300); // 1500 chars
        let text = format!("{para}\n\n{para}\n\n{para}");
        let chunks = split_mrkdwn(&text, 3000);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 3000);
        }
        let rejoined = chunks.join("\n");
        assert_eq!(
            rejoined.split_whitespace().count(),
            text.split_whitespace().count()
        );
    }

    #[test]
    fn split_counts_characters_not_bytes() {
        let text = "é".repeat(5000);
        let chunks = split_mrkdwn(&text, 3000);
        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().all(|c| c.chars().count() <= 3000));
        assert_eq!(
            chunks.iter().map(|c| c.chars().count()).sum::<usize>(),
            5000
        );
    }

    #[test]
    fn split_keeps_code_fences_balanced() {
        let body = "let x = 1;\n".repeat(600);
        let text = format!("```\n{body}```");
        let chunks = split_mrkdwn(&text, 3000);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 3000, "{}", chunk.len());
            assert_eq!(chunk.matches("```").count() % 2, 0, "unbalanced: {chunk}");
        }
    }

    #[test]
    fn split_never_cuts_a_link_token() {
        let link = "<https://example.com/very/long/path|label>";
        let text = format!("{}{}", "x".repeat(2990), link);
        let chunks = split_mrkdwn(&text, 3000);
        assert!(chunks.iter().any(|c| c.contains(link)));
    }

    #[test]
    fn truncate_marks_cut() {
        assert_eq!(truncate_chars("abcdef", 4), "abc…");
        assert_eq!(truncate_chars("abc", 4), "abc");
    }
}
