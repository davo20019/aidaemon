//! Markdown to Telegram HTML conversion and tag-aware message splitting.
//!
//! Telegram's HTML parse mode accepts a small tag set (`b`, `i`, `s`, `code`,
//! `pre`, `a`, ...) and requires `<`, `>` and `&` to be escaped everywhere
//! else. LLM output is loose Markdown, so the converter is deliberately
//! forgiving: anything it does not recognise is passed through escaped.

pub use crate::plain::{markdown_to_plain_fallback, split_text};

/// Escape `<`, `>` and `&` for Telegram HTML text.
pub fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

/// Convert common LLM Markdown to Telegram-compatible HTML.
///
/// Supports fenced code blocks (with language), headings (rendered bold),
/// `-`/`*` bullets, tables (rendered as `header: value` cards), and inline
/// `**bold**`, `*italic*`/`_italic_`, `~~strike~~`, `` `code` `` and
/// `[text](url)` links. LaTeX math, which Telegram cannot render, is
/// rewritten as plain text first ([`aidaemon_channels::latex::strip_latex`]).
pub fn markdown_to_html(md: &str) -> String {
    render_html(&aidaemon_channels::latex::strip_latex(md))
}

/// [`markdown_to_html`] with the LaTeX left to the caller.
pub(crate) fn render_html(md: &str) -> String {
    let mut out = String::with_capacity(md.len() + md.len() / 4);
    let lines: Vec<&str> = md.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        if let Some(info) = line.trim_start().strip_prefix("```") {
            let lang = info
                .split_whitespace()
                .next()
                .filter(|l| {
                    l.chars()
                        .all(|c| c.is_ascii_alphanumeric() || "+-_#.".contains(c))
                })
                .unwrap_or("");
            i += 1;
            let mut code = String::new();
            let mut first = true;
            while i < lines.len() && !lines[i].trim_start().starts_with("```") {
                if !first {
                    code.push('\n');
                }
                first = false;
                code.push_str(lines[i]);
                i += 1;
            }
            if i < lines.len() {
                i += 1; // closing fence
            }
            if lang.is_empty() {
                out.push_str("<pre><code>");
            } else {
                out.push_str("<pre><code class=\"language-");
                out.push_str(lang);
                out.push_str("\">");
            }
            out.push_str(&html_escape(&code));
            out.push_str("</code></pre>\n");
            continue;
        }

        if line.trim_start().starts_with('|') {
            let mut table = Vec::new();
            while i < lines.len() && lines[i].trim_start().starts_with('|') {
                table.push(lines[i]);
                i += 1;
            }
            render_table(&table, &mut out);
            continue;
        }

        let escaped = html_escape(line);
        if let Some(heading) = strip_heading(&escaped) {
            out.push_str("<b>");
            out.push_str(&convert_inline(heading));
            out.push_str("</b>\n");
            i += 1;
            continue;
        }

        let trimmed = escaped.trim_start();
        let indent = &escaped[..escaped.len() - trimmed.len()];
        let processed = match trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            Some(rest) => format!("{indent}• {}", convert_inline(rest)),
            None => convert_inline(&escaped),
        };
        out.push_str(&processed);
        out.push('\n');
        i += 1;
    }

    if out.ends_with('\n') {
        out.pop();
    }
    out
}

fn render_table(table: &[&str], out: &mut String) {
    let parse_row = |row: &str| -> Vec<String> {
        row.split('|')
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect()
    };
    let is_separator = |row: &str| {
        row.chars()
            .all(|c| matches!(c, '|' | '-' | ':' | ' ' | '\t'))
    };
    let headers = table.first().map(|r| parse_row(r)).unwrap_or_default();
    let rows: Vec<Vec<String>> = table
        .iter()
        .skip(1)
        .filter(|r| !is_separator(r))
        .map(|r| parse_row(r))
        .collect();
    if rows.is_empty() {
        // Header-only table: show the header cells as one line.
        out.push_str(&convert_inline(&html_escape(&headers.join(" | "))));
        out.push('\n');
        return;
    }
    for (ri, row) in rows.iter().enumerate() {
        if ri > 0 {
            out.push('\n');
        }
        for (ci, cell) in row.iter().enumerate() {
            let cell = convert_inline(&html_escape(cell));
            match headers.get(ci) {
                Some(h) => {
                    out.push_str("<b>");
                    out.push_str(&html_escape(h));
                    out.push_str("</b>: ");
                    out.push_str(&cell);
                }
                None => out.push_str(&cell),
            }
            out.push('\n');
        }
    }
}

fn strip_heading(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let after = trimmed.trim_start_matches('#');
    if trimmed.len() - after.len() > 6 || !after.starts_with(' ') {
        return None;
    }
    Some(after.trim())
}

/// Inline formatting over already-escaped text.
fn convert_inline(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    convert_inline_chars(&chars, &mut out);
    out
}

fn convert_inline_chars(chars: &[char], out: &mut String) {
    let len = chars.len();
    let mut i = 0;
    while i < len {
        let c = chars[i];

        if c == '`' {
            if let Some(end) = find_char(chars, '`', i + 1) {
                out.push_str("<code>");
                out.extend(&chars[i + 1..end]);
                out.push_str("</code>");
                i = end + 1;
                continue;
            }
        }

        if (c == '*' || c == '~') && i + 1 < len && chars[i + 1] == c {
            if let Some(end) = find_double(chars, c, i + 2) {
                if end > i + 2 {
                    let tag = if c == '*' { "b" } else { "s" };
                    out.push('<');
                    out.push_str(tag);
                    out.push('>');
                    convert_inline_chars(&chars[i + 2..end], out);
                    out.push_str("</");
                    out.push_str(tag);
                    out.push('>');
                    i = end + 2;
                    continue;
                }
            }
        }

        if c == '[' {
            if let Some((text_end, url_start, url_end)) = parse_link(chars, i) {
                let url: String = chars[url_start..url_end].iter().collect();
                out.push_str("<a href=\"");
                out.push_str(&url.replace('"', "&quot;"));
                out.push_str("\">");
                convert_inline_chars(&chars[i + 1..text_end], out);
                out.push_str("</a>");
                i = url_end + 1;
                continue;
            }
        }

        // _italic_ only at word boundaries, so snake_case stays intact.
        if c == '_' && (i == 0 || !chars[i - 1].is_alphanumeric()) {
            if let Some(end) = find_char(chars, '_', i + 1) {
                let boundary_after = end + 1 >= len || !chars[end + 1].is_alphanumeric();
                if end > i + 1 && boundary_after && !chars[i + 1].is_whitespace() {
                    out.push_str("<i>");
                    convert_inline_chars(&chars[i + 1..end], out);
                    out.push_str("</i>");
                    i = end + 1;
                    continue;
                }
            }
        }

        if c == '*' && (i + 1 >= len || chars[i + 1] != '*') {
            if let Some(end) = find_single_star(chars, i + 1) {
                if end > i + 1 && !chars[i + 1].is_whitespace() {
                    out.push_str("<i>");
                    convert_inline_chars(&chars[i + 1..end], out);
                    out.push_str("</i>");
                    i = end + 1;
                    continue;
                }
            }
        }

        out.push(c);
        i += 1;
    }
}

fn find_char(chars: &[char], c: char, start: usize) -> Option<usize> {
    (start..chars.len()).find(|&j| chars[j] == c)
}

fn find_double(chars: &[char], c: char, start: usize) -> Option<usize> {
    (start..chars.len().saturating_sub(1)).find(|&j| chars[j] == c && chars[j + 1] == c)
}

fn find_single_star(chars: &[char], start: usize) -> Option<usize> {
    (start..chars.len()).find(|&j| chars[j] == '*' && (j + 1 >= chars.len() || chars[j + 1] != '*'))
}

/// `[text](url)` → (index of `]`, url start, index of `)`).
fn parse_link(chars: &[char], start: usize) -> Option<(usize, usize, usize)> {
    let close = find_char(chars, ']', start + 1)?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let paren = find_char(chars, ')', close + 2)?;
    if paren == close + 2 || chars[close + 2..paren].iter().any(|c| c.is_whitespace()) {
        return None;
    }
    Some((close, close + 2, paren))
}

/// Turn Telegram HTML back into readable plain text (tags removed, entities
/// decoded, links rendered as `text (url)`). Used when Telegram rejects the
/// HTML of a chunk.
pub fn html_to_plain(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut href_stack: Vec<Option<(String, usize)>> = Vec::new();
    for atom in atoms(html) {
        match atom {
            Atom::Tag(tag) => {
                let Some(info) = parse_tag(tag) else { continue };
                if info.name != "a" {
                    continue;
                }
                if info.closing {
                    if let Some(Some((href, start))) = href_stack.pop() {
                        if out[start..] != href[..] && !href.is_empty() {
                            out.push_str(" (");
                            out.push_str(&href);
                            out.push(')');
                        }
                    }
                } else {
                    let href = extract_href(tag).map(|h| decode_entities(&h));
                    href_stack.push(href.map(|h| (h, out.len())));
                }
            }
            Atom::Entity(e) => out.push_str(&decode_entities(e)),
            Atom::Char(c) => out.push(c),
        }
    }
    out
}

fn extract_href(tag: &str) -> Option<String> {
    let start = tag.find("href=\"")? + 6;
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

fn decode_entities(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&amp;", "&")
}

/// Telegram-relevant length of a string: UTF-16 code units, which is how the
/// Bot API counts message length.
pub fn tg_len(s: &str) -> usize {
    s.encode_utf16().count()
}

#[derive(Debug, Clone, Copy)]
enum Atom<'a> {
    Tag(&'a str),
    Entity(&'a str),
    Char(char),
}

impl Atom<'_> {
    fn len(&self) -> usize {
        match self {
            Atom::Tag(s) | Atom::Entity(s) => tg_len(s),
            Atom::Char(c) => c.len_utf16(),
        }
    }
}

fn atoms(html: &str) -> Vec<Atom<'_>> {
    let mut out = Vec::new();
    let mut rest = html;
    while let Some(c) = rest.chars().next() {
        if c == '<' {
            if let Some(end) = rest.find('>') {
                out.push(Atom::Tag(&rest[..=end]));
                rest = &rest[end + 1..];
                continue;
            }
        }
        if c == '&' {
            if let Some(end) = rest.find(';') {
                let body = &rest[1..end];
                if end <= 10
                    && !body.is_empty()
                    && body.chars().all(|c| c.is_ascii_alphanumeric() || c == '#')
                {
                    out.push(Atom::Entity(&rest[..=end]));
                    rest = &rest[end + 1..];
                    continue;
                }
            }
        }
        out.push(Atom::Char(c));
        rest = &rest[c.len_utf8()..];
    }
    out
}

struct TagInfo<'a> {
    name: &'a str,
    closing: bool,
}

fn parse_tag(tag: &str) -> Option<TagInfo<'_>> {
    let inner = tag.strip_prefix('<')?.strip_suffix('>')?;
    let (closing, inner) = match inner.strip_prefix('/') {
        Some(rest) => (true, rest),
        None => (false, inner),
    };
    let name = inner
        .split(|c: char| c.is_whitespace() || c == '/')
        .next()
        .filter(|n| !n.is_empty())?;
    Some(TagInfo { name, closing })
}

/// An open tag: its name and the raw opening tag used to reopen it.
type OpenTag<'a> = (&'a str, &'a str);

fn closing_tags(stack: &[OpenTag<'_>]) -> String {
    stack
        .iter()
        .rev()
        .map(|(name, _)| format!("</{name}>"))
        .collect()
}

fn closing_len(stack: &[OpenTag<'_>]) -> usize {
    stack.iter().map(|(name, _)| name.len() + 3).sum()
}

fn opening_tags(stack: &[OpenTag<'_>]) -> String {
    stack.iter().map(|(_, raw)| *raw).collect()
}

/// Split Telegram HTML into chunks of at most `max_len` UTF-16 units each.
///
/// Splits prefer paragraph breaks, then line breaks, then spaces. Tags open
/// at a split point are closed at the end of the chunk and reopened at the
/// start of the next one, so every chunk is well-formed on its own. Entities
/// such as `&amp;` are never cut. The length counts the markup too, which is
/// conservative (Telegram counts only the visible text).
pub fn split_html(html: &str, max_len: usize) -> Vec<String> {
    let max_len = max_len.max(32);
    if tg_len(html) <= max_len {
        return vec![html.to_string()];
    }
    let atoms = atoms(html);
    let mut chunks = Vec::new();
    let mut stack: Vec<OpenTag<'_>> = Vec::new();
    let mut i = 0;

    while i < atoms.len() {
        // Skip leading line breaks at a chunk start.
        while i < atoms.len() && matches!(atoms[i], Atom::Char('\n')) {
            i += 1;
        }
        if i >= atoms.len() {
            break;
        }

        let prefix = opening_tags(&stack);
        let mut used = tg_len(&prefix);
        let mut j = i;
        // Best split candidates: (atom index to split before, stack there).
        let mut para: Option<(usize, Vec<OpenTag<'_>>)> = None;
        let mut line: Option<(usize, Vec<OpenTag<'_>>)> = None;
        let mut space: Option<(usize, Vec<OpenTag<'_>>)> = None;
        let mut overflow = false;

        while j < atoms.len() {
            let atom = atoms[j];
            // Stack after this atom, to know closing cost.
            let mut next_stack = stack.clone();
            if let Atom::Tag(raw) = atom {
                if let Some(info) = parse_tag(raw) {
                    if info.closing {
                        if let Some(pos) = next_stack.iter().rposition(|(n, _)| *n == info.name) {
                            next_stack.truncate(pos);
                        }
                    } else {
                        next_stack.push((info.name, raw));
                    }
                }
            }
            let cost = atom.len();
            if used + cost + closing_len(&next_stack) > max_len && j > i {
                overflow = true;
                break;
            }
            // Record candidates before consuming a break character.
            if let Atom::Char(c) = atom {
                if c == '\n' {
                    if j > i && matches!(atoms[j - 1], Atom::Char('\n')) {
                        para = Some((j, stack.clone()));
                    }
                    line = Some((j, stack.clone()));
                } else if c == ' ' {
                    space = Some((j, stack.clone()));
                }
            }
            used += cost;
            stack = next_stack;
            j += 1;
        }

        if !overflow {
            let body: String = atoms[i..j].iter().map(atom_str).collect();
            chunks.push(format!("{prefix}{body}{}", closing_tags(&stack)));
            break;
        }

        // Choose the split point: a natural break in the second half of the
        // chunk if possible, else any natural break, else a hard cut at `j`.
        let half = i + (j - i) / 2;
        let pick = [&para, &line, &space]
            .into_iter()
            .flatten()
            .find(|(idx, _)| *idx > half)
            .or_else(|| {
                [&para, &line, &space]
                    .into_iter()
                    .flatten()
                    .find(|(idx, _)| *idx > i)
            })
            .cloned();
        let (split, split_stack) = pick.unwrap_or((j, stack.clone()));

        let body: String = atoms[i..split].iter().map(atom_str).collect();
        let body = body.trim_end_matches([' ', '\n']);
        chunks.push(format!("{prefix}{body}{}", closing_tags(&split_stack)));
        stack = split_stack;
        i = split;
        // Drop the separator we split on.
        if matches!(atoms.get(i), Some(Atom::Char(' '))) {
            i += 1;
        }
    }
    chunks.retain(|c| !html_to_plain(c).trim().is_empty());
    chunks
}

fn atom_str(atom: &Atom<'_>) -> String {
    match atom {
        Atom::Tag(s) | Atom::Entity(s) => (*s).to_string(),
        Atom::Char(c) => c.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_html_specials() {
        assert_eq!(
            markdown_to_html("a < b && c > d"),
            "a &lt; b &amp;&amp; c &gt; d"
        );
        assert_eq!(
            markdown_to_html("<script>alert(1)</script>"),
            "&lt;script&gt;alert(1)&lt;/script&gt;"
        );
    }

    #[test]
    fn inline_formatting() {
        assert_eq!(markdown_to_html("**bold**"), "<b>bold</b>");
        assert_eq!(markdown_to_html("*it* and _it_"), "<i>it</i> and <i>it</i>");
        assert_eq!(markdown_to_html("~~gone~~"), "<s>gone</s>");
        assert_eq!(markdown_to_html("use `a<b`"), "use <code>a&lt;b</code>");
        assert_eq!(
            markdown_to_html("**bold _and it_**"),
            "<b>bold <i>and it</i></b>"
        );
        assert_eq!(markdown_to_html("snake_case_name"), "snake_case_name");
        assert_eq!(markdown_to_html("2 * 3 * 4"), "2 * 3 * 4");
    }

    #[test]
    fn links_are_escaped() {
        assert_eq!(
            markdown_to_html("[Docs](https://example.com/?a=1&b=\"2\")"),
            "<a href=\"https://example.com/?a=1&amp;b=&quot;2&quot;\">Docs</a>"
        );
        assert_eq!(markdown_to_html("[not a link] (x)"), "[not a link] (x)");
    }

    #[test]
    fn code_blocks_with_language() {
        assert_eq!(
            markdown_to_html("```rust\nfn main() { a < b }\n```"),
            "<pre><code class=\"language-rust\">fn main() { a &lt; b }</code></pre>"
        );
        assert_eq!(
            markdown_to_html("```\n**not bold**\n```"),
            "<pre><code>**not bold**</code></pre>"
        );
        // Unterminated fence still produces balanced HTML.
        assert_eq!(markdown_to_html("```\nx"), "<pre><code>x</code></pre>");
    }

    #[test]
    fn headings_lists_tables() {
        assert_eq!(markdown_to_html("## Title"), "<b>Title</b>");
        assert_eq!(markdown_to_html("- one\n* two"), "• one\n• two");
        assert_eq!(
            markdown_to_html("| Name | Role |\n|---|---|\n| Alice Rivera | Admin |"),
            "<b>Name</b>: Alice Rivera\n<b>Role</b>: Admin"
        );
    }

    #[test]
    fn plain_fallback() {
        assert_eq!(
            html_to_plain("<b>Hi</b> &lt;x&gt; <a href=\"https://example.com\">site</a>"),
            "Hi <x> site (https://example.com)"
        );
    }

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(split_html("hello", 4096), vec!["hello".to_string()]);
    }

    #[test]
    fn split_respects_limit_and_lines() {
        let text: String = (0..500)
            .map(|n| format!("line number {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = split_html(&text, 4096);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(tg_len(c) <= 4096);
            assert!(c.starts_with("line number"));
        }
        let rejoined = chunks.join("\n");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn split_closes_and_reopens_tags() {
        let code = (0..400)
            .map(|n| format!("let x{n} = a &lt; b;"))
            .collect::<Vec<_>>()
            .join("\n");
        let html = format!("<pre><code class=\"language-rust\">{code}</code></pre>");
        let chunks = split_html(&html, 1000);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(tg_len(c) <= 1000, "chunk too long: {}", tg_len(c));
            assert!(c.starts_with("<pre><code class=\"language-rust\">"), "{c}");
            assert!(c.ends_with("</code></pre>"), "{c}");
            // No cut entities.
            assert!(!c.contains("&l<") && !c.ends_with('&'));
        }
    }

    #[test]
    fn split_without_breaks_hard_cuts_on_atoms() {
        let html = "&amp;".repeat(3000);
        let chunks = split_html(&html, 4096);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(tg_len(c) <= 4096);
            assert_eq!(c.len() % 5, 0);
        }
        assert_eq!(chunks.concat(), html);
    }

    #[test]
    fn split_counts_utf16() {
        let html = "😀".repeat(3000); // 6000 UTF-16 units
        let chunks = split_html(&html, 4096);
        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().all(|c| tg_len(c) <= 4096));
    }
}

/// Rendering and splitting as the daemon's Telegram bot uses them.
#[cfg(test)]
mod bot_format_tests {

    #[test]
    fn plain_multi_line_replies_keep_one_field_per_line() {
        let reply =
            "phase=SYNTHETIC\nblog_schedule=absent\nlatest_run=unknown\ncontroller=not_proven";
        assert_eq!(markdown_to_telegram_html(reply), reply);
        assert_eq!(markdown_to_telegram_plain_fallback(reply), reply);
        assert_eq!(markdown_to_telegram_html(reply).lines().count(), 4);
    }
    use super::markdown_to_html as markdown_to_telegram_html;
    use super::*;
    use crate::plain::{
        markdown_to_plain_fallback as markdown_to_telegram_plain_fallback,
        split_text as split_telegram_text,
    };
    use aidaemon_channels::split_message;

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
