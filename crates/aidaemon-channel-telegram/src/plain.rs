//! Plain-text fallbacks: Markdown read as plain text when Telegram rejects
//! the HTML, and plain text split to Telegram's length.

use aidaemon_channels::markdown::Inline;

use crate::format::tg_len;

/// Convert common LLM markdown to readable plain text for Telegram fallbacks.
pub fn markdown_to_plain_fallback(md: &str) -> String {
    let md = aidaemon_channels::latex::strip_latex(md);
    let mut result = String::with_capacity(md.len());
    let mut in_code_block = false;

    for line in md.lines() {
        // The same line rules as the HTML converter: an indented fence, a
        // heading of at most six `#`, and a bullet keeping its indent.
        if line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }

        let processed = if in_code_block {
            line.to_string()
        } else {
            let line = crate::format::strip_heading(line).unwrap_or(line);
            let trimmed = line.trim_start();
            let indent = &line[..line.len() - trimmed.len()];
            match trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
            {
                Some(rest) => format!("{indent}• {}", convert_inline_formatting_to_plain(rest)),
                None => convert_inline_formatting_to_plain(line),
            }
        };

        result.push_str(&processed);
        result.push('\n');
    }

    if result.ends_with('\n') {
        result.pop();
    }
    result
}

fn convert_inline_formatting_to_plain(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    render(&aidaemon_channels::markdown::parse(s), &mut out);
    out
}

/// Plain text for the shared inline tree: marks dropped, a link as
/// `text (url)`.
fn render(inlines: &[Inline], out: &mut String) {
    for inline in inlines {
        match inline {
            Inline::Text(text) | Inline::Code(text) => out.push_str(text),
            Inline::Bold(inner) | Inline::Italic(inner) | Inline::Strike(inner) => {
                render(inner, out)
            }
            Inline::Link { text, url } => {
                render(text, out);
                out.push_str(" (");
                out.push_str(url);
                out.push(')');
            }
            Inline::Angle(inner) => {
                out.push('<');
                out.push_str(inner);
                out.push('>');
            }
        }
    }
}

/// Split plain text for Telegram into parts of at most `max_units` UTF-16
/// code units, the unit the Bot API counts. The generic splitter picks the
/// cut points by characters, and a character outside the Basic Multilingual
/// Plane (most emoji) is two units, so a part still too long is cut again by
/// units, at the last whitespace that fits when there is one.
pub fn split_text(text: &str, max_units: usize) -> Vec<String> {
    let max_units = max_units.max(2);
    let mut parts = Vec::new();
    for part in aidaemon_channels::split_message(text, max_units) {
        if tg_len(&part) <= max_units {
            parts.push(part);
            continue;
        }
        let mut rest = part.as_str();
        while tg_len(rest) > max_units {
            let mut units = 0;
            let mut end = 0;
            let mut last_space = None;
            for (index, c) in rest.char_indices() {
                if units + c.len_utf16() > max_units {
                    break;
                }
                units += c.len_utf16();
                end = index + c.len_utf8();
                if c.is_whitespace() {
                    last_space = Some(end);
                }
            }
            let cut = last_space.filter(|&cut| cut > 0).unwrap_or(end);
            let piece = rest[..cut].trim_end();
            if !piece.is_empty() {
                parts.push(piece.to_string());
            }
            rest = rest[cut..].trim_start();
        }
        if !rest.is_empty() {
            parts.push(rest.to_string());
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::markdown_to_plain_fallback;

    #[test]
    fn line_rules_match_the_html_converter() {
        assert_eq!(
            markdown_to_plain_fallback("  ```\ncode\n  ```\n####### seven\n  - item"),
            "code\n####### seven\n  • item"
        );
    }

    #[test]
    fn snake_case_keeps_its_underscores() {
        assert_eq!(
            markdown_to_plain_fallback("set my_var_name and _this_"),
            "set my_var_name and this"
        );
    }
}
