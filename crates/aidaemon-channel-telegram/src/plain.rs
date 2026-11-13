//! Plain-text fallbacks: Markdown read as plain text when Telegram rejects
//! the HTML, and plain text split to Telegram's length.

use crate::format::tg_len;

/// Convert common LLM markdown to readable plain text for Telegram fallbacks.
pub fn markdown_to_plain_fallback(md: &str) -> String {
    let md = aidaemon_channels::latex::strip_latex(md);
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
        if chars[i] == '`'
            && let Some(end) = find_char(&chars, '`', i + 1) {
                result.extend(chars[i + 1..end].iter());
                i = end + 1;
                continue;
            }

        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*'
            && let Some(end) = find_double_char(&chars, '*', i + 2) {
                result.extend(chars[i + 2..end].iter());
                i = end + 2;
                continue;
            }

        if chars[i] == '['
            && let Some((text, url, end)) = parse_link(&chars, i) {
                result.push_str(&text);
                if !url.is_empty() {
                    result.push_str(" (");
                    result.push_str(&url);
                    result.push(')');
                }
                i = end;
                continue;
            }

        if chars[i] == '_'
            && let Some(end) = find_char(&chars, '_', i + 1) {
                result.extend(chars[i + 1..end].iter());
                i = end + 1;
                continue;
            }

        if chars[i] == '*' && (i + 1 >= len || chars[i + 1] != '*')
            && let Some(end) = find_single_star(&chars, i + 1) {
                result.extend(chars[i + 1..end].iter());
                i = end + 1;
                continue;
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
