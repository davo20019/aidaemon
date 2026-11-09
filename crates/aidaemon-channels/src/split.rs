//! Splitting long Markdown replies into messages that fit a channel.
//!
//! Lengths are counted in characters (Unicode scalar values), matching
//! [`crate::ChannelCapabilities::max_message_len`].

/// The smallest limit the splitter honours. Smaller limits are raised to
/// this so a reopened code fence always leaves room for content.
pub const MIN_SPLIT_LEN: usize = 32;

const FENCE_CLOSE_COST: usize = 4; // "\n```"

/// Split `text` into parts of at most `max_chars` characters.
///
/// Cut points are chosen in this order: a paragraph boundary (blank line)
/// outside a fenced code block, a line boundary outside a fence, a line
/// boundary inside a fence, then a hard cut. When a part has to end inside a
/// fenced code block, the fence is closed at the end of that part and
/// reopened (with the same info string) at the start of the next, so every
/// part renders on its own.
pub fn split_message(text: &str, max_chars: usize) -> Vec<String> {
    let max = max_chars.max(MIN_SPLIT_LEN);
    let text = text.trim_end();
    if text.chars().count() <= max {
        return if text.is_empty() {
            Vec::new()
        } else {
            vec![text.to_string()]
        };
    }

    let mut parts = Vec::new();
    let mut rest = text.to_string();
    let mut reopen: Option<String> = None;
    loop {
        let src = match reopen.take() {
            Some(fence) => format!("{fence}\n{rest}"),
            None => rest,
        };
        if src.chars().count() <= max {
            let trimmed = src.trim_end();
            if !trimmed.is_empty() {
                parts.push(trimmed.to_string());
            }
            break;
        }
        let cut = choose_cut(&src, max);
        let mut part = src[..cut.at].trim_end().to_string();
        if let Some(fence) = &cut.open_fence {
            part.push('\n');
            part.push_str(fence_marker(fence));
            // A very long info string would eat the budget of every part.
            reopen = Some(if fence.chars().count() > max / 2 {
                fence_marker(fence).to_string()
            } else {
                fence.clone()
            });
        }
        if !part.trim().is_empty() {
            parts.push(part);
        }
        rest = src[cut.at..].trim_start_matches('\n').to_string();
        if rest.trim().is_empty() {
            break;
        }
    }
    parts
}

struct Cut {
    /// Byte offset to cut at.
    at: usize,
    /// The opening fence line still open at the cut, if any.
    open_fence: Option<String>,
}

fn fence_marker(fence_line: &str) -> &'static str {
    if fence_line.trim_start().starts_with("~~~") {
        "~~~"
    } else {
        "```"
    }
}

fn is_fence(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("```") || t.starts_with("~~~")
}

fn choose_cut(src: &str, max: usize) -> Cut {
    let mut open: Option<String> = None;
    let mut chars_so_far = 0usize;
    let mut byte_pos = 0usize;
    let mut best_para: Option<usize> = None;
    let mut best_line: Option<usize> = None;
    let mut best_fenced: Option<(usize, String)> = None;

    let lines: Vec<&str> = src.split_inclusive('\n').collect();
    for (i, line) in lines.iter().enumerate() {
        let line_chars = line.chars().count();
        if chars_so_far + line_chars > max {
            break;
        }
        chars_so_far += line_chars;
        byte_pos += line.len();
        let body = line.trim_end_matches('\n');
        if is_fence(body) {
            match &open {
                Some(opening) if body.trim_start().starts_with(fence_marker(opening)) => {
                    open = None;
                }
                Some(_) => {}
                None => open = Some(body.to_string()),
            }
        }
        match &open {
            None => {
                best_line = Some(byte_pos);
                let next_blank = lines.get(i + 1).is_some_and(|next| next.trim().is_empty());
                if body.trim().is_empty() || next_blank {
                    best_para = Some(byte_pos);
                }
            }
            Some(opening) => {
                if chars_so_far + FENCE_CLOSE_COST <= max && !is_fence(body) {
                    best_fenced = Some((byte_pos, opening.clone()));
                }
            }
        }
    }

    if let Some(at) = best_para.or(best_line) {
        return Cut {
            at,
            open_fence: None,
        };
    }
    if let Some((at, fence)) = best_fenced {
        return Cut {
            at,
            open_fence: Some(fence),
        };
    }
    hard_cut(src, max)
}

/// No usable line boundary: cut at a character boundary.
fn hard_cut(src: &str, max: usize) -> Cut {
    let byte_at = |n: usize| src.char_indices().nth(n).map_or(src.len(), |(i, _)| i);
    let at = byte_at(max);
    let open = fence_open_before(src, at);
    if open.is_some() {
        let at = byte_at(max.saturating_sub(FENCE_CLOSE_COST).max(1));
        return Cut {
            at,
            open_fence: fence_open_before(src, at),
        };
    }
    Cut {
        at,
        open_fence: None,
    }
}

/// The fence left open by the complete lines before `pos`.
fn fence_open_before(src: &str, pos: usize) -> Option<String> {
    let mut open: Option<String> = None;
    let mut consumed = 0;
    for line in src.split_inclusive('\n') {
        consumed += line.len();
        if consumed > pos {
            break;
        }
        let body = line.trim_end_matches('\n');
        if is_fence(body) {
            match &open {
                Some(opening) if body.trim_start().starts_with(fence_marker(opening)) => {
                    open = None
                }
                Some(_) => {}
                None => open = Some(body.to_string()),
            }
        }
    }
    open
}

/// Marks text that [`crate::Channel::edit`] cut to fit.
pub const TRUNCATION_MARK: char = '\u{2026}';

/// `text` as one message of at most `max_chars`: over the limit it is cut at
/// the end and finished with [`TRUNCATION_MARK`]. This is the rule every
/// [`crate::Channel::edit`] follows; an adapter that counts differently
/// (Telegram's UTF-16 units over HTML) applies the same rule to its own
/// measure.
pub fn truncate_for_edit(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    out.push(TRUNCATION_MARK);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_edit_is_cut_at_the_end_and_marked() {
        assert_eq!(truncate_for_edit("short", 5), "short");
        assert_eq!(truncate_for_edit("toolong", 5), "tool\u{2026}");
        assert_eq!(truncate_for_edit("😀😀😀", 2), "😀\u{2026}");
    }

    fn len(s: &str) -> usize {
        s.chars().count()
    }

    #[test]
    fn short_text_is_one_part() {
        assert_eq!(split_message("hello", 100), vec!["hello"]);
        assert!(split_message("   ", 100).is_empty());
    }

    #[test]
    fn prefers_paragraph_boundaries() {
        let a = "a".repeat(30);
        let b = "b".repeat(30);
        let c = "c".repeat(30);
        let text = format!("{a}\n{b}\n\n{c}");
        let parts = split_message(&text, 70);
        assert_eq!(parts, vec![format!("{a}\n{b}"), c]);
    }

    #[test]
    fn falls_back_to_lines_then_hard_cuts() {
        let line = "x".repeat(40);
        let text = format!("{line}\n{line}\n{line}");
        let parts = split_message(&text, 50);
        assert_eq!(parts.len(), 3);
        assert!(parts.iter().all(|p| len(p) <= 50));

        let long = "y".repeat(120);
        let parts = split_message(&long, 50);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts.concat(), long);
    }

    #[test]
    fn multibyte_characters_are_counted_as_chars() {
        let text = "é".repeat(100);
        let parts = split_message(&text, 40);
        assert!(parts.iter().all(|p| len(p) <= 40));
        assert_eq!(parts.concat(), text);
    }

    #[test]
    fn does_not_split_inside_a_code_fence_when_avoidable() {
        let intro = "Here is the code:".to_string();
        let code = format!("```rust\n{}\n```", "let x = 1;\n".repeat(4).trim_end());
        let outro = "That is all.".to_string();
        let text = format!("{intro}\n\n{code}\n\n{outro}");
        let parts = split_message(&text, len(&code) + 5);
        assert!(parts.contains(&code), "{parts:#?}");
        for p in &parts {
            assert_eq!(p.matches("```").count() % 2, 0, "unbalanced fence: {p}");
        }
    }

    #[test]
    fn oversized_code_block_is_closed_and_reopened() {
        let body: String = (0..30).map(|i| format!("line {i:02}\n")).collect();
        let text = format!("```python\n{body}```");
        let parts = split_message(&text, 60);
        assert!(parts.len() > 1);
        for p in &parts {
            assert!(len(p) <= 60, "{p}");
            assert!(p.starts_with("```python"), "{p}");
            assert!(p.ends_with("```"), "{p}");
        }
        let joined: String = parts.join("\n");
        for i in 0..30 {
            assert!(joined.contains(&format!("line {i:02}")));
        }
    }
}
