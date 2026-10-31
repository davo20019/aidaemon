//! Outbound text handling: Discord Markdown is close enough to standard
//! Markdown that nothing is converted; long messages are split at line
//! boundaries and open code fences are closed and reopened across chunks.

const FENCE_CLOSE_RESERVE: usize = 4; // "\n```"

/// Split `text` into chunks of at most `max_chars` characters (Unicode
/// scalar values, which is how Discord counts). Prefers line boundaries; a
/// chunk that ends inside a fenced code block is closed with ```` ``` ```` and
/// the next chunk reopens the fence with the same opener (language tag kept).
pub fn split_message(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }
    let budget = max_chars.saturating_sub(FENCE_CLOSE_RESERVE).max(1);

    let mut chunks = Vec::new();
    let mut fence: Option<String> = None;
    let mut cur = String::new();
    let mut cur_len = 0usize;
    // Length of the reopened fence prefix at the start of `cur`.
    let mut fresh_len = 0usize;

    let flush = |cur: &mut String,
                 cur_len: &mut usize,
                 fresh_len: &mut usize,
                 fence: &Option<String>,
                 chunks: &mut Vec<String>| {
        let mut chunk = std::mem::take(cur);
        if let Some(_opener) = fence {
            if !chunk.ends_with('\n') {
                chunk.push('\n');
            }
            chunk.push_str("```");
        }
        let trimmed = chunk.trim_end();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }
        if let Some(opener) = fence {
            cur.push_str(opener);
            cur.push('\n');
        }
        *cur_len = cur.chars().count();
        *fresh_len = *cur_len;
    };

    for line in text.split_inclusive('\n') {
        let is_fence_line = line.trim_start().starts_with("```");
        // Drop blank lines at the start of a chunk (outside code blocks).
        if cur_len == fresh_len && fence.is_none() && line.trim().is_empty() {
            continue;
        }
        let mut rest = line;
        while !rest.is_empty() {
            let rest_len = rest.chars().count();
            if cur_len + rest_len <= budget {
                cur.push_str(rest);
                cur_len += rest_len;
                break;
            }
            if cur_len > fresh_len {
                flush(&mut cur, &mut cur_len, &mut fresh_len, &fence, &mut chunks);
                continue;
            }
            // A single line longer than a chunk: hard split it.
            let take = budget.saturating_sub(cur_len).max(1);
            let split_at = rest.char_indices().nth(take).map_or(rest.len(), |(i, _)| i);
            cur.push_str(&rest[..split_at]);
            cur_len += take;
            rest = &rest[split_at..];
            flush(&mut cur, &mut cur_len, &mut fresh_len, &fence, &mut chunks);
        }
        if is_fence_line {
            fence = match fence {
                None => Some(line.trim().to_string()),
                Some(_) => None,
            };
        }
    }
    if cur_len > fresh_len {
        let trimmed = cur.trim_end();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }
    }
    chunks
}

/// Truncate to `max_chars` characters, ending with an ellipsis if cut.
pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(split_message("hello", 2000), vec!["hello"]);
    }

    #[test]
    fn splits_at_lines_within_limit() {
        let text = (0..300)
            .map(|i| format!("line number {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = split_message(&text, 2000);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.chars().count() <= 2000);
        }
        assert_eq!(chunks.join("\n"), text);
    }

    #[test]
    fn fences_are_closed_and_reopened() {
        let mut text = String::from("Intro\n```rust\n");
        for i in 0..200 {
            text.push_str(&format!("let value_{i} = {i};\n"));
        }
        text.push_str("```\nOutro");
        let chunks = split_message(&text, 2000);
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.chars().count() <= 2000, "chunk too long");
            assert_eq!(c.matches("```").count() % 2, 0, "unbalanced: {c}");
        }
        assert!(chunks[0].ends_with("```"));
        assert!(chunks[1].starts_with("```rust\n"));
        assert!(chunks.last().unwrap().ends_with("Outro"));
    }

    #[test]
    fn hard_splits_long_lines_on_char_boundaries() {
        let text = "é".repeat(4500);
        let chunks = split_message(&text, 2000);
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.chars().count() <= 2000));
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn truncates_with_ellipsis() {
        assert_eq!(truncate_chars("abcdef", 4), "abc…");
        assert_eq!(truncate_chars("abc", 4), "abc");
    }
}
