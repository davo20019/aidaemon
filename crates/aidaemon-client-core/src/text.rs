//! Text from the daemon as a reader should see it.

/// `text` without terminal escape sequences (colours, cursor moves, titles)
/// and carriage-return overwrites: what a reader of the plain text sees.
pub fn plain_output(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\u{1b}' => match chars.next() {
                // CSI: parameters, then one final byte in @..~.
                Some('[') => {
                    for c in chars.by_ref() {
                        if ('@'..='~').contains(&c) {
                            break;
                        }
                    }
                }
                // OSC: to BEL or ESC \.
                Some(']') => {
                    while let Some(c) = chars.next() {
                        if c == '\u{7}' {
                            break;
                        }
                        if c == '\u{1b}' && chars.peek() == Some(&'\\') {
                            chars.next();
                            break;
                        }
                    }
                }
                _ => {}
            },
            // A carriage return alone redraws its line (a progress bar):
            // keep what came after it.
            '\r' if chars.peek() != Some(&'\n') => {
                if let Some(start) = out.rfind('\n') {
                    out.truncate(start + 1);
                } else {
                    out.clear();
                }
            }
            '\r' => {}
            c if c.is_control() && c != '\n' && c != '\t' => {}
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::plain_output;

    #[test]
    fn escapes_and_redrawn_lines_leave_the_plain_text() {
        assert_eq!(
            plain_output(
                "\u{1b}[31mFAILED\u{1b}[0m\nbuilding 10%\rbuilding 99%\n\u{1b}]0;title\u{7}done"
            ),
            "FAILED\nbuilding 99%\ndone"
        );
    }
}

/// A reply's Markdown as it is read aloud: the words a reader sees, without
/// the marks that draw them. A fenced code block is not read, only said to
/// be there; a link reads as its text; emphasis, heading and quote marks,
/// list bullets and table rules go. Paragraphs stay apart, so a voice pauses
/// between them. (Markdown syntax is a machine format the daemon's replies
/// use; this reads it, never prose.)
pub fn spoken_text(markdown: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut in_fence = false;
    for line in markdown.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            if !in_fence {
                out.push("(code)".to_string());
            }
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        // A table's rule row (|---|:--:|) says nothing.
        if !trimmed.is_empty()
            && trimmed
                .chars()
                .all(|c| matches!(c, '|' | '-' | ':' | ' ' | '='))
            && trimmed.contains('-')
        {
            continue;
        }
        let mut text = trimmed;
        // Block marks at the line's start: headings, quotes, bullets.
        loop {
            let before = text;
            text = text.trim_start_matches('#').trim_start();
            text = text.trim_start_matches('>').trim_start();
            for bullet in ["- [ ] ", "- [x] ", "- ", "* ", "+ "] {
                if let Some(rest) = text.strip_prefix(bullet) {
                    text = rest;
                }
            }
            if text == before {
                break;
            }
        }
        // Table cells read as a list; runs of spaces as one.
        let text = inline_spoken(text)
            .split('|')
            .map(|cell| cell.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|cell| !cell.is_empty())
            .collect::<Vec<_>>()
            .join(", ");
        let text = text.trim();
        if text.is_empty() {
            if out.last().is_some_and(|last| !last.is_empty()) {
                out.push(String::new());
            }
        } else {
            out.push(text.to_string());
        }
    }
    let mut spoken = String::new();
    for line in out {
        if line.is_empty() {
            spoken.push_str("\n\n");
        } else {
            if !spoken.is_empty() && !spoken.ends_with('\n') {
                spoken.push('\n');
            }
            spoken.push_str(&line);
        }
    }
    spoken.trim().to_string()
}

/// One line's inline Markdown as it reads: `[text](url)` and `![alt](url)`
/// as their text, code spans as their contents, emphasis marks dropped.
fn inline_spoken(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let image = c == '!' && chars.get(i + 1) == Some(&'[');
        if c == '[' || image {
            let open = if image { i + 1 } else { i };
            if let Some(close) = (open + 1..chars.len()).find(|&j| chars[j] == ']')
                && chars.get(close + 1) == Some(&'(')
                && let Some(end) = (close + 2..chars.len()).find(|&j| chars[j] == ')')
            {
                out.extend(&chars[open + 1..close]);
                i = end + 1;
                continue;
            }
        }
        match c {
            '*' | '_' | '`' | '~' => {}
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod spoken_tests {
    use super::spoken_text;

    #[test]
    fn markdown_reads_as_its_words() {
        let reply = "## Result\n\nThe capital is **Quito**, see [the map](https://example.com/map).\n\n\
                     - first `item`\n- second\n\n```rust\nfn main() {}\n```\n\n\
                     | City | Altitude |\n|---|---:|\n| Quito | 2,850 m |\n\n> A _quoted_ line";
        assert_eq!(
            spoken_text(reply),
            "Result\n\nThe capital is Quito, see the map.\n\nfirst item\nsecond\n\n(code)\n\n\
             City, Altitude\nQuito, 2,850 m\n\nA quoted line"
        );
    }

    #[test]
    fn plain_text_is_read_as_it_is() {
        assert_eq!(spoken_text("Just a sentence."), "Just a sentence.");
        assert_eq!(spoken_text(""), "");
    }
}
