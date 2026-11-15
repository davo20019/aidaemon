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
