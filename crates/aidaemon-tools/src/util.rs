//! Small helpers shared by the tools.

use serde::de::DeserializeOwned;
use serde_json::Value;

use aidaemon_core::ToolOutput;

/// Parse tool arguments into a typed struct, mapping serde errors to a
/// message the model can act on.
pub(crate) fn parse_args<T: DeserializeOwned>(tool: &str, arguments: Value) -> Result<T, String> {
    serde_json::from_value(arguments).map_err(|e| format!("invalid arguments for {tool}: {e}"))
}

/// Shorthand for an error the model should react to.
pub(crate) fn fail(message: impl Into<String>) -> ToolOutput {
    ToolOutput::error(message)
}

/// Heuristic used by `git` and `ripgrep`: a NUL byte in the first 8 KiB
/// means binary.
pub(crate) fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8192).any(|b| *b == 0)
}

/// Truncate to at most `max` characters on a char boundary, appending a
/// marker when cut.
pub(crate) fn truncate_chars(text: &str, max: usize) -> (String, bool) {
    match text.char_indices().nth(max) {
        Some((idx, _)) => (format!("{}…", &text[..idx]), true),
        None => (text.to_string(), false),
    }
}

/// 1-based line number of a byte offset.
pub(crate) fn line_of(text: &str, byte: usize) -> usize {
    text.as_bytes()[..byte]
        .iter()
        .filter(|b| **b == b'\n')
        .count()
        + 1
}

/// Bounded head + tail byte buffer for streaming output. Keeps the first
/// `head_cap` bytes and the last `tail_cap` bytes, counting everything.
#[derive(Debug)]
pub(crate) struct HeadTail {
    head: Vec<u8>,
    tail: std::collections::VecDeque<u8>,
    head_cap: usize,
    tail_cap: usize,
    total: usize,
}

impl HeadTail {
    pub(crate) fn new(max_bytes: usize) -> Self {
        let head_cap = max_bytes / 2;
        Self {
            head: Vec::new(),
            tail: std::collections::VecDeque::new(),
            head_cap,
            tail_cap: max_bytes - head_cap,
            total: 0,
        }
    }

    pub(crate) fn push(&mut self, mut data: &[u8]) {
        self.total += data.len();
        if self.head.len() < self.head_cap {
            let take = (self.head_cap - self.head.len()).min(data.len());
            self.head.extend_from_slice(&data[..take]);
            data = &data[take..];
        }
        if self.tail_cap == 0 {
            return;
        }
        if data.len() >= self.tail_cap {
            self.tail.clear();
            self.tail.extend(&data[data.len() - self.tail_cap..]);
        } else {
            let overflow = (self.tail.len() + data.len()).saturating_sub(self.tail_cap);
            self.tail.drain(..overflow);
            self.tail.extend(data);
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    pub(crate) fn truncated(&self) -> bool {
        self.total > self.head.len() + self.tail.len()
    }

    /// Render as text with an omission notice in the middle when truncated.
    pub(crate) fn render(&self) -> String {
        let head = String::from_utf8_lossy(&self.head);
        let tail: Vec<u8> = self.tail.iter().copied().collect();
        let tail = String::from_utf8_lossy(&tail);
        if self.truncated() {
            let omitted = self.total - self.head.len() - self.tail.len();
            format!("{head}\n[... {omitted} bytes of output omitted ...]\n{tail}")
        } else {
            format!("{head}{tail}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_tail_keeps_both_ends() {
        let mut buf = HeadTail::new(10);
        buf.push(b"0123");
        buf.push(b"456789abcdefghij");
        assert_eq!(buf.total(), 20);
        assert!(buf.truncated());
        let out = buf.render();
        assert!(out.starts_with("01234\n"), "{out}");
        assert!(out.ends_with("\nfghij"), "{out}");
        assert!(out.contains("10 bytes of output omitted"), "{out}");

        let mut small = HeadTail::new(10);
        small.push(b"abc");
        small.push(b"def");
        assert!(!small.truncated());
        assert_eq!(small.render(), "abcdef");
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        assert_eq!(truncate_chars("héllo", 2), ("hé…".to_string(), true));
        assert_eq!(truncate_chars("hi", 5), ("hi".to_string(), false));
    }
}
