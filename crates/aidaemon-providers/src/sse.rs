//! Incremental Server-Sent Events parser.
//!
//! Fed raw body chunks in whatever sizes the network delivers; yields one
//! [`SseEvent`] per blank-line-terminated event. Lines are split on `\n`
//! (with an optional trailing `\r`), which is safe for UTF-8 because `\n`
//! never appears inside a multi-byte sequence.

/// One dispatched SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SseEvent {
    /// The `event:` field, if the server sent one.
    pub event: Option<String>,
    /// All `data:` lines joined with `\n`.
    pub data: String,
}

/// The most one line, and one event's joined data, may hold. A response
/// event carrying an image is a few MiB; past this the server (or anything
/// between it and us on a plain-http endpoint) is streaming garbage, and
/// buffering it would grow the daemon's memory without bound.
const MAX_EVENT_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Default)]
pub(crate) struct SseParser {
    buf: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
    data_bytes: usize,
    /// Inside a line that passed the cap: dropped up to its newline.
    skipping_line: bool,
    /// The current event passed the cap: dropped up to its blank line.
    skipping_event: bool,
}

impl SseParser {
    /// Feed one chunk; returns every event completed by it.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let raw: Vec<u8> = self.buf.drain(..=pos).collect();
            if std::mem::take(&mut self.skipping_line) {
                continue;
            }
            let line = String::from_utf8_lossy(&raw);
            let line = line.trim_end_matches('\n').trim_end_matches('\r');
            self.line(line, &mut out);
        }
        if self.buf.len() > MAX_EVENT_BYTES {
            self.buf.clear();
            self.skipping_line = true;
            self.drop_event();
        }
        out
    }

    fn drop_event(&mut self) {
        self.event = None;
        self.data.clear();
        self.data_bytes = 0;
        self.skipping_event = true;
    }

    /// The body ended: flush a trailing unterminated line and event.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut out = Vec::new();
        if !self.buf.is_empty() && !self.skipping_line {
            let raw = std::mem::take(&mut self.buf);
            let line = String::from_utf8_lossy(&raw);
            let line = line.trim_end_matches('\r').to_string();
            self.line(&line, &mut out);
        }
        self.dispatch(&mut out);
        out
    }

    fn line(&mut self, line: &str, out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            if std::mem::take(&mut self.skipping_event) {
                return;
            }
            self.dispatch(out);
            return;
        }
        if self.skipping_event {
            return;
        }
        if line.starts_with(':') {
            return; // comment / keep-alive
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => {
                self.data_bytes += value.len() + 1;
                if self.data_bytes > MAX_EVENT_BYTES {
                    self.drop_event();
                } else {
                    self.data.push(value.to_string());
                }
            }
            _ => {}
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        let event = self.event.take();
        self.data_bytes = 0;
        if self.data.is_empty() {
            return;
        }
        out.push(SseEvent {
            event,
            data: std::mem::take(&mut self.data).join("\n"),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_oversized_line_is_dropped_and_parsing_resumes() {
        let mut parser = SseParser::default();
        let chunk = vec![b'x'; 1024 * 1024];
        assert!(parser.push(b"data: ").is_empty());
        for _ in 0..40 {
            assert!(parser.push(&chunk).is_empty());
        }
        assert!(parser.buf.len() <= MAX_EVENT_BYTES);
        let out = parser.push(b"\n\ndata: ok\n\n");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data, "ok");
    }

    #[test]
    fn an_event_over_the_cap_is_dropped_whole() {
        let mut parser = SseParser::default();
        let line = format!("data: {}\n", "y".repeat(1024 * 1024));
        for _ in 0..40 {
            assert!(parser.push(line.as_bytes()).is_empty());
        }
        assert!(parser.push(b"\n").is_empty());
        let out = parser.push(b"data: next\n\n");
        assert_eq!(out[0].data, "next");
    }

    #[test]
    fn splits_events_across_arbitrary_chunks() {
        let body =
            b"event: a\r\ndata: {\"x\":1}\r\n\r\n: ping\n\ndata: one\ndata: two\n\ndata: [DONE]";
        let mut parser = SseParser::default();
        let mut events = Vec::new();
        for chunk in body.chunks(3) {
            events.extend(parser.push(chunk));
        }
        events.extend(parser.finish());
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: Some("a".into()),
                    data: "{\"x\":1}".into()
                },
                SseEvent {
                    event: None,
                    data: "one\ntwo".into()
                },
                SseEvent {
                    event: None,
                    data: "[DONE]".into()
                },
            ]
        );
    }

    #[test]
    fn multibyte_text_split_mid_character() {
        let body = "data: héllo ✓\n\n".as_bytes();
        let mut parser = SseParser::default();
        let mut events = Vec::new();
        for chunk in body.chunks(1) {
            events.extend(parser.push(chunk));
        }
        assert_eq!(events[0].data, "héllo ✓");
    }
}
