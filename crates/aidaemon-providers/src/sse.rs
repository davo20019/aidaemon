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

#[derive(Debug, Default)]
pub(crate) struct SseParser {
    buf: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

impl SseParser {
    /// Feed one chunk; returns every event completed by it.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let raw: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&raw);
            let line = line.trim_end_matches('\n').trim_end_matches('\r');
            self.line(line, &mut out);
        }
        out
    }

    /// The body ended: flush a trailing unterminated line and event.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut out = Vec::new();
        if !self.buf.is_empty() {
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
            self.dispatch(out);
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
            "data" => self.data.push(value.to_string()),
            _ => {}
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        let event = self.event.take();
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
