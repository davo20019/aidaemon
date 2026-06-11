//! SSE stream accumulation for OpenAI-compatible chat completions.
//!
//! Streaming transport, non-streaming contract: deltas are accumulated into
//! a JSON body identical in shape to a non-streaming `chat/completions`
//! response, so the existing parse path (finish_reason handling, usage,
//! reasoning, tool calls) applies unchanged.
//!
//! Self-recovery: when a stream dies mid-response but text was already
//! received, finalization stamps `finish_reason: "length"` — the agent
//! loop's truncation-recovery machinery (continuation with overlap
//! detection) then treats it exactly like a token-limit cutoff instead of
//! losing the whole response.

use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Incremental SSE event framer. Feed raw bytes as they arrive; it yields
/// complete `data:` payloads, holding partial events across chunk
/// boundaries. Comment lines (`: ping`) and other fields are ignored.
#[derive(Debug, Default)]
pub(crate) struct SseFramer {
    buffer: String,
}

impl SseFramer {
    /// Feed a chunk of bytes; returns the complete data payloads it closed.
    /// The `[DONE]` sentinel is returned as a payload like any other.
    pub(crate) fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));
        let mut payloads = Vec::new();
        // Events are separated by a blank line. Process every complete
        // event; whatever follows the last separator stays buffered.
        while let Some(boundary) = self.buffer.find("\n\n") {
            let event: String = self.buffer.drain(..boundary + 2).collect();
            for line in event.lines() {
                if let Some(data) = line.strip_prefix("data:") {
                    payloads.push(data.trim().to_string());
                }
                // Comment lines (": ping") and other SSE fields are ignored.
            }
        }
        payloads
    }
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Accumulates streaming deltas into a non-streaming-shaped response body.
#[derive(Debug, Default)]
pub(crate) struct StreamAccumulator {
    content: String,
    reasoning: String,
    tool_calls: BTreeMap<u64, PartialToolCall>,
    finish_reason: Option<String>,
    usage: Option<Value>,
    saw_done: bool,
}

impl StreamAccumulator {
    /// Apply one SSE data payload. Returns false for the `[DONE]` sentinel
    /// (stream complete), true otherwise. Unparseable payloads are skipped.
    pub(crate) fn apply_payload(&mut self, payload: &str) -> bool {
        if payload == "[DONE]" {
            self.saw_done = true;
            return false;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
            return true;
        };
        if let Some(usage) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(usage.clone());
        }
        let Some(choice) = chunk["choices"].get(0) else {
            return true;
        };
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finish_reason = Some(reason.to_string());
        }
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str() {
            self.content.push_str(text);
        }
        // Reasoning deltas: "reasoning_content" (DeepSeek/Kimi style) or
        // "reasoning" (OpenRouter style).
        for key in ["reasoning_content", "reasoning"] {
            if let Some(text) = delta[key].as_str() {
                self.reasoning.push_str(text);
            }
        }
        if let Some(tool_deltas) = delta["tool_calls"].as_array() {
            for td in tool_deltas {
                let index = td["index"].as_u64().unwrap_or(0);
                let entry = self.tool_calls.entry(index).or_default();
                if let Some(id) = td["id"].as_str() {
                    entry.id.push_str(id);
                }
                if let Some(name) = td["function"]["name"].as_str() {
                    entry.name.push_str(name);
                }
                if let Some(args) = td["function"]["arguments"].as_str() {
                    entry.arguments.push_str(args);
                }
            }
        }
        true
    }

    /// Whether any visible output (text or tool-call fragments) arrived.
    pub(crate) fn has_partial_output(&self) -> bool {
        !self.content.is_empty() || !self.tool_calls.is_empty()
    }

    /// Build a response body in the non-streaming `chat/completions` shape.
    ///
    /// `interrupted` marks a stream that died before `[DONE]`/finish_reason:
    /// with partial text present the finish_reason becomes "length" so the
    /// caller's truncation recovery takes over.
    pub(crate) fn finalize(self, interrupted: bool) -> Value {
        let finish_reason = match (&self.finish_reason, interrupted) {
            (Some(reason), _) => Some(reason.clone()),
            // Died mid-stream with partial text: report a token-limit-style
            // cutoff so truncation recovery continues the response.
            (None, true) if !self.content.is_empty() => Some("length".to_string()),
            (None, _) => None,
        };

        let mut message = json!({ "role": "assistant" });
        message["content"] = if self.content.is_empty() {
            Value::Null
        } else {
            json!(self.content)
        };
        if !self.reasoning.is_empty() {
            // Emitted as "reasoning" — the key the non-streaming parse path
            // reads for thinking extraction.
            message["reasoning"] = json!(self.reasoning);
        }
        if !self.tool_calls.is_empty() {
            let calls: Vec<Value> = self
                .tool_calls
                .into_values()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": { "name": tc.name, "arguments": tc.arguments },
                    })
                })
                .collect();
            message["tool_calls"] = json!(calls);
        }

        let mut body = json!({
            "choices": [{
                "message": message,
                "finish_reason": finish_reason,
            }],
        });
        if let Some(usage) = self.usage {
            body["usage"] = usage;
        }
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(payload: &str) -> String {
        format!("data: {payload}\n\n")
    }

    #[test]
    fn framer_yields_payloads_and_holds_partials_across_chunks() {
        let mut framer = SseFramer::default();
        // One complete event plus the head of a second one.
        let first = framer.feed(b"data: {\"a\":1}\n\ndata: {\"b\"");
        assert_eq!(first, vec!["{\"a\":1}".to_string()]);
        // Rest of the second event arrives.
        let second = framer.feed(b":2}\n\n");
        assert_eq!(second, vec!["{\"b\":2}".to_string()]);
    }

    #[test]
    fn framer_ignores_comments_and_handles_done() {
        let mut framer = SseFramer::default();
        let got = framer.feed(b": keep-alive ping\n\ndata: [DONE]\n\n");
        assert_eq!(got, vec!["[DONE]".to_string()]);
    }

    #[test]
    fn accumulates_text_content_across_deltas() {
        let mut acc = StreamAccumulator::default();
        assert!(acc.apply_payload(
            r#"{"choices":[{"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}"#
        ));
        assert!(
            acc.apply_payload(r#"{"choices":[{"delta":{"content":"lo!"},"finish_reason":null}]}"#)
        );
        assert!(acc.apply_payload(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#));
        assert!(!acc.apply_payload("[DONE]"));

        let body = acc.finalize(false);
        assert_eq!(body["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn merges_tool_call_deltas_by_index() {
        let mut acc = StreamAccumulator::default();
        acc.apply_payload(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}"#,
        );
        acc.apply_payload(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]},"finish_reason":null}]}"#,
        );
        acc.apply_payload(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}},{"index":1,"id":"call_2","function":{"name":"web_search","arguments":"{}"}}]},"finish_reason":null}]}"#,
        );
        acc.apply_payload(r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#);

        let body = acc.finalize(false);
        let calls = body["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["function"]["name"], "read_file");
        assert_eq!(calls[0]["function"]["arguments"], "{\"path\":\"a.txt\"}");
        assert_eq!(calls[1]["function"]["name"], "web_search");
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn captures_usage_and_reasoning() {
        let mut acc = StreamAccumulator::default();
        acc.apply_payload(
            r#"{"choices":[{"delta":{"reasoning_content":"thinking hard"},"finish_reason":null}]}"#,
        );
        acc.apply_payload(
            r#"{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":7}}"#,
        );
        let body = acc.finalize(false);
        assert_eq!(body["usage"]["prompt_tokens"], 12);
        assert_eq!(
            body["choices"][0]["message"]["reasoning"], "thinking hard",
            "reasoning must land on the key the non-streaming parse reads"
        );
        assert_eq!(body["choices"][0]["message"]["content"], "answer");
    }

    #[test]
    fn interrupted_stream_with_partial_text_finalizes_as_length_cutoff() {
        let mut acc = StreamAccumulator::default();
        acc.apply_payload(
            r#"{"choices":[{"delta":{"content":"partial answer that was cut"},"finish_reason":null}]}"#,
        );
        assert!(acc.has_partial_output());
        let body = acc.finalize(true);
        assert_eq!(
            body["choices"][0]["finish_reason"], "length",
            "interrupted partial must reuse the truncation-recovery path"
        );
        assert_eq!(
            body["choices"][0]["message"]["content"],
            "partial answer that was cut"
        );
    }

    #[test]
    fn full_round_trip_through_framer_and_accumulator() {
        let mut framer = SseFramer::default();
        let mut acc = StreamAccumulator::default();
        let stream_bytes = [
            chunk(r#"{"choices":[{"delta":{"role":"assistant","content":"A"},"finish_reason":null}]}"#),
            chunk(r#"{"choices":[{"delta":{"content":"B"},"finish_reason":"stop"}]}"#),
            chunk("[DONE]"),
        ]
        .concat();
        // Feed in awkward 7-byte slices to exercise partial-event handling.
        for piece in stream_bytes.as_bytes().chunks(7) {
            for payload in framer.feed(piece) {
                acc.apply_payload(&payload);
            }
        }
        let body = acc.finalize(false);
        assert_eq!(body["choices"][0]["message"]["content"], "AB");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
    }
}
