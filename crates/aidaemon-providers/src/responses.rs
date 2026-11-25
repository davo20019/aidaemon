//! What both providers on OpenAI's Responses API (xAI, ChatGPT) send the
//! same way: function calls, and the results of tool calls. Each keeps its
//! own message items (ChatGPT's developer role and reasoning replay).

use std::collections::HashSet;

use aidaemon_core::{Image, Message, ToolCall};
use serde_json::{Value, json};

use crate::transport::raw_arguments;

/// A `function_call` item, its arguments replayed as the model sent them.
pub(crate) fn function_call(call: &ToolCall) -> Value {
    let arguments = raw_arguments(&call.arguments)
        .map(str::to_string)
        .unwrap_or_else(|| call.arguments.to_string());
    json!({
        "type": "function_call",
        "call_id": call.id,
        "name": call.name,
        "arguments": arguments,
    })
}

/// An `input_image` content part.
pub(crate) fn input_image(image: &Image) -> Value {
    json!({"type": "input_image", "image_url": image.to_url()})
}

/// The tool results of one request: a result whose call was never sent is
/// dropped (the API refuses an orphan `function_call_output`), and images
/// a tool returned (which an output item cannot carry) are gathered for a
/// user message after the run of results.
#[derive(Default)]
pub(crate) struct ToolResults {
    pending: HashSet<String>,
    images: Vec<Value>,
}

impl ToolResults {
    /// `call` was sent; its result may follow.
    pub(crate) fn sent(&mut self, call: &ToolCall) {
        if !call.id.is_empty() {
            self.pending.insert(call.id.clone());
        }
    }

    /// The `function_call_output` item for a tool message, `None` for an
    /// orphan. Its images wait for [`Self::take_images`].
    pub(crate) fn output(&mut self, message: &Message) -> Option<Value> {
        let call_id = message.tool_call_id.as_deref().unwrap_or_default();
        if !self.pending.remove(call_id) {
            return None;
        }
        if !message.images.is_empty() {
            self.images.push(json!({
                "type": "input_text",
                "text": format!("[Images returned by tool call {call_id}]"),
            }));
            self.images.extend(message.images.iter().map(input_image));
        }
        Some(json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": message.content,
        }))
    }

    /// The content of the user message that carries the tool images
    /// gathered so far, if any.
    pub(crate) fn take_images(&mut self) -> Option<Vec<Value>> {
        (!self.images.is_empty()).then(|| std::mem::take(&mut self.images))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_orphan_result_is_dropped_and_tool_images_wait_for_a_user_message() {
        let mut results = ToolResults::default();
        assert!(
            results
                .output(&Message::tool_result("never-sent", "x"))
                .is_none()
        );
        results.sent(&ToolCall::new("c1", "screenshot", json!({})));
        let mut shot = Message::tool_result("c1", "taken");
        shot.images = vec![Image::url("https://example.com/shot.png")];
        let item = results.output(&shot).unwrap();
        assert_eq!(item["type"], "function_call_output");
        assert_eq!(item["call_id"], "c1");
        let images = results.take_images().unwrap();
        assert_eq!(images[1]["image_url"], "https://example.com/shot.png");
        assert!(results.take_images().is_none());
        // A result is answered once.
        assert!(results.output(&shot).is_none());
    }
}
