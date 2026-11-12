//! Provider- and agent-neutral helpers for structured chat content.

use serde_json::Value;

/// Extract plain text from OpenAI-style string or multimodal content.
pub fn content_value_as_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(blocks) => {
            let parts: Vec<String> = blocks
                .iter()
                .filter_map(|block| {
                    (block.get("type").and_then(Value::as_str) == Some("text"))
                        .then(|| {
                            block
                                .get("text")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        })
                        .flatten()
                })
                .collect();
            (!parts.is_empty()).then(|| parts.join("\n"))
        }
        _ => None,
    }
}

/// Whether a rendered user message matches the current turn's user text.
pub fn user_message_content_matches(content: &Value, user_text: &str) -> bool {
    content_value_as_text(content).is_some_and(|text| text == user_text)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::content_value_as_text;

    #[test]
    fn extracts_text_from_mixed_multimodal_blocks() {
        let content = json!([
            {"type": "text", "text": "first"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "text", "text": "second"}
        ]);
        assert_eq!(
            content_value_as_text(&content).as_deref(),
            Some("first\nsecond")
        );
    }
}
