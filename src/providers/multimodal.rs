//! OpenAI-style multimodal content translation for provider adapters.

use serde_json::Value;

/// Strip image and audio blocks from message content for text-only provider retry.
pub fn strip_multimodal_blocks_from_messages(messages: &mut [Value]) {
    for msg in messages.iter_mut() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        if let Some(content) = msg.get("content") {
            msg["content"] = strip_multimodal_blocks_from_content_value(content);
        }
    }
}

pub fn messages_contain_multimodal_blocks(messages: &[Value]) -> bool {
    messages.iter().any(|msg| {
        msg.get("role").and_then(|r| r.as_str()) == Some("user")
            && msg
                .get("content")
                .and_then(|c| c.as_array())
                .is_some_and(|blocks| {
                    blocks.iter().any(|block| {
                        matches!(
                            block.get("type").and_then(|t| t.as_str()),
                            Some("image_url") | Some("input_audio")
                        )
                    })
                })
    })
}

fn strip_multimodal_blocks_from_content_value(content: &Value) -> Value {
    match content {
        Value::String(s) => Value::String(s.clone()),
        Value::Array(blocks) => {
            let text_blocks: Vec<Value> = blocks
                .iter()
                .filter(|block| block.get("type").and_then(|t| t.as_str()) == Some("text"))
                .cloned()
                .collect();
            if text_blocks.is_empty() {
                Value::String(String::new())
            } else if text_blocks.len() == 1 {
                text_blocks[0]["text"]
                    .as_str()
                    .map(|s| Value::String(s.to_string()))
                    .unwrap_or(Value::String(String::new()))
            } else {
                Value::Array(text_blocks)
            }
        }
        other => other.clone(),
    }
}
