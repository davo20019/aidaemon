//! OpenAI-style multimodal content translation for provider adapters.

use serde_json::{json, Value};

fn openai_format_to_gemini_mime(format: &str) -> Option<&'static str> {
    match format {
        "opus" => Some("audio/ogg"),
        "mp3" => Some("audio/mp3"),
        "wav" => Some("audio/wav"),
        "flac" => Some("audio/flac"),
        "aac" => Some("audio/aac"),
        _ => None,
    }
}

/// Convert OpenAI-style user content (string or block array) to Anthropic blocks.
pub fn openai_content_to_anthropic_blocks(content: &Value) -> Value {
    match content {
        Value::String(text) => json!(text),
        Value::Array(blocks) => {
            let mapped: Vec<Value> = blocks
                .iter()
                .filter_map(openai_block_to_anthropic)
                .collect();
            if mapped.is_empty() {
                json!("")
            } else if mapped.len() == 1
                && mapped[0].get("type").and_then(|t| t.as_str()) == Some("text")
            {
                mapped[0]["text"].clone()
            } else {
                Value::Array(mapped)
            }
        }
        other => other.clone(),
    }
}

fn openai_block_to_anthropic(block: &Value) -> Option<Value> {
    match block.get("type").and_then(|t| t.as_str()) {
        Some("text") => Some(json!({
            "type": "text",
            "text": block.get("text").and_then(|t| t.as_str()).unwrap_or(""),
        })),
        Some("image_url") => {
            let url = block
                .get("image_url")
                .and_then(|u| u.get("url"))
                .and_then(|u| u.as_str())?;
            parse_data_url_image(url).map(|(media_type, data)| {
                json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    }
                })
            })
        }
        // Anthropic has no native audio input — drop.
        Some("input_audio") => None,
        _ => None,
    }
}

/// Convert OpenAI-style user content to Gemini parts.
pub fn openai_content_to_gemini_parts(content: &Value) -> Vec<Value> {
    match content {
        Value::String(text) => vec![json!({ "text": text })],
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(openai_block_to_gemini_part)
            .collect(),
        _ => vec![json!({ "text": "" })],
    }
}

fn openai_block_to_gemini_part(block: &Value) -> Option<Value> {
    match block.get("type").and_then(|t| t.as_str()) {
        Some("text") => Some(json!({
            "text": block.get("text").and_then(|t| t.as_str()).unwrap_or(""),
        })),
        Some("image_url") => {
            let url = block
                .get("image_url")
                .and_then(|u| u.get("url"))
                .and_then(|u| u.as_str())?;
            parse_data_url_image(url).map(|(mime_type, data)| {
                json!({
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": data,
                    }
                })
            })
        }
        Some("input_audio") => {
            let input = block.get("input_audio")?;
            let data = input.get("data").and_then(|d| d.as_str())?;
            let format = input.get("format").and_then(|f| f.as_str())?;
            let mime_type = openai_format_to_gemini_mime(format)?;
            Some(json!({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": data,
                }
            }))
        }
        _ => None,
    }
}

fn parse_data_url_image(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(";base64,")?;
    if !meta.starts_with("image/") {
        return None;
    }
    Some((meta.to_string(), data.to_string()))
}

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

pub fn messages_contain_audio_blocks(messages: &[Value]) -> bool {
    messages.iter().any(|msg| {
        msg.get("role").and_then(|r| r.as_str()) == Some("user")
            && msg
                .get("content")
                .and_then(|c| c.as_array())
                .is_some_and(|blocks| {
                    blocks.iter().any(|block| {
                        block.get("type").and_then(|t| t.as_str()) == Some("input_audio")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_maps_text_and_image_blocks() {
        let content = json!([
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
        ]);
        let blocks = openai_content_to_anthropic_blocks(&content);
        let arr = blocks.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[1]["type"], "image");
        assert_eq!(arr[1]["source"]["media_type"], "image/png");
    }

    #[test]
    fn anthropic_drops_input_audio() {
        let content = json!([
            {"type": "text", "text": "listen"},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "opus"}}
        ]);
        let blocks = openai_content_to_anthropic_blocks(&content);
        assert_eq!(blocks.as_str(), Some("listen"));
    }

    #[test]
    fn gemini_maps_multiple_images() {
        let content = json!([
            {"type": "text", "text": "compare"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}}
        ]);
        let parts = openai_content_to_gemini_parts(&content);
        assert_eq!(parts.len(), 3);
        assert!(parts[1].get("inlineData").is_some());
        assert!(parts[2].get("inlineData").is_some());
    }

    #[test]
    fn gemini_maps_input_audio_ogg() {
        let content = json!([
            {"type": "text", "text": "transcribe"},
            {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "opus"}}
        ]);
        let parts = openai_content_to_gemini_parts(&content);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1]["inlineData"]["mimeType"], "audio/ogg");
        assert_eq!(parts[1]["inlineData"]["data"], "QUJD");
    }
}
