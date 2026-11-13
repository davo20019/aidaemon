//! How an MCP tool result reads to the model.
//!
//! The transport lives in `aidaemon_mcp`. The daemon keeps its own text for
//! non-text blocks and treats `isError` as a failed call.

use serde_json::Value;

fn truncate_for_preview(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        input.to_string()
    } else {
        let head: String = input.chars().take(max_chars).collect();
        format!("{}...", head)
    }
}

fn render_mcp_content_block(block: &Value) -> Option<String> {
    let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");

    match block_type {
        "text" => block
            .get("text")
            .and_then(Value::as_str)
            .map(|t| t.to_string()),
        "image" => {
            let mime = block
                .get("mimeType")
                .or_else(|| block.get("mime_type"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");

            if let Some(url) = block.get("url").and_then(Value::as_str) {
                return Some(format!(
                    "[MCP image content: mime={}, url={}]",
                    mime,
                    truncate_for_preview(url, 160)
                ));
            }

            if let Some(data) = block.get("data").and_then(Value::as_str) {
                let approx_bytes = data.len().saturating_mul(3) / 4;
                return Some(format!(
                    "[MCP image content: mime={}, inline_data~{} bytes]",
                    mime, approx_bytes
                ));
            }

            Some(format!("[MCP image content: mime={}]", mime))
        }
        "resource" => {
            let uri = block
                .get("resource")
                .and_then(|r| r.get("uri"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            Some(format!("[MCP resource content: {}]", uri))
        }
        "" => block
            .get("text")
            .and_then(Value::as_str)
            .map(|t| t.to_string()),
        other => Some(format!("[MCP {} content block]", other)),
    }
}

pub(super) fn format_tool_call_result(result: &Value) -> anyhow::Result<String> {
    let mut rendered_blocks = Vec::new();

    if let Some(content) = result.get("content").and_then(Value::as_array) {
        for block in content {
            if let Some(rendered) = render_mcp_content_block(block) {
                if !rendered.trim().is_empty() {
                    rendered_blocks.push(rendered);
                }
            }
        }
    }

    if rendered_blocks.is_empty() {
        if let Some(structured) = result.get("structuredContent").filter(|v| !v.is_null()) {
            rendered_blocks.push(format!("[MCP structured content]\n{}", structured));
        }
    }

    let rendered = rendered_blocks.join("\n");
    let is_error = result
        .get("isError")
        .or_else(|| result.get("is_error"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if is_error {
        let err_msg = if rendered.trim().is_empty() {
            result.to_string()
        } else {
            rendered
        };
        anyhow::bail!("MCP tool reported isError=true: {}", err_msg);
    }

    if rendered.trim().is_empty() {
        Ok(result.to_string())
    } else {
        Ok(rendered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn formats_text_blocks() {
        let result = json!({
            "content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"}
            ],
            "isError": false
        });

        let out = format_tool_call_result(&result).unwrap();
        assert_eq!(out, "line one\nline two");
    }

    #[test]
    fn preserves_non_text_blocks_as_placeholders() {
        let result = json!({
            "content": [
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "data": "aGVsbG8="
                }
            ],
            "isError": false
        });

        let out = format_tool_call_result(&result).unwrap();
        assert!(out.contains("MCP image content"));
        assert!(out.contains("image/png"));
    }

    #[test]
    fn respects_is_error_flag() {
        let result = json!({
            "content": [{"type": "text", "text": "Not allowed"}],
            "isError": true
        });

        let err = format_tool_call_result(&result).unwrap_err().to_string();
        assert!(err.contains("isError=true"));
        assert!(err.contains("Not allowed"));
    }
}
