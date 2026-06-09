//! Lazy image encoding for LLM vision (current turn only).

use std::path::Path;

use base64::Engine;
use serde_json::{json, Value};
use tracing::warn;

use crate::config::VisionConfig;
use crate::traits::MessageAttachment;

use crate::agent::turn_render::RenderMode;

pub const VISION_SKIPPED_SYSTEM_HINT: &str = "[SYSTEM] User attached image(s) saved to disk; vision is disabled, unsupported, or the file could not be read — acknowledge receipt and explain you cannot view the image directly.";

/// Read and base64-encode an image file when within size limits.
pub fn encode_image_attachment(
    path: &Path,
    mime: &str,
    max_bytes: u64,
) -> Result<String, EncodeImageError> {
    let metadata = std::fs::metadata(path).map_err(EncodeImageError::Io)?;
    if metadata.len() > max_bytes {
        return Err(EncodeImageError::TooLarge {
            size_bytes: metadata.len(),
            max_bytes,
        });
    }
    let bytes = std::fs::read(path).map_err(EncodeImageError::Io)?;
    if bytes.len() as u64 > max_bytes {
        return Err(EncodeImageError::TooLarge {
            size_bytes: bytes.len() as u64,
            max_bytes,
        });
    }
    if !mime.starts_with("image/") {
        return Err(EncodeImageError::UnsupportedMime(mime.to_string()));
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

#[derive(Debug)]
pub enum EncodeImageError {
    Io(std::io::Error),
    TooLarge { size_bytes: u64, max_bytes: u64 },
    UnsupportedMime(String),
}

impl std::fmt::Display for EncodeImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "failed to read image: {e}"),
            Self::TooLarge {
                size_bytes,
                max_bytes,
            } => write!(
                f,
                "image size {size_bytes} bytes exceeds vision cap {max_bytes} bytes"
            ),
            Self::UnsupportedMime(m) => write!(f, "unsupported image mime: {m}"),
        }
    }
}

/// Result of building wire-format user content for a message with attachments.
pub struct MultimodalBuildResult {
    pub content: Value,
    pub vision_skipped: bool,
}

/// Build OpenAI-style multimodal `content` for a user message.
pub fn build_multimodal_content(
    text: &str,
    attachments: &[MessageAttachment],
    mode: RenderMode,
    vision: &VisionConfig,
) -> MultimodalBuildResult {
    let text = text.trim();
    let encode_vision = matches!(mode, RenderMode::Current) && vision.enabled;

    if !encode_vision || attachments.is_empty() {
        return MultimodalBuildResult {
            content: Value::String(text.to_string()),
            vision_skipped: !attachments.is_empty(),
        };
    }

    let image_attachments: Vec<_> = attachments
        .iter()
        .filter(|a| vision.mime_allowed(&a.mime_type))
        .collect();

    if image_attachments.is_empty() {
        return MultimodalBuildResult {
            content: Value::String(text.to_string()),
            vision_skipped: !attachments.is_empty(),
        };
    }

    let mut blocks = Vec::new();
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }

    let mut encoded_any = false;
    let mut skipped_any = false;

    for attachment in image_attachments {
        let path = Path::new(&attachment.local_path);
        match encode_image_attachment(path, &attachment.mime_type, vision.max_image_bytes) {
            Ok(data) => {
                encoded_any = true;
                blocks.push(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{};base64,{}", attachment.mime_type, data)
                    }
                }));
            }
            Err(err) => {
                skipped_any = true;
                warn!(
                    path = %attachment.local_path,
                    mime = %attachment.mime_type,
                    error = %err,
                    "Skipping vision encoding for attachment"
                );
            }
        }
    }

    if !encoded_any {
        return MultimodalBuildResult {
            content: Value::String(text.to_string()),
            vision_skipped: true,
        };
    }

    MultimodalBuildResult {
        content: Value::Array(blocks),
        vision_skipped: skipped_any,
    }
}

/// Extract plain text from OpenAI-style string or multimodal content.
pub fn content_value_as_text(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(blocks) => {
            let parts: Vec<String> = blocks
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                        block
                            .get("text")
                            .and_then(|t| t.as_str())
                            .map(str::to_string)
                    } else {
                        None
                    }
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
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
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn vision_config() -> VisionConfig {
        VisionConfig {
            enabled: true,
            max_image_bytes: 4 * 1_048_576,
            mime_types: default_vision_mime_types(),
        }
    }

    fn default_vision_mime_types() -> Vec<String> {
        vec![
            "image/jpeg".to_string(),
            "image/png".to_string(),
            "image/gif".to_string(),
            "image/webp".to_string(),
        ]
    }

    #[test]
    fn build_multimodal_content_multiple_images() {
        let mut f1 = NamedTempFile::new().unwrap();
        f1.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
            .unwrap();
        let mut f2 = NamedTempFile::new().unwrap();
        f2.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
            .unwrap();

        let attachments = vec![
            MessageAttachment {
                local_path: f1.path().to_string_lossy().into_owned(),
                filename: "a.png".to_string(),
                mime_type: "image/png".to_string(),
                size_bytes: 8,
            },
            MessageAttachment {
                local_path: f2.path().to_string_lossy().into_owned(),
                filename: "b.png".to_string(),
                mime_type: "image/png".to_string(),
                size_bytes: 8,
            },
        ];

        let result = build_multimodal_content(
            "compare these",
            &attachments,
            RenderMode::Current,
            &vision_config(),
        );
        let blocks = result.content.as_array().unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image_url");
        assert_eq!(blocks[2]["type"], "image_url");
    }

    #[test]
    fn archived_mode_keeps_text_only() {
        let attachments = vec![MessageAttachment {
            local_path: "/tmp/missing.png".to_string(),
            filename: "missing.png".to_string(),
            mime_type: "image/png".to_string(),
            size_bytes: 1,
        }];
        let result = build_multimodal_content(
            "stub text",
            &attachments,
            RenderMode::Archived {
                terminal_state: crate::events::TerminalState::Completed,
            },
            &vision_config(),
        );
        assert!(result.content.is_string());
        assert_eq!(result.content.as_str(), Some("stub text"));
    }

    #[test]
    fn missing_file_falls_back_to_text() {
        let attachments = vec![MessageAttachment {
            local_path: "/tmp/definitely-not-a-real-file-vision-test.png".to_string(),
            filename: "x.png".to_string(),
            mime_type: "image/png".to_string(),
            size_bytes: 1,
        }];
        let result =
            build_multimodal_content("hello", &attachments, RenderMode::Current, &vision_config());
        assert!(result.content.is_string());
        assert!(result.vision_skipped);
    }
}
