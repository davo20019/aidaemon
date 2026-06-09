//! Compose text + image + audio blocks for user messages with attachments.

use serde_json::{json, Value};

use crate::config::{AudioConfig, VisionConfig};
use crate::traits::MessageAttachment;

use crate::agent::audio::build_audio_blocks;
use crate::agent::turn_render::RenderMode;
use crate::agent::vision::build_image_blocks;

/// Result of building wire-format user content for a message with attachments.
pub struct AttachmentBuildResult {
    pub content: Value,
    pub vision_skipped: bool,
    pub audio_skipped: bool,
}

fn has_vision_eligible(attachments: &[MessageAttachment], vision: &VisionConfig) -> bool {
    attachments
        .iter()
        .any(|a| vision.mime_allowed(&a.mime_type))
}

fn has_audio_eligible(attachments: &[MessageAttachment], audio: &AudioConfig) -> bool {
    attachments.iter().any(|a| audio.mime_allowed(&a.mime_type))
}

/// Build OpenAI-style multimodal `content` for a user message with attachments.
pub fn build_attachment_content(
    text: &str,
    attachments: &[MessageAttachment],
    mode: RenderMode,
    vision: &VisionConfig,
    audio: &AudioConfig,
    model: &str,
) -> AttachmentBuildResult {
    let text = text.trim();
    let encode_media = matches!(mode, RenderMode::Current);

    if attachments.is_empty() {
        return AttachmentBuildResult {
            content: Value::String(text.to_string()),
            vision_skipped: false,
            audio_skipped: false,
        };
    }

    if !encode_media {
        return AttachmentBuildResult {
            content: Value::String(text.to_string()),
            vision_skipped: has_vision_eligible(attachments, vision),
            audio_skipped: has_audio_eligible(attachments, audio),
        };
    }

    let image_result = build_image_blocks(attachments, mode, vision);
    let audio_result = build_audio_blocks(attachments, mode, audio, model);

    let vision_skipped = has_vision_eligible(attachments, vision)
        && (!image_result.encoded_any || image_result.skipped_any);
    let audio_skipped = has_audio_eligible(attachments, audio)
        && (!audio_result.encoded_any || audio_result.skipped_any);

    if !image_result.encoded_any && !audio_result.encoded_any {
        return AttachmentBuildResult {
            content: Value::String(text.to_string()),
            vision_skipped,
            audio_skipped,
        };
    }

    let mut blocks = Vec::new();
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    blocks.extend(image_result.blocks);
    blocks.extend(audio_result.blocks);

    AttachmentBuildResult {
        content: Value::Array(blocks),
        vision_skipped,
        audio_skipped,
    }
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
            mime_types: vec![
                "image/png".to_string(),
                "image/jpeg".to_string(),
                "image/gif".to_string(),
                "image/webp".to_string(),
            ],
        }
    }

    fn audio_config() -> AudioConfig {
        AudioConfig {
            enabled: true,
            max_audio_bytes: 1_048_576,
            mime_types: vec!["audio/ogg".to_string()],
            model_patterns: vec!["gemini-2".to_string()],
        }
    }

    #[test]
    fn mixed_image_and_audio_ordering() {
        let mut png = NamedTempFile::new().unwrap();
        png.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
            .unwrap();
        let mut ogg = NamedTempFile::new().unwrap();
        ogg.write_all(&[1, 2, 3, 4]).unwrap();

        let attachments = vec![
            MessageAttachment {
                local_path: png.path().to_string_lossy().into_owned(),
                filename: "a.png".to_string(),
                mime_type: "image/png".to_string(),
                size_bytes: 8,
                ..Default::default()
            },
            MessageAttachment {
                local_path: ogg.path().to_string_lossy().into_owned(),
                filename: "voice.ogg".to_string(),
                mime_type: "audio/ogg".to_string(),
                size_bytes: 4,
                ..Default::default()
            },
        ];

        let result = build_attachment_content(
            "check both",
            &attachments,
            RenderMode::Current,
            &vision_config(),
            &audio_config(),
            "gemini-2.0-flash",
        );
        let blocks = result.content.as_array().unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image_url");
        assert_eq!(blocks[2]["type"], "input_audio");
    }
}
