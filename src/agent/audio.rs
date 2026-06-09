//! Lazy audio encoding for native multimodal LLM input (current turn only).

use std::path::Path;

use base64::Engine;
use serde_json::{json, Value};
use tracing::warn;

use crate::config::AudioConfig;
use crate::traits::MessageAttachment;

use crate::agent::turn_render::RenderMode;

pub const AUDIO_SKIPPED_SYSTEM_HINT: &str = "[SYSTEM] User attached audio saved to disk; native audio input is disabled, unsupported for this model, or the file could not be read — acknowledge receipt and ask them to describe it or switch to an audio-capable model.";

/// OpenAI `input_audio.format` value for a MIME type.
pub fn mime_to_openai_format(mime: &str) -> Option<&'static str> {
    match mime {
        "audio/ogg" | "audio/webm" => Some("opus"),
        "audio/mpeg" | "audio/mp3" => Some("mp3"),
        "audio/wav" => Some("wav"),
        "audio/flac" => Some("flac"),
        "audio/aac" => Some("aac"),
        _ => None,
    }
}

/// Gemini `inlineData.mimeType` for an OpenAI audio format token.
#[allow(dead_code)]
pub fn openai_format_to_gemini_mime(format: &str) -> Option<&'static str> {
    match format {
        "opus" => Some("audio/ogg"),
        "mp3" => Some("audio/mp3"),
        "wav" => Some("audio/wav"),
        "flac" => Some("audio/flac"),
        "aac" => Some("audio/aac"),
        _ => None,
    }
}

/// Read and base64-encode an audio file when within size limits.
pub fn encode_audio_attachment(
    path: &Path,
    mime: &str,
    max_bytes: u64,
) -> Result<(String, String), EncodeAudioError> {
    let openai_format = mime_to_openai_format(mime)
        .ok_or_else(|| EncodeAudioError::UnsupportedMime(mime.to_string()))?;
    let metadata = std::fs::metadata(path).map_err(EncodeAudioError::Io)?;
    if metadata.len() > max_bytes {
        return Err(EncodeAudioError::TooLarge {
            size_bytes: metadata.len(),
            max_bytes,
        });
    }
    let bytes = std::fs::read(path).map_err(EncodeAudioError::Io)?;
    if bytes.len() as u64 > max_bytes {
        return Err(EncodeAudioError::TooLarge {
            size_bytes: bytes.len() as u64,
            max_bytes,
        });
    }
    if !mime.starts_with("audio/") {
        return Err(EncodeAudioError::UnsupportedMime(mime.to_string()));
    }
    Ok((
        base64::engine::general_purpose::STANDARD.encode(bytes),
        openai_format.to_string(),
    ))
}

#[derive(Debug)]
pub enum EncodeAudioError {
    Io(std::io::Error),
    TooLarge { size_bytes: u64, max_bytes: u64 },
    UnsupportedMime(String),
}

impl std::fmt::Display for EncodeAudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "failed to read audio: {e}"),
            Self::TooLarge {
                size_bytes,
                max_bytes,
            } => write!(
                f,
                "audio size {size_bytes} bytes exceeds audio cap {max_bytes} bytes"
            ),
            Self::UnsupportedMime(m) => write!(f, "unsupported audio mime: {m}"),
        }
    }
}

pub fn build_input_audio_block(data: &str, format: &str) -> Value {
    json!({
        "type": "input_audio",
        "input_audio": {
            "data": data,
            "format": format,
        }
    })
}

pub struct AudioBlocksResult {
    pub blocks: Vec<Value>,
    pub encoded_any: bool,
    pub skipped_any: bool,
}

/// Build `input_audio` blocks for eligible audio attachments.
pub fn build_audio_blocks(
    attachments: &[MessageAttachment],
    mode: RenderMode,
    audio: &AudioConfig,
    model: &str,
) -> AudioBlocksResult {
    let encode =
        matches!(mode, RenderMode::Current) && audio.enabled && audio.model_supports_audio(model);

    if !encode {
        return AudioBlocksResult {
            blocks: Vec::new(),
            encoded_any: false,
            skipped_any: attachments.iter().any(|a| audio.mime_allowed(&a.mime_type)),
        };
    }

    let audio_attachments: Vec<_> = attachments
        .iter()
        .filter(|a| audio.mime_allowed(&a.mime_type))
        .collect();

    if audio_attachments.is_empty() {
        return AudioBlocksResult {
            blocks: Vec::new(),
            encoded_any: false,
            skipped_any: false,
        };
    }

    let mut blocks = Vec::new();
    let mut encoded_any = false;
    let mut skipped_any = false;

    for attachment in audio_attachments {
        let path = Path::new(&attachment.local_path);
        match encode_audio_attachment(path, &attachment.mime_type, audio.max_audio_bytes) {
            Ok((data, format)) => {
                encoded_any = true;
                blocks.push(build_input_audio_block(&data, &format));
            }
            Err(err) => {
                skipped_any = true;
                warn!(
                    path = %attachment.local_path,
                    mime = %attachment.mime_type,
                    error = %err,
                    "Skipping audio encoding for attachment"
                );
            }
        }
    }

    AudioBlocksResult {
        blocks,
        encoded_any,
        skipped_any,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn audio_config() -> AudioConfig {
        AudioConfig {
            enabled: true,
            max_audio_bytes: 1_048_576,
            mime_types: vec!["audio/ogg".to_string(), "audio/wav".to_string()],
            model_patterns: vec!["gemini-2".to_string()],
        }
    }

    #[test]
    fn mime_to_openai_format_maps_voice_ogg() {
        assert_eq!(mime_to_openai_format("audio/ogg"), Some("opus"));
        assert_eq!(mime_to_openai_format("audio/wav"), Some("wav"));
        assert!(mime_to_openai_format("audio/unknown").is_none());
    }

    #[test]
    fn encode_rejects_non_audio_mime() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 4]).unwrap();
        let err = encode_audio_attachment(f.path(), "image/png", 1024).unwrap_err();
        assert!(matches!(err, EncodeAudioError::UnsupportedMime(_)));
    }

    #[test]
    fn build_audio_blocks_requires_eligible_model() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[1, 2, 3, 4]).unwrap();
        let attachments = vec![MessageAttachment {
            local_path: f.path().to_string_lossy().into_owned(),
            filename: "voice.ogg".to_string(),
            mime_type: "audio/ogg".to_string(),
            size_bytes: 4,
            ..Default::default()
        }];
        let result = build_audio_blocks(
            &attachments,
            RenderMode::Current,
            &audio_config(),
            "gpt-4o-mini",
        );
        assert!(!result.encoded_any);
        assert!(result.skipped_any);
        assert!(result.blocks.is_empty());
    }

    #[test]
    fn build_audio_blocks_encodes_for_eligible_model() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[1, 2, 3, 4]).unwrap();
        let attachments = vec![MessageAttachment {
            local_path: f.path().to_string_lossy().into_owned(),
            filename: "voice.ogg".to_string(),
            mime_type: "audio/ogg".to_string(),
            size_bytes: 4,
            ..Default::default()
        }];
        let result = build_audio_blocks(
            &attachments,
            RenderMode::Current,
            &audio_config(),
            "gemini-2.0-flash",
        );
        assert!(result.encoded_any);
        assert_eq!(result.blocks.len(), 1);
        assert_eq!(result.blocks[0]["type"], "input_audio");
        assert_eq!(result.blocks[0]["input_audio"]["format"], "opus");
    }
}
