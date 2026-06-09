//! Local Whisper STT fallback when native `input_audio` is skipped.

use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::process::Command;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::config::{AudioConfig, SttConfig};
use crate::traits::MessageAttachment;

use crate::agent::audio::encode_audio_attachment;

pub const TRANSCRIPTION_PREFIX: &str = "[Transcription of ";
pub const STT_FAILED_SYSTEM_HINT: &str = "[SYSTEM] User attached audio saved to disk; local speech-to-text failed or is disabled — acknowledge receipt and ask them to type what they said.";

/// Whether persisted/rendered user text already includes an STT block.
pub fn content_has_transcription(text: &str) -> bool {
    text.contains(TRANSCRIPTION_PREFIX)
}

/// Format a transcription line appended to the user message.
pub fn format_transcription_line(filename: &str, transcript: &str) -> String {
    format!("{TRANSCRIPTION_PREFIX}{filename}]: {transcript}")
}

/// True when native multimodal audio blocks would be sent for these attachments.
pub fn native_audio_will_encode(
    audio: &AudioConfig,
    model: &str,
    attachments: &[MessageAttachment],
) -> bool {
    if !audio.enabled || attachments.is_empty() || !audio.model_supports_audio(model) {
        return false;
    }

    attachments.iter().any(|attachment| {
        if !audio.mime_allowed(&attachment.mime_type) {
            return false;
        }
        let path = Path::new(&attachment.local_path);
        encode_audio_attachment(path, &attachment.mime_type, audio.max_audio_bytes).is_ok()
    })
}

/// Whether Whisper fallback should run for this turn.
pub fn should_run_stt_fallback(
    stt: &SttConfig,
    audio: &AudioConfig,
    model: &str,
    attachments: &[MessageAttachment],
) -> bool {
    if !stt.enabled || attachments.is_empty() || native_audio_will_encode(audio, model, attachments)
    {
        return false;
    }

    attachments
        .iter()
        .any(|attachment| stt.mime_allowed(&attachment.mime_type))
}

/// Append Whisper transcriptions to inbound user text (fallback mode only).
pub async fn maybe_enrich_user_text(
    user_text: &str,
    attachments: &[MessageAttachment],
    stt: &SttConfig,
    audio: &AudioConfig,
    model: &str,
) -> String {
    if !should_run_stt_fallback(stt, audio, model, attachments) {
        return user_text.to_string();
    }

    let mut enriched = user_text.to_string();
    let mut transcribed_any = false;

    for attachment in attachments {
        if !stt.mime_allowed(&attachment.mime_type) {
            continue;
        }
        let path = Path::new(&attachment.local_path);
        match transcribe_attachment(path, &attachment.mime_type, stt).await {
            Ok(transcript) if !transcript.trim().is_empty() => {
                transcribed_any = true;
                if !enriched.is_empty() {
                    enriched.push('\n');
                }
                enriched.push_str(&format_transcription_line(
                    &attachment.filename,
                    transcript.trim(),
                ));
                info!(
                    path = %attachment.local_path,
                    filename = %attachment.filename,
                    chars = transcript.trim().len(),
                    "STT transcription appended to user message"
                );
            }
            Ok(_) => {
                warn!(
                    path = %attachment.local_path,
                    filename = %attachment.filename,
                    "Whisper returned empty transcript"
                );
            }
            Err(err) => {
                warn!(
                    path = %attachment.local_path,
                    filename = %attachment.filename,
                    error = %err,
                    "Whisper transcription failed"
                );
            }
        }
    }

    if !transcribed_any {
        return user_text.to_string();
    }

    enriched
}

#[derive(Debug)]
pub enum SttError {
    Io(std::io::Error),
    UnsupportedMime(String),
    TooLarge { size_bytes: u64, max_bytes: u64 },
    MissingBinary { path: PathBuf },
    MissingModel { path: PathBuf },
    FfmpegFailed(String),
    WhisperFailed(String),
    Timeout,
    EmptyTranscript,
}

impl std::fmt::Display for SttError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::UnsupportedMime(mime) => write!(f, "unsupported mime type: {mime}"),
            Self::TooLarge {
                size_bytes,
                max_bytes,
            } => write!(
                f,
                "audio too large for STT ({size_bytes} bytes > {max_bytes} byte limit)"
            ),
            Self::MissingBinary { path } => write!(f, "stt cli not found: {}", path.display()),
            Self::MissingModel { path } => write!(f, "stt model not found: {}", path.display()),
            Self::FfmpegFailed(msg) => write!(f, "ffmpeg failed: {msg}"),
            Self::WhisperFailed(msg) => write!(f, "whisper-cli failed: {msg}"),
            Self::Timeout => write!(f, "stt timed out"),
            Self::EmptyTranscript => write!(f, "empty transcript"),
        }
    }
}

impl From<std::io::Error> for SttError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

async fn transcribe_attachment(
    path: &Path,
    mime: &str,
    stt: &SttConfig,
) -> Result<String, SttError> {
    if !stt.mime_allowed(mime) {
        return Err(SttError::UnsupportedMime(mime.to_string()));
    }
    if !path.is_file() {
        return Err(SttError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("audio file not found: {}", path.display()),
        )));
    }

    let metadata = std::fs::metadata(path)?;
    if metadata.len() > stt.max_audio_bytes {
        return Err(SttError::TooLarge {
            size_bytes: metadata.len(),
            max_bytes: stt.max_audio_bytes,
        });
    }
    if !stt.cli_path.is_file() {
        return Err(SttError::MissingBinary {
            path: stt.cli_path.clone(),
        });
    }
    if !stt.model_path.is_file() {
        return Err(SttError::MissingModel {
            path: stt.model_path.clone(),
        });
    }

    let work_dir = std::env::temp_dir().join(format!("aidaemon-stt-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&work_dir)?;
    let _cleanup = SttWorkDir(work_dir.clone());

    let wav_path = if mime == "audio/wav" || mime == "audio/x-wav" {
        path.to_path_buf()
    } else {
        let converted = work_dir.join("input.wav");
        convert_to_wav(path, &converted, stt).await?;
        converted
    };

    let output_prefix = work_dir.join("out");
    let mut cmd = Command::new(&stt.cli_path);
    cmd.arg("-m")
        .arg(&stt.model_path)
        .arg("-f")
        .arg(&wav_path)
        .arg("--no-timestamps")
        .arg("-otxt")
        .arg("-of")
        .arg(&output_prefix);
    if stt.language != "auto" {
        cmd.arg("-l").arg(&stt.language);
    }

    let run = timeout(Duration::from_secs(stt.timeout_secs), cmd.output())
        .await
        .map_err(|_| SttError::Timeout)??;

    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        let stdout = String::from_utf8_lossy(&run.stdout);
        return Err(SttError::WhisperFailed(format!(
            "exit {:?}: {stderr}{stdout}",
            run.status.code()
        )));
    }

    let txt_path = output_prefix.with_extension("txt");
    if txt_path.is_file() {
        let transcript = std::fs::read_to_string(&txt_path)?;
        let trimmed = transcript.trim().to_string();
        if trimmed.is_empty() {
            return Err(SttError::EmptyTranscript);
        }
        return Ok(trimmed);
    }

    let stdout = String::from_utf8_lossy(&run.stdout);
    let transcript = extract_transcript_from_stdout(&stdout);
    if transcript.is_empty() {
        return Err(SttError::EmptyTranscript);
    }
    Ok(transcript)
}

async fn convert_to_wav(src: &Path, dest: &Path, stt: &SttConfig) -> Result<(), SttError> {
    let output = Command::new(&stt.ffmpeg_path)
        .args([
            "-y",
            "-i",
            src.to_string_lossy().as_ref(),
            "-ac",
            "1",
            "-ar",
            "16000",
            dest.to_string_lossy().as_ref(),
        ])
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SttError::FfmpegFailed(stderr.trim().to_string()));
    }
    Ok(())
}

fn extract_transcript_from_stdout(stdout: &str) -> String {
    stdout
        .lines()
        .map(str::trim)
        .filter(|line| {
            !line.is_empty()
                && !line.starts_with("whisper_")
                && !line.starts_with("ggml_")
                && !line.starts_with("load_backend:")
                && !line.starts_with("main:")
                && !line.starts_with("system_info:")
        })
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

struct SttWorkDir(PathBuf);

impl Drop for SttWorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AudioConfig;

    fn stt_config() -> SttConfig {
        SttConfig {
            enabled: true,
            cli_path: PathBuf::from("/opt/homebrew/bin/whisper-cli"),
            model_path: PathBuf::from("/tmp/model.bin"),
            ffmpeg_path: PathBuf::from("ffmpeg"),
            language: "en".to_string(),
            max_audio_bytes: 25 * 1_048_576,
            timeout_secs: 120,
            mime_types: vec!["audio/ogg".to_string(), "audio/wav".to_string()],
        }
    }

    fn audio_config(patterns: &[&str], mime_types: &[&str]) -> AudioConfig {
        AudioConfig {
            enabled: true,
            max_audio_bytes: 10 * 1_048_576,
            mime_types: mime_types.iter().map(|s| s.to_string()).collect(),
            model_patterns: patterns.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn attachment(mime: &str) -> MessageAttachment {
        MessageAttachment {
            local_path: "/tmp/voice.ogg".to_string(),
            filename: "voice.ogg".to_string(),
            mime_type: mime.to_string(),
            size_bytes: 1024,
            provenance: crate::traits::AttachmentProvenance::Inbound,
            source_tool: None,
        }
    }

    #[test]
    fn should_run_stt_when_native_audio_ineligible() {
        let attachments = vec![attachment("audio/ogg")];
        assert!(should_run_stt_fallback(
            &stt_config(),
            &audio_config(&["gemini-2"], &["audio/ogg"]),
            "gemma-4-26b",
            &attachments,
        ));
    }

    #[test]
    fn should_not_run_stt_when_native_audio_eligible() {
        let mut tmp = tempfile::NamedTempFile::new().expect("temp wav");
        std::io::Write::write_all(&mut tmp, b"RIFF....WAVEfmt ").expect("write wav");
        let path = tmp.path().to_string_lossy().into_owned();
        let attachments = vec![MessageAttachment {
            local_path: path,
            filename: "voice.wav".to_string(),
            mime_type: "audio/wav".to_string(),
            size_bytes: 16,
            provenance: crate::traits::AttachmentProvenance::Inbound,
            source_tool: None,
        }];
        assert!(!should_run_stt_fallback(
            &stt_config(),
            &audio_config(&["gemma"], &["audio/wav"]),
            "gemma-4-26b",
            &attachments,
        ));
    }

    #[test]
    fn should_not_run_stt_when_disabled() {
        let mut stt = stt_config();
        stt.enabled = false;
        let attachments = vec![attachment("audio/ogg")];
        assert!(!should_run_stt_fallback(
            &stt,
            &audio_config(&["gemini-2"], &["audio/ogg"]),
            "gemma-4-26b",
            &attachments,
        ));
    }

    #[test]
    fn content_has_transcription_detects_marker() {
        let line = format_transcription_line("voice.ogg", "hello there");
        assert!(content_has_transcription(&line));
        assert!(!content_has_transcription("[File received: voice.ogg]"));
    }

    #[test]
    fn extract_transcript_from_stdout_skips_logs() {
        let stdout = "main: processing '/tmp/a.wav'\n\n Okay, who is my dad?";
        assert_eq!(
            extract_transcript_from_stdout(stdout),
            "Okay, who is my dad?"
        );
    }
}
