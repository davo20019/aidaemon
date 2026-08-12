use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::config::NodeSpeechConfig;

pub fn configured_synthesizer(
    config: &NodeSpeechConfig,
) -> anyhow::Result<std::sync::Arc<dyn NodeSpeechSynthesizer>> {
    if !config.enabled {
        return Ok(std::sync::Arc::new(DisabledSpeech));
    }
    anyhow::ensure!(
        config.provider == "macos_say",
        "unsupported Node speech provider"
    );
    Ok(std::sync::Arc::new(MacOsSaySpeech::new(config.clone())))
}

#[derive(Debug, Clone)]
pub struct SpeechArtifact {
    pub path: PathBuf,
    pub content_type: String,
    pub size_bytes: u64,
}

#[async_trait]
pub trait NodeSpeechSynthesizer: Send + Sync {
    async fn synthesize(
        &self,
        text: &str,
        output_dir: &Path,
        max_bytes: usize,
    ) -> anyhow::Result<Option<SpeechArtifact>>;
}

pub struct DisabledSpeech;
#[async_trait]
impl NodeSpeechSynthesizer for DisabledSpeech {
    async fn synthesize(
        &self,
        _text: &str,
        _output_dir: &Path,
        _max_bytes: usize,
    ) -> anyhow::Result<Option<SpeechArtifact>> {
        Ok(None)
    }
}

pub struct MacOsSaySpeech {
    config: NodeSpeechConfig,
}
impl MacOsSaySpeech {
    pub fn new(config: NodeSpeechConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl NodeSpeechSynthesizer for MacOsSaySpeech {
    async fn synthesize(
        &self,
        text: &str,
        output_dir: &Path,
        max_bytes: usize,
    ) -> anyhow::Result<Option<SpeechArtifact>> {
        anyhow::ensure!(
            cfg!(target_os = "macos"),
            "macos_say speech provider requires macOS"
        );
        anyhow::ensure!(
            !text.trim().is_empty() && text.chars().count() <= 2_000,
            "speech text is outside the supported bounds"
        );
        tokio::fs::create_dir_all(output_dir).await?;
        let id = uuid::Uuid::new_v4().simple().to_string();
        let aiff = output_dir.join(format!("response-{id}.aiff"));
        let wav = output_dir.join(format!("response-{id}.wav"));
        let say_status = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.timeout_seconds),
            tokio::process::Command::new("/usr/bin/say")
                .arg("-v")
                .arg(&self.config.voice)
                .arg("-o")
                .arg(&aiff)
                .arg(text)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("speech synthesis timed out"))??;
        anyhow::ensure!(say_status.success(), "speech synthesis command failed");
        let ffmpeg_status = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.timeout_seconds),
            tokio::process::Command::new("/opt/homebrew/bin/ffmpeg")
                .args(["-nostdin", "-loglevel", "error", "-y", "-i"])
                .arg(&aiff)
                .args([
                    "-af",
                    "highpass=f=120,lowpass=f=9500,equalizer=f=2800:t=q:w=1:g=2",
                    "-ar",
                    &self.config.sample_rate_hz.to_string(),
                    "-ac",
                    "2",
                    "-c:a",
                    "pcm_s16le",
                    "-fflags",
                    "+bitexact",
                    "-flags:a",
                    "+bitexact",
                ])
                .arg(&wav)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("speech conversion timed out"))??;
        let _ = tokio::fs::remove_file(&aiff).await;
        anyhow::ensure!(ffmpeg_status.success(), "speech conversion failed");
        let size_bytes = tokio::fs::metadata(&wav).await?.len();
        if size_bytes == 0 || size_bytes > max_bytes as u64 {
            let _ = tokio::fs::remove_file(&wav).await;
            anyhow::bail!("synthesized response audio exceeds the configured limit");
        }
        Ok(Some(SpeechArtifact {
            path: wav,
            content_type: "audio/wav".to_string(),
            size_bytes,
        }))
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn macos_say_produces_bounded_canonical_stereo_wav() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = NodeSpeechConfig::default();
        config.enabled = true;
        let artifact = MacOsSaySpeech::new(config)
            .synthesize("Hi Bella.", directory.path(), 512 * 1024)
            .await
            .unwrap()
            .unwrap();
        let bytes = std::fs::read(&artifact.path).unwrap();
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[36..40], b"data");
        assert!(bytes.len() < 512 * 1024);
    }
}
