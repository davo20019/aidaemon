//! Shared Whisper STT discovery and config helpers for wizard and manage_config.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Result of probing the local Whisper STT stack.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SttProbe {
    pub whisper_cli: Option<PathBuf>,
    pub ffmpeg: Option<PathBuf>,
    pub model_path: Option<PathBuf>,
}

impl SttProbe {
    pub fn is_complete(&self) -> bool {
        self.whisper_cli.is_some() && self.ffmpeg.is_some() && self.model_path.is_some()
    }

    /// Probe with optional path overrides (used by manage_config).
    /// An explicit override that does not exist clears that component (no auto fallback).
    pub fn resolve(
        cli_override: Option<&str>,
        model_override: Option<&str>,
        ffmpeg_override: Option<&str>,
    ) -> Self {
        let auto = probe_stt_environment();
        SttProbe {
            whisper_cli: resolve_cli_path(cli_override, auto.whisper_cli),
            ffmpeg: resolve_ffmpeg_path(ffmpeg_override, auto.ffmpeg),
            model_path: resolve_model_path(model_override, auto.model_path),
        }
    }
}

pub fn probe_stt_environment() -> SttProbe {
    SttProbe {
        whisper_cli: discover_whisper_cli(),
        ffmpeg: discover_ffmpeg(),
        model_path: discover_whisper_model(),
    }
}

pub fn discover_whisper_cli() -> Option<PathBuf> {
    const CANDIDATES: &[&str] = &[
        "/opt/homebrew/bin/whisper-cli",
        "/usr/local/bin/whisper-cli",
    ];
    for candidate in CANDIDATES {
        let path = PathBuf::from(candidate);
        if is_executable(&path) {
            return Some(path);
        }
    }
    which_sync("whisper-cli")
}

pub fn discover_ffmpeg() -> Option<PathBuf> {
    const CANDIDATES: &[&str] = &["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"];
    for candidate in CANDIDATES {
        let path = PathBuf::from(candidate);
        if is_executable(&path) {
            return Some(path);
        }
    }
    which_sync("ffmpeg")
}

pub fn discover_whisper_model() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let candidates = [
        home.join("models/whisper/ggml-medium.en.bin"),
        home.join("models/whisper/ggml-medium.bin"),
        home.join("models/whisper/ggml-base.en.bin"),
        home.join("models/whisper/ggml-small.en.bin"),
        home.join("models/whisper/ggml-large-v3.bin"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

/// Human-readable probe report for CLI / tool responses.
pub fn format_stt_probe_report(probe: &SttProbe) -> String {
    let mut lines = vec!["Whisper STT probe:".to_string()];
    lines.push(format!(
        "- whisper-cli: {}",
        probe
            .whisper_cli
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "not found".to_string())
    ));
    lines.push(format!(
        "- ffmpeg: {}",
        probe
            .ffmpeg
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "not found".to_string())
    ));
    lines.push(format!(
        "- model: {}",
        probe
            .model_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "not found".to_string())
    ));
    lines.join("\n")
}

/// TOML fragment for wizard-generated config (`[files]` + `[files.stt]`).
pub fn stt_config_section(probe: &SttProbe, enabled: bool, language: &str) -> String {
    if !enabled {
        return "\n# Uncomment for Whisper voice-note transcription (fallback when native audio is unavailable):\n# [files]\n# enabled = true\n# [files.stt]\n# enabled = true\n# cli_path = \"/opt/homebrew/bin/whisper-cli\"\n# model_path = \"~/models/whisper/ggml-medium.en.bin\"\n# ffmpeg_path = \"ffmpeg\"\n# language = \"en\"\n".to_string();
    }

    let cli = path_for_toml(
        probe.whisper_cli.as_deref(),
        "/opt/homebrew/bin/whisper-cli",
    );
    let model = path_for_toml(
        probe.model_path.as_deref(),
        "~/models/whisper/ggml-medium.en.bin",
    );
    let ffmpeg = probe
        .ffmpeg
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "ffmpeg".to_string());

    format!(
        r#"
[files]
enabled = true

[files.stt]
enabled = true
cli_path = "{cli}"
model_path = "{model}"
ffmpeg_path = "{ffmpeg}"
language = "{language}"
max_audio_mb = 25
timeout_secs = 120
"#
    )
}

pub fn apply_stt_to_config_table(
    doc: &mut toml::Table,
    probe: &SttProbe,
    enabled: bool,
    language: &str,
) -> anyhow::Result<()> {
    let files = doc
        .entry("files")
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    let files_table = files
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("'files' is not a table"))?;

    let cli = path_for_toml(
        probe.whisper_cli.as_deref(),
        "/opt/homebrew/bin/whisper-cli",
    );
    let model = path_for_toml(
        probe.model_path.as_deref(),
        "~/models/whisper/ggml-medium.en.bin",
    );
    let ffmpeg = probe
        .ffmpeg
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "ffmpeg".to_string());

    let mut stt = toml::Table::new();
    stt.insert("enabled".to_string(), toml::Value::Boolean(enabled));
    stt.insert("cli_path".to_string(), toml::Value::String(cli));
    stt.insert("model_path".to_string(), toml::Value::String(model));
    stt.insert("ffmpeg_path".to_string(), toml::Value::String(ffmpeg));
    stt.insert(
        "language".to_string(),
        toml::Value::String(language.to_string()),
    );
    stt.insert("max_audio_mb".to_string(), toml::Value::Integer(25));
    stt.insert("timeout_secs".to_string(), toml::Value::Integer(120));

    files_table.insert("stt".to_string(), toml::Value::Table(stt));
    Ok(())
}

fn path_for_toml(found: Option<&Path>, fallback: &str) -> String {
    found
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| fallback.to_string())
}

fn resolve_cli_path(override_path: Option<&str>, auto: Option<PathBuf>) -> Option<PathBuf> {
    match override_path.filter(|s| !s.is_empty()) {
        Some(path) => {
            let expanded = expand_tilde(path);
            is_executable(&expanded).then_some(expanded)
        }
        None => auto,
    }
}

fn resolve_model_path(override_path: Option<&str>, auto: Option<PathBuf>) -> Option<PathBuf> {
    match override_path.filter(|s| !s.is_empty()) {
        Some(path) => {
            let expanded = expand_tilde(path);
            expanded.is_file().then_some(expanded)
        }
        None => auto,
    }
}

fn resolve_ffmpeg_path(override_path: Option<&str>, auto: Option<PathBuf>) -> Option<PathBuf> {
    match override_path.filter(|s| !s.is_empty()) {
        Some(path) => {
            let expanded = expand_tilde(path);
            if is_executable(&expanded) {
                Some(expanded)
            } else {
                which_sync(path)
            }
        }
        None => auto,
    }
}

fn expand_tilde(path: &str) -> PathBuf {
    PathBuf::from(shellexpand::tilde(path).into_owned())
}

fn is_executable(path: &Path) -> bool {
    path.is_file()
}

fn which_sync(binary: &str) -> Option<PathBuf> {
    Command::new("sh")
        .args(["-c", &format!("command -v {binary}")])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| PathBuf::from(s.trim()))
        .filter(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stt_config_section_disabled_is_commented() {
        let section = stt_config_section(&SttProbe::default(), false, "en");
        assert!(section.contains("# [files.stt]"));
        assert!(!section.contains("enabled = true\ncli_path"));
    }

    #[test]
    fn stt_config_section_enabled_includes_paths() {
        let probe = SttProbe {
            whisper_cli: Some(PathBuf::from("/opt/homebrew/bin/whisper-cli")),
            ffmpeg: Some(PathBuf::from("/opt/homebrew/bin/ffmpeg")),
            model_path: Some(PathBuf::from("/Users/me/models/whisper/ggml-medium.en.bin")),
        };
        let section = stt_config_section(&probe, true, "en");
        assert!(section.contains("[files.stt]"));
        assert!(section.contains("enabled = true"));
        assert!(section.contains("whisper-cli"));
        assert!(section.contains("ggml-medium.en.bin"));
        assert!(section.contains("language = \"en\""));
    }

    #[test]
    fn apply_stt_to_config_table_merges_under_files() {
        let mut doc: toml::Table = toml::Table::new();
        let probe = SttProbe {
            whisper_cli: Some(PathBuf::from("/bin/whisper-cli")),
            ffmpeg: Some(PathBuf::from("/bin/ffmpeg")),
            model_path: Some(PathBuf::from("/models/ggml-medium.en.bin")),
        };
        apply_stt_to_config_table(&mut doc, &probe, true, "en").unwrap();
        let files = doc.get("files").and_then(toml::Value::as_table).unwrap();
        let stt = files.get("stt").and_then(toml::Value::as_table).unwrap();
        assert_eq!(
            stt.get("enabled").and_then(toml::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            stt.get("cli_path").and_then(toml::Value::as_str),
            Some("/bin/whisper-cli")
        );
    }

    #[test]
    fn resolve_honors_overrides() {
        let dir = tempfile::tempdir().unwrap();
        let cli = dir.path().join("whisper-cli");
        std::fs::write(&cli, b"").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let model = dir.path().join("model.bin");
        std::fs::write(&model, b"fake").unwrap();

        let probe = SttProbe::resolve(
            Some(cli.to_str().unwrap()),
            Some(model.to_str().unwrap()),
            None,
        );
        assert_eq!(probe.whisper_cli, Some(cli));
        assert_eq!(probe.model_path, Some(model));
    }
}
