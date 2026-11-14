//! What first-run setup has left, kept beside the config file
//! (`setup.json`) from the moment a model is connected until the owner has
//! finished or skipped the chat step. An install with no such file is set
//! up.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Progress {
    /// The chat step is done (finished or skipped).
    #[serde(default)]
    pub channel_done: bool,
    /// A bot connected in the chat step: its platform and name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot: Option<(String, String)>,
    /// The first conversation's greeting was posted.
    #[serde(default)]
    pub greeted: bool,
    /// Set up from another device: the daemon runs on a server.
    #[serde(default)]
    pub on_server: bool,
    /// The model connected, in plain words.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
}

pub(crate) fn path(config_path: &Path) -> PathBuf {
    config_path.with_file_name("setup.json")
}

/// What is left; `None` when nothing is (or the install predates this).
pub(crate) fn load(config_path: &Path) -> Option<Progress> {
    let text = std::fs::read_to_string(path(config_path)).ok()?;
    let progress: Progress = serde_json::from_str(&text).ok()?;
    (!progress.channel_done || !progress.greeted).then_some(progress)
}

pub(crate) fn save(config_path: &Path, progress: &Progress) -> anyhow::Result<()> {
    let path = path(config_path);
    std::fs::write(&path, serde_json::to_string_pretty(progress)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_install_without_the_file_is_set_up_and_a_finished_one_too() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        assert_eq!(load(&config), None);
        let mut progress = Progress {
            model: "Synthetic model".into(),
            ..Progress::default()
        };
        save(&config, &progress).unwrap();
        assert_eq!(load(&config), Some(progress.clone()));
        progress.channel_done = true;
        progress.greeted = true;
        save(&config, &progress).unwrap();
        assert_eq!(load(&config), None);
    }
}
