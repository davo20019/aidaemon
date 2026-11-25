//! How a turn's messages are rendered for the model: the mode (the current
//! turn in full, earlier ones archived or adjacent) and the options, plus the
//! render cache. The renderer itself is `turn::turn_render`; these sit below it
//! so the media and turn-runtime modules can name them.

use aidaemon_events::TerminalState;

use crate::settings::{AudioConfig, SttConfig, VisionConfig};

pub mod cache;

#[derive(Clone, Debug)]
pub struct RenderOptions {
    pub vision: VisionConfig,
    pub audio: AudioConfig,
    pub stt: SttConfig,
    pub model: String,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            vision: VisionConfig {
                enabled: true,
                max_image_bytes: 4 * 1_048_576,
                mime_types: vec![
                    "image/jpeg".to_string(),
                    "image/png".to_string(),
                    "image/gif".to_string(),
                    "image/webp".to_string(),
                ],
                model_patterns: vec!["gpt-4o".to_string()],
            },
            audio: AudioConfig::from_files(&crate::settings::FilesConfig::default()),
            stt: SttConfig::from_files(&crate::settings::FilesConfig::default()),
            model: String::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    Current,
    Archived { terminal_state: TerminalState },
    Adjacent { terminal_state: TerminalState },
}
