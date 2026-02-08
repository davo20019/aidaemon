use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::traits::Tool;
use crate::types::{MediaKind, MediaMessage};

/// Blocked path patterns for outbound file sends (security).
const BLOCKED_PATTERNS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".env",
    "credentials",
    ".key",
    ".pem",
    ".aws/credentials",
    ".netrc",
    ".docker/config.json",
    "config.toml",
];

pub struct SendFileTool {
    media_tx: mpsc::Sender<MediaMessage>,
    outbox_dirs: Vec<PathBuf>,
    inbox_dir: PathBuf,
}

impl SendFileTool {
    pub fn new(
        media_tx: mpsc::Sender<MediaMessage>,
        outbox_dirs: &[String],
        inbox_dir: &str,
    ) -> Self {
        let outbox_dirs: Vec<PathBuf> = outbox_dirs
            .iter()
            .map(|d| {
                let expanded = shellexpand::tilde(d).to_string();
                PathBuf::from(expanded)
            })
            .collect();
        let inbox_dir = PathBuf::from(shellexpand::tilde(inbox_dir).to_string());
        Self {
            media_tx,
            outbox_dirs,
            inbox_dir,
        }
    }

    fn is_path_allowed(&self, canonical: &Path) -> bool {
        // Allow files in inbox dir (agent returning processed files)
        if canonical.starts_with(&self.inbox_dir) {
            return true;
        }
        // Check against allowed outbox dirs
        self.outbox_dirs.iter().any(|d| canonical.starts_with(d))
    }

    fn is_path_blocked(path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        for pattern in BLOCKED_PATTERNS {
            if pattern.starts_with('.') || pattern.starts_with('/') {
                // Component-based check: .ssh, .gnupg, .env, .aws/credentials, etc.
                if path_str.contains(&format!("/{}", pattern))
                    || path_str.contains(&format!("/{}/", pattern))
                {
                    return true;
                }
            } else if pattern.starts_with("*.") {
                // Extension-based check (not used currently but future-proof)
                let ext = &pattern[1..]; // ".key", ".pem"
                if path_str.ends_with(ext) {
                    return true;
                }
            } else {
                // Exact filename check
                if let Some(name) = path.file_name() {
                    if name.to_string_lossy() == *pattern {
                        return true;
                    }
                }
                // Also check as path component
                if path_str.contains(&format!("/{}", pattern))
                    || path_str.contains(&format!("/{}/", pattern))
                {
                    return true;
                }
            }
        }
        // Also block files ending with .key or .pem
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy();
            if ext == "key" || ext == "pem" {
                return true;
            }
        }
        false
    }
}

#[async_trait]
impl Tool for SendFileTool {
    fn name(&self) -> &str {
        "send_file"
    }

    fn description(&self) -> &str {
        "Send a file to the user in the current chat. ALWAYS use this tool when the user asks you to send, share, or deliver a file. Validates the path is within allowed directories and not a sensitive file."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "send_file",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to send"
                    },
                    "caption": {
                        "type": "string",
                        "description": "Optional caption for the file"
                    }
                },
                "required": ["file_path"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;

        let file_path = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: file_path"))?;

        let caption = args.get("caption").and_then(|v| v.as_str()).unwrap_or("");

        let session_id = args
            .get("_session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Expand ~ in the path
        let expanded = shellexpand::tilde(file_path).to_string();
        let path = Path::new(&expanded);

        // Check file exists
        if !path.exists() {
            return Ok(format!("Error: File not found: {}", file_path));
        }

        // Must be a regular file
        let metadata = std::fs::metadata(path)?;
        if !metadata.is_file() {
            return Ok(format!("Error: Not a regular file: {}", file_path));
        }

        // Canonicalize to resolve symlinks and prevent traversal
        let canonical = path.canonicalize()?;

        // Check against allowed directories
        if !self.is_path_allowed(&canonical) {
            return Ok(format!(
                "Error: File is outside allowed directories. Path: {}",
                file_path
            ));
        }

        // Check against blocked patterns
        if Self::is_path_blocked(&canonical) {
            return Ok(format!(
                "Error: Sending this file is blocked for security reasons: {}",
                file_path
            ));
        }

        // Extract filename for display
        let filename = canonical
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "file".to_string());

        let file_size = metadata.len();
        let size_display = if file_size > 1_048_576 {
            format!("{:.1} MB", file_size as f64 / 1_048_576.0)
        } else {
            format!("{:.0} KB", file_size as f64 / 1024.0)
        };

        self.media_tx
            .send(MediaMessage {
                session_id: session_id.to_string(),
                caption: caption.to_string(),
                kind: MediaKind::Document {
                    file_path: canonical.to_string_lossy().to_string(),
                    filename: filename.clone(),
                },
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send file: {}", e))?;

        Ok(format!("File sent: {} ({})", filename, size_display))
    }
}
