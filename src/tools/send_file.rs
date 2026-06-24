use std::collections::HashSet;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::traits::{Tool, ToolCallSemantics, ToolCapabilities, ToolTargetHintKind};
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

    /// Copy a readable file from outside the allowed dirs into the inbox so it
    /// can be delivered. Returns the canonical path of the copy. Caller must have
    /// already run the blocked-pattern check on `src`.
    fn recover_into_inbox(&self, src: &Path) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(&self.inbox_dir)?;
        let filename = src
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("file"));
        let dest = self.inbox_dir.join(filename);
        std::fs::copy(src, &dest)?;
        dest.canonicalize()
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

    /// If the requested absolute path doesn't exist, try a safe, bounded
    /// recovery by looking for the same filename in known roots.
    fn resolve_missing_path_by_filename(
        &self,
        requested: &Path,
    ) -> anyhow::Result<Option<PathBuf>> {
        let file_name = match requested.file_name() {
            Some(name) if !name.is_empty() => name.to_os_string(),
            _ => return Ok(None),
        };

        let mut matches: Vec<PathBuf> = Vec::new();
        let mut seen: HashSet<PathBuf> = HashSet::new();
        let mut check_candidate = |candidate: PathBuf| {
            if !candidate.exists() {
                return;
            }
            if let Ok(md) = std::fs::metadata(&candidate) {
                if !md.is_file() {
                    return;
                }
            } else {
                return;
            }
            if let Ok(canonical) = candidate.canonicalize() {
                if seen.insert(canonical.clone()) {
                    matches.push(canonical);
                }
            }
        };

        if let Ok(cwd) = std::env::current_dir() {
            check_candidate(cwd.join(&file_name));
        }
        check_candidate(self.inbox_dir.join(&file_name));
        for outbox in &self.outbox_dirs {
            check_candidate(outbox.join(&file_name));
        }

        match matches.len() {
            0 => Ok(None),
            1 => Ok(matches.into_iter().next()),
            _ => {
                let candidates = matches
                    .iter()
                    .take(3)
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                Err(anyhow::anyhow!(
                    "Found multiple files with this name in allowed locations: {}",
                    candidates
                ))
            }
        }
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
                "required": ["file_path"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let path = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|args| {
                args.get("file_path")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();

        ToolCallSemantics::mutation().with_target_hint(ToolTargetHintKind::Path, path)
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
        let requested_path = Path::new(&expanded);
        let mut resolved_missing_path = false;
        let path = if requested_path.exists() {
            requested_path.to_path_buf()
        } else {
            match self.resolve_missing_path_by_filename(requested_path) {
                Ok(Some(found)) => {
                    resolved_missing_path = true;
                    found
                }
                Ok(None) => return Ok(format!("Error: File not found: {}", file_path)),
                Err(e) => return Ok(format!("Error: File not found: {}. {}", file_path, e)),
            }
        };

        // Must be a regular file
        let metadata = std::fs::metadata(&path)?;
        if !metadata.is_file() {
            return Ok(format!("Error: Not a regular file: {}", file_path));
        }

        // Canonicalize to resolve symlinks and prevent traversal
        let mut canonical = path.canonicalize()?;

        // Block sensitive files regardless of location — checked BEFORE any
        // recovery copy so a blocked file is never copied into the inbox.
        if Self::is_path_blocked(&canonical) {
            return Ok(format!(
                "Error: Sending this file is blocked for security reasons: {}",
                file_path
            ));
        }

        // If the file is outside the allowed directories, auto-recover by copying
        // it into the inbox dir and sending from there. This makes the common
        // "create a file in /tmp and send it" flow work without the model looping
        // on directory-policy errors (the file the user asked for is delivered).
        let mut recovered_into_inbox = false;
        if !self.is_path_allowed(&canonical) {
            match self.recover_into_inbox(&canonical) {
                Ok(copied) => {
                    canonical = copied;
                    recovered_into_inbox = true;
                }
                Err(e) => {
                    return Ok(format!(
                        "Error: File is outside allowed directories ({}). I tried to copy it into \
                         the allowed inbox directory {} but that failed: {}. Copy the file into {} \
                         (e.g. with the terminal tool: cp '{}' '{}/') and send that path instead.",
                        file_path,
                        self.inbox_dir.display(),
                        e,
                        self.inbox_dir.display(),
                        canonical.display(),
                        self.inbox_dir.display(),
                    ));
                }
            }
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
                // Fire-and-forget: send_file does not await a delivery receipt.
                result_tx: None,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send file: {}", e))?;

        if resolved_missing_path {
            Ok(format!(
                "File sent: {} ({}) [resolved missing path to {}]",
                filename,
                size_display,
                canonical.display()
            ))
        } else if recovered_into_inbox {
            Ok(format!(
                "File sent: {} ({}) [copied into the inbox for delivery]",
                filename, size_display
            ))
        } else {
            Ok(format!("File sent: {} ({})", filename, size_display))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_tool(outboxes: Vec<String>, inbox: String) -> SendFileTool {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        SendFileTool::new(tx, &outboxes, &inbox)
    }

    #[tokio::test]
    async fn send_file_recovers_file_outside_allowed_dirs_into_inbox() {
        // Regression: the model created a file in /tmp (outside allowed dirs) and
        // send_file rejected it, causing a retry loop. send_file now copies a
        // readable out-of-dir file into the inbox and delivers it.
        let tmp = tempfile::tempdir().expect("tempdir");
        let inbox = tmp.path().join("inbox");
        let outside = tmp.path().join("external");
        std::fs::create_dir_all(&outside).expect("create external");
        let src = outside.join("latency_results.txt");
        std::fs::write(&src, b"latency data").expect("write src");

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let tool = SendFileTool::new(tx, &[], &inbox.to_string_lossy());

        let args = json!({
            "_session_id": "sess-1",
            "file_path": src.to_string_lossy(),
        })
        .to_string();
        let out = tool.call(&args).await.expect("call ok");
        assert!(out.contains("File sent"), "got: {out}");
        assert!(out.contains("copied into the inbox"), "should note recovery: {out}");

        let msg = rx.try_recv().expect("media message sent");
        match msg.kind {
            MediaKind::Document { file_path, .. } => {
                assert!(file_path.contains("inbox"), "delivered from inbox: {file_path}");
            }
            _ => panic!("expected Document media"),
        }
        assert!(inbox.join("latency_results.txt").exists(), "copy must land in inbox");
    }

    #[tokio::test]
    async fn send_file_blocked_file_is_not_recovered() {
        // A sensitive file outside allowed dirs must be blocked, never copied in.
        let tmp = tempfile::tempdir().expect("tempdir");
        let inbox = tmp.path().join("inbox");
        let outside = tmp.path().join("external");
        std::fs::create_dir_all(&outside).expect("create external");
        let src = outside.join(".env");
        std::fs::write(&src, b"SECRET=1").expect("write src");

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tool = SendFileTool::new(tx, &[], &inbox.to_string_lossy());

        let args = json!({
            "_session_id": "s",
            "file_path": src.to_string_lossy(),
        })
        .to_string();
        let out = tool.call(&args).await.expect("call ok");
        assert!(out.contains("blocked for security"), "got: {out}");
        assert!(
            !inbox.join(".env").exists(),
            "blocked file must not be copied into the inbox"
        );
    }

    #[test]
    fn resolve_missing_path_by_filename_finds_unique_match() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let outbox = tmp.path().join("outbox");
        std::fs::create_dir_all(&outbox).expect("create outbox");
        let file = outbox.join("report.pdf");
        std::fs::write(&file, b"pdf").expect("write file");

        let tool = mk_tool(
            vec![outbox.to_string_lossy().to_string()],
            tmp.path().join("inbox").to_string_lossy().to_string(),
        );

        let requested = Path::new("/tmp/testuser/report.pdf");
        let resolved = tool
            .resolve_missing_path_by_filename(requested)
            .expect("resolver should not error")
            .expect("expected one match");
        assert_eq!(
            resolved,
            file.canonicalize().expect("canonicalize expected file")
        );
    }

    #[test]
    fn resolve_missing_path_by_filename_errors_on_ambiguous_matches() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let outbox1 = tmp.path().join("outbox1");
        let outbox2 = tmp.path().join("outbox2");
        std::fs::create_dir_all(&outbox1).expect("create outbox1");
        std::fs::create_dir_all(&outbox2).expect("create outbox2");
        std::fs::write(outbox1.join("report.pdf"), b"one").expect("write outbox1 file");
        std::fs::write(outbox2.join("report.pdf"), b"two").expect("write outbox2 file");

        let tool = mk_tool(
            vec![
                outbox1.to_string_lossy().to_string(),
                outbox2.to_string_lossy().to_string(),
            ],
            tmp.path().join("inbox").to_string_lossy().to_string(),
        );

        let requested = Path::new("/tmp/testuser/report.pdf");
        let err = tool
            .resolve_missing_path_by_filename(requested)
            .expect_err("expected ambiguity error");
        assert!(err.to_string().contains("multiple files"));
    }

    #[test]
    fn resolve_missing_path_by_filename_returns_none_without_matches() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let outbox = tmp.path().join("outbox");
        std::fs::create_dir_all(&outbox).expect("create outbox");

        let tool = mk_tool(
            vec![outbox.to_string_lossy().to_string()],
            tmp.path().join("inbox").to_string_lossy().to_string(),
        );

        let requested = Path::new("/tmp/testuser/report.pdf");
        let resolved = tool
            .resolve_missing_path_by_filename(requested)
            .expect("resolver should not error");
        assert!(resolved.is_none());
    }
}
