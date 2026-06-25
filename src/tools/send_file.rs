use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::tools::file_delivery::{prepare_delivery, DeliveryError};
use crate::traits::{Tool, ToolCallSemantics, ToolCapabilities, ToolTargetHintKind};
use crate::types::{MediaKind, MediaMessage};

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

        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        let ready = match prepare_delivery(file_path, &cwd, &self.inbox_dir, &self.outbox_dirs) {
            Ok(r) => r,
            Err(DeliveryError::FileNotFound(_)) => {
                return Ok(format!("Error: File not found: {}", file_path));
            }
            Err(DeliveryError::Ambiguous(candidates)) => {
                let names = candidates
                    .iter()
                    .take(3)
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Ok(format!(
                    "Error: File not found: {}. Found multiple files with this name in allowed locations: {}",
                    file_path, names
                ));
            }
            Err(DeliveryError::NotRegularFile(_)) => {
                return Ok(format!("Error: Not a regular file: {}", file_path));
            }
            Err(DeliveryError::Blocked(_)) => {
                return Ok(format!(
                    "Error: Sending this file is blocked for security reasons: {}",
                    file_path
                ));
            }
            Err(DeliveryError::OutsideAllowedDirs(_)) => {
                return Ok(format!(
                    "Error: File is outside allowed directories: {}. Only files in the allowed \
                     output directories or a system temp dir can be sent. Move the file into {} \
                     and send that path instead.",
                    file_path,
                    self.inbox_dir.display(),
                ));
            }
            Err(DeliveryError::RecoveryFailed { path, error }) => {
                return Ok(format!(
                    "Error: File is outside allowed directories ({}). I tried to copy it into \
                     the allowed inbox directory {} but that failed: {}. Copy the file into {} \
                     (e.g. with the terminal tool: cp '{}' '{}/') and send that path instead.",
                    file_path,
                    self.inbox_dir.display(),
                    error,
                    self.inbox_dir.display(),
                    path.display(),
                    self.inbox_dir.display(),
                ));
            }
        };

        let size_display = if ready.size_bytes > 1_048_576 {
            format!("{:.1} MB", ready.size_bytes as f64 / 1_048_576.0)
        } else {
            format!("{:.0} KB", ready.size_bytes as f64 / 1024.0)
        };

        self.media_tx
            .send(MediaMessage {
                session_id: session_id.to_string(),
                caption: caption.to_string(),
                kind: MediaKind::Document {
                    file_path: ready.canonical_path.to_string_lossy().to_string(),
                    filename: ready.filename.clone(),
                },
                // Fire-and-forget: send_file does not await a delivery receipt.
                result_tx: None,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send file: {}", e))?;

        // Determine success message: resolved_missing_path is detected by comparing
        // the canonical path with a fresh expansion of the requested path.
        let expanded_requested = shellexpand::tilde(file_path).to_string();
        let resolved_missing_path = !PathBuf::from(&expanded_requested).exists()
            || PathBuf::from(&expanded_requested)
                .canonicalize()
                .map(|c| c != ready.canonical_path)
                .unwrap_or(false);

        if ready.recovered_into_inbox {
            Ok(format!(
                "File sent: {} ({}) [copied into the inbox for delivery]",
                ready.filename, size_display
            ))
        } else if resolved_missing_path {
            Ok(format!(
                "File sent: {} ({}) [resolved missing path to {}]",
                ready.filename,
                size_display,
                ready.canonical_path.display()
            ))
        } else {
            Ok(format!("File sent: {} ({})", ready.filename, size_display))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::file_delivery::{
        is_recoverable_source, resolve_missing_path_by_filename, ResolveResult,
    };
    use std::path::Path;

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
        assert!(
            out.contains("copied into the inbox"),
            "should note recovery: {out}"
        );

        let msg = rx.try_recv().expect("media message sent");
        match msg.kind {
            MediaKind::Document { file_path, .. } => {
                assert!(
                    file_path.contains("inbox"),
                    "delivered from inbox: {file_path}"
                );
            }
            _ => panic!("expected Document media"),
        }
        assert!(
            inbox.join("latency_results.txt").exists(),
            "copy must land in inbox"
        );
    }

    #[test]
    fn is_recoverable_source_only_allows_temp_roots() {
        // A file in the system temp dir is recoverable...
        let tmp = tempfile::tempdir().expect("tempdir");
        let in_temp = tmp.path().join("out.txt");
        std::fs::write(&in_temp, b"x").expect("write");
        let in_temp = in_temp.canonicalize().expect("canon");
        assert!(is_recoverable_source(&in_temp));

        // ...but arbitrary paths outside temp are NOT (no arbitrary exfiltration).
        assert!(!is_recoverable_source(Path::new("/etc/hosts")));
        if let Some(home) = dirs::home_dir() {
            assert!(!is_recoverable_source(&home.join("Documents/secret.pdf")));
        }
    }

    #[tokio::test]
    async fn send_file_rejects_non_temp_outside_path_without_copying() {
        // A readable file outside both allowed dirs and temp roots must be
        // refused with an error and NEVER copied into the inbox.
        let home = match dirs::home_dir() {
            Some(h) => h,
            None => return, // can't run meaningfully without a home dir
        };
        let inbox = home.join(".aidaemon-test-inbox-reject");
        let _ = std::fs::remove_dir_all(&inbox);
        // Source: a real readable file outside temp (the binary's own Cargo.toml).
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tool = SendFileTool::new(tx, &[], &inbox.to_string_lossy());
        let args = json!({
            "_session_id": "s",
            "file_path": manifest.to_string_lossy(),
        })
        .to_string();
        let out = tool.call(&args).await.expect("call ok");
        assert!(out.contains("outside allowed directories"), "got: {out}");
        assert!(
            !inbox.join("Cargo.toml").exists(),
            "non-temp file must not be copied into the inbox"
        );
        let _ = std::fs::remove_dir_all(&inbox);
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

        let outboxes = vec![outbox.clone()];
        let inbox = tmp.path().join("inbox");

        let requested = Path::new("/tmp/testuser/report.pdf");
        let result = resolve_missing_path_by_filename(requested, tmp.path(), &inbox, &outboxes);
        let resolved = match result.expect("expected one match") {
            ResolveResult::Found(p) => p,
            ResolveResult::Ambiguous(_) => panic!("expected unique match"),
        };
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

        let outboxes = vec![outbox1, outbox2];
        let inbox = tmp.path().join("inbox");

        let requested = Path::new("/tmp/testuser/report.pdf");
        let result = resolve_missing_path_by_filename(requested, tmp.path(), &inbox, &outboxes);
        match result.expect("expected a result") {
            ResolveResult::Ambiguous(candidates) => {
                assert!(candidates.len() >= 2, "expected multiple candidates");
            }
            ResolveResult::Found(_) => panic!("expected ambiguity"),
        }
    }

    #[test]
    fn resolve_missing_path_by_filename_returns_none_without_matches() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let outbox = tmp.path().join("outbox");
        std::fs::create_dir_all(&outbox).expect("create outbox");

        let outboxes = vec![outbox];
        let inbox = tmp.path().join("inbox");

        let requested = Path::new("/tmp/testuser/report.pdf");
        let result = resolve_missing_path_by_filename(requested, tmp.path(), &inbox, &outboxes);
        assert!(result.is_none(), "expected no match");
    }
}
