//! `write_file`: create, overwrite or append.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use aidaemon_core::{AccessManifest, ToolAnnotations};

use super::{FilePolicy, SharedBackend, ToolText, path_arg, workspace_parts};
use crate::backend::WriteMode;
use crate::outcome::{OutcomeTool, core_tool};
use crate::semantics::{
    DurableReplayDecision, ToolCallContext, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolMutationEffects, ToolRole, ToolTargetHintKind,
};
use crate::workspace::Workspace;

const MAX_CONTENT_SIZE: usize = 1024 * 1024; // 1MB

pub struct WriteFile {
    backend: SharedBackend,
    policy: Arc<dyn FilePolicy>,
    text: ToolText,
}

impl WriteFile {
    /// A tool confined to `workspace`.
    pub fn new(workspace: Workspace) -> Self {
        let (backend, policy) = workspace_parts(workspace);
        Self::on_backend(backend, policy)
    }

    /// A tool working through `backend` under an application's `policy`.
    pub fn on_backend(backend: SharedBackend, policy: Arc<dyn FilePolicy>) -> Self {
        Self {
            backend,
            policy,
            text: ToolText {
                short: "Write or create a file with given content".into(),
                schema: "Create a file, overwrite one, or append to one. Parent directories are created. Prefer edit_file for changing part of an existing file.".into(),
            },
        }
    }

    /// Replace the model-facing descriptions.
    pub fn with_text(mut self, text: ToolText) -> Self {
        self.text = text;
        self
    }
}

#[async_trait]
impl OutcomeTool for WriteFile {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        &self.text.short
    }

    fn schema(&self) -> Value {
        json!({
            "name": "write_file",
            "description": self.text.schema,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to write to (supports ~ expansion)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create parent directories if they don't exist (default: true)"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "Write mode. \"overwrite\" (default) replaces the file (a backup is kept). \"append\" adds to the end of the file without a backup — use it to build a large file in chunks across multiple calls instead of emitting everything in one call (which can exceed the output token limit)."
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: Value = serde_json::from_str(arguments)?;
        // Parameter aliasing: models often use "file_path" or "file" instead of "path"
        let path_str = args["path"]
            .as_str()
            .or_else(|| args["file_path"].as_str())
            .or_else(|| args["file"].as_str())
            .or_else(|| args["filename"].as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;
        let content = args["content"]
            .as_str()
            .or_else(|| args["data"].as_str())
            .or_else(|| args["text"].as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: content"))?;
        let create_dirs = args["create_dirs"].as_bool().unwrap_or(true);

        if content.len() > MAX_CONTENT_SIZE {
            anyhow::bail!(
                "Content too large: {} bytes (max {})",
                content.len(),
                MAX_CONTENT_SIZE
            );
        }

        let backend = self.backend.clone();
        let path = backend.resolve_path(path_str).await?;

        // Create parent dirs if requested
        if create_dirs {
            if let Some(parent) = path.parent() {
                backend.create_dir_all(&parent).await?;
            }
        } else if let Some(parent) = path.parent()
            && !backend
                .metadata(&parent)
                .await
                .is_ok_and(|metadata| metadata.is_dir())
        {
            anyhow::bail!(
                "Parent directory does not exist: {}. Set create_dirs=true to create it.",
                parent
            );
        }

        // Write mode: "overwrite" (default, replaces with backup) or "append"
        // (additive, no backup — lets the model build a large file in chunks
        // instead of one oversized generation).
        let mode = args["mode"].as_str().unwrap_or("overwrite");
        if mode != "overwrite" && mode != "append" {
            anyhow::bail!(
                "Invalid mode: \"{}\" (expected \"overwrite\" or \"append\")",
                mode
            );
        }

        // Held until the write has landed.
        let write_guard = match crate::semantics::file_fence(facts) {
            Some(fence) => Some(
                fence
                    .begin_write(path_str)
                    .await
                    .map_err(|why| anyhow::anyhow!(why))?,
            ),
            None => None,
        };

        if mode == "append" {
            let existing_size = backend
                .metadata(&path)
                .await
                .map(|metadata| metadata.len)
                .unwrap_or(0);
            backend
                .write(&path, content.as_bytes(), WriteMode::Append, create_dirs)
                .await?;
            if let Some(write_guard) = write_guard {
                write_guard.committed().await;
            }

            let appended = content.len();
            let appended_lines = content.lines().count();
            let total = existing_size + appended as u64;
            let delivery = self
                .policy
                .delivery_hint()
                .map(|hint| format!(" {hint}"))
                .unwrap_or_default();
            return Ok(ToolCallOutcome::from_output(format!(
                "Appended {} bytes ({} lines) to {} (now {} bytes total). \
                 To add more, call write_file again with mode=\"append\".{}",
                appended, appended_lines, path_str, total, delivery
            )));
        }

        // Backup existing file
        let existing_metadata = backend.metadata(&path).await.ok();
        let existed = existing_metadata.is_some();
        let old_size = if existed {
            let size = existing_metadata
                .as_ref()
                .map_or(0, |metadata| metadata.len);
            // One backup per file, outside the user's tree when local.
            self.policy.backup_before_overwrite(&backend, &path).await;
            Some(size)
        } else {
            None
        };

        backend
            .write(&path, content.as_bytes(), WriteMode::Overwrite, create_dirs)
            .await?;
        if let Some(write_guard) = write_guard {
            write_guard.committed().await;
        }

        let new_size = content.len();
        let line_count = content.lines().count();

        let action = if existed { "Updated" } else { "Created" };
        let size_info = if let Some(old) = old_size {
            let backup = if self.policy.keeps_backups() {
                ", backup saved"
            } else {
                ""
            };
            format!(" (was {} bytes{})", old, backup)
        } else {
            String::new()
        };

        let diagnostics = self
            .policy
            .post_write_diagnostics(&*backend, std::path::Path::new(path.as_str()))
            .await;

        Ok(ToolCallOutcome::from_output(format!(
            "{} {}\n{} bytes, {} lines{}{}",
            action, path_str, new_size, line_count, size_info, diagnostics
        )))
    }
}

core_tool!(
    WriteFile,
    // Append mode writes more each time, so a repeat is not a no-op.
    annotations: ToolAnnotations::WRITE,
    access: |this, arguments| AccessManifest {
        writes: path_arg(arguments)
            .map(|path| vec![this.policy.declare_path(path)])
            .unwrap_or_default(),
        ..Default::default()
    }
);

#[async_trait::async_trait]
impl crate::semantics::ToolSemantics for WriteFile {
    fn dispatch_facts(&self) -> crate::semantics::ToolDispatchFacts {
        crate::semantics::ToolDispatchFacts {
            offering: crate::semantics::Offering::Core,
            checkpoints_writes: true,
            file_guard: crate::semantics::FileGuard::Writes,
            ..Default::default()
        }
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            // Not high-impact: creates backups before overwriting, user-scoped
            // file system only, and dedicated file tools are intentionally
            // available without terminal approval. Sensitive paths are blocked
            // before writing.
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let path = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|args| {
                for key in ["path", "file_path", "file", "filename"] {
                    if let Some(path) = args.get(key).and_then(|value| value.as_str()) {
                        return Some(path.to_string());
                    }
                }
                None
            })
            .unwrap_or_default();

        ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE)
            .with_target_hint(ToolTargetHintKind::Path, path)
    }

    fn call_file_write(&self, arguments: &str) -> Option<crate::semantics::FileWrite> {
        let args: Value = serde_json::from_str(arguments).ok()?;
        Some(crate::semantics::FileWrite {
            path: path_arg(&args)?.to_string(),
            kind: if args["mode"].as_str() == Some("append") {
                crate::semantics::FileWriteKind::Append
            } else {
                crate::semantics::FileWriteKind::Replace
            },
        })
    }

    fn file_write_result(
        &self,
        arguments: &str,
        current: Option<&[u8]>,
    ) -> Option<Result<Vec<u8>, String>> {
        let args: Value = serde_json::from_str(arguments).ok()?;
        let content = ["content", "data", "text"]
            .iter()
            .find_map(|key| args.get(*key).and_then(Value::as_str))?;
        let mut result = match args["mode"].as_str() {
            Some("append") => current.unwrap_or_default().to_vec(),
            Some("overwrite") | None => Vec::new(),
            Some(_) => return None,
        };
        result.extend_from_slice(content.as_bytes());
        Some(Ok(result))
    }

    /// A write of the target path (absolute; a relative path resolves where
    /// the execution backend puts it, which is not known here).
    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        self.policy.write_effects(&*self.backend, arguments)
    }

    fn call_annotations_in(
        &self,
        arguments: &str,
        _roots: &crate::semantics::TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        let semantics = self.call_semantics(arguments);
        let mut annotations =
            crate::semantics::default_call_annotations(self.capabilities(), &semantics);
        // A write that can plant git configuration changes what later
        // "read-only" git commands execute: treat it as a configuration change.
        let plants_git_config = semantics
            .target_hints
            .iter()
            .any(|hint| self.policy.plants_configuration(&hint.value));
        annotations.egress |= plants_git_config;
        annotations
    }

    async fn durable_replay_decision(&self, arguments: &str) -> DurableReplayDecision {
        let Ok(args) = serde_json::from_str::<Value>(arguments) else {
            return DurableReplayDecision::Replay;
        };
        // Appending the same bytes twice is not safe. Keep the durable receipt
        // authoritative for append calls; callers must issue a new operation.
        if args["mode"].as_str().unwrap_or("overwrite") != "overwrite" {
            return DurableReplayDecision::Replay;
        }
        let Some(path_str) = args["path"]
            .as_str()
            .or_else(|| args["file_path"].as_str())
            .or_else(|| args["file"].as_str())
            .or_else(|| args["filename"].as_str())
        else {
            return DurableReplayDecision::Replay;
        };
        let Some(content) = args["content"]
            .as_str()
            .or_else(|| args["data"].as_str())
            .or_else(|| args["text"].as_str())
        else {
            return DurableReplayDecision::Replay;
        };

        let backend = self.backend.clone();
        let path = match backend.resolve_path(path_str).await {
            Ok(path) => path,
            Err(error) => {
                return DurableReplayDecision::Block {
                    reason: format!(
                        "the prior file-write target can no longer be resolved safely: {error}"
                    ),
                };
            }
        };
        match backend.read(&path).await {
            Ok(current) if current == content.as_bytes() => DurableReplayDecision::Replay,
            Ok(_) => DurableReplayDecision::Block {
                reason: format!(
                    "the file at {path_str} exists but no longer matches the successful write receipt; inspect and reconcile the current content before overwriting it"
                ),
            },
            Err(read_error) => {
                let Some(parent) = path.parent() else {
                    return DurableReplayDecision::Block {
                        reason: format!(
                            "the prior file-write target could not be inspected: {read_error}"
                        ),
                    };
                };
                match backend.read_dir(&parent).await {
                    Ok(entries) if entries.iter().all(|entry| entry.path != path) => {
                        DurableReplayDecision::Reexecute {
                            reason: format!(
                                "the exact overwrite target {path_str} is absent, so the prior successful receipt no longer satisfies current state"
                            ),
                        }
                    }
                    Ok(_) => DurableReplayDecision::Block {
                        reason: format!(
                            "the file at {path_str} still exists but could not be read; refusing to overwrite unreconciled current state"
                        ),
                    },
                    Err(parent_error) => DurableReplayDecision::Block {
                        reason: format!(
                            "the prior file-write target and its parent could not be inspected safely: {read_error}; {parent_error}"
                        ),
                    },
                }
            }
        }
    }
}
