use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::events::{
    Event, EventStore, EventType, ResourceInvalidatedData, ResourceRegisteredData,
};
use crate::execution::{active_execution_backend, BackendKind};
use crate::tools::file_delivery::{prepare_delivery, DeliveryError};
use crate::traits::{
    Tool, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolMutationEffects, ToolOutcomeStatus, ToolTargetHintKind,
};
use crate::types::{MediaKind, MediaMessage};

pub struct SendFileTool {
    media_tx: mpsc::Sender<MediaMessage>,
    outbox_dirs: Vec<PathBuf>,
    inbox_dir: PathBuf,
    event_store: Option<Arc<EventStore>>,
    /// How long to wait for the media listener's delivery receipt before
    /// reporting the file as queued-but-pending. Long enough to cover one
    /// full channel rate-limit retry sleep (capped at 60s) plus the upload.
    receipt_timeout: std::time::Duration,
}

const DELIVERY_RECEIPT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(70);

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
            event_store: None,
            receipt_timeout: DELIVERY_RECEIPT_TIMEOUT,
        }
    }

    pub fn with_event_store(mut self, event_store: Arc<EventStore>) -> Self {
        self.event_store = Some(event_store);
        self
    }

    async fn resolve_resource(
        &self,
        session_id: &str,
        resource_id: &str,
    ) -> Result<ResourceRegisteredData, String> {
        if session_id.is_empty() {
            return Err("resource_id requires the current session context".to_string());
        }
        let store = self
            .event_store
            .as_ref()
            .ok_or_else(|| "the resource registry is unavailable".to_string())?;
        let resource = store
            .get_resource(session_id, resource_id)
            .await
            .map_err(|error| format!("could not read the resource registry: {error}"))?
            .ok_or_else(|| {
                format!(
                    "resource {resource_id} is unknown, expired, or invalidated in this session"
                )
            })?;
        if resource.kind != "file" {
            return Err(format!(
                "resource {resource_id} is {}, not a deliverable file",
                resource.kind
            ));
        }
        Ok(resource)
    }

    async fn invalidate_resource(&self, session_id: &str, resource_id: &str, reason: &str) {
        let Some(store) = &self.event_store else {
            return;
        };
        let data = ResourceInvalidatedData {
            schema_version: ResourceInvalidatedData::SCHEMA_VERSION,
            resource_id: resource_id.to_string(),
            reason: reason.to_string(),
            turn_id: None,
        };
        if let Ok(value) = serde_json::to_value(data) {
            let _ = store
                .append(Event::new(
                    session_id,
                    EventType::ResourceInvalidated,
                    value,
                ))
                .await;
        }
    }

    #[cfg(test)]
    pub(crate) fn with_receipt_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.receipt_timeout = timeout;
        self
    }
}

#[async_trait]
impl Tool for SendFileTool {
    fn name(&self) -> &str {
        "send_file"
    }

    fn description(&self) -> &str {
        "Send an exact file resource to the user in the current chat. Prefer resource_id from an attachment or artifact result; otherwise provide an exact file_path. Never guess a path or substitute another file with the same name."
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
                        "description": "Exact absolute path to send. Use only when no resource_id is available."
                    },
                    "resource_id": {
                        "type": "string",
                        "description": "Exact opaque resource handle (res_...) from this conversation. Preferred for attachments and tool-created artifacts."
                    },
                    "caption": {
                        "type": "string",
                        "description": "Optional caption for the file"
                    }
                },
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

    fn stable_observation_subjects(&self) -> Vec<crate::traits::StableObservationSubject> {
        vec![crate::traits::StableObservationSubject::namespace(
            crate::channels::attachments::RESOURCE_ID_PREFIX,
            "session resource handles returned by tools",
        )]
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let args = serde_json::from_str::<Value>(arguments).ok();
        let resource_id = args
            .as_ref()
            .and_then(|args| args.get("resource_id"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let path = args
            .as_ref()
            .and_then(|args| args.get("file_path"))
            .and_then(Value::as_str)
            .unwrap_or_default();

        let semantics = ToolCallSemantics::mutation_with(ToolMutationEffects::EXTERNAL_DELIVERY);
        if resource_id.is_empty() {
            semantics.with_target_hint(ToolTargetHintKind::Path, path)
        } else {
            semantics.with_target_hint(ToolTargetHintKind::ResourceId, resource_id)
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;

        let caption = args.get("caption").and_then(|v| v.as_str()).unwrap_or("");

        let session_id = args
            .get("_session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let resource_id = args
            .get("resource_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let explicit_path = args
            .get("file_path")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if resource_id.is_some() && explicit_path.is_some() {
            return Ok(
                "Error: Provide exactly one of resource_id or file_path, not both.".to_string(),
            );
        }

        let registered_resource = if let Some(resource_id) = resource_id {
            match self.resolve_resource(session_id, resource_id).await {
                Ok(resource) => Some(resource),
                Err(reason) => return Ok(format!("Error: {reason}")),
            }
        } else {
            None
        };
        let file_path = registered_resource
            .as_ref()
            .map(|resource| resource.locator.as_str())
            .or(explicit_path);
        let Some(file_path) = file_path else {
            return Ok("Error: Missing resource_id or file_path.".to_string());
        };

        let backend = active_execution_backend();
        let mut requested_for_delivery = file_path.to_string();
        let mut exported_from_backend = false;
        if registered_resource.is_none() && backend.kind() != BackendKind::Local {
            let backend_path = match backend.resolve_path(file_path).await {
                Ok(path) => path,
                Err(error) => return Ok(format!("Error: Invalid execution path: {error}")),
            };
            let metadata = match backend.metadata(&backend_path).await {
                Ok(metadata) => metadata,
                Err(_) => return Ok(format!("Error: File not found: {file_path}")),
            };
            if !metadata.is_file() {
                return Ok(format!("Error: Not a regular file: {file_path}"));
            }
            let canonical = backend
                .canonicalize(&backend_path)
                .await
                .unwrap_or(backend_path);
            if crate::tools::file_delivery::is_path_blocked(std::path::Path::new(
                canonical.as_str(),
            )) {
                return Ok(format!(
                    "Error: Sending this file is blocked for security reasons: {file_path}"
                ));
            }
            let filename = canonical.file_name().unwrap_or("file");
            let local_staging = self.inbox_dir.join(filename);
            if let Err(error) = backend.export_local_file(&canonical, &local_staging).await {
                return Ok(format!(
                    "Error: Could not export {} from the {} execution backend for delivery: {}",
                    file_path,
                    backend.kind().as_str(),
                    error
                ));
            }
            requested_for_delivery = local_staging.to_string_lossy().into_owned();
            exported_from_backend = true;
        }

        if let Some(resource) = &registered_resource {
            let Some(expected_sha256) = resource.sha256.as_deref() else {
                return Ok(format!(
                    "Error: Resource {} has no integrity receipt and cannot be delivered safely.",
                    resource.resource_id
                ));
            };
            let actual_sha256 =
                crate::channels::attachments::sha256_file(Path::new(&requested_for_delivery));
            if actual_sha256.as_deref() != Some(expected_sha256) {
                self.invalidate_resource(
                    session_id,
                    &resource.resource_id,
                    "file content changed after registration",
                )
                .await;
                return Ok(format!(
                    "Error: Resource {} changed after it was registered and was invalidated. Recreate or reattach the file before sending it.",
                    resource.resource_id
                ));
            }
        }

        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let ready = match prepare_delivery(
            &requested_for_delivery,
            &cwd,
            &self.inbox_dir,
            &self.outbox_dirs,
        ) {
            Ok(r) => r,
            Err(DeliveryError::FileNotFound(_)) => {
                return Ok(format!("Error: File not found: {}", file_path));
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

        // Await the media listener's delivery receipt so "File sent" is a
        // statement of fact, not a queue acknowledgment. Without this the
        // model tells the user "I've sent it" while the document waits out a
        // channel rate-limit retry (observed live: ~60s gap), and a shed
        // delivery would leave the claim standing with no file at all.
        let (receipt_tx, receipt_rx) = tokio::sync::oneshot::channel();
        self.media_tx
            .send(MediaMessage {
                session_id: session_id.to_string(),
                caption: caption.to_string(),
                kind: MediaKind::Document {
                    file_path: ready.canonical_path.to_string_lossy().to_string(),
                    filename: ready.filename.clone(),
                },
                result_tx: Some(receipt_tx),
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send file: {}", e))?;

        match tokio::time::timeout(self.receipt_timeout, receipt_rx).await {
            Ok(Ok(Ok(()))) => {} // delivered — fall through to the success text
            Ok(Ok(Err(reason))) => {
                return Ok(format!(
                    "The file {} could not be delivered: {}. Do NOT tell the user it was sent; \
                     report the delivery failure instead.",
                    ready.filename, reason
                ));
            }
            // Receipt channel dropped or timed out: the message is still in the
            // delivery queue (e.g. waiting out a rate-limit retry) — report
            // pending honestly instead of claiming completion.
            Ok(Err(_)) | Err(_) => {
                return Ok(format!(
                    "File queued for delivery: {} ({}). The channel is congested; it should \
                     arrive shortly. Phrase your reply as \"sending\" (in progress), not \"sent\".",
                    ready.filename, size_display
                ));
            }
        }

        if exported_from_backend {
            Ok(format!(
                "File sent: {} ({}) [exported from the {} execution backend]",
                ready.filename,
                size_display,
                backend.kind().as_str()
            ))
        } else if ready.recovered_into_inbox {
            Ok(format!(
                "File sent: {} ({}) [copied into the inbox for delivery]",
                ready.filename, size_display
            ))
        } else {
            Ok(format!("File sent: {} ({})", ready.filename, size_display))
        }
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<mpsc::Sender<crate::types::StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let output = self.call(arguments).await?;
        let outcome_status = if output.starts_with("File sent:") {
            ToolOutcomeStatus::Succeeded
        } else if output.starts_with("File queued for delivery:") {
            ToolOutcomeStatus::Backgrounded
        } else {
            ToolOutcomeStatus::FailedPermanent
        };
        Ok(ToolCallOutcome {
            metadata: ToolCallMetadata {
                outcome_status: Some(outcome_status),
                background_started: outcome_status == ToolOutcomeStatus::Backgrounded,
                semantics: self.call_semantics(arguments),
                ..ToolCallMetadata::default()
            },
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{ResourceProvenance, ResourceRegisteredData};
    use crate::tools::file_delivery::is_recoverable_source;
    use std::path::Path;

    async fn test_event_store() -> Arc<EventStore> {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("connect test event store");
        Arc::new(
            EventStore::new(pool)
                .await
                .expect("migrate test event store"),
        )
    }

    async fn register_file_resource(
        store: &EventStore,
        session_id: &str,
        resource_id: &str,
        path: &Path,
    ) {
        let data = ResourceRegisteredData {
            schema_version: 1,
            resource_id: resource_id.to_string(),
            kind: "file".to_string(),
            locator: path.to_string_lossy().into_owned(),
            display_name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            mime_type: Some("application/pdf".to_string()),
            size_bytes: std::fs::metadata(path).ok().map(|metadata| metadata.len()),
            sha256: crate::channels::attachments::sha256_file(path),
            provenance: ResourceProvenance::ToolArtifact,
            produced_by_tool_call_id: Some("browser-call-1".to_string()),
            source_tool: Some("browser".to_string()),
            task_id: Some("task-1".to_string()),
            turn_id: Some("turn-1".to_string()),
        };
        store
            .append(Event::new(
                session_id,
                EventType::ResourceRegistered,
                serde_json::to_value(data).unwrap(),
            ))
            .await
            .expect("register resource");
    }

    #[tokio::test]
    async fn send_file_awaits_delivery_receipt_before_claiming_sent() {
        // Live repro (2026-07-02): send_file enqueued fire-and-forget and
        // returned "File sent"; the model told the user "I've sent it" while
        // the document sat behind a Telegram 429 retry for ~a minute. The
        // tool must only say "File sent" after the media listener confirms
        // delivery.
        let tmp = tempfile::tempdir().expect("tempdir");
        let inbox = tmp.path().join("inbox");
        std::fs::create_dir_all(&inbox).expect("create inbox");
        let f = inbox.join("report.txt");
        std::fs::write(&f, b"data").expect("write");

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let tool = std::sync::Arc::new(
            SendFileTool::new(tx, &[], &inbox.to_string_lossy())
                .with_receipt_timeout(std::time::Duration::from_secs(5)),
        );
        let args = json!({"_session_id": "sess-1", "file_path": f.to_string_lossy()}).to_string();

        let call_tool = tool.clone();
        let call =
            tokio::spawn(async move { call_tool.call_with_status_outcome(&args, None).await });
        // Deliver the receipt like the hub's media_listener does.
        let mut msg = rx.recv().await.expect("media message enqueued");
        let receipt = msg
            .result_tx
            .take()
            .expect("send_file must request a receipt");
        receipt.send(Ok(())).expect("receipt delivered");
        let outcome = call.await.expect("join").expect("call ok");
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        let out = outcome.output;
        assert!(out.contains("File sent"), "got: {out}");
    }

    #[tokio::test]
    async fn send_file_reports_failed_delivery_honestly() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let inbox = tmp.path().join("inbox");
        std::fs::create_dir_all(&inbox).expect("create inbox");
        let f = inbox.join("report.txt");
        std::fs::write(&f, b"data").expect("write");

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let tool = std::sync::Arc::new(
            SendFileTool::new(tx, &[], &inbox.to_string_lossy())
                .with_receipt_timeout(std::time::Duration::from_secs(5)),
        );
        let args = json!({"_session_id": "sess-1", "file_path": f.to_string_lossy()}).to_string();
        let call_tool = tool.clone();
        let call =
            tokio::spawn(async move { call_tool.call_with_status_outcome(&args, None).await });
        let mut msg = rx.recv().await.expect("media message enqueued");
        let receipt = msg.result_tx.take().expect("receipt requested");
        receipt
            .send(Err("system overload".to_string()))
            .expect("receipt delivered");
        let outcome = call.await.expect("join").expect("call ok");
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::FailedPermanent)
        );
        let out = outcome.output;
        assert!(
            out.contains("could not be delivered") && out.contains("system overload"),
            "must surface the delivery failure to the model, got: {out}"
        );
        assert!(!out.contains("File sent"), "must not claim success: {out}");
    }

    #[tokio::test]
    async fn send_file_reports_pending_when_receipt_times_out() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let inbox = tmp.path().join("inbox");
        std::fs::create_dir_all(&inbox).expect("create inbox");
        let f = inbox.join("report.txt");
        std::fs::write(&f, b"data").expect("write");

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let tool = SendFileTool::new(tx, &[], &inbox.to_string_lossy())
            .with_receipt_timeout(std::time::Duration::from_millis(100));
        let args = json!({"_session_id": "sess-1", "file_path": f.to_string_lossy()}).to_string();
        // Nobody answers the receipt (congested channel) → honest pending text.
        let outcome = tool
            .call_with_status_outcome(&args, None)
            .await
            .expect("call ok");
        let _keep_queue_alive = rx.recv().await; // message was still enqueued
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Backgrounded)
        );
        let out = outcome.output;
        assert!(
            out.contains("queued for delivery"),
            "must report pending, not sent: {out}"
        );
        assert!(!out.contains("File sent:"), "must not claim sent: {out}");
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
        let tool = std::sync::Arc::new(
            SendFileTool::new(tx, &[], &inbox.to_string_lossy())
                .with_receipt_timeout(std::time::Duration::from_secs(5)),
        );

        let args = json!({
            "_session_id": "sess-1",
            "file_path": src.to_string_lossy(),
        })
        .to_string();
        let call_tool = tool.clone();
        let call =
            tokio::spawn(async move { call_tool.call_with_status_outcome(&args, None).await });
        let mut msg = rx.recv().await.expect("media message sent");
        msg.result_tx
            .take()
            .expect("receipt requested")
            .send(Ok(()))
            .expect("receipt delivered");
        let outcome = call.await.expect("join").expect("call ok");
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        let out = outcome.output;
        assert!(out.contains("File sent"), "got: {out}");
        assert!(
            out.contains("copied into the inbox"),
            "should note recovery: {out}"
        );
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

    #[tokio::test]
    async fn send_file_resolves_exact_session_resource_and_checks_digest() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let inbox = tmp.path().join("inbox");
        std::fs::create_dir_all(&inbox).expect("create inbox");
        let artifact = inbox.join("brief.pdf");
        std::fs::write(&artifact, b"%PDF-1.7 exact artifact").expect("write artifact");

        let store = test_event_store().await;
        register_file_resource(&store, "sess-resource", "res_exact", &artifact).await;
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let tool = Arc::new(
            SendFileTool::new(tx, &[], &inbox.to_string_lossy())
                .with_event_store(store)
                .with_receipt_timeout(std::time::Duration::from_secs(5)),
        );
        let args = json!({
            "_session_id": "sess-resource",
            "resource_id": "res_exact"
        })
        .to_string();

        let call_tool = tool.clone();
        let call =
            tokio::spawn(async move { call_tool.call_with_status_outcome(&args, None).await });
        let mut message = rx.recv().await.expect("media enqueued");
        match &message.kind {
            MediaKind::Document { file_path, .. } => {
                assert_eq!(Path::new(file_path), artifact.canonicalize().unwrap());
            }
            _ => panic!("expected document"),
        }
        message
            .result_tx
            .take()
            .expect("delivery receipt")
            .send(Ok(()))
            .expect("deliver receipt");
        let outcome = call.await.expect("join").expect("call");
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        assert_eq!(
            outcome.metadata.semantics.target_hints[0].kind,
            ToolTargetHintKind::ResourceId
        );
    }

    #[tokio::test]
    async fn send_file_invalidates_resource_when_content_changes() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let inbox = tmp.path().join("inbox");
        std::fs::create_dir_all(&inbox).expect("create inbox");
        let artifact = inbox.join("brief.pdf");
        std::fs::write(&artifact, b"%PDF-1.7 original").expect("write artifact");
        let store = test_event_store().await;
        register_file_resource(&store, "sess-resource", "res_changed", &artifact).await;
        std::fs::write(&artifact, b"%PDF-1.7 replaced").expect("replace artifact");

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let tool =
            SendFileTool::new(tx, &[], &inbox.to_string_lossy()).with_event_store(store.clone());
        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "_session_id": "sess-resource",
                    "resource_id": "res_changed"
                })
                .to_string(),
                None,
            )
            .await
            .expect("call");

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::FailedPermanent)
        );
        assert!(outcome.output.contains("changed after it was registered"));
        assert!(rx.try_recv().is_err(), "changed content must not be queued");
        assert!(store
            .get_resource("sess-resource", "res_changed")
            .await
            .unwrap()
            .is_none());
    }

    #[test]
    fn send_file_schema_prefers_resource_id_without_requiring_a_path() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tool = SendFileTool::new(tx, &[], "/tmp/aidaemon-inbox");
        let schema = tool.schema();
        assert!(schema["parameters"]["properties"]["resource_id"].is_object());
        assert!(schema["parameters"].get("required").is_none());
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
}
