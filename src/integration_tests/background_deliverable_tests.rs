// Integration tests for Task 4: deliver-once ledger + deliverable-aware
// completion in the terminal background notifier.
//
// Scenario (the repro the design fixes): a command runs longer than
// `initial_timeout`, writes ONLY to a result file (empty stdout), then exits 0.
// The notifier must NOT take the trivial-output skip; it must attribute the
// produced file and deliver it directly via `send_media` (Document), guarded by
// a process-scoped deliver-once ledger.

use crate::channels::{ChannelHub, SessionMap};
use crate::state::SqliteStateStore;
use crate::tools::command_risk::PermissionMode;
use crate::tools::terminal::{ApprovalRequest, TerminalTool};
use crate::traits::{Channel, ChannelCapabilities, Tool};
use crate::types::{MediaKind, MediaMessage};
use async_trait::async_trait;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex, RwLock};

/// A media-capable test channel that records every `send_media` Document
/// delivery and every text message, so the test can assert on real file
/// delivery (not a text-only fallback).
#[derive(Default)]
struct MediaCaptureChannel {
    documents: Mutex<Vec<(String /*filename*/, String /*caption*/)>>,
    texts: Mutex<Vec<String>>,
}

impl MediaCaptureChannel {
    async fn document_count(&self) -> usize {
        self.documents.lock().await.len()
    }
    async fn documents(&self) -> Vec<(String, String)> {
        self.documents.lock().await.clone()
    }
    async fn texts(&self) -> Vec<String> {
        self.texts.lock().await.clone()
    }
}

#[async_trait]
impl Channel for MediaCaptureChannel {
    fn name(&self) -> String {
        "media_test".to_string()
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            markdown: true,
            inline_buttons: false,
            media: true,
            max_message_len: 4096,
        }
    }

    async fn send_text(&self, _session_id: &str, text: &str) -> anyhow::Result<()> {
        self.texts.lock().await.push(text.to_string());
        Ok(())
    }

    async fn send_media(&self, _session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        if let MediaKind::Document { filename, .. } = &media.kind {
            self.documents
                .lock()
                .await
                .push((filename.clone(), media.caption.clone()));
        }
        Ok(())
    }

    async fn request_approval(
        &self,
        _session_id: &str,
        _command: &str,
        _risk_level: crate::tools::command_risk::RiskLevel,
        _warnings: &[String],
        _permission_mode: PermissionMode,
    ) -> anyhow::Result<crate::types::ApprovalResponse> {
        Ok(crate::types::ApprovalResponse::AllowOnce)
    }
}

async fn make_terminal_with_hub(
    inbox_dir: PathBuf,
    outbox_dirs: Vec<PathBuf>,
    session_id: &str,
) -> (
    Arc<TerminalTool>,
    Arc<SqliteStateStore>,
    Arc<MediaCaptureChannel>,
) {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    // Keep the temp DB file alive for the test duration.
    std::mem::forget(db_file);
    let embedding_service = Arc::new(crate::memory::embeddings::EmbeddingService::new().unwrap());
    let state = Arc::new(
        SqliteStateStore::new(&db_path, 100, None, embedding_service)
            .await
            .unwrap(),
    );
    let pool = state.pool();
    let plan_store = Arc::new(crate::plans::PlanStore::new(pool.clone()).await.unwrap());

    let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
    let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);

    let tool = Arc::new(
        TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1, // 1-second initial timeout — moves long commands to background fast
            8000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>)
        .with_delivery_dirs(inbox_dir, outbox_dirs),
    );

    // Wire a media-capable channel + hub.
    let channel = Arc::new(MediaCaptureChannel::default());
    let session_map: SessionMap = Arc::new(RwLock::new(std::collections::HashMap::new()));
    session_map
        .write()
        .await
        .insert(session_id.to_string(), "media_test".to_string());
    let hub = Arc::new(ChannelHub::new(
        vec![channel.clone() as Arc<dyn Channel>],
        session_map,
    ));
    tool.set_hub(Arc::downgrade(&hub));
    tool.set_plan_store(plan_store);
    // Keep the hub alive for the duration of the notifier task.
    std::mem::forget(hub);

    (tool, state, channel)
}

#[tokio::test]
async fn empty_stdout_background_command_delivers_result_file() {
    let tmp = tempfile::tempdir().unwrap();
    let inbox = tmp.path().join("inbox");
    std::fs::create_dir_all(&inbox).unwrap();

    // A script that writes a result file but prints nothing to stdout.
    let script_path = std::env::temp_dir().join(format!("bgd_probe_{}.py", std::process::id()));
    let result_path =
        std::env::temp_dir().join(format!("bgd_probe_results_{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&result_path);
    let script = format!(
        "import time\noutput_path = \"{}\"\ntime.sleep(2)\nwith open(output_path, \"w\") as f:\n    f.write(\"latency: 42ms\\n\")\n",
        result_path.display()
    );
    std::fs::write(&script_path, script).unwrap();

    let session_id = "sess_bgd_empty";
    let (tool, _state, channel) =
        make_terminal_with_hub(inbox, vec![], session_id).await;

    let command = format!("python3 {}", script_path.display());
    let call = serde_json::json!({
        "action": "run",
        "command": command,
        "_session_id": session_id,
        "_user_role": "Owner",
    })
    .to_string();

    let response = tool.call(&call).await.unwrap();
    assert!(
        response.contains("Moved to background (pid="),
        "command should have been moved to background: {response}"
    );

    // Wait for completion + deliverable delivery.
    let mut delivered = false;
    for _ in 0..80 {
        if channel.document_count().await >= 1 {
            delivered = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    let docs = channel.documents().await;
    let texts = channel.texts().await;
    assert!(
        delivered,
        "the result file must be delivered as a Document media message; docs={docs:?}, texts={texts:?}"
    );
    let filename = result_path.file_name().unwrap().to_string_lossy().to_string();
    assert!(
        docs.iter().any(|(f, _)| f == &filename),
        "delivered document should be the result file {filename}; got {docs:?}"
    );
    // No "Activity summary"/"no results" give-up text should be the answer.
    assert!(
        !texts.iter().any(|t| t.to_lowercase().contains("does not show")
            || t.to_lowercase().contains("no results")
            || t.to_lowercase().contains("activity summary")),
        "must not emit a give-up text answer; texts={texts:?}"
    );

    // Deliver-once: a second completion notification path must not re-send.
    // Calling the (private) ledger via a second deliverable_send_once claim
    // would be the unit-level check; here we assert no duplicate document was
    // sent during the polling window above (only one delivery happened).
    assert_eq!(
        channel.document_count().await,
        1,
        "deliver-once: the result file must be sent exactly once"
    );

    let _ = std::fs::remove_file(&script_path);
    let _ = std::fs::remove_file(&result_path);
}

// Task 6: a detached command that was structured to produce an explicit output
// file but is idle-reaped before the file ever appears must close out with an
// HONEST failure ("expected output file ... never appeared"), not a silent end or
// the generic whole-disk-scan guidance (this command was not a scan).
#[tokio::test]
async fn reaped_command_without_produced_file_sends_honest_failure() {
    let tmp = tempfile::tempdir().unwrap();
    let inbox = tmp.path().join("inbox");
    std::fs::create_dir_all(&inbox).unwrap();

    let missing_path = std::env::temp_dir().join(format!(
        "bgd_unfulfilled_{}_{}.txt",
        std::process::id(),
        "reap"
    ));
    let _ = std::fs::remove_file(&missing_path);

    let session_id = "sess_bgd_reap";
    let (tool, _state, channel) = make_terminal_with_hub(inbox, vec![], session_id).await;

    // A command that hangs (so the idle-reaper, not the completion notifier,
    // handles it) and carries an explicit `--output` target it never writes.
    let command = format!(
        "python3 -c \"import time; time.sleep(30)\" --output {}",
        missing_path.display()
    );
    let call = serde_json::json!({
        "action": "run",
        "command": command,
        "_session_id": session_id,
        "_user_role": "Owner",
    })
    .to_string();

    let response = tool.call(&call).await.unwrap();
    assert!(
        response.contains("Moved to background (pid="),
        "command should have been moved to background: {response}"
    );

    // Force an immediate reap (zero idle + zero max-runtime thresholds).
    let reaped = tool
        .reap_stale_background_processes_with(Duration::from_secs(0), Duration::from_secs(0))
        .await;
    assert!(reaped >= 1, "the hung background command should have been reaped");

    // Allow the reaper's notification to flush.
    let mut got_honest = false;
    for _ in 0..40 {
        let texts = channel.texts().await;
        if texts.iter().any(|t| {
            let l = t.to_lowercase();
            l.contains("never appeared") && l.contains("nothing to send")
        }) {
            got_honest = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let texts = channel.texts().await;
    assert!(
        got_honest,
        "reaper must send an honest unfulfilled-deliverable failure; texts={texts:?}"
    );
    // No file was produced, so nothing should have been delivered as a document.
    assert_eq!(
        channel.document_count().await,
        0,
        "no document should be sent when the produced file never appeared"
    );

    let _ = std::fs::remove_file(&missing_path);
}
