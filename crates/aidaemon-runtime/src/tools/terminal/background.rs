//! Running and finished background processes, re-engagement limits and
//! delivery of a background run's files and text.

use super::*;

/// A background process being tracked after it exceeded the initial timeout.
///
/// Process lifecycle modes:
/// 1. **Task-owned** (`detached=false`, `notifier_active=false`): killed on task-end.
/// 2. **Background with notifier** (`detached=false`, `notifier_active=true`): survives
///    task-end so the notifier can deliver the result. Killed when the notifier finishes.
/// 3. **Detached** (`detached=true`): survives task-end and notifier. Requires explicit kill.
pub(super) struct RunningProcess {
    pub(super) command: String,
    /// The plain title the call gave the command, for the owner's lists.
    pub(super) title: Option<String>,
    pub(super) declared_write_paths: Vec<String>,
    pub(super) semantics: ToolCallSemantics,
    pub(super) dedupe_key: Option<String>,
    pub(super) owner_task_id: Option<String>,
    /// The task that started it, kept when the task disowns it: the chat's
    /// list of running work groups jobs by the request that started them.
    pub(super) origin_task_id: Option<String>,
    pub(super) detached: bool,
    pub(super) started_at: Instant,
    pub(super) stdout_buf: Arc<Mutex<Vec<u8>>>,
    pub(super) stderr_buf: Arc<Mutex<Vec<u8>>>,
    pub(super) reader_handle: JoinHandle<Option<i32>>,
    pub(super) process_handle: ProcessHandle,
    pub(super) notify_on_completion: Arc<AtomicBool>,
    /// True only when the background notifier tokio task was actually spawned
    /// and is actively monitoring this process for completion/progress delivery.
    /// Used by `cleanup_task_processes` to decide whether to kill or disown.
    pub(super) notifier_active: bool,
    /// Session/goal that launched this command, captured so the idle-reaper can
    /// notify the user when it auto-stops a hung background process. Empty for
    /// task-owned processes that never enter the notifier path.
    pub(super) notify_session_id: String,
    pub(super) notify_goal_id: String,
    /// The run it is in the runtime's records, once it went to the
    /// background as one: the chat's list of running work names it by this.
    pub(super) run_id: Option<String>,
    /// Idle-reap bookkeeping: total output bytes observed at the last sweep and
    /// the instant any progress signal last advanced. A notifier-active,
    /// non-detached process that makes no objective progress becomes a stall
    /// candidate after its workload-specific threshold; termination normally
    /// requires another conclusive sample. Genuinely working processes keep
    /// resetting `last_progress_at` and survive the stall path.
    pub(super) last_progress_len: usize,
    pub(super) last_progress_at: Instant,
    /// Cumulative CPU time (ms) observed at the last sweep. Advancing CPU time is
    /// progress even when the process is silent (a busy scan statting files).
    pub(super) last_cpu_ms: u64,
    /// Cumulative disk read+written bytes observed at the last sweep. Advancing
    /// disk I/O is progress even when the process is silent.
    pub(super) last_io_bytes: u64,
    /// Stable summary of the tracked process subtree. A child starting or
    /// finishing is observable progress even when aggregate counters fall as a
    /// completed child leaves the tree.
    pub(super) last_tree_fingerprint: u64,
    /// Number of consecutive, independently sampled threshold crossings. The
    /// first crossing is only a suspected stall; termination requires the
    /// launch-time contract's confirmation count.
    pub(super) idle_confirmations: u8,
    pub(super) progress_contract: BackgroundProgressContract,
}

/// Finalized background process output retained briefly so `action="check"`
/// can still return results after automatic reaping.
pub(super) struct CompletedProcess {
    pub(super) output: String,
    pub(super) metadata: ToolCallMetadata,
    pub(super) completed_at: Instant,
}

/// Max agent re-engagements from the completions of one background job (the
/// same command, the same CLI agent) per session within [`REENGAGE_WINDOW`].
/// Beyond it the raw output is delivered instead of re-entering the agent
/// loop. Guards against runaway cycles where a re-engaged task stalls,
/// spawns the same background command again, and its completion re-engages
/// again (observed 2026-06-06: a stalled task re-spawned whole-home `find`
/// scans, burning ~24k-token LLM calls for ~30 minutes). Different jobs are
/// progress: build, then test, then check (live 2026-11-15: a coding task
/// was cut off after Codex, a format check and a test run).
pub(super) const MAX_REENGAGEMENTS_PER_JOB: usize = 2;
/// Max re-engagements per session within [`REENGAGE_WINDOW`], whatever the
/// jobs: a backstop.
pub(super) const MAX_REENGAGEMENTS_PER_WINDOW: usize = 12;
/// Sliding window for the re-engagement cap.
pub(super) const REENGAGE_WINDOW: Duration = Duration::from_secs(600);
pub(super) const BACKGROUND_DELIVERY_DEDUPE_WINDOW: Duration = Duration::from_secs(600);

/// Sliding-window limiter for background-completion agent re-engagements:
/// `job` names what completed (a command line, a CLI agent). Records `now`
/// and returns `true` when the session still has budget for it; returns
/// `false` (recording nothing) once that job repeated past its cap or the
/// session reached its total.
pub fn reengagement_allowed(
    log: &mut HashMap<String, std::collections::VecDeque<Instant>>,
    session_id: &str,
    job: &str,
    now: Instant,
) -> bool {
    let job_key = format!(
        "{session_id}\u{1f}{}",
        job.split_whitespace().collect::<Vec<_>>().join(" ")
    );
    let mut within = |key: &str, cap: usize| {
        let entries = log.entry(key.to_string()).or_default();
        while entries
            .front()
            .is_some_and(|t| now.duration_since(*t) >= REENGAGE_WINDOW)
        {
            entries.pop_front();
        }
        entries.len() < cap
    };
    if !within(session_id, MAX_REENGAGEMENTS_PER_WINDOW)
        || !within(&job_key, MAX_REENGAGEMENTS_PER_JOB)
    {
        return false;
    }
    for key in [session_id.to_string(), job_key] {
        log.entry(key).or_default().push_back(now);
    }
    true
}

pub(super) fn normalize_background_delivery_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(super) fn background_delivery_allowed(
    log: &mut HashMap<String, HashMap<String, Instant>>,
    session_id: &str,
    text: &str,
    now: Instant,
) -> bool {
    let normalized = normalize_background_delivery_text(text);
    if normalized.is_empty() {
        return false;
    }

    let entries = log.entry(session_id.to_string()).or_default();
    entries.retain(|_, sent_at| now.duration_since(*sent_at) < BACKGROUND_DELIVERY_DEDUPE_WINDOW);
    if entries.contains_key(&normalized) {
        return false;
    }

    entries.insert(normalized, now);
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeliverableDeliveryState {
    Delivering,
    Delivered,
    Failed,
}

#[derive(Debug)]
pub(super) enum DeliverableAttribution {
    One(PathBuf),
    Ambiguous(Vec<PathBuf>),
    ExpectedMissing(Vec<PathBuf>),
    None,
}

pub(super) fn format_deliverable_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .take(5)
        .map(|p| format!("`{}`", p.display()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Completion-time deliverable attribution for a finished background command.
///
/// Uses exact write targets from the enforced process manifest. Shell source,
/// referenced scripts, and checklist prose cannot create artifact authority.
pub(super) async fn attribute_background_deliverable(
    backend: &SharedExecutionBackend,
    declared_write_paths: &[String],
    command_start: std::time::SystemTime,
    command_end: std::time::SystemTime,
) -> DeliverableAttribution {
    let ctx = aidaemon_tools::background_deliverable::attribute_declared_deliverables_backend(
        backend.clone(),
        declared_write_paths,
        command_start,
        command_end,
    )
    .await;
    match aidaemon_tools::background_deliverable::auto_send_decision(&ctx) {
        aidaemon_tools::background_deliverable::AutoSendDecision::One(p) => {
            DeliverableAttribution::One(p)
        }
        aidaemon_tools::background_deliverable::AutoSendDecision::Ambiguous(paths) => {
            DeliverableAttribution::Ambiguous(paths)
        }
        aidaemon_tools::background_deliverable::AutoSendDecision::None => {
            if !ctx.unconfirmed_candidates.is_empty() {
                DeliverableAttribution::ExpectedMissing(ctx.unconfirmed_candidates)
            } else {
                DeliverableAttribution::None
            }
        }
    }
}

/// Map a [`file_delivery::DeliveryError`] to a short, honest user-facing reason.
pub(super) fn describe_delivery_error(
    err: &aidaemon_tools::file_delivery::DeliveryError,
) -> String {
    use aidaemon_tools::file_delivery::DeliveryError;
    match err {
        DeliveryError::FileNotFound(_) => "the file no longer exists".to_string(),
        DeliveryError::NotRegularFile(_) => "it is not a regular file".to_string(),
        DeliveryError::Blocked(_) => "the path is blocked for security reasons".to_string(),
        DeliveryError::OutsideAllowedDirs(_) => {
            "it is outside the allowed delivery directories".to_string()
        }
        DeliveryError::RecoveryFailed { error, .. } => {
            format!("recovery into the inbox failed ({error})")
        }
    }
}

/// Text files up to this size are quoted in the delivery caption (Telegram
/// captions stop at 1024 characters).
pub(super) const SMALL_TEXT_PREVIEW_BYTES: usize = 600;

/// The trimmed content of a small UTF-8 text file without control
/// characters, so a delivery can answer "what does it contain" inline.
pub(super) fn small_text_preview(path: &std::path::Path) -> Option<String> {
    let metadata = std::fs::metadata(path).ok()?;
    if metadata.len() as usize > SMALL_TEXT_PREVIEW_BYTES {
        return None;
    }
    let text = String::from_utf8(std::fs::read(path).ok()?).ok()?;
    if text
        .chars()
        .any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t'))
    {
        return None;
    }
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

/// Build the user-facing caption for an auto-delivered background result file.
/// Never embeds the shell command — only the filename and an optional summary.
pub(super) fn build_deliverable_caption(filename: &str, summary: Option<&str>) -> String {
    match summary {
        Some(s) if !s.trim().is_empty() => format!("{}\n\n📄 {}", s.trim(), filename),
        _ => format!("Done — result file attached.\n\n📄 {}", filename),
    }
}

/// Deliver one attributed produced-output file directly to the session's
/// channel, guarded by the process-scoped deliver-once ledger. On a successful
/// document send, the conservative single-item checklist write-back marks the
/// delivery step complete. On any failure, an honest text message is sent
/// instead so the user is never left thinking a file was delivered when it was
/// not. Returns nothing; all paths are best-effort with logging.
#[allow(clippy::too_many_arguments)]
pub(super) async fn deliver_attributed_background_file(
    backend: &SharedExecutionBackend,
    path: &std::path::Path,
    session_id: &str,
    _command_summary: &str,
    inbox_dir: &std::path::Path,
    outbox_dirs: &[PathBuf],
    delivered_deliverables: &Arc<Mutex<HashMap<(String, String), DeliverableDeliveryState>>>,
    plan_store: Option<&Arc<aidaemon_work::plans::PlanStore>>,
    hub: Option<&Arc<dyn OutboundRouter>>,
    state: Option<&Arc<dyn NotificationStore>>,
    goal_id: &str,
    pid: u32,
) {
    let original_name = path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());
    let mut delivery_path = path.to_path_buf();
    if backend.kind() != BackendKind::Local {
        let backend_path = match backend.resolve_path(&path.to_string_lossy()).await {
            Ok(path) => path,
            Err(error) => {
                let msg = format!(
                    "⚠️ The background command finished and referenced `{original_name}`, \
                     but its execution-side path was invalid: {error}."
                );
                deliver_background_text(hub, state, session_id, goal_id, &msg, pid).await;
                return;
            }
        };
        let canonical = backend
            .canonicalize(&backend_path)
            .await
            .unwrap_or(backend_path);
        if aidaemon_tools::file_delivery::is_path_blocked(std::path::Path::new(canonical.as_str()))
        {
            let msg = format!(
                "⚠️ The background command produced `{original_name}`, but that path is blocked \
                 from delivery for security reasons."
            );
            deliver_background_text(hub, state, session_id, goal_id, &msg, pid).await;
            return;
        }
        delivery_path = inbox_dir.join(canonical.file_name().unwrap_or(original_name.as_str()));
        if let Err(error) = backend.export_local_file(&canonical, &delivery_path).await {
            let msg = format!(
                "⚠️ The background command produced `{original_name}`, but exporting it from \
                 the {} execution backend failed: {error}.",
                backend.kind().as_str()
            );
            deliver_background_text(hub, state, session_id, goal_id, &msg, pid).await;
            return;
        }
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let ready = match aidaemon_tools::file_delivery::prepare_delivery(
        &delivery_path.to_string_lossy(),
        &cwd,
        inbox_dir,
        outbox_dirs,
    ) {
        Ok(r) => r,
        Err(e) => {
            let msg = format!(
                "⚠️ The background command finished and produced `{}`, but I couldn't deliver it: {}.",
                original_name,
                describe_delivery_error(&e)
            );
            deliver_background_text(hub, state, session_id, goal_id, &msg, pid).await;
            return;
        }
    };

    let canonical = ready.canonical_path.to_string_lossy().to_string();
    // Deliver-once: the FIRST claimant for this (session, canonical path) sends;
    // duplicate completion paths / the reaper / a re-engaged send get `false`.
    {
        let mut ledger = delivered_deliverables.lock().await;
        let key = (session_id.to_string(), canonical.clone());
        match ledger.get(&key) {
            Some(DeliverableDeliveryState::Delivered | DeliverableDeliveryState::Delivering) => {
                info!(
                    pid,
                    session_id = %session_id,
                    "Deliver-once: background deliverable already sent or in flight; skipping re-send"
                );
                return;
            }
            Some(DeliverableDeliveryState::Failed) | None => {
                ledger.insert(key, DeliverableDeliveryState::Delivering);
            }
        }
    }

    let mark_delivery_state = |state: DeliverableDeliveryState| {
        let delivered_deliverables = delivered_deliverables.clone();
        let session_id = session_id.to_string();
        let canonical = canonical.clone();
        async move {
            delivered_deliverables
                .lock()
                .await
                .insert((session_id, canonical), state);
        }
    };

    let preview = small_text_preview(&ready.canonical_path)
        .map(|text| format!("Done. {} contains:\n{}", ready.filename, text));
    let caption = build_deliverable_caption(&ready.filename, preview.as_deref());
    let msg = MediaMessage {
        session_id: session_id.to_string(),
        caption: caption.clone(),
        kind: MediaKind::Document {
            file_path: ready.canonical_path.to_string_lossy().to_string(),
            filename: ready.filename.clone(),
        },
        // No result_tx: `send_media_strict` already returns Err on a non-media
        // channel or send failure, so its `Ok` IS the honest success signal — a
        // text fallback can never masquerade as a document delivery.
        result_tx: None,
    };

    let sent = match hub {
        Some(hub) => match hub.send_media_strict(session_id, &msg).await {
            Ok(()) => true,
            Err(e) => {
                warn!(
                    pid,
                    error = %e,
                    session_id = %session_id,
                    "Failed to deliver attributed background result file"
                );
                false
            }
        },
        None => false,
    };

    if sent {
        mark_delivery_state(DeliverableDeliveryState::Delivered).await;
        info!(
            pid,
            session_id = %session_id,
            filename = %ready.filename,
            "Delivered attributed background result file as a document"
        );
        // Reconcile only an explicit typed delivery contract for this exact
        // canonical path. Checklist prose cannot authorize completion.
        if let Some(ps) = plan_store {
            let semantics =
                ToolCallSemantics::mutation_with(ToolMutationEffects::EXTERNAL_DELIVERY)
                    .with_target_hint(ToolTargetHintKind::Path, canonical.clone());
            let receipt_id = format!("background-delivery:{pid}");
            if let Err(e) = ps
                .reconcile_checklist_after_tool_success(
                    session_id,
                    &semantics,
                    &receipt_id,
                    "Background file delivered",
                )
                .await
            {
                warn!(
                    pid,
                    error = %e,
                    session_id = %session_id,
                    "Failed to mark delivery checklist step complete after sending file"
                );
            }
        }
    } else {
        mark_delivery_state(DeliverableDeliveryState::Failed).await;
        let msg = format!(
            "⚠️ The background command finished and produced `{original_name}`, but I couldn't deliver the file to this channel."
        );
        deliver_background_text(hub, state, session_id, goal_id, &msg, pid).await;
    }
}

pub(super) async fn deliver_background_text(
    hub: Option<&Arc<dyn OutboundRouter>>,
    state: Option<&Arc<dyn NotificationStore>>,
    session_id: &str,
    goal_id: &str,
    message: &str,
    pid: u32,
) {
    let mut delivered = false;
    if let Some(hub) = hub {
        match hub.send_text(session_id, message).await {
            Ok(()) => delivered = true,
            Err(e) => warn!(
                pid,
                error = %e,
                session_id = %session_id,
                "Failed to deliver background deliverable status text"
            ),
        }
    }
    if !delivered && let Some(state) = state {
        let entry =
            crate::tools::traits::NotificationEntry::new(goal_id, session_id, "progress", message);
        if let Err(e) = state.enqueue_notification(&entry).await {
            warn!(
                pid,
                error = %e,
                session_id = %session_id,
                goal_id = %goal_id,
                "Failed to enqueue background deliverable status text"
            );
        }
    }
}
