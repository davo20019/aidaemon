//! The process table: dedupe keys, termination, task cleanup and
//! reaping.

use super::*;

impl TerminalTool {
    pub(super) fn dedupe_scope_key(
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

    pub(super) fn dedupe_key_for_run(
        command: &str,
        notify_session_id: &str,
        notify_goal_id: Option<&str>,
        task_id: Option<&str>,
    ) -> String {
        let scope = Self::dedupe_scope_key(notify_session_id, notify_goal_id, task_id);
        let normalized = normalize_command_for_dedupe(command);
        format!("{}|{}", scope, normalized)
    }

    pub(super) async fn insert_indexes_for_process(
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

        if !detached && let Some(task_id) = owner_task_id {
            let mut task_map = self.task_processes.lock().await;
            task_map.entry(task_id.to_string()).or_default().insert(pid);
        }
    }

    pub(super) async fn remove_indexes_for_process(&self, pid: u32, proc: &RunningProcess) {
        if let Some(key) = proc.dedupe_key.as_ref() {
            let mut dedupe = self.running_by_dedupe_key.lock().await;
            if dedupe.get(key).copied() == Some(pid) {
                dedupe.remove(key);
            }
        }

        if !proc.detached
            && let Some(task_id) = proc.owner_task_id.as_ref()
        {
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

    pub(super) async fn resolve_duplicate_running_pid(&self, dedupe_key: &str) -> Option<u32> {
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

    pub(super) async fn terminate_running_process(
        &self,
        pid: u32,
        proc: RunningProcess,
        reason: &str,
    ) -> anyhow::Result<(String, Option<crate::tools::traits::TruncationInfo>)> {
        self.terminate_running_process_verified(pid, proc, reason)
            .await
            .map(|(output, truncation, _)| (output, truncation))
    }

    /// [`Self::terminate_running_process`], and whether the process is
    /// known to have exited: its wait returned, rather than being abandoned.
    pub(super) async fn terminate_running_process_verified(
        &self,
        pid: u32,
        proc: RunningProcess,
        reason: &str,
    ) -> anyhow::Result<(String, Option<crate::tools::traits::TruncationInfo>, bool)> {
        proc.notify_on_completion.store(false, Ordering::Relaxed);
        let notify_session = proc.notify_session_id.clone();
        let started_at = proc.started_at;
        let command = proc.command.clone();
        let stdout_buf = proc.stdout_buf.clone();
        let stderr_buf = proc.stderr_buf.clone();
        let reader_handle = proc.reader_handle;
        let process_handle = proc.process_handle;

        if !reader_handle.is_finished() {
            self.backend
                .terminate(&process_handle, Duration::from_secs(2))
                .await
                .ok();
            let finished = tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    if reader_handle.is_finished() {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            })
            .await;
            if finished.is_err() && !reader_handle.is_finished() {
                if let Some(pid) = process_handle.local_pid() {
                    send_sigkill(pid);
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }

        let exited = reader_handle.is_finished();
        if !exited {
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
        let (formatted, truncation) = format_output(
            &stdout,
            &stderr,
            self.max_output_chars,
            self.spill_session(&notify_session),
        );
        output.push_str(&formatted);
        Ok((output, truncation, exited))
    }

    pub(super) async fn cleanup_task_processes(&self, task_id: &str) -> anyhow::Result<usize> {
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
                    command = %LoggedCommand(&proc.command),
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

    /// Whether every command runs unasked: AIdaemon's level is Full access.
    pub async fn trust_all_enabled(&self) -> bool {
        self.daemon_permissions
            .get()
            .unwrap_or(&self.permissions)
            .trust_all()
    }

    /// Follow the daemon's permission level (shared with its engine and
    /// Settings) instead of this terminal's own. Set once, at start.
    pub fn set_permissions(&self, permissions: crate::permissions::Permissions) {
        let _ = self.daemon_permissions.set(permissions);
    }

    /// The reply to a model still calling the retired `trust_all` action.
    pub(super) const TRUST_ALL_RETIRED: &'static str = "Nothing changed: trust-all mode is not a terminal \
        action. Only the owner can turn it on or off, with the /trustall command.";

    pub(super) fn prune_completed_map(completed: &mut HashMap<u32, CompletedProcess>) {
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
    pub(super) async fn reap_finished(&self) {
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
            let semantics = proc.semantics.clone();
            let exit_code = proc.reader_handle.await.ok().flatten();
            let stdout = String::from_utf8_lossy(&proc.stdout_buf.lock().await).to_string();
            let stderr = String::from_utf8_lossy(&proc.stderr_buf.lock().await).to_string();
            let mut output = format!(
                "[Process pid={} finished after {:.0}s]\n",
                pid,
                proc.started_at.elapsed().as_secs_f64()
            );
            let (formatted, truncation) = format_output(
                &stdout,
                &stderr,
                self.max_output_chars,
                self.spill_session(&proc.notify_session_id),
            );
            output.push_str(&formatted);
            push_exit_status(&mut output, exit_code);

            let mut metadata = background_completion_metadata(exit_code);
            metadata.truncation = truncation;
            metadata.semantics = semantics;
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
            info!(pid, command = %LoggedCommand(&proc.command), "Reaped finished background process");
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

    /// Sample process-tree resource/liveness metadata for each pid via `sysinfo`.
    ///
    /// Cross-platform: `sysinfo` covers macOS/Linux/Windows, so there are no
    /// per-OS `cfg` blocks. A pid that has exited (or that the OS won't stat)
    /// simply has no map entry — the caller treats a missing sample as "no
    /// progress evidence" and falls back to output-based progress.
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
    pub(super) fn sample_process_resources(pids: &[u32]) -> HashMap<u32, ProcessResourceSample> {
        use sysinfo::{ProcessRefreshKind, ProcessStatus, ProcessesToUpdate, System};

        if pids.is_empty() {
            return HashMap::new();
        }

        // Refresh ALL processes (not just the tracked roots) so descendant
        // CPU/IO is visible. cpu + disk only — we never need names/cmdlines here.
        // An unreadable process table leaves the snapshot empty: no samples,
        // the same as a pid that has already exited.
        let mut sys = System::new();
        sys.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing().with_cpu().with_disk_usage(),
        );

        // Build a flat per-pid sample map and a parent→children index from the
        // full snapshot, then delegate the subtree summing to the pure helper.
        let procs = sys.processes();
        let mut per_pid: HashMap<u32, (u64, u64, bool)> = HashMap::with_capacity(procs.len());
        let mut children_of: HashMap<u32, Vec<u32>> = HashMap::new();
        for (pid, proc) in procs {
            let pid_u32 = pid.as_u32();
            let cpu_ms = proc.accumulated_cpu_time();
            let du = proc.disk_usage();
            let io_bytes = du.total_read_bytes.saturating_add(du.total_written_bytes);
            per_pid.insert(
                pid_u32,
                (cpu_ms, io_bytes, proc.status() == ProcessStatus::Run),
            );
            if let Some(parent) = proc.parent() {
                children_of
                    .entry(parent.as_u32())
                    .or_default()
                    .push(pid_u32);
            }
        }

        sum_subtree_resources(pids, &children_of, &per_pid)
    }

    /// Stall + max-runtime variant. `stall_threshold` is the configurable base;
    /// the launch-time contract selects the effective no-progress window and
    /// confirmation count. `max_runtime` remains hard only for generic work.
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
            last_tree_fingerprint: u64,
            idle_confirmations: u8,
            progress_contract: BackgroundProgressContract,
            started_at: Instant,
            session_id: String,
            owner_task_id: Option<String>,
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
                    last_tree_fingerprint: p.last_tree_fingerprint,
                    idle_confirmations: p.idle_confirmations,
                    progress_contract: p.progress_contract,
                    started_at: p.started_at,
                    session_id: p.notify_session_id.clone(),
                    owner_task_id: p.owner_task_id.clone(),
                })
                .collect()
        };
        if candidates.is_empty() {
            return 0;
        }

        // Phase 2: sample resources + output (no `running` lock held). A process
        // is "making progress" if CPU time, disk I/O, output, or its process
        // tree changed; that refreshes its progress clock. Threshold crossings
        // are confirmed on another conclusive sweep before termination. Generic
        // commands additionally retain the absolute max-runtime leak backstop.
        let pids: Vec<u32> = candidates.iter().map(|c| c.pid).collect();
        let samples = Self::sample_process_resources(&pids);

        // Bookkeeping updates carry the latest conclusive sample even when it
        // was flat. This makes process-tree transitions comparable on the next
        // sweep without pretending that a baseline refresh is semantic progress.
        let mut to_update: Vec<(u32, usize, Option<ProcessResourceSample>, bool, u8)> = Vec::new();
        let mut to_reap: Vec<(u32, ReapReason)> = Vec::new();
        let mut watchdog_events: Vec<(String, String, DecisionPointData)> = Vec::new();
        for c in candidates {
            let len = c.stdout_buf.lock().await.len() + c.stderr_buf.lock().await.len();
            let sampled = samples.get(&c.pid).copied();
            // Missing sample (process gone / OS denied) is INCONCLUSIVE, not
            // evidence of a stall. Output growth can still prove progress, but
            // a local process is never stall-killed solely because OS sampling
            // disappeared.
            let sample = sampled.unwrap_or(ProcessResourceSample {
                cpu_ms: c.last_cpu_ms,
                io_bytes: c.last_io_bytes,
                tree_fingerprint: c.last_tree_fingerprint,
                process_count: 0,
                runnable_count: 0,
            });

            // A local ssh/docker client is only a transport process; its host
            // CPU/IO says nothing about remote work. Treat a live remote
            // transport as progress for the stall policy while retaining the
            // absolute max-runtime backstop.
            let made_progress = self.backend.kind() != BackendKind::Local
                || len > c.last_progress_len
                || sampled.is_some_and(|sample| {
                    process_made_progress(
                        ProcessResourceSample {
                            cpu_ms: c.last_cpu_ms,
                            io_bytes: c.last_io_bytes,
                            tree_fingerprint: c.last_tree_fingerprint,
                            process_count: 0,
                            runnable_count: 0,
                        },
                        c.last_progress_len,
                        len,
                        sample,
                    )
                });

            let total_runtime = c.started_at.elapsed();
            let no_progress_elapsed = if made_progress {
                Duration::ZERO
            } else {
                c.last_progress_at.elapsed()
            };
            let effective_stall = c
                .progress_contract
                .effective_stall_threshold(stall_threshold);

            if should_idle_reap(
                true,
                false,
                no_progress_elapsed,
                total_runtime,
                effective_stall,
                max_runtime,
                c.progress_contract.hard_max_runtime,
            ) {
                let reason = if c.progress_contract.hard_max_runtime
                    && total_runtime >= max_runtime
                    && no_progress_elapsed < effective_stall
                {
                    ReapReason::MaxRuntime
                } else {
                    ReapReason::Stalled
                };

                // A stall requires a real OS sample plus another independent
                // threshold-crossing sweep. Max-runtime remains an immediate
                // leak backstop only for the generic contract.
                let conclusive_stall = reason != ReapReason::Stalled || sampled.is_some();
                let confirmations = if reason == ReapReason::Stalled && conclusive_stall {
                    c.idle_confirmations.saturating_add(1)
                } else {
                    0
                };
                let confirmed = stall_threshold.is_zero()
                    || reason == ReapReason::MaxRuntime
                    || (conclusive_stall
                        && confirmations >= c.progress_contract.idle_confirmations_required);
                if confirmed {
                    to_reap.push((c.pid, reason));
                }
                to_update.push((c.pid, len, sampled, made_progress, confirmations));

                if conclusive_stall && let Some(task_id) = c.owner_task_id.as_deref() {
                    watchdog_events.push((
                            c.session_id.clone(),
                            task_id.to_string(),
                            DecisionPointData {
                                decision_type: DecisionType::ExecutionFailureClassification,
                                task_id: task_id.to_string(),
                                iteration: 0,
                                severity: DiagnosticSeverity::Warning,
                                code: Some("background_progress_watchdog".to_string()),
                                metadata: json!({
                                    "pid": c.pid,
                                    "action": if confirmed { "reap" } else { "confirm_idle" },
                                    "reason": reason.as_str(),
                                    "workload": c.progress_contract.workload.as_str(),
                                    "stall_threshold_ms": effective_stall.as_millis().min(u64::MAX as u128) as u64,
                                    "no_progress_ms": no_progress_elapsed.as_millis().min(u64::MAX as u128) as u64,
                                    "runtime_ms": total_runtime.as_millis().min(u64::MAX as u128) as u64,
                                    "idle_confirmation": confirmations,
                                    "idle_confirmations_required": c.progress_contract.idle_confirmations_required,
                                    "sample_present": sampled.is_some(),
                                    "cpu_ms_previous": c.last_cpu_ms,
                                    "cpu_ms_current": sample.cpu_ms,
                                    "io_bytes_previous": c.last_io_bytes,
                                    "io_bytes_current": sample.io_bytes,
                                    "output_bytes_previous": c.last_progress_len,
                                    "output_bytes_current": len,
                                    "process_count": sample.process_count,
                                    "runnable_count": sample.runnable_count,
                                    "tree_changed": c.last_tree_fingerprint != 0
                                        && sample.tree_fingerprint != c.last_tree_fingerprint,
                                }),
                                summary: if confirmed {
                                    format!(
                                        "Background {} command confirmed {} after {} idle samples",
                                        c.progress_contract.workload.as_str(),
                                        reason.as_str(),
                                        confirmations
                                    )
                                } else {
                                    format!(
                                        "Background {} command suspected idle; awaiting confirmation",
                                        c.progress_contract.workload.as_str()
                                    )
                                },
                            },
                        ));
                }
            } else {
                to_update.push((c.pid, len, sampled, made_progress, 0));
            }
        }

        // Persist watchdog decisions before termination so a crash during kill
        // cannot erase the evidence used to classify the process as stalled.
        if let Some(store) = &self.event_store {
            for (session_id, task_id, event) in watchdog_events {
                if session_id.is_empty() {
                    continue;
                }
                let emitter = aidaemon_events::EventEmitter::new(store.clone(), session_id)
                    .with_task_id(task_id);
                if let Err(error) = emitter.emit(EventType::DecisionPoint, event).await {
                    warn!(%error, "Failed to persist background watchdog sample");
                }
            }
        }

        // Phase 3a: refresh sample baselines, confirmation count, and—only for
        // genuine progress—the semantic progress clock.
        if !to_update.is_empty() {
            let now = Instant::now();
            let mut running = self.running.lock().await;
            for (pid, len, sample, made_progress, idle_confirmations) in to_update {
                if let Some(proc) = running.get_mut(&pid) {
                    if made_progress {
                        proc.last_progress_len = len;
                        proc.last_progress_at = now;
                    }
                    if let Some(sample) = sample {
                        proc.last_cpu_ms = sample.cpu_ms;
                        proc.last_io_bytes = sample.io_bytes;
                        proc.last_tree_fingerprint = sample.tree_fingerprint;
                    }
                    proc.idle_confirmations = if made_progress { 0 } else { idle_confirmations };
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
            // Captured here because `proc` is moved into
            // `terminate_running_process` below.
            let declared_write_paths = proc.declared_write_paths.clone();
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
                // Task 6: honest closeout for an unfulfilled deliverable. If this
                // command was structured to produce an explicit output file
                // (attributable) but the file never appeared and nothing was
                // delivered for this session, say so plainly instead of the generic
                // whole-disk-scan guidance — that command was not a scan, it was
                // supposed to hand back a file. Skipped when nothing is
                // attributable (servers, watchers, scans → unchanged behavior).
                let unfulfilled_deliverable: Option<String> = {
                    let command_end = std::time::SystemTime::now();
                    let command_start = command_end
                        .checked_sub(Duration::from_secs(runtime_secs))
                        .unwrap_or(command_end);
                    match attribute_background_deliverable(
                        &self.backend,
                        &declared_write_paths,
                        command_start,
                        command_end,
                    )
                    .await
                    {
                        DeliverableAttribution::One(path) => {
                            let file_appeared =
                                match self.backend.resolve_path(&path.to_string_lossy()).await {
                                    Ok(path) => self.backend.metadata(&path).await.is_ok(),
                                    Err(_) => false,
                                };
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
                        DeliverableAttribution::Ambiguous(_) | DeliverableAttribution::None => None,
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
                if let Some(hub) = self.get_hub()
                    && hub.send_text(&session_id, &message).await.is_ok()
                {
                    delivered = true;
                }
                if !delivered && let Some(ref state) = self.notifications {
                    let entry = crate::tools::traits::NotificationEntry::new(
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
            reaped += 1;
        }
        reaped
    }
}
