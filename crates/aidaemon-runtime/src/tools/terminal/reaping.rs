//! Background process supervision: progress contracts, idle reaping and
//! process-tree resource sampling.

use super::*;

/// Max bytes per stream buffer (1 MB) to prevent unbounded memory growth.
pub(super) const BUFFER_CAP: usize = 1_048_576;

/// Maximum wall time for a background completion to re-enter the agent loop.
/// The raw command/worker result is always delivered when this budget expires,
/// so a slow model or a wedged continuation can never strand the user's final
/// answer or monopolize the global re-engagement serializer indefinitely.
/// It is how long a continuation may go without progress (and how long it
/// may wait for its slot), never how long it may work: ten minutes quiet
/// means wedged; real work (fix, rebuild, re-test) keeps beating.
pub(super) const BACKGROUND_CONTINUATION_TIMEOUT: Duration = Duration::from_secs(600);
/// `detach=true` is a launch-and-return contract: the process is given this
/// brief startup window so an immediate failure (bad binary, usage error) is
/// still reported synchronously with its exit code, then it is handed to
/// background tracking with a typed `process_state` receipt.
pub(super) const DETACH_STARTUP_GRACE: Duration = Duration::from_millis(1500);
/// How long a sub-agent's own foreground command keeps its call. Nothing
/// else reads the result of a sub-agent's command (its session has no
/// reader), so moving it to the background only hands the run a job to
/// wait on, poll or kill (live 2026-11-12: a counting helper's single loop
/// went to the background at 30 s and the helper killed it at 56 of 60).
/// Kept under the tool watchdog's default (300 s); past it the command
/// goes to the background and the run's stop is held until it ends.
pub(super) const SUB_AGENT_FOREGROUND_WAIT: Duration = Duration::from_secs(240);

/// How long a run waits for its command before moving it to the background:
/// a detached or background launch only for the startup grace; a sub-agent's
/// command (`notifications_reach` false: nothing else reports its result) up
/// to [`SUB_AGENT_FOREGROUND_WAIT`], one call for one long job; anything else
/// the configured initial wait.
pub(super) fn foreground_wait(
    initial: Duration,
    backgrounded: bool,
    notifications_reach: bool,
) -> Duration {
    if backgrounded {
        initial.min(DETACH_STARTUP_GRACE)
    } else if !notifications_reach {
        initial.max(SUB_AGENT_FOREGROUND_WAIT)
    } else {
        initial
    }
}
/// Per-stream tail of startup output returned with a background launch.
pub(super) const PARTIAL_OUTPUT_TAIL_BYTES: usize = 1500;

/// How often a running command's output count is sent as its progress.
pub(super) const COMMAND_OUTPUT_PROGRESS_EVERY: Duration = Duration::from_secs(3);

/// How many lines a command has printed so far on both streams (a last
/// line without its newline counts): what its card shows while it runs.
/// Never the lines themselves: a command's output is its own data, and a
/// line can carry a credential (live 2026-11-13: `drush site:install`
/// printed "User password: …" and the card showed it as "last output").
pub fn printed_lines(stdout: &[u8], stderr: &[u8]) -> u64 {
    let count = |output: &[u8]| {
        let breaks = output.iter().filter(|byte| **byte == b'\n').count() as u64;
        breaks + u64::from(output.last().is_some_and(|byte| *byte != b'\n'))
    };
    count(stdout) + count(stderr)
}

/// A disowned background process (notifier-active, non-detached) that makes no
/// progress (no CPU time, disk I/O, output growth, or process-tree change) for
/// this long is treated as a stall candidate by the heartbeat reaper. The
/// launch-time progress contract can extend this fallback threshold, and a
/// second conclusive sample is normally required before termination. The
/// observed failure: a whole-disk
/// `du -ah ~ | sort | head` that emitted zero bytes and ran for ~11 hours
/// without exiting. Detached processes (dev servers started with `detach=true`)
/// are exempt, and any process that is genuinely working — advancing CPU time,
/// statting files (disk I/O), or streaming output — resets its progress clock
/// and is never reaped on the stall path.
pub const BACKGROUND_IDLE_REAP_SECS: u64 = 300;

/// A total-runtime cap the reaper's tests exercise. The daemon sets none by
/// default (`background_max_runtime_secs = 0`): work is bounded by
/// inactivity, never by how long it takes; an owner who wants a cap sets
/// one, and recognized long workloads treat it as soft while active.
pub const BACKGROUND_MAX_RUNTIME_SECS: u64 = 1200;

/// Why a background process was reaped — drives the user-facing wording and the
/// `terminate_running_process` reason string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReapReason {
    /// No CPU/disk/output progress for the stall threshold.
    Stalled,
    /// Total runtime hit the max-runtime backstop (busy-loop / too-slow).
    MaxRuntime,
}

/// Deterministic launch-time context for background-process supervision.
///
/// This is deliberately derived from the parsed executable/arguments once, when
/// the process starts. The heartbeat never asks an LLM to reinterpret a command
/// and never uses substring matching to decide whether a process is healthy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BackgroundWorkload {
    Generic,
    FilesystemTraversal { broad_or_multi_target: bool },
    BuildOrTest,
    NetworkTransfer,
}

impl BackgroundWorkload {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::FilesystemTraversal {
                broad_or_multi_target: true,
            } => "filesystem_traversal_broad",
            Self::FilesystemTraversal {
                broad_or_multi_target: false,
            } => "filesystem_traversal_bounded",
            Self::BuildOrTest => "build_or_test",
            Self::NetworkTransfer => "network_transfer",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BackgroundProgressContract {
    pub(super) workload: BackgroundWorkload,
    /// Expected long-silent work receives a larger no-progress window.
    pub(super) stall_multiplier: u32,
    /// A threshold crossing is only a suspicion. Require another independent
    /// heartbeat sample before terminating for a stall.
    pub(super) idle_confirmations_required: u8,
    /// Generic commands retain the absolute runtime leak backstop. Recognized
    /// long-running work is never killed merely because wall time elapsed while
    /// objective activity is still visible.
    pub(super) hard_max_runtime: bool,
}

impl Default for BackgroundProgressContract {
    fn default() -> Self {
        Self {
            workload: BackgroundWorkload::Generic,
            stall_multiplier: 1,
            idle_confirmations_required: 2,
            hard_max_runtime: true,
        }
    }
}

impl BackgroundProgressContract {
    pub(super) fn effective_stall_threshold(self, base: Duration) -> Duration {
        base.saturating_mul(self.stall_multiplier)
    }
}

/// One OS resource snapshot for the tracked shell plus all descendants.
/// Process-tree churn is a progress signal beyond CPU/disk/output counters.
/// Process and runnable counts are retained as diagnostic context only: a
/// point-in-time runnable state does not prove that work advanced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ProcessResourceSample {
    pub(super) cpu_ms: u64,
    pub(super) io_bytes: u64,
    pub(super) tree_fingerprint: u64,
    pub(super) process_count: u32,
    pub(super) runnable_count: u32,
}

impl ReapReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            ReapReason::Stalled => "stalled",
            ReapReason::MaxRuntime => "max_runtime",
        }
    }
    pub(super) fn terminate_reason(self) -> &'static str {
        match self {
            ReapReason::Stalled => "idle: no progress",
            ReapReason::MaxRuntime => "idle: max runtime",
        }
    }
}

/// Cross-platform progress signal for one sweep. A process is "making progress"
/// if any cumulative CPU time, disk I/O bytes, or output bytes grew, or if the
/// process tree changed since the last sweep. Pure + injectable so the policy is
/// unit-testable without spawning real processes or sampling the OS.
///
/// `*_now` values that could not be sampled (process gone / OS denied the stat)
/// MUST be passed as the previous value (no change) by the caller, so a missing
/// signal simply contributes nothing rather than being mistaken for progress.
pub(super) fn process_made_progress(
    previous: ProcessResourceSample,
    output_len_prev: usize,
    output_len_now: usize,
    sample_now: ProcessResourceSample,
) -> bool {
    sample_now.cpu_ms > previous.cpu_ms
        || sample_now.io_bytes > previous.io_bytes
        || output_len_now > output_len_prev
        || (previous.tree_fingerprint != 0
            && sample_now.tree_fingerprint != previous.tree_fingerprint)
}

/// Pure decision: should a tracked background process be idle-reaped?
///
/// Reaped only when it is notifier-active (the user was promised a result) and
/// not detached (detached = "survives, requires explicit kill"), AND either:
///   - it has made no progress (no CPU/IO/output advance) for at least
///     `stall_threshold` — truly stalled; or
///   - its total runtime has reached `max_runtime` and its launch-time contract
///     retains the hard wall-time backstop.
///
/// Kept as a free function so the policy is unit-testable without spawning real
/// processes.
pub(super) fn should_idle_reap(
    notifier_active: bool,
    detached: bool,
    no_progress_elapsed: Duration,
    total_runtime: Duration,
    stall_threshold: Duration,
    max_runtime: Duration,
    hard_max_runtime: bool,
) -> bool {
    if !notifier_active || detached {
        return false;
    }
    no_progress_elapsed >= stall_threshold || (hard_max_runtime && total_runtime >= max_runtime)
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
/// - `per_pid`: pid → `(cpu_ms, io_bytes, runnable)` sample for that one pid.
/// - A `visited` set guards against malformed cyclic maps (a real process tree
///   never cycles, but this is cheap insurance against an infinite loop).
/// - A root absent from `per_pid` (e.g. exited between snapshot and lookup)
///   yields no map entry; the caller carries forward the previous sample, so an
///   absent entry is safe.
pub(super) fn sum_subtree_resources(
    roots: &[u32],
    children_of: &HashMap<u32, Vec<u32>>,
    per_pid: &HashMap<u32, (u64, u64, bool)>,
) -> HashMap<u32, ProcessResourceSample> {
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
        let mut tree_fingerprint: u64 = 0;
        let mut runnable_count: u32 = 0;
        while let Some(pid) = stack.pop() {
            if !visited.insert(pid) {
                continue; // cycle guard / already counted
            }
            if visited.len() > MAX_SUBTREE_NODES {
                break;
            }
            if let Some(&(cpu, io, runnable)) = per_pid.get(&pid) {
                cpu_sum = cpu_sum.saturating_add(cpu);
                io_sum = io_sum.saturating_add(io);
                // Order-independent, cheap fingerprint. PID reuse cannot occur
                // inside one live root without a corresponding tree transition.
                tree_fingerprint ^= (pid as u64)
                    .wrapping_mul(0x9E37_79B1_85EB_CA87)
                    .rotate_left(pid % 63);
                if runnable {
                    runnable_count = runnable_count.saturating_add(1);
                }
            }
            if let Some(children) = children_of.get(&pid) {
                for &child in children {
                    if !visited.contains(&child) {
                        stack.push(child);
                    }
                }
            }
        }
        out.insert(
            root,
            ProcessResourceSample {
                cpu_ms: cpu_sum,
                io_bytes: io_sum,
                tree_fingerprint,
                process_count: visited.len().min(u32::MAX as usize) as u32,
                runnable_count,
            },
        );
    }
    out
}
