//! Process-wide agent-activity signal for background-work deferral.
//!
//! The local llama.cpp server multiplexes two KV slots: interactive turns pin
//! slot 0; everything else (goal runs, spawned specialists, the structured
//! answer pass, the background memory pipeline) shares slot 1. Memory-pipeline
//! jobs interleaving with an agent task evict the task's warm prefix and steal
//! prefill/decode throughput — measured 2026-07-03: a scheduled goal run went
//! fully cold three times mid-run (cached=9) because consolidation/extraction
//! fired in its idle gaps, inflating the run's fresh-token budget charge ~5x
//! until it tripped its per-run ceiling.
//!
//! The pipeline's work is inherently deferrable (consolidating a conversation
//! ten minutes later loses nothing), so pipeline jobs consult this gate and
//! yield while any agent task is in flight. Correctness-critical heartbeat
//! work (watchdogs, goal dispatch, orphan reclaim) must NEVER consult it —
//! the watchdog's whole job is to run while tasks are active.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

static ACTIVE_AGENT_TASKS: AtomicUsize = AtomicUsize::new(0);

/// RAII marker for an in-flight agent task (interactive turn, goal run, or
/// spawned specialist). Hold it for the duration of the work.
pub struct AgentActivityGuard(());

impl AgentActivityGuard {
    pub fn acquire() -> Self {
        ACTIVE_AGENT_TASKS.fetch_add(1, Ordering::SeqCst);
        Self(())
    }
}

impl Drop for AgentActivityGuard {
    fn drop(&mut self) {
        ACTIVE_AGENT_TASKS.fetch_sub(1, Ordering::SeqCst);
    }
}

pub fn agent_busy() -> bool {
    ACTIVE_AGENT_TASKS.load(Ordering::SeqCst) > 0
}

/// Heartbeat-tick decision: should a defer-flagged job skip this tick?
/// Pure so the policy is unit-testable: defer only when the agent is busy AND
/// the job is not badly overdue (starvation cap: once a job has waited 3x its
/// interval, it runs regardless — steady traffic must not starve consolidation
/// forever).
pub fn should_defer_heartbeat_job(
    defer_flag: bool,
    busy: bool,
    elapsed_since_last_run: Option<Duration>,
    interval: Duration,
) -> bool {
    if !defer_flag || !busy {
        return false;
    }
    match elapsed_since_last_run {
        // Never ran yet (daemon start): defer until first idle moment.
        None => true,
        Some(elapsed) => elapsed < interval.saturating_mul(3),
    }
}

/// Event-driven jobs (summarization, progressive extraction) wait for an idle
/// moment instead of skipping — their trigger won't recur. Returns `true` if
/// idle was reached, `false` if the cap expired (caller proceeds anyway; the
/// cap only bounds the wait, it never cancels the work).
pub async fn wait_until_agent_idle(max_wait: Duration, poll: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + max_wait;
    loop {
        if !agent_busy() {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(poll).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_raises_and_lowers_busy_counter() {
        let before = ACTIVE_AGENT_TASKS.load(Ordering::SeqCst);
        let g1 = AgentActivityGuard::acquire();
        let g2 = AgentActivityGuard::acquire();
        assert_eq!(ACTIVE_AGENT_TASKS.load(Ordering::SeqCst), before + 2);
        assert!(agent_busy());
        drop(g1);
        assert_eq!(ACTIVE_AGENT_TASKS.load(Ordering::SeqCst), before + 1);
        drop(g2);
        assert_eq!(ACTIVE_AGENT_TASKS.load(Ordering::SeqCst), before);
    }

    #[test]
    fn defer_policy_is_flag_and_busy_scoped_with_starvation_cap() {
        let i = Duration::from_secs(600);
        // Not flagged, or not busy: never defer.
        assert!(!should_defer_heartbeat_job(false, true, None, i));
        assert!(!should_defer_heartbeat_job(true, false, None, i));
        // Flagged + busy: defer while fresh...
        assert!(should_defer_heartbeat_job(true, true, None, i));
        assert!(should_defer_heartbeat_job(
            true,
            true,
            Some(Duration::from_secs(1200)),
            i
        ));
        // ...but a job overdue by 3x its interval runs regardless.
        assert!(!should_defer_heartbeat_job(
            true,
            true,
            Some(Duration::from_secs(1800)),
            i
        ));
    }

    #[tokio::test]
    async fn wait_until_idle_times_out_while_guard_held() {
        let _g = AgentActivityGuard::acquire();
        let reached_idle =
            wait_until_agent_idle(Duration::from_millis(60), Duration::from_millis(10)).await;
        assert!(!reached_idle, "must report timeout while busy");
    }
}
