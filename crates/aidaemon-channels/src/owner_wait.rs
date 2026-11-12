//! How long a turn has waited on its owner. A turn's hang limits leave
//! that time out: waiting on a person is not the turn working. Internal to
//! aidaemon, no stability promise.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::time::Instant;

tokio::task_local! {
    static TURN_OWNER_WAIT: OwnerWaitClock;
}

/// How long a turn has spent waiting for its owner to answer a card. Waiting
/// on a person is not the turn working, so the turn's hang limit leaves it out.
#[derive(Clone, Default)]
pub struct OwnerWaitClock(Arc<Mutex<OwnerWaitState>>);

#[derive(Default)]
struct OwnerWaitState {
    /// Cards open now (a turn's child agents can ask at the same time).
    open: usize,
    /// When the open cards began; overlapping cards count once.
    since: Option<Instant>,
    /// Waits already finished.
    waited: Duration,
}

/// Ends one card's wait when dropped, however the wait ends.
pub struct OwnerWaiting(OwnerWaitClock);

impl OwnerWaitClock {
    /// Total time spent waiting so far, including a wait still open.
    pub fn waited(&self) -> Duration {
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        state.waited + state.since.map_or(Duration::ZERO, |since| since.elapsed())
    }

    /// The clock of the turn running this task, if any (scheduled work has
    /// none, and no turn limit to exempt).
    pub fn current() -> Option<Self> {
        TURN_OWNER_WAIT.try_with(Clone::clone).ok()
    }

    /// Run `future` with `clock` as its turn's clock. Task locals are not
    /// inherited by spawned tasks, so a child agent's task re-enters it here.
    pub async fn within<F: std::future::Future>(clock: Option<Self>, future: F) -> F::Output {
        match clock {
            Some(clock) => TURN_OWNER_WAIT.scope(clock, future).await,
            None => future.await,
        }
    }

    /// Mark a card as waiting on the owner in the current turn.
    pub fn begin_waiting() -> Option<OwnerWaiting> {
        let clock = Self::current()?;
        {
            let mut state = clock.0.lock().unwrap_or_else(|e| e.into_inner());
            state.open += 1;
            if state.open == 1 {
                state.since = Some(Instant::now());
            }
        }
        Some(OwnerWaiting(clock))
    }
}

/// A limit on working time: time spent waiting for the owner does not count.
#[derive(Clone)]
pub struct WorkDeadline {
    started: tokio::time::Instant,
    limit: std::time::Duration,
    owner_wait: OwnerWaitClock,
    /// The clock's total when this deadline started: waits before then do
    /// not extend it.
    waited_at_start: std::time::Duration,
}

impl WorkDeadline {
    pub fn starting_now(limit: std::time::Duration, owner_wait: OwnerWaitClock) -> Self {
        Self {
            started: tokio::time::Instant::now(),
            limit,
            waited_at_start: owner_wait.waited(),
            owner_wait,
        }
    }

    pub fn limit(&self) -> std::time::Duration {
        self.limit
    }

    /// Working time left: time since the start, less the owner waits since
    /// the start.
    pub fn remaining(&self) -> std::time::Duration {
        let waited_since_start = self
            .owner_wait
            .waited()
            .saturating_sub(self.waited_at_start);
        let worked = self.started.elapsed().saturating_sub(waited_since_start);
        self.limit.saturating_sub(worked)
    }

    /// Resolves once the limit has been worked, checking every `check`.
    pub async fn reached(&self, check: std::time::Duration) {
        loop {
            let left = self.remaining();
            if left.is_zero() {
                return;
            }
            tokio::time::sleep(left.min(check)).await;
        }
    }
}

impl Drop for OwnerWaiting {
    fn drop(&mut self) {
        let mut state = self.0 .0.lock().unwrap_or_else(|e| e.into_inner());
        state.open = state.open.saturating_sub(1);
        if state.open == 0 {
            if let Some(since) = state.since.take() {
                state.waited += since.elapsed();
            }
        }
    }
}
