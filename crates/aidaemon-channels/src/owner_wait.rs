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
pub struct OwnerWaitClock(Arc<Mutex<OwnerWaitState>>, Option<Box<OwnerWaitClock>>);

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
pub struct OwnerWaiting {
    clock: OwnerWaitClock,
    /// The same wait on the enclosing clock, ended with this one.
    _parent: Option<Box<OwnerWaiting>>,
}

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

    /// A clock of its own for one part of the turn (a sub-agent): its waits
    /// are measured apart from its siblings', and count on this clock too.
    pub fn nested(&self) -> Self {
        Self(Arc::default(), Some(Box::new(self.clone())))
    }

    /// Mark a card as waiting on the owner in the current turn.
    pub fn begin_waiting() -> Option<OwnerWaiting> {
        Some(Self::current()?.begin())
    }

    fn begin(&self) -> OwnerWaiting {
        {
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            state.open += 1;
            if state.open == 1 {
                state.since = Some(Instant::now());
            }
        }
        OwnerWaiting {
            clock: self.clone(),
            _parent: self.1.as_ref().map(|parent| Box::new(parent.begin())),
        }
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

/// When a run last made progress: the end of a model or tool call, or any
/// event of its own. A call in flight is progress being made (each call has
/// its own bound: the model call timeout, a tool's timeout or stall
/// watchdog), and time spent waiting for the owner never counts as idle. A
/// child agent is stopped only after it has been idle for its limit, never
/// for how long it has been running: long legitimate work (an install that
/// takes minutes) keeps making calls.
#[derive(Clone)]
pub struct ProgressClock(Arc<Mutex<ProgressState>>, OwnerWaitClock);

struct ProgressState {
    /// The last progress: a call began or ended, or the run sent an event.
    last: Instant,
    /// The owner clock's total at `last`: waits after it are not idle.
    waited_at_last: Duration,
    /// Calls running now.
    in_flight: usize,
}

/// Marks one call as running until dropped, however the call ends.
pub struct InFlight(ProgressClock);

impl ProgressClock {
    pub fn starting_now(owner_wait: OwnerWaitClock) -> Self {
        Self(
            Arc::new(Mutex::new(ProgressState {
                last: Instant::now(),
                waited_at_last: owner_wait.waited(),
                in_flight: 0,
            })),
            owner_wait,
        )
    }

    /// Record progress now.
    pub fn touch(&self) {
        let waited = self.1.waited();
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        state.last = Instant::now();
        state.waited_at_last = waited;
    }

    /// A call begins; it is progress until the returned guard drops.
    pub fn begin(&self) -> InFlight {
        self.touch();
        self.0.lock().unwrap_or_else(|e| e.into_inner()).in_flight += 1;
        InFlight(self.clone())
    }

    /// How long the run has made no progress: zero while a call runs, else
    /// the time since the last progress less the owner waits since then.
    pub fn idle(&self) -> Duration {
        let waited = self.1.waited();
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.in_flight > 0 {
            return Duration::ZERO;
        }
        state
            .last
            .elapsed()
            .saturating_sub(waited.saturating_sub(state.waited_at_last))
    }

    /// Resolves once the run has been idle for `limit`, checking every
    /// `check`.
    pub async fn idle_for(&self, limit: Duration, check: Duration) {
        loop {
            let idle = self.idle();
            if idle >= limit {
                return;
            }
            tokio::time::sleep((limit - idle).min(check)).await;
        }
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        {
            let mut state = self.0 .0.lock().unwrap_or_else(|e| e.into_inner());
            state.in_flight = state.in_flight.saturating_sub(1);
        }
        self.0.touch();
    }
}

impl Drop for OwnerWaiting {
    fn drop(&mut self) {
        let mut state = self.clock.0.lock().unwrap_or_else(|e| e.into_inner());
        state.open = state.open.saturating_sub(1);
        if state.open == 0 {
            if let Some(since) = state.since.take() {
                state.waited += since.elapsed();
            }
        }
    }
}

#[cfg(test)]
mod progress_tests {
    use super::*;

    const STEP: Duration = Duration::from_millis(60);

    #[tokio::test]
    async fn a_call_in_flight_is_never_idle_and_its_end_restarts_the_clock() {
        let clock = ProgressClock::starting_now(OwnerWaitClock::default());
        tokio::time::sleep(STEP).await;
        assert!(clock.idle() >= STEP);
        let call = clock.begin();
        tokio::time::sleep(STEP * 3).await;
        assert_eq!(clock.idle(), Duration::ZERO, "a long call is progress");
        drop(call);
        assert!(clock.idle() < STEP);
        tokio::time::sleep(STEP).await;
        assert!(clock.idle() >= STEP);
        clock.touch();
        assert!(clock.idle() < STEP);
    }

    /// A sub-agent's own clock measures only its waits, and they still
    /// count for the turn it belongs to.
    #[tokio::test]
    async fn a_nested_clock_measures_its_own_waits_and_counts_for_the_turn() {
        let turn = OwnerWaitClock::default();
        let helper = turn.nested();
        let sibling = turn.nested();
        OwnerWaitClock::within(Some(helper.clone()), async {
            let card = OwnerWaitClock::begin_waiting();
            tokio::time::sleep(STEP * 2).await;
            drop(card);
        })
        .await;
        assert!(helper.waited() >= STEP * 2, "{:?}", helper.waited());
        assert!(turn.waited() >= STEP * 2, "{:?}", turn.waited());
        assert_eq!(sibling.waited(), Duration::ZERO);
    }

    #[tokio::test]
    async fn waiting_on_the_owner_is_not_idle() {
        let owner = OwnerWaitClock::default();
        let clock = ProgressClock::starting_now(owner.clone());
        OwnerWaitClock::within(Some(owner), async {
            let card = OwnerWaitClock::begin_waiting();
            tokio::time::sleep(STEP * 4).await;
            drop(card);
        })
        .await;
        assert!(clock.idle() < STEP * 2, "{:?}", clock.idle());
    }
}
