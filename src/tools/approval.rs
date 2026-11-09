//! `ApprovalBroker` — the durable boundary around approval delivery.
//!
//! The broker is created once at startup from the channel's transmit half and
//! handed to every tool/struct that needs to request command approval, replacing
//! the previously hand-threaded raw `mpsc::Sender<ApprovalRequest>`. When an
//! event store is attached, the broker persists a versioned request before
//! delivery and a matching resolution before forwarding the answer. This makes
//! the human wait state reconstructable instead of keeping it solely in an
//! in-memory oneshot channel.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::SendError;
use tokio::time::Instant;
use tracing::warn;
use uuid::Uuid;

use super::terminal::ApprovalRequest;
use crate::events::{
    Event, EventStore, EventType, InteractionRequestedData, InteractionResolvedData,
};
use crate::types::{ApprovalKind, ApprovalResponse};

tokio::task_local! {
    static TURN_OWNER_WAIT: OwnerWaitClock;
}

/// How long a turn has spent waiting for its owner to answer a card. Waiting
/// on a person is not the turn working, so the turn's hang limit leaves it out.
#[derive(Clone, Default)]
pub(crate) struct OwnerWaitClock(Arc<Mutex<OwnerWaitState>>);

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
pub(crate) struct OwnerWaiting(OwnerWaitClock);

impl OwnerWaitClock {
    /// Total time spent waiting so far, including a wait still open.
    pub(crate) fn waited(&self) -> Duration {
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        state.waited + state.since.map_or(Duration::ZERO, |since| since.elapsed())
    }

    /// The clock of the turn running this task, if any (scheduled work has
    /// none, and no turn limit to exempt).
    pub(crate) fn current() -> Option<Self> {
        TURN_OWNER_WAIT.try_with(Clone::clone).ok()
    }

    /// Run `future` with `clock` as its turn's clock. Task locals are not
    /// inherited by spawned tasks, so a child agent's task re-enters it here.
    pub(crate) async fn within<F: std::future::Future>(
        clock: Option<Self>,
        future: F,
    ) -> F::Output {
        match clock {
            Some(clock) => TURN_OWNER_WAIT.scope(clock, future).await,
            None => future.await,
        }
    }

    /// Mark a card as waiting on the owner in the current turn.
    pub(crate) fn begin_waiting() -> Option<OwnerWaiting> {
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
pub(crate) struct WorkDeadline {
    started: tokio::time::Instant,
    limit: std::time::Duration,
    owner_wait: OwnerWaitClock,
}

impl WorkDeadline {
    pub(crate) fn starting_now(limit: std::time::Duration, owner_wait: OwnerWaitClock) -> Self {
        Self {
            started: tokio::time::Instant::now(),
            limit,
            owner_wait,
        }
    }

    pub(crate) fn limit(&self) -> std::time::Duration {
        self.limit
    }

    /// Working time left.
    pub(crate) fn remaining(&self) -> std::time::Duration {
        let worked = self
            .started
            .elapsed()
            .saturating_sub(self.owner_wait.waited());
        self.limit.saturating_sub(worked)
    }

    /// Resolves once the limit has been worked, checking every `check`.
    pub(crate) async fn reached(&self, check: std::time::Duration) {
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

/// Named wrapper around the approval channel's transmit half.
///
/// Cloning is cheap (clones the inner `mpsc::Sender`) and yields a handle to the
/// same underlying channel, matching the prior raw-sender behavior.
#[derive(Clone)]
pub struct ApprovalBroker {
    tx: mpsc::Sender<ApprovalRequest>,
    event_store: Option<Arc<EventStore>>,
}

impl ApprovalBroker {
    /// Wrap an existing approval channel sender.
    pub fn new(tx: mpsc::Sender<ApprovalRequest>) -> Self {
        Self {
            tx,
            event_store: None,
        }
    }

    /// Attach the canonical event store. Kept as a builder so isolated tool
    /// tests can continue using an in-memory-only broker.
    pub fn with_event_store(mut self, event_store: Arc<EventStore>) -> Self {
        self.event_store = Some(event_store);
        self
    }

    /// Send an approval request to the channel, delegating to the inner sender.
    ///
    /// Semantics are identical to `mpsc::Sender::send`: awaits available capacity
    /// and returns `Err(SendError)` if the receiver has been dropped.
    pub async fn send(&self, req: ApprovalRequest) -> Result<(), SendError<ApprovalRequest>> {
        let Some(event_store) = self.event_store.clone() else {
            return self.tx.send(req).await;
        };

        let interaction_id = format!("int_{}", Uuid::new_v4().simple());
        let action_sha256 = format!("{:x}", Sha256::digest(req.command.as_bytes()));
        let requested = InteractionRequestedData {
            schema_version: InteractionRequestedData::SCHEMA_VERSION,
            interaction_id: interaction_id.clone(),
            interaction_kind: interaction_kind(&req.kind).to_string(),
            action: req.command.clone(),
            action_sha256,
            risk_level: req.risk_level.to_string(),
            warnings: req.warnings.clone(),
            permission_mode: req.permission_mode.to_string(),
        };
        if let Err(error) = event_store
            .append(Event::new(
                &req.session_id,
                EventType::InteractionRequested,
                serde_json::to_value(requested).expect("interaction request serializes"),
            ))
            .await
        {
            // The approval response is an execution capability. If its request
            // cannot be made durable, do not create a process-local path that
            // could authorize an effect and then disappear on restart.
            warn!(
                %error,
                session_id = %req.session_id,
                %interaction_id,
                "Refusing non-durable approval interaction"
            );
            return Err(SendError(req));
        }

        let ApprovalRequest {
            command,
            session_id,
            risk_level,
            warnings,
            permission_mode,
            response_tx,
            kind,
        } = req;

        let (broker_response_tx, broker_response_rx) = tokio::sync::oneshot::channel();
        let forwarded = ApprovalRequest {
            command,
            session_id: session_id.clone(),
            risk_level,
            warnings,
            permission_mode,
            response_tx: broker_response_tx,
            kind,
        };
        if let Err(error) = self.tx.send(forwarded).await {
            if let Err(persist_error) =
                persist_resolution(&event_store, &session_id, &interaction_id, "unavailable").await
            {
                warn!(
                    error = %persist_error,
                    %session_id,
                    %interaction_id,
                    "Failed to persist unavailable approval resolution"
                );
            }
            drop(response_tx);
            return Err(error);
        }

        let mut response_tx = response_tx;
        tokio::spawn(async move {
            // A requester that stops waiting (a killed or cancelled turn)
            // drops the delivered card's receiver with this select, so the
            // channel expires the card instead of leaving live buttons.
            let answer = tokio::select! {
                biased;
                answer = broker_response_rx => Some(answer),
                _ = response_tx.closed() => None,
            };
            match answer {
                Some(Ok(response)) => {
                    if let Err(error) = persist_resolution(
                        &event_store,
                        &session_id,
                        &interaction_id,
                        approval_resolution(&response),
                    )
                    .await
                    {
                        warn!(
                            %error,
                            %session_id,
                            %interaction_id,
                            "Refusing non-durable approval resolution"
                        );
                        drop(response_tx);
                    } else {
                        let _ = response_tx.send(response);
                    }
                }
                unanswered => {
                    let resolution = if unanswered.is_none() {
                        "abandoned"
                    } else {
                        "unavailable"
                    };
                    if let Err(error) =
                        persist_resolution(&event_store, &session_id, &interaction_id, resolution)
                            .await
                    {
                        warn!(
                            %error,
                            %session_id,
                            %interaction_id,
                            "Failed to persist unavailable approval resolution"
                        );
                    }
                    drop(response_tx);
                }
            }
        });
        Ok(())
    }
}

fn interaction_kind(kind: &ApprovalKind) -> &'static str {
    match kind {
        ApprovalKind::Command => "command_approval",
        ApprovalKind::CommandOnce => "command_approval_once",
        ApprovalKind::GoalConfirmation => "goal_confirmation",
        ApprovalKind::AutopilotConfirmation => "autopilot_confirmation",
    }
}

fn approval_resolution(response: &ApprovalResponse) -> &'static str {
    match response {
        ApprovalResponse::AllowOnce => "approved_once",
        ApprovalResponse::AllowSession => "approved_session",
        ApprovalResponse::AllowAlways => "approved_always",
        ApprovalResponse::Deny => "denied",
    }
}

async fn persist_resolution(
    event_store: &EventStore,
    session_id: &str,
    interaction_id: &str,
    resolution: &str,
) -> anyhow::Result<()> {
    let data = InteractionResolvedData {
        schema_version: InteractionResolvedData::SCHEMA_VERSION,
        interaction_id: interaction_id.to_string(),
        resolution: resolution.to_string(),
    };
    event_store
        .append(Event::new(
            session_id,
            EventType::InteractionResolved,
            serde_json::to_value(data).expect("interaction resolution serializes"),
        ))
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::command_risk::{PermissionMode, RiskLevel};

    #[tokio::test]
    async fn waiting_for_the_owner_does_not_use_up_a_work_deadline() {
        use std::time::Duration;
        let clock = OwnerWaitClock::default();
        OwnerWaitClock::within(Some(clock.clone()), async {
            let deadline = WorkDeadline::starting_now(Duration::from_millis(300), clock.clone());
            let waiting = OwnerWaitClock::begin_waiting();
            tokio::time::sleep(Duration::from_millis(500)).await;
            assert!(
                !deadline.remaining().is_zero(),
                "a card was open throughout"
            );
            drop(waiting);
            tokio::time::timeout(
                Duration::from_secs(5),
                deadline.reached(Duration::from_millis(10)),
            )
            .await
            .expect("working time runs out once the card closes");
        })
        .await;
    }

    #[tokio::test]
    async fn durable_broker_projects_pending_then_resolved_interaction() {
        let db = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", db.path().display()))
            .await
            .unwrap();
        let store = Arc::new(EventStore::new(pool).await.unwrap());
        let (tx, mut rx) = mpsc::channel(1);
        let broker = ApprovalBroker::new(tx).with_event_store(store.clone());
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        broker
            .send(ApprovalRequest {
                command: "deploy release-42".to_string(),
                session_id: "session-approval".to_string(),
                risk_level: RiskLevel::High,
                warnings: vec!["external write".to_string()],
                permission_mode: PermissionMode::Cautious,
                response_tx,
                kind: ApprovalKind::Command,
            })
            .await
            .unwrap();

        let pending = store
            .get_pending_interactions("session-approval")
            .await
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].action, "deploy release-42");
        assert_eq!(pending[0].interaction_kind, "command_approval");
        assert_eq!(pending[0].action_sha256.len(), 64);

        let delivered = rx.recv().await.unwrap();
        delivered
            .response_tx
            .send(ApprovalResponse::AllowOnce)
            .unwrap();
        assert!(matches!(
            response_rx.await.unwrap(),
            ApprovalResponse::AllowOnce
        ));
        assert!(store
            .get_pending_interactions("session-approval")
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn durable_broker_refuses_request_when_request_cannot_be_persisted() {
        let db = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", db.path().display()))
            .await
            .unwrap();
        let store = Arc::new(EventStore::new(pool).await.unwrap());
        store.pool().close().await;
        let (tx, mut rx) = mpsc::channel(1);
        let broker = ApprovalBroker::new(tx).with_event_store(store);
        let (response_tx, _response_rx) = tokio::sync::oneshot::channel();

        let result = broker
            .send(ApprovalRequest {
                command: "deploy release-42".to_string(),
                session_id: "session-approval".to_string(),
                risk_level: RiskLevel::High,
                warnings: vec!["external write".to_string()],
                permission_mode: PermissionMode::Cautious,
                response_tx,
                kind: ApprovalKind::Command,
            })
            .await;

        assert!(result.is_err());
        assert!(rx.try_recv().is_err(), "non-durable request was delivered");
    }

    #[tokio::test]
    async fn durable_broker_refuses_response_when_resolution_cannot_be_persisted() {
        let db = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", db.path().display()))
            .await
            .unwrap();
        let store = Arc::new(EventStore::new(pool).await.unwrap());
        let (tx, mut rx) = mpsc::channel(1);
        let broker = ApprovalBroker::new(tx).with_event_store(store.clone());
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        broker
            .send(ApprovalRequest {
                command: "deploy release-42".to_string(),
                session_id: "session-approval".to_string(),
                risk_level: RiskLevel::High,
                warnings: vec!["external write".to_string()],
                permission_mode: PermissionMode::Cautious,
                response_tx,
                kind: ApprovalKind::Command,
            })
            .await
            .unwrap();
        let delivered = rx.recv().await.unwrap();
        store.pool().close().await;
        delivered
            .response_tx
            .send(ApprovalResponse::AllowOnce)
            .unwrap();

        assert!(
            response_rx.await.is_err(),
            "non-durable resolution authorized execution"
        );
    }

    /// A turn killed while its card is open must close the delivered card,
    /// so the channel expires it instead of leaving buttons nothing listens to.
    #[tokio::test(start_paused = true)]
    async fn overlapping_owner_waits_count_once() {
        let clock = OwnerWaitClock::default();
        OwnerWaitClock::within(Some(clock.clone()), async {
            let first = OwnerWaitClock::begin_waiting();
            tokio::time::advance(Duration::from_secs(10)).await;
            let second = OwnerWaitClock::begin_waiting();
            tokio::time::advance(Duration::from_secs(10)).await;
            drop(first);
            tokio::time::advance(Duration::from_secs(10)).await;
            drop(second);
            tokio::time::advance(Duration::from_secs(10)).await;
        })
        .await;
        assert_eq!(clock.waited(), Duration::from_secs(30));
    }

    #[tokio::test]
    async fn a_requester_that_stops_waiting_closes_the_delivered_card() {
        let db = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", db.path().display()))
            .await
            .unwrap();
        let store = Arc::new(EventStore::new(pool).await.unwrap());
        let (tx, mut rx) = mpsc::channel(1);
        let broker = ApprovalBroker::new(tx).with_event_store(store.clone());
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        broker
            .send(ApprovalRequest {
                command: "activate a goal".to_string(),
                session_id: "session-abandoned".to_string(),
                risk_level: RiskLevel::High,
                warnings: Vec::new(),
                permission_mode: PermissionMode::Cautious,
                response_tx,
                kind: ApprovalKind::GoalConfirmation,
            })
            .await
            .unwrap();
        let mut delivered = rx.recv().await.unwrap();
        drop(response_rx);

        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            delivered.response_tx.closed(),
        )
        .await
        .expect("the delivered card outlived the requester");
    }
}
