//! `ApprovalBroker`, the durable boundary around approval delivery, lives in
//! `aidaemon-runtime`; the old paths stay.

pub use aidaemon_runtime::approvals::broker::ApprovalBroker;

#[cfg(test)]
use aidaemon_channels::owner_wait::OwnerWaitClock;
pub(crate) use aidaemon_channels::owner_wait::ProgressClock;

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_channels::owner_wait::WorkDeadline;
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::sync::mpsc;

    use crate::events::EventStore;
    use crate::events::EventStoreForTests as _;
    use crate::tools::command_risk::{PermissionMode, RiskLevel};
    use crate::tools::terminal::ApprovalRequest;
    use crate::types::{ApprovalKind, ApprovalResponse};

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
                expires_at_ms: None,
                card: None,
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
                expires_at_ms: None,
                card: None,
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
                expires_at_ms: None,
                card: None,
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
    /// Waits that ended before a deadline started do not extend it.
    #[tokio::test(start_paused = true)]
    async fn earlier_owner_waits_do_not_extend_a_later_deadline() {
        let clock = OwnerWaitClock::default();
        OwnerWaitClock::within(Some(clock.clone()), async {
            let earlier = OwnerWaitClock::begin_waiting();
            tokio::time::advance(Duration::from_secs(60)).await;
            drop(earlier);
            let deadline = WorkDeadline::starting_now(Duration::from_secs(30), clock.clone());
            tokio::time::advance(Duration::from_secs(10)).await;
            assert_eq!(deadline.remaining(), Duration::from_secs(20));
            let later = OwnerWaitClock::begin_waiting();
            tokio::time::advance(Duration::from_secs(5)).await;
            drop(later);
            assert_eq!(deadline.remaining(), Duration::from_secs(20));
            tokio::time::advance(Duration::from_secs(20)).await;
            assert!(deadline.remaining().is_zero());
        })
        .await;
    }

    /// A wait already open when a deadline starts counts only from then.
    #[tokio::test(start_paused = true)]
    async fn an_open_owner_wait_counts_only_after_a_deadline_starts() {
        let clock = OwnerWaitClock::default();
        OwnerWaitClock::within(Some(clock.clone()), async {
            let open = OwnerWaitClock::begin_waiting();
            tokio::time::advance(Duration::from_secs(60)).await;
            let deadline = WorkDeadline::starting_now(Duration::from_secs(30), clock.clone());
            tokio::time::advance(Duration::from_secs(5)).await;
            drop(open);
            tokio::time::advance(Duration::from_secs(10)).await;
            assert_eq!(deadline.remaining(), Duration::from_secs(20));
        })
        .await;
    }

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
                expires_at_ms: None,
                card: None,
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
