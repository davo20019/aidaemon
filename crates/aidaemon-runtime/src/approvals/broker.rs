//! `ApprovalBroker` — the durable boundary around approval delivery.
//!
//! The broker is created once at startup from the channel's transmit half and
//! handed to every tool/struct that needs to request command approval, replacing
//! the previously hand-threaded raw `mpsc::Sender<ApprovalRequest>`. When an
//! event store is attached, the broker persists a versioned request before
//! delivery and a matching resolution before forwarding the answer. This makes
//! the human wait state reconstructable instead of keeping it solely in an
//! in-memory oneshot channel.

use std::sync::Arc;

use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::SendError;
use tracing::warn;
use uuid::Uuid;

use aidaemon_channels::hub::ApprovalRequest;
use aidaemon_events::{
    Event, EventStore, EventType, InteractionRequestedData, InteractionResolvedData,
};
use aidaemon_tools::semantics::{ApprovalKind, ApprovalResponse};

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
        let action_sha256 = hex::encode(Sha256::digest(req.command.as_bytes()));
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
