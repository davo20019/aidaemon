//! The per-owner event bus: one daemon event fans out to every connected
//! client, and a durable event is first appended to the `client_events` log
//! (which gives it its `seq`), so a client that reconnects can read what it
//! missed. Appending and broadcasting happen under one lock, so live frames
//! leave in `seq` order: a subscriber that subscribes first and then reads
//! the log never sees a gap.

use std::sync::Arc;

use aidaemon_client_protocol::{Event, ServerFrame};
use aidaemon_store_sqlite::SqliteClientEventLog;
use tokio::sync::{Mutex, broadcast};
use tracing::warn;

/// v1 has one owner; the log is keyed by owner for when it has more.
pub const OWNER: &str = "owner";
/// The most frames a slow subscriber may fall behind before it re-reads
/// the log.
const LIVE_BUFFER: usize = 1024;

pub(crate) fn now_ms() -> u64 {
    u64::try_from(chrono::Utc::now().timestamp_millis()).unwrap_or(0)
}

/// What an event is about, so the log can find it again: the approval it
/// asks or settles.
fn ref_id(event: &Event) -> Option<&str> {
    match event {
        Event::ApprovalRequested { approval } => Some(&approval.id),
        Event::ApprovalResolved { approval_id, .. } => Some(approval_id),
        Event::TurnStarted { turn_id, .. }
        | Event::TurnCompleted { turn_id, .. }
        | Event::UserActionRequired { turn_id, .. } => Some(turn_id),
        Event::TaskUpdated { task } => Some(&task.id),
        _ => None,
    }
}

pub struct EventBus {
    log: SqliteClientEventLog,
    live: broadcast::Sender<Arc<ServerFrame>>,
    append: Mutex<()>,
}

impl EventBus {
    pub fn new(log: SqliteClientEventLog) -> Self {
        let (live, _) = broadcast::channel(LIVE_BUFFER);
        Self {
            log,
            live,
            append: Mutex::new(()),
        }
    }

    pub fn log(&self) -> &SqliteClientEventLog {
        &self.log
    }

    /// Frames from now on.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<ServerFrame>> {
        self.live.subscribe()
    }

    /// Log (when durable) and broadcast `event`. Returns its `seq`.
    pub async fn publish(&self, session_id: Option<String>, event: Event) -> Option<u64> {
        let at_ms = now_ms();
        // The one redaction boundary every frame to the owner's app passes.
        let event = owner_event(event);
        if !event.is_durable() {
            let _ = self.live.send(Arc::new(ServerFrame::Event {
                seq: None,
                at_ms,
                session_id,
                event,
            }));
            return None;
        }
        let payload = match serde_json::to_string(&event) {
            Ok(payload) => payload,
            Err(error) => {
                warn!(%error, "Client event could not be encoded");
                return None;
            }
        };
        let _order = self.append.lock().await;
        let seq = match self
            .log
            .append(
                OWNER,
                at_ms,
                session_id.as_deref(),
                event.kind(),
                ref_id(&event),
                &payload,
            )
            .await
        {
            Ok(seq) => seq,
            Err(error) => {
                // Shown live but not kept: a resuming client gets a snapshot
                // only if it falls outside the log, so say so loudly.
                warn!(%error, kind = event.kind(), "Client event could not be logged");
                let _ = self.live.send(Arc::new(ServerFrame::Event {
                    seq: None,
                    at_ms,
                    session_id,
                    event,
                }));
                return None;
            }
        };
        let _ = self.live.send(Arc::new(ServerFrame::Event {
            seq: Some(seq),
            at_ms,
            session_id,
            event,
        }));
        Some(seq)
    }

    /// Up to `limit` logged frames after `seq`, oldest first.
    pub async fn after(&self, seq: u64, limit: u32) -> anyhow::Result<Vec<ServerFrame>> {
        let rows = self.log.after(OWNER, seq, limit).await?;
        Ok(rows
            .into_iter()
            .map(|row| ServerFrame::Event {
                seq: Some(row.seq),
                at_ms: row.at_ms,
                session_id: row.session_id,
                // A row this build cannot read is passed on as unknown, so
                // the client's `seq` still advances over it.
                event: serde_json::from_str(&row.payload).unwrap_or(Event::Unknown),
            })
            .collect())
    }

    /// The oldest and newest `seq` kept.
    pub async fn bounds(&self) -> anyhow::Result<Option<(u64, u64)>> {
        Ok(self.log.bounds(OWNER).await?)
    }

    /// The newest `seq`, 0 for an empty log.
    pub async fn latest(&self) -> u64 {
        self.bounds()
            .await
            .ok()
            .flatten()
            .map_or(0, |(_, latest)| latest)
    }

    /// Whether a client that applied everything up to `seq` can resume
    /// from the log (else it needs a snapshot).
    pub async fn can_resume(&self, seq: u64) -> bool {
        match self.bounds().await {
            Ok(Some((oldest, latest))) => seq <= latest && seq.saturating_add(1) >= oldest,
            // Nothing logged: only a client that has seen nothing is in step.
            Ok(None) => seq == 0,
            Err(_) => false,
        }
    }
}

/// `event` with every string in it through the channels' one redaction
/// boundary ([`aidaemon_channels::outbound`]).
fn owner_event(event: Event) -> Event {
    let Ok(value) = serde_json::to_value(&event) else {
        return event;
    };
    let redacted = aidaemon_channels::outbound::redact_body(&value);
    if redacted == value {
        return event;
    }
    serde_json::from_value(redacted).unwrap_or(event)
}
