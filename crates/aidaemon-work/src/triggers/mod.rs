//! Events that wake work: what triggers and channels hand the scheduler, the
//! bus that carries them, and the trigger sources (the email trigger behind
//! the `email-trigger` feature).

use std::fmt;

use tokio::sync::broadcast;

#[cfg(feature = "email-trigger")]
mod email;
#[cfg(feature = "email-trigger")]
pub use email::TriggerManager;

pub type EventSender = broadcast::Sender<TriggerEvent>;
pub type EventReceiver = broadcast::Receiver<TriggerEvent>;

/// Create a new event bus (broadcast channel).
pub fn event_bus(capacity: usize) -> (EventSender, EventReceiver) {
    broadcast::channel(capacity)
}

/// Which trigger sources run.
#[derive(Debug, Clone, Default)]
pub struct TriggerSettings {
    pub email: Option<EmailTriggerSettings>,
}

/// An IMAP mailbox to watch for new mail.
#[derive(Clone)]
pub struct EmailTriggerSettings {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub folder: String,
}

impl fmt::Debug for EmailTriggerSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmailTriggerSettings")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("username", &self.username)
            .field("password", &"[REDACTED]")
            .field("folder", &self.folder)
            .finish()
    }
}

/// An event emitted by triggers or channels (not the event-sourcing store).
#[derive(Debug, Clone)]
pub struct TriggerEvent {
    pub source: String,
    pub session_id: String,
    pub content: String,
    /// Whether this event originates from an explicitly trusted source
    /// (e.g., a scheduled task marked `trusted = true` in config).
    pub trusted: bool,
    /// Optional content-free structural signal for matching Autopilot mandates.
    /// The human-readable `content` is never used for wake routing.
    pub mandate_signal: Option<crate::mandates::MandateWakeSignal>,
}
