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

/// The prefix of every session an automated trigger's turn runs in. A
/// trigger's turn is an untrusted source by this typed id the daemon itself
/// writes, never by what the source is called.
pub const TRIGGER_SESSION_PREFIX: &str = "trigger:";

/// The session a trigger from `source` runs its turns in.
pub fn trigger_session_id(source: &str) -> String {
    format!("{TRIGGER_SESSION_PREFIX}{source}")
}

/// Check if a session ID indicates it was triggered by an automated source
/// (e.g., email trigger) rather than direct user interaction via Telegram.
///
/// Daemon-written id markers: every trigger turn runs in a session under
/// [`TRIGGER_SESSION_PREFIX`] (the trigger handler
/// puts it there whatever the source names it). The older `email_trigger`
/// and `event_` ids, and any id with "trigger" in it, still count, so a
/// stored session from before is never treated as the owner's.
pub fn is_trigger_session(session_id: &str) -> bool {
    session_id.starts_with(TRIGGER_SESSION_PREFIX)
        || session_id.contains("trigger")
        || session_id.starts_with("event_")
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

#[cfg(test)]
mod trigger_session_tests {
    #[test]
    fn every_trigger_session_id_is_a_trigger_session() {
        for source in ["email", "webhook", "calendar"] {
            let id = super::trigger_session_id(source);
            assert!(super::is_trigger_session(&id), "{id}");
        }
        assert!(super::is_trigger_session("email_trigger"));
        assert!(!super::is_trigger_session("telegram:synthetic-user-1"));
    }
}
