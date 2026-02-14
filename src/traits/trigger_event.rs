/// An event emitted by triggers or channels (not the event-sourcing store).
#[derive(Debug, Clone)]
pub struct TriggerEvent {
    pub source: String,
    pub session_id: String,
    pub content: String,
    /// Whether this event originates from an explicitly trusted source
    /// (e.g., a scheduled task marked `trusted = true` in config).
    pub trusted: bool,
}

