//! The coding-agent asks being judged now. A card an ask raises names the
//! run that asked ([`asker_of`]); the judge in `tools::agent_mode` records
//! the ask for the judgment's length ([`begin`] / [`end`]).

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// A coding agent's run, as a card it raises names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asker {
    /// The run's id.
    pub run: String,
    /// Its agent ("Codex").
    pub agent: String,
}

/// The asks being judged now, by the id of the daemon call each is judged
/// as (unique per ask).
static ASKING: LazyLock<Mutex<HashMap<String, Asker>>> = LazyLock::new(Default::default);

/// Record that daemon call `call_id` is `asker`'s ask, while it is judged.
pub fn begin(call_id: &str, asker: Asker) {
    ASKING
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(call_id.to_string(), asker);
}

/// The judgment of daemon call `call_id` is over.
pub fn end(call_id: &str) {
    ASKING
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .remove(call_id);
    HELD_FULL_ACCESS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .remove(call_id);
}

/// The run whose agent's ask is daemon call `call_id`, while it is judged.
pub fn asker_of(call_id: &str) -> Option<Asker> {
    ASKING
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(call_id)
        .cloned()
}

/// The asks judged now of agents the owner gave full access that a run's
/// device ceiling held to a checked mode, by daemon call id. Such an ask at
/// or below the ceiling gets the owner's full-access answer (no card); one
/// above it still asks on the phone (`approvals::device_ceiling`).
static HELD_FULL_ACCESS: LazyLock<Mutex<std::collections::HashSet<String>>> =
    LazyLock::new(Default::default);

/// Daemon call `call_id` is an ask of an agent held from full access, while
/// it is judged (cleared by [`end`]).
pub fn begin_held(call_id: &str) {
    HELD_FULL_ACCESS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(call_id.to_string());
}

/// Whether daemon call `call_id` is an ask of an agent the owner gave full
/// access and the device ceiling held to a checked mode.
pub fn held_from_full_access(call_id: &str) -> bool {
    HELD_FULL_ACCESS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .contains(call_id)
}
