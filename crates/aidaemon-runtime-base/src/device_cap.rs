//! The device ceiling of one request's runs: a message from a device the
//! daemon does not trust like the owner's own app (an e-ink reader paired
//! over a public bridge, typed by the client gateway as
//! `ChannelContext::sender_ceiling`) caps the run it starts or steers at
//! Medium for the rest of that run's life, and every sub-agent of its tree
//! with it.
//!
//! The cap is a live, shared flag read before every call (as the chat's
//! `/readonly` ceiling is): a steer from such a device into a run already
//! going caps the calls that run makes from then on. It only ever rises.
//! The kernel hook that enforces it and the resolver that asks the owner
//! live in `aidaemon-runtime-core` (`approvals::device_ceiling`).

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use aidaemon_core::{ApprovalRisk, RunId};

/// One call of one run, as the cap's records key it: the run and the
/// call's own key (its id, tool and arguments, so a provider that reuses
/// call ids never carries one call's record over to another).
type CallKey = (RunId, String);

#[derive(Debug, Default)]
struct DeviceCapState {
    capped: AtomicBool,
    /// Calls the owner allowed on a High (or Critical) card while the run
    /// was capped: such a card is never answered by the capped device, so
    /// the cap needs no second card for the same call. Each is good once.
    passes: Mutex<HashSet<CallKey>>,
    /// The risk a call's own tool rated it when its approval rule asked
    /// (whether or not a grant then covered the ask): the tool's typed
    /// judgment, read by the cap for the same call.
    rated: Mutex<HashMap<CallKey, ApprovalRisk>>,
}

/// The device ceiling of a request's runs. Clones share it; the default is
/// uncapped. A runtime handle, never serialized.
#[derive(Debug, Clone, Default)]
pub struct DeviceCap(Arc<DeviceCapState>);

impl DeviceCap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap the runs at Medium from now on. Never lowered.
    pub fn raise(&self) {
        self.0.capped.store(true, Ordering::SeqCst);
    }

    /// Whether the runs are capped now.
    pub fn is_capped(&self) -> bool {
        self.0.capped.load(Ordering::SeqCst)
    }

    /// The owner allowed the call `call_id` (its key) of `run` on a card
    /// the capped device cannot answer.
    pub fn allow_call(&self, run: RunId, call_id: &str) {
        self.0
            .passes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((run, call_id.to_string()));
    }

    /// Whether the owner allowed this call on such a card; consumes it.
    pub fn take_pass(&self, run: RunId, call_id: &str) -> bool {
        self.0
            .passes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(run, call_id.to_string()))
    }

    /// The call's own tool rated it `risk` when its approval rule asked.
    /// The highest rating of a call is kept.
    pub fn rate(&self, run: RunId, call_id: &str, risk: ApprovalRisk) {
        let mut rated = self.0.rated.lock().unwrap_or_else(|e| e.into_inner());
        let entry = rated.entry((run, call_id.to_string())).or_insert(risk);
        if risk_rank(risk) > risk_rank(*entry) {
            *entry = risk;
        }
    }

    /// The risk the call's own tool gave it, if its rule asked.
    pub fn rating(&self, run: RunId, call_id: &str) -> Option<ApprovalRisk> {
        self.0
            .rated
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(run, call_id.to_string()))
            .copied()
    }
}

/// An approval risk's order: Low < Medium < High < Critical (a risk this
/// crate does not know ranks as High).
pub fn risk_rank(risk: ApprovalRisk) -> u8 {
    match risk {
        ApprovalRisk::Low => 0,
        ApprovalRisk::Medium => 1,
        ApprovalRisk::Critical => 3,
        _ => 2,
    }
}

/// Whether `risk` is above the Medium ceiling.
pub fn above_medium(risk: ApprovalRisk) -> bool {
    risk_rank(risk) > risk_rank(ApprovalRisk::Medium)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cap_only_rises_and_clones_share_it() {
        let cap = DeviceCap::new();
        let shared = cap.clone();
        assert!(!cap.is_capped());
        shared.raise();
        assert!(cap.is_capped());
    }

    #[test]
    fn a_pass_is_for_one_call_once() {
        let cap = DeviceCap::new();
        let run = RunId::new();
        cap.allow_call(run, "call-1");
        assert!(!cap.take_pass(run, "call-2"));
        assert!(!cap.take_pass(RunId::new(), "call-1"));
        assert!(cap.take_pass(run, "call-1"));
        assert!(!cap.take_pass(run, "call-1"));
    }

    #[test]
    fn a_call_keeps_its_highest_rating() {
        let cap = DeviceCap::new();
        let run = RunId::new();
        cap.rate(run, "c", ApprovalRisk::High);
        cap.rate(run, "c", ApprovalRisk::Medium);
        assert_eq!(cap.rating(run, "c"), Some(ApprovalRisk::High));
        assert!(above_medium(ApprovalRisk::High));
        assert!(!above_medium(ApprovalRisk::Medium));
        assert!(!above_medium(ApprovalRisk::Low));
    }
}
