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

/// The channel context of an internal continuation that answers a request
/// once its runs end (or after a restart): a capped request's answer runs
/// capped too, as if the capped device had sent it.
pub fn continuation_context(capped: bool) -> aidaemon_channels::context::ChannelContext {
    let mut ctx = aidaemon_channels::context::ChannelContext::internal();
    if capped {
        ctx.sender_ceiling = Some(aidaemon_channels::context::SenderCeiling::Medium);
    }
    ctx
}

/// The settings key of the chats whose latest request a device capped.
const CAPPED_SESSIONS_KEY: &str = "runtime.device_capped_sessions";
/// Chats remembered at once; the oldest are forgotten first.
const CAPPED_SESSIONS_MAX: usize = 500;
static CAPPED_SESSIONS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn read_capped(settings: &dyn aidaemon_tools::services::SettingsStore) -> Vec<String> {
    settings
        .get_setting(CAPPED_SESSIONS_KEY)
        .await
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

/// Record whether chat `session_id`'s latest request came capped from its
/// device, so a turn a restart cut off, or a job it started again, is
/// continued capped too (the cap itself lives in memory only).
pub async fn mark_session(
    settings: &dyn aidaemon_tools::services::SettingsStore,
    session_id: &str,
    capped: bool,
) {
    let _held = CAPPED_SESSIONS.lock().await;
    let mut sessions = read_capped(settings).await;
    let known = sessions.iter().any(|s| s == session_id);
    if known == capped {
        return;
    }
    sessions.retain(|s| s != session_id);
    if capped {
        sessions.push(session_id.to_string());
        let over = sessions.len().saturating_sub(CAPPED_SESSIONS_MAX);
        sessions.drain(..over);
    }
    if let Ok(raw) = serde_json::to_string(&sessions)
        && let Err(error) = settings.set_setting(CAPPED_SESSIONS_KEY, &raw).await
    {
        tracing::warn!(%error, "Could not record a chat's device ceiling");
    }
}

/// Whether chat `session_id`'s latest request came capped from its device.
pub async fn capped_session(
    settings: &dyn aidaemon_tools::services::SettingsStore,
    session_id: &str,
) -> bool {
    read_capped(settings).await.iter().any(|s| s == session_id)
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

    #[derive(Default)]
    struct Settings(Mutex<HashMap<String, String>>);

    #[async_trait::async_trait]
    impl aidaemon_tools::services::SettingsStore for Settings {
        async fn get_setting(&self, key: &str) -> anyhow::Result<Option<String>> {
            Ok(self.0.lock().unwrap().get(key).cloned())
        }
        async fn set_setting(&self, key: &str, value: &str) -> anyhow::Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert(key.to_string(), value.to_string());
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_capped_chat_is_remembered_until_an_uncapped_message() {
        let settings = Settings::default();
        assert!(!capped_session(&settings, "app:synthetic-1").await);
        mark_session(&settings, "app:synthetic-1", true).await;
        mark_session(&settings, "app:synthetic-2", true).await;
        assert!(capped_session(&settings, "app:synthetic-1").await);
        mark_session(&settings, "app:synthetic-1", false).await;
        assert!(!capped_session(&settings, "app:synthetic-1").await);
        assert!(capped_session(&settings, "app:synthetic-2").await);
    }

    #[test]
    fn a_capped_continuation_carries_the_ceiling() {
        assert!(continuation_context(false).sender_ceiling.is_none());
        assert_eq!(
            continuation_context(true).sender_ceiling,
            Some(aidaemon_channels::context::SenderCeiling::Medium)
        );
    }
}
