//! The owner's approvals for computer-use sessions and actions.
//!
//! The tool never asks the owner itself: the agent kernel asks with the card
//! `ComputerUseTool::owner_approval_request` returns and hands the answer back
//! through `record_owner_answer`, which keeps it here. At execution the tool
//! checks for an approval and refuses a call that has none.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::settings::ComputerUseConfig;
use aidaemon_tools::owner_answers::OwnerAnsweredCalls;

use super::telemetry::MutationBudget;

/// The warnings on a per-app card.
pub const APP_WARNINGS: [&str; 2] = [
    "Captures screenshots (may expose private content) and can click/type in this app.",
    "Send/delete/purchase-style actions will still ask for confirmation each time.",
];

/// The warning that marks a consequential action's card (once-only).
pub const CONSEQUENTIAL_WARNING: &str = "This action may send, delete, purchase, or publish.";

/// Said on a consequential action's card.
pub const ONE_TIME_WARNING: &str =
    "Persistent allow is not available here — any approval applies to this action only.";

/// The one-call subject of an approval to control the running app `bundle_id`.
pub fn app_subject(bundle_id: &str) -> String {
    format!("computer_use app {bundle_id}")
}

/// The one-call subject of an approval to launch and control the app the
/// call names, asked before it runs (its bundle id is not known yet).
pub fn requested_app_subject(app: &str) -> String {
    format!("computer_use app named {}", app.trim().to_ascii_lowercase())
}

#[derive(Clone, Default)]
pub struct ApprovalState {
    /// Apps approved for inspect+control in a session, keyed by (session, bundle).
    /// A single per-app grant covers both screenshots and click/type — the user
    /// is asked once per app, not once per scope. Consequential actions still
    /// require a separate point-of-action confirmation (see `ensure_consequential`).
    app_approved: Arc<Mutex<HashSet<(String, String)>>>,
    /// Approvals good for one call: Allow Once on an app, and every
    /// consequential action.
    answers: Arc<OwnerAnsweredCalls>,
    mutating_actions: Arc<Mutex<HashMap<String, u32>>>,
    /// The chat whose grants a session's app approvals are (a sub-agent's
    /// session maps to its parent chat), from the dispatcher's typed
    /// `grant_session`.
    grant_sessions: Arc<std::sync::Mutex<HashMap<String, String>>>,
}

impl ApprovalState {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn clear_task(&self, task_id: &str) {
        self.mutating_actions.lock().await.remove(task_id);
    }

    #[cfg(test)]
    pub async fn approve_all_for_test(&self, session_id: &str, bundle_id: &str) {
        self.approve_app_for_session(session_id, bundle_id).await;
    }

    pub async fn mutations_used(&self, task_id: &str) -> u32 {
        self.mutating_actions
            .lock()
            .await
            .get(task_id)
            .copied()
            .unwrap_or(0)
    }

    pub fn mutation_budget(config: &ComputerUseConfig, used: u32) -> MutationBudget {
        MutationBudget {
            used,
            max: config.max_mutating_actions,
        }
    }

    pub async fn record_mutating_action(
        &self,
        task_id: &str,
        config: &ComputerUseConfig,
    ) -> Result<MutationBudget, String> {
        let mut counts = self.mutating_actions.lock().await;
        let count = counts.entry(task_id.to_string()).or_insert(0);
        *count = count.saturating_add(1);
        let budget = Self::mutation_budget(config, *count);
        if *count > config.max_mutating_actions {
            return Err(format!(
                "computer_use mutating action budget exceeded (max {})",
                config.max_mutating_actions
            ));
        }
        Ok(budget)
    }

    /// Whether operating the app needs the owner's approval in this session.
    /// `None`: the app is not running yet, so nothing covers it.
    pub async fn app_needs_approval(
        &self,
        config: &ComputerUseConfig,
        session_id: &str,
        bundle_id: Option<&str>,
    ) -> bool {
        let Some(bundle_id) = bundle_id else {
            return true;
        };
        if config
            .always_allowed_apps
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(bundle_id))
        {
            return false;
        }
        !self
            .app_approved
            .lock()
            .await
            .contains(&(self.grant_key(session_id), bundle_id.to_string()))
    }

    /// Record that `session_id`'s app approvals are those of `grant_session`.
    pub fn bind_grant_session(&self, session_id: &str, grant_session: &str) {
        if session_id.is_empty() || grant_session.is_empty() || session_id == grant_session {
            return;
        }
        self.grant_sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(session_id.to_string(), grant_session.to_string());
    }

    fn grant_key(&self, session_id: &str) -> String {
        self.grant_sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(session_id)
            .cloned()
            .unwrap_or_else(|| session_id.to_string())
    }

    /// The owner allowed operating `bundle_id` for the session (for a
    /// sub-agent, its parent chat, whose other agents share it).
    pub async fn approve_app_for_session(&self, session_id: &str, bundle_id: &str) {
        let key = self.grant_key(session_id);
        self.app_approved
            .lock()
            .await
            .insert((key, bundle_id.to_string()));
    }

    /// The owner approved `subject` for one call in the session.
    pub fn approve_once(&self, session_id: &str, subject: &str) {
        self.answers.approve(session_id, subject);
    }

    /// Ensure the owner approved operating `app` in this call.
    ///
    /// One combined approval per app grants both inspection (screenshots) and
    /// control (click/type): for the session after Allow Session, or for one
    /// call. A one-call approval asked before the app was running is keyed by
    /// the name the call gave (`requested_app`). This does NOT authorize
    /// consequential actions (send/delete/purchase/publish) — those are gated
    /// separately at point of action.
    pub async fn ensure_app(
        &self,
        config: &ComputerUseConfig,
        session_id: &str,
        bundle_id: &str,
        app_name: &str,
        requested_app: &str,
    ) -> Result<(), String> {
        if !self
            .app_needs_approval(config, session_id, Some(bundle_id))
            .await
        {
            return Ok(());
        }
        if self.answers.take(session_id, &app_subject(bundle_id))
            || self
                .answers
                .take(session_id, &requested_app_subject(requested_app))
        {
            return Ok(());
        }
        Err(format!(
            "computer_use control of '{app_name}' ({bundle_id}) needs the owner's approval \
             through the agent kernel, and none was given for this call"
        ))
    }

    /// Ensure the owner approved this exact consequential action (`subject`).
    /// Each approval covers one action; none persists.
    pub fn ensure_consequential(&self, session_id: &str, subject: &str) -> Result<(), String> {
        if self.answers.take(session_id, subject) {
            Ok(())
        } else {
            Err(
                "consequential computer_use action needs the owner's approval of this exact \
                 action through the agent kernel, and none was given"
                    .to_string(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sub-agent's computer_use session shares its parent chat's app
    /// approvals; an unrelated session does not.
    #[tokio::test]
    async fn a_sub_agent_inherits_its_chats_app_approvals() {
        let state = ApprovalState::new();
        let config = ComputerUseConfig::default();
        state
            .approve_app_for_session("telegram:synthetic-user-1", "com.example.App")
            .await;
        state.bind_grant_session("specialist:child", "telegram:synthetic-user-1");
        assert!(
            !state
                .app_needs_approval(&config, "specialist:child", Some("com.example.App"))
                .await
        );
        assert!(
            state
                .app_needs_approval(&config, "specialist:other", Some("com.example.App"))
                .await
        );
    }

    #[tokio::test]
    async fn app_allow_once_does_not_become_session_approval() {
        let state = ApprovalState::new();
        let config = ComputerUseConfig::default();
        state.approve_once("session", &app_subject("com.example.App"));

        for expect_ok in [true, false] {
            let result = state
                .ensure_app(&config, "session", "com.example.App", "Example", "Example")
                .await;
            assert_eq!(result.is_ok(), expect_ok, "{result:?}");
        }
        assert!(
            state
                .app_needs_approval(&config, "session", Some("com.example.App"))
                .await
        );
    }

    #[tokio::test]
    async fn app_allow_session_is_reused() {
        let state = ApprovalState::new();
        let config = ComputerUseConfig::default();
        state
            .approve_app_for_session("session", "com.example.App")
            .await;

        for _ in 0..2 {
            state
                .ensure_app(&config, "session", "com.example.App", "Example", "Example")
                .await
                .unwrap();
        }
        assert!(
            state
                .app_needs_approval(&config, "other-session", Some("com.example.App"))
                .await
        );
    }

    #[tokio::test]
    async fn approval_asked_before_launch_covers_the_named_app_once() {
        let state = ApprovalState::new();
        let config = ComputerUseConfig::default();
        state.approve_once("session", &requested_app_subject(" Example "));

        assert!(state
            .ensure_app(&config, "session", "com.example.App", "Example", "example")
            .await
            .is_ok());
        assert!(state
            .ensure_app(&config, "session", "com.example.App", "Example", "example")
            .await
            .is_err());
    }

    #[test]
    fn consequential_approval_covers_one_exact_action() {
        let state = ApprovalState::new();
        state.approve_once("session", "delete-subject");
        assert!(state
            .ensure_consequential("session", "other-subject")
            .is_err());
        assert!(state
            .ensure_consequential("other-session", "delete-subject")
            .is_err());
        assert!(state
            .ensure_consequential("session", "delete-subject")
            .is_ok());
        assert!(state
            .ensure_consequential("session", "delete-subject")
            .is_err());
    }
}
