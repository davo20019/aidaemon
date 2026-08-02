//! Channel approval flows for computer-use sessions and actions.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::Mutex;
use tracing::warn;

use crate::config::ComputerUseConfig;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::tools::ApprovalBroker;
use crate::types::ApprovalResponse;

use super::telemetry::MutationBudget;

#[derive(Clone, Default)]
pub struct ApprovalState {
    /// Apps approved for inspect+control in a session, keyed by (session, bundle).
    /// A single per-app grant covers both screenshots and click/type — the user
    /// is asked once per app, not once per scope. Consequential actions still
    /// require a separate point-of-action confirmation (see `ensure_consequential`).
    app_approved: Arc<Mutex<HashSet<(String, String)>>>,
    mutating_actions: Arc<Mutex<HashMap<String, u32>>>,
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
        self.app_approved
            .lock()
            .await
            .insert((session_id.to_string(), bundle_id.to_string()));
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

    /// Ensure the user has approved aidaemon to operate `app` in this session.
    ///
    /// One combined prompt per app grants both inspection (screenshots) and
    /// control (click/type). Persists for the session on Allow Always. This does
    /// NOT authorize consequential actions (send/delete/purchase/publish) — those
    /// are gated separately at point of action.
    pub async fn ensure_app(
        &self,
        approval_tx: &ApprovalBroker,
        config: &ComputerUseConfig,
        session_id: &str,
        task_id: &str,
        bundle_id: &str,
        app_name: &str,
    ) -> Result<(), String> {
        if config
            .always_allowed_apps
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(bundle_id))
        {
            return Ok(());
        }
        let key = (session_id.to_string(), bundle_id.to_string());
        if self.app_approved.lock().await.contains(&key) {
            return Ok(());
        }
        let summary = format!("Allow aidaemon to inspect and control '{app_name}' ({bundle_id})?");
        let response = request_approval(
            approval_tx,
            session_id,
            &summary,
            RiskLevel::High,
            vec![
                "Captures screenshots (may expose private content) and can click/type in this app."
                    .to_string(),
                "Send/delete/purchase-style actions will still ask for confirmation each time."
                    .to_string(),
            ],
            Some(task_id),
            crate::types::ApprovalKind::Command,
        )
        .await?;
        match response {
            ApprovalResponse::AllowOnce
            | ApprovalResponse::AllowSession
            | ApprovalResponse::AllowAlways => {
                self.app_approved.lock().await.insert(key);
                Ok(())
            }
            ApprovalResponse::Deny => Err(format!("computer_use denied for {app_name}")),
        }
    }

    pub async fn ensure_consequential(
        &self,
        approval_tx: &ApprovalBroker,
        session_id: &str,
        task_id: &str,
        summary: &str,
    ) -> Result<(), String> {
        let response = request_approval(
            approval_tx,
            session_id,
            summary,
            RiskLevel::Critical,
            vec![
                "This action may send, delete, purchase, or publish.".to_string(),
                "Persistent allow is not available here — any approval applies to this action \
                 only."
                    .to_string(),
            ],
            Some(task_id),
            crate::types::ApprovalKind::CommandOnce,
        )
        .await?;
        match response {
            // The user said yes — a session/always response is still consent for
            // this action. Persistent grants are intentionally never stored for
            // consequential actions, so all allows degrade to one-time.
            ApprovalResponse::AllowOnce
            | ApprovalResponse::AllowSession
            | ApprovalResponse::AllowAlways => Ok(()),
            ApprovalResponse::Deny => {
                Err("consequential computer_use action denied by user".to_string())
            }
        }
    }
}

async fn request_approval(
    approval_tx: &ApprovalBroker,
    session_id: &str,
    command: &str,
    risk_level: RiskLevel,
    warnings: Vec<String>,
    _task_id: Option<&str>,
    kind: crate::types::ApprovalKind,
) -> Result<ApprovalResponse, String> {
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    if let Err(send_err) = approval_tx
        .send(ApprovalRequest {
            command: command.to_string(),
            session_id: session_id.to_string(),
            risk_level,
            warnings,
            permission_mode: PermissionMode::Default,
            response_tx,
            kind,
        })
        .await
    {
        return Err(format!("Approval channel closed: {send_err}"));
    }

    // Keep the per-approval wait comfortably below the agent's tool-call
    // watchdog (300s): if the approval wait reached the watchdog, the tool call
    // is killed mid-await, the turn ends abnormally, and the running-task entry
    // can leak (the queue then stalls behind a phantom task). 120s gives the
    // user time to respond while still resolving cleanly before the watchdog
    // fires. Child sessions are routed to the parent channel and get the same
    // response window.
    let timeout_secs = 120;
    match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), response_rx).await {
        Ok(Ok(response)) => Ok(response),
        Ok(Err(_)) => {
            warn!(command, "computer_use approval response channel closed");
            Err("computer_use approval response channel closed".to_string())
        }
        Err(_) => {
            warn!(command, "computer_use approval timed out");
            Err(format!(
                "computer_use approval timed out after {timeout_secs}s"
            ))
        }
    }
}
