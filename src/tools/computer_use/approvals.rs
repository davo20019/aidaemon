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

#[derive(Clone, Default)]
pub struct ApprovalState {
    session_approved: Arc<Mutex<HashSet<String>>>,
    inspect_approved: Arc<Mutex<HashSet<(String, String)>>>,
    mutate_approved: Arc<Mutex<HashSet<(String, String)>>>,
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
        self.session_approved
            .lock()
            .await
            .insert(session_id.to_string());
        let key = (session_id.to_string(), bundle_id.to_string());
        self.inspect_approved.lock().await.insert(key.clone());
        self.mutate_approved.lock().await.insert(key);
    }

    pub async fn record_mutating_action(
        &self,
        task_id: &str,
        config: &ComputerUseConfig,
    ) -> Result<(), String> {
        let mut counts = self.mutating_actions.lock().await;
        let count = counts.entry(task_id.to_string()).or_insert(0);
        *count = count.saturating_add(1);
        if *count > config.max_mutating_actions {
            return Err(format!(
                "computer_use mutating action budget exceeded (max {})",
                config.max_mutating_actions
            ));
        }
        Ok(())
    }

    pub async fn ensure_session(
        &self,
        approval_tx: &ApprovalBroker,
        session_id: &str,
        task_id: &str,
    ) -> Result<(), String> {
        if self.session_approved.lock().await.contains(session_id) {
            return Ok(());
        }
        let summary = "Enable native desktop computer use for this chat session? \
                       This allows inspecting app windows and screenshots.";
        let response = request_approval(
            approval_tx,
            session_id,
            summary,
            RiskLevel::Medium,
            vec!["Screenshots may expose private on-screen content.".to_string()],
            Some(task_id),
        )
        .await?;
        match response {
            ApprovalResponse::AllowOnce | ApprovalResponse::AllowSession => {
                self.session_approved
                    .lock()
                    .await
                    .insert(session_id.to_string());
                Ok(())
            }
            ApprovalResponse::AllowAlways => {
                self.session_approved
                    .lock()
                    .await
                    .insert(session_id.to_string());
                Ok(())
            }
            ApprovalResponse::Deny => Err("computer_use session denied by user".to_string()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn ensure_app_scope(
        &self,
        approval_tx: &ApprovalBroker,
        config: &ComputerUseConfig,
        session_id: &str,
        task_id: &str,
        bundle_id: &str,
        app_name: &str,
        mutate: bool,
    ) -> Result<(), String> {
        if config
            .always_allowed_apps
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(bundle_id))
        {
            return Ok(());
        }
        let key = (session_id.to_string(), bundle_id.to_string());
        let store = if mutate {
            &self.mutate_approved
        } else {
            &self.inspect_approved
        };
        if store.lock().await.contains(&key) {
            return Ok(());
        }
        let (summary, risk, warnings) = if mutate {
            (
                format!("Allow computer_use to control '{app_name}' ({bundle_id})?"),
                RiskLevel::High,
                vec![
                    "Mutating actions move focus and may click/type in the target app.".to_string(),
                ],
            )
        } else {
            (
                format!("Allow computer_use to inspect '{app_name}' ({bundle_id})?"),
                RiskLevel::Medium,
                vec!["Inspection captures an accessibility tree and screenshot.".to_string()],
            )
        };
        let response = request_approval(
            approval_tx,
            session_id,
            &summary,
            risk,
            warnings,
            Some(task_id),
        )
        .await?;
        match response {
            ApprovalResponse::AllowOnce | ApprovalResponse::AllowSession => {
                store.lock().await.insert(key);
                Ok(())
            }
            ApprovalResponse::AllowAlways => {
                store.lock().await.insert(key);
                Ok(())
            }
            ApprovalResponse::Deny => Err(format!(
                "computer_use {} denied for {app_name}",
                if mutate { "control" } else { "inspection" }
            )),
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
            vec!["This action may send, delete, purchase, or publish.".to_string()],
            Some(task_id),
        )
        .await?;
        match response {
            ApprovalResponse::AllowOnce => Ok(()),
            ApprovalResponse::AllowSession | ApprovalResponse::AllowAlways => Err(
                "Persistent allow is not supported for consequential computer_use actions"
                    .to_string(),
            ),
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
            kind: Default::default(),
        })
        .await
    {
        return Err(format!("Approval channel closed: {send_err}"));
    }

    let timeout_secs = if session_id.starts_with("sub-") || session_id.starts_with("specialist:") {
        10
    } else {
        300
    };
    match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), response_rx).await {
        Ok(Ok(response)) => Ok(response),
        Ok(Err(_)) => {
            warn!(command, "computer_use approval response channel closed");
            Ok(ApprovalResponse::Deny)
        }
        Err(_) => {
            warn!(command, "computer_use approval timed out");
            Ok(ApprovalResponse::Deny)
        }
    }
}
