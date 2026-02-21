use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::warn;

use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::traits::{StateStore, Tool};
use crate::types::{ApprovalResponse, FactPrivacy};

/// Timeout for approval requests (5 minutes).
const APPROVAL_TIMEOUT_SECS: u64 = 300;

pub struct ShareMemoryTool {
    state: Arc<dyn StateStore>,
    approval_tx: mpsc::Sender<ApprovalRequest>,
}

impl ShareMemoryTool {
    pub fn new(state: Arc<dyn StateStore>, approval_tx: mpsc::Sender<ApprovalRequest>) -> Self {
        Self { state, approval_tx }
    }

    async fn request_approval(
        &self,
        session_id: &str,
        description: &str,
    ) -> anyhow::Result<ApprovalResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                command: description.to_string(),
                session_id: session_id.to_string(),
                risk_level: RiskLevel::Medium,
                warnings: vec![
                    "This will make this memory visible across all channels".to_string(),
                ],
                permission_mode: PermissionMode::Default,
                response_tx,
                kind: Default::default(),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;
        match tokio::time::timeout(Duration::from_secs(APPROVAL_TIMEOUT_SECS), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                warn!("Approval response channel closed for share_memory");
                Ok(ApprovalResponse::Deny)
            }
            Err(_) => {
                warn!(
                    "Approval request timed out for share_memory ({}s)",
                    APPROVAL_TIMEOUT_SECS
                );
                Ok(ApprovalResponse::Deny)
            }
        }
    }
}

#[derive(Deserialize)]
struct ShareArgs {
    category: String,
    key: String,
    #[serde(default)]
    _session_id: String,
}

#[async_trait]
impl Tool for ShareMemoryTool {
    fn name(&self) -> &str {
        "share_memory"
    }

    fn description(&self) -> &str {
        "Make a memory permanently shareable in the current channel"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "share_memory",
            "description": "Make a memory permanently shareable in the current channel. Use when the owner approves sharing cross-channel information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of the fact to share"
                    },
                    "key": {
                        "type": "string",
                        "description": "The key of the fact to share"
                    }
                },
                "required": ["category", "key"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ShareArgs = serde_json::from_str(arguments)?;

        // Find the fact by category + key
        let facts = self.state.get_facts(Some(&args.category)).await?;
        let fact = facts
            .iter()
            .find(|f| f.key == args.key && f.superseded_at.is_none());

        match fact {
            Some(f) => {
                // Request user approval before sharing
                let description = format!(
                    "Share memory globally: [{}] {} = \"{}\"",
                    args.category,
                    args.key,
                    if f.value.len() > 80 {
                        format!("{}...", &f.value[..80])
                    } else {
                        f.value.clone()
                    }
                );
                match self
                    .request_approval(&args._session_id, &description)
                    .await?
                {
                    ApprovalResponse::Deny => Ok("Memory sharing denied by user.".to_string()),
                    ApprovalResponse::AllowOnce
                    | ApprovalResponse::AllowSession
                    | ApprovalResponse::AllowAlways => {
                        self.state
                            .update_fact_privacy(f.id, FactPrivacy::Global)
                            .await?;
                        Ok(format!(
                            "Memory shared: [{}] {} is now globally accessible.",
                            args.category, args.key
                        ))
                    }
                }
            }
            None => Ok(format!(
                "No active fact found with category '{}' and key '{}'.",
                args.category, args.key
            )),
        }
    }
}
