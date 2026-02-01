use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, RwLock};
use tracing::info;

use crate::traits::Tool;

/// Response to an approval request from the user.
#[derive(Debug, Clone)]
pub enum ApprovalResponse {
    AllowOnce,
    AllowAlways,
    Deny,
}

/// A request sent to the user for command approval.
pub struct ApprovalRequest {
    pub command: String,
    pub response_tx: tokio::sync::oneshot::Sender<ApprovalResponse>,
}

pub struct TerminalTool {
    allowed_prefixes: Arc<RwLock<Vec<String>>>,
    approval_tx: mpsc::Sender<ApprovalRequest>,
}

impl TerminalTool {
    pub fn new(
        allowed_prefixes: Vec<String>,
        approval_tx: mpsc::Sender<ApprovalRequest>,
    ) -> Self {
        Self {
            allowed_prefixes: Arc::new(RwLock::new(allowed_prefixes)),
            approval_tx,
        }
    }

    async fn is_allowed(&self, command: &str) -> bool {
        let prefixes = self.allowed_prefixes.read().await;
        if prefixes.iter().any(|p| p == "*") {
            return true;
        }
        let trimmed = command.trim();
        prefixes
            .iter()
            .any(|prefix| trimmed.starts_with(prefix.as_str()))
    }

    async fn request_approval(&self, command: &str) -> anyhow::Result<ApprovalResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                command: command.to_string(),
                response_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;

        response_rx
            .await
            .map_err(|_| anyhow::anyhow!("Approval response channel closed"))
    }

    async fn add_prefix(&self, command: &str) {
        let prefix = command
            .split_whitespace()
            .next()
            .unwrap_or(command.trim());
        let mut prefixes = self.allowed_prefixes.write().await;
        if !prefixes.contains(&prefix.to_string()) {
            info!(prefix, "Adding to allowed command prefixes");
            prefixes.push(prefix.to_string());
        }
    }
}

#[derive(Deserialize)]
struct TerminalArgs {
    command: String,
}

#[async_trait]
impl Tool for TerminalTool {
    fn name(&self) -> &str {
        "terminal"
    }

    fn description(&self) -> &str {
        "Execute a shell command. If a command is not pre-approved, the user will be asked to authorize it."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "terminal",
            "description": "Execute any command available on this system — shell commands, CLI tools (python, node, claude, gemini, cargo, docker, git, etc.), scripts, and anything else installed. If the command is not pre-approved, the user will be asked to authorize it in real time via an inline button. Never assume a command is unavailable — try it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute (any valid shell command or CLI tool)"
                    }
                },
                "required": ["command"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: TerminalArgs = serde_json::from_str(arguments)?;

        if !self.is_allowed(&args.command).await {
            // Request approval from the user via Telegram
            match self.request_approval(&args.command).await {
                Ok(ApprovalResponse::AllowOnce) => {
                    // proceed below
                }
                Ok(ApprovalResponse::AllowAlways) => {
                    self.add_prefix(&args.command).await;
                    // proceed below
                }
                Ok(ApprovalResponse::Deny) => {
                    return Ok("Command denied by user.".to_string());
                }
                Err(e) => {
                    return Ok(format!("Could not get approval: {}", e));
                }
            }
        }

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&args.command)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push_str("\n--- stderr ---\n");
            }
            result.push_str(&stderr);
        }
        if result.is_empty() {
            result.push_str("(no output)");
        }

        // Truncate very long output
        if result.len() > 4000 {
            result.truncate(4000);
            result.push_str("\n... (truncated)");
        }

        Ok(result)
    }
}
