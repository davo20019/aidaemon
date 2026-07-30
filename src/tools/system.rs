use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::execution::{active_execution_backend, ExecutionRequest, SharedExecutionBackend};
use crate::traits::{Tool, ToolCapabilities, ToolRole};

pub struct SystemInfoTool;

#[async_trait]
impl Tool for SystemInfoTool {
    fn name(&self) -> &str {
        "system_info"
    }

    fn description(&self) -> &str {
        "Get system information for the configured execution environment"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "system_info",
            "description": "Get system information for the configured execution environment, including hostname, OS, uptime, and memory",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Universal
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        let backend = active_execution_backend();
        let mut info = format!(
            "Execution target: {} ({})\nWorkspace: {}\n",
            backend.kind().as_str(),
            backend.id(),
            backend.workspace_root()
        );

        // Current date and time
        if let Some(date) = command_output(backend.clone(), "date", &[]).await {
            info.push_str(&format!("Date: {}\n", date));
        }

        // Hostname
        if let Some(hostname) = command_output(backend.clone(), "hostname", &[]).await {
            info.push_str(&format!("Hostname: {}\n", hostname));
        }

        // OS info
        if let Some(uname) = command_output(backend.clone(), "uname", &["-a"]).await {
            info.push_str(&format!("OS: {}\n", uname));
        }

        // Uptime
        if let Some(uptime) = command_output(backend.clone(), "uptime", &[]).await {
            info.push_str(&format!("Uptime: {}\n", uptime));
        }

        // Probe the execution target rather than compiling this decision for
        // the daemon host's operating system.
        let memory = if backend.executable_exists("free").await.unwrap_or(false) {
            command_output(backend.clone(), "free", &["-h"]).await
        } else if backend.executable_exists("vm_stat").await.unwrap_or(false) {
            command_output(backend, "vm_stat", &[]).await
        } else {
            None
        };
        if let Some(memory) = memory {
            info.push_str(&format!("Memory:\n{}\n", memory));
        }

        Ok(info)
    }
}

async fn command_output(
    backend: SharedExecutionBackend,
    program: &str,
    args: &[&str],
) -> Option<String> {
    let output = backend
        .execute(
            ExecutionRequest::argv(
                program,
                args.iter()
                    .map(|argument| (*argument).to_string())
                    .collect(),
            ),
            Duration::from_secs(5),
        )
        .await
        .ok()?;
    (output.exit_code == 0)
        .then(|| output.stdout_lossy().trim().to_string())
        .filter(|text| !text.is_empty())
}
