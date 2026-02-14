use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities, ToolRole};

pub struct SystemInfoTool;

#[async_trait]
impl Tool for SystemInfoTool {
    fn name(&self) -> &str {
        "system_info"
    }

    fn description(&self) -> &str {
        "Get system information including CPU and memory usage"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "system_info",
            "description": "Get system information including CPU and memory usage",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
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
        let mut info = String::new();

        // Hostname
        if let Ok(output) = tokio::process::Command::new("hostname").output().await {
            let hostname = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info.push_str(&format!("Hostname: {}\n", hostname));
        }

        // OS info
        #[cfg(unix)]
        if let Ok(output) = tokio::process::Command::new("uname")
            .arg("-a")
            .output()
            .await
        {
            let uname = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info.push_str(&format!("OS: {}\n", uname));
        }

        #[cfg(windows)]
        if let Ok(output) = tokio::process::Command::new("cmd")
            .args(["/C", "ver"])
            .output()
            .await
        {
            let ver = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info.push_str(&format!("OS: {}\n", ver));
        }

        // Uptime
        #[cfg(unix)]
        if let Ok(output) = tokio::process::Command::new("uptime").output().await {
            let uptime = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info.push_str(&format!("Uptime: {}\n", uptime));
        }

        #[cfg(windows)]
        if let Ok(output) = tokio::process::Command::new("wmic")
            .args(["os", "get", "LastBootUpTime"])
            .output()
            .await
        {
            let boot = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info.push_str(&format!("Last Boot: {}\n", boot));
        }

        // Memory (works on Linux, macOS, and Windows)
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = tokio::process::Command::new("free")
                .arg("-h")
                .output()
                .await
            {
                let mem = String::from_utf8_lossy(&output.stdout).trim().to_string();
                info.push_str(&format!("Memory:\n{}\n", mem));
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = tokio::process::Command::new("vm_stat").output().await {
                let mem = String::from_utf8_lossy(&output.stdout).trim().to_string();
                info.push_str(&format!("Memory:\n{}\n", mem));
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = tokio::process::Command::new("wmic")
                .args([
                    "OS",
                    "get",
                    "FreePhysicalMemory,TotalVisibleMemorySize",
                    "/FORMAT:LIST",
                ])
                .output()
                .await
            {
                let mem = String::from_utf8_lossy(&output.stdout).trim().to_string();
                info.push_str(&format!("Memory:\n{}\n", mem));
            }
        }

        if info.is_empty() {
            info.push_str("Could not retrieve system information.");
        }

        Ok(info)
    }
}
