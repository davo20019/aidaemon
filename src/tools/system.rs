use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::Tool;

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

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        let mut info = String::new();

        // Hostname
        if let Ok(output) = tokio::process::Command::new("hostname")
            .output()
            .await
        {
            let hostname = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info.push_str(&format!("Hostname: {}\n", hostname));
        }

        // OS info
        if let Ok(output) = tokio::process::Command::new("uname")
            .arg("-a")
            .output()
            .await
        {
            let uname = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info.push_str(&format!("OS: {}\n", uname));
        }

        // Uptime
        if let Ok(output) = tokio::process::Command::new("uptime")
            .output()
            .await
        {
            let uptime = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info.push_str(&format!("Uptime: {}\n", uptime));
        }

        // Memory (works on both Linux and macOS)
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
            if let Ok(output) = tokio::process::Command::new("vm_stat")
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
