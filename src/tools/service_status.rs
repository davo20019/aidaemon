use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolRole};

use super::fs_utils;

pub struct ServiceStatusTool;

const VALID_SECTIONS: &[&str] = &["ports", "docker", "processes"];

#[async_trait]
impl Tool for ServiceStatusTool {
    fn name(&self) -> &str {
        "service_status"
    }

    fn description(&self) -> &str {
        "Check running services, listening ports, and Docker containers"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "service_status",
            "description": "Check running services: listening ports, Docker containers, and relevant processes. Use this instead of terminal lsof/ss/docker ps commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "services": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by service names or port numbers (optional)"
                    },
                    "include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["ports", "docker", "processes"]
                        },
                        "description": "Sections to include (default: all). Options: ports, docker, processes"
                    }
                },
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Universal
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let filters: Vec<String> = args["services"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let sections: Vec<String> = if let Some(arr) = args["include"].as_array() {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            VALID_SECTIONS.iter().map(|s| s.to_string()).collect()
        };

        let mut output = String::new();

        for section in &sections {
            let section_output = match section.as_str() {
                "ports" => get_listening_ports(&filters).await,
                "docker" => get_docker_status(&filters).await,
                "processes" => get_dev_processes(&filters).await,
                _ => continue,
            };

            if !section_output.is_empty() {
                output.push_str(&format!(
                    "## {}\n{}\n",
                    capitalize_first(section),
                    section_output
                ));
            }
        }

        if output.is_empty() {
            output.push_str("No services detected.");
        }

        Ok(output)
    }
}

fn capitalize_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

async fn get_listening_ports(filters: &[String]) -> String {
    let cmd = if cfg!(target_os = "macos") {
        "lsof -iTCP -sTCP:LISTEN -nP 2>/dev/null"
    } else {
        "ss -tlnp 2>/dev/null"
    };

    match fs_utils::run_cmd(cmd, None, 10).await {
        Ok(out) if out.exit_code == 0 && !out.stdout.trim().is_empty() => {
            if filters.is_empty() {
                out.stdout
            } else {
                filter_output(&out.stdout, filters)
            }
        }
        _ => String::new(),
    }
}

async fn get_docker_status(filters: &[String]) -> String {
    // Check if docker is available
    let which_result = tokio::process::Command::new("which")
        .arg("docker")
        .output()
        .await;

    if which_result.map(|r| r.status.success()).unwrap_or(false) {
        match fs_utils::run_cmd(
            "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null",
            None,
            10,
        )
        .await
        {
            Ok(out) if out.exit_code == 0 && !out.stdout.trim().is_empty() => {
                if filters.is_empty() {
                    out.stdout
                } else {
                    filter_output(&out.stdout, filters)
                }
            }
            _ => String::from("Docker not running or no containers\n"),
        }
    } else {
        String::from("Docker not installed\n")
    }
}

async fn get_dev_processes(filters: &[String]) -> String {
    // Look for common dev server processes
    let patterns = [
        "node", "python", "ruby", "java", "go ", "cargo", "npm", "webpack", "vite", "next", "nuxt",
        "rails", "flask", "django", "uvicorn", "gunicorn", "nginx", "postgres", "mysql", "redis",
        "mongo",
    ];

    let grep_pattern = if !filters.is_empty() {
        filters.join("\\|")
    } else {
        patterns.join("\\|")
    };

    let cmd = format!(
        "ps aux | grep -i '{}' | grep -v grep | head -20",
        grep_pattern
    );

    match fs_utils::run_cmd(&cmd, None, 10).await {
        Ok(out) if out.exit_code == 0 && !out.stdout.trim().is_empty() => {
            // Format: simplify ps output
            let mut result = String::new();
            for line in out.stdout.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 11 {
                    let user = parts[0];
                    let pid = parts[1];
                    let cpu = parts[2];
                    let mem = parts[3];
                    let cmd_parts = &parts[10..];
                    let cmd_str: String = cmd_parts.join(" ");
                    // Truncate long commands
                    let cmd_display = if cmd_str.len() > 80 {
                        format!("{}...", &cmd_str[..80])
                    } else {
                        cmd_str
                    };
                    result.push_str(&format!(
                        "  PID {} ({}): CPU {}% MEM {}% {}\n",
                        pid, user, cpu, mem, cmd_display
                    ));
                }
            }
            result
        }
        _ => String::new(),
    }
}

fn filter_output(output: &str, filters: &[String]) -> String {
    let mut result = String::new();
    // Always include header line
    if let Some(header) = output.lines().next() {
        result.push_str(header);
        result.push('\n');
    }
    for line in output.lines().skip(1) {
        let lower = line.to_lowercase();
        if filters.iter().any(|f| lower.contains(&f.to_lowercase())) {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = ServiceStatusTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "service_status");
        assert!(schema["description"].as_str().unwrap().len() > 0);
        assert!(schema["parameters"]["properties"]["include"].is_object());
    }

    #[tokio::test]
    async fn test_service_status_runs() {
        let args = json!({"include": ["ports"]}).to_string();
        let result = ServiceStatusTool.call(&args).await.unwrap();
        // Should return something (ports section or "No services")
        assert!(!result.is_empty());
    }

    #[test]
    fn test_filter_output() {
        let output = "HEADER\nline with node\nline with python\nline with rust\n";
        let filtered = filter_output(output, &["node".to_string()]);
        assert!(filtered.contains("HEADER"));
        assert!(filtered.contains("node"));
        assert!(!filtered.contains("python"));
    }

    #[test]
    fn test_capitalize_first() {
        assert_eq!(capitalize_first("ports"), "Ports");
        assert_eq!(capitalize_first("docker"), "Docker");
    }
}
