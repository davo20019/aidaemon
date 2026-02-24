use std::path::Path;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities, ToolRole};

use super::fs_utils;

pub struct CheckEnvironmentTool;

/// Runtimes/tools to check with their version flags.
const TOOLS_TO_CHECK: &[(&str, &str)] = &[
    ("rustc", "--version"),
    ("cargo", "--version"),
    ("node", "--version"),
    ("npm", "--version"),
    ("npx", "--version"),
    ("bun", "--version"),
    ("deno", "--version"),
    ("python3", "--version"),
    ("python", "--version"),
    ("pip3", "--version"),
    ("pip", "--version"),
    ("go", "version"),
    ("java", "-version"),
    ("javac", "-version"),
    ("ruby", "--version"),
    ("php", "--version"),
    ("docker", "--version"),
    ("docker-compose", "--version"),
    ("git", "--version"),
    ("make", "--version"),
    ("cmake", "--version"),
    ("gcc", "--version"),
    ("g++", "--version"),
    ("clang", "--version"),
];

/// Config files to look for.
const CONFIG_FILES: &[(&str, &str)] = &[
    (".nvmrc", "Node version"),
    (".node-version", "Node version"),
    (".python-version", "Python version"),
    (".ruby-version", "Ruby version"),
    (".tool-versions", "asdf versions"),
    ("Dockerfile", "Docker"),
    ("docker-compose.yml", "Docker Compose"),
    ("docker-compose.yaml", "Docker Compose"),
    (".env", "Environment variables"),
    (".env.local", "Local environment"),
    ("rust-toolchain.toml", "Rust toolchain"),
    ("rust-toolchain", "Rust toolchain"),
    (".editorconfig", "Editor config"),
    (".prettierrc", "Prettier"),
    (".eslintrc.json", "ESLint"),
    ("tsconfig.json", "TypeScript"),
];

#[async_trait]
impl Tool for CheckEnvironmentTool {
    fn name(&self) -> &str {
        "check_environment"
    }

    fn description(&self) -> &str {
        "Check available development tools, runtimes, and config files"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "check_environment",
            "description": "Check which development tools and runtimes are available, their versions, and what config files exist. Use this instead of running multiple 'which' and '--version' commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to check for config files (default: current directory)"
                    }
                },
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let path_str = args["path"].as_str().unwrap_or(".");
        let check_dir = fs_utils::validate_path(path_str)?;

        let mut output = String::new();

        // Check tools in parallel
        output.push_str("## Available Tools\n\n");

        let mut handles = Vec::new();
        for (tool, flag) in TOOLS_TO_CHECK {
            let tool = tool.to_string();
            let flag = flag.to_string();
            handles.push(tokio::spawn(async move { check_tool(&tool, &flag).await }));
        }

        let results = futures::future::join_all(handles).await;
        let mut found = Vec::new();
        let mut not_found = Vec::new();

        for (i, result) in results.into_iter().enumerate() {
            let (tool_name, _) = TOOLS_TO_CHECK[i];
            match result {
                Ok(Some(version)) => found.push((tool_name, version)),
                Ok(None) => not_found.push(tool_name),
                Err(_) => not_found.push(tool_name),
            }
        }

        for (name, version) in &found {
            output.push_str(&format!("  {} {}\n", pad_right(name, 18), version));
        }

        if !not_found.is_empty() {
            output.push_str(&format!("\nNot found: {}\n", not_found.join(", ")));
        }

        // Check config files
        let configs = check_config_files(&check_dir).await;
        if !configs.is_empty() {
            output.push_str("\n## Config Files\n\n");
            for (file, desc, content) in &configs {
                output.push_str(&format!("  {} ({})", file, desc));
                if let Some(c) = content {
                    output.push_str(&format!(" → {}", c));
                }
                output.push('\n');
            }
        }

        Ok(output)
    }
}

fn pad_right(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - s.len()))
    }
}

async fn check_tool(name: &str, flag: &str) -> Option<String> {
    // First check if tool exists
    let which_result = tokio::process::Command::new("which")
        .arg(name)
        .output()
        .await
        .ok()?;

    if !which_result.status.success() {
        return None;
    }

    // Get version
    let cmd = format!("{} {}", name, flag);
    match fs_utils::run_cmd(&cmd, None, 5).await {
        Ok(out) => {
            if out.exit_code == 0 {
                // Some tools output to stderr (java -version)
                let version_str = if out.stdout.trim().is_empty() {
                    out.stderr.trim().to_string()
                } else {
                    out.stdout.trim().to_string()
                };
                // Take first line only
                Some(
                    version_str
                        .lines()
                        .next()
                        .unwrap_or(&version_str)
                        .to_string(),
                )
            } else {
                Some("(installed, version unknown)".to_string())
            }
        }
        Err(_) => Some("(installed, version check timed out)".to_string()),
    }
}

async fn check_config_files(dir: &Path) -> Vec<(String, String, Option<String>)> {
    let mut configs = Vec::new();

    for (file, desc) in CONFIG_FILES {
        let path = dir.join(file);
        if path.exists() {
            // Read small config files for their content
            let content = if *file == ".nvmrc"
                || *file == ".node-version"
                || *file == ".python-version"
                || *file == ".ruby-version"
                || *file == "rust-toolchain"
            {
                tokio::fs::read_to_string(&path)
                    .await
                    .ok()
                    .map(|c| c.trim().to_string())
            } else {
                None
            };
            configs.push((file.to_string(), desc.to_string(), content));
        }
    }

    configs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = CheckEnvironmentTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "check_environment");
        assert!(!schema["description"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_check_environment_runs() {
        let args = json!({}).to_string();
        let result = CheckEnvironmentTool.call(&args).await.unwrap();
        // Should at least find git (we're in a git repo)
        assert!(result.contains("Available Tools"));
    }

    #[tokio::test]
    async fn test_check_environment_with_config_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".nvmrc"), "18.17.0").unwrap();
        std::fs::write(dir.path().join("Dockerfile"), "FROM node:18").unwrap();

        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = CheckEnvironmentTool.call(&args).await.unwrap();
        assert!(result.contains(".nvmrc"));
        assert!(result.contains("18.17.0"));
        assert!(result.contains("Dockerfile"));
    }

    #[tokio::test]
    async fn test_check_tool_git() {
        let result = check_tool("git", "--version").await;
        assert!(result.is_some());
        assert!(result.unwrap().contains("git"));
    }

    #[tokio::test]
    async fn test_check_tool_nonexistent() {
        let result = check_tool("nonexistent_tool_xyz_12345", "--version").await;
        assert!(result.is_none());
    }

    #[test]
    fn test_pad_right() {
        assert_eq!(pad_right("git", 10), "git       ");
        assert_eq!(pad_right("docker-compose", 10), "docker-compose");
    }
}
