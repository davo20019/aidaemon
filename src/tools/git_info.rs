use std::path::Path;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities, ToolRole};

use super::fs_utils;

pub struct GitInfoTool;

const VALID_SECTIONS: &[&str] = &["status", "log", "branches", "remotes", "diff", "stash"];

#[async_trait]
impl Tool for GitInfoTool {
    fn name(&self) -> &str {
        "git_info"
    }

    fn description(&self) -> &str {
        "Get comprehensive git repository information in one call"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "git_info",
            "description": "Get comprehensive git repository information: status, log, branches, remotes, diff, and stash — all in one call instead of multiple terminal commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the git repository (default: current directory)"
                    },
                    "include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["status", "log", "branches", "remotes", "diff", "stash"]
                        },
                        "description": "Sections to include (default: all). Options: status, log, branches, remotes, diff, stash"
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
        let repo_dir = fs_utils::validate_path(path_str)?;

        if !repo_dir.join(".git").exists() && !repo_dir.join("../.git").exists() {
            // Check if we're in a subdirectory of a git repo
            if let Ok(out) = fs_utils::run_cmd("git rev-parse --git-dir", Some(&repo_dir), 5).await
            {
                if out.exit_code != 0 {
                    anyhow::bail!("Not a git repository: {}", path_str);
                }
            } else {
                anyhow::bail!("Not a git repository: {}", path_str);
            }
        }

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
                "status" => get_status(&repo_dir).await,
                "log" => get_log(&repo_dir).await,
                "branches" => get_branches(&repo_dir).await,
                "remotes" => get_remotes(&repo_dir).await,
                "diff" => get_diff(&repo_dir).await,
                "stash" => get_stash(&repo_dir).await,
                _ => continue,
            };

            if !section_output.is_empty() {
                output.push_str(&format!("## {}\n{}\n", capitalize(section), section_output));
            }
        }

        if output.is_empty() {
            output.push_str("No git information available.");
        }

        Ok(output)
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

async fn get_status(dir: &Path) -> String {
    let mut result = String::new();

    // Branch
    if let Ok(out) = fs_utils::run_cmd("git rev-parse --abbrev-ref HEAD", Some(dir), 5).await {
        if out.exit_code == 0 {
            result.push_str(&format!("Branch: {}\n", out.stdout.trim()));
        }
    }

    // Status
    if let Ok(out) = fs_utils::run_cmd("git status --porcelain", Some(dir), 5).await {
        if out.exit_code == 0 {
            let lines: Vec<&str> = out.stdout.lines().collect();
            if lines.is_empty() {
                result.push_str("Working tree: clean\n");
            } else {
                result.push_str(&format!("Changed files ({}):\n", lines.len()));
                for line in lines.iter().take(30) {
                    result.push_str(&format!("  {}\n", line));
                }
                if lines.len() > 30 {
                    result.push_str(&format!("  ... and {} more\n", lines.len() - 30));
                }
            }
        }
    }

    // Ahead/behind
    if let Ok(out) = fs_utils::run_cmd(
        "git rev-list --left-right --count HEAD...@{upstream}",
        Some(dir),
        5,
    )
    .await
    {
        if out.exit_code == 0 {
            let parts: Vec<&str> = out.stdout.trim().split('\t').collect();
            if parts.len() == 2 {
                let ahead = parts[0].trim();
                let behind = parts[1].trim();
                if ahead != "0" || behind != "0" {
                    result.push_str(&format!("Ahead: {}, Behind: {}\n", ahead, behind));
                }
            }
        }
    }

    result
}

async fn get_log(dir: &Path) -> String {
    if let Ok(out) = fs_utils::run_cmd(
        "git log --oneline -10 --format='%h %s (%cr, %an)'",
        Some(dir),
        5,
    )
    .await
    {
        if out.exit_code == 0 && !out.stdout.trim().is_empty() {
            return out.stdout.clone();
        }
    }
    String::new()
}

async fn get_branches(dir: &Path) -> String {
    if let Ok(out) = fs_utils::run_cmd(
        "git branch -a --format='%(refname:short) %(upstream:short) %(upstream:track)'",
        Some(dir),
        5,
    )
    .await
    {
        if out.exit_code == 0 && !out.stdout.trim().is_empty() {
            return out.stdout.clone();
        }
    }
    String::new()
}

async fn get_remotes(dir: &Path) -> String {
    if let Ok(out) = fs_utils::run_cmd("git remote -v", Some(dir), 5).await {
        if out.exit_code == 0 && !out.stdout.trim().is_empty() {
            return out.stdout.clone();
        }
    }
    String::new()
}

async fn get_diff(dir: &Path) -> String {
    if let Ok(out) = fs_utils::run_cmd("git diff --stat", Some(dir), 10).await {
        if out.exit_code == 0 && !out.stdout.trim().is_empty() {
            return out.stdout.clone();
        }
    }
    String::new()
}

async fn get_stash(dir: &Path) -> String {
    if let Ok(out) = fs_utils::run_cmd("git stash list", Some(dir), 5).await {
        if out.exit_code == 0 && !out.stdout.trim().is_empty() {
            return out.stdout.clone();
        }
    }
    String::from("No stashes\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = GitInfoTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "git_info");
        assert!(schema["description"].as_str().unwrap().len() > 0);
        assert!(schema["parameters"]["properties"]["include"].is_object());
    }

    #[tokio::test]
    async fn test_git_info_on_current_repo() {
        // This test assumes we're running from within a git repo
        let args = json!({"include": ["status"]}).to_string();
        let result = GitInfoTool.call(&args).await;
        // Should succeed if running from aidaemon project
        if let Ok(output) = result {
            assert!(output.contains("Status") || output.contains("Branch"));
        }
    }

    #[tokio::test]
    async fn test_git_info_not_a_repo() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = GitInfoTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Not a git"));
    }

    #[test]
    fn test_capitalize() {
        assert_eq!(capitalize("status"), "Status");
        assert_eq!(capitalize("log"), "Log");
        assert_eq!(capitalize(""), "");
    }
}
