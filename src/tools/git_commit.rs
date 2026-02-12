use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities, ToolRole};

use super::fs_utils;

pub struct GitCommitTool;

#[async_trait]
impl Tool for GitCommitTool {
    fn name(&self) -> &str {
        "git_commit"
    }

    fn description(&self) -> &str {
        "Stage and commit changes to git"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "git_commit",
            "description": "Stage files and create a git commit. Use this instead of running separate git add + git commit terminal commands. Validates changes exist before committing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Commit message"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific files to stage (default: all changed files)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the git repository (default: current directory)"
                    }
                },
                "required": ["message"],
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let message = args["message"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: message"))?;
        let files: Option<Vec<String>> = args["files"].as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        });
        let path_str = args["path"].as_str().unwrap_or(".");
        let repo_dir = fs_utils::validate_path(path_str)?;

        // Verify git repo
        let check = fs_utils::run_cmd("git rev-parse --git-dir", Some(&repo_dir), 5).await?;
        if check.exit_code != 0 {
            anyhow::bail!("Not a git repository: {}", path_str);
        }

        // Check for changes
        let status = fs_utils::run_cmd("git status --porcelain", Some(&repo_dir), 5).await?;
        if status.exit_code != 0 {
            anyhow::bail!("Failed to get git status: {}", status.stderr);
        }

        let changed_files: Vec<&str> = status.stdout.lines().collect();
        if changed_files.is_empty() {
            anyhow::bail!("No changes to commit");
        }

        // Stage files
        if let Some(ref specific_files) = files {
            if specific_files.is_empty() {
                anyhow::bail!("Empty files list provided. Omit 'files' to stage all changes.");
            }
            for file in specific_files {
                let add_cmd = format!("git add -- '{}'", file.replace('\'', "'\\''"));
                let add_result = fs_utils::run_cmd(&add_cmd, Some(&repo_dir), 5).await?;
                if add_result.exit_code != 0 {
                    anyhow::bail!("Failed to stage '{}': {}", file, add_result.stderr);
                }
            }
        } else {
            let add_result = fs_utils::run_cmd("git add -A", Some(&repo_dir), 10).await?;
            if add_result.exit_code != 0 {
                anyhow::bail!("Failed to stage changes: {}", add_result.stderr);
            }
        }

        // Commit
        let escaped_message = message.replace('\'', "'\\''");
        let commit_cmd = format!("git commit -m '{}'", escaped_message);
        let commit_result = fs_utils::run_cmd(&commit_cmd, Some(&repo_dir), 30).await?;

        if commit_result.exit_code != 0 {
            anyhow::bail!("Commit failed: {}", commit_result.stderr);
        }

        // Get commit hash
        let hash_result =
            fs_utils::run_cmd("git log -1 --format='%H %h %s'", Some(&repo_dir), 5).await?;

        let mut output = String::new();
        output.push_str("Commit successful!\n\n");

        if hash_result.exit_code == 0 {
            output.push_str(&format!("{}\n", hash_result.stdout.trim()));
        }

        // Show what was committed
        let staged_files = if let Some(ref specific) = files {
            specific.join(", ")
        } else {
            format!("{} files", changed_files.len())
        };
        output.push_str(&format!("\nStaged: {}", staged_files));

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = GitCommitTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "git_commit");
        assert!(schema["description"].as_str().unwrap().len() > 0);
        assert!(schema["parameters"]["properties"]["message"].is_object());
    }

    #[tokio::test]
    async fn test_git_commit_no_repo() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({
            "message": "test commit",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = GitCommitTool.call(&args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_git_commit_no_changes() {
        let dir = tempfile::tempdir().unwrap();
        // Init a git repo with an initial commit
        fs_utils::run_cmd("git init", Some(dir.path()), 5)
            .await
            .unwrap();
        fs_utils::run_cmd("git config user.email 'test@test.com'", Some(dir.path()), 5)
            .await
            .unwrap();
        fs_utils::run_cmd("git config user.name 'Test'", Some(dir.path()), 5)
            .await
            .unwrap();
        std::fs::write(dir.path().join("README.md"), "# Test").unwrap();
        fs_utils::run_cmd("git add -A && git commit -m 'init'", Some(dir.path()), 5)
            .await
            .unwrap();

        let args = json!({
            "message": "empty commit",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = GitCommitTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No changes"));
    }

    #[tokio::test]
    async fn test_git_commit_success() {
        let dir = tempfile::tempdir().unwrap();
        fs_utils::run_cmd("git init", Some(dir.path()), 5)
            .await
            .unwrap();
        fs_utils::run_cmd("git config user.email 'test@test.com'", Some(dir.path()), 5)
            .await
            .unwrap();
        fs_utils::run_cmd("git config user.name 'Test'", Some(dir.path()), 5)
            .await
            .unwrap();

        std::fs::write(dir.path().join("file.txt"), "hello").unwrap();

        let args = json!({
            "message": "add file",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = GitCommitTool.call(&args).await.unwrap();
        assert!(result.contains("Commit successful"));
        assert!(result.contains("add file"));
    }

    #[tokio::test]
    async fn test_git_commit_specific_files() {
        let dir = tempfile::tempdir().unwrap();
        fs_utils::run_cmd("git init", Some(dir.path()), 5)
            .await
            .unwrap();
        fs_utils::run_cmd("git config user.email 'test@test.com'", Some(dir.path()), 5)
            .await
            .unwrap();
        fs_utils::run_cmd("git config user.name 'Test'", Some(dir.path()), 5)
            .await
            .unwrap();

        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        std::fs::write(dir.path().join("b.txt"), "b").unwrap();

        let args = json!({
            "message": "add a only",
            "files": ["a.txt"],
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = GitCommitTool.call(&args).await.unwrap();
        assert!(result.contains("Commit successful"));

        // b.txt should still be untracked
        let status = fs_utils::run_cmd("git status --porcelain", Some(dir.path()), 5)
            .await
            .unwrap();
        assert!(status.stdout.contains("b.txt"));
    }
}
