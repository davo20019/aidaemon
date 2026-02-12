use std::path::{Path, PathBuf};

use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities, ToolRole};

use super::fs_utils;

pub struct SearchFilesTool;

const MAX_RESULTS: usize = 200;
const DEFAULT_MAX_RESULTS: usize = 50;
const MAX_FILES_SCANNED: usize = 10_000;
const MAX_DEPTH: usize = 20;

#[async_trait]
impl Tool for SearchFilesTool {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        "Search for files by name pattern or content regex"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "search_files",
            "description": "Search for files by name pattern (glob) and/or content (regex). Use this instead of terminal find/grep. Automatically skips .git, node_modules, target, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search file contents for"
                    },
                    "glob": {
                        "type": "string",
                        "description": "Filename glob pattern (e.g., '*.rs', '*.ts', 'Cargo.*')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 50, max: 200)"
                    }
                },
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
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
        let content_pattern = args["pattern"].as_str();
        let glob_pattern = args["glob"].as_str();
        let path_str = args["path"].as_str().unwrap_or(".");
        let max_results = args["max_results"]
            .as_u64()
            .map(|n| (n as usize).min(MAX_RESULTS))
            .unwrap_or(DEFAULT_MAX_RESULTS);

        if content_pattern.is_none() && glob_pattern.is_none() {
            anyhow::bail!("At least one of 'pattern' (content regex) or 'glob' (filename pattern) is required");
        }

        let search_dir = fs_utils::validate_path(path_str)?;
        if !search_dir.exists() {
            anyhow::bail!("Directory not found: {}", path_str);
        }

        let content_regex = if let Some(pat) = content_pattern {
            Some(Regex::new(pat).map_err(|e| anyhow::anyhow!("Invalid regex '{}': {}", pat, e))?)
        } else {
            None
        };

        let glob_regex = if let Some(g) = glob_pattern {
            Some(glob_to_regex(g)?)
        } else {
            None
        };

        let mut results = Vec::new();
        let mut files_scanned = 0;

        walk_dir(
            &search_dir,
            &content_regex,
            &glob_regex,
            max_results,
            0,
            &mut results,
            &mut files_scanned,
        )
        .await;

        if results.is_empty() {
            return Ok(format!(
                "No matches found ({} files scanned in {})",
                files_scanned, path_str
            ));
        }

        let mut output = format!(
            "Found {} match{} ({} files scanned):\n\n",
            results.len(),
            if results.len() == 1 { "" } else { "es" },
            files_scanned
        );

        for result in &results {
            output.push_str(&result.format());
            output.push('\n');
        }

        if results.len() >= max_results {
            output.push_str(&format!(
                "\n(Results capped at {}. Use a more specific pattern or glob to narrow results.)",
                max_results
            ));
        }

        Ok(output)
    }
}

struct SearchResult {
    path: PathBuf,
    matches: Vec<(usize, String)>, // (line_number, line_content)
}

impl SearchResult {
    fn format(&self) -> String {
        let path_str = self.path.display();
        if self.matches.is_empty() {
            format!("{}", path_str)
        } else {
            let mut s = format!("{}:", path_str);
            for (line_num, line) in &self.matches {
                let truncated = if line.len() > 200 {
                    format!("{}...", &line[..200])
                } else {
                    line.clone()
                };
                s.push_str(&format!("\n  {:>4}: {}", line_num, truncated));
            }
            s
        }
    }
}

fn glob_to_regex(glob: &str) -> anyhow::Result<Regex> {
    let mut regex = String::from("^");
    for c in glob.chars() {
        match c {
            '*' => regex.push_str(".*"),
            '?' => regex.push('.'),
            '.' => regex.push_str("\\."),
            '[' => regex.push('['),
            ']' => regex.push(']'),
            '{' => regex.push('('),
            '}' => regex.push(')'),
            ',' => regex.push('|'),
            c => regex.push(c),
        }
    }
    regex.push('$');
    Regex::new(&regex).map_err(|e| anyhow::anyhow!("Invalid glob pattern '{}': {}", glob, e))
}

fn walk_dir<'a>(
    dir: &'a Path,
    content_regex: &'a Option<Regex>,
    glob_regex: &'a Option<Regex>,
    max_results: usize,
    depth: usize,
    results: &'a mut Vec<SearchResult>,
    files_scanned: &'a mut usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
    Box::pin(async move {
        if depth > MAX_DEPTH || results.len() >= max_results || *files_scanned >= MAX_FILES_SCANNED
        {
            return;
        }

        let mut entries = match tokio::fs::read_dir(dir).await {
            Ok(e) => e,
            Err(_) => return,
        };

        let mut subdirs = Vec::new();

        while let Ok(Some(entry)) = entries.next_entry().await {
            if results.len() >= max_results || *files_scanned >= MAX_FILES_SCANNED {
                break;
            }

            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();

            if let Ok(file_type) = entry.file_type().await {
                if file_type.is_dir() {
                    if !fs_utils::should_skip_dir(&file_name) && !file_name.starts_with('.') {
                        subdirs.push(path);
                    }
                    continue;
                }

                if !file_type.is_file() {
                    continue;
                }
            } else {
                continue;
            }

            // Check glob match on filename
            if let Some(ref glob_re) = glob_regex {
                if !glob_re.is_match(&file_name) {
                    continue;
                }
            }

            *files_scanned += 1;

            // If content pattern specified, search contents
            if let Some(ref content_re) = content_regex {
                if let Ok(content) = tokio::fs::read_to_string(&path).await {
                    let mut matches = Vec::new();
                    for (i, line) in content.lines().enumerate() {
                        if content_re.is_match(line) {
                            matches.push((i + 1, line.to_string()));
                            if matches.len() >= 5 {
                                break; // Cap matches per file
                            }
                        }
                    }
                    if !matches.is_empty() {
                        results.push(SearchResult {
                            path: path.clone(),
                            matches,
                        });
                    }
                }
            } else {
                // Glob-only: just list the file
                results.push(SearchResult {
                    path: path.clone(),
                    matches: vec![],
                });
            }
        }

        // Recurse into subdirectories
        for subdir in subdirs {
            walk_dir(
                &subdir,
                content_regex,
                glob_regex,
                max_results,
                depth + 1,
                results,
                files_scanned,
            )
            .await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = SearchFilesTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "search_files");
        assert!(schema["description"].as_str().unwrap().len() > 0);
    }

    #[test]
    fn test_glob_to_regex() {
        let re = glob_to_regex("*.rs").unwrap();
        assert!(re.is_match("main.rs"));
        assert!(!re.is_match("main.py"));

        let re = glob_to_regex("Cargo.*").unwrap();
        assert!(re.is_match("Cargo.toml"));
        assert!(re.is_match("Cargo.lock"));
    }

    #[tokio::test]
    async fn test_search_by_glob() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("test.py"), "def main(): pass").unwrap();

        let args = json!({
            "glob": "*.rs",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = SearchFilesTool.call(&args).await.unwrap();
        assert!(result.contains("test.rs"));
        assert!(!result.contains("test.py"));
    }

    #[tokio::test]
    async fn test_search_by_content() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello world\nfoo bar\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "goodbye world\n").unwrap();

        let args = json!({
            "pattern": "hello",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = SearchFilesTool.call(&args).await.unwrap();
        assert!(result.contains("a.txt"));
        assert!(result.contains("hello world"));
        assert!(!result.contains("b.txt"));
    }

    #[tokio::test]
    async fn test_search_no_results() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\n").unwrap();

        let args = json!({
            "pattern": "nonexistent_pattern_xyz",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = SearchFilesTool.call(&args).await.unwrap();
        assert!(result.contains("No matches"));
    }

    #[tokio::test]
    async fn test_search_requires_pattern_or_glob() {
        let args = json!({"path": "/tmp"}).to_string();
        let result = SearchFilesTool.call(&args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_skips_ignored_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let node_modules = dir.path().join("node_modules");
        std::fs::create_dir(&node_modules).unwrap();
        std::fs::write(node_modules.join("hidden.js"), "should not find").unwrap();
        std::fs::write(dir.path().join("visible.js"), "should find").unwrap();

        let args = json!({
            "glob": "*.js",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = SearchFilesTool.call(&args).await.unwrap();
        assert!(result.contains("visible.js"));
        assert!(!result.contains("hidden.js"));
    }
}
