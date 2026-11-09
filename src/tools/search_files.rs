//! `search_files` is implemented in `aidaemon-tools`. This module holds the
//! daemon's wording and policy for it, and the tests of its behaviour on the
//! daemon's execution backend.

#[cfg(test)]
use crate::traits::Tool;
use aidaemon_tools::ToolText;
#[cfg(test)]
use serde_json::json;

use super::file_policy::{daemon_file_policy, DaemonPaths};
use super::kernel_tool::KernelTool;
use crate::execution::SharedExecutionBackend;

pub type SearchFilesTool = KernelTool<aidaemon_tools::SearchFiles>;

impl SearchFilesTool {
    /// Without the daemon's own files protected; production passes them.
    #[cfg(test)]
    pub fn new(backend: SharedExecutionBackend) -> Self {
        Self::on_backend(backend, DaemonPaths::default())
    }

    pub fn on_backend(backend: SharedExecutionBackend, paths: DaemonPaths) -> Self {
        Self::wrap(
            aidaemon_tools::SearchFiles::on_backend(backend, daemon_file_policy(paths)).with_text(ToolText {
                short: "Search for files by name pattern or content regex".to_string(),
                schema: "Search for files by name pattern (glob) and/or content (regex). Use this instead of terminal find/grep. Automatically skips .git, node_modules, target, etc.".to_string(),
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn default_path_no_match_suggests_user_directories() {
        // No 'path' arg → tool defaults to cwd. When nothing matches a
        // filename-style glob, the output must point the model at common
        // user locations instead of dead-ending.
        let args = serde_json::json!({
            "glob": "*zz-definitely-not-present-xyz (1).pdf"
        })
        .to_string();
        let result = SearchFilesTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();

        assert!(result.contains("No matches found"));
        assert!(
            result.contains("~/Downloads"),
            "no-match default-path output must suggest user dirs, got: {result}"
        );
        assert!(result.contains("'path'"));
    }

    #[tokio::test]
    async fn collaboration_search_skips_sensitive_files() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("safe.txt"), "COLLAB_NEEDLE").unwrap();
        std::fs::write(root.path().join(".env.local"), "COLLAB_NEEDLE=secret").unwrap();
        let canonical_root = std::fs::canonicalize(root.path()).unwrap();
        let args = serde_json::json!({
            "pattern": "COLLAB_NEEDLE",
            "path": canonical_root,
        })
        .to_string();
        let facts = crate::traits::ToolCallContext {
            workspace_collaboration_root: Some(canonical_root.to_string_lossy().into_owned()),
            ..Default::default()
        };

        let result = SearchFilesTool::new(crate::execution::active_execution_backend())
            .call_with_execution_context(&args, None, facts)
            .await
            .unwrap()
            .output;
        assert!(result.contains("safe.txt"));
        assert!(!result.contains(".env.local"));
        assert!(!result.contains("secret"));
    }

    #[tokio::test]
    async fn explicit_path_no_match_has_no_user_dir_hint() {
        let dir = tempfile::tempdir().unwrap();
        let args = serde_json::json!({
            "glob": "*zz-definitely-not-present-xyz.pdf",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();
        let result = SearchFilesTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();

        assert!(result.contains("No matches found"));
        assert!(!result.contains("~/Downloads"));
    }

    #[tokio::test]
    async fn scanned_dir_is_never_glued_to_closing_punctuation() {
        // Models copy-paste the scanned dir from tool output into their next
        // call. "(N files scanned in /path)" produced paths like
        // "/Users/x/projects)" — the dir must end its line cleanly.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hit.rs"), "fn main() {}").unwrap();
        let dir_str = dir.path().to_str().unwrap();

        for glob in ["*zz-no-match-xyz.pdf", "*.rs"] {
            let args = serde_json::json!({ "glob": glob, "path": dir_str }).to_string();
            let result = SearchFilesTool::new(crate::execution::active_execution_backend())
                .call(&args)
                .await
                .unwrap();
            for bad_suffix in [")", "):"] {
                assert!(
                    !result.contains(&format!("{}{}", dir_str, bad_suffix)),
                    "scanned dir must not be followed by {:?}, got: {}",
                    bad_suffix,
                    result
                );
            }
            assert!(
                result.contains("files scanned in "),
                "scanned-dir marker must survive for project_dir extraction, got: {}",
                result
            );
        }
    }

    #[tokio::test]
    async fn trailing_punctuation_on_missing_path_is_recovered() {
        // A path copy-pasted from prose/old output may carry trailing ')' or
        // ':'. When the literal path doesn't exist but the trimmed one does,
        // search there instead of failing.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hit.rs"), "fn main() {}").unwrap();

        for suffix in [")", "):", ","] {
            let args = serde_json::json!({
                "glob": "*.rs",
                "path": format!("{}{}", dir.path().to_str().unwrap(), suffix)
            })
            .to_string();
            let result = SearchFilesTool::new(crate::execution::active_execution_backend())
                .call(&args)
                .await
                .unwrap_or_else(|e| {
                    panic!(
                        "path with trailing {:?} must recover, got error: {}",
                        suffix, e
                    )
                });
            assert!(result.contains("hit.rs"), "got: {}", result);
        }
    }

    #[tokio::test]
    async fn trailing_paren_path_that_exists_is_not_trimmed() {
        // Directories with a literal trailing paren (e.g. "Folder (1)") must
        // still be searched as-is.
        let dir = tempfile::tempdir().unwrap();
        let paren_dir = dir.path().join("Copy (1)");
        std::fs::create_dir(&paren_dir).unwrap();
        std::fs::write(paren_dir.join("hit.rs"), "fn main() {}").unwrap();

        let args = serde_json::json!({
            "glob": "*.rs",
            "path": paren_dir.to_str().unwrap()
        })
        .to_string();
        let result = SearchFilesTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("hit.rs"), "got: {}", result);
    }

    #[test]
    fn test_schema_has_required_fields() {
        let tool = SearchFilesTool::new(crate::execution::active_execution_backend());
        let schema = tool.schema();
        assert_eq!(schema["name"], "search_files");
        assert!(!schema["description"].as_str().unwrap().is_empty());
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

        let result = SearchFilesTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
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

        let result = SearchFilesTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("a.txt"));
        assert!(result.contains("hello world"));
        assert!(!result.contains("b.txt"));
    }

    #[tokio::test]
    async fn test_content_search_skips_oversized_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut large = vec![b'a'; 1_048_577];
        large.extend_from_slice(b"\nneedle_in_large_file\n");
        std::fs::write(dir.path().join("large.log"), large).unwrap();

        let args = json!({
            "pattern": "needle_in_large_file",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = SearchFilesTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(!result.contains("needle_in_large_file"));
        assert!(result.contains("Skipped 1 oversized file"));
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

        let result = SearchFilesTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("No matches"));
        assert!(result.contains(dir.path().to_str().unwrap()));
    }

    #[tokio::test]
    async fn test_search_requires_pattern_or_glob() {
        let args = json!({"path": "/tmp"}).to_string();
        let result = SearchFilesTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await;
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

        let result = SearchFilesTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("visible.js"));
        assert!(!result.contains("hidden.js"));
    }

    #[tokio::test]
    async fn test_search_warns_when_path_omitted() {
        let args = json!({
            "glob": "*",
            "max_results": 1
        })
        .to_string();

        let result = SearchFilesTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("defaulted to current directory"));
    }
}
