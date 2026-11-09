//! `edit_file` is implemented in `aidaemon-tools`. This module holds the daemon's
//! wording and policy for it, and the tests of its behaviour on the daemon's
//! execution backend.

use aidaemon_tools::ToolText;

#[cfg(test)]
use super::fs_utils;
#[cfg(test)]
use crate::traits::Tool;
#[cfg(test)]
use serde_json::json;

use super::file_policy::daemon_file_policy;
use super::kernel_tool::KernelTool;
use crate::execution::SharedExecutionBackend;

pub type EditFileTool = KernelTool<aidaemon_tools::EditFile>;

impl EditFileTool {
    pub fn new(backend: SharedExecutionBackend) -> Self {
        Self::wrap(
            aidaemon_tools::EditFile::on_backend(backend, daemon_file_policy()).with_text(ToolText {
                short: "Find and replace text in a file".to_string(),
                schema: "Find and replace text in a file. Use this instead of terminal sed/awk. Shows context around the change. Fails safely if the text isn't found or is ambiguous. On not-found/ambiguous text, read_file the same path and retry once before asking the user.".to_string(),
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ToolSemantics;

    #[test]
    fn git_config_targets_are_configuration_egress() {
        let tool = EditFileTool::new(crate::execution::active_execution_backend());
        let ann = |path: &str| {
            tool.call_annotations_in(
                &json!({"path": path}).to_string(),
                &crate::traits::TrustRoots::new(&[]),
            )
        };
        let config = ann("/tmp/aidaemon-test/repo/.git/config");
        assert!(config.egress && !config.read_only);
        assert!(ann("/tmp/aidaemon-test/repo/.gitattributes").egress);
        let plain = ann("/tmp/aidaemon-test/repo/src/lib.rs");
        assert!(!plain.egress && !plain.read_only);
    }
    use std::io::Write;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = EditFileTool::new(crate::execution::active_execution_backend());
        let schema = tool.schema();
        assert_eq!(schema["name"], "edit_file");
        assert!(!schema["description"].as_str().unwrap().is_empty());
        assert!(schema["parameters"]["properties"]["old_text"].is_object());
    }

    #[tokio::test]
    async fn test_edit_single_replace() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "fn main() {{\n    println!(\"hello\");\n}}\n").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "old_text": "hello",
            "new_text": "world"
        })
        .to_string();

        let result = EditFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("replaced 1 occurrence"));

        let content = tokio::fs::read_to_string(f.path()).await.unwrap();
        assert!(content.contains("world"));
        assert!(!content.contains("hello"));
    }

    #[tokio::test]
    async fn test_edit_multiple_without_flag() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "foo bar foo baz foo").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "old_text": "foo",
            "new_text": "qux"
        })
        .to_string();

        let result = EditFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("3 occurrences"));
    }

    #[tokio::test]
    async fn test_edit_replace_all() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "foo bar foo baz foo").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "old_text": "foo",
            "new_text": "qux",
            "replace_all": true
        })
        .to_string();

        let result = EditFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("replaced 3 occurrences"));

        let content = tokio::fs::read_to_string(f.path()).await.unwrap();
        assert_eq!(content, "qux bar qux baz qux\n");
    }

    #[tokio::test]
    async fn test_edit_text_not_found() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "hello world").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "old_text": "nonexistent",
            "new_text": "replacement"
        })
        .to_string();

        let result = EditFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found"));
        assert!(err.contains("Self-recovery steps"));
    }

    #[tokio::test]
    async fn test_edit_newline_normalization_recovery() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "alpha\r\nbeta\r\ngamma\r\n").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "old_text": "beta\n",
            "new_text": "BETA\n"
        })
        .to_string();

        let result = EditFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("replaced 1 occurrence"));
        assert!(result.contains("newline-normalized match"));

        let content = tokio::fs::read_to_string(f.path()).await.unwrap();
        assert!(content.contains("BETA\r\n"));
        assert!(!content.contains("beta\r\n"));
    }

    #[tokio::test]
    async fn test_edit_file_not_found() {
        let args = json!({
            "path": "/tmp/nonexistent_edit_test_12345.txt",
            "old_text": "a",
            "new_text": "b"
        })
        .to_string();

        let result = EditFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await;
        assert!(result.is_err());
    }

    /// Regression: edit_file must refuse to touch sensitive paths
    /// (~/.ssh/*, *.env, ~/.gnupg/*, etc.) the same way write_file does.
    /// Without this guard, the agent could rewrite SSH keys or env files
    /// in place via find-and-replace.
    #[tokio::test]
    async fn test_edit_file_blocks_sensitive_path() {
        // Use a path under HOME/.ssh which `is_sensitive_path` flags.
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let ssh_path = format!("{}/.ssh/edit_file_sensitivity_probe", home);

        let args = json!({
            "path": ssh_path,
            "old_text": "a",
            "new_text": "b"
        })
        .to_string();

        let result = EditFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await;
        assert!(result.is_err(), "expected error for sensitive path");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("sensitive"),
            "expected 'sensitive' in error, got: {err}"
        );
    }

    #[tokio::test]
    async fn edit_file_allows_unrelated_config_toml_and_refuses_the_daemons_own() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("config.toml");
        let own = dir.path().join("daemon-config.toml");
        std::fs::write(&project, "mode = \"a\"\n").unwrap();
        std::fs::write(&own, "mode = \"a\"\n").unwrap();
        let _guard = fs_utils::DaemonPathsGuard::set(vec![own.clone()]);
        let edit = |p: &std::path::Path| {
            json!({"path": p.to_str().unwrap(), "old_text": "a", "new_text": "b"}).to_string()
        };
        EditFileTool::new(crate::execution::active_execution_backend())
            .call(&edit(&project))
            .await
            .unwrap();
        let err = EditFileTool::new(crate::execution::active_execution_backend())
            .call(&edit(&own))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("sensitive"), "got: {err}");
        assert_eq!(std::fs::read_to_string(&own).unwrap(), "mode = \"a\"\n");
    }

    #[test]
    fn test_edit_file_capabilities_match_no_approval_tool_guidance() {
        let caps = EditFileTool::new(crate::execution::active_execution_backend()).capabilities();
        assert!(
            !caps.needs_approval,
            "edit_file is documented as a dedicated file tool that does not require approval"
        );
        assert!(!caps.high_impact_write);
    }

    #[test]
    fn declares_write_of_the_absolute_path() {
        let dir = format!("/tmp/aidaemon-test/{}", uuid::Uuid::new_v4());
        let path = format!("{dir}/notes/../acme.md");
        for key in ["path", "file_path"] {
            let effects = EditFileTool::new(crate::execution::active_execution_backend())
                .call_effects(&json!({key: path}).to_string());
            assert!(effects.complete, "{key}");
            assert_eq!(
                effects.keys,
                vec![crate::tools::command_effects::path_effect(
                    aidaemon_core::EffectClass::Write,
                    &format!("{dir}/acme.md"),
                    None
                )
                .unwrap()]
            );
        }
        // A relative path has no known working directory here.
        let relative = EditFileTool::new(crate::execution::active_execution_backend())
            .call_effects(&json!({"path": "acme.md"}).to_string());
        assert_eq!(relative, aidaemon_core::CallEffects::unknown());
    }
}
