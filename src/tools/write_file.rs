//! `write_file` is implemented in `aidaemon-tools`. This module holds the daemon's
//! wording and policy for it, and the tests of its behaviour on the daemon's
//! execution backend.

use aidaemon_tools::ToolText;

#[cfg(test)]
use crate::traits::{DurableReplayDecision, Tool};
#[cfg(test)]
use serde_json::json;

use super::file_policy::{daemon_file_policy, DaemonPaths};
use crate::execution::SharedExecutionBackend;

/// The daemon's constructors for the SDK's [`aidaemon_tools::WriteFile`].
pub struct WriteFileTool;

#[allow(clippy::new_ret_no_self)]
impl WriteFileTool {
    /// Without the daemon's own files protected; production passes them.
    #[cfg(test)]
    pub fn new(backend: SharedExecutionBackend) -> aidaemon_tools::WriteFile {
        Self::on_backend(backend, DaemonPaths::default())
    }

    pub fn on_backend(
        backend: SharedExecutionBackend,
        paths: DaemonPaths,
    ) -> aidaemon_tools::WriteFile {
        aidaemon_tools::WriteFile::on_backend(backend, daemon_file_policy(paths)).with_text(ToolText {
                short: "Write or create a file with given content".to_string(),
                schema: "Write content to a file, creating it if it doesn't exist. Use this instead of terminal echo/cat for file creation. Creates a backup of existing files before overwriting. ALWAYS prefer write_file over terminal heredocs (cat > file << EOF) — heredocs trigger the approval flow.".to_string(),
            })
    }
}

#[cfg(test)]
mod durable_replay_tests {
    use super::*;
    use crate::traits::ToolSemantics;

    #[tokio::test]
    async fn overwrite_receipt_reconciles_exact_current_file_state() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("result.md");
        let arguments = json!({
            "path": path.to_string_lossy(),
            "content": "expected content\n"
        })
        .to_string();
        let tool = WriteFileTool::new(crate::execution::active_execution_backend());

        assert!(matches!(
            tool.durable_replay_decision(&arguments).await,
            DurableReplayDecision::Reexecute { .. }
        ));

        std::fs::write(&path, "expected content\n").unwrap();
        assert_eq!(
            tool.durable_replay_decision(&arguments).await,
            DurableReplayDecision::Replay
        );

        std::fs::write(&path, "concurrent content\n").unwrap();
        assert!(matches!(
            tool.durable_replay_decision(&arguments).await,
            DurableReplayDecision::Block { .. }
        ));
    }

    #[tokio::test]
    async fn append_receipt_is_never_automatically_reexecuted() {
        let directory = tempfile::tempdir().unwrap();
        let arguments = json!({
            "path": directory.path().join("result.md").to_string_lossy(),
            "content": "one chunk\n",
            "mode": "append"
        })
        .to_string();

        assert_eq!(
            WriteFileTool::new(crate::execution::active_execution_backend())
                .durable_replay_decision(&arguments)
                .await,
            DurableReplayDecision::Replay
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ToolCalls as _;
    use crate::traits::ToolSemantics;

    #[test]
    fn git_config_targets_are_configuration_egress() {
        let tool = WriteFileTool::new(crate::execution::active_execution_backend());
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

    #[test]
    fn test_schema_has_required_fields() {
        let tool = WriteFileTool::new(crate::execution::active_execution_backend());
        let schema = tool.schema();
        assert_eq!(schema["name"], "write_file");
        assert!(!schema["description"].as_str().unwrap().is_empty());
        assert!(schema["parameters"]["properties"]["path"].is_object());
        assert!(schema["parameters"]["properties"]["content"].is_object());
    }

    #[tokio::test]
    async fn write_file_allows_unrelated_project_config_toml() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(&config, "title = \"a\"\n").unwrap();
        let args =
            json!({"path": config.to_str().unwrap(), "content": "title = \"b\"\n"}).to_string();
        WriteFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        let args = json!({"path": config.to_str().unwrap()}).to_string();
        let read = crate::tools::read_file::ReadFileTool::new(
            crate::execution::active_execution_backend(),
        )
        .call(&args)
        .await
        .unwrap();
        assert!(read.contains("title = \"b\""), "{read}");
    }

    #[tokio::test]
    async fn write_file_refuses_the_daemons_own_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("daemon-config.toml");
        std::fs::write(&config, "[provider]\nname = \"synthetic\"\n").unwrap();
        let paths = crate::tools::DaemonPaths::new(vec![config.clone()]);
        let args = json!({"path": config.to_str().unwrap(), "content": "x = 1"}).to_string();
        let err = crate::tools::guarded_call_protecting(
            paths,
            &WriteFileTool::new(crate::execution::active_execution_backend()),
            &args,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("sensitive"), "got: {err}");
        assert!(std::fs::read_to_string(&config)
            .unwrap()
            .contains("synthetic"));
    }

    #[tokio::test]
    async fn test_write_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("new_file.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "Hello, world!"
        })
        .to_string();

        let result = WriteFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("Created"));
        assert!(result.contains("13 bytes"));

        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_write_overwrite_with_backup() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("existing.txt");
        tokio::fs::write(&file_path, "old content").await.unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "new content"
        })
        .to_string();

        let result = WriteFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("Updated"));
        assert!(result.contains("backup saved"));

        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "new content");

        // The backup lives under the daemon's data dir, not beside the file.
        let backup = crate::tools::fs_utils::local_backup_path(
            &crate::tools::file_policy::file_backup_root().unwrap(),
            &file_path,
        )
        .unwrap();
        let backup_content = tokio::fs::read_to_string(&backup).await.unwrap();
        assert_eq!(backup_content, "old content");
        assert!(!file_path.with_extension("txt.bak").exists());
        let _ = tokio::fs::remove_file(&backup).await;
    }

    #[tokio::test]
    async fn test_write_sensitive_path_blocked() {
        let args = json!({
            "path": "/tmp/.ssh/test_key",
            "content": "secret"
        })
        .to_string();

        let result = crate::tools::guarded_call(
            &WriteFileTool::new(crate::execution::active_execution_backend()),
            &args,
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sensitive"));
    }

    #[tokio::test]
    async fn test_write_create_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("a").join("b").join("file.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "nested",
            "create_dirs": true
        })
        .to_string();

        let result = WriteFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("Created"));
    }

    #[tokio::test]
    async fn test_write_auto_creates_parent_dirs_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("auto_created").join("file.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "hello"
        })
        .to_string();

        let result = WriteFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("Created"));
        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "hello");
    }

    #[tokio::test]
    async fn test_write_no_parent_dir_when_create_dirs_false() {
        let args = json!({
            "path": "/tmp/nonexistent_dir_12345/file.txt",
            "content": "hello",
            "create_dirs": false
        })
        .to_string();

        let result = WriteFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Parent directory"));
    }

    #[test]
    fn test_write_file_capabilities_match_no_approval_tool_guidance() {
        let caps = WriteFileTool::new(crate::execution::active_execution_backend()).capabilities();
        assert!(
            !caps.needs_approval,
            "write_file is documented as a dedicated file tool that does not require approval"
        );
        assert!(!caps.high_impact_write);
    }

    #[tokio::test]
    async fn test_append_creates_new_file_with_hint() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("appended.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "chunk one\n",
            "mode": "append"
        })
        .to_string();

        let result = WriteFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("Appended"), "result was: {result}");
        // Continuation hint must tell the model how to add more.
        assert!(result.contains("mode=\"append\""), "result was: {result}");

        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "chunk one\n");
    }

    #[tokio::test]
    async fn test_append_preserves_existing_and_no_backup() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("doc.txt");
        tokio::fs::write(&file_path, "first\n").await.unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "second\n",
            "mode": "append"
        })
        .to_string();
        WriteFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();

        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "first\nsecond\n");

        // Append must NOT create a backup (it is additive).
        let backup = file_path.with_extension("txt.bak");
        assert!(!backup.exists(), "append must not create a backup");
    }

    #[tokio::test]
    async fn test_invalid_mode_errors_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("untouched.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "x",
            "mode": "garbage"
        })
        .to_string();

        let result = WriteFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid mode"));
        assert!(!file_path.exists(), "invalid mode must not write the file");
    }

    #[tokio::test]
    async fn test_append_sensitive_path_blocked() {
        let args = json!({
            "path": "/tmp/.ssh/test_key",
            "content": "secret",
            "mode": "append"
        })
        .to_string();

        let result = crate::tools::guarded_call(
            &WriteFileTool::new(crate::execution::active_execution_backend()),
            &args,
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sensitive"));
    }

    #[tokio::test]
    async fn test_explicit_overwrite_mode_matches_default() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("ow.txt");
        tokio::fs::write(&file_path, "old").await.unwrap();
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "new",
            "mode": "overwrite"
        })
        .to_string();

        let result = WriteFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("Updated"));
        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "new");
    }

    #[test]
    fn declares_write_of_the_absolute_path() {
        let dir = format!("/tmp/aidaemon-test/{}", uuid::Uuid::new_v4());
        let path = format!("{dir}/notes/../acme.md");
        for key in ["path", "file_path"] {
            let effects = WriteFileTool::new(crate::execution::active_execution_backend())
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
        let relative = WriteFileTool::new(crate::execution::active_execution_backend())
            .call_effects(&json!({"path": "acme.md"}).to_string());
        assert_eq!(relative, aidaemon_core::CallEffects::unknown());
    }
}
