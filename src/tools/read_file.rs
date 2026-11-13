//! `read_file` is implemented in `aidaemon-tools`. This module holds the daemon's
//! wording and policy for it, and the tests of its behaviour on the daemon's
//! execution backend.

use aidaemon_tools::ToolText;

#[cfg(test)]
use crate::traits::{
    AttachmentProvenance, ReadFileResultMetadata, ReadFileSelectionMetadata, Tool,
};
#[cfg(test)]
use aidaemon_tools::render_read_file_output_within;
#[cfg(test)]
use aidaemon_tools::{render_read_file_output, MAX_READ_CHARS};
#[cfg(test)]
use serde_json::json;

use super::file_policy::{daemon_file_policy, DaemonPaths};
use crate::execution::SharedExecutionBackend;

/// The daemon's constructors for the SDK's [`aidaemon_tools::ReadFile`].
pub struct ReadFileTool;

#[allow(clippy::new_ret_no_self)]
impl ReadFileTool {
    /// Without the daemon's own files protected; production passes them.
    #[cfg(test)]
    pub fn new(backend: SharedExecutionBackend) -> aidaemon_tools::ReadFile {
        Self::on_backend(backend, DaemonPaths::default())
    }

    pub fn on_backend(
        backend: SharedExecutionBackend,
        paths: DaemonPaths,
    ) -> aidaemon_tools::ReadFile {
        aidaemon_tools::ReadFile::on_backend(backend, daemon_file_policy(paths)).with_text(ToolText {
                short: "Read file contents with line numbers and metadata. Read a fitting file in full once; for large files, search first and then read one exact non-overlapping range.".to_string(),
                schema: "Read file contents with line numbers and metadata. Use this instead of terminal cat/head/tail. Read a file in full once when it fits the limit; oversized reads return the first page plus an exact continuation call to follow. For a large file with a known target, use search_files first, then read one exact surrounding range. Sequential ranges must not overlap; previously covered content is replayed from the task-local artifact.".to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ToolCalls as _;
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = ReadFileTool::new(crate::execution::active_execution_backend());
        let schema = tool.schema();
        assert_eq!(schema["name"], "read_file");
        assert!(!schema["description"].as_str().unwrap().is_empty());
        assert!(schema["parameters"]["properties"]["path"].is_object());
    }

    async fn read_in_tempdir(name: &str, body: &str) -> anyhow::Result<String> {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(name);
        std::fs::write(&file, body).unwrap();
        let args = json!({"path": file.to_str().unwrap()}).to_string();
        crate::tools::guarded_call(
            &ReadFileTool::new(crate::execution::active_execution_backend()),
            &args,
        )
        .await
    }

    #[tokio::test]
    async fn read_file_refuses_ssh_private_key() {
        let dir = tempfile::tempdir().unwrap();
        let ssh = dir.path().join(".ssh");
        std::fs::create_dir(&ssh).unwrap();
        let key = ssh.join("id_ed25519");
        std::fs::write(&key, "synthetic-private-key-material\n").unwrap();
        let args = json!({"path": key.to_str().unwrap()}).to_string();
        let out = crate::tools::guarded_call(
            &ReadFileTool::new(crate::execution::active_execution_backend()),
            &args,
        )
        .await;
        let text = match out {
            Ok(text) => text,
            Err(error) => error.to_string(),
        };
        assert!(text.contains("read_file does not open credential or secret files"));
        assert!(!text.contains("synthetic-private-key-material"));
    }

    #[tokio::test]
    async fn read_file_refuses_dotenv() {
        let text = match read_in_tempdir(".env", "API_TOKEN=synthetic-token\n").await {
            Ok(text) => text,
            Err(error) => error.to_string(),
        };
        assert!(text.contains("read_file does not open credential or secret files"));
        assert!(text.contains("Ask the owner to share what is needed."));
        assert!(!text.contains("synthetic-token"));
    }

    #[tokio::test]
    async fn read_file_allows_env_example() {
        for name in [".env.example", ".env.sample", ".env.template"] {
            let text = read_in_tempdir(name, "API_TOKEN=changeme\n").await.unwrap();
            assert!(text.contains("API_TOKEN=changeme"), "{name}: {text}");
        }
    }

    #[tokio::test]
    async fn read_file_allows_git_head_and_config() {
        let dir = tempfile::tempdir().unwrap();
        let git = dir.path().join(".git");
        std::fs::create_dir(&git).unwrap();
        std::fs::write(git.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        std::fs::write(git.join("config"), "[core]\n\tbare = false\n").unwrap();
        for (name, needle) in [("HEAD", "refs/heads/main"), ("config", "bare = false")] {
            let args = json!({"path": git.join(name).to_str().unwrap()}).to_string();
            let text = ReadFileTool::new(crate::execution::active_execution_backend())
                .call(&args)
                .await
                .unwrap();
            assert!(text.contains(needle), "{name}: {text}");
        }
    }

    #[tokio::test]
    async fn read_file_refuses_the_daemons_own_config() {
        let dir = tempfile::tempdir().unwrap();
        let own = dir.path().join("daemon-config.toml");
        std::fs::write(&own, "token = \"synthetic-secret\"\n").unwrap();
        let paths = crate::tools::DaemonPaths::new(vec![own.clone()]);
        let args = json!({"path": own.to_str().unwrap()}).to_string();
        let text = crate::tools::guarded_call_protecting(
            paths,
            &ReadFileTool::new(crate::execution::active_execution_backend()),
            &args,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(text.contains("does not open credential or secret files"));
        assert!(text.contains("manage_config action 'read'"), "{text}");
        assert!(!text.contains("synthetic-secret"));
        let other = dir.path().join("config.toml");
        std::fs::write(&other, "title = \"ok\"\n").unwrap();
        let args = json!({"path": other.to_str().unwrap()}).to_string();
        assert!(
            ReadFileTool::new(crate::execution::active_execution_backend())
                .call(&args)
                .await
                .unwrap()
                .contains("title")
        );
    }

    #[tokio::test]
    async fn read_file_refuses_env_example_under_a_secret_directory() {
        let dir = tempfile::tempdir().unwrap();
        let ssh = dir.path().join(".ssh");
        std::fs::create_dir(&ssh).unwrap();
        std::fs::write(ssh.join(".env.example"), "synthetic-secret\n").unwrap();
        let args = json!({"path": ssh.join(".env.example").to_str().unwrap()}).to_string();
        let text = crate::tools::guarded_call(
            &ReadFileTool::new(crate::execution::active_execution_backend()),
            &args,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(text.contains("does not open credential or secret files"));
        assert!(!text.contains("synthetic-secret"));
    }

    #[tokio::test]
    async fn read_file_allows_regular_source() {
        let text = read_in_tempdir("main.rs", "fn main() {}\n").await.unwrap();
        assert!(text.contains("fn main() {}"));
    }

    #[tokio::test]
    async fn test_read_file_basic() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "line one\nline two\nline three\n").unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();
        let result = ReadFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("line one"));
        assert!(result.contains("line two"));
        assert!(result.contains("line three"));
        assert!(result.contains("3 lines"));
        assert!(result.contains("bytes"));
        assert!(result.contains("modified"));
    }

    #[tokio::test]
    async fn test_read_file_outcome_includes_complete_typed_metadata() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "line one\nline two\nline three\n").unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();
        let metadata = outcome
            .metadata
            .read_file
            .expect("text reads should expose typed metadata");

        assert_eq!(metadata.display_path, f.path().to_str().unwrap());
        assert_eq!(metadata.returned_start_line, Some(1));
        assert_eq!(metadata.returned_end_line, Some(3));
        assert_eq!(metadata.total_lines, 3);
        assert_eq!(
            metadata.selected_lines,
            vec!["line one", "line two", "line three"]
        );
        assert!(matches!(
            metadata.selection,
            ReadFileSelectionMetadata::Full
        ));
    }

    #[tokio::test]
    async fn test_read_file_range_outcome_metadata_preserves_raw_lines() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "a\nb\nc\nd\ne\n").unwrap();
        let args =
            json!({"path": f.path().to_str().unwrap(), "start_line": 2, "end_line": 4}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();
        let metadata = outcome.metadata.read_file.unwrap();

        assert_eq!(metadata.returned_start_line, Some(2));
        assert_eq!(metadata.returned_end_line, Some(4));
        assert_eq!(metadata.selected_lines, vec!["b", "c", "d"]);
        assert!(matches!(
            metadata.selection,
            ReadFileSelectionMetadata::BoundedRange { .. }
        ));
    }

    #[tokio::test]
    async fn test_empty_file_is_a_typed_full_artifact() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();
        let metadata = outcome.metadata.read_file.unwrap();

        assert!(matches!(
            metadata.selection,
            ReadFileSelectionMetadata::Full
        ));
        assert_eq!(metadata.total_lines, 0);
        assert_eq!(metadata.returned_start_line, None);
        assert_eq!(metadata.returned_end_line, None);
        assert!(metadata.selected_lines.is_empty());
    }

    #[tokio::test]
    async fn test_binary_file_has_no_typed_read_artifact() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0xFF, 0xD8, 0xFF, 0x00, 0x10, 0x00]).unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();

        assert!(outcome.metadata.read_file.is_none());
    }

    #[tokio::test]
    async fn test_read_file_line_range() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "a\nb\nc\nd\ne\n").unwrap();
        let args =
            json!({"path": f.path().to_str().unwrap(), "start_line": 2, "end_line": 4}).to_string();
        let result = ReadFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("b"));
        assert!(result.contains("c"));
        assert!(result.contains("d"));
        assert!(result.contains("lines 2-4 of 5"));
        assert!(!result.contains("| a"));
        assert!(!result.contains("| e"));
    }

    #[tokio::test]
    async fn conflicting_range_modes_fail_without_reading() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "a\nb\nc\n").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "start_line": 1,
            "end_line": 2,
            "tail_lines": 1
        })
        .to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();
        assert!(outcome
            .output
            .contains("range modes are mutually exclusive"));
        assert!(outcome.metadata.contract_rejected);
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(crate::traits::ToolOutcomeStatus::Blocked)
        );
        assert!(outcome.metadata.read_file.is_none());
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let args = json!({"path": "/tmp/nonexistent_read_file_test_12345.txt"}).to_string();
        let result = ReadFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_read_file_jpeg_image_attaches_vision_metadata() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0xFF, 0xD8, 0xFF, 0x00, 0x10, 0x00]).unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();

        assert!(outcome.output.contains("Image file"));
        assert!(outcome.output.contains("image/jpeg"));
        assert!(outcome.output.contains("Attached for vision analysis"));
        assert!(outcome.metadata.read_file.is_none());
        let attachment = outcome
            .metadata
            .attachments
            .first()
            .expect("jpeg image attachment");
        assert_eq!(attachment.mime_type, "image/jpeg");
        assert_eq!(attachment.provenance, AttachmentProvenance::ToolObservation);
        assert_eq!(attachment.source_tool.as_deref(), Some("read_file"));
        assert!(Path::new(&attachment.local_path).exists());
    }

    #[tokio::test]
    async fn test_read_file_png_image_without_null_bytes() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x01, 0x02])
            .unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();

        assert!(outcome.output.contains("Image file"));
        assert!(outcome.output.contains("image/png"));
        assert_eq!(outcome.metadata.attachments.len(), 1);
    }

    #[tokio::test]
    async fn test_pdf_file_returns_extraction_hint_not_dead_end() {
        let mut f = tempfile::Builder::new().suffix(".pdf").tempfile().unwrap();
        f.write_all(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n")
            .unwrap();
        let path = f.path().to_str().unwrap().to_string();
        let args = json!({"path": &path}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();

        assert!(outcome.output.contains("PDF"));
        assert!(
            outcome.output.contains("pdftotext"),
            "must give an exact terminal extraction command, got: {}",
            outcome.output
        );
        assert!(
            outcome.output.contains(&path),
            "extraction command must include the file path"
        );
        assert!(!outcome.output.contains("cannot display contents"));
        assert!(outcome.metadata.read_file.is_none());
    }

    #[tokio::test]
    async fn test_pdf_detected_by_magic_even_without_null_bytes() {
        // A small text-only PDF can pass the NUL-byte binary check; raw PDF
        // syntax must never be dumped as if it were the document text.
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "%PDF-1.4\nplain ascii body without nulls\n").unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();

        assert!(outcome.output.contains("pdftotext"));
        assert!(!outcome.output.contains("plain ascii body"));
    }

    #[tokio::test]
    async fn test_docx_file_returns_extraction_hint() {
        let mut f = tempfile::Builder::new().suffix(".docx").tempfile().unwrap();
        f.write_all(b"PK\x03\x04docx-zip-payload").unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();

        assert!(
            outcome.output.contains("Word document"),
            "got: {}",
            outcome.output
        );
        assert!(
            outcome.output.contains("textutil") || outcome.output.contains("pandoc"),
            "must suggest a concrete extraction command, got: {}",
            outcome.output
        );
        assert!(!outcome.output.contains("cannot display contents"));
    }

    #[tokio::test]
    async fn test_generic_binary_stub_suggests_identifying_format() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0x00, 0x01, 0x02, 0x03, 0x04]).unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();

        assert!(outcome.output.contains("cannot display contents"));
        assert!(
            outcome.output.contains("terminal"),
            "binary stub must point to the terminal tool instead of dead-ending, got: {}",
            outcome.output
        );
    }

    #[tokio::test]
    async fn test_read_file_non_image_binary_still_stubbed() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0x00, 0x01, 0x02, 0x03, 0x04]).unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();
        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();

        assert!(outcome.output.contains("Binary file"));
        assert!(outcome.output.contains("cannot display contents"));
        assert!(outcome.metadata.attachments.is_empty());
    }

    #[tokio::test]
    async fn test_read_file_directory() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = ReadFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("directory"));
    }

    #[tokio::test]
    async fn test_read_file_empty() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();
        let result = ReadFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();
        assert!(result.contains("0 lines"));
        assert!(result.contains("0 bytes"));
    }

    #[tokio::test]
    async fn test_long_lines_truncated_with_marker() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "short line").unwrap();
        writeln!(f, "{}", "z".repeat(10_000)).unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();

        assert!(outcome.output.contains("line truncated"));
        assert!(!outcome.output.contains(&"z".repeat(3_000)));
        let metadata = outcome.metadata.read_file.unwrap();
        assert!(
            metadata.selected_lines[1].chars().count() < 2_100,
            "stored line should be capped, got {} chars",
            metadata.selected_lines[1].chars().count()
        );
        assert_eq!(metadata.selected_lines[0], "short line");
    }

    #[tokio::test]
    async fn test_unbounded_range_read_is_capped_with_continuation_hint() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for i in 1..=25_000 {
            writeln!(f, "line number {} with some padding text", i).unwrap();
        }
        let args = json!({"path": f.path().to_str().unwrap(), "start_line": 1, "end_line": 25_000})
            .to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();
        let metadata = outcome.metadata.read_file.clone().unwrap();

        assert!(metadata.truncated, "capped range read must set truncated");
        assert!(
            metadata.selected_lines.len() < 25_000,
            "returned {} lines, expected capped subset",
            metadata.selected_lines.len()
        );
        assert_eq!(metadata.total_lines, 25_000);
        let end = metadata.returned_end_line.unwrap();
        assert!(
            outcome
                .output
                .contains(&format!("\"start_line\": {}", end + 1)),
            "output must include an exact continuation call; output tail: {}",
            &outcome.output[outcome.output.len().saturating_sub(400)..]
        );
        assert!(outcome.output.contains("NOT returned"));
    }

    #[tokio::test]
    async fn test_full_read_of_large_file_returns_first_page() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for i in 1..=25_000 {
            writeln!(f, "line number {} with some padding text", i).unwrap();
        }
        assert!(f.as_file().metadata().unwrap().len() > MAX_READ_CHARS as u64);
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .expect("oversized full read should auto-page, not error");
        let metadata = outcome.metadata.read_file.clone().unwrap();

        assert!(metadata.truncated);
        assert_eq!(metadata.returned_start_line, Some(1));
        assert_eq!(metadata.total_lines, 25_000);
        assert!(outcome.output.contains("| line number 1 "));
        assert!(outcome.output.contains("NOT returned"));
        let end = metadata.returned_end_line.unwrap();
        assert!(outcome
            .output
            .contains(&format!("\"start_line\": {}", end + 1)));
    }

    #[tokio::test]
    async fn test_huge_tail_lines_request_is_bounded() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "a\nb\nc\n").unwrap();
        let args =
            json!({"path": f.path().to_str().unwrap(), "tail_lines": 1_000_000_u64}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();
        let metadata = outcome.metadata.read_file.unwrap();

        assert_eq!(metadata.selected_lines, vec!["a", "b", "c"]);
        assert!(!metadata.truncated);
    }

    #[tokio::test]
    async fn test_small_full_read_has_no_truncation_hint() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "alpha\nbeta\ngamma\n").unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool::new(crate::execution::active_execution_backend())
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();
        let metadata = outcome.metadata.read_file.unwrap();

        assert!(!metadata.truncated);
        assert!(!outcome.output.contains("NOT returned"));
        assert!(outcome.output.contains("3 lines"));
    }

    fn render_test_metadata(line_count: usize) -> ReadFileResultMetadata {
        ReadFileResultMetadata {
            display_path: "/tmp/example.txt".to_string(),
            canonical_path: "/tmp/example.txt".to_string(),
            selection: ReadFileSelectionMetadata::Full,
            returned_start_line: Some(1),
            returned_end_line: Some(line_count),
            total_lines: line_count,
            file_size: (line_count * 32) as u64,
            modified: Some("2026-06-10T00:00:00+00:00".to_string()),
            selected_lines: (1..=line_count)
                .map(|i| format!("content line {} with some padding", i))
                .collect(),
            truncated: false,
        }
    }

    #[test]
    fn test_render_within_budget_identical_when_it_fits() {
        let metadata = render_test_metadata(5);
        assert_eq!(
            render_read_file_output_within(&metadata, 10_000),
            render_read_file_output(&metadata)
        );
    }

    #[test]
    fn test_render_within_budget_cuts_on_line_boundary_with_hint() {
        let metadata = render_test_metadata(500);
        let out = render_read_file_output_within(&metadata, 2_000);

        assert!(
            out.chars().count() <= 2_400,
            "output should respect budget (+slack), got {} chars",
            out.chars().count()
        );
        assert!(out.contains("NOT returned"));
        assert!(out.contains("\"start_line\": "));
        // Every rendered content line must be complete — cut between lines,
        // never inside one.
        for line in out.lines() {
            if let Some((_, content)) = line.split_once(" | ") {
                assert!(
                    content.ends_with("with some padding"),
                    "line cut mid-content: {line:?}"
                );
            }
        }
        // The hint must point at the first line that was dropped.
        let last_shown = out
            .lines()
            .filter_map(|line| {
                line.split_once(" | ")
                    .and_then(|(n, _)| n.trim().parse::<usize>().ok())
            })
            .max()
            .unwrap();
        assert!(out.contains(&format!("\"start_line\": {}", last_shown + 1)));
    }

    #[test]
    fn test_render_within_budget_tail_keeps_most_recent_lines() {
        let mut metadata = render_test_metadata(500);
        metadata.selection = ReadFileSelectionMetadata::Tail {
            requested_lines: 500,
        };
        let out = render_read_file_output_within(&metadata, 2_000);

        assert!(
            out.contains("| content line 500 "),
            "tail render must keep the final lines"
        );
        assert!(
            !out.contains("| content line 1 "),
            "tail render must drop the oldest lines first"
        );
    }

    #[test]
    fn test_render_within_budget_always_keeps_at_least_one_line() {
        let metadata = render_test_metadata(50);
        let out = render_read_file_output_within(&metadata, 10);
        assert!(out.contains("| content line 1 "));
    }

    #[tokio::test]
    async fn test_read_large_file_with_line_range() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for i in 1..=25_000 {
            writeln!(f, "line {}", i).unwrap();
        }
        assert!(f.as_file().metadata().unwrap().len() > MAX_READ_CHARS as u64);

        let args =
            json!({"path": f.path().to_str().unwrap(), "start_line": 24998, "end_line": 25000})
                .to_string();
        let result = ReadFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();

        assert!(result.contains("lines 24998-25000 of 25000"));
        assert!(result.contains("24998 | line 24998"));
        assert!(result.contains("25000 | line 25000"));
        assert!(!result.contains("24997 | line 24997"));
    }

    #[tokio::test]
    async fn test_read_large_file_with_tail_lines() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for i in 1..=25_000 {
            writeln!(f, "line {}", i).unwrap();
        }
        assert!(f.as_file().metadata().unwrap().len() > MAX_READ_CHARS as u64);

        let args = json!({"path": f.path().to_str().unwrap(), "tail_lines": 3}).to_string();
        let result = ReadFileTool::new(crate::execution::active_execution_backend())
            .call(&args)
            .await
            .unwrap();

        assert!(result.contains("last 3 lines of 25000"));
        assert!(result.contains("24998 | line 24998"));
        assert!(result.contains("25000 | line 25000"));
        assert!(!result.contains("24997 | line 24997"));
    }
}
