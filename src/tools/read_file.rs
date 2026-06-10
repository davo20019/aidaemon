use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

use crate::traits::{
    AttachmentProvenance, MessageAttachment, ReadFileResultMetadata, ReadFileSelectionMetadata,
    Tool, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolRole,
    ToolTargetHintKind, ToolVerificationMode,
};
use crate::types::StatusUpdate;

use super::fs_utils;

pub struct ReadFileTool;

/// Per-call cap on returned content (bytes of line text). Reads that would
/// exceed it return the first lines that fit plus a continuation hint
/// instead of an unbounded payload.
const MAX_READ_CHARS: usize = 100 * 1024;
/// Individual lines longer than this (minified JS, JSONL, ...) are truncated
/// with a marker; line-based paging can't help inside a single line.
const MAX_LINE_CHARS: usize = 2_000;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read file contents with line numbers and metadata. Read a fitting file in full once; for large files, search first and then read one exact non-overlapping range."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "read_file",
            "description": "Read file contents with line numbers and metadata. Use this instead of terminal cat/head/tail. Read a file in full once when it fits the limit; oversized reads return the first page plus an exact continuation call to follow. For a large file with a known target, use search_files first, then read one exact surrounding range. Sequential ranges must not overlap; previously covered content is replayed from the task-local artifact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (supports ~ expansion)"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line number (1-based, inclusive). Omit to start from beginning."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line number (1-based, inclusive). Omit to read to end."
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "Read the last N lines of the file. Useful for large logs."
                    }
                },
                "required": ["path"],
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

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let path = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|args| {
                for key in ["path", "file_path", "file", "filename"] {
                    if let Some(path) = args.get(key).and_then(|value| value.as_str()) {
                        return Some(path.to_string());
                    }
                }
                None
            })
            .unwrap_or_default();

        ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent)
            .with_target_hint(ToolTargetHintKind::Path, path)
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        Ok(self.read_outcome(arguments).await?.output)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<tokio::sync::mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.read_outcome(arguments).await
    }
}

impl ReadFileTool {
    async fn read_outcome(&self, arguments: &str) -> anyhow::Result<ToolCallOutcome> {
        let args: Value = serde_json::from_str(arguments)?;
        // Parameter aliasing: models often use "file_path" or "file" instead of "path"
        let path_str = args["path"]
            .as_str()
            .or_else(|| args["file_path"].as_str())
            .or_else(|| args["file"].as_str())
            .or_else(|| args["filename"].as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;

        let path = fs_utils::validate_path(path_str)?;

        if !path.exists() {
            anyhow::bail!("File not found: {}", path_str);
        }

        let metadata = tokio::fs::metadata(&path).await?;

        if metadata.is_dir() {
            anyhow::bail!("Path is a directory, not a file: {}", path_str);
        }

        let file_size = metadata.len();
        let modified = format_modified_rfc3339(&metadata);

        if let Some(mime_type) = sniff_file_image_mime(&path).await? {
            return Ok(image_file_outcome(
                path_str,
                &path,
                file_size,
                modified.as_deref(),
                mime_type,
            ));
        }

        // Documents (PDF, Word) can't be decoded here, but dead-ending with
        // "binary" makes models give up or ask the user to convert the file.
        // Return an exact terminal extraction command instead. Checked before
        // the NUL-byte binary test: small text-only PDFs contain no NUL bytes
        // and would otherwise be dumped as raw PDF syntax.
        if let Some(kind) = sniff_document_kind(&path).await? {
            return Ok(ToolCallOutcome::from_output(document_extraction_stub(
                kind,
                path_str,
                file_size,
                modified.as_deref(),
            )));
        }

        // Check for binary
        if fs_utils::is_binary_file(&path).await? {
            let mut out = format!("Binary file: {}\nSize: {} bytes\n", path_str, file_size);
            if let Some(modified) = &modified {
                out.push_str(&format!("Modified: {}\n", modified));
            }
            out.push_str("Type: binary (cannot display contents).\n");
            out.push_str(&format!(
                "If you need its contents, run the terminal tool with: file \"{}\" to identify the format, then choose a matching extraction command.",
                shell_safe_path(path_str)
            ));
            return Ok(ToolCallOutcome::from_output(out));
        }

        let start = args["start_line"]
            .as_u64()
            .map(|n| (n as usize).saturating_sub(1))
            .unwrap_or(0);

        let end = args["end_line"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(usize::MAX);

        let tail_lines = args["tail_lines"]
            .as_u64()
            .or_else(|| args["last_lines"].as_u64())
            .or_else(|| args["last_n_lines"].as_u64())
            .map(|n| n as usize);

        if matches!(tail_lines, Some(0)) {
            anyhow::bail!("tail_lines must be at least 1");
        }

        let uses_subset = start > 0 || end != usize::MAX || tail_lines.is_some();
        // Oversized full reads no longer error: the per-call cap below
        // returns the first page with an explicit continuation hint, which
        // small-context models follow far more reliably than an error.

        let selection = if let Some(count) = tail_lines {
            ReadSelection::Tail { count }
        } else if uses_subset {
            ReadSelection::Range {
                start,
                end_exclusive: (end != usize::MAX).then_some(end),
            }
        } else {
            ReadSelection::Full
        };

        let selected = read_selected_lines(&path, selection).await?;
        let total_lines = selected.total_lines;
        let selection_metadata = match selection {
            ReadSelection::Full => ReadFileSelectionMetadata::Full,
            ReadSelection::Range {
                start,
                end_exclusive: Some(end),
            } => ReadFileSelectionMetadata::BoundedRange {
                start_line: start + 1,
                end_line: end,
            },
            ReadSelection::Range {
                start,
                end_exclusive: None,
            } => ReadFileSelectionMetadata::OpenEndedRange {
                start_line: start + 1,
            },
            ReadSelection::Tail { count } => ReadFileSelectionMetadata::Tail {
                requested_lines: count,
            },
        };
        let canonical_path = tokio::fs::canonicalize(&path)
            .await
            .unwrap_or_else(|_| path.clone())
            .to_string_lossy()
            .into_owned();
        let read_metadata = ReadFileResultMetadata {
            display_path: path_str.to_string(),
            canonical_path,
            selection: selection_metadata,
            returned_start_line: (!selected.lines.is_empty()).then_some(selected.start_index + 1),
            returned_end_line: (!selected.lines.is_empty()).then_some(selected.end_display),
            total_lines,
            file_size,
            modified: modified.clone(),
            selected_lines: selected.lines.clone(),
            truncated: selected.truncated,
        };

        if total_lines == 0 {
            let header =
                format_text_file_header(path_str, "0 lines, empty", file_size, modified.as_deref());
            return Ok(ToolCallOutcome {
                output: header.trim_end().to_string(),
                metadata: ToolCallMetadata {
                    read_file: Some(read_metadata),
                    ..ToolCallMetadata::default()
                },
            });
        }

        if selected.start_index >= total_lines {
            anyhow::bail!(
                "start_line {} exceeds total lines {} in file",
                selected.start_index + 1,
                total_lines
            );
        }

        Ok(ToolCallOutcome {
            output: render_read_file_output(&read_metadata),
            metadata: ToolCallMetadata {
                read_file: Some(read_metadata),
                ..ToolCallMetadata::default()
            },
        })
    }
}

pub(crate) fn render_read_file_output(metadata: &ReadFileResultMetadata) -> String {
    if metadata.total_lines == 0 {
        return format_text_file_header(
            &metadata.display_path,
            "0 lines, empty",
            metadata.file_size,
            metadata.modified.as_deref(),
        )
        .trim_end()
        .to_string();
    }

    let start_line = metadata.returned_start_line.unwrap_or(1);
    let end_line = metadata.returned_end_line.unwrap_or(start_line);
    let header_summary = match metadata.selection {
        ReadFileSelectionMetadata::Full if !metadata.truncated => {
            format!("{} lines", metadata.total_lines)
        }
        ReadFileSelectionMetadata::Full
        | ReadFileSelectionMetadata::BoundedRange { .. }
        | ReadFileSelectionMetadata::OpenEndedRange { .. } => {
            format!(
                "lines {}-{} of {}",
                start_line, end_line, metadata.total_lines
            )
        }
        ReadFileSelectionMetadata::Tail { .. } => format!(
            "last {} lines of {}",
            metadata.selected_lines.len(),
            metadata.total_lines
        ),
    };
    let header = format_text_file_header(
        &metadata.display_path,
        &header_summary,
        metadata.file_size,
        metadata.modified.as_deref(),
    );
    let selected_content = metadata.selected_lines.join("\n");
    let formatted = fs_utils::format_with_line_numbers(&selected_content, start_line - 1);
    match continuation_hint(metadata) {
        Some(hint) => format!("{}{}\n{}", header, formatted, hint),
        None => format!("{}{}", header, formatted),
    }
}

/// Render a read result within a character budget, cutting on line
/// boundaries and appending an explicit continuation hint when lines had to
/// be dropped. Results that fit are rendered unchanged, so big-budget models
/// see exactly what they do today.
///
/// For tail selections the most recent lines are kept (the end of a log is
/// the signal); for everything else the first lines are kept so sequential
/// paging works.
pub(crate) fn render_read_file_output_within(
    metadata: &ReadFileResultMetadata,
    max_chars: usize,
) -> String {
    let full = render_read_file_output(metadata);
    if full.chars().count() <= max_chars {
        return full;
    }

    // Reserve room for the header and continuation hint, then keep whole
    // numbered lines until the budget is spent.
    const HINT_RESERVE: usize = 360;
    let start_line = metadata.returned_start_line.unwrap_or(1);
    let end_line = metadata.returned_end_line.unwrap_or(start_line);
    let keep_from_end = matches!(metadata.selection, ReadFileSelectionMetadata::Tail { .. });
    let number_width = end_line.to_string().len().max(3) + 3; // "NNN | "
    let budget = max_chars.saturating_sub(HINT_RESERVE + 120);

    let mut kept: Vec<String> = Vec::new();
    let mut used = 0usize;
    let lines: Box<dyn Iterator<Item = &String>> = if keep_from_end {
        Box::new(metadata.selected_lines.iter().rev())
    } else {
        Box::new(metadata.selected_lines.iter())
    };
    for line in lines {
        let cost = number_width + line.chars().count() + 1;
        if !kept.is_empty() && used + cost > budget {
            break;
        }
        used += cost;
        kept.push(line.clone());
    }
    if keep_from_end {
        kept.reverse();
    }

    let mut adjusted = metadata.clone();
    adjusted.truncated = true;
    if keep_from_end {
        adjusted.returned_start_line = Some(end_line.saturating_sub(kept.len()).saturating_add(1));
    } else {
        adjusted.returned_end_line = Some(start_line + kept.len() - 1);
    }
    adjusted.selected_lines = kept;
    render_read_file_output(&adjusted)
}

/// Build an explicit, copyable next-call instruction for truncated reads.
/// Small-context models follow a literal tool-call template far more
/// reliably than a generic "output was truncated" notice.
pub(crate) fn continuation_hint(metadata: &ReadFileResultMetadata) -> Option<String> {
    if !metadata.truncated {
        return None;
    }
    let start_line = metadata.returned_start_line?;
    let end_line = metadata.returned_end_line?;
    if matches!(metadata.selection, ReadFileSelectionMetadata::Tail { .. }) {
        return Some(format!(
            "[NOTE: tail output truncated to fit the output limit — only the last {} lines were returned; lines before {} are NOT visible to you. Use start_line/end_line to read earlier sections.]",
            metadata.selected_lines.len(),
            start_line
        ));
    }
    let shown = end_line
        .saturating_sub(start_line)
        .saturating_add(1)
        .max(50);
    let next_start = end_line + 1;
    let next_end = end_line.saturating_add(shown).min(metadata.total_lines);
    Some(format!(
        "[NOTE: output truncated at line {} — lines {}-{} of {} were NOT returned and are NOT visible to you. Do not guess their content. To continue reading, call read_file with {{\"path\": \"{}\", \"start_line\": {}, \"end_line\": {}}}.]",
        end_line,
        next_start,
        metadata.total_lines,
        metadata.total_lines,
        metadata.display_path,
        next_start,
        next_end
    ))
}

#[derive(Clone, Copy)]
enum ReadSelection {
    Full,
    Range {
        start: usize,
        end_exclusive: Option<usize>,
    },
    Tail {
        count: usize,
    },
}

struct SelectedLines {
    lines: Vec<String>,
    total_lines: usize,
    start_index: usize,
    end_display: usize,
    /// True when the per-call output cap dropped lines the request asked for.
    truncated: bool,
}

/// Truncate a single overlong line, keeping a marker so the model knows
/// content is missing rather than absent.
fn cap_line(line: String) -> String {
    if line.len() <= MAX_LINE_CHARS {
        return line;
    }
    let total_chars = line.chars().count();
    if total_chars <= MAX_LINE_CHARS {
        return line;
    }
    let mut kept: String = line.chars().take(MAX_LINE_CHARS).collect();
    kept.push_str(&format!("… [line truncated; {} chars total]", total_chars));
    kept
}

async fn read_selected_lines(
    path: &Path,
    selection: ReadSelection,
) -> anyhow::Result<SelectedLines> {
    let file = tokio::fs::File::open(path).await?;
    let mut reader = BufReader::new(file).lines();

    match selection {
        ReadSelection::Full | ReadSelection::Range { .. } => {
            let (start, end_exclusive) = match selection {
                ReadSelection::Range {
                    start,
                    end_exclusive,
                } => (start, end_exclusive),
                _ => (0, None),
            };

            let mut lines = Vec::new();
            let mut total_lines: usize = 0;
            let mut stored_bytes: usize = 0;
            let mut capped = false;

            while let Some(line) = reader.next_line().await? {
                total_lines += 1;
                let zero_based_index = total_lines - 1;
                let in_range = zero_based_index >= start
                    && end_exclusive.is_none_or(|end| zero_based_index < end);
                if in_range && !capped {
                    let line = cap_line(line);
                    if !lines.is_empty() && stored_bytes + line.len() > MAX_READ_CHARS {
                        capped = true;
                    } else {
                        stored_bytes += line.len() + 1;
                        lines.push(line);
                    }
                }
                // Keep counting to EOF so total_lines (and the continuation
                // hint) stay accurate even after the cap is hit.
            }

            let (end_display, truncated) = if capped {
                (start + lines.len(), true)
            } else {
                (end_exclusive.unwrap_or(total_lines).min(total_lines), false)
            };
            Ok(SelectedLines {
                lines,
                total_lines,
                start_index: start,
                end_display,
                truncated,
            })
        }
        ReadSelection::Tail { count } => {
            let mut lines: VecDeque<String> = VecDeque::new();
            let mut total_lines: usize = 0;
            let mut stored_bytes: usize = 0;

            while let Some(line) = reader.next_line().await? {
                total_lines += 1;
                if lines.len() == count {
                    if let Some(dropped) = lines.pop_front() {
                        stored_bytes = stored_bytes.saturating_sub(dropped.len() + 1);
                    }
                }
                let line = cap_line(line);
                stored_bytes += line.len() + 1;
                lines.push_back(line);
                // Bound memory and output: keep only the most recent lines
                // that fit the per-call cap.
                while stored_bytes > MAX_READ_CHARS && lines.len() > 1 {
                    if let Some(dropped) = lines.pop_front() {
                        stored_bytes = stored_bytes.saturating_sub(dropped.len() + 1);
                    }
                }
            }

            let lines: Vec<String> = lines.into_iter().collect();
            let start_index = total_lines.saturating_sub(lines.len());
            let truncated = lines.len() < count.min(total_lines);
            Ok(SelectedLines {
                lines,
                total_lines,
                start_index,
                end_display: total_lines,
                truncated,
            })
        }
    }
}

/// Document formats `read_file` cannot decode itself but whose text the
/// model can extract with one terminal command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DocumentKind {
    Pdf,
    Word,
}

/// Detect PDF (magic bytes) and Word documents (zip/OLE magic + extension).
async fn sniff_document_kind(path: &Path) -> anyhow::Result<Option<DocumentKind>> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut header = [0u8; 8];
    let n = file.read(&mut header).await?;
    let header = &header[..n];

    if header.starts_with(b"%PDF") {
        return Ok(Some(DocumentKind::Pdf));
    }

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());
    let is_zip = header.starts_with(b"PK\x03\x04");
    let is_ole = header.starts_with(&[0xD0, 0xCF, 0x11, 0xE0]);
    match extension.as_deref() {
        Some("docx") if is_zip => Ok(Some(DocumentKind::Word)),
        Some("doc") if is_ole => Ok(Some(DocumentKind::Word)),
        _ => Ok(None),
    }
}

/// Escape double quotes so the suggested command stays a single shell token
/// even for paths with spaces or quotes.
fn shell_safe_path(path: &str) -> String {
    path.replace('"', "\\\"")
}

fn document_extraction_stub(
    kind: DocumentKind,
    display_path: &str,
    file_size: u64,
    modified: Option<&str>,
) -> String {
    let label = match kind {
        DocumentKind::Pdf => "PDF document",
        DocumentKind::Word => "Word document",
    };
    let mut out = format!("{}: {}\nSize: {} bytes\n", label, display_path, file_size);
    if let Some(modified) = modified {
        out.push_str(&format!("Modified: {}\n", modified));
    }
    let quoted = shell_safe_path(display_path);
    match kind {
        DocumentKind::Pdf => {
            out.push_str(
                "This tool cannot extract PDF text directly. Run the terminal tool to extract it:\n",
            );
            out.push_str(&format!("  pdftotext -layout \"{}\" -\n", quoted));
            #[cfg(target_os = "macos")]
            out.push_str(&format!(
                "If pdftotext is not installed, use Spotlight's extracted text:\n  mdls -raw -name kMDItemTextContent \"{}\"\n",
                quoted
            ));
            #[cfg(not(target_os = "macos"))]
            out.push_str(
                "If pdftotext is not installed, install poppler-utils (e.g. apt install poppler-utils) or use pandoc.\n",
            );
        }
        DocumentKind::Word => {
            out.push_str(
                "This tool cannot extract Word document text directly. Run the terminal tool to extract it:\n",
            );
            #[cfg(target_os = "macos")]
            out.push_str(&format!("  textutil -convert txt -stdout \"{}\"\n", quoted));
            #[cfg(not(target_os = "macos"))]
            out.push_str(&format!("  pandoc -t plain \"{}\"\n", quoted));
        }
    }
    out.push_str("Do not ask the user to convert the file — extract it yourself.");
    out
}

async fn sniff_file_image_mime(path: &Path) -> anyhow::Result<Option<&'static str>> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut header = [0u8; 16];
    let n = file.read(&mut header).await?;
    Ok(crate::channels::attachments::sniff_image_mime(&header[..n]))
}

fn image_file_outcome(
    display_path: &str,
    path: &Path,
    file_size: u64,
    modified: Option<&str>,
    mime_type: &str,
) -> ToolCallOutcome {
    let mut output = format!("Image file: {}\nSize: {} bytes\n", display_path, file_size);
    if let Some(modified) = modified {
        output.push_str(&format!("Modified: {}\n", modified));
    }
    output.push_str(&format!("Type: {mime_type}\n"));
    output.push_str("Attached for vision analysis.");

    let canonical_path = std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned();
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image")
        .to_string();

    ToolCallOutcome {
        output,
        metadata: ToolCallMetadata {
            attachments: vec![MessageAttachment {
                local_path: canonical_path,
                filename,
                mime_type: mime_type.to_string(),
                size_bytes: file_size,
                provenance: AttachmentProvenance::ToolObservation,
                source_tool: Some("read_file".to_string()),
            }],
            ..ToolCallMetadata::default()
        },
    }
}

fn format_modified_rfc3339(metadata: &std::fs::Metadata) -> Option<String> {
    let modified = metadata.modified().ok()?;
    let modified_utc: chrono::DateTime<chrono::Utc> = modified.into();
    Some(modified_utc.to_rfc3339())
}

fn format_text_file_header(
    path: &str,
    line_summary: &str,
    file_size: u64,
    modified: Option<&str>,
) -> String {
    match modified {
        Some(modified) => format!(
            "File: {} ({}, {} bytes, modified {})\n",
            path, line_summary, file_size, modified
        ),
        None => format!("File: {} ({}, {} bytes)\n", path, line_summary, file_size),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = ReadFileTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "read_file");
        assert!(!schema["description"].as_str().unwrap().is_empty());
        assert!(schema["parameters"]["properties"]["path"].is_object());
    }

    #[tokio::test]
    async fn test_read_file_basic() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "line one\nline two\nline three\n").unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();
        let result = ReadFileTool.call(&args).await.unwrap();
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

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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
        let result = ReadFileTool.call(&args).await.unwrap();
        assert!(result.contains("b"));
        assert!(result.contains("c"));
        assert!(result.contains("d"));
        assert!(result.contains("lines 2-4 of 5"));
        assert!(!result.contains("| a"));
        assert!(!result.contains("| e"));
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let args = json!({"path": "/tmp/nonexistent_read_file_test_12345.txt"}).to_string();
        let result = ReadFileTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_read_file_jpeg_image_attaches_vision_metadata() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0xFF, 0xD8, 0xFF, 0x00, 0x10, 0x00]).unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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
        let outcome = ReadFileTool
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
        let result = ReadFileTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("directory"));
    }

    #[tokio::test]
    async fn test_read_file_empty() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();
        let result = ReadFileTool.call(&args).await.unwrap();
        assert!(result.contains("0 lines"));
        assert!(result.contains("0 bytes"));
    }

    #[tokio::test]
    async fn test_long_lines_truncated_with_marker() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "short line").unwrap();
        writeln!(f, "{}", "z".repeat(10_000)).unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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

        let outcome = ReadFileTool
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
        let result = ReadFileTool.call(&args).await.unwrap();

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
        let result = ReadFileTool.call(&args).await.unwrap();

        assert!(result.contains("last 3 lines of 25000"));
        assert!(result.contains("24998 | line 24998"));
        assert!(result.contains("25000 | line 25000"));
        assert!(!result.contains("24997 | line 24997"));
    }
}
