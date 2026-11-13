//! `read_file`: numbered lines with typed metadata, plus attachments for
//! images and extraction hints for documents.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use aidaemon_core::{AccessManifest, ToolAnnotations};

use super::{FilePolicy, SharedBackend, ToolText, path_arg, workspace_parts};
use crate::backend::{BackendMetadata, BackendPath};
use crate::outcome::{OutcomeTool, core_tool};
use crate::semantics::{
    AttachmentProvenance, MessageAttachment, ReadFileResultMetadata, ReadFileSelectionMetadata,
    ToolCallContext, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolRole, ToolTargetHintKind, ToolVerificationMode,
};
use crate::workspace::Workspace;

/// Per-call cap on returned content (bytes of line text). Reads that would
/// exceed it return the first lines that fit plus a continuation hint
/// instead of an unbounded payload.
pub const MAX_READ_CHARS: usize = 100 * 1024;
/// Individual lines longer than this (minified JS, JSONL, ...) are truncated
/// with a marker; line-based paging can't help inside a single line.
const MAX_LINE_CHARS: usize = 2_000;

pub struct ReadFile {
    backend: SharedBackend,
    policy: Arc<dyn FilePolicy>,
    text: ToolText,
}

impl ReadFile {
    /// A tool confined to `workspace`.
    pub fn new(workspace: Workspace) -> Self {
        let (backend, policy) = workspace_parts(workspace);
        Self::on_backend(backend, policy)
    }

    /// A tool working through `backend` under an application's `policy`.
    pub fn on_backend(backend: SharedBackend, policy: Arc<dyn FilePolicy>) -> Self {
        Self {
            backend,
            policy,
            text: ToolText {
                short: "Read a text file with line numbers and metadata".into(),
                schema: "Read a file with line numbers and metadata. Read a file in full once when it fits the limit; oversized reads return the first page plus an exact continuation call to follow.".into(),
            },
        }
    }

    /// Replace the model-facing descriptions.
    pub fn with_text(mut self, text: ToolText) -> Self {
        self.text = text;
        self
    }
}

#[async_trait]
impl OutcomeTool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        &self.text.short
    }

    fn schema(&self) -> Value {
        json!({
            "name": "read_file",
            "description": self.text.schema,
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
                        "minimum": 1,
                        "description": "Read the last N lines of the file. Useful for large logs."
                    }
                },
                "required": ["path"],
                "allOf": [
                    {"not": {"required": ["tail_lines", "start_line"]}},
                    {"not": {"required": ["tail_lines", "end_line"]}}
                ],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.read_outcome(arguments, crate::semantics::file_fence(facts))
            .await
    }
}

core_tool!(
    ReadFile,
    annotations: ToolAnnotations::READ_ONLY,
    access: |this, arguments| AccessManifest {
        reads: path_arg(arguments)
            .map(|path| vec![this.policy.declare_path(path)])
            .unwrap_or_default(),
        ..Default::default()
    }
);

impl crate::semantics::ToolSemantics for ReadFile {
    fn dispatch_facts(&self) -> crate::semantics::ToolDispatchFacts {
        crate::semantics::ToolDispatchFacts {
            same_tool_streak: crate::semantics::SameToolStreak::Versatile,
            file_guard: crate::semantics::FileGuard::Reads,
            ..Default::default()
        }
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

    fn output_trust(
        &self,
        arguments: &str,
        roots: &crate::semantics::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        crate::semantics::file_output_trust(
            arguments,
            &["path", "file_path", "file", "filename"],
            roots,
        )
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
}

impl ReadFile {
    pub(crate) async fn read_outcome(
        &self,
        arguments: &str,
        fence: Option<&Arc<dyn crate::semantics::FileFence>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: Value = serde_json::from_str(arguments)?;
        // Parameter aliasing: models often use "file_path" or "file" instead of "path"
        let path_str = args["path"]
            .as_str()
            .or_else(|| args["file_path"].as_str())
            .or_else(|| args["file"].as_str())
            .or_else(|| args["filename"].as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;

        let has_start = args.get("start_line").is_some_and(|value| !value.is_null());
        let has_end = args.get("end_line").is_some_and(|value| !value.is_null());
        let tail_value = args
            .get("tail_lines")
            .or_else(|| args.get("last_lines"))
            .or_else(|| args.get("last_n_lines"));
        let has_tail = tail_value.is_some_and(|value| !value.is_null());
        if has_tail && (has_start || has_end) {
            return Ok(ToolCallOutcome::contract_rejection(
                "read_file range modes are mutually exclusive: use start_line/end_line or tail_lines, not both",
            ));
        }
        let requested_start = args.get("start_line").and_then(Value::as_u64);
        let requested_end = args.get("end_line").and_then(Value::as_u64);
        if requested_start.is_some_and(|line| line == 0)
            || requested_end.is_some_and(|line| line == 0)
        {
            return Ok(ToolCallOutcome::contract_rejection(
                "read_file line numbers are 1-based and must be positive",
            ));
        }
        if matches!((requested_start, requested_end), (Some(start), Some(end)) if end < start) {
            return Ok(ToolCallOutcome::contract_rejection(
                "read_file end_line must be greater than or equal to start_line",
            ));
        }

        let backend = self.backend.clone();
        let path = backend.resolve_path(path_str).await?;
        let metadata = backend
            .metadata(&path)
            .await
            .map_err(|_| anyhow::anyhow!("File not found: {}", path_str))?;

        if metadata.is_dir() {
            anyhow::bail!("Path is a directory, not a file: {}", path_str);
        }

        let file_size = metadata.len;
        let modified = format_modified_rfc3339(&metadata);
        let bytes = backend.read(&path).await?;
        if let Some(fence) = fence {
            fence.observed(path_str, &bytes).await;
        }

        if let Some(mime_type) =
            sniff_file_image_mime(&bytes).filter(|_| self.policy.attaches_images())
        {
            return image_file_outcome(
                backend,
                path_str,
                &path,
                file_size,
                modified.as_deref(),
                mime_type,
            )
            .await;
        }

        // Documents (PDF, Word) can't be decoded here, but dead-ending with
        // "binary" makes models give up or ask the user to convert the file.
        // Return an exact terminal extraction command instead. Checked before
        // the NUL-byte binary test: small text-only PDFs contain no NUL bytes
        // and would otherwise be dumped as raw PDF syntax.
        if let Some(kind) = sniff_document_kind(&path, &bytes) {
            return Ok(ToolCallOutcome::from_output(document_extraction_stub(
                kind,
                path_str,
                file_size,
                modified.as_deref(),
                self.policy.shell_tool_name(),
            )));
        }

        // Check for binary
        if bytes.iter().take(8192).any(|byte| *byte == 0) {
            let mut out = format!("Binary file: {}\nSize: {} bytes\n", path_str, file_size);
            if let Some(modified) = &modified {
                out.push_str(&format!("Modified: {}\n", modified));
            }
            out.push_str("Type: binary (cannot display contents).\n");
            out.push_str(&format!(
                "If you need its contents, run the {} tool with: file \"{}\" to identify the format, then choose a matching extraction command.",
                self.policy.shell_tool_name(),
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

        let tail_lines = tail_value
            .and_then(Value::as_u64)
            // Be tolerant of older/smaller models that emit zero to mean
            // "no tail selection". The schema prevents new strict calls from
            // doing so, while this fallback avoids a useless retry loop.
            .filter(|n| *n > 0)
            .map(|n| n as usize);

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

        let text = String::from_utf8(bytes)
            .map_err(|_| anyhow::anyhow!("Cannot display non-UTF-8 file: {}", path_str))?;
        let selected = read_selected_lines(&text, selection);
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
        let canonical_path = backend
            .canonicalize(&path)
            .await
            .unwrap_or_else(|_| path.clone())
            .to_string();
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
                finish: None,
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
            finish: None,
        })
    }
}

pub fn render_read_file_output(metadata: &ReadFileResultMetadata) -> String {
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
    let formatted = super::format_with_line_numbers(&selected_content, start_line - 1);
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
pub fn render_read_file_output_within(
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

fn read_selected_lines(content: &str, selection: ReadSelection) -> SelectedLines {
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

            for line in content.lines() {
                total_lines += 1;
                let zero_based_index = total_lines - 1;
                let in_range = zero_based_index >= start
                    && end_exclusive.is_none_or(|end| zero_based_index < end);
                if in_range && !capped {
                    let line = cap_line(line.to_string());
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
            SelectedLines {
                lines,
                total_lines,
                start_index: start,
                end_display,
                truncated,
            }
        }
        ReadSelection::Tail { count } => {
            let mut lines: VecDeque<String> = VecDeque::new();
            let mut total_lines: usize = 0;
            let mut stored_bytes: usize = 0;

            for line in content.lines() {
                total_lines += 1;
                if lines.len() == count
                    && let Some(dropped) = lines.pop_front()
                {
                    stored_bytes = stored_bytes.saturating_sub(dropped.len() + 1);
                }
                let line = cap_line(line.to_string());
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
            SelectedLines {
                lines,
                total_lines,
                start_index,
                end_display: total_lines,
                truncated,
            }
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
fn sniff_document_kind(path: &BackendPath, bytes: &[u8]) -> Option<DocumentKind> {
    let header = &bytes[..bytes.len().min(8)];
    if header.starts_with(b"%PDF") {
        return Some(DocumentKind::Pdf);
    }

    let extension = path.extension().map(|ext| ext.to_ascii_lowercase());
    let is_zip = header.starts_with(b"PK\x03\x04");
    let is_ole = header.starts_with(&[0xD0, 0xCF, 0x11, 0xE0]);
    match extension.as_deref() {
        Some("docx") if is_zip => Some(DocumentKind::Word),
        Some("doc") if is_ole => Some(DocumentKind::Word),
        _ => None,
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
    shell: &str,
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
            out.push_str(&format!(
                "This tool cannot extract PDF text directly. Run the {shell} tool to extract it:\n"
            ));
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
            out.push_str(&format!(
                "This tool cannot extract Word document text directly. Run the {shell} tool to extract it:\n"
            ));
            #[cfg(target_os = "macos")]
            out.push_str(&format!("  textutil -convert txt -stdout \"{}\"\n", quoted));
            #[cfg(not(target_os = "macos"))]
            out.push_str(&format!("  pandoc -t plain \"{}\"\n", quoted));
        }
    }
    out.push_str("Do not ask the user to convert the file — extract it yourself.");
    out
}

fn sniff_file_image_mime(bytes: &[u8]) -> Option<&'static str> {
    crate::util::sniff_image_mime(&bytes[..bytes.len().min(16)])
}

async fn image_file_outcome(
    backend: SharedBackend,
    display_path: &str,
    path: &BackendPath,
    file_size: u64,
    modified: Option<&str>,
    mime_type: &str,
) -> anyhow::Result<ToolCallOutcome> {
    let mut output = format!("Image file: {}\nSize: {} bytes\n", display_path, file_size);
    if let Some(modified) = modified {
        output.push_str(&format!("Modified: {}\n", modified));
    }
    output.push_str(&format!("Type: {mime_type}\n"));
    output.push_str("Attached for vision analysis.");

    let filename = path.file_name().unwrap_or("image").to_string();
    let staging_dir = std::env::temp_dir().join("aidaemon-execution-observations");
    tokio::fs::create_dir_all(&staging_dir).await?;
    let local_path = staging_dir.join(format!("{}-{}", uuid::Uuid::new_v4(), filename));
    backend.export_local_file(path, &local_path).await?;

    Ok(ToolCallOutcome {
        output,
        metadata: ToolCallMetadata {
            attachments: vec![MessageAttachment {
                local_path: local_path.to_string_lossy().into_owned(),
                filename,
                mime_type: mime_type.to_string(),
                size_bytes: file_size,
                provenance: AttachmentProvenance::ToolObservation,
                source_tool: Some("read_file".to_string()),
                ..MessageAttachment::default()
            }],
            ..ToolCallMetadata::default()
        },
        finish: None,
    })
}

fn format_modified_rfc3339(metadata: &BackendMetadata) -> Option<String> {
    let modified = metadata.modified?;
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
