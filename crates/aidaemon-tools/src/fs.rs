//! `read_file`, `write_file` and `edit_file`.

use std::path::Path;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use aidaemon_core::{
    AccessManifest, Tool, ToolAnnotations, ToolContext, ToolOutput, ToolResult, ToolSpec,
};

use crate::util::{fail, line_of, looks_binary, parse_args, truncate_chars};
use crate::workspace::Workspace;

/// Default number of lines `read_file` returns.
pub const DEFAULT_READ_LIMIT: usize = 2000;
/// Lines longer than this are cut in `read_file` output.
pub const MAX_LINE_CHARS: usize = 2000;
/// Files larger than this are refused by `read_file` and `edit_file`.
pub const MAX_FILE_BYTES: u64 = 20 * 1024 * 1024;

fn path_arg(arguments: &Value) -> Option<&str> {
    arguments.get("path").and_then(Value::as_str)
}

/// Read a text file and refuse binaries or oversized files with a message.
async fn read_text(ws: &Workspace, path: &Path) -> Result<String, String> {
    let shown = ws.display(path);
    let meta = match tokio::fs::metadata(path).await {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "`{shown}` does not exist. Use glob to find the right path."
            ))
        }
        Err(e) => return Err(format!("cannot read `{shown}`: {e}")),
    };
    if meta.is_dir() {
        return Err(format!(
            "`{shown}` is a directory. Use glob (for example `{shown}/*`) to list it."
        ));
    }
    if meta.len() > MAX_FILE_BYTES {
        return Err(format!(
            "`{shown}` is {} bytes, over the {MAX_FILE_BYTES}-byte limit. Use grep to search it.",
            meta.len()
        ));
    }
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| format!("cannot read `{shown}`: {e}"))?;
    if looks_binary(&bytes) {
        return Err(format!(
            "`{shown}` looks like a binary file ({} bytes); only text files can be read or edited.",
            bytes.len()
        ));
    }
    Ok(match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    })
}

// ---------------------------------------------------------------- read_file

/// Read a text file as numbered lines (`cat -n` style), paged with
/// `offset`/`limit`.
#[derive(Debug, Clone)]
pub struct ReadFile {
    workspace: Workspace,
}

impl ReadFile {
    pub fn new(workspace: Workspace) -> Self {
        Self { workspace }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadArgs {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

#[async_trait]
impl Tool for ReadFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: format!(
                "Read a UTF-8 text file in the workspace. Returns numbered lines \
                 (line number, tab, text). By default returns the first {DEFAULT_READ_LIMIT} \
                 lines; for longer files page with `offset` (1-based start line) and `limit`. \
                 Lines over {MAX_LINE_CHARS} characters are cut. Binary files are refused. \
                 Use grep to locate text in large files instead of reading them whole."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace root."},
                    "offset": {"type": "integer", "minimum": 1, "description": "1-based line to start from. Default 1."},
                    "limit": {"type": "integer", "minimum": 1, "description": format!("Maximum lines to return. Default {DEFAULT_READ_LIMIT}.")}
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }

    fn access(&self, arguments: &Value) -> AccessManifest {
        AccessManifest {
            reads: path_arg(arguments)
                .map(|p| vec![self.workspace.declare(p)])
                .unwrap_or_default(),
            ..Default::default()
        }
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> ToolResult {
        let args: ReadArgs = match parse_args("read_file", arguments) {
            Ok(a) => a,
            Err(e) => return Ok(fail(e)),
        };
        let path = match self.workspace.resolve(&args.path) {
            Ok(p) => p,
            Err(e) => return Ok(fail(e.to_string())),
        };
        let shown = self.workspace.display(&path);
        let text = match read_text(&self.workspace, &path).await {
            Ok(t) => t,
            Err(e) => return Ok(fail(e)),
        };
        let offset = args.offset.unwrap_or(1).max(1);
        let limit = args.limit.unwrap_or(DEFAULT_READ_LIMIT).max(1);

        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();
        if total == 0 {
            return Ok(ToolOutput::text(format!("(`{shown}` is empty)")).with_details(
                json!({"path": shown, "total_lines": 0, "start_line": 0, "end_line": 0, "truncated": false}),
            ));
        }
        if offset > total {
            return Ok(fail(format!(
                "offset {offset} is past the end of `{shown}`, which has {total} lines."
            )));
        }
        let end = (offset - 1 + limit).min(total);
        let mut out = String::new();
        let mut cut_lines = 0usize;
        for (i, line) in lines[offset - 1..end].iter().enumerate() {
            let (line, cut) = truncate_chars(line, MAX_LINE_CHARS);
            cut_lines += cut as usize;
            out.push_str(&format!("{:>6}\t{}\n", offset + i, line));
        }
        let truncated = end < total || offset > 1;
        if end < total {
            out.push_str(&format!(
                "\n[Showing lines {offset}-{end} of {total}. Call read_file with offset={} to continue.]",
                end + 1
            ));
        } else if offset > 1 {
            out.push_str(&format!(
                "\n[Showing lines {offset}-{end} of {total} (end of file).]"
            ));
        }
        if cut_lines > 0 {
            out.push_str(&format!(
                "\n[{cut_lines} line(s) longer than {MAX_LINE_CHARS} characters were cut.]"
            ));
        }
        Ok(ToolOutput::text(out).with_details(json!({
            "path": shown,
            "start_line": offset,
            "end_line": end,
            "total_lines": total,
            "truncated": truncated,
        })))
    }
}

// --------------------------------------------------------------- write_file

/// Create or overwrite a file, creating parent directories.
#[derive(Debug, Clone)]
pub struct WriteFile {
    workspace: Workspace,
}

impl WriteFile {
    pub fn new(workspace: Workspace) -> Self {
        Self { workspace }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteArgs {
    path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description: "Create a file, or overwrite an existing one, with the given content. \
                 Parent directories are created. Prefer edit_file for changing part of an \
                 existing file; use this for new files or complete rewrites."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace root."},
                    "content": {"type": "string", "description": "The complete file content."}
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        // Writing the same content twice leaves the same file.
        ToolAnnotations {
            idempotent: true,
            ..ToolAnnotations::WRITE
        }
    }

    fn access(&self, arguments: &Value) -> AccessManifest {
        AccessManifest {
            writes: path_arg(arguments)
                .map(|p| vec![self.workspace.declare(p)])
                .unwrap_or_default(),
            ..Default::default()
        }
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> ToolResult {
        let args: WriteArgs = match parse_args("write_file", arguments) {
            Ok(a) => a,
            Err(e) => return Ok(fail(e)),
        };
        let path = match self.workspace.resolve(&args.path) {
            Ok(p) => p,
            Err(e) => return Ok(fail(e.to_string())),
        };
        let shown = self.workspace.display(&path);
        let existed = match tokio::fs::metadata(&path).await {
            Ok(m) if m.is_dir() => {
                return Ok(fail(format!("`{shown}` is a directory; give a file path.")))
            }
            Ok(_) => true,
            Err(_) => false,
        };
        if let Some(parent) = path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return Ok(fail(format!(
                    "cannot create directories for `{shown}`: {e}"
                )));
            }
        }
        if let Err(e) = tokio::fs::write(&path, args.content.as_bytes()).await {
            return Ok(fail(format!("cannot write `{shown}`: {e}")));
        }
        let lines = args.content.lines().count();
        let verb = if existed { "Overwrote" } else { "Created" };
        Ok(ToolOutput::text(format!(
            "{verb} `{shown}` ({} bytes, {lines} lines).",
            args.content.len()
        ))
        .with_details(json!({
            "path": shown,
            "bytes": args.content.len(),
            "lines": lines,
            "created": !existed,
        })))
    }
}

// ---------------------------------------------------------------- edit_file

/// Exact string replacement. Fails without changing anything when the text
/// is missing or ambiguous.
#[derive(Debug, Clone)]
pub struct EditFile {
    workspace: Workspace,
}

impl EditFile {
    pub fn new(workspace: Workspace) -> Self {
        Self { workspace }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EditArgs {
    path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

/// Lines of context shown around an edit.
const SNIPPET_CONTEXT: usize = 3;

#[async_trait]
impl Tool for EditFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit_file".into(),
            description: "Replace an exact string in a text file. `old_string` must match the \
                 file byte for byte, including whitespace and indentation (do not include the \
                 line-number prefix from read_file). If it matches more than once, add \
                 surrounding lines to make it unique, or set `replace_all` to change every \
                 occurrence. Nothing is written when the match is missing or ambiguous. \
                 Returns the edited region with line numbers."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace root."},
                    "old_string": {"type": "string", "description": "Exact text to replace. Must be non-empty."},
                    "new_string": {"type": "string", "description": "Replacement text. Must differ from old_string."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence. Default false."}
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::WRITE
    }

    fn validate(&self, arguments: &Value) -> Result<(), String> {
        let Some(obj) = arguments.as_object() else {
            return Err("arguments must be a JSON object".into());
        };
        for key in ["path", "old_string", "new_string"] {
            if !obj.get(key).is_some_and(Value::is_string) {
                return Err(format!("`{key}` is required and must be a string"));
            }
        }
        let old = obj["old_string"].as_str().unwrap_or_default();
        if old.is_empty() {
            return Err("`old_string` must not be empty; use write_file to create a file".into());
        }
        if Some(old) == obj["new_string"].as_str() {
            return Err("`old_string` and `new_string` are identical; nothing to change".into());
        }
        Ok(())
    }

    fn access(&self, arguments: &Value) -> AccessManifest {
        let paths: Vec<String> = path_arg(arguments)
            .map(|p| vec![self.workspace.declare(p)])
            .unwrap_or_default();
        AccessManifest {
            reads: paths.clone(),
            writes: paths,
            ..Default::default()
        }
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> ToolResult {
        if let Err(e) = self.validate(&arguments) {
            return Ok(fail(e));
        }
        let args: EditArgs = match parse_args("edit_file", arguments) {
            Ok(a) => a,
            Err(e) => return Ok(fail(e)),
        };
        let path = match self.workspace.resolve(&args.path) {
            Ok(p) => p,
            Err(e) => return Ok(fail(e.to_string())),
        };
        let shown = self.workspace.display(&path);
        let text = match read_text(&self.workspace, &path).await {
            Ok(t) => t,
            Err(e) => return Ok(fail(e)),
        };

        let positions: Vec<usize> = text
            .match_indices(&args.old_string)
            .map(|(i, _)| i)
            .collect();
        if positions.is_empty() {
            let mut msg = format!("`old_string` was not found in `{shown}`. No changes made.");
            let trimmed = args.old_string.trim();
            if !trimmed.is_empty() && trimmed != args.old_string && text.contains(trimmed) {
                msg.push_str(
                    " The text matches once leading/trailing whitespace is removed; \
                     check indentation and line endings.",
                );
            } else {
                msg.push_str(" Read the file again to copy the exact current text.");
            }
            return Ok(fail(msg).with_details(json!({"path": shown, "matches": 0})));
        }
        if positions.len() > 1 && !args.replace_all {
            let lines: Vec<usize> = positions.iter().map(|p| line_of(&text, *p)).collect();
            let shown_lines: Vec<String> = lines.iter().take(20).map(|l| l.to_string()).collect();
            let more = if lines.len() > 20 { ", ..." } else { "" };
            return Ok(fail(format!(
                "`old_string` matches {} times in `{shown}` (lines {}{more}). No changes made. \
                 Include more surrounding context to make it unique, or set replace_all=true.",
                positions.len(),
                shown_lines.join(", "),
            ))
            .with_details(json!({"path": shown, "matches": positions.len(), "lines": lines})));
        }

        let updated = if args.replace_all {
            text.replace(&args.old_string, &args.new_string)
        } else {
            text.replacen(&args.old_string, &args.new_string, 1)
        };
        if let Err(e) = tokio::fs::write(&path, updated.as_bytes()).await {
            return Ok(fail(format!("cannot write `{shown}`: {e}")));
        }

        // Snippet around the first replacement, in the updated file.
        let first_line = line_of(&text, positions[0]);
        let new_lines = args.new_string.lines().count().max(1);
        let all: Vec<&str> = updated.lines().collect();
        let start = first_line.saturating_sub(SNIPPET_CONTEXT).max(1);
        let end = (first_line + new_lines - 1 + SNIPPET_CONTEXT).min(all.len());
        let mut snippet = String::new();
        if start <= end {
            for (i, line) in all[start - 1..end].iter().enumerate() {
                let (line, _) = truncate_chars(line, MAX_LINE_CHARS);
                snippet.push_str(&format!("{:>6}\t{}\n", start + i, line));
            }
        }
        let count = positions.len();
        let summary = if count == 1 {
            format!("Edited `{shown}` (1 replacement at line {first_line}).")
        } else {
            format!("Edited `{shown}` ({count} replacements; first at line {first_line}).")
        };
        Ok(
            ToolOutput::text(format!("{summary}\n{snippet}")).with_details(json!({
                "path": shown,
                "replacements": count,
                "first_line": first_line,
            })),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{ctx, ws};

    #[tokio::test]
    async fn read_numbers_lines_and_pages() {
        let (_d, ws) = ws();
        let body: String = (1..=2500).map(|i| format!("line {i}\n")).collect();
        std::fs::write(ws.root().join("big.txt"), body).unwrap();
        let tool = ReadFile::new(ws.clone());

        let out = tool.call(json!({"path": "big.txt"}), &ctx()).await.unwrap();
        assert!(!out.is_error);
        assert!(
            out.content.starts_with("     1\tline 1\n"),
            "{}",
            &out.content[..40]
        );
        assert!(out.content.contains("  2000\tline 2000\n"));
        assert!(!out.content.contains("line 2001\n"));
        assert!(
            out.content
                .contains("Showing lines 1-2000 of 2500. Call read_file with offset=2001"),
            "{}",
            out.content.lines().last().unwrap()
        );
        assert_eq!(out.details["truncated"], true);

        let out = tool
            .call(json!({"path": "big.txt", "offset": 2001}), &ctx())
            .await
            .unwrap();
        assert!(out.content.starts_with("  2001\tline 2001\n"));
        assert!(out.content.contains("end of file"));

        let out = tool
            .call(json!({"path": "big.txt", "offset": 10, "limit": 2}), &ctx())
            .await
            .unwrap();
        assert!(out
            .content
            .starts_with("    10\tline 10\n    11\tline 11\n"));
        assert!(out.content.contains("offset=12"));

        let out = tool
            .call(json!({"path": "big.txt", "offset": 9999}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("2500 lines"));
    }

    #[tokio::test]
    async fn read_refuses_binary_missing_dir_and_escape() {
        let (_d, ws) = ws();
        std::fs::write(ws.root().join("blob.bin"), [0u8, 159, 146, 150, 0, 1]).unwrap();
        std::fs::create_dir(ws.root().join("dir")).unwrap();
        let tool = ReadFile::new(ws.clone());
        let out = tool
            .call(json!({"path": "blob.bin"}), &ctx())
            .await
            .unwrap();
        assert!(
            out.is_error && out.content.contains("binary"),
            "{}",
            out.content
        );
        let out = tool
            .call(json!({"path": "nope.txt"}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error && out.content.contains("does not exist"));
        let out = tool.call(json!({"path": "dir"}), &ctx()).await.unwrap();
        assert!(out.is_error && out.content.contains("directory"));
        let out = tool
            .call(json!({"path": "../../etc/passwd"}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error && out.content.contains("outside the workspace"));
        let out = tool
            .call(json!({"path": "/etc/hostname"}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error && out.content.contains("outside the workspace"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn file_tools_refuse_symlink_escape() {
        let (_d, ws) = ws();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "synthetic secret").unwrap();
        std::os::unix::fs::symlink(outside.path(), ws.root().join("out")).unwrap();
        let read = ReadFile::new(ws.clone());
        let out = read
            .call(json!({"path": "out/secret.txt"}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error && out.content.contains("outside the workspace"));
        let write = WriteFile::new(ws.clone());
        let out = write
            .call(json!({"path": "out/new.txt", "content": "x"}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(!outside.path().join("new.txt").exists());
        let write_escape = write
            .call(json!({"path": "../escape.txt", "content": "x"}), &ctx())
            .await
            .unwrap();
        assert!(write_escape.is_error);
    }

    #[tokio::test]
    async fn write_creates_parents_and_reports() {
        let (_d, ws) = ws();
        let tool = WriteFile::new(ws.clone());
        let out = tool
            .call(
                json!({"path": "a/b/c.txt", "content": "one\ntwo\n"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert!(!out.is_error, "{}", out.content);
        assert!(out.content.starts_with("Created `a/b/c.txt`"));
        assert_eq!(
            std::fs::read_to_string(ws.root().join("a/b/c.txt")).unwrap(),
            "one\ntwo\n"
        );
        let out = tool
            .call(json!({"path": "a/b/c.txt", "content": "new"}), &ctx())
            .await
            .unwrap();
        assert!(out.content.starts_with("Overwrote"));
        let access = tool.access(&json!({"path": "a/b/c.txt"}));
        assert_eq!(
            access.writes,
            vec![ws.root().join("a/b/c.txt").display().to_string()]
        );
        assert!(!tool.annotations().read_only);
    }

    #[tokio::test]
    async fn edit_replaces_unique_match_with_snippet() {
        let (_d, ws) = ws();
        std::fs::write(
            ws.root().join("main.rs"),
            "fn main() {\n    let x = 1;\n    println!(\"{x}\");\n}\n",
        )
        .unwrap();
        let tool = EditFile::new(ws.clone());
        let out = tool
            .call(
                json!({"path": "main.rs", "old_string": "let x = 1;", "new_string": "let x = 2;"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert!(!out.is_error, "{}", out.content);
        assert!(out.content.contains("1 replacement at line 2"));
        assert!(
            out.content.contains("     2\t    let x = 2;"),
            "{}",
            out.content
        );
        assert!(std::fs::read_to_string(ws.root().join("main.rs"))
            .unwrap()
            .contains("let x = 2;"));
    }

    #[tokio::test]
    async fn edit_not_found_and_ambiguous_change_nothing() {
        let (_d, ws) = ws();
        let original = "a = 1\nb = 2\na = 1\nc = 3\na = 1\n";
        std::fs::write(ws.root().join("cfg.txt"), original).unwrap();
        let tool = EditFile::new(ws.clone());

        let out = tool
            .call(
                json!({"path": "cfg.txt", "old_string": "z = 9", "new_string": "z = 0"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("not found"), "{}", out.content);

        let out = tool
            .call(
                json!({"path": "cfg.txt", "old_string": "  b = 2", "new_string": "b = 5"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert!(out.is_error && out.content.contains("whitespace"));

        let out = tool
            .call(
                json!({"path": "cfg.txt", "old_string": "a = 1", "new_string": "a = 7"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(
            out.content.contains("matches 3 times") && out.content.contains("lines 1, 3, 5"),
            "{}",
            out.content
        );
        assert_eq!(out.details["lines"], json!([1, 3, 5]));
        assert_eq!(
            std::fs::read_to_string(ws.root().join("cfg.txt")).unwrap(),
            original
        );

        let out = tool
            .call(
                json!({"path": "cfg.txt", "old_string": "a = 1", "new_string": "a = 7", "replace_all": true}),
                &ctx(),
            )
            .await
            .unwrap();
        assert!(!out.is_error, "{}", out.content);
        assert!(out.content.contains("3 replacements"));
        assert_eq!(
            std::fs::read_to_string(ws.root().join("cfg.txt")).unwrap(),
            "a = 7\nb = 2\na = 7\nc = 3\na = 7\n"
        );
    }

    #[test]
    fn edit_validation_runs_before_io() {
        let (_d, ws) = ws();
        let tool = EditFile::new(ws);
        assert!(tool
            .validate(&json!({"path": "x", "old_string": "", "new_string": "y"}))
            .is_err());
        assert!(tool
            .validate(&json!({"path": "x", "old_string": "y", "new_string": "y"}))
            .is_err());
        assert!(tool
            .validate(&json!({"path": "x", "old_string": "y"}))
            .is_err());
        assert!(tool
            .validate(&json!({"path": "x", "old_string": "a", "new_string": "b"}))
            .is_ok());
    }
}
